"""Sinh biến thể 'correct_paraphrase' cho Vấn đề 3 — bài làm ĐÚNG nhưng viết khác đi.

  python -m bench.paraphrase        (chạy SAU khi pha B có 32B, TRƯỚC bench.run)

Vì sao cần biến thể này:
  Đây là ca đáng giá nhất của bài toán chấm tự động. Sinh viên hiếm khi viết y hệt
  đáp án của giảng viên. Nếu AI đánh trượt bài đúng chỉ vì khác câu chữ thì tính
  năng chấm bài KHÔNG dùng được, dù accuracy tổng trông vẫn đẹp.
  -> đo riêng bằng chỉ số "False-Fail Rate".

Vì sao nhãn 'correct' vẫn đáng tin dù do LLM viết:
  Sau khi 32B viết lại, ta KIỂM BẰNG CODE rằng mọi token số của đáp án gốc còn
  nguyên trong bản viết lại (và không có số lạ xuất hiện). Không đạt -> LOẠI, không
  đưa vào bộ test. Nghĩa là nhãn không dựa vào phán đoán của AI, mà dựa vào một
  bất biến kiểm được: viết lại mà mọi con số giữ nguyên thì kết quả toán không đổi.
  Đánh đổi trung thực: với đáp án thuần chữ (vd 'Biểu thức là P ∧ ¬Q') thì không có
  số để kiểm -> ta bỏ qua, KHÔNG sinh paraphrase cho những câu đó (thà ít mà chắc).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from . import config
from .client import chat_stream
from .judge import numbers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Sinh bằng model BASELINE (Qwen2.5-7B-Instruct), CỐ Ý không dùng 32B. Ba lý do:
#   1. Chống thiên lệch: 32B là model ta muốn kết luận là tốt nhất. Lấy chính nó
#      sinh dữ liệu test cho nó là mời hội đồng vặn. Dùng baseline thì nếu có thiên
#      lệch, thiên lệch NGHIÊNG VỀ Qwen — tức chống lại kết luận của ta. 32B mà vẫn
#      thắng trên dữ liệu do đối thủ sinh thì luận điểm mạnh hơn hẳn.
#   2. Đây là việc viết lại câu chữ, không phải suy luận — 7B-Instruct thừa sức, và
#      dù nó viết dở thì bộ kiểm bằng code vẫn chặn (verify()).
#   3. Chạy được ngay trong pha A -> đỡ một lần nạp/tháo 32B (~10 phút).
GEN = config.MODEL_BY_KEY["qwen7b"]

PROMPT = """Viết lại câu trả lời toán học sau bằng cách diễn đạt KHÁC, như một sinh viên tự viết.

BẮT BUỘC:
- Giữ NGUYÊN mọi con số và mọi kết quả. Không làm tròn, không đổi đơn vị, không rút gọn khác đi.
- Chỉ đổi cách hành văn, trật tự câu, từ ngữ.
- Không thêm lời giải thích thừa, không thêm số mới.

Câu trả lời gốc: {ans}

Chỉ in ra bản viết lại, không thêm gì khác. Dòng cuối cùng bắt đầu bằng "VIẾT LẠI:" rồi đến nội dung."""


def verify(orig: str, para: str) -> bool:
    """Bản viết lại hợp lệ khi tập số KHÔNG đổi và nội dung thực sự khác chữ."""
    if not para or len(para.split()) < 3:
        return False
    if numbers(orig) != numbers(para):
        return False
    # Phải khác đáng kể, nếu không thì trùng với biến thể correct_verbatim.
    return para.strip().lower() != orig.strip().lower()


def main():
    rows = json.loads((OUT / "v3_variants.json").read_text(encoding="utf-8"))
    # Chỉ lấy câu gốc (biến thể verbatim) và CHỈ những câu có số để kiểm được.
    base = [r for r in rows if r["variant"] == "correct_verbatim" and numbers(r["expected_answer"])]
    print(f"Ứng viên có số để kiểm: {len(base)}/{len([r for r in rows if r['variant']=='correct_verbatim'])}")

    def one(r, client):
        res = chat_stream(base_url=GEN.base_url, model=GEN.hf_id,
                          messages=[{"role": "user", "content": PROMPT.format(ans=r["expected_answer"])}],
                          temperature=0.7, top_p=0.95, max_tokens=2048,
                          timeout_s=config.REQUEST_TIMEOUT_S, client=client)
        if not res.ok:
            return None
        txt = res.answer_text or res.text
        i = txt.rfind("VIẾT LẠI:")
        para = (txt[i + len("VIẾT LẠI:"):] if i != -1 else txt).strip().split("\n")[0].strip()
        if not verify(r["expected_answer"], para):
            return None
        return {**{k: v for k, v in r.items() if k not in ("variant", "student_answer",
                                                           "label", "label_src")},
                "variant": "correct_paraphrase", "student_answer": para,
                "label": "correct", "label_src": "llm+code_verified"}

    out = []
    with httpx.Client(timeout=httpx.Timeout(config.REQUEST_TIMEOUT_S, connect=30.0),
                      limits=httpx.Limits(max_connections=32)) as client:
        with ThreadPoolExecutor(max_workers=GEN.quality_concurrency) as ex:
            futs = [ex.submit(one, r, client) for r in base]
            for k, f in enumerate(as_completed(futs), 1):
                v = f.result()
                if v:
                    out.append(v)
                if k % 20 == 0:
                    print(f"  {k}/{len(base)} — giữ được {len(out)}", flush=True)

    kept = len(out)
    print(f"Sinh {kept}/{len(base)} bản viết lại ĐẠT kiểm tra số "
          f"({kept/max(len(base),1):.0%}). Số bị loại: {len(base)-kept}")
    (OUT / "v3_paraphrases.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    # Gộp vào bộ biến thể chính để bench.run dùng.
    merged = [r for r in rows if r["variant"] != "correct_paraphrase"] + out
    (OUT / "v3_variants.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"Tổng biến thể V3: {len(merged)}")


if __name__ == "__main__":
    main()
