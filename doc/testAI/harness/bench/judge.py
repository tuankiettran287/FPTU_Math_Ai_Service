"""Chấm kết quả: đối chiếu máy móc trước, LLM giám khảo sau.

  python -m bench.judge

Trình tự cho V2 (giải toán):
  1. So khớp máy móc (chuẩn hoá chuỗi + so tập số). Khớp -> ĐÚNG, không cần AI.
  2. Còn lại -> giám khảo 32B chấm CÓ THAM CHIẾU (được xem đáp án chuẩn).

Vì sao chấm có tham chiếu thì thiên lệch tự-ưa-thích không đáng ngại:
  Thiên lệch LLM-judge chủ yếu xảy ra khi chấm "hay/dở" không có chuẩn. Ở đây
  giám khảo chỉ phải trả lời "kết quả này có khớp đáp án chuẩn không" — một phép
  đối chiếu, không phải đánh giá thẩm mỹ.

Và quan trọng nhất: ĐỘ TIN CẬY CỦA GIÁM KHẢO ĐƯỢC ĐO, không phải giả định.
Bộ V3 có nhãn đúng/sai biết trước do code sinh ra -> cho giám khảo chấm chính bộ
đó và đối chiếu với nhãn -> ra accuracy của giám khảo (xem calibrate_judge).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from . import config, tasks
from .client import chat_stream

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

JUDGE = config.MODEL_BY_KEY[config.JUDGE_KEY]

NUM_RE = re.compile(r"-?\d+(?:[./]\d+)?(?:[.,]\d+)?")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ── Bước 1: đối chiếu máy móc ────────────────────────────────────────────────
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*")
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[.;,]+$", "", s)
    return s


def numbers(s: str) -> list[str]:
    out = []
    for m in NUM_RE.finditer(norm(s)):
        t = m.group().replace(",", ".")
        try:
            f = float(t) if "/" not in t else eval(t)  # noqa: S307  (chỉ số, regex đã chặn)
            out.append(f"{float(f):.6g}")
        except Exception:
            out.append(t)
    return sorted(out)


def mechanical(final: str | None, expected: str) -> bool | None:
    """True = chắc chắn đúng. None = không kết luận được -> nhờ giám khảo.

    Cố ý KHÔNG bao giờ trả False: chuỗi khác nhau không có nghĩa là sai
    (vd '1/2' vs '0.5', 'x=1/2 và x=1' vs 'x=1, x=0.5').
    """
    if not final:
        return None
    if norm(final) == norm(expected):
        return True
    ne, nf = numbers(expected), numbers(final)
    # Đáp án chuẩn có số, và model nêu ĐÚNG y hệt tập số đó -> coi như đúng.
    if ne and ne == nf:
        return True
    return None


# ── Bước 2: giám khảo LLM ────────────────────────────────────────────────────
J_V2_RAG = """Bạn là giám khảo chấm toán, tuyệt đối khách quan.

CÂU HỎI: {q}

ĐÁP ÁN CHUẨN (chân lý, không được nghi ngờ): {exp}

TÀI LIỆU ĐÃ CUNG CẤP CHO THÍ SINH:
{ctx}

BÀI LÀM CỦA THÍ SINH:
{ans}

Trả lời ĐÚNG 4 dòng sau, không thêm gì khác:
CORRECT: yes/no          (kết quả cuối của thí sinh có tương đương ĐÁP ÁN CHUẨN không; chấp nhận khác cách viết, vd 1/2 = 0.5)
CONTEXT_SUFFICIENT: yes/no   (TÀI LIỆU trên có chứa đủ định lý/công thức/phương pháp cần để giải câu này không)
GROUNDED: yes/partial/no     (bài làm có thực sự dựa vào TÀI LIỆU không; 'no' nếu phớt lờ hoặc mâu thuẫn tài liệu)
ERROR: none/retrieval/calculation/hallucination
   none = làm đúng
   retrieval = sai VÌ tài liệu không có/không đúng thứ cần
   calculation = tài liệu đủ, dùng đúng phương pháp, nhưng tính toán/biến đổi sai
   hallucination = bịa công thức hoặc dữ kiện, hoặc phớt lờ tài liệu để dùng kiến thức sai"""

J_V2_PURE = """Bạn là giám khảo chấm toán, tuyệt đối khách quan.

CÂU HỎI: {q}

ĐÁP ÁN CHUẨN (chân lý, không được nghi ngờ): {exp}

BÀI LÀM CỦA THÍ SINH:
{ans}

Trả lời ĐÚNG 2 dòng sau, không thêm gì khác:
CORRECT: yes/no      (kết quả cuối có tương đương ĐÁP ÁN CHUẨN không; chấp nhận khác cách viết)
ERROR: none/calculation/hallucination
   none = đúng
   calculation = đúng phương pháp nhưng tính sai
   hallucination = bịa công thức/định lý hoặc dữ kiện không có trong đề"""

J_V1 = """Bạn là giám khảo thẩm định ĐỀ THI toán do AI ra, cho môn {subj}, chủ đề "{topic}", mức khó dự kiến: {diff}.

ĐỀ AI RA: {q}

ĐÁP ÁN AI TỰ KHAI: {a}

Trả lời ĐÚNG 5 dòng, không thêm gì khác:
SOLVABLE: yes/no        (đề có đủ dữ kiện để giải ra kết quả xác định không)
ON_TOPIC: yes/no        (có đúng chủ đề "{topic}" của môn {subj} không)
ANSWER_CORRECT: yes/no  (đáp án AI tự khai có đúng với đề nó vừa ra không — bạn hãy tự giải lại để kiểm)
DIFFICULTY: easy/medium/hard   (mức khó THỰC TẾ bạn đánh giá)
WELLFORMED: yes/no      (đề rõ nghĩa, không nhập nhằng, không thiếu/thừa dữ kiện)"""


def parse_judge(text: str) -> dict:
    """Bóc các dòng NHÃN: giá trị. Chỉ lấy lần xuất hiện CUỐI (sau <think>)."""
    out = {}
    for key in ("CORRECT", "CONTEXT_SUFFICIENT", "GROUNDED", "ERROR", "SOLVABLE",
                "ON_TOPIC", "ANSWER_CORRECT", "DIFFICULTY", "WELLFORMED"):
        m = None
        for m in re.finditer(rf"^\s*{key}\s*:\s*([A-Za-zÀ-ỹ]+)", text, re.M | re.I):
            pass
        if m:
            out[key.lower()] = m.group(1).strip().lower()
    return out


_JSTAT = {"n": 0, "truncated": 0, "no_label": 0, "error": 0}


def ask_judge(prompt: str, client: httpx.Client) -> tuple[dict, str]:
    r = chat_stream(base_url=JUDGE.base_url, model=JUDGE.hf_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.JUDGE_TEMPERATURE, top_p=0.95,
                    max_tokens=config.JUDGE_MAX_TOKENS,
                    timeout_s=config.REQUEST_TIMEOUT_S, client=client)
    _JSTAT["n"] += 1
    if not r.ok:
        _JSTAT["error"] += 1
        return {"_error": r.error or "fail"}, ""
    parsed = parse_judge(r.answer_text or r.text)
    # Giám khảo là R1 -> nghĩ trong <think> rồi mới in nhãn. Bị cắt vì hết max_tokens
    # nghĩa là nhãn không bao giờ được in ra. Phải đếm và báo, không được nuốt lặng.
    if r.truncated:
        _JSTAT["truncated"] += 1
    if not parsed:
        _JSTAT["no_label"] += 1
    return parsed, r.text


def judge_health() -> dict:
    s = dict(_JSTAT)
    n = max(s["n"], 1)
    s["truncated_pct"] = round(100 * s["truncated"] / n, 2)
    s["no_label_pct"] = round(100 * s["no_label"] / n, 2)
    return s


# ── Hiệu chuẩn giám khảo bằng nhãn thật của V3 ───────────────────────────────
def calibrate_judge(client: httpx.Client) -> dict:
    """Cho giám khảo chấm chính bộ V3 (nhãn do code sinh, chắc chắn đúng).

    Kết quả = accuracy của giám khảo. Đây là con số để trả lời hội đồng khi bị hỏi
    'lấy gì bảo đảm giám khảo AI chấm đúng?'.
    """
    rows = json.loads((OUT / "v3_variants.json").read_text(encoding="utf-8"))
    log(f"Hiệu chuẩn giám khảo trên {len(rows)} mẫu có nhãn thật...")

    def one(r):
        p = J_V2_PURE.format(q=r["input_question"], exp=r["expected_answer"],
                             ans=r["student_answer"])
        j, _ = ask_judge(p, client)
        pred = {"yes": "correct", "no": "incorrect"}.get(j.get("correct"))
        return {"test_id": r["test_id"], "variant": r["variant"],
                "label": r["label"], "judge_pred": pred}

    res = []
    with ThreadPoolExecutor(max_workers=JUDGE.quality_concurrency) as ex:
        for f in as_completed([ex.submit(one, r) for r in rows]):
            res.append(f.result())
    ok = [x for x in res if x["judge_pred"]]
    acc = sum(1 for x in ok if x["judge_pred"] == x["label"]) / max(len(ok), 1)
    by_var = {}
    for v in sorted({x["variant"] for x in res}):
        s = [x for x in ok if x["variant"] == v]
        by_var[v] = round(sum(1 for x in s if x["judge_pred"] == x["label"]) / max(len(s), 1), 4)
    summary = {"n": len(res), "n_parsed": len(ok), "accuracy": round(acc, 4),
               "by_variant": by_var, "detail": res}
    (OUT / "judge_calibration.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  Độ chính xác giám khảo: {acc:.1%} (theo biến thể: {by_var})")
    return summary


# ── Chấm toàn bộ ─────────────────────────────────────────────────────────────
def load_rows(model: str, mode: str) -> list[dict]:
    p = OUT / f"rows_{model}_{mode}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_previous_grades(model: str, mode: str) -> dict[str, dict]:
    """Kết quả chấm của lượt trước, tra theo task_id.

    Cho phép CHẤM NỐI TIẾP: khi bổ sung bộ test mới cho model đã chấm xong bộ cũ,
    ta chỉ chấm phần chưa có. Không có cơ chế này thì phải chấm lại từ đầu — với
    5 model x 2 chế độ x ~200 lượt giám khảo thì mất cả tiếng GPU cho việc đã làm rồi.
    """
    p = OUT / f"graded_{model}_{mode}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        # Chỉ tính là "đã chấm" khi thực sự có kết luận, không phải dòng lỗi dở dang.
        if r.get("graded") and ("correct" in r or r.get("graded") == "error"):
            out[r["task_id"]] = r
    return out


def load_texts(model: str, mode: str) -> dict[str, str]:
    p = OUT / f"texts_{model}_{mode}.jsonl"
    if not p.exists():
        return {}
    d = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            o = json.loads(l)
            d[o["task_id"]] = o["text"]
    return d


def reextract(rows: list[dict], texts: dict[str, str]) -> dict:
    """Bóc lại final_answer/verdict/gen_* từ VĂN BẢN GỐC đã lưu.

    Chạy offline, không tốn GPU. Cần thiết vì bản bóc lúc chạy dùng luật 'lấy dòng
    ngay sau marker' -> hỏng với phong cách markdown/LaTeX \\boxed{} của R1
    (128/180 với R1-7B so với 2/180 với Qwen). Sai lệch hẳn về một họ model.
    Giữ nguyên format_ok của lần chạy đầu: nó đo 'model CÓ in marker không' —
    đó là chỉ số tuân thủ chỉ dẫn thật, không liên quan tới lỗi bóc của mình.
    """
    stat = {"v2_fixed": 0, "v2_total": 0, "v3_fixed": 0, "v1_fixed": 0}
    for r in rows:
        t = texts.get(r.get("task_id"))
        if not t or not r.get("ok"):
            continue
        if r["problem"] in ("V2", "V2H"):
            stat["v2_total"] += 1
            new = tasks.extract_final(t)
            if new and new != r.get("final_answer"):
                r["final_answer"] = new
                stat["v2_fixed"] += 1
        elif r["problem"] == "V3":
            new = tasks.extract_verdict(t)
            if new and new != r.get("verdict"):
                r["verdict"] = new
                stat["v3_fixed"] += 1
        elif r["problem"] == "V1":
            q, a = tasks.extract_generated(t)
            if q and (q != r.get("gen_question") or a != r.get("gen_answer")):
                r["gen_question"], r["gen_answer"] = q, a
                stat["v1_fixed"] += 1
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--only", default=None,
                    help="chỉ chấm các model này (phân tách bằng dấu phẩy). "
                         "Dùng khi bổ sung model mới -> khỏi chấm lại model đã xong.")
    ap.add_argument("--regrade", action="store_true",
                    help="chấm lại TỪ ĐẦU, bỏ qua kết quả chấm cũ. Mặc định là chấm "
                         "nối tiếp: dòng nào đã có kết luận thì giữ nguyên.")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    items = tasks.load_test_set(ROOT / "testset")
    # Bộ khó đã được load_hard_set() thêm tiền tố HARD_ nên gộp chung dict là an toàn:
    # không có test_id nào đụng nhau. Không có tiền tố thì bộ khó sẽ ghi đè bộ dễ ở đây.
    hard = tasks.load_hard_set(ROOT / "testset")
    by_id = {i["test_id"]: i for i in items + hard}
    assert len(by_id) == len(items) + len(hard), "test_id bị trùng giữa hai bộ!"
    ctx = json.loads((OUT / "contexts.json").read_text(encoding="utf-8"))
    v1_tasks = {t.task_id: t for t in tasks.build_v1(items, n_per_subject=10)}

    with httpx.Client(timeout=httpx.Timeout(config.REQUEST_TIMEOUT_S, connect=30.0),
                      limits=httpx.Limits(max_connections=32)) as client:
        if not args.skip_calibration:
            calibrate_judge(client)

        graded_all = []
        for spec in config.MODELS:
            if only and spec.key not in only:
                continue
            for mode in ("pure", "rag"):
                rows = load_rows(spec.key, mode)
                if not rows:
                    log(f"bỏ qua {spec.key}/{mode} (chưa có kết quả)")
                    continue
                texts = load_texts(spec.key, mode)
                prev = {} if args.regrade else load_previous_grades(spec.key, mode)
                log(f"Chấm {spec.key}/{mode}: {len(rows)} dòng"
                    + (f" (đã chấm trước đó: {len(prev)})" if prev else ""))
                st = reextract(rows, texts)
                log(f"  bóc lại từ văn bản gốc: V2 sửa {st['v2_fixed']}/{st['v2_total']}, "
                    f"V3 sửa {st['v3_fixed']}, V1 sửa {st['v1_fixed']}")

                need_judge = []
                reused = 0
                for i, r in enumerate(rows):
                    # Dùng lại kết quả chấm cũ nếu có -> chỉ tốn GPU cho phần mới.
                    old = prev.get(r.get("task_id"))
                    if old is not None:
                        rows[i] = old
                        reused += 1
                        continue
                    if not r.get("ok"):
                        r["graded"] = "error"
                        continue
                    if r["problem"] == "V3":
                        # Không cần AI: so phán quyết model với nhãn code sinh.
                        r["correct"] = (r.get("verdict") == r.get("label"))
                        r["graded"] = "auto"
                        continue
                    if r["problem"] in ("V2", "V2H"):
                        mech = mechanical(r.get("final_answer"), by_id[r["task_id"]]["expected_answer"])
                        if mech is True:
                            r["correct"] = True
                            r["graded"] = "mechanical"
                            r["error_category"] = "none"
                            continue
                    need_judge.append(r)

                log(f"  dùng lại kết quả cũ: {reused}, "
                    f"máy móc xử được: {sum(1 for r in rows if r.get('graded')=='mechanical')}, "
                    f"cần giám khảo: {len(need_judge)}")

                def do(r):
                    tid = r["task_id"]
                    if r["problem"] in ("V2", "V2H"):
                        it = by_id[tid]
                        if r["mode"] == "rag":
                            p = J_V2_RAG.format(q=it["input_question"], exp=it["expected_answer"],
                                                ctx=ctx[tid]["context"][:12000],
                                                ans=(texts.get(tid) or "")[-6000:])
                        else:
                            p = J_V2_PURE.format(q=it["input_question"], exp=it["expected_answer"],
                                                 ans=(texts.get(tid) or "")[-6000:])
                        j, _ = ask_judge(p, client)
                        r["correct"] = j.get("correct") == "yes"
                        r["error_category"] = j.get("error", "unknown")
                        r["context_sufficient"] = j.get("context_sufficient")
                        r["grounded"] = j.get("grounded")
                        r["graded"] = "judge"
                    elif r["problem"] == "V1":
                        t = v1_tasks[tid]
                        p = J_V1.format(subj=t.subject, topic=t.topic, diff=t.difficulty,
                                        q=(r.get("gen_question") or "")[:3000],
                                        a=(r.get("gen_answer") or "")[:2000])
                        j, _ = ask_judge(p, client)
                        r["solvable"] = j.get("solvable") == "yes"
                        r["on_topic"] = j.get("on_topic") == "yes"
                        r["gen_answer_correct"] = j.get("answer_correct") == "yes"
                        r["judged_difficulty"] = j.get("difficulty")
                        r["wellformed"] = j.get("wellformed") == "yes"
                        r["difficulty_match"] = (j.get("difficulty") == t.difficulty)
                        # Đề coi là ĐẠT khi giải được + đúng chủ đề + đáp án tự khai đúng + rõ nghĩa
                        r["correct"] = bool(r["solvable"] and r["on_topic"]
                                            and r["gen_answer_correct"] and r["wellformed"])
                        r["graded"] = "judge"
                    return r

                with ThreadPoolExecutor(max_workers=JUDGE.quality_concurrency) as ex:
                    futs = [ex.submit(do, r) for r in need_judge]
                    for k, f in enumerate(as_completed(futs), 1):
                        f.result()
                        if k % 25 == 0:
                            log(f"    giám khảo {k}/{len(need_judge)}")

                with open(OUT / f"graded_{spec.key}_{mode}.jsonl", "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                graded_all += rows
    h = judge_health()
    (OUT / "judge_health.json").write_text(json.dumps(h, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    log(f"Chấm xong {len(graded_all)} dòng.")
    log(f"Sức khoẻ giám khảo: {h['n']} lượt | bị cắt {h['truncated_pct']}% | "
        f"không ra nhãn {h['no_label_pct']}% | lỗi {h['error']}")
    if h["no_label_pct"] > 5:
        log("⚠️ >5% lượt chấm không ra nhãn — số liệu accuracy sẽ bị lệch, cần xem lại "
            "JUDGE_MAX_TOKENS hoặc prompt giám khảo.")


if __name__ == "__main__":
    main()
