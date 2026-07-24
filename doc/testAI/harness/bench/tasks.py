"""Dựng 3 bộ nhiệm vụ tương ứng 3 vấn đề của đồ án.

  V1 — ra đề toán          -> không có đáp án chuẩn, chấm bằng rubric + kiểm chéo
  V2 — giải bài tập toán   -> có expected_answer, chấm có tham chiếu
  V3 — chấm bài sinh viên  -> nhãn BIẾT TRƯỚC, chấm như phân loại nhị phân

Điểm mấu chốt của V3: bộ test không có "bài làm sinh viên". Ta phải sinh ra, nhưng
nhãn phải ĐÁNG TIN, nếu không thì mọi con số accuracy đều vô nghĩa.
Cách bảo đảm — xem docstring của build_v3().
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

FINAL_MARK = "ĐÁP ÁN CUỐI:"
VERDICT_MARK = "KẾT LUẬN:"
GEN_Q_MARK = "ĐỀ BÀI:"
GEN_A_MARK = "ĐÁP ÁN:"

# System prompt bám theo prompt.txt của production (đóng vai GV toán FPT + bắt buộc
# suy luận từng bước = kỹ thuật Chain-of-Thought). Dùng CHUNG cho cả 3 model để so
# sánh công bằng; chỉ tham số sinh (temp/top_p/max_tokens) là riêng theo model card.
SYSTEM = (
    "Bạn là FPTU_MATHAI — trợ giảng toán của Đại học FPT, phụ trách 3 môn: "
    "MAE101 (Toán cho kỹ thuật), MAD101 (Toán rời rạc), MAS291 (Xác suất & thống kê).\n"
    "Nguyên tắc bắt buộc:\n"
    "- Suy luận từng bước, rõ ràng, bằng tiếng Việt.\n"
    "- Dùng thuật ngữ và ký hiệu toán chuẩn xác.\n"
    "- KHÔNG bịa định lý, công thức hay dữ kiện không có trong đề.\n"
    "- Luôn kiểm tra lại phép tính ở bước cuối trước khi kết luận."
)

RAG_RULE = (
    "Dưới đây là tài liệu tham khảo được trích xuất từ kho học liệu của trường. "
    "Hãy ƯU TIÊN dùng định lý/công thức/phương pháp trong tài liệu này. "
    "Khi dùng, trích dẫn số hiệu đoạn, ví dụ [1], [2]. "
    "Nếu tài liệu KHÔNG chứa thứ bạn cần, hãy nói rõ 'tài liệu không đủ' rồi mới "
    "dùng kiến thức của bạn."
)


@dataclass
class Task:
    task_id: str
    problem: str              # 'V1' | 'V2' | 'V3'
    subject: str
    topic: str
    difficulty: str
    query: str                # dùng để RAG truy hồi
    user_prompt: str          # phần đề bài (chưa gắn ngữ cảnh)
    meta: dict[str, Any] = field(default_factory=dict)


def load_test_set(root: Path) -> list[dict]:
    items = []
    for f in ("FPT_Test_Set_60_MAE101.json", "FPT_Test_Set_60_MAD101.json",
              "FPT_Test_Set_60_MAS291.json"):
        items += json.loads((root / f).read_text(encoding="utf-8"))
    return items


HARD_FILE = "FPT_Math_Hard_Test_Set_180_MAE101_MAD101_MAS291.json"
HARD_PREFIX = "HARD_"


def load_hard_set(root: Path) -> list[dict]:
    """Bộ 180 câu KHÓ (medium 48 / hard 132, không có easy).

    ⚠️ test_id của bộ khó TRÙNG HỆT bộ dễ ('TEST_MAE101_001' có ở cả hai) -> phải
    thêm tiền tố, nếu không kết quả hai bộ ghi đè lên nhau mà không báo lỗi gì.
    """
    p = root / HARD_FILE
    if not p.exists():
        return []
    items = json.loads(p.read_text(encoding="utf-8"))
    for it in items:
        it["test_id"] = HARD_PREFIX + it["test_id"]
    return items


def build_v2_hard(items: list[dict]) -> list[Task]:
    """Giống build_v2 nhưng gắn nhãn problem='V2H' để tách số liệu khi báo cáo.

    Vì sao cần bộ khó: bộ 180 câu ban đầu bị HIỆU ỨNG TRẦN — model baseline 7B đã
    đạt 94.4% (kể cả nhóm câu 'hard' cũng 91.5%), chỉ còn 5.6 điểm dư địa nên
    không phân biệt nổi model nào hơn. Giữ cả hai bộ cho thấy đúng điều đó:
    dễ thì model nào cũng làm được, khó mới lộ ra khác biệt.
    """
    out = []
    for it in items:
        out.append(Task(
            task_id=it["test_id"], problem="V2H",
            subject=it["subject"], topic=it["topic"], difficulty=it["difficulty"],
            query=it["input_question"],
            user_prompt=(
                f"Giải bài toán sau (môn {it['subject']}, chủ đề {it['topic']}).\n\n"
                f"Bài toán: {it['input_question']}\n\n"
                f"Trình bày lời giải từng bước. Dòng CUỐI CÙNG bắt buộc theo đúng định dạng:\n"
                f"{FINAL_MARK} <đáp án ngắn gọn, không giải thích thêm>"
            ),
            meta={"expected_answer": it["expected_answer"],
                  "expected_context": it["expected_context"]},
        ))
    return out


# ── V2: giải bài tập ─────────────────────────────────────────────────────────
def build_v2(items: list[dict]) -> list[Task]:
    out = []
    for it in items:
        out.append(Task(
            task_id=it["test_id"], problem="V2",
            subject=it["subject"], topic=it["topic"], difficulty=it["difficulty"],
            query=it["input_question"],
            user_prompt=(
                f"Giải bài toán sau (môn {it['subject']}, chủ đề {it['topic']}).\n\n"
                f"Bài toán: {it['input_question']}\n\n"
                f"Trình bày lời giải từng bước. Dòng CUỐI CÙNG bắt buộc theo đúng định dạng:\n"
                f"{FINAL_MARK} <đáp án ngắn gọn, không giải thích thêm>"
            ),
            meta={"expected_answer": it["expected_answer"],
                  "expected_context": it["expected_context"]},
        ))
    return out


# ── V1: ra đề ────────────────────────────────────────────────────────────────
def build_v1(items: list[dict], n_per_subject: int = 10) -> list[Task]:
    """Lấy (môn, chủ đề, độ khó) từ bộ test làm ĐỀ BÀI cho việc ra đề.

    Không có đáp án chuẩn -> chấm bằng rubric + kiểm chéo (bắt 32B giải lại đề vừa
    sinh, xem đề có giải được và có khớp đáp án model tự khai không).
    """
    rng = random.Random(config.SEED)
    out = []
    by_subj: dict[str, list[dict]] = {}
    for it in items:
        by_subj.setdefault(it["subject"], []).append(it)
    for subj, lst in by_subj.items():
        # Chia đều 3 mức độ khó để đề sinh ra phủ hết phổ.
        picked = []
        for d in ("easy", "medium", "hard"):
            pool = [x for x in lst if x["difficulty"] == d]
            rng.shuffle(pool)
            picked += pool[: max(1, n_per_subject // 3)]
        for i, it in enumerate(picked[:n_per_subject]):
            out.append(Task(
                task_id=f"GEN_{subj}_{i+1:02d}", problem="V1",
                subject=subj, topic=it["topic"], difficulty=it["difficulty"],
                query=f"{subj} {it['topic']} lý thuyết công thức bài tập mẫu",
                user_prompt=(
                    f"Hãy RA MỘT ĐỀ BÀI TẬP mới cho môn {subj}, chủ đề \"{it['topic']}\", "
                    f"mức độ khó: {it['difficulty']}.\n"
                    f"Yêu cầu: đề phải tự chứa đủ dữ kiện để giải, có đáp án xác định, "
                    f"đúng phạm vi chủ đề, KHÔNG chép lại đề có sẵn.\n\n"
                    f"Trả lời theo ĐÚNG định dạng sau, không thêm gì khác:\n"
                    f"{GEN_Q_MARK} <nội dung đề bài>\n"
                    f"{GEN_A_MARK} <đáp án đúng của đề bạn vừa ra>"
                ),
                meta={"seed_topic": it["topic"]},
            ))
    return out


# ── V3: chấm bài sinh viên ───────────────────────────────────────────────────
NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _perturb_number(ans: str, rng: random.Random) -> str | None:
    """Đổi ĐÚNG MỘT con số trong đáp án -> đáp án chắc chắn SAI.

    Nhãn tin được 100%: đổi giá trị số thì kết quả không còn khớp đáp án chuẩn.
    Tránh đổi số nằm trong ngữ cảnh vô hại bằng cách chỉ đổi số CUỐI (thường là
    kết quả), và bảo đảm giá trị mới khác giá trị cũ.
    """
    ms = list(NUM_RE.finditer(ans))
    if not ms:
        return None
    m = ms[-1]
    old = m.group()
    try:
        val = float(old.replace(",", "."))
    except ValueError:
        return None
    delta = rng.choice([1, 2, -1, -2, 3])
    new_val = val + delta
    new = str(int(new_val)) if float(new_val).is_integer() and "." not in old and "," not in old \
        else f"{new_val:.2f}"
    if new == old:
        return None
    return ans[: m.start()] + new + ans[m.end():]


def _truncate(ans: str) -> str | None:
    """Cắt bỏ phần kết luận -> bài làm thiếu bước, chưa ra đáp án cuối."""
    words = ans.split()
    if len(words) < 6:
        return None
    return " ".join(words[: max(3, int(len(words) * 0.45))]) + " ..."


def build_v3_variants(items: list[dict], n_per_subject: int = 15) -> list[dict]:
    """Sinh bài làm sinh viên CÓ NHÃN BIẾT TRƯỚC.

    Vì sao nhãn đáng tin (đây là chỗ hội đồng sẽ vặn):
      - correct_verbatim : chính là expected_answer -> hiển nhiên ĐÚNG.
      - arithmetic_slip  : đổi một con số bằng code -> hiển nhiên SAI. Không nhờ AI.
      - incomplete       : cắt cụt trước khi ra kết luận -> SAI (chưa trả lời xong).
      - correct_paraphrase: nhờ 32B viết lại lời văn, SAU ĐÓ kiểm bằng code rằng
        MỌI token số trong đáp án gốc vẫn còn nguyên. Không đạt -> LOẠI khỏi bộ test.
        (sinh ở giai đoạn sau, trong prepare_paraphrases)
    Ba loại đầu không phụ thuộc phán đoán của bất kỳ AI nào.
    """
    rng = random.Random(config.SEED)
    by_subj: dict[str, list[dict]] = {}
    for it in items:
        by_subj.setdefault(it["subject"], []).append(it)
    rows = []
    for subj, lst in by_subj.items():
        picked = []
        for d in ("easy", "medium", "hard"):
            pool = [x for x in lst if x["difficulty"] == d]
            rng.shuffle(pool)
            picked += pool[: max(1, n_per_subject // 3)]
        for it in picked[:n_per_subject]:
            exp = it["expected_answer"]
            rows.append({**it, "variant": "correct_verbatim",
                         "student_answer": exp, "label": "correct", "label_src": "code"})
            bad = _perturb_number(exp, rng)
            if bad:
                rows.append({**it, "variant": "arithmetic_slip",
                             "student_answer": bad, "label": "incorrect", "label_src": "code"})
            inc = _truncate(exp)
            if inc:
                rows.append({**it, "variant": "incomplete",
                             "student_answer": inc, "label": "incorrect", "label_src": "code"})
    return rows


def v3_to_tasks(rows: list[dict]) -> list[Task]:
    out = []
    for i, r in enumerate(rows):
        out.append(Task(
            task_id=f"GRADE_{r['test_id']}_{r['variant']}", problem="V3",
            subject=r["subject"], topic=r["topic"], difficulty=r["difficulty"],
            query=r["input_question"],
            user_prompt=(
                f"Bạn đang chấm bài cho sinh viên môn {r['subject']}.\n\n"
                f"Câu hỏi: {r['input_question']}\n\n"
                f"Đáp án chuẩn của giảng viên: {r['expected_answer']}\n\n"
                f"Bài làm của sinh viên: {r['student_answer']}\n\n"
                f"Nhiệm vụ: xác định bài làm của sinh viên ĐÚNG hay SAI so với đáp án chuẩn. "
                f"Chấp nhận cách diễn đạt/trình bày khác nếu kết quả toán học tương đương. "
                f"Bài làm chưa ra kết quả cuối cùng thì tính là SAI.\n"
                f"Giải thích ngắn gọn, rồi dòng CUỐI CÙNG bắt buộc theo đúng định dạng:\n"
                f"{VERDICT_MARK} ĐÚNG   (hoặc)   {VERDICT_MARK} SAI"
            ),
            meta={"label": r["label"], "variant": r["variant"],
                  "expected_answer": r["expected_answer"],
                  "student_answer": r["student_answer"], "label_src": r["label_src"]},
        ))
    return out


# ── Ghép prompt cuối cùng ────────────────────────────────────────────────────
def render(task: Task, context: str | None) -> list[dict[str, str]]:
    if context:
        user = f"{RAG_RULE}\n\n===== TÀI LIỆU THAM KHẢO =====\n{context}\n===== HẾT =====\n\n{task.user_prompt}"
    else:
        user = task.user_prompt
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


# ── Bóc kết quả từ output ────────────────────────────────────────────────────
_BOXED = re.compile(r"\\boxed\{")
# Rác bao quanh đáp án: đậm markdown, mở/đóng môi trường LaTeX, gạch đầu dòng...
_JUNK_LINE = re.compile(r"^[\s*_`:\-—]*(?:\\\[|\\\]|\\\(|\\\)|\$\$?)?[\s*_`:\-—]*$")


def _boxed_contents(s: str) -> list[str]:
    """Bóc nội dung mọi \\boxed{...}, có đếm ngoặc lồng nhau.

    Regex thuần không làm được vì \\boxed{\\frac{1}{2}} có ngoặc lồng.
    """
    out = []
    for m in _BOXED.finditer(s):
        i = m.end()          # ngay sau '{'
        depth, buf = 1, []
        while i < len(s) and depth:
            c = s[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        v = "".join(buf).strip()
        if v:
            out.append(v)
    return out


def extract_final(text: str) -> str | None:
    """Bóc đáp án cuối, chịu được phong cách trình bày của CẢ HAI họ model.

    Vì sao phức tạp: model instruct (Qwen) viết thẳng 'ĐÁP ÁN CUỐI: f(2)=3', nhưng
    R1 được huấn luyện để trả lời bằng markdown + LaTeX \\boxed{}:
        **ĐÁP ÁN CUỐI:**
        \\[
        f(2) = \\boxed{3} \\quad ; \\quad x = \\boxed{1}
        \\]
    Lấy 'dòng ngay sau marker' sẽ ra '**' hoặc '\\[' -> mất trắng đáp án (đo thật:
    hỏng 128/180 với R1-7B nhưng chỉ 2/180 với Qwen). Bóc kiểu đó không chỉ sai,
    mà còn sai LỆCH HẲN VỀ MỘT HỌ MODEL -> nếu dùng để chấm thì bất công.
    """
    i = text.rfind(FINAL_MARK)
    seg = text[i + len(FINAL_MARK):] if i != -1 else text[-600:]
    seg = seg.strip()
    # \boxed{} là cách DeepSeek/R1 đánh dấu kết quả -> ưu tiên tuyệt đối.
    bx = _boxed_contents(seg)
    if bx:
        return " ; ".join(bx)[:300]
    # Không có boxed: lấy dòng đầu tiên CÓ NỘI DUNG (bỏ qua '**', '\[', ...).
    for line in seg.split("\n"):
        s = line.strip().strip("*`_ ").strip()
        if s and not _JUNK_LINE.match(line):
            return s[:300]
    return None


def extract_verdict(text: str) -> str | None:
    i = text.rfind(VERDICT_MARK)
    seg = text[i + len(VERDICT_MARK):] if i != -1 else text[-200:]
    seg = seg.strip().upper()
    # Bắt cả trường hợp model quên nhãn nhưng vẫn nói rõ đúng/sai ở cuối.
    first = re.search(r"\b(ĐÚNG|DUNG|SAI|CORRECT|INCORRECT|WRONG)\b", seg)
    if not first:
        return None
    w = first.group(1)
    return "correct" if w in ("ĐÚNG", "DUNG", "CORRECT") else "incorrect"


def extract_generated(text: str) -> tuple[str | None, str | None]:
    qi, ai = text.rfind(GEN_Q_MARK), text.rfind(GEN_A_MARK)
    if qi == -1:
        return None, None
    q = text[qi + len(GEN_Q_MARK): ai if ai > qi else len(text)].strip()
    a = text[ai + len(GEN_A_MARK):].strip() if ai > qi else None
    return (q or None), (a or None)


def cited_refs(text: str) -> set[int]:
    """Các số hiệu đoạn model trích dẫn -> dùng cho Groundedness."""
    return {int(x) for x in re.findall(r"\[(\d{1,2})\]", text)}
