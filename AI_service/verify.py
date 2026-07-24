"""Kiểm tra bằng công cụ toán (sympy) — KHÔNG tin LLM tự soát.

- so sánh đáp án tương đương (biểu thức/số), không so chuỗi.
- lọc phương án nhiễu (distractor) trùng đáp án đúng.
"""
import re
from typing import Any


def _to_expr(text: Any):
    """Chuyển chuỗi/đáp án thành sympy expr; trả None nếu không parse được."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # gỡ latex/ký hiệu hay gặp
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = s.replace("\\pi", "pi").replace("^", "**")
    s = re.sub(r"[a-zA-Z]+\s*=\s*", "", s)  # bỏ "x = ..."
    s = s.strip().rstrip(".")
    try:
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application, parse_expr, standard_transformations,
        )
        transformations = standard_transformations + (implicit_multiplication_application,)
        return parse_expr(s, transformations=transformations, evaluate=True)
    except Exception:
        try:
            from sympy import sympify
            return sympify(s)
        except Exception:
            return None


def is_math_like(text: Any) -> bool:
    """Đáp án có 'giống toán' để đáng áp sympy không (số/biểu thức ngắn),
    tránh chấm nhầm câu trả lời bằng văn xuôi."""
    if text is None:
        return False
    s = str(text).strip()
    if not s or len(s) > 80:
        return False
    if re.search(r"[0-9]", s) or re.search(r"[+\-*/^=(){}\[\]]", s):
        # ít chữ cái (không phải câu văn)
        letters = len(re.findall(r"[A-Za-zÀ-ỹ]", s))
        return letters <= 12
    return False


def answers_equivalent(a: Any, b: Any) -> bool | None:
    """True/False nếu quyết định được bằng sympy; None nếu không tính được
    (khi đó phải nhờ LLM/GV)."""
    if a is None or b is None:
        return None
    sa, sb = str(a).strip(), str(b).strip()
    if sa == sb:
        return True
    ea, eb = _to_expr(a), _to_expr(b)
    if ea is None or eb is None:
        # so khớp số nếu cả hai là số
        na, nb = _num(sa), _num(sb)
        if na is not None and nb is not None:
            return abs(na - nb) < 1e-6
        return None
    try:
        from sympy import simplify
        diff = simplify(ea - eb)
        if diff == 0:
            return True
        # thử số hoá
        val = complex(diff)
        return abs(val) < 1e-6
    except Exception:
        try:
            return bool(ea.equals(eb))
        except Exception:
            return None


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


def check_distractors(correct: Any, options: list[dict[str, Any]],
                      correct_key: Any = None) -> dict[str, Any]:
    """Trả về danh sách phương án NHIỄU trùng đáp án đúng hoặc trùng nhau (cần loại/sinh lại).
    Bỏ qua chính phương án đúng (correct_key) — nó ĐÚNG là bằng đáp án."""
    dupes = []
    seen_texts: set[str] = set()
    ck = str(correct_key).strip().upper() if correct_key is not None else None
    for opt in options:
        text = opt.get("text") if isinstance(opt, dict) else opt
        key = opt.get("key") if isinstance(opt, dict) else None
        if ck is not None and str(key).strip().upper() == ck:
            continue  # phương án đúng, không tính là nhiễu
        norm = str(text).strip().lower()
        eq = answers_equivalent(text, correct)
        is_dup_text = norm in seen_texts
        seen_texts.add(norm)
        if eq is True or is_dup_text:
            dupes.append({"key": key, "text": text, "reason": "trung_dap_an" if eq is True else "trung_option"})
    return {"ok": len(dupes) == 0, "duplicates": dupes}


def verify_final_answer(claimed_answer: Any, reference_answer: Any) -> dict[str, Any]:
    """Đối chiếu đáp án cuối AI đưa ra với đáp án tham chiếu (nếu có)."""
    eq = answers_equivalent(claimed_answer, reference_answer)
    return {
        "verifiable": eq is not None,
        "match": bool(eq) if eq is not None else None,
        "claimed": None if claimed_answer is None else str(claimed_answer),
        "reference": None if reference_answer is None else str(reference_answer),
    }
