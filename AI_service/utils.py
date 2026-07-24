import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc):%Y%m%d%H%M%S}_{uuid4().hex[:8]}"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def loads_json(text: str) -> Any:
    return json.loads(text)


# Model nền Qwen → thỉnh thoảng chèn chữ Hán/Nhật/Hàn vào output tiếng Việt.
# Guard dùng CHUNG cho mọi feature có output hiển thị cho người dùng
# (benchmark 15/07: 32B rò CJK ~20% câu trả lời nếu không chặn).
_CJK_RE = re.compile(r"[一-鿿぀-ヿ]")


def has_cjk(*texts: Any) -> bool:
    return any(_CJK_RE.search(str(t or "")) for t in texts)


def dict_has_cjk(obj: Any) -> bool:
    """Quét ĐỆ QUY mọi chuỗi trong dict/list để phát hiện chữ Hán.

    Dùng cho guard dùng chung ở llm.generate_json: output của mọi feature là JSON
    lồng nhau (feedback, steps[].content, mistakes[].what…), rò CJK có thể nằm ở
    bất kỳ tầng nào. Quét phẳng từng field thủ công như bản cũ dễ bỏ sót field mới.
    """
    if isinstance(obj, str):
        return bool(_CJK_RE.search(obj))
    if isinstance(obj, dict):
        return any(dict_has_cjk(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(dict_has_cjk(v) for v in obj)
    return False


# ── bóc đáp án cuối từ text tự do ─────────────────────────────────────────────
# R1 trả lời markdown + LaTeX: đáp án nằm trong \boxed{} (có ngoặc lồng) hoặc \( \),
# thường ở dòng SAU marker. Bóc kiểu "lấy dòng ngay sau marker" hỏng 128/180 câu
# với R1 (đo thật ở benchmark) → port lại extract_final() của harness.
_BOXED_RE = re.compile(r"\\boxed\{")
_JUNK_LINE_RE = re.compile(r"^[\s*_`:\-—]*(?:\\\[|\\\]|\\\(|\\\)|\$\$?)?[\s*_`:\-—]*$")


def _boxed_contents(s: str) -> list[str]:
    """Bóc nội dung mọi \\boxed{...}, có đếm ngoặc lồng (\\boxed{\\frac{1}{2}})."""
    out = []
    for m in _BOXED_RE.finditer(s):
        i = m.end()
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


def extract_final_answer(text: Any, marker: str = "ĐÁP ÁN CUỐI:") -> str | None:
    """Bóc đáp án cuối, chịu được cả hai phong cách:
    - Qwen: 'ĐÁP ÁN CUỐI: f(2)=3' ngay trên 1 dòng;
    - R1:   markdown + \\boxed{}/\\( \\) ở dòng sau marker.
    """
    if text is None:
        return None
    s = str(text)
    i = s.rfind(marker)
    seg = s[i + len(marker):] if i != -1 else s[-600:]
    seg = seg.strip()
    bx = _boxed_contents(seg)
    if bx:
        return " ; ".join(bx)[:300]
    # 32B không dùng \boxed mà dùng \( \) — gỡ vỏ inline-math trước khi lấy dòng.
    for line in seg.split("\n"):
        raw = line.strip()
        if not raw or _JUNK_LINE_RE.match(line):
            continue
        cleaned = raw.strip("*`_ ").strip()
        m = re.fullmatch(r"\\\((.+)\\\)", cleaned)
        if m:
            cleaned = m.group(1).strip()
        if cleaned:
            return cleaned[:300]
    return None


def strip_reasoning_blocks(text: str) -> str:
    # 1) Xoá cặp <think>…</think> đầy đủ.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 2) DeepSeek-R1 THƯỜNG emit reasoning KHÔNG có tag mở (template tự mở) → chỉ còn
    #    </think> lơ lửng ở cuối phần suy luận. Bỏ TẤT CẢ nội dung tới & gồm </think> cuối.
    lower = text.lower()
    if "</think>" in lower:
        text = text[lower.rfind("</think>") + len("</think>"):]
    # 3) Bỏ mọi tag <think>/</think> lẻ còn sót.
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def _extract_balanced_json(text: str, opening: str, closing: str) -> str | None:
    start = text.find(opening)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_reasoning_blocks(text)
    candidate = _extract_balanced_json(cleaned, "{", "}")
    if not candidate:
        raise ValueError(f"Model output does not contain a JSON object: {cleaned[:500]}")

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object.")
    return payload


def parse_json_array_or_object(text: str) -> Any:
    cleaned = strip_reasoning_blocks(text)
    object_candidate = _extract_balanced_json(cleaned, "{", "}")
    array_candidate = _extract_balanced_json(cleaned, "[", "]")

    candidates = [value for value in [object_candidate, array_candidate] if value]
    if not candidates:
        raise ValueError(f"Model output does not contain JSON: {cleaned[:500]}")

    candidate = min(candidates, key=lambda value: cleaned.find(value))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model output JSON is invalid: {exc}") from exc
