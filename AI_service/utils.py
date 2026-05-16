import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"AI output does not contain a JSON object:\n{text}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI output is not valid JSON: {exc}\n{text}") from exc
    if not isinstance(payload, dict):
        raise ValueError("AI output JSON must be an object.")
    return payload


def parse_json_or_raw(text: str) -> Dict[str, Any]:
    try:
        return parse_json_object(text)
    except ValueError:
        return {"raw_output": text}


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"


def checked_identifier(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"{label} must be a safe SQL identifier, got: {value}")
    return value


def missing_dependency_error(exc: ModuleNotFoundError) -> SystemExit:
    return SystemExit(
        f"Missing Python package: {exc.name}. Install dependencies first with: "
        "python -m pip install -r requirements.txt"
    )


def read_question_from_image(image_path: Path) -> str:
    try:
        from PIL import Image
        import pytesseract
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Cannot OCR image because package '{exc.name}' is missing. "
            "Install OCR dependencies, or pass the problem text with --question/--ocr-text."
        ) from exc

    if not image_path.exists():
        raise SystemExit(f"Image file not found: {image_path}")

    text = pytesseract.image_to_string(Image.open(image_path), lang="eng").strip()
    if not text:
        raise SystemExit("OCR did not detect any text. Pass the problem text with --question/--ocr-text.")
    return text

