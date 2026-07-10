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


def strip_reasoning_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
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
