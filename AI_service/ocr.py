"""OCR ảnh → text/LaTeX cho chatbot (đề chụp hoặc viết tay).

DeepSeek-R1-Distill là model CHỈ-VĂN-BẢN, không nhìn được ảnh. Nên khi người dùng
tải ảnh lên chatbot, ta OCR ảnh thành text (kèm công thức LaTeX nếu có) rồi mới đưa
vào LLM để giải/gợi ý.

CHẠY TRÊN POD (GPU) — KHÔNG load model OCR trên máy dev. Backend cấu hình qua
`OCR_BACKEND` (rapidocr | pix2tex | vlm | none). Các thư viện OCR chỉ import khi dùng
để không phá vỡ môi trường vLLM đã pin.
"""
from __future__ import annotations

import base64
import binascii
import io
from functools import lru_cache
from typing import Any

from . import config


class OcrUnavailable(RuntimeError):
    """OCR chưa cài/không cấu hình → FE báo người dùng thay vì crash."""


def _decode_image(image_base64: str) -> bytes:
    data = image_base64.strip()
    # Chấp nhận data URI "data:image/png;base64,...."
    if data.startswith("data:"):
        _, _, data = data.partition(",")
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Ảnh base64 không hợp lệ: {exc}") from exc
    if not raw:
        raise ValueError("Ảnh rỗng.")
    if len(raw) > config.OCR_MAX_IMAGE_BYTES:
        raise ValueError(f"Ảnh quá lớn (> {config.OCR_MAX_IMAGE_BYTES // (1024*1024)}MB).")
    return raw


# ── backend: RapidOCR (onnxruntime, CPU) — text in ấn ─────────────────────────
@lru_cache(maxsize=1)
def _rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:  # noqa: BLE001
        raise OcrUnavailable(
            "Chưa cài rapidocr-onnxruntime trên pod (pip install rapidocr-onnxruntime)."
        ) from exc
    return RapidOCR()


def _ocr_rapidocr(raw: bytes) -> str:
    import numpy as np
    from PIL import Image

    engine = _rapidocr()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    result, _ = engine(np.array(img))
    if not result:
        return ""
    # result: list[[box, text, score]] theo thứ tự đọc.
    return "\n".join(line[1] for line in result if len(line) >= 2 and line[1]).strip()


# ── backend: pix2tex (LaTeX-OCR) — công thức toán → LaTeX ─────────────────────
@lru_cache(maxsize=1)
def _pix2tex():
    try:
        from pix2tex.cli import LatexOCR
    except Exception as exc:  # noqa: BLE001
        raise OcrUnavailable(
            "Chưa cài pix2tex trên pod (pip install 'pix2tex[gui]' hoặc pix2tex)."
        ) from exc
    return LatexOCR()


def _ocr_pix2tex(raw: bytes) -> str:
    from PIL import Image

    model = _pix2tex()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    latex = model(img)
    return f"$$ {latex} $$" if latex else ""


# ── backend: VLM (vision model OpenAI-compatible) — hiểu ảnh tốt nhất ──────────
def _ocr_vlm(raw: bytes, prompt: str = "") -> str:
    import httpx

    if not config.OCR_VLM_BASE_URL:
        raise OcrUnavailable("Chưa cấu hình OCR_VLM_BASE_URL cho backend 'vlm'.")
    b64 = base64.b64encode(raw).decode()
    instruction = prompt or (
        "Chép lại CHÍNH XÁC toàn bộ nội dung trong ảnh (đề toán), giữ nguyên công thức "
        "dưới dạng LaTeX ($...$). Chỉ trả về nội dung, không giải thích."
    )
    body = {
        "model": config.OCR_VLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": 1500,
    }
    url = config.OCR_VLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {config.OCR_VLM_API_KEY}"}
    with httpx.Client(timeout=config.OCR_TIMEOUT) as http:
        r = http.post(url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


# ── API công khai ─────────────────────────────────────────────────────────────
def image_to_text(image_base64: str, *, prompt: str = "") -> str:
    """OCR 1 ảnh (base64) → text (kèm LaTeX nếu backend hỗ trợ). Có thể trả "" nếu ảnh trống."""
    backend = config.OCR_BACKEND
    if backend == "none":
        raise OcrUnavailable("OCR đang tắt (OCR_BACKEND=none).")
    raw = _decode_image(image_base64)
    if backend == "vlm":
        return _ocr_vlm(raw, prompt=prompt)
    if backend == "pix2tex":
        return _ocr_pix2tex(raw)
    return _ocr_rapidocr(raw)  # mặc định


def ocr(*, image_base64: str, prompt: str = "", language: str = "vi") -> dict[str, Any]:
    """Endpoint /ocr: trả text đã trích + backend đã dùng."""
    try:
        text = image_to_text(image_base64, prompt=prompt)
        return {"text": text, "backend": config.OCR_BACKEND, "status": "ok"}
    except OcrUnavailable as exc:
        return {"text": "", "backend": config.OCR_BACKEND, "status": "unavailable", "error": str(exc)}
    except ValueError as exc:
        return {"text": "", "backend": config.OCR_BACKEND, "status": "invalid", "error": str(exc)}
