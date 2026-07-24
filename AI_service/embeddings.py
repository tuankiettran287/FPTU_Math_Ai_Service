"""Embedding bằng bge-m3 (đa ngôn ngữ, 1024 chiều) — dùng cho tìm câu tương tự
và RAG.

Hai backend (config EMBEDDING_BACKEND):
  - "remote" (mặc định): gọi HTTP endpoint OpenAI-compatible /embeddings trên GPU
    pod. Máy dev KHÔNG có GPU nên KHÔNG bao giờ load model tại chỗ.
  - "local": nạp SentenceTransformer trong tiến trình (chỉ khi máy đủ khoẻ).
"""
import threading
from typing import Any

from .config import (
    EMBEDDING_API_KEY, EMBEDDING_API_MODEL, EMBEDDING_API_URL, EMBEDDING_BACKEND,
    EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_TIMEOUT,
)

_model = None
_load_lock = threading.Lock()


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_model():
    """Chỉ dùng cho backend 'local'. Không gọi ở máy không có GPU."""
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL, device=_resolve_device(EMBEDDING_DEVICE))
        return _model


def _embed_remote(texts: list[str]) -> list[list[float]]:
    import httpx

    url = EMBEDDING_API_URL.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}"}
    out: list[list[float]] = []
    with httpx.Client(timeout=EMBEDDING_TIMEOUT) as http:
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            resp = http.post(url, headers=headers, json={"model": EMBEDDING_API_MODEL, "input": batch})
            resp.raise_for_status()
            data = resp.json()["data"]
            out.extend(item["embedding"] for item in data)
    return out


def _embed_local(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Trả về list vector (mỗi vector EMBEDDING_DIM chiều)."""
    if not texts:
        return []
    if EMBEDDING_BACKEND == "local":
        return _embed_local(texts)
    return _embed_remote(texts)


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def _join(*parts: Any) -> str:
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, (list, tuple)):
            out.extend(str(x) for x in p if x)
        else:
            out.append(str(p))
    return " \n".join(out)


def build_question_text(record: dict[str, Any]) -> str:
    """Ghép các trường có ý nghĩa của 1 câu hỏi thành 'document' để embed.

    Chấp nhận cả bản ghi dataset thô lẫn QuestionBankItem đã chuẩn hoá.
    """
    q = record.get("question")
    if isinstance(q, dict):
        q_text = _join(q.get("text"), q.get("latex"))
        options = q.get("options") or []
        opt_text = _join(*[o.get("text") if isinstance(o, dict) else o for o in options])
    else:
        q_text = str(q or "")
        opt_text = ""

    sol = record.get("solution")
    final_answer = sol.get("final_answer") if isinstance(sol, dict) else ""

    header = _join(
        record.get("subject"), record.get("course"), record.get("chapter"),
        record.get("topic"), record.get("subtopic"),
    )
    return _join(
        header,
        q_text,
        opt_text,
        record.get("concepts_used"),
        final_answer,
    ).strip()[:8000]


def dim() -> int:
    return EMBEDDING_DIM
