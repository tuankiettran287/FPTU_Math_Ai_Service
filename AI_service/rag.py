"""RAG cho giải thích lỗi & tra cứu khái niệm.

- retrieve(): embed truy vấn → tìm đoạn tài liệu gần nhất (document_chunks).
- rewrite_query(): với câu hỏi ngắn/mơ hồ ("đạo hàm là gì"), nhờ LLM viết lại cho
  rõ nghĩa (kèm ngữ cảnh môn/chương) trước khi embed.

Khi kho tài liệu chưa được nạp (document_chunks rỗng) thì retrieve trả [] — feature
vẫn chạy (LLM trả lời, không trích nguồn).
"""
from typing import Any

from . import db
from .llm import client


def rewrite_query(query: str, *, subject: str | None = None, chapter: str | None = None) -> str:
    ctx = " ".join(x for x in [subject, chapter] if x)
    try:
        text = client.chat(
            [
                {"role": "system", "content": "Bạn viết lại câu hỏi toán ngắn/mơ hồ thành một truy vấn tìm kiếm rõ nghĩa (1 câu, tiếng Việt), thêm ngữ cảnh môn/chương nếu có. Chỉ trả về câu truy vấn."},
                {"role": "user", "content": f"Ngữ cảnh: {ctx or 'không rõ'}\nCâu hỏi: {query}\nTruy vấn tìm kiếm:"},
            ],
            task="concept_lookup", max_new_tokens=120, temperature=0.0,
        )
        text = text.strip().splitlines()[0].strip() if text.strip() else query
        return text or query
    except Exception:
        return query


def retrieve(query: str, *, k: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    from .embeddings import embed_text
    try:
        vec = embed_text(query)
    except Exception:
        return []
    try:
        return db.search_chunks(vec, k=k, filters=filters)
    except Exception:
        return []


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Ghép các đoạn thành context cho prompt (kèm nhãn nguồn để trích dẫn)."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        src = c.get("title") or c.get("document_id") or "tài liệu"
        loc = " · ".join(x for x in [c.get("chapter"), c.get("section")] if x)
        parts.append(f"[{i}] ({src}{' — ' + loc if loc else ''})\n{c.get('content', '')}")
    return "\n\n".join(parts)


def sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Danh sách nguồn (tài liệu/file) để trả về cho BE hiển thị."""
    seen, out = set(), []
    for c in chunks:
        doc = c.get("document_id")
        if doc in seen:
            continue
        seen.add(doc)
        out.append({
            "document_id": doc, "title": c.get("title"), "source_path": c.get("source_path"),
            "subject": c.get("subject"), "course": c.get("course"),
            "chapter": c.get("chapter"), "section": c.get("section"),
        })
    return out
