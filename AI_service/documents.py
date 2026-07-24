"""Ingest tài liệu (PDF/DOCX/TXT) → chunk theo mục → embed → lưu document_chunks.
Phục vụ RAG: tìm kiếm tài liệu (màn 1/6), hỏi-đáp có trích nguồn, và sinh quiz từ file.

BE gửi presigned URL (KHÔNG gửi credential). AI tải file, trích text, chunk, embed.
"""
import io
import re
from typing import Any

import httpx

from . import db, rag
from .embeddings import embed_texts
from .llm import client


# ── trích text ────────────────────────────────────────────────────────────────
def _download(url: str, timeout: float = 120) -> bytes:
    with httpx.Client(timeout=timeout, follow_redirects=True) as http:
        r = http.get(url)
        r.raise_for_status()
        return r.content


def extract_text(data: bytes, extension: str) -> str:
    ext = (extension or "").lower().lstrip(".")
    if ext == "pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if ext in ("docx", "doc"):
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    # txt/md/khác
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return ""


# ── chunk theo mục (định lý/mục con), không cắt cơ học giữa câu ────────────────
_SECTION_RE = re.compile(
    r"^(?:chương|chapter|section|mục|bài|định lý|theorem|định nghĩa|definition|ví dụ|example)\b"
    r"|^\d+(?:\.\d+)*\s+\S",
    re.IGNORECASE,
)


def chunk_text(text: str, target: int = 900, overlap: int = 120) -> list[dict[str, Any]]:
    """Chia theo đoạn/mục; gộp tới ~target ký tự, ưu tiên ranh giới mục."""
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[dict[str, Any]] = []
    buf = ""
    section = None
    for p in paras:
        first_line = p.splitlines()[0][:80] if p else ""
        is_header = bool(_SECTION_RE.match(p.strip()))
        # gặp header mới hoặc buf quá dài → chốt chunk
        if buf and (is_header or len(buf) + len(p) > target):
            chunks.append({"content": buf.strip(), "section": section})
            tail = buf[-overlap:] if len(buf) > overlap else ""
            buf = tail
        if is_header:
            section = first_line
        buf += ("\n\n" if buf else "") + p
    if buf.strip():
        chunks.append({"content": buf.strip(), "section": section})
    return chunks


# ── ingest ────────────────────────────────────────────────────────────────────
def ingest(*, document_id: str, file_url: str = "", text: str = "", extension: str = "pdf",
           title: str = "", subject: str = "", course: str = "", chapter: str = "",
           metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tải + trích + chunk + embed + lưu. Trả {document_id, indexed_chunks, status}."""
    try:
        raw_text = text or extract_text(_download(file_url), extension)
    except Exception as exc:
        return {"document_id": document_id, "indexed_chunks": 0, "status": "failed", "error": str(exc)}

    pieces = chunk_text(raw_text)
    if not pieces:
        return {"document_id": document_id, "indexed_chunks": 0, "status": "not_supported",
                "error": "Không trích được nội dung văn bản."}

    try:
        vectors = embed_texts([c["content"] for c in pieces])
    except Exception as exc:
        return {"document_id": document_id, "indexed_chunks": 0, "status": "failed",
                "error": f"Embedding lỗi: {exc}"}

    db.upsert_document(document_id, subject=subject, course=course, title=title,
                       source_path=file_url or None, metadata=metadata or {})
    db.delete_document_chunks(document_id)  # ingest lại → xoá chunk cũ
    # Ghi tất cả chunk trong 1 connection (KHÔNG mở connection mỗi chunk — xem upsert_chunks_bulk).
    db.upsert_chunks_bulk([
        {"document_id": document_id, "ordinal": i, "content": c["content"], "embedding": v,
         "subject": subject, "course": course, "chapter": chapter, "section": c.get("section"),
         "metadata": {"title": title}}
        for i, (c, v) in enumerate(zip(pieces, vectors))
    ])
    return {"document_id": document_id, "indexed_chunks": len(pieces), "status": "indexed"}


def delete(document_id: str) -> dict[str, Any]:
    db.delete_document(document_id)
    return {"document_id": document_id, "status": "deleted"}


# ── search / ask ──────────────────────────────────────────────────────────────
def _filters(subject: str | None, chapter: str | None, allowed_ids: list[str] | None) -> dict[str, Any]:
    f: dict[str, Any] = {}
    if subject:
        f["subject"] = subject
    if chapter:
        f["chapter"] = chapter
    return f


def search(*, query: str, subject: str = "", chapter: str = "", limit: int = 5,
           allowed_document_ids: list[str] | None = None) -> dict[str, Any]:
    chunks = rag.retrieve(query, k=limit, filters=_filters(subject, chapter, allowed_document_ids))
    if allowed_document_ids:
        allow = set(allowed_document_ids)
        chunks = [c for c in chunks if c.get("document_id") in allow]
    results = [{
        "chunk_id": c.get("chunk_id"), "document_id": c.get("document_id"),
        "title": c.get("title"), "section_title": c.get("section"),
        "chapter_name": c.get("chapter"), "content": c.get("content"),
        "content_preview": (c.get("content") or "")[:240], "score": c.get("score", 0.0),
        "subject_code": c.get("subject"),
    } for c in chunks]
    return {"query": query, "results": results}


def ask(*, message: str, subject: str = "", chapter: str = "", limit: int = 5,
        allowed_document_ids: list[str] | None = None, history: list[dict] | None = None,
        analysis_context: str = "", language: str = "vi") -> dict[str, Any]:
    chunks = rag.retrieve(message, k=limit, filters=_filters(subject, chapter, allowed_document_ids))
    if allowed_document_ids:
        allow = set(allowed_document_ids)
        chunks = [c for c in chunks if c.get("document_id") in allow]
    context = rag.format_context(chunks)

    lang_rule = ("Answer in ENGLISH." if str(language).lower() == "en"
                 else "Trả lời bằng TIẾNG VIỆT.")
    messages = [{"role": "system", "content":
                 "Bạn là trợ lý học tập. Trả lời dựa trên CONTEXT tài liệu (trích [1],[2]... khi dùng). "
                 "Nếu có DỮ LIỆU LỚP thì dùng để phân tích. Không bịa nguồn. " + lang_rule}]
    for turn in (history or [])[-5:]:
        role = "assistant" if str(turn.get("role")) == "assistant" else "user"
        messages.append({"role": role, "content": str(turn.get("content", ""))})
    user = message
    if analysis_context:
        user += f"\n\nDỮ LIỆU LỚP:\n{analysis_context}"
    if context:
        user += f"\n\nCONTEXT:\n{context}"
    messages.append({"role": "user", "content": user})

    answer = client.chat(messages, task="concept_lookup", max_new_tokens=1200, temperature=0.2)
    from .utils import strip_reasoning_blocks
    return {"answer": strip_reasoning_blocks(answer), "sources": rag.sources(chunks)}


def quiz_from_url(*, file_url: str, extension: str = "pdf", count: int = 5,
                  subject: str = "", chapter: str = "", language: str = "vi") -> dict[str, Any]:
    """Sinh MCQ từ nội dung 1 file (Resource Hub)."""
    from .features import generate_questions
    try:
        text = extract_text(_download(file_url), extension)
    except Exception as exc:
        return {"questions": [], "status": "failed", "error": str(exc)}
    if not text.strip():
        return {"questions": [], "status": "not_supported", "error": "Không trích được nội dung."}

    res = generate_questions(subject=subject or "MAE", count=count, qtype="mcq", chapter=chapter,
                             language=language, extra_instructions="Dựa trên nội dung tài liệu sau:\n" + text[:6000])
    questions = []
    for q in res.get("questions", []):
        qq = q.get("question", {})
        opts = {o.get("key"): o.get("text") for o in qq.get("options", [])}
        questions.append({
            "question": qq.get("text", ""), "options": opts,
            "correct_key": (q.get("evaluation") or {}).get("correct_key", "A"),
            "explanation": (q.get("metadata") or {}).get("explanation"),
        })
    return {"questions": questions, "status": "ok"}
