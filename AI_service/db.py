"""Tầng dữ liệu Postgres + pgvector cho AI service.

AI service giữ DB riêng (tra cứu câu hỏi theo id, tìm câu tương tự bằng embedding,
lưu lịch sử tương tác & chấm bài). BE không cần biết AI lưu ở đâu — mọi endpoint
đều trả đủ kết quả trong response.
"""
import json
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg

from .config import DATABASE_URL, EMBEDDING_DIM


# ── kết nối ───────────────────────────────────────────────────────────────────
def _dsn(database_url: str = DATABASE_URL) -> str:
    # psycopg chấp nhận cả 'postgresql://...' lẫn key=value; giữ nguyên URL.
    return database_url


@contextmanager
def connect(database_url: str = DATABASE_URL) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(_dsn(database_url))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _vec(values: list[float]) -> str:
    """Định dạng vector cho pgvector: '[v1,v2,...]' (dùng với ::vector)."""
    return "[" + ",".join(f"{float(x):.7g}" for x in values) + "]"


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


# ── khởi tạo schema ───────────────────────────────────────────────────────────
def init_db(database_url: str = DATABASE_URL) -> None:
    with connect(database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS math_question_bank (
                id              TEXT PRIMARY KEY,
                subject         TEXT,
                assest_format   TEXT,
                course          TEXT,
                chapter         TEXT,
                topic           TEXT,
                subtopic        TEXT,
                difficulty      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                question_type   TEXT,
                question        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                solution        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                concepts_used   JSONB NOT NULL DEFAULT '[]'::jsonb,
                prerequisites   JSONB NOT NULL DEFAULT '[]'::jsonb,
                common_mistakes JSONB NOT NULL DEFAULT '[]'::jsonb,
                hints           JSONB NOT NULL DEFAULT '[]'::jsonb,
                evaluation      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding       VECTOR({EMBEDDING_DIM}),
                document        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS mqb_subject_idx ON math_question_bank(subject);")
        conn.execute("CREATE INDEX IF NOT EXISTS mqb_chapter_idx ON math_question_bank(chapter);")
        conn.execute("CREATE INDEX IF NOT EXISTS mqb_qtype_idx ON math_question_bank(question_type);")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS mqb_embedding_idx ON math_question_bank "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        )

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS ai_interactions (
                id            UUID PRIMARY KEY,
                feature       TEXT NOT NULL,
                role_context  TEXT,
                user_id       TEXT,
                student_id    TEXT,
                class_id      TEXT,
                question_id   TEXT,
                input         JSONB NOT NULL,
                output        JSONB NOT NULL,
                embedding     VECTOR({EMBEDDING_DIM}),
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS ai_interactions_feature_idx ON ai_interactions(feature);")

        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS ai_evaluations (
                id              UUID PRIMARY KEY,
                question_id     TEXT,
                student_id      TEXT,
                class_id        TEXT,
                submission_id   TEXT,
                student_answer  TEXT NOT NULL,
                verdict         TEXT,
                score           NUMERIC,
                max_score       NUMERIC,
                feedback        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding       VECTOR({EMBEDDING_DIM}),
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )

        # RAG tài liệu (dùng cho giải thích lỗi / tra cứu khái niệm — Phase 3).
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id           TEXT PRIMARY KEY,
                subject      TEXT,
                course       TEXT,
                title        TEXT,
                source_path  TEXT,
                metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id           UUID PRIMARY KEY,
                document_id  TEXT REFERENCES documents(id) ON DELETE CASCADE,
                subject      TEXT,
                course       TEXT,
                chapter      TEXT,
                section      TEXT,
                ordinal      INTEGER,
                content      TEXT NOT NULL,
                embedding    VECTOR({EMBEDDING_DIM}),
                metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx ON document_chunks "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
        )

        # Khung năng lực (competency framework) — dữ liệu MẬT, chỉ nạp bằng
        # tools/import_competency_framework.py. KHÔNG có endpoint thêm/sửa/xoá.
        # kind: 'shared_policy' | 'gmc' | 'subject' (subject_code chỉ có ở kind='subject').
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS competency_framework (
                id            UUID PRIMARY KEY,
                framework_id  TEXT NOT NULL,
                version       TEXT NOT NULL,
                kind          TEXT NOT NULL,
                subject_code  TEXT NOT NULL DEFAULT '',
                payload       JSONB NOT NULL,
                imported_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE (framework_id, kind, subject_code)
            );
            """
        )


# ── math_question_bank ────────────────────────────────────────────────────────
_QB_COLS = [
    "id", "subject", "assest_format", "course", "chapter", "topic", "subtopic",
    "difficulty", "question_type", "question", "solution", "concepts_used",
    "prerequisites", "common_mistakes", "hints", "evaluation", "metadata",
]


def _qb_row(record: dict[str, Any], embedding: list[float] | None) -> list[Any]:
    def js(key: str, default: Any) -> str:
        val = record.get(key)
        return _j(val if val is not None else default)

    diff = record.get("difficulty")
    if not isinstance(diff, (dict, list)):
        diff = {"level": diff} if diff else {}

    return [
        str(record.get("id") or uuid.uuid4().hex),
        record.get("subject"),
        record.get("assest_format"),
        record.get("course"),
        record.get("chapter"),
        record.get("topic"),
        record.get("subtopic"),
        _j(diff),
        record.get("question_type"),
        js("question", {}),
        js("solution", {}),
        js("concepts_used", []),
        js("prerequisites", []),
        js("common_mistakes", []),
        js("hints", []),
        js("evaluation", {}),
        js("metadata", {}),
        _vec(embedding) if embedding else None,
        _j(record),  # document = bản ghi gốc đầy đủ
    ]


_UPSERT_SQL = """
INSERT INTO math_question_bank (
    id, subject, assest_format, course, chapter, topic, subtopic,
    difficulty, question_type, question, solution, concepts_used,
    prerequisites, common_mistakes, hints, evaluation, metadata,
    embedding, document
) VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s::jsonb, %s, %s::jsonb, %s::jsonb, %s::jsonb,
    %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb,
    %s::vector, %s::jsonb
)
ON CONFLICT (id) DO UPDATE SET
    subject = EXCLUDED.subject, assest_format = EXCLUDED.assest_format,
    course = EXCLUDED.course, chapter = EXCLUDED.chapter, topic = EXCLUDED.topic,
    subtopic = EXCLUDED.subtopic, difficulty = EXCLUDED.difficulty,
    question_type = EXCLUDED.question_type, question = EXCLUDED.question,
    solution = EXCLUDED.solution, concepts_used = EXCLUDED.concepts_used,
    prerequisites = EXCLUDED.prerequisites, common_mistakes = EXCLUDED.common_mistakes,
    hints = EXCLUDED.hints, evaluation = EXCLUDED.evaluation, metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding, document = EXCLUDED.document, updated_at = now()
"""


def upsert_question(record: dict[str, Any], embedding: list[float] | None = None,
                    database_url: str = DATABASE_URL) -> str:
    row = _qb_row(record, embedding)
    with connect(database_url) as conn:
        conn.execute(_UPSERT_SQL, row)
    return row[0]


def upsert_questions_bulk(items: list[tuple[dict[str, Any], list[float] | None]],
                          conn: psycopg.Connection | None = None,
                          database_url: str = DATABASE_URL) -> int:
    rows = [_qb_row(rec, emb) for rec, emb in items]
    if conn is not None:
        conn.cursor().executemany(_UPSERT_SQL, rows)
        return len(rows)
    with connect(database_url) as c:
        c.cursor().executemany(_UPSERT_SQL, rows)
    return len(rows)


def get_question(question_id: str, database_url: str = DATABASE_URL) -> dict[str, Any] | None:
    with connect(database_url) as conn:
        row = conn.execute(
            "SELECT document FROM math_question_bank WHERE id = %s", [question_id]
        ).fetchone()
    return row[0] if row else None


def set_question_embedding(question_id: str, embedding: list[float],
                           database_url: str = DATABASE_URL) -> bool:
    """Ghi embedding cho 1 câu hỏi đã có (BE gọi sau khi admin thêm câu mới)."""
    with connect(database_url) as conn:
        cur = conn.execute(
            "UPDATE math_question_bank SET embedding = %s::vector, updated_at = now() WHERE id = %s",
            [_vec(embedding), question_id],
        )
        return cur.rowcount > 0


def count_questions(database_url: str = DATABASE_URL) -> int:
    with connect(database_url) as conn:
        return int(conn.execute("SELECT count(*) FROM math_question_bank").fetchone()[0])


def list_questions(limit: int = 50, offset: int = 0,
                   database_url: str = DATABASE_URL) -> tuple[int, list[dict[str, Any]]]:
    with connect(database_url) as conn:
        total = int(conn.execute("SELECT count(*) FROM math_question_bank").fetchone()[0])
        rows = conn.execute(
            "SELECT document FROM math_question_bank ORDER BY created_at DESC LIMIT %s OFFSET %s",
            [limit, offset],
        ).fetchall()
    return total, [r[0] for r in rows]


def _filter_clause(filters: dict[str, Any] | None, prefix: str = "") -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    clauses, params = [], []
    for col in ("subject", "course", "chapter", "topic", "question_type"):
        val = filters.get(col)
        if val:
            if isinstance(val, (list, tuple)):
                clauses.append(f"{prefix}{col} = ANY(%s)")
                params.append(list(val))
            else:
                clauses.append(f"{prefix}{col} = %s")
                params.append(val)
    level = filters.get("difficulty")
    if level:
        clauses.append(f"{prefix}difficulty->>'level' = %s")
        params.append(level)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def search_questions(embedding: list[float], k: int = 5, filters: dict[str, Any] | None = None,
                     database_url: str = DATABASE_URL) -> list[dict[str, Any]]:
    """Tìm câu tương tự bằng cosine distance (kèm lọc metadata tuỳ chọn)."""
    where, params = _filter_clause(filters)
    sql = (
        "SELECT document, 1 - (embedding <=> %s::vector) AS score "
        f"FROM math_question_bank{where} "
        "ORDER BY embedding <=> %s::vector LIMIT %s"
    )
    vec = _vec(embedding)
    args = [vec, *params, vec, k]
    with connect(database_url) as conn:
        rows = conn.execute(sql, args).fetchall()
    out = []
    for doc, score in rows:
        item = dict(doc)
        item["_score"] = float(score)
        out.append(item)
    return out


def draw_questions(filters: dict[str, Any] | None = None, k: int = 5, random: bool = True,
                   database_url: str = DATABASE_URL) -> list[dict[str, Any]]:
    """Rút câu theo bộ lọc (dùng cho ra đề arena/exam khi cần lấy từ ngân hàng)."""
    where, params = _filter_clause(filters)
    order = "random()" if random else "created_at DESC"
    sql = f"SELECT document FROM math_question_bank{where} ORDER BY {order} LIMIT %s"
    with connect(database_url) as conn:
        rows = conn.execute(sql, [*params, k]).fetchall()
    return [r[0] for r in rows]


# ── documents / RAG ───────────────────────────────────────────────────────────
def upsert_document(doc_id: str, *, subject: str | None = None, course: str | None = None,
                    title: str | None = None, source_path: str | None = None,
                    metadata: dict[str, Any] | None = None, database_url: str = DATABASE_URL) -> str:
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO documents (id, subject, course, title, source_path, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET subject=EXCLUDED.subject, course=EXCLUDED.course,
                title=EXCLUDED.title, source_path=EXCLUDED.source_path, metadata=EXCLUDED.metadata
            """,
            [doc_id, subject, course, title, source_path, _j(metadata or {})],
        )
    return doc_id


def upsert_chunk(*, document_id: str, ordinal: int, content: str, embedding: list[float] | None,
                 subject: str | None = None, course: str | None = None, chapter: str | None = None,
                 section: str | None = None, metadata: dict[str, Any] | None = None,
                 database_url: str = DATABASE_URL) -> str:
    cid = str(uuid.uuid4())
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO document_chunks
                (id, document_id, subject, course, chapter, section, ordinal, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
            """,
            [cid, document_id, subject, course, chapter, section, ordinal, content,
             _vec(embedding) if embedding else None, _j(metadata or {})],
        )
    return cid


def upsert_chunks_bulk(items: list[dict[str, Any]], database_url: str = DATABASE_URL) -> int:
    """Ghi NHIỀU chunk trong DUY NHẤT 1 connection.

    Mở connection Postgres qua SSH tunnel tới VM tốn ~2s/lần → nếu mỗi chunk gọi
    upsert_chunk riêng (mở/đóng connection mỗi lần) thì 1 tài liệu 65 chunk ≈ 65×2s ≈ 150s
    → vượt timeout HttpClient/proxy (120s) → 524. Gom tất cả insert vào 1 connection.

    items: list dict {document_id, ordinal, content, embedding, subject?, course?, chapter?,
                      section?, metadata?}
    """
    if not items:
        return 0
    rows = [
        (str(uuid.uuid4()), it["document_id"], it.get("subject"), it.get("course"),
         it.get("chapter"), it.get("section"), it["ordinal"], it["content"],
         _vec(it["embedding"]) if it.get("embedding") else None, _j(it.get("metadata") or {}))
        for it in items
    ]
    with connect(database_url) as conn:
        conn.cursor().executemany(
            """
            INSERT INTO document_chunks
                (id, document_id, subject, course, chapter, section, ordinal, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)
            """,
            rows,
        )
    return len(rows)


def delete_document(document_id: str, database_url: str = DATABASE_URL) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM documents WHERE id = %s", [document_id])  # cascade → chunks


def delete_document_chunks(document_id: str, database_url: str = DATABASE_URL) -> None:
    with connect(database_url) as conn:
        conn.execute("DELETE FROM document_chunks WHERE document_id = %s", [document_id])


def search_chunks(embedding: list[float], k: int = 5, filters: dict[str, Any] | None = None,
                  database_url: str = DATABASE_URL) -> list[dict[str, Any]]:
    """Tìm đoạn tài liệu gần nhất (cosine) — cho RAG giải thích lỗi / tra cứu khái niệm."""
    where, params = _filter_clause(filters, prefix="c.")
    sql = (
        "SELECT c.id, c.document_id, c.content, c.chapter, c.section, c.metadata, "
        "d.title, d.source_path, d.subject, d.course, 1 - (c.embedding <=> %s::vector) AS score "
        "FROM document_chunks c LEFT JOIN documents d ON d.id = c.document_id"
        + where
        + " ORDER BY c.embedding <=> %s::vector LIMIT %s"
    )
    vec = _vec(embedding)
    with connect(database_url) as conn:
        rows = conn.execute(sql, [vec, *params, vec, k]).fetchall()
    out = []
    for r in rows:
        out.append({
            "chunk_id": str(r[0]), "document_id": r[1], "content": r[2], "chapter": r[3],
            "section": r[4], "metadata": r[5], "title": r[6], "source_path": r[7],
            "subject": r[8], "course": r[9], "score": float(r[10]),
        })
    return out


# ── competency_framework (đọc/ghi qua import script, KHÔNG có API ghi) ────────
def upsert_framework_part(*, framework_id: str, version: str, kind: str,
                          payload: dict[str, Any], subject_code: str = "",
                          database_url: str = DATABASE_URL) -> None:
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO competency_framework (id, framework_id, version, kind, subject_code, payload)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (framework_id, kind, subject_code) DO UPDATE SET
                version = EXCLUDED.version, payload = EXCLUDED.payload, imported_at = now()
            """,
            [str(uuid.uuid4()), framework_id, version, kind, subject_code, _j(payload)],
        )


def get_framework_subject(subject_code: str,
                          database_url: str = DATABASE_URL) -> dict[str, Any] | None:
    """Trả gói framework đủ để sinh đề/chấm cho 1 môn:
    payload môn + shared_policy + khung GMC chung. None nếu môn chưa được import."""
    with connect(database_url) as conn:
        rows = conn.execute(
            """
            SELECT kind, payload, version FROM competency_framework
            WHERE (kind = 'subject' AND subject_code = %s) OR kind IN ('shared_policy', 'gmc')
            """,
            [subject_code],
        ).fetchall()
    parts = {kind: payload for kind, payload, _ in rows}
    if "subject" not in parts:
        return None
    version = next((v for k, _, v in rows if k == "subject"), "")
    return {
        "subject_code": subject_code,
        "version": version,
        "subject": parts["subject"],
        "shared_policy": parts.get("shared_policy") or {},
        "gmc": parts.get("gmc") or {},
    }


def list_framework_subjects(database_url: str = DATABASE_URL) -> list[str]:
    with connect(database_url) as conn:
        rows = conn.execute(
            "SELECT subject_code FROM competency_framework WHERE kind = 'subject' ORDER BY subject_code"
        ).fetchall()
    return [r[0] for r in rows]


# ── ai_interactions ───────────────────────────────────────────────────────────
def save_interaction(feature: str, input_payload: dict[str, Any], output_payload: dict[str, Any],
                     *, role_context: str | None = None, user_id: str | None = None,
                     student_id: str | None = None, class_id: str | None = None,
                     question_id: str | None = None, embedding: list[float] | None = None,
                     database_url: str = DATABASE_URL) -> str:
    iid = str(uuid.uuid4())
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO ai_interactions
                (id, feature, role_context, user_id, student_id, class_id, question_id, input, output, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::vector)
            """,
            [iid, feature, role_context, user_id, student_id, class_id, question_id,
             _j(input_payload), _j(output_payload), _vec(embedding) if embedding else None],
        )
    return iid


# ── ai_evaluations ────────────────────────────────────────────────────────────
def save_ai_evaluation(*, student_answer: str, feedback: dict[str, Any] | list[Any] | str,
                       question_id: str | None = None, student_id: str | None = None,
                       class_id: str | None = None, submission_id: str | None = None,
                       verdict: str | None = None, score: float | None = None,
                       max_score: float | None = None, embedding: list[float] | None = None,
                       database_url: str = DATABASE_URL) -> str:
    eid = str(uuid.uuid4())
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO ai_evaluations
                (id, question_id, student_id, class_id, submission_id, student_answer,
                 verdict, score, max_score, feedback, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            """,
            [eid, question_id, student_id, class_id, submission_id, student_answer,
             verdict, score, max_score, _j(feedback), _vec(embedding) if embedding else None],
        )
    return eid


# ── shim tương thích cho main.py hiện tại (Phase 1 giữ 3 endpoint cũ chạy được) ─
def _embed_record(record: dict[str, Any]) -> list[float] | None:
    try:
        from .embeddings import build_question_text, embed_text
        text = build_question_text(record)
        return embed_text(text) if text else None
    except Exception:
        return None


def save_problem(problem, source_task: str, database_url: str = DATABASE_URL) -> str:
    record = problem.model_dump(mode="json") if hasattr(problem, "model_dump") else dict(problem)
    record.setdefault("metadata", {})
    if isinstance(record["metadata"], dict):
        record["metadata"]["source_task"] = source_task
    return upsert_question(record, _embed_record(record), database_url)


def get_problem(problem_id: str, database_url: str = DATABASE_URL):
    from .schemas import QuestionBankItem
    doc = get_question(problem_id, database_url)
    return QuestionBankItem.model_validate(doc) if doc else None


def list_problems(limit: int = 50, offset: int = 0, database_url: str = DATABASE_URL):
    from .schemas import QuestionBankItem
    total, docs = list_questions(limit, offset, database_url)
    return total, [QuestionBankItem.model_validate(d) for d in docs]


def save_exam_generation(exam_id: str, request_payload: dict[str, Any],
                         coverage_report: dict[str, Any], problem_ids: list[str],
                         database_url: str = DATABASE_URL) -> None:
    save_interaction(
        "exam_generation", request_payload,
        {"exam_id": exam_id, "coverage_report": coverage_report, "problem_ids": problem_ids},
        database_url=database_url,
    )


def save_evaluation(result, request_payload: dict[str, Any], database_url: str = DATABASE_URL) -> str:
    return save_ai_evaluation(
        student_answer=str(request_payload.get("student_answer") or ""),
        feedback=result.model_dump(mode="json") if hasattr(result, "model_dump") else result,
        question_id=result.question_id, student_id=request_payload.get("student_id"),
        class_id=request_payload.get("class_id"), submission_id=request_payload.get("submission_id"),
        verdict=result.verdict, score=result.score, max_score=result.max_score,
        database_url=database_url,
    )


def get_evaluation(evaluation_id: str, database_url: str = DATABASE_URL):
    from .schemas import GradeResult
    with connect(database_url) as conn:
        row = conn.execute(
            "SELECT feedback FROM ai_evaluations WHERE id = %s", [evaluation_id]
        ).fetchone()
    if not row:
        return None
    try:
        return GradeResult.model_validate(row[0])
    except Exception:
        return None
