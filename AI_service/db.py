import argparse
from typing import Any, Dict, List, Sequence
from uuid import uuid4

from .config import (
    AI_CLASS_ANALYTICS_TABLE,
    AI_EVALUATIONS_TABLE,
    AI_INTERACTIONS_TABLE,
    JSON_COLUMNS,
    TEXT_COLUMNS,
)
from .utils import checked_identifier, missing_dependency_error, vector_literal


def ensure_question_schema(conn, table_name: str, vector_dim: int) -> None:
    try:
        from psycopg import sql
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    checked_identifier(table_name, "table name")
    table = sql.Identifier(table_name)
    dim = sql.SQL(str(int(vector_dim)))
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    subject TEXT,
                    course TEXT,
                    chapter TEXT,
                    topic TEXT,
                    subtopic TEXT,
                    difficulty JSONB NOT NULL,
                    question_type TEXT,
                    question JSONB NOT NULL,
                    solution JSONB NOT NULL,
                    concepts_used JSONB NOT NULL,
                    prerequisites JSONB NOT NULL,
                    common_mistakes JSONB NOT NULL,
                    hints JSONB NOT NULL,
                    evaluation JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({dim}) NOT NULL,
                    document JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            ).format(table=table, dim=dim)
        )
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} ON {table} "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            ).format(index=sql.Identifier(f"{table_name}_embedding_idx"), table=table)
        )
    conn.commit()


def ensure_ai_tables(conn, question_table: str, vector_dim: int) -> None:
    try:
        from psycopg import sql
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    ensure_question_schema(conn, question_table, vector_dim)
    dim = sql.SQL(str(int(vector_dim)))
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id UUID PRIMARY KEY,
                    feature TEXT NOT NULL,
                    role_context TEXT,
                    user_id TEXT,
                    student_id TEXT,
                    class_id TEXT,
                    question_id TEXT,
                    input JSONB NOT NULL,
                    output JSONB NOT NULL,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            ).format(table=sql.Identifier(AI_INTERACTIONS_TABLE), dim=dim)
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id UUID PRIMARY KEY,
                    question_id TEXT,
                    student_id TEXT,
                    class_id TEXT,
                    submission_id TEXT,
                    student_answer TEXT NOT NULL,
                    verdict TEXT,
                    score NUMERIC,
                    max_score NUMERIC,
                    feedback JSONB NOT NULL,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            ).format(table=sql.Identifier(AI_EVALUATIONS_TABLE), dim=dim)
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {table} (
                    id UUID PRIMARY KEY,
                    class_id TEXT NOT NULL,
                    course TEXT,
                    input JSONB NOT NULL,
                    analysis JSONB NOT NULL,
                    embedding vector({dim}),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            ).format(table=sql.Identifier(AI_CLASS_ANALYTICS_TABLE), dim=dim)
        )
        cur.execute(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} ON {table} "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            ).format(
                index=sql.Identifier(f"{AI_INTERACTIONS_TABLE}_embedding_idx"),
                table=sql.Identifier(AI_INTERACTIONS_TABLE),
            )
        )
    conn.commit()


def upsert_question_item(conn, table_name: str, item: Dict[str, Any], embedding: Sequence[float]) -> None:
    try:
        from psycopg import sql
        from psycopg.types.json import Jsonb
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    table = sql.Identifier(table_name)
    columns = TEXT_COLUMNS + JSON_COLUMNS + ["embedding", "document"]
    values = []
    for column in TEXT_COLUMNS:
        values.append(item[column])
    for column in JSON_COLUMNS:
        values.append(Jsonb(item[column]))
    values.append(vector_literal(embedding))
    values.append(Jsonb(item))

    assignments = [
        sql.SQL("{column} = EXCLUDED.{column}").format(column=sql.Identifier(column))
        for column in columns
        if column != "id"
    ]
    assignments.append(sql.SQL("updated_at = now()"))

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {table} ({columns})
                VALUES ({placeholders})
                ON CONFLICT (id) DO UPDATE SET {assignments}
                """
            ).format(
                table=table,
                columns=sql.SQL(", ").join(sql.Identifier(column) for column in columns),
                placeholders=sql.SQL(", ").join(
                    sql.SQL("%s::vector") if column == "embedding" else sql.SQL("%s")
                    for column in columns
                ),
                assignments=sql.SQL(", ").join(assignments),
            ),
            values,
        )
    conn.commit()


def fetch_question_document(conn, table_name: str, question_id: str) -> Dict[str, Any]:
    try:
        from psycopg import sql
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    checked_identifier(table_name, "table name")
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT document FROM {table} WHERE id = %s").format(table=sql.Identifier(table_name)),
            [question_id],
        )
        row = cur.fetchone()
    if not row:
        raise SystemExit(f"Question id not found in database: {question_id}")
    return row[0]


def search_question_documents(conn, table_name: str, query_embedding: str, limit: int) -> List[Dict[str, Any]]:
    try:
        from psycopg import sql
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    checked_identifier(table_name, "table name")
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT document, 1 - (embedding <=> %s::vector) AS similarity
                FROM {table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """
            ).format(table=sql.Identifier(table_name)),
            [query_embedding, query_embedding, limit],
        )
        rows = cur.fetchall()
    results = []
    for document, similarity in rows:
        if isinstance(document, dict):
            document = dict(document)
            document["_similarity"] = float(similarity)
            results.append(document)
    return results


def save_interaction(
    conn,
    args: argparse.Namespace,
    feature: str,
    input_payload: Dict[str, Any],
    output_payload: Dict[str, Any],
    embedding_text: str,
) -> str:
    try:
        from psycopg import sql
        from psycopg.types.json import Jsonb
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    embedder = SentenceTransformer(args.embedding_model)
    embedding = vector_literal(embedder.encode(embedding_text, normalize_embeddings=True).tolist())
    interaction_id = str(uuid4())
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {table} (
                    id, feature, role_context, user_id, student_id, class_id, question_id,
                    input, output, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                """
            ).format(table=sql.Identifier(AI_INTERACTIONS_TABLE)),
            [
                interaction_id,
                feature,
                getattr(args, "role_context", None),
                getattr(args, "user_id", None),
                getattr(args, "student_id", None),
                getattr(args, "class_id", None),
                getattr(args, "question_id", None),
                Jsonb(input_payload),
                Jsonb(output_payload),
                embedding,
            ],
        )
    conn.commit()
    return interaction_id


def save_evaluation(
    conn,
    args: argparse.Namespace,
    feedback: Dict[str, Any],
    student_answer: str,
    embedding_text: str,
) -> str:
    try:
        from psycopg import sql
        from psycopg.types.json import Jsonb
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    embedder = SentenceTransformer(args.embedding_model)
    embedding = vector_literal(embedder.encode(embedding_text, normalize_embeddings=True).tolist())
    evaluation_id = str(uuid4())
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {table} (
                    id, question_id, student_id, class_id, submission_id, student_answer,
                    verdict, score, max_score, feedback, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                """
            ).format(table=sql.Identifier(AI_EVALUATIONS_TABLE)),
            [
                evaluation_id,
                getattr(args, "question_id", None),
                getattr(args, "student_id", None),
                getattr(args, "class_id", None),
                getattr(args, "submission_id", None),
                student_answer,
                feedback.get("verdict"),
                feedback.get("score"),
                feedback.get("max_score"),
                Jsonb(feedback),
                embedding,
            ],
        )
    conn.commit()
    return evaluation_id


def save_class_analytics(
    conn,
    args: argparse.Namespace,
    input_payload: Any,
    analysis_payload: Dict[str, Any],
    embedding_text: str,
) -> str:
    try:
        from psycopg import sql
        from psycopg.types.json import Jsonb
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc

    embedder = SentenceTransformer(args.embedding_model)
    embedding = vector_literal(embedder.encode(embedding_text, normalize_embeddings=True).tolist())
    analytics_id = str(uuid4())
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {table} (id, class_id, course, input, analysis, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """
            ).format(table=sql.Identifier(AI_CLASS_ANALYTICS_TABLE)),
            [
                analytics_id,
                args.class_id,
                args.course,
                Jsonb(input_payload),
                Jsonb(analysis_payload),
                embedding,
            ],
        )
    conn.commit()
    return analytics_id
