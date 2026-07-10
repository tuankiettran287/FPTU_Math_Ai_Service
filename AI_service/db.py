import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import DATABASE_URL
from .schemas import GradeResult, QuestionBankItem
from .utils import dumps_json, loads_json, now_iso


def _sqlite_path(database_url: str = DATABASE_URL) -> Path:
    if database_url == "sqlite:///:memory:":
        return Path(":memory:")
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Only sqlite:/// DATABASE_URL is supported by this FastAPI service.")
    raw_path = database_url.removeprefix("sqlite:///")
    return Path(raw_path)


@contextmanager
def connect(database_url: str = DATABASE_URL) -> Iterator[sqlite3.Connection]:
    db_path = _sqlite_path(database_url)
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(database_url: str = DATABASE_URL) -> None:
    with connect(database_url) as conn:
        conn.executescript(
            """
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS problems (
                id TEXT PRIMARY KEY,
                source_task TEXT NOT NULL,
                subject TEXT NOT NULL,
                course TEXT NOT NULL,
                chapter TEXT NOT NULL,
                topic TEXT NOT NULL,
                subtopic TEXT,
                difficulty TEXT NOT NULL,
                question_type TEXT NOT NULL,
                document TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_problems_course ON problems(course);
            CREATE INDEX IF NOT EXISTS idx_problems_topic ON problems(topic);
            CREATE INDEX IF NOT EXISTS idx_problems_source_task ON problems(source_task);

            CREATE TABLE IF NOT EXISTS exam_generations (
                id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                coverage_report TEXT NOT NULL,
                problem_ids TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                question_id TEXT,
                student_id TEXT,
                class_id TEXT,
                submission_id TEXT,
                verdict TEXT NOT NULL,
                score REAL,
                max_score REAL NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(question_id) REFERENCES problems(id)
            );
            """
        )


def save_problem(problem: QuestionBankItem, source_task: str, database_url: str = DATABASE_URL) -> str:
    now = now_iso()
    document = problem.model_dump(mode="json")
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO problems (
                id, source_task, subject, course, chapter, topic, subtopic,
                difficulty, question_type, document, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_task = excluded.source_task,
                subject = excluded.subject,
                course = excluded.course,
                chapter = excluded.chapter,
                topic = excluded.topic,
                subtopic = excluded.subtopic,
                difficulty = excluded.difficulty,
                question_type = excluded.question_type,
                document = excluded.document,
                updated_at = excluded.updated_at
            """,
            [
                problem.id,
                source_task,
                problem.subject,
                problem.course,
                problem.chapter,
                problem.topic,
                problem.subtopic,
                problem.difficulty.level,
                problem.question_type,
                dumps_json(document),
                now,
                now,
            ],
        )
    return problem.id


def save_exam_generation(
    exam_id: str,
    request_payload: dict[str, Any],
    coverage_report: dict[str, Any],
    problem_ids: list[str],
    database_url: str = DATABASE_URL,
) -> None:
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO exam_generations (id, request_json, coverage_report, problem_ids, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                exam_id,
                dumps_json(request_payload),
                dumps_json(coverage_report),
                dumps_json(problem_ids),
                now_iso(),
            ],
        )


def get_problem(problem_id: str, database_url: str = DATABASE_URL) -> QuestionBankItem | None:
    with connect(database_url) as conn:
        row = conn.execute("SELECT document FROM problems WHERE id = ?", [problem_id]).fetchone()
    if not row:
        return None
    return QuestionBankItem.model_validate(loads_json(row["document"]))


def list_problems(limit: int = 50, offset: int = 0, database_url: str = DATABASE_URL) -> tuple[int, list[QuestionBankItem]]:
    with connect(database_url) as conn:
        total = conn.execute("SELECT COUNT(*) AS total FROM problems").fetchone()["total"]
        rows = conn.execute(
            "SELECT document FROM problems ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
    return int(total), [QuestionBankItem.model_validate(loads_json(row["document"])) for row in rows]


def save_evaluation(
    result: GradeResult,
    request_payload: dict[str, Any],
    database_url: str = DATABASE_URL,
) -> str:
    with connect(database_url) as conn:
        conn.execute(
            """
            INSERT INTO evaluations (
                id, question_id, student_id, class_id, submission_id, verdict,
                score, max_score, request_json, result_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.id,
                result.question_id,
                request_payload.get("student_id"),
                request_payload.get("class_id"),
                request_payload.get("submission_id"),
                result.verdict,
                result.score,
                result.max_score,
                dumps_json(request_payload),
                dumps_json(result.model_dump(mode="json")),
                result.created_at,
            ],
        )
    return result.id


def get_evaluation(evaluation_id: str, database_url: str = DATABASE_URL) -> GradeResult | None:
    with connect(database_url) as conn:
        row = conn.execute("SELECT result_json FROM evaluations WHERE id = ?", [evaluation_id]).fetchone()
    if not row:
        return None
    return GradeResult.model_validate(loads_json(row["result_json"]))
