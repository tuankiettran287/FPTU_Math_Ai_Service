import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from .config import DATA_BANK_FIELDS
from .utils import as_list, now_iso, read_question_from_image, read_text_file


def normalize_steps(value: Any) -> List[Dict[str, Any]]:
    steps = as_list(value)
    normalized = []
    for index, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            normalized.append(
                {
                    "step_number": step.get("step_number", index),
                    "title": step.get("title", f"Step {index}"),
                    "content": step.get("content") or step.get("reasoning") or "",
                    "formula": step.get("formula") or step.get("latex"),
                }
            )
        else:
            normalized.append(
                {
                    "step_number": index,
                    "title": f"Step {index}",
                    "content": str(step),
                    "formula": None,
                }
            )
    return normalized


def normalize_item(raw: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    generated_at = now_iso()
    item: Dict[str, Any] = {field: raw.get(field) for field in DATA_BANK_FIELDS}

    item["id"] = str(item.get("id") or f"MATHAI_GEN_{datetime.now():%Y%m%d%H%M%S}_{uuid4().hex[:8]}")
    item["subject"] = str(item.get("subject") or args.subject)
    item["course"] = str(item.get("course") or args.course)
    item["chapter"] = str(item.get("chapter") or args.chapter)
    item["topic"] = str(item.get("topic") or args.topic)
    item["subtopic"] = str(item.get("subtopic") or args.subtopic or args.topic)
    item["question_type"] = str(item.get("question_type") or args.question_type)

    difficulty = item.get("difficulty")
    if not isinstance(difficulty, dict):
        difficulty = {"level": difficulty or args.difficulty}
    difficulty.setdefault("level", args.difficulty)
    difficulty.setdefault("score", {"easy": 3, "medium": 5, "hard": 8}.get(str(difficulty["level"]), 5))
    difficulty.setdefault("estimated_time_minutes", {"easy": 5, "medium": 10, "hard": 15}.get(str(difficulty["level"]), 10))
    difficulty.setdefault("cognitive_level", "apply")
    item["difficulty"] = difficulty

    question = item.get("question")
    if not isinstance(question, dict):
        question = {"text": question or raw.get("question_text") or "", "latex": raw.get("latex"), "image": None}
    question.setdefault("text", "")
    question.setdefault("latex", None)
    question.setdefault("image", None)
    item["question"] = question

    solution = item.get("solution")
    if not isinstance(solution, dict):
        solution = {}
    solution.setdefault("final_answer", raw.get("expected_answer") or raw.get("answer") or "")
    solution.setdefault("steps", normalize_steps(raw.get("solution_steps") or raw.get("steps")))
    solution.setdefault("alternative_solutions", as_list(raw.get("alternative_solutions")))
    item["solution"] = solution

    item["concepts_used"] = as_list(item.get("concepts_used") or raw.get("concepts"))
    item["prerequisites"] = as_list(item.get("prerequisites"))
    item["common_mistakes"] = as_list(item.get("common_mistakes"))
    item["hints"] = as_list(item.get("hints"))

    evaluation = item.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    evaluation.setdefault("answer_verifiable", True)
    evaluation.setdefault("step_verifiable", bool(solution.get("steps")))
    item["evaluation"] = evaluation

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.setdefault("source", "AI Generated")
    metadata.setdefault("language", args.language)
    metadata.setdefault("created_by", "AI")
    metadata.setdefault("verified", False)
    metadata.setdefault("generated_at", generated_at)
    item["metadata"] = metadata

    return item


def text_for_embedding(item: Dict[str, Any]) -> str:
    question = item.get("question") or {}
    solution = item.get("solution") or {}
    difficulty = item.get("difficulty") or {}
    parts = [
        item.get("course"),
        item.get("chapter"),
        item.get("topic"),
        item.get("subtopic"),
        difficulty.get("level"),
        item.get("question_type"),
        question.get("text"),
        question.get("latex"),
        solution.get("final_answer"),
    ]
    for step in solution.get("steps") or []:
        if isinstance(step, dict):
            parts.extend([step.get("title"), step.get("content"), step.get("formula")])
    parts.extend(item.get("concepts_used") or [])
    parts.extend(item.get("hints") or [])
    return "\n".join(str(part) for part in parts if part)


def load_json_bank(path: Path) -> Iterable[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    for row in data:
        if isinstance(row, dict):
            yield row


def read_problem_text_from_args(args: argparse.Namespace) -> str:
    question_text = getattr(args, "question", "") or getattr(args, "ocr_text", "")
    if not question_text and getattr(args, "question_file", None):
        question_text = read_text_file(args.question_file)
    if not question_text and getattr(args, "image_path", None):
        question_text = read_question_from_image(args.image_path)
    return question_text.strip()


def read_answer_text_from_args(args: argparse.Namespace) -> str:
    answer_text = getattr(args, "student_answer", "") or getattr(args, "answer_ocr_text", "")
    if not answer_text and getattr(args, "student_answer_file", None):
        answer_text = read_text_file(args.student_answer_file)
    if not answer_text and getattr(args, "answer_image_path", None):
        answer_text = read_question_from_image(args.answer_image_path)
    return answer_text.strip()


def build_question_payload(conn, args: argparse.Namespace, fetch_question_document) -> Dict[str, Any]:
    if getattr(args, "question_id", ""):
        return fetch_question_document(conn, args.table, args.question_id)

    question_text = read_problem_text_from_args(args)
    if not question_text:
        raise SystemExit("Pass --question-id, --question, --question-file, --ocr-text, or --image-path.")

    return normalize_item(
        {
            "id": f"QUESTION_{uuid4().hex[:12]}",
            "question": {
                "text": question_text,
                "latex": getattr(args, "latex", "") or None,
                "image": str(args.image_path) if getattr(args, "image_path", None) else None,
            },
            "metadata": {
                "source": "User Uploaded Image" if getattr(args, "image_path", None) else "User Input",
                "language": getattr(args, "language", "en"),
                "created_by": "User",
                "verified": False,
            },
        },
        args,
    )
