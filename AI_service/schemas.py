from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .utils import as_list, new_id, now_iso


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Difficulty(ApiModel):
    level: str = Field(default="medium")
    score: float | None = None
    estimated_time_minutes: int | None = None
    cognitive_level: str = Field(default="apply")


class ChoiceOption(ApiModel):
    key: str
    text: str
    latex: str | None = None


class QuestionContent(ApiModel):
    text: str
    latex: str | None = None
    image: str | None = None
    options: list[ChoiceOption] = Field(default_factory=list)


class SolutionStep(ApiModel):
    step_number: int
    title: str
    content: str
    formula: str | None = None


class Solution(ApiModel):
    final_answer: str
    steps: list[SolutionStep] = Field(default_factory=list)
    alternative_solutions: list[dict[str, Any]] = Field(default_factory=list)


class QuestionBankItem(ApiModel):
    id: str
    subject: str = "MAE"
    course: str = "Mathematics for Engineering"
    chapter: str = "General"
    topic: str = "General"
    subtopic: str = ""
    difficulty: Difficulty = Field(default_factory=Difficulty)
    question_type: str = "problem_solving"
    question: QuestionContent
    solution: Solution
    concepts_used: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    common_mistakes: list[str] = Field(default_factory=list)
    hints: list[str] = Field(default_factory=list)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationControls(ApiModel):
    max_new_tokens: int | None = Field(default=None, ge=256, le=12000)
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    top_p: float | None = Field(default=None, ge=0.1, le=1.0)


class GenerateExamRequest(GenerationControls):
    document_text: str = Field(..., min_length=20)
    subject: str = "MAE"
    course: str = "Mathematics for Engineering"
    chapter: str = "General"
    topic: str = "General"
    subtopic: str = ""
    difficulty: str = "medium"
    question_count: int = Field(default=10, ge=1, le=50)
    choices_per_question: int = Field(default=4, ge=2, le=6)
    language: str = "vi"
    coverage_requirements: list[str] = Field(default_factory=list)


class SolveProblemRequest(GenerationControls):
    problem_text: str = Field(..., min_length=3)
    subject: str = "MAE"
    course: str = "Mathematics for Engineering"
    chapter: str = "General"
    topic: str = "General"
    subtopic: str = ""
    difficulty: str = "medium"
    problem_kind: str = "math"
    language: str = "vi"
    latex: str | None = None


class GradeSubmissionRequest(GenerationControls):
    student_answer: str = Field(..., min_length=1)
    question_id: str | None = None
    question_text: str | None = None
    standard_solution: dict[str, Any] | str | None = None
    rubric: str | None = None
    max_score: float = Field(default=10.0, gt=0)
    student_id: str | None = None
    class_id: str | None = None
    submission_id: str | None = None
    language: str = "vi"


class GeneratedExamResponse(ApiModel):
    exam_id: str
    saved_count: int
    coverage_report: dict[str, Any]
    problems: list[QuestionBankItem]


class SolveProblemResponse(ApiModel):
    problem_id: str
    problem: QuestionBankItem


class GradeResult(ApiModel):
    id: str
    question_id: str | None = None
    verdict: str
    is_correct: bool | None = None
    score: float | None = None
    max_score: float
    feedback: str | dict[str, Any] | list[Any]
    expected_answer: str | None = None
    mistakes: list[Any] = Field(default_factory=list)
    step_feedback: list[Any] = Field(default_factory=list)
    suggested_fix: str | list[Any] | dict[str, Any] | None = None
    explanation: str | None = None
    confidence: float | None = None
    created_at: str


class GradeSubmissionResponse(ApiModel):
    evaluation_id: str
    result: GradeResult


class ProblemListResponse(ApiModel):
    total: int
    items: list[QuestionBankItem]


def _get(context: Any, name: str, default: Any = None) -> Any:
    if isinstance(context, dict):
        return context.get(name, default)
    return getattr(context, name, default)


def _normalize_difficulty(raw: Any, default_level: str) -> dict[str, Any]:
    if isinstance(raw, dict):
        difficulty = dict(raw)
    else:
        difficulty = {"level": raw or default_level}
    level = str(difficulty.get("level") or default_level)
    difficulty["level"] = level
    difficulty.setdefault("score", {"easy": 3, "medium": 5, "hard": 8}.get(level, 5))
    difficulty.setdefault("estimated_time_minutes", {"easy": 5, "medium": 10, "hard": 15}.get(level, 10))
    difficulty.setdefault("cognitive_level", "apply")
    return difficulty


def _normalize_question(raw: dict[str, Any], fallback_text: str) -> dict[str, Any]:
    question = raw.get("question")
    if isinstance(question, dict):
        normalized = dict(question)
    else:
        normalized = {"text": question or raw.get("question_text") or fallback_text}

    normalized.setdefault("text", fallback_text)
    normalized.setdefault("latex", raw.get("latex"))
    normalized.setdefault("image", None)

    options = (
        normalized.get("options")
        or normalized.get("choices")
        or raw.get("options")
        or raw.get("choices")
        or raw.get("answers")
        or []
    )
    normalized["options"] = _normalize_options(options)
    return normalized


def _normalize_options(options: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if isinstance(options, dict):
        options = [{"key": key, "text": value} for key, value in options.items()]
    for index, option in enumerate(as_list(options), start=1):
        key = chr(64 + index)
        if isinstance(option, dict):
            normalized.append(
                {
                    "key": str(option.get("key") or option.get("label") or key),
                    "text": str(option.get("text") or option.get("content") or option.get("answer") or ""),
                    "latex": option.get("latex"),
                }
            )
        else:
            normalized.append({"key": key, "text": str(option), "latex": None})
    return [option for option in normalized if option["text"]]


def _normalize_steps(value: Any) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for index, step in enumerate(as_list(value), start=1):
        if isinstance(step, dict):
            steps.append(
                {
                    "step_number": int(step.get("step_number") or index),
                    "title": str(step.get("title") or f"Bước {index}"),
                    "content": str(step.get("content") or step.get("reasoning") or step.get("explanation") or ""),
                    "formula": step.get("formula") or step.get("latex"),
                }
            )
        else:
            steps.append({"step_number": index, "title": f"Bước {index}", "content": str(step), "formula": None})
    return [step for step in steps if step["content"] or step["formula"]]


def _normalize_solution(raw: dict[str, Any]) -> dict[str, Any]:
    solution = raw.get("solution")
    if isinstance(solution, dict):
        normalized = dict(solution)
    else:
        normalized = {}
    normalized.setdefault("final_answer", raw.get("final_answer") or raw.get("answer") or raw.get("correct_answer") or "")
    normalized["final_answer"] = str(normalized.get("final_answer") or "")
    normalized["steps"] = _normalize_steps(
        normalized.get("steps")
        or raw.get("solution_steps")
        or raw.get("steps")
        or raw.get("explanation_steps")
    )
    normalized["alternative_solutions"] = as_list(normalized.get("alternative_solutions"))
    return normalized


def normalize_question_bank_item(
    raw_item: dict[str, Any],
    context: Any,
    source_task: str,
    fallback_question_text: str = "",
) -> QuestionBankItem:
    item = dict(raw_item)
    language = _get(context, "language", "vi")

    normalized = {
        "id": str(item.get("id") or new_id("MATHAI")),
        "subject": str(item.get("subject") or _get(context, "subject", "MAE")),
        "course": str(item.get("course") or _get(context, "course", "Mathematics for Engineering")),
        "chapter": str(item.get("chapter") or _get(context, "chapter", "General")),
        "topic": str(item.get("topic") or _get(context, "topic", "General")),
        "subtopic": str(item.get("subtopic") or _get(context, "subtopic", "") or _get(context, "topic", "General")),
        "difficulty": _normalize_difficulty(item.get("difficulty"), _get(context, "difficulty", "medium")),
        "question_type": str(item.get("question_type") or item.get("type") or "problem_solving"),
        "question": _normalize_question(item, fallback_question_text),
        "solution": _normalize_solution(item),
        "concepts_used": [str(value) for value in as_list(item.get("concepts_used") or item.get("concepts"))],
        "prerequisites": [str(value) for value in as_list(item.get("prerequisites"))],
        "common_mistakes": [str(value) for value in as_list(item.get("common_mistakes"))],
        "hints": [str(value) for value in as_list(item.get("hints"))],
        "evaluation": item.get("evaluation") if isinstance(item.get("evaluation"), dict) else {},
        "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
    }

    normalized["evaluation"].setdefault("answer_verifiable", True)
    normalized["evaluation"].setdefault("step_verifiable", bool(normalized["solution"]["steps"]))
    normalized["metadata"].update(
        {
            "source": normalized["metadata"].get("source", "DeepSeek R1 Distill Qwen 7B"),
            "language": normalized["metadata"].get("language", language),
            "created_by": normalized["metadata"].get("created_by", "AI"),
            "verified": normalized["metadata"].get("verified", False),
            "source_task": source_task,
            "generated_at": normalized["metadata"].get("generated_at", now_iso()),
        }
    )
    return QuestionBankItem.model_validate(normalized)


def normalize_grade_result(raw_result: dict[str, Any], request: GradeSubmissionRequest, question_id: str | None) -> GradeResult:
    verdict = str(raw_result.get("verdict") or raw_result.get("result") or "SAI").upper()
    if verdict in {"TRUE", "CORRECT", "ĐÚNG", "DUNG"}:
        verdict = "DUNG"
    elif verdict in {"FALSE", "INCORRECT", "SAI"}:
        verdict = "SAI"

    result = {
        "id": str(raw_result.get("id") or new_id("EVAL")),
        "question_id": question_id,
        "verdict": verdict,
        "is_correct": raw_result.get("is_correct"),
        "score": raw_result.get("score"),
        "max_score": raw_result.get("max_score") or request.max_score,
        "feedback": raw_result.get("feedback") or raw_result.get("comment") or "",
        "expected_answer": raw_result.get("expected_answer") or raw_result.get("standard_answer"),
        "mistakes": as_list(raw_result.get("mistakes") or raw_result.get("errors")),
        "step_feedback": as_list(raw_result.get("step_feedback")),
        "suggested_fix": raw_result.get("suggested_fix") or raw_result.get("fix"),
        "explanation": raw_result.get("explanation") or raw_result.get("reason"),
        "confidence": raw_result.get("confidence"),
        "created_at": raw_result.get("created_at") or now_iso(),
    }
    if result["is_correct"] is None:
        result["is_correct"] = result["verdict"] == "DUNG"
    return GradeResult.model_validate(result)
