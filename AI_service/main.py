from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from .config import APP_NAME, APP_VERSION, DATABASE_URL, MODEL_NAME
from .db import get_evaluation, get_problem, init_db, list_problems, save_evaluation, save_exam_generation, save_problem
from .llm import client
from .schemas import (
    GenerateExamRequest,
    GeneratedExamResponse,
    GradeSubmissionRequest,
    GradeSubmissionResponse,
    ProblemListResponse,
    QuestionBankItem,
    SolveProblemRequest,
    SolveProblemResponse,
    normalize_grade_result,
    normalize_question_bank_item,
)
from .utils import new_id


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db(DATABASE_URL)
    yield


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="FastAPI service for exam generation, problem solving, and student-answer grading.",
    lifespan=lifespan,
)


def _generate_json(task: str, payload: dict[str, Any], controls: Any) -> dict[str, Any]:
    try:
        return client.generate_json(
            task=task,
            payload=payload,
            max_new_tokens=getattr(controls, "max_new_tokens", None),
            temperature=getattr(controls, "temperature", None),
            top_p=getattr(controls, "top_p", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _extract_exam_questions(model_payload: dict[str, Any]) -> list[dict[str, Any]]:
    questions = model_payload.get("questions") or model_payload.get("items") or model_payload.get("problems")
    if not isinstance(questions, list) or not questions:
        raise HTTPException(status_code=502, detail="Model output must contain a non-empty questions array.")
    normalized = [item for item in questions if isinstance(item, dict)]
    if not normalized:
        raise HTTPException(status_code=502, detail="Model output questions array does not contain JSON objects.")
    return normalized


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "model": MODEL_NAME,
        "database": DATABASE_URL,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": client.is_loaded,
    }


@app.get("/api/v1/schema/problem")
def problem_schema() -> dict[str, Any]:
    return QuestionBankItem.model_json_schema()


@app.post("/api/v1/exams/generate-from-document", response_model=GeneratedExamResponse)
def generate_exam_from_document(request: GenerateExamRequest) -> GeneratedExamResponse:
    payload = request.model_dump(mode="json")
    model_payload = _generate_json("exam_generation", payload, request)
    raw_questions = _extract_exam_questions(model_payload)

    exam_id = new_id("EXAM")
    problems: list[QuestionBankItem] = []
    for raw_question in raw_questions:
        raw_question.setdefault("question_type", "multiple_choice")
        problem = normalize_question_bank_item(
            raw_question,
            request,
            source_task="exam_generation",
        )
        problem.metadata["exam_id"] = exam_id
        save_problem(problem, source_task="exam_generation")
        problems.append(problem)

    coverage_report = model_payload.get("coverage_report")
    if not isinstance(coverage_report, dict):
        coverage_report = {
            "covered_items": request.coverage_requirements,
            "question_count": len(problems),
            "note": "Model did not provide a detailed coverage_report.",
        }

    save_exam_generation(
        exam_id=exam_id,
        request_payload=payload,
        coverage_report=coverage_report,
        problem_ids=[problem.id for problem in problems],
    )
    return GeneratedExamResponse(
        exam_id=exam_id,
        saved_count=len(problems),
        coverage_report=coverage_report,
        problems=problems,
    )


@app.post("/api/v1/problems/solve", response_model=SolveProblemResponse)
def solve_problem(request: SolveProblemRequest) -> SolveProblemResponse:
    payload = request.model_dump(mode="json")
    model_payload = _generate_json("solve_problem", payload, request)
    raw_problem = model_payload.get("problem") or model_payload.get("item") or model_payload
    if not isinstance(raw_problem, dict):
        raise HTTPException(status_code=502, detail="Model output must contain a problem object.")

    problem = normalize_question_bank_item(
        raw_problem,
        request,
        source_task="solve_problem",
        fallback_question_text=request.problem_text,
    )
    problem.question.text = problem.question.text or request.problem_text
    if request.latex and not problem.question.latex:
        problem.question.latex = request.latex

    save_problem(problem, source_task="solve_problem")
    return SolveProblemResponse(problem_id=problem.id, problem=problem)


@app.post("/api/v1/submissions/grade", response_model=GradeSubmissionResponse)
def grade_submission(request: GradeSubmissionRequest) -> GradeSubmissionResponse:
    stored_problem = None
    if request.question_id:
        stored_problem = get_problem(request.question_id)
        if stored_problem is None:
            raise HTTPException(status_code=404, detail=f"Question not found: {request.question_id}")

    if stored_problem is None and not request.question_text:
        raise HTTPException(status_code=400, detail="Pass question_id or question_text.")

    question_payload: dict[str, Any]
    if stored_problem is not None:
        question_payload = stored_problem.model_dump(mode="json")
        if request.standard_solution:
            question_payload["teacher_standard_solution"] = request.standard_solution
    else:
        question_payload = {
            "question": {"text": request.question_text},
            "solution": request.standard_solution,
        }

    payload = request.model_dump(mode="json")
    payload["question_payload"] = question_payload

    model_payload = _generate_json("grade_submission", payload, request)
    result = normalize_grade_result(model_payload, request, request.question_id)
    save_evaluation(result, payload)
    return GradeSubmissionResponse(evaluation_id=result.id, result=result)


@app.get("/api/v1/problems", response_model=ProblemListResponse)
def get_problem_list(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ProblemListResponse:
    total, items = list_problems(limit=limit, offset=offset)
    return ProblemListResponse(total=total, items=items)


@app.get("/api/v1/problems/{problem_id}", response_model=QuestionBankItem)
def get_problem_detail(problem_id: str) -> QuestionBankItem:
    problem = get_problem(problem_id)
    if problem is None:
        raise HTTPException(status_code=404, detail=f"Problem not found: {problem_id}")
    return problem


@app.get("/api/v1/evaluations/{evaluation_id}")
def get_evaluation_detail(evaluation_id: str) -> dict[str, Any]:
    evaluation = get_evaluation(evaluation_id)
    if evaluation is None:
        raise HTTPException(status_code=404, detail=f"Evaluation not found: {evaluation_id}")
    return evaluation.model_dump(mode="json")
