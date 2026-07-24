from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from .config import APP_NAME, APP_VERSION, DATABASE_URL, LLM_BACKEND, MODEL_NAME, VLLM_MODEL
from .db import get_evaluation, get_problem, init_db, list_problems, save_evaluation, save_exam_generation, save_problem
from .llm import client
from . import competency, documents, features, ocr
from .schemas import (
    AnalyzeLearningRequest,
    ArenaQuestionsRequest,
    CompetencyGenerateRequest,
    CompetencyGradeRequest,
    CompetencyHintRequest,
    ConceptLookupRequest,
    DocumentAskRequest,
    DocumentSearchRequest,
    ExplainMistakeRequest,
    FromMistakesRequest,
    IngestDocumentRequest,
    OcrRequest,
    QuizFromUrlRequest,
    GenerateExamRequest,
    GeneratedExamResponse,
    GenerateQuestionsRequest,
    GradeRequest,
    GradeEssayRequest,
    GradeSubmissionRequest,
    GradeSubmissionResponse,
    ProblemListResponse,
    QuestionBankItem,
    SolveProblemRequest,
    SolveProblemResponse,
    SolveRequest,
    StudyPathRequest,
    normalize_grade_result,
    normalize_question_bank_item,
)
from .utils import new_id


def _feature(fn, **kwargs) -> dict[str, Any]:
    try:
        return fn(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Model returned invalid JSON: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
    # Báo model ĐANG THỰC SỰ phục vụ: khi backend=vllm thì đó là VLLM_MODEL
    # (--served-model-name của vLLM), KHÔNG phải MODEL_NAME. MODEL_NAME chỉ là repo HF
    # cho nhánh fallback transformers → báo nó ra là nói sai model đang chạy.
    serving = VLLM_MODEL if LLM_BACKEND == "vllm" else MODEL_NAME
    return {
        "status": "ok",
        "model": serving,
        "backend": LLM_BACKEND,
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


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — 9 tính năng AI
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/generate/questions")           # 1. Ra đề MCQ/tự luận
def api_generate_questions(req: GenerateQuestionsRequest) -> dict[str, Any]:
    return _feature(features.generate_questions, **req.model_dump())


@app.post("/api/v1/grade")                          # 2. Chấm không rubric (2 lớp)
def api_grade(req: GradeRequest) -> dict[str, Any]:
    return _feature(features.grade, **req.model_dump())


@app.post("/api/v1/grade/essay")                    # 2b. Chấm tự luận (có ý a/b/c)
def api_grade_essay(req: GradeEssayRequest) -> dict[str, Any]:
    payload = req.model_dump()
    # sub_questions là list[EssaySubQuestion] → features.grade_essay nhận dict thuần.
    payload["sub_questions"] = [s.model_dump() for s in req.sub_questions]
    return _feature(features.grade_essay, **payload)


@app.post("/api/v1/explain-mistake")                # 3. Giải thích lỗi + lý thuyết (RAG)
def api_explain_mistake(req: ExplainMistakeRequest) -> dict[str, Any]:
    return _feature(features.explain_mistake, **req.model_dump())


@app.post("/api/v1/concept")                        # 4. Tra cứu khái niệm (RAG + rewrite)
def api_concept(req: ConceptLookupRequest) -> dict[str, Any]:
    return _feature(features.concept_lookup, **req.model_dump())


@app.post("/api/v1/analyze-learning")               # 5. Phân tích học tập
def api_analyze_learning(req: AnalyzeLearningRequest) -> dict[str, Any]:
    return _feature(features.analyze_learning, **req.model_dump())


@app.post("/api/v1/solve")                          # 6. Giải full / gợi ý từng bước (nhận cả ảnh/file)
def api_solve(req: SolveRequest) -> dict[str, Any]:
    return _feature(features.solve, **req.model_dump())


@app.post("/api/v1/ocr")                            # OCR ảnh → text/LaTeX (chatbot tải ảnh)
def api_ocr(req: OcrRequest) -> dict[str, Any]:
    return _feature(ocr.ocr, **req.model_dump())


@app.post("/api/v1/questions/embed")                # Embed 1 câu hỏi vừa thêm (admin QB)
def api_embed_question(body: dict[str, Any]) -> dict[str, Any]:
    from .db import get_question, set_question_embedding
    from .embeddings import build_question_text, embed_text
    qid = (body or {}).get("id")
    if not qid:
        raise HTTPException(status_code=400, detail="Thiếu id câu hỏi.")
    doc = get_question(qid)  # trả về document (bản ghi gốc) hoặc None
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy câu hỏi: {qid}")
    try:
        vector = embed_text(build_question_text(doc))
        ok = set_question_embedding(qid, vector)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Embedding lỗi: {exc}") from exc
    return {"id": qid, "embedded": bool(ok)}


@app.post("/api/v1/arena/questions")                # 7. Ra đề arena theo ELO
def api_arena_questions(req: ArenaQuestionsRequest) -> dict[str, Any]:
    return _feature(features.arena_questions, **req.model_dump())


@app.post("/api/v1/generate/from-mistakes")         # 8. Ra đề theo lỗi sai
def api_from_mistakes(req: FromMistakesRequest) -> dict[str, Any]:
    return _feature(features.generate_from_mistakes, **req.model_dump())


@app.post("/api/v1/study-path")                     # 9. Lộ trình học
def api_study_path(req: StudyPathRequest) -> dict[str, Any]:
    return _feature(features.study_path, **req.model_dump())


# ══════════════════════════════════════════════════════════════════════════════
#  Đánh giá năng lực (competency) — framework MẬT, chỉ đọc; import bằng script.
#  Câu hỏi sinh ra KHÔNG lưu vào math_question_bank (BE giữ snapshot riêng).
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/competency/framework")            # BE đọc khung để build blueprint + FE hiển thị tên
def api_competency_framework(subject: str = Query(min_length=3)) -> dict[str, Any]:
    return _feature(competency.framework, subject=subject)


@app.post("/api/v1/competency/generate")            # sinh + kiểm định bộ câu tự luận
def api_competency_generate(req: CompetencyGenerateRequest) -> dict[str, Any]:
    payload = req.model_dump()
    payload["targets"] = [t.model_dump() for t in req.targets]
    return _feature(competency.generate, **payload)


@app.post("/api/v1/competency/grade")               # chấm 2 khung độc lập (KHÔNG RAG)
def api_competency_grade(req: CompetencyGradeRequest) -> dict[str, Any]:
    payload = req.model_dump()
    payload["hint_events"] = [e.model_dump() for e in req.hint_events]
    return _feature(competency.grade, **payload)


@app.post("/api/v1/competency/hint")                # gợi ý 5 mức trong lúc làm bài
def api_competency_hint(req: CompetencyHintRequest) -> dict[str, Any]:
    return _feature(competency.hint, **req.model_dump())


# ── RAG tài liệu (Phase 4) ────────────────────────────────────────────────────
@app.post("/api/v1/documents/ingest")               # index tài liệu (admin upload — màn 9)
def api_ingest(req: IngestDocumentRequest) -> dict[str, Any]:
    return _feature(documents.ingest, **req.model_dump())


@app.delete("/api/v1/documents/{document_id}")      # xoá tài liệu khỏi RAG
def api_delete_document(document_id: str) -> dict[str, Any]:
    return _feature(documents.delete, document_id=document_id)


@app.post("/api/v1/documents/search")               # tìm tài liệu (màn 1/6)
def api_doc_search(req: DocumentSearchRequest) -> dict[str, Any]:
    return _feature(documents.search, **req.model_dump())


@app.post("/api/v1/documents/ask")                  # hỏi-đáp có trích nguồn (màn 2/6)
def api_doc_ask(req: DocumentAskRequest) -> dict[str, Any]:
    return _feature(documents.ask, **req.model_dump())


@app.post("/api/v1/documents/quiz")                 # sinh quiz từ file (Resource Hub)
def api_doc_quiz(req: QuizFromUrlRequest) -> dict[str, Any]:
    return _feature(documents.quiz_from_url, **req.model_dump())
