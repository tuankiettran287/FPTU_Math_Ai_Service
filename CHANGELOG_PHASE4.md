# Phase 4 — Migrate BE sang service AI mới (bỏ token) + RAG tài liệu

> Ngày: 2026-07-11. Chuyển toàn bộ điểm dùng AI của BE sang gọi service AI mới
> (1 URL, KHÔNG token), và bổ sung hạ tầng RAG tài liệu ở AI service.

## Chiến lược
Giữ nguyên 3 interface (`IArenaQuestionClient`, `IAiServiceClient`, `IRagServiceClient`)
nhưng **viết lại phần impl** để gọi endpoint mới → **tự động migrate mọi controller**
(AIController, AiSearchController, TeacherAiController, DocumentController, ResourceHub,
SelfStudyExam, ArenaQuestionService…) mà KHÔNG phải sửa từng cái.

## BE (`BE_KLTN`)
- **Config**: thêm `MathAiService:BaseUrl` (=`http://localhost:8080` dev). `DependencyInjection`
  trỏ CẢ 3 client về key này (fallback key cũ), timeout 180s. Bỏ cơ chế token.
- **`ArenaQuestionClient`** → `POST /api/v1/generate/questions` theo từng mức độ khó, map về
  `ArenaGeneratedQuestion`. Lỗi/pod tắt → ném `ArenaAiNotReadyException` (fallback pool). ⇒ mở
  khoá AI cho Đấu 1v1 / nhiệm vụ Quiz / contest auto.
- **`AiServiceClient`** (4 method): GenerateExam→`/generate/questions`, SolveProblem→`/solve`,
  ExplainWrongAnswer→`/explain-mistake` (tra câu hỏi theo `question_id`), AnalyzeClass→`/analyze-learning`.
- **`RagServiceClient`** (7 method): IndexDocumentUrl→`/documents/ingest`, Delete→`DELETE /documents/{id}`,
  Search→`/documents/search`, Ask/TeacherAssistant→`/documents/ask`, QuizFromUrl→`/documents/quiz`,
  LessonPlan→`/documents/ask` (best-effort, tab GV đã bỏ).

## AI service — hạ tầng RAG tài liệu (mới)
- **`AI_service/documents.py`**: tải file (presigned URL) → trích text (**pypdf** cho PDF, python-docx
  cho DOCX, txt) → **chunk theo mục** (nhận diện "Chương/Định lý/1.1…", không cắt cơ học) → embed
  bge-m3 → lưu `documents`/`document_chunks`. Kèm `search`, `ask` (RAG + trích nguồn), `quiz_from_url`.
- `db.py`: thêm `delete_document`, `delete_document_chunks`.
- `main.py`: 5 endpoint `POST /documents/{ingest,search,ask,quiz}` + `DELETE /documents/{id}`.
- `explain_mistake` nhận thêm `question_id`/`correct_answer` (tra câu hỏi trong ngân hàng để đủ ngữ cảnh).

## Đã kiểm chứng THẬT (pod RunPod A6000 + AI service :8080 + Postgres local)
- **Embedding backfill: 14513/14513** (DB local persist qua việc xoá/đổi pod).
- `POST /generate/questions` (MCQ) → 2 câu hợp lệ, đáp án đúng, distractor sạch. ✔
- `POST /solve` "đạo hàm x²+3x tại 2" → **7** (đúng), sympy verify match. ✔
- **RAG**: ingest 1 tài liệu (2 chunk) → `search` "mệnh đề là gì" trả đúng chunk (score 0.665) →
  `ask` trả lời có trích `[1]` + source. ✔
- BE build (Infrastructure + API) **succeeded** với cả 3 client mới.

## Còn lại (FE — sẽ làm tiếp)
- **Ra đề free-text**: bỏ ép chọn môn/chương ở "Tạo Quiz AI" & ra đề GV — cho nhập text tự do,
  ra bất kỳ dạng nào (map sang `/generate/questions` với `extra_instructions`, subject optional).
- **Gộp tab Solve + Hints** (AI Tutor học sinh, màn 7+8) làm 1.
- Kiểm tra luồng admin upload (màn 9) → `DocumentController` gọi `IndexDocumentUrl` → nay index thật.
