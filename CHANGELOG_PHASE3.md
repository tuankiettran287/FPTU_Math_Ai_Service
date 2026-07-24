# Phase 3 — 9 tính năng AI + RAG + verify (sympy)

> Ngày: 2026-07-10. Mỗi tính năng = 1 endpoint, dùng `client.generate_json` (2 giai
> đoạn + guided_json), verify bằng sympy/code nơi tính được, lưu DB riêng và trả đủ
> JSON cho BE. **Test offline** (stub LLM) toàn bộ 9 tính năng — chờ pod để chạy model thật.

## Module mới
- **`AI_service/verify.py`** — sympy: `answers_equivalent` (so biểu thức tương đương, không
  so chuỗi), `check_distractors` (loại phương án nhiễu trùng đáp án đúng, bỏ qua chính
  đáp án), `verify_final_answer`, `is_math_like` (cổng để không chấm nhầm câu văn xuôi).
- **`AI_service/rag.py`** — `retrieve` (embed truy vấn → `db.search_chunks`), `rewrite_query`
  (LLM viết lại câu hỏi ngắn/mơ hồ), `format_context`, `sources` (trả cả file tài liệu).
  Kho tài liệu rỗng → trả [] gọn (feature vẫn chạy).
- **`AI_service/features.py`** — 9 hàm nghiệp vụ.
- `db.py` thêm `upsert_document/upsert_chunk/search_chunks` (RAG) + `_filter_clause(prefix)`.
- `schemas.py` thêm 9 request model. `main.py` thêm 9 endpoint.

## 9 endpoint
| # | Endpoint | Tính năng | Điểm nhấn |
|---|----------|-----------|-----------|
| 1 | `POST /api/v1/generate/questions` | Ra đề MCQ/tự luận | few-shot theo dạng; **code-check distractor ≠ đáp án**; sympy verify đáp án khi tính được; lưu vào ngân hàng |
| 2 | `POST /api/v1/grade` | Chấm không rubric | **Lớp 1 sympy** (so đáp án cuối) + **Lớp 2 LLM chấm 2 lần (self-consistency)** → lệch thì `needs_teacher_review`, confidence thấp; tự liệt kê bước bắt buộc, không ép trình bày |
| 3 | `POST /api/v1/explain-mistake` | Giải thích lỗi + chỉ lý thuyết | **RAG** lấy đoạn tài liệu, trích [1],[2]…; trả `sources` |
| 4 | `POST /api/v1/concept` | Tra cứu khái niệm | **query rewriting** trước khi embed; trả cả **file tài liệu** (`sources`) |
| 5 | `POST /api/v1/analyze-learning` | Phân tích học tập | mastery theo chủ đề, mẫu lỗi, xu hướng tuần/tháng, đề xuất |
| 6 | `POST /api/v1/solve` | Giải full / gợi ý | `mode=hint`: sinh full **ẩn** → lệnh 2 chỉ trích gợi ý bước hiện tại, **không lộ đáp án** |
| 7 | `POST /api/v1/arena/questions` | Ra đề arena theo ELO | ELO→độ khó (easy/medium/hard), MCQ theo bậc track |
| 8 | `POST /api/v1/generate/from-mistakes` | Ra đề theo lỗi sai | gom chủ đề từ sổ tay lỗi → ra câu rèn đúng chỗ yếu |
| 9 | `POST /api/v1/study-path` | Lộ trình học | milestone có thứ tự (chủ đề/hoạt động/thời lượng) từ mục tiêu/trình độ/điểm yếu |

## Đã kiểm chứng (offline, stub LLM — không cần pod)
- Cả 9 hàm chạy end-to-end, ghi `ai_interactions`/`ai_evaluations`, dọn sạch câu test.
- **Grade**: lớp 1 sympy `4 ≡ 4` → match; 2 lần chấm đồng thuận → confidence 0.95, không cần review. Khi 2 lần lệch → `needs_teacher_review=True`, confidence 0.4.
- **Distractor**: đã sửa lỗi cờ nhầm chính đáp án đúng → `ok=True` khi các nhiễu khác đáp án.
- 12 POST route (9 mới + 3 cũ) đăng ký OK.

## Còn lại / lưu ý
- **Cần pod** (vLLM + embedding) để chạy model thật; guided_json chỉ hiệu lực trên vLLM.
- **RAG (feature 3,4)** cần **nạp tài liệu** vào `documents`/`document_chunks` (chunk theo
  định lý/mục — TODO ingest). Chưa có tài liệu thì trả lời không trích nguồn.
- **question_type** trong ngân hàng nhiều dạng tự do → khi ra đề/arena nên chuẩn hoá.
- **Phase 4**: migrate BE (RagServiceClient, ArenaQuestionClient…) gọi 9 endpoint này, bỏ token.

## Việc phụ đã làm kèm
- **Bỏ tab "Lesson Planning"** khỏi AI Tutor giáo viên (FE `teacher/chat/page.tsx`) — còn
  Document Search + AI Chat. BE lesson-plan để dormant.
