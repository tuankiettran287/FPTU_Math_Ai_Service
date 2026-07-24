# Phase 1 — Nền tảng dữ liệu (DB Postgres/pgvector + Import dataset)

> Ngày: 2026-07-10. Mục tiêu Phase 1: dựng lại nền tảng DB cho AI service và nạp
> toàn bộ ngân hàng câu hỏi. **Chưa** đụng tới serving model (Phase 2) hay các
> tính năng AI (Phase 3).

## Ràng buộc môi trường (quan trọng)
- **Máy dev không có GPU → KHÔNG tải/chạy bất kỳ model nào (LLM lẫn embedding) trên máy này.**
  Mọi model sẽ chạy trên **GPU pod (RunPod)**. Vì vậy Phase 1 import **không sinh
  embedding tại chỗ** — cột `embedding` để `NULL`, sẽ backfill sau ở nơi có GPU.
- DB: container Docker `fptu_mathai_db` = `pgvector/pgvector:pg17` tại `localhost:5433`,
  database **`fptu_mathai_final`** (postgres/postgres).

## Đã thay đổi

### 1. `AI_service/config.py`
- `DATABASE_URL` mặc định → Postgres `fptu_mathai_final` (thay SQLite).
- LLM backend: `LLM_BACKEND=vllm` (mặc định) + `VLLM_BASE_URL/VLLM_MODEL/VLLM_API_KEY/VLLM_TIMEOUT`
  để gọi vLLM OpenAI-compatible trên pod (Phase 2). Vẫn giữ nhánh `transformers` để dev.
- Embedding: **bge-m3, `EMBEDDING_DIM=1024`**. Backend `remote` (mặc định, gọi endpoint
  `/embeddings` trên pod) hoặc `local`. Máy dev không load model.
- `TASK_LIMITS` + `task_limits(task)`: ngân sách token (`think`/`json`) và `timeout`
  **riêng theo từng loại task** (không dùng 1 con số chung) — chuẩn bị cho tách sinh
  2 giai đoạn `<think>` tự do → ép JSON kết luận ở Phase 2.

### 2. `AI_service/db.py` — chuyển hẳn sang Postgres + pgvector (psycopg3)
- Đổi bảng `problems` (SQLite cũ) → **`math_question_bank`** đúng schema yêu cầu:
  `id, subject, assest_format, course, chapter, topic, subtopic, difficulty(JSONB),
  question_type, question/solution/concepts_used/prerequisites/common_mistakes/hints/
  evaluation/metadata (JSONB), embedding VECTOR(1024) (nullable), document(JSONB=bản ghi gốc),
  created_at, updated_at` + index ivfflat cosine.
- Thêm bảng **`ai_interactions`**, **`ai_evaluations`** (đúng schema yêu cầu) và
  **`documents` / `document_chunks`** (scaffold RAG cho Phase 3, ivfflat cosine).
- API: `init_db, upsert_question(s_bulk), get_question, list_questions, count_questions,
  search_questions(vector cosine + lọc), draw_questions(lọc/random), save_interaction,
  save_ai_evaluation` + **shim tương thích** (`save_problem/get_problem/list_problems/
  save_evaluation/get_evaluation/save_exam_generation`) để `main.py` cũ vẫn chạy.

### 3. `AI_service/embeddings.py` (mới)
- bge-m3, 2 backend: `remote` (HTTP OpenAI-compatible, mặc định — **không load trên laptop**)
  và `local` (SentenceTransformer). `build_question_text()` ghép các trường có nghĩa của
  câu hỏi thành "document" để embed.

### 4. `AI_service/schemas.py`
- Thêm field `assest_format` vào `QuestionBankItem` (dataset có trường này).

### 5. `tools/import_datasets.py` (mới)
- Parser **salvage**: quét từng object top-level (cân bằng ngoặc), tự sửa lỗi JSON hay
  gặp trong các file MAD/MAE/MAS (key `assest_format` lặp, thiếu dấu phẩy) — bỏ qua bản
  ghi không cứu được thay vì hỏng cả file.
- Dedupe theo `id` (Data_Bank xử lý trước vì sạch, rồi bổ sung folder).
- Mặc định import **embedding=NULL**; cờ `--embed` để sinh embedding khi có endpoint.

### 6. `tools/backfill_embeddings.py` (mới)
- Điền embedding cho các dòng đang NULL — chạy ở nơi có endpoint embedding (pod).

## Kết quả chạy thật (Postgres 5433)
- Đã tạo DB `fptu_mathai_final`, bật extension `vector`, tạo toàn bộ bảng.
- **Import 14.511 câu hỏi** (unique theo id), ~5 giây:
  - Nguồn: `Data_Bank (1).json` (10.132 bản ghi → 7.630 id unique) + folder `MAD/MAE/MAS`
    (JSON lỗi, salvage được phần lớn; 1 file MAD chỉ cứu 221/240).
  - Phân bố: **MAD 6.752 · MAE 4.403 · MAS 3.356**; 87 chương.
  - question_type: 6.918 `multiple_choice`, 4.454 `problem_solving`, 1.002 `proof`, còn lại
    đuôi dài nhiều dạng tự do (sẽ chuẩn hoá khi làm feature ra đề/arena ở Phase 3).
  - `embedding` = NULL toàn bộ (đúng chủ đích, backfill sau).

## Việc còn lại (phase sau)
- **Backfill embedding** khi pod embedding sẵn sàng: `python -m tools.backfill_embeddings`.
- **Phase 2** — serving vLLM: chuyển `llm.py` sang gọi vLLM (continuous batching,
  guided_decoding, quantize AWQ/FP8 tuỳ VRAM pod), timeout/max_new_tokens theo route,
  tách sinh 2 giai đoạn (`<think>` tự do → ép JSON kết luận).
- **Phase 3** — 9 tính năng AI + RAG tài liệu (chunk theo định lý/mục).
- **Phase 4** — migrate BE (RagServiceClient, ArenaQuestionClient, …) sang gọi service mới, bỏ token.

## Cách chạy lại import
```bash
# tạo/khởi tạo schema + import (embedding NULL)
python -m tools.import_datasets              # đầy đủ
python -m tools.import_datasets --dry-run    # chỉ đếm
python -m tools.import_datasets --limit 20   # thử nhanh
```
