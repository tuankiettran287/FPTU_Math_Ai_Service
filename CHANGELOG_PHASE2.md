# Phase 2 — Serving vLLM + tách sinh 2 giai đoạn

> Ngày: 2026-07-10. Mục tiêu: chuyển inference từ transformers thuần (tuần tự, dễ
> nghẽn) sang **vLLM** trên GPU pod, và tách sinh 2 giai đoạn để JSON ổn định.

## Đã thay đổi

### `AI_service/llm.py` — viết lại thành `LLMClient`
- **Bỏ `self._generate_lock`** (nút thắt: mỗi lúc chỉ 1 request). Backend vLLM lo
  **continuous batching** phía server → nhiều tính năng chạy song song được.
- **Backend vLLM (mặc định)**: gọi `POST {VLLM_BASE_URL}/chat/completions`
  (OpenAI-compatible) qua httpx. Lỗi mạng → `RuntimeError` (main.py trả 500 gọn).
- **Tách sinh 2 giai đoạn** trong `generate_json`:
  1. *Suy luận tự do*: system = prompt.txt + hướng dẫn suy luận trong `<think>`, rồi
     kết luận bằng văn xuôi (ngân sách token `think`, KHÔNG ép JSON).
  2. *Ép JSON*: bỏ `<think>`, đưa phần kết luận cho một system "bộ định dạng",
     `temperature=0`, dùng **`guided_json`** (guided_decoding của vLLM) khi có schema —
     không cần retry JSON thủ công.
- **Timeout + max_new_tokens riêng theo task** (`config.TASK_LIMITS` → `think`/`json`/
  `timeout`), không dùng một con số chung.
- Thêm `chat()` (hội thoại tự do, cho RAG ở Phase 3) và `generate_text()`.
- **transformers vẫn giữ làm fallback** (`LLM_BACKEND=transformers`) cho dev, lazy-load.
- Tương thích ngược: `client.generate_json(task, payload, max_new_tokens, temperature,
  top_p)` như cũ → `main.py` không phải sửa.

### `AI_service/config.py`
- Nạp `.env` (python-dotenv, optional).
- Đã có sẵn từ Phase 1: `LLM_BACKEND`, `VLLM_BASE_URL/MODEL/API_KEY/TIMEOUT`, `TASK_LIMITS`.

### Mới
- **`.env.example`** — toàn bộ biến môi trường (DB, vLLM, embedding).
- **`POD_SETUP.md`** — hướng dẫn dựng trên RunPod: vLLM serve DeepSeek-R1-Distill-Qwen-7B
  (bf16 / FP8 / AWQ theo VRAM) cổng 8000, và bge-m3 `--task embed` cổng 8001; cách trỏ
  service tới pod + backfill embedding.
- `requirements.txt`: thêm `python-dotenv`.

## Đã kiểm chứng
- **Logic 2 giai đoạn** (offline, stub `_complete`): stage 1 dùng ngân sách `think`
  (3500) không guided; stage 2 dùng ngân sách `json` (3000), `temperature=0`,
  `guided_json` bật; kết quả parse đúng JSON. ✔
- **Service boot thật** (`uvicorn AI_service.main:app`): `/health` ok, `/` báo đúng
  Postgres, `/api/v1/problems` trả **14.511** từ DB. ✔
- Chưa gọi vLLM thật vì cần pod — sẽ chạy khi `VLLM_BASE_URL` trỏ tới pod sống.

## Việc còn lại
- Dựng pod theo `POD_SETUP.md`, set `.env`, rồi:
  - `curl $VLLM_BASE_URL/models` để chắc LLM sống;
  - `python -m tools.backfill_embeddings` để điền embedding.
- **Phase 3**: 9 tính năng (ra đề MCQ/tự luận, chấm không rubric 2 lớp + self-consistency,
  giải thích lỗi + RAG, tra cứu khái niệm + RAG, phân tích học tập, giải full vs gợi ý,
  ra đề arena theo ELO, ra đề theo lỗi sai, lộ trình học) — mỗi feature 1 endpoint,
  dùng `generate_json(..., json_schema=...)` + verify bằng sympy/code nơi tính được.
- **Phase 4**: migrate BE (RagServiceClient, ArenaQuestionClient…) sang gọi service mới, bỏ token.
