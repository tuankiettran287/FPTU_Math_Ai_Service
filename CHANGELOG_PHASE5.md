# Phase 5 — Ra đề tự do + gộp Giải/Gợi ý + OCR ảnh/file chatbot + gỡ AI cũ + all-on-pod

> Ngày: 2026-07-11. Xử lý 3 flag còn lại của Phase 4 + yêu cầu mới: bỏ AI service cũ,
> thêm OCR/tải file-ảnh cho chatbot, chuyển topology "toàn bộ AI trên RunPod, DB ở VPS".

## 1) Ra đề TỰ DO (bỏ ép chọn môn/chương)
- **FE** `practice/page.tsx` (`GenerateQuizModal`): ô prompt tự do là chính (nhập bất kỳ
  dạng đề nào bằng lời + gợi ý mẫu), môn/chương gập lại thành **tuỳ chọn**.
- **BE** `SelfStudyExamController.Generate`: `SubjectId` optional, thêm `Prompt` →
  map vào `GenerateExamRequest.AdditionalInstructions` (`/generate/questions` extra_instructions).
- **DB**: `SelfStudyExamAttempt.SubjectId` → **nullable** (migration `MakeExamAttemptSubjectNullable`).
  Các analytics theo môn (sổ tay lỗi/SRS/GV theo dõi) loại đề tự do (SubjectId null).
- **AI** `features.generate_questions`: xử lý subject rỗng, **ưu tiên `extra_instructions`** là đặc tả chính.

## 2) Gộp tab Giải bài + Gợi ý (AI Tutor SV, màn 7+8)
- **FE** `chat/page.tsx`: 3 mode → **2** ("Find documents" + "Giải bài & Gợi ý"), thêm
  sub-toggle **Lời giải đầy đủ / Chỉ gợi ý**. Bỏ preview giả, gọi thật `/api/ai/solve`.
- **BE** `SolveProblemRequest.Mode` (full|hint) → `AiServiceClient` truyền `mode`; hint đọc
  field `hint`+`total_steps` (không lộ đáp án). `AIController` nhận `mode`.

## 3) Màn 9 — admin tải tài liệu → RAG index
- Xác nhận luồng: Upload → `POST /documents/{id}/ai-index` → `IndexDocumentUrlAsync` →
  AI `/documents/ingest` (trả `status:"indexed"`, khớp check BE). **Nới định dạng**:
  `DocumentController` cho index cả **PDF/DOC/DOCX/TXT/MD** (trước chỉ PDF) — khớp
  `documents.extract_text`.

## 4) Gỡ AI service cũ
- **appsettings**: xoá `AiService`(:8000)/`RagService`(:8001); giữ **duy nhất `MathAiService`**
  (+ dời `IndexUrlExpiryMinutes` vào đây). `DependencyInjection`: bỏ fallback key cũ,
  gom `MathAiConfig()` dùng chung cho cả 3 client.

## 5) OCR + tải ảnh/file ở chatbot (DeepSeek không nhìn ảnh)
- **AI** `ocr.py`: OCR ảnh→text/LaTeX, backend cấu hình `OCR_BACKEND`:
  `rapidocr` (onnxruntime, CPU, mặc định) · `pix2tex` (công thức→LaTeX) ·
  `vlm` (gửi ảnh tới VLM OpenAI-compatible `OCR_VLM_BASE_URL`) · `none`. Endpoint `POST /ocr`.
- `features.solve`: nhận `image_base64` (→OCR) và `file_url`/`file_base64`+`file_extension`
  (→`extract_text`); `problem_text` optional.
- **BE**: `SolveProblemRequest`/`SolveProblemDto` + `AiServiceClient` truyền image/file;
  `AIController` yêu cầu có đề HOẶC ảnh/file; **KHÔNG log base64** vào `AiInteractionLog`.
- **FE** `chat/page.tsx`: đọc tệp đính kèm → base64; ảnh→`imageBase64`, file→`fileBase64`+ext;
  giới hạn 8MB; nhận .txt.

## 6) Topology mới: TOÀN BỘ AI trên RunPod, DB ở VPS
- `pod_start.sh`: thêm cài deps FastAPI+DB+đọc file+OCR, và **chạy FastAPI :8080 trong tmux 'api'**
  (cùng pod → gọi vLLM/embed qua `localhost:8000/8001`, override trong lệnh). Dọn thêm cổng/session 8080/'api'.
- `.env`: pod hiện tại **s84oqayiufbf7f**; ghi rõ 2 chế độ (LOCAL vs ALL-ON-POD) cho DATABASE_URL/VLLM/EMBEDDING; thêm `OCR_BACKEND`.
- `config.py`: thêm cấu hình OCR.

## Kiểm chứng
- **BE build succeeded** (0 error) sau tất cả thay đổi.
- **FE tsc** (chat/page, practice/page, actions) — sạch.
- **Pod s84oqayiufbf7f**: vLLM DeepSeek-R1 (:8000) + bge-m3 (:8001) **đang serve** (đã tải lại,
  `/v1/models` OK, embed trả vector 1024). AI **reconnected ở tầng model**.
- Python `py_compile` các file AI mới/sửa: OK.

## ⚠️ Còn phải làm tay (không code được từ đây)
- **FastAPI-on-pod cần DB reach được**: hiện DB local (laptop 5433) pod không thấy. Muốn "all-on-pod"
  phải dựng Postgres trên **VPS** + `pg_dump/restore` DB `fptu_mathai_final` (14.5k câu + embedding) sang VPS,
  mở cho pod (firewall theo IP pod hoặc SSH reverse tunnel), rồi set `DATABASE_URL` (pod .env) + `MathAiService:BaseUrl`
  (BE) = `https://s84oqayiufbf7f-8080.proxy.runpod.net`. Trong lúc chưa có VPS DB: **chạy FastAPI local** (uvicorn,
  DB local) + models trên pod — hoạt động ngay, `MathAiService:BaseUrl=http://localhost:8080`.
- **Đồng bộ code lên pod** (git/VS Code Remote vào `/workspace/FPTU_Math_Ai_Service-main`) rồi `bash deploy/pod_start.sh`
  để FastAPI :8080 chạy trên pod. OCR `rapidocr` cài trong pod_start.
- OCR mặc định `rapidocr` (text in ấn). Ảnh viết tay/công thức phức tạp → cân nhắc `pix2tex` hoặc serve thêm VLM (`OCR_BACKEND=vlm`).
