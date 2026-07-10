# Tài Liệu Phạm Vi FPTU_MATHAI

## 1. Mục Tiêu

FPTU_MATHAI là service AI hỗ trợ hệ thống học toán của sinh viên FPT University. Phiên bản hiện tại tập trung vào 3 bài toán cần AI:

1. Đọc hiểu tài liệu và tự động sinh đề thi trắc nghiệm.
2. Giải bài toán toán học hoặc bài toán logic.
3. Chấm bài làm sinh viên và đưa phản hồi sửa lỗi.

Service được triển khai bằng FastAPI, dùng model `DeepSeek-R1-Distill-Qwen-7B` local, không fine-tune.

## 2. Phạm Vi AI

### 2.1 Sinh Đề Từ Tài Liệu

Giáo viên hoặc backend gửi nội dung tài liệu đã extract/OCR vào API. AI đọc nội dung, xác định các ý chính và sinh bộ câu hỏi trắc nghiệm.

Tiêu chí đánh giá:

- Hiểu đúng nội dung tài liệu.
- Bao phủ các khái niệm chính.
- Sinh câu hỏi rõ ràng, không nhập nhằng.
- Có đáp án đúng và phương án nhiễu hợp lý.
- Có giải thích vì sao đáp án đúng.

Endpoint:

```txt
POST /api/v1/exams/generate-from-document
```

### 2.2 Giải Bài Toán Và Bài Toán Logic

Sinh viên hoặc backend gửi đề bài dạng text. AI phân tích đề, lập luận từng bước và trả lời theo schema ngân hàng bài toán.

Tiêu chí đánh giá:

- Hiểu đúng đề bài.
- Lập luận đầy đủ.
- Đáp án cuối chính xác.
- Có khả năng xử lý nhiều mức độ khó.
- Có gợi ý, lỗi thường gặp và khái niệm liên quan.

Endpoint:

```txt
POST /api/v1/problems/solve
```

### 2.3 Chấm Bài Làm Sinh Viên

Backend gửi bài làm sinh viên kèm `question_id` đã có trong DB, hoặc gửi trực tiếp đề bài và lời giải chuẩn. AI so sánh, chấm điểm và đưa phản hồi.

Tiêu chí đánh giá:

- So sánh đúng bài làm với đáp án chuẩn.
- Kết luận đúng/sai.
- Phát hiện lỗi sai cụ thể.
- Giải thích nguyên nhân.
- Đề xuất hướng sửa.

Endpoint:

```txt
POST /api/v1/submissions/grade
```

## 3. Phạm Vi Không Làm Trong AI Service

Các phần sau thuộc backend/frontend chính, không nằm trong service AI hiện tại:

- Quản lý tài khoản, role, lớp học.
- Giao bài, nộp bài, deadline, retry.
- Dashboard sinh viên/giáo viên/admin.
- Notification, chat real-time, video call.
- Import Excel, quản lý file học liệu.
- Workflow giáo viên duyệt câu hỏi.
- Ghi nhận điểm cuối cùng và quyền override.

AI service chỉ trả kết quả đề xuất để hệ thống chính lưu workflow nghiệp vụ.

## 4. Dữ Liệu Và Database

Service dùng SQLite mặc định. Database tự tạo khi app startup.

Bảng chính:

- `problems`: lưu câu hỏi/bài toán được sinh hoặc được giải. Trường `document` là JSON theo schema ngân hàng bài toán.
- `exam_generations`: lưu lịch sử sinh đề từ tài liệu, coverage report và danh sách câu hỏi đã tạo.
- `evaluations`: lưu kết quả chấm bài sinh viên.

Schema bài toán gồm:

```txt
id, subject, course, chapter, topic, subtopic, difficulty,
question_type, question, solution, concepts_used, prerequisites,
common_mistakes, hints, evaluation, metadata
```

`Data_Bank.json` có thể dùng làm dữ liệu tham chiếu/schema seed, nhưng service mới không phụ thuộc vào script train.

## 5. Model Và Prompt

Model:

```txt
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Cấu hình local:

```powershell
$env:DEEPSEEK_MODEL_PATH="D:\models\DeepSeek-R1-Distill-Qwen-7B"
$env:MODEL_LOCAL_FILES_ONLY="true"
```

Prompt:

```txt
prompt.txt
```

Mọi task đều dùng cùng file prompt và phân biệt bằng `TASK_ID`:

- `exam_generation`
- `solve_problem`
- `grade_submission`

## 6. Luồng Tích Hợp

### Sinh đề

1. Backend extract text từ tài liệu.
2. Backend gọi `POST /api/v1/exams/generate-from-document`.
3. AI trả danh sách câu hỏi và coverage report.
4. Service lưu câu hỏi vào `problems`.
5. Giáo viên review trước khi dùng chính thức.

### Giải bài

1. Backend gửi đề bài vào `POST /api/v1/problems/solve`.
2. AI trả lời giải theo từng bước.
3. Service lưu bài toán và lời giải vào `problems`.
4. Backend hiển thị cho sinh viên.

### Chấm bài

1. Backend gửi `question_id` và bài làm sinh viên.
2. AI lấy đề/lời giải từ DB, so sánh và chấm.
3. Service lưu kết quả vào `evaluations`.
4. Backend cho giáo viên xem và có thể override.

## 7. Rủi Ro

- LLM vẫn có thể sinh câu hỏi hoặc lời giải sai; cần review với nội dung quan trọng.
- Nếu text tài liệu extract/OCR sai, câu hỏi sinh ra cũng có thể sai.
- Model 7B cần RAM/VRAM đủ lớn; lần gọi đầu tiên sẽ chậm do load model.
- AI chấm bài chỉ nên là điểm đề xuất, không phải điểm cuối cùng nếu chưa có giáo viên xác nhận.

## 8. Kết Luận Scope

Repo hiện tại đã được rút gọn để phục vụ đúng 3 chức năng AI cốt lõi. Không còn pipeline fine-tune, LoRA, Gradio demo, vector search hoặc các file dữ liệu sinh ra không cần thiết.
