# FPTU_MATHAI

FastAPI service cho 3 chức năng AI cốt lõi của hệ thống học toán FPTU:

1. Đọc hiểu tài liệu và tự động sinh đề thi trắc nghiệm.
2. Giải bài toán toán học hoặc bài toán logic theo từng bước.
3. Chấm bài làm sinh viên, phát hiện lỗi sai và đề xuất hướng sửa.

Service dùng model `DeepSeek-R1-Distill-Qwen-7B` local qua `transformers`, không fine-tune và không dùng LoRA. Prompt nghiệp vụ nằm tập trung trong một file duy nhất: `prompt.txt`.

## Cấu Trúc

```txt
FPTU_MathAI/
  AI_service/
    main.py       # FastAPI app và routes
    llm.py        # lazy-load DeepSeek R1 Distill Qwen 7B local
    db.py         # SQLite database
    schemas.py    # request/response model và schema Data_Bank
    config.py     # cấu hình môi trường
    utils.py      # helper JSON, ID, thời gian
  main.py         # chạy uvicorn local
  prompt.txt      # prompt duy nhất cho 3 task AI
  requirements.txt
  README.md
  AI_FEATURES.md
  Doc.Md
```

## Cài Đặt

```powershell
python -m pip install -r requirements.txt
```

Nếu model đã tải sẵn ở máy local, trỏ biến môi trường tới thư mục model:

```powershell
$env:DEEPSEEK_MODEL_PATH="D:\models\DeepSeek-R1-Distill-Qwen-7B"
```

Nếu không set biến này, service dùng fallback `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` và phụ thuộc vào cache Hugging Face có sẵn.

Mặc định service chỉ đọc model từ local/cache:

```powershell
$env:MODEL_LOCAL_FILES_ONLY="true"
```

Nếu muốn cho phép `transformers` tải model từ Hugging Face, đổi biến này thành `false`.

Database mặc định:

```powershell
$env:DATABASE_URL="sqlite:///F:\FPTU_MathAI\mathai.db"
```

## Chạy API

```powershell
python main.py
```

Hoặc:

```powershell
uvicorn AI_service.main:app --host 127.0.0.1 --port 8000
```

Swagger UI:

```txt
http://127.0.0.1:8000/docs
```

Model được lazy-load: `/health` không load model, endpoint sinh/chấm đầu tiên mới load model.

## Endpoint Chính

### 1. Sinh đề từ tài liệu

`POST /api/v1/exams/generate-from-document`

```json
{
  "document_text": "Nội dung tài liệu hoặc nội dung đã OCR...",
  "subject": "MAE",
  "course": "Mathematics for Engineering",
  "chapter": "Derivatives",
  "topic": "Basic derivatives",
  "difficulty": "medium",
  "question_count": 5,
  "choices_per_question": 4,
  "language": "vi"
}
```

Kết quả trả về `exam_id`, `coverage_report` và danh sách câu hỏi đã lưu vào DB.

### 2. Giải bài toán

`POST /api/v1/problems/solve`

```json
{
  "problem_text": "Giải phương trình 2x + 3 = 7.",
  "subject": "MAE",
  "course": "Mathematics for Engineering",
  "chapter": "Algebra",
  "topic": "Linear equations",
  "difficulty": "easy",
  "problem_kind": "math",
  "language": "vi"
}
```

Kết quả là một bài toán theo schema ngân hàng câu hỏi, gồm đề bài, lời giải từng bước và đáp án cuối.

### 3. Chấm bài làm sinh viên

`POST /api/v1/submissions/grade`

```json
{
  "question_id": "MATHAI_20260710093000_ab12cd34",
  "student_answer": "2x + 3 = 7 nên x = 5",
  "max_score": 10,
  "rubric": "Đáp án 4 điểm, biến đổi 4 điểm, trình bày 2 điểm",
  "student_id": "SE180001",
  "submission_id": "SUB_001",
  "language": "vi"
}
```

Nếu chưa có `question_id`, truyền trực tiếp:

```json
{
  "question_text": "Giải phương trình 2x + 3 = 7.",
  "standard_solution": "x = 2",
  "student_answer": "x = 5",
  "max_score": 10
}
```

Kết quả gồm `verdict` (`DUNG` hoặc `SAI`), điểm, lỗi sai, nhận xét và hướng sửa.

## Endpoint Tra Cứu

- `GET /health`
- `GET /api/v1/schema/problem`
- `GET /api/v1/problems`
- `GET /api/v1/problems/{problem_id}`
- `GET /api/v1/evaluations/{evaluation_id}`

## Database

SQLite tự tạo các bảng:

- `problems`: lưu bài toán/câu hỏi được sinh hoặc được giải, trường `document` chứa JSON theo schema `Data_Bank`.
- `exam_generations`: lưu request sinh đề, coverage report và danh sách `problem_id`.
- `evaluations`: lưu kết quả chấm bài sinh viên.

Schema bài toán giữ các field chính:

```txt
id, subject, course, chapter, topic, subtopic, difficulty,
question_type, question, solution, concepts_used, prerequisites,
common_mistakes, hints, evaluation, metadata
```

## Ghi Chú

- Service không train, không fine-tune, không dùng adapter.
- Prompt nghiệp vụ chỉ nằm trong `prompt.txt`.
- Các câu hỏi AI sinh có `metadata.verified = false`; giáo viên nên review trước khi đưa vào đề chính thức.
- API nhận text tài liệu. Nếu tài liệu là PDF, ảnh hoặc DOCX, backend/UI nên OCR hoặc extract text trước rồi gửi vào endpoint.
