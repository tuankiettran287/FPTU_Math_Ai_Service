# AI Features

Tài liệu này mô tả các chức năng AI đang được code trong `AI_service`. Phiên bản hiện tại là FastAPI service, dùng `DeepSeek-R1-Distill-Qwen-7B` local, không fine-tune.

## Kiến Trúc

```txt
Client / Backend chính
        |
        v
FastAPI: AI_service.main
        |
        +-- llm.py      -> load model local và sinh JSON
        +-- prompt.txt  -> prompt duy nhất cho 3 task
        +-- schemas.py  -> chuẩn hóa schema bài toán
        +-- db.py       -> SQLite lưu bài toán và kết quả chấm
```

## 1. Đọc Hiểu Tài Liệu Và Sinh Đề Thi

Endpoint:

```txt
POST /api/v1/exams/generate-from-document
```

Input chính:

- `document_text`: nội dung tài liệu đã extract/OCR.
- `course`, `chapter`, `topic`, `subtopic`.
- `difficulty`: `easy`, `medium`, `hard`.
- `question_count`: số câu cần sinh.
- `choices_per_question`: số phương án trắc nghiệm.
- `coverage_requirements`: các ý bắt buộc cần bao phủ nếu giáo viên cung cấp.

AI cần đảm bảo:

- Hiểu đúng nội dung tài liệu.
- Sinh câu hỏi trắc nghiệm có một đáp án đúng.
- Sinh đủ đáp án và phương án nhiễu hợp lý.
- Có `coverage_report` để mô tả phần kiến thức đã bao phủ.
- Câu hỏi và lời giải được lưu vào bảng `problems` theo schema ngân hàng câu hỏi.

Output chính:

- `exam_id`
- `saved_count`
- `coverage_report`
- `problems[]`

## 2. Giải Bài Toán Và Bài Toán Logic

Endpoint:

```txt
POST /api/v1/problems/solve
```

Input chính:

- `problem_text`: đề bài.
- `problem_kind`: `math` hoặc `logic`.
- `course`, `chapter`, `topic`.
- `difficulty`.
- `latex`: công thức LaTeX nếu frontend tách riêng.

AI cần đảm bảo:

- Hiểu đề bài và nhận ra dữ kiện quan trọng.
- Lập luận từng bước.
- Đưa ra đáp án cuối trong `solution.final_answer`.
- Ghi các khái niệm dùng trong `concepts_used`.
- Ghi lỗi thường gặp và gợi ý học tập.
- Lưu bài toán và lời giải vào database.

Output là một object theo schema `QuestionBankItem`.

## 3. Chấm Bài Làm Sinh Viên

Endpoint:

```txt
POST /api/v1/submissions/grade
```

Input chính:

- `question_id`: lấy đề và lời giải chuẩn từ DB.
- Hoặc `question_text` + `standard_solution`: chấm trực tiếp khi đề chưa có trong DB.
- `student_answer`: bài làm sinh viên.
- `rubric`: thang điểm hoặc tiêu chí chấm.
- `max_score`.
- `student_id`, `class_id`, `submission_id` nếu backend cần tracking.

AI cần đảm bảo:

- So sánh bài làm với đáp án/lời giải chuẩn.
- Kết luận `DUNG` hoặc `SAI`.
- Phát hiện lỗi sai cụ thể.
- Nhận xét từng bước nếu bài làm có nhiều bước.
- Giải thích nguyên nhân sai.
- Đề xuất hướng sửa.

Output chính:

- `evaluation_id`
- `verdict`
- `is_correct`
- `score`
- `max_score`
- `feedback`
- `mistakes`
- `step_feedback`
- `suggested_fix`
- `explanation`
- `confidence`

Kết quả chấm được lưu vào bảng `evaluations`.

## Schema Lưu Bài Toán

Mỗi bài toán/câu hỏi được lưu trong bảng `problems.document` theo schema:

```json
{
  "id": "MATHAI_...",
  "subject": "MAE",
  "course": "Mathematics for Engineering",
  "chapter": "General",
  "topic": "General",
  "subtopic": "",
  "difficulty": {
    "level": "medium",
    "score": 5,
    "estimated_time_minutes": 10,
    "cognitive_level": "apply"
  },
  "question_type": "problem_solving",
  "question": {
    "text": "Đề bài",
    "latex": null,
    "image": null,
    "options": []
  },
  "solution": {
    "final_answer": "Đáp án cuối",
    "steps": [],
    "alternative_solutions": []
  },
  "concepts_used": [],
  "prerequisites": [],
  "common_mistakes": [],
  "hints": [],
  "evaluation": {
    "answer_verifiable": true,
    "step_verifiable": true
  },
  "metadata": {
    "source": "DeepSeek R1 Distill Qwen 7B",
    "language": "vi",
    "created_by": "AI",
    "verified": false
  }
}
```

## Nguyên Tắc Tích Hợp

- Backend chính gọi FastAPI qua HTTP, không gọi trực tiếp model.
- Tài liệu dạng file nên được extract/OCR trước khi gửi vào `document_text`.
- AI chỉ là nguồn đề xuất; giáo viên vẫn nên review câu hỏi, đáp án và điểm.
- Không có workflow training trong repo hiện tại.
