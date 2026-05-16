# AI Features

Tài liệu này mô tả các chức năng AI đã được tách vào folder `AI_service`.

Entry point chính:

```powershell
python -m AI_service.main --help
```

Root `main.py` chỉ là wrapper để vẫn chạy được:

```powershell
python main.py --help
```

## Kiến Trúc AI Service

```txt
AI_service/
  main.py      # khai báo CLI command
  commands.py  # xử lý từng nghiệp vụ AI
  db.py        # PostgreSQL, pgvector, insert/search
  llm.py       # load DeepSeek + LoRA
  prompts.py   # prompt cho từng chức năng
  schemas.py   # chuẩn hóa dữ liệu giống Data_Bank.json
  utils.py     # helper JSON, OCR, vector
  config.py    # cấu hình mặc định
```

## Model Mặc Định

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- LoRA adapter: `outputs/deepseek-fptu-mathai-lora`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Database: PostgreSQL + `pgvector`

## Bảng Database

`AI_service` tự tạo các bảng sau:

| Bảng | Mục đích |
| --- | --- |
| `math_question_bank` | Lưu đề, lời giải, metadata, document JSON và vector embedding |
| `ai_interactions` | Lưu lịch sử tương tác AI của student/teacher/admin |
| `ai_evaluations` | Lưu kết quả AI chấm bài tự luận |
| `ai_class_analytics` | Lưu kết quả AI phân tích lớp |

## 1. AI Sinh Câu Hỏi Tự Động

Dùng cho giáo viên tạo bài tập hoặc bài kiểm tra theo chương, topic và độ khó.

```powershell
python -m AI_service.main generate `
  --chapter "Linear Algebra" `
  --topic "Vector Spaces" `
  --difficulty medium `
  --count 5
```

Kết quả được lưu vào `math_question_bank` theo schema giống `Data_Bank.json` và có vector embedding.

## 2. AI Giải Đề User Upload

Dùng cho học sinh upload ảnh hoặc text đề bài để AI giải từng bước.

Từ ảnh:

```powershell
python -m AI_service.main solve-upload `
  --image-path data/uploads/problem.png `
  --chapter "Calculus" `
  --topic "Derivative"
```

Từ text OCR:

```powershell
python -m AI_service.main solve-upload `
  --ocr-text "Find the derivative of f(x)=x^3+2x." `
  --chapter "Calculus" `
  --topic "Derivative"
```

Đề upload và lời giải đều được lưu vào database và vector hóa.

## 3. AI Chấm Bài Tự Luận

Dùng cho feature giáo viên hoặc hệ thống chấm bài tự luận.

```powershell
python -m AI_service.main evaluate-answer `
  --question-id MAE_MATH_0001 `
  --student-answer "x = 2" `
  --rubric "10 điểm: đáp án 4 điểm, biến đổi 4 điểm, trình bày 2 điểm" `
  --student-id SE180001 `
  --class-id MAE101_SP26 `
  --submission-id SUB_001
```

Output JSON có các field chính:

- `verdict`: `DUNG` hoặc `SAI`
- `is_correct`
- `score`
- `max_score`
- `feedback`
- `expected_answer`
- `mistakes`
- `step_feedback`
- `suggested_fix`
- `confidence`

Kết quả được lưu vào `ai_evaluations` và `ai_interactions`.

## 4. AI Giải Thích Câu Sai

Dùng sau khi sinh viên nộp bài và có câu sai.

```powershell
python -m AI_service.main explain-wrong `
  --question-id MAE_MATH_0001 `
  --student-answer "x = 5" `
  --student-id SE180001 `
  --class-id MAE101_SP26
```

Nếu chưa có `question-id`, truyền đề trực tiếp:

```powershell
python -m AI_service.main explain-wrong `
  --question "Solve 2x+3=7" `
  --student-answer "x=5"
```

## 5. AI Tự Kiểm Tra Năng Lực

Sinh bộ câu hỏi diagnostic:

```powershell
python -m AI_service.main self-assess `
  --student-id SE180001 `
  --course "Mathematics for Engineering" `
  --topics "Linear Algebra, Calculus" `
  --num-questions 10
```

Chấm diagnostic từ file câu trả lời:

```powershell
python -m AI_service.main self-assess `
  --student-id SE180001 `
  --answers-json data/student_answers.json
```

Output gồm level ước lượng, điểm mạnh, điểm yếu và lộ trình học đề xuất.

## 6. AI Phân Tích Lớp

Dùng cho giáo viên xem lớp yếu chương nào, topic nào, dạng bài nào.

```powershell
python -m AI_service.main analyze-class `
  --class-id MAE101_SP26 `
  --input-json data/class_records.json
```

File `class_records.json` nên chứa danh sách submission hoặc score theo format backend tự định nghĩa, ví dụ:

```json
[
  {
    "student_id": "SE180001",
    "question_id": "MAE_MATH_0001",
    "chapter": "Linear Algebra",
    "topic": "Vector Spaces",
    "score": 6.5,
    "mistakes": ["confused basis with spanning set"]
  }
]
```

Output có:

- `summary`
- `grade_distribution`
- `weak_chapters`
- `weak_topics`
- `common_mistakes`
- `students_need_support`
- `recommended_actions`
- `suggested_review_questions`

## 7. AI Chatbox Cho Giáo Viên

Chatbox dùng RAG từ `math_question_bank` và có thể nhận thêm analytics của lớp.

```powershell
python -m AI_service.main teacher-chat `
  --class-id MAE101_SP26 `
  --message "Lớp này yếu chương nào nhất và nên ôn dạng bài nào?"
```

Kèm file phân tích lớp:

```powershell
python -m AI_service.main teacher-chat `
  --class-id MAE101_SP26 `
  --analytics-json data/latest_class_analysis.json `
  --message "Hãy đề xuất kế hoạch ôn tập 2 tuần."
```

## 8. AI Phân Loại Bài Toán

Dùng để tag câu hỏi theo chương, topic, subtopic, độ khó và dạng bài.

```powershell
python -m AI_service.main classify-question `
  --question "Prove that a set of vectors is linearly independent..."
```

## 9. Import Và Search Vector

Import `Data_Bank.json` vào PostgreSQL:

```powershell
python -m AI_service.main import-json --input Data_Bank.json
```

Search theo vector:

```powershell
python -m AI_service.main search `
  --query "linear independence vector spaces" `
  --limit 5
```

## Mapping Với Doc.Md

| Feature trong Doc.Md | Command |
| --- | --- |
| 9. AI giải thích từng câu sai | `explain-wrong` |
| 29. Tự kiểm tra năng lực | `self-assess` |
| 42. AI sinh câu hỏi tự động | `generate` |
| 51. AI chấm bài tự luận | `evaluate-answer` |
| 55. Xếp loại sinh viên tự động | `analyze-class` |
| 56. AI phân tích điểm yếu | `analyze-class` |
| 57. AI chatbox cho giáo viên | `teacher-chat` |

## Lưu Ý

- Các command trả output dạng JSON để backend dễ parse.
- AI chấm bài chỉ nên là điểm đề xuất. Giáo viên vẫn cần quyền override.
- Các câu hỏi AI sinh ra nên có trạng thái `verified = false` cho đến khi giáo viên duyệt.
- Với ảnh chụp đề, có thể OCR ở backend rồi truyền `--ocr-text` để ổn định hơn.
