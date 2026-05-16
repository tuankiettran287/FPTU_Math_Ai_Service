# FPTU_MATHAI

FPTU_MATHAI là phần AI service cho hệ thống học toán của sinh viên FPT. Mục tiêu chính là xây dựng ngân hàng đề toán, fine-tune DeepSeek bằng LoRA, sinh đề, giải toán, chấm bài tự luận, phân tích điểm yếu và lưu dữ liệu dạng vector trong PostgreSQL.

Project hiện tập trung vào các môn toán phục vụ sinh viên IT như Mathematics for Engineering, Linear Algebra, Calculus và Discrete Mathematics.

## Cấu Trúc Dự Án

```txt
FPTU_MathAI/
  AI_service/
    main.py        # CLI chính để chạy toàn bộ chức năng AI
    commands.py    # logic từng chức năng AI
    db.py          # PostgreSQL + pgvector
    llm.py         # load DeepSeek base + LoRA adapter
    prompts.py     # prompt cho từng task
    schemas.py     # chuẩn hóa schema giống Data_Bank.json
    utils.py       # helper JSON, OCR, vector
    config.py      # cấu hình mặc định
  scripts/
    prepare_sft_data.py
    train_deepseek_lora.py
    infer_mathai.py
    generate_question_bank.py
  data/sft/
    train.jsonl
    valid.jsonl
    dataset_stats.json
  Data_Bank.json
  AI_FEATURES.md
  Doc.Md
  requirements.txt
  main.py          # wrapper gọi AI_service.main
```

## Model Và Database

Model mặc định:

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- LoRA adapter: `outputs/deepseek-fptu-mathai-lora`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector database: PostgreSQL + `pgvector`

Các bảng chính do `AI_service` tự tạo:

- `math_question_bank`: lưu đề, lời giải, metadata và `embedding vector(...)`
- `ai_interactions`: lưu lịch sử gọi AI
- `ai_evaluations`: lưu kết quả AI chấm bài tự luận
- `ai_class_analytics`: lưu kết quả phân tích điểm yếu của lớp

## Cài Đặt

```powershell
python -m pip install -r requirements.txt
```

Cấu hình PostgreSQL:

```powershell
$env:DATABASE_URL="postgresql://postgres:postgres@localhost:5432/fptu_mathai"
```

PostgreSQL cần cài extension `pgvector`. Script sẽ tự chạy:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

OCR ảnh dùng `pytesseract`, nên nếu chạy chức năng đọc ảnh trực tiếp thì Windows cần cài thêm Tesseract OCR. Nếu backend đã OCR ảnh thành text thì chỉ cần truyền text bằng `--ocr-text`.

## Chuẩn Bị Dữ Liệu Fine-Tune

Tạo SFT dataset từ `Data_Bank.json`:

```powershell
python scripts/prepare_sft_data.py `
  --input Data_Bank.json `
  --output-dir data/sft `
  --validation-ratio 0.1
```

Kết quả:

- `data/sft/train.jsonl`
- `data/sft/valid.jsonl`
- `data/sft/dataset_stats.json`

## Fine-Tune DeepSeek Bằng LoRA

```powershell
python scripts/train_deepseek_lora.py `
  --model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B `
  --train-file data/sft/train.jsonl `
  --valid-file data/sft/valid.jsonl `
  --output-dir outputs/deepseek-fptu-mathai-lora `
  --epochs 3 `
  --batch-size 1 `
  --grad-accum 8
```

Nếu chạy Linux/Colab và có GPU phù hợp, có thể thêm `--use-4bit`. Trên Windows, `bitsandbytes` có thể không ổn định.

## Chạy AI Service

Có thể chạy bằng entrypoint mới:

```powershell
python -m AI_service.main --help
```

Hoặc wrapper cũ:

```powershell
python main.py --help
```

Các command chính:

```powershell
python -m AI_service.main generate
python -m AI_service.main solve-upload
python -m AI_service.main evaluate-answer
python -m AI_service.main explain-wrong
python -m AI_service.main self-assess
python -m AI_service.main analyze-class
python -m AI_service.main teacher-chat
python -m AI_service.main classify-question
python -m AI_service.main import-json
python -m AI_service.main search
```

## Import Data_Bank Vào Vector DB

```powershell
python -m AI_service.main import-json --input Data_Bank.json
```

Mỗi câu hỏi sẽ được lưu theo schema gốc và có thêm vector embedding để search.

## Sinh Đề Và Lưu Vector

```powershell
python -m AI_service.main generate `
  --chapter "Linear Algebra" `
  --topic "Vector Spaces" `
  --difficulty medium `
  --count 5
```

Output được lưu vào `math_question_bank`.

## Giải Đề User Upload

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

## Chấm Bài Tự Luận

```powershell
python -m AI_service.main evaluate-answer `
  --question-id MAE_MATH_0001 `
  --student-answer "x = 2" `
  --student-id SE180001 `
  --class-id MAE101_SP26 `
  --submission-id SUB_001
```

AI trả về verdict `DUNG` hoặc `SAI`, điểm, feedback, lỗi sai và gợi ý sửa.

## Search Vector

```powershell
python -m AI_service.main search `
  --query "linear independence vector spaces" `
  --limit 5
```

## Ghi Chú Kỹ Thuật

- `AI_service` không thay thế backend chính. Backend có thể gọi CLI này hoặc import module Python.
- Các output AI được ép trả về JSON để backend dễ parse.
- Câu hỏi sinh mới nên có bước giáo viên review trước khi đưa vào đề chính thức.
- Dataset hiện chưa đủ lớn cho một model toán mạnh. Nên tiếp tục bổ sung đề thật, lời giải đúng và dữ liệu lỗi sai của sinh viên.
