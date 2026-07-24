import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

# Nạp .env nếu có python-dotenv (không bắt buộc).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass

APP_NAME = "FPTU_MATHAI AI Service"
APP_VERSION = "3.0.0"

PROMPT_PATH = Path(os.getenv("PROMPT_PATH", ROOT_DIR / "prompt.txt"))

# ──────────────────────────────────────────────────────────────────────────────
# LLM backend
#   - "vllm": gọi tới một máy chủ vLLM (OpenAI-compatible) chạy trên GPU pod.
#   - "transformers": chạy trực tiếp bằng transformers (chỉ để dev/test, chậm).
# Model không fine-tune, chỉ inference.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

LLM_BACKEND = os.getenv("LLM_BACKEND", "vllm").lower()  # "vllm" | "transformers"

# vLLM (OpenAI-compatible) — ví dụ RunPod: https://<pod>-8000.proxy.runpod.net/v1
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM chạy local không cần key thật
VLLM_MODEL = os.getenv("VLLM_MODEL", MODEL_NAME)
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "180"))

# transformers fallback
MODEL_LOCAL_FILES_ONLY = os.getenv("MODEL_LOCAL_FILES_ONLY", "true").lower() not in {"0", "false", "no"}

# ──────────────────────────────────────────────────────────────────────────────
# Database — Postgres + pgvector (AI service giữ DB riêng của nó)
# ──────────────────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/fptu_mathai_final",
)

# ──────────────────────────────────────────────────────────────────────────────
# Embedding — bge-m3 (đa ngôn ngữ, tốt cho tiếng Việt + LaTeX). 1024 chiều.
#   backend "remote": gọi HTTP endpoint embedding trên GPU pod (OpenAI-compatible
#                     /embeddings). MẶC ĐỊNH — máy dev không có GPU, KHÔNG load model.
#   backend "local":  chạy SentenceTransformer tại chỗ (chỉ khi máy đủ khoẻ).
# ──────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "remote").lower()  # "remote" | "local"
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
# remote endpoint (OpenAI-compatible). Ví dụ RunPod: https://<pod>-8001.proxy.runpod.net/v1
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:8001/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "EMPTY")
EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", EMBEDDING_MODEL)
EMBEDDING_TIMEOUT = float(os.getenv("EMBEDDING_TIMEOUT", "120"))
# local backend
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # auto | cpu | cuda

# ──────────────────────────────────────────────────────────────────────────────
# OCR — DeepSeek-R1 KHÔNG nhìn được ảnh, nên ảnh (đề chụp/viết tay) phải OCR ra text
# TRƯỚC khi đưa vào LLM. Chạy TRÊN POD (không load model lên máy dev).
#   backend "rapidocr": RapidOCR (onnxruntime, CPU) — text in ấn, không đụng torch/vLLM.
#   backend "pix2tex" : LaTeX-OCR — chuyên công thức toán → LaTeX (cần torch, đã có sẵn).
#   backend "vlm"     : gửi ảnh tới 1 endpoint vision OpenAI-compatible (OCR_VLM_BASE_URL)
#                       — khi bạn serve thêm 1 VLM (VD Qwen2-VL) trên pod. Hiểu ảnh tốt nhất.
#   backend "none"    : tắt OCR (trả lỗi rõ ràng để FE báo người dùng).
# ──────────────────────────────────────────────────────────────────────────────
OCR_BACKEND = os.getenv("OCR_BACKEND", "rapidocr").lower()
OCR_VLM_BASE_URL = os.getenv("OCR_VLM_BASE_URL", "")          # vd http://localhost:8002/v1
OCR_VLM_MODEL = os.getenv("OCR_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
OCR_VLM_API_KEY = os.getenv("OCR_VLM_API_KEY", "EMPTY")
OCR_TIMEOUT = float(os.getenv("OCR_TIMEOUT", "120"))
OCR_MAX_IMAGE_BYTES = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))  # 8MB

# ──────────────────────────────────────────────────────────────────────────────
# Sinh: tách 2 giai đoạn — để <think> tự do, ép JSON ở đoạn kết luận.
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))

# Ngân sách token + timeout riêng theo từng loại task (không dùng 1 con số chung).
# think_tokens = phần suy luận tự do; json_tokens = phần ép JSON kết luận.
TASK_LIMITS: dict[str, dict[str, int]] = {
    "exam_generation":   {"think": 3500, "json": 6000, "timeout": 240},
    "generate_mcq":      {"think": 3000, "json": 4000, "timeout": 180},
    "generate_essay":    {"think": 3000, "json": 4000, "timeout": 200},
    # think 5500: benchmark 15/07 đo token <think> khi giải đề khó — p99=2.504, max=4.691.
    # Trần cũ 3500 làm 1/360 câu khó bị cắt JSON (hỏng hẳn, không suy giảm dần). Ngân sách
    # thừa không tốn gì nếu model không dùng tới.
    "solve_full":        {"think": 5500, "json": 2500, "timeout": 240},
    "solve_hint":        {"think": 2500, "json": 1200, "timeout": 120},
    "grade_submission":  {"think": 3500, "json": 3000, "timeout": 220},
    # Chấm tự luận trả THÊM cách giải chi tiết + lỗi theo từng bước + điểm từng ý a/b/c
    # → JSON dài hơn grade_submission nhiều.
    # timeout 600: chạy trên 32B (chậm hơn 7B ~3–4×) và sinh ~9k token; 280s là KHÔNG đủ.
    # Chấm chạy nền (EssayGradingWorker) nên chờ lâu không ảnh hưởng SV.
    "grade_essay":       {"think": 4000, "json": 5000, "timeout": 600},
    "explain_mistake":   {"think": 3000, "json": 2500, "timeout": 160},
    "concept_lookup":    {"think": 2500, "json": 2000, "timeout": 140},
    "arena_generation":  {"think": 2500, "json": 4000, "timeout": 180},
    "mistake_generation":{"think": 2500, "json": 4000, "timeout": 200},
    "learning_analysis": {"think": 2500, "json": 3000, "timeout": 180},
    "study_path":        {"think": 2500, "json": 3000, "timeout": 180},
    # ── Đánh giá năng lực ─────────────────────────────────────────────────────
    # Sinh 1 câu tự luận kèm accepted_methods + grading_notes (JSON dài hơn essay thường).
    "competency_generate":    {"think": 3000, "json": 4000, "timeout": 240},
    # Giải độc lập để đối chiếu đáp án — đề khó cần tới 4.691 token nghĩ (đo thật
    # ở benchmark, p99=2.504/max=4.691) → 5500 cho dư biên; hết ngân sách think
    # giữa chừng thì JSON không bao giờ được in ra (hỏng hẳn, không suy giảm dần).
    "competency_solve_check": {"think": 5500, "json": 1500, "timeout": 300},
    # Critic chấm 7 tiêu chí kiểm định đề.
    "competency_critic":      {"think": 3000, "json": 3000, "timeout": 240},
    # Chấm bài đánh giá năng lực: 2 call riêng (khung syllabus / khung GMC),
    # chạy nền qua worker → timeout dài như grade_essay.
    "competency_grade":       {"think": 4000, "json": 5000, "timeout": 600},
    "competency_hint":        {"think": 2500, "json": 1500, "timeout": 120},
    "default":           {"think": 3000, "json": 3000, "timeout": 180},
}


# ── Fix #3 (benchmark 15/07): RAG làm HẠI tác vụ đối chiếu bài làm ─────────────
# grade/grade_essay đã KHÔNG dùng RAG (đúng). explain_mistake vẫn dùng — cùng bản chất
# "đối chiếu bài sai với đáp án" nên NGHI hại, nhưng CHƯA đo riêng. Giữ bật mặc định;
# đặt EXPLAIN_MISTAKE_USE_RAG=false để tắt sau khi đo mà không cần sửa code.
EXPLAIN_MISTAKE_USE_RAG = os.getenv("EXPLAIN_MISTAKE_USE_RAG", "true").lower() not in {"0", "false", "no"}


def task_limits(task: str) -> dict[str, int]:
    return TASK_LIMITS.get(task, TASK_LIMITS["default"])


DATA_BANK_FIELDS = [
    "id", "subject", "assest_format", "course", "chapter", "topic", "subtopic",
    "difficulty", "question_type", "question", "solution", "concepts_used",
    "prerequisites", "common_mistakes", "hints", "evaluation", "metadata",
]
