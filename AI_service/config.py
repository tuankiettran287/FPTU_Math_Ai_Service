import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

APP_NAME = "FPTU_MATHAI AI Service"
APP_VERSION = "2.0.0"

PROMPT_PATH = Path(os.getenv("PROMPT_PATH", ROOT_DIR / "prompt.txt"))

# Use a local folder in DEEPSEEK_MODEL_PATH when the model has already been
# downloaded. The Hugging Face id is kept as a fallback for machines with cache.
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_NAME = os.getenv("DEEPSEEK_MODEL_PATH", os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME))
MODEL_LOCAL_FILES_ONLY = os.getenv("MODEL_LOCAL_FILES_ONLY", "true").lower() not in {"0", "false", "no"}

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'mathai.db'}")

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4096"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.9"))

DATA_BANK_FIELDS = [
    "id",
    "subject",
    "course",
    "chapter",
    "topic",
    "subtopic",
    "difficulty",
    "question_type",
    "question",
    "solution",
    "concepts_used",
    "prerequisites",
    "common_mistakes",
    "hints",
    "evaluation",
    "metadata",
]
