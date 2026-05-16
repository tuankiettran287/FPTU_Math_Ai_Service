import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
DEFAULT_LORA_ADAPTER = ROOT_DIR / "outputs" / "deepseek-fptu-mathai-lora"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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

TEXT_COLUMNS = ["id", "subject", "course", "chapter", "topic", "subtopic", "question_type"]
JSON_COLUMNS = [
    "difficulty",
    "question",
    "solution",
    "concepts_used",
    "prerequisites",
    "common_mistakes",
    "hints",
    "evaluation",
    "metadata",
]

AI_INTERACTIONS_TABLE = "ai_interactions"
AI_EVALUATIONS_TABLE = "ai_evaluations"
AI_CLASS_ANALYTICS_TABLE = "ai_class_analytics"

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/fptu_mathai"
DEFAULT_TABLE = "math_question_bank"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

