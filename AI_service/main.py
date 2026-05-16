import argparse
import os
from pathlib import Path

from .commands import (
    analyze_class,
    classify_question,
    evaluate_answer,
    explain_wrong_answer,
    generate_and_store,
    import_json,
    search,
    self_assess,
    solve_uploaded_and_store,
    teacher_chat,
)
from .config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DB_URL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LORA_ADAPTER,
    DEFAULT_TABLE,
)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", DEFAULT_DB_URL))
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--subject", default="MAE")
    parser.add_argument("--course", default="Mathematics for Engineering")
    parser.add_argument("--chapter", default="Functions and Graphs")
    parser.add_argument("--topic", default="Trigonometry")
    parser.add_argument("--subtopic", default="")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--question-type", default="problem_solving")
    parser.add_argument("--language", default="en")


def add_model_args(parser: argparse.ArgumentParser, temperature: float = 0.2) -> None:
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_LORA_ADAPTER, type=Path)
    parser.add_argument("--max-new-tokens", default=1200, type=int)
    parser.add_argument("--temperature", default=temperature, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--retries", default=2, type=int)


def add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--user-id", default="")
    parser.add_argument("--student-id", default="")
    parser.add_argument("--class-id", default="")
    parser.add_argument("--role-context", default="")


def add_question_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--question-id", default="")
    parser.add_argument("--question", default="")
    parser.add_argument("--question-file", default=None, type=Path)
    parser.add_argument("--ocr-text", default="")
    parser.add_argument("--latex", default="")
    parser.add_argument("--image-path", default=None, type=Path)


def add_answer_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--student-answer", default="")
    parser.add_argument("--student-answer-file", default=None, type=Path)
    parser.add_argument("--answer-ocr-text", default="")
    parser.add_argument("--answer-image-path", default=None, type=Path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FPTU_MATHAI AI service.")
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate", help="Generate questions with AI and save them to PostgreSQL.")
    add_common_args(generate_parser)
    add_context_args(generate_parser)
    add_model_args(generate_parser, temperature=0.4)
    generate_parser.add_argument("--count", default=1, type=int)
    generate_parser.set_defaults(func=generate_and_store)

    solve_parser = subparsers.add_parser("solve-upload", help="Solve a user-uploaded problem and save it as a vector.")
    add_common_args(solve_parser)
    add_context_args(solve_parser)
    add_model_args(solve_parser, temperature=0.2)
    add_question_input_args(solve_parser)
    solve_parser.set_defaults(func=solve_uploaded_and_store)

    evaluate_parser = subparsers.add_parser("evaluate-answer", help="AI grade a student answer with verdict, score, and feedback.")
    add_common_args(evaluate_parser)
    add_context_args(evaluate_parser)
    add_model_args(evaluate_parser, temperature=0.1)
    add_question_input_args(evaluate_parser)
    add_answer_input_args(evaluate_parser)
    evaluate_parser.add_argument("--rubric", default="")
    evaluate_parser.add_argument("--submission-id", default="")
    evaluate_parser.set_defaults(func=evaluate_answer)

    explain_parser = subparsers.add_parser("explain-wrong", help="Explain a wrong answer after submission.")
    add_common_args(explain_parser)
    add_context_args(explain_parser)
    add_model_args(explain_parser, temperature=0.2)
    add_question_input_args(explain_parser)
    add_answer_input_args(explain_parser)
    explain_parser.add_argument("--evaluation-json", default=None, type=Path)
    explain_parser.set_defaults(func=explain_wrong_answer)

    assess_parser = subparsers.add_parser("self-assess", help="Generate or grade a self-study diagnostic assessment.")
    add_common_args(assess_parser)
    add_context_args(assess_parser)
    add_model_args(assess_parser, temperature=0.3)
    assess_parser.add_argument("--topics", default="Discrete Mathematics, Linear Algebra, Calculus")
    assess_parser.add_argument("--num-questions", default=10, type=int)
    assess_parser.add_argument("--answers-json", default=None, type=Path)
    assess_parser.set_defaults(func=self_assess)

    analyze_parser = subparsers.add_parser("analyze-class", help="Analyze class weak chapters, weak topics, and support actions.")
    add_common_args(analyze_parser)
    add_model_args(analyze_parser, temperature=0.2)
    analyze_parser.add_argument("--user-id", default="")
    analyze_parser.add_argument("--role-context", default="teacher")
    analyze_parser.add_argument("--class-id", required=True)
    analyze_parser.add_argument("--input-json", required=True, type=Path)
    analyze_parser.set_defaults(func=analyze_class)

    teacher_chat_parser = subparsers.add_parser("teacher-chat", help="Teacher AI chatbox over class analytics and question bank.")
    add_common_args(teacher_chat_parser)
    add_context_args(teacher_chat_parser)
    add_model_args(teacher_chat_parser, temperature=0.2)
    teacher_chat_parser.add_argument("--message", required=True)
    teacher_chat_parser.add_argument("--analytics-json", default=None, type=Path)
    teacher_chat_parser.add_argument("--context-limit", default=5, type=int)
    teacher_chat_parser.set_defaults(func=teacher_chat)

    classify_parser = subparsers.add_parser("classify-question", help="Classify a math problem into chapter/topic/difficulty/type.")
    add_common_args(classify_parser)
    add_context_args(classify_parser)
    add_model_args(classify_parser, temperature=0.1)
    add_question_input_args(classify_parser)
    classify_parser.set_defaults(func=classify_question)

    import_parser = subparsers.add_parser("import-json", help="Import Data_Bank.json rows into PostgreSQL with embeddings.")
    add_common_args(import_parser)
    import_parser.add_argument("--input", default=Path("Data_Bank.json"), type=Path)
    import_parser.set_defaults(func=import_json)

    search_parser = subparsers.add_parser("search", help="Search saved questions by vector similarity.")
    search_parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", DEFAULT_DB_URL))
    search_parser.add_argument("--table", default=DEFAULT_TABLE)
    search_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--limit", default=5, type=int)
    search_parser.set_defaults(func=search)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
