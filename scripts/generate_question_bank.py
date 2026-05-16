import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from infer_mathai import generate, load_model


def extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"raw_output": text, "parse_error": "No JSON object found"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"raw_output": text, "parse_error": str(exc)}


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additional FPTU_MATHAI question-bank candidates.")
    parser.add_argument("--base-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--chapters", default="Discrete Mathematics,Linear Algebra,Calculus")
    parser.add_argument("--topics", default="")
    parser.add_argument("--difficulties", default="easy,medium,hard")
    parser.add_argument("--per-combination", default=3, type=int)
    parser.add_argument("--output", default=Path("data/generated_bank.jsonl"), type=Path)
    parser.add_argument("--max-new-tokens", default=900, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.base_model, args.adapter)
    chapters = split_csv(args.chapters)
    topics = split_csv(args.topics) or ["core topic"]
    difficulties = split_csv(args.difficulties)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for chapter in chapters:
            for topic in topics:
                for difficulty in difficulties:
                    for index in range(args.per_combination):
                        prompt = (
                            "Tạo một bài toán mới cho ngân hàng đề FPTU_MATHAI. "
                            "Ưu tiên sinh viên IT FPT, nội dung toán nền tảng cho lập trình/AI/data. "
                            "Trả về đúng một JSON object có các field: id, subject, course, chapter, topic, "
                            "subtopic, difficulty, question_type, question, solution, concepts_used, "
                            "prerequisites, common_mistakes, hints, evaluation, metadata. "
                            "Không copy y nguyên bài có sẵn.\n\n"
                            f"Chương: {chapter}\nTopic: {topic}\nĐộ khó: {difficulty}\nMẫu số: {index + 1}"
                        )
                        text = generate(
                            tokenizer,
                            model,
                            prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                        )
                        row = extract_json(text)
                        row["_generation_context"] = {
                            "chapter": chapter,
                            "topic": topic,
                            "difficulty": difficulty,
                            "index": index + 1,
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        print(f"generated {chapter} | {topic} | {difficulty} | {index + 1}")


if __name__ == "__main__":
    main()
