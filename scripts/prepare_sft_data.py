import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = (
    "Bạn là FPTU_MATHAI, trợ lý AI chuyên giải toán và ra đề toán cho sinh viên IT FPT. "
    "Luôn ưu tiên tính đúng đắn, trình bày từng bước rõ ràng, dùng LaTeX khi cần, "
    "và khi đánh giá bài làm phải kết luận ĐÚNG hoặc SAI trước khi giải thích."
)

DIFFICULTY_VI = {
    "easy": "dễ",
    "medium": "vừa",
    "hard": "khó",
}


def load_bank(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Data_Bank.json must be a JSON array.")
    return [item for item in data if isinstance(item, dict)]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    return str(value)


def question_text(item: Dict[str, Any]) -> str:
    q = item.get("question") or {}
    parts = []
    if q.get("text"):
        parts.append(f"Đề bài: {clean_text(q.get('text'))}")
    if q.get("latex"):
        parts.append(f"Công thức/LaTeX: {clean_text(q.get('latex'))}")
    if q.get("answer_format"):
        parts.append(f"Định dạng trả lời: {clean_text(q.get('answer_format'))}")
    return "\n".join(parts).strip()


def step_text(step: Dict[str, Any]) -> str:
    number = step.get("step_number", "")
    title = clean_text(step.get("title"))
    content = clean_text(step.get("content"))
    formula = clean_text(step.get("formula") or step.get("latex"))
    reasoning = clean_text(step.get("reasoning"))
    line = f"Bước {number}: {title}".strip()
    details = [x for x in [content, f"Công thức: {formula}" if formula else "", reasoning] if x]
    return f"{line}\n" + "\n".join(details)


def solution_text(item: Dict[str, Any]) -> str:
    solution = item.get("solution") or {}
    final_answer = clean_text(solution.get("final_answer"))
    steps = solution.get("steps") or []
    lines = []
    for step in steps:
        if isinstance(step, dict):
            lines.append(step_text(step))
    if final_answer:
        lines.append(f"Kết luận: {final_answer}")
    return "\n\n".join(lines).strip()


def exercise_json(item: Dict[str, Any]) -> str:
    q = item.get("question") or {}
    difficulty = item.get("difficulty") or {}
    solution = item.get("solution") or {}
    payload = {
        "course": item.get("course"),
        "chapter": item.get("chapter"),
        "topic": item.get("topic"),
        "subtopic": item.get("subtopic"),
        "difficulty": difficulty.get("level"),
        "question_type": item.get("question_type"),
        "question": {
            "text": q.get("text"),
            "latex": q.get("latex"),
        },
        "expected_answer": solution.get("final_answer"),
        "solution_outline": [
            {
                "step_number": s.get("step_number"),
                "title": s.get("title"),
                "formula": s.get("formula") or s.get("latex"),
            }
            for s in solution.get("steps", [])
            if isinstance(s, dict)
        ],
        "hints": item.get("hints") or [],
        "common_mistakes": item.get("common_mistakes") or [],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def make_message(instruction: str, answer: str, task: str, source_id: str) -> Dict[str, Any]:
    return {
        "task": task,
        "source_id": source_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
    }


def build_examples(item: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    source_id = clean_text(item.get("id"))
    chapter = clean_text(item.get("chapter"))
    topic = clean_text(item.get("topic"))
    subtopic = clean_text(item.get("subtopic"))
    difficulty = clean_text((item.get("difficulty") or {}).get("level"))
    difficulty_vi = DIFFICULTY_VI.get(difficulty, difficulty)
    q_text = question_text(item)
    s_text = solution_text(item)
    final_answer = clean_text((item.get("solution") or {}).get("final_answer"))

    if q_text and s_text:
        yield make_message(
            instruction=(
                "Giải bài toán sau cho sinh viên FPT. Trình bày từng bước, nêu công thức chính, "
                "và kết luận đáp án cuối.\n\n"
                f"{q_text}"
            ),
            answer=s_text,
            task="solve_math",
            source_id=source_id,
        )

    yield make_message(
        instruction=(
            "Tạo một bài tập toán theo đúng chương, chủ đề và độ khó sau. "
            "Trả về JSON có đề bài, đáp án, lời giải tóm tắt, gợi ý và lỗi thường gặp.\n\n"
            f"Chương: {chapter}\nChủ đề: {topic}\nChủ đề con: {subtopic}\nĐộ khó: {difficulty_vi}"
        ),
        answer=exercise_json(item),
        task="generate_exercise",
        source_id=source_id,
    )

    if q_text and final_answer:
        correct_eval = {
            "verdict": "ĐÚNG",
            "is_correct": True,
            "score": 1.0,
            "feedback": "Bài làm khớp với đáp án kỳ vọng.",
            "expected_answer": final_answer,
        }
        yield make_message(
            instruction=(
                "Đánh giá bài tự luận sau. Dòng đầu tiên phải kết luận ĐÚNG hoặc SAI, "
                "sau đó giải thích ngắn gọn.\n\n"
                f"{q_text}\n\nBài làm sinh viên:\n{final_answer}"
            ),
            answer="ĐÚNG\n" + json.dumps(correct_eval, ensure_ascii=False, indent=2),
            task="evaluate_answer",
            source_id=source_id,
        )

    mistakes = [clean_text(x) for x in (item.get("common_mistakes") or []) if clean_text(x)]
    for mistake in mistakes[:2]:
        wrong_eval = {
            "verdict": "SAI",
            "is_correct": False,
            "score": 0.0,
            "feedback": f"Bài làm mắc lỗi: {mistake}",
            "expected_answer": final_answer,
            "next_hint": "So sánh từng bước biến đổi với công thức chuẩn và kiểm tra điều kiện áp dụng.",
        }
        yield make_message(
            instruction=(
                "Đánh giá bài tự luận sau. Dòng đầu tiên phải kết luận ĐÚNG hoặc SAI, "
                "rồi chỉ ra lỗi sai chính và gợi ý sửa.\n\n"
                f"{q_text}\n\nBài làm sinh viên:\n{mistake}"
            ),
            answer="SAI\n" + json.dumps(wrong_eval, ensure_ascii=False, indent=2),
            task="evaluate_answer",
            source_id=source_id,
        )

        yield make_message(
            instruction=(
                "Sinh viên vừa làm sai bài toán này. Hãy giải thích lỗi sai dễ hiểu và đưa ra hướng sửa, "
                "không đưa thêm bài mới.\n\n"
                f"{q_text}\n\nLỗi sinh viên: {mistake}"
            ),
            answer=(
                f"Lỗi chính: {mistake}\n\n"
                "Cách sửa: quay lại bước dùng định nghĩa hoặc công thức nền tảng, kiểm tra điều kiện áp dụng, "
                "rồi thay số/biến cẩn thận. "
                f"Đáp án đúng cần hướng tới là: {final_answer}" if final_answer else f"Lỗi chính: {mistake}"
            ),
            task="explain_mistake",
            source_id=source_id,
        )

    hints = [clean_text(x) for x in (item.get("hints") or []) if clean_text(x)]
    if hints and q_text:
        yield make_message(
            instruction=(
                "Tôi là sinh viên IT FPT và chưa biết bắt đầu bài này từ đâu. "
                "Hãy hướng dẫn từng gợi ý, chưa cần giải hết ngay.\n\n"
                f"{q_text}"
            ),
            answer="\n".join(f"Gợi ý {idx + 1}: {hint}" for idx, hint in enumerate(hints[:4])),
            task="chat_tutor",
            source_id=source_id,
        )


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-task SFT data for FPTU_MATHAI.")
    parser.add_argument("--input", default="Data_Bank.json", type=Path)
    parser.add_argument("--output-dir", default=Path("data/sft"), type=Path)
    parser.add_argument("--validation-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    items = load_bank(args.input)
    examples: List[Dict[str, Any]] = []
    for item in items:
        examples.extend(build_examples(item))

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    valid_size = max(1, int(len(examples) * args.validation_ratio))
    valid_rows = examples[:valid_size]
    train_rows = examples[valid_size:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_count = write_jsonl(args.output_dir / "train.jsonl", train_rows)
    valid_count = write_jsonl(args.output_dir / "valid.jsonl", valid_rows)

    stats = {
        "source_items": len(items),
        "train_examples": train_count,
        "valid_examples": valid_count,
        "tasks": Counter(x["task"] for x in examples),
        "chapters": Counter(clean_text(x.get("chapter")) for x in items),
        "difficulty": Counter(clean_text((x.get("difficulty") or {}).get("level")) for x in items),
    }
    serializable_stats = {
        k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()
    }
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(serializable_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(serializable_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
