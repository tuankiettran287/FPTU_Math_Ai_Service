import argparse
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "Bạn là FPTU_MATHAI, trợ lý AI chuyên giải toán và ra đề toán cho sinh viên IT FPT. "
    "Luôn ưu tiên tính đúng đắn, trình bày từng bước rõ ràng, dùng LaTeX khi cần, "
    "và khi đánh giá bài làm phải kết luận ĐÚNG hoặc SAI trước khi giải thích."
)


def fallback_chat_template(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    text = ""
    for message in messages:
        text += f"<|{message['role']}|>\n{message['content']}\n"
    if add_generation_prompt:
        text += "<|assistant|>\n"
    return text


def render_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return fallback_chat_template(messages, add_generation_prompt=True)


def load_model(base_model: str, adapter: str | None):
    tokenizer = AutoTokenizer.from_pretrained(adapter or base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def build_user_prompt(args: argparse.Namespace) -> str:
    if args.task == "generate":
        return (
            "Tạo một bài tập toán mới cho sinh viên IT FPT. Trả về JSON gồm: "
            "chapter, topic, difficulty, question, expected_answer, solution_steps, hints, common_mistakes.\n\n"
            f"Chương: {args.chapter}\nChủ đề: {args.topic}\nĐộ khó: {args.difficulty}"
        )
    if args.task == "solve":
        latex = f"\nCông thức/LaTeX: {args.latex}" if args.latex else ""
        return (
            "Giải bài toán sau từng bước. Nêu công thức chính và kết luận đáp án cuối.\n\n"
            f"Đề bài: {args.question}{latex}"
        )
    if args.task == "evaluate":
        return (
            "Đánh giá bài tự luận sau. Dòng đầu tiên phải kết luận ĐÚNG hoặc SAI. "
            "Sau đó giải thích lỗi hoặc xác nhận đúng, rồi đưa đáp án kỳ vọng nếu cần.\n\n"
            f"Đề bài: {args.question}\n\nBài làm sinh viên:\n{args.student_answer}"
        )
    raise ValueError(f"Unsupported one-shot task: {args.task}")


def generate(tokenizer, model, user_prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FPTU_MATHAI DeepSeek LoRA inference.")
    parser.add_argument("--base-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--task", choices=["chat", "generate", "solve", "evaluate"], required=True)
    parser.add_argument("--chapter", default="Functions and Graphs")
    parser.add_argument("--topic", default="Mathematics")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--question", default="")
    parser.add_argument("--latex", default="")
    parser.add_argument("--student-answer", default="")
    parser.add_argument("--max-new-tokens", default=768, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.base_model, args.adapter)

    if args.task == "chat":
        print("FPTU_MATHAI chat. Gõ 'exit' để thoát.")
        history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        while True:
            user_text = input("Bạn: ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            history.append({"role": "user", "content": user_text})
            prompt = render_prompt(tokenizer, history)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                output[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            ).strip()
            print(f"AI: {response}\n")
            history.append({"role": "assistant", "content": response})
        return

    print(generate(tokenizer, model, build_user_prompt(args), args.max_new_tokens, args.temperature, args.top_p))


if __name__ == "__main__":
    main()
