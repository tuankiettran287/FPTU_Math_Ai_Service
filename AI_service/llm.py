import argparse
from pathlib import Path
from typing import Any, Dict

from .utils import missing_dependency_error, parse_json_object, parse_json_or_raw


def adapter_path_or_exit(args: argparse.Namespace) -> str | None:
    adapter = str(args.adapter) if getattr(args, "adapter", None) else None
    if adapter and not Path(adapter).exists():
        raise SystemExit(
            f"LoRA adapter not found: {adapter}\n"
            "Train first with scripts/train_deepseek_lora.py, or pass the correct --adapter path."
        )
    return adapter


def load_ai_runtime():
    try:
        from infer_mathai import generate as generate_text
        from infer_mathai import load_model as load_ai_model
    except ModuleNotFoundError as exc:
        raise missing_dependency_error(exc) from exc
    return generate_text, load_ai_model


def load_llm(args: argparse.Namespace):
    generate_text, load_ai_model = load_ai_runtime()
    tokenizer, model = load_ai_model(args.base_model, adapter_path_or_exit(args))
    return generate_text, tokenizer, model


def generate_json_with_retries(
    generate_text,
    tokenizer,
    model,
    prompt: str,
    args: argparse.Namespace,
    strict_json: bool = True,
) -> Dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, args.retries + 2):
        output = generate_text(
            tokenizer,
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        if not strict_json:
            return parse_json_or_raw(output)
        try:
            return parse_json_object(output)
        except ValueError as exc:
            last_error = exc
            print(f"invalid JSON, attempt {attempt}: {exc}")
    raise RuntimeError("Could not generate valid JSON") from last_error
