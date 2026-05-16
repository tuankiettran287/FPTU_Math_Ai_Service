import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def fallback_chat_template(messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    chunks = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        chunks.append(f"<|{role}|>\n{content}\n")
    if add_generation_prompt:
        chunks.append("<|assistant|>\n")
    return "".join(chunks)


def render_chat(tokenizer: AutoTokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return fallback_chat_template(messages, add_generation_prompt=add_generation_prompt)


def tokenize_example(example: Dict[str, Any], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    messages = example["messages"]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError("Each row must contain messages ending with an assistant response.")

    prompt_messages = messages[:-1]
    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)

    full = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
    prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    prompt_len = min(len(prompt["input_ids"]), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    if all(label == -100 for label in labels) and input_ids:
        labels[-1] = input_ids[-1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class DataCollatorForCausalLM:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            batch["labels"].append(item["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek for FPTU_MATHAI with LoRA/QLoRA.")
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--train-file", default="data/sft/train.jsonl", type=Path)
    parser.add_argument("--valid-file", default="data/sft/valid.jsonl", type=Path)
    parser.add_argument("--output-dir", default="outputs/deepseek-fptu-mathai-lora", type=Path)
    parser.add_argument("--max-length", default=2048, type=int)
    parser.add_argument("--epochs", default=3.0, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--grad-accum", default=8, type=int)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--warmup-ratio", default=0.03, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--save-steps", default=100, type=int)
    parser.add_argument("--logging-steps", default=10, type=int)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--lora-r", default=16, type=int)
    parser.add_argument("--lora-alpha", default=32, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and not args.use_4bit else None,
    )
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.valid_file)},
    )
    tokenized = raw.map(
        lambda row: tokenize_example(row, tokenizer, args.max_length),
        remove_columns=raw["train"].column_names,
        desc="Tokenizing chat data",
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForCausalLM(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    run_config = vars(args)
    run_config["output_dir"] = str(args.output_dir)
    run_config["train_file"] = str(args.train_file)
    run_config["valid_file"] = str(args.valid_file)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
