import json
import threading
from typing import Any

from .config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MODEL_LOCAL_FILES_ONLY,
    MODEL_NAME,
    PROMPT_PATH,
)
from .utils import parse_json_object, read_text


def _missing_dependency_error(exc: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        f"Missing Python package '{exc.name}'. Install dependencies with: python -m pip install -r requirements.txt"
    )


class LocalDeepSeekClient:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ModuleNotFoundError as exc:
                raise _missing_dependency_error(exc) from exc

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                local_files_only=MODEL_LOCAL_FILES_ONLY,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": MODEL_LOCAL_FILES_ONLY,
                "low_cpu_mem_usage": True,
            }
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float32

            model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model

    def _render_messages(self, task: str, payload: dict[str, Any]) -> str:
        self.load()
        assert self._tokenizer is not None

        base_prompt = read_text(PROMPT_PATH)
        user_prompt = (
            f"TASK_ID: {task}\n\n"
            "INPUT_JSON:\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Return only one valid JSON object. Do not add markdown or prose outside JSON."
        )
        messages = [
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(f"{message['role'].upper()}:\n{message['content']}" for message in messages) + "\n\nASSISTANT:\n"

    def generate_text(
        self,
        task: str,
        payload: dict[str, Any],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        self.load()
        assert self._tokenizer is not None
        assert self._model is not None

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise _missing_dependency_error(exc) from exc

        prompt = self._render_messages(task, payload)
        input_device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt").to(input_device)
        generation_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        do_sample = generation_temperature > 0
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens or DEFAULT_MAX_NEW_TOKENS,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = generation_temperature
            generation_kwargs["top_p"] = DEFAULT_TOP_P if top_p is None else top_p

        with self._generate_lock:
            with torch.no_grad():
                output = self._model.generate(**generation_kwargs)

        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_json(
        self,
        task: str,
        payload: dict[str, Any],
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        text = self.generate_text(
            task=task,
            payload=payload,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return parse_json_object(text)


client = LocalDeepSeekClient()
