"""LLM client — mặc định gọi vLLM (OpenAI-compatible) trên GPU pod.

Điểm chính so với bản cũ:
- Bỏ `self._generate_lock` (nút thắt tuần tự). vLLM tự lo continuous batching phía
  server nên nhiều request chạy song song được.
- **Tách sinh 2 giai đoạn**: (1) để model suy luận tự do trong <think> rồi kết luận
  bằng văn xuôi; (2) ép JSON ở đoạn kết luận bằng guided_decoding (`guided_json`).
- Timeout + max_new_tokens **riêng theo từng task** (config.TASK_LIMITS), không dùng
  một con số chung.
- transformers vẫn giữ làm fallback dev (LLM_BACKEND=transformers), nhưng KHÔNG chạy
  trên máy không đủ khoẻ.
"""
import json
import threading
from typing import Any

from .config import (
    DEFAULT_TEMPERATURE, DEFAULT_TOP_P, LLM_BACKEND,
    MODEL_LOCAL_FILES_ONLY, MODEL_NAME, PROMPT_PATH, VLLM_API_KEY, VLLM_BASE_URL,
    VLLM_MODEL, VLLM_TIMEOUT, task_limits,
)
from .utils import dict_has_cjk, parse_json_object, read_text, strip_reasoning_blocks


_REASON_SUFFIX = (
    "\n\nHãy suy luận thật kỹ trong cặp thẻ <think></think>, kiểm tra lại đáp án cuối "
    "và các bước. Sau </think>, trình bày KẾT LUẬN rõ ràng bằng văn xuôi (đáp án cuối, "
    "các bước chính, phương án nhiễu nếu là trắc nghiệm). Ở bước này CHƯA cần JSON."
)
_FORMAT_SYSTEM = (
    "Bạn là bộ ĐỊNH DẠNG. Chuyển nội dung được cung cấp thành DUY NHẤT một JSON object "
    "hợp lệ theo yêu cầu. Không thêm bất kỳ chữ nào ngoài JSON, không markdown, không "
    "thẻ <think>. Giữ nguyên nội dung toán học/công thức, không tự bịa thêm."
)
# Guard rò chữ Hán dùng CHUNG: nền model là Qwen (Trung Quốc) nên hay chèn chữ Hán vào
# câu tiếng Việt. Benchmark 15/07 đo được 32B rò CJK vào ~20% câu trả lời. Khi phát hiện,
# sinh LẠI 1 lần với chỉ thị nghiêm ngặt + nhiệt độ thấp hơn (chép đúng cách grade_essay
# đã làm, nay gom về một chỗ để MỌI feature dùng — không hàm nào quên guard được nữa).
_CJK_STRICT_SUFFIX = (
    "\n\nBẮT BUỘC: viết TOÀN BỘ bằng tiếng Việt có dấu, CHỈ dùng chữ Latinh. "
    "TUYỆT ĐỐI KHÔNG dùng chữ Hán/Trung/Nhật/Hàn hay bất kỳ ký tự CJK nào."
)


def _missing_dependency_error(exc: ModuleNotFoundError) -> RuntimeError:
    return RuntimeError(
        f"Missing Python package '{exc.name}'. Install: python -m pip install -r requirements.txt"
    )


class LLMClient:
    def __init__(self) -> None:
        self.backend = LLM_BACKEND
        # transformers fallback (lazy)
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()

    # ── health ────────────────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        if self.backend == "vllm":
            return True  # model nằm ở pod
        return self._tokenizer is not None and self._model is not None

    # ── low-level completion ──────────────────────────────────────────────────
    def _complete(self, messages: list[dict[str, str]], *, max_tokens: int,
                  temperature: float, top_p: float, guided_json: dict[str, Any] | None,
                  timeout: float) -> str:
        if self.backend == "vllm":
            return self._complete_vllm(messages, max_tokens, temperature, top_p, guided_json, timeout)
        return self._complete_transformers(messages, max_tokens, temperature, top_p)

    def _complete_vllm(self, messages, max_tokens, temperature, top_p, guided_json, timeout) -> str:
        import httpx

        body: dict[str, Any] = {
            "model": VLLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if guided_json is not None:
            # vLLM OpenAI server chấp nhận guided_json (guided_decoding) như field mở rộng.
            body["guided_json"] = guided_json
        url = VLLM_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
        try:
            with httpx.Client(timeout=timeout or VLLM_TIMEOUT) as http:
                resp = http.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"vLLM request failed ({VLLM_BASE_URL}): {exc}") from exc
        return data["choices"][0]["message"]["content"]

    # transformers fallback (dev) ------------------------------------------------
    def _load_transformers(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ModuleNotFoundError as exc:
                raise _missing_dependency_error(exc) from exc

            tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True,
                                                local_files_only=MODEL_LOCAL_FILES_ONLY)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            kwargs: dict[str, Any] = {"trust_remote_code": True,
                                      "local_files_only": MODEL_LOCAL_FILES_ONLY,
                                      "low_cpu_mem_usage": True}
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = torch.bfloat16
            else:
                kwargs["torch_dtype"] = torch.float32
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
            model.eval()
            self._tokenizer, self._model = tok, model

    def _complete_transformers(self, messages, max_tokens, temperature, top_p) -> str:
        self._load_transformers()
        import torch

        tok, model = self._tokenizer, self._model
        if getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages) + "\n\nASSISTANT:\n"
        inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
        do_sample = temperature > 0
        gen: dict[str, Any] = {**inputs, "max_new_tokens": max_tokens, "do_sample": do_sample,
                               "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id}
        if do_sample:
            gen["temperature"] = temperature
            gen["top_p"] = top_p
        with torch.no_grad():
            out = model.generate(**gen)
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # ── high-level API ────────────────────────────────────────────────────────
    def chat(self, messages: list[dict[str, str]], *, task: str = "default",
             max_new_tokens: int | None = None, temperature: float | None = None,
             top_p: float | None = None) -> str:
        limits = task_limits(task)
        return self._complete(
            messages,
            max_tokens=max_new_tokens or limits["think"],
            temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
            top_p=DEFAULT_TOP_P if top_p is None else top_p,
            guided_json=None,
            timeout=limits["timeout"],
        )

    def generate_text(self, task: str, payload: dict[str, Any], *,
                      system_prompt: str | None = None, max_new_tokens: int | None = None,
                      temperature: float | None = None, top_p: float | None = None) -> str:
        base = system_prompt or read_text(PROMPT_PATH)
        messages = [
            {"role": "system", "content": base},
            {"role": "user", "content": _user_block(task, payload)},
        ]
        limits = task_limits(task)
        return self._complete(
            messages,
            max_tokens=max_new_tokens or limits["think"],
            temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
            top_p=DEFAULT_TOP_P if top_p is None else top_p,
            guided_json=None,
            timeout=limits["timeout"],
        )

    def generate_json(self, task: str, payload: dict[str, Any], *,
                      json_schema: dict[str, Any] | None = None,
                      system_prompt: str | None = None, two_stage: bool = True,
                      max_new_tokens: int | None = None, temperature: float | None = None,
                      top_p: float | None = None,
                      cjk_guard_lang: str | None = None) -> dict[str, Any]:
        """Sinh JSON theo 2 giai đoạn (suy luận tự do → ép JSON).

        `cjk_guard_lang="vi"` bật guard rò chữ Hán dùng chung: nếu output có ký tự CJK
        thì sinh LẠI một lần với chỉ thị nghiêm ngặt + nhiệt độ thấp hơn. Bỏ trống thì
        không guard (giữ nguyên hành vi cũ cho các call không phải tiếng Việt).
        """
        result = self._generate_json_once(
            task, payload, json_schema=json_schema, system_prompt=system_prompt,
            two_stage=two_stage, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
        )
        # Guard rò CJK — chỉ chạy thêm 1 lượt khi thật sự phát hiện chữ Hán (≈20% câu).
        if cjk_guard_lang == "vi" and dict_has_cjk(result):
            base = system_prompt or read_text(PROMPT_PATH)
            temp = DEFAULT_TEMPERATURE if temperature is None else temperature
            retry = self._generate_json_once(
                task, payload, json_schema=json_schema,
                system_prompt=base + _CJK_STRICT_SUFFIX, two_stage=two_stage,
                max_new_tokens=max_new_tokens, temperature=max(0.0, temp * 0.5), top_p=top_p,
            )
            # Chỉ nhận bản retry nếu nó SẠCH — bản lỗi cũ vẫn hơn một bản lỗi mới.
            if not dict_has_cjk(retry):
                result = retry
        return result

    def _generate_json_once(self, task: str, payload: dict[str, Any], *,
                            json_schema: dict[str, Any] | None = None,
                            system_prompt: str | None = None, two_stage: bool = True,
                            max_new_tokens: int | None = None, temperature: float | None = None,
                            top_p: float | None = None) -> dict[str, Any]:
        """Một lượt sinh JSON 2 giai đoạn (suy luận tự do → ép JSON)."""
        limits = task_limits(task)
        base = system_prompt or read_text(PROMPT_PATH)
        temp = DEFAULT_TEMPERATURE if temperature is None else temperature
        tp = DEFAULT_TOP_P if top_p is None else top_p

        # Giai đoạn 1 — suy luận tự do
        reason = self._complete(
            [
                {"role": "system", "content": base + _REASON_SUFFIX},
                {"role": "user", "content": _user_block(task, payload)},
            ],
            max_tokens=max_new_tokens or limits["think"],
            temperature=temp, top_p=tp, guided_json=None, timeout=limits["timeout"],
        )
        conclusion = strip_reasoning_blocks(reason)  # bỏ <think>, giữ kết luận

        if not two_stage:
            return parse_json_object(reason)

        # Giai đoạn 2 — ép JSON ở kết luận (guided_decoding nếu có schema)
        fmt_user = (
            f"TASK_ID: {task}\n\nNỘI DUNG CẦN CHUYỂN THÀNH JSON:\n{conclusion}\n\n"
            + (f"JSON phải tuân thủ schema sau:\n{json.dumps(json_schema, ensure_ascii=False)}\n\n"
               if json_schema else "")
            + "Trả về đúng MỘT JSON object hợp lệ, không kèm chữ nào khác."
        )
        text = self._complete(
            [
                {"role": "system", "content": _FORMAT_SYSTEM},
                {"role": "user", "content": fmt_user},
            ],
            max_tokens=limits["json"], temperature=0.0, top_p=1.0,
            guided_json=json_schema, timeout=limits["timeout"],
        )
        return parse_json_object(text)


def _user_block(task: str, payload: dict[str, Any]) -> str:
    return (
        f"TASK_ID: {task}\n\nINPUT_JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


client = LLMClient()
