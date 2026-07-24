"""Gọi vLLM qua HTTP streaming và đo hiệu năng ở mức từng token.

Bắt buộc phải STREAM: TTFT (time-to-first-token) không thể đo được nếu chờ cả
response trả về một cục. Với model suy luận, ta còn tách được:
  - thinking time  = từ token đầu đến khi thấy '</think>'
  - answer time    = phần còn lại
=> chứng minh 32B chậm vì nó NGHĨ LÂU, không phải vì code chậm.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

THINK_CLOSE = "</think>"
THINK_OPEN = "<think>"


@dataclass
class CallResult:
    ok: bool = False
    error: str | None = None
    timeout: bool = False
    truncated: bool = False          # finish_reason == 'length' -> bị cắt vì hết max_tokens

    text: str = ""                   # toàn bộ output (kể cả <think>)
    think_text: str = ""
    answer_text: str = ""            # phần sau </think>; model thường = toàn bộ

    ttft_ms: float | None = None     # thời gian tới token ĐẦU TIÊN
    total_latency_s: float = 0.0
    think_time_s: float | None = None    # tới lúc đóng </think>
    answer_time_s: float | None = None   # phần sinh đáp án
    prefill_time_s: float | None = None  # xấp xỉ = TTFT (xử lý prompt + context)
    decode_time_s: float = 0.0           # total - ttft

    prompt_tokens: int = 0
    completion_tokens: int = 0
    think_tokens: int = 0
    answer_tokens: int = 0

    tokens_per_sec: float | None = None  # tốc độ sinh (chỉ tính pha decode)
    tpot_ms: float | None = None         # time-per-output-token trung bình

    def finalize(self) -> "CallResult":
        if self.ttft_ms is not None:
            self.prefill_time_s = self.ttft_ms / 1000.0
            self.decode_time_s = max(self.total_latency_s - self.prefill_time_s, 1e-9)
        if self.completion_tokens > 0 and self.decode_time_s > 0:
            self.tokens_per_sec = self.completion_tokens / self.decode_time_s
            if self.completion_tokens > 1:
                self.tpot_ms = self.decode_time_s * 1000.0 / (self.completion_tokens - 1)
        return self

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        # Text dài -> lưu riêng, không nhét vào bảng số liệu.
        for k in ("text", "think_text", "answer_text"):
            d.pop(k, None)
        return d


def _split_think(text: str) -> tuple[str, str]:
    """Tách phần <think>...</think> khỏi đáp án.

    R1 khi dùng chat template thường KHÔNG in '<think>' mở đầu (template đã thêm sẵn),
    chỉ in nội dung nghĩ rồi '</think>'. Nên bắt theo '</think>' là chính.
    """
    i = text.find(THINK_CLOSE)
    if i == -1:
        return "", text
    think = text[:i].replace(THINK_OPEN, "")
    return think.strip(), text[i + len(THINK_CLOSE):].strip()


def chat_stream(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
    client: httpx.Client | None = None,
) -> CallResult:
    r = CallResult()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": True,
        # Xin usage thật từ vLLM thay vì tự đếm -> số token chính xác cho tính chi phí.
        "stream_options": {"include_usage": True},
    }
    own = client is None
    c = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0))
    t0 = time.perf_counter()
    buf: list[str] = []
    think_closed_at: float | None = None
    try:
        with c.stream("POST", f"{base_url}/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                resp.read()
                r.error = f"HTTP {resp.status_code}: {resp.text[:300]}"
                r.total_latency_s = time.perf_counter() - t0
                return r.finalize()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    u = ev["usage"]
                    r.prompt_tokens = u.get("prompt_tokens", 0) or 0
                    r.completion_tokens = u.get("completion_tokens", 0) or 0
                for ch in ev.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        if r.ttft_ms is None:
                            r.ttft_ms = (time.perf_counter() - t0) * 1000.0
                        buf.append(piece)
                        if think_closed_at is None and THINK_CLOSE in "".join(buf[-8:]):
                            think_closed_at = time.perf_counter()
                    if ch.get("finish_reason") == "length":
                        r.truncated = True
        r.total_latency_s = time.perf_counter() - t0
        r.text = "".join(buf)
        r.think_text, r.answer_text = _split_think(r.text)
        if think_closed_at is not None:
            r.think_time_s = think_closed_at - t0
            r.answer_time_s = max(r.total_latency_s - r.think_time_s, 0.0)
        # Ước lượng token nghĩ/đáp theo tỉ lệ ký tự (vLLM không tách sẵn 2 phần này).
        if r.completion_tokens and r.text:
            ratio = len(r.think_text) / max(len(r.text), 1)
            r.think_tokens = int(round(r.completion_tokens * ratio))
            r.answer_tokens = r.completion_tokens - r.think_tokens
        r.ok = bool(r.text)
        if not r.ok:
            r.error = r.error or "rỗng"
    except httpx.TimeoutException:
        r.timeout = True
        r.error = f"timeout > {timeout_s}s"
        r.total_latency_s = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        r.error = f"{type(e).__name__}: {e}"[:300]
        r.total_latency_s = time.perf_counter() - t0
    finally:
        if own:
            c.close()
    return r.finalize()


def embed(texts: list[str], *, port: int, model: str, timeout_s: float = 120.0) -> list[list[float]]:
    with httpx.Client(timeout=timeout_s) as c:
        resp = c.post(f"http://127.0.0.1:{port}/v1/embeddings",
                      json={"model": model, "input": texts})
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def wait_ready(port: int, timeout_s: float = 2400) -> float:
    """Chờ vLLM sẵn sàng. Trả về số giây -> đây chính là Model Load Time."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            with httpx.Client(timeout=5) as c:
                if c.get(f"http://127.0.0.1:{port}/v1/models").status_code == 200:
                    return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"cổng {port} không sẵn sàng sau {timeout_s}s")
