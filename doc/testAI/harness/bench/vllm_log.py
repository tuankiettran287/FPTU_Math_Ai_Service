"""Bóc số liệu VRAM THẬT từ log khởi động của vLLM.

  python -m bench.vllm_log        -> results/vllm_mem.json

Vì sao cần file này thay vì chỉ đọc nvidia-smi:
  vLLM CẤP PHÁT TRƯỚC toàn bộ ngân sách KV theo --gpu-memory-utilization ngay lúc
  khởi động. Nên `nvidia-smi` luôn báo đúng bằng util × tổng_VRAM, bất kể model to
  hay nhỏ. Đo thật trên pod này:
      Qwen2.5-7B  util 0.30 -> nvidia-smi 30.5GB   (trọng số THẬT chỉ 14.2488 GiB)
      R1-Distill-7B util 0.55 -> nvidia-smi 54.8GB (trọng số THẬT 14.2717 GiB)
  Hai model gần như y hệt nhau về trọng số, nhưng nvidia-smi lệch 24GB — toàn bộ
  chênh lệch là KV cache do NGƯỜI CHẠY cấu hình.
  => Câu "model nào ngốn bao nhiêu VRAM" chỉ trả lời đúng bằng TRỌNG SỐ,
     lấy từ log 'Model loading took X GiB'.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
LOGS = Path("/workspace/logs")

# tên log trên pod -> model key trong bench.config
# Phải khớp tên file `tee` trong serve_phase.sh. Thiếu một dòng ở đây thì model đó
# mất sạch số VRAM trong báo cáo mà không có lỗi nào báo.
LOGMAP = {"qwen": "qwen7b", "r17b": "r1_7b", "llm32": "r1_32b",
          "llama": "llama8b", "r114b": "r1_14b"}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PATTERNS = {
    "weights_gib": re.compile(r"Model loading took\s+([\d.]+)\s*GiB"),
    "load_seconds_engine": re.compile(r"Model loading took\s+[\d.]+\s*GiB and\s+([\d.]+)\s*seconds"),
    "kv_cache_gib": re.compile(r"Available KV cache memory:\s+([\d.]+)\s*GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s+([\d,]+)\s*tokens"),
    "cuda_graph_gib": re.compile(r"Graph capturing finished in \d+ secs, took\s+([\d.]+)\s*GiB"),
}


def parse(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = ANSI.sub("", path.read_text(errors="ignore"))
    out: dict = {}
    for key, rx in PATTERNS.items():
        # Lấy match CUỐI, không phải đầu. Hiện `tee` ghi đè log mỗi lần dựng lại nên
        # mỗi file chỉ có 1 lượt chạy — nhưng nếu ai đó đổi sang `tee -a`, hoặc một
        # lượt chạy hỏng để lại log cũ, thì match đầu sẽ là số của cấu hình CŨ và ta
        # âm thầm báo cáo sai VRAM. Lấy match cuối luôn ứng với lần dựng gần nhất.
        m = None
        for m in rx.finditer(txt):
            pass
        if m:
            v = m.group(1).replace(",", "")
            out[key] = float(v) if "." in v else int(v)
    if "weights_gib" in out and "kv_cache_gib" in out:
        out["total_reserved_gib"] = round(
            out["weights_gib"] + out["kv_cache_gib"] + out.get("cuda_graph_gib", 0.0), 2)
    return out


def main():
    res = {}
    for log_name, key in LOGMAP.items():
        d = parse(LOGS / f"{log_name}.log")
        if d:
            res[key] = d
            print(f"{key:8s} trọng số {d.get('weights_gib')} GiB | "
                  f"KV {d.get('kv_cache_gib')} GiB ({d.get('kv_cache_tokens'):,} token) | "
                  f"graph {d.get('cuda_graph_gib')} GiB | "
                  f"tổng cấp phát {d.get('total_reserved_gib')} GiB")
        else:
            print(f"{key:8s} — chưa có log")
    OUT.mkdir(exist_ok=True)
    (OUT / "vllm_mem.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print("Đã ghi:", OUT / "vllm_mem.json")


if __name__ == "__main__":
    main()
