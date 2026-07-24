"""Lấy mẫu GPU bằng nvidia-smi trong luồng nền.

Đo được: VRAM (idle/peak/incremental), GPU utilization, công suất (W) và từ đó
tính điện năng cho mỗi câu trả lời (Wh) = tích phân công suất theo thời gian.

⚠️ VRAM PHẢI đo theo TIẾN TRÌNH, không lấy `--query-gpu=memory.used`.
   Ở pha A có 2 model + bge dùng CHUNG một card -> memory.used là TỔNG của cả ba.
   Báo cáo cần "model nào ngốn bao nhiêu" nên phải quy về từng PID:
     cổng (8003/8004/8002) -> PID (ss -ltnp) -> VRAM (nvidia-smi --query-compute-apps)
   Công suất và GPU-util thì KHÔNG tách được theo tiến trình (phần cứng chỉ báo mức
   toàn card) -> ở pha A hai model chạy lần lượt chứ không đồng thời, nên số đo trong
   cửa sổ thời gian của model nào thì quy cho model đó; vẫn ghi rõ đây là mức toàn card.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass  # noqa: F401  (Window dùng)

QUERY = "memory.used,utilization.gpu,power.draw"


def sample_once() -> tuple[float, float, float]:
    """(VRAM toàn card MiB, util %, power W)"""
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        text=True, timeout=15).strip().splitlines()[0]
    a, b, c = (x.strip() for x in out.split(","))
    return float(a), float(b), float(c)


def pids_on_port(port: int) -> set[int]:
    """PID đang LISTEN trên cổng. vLLM fork nhiều tiến trình con (EngineCore giữ
    phần lớn VRAM) nên phải lấy cả cây tiến trình, không chỉ mỗi PID của server."""
    try:
        out = subprocess.check_output(["ss", "-ltnp"], text=True, timeout=10)
    except Exception:
        return set()
    roots = set()
    for line in out.splitlines():
        if f":{port} " in line or line.rstrip().endswith(f":{port}"):
            roots.update(int(m) for m in re.findall(r"pid=(\d+)", line))
    if not roots:
        return set()
    # Mở rộng ra toàn bộ con cháu.
    try:
        ps = subprocess.check_output(["ps", "-eo", "pid,ppid"], text=True, timeout=10)
    except Exception:
        return roots
    children: dict[int, list[int]] = {}
    for line in ps.splitlines()[1:]:
        p = line.split()
        if len(p) >= 2 and p[0].isdigit() and p[1].isdigit():
            children.setdefault(int(p[1]), []).append(int(p[0]))
    seen, stack = set(roots), list(roots)
    while stack:
        cur = stack.pop()
        for ch in children.get(cur, []):
            if ch not in seen:
                seen.add(ch); stack.append(ch)
    return seen


def vram_of_pids(pids: set[int]) -> float:
    """Tổng VRAM (MiB) mà tập PID này đang giữ trên GPU."""
    if not pids:
        return 0.0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
    except Exception:
        return 0.0
    tot = 0.0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) in pids:
            try:
                tot += float(parts[1])
            except ValueError:
                pass
    return tot


@dataclass
class Window:
    """Thống kê GPU trong một khoảng thời gian."""
    n: int = 0
    vram_mean: float = 0.0        # TOÀN CARD
    vram_peak: float = 0.0        # TOÀN CARD
    model_vram_mean: float = 0.0  # RIÊNG model đang đo (theo PID)
    model_vram_peak: float = 0.0
    util_mean: float = 0.0
    util_peak: float = 0.0
    power_mean: float = 0.0
    power_peak: float = 0.0
    duration_s: float = 0.0
    energy_wh: float = 0.0


class Monitor:
    def __init__(self, interval_s: float = 1.0):
        self.interval = interval_s
        # (t, vram_toàn_card, util, power, vram_riêng_model)
        self._samples: list[tuple[float, float, float, float, float]] = []
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pids: set[int] = set()

    def track_port(self, port: int) -> set[int]:
        """Khoá theo cây tiến trình của model đang đo. Gọi lại mỗi khi đổi model."""
        self._pids = pids_on_port(port)
        return self._pids

    def start(self):
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                v, u, p = sample_once()
                mv = vram_of_pids(self._pids) if self._pids else 0.0
                with self._lock:
                    self._samples.append((time.perf_counter(), v, u, p, mv))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)

    def mark(self) -> float:
        return time.perf_counter()

    def window(self, t0: float, t1: float, n_answers: int = 1) -> Window:
        with self._lock:
            s = [x for x in self._samples if t0 <= x[0] <= t1]
        w = Window(n=len(s), duration_s=max(t1 - t0, 0.0))
        if not s:
            return w
        vs = [x[1] for x in s]; us = [x[2] for x in s]; ps = [x[3] for x in s]
        mvs = [x[4] for x in s if x[4] > 0]
        w.vram_mean = sum(vs) / len(vs); w.vram_peak = max(vs)
        if mvs:
            w.model_vram_mean = sum(mvs) / len(mvs); w.model_vram_peak = max(mvs)
        w.util_mean = sum(us) / len(us); w.util_peak = max(us)
        w.power_mean = sum(ps) / len(ps); w.power_peak = max(ps)
        # Tích phân hình thang -> Wh, rồi chia cho số câu.
        e = 0.0
        for i in range(1, len(s)):
            dt = s[i][0] - s[i - 1][0]
            e += (s[i][3] + s[i - 1][3]) / 2.0 * dt
        w.energy_wh = e / 3600.0 / max(n_answers, 1)
        return w
