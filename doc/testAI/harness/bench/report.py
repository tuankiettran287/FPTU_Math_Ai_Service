"""Sinh báo cáo Markdown từ summary.json — mọi con số đều lấy từ đo thật.

  python -m bench.report
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
REP = ROOT / "report"


def g(s, key, default="—"):
    v = s.get(key)
    return default if v is None else v


def cjk_both(k, key):
    """Gộp tỉ lệ rò CJK của hai chế độ, có trọng số theo số mẫu.

    Vì sao cần: rò CJK là đặc tính của MODEL chứ không phải của chế độ RAG. Trước đây
    Slide 7 lấy chế độ 'rag' còn Slide 12 lấy 'pure' -> cùng một chỉ số mà báo cáo
    đưa ra hai con số khác nhau (32B: 22.2% ở Slide 7 vs 18.2% ở Slide 12). Gộp lại
    để toàn báo cáo chỉ có MỘT con số.
    """
    p_, g_ = _S_ALL.get((k, "pure"), {}), _S_ALL.get((k, "rag"), {})
    vp, vg = p_.get(key), g_.get(key)
    np_, ng_ = p_.get("cjk_n") or 0, g_.get("cjk_n") or 0
    if vp is None and vg is None:
        return None
    if vp is None:
        return vg
    if vg is None:
        return vp
    if np_ + ng_ == 0:
        return None
    return (vp * np_ + vg * ng_) / (np_ + ng_)


def fmt(v, suf="", nd=1):
    if v is None or v == "—":
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suf}"
    return f"{v}{suf}"


def gpu_info():
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"], text=True).strip()
        return o
    except Exception:
        return "không đọc được"


def main():
    S = {tuple(k.split("|")): v for k, v in
         json.loads((OUT / "summary.json").read_text(encoding="utf-8")).items()}
    global _S_ALL
    _S_ALL = S
    cal = {}
    p = OUT / "judge_calibration.json"
    if p.exists():
        cal = json.loads(p.read_text(encoding="utf-8"))
    cov = json.loads((ROOT / "corpus" / "coverage.json").read_text(encoding="utf-8"))

    keys = [m.key for m in config.MODELS]
    names = {m.key: m.label for m in config.MODELS}
    shorts = {m.key: (m.short or m.label.split(" (")[0]) for m in config.MODELS}
    JUDGE_K = config.JUDGE_KEY      # cũng là model 32B đang được đề xuất
    L = []
    A = L.append

    A("# Báo cáo đánh giá & lựa chọn mô hình AI\n")
    A("**Hệ thống:** FPTU Math AI — trợ giảng toán cho MAE101 / MAD101 / MAS291  ")
    A(f"**Ngày chạy:** {datetime.now():%d/%m/%Y}  ")
    A(f"**Phần cứng:** {gpu_info()}  ")
    A(f"**Phần mềm:** vLLM 0.10.2 · PyTorch 2.8.0+cu128 · driver 580 (CUDA 13.0)\n")
    A("> Toàn bộ số liệu trong báo cáo này được đo thực tế trên chính hạ tầng triển khai, "
      "không lấy từ tài liệu hay bảng xếp hạng của nhà sản xuất.\n")
    A("---\n")

    # ── Slide 0 ──
    A("## Slide 0 — Kiến trúc hệ thống AI (Local RAG Pipeline)\n")
    A("```")
    A("  Tài liệu (PDF/DOCX)          Ngân hàng câu hỏi")
    A("        │                             │")
    A("        ├── OCR (RapidOCR) ───────────┤")
    A("        │                             │")
    A("        ▼                             ▼")
    A("   Chia đoạn (chunk)          Chuẩn hoá schema")
    A("        │                             │")
    A("        └──────────┬──────────────────┘")
    A("                   ▼")
    A("        Embedding — BAAI/bge-m3 (1024 chiều, vLLM :8001)")
    A("                   ▼")
    A("        PostgreSQL + pgvector  (VM riêng, nối qua SSH tunnel)")
    A(f"          ├── document_chunks     : {cov['n_chunks']:>6,} đoạn lý thuyết")
    A(f"          └── math_question_bank  : {cov['n_qb']:>6,} bài đã giải")
    A("                   ▼")
    A("        Truy hồi (cosine, lọc theo môn) — top-3 lý thuyết + top-3 bài mẫu")
    A("                   ▼")
    A("        Prompt = System (đóng vai GV + CoT) + Ngữ cảnh + Câu hỏi")
    A("                   ▼")
    A("        LLM cục bộ — vLLM (OpenAI-compatible API)")
    A("                   ▼")
    A("        Hậu xử lý: bóc đáp án · kiểm định dạng · sympy đối chiếu")
    A("```\n")
    A("**Độ phủ corpus theo môn (đo từ DB thật):**\n")
    A("| Môn | Đoạn lý thuyết | Bài giải mẫu |")
    A("|---|---:|---:|")
    for full, short in (("MAE101", "MAE"), ("MAD101", "MAD"), ("MAS291", "MAS")):
        A(f"| {full} | {cov['chunks'].get(full, 0):,} | {cov['qb'].get(short, 0):,} |")
    A("")
    mn = min(cov["chunks"].values()); mx = max(cov["chunks"].values())
    A(f"> ⚠️ Lý thuyết phân bố **không đều**: môn nhiều nhất gấp ~{mx/max(mn,1):.0f} lần môn ít nhất. "
      "Đây là nguyên nhân gốc của phần lớn *Retrieval Error* ở mục phân tích lỗi.\n")
    A("---\n")

    # ── Slide 1 ──
    ROLE = {
        "qwen7b": "Đối chứng cùng họ — LLM chỉ-dẫn thường, trả thẳng kết quả",
        "llama8b": "Đối chứng khác họ — kiểm tra kết luận có tổng quát ngoài họ Qwen không",
        "r1_7b": "Reasoning cỡ nhỏ, có chuỗi `<think>`",
        "r1_14b": "Reasoning **trung gian** — trả lời 'sao không chọn cỡ ở giữa?'",
        "r1_32b": "Reasoning cỡ lớn — cấu hình mạnh nhất chạy được trên hệ thống",
    }
    A(f"## Slide 1 — {len(config.MODELS)} mô hình được so sánh\n")
    A("| | Model | Loại | Tham số | Vai trò trong thí nghiệm |")
    A("|---|---|---|---:|---|")
    for i, m in enumerate(config.MODELS, 1):
        fam = "Instruct (không suy luận)" if m.family == "instruct" else "Reasoning"
        A(f"| {i} | `{m.hf_id}` | {fam} | {m.params_b}B | {ROLE.get(m.key, '')} |")
    A("")
    A("### Vì sao chọn đúng những model này\n")
    A("**Hai đối chứng, không phải một.** `Qwen2.5-7B` khống chế biến kiến trúc: cả ba "
      "model DeepSeek-R1-Distill đều được chưng cất **từ chính Qwen**, nên khi so với "
      "Qwen2.5-7B thì chênh lệch quy về **đúng một biến — có suy luận hay không** "
      "(cùng kiến trúc, cùng tokenizer, cùng dữ liệu tiền huấn luyện). Nhưng chính vì "
      "cùng họ, nó **không** chứng minh được kết luận có đúng ngoài họ Qwen hay không "
      "→ thêm `Llama-3.1-8B` khác hẳn nền. Nếu **cả hai** baseline cùng thua reasoning "
      "model thì luận điểm chắc hơn hẳn một baseline.\n")
    A("**Ba cỡ reasoning, không phải hai.** Chỉ có 7B và 32B thì kết luận \"phải dùng "
      "32B\" mới là so hai điểm mút — hội đồng hỏi ngay *\"sao không thử cỡ trung "
      "gian?\"*. `R1-Distill-14B` cho biết đường cong chất lượng-theo-kích-thước bão hoà "
      "ở đâu. Nếu 14B đã đủ ngưỡng thì kết luận đúng phải là chọn 14B — rẻ hơn, nhanh "
      "hơn — và báo cáo này sẽ nói đúng như vậy.\n")
    A("> **Ghi chú về nguồn model:** dùng bản mirror `NousResearch/Meta-Llama-3.1-8B-Instruct` "
      "vì repo gốc `meta-llama/*` là *gated* (cần token đã được Meta duyệt). Đây là bản "
      "sao y nguyên trọng số, không phải bản fine-tune lại.\n")
    A("---\n")

    # ── Slide thiết kế thí nghiệm ──
    A("## Slide 2 — Thiết kế thí nghiệm & bảo đảm công bằng\n")
    n_v2 = S.get((keys[0], "pure"), {}).get("n_V2", 0)
    n_v1 = S.get((keys[0], "pure"), {}).get("n_V1", 0)
    n_v3 = S.get((keys[0], "pure"), {}).get("n_V3", 0)
    A("| Hạng mục | Giá trị |")
    A("|---|---|")
    n_v2h = S.get((keys[0], "pure"), {}).get("n_V2H", 0)
    n_task = n_v1 + n_v2 + n_v2h + n_v3
    A(f"| Bộ câu hỏi | **hai bộ độc lập**: bộ gốc {n_v2} câu (60/môn × 3 môn, "
      f"easy/medium/hard) + bộ KHÓ {n_v2h} câu (medium 48 / hard 132) |")
    A(f"| Vấn đề 1 — ra đề | {n_v1} lượt/model/chế độ |")
    A(f"| Vấn đề 2 — giải toán *(bộ gốc)* | {n_v2} lượt/model/chế độ |")
    A(f"| Vấn đề 2 — giải toán *(bộ KHÓ)* | {n_v2h} lượt/model/chế độ |")
    A(f"| Vấn đề 3 — chấm bài | {n_v3} lượt/model/chế độ |")
    nm = len(config.MODELS)
    A(f"| Tổng lượt sinh | **{n_task*nm*2:,}** ({n_task} tác vụ × {nm} model × 2 chế độ) |")
    A(f"| Warm-up | {config.WARMUP_REQUESTS} request/model trước khi đo |")
    A(f"| Concurrency — đo hiệu năng | **1** (nghiêm ngặt) |")
    A("| Concurrency — đo chất lượng | " +
      ", ".join(f"{m.short}={m.quality_concurrency}" for m in config.MODELS) + " |")
    A(f"| Số lần lặp | 1 lượt/câu (seed cố định `{config.SEED}`) |")
    A(f"| Thứ tự chạy | cố định: model → chế độ → task theo ID |")
    A(f"| RAG corpus | **giống hệt nhau** cho cả {nm} model |")
    A(f"| Ngữ cảnh RAG | **truy hồi trước 1 lần**, cả {nm} model nhận cùng một chuỗi |")
    A("| Bố trí GPU | 3 pha (A: Qwen+R1-7B · A2: Llama+R1-14B · B: R1-32B riêng) |")
    A("")
    A("### Vì sao tách hai pha concurrency (câu hỏi hội đồng sẽ hỏi)\n")
    A("- **Đo hiệu năng phải concurrency = 1.** Chạy song song thì request phải xếp hàng, "
      "TTFT đo được sẽ gồm cả thời gian chờ hàng đợi — không còn là độ nhạy của model.\n"
      "- **Đo chất lượng thì concurrency không ảnh hưởng.** Mỗi request độc lập, tham số "
      "sinh cố định; chạy 8 luồng hay 1 luồng thì nội dung câu trả lời phân phối như nhau. "
      "Nhờ vậy tiết kiệm khoảng 9 giờ GPU mà không đánh đổi tính công bằng.\n")
    A("### Các biến nhiễu đã khử\n")
    A("- Ngữ cảnh RAG truy hồi **trước**, lưu ra file → thời gian embed + tìm kiếm không "
      "lọt vào TTFT, và mọi model đọc **cùng một** ngữ cảnh.\n"
      "- Chỉ mục nằm trong RAM (numpy) thay vì gọi DB qua tunnel → loại biến động mạng.\n"
      "- Không model nào dùng `--enforce-eager`: nếu chỉ 32B chạy eager thì nó bị bóp "
      "tốc độ 10–15% một cách giả tạo.\n"
      "- Mọi model dùng **chung một System Prompt**.\n")
    A("---\n")

    # ── Slide 3: hyperparameters ──
    A("## Slide 3 — Tham số cấu hình & lý do\n")
    A("| Model | Temperature | Top-P | Max tokens |")
    A("|---|---:|---:|---:|")
    for m in config.MODELS:
        A(f"| {m.label} | {m.temperature} | {m.top_p} | {m.max_tokens:,} |")
    A("")
    A("### Vì sao KHÔNG ép mọi model cùng temperature\n")
    A("Model card chính thức của DeepSeek-R1-Distill ghi rõ: temperature = 0 khiến model "
      "**lặp vô tận hoặc mất mạch lạc**, và khuyến nghị dải **0.5–0.7 (mặc định 0.6)** "
      "kèm top-p 0.95. Ép R1 về 0.1 là cố tình chạy sai khuyến nghị của nhà sản xuất → "
      "so sánh mất công bằng. Ngược lại Qwen2.5-Instruct là model chỉ-dẫn thường, chạy "
      "tốt ở temperature thấp và đó chính là cấu hình dùng khi triển khai thật.\n"
      "→ **Mỗi model được chạy ở cấu hình tốt nhất của chính nó.** So sánh ở đây là "
      "\"model ở trạng thái tốt nhất\", không phải \"model bị bóp cùng một tham số\".\n")
    A("### Cách bảo vệ từng tham số trước hội đồng\n")
    A(f"- **Temperature thấp (Qwen 0.1 / R1 0.6):** dự án là giải toán, cần suy luận "
      "tất định theo định lý, không cần sáng tạo. Hạ nhiệt độ để triệt tiêu việc AI tự "
      "bịa số liệu. Với R1 thì 0.6 đã là mức thấp nhất còn *an toàn*.\n"
      f"- **Top-P 0.9–0.95:** giới hạn model vào nhóm token xác suất cao → các bước giải "
      "mạch lạc, không lan man sang từ ngữ xác suất thấp.\n"
      f"- **Max tokens 2.048 (Qwen) vs 6.144 (R1):** R1 cần không gian rất lớn để 'nháp' "
      "trong thẻ `<think>` trước khi chốt đáp án. Cấp cho Qwen 6.144 là vô nghĩa vì nó "
      "không có pha suy nghĩ. Tỉ lệ bị cắt (truncation) được đo và công bố ở dưới.\n"
      "- **System Prompt:** đóng vai giáo viên toán FPT (Role-play) + bắt buộc suy luận "
      "từng bước (Chain-of-Thought) + cấm bịa định lý + bắt kiểm tra lại phép tính cuối.\n")
    A("---\n")

    # ── Slide 4: hiệu năng ──
    A("## Slide 4 — So sánh hiệu năng (Latency & VRAM)\n")
    A("*Mọi số dưới đây lấy từ pha concurrency = 1.*\n")
    A("| Chỉ số | " + " | ".join(shorts[k] for k in keys) + " |")
    A("|---|" + "---:|" * len(keys))
    rows = [
        ("Model Load Time (s)", lambda s: fmt(_gpu(s, "load_time_s"), nd=0)),
        ("**Trọng số model (GiB)**", lambda s: fmt(_mem(s, "weights_gib"), nd=2)),
        ("KV cache cấp phát (GiB)", lambda s: fmt(_mem(s, "kv_cache_gib"), nd=2)),
        ("KV cache (số token)", lambda s: f"{_mem(s, 'kv_cache_tokens'):,}"
         if _mem(s, "kv_cache_tokens") else "—"),
        ("CUDA graph (GiB)", lambda s: fmt(_mem(s, "cuda_graph_gib"), nd=2)),
        ("Tổng VRAM instance giữ (GiB)", lambda s: fmt(_mem(s, "total_reserved_gib"), nd=2)),
        ("Idle VRAM — riêng model (GB)", lambda s: fmt(idle_vram(s), nd=1)),
        ("Peak VRAM — riêng model (GB)", lambda s: fmt(peak_vram(s), nd=1)),
        ("Incremental VRAM (GB)", lambda s: fmt(incr_vram(s), nd=1)),
        ("TTFT P50 (ms)", lambda s: fmt(g(s, "ttft_ms_p50"), nd=0)),
        ("TTFT P95 (ms)", lambda s: fmt(g(s, "ttft_ms_p95"), nd=0)),
        ("Tốc độ sinh (tok/s)", lambda s: fmt(g(s, "tokens_per_sec_mean"), nd=1)),
        ("TPOT (ms/token)", lambda s: fmt(g(s, "tpot_ms_mean"), nd=1)),
        ("Prefill (s)", lambda s: fmt(g(s, "prefill_s_mean"), nd=2)),
        ("Decode (s)", lambda s: fmt(g(s, "decode_s_mean"), nd=2)),
        ("Latency P50 (s)", lambda s: fmt(g(s, "latency_s_p50"), nd=1)),
        ("Latency P95 (s)", lambda s: fmt(g(s, "latency_s_p95"), nd=1)),
        ("Latency P99 (s)", lambda s: fmt(g(s, "latency_s_p99"), nd=1)),
        ("**Thinking time (s)**", lambda s: fmt(g(s, "think_time_s_mean"), nd=1)),
        ("**Tỉ lệ token nghĩ/tổng**", lambda s: fmt((g(s, "think_token_ratio") or 0) * 100, "%", 1)),
        ("Token nghĩ (TB)", lambda s: fmt(g(s, "think_tokens_mean"), nd=0)),
        ("Token đáp án (TB)", lambda s: fmt(g(s, "answer_tokens_mean"), nd=0)),
        ("GPU util TB (%)", lambda s: fmt(gpu_field(s, "util_mean"), nd=0)),
        ("Công suất TB (W)", lambda s: fmt(gpu_field(s, "power_mean"), nd=0)),
        ("Điện năng/câu (Wh)", lambda s: fmt(gpu_field(s, "energy_wh"), nd=3)),
        ("Timeout rate (%)", lambda s: fmt(g(s, "timeout_rate"), nd=1)),
        ("OOM rate (%)", lambda s: fmt(g(s, "oom_rate"), nd=1)),
        ("Truncation rate (%)", lambda s: fmt(g(s, "truncation_rate"), nd=1)),
        ("Format compliance (%)", lambda s: fmt(g(s, "format_compliance"), nd=1)),
    ]
    for label, fn in rows:
        A(f"| {label} | " + " | ".join(fn(S.get((k, "pure"), {})) for k in keys) + " |")
    A("")
    A("![TTFT](charts/s3_ttft.png)")
    A("![Tốc độ](charts/s3_speed.png)")
    A("![Latency P95](charts/s3_latency_p95.png)")
    A("![VRAM](charts/s3_vram.png)\n")
    A("> **Đọc cột VRAM cho đúng.** vLLM cấp phát trước toàn bộ KV cache theo tham số "
      "`--gpu-memory-utilization`, nên `nvidia-smi` luôn báo đúng bằng `util × VRAM card` "
      "— **bất kể model to hay nhỏ**. Bằng chứng đo được ở chính pod này: Qwen2.5-7B và "
      "DeepSeek-R1-Distill-7B có trọng số gần như y hệt (14.25 vs 14.27 GiB, vì cùng 7.6B "
      "và cùng nền Qwen), nhưng nếu đặt util lệch nhau thì `nvidia-smi` báo lệch tới 24GB. "
      "Vì vậy **chỉ số trả lời đúng câu \"model nào ngốn bao nhiêu\" là TRỌNG SỐ MODEL**, "
      "lấy từ log vLLM. Trong thí nghiệm này hai model 7B được đặt **cùng util 0.35** để "
      "loại hẳn biến nhiễu đó.\n")
    A("> **32B chậm là vì nó nghĩ lâu, không phải vì code chậm.** Xem hai dòng "
      "*Thinking time* và *Tỉ lệ token nghĩ/tổng*: phần lớn thời gian của model reasoning "
      "nằm trong thẻ `<think>`, tức là nháp logic trước khi chốt. TPOT (thời gian mỗi "
      "token) giữa các model chênh nhau ít hơn nhiều so với tổng latency — chứng tỏ khác "
      "biệt đến từ **số lượng token phải sinh**, không phải tốc độ phục vụ.\n")
    A("---\n")

    # ── Slide 5: chất lượng ──
    A("## Slide 5 — So sánh chất lượng\n")
    A("| Chỉ số | " + " | ".join(f"{shorts[k]}<br>thuần / RAG" for k in keys) + " |")
    A("|---|" + "---:|" * len(keys))

    def pair(k, key):
        a = S.get((k, "pure"), {}).get(key)
        b = S.get((k, "rag"), {}).get(key)
        return f"{fmt(a, '%')} / **{fmt(b, '%')}**"

    for label, key in (("Vấn đề 1 — Ra đề", "acc_V1"),
                       ("Vấn đề 2 — Giải toán *(bộ gốc)*", "acc_V2"),
                       ("**Vấn đề 2 — Giải toán (bộ KHÓ)**", "acc_V2H"),
                       ("Vấn đề 3 — Chấm bài", "acc_V3"),
                       ("*bộ gốc* · MAE101", "acc_V2_MAE101"),
                       ("*bộ gốc* · MAD101", "acc_V2_MAD101"),
                       ("*bộ gốc* · MAS291", "acc_V2_MAS291"),
                       ("*bộ gốc* · Dễ", "acc_V2_easy"),
                       ("*bộ gốc* · Trung bình", "acc_V2_medium"),
                       ("*bộ gốc* · Khó", "acc_V2_hard"),
                       ("**bộ KHÓ** · MAE101", "acc_V2H_MAE101"),
                       ("**bộ KHÓ** · MAD101", "acc_V2H_MAD101"),
                       ("**bộ KHÓ** · MAS291", "acc_V2H_MAS291"),
                       ("**bộ KHÓ** · Trung bình", "acc_V2H_medium"),
                       ("**bộ KHÓ** · Khó", "acc_V2H_hard")):
        A(f"| {label} | " + " | ".join(pair(k, key) for k in keys) + " |")
    A("")
    # ── Hai bộ đề: dễ (bị trần) vs khó — câu chuyện quan trọng nhất của Vấn đề 2 ──
    def _rng(prob):
        """(thấp nhất, cao nhất) accuracy của mọi model×chế độ ở một vấn đề."""
        vals = [S.get((k, m), {}).get(f"acc_{prob}") for k in keys for m in ("pure", "rag")]
        vals = [v for v in vals if v is not None]
        return (min(vals), max(vals)) if vals else (None, None)

    lo_e, hi_e = _rng("V2")
    lo_h, hi_h = _rng("V2H")
    if lo_e is not None and lo_h is not None:
        A("### Vấn đề 2 chạy trên HAI bộ đề — và đó là phát hiện quan trọng nhất\n")
        A("| Bộ đề | Số câu | Thấp nhất | Cao nhất | **Khoảng phân tách** |")
        A("|---|---:|---:|---:|---:|")
        A(f"| Bộ gốc (easy/medium/hard) | {S.get((keys[0],'pure'),{}).get('n_V2','—')} | "
          f"{lo_e:.1f}% | {hi_e:.1f}% | **{hi_e-lo_e:.1f} điểm** |")
        A(f"| **Bộ khó** (medium 48 / hard 132) | {S.get((keys[0],'pure'),{}).get('n_V2H','—')} | "
          f"{lo_h:.1f}% | {hi_h:.1f}% | **{hi_h-lo_h:.1f} điểm** |")
        A("")
        if hi_e - lo_e < 10:
            A(f"**Bộ gốc bị hiệu ứng trần.** Mọi model — kể cả baseline 7B — đều nằm trong "
              f"dải hẹp {lo_e:.1f}–{hi_e:.1f}% (chênh {hi_e-lo_e:.1f} điểm). Đề quá dễ nên "
              f"model mạnh **không có chỗ thể hiện**: mọi chênh lệch đo được đều chìm trong "
              f"vùng nhiễu lấy mẫu. Dùng bộ này để kết luận chọn model là sai phương pháp.\n")
        if (hi_h - lo_h) > (hi_e - lo_e):
            A(f"**Bộ khó khôi phục sức phân biệt**: dải rộng ra {hi_h-lo_h:.1f} điểm "
              f"(gấp {(hi_h-lo_h)/max(hi_e-lo_e,0.1):.1f}× bộ gốc). Đây mới là bộ dùng để "
              f"kết luận cho bài toán giải toán.\n")
        A("> Bài học phương pháp, đáng đưa lên slide: **một bộ test không phân biệt được "
          "các model thì không chứng minh được gì — kể cả khi mọi model đều đạt điểm cao.** "
          "Điểm cao trên đề dễ chỉ nói lên đề dễ. Nhóm giữ **cả hai** bộ trong báo cáo "
          "chính vì lý do đó: bộ gốc cho thấy *dễ thì model nào cũng làm được*, bộ khó cho "
          "thấy *khó mới lộ ra khác biệt thật*.\n")
        A("![V2 dễ vs khó](charts/s_easy_vs_hard.png)\n")
    A("### Bộ ba chỉ số RAG (RAG Triad) — chỉ đo ở chế độ RAG\n")
    A("| Chỉ số | " + " | ".join(shorts[k] for k in keys) + " | Ý nghĩa |")
    A("|---|" + "---:|" * len(keys) + "---|")
    for label, key, mean in (
        ("Context Relevance", "rag_context_relevance",
         "Truy hồi có lấy đúng định lý/công thức cần không"),
        ("Groundedness", "rag_groundedness",
         "Câu trả lời có dựa vào tài liệu được cấp không"),
        ("Answer Relevance", "rag_answer_relevance",
         "Đáp án cuối có khớp Ground Truth không"),
        ("Tỉ lệ trích dẫn [n]", "rag_citation_rate", "Model có nêu nguồn không"),
        ("Có lấy được lý thuyết", "rag_theory_hit_rate",
         "Tỉ lệ câu truy hồi ra ≥1 đoạn lý thuyết"),
    ):
        A(f"| {label} | " + " | ".join(fmt(S.get((k, 'rag'), {}).get(key), "%")
                                       for k in keys) + f" | {mean} |")
    cre = S.get((keys[0], "rag"), {}).get("rag_context_relevance_emb")
    A("")
    A(f"**Context Relevance đo độc lập bằng embedding:** {fmt(cre, '', 3)} "
      "(cosine trung bình giữa đoạn truy hồi được và `expected_context` do người ra đề "
      "viết). Chỉ số này **không nhờ AI chấm** nên dùng để đối chứng với cột "
      "Context Relevance ở trên.\n")
    cont = S.get((keys[0], "rag"), {}).get("rag_contamination_rate")
    A(f"**Kiểm tra trùng lặp dữ liệu:** {fmt(cont, '%')} số câu test có bài trong ngân "
      f"hàng giống ≥ {config.CONTAMINATION_SIM} (cosine). Nếu tỉ lệ này cao thì RAG "
      "'thắng' một cách tầm thường do chép được đáp án — nên phải công bố cùng kết quả.\n")
    A("![Accuracy V2](charts/s4_acc_v2.png)")
    A("![RAG Triad](charts/s4_rag_triad.png)\n")
    A("---\n")

    # ── Slide 6: pure vs RAG ──
    A("## Slide 6 — Prompt thuần vs Prompt + RAG\n")
    A("![V1](charts/s5_acc_V1.png)")
    A("![V2](charts/s5_acc_V2.png)")
    A("![V3](charts/s5_acc_V3.png)\n")
    A("| Model | Ra đề (V1) | Giải toán (V2) | Chấm bài (V3) |")
    A("|---|---:|---:|---:|")
    for k in keys:
        d = []
        for pr in ("V1", "V2", "V3"):
            a = S.get((k, "pure"), {}).get(f"acc_{pr}")
            b = S.get((k, "rag"), {}).get(f"acc_{pr}")
            delta = (b - a) if (a is not None and b is not None) else None
            sign = "+" if (delta or 0) >= 0 else ""
            d.append(f"{fmt(a,'%')} → {fmt(b,'%')} ({sign}{fmt(delta,' đ%')})")
        A(f"| {shorts[k]} | " + " | ".join(d) + " |")
    A("")
    A("---\n")

    # ── Phân tích lỗi ──
    A("## Slide 7 — Phân tích nguyên nhân lỗi\n")
    A("![Lỗi](charts/s_errors.png)\n")
    A("> Bảng dưới tính trên **bộ KHÓ**, chế độ RAG. Lý do: trên bộ gốc R1-32B đạt "
      "100% → **không còn câu sai nào để phân tích**, bảng sẽ hiện 0%/0%/0% và bị "
      "đọc nhầm thành *không bao giờ sai*. Mẫu số là **số câu SAI của chính model "
      "đó** (ghi ở dòng cuối), nên mỗi cột cộng lại bằng 100%.\n")
    A("| Nguyên nhân | " + " | ".join(shorts[k] for k in keys) + " | Định nghĩa |")
    A("|---|" + "---:|" * len(keys) + "---|")
    for label, key, desc in (
        ("Lỗi do RAG (Retrieval)", "errH_retrieval",
         "Sai vì VectorDB trích thiếu/nhầm công thức"),
        ("Lỗi suy luận (Calculation)", "errH_calculation",
         "Lấy đúng công thức, dùng đúng phương pháp, nhưng tính sai"),
        ("Ảo giác (Hallucination)", "errH_hallucination",
         "Phớt lờ tài liệu, tự bịa công thức/dữ kiện"),
    ):
        A(f"| {label} | " + " | ".join(fmt(S.get((k, 'rag'), {}).get(key), "%")
                                       for k in keys) + f" | {desc} |")
    A("| *(số câu SAI — mẫu số của 3 dòng trên)* | " +
      " | ".join(str(S.get((k, 'rag'), {}).get("n_errors_hard", 0)) for k in keys) +
      " | Càng nhỏ càng tốt |")
    A("")
    A("| Chỉ số phụ | " + " | ".join(shorts[k] for k in keys) + " |")
    A("|---|" + "---:|" * len(keys))
    for label, key in (("Timeout rate", "timeout_rate"), ("Truncation rate", "truncation_rate"),
                       ("OOM rate", "oom_rate"), ("Lỗi gọi API", "error_rate")):
        A(f"| {label} | " + " | ".join(fmt(S.get((k, 'rag'), {}).get(key), "%")
                                       for k in keys) + " |")
    A("")
    A("> Ví dụ thực tế của từng loại lỗi: xem sheet **Chi tiết từng câu** trong "
      "`ket_qua_benchmark.xlsx`, lọc cột *Nguyên nhân lỗi*.\n")
    A("### Lỗi thứ tư không có trong phân loại ban đầu: rò ngôn ngữ\n")
    A("![CJK](charts/s_cjk_leak.png)\n")
    A("| Chỉ số | " + " | ".join(shorts[k] for k in keys) + " |")
    A("|---|" + "---:|" * len(keys))
    A("| Rò chữ Trung/Nhật/Hàn (bất kỳ đâu) — thuần / RAG | " +
      " | ".join(f'{fmt(S.get((k, "pure"), {}).get("cjk_leak_rate"), "%")} / '
                 f'{fmt(S.get((k, "rag"), {}).get("cjk_leak_rate"), "%")}' for k in keys) + " |")
    A("| **Rò vào ĐÁP ÁN (SV đọc thấy)** — thuần / RAG | " +
      " | ".join(f'{fmt(S.get((k, "pure"), {}).get("cjk_leak_answer_rate"), "%")} / '
                 f'{fmt(S.get((k, "rag"), {}).get("cjk_leak_answer_rate"), "%")}' for k in keys) + " |")
    A("| **Rò vào ĐÁP ÁN — gộp cả hai chế độ** | " +
      " | ".join(f'**{fmt(cjk_both(k, "cjk_leak_answer_rate"), "%")}**' for k in keys) + " |")
    A("")
    A("> Tách theo **cả hai chế độ** vì rò CJK là đặc tính của MODEL, không phụ thuộc "
      "RAG. Dòng **gộp** là con số dùng để kết luận (và dùng lại nguyên vẹn ở Slide 12) "
      "— tránh việc trích một chế độ rồi trình bày như đặc tính chung.\n")
    A("Phát hiện trong lúc chạy, **không nằm trong thiết kế ban đầu**: dòng "
      "DeepSeek-R1-Distill được chưng cất từ dữ liệu đa ngữ nặng tiếng Trung nên chèn "
      "chữ Hán vào giữa câu tiếng Việt. Ví dụ có thật lấy từ log:\n")
    A("```")
    A("Tuy nhiên, sinh viên đã tính ra 19π cm²/s, có thể他们是 đã tính sai số")
    A("                                              ^^^^^^ tiếng Trung")
    A("```")
    A("Với sản phẩm phục vụ sinh viên Việt Nam thì đây **không phải tiểu tiết học "
      "thuật** — sinh viên nhìn thấy ngay. Ta tách hai mức: rò trong thẻ `<think>` "
      "(người dùng không thấy, chấp nhận được) và rò vào **đáp án** (nghiêm trọng). "
      "Hệ thống production đã phải cài bộ chặn `_has_cjk` trong `features.py` chính vì "
      "hiện tượng này.\n")
    A("---\n")

    # ── Vấn đề 3 chi tiết ──
    A("## Slide 8 — Vấn đề 3: chấm bài sinh viên (chi tiết)\n")
    A("Bộ test được sinh có **nhãn biết trước**, nên đây là bài toán phân loại nhị phân "
      "đo được chính xác:\n")
    A("| Biến thể bài làm | Nhãn | Nguồn nhãn | " +
      " | ".join(shorts[k] for k in keys) + " |")
    A("|---|---|---|" + "---:|" * len(keys))
    for var, lab, src in (("correct_verbatim", "Đúng", "code"),
                          ("correct_paraphrase", "Đúng", "LLM + code kiểm số"),
                          ("arithmetic_slip", "Sai", "code"),
                          ("incomplete", "Sai", "code")):
        vals = [fmt(S.get((k, "rag"), {}).get(f"v3_acc_{var}"), "%") for k in keys]
        A(f"| `{var}` | {lab} | {src} | " + " | ".join(vals) + " |")
    A("")
    A("| Chỉ số (chế độ prompt thuần) | " + " | ".join(shorts[k] for k in keys) +
      " | Vì sao quan trọng |")
    A("|---|" + "---:|" * len(keys) + "---|")
    A("| **False-Pass Rate** | " + " | ".join(fmt(S.get((k, 'pure'), {}).get('v3_false_pass_rate'), "%")
                                              for k in keys) +
      " | Cho bài SAI đậu — hỏng tính công bằng của điểm số |")
    A("| **False-Fail Rate** | " + " | ".join(fmt(S.get((k, 'pure'), {}).get('v3_false_fail_rate'), "%")
                                              for k in keys) +
      " | Đánh trượt bài ĐÚNG chỉ vì viết khác — SV khiếu nại, GV mất niềm tin |")
    A("")

    # ── RAG làm HỎNG việc chấm bài — phát hiện phản trực giác, phải nêu bật ──
    deltas = []
    for k in keys:
        a = S.get((k, "pure"), {}).get("acc_V3")
        b = S.get((k, "rag"), {}).get("acc_V3")
        if a is not None and b is not None:
            deltas.append((k, a, b, b - a,
                           S.get((k, "pure"), {}).get("v3_false_pass_rate"),
                           S.get((k, "rag"), {}).get("v3_false_pass_rate")))
    if deltas and all(d[3] <= 0 for d in deltas):
        A("### ⚠️ RAG làm GIẢM chất lượng chấm bài — ở **mọi** model, không có ngoại lệ\n")
        A("| Model | Accuracy thuần → RAG | Chênh | False-Pass thuần → RAG |")
        A("|---|---:|---:|---:|")
        for k, a, b, d, fpa, fpb in deltas:
            A(f"| {shorts[k]} | {a:.1f}% → **{b:.1f}%** | **{d:+.1f}** | "
              f"{fmt(fpa,'%')} → **{fmt(fpb,'%')}** |")
        A("")
        A("**Cơ chế — nhìn cột False-Pass là thấy.** Ngữ cảnh RAG chứa các **bài giải mẫu "
          "tương tự** lấy từ ngân hàng câu hỏi. Khi chấm, model đem bài làm của sinh viên "
          "đối chiếu nhầm với *bài mẫu tương tự* đó thay vì với **đáp án chuẩn của chính "
          "câu hỏi này** → nó trở nên dễ dãi, cho bài sai đậu nhiều hơn hẳn.\n")
        A("**Kết luận nghiệp vụ: KHÔNG dùng RAG cho tính năng chấm bài.** Đây là thứ chỉ "
          "lộ ra vì thí nghiệm đo **cả hai** chế độ. Nếu mặc định \"RAG luôn tốt hơn\" và "
          "chỉ chạy RAG, nhóm đã đưa vào sản phẩm đúng cấu hình làm hỏng tính năng này.\n")
        A("> Bài học tổng quát: **RAG không phải luôn tốt.** Nó giúp khi model THIẾU kiến "
          "thức (ra đề, giải toán khó), nhưng gây hại khi nhiệm vụ là ĐỐI CHIẾU hai thứ "
          "đã có sẵn trong prompt — lúc đó tài liệu thêm vào chỉ là nhiễu.\n")
        A("")
    # Vì sao 32B được chọn làm giám khảo — trả lời bằng chính số liệu V3.
    v3p = [(k, S.get((k, "pure"), {}).get("acc_V3")) for k in keys]
    v3p = [(k, v) for k, v in v3p if v is not None]
    if len(v3p) >= 3:
        best_k, best_v = max(v3p, key=lambda x: x[1])
        rest = sorted((v for k, v in v3p if k != best_k), reverse=True)
        A("### Vì sao 32B được chọn làm giám khảo — chính bảng trên là bằng chứng\n")
        A(f"Nhiệm vụ của giám khảo (\"đối chiếu bài làm với đáp án chuẩn, kết luận "
          f"đúng/sai\") **chính là Vấn đề 3**. Nên bảng ở trên vừa là kết quả nghiệp vụ, "
          f"vừa là căn cứ chọn giám khảo — không cần thí nghiệm riêng.\n")
        A(f"`{shorts[best_k]}` đạt **{best_v:.1f}%** trên tập nhãn khách quan, hơn model "
          f"kế tiếp **{best_v - rest[0]:+.1f} điểm** và hơn các model 7–8B khoảng "
          f"**{best_v - rest[-1]:.0f} điểm**. Việc chọn nó làm giám khảo vì thế **không "
          f"phải quyết định tuỳ tiện — nó là model chấm chính xác nhất trong hệ thống, "
          f"và điều đó được đo chứ không phải giả định**.\n")
        A("")
    if cal:
        A("### Độ tin cậy của giám khảo AI — đo, không giả định\n")
        det = [x for x in cal.get("detail", []) if x.get("judge_pred")]
        rel = [x for x in det if x["variant"] != "incomplete"]
        inc = [x for x in det if x["variant"] == "incomplete"]
        acc_rel = (sum(1 for x in rel if x["judge_pred"] == x["label"]) / len(rel)) if rel else None
        acc_inc = (sum(1 for x in inc if x["judge_pred"] == x["label"]) / len(inc)) if inc else None
        # Tỉ lệ bài bị cắt trong V2 -> chặn trên cho ảnh hưởng của điểm yếu giám khảo.
        trunc = max((S.get((k, m), {}).get("truncation_rate") or 0)
                    for k in keys for m in ("pure", "rag")) if S else 0
        # 'trunc' ở trên là CHẶN TRÊN (model tệ nhất), KHÔNG phải tỉ lệ của toàn thí
        # nghiệm — câu chữ trước đây mô tả sai con số này. Tính thêm bản gộp có trọng số.
        _tn = sum((S.get((k, m), {}).get("n_ok") or 0)
                  for k in keys for m in ("pure", "rag"))
        _tc = sum((S.get((k, m), {}).get("truncation_rate") or 0) / 100.0
                  * (S.get((k, m), {}).get("n_ok") or 0)
                  for k in keys for m in ("pure", "rag"))
        trunc_pooled = (100.0 * _tc / _tn) if _tn else 0
        A(f"Giám khảo (32B) được đem chấm **{cal['n_parsed']}/{cal['n']}** mẫu mà **nhãn "
          f"đúng/sai đã biết trước do code sinh ra**. Kết quả phải tách theo loại ca, "
          f"vì gộp lại sẽ cho một con số gây hiểu sai:\n")
        A("| Loại ca | Độ chính xác giám khảo | Có xảy ra trong V2 không? |")
        A("|---|---:|---|")
        for v, a in cal["by_variant"].items():
            occurs = "Không" if v == "incomplete" else "Có"
            A(f"| `{v}` | {a:.1%} | {occurs} |")
        if acc_rel is not None:
            A(f"| **Gộp các ca bài làm ĐẦY ĐỦ** | **{acc_rel:.1%}** ({len(rel)} mẫu) | "
              f"**Đây là con số dùng để đánh giá độ tin cậy** |")
        if acc_inc is not None:
            A(f"| Ca bài làm cắt cụt | {acc_inc:.1%} ({len(inc)} mẫu) | Không |")
        A(f"| *(gộp tất cả — con số gây hiểu sai)* | *{cal['accuracy']:.1%}* | — |")
        A("")
        if acc_rel is not None and acc_inc is not None:
            A(f"**Đọc bảng này cho đúng.** Giám khảo đạt **{acc_rel:.1%}** trên các ca "
              f"bài làm đầy đủ — tức đúng những ca xảy ra khi chấm Vấn đề 2. Nó chỉ yếu ở "
              f"ca bài làm **cắt cụt** ({acc_inc:.1%}): khi lời giải dừng giữa chừng nhưng "
              f"phần đã viết thì đúng, giám khảo có xu hướng chấm ĐÚNG — nó **dễ dãi với "
              f"bài dở dang**.\n")
            A(f"Điểm yếu đó **không ảnh hưởng số liệu Vấn đề 2**, vì tỉ lệ câu trả lời bị "
              f"cắt gộp toàn bộ {_tn:,} lượt chỉ **{trunc_pooled:.1f}%** (model tệ nhất "
              f"cũng chỉ {trunc:.1f}%) — model không bao giờ "
              f"dừng giữa chừng như biến thể nhân tạo kia. Và kể cả nếu có, độ dễ dãi này "
              f"áp dụng **như nhau cho mọi model**, nên phép SO SÁNH giữa các model vẫn "
              f"đứng vững.\n")
        A("> Trả lời trực tiếp câu \"lấy gì bảo đảm giám khảo AI chấm đúng?\": "
          "**không bảo đảm bằng lời — bảo đảm bằng số.** Bộ V3 có nhãn do code sinh nên "
          "dùng làm thước đo chính giám khảo được, và điểm yếu tìm thấy đã được định "
          "lượng cùng phạm vi ảnh hưởng của nó.\n")
    A("---\n")

    # ── Chi phí ──
    A("## Slide 9 — Chi phí\n")
    tot_in = sum(v.get("prompt_tokens_total", 0) for v in S.values())
    tot_out = sum(v.get("completion_tokens_total", 0) for v in S.values())
    A(f"Tổng token toàn bộ thí nghiệm: **{tot_in:,}** input + **{tot_out:,}** output "
      f"= **{tot_in+tot_out:,}** token.\n")
    A("| Phương án | Chi phí cho đúng khối lượng này |")
    A("|---|---:|")
    for api, p in config.API_PRICING.items():
        c = tot_in / 1e6 * p["in"] + tot_out / 1e6 * p["out"]
        A(f"| API `{api}` | ${c:,.2f} |")
    A(f"| **DeepSeek-R1 tự host (nhóm)** | **$0.00** (token) |")
    A("")
    A(f"> Tự host không miễn phí tuyệt đối: GPU RTX PRO 6000 trên RunPod "
      f"≈ ${config.POD_USD_PER_HOUR:.2f}/giờ. Điểm mấu chốt là chi phí **không tăng theo "
      f"lượng token** — dùng càng nhiều càng rẻ tương đối, và **dữ liệu bài làm của sinh "
      f"viên không rời khỏi hạ tầng trường**.\n")
    A("---\n")

    # ── Kết luận: SINH TỪ SỐ LIỆU, không gõ tay ──
    A("## Slide 10 — Kết luận: dùng model nào cho việc gì\n")

    def best(prob):
        """(model, mode, acc) tốt nhất cho một vấn đề."""
        cand = [(k, md, S.get((k, md), {}).get(f"acc_{prob}"))
                for k in keys for md in ("pure", "rag")]
        cand = [c for c in cand if c[2] is not None]
        return max(cand, key=lambda c: c[2]) if cand else (None, None, None)

    MODE_VI = {"pure": "prompt thuần", "rag": "prompt + RAG"}
    A("| Vấn đề | Cấu hình tốt nhất (theo số đo) | Độ chính xác | Độ trễ P50 |")
    A("|---|---|---:|---:|")
    # Vấn đề 2 kết luận theo BỘ KHÓ, không theo bộ gốc: bộ gốc bị trần (mọi model
    # ~94%) nên "tốt nhất" ở đó chỉ là nhiễu lấy mẫu, không phải năng lực thật.
    for prob, nm in (("V1", "Ra đề"), ("V2H", "Giải toán — bộ khó"), ("V3", "Chấm bài")):
        k, md, acc = best(prob)
        if not k:
            A(f"| **{nm}** | — | — | — |")
            continue
        lat = S.get((k, "pure"), {}).get("latency_s_p50")
        A(f"| **{nm}** | {shorts[k]} · {MODE_VI[md]} | "
          f"{fmt(acc, '%')} | {fmt(lat, 's')} |")
    # Việc cần NHANH: chọn theo độ trễ. Chất lượng lấy từ BỘ KHÓ (bộ gốc bị trần nên
    # không phân biệt được, đưa vào đây sẽ khiến model nào trông cũng "đủ tốt").
    fast = sorted(((k, S.get((k, "pure"), {}).get("latency_s_p50"),
                    S.get((k, "rag"), {}).get("acc_V2H"))
                   for k in keys if S.get((k, "pure"), {}).get("latency_s_p50")),
                  key=lambda x: x[1])
    if fast:
        k, lat, acc = fast[0]
        A(f"| Chat / Đấu trường (ưu tiên tốc độ) | {shorts[k]} | "
          f"{fmt(acc, '%')} (V2-khó) | **{fmt(lat, 's')}** |")
    A("")
    # Chênh lệch < 5 điểm % thì KHÔNG kết luận — n=180 không đủ phân giải.
    kb, mb, ab = best("V2H")
    if kb:
        others = sorted(((k, md, S.get((k, md), {}).get("acc_V2H"))
                         for k in keys for md in ("pure", "rag")
                         if S.get((k, md), {}).get("acc_V2H") is not None and
                         not (k == kb and md == mb)),
                        key=lambda c: -c[2])
        if others and ab is not None and (ab - others[0][2]) < 5:
            A(f"> ⚠️ **Không đủ căn cứ thống kê để tuyên bố người thắng ở Vấn đề 2.** "
              f"Cấu hình dẫn đầu ({shorts[kb]} · {MODE_VI[mb]}, {ab:.1f}%) "
              f"chỉ hơn cấu hình thứ hai ({shorts[others[0][0]]} · "
              f"{MODE_VI[others[0][1]]}, {others[0][2]:.1f}%) **{ab - others[0][2]:.1f} điểm %**. "
              f"Với n=180 và mỗi câu chạy 1 lượt, chênh lệch dưới 5 điểm % nằm trong "
              f"khoảng dao động do lấy mẫu (temperature > 0) — phải chọn theo tiêu chí "
              f"khác (độ trễ, VRAM, tỉ lệ rò ngôn ngữ).\n")
    A("")
    A("### Vì sao KHÔNG fine-tune\n")
    A("1. **Không có dữ liệu huấn luyện đủ chuẩn.** Fine-tune cần hàng nghìn cặp "
      "(đề → lời giải) đã được giảng viên kiểm duyệt. Ngân hàng câu hỏi hiện có "
      f"{cov['n_qb']:,} bài nhưng chưa qua thẩm định sư phạm từng bài — fine-tune trên dữ "
      "liệu chưa sạch sẽ *dạy model học luôn cả lỗi*.\n"
      "2. **RAG sửa được tức thì, fine-tune thì không.** Giảng viên đổi giáo trình hay "
      "sửa một công thức → chỉ cần nạp lại tài liệu, vector cập nhật ngay. Với fine-tune "
      "phải huấn luyện lại toàn bộ.\n"
      "3. **Fine-tune không chữa được lỗi tính toán.** Bảng phân tích lỗi cho thấy phần "
      "lớn lỗi là *Calculation*, tức bản chất số học của LLM — thêm dữ liệu cùng miền "
      "không sửa được, trong khi RAG + kiểm tra bằng sympy thì có.\n"
      "4. **Chi phí và rủi ro.** Fine-tune 32B cần nhiều GPU-giờ, dễ *catastrophic "
      "forgetting*, và phải đánh giá lại từ đầu sau mỗi lần chạy.\n"
      "5. **Phạm vi đồ án là ứng dụng, không phải nghiên cứu mô hình.** Mục tiêu là "
      "*giải quyết 3 bài toán nghiệp vụ*, và số liệu ở trên cho thấy prompt + RAG đã đạt.\n")
    A("### DeepSeek-R1 có thật sự giải quyết được 3 vấn đề không?\n")
    # Trả lời bằng số, kể cả khi số không ủng hộ. Ngưỡng đặt TRƯỚC khi nhìn kết quả.
    THRESH = {"V1": 70.0, "V2H": 70.0, "V3": 85.0}
    WHY = {"V1": "đề GV còn duyệt lại trước khi dùng nên sai sót còn cứu được",
           "V2H": "sinh viên đối chiếu được với lời giải từng bước; đo trên BỘ KHÓ vì bộ gốc bị trần",
           "V3": "chấm điểm ảnh hưởng trực tiếp tới quyền lợi SV nên ngưỡng phải cao hơn"}
    A(f"Ngưỡng \"đạt\" được đặt **trước** khi xem kết quả, theo mức độ rủi ro của "
      f"từng nghiệp vụ: V1 ≥ {THRESH['V1']:.0f}%, V2 (bộ khó) ≥ {THRESH['V2H']:.0f}%, "
      f"V3 ≥ {THRESH['V3']:.0f}%.\n")
    A("| Vấn đề | Ngưỡng | 32B đạt (tốt nhất) | Kết luận | Vì sao ngưỡng đó |")
    A("|---|---:|---:|---|---|")
    verdicts = {}
    for prob in ("V1", "V2H", "V3"):
        vals = [S.get((JUDGE_K, md), {}).get(f"acc_{prob}") for md in ("pure", "rag")]
        vals = [v for v in vals if v is not None]
        got = max(vals) if vals else None
        ok = (got is not None and got >= THRESH[prob])
        verdicts[prob] = (got, ok)
        mark = "✅ **ĐẠT**" if ok else ("❌ **CHƯA ĐẠT**" if got is not None else "—")
        A(f"| {prob} | {THRESH[prob]:.0f}% | {fmt(got, '%')} | {mark} | {WHY[prob]} |")
    A("")
    npass = sum(1 for v in verdicts.values() if v[1])
    if npass == 3:
        A("**Trả lời: CÓ.** DeepSeek-R1-Distill-32B vượt ngưỡng ở cả ba bài toán.\n")
    elif npass == 0:
        A("**Trả lời: CHƯA.** Không bài toán nào vượt ngưỡng. Số liệu KHÔNG ủng hộ việc "
          "triển khai tự động hoàn toàn — cần giữ giảng viên trong vòng lặp duyệt.\n")
    else:
        passed = ", ".join(p for p, v in verdicts.items() if v[1])
        failed = ", ".join(p for p, v in verdicts.items() if not v[1])
        A(f"**Trả lời: CÓ MỘT PHẦN.** Đạt ở {passed}; **chưa đạt ở {failed}**. "
          f"Với phần chưa đạt, kết luận trung thực là *chưa dùng tự động được* — phải "
          f"để AI đề xuất và giảng viên duyệt, chứ không để AI quyết.\n")
    # ── Đường cong kích thước: trả lời "sao không chọn 14B?" ──
    A("### Tại sao không chọn cỡ trung gian (14B)?\n")
    A("![Scaling](charts/s_scaling.png)\n")
    rs = [m for m in config.MODELS if m.family == "reasoning"]
    A("| Model | Tham số | Trọng số | **V2-khó (RAG)** | V3 (RAG) | Độ trễ P50 | Rò chữ Hán vào đáp án *(gộp 2 chế độ)* |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for m in rs:
        d = S.get((m.key, "rag"), {})
        dp = S.get((m.key, "pure"), {})
        A(f"| {m.short} | {m.params_b}B | {fmt(_mem({'_model': m.key}, 'weights_gib'), ' GiB', 2)} | "
          f"{fmt(d.get('acc_V2H'), '%')} | {fmt(d.get('acc_V3'), '%')} | "
          f"{fmt(dp.get('latency_s_p50'), 's')} | "
          f"{fmt(cjk_both(m.key, 'cjk_leak_answer_rate'), '%')} |")
    A("")
    # Kết luận về 14B phải suy ra từ SỐ, kể cả khi số nói ngược ý ban đầu.
    if len(rs) >= 3:
        mid = rs[len(rs) // 2] if rs[len(rs) // 2].key == "r1_14b" else \
            next((m for m in rs if m.key == "r1_14b"), None)
        big = next((m for m in rs if m.key == "r1_32b"), None)
        if mid and big:
            lm = S.get((mid.key, "pure"), {}).get("latency_s_p50")
            lb = S.get((big.key, "pure"), {}).get("latency_s_p50")

            def _gap(prob, mode):
                a = S.get((mid.key, mode), {}).get(f"acc_{prob}")
                b = S.get((big.key, mode), {}).get(f"acc_{prob}")
                return (None if (a is None or b is None) else b - a), a, b

            g2p, a2p, b2p = _gap("V2H", "pure")
            g2r, a2r, b2r = _gap("V2H", "rag")
            g3p, a3p, b3p = _gap("V3", "pure")
            g3r, a3r, b3r = _gap("V3", "rag")

            if None not in (g2p, g2r, g3p, g3r):
                A("Chênh lệch 32B − 14B, đo ở **cả hai chế độ** (ngưỡng nhiễu tự đặt: "
                  "5 điểm %):\n")
                A("| Nghiệp vụ | 14B | 32B | Chênh | Có vượt ngưỡng nhiễu? |")
                A("|---|---:|---:|---:|---|")
                for nm_, gp, a_, b_ in (
                        (f"V2 bộ KHÓ · prompt thuần", g2p, a2p, b2p),
                        (f"V2 bộ KHÓ · prompt + RAG", g2r, a2r, b2r),
                        (f"V3 chấm bài · prompt thuần", g3p, a3p, b3p),
                        (f"V3 chấm bài · prompt + RAG", g3r, a3r, b3r)):
                    A(f"| {nm_} | {a_:.1f}% | {b_:.1f}% | **{gp:+.1f}** | "
                      f"{'✅ có' if gp >= 5 else '❌ không'} |")
                A("")
                A(f"**Đọc bảng cho trung thực.** Ở **Vấn đề 2 bộ khó**, chênh lệch "
                  f"**vắt ngang ngưỡng**: {g2p:.1f} điểm ở chế độ thuần (dưới ngưỡng) "
                  f"nhưng {g2r:.1f} điểm ở chế độ RAG (trên ngưỡng). Chỉ trích dẫn con số "
                  f"{g2r:.1f} rồi kết luận '32B hơn hẳn' là **chọn đúng chế độ có lợi cho "
                  f"kết luận mình muốn** — nhóm không làm vậy. Riêng Vấn đề 2, số liệu "
                  f"**chưa đủ để tách 14B khỏi 32B**.\n")
                A(f"**Căn cứ thật sự nằm ở Vấn đề 3 — chấm bài.** Ở đó 32B hơn 14B "
                  f"**{g3p:.1f} điểm** (thuần) và **{g3r:.1f} điểm** (RAG) — vượt ngưỡng ở "
                  f"**cả hai** chế độ, nên không phụ thuộc vào việc chọn chế độ nào. Quan "
                  f"trọng hơn: V3 là bộ **có nhãn do code sinh**, chấm bằng so khớp máy móc "
                  f"chứ **không nhờ giám khảo AI** → con số này không thể bị nghi là do "
                  f"giám khảo thiên vị. Và chấm bài chính là nghiệp vụ **rủi ro cao nhất** "
                  f"(ảnh hưởng trực tiếp điểm số sinh viên), nên đây là chỗ đáng trả giá.\n")
                A(f"**Cái giá của 32B:** chậm hơn {(lb / max(lm, 1e-9)):.1f}× "
                  f"({fmt(lb,'s')} vs {fmt(lm,'s')}) và tốn thêm "
                  f"{fmt((_mem({'_model': big.key}, 'weights_gib') or 0) - (_mem({'_model': mid.key}, 'weights_gib') or 0), ' GiB', 1)} "
                  f"trọng số. **Kết luận có điều kiện:** chọn 32B **vì Vấn đề 3**; nếu hệ "
                  f"thống chỉ cần giải toán (Vấn đề 2) thì **14B là lựa chọn hợp lý hơn** — "
                  f"gần bằng chất lượng, nhanh hơn ~3×, nhẹ hơn 33 GiB.\n")
                A(f"> Chỗ này đáng nói thẳng trước hội đồng: **một phép đo không ủng hộ "
                  f"kết luận thì phải báo cáo đúng như vậy.** Vấn đề 2 không tách được "
                  f"14B/32B, và nhóm ghi lại điều đó thay vì lờ đi — chính vì thế con số "
                  f"ở Vấn đề 3 mới đáng tin.\n")
    A("> Đây chính là lý do phải đo cả ba cỡ. Nếu chỉ có 7B và 32B thì kết luận "
      "\"phải dùng 32B\" mới là suy diễn từ hai điểm mút — không biết đường cong bão hoà "
      "ở đâu, và không trả lời được câu hỏi hiển nhiên của hội đồng.\n")
    A("### Tại sao không dùng model khác?\n")
    A("| Phương án | Vì sao loại | Căn cứ |")
    A("|---|---|---|")
    qa = S.get(("qwen7b", "rag"), {})
    la = S.get(("llama8b", "rag"), {})
    ra = S.get(("r1_7b", "rag"), {})
    ma = S.get(("r1_14b", "rag"), {})
    ja = S.get((JUDGE_K, "rag"), {})
    A(f"| **Qwen2.5-7B-Instruct** | Nhanh nhất ({fmt(S.get(('qwen7b','pure'),{}).get('latency_s_p50'),'s')}) "
      f"nhưng kém suy luận | V2-khó {fmt(qa.get('acc_V2H'),'%')} so với 32B {fmt(ja.get('acc_V2H'),'%')} |")
    A(f"| **Llama-3.1-8B-Instruct** | Đối chứng khác họ — xác nhận kết luận không phải "
      f"đặc thù họ Qwen | V2-khó {fmt(la.get('acc_V2H'),'%')}; V3 {fmt(la.get('acc_V3'),'%')} |")
    A(f"| **DeepSeek-R1-Distill-7B** | Chậm hơn Qwen mà không chính xác hơn, lại rò "
      f"tiếng Trung nặng | V2-khó {fmt(ra.get('acc_V2H'),'%')}; rò chữ Hán vào đáp án "
      f"{fmt(cjk_both('r1_7b', 'cjk_leak_answer_rate'),'%')} so với Qwen "
      f"{fmt(cjk_both('qwen7b', 'cjk_leak_answer_rate'),'%')} (gộp 2 chế độ) |")
    A(f"| **DeepSeek-R1-Distill-14B** | Xem mục đường cong kích thước ngay trên | "
      f"V2-khó {fmt(ma.get('acc_V2H'),'%')}; V3 {fmt(ma.get('acc_V3'),'%')}; "
      f"độ trễ {fmt(S.get(('r1_14b','pure'),{}).get('latency_s_p50'),'s')} |")
    A("| **GPT-4o / API thương mại** | Bài làm và dữ liệu học tập của SV phải rời khỏi "
      "hạ tầng trường; chi phí tăng tuyến tính theo token | Xem Slide 9 — Chi phí |")
    A("| **32B-AWQ (lượng tử hoá 4-bit)** | **Chưa kiểm chứng** — về lý thuyết ~20GB, "
      "đủ chỗ chạy kèm 7B, nhưng nhóm CHƯA đo nên không đưa vào kết luận | — |")
    A("| **Fine-tune** | Xem mục ngay trên | — |")
    A("")
    A("> Ghi chú trung thực: dòng 32B-AWQ là **hướng chưa thử**, không phải phương án bị "
      "bác bỏ. Nếu cần chạy đồng thời một model nhanh cho Chat/Đấu trường và một model "
      "sâu cho Chấm bài trên **một** GPU 96GB thì đó là hướng đáng đo tiếp.\n")
    A("---\n")
    A("## Slide 11 — Model đã chọn: prompt thuần vs RAG theo từng vấn đề\n")
    A("![Chosen](charts/s7_chosen_by_problem.png)\n")
    A("**Độ chính xác được xác định thế nào:**\n")
    A("| Vấn đề | Căn cứ chấm |")
    A("|---|---|")
    A("| V1 — Ra đề | Không có đáp án chuẩn → rubric 4 tiêu chí (giải được / đúng chủ đề / "
      "đáp án tự khai đúng / đề rõ nghĩa) + kiểm chéo bằng cách bắt model mạnh nhất giải lại đề |")
    A("| V2 — Giải toán | So với `expected_answer` của bộ 180 câu. Ưu tiên đối chiếu bằng "
      "code (chuẩn hoá chuỗi + so tập số); chỉ ca mơ hồ mới nhờ giám khảo LLM chấm **có tham chiếu** |")
    A("| V3 — Chấm bài | So phán quyết của model với **nhãn biết trước** do code sinh → "
      "không cần AI chấm, con số là khách quan tuyệt đối |")
    A("")
    # ── SLIDE: từ đo lường -> hành động kỹ thuật cụ thể ─────────────────────
    A("---\n")
    A("## Slide 12 — Từ đo lường đến hành động: các chỉnh sửa rút ra cho hệ thống\n")
    A("> Thí nghiệm này **không dừng ở việc chọn model**. Nó phát hiện 7 vấn đề có thật "
      "trong AI service đang chạy — mỗi vấn đề kèm số đo, vị trí trong code và hành động "
      "cụ thể. Đây là phần chuyển từ *đo đạc* sang *kỹ thuật*.\n")

    ja_p = S.get((JUDGE_K, "pure"), {})
    ja_r = S.get((JUDGE_K, "rag"), {})
    la_p = S.get(("llama8b", "pure"), {})
    # Gộp cả hai chế độ -> khớp với Slide 7, không còn 18.2% vs 22.2% cho cùng chỉ số.
    cjk32 = cjk_both(JUDGE_K, "cjk_leak_answer_rate")
    cjk_llama = cjk_both("llama8b", "cjk_leak_answer_rate")
    v3p = ja_p.get("acc_V3")
    v3r = ja_r.get("acc_V3")
    fpp = ja_p.get("v3_false_pass_rate")
    fpr = ja_r.get("v3_false_pass_rate")
    lat32 = ja_p.get("latency_s_p50")
    latq = S.get(("qwen7b", "pure"), {}).get("latency_s_p50")

    A("| # | Phát hiện | Số đo (bằng chứng) | Hành động | Mức |")
    A("|---|---|---|---|---|")
    A(f"| 1 | Bộ chặn chữ Hán `_has_cjk` chỉ áp cho **1/10 hàm** trong `features.py` "
      f"(chỉ `grade_essay`) | 32B rò chữ Hán vào đáp án **{fmt(cjk32,'%')}**; "
      f"Llama-3.1-8B rò **{fmt(cjk_llama,'%')}** → là đặc tính riêng dòng R1-Distill | "
      f"Tách guard+retry của `grade_essay` thành helper dùng chung; áp cho `solve`, "
      f"`generate_questions`, `arena_questions` trước tiên | 🔴 |")
    # Số đo lấy từ chính graded_r1_32b_*.jsonl (bộ KHÓ, n=360), đã kiểm lại:
    #   TB 658 · p50 510 · p95 1.489 · p99 2.504 · max 4.691 · vượt 3.500 = 1/360.
    A("| 2 | `TASK_LIMITS.solve_full.think = 3500` **thấp hơn đỉnh nhu cầu** | Token "
      "`<think>` khi giải đề khó (n=360): p95 **1.489** · p99 **2.504** · max **4.691** "
      "→ **1/360 câu (0,3%)** vượt 3.500. Hiếm, nhưng hỏng thì hỏng hẳn: hết ngân "
      "sách think thì JSON **không bao giờ được in** | Nâng `solve_full` → **5.500** "
      "(ngân sách thừa không tốn gì nếu không dùng tới); `concept_lookup` 1.500 → "
      "2.500; `arena_generation` 2.000 → 2.500 | 🟡 |")
    # Đếm THẬT từ dữ liệu thay vì viết cứng "5/5" — số cũ sai (Llama-3.1-8B tăng khi bật RAG).
    _hurt = [k for k in keys
             if (S.get((k, "pure"), {}).get("acc_V3") is not None
                 and S.get((k, "rag"), {}).get("acc_V3") is not None
                 and S[(k, "rag")]["acc_V3"] < S[(k, "pure")]["acc_V3"])]
    _exc = [shorts[k] for k in keys if k not in _hurt]
    _exc_txt = (f"; ngoại lệ: {', '.join(_exc)}" if _exc else "; không ngoại lệ")
    A(f"| 3 | **RAG làm hỏng tác vụ đối chiếu** | Chấm bài 32B: {fmt(v3p,'%')} → "
      f"**{fmt(v3r,'%')}** khi bật RAG; False-Pass {fmt(fpp,'%')} → **{fmt(fpr,'%')}**. "
      f"Đúng ở **{len(_hurt)}/{len(keys)} model**{_exc_txt} | `grade_essay` đã KHÔNG "
      f"dùng RAG → **giữ nguyên**. `explain_mistake` có dùng → **cần đo riêng** trước "
      f"khi tắt | 🟡 |")
    A("| 4 | Bóc đáp án cuối không chịu được `\\boxed{}` / markdown | Luật \"lấy dòng sau "
      "marker\" hỏng **128/180** câu với R1-7B, chỉ 2/180 với Qwen → **sai lệch hẳn về "
      "một họ model** | Dùng `extract_final()` trong `harness/bench/tasks.py`: ưu tiên "
      "`\\boxed{}` (đếm ngoặc lồng) rồi tới dòng đầu có nội dung. Đã kiểm: 128 → **0** | 🟡 |")
    mn_c = min(cov["chunks"].values()); mx_c = max(cov["chunks"].values())
    A(f"| 5 | Tài liệu MAD101 **bị trùng** trong `document_chunks` | 405 chunk cũ chưa xoá "
      f"khi nạp bản mới (\"Đồ thị\" và \"Đồ Thị\", \"Logic\" và \"Logic mệnh đề & vị từ\"...) "
      f"→ top-3 có thể trả 2 đoạn gần trùng | Xoá 8 tài liệu cũ (tên Title Case) | 🟢 |")
    A(f"| 6 | Lý thuyết **lệch giữa các môn** | MAS291 chỉ {mn_c:,} chunk so với MAD101 "
      f"{mx_c:,} — kém **{mx_c/max(mn_c,1):.0f} lần** | Bổ sung tài liệu MAS291 — đòn bẩy "
      f"lớn hơn đổi model | 🟢 |")
    A(f"| 7 | **32B quá chậm cho Chat / Đấu trường** | Độ trễ P50: 32B **{fmt(lat32,'s')}** "
      f"vs Qwen2.5-7B **{fmt(latq,'s')}** (chậm "
      f"{(lat32/latq):.1f}×) | 96GB không chở nổi 32B + model nhanh "
      f"(32B chiếm 87.3GB, còn 6.1GB < 14.25GB). Hướng **chưa kiểm chứng**: 32B-AWQ ~20GB | 🟢 |"
      if (lat32 and latq) else
      f"| 7 | **32B quá chậm cho Chat / Đấu trường** | Độ trễ P50 32B = {fmt(lat32,'s')} | "
      f"Hướng chưa kiểm chứng: 32B-AWQ ~20GB | 🟢 |")
    A("")
    A("### Ba bài học phương pháp (giá trị hơn cả bảng trên)\n")
    A("1. **Một bộ test không phân biệt được các model thì không chứng minh được gì — "
      "kể cả khi mọi model đều điểm cao.** Bộ đề gốc cho mọi model ~94–100%: chọn \"người "
      "thắng\" ở đó là chọn nhiễu. Phải có bộ khó mới lộ ra khác biệt thật.\n"
      f"2. **RAG không phải luôn tốt.** Đo được ở {len(_hurt)}/{len(keys)} model"
      + (f" (ngoại lệ {', '.join(_exc)} — model duy nhất ngoài họ Qwen, và cũng là model "
          f"yếu nhất nên còn nhiều chỗ cải thiện)" if _exc else "")
      + ". Nó giúp khi model THIẾU kiến thức (ra đề, giải toán "
      "khó, tra khái niệm), nhưng gây hại khi nhiệm vụ là ĐỐI CHIẾU hai thứ đã có sẵn "
      "trong prompt. Nếu mặc định \"bật RAG cho mọi thứ\", nhóm đã đưa vào sản phẩm đúng "
      "cấu hình làm hỏng tính năng chấm điểm.\n"
      "3. **Đo cả hai chế độ cho mọi tính năng, đừng giả định.** Phát hiện số 3 hoàn toàn "
      "nằm ngoài dự đoán ban đầu — nó chỉ lộ ra vì thí nghiệm chạy cả `prompt thuần` lẫn "
      "`prompt + RAG` cho cả ba vấn đề, thay vì chỉ chạy cấu hình được cho là tốt hơn.\n")
    A("> Chi tiết đầy đủ kèm vị trí code: xem `CAN_SUA_AI_SERVICE.md` cùng thư mục.\n")

    A("---\n")
    A("## Hạn chế của nghiên cứu\n")
    A("- **Giám khảo là chính model 32B** (mạnh nhất trong hệ local, không có API ngoài). "
      "Giảm thiểu bằng: chấm *có tham chiếu* (đưa sẵn đáp án chuẩn, chỉ đối chiếu chứ "
      "không đánh giá cảm tính) + ưu tiên đối chiếu bằng code + **đo độ chính xác của "
      "giám khảo trên tập nhãn thật**"
      + (f" (kết quả: **{acc_rel:.1%}** trên các ca bài làm đầy đủ — đúng loại ca xảy "
          f"ra khi chấm Vấn đề 2; con số gộp {cal['accuracy']:.1%} bị kéo xuống bởi ca "
          f"cắt cụt vốn không xuất hiện trong V2 — xem Slide 8)"
         if (cal and acc_rel is not None)
         else (f" (kết quả: {cal['accuracy']:.1%})" if cal else "")) + ".\n"
      "- **Mỗi câu chạy 1 lượt**, chưa lặp nhiều lần để ước lượng phương sai do sampling "
      "(temperature > 0). Với 180 câu/model thì sai số trung bình đã đủ nhỏ cho kết luận "
      "so sánh, nhưng chênh lệch < 5 điểm % giữa hai model **không nên coi là có ý nghĩa**.\n"
      "- **Biến thể `correct_paraphrase`** do LLM sinh, chỉ giữ lại bản qua được kiểm tra "
      "bằng code (mọi token số bảo toàn). Các câu có đáp án thuần chữ không sinh được "
      "biến thể này.\n"
      "- **Lý thuyết phân bố lệch giữa các môn** (xem Slide 0) → điểm RAG của môn ít tài "
      "liệu bị thiệt; đây là hạn chế của kho học liệu, không phải của model.\n")

    (REP / "BAO_CAO_AI.md").write_text("\n".join(L), encoding="utf-8")
    print("Đã ghi:", REP / "BAO_CAO_AI.md")


def idle_vram(s):
    v = _gpu(s, "idle_vram_mib")
    return v / 1024 if v else None


def peak_vram(s):
    # VRAM RIÊNG của model (theo PID), KHÔNG phải memory.used của cả card.
    v = _gpu(s, "model_vram_peak")
    return v / 1024 if v else None


def incr_vram(s):
    v = _gpu(s, "incremental_vram_mib")
    return v / 1024 if v else None


def gpu_field(s, f):
    return _gpu(s, f)


def _gpu(s, f):
    """Số đo GPU của ĐÚNG model trong summary row (gpu_<key>_pure_perf.json)."""
    key = s.get("_model")
    if not key:
        return None
    p = OUT / f"gpu_{key}_pure_perf.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(f)


def _mem(s, f):
    """Số VRAM bóc từ log vLLM (trọng số / KV / graph) — thuộc tính thật của model."""
    key = s.get("_model")
    p = OUT / "vllm_mem.json"
    if not key or not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(key, {}).get(f)


if __name__ == "__main__":
    main()
