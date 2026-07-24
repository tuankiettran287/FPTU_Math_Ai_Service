"""Cấu hình thí nghiệm — MỌI tham số đều đặt tường minh, không để mặc định.

Vì sao KHÔNG ép cả 3 model cùng temperature:
  DeepSeek ghi rõ trên model card của R1-Distill rằng temperature=0 khiến model
  "lặp vô tận hoặc mất mạch lạc", và khuyến nghị 0.5-0.7 (mặc định 0.6), top_p 0.95.
  Ép R1 về 0.1 = cố tình chạy model sai khuyến nghị của chính nhà sản xuất -> so sánh
  mất công bằng. Ngược lại Qwen2.5-Instruct là model chỉ-dẫn thường, chạy tốt ở temp
  thấp và đó là cấu hình dùng thật khi triển khai.
  => Mỗi model chạy ở CẤU HÌNH TỐT NHẤT CỦA NÓ. So sánh là "model ở trạng thái tốt
     nhất", không phải "model bị bóp cùng một tham số".
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str                  # định danh ngắn dùng trong file kết quả
    hf_id: str                # tên vLLM đang phục vụ (phải khớp --served-model-name / repo id)
    label: str                # tên hiển thị trên báo cáo
    port: int
    family: str               # 'instruct' | 'reasoning'
    params_b: float
    temperature: float
    top_p: float
    max_tokens: int
    phase: str                # 'A' = 2 model nhỏ chạy chung | 'B' = 32B chạy riêng
    temp_rationale: str
    # Concurrency của PHA CHẤT LƯỢNG, đặt theo ngân sách KV cache của từng model.
    # KHÔNG ảnh hưởng công bằng (chất lượng mỗi request độc lập), nhưng đặt quá cao so
    # với KV thì vLLM phải preempt/swap -> chậm hơn chứ không nhanh hơn.
    #
    # Cách tính trần cho 32B: 64 lớp, 8 KV head (GQA), head_dim 128, bf16.
    #   mỗi token/lớp = 2(K+V) x 8 x 128 x 2 byte = 4 KiB -> x64 lớp = 256 KiB/token
    #   KV 24GB / 256KiB ~= 98k token.  Mỗi request ~7k token -> trần ~14 luồng.
    #   Đặt 12 để còn biên. 7B nhẹ hơn nhiều (28 lớp) nên 16 thoải mái.
    quality_concurrency: int = 8
    # Tên ngắn cho trục biểu đồ: nhãn đầy đủ (vd 'DeepSeek-R1-Distill-Qwen-32B')
    # chồng lên nhau khi có 5 model trên một trục x.
    short: str = ""

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


MODELS: list[ModelSpec] = [
    ModelSpec(
        key="qwen7b", hf_id="Qwen/Qwen2.5-7B-Instruct",
        label="Qwen2.5-7B-Instruct (baseline, không suy luận)",
        port=8003, family="instruct", params_b=7.6,
        temperature=0.1, top_p=0.9, max_tokens=2048, phase="A", quality_concurrency=16,
        short="Qwen2.5-7B",
        temp_rationale="Model chỉ-dẫn thường: temp thấp (0.1) cho lời giải xác định, "
                       "triệt tiêu sáng tạo. Đây là cấu hình triển khai thật.",
    ),
    ModelSpec(
        key="r1_7b", hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        label="DeepSeek-R1-Distill-Qwen-7B (suy luận, nhỏ)",
        port=8004, family="reasoning", params_b=7.6,
        temperature=0.6, top_p=0.95, max_tokens=6144, phase="A", quality_concurrency=16,
        short="R1-7B",
        temp_rationale="Theo model card DeepSeek: temp 0.5-0.7 (khuyến nghị 0.6), top_p 0.95. "
                       "temp=0 gây lặp vô tận. max_tokens lớn vì cần chỗ 'nháp' trong <think>.",
    ),
    ModelSpec(
        key="r1_32b", hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        label="DeepSeek-R1-Distill-Qwen-32B (suy luận, lớn)",
        port=8002, family="reasoning", params_b=32.8,
        temperature=0.6, top_p=0.95, max_tokens=6144, phase="B", quality_concurrency=12,
        short="R1-32B",
        temp_rationale="Giống R1-7B — cùng họ, cùng khuyến nghị của DeepSeek.",
    ),
    # ── Bổ sung: bịt hai lỗ hổng lập luận ────────────────────────────────────
    ModelSpec(
        key="llama8b", hf_id="NousResearch/Meta-Llama-3.1-8B-Instruct",
        label="Llama-3.1-8B-Instruct (baseline khác họ)",
        port=8005, family="instruct", params_b=8.0,
        temperature=0.1, top_p=0.9, max_tokens=2048, phase="A2", quality_concurrency=16,
        short="Llama-3.1-8B",
        temp_rationale="Model chỉ-dẫn thường, cùng cấu hình với Qwen2.5-7B để so trực tiếp.",
    ),
    ModelSpec(
        key="r1_14b", hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        label="DeepSeek-R1-Distill-Qwen-14B (suy luận, trung gian)",
        port=8006, family="reasoning", params_b=14.8,
        temperature=0.6, top_p=0.95, max_tokens=6144, phase="A2", quality_concurrency=12,
        short="R1-14B",
        temp_rationale="Giống R1-7B/32B — cùng họ, cùng khuyến nghị DeepSeek.",
    ),
]

# Vì sao thêm 2 model này (cả hai đều bịt một lỗ hổng lập luận CÓ THẬT):
#
#  • R1-Distill-14B — trả lời "sao nhảy thẳng 7B lên 32B, không thử trung gian?".
#    Không có 14B thì kết luận "phải dùng 32B" chỉ là so 2 điểm mút; có 14B mới biết
#    đường cong chất lượng-theo-kích-thước bão hoà ở đâu. Nếu 14B đã đủ ngưỡng thì
#    kết luận đúng phải là chọn 14B (rẻ hơn, nhanh hơn) — và ta phải dám nói vậy.
#
#  • Llama-3.1-8B-Instruct — Qwen2.5-7B khống chế biến KIẾN TRÚC (R1-Distill vốn
#    chưng cất từ Qwen), nhưng vì thế nó KHÔNG chứng minh được kết luận có tổng quát
#    ngoài họ Qwen hay không. Llama khác hẳn nền -> nếu cả hai baseline cùng thua
#    reasoning model thì luận điểm mạnh hơn nhiều so với chỉ một baseline.
#
# LƯU Ý PHẢI GHI VÀO BÁO CÁO: dùng bản mirror NousResearch/Meta-Llama-3.1-8B-Instruct
# vì repo gốc meta-llama/* là GATED (cần token đã duyệt). Đây là bản sao y nguyên
# trọng số, không phải bản fine-tune lại.

MODEL_BY_KEY = {m.key: m for m in MODELS}

# Model làm giám khảo. Dùng 32B vì nó mạnh nhất trong hệ thống local.
# LƯU Ý THIÊN LỆCH (ghi vào báo cáo): 32B vừa là thí sinh vừa là giám khảo.
# Giảm thiểu bằng 3 cách:
#   1. Chấm CÓ THAM CHIẾU (đưa expected_answer làm chuẩn) — không phải chấm "hay/dở"
#      theo cảm tính, mà là "có khớp đáp án chuẩn không". Thiên lệch tự-ưa-thích
#      của LLM-judge chủ yếu xuất hiện ở chấm KHÔNG tham chiếu.
#   2. Đối chiếu máy móc trước (sympy + so khớp số) — giám khảo chỉ xử ca mơ hồ.
#   3. Lấy mẫu 60 ca cho người kiểm tra tay -> báo cáo độ đồng thuận người-máy.
JUDGE_KEY = "r1_32b"
# KHÔNG đặt 0.0: giám khảo cũng là R1, mà R1 ở temp=0 rơi vào lặp vô tận (chính
# DeepSeek cảnh báo). 0.2 = thấp nhất còn an toàn -> gần như tái lập được mà không
# treo. Độ ổn định của giám khảo không phải giả định: đo bằng calibrate_judge().
JUDGE_TEMPERATURE = 0.2
# 1024 là BẪY: giám khảo cũng là R1 nên nó sinh <think> trước mọi câu trả lời, thường
# 500-2000 token. Cắt ở 1024 thì rất nhiều lượt chấm bị dừng giữa lúc đang nghĩ,
# chưa kịp in dòng "CORRECT: yes/no" -> mất kết quả hàng loạt mà nhìn log không rõ vì sao.
# 3072 cho đủ chỗ nghĩ + in nhãn. Tỉ lệ chấm hỏng được đo trong calibrate_judge.
JUDGE_MAX_TOKENS = 3072

EMBED_PORT = 8001
EMBED_MODEL = "BAAI/bge-m3"

# ── Thiết kế thí nghiệm ──────────────────────────────────────────────────────
# Pha ĐO HIỆU NĂNG: concurrency=1 nghiêm ngặt, subset cân bằng môn+độ khó.
#   -> TTFT/TPOT/tokens-s/P95 sạch, không bị nhiễu do hàng đợi.
# Pha ĐO CHẤT LƯỢNG: concurrency=8 trên full 180 câu.
#   -> Chất lượng KHÔNG phụ thuộc concurrency (mỗi request độc lập, temp cố định),
#      nên chạy song song vẫn công bằng mà tiết kiệm ~9 giờ GPU.
PERF_CONCURRENCY = 1
PERF_SUBSET_PER_SUBJECT = 10      # 10 câu x 3 môn = 30 câu, chia đều easy/medium/hard
QUALITY_CONCURRENCY = 8
WARMUP_REQUESTS = 3               # làm nóng CUDA graph / cache trước khi đo
REQUEST_TIMEOUT_S = 600
SEED = 20260715                   # cố định để chọn subset tái lập được

RETRIEVAL_K_CHUNKS = 3            # lý thuyết
RETRIEVAL_K_QB = 3                # bài giải mẫu
# Ngưỡng cảnh báo trùng lặp: nếu câu test giống hệt 1 bài trong ngân hàng thì
# RAG "thắng" một cách tầm thường (chỉ việc chép đáp án). Phải đo và công bố.
CONTAMINATION_SIM = 0.95

# ── Giá API để quy đổi chi phí (USD / 1 triệu token, niêm yết 2026-07) ────────
# Dùng để trả lời "tự host có rẻ hơn gọi API không".
API_PRICING = {
    "gpt-4o":        {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":   {"in": 0.15, "out": 0.60},
    "deepseek-chat": {"in": 0.27, "out": 1.10},
}
POD_USD_PER_HOUR = 2.00   # RTX PRO 6000 Blackwell 96GB trên RunPod
