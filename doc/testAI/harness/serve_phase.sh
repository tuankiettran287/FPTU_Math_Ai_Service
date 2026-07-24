#!/usr/bin/env bash
# Dựng model cho một pha benchmark:  bash serve_phase.sh A|B
#
# ⚠️ gpu-memory-utilization = PHẦN RIÊNG của instance, tính trên TỔNG VRAM của card:
#      budget_i ≈ TỔNG_VRAM × util_i          (KHÔNG trừ phần model khác đang giữ)
#    => nhiều instance thì các util phải CỘNG LẠI < 1.0, chứ không phải tăng dần.
#
#    ĐO THẬT trên pod này (vLLM 0.10.2, RTX PRO 6000 96GB, 2026-07-15):
#      bge 2.4GB + qwen(0.30) + r17b(0.55)  ->  87.9GB thực tế
#      khớp với 2.4 + 95.6*0.30 + 95.6*0.55 = 83.7GB (+ overhead)
#      Nếu util là ngưỡng cộng dồn thì tổng đã phải là ~52.6GB — không phải vậy.
#    Hệ quả: tổng util của MỌI instance phải chừa chỗ cho bge (~0.025) và overhead.
#
# KHÔNG dùng --enforce-eager ở bất kỳ model nào: eager bỏ CUDA graph, chậm 10-15%.
# Nếu chỉ 32B chạy eager thì nó bị bóp tốc độ giả tạo -> so sánh hiệu năng mất công bằng.
set -e
PHASE="${1:?dùng: serve_phase.sh A|A2|B}"
export HF_HOME=/workspace/hf
V=/usr/local/bin/vllm

# Model Load Time PHẢI đo ở đây. bench.run gọi wait_ready() khi model đã sẵn sàng từ
# trước nên luôn ra ~0s — vô nghĩa. Ghi ra JSON để bench.run đọc lại.
LOADFILE=/workspace/testai/results/load_times.json
mkdir -p /workspace/testai/results
[ -f "$LOADFILE" ] || echo '{}' > "$LOADFILE"

record_load() {  # $1=model_key $2=giây
  python3 - "$LOADFILE" "$1" "$2" <<'PY'
import json, sys
f, k, v = sys.argv[1], sys.argv[2], float(sys.argv[3])
try:
    d = json.load(open(f))
except Exception:
    d = {}
d[k] = v
json.dump(d, open(f, "w"), indent=1)
PY
}

# ─────────────────────────────────────────────────────────────────────────────
# MỘT chỗ duy nhất biết danh sách model. MỌI pha đều gọi hàm này trước khi dựng.
#
# Vì sao phải gom về một hàm: trước đây mỗi pha tự liệt kê session cần kill. Thêm
# pha A2 (llama, r114b) thì pha A không biết chúng tồn tại -> Qwen khởi động khi
# Llama+14B vẫn ôm 81GB -> "Engine core initialization failed".
# Đã dính lỗi này HAI LẦN (r17b sót ở pha A, rồi llama/r114b sót ở pha A).
# Liệt kê tay từng pha là mô hình sai; danh sách phải nằm ở đúng một chỗ.
ALL_LLM_SESSIONS="llm32 qwen r17b llama r114b"
ALL_LLM_PORTS="8002 8003 8004 8005 8006"   # KHÔNG có 8001 (bge) và 8080 (FastAPI)

stop_all_llms() {
  for s in $ALL_LLM_SESSIONS; do tmux kill-session -t "$s" 2>/dev/null || true; done
  for p in $ALL_LLM_PORTS;    do fuser -k "$p"/tcp 2>/dev/null || true; done
  # Cổng đóng KHÔNG có nghĩa là VRAM đã trả — driver thu hồi sau vài giây.
  # Ngưỡng 6000 MiB = chỉ còn bge-m3 (~2.4GB) + biên.
  local i u
  for i in $(seq 1 36); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u" -lt 6000 ] && break
    sleep 5
  done
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo ">> VRAM sau khi dọn: ${u} MiB (chỉ nên còn bge ~2400)"
  if [ "$u" -ge 6000 ]; then
    # Dừng HẲN thay vì dựng model rồi chết vì thiếu VRAM: chết ở đây thì log rõ ràng,
    # chết ở kia thì phải đi đọc log vLLM mới hiểu.
    echo "!! VRAM chưa được trả (${u} MiB) — tiến trình nào đó còn giữ. Dừng."
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    return 1
  fi
  return 0
}

wait_ready() {  # $1=port $2=tên(=tên log) $3=giây tối đa $4=model_key
  local t0=$(date +%s)
  until curl -s -m 3 -o /dev/null "http://127.0.0.1:$1/v1/models"; do
    local now=$(( $(date +%s) - t0 ))
    if [ "$now" -ge "$3" ]; then
      echo "!! $2 :$1 chưa lên sau ${3}s — 25 dòng log cuối:"
      tail -25 "/workspace/logs/$2.log" 2>/dev/null
      return 1
    fi
    sleep 5
  done
  local el=$(( $(date +%s) - t0 ))
  echo ">> $2 :$1 READY (${el}s)"
  [ -n "$4" ] && record_load "$4" "$el"
}

if [ "$PHASE" = "A" ]; then
  stop_all_llms || exit 1

  # CẢ HAI model 7B đặt CÙNG util 0.35 -> cùng ngân sách KV.
  # Vì sao phải bằng nhau: đo thật cho thấy trọng số hai model gần như y hệt
  # (Qwen 14.2488 GiB, R1-7B 14.2717 GiB — cùng 7.6B, cùng nền Qwen). Nếu đặt util
  # lệch (0.30 vs 0.55) thì VRAM đo được lệch 30GB vs 55GB — nhưng đó là DO CẤU HÌNH
  # của mình, không phải bản chất model. Đưa số đó lên slide là số liệu sai.
  # budget ≈ 95.6*0.35 ≈ 33.5GB (trọng số 14.25 + KV ~18.8 + CUDA graph 0.45)
  echo ">> Qwen2.5-7B-Instruct :8003"
  tmux new-session -d -s qwen "HF_HOME=$HF_HOME $V serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 --port 8003 --max-model-len 16384 --gpu-memory-utilization 0.35 \
    2>&1 | tee /workspace/logs/qwen.log"
  wait_ready 8003 qwen 900 qwen7b

  # R1-7B: CÙNG util 0.35 với Qwen (xem lý do ở trên).
  # Tổng: bge 2.4 + 33.5 + 33.5 = 69.4GB < 95.6GB -> an toàn.
  echo ">> DeepSeek-R1-Distill-Qwen-7B :8004"
  tmux new-session -d -s r17b "HF_HOME=$HF_HOME $V serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --host 0.0.0.0 --port 8004 --max-model-len 16384 --gpu-memory-utilization 0.35 \
    2>&1 | tee /workspace/logs/r17b.log"
  wait_ready 8004 r17b 900 r1_7b

elif [ "$PHASE" = "A2" ]; then
  # Bổ sung: Llama-3.1-8B (baseline khác họ) + R1-Distill-14B (reasoning trung gian).
  # Không ở chung với 32B được: 32B đã chiếm 87GB/96GB.
  stop_all_llms || exit 1

  # Llama-3.1-8B: trọng số ~16GB. util 0.30 -> 28.7GB -> KV ~12GB.
  # 32 lớp x 8 KV head x 128 dim x bf16 = 128 KiB/token -> ~93k token. conc=16 x ~3k = 48k. Dư.
  echo ">> Llama-3.1-8B-Instruct :8005"
  tmux new-session -d -s llama "HF_HOME=$HF_HOME $V serve NousResearch/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8005 --max-model-len 16384 --gpu-memory-utilization 0.30 \
    2>&1 | tee /workspace/logs/llama.log"
  wait_ready 8005 llama 900 llama8b

  # R1-14B: trọng số ~29.5GB. util 0.50 -> 47.8GB -> KV ~17.8GB.
  # 48 lớp x 8 KV head x 128 dim x bf16 = 192 KiB/token -> ~92k token. conc=12 x ~9k = 108k
  # -> hơi vượt, vLLM tự xếp hàng phần dư (không lỗi, chỉ là conc hiệu dụng ~10).
  echo ">> DeepSeek-R1-Distill-Qwen-14B :8006"
  tmux new-session -d -s r114b "HF_HOME=$HF_HOME $V serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --host 0.0.0.0 --port 8006 --max-model-len 16384 --gpu-memory-utilization 0.50 \
    2>&1 | tee /workspace/logs/r114b.log"
  wait_ready 8006 r114b 1200 r1_14b
  # Tổng: bge 2.4 + 28.7 + 47.8 = 78.9GB < 95.6GB.
else
  # Pha B: 32B KHÔNG ở chung được với model nào khác.
  # Đo thật: trọng số 61.06GB + KV 18.78 + graph 0.82 -> chiếm 87.3GB/95.6GB.
  stop_all_llms || exit 1

  # Chỉ còn bge (~2.4GB). util 0.90 -> budget ≈ 95.6*0.90 ≈ 86GB; +bge = 88.4GB < 95.6GB.
  # Trọng số 32B đo thật = 61.06GB -> còn ~25GB cho KV: thoải mái cho conc=8.
  # Đo thật phiên trước: util 0.70/0.72 CHẾT (chỉ còn ~6GB cho KV+activation),
  # util 0.76 sống (72.7GB). Nay 7B đã tắt nên nới lên 0.90 được, và nhờ vậy KHÔNG
  # phải dùng --enforce-eager -> giữ CUDA graph, không bóp tốc độ 32B một cách giả tạo.
  echo ">> DeepSeek-R1-Distill-Qwen-32B :8002"
  tmux kill-session -t llm32 2>/dev/null || true
  tmux new-session -d -s llm32 "HF_HOME=$HF_HOME $V serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --host 0.0.0.0 --port 8002 --max-model-len 16384 --gpu-memory-utilization 0.90 \
    2>&1 | tee /workspace/logs/llm32.log"
  wait_ready 8002 llm32 2400 r1_32b
fi

echo "--- VRAM sau khi dựng pha $PHASE ---"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
tmux ls
