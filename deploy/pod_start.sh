#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Chạy TRÊN RunPod pod (RTX A6000 48GB, image runpod/pytorch cu128/torch280).
# Dựng 2 dịch vụ model trong tmux: vLLM LLM (:8000) + embedding bge-m3 (:8001).
# 48GB VRAM dư sức chạy DeepSeek-R1-Distill-Qwen-7B bf16 — KHÔNG cần quantize.
#
#   bash /workspace/pod_start.sh
#
# GHI CHÚ QUAN TRỌNG (đã gặp thực tế khi setup):
#  • PIN vllm==0.10.2 + transformers==4.55.4: image pod có vllm/torch/transformers
#    quá mới → torch cu130 (driver too old) và tokenizer thiếu attr. Bản pin này khớp
#    driver 570 (CUDA 12.8) và chạy được.
#  • Image chạy sẵn nginx trên :8001 → phải dừng để nhường cổng.
#  • Dùng tmux (KHÔNG nohup): RunPod dọn tiến trình khi phiên SSH đóng; tmux server
#    persist. TUYỆT ĐỐI không pkill theo chuỗi "vllm" (khớp luôn shell đang chạy) —
#    dọn theo PORT.
# ─────────────────────────────────────────────────────────────────────────────
set -e

export HF_HOME="${HF_HOME:-/workspace/hf}"       # cache model vào volume /workspace
export PIP_BREAK_SYSTEM_PACKAGES=1               # Ubuntu 24.04 externally-managed (PEP 668)
# Thư mục code AI service trên pod (đồng bộ bằng git / VS Code Remote-SSH).
export APP_DIR="${APP_DIR:-/workspace/FPTU_Math_Ai_Service-main}"
mkdir -p "$HF_HOME" /workspace/logs

# ── TUNNEL 2 CHIỀU pod↔VM (tự dựng nếu chưa có) ──────────────────────────────
# Nhờ tunnel này mà: (1) FastAPI trên pod đọc Postgres VM qua localhost:15432 (-L);
# (2) BE (docker) trên VM gọi FastAPI pod qua host:18080 CỐ ĐỊNH (-R), không dùng
# URL proxy runpod (proxy cắt >~100s → 524). Pod đổi ID vẫn KHÔNG phải sửa gì bên VM.
#
# Yêu cầu (làm 1 lần cho mỗi pod mới, KHÔNG đổi giữa các lần):
#   • RunPod secret FPTU_VM_TUNNEL_KEY_B64 = base64 của private key runpod-tunnel@VM.
#   • VM sshd (Match User runpod-tunnel): AllowTcpForwarding yes + GatewayPorts yes.
VM_HOST="${VM_HOST:-34.124.220.236}"
VM_USER="${VM_USER:-runpod-tunnel}"
VM_KEY="${VM_KEY:-/root/.ssh/runpod_fptu_vm_ed25519}"
if [ ! -f "$VM_KEY" ] && [ -n "${FPTU_VM_TUNNEL_KEY_B64:-}" ]; then
  mkdir -p /root/.ssh && chmod 700 /root/.ssh
  printf '%s' "$FPTU_VM_TUNNEL_KEY_B64" | base64 -d > "$VM_KEY" && chmod 600 "$VM_KEY"
fi
if [ -f "$VM_KEY" ] && ! tmux has-session -t tunnel 2>/dev/null; then
  echo ">> Dựng tunnel pod↔VM (-L 15432 DB, -R 18080 AI)..."
  tmux new-session -d -s tunnel \
    "ssh -N -L 15432:127.0.0.1:5432 -R 18080:localhost:8080 \
       -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
       -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
       -i $VM_KEY $VM_USER@$VM_HOST"
  sleep 5
fi

echo ">> Cài vLLM + transformers (pin bản tương thích driver)..."
pip install -q --break-system-packages "vllm==0.10.2" "transformers==4.55.4"

echo ">> Cài deps AI service (FastAPI + DB + đọc file + OCR)..."
# fastapi/uvicorn = web; psycopg[binary]+pgvector = DB; pypdf/python-docx = đọc file;
# sympy = kiểm đáp án; rapidocr-onnxruntime+pillow = OCR ảnh (CPU, KHÔNG đụng torch/vLLM).
pip install -q --break-system-packages \
  "fastapi" "uvicorn[standard]" "psycopg[binary]" "pgvector" "httpx" "python-dotenv" \
  "pypdf" "python-docx" "sympy" "pillow" "rapidocr-onnxruntime" || true

echo ">> Dừng nginx (image chiếm :8001) + giải phóng cổng..."
nginx -s stop 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true
fuser -k 8080/tcp 2>/dev/null || true
# Kill ĐÚNG session llm/embed/api (KHÔNG dùng 'tmux kill-server' — nếu script này chạy
# trong 1 tmux session thì kill-server sẽ tự giết luôn chính nó).
tmux kill-session -t llm 2>/dev/null || true
tmux kill-session -t llm32 2>/dev/null || true
tmux kill-session -t embed 2>/dev/null || true
tmux kill-session -t api 2>/dev/null || true
# KHÔNG kill session 'tunnel' — FastAPI cần tunnel DB sống trước khi khởi động.
sleep 3

# ── VRAM trên RTX PRO 6000 Blackwell 96GB — CÁC SỐ NÀY LÀ ĐO THẬT, ĐỪNG ĐOÁN ──
#
# ⚠️ `gpu-memory-utilization` KHÔNG phải "phần dành riêng cho instance này".
#    vLLM tính: budget = TỔNG_VRAM × util − (phần MỌI tiến trình đang chiếm).
#    ⇒ instance thứ 2 phải đặt util CỘNG DỒN, không phải phần thêm.
#    Đặt sai kiểu 0.76 + 0.18 → con thứ 2 ra budget ÂM → chết
#    "No available memory for the cache blocks".
#
# ĐO THẬT (2026-07-15, DeepSeek-R1-Distill-Qwen-32B bf16, max-model-len 16384):
#   • Trọng số 32B      = 61.06 GiB
#   • util 0.70 / 0.72  → CHẾT (activation lúc profiling ăn hết, KV = 0)
#     (kể cả khi đã --enforce-eager và --max-num-batched-tokens 2048)
#   • util 0.76         → SỐNG, chiếm 73 GiB, còn trống ~22.6 GiB
#   • 7B bf16 cần ~20 GiB + bge-m3 ~2.5 GiB = 22.5 GiB → VỪA KHÍT 0.1 GiB ⇒ KHÔNG khả thi.
#     Ép 7B vào (util cộng dồn 0.97) thì đỉnh activation của nó GIẾT luôn 32B.
#   ⇒ CHỐT: 96GB chỉ chở nổi **32B bf16 + bge-m3**. Muốn thêm 7B phải dùng 32B-AWQ
#     (~20GB) hoặc GPU thứ 2.
#
# ⚠️ DISK: 32B ~62GB + bge ~2.3GB. Volume /workspace phải ≥100GB.
#    Volume 80GB chết giữa chừng "[Errno 122] Disk quota exceeded" + HF tự xoá ngược
#    cache. `df` báo dung lượng CẢ CỤM (hàng trăm TB) nên KHÔNG hề thấy trước được.
LLM32_MODEL="${LLM32_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"

wait_ready() {  # $1=port  $2=tên  $3=giây tối đa
  local i=0
  until curl -s -m 3 -o /dev/null "http://127.0.0.1:$1/v1/models"; do
    i=$((i+5)); [ "$i" -ge "$3" ] && { echo "!! $2 :$1 chưa lên sau ${3}s"; return 1; }
    sleep 5
  done
  echo ">> $2 :$1 READY (${i}s)"
}

# --enforce-eager: bỏ CUDA graph → nhường vài GiB cho KV. Chậm ~10-15% nhưng 32B chỉ
# chạy nền (chấm tự luận / sinh đề) nên không ảnh hưởng SV.
# KHÔNG đặt --served-model-name: để vLLM phục vụ đúng repo ID HuggingFace.
# Đặt bí danh (vd "deepseek-32b") sinh ra 2 tên cho 1 model → .env phải khớp bí danh,
# trong khi MODEL_NAME lại phải là repo thật (nhánh fallback gọi from_pretrained).
# Một tên duy nhất cho đỡ nhầm.
echo ">> LLM-32B :8002 ($LLM32_MODEL, bf16) — TOÀN BỘ tính năng..."
tmux new-session -d -s llm32 \
  "HF_HOME=$HF_HOME /usr/local/bin/vllm serve $LLM32_MODEL \
     --host 0.0.0.0 --port 8002 \
     --max-model-len 16384 --gpu-memory-utilization 0.76 --enforce-eager \
     2>&1 | tee /workspace/logs/vllm_32b.log"
wait_ready 8002 "LLM-32B" 2400 || true

# util 0.05: bge-m3 task=embed không cần KV cache nên rất nhẹ.
echo ">> Embedding :8001 (bge-m3, 1024 chiều) trong tmux 'embed'..."
tmux new-session -d -s embed \
  "HF_HOME=$HF_HOME /usr/local/bin/vllm serve BAAI/bge-m3 --task embed \
     --host 0.0.0.0 --port 8001 --served-model-name BAAI/bge-m3 \
     --gpu-memory-utilization 0.05 \
     2>&1 | tee /workspace/logs/vllm_embed.log"
wait_ready 8001 "Embed" 600 || true

# FastAPI AI service :8080 trong tmux 'api' (CÙNG pod → gọi vLLM/embed qua localhost).
# .env trong APP_DIR quyết định DATABASE_URL (trỏ Postgres VPS) + VLLM/EMBEDDING localhost.
if [ -d "$APP_DIR/AI_service" ]; then
  echo ">> FastAPI AI service :8080 trong tmux 'api' (APP_DIR=$APP_DIR)..."
  # :8002 = 32B (KHÔNG phải 8000). VLLM_MODEL phải khớp ĐÚNG tên vLLM đang phục vụ.
  tmux new-session -d -s api \
    "cd $APP_DIR && VLLM_BASE_URL=http://localhost:8002/v1 VLLM_MODEL=$LLM32_MODEL \
       EMBEDDING_API_URL=http://localhost:8001/v1 \
       /usr/bin/python3 -m uvicorn AI_service.main:app --host 0.0.0.0 --port 8080 \
       2>&1 | tee /workspace/logs/aiservice.log"
else
  echo "!! KHÔNG thấy $APP_DIR/AI_service — bỏ qua FastAPI. Đồng bộ code rồi chạy lại (hoặc set APP_DIR)."
fi

sleep 2
tmux ls || true
cat <<'EOF'

>> Đang tải & nạp model. Theo dõi:
     tail -f /workspace/logs/vllm_llm.log      # đợi "Uvicorn running on ... 8000"
     tail -f /workspace/logs/vllm_embed.log
     tmux attach -t llm                        # xem trực tiếp (Ctrl-b d để thoát)
>> Kiểm tra khi sẵn sàng:
     curl -s localhost:8000/v1/models
     curl -s localhost:8001/v1/embeddings -H 'Content-Type: application/json' \
       -d '{"model":"BAAI/bge-m3","input":["đạo hàm là gì"]}' | head -c 200
>> Phải EXPOSE HTTP 8000 & 8001 khi tạo pod. URL: https://<pod-id>-8000.proxy.runpod.net
EOF
