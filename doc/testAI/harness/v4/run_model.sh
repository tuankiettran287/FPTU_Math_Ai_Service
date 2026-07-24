#!/usr/bin/env bash
# Chạy V4 cho MỘT model (dùng sau khi 32B đã xong, đã kill llm32 để nhường VRAM).
#   bash run_model.sh <key> <hf_id> <port> [max_model_len]
# Serve vLLM → trỏ FastAPI vào nó → chạy v4_bench. Model ≤14B nên serve một mình,
# util 0.55 là dư. Chấm dùng ĐÚNG pipeline production (temp cố định trong code service)
# → so sánh công bằng "cắm model X vào cùng bộ máy chấm".
set -e
KEY="$1"; HF_ID="$2"; PORT="$3"; MML="${4:-16384}"
export HF_HOME=/workspace/hf

# Volume pod sát trần ~150G → chỉ giữ 32B + model đang chạy. Xoá mọi model xoay vòng
# KHÁC target (giữ 32B và bge), rồi tải target nếu thiếu.
echo ">> [$KEY] dọn disk + đảm bảo $HF_ID có mặt"
CACHE_NAME="models--$(echo "$HF_ID" | sed 's#/#--#g')"   # HF cache dùng '--' ngăn cách
for d in /workspace/hf/hub/models--*; do
  b=$(basename "$d")
  case "$b" in
    *DeepSeek-R1-Distill-Qwen-32B|*bge-m3|"$CACHE_NAME") ;;   # giữ
    *) echo "   xoá $b"; rm -rf "$d" ;;
  esac
done
if [ ! -d "/workspace/hf/hub/$CACHE_NAME" ]; then
  echo ">> [$KEY] tải $HF_ID ..."
  huggingface-cli download "$HF_ID" --quiet 2>&1 | tail -2 || \
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_ID')"
fi

echo ">> [$KEY] serve $HF_ID :$PORT"
fuser -k "$PORT/tcp" 2>/dev/null || true
tmux kill-session -t "m_$KEY" 2>/dev/null || true
sleep 2
tmux new-session -d -s "m_$KEY" \
  "HF_HOME=$HF_HOME /usr/local/bin/vllm serve $HF_ID \
     --host 0.0.0.0 --port $PORT --served-model-name $HF_ID \
     --max-model-len $MML --gpu-memory-utilization 0.55 --enforce-eager \
     2>&1 | tee /workspace/logs/vllm_$KEY.log"

echo ">> [$KEY] chờ vLLM ready..."
i=0
until curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
  i=$((i+5)); [ "$i" -ge 1800 ] && { echo "!! $KEY chưa lên sau 1800s"; exit 1; }
  sleep 5
done
echo ">> [$KEY] vLLM READY (${i}s)"

echo ">> [$KEY] trỏ FastAPI vào :$PORT"
tmux kill-session -t api 2>/dev/null || true
sleep 2
tmux new-session -d -s api \
  "cd /workspace/FPTU_Math_Ai_Service-main && \
   VLLM_BASE_URL=http://localhost:$PORT/v1 VLLM_MODEL=$HF_ID \
   EMBEDDING_API_URL=http://localhost:8001/v1 \
   /usr/bin/python3 -m uvicorn AI_service.main:app --host 0.0.0.0 --port 8080 \
   2>&1 | tee /workspace/logs/aiservice.log"
sleep 8
until curl -s -m 3 -o /dev/null localhost:8080/health; do sleep 3; done
echo ">> [$KEY] FastAPI READY, model=$(curl -s localhost:8080/health)"

echo ">> [$KEY] chạy V4 135 câu..."
cd /workspace/v4
python3 v4_bench.py --model "$KEY" --base http://localhost:8080 \
  --testdir /workspace/testset_v4 --out /workspace/logs/v4 --concurrency 8 \
  2>&1 | tee "/workspace/logs/v4/run_$KEY.log"

echo ">> [$KEY] xong — kill vLLM :$PORT nhường VRAM"
tmux kill-session -t "m_$KEY" 2>/dev/null || true
fuser -k "$PORT/tcp" 2>/dev/null || true
echo ">> [$KEY] DONE"
