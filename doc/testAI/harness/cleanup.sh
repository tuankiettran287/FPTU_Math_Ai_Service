#!/usr/bin/env bash
# Dọn sạch mọi thứ của việc TEST khỏi pod, trả pod về đúng trạng thái production.
#   bash /workspace/testai/cleanup.sh
#
# ⚠️ CHỈ chạy SAU KHI đã tải report/ + results/ về máy.
#
# Vì sao dọn được an toàn: toàn bộ harness nằm gọn trong /workspace/testai và KHÔNG
# hề sửa file nào của dự án thật. Service AI thật (/workspace/FPTU_Math_Ai_Service-main)
# là bản sao nguyên vẹn từ repo, không đụng tới.
set -u

echo "=== Trạng thái trước khi dọn ==="
tmux ls 2>/dev/null || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo
echo "=== 1. Tháo 2 model chỉ dùng để test (Qwen2.5-7B, R1-Distill-7B) ==="
# 32B + bge-m3 GIỮ LẠI: đó là cấu hình production.
tmux kill-session -t qwen 2>/dev/null || true
tmux kill-session -t r17b 2>/dev/null || true
tmux kill-session -t serveA 2>/dev/null || true
tmux kill-session -t bench 2>/dev/null || true
fuser -k 8003/tcp 2>/dev/null || true
fuser -k 8004/tcp 2>/dev/null || true
sleep 5

echo "=== 2. Xoá code + dữ liệu test ==="
rm -rf /workspace/testai
rm -rf /workspace/dryrun   # bản chạy thử khô để kiểm tra đường code analyze/report
rm -f /workspace/logs/qwen.log /workspace/logs/r17b.log /workspace/logs/serveA.log \
      /workspace/logs/bench.log /workspace/logs/dl_qwen.log /workspace/logs/dl_r1_7b.log

echo "=== 3. Xoá cache 2 model chỉ dùng để test (~30GB) ==="
# Giữ 32B + bge-m3. Bỏ comment 2 dòng dưới nếu muốn giữ lại cho lần chạy sau.
rm -rf /workspace/hf/hub/models--Qwen--Qwen2.5-7B-Instruct
rm -rf /workspace/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B

echo
echo "=== Trạng thái sau khi dọn (production: 32B + bge-m3 + FastAPI + tunnel) ==="
tmux ls 2>/dev/null || true
echo "--- VRAM ---"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "--- Disk ---"; du -sh /workspace/hf 2>/dev/null
echo "--- Cổng ---"; ss -ltn 2>/dev/null | grep -oE ':(8001|8002|8080)' | sort -u | tr '\n' ' '; echo
echo "--- FastAPI health ---"; curl -s -m 10 http://127.0.0.1:8080/health || echo "FastAPI chưa lên"
echo
echo "Còn lại đúng những gì dự án thật cần:"
echo "  /workspace/FPTU_Math_Ai_Service-main  (service AI, nguyên vẹn)"
echo "  /workspace/hf  (32B + bge-m3)"
echo "  tmux: tunnel(DB) + embed(8001) + llm32(8002) + api(8080)"
