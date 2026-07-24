#!/usr/bin/env bash
# Chạy trọn thí nghiệm. Gọi khi PHA A đã sẵn sàng (serve_phase.sh A xong).
#   tmux new-session -d -s bench 'bash /workspace/testai/run_all.sh > /workspace/logs/bench.log 2>&1'
#
# Thứ tự cố định và có lý do:
#   - paraphrase chạy TRƯỚC bench.run: nó bổ sung biến thể vào v3_variants.json, mà
#     pha A và pha B BẮT BUỘC phải test trên cùng một bộ. Chạy sau sẽ khiến hai pha
#     dùng hai bộ test khác nhau -> so sánh vô nghĩa.
#   - paraphrase dùng Qwen (pha A) nên phải nằm trong pha A -> đỡ một lần nạp 32B.
#   - vllm_log chạy CUỐI: lúc đó có đủ log của cả 3 model.
set -e
cd /workspace/testai
export PYTHONUNBUFFERED=1

step() { echo; echo "############ $* ############"; date '+%H:%M:%S'; }

step "1/8 Sinh biến thể correct_paraphrase (bằng Qwen baseline)"
python3 -m bench.paraphrase

step "2/8 Benchmark PHA A (Qwen2.5-7B + R1-Distill-7B)"
python3 -m bench.run --phase A

step "3/8 Dựng PHA B (32B) — tháo 2 model 7B"
bash serve_phase.sh B

step "4/8 Benchmark PHA B (R1-Distill-32B)"
python3 -m bench.run --phase B

step "5/8 Bóc VRAM thật từ log vLLM (cả 3 model)"
python3 -m bench.vllm_log

step "6/8 Chấm (đối chiếu code trước, giám khảo 32B sau) + hiệu chuẩn giám khảo"
python3 -m bench.judge

step "7/8 Tổng hợp số liệu + Excel + biểu đồ"
python3 -m bench.analyze

step "8/8 Sinh báo cáo Markdown"
python3 -m bench.report

step "HOÀN TẤT"
ls -la report/
