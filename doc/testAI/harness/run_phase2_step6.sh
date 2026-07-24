#!/usr/bin/env bash
# BAN CHAY TIEP: bo buoc 1 (dung pha A2) vi Llama+14B DA nap san tren GPU.
# Vòng 2: thêm Llama-3.1-8B + R1-Distill-14B, và chạy BỘ TEST KHÓ (V2H) cho cả 5 model.
# Chạy SAU KHI bench.run --phase B --problems V2H (32B trên bộ khó) đã xong.
#
#   tmux new-session -d -s bench2 'bash /workspace/testai/run_phase2.sh > /workspace/logs/bench2.log 2>&1'
#
# ── RÀNG BUỘC ĐĨA (đã dính sự cố thật) ──────────────────────────────────────
# Volume pod chỉ ~140GB, KHÔNG phải 200GB. `df` báo dung lượng cả cụm MooseFS
# (hàng trăm TB) nên KHÔNG bao giờ nhìn thấy quota trước khi vỡ.
#   32B 62 + 14B 28 + Llama 15 + Qwen 15 + R1-7B 15 + bge 4.3 = 139GB -> VỠ.
# => phải xoá model đã dùng xong trước khi tải model tiếp theo.
# Lần trước quota vỡ đã giết tiến trình chấm VÀ làm hỏng file code đang ghi.
#
# ── RÀNG BUỘC VRAM ─────────────────────────────────────────────────────────
# 32B chiếm 87.3GB/95.6GB -> không ở chung với model nào. Mà giám khảo LẠI là 32B.
# => trình tự bắt buộc: chạy model nhỏ -> tháo -> dựng 32B -> chấm.
set -e
cd /workspace/testai
export PYTHONUNBUFFERED=1
step() { echo; echo "############ $* ############"; date "+%H:%M:%S"; df_report; }
df_report() { echo "    [đĩa: $(du -sh /workspace/hf 2>/dev/null | cut -f1)]"; }
step "6/8 Dựng PHA A + chạy RIÊNG bộ khó cho Qwen + R1-7B"
bash serve_phase.sh A
# --append: 2 model này đã có kết quả bộ dễ. Thiếu cờ này sẽ ghi đè mất.
python3 -m bench.run --phase A --problems V2H --append

step "7/8 Dựng lại 32B (giám khảo) + chấm phần CHƯA chấm"
bash serve_phase.sh B
# Không --only: judge tự chấm nối tiếp, dòng nào đã có kết luận thì giữ nguyên.
python3 -m bench.judge --skip-calibration

step "8/8 Tổng hợp 5 model + Excel + biểu đồ + báo cáo"
python3 -m bench.vllm_log
python3 -m bench.analyze
python3 -m bench.report

step "HOÀN TẤT"
ls -la report/ report/charts/
