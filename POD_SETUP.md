# Dựng model trên GPU pod (RunPod) — vLLM + Embedding

> Máy dev **không có GPU** nên model chạy trên pod. AI service (FastAPI) chạy ở máy
> dev / server thường, gọi tới pod qua HTTP. Không dùng token/paid API — vLLM tự host.

Có **2 dịch vụ** cần chạy trên pod (2 cổng):
1. **LLM** — vLLM serve `DeepSeek-R1-Distill-Qwen-7B` (OpenAI-compatible) — cổng 8000.
2. **Embedding** — vLLM serve `bge-m3` (task embed) — cổng 8001.

## 0. Yêu cầu GPU & mức quantize
7B ở bf16 ≈ **15GB VRAM** (chưa tính KV-cache). Chọn theo VRAM pod:

| VRAM pod        | Khuyến nghị LLM                                  |
|-----------------|--------------------------------------------------|
| ≥ 40GB (A100/L40S) | bf16 nguyên bản, batch lớn — không cần quantize |
| 24GB (4090/L4/A10) | **FP8** (Ada/Hopper) hoặc **AWQ 4-bit**; KV-cache thoải mái |
| 16GB            | **AWQ 4-bit** bắt buộc, giảm `--max-model-len`   |

bge-m3 rất nhẹ (~2GB), chạy chung pod được.

## 1. LLM — vLLM (cổng 8000)
```bash
pip install "vllm>=0.6.0"

# bf16 (GPU ≥ 40GB)
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --port 8000 --host 0.0.0.0 \
  --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --max-model-len 16384 --gpu-memory-utilization 0.90

# FP8 (GPU 24GB Ada/Hopper) — thêm:
#   --quantization fp8
# AWQ 4-bit (VRAM nhỏ) — dùng bản đã quantize sẵn, ví dụ:
#   vllm serve casperhansen/deepseek-r1-distill-qwen-7b-awq --quantization awq ...
```
vLLM lo **continuous batching** (nhiều request song song) và **guided_decoding**
(`guided_json`) — service đã dùng ở giai đoạn 2 để ép JSON, không cần retry thủ công.

## 2. Embedding — bge-m3 (cổng 8001)
```bash
vllm serve BAAI/bge-m3 --task embed --port 8001 --host 0.0.0.0 \
  --served-model-name BAAI/bge-m3
```
Cho ra `/v1/embeddings` OpenAI-compatible (1024 chiều). Nếu không muốn vLLM cho
embedding, có thể chạy `sentence-transformers` sau một FastAPI nhỏ — miễn là cùng
định dạng `/v1/embeddings`.

## 3. Trỏ AI service tới pod
Trên máy chạy AI service, tạo `.env` (xem `.env.example`):
```env
LLM_BACKEND=vllm
VLLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1
VLLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

EMBEDDING_BACKEND=remote
EMBEDDING_API_URL=https://<pod-id>-8001.proxy.runpod.net/v1
EMBEDDING_API_MODEL=BAAI/bge-m3
```
> RunPod: mở HTTP port 8000/8001, dùng URL proxy `https://<pod-id>-<port>.proxy.runpod.net`.

## 4. Backfill embedding cho ngân hàng câu hỏi
Sau khi endpoint embedding sống, điền vector cho 14.511 câu (đang NULL):
```bash
python -m tools.backfill_embeddings           # chạy ở máy có .env trỏ tới pod
```

## 5. Kiểm tra nhanh
```bash
# LLM
curl $VLLM_BASE_URL/models
# Embedding
curl -s $EMBEDDING_API_URL/embeddings -H "Content-Type: application/json" \
  -d '{"model":"BAAI/bge-m3","input":["đạo hàm là gì"]}' | head -c 200
# AI service (máy dev)
curl localhost:8080/health
```
