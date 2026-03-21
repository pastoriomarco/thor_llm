# Qwen3.5-35B-A3B-FP8 on Jetson Thor

Model source:
- `Qwen/Qwen3.5-35B-A3B-FP8`

## 1) Download model persistently

```bash
IMAGE=ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04

docker run --rm -it \
  --runtime nvidia --gpus all \
  --ipc=host --network host \
  -e HF_HOME=/data/models/huggingface \
  -e HF_HUB_CACHE=/data/models/huggingface/hub \
  -e TRANSFORMERS_CACHE=/data/models/huggingface/hub \
  -v "$HOME/thor-hf-cache:/data/models/huggingface" \
  "$IMAGE" \
  bash -lc 'python3 - << "PY"
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen3.5-35B-A3B-FP8",
    resume_download=True,
    max_workers=1,
    cache_dir="/data/models/huggingface/hub",
)
print("download complete")
PY'
```

## 2) Serve model (1 SEQ, MAX CONTEXT FP16 KV CACHE)

```bash
IMAGE=ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04 && \
docker run --rm -it \
  --runtime nvidia --gpus all \
  --ipc=host --network host \
  -e HF_HOME=/data/models/huggingface \
  -e HF_HUB_CACHE=/data/models/huggingface/hub \
  -e TRANSFORMERS_CACHE=/data/models/huggingface/hub \
  -v "$HOME/thor-hf-cache:/data/models/huggingface" \
  -v "$HOME/thor-vllm-cache:/root/.cache/vllm" \
  -v "$HOME/thor-torch-cache:/root/.cache/torch" \
  "$IMAGE" \
  vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --download-dir /data/models/huggingface/hub \
    --served-model-name Qwen3.5-35B-A3B-FP8 \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 1
```

### FP8 KV CACHE, 64K CONTEXT, 20 SEQ

Measured on Thor with:
- `Available KV cache memory: 56.81 GiB`
- `GPU KV cache size: 1,353,408 tokens`

That fits `20 x 65,536 = 1,310,720` tokens with margin.

```bash
IMAGE=ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04 && \
docker run --rm -it \
  --runtime nvidia --gpus all \
  --ipc=host --network host \
  -e HF_HOME=/data/models/huggingface \
  -e HF_HUB_CACHE=/data/models/huggingface/hub \
  -e TRANSFORMERS_CACHE=/data/models/huggingface/hub \
  -v "$HOME/thor-hf-cache:/data/models/huggingface" \
  -v "$HOME/thor-vllm-cache:/root/.cache/vllm" \
  -v "$HOME/thor-torch-cache:/root/.cache/torch" \
  "$IMAGE" \
  vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
    --download-dir /data/models/huggingface/hub \
    --served-model-name Qwen3.5-35B-A3B-FP8 \
    --tensor-parallel-size 1 \
    --max-model-len 65536 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 20
```

## 3) Verify model id

```bash
curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id'
```

Expected id:
- `Qwen3.5-35B-A3B-FP8`
