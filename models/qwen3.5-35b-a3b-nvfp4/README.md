# Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 on Jetson Thor

## 1) Select image

```bash
IMAGE=ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04
```

## 2) Download weights persistently to HF hub cache (optional pre-download)

```bash
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
    "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
    resume_download=True,
    max_workers=1,
    cache_dir="/data/models/huggingface/hub",
)
print("download complete")
PY'
```

## 3) Serve model

```bash
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
  vllm serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
    --download-dir /data/models/huggingface/hub \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.8
```

## 4) Smoke test from host

```bash
curl -s http://127.0.0.1:8000/v1/models | jq
```
