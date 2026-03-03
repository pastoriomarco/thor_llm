# Qwen3.5-122B-A10B-NVFP4-resharded on Jetson Thor

Model source:
- `patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded`

## 1) Download model persistently

```bash
IMAGE=ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04
HOST_UID=$(id -u)
HOST_GID=$(id -g)

docker run --rm -it \
  -e HF_HOME=/data/models/huggingface \
  -e HF_HUB_CACHE=/data/models/huggingface/hub \
  -e HOST_UID="$HOST_UID" \
  -e HOST_GID="$HOST_GID" \
  -v "$HOME/thor-hf-cache:/data/models/huggingface" \
  --entrypoint bash \
  "$IMAGE" -lc '
    hf download patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded \
      --local-dir /data/models/huggingface/hub/qwen-3.5-122b-a10b-nvfp4-resharded
    chown -R ${HOST_UID}:${HOST_GID} /data/models/huggingface/hub/qwen-3.5-122b-a10b-nvfp4-resharded
  '
```

## 2) Serve model

```bash
IMAGE=ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04 && \
docker run --rm -it \
  --runtime nvidia --gpus all \
  --ipc=host --network host \
  -e HF_HOME=/data/models/huggingface \
  -e HF_HUB_CACHE=/data/models/huggingface/hub \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -v "$HOME/thor-hf-cache:/data/models/huggingface" \
  -v "$HOME/thor-vllm-cache:/root/.cache/vllm" \
  -v "$HOME/thor-torch-cache:/root/.cache/torch" \
  "$IMAGE" \
  vllm serve /data/models/huggingface/hub/qwen-3.5-122b-a10b-nvfp4-resharded/resharded \
    --served-model-name Qwen3.5-122B-A10B-NVFP4-resharded \
    --quantization compressed-tensors \
    --attention-backend FLASHINFER \
    --language-model-only \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 4096
```

## 3) Verify model id

```bash
curl -s http://127.0.0.1:8000/v1/models | jq -r '.data[].id'
```

Expected id:
- `Qwen3.5-122B-A10B-NVFP4-resharded`
