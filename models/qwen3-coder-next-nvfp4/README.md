# GadflyII/Qwen3-Coder-Next-NVFP4 on Jetson Thor

Current recommended flow uses the NVIDIA vLLM container directly, with unified HF cache paths under `/data/models/huggingface/hub`.

## 1) Select image

```bash
IMAGE=ghcr.io/nvidia-ai-iot/vllm:0.16.0-g15d76f74e-r38.2-arm64-sbsa-cu130-24.04
```

## 2) Download model weights persistently (optional pre-download)

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
    "GadflyII/Qwen3-Coder-Next-NVFP4",
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
  vllm serve GadflyII/Qwen3-Coder-Next-NVFP4 \
  --download-dir /data/models/huggingface/hub \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code
```

Do not use `--reasoning-parser qwen3` for this model.

## 4) Smoke test from host

```bash
curl -s http://127.0.0.1:8000/v1/models | jq
```

## Previous version

Previous workflow (kept for reference) used a derived image (`vllm-thor:qwen3next`) with dependency/runtime fixes.

### Pull base image

```bash
docker pull ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor
```

### Build derived image

Create `Dockerfile.qwen3next-thor`:

```dockerfile
FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor

SHELL ["/bin/bash", "-lc"]

RUN set -euo pipefail && \
    unset PIP_CONSTRAINT PIP_APPEND_CONSTRAINT PIP_INDEX_URL PIP_EXTRA_INDEX_URL \
          PIP_TRUSTED_HOST PIP_WHEEL_DIR PIP_NO_CACHE_DIR PIP_VERBOSE && \
    python3 -m pip install -U "setuptools<81" "protobuf==5.29.6" && \
    python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true && \
    python3 -m pip install -U flashinfer-python==0.6.3 flashinfer-cubin==0.6.3 && \
    python3 -m pip install -U "transformers==5.2.0" && \
    python3 -m pip install -U opencv-python-headless pydot treelib "mistral-common[image]>=1.9.0" && \
    python3 -c "import transformers; print('transformers', transformers.__version__)" && \
    python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"
```

Build:

```bash
docker build -t vllm-thor:qwen3next -f Dockerfile.qwen3next-thor .
```

### Runtime sanity check

```bash
docker run --rm -it \
  --runtime nvidia --gpus all \
  --ipc=host \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH \
  vllm-thor:qwen3next \
  bash -lc '
set -euo pipefail
python3 -c "import vllm._C; print(\"vllm._C OK\")"
python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"avail\", torch.cuda.is_available())"
python3 -c "import transformers, flashinfer; print(\"transformers\", transformers.__version__, \"flashinfer\", flashinfer.__version__)"
'
```

### Serve model (previous version)

```bash
docker run --rm -it \
  --runtime nvidia --gpus all \
  --ipc=host --network host \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH \
  -e HF_HOME=/data/models/huggingface \
  -e HF_HUB_CACHE=/data/models/huggingface/hub \
  -e TRANSFORMERS_CACHE=/data/models/huggingface/hub \
  -e HF_HUB_DISABLE_XET=1 \
  -v "$HOME/thor-hf-cache:/data/models/huggingface" \
  -v "$HOME/thor-vllm-cache:/root/.cache/vllm" \
  -v "$HOME/thor-torch-cache:/root/.cache/torch" \
  vllm-thor:qwen3next \
  bash -lc '
set -euo pipefail
python3 -c "import vllm._C; print(\"vllm._C OK\")"
exec vllm serve GadflyII/Qwen3-Coder-Next-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.7 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
'
```
