# thor_llm

**End-to-end runbook** to starts a working setup with Qwen3-Coder-Next-NVFP4 on Jetson Thor.

---

# 0) Host setup (once)

Create persistent directories:

```bash
mkdir -p ~/thor-hf-cache
mkdir -p ~/thor-vllm-cache
mkdir -p ~/thor-torch-cache
```

Optional: show sizes:

```bash
du -sh ~/thor-hf-cache ~/thor-vllm-cache ~/thor-torch-cache 2>/dev/null || true
```

Only if you need Hugging Face auth:

```bash
export HF_TOKEN="YOUR_HF_TOKEN"
```

---

# 1) Pull the vendor base image (Thor)

```bash
docker pull ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor
```

---

# 2) Download the model weights (persistently, resumable)

This writes into `~/thor-hf-cache` on the host.

```bash
docker run --rm -it \
  --runtime nvidia --gpus all \
  --ipc=host --network host \
  -e HF_HOME=/data/models/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HUB_MAX_WORKERS=1 \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=600 \
  -e HF_HUB_ETAG_TIMEOUT=60 \
  -e HF_HUB_DISABLE_XET=1 \
  -v "$HOME/thor-hf-cache:/data/models/huggingface" \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
  bash -lc 'python3 - <<'"'"'PY'"'"'
from huggingface_hub import snapshot_download
snapshot_download("GadflyII/Qwen3-Coder-Next-NVFP4", resume_download=True, max_workers=1)
print("download complete")
PY'
```

Verify shards exist on host:

```bash
find ~/thor-hf-cache -type f -name "model-00001-of-*.safetensors" | head
du -sh ~/thor-hf-cache
```

---

# 3) Build your derived image (deps + fixes baked in)

Create `Dockerfile.qwen3next-thor`:

```dockerfile
FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor

SHELL ["/bin/bash", "-lc"]

# IMPORTANT:
# - Don't set LD_PRELOAD in Dockerfile (it breaks during docker build because driver isn't mounted)
# - Don't import vllm._C at build-time (same reason)
# - We *do* install the packages you validated and keep runtime fixes for docker run.
RUN set -euo pipefail && \
    # disable the base-image pip pinning for THIS build layer
    unset PIP_CONSTRAINT PIP_APPEND_CONSTRAINT PIP_INDEX_URL PIP_EXTRA_INDEX_URL \
          PIP_TRUSTED_HOST PIP_WHEEL_DIR PIP_NO_CACHE_DIR PIP_VERBOSE && \
    \
    # core pins you already validated
    python3 -m pip install -U "setuptools<81" "protobuf==5.29.6" && \
    \
    # flashinfer: ensure python + cubin versions match
    python3 -m pip uninstall -y flashinfer-python flashinfer-cubin flashinfer-jit-cache || true && \
    python3 -m pip install -U flashinfer-python==0.6.3 flashinfer-cubin==0.6.3 && \
    \
    # model-card requirement (you proved it works)
    python3 -m pip install -U "transformers==5.2.0" && \
    \
    # clean up common vLLM extras
    python3 -m pip install -U opencv-python-headless pydot treelib "mistral-common[image]>=1.9.0" && \
    \
    # build-time checks that don't require CUDA driver symbols
    python3 -c "import transformers; print('transformers', transformers.__version__)" && \
    python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"
```

Build:

```bash
docker build -t vllm-thor:qwen3next -f Dockerfile.qwen3next-thor .
```

---

# 4) Runtime sanity check (driver fix + imports)

This is the check you *can’t* do in Docker build.

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

If you see `vllm._C OK`, the CUDA symbol issue is resolved.

---

# 5) Serve with all persistence enabled (weights + vLLM cache + torch cache)

```bash
docker run --rm -it \
  --runtime nvidia --gpus all \
  --ipc=host --network host \
  -e LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1 \
  -e LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH \
  -e HF_HOME=/data/models/huggingface \
  -e TRANSFORMERS_CACHE=/data/models/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -v "$HOME/thor-hf-cache:/data/models/huggingface" \
  -v "$HOME/thor-vllm-cache:/root/.cache/vllm" \
  -v "$HOME/thor-torch-cache:/root/.cache/torch" \
  vllm-thor:qwen3next \
  bash -lc '
set -euo pipefail
python3 -c "import vllm._C; print(\"vllm._C OK\")"
echo "Caches:"
ls -la /root/.cache/vllm || true
ls -la /root/.cache/torch || true

exec vllm serve GadflyII/Qwen3-Coder-Next-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.7 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
'
```

---

# 6) Smoke test from host

Models:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq
```

Completion:

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"GadflyII/Qwen3-Coder-Next-NVFP4",
    "prompt":"Write a Python function factorial(n).",
    "max_tokens":120,
    "temperature":0.2
  }' | jq -r '.choices[0].text'
```

---

