# thor_llm

Runbooks for serving LLMs on Jetson Thor with persistent Hugging Face and runtime caches.

## Directory layout

- `models/qwen3-coder-next-nvfp4`: `GadflyII/Qwen3-Coder-Next-NVFP4` flow. Current version uses the NVIDIA container directly; previous derived-image workflow is preserved at the end of that README.
- `models/qwen3.5-35b-a3b-nvfp4`: `Kbenkhaled/Qwen3.5-35B-A3B-NVFP4` flow (vendor image, unified HF cache layout).
- `models/qwen3.5-122b-a10b-nvfp4-resharded`: `patrickbdevaney/qwen-3.5-122b-a10b-nvfp4-resharded` flow (persistent download + optimized serve flags for Thor).

## Shared host setup (once)

```bash
mkdir -p ~/thor-hf-cache/hub ~/thor-vllm-cache ~/thor-torch-cache
```

Optional: if a model requires auth.

```bash
export HF_TOKEN="YOUR_HF_TOKEN"
```

## Cache conventions

Use one cache root for all HF-related tools to avoid duplicate `models--*` folders outside `/hub`:

```bash
-e HF_HOME=/data/models/huggingface \
-e HF_HUB_CACHE=/data/models/huggingface/hub \
-e TRANSFORMERS_CACHE=/data/models/huggingface/hub
```

Mounts used in all runbooks:

```bash
-v "$HOME/thor-hf-cache:/data/models/huggingface" \
-v "$HOME/thor-vllm-cache:/root/.cache/vllm" \
-v "$HOME/thor-torch-cache:/root/.cache/torch"
```
