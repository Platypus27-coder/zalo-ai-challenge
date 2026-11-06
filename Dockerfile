# ==============================================================================
# Dockerfile — LifeMate Agent (Zalo AI Challenge 2026)
# Optimized for: 1x NVIDIA A30 GPU (24GB VRAM), Offline 100%
# ==============================================================================
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0

# Install audio & system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt requirements-gpu.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt

# Copy source code and scripts
COPY config.py /workspace/
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/

# Copy offline model weights
# (Run `python scripts/download_models.py` before building image)
COPY models/ /workspace/models/

# Default entrypoint for competition platform
ENTRYPOINT ["python", "scripts/run_offline.py"]
CMD ["--data-dir", "/data/public_test", "--output", "result/submission.jsonl"]
