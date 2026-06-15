FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="E-Labs AI Studio" description="LTX 2.3 video generation — T2V + I2V on RunPod"

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY download_models.sh .

# Models download to persistent network volume at /workspace
# Requires HUGGINGFACE_ACCESS_TOKEN env var for private model access

CMD ["python3", "-u", "handler.py"]
