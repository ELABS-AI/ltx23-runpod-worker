# A3: RunPod worker-comfyui base + LTX-2.3 custom nodes + handler
# Uses the proven worker-comfyui infrastructure (ComfyUI on 8188, handler as main process)
# Models are downloaded to network volume by download_models.sh on first boot

FROM runpod/worker-comfyui:5.8.5-base

LABEL maintainer="E-Labs AI Studio" description="LTX 2.3 video generation — T2V + I2V on RunPod"

# ── LTX-2.3 Custom Nodes ──────────────────────────────────────────────────
# Required for LTX-2.3 workflow nodes (audio VAE, upscaler, conditioning, etc.)
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo \
    /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    cd /comfyui/custom_nodes/ComfyUI-LTXVideo && \
    pip install -r requirements.txt && \
    echo "LTX-2.3 custom nodes installed"

# ── Handler ────────────────────────────────────────────────────────────────
# Replace default handler with LTX-2.3 workflow builder
# Accepts prompt/width/height/duration/fps/seed/image_url
# Builds correct two-pass T2V or I2V workflow, submits to local ComfyUI
COPY handler.py /handler.py

# ── Model Download Script ─────────────────────────────────────────────────
# Downloads models to network volume at /workspace/ComfyUI/models/
# Runs on first boot (handled by start.sh or manually triggered)
COPY download_models.sh /download_models.sh
RUN chmod +x /download_models.sh

# start.sh from the base image handles:
#   1. GPU pre-flight check
#   2. Starting ComfyUI on port 8188 (background)
#   3. Starting /handler.py as the main process (foreground)
# Our handler.py replaces the default, building LTX workflows on the fly.
