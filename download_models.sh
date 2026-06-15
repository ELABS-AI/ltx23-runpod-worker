#!/bin/bash
# LTX 2.3 Model Downloader — runs on first worker start
# Downloads all models to the persistent network volume at /workspace

set -e

MODELS_DIR="/workspace/ComfyUI/models"
HF_TOKEN="$HUGGINGFACE_ACCESS_TOKEN"

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: No HUGGINGFACE_ACCESS_TOKEN set. Public downloads only."
fi

AUTH_HEADER="Authorization: Bearer $HF_TOKEN"

mkdir -p "$MODELS_DIR/checkpoints" "$MODELS_DIR/loras" \
         "$MODELS_DIR/text_encoders" "$MODELS_DIR/latent_upscale_models"

# ── 1. Main checkpoint — FP8 distilled (22GB) ──
CKPT="$MODELS_DIR/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
if [ ! -f "$CKPT" ]; then
    echo "Downloading LTX 2.3 FP8 checkpoint (22GB)..."
    wget --header="$AUTH_HEADER" \
        "https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" \
        -O "$CKPT"
else
    echo "Checkpoint already present, skipping"
fi

# ── 2. Distilled LoRA ──
LORA="$MODELS_DIR/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors"
if [ ! -f "$LORA" ]; then
    echo "Downloading distilled LoRA..."
    wget --header="$AUTH_HEADER" \
        "https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors" \
        -O "$LORA"
else
    echo "LoRA already present, skipping"
fi

# ── 3. Gemma text encoder (FP4) ──
TE="$MODELS_DIR/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"
if [ ! -f "$TE" ]; then
    echo "Downloading Gemma 3 12B FP4 text encoder..."
    wget --header="$AUTH_HEADER" \
        "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
        -O "$TE"
else
    echo "Text encoder already present, skipping"
fi

# ── 4. Spatial upscaler (2x) ──
UPSCALER="$MODELS_DIR/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
if [ ! -f "$UPSCALER" ]; then
    echo "Downloading latent upscaler..."
    wget --header="$AUTH_HEADER" \
        "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
        -O "$UPSCALER"
else
    echo "Upscaler already present, skipping"
fi

echo "All LTX 2.3 models downloaded successfully"
du -sh "$MODELS_DIR/checkpoints" "$MODELS_DIR/loras" \
       "$MODELS_DIR/text_encoders" "$MODELS_DIR/latent_upscale_models"
