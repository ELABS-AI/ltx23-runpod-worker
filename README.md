# LTX 2.3 Serverless Worker for RunPod

[![Deploy on RunPod](https://img.shields.io/badge/RunPod-Deploy-orange?logo=runpod)](https://console.runpod.io)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)

**Text-to-video and image-to-video generation** with Lightricks LTX 2.3 (22B parameters). Two-pass architecture with latent upscaling and synchronized audio.

## Architecture

Two-pass generation pipeline validated against the official Comfy-Org LTX 2.3 workflow:

```
Pass 1 (low-res):  9-step euler sampling at half resolution
          ↓
Latent Upscale:    2x spatial upscaler
          ↓
Pass 2 (high-res): 4-step euler sampling at full resolution
          ↓
Decode:            Tiled VAE decode + Audio VAE decode
          ↓
Output:            CreateVideo (MP4 with synchronized audio)
```

## Models (auto-downloaded to network volume)

| Model | Size | Source |
|-------|------|--------|
| `ltx-2.3-22b-dev-fp8.safetensors` | ~22GB | Lightricks/LTX-2.3-fp8 |
| `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` | ~2GB | Comfy-Org/ltx-2.3 |
| `gemma_3_12B_it_fp4_mixed.safetensors` | ~8GB | Comfy-Org/ltx-2 |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~1GB | Lightricks/LTX-2.3 |

**Total:** ~33GB (recommend 60GB network volume)

## Deployment

1. Create a 60GB network volume in RunPod console
2. Create serverless endpoint using this repo's prebuilt image
3. Set env vars:
   - `HUGGINGFACE_ACCESS_TOKEN` — HF read token (required for model downloads)
4. Attach network volume to endpoint

## API

### Text-to-Video

```json
{
  "input": {
    "prompt": "A serene mountain lake at dawn, mist rising off the water, cinematic lighting",
    "width": 1280,
    "height": 720,
    "duration": 5,
    "fps": 25,
    "seed": 42
  }
}
```

### Image-to-Video

```json
{
  "input": {
    "prompt": "A serene mountain lake at dawn, mist rising off the water",
    "image_url": "https://example.com/input.jpg",
    "width": 544,
    "height": 960,
    "duration": 5,
    "fps": 25
  }
}
```

### Output

```json
{
  "video_b64": "<base64-encoded MP4>",
  "prompt": "...",
  "seed": 42,
  "width": 1280,
  "height": 720,
  "duration": 5,
  "fps": 25,
  "wall_time_s": 12.3,
  "status": "COMPLETED"
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description of the video |
| `width` | int | 1280 | Output width (512-2048, multiple of 64) |
| `height` | int | 720 | Output height (512-2048, multiple of 64) |
| `duration` | int | 5 | Duration in seconds (2-30) |
| `fps` | int | 25 | Frames per second (8-30) |
| `seed` | int | random | Random seed |
| `image_url` | string | null | Input image URL for I2V mode |

## GPU Requirements

- Minimum: >=24GB VRAM (for FP8 model at 720p)
- Recommended: RTX 4090, L40S, A5000 (>=24GB)
- CUDA: 12.4+

## Cost Estimate

- GPU: $0.46-$0.69/hr (RTX 3090 to RTX 4090)
- Per 5s video: ~$0.04-0.06 at low resolution
- Network volume: ~$4.20/mo

## License

Apache-2.0. Based on [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3).
