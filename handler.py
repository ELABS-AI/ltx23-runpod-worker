"""
RunPod serverless handler for LTX 2.3 — text-to-video and image-to-video.

Architecture:
  - Lightricks LTX 2.3 (22B) via ComfyUI native support
  - Two-pass generation: low-res (9-step) -> latent upscale 2x -> high-res (4-step)
  - Audio VAE: separate video + audio latent pipelines, combined at output
  - Distilled LoRA applied at strength 0.5

Input schema (via RunPod serverless job):
  {
    "input": {
      "prompt": "A serene mountain lake at dawn",       // REQUIRED
      "width": 1280,                                    // optional — output width
      "height": 720,                                    // optional — output height
      "duration": 5,                                    // optional — seconds
      "fps": 25,                                        // optional — frames per second
      "seed": null,                                     // optional — null = random
      "image_url": null                                 // optional — I2V mode
    }
  }

Output:
  {
    "video_b64": "<base64-encoded MP4>",
    "audio_b64": "<base64-encoded WAV>",
    "prompt": "...",
    "seed": 42,
    "wall_time_s": 12.3,
    "status": "COMPLETED"
  }

Gemma prompt enhancement: NOT YET WIRED.
  — The input schema accepts `prompt` directly so the Studio can pass it without Gemma.
  — To add Gemma enhancement later: wire TextGenerateLTX2Prompt + ComfySwitchNode
    before the CLIPTextEncode in the workflow below. No input schema change needed.
"""

import base64
import json
import os
import random
import time
import requests as req
import traceback


# ── Workflow Builder ──────────────────────────────────────────────────────────

def build_t2v_workflow(
    prompt: str,
    negative_prompt: str = "pc game, console game, video game, cartoon, childish, ugly",
    width: int = 1280,
    height: int = 720,
    duration: int = 5,
    fps: int = 25,
    seed: int | None = None,
) -> dict:
    """
    Build an LTX 2.3 T2V workflow in ComfyUI API format.

    Two-pass architecture validated against official Comfy-Org template
    (video_ltx2_3_t2v.json rev 0).

    Pass 1: Low-resolution at half dimensions, 9-step euler sampling
    Pass 2: Latent upscale 2x, then 4-step euler sampling at full resolution
    Audio: Separate VAE pipeline, combined via LTXVConcatAVLatent
    """
    if seed is None:
        seed = random.randint(0, 2 ** 31 - 1)

    w = width // 2   # half-res for pass 1
    h = height // 2
    length = duration * fps + 1

    workflow = {
        # ── Model Loading ──
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
                "strength_model": 0.5,
            },
        },
        "3": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder_name": "gemma_3_12B_it_fp4_mixed.safetensors",
                "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors",
                "prompt_template": "default",
            },
        },
        "4": {
            "class_type": "LTXVAudioVAELoader",
            "inputs": {
                "model": ["1", 0],
            },
        },
        "5": {
            "class_type": "LatentUpscaleModelLoader",
            "inputs": {
                "model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            },
        },

        # ── Latent Creation ──
        "6": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": w,
                "height": h,
                "length": length,
                "batch_size": 1,
            },
        },
        "7": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "length": length,
                "fps": fps,
                "batch_size": 1,
            },
        },

        # ── Conditioning ──
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["3", 0],
            },
        },
        "9": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["3", 0],
            },
        },
        "10": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["8", 0],
                "negative": ["9", 0],
                "fps": fps,
            },
        },

        # ── Sampling Infrastructure (shared) ──
        "11": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "12": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                          "0.909375, 0.725, 0.421875, 0.0"
            },
        },
        "13": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["2", 0],
                "cfg": 1.0,
            },
        },
        "14": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "15": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "0.85, 0.725, 0.4219, 0.0"
            },
        },
        "16": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["2", 0],
                "cfg": 1.0,
            },
        },

        # ── Pass 1: Low-Resolution Sampling ──
        "17": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
        },
        "18": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["17", 0],
                "guider": ["13", 0],
                "sampler": ["11", 0],
                "sigmas": ["12", 0],
                "latent": ["6", 0],
            },
        },
        "19": {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "samples": ["18", 0],
                "strength": 0.7,
                "override": False,
            },
        },

        # ── Audio + Video Combine (Pass 1) ──
        "20": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {
                "video": ["19", 0],
                "audio": ["7", 0],
            },
        },
        "21": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {
                "av": ["20", 0],
            },
        },

        # ── Latent Upscale 2x ──
        "22": {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {
                "model": ["5", 0],
                "video": ["21", 0],
                "audio": ["21", 1],
            },
        },

        # ── Pass 2: High-Resolution Sampling ──
        "23": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
        },
        "24": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["23", 0],
                "guider": ["16", 0],
                "sampler": ["14", 0],
                "sigmas": ["15", 0],
                "latent": ["22", 0],
            },
        },
        "25": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {
                "av": ["24", 0],
            },
        },

        # ── Decode ──
        "26": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["25", 0],
                "vae": ["1", 2],
                "tile_size": 768,
                "overlap": 64,
                "threshold": 4096,
            },
        },
        "27": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {
                "audio": ["25", 1],
                "vae": ["4", 0],
            },
        },

        # ── Output ──
        "28": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["26", 0],
                "audio": ["27", 0],
                "fps": fps,
                "format": "video/mp4",
            },
        },
    }

    return workflow


def build_i2v_workflow(
    prompt: str,
    image_url: str,
    negative_prompt: str = "pc game, console game, video game, cartoon, childish, ugly",
    width: int = 1280,
    height: int = 720,
    duration: int = 5,
    fps: int = 25,
    seed: int | None = None,
) -> dict:
    """
    Build an LTX 2.3 I2V workflow.

    Same two-pass architecture as T2V but with LoadImage + LTXVPreprocess
    replacing EmptyLTXVLatentVideo.
    """
    if seed is None:
        seed = random.randint(0, 2 ** 31 - 1)

    w = width // 2
    h = height // 2
    length = duration * fps + 1

    workflow = {
        # ── Model Loading (same as T2V) ──
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors"},
        },
        "2": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": "ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors",
                "strength_model": 0.5,
            },
        },
        "3": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder_name": "gemma_3_12B_it_fp4_mixed.safetensors",
                "ckpt_name": "ltx-2.3-22b-dev-fp8.safetensors",
                "prompt_template": "default",
            },
        },
        "4": {
            "class_type": "LTXVAudioVAELoader",
            "inputs": {"model": ["1", 0]},
        },
        "5": {
            "class_type": "LatentUpscaleModelLoader",
            "inputs": {"model_name": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"},
        },

        # ── Image Preprocessing ──
        "6": {
            "class_type": "LoadImage",
            "inputs": {"url": image_url},
        },
        "7": {
            "class_type": "ResizeImagesByLongerEdge",
            "inputs": {
                "images": ["6", 0],
                "max_size": 1536,
            },
        },
        "8": {
            "class_type": "LTXVPreprocess",
            "inputs": {
                "images": ["7", 0],
                "num_frames": length,
            },
        },
        "9": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": w,
                "height": h,
                "length": length,
                "batch_size": 1,
            },
        },
        "10": {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "image": ["8", 0],
                "video": ["9", 0],
                "strength": 1.0,
                "override": False,
            },
        },

        # ── Audio Latent ──
        "11": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "length": length,
                "fps": fps,
                "batch_size": 1,
            },
        },

        # ── Conditioning (same as T2V) ──
        "12": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["3", 0]},
        },
        "13": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["3", 0]},
        },
        "14": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["12", 0],
                "negative": ["13", 0],
                "fps": fps,
            },
        },

        # ── Sampling Infrastructure ──
        "15": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "16": {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                          "0.909375, 0.725, 0.421875, 0.0"
            },
        },
        "17": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["2", 0], "cfg": 1.0},
        },

        # ── Pass 1: Low-Res Sampling ──
        "18": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "19": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["18", 0],
                "guider": ["17", 0],
                "sampler": ["15", 0],
                "sigmas": ["16", 0],
                "latent": ["10", 0],
            },
        },

        # ── AV Combine + Upscale ──
        "20": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video": ["19", 0], "audio": ["11", 0]},
        },
        "21": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av": ["20", 0]},
        },
        "22": {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {"model": ["5", 0], "video": ["21", 0], "audio": ["21", 1]},
        },

        # ── Pass 2: High-Res Sampling ──
        "23": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "24": {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": "0.85, 0.725, 0.4219, 0.0"},
        },
        "25": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["2", 0], "cfg": 1.0},
        },
        "26": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "27": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["26", 0],
                "guider": ["25", 0],
                "sampler": ["23", 0],
                "sigmas": ["24", 0],
                "latent": ["22", 0],
            },
        },
        "28": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av": ["27", 0]},
        },

        # ── Decode ──
        "29": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": ["28", 0],
                "vae": ["1", 2],
                "tile_size": 768,
                "overlap": 64,
                "threshold": 4096,
            },
        },
        "30": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"audio": ["28", 1], "vae": ["4", 0]},
        },

        # ── Output ──
        "31": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["29", 0],
                "audio": ["30", 0],
                "fps": fps,
                "format": "video/mp4",
            },
        },
    }

    return workflow


# ── ComfyUI Submission ────────────────────────────────────────────────────────

def run_comfy_job(workflow: dict, timeout: int = 600) -> dict:
    """
    Submit workflow to local ComfyUI, poll for completion, return output.

    ComfyUI is expected at the address in COMFY_NODES env var
    (default 127.0.0.1:8188).
    """
    comfy_addr = os.environ.get("COMFY_NODES", "127.0.0.1:8188")
    api_url = f"http://{comfy_addr}/prompt"

    # Queue the workflow
    resp = req.post(api_url, json={"prompt": workflow}, timeout=30)
    resp.raise_for_status()
    prompt_id = resp.json().get("prompt_id", "")
    print(f"[Worker] Queued workflow: prompt_id={prompt_id}", flush=True)

    # Poll for completion
    history_url = f"http://{comfy_addr}/history/{prompt_id}"
    deadline = time.time() + timeout
    result = {}

    while time.time() < deadline:
        time.sleep(1.0)
        try:
            hist = req.get(history_url, timeout=10)
            if hist.status_code == 200:
                data = hist.json()
                if data and prompt_id in data:
                    outputs = data[prompt_id].get("outputs", {})
                    for node_id, node_out in outputs.items():
                        if "video" in node_out or "gifs" in node_out:
                            result = node_out
                            break
                    if result:
                        print(f"[Worker] Job completed after {time.time() - deadline + timeout:.1f}s", flush=True)
                        return result
                    # Check for status info (errors etc.)
                    status = data[prompt_id].get("status", {})
                    if status.get("exec_info", {}).get("queue_remaining", -1) == 0:
                        # Queue is empty but no output — likely error
                        print(f"[Worker] Queue empty, no output produced", flush=True)
                        break
        except req.RequestException as e:
            print(f"[Worker] Poll error: {e}", flush=True)
            time.sleep(2.0)

    return result


# ── RunPod Serverless Handler ─────────────────────────────────────────────────

def handler(job):
    """
    RunPod serverless handler: text prompt -> base64 video.
    """
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "")

    if not prompt:
        return {"error": "Missing required field: prompt", "status": "FAILED"}

    # Parse parameters
    width = int(job_input.get("width", 1280))
    height = int(job_input.get("height", 720))
    duration = int(job_input.get("duration", 5))
    fps = int(job_input.get("fps", 25))
    seed_raw = job_input.get("seed", None)
    seed = int(seed_raw) if seed_raw is not None else None
    image_url = job_input.get("image_url", None)

    # Validate dimensions
    width = max(512, min(2048, width // 64 * 64))
    height = max(512, min(2048, height // 64 * 64))
    duration = max(2, min(30, duration))
    fps = max(8, min(30, fps))

    print(f"[Worker] Starting LTX 2.3 generation: prompt='{prompt[:80]}'", flush=True)
    print(f"  mode={'I2V' if image_url else 'T2V'}, {width}x{height}, "
          f"{duration}s @ {fps}fps, seed={seed or 'random'}", flush=True)

    t_start = time.time()

    try:
        # Build workflow
        if image_url:
            workflow = build_i2v_workflow(
                prompt=prompt, image_url=image_url,
                width=width, height=height,
                duration=duration, fps=fps, seed=seed,
            )
        else:
            workflow = build_t2v_workflow(
                prompt=prompt,
                width=width, height=height,
                duration=duration, fps=fps, seed=seed,
            )

        # Submit to ComfyUI
        result = run_comfy_job(workflow)

        if not result:
            wall_time = time.time() - t_start
            return {
                "error": "Generation produced no output — check ComfyUI logs",
                "wall_time_s": round(wall_time, 1),
                "status": "FAILED",
            }

        wall_time = time.time() - t_start
        print(f"[Worker] Generation complete in {wall_time:.1f}s", flush=True)

        # Encode result as base64
        video_data = result.get("video", "") or result.get("gifs", [{}])[0].get("data", "")

        if not video_data:
            return {
                "error": "Output missing video data",
                "wall_time_s": round(wall_time, 1),
                "status": "FAILED",
            }

        return {
            "video_b64": video_data if isinstance(video_data, str) else "",
            "prompt": prompt,
            "seed": seed or 0,
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "wall_time_s": round(wall_time, 1),
            "status": "COMPLETED",
        }

    except Exception as exc:
        traceback.print_exc()
        return {
            "error": f"LTX 2.3 generation failed: {str(exc)}",
            "traceback": traceback.format_exc(),
            "status": "FAILED",
        }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
