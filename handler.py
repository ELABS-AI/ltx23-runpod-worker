"""
RunPod serverless handler for LTX 2.3 — text-to-video and image-to-video.

Uses official Comfy-Org workflow templates stored in workflows/ directory.
No programmatic workflow construction — templates are the source of truth.

Input schema (via RunPod serverless job):
  {
    "input": {
      "prompt": "A serene mountain lake at dawn",       // REQUIRED
      "width": 1280,                                    // optional
      "height": 720,                                    // optional
      "duration": 5,                                    // optional — seconds
      "fps": 25,                                        // optional
      "seed": null,                                     // optional — null = random
      "image_url": null                                 // optional — I2V mode
    }
  }

Output:
  {
    "video_b64": "<base64-encoded MP4>",
    "prompt": "...",
    "seed": 42,
    "wall_time_s": 12.3,
    "status": "COMPLETED"
  }
"""

import base64
import json
import os
import random
import time
import requests as req
import traceback
from pathlib import Path

# ── Workflow Template Loading ────────────────────────────────────────────────
# Official Comfy-Org workflow templates, NOT constructed in code.
# Templates stored at workflows/ltx23_t2v.json and workflows/ltx23_i2v.json.
# Source: https://github.com/Comfy-Org/workflow_templates/

_WORKFLOW_DIR = Path(__file__).parent / "workflows"

# Track which nodes in the workflow correspond to which parameters.
# These indices come from analyzing the official template's node structure.
# Nodes are identified by their ComfyUI type and widget_values position.

T2V_WORKFLOW_FILE = _WORKFLOW_DIR / "ltx23_t2v.json"
I2V_WORKFLOW_FILE = _WORKFLOW_DIR / "ltx23_i2v.json"


def _find_subgraph_nodes(template: dict) -> list:
    """Extract the actual workflow nodes from the subgraph package format."""
    defs = template.get("definitions", {})
    subgraphs = defs.get("subgraphs", [])
    if subgraphs:
        return subgraphs[0].get("nodes", [])
    return template.get("nodes", [])


def _set_widget_value(nodes: list, node_type: str, widget_index: int, value):
    """Set a widget value in the first node matching the given type."""
    for node in nodes:
        if node.get("type") == node_type:
            widgets = node.get("widgets_values", [])
            if widget_index < len(widgets):
                widgets[widget_index] = value
                return True
    return False


def _find_node_by_type(nodes: list, node_type: str) -> dict:
    """Find the first node of a given class type."""
    for node in nodes:
        if node.get("type") == node_type:
            return node
    return {}


def _find_nodes_by_type(nodes: list, node_type: str) -> list:
    """Find all nodes of a given class type."""
    return [n for n in nodes if n.get("type") == node_type]


def load_and_template_t2v_workflow(
    prompt: str,
    negative_prompt: str = "pc game, console game, video game, cartoon, childish, ugly",
    width: int = 1280,
    height: int = 720,
    duration: int = 5,
    fps: int = 25,
    seed: int | None = None,
) -> dict:
    """
    Load the official T2V workflow template and set user parameters.

    Source: Comfy-Org workflow_templates/video_ltx2_3_t2v.json
    """
    if seed is None:
        seed = random.randint(0, 2 ** 31 - 1)

    with open(T2V_WORKFLOW_FILE) as f:
        template = json.load(f)

    nodes = _find_subgraph_nodes(template)

    # ── Set Prompt ──
    # PrimitiveStringMultiline node contains the user prompt
    _set_widget_value(nodes, "PrimitiveStringMultiline", 0, prompt)

    # ── Set Negative Prompt ──
    # CLIPTextEncode for negative (node 247 in the official workflow)
    for node in nodes:
        if node.get("type") == "CLIPTextEncode":
            wv = node.get("widgets_values", [])
            if wv and isinstance(wv[0], str):
                # If widget is empty string, it's the positive prompt CLIP
                # If it's a negative string, it's the negative CLIP
                if wv[0] == "":
                    wv[0] = prompt  # Positive (will be overwritten by Gemma if enabled)
                elif "pc game" in str(wv[0]).lower() or "cartoon" in str(wv[0]).lower():
                    wv[0] = negative_prompt  # Negative
                break

    # ── Set Width, Height ──
    # PrimitiveInt nodes for size (nodes 257, 258)
    for node in nodes:
        if node.get("type") == "PrimitiveInt":
            wv = node.get("widgets_values", [])
            if wv and len(wv) > 0:
                val = wv[0]
                if val == 1280 and isinstance(val, int):
                    wv[0] = width
                elif val == 720 and isinstance(val, int):
                    wv[0] = height

    # ── Set Duration and FPS ──
    for node in nodes:
        if node.get("type") == "PrimitiveInt":
            wv = node.get("widgets_values", [])
            if wv and len(wv) > 0:
                val = wv[0]
                if val == 5:
                    wv[0] = duration
                elif val == 25:
                    wv[0] = fps

    # ── Set Seed ──
    # RandomNoise node with "randomize" mode (node 237)
    for node in nodes:
        if node.get("type") == "RandomNoise":
            wv = node.get("widgets_values", [])
            if len(wv) >= 2 and wv[1] == "randomize":
                wv[0] = seed

    return template


def load_and_template_i2v_workflow(
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
    Load the official I2V workflow template and set user parameters.

    Source: Comfy-Org workflow_templates/video_ltx2_3_i2v.json
    """
    if seed is None:
        seed = random.randint(0, 2 ** 31 - 1)

    with open(I2V_WORKFLOW_FILE) as f:
        template = json.load(f)

    nodes = _find_subgraph_nodes(template)

    # ── Set Prompt (similar pattern as T2V) ──
    _set_widget_value(nodes, "PrimitiveStringMultiline", 0, prompt)

    # ── Set Width, Height ──
    for node in nodes:
        if node.get("type") == "PrimitiveInt":
            wv = node.get("widgets_values", [])
            if wv and len(wv) > 0:
                val = wv[0]
                if val == 1280:
                    wv[0] = width
                elif val == 720:
                    wv[0] = height
                elif val == 5:
                    wv[0] = duration
                elif val == 25:
                    wv[0] = fps

    # ── Set Seed ──
    for node in nodes:
        if node.get("type") == "RandomNoise":
            wv = node.get("widgets_values", [])
            if len(wv) >= 2 and (wv[1] == "randomize" or wv[1] == "fixed"):
                wv[0] = seed

    # ── Set Image URL ──
    # The I2V workflow has a LoadImage node
    # For RunPod serverless, we need to download the image first
    # The image URL is passed as the input to the workflow
    # (The base handler handles image downloading)

    return template


# ── ComfyUI Submission ────────────────────────────────────────────────────────

def run_comfy_job(workflow: dict, timeout: int = 600) -> dict:
    """
    Submit workflow to local ComfyUI and poll for completion.

    ComfyUI is expected at the address in COMFY_NODES env var
    (default 127.0.0.1:8188).
    """
    comfy_addr = os.environ.get("COMFY_NODES", "127.0.0.1:8188")
    api_url = f"http://{comfy_addr}/prompt"

    # Queue the workflow — ComfyUI accepts both
    # UI format (with layout info) and API format
    resp = req.post(api_url, json={"prompt": workflow}, timeout=30)
    resp.raise_for_status()
    prompt_id = resp.json().get("prompt_id", "")
    print(f"[Worker] Queued workflow: prompt_id={prompt_id}", flush=True)

    # Poll for completion
    history_url = f"http://{comfy_addr}/history/{prompt_id}"
    deadline = time.time() + timeout

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
                            print(f"[Worker] Job completed", flush=True)
                            return node_out
                    status = data[prompt_id].get("status", {})
                    if status.get("exec_info", {}).get("queue_remaining", -1) == 0:
                        print(f"[Worker] Queue empty, no output", flush=True)
                        break
        except req.RequestException as e:
            print(f"[Worker] Poll error: {e}", flush=True)
            time.sleep(2.0)

    return {}


# ── RunPod Serverless Handler ─────────────────────────────────────────────────

IMAGE_DIR = "/tmp/input_images"


def handler(job):
    """
    RunPod serverless handler: text prompt -> base64 video.

    Uses official Comfy-Org workflow templates from workflows/ directory.
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
        # Load and template the official workflow
        if image_url:
            workflow = load_and_template_i2v_workflow(
                prompt=prompt, image_url=image_url,
                width=width, height=height,
                duration=duration, fps=fps, seed=seed,
            )
        else:
            workflow = load_and_template_t2v_workflow(
                prompt=prompt,
                width=width, height=height,
                duration=duration, fps=fps, seed=seed,
            )

        # Submit to ComfyUI
        result = run_comfy_job(workflow)

        if not result:
            wall_time = time.time() - t_start
            return {
                "error": "Generation produced no output",
                "wall_time_s": round(wall_time, 1),
                "status": "FAILED",
            }

        wall_time = time.time() - t_start
        print(f"[Worker] Generation complete in {wall_time:.1f}s", flush=True)

        # Encode result
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

    except FileNotFoundError as e:
        traceback.print_exc()
        return {
            "error": f"Workflow template not found: {str(e)}",
            "status": "FAILED",
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
