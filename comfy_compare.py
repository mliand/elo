import argparse
import json
import os
import time
import uuid
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

# Requires: websocket-client
import websocket  # type: ignore


def queue_prompt(server_address: str, client_id: str, prompt: dict) -> str:
    payload = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["prompt_id"]


def get_image(server_address: str, filename: str, subfolder: str, folder_type: str) -> bytes:
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()


def get_history(server_address: str, prompt_id: str) -> dict:
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())


def wait_for_completion(ws: websocket.WebSocket, prompt_id: str, timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while True:
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for ComfyUI execution to finish")
        out = ws.recv()
        if isinstance(out, str):
            try:
                message = json.loads(out)
            except Exception:
                continue
            if message.get("type") == "executing":
                data = message.get("data", {})
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    return  # done


def _find_first_node_id_by_class(workflow: dict, class_type: str) -> str | None:
    for nid, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return nid
    return None


def _get_ref_node_id(workflow: dict, node_id: str, input_key: str) -> str | None:
    node = workflow.get(node_id, {})
    ref = node.get("inputs", {}).get(input_key)
    if isinstance(ref, list) and len(ref) >= 1 and isinstance(ref[0], str):
        return ref[0]
    return None


def build_prompt_workflow(
    base_workflow: dict,
    model_name: str,
    prompt_text: str | None,
    negative_text: str | None,
    seed: int | None,
    width: int | None,
    height: int | None,
    steps: int | None,
    cfg: float | None,
    sampler_name: str | None,
    scheduler: str | None,
) -> dict:
    wf = json.loads(json.dumps(base_workflow))  # deep copy

    # 1) Locate the KSampler node (assume first one)
    ksampler_id = _find_first_node_id_by_class(wf, "KSampler")
    if not ksampler_id:
        raise ValueError("KSampler node not found in workflow")

    # 2) From KSampler, find references to model, positive, negative, and latent_image nodes
    ckpt_node_id = _get_ref_node_id(wf, ksampler_id, "model")
    pos_node_id = _get_ref_node_id(wf, ksampler_id, "positive")
    neg_node_id = _get_ref_node_id(wf, ksampler_id, "negative")
    latent_node_id = _get_ref_node_id(wf, ksampler_id, "latent_image")

    # 3) Set checkpoint (prefer via referenced node, fallback to first CheckpointLoaderSimple)
    target_ckpt_id = ckpt_node_id or _find_first_node_id_by_class(wf, "CheckpointLoaderSimple")
    if not target_ckpt_id:
        raise ValueError("CheckpointLoaderSimple node not found in workflow")
    wf[target_ckpt_id].setdefault("inputs", {})["ckpt_name"] = model_name

    # 4) Update prompt and negative prompt text via the nodes feeding KSampler
    if pos_node_id and prompt_text is not None:
        wf[pos_node_id].setdefault("inputs", {})["text"] = prompt_text
    if neg_node_id and negative_text is not None:
        wf[neg_node_id].setdefault("inputs", {})["text"] = negative_text

    # 5) Update KSampler settings
    kin = wf[ksampler_id].setdefault("inputs", {})
    if seed is not None:
        kin["seed"] = seed
    if steps is not None:
        kin["steps"] = steps
    if cfg is not None:
        kin["cfg"] = cfg
    if sampler_name is not None:
        kin["sampler_name"] = sampler_name
    if scheduler is not None:
        kin["scheduler"] = scheduler

    # 6) Update latent image size if applicable
    if latent_node_id and latent_node_id in wf and wf[latent_node_id].get("class_type") == "EmptyLatentImage":
        lin = wf[latent_node_id].setdefault("inputs", {})
        if width is not None:
            lin["width"] = width
        if height is not None:
            lin["height"] = height

    return wf


def run_once(
    server: str,
    workflow: dict,
    out_dir: Path,
    model_name_for_naming: str,
    base_name: str | None = None,
    unified_core_only: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client_id = str(uuid.uuid4())

    ws = websocket.WebSocket()
    ws.connect(f"ws://{server}/ws?clientId={client_id}")

    prompt_id = queue_prompt(server, client_id, workflow)
    wait_for_completion(ws, prompt_id)

    history = get_history(server, prompt_id)[prompt_id]
    saved_paths: list[Path] = []
    ts = datetime.now().strftime("%Y%m%d%H%M%S")

    outputs = history.get("outputs", {})
    for node_id, node_output in outputs.items():
        # Images
        for kind in ("images", "videos"):
            if kind not in node_output:
                continue
            for idx, asset in enumerate(node_output[kind]):
                blob = get_image(server, asset["filename"], asset["subfolder"], asset["type"])
                # Try to preserve extension from filename
                _, ext = os.path.splitext(asset["filename"]) or ("", ".png")
                ext = ext if ext else ".png"
                core = base_name if base_name else model_name_for_naming
                if unified_core_only:
                    fname = f"{core}_{ts}_{node_id}_{idx}{ext}"
                else:
                    fname = f"{core}_{model_name_for_naming}_{ts}_{node_id}_{idx}{ext}"
                path = out_dir / fname
                with open(path, "wb") as f:
                    f.write(blob)
                saved_paths.append(path)

    return saved_paths


def _slugify(text: str, max_len: int = 60) -> str:
    # Simple slug that keeps CJK and alphanumerics; replace spaces and punctuation with '_'
    import re
    s = re.sub(r"[\s\t\n]+", "_", text.strip())
    s = re.sub(r"[^\w\u4e00-\u9fff\-_.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def _save_caption_alongside(image_paths: list[Path], caption: str):
    for p in image_paths:
        txt_path = p.with_suffix(".txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
        except Exception:
            pass


def _make_pair_grid(img_a: Path, img_b: Path, out_path: Path):
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Pillow not installed. Please `pip install pillow`. ") from e

    im_a = Image.open(img_a).convert("RGB")
    im_b = Image.open(img_b).convert("RGB")
    h = max(im_a.height, im_b.height)
    # Scale shorter image to same height for neat grid
    def _resize_to_height(im, target_h):
        if im.height == target_h:
            return im
        w = int(im.width * (target_h / im.height))
        return im.resize((w, target_h), Image.LANCZOS)

    im_a = _resize_to_height(im_a, h)
    im_b = _resize_to_height(im_b, h)
    grid = Image.new("RGB", (im_a.width + im_b.width, h), (0, 0, 0))
    grid.paste(im_a, (0, 0))
    grid.paste(im_b, (im_a.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Compare ComfyUI outputs across multiple checkpoints.")
    parser.add_argument("--server", default="10.30.100.201:8844", help="ComfyUI server host:port")
    parser.add_argument("--workflow", required=True, help="Path to workflow api JSON")
    parser.add_argument("--out", default="output", help="Output directory")
    parser.add_argument("--prompt", default=None, help="Positive prompt text")
    parser.add_argument("--negative", default=None, help="Negative prompt text")
    parser.add_argument("--seed", type=int, default=None, help="Seed")
    parser.add_argument("--width", type=int, default=None, help="Width (if EmptyLatentImage present)")
    parser.add_argument("--height", type=int, default=None, help="Height (if EmptyLatentImage present)")
    parser.add_argument("--steps", type=int, default=None, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=None, help="CFG scale")
    parser.add_argument("--sampler", default=None, help="Sampler name (e.g., dpmpp_2m)")
    parser.add_argument("--scheduler", default=None, help="Scheduler (e.g., sgm_uniform)")
    # Two-model quick compare mode
    parser.add_argument("--model-a", dest="model_a", default=None, help="Checkpoint A filename")
    parser.add_argument("--model-b", dest="model_b", default=None, help="Checkpoint B filename")
    # Multi-model mode (fallback)
    parser.add_argument("--models", nargs="+", default=None, help="List of checkpoint filenames to test")
    # Extras
    parser.add_argument("--save-caption", action="store_true", help="Save prompt as .txt next to image")
    parser.add_argument("--pair-grid", action="store_true", help="Create side-by-side grid for two-model mode")
    parser.add_argument("--pairs-out", default=None, help="Output dir for pair grids (default: <out>/pairs)")

    args = parser.parse_args()
    out_base = Path(args.out)

    with open(args.workflow, "r", encoding="utf-8") as f:
        base_wf = json.load(f)

    # Determine mode
    two_model_mode = args.model_a is not None and args.model_b is not None
    if two_model_mode:
        models = [args.model_a, args.model_b]
    else:
        if not args.models:
            raise ValueError("Specify either --model-a and --model-b, or --models ...")
        models = args.models

    prompt_for_name = args.prompt or "prompt"
    base_name = _slugify(f"{prompt_for_name}_{args.seed if args.seed is not None else ''}")

    all_saved: list[Path] = []
    per_model_saved: dict[str, list[Path]] = {}
    for model in models:
        wf = build_prompt_workflow(
            base_wf,
            model_name=model,
            prompt_text=args.prompt,
            negative_text=args.negative,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
            sampler_name=args.sampler,
            scheduler=args.scheduler,
        )
        model_safe = Path(model).stem
        out_dir = out_base / model_safe
        saved = run_once(args.server, wf, out_dir, model_safe, base_name=base_name)
        if args.save_caption and args.prompt is not None:
            _save_caption_alongside(saved, args.prompt)
        per_model_saved[model_safe] = saved
        all_saved.extend(saved)

    # If two-model mode and both produced at least one image, create a grid
    if two_model_mode and args.pair_grid:
        a_key = Path(args.model_a).stem
        b_key = Path(args.model_b).stem
        imgs_a = per_model_saved.get(a_key, [])
        imgs_b = per_model_saved.get(b_key, [])
        if imgs_a and imgs_b:
            # Take the first from each model to compose
            grid_dir = Path(args.pairs_out) if args.pairs_out else out_base / "pairs"
            grid_name = f"{base_name}_{a_key}_vs_{b_key}.png"
            grid_path = grid_dir / grid_name
            _make_pair_grid(imgs_a[0], imgs_b[0], grid_path)
            print(f"Pair grid: {grid_path}")

    print("Saved files:")
    for p in all_saved:
        print(p)


if __name__ == "__main__":
    main()
