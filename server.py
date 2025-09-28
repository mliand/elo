from __future__ import annotations
import io
import json
import mimetypes
import os
import random
import threading
import traceback
from http import HTTPStatus
import socket
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List
from urllib.parse import urlparse, parse_qs

from backend.elo import update_ratings
from backend.storage import DataStore, ImageRec
from backend.events import EventBroker
from backend.genqueue import GenQueue
import atexit


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"

# Default negative prompt for real-time generation
DEFAULT_RT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, "
    "fewer digits, cropped, worst quality, low quality, low score, bad score, "
    "average score, signature, watermark, username, blurry"
)


def rt_generate_impl(ds: DataStore,
                     prompt_text: str,
                     negative_text: str,
                     models: List[str],
                     *,
                     comfy_server: str,
                     workflow_path: str,
                     seed: int | None,
                     width: int,
                     height: int,
                     steps: int,
                     cfg: float,
                     sampler: str | None,
                     scheduler: str | None) -> Dict[str, Any]:
    try:
        import comfy_compare as cc  # type: ignore
    except Exception:
        raise RuntimeError("需要安装 websocket-client，且存在 comfy_compare.py")

    # Load workflow JSON from file path
    wf_path = Path(workflow_path)
    if not wf_path.is_absolute():
        wf_path = (BASE_DIR / workflow_path).resolve()
    if not wf_path.exists():
        raise RuntimeError("workflow 文件不存在")
    try:
        base_wf = json.loads(wf_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"workflow 读取失败: {e}")

    # decide a unified stem
    try:
        stem_core = cc._slugify(f"{(prompt_text or 'prompt').strip()}_{seed if seed is not None else ''}")
    except Exception:
        stem_core = (prompt_text or "prompt").strip().replace(" ", "_")[:40]
    from uuid import uuid4
    unified_stem = f"{stem_core}_{str(uuid4())[:8]}"

    # ensure caption saved
    cap_dir = (BASE_DIR / ds.config.get("caption_dir", "caption")).resolve()
    cap_dir.mkdir(parents=True, exist_ok=True)
    try:
        (cap_dir / f"{unified_stem}.txt").write_text((prompt_text or "").strip(), encoding="utf-8")
    except Exception:
        pass

    results: List[Dict[str, Any]] = []
    have: set[str] = set()
    for model in models:
        wf = cc.build_prompt_workflow(
            base_wf,
            model_name=model,
            prompt_text=prompt_text,
            negative_text=negative_text,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
        )
        model_safe = Path(model).stem
        out_model_dir = (BASE_DIR / ds.config.get("models_dir", "models") / model_safe).resolve()
        out_model_dir.mkdir(parents=True, exist_ok=True)
        if model_safe in have:
            continue
        saved_paths = cc.run_once(
            comfy_server,
            wf,
            out_model_dir,
            model_safe,
            base_name=unified_stem,
            unified_core_only=True,
        )
        if not saved_paths:
            raise RuntimeError("未收到生成图像")
        first = saved_paths[0]
        ext = first.suffix or ".png"
        final_name = f"{unified_stem}{ext}"
        final_path = out_model_dir / final_name
        try:
            if first != final_path:
                if final_path.exists():
                    from uuid import uuid4
                    final_name = f"{unified_stem}_{str(uuid4())[:4]}{ext}"
                    final_path = out_model_dir / final_name
                first.replace(final_path)
        except Exception:
            try:
                data = first.read_bytes()
                final_path.write_bytes(data)
            except Exception:
                pass
        have.add(model_safe)
        results.append({
            "model": model_safe,
            "filename": final_path.name,
            "relpath": f"/local/model/{model_safe}/{final_path.name}",
            "rating": ds.get_rating(f"model:{model_safe}"),
        })

    by_model = {r["model"]: r for r in results}
    mlist = list(by_model.keys())
    if len(mlist) < 2:
        raise RuntimeError("生成结果不足两张")
    # pick 2 randomly; balancing is less relevant for fresh gens
    import random as _r
    ma, mb = (mlist[0], mlist[1]) if len(mlist) == 2 else _r.sample(mlist, 2)
    a_rec = {
        "id": f"model:{ma}",
        "model": ma,
        "stem": unified_stem,
        "filename": by_model[ma]["filename"],
        "relpath": by_model[ma]["relpath"],
        "rating": ds.get_rating(f"model:{ma}"),
    }
    b_rec = {
        "id": f"model:{mb}",
        "model": mb,
        "stem": unified_stem,
        "filename": by_model[mb]["filename"],
        "relpath": by_model[mb]["relpath"],
        "rating": ds.get_rating(f"model:{mb}"),
    }
    pairs = [{"a": a_rec, "b": b_rec}]
    return {
        "stem": unified_stem,
        "caption": (prompt_text or "").strip(),
        "results": results,
        "pairs": pairs,
    }


def build_handler(ds: DataStore, broker: EventBroker, genq: GenQueue | None = None):
    class AppHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        # ---------- helpers ----------
        def _json(self, status: int, obj: Any):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _bad(self, msg: str, code: int = 400):
            self._json(code, {"error": msg})

        def _parse_json_body(self) -> Any:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return None
            data = self.rfile.read(length)
            try:
                return json.loads(data.decode("utf-8"))
            except Exception:
                return None

        def _serve_image(self, path_part: str):
            # path_part is the portion after /images/
            file_path = IMAGES_DIR / path_part
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            ctype, _ = mimetypes.guess_type(str(file_path))
            try:
                data = file_path.read_bytes()
            except Exception:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Failed to read file")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_local(self, kind: str, filename: str):
            cfg = ds.config
            base_dirname = cfg.get("a_dir") if kind == "a" else cfg.get("b_dir")
            base = (BASE_DIR / base_dirname).resolve()
            safe_name = filename.lstrip("/\\")
            target = (base / safe_name).resolve()
            if not str(target).startswith(str(base)):
                return self._bad("非法路径", 400)
            if not target.exists() or not target.is_file():
                return self._bad("文件不存在", 404)
            ctype, _ = mimetypes.guess_type(str(target))
            try:
                data = target.read_bytes()
            except Exception:
                return self._bad("读取失败", 500)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _serve_model_file(self, model: str, filename: str):
            cfg = ds.config
            models_dirname = cfg.get("models_dir", "models")
            base = (BASE_DIR / models_dirname / model).resolve()
            safe_name = filename.lstrip("/\\")
            target = (base / safe_name).resolve()
            if not str(target).startswith(str(base)):
                return self._bad("非法路径", 400)
            if not target.exists() or not target.is_file():
                return self._bad("文件不存在", 404)
            ctype, _ = mimetypes.guess_type(str(target))
            try:
                data = target.read_bytes()
            except Exception:
                return self._bad("读取失败", 500)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # ---------- cache helpers ----------
        def _caption_dir(self) -> Path:
            return (BASE_DIR / ds.config.get("caption_dir", "caption")).resolve()

        def _models_dir(self) -> Path:
            return (BASE_DIR / ds.config.get("models_dir", "models")).resolve()

        def _find_stem_by_caption_text(self, text: str) -> str | None:
            cap_dir = self._caption_dir()
            if not cap_dir.exists():
                return None
            text_norm = (text or "").strip()
            cand = []
            try:
                for f in cap_dir.glob("*.txt"):
                    try:
                        t = f.read_text(encoding='utf-8').strip()
                    except Exception:
                        t = ''
                    if t == text_norm:
                        try:
                            mtime = f.stat().st_mtime
                        except Exception:
                            mtime = 0
                        cand.append((mtime, f.stem))
            except Exception:
                return None
            if not cand:
                return None
            # pick most recent
            cand.sort(key=lambda x: x[0], reverse=True)
            return cand[0][1]

        def _existing_results_for_stem(self, stem: str, models: list[str]) -> list[Dict[str, Any]]:
            allowed = [".png", ".webp", ".jpg", ".jpeg", ".bmp"]
            mdir = self._models_dir()
            out = []
            for m in models:
                model_safe = Path(m).stem
                base = (mdir / model_safe).resolve()
                if not base.exists() or not base.is_dir():
                    continue
                found_path = None
                for ext in allowed:
                    p = base / f"{stem}{ext}"
                    if p.exists() and p.is_file():
                        found_path = p
                        break
                if not found_path:
                    # try any file that startswith stem (in case of suffix variants)
                    try:
                        for f in base.iterdir():
                            if f.is_file() and f.name.startswith(stem) and f.suffix.lower() in allowed:
                                found_path = f
                                break
                    except Exception:
                        pass
                if found_path:
                    out.append({
                        "model": model_safe,
                        "filename": found_path.name,
                        "relpath": f"/local/model/{model_safe}/{found_path.name}",
                        "rating": ds.get_rating(f"model:{model_safe}"),
                    })
            return out

        # ---------- API ----------
        def do_GET(self):
            parsed = urlparse(self.path)
            # redirect root to real-time compare page by default
            if parsed.path in ("/", "/index.html"):
                self.send_response(302)
                self.send_header("Location", "/rt.html")
                self.end_headers()
                return
            if parsed.path in ("/admin.html", "/admin"):
                # hide admin on this port
                self.send_error(404, "Not Found")
                return
            if parsed.path.startswith("/api/"):
                try:
                    return self._handle_api_get(parsed)
                except Exception as e:
                    traceback.print_exc()
                    return self._bad("Server error", 500)
            if parsed.path.startswith("/images/"):
                return self._serve_image(parsed.path.split("/images/")[-1])
            if parsed.path.startswith("/local/a/"):
                return self._serve_local("a", parsed.path.split("/local/a/")[-1])
            if parsed.path.startswith("/local/b/"):
                return self._serve_local("b", parsed.path.split("/local/b/")[-1])
            if parsed.path.startswith("/local/model/"):
                # /local/model/<model>/<filename>
                parts = parsed.path.split("/local/model/")[-1].split("/", 1)
                if len(parts) == 2:
                    model, fname = parts
                    return self._serve_model_file(model, fname)
            # static files
            return super().do_GET()

        def _handle_api_get(self, parsed):
            path = parsed.path
            qs = parse_qs(parsed.query or "")
            if path == "/api/config":
                # Return current config and computed defaults
                return self._json(200, {"config": ds.config})
            if path == "/api/prompts":
                # Read prompt list from prompt.json at project root
                pfile = BASE_DIR / "prompt.json"
                if not pfile.exists():
                    return self._json(200, {"prompts": []})
                try:
                    data = json.loads(pfile.read_text(encoding="utf-8"))
                except Exception as e:
                    return self._bad(f"prompt.json 解析失败: {e}", 500)
                # support both array-of-strings and {prompts:[...]}
                prompts = []
                if isinstance(data, dict) and "prompts" in data:
                    prompts = data.get("prompts") or []
                elif isinstance(data, list):
                    prompts = data
                # normalize into list of {text, negative}
                norm = []
                for item in prompts:
                    if isinstance(item, str):
                        norm.append({"text": item, "negative": ""})
                    elif isinstance(item, dict):
                        norm.append({"text": str(item.get("text", "")), "negative": str(item.get("negative", ""))})
                return self._json(200, {"prompts": norm})
            if path == "/api/reload":
                # Reload ratings/history/config from disk
                try:
                    ds._load_all()
                    return self._json(200, {"ok": True, "message": "reloaded from disk"})
                except Exception as e:
                    return self._bad(f"reload failed: {e}", 500)
            if path == "/api/history":
                # return raw history list
                return self._json(200, {"history": ds.history})
            if path == "/api/images":
                # Provide a flat list of available model images (first per-stem found)
                stems = self._scan_model_stems()
                out = []
                for stem, m in stems.items():
                    for model, p in m.items():
                        out.append({
                            "id": f"{model}:{stem}",
                            "model": model,
                            "stem": stem,
                            "filename": os.path.basename(p),
                            "relpath": f"/local/model/{model}/{os.path.basename(p)}",
                            "rating": ds.get_rating(f"model:{model}"),
                        })
                return self._json(200, {"images": out})

            if path == "/api/ratings":
                # Build model-level leaderboard
                models = self._scan_models_list()
                appearances = self._count_model_appearances()
                items = []
                for model in models:
                    items.append((model, ds.get_rating(f"model:{model}"), appearances.get(model, 0)))
                items.sort(key=lambda t: t[1], reverse=True)
                ranks = []
                # pick a sample image for preview per model
                samples = self._pick_model_samples()
                for idx, (model, rating, apps) in enumerate(items, start=1):
                    sample = samples.get(model)
                    ranks.append({
                        "rank": idx,
                        "model": model,
                        "rating": rating,
                        "appearances": apps,
                        "relpath": sample["relpath"] if sample else "",
                        "filename": sample["filename"] if sample else model,
                    })
                return self._json(200, {"ratings": ranks})

            if path == "/api/next_pair":
                prompt = None
                try:
                    prompt = (qs.get('prompt') or [None])[0]
                except Exception:
                    prompt = None
                pair, cap = self._select_model_pair_with_caption(prompt)
                if not pair:
                    return self._bad("需要至少两张图片")
                a, b = pair
                return self._json(200, {"a": a, "b": b, "caption": cap})

            if path.startswith("/api/annotations/"):
                img_id = path.split("/api/annotations/")[-1]
                return self._json(200, {"id": img_id, "annotations": ds.get_annotations(img_id)})

            if path == "/api/export":
                models = self._scan_models_list()
                appearances = self._count_model_appearances()
                rows = ["model,rating,appearances"]
                for m in models:
                    rows.append(f"{m},{ds.get_rating('model:'+m):.3f},{appearances.get(m,0)}")
                csv = "\n".join(rows) + "\n"
                data = csv.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=ratings.csv")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            if path == "/api/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = broker.subscribe()
                try:
                    # send initial snapshot
                    snapshot = self._snapshot()
                    self._sse_event({"type": "snapshot", "payload": snapshot})
                    while True:
                        evt = q.get()
                        self._sse_event(evt)
                except BrokenPipeError:
                    pass
                except ConnectionResetError:
                    pass
                finally:
                    broker.unsubscribe(q)
                return

            return self._bad("Unknown endpoint", 404)

        def _sse_event(self, payload: Dict[str, Any]):
            data = json.dumps(payload, ensure_ascii=False)
            out = f"data: {data}\n\n".encode("utf-8")
            try:
                self.wfile.write(out)
                self.wfile.flush()
            except Exception:
                raise

        def _snapshot(self) -> Dict[str, Any]:
            models = self._scan_models_list()
            appearances = self._count_model_appearances()
            items = [(m, ds.get_rating(f"model:{m}"), appearances.get(m, 0)) for m in models]
            items.sort(key=lambda t: t[1], reverse=True)
            samples = self._pick_model_samples()
            ranks = []
            for idx, (model, rating, apps) in enumerate(items, start=1):
                smp = samples.get(model)
                ranks.append({
                    "rank": idx,
                    "model": model,
                    "rating": rating,
                    "appearances": apps,
                    "relpath": smp["relpath"] if smp else "",
                    "filename": smp["filename"] if smp else model,
                })
            return {"ratings": ranks}

        # ------ end helpers ------

        # shared small caches across handler instances
        _CACHE_TTL_SEC = 2.0
        _cache_lock = threading.RLock()
        _cache_stems: Dict[str, Dict[str, str]] | None = None
        _cache_stems_at: float = 0.0

        def _scan_model_stems(self) -> Dict[str, Dict[str, str]]:
            # returns mapping: stem -> { model -> file_path }
            cfg = ds.config
            models_dir = (BASE_DIR / cfg.get("models_dir", "models")).resolve()
            allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            # use TTL cache to reduce filesystem scans under load
            with self.__class__._cache_lock:
                if (self.__class__._cache_stems is not None) and (time.monotonic() - self.__class__._cache_stems_at < self.__class__._CACHE_TTL_SEC):
                    # return a shallow copy to avoid external mutation
                    return dict(self.__class__._cache_stems)
            stems: Dict[str, Dict[str, str]] = {}
            if models_dir.exists():
                for model_dir in models_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name
                    for f in model_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in allowed:
                            stem = f.stem
                            stems.setdefault(stem, {})[model] = str(f)
            else:
                # fallback to old a/b layout
                ab = {"a": (BASE_DIR / cfg.get("a_dir")).resolve(), "b": (BASE_DIR / cfg.get("b_dir")).resolve()}
                amap = {k: {f.stem: f for f in p.iterdir() if f.is_file() and f.suffix.lower() in allowed} if p.exists() else {} for k, p in ab.items()}
                common = set(amap["a"].keys()) & set(amap["b"].keys())
                for stem in common:
                    stems.setdefault(stem, {})["a"] = str(amap["a"][stem])
                    stems.setdefault(stem, {})["b"] = str(amap["b"][stem])
            # update cache
            with self.__class__._cache_lock:
                self.__class__._cache_stems = stems
                self.__class__._cache_stems_at = time.monotonic()
            return stems

        def _scan_models_list(self) -> List[str]:
            stems = self._scan_model_stems()
            models = set()
            for m in stems.values():
                models.update(m.keys())
            return sorted(models)

        def _pick_model_samples(self) -> Dict[str, Dict[str, str]]:
            # choose one sample image per model for preview
            stems = self._scan_model_stems()
            out: Dict[str, Dict[str, str]] = {}
            for stem, by_model in stems.items():
                for model, p in by_model.items():
                    if model not in out:
                        out[model] = {
                            "filename": os.path.basename(p),
                            "relpath": f"/local/model/{model}/{os.path.basename(p)}",
                        }
            return out

        def _count_model_appearances(self) -> Dict[str, int]:
            # use incremental counts maintained by DataStore
            try:
                return ds.get_model_appearances()
            except Exception:
                return {}

        def _pick_balanced_pair_models(self, model_names: list[str]) -> tuple[str, str]:
            """Pick two model names with lowest appearances to balance sampling.
            If multiple share the same minimal count, choose randomly among them.
            """
            import random as _r
            if not model_names or len(model_names) < 2:
                raise ValueError("需要至少两个模型用于配对")
            counts = self._count_model_appearances()
            def c(m: str) -> int:
                try: return int(counts.get(m, 0))
                except Exception: return 0
            remaining = list(model_names)
            # first pick among minimal count
            min1 = min(c(m) for m in remaining)
            cand1 = [m for m in remaining if c(m) == min1] or remaining
            m1 = _r.choice(cand1)
            remaining.remove(m1)
            # second pick among minimal of remaining
            min2 = min(c(m) for m in remaining)
            cand2 = [m for m in remaining if c(m) == min2] or remaining
            m2 = _r.choice(cand2)
            return (m1, m2)

        def _select_model_pair_with_caption(self, prompt: str | None = None) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]] | None, str]:
            stems = self._scan_model_stems()
            # choose stems with at least two models
            valid = [s for s, by_model in stems.items() if len(by_model) >= 2]
            if not valid:
                return None, ""
            stem = None
            if prompt:
                p = prompt.strip()
                if p:
                    # 1) try exact caption match; 2) substring; 3) exact stem; 4) substring stem
                    cap_dir = (BASE_DIR / ds.config.get("caption_dir")).resolve()
                    found = None
                    for s in valid:
                        cf = cap_dir / f"{s}.txt"
                        if cf.exists():
                            try:
                                txt = cf.read_text(encoding='utf-8').strip()
                            except Exception:
                                txt = ''
                            if txt == p:
                                found = s; break
                    if not found:
                        p_low = p.lower()
                        for s in valid:
                            cf = cap_dir / f"{s}.txt"
                            if cf.exists():
                                try:
                                    txt = cf.read_text(encoding='utf-8').strip().lower()
                                except Exception:
                                    txt = ''
                                if p_low in txt:
                                    found = s; break
                    if not found and p in valid:
                        found = p
                    if not found:
                        for s in valid:
                            if p.lower() in s.lower():
                                found = s; break
                    stem = found
            if not stem:
                stem = random.choice(valid)
            by_model = stems[stem]
            models = list(by_model.keys())
            m1, m2 = random.sample(models, 2)
            p1, p2 = by_model[m1], by_model[m2]
            a_rec = {
                "id": f"model:{m1}",
                "model": m1,
                "stem": stem,
                "filename": os.path.basename(p1),
                "relpath": f"/local/model/{m1}/{os.path.basename(p1)}",
                "rating": ds.get_rating(f"model:{m1}"),
            }
            b_rec = {
                "id": f"model:{m2}",
                "model": m2,
                "stem": stem,
                "filename": os.path.basename(p2),
                "relpath": f"/local/model/{m2}/{os.path.basename(p2)}",
                "rating": ds.get_rating(f"model:{m2}"),
            }
            cap_dir = (BASE_DIR / ds.config.get("caption_dir")).resolve()
            cap_file = cap_dir / f"{stem}.txt"
            caption = ""
            if cap_file.exists():
                try:
                    caption = cap_file.read_text(encoding="utf-8").strip()
                except Exception:
                    caption = ""
            return (a_rec, b_rec) if random.random() < 0.5 else (b_rec, a_rec), caption

        def do_POST(self):
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                return self._bad("POST not allowed", 405)
            try:
                return self._handle_api_post(parsed)
            except Exception:
                traceback.print_exc()
                return self._bad("Server error", 500)

        def do_OPTIONS(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith('/api/'):
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                return
            return super().do_OPTIONS()

        def _parse_multipart(self):
            """Parse multipart/form-data using email parser (Python stdlib).
            Returns list of {name, filename, data} for file parts.
            """
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                return []
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except Exception:
                length = 0
            body = self.rfile.read(length) if length > 0 else b""
            from email.parser import BytesParser
            from email.policy import default
            parser = BytesParser(policy=default)
            # Construct a minimal message for parsing
            wire = (f"Content-Type: {content_type}\r\n\r\n").encode("utf-8") + body
            try:
                msg = parser.parsebytes(wire)
            except Exception:
                return []
            out = []
            if msg.is_multipart():
                for part in msg.iter_parts():
                    cd = part.get("Content-Disposition", "")
                    if "form-data" not in cd:
                        continue
                    name = part.get_param("name", header="content-disposition")
                    filename = part.get_filename()
                    payload = part.get_payload(decode=True) or b""
                    if filename:  # only keep files
                        out.append({"name": name, "filename": filename, "data": payload})
            return out

        def _handle_api_post(self, parsed):
            path = parsed.path
            # ---- generation queue APIs ----
            if path == "/api/rt_enqueue":
                if genq is None:
                    return self._bad("生成队列未启用", 503)
                body = self._parse_json_body() or {}
                # allow direct prompt/negative or pick from prompt.json by index
                prompt_text = body.get("prompt")
                negative_text = body.get("negative") or ds.config.get("rt_negative") or DEFAULT_RT_NEGATIVE
                if not prompt_text:
                    # fallback from prompt.json if index provided
                    idx = body.get("prompt_index")
                    pfile = BASE_DIR / "prompt.json"
                    if pfile.exists():
                        try:
                            pdata = json.loads(pfile.read_text(encoding="utf-8"))
                            plist = pdata.get("prompts") if isinstance(pdata, dict) else pdata
                            if isinstance(plist, list):
                                if idx is not None:
                                    try:
                                        i = int(idx)
                                        pick = plist[i] if 0 <= i < len(plist) else None
                                    except Exception:
                                        pick = None
                                else:
                                    pick = plist[0] if plist else None
                                if isinstance(pick, str):
                                    prompt_text = pick
                                elif isinstance(pick, dict):
                                    prompt_text = str(pick.get("text", ""))
                                    if not negative_text:
                                        negative_text = str(pick.get("negative", ""))
                        except Exception:
                            pass
                models = body.get("models") or ds.config.get("rt_models") or []
                if isinstance(models, str):
                    models = [m.strip() for m in models.split(",") if m.strip()]
                if not isinstance(models, list):
                    return self._bad("models 参数需要为列表或逗号分隔字符串")
                models = [str(m).strip() for m in models if str(m).strip()]
                if len(models) < 2:
                    return self._bad("需要至少2个模型")
                comfy_server = (body.get("comfy_server") or ds.config.get("comfy_server") or "127.0.0.1:8188").strip()
                workflow_path = body.get("workflow") or ds.config.get("rt_workflow")
                if not workflow_path:
                    return self._bad("缺少 workflow 路径")
                # int/float params
                def _to_int(k, default=None):
                    v = body.get(k)
                    if v is None:
                        return default
                    try: return int(v)
                    except Exception: return default
                def _to_float(k, default=None):
                    v = body.get(k)
                    if v is None:
                        return default
                    try: return float(v)
                    except Exception: return default
                seed = _to_int("seed")
                width = _to_int("width", 1024)
                height = _to_int("height", 1024)
                steps = _to_int("steps", 30)
                cfg = _to_float("cfg", 4.5) or 4.5
                sampler = body.get("sampler")
                scheduler = body.get("scheduler")

                payload = {
                    "prompt": prompt_text or "",
                    "negative": negative_text or "",
                    "models": models,
                    "comfy_server": comfy_server,
                    "workflow": workflow_path,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler": sampler,
                    "scheduler": scheduler,
                }

                # submit to queue
                job_id = genq.submit(payload)
                return self._json(200, {"job_id": job_id, "queued": True})

            if path.startswith("/api/rt_job/"):
                if genq is None:
                    return self._bad("生成队列未启用", 503)
                job_id = path.split("/api/rt_job/")[-1]
                job = genq.get(job_id)
                if not job:
                    return self._bad("job 不存在", 404)
                return self._json(200, job)

            if path == "/api/rt_queue":
                if genq is None:
                    return self._json(200, {"enabled": False})
                return self._json(200, {"enabled": True, "stats": genq.stats()})
            if path == "/api/rt_next_from_prompts":
                body = self._parse_json_body() or {}
                # load prompt.json
                pfile = BASE_DIR / "prompt.json"
                if not pfile.exists():
                    return self._bad("prompt.json 不存在，请在项目根目录创建")
                try:
                    pdata = json.loads(pfile.read_text(encoding="utf-8"))
                except Exception as e:
                    return self._bad(f"prompt.json 解析失败: {e}")
                prompts = []
                if isinstance(pdata, dict) and "prompts" in pdata:
                    prompts = pdata.get("prompts") or []
                elif isinstance(pdata, list):
                    prompts = pdata
                # choose prompt
                idx = body.get("prompt_index")
                pick = None
                if idx is not None:
                    try:
                        idx = int(idx)
                        if 0 <= idx < len(prompts):
                            pick = prompts[idx]
                    except Exception:
                        pick = None
                if pick is None:
                    if not prompts:
                        return self._bad("prompt.json 中没有任何提示")
                    pick = random.choice(prompts)
                if isinstance(pick, str):
                    prompt_text = pick
                    negative_text = ds.config.get("rt_negative") or DEFAULT_RT_NEGATIVE
                else:
                    prompt_text = str(pick.get("text", ""))
                    negative_text = str(pick.get("negative", ""))
                    if not negative_text:
                        negative_text = ds.config.get("rt_negative") or DEFAULT_RT_NEGATIVE

                # models input (two or three), fallback to config
                models = body.get("models") or ds.config.get("rt_models") or []
                if isinstance(models, str):
                    models = [m.strip() for m in models.split(",") if m.strip()]
                if not isinstance(models, list):
                    return self._bad("models 参数需要为列表或逗号分隔字符串")
                models = [str(m).strip() for m in models if str(m).strip()]
                if len(models) < 2:
                    return self._bad("需要至少2个模型")
                # allow 2+ models (no upper cap)

                comfy_server = (body.get("comfy_server") or ds.config.get("comfy_server") or "127.0.0.1:8188").strip()
                workflow_path = body.get("workflow") or ds.config.get("rt_workflow")
                if not workflow_path:
                    return self._bad("缺少 workflow 路径")

                try:
                    seed = int(body.get("seed")) if body.get("seed") is not None else None
                except Exception:
                    seed = None
                if seed is None:
                    try:
                        seed = random.randint(0, 2**31 - 1)
                    except Exception:
                        seed = 0

                width = body.get("width"); height = body.get("height")
                steps = body.get("steps"); cfg = body.get("cfg")
                sampler = body.get("sampler"); scheduler = body.get("scheduler")
                def _opt_int(x):
                    try:
                        return int(x) if x is not None and str(x) != '' else None
                    except Exception:
                        return None
                def _opt_float(x):
                    try:
                        return float(x) if x is not None and str(x) != '' else None
                    except Exception:
                        return None
                width = _opt_int(width); height = _opt_int(height)
                steps = _opt_int(steps); cfg = _opt_float(cfg)

                try:
                    import comfy_compare as cc  # type: ignore
                except Exception:
                    return self._bad("需要安装 websocket-client，且存在 comfy_compare.py")

                wf_file = (BASE_DIR / workflow_path).resolve() if not os.path.isabs(str(workflow_path)) else Path(workflow_path)
                if not wf_file.exists():
                    return self._bad("workflow 文件不存在")
                try:
                    base_wf = json.loads(wf_file.read_text(encoding="utf-8"))
                except Exception as e:
                    return self._bad(f"workflow 读取失败: {e}")

                # Try cache by exact prompt text
                unified_stem = None
                existing_results: List[Dict[str, Any]] = []
                stem_hit = self._find_stem_by_caption_text(prompt_text)
                if stem_hit:
                    existing_results = self._existing_results_for_stem(stem_hit, models)
                    # even if已有两张，也继续用该stem并补齐缺失模型，保证新模型也生成
                    unified_stem = stem_hit

                if unified_stem is None:
                    # unified stem from prompt + seed
                    try:
                        stem_core = cc._slugify(f"{(prompt_text or 'prompt').strip()}_{seed if seed is not None else ''}")
                    except Exception:
                        stem_core = (prompt_text or "prompt").strip().replace(" ", "_")[:40]
                    from uuid import uuid4
                    unified_stem = f"{stem_core}_{str(uuid4())[:8]}"

                # save caption
                cap_dir = (BASE_DIR / ds.config.get("caption_dir", "caption")).resolve()
                cap_dir.mkdir(parents=True, exist_ok=True)
                try:
                    (cap_dir / f"{unified_stem}.txt").write_text((prompt_text or "").strip(), encoding="utf-8")
                except Exception:
                    pass

                results = list(existing_results) if existing_results else []
                have = {r["model"] for r in results}
                for model in models:
                    try:
                        wf = cc.build_prompt_workflow(
                            base_wf,
                            model_name=model,
                            prompt_text=prompt_text,
                            negative_text=negative_text,
                            seed=seed,
                            width=width,
                            height=height,
                            steps=steps,
                            cfg=cfg,
                            sampler_name=sampler,
                            scheduler=scheduler,
                        )
                    except Exception as e:
                        return self._bad(f"构建工作流失败: {e}")

                    model_safe = Path(model).stem
                    out_model_dir = (BASE_DIR / ds.config.get("models_dir", "models") / model_safe).resolve()
                    out_model_dir.mkdir(parents=True, exist_ok=True)
                    if model_safe in have:
                        continue  # cached
                    try:
                        saved_paths = cc.run_once(
                            comfy_server,
                            wf,
                            out_model_dir,
                            model_safe,
                            base_name=unified_stem,
                            unified_core_only=True,
                        )
                    except Exception as e:
                        return self._bad(f"生成失败: {e}")
                    if not saved_paths:
                        return self._bad("未收到生成图像")
                    first = saved_paths[0]
                    ext = first.suffix or ".png"
                    final_name = f"{unified_stem}{ext}"
                    final_path = out_model_dir / final_name
                    try:
                        if first != final_path:
                            if final_path.exists():
                                final_name = f"{unified_stem}_{str(uuid4())[:4]}{ext}"
                                final_path = out_model_dir / final_name
                            first.replace(final_path)
                    except Exception:
                        try:
                            data = first.read_bytes()
                            final_path.write_bytes(data)
                        except Exception:
                            pass
                    results.append({
                        "model": model_safe,
                        "filename": final_path.name,
                        "relpath": f"/local/model/{model_safe}/{final_path.name}",
                        "rating": ds.get_rating(f"model:{model_safe}"),
                    })

                # build a single random pair
                by_model = {r["model"]: r for r in results}
                mlist = list(by_model.keys())
                if len(mlist) < 2:
                    return self._bad("生成结果不足两张")
                try:
                    ma, mb = self._pick_balanced_pair_models(mlist)
                except Exception:
                    import random as _r
                    ma, mb = _r.sample(mlist, 2)
                a_rec = {
                    "id": f"model:{ma}",
                    "model": ma,
                    "stem": unified_stem,
                    "filename": by_model[ma]["filename"],
                    "relpath": by_model[ma]["relpath"],
                    "rating": ds.get_rating(f"model:{ma}"),
                }
                b_rec = {
                    "id": f"model:{mb}",
                    "model": mb,
                    "stem": unified_stem,
                    "filename": by_model[mb]["filename"],
                    "relpath": by_model[mb]["relpath"],
                    "rating": ds.get_rating(f"model:{mb}"),
                }
                pairs = [{"a": a_rec, "b": b_rec}]

                return self._json(200, {
                    "stem": unified_stem,
                    "caption": (prompt_text or "").strip(),
                    "results": results,
                    "pairs": pairs,
                })
            if path == "/api/rt_generate":
                body = self._parse_json_body() or {}
                try:
                    # Lazy import to keep optional dependency local
                    import comfy_compare as cc  # type: ignore
                except Exception:
                    return self._bad("需要安装 websocket-client，且存在 comfy_compare.py")

                comfy_server = (body.get("comfy_server") or ds.config.get("comfy_server") or "127.0.0.1:8188").strip()
                workflow_path = body.get("workflow") or ds.config.get("rt_workflow")
                models = body.get("models") or ds.config.get("rt_models") or []
                if not isinstance(models, list):
                    return self._bad("models 参数需要为列表")
                # allow 2+ models (no upper cap)
                models = [str(m).strip() for m in models if str(m).strip()]
                if len(models) < 2:
                    return self._bad("需要至少2个模型名称")
                try:
                    seed = int(body.get("seed")) if body.get("seed") is not None else None
                except Exception:
                    seed = None
                if seed is None:
                    try:
                        seed = random.randint(0, 2**31 - 1)
                    except Exception:
                        seed = 0
                prompt = body.get("prompt")
                negative = body.get("negative") or ds.config.get("rt_negative") or DEFAULT_RT_NEGATIVE
                width = body.get("width"); height = body.get("height")
                steps = body.get("steps"); cfg = body.get("cfg")
                sampler = body.get("sampler"); scheduler = body.get("scheduler")
                # optional numeric conversions
                def _opt_int(x):
                    try:
                        return int(x) if x is not None and str(x) != '' else None
                    except Exception:
                        return None
                def _opt_float(x):
                    try:
                        return float(x) if x is not None and str(x) != '' else None
                    except Exception:
                        return None
                width = _opt_int(width); height = _opt_int(height)
                steps = _opt_int(steps); cfg = _opt_float(cfg)

                if not workflow_path:
                    return self._bad("缺少 workflow 路径")
                wf_file = (BASE_DIR / workflow_path).resolve() if not os.path.isabs(str(workflow_path)) else Path(workflow_path)
                if not wf_file.exists():
                    return self._bad("workflow 文件不存在")

                # load workflow
                try:
                    base_wf = json.loads(wf_file.read_text(encoding="utf-8"))
                except Exception as e:
                    return self._bad(f"workflow 读取失败: {e}")

                # cache lookup by exact prompt text
                unified_stem = None
                existing_results: List[Dict[str, Any]] = []
                if prompt and str(prompt).strip():
                    stem_hit = self._find_stem_by_caption_text(str(prompt).strip())
                    if stem_hit:
                        existing_results = self._existing_results_for_stem(stem_hit, models)
                        # 不管已有几张，都复用该 stem 并补齐缺失模型
                        unified_stem = stem_hit

                if unified_stem is None:
                    # choose a unified stem for this generation: prompt + seed + short uuid
                    try:
                        stem_core = cc._slugify(f"{(prompt or 'prompt').strip()}_{seed if seed is not None else ''}")
                    except Exception:
                        # simple fallback
                        stem_core = (prompt or "prompt").strip().replace(" ", "_")[:40]
                    from uuid import uuid4
                    unified_stem = f"{stem_core}_{str(uuid4())[:8]}"

                # ensure caption saved for stem so UI can display prompt
                cap_dir = (BASE_DIR / ds.config.get("caption_dir", "caption")).resolve()
                cap_dir.mkdir(parents=True, exist_ok=True)
                try:
                    (cap_dir / f"{unified_stem}.txt").write_text((prompt or "").strip(), encoding="utf-8")
                except Exception:
                    pass

                try:
                    out = rt_generate_impl(
                        ds,
                        prompt_text=(prompt or ""),
                        negative_text=(negative or (ds.config.get("rt_negative") or DEFAULT_RT_NEGATIVE)),
                        models=models,
                        comfy_server=comfy_server,
                        workflow_path=workflow_path,
                        seed=seed,
                        width=width,
                        height=height,
                        steps=steps,
                        cfg=cfg,
                        sampler=sampler,
                        scheduler=scheduler,
                    )
                except Exception as e:
                    return self._bad(f"生成失败: {e}")
                return self._json(200, out)
            if path == "/api/reset":
                body = self._parse_json_body() or {}
                scope = (body.get("scope") or "ratings").lower()  # ratings | all
                # optionally adjust initial rating
                if "initial_rating" in body:
                    try:
                        ds.config["initial_rating"] = float(body["initial_rating"])
                        ds._save_config()
                    except Exception:
                        pass
                if scope == "all":
                    ds.ratings = {}
                    ds.history = []
                    try:
                        ds.clear_model_appearances()
                    except Exception:
                        pass
                    ds._save_ratings(); ds._save_history()
                    return self._json(200, {"ok": True, "reset": "all"})
                else:
                    ds.ratings = {}
                    ds._save_ratings()
                    return self._json(200, {"ok": True, "reset": "ratings"})
            if path == "/api/config":
                body = self._parse_json_body() or {}
                changed = {}
                for key in ("initial_rating", "k_factor", "a_dir", "b_dir", "caption_dir", "models_dir", "comfy_server", "rt_workflow", "rt_models", "rt_negative", "flush_interval_sec", "flush_max_ops"):
                    if key in body and body[key] is not None:
                        ds.config[key] = body[key]
                        changed[key] = body[key]
                ds._save_config()
                # apply flush config live
                try:
                    if "flush_interval_sec" in changed:
                        ds.flush_interval_sec = float(changed["flush_interval_sec"])  # type: ignore[arg-type]
                    if "flush_max_ops" in changed:
                        ds.flush_max_ops = int(changed["flush_max_ops"])  # type: ignore[arg-type]
                except Exception:
                    pass
                return self._json(200, {"ok": True, "changed": changed, "config": ds.config})
            if path == "/api/upload":
                files = self._parse_multipart()
                if not files:
                    return self._bad("需要 multipart/form-data 文件字段")
                saved = []
                for part in files:
                    rec = ds.add_image_bytes(part["filename"], part["data"])
                    saved.append({**rec.__dict__, "rating": ds.get_rating(rec.id)})
                return self._json(200, {"saved": saved})

            if path == "/api/compare":
                body = self._parse_json_body() or {}
                # prefer model-level ids like model:<name>
                winner_id = body.get("winner_id") or ("model:" + body.get("winner_model", ""))
                loser_id = body.get("loser_id") or ("model:" + body.get("loser_model", ""))
                tie = bool(body.get("tie", False))
                if not winner_id or not loser_id or winner_id == loser_id:
                    return self._bad("参数错误：需要不同的 winner_id/loser_id")
                r_w = ds.get_rating(winner_id)
                r_l = ds.get_rating(loser_id)
                res = update_ratings(r_w, r_l, k=int(ds.config.get("k_factor", 32)), tie=tie)
                # apply updates and record history atomically
                hist_rec = {
                    "timestamp": dsr_iso(),
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "tie": tie,
                    "winner_rating_before": res.r1_before,
                    "loser_rating_before": res.r2_before,
                    "winner_rating_after": res.r1_after,
                    "loser_rating_after": res.r2_after,
                    "rating_change": res.delta,
                }
                try:
                    ds.apply_rating_update_and_history({winner_id: res.r1_after, loser_id: res.r2_after}, hist_rec)
                except Exception:
                    # fallback to legacy on any unexpected issue
                    ds.set_rating(winner_id, res.r1_after)
                    ds.set_rating(loser_id, res.r2_after)
                    ds.append_history(hist_rec)
                # broadcast update
                broker.broadcast({
                    "type": "rating_update",
                    "payload": {
                        "winner_id": winner_id,
                        "loser_id": loser_id,
                        "tie": tie,
                        "winner_after": res.r1_after,
                        "loser_after": res.r2_after,
                        "ratings": self._snapshot()["ratings"],
                    },
                })
                return self._json(200, {
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "tie": tie,
                    "winner_after": res.r1_after,
                    "loser_after": res.r2_after,
                })

            if path.startswith("/api/annotations/"):
                img_id = path.split("/api/annotations/")[-1]
                body = self._parse_json_body()
                if body is None:
                    return self._bad("需要 JSON body")
                ds.set_annotations(img_id, body)
                return self._json(200, {"ok": True})

            if path == "/api/import_ab":
                body = self._parse_json_body() or {}
                a_dir = Path(body.get("a_dir") or (BASE_DIR / "a")).resolve()
                b_dir = Path(body.get("b_dir") or (BASE_DIR / "b")).resolve()
                caption_dir = Path(body.get("caption_dir") or (BASE_DIR / "caption")).resolve()
                allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                def scan(p: Path):
                    return {f.stem: f for f in p.iterdir() if f.is_file() and f.suffix.lower() in allowed} if p.exists() else {}
                a_map = scan(a_dir)
                b_map = scan(b_dir)
                common = sorted(set(a_map.keys()) & set(b_map.keys()))
                imported = 0
                for stem in common:
                    for f in (a_map[stem], b_map[stem]):
                        try:
                            ds.add_image_file(f)
                            imported += 1
                        except Exception:
                            pass
                return self._json(200, {"pairs": len(common), "images": imported})

            return self._bad("Unknown endpoint", 404)

    return AppHandler


def dsr_iso() -> str:
    # small helper to avoid importing iso_now back
    from datetime import datetime
    return datetime.now().isoformat()


def run(host: str = "0.0.0.0", port: int = 8765):
    ds = DataStore(BASE_DIR)
    broker = EventBroker()
    genq = GenQueue()

    # set the generation runner
    def _runner(payload: Dict[str, Any]) -> Dict[str, Any]:
        # construct a minimal helper handler to reuse implementation
        class _H:
            pass
        h = _H()
        # attach needed methods from enclosed build_handler via closure later
        # but we cannot call before Handler defined; so we duplicate minimal logic here by
        # reusing the same impl through a temporary AppHandler instance.
        # Instead, we define a tiny local function after Handler creation.
        raise RuntimeError("runner not attached")

    # Build handler first so we can bind the runner to its method.
    Handler = build_handler(ds, broker, genq)

    # now set runner to call shared impl
    def _runner(payload: Dict[str, Any]) -> Dict[str, Any]:
        # Extract params
        prompt_text = str(payload.get("prompt") or "")
        negative_text = str(payload.get("negative") or (ds.config.get("rt_negative") or DEFAULT_RT_NEGATIVE))
        models = list(payload.get("models") or [])
        comfy_server = str(payload.get("comfy_server") or ds.config.get("comfy_server") or "127.0.0.1:8188")
        workflow = str(payload.get("workflow") or (ds.config.get("rt_workflow") or ""))
        seed = payload.get("seed")
        width = int(payload.get("width") or 1024)
        height = int(payload.get("height") or 1024)
        steps = int(payload.get("steps") or 30)
        cfg = float(payload.get("cfg") or 4.5)
        sampler = payload.get("sampler")
        scheduler = payload.get("scheduler")
        return rt_generate_impl(
            ds,
            prompt_text, negative_text, models,
            comfy_server=comfy_server, workflow_path=workflow,
            seed=seed, width=width, height=height,
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler,
        )

    genq.set_runner(_runner)
    server = ThreadingHTTPServer((host, port), Handler)
    # ensure handler threads do not block shutdown
    try:
        server.daemon_threads = True  # type: ignore[attr-defined]
    except Exception:
        pass
    # flush data on process exit
    try:
        atexit.register(ds.close)
    except Exception:
        pass
    try:
        atexit.register(genq.close)
    except Exception:
        pass
    # try to detect a LAN IP for convenience
    def _detect_lan_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
            return ip
        except Exception:
            # fallback to hostname or loopback
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return "127.0.0.1"

    lan_ip = _detect_lan_ip()
    print(f"[elo] server listening on http://{host}:{port}")
    if host == "0.0.0.0":
        print(f"[elo] access on LAN:   http://{lan_ip}:{port}")
    print(f"[elo] web root: {WEB_DIR}")
    print(f"[elo] images dir: {IMAGES_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[elo] shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
