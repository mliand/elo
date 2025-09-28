from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .elo import DEFAULT_INITIAL_RATING


def iso_now() -> str:
    return datetime.now().isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any):
    if not path.exists():
        return default
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return default
        return json.loads(text)
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@dataclass
class ImageRec:
    id: str
    filename: str
    relpath: str  # web path, e.g. /images/uuid_name.png
    created_at: str


class DataStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.data_dir = self.base_dir / "data"
        self.images_dir = self.data_dir / "images"

        ensure_dir(self.data_dir)
        ensure_dir(self.images_dir)

        self.images_file = self.data_dir / "images.json"
        self.annotations_file = self.data_dir / "annotations.json"

        # Keep ratings/history files compatible with previous Gradio app
        self.ratings_file = self.base_dir / "ratings.json"
        self.history_file = self.base_dir / "history.json"
        self.config_file = self.base_dir / "config.json"

        self._lock = threading.RLock()

        # flush control
        self._dirty_images = False
        self._dirty_annotations = False
        self._dirty_ratings = False
        self._dirty_history = False
        self._dirty_config = False
        self._pending_ops = 0

        # runtime stats derived from history (not persisted)
        self._model_appearances: Dict[str, int] = {}

        # auto flush settings (may be overridden by config at load)
        self.flush_interval_sec = 1.0
        self.flush_max_ops = 64

        # background flusher
        self._stop_flag = False
        self._flusher: Optional[threading.Thread] = None

        # in-memory
        self.images: Dict[str, ImageRec] = {}
        self.annotations: Dict[str, Any] = {}
        self.ratings: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}

        self._load_all()
        self._start_flusher()

    # ---------- load/save ----------
    def _load_all(self) -> None:
        with self._lock:
            imgs = read_json(self.images_file, default={})
            self.images = {k: ImageRec(**v) for k, v in imgs.items()} if isinstance(imgs, dict) else {}
            self.annotations = read_json(self.annotations_file, default={})
            self.ratings = read_json(self.ratings_file, default={})
            self.history = read_json(self.history_file, default=[])
            self.config = read_json(self.config_file, default={}) or {}
            # defaults
            if "initial_rating" not in self.config:
                self.config["initial_rating"] = DEFAULT_INITIAL_RATING
            if "k_factor" not in self.config:
                self.config["k_factor"] = 64
            if "a_dir" not in self.config:
                self.config["a_dir"] = "a"
            if "b_dir" not in self.config:
                self.config["b_dir"] = "b"
            if "caption_dir" not in self.config:
                self.config["caption_dir"] = "caption"
            if "models_dir" not in self.config:
                self.config["models_dir"] = "models"
            # optional flush controls from config
            try:
                if "flush_interval_sec" in self.config:
                    self.flush_interval_sec = float(self.config.get("flush_interval_sec", 1.0))
                if "flush_max_ops" in self.config:
                    self.flush_max_ops = int(self.config.get("flush_max_ops", 64))
            except Exception:
                pass
            self._save_config()

            # rebuild appearances from history for quick stats
            self._rebuild_model_appearances_locked()

    def _save_images(self) -> None:
        write_json(self.images_file, {k: asdict(v) for k, v in self.images.items()})

    def _save_annotations(self) -> None:
        write_json(self.annotations_file, self.annotations)

    def _save_ratings(self) -> None:
        write_json(self.ratings_file, self.ratings)

    def _save_history(self) -> None:
        write_json(self.history_file, self.history)

    def _save_config(self) -> None:
        write_json(self.config_file, self.config)

    def _start_flusher(self) -> None:
        if self._flusher is not None:
            return
        def _run():
            import time
            while not self._stop_flag:
                time.sleep(max(0.1, float(self.flush_interval_sec)))
                try:
                    self.flush()
                except Exception:
                    # best-effort; avoid crashing background thread
                    pass
        t = threading.Thread(target=_run, name="DataStoreFlusher", daemon=True)
        t.start()
        self._flusher = t

    def close(self) -> None:
        self._stop_flag = True
        t = self._flusher
        if t:
            t.join(timeout=1.0)
        # final flush
        try:
            self.flush(force=True)
        except Exception:
            pass

    def flush(self, force: bool = False) -> None:
        """Persist dirty in-memory state to disk. Called periodically."""
        with self._lock:
            if not force and self._pending_ops < self.flush_max_ops and not (
                self._dirty_images or self._dirty_annotations or self._dirty_ratings or self._dirty_history or self._dirty_config
            ):
                return
            if self._dirty_images:
                self._save_images()
                self._dirty_images = False
            if self._dirty_annotations:
                self._save_annotations()
                self._dirty_annotations = False
            if self._dirty_ratings:
                self._save_ratings()
                self._dirty_ratings = False
            if self._dirty_history:
                self._save_history()
                self._dirty_history = False
            if self._dirty_config:
                self._save_config()
                self._dirty_config = False
            self._pending_ops = 0

    # ---------- images ----------
    def list_images(self) -> List[ImageRec]:
        with self._lock:
            return list(self.images.values())

    def add_image_bytes(self, filename: str, data: bytes) -> ImageRec:
        ext = Path(filename).suffix or ".bin"
        safe_name = Path(filename).name.replace(" ", "_")
        img_id = str(uuid.uuid4())
        stored_name = f"{img_id}_{safe_name}"
        relpath = f"/images/{stored_name}"
        with self._lock:
            # save file
            (self.images_dir / stored_name).write_bytes(data)
            rec = ImageRec(id=img_id, filename=safe_name, relpath=relpath, created_at=iso_now())
            self.images[img_id] = rec
            # set initial rating if not exists
            if img_id not in self.ratings:
                self.ratings[img_id] = float(self.config.get("initial_rating", DEFAULT_INITIAL_RATING))
                self._dirty_ratings = True
            self._dirty_images = True
            self._pending_ops += 1
            return rec

    def add_image_file(self, file_path: Path) -> ImageRec:
        file_path = Path(file_path)
        data = file_path.read_bytes()
        return self.add_image_bytes(file_path.name, data)

    def get_image_file(self, relpath: str) -> Path:
        # relpath like /images/<file>
        name = relpath.split("/images/")[-1]
        return self.images_dir / name

    # ---------- annotations ----------
    def get_annotations(self, img_id: str) -> Any:
        with self._lock:
            return self.annotations.get(img_id, [])

    def set_annotations(self, img_id: str, ann: Any) -> None:
        with self._lock:
            self.annotations[img_id] = ann
            self._dirty_annotations = True
            self._pending_ops += 1

    # ---------- ratings/history ----------
    def get_rating(self, img_id: str) -> float:
        with self._lock:
            return float(self.ratings.get(img_id, self.config.get("initial_rating", DEFAULT_INITIAL_RATING)))

    def set_rating(self, img_id: str, rating: float) -> None:
        with self._lock:
            self.ratings[img_id] = float(rating)
            self._dirty_ratings = True
            self._pending_ops += 1

    def append_history(self, rec: Dict[str, Any]) -> None:
        with self._lock:
            self.history.append(rec)
            self._update_model_appearances_from_rec_locked(rec)
            self._dirty_history = True
            self._pending_ops += 1

    def top_ratings(self) -> List[Tuple[str, float]]:
        with self._lock:
            return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)

    # ---------- export ----------
    def export_csv(self) -> str:
        # return CSV text of ratings
        rows = ["id,filename,relpath,rating"]
        with self._lock:
            for img_id, rating in self.ratings.items():
                rec = self.images.get(img_id)
                if not rec:
                    continue
                rows.append(f"{img_id},{rec.filename},{rec.relpath},{rating:.3f}")
        return "\n".join(rows) + "\n"

    # ---------- appearances (models) ----------
    def _rebuild_model_appearances_locked(self) -> None:
        cnt: Dict[str, int] = {}
        try:
            for h in self.history:
                for k in (h.get("winner_id"), h.get("loser_id")):
                    if isinstance(k, str) and k.startswith("model:"):
                        name = k.split(":", 1)[1]
                        cnt[name] = cnt.get(name, 0) + 1
        except Exception:
            pass
        self._model_appearances = cnt

    def _update_model_appearances_from_rec_locked(self, rec: Dict[str, Any]) -> None:
        for k in (rec.get("winner_id"), rec.get("loser_id")):
            if isinstance(k, str) and k.startswith("model:"):
                name = k.split(":", 1)[1]
                self._model_appearances[name] = self._model_appearances.get(name, 0) + 1

    def get_model_appearances(self) -> Dict[str, int]:
        with self._lock:
            # return a shallow copy to avoid external mutation
            return dict(self._model_appearances)

    def clear_model_appearances(self) -> None:
        with self._lock:
            self._model_appearances = {}

    # ---------- batched helpers ----------
    def apply_rating_update_and_history(self, updates: Dict[str, float], history_rec: Dict[str, Any]) -> None:
        with self._lock:
            for k, v in updates.items():
                self.ratings[k] = float(v)
            self.history.append(history_rec)
            self._update_model_appearances_from_rec_locked(history_rec)
            self._dirty_ratings = True
            self._dirty_history = True
            self._pending_ops += 1
