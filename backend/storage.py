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

        # in-memory
        self.images: Dict[str, ImageRec] = {}
        self.annotations: Dict[str, Any] = {}
        self.ratings: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}

        self._load_all()

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
                self.config["k_factor"] = 32
            self._save_config()

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
            self._save_images()
            self._save_ratings()
            return rec

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
            self._save_annotations()

    # ---------- ratings/history ----------
    def get_rating(self, img_id: str) -> float:
        with self._lock:
            return float(self.ratings.get(img_id, self.config.get("initial_rating", DEFAULT_INITIAL_RATING)))

    def set_rating(self, img_id: str, rating: float) -> None:
        with self._lock:
            self.ratings[img_id] = float(rating)
            self._save_ratings()

    def append_history(self, rec: Dict[str, Any]) -> None:
        with self._lock:
            self.history.append(rec)
            self._save_history()

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

