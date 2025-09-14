from __future__ import annotations

import cgi
import io
import json
import mimetypes
import os
import random
import threading
import traceback
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Tuple, Dict, Any, List
from urllib.parse import urlparse, parse_qs

from backend.elo import update_ratings
from backend.storage import DataStore, ImageRec
from backend.events import EventBroker


BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"


def build_handler(ds: DataStore, broker: EventBroker):
    class AppHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        # ---------- helpers ----------
        def _json(self, status: int, obj: Any):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
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

        # ---------- API ----------
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                try:
                    return self._handle_api_get(parsed)
                except Exception as e:
                    traceback.print_exc()
                    return self._bad("Server error", 500)
            if parsed.path.startswith("/images/"):
                return self._serve_image(parsed.path.split("/images/")[-1])
            # static files
            return super().do_GET()

        def _handle_api_get(self, parsed):
            path = parsed.path
            qs = parse_qs(parsed.query or "")
            if path == "/api/images":
                imgs = [
                    {
                        **rec.__dict__,
                        "rating": ds.get_rating(rec.id),
                    }
                    for rec in ds.list_images()
                ]
                imgs.sort(key=lambda r: r.get("created_at", ""))
                return self._json(200, {"images": imgs})

            if path == "/api/ratings":
                ranks = []
                for idx, (img_id, rating) in enumerate(ds.top_ratings(), start=1):
                    rec = next((r for r in ds.list_images() if r.id == img_id), None)
                    if rec:
                        ranks.append({
                            "rank": idx,
                            "id": img_id,
                            "filename": rec.filename,
                            "relpath": rec.relpath,
                            "rating": rating,
                        })
                return self._json(200, {"ratings": ranks})

            if path == "/api/next_pair":
                pair = self._select_pair()
                if not pair:
                    return self._bad("需要至少两张图片")
                a, b = pair
                return self._json(200, {
                    "a": {**a.__dict__, "rating": ds.get_rating(a.id)},
                    "b": {**b.__dict__, "rating": ds.get_rating(b.id)},
                })

            if path.startswith("/api/annotations/"):
                img_id = path.split("/api/annotations/")[-1]
                return self._json(200, {"id": img_id, "annotations": ds.get_annotations(img_id)})

            if path == "/api/export":
                csv = ds.export_csv()
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
            ranks = []
            for idx, (img_id, rating) in enumerate(ds.top_ratings(), start=1):
                rec = next((r for r in ds.list_images() if r.id == img_id), None)
                if rec:
                    ranks.append({
                        "rank": idx,
                        "id": img_id,
                        "filename": rec.filename,
                        "relpath": rec.relpath,
                        "rating": rating,
                    })
            return {"ratings": ranks}

        def _select_pair(self) -> Tuple[ImageRec, ImageRec] | None:
            imgs = ds.list_images()
            if len(imgs) < 2:
                return None
            # pick a pivot and a close-opponent by rating
            pivot = random.choice(imgs)
            pivot_rating = ds.get_rating(pivot.id)
            others = [i for i in imgs if i.id != pivot.id]
            others.sort(key=lambda r: abs(ds.get_rating(r.id) - pivot_rating))
            opponent = others[0] if others else random.choice(imgs)
            if opponent.id == pivot.id:
                # ensure distinct
                opponent = random.choice([i for i in imgs if i.id != pivot.id])
            # randomize left/right
            return (pivot, opponent) if random.random() < 0.5 else (opponent, pivot)

        def do_POST(self):
            parsed = urlparse(self.path)
            if not parsed.path.startswith("/api/"):
                return self._bad("POST not allowed", 405)
            try:
                return self._handle_api_post(parsed)
            except Exception:
                traceback.print_exc()
                return self._bad("Server error", 500)

        def _handle_api_post(self, parsed):
            path = parsed.path
            if path == "/api/upload":
                ctype, pdict = cgi.parse_header(self.headers.get("Content-Type"))
                if ctype != "multipart/form-data":
                    return self._bad("需要 multipart/form-data")
                pdict["boundary"] = pdict["boundary"].encode("utf-8") if isinstance(pdict.get("boundary"), str) else pdict.get("boundary")
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                })
                files = form["files"] if "files" in form else None
                if files is None:
                    # try any file fields
                    files = [item for item in form.list or [] if item.filename]
                else:
                    files = files if isinstance(files, list) else [files]
                saved = []
                for item in files:
                    if not getattr(item, "filename", None):
                        continue
                    data = item.file.read()
                    rec = ds.add_image_bytes(item.filename, data)
                    saved.append({**rec.__dict__, "rating": ds.get_rating(rec.id)})
                return self._json(200, {"saved": saved})

            if path == "/api/compare":
                body = self._parse_json_body() or {}
                winner_id = body.get("winner_id")
                loser_id = body.get("loser_id")
                tie = bool(body.get("tie", False))
                if not winner_id or not loser_id or winner_id == loser_id:
                    return self._bad("参数错误：需要不同的 winner_id/loser_id")
                r_w = ds.get_rating(winner_id)
                r_l = ds.get_rating(loser_id)
                res = update_ratings(r_w, r_l, k=int(ds.config.get("k_factor", 32)), tie=tie)
                ds.set_rating(winner_id, res.r1_after)
                ds.set_rating(loser_id, res.r2_after)
                ds.append_history({
                    "timestamp": dsr_iso(),
                    "winner_id": winner_id,
                    "loser_id": loser_id,
                    "tie": tie,
                    "winner_rating_before": res.r1_before,
                    "loser_rating_before": res.r2_before,
                    "winner_rating_after": res.r1_after,
                    "loser_rating_after": res.r2_after,
                    "rating_change": res.delta,
                })
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

            return self._bad("Unknown endpoint", 404)

    return AppHandler


def dsr_iso() -> str:
    # small helper to avoid importing iso_now back
    from datetime import datetime
    return datetime.now().isoformat()


def run(host: str = "0.0.0.0", port: int = 8765):
    ds = DataStore(BASE_DIR)
    broker = EventBroker()

    Handler = build_handler(ds, broker)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[elo] server running at http://{host}:{port}")
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

