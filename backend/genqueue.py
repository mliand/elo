from __future__ import annotations

import threading
import queue
import time
import uuid
from typing import Any, Callable, Dict, Optional


class GenQueue:
    """Simple in-process generation job queue with a single worker thread.

    Each job is executed by a user-provided runner callable and the result
    is stored for later retrieval.
    """

    def __init__(self):
        self._q: "queue.Queue[str]" = queue.Queue()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._runner: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
        self._stop = False
        self._worker: Optional[threading.Thread] = None

    def set_runner(self, runner: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self._runner = runner
        if self._worker is None:
            self._start()

    def _start(self) -> None:
        def loop():
            while not self._stop:
                try:
                    job_id = self._q.get(timeout=0.25)
                except queue.Empty:
                    continue
                with self._lock:
                    job = self._jobs.get(job_id)
                    if not job:
                        continue
                    job["status"] = "running"
                    job["started_at"] = time.time()
                try:
                    if not self._runner:
                        raise RuntimeError("runner not set")
                    payload = job.get("payload", {})
                    res = self._runner(payload)
                    with self._lock:
                        job["status"] = "done"
                        job["result"] = res
                        job["finished_at"] = time.time()
                except Exception as e:
                    with self._lock:
                        job["status"] = "error"
                        job["error"] = str(e)
                        job["finished_at"] = time.time()
                finally:
                    self._q.task_done()

        t = threading.Thread(target=loop, name="GenQueueWorker", daemon=True)
        t.start()
        self._worker = t

    def close(self) -> None:
        self._stop = True
        t = self._worker
        if t:
            t.join(timeout=1.0)

    def submit(self, payload: Dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "payload": payload,
                "created_at": time.time(),
            }
        self._q.put(job_id)
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            j = self._jobs.get(job_id)
            if not j:
                return None
            return dict(j)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            queued = sum(1 for j in self._jobs.values() if j.get("status") == "queued")
            running = sum(1 for j in self._jobs.values() if j.get("status") == "running")
            done = sum(1 for j in self._jobs.values() if j.get("status") == "done")
            err = sum(1 for j in self._jobs.values() if j.get("status") == "error")
            return {
                "queued": queued,
                "running": running,
                "done": done,
                "error": err,
                "total": len(self._jobs),
                "qsize": self._q.qsize(),
            }

