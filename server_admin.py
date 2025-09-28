from __future__ import annotations

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


def build_handler(api_base: str | None = None, api_port: int = 8765, allow_fallback: bool = True):
    class AdminHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        # ---- proxy helpers ----
        def _backend_target(self):
            # returns (scheme, host, port)
            if api_base:
                try:
                    from urllib.parse import urlparse as _u
                    u = _u(api_base)
                    scheme = u.scheme or 'http'
                    host = u.hostname or '127.0.0.1'
                    port = u.port or (443 if scheme == 'https' else 80)
                    return scheme, host, port
                except Exception:
                    pass
            # fallback to same host with provided port
            try:
                # use the Host header's hostname if available for LAN access
                host_hdr = (self.headers.get('Host') or '').split(':')[0] or '127.0.0.1'
            except Exception:
                host_hdr = '127.0.0.1'
            return 'http', host_hdr, int(api_port or 8765)

        def _fallback_handle(self, method: str) -> bool:
            """Return True if a local fallback handled the request."""
            if not allow_fallback:
                return False
            parsed = urlparse(self.path)
            p = parsed.path
            try:
                # Fallback: serve minimal /api/ratings and /api/history from disk
                if method == 'GET' and p == '/api/ratings':
                    data = self._build_local_ratings()
                    blob = json.dumps({"ratings": data}, ensure_ascii=False).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Content-Length', str(len(blob)))
                    self.end_headers()
                    self.wfile.write(blob)
                    return True
                if method == 'GET' and p == '/api/history':
                    hist_path = BASE_DIR / 'history.json'
                    try:
                        hist = json.loads(hist_path.read_text(encoding='utf-8')) if hist_path.exists() else []
                    except Exception:
                        hist = []
                    blob = json.dumps({"history": hist}, ensure_ascii=False).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Content-Length', str(len(blob)))
                    self.end_headers()
                    self.wfile.write(blob)
                    return True
            except Exception:
                return False
            return False

        def _scan_model_stems_local(self) -> dict:
            models_dir = (BASE_DIR / 'models').resolve()
            allowed = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            stems = {}
            if models_dir.exists():
                for model_dir in models_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name
                    for f in model_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in allowed:
                            stem = f.stem
                            stems.setdefault(stem, {})[model] = str(f)
            return stems

        def _build_local_ratings(self) -> list:
            # read ratings.json and history.json, construct leaderboard
            rfile = BASE_DIR / 'ratings.json'
            try:
                ratings = json.loads(rfile.read_text(encoding='utf-8')) if rfile.exists() else {}
            except Exception:
                ratings = {}
            hfile = BASE_DIR / 'history.json'
            try:
                history = json.loads(hfile.read_text(encoding='utf-8')) if hfile.exists() else []
            except Exception:
                history = []
            stems = self._scan_model_stems_local()
            models = set()
            for m in stems.values():
                models.update(m.keys())
            for k in list(ratings.keys()):
                if isinstance(k, str) and k.startswith('model:'):
                    models.add(k.split(':', 1)[1])
            # appearances count from history
            appear = {}
            try:
                for rec in history:
                    for key in (rec.get('winner_id'), rec.get('loser_id')):
                        if isinstance(key, str) and key.startswith('model:'):
                            name = key.split(':', 1)[1]
                            appear[name] = appear.get(name, 0) + 1
            except Exception:
                pass
            # pick one sample per model if exists
            samples = {}
            for stem, by_model in stems.items():
                for model, p in by_model.items():
                    if model not in samples:
                        import os as _os
                        samples[model] = {
                            'filename': _os.path.basename(p),
                            'relpath': f"/local/model/{model}/{_os.path.basename(p)}",
                        }
            items = []
            for model in sorted(models):
                r = ratings.get(f'model:{model}', 1200.0)
                apps = int(appear.get(model, 0))
                smp = samples.get(model)
                items.append((model, float(r), apps, smp))
            items.sort(key=lambda x: x[1], reverse=True)
            ranks = []
            for i, (model, rating, apps, smp) in enumerate(items, start=1):
                ranks.append({
                    'rank': i,
                    'model': model,
                    'rating': rating,
                    'appearances': apps,
                    'relpath': smp['relpath'] if smp else '',
                    'filename': smp['filename'] if smp else model,
                })
            return ranks

        def _proxy(self, method: str):
            import http.client
            from urllib.parse import urlsplit
            scheme, host, port = self._backend_target()
            path = self.path
            # remove any admin origin prefix; forward full /api/... path as-is
            body = None
            if method in ('POST', 'PUT', 'PATCH'):
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                except Exception:
                    length = 0
                body = self.rfile.read(length) if length > 0 else None
            Conn = http.client.HTTPSConnection if scheme == 'https' else http.client.HTTPConnection
            conn = Conn(host, port, timeout=60)
            # forward a minimal header set
            fwd_headers = {
                'Accept': self.headers.get('Accept', '*/*'),
                'Content-Type': self.headers.get('Content-Type', ''),
                'User-Agent': self.headers.get('User-Agent', 'elo-admin-proxy'),
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
            try:
                conn.request(method, path, body=body, headers={k: v for k, v in fwd_headers.items() if v})
                resp = conn.getresponse()
                # relay status and headers
                self.send_response(resp.status)
                # copy headers except hop-by-hop/length (will stream)
                for hk, hv in resp.getheaders():
                    hkl = hk.lower()
                    if hkl in ('transfer-encoding', 'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 'upgrade'):
                        continue
                    # For SSE, backend sets text/event-stream; keep it
                    self.send_header(hk, hv)
                self.end_headers()
                # stream body
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    try:
                        self.wfile.flush()
                    except Exception:
                        pass
            except Exception as e:
                # try local fallback for common GET endpoints
                if self._fallback_handle(method):
                    return
                else:
                    try:
                        self.send_response(502)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        payload = json.dumps({'error': f'proxy failed: {e}'}).encode('utf-8')
                        self.send_header('Content-Length', str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                    except Exception:
                        pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        def do_GET(self):
            parsed = urlparse(self.path)
            # Only serve admin and static assets
            if parsed.path in ("/", "/index.html", "/compare.html"):
                self.send_response(302)
                self.send_header("Location", "/admin.html")
                self.end_headers()
                return
            if parsed.path.startswith('/api/'):
                return self._proxy('GET')
            if parsed.path == "/admin.html":
                # Inject API_BASE to point to the main backend
                p = WEB_DIR / "admin.html"
                if not p.exists():
                    self.send_error(404)
                    return
                html = p.read_text(encoding="utf-8")
                # Prefer explicit base if provided; otherwise derive dynamically on client
                if api_base:
                    inj = f"<script>window.API_BASE = {json.dumps(api_base)};</script>"
                else:
                    # Use current page protocol + hostname, but with backend port
                    inj = (
                        "<script>(function(){try{var loc=window.location;"
                        f"var base=loc.protocol+'//'+loc.hostname+':{api_port}';"
                        "if(!window.API_BASE){window.API_BASE=base;}}catch(e){}})();</script>"
                    )
                html = html.replace("</head>", inj + "\n</head>")
                data = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith('/api/'):
                return self._proxy('POST')
            return super().do_POST()

    return AdminHandler


def run(host: str = "0.0.0.0", port: int = 8766, *, api_base: str | None = None, api_port: int = 8765, allow_fallback: bool = True):
    Handler = build_handler(api_base=api_base, api_port=api_port, allow_fallback=allow_fallback)
    print(f"[elo-admin] serving admin at http://{host}:{port}/admin.html")
    if api_base:
        print(f"[elo-admin] API_BASE set to {api_base}")
    else:
        print(f"[elo-admin] API_BASE derived as http(s)://<host>:{api_port}")
    print(f"[elo-admin] local fallback {'ENABLED' if allow_fallback else 'DISABLED'}")
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[elo-admin] shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Serve ELO admin UI")
    parser.add_argument("--host", default=os.environ.get("ELO_ADMIN_HOST", "0.0.0.0"), help="Bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("ELO_ADMIN_PORT", "8766")), help="Bind port (default 8766)")
    parser.add_argument("--api-base", default=os.environ.get("ELO_API_BASE"), help="Absolute backend base URL, e.g. http://10.0.0.2:8765")
    parser.add_argument("--api-port", type=int, default=int(os.environ.get("ELO_API_PORT", "8765")), help="Backend port if deriving base (default 8765)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable local JSON fallback when proxy fails")
    args = parser.parse_args()
    allow_fallback = not args.no_fallback
    # env var toggle also supported: ELO_NO_FALLBACK=1
    if os.environ.get("ELO_NO_FALLBACK") in ("1", "true", "yes", "on"):
        allow_fallback = False
    run(args.host, args.port, api_base=args.api_base, api_port=args.api_port, allow_fallback=allow_fallback)
