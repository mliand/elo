from __future__ import annotations

from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


class AdminHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        # Only serve admin and static assets
        if parsed.path in ("/", "/index.html", "/compare.html"):
            self.send_response(302)
            self.send_header("Location", "/admin.html")
            self.end_headers()
            return
        if parsed.path == "/admin.html":
            # Inject API_BASE to point to the main backend
            p = WEB_DIR / "admin.html"
            if not p.exists():
                self.send_error(404)
                return
            html = p.read_text(encoding="utf-8")
            inj = "<script>window.API_BASE = 'http://localhost:8765';</script>"
            html = html.replace("</head>", inj + "\n</head>")
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()


def run(host: str = "0.0.0.0", port: int = 8766):
    print(f"[elo-admin] serving admin at http://{host}:{port}/admin.html")
    server = ThreadingHTTPServer((host, port), AdminHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[elo-admin] shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run()

