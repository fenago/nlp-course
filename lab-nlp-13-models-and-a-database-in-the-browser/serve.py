"""serve.py: the web server behind the App tab.

It does three small jobs and nothing else. It serves the app you are building
(the files in ~/exercises/app, re-read on every request so an edit shows on the
next reload). It serves the JavaScript libraries and the ONNX model files that
are baked into the image under /opt/nlplab/web, so the page never needs a CDN
or Hugging Face. And it accepts the reports the page posts back, because the
checks run on this machine and cannot see inside your browser: each report is
written to ~/exercises/out/13_<name>.json.

Every page it serves carries a Content-Security-Policy that allows this server
and nothing else, so if any code in the page tries to fetch from another
address, the browser refuses and says so in the console.

The session starts it for you (workshop/supervisor/app.conf). To restart it:
    supervisorctl restart app
"""
import json
import os
import sys
import tempfile
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

PORT = int(os.environ.get("APP_PORT", "8013"))
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app")
OUT = os.path.join(HERE, "out")
WEB = os.environ.get("NLPLAB_WEB", "/opt/nlplab/web")
ROOTS = {
    "/vendor/": os.path.join(WEB, "node_modules"),
    "/models/": os.path.join(WEB, "models"),
}
REPORTS = {"step1", "step2", "step3", "own", "resources"}

CSP = ("default-src 'self'; "
       "script-src 'self' 'wasm-unsafe-eval' blob:; "
       "worker-src 'self' blob:; "
       "connect-src 'self' blob: data:; "
       "img-src 'self' data:; "
       "style-src 'self' 'unsafe-inline'")

TYPES = {".wasm": "application/wasm", ".mjs": "text/javascript",
         ".js": "text/javascript", ".json": "application/json",
         ".onnx": "application/octet-stream", ".html": "text/html; charset=utf-8",
         ".css": "text/css", ".txt": "text/plain; charset=utf-8"}


def write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


class Handler(SimpleHTTPRequestHandler):
    server_version = "KittiwakeApp/1.0"

    def translate_path(self, path):
        path = unquote(urlsplit(path).path)
        for prefix, root in ROOTS.items():
            if path.startswith(prefix):
                base = root
                rest = path[len(prefix):]
                break
        else:
            base, rest = APP, path.lstrip("/") or "index.html"
        full = os.path.realpath(os.path.join(base, rest))
        # Never serve anything outside the three folders above.
        if not full.startswith(os.path.realpath(base) + os.sep) and full != os.path.realpath(base):
            return os.path.join(APP, "__nothing__")
        return full

    def guess_type(self, path):
        return TYPES.get(os.path.splitext(path)[1], "application/octet-stream")

    def end_headers(self):
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        if urlsplit(self.path).path.startswith(("/vendor/", "/models/")):
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 8_000_000:
                raise ValueError("report too large")
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError as e:
            return self._reply(400, {"ok": False, "error": str(e)})
        if path.startswith("/report/") and path[8:] in REPORTS:
            body["_received"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            dest = os.path.join(OUT, f"13_{path[8:]}.json")
            if path[8:] == "resources":
                # Keep every URL and refusal from every page load, not just the last.
                try:
                    old = json.load(open(dest))
                except (OSError, ValueError):
                    old = {}
                for field in ("urls", "violations"):
                    seen = old.get(field, [])
                    body[field] = seen + [x for x in body.get(field, []) if x not in seen]
            write_atomic(dest, body)
            return self._reply(200, {"ok": True, "saved": f"out/13_{path[8:]}.json"})
        if path == "/predict":
            p = os.path.join(OUT, "predictions.json")
            try:
                preds = json.load(open(p))
            except (OSError, ValueError):
                preds = {}
            key = str(body.get("key", ""))[:40]
            if key:
                entry = preds.get(key, {})
                for field in ("guess", "actual"):
                    if body.get(field) is not None:
                        entry[field] = str(body[field])[:200]
                preds[key] = entry
                write_atomic(p, preds)
            return self._reply(200, {"ok": True})
        return self._reply(404, {"ok": False, "error": "unknown report"})

    def _reply(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(f"serving {APP} on port {PORT}; libraries and models from {WEB}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
