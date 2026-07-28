"""Stdlib-only HTTP server for the prediction-visualization app.

No third-party dependencies (no Flask/pandas) - just Python's http.server, so
it runs anywhere with a plain Python 3. Start it with:

    python visualization_app/server.py

then open http://127.0.0.1:8070 in a browser.

Routes:
  GET /                      -> the single-page UI (static/index.html)
  GET /static/<file>         -> JS / CSS assets
  GET /api/files             -> list discovered prediction CSVs
  GET /api/data?file=<path>  -> parsed rows + columns + kind for one CSV
  GET /api/image?dataset=&path=  -> the resolved local image bytes
"""

import json
import mimetypes
import os
import posixpath
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
import dataloader
import imageresolver

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")


class Handler(BaseHTTPRequestHandler):
    # Quieter logging (one line per request is enough).
    def log_message(self, fmt, *args):
        print("[server]", fmt % args)

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj, status=200):
        payload = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_bytes(self, data, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, msg, status=400):
        self._send_json({"error": msg}, status=status)

    # -- routing ----------------------------------------------------------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        try:
            if route == "/" or route == "/index.html":
                return self._serve_static("index.html")
            if route.startswith("/static/"):
                return self._serve_static(route[len("/static/"):])
            if route == "/api/files":
                return self._api_files()
            if route == "/api/data":
                return self._api_data(qs)
            if route == "/api/image":
                return self._api_image(qs)
            return self._send_error_json("not found: " + route, status=404)
        except BrokenPipeError:
            pass  # client navigated away mid-response; nothing to do.
        except Exception as exc:  # noqa: BLE001 - surface any error to the client
            self._send_error_json(f"server error: {exc}", status=500)

    # -- static -----------------------------------------------------------
    def _serve_static(self, relpath):
        # Prevent path traversal outside STATIC_DIR.
        safe = posixpath.normpath("/" + relpath).lstrip("/")
        full = os.path.join(STATIC_DIR, safe)
        if not os.path.isfile(full):
            return self._send_error_json("static not found: " + relpath, status=404)
        ctype, _ = mimetypes.guess_type(full)
        with open(full, "rb") as f:
            self._send_bytes(f.read(), ctype or "application/octet-stream")

    # -- API --------------------------------------------------------------
    def _api_files(self):
        self._send_json({"files": dataloader.discover_csvs()})

    def _api_data(self, qs):
        path = (qs.get("file") or [None])[0]
        if not path:
            return self._send_error_json("missing ?file=")
        path = os.path.abspath(path)
        # Only allow files inside the configured search dirs.
        if not self._path_allowed(path):
            return self._send_error_json("file not in allowed search dirs", status=403)
        if not os.path.isfile(path):
            return self._send_error_json("file not found: " + path, status=404)
        data = dataloader.load_csv(path)
        data["file"] = path
        self._send_json(data)

    def _api_image(self, qs):
        stored = (qs.get("path") or [None])[0]
        dataset = (qs.get("dataset") or [None])[0]
        if not stored:
            return self._send_error_json("missing ?path=")
        local = imageresolver.resolve(dataset, stored)
        if not local:
            return self._send_error_json("image not found locally", status=404)
        ctype, _ = mimetypes.guess_type(local)
        with open(local, "rb") as f:
            self._send_bytes(f.read(), ctype or "image/jpeg")

    def _path_allowed(self, path):
        for base in config.CSV_SEARCH_DIRS:
            if base and os.path.isdir(base) and \
               os.path.commonpath([os.path.abspath(base), path]) == os.path.abspath(base):
                return True
        return False


def main():
    os.chdir(HERE)  # so relative imports of config/dataloader resolve
    server = ThreadingHTTPServer((config.HOST, config.PORT), Handler)
    url = f"http://{config.HOST}:{config.PORT}"
    print(f"Prediction viewer running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
