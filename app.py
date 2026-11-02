"""Web interface for the semantic ranking module. Standard library server + numpy.

    python app.py            ->  http://127.0.0.1:8091
    python app.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from semantic_search import MODELS, QueryError, SemanticSearchEngine

WEB = Path(__file__).resolve().parent / "web"
ENGINE = SemanticSearchEngine.from_data_dir()

EXAMPLES = [
    {"query": "equipment left on after the factory closes", "note": "finds night & Sunday incidents"},
    {"query": "shift heavy loads to cheaper hours", "note": "never says peak or tariff"},
    {"query": "rotating machine mechanical wear", "note": "bearings, vibration"},
    {"query": "cold room warming up", "note": "door seals, heater"},
    {"query": "machine switching on and off too often", "note": "short cycling"},
    {"query": "dirty heat exchanger surfaces", "note": "a known failure"},
]


def meta() -> dict:
    return {
        "document_count": len(ENGINE.documents),
        "matrix_shape": list(ENGINE.dtm.shape),
        "default_k": ENGINE.default_k,
        "max_k": ENGINE.max_k,
        "energy": [round(float(e), 3) for e in ENGINE.svd.energy],
        "models": list(MODELS),
        "examples": EXAMPLES,
    }


def search(params: dict) -> dict:
    def one(name, default=""):
        return params.get(name, [default])[0]

    model = one("model", "lsi")
    k = min(max(int(one("k", str(ENGINE.default_k))), 1), ENGINE.max_k)
    top = min(max(int(one("top", "10")), 1), 40)
    min_score = min(max(float(one("min", "0.10")), -1.0), 1.0)
    return ENGINE.search(one("q"), model=model, k=k, top=top, min_score=min_score)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        params = parse_qs(url.query)
        try:
            if url.path == "/api/meta":
                return self.send_json(meta())
            if url.path == "/api/concepts":
                k = int(params.get("k", [str(ENGINE.default_k)])[0])
                return self.send_json({"k": ENGINE.lsi(k).k, "concepts": ENGINE.concepts(k)})
            if url.path == "/api/search":
                return self.send_json(search(params))
        except QueryError as error:
            return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except ValueError:
            return self.send_json({"error": "Invalid number in the request."}, HTTPStatus.BAD_REQUEST)
        return self.send_static(url.path)

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, path: str):
        target = (WEB / ("index.html" if path in ("", "/") else path.lstrip("/"))).resolve()
        if WEB not in target.parents or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if "/api/search" in self.path:
            q = parse_qs(urlparse(self.path).query)
            print(f"  search  [{q.get('model', ['lsi'])[0]}]  {q.get('q', [''])[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"VoltMize semantic search running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
