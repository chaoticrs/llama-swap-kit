"""HTTP worker for a model that needs its own Python code (no native runtime supports it).

llama-swap starts it on demand and kills it to free memory, exactly like the native backends:
    bin/uv/uv.exe run --project workers/<id> python workers/<id>/server.py --port <port>

Contract with llama-swap / clients:
    GET  /health  -> 200 once the model is loaded (llama-swap waits for this)
    POST /run     -> JSON in, JSON out; send/receive files as base64 strings
Reach it through llama-swap at  http://127.0.0.1:8080/upstream/<id>/run

To adapt it: add the model's packages to pyproject.toml, then fill in load() and run().
As shipped, load() does nothing and run() echoes the request, so the plumbing can be tested.
"""
import argparse
import base64
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # project root; put downloaded weights under ROOT / "models" / ...

MODEL = None


def load():
    """Load weights once at startup and return whatever run() needs.

    Example (image-to-3D):
        from idol import IdolPipeline
        return IdolPipeline.from_pretrained("speridlabs/idol")
    """
    return None


def run(model, req: dict) -> dict:
    """Handle one request. Decode inputs from base64, encode outputs to base64.

    Example:
        img = HERE / "in.png"; img.write_bytes(base64.b64decode(req["image"]))
        out = model(str(img)); out.gaussian.save_ply(HERE / "out.ply")
        return {"ply": base64.b64encode((HERE / "out.ply").read_bytes()).decode()}
    """
    return {"echo": req}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, {"error": "POST /run"})
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._send(200, run(MODEL, json.loads(body or b"{}")))
        except Exception as e:  # report to the client instead of dropping the connection
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{HERE.name}] {fmt % args}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    global MODEL
    MODEL = load()  # before binding: llama-swap keeps polling /health until the port answers
    # Single-threaded on purpose: one request at a time on the GPU.
    print(f"[{HERE.name}] ready on {args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
