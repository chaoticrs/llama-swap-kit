"""Text-to-speech worker: wraps a Python TTS model behind the same OpenAI endpoint the native engines serve,
so it can fill the `tts` role (see CONTRACT.md) and clients never notice the swap.

llama-swap starts it on demand, exactly like the native backends:
    bin/uv/uv.exe run --project workers/<id> python workers/<id>/server.py --port <port> --model-dir models/tts/<id>

Endpoints:
    GET  /health            -> 200 once the model is loaded (llama-swap waits for this)
    GET  /v1/audio/voices   -> {"voices": [...]}
    POST /v1/audio/speech   -> JSON {"input": text, "voice": name (optional)} -> WAV bytes (16-bit mono)
An unknown or missing voice falls back to the first voice, so clients written for another TTS model keep working.

To adapt it: add the model's packages to pyproject.toml, then fill in load(). As shipped, it "speaks" a short
tone per request, so the plumbing can be tested before any model code exists.
"""
import argparse
import io
import json
import math
import struct
import sys
import traceback
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # project root

TTS = None


class ToneTTS:
    """Placeholder model: replace with the real one in load()."""
    voices = ["tone"]
    sample_rate = 24000

    def synth(self, text, voice):
        n = int(self.sample_rate * min(2.0, 0.05 * max(1, len(text))))
        return [0.2 * math.sin(2 * math.pi * 440 * i / self.sample_rate) for i in range(n)]


def load(model_dir: Path):
    """Load the model once and return an object with:
        .voices       list of voice names (the first one is the default)
        .sample_rate  output sample rate in Hz
        .synth(text, voice) -> mono float samples in [-1, 1] (list or numpy array)

    model_dir is models/tts/<id>, where `manage.py add` downloaded the repo (code and weights). Example:
        sys.path.insert(0, str(model_dir))
        from some_tts import Model
        ...
    """
    return ToneTTS()


def to_wav(samples, sr):
    try:
        import numpy as np
        pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * 32767).astype("<i2").tobytes()
    except ImportError:
        pcm = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path.split("?")[0] == "/v1/audio/voices":
            self._send(200, {"voices": list(TTS.voices)})
        elif self.path.split("?")[0] in ("/", "/v1/models"):  # llama-swap's preload probes these
            self._send(200, {"object": "list", "data": [{"id": HERE.name, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/audio/speech":
            return self._send(404, {"error": "POST /v1/audio/speech"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = str(req.get("input", "")).strip()
            if not text:
                return self._send(400, {"error": "'input' is required"})
            voice = req.get("voice") if req.get("voice") in TTS.voices else TTS.voices[0]
            data = to_wav(TTS.synth(text, voice), TTS.sample_rate)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
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
    ap.add_argument("--model-dir", type=Path, default=ROOT / "models" / "tts" / HERE.name)
    args = ap.parse_args()

    global TTS
    TTS = load(args.model_dir)  # before binding: llama-swap keeps polling /health until the port answers
    print(f"[{HERE.name}] ready on {args.host}:{args.port}, voices: {', '.join(TTS.voices)}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()  # one request at a time


if __name__ == "__main__":
    main()
