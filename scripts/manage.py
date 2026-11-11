"""Model manager for llama-swap-kit.

Identifies a model (local file or Hugging Face repo), downloads/converts it if needed,
registers it in models/registry.json and regenerates config/llama-swap.yaml.

    python scripts/manage.py inspect <hf-repo | file>      classify only, downloads nothing
    python scripts/manage.py add <hf-repo | file> [opts]   register (download/convert as needed)
    python scripts/manage.py list
    python scripts/manage.py remove <id> [--delete-files]
    python scripts/manage.py render                        rewrite config from the registry
    python scripts/manage.py detect                        size the pools to this PC, re-estimate, render
    python scripts/manage.py hardware                      print detected GPU/RAM + build variant (JSON)
    python scripts/manage.py role [<role> <model-id>]      list roles / point a role at a model
    python scripts/manage.py check [--all]                 test CONTRACT.md against the running stack

(`manage.ps1` in the project root runs this with the uv-managed Python; no system Python needed.)

Routes (see scripts/folder_description.md):
    GGUF LLM / vision      -> llama.cpp   (llama-server)
    HF transformers LLM    -> converted once to GGUF, then llama.cpp
    diffusion model        -> stable-diffusion.cpp (sd-server)  image or video
    whisper ggml .bin      -> whisper.cpp (whisper-server)
    audio.cpp GGUF         -> audio.cpp   (audiocpp_server)     TTS / ASR
    anything else          -> Python worker scaffold in workers/<id>

Standard library only. HF_TOKEN in the environment is used for gated repos.
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"
MODELS = ROOT / "models"
CONFIG = ROOT / "config"
TOOLS = ROOT / "tools"
WORKERS = ROOT / "workers"
REGISTRY = MODELS / "registry.json"
HF = "https://huggingface.co"

DEFAULT_POOLS = {"vram": 3800, "ram": 11000}  # MiB; replaced by `detect` with this machine's numbers
VRAM_HEADROOM = 250    # MiB of VRAM left for the desktop / other apps
MIN_GPU_VRAM = 3000    # non-NVIDIA GPUs with less dedicated VRAM (integrated ones report ~128-2047) get the CPU build
GPU_DLLS = {"cuda": "ggml-cuda.dll", "vulkan": "ggml-vulkan.dll"}

# Roles are the stable names other layers request (`"model": "agent"`); the registry maps each to the model
# that currently fills it, and render() turns them into llama-swap aliases. See CONTRACT.md.
ROLE_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
STANDARD_ROLES = {
    "chat": "general assistant LLM (text in, text out)",
    "agent": "always-ready LLM that does tool calling for daemons/agents",
    "vision": "LLM that also accepts images",
    "reasoner": "strongest (slower) LLM for hard questions",
    "embed": "embedding model (/v1/embeddings)",
    "stt": "speech to text (/v1/audio/transcriptions)",
    "tts": "text to speech (/v1/audio/speech)",
    "image": "text/image to image (/v1/images/generations)",
    "video": "video generation (sd.cpp /upstream/<model-id>/sdcpp/v1/vid_gen)",
}
DEFAULT_ROLES = {"llm": ["chat", "agent"], "stt": ["stt"], "tts": ["tts"], "image": ["image"], "video": ["video"]}
AUDIO_PORTS = range(5700, 5800)  # audio.cpp takes its port from a config file, not ${PORT}

IMAGE_ARCHS = {"flux", "flux2", "sd1", "sd2", "sdxl", "sd3", "aura", "chroma", "lumina2", "hidream",
               "qwen_image", "z_image", "pixart", "anima", "ernie_image", "ovis_image"}
VIDEO_ARCHS = {"wan", "ltxv", "ltx", "hyvid", "hunyuan_video", "cosmos", "mochi"}
ASR_FAMILY = re.compile(r"asr|whisper|transcribe|parakeet|canary|moonshine|bark3|citrinet|owsm")
GGUF_SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
QUANT_PREFERENCE = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q4_0", "Q6_K", "Q8_0"]
SKIP_FOR_CONVERT = re.compile(r"\.(gguf|onnx|h5|msgpack|ot|png|jpe?g|gif|webp|mp4|md|pdf)$|^original/|^onnx/", re.I)


# ----------------------------------------------------------------------------- HTTP

class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """HF redirects downloads to a CDN; don't leak the token to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_StripAuthOnRedirect)


def _request(url, headers=None):
    h = {"User-Agent": "localai-manage/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith(HF):
        h["Authorization"] = f"Bearer {token}"
    h.update(headers or {})
    return urllib.request.Request(url, headers=h)


def http_json(url):
    with _opener.open(_request(url), timeout=60) as r:
        return json.load(r)


def hf_url(repo, path):
    return f"{HF}/{repo}/resolve/main/{urllib.parse.quote(path, safe='/')}"


def hf_info(repo):
    try:
        info = http_json(f"{HF}/api/models/{repo}?blobs=true")
    except urllib.error.HTTPError as e:
        hint = " (gated or private: set HF_TOKEN)" if e.code in (401, 403) else ""
        sys.exit(f"cannot read HF repo {repo}: HTTP {e.code}{hint}")
    info["files"] = {s["rfilename"]: s.get("size") or 0 for s in info.get("siblings", [])}
    return info


def download(url, dest: Path, size=0):
    """Resumable download (keeps a .part file until complete)."""
    if dest.exists() and (not size or dest.stat().st_size == size):
        print(f"  [have] {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    with _opener.open(_request(url, {"Range": f"bytes={start}-"} if start else None), timeout=60) as r:
        if start and r.status != 206:
            start = 0
        total = start + int(r.headers.get("Content-Length") or 0)
        done, last = start, 0.0
        with open(part, "ab" if start else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 1:
                    last = time.time()
                    print(f"\r  {dest.name}: {done / 2**20:,.0f} / {total / 2**20:,.0f} MiB", end="", flush=True)
    print(f"\r  {dest.name}: {done / 2**20:,.0f} MiB done" + " " * 20)
    part.replace(dest)


class RangeReader:
    """File-like read() over HTTP Range requests, to sniff headers without downloading."""

    def __init__(self, url, chunk=1 << 20):
        self.url, self.chunk, self.pos, self.buf, self.buf_start = url, chunk, 0, b"", 0

    def read(self, n):
        out = bytearray()
        while n > 0:
            off = self.pos - self.buf_start
            if not 0 <= off < len(self.buf):
                end = self.pos + max(n, self.chunk) - 1
                with _opener.open(_request(self.url, {"Range": f"bytes={self.pos}-{end}"}), timeout=60) as r:
                    self.buf, self.buf_start, off = r.read(), self.pos, 0
                if not self.buf:
                    break
            piece = self.buf[off:off + n]
            out += piece
            self.pos += len(piece)
            n -= len(piece)
        return bytes(out)

    def skip(self, n):
        self.pos += n


def _skip(f, n):
    f.skip(n) if hasattr(f, "skip") else f.seek(n, 1)


# ----------------------------------------------------------------------------- format sniffing

_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf_meta(f, until=None):
    """Metadata keys up to the tokenizer block (enough to identify the model), or up to key `until`."""
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    f.read(4)  # version
    _, n_kv = struct.unpack("<QQ", f.read(16))

    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def rval(t):
        if t == 8:
            return rstr()
        if t == 9:
            (at,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            if at in _GGUF_SCALAR and n > 64:  # big numeric arrays: skip the bytes
                _skip(f, n * struct.calcsize(_GGUF_SCALAR[at]))
                return f"<{n} values>"
            return [rval(at) for _ in range(n)]
        fmt = _GGUF_SCALAR[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    meta = {}
    for _ in range(n_kv):
        key = rstr()
        if key.startswith("tokenizer.") and "general.architecture" in meta:
            break
        (t,) = struct.unpack("<I", f.read(4))
        meta[key] = rval(t)
        if until:
            if until in meta:
                break
        elif meta.get("general.architecture") == "audiocpp" and "audiocpp.model_spec.family" in meta:
            break
    return meta


def audiocpp_voices(path):
    """Preset voice ids packed inside an audio.cpp GGUF (embedded files named voices/<id>.bin)."""
    try:
        with open(path, "rb") as f:
            names = read_gguf_meta(f, until="audiocpp.embedded_files.names").get("audiocpp.embedded_files.names") or []
    except (OSError, ValueError, struct.error):
        return []
    voices = sorted(n[len("voices/"):-len(".bin")] for n in names if n.startswith("voices/") and n.endswith(".bin"))
    # The UI preselects the first voice; prefer Kokoro's recommended default when present.
    return sorted(voices, key=lambda v: v != "af_heart")


def read_safetensors_names(f):
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 200_000_000:
        raise ValueError("not a safetensors file")
    header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    return set(header), meta, sum(
        (v["data_offsets"][1] - v["data_offsets"][0]) for v in header.values())


def classify_tensors(names):
    """Identify a safetensors file from its tensor names: (kind, detail, extra)."""
    def has(prefix):
        return any(n.startswith(prefix) for n in names)

    def contains(part):
        return any(part in n for n in names)

    if has("model.diffusion_model.") or has("first_stage_model."):
        if has("conditioner.embedders.1"):
            return "image", "SDXL checkpoint", {"model_flag": "-m"}
        if contains("joint_blocks"):
            return "image", "SD3 checkpoint", {"model_flag": "-m"}
        if contains("double_blocks"):
            return "image", "Flux checkpoint", {"model_flag": "-m"}
        return "image", "SD 1.x/2.x checkpoint", {"model_flag": "-m"}
    if has("double_blocks.") and has("single_blocks."):
        return "image", "Flux-style diffusion model (needs --vae, --clip_l, --t5xxl)", {"model_flag": "--diffusion-model"}
    if has("joint_blocks."):
        return "image", "SD3 diffusion model (needs text encoders + vae)", {"model_flag": "--diffusion-model"}
    if has("patch_embedding.") and contains("cross_attn"):
        return "video", "Wan diffusion model (needs --vae, --t5xxl)", {"model_flag": "--diffusion-model"}
    if has("model.layers.") or has("model.language_model.") or has("transformer.h.") or has("language_model.model."):
        return "llm", "Hugging Face transformers LLM (needs conversion to GGUF)", {}
    if has("model.encoder.layers.") and has("model.decoder.layers.") and contains("conv1"):
        return "stt", "Whisper in HF format (use a ggml-*.bin from ggerganov/whisper.cpp instead)", {}
    if has("encoder.down") and has("decoder.up"):
        return "component", "VAE (pass to an image model with --args \"--vae <path>\")", {}
    if has("text_model.encoder."):
        return "component", "CLIP text encoder (--clip_l / --clip_g)", {}
    if has("encoder.block.0.layer"):
        return "component", "T5 text encoder (--t5xxl)", {}
    return "unknown", "unrecognised tensor layout; needs its own code (Python worker)", {}


def classify_gguf(meta):
    arch = meta.get("general.architecture", "")
    if arch == "audiocpp":
        fam = meta.get("audiocpp.model_spec.family", "")
        task = "asr" if ASR_FAMILY.search(fam) else "tts"
        return ("stt" if task == "asr" else "tts"), f"audio.cpp {fam}", {"family": fam, "task": task}
    if arch == "clip" or "clip.projector_type" in meta or "clip.has_vision_encoder" in meta:
        return "mmproj", "multimodal projector (attach to an LLM with --mmproj)", {}
    if arch in IMAGE_ARCHS:
        return "image", f"{arch} diffusion model (may need --vae / text encoders)", {"model_flag": "--diffusion-model"}
    if arch in VIDEO_ARCHS:
        return "video", f"{arch} video diffusion model (needs --vae / text encoder)", {"model_flag": "--diffusion-model"}
    name = meta.get("general.name", "")
    ctx = meta.get(f"{arch}.context_length")
    return "llm", f"{arch} LLM" + (f" '{name}'" if name else "") + (f", ctx {ctx}" if ctx else ""), {"arch": arch}


def sniff(open_fn, filename):
    """Classify one model file. open_fn() returns a binary file-like object."""
    low = filename.lower()
    f = open_fn()
    try:
        if low.endswith(".gguf"):
            return classify_gguf(read_gguf_meta(f))
        if low.endswith((".safetensors", ".sft")):
            names, meta, _ = read_safetensors_names(f)
            return classify_tensors(names)
        if low.endswith(".bin") and f.read(4) == b"lmgg":
            return "stt", "whisper.cpp ggml model", {}
    finally:
        if hasattr(f, "close"):
            f.close()
    return "unknown", f"unsupported file type: {Path(filename).suffix}", {}


# ----------------------------------------------------------------------------- registry

def load_registry():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"pools": dict(DEFAULT_POOLS), "listen": "127.0.0.1:8080", "models": {}}


def save_registry(reg):
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- hardware

def _total_ram_mb():
    if os.name == "nt":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total", "avail", "pagetotal", "pageavail",
                                                         "virttotal", "virtavail", "extavail")]
        ms = MemStatus()
        ms.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.total // 2**20
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20


def _nvidia_gpus():
    """[(name, MiB)] from nvidia-smi, which ships with every NVIDIA driver."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.splitlines():
        name, _, mem = line.rpartition(",")
        if name and mem.strip().isdigit():
            gpus.append((name.strip(), int(mem)))
    return gpus


def _windows_gpus():
    """[(name, dedicated MiB)] for every display adapter, from the drivers' registry entries
    (the 64-bit qwMemorySize value; WMI's AdapterRAM is capped at 4 GiB)."""
    if os.name != "nt":
        return []
    import winreg
    cls = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, cls)
    except OSError:
        return []
    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            try:
                with winreg.OpenKey(root, sub) as k:
                    name = winreg.QueryValueEx(k, "DriverDesc")[0]
                    try:
                        mem = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0]
                    except OSError:
                        mem = winreg.QueryValueEx(k, "HardwareInformation.MemorySize")[0]
            except OSError:
                continue
            if isinstance(mem, bytes):
                mem = int.from_bytes(mem[:8], "little")
            gpus.append((str(name), int(mem) // 2**20))
    return gpus


def detect_hardware():
    """What this PC has and which backend build suits it: cuda (NVIDIA), vulkan (AMD/Intel with
    enough dedicated VRAM) or cpu."""
    nv = _nvidia_gpus()
    if nv:
        gpu, vram = max(nv, key=lambda g: g[1])
        variant = "cuda"
    else:
        others = [g for g in _windows_gpus() if "nvidia" not in g[0].lower()]
        gpu, vram = max(others, key=lambda g: g[1], default=("none", 0))
        variant = "vulkan" if vram >= MIN_GPU_VRAM else "cpu"
    return {"variant": variant, "gpu": gpu, "vram_total_mb": vram, "ram_total_mb": _total_ram_mb()}


def installed_variant():
    """Build variant of the binaries in bin/ (setup.ps1 installs the same one for every GPU backend)."""
    for variant, dll in GPU_DLLS.items():
        if (BIN / "llama" / dll).exists():
            return variant
    return "cpu"


def pools_for(hw, variant):
    vram = 0 if variant == "cpu" else max(0, hw["vram_total_mb"] - VRAM_HEADROOM) // 100 * 100
    ram = hw["ram_total_mb"]
    ram = max(1000, ram - max(4096, ram * 3 // 10)) // 100 * 100  # keep 4 GiB / 30% for Windows + apps
    return {"vram": vram, "ram": ram}


def model_size_mb(m):
    """Weights size, for re-estimating entries registered before size_mb was stored."""
    if m.get("size_mb"):
        return m["size_mb"]
    if m["kind"] == "worker":
        return max(0, m.get("ram_mb", 0) - 1500)  # inverse of estimate(); weights live in the HF cache
    p = Path(m["path"]) if Path(m["path"]).is_absolute() else ROOT / m["path"]
    files = [p] if p.is_file() else list(p.rglob("*")) if p.is_dir() else []
    if m.get("mmproj"):
        files.append(ROOT / m["mmproj"])
    sizes = [f.stat().st_size for f in files if f.is_file()]
    return round(sum(sizes) / 2**20) if sizes else None


def reestimate(reg, variant):
    """Recompute vram_mb/ram_mb for the current pools; entries given --vram/--ram keep theirs."""
    changed = []
    for mid, m in sorted(reg["models"].items()):
        if m.get("mem_manual"):
            continue
        size = model_size_mb(m)
        if size is None:
            continue
        m["size_mb"] = size
        cpu = m.get("device") == "cpu" or variant == "cpu"
        vram, ram = estimate(m["kind"], m["backend"], size, cpu, reg["pools"])
        if abs(vram - m.get("vram_mb", 0)) > 16 or abs(ram - m.get("ram_mb", 0)) > 16:  # don't log rounding
            changed.append(f"{mid}: vram {m.get('vram_mb')} -> {vram}, ram {m.get('ram_mb')} -> {ram}")
        m["vram_mb"], m["ram_mb"] = vram, ram
    return changed


def make_id(name):
    stem = re.sub(r"\.(gguf|safetensors|sft|bin|ckpt)$", "", Path(name).name, flags=re.I)
    stem = GGUF_SPLIT.sub("", stem + ".gguf").removesuffix(".gguf")
    return re.sub(r"[^a-z0-9.]+", "-", stem.lower()).strip("-")


def rel(p: Path):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def absp(p):
    path = (ROOT / p) if not Path(p).is_absolute() else Path(p)
    s = path.as_posix()
    return f'"{s}"' if " " in s else s


# VRAM a fully offloaded LLM needs on top of its file: 8k KV cache, compute buffers, CUDA context.
# Measured on an RTX 3050 4 GB: Qwen3-4B Q4_K_M, 2382 MiB file -> 3701 MiB with -ngl 99 -np 1.
FULL_OFFLOAD_MB = 1350


def estimate(kind, backend, size_mb, cpu, pools):
    """Rough (vram_mb, ram_mb) for the capacity router. Override with --vram/--ram."""
    v = pools["vram"]
    if cpu:
        # audio.cpp's runtime adds ~1.3 GB on its own (Kokoro: 190 MB file, 1.7 GB resident, measured)
        return 0, int(size_mb * 1.2 + (1500 if backend == "audiocpp" else 400))
    if kind in ("llm",):
        if size_mb + FULL_OFFLOAD_MB <= v:  # fits entirely: backend_command() then forces every layer onto the GPU
            return int(size_mb + FULL_OFFLOAD_MB), 600
        need = size_mb + 700  # weights + KV cache at 8k ctx + compute buffers
        vram = int(min(v, need))
        return vram, int(max(0, need - vram) + 600)
    if kind in ("image", "video", "worker"):
        return v, int(size_mb + 1500)  # diffusion activations use whatever VRAM is left
    if backend == "audiocpp":
        return min(v, int(size_mb * 1.4 + 500)), 600
    return min(v, int(size_mb * 1.3 + 300)), 500


def register(reg, mid, entry, force=False):
    if mid in reg["models"] and not force:
        sys.exit(f"id '{mid}' already registered; pick another with --id or remove it first")
    if mid in reg.get("roles", {}):
        sys.exit(f"'{mid}' is a role name; model ids and role names must differ (pick another --id)")
    entry = {k: v for k, v in entry.items() if v not in (None, "", {}, [])}
    if entry.get("backend") == "audiocpp" and "port" not in entry:
        used = {m.get("port") for m in reg["models"].values()}
        entry["port"] = next(p for p in AUDIO_PORTS if p not in used)
    reg["models"][mid] = entry
    # The first model of a kind fills that kind's unset roles, so a fresh install works with no `role` commands.
    roles = reg.setdefault("roles", {})
    for role in DEFAULT_ROLES.get(entry["kind"], []):
        if role not in roles and role not in reg["models"]:
            roles[role] = mid
            print(f"role '{role}' -> '{mid}' (first {entry['kind']} model; change with `role {role} <model>`)")
    save_registry(reg)
    render(reg)
    print(f"\nregistered '{mid}'  ({entry['kind']} via {entry['backend']}, "
          f"vram {entry['vram_mb']} MiB / ram {entry['ram_mb']} MiB, ttl {entry['ttl']}s)")


def build_entry(args, reg, kind, backend, path, detail, extra, size_mb, source):
    cpu = args.cpu or (backend == "whisper")  # the bundled whisper build is CPU/OpenBLAS
    vram, ram = estimate(kind, backend, size_mb, cpu or installed_variant() == "cpu", reg["pools"])
    default_ttl = 0 if args.always_on else (300 if kind in ("image", "video", "worker") else 900)
    return {
        "kind": kind,
        "backend": backend,
        "path": rel(path),
        "mmproj": rel(args.mmproj) if getattr(args, "mmproj", None) else None,
        "model_flag": extra.get("model_flag"),
        "family": args.family or extra.get("family"),
        "task": args.task or extra.get("task"),
        "device": "cpu" if cpu else "gpu",
        "always_on": bool(args.always_on),
        "ttl": args.ttl if args.ttl is not None else default_ttl,
        "vram_mb": args.vram if args.vram is not None else vram,
        "ram_mb": args.ram if args.ram is not None else ram,
        "mem_manual": args.vram is not None or args.ram is not None or None,  # `detect` leaves these alone
        "size_mb": round(size_mb) or None,
        "args": args.args,
        "description": detail,
        "source": source,
    }


BACKEND_FOR = {"llm": "llama", "image": "sd", "video": "sd"}


def backend_for(kind, extra):
    if "family" in extra:
        return "audiocpp"
    if kind == "stt":
        return "whisper"
    return BACKEND_FOR.get(kind)


# ----------------------------------------------------------------------------- config rendering

def q(s):
    return json.dumps(str(s))  # a JSON string is a valid YAML double-quoted scalar


def backend_command(mid, m):
    """(cmd, proxy, checkEndpoint) for one registry entry."""
    b = BIN.as_posix()
    extra = (" " + m["args"]) if m.get("args") else ""
    path = absp(m["path"])
    if m["backend"] == "llama":
        cmd = f"{b}/llama/llama-server.exe --host 127.0.0.1 --port ${{PORT}} -m {path}"
        if m.get("mmproj"):
            cmd += f" --mmproj {absp(m['mmproj'])}"
        if m.get("device") == "cpu":
            cmd += " -ngl 0"
        elif ("-ngl" not in extra and "--gpu-layers" not in extra
              and m.get("vram_mb", 0) >= m.get("size_mb", 1e9) + FULL_OFFLOAD_MB):
            # llama.cpp's automatic fit is cautious (it left 1.8 GB of a 4 GB card unused: 30 vs 57 tok/s),
            # so when estimate() says the whole model fits, say so explicitly
            cmd += " -ngl 99"
        if "-np " not in extra and "--parallel" not in extra:
            cmd += " -np 1"  # one user: a single slot gets the whole context instead of a quarter of it
        if "-c " not in extra and "--ctx-size" not in extra:
            cmd += " -c 8192"
        return cmd + extra, None, "/health"
    if m["backend"] == "sd":
        flag = m.get("model_flag") or "-m"
        cmd = f"{b}/sd/sd-server.exe --listen-ip 127.0.0.1 --listen-port ${{PORT}} {flag} {path}"
        return cmd + extra, None, "/v1/models"
    if m["backend"] == "whisper":
        cmd = (f"{b}/whisper/whisper-server.exe --host 127.0.0.1 --port ${{PORT}} -m {path}"
               f" -l auto -t 4 --inference-path /v1/audio/transcriptions")
        return cmd + extra, None, "/health"
    if m["backend"] == "audiocpp":
        cfg = CONFIG / "audiocpp" / f"{mid}.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps({
            "host": "127.0.0.1",
            "port": m["port"],
            "backend": "cpu" if m.get("device") == "cpu" else installed_variant(),
            "threads": 4,
            "lazy_load": False,
            "model_spec_override": (BIN / "audiocpp" / "model_specs").as_posix(),
            "models": [{"id": mid, "family": m["family"], "path": (ROOT / m["path"]).as_posix(),
                        "task": m.get("task", "tts"), "mode": "offline"}],
        }, indent=2) + "\n", encoding="utf-8")
        cmd = f"{b}/audiocpp/audiocpp_server.exe --config {absp(cfg)} --no-ui"
        return cmd + extra, f"http://127.0.0.1:{m['port']}", "/health"
    if m["backend"] == "worker":
        wdir = absp(m["path"])
        cmd = f"{b}/uv/uv.exe run --project {wdir} python {wdir}/server.py --port ${{PORT}}"
        if m.get("model_dir"):
            cmd += f" --model-dir {absp(m['model_dir'])}"
        return cmd + extra, None, "/health"
    raise ValueError(f"unknown backend {m['backend']}")


def render(reg=None):
    reg = reg or load_registry()
    CONFIG.mkdir(exist_ok=True)
    pools = reg["pools"]
    out = [
        "# GENERATED by scripts/manage.py from models/registry.json. Edit the registry, then run",
        "# `python scripts/manage.py render` (llama-swap reloads this file automatically).",
        "healthCheckTimeout: 600",
        "logLevel: info",
        "startPort: 5800",
        "",
        "# capacity router: every model declares the MiB it needs from each pool; any set of",
        "# models that fits stays loaded, otherwise the cheapest-to-evict ones are unloaded.",
        "routing:",
        "  router:",
        "    use: capacity",
        "    settings:",
        "      capacity:",
        "        pools:",
        f"          vram: {pools['vram']}",
        f"          ram: {pools['ram']}",
        "",
    ]
    # A registry copied from another PC may list models whose files are not here; leave them out
    # of the config (with a warning) instead of letting llama-swap fail when they are requested.
    models = {}
    for mid, m in sorted(reg["models"].items()):
        p = Path(m["path"]) if Path(m["path"]).is_absolute() else ROOT / m["path"]
        if p.exists():
            models[mid] = m
        else:
            print(f"warning: '{mid}' skipped, {m['path']} not found "
                  f"(re-add it or `remove {mid}`)", file=sys.stderr)
    # roles -> llama-swap aliases (requests may name the role instead of the model id)
    aliases = {}
    for role, mid in sorted(reg.get("roles", {}).items()):
        if mid in models:
            aliases.setdefault(mid, []).append(role)
        else:
            print(f"warning: role '{role}' points to '{mid}', which is not available; requests for "
                  f"'{role}' will fail (`role {role} <model>` to reassign)", file=sys.stderr)
    if aliases:
        out += ["# roles appear in /v1/models next to the real ids", "includeAliasesInList: true", ""]
    always = [mid for mid, m in models.items() if m.get("always_on")]
    if always:
        out += ["hooks:", "  on_startup:", "    preload:"] + [f"      - {q(mid)}" for mid in always] + [""]
    out.append("models:")
    for mid, m in models.items():
        cmd, proxy, check = backend_command(mid, m)
        out += [f"  {q(mid)}:",
                f"    description: {q(m.get('description', ''))}",
                f"    cmd: {q(cmd)}"]
        if mid in aliases:
            out.append(f"    aliases: [{', '.join(q(r) for r in aliases[mid])}]")
            # llama-swap forwards the requested name; backends that check it (audio.cpp) must see the real id
            out.append(f"    useModelName: {q(mid)}")
        if proxy:
            out.append(f"    proxy: {q(proxy)}")
        out += [f"    checkEndpoint: {q(check)}",
                f"    ttl: {m.get('ttl', 0)}",
                "    capacities:",
                f"      vram: {m.get('vram_mb', 0)}",
                f"      ram: {m.get('ram_mb', 0)}"]
        if m.get("always_on"):
            out.append("    evict_cost: 100")
        out += capabilities_yaml(m)
    if not models:
        out[-1] = "models: {}"
    (CONFIG / "llama-swap.yaml").write_text("\n".join(out) + "\n", encoding="utf-8")
    sync_descriptions(reg)


def capabilities_yaml(m):
    """What a model takes and produces, so the UI lists it on the right playground tab (and its voices)."""
    kind = m["kind"]
    io = {"llm": (["text", "image"] if m.get("mmproj") else ["text"], ["text"]),
          "stt": (["audio"], ["text"]),
          "tts": (["text"], ["audio"]),
          "image": (["text", "image"], ["image"]),
          "video": (["text", "image"], ["video"])}.get(kind)
    if not io:
        return []
    lines = ["    capabilities:",
             f"      in: [{', '.join(q(x) for x in io[0])}]",
             f"      out: [{', '.join(q(x) for x in io[1])}]"]
    if kind == "tts":
        voices = audiocpp_voices(ROOT / m["path"]) if m["backend"] == "audiocpp" else m.get("voices")
        if voices:
            lines.append(f"      voices: [{', '.join(q(v) for v in voices)}]")
    return lines


DESC_START, DESC_END = "<!-- registry:start -->", "<!-- registry:end -->"


def sync_descriptions(reg):
    """Keep the generated 'registered models' section of each models/<kind> folder_description.md current."""
    # models/<kind>/ only: these descriptions are per-machine (untracked); workers/folder_description.md is
    # shared in git, so its list is not generated (`manage.py list` shows registered workers)
    dirs = {(MODELS / k).resolve() for k in ("llm", "image", "video", "stt", "tts", "unknown")}
    by_dir = {}
    for mid, m in sorted(reg["models"].items()):
        d = (ROOT / m["path"]).resolve().parent if m["backend"] != "worker" else WORKERS.resolve()
        if d.parent in dirs:  # models/<kind>/<id>/file -> listed under models/<kind>
            d = d.parent
        line = f"- `{mid}` -> `{rel(ROOT / m['path'])}`: {m.get('description', '')} ({m['backend']}, {m.get('device')})"
        by_dir.setdefault(d, []).append(line)
    for d in (dirs | set(by_dir)) - {WORKERS.resolve()}:
        desc = d / "folder_description.md"
        if not desc.exists():
            continue
        text = desc.read_text(encoding="utf-8")
        body = "\n".join(by_dir.get(d, ["- (none)"]))
        block = f"{DESC_START}\n## Registered models (generated by scripts/manage.py)\n{body}\n{DESC_END}"
        if DESC_START in text:
            text = re.sub(re.escape(DESC_START) + ".*?" + re.escape(DESC_END), lambda _: block, text, flags=re.S)
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
        desc.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- HF routes

def pick_gguf(files, want):
    """Pick one GGUF (all parts if split) from a repo listing."""
    models = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
    groups = {}
    for f in models:
        groups.setdefault(GGUF_SPLIT.sub(".gguf", f), []).append(f)
    if not groups:
        return None
    sizes = {k: sum(files[p] for p in v) for k, v in groups.items()}
    print("  available GGUF files:")
    for k in sorted(groups, key=sizes.get):
        print(f"    {sizes[k] / 2**30:6.2f} GiB  {k}")
    for pref in ([want] if want else []) + ([] if want else QUANT_PREFERENCE):
        hits = [k for k in groups if pref.lower() in Path(k).name.lower()]
        if hits:
            return sorted(groups[min(hits, key=sizes.get)])
    if want:
        sys.exit(f"no GGUF matching --quant {want}")
    return sorted(groups[min(groups, key=sizes.get)])


def pick_mmproj(files):
    mm = [f for f in files if f.lower().endswith(".gguf") and "mmproj" in f.lower()]
    for pref in ("f16", "bf16", "q8", ""):
        hits = [f for f in mm if pref in f.lower()]
        if hits:
            return min(hits, key=files.get)
    return None


def suggest_audiocpp(repo):
    """Point TTS/ASR repos at a ready-made audio.cpp GGUF package when one exists."""
    try:
        tree = http_json(f"{HF}/api/models/audio-cpp/audio.cpp-gguf/tree/main")
    except Exception:
        return []
    # every word of the repo name must appear: "Underdog-Woof" must not match "Underdog-Bark-3.0-GGUF"
    words = [w for w in re.split(r"[^a-z0-9]+", repo.split("/")[-1].lower()) if len(w) > 2 and w != "gguf"]
    return [t["path"] for t in tree if t["type"] == "directory" and words
            and all(w in t["path"].lower() for w in words)]


def plan_hf(repo, args):
    """Decide how a HF repo is served. Returns a dict describing the route."""
    info = hf_info(repo)
    files, cfg = info["files"], info.get("config") or {}
    tag, lib = info.get("pipeline_tag") or "", info.get("library_name") or ""
    total_mb = sum(files.values()) / 2**20
    print(f"repo {repo}: pipeline={tag or '?'} library={lib or '?'} files={len(files)} ({total_mb:,.0f} MiB)")

    if args.file:
        if args.file not in files:
            sys.exit(f"--file {args.file} not in repo")
        chosen = [args.file]
    elif any(f.lower().endswith(".gguf") for f in files):
        chosen = pick_gguf(files, args.quant)
    elif any(re.match(r"(.*/)?ggml-.*\.bin$", f) for f in files):
        print("  whisper ggml files:", ", ".join(f for f in files if f.endswith(".bin")))
        sys.exit("pick one with --file (e.g. ggml-small-q5_1.bin)")
    else:
        chosen = None

    if chosen:
        kind, detail, extra = sniff(lambda: RangeReader(hf_url(repo, chosen[0])), chosen[0])
        plan = {"route": "file", "repo": repo, "files": chosen, "kind": kind, "detail": detail, "extra": extra,
                "size_mb": sum(files[f] for f in chosen) / 2**20}
        if kind == "llm" and pick_mmproj(files):  # vision/audio-capable LLM repos ship a projector
            plan["mmproj"] = pick_mmproj(files)
        return plan

    archs = cfg.get("architectures") or []
    st = [f for f in files if f.endswith(".safetensors")]
    if lib == "transformers" and archs and st and tag in ("text-generation", "image-text-to-text", ""):
        kind, detail, _ = sniff(lambda: RangeReader(hf_url(repo, st[0])), st[0])
        if kind == "llm":
            return {"route": "convert", "repo": repo, "files": files, "arch": archs[0], "kind": "llm",
                    "detail": f"{archs[0]} (converted from {repo})", "size_mb": sum(files[f] for f in st) / 2**20}
    # Diffusion repos often ship single-file checkpoints at the top level next to the diffusers
    # folders (unet/, vae/, ...); sd.cpp runs those. Prefer fp16 variants, then the smallest.
    top = sorted((f for f in files if "/" not in f and f.endswith((".safetensors", ".sft", ".ckpt"))),
                 key=lambda f: ("fp16" not in f.lower(), files[f]))
    for f in top[:3]:
        if f.endswith(".ckpt"):
            continue
        kind, detail, extra = sniff(lambda: RangeReader(hf_url(repo, f)), f)
        if kind in ("image", "video"):
            return {"route": "file", "repo": repo, "files": [f], "kind": kind, "detail": detail, "extra": extra,
                    "size_mb": files[f] / 2**20}

    reason = f"library '{lib or 'unknown'}', pipeline '{tag or 'unknown'}' has no native runtime here"
    plan = {"route": "worker", "repo": repo, "kind": "worker", "detail": f"{repo}: {reason}",
            "size_mb": total_mb, "library": lib, "pipeline": tag}
    if tag in ("text-to-speech", "automatic-speech-recognition", "text-to-audio", "audio-to-audio"):
        plan["audiocpp"] = suggest_audiocpp(repo)
    return plan


def llama_build_tag():
    out = subprocess.run([str(BIN / "llama" / "llama-server.exe"), "--version"], capture_output=True, text=True,
                         env=_env()).stderr
    m = re.search(r"build (\d+)", out)
    if not m:
        sys.exit("cannot read llama.cpp build number from llama-server --version")
    return f"b{m.group(1)}"


def _env():
    env = dict(os.environ)
    env["PATH"] = str(BIN / "cuda") + os.pathsep + env.get("PATH", "")
    env.setdefault("UV_CACHE_DIR", str(ROOT / ".uv-cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(TOOLS / "python"))
    env.setdefault("HF_HOME", str(MODELS / "hf-cache"))
    return env


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, env=_env(), **kw)


def ensure_converter():
    """llama.cpp's convert_hf_to_gguf.py (matching the installed build) in its own venv."""
    src, venv = TOOLS / "llama.cpp-src", TOOLS / "convert-venv"
    if not (src / "convert_hf_to_gguf.py").exists():
        tag = llama_build_tag()
        zpath = ROOT / "_downloads" / f"llama.cpp-{tag}-src.zip"
        download(f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{tag}.zip", zpath)
        with zipfile.ZipFile(zpath) as z:
            top = z.namelist()[0].split("/")[0]
            for name in z.namelist():
                target = src / name[len(top) + 1:]
                if name.endswith("/") or not name[len(top) + 1:]:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
    py = venv / "Scripts" / "python.exe"
    if not py.exists():
        uv = BIN / "uv" / "uv.exe"
        run([uv, "venv", "--python", "3.12", venv])
        run([uv, "pip", "install", "--python", py, "--index-strategy", "unsafe-best-match",
             "-r", src / "requirements" / "requirements-convert_hf_to_gguf.txt"])
    return src, py


def converter_supports(src, arch):
    needle = f'"{arch}"'
    return any(needle in p.read_text(encoding="utf-8", errors="ignore")
               for p in [src / "convert_hf_to_gguf.py", *src.glob("conversion/**/*.py")] if p.exists())


def do_convert(plan, args, mid, local_dir=None):
    src, py = ensure_converter()
    if not converter_supports(src, plan["arch"]):
        sys.exit(f"llama.cpp {llama_build_tag()} cannot convert architecture {plan['arch']}; "
                 f"use `add {plan['repo']} --worker` to wrap it in Python instead")
    hf_dir = Path(local_dir) if local_dir else ROOT / "_downloads" / "hf" / plan["repo"].replace("/", "--")
    st_present = any(f.endswith(".safetensors") for f in plan["files"])
    for f, size in plan["files"].items():
        if SKIP_FOR_CONVERT.search(f) or (st_present and f.endswith((".bin", ".pt", ".pth"))):
            continue
        download(hf_url(plan["repo"], f), hf_dir / f, size)
    out_dir = MODELS / "llm" / mid
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / f"{mid}-f16.gguf"
    run([py, src / "convert_hf_to_gguf.py", hf_dir, "--outfile", f16, "--outtype", "auto"])
    cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    mmproj = None
    if "vision_config" in cfg or "audio_config" in cfg:
        mmproj = out_dir / f"mmproj-{mid}-f16.gguf"
        run([py, src / "convert_hf_to_gguf.py", hf_dir, "--mmproj", "--outfile", mmproj, "--outtype", "f16"])
    qtype = args.quant or ("Q4_K_M" if plan["size_mb"] > 5500 else "Q8_0")  # ~>3B params at bf16
    final = out_dir / f"{mid}-{qtype}.gguf"
    run([BIN / "llama" / "llama-quantize.exe", f16, final, qtype])
    f16.unlink()  # intermediate written by this script
    if not args.keep_source and not local_dir:
        shutil.rmtree(hf_dir)  # HF download made by this script; the GGUF is what we keep
    write_model_desc(out_dir, mid, f"{plan['detail']}, quantized to {qtype}", [final.name] + ([mmproj.name] if mmproj else []))
    return final, mmproj, final.stat().st_size / 2**20


TTS_PIPELINES = ("text-to-speech", "text-to-audio")
SNAPSHOT_MAX_MB = 4096  # TTS repos up to this size are downloaded whole (code + weights) into models/tts/<id>


def snapshot_repo(repo, dest):
    """Download every file of a HF repo into dest (skipping files already there). Returns total MiB."""
    files = hf_info(repo)["files"]
    total = sum(files.values()) / 2**20
    if total > SNAPSHOT_MAX_MB:
        sys.exit(f"{repo} is {total:,.0f} MiB; too big to download whole. Use the generic worker (--worker without "
                 f"a TTS repo) and let the model's own code fetch its weights.")
    for f, size in sorted(files.items()):
        target = dest / f
        if not (target.exists() and target.stat().st_size == size):
            download(hf_url(repo, f), target, size)
    return total


def add_requirements(wdir, model_dir):
    """Copy plain `package>=x` lines from the repo's requirements.txt into the worker's pyproject."""
    req = model_dir / "requirements.txt"
    if not req.exists():
        return []
    deps = [ln.split("#")[0].strip() for ln in req.read_text(encoding="utf-8").splitlines()]
    deps = [d for d in deps if d and not d.startswith(("-", "git+", "http"))]
    if deps:
        pp = wdir / "pyproject.toml"
        pp.write_text(pp.read_text(encoding="utf-8").replace(
            "dependencies = []", "dependencies = [" + ", ".join(json.dumps(d) for d in deps) + "]"), encoding="utf-8")
    return deps


def scaffold_worker(plan, mid, template="_template"):
    wdir = WORKERS / mid
    if wdir.exists():
        sys.exit(f"{rel(wdir)} already exists")
    shutil.copytree(WORKERS / template, wdir, ignore=shutil.ignore_patterns(".venv", "__pycache__", "folder_description.md"))
    pp = wdir / "pyproject.toml"
    pp.write_text(re.sub(r'"worker-template[a-z-]*"', f'"worker-{mid}"', pp.read_text(encoding="utf-8"), count=1),
                  encoding="utf-8")
    (wdir / "folder_description.md").write_text(
        f"# workers/{mid}\n\n"
        f"Python worker for `{plan['repo']}` ({plan.get('pipeline') or 'unknown task'}, library "
        f"`{plan.get('library') or 'unknown'}`). Created by `scripts/manage.py add` because no native runtime "
        f"in this stack supports it. Weights are fetched by the model's own code into `models/hf-cache`.\n\n"
        f"- `server.py`: HTTP wrapper; fill in `load()` and `run()` from the model card's example code.\n"
        f"- `pyproject.toml`: the worker's dependencies; add the model's package, then run "
        f"`bin\\uv\\uv.exe sync --project workers\\{mid}`.\n"
        f"- `uv.lock`: exact resolved package versions, written by uv.\n"
        f"- `.venv/`: created by uv; safe to delete (rebuilt on next start).\n\n"
        f"Call it through llama-swap: `POST http://127.0.0.1:8080/upstream/{mid}/run` with a JSON body.\n"
        f"Model card: {HF}/{plan['repo']}\n", encoding="utf-8")
    return wdir


def write_model_desc(folder, mid, detail, files):
    (folder / "folder_description.md").write_text(
        f"# {rel(folder)}\n\nFiles for model `{mid}` ({detail}), downloaded by `scripts/manage.py add`.\n\n"
        + "\n".join(f"- `{f}`" for f in files) + "\n\nRemove with `python scripts/manage.py remove "
        f"{mid} --delete-files`.\n", encoding="utf-8")


# ----------------------------------------------------------------------------- commands

def cmd_inspect(args):
    src = args.source
    if Path(src).exists():
        p = Path(src)
        kind, detail, extra = sniff(lambda: open(p, "rb"), p.name)
        print(f"{p.name}: {kind} - {detail}" + (f" {extra}" if extra else ""))
    else:
        plan = plan_hf(src, args)
        print(json.dumps({k: v for k, v in plan.items() if k != "files" or isinstance(v, list)}, indent=2))


def add_tts_worker(args, reg, src, plan, mid):
    """TTS repo with its own Python code: download it whole, wrap it in workers/_template_tts (OpenAI speech
    endpoint), register it as kind `tts` so it can fill the `tts` role."""
    model_dir = MODELS / "tts" / mid
    print(f"\nText-to-speech repo: downloading it into {rel(model_dir)} and wrapping it as a tts worker")
    size_mb = snapshot_repo(src, model_dir)
    wdir = WORKERS / mid
    if not (wdir.exists() and args.force):  # re-adding with --force keeps your filled-in server.py
        wdir = scaffold_worker(plan, mid, template="_template_tts")
    deps = add_requirements(wdir, model_dir)
    run([BIN / "uv" / "uv.exe", "sync", "--project", wdir])
    entry = build_entry(args, reg, "tts", "worker", wdir, f"{src}: Python TTS worker", {}, size_mb, f"hf:{src}")
    entry["model_dir"] = rel(model_dir)
    entry["voices"] = [v.strip() for v in (args.voices or "").split(",") if v.strip()]
    register(reg, mid, entry, args.force)
    print(f"\ndependencies from requirements.txt: {', '.join(deps) or '(none)'}"
          f"\nnext: fill in load() in {rel(wdir)}/server.py (the model's code is in {rel(model_dir)}), then "
          f"`role tts {mid}` and `check`. Until then it answers with a test tone."
          + ("" if entry["voices"] else "\nTip: pass --voices a,b to list its voices in /v1/models (re-add with --force)."))


def cmd_add(args):
    reg = load_registry()
    src = args.source
    if Path(src).exists():
        p = Path(src).resolve()
        folder = p if p.is_dir() else p.parent
        if p.is_dir() or (p.suffix.lower() in (".safetensors", ".sft") and (folder / "config.json").exists()):
            cfg_path = folder / "config.json"
            sts = sorted(folder.glob("*.safetensors"))
            if not cfg_path.exists() or not sts:
                sys.exit(f"{folder} needs config.json, tokenizer files and *.safetensors to be converted; "
                         f"download the whole model repo (`add <owner/repo>` does this for you)")
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            arch = (cfg.get("architectures") or ["?"])[0]
            kind, detail, _ = sniff(lambda: open(sts[0], "rb"), sts[0].name)
            if kind != "llm":
                sys.exit(f"{folder.name}: {detail}; only transformers LLMs can be converted from a folder")
            mid = args.id or make_id(folder.name)
            plan = {"repo": folder.name, "arch": arch, "files": {f.name: 0 for f in sts}, "detail": f"{arch} (converted from {folder})",
                    "size_mb": sum(f.stat().st_size for f in sts) / 2**20}
            final, mmproj, size_mb = do_convert(plan, args, mid, local_dir=folder)
            args.mmproj = args.mmproj or mmproj
            entry = build_entry(args, reg, "llm", "llama", final, plan["detail"], {}, size_mb, f"local:{folder.name}")
            return register(reg, mid, entry, args.force)
        kind, detail, extra = sniff(lambda: open(p, "rb"), p.name)
        print(f"{p.name}: {kind} - {detail}")
        mid = args.id or make_id(p.name)
        if kind == "llm" and p.suffix.lower() in (".safetensors", ".sft"):
            sys.exit(f"{p.name} is an LLM in safetensors format, which llama.cpp cannot load, and there is no "
                     f"config.json/tokenizer next to it to convert it. Put the whole model folder together "
                     f"(or run `add <owner/repo>`), then `add <folder>`.")
        if args.worker:
            plan = {"repo": p.name, "kind": "worker", "detail": f"Python worker for local file {p.name}"}
            wdir = scaffold_worker(plan, mid)
            run([BIN / "uv" / "uv.exe", "sync", "--project", wdir])
            entry = build_entry(args, reg, "worker", "worker", wdir, plan["detail"], {},
                                p.stat().st_size / 2**20, f"local:{p.name}")
            return register(reg, mid, entry, args.force)
        if kind in ("unknown", "component", "mmproj"):
            sys.exit(f"cannot serve a {kind} file on its own. "
                     + ("Attach it to an LLM with --mmproj." if kind == "mmproj" else
                        "Pass it via --args to the model that uses it." if kind == "component" else
                        "Find the repo it came from and `add <repo>` (or --worker)."))
        backend = backend_for(kind, extra)
        entry = build_entry(args, reg, kind, backend, p, detail, extra, p.stat().st_size / 2**20, f"local:{p.name}")
        return register(reg, mid, entry, args.force)

    if args.worker:
        info = hf_info(src)
        plan = {"route": "worker", "repo": src, "kind": "worker", "detail": src,
                "size_mb": sum(info["files"].values()) / 2**20,
                "pipeline": info.get("pipeline_tag") or "", "library": info.get("library_name") or ""}
    else:
        plan = plan_hf(src, args)
    print(f"  -> {plan['kind']}: {plan['detail']}  [route: {plan['route']}]")
    mid = args.id or make_id(plan["files"][0] if plan["route"] == "file" else src.split("/")[-1])

    if plan["route"] == "worker":
        if plan.get("audiocpp"):
            print("\nThis looks like an audio model. audio.cpp has native GGUF packages that may match:")
            for d in plan["audiocpp"]:
                print(f"    python scripts/manage.py add audio-cpp/audio.cpp-gguf --file {d}/<file>.gguf")
            print("Run with --worker to wrap the original Python code instead.")
            if not args.worker:
                return
        if plan.get("pipeline") in TTS_PIPELINES:
            return add_tts_worker(args, reg, src, plan, mid)
        if plan["size_mb"] > reg["pools"]["vram"]:
            print(f"warning: repo is {plan['size_mb']:,.0f} MiB, more than the {reg['pools']['vram']} MiB VRAM pool; "
                  f"it may need CPU offload or may not fit at all")
        wdir = scaffold_worker(plan, mid)
        run([BIN / "uv" / "uv.exe", "sync", "--project", wdir])
        entry = build_entry(args, reg, "worker", "worker", wdir, plan["detail"], {}, plan["size_mb"], f"hf:{src}")
        register(reg, mid, entry, args.force)
        print(f"\nnext: add the model's packages to {rel(wdir)}/pyproject.toml, fill in load()/run() in "
              f"{rel(wdir)}/server.py, then `bin\\uv\\uv.exe sync --project {rel(wdir)}`")
        return

    if plan["route"] == "convert":
        final, mmproj, size_mb = do_convert(plan, args, mid)
        args.mmproj = args.mmproj or mmproj
        entry = build_entry(args, reg, "llm", "llama", final, plan["detail"], {}, size_mb, f"hf:{src}")
        return register(reg, mid, entry, args.force)

    kind, extra = plan["kind"], plan["extra"]
    if kind in ("unknown", "component", "mmproj"):
        sys.exit(f"{plan['files'][0]} is a {kind} ({plan['detail']}); nothing to serve on its own")
    folder = MODELS / kind / mid
    for f in plan["files"]:
        download(hf_url(src, f), folder / Path(f).name, 0)
    if plan.get("mmproj") and not args.mmproj:
        download(hf_url(src, plan["mmproj"]), folder / Path(plan["mmproj"]).name)
        args.mmproj = folder / Path(plan["mmproj"]).name
    write_model_desc(folder, mid, plan["detail"], [Path(f).name for f in plan["files"]]
                     + ([Path(plan["mmproj"]).name] if plan.get("mmproj") else []))
    if kind in ("image", "video") and extra.get("model_flag") == "--diffusion-model" and not args.args:
        print("note: standalone diffusion models need their VAE/text encoders; add them with "
              "--args \"--vae <file> --t5xxl <file> ...\" (re-run add with --force)")
    entry = build_entry(args, reg, kind, backend_for(kind, extra), folder / Path(plan["files"][0]).name,
                        plan["detail"], extra, plan["size_mb"], f"hf:{src}")
    register(reg, mid, entry, args.force)


def cmd_hardware(args):
    print(json.dumps({**detect_hardware(), "installed": installed_variant()}, indent=2))


def cmd_detect(args):
    reg = load_registry()
    hw = detect_hardware()
    variant = installed_variant()
    reg["hardware"] = hw
    print(f"GPU: {hw['gpu']} ({hw['vram_total_mb']} MiB)   RAM: {hw['ram_total_mb']} MiB   "
          f"build: {variant}")
    if hw["variant"] != variant:
        print(f"warning: bin/ has the {variant} build but this PC suits {hw['variant']}; run "
              f"`powershell -ExecutionPolicy Bypass -File scripts\\setup.ps1` to switch", file=sys.stderr)
    if reg.get("pools_auto", True):
        reg["pools_auto"] = True
        reg["pools"] = pools_for(hw, variant)
        print(f"pools: vram {reg['pools']['vram']} MiB, ram {reg['pools']['ram']} MiB  "
              f"(set \"pools_auto\": false in models/registry.json to keep your own)")
    else:
        print(f"pools: vram {reg['pools']['vram']} MiB, ram {reg['pools']['ram']} MiB  (pools_auto is false)")
    for line in reestimate(reg, variant):
        print(f"  re-estimated {line}")
    for mid, m in sorted(reg["models"].items()):
        if m.get("vram_mb", 0) > reg["pools"]["vram"] or m.get("ram_mb", 0) > reg["pools"]["ram"]:
            print(f"warning: '{mid}' needs vram {m.get('vram_mb')} / ram {m.get('ram_mb')} MiB, more than this "
                  f"PC's pools, so it cannot be loaded here (pick a smaller quant or --cpu)", file=sys.stderr)
    save_registry(reg)
    render(reg)


def cmd_list(args):
    reg = load_registry()
    print(f"pools: vram {reg['pools']['vram']} MiB, ram {reg['pools']['ram']} MiB")
    roles = {}
    for role, mid in reg.get("roles", {}).items():
        roles.setdefault(mid, []).append(role)
    for mid, m in sorted(reg["models"].items()):
        flag = " always-on" if m.get("always_on") else f" ttl={m.get('ttl')}s"
        tag = f"  roles: {', '.join(sorted(roles[mid]))}" if mid in roles else ""
        print(f"  {mid:32} {m['kind']:7} {m['backend']:9} {m.get('device', ''):4} "
              f"vram={m.get('vram_mb'):>5} ram={m.get('ram_mb'):>5}{flag}  {m['path']}{tag}")


def cmd_role(args):
    reg = load_registry()
    roles = reg.setdefault("roles", {})
    if not args.name:
        for role in sorted(set(roles) | set(STANDARD_ROLES)):
            target = roles.get(role, "-")
            print(f"  {role:10} -> {target:28} {STANDARD_ROLES.get(role, '(custom role)')}")
        return
    if not ROLE_RE.match(args.name):
        sys.exit(f"role names are lowercase letters, digits and '-', starting with a letter: '{args.name}'")
    if args.unset:
        if roles.pop(args.name, None) is None:
            sys.exit(f"role '{args.name}' is not set")
        print(f"role '{args.name}' unset")
    else:
        if not args.model:
            sys.exit(f"usage: role {args.name} <model-id>   (or --unset)")
        if args.model not in reg["models"]:
            sys.exit(f"no model '{args.model}' (see `list`)")
        if args.name in reg["models"]:
            sys.exit(f"'{args.name}' is a model id; role names and model ids must differ")
        old = roles.get(args.name)
        roles[args.name] = args.model
        print(f"role '{args.name}' -> '{args.model}'" + (f"  (was '{old}')" if old and old != args.model else ""))
    save_registry(reg)
    render(reg)


# ----------------------------------------------------------------------------- contract check

def _api(base, path, body=None, raw=None, ctype=None, timeout=600):
    """POST JSON (body) or raw bytes, or GET when neither; returns (status, bytes)."""
    data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", ctype or "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _multipart(fields, file_field, filename, content):
    boundary = f"----localai{int(time.time() * 1000)}"
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items()]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                 f'Content-Type: audio/wav\r\n\r\n'.encode() + content + b"\r\n")
    return b"".join(parts) + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


AGENT_TOOL = {"type": "function", "function": {
    "name": "get_time", "description": "Returns the current local time.",
    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []}}}


def cmd_check(args):
    """Exercise the public contract (CONTRACT.md) against the running stack, through roles only."""
    reg = load_registry()
    base = f"http://{reg.get('listen', '127.0.0.1:8080')}"
    results = []

    def report(name, ok, detail):
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL' if ok is False else 'SKIP'}] {name:28} {detail}")

    try:
        status, _ = _api(base, "/health", timeout=5)
    except OSError as e:
        sys.exit(f"stack not reachable at {base} ({e}); start it with start.ps1")
    report("llama-swap /health", status == 200, f"HTTP {status}")
    status, body = _api(base, "/v1/models", timeout=30)
    listed = {m["id"] for m in json.loads(body).get("data", [])} if status == 200 else set()
    report("/v1/models", status == 200, f"{len(listed)} ids")

    roles = reg.get("roles", {})
    missing = sorted(set(roles) - listed)
    report("roles listed in /v1/models", not missing, "all present" if not missing else f"missing {missing}")

    wav = None
    order = sorted(roles, key=lambda r: (reg["models"].get(roles[r], {}).get("kind") == "stt", r))  # tts before stt
    for role in order:
        m = reg["models"].get(roles[role])
        if not m:
            report(f"role {role}", False, f"points to unknown model '{roles[role]}'")
            continue
        kind, t0 = m["kind"], time.time()
        label = f"role {role} ({roles[role]})"
        try:
            if kind == "llm" and role != "embed":
                st, b = _api(base, "/v1/chat/completions", {"model": role, "max_tokens": 16,
                             "messages": [{"role": "user", "content": "Reply with the word ok."}]})
                ok = st == 200 and json.loads(b)["choices"][0]["message"] is not None
                report(label + " chat", ok, f"HTTP {st}, {time.time() - t0:.1f}s")
                if role == "agent":
                    t0 = time.time()
                    st, b = _api(base, "/v1/chat/completions", {
                        "model": role, "max_tokens": 1024, "tools": [AGENT_TOOL], "tool_choice": "auto",  # room to reason first
                        "messages": [{"role": "user", "content": "What time is it? Use the get_time tool."}]})
                    calls = json.loads(b)["choices"][0]["message"].get("tool_calls") if st == 200 else None
                    report(label + " tool call", bool(calls),
                           f"called {calls[0]['function']['name']}" if calls else f"no tool_calls (HTTP {st})")
            elif role == "embed":
                st, b = _api(base, "/v1/embeddings", {"model": role, "input": "hello"})
                ok = st == 200 and len(json.loads(b)["data"][0]["embedding"]) > 0
                report(label, ok, f"HTTP {st}, {time.time() - t0:.1f}s")
            elif kind == "tts":
                body = {"model": role, "input": "Hello from the contract check."}
                voices = audiocpp_voices(ROOT / m["path"]) if m["backend"] == "audiocpp" else m.get("voices") or []
                if voices:
                    body["voice"] = voices[0]
                st, b = _api(base, "/v1/audio/speech", body)
                ok = st == 200 and b[:4] == b"RIFF"
                wav = b if ok else wav
                report(label, ok, f"HTTP {st}, {len(b)} bytes, {time.time() - t0:.1f}s")
            elif kind == "stt":
                if wav is None:
                    report(label, None, "needs audio from a working tts role")
                    continue
                raw, ctype = _multipart({"model": role}, "file", "check.wav", wav)
                st, b = _api(base, "/v1/audio/transcriptions", raw=raw, ctype=ctype)
                text = json.loads(b).get("text", "").strip() if st == 200 else ""
                report(label, bool(text), f"heard {text[:40]!r}, {time.time() - t0:.1f}s")
            elif kind == "image" and args.all:
                st, b = _api(base, "/v1/images/generations", {"model": role, "prompt": "a red circle",
                                                               "size": "512x512", "n": 1})
                ok = st == 200 and bool(json.loads(b).get("data"))
                report(label, ok, f"HTTP {st}, {time.time() - t0:.1f}s")
            else:
                report(label, None, "slow / evicts the GPU model; run `check --all`" if kind == "image"
                       else f"no automatic probe for {kind}")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            report(label, False, f"{type(e).__name__}: {e}")

    failed = results.count(False)
    print(f"\n{results.count(True)} passed, {failed} failed, {results.count(None)} skipped")
    sys.exit(1 if failed else 0)


def cmd_remove(args):
    reg = load_registry()
    m = reg["models"].pop(args.id, None)
    if not m:
        sys.exit(f"no model '{args.id}'")
    for role in [r for r, mid in reg.get("roles", {}).items() if mid == args.id]:
        del reg["roles"][role]
        print(f"role '{role}' unset (it pointed to '{args.id}')")
    save_registry(reg)
    render(reg)
    (CONFIG / "audiocpp" / f"{args.id}.json").unlink(missing_ok=True)
    print(f"removed '{args.id}' from the registry")
    if args.delete_files:
        target = ROOT / m["path"]
        owned = target.parent if target.parent.name == args.id else target
        if MODELS in owned.parents or WORKERS in owned.parents:
            shutil.rmtree(owned) if owned.is_dir() else owned.unlink()
            print(f"deleted {rel(owned)}")
        else:
            print(f"left {rel(owned)} in place (outside models/ and workers/)")
        model_dir = ROOT / m["model_dir"] if m.get("model_dir") else None  # TTS worker's downloaded repo
        if model_dir and model_dir.is_dir() and MODELS in model_dir.parents:
            shutil.rmtree(model_dir)
            print(f"deleted {rel(model_dir)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def source_opts(p):
        p.add_argument("source", help="Hugging Face repo id (owner/name) or a local model file")
        p.add_argument("--file", help="exact file inside the HF repo")
        p.add_argument("--quant", help="GGUF quant to pick / quantize to, e.g. Q4_K_M, Q8_0")

    p = sub.add_parser("inspect", help="classify a model without downloading it")
    source_opts(p)
    p = sub.add_parser("add", help="download/convert as needed and register a model")
    source_opts(p)
    p.add_argument("--id", help="name used as `model` in API requests")
    p.add_argument("--cpu", action="store_true", help="run on CPU only (no VRAM)")
    p.add_argument("--always-on", action="store_true", help="load at startup and never unload")
    p.add_argument("--ttl", type=int, help="seconds idle before unloading (0 = never)")
    p.add_argument("--args", default="", help="extra backend arguments, e.g. \"-c 16384\" or \"--vae x.safetensors\"")
    p.add_argument("--mmproj", help="multimodal projector GGUF to attach to an LLM")
    p.add_argument("--family", help="audio.cpp family override")
    p.add_argument("--task", help="audio.cpp task override (tts, asr, ...)")
    p.add_argument("--vram", type=int, help="override VRAM estimate (MiB)")
    p.add_argument("--ram", type=int, help="override RAM estimate (MiB)")
    p.add_argument("--worker", action="store_true", help="force the Python worker route")
    p.add_argument("--voices", help="TTS worker voice names, comma-separated; the first is the default")
    p.add_argument("--keep-source", action="store_true", help="keep the HF download after converting")
    p.add_argument("--force", action="store_true", help="replace an existing registry entry with the same id")
    sub.add_parser("list", help="show registered models")
    p = sub.add_parser("remove", help="unregister a model")
    p.add_argument("id")
    p.add_argument("--delete-files", action="store_true", help="also delete its files under models/ or workers/")
    sub.add_parser("render", help="regenerate config/llama-swap.yaml from the registry")
    sub.add_parser("detect", help="size the VRAM/RAM pools to this PC, re-estimate models, render")
    sub.add_parser("hardware", help="print detected GPU, RAM and suitable build variant as JSON")
    p = sub.add_parser("role", help="list roles, or point a role (agent, chat, tts...) at a model")
    p.add_argument("name", nargs="?", help="role name; omit to list")
    p.add_argument("model", nargs="?", help="model id to fill the role")
    p.add_argument("--unset", action="store_true", help="remove the role")
    p = sub.add_parser("check", help="test the public contract against the running stack (roles only)")
    p.add_argument("--all", action="store_true", help="also probe slow roles (image)")

    args = ap.parse_args()
    {"inspect": cmd_inspect, "add": cmd_add, "list": cmd_list, "remove": cmd_remove,
     "render": lambda a: render(), "detect": cmd_detect, "hardware": cmd_hardware,
     "role": cmd_role, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
