"""Central configuration for ShortsAgent.

Values come from environment variables (optionally loaded from a ``.env`` file)
with sensible defaults, so the app runs out of the box on a laptop.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

try:  # python-dotenv is optional but recommended
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent

# --- storage -------------------------------------------------------------
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage"))
AUDIO_DIR = STORAGE_DIR / "audio"
VIDEOS_DIR = STORAGE_DIR / "videos"
CLIPS_DIR = STORAGE_DIR / "clips"
TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"
MUSIC_DIR = STORAGE_DIR / "music"
DB_PATH = Path(os.getenv("DB_PATH", STORAGE_DIR / "shortsagent.db"))

ASSETS_DIR = BASE_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
MODELS_DIR = ASSETS_DIR / "models"

# --- fonts ---------------------------------------------------------------
FONT_FAMILY = os.getenv("FONT_FAMILY", "Anton")
FONT_FILE = os.getenv("FONT_FILE", "Anton-Regular.ttf")
FONT_URL = os.getenv(
    "FONT_URL",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/anton/Anton-Regular.ttf",
)
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

# --- LLM -----------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "gemini" | "groq"
LLM_FALLBACK = os.getenv("LLM_FALLBACK", "")  # optional fallback provider name
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))

# --- pipeline defaults ---------------------------------------------------
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
MAX_WORKERS = max(1, min(2, int(os.getenv("MAX_WORKERS", "1"))))
OUT_W, OUT_H = 1080, 1920
CLIP_MIN_S, CLIP_MAX_S = 20.0, 50.0
WINDOW_S, WINDOW_OVERLAP_S = 240.0, 30.0
SNAP_PAD_S = 0.2
SECTION_MARGIN_S = 4.0  # extra seconds downloaded around each clip for trimming room
HOOK_DURATION_S = 3.0

log = logging.getLogger("shortsagent")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once and silence Werkzeug HTTP polling noise."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def ensure_dirs() -> None:
    """Create all storage folders."""
    for d in (AUDIO_DIR, VIDEOS_DIR, CLIPS_DIR, TRANSCRIPTS_DIR, MUSIC_DIR, FONTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_ffmpeg() -> bool:
    """Return True if ffmpeg and ffprobe are on PATH; print a clear message otherwise."""
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        print(
            "\n*** ERROR: %s not found on PATH. ***\n"
            "ShortsAgent needs FFmpeg.\n"
            "  Windows: winget install Gyan.FFmpeg   (or https://www.gyan.dev/ffmpeg/builds/)\n"
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg\n" % ", ".join(missing)
        )
        return False
    return True


_CUDA: bool | None = None


def has_cuda() -> bool:
    """Detect an NVIDIA GPU usable for Whisper (CTranslate2) and NVENC."""
    global _CUDA
    if _CUDA is not None:
        return _CUDA
    ok = False
    try:
        import ctranslate2

        ok = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        ok = False
    _CUDA = ok
    return ok


_NVENC: bool | None = None


def has_nvenc() -> bool:
    """True when a CUDA GPU exists AND ffmpeg can actually open h264_nvenc."""
    global _NVENC
    if _NVENC is not None:
        return _NVENC
    ok = False
    if has_cuda() or shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "color=c=black:s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=20,
            )
            ok = r.returncode == 0
        except Exception:
            ok = False
    _NVENC = ok
    return ok
