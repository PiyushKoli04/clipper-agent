"""Stage 1 & 4: YouTube URL validation, audio download, and section download (yt-dlp)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import config

log = config.log.getChild("download")

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
             "youtu.be", "www.youtu.be"}


class DownloadError(RuntimeError):
    """Raised when yt-dlp fails or a URL is invalid."""


def parse_youtube_url(url: str) -> tuple[str, str]:
    """Validate a YouTube URL and return ``(video_id, canonical_url)``.

    Accepts watch, youtu.be, /shorts/, /live/ and /embed/ links. Everything else
    (other hosts, playlists without a video id, garbage) raises ``ValueError``.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL is empty")
    url = url.strip()
    if len(url) > 300 or any(ch in url for ch in " \t\r\n\"'<>`;|&$"):
        raise ValueError("URL contains invalid characters")
    if "://" not in url:
        url = "https://" + url
    p = urlparse(url)
    if p.scheme not in ("http", "https") or (p.hostname or "").lower() not in _YT_HOSTS:
        raise ValueError("Not a YouTube URL")
    host = p.hostname.lower()
    vid: str | None = None
    if host.endswith("youtu.be"):
        vid = p.path.strip("/").split("/")[0]
    elif p.path == "/watch":
        vid = (parse_qs(p.query).get("v") or [None])[0]
    else:
        m = re.match(r"^/(shorts|live|embed|v)/([^/?#]+)", p.path)
        vid = m.group(2) if m else None
    if not vid or not _ID_RE.match(vid):
        raise ValueError("Could not find a valid 11-character YouTube video id in the URL")
    return vid, f"https://www.youtube.com/watch?v={vid}"


def download_audio(url: str, video_id: str) -> dict[str, Any]:
    """Download audio only and return ``{title, duration, audio_path}``.

    The audio file is cached: if it already exists it is not downloaded again
    (metadata is still fetched, which is cheap).
    """
    import yt_dlp

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    existing = next(iter(sorted(config.AUDIO_DIR.glob(f"{video_id}.*"))), None)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(config.AUDIO_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=existing is None)
            if existing is None:
                existing = Path(ydl.prepare_filename(info))
    except Exception as e:  # yt-dlp raises many types
        raise DownloadError(f"Audio download failed: {e}") from e
    if not existing or not Path(existing).exists():
        raise DownloadError("Audio file missing after download")
    return {
        "title": info.get("title") or video_id,
        "duration": float(info.get("duration") or 0),
        "audio_path": str(existing),
    }


def download_section(url: str, video_id: str, start: float, end: float,
                     max_height: int = 1080) -> Path:
    """Download only ``[start, end]`` seconds (video+audio, <= ``max_height``).

    Uses yt-dlp's ``download_ranges`` (equivalent to ``--download-sections``)
    with forced keyframes at cuts so the section is frame-accurate.
    """
    import yt_dlp
    from yt_dlp.utils import download_range_func

    config.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    start = max(0.0, float(start))
    end = float(end)
    if end <= start:
        raise DownloadError("Section end must be after start")
    tag = f"{video_id}_{int(start * 1000)}_{int(end * 1000)}"
    cached = next(iter(sorted(config.VIDEOS_DIR.glob(f"{tag}.*"))), None)
    if cached and cached.stat().st_size > 0:
        return cached

    opts = {
        "format": (f"bestvideo[height<={max_height}][vcodec^=avc1]+bestaudio[ext=m4a]/"
                   f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best"),
        "merge_output_format": "mp4/mkv",
        "outtmpl": str(config.VIDEOS_DIR / f"{tag}.%(ext)s"),
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise DownloadError(f"Section download failed: {e}") from e
    out = next(iter(sorted(config.VIDEOS_DIR.glob(f"{tag}.*"))), None)
    if out is None or out.stat().st_size == 0:
        raise DownloadError("Section file missing after download")
    return out
