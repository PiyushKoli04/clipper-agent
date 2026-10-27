"""Stage 7: single-pass FFmpeg render to 1080x1920 (9:16) with burned-in ASS overlay.

All FFmpeg calls are argument lists (never ``shell=True``); on failure the exact
command is logged. The original audio stream is mapped straight through
(``-map 0:a``) unless the user explicitly enables loudness normalisation,
silence trimming or background music.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import config
from pipeline import crop as crop_mod
from pipeline import overlay

log = config.log.getChild("render")

W, H, FPS = config.OUT_W, config.OUT_H, 30


class RenderError(RuntimeError):
    """FFmpeg/ffprobe failure."""


@dataclass
class RenderOptions:
    """Everything that influences a single clip render."""

    layout: str = "blur"                 # 'blur' | 'smart'
    overlay_mode: str = "hook"           # 'hook' | 'captions' | 'both'
    hook_text: str = ""
    hook_style: str = "bold_outline"
    caption_style: str = "white_yellow"
    hook_duration: float = config.HOOK_DURATION_S
    loudnorm: bool = False
    zoom: bool = False
    silence_trim: bool = False
    progress_bar: bool = False
    music_path: str | None = None
    music_db: float = -22.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderOptions":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "extra"}
        return cls(**known)


# --------------------------------------------------------------------------
# ffprobe / ffmpeg helpers
# --------------------------------------------------------------------------
def probe(path: str | Path) -> dict[str, Any]:
    """Return ``{width,height,duration,fps,has_audio,audio_codec}`` for a media file."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("ffprobe failed: %s\n%s", shlex.join(cmd), r.stderr[-2000:])
        raise RenderError(f"ffprobe failed for {path}: {r.stderr.strip()[-300:]}")
    data = json.loads(r.stdout)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    if v is None:
        raise RenderError(f"No video stream in {path}")
    num, _, den = (v.get("avg_frame_rate") or "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return {
        "width": int(v["width"]), "height": int(v["height"]), "fps": fps,
        "duration": float(data["format"].get("duration") or 0),
        "has_audio": a is not None, "audio_codec": a.get("codec_name") if a else None,
        "audio_duration": float(a.get("duration") or 0) if a else 0.0,
    }


def run_ffmpeg(cmd: list[str], on_progress: Callable[[float], None] | None = None,
               total: float = 0.0) -> None:
    """Run FFmpeg, optionally reporting progress (0..1); log the exact command on failure."""
    full = list(cmd)
    if on_progress and total > 0:
        full = [cmd[0], "-progress", "pipe:1", "-nostats", *cmd[1:]]
    proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        if on_progress and total > 0 and proc.stdout:
            for line in proc.stdout:
                if line.startswith("out_time_ms="):
                    try:
                        on_progress(min(1.0, int(line.split("=")[1]) / 1e6 / total))
                    except ValueError:
                        pass
        err = proc.stderr.read() if proc.stderr else ""
        rc = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if rc != 0:
        log.error("FFmpeg failed (exit %s). Command:\n%s\nstderr tail:\n%s", rc, shlex.join(full), err[-3000:])
        raise RenderError(f"FFmpeg failed (exit {rc}): {err.strip().splitlines()[-1] if err.strip() else 'unknown error'}")


def _esc_filter_path(p: str | Path) -> str:
    """Escape a path for use inside an FFmpeg filter option (handles Windows drive colons)."""
    s = str(Path(p).resolve()).replace("\\", "/")
    return s.replace(":", "\\:").replace("'", "\\'")


# --------------------------------------------------------------------------
# silence trimming
# --------------------------------------------------------------------------
def silence_plan(words: list[dict[str, Any]], clip_len: float, min_gap: float = 0.5,
                 keep_gap: float = 0.2) -> tuple[list[tuple[float, float]], Callable[[float], float], float]:
    """Find gaps > ``min_gap`` between words and shorten them to ``keep_gap``.

    Returns ``(removed_intervals, time_map, new_length)`` where ``time_map`` converts a
    clip-relative time to the trimmed timeline.
    """
    removed: list[tuple[float, float]] = []
    gaps: list[tuple[float, float]] = []
    prev_end = 0.0
    for w in sorted(words, key=lambda x: x["start"]):
        gaps.append((prev_end, min(w["start"], clip_len)))
        prev_end = max(prev_end, w["end"])
    gaps.append((min(prev_end, clip_len), clip_len))
    for a, b in gaps:
        if b - a > min_gap:
            cut = ((a + keep_gap / 2), (b - keep_gap / 2))
            if cut[1] - cut[0] > 0.05:
                removed.append(cut)

    def tmap(t: float) -> float:
        return t - sum(min(max(t - a, 0.0), b - a) for a, b in removed)

    return removed, tmap, clip_len - sum(b - a for a, b in removed)


# --------------------------------------------------------------------------
# filter graph
# --------------------------------------------------------------------------
def build_filter_graph(opts: RenderOptions, plan: crop_mod.CropPlan | None, out_len: float,
                       removed: list[tuple[float, float]], ass_path: Path | None,
                       has_audio: bool, music_idx: int | None) -> tuple[str, str, str | None]:
    """Return ``(filter_complex, video_label, audio_label_or_None)``."""
    v: list[str] = [f"[0:v]fps={FPS},setsar=1"]
    g: list[str] = []

    # --- layout -------------------------------------------------------------
    if opts.layout == "smart" and plan is not None and plan.mode != "fill":
        v.append(f"crop=w={plan.crop_w}:h={plan.crop_h}:x='{plan.x_expr()}':y=0,"
                 f"scale={W}:{H}:flags=lanczos,setsar=1[lay]")
        g.append(",".join(v))
    elif opts.layout == "smart":  # already portrait/narrow: fill the frame
        v.append(f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[lay]")
        g.append(",".join(v))
    else:  # blur layout: fitted video centred over a blurred, scaled copy
        v.append("split=2[bgsrc][fgsrc]")
        g.append(",".join(v))
        g.append(f"[bgsrc]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
                 f"boxblur=luma_radius=12:luma_power=2,scale={W}:{H}:flags=bicubic,setsar=1[bg]")
        g.append(f"[fgsrc]scale={W}:-2:flags=lanczos[fg]")
        g.append("[bg][fg]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1[lay]")
    cur = "lay"

    # --- silence trimming (video) -------------------------------------------
    if removed:
        cond = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in removed)
        g.append(f"[{cur}]select='not({cond})',setpts=N/({FPS}*TB)[trim]")
        cur = "trim"

    # --- slow zoom 1.0 -> 1.06 ------------------------------------------------
    if opts.zoom:
        d = max(out_len, 0.1)
        g.append(f"[{cur}]scale=w='trunc({W}*(1+0.06*t/{d:.3f})/2)*2':h='trunc({H}*(1+0.06*t/{d:.3f})/2)*2'"
                 f":eval=frame:flags=bicubic,crop={W}:{H}[zoom]")
        cur = "zoom"

    # --- progress bar -----------------------------------------------------------
    if opts.progress_bar:
        d = max(out_len, 0.1)
        g.append(f"color=c=white:s={W}x10:r={FPS}[barsrc]")
        g.append(f"[{cur}][barsrc]overlay=x='-w+w*t/{d:.3f}':y={H - 14}:shortest=1[bar]")
        cur = "bar"

    # --- subtitles (hook / captions) ---------------------------------------------
    if ass_path is not None:
        g.append(f"[{cur}]ass=filename='{_esc_filter_path(ass_path)}':fontsdir='{_esc_filter_path(config.FONTS_DIR)}'[sub]")
        cur = "sub"
    g.append(f"[{cur}]format=yuv420p[vout]")

    # --- audio ----------------------------------------------------------------------
    a_label: str | None = None
    if has_audio:
        need_chain = bool(removed) or opts.loudnorm or music_idx is not None
        if need_chain:
            chain = ["[0:a]aresample=48000"]
            if removed:
                cond = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in removed)
                chain.append(f"aselect='not({cond})',asetpts=N/SR/TB")
            if opts.loudnorm:
                chain.append("loudnorm=I=-16:TP=-1.5:LRA=11")
            if music_idx is not None:
                g.append(",".join(chain) + "[a0]")
                g.append(f"[{music_idx}:a]aresample=48000,volume={opts.music_db}dB[mus]")
                g.append("[a0][mus]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]")
            else:
                g.append(",".join(chain) + "[aout]")
            a_label = "[aout]"
    return ";".join(g), "[vout]", a_label


def encoder_args() -> list[str]:
    """H.264 encoder args: NVENC when available, else libx264 (crf 20, preset medium)."""
    if config.has_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
    return ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]


def render_clip(src: str | Path, out: str | Path, ss: float, dur: float,
                words: list[dict[str, Any]], opts: RenderOptions,
                on_progress: Callable[[float], None] | None = None,
                crop_plan: crop_mod.CropPlan | None = None) -> Path:
    """Render ``src[ss:ss+dur]`` to ``out`` as a 1080x1920 H.264/AAC MP4.

    ``words`` are clip-relative word timings (0 = first frame of the clip) used for
    captions and silence detection.
    """
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    dur = min(dur, max(0.1, info["duration"] - ss)) if info["duration"] else dur

    removed: list[tuple[float, float]] = []
    tmap: Callable[[float], float] = lambda t: t  # noqa: E731
    out_len = dur
    if opts.silence_trim and words:
        removed, tmap, out_len = silence_plan(words, dur)
        log.info("Silence trim: removing %d gaps, %.1fs -> %.1fs", len(removed), dur, out_len)

    plan = crop_plan
    if opts.layout == "smart" and plan is None:
        plan = crop_mod.plan_crop(str(src), ss, dur)

    ass_path: Path | None = None
    if opts.overlay_mode in overlay.OVERLAY_MODES:
        subs = overlay.build_ass(opts.overlay_mode, words, dur, hook_text=opts.hook_text,
                                 hook_style=opts.hook_style, caption_style=opts.caption_style,
                                 hook_duration=opts.hook_duration, time_map=tmap)
        if len(subs):
            ass_path = overlay.write_ass(subs, out.with_suffix(".ass"))

    music_idx = None
    inputs = ["-ss", f"{ss:.3f}", "-t", f"{dur:.3f}", "-i", str(src)]
    if opts.music_path and info["has_audio"]:
        mp = Path(opts.music_path)
        if mp.exists():
            inputs += ["-stream_loop", "-1", "-i", str(mp)]
            music_idx = 1
        else:
            log.warning("Music file not found: %s (skipping)", mp)

    graph, vlabel, alabel = build_filter_graph(opts, plan, out_len, removed, ass_path,
                                               info["has_audio"], music_idx)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", graph, "-map", vlabel]
    if info["has_audio"]:
        cmd += ["-map", alabel] if alabel else ["-map", "0:a"]
    cmd += [*encoder_args(), "-pix_fmt", "yuv420p", "-r", str(FPS)]
    if info["has_audio"]:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-t", f"{out_len:.3f}", "-movflags", "+faststart", str(out)]
    run_ffmpeg(cmd, on_progress, out_len)
    return out


def verify_audio(src: str | Path, out: str | Path, ss: float = 0.0, dur: float | None = None) -> dict[str, Any]:
    """Compare the audio of a rendered clip against its source segment.

    Returns ``{has_audio, duration_delta, correlation}``; correlation ~1.0 means the
    original audio plays through untouched (modulo AAC re-encoding).
    """
    import numpy as np

    pi, po = probe(src), probe(out)
    res: dict[str, Any] = {"has_audio": po["has_audio"], "src_has_audio": pi["has_audio"]}
    if not (po["has_audio"] and pi["has_audio"]):
        return res
    length = dur or po["duration"]

    def pcm(path: str | Path, start: float) -> "np.ndarray":
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path),
               "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
        r = subprocess.run(cmd, capture_output=True)
        return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32)

    a, b = pcm(src, ss), pcm(out, 0.0)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best = 0.0
    for lag in range(-1600, 1601, 40):  # +-100 ms for AAC priming delay
        x, y = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        m = min(len(x), len(y))
        if m > 1000 and x[:m].std() > 0 and y[:m].std() > 0:
            best = max(best, float(np.corrcoef(x[:m], y[:m])[0, 1]))
    res.update(duration_delta=abs(po["audio_duration"] - po["duration"]), correlation=best)
    return res
