"""Pipeline orchestration shared by the web worker and the CLI.

``run_pipeline`` does URL -> audio -> transcript -> highlights -> sections -> renders.
``render_clip_row`` (re)renders a single clip from the database.
Each stage lives in its own module and can be tested independently.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import config
import db
from llm import get_provider
from pipeline import download, highlights, render, transcribe

log = config.log.getChild("runner")
ProgressCb = Callable[[str, float], None]  # (stage label, overall percent 0..100)

DEFAULT_PARAMS: dict[str, Any] = {
    "num_clips": 5, "layout": "blur", "overlay_mode": "hook", "caption_style": "white_yellow",
    "hook_style": "bold_outline", "whisper_model": config.WHISPER_MODEL, "llm_provider": config.LLM_PROVIDER,
    "hook_duration": config.HOOK_DURATION_S, "loudnorm": False, "zoom": False, "silence_trim": False,
    "progress_bar": False, "music": None, "force_new": False,
    # CLI/testing extras
    "local_file": None, "start": None, "end": None, "hook_text": None,
}
STYLE_KEYS = ("layout", "hook_style", "caption_style", "hook_duration", "loudnorm", "zoom",
              "silence_trim", "progress_bar", "music")


def normalize_params(p: dict[str, Any]) -> dict[str, Any]:
    """Merge with defaults and validate enum/range values. Raises ``ValueError`` on bad input."""
    from pipeline import overlay

    out = {**DEFAULT_PARAMS, **{k: v for k, v in p.items() if v is not None or k in ("music",)}}
    out["num_clips"] = max(1, min(20, int(out["num_clips"])))
    out["force_new"] = bool(out.get("force_new", False))
    if out["layout"] not in ("blur", "smart"):
        raise ValueError("layout must be 'blur' or 'smart'")
    if out["overlay_mode"] not in overlay.OVERLAY_MODES:
        raise ValueError("overlay_mode must be one of %s" % (overlay.OVERLAY_MODES,))
    if out["hook_style"] not in overlay.HOOK_STYLES:
        raise ValueError("hook_style must be one of %s" % list(overlay.HOOK_STYLES))
    if out["caption_style"] not in overlay.CAPTION_STYLES:
        raise ValueError("caption_style must be one of %s" % list(overlay.CAPTION_STYLES))
    if out["whisper_model"] not in ("tiny", "base", "small", "medium", "large-v3", "large-v2", "distil-large-v3"):
        raise ValueError("unknown whisper_model")
    out["hook_duration"] = max(1.0, min(6.0, float(out["hook_duration"])))
    for k in ("loudnorm", "zoom", "silence_trim", "progress_bar"):
        out[k] = bool(out[k])
    if out.get("music"):
        mp = Path(out["music"])
        if not mp.is_absolute():
            mp = config.MUSIC_DIR / mp.name  # only files inside storage/music by name
        if not mp.exists():
            raise ValueError(f"music file not found: {mp.name}")
        out["music"] = str(mp)
    else:
        out["music"] = None
    return out


def style_from_params(p: dict[str, Any]) -> dict[str, Any]:
    return {k: p[k] for k in STYLE_KEYS if k in p}


def clip_words(transcript: dict[str, Any], start: float, end: float) -> list[dict[str, Any]]:
    """Words inside ``[start, end]`` shifted so 0 = clip start."""
    out = []
    for w in transcribe.flatten_words(transcript):
        if w["end"] > start and w["start"] < end:
            out.append({"word": w["word"], "start": max(0.0, w["start"] - start),
                        "end": min(end - start, w["end"] - start)})
    return out


def _noop(stage: str, pct: float) -> None:  # pragma: no cover
    pass


def _prepare_source(p: dict[str, Any]) -> tuple[str, str, str, float, str]:
    """Return ``(video_id, url, title, duration, audio_path)`` for a URL or a local file."""
    if p.get("local_file"):
        f = Path(p["local_file"]).expanduser().resolve()
        if not f.exists():
            raise ValueError(f"local file not found: {f}")
        vid = "local-" + hashlib.sha1(str(f).encode()).hexdigest()[:8]
        info = render.probe(f)
        return vid, f.as_uri(), f.stem, info["duration"], str(f)
    vid, url = download.parse_youtube_url(p["url"])
    meta = download.download_audio(url, vid)
    return vid, url, meta["title"], meta["duration"], meta["audio_path"]


def run_pipeline(params: dict[str, Any], on_progress: ProgressCb = _noop,
                 job_id: str | None = None) -> str:
    """Run the whole pipeline; returns the video id. Clips end up rendered in ``storage/clips``.

    The pipeline is **resumable**: transcripts, highlights, and rendered clips are
    cached on disk / in the DB.  If a previous run crashed (e.g. LLM 503), a retry
    for the same video will skip the stages that already completed.
    """
    p = normalize_params(params)
    config.ensure_dirs()
    db.init_db()

    on_progress("Downloading audio", 2)
    vid, url, title, duration, audio = _prepare_source(p)
    db.upsert_video(vid, url, title, duration, audio)
    if job_id:
        db.update_job(job_id, video_id=vid)

    # --- transcript ---------------------------------------------------------
    need_transcript = p["start"] is None or p["overlay_mode"] != "hook" or p["silence_trim"]
    transcript: dict[str, Any] = {"segments": [], "duration": duration}
    if need_transcript:
        on_progress("Transcribing", 8)
        transcript = transcribe.transcribe(
            vid, audio, p["whisper_model"], duration_hint=duration,
            progress=lambda f: on_progress("Transcribing", 8 + 27 * f))
        db.save_transcript_meta(vid, transcript.get("source", "whisper"), transcript.get("model", ""),
                                str(transcribe.transcript_path(vid)))

    force_new = bool(p.get("force_new", False))
    existing_clips = db.list_clips(vid)
    exclude_ranges = [(c["start"], c["end"]) for c in existing_clips if c.get("status") == "done"]

    # --- highlights -----------------------------------------------------------
    if p["start"] is not None and p["end"] is not None:  # manual single clip (CLI / testing)
        hook = p["hook_text"] or "Wait for the ending"
        hl = [highlights.Highlight(float(p["start"]), float(p["end"]), 10.0, hook, hook,
                                   highlights.normalize_hooks(hook, [])[1], "manual range")]
    else:
        # If force_new is requested, bypass cache and find fresh highlights
        cached_hl = None if force_new else highlights.load_highlights(vid)
        if cached_hl:
            on_progress("Restoring highlights (cached)", 50)
            log.info("Restored %d highlights from cache for %s — skipping LLM", len(cached_hl), vid)
            hl = cached_hl
        else:
            on_progress("Finding highlights", 36)
            provider = get_provider(p["llm_provider"])
            hl = highlights.select_highlights(
                transcript, provider, p["num_clips"], duration,
                progress=lambda f: on_progress("Finding highlights", 36 + 14 * f),
                exclude_ranges=exclude_ranges if (force_new and exclude_ranges) else None,
            )
            if not hl:
                raise RuntimeError("The LLM found no usable highlights. Try a different model/provider or a longer video.")
            # Cache highlights
            highlights.save_highlights(vid, hl)

    # --- insert clips (skip if already present from a cached run) ---------------
    if existing_clips and not force_new:
        # Check if the existing clips match the highlights (same run)
        existing_done = [c for c in existing_clips if c["status"] == "done"
                         and c.get("output_path") and Path(c["output_path"]).exists()]
        existing_pending = [c for c in existing_clips if c["status"] != "done"
                            or not c.get("output_path") or not Path(c["output_path"]).exists()]
        if existing_done and not existing_pending:
            log.info("All %d clips already rendered for %s — nothing to do", len(existing_done), vid)
            on_progress("Done", 100)
            return vid
        if existing_done:
            log.info("Resuming: %d clips done, %d still need rendering for %s",
                     len(existing_done), len(existing_pending), vid)
            # Only re-render the ones that aren't done
            ids = [c["id"] for c in existing_pending]
        else:
            # Clips exist but none are done — re-create them
            db.delete_clips_for_video(vid)
            ids = []
            for h in hl:
                ids.append(db.insert_clip(vid, h.start, h.end, h.score, h.title, h.reason, h.hook_text,
                                          h.hooks, p["overlay_mode"], style_from_params(p)))
    else:
        # Brand new video or force_new: append fresh clips without overwriting old ones
        ids = []
        for h in hl:
            ids.append(db.insert_clip(vid, h.start, h.end, h.score, h.title, h.reason, h.hook_text,
                                      h.hooks, p["overlay_mode"], style_from_params(p)))

    # --- sections + render ---------------------------------------------------------
    for n, cid in enumerate(ids):
        lo = 50 + 50 * n / len(ids)
        span = 50 / len(ids)
        render_clip_row(cid, lambda stage, f, lo=lo, span=span, n=n: on_progress(
            f"{stage} (clip {n + 1}/{len(ids)})", lo + span * f), local_file=p.get("local_file"))
    on_progress("Done", 100)
    return vid


def render_clip_row(clip_id: int, on_progress: Callable[[str, float], None] | None = None,
                    local_file: str | None = None) -> Path:
    """(Re)render one clip from its DB row. Downloads the source section if needed.

    ``on_progress(stage, fraction 0..1)``. Updates ``clips.status`` / ``output_path`` / ``error``.
    """
    cb = on_progress or (lambda s, f: None)
    clip = db.get_clip(clip_id)
    if clip is None:
        raise KeyError(f"clip {clip_id} not found")
    video = db.get_video(clip["video_id"])
    db.update_clip(clip_id, status="rendering", error=None)
    try:
        style = json.loads(clip["style"] or "{}")
        start, end = float(clip["start"]), float(clip["end"])
        if end - start < 3:
            raise ValueError("Clip must be at least 3 seconds long")
        vdur = float(video["duration"] or 0)
        if vdur and end > vdur + 0.01:
            end = vdur

        # 1) source section ---------------------------------------------------------
        src, src_start, src_end = clip["src_path"], clip["src_start"], clip["src_end"]
        is_local = video["url"].startswith("file://")
        covered = (src and Path(src).exists() and src_start is not None
                   and src_start <= start + 1e-3 and src_end >= end - 1e-3)
        if not covered:
            if is_local:
                src, src_start, src_end = str(Path(video["audio_path"])), 0.0, vdur
            else:
                cb("Downloading section", 0.0)
                a = max(0.0, start - config.SECTION_MARGIN_S)
                b = min(vdur or end + config.SECTION_MARGIN_S, end + config.SECTION_MARGIN_S)
                src, src_start, src_end = str(download.download_section(video["url"], video["id"], a, b)), a, b
            db.update_clip(clip_id, src_path=src, src_start=src_start, src_end=src_end)

        # 2) words (captions / silence trim) ---------------------------------------------
        mode = clip["overlay_mode"]
        words: list[dict[str, Any]] = []
        if mode in ("captions", "both") or style.get("silence_trim"):
            tr = transcribe.load_cached(video["id"])
            if tr is None:
                raise RuntimeError("No transcript cached for this video; re-run the job to enable captions")
            words = clip_words(tr, start, end)

        # 3) render --------------------------------------------------------------------------
        opts = render.RenderOptions(
            layout=style.get("layout", "blur"), overlay_mode=mode, hook_text=clip["hook_text"] or "",
            hook_style=style.get("hook_style", "bold_outline"), caption_style=style.get("caption_style", "white_yellow"),
            hook_duration=float(style.get("hook_duration", config.HOOK_DURATION_S)),
            loudnorm=bool(style.get("loudnorm")), zoom=bool(style.get("zoom")),
            silence_trim=bool(style.get("silence_trim")), progress_bar=bool(style.get("progress_bar")),
            music_path=style.get("music") or None)
        out = config.CLIPS_DIR / f"{video['id']}_c{clip_id}.mp4"
        cb("Rendering", 0.0)
        render.render_clip(src, out, start - float(src_start), end - start, words, opts,
                           on_progress=lambda f: cb("Rendering", f))
        db.update_clip(clip_id, status="done", output_path=str(out), error=None)
        return out
    except Exception as e:
        log.exception("Render of clip %s failed", clip_id)
        db.update_clip(clip_id, status="failed", error=str(e)[:500])
        raise
