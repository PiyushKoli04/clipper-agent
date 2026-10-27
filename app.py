"""ShortsAgent - Flask app (local use only).

Run:  python app.py   ->  http://127.0.0.1:5000
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import config
import db
import worker
from pipeline import download, overlay, runner

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # music uploads
MUSIC_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


# --------------------------------------------------------------------------
# pages & static-ish assets
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/fonts/<path:name>")
def fonts(name: str):
    """Serve the bundled font so the live hook preview matches the render."""
    return send_from_directory(config.FONTS_DIR, name)


@app.get("/clips/<path:name>")
def clip_file(name: str):
    """Serve a rendered clip; ``?download=1`` forces a download with a friendly filename."""
    as_attachment = request.args.get("download") == "1"
    dl_name = None
    if as_attachment:
        m = re.search(r"_c(\d+)\.mp4$", name)
        clip = db.get_clip(int(m.group(1))) if m else None
        base = re.sub(r"[^\w\- ]+", "", (clip or {}).get("title") or "short").strip().replace(" ", "_")[:60]
        dl_name = f"{base or 'short'}.mp4"
    return send_from_directory(config.CLIPS_DIR, name, as_attachment=as_attachment,
                               download_name=dl_name, conditional=True, max_age=0)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/options")
def options():
    music = sorted(p.name for p in config.MUSIC_DIR.glob("*") if p.suffix.lower() in MUSIC_EXT)
    cell = overlay.TextMeasurer(overlay.ensure_font())._cell  # libass sizes fonts by ascent+descent
    return jsonify({
        "em_per_fs": (1000.0 / cell) if cell else 0.83,
        "hook_styles": {k: v["label"] for k, v in overlay.HOOK_STYLES.items()},
        "caption_styles": {k: v["label"] for k, v in overlay.CAPTION_STYLES.items()},
        "defaults": {**runner.DEFAULT_PARAMS, "music": None},
        "music": music, "font_family": config.FONT_FAMILY, "font_file": config.FONT_FILE,
        "providers": ["gemini", "groq", "nemotron", "ollama"],
    })


@app.post("/api/music")
def upload_music():
    f = request.files.get("file")
    if not f or not f.filename:
        return err("No file uploaded")
    name = secure_filename(f.filename)
    if Path(name).suffix.lower() not in MUSIC_EXT:
        return err("Unsupported audio type")
    f.save(config.MUSIC_DIR / name)
    return jsonify({"name": name})


@app.get("/api/videos/check")
def check_video():
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"exists": False})
    try:
        vid, _ = download.parse_youtube_url(url)
    except ValueError:
        return jsonify({"exists": False})
    v = db.get_video(vid)
    if not v:
        return jsonify({"exists": False, "video_id": vid})
    clips = [_clip_json(c) for c in db.list_clips(vid)]
    done_count = sum(1 for c in clips if c["status"] == "done")
    return jsonify({
        "exists": True,
        "video_id": vid,
        "title": v.get("title") or vid,
        "clip_count": len(clips),
        "done_count": done_count,
        "clips": clips,
    })


@app.post("/api/jobs")
def create_job():
    data = request.get_json(silent=True) or {}
    try:
        vid, url = download.parse_youtube_url(str(data.get("url", "")))
    except ValueError as e:
        return err(str(e))
    allowed = ("num_clips", "layout", "overlay_mode", "caption_style", "hook_style", "whisper_model",
               "llm_provider", "hook_duration", "loudnorm", "zoom", "silence_trim", "progress_bar", "music", "force_new")
    params = {k: data[k] for k in allowed if k in data}
    params["url"] = url
    try:
        job_id = worker.submit_job(params)
    except (ValueError, TypeError) as e:
        return err(str(e))
    return jsonify({"job_id": job_id, "video_id": vid}), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    j = db.get_job(job_id)
    if not j:
        return err("Job not found", 404)
    return jsonify({k: j[k] for k in ("id", "video_id", "status", "stage", "progress", "error")})


@app.post("/api/jobs/<job_id>/retry")
def retry_job(job_id: str):
    """Re-submit a failed job. The pipeline will resume from cached stages."""
    j = db.get_job(job_id)
    if not j:
        return err("Job not found", 404)
    if j["status"] not in ("failed",):
        return err("Only failed jobs can be retried", 409)
    try:
        params = json.loads(j["params_json"] or "{}")
    except (json.JSONDecodeError, TypeError):
        return err("Could not parse saved job params")
    if not params:
        return err("No saved params for this job")
    try:
        new_job_id = worker.submit_job(params)
    except (ValueError, TypeError) as e:
        return err(str(e))
    return jsonify({"job_id": new_job_id, "video_id": j.get("video_id")}), 202


def _clip_json(c: dict[str, Any]) -> dict[str, Any]:
    out_url = None
    if c["output_path"] and Path(c["output_path"]).exists():
        p = Path(c["output_path"])
        out_url = f"/clips/{p.name}?v={int(p.stat().st_mtime)}"
    return {
        "id": c["id"], "video_id": c["video_id"], "start": c["start"], "end": c["end"], "score": c["score"],
        "title": c["title"], "reason": c["reason"], "hook_text": c["hook_text"],
        "hooks": json.loads(c["hook_alternatives_json"] or "[]"), "overlay_mode": c["overlay_mode"],
        "style": json.loads(c["style"] or "{}"), "status": c["status"], "error": c["error"],
        "url": out_url, "download_url": (out_url.split("?")[0] + "?download=1") if out_url else None,
    }


@app.get("/api/history")
def history():
    videos = db.list_all_videos()
    return jsonify({
        "videos": [
            {
                "id": v["id"],
                "url": v["url"],
                "title": v["title"] or v["id"],
                "duration": v["duration"],
                "created_at": v.get("created_at"),
                "clip_count": v.get("clip_count", 0),
                "done_count": v.get("done_count", 0),
            }
            for v in videos
        ]
    })


@app.get("/api/videos/<video_id>/clips")
def list_clips(video_id: str):
    v = db.get_video(video_id)
    if not v:
        return err("Video not found", 404)
    return jsonify({"video": {"id": v["id"], "title": v["title"], "duration": v["duration"]},
                    "clips": [_clip_json(c) for c in db.list_clips(video_id)]})


@app.patch("/api/clips/<int:clip_id>")
def patch_clip(clip_id: int):
    c = db.get_clip(clip_id)
    if not c:
        return err("Clip not found", 404)
    data = request.get_json(silent=True) or {}
    video = db.get_video(c["video_id"]) or {}
    upd: dict[str, Any] = {}
    try:
        start = float(data.get("start", c["start"]))
        end = float(data.get("end", c["end"]))
    except (TypeError, ValueError):
        return err("start/end must be numbers")
    if "start" in data or "end" in data:
        if start < 0 or end <= start:
            return err("end must be greater than start (and start >= 0)")
        if not 3 <= end - start <= 120:
            return err("Clip length must be between 3 and 120 seconds")
        if video.get("duration") and end > video["duration"] + 0.5:
            return err(f"end is past the end of the video ({video['duration']:.0f}s)")
        upd.update(start=start, end=end)
    if "hook_text" in data:
        t = " ".join(str(data["hook_text"]).split())
        if len(t) > 90:
            return err("hook_text is limited to 90 characters")
        upd["hook_text"] = t
    if "overlay_mode" in data:
        if data["overlay_mode"] not in overlay.OVERLAY_MODES:
            return err("overlay_mode must be hook, captions or both")
        upd["overlay_mode"] = data["overlay_mode"]
    if "style" in data:
        if not isinstance(data["style"], dict):
            return err("style must be an object")
        merged = {**json.loads(c["style"] or "{}"), **{k: v for k, v in data["style"].items() if k in runner.STYLE_KEYS}}
        try:  # validate through the same code as job creation
            runner.normalize_params({**runner.DEFAULT_PARAMS, **merged, "overlay_mode": upd.get("overlay_mode", c["overlay_mode"])})
        except ValueError as e:
            return err(str(e))
        upd["style"] = merged
    if upd:
        db.update_clip(clip_id, **upd)
    return jsonify(_clip_json(db.get_clip(clip_id)))


@app.post("/api/clips/<int:clip_id>/render")
def render_clip(clip_id: int):
    c = db.get_clip(clip_id)
    if not c:
        return err("Clip not found", 404)
    if c["status"] == "rendering":
        return err("Already rendering", 409)
    worker.submit_render(clip_id)
    return jsonify(_clip_json(db.get_clip(clip_id))), 202


def main() -> None:
    config.setup_logging()
    if not config.check_ffmpeg():
        sys.exit(1)
    config.ensure_dirs()
    db.init_db()
    db.fail_stale_jobs()
    overlay.ensure_font()
    print(f"GPU: {'CUDA available (Whisper float16, NVENC if usable)' if config.has_cuda() else 'none - using CPU (int8) + libx264'}")
    print("ShortsAgent running at http://127.0.0.1:5000  (local use only)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
