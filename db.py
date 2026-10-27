"""SQLite persistence layer (stdlib ``sqlite3``).

A fresh connection is opened per call so it is safe to use from Flask request
threads and background worker threads alike.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,            -- YouTube video id
    url TEXT NOT NULL,
    title TEXT,
    duration REAL,
    audio_path TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS transcripts (
    video_id TEXT PRIMARY KEY REFERENCES videos(id) ON DELETE CASCADE,
    source TEXT,                    -- whisper | youtube-transcript-api
    model TEXT,
    json_path TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start REAL NOT NULL,
    end REAL NOT NULL,
    score REAL,
    title TEXT,
    reason TEXT,
    hook_text TEXT,
    hook_alternatives_json TEXT,
    overlay_mode TEXT DEFAULT 'hook',
    style TEXT DEFAULT '{}',        -- JSON: layout, hook_style, caption_style, polish options
    status TEXT DEFAULT 'pending',  -- pending | rendering | done | failed
    error TEXT,
    output_path TEXT,
    src_path TEXT,                  -- downloaded section (with margin)
    src_start REAL,
    src_end REAL,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    video_id TEXT,
    status TEXT DEFAULT 'queued',   -- queued | running | done | failed
    stage TEXT DEFAULT 'queued',
    progress REAL DEFAULT 0,
    error TEXT,
    params_json TEXT,
    created_at REAL
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Yield a connection with Row factory; commit on success, rollback on error."""
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they do not exist."""
    config.ensure_dirs()
    with connect() as c:
        c.executescript(SCHEMA)


def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(r) if r is not None else None


# ---- videos / transcripts ------------------------------------------------
def upsert_video(video_id: str, url: str, title: str | None = None,
                 duration: float | None = None, audio_path: str | None = None) -> None:
    with connect() as c:
        c.execute(
            """INSERT INTO videos (id, url, title, duration, audio_path, created_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 url=excluded.url,
                 title=COALESCE(excluded.title, videos.title),
                 duration=COALESCE(excluded.duration, videos.duration),
                 audio_path=COALESCE(excluded.audio_path, videos.audio_path)""",
            (video_id, url, title, duration, audio_path, time.time()),
        )


def get_video(video_id: str) -> dict[str, Any] | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone())


def list_all_videos() -> list[dict[str, Any]]:
    """Return all saved videos with their clips count, newest first."""
    with connect() as c:
        rows = c.execute(
            """SELECT v.*,
                      COUNT(c.id) as clip_count,
                      SUM(CASE WHEN c.status='done' THEN 1 ELSE 0 END) as done_count
               FROM videos v
               LEFT JOIN clips c ON c.video_id = v.id
               GROUP BY v.id
               ORDER BY v.created_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]


def save_transcript_meta(video_id: str, source: str, model: str, json_path: str) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO transcripts (video_id, source, model, json_path, created_at) VALUES (?,?,?,?,?)",
            (video_id, source, model, json_path, time.time()),
        )


def get_transcript_meta(video_id: str) -> dict[str, Any] | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM transcripts WHERE video_id=?", (video_id,)).fetchone())


# ---- clips ---------------------------------------------------------------
def insert_clip(video_id: str, start: float, end: float, score: float, title: str,
                reason: str, hook_text: str, hooks: list[str], overlay_mode: str,
                style: dict[str, Any]) -> int:
    with connect() as c:
        cur = c.execute(
            """INSERT INTO clips (video_id,start,end,score,title,reason,hook_text,
                                  hook_alternatives_json,overlay_mode,style,status,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (video_id, start, end, score, title, reason, hook_text, json.dumps(hooks),
             overlay_mode, json.dumps(style), "pending", time.time()),
        )
        return int(cur.lastrowid)


def delete_clips_for_video(video_id: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM clips WHERE video_id=?", (video_id,))


def get_clip(clip_id: int) -> dict[str, Any] | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone())


def list_clips(video_id: str) -> list[dict[str, Any]]:
    with connect() as c:
        rows = c.execute("SELECT * FROM clips WHERE video_id=? ORDER BY score DESC, id", (video_id,)).fetchall()
        return [dict(r) for r in rows]


def update_clip(clip_id: int, **fields: Any) -> None:
    """Update arbitrary clip columns. ``style`` / hook list dicts are JSON-encoded."""
    if not fields:
        return
    if isinstance(fields.get("style"), dict):
        fields["style"] = json.dumps(fields["style"])
    if isinstance(fields.get("hook_alternatives_json"), list):
        fields["hook_alternatives_json"] = json.dumps(fields["hook_alternatives_json"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE clips SET {cols} WHERE id=?", (*fields.values(), clip_id))


# ---- jobs ----------------------------------------------------------------
def create_job(params: dict[str, Any], video_id: str | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    with connect() as c:
        c.execute(
            "INSERT INTO jobs (id, video_id, status, stage, progress, params_json, created_at) VALUES (?,?,?,?,?,?,?)",
            (job_id, video_id, "queued", "queued", 0, json.dumps(params), time.time()),
        )
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def get_job(job_id: str) -> dict[str, Any] | None:
    with connect() as c:
        return _row(c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())


def fail_stale_jobs() -> None:
    """On startup, mark jobs/clips left 'running' by a previous crash as failed."""
    with connect() as c:
        c.execute("UPDATE jobs SET status='failed', error='Interrupted by app restart' WHERE status IN ('queued','running')")
        c.execute("UPDATE clips SET status='failed', error='Interrupted by app restart' WHERE status='rendering'")
