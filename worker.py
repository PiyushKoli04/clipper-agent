"""Background job execution (ThreadPoolExecutor, 1-2 workers).

Nothing heavy ever runs inside a Flask request: the app only creates DB rows and
submits callables here. Job/clip state is persisted in SQLite so the UI can poll.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import config
import db
from pipeline import runner

log = config.log.getChild("worker")
_executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS, thread_name_prefix="shorts")
_render_lock = threading.Lock()


def _run_job(job_id: str, params: dict[str, Any]) -> None:
    db.update_job(job_id, status="running", stage="Starting", progress=1)
    last = [-1.0]
    last_stage = [""]

    def on_progress(stage: str, pct: float) -> None:
        pct_round = round(pct, 1)
        if stage != last_stage[0] or (pct - last[0]) >= 5.0 or pct >= 100:
            log.info("[%s] %s (%.0f%%)", job_id, stage, pct_round)
            last_stage[0] = stage
        if abs(pct - last[0]) >= 0.5 or pct >= 100:  # throttle DB writes
            last[0] = pct
            db.update_job(job_id, stage=stage, progress=pct_round)

    try:
        runner.run_pipeline(params, on_progress, job_id=job_id)
        db.update_job(job_id, status="done", stage="Done", progress=100)
    except Exception as e:  # any crash -> failed + message, never a silent hang
        log.exception("Job %s failed", job_id)
        db.update_job(job_id, status="failed", error=f"{type(e).__name__}: {e}"[:800])


def submit_job(params: dict[str, Any]) -> str:
    """Validate params, create the job row and queue it. Raises ``ValueError`` on bad params."""
    normalized = runner.normalize_params(params)
    job_id = db.create_job({k: v for k, v in params.items() if k != "music"} | {"music": normalized["music"]})
    _executor.submit(_run_job, job_id, normalized)
    return job_id


def _render_one(clip_id: int) -> None:
    try:
        runner.render_clip_row(clip_id)
    except Exception:
        pass  # render_clip_row already stored status=failed + error


def submit_render(clip_id: int) -> None:
    """Queue a single-clip re-render."""
    db.update_clip(clip_id, status="rendering", error=None)
    _executor.submit(_render_one, clip_id)
