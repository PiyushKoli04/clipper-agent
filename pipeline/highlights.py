"""Stage 3: pick the most engaging 20-50 s moments with an LLM.

Flow: transcript -> overlapping 4-minute windows -> LLM (strict JSON, retried) ->
validation -> snap to sentence boundaries via word timestamps (+0.2 s padding) ->
dedupe overlaps -> rank by score -> top N.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import config
from llm import LLMError, LLMProvider
from pipeline.transcribe import flatten_words

log = config.log.getChild("highlights")

FALLBACK_HOOKS = ["Wait for the ending", "Nobody expected this", "Watch till the end",
                  "You need to see this", "This changes everything"]

SYSTEM_PROMPT = (
    "You are an expert short-form video editor who finds viral moments in long videos (including English, Hindi, and Hinglish podcasts). "
    "You ALWAYS answer with a single valid JSON object and nothing else - no markdown, no commentary."
)

PROMPT_TEMPLATE = """Below is a transcript excerpt of a YouTube video (English, Hindi, or Hinglish). Each line starts with its timestamp in SECONDS: [123.4] text.
The excerpt covers {a:.0f}s to {b:.0f}s of the video.

Find up to {k} moments that would work as standalone YouTube Shorts.

Rules for every clip:
- Length between {lo:.0f} and {hi:.0f} seconds (end - start).
- Self-contained: makes sense without the rest of the video.
- Strong first 3 seconds (a hook), and a payoff or clean ending.
- Contains emotion, a story, a surprising claim, a strong opinion or a concrete number.
- start and end are in SECONDS (numbers) and must come from the timestamps above.
- score is an integer 1-10 for how likely the clip is to hold viewers.
- hooks: exactly 3 DIFFERENT on-screen hook lines, each 3-8 words, curiosity-driven, that TEASE the clip WITHOUT spoiling it (in English or Hinglish/Hindi as appropriate).
  Good examples: "See what happens next...", "Nobody expected this", "Yeh kya bol diya?", "Wait for the ending".
- hook_text is your best hook (must equal hooks[0]).
- title: a short catchy title (max 8 words). reason: one sentence on why it works.

Return ONLY this JSON shape:
{{"clips": [{{"start": 12.5, "end": 47.0, "score": 8, "hook_text": "...", "hooks": ["...", "...", "..."], "title": "...", "reason": "..."}}]}}
If nothing qualifies return {{"clips": []}}.

TRANSCRIPT:
{text}
"""


@dataclass
class Highlight:
    """One selected clip."""

    start: float
    end: float
    score: float
    title: str
    hook_text: str
    hooks: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# windows & prompt
# --------------------------------------------------------------------------
def make_windows(duration: float, window: float = config.WINDOW_S,
                 overlap: float = config.WINDOW_OVERLAP_S) -> list[tuple[float, float]]:
    """Overlapping ``window``-second windows covering ``[0, duration]``."""
    if duration <= window:
        return [(0.0, duration)]
    step = window - overlap
    out, a = [], 0.0
    while a < duration - overlap:
        out.append((a, min(duration, a + window)))
        a += step
    return out


def window_text(segments: list[dict[str, Any]], a: float, b: float) -> str:
    """Transcript lines ``[sec] text`` for segments inside ``[a, b]``."""
    return "\n".join(f"[{s['start']:.1f}] {s['text']}" for s in segments
                     if s["end"] > a and s["start"] < b and s["text"].strip())


# --------------------------------------------------------------------------
# JSON handling
# --------------------------------------------------------------------------
_FENCE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$", re.M)


def parse_llm_json(raw: str) -> list[dict[str, Any]]:
    """Parse an LLM reply into a list of clip dicts.

    Strips markdown fences, extracts the outermost JSON value, repairs trailing commas,
    and accepts either ``{"clips": [...]}``, a bare list, or a single clip object.
    Raises ``ValueError`` if nothing parseable is found.
    """
    if not raw or not raw.strip():
        raise ValueError("empty reply")
    text = _FENCE.sub("", raw.strip()).strip()
    candidates = [text]
    m = re.search(r"[\[{]", text)
    if m:
        end = max(text.rfind("}"), text.rfind("]"))
        if end > m.start():
            candidates.append(text[m.start(): end + 1])
    last_err: Exception | None = None
    for cand in candidates:
        for fixed in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                data = json.loads(fixed)
            except json.JSONDecodeError as e:
                last_err = e
                continue
            if isinstance(data, dict):
                for key in ("clips", "highlights", "moments", "results"):
                    if isinstance(data.get(key), list):
                        return [d for d in data[key] if isinstance(d, dict)]
                return [data] if "start" in data and "end" in data else []
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
    raise ValueError(f"invalid JSON: {last_err}")


def ask_json(provider: LLMProvider, prompt: str, retries: int = 3) -> list[dict[str, Any]]:
    """Call the LLM and parse strict JSON, retrying with a firmer instruction on bad output."""
    nudge = ""
    err: Exception | None = None
    for attempt in range(retries):
        raw = provider.complete(prompt + nudge, system=SYSTEM_PROMPT, temperature=max(0.1, 0.4 - 0.15 * attempt))
        try:
            return parse_llm_json(raw)
        except ValueError as e:
            err = e
            log.warning("LLM returned invalid JSON (attempt %d/%d): %s", attempt + 1, retries, e)
            nudge = "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object described above."
    raise LLMError(f"LLM did not return valid JSON after {retries} attempts: {err}")


# --------------------------------------------------------------------------
# candidate cleanup
# --------------------------------------------------------------------------
def _to_seconds(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"\d+(\.\d+)?", s):
            return float(s)
        m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", s)
        if m:
            h, mi, sec = m.groups()
            return int(h or 0) * 3600 + int(mi) * 60 + float(sec)
    return None


def clean_hook(h: Any, max_words: int = 8) -> str | None:
    """Normalise one hook line: strip quotes/emoji-ish noise, enforce 2..8 words."""
    if not isinstance(h, str):
        return None
    h = re.sub(r"^[\s\"'“”‘’\-•*\d.)]+|[\s\"'“”‘’]+$", "", h)
    h = " ".join(h.split())
    words = h.split()
    if len(words) < 2:
        return None
    if len(words) > max_words:
        h = " ".join(words[:max_words]).rstrip(",;:-") + "..."
    return h


def normalize_hooks(hook_text: Any, hooks: Any) -> tuple[str, list[str]]:
    """Return ``(best_hook, [3 unique hooks])`` guaranteeing exactly three variants."""
    pool: list[str] = []
    for h in ([hook_text] if hook_text else []) + (hooks if isinstance(hooks, list) else []):
        c = clean_hook(h)
        if c and c.lower() not in {p.lower() for p in pool}:
            pool.append(c)
    for fb in FALLBACK_HOOKS:
        if len(pool) >= 3:
            break
        if fb.lower() not in {p.lower() for p in pool}:
            pool.append(fb)
    pool = pool[:3]
    return pool[0], pool


# --------------------------------------------------------------------------
# sentence snapping
# --------------------------------------------------------------------------
_END_PUNCT = (".", "?", "!", "…", ".\"", "?\"", "!\"")


def sentence_bounds(words: list[dict[str, Any]], gap: float = 0.8) -> tuple[list[int], list[int]]:
    """Indices of words that start / end a sentence (punctuation or a pause > ``gap`` s)."""
    starts, ends = [], []
    for i, w in enumerate(words):
        prev = words[i - 1] if i else None
        nxt = words[i + 1] if i + 1 < len(words) else None
        if prev is None or prev["word"].endswith(_END_PUNCT) or w["start"] - prev["end"] > gap:
            starts.append(i)
        if nxt is None or w["word"].endswith(_END_PUNCT) or nxt["start"] - w["end"] > gap:
            ends.append(i)
    return starts, ends


def snap_clip(words: list[dict[str, Any]], start: float, end: float, video_dur: float,
              lo: float = config.CLIP_MIN_S, hi: float = config.CLIP_MAX_S,
              pad: float = config.SNAP_PAD_S, tol: float = 8.0) -> tuple[float, float] | None:
    """Snap ``[start, end]`` to sentence boundaries, enforce ``lo..hi`` seconds, add padding.

    Returns ``None`` if no valid snapped range exists.
    """
    if not words:
        return None
    s_idx, e_idx = sentence_bounds(words)
    if not s_idx or not e_idx:
        return None
    starts = [(i, words[i]["start"]) for i in s_idx]
    ends = [(i, words[i]["end"]) for i in e_idx]
    si, st = min(starts, key=lambda p: abs(p[1] - start))
    if abs(st - start) > tol:  # no sentence start nearby: use the nearest word start
        si = min(range(len(words)), key=lambda i: abs(words[i]["start"] - start))
        st = words[si]["start"]
    eligible = [p for p in ends if p[1] > st]
    if not eligible:
        return None
    ei, en = min(eligible, key=lambda p: abs(p[1] - end))
    # the padded length must land inside [lo, hi]
    max_len, min_len = hi - 2 * pad, lo
    if en - st > max_len:  # too long: pull the end back to an earlier sentence end
        shorter = [p for p in eligible if p[1] - st <= max_len and p[1] - st >= min_len]
        if shorter:
            ei, en = max(shorter, key=lambda p: p[1])
        else:  # a single run-on sentence: hard-cut at the last word that fits
            fit = [i for i in range(si, len(words)) if words[i]["end"] - st <= max_len]
            if not fit:
                return None
            en = words[fit[-1]]["end"]
    if en - st < min_len:  # too short: extend to a later sentence end
        longer = [p for p in eligible if min_len <= p[1] - st <= max_len]
        if not longer:
            return None
        ei, en = min(longer, key=lambda p: p[1])
    a, b = max(0.0, st - pad), min(video_dur or en + pad, en + pad)
    return (round(a, 3), round(b, 3)) if b - a >= lo - 0.5 else None


def dedupe(items: list[Highlight], max_overlap: float = 0.3) -> list[Highlight]:
    """Greedy by score: drop clips that overlap an already-kept clip by > ``max_overlap`` of the shorter one."""
    kept: list[Highlight] = []
    for h in sorted(items, key=lambda x: (-x.score, x.start)):
        ok = True
        for k in kept:
            inter = min(h.end, k.end) - max(h.start, k.start)
            if inter > 0 and inter / min(h.end - h.start, k.end - k.start) > max_overlap:
                ok = False
                break
        if ok:
            kept.append(h)
    return kept


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def select_highlights(transcript: dict[str, Any], provider: LLMProvider, num_clips: int = 5,
                      video_duration: float | None = None,
                      progress: Callable[[float], None] | None = None,
                      exclude_ranges: list[tuple[float, float]] | None = None) -> list[Highlight]:
    """Return the top ``num_clips`` highlights for ``transcript`` (sorted by score, best first).
    
    If ``exclude_ranges`` is provided (e.g. from previously rendered clips), any
    candidate overlapping those timestamps is discarded, guaranteeing fresh clips.
    """
    words = flatten_words(transcript)
    if not words:
        raise ValueError("Transcript has no words")
    duration = float(video_duration or transcript.get("duration") or words[-1]["end"])
    windows = make_windows(duration)
    per_window = min(6, max(2, math.ceil(num_clips * 1.6 / len(windows)) + 1))
    candidates: list[Highlight] = []
    for wi, (a, b) in enumerate(windows):
        text = window_text(transcript["segments"], a, b)
        if len(text) < 80:
            continue
        log.info("Analyzing transcript window %d/%d (%.0fs - %.0fs)...", wi + 1, len(windows), a, b)
        
        exclude_rule = ""
        if exclude_ranges:
            overlapping_ranges = [r for r in exclude_ranges if not (r[1] < a or r[0] > b)]
            if overlapping_ranges:
                ranges_str = ", ".join(f"[{r[0]:.1f}s - {r[1]:.1f}s]" for r in overlapping_ranges)
                exclude_rule = f"\n- CRITICAL: Do NOT pick clips overlapping with these already-generated moments: {ranges_str}."

        prompt = PROMPT_TEMPLATE.format(a=a, b=b, k=per_window, lo=config.CLIP_MIN_S,
                                        hi=config.CLIP_MAX_S, text=text)
        if exclude_rule:
            prompt = prompt.replace("Rules for every clip:", f"Rules for every clip:{exclude_rule}")

        for d in ask_json(provider, prompt):
            s, e = _to_seconds(d.get("start")), _to_seconds(d.get("end"))
            if s is None or e is None or e <= s or e < a - 2 or s > b + 2:
                continue
            snapped = snap_clip(words, s, e, duration)
            if not snapped:
                log.info("Dropped candidate %.1f-%.1f (cannot snap into %ss-%ss)", s, e,
                         config.CLIP_MIN_S, config.CLIP_MAX_S)
                continue
            
            # Verify candidate doesn't overlap existing clips
            if exclude_ranges:
                overlap_found = False
                for ex_s, ex_e in exclude_ranges:
                    inter = min(snapped[1], ex_e) - max(snapped[0], ex_s)
                    if inter > 0 and (inter / min(snapped[1] - snapped[0], ex_e - ex_s)) > 0.15:
                        overlap_found = True
                        break
                if overlap_found:
                    log.info("Skipped candidate %.1f-%.1f (overlaps an existing clip)", snapped[0], snapped[1])
                    continue

            try:
                score = max(1.0, min(10.0, float(d.get("score", 5))))
            except (TypeError, ValueError):
                score = 5.0
            hook, hooks = normalize_hooks(d.get("hook_text"), d.get("hooks"))
            candidates.append(Highlight(snapped[0], snapped[1], score,
                                        str(d.get("title") or hook)[:90], hook, hooks,
                                        str(d.get("reason") or "")[:300]))
        if progress:
            progress((wi + 1) / len(windows))
    final = dedupe(candidates)[:max(1, num_clips)]
    log.info("Selected %d highlights from %d candidates", len(final), len(candidates))
    return final


# --------------------------------------------------------------------------
# highlight cache (survives crashes / restarts)
# --------------------------------------------------------------------------
_CACHE_DIR = config.STORAGE_DIR / "highlights_cache"


def _cache_path(video_id: str) -> Path:
    return _CACHE_DIR / f"{video_id}.json"


def save_highlights(video_id: str, items: list[Highlight]) -> None:
    """Persist highlights to disk so the LLM stage can be skipped on retry."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = [h.to_dict() for h in items]
    _cache_path(video_id).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    log.info("Cached %d highlights for %s", len(items), video_id)


def load_highlights(video_id: str) -> list[Highlight] | None:
    """Load previously cached highlights, or return ``None`` if the cache misses."""
    p = _cache_path(video_id)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            return None
        out = []
        for d in raw:
            out.append(Highlight(
                start=float(d["start"]), end=float(d["end"]),
                score=float(d.get("score", 5)),
                title=str(d.get("title", "")),
                hook_text=str(d.get("hook_text", "")),
                hooks=d.get("hooks", []),
                reason=str(d.get("reason", "")),
            ))
        log.info("Loaded %d cached highlights for %s", len(out), video_id)
        return out
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        log.warning("Corrupt highlights cache for %s (%s), ignoring", video_id, e)
        return None


def clear_highlights_cache(video_id: str) -> None:
    """Remove the cached highlights for a video (e.g. when user wants a fresh run)."""
    p = _cache_path(video_id)
    if p.exists():
        p.unlink(missing_ok=True)
        log.info("Cleared highlights cache for %s", video_id)

