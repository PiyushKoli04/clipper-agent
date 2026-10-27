"""Stage 2: speech-to-text with faster-whisper (word timestamps + VAD), cached as JSON.

Transcript format (JSON, also the in-memory dict)::

    {"source": "whisper|youtube-transcript-api", "language": "en", "duration": 123.4,
     "segments": [{"start": 0.0, "end": 3.2, "text": "...",
                   "words": [{"word": "Hello", "start": 0.0, "end": 0.4}, ...]}]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import config

log = config.log.getChild("transcribe")
ProgressCb = Callable[[float], None]


class TranscribeError(RuntimeError):
    """Raised when both Whisper and the transcript-API fallback fail."""


def transcript_path(video_id: str) -> Path:
    return config.TRANSCRIPTS_DIR / f"{video_id}.json"


def load_cached(video_id: str) -> dict[str, Any] | None:
    p = transcript_path(video_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Corrupt transcript cache %s, ignoring", p)
    return None


def flatten_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one flat, time-ordered list of ``{word,start,end}`` dicts."""
    words: list[dict[str, Any]] = []
    for seg in transcript.get("segments", []):
        words.extend(seg.get("words") or [])
    words.sort(key=lambda w: w["start"])
    return words


def transcribe_whisper(audio_path: str, model_size: str = "small",
                       progress: ProgressCb | None = None,
                       duration_hint: float = 0.0,
                       task: str = "transcribe",
                       language: str | None = None) -> dict[str, Any]:
    """Transcribe with faster-whisper. Uses float16 on CUDA, int8 on CPU.
    
    Tuned for multilingual & Indian podcasts (Hindi, Hinglish, English) with generous
    speech padding and Hindi/Hinglish token priming.
    """
    from faster_whisper import WhisperModel

    cuda = config.has_cuda()
    device, compute = ("cuda", "float16") if cuda else ("cpu", "int8")
    log.info("Loading Whisper '%s' on %s (%s)", model_size, device, compute)
    model = WhisperModel(model_size, device=device, compute_type=compute)
    
    # Robust VAD parameters so rapid conversational Hindi pauses aren't cut mid-sentence
    vad_params = {"min_silence_duration_ms": 1000, "speech_pad_ms": 400}
    
    opts: dict[str, Any] = {
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": vad_params,
        "initial_prompt": "Podcast conversation with Hindi, Hinglish, and English speech.",
        "task": task,
    }
    if language:
        opts["language"] = language

    seg_iter, info = model.transcribe(audio_path, **opts)
    log.info("Whisper detected language: %s (probability: %.2f)", info.language, getattr(info, "language_probability", 1.0))

    total = info.duration or duration_hint or 1.0
    segments: list[dict[str, Any]] = []
    for s in seg_iter:  # generator: iterating performs the transcription
        words = [{"word": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                 for w in (s.words or []) if w.word.strip()]
        segments.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip(), "words": words})
        if progress:
            progress(min(1.0, s.end / total))
    return {"source": "whisper", "model": model_size, "language": info.language,
            "duration": float(info.duration), "segments": segments}


def transcribe_youtube_api(video_id: str) -> dict[str, Any]:
    """Fallback: YouTube's own captions. Supports Hindi, Hinglish, and English transcripts."""
    from youtube_transcript_api import YouTubeTranscriptApi

    raw = []
    try:
        t_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try manual or generated Hindi / English transcripts
        t_obj = None
        try:
            t_obj = t_list.find_transcript(["hi", "en", "hi-Latn", "en-IN"])
        except Exception:
            # Fall back to whatever transcript YouTube provides
            t_obj = next(iter(t_list), None)

        if t_obj:
            raw = t_obj.fetch()
        elif hasattr(YouTubeTranscriptApi, "get_transcript"):
            raw = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi", "en", "hi-Latn", "en-IN"])
        else:
            fetched = YouTubeTranscriptApi().fetch(video_id)
            raw = [{"text": r.text, "start": r.start, "duration": r.duration} for r in fetched]
    except Exception as e:
        # Final attempt with direct get_transcript
        try:
            raw = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi", "en", "hi-Latn", "en-IN"])
        except Exception:
            raise TranscribeError(f"YouTube transcript fallback failed: {e}") from e

    segments = []
    for r in raw:
        text = " ".join(str(r["text"]).replace("\n", " ").split())
        toks = text.split()
        if not toks:
            continue
        start, dur = float(r["start"]), max(0.2, float(r["duration"]))
        step = dur / len(toks)
        words = [{"word": t, "start": start + i * step, "end": start + (i + 1) * step * 0.95}
                 for i, t in enumerate(toks)]
        segments.append({"start": start, "end": start + dur, "text": text, "words": words})
    if not segments:
        raise TranscribeError("YouTube returned an empty transcript")
    return {"source": "youtube-transcript-api", "model": "none", "language": "unknown",
            "duration": segments[-1]["end"], "segments": segments}


def transcribe(video_id: str, audio_path: str, model_size: str = "small",
               progress: ProgressCb | None = None, duration_hint: float = 0.0,
               use_cache: bool = True) -> dict[str, Any]:
    """Return a transcript, using the JSON cache when present.

    Tries faster-whisper first; on any failure falls back to youtube-transcript-api.
    """
    if use_cache:
        cached = load_cached(video_id)
        if cached:
            log.info("Using cached transcript for %s", video_id)
            return cached
    try:
        result = transcribe_whisper(audio_path, model_size, progress, duration_hint)
    except Exception as e:
        log.warning("Whisper failed (%s); trying youtube-transcript-api", e)
        result = transcribe_youtube_api(video_id)
    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path(video_id).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result
