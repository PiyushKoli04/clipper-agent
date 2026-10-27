#!/usr/bin/env python3
"""ShortsAgent CLI - runs the same pipeline as the web UI.

Examples
--------
  # automatic: LLM picks 3 clips, hook-only overlay
  python cli.py https://youtu.be/VIDEOID --mode hook --clips 3

  # test ONE clip with a manual range and hook (no LLM needed)
  python cli.py https://youtu.be/VIDEOID --mode both --start 60 --end 95 --hook "He said WHAT?"

  # work on a local video file instead of YouTube
  python cli.py --file my_video.mp4 --mode captions --start 5 --end 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import config
import db
from pipeline import overlay, render, runner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Turn a YouTube video into vertical Shorts.")
    ap.add_argument("url", nargs="?", help="YouTube URL")
    ap.add_argument("--file", help="use a local video file instead of a YouTube URL (testing)")
    ap.add_argument("--mode", choices=["hook", "captions", "both"], default="hook", help="overlay mode")
    ap.add_argument("--clips", type=int, default=3, help="number of clips (LLM mode)")
    ap.add_argument("--layout", choices=["blur", "smart"], default="blur")
    ap.add_argument("--hook-style", choices=list(overlay.HOOK_STYLES), default="bold_outline")
    ap.add_argument("--caption-style", choices=list(overlay.CAPTION_STYLES), default="white_yellow")
    ap.add_argument("--hook-duration", type=float, default=config.HOOK_DURATION_S)
    ap.add_argument("--whisper", default=config.WHISPER_MODEL, help="tiny|base|small|medium|large-v3")
    ap.add_argument("--llm", default=config.LLM_PROVIDER, choices=["ollama", "gemini", "groq"])
    ap.add_argument("--start", type=float, help="manual clip start (seconds); skips the LLM")
    ap.add_argument("--end", type=float, help="manual clip end (seconds)")
    ap.add_argument("--hook", help="hook text for a manual clip")
    ap.add_argument("--loudnorm", action="store_true", help="loudness-normalise audio (off by default)")
    ap.add_argument("--zoom", action="store_true", help="slow zoom-in 1.0 -> 1.06")
    ap.add_argument("--silence-trim", action="store_true", help="shorten pauses > 0.5s")
    ap.add_argument("--progress-bar", action="store_true", help="thin progress bar")
    ap.add_argument("--music", help="file name in storage/music mixed at -22 dB")
    ap.add_argument("--verify-audio", action="store_true", help="ffprobe + correlate audio with the source")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    config.setup_logging(10 if a.verbose else 20)
    if not config.check_ffmpeg():
        return 2
    if not (a.url or a.file):
        ap.error("give a YouTube URL or --file")
    if (a.start is None) != (a.end is None):
        ap.error("--start and --end must be given together")

    params = {
        "url": a.url, "local_file": a.file, "overlay_mode": a.mode, "num_clips": a.clips, "layout": a.layout,
        "hook_style": a.hook_style, "caption_style": a.caption_style, "hook_duration": a.hook_duration,
        "whisper_model": a.whisper, "llm_provider": a.llm, "start": a.start, "end": a.end, "hook_text": a.hook,
        "loudnorm": a.loudnorm, "zoom": a.zoom, "silence_trim": a.silence_trim, "progress_bar": a.progress_bar,
        "music": a.music,
    }
    last = [""]

    def show(stage: str, pct: float) -> None:
        line = f"[{pct:5.1f}%] {stage}"
        if line != last[0]:
            print(line, flush=True)
            last[0] = line

    try:
        vid = runner.run_pipeline(params, show)
    except Exception as e:  # clear message, non-zero exit
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    print("\nClips:")
    for c in db.list_clips(vid):
        print(f"  #{c['id']} score={c['score']:.0f} {c['start']:.1f}-{c['end']:.1f}s  {c['status']}  "
              f"hook={c['hook_text']!r}\n      {c['output_path']}")
        if a.verify_audio and c["output_path"]:
            info = render.probe(c["output_path"])
            print(f"      ffprobe: {info['width']}x{info['height']} audio={info['audio_codec']} dur={info['duration']:.2f}s")
            if c["src_path"]:
                v = render.verify_audio(c["src_path"], c["output_path"], c["start"] - c["src_start"], c["end"] - c["start"])
                print("      audio check:", json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
