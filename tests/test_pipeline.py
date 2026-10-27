"""Self-contained checks (no network, no Whisper, no LLM needed).

Run:  python tests/test_pipeline.py
Needs ffmpeg/ffprobe on PATH. Generates a synthetic video with a tone, renders all three
overlay modes and verifies with ffprobe + waveform correlation that the original audio survives.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import crop, download, highlights, render  # noqa: E402


def test_urls():
    assert download.parse_youtube_url("https://youtu.be/dQw4w9WgXcQ?t=3")[0] == "dQw4w9WgXcQ"
    for bad in ("https://evil.com/watch?v=dQw4w9WgXcQ", "https://youtube.com/watch?v=x", "javascript:alert(1)"):
        try:
            download.parse_youtube_url(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_json_and_snapping():
    raw = '```json\n{"clips":[{"start":10,"end":40,},]}\n```'
    assert highlights.parse_llm_json(raw)[0]["start"] == 10
    words, t = [], 0.0
    for _ in range(30):
        for i in range(10):
            words.append({"word": f"w{i}" + ("." if i == 9 else ""), "start": t, "end": t + 0.4})
            t += 0.45
        t += 0.2
    a, b = highlights.snap_clip(words, 5, 120, t)
    assert 20 <= b - a <= 50


def test_crop_plan():
    moving = [(i * 0.25, 300 + 68 * i * 0.25) for i in range(40)]
    assert crop.plan_from_samples(moving, 1280, 720).mode == "dynamic"
    assert crop.plan_from_samples([(i * .25, 640.0) for i in range(40)], 1280, 720).mode == "static"
    assert crop.plan_from_samples([(0, None)], 1280, 720).mode == "center"


def test_render_modes_keep_audio():
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=8",
                    "-f", "lavfi", "-i", "sine=frequency=440:d=8", "-c:v", "libx264", "-c:a", "aac", "-shortest",
                    str(src)], check=True)
    words = [{"word": w, "start": 0.3 + i * 0.35, "end": 0.3 + i * 0.35 + 0.3}
             for i, w in enumerate("this is a test of the word by word captions system".split())]
    for mode in ("hook", "captions", "both"):
        out = tmp / f"{mode}.mp4"
        opts = render.RenderOptions(overlay_mode=mode, hook_text="Nobody expected this")
        render.render_clip(src, out, 0.5, 6.0, words, opts)
        info = render.probe(out)
        assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"], mode
        v = render.verify_audio(src, out, 0.5, 6.0)
        assert v["correlation"] > 0.98, (mode, v)
        print(f"  {mode:8s} ok  audio correlation={v['correlation']:.4f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL OK")
