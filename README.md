# ShortsAgent

Local YouTube → Shorts tool. Paste a link; it transcribes the video, asks an LLM for the most
engaging 20–50 s moments, downloads only those sections, and renders vertical 1080×1920 clips with a
hook title and/or word-by-word captions. Flask + vanilla JS + SQLite, FFmpeg for all video work.
Free and open-source end to end, for personal local use (no auth, binds to 127.0.0.1).

## Requirements
- Python 3.10 – 3.12 (MediaPipe wheels don't cover newer versions yet)
- **FFmpeg** with libass (any normal build) – the app prints a clear message at startup if it's missing
- Optional: NVIDIA GPU (auto-detected → Whisper `float16` + `h264_nvenc`; otherwise CPU `int8` + `libx264`)

## Install

### Linux / macOS
```bash
# FFmpeg:  sudo apt install ffmpeg   |   brew install ffmpeg
cd shortsagent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Windows (PowerShell)
```powershell
winget install Gyan.FFmpeg          # then open a NEW terminal so PATH updates
cd shortsagent
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

The bold font (Anton, OFL) is already in `assets/fonts/`; if it is missing the app downloads it on start.
Face-tracking uses MediaPipe's small face model, auto-downloaded to `assets/models/` on first smart-crop
use (falls back to an OpenCV Haar cascade, then to a centre crop).

## Choose an LLM (highlight selection)
**Ollama (default, offline)**
1. Install from https://ollama.com
2. `ollama pull llama3.1:8b` (any instruction-tuned model works; set `OLLAMA_MODEL` in `.env`)
3. Make sure it's running (`ollama serve`; the desktop app does this for you)

**Gemini (free API key)**
1. Create a key at https://aistudio.google.com/apikey
2. In `.env`: `GEMINI_API_KEY=your_key` (optionally `LLM_PROVIDER=gemini`, `GEMINI_MODEL=...`)

You can also pick the provider per job in the UI. Providers live in `llm.py`.

## Run
```bash
python app.py        # → http://127.0.0.1:5000
```
Paste a YouTube URL, set the options, click **Generate**. Cards appear as clips render; open **Edit clip**
to change the hook (3 LLM variants or custom), overlay mode, styles, trim start/end, with a live CSS preview
of the hook, then **Re-render**.

### CLI (same pipeline, no web UI)
```bash
python cli.py <url> --mode hook|captions|both [--clips 3] [--layout blur|smart]
python cli.py <url> --mode both --start 60 --end 95 --hook "He said WHAT?" --verify-audio   # one manual clip, no LLM
python cli.py --file my_video.mp4 --mode hook --start 5 --end 40                               # local file, no YouTube
```
Polish flags: `--loudnorm --zoom --silence-trim --progress-bar --music track.mp3` (put music files in `storage/music/`).
Outputs land in `storage/clips/`.

## Overlay modes
| Mode | What you get |
|---|---|
| `hook` | Styled title in the upper third for ~3 s (pop-in 0.2 s, fade-out 0.3 s). Original audio untouched. 4 presets: bold white + black outline, yellow highlight box, rounded dark box, gradient pill. |
| `captions` | 2–4 words at a time, active word highlighted, pop animation, centred at ~68 % of frame height. 3 colour presets. |
| `both` | Hook at ~19 % height, captions at ~68 % – never overlapping. |

Audio: the source stream is mapped straight through (`-map 0:a`) and only re-encoded to AAC for the MP4.
It is changed only if you enable *loudness normalise*, *silence trim* or *music* (music is mixed at −22 dB with `amix`, original stays dominant).

## Testing each stage
1. **Render + audio** – `python tests/test_pipeline.py` (synthetic video; verifies 1080×1920, AAC present, waveform correlation ≈ 1.0 for all three modes).
   On real clips: `python cli.py <url> --mode both --start 60 --end 95 --verify-audio`, or by hand:
   `ffprobe -v error -show_streams storage/clips/<file>.mp4`.
2. **Highlights** – start with `--clips 3` on a short video; check the log for "Selected N highlights". JSON problems are retried 3× and logged.
3. **Web app** – `python app.py`, open the page; job progress is polled every 2 s. API: `POST /api/jobs`, `GET /api/jobs/<id>`, `GET /api/videos/<id>/clips`, `PATCH /api/clips/<id>`, `POST /api/clips/<id>/render`.
4. **Smart crop** – choose *Smart crop* on a talking-head video; the log prints the plan (`static` / `dynamic` / `center`).

## Layout
```
app.py config.py db.py worker.py llm.py cli.py
pipeline/  download.py transcribe.py highlights.py crop.py overlay.py render.py runner.py
templates/index.html  static/style.css  static/app.js
assets/fonts/  storage/{audio,videos,clips,transcripts,music}   tests/test_pipeline.py
```

## Troubleshooting
- **"ffmpeg not found"** – install it and reopen the terminal.
- **Ollama connection error** – start Ollama and pull the model; or switch to Gemini.
- **yt-dlp errors** – `pip install -U yt-dlp` (YouTube changes often).
- **Captions look like a different font** – make sure `assets/fonts/Anton-Regular.ttf` exists.
- **Slow on CPU** – use Whisper `tiny`/`base`, fewer clips, or a smaller Ollama model.
- Download only content you have the rights to repurpose.
