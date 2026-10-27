"""Stage 6: build ASS subtitle files for the three overlay modes.

* ``hook``     - styled title for the first N seconds (pop-in, fade-out), no captions
* ``captions`` - word-by-word karaoke-style captions (2-4 words, active word highlighted)
* ``both``     - hook in the top third + captions in the lower-middle

Everything is produced with pysubs2 and burned in later by FFmpeg's ``ass`` filter.
This module never touches audio.
"""
from __future__ import annotations

import math
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pysubs2
from pysubs2 import Color, SSAEvent, SSAStyle

import config

log = config.log.getChild("overlay")

W, H = config.OUT_W, config.OUT_H
TimeMap = Callable[[float], float]

OVERLAY_MODES = ("hook", "captions", "both")

HOOK_STYLES: dict[str, dict[str, Any]] = {
    "bold_outline": {"label": "Bold white + black outline", "text": "FFFFFF", "outline": "000000",
                     "bord": 9, "shadow": 3, "box": None},
    "yellow_box": {"label": "Yellow highlight box", "text": "111111", "outline": "111111",
                   "bord": 0, "shadow": 0, "box": {"kind": "lines", "color": "FFE11A", "radius": 14,
                                                    "padx": 30, "pady": 6, "alpha": 0}},
    "rounded_dark": {"label": "Rounded dark box", "text": "FFFFFF", "outline": "000000",
                     "bord": 0, "shadow": 0, "box": {"kind": "block", "color": "0B0B10", "radius": 44,
                                                      "padx": 48, "pady": 32, "alpha": 38}},
    "gradient_pill": {"label": "Gradient pill", "text": "FFFFFF", "outline": "000000",
                      "bord": 2, "shadow": 2, "box": {"kind": "gradient", "colors": ("FF512F", "DD2476"),
                                                       "radius": None, "padx": 56, "pady": 30, "alpha": 0}},
}

CAPTION_STYLES: dict[str, dict[str, Any]] = {
    "white_yellow": {"label": "White + yellow highlight", "text": "FFFFFF", "active": "FFE11A",
                     "outline": "000000", "bord": 8, "shadow": 3},
    "white_clean": {"label": "Pure white (clean outline)", "text": "FFFFFF", "active": "FFFFFF",
                    "outline": "000000", "bord": 8, "shadow": 3},
    "white_green": {"label": "White + green highlight", "text": "FFFFFF", "active": "2CFF6B",
                    "outline": "000000", "bord": 8, "shadow": 3},
    "white_pink": {"label": "White + pink highlight", "text": "FFFFFF", "active": "FF4FA3",
                   "outline": "1A0B2E", "bord": 9, "shadow": 3},
}

# vertical anchors (fraction of frame height)
HOOK_CY = 0.19
CAPTION_CY = 0.685


# --------------------------------------------------------------------------
# fonts & measuring
# --------------------------------------------------------------------------
def font_path() -> Path:
    return config.FONTS_DIR / config.FONT_FILE


def ensure_font() -> Path | None:
    """Make sure the bundled bold font exists (download once if missing).

    Returns the font path, or ``None`` if it is unavailable (FFmpeg then falls
    back to a system font and text-fitting uses a width heuristic).
    """
    p = font_path()
    if p.exists() and p.stat().st_size > 10_000:
        return p
    try:
        config.FONTS_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Downloading font %s", config.FONT_URL)
        urllib.request.urlretrieve(config.FONT_URL, p)  # noqa: S310 - fixed https URL from config
        return p if p.stat().st_size > 10_000 else None
    except Exception as e:
        log.warning("Could not download font (%s). Drop a .ttf into %s", e, config.FONTS_DIR)
        return None


class TextMeasurer:
    """Measure text width the way libass will render it.

    libass sizes fonts so that (ascent + descent) == ASS ``Fontsize``, so we
    derive the PIL em-size from the font's cell height.
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._cell = None
        if path and path.exists():
            try:
                from PIL import ImageFont

                a, d = ImageFont.truetype(str(path), 1000).getmetrics()
                self._cell = float(a + d)
            except Exception as e:  # pragma: no cover
                log.warning("PIL font measure unavailable: %s", e)

    @lru_cache(maxsize=256)
    def _font(self, em: int):
        from PIL import ImageFont

        return ImageFont.truetype(str(self.path), em)

    def width(self, text: str, fontsize: float) -> float:
        if not text:
            return 0.0
        if self._cell:
            em = max(1, round(fontsize * 1000.0 / self._cell))
            return float(self._font(em).getlength(text))
        return 0.46 * fontsize * len(text)  # heuristic for condensed bold faces


def wrap_balanced(words: list[str], fontsize: float, max_w: float, m: TextMeasurer) -> list[str]:
    """Greedy wrap, then tighten the width so lines come out visually balanced."""

    def greedy(limit: float) -> list[str]:
        lines, cur = [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if cur and m.width(trial, fontsize) > limit:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    base = greedy(max_w)
    n = len(base)
    if n <= 1:
        return base
    lo = max(m.width(w, fontsize) for w in words)
    hi = max_w
    for _ in range(14):  # binary search the narrowest width keeping n lines
        mid = (lo + hi) / 2
        if len(greedy(mid)) <= n:
            hi = mid
        else:
            lo = mid
    return greedy(hi)


def fit_hook(text: str, m: TextMeasurer, max_w: float, max_lines: int = 3,
             pad_x: float = 0.0) -> tuple[list[str], int]:
    """Pick the largest font size (<=150) whose wrapped text is <= ``max_lines`` lines
    and whose widest line fits ``max_w`` (minus box padding)."""
    words = text.split()
    avail = max_w - 2 * pad_x
    for fs in range(150, 43, -4):
        lines = wrap_balanced(words, fs, avail, m)
        if len(lines) <= max_lines and max(m.width(l, fs) for l in lines) <= avail:
            return lines, fs
    return wrap_balanced(words, 44, avail, m)[:max_lines], 44


# --------------------------------------------------------------------------
# ASS helpers
# --------------------------------------------------------------------------
def _rgb(hexstr: str) -> tuple[int, int, int]:
    hexstr = hexstr.lstrip("#")
    return int(hexstr[0:2], 16), int(hexstr[2:4], 16), int(hexstr[4:6], 16)


def _ass_c(hexstr: str) -> str:
    """'RRGGBB' -> ASS inline colour '&HBBGGRR&'."""
    r, g, b = _rgb(hexstr)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _ass_a(alpha: int) -> str:
    """0 (opaque) .. 255 (transparent) -> '&HAA&'."""
    return f"&H{max(0, min(255, alpha)):02X}&"


def _lerp_hex(a: str, b: str, t: float) -> str:
    ra, rb = _rgb(a), _rgb(b)
    return "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ra, rb))


def _ts(t: float) -> int:
    return max(0, int(round(t * 1000)))


def _rounded_rect_path(w: float, h: float, r: float) -> str:
    """ASS vector drawing of a rounded rectangle whose bbox is (0,0)-(w,h)."""
    r = max(0.0, min(r, w / 2, h / 2))
    k = 0.5523 * r
    f = lambda v: f"{v:.1f}"  # noqa: E731
    if r < 0.5:
        return f"m 0 0 l {f(w)} 0 {f(w)} {f(h)} 0 {f(h)}"
    return (f"m {f(r)} 0 l {f(w - r)} 0 "
            f"b {f(w - r + k)} 0 {f(w)} {f(r - k)} {f(w)} {f(r)} "
            f"l {f(w)} {f(h - r)} "
            f"b {f(w)} {f(h - r + k)} {f(w - r + k)} {f(h)} {f(w - r)} {f(h)} "
            f"l {f(r)} {f(h)} "
            f"b {f(r - k)} {f(h)} 0 {f(h - r + k)} 0 {f(h - r)} "
            f"l 0 {f(r)} "
            f"b 0 {f(r - k)} {f(r - k)} 0 {f(r)} 0")


def _pop_tags(scale0: int = 72, over: int = 106) -> str:
    """Scale-in with slight overshoot (~0.22s total)."""
    return (f"\\fscx{scale0}\\fscy{scale0}"
            f"\\t(0,150,\\fscx{over}\\fscy{over})\\t(150,230,\\fscx100\\fscy100)")


# --------------------------------------------------------------------------
# hook
# --------------------------------------------------------------------------
def _add_hook(subs: pysubs2.SSAFile, text: str, style_key: str, duration: float,
              m: TextMeasurer) -> None:
    st = HOOK_STYLES.get(style_key, HOOK_STYLES["bold_outline"])
    box = st["box"]
    text = " ".join(text.upper().split())
    pad_x = box["padx"] if box else 0
    lines, fs = fit_hook(text, m, 0.85 * W, 3, pad_x)

    style = SSAStyle(
        fontname=config.FONT_FAMILY, fontsize=fs, bold=False,
        primarycolor=Color(*_rgb(st["text"]), 0), outlinecolor=Color(*_rgb(st["outline"]), 0),
        backcolor=Color(0, 0, 0, 90), outline=st["bord"], shadow=st["shadow"],
        alignment=5, borderstyle=1, marginl=0, marginr=0, marginv=0, spacing=1.0,
    )
    subs.styles["Hook"] = style

    cx, cy = W / 2, H * HOOK_CY
    end = duration
    fade_out = min(300, int(duration * 1000 / 3))
    common = f"\\an5\\pos({cx:.0f},{cy:.0f})"
    line_h = fs * 1.0
    block_h = line_h * len(lines)
    widths = [m.width(l, fs) for l in lines]
    pop = _pop_tags()

    def drawing_event(dw: float, dh: float, ox: float, oy: float, color: str, radius: float,
                      alpha: int, layer: int, clip: tuple[float, float, float, float] | None = None) -> None:
        """Drawing of size dw x dh centred at (ox, oy); pops about its own centre."""
        tags = (f"\\an5\\pos({ox:.0f},{oy:.0f})\\p1\\bord0\\shad0\\1c{_ass_c(color)}\\1a{_ass_a(alpha)}"
                f"{pop}\\fad(0,{fade_out})")
        if clip:
            tags += _clip_anim(clip, ox, oy)
        subs.append(SSAEvent(start=0, end=_ts(end), layer=layer, style="Hook",
                             text="{" + tags + "}" + _rounded_rect_path(dw, dh, radius) + "{\\p0}"))

    if box:
        if box["kind"] == "block":
            bw = max(widths) + 2 * box["padx"]
            bh = block_h + 2 * box["pady"]
            drawing_event(bw, bh, cx, cy, box["color"], box["radius"], box["alpha"], 0)
        elif box["kind"] == "lines":  # per-line highlighter bars
            for i, w in enumerate(widths):
                ly = cy - block_h / 2 + line_h * (i + 0.5)
                drawing_event(w + 2 * box["padx"], line_h * 0.92 + 2 * box["pady"] * 0.5, cx, ly,
                              box["color"], box["radius"], box["alpha"], 0)
        elif box["kind"] == "gradient":
            bw = max(widths) + 2 * box["padx"]
            bh = block_h + 2 * box["pady"]
            radius = min(bh / 2, bw / 2)
            n = 26
            c0, c1 = box["colors"]
            for i in range(n):
                x0 = cx - bw / 2 + bw * i / n
                x1 = cx - bw / 2 + bw * (i + 1) / n + 1.0  # 1px overlap hides seams
                col = _lerp_hex(c0, c1, (i + 0.5) / n)
                drawing_event(bw, bh, cx, cy, col, radius, box["alpha"], 0,
                              clip=(x0, cy - bh / 2 - 4, x1, cy + bh / 2 + 4))

    tags = f"{common}{pop}\\fad(0,{fade_out})"
    subs.append(SSAEvent(start=0, end=_ts(end), layer=1, style="Hook",
                         text="{" + tags + "}" + "\\N".join(lines)))


def _clip_anim(rect: tuple[float, float, float, float], cx: float, cy: float) -> str:
    """Rect \\clip that follows the same 72% -> 106% -> 100% scale animation about (cx, cy)."""

    def sc(s: float) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        return (round(cx + (x0 - cx) * s), round(cy + (y0 - cy) * s),
                round(cx + (x1 - cx) * s), round(cy + (y1 - cy) * s))

    a, b, c = sc(0.72), sc(1.06), sc(1.0)
    return (f"\\clip({a[0]},{a[1]},{a[2]},{a[3]})"
            f"\\t(0,150,\\clip({b[0]},{b[1]},{b[2]},{b[3]}))"
            f"\\t(150,230,\\clip({c[0]},{c[1]},{c[2]},{c[3]}))")


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------
_SENT_END = (".", "?", "!", "…")


def group_words(words: list[dict[str, Any]], m: TextMeasurer, fs: int,
                max_words: int = 4, max_w: float = 0.88 * W) -> list[list[dict[str, Any]]]:
    """Split words into display groups of 2-4 words (never wider than ``max_w`` on 2 lines)."""
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        cur.append(w)
        text = " ".join(x["word"].upper() for x in cur)
        nxt = words[i + 1] if i + 1 < len(words) else None
        too_wide = m.width(text, fs) > 2 * max_w * 0.92
        sentence_end = w["word"].endswith(_SENT_END)
        long_gap = nxt is not None and nxt["start"] - w["end"] > 0.6
        if len(cur) >= max_words or too_wide or sentence_end or long_gap or nxt is None:
            if len(cur) == 1 and groups and not (sentence_end or long_gap) and nxt is None:
                groups[-1].extend(cur)  # avoid a dangling one-word group at the end
            else:
                groups.append(cur)
            cur = []
    return groups


def _split_two_lines(ws: list[str], fs: int, m: TextMeasurer, max_w: float) -> list[list[int]]:
    """Return word indices per line (1 line if it fits, else best 2-line split)."""
    full = " ".join(ws)
    if m.width(full, fs) <= max_w or len(ws) < 2:
        return [list(range(len(ws)))]
    best, best_cost = 1, float("inf")
    for k in range(1, len(ws)):
        a, b = " ".join(ws[:k]), " ".join(ws[k:])
        cost = max(m.width(a, fs), m.width(b, fs))
        if cost < best_cost:
            best, best_cost = k, cost
    return [list(range(best)), list(range(best, len(ws)))]


def _add_captions(subs: pysubs2.SSAFile, words: list[dict[str, Any]], style_key: str,
                  m: TextMeasurer, tmap: TimeMap, clip_len: float) -> None:
    st = CAPTION_STYLES.get(style_key, CAPTION_STYLES["white_yellow"])
    fs = 84
    subs.styles["Caption"] = SSAStyle(
        fontname=config.FONT_FAMILY, fontsize=fs, bold=False,
        primarycolor=Color(*_rgb(st["text"]), 0), outlinecolor=Color(*_rgb(st["outline"]), 0),
        backcolor=Color(0, 0, 0, 100), outline=st["bord"], shadow=st["shadow"],
        alignment=5, borderstyle=1, marginl=60, marginr=60, marginv=0, spacing=1.0,
    )
    words = [w for w in words if w["word"].strip() and w["end"] > 0]
    groups = group_words(words, m, fs)
    cx, cy = W / 2, H * CAPTION_CY
    act = _ass_c(st["active"])
    for gi, g in enumerate(groups):
        texts = [w["word"].upper() for w in g]
        lines = _split_two_lines(texts, fs, m, 0.88 * W)
        g_next_start = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else clip_len
        for wi, w in enumerate(g):
            start = max(0.0, w["start"])
            if wi + 1 < len(g):
                end = g[wi + 1]["start"]
            else:
                hold = w["end"] + 0.12
                end = g_next_start if g_next_start - w["end"] < 0.3 else hold
            end = min(max(end, start + 0.08), clip_len)
            if start >= clip_len:
                continue
            parts: list[str] = []
            for li, idxs in enumerate(lines):
                seg = []
                for j in idxs:
                    if j == wi:
                        seg.append("{\\1c" + act + "\\t(0,70,\\fscx112\\fscy112)\\t(70,150,\\fscx100\\fscy100)}"
                                   + texts[j] + "{\\r}")
                    else:
                        seg.append(texts[j])
                parts.append(" ".join(seg))
            body = "\\N".join(parts)
            subs.append(SSAEvent(start=_ts(tmap(start)), end=_ts(tmap(end)), layer=2, style="Caption",
                                 text="{\\an5\\pos(%d,%d)}" % (cx, cy) + body))


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def build_ass(mode: str, words: list[dict[str, Any]], clip_len: float, hook_text: str | None = None,
              hook_style: str = "bold_outline", caption_style: str = "white_yellow",
              hook_duration: float = config.HOOK_DURATION_S,
              time_map: TimeMap | None = None) -> pysubs2.SSAFile:
    """Build the subtitle file for ``mode`` (``hook`` | ``captions`` | ``both``).

    ``words`` must be clip-relative (0 = first frame of the clip). ``time_map``
    converts clip-relative seconds into output seconds (used when silence is trimmed).
    """
    if mode not in OVERLAY_MODES:
        raise ValueError(f"Unknown overlay mode {mode!r}; expected one of {OVERLAY_MODES}")
    tmap = time_map or (lambda t: t)
    m = TextMeasurer(ensure_font())
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = str(W)
    subs.info["PlayResY"] = str(H)
    subs.info["WrapStyle"] = "2"  # we break lines ourselves
    subs.info["ScaledBorderAndShadow"] = "yes"
    if mode in ("hook", "both") and hook_text and hook_text.strip():
        _add_hook(subs, hook_text, hook_style, max(1.0, min(hook_duration, clip_len)), m)
    if mode in ("captions", "both"):
        _add_captions(subs, words, caption_style, m, tmap, clip_len)
    return subs


def write_ass(subs: pysubs2.SSAFile, path: str | Path) -> Path:
    """Write the file as ASS and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(path), format_="ass")
    return path
