"""Stage 5: smart 9:16 crop planning with face tracking.

* Detects faces about every 0.25 s (MediaPipe Tasks API, legacy MediaPipe
  ``solutions`` API, or an OpenCV Haar cascade as last resort).
* Smooths the face centre with an exponential moving average (EMA).
* Emits a piecewise-linear crop-x expression for FFmpeg's ``crop`` filter, or a
  single static x when the subject barely moves, or a centre crop when no face is found.
"""
from __future__ import annotations

import statistics
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import config

log = config.log.getChild("crop")

SAMPLE_STEP_S = 0.25
EMA_ALPHA = 0.25
STATIC_THRESHOLD = 0.10      # subject moves < 10% of crop width -> static crop
KEYFRAME_EVERY_S = 0.5
MIN_KEY_DELTA_PX = 6


@dataclass
class CropPlan:
    """How to cut a 9:16 window out of the source frame."""

    mode: str                       # 'center' | 'static' | 'dynamic' | 'fill'
    src_w: int
    src_h: int
    crop_w: int
    crop_h: int
    x_static: float = 0.0
    keyframes: list[tuple[float, float]] = field(default_factory=list)  # (t seconds, x px)
    faces_found: int = 0

    def x_expr(self) -> str:
        """FFmpeg expression for the crop x offset as a function of ``t``."""
        if self.mode != "dynamic" or len(self.keyframes) < 2:
            return f"{self.x_static:.1f}"
        (t0, x0), rest = self.keyframes[0], self.keyframes[1:]
        expr = f"{x0:.1f}"
        prev_t, prev_x = t0, x0
        for t, x in rest:
            dt = max(1e-3, t - prev_t)
            expr += f"+({x - prev_x:.1f})*clip((t-{prev_t:.3f})/{dt:.3f},0,1)"
            prev_t, prev_x = t, x
        return expr


# --------------------------------------------------------------------------
# face detection backends
# --------------------------------------------------------------------------
class FaceFinder:
    """Return face boxes ``(cx, cy, w, h)`` in pixel coordinates of the given frame."""

    def __init__(self) -> None:
        self.backend = "none"
        self._mp = None
        self._haar = None
        self._init()

    def _init(self) -> None:
        # 1) MediaPipe Tasks API (current releases)
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            model = config.MODELS_DIR / "blaze_face_short_range.tflite"
            if not model.exists():
                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                log.info("Downloading MediaPipe face model...")
                urllib.request.urlretrieve(config.FACE_MODEL_URL, model)  # noqa: S310
            opts = vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model)),
                min_detection_confidence=0.5)
            self._mp = (mp, vision.FaceDetector.create_from_options(opts))
            self.backend = "mediapipe-tasks"
            return
        except Exception as e:
            log.info("MediaPipe Tasks unavailable (%s)", e)
        # 2) MediaPipe legacy solutions API
        try:
            import mediapipe as mp

            det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            self._mp = (mp, det)
            self.backend = "mediapipe-solutions"
            return
        except Exception as e:
            log.info("MediaPipe solutions unavailable (%s)", e)
        # 3) OpenCV Haar cascade
        try:
            import cv2

            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not self._haar.empty():
                self.backend = "opencv-haar"
        except Exception as e:  # pragma: no cover
            log.warning("No face detector available: %s", e)

    def detect(self, frame_bgr: Any) -> list[tuple[float, float, float, float]]:
        import cv2

        h, w = frame_bgr.shape[:2]
        try:
            if self.backend == "mediapipe-tasks":
                mp, det = self._mp
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                out = []
                for d in res.detections:
                    b = d.bounding_box
                    out.append((b.origin_x + b.width / 2, b.origin_y + b.height / 2, b.width, b.height))
                return out
            if self.backend == "mediapipe-solutions":
                _, det = self._mp
                res = det.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                out = []
                for d in res.detections or []:
                    r = d.location_data.relative_bounding_box
                    out.append(((r.xmin + r.width / 2) * w, (r.ymin + r.height / 2) * h, r.width * w, r.height * h))
                return out
            if self.backend == "opencv-haar":
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(max(30, w // 20),) * 2)
                return [(x + fw / 2, y + fh / 2, fw, fh) for (x, y, fw, fh) in faces]
        except Exception as e:
            log.debug("detect failed: %s", e)
        return []


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def sample_face_centers(src: str, ss: float, dur: float, finder: FaceFinder | None = None,
                        detect_width: int = 640) -> tuple[list[tuple[float, float | None]], int, int]:
    """Return ``([(t, face_cx_px_or_None), ...], src_w, src_h)`` for ``[ss, ss+dur]`` of ``src``.

    ``t`` is relative to ``ss``. The face centre is in source pixel coordinates.
    """
    import cv2

    finder = finder or FaceFinder()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {src}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale = min(1.0, detect_width / max(1, src_w))
    cap.set(cv2.CAP_PROP_POS_MSEC, ss * 1000.0)
    start_idx = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    step_frames = max(1, int(round(SAMPLE_STEP_S * fps)))
    n_frames = int(dur * fps)
    samples: list[tuple[float, float | None]] = []
    prev_cx: float | None = None
    for i in range(0, n_frames, step_frames):
        # advance to frame i (grab = decode without conversion)
        target = start_idx + i
        cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        while cur < target:
            if not cap.grab():
                break
            cur += 1
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        faces = finder.detect(small)
        cx = None
        if faces:
            # prefer the largest face; with a previous centre, prefer a nearby one of similar size
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)
            best = faces[0]
            if prev_cx is not None and len(faces) > 1:
                near = min(faces, key=lambda f: abs(f[0] / scale - prev_cx))
                if near[2] * near[3] > 0.5 * best[2] * best[3]:
                    best = near
            cx = best[0] / scale
            prev_cx = cx
        samples.append((i / fps, cx))
    cap.release()
    return samples, src_w, src_h


def plan_from_samples(samples: list[tuple[float, float | None]], src_w: int, src_h: int) -> CropPlan:
    """Pure function: turn face samples into a :class:`CropPlan` (easy to unit test)."""
    target_ar = config.OUT_W / config.OUT_H  # 0.5625
    if src_w / src_h <= target_ar + 0.01:
        return CropPlan("fill", src_w, src_h, src_w, src_h)
    crop_h = _even(src_h)
    crop_w = _even(min(src_w, src_h * target_ar))
    max_x = src_w - crop_w
    centers = [c for _, c in samples if c is not None]
    if not centers:
        return CropPlan("center", src_w, src_h, crop_w, crop_h, x_static=max_x / 2)

    # fill gaps by holding the last known centre (and back-fill the start)
    filled: list[float] = []
    last = centers[0]
    for _, c in samples:
        last = c if c is not None else last
        filled.append(last)
    # EMA smoothing
    sm = [filled[0]]
    for c in filled[1:]:
        sm.append(EMA_ALPHA * c + (1 - EMA_ALPHA) * sm[-1])
    xs = [min(max(c - crop_w / 2, 0.0), max_x) for c in sm]
    times = [t for t, _ in samples]

    if max(xs) - min(xs) < STATIC_THRESHOLD * crop_w:
        return CropPlan("static", src_w, src_h, crop_w, crop_h, x_static=statistics.median(xs),
                        faces_found=len(centers))

    # thin to keyframes every ~0.5 s, dropping tiny moves
    keys: list[tuple[float, float]] = [(times[0], xs[0])]
    for t, x in zip(times, xs):
        if t - keys[-1][0] >= KEYFRAME_EVERY_S - 1e-6:
            if abs(x - keys[-1][1]) >= MIN_KEY_DELTA_PX:
                keys.append((t, x))
            else:
                keys.append((t, keys[-1][1]))
    if keys[-1][0] < times[-1]:
        keys.append((times[-1] + SAMPLE_STEP_S, xs[-1]))
    if len(keys) > 120:  # keep expression/command length sane
        stride = len(keys) // 120 + 1
        keys = keys[::stride] + [keys[-1]]
    return CropPlan("dynamic", src_w, src_h, crop_w, crop_h, x_static=keys[0][1],
                    keyframes=keys, faces_found=len(centers))


def plan_crop(src: str, ss: float, dur: float) -> CropPlan:
    """Analyse ``src[ss:ss+dur]`` and return a crop plan. Falls back to centre crop on any error."""
    try:
        samples, w, h = sample_face_centers(src, ss, dur)
        plan = plan_from_samples(samples, w, h)
        log.info("Crop plan: %s (faces in %d/%d samples)", plan.mode, plan.faces_found, len(samples))
        return plan
    except Exception as e:
        log.warning("Smart crop failed (%s); using centre crop", e)
        import cv2

        cap = cv2.VideoCapture(src)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not w or not h:
            raise
        return plan_from_samples([], w, h)
