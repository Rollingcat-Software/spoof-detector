"""Shared helpers used by all benchmark pipelines."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from tests.benchmark.runner import Sample

logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_frames(sample: Sample, *, max_frames: int = 30) -> list[np.ndarray]:
    """Decode a sample into BGR frames.

    - If payload is a path to an image, returns [frame].
    - If payload is a video path, samples up to max_frames evenly-spaced.
    - If payload is already an ndarray, returns [payload].

    cv2 is imported lazily so importing this module doesn't pull in OpenCV
    when only the metrics are needed.
    """
    if isinstance(sample.payload, np.ndarray):
        return [sample.payload]
    if not isinstance(sample.payload, str):
        raise TypeError(f"unsupported payload type: {type(sample.payload)}")

    path = Path(sample.payload)
    suffix = path.suffix.lower()
    import cv2  # lazy

    if suffix in IMAGE_EXTS:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("cv2.imread returned None for %s", path)
            return []
        return [frame]

    if suffix in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total == 0:
            cap.release()
            return []
        idxs = np.linspace(0, total - 1, num=min(max_frames, total)).astype(int)
        frames: list[np.ndarray] = []
        for target in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(target))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
        return frames

    raise ValueError(f"cannot decode file with extension: {suffix}")


def build_image_pipeline():
    """Build Ahmet's image-only pipeline (no temporal analyzers).

    Returns (pipeline, fuser).
    """
    from src.application.pipeline import SpoofDetectionPipeline
    from src.application.face_tracker import FaceTracker
    from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
    from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser
    from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
    from src.infrastructure.analyzers.device_boundary_analyzer import DeviceBoundaryAnalyzer
    from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
    from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer

    detector = MediaPipeFaceDetector()
    tracker = FaceTracker()
    face_analyzers = [
        MiniFASNetAnalyzer(),
        DeviceBoundaryAnalyzer(),
        TextureAnalyzer(),
        MoireAnalyzer(),
    ]
    frame_analyzers: list = []  # image-only — no whole-frame temporal
    fuser = MultiClassFuser()
    pipeline = SpoofDetectionPipeline(detector, tracker, face_analyzers, frame_analyzers, fuser)
    return pipeline, fuser


def build_video_pipeline():
    """Build Aysenur's video-only pipeline (temporal analyzers, no MiniFASNet).

    Includes blink, rPPG, screen-replay, micro-tremor, screen-flicker,
    landmark-variance, temporal — every analyzer that needs >1 frame.
    """
    from src.application.pipeline import SpoofDetectionPipeline
    from src.application.face_tracker import FaceTracker
    from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
    from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser
    from src.infrastructure.analyzers.blink_analyzer import BlinkAnalyzer
    from src.infrastructure.analyzers.rppg_analyzer import RPPGAnalyzer
    from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
    from src.infrastructure.analyzers.micro_tremor_analyzer import MicroTremorAnalyzer
    from src.infrastructure.analyzers.screen_flicker_analyzer import ScreenFlickerAnalyzer
    from src.infrastructure.analyzers.landmark_variance_analyzer import LandmarkVarianceAnalyzer
    from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer

    detector = MediaPipeFaceDetector()
    tracker = FaceTracker()
    face_analyzers = [
        BlinkAnalyzer(),
        RPPGAnalyzer(),
        ScreenReplayAnalyzer(),
        MicroTremorAnalyzer(),
        LandmarkVarianceAnalyzer(),
        TemporalAnalyzer(),
    ]
    frame_analyzers = [ScreenFlickerAnalyzer()]
    fuser = MultiClassFuser()
    pipeline = SpoofDetectionPipeline(detector, tracker, face_analyzers, frame_analyzers, fuser)
    return pipeline, fuser


def build_hybrid_pipeline():
    """Build the published hybrid pipeline — image + video analyzers fused.

    Every productized analyzer in `src/infrastructure/analyzers/` is wired.
    Calibrated weights (from MultiClassFuser) suppress anti-correlated
    analyzers (texture, moire) automatically.
    """
    from src.application.pipeline import SpoofDetectionPipeline
    from src.application.face_tracker import FaceTracker
    from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
    from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser
    from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
    from src.infrastructure.analyzers.device_boundary_analyzer import DeviceBoundaryAnalyzer
    from src.infrastructure.analyzers.texture_analyzer import TextureAnalyzer
    from src.infrastructure.analyzers.moire_analyzer import MoireAnalyzer
    from src.infrastructure.analyzers.blink_analyzer import BlinkAnalyzer
    from src.infrastructure.analyzers.rppg_analyzer import RPPGAnalyzer
    from src.infrastructure.analyzers.screen_replay_analyzer import ScreenReplayAnalyzer
    from src.infrastructure.analyzers.micro_tremor_analyzer import MicroTremorAnalyzer
    from src.infrastructure.analyzers.screen_flicker_analyzer import ScreenFlickerAnalyzer
    from src.infrastructure.analyzers.landmark_variance_analyzer import LandmarkVarianceAnalyzer
    from src.infrastructure.analyzers.temporal_analyzer import TemporalAnalyzer
    from src.infrastructure.analyzers.background_grid_analyzer import BackgroundGridAnalyzer
    from src.infrastructure.analyzers.ar_filter_analyzer import ARFilterAnalyzer

    detector = MediaPipeFaceDetector()
    tracker = FaceTracker()
    face_analyzers = [
        # Image-level (Ahmet's track)
        MiniFASNetAnalyzer(),
        DeviceBoundaryAnalyzer(),
        TextureAnalyzer(),
        MoireAnalyzer(),
        ARFilterAnalyzer(),
        # Video-level (Aysenur's track)
        BlinkAnalyzer(),
        RPPGAnalyzer(),
        ScreenReplayAnalyzer(),
        MicroTremorAnalyzer(),
        LandmarkVarianceAnalyzer(),
        TemporalAnalyzer(),
    ]
    frame_analyzers = [
        ScreenFlickerAnalyzer(),
        BackgroundGridAnalyzer(),
    ]
    fuser = MultiClassFuser()
    pipeline = SpoofDetectionPipeline(detector, tracker, face_analyzers, frame_analyzers, fuser)
    return pipeline, fuser


def aggregate_frame_scores(scores: list[float], *, mode: str = "peak_sensitive") -> float:
    """Aggregate per-frame live-ness scores into a session score.

    "mean"           — simple average. Used for image_only baseline.
    "peak_sensitive" — 0.5 * mean + 0.5 * worst-window. Published method.
                       Prevents spoof-burst dilution in mixed sessions.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if mode == "mean":
        return float(arr.mean())
    if mode == "peak_sensitive":
        # Worst 10% window
        k = max(1, len(arr) // 10)
        worst = float(np.sort(arr)[:k].mean())
        return 0.5 * float(arr.mean()) + 0.5 * worst
    raise ValueError(f"unknown aggregation mode: {mode}")
