"""Hybrid image+video pipeline — the published method.

Combines:
  Image-level (Ahmet's track):
    - MiniFASNet ONNX (per-frame, +94.7 discrimination gap)
    - Device boundary (per-frame, phone-bezel detection)
    - Texture / moire (per-frame, anti-correlated → suppressed by fuser)
    - AR-filter analyzer (per-frame heuristic)

  Video-level (Aysenur's track):
    - Blink (EAR over time)
    - rPPG (skin pulse via FFT)
    - Screen-replay anti-spoof (specular + skin-color FFT)
    - Micro-tremor (8–12 Hz oscillation)
    - Landmark-variance (zero variance = photo)
    - Temporal-consistency (cross-frame motion plausibility)

  Whole-frame:
    - Screen-flicker (50/60 Hz LCD/OLED detection)
    - Background-grid (proctoring scene stability)

Multi-class fuser computes calibrated weighted vote → per-category
probability distribution. Session-level aggregation uses peak-sensitive
verdict to prevent spoof-burst dilution.
"""
from __future__ import annotations

import logging
import numpy as np

from tests.benchmark.runner import Sample
from tests.benchmark.pipelines._common import (
    load_frames,
    build_hybrid_pipeline,
    aggregate_frame_scores,
)

logger = logging.getLogger(__name__)


def score_sample(sample: Sample, *, max_frames: int = 30) -> float:
    pipeline, _fuser = build_hybrid_pipeline()
    frames = load_frames(sample, max_frames=max_frames)

    per_frame_scores: list[float] = []
    for frame in frames:
        analysis = pipeline.process(frame)
        if not analysis.classifications:
            continue
        best = max(analysis.classifications.values(), key=lambda c: c.confidence or 0.0)
        per_frame_scores.append(float(best.p_real))

    if not per_frame_scores:
        return 0.5
    # If we have a video (>= 2 frames), use peak-sensitive (the published method).
    # If single image, mean == only sample → falls back to that sample.
    mode = "peak_sensitive" if len(per_frame_scores) >= 2 else "mean"
    return aggregate_frame_scores(per_frame_scores, mode=mode)
