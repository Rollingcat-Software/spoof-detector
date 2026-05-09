"""Image-only pipeline (Ahmet's contribution).

For each sample:
  1. If sample.payload is a video path, sample N evenly-spaced frames.
  2. For each frame, run face detection → crop → MiniFASNet + texture/moire/AR.
  3. Average the per-frame live-ness scores → final score.

This is the strong per-frame baseline. Aysenur's temporal signals are
deliberately NOT applied — that's the ablation point in the paper.
"""
from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable

import numpy as np

from tests.benchmark.runner import Sample
from tests.benchmark.pipelines._common import (
    load_frames,
    build_image_pipeline,
    aggregate_frame_scores,
)

logger = logging.getLogger(__name__)


def score_sample(sample: Sample, *, max_frames: int = 30) -> float:
    """Return [0,1] live-ness score for one sample (image OR video)."""
    pipeline, fuser = build_image_pipeline()
    frames = load_frames(sample, max_frames=max_frames)

    from src.domain.models import SpoofCategory
    per_frame_scores: list[float] = []
    for frame in frames:
        analysis = pipeline.process(frame)
        if not analysis.classifications:
            continue
        # Take the highest-confidence face (largest crop area as proxy)
        best = max(analysis.classifications.values(), key=lambda c: c.confidence or 0.0)
        # P(REAL) — bona-fide probability — taken from the multi-class distribution.
        p_real = float(best.probabilities.get(SpoofCategory.REAL, 0.0))
        per_frame_scores.append(p_real)

    if not per_frame_scores:
        return 0.5  # uncertain — no face detected
    return aggregate_frame_scores(per_frame_scores, mode="mean")
