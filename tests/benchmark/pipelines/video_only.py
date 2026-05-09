"""Video-only pipeline (Aysenur's contribution).

For each sample (video required — image samples scored 0.5 = uncertain):
  1. Decode 30 evenly-spaced frames.
  2. Run blink, rPPG, screen-replay, micro-tremor, screen-flicker,
     landmark-variance, temporal analyzers.
  3. Aggregate via peak-sensitive verdict (Aysenur's session model).
"""
from __future__ import annotations

import logging
import numpy as np

from tests.benchmark.runner import Sample
from tests.benchmark.pipelines._common import (
    load_frames,
    build_video_pipeline,
    aggregate_frame_scores,
)

logger = logging.getLogger(__name__)


def score_sample(sample: Sample, *, max_frames: int = 30) -> float:
    pipeline, _fuser = build_video_pipeline()
    frames = load_frames(sample, max_frames=max_frames)
    if len(frames) < 2:
        # Video-only pipeline needs temporal context.
        return 0.5

    per_frame_scores: list[float] = []
    for frame in frames:
        analysis = pipeline.process(frame)
        if not analysis.classifications:
            continue
        best = max(analysis.classifications.values(), key=lambda c: c.confidence or 0.0)
        per_frame_scores.append(float(best.p_real))

    if not per_frame_scores:
        return 0.5
    return aggregate_frame_scores(per_frame_scores, mode="peak_sensitive")
