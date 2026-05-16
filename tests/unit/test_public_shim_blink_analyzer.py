"""Pin the public ``spoof_detector.infrastructure.analyzers.blink_analyzer``
re-export added 2026-05-12.

Downstream consumers (FIVUCSAS biometric-processor /verify wiring) import
``compute_ear`` from this path so they don't reach into ``src.*``. Removing
the shim would silently break those callers — this test catches that.
"""

from __future__ import annotations

import numpy as np


def test_public_blink_analyzer_exports_resolve():
    """The public path must expose the same objects as ``src.*``."""
    from spoof_detector.infrastructure.analyzers.blink_analyzer import (
        BlinkAnalyzer,
        LEFT_EYE,
        RIGHT_EYE,
        compute_ear,
    )

    assert callable(compute_ear)
    assert isinstance(LEFT_EYE, list) and len(LEFT_EYE) == 6
    assert isinstance(RIGHT_EYE, list) and len(RIGHT_EYE) == 6
    assert BlinkAnalyzer.EAR_THRESHOLD == 0.18


def test_public_compute_ear_matches_src_implementation():
    """Public alias must point at the SAME function object as ``src``."""
    from spoof_detector.infrastructure.analyzers.blink_analyzer import compute_ear as public
    from src.infrastructure.analyzers.blink_analyzer import compute_ear as private

    assert public is private


def test_public_compute_ear_round_trip_on_open_eye_landmark():
    """Smoke: a synthetic 'open eye' rectangle gives a sensible EAR."""
    from spoof_detector.infrastructure.analyzers.blink_analyzer import (
        LEFT_EYE,
        compute_ear,
    )

    # Build a 478-point array so LEFT_EYE indices resolve. We only need
    # the 6 LEFT_EYE indices populated with a visibly-open rectangle.
    lm = np.zeros((478, 3), dtype=np.float64)
    p1, p2, p3, p4, p5, p6 = LEFT_EYE
    lm[p1] = [0.0, 0.5, 0.0]   # left corner
    lm[p4] = [1.0, 0.5, 0.0]   # right corner (horizontal width = 1.0)
    lm[p2] = [0.3, 0.2, 0.0]   # top inner
    lm[p3] = [0.7, 0.2, 0.0]   # top outer
    lm[p5] = [0.7, 0.8, 0.0]   # bottom outer
    lm[p6] = [0.3, 0.8, 0.0]   # bottom inner

    ear = compute_ear(lm, LEFT_EYE)
    # Vertical distances = 0.6 each, horizontal = 1.0 → EAR = (0.6 + 0.6) / 2 = 0.6
    assert 0.55 < ear < 0.65
