"""Public re-export of ``src.infrastructure.analyzers.blink_analyzer``.

This module exposes the BlinkAnalyzer + EAR helpers (calibrated 2026-05-11
in the perf/blink-cache-and-ear-calibration branch) under the stable
``spoof_detector.*`` namespace, so downstream consumers (e.g. FIVUCSAS
biometric-processor) don't reach into the unstable ``src.*`` import path.

Re-exports:
    compute_ear         — Eye Aspect Ratio from 6-point landmark indices
    BlinkAnalyzer       — multi-frame blink detector with per-frame cache
    BlinkState          — per-face tracking state dataclass
    LEFT_EYE, RIGHT_EYE — MediaPipe FaceMesh eye landmark index lists
"""

# pylint: disable=wildcard-import
from src.infrastructure.analyzers.blink_analyzer import (  # noqa: F401
    BlinkAnalyzer,
    BlinkState,
    LEFT_EYE,
    RIGHT_EYE,
    compute_ear,
)
