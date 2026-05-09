"""Pre-liveness usability gates.

These gates are pixel-and-landmark forensics that decide whether a frame
is even **usable** for downstream liveness/identity scoring. They never
make a "spoof vs live" decision — they only short-circuit obviously
unusable frames (occlusion, bad lighting, no detectable face).

Originally authored by Aysenur (FIVUCSAS R&D, 2026-05) and ported here
2026-05-09 to live next to the rest of the spoof-detection engine.
"""

from src.gates.critical_region_visibility import (
    CriticalRegionVisibilityGate,
    CriticalRegionVisibilityResult,
)
from src.gates.face_usability import FaceUsabilityGate, FaceUsabilityResult
from src.gates.illumination import (
    FaceQualityIlluminationGate,
    FaceQualityIlluminationResult,
)
from src.gates.landmarks import HeadPose, Landmark, LandmarkResult

__all__ = [
    "CriticalRegionVisibilityGate",
    "CriticalRegionVisibilityResult",
    "FaceQualityIlluminationGate",
    "FaceQualityIlluminationResult",
    "FaceUsabilityGate",
    "FaceUsabilityResult",
    "HeadPose",
    "Landmark",
    "LandmarkResult",
]
