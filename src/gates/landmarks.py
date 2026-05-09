"""Lightweight landmark dataclasses.

Mirrors the dataclass-only subset of the FIVUCSAS biometric-processor
`app.domain.entities.face_landmarks` module so the gates in this package
can be imported and tested without pulling in a Pydantic dependency.

These types are intentionally minimal — only the fields read by the gates
are reproduced. The owning service (FIVUCSAS biometric-processor) is free
to subclass or duck-type a richer landmark object as long as the
attributes used here are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Landmark:
    """Single facial landmark point.

    Attributes:
        id: Landmark index.
        x: X coordinate (pixels, frame-space).
        y: Y coordinate (pixels, frame-space).
        z: Z coordinate (optional, for 3D landmarks).
    """

    id: int
    x: int
    y: int
    z: Optional[float] = None


@dataclass
class HeadPose:
    """Head pose estimation in degrees.

    Attributes:
        pitch: Rotation around X-axis (up/down).
        yaw: Rotation around Y-axis (left/right).
        roll: Rotation around Z-axis (tilt).
    """

    pitch: float
    yaw: float
    roll: float


@dataclass
class LandmarkResult:
    """Face landmark detection result.

    Attributes:
        model: Name of the detector that produced these landmarks.
        landmark_count: Number of landmarks detected.
        landmarks: List of landmark points.
        regions: Optional region → list-of-landmark-indices mapping.
        head_pose: Optional head pose estimate.
    """

    model: str
    landmark_count: int
    landmarks: List[Landmark] = field(default_factory=list)
    regions: Dict[str, List[int]] = field(default_factory=dict)
    head_pose: Optional[HeadPose] = None
