# This file is a copy from biometric-processor/app/infrastructure/ml/liveness/stub_liveness_detector.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Stub liveness detector implementation (to be replaced in Sprint 3)."""

import logging

import numpy as np

from app.domain.entities.liveness_result import LivenessResult

logger = logging.getLogger(__name__)


class StubLivenessDetector:
    """Stub liveness detector for MVP.

    This is a temporary implementation that always returns a passing liveness score.
    Will be replaced with actual smile/blink detection in Sprint 3.

    Following Open/Closed Principle: Can be swapped with real implementation
    without changing client code.
    """

    def __init__(self, default_score: float = 85.0) -> None:
        """Initialize stub liveness detector.

        Args:
            default_score: Default liveness score to return (0-100)
        """
        self._default_score = default_score

        logger.warning(
            "Using StubLivenessDetector - always returns passing score. "
            "Replace with real implementation in Sprint 3!"
        )

    async def check_liveness(self, image: np.ndarray) -> LivenessResult:
        """Stub liveness check - always returns passing score.

        Args:
            image: Input image (not actually used in stub)

        Returns:
            LivenessResult with passing score

        Note:
            This is a STUB implementation for MVP.
            Actual liveness detection will be implemented in Sprint 3.
        """
        logger.debug(f"Stub liveness check - returning default score {self._default_score}")

        return LivenessResult(
            is_live=True,
            liveness_score=self._default_score,
            challenge="none",
            challenge_completed=True,
            details={"backend_score": self._default_score / 100.0, "stub": True},
        )

    def get_challenge_type(self) -> str:
        """Get the type of liveness challenge.

        Returns:
            Challenge type (stub)
        """
        return "none"

    def get_liveness_threshold(self) -> float:
        """Get the liveness threshold.

        Returns:
            Threshold (80.0)
        """
        return 80.0
