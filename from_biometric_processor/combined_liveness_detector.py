# This file is a copy from biometric-processor/app/infrastructure/ml/liveness/combined_liveness_detector.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Combined liveness detector using both passive and active detection.

This detector combines texture-based (passive) and facial action (active)
liveness detection for improved accuracy and anti-spoofing protection.
"""

import logging

import numpy as np

from app.domain.entities.liveness_result import LivenessResult
from app.domain.interfaces.liveness_detector import ILivenessDetector
from app.infrastructure.ml.liveness.active_liveness_detector import ActiveLivenessDetector
from app.infrastructure.ml.liveness.texture_liveness_detector import TextureLivenessDetector

logger = logging.getLogger(__name__)


class CombinedLivenessDetector(ILivenessDetector):
    """Liveness detector combining passive and active methods.

    Combines:
    - Texture-based analysis (passive): Detects printed photos, screens
    - Facial action analysis (active): Detects smile, blink

    The combined approach provides:
    - Higher accuracy against spoofing attacks
    - Multiple detection vectors for robustness
    - Weighted scoring from both methods
    """

    def __init__(
        self,
        texture_weight: float = 0.4,
        active_weight: float = 0.6,
        liveness_threshold: float = 65.0,
        texture_threshold: float = 60.0,
        active_threshold: float = 70.0,
    ) -> None:
        """Initialize combined liveness detector.

        Args:
            texture_weight: Weight for texture-based score (0-1)
            active_weight: Weight for active detection score (0-1)
            liveness_threshold: Overall liveness threshold (0-100)
            texture_threshold: Threshold for texture detector
            active_threshold: Threshold for active detector
        """
        if not abs(texture_weight + active_weight - 1.0) < 0.01:
            raise ValueError("Weights must sum to 1.0")

        self._texture_weight = texture_weight
        self._active_weight = active_weight
        self._liveness_threshold = liveness_threshold

        # Initialize sub-detectors
        self._texture_detector = TextureLivenessDetector(
            liveness_threshold=texture_threshold
        )
        self._active_detector = ActiveLivenessDetector(
            liveness_threshold=active_threshold
        )

        logger.info(
            f"CombinedLivenessDetector initialized: "
            f"texture_weight={texture_weight}, active_weight={active_weight}, "
            f"threshold={liveness_threshold}"
        )

    async def check_liveness(self, image: np.ndarray) -> LivenessResult:
        """Check if image shows a live person using combined methods.

        Args:
            image: Face image as numpy array (BGR format)

        Returns:
            LivenessResult with combined liveness determination
        """
        return await self.detect(image)

    async def detect(
        self,
        image: np.ndarray,
        challenge: str = "combined",
    ) -> LivenessResult:
        """Detect liveness using combined passive and active methods.

        Args:
            image: Face image as numpy array (BGR format)
            challenge: Challenge type (default: combined)

        Returns:
            LivenessResult with combined liveness determination
        """
        logger.info("Starting combined liveness detection")

        # Run texture detector (always available)
        texture_result = await self._texture_detector.detect(image)

        # Try active detector, fall back to texture-only if unavailable
        active_result = None
        try:
            active_result = await self._active_detector.detect(image)
        except Exception as e:
            logger.warning(f"Active liveness detection failed (possibly MediaPipe unavailable): {e}")
            logger.info("Falling back to texture-only liveness detection")

        # Calculate score based on available results
        if active_result is not None:
            combined_score = (
                texture_result.score * self._texture_weight +
                active_result.score * self._active_weight
            )
            combined_confidence = (
                texture_result.confidence * self._texture_weight
                + active_result.confidence * self._active_weight
            )
            challenge_completed = (
                texture_result.challenge_completed and
                active_result.challenge_completed
            )
            used_challenge = challenge
            logger.info(
                f"Combined liveness detection complete: "
                f"score={combined_score:.2f}, "
                f"texture_score={texture_result.score:.2f}, "
                f"active_score={active_result.score:.2f}"
            )
            details = {
                **texture_result.details,
                **active_result.details,
                "passive_score": texture_result.score,
                "active_score": active_result.score,
                "combined_score": combined_score,
            }
        else:
            # Texture-only fallback
            combined_score = texture_result.score
            combined_confidence = texture_result.confidence
            challenge_completed = texture_result.challenge_completed
            used_challenge = "texture"
            logger.info(
                f"Texture-only liveness detection complete: "
                f"score={combined_score:.2f}"
            )
            details = {
                **texture_result.details,
                "passive_score": texture_result.score,
                "combined_score": combined_score,
            }

        # Determine liveness
        is_live = combined_score >= self._liveness_threshold

        return LivenessResult(
            is_live=is_live,
            score=combined_score,
            challenge=used_challenge,
            challenge_completed=challenge_completed,
            confidence=combined_confidence,
            details=details,
        )

    def get_challenge_type(self) -> str:
        """Get the type of liveness challenge used.

        Returns:
            Challenge type
        """
        return "combined"

    def get_liveness_threshold(self) -> float:
        """Get the threshold for considering result as live.

        Returns:
            Liveness score threshold (0-100)
        """
        return self._liveness_threshold

    def get_threshold(self) -> float:
        """Get the liveness threshold.

        Returns:
            Current liveness threshold
        """
        return self._liveness_threshold

    def set_threshold(self, threshold: float) -> None:
        """Set the liveness threshold.

        Args:
            threshold: New threshold value (0-100)

        Raises:
            ValueError: If threshold is out of range
        """
        if not 0 <= threshold <= 100:
            raise ValueError(f"Threshold must be between 0 and 100, got {threshold}")
        self._liveness_threshold = threshold
        logger.info(f"Liveness threshold updated to {threshold}")
