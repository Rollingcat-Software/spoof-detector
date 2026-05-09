# This file is a copy from biometric-processor/app/domain/exceptions/liveness_errors.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Liveness detection related errors."""

from app.domain.exceptions.base import BiometricProcessorError


class LivenessCheckFailedError(BiometricProcessorError):
    """Raised when liveness check fails (spoofing detected or low score).

    This indicates:
    - Possible spoofing attack (photo, video, mask)
    - Insufficient liveness indicators
    - Challenge not completed properly
    """

    def __init__(
        self, liveness_score: float, min_threshold: float = 80.0, challenge: str = "unknown"
    ) -> None:
        """Initialize liveness check failed error.

        Args:
            liveness_score: Actual liveness score (0-100)
            min_threshold: Minimum acceptable liveness score
            challenge: Type of liveness challenge used
        """
        super().__init__(
            message=(
                f"Liveness check failed (score: {liveness_score:.0f}/100, "
                f"minimum: {min_threshold:.0f}). Challenge: {challenge}. "
                "Please ensure you're a live person performing the requested action."
            ),
            error_code="LIVENESS_CHECK_FAILED",
        )
        self.liveness_score = liveness_score
        self.min_threshold = min_threshold
        self.challenge = challenge

    def to_dict(self) -> dict:
        """Include liveness details in error response."""
        result = super().to_dict()
        result["liveness_score"] = self.liveness_score
        result["min_threshold"] = self.min_threshold
        result["challenge"] = self.challenge
        return result


class LivenessCheckError(BiometricProcessorError):
    """Raised when liveness check encounters an error during processing.

    This is different from LivenessCheckFailedError - it indicates
    a technical error rather than a failed liveness test.
    """

    def __init__(self, reason: str = "Unknown error") -> None:
        """Initialize liveness check error.

        Args:
            reason: Detailed reason for error
        """
        super().__init__(
            message=f"Liveness check error: {reason}",
            error_code="LIVENESS_CHECK_ERROR",
        )
        self.reason = reason

    def to_dict(self) -> dict:
        """Include failure reason in error response."""
        result = super().to_dict()
        result["reason"] = self.reason
        return result
