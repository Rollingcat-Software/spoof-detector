# This file is a copy from biometric-processor/app/domain/entities/liveness_result.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Liveness detection result entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True, init=False)
class LivenessResult:
    """Result of liveness detection operation.

    This is an immutable value object representing liveness check outcome.
    Following Single Responsibility Principle - only contains liveness data.

    Attributes:
        is_live: Whether the subject is determined to be a live person
        score: Decision score (0-100), compared against the detector threshold
        confidence: Normalized confidence score (0-1)
        challenge: Type of challenge used (e.g., "smile", "blink")
        challenge_completed: Whether the challenge was successfully completed
        details: Optional per-signal breakdown for calibration and observability

    Liveness Score Guidelines:
        - 0-50: Likely spoof (reject)
        - 51-80: Uncertain (additional verification recommended)
        - 81-100: Live person (accept)

    Note:
        This class is immutable (frozen) to ensure data integrity.
    """

    is_live: bool
    score: float
    challenge: str
    challenge_completed: bool
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        is_live: bool,
        score: Optional[float] = None,
        challenge: str,
        challenge_completed: bool,
        confidence: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
        liveness_score: Optional[float] = None,
    ) -> None:
        """Initialize result.

        `liveness_score` is accepted as a backward-compatible alias for `score`.
        """
        resolved_score = score if score is not None else liveness_score
        if resolved_score is None:
            raise TypeError("LivenessResult requires 'score'")
        if score is not None and liveness_score is not None and score != liveness_score:
            raise ValueError("score and liveness_score must match when both are provided")

        object.__setattr__(self, "is_live", is_live)
        object.__setattr__(self, "score", float(resolved_score))
        object.__setattr__(self, "challenge", challenge)
        object.__setattr__(self, "challenge_completed", challenge_completed)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "details", details or {})
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate liveness result data."""
        if not 0 <= self.score <= 100:
            raise ValueError(f"Liveness score must be 0-100, got {self.score}")

        if self.confidence == 0.0 and self.score > 0:
            object.__setattr__(self, "confidence", self.score / 100.0)

        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

        if not self.challenge:
            raise ValueError("Challenge type cannot be empty")

    @property
    def liveness_score(self) -> float:
        """Backward-compatible alias for legacy call sites."""
        return self.score

    def get_confidence_level(self) -> str:
        """Get confidence level as string.

        Returns:
            Confidence level: "low", "medium", or "high"
        """
        if self.score < 50:
            return "low"
        elif self.score < 81:
            return "medium"
        else:
            return "high"

    def is_spoof_suspected(self, threshold: float = 50.0) -> bool:
        """Check if spoof attack is suspected.

        Args:
            threshold: Minimum acceptable liveness score

        Returns:
            True if liveness score below threshold
        """
        return self.score < threshold

    def requires_additional_verification(self, threshold: float = 80.0) -> bool:
        """Check if additional verification is recommended.

        Args:
            threshold: Threshold for requiring additional verification

        Returns:
            True if liveness score is uncertain (between 50-80)
        """
        return 50 <= self.score < threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "is_live": self.is_live,
            "score": self.score,
            "liveness_score": self.score,
            "confidence": self.confidence,
            "challenge": self.challenge,
            "challenge_completed": self.challenge_completed,
            "details": self.details,
            "confidence_level": self.get_confidence_level(),
            "spoof_suspected": self.is_spoof_suspected(),
            "requires_additional_verification": self.requires_additional_verification(),
        }
