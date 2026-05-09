# This file is a copy from biometric-processor/app/domain/entities/liveness_report.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Liveness detection report domain entities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class ComponentAnalysis:
    """Analysis of single liveness component.

    Attributes:
        score: Component score (0-100)
        weight: Component weight in overall score
        details: Detailed analysis data
        verdict: Human-readable verdict
    """

    score: float
    weight: float
    details: Dict[str, Any]
    verdict: str


@dataclass
class PassiveAnalysis:
    """Passive liveness analysis result.

    Attributes:
        score: Overall passive score
        passed: Whether passive check passed
        texture: Texture analysis component
        color: Color distribution component
        frequency: Frequency analysis component
        moire: Moire pattern detection component
    """

    score: float
    passed: bool
    texture: ComponentAnalysis
    color: ComponentAnalysis
    frequency: ComponentAnalysis
    moire: ComponentAnalysis


@dataclass
class ActiveAnalysis:
    """Active liveness analysis result.

    Attributes:
        score: Overall active score
        passed: Whether active check passed
        eye_analysis: Eye analysis component
        mouth_analysis: Mouth analysis component
    """

    score: float
    passed: bool
    eye_analysis: ComponentAnalysis
    mouth_analysis: ComponentAnalysis


@dataclass
class LivenessReport:
    """Complete liveness detection report.

    Attributes:
        is_live: Final liveness determination
        overall_score: Combined liveness score
        threshold: Score threshold used
        detection_mode: Mode used (passive/active/combined)
        passive: Passive analysis results
        active: Active analysis results
        recommendations: Suggestions for improvement
        risk_level: Risk assessment (low/medium/high)
    """

    is_live: bool
    overall_score: float
    threshold: float
    detection_mode: str
    passive: Optional[PassiveAnalysis] = None
    active: Optional[ActiveAnalysis] = None
    recommendations: List[str] = field(default_factory=list)
    risk_level: str = "low"


# Pydantic models for API responses


class ComponentAnalysisResponse(BaseModel):
    """API response model for component analysis."""

    score: float
    weight: float
    details: Dict[str, Any]
    verdict: str


class PassiveAnalysisResponse(BaseModel):
    """API response model for passive analysis."""

    score: float
    passed: bool
    components: Dict[str, ComponentAnalysisResponse]


class ActiveAnalysisResponse(BaseModel):
    """API response model for active analysis."""

    score: float
    passed: bool
    components: Dict[str, ComponentAnalysisResponse]


class LivenessReportResponse(BaseModel):
    """API response model for liveness report."""

    is_live: bool
    overall_score: float
    threshold: float
    detection_mode: str
    analysis: Dict[str, Any]
    recommendations: List[str]
    risk_level: str

    @classmethod
    def from_result(cls, result: LivenessReport) -> "LivenessReportResponse":
        """Create response from domain result."""
        analysis = {}

        if result.passive:
            analysis["passive"] = {
                "score": result.passive.score,
                "passed": result.passive.passed,
                "components": {
                    "texture_analysis": {
                        "score": result.passive.texture.score,
                        "weight": result.passive.texture.weight,
                        "details": result.passive.texture.details,
                        "verdict": result.passive.texture.verdict,
                    },
                    "color_distribution": {
                        "score": result.passive.color.score,
                        "weight": result.passive.color.weight,
                        "details": result.passive.color.details,
                        "verdict": result.passive.color.verdict,
                    },
                    "frequency_analysis": {
                        "score": result.passive.frequency.score,
                        "weight": result.passive.frequency.weight,
                        "details": result.passive.frequency.details,
                        "verdict": result.passive.frequency.verdict,
                    },
                    "moire_detection": {
                        "score": result.passive.moire.score,
                        "weight": result.passive.moire.weight,
                        "details": result.passive.moire.details,
                        "verdict": result.passive.moire.verdict,
                    },
                },
            }

        if result.active:
            analysis["active"] = {
                "score": result.active.score,
                "passed": result.active.passed,
                "components": {
                    "eye_analysis": {
                        "score": result.active.eye_analysis.score,
                        "details": result.active.eye_analysis.details,
                    },
                    "mouth_analysis": {
                        "score": result.active.mouth_analysis.score,
                        "details": result.active.mouth_analysis.details,
                    },
                },
            }

        return cls(
            is_live=result.is_live,
            overall_score=result.overall_score,
            threshold=result.threshold,
            detection_mode=result.detection_mode,
            analysis=analysis,
            recommendations=result.recommendations,
            risk_level=result.risk_level,
        )
