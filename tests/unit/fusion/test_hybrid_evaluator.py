"""Unit tests for hybrid liveness fusion (PR 3/5 of Aysenur cherry-pick).

The downstream `LiveCameraAnalysisUseCase` integration test from Aysenur's
branch is intentionally **deferred to PR 4/5** of this cherry-pick chain —
it depends on `LiveCameraAnalysisUseCase.__init__` accepting `settings` and
`hybrid_fusion_evaluator` kwargs, which are part of the wiring PRs.
"""

import pytest

from src.fusion.hybrid_evaluator import (
    HybridFusionEvaluator,
    FusionWeights,
)


def test_hybrid_fusion_evaluator_detects_clear_spoof_case() -> None:
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.95,
        custom_signals={
            "flash_response_score": 0.0,
            "flash_response_samples": 2,
            "rppg_live_signal": False,
            "rppg_available": True,
            "moire_score": 0.9,
            "device_replay_score": 0.8,
        },
    )

    assert result.is_spoof is True
    assert result.spoof_score > 0.8
    assert "SPOOF detected" in result.reasoning


def test_hybrid_fusion_evaluator_detects_clear_live_case() -> None:
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.05,
        custom_signals={
            "flash_response_score": 0.95,
            "flash_response_samples": 2,
            "rppg_live_signal": True,
            "rppg_available": True,
            "moire_score": 0.1,
            "device_replay_score": 0.15,
        },
    )

    assert result.is_spoof is False
    assert result.spoof_score < 0.3
    assert "LIVE verified" in result.reasoning


def test_hybrid_fusion_evaluator_flicker_override_forces_spoof() -> None:
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.05,
        custom_signals={
            "flicker_score": 0.91,
            "flash_response_score": 0.95,
            "flash_response_samples": 2,
            "moire_score": 0.05,
            "device_replay_score": 0.10,
        },
    )

    assert result.is_spoof is True
    assert result.spoof_score == pytest.approx(0.90)
    assert result.confidence == pytest.approx(0.90)
    assert result.breakdown["flicker"] == pytest.approx(0.91)
    assert result.reasoning == "Very high flicker detected (0.91)"


def test_hybrid_fusion_evaluator_flicker_below_override_uses_normal_fusion() -> None:
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.05,
        custom_signals={
            "flicker_score": 0.65,
            "flash_response_score": 0.95,
            "flash_response_samples": 2,
            "moire_score": 0.10,
            "device_replay_score": 0.15,
        },
    )

    assert result.is_spoof is False
    assert result.spoof_score < 0.30
    assert "LIVE verified" in result.reasoning


def test_hybrid_fusion_evaluator_weak_minifasnet_neutral_signals_remains_consistent() -> None:
    """Edge case: weak MiniFASNet (0.65) with neutral/missing custom signals.

    With the current 4-weight scheme (0.30/0.30/0.20/0.20) the fused score
    sits very close to the default threshold (0.5). This test pins the
    *behaviour* — that the evaluator returns a consistent verdict and a
    bounded score in the uncertainty zone — rather than a specific decimal,
    so it survives weight tuning.
    """
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.65,
        custom_signals={
            "flash_response_samples": 0,
            "flash_response_score": None,
            "rppg_available": False,
            "rppg_live_signal": None,
            "moire_score": 0.30,
            "device_replay_score": 0.25,
        },
    )

    # Score should be in the uncertainty band, not extreme either way.
    assert 0.30 < result.spoof_score < 0.55
    # Whatever the verdict, the reasoning string must be coherent.
    assert "score=" in result.reasoning


def test_hybrid_fusion_evaluator_threshold_boundary_low_side_is_live() -> None:
    """Mixed-but-mostly-live signals stay below threshold and report LIVE."""
    evaluator = HybridFusionEvaluator(threshold=0.55)

    result = evaluator.evaluate(
        pretrained_spoof_score=0.40,
        custom_signals={
            "flash_response_score": 0.70,
            "flash_response_samples": 2,
            "rppg_live_signal": True,
            "rppg_available": True,
            "moire_score": 0.30,
            "device_replay_score": 0.30,
        },
    )

    assert result.is_spoof is False
    assert result.spoof_score < 0.55
    assert "LIVE" in result.reasoning


def test_hybrid_fusion_evaluator_threshold_boundary_high_side_is_spoof() -> None:
    """Multiple spoof indicators above threshold trigger SPOOF verdict."""
    evaluator = HybridFusionEvaluator(threshold=0.55)

    result = evaluator.evaluate(
        pretrained_spoof_score=0.70,
        custom_signals={
            "flash_response_score": 0.20,
            "flash_response_samples": 2,
            "rppg_live_signal": False,
            "rppg_available": True,
            "moire_score": 0.70,
            "device_replay_score": 0.65,
        },
    )

    assert result.is_spoof is True
    assert result.spoof_score >= 0.55
    assert "SPOOF detected" in result.reasoning


def test_hybrid_fusion_evaluator_spoof_with_one_strong_live_signal() -> None:
    """Adversarial case: Spoof attack with one strong contradictory signal.

    If a spoof video manages to trigger a false rPPG signal, fusion should
    still catch it via other indicators.
    """
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.90,  # Clear spoof signature
        custom_signals={
            "flash_response_score": 0.05,  # No flash response
            "flash_response_samples": 2,
            "rppg_live_signal": True,  # FALSE POSITIVE from rPPG (video artifact)
            "rppg_available": True,
            "moire_score": 0.88,  # Clear moire from screen
            "device_replay_score": 0.85,  # Clear replay from screen
        },
    )

    # Multiple strong spoof indicators should overcome single rPPG false positive
    assert result.is_spoof is True, (
        f"Should detect spoof despite rPPG false positive. "
        f"Score: {result.spoof_score:.3f}"
    )
    assert result.spoof_score > 0.65


def test_hybrid_fusion_evaluator_missing_all_custom_signals() -> None:
    """Fallback case: What if all custom signals are unavailable?

    Should fall back to MiniFASNet with neutral padding.
    """
    evaluator = HybridFusionEvaluator()

    result = evaluator.evaluate(
        pretrained_spoof_score=0.80,  # Strong spoof from MiniFASNet
        custom_signals={
            # Everything missing/null
            "flash_response_score": None,
            "flash_response_samples": None,
            "rppg_available": False,
            "rppg_live_signal": None,
            "moire_score": None,
            "device_replay_score": None,
        },
    )

    # With all custom signals neutral (0.5), should still detect spoof from MiniFASNet
    # Score = 0.25*0.80 + 0.75*0.5 = 0.20 + 0.375 = 0.575 > 0.55
    assert result.is_spoof is True
    assert result.spoof_score > 0.55
    assert "backend_score" in result.breakdown or "pretrained" in result.breakdown


def test_hybrid_fusion_evaluator_weights_normalize_correctly() -> None:
    """Verify that FusionWeights properly normalize."""
    weights = FusionWeights(
        pretrained_model=0.30,
        flash_response=0.30,
        moire_pattern=0.20,
        device_replay=0.20,
    )

    assert abs(sum([
        weights.pretrained_model,
        weights.flash_response,
        weights.moire_pattern,
        weights.device_replay,
    ]) - 1.0) < 1e-6


def test_hybrid_fusion_evaluator_invalid_weights_raise() -> None:
    """Verify that invalid weights are caught."""
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        FusionWeights(
            pretrained_model=0.5,  # Sum will be > 1.0
            flash_response=0.5,
            moire_pattern=0.1,
            device_replay=0.1,
        )


# NOTE: Aysenur's branch also asserted that LiveCameraAnalysisUseCase wires the
# hybrid evaluator behind LIVENESS_FUSION_ENABLED. That integration test is
# intentionally deferred to PRs 4 and 5 of this cherry-pick chain (which add
# the wiring + the env-flag gate). See the PR description.
