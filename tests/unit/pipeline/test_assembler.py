"""End-to-end unit tests for AntispoofPipelineAssembler (PR 5/5).

We mock at the analyzer boundary — face_usability_gate, device_spoof_risk_evaluator,
and hybrid_fusion_evaluator — so these tests don't require real images, OpenCV
detectors, or the ONNX MiniFASNet model. The goal is to pin the *assembler's*
glue logic, not re-test the analyzers (which have their own unit tests in
PRs 1, 3, 4).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.pipeline.assembler import (
    AntispoofPipelineAssembler,
    AntispoofPipelineResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def frame_bgr() -> np.ndarray:
    return np.full((120, 120, 3), 80, dtype=np.uint8)


@pytest.fixture
def landmark_result():
    """Opaque sentinel — the gate is mocked, so we just need a non-None token."""
    return SimpleNamespace(face_present=True)


def _make_usability_gate(usable: bool, reason: str = "") -> Mock:
    gate = Mock()
    gate.evaluate.return_value = SimpleNamespace(
        usable=usable,
        quality_reason=reason,
        physical_occlusion_reason="",
    )
    return gate


def _make_device_evaluator(device_replay_risk: float, **extra: float) -> Mock:
    ev = Mock()
    payload = {
        "moire_risk": 0.1,
        "reflection_risk": 0.1,
        "flicker_risk": 0.0,
        "flash_response_score": 0.0,
        "device_replay_risk": device_replay_risk,
    }
    payload.update(extra)
    ev.evaluate.return_value = SimpleNamespace(
        to_dict=Mock(return_value=payload),
    )
    return ev


def _make_fusion_evaluator(is_spoof: bool, score: float, reasoning: str) -> Mock:
    ev = Mock()
    ev.evaluate.return_value = SimpleNamespace(
        is_spoof=is_spoof,
        spoof_score=score,
        reasoning=reasoning,
    )
    return ev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_allow_when_no_evaluators_configured(frame_bgr) -> None:
    asm = AntispoofPipelineAssembler()
    result = asm.evaluate(frame_bgr=frame_bgr)
    assert isinstance(result, AntispoofPipelineResult)
    assert result.recommended_action == "allow"
    assert result.layers_evaluated == ()
    assert result.face_usability_block is None
    assert result.device_replay_risk is None
    assert result.hybrid_fusion_is_spoof is None


def test_handles_empty_frame_gracefully() -> None:
    asm = AntispoofPipelineAssembler(
        device_spoof_risk_evaluator=_make_device_evaluator(0.05),
    )
    result = asm.evaluate(frame_bgr=np.empty((0, 0, 3), dtype=np.uint8))
    assert result.recommended_action == "allow"
    assert result.layers_evaluated == ()


def test_face_usability_block_short_circuits_to_block(frame_bgr, landmark_result) -> None:
    gate = _make_usability_gate(usable=False, reason="occluded_lower_face")
    device = _make_device_evaluator(0.05)
    fusion = _make_fusion_evaluator(is_spoof=False, score=0.10, reasoning="LIVE")

    asm = AntispoofPipelineAssembler(
        face_usability_gate=gate,
        device_spoof_risk_evaluator=device,
        hybrid_fusion_evaluator=fusion,
    )
    result = asm.evaluate(
        frame_bgr=frame_bgr,
        landmark_result=landmark_result,
        pretrained_spoof_score=0.10,
    )
    assert result.face_usability_block is True
    assert result.face_usability_reason == "occluded_lower_face"
    assert result.recommended_action == "block"
    assert "face_usability" in result.layers_evaluated


def test_hybrid_fusion_spoof_drives_block(frame_bgr) -> None:
    """No usability gate; fusion verdict alone can block."""
    device = _make_device_evaluator(0.7, moire_risk=0.6, flicker_risk=0.4)
    fusion = _make_fusion_evaluator(
        is_spoof=True, score=0.82, reasoning="SPOOF detected"
    )
    asm = AntispoofPipelineAssembler(
        device_spoof_risk_evaluator=device,
        hybrid_fusion_evaluator=fusion,
    )
    result = asm.evaluate(frame_bgr=frame_bgr, pretrained_spoof_score=0.85)
    assert result.hybrid_fusion_is_spoof is True
    assert result.hybrid_fusion_score == pytest.approx(0.82)
    assert result.recommended_action == "block"
    assert "device_spoof_risk" in result.layers_evaluated
    assert "hybrid_fusion" in result.layers_evaluated


def test_high_device_risk_alone_recommends_review(frame_bgr) -> None:
    """Device risk > 0.65 with a passing fusion verdict → review (not block)."""
    device = _make_device_evaluator(0.72)
    fusion = _make_fusion_evaluator(
        is_spoof=False, score=0.45, reasoning="LIVE verified"
    )
    asm = AntispoofPipelineAssembler(
        device_spoof_risk_evaluator=device,
        hybrid_fusion_evaluator=fusion,
    )
    result = asm.evaluate(frame_bgr=frame_bgr, pretrained_spoof_score=0.20)
    assert result.recommended_action == "review"
    assert result.device_replay_risk == pytest.approx(0.72)
    assert result.hybrid_fusion_is_spoof is False


def test_device_evaluator_failure_is_swallowed(frame_bgr) -> None:
    device = Mock()
    device.evaluate.side_effect = RuntimeError("boom")
    fusion = _make_fusion_evaluator(is_spoof=False, score=0.3, reasoning="LIVE")
    asm = AntispoofPipelineAssembler(
        device_spoof_risk_evaluator=device,
        hybrid_fusion_evaluator=fusion,
    )
    result = asm.evaluate(frame_bgr=frame_bgr, pretrained_spoof_score=0.20)
    # Device layer skipped, fusion not run (no device_signals).
    assert result.device_replay_risk is None
    assert result.hybrid_fusion_is_spoof is None
    assert result.recommended_action == "allow"
    assert "device_spoof_risk" not in result.layers_evaluated


def test_cutout_enabled_appends_observability_token(frame_bgr) -> None:
    device = _make_device_evaluator(0.20)
    asm = AntispoofPipelineAssembler(device_spoof_risk_evaluator=device)
    result = asm.evaluate(frame_bgr=frame_bgr, cutout_enabled=True)
    assert "cutout_anomaly_forced" in result.layers_evaluated


def test_pretrained_score_provider_is_used_when_score_omitted(frame_bgr) -> None:
    device = _make_device_evaluator(0.30, moire_risk=0.40)
    fusion = _make_fusion_evaluator(is_spoof=True, score=0.71, reasoning="SPOOF")
    provider = Mock(return_value=0.91)

    asm = AntispoofPipelineAssembler(
        device_spoof_risk_evaluator=device,
        hybrid_fusion_evaluator=fusion,
        pretrained_spoof_score_provider=provider,
    )
    result = asm.evaluate(frame_bgr=frame_bgr)  # no pretrained_spoof_score
    provider.assert_called_once()
    assert result.hybrid_fusion_is_spoof is True
    assert result.recommended_action == "block"


def test_to_dict_is_jsonable(frame_bgr) -> None:
    """to_dict must produce only built-in types so callers can serialize."""
    device = _make_device_evaluator(0.30)
    fusion = _make_fusion_evaluator(
        is_spoof=False, score=0.30, reasoning="LIVE verified"
    )
    asm = AntispoofPipelineAssembler(
        device_spoof_risk_evaluator=device,
        hybrid_fusion_evaluator=fusion,
    )
    result = asm.evaluate(frame_bgr=frame_bgr, pretrained_spoof_score=0.20)

    import json

    blob = json.dumps(result.to_dict())
    assert "device_replay_risk" in blob
    assert "hybrid_fusion_score" in blob
    assert "layers_evaluated" in blob
