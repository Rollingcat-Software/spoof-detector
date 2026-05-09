"""Anti-spoof pipeline assembler (Aysenur cherry-pick 5/5).

Composes three independent anti-spoof modules — already merged via
PRs 1, 3, 4 of the same cherry-pick chain — into a single feature-flag-gated
adapter that callers (e.g. /verify) can opt into:

    1. ``FaceUsabilityGate`` (PR 1)        — pre-liveness frame quality + occlusion
    2. ``DeviceSpoofRiskEvaluator`` (PR 4) — device replay / moire / flash / cutout signals
    3. ``HybridFusionEvaluator`` (PR 3)    — fuses pretrained MiniFASNet score + device signals

Each component is gated by its own env flag (``ANTISPOOF_USABILITY_GATE_ENABLED``,
``ANTISPOOF_DEVICE_RISK_ENABLED``, ``ANTISPOOF_FUSION_ENABLED``,
``ANTISPOOF_CUTOUT_ENABLED``) so an operator can roll out the layers in any
order. **All flags default OFF**; this module is invisible to prod until
explicitly opted into.

The module is purely additive: it returns a structured ``AntispoofPipelineResult``
that callers attach to their response payload but never blocks on (the result's
``recommended_action`` is informational only). Hard-blocking is reserved for
a separate, future PR after the signals have been observed in shadow mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AntispoofPipelineResult:
    """Combined output of the three anti-spoof layers.

    All fields are nullable; a ``None`` value means that layer was either
    disabled by its flag or its evaluator raised an exception (which is
    swallowed — failures must never break verification).
    """

    face_usability_block: Optional[bool] = None
    """True iff face_usability_gate produced a blocking verdict."""

    face_usability_reason: Optional[str] = None
    """Human-readable reason for the block (e.g. 'occluded', 'no_face')."""

    device_replay_risk: Optional[float] = None
    """[0, 1] device-replay risk produced by DeviceSpoofRiskEvaluator."""

    device_signals: Optional[dict] = None
    """Full device-spoof breakdown (moire/reflection/flicker/flash/cutout/...)."""

    hybrid_fusion_is_spoof: Optional[bool] = None
    """Verdict from HybridFusionEvaluator combining pretrained+device signals."""

    hybrid_fusion_score: Optional[float] = None
    """[0, 1] spoof score from HybridFusionEvaluator."""

    hybrid_fusion_reasoning: Optional[str] = None

    recommended_action: str = "allow"
    """Soft recommendation: 'allow' | 'review' | 'block'. Caller decides."""

    layers_evaluated: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["layers_evaluated"] = list(self.layers_evaluated)
        return d


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


class AntispoofPipelineAssembler:
    """Run the configured layers and assemble a single result.

    Constructor takes the three evaluators as kwargs so they can be lazily
    injected (avoids paying detector-construction cost when all flags are off).
    Each evaluator may be ``None``; the corresponding layer is skipped.
    """

    def __init__(
        self,
        *,
        face_usability_gate: Any | None = None,
        device_spoof_risk_evaluator: Any | None = None,
        hybrid_fusion_evaluator: Any | None = None,
        pretrained_spoof_score_provider: Any | None = None,
    ) -> None:
        self._face_usability_gate = face_usability_gate
        self._device_spoof_risk_evaluator = device_spoof_risk_evaluator
        self._hybrid_fusion_evaluator = hybrid_fusion_evaluator
        self._pretrained_spoof_score_provider = pretrained_spoof_score_provider

    # -- public ----------------------------------------------------------

    def evaluate(
        self,
        *,
        frame_bgr: np.ndarray,
        landmark_result: Any | None = None,
        face_bounding_box: Optional[tuple[int, int, int, int]] = None,
        pretrained_spoof_score: Optional[float] = None,
        cutout_enabled: bool = False,
    ) -> AntispoofPipelineResult:
        """Run every layer the assembler was constructed with.

        Parameters
        ----------
        frame_bgr
            BGR frame from the request. Must be a non-empty ``np.ndarray``.
        landmark_result
            Optional ``LandmarkResult`` used by the face-usability gate.
            If absent and the gate is configured, the gate is skipped.
        face_bounding_box
            Optional (x, y, w, h) tuple in pixel coords; threaded through to
            DeviceSpoofRiskEvaluator for the screen-frame heuristic.
        pretrained_spoof_score
            Optional pretrained-model spoof probability in [0, 1]. When
            absent, the assembler will pull it from
            ``pretrained_spoof_score_provider`` if one was injected.
        cutout_enabled
            When True, forces the cutout/focal-blur anomaly score to be
            included in the device-replay fusion (instructional override
            for the ``ANTISPOOF_CUTOUT_ENABLED`` flag — DeviceSpoofRisk
            already runs the detector internally, so toggling this is a
            no-op for the underlying maths but is propagated to the
            ``layers_evaluated`` tuple for observability).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return AntispoofPipelineResult(recommended_action="allow")

        layers: list[str] = []
        face_block: Optional[bool] = None
        face_reason: Optional[str] = None
        device_risk: Optional[float] = None
        device_signals: Optional[dict] = None
        fusion_is_spoof: Optional[bool] = None
        fusion_score: Optional[float] = None
        fusion_reasoning: Optional[str] = None

        # -- Layer 1: face usability gate ---------------------------------
        if self._face_usability_gate is not None and landmark_result is not None:
            try:
                gate_result = self._face_usability_gate.evaluate(
                    frame_bgr=frame_bgr,
                    landmark_result=landmark_result,
                )
                face_block = bool(getattr(gate_result, "usable", True) is False)
                if face_block:
                    face_reason = (
                        getattr(gate_result, "quality_reason", None)
                        or getattr(gate_result, "physical_occlusion_reason", None)
                        or "unusable_face"
                    )
                layers.append("face_usability")
            except Exception:  # noqa: BLE001 — fail-soft
                pass

        # -- Layer 2: device-spoof risk evaluator -------------------------
        if self._device_spoof_risk_evaluator is not None:
            try:
                assessment = self._device_spoof_risk_evaluator.evaluate(
                    frame_bgr=frame_bgr,
                    face_bounding_box=face_bounding_box,
                )
                device_signals = assessment.to_dict()
                device_risk = float(device_signals.get("device_replay_risk", 0.0))
                layers.append("device_spoof_risk")
                if cutout_enabled:
                    layers.append("cutout_anomaly_forced")
            except Exception:  # noqa: BLE001 — fail-soft
                pass

        # -- Layer 3: hybrid fusion evaluator -----------------------------
        if self._hybrid_fusion_evaluator is not None:
            score = pretrained_spoof_score
            if score is None and self._pretrained_spoof_score_provider is not None:
                try:
                    score = float(
                        self._pretrained_spoof_score_provider(frame_bgr=frame_bgr)
                    )
                except Exception:  # noqa: BLE001
                    score = None

            if score is not None and device_signals is not None:
                try:
                    custom_signals = {
                        "flicker_score": float(device_signals.get("flicker_risk", 0.0)),
                        "flash_response_score": float(
                            device_signals.get("flash_response_score", 0.0) or 0.0
                        ),
                        "flash_response_samples": 1,  # we only have one frame here
                        "moire_score": float(device_signals.get("moire_risk", 0.0)),
                        "device_replay_score": float(device_risk or 0.0),
                    }
                    fusion = self._hybrid_fusion_evaluator.evaluate(
                        pretrained_spoof_score=float(score),
                        custom_signals=custom_signals,
                    )
                    fusion_is_spoof = bool(fusion.is_spoof)
                    fusion_score = float(fusion.spoof_score)
                    fusion_reasoning = str(fusion.reasoning)
                    layers.append("hybrid_fusion")
                except Exception:  # noqa: BLE001
                    pass

        # -- Recommended action -------------------------------------------
        action = self._recommend(
            face_block=face_block,
            device_risk=device_risk,
            fusion_is_spoof=fusion_is_spoof,
        )

        return AntispoofPipelineResult(
            face_usability_block=face_block,
            face_usability_reason=face_reason,
            device_replay_risk=device_risk,
            device_signals=device_signals,
            hybrid_fusion_is_spoof=fusion_is_spoof,
            hybrid_fusion_score=fusion_score,
            hybrid_fusion_reasoning=fusion_reasoning,
            recommended_action=action,
            layers_evaluated=tuple(layers),
        )

    # -- private ---------------------------------------------------------

    @staticmethod
    def _recommend(
        *,
        face_block: Optional[bool],
        device_risk: Optional[float],
        fusion_is_spoof: Optional[bool],
    ) -> str:
        """Soft recommendation. Caller decides whether to enforce."""
        if face_block is True:
            return "block"
        if fusion_is_spoof is True:
            return "block"
        if device_risk is not None and device_risk >= 0.65:
            return "review"
        return "allow"
