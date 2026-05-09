# This file is a copy from biometric-processor/app/infrastructure/ml/liveness/uniface_liveness_detector.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""UniFace MiniFASNet-based liveness detector.

This detector uses the UniFace library's MiniFASNet ONNX model for
anti-spoofing / liveness detection. MiniFASNet is a lightweight face
anti-spoofing network that classifies face images as real or spoofed.

Key advantages over texture-based heuristics:
- Deep learning model trained specifically for anti-spoofing
- ONNX runtime inference (fast, no TensorFlow/PyTorch dependency)
- Handles print attacks, screen replay, and mask attacks
- Lightweight enough for real-time use (~10ms per inference)

Follows the same ILivenessDetector protocol as other implementations
(Liskov Substitution Principle / Open-Closed Principle).
"""

import asyncio
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np

from app.domain.entities.liveness_result import LivenessResult
from app.domain.exceptions.liveness_errors import LivenessCheckError
from app.domain.interfaces.liveness_detector import ILivenessDetector

if TYPE_CHECKING:
    # Type-only import: keeps graceful-degradation behavior at runtime
    # (uniface may not be installed) while letting static analysers see the
    # real .predict() signature on self._model.
    from uniface.spoofing import MiniFASNet

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_shared_minifasnet() -> "MiniFASNet":
    """Process-wide cached MiniFASNet ONNX session.

    Copilot post-merge round 8 (PR #64): `get_liveness_detector()` is
    intentionally NOT cached (the enhanced/hybrid detectors keep per-session
    blink state which must not leak across requests). That meant the
    startup warm-up loaded MiniFASNet onto a throw-away
    `UniFaceLivenessDetector` and request handlers still constructed fresh
    detectors paying the first-call ONNX session-init cost — the warm-up
    was effectively dead code.

    Caching the underlying MiniFASNet at module level (not the detector
    instance) preserves blink-state isolation while ensuring the heavy
    ONNX session is initialised exactly once per process. Every later
    `UniFaceLivenessDetector` instance — including the inner detector
    inside `HybridLivenessDetector` — picks up the same shared model.

    Raises:
        LivenessCheckError: If uniface is not installed or the model
            fails to construct. The original `ImportError` is preserved
            as `__cause__` so callers that need to disambiguate can walk
            the cause chain.
    """
    try:
        from uniface.spoofing import MiniFASNet
        model = MiniFASNet()
        logger.info("UniFace MiniFASNet model loaded (process-wide shared session)")
        return model
    except ImportError as e:
        logger.error(f"uniface package not installed: {e}")
        # Preserve the ImportError as __cause__ via `from e` so callers that
        # need to disambiguate "missing dependency" from "model crashed"
        # can walk `exc.__cause__`.
        raise LivenessCheckError(
            "uniface package is required for UniFace liveness detection. "
            "Install it with: pip install 'uniface>=3.0.0'"
        ) from e
    except Exception as e:
        logger.error(f"Failed to load MiniFASNet model: {e}", exc_info=True)
        raise LivenessCheckError(f"Failed to initialize MiniFASNet: {e}") from e


class UniFaceLivenessDetector(ILivenessDetector):
    """Liveness detector using UniFace MiniFASNet ONNX model.

    This implementation wraps the uniface.spoofing.MiniFASNet model
    to provide anti-spoofing predictions. The model outputs a confidence
    score indicating whether a face is real or spoofed.

    Follows Single Responsibility Principle: only handles MiniFASNet-based
    liveness detection. Dependencies are injected via constructor parameters.

    Attributes:
        _model: Lazy-loaded MiniFASNet model instance
        _liveness_threshold: Score threshold for considering a face as live (0-100)
    """

    def __init__(
        self,
        liveness_threshold: float = 70.0,
    ) -> None:
        """Initialize UniFace liveness detector.

        Args:
            liveness_threshold: Overall score threshold for liveness (0-100).
                Scores above this threshold are considered live.
        """
        self._liveness_threshold = liveness_threshold
        self._model: Optional["MiniFASNet"] = None
        # Per-instance async lock guarding lazy model construction. Multiple
        # concurrent check_liveness() coroutines (each offloading via
        # asyncio.to_thread) would otherwise race in _ensure_model_loaded
        # and instantiate MiniFASNet twice (Copilot post-merge PR #57).
        self._model_lock: asyncio.Lock = asyncio.Lock()

        logger.info(
            f"UniFaceLivenessDetector initialized: "
            f"liveness_threshold={liveness_threshold}"
        )

    async def _ensure_model_loaded(self) -> "MiniFASNet":
        """Lazy-load the MiniFASNet model on first use (async, thread-safe).

        Uses canonical double-checked locking: an unlocked fast-path read
        when the model is already loaded, then a lock-protected re-check
        before binding so only one coroutine ever populates ``self._model``.

        Returns the loaded model directly so callers don't have to deal
        with `Optional[MiniFASNet]` at the call site (Copilot post-merge
        finding on PR #59 — `.predict(...)` was failing mypy because
        `self._model` stayed typed `Optional` after the await).

        The actual MiniFASNet construction lives in the module-level
        `_get_shared_minifasnet()` `lru_cache`, so once any instance has
        triggered the load every later instance picks up the same ONNX
        session for free (Copilot post-merge round 8 / PR #64).

        Raises:
            LivenessCheckError: If uniface is not installed or model fails to load
        """
        # Fast path — already loaded, no lock needed
        if self._model is not None:
            return self._model

        async with self._model_lock:
            # Re-check after acquiring the lock: another coroutine may have
            # finished loading while we were awaiting the lock.
            if self._model is not None:
                return self._model

            # Resolve through the process-wide cache. Errors are already
            # wrapped in LivenessCheckError by `_get_shared_minifasnet`.
            self._model = _get_shared_minifasnet()
            return self._model

    def warm_model_sync(self) -> None:
        """Synchronously preload the MiniFASNet model.

        Intended for application startup (`initialize_dependencies`),
        during startup before serving requests. Even though FastAPI's
        ``lifespan`` is itself an `async` context, this hook is purely
        synchronous: it forces the module-level
        ``_get_shared_minifasnet()`` cache to populate so the heavy ONNX
        session-init cost is paid by the operator, never by an end user
        on the first `/verify` call after a deploy.

        Copilot post-merge round 8 (PR #64): the previous implementation
        wrote to a per-instance ``self._model``. Combined with
        ``get_liveness_detector()`` being intentionally non-cached (so
        per-session blink state in the enhanced/hybrid detectors stays
        isolated), the warmed instance was thrown away and request
        handlers still paid the first-call MiniFASNet load cost. Routing
        through the module-level cache fixes that without re-introducing
        cross-session blink-state leakage.

        Raises:
            LivenessCheckError: If uniface is not installed or model fails to load.
        """
        # Bind the cached model to this instance too so the very first
        # `check_liveness()` skips the lock + cache lookup.
        if self._model is None:
            self._model = _get_shared_minifasnet()
        logger.info("UniFace MiniFASNet model pre-loaded (sync warm-up)")

    async def check_liveness(self, image: np.ndarray) -> LivenessResult:
        """Check if image shows a live person using MiniFASNet.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format

        Returns:
            LivenessResult containing liveness score and challenge information

        Raises:
            LivenessCheckError: When liveness check fails
        """
        logger.info("Starting UniFace MiniFASNet liveness detection")

        try:
            # Validate input
            if image is None or image.size == 0:
                raise LivenessCheckError("Invalid input image")

            # Ensure model is loaded (async, thread-safe lazy init).
            # Bind the loaded model to a local so `.predict(...)` below is
            # typed as `MiniFASNet`, not `Optional[MiniFASNet]`.
            model = await self._ensure_model_loaded()

            # Convert BGR to RGB (UniFace expects RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run MiniFASNet prediction
            # The model returns a list of predictions per detected face
            h, w = image_rgb.shape[:2]
            bbox = [0, 0, w, h]
            # P2.11: ONNX inference is CPU-bound — offload off the event loop
            # via the project's shared ThreadPoolManager so ML_THREAD_POOL_SIZE
            # is honored (was asyncio.to_thread, which uses the default executor).
            from app.core.container import get_thread_pool
            raw_prediction = await get_thread_pool().run_blocking(
                model.predict, image_rgb, bbox
            )
            predictions = self._normalize_predictions(raw_prediction)

            if not predictions:
                # No face detected by MiniFASNet - fall back to "real" with low confidence
                logger.warning(
                    "MiniFASNet did not detect any face, "
                    "returning indeterminate failure result"
                )
                return LivenessResult(
                    is_live=False,
                    score=0.0,
                    challenge="uniface_minifasnet",
                    challenge_completed=False,
                    confidence=0.0,
                    details={
                        "backend_score": 0.0,
                        "fallback_reason": "no_face_detected",
                        "indeterminate": True,
                    },
                )

            # Use the first (or largest) prediction
            prediction = predictions[0]

            # Extract the anti-spoofing score
            # MiniFASNet returns a score where higher = more likely real
            # The exact attribute depends on the uniface API
            score = self._extract_score(prediction)

            # Convert to 0-100 scale
            liveness_score = min(100.0, max(0.0, score * 100.0))

            is_live = liveness_score >= self._liveness_threshold

            logger.info(
                f"UniFace liveness detection complete: "
                f"score={liveness_score:.2f}, is_live={is_live}"
            )

            return LivenessResult(
                is_live=is_live,
                score=liveness_score,
                challenge="uniface_minifasnet",
                challenge_completed=True,
                confidence=score,
                details={"backend_score": score},
            )

        except LivenessCheckError:
            raise
        except Exception as e:
            logger.error(f"UniFace liveness check error: {e}", exc_info=True)
            logger.warning(
                "Returning indeterminate failure result due to MiniFASNet error"
            )
            return LivenessResult(
                is_live=False,
                score=0.0,
                challenge="uniface_minifasnet",
                challenge_completed=False,
                confidence=0.0,
                details={
                    "backend_score": 0.0,
                    "fallback_reason": "model_inference_failed",
                    "indeterminate": True,
                },
            )

    def _normalize_predictions(self, raw_prediction: Any) -> list[Any]:
        """Normalize UniFace prediction output to a list.

        UniFace can return either:
        - a list/tuple of predictions
        - a single SpoofingResult-like object
        - None
        """
        if raw_prediction is None:
            return []

        if isinstance(raw_prediction, (list, tuple)):
            return list(raw_prediction)

        return [raw_prediction]

    def _extract_score(self, prediction) -> float:
        """Extract the anti-spoofing confidence score from a MiniFASNet prediction.

        Handles different prediction formats that MiniFASNet may return.

        Args:
            prediction: A single prediction result from MiniFASNet

        Returns:
            Confidence score in range [0.0, 1.0] where higher = more likely real
        """
        # MiniFASNet prediction may be a dict, tuple, or object with attributes
        if isinstance(prediction, dict):
            # Try common key names
            for key in ("score", "confidence", "real_score", "liveness"):
                if key in prediction:
                    return float(prediction[key])
            # If dict has 'label' and 'score' pattern
            if "label" in prediction:
                score = float(prediction.get("score", prediction.get("confidence", 0.5)))
                # If label indicates spoof, invert the score
                label = str(prediction["label"]).lower()
                if label in ("spoof", "fake", "attack", "0"):
                    return 1.0 - score
                return score

        elif isinstance(prediction, (list, tuple)):
            # Assume [label, score] or [score]
            if len(prediction) >= 2:
                label = prediction[0]
                score = float(prediction[1])
                if isinstance(label, str) and label.lower() in ("spoof", "fake", "attack"):
                    return 1.0 - score
                return score
            elif len(prediction) == 1:
                return float(prediction[0])

        elif isinstance(prediction, (int, float)):
            return float(prediction)

        elif hasattr(prediction, "score"):
            return float(prediction.score)

        elif hasattr(prediction, "confidence"):
            return float(prediction.confidence)

        # Fallback: moderate confidence
        logger.warning(
            f"Unknown MiniFASNet prediction format: {type(prediction)}, "
            f"using default score 0.5"
        )
        return 0.5

    def get_challenge_type(self) -> str:
        """Get the type of liveness challenge used.

        Returns:
            Challenge type identifier
        """
        return "uniface_minifasnet"

    def get_liveness_threshold(self) -> float:
        """Get the threshold for considering result as live.

        Returns:
            Liveness score threshold (0-100)
        """
        return self._liveness_threshold
