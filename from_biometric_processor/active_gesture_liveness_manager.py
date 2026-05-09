# This file is a copy from biometric-processor/app/application/services/active_gesture_liveness_manager.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Active gesture liveness manager (landmarks-only, server-side).

Design (2026-04-24, Phase 1)
----------------------------
MediaPipe hand inference runs CLIENT-SIDE. The server accepts 21-point hand
landmark arrays plus two anti-spoof telemetry scalars (tremor_variance,
brightness_std) and verifies challenges with deterministic geometry + DTW.
No TFLite / MediaPipe runtime is loaded on the server — the only dependency
beyond stdlib is numpy (already in requirements.txt for face liveness).

The public API intentionally mirrors :class:`ActiveLivenessManager` so the
existing :class:`StartActiveLivenessUseCase` and
:class:`ProcessActiveLivenessFrameUseCase` can inject this manager with no
changes, by switching on session modality in the container.

Challenges implemented:
    FINGER_COUNT   — open-finger count via wrist-PIP / wrist-TIP ratio
    SHAPE_TRACE    — DTW against template catalog (loaded from JSON)
    WAVE           — zero-crossing + amplitude on wrist x-coord history
    HAND_FLIP      — sign change on palm-normal vector proxy
    FINGER_TAP     — index tip ↔ middle tip proximity over window
    PINCH          — thumb tip ↔ index tip distance
    PEEK_A_BOO     — monotonic hand-covers-face sequence (client flag verified)
    MATH           — reuses FINGER_COUNT with a random target
    HOLD_POSITION  — wrist std-dev below threshold over a window

Port sources (prototype):
    practice-and-test/GestureAnalysis/gesture_validator.py
    practice-and-test/GestureAnalysis/shape_tracer.py
    practice-and-test/GestureAnalysis/motion_analyzer.py
    practice-and-test/GestureAnalysis/anti_spoof.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from app.api.schemas.active_liveness import (
    ActiveLivenessConfig,
    ActiveLivenessResponse,
    ActiveLivenessSession,
    Challenge,
    ChallengeResult,
    ChallengeStatus,
    ChallengeType,
    get_challenge_instruction,
)
from app.api.schemas.gesture_liveness import (
    GestureChallengeType,
    GestureFramePayload,
    GestureLivenessConfig,
    HandLandmark,
    ShapeTemplate,
    ShapeTemplateCatalog,
    is_gesture_challenge,
    to_gesture_challenge_type,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MediaPipe Hand Landmarker index constants
# ---------------------------------------------------------------------------
# (Mirror of the prototype's gesture_validator._LM, copy-pasted to keep the
# server code free of any prototype imports.)

_WRIST = 0
_THUMB_CMC, _THUMB_MCP, _THUMB_IP, _THUMB_TIP = 1, 2, 3, 4
_INDEX_MCP, _INDEX_PIP, _INDEX_DIP, _INDEX_TIP = 5, 6, 7, 8
_MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_DIP, _MIDDLE_TIP = 9, 10, 11, 12
_RING_MCP, _RING_PIP, _RING_DIP, _RING_TIP = 13, 14, 15, 16
_PINKY_MCP, _PINKY_PIP, _PINKY_DIP, _PINKY_TIP = 17, 18, 19, 20


_FINGER_JOINTS: Dict[str, Tuple[int, int]] = {
    "INDEX": (_INDEX_PIP, _INDEX_TIP),
    "MIDDLE": (_MIDDLE_PIP, _MIDDLE_TIP),
    "RING": (_RING_PIP, _RING_TIP),
    "PINKY": (_PINKY_PIP, _PINKY_TIP),
}


def _euclid(a: HandLandmark, b: HandLandmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _hand_scale(landmarks: Sequence[HandLandmark]) -> float:
    return _euclid(landmarks[_WRIST], landmarks[_MIDDLE_MCP])


def _finger_ratio(landmarks: Sequence[HandLandmark], finger: str) -> float:
    """Ported from GestureAnalysis/gesture_validator.finger_ratio.

    Returns a normalised "openness" metric. Higher = more extended.
    """

    hs = _hand_scale(landmarks)
    if hs < 1e-9:
        return 0.0

    wrist = landmarks[_WRIST]
    if finger == "THUMB":
        return _euclid(landmarks[_THUMB_TIP], landmarks[_PINKY_MCP]) / hs

    pip_id, tip_id = _FINGER_JOINTS[finger]
    dist_tip = _euclid(wrist, landmarks[tip_id])
    dist_pip = _euclid(wrist, landmarks[pip_id])
    return (dist_tip - dist_pip) / hs


def _count_open_fingers(
    landmarks: Sequence[HandLandmark],
    finger_open_threshold: float,
    thumb_open_threshold: float,
) -> int:
    """Count extended fingers on one hand using the v6b distance-difference rule."""

    count = 0
    if _finger_ratio(landmarks, "THUMB") > thumb_open_threshold:
        count += 1
    for finger in ("INDEX", "MIDDLE", "RING", "PINKY"):
        if _finger_ratio(landmarks, finger) > finger_open_threshold:
            count += 1
    return count


# ---------------------------------------------------------------------------
# DTW (straight port from shape_tracer.dtw_normalised_cost)
# ---------------------------------------------------------------------------


def _resample(path: Sequence[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    if len(path) == 0:
        return [(0.0, 0.0)] * n
    if len(path) == 1:
        return [path[0]] * n
    dists: List[float] = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i - 1][0]
        dy = path[i][1] - path[i - 1][1]
        dists.append(dists[-1] + math.sqrt(dx * dx + dy * dy))
    total = dists[-1]
    if total < 1e-9:
        return [path[0]] * n
    out: List[Tuple[float, float]] = []
    j = 0
    for k in range(n):
        target = k * total / (n - 1)
        while j < len(dists) - 2 and dists[j + 1] < target:
            j += 1
        seg = dists[j + 1] - dists[j]
        t = (target - dists[j]) / seg if seg > 1e-9 else 0.0
        x = path[j][0] + t * (path[j + 1][0] - path[j][0])
        y = path[j][1] + t * (path[j + 1][1] - path[j][1])
        out.append((x, y))
    return out


def _centroid_normalise(path: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    n = len(path)
    if n == 0:
        return []
    cx = sum(p[0] for p in path) / n
    cy = sum(p[1] for p in path) / n
    centred = [(p[0] - cx, p[1] - cy) for p in path]
    max_r = max(math.sqrt(x * x + y * y) for x, y in centred) or 1.0
    return [(x / max_r, y / max_r) for x, y in centred]


def _dtw_cost(path_a: Sequence[Tuple[float, float]], path_b: Sequence[Tuple[float, float]]) -> float:
    n, m = len(path_a), len(path_b)
    if n == 0 or m == 0:
        return float("inf")
    a = np.asarray(path_a, dtype=np.float32)
    b = np.asarray(path_b, dtype=np.float32)
    diff = a[:, None, :] - b[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(D[i - 1, j - 1])
            dtw[i, j] = cost + min(
                float(dtw[i - 1, j]),
                float(dtw[i, j - 1]),
                float(dtw[i - 1, j - 1]),
            )
    return float(dtw[n, m]) / max(n, m)


# ---------------------------------------------------------------------------
# Shape template catalog (lazy-loaded from JSON, mtime-based ETag)
# ---------------------------------------------------------------------------


_DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent.parent / "resources" / "gesture_shape_templates.json"
)


@dataclass
class _CatalogCacheEntry:
    mtime_ns: int
    version: str
    catalog: ShapeTemplateCatalog


_shape_catalog_cache: Optional[_CatalogCacheEntry] = None


def _get_template_path() -> Path:
    override = os.getenv("GESTURE_SHAPE_TEMPLATES_PATH", "").strip()
    if override:
        return Path(override)
    return _DEFAULT_TEMPLATE_PATH


def load_shape_template_catalog() -> ShapeTemplateCatalog:
    """Return the shape-template catalog; cached until file mtime changes."""

    global _shape_catalog_cache
    path = _get_template_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Gesture shape templates not found at {path}. "
            "Ensure app/resources/gesture_shape_templates.json is deployed."
        )
    mtime_ns = path.stat().st_mtime_ns
    if _shape_catalog_cache and _shape_catalog_cache.mtime_ns == mtime_ns:
        return _shape_catalog_cache.catalog
    raw = json.loads(path.read_text())
    templates = [ShapeTemplate(**t) for t in raw.get("templates", [])]
    # Use mtime_ns as the opaque ETag version — stable across restarts for the
    # same file, and trivially bumps when ops overwrite the asset.
    version = f"mtime-{mtime_ns}"
    catalog = ShapeTemplateCatalog(version=version, templates=templates)
    _shape_catalog_cache = _CatalogCacheEntry(mtime_ns, version, catalog)
    return catalog


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ActiveGestureLivenessManager:
    """Server-side gesture liveness manager.

    Public API mirrors :class:`ActiveLivenessManager` so the generic use
    cases can route frames to either manager based on session modality.
    """

    # Default thresholds (v6b tightened values from the prototype).
    DEFAULT_FINGER_OPEN_THRESHOLD = 0.20
    DEFAULT_FINGER_CLOSE_THRESHOLD = 0.12
    DEFAULT_THUMB_OPEN_THRESHOLD = 0.75
    DEFAULT_THUMB_CLOSE_THRESHOLD = 0.60
    DEFAULT_WAVE_MIN_FREQ_HZ = 1.0
    DEFAULT_WAVE_MAX_FREQ_HZ = 4.0
    DEFAULT_WAVE_MIN_REVERSALS = 2
    DEFAULT_WAVE_MIN_SWING = 0.10
    DEFAULT_DTW_COST_MAX = 0.25
    DEFAULT_DTW_RESAMPLE_N = 50
    DEFAULT_TREMOR_VARIANCE_MIN = 3e-4
    DEFAULT_BRIGHTNESS_STD_MIN = 0.05
    DEFAULT_FINGER_TAP_MAX_DIST_SCALED = 0.08
    DEFAULT_PINCH_MAX_DIST_SCALED = 0.12
    DEFAULT_HOLD_MAX_VARIANCE = 2e-3

    def __init__(
        self,
        finger_open_distance: float = DEFAULT_FINGER_OPEN_THRESHOLD,
        thumb_open_distance: float = DEFAULT_THUMB_OPEN_THRESHOLD,
        wave_min_freq_hz: float = DEFAULT_WAVE_MIN_FREQ_HZ,
        wave_max_freq_hz: float = DEFAULT_WAVE_MAX_FREQ_HZ,
        dtw_cost_max: float = DEFAULT_DTW_COST_MAX,
        tremor_variance_min: float = DEFAULT_TREMOR_VARIANCE_MIN,
        brightness_std_min: float = DEFAULT_BRIGHTNESS_STD_MIN,
        finger_tap_max_dist_scaled: float = DEFAULT_FINGER_TAP_MAX_DIST_SCALED,
        pinch_max_dist_scaled: float = DEFAULT_PINCH_MAX_DIST_SCALED,
        hold_max_variance: float = DEFAULT_HOLD_MAX_VARIANCE,
    ) -> None:
        self._finger_open_th = finger_open_distance
        self._thumb_open_th = thumb_open_distance
        self._wave_min_freq = wave_min_freq_hz
        self._wave_max_freq = wave_max_freq_hz
        self._dtw_cost_max = dtw_cost_max
        self._tremor_min = tremor_variance_min
        self._brightness_min = brightness_std_min
        self._finger_tap_max = finger_tap_max_dist_scaled
        self._pinch_max = pinch_max_dist_scaled
        self._hold_max_variance = hold_max_variance
        logger.info(
            "ActiveGestureLivenessManager initialised "
            "(finger_open=%s, thumb_open=%s, wave=%s-%sHz, dtw_max=%s, "
            "tremor_min=%s, brightness_min=%s)",
            finger_open_distance,
            thumb_open_distance,
            wave_min_freq_hz,
            wave_max_freq_hz,
            dtw_cost_max,
            tremor_variance_min,
            brightness_std_min,
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self, config: Optional[ActiveLivenessConfig | GestureLivenessConfig] = None
    ) -> ActiveLivenessSession:
        if config is None:
            config = GestureLivenessConfig()
        elif isinstance(config, ActiveLivenessConfig) and not isinstance(
            config, GestureLivenessConfig
        ):
            # Widen plain ActiveLivenessConfig into a gesture config.
            config = GestureLivenessConfig(
                num_challenges=config.num_challenges,
                challenge_timeout=config.challenge_timeout,
                randomize=config.randomize,
                session_timeout_seconds=config.session_timeout_seconds,
                required_challenges=config.required_challenges,
            )

        created_at = time.time()
        challenges = self._generate_challenges(config)
        gesture_state = self._new_gesture_state(challenges, config)
        session = ActiveLivenessSession(
            session_id=str(uuid.uuid4()),
            modality="gesture",
            challenges=challenges,
            current_challenge_index=0,
            started_at=created_at,
            expires_at=created_at + config.session_timeout_seconds,
            last_activity_at=created_at,
            current_challenge_started_at=created_at,
            gesture_state=gesture_state,
        )
        logger.info(
            "Created active gesture liveness session %s with %s challenges",
            session.session_id,
            len(challenges),
        )
        return session

    def _generate_challenges(self, config: GestureLivenessConfig) -> List[Challenge]:
        # Prefer the gesture-typed override; fall back to the base-config list;
        # else pick at random.
        pool: List[ChallengeType]
        if config.required_gesture_challenges:
            pool = [ChallengeType(ct.value) for ct in config.required_gesture_challenges]
        elif config.required_challenges:
            pool = [ct for ct in config.required_challenges if is_gesture_challenge(ct)]
            if not pool:
                pool = list(self._default_pool())
        else:
            pool = list(self._default_pool())
            if config.randomize:
                random.shuffle(pool)

        pool = pool[: max(1, config.num_challenges)]
        return [
            Challenge(
                type=challenge_type,
                instruction=get_challenge_instruction(challenge_type),
                timeout_seconds=config.challenge_timeout,
            )
            for challenge_type in pool
        ]

    @staticmethod
    def _default_pool() -> Tuple[ChallengeType, ...]:
        return (
            ChallengeType.FINGER_COUNT,
            ChallengeType.WAVE,
            ChallengeType.FINGER_TAP,
            ChallengeType.PINCH,
            ChallengeType.HAND_FLIP,
            ChallengeType.HOLD_POSITION,
        )

    def _new_gesture_state(
        self, challenges: Sequence[Challenge], config: GestureLivenessConfig
    ) -> Dict[str, Any]:
        """Build the scratch-state dict stored on the session.

        Per-challenge sub-state is keyed by challenge index so challenges can
        accumulate their own windows of samples across frames.
        """

        per_challenge: Dict[str, Dict[str, Any]] = {}
        for idx, challenge in enumerate(challenges):
            gtype = challenge.type
            if gtype == ChallengeType.FINGER_COUNT:
                target = config.target_finger_count
                if target is None:
                    # Ask for 1..5 by default (avoid 0 = fist, avoid two-hand).
                    target = random.randint(1, 5)
                per_challenge[str(idx)] = {"target": int(target), "matched_frames": 0}
            elif gtype == ChallengeType.MATH:
                target = config.target_finger_count
                if target is None:
                    target = random.randint(1, 5)
                a = random.randint(0, target)
                b = target - a
                per_challenge[str(idx)] = {
                    "target": int(target),
                    "question": f"{a} + {b}",
                    "matched_frames": 0,
                }
            elif gtype == ChallengeType.SHAPE_TRACE:
                try:
                    templates = load_shape_template_catalog().templates
                    picked = random.choice(templates) if templates else None
                except Exception:  # pragma: no cover - defensive
                    logger.warning("Shape templates unavailable; SHAPE_TRACE will fail closed")
                    picked = None
                per_challenge[str(idx)] = {
                    "template_key": picked.key if picked else None,
                    "trace": [],
                }
            elif gtype == ChallengeType.WAVE:
                per_challenge[str(idx)] = {"samples": []}  # [(t_s, wrist_x), ...]
            elif gtype == ChallengeType.HAND_FLIP:
                per_challenge[str(idx)] = {"normal_signs": []}
            elif gtype == ChallengeType.FINGER_TAP:
                per_challenge[str(idx)] = {"tap_frames": 0}
            elif gtype == ChallengeType.PINCH:
                per_challenge[str(idx)] = {"pinch_frames": 0}
            elif gtype == ChallengeType.PEEK_A_BOO:
                per_challenge[str(idx)] = {"sequence": []}  # [bool,...]
            elif gtype == ChallengeType.HOLD_POSITION:
                per_challenge[str(idx)] = {"wrist_samples": []}  # [(x,y),...]
        return {"per_challenge": per_challenge}

    def get_current_challenge(self, session: ActiveLivenessSession) -> Optional[Challenge]:
        if session.is_complete:
            return None
        if session.current_challenge_index >= len(session.challenges):
            return None
        return session.challenges[session.current_challenge_index]

    def is_expired(self, session: ActiveLivenessSession, now: Optional[float] = None) -> bool:
        return (now or time.time()) >= session.expires_at

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    async def process_frame(
        self,
        session: ActiveLivenessSession,
        landmarks_payload: GestureFramePayload,
    ) -> ActiveLivenessResponse:
        current_challenge = self.get_current_challenge(session)
        if current_challenge is None:
            return self.build_response(session=session)

        if current_challenge.status == ChallengeStatus.PENDING:
            current_challenge.status = ChallengeStatus.IN_PROGRESS
            session.current_challenge_started_at = time.time()

        elapsed = time.time() - session.current_challenge_started_at
        time_remaining = max(0.0, current_challenge.timeout_seconds - elapsed)

        # 1) Anti-spoof gate. Reject BEFORE running geometry so a trivially
        #    static frame cannot accumulate "progress".
        spoof_reason = self._anti_spoof_reject(landmarks_payload)
        if spoof_reason is not None:
            detection = ChallengeResult(
                challenge_type=current_challenge.type,
                detected=False,
                confidence=0.0,
                details={"anti_spoof_rejected": True, "reason": spoof_reason},
            )
            return self.build_response(
                session=session,
                detection=detection,
                feedback="Spoof check failed — please use a real hand in a lit scene.",
            )

        if time_remaining <= 0:
            current_challenge.attempts += 1
            feedback = "Challenge failed."
            if current_challenge.attempts >= current_challenge.max_attempts:
                current_challenge.status = ChallengeStatus.FAILED
                self._advance_to_next_challenge(session)
            else:
                session.current_challenge_started_at = time.time()
                feedback = "Time's up! Try again."
                # Reset only the current-challenge scratch state on retry.
                self._reset_challenge_state(session, session.current_challenge_index, session.challenges)
            return self.build_response(session=session, feedback=feedback)

        # 2) Geometry.
        detection = self._detect_challenge(session, current_challenge, landmarks_payload)
        feedback = self._get_guidance(current_challenge.type, detection)

        if detection.detected:
            current_challenge.status = ChallengeStatus.COMPLETED
            current_challenge.confidence = detection.confidence
            self._advance_to_next_challenge(session)
            if session.is_complete:
                return self.build_response(
                    session=session,
                    detection=detection,
                    feedback="Challenge sequence completed.",
                )
            feedback = "Great job!"

        return self.build_response(session=session, detection=detection, feedback=feedback)

    def _anti_spoof_reject(self, payload: GestureFramePayload) -> Optional[str]:
        """Return None if the frame passes anti-spoof; else a human-readable reason."""

        if payload.tremor_variance < self._tremor_min:
            return (
                f"tremor_variance={payload.tremor_variance:.6f} < "
                f"min={self._tremor_min:.6f} (frame appears static)"
            )
        if payload.brightness_std < self._brightness_min:
            return (
                f"brightness_std={payload.brightness_std:.6f} < "
                f"min={self._brightness_min:.6f} (scene brightness too uniform)"
            )
        return None

    def _advance_to_next_challenge(self, session: ActiveLivenessSession) -> None:
        session.current_challenge_index += 1
        session.current_challenge_started_at = time.time()
        if session.current_challenge_index >= len(session.challenges):
            self._complete_session(session)

    def _complete_session(self, session: ActiveLivenessSession) -> None:
        completed = sum(1 for c in session.challenges if c.status == ChallengeStatus.COMPLETED)
        total = len(session.challenges)
        session.completed_at = time.time()
        session.is_complete = True
        session.overall_score = (completed / total) * 100 if total else 0.0
        session.passed = completed >= max(1, int(total * 0.6))

    def _reset_challenge_state(
        self,
        session: ActiveLivenessSession,
        challenge_index: int,
        challenges: Sequence[Challenge],
    ) -> None:
        """Clear accumulated sample buffers for a challenge being retried."""

        state = session.gesture_state.get("per_challenge", {}).get(str(challenge_index))
        if not state:
            return
        for key in ("matched_frames", "tap_frames", "pinch_frames"):
            if key in state:
                state[key] = 0
        for key in ("samples", "normal_signs", "sequence", "wrist_samples", "trace"):
            if key in state:
                state[key] = []

    # ------------------------------------------------------------------
    # Per-challenge detectors
    # ------------------------------------------------------------------

    def _detect_challenge(
        self,
        session: ActiveLivenessSession,
        challenge: Challenge,
        payload: GestureFramePayload,
    ) -> ChallengeResult:
        gtype = challenge.type
        per = session.gesture_state.setdefault("per_challenge", {}).setdefault(
            str(session.current_challenge_index), {}
        )
        # Pick one primary hand for single-hand challenges. Prefer right, fall
        # back to left, fail if neither is present.
        primary = payload.landmarks_right or payload.landmarks_left

        if primary is None:
            return ChallengeResult(
                challenge_type=gtype,
                detected=False,
                confidence=0.0,
                details={"error": "No hand landmarks in frame"},
            )

        if gtype == ChallengeType.FINGER_COUNT:
            return self._detect_finger_count(primary, per)
        if gtype == ChallengeType.MATH:
            return self._detect_math(primary, per)
        if gtype == ChallengeType.WAVE:
            return self._detect_wave(primary, per, payload.frame_time_ms)
        if gtype == ChallengeType.HAND_FLIP:
            return self._detect_hand_flip(primary, per)
        if gtype == ChallengeType.FINGER_TAP:
            return self._detect_finger_tap(primary, per)
        if gtype == ChallengeType.PINCH:
            return self._detect_pinch(primary, per)
        if gtype == ChallengeType.PEEK_A_BOO:
            return self._detect_peek_a_boo(payload, per)
        if gtype == ChallengeType.HOLD_POSITION:
            return self._detect_hold_position(primary, per)
        if gtype == ChallengeType.SHAPE_TRACE:
            return self._detect_shape_trace(primary, per)

        return ChallengeResult(
            challenge_type=gtype,
            detected=False,
            confidence=0.0,
            details={"error": f"Unsupported gesture challenge: {gtype}"},
        )

    # ----- finger-count / math -----

    def _detect_finger_count(
        self, landmarks: Sequence[HandLandmark], per: Dict[str, Any]
    ) -> ChallengeResult:
        target = int(per.get("target", 1))
        n = _count_open_fingers(landmarks, self._finger_open_th, self._thumb_open_th)
        matched = per.get("matched_frames", 0)
        if n == target:
            matched += 1
        else:
            matched = 0  # reset on mismatch so a flicker doesn't accrue
        per["matched_frames"] = matched
        # Require 3 consecutive frames to count as a confirmed match.
        detected = matched >= 3
        return ChallengeResult(
            challenge_type=ChallengeType.FINGER_COUNT,
            detected=detected,
            confidence=1.0 if detected else min(1.0, matched / 3.0),
            details={"target": target, "observed": n, "consecutive_matches": matched},
        )

    def _detect_math(
        self, landmarks: Sequence[HandLandmark], per: Dict[str, Any]
    ) -> ChallengeResult:
        result = self._detect_finger_count(landmarks, per)
        # Expose the math question so the client can render it.
        result.details["question"] = per.get("question")
        return ChallengeResult(
            challenge_type=ChallengeType.MATH,
            detected=result.detected,
            confidence=result.confidence,
            details=result.details,
        )

    # ----- wave -----

    def _detect_wave(
        self,
        landmarks: Sequence[HandLandmark],
        per: Dict[str, Any],
        frame_time_ms: int,
    ) -> ChallengeResult:
        samples: List[List[float]] = per.setdefault("samples", [])
        samples.append([float(frame_time_ms) / 1000.0, float(landmarks[_WRIST].x)])
        # Keep the last ~2 s of samples.
        if len(samples) > 120:
            del samples[: len(samples) - 120]

        if len(samples) < 10:
            return ChallengeResult(
                challenge_type=ChallengeType.WAVE,
                detected=False,
                confidence=0.0,
                details={"reason": "accumulating samples", "n_samples": len(samples)},
            )

        xs = [s[1] for s in samples]
        total_disp = max(xs) - min(xs)
        if total_disp < 0.20:
            return ChallengeResult(
                challenge_type=ChallengeType.WAVE,
                detected=False,
                confidence=0.0,
                details={"reason": "insufficient displacement", "total_disp": total_disp},
            )

        # Count sign changes of first-difference with an amplitude gate.
        reversals = 0
        reversal_times: List[float] = []
        last_dir = 0
        anchor_x = xs[0]
        anchor_t = samples[0][0]
        for i in range(1, len(xs)):
            dx = xs[i] - anchor_x
            if abs(dx) < self.DEFAULT_WAVE_MIN_SWING:
                continue
            direction = 1 if dx > 0 else -1
            if last_dir != 0 and direction != last_dir:
                reversals += 1
                reversal_times.append(samples[i][0])
            last_dir = direction
            anchor_x = xs[i]
            anchor_t = samples[i][0]

        detected = reversals >= self.DEFAULT_WAVE_MIN_REVERSALS
        freq_hz = 0.0
        if len(reversal_times) >= 2:
            duration = reversal_times[-1] - reversal_times[0]
            if duration > 0:
                freq_hz = (reversals / duration) / 2.0
                if freq_hz < self._wave_min_freq or freq_hz > self._wave_max_freq:
                    detected = False

        return ChallengeResult(
            challenge_type=ChallengeType.WAVE,
            detected=detected,
            confidence=1.0 if detected else min(1.0, reversals / self.DEFAULT_WAVE_MIN_REVERSALS),
            details={
                "reversals": reversals,
                "total_disp": total_disp,
                "freq_hz": freq_hz,
            },
        )

    # ----- hand flip -----

    def _detect_hand_flip(
        self, landmarks: Sequence[HandLandmark], per: Dict[str, Any]
    ) -> ChallengeResult:
        # Palm normal proxy: use the cross product of (index_mcp - wrist) and
        # (pinky_mcp - wrist). The Z component's sign flips when the palm faces
        # the camera vs when the back of the hand faces the camera.
        wrist = landmarks[_WRIST]
        i = landmarks[_INDEX_MCP]
        p = landmarks[_PINKY_MCP]
        v1 = (i.x - wrist.x, i.y - wrist.y, i.z - wrist.z)
        v2 = (p.x - wrist.x, p.y - wrist.y, p.z - wrist.z)
        # cross product Z component:
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        signs: List[int] = per.setdefault("normal_signs", [])
        sign = 1 if cross_z > 0 else (-1 if cross_z < 0 else 0)
        if sign != 0 and (not signs or signs[-1] != sign):
            signs.append(sign)
        detected = len({s for s in signs}) >= 2  # observed both orientations
        return ChallengeResult(
            challenge_type=ChallengeType.HAND_FLIP,
            detected=detected,
            confidence=1.0 if detected else 0.5 if signs else 0.0,
            details={"sign_sequence": list(signs), "cross_z": cross_z},
        )

    # ----- finger tap -----

    def _detect_finger_tap(
        self, landmarks: Sequence[HandLandmark], per: Dict[str, Any]
    ) -> ChallengeResult:
        hs = _hand_scale(landmarks) or 1e-6
        idx_tip = landmarks[_INDEX_TIP]
        mid_tip = landmarks[_MIDDLE_TIP]
        dist = _euclid(idx_tip, mid_tip) / hs
        tap_frames = per.get("tap_frames", 0)
        if dist < self._finger_tap_max:
            tap_frames += 1
        else:
            tap_frames = max(0, tap_frames)  # don't reset — allow stopping and resuming
        per["tap_frames"] = tap_frames
        detected = tap_frames >= 2  # 2 frames = a deliberate tap, not noise
        return ChallengeResult(
            challenge_type=ChallengeType.FINGER_TAP,
            detected=detected,
            confidence=1.0 if detected else min(1.0, tap_frames / 2.0),
            details={"dist_scaled": dist, "tap_frames": tap_frames},
        )

    # ----- pinch -----

    def _detect_pinch(
        self, landmarks: Sequence[HandLandmark], per: Dict[str, Any]
    ) -> ChallengeResult:
        hs = _hand_scale(landmarks) or 1e-6
        thumb_tip = landmarks[_THUMB_TIP]
        idx_tip = landmarks[_INDEX_TIP]
        dist = _euclid(thumb_tip, idx_tip) / hs
        # Also gate on z-axis proximity so a hand facing away can't fake a pinch.
        z_delta = abs(thumb_tip.z - idx_tip.z)
        pinch_frames = per.get("pinch_frames", 0)
        if dist < self._pinch_max and z_delta < 0.15:
            pinch_frames += 1
        else:
            pinch_frames = max(0, pinch_frames - 1)
        per["pinch_frames"] = pinch_frames
        detected = pinch_frames >= 2
        return ChallengeResult(
            challenge_type=ChallengeType.PINCH,
            detected=detected,
            confidence=1.0 if detected else min(1.0, pinch_frames / 2.0),
            details={"dist_scaled": dist, "z_delta": z_delta, "pinch_frames": pinch_frames},
        )

    # ----- peek-a-boo -----

    def _detect_peek_a_boo(
        self, payload: GestureFramePayload, per: Dict[str, Any]
    ) -> ChallengeResult:
        sequence: List[bool] = per.setdefault("sequence", [])
        # The client MUST flag face-covered frames; the server just checks the
        # monotonic pattern [covered -> revealed]. A single-frame flag is not
        # enough.
        if payload.face_covered is None:
            return ChallengeResult(
                challenge_type=ChallengeType.PEEK_A_BOO,
                detected=False,
                confidence=0.0,
                details={"error": "client_face_covered_flag_missing"},
            )
        # Only record state transitions to keep the sequence short.
        if not sequence or sequence[-1] != bool(payload.face_covered):
            sequence.append(bool(payload.face_covered))
        # Accept: saw "covered" for at least one transition, followed by "revealed".
        detected = (
            True in sequence
            and False in sequence
            and sequence.index(True) < (len(sequence) - 1)
            and sequence[-1] is False
        )
        return ChallengeResult(
            challenge_type=ChallengeType.PEEK_A_BOO,
            detected=detected,
            confidence=1.0 if detected else 0.5 if True in sequence else 0.0,
            details={"sequence_len": len(sequence)},
        )

    # ----- hold position -----

    def _detect_hold_position(
        self, landmarks: Sequence[HandLandmark], per: Dict[str, Any]
    ) -> ChallengeResult:
        samples: List[List[float]] = per.setdefault("wrist_samples", [])
        wrist = landmarks[_WRIST]
        samples.append([float(wrist.x), float(wrist.y)])
        if len(samples) > 60:  # ~2 s at 30 fps
            del samples[: len(samples) - 60]
        if len(samples) < 15:
            return ChallengeResult(
                challenge_type=ChallengeType.HOLD_POSITION,
                detected=False,
                confidence=0.0,
                details={"reason": "need more samples", "n": len(samples)},
            )
        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        var = sum((x - mx) ** 2 + (y - my) ** 2 for x, y in zip(xs, ys)) / len(xs)
        detected = var < self._hold_max_variance
        return ChallengeResult(
            challenge_type=ChallengeType.HOLD_POSITION,
            detected=detected,
            confidence=1.0 if detected else 0.0,
            details={"variance": var, "n_samples": len(samples)},
        )

    # ----- shape trace -----

    def _detect_shape_trace(
        self, landmarks: Sequence[HandLandmark], per: Dict[str, Any]
    ) -> ChallengeResult:
        template_key = per.get("template_key")
        if not template_key:
            return ChallengeResult(
                challenge_type=ChallengeType.SHAPE_TRACE,
                detected=False,
                confidence=0.0,
                details={"error": "template unavailable"},
            )
        idx_tip = landmarks[_INDEX_TIP]
        trace: List[List[float]] = per.setdefault("trace", [])
        trace.append([float(idx_tip.x), float(idx_tip.y)])
        if len(trace) > 200:
            del trace[: len(trace) - 200]

        try:
            catalog = load_shape_template_catalog()
        except Exception as exc:  # pragma: no cover - defensive
            return ChallengeResult(
                challenge_type=ChallengeType.SHAPE_TRACE,
                detected=False,
                confidence=0.0,
                details={"error": f"catalog load failed: {exc}"},
            )
        template = next((t for t in catalog.templates if t.key == template_key), None)
        if template is None or len(trace) < template.min_trace_points:
            return ChallengeResult(
                challenge_type=ChallengeType.SHAPE_TRACE,
                detected=False,
                confidence=0.0,
                details={
                    "template": template_key,
                    "trace_len": len(trace),
                    "min_required": template.min_trace_points if template else None,
                },
            )

        resampled_trace = _resample(
            [tuple(p) for p in trace], self.DEFAULT_DTW_RESAMPLE_N
        )
        resampled_template = _resample(
            [tuple(p) for p in template.points], self.DEFAULT_DTW_RESAMPLE_N
        )
        norm_trace = _centroid_normalise(resampled_trace)
        norm_template = _centroid_normalise(resampled_template)
        cost = _dtw_cost(norm_trace, norm_template)
        detected = cost <= self._dtw_cost_max
        similarity = max(0.0, 1.0 - cost / max(self._dtw_cost_max, 1e-6))
        return ChallengeResult(
            challenge_type=ChallengeType.SHAPE_TRACE,
            detected=detected,
            confidence=min(1.0, similarity) if detected else 0.0,
            details={
                "template": template_key,
                "dtw_cost": cost,
                "threshold": self._dtw_cost_max,
                "similarity": similarity,
                "trace_len": len(trace),
            },
        )

    # ------------------------------------------------------------------
    # Guidance + response building (mirrors ActiveLivenessManager)
    # ------------------------------------------------------------------

    def _get_guidance(
        self, challenge_type: ChallengeType, detection: ChallengeResult
    ) -> str:
        d = detection.details or {}
        if challenge_type == ChallengeType.FINGER_COUNT:
            target = d.get("target")
            observed = d.get("observed")
            if observed != target:
                return f"Show exactly {target} fingers (currently {observed})"
            return "Hold steady..."
        if challenge_type == ChallengeType.MATH:
            q = d.get("question", "the sum")
            return f"Show the answer to {q} with your fingers"
        if challenge_type == ChallengeType.WAVE:
            return "Keep waving side to side"
        if challenge_type == ChallengeType.HAND_FLIP:
            return "Now flip your hand the other way"
        if challenge_type == ChallengeType.FINGER_TAP:
            return "Tap your index and middle fingertips"
        if challenge_type == ChallengeType.PINCH:
            return "Pinch thumb and index tight"
        if challenge_type == ChallengeType.PEEK_A_BOO:
            return "Cover your face, then reveal it"
        if challenge_type == ChallengeType.HOLD_POSITION:
            return "Hold your hand still"
        if challenge_type == ChallengeType.SHAPE_TRACE:
            return "Trace the shape with your index finger"
        return ""

    def build_response(
        self,
        session: ActiveLivenessSession,
        detection: Optional[ChallengeResult] = None,
        feedback: str = "",
    ) -> ActiveLivenessResponse:
        completed = sum(
            1 for c in session.challenges if c.status == ChallengeStatus.COMPLETED
        )
        total = len(session.challenges)
        current_challenge = self.get_current_challenge(session)
        time_remaining = 0.0
        instruction = "Session complete" if session.is_complete else ""
        if current_challenge is not None:
            elapsed = max(0.0, time.time() - session.current_challenge_started_at)
            time_remaining = max(0.0, current_challenge.timeout_seconds - elapsed)
            instruction = current_challenge.instruction
        elif session.is_complete:
            feedback = feedback or (
                f"Passed {completed}/{total} challenges"
                if session.passed
                else f"Only {completed}/{total} challenges passed"
            )
            instruction = (
                "All challenges completed! Liveness verified."
                if session.passed
                else "Session complete. Please try again."
            )

        progress = completed / total if total else 0.0
        if session.is_complete:
            progress = 1.0

        return ActiveLivenessResponse(
            session_id=session.session_id,
            current_challenge=current_challenge,
            challenge=current_challenge,
            challenge_progress=progress,
            time_remaining=time_remaining,
            detection=detection,
            challenges_completed=completed,
            challenges_total=total,
            session_complete=session.is_complete,
            session_passed=session.passed,
            overall_score=session.overall_score,
            instruction=instruction,
            feedback=feedback,
        )


__all__ = [
    "ActiveGestureLivenessManager",
    "load_shape_template_catalog",
]
