# This file is a copy from biometric-processor/app/infrastructure/ml/liveness/active_liveness_detector.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Active liveness detector using facial landmark analysis.

This detector uses MediaPipe Face Mesh to detect facial landmarks
and analyze facial actions (smile, blink) for liveness verification.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from app.domain.entities.liveness_result import LivenessResult
from app.domain.interfaces.liveness_detector import ILivenessDetector

logger = logging.getLogger(__name__)

# MediaPipe Face Mesh landmark indices
# Eye landmarks for EAR calculation
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Mouth landmarks for MAR calculation
UPPER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_INDICES = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14


class ActiveLivenessDetector(ILivenessDetector):
    """Liveness detector using facial action analysis.

    This implementation detects:
    - Smile: Using Mouth Aspect Ratio (MAR)
    - Blink: Using Eye Aspect Ratio (EAR)

    Uses MediaPipe Face Mesh for 468 facial landmark detection.
    """

    def __init__(
        self,
        ear_threshold: float = 0.25,
        mar_threshold: float = 0.6,
        liveness_threshold: float = 70.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize active liveness detector.

        Args:
            ear_threshold: Eye Aspect Ratio threshold for blink detection.
                          EAR < threshold indicates eyes are closed.
            mar_threshold: Mouth Aspect Ratio threshold for smile detection.
                          MAR > threshold indicates smiling.
            liveness_threshold: Overall liveness score threshold (0-100).
            min_detection_confidence: MediaPipe detection confidence.
            min_tracking_confidence: MediaPipe tracking confidence.
        """
        self._ear_threshold = ear_threshold
        self._mar_threshold = mar_threshold
        self._liveness_threshold = liveness_threshold
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._face_mesh = None

        logger.info(
            f"ActiveLivenessDetector initialized: "
            f"EAR threshold={ear_threshold}, MAR threshold={mar_threshold}, "
            f"liveness threshold={liveness_threshold}"
        )

    def _get_face_mesh(self):
        """Lazy initialization of MediaPipe Face Mesh."""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self._min_detection_confidence,
                    min_tracking_confidence=self._min_tracking_confidence,
                )
                logger.info("MediaPipe Face Mesh initialized")
            except ImportError:
                logger.error("MediaPipe not installed. Run: pip install mediapipe")
                raise
        return self._face_mesh

    async def check_liveness(self, image: np.ndarray) -> LivenessResult:
        """Check if image shows a live person using facial action analysis.

        Args:
            image: Face image as numpy array (BGR format)

        Returns:
            LivenessResult with liveness determination
        """
        return await self.detect(image)

    async def detect(
        self,
        image: np.ndarray,
        challenge: str = "smile_blink",
    ) -> LivenessResult:
        """Detect liveness using facial landmark analysis.

        Args:
            image: Face image as numpy array (BGR format)
            challenge: Challenge type (default: smile_blink)

        Returns:
            LivenessResult with liveness determination
        """
        logger.info("Starting active liveness detection")

        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get facial landmarks
        face_mesh = self._get_face_mesh()
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            logger.warning("No face landmarks detected")
            return LivenessResult(
                is_live=False,
                liveness_score=0.0,
                challenge=challenge,
                challenge_completed=False,
                details={
                    "ear": 0.0,
                    "mar": 0.0,
                    "eyes_open": False,
                    "smiling": False,
                },
            )

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]

        # Convert normalized landmarks to pixel coordinates
        landmark_points = [
            (int(lm.x * w), int(lm.y * h)) for lm in landmarks
        ]

        # Calculate Eye Aspect Ratio (EAR)
        left_ear = self._calculate_ear(landmark_points, LEFT_EYE_INDICES)
        right_ear = self._calculate_ear(landmark_points, RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        # Calculate Mouth Aspect Ratio (MAR)
        mar = self._calculate_mar(landmark_points)

        # Analyze facial state
        eyes_open = avg_ear > self._ear_threshold
        is_smiling = mar > self._mar_threshold

        # Calculate liveness score
        liveness_score = self._calculate_liveness_score(
            avg_ear, mar, eyes_open, is_smiling
        )

        is_live = liveness_score >= self._liveness_threshold

        logger.info(
            f"Active liveness detection complete: "
            f"score={liveness_score:.2f}, is_live={is_live}, "
            f"EAR={avg_ear:.3f}, MAR={mar:.3f}, "
            f"eyes_open={eyes_open}, smiling={is_smiling}"
        )

        return LivenessResult(
            is_live=is_live,
            liveness_score=liveness_score,
            challenge=challenge,
            challenge_completed=is_live,
            details={
                "ear": avg_ear,
                "mar": mar,
                "eyes_open": eyes_open,
                "smiling": is_smiling,
            },
        )

    def _calculate_ear(
        self,
        landmarks: List[Tuple[int, int]],
        eye_indices: List[int],
    ) -> float:
        """Calculate Eye Aspect Ratio (EAR).

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Where p1-p6 are the eye landmarks:
        - p1, p4: horizontal eye corners
        - p2, p6: upper vertical landmarks
        - p3, p5: lower vertical landmarks

        Args:
            landmarks: List of (x, y) landmark coordinates
            eye_indices: Indices of eye landmarks

        Returns:
            Eye Aspect Ratio value (0.0 to ~0.4)
        """
        try:
            # Get eye landmark points
            p1 = np.array(landmarks[eye_indices[0]])  # Left corner
            p2 = np.array(landmarks[eye_indices[1]])  # Upper left
            p3 = np.array(landmarks[eye_indices[2]])  # Upper right
            p4 = np.array(landmarks[eye_indices[3]])  # Right corner
            p5 = np.array(landmarks[eye_indices[4]])  # Lower right
            p6 = np.array(landmarks[eye_indices[5]])  # Lower left

            # Calculate distances
            vertical_1 = np.linalg.norm(p2 - p6)
            vertical_2 = np.linalg.norm(p3 - p5)
            horizontal = np.linalg.norm(p1 - p4)

            if horizontal == 0:
                return 0.0

            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return float(ear)
        except (IndexError, ValueError) as e:
            logger.warning(f"EAR calculation failed: {e}")
            return 0.0

    def _calculate_mar(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculate Mouth Aspect Ratio (MAR).

        MAR = vertical_distance / horizontal_distance

        Higher MAR indicates mouth is more open (smiling).

        Args:
            landmarks: List of (x, y) landmark coordinates

        Returns:
            Mouth Aspect Ratio value
        """
        try:
            # Get mouth corner points
            left_corner = np.array(landmarks[MOUTH_CORNER_LEFT])
            right_corner = np.array(landmarks[MOUTH_CORNER_RIGHT])

            # Get lip center points
            upper_lip = np.array(landmarks[UPPER_LIP_CENTER])
            lower_lip = np.array(landmarks[LOWER_LIP_CENTER])

            # Calculate distances
            horizontal = np.linalg.norm(right_corner - left_corner)
            vertical = np.linalg.norm(lower_lip - upper_lip)

            if horizontal == 0:
                return 0.0

            mar = vertical / horizontal
            return float(mar)
        except (IndexError, ValueError) as e:
            logger.warning(f"MAR calculation failed: {e}")
            return 0.0

    def _calculate_liveness_score(
        self,
        ear: float,
        mar: float,
        eyes_open: bool,
        is_smiling: bool,
    ) -> float:
        """Calculate overall liveness score.

        Scoring factors:
        - Eyes open: 40 points
        - Natural EAR range (0.2-0.35): 20 points
        - Smile detected: 30 points
        - Natural MAR range: 10 points

        Args:
            ear: Eye Aspect Ratio
            mar: Mouth Aspect Ratio
            eyes_open: Whether eyes are detected as open
            is_smiling: Whether smile is detected

        Returns:
            Liveness score (0-100)
        """
        score = 0.0

        # Eyes open score (40 points max)
        if eyes_open:
            score += 40.0
            # Bonus for natural EAR range (typical: 0.2-0.35)
            if 0.2 <= ear <= 0.35:
                score += 20.0
            elif 0.15 <= ear <= 0.4:
                score += 10.0

        # Smile score (30 points max)
        if is_smiling:
            score += 30.0

        # Natural facial proportions (10 points)
        # MAR typically ranges 0.3-0.8 for natural expressions
        if 0.3 <= mar <= 0.8:
            score += 10.0

        return min(100.0, score)

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

    def get_challenge_type(self) -> str:
        """Get the type of liveness challenge used.

        Returns:
            Challenge type
        """
        return "smile_blink"

    def get_liveness_threshold(self) -> float:
        """Get the threshold for considering result as live.

        Returns:
            Liveness score threshold (0-100)
        """
        return self._liveness_threshold
