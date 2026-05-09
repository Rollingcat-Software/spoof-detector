# This file is a copy from biometric-processor/app/infrastructure/ml/liveness/texture_liveness_detector.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Texture-based liveness detector using image analysis heuristics.

This detector analyzes image properties to detect potential spoofing attacks
such as printed photos or screen displays. It uses multiple heuristics:
- Texture analysis (Laplacian variance)
- Color distribution analysis
- Frequency domain analysis
- Moiré pattern detection
"""

import logging

import cv2
import numpy as np

from app.domain.entities.liveness_result import LivenessResult
from app.domain.interfaces.liveness_detector import ILivenessDetector

logger = logging.getLogger(__name__)


class TextureLivenessDetector(ILivenessDetector):
    """Liveness detector using texture and image analysis.

    This implementation uses multiple heuristics to detect spoofing:
    1. Texture variance - real faces have more texture variation
    2. Color naturalness - screens have different color distributions
    3. Frequency analysis - printed photos have different frequency patterns
    4. Moiré detection - screens often show moiré patterns
    """

    def __init__(
        self,
        texture_threshold: float = 100.0,
        color_threshold: float = 0.3,
        frequency_threshold: float = 0.5,
        liveness_threshold: float = 60.0,
    ) -> None:
        """Initialize texture-based liveness detector.

        Args:
            texture_threshold: Minimum Laplacian variance for live face
            color_threshold: Maximum color distribution deviation
            frequency_threshold: Frequency analysis threshold
            liveness_threshold: Overall score threshold for liveness
        """
        self._texture_threshold = texture_threshold
        self._color_threshold = color_threshold
        self._frequency_threshold = frequency_threshold
        self._liveness_threshold = liveness_threshold

        logger.info(
            f"TextureLivenessDetector initialized with thresholds: "
            f"texture={texture_threshold}, color={color_threshold}, "
            f"frequency={frequency_threshold}, liveness={liveness_threshold}"
        )

    async def check_liveness(self, image: np.ndarray) -> LivenessResult:
        """Check if image shows a live person using texture analysis.

        Args:
            image: Face image as numpy array (BGR format)

        Returns:
            LivenessResult with liveness determination
        """
        return await self.detect(image)

    async def detect(
        self,
        image: np.ndarray,
        challenge: str = "texture_analysis",
    ) -> LivenessResult:
        """Detect liveness using texture and image analysis.

        Args:
            image: Face image as numpy array (BGR format)
            challenge: Challenge type (default: texture_analysis)

        Returns:
            LivenessResult with liveness determination
        """
        logger.info("Starting texture-based liveness detection")

        # Calculate individual scores
        texture_score = self._calculate_texture_score(image)
        color_score = self._calculate_color_score(image)
        frequency_score = self._calculate_frequency_score(image)
        moire_score = self._calculate_moire_score(image)

        # Weight the scores (texture is most important)
        weights = {
            "texture": 0.35,
            "color": 0.25,
            "frequency": 0.25,
            "moire": 0.15,
        }

        combined_score = (
            texture_score * weights["texture"]
            + color_score * weights["color"]
            + frequency_score * weights["frequency"]
            + moire_score * weights["moire"]
        )

        # Normalize to 0-100
        liveness_score = min(100.0, max(0.0, combined_score))

        is_live = liveness_score >= self._liveness_threshold

        logger.info(
            f"Liveness detection complete: score={liveness_score:.2f}, "
            f"is_live={is_live} (texture={texture_score:.2f}, "
            f"color={color_score:.2f}, frequency={frequency_score:.2f}, "
            f"moire={moire_score:.2f})"
        )

        return LivenessResult(
            is_live=is_live,
            liveness_score=liveness_score,
            challenge=challenge,
            challenge_completed=True,
            details={
                "texture": texture_score,
                "color": color_score,
                "frequency": frequency_score,
                "moire": moire_score,
            },
        )

    def _calculate_texture_score(self, image: np.ndarray) -> float:
        """Calculate texture score using Laplacian variance.

        Real faces have more texture variation than printed photos.

        Args:
            image: Input image (BGR)

        Returns:
            Texture score (0-100)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Normalize score: higher variance = more texture = more likely live
        # Typical range: 50-500 for real faces, 10-100 for printed
        if variance >= self._texture_threshold:
            score = min(100.0, 50.0 + (variance - self._texture_threshold) * 0.2)
        else:
            score = max(0.0, (variance / self._texture_threshold) * 50.0)

        return score

    def _calculate_color_score(self, image: np.ndarray) -> float:
        """Calculate color naturalness score.

        Screens and printed photos often have unnatural color distributions.

        Args:
            image: Input image (BGR)

        Returns:
            Color naturalness score (0-100)
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Analyze saturation distribution
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation)

        # Analyze value (brightness) distribution
        value = hsv[:, :, 2]
        val_std = np.std(value)

        # Real faces have moderate saturation and varied brightness
        # Screens often have high saturation, printed photos have low variation

        # Ideal ranges for real faces
        ideal_sat_mean = 80  # Moderate saturation
        ideal_val_std = 50  # Good brightness variation

        # Calculate deviations
        sat_deviation = abs(sat_mean - ideal_sat_mean) / 128.0
        val_deviation = abs(val_std - ideal_val_std) / 64.0

        # Lower deviation = higher score
        combined_deviation = (sat_deviation + val_deviation) / 2.0

        if combined_deviation <= self._color_threshold:
            score = 100.0 - (combined_deviation / self._color_threshold) * 30.0
        else:
            score = max(0.0, 70.0 - (combined_deviation - self._color_threshold) * 100.0)

        return score

    def _calculate_frequency_score(self, image: np.ndarray) -> float:
        """Calculate frequency domain score.

        Printed photos often have different frequency characteristics
        due to printing patterns.

        Args:
            image: Input image (BGR)

        Returns:
            Frequency score (0-100)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Calculate ratio of high to low frequencies
        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2

        # Low frequency region (center)
        low_freq_region = magnitude[
            center_row - rows // 8 : center_row + rows // 8,
            center_col - cols // 8 : center_col + cols // 8,
        ]

        # High frequency region (outer)
        high_freq_mask = np.ones_like(magnitude, dtype=bool)
        high_freq_mask[
            center_row - rows // 4 : center_row + rows // 4,
            center_col - cols // 4 : center_col + cols // 4,
        ] = False
        high_freq_region = magnitude[high_freq_mask]

        # Calculate ratio
        low_mean = np.mean(low_freq_region) + 1e-6
        high_mean = np.mean(high_freq_region) + 1e-6
        freq_ratio = high_mean / low_mean

        # Real faces have balanced frequencies
        # Printed photos often have higher high-frequency content (printing dots)
        if freq_ratio < self._frequency_threshold:
            score = 100.0 - (1.0 - freq_ratio / self._frequency_threshold) * 40.0
        elif freq_ratio > self._frequency_threshold * 2:
            # Too much high frequency = possibly printed
            score = max(0.0, 60.0 - (freq_ratio - self._frequency_threshold * 2) * 50.0)
        else:
            score = 80.0

        return score

    def _calculate_moire_score(self, image: np.ndarray) -> float:
        """Detect moiré patterns that indicate screen display.

        Screens often produce moiré patterns when photographed.

        Args:
            image: Input image (BGR)

        Returns:
            Moiré detection score (0-100, higher = no moiré = more likely live)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply band-pass filter to detect periodic patterns
        # Moiré patterns typically appear as regular, high-frequency patterns

        # Use multiple Gabor filters at different orientations
        moire_detected = 0.0

        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=5.0,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0,
            )

            # Apply filter
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)

            # Check for strong periodic response
            response_std = np.std(filtered)
            if response_std > 30:  # Strong periodic pattern detected
                moire_detected += 0.25

        # Higher moiré detection = lower liveness score
        score = 100.0 - (moire_detected * 100.0)

        return score

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
        return "texture_analysis"

    def get_liveness_threshold(self) -> float:
        """Get the threshold for considering result as live.

        Returns:
            Liveness score threshold (0-100)
        """
        return self._liveness_threshold
