# This file is a copy from biometric-processor/app/infrastructure/ml/liveness/threshold_calibrator.py
# Source SHA: 9d17359e18186abc317f4fe4b903ab8e82626297 (bio main as of 2026-05-09).
# Synced: 2026-05-09. DO NOT edit here without syncing back to biometric-processor.

"""Utilities for calibrating liveness thresholds from labeled score logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """Calibration output summary."""

    optimal_threshold: float
    far: float
    frr: float
    eer: float
    sample_count: int
    real_count: int
    spoof_count: int
    target_far: float
    strategy: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialize calibration result."""
        return {
            "optimal_threshold": round(self.optimal_threshold, 4),
            "far": round(self.far, 6),
            "frr": round(self.frr, 6),
            "eer": round(self.eer, 6),
            "sample_count": self.sample_count,
            "real_count": self.real_count,
            "spoof_count": self.spoof_count,
            "target_far": self.target_far,
            "strategy": self.strategy,
        }


class ThresholdCalibrator:
    """Calibrate liveness thresholds from labeled JSONL logs."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)

    def load_scores(self) -> tuple[list[float], list[float]]:
        """Load real and spoof scores from JSONL logs.

        Accepted labels:
        - real/live/genuine/bonafide
        - spoof/fake/attack/imposter

        Accepted score keys:
        - score
        - liveness_score
        """
        real_scores: list[float] = []
        spoof_scores: list[float] = []

        with self.log_path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                entry = json.loads(stripped)
                label = self._normalize_label(entry)
                score = self._extract_score(entry)

                if label is None or score is None:
                    continue

                if label == "real":
                    real_scores.append(score)
                elif label == "spoof":
                    spoof_scores.append(score)
                else:
                    raise ValueError(
                        f"Unsupported normalized label '{label}' at line {line_number}"
                    )

        return real_scores, spoof_scores

    def find_optimal_threshold(
        self,
        real_scores: list[float],
        spoof_scores: list[float],
        target_far: float = 0.01,
        strategy: str = "target_far",
    ) -> ThresholdCalibrationResult:
        """Find an optimal threshold from labeled score sets.

        Strategies:
        - ``target_far``: pick the lowest threshold that satisfies the FAR target,
          then minimize FRR among those candidates.
        - ``eer``: minimize absolute FAR/FRR distance.
        """
        self._validate_inputs(real_scores, spoof_scores, target_far)

        thresholds = self._candidate_thresholds(real_scores, spoof_scores)
        evaluations = [
            self._evaluate_threshold(threshold, real_scores, spoof_scores)
            for threshold in thresholds
        ]

        if strategy == "target_far":
            result = self._select_target_far_threshold(evaluations, target_far)
        elif strategy == "eer":
            result = self._select_eer_threshold(evaluations, target_far)
        else:
            raise ValueError("strategy must be 'target_far' or 'eer'")

        return ThresholdCalibrationResult(
            optimal_threshold=result["threshold"],
            far=result["far"],
            frr=result["frr"],
            eer=result["eer"],
            sample_count=len(real_scores) + len(spoof_scores),
            real_count=len(real_scores),
            spoof_count=len(spoof_scores),
            target_far=target_far,
            strategy=strategy,
        )

    def write_env_threshold(
        self,
        env_path: str | Path,
        threshold: float,
        key: str = "LIVENESS_THRESHOLD",
    ) -> None:
        """Update or append a threshold entry in an env file."""
        path = Path(env_path)
        threshold_value = f"{threshold:.4f}".rstrip("0").rstrip(".")
        new_line = f"{key}={threshold_value}"

        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            lines = []

        replaced = False
        updated_lines: list[str] = []
        for line in lines:
            if line.startswith(f"{key}="):
                updated_lines.append(new_line)
                replaced = True
            else:
                updated_lines.append(line)

        if not replaced:
            if updated_lines and updated_lines[-1] != "":
                updated_lines.append("")
            updated_lines.append(new_line)

        path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    def _validate_inputs(
        self,
        real_scores: list[float],
        spoof_scores: list[float],
        target_far: float,
    ) -> None:
        if not real_scores:
            raise ValueError("No real scores found in calibration data")
        if not spoof_scores:
            raise ValueError("No spoof scores found in calibration data")
        if not 0.0 <= target_far <= 1.0:
            raise ValueError("target_far must be between 0.0 and 1.0")

    def _candidate_thresholds(
        self,
        real_scores: list[float],
        spoof_scores: list[float],
    ) -> list[float]:
        combined = sorted(set(real_scores + spoof_scores))
        if not combined:
            return [70.0]

        epsilon = 1e-6
        candidates = [max(0.0, combined[0] - epsilon), *combined, min(100.0, combined[-1] + epsilon)]
        return sorted(set(candidates))

    def _evaluate_threshold(
        self,
        threshold: float,
        real_scores: Iterable[float],
        spoof_scores: Iterable[float],
    ) -> dict[str, float]:
        real_scores = list(real_scores)
        spoof_scores = list(spoof_scores)
        far = sum(1 for score in spoof_scores if score >= threshold) / max(len(spoof_scores), 1)
        frr = sum(1 for score in real_scores if score < threshold) / max(len(real_scores), 1)
        eer = (far + frr) / 2.0
        return {
            "threshold": threshold,
            "far": far,
            "frr": frr,
            "eer": eer,
            "eer_distance": abs(far - frr),
        }

    def _select_target_far_threshold(
        self,
        evaluations: list[dict[str, float]],
        target_far: float,
    ) -> dict[str, float]:
        acceptable = [item for item in evaluations if item["far"] <= target_far]
        if acceptable:
            return min(
                acceptable,
                key=lambda item: (item["frr"], item["far"], item["threshold"]),
            )

        return min(
            evaluations,
            key=lambda item: (abs(item["far"] - target_far), item["frr"], item["threshold"]),
        )

    def _select_eer_threshold(
        self,
        evaluations: list[dict[str, float]],
        target_far: float,
    ) -> dict[str, float]:
        del target_far
        return min(
            evaluations,
            key=lambda item: (item["eer_distance"], item["frr"], item["threshold"]),
        )

    def _normalize_label(self, entry: dict[str, Any]) -> str | None:
        raw_label = entry.get("label")
        if raw_label is None:
            return None

        label = str(raw_label).strip().lower()
        if label in {"real", "live", "genuine", "bonafide", "bona_fide"}:
            return "real"
        if label in {"spoof", "fake", "attack", "imposter", "impostor"}:
            return "spoof"
        return None

    def _extract_score(self, entry: dict[str, Any]) -> float | None:
        score = entry.get("score", entry.get("liveness_score"))
        if score is None:
            return None
        score = float(score)
        if not 0.0 <= score <= 100.0:
            raise ValueError(f"Score out of range: {score}")
        return score
