"""Generic benchmark runner.

Reads samples (a stream of `Sample` records — bonafide or attack frames /
videos), pushes each through a configured pipeline, collects scores, and
produces a `BenchmarkResult` with all canonical FAS metrics filled in.

The runner is dataset- and pipeline-agnostic — adapters live in
`tests/benchmark/datasets/` and pipeline factories in
`tests/benchmark/pipelines/`.
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from src.metrics import classification_report, roc_curve

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """A single labelled FAS evaluation unit.

    For frame-level evaluation: one Sample = one frame.
    For session-level evaluation: one Sample = one video / capture.
    """

    sample_id: str
    is_bonafide: bool
    attack_type: str | None  # None for bonafide; otherwise "print", "replay", "mask", "ar_filter", ...
    payload: object  # frame ndarray, video path, etc. — runner passes to pipeline
    metadata: dict = field(default_factory=dict)


@dataclass
class ProtocolSpec:
    """An evaluation protocol — typically several per dataset.

    OULU-NPU defines 4 protocols (P1: train/test session split,
    P2: subject-disjoint, P3: PAI-disjoint, P4: cross-everything).
    SiW defines 3. CASIA-SURF has its own.
    """

    name: str
    description: str
    train_filter: Callable[[Sample], bool] | None = None
    test_filter: Callable[[Sample], bool] | None = None


@dataclass
class BenchmarkResult:
    dataset: str
    protocol: str
    pipeline_name: str
    n_samples: int
    elapsed_sec: float
    metrics: dict
    per_sample: list[dict] = field(default_factory=list)

    def to_paper_row(self) -> dict[str, str]:
        """Format the headline numbers as a single paper-table row."""
        m = self.metrics
        return {
            "dataset": self.dataset,
            "protocol": self.protocol,
            "method": self.pipeline_name,
            "APCER_max": f"{m['apcer_max']*100:.2f}%",
            "BPCER":     f"{m['bpcer']*100:.2f}%",
            "ACER":      f"{m['acer']*100:.2f}%",
            "EER":       f"{m['eer']*100:.2f}%",
            "AUC":       f"{m['auc']:.4f}",
            "N":         f"{self.n_samples}",
        }


def run_benchmark(
    samples: Iterable[Sample],
    score_fn: Callable[[Sample], float],
    *,
    dataset: str,
    protocol: str = "default",
    pipeline_name: str = "spoof-detector",
    progress_every: int = 100,
) -> BenchmarkResult:
    """Run a pipeline against a stream of samples.

    Args:
        samples: iterable of Sample
        score_fn: function that takes a Sample and returns a [0,1] live-ness score.
                  This is where the pipeline-of-the-day plugs in.
        dataset/protocol/pipeline_name: bookkeeping labels for results.

    Returns:
        BenchmarkResult with every canonical metric pre-computed.
    """
    t0 = time.perf_counter()
    scores: list[float] = []
    is_bonafide: list[bool] = []
    attack_types: list[str | None] = []
    per_sample: list[dict] = []

    n = 0
    for sample in samples:
        s = float(score_fn(sample))
        scores.append(s)
        is_bonafide.append(sample.is_bonafide)
        attack_types.append(sample.attack_type or "unknown" if not sample.is_bonafide else None)
        per_sample.append({
            "sample_id": sample.sample_id,
            "is_bonafide": sample.is_bonafide,
            "attack_type": sample.attack_type,
            "score": s,
        })
        n += 1
        if progress_every and n % progress_every == 0:
            logger.info("benchmark: processed %d samples", n)

    elapsed = time.perf_counter() - t0
    if n == 0:
        return BenchmarkResult(
            dataset=dataset,
            protocol=protocol,
            pipeline_name=pipeline_name,
            n_samples=0,
            elapsed_sec=elapsed,
            metrics={"error": "empty stream"},
        )

    report = classification_report(
        scores,
        is_bonafide,
        attack_types,
    )
    return BenchmarkResult(
        dataset=dataset,
        protocol=protocol,
        pipeline_name=pipeline_name,
        n_samples=n,
        elapsed_sec=elapsed,
        metrics=report,
        per_sample=per_sample,
    )


def write_paper_row_csv(results: list[BenchmarkResult], out_path: Path) -> None:
    """Append paper-table rows to a CSV that can be \\input{} into LaTeX."""
    import csv
    rows = [r.to_paper_row() for r in results]
    if not rows:
        return
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
