"""Benchmark harness: dataset-agnostic evaluation runner.

This package is the bridge between the spoof-detector engine and the
academic FAS benchmarks (OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof).

The benchmarks themselves are NOT bundled (each requires individual EULA
acceptance / institutional access). Adapters in `datasets/` know how to
walk the directory layout each dataset provides and emit the canonical
`Sample` records the runner consumes.

Public API:
    Sample, BenchmarkResult, run_benchmark, ProtocolSpec

To register a new dataset:
    1. Implement an adapter in `datasets/<name>.py` that yields Sample(s).
    2. Add the protocol spec in `protocols.py`.
    3. Run `python -m tests.benchmark.run --dataset <name>`.
"""
from tests.benchmark.runner import (
    Sample,
    BenchmarkResult,
    ProtocolSpec,
    run_benchmark,
)

__all__ = ["Sample", "BenchmarkResult", "ProtocolSpec", "run_benchmark"]
