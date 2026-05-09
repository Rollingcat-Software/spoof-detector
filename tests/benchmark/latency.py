"""Per-stage latency benchmark.

Measures wall-clock time per pipeline stage over N frames, reporting
mean and p50/p95/p99 percentiles. This is paper Table 4.

Usage:
    python -m tests.benchmark.latency --pipeline hybrid --n-frames 200

Output:
    paper/figures/latency_<pipeline>.json — per-stage stats
    stdout — formatted table

The frames are sampled from the in-house bonafide set so we exercise the
real face-detection + analyzer code paths (instead of mocked inputs that
might short-circuit).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _load_test_frames(n: int) -> list[np.ndarray]:
    """Sample N test frames from the in-house bonafide set."""
    src = Path("data/in_house/bonafide")
    paths = sorted(src.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no test frames at {src} — run synthesize_attacks first")
    frames: list[np.ndarray] = []
    while len(frames) < n:
        for p in paths:
            if len(frames) >= n:
                break
            img = cv2.imread(str(p))
            if img is None:
                continue
            frames.append(img)
    return frames


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
    s = sorted(values)
    return {
        "mean": float(statistics.fmean(values)),
        "p50": float(s[len(s) // 2]),
        "p95": float(s[int(0.95 * len(s))]),
        "p99": float(s[int(0.99 * len(s))]),
        "n": len(values),
    }


def benchmark_pipeline(pipeline_name: str, n_frames: int) -> dict:
    """Run N frames through the named pipeline, recording per-stage times."""
    from tests.benchmark.pipelines._common import (
        build_image_pipeline,
        build_video_pipeline,
        build_hybrid_pipeline,
    )

    if pipeline_name == "image_only":
        pipeline, _ = build_image_pipeline()
    elif pipeline_name == "video_only":
        pipeline, _ = build_video_pipeline()
    elif pipeline_name == "hybrid":
        pipeline, _ = build_hybrid_pipeline()
    else:
        raise SystemExit(f"unknown pipeline: {pipeline_name}")

    frames = _load_test_frames(n_frames)
    logger.info("loaded %d test frames; starting benchmark...", len(frames))

    # Warm-up: 5 frames so MiniFASNet ONNX session compiles, MediaPipe initialises.
    for f in frames[:5]:
        pipeline.process(f)

    detect_ms: list[float] = []
    track_ms: list[float] = []
    face_analyze_ms: list[float] = []
    fuse_ms: list[float] = []
    total_ms: list[float] = []
    per_analyzer_ms: dict[str, list[float]] = {}

    for f in frames:
        t0 = time.perf_counter()
        analysis = pipeline.process(f)
        t1 = time.perf_counter()
        total_ms.append((t1 - t0) * 1000)

        # Per-analyzer breakdown is in analysis.classifications[*].analyzer_results
        for cls in analysis.classifications.values():
            for name, ar in cls.analyzer_results.items():
                per_analyzer_ms.setdefault(name, []).append(float(ar.elapsed_ms))

    out = {
        "pipeline": pipeline_name,
        "n_frames": len(frames),
        "total": _percentiles(total_ms),
        "per_analyzer": {
            name: _percentiles(vs) for name, vs in sorted(per_analyzer_ms.items())
        },
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline", default="hybrid",
                   choices=["image_only", "video_only", "hybrid"])
    p.add_argument("--n-frames", type=int, default=200)
    p.add_argument("--out", default="paper/figures")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    result = benchmark_pipeline(args.pipeline, args.n_frames)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"latency_{args.pipeline}.json"
    out_file.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n=== Latency: pipeline={result['pipeline']}, N={result['n_frames']} frames ===\n")
    t = result["total"]
    print(f"{'TOTAL':<24s} mean={t['mean']:6.1f} ms   p50={t['p50']:6.1f}   p95={t['p95']:6.1f}   p99={t['p99']:6.1f}")
    print()
    print(f"{'per-analyzer':<24s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'N':>6s}")
    print("-" * 70)
    for name, st in sorted(result["per_analyzer"].items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {name:<22s} {st['mean']:7.1f} {st['p50']:7.1f} {st['p95']:7.1f} {st['p99']:7.1f} {st['n']:>6d}")

    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
