"""Capture per-sample per-analyzer scores for downstream learning.

For each sample in the dataset, runs the full hybrid pipeline and writes
the per-analyzer raw scores (0-100) plus the label to a JSON file. That
file is the input to:

  - tests/benchmark/train_logistic_fuser.py — learns optimal fuser weights
  - tests/benchmark/calibration_sweep.py    — sweeps one analyzer's weight
  - tests/benchmark/quality_gate_eval.py    — filters by face-detection quality

Usage:
    python -m tests.benchmark.capture_analyzer_scores \\
        --dataset casia_fasd --root /tmp/fas_datasets/akahana_casiafasd/extracted \\
        --split train --out paper/figures/captures/casia_fasd_train.json

Output JSON shape:
    {
      "dataset": str,
      "split": str,
      "n_samples": int,
      "elapsed_sec": float,
      "per_sample": [
        {
          "sample_id": str,
          "is_bonafide": bool,
          "attack_type": str|null,
          "analyzer_scores": {analyzer_name: float[0,100]},
          "face_detected": bool,
          "n_faces": int,
          "face_area_px": int,
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def capture_one(sample, pipeline) -> dict | None:
    """Run pipeline.process on a sample, return per-analyzer score dict + face metadata."""
    from tests.benchmark.pipelines._common import load_frames

    frames = load_frames(sample, max_frames=1)
    if not frames:
        return None
    analysis = pipeline.process(frames[0])

    record: dict = {
        "sample_id": sample.sample_id,
        "is_bonafide": sample.is_bonafide,
        "attack_type": sample.attack_type,
        "n_faces": len(analysis.faces),
        "face_detected": bool(analysis.classifications),
    }

    if not analysis.classifications:
        record["analyzer_scores"] = {}
        record["face_area_px"] = 0
        return record

    # Pick highest-confidence face
    best_face_id = max(
        analysis.classifications.keys(),
        key=lambda fid: analysis.classifications[fid].confidence or 0.0,
    )
    best_cls = analysis.classifications[best_face_id]
    best_face = next((f for f in analysis.faces if f.face_id == best_face_id), None)

    record["analyzer_scores"] = {
        name: float(ar.score)
        for name, ar in best_cls.analyzer_results.items()
    }
    if best_face and best_face.bbox is not None:
        bbox = best_face.bbox
        record["face_area_px"] = int(
            (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        )
    else:
        record["face_area_px"] = 0
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=False)
    p.add_argument("--split", default=None,
                   help="Pass to dataset adapter if it supports splits (CASIA-FASD: train/test).")
    p.add_argument("--protocol", default="capture")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    from tests.benchmark.pipelines._common import build_hybrid_pipeline

    # Special-case: CASIA-FASD adapter takes split kwarg
    if args.dataset == "casia_fasd":
        from tests.benchmark.datasets.casia_fasd import iter_casia_fasd
        if args.root is None:
            raise SystemExit("--root required for casia_fasd")
        samples_iter = iter_casia_fasd(args.root, split=args.split or "test")
    else:
        from tests.benchmark.run import _load_adapter
        samples_iter = _load_adapter(args.dataset, args.root, args.protocol)

    if args.limit:
        from itertools import islice
        samples_iter = islice(samples_iter, args.limit)

    logger.info("building hybrid pipeline...")
    pipeline, _ = build_hybrid_pipeline()
    logger.info("walking samples...")

    out: list[dict] = []
    t0 = time.perf_counter()
    for i, sample in enumerate(samples_iter):
        if i % 100 == 0:
            logger.info("processed %d", i)
        record = capture_one(sample, pipeline)
        if record is not None:
            out.append(record)

    elapsed = time.perf_counter() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "split": args.split or args.protocol,
        "n_samples": len(out),
        "elapsed_sec": elapsed,
        "per_sample": out,
    }, indent=2, default=str))

    bf = sum(1 for r in out if r["is_bonafide"])
    detected = sum(1 for r in out if r["face_detected"])
    print(f"\n=== captured {len(out)} samples in {elapsed:.1f}s ===")
    print(f"  bonafide={bf}  attack={len(out)-bf}")
    print(f"  face_detected={detected} ({100*detected/len(out):.1f}%)")
    if out:
        analyzer_keys = sorted(out[0]["analyzer_scores"].keys()) if out[0].get("analyzer_scores") else []
        print(f"  analyzers captured: {analyzer_keys}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
