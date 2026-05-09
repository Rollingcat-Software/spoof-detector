"""End-to-end benchmark runner — one command produces paper-ready numbers.

Usage:
    python -m tests.benchmark.run --dataset oulu_npu --root /data/oulu --protocol P1
    python -m tests.benchmark.run --dataset siw       --root /data/siw
    python -m tests.benchmark.run --dataset in_house

Outputs:
    paper/figures/results_<dataset>_<protocol>.json — full per-sample log
    paper/figures/results_<dataset>_<protocol>.csv  — paper table row

The pipeline used is `HybridImageVideoPipeline` from
`tests.benchmark.pipelines.hybrid` — Ahmet's image-level analyzers fused
with Aysenur's video-level analyzers (rPPG / blink / screen-replay).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tests.benchmark.runner import run_benchmark, write_paper_row_csv

logger = logging.getLogger(__name__)


def _load_adapter(dataset: str, root: str | None, protocol: str):
    """Lazy-load the chosen adapter so missing optional deps don't break -h."""
    if dataset == "oulu_npu":
        from tests.benchmark.datasets.oulu_npu import iter_oulu_npu
        if root is None:
            raise SystemExit("--root required for oulu_npu")
        return iter_oulu_npu(root)
    if dataset == "siw":
        from tests.benchmark.datasets.siw import iter_siw
        if root is None:
            raise SystemExit("--root required for siw")
        return iter_siw(root)
    if dataset == "casia_surf":
        from tests.benchmark.datasets.casia_surf import iter_casia_surf
        if root is None:
            raise SystemExit("--root required for casia_surf")
        return iter_casia_surf(root)
    if dataset == "celeba_spoof":
        from tests.benchmark.datasets.celeba_spoof import iter_celeba_spoof
        if root is None:
            raise SystemExit("--root required for celeba_spoof")
        return iter_celeba_spoof(root)
    if dataset == "in_house":
        from tests.benchmark.datasets.in_house import iter_in_house
        return iter_in_house(root)
    raise SystemExit(f"unknown dataset: {dataset}")


def _load_pipeline(name: str):
    """Lazy-load the pipeline factory."""
    if name == "hybrid":
        from tests.benchmark.pipelines.hybrid import score_sample
        return score_sample
    if name == "image_only":
        from tests.benchmark.pipelines.image_only import score_sample
        return score_sample
    if name == "video_only":
        from tests.benchmark.pipelines.video_only import score_sample
        return score_sample
    if name == "minifasnet_only":
        from tests.benchmark.pipelines.minifasnet_only import score_sample
        return score_sample
    raise SystemExit(f"unknown pipeline: {name}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a paper-grade FAS benchmark.")
    p.add_argument("--dataset", required=True,
                   choices=["oulu_npu", "siw", "casia_surf", "celeba_spoof", "in_house"])
    p.add_argument("--root", help="Path to dataset root.")
    p.add_argument("--protocol", default="default", help="Protocol name (dataset-specific).")
    p.add_argument("--pipeline", default="hybrid",
                   choices=["hybrid", "image_only", "video_only", "minifasnet_only"])
    p.add_argument("--out", default="paper/figures",
                   help="Output directory for results.")
    p.add_argument("-v", "--verbose", action="count", default=0)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    samples = _load_adapter(args.dataset, args.root, args.protocol)
    score_fn = _load_pipeline(args.pipeline)

    result = run_benchmark(
        samples,
        score_fn,
        dataset=args.dataset,
        protocol=args.protocol,
        pipeline_name=args.pipeline,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"results_{args.dataset}_{args.protocol}_{args.pipeline}"

    (out_dir / f"{stem}.json").write_text(json.dumps({
        "dataset": result.dataset,
        "protocol": result.protocol,
        "pipeline_name": result.pipeline_name,
        "n_samples": result.n_samples,
        "elapsed_sec": result.elapsed_sec,
        "metrics": result.metrics,
        "per_sample": result.per_sample,
    }, indent=2, default=str))

    write_paper_row_csv([result], out_dir / f"{stem}.csv")

    print(json.dumps(result.to_paper_row(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
