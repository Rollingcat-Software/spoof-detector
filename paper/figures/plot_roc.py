"""Render per-dataset ROC curves from the JSON results.

Usage:
    python paper/figures/plot_roc.py

Writes:
    paper/figures/roc_<dataset>_<protocol>.png

Each plot overlays minifasnet_only / image_only / hybrid for the same
(dataset, protocol). Datasets / protocols with only one pipeline run get
a single curve. Bootstrap-CI band rendered around the curve when the JSON
includes per-sample scores (used to compute the band on-the-fly).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

FIG_DIR = Path(__file__).parent
RESULTS_GLOB = "results_*.json"

# Method-specific styling so reviewers can tell curves apart at a glance
STYLES = {
    "minifasnet_only": {"label": "minifasnet_only (baseline)", "linestyle": "--"},
    "image_only":      {"label": "image_only (Ahmet's track)", "linestyle": "-"},
    "video_only":      {"label": "video_only (Aysenur's track)", "linestyle": ":"},
    "hybrid":          {"label": "hybrid (published)", "linestyle": "-", "linewidth": 2.5},
}


def load_grouped_results() -> dict[tuple[str, str], list[dict]]:
    """Group results JSONs by (dataset, protocol)."""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in sorted(FIG_DIR.glob(RESULTS_GLOB)):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        if "per_sample" not in data or not data["per_sample"]:
            continue
        key = (data["dataset"], data["protocol"])
        grouped[key].append(data)
    return grouped


def render_roc_for_group(dataset: str, protocol: str, runs: list[dict]) -> Path | None:
    """Render one PNG with all pipelines overlaid for the given (dataset, protocol)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping ROC render. `pip install matplotlib`.")
        return None

    from src.metrics.standard import roc_curve

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
    has_data = False
    for run in runs:
        s = run["per_sample"]
        scores = [x["score"] for x in s]
        is_bf = [x["is_bonafide"] for x in s]
        types = [x.get("attack_type") for x in s]
        n_bf = sum(is_bf)
        n_at = len(is_bf) - n_bf
        if n_bf == 0 or n_at == 0:
            continue
        roc = roc_curve(scores, is_bf, types, n_points=120)
        far = [pt.far for pt in roc.points]
        tpr = [1 - pt.frr for pt in roc.points]
        # Sort by far for clean line
        order = sorted(range(len(far)), key=lambda i: far[i])
        far = [far[i] for i in order]
        tpr = [tpr[i] for i in order]
        style = STYLES.get(run["pipeline_name"], {})
        ax.plot(far, tpr,
                label=f"{style.get('label', run['pipeline_name'])} (AUC={roc.auc:.3f}, EER={roc.eer*100:.1f}%)",
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 1.5),
                alpha=0.9)
        has_data = True

    if not has_data:
        plt.close(fig)
        return None

    # Reference: random classifier
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8, alpha=0.5,
            label="random (AUC=0.500)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("APCER (false-accept rate of attacks)")
    ax.set_ylabel("1 − BPCER (true-accept rate of bonafide)")
    ax.set_title(f"{dataset} / {protocol}")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()

    out_path = FIG_DIR / f"roc_{dataset}_{protocol}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    grouped = load_grouped_results()
    if not grouped:
        print("no results JSONs to plot — run tests.benchmark.run first")
        return
    for (dataset, protocol), runs in sorted(grouped.items()):
        out = render_roc_for_group(dataset, protocol, runs)
        if out:
            print(f"wrote {out}  ({len(runs)} pipeline(s))")
    print(f"\nrendered {len(grouped)} ROC group(s)")


if __name__ == "__main__":
    main()
