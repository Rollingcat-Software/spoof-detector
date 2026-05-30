"""1-D calibration sweep over an analyzer's fuser weight.

Paper §5.4 references this script. It sweeps the weight of one analyzer
from 0.0 to 1.0 in 0.05 steps (configurable), re-fuses the per-sample
analyzer scores using the swept weight, and reports ACER + AUC at each
step.

Usage:
    # 1. Run the leave-one-out captures once (gives us per-analyzer scores per sample)
    python -m tests.benchmark.ablation_leave_one_out \\
        --dataset in_house --root data/in_house_replay --protocol replay_n100

    # 2. Sweep one analyzer's weight
    python -m tests.benchmark.calibration_sweep \\
        --capture paper/figures/ablation_loo_in_house_replay_n100.json \\
        --analyzer texture \\
        --out paper/figures

Outputs:
    paper/figures/calibration_sweep_<analyzer>.json — per-step (weight, ACER, AUC)
    paper/figures/calibration_sweep_<analyzer>.png  — line chart

The PNG visualises the ACER curve so reviewers can read off the optimal
weight directly.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _sweep_one(per_sample: list[dict], analyzer: str, weights: list[float]) -> list[dict]:
    """For each weight value, override the named analyzer's weight and recompute ACER + AUC."""
    from src.metrics import classification_report
    from src.infrastructure.fusion.multi_class_fuser import DEFAULT_ANALYZER_WEIGHTS
    from src.domain.models import SpoofCategory
    from src.domain.taxonomy import SPOOF_SIGNAL_MAP

    rows = []
    is_bonafide = [s["is_bonafide"] for s in per_sample]
    attack_types = [s["attack_type"] or "unknown" for s in per_sample]

    for w in weights:
        weights_dict = dict(DEFAULT_ANALYZER_WEIGHTS)
        weights_dict[analyzer] = float(w)

        scores: list[float] = []
        for s in per_sample:
            cat_evidence = {cat: 0.0 for cat in SpoofCategory}
            for name, score in s["analyzer_scores"].items():
                weight = weights_dict.get(name, 0.0)
                if weight <= 0:
                    continue
                score_norm = score / 100.0
                real_evidence = weight * score_norm
                spoof_evidence = weight * (1 - score_norm)
                cat_evidence[SpoofCategory.REAL] += real_evidence
                signal_map = SPOOF_SIGNAL_MAP.get(name, {})
                for cat, sw in signal_map.items():
                    cat_evidence[cat] += spoof_evidence * sw
            max_e = max(cat_evidence.values())
            exps = {c: np.exp(e - max_e) for c, e in cat_evidence.items()}
            z = sum(exps.values()) or 1e-9
            scores.append(float(exps[SpoofCategory.REAL] / z))

        # Sweep compares ACER across weight values on one fixed capture; no Dev
        # split is wired, so this uses the (biased) EER-on-test operating point —
        # opt-in. The swept curve is a heuristic visualisation, not a paper headline.
        report = classification_report(
            scores, is_bonafide, attack_types, allow_test_set_threshold=True
        )
        rows.append({
            "weight": float(w),
            "acer": float(report["acer"]),
            "apcer": float(report["apcer_max"]),
            "bpcer": float(report["bpcer"]),
            "eer":  float(report["eer"]),
            "auc":  float(report["auc"]),
        })
    return rows


def _plot(rows: list[dict], analyzer: str, out_dir: Path) -> Path | None:
    """Render the ACER + AUC sweep as a 2-axis line plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return None

    weights = [r["weight"] for r in rows]
    acer_vals = [r["acer"] * 100 for r in rows]
    auc_vals = [r["auc"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    color1 = "#d35400"
    color2 = "#2c3e50"

    line1, = ax1.plot(weights, acer_vals, color=color1, marker="o", linewidth=1.6, label="ACER")
    ax1.set_xlabel(f"weight of `{analyzer}` analyzer")
    ax1.set_ylabel("ACER (%)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    line2, = ax2.plot(weights, auc_vals, color=color2, marker="s", linewidth=1.6, linestyle="--", label="AUC")
    ax2.set_ylabel("AUC", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Annotate optimum
    best_idx = int(np.argmin(acer_vals))
    ax1.axvline(weights[best_idx], color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.annotate(
        f"min ACER {acer_vals[best_idx]:.1f}%\nat weight {weights[best_idx]:.2f}",
        xy=(weights[best_idx], acer_vals[best_idx]),
        xytext=(weights[best_idx] + 0.05, acer_vals[best_idx] + 1.5),
        fontsize=8, color="dimgrey",
        arrowprops={"arrowstyle": "->", "color": "dimgrey", "alpha": 0.7},
    )

    ax1.set_title(f"calibration sweep — `{analyzer}` weight")
    ax1.grid(True, alpha=0.25, linestyle="--")
    fig.legend(handles=[line1, line2], loc="upper right", fontsize=9)
    fig.tight_layout()

    out_path = out_dir / f"calibration_sweep_{analyzer}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--capture", required=True,
                   help="JSON from ablation_leave_one_out.py with per-sample analyzer scores.")
    p.add_argument("--analyzer", required=True,
                   help="Analyzer name to sweep (e.g. texture, moire, ar_filter).")
    p.add_argument("--w-min", type=float, default=0.0)
    p.add_argument("--w-max", type=float, default=1.0)
    p.add_argument("--w-step", type=float, default=0.05)
    p.add_argument("--out", default="paper/figures")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    capture = json.loads(Path(args.capture).read_text())
    if "leave_one_out" not in capture:
        # The leave_one_out output uses a different shape; re-derive from baseline run.
        logger.error("capture must be the output of ablation_leave_one_out.py")
        return 2

    # The ablation script doesn't persist per-sample analyzer scores by default —
    # we need to re-run the capture step here.
    logger.info("capture file structure check passed; re-running per-sample capture for sweep")
    from tests.benchmark.run import _load_adapter
    from tests.benchmark.ablation_leave_one_out import _full_pipeline_with_per_analyzer_capture

    # The capture only knows the dataset+protocol, so we re-walk the same samples.
    dataset = capture["dataset"]
    protocol = capture["protocol"]
    # Best-guess root from earlier ablation runs:
    candidate_roots = {
        "in_house": ["data/in_house", "data/in_house_replay"],
        "casia_fasd": ["/tmp/fas_datasets/akahana_casiafasd/extracted"],
    }
    root = None
    for r in candidate_roots.get(dataset, []):
        if Path(r).exists():
            root = r
            break

    samples = list(_load_adapter(dataset, root, protocol))
    logger.info("walking %d samples through the pipeline once for per-analyzer capture...", len(samples))
    per_sample = _full_pipeline_with_per_analyzer_capture(samples)

    # Sweep
    weights = [round(args.w_min + i * args.w_step, 4)
               for i in range(int((args.w_max - args.w_min) / args.w_step) + 1)]
    rows = _sweep_one(per_sample, args.analyzer, weights)

    # Output
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"calibration_sweep_{args.analyzer}.json"
    out_json.write_text(json.dumps({
        "analyzer": args.analyzer,
        "dataset": dataset,
        "protocol": protocol,
        "n_samples": len(per_sample),
        "rows": rows,
    }, indent=2))
    print(f"wrote {out_json}")

    plot_path = _plot(rows, args.analyzer, out_dir)
    if plot_path:
        print(f"wrote {plot_path}")

    # Print sweep table
    best = min(rows, key=lambda r: r["acer"])
    print(f"\n=== Sweep: `{args.analyzer}` weight 0.0 → 1.0 (step {args.w_step}) ===\n")
    print(f"{'weight':>8s} {'ACER':>8s} {'EER':>8s} {'AUC':>8s}")
    print("-" * 40)
    for r in rows:
        marker = " *" if r is best else ""
        print(f"{r['weight']:>8.2f} {r['acer']*100:>7.2f}% {r['eer']*100:>7.2f}% {r['auc']:>8.4f}{marker}")
    print(f"\nOptimal weight: {best['weight']:.2f}  →  ACER {best['acer']*100:.2f}%, AUC {best['auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
