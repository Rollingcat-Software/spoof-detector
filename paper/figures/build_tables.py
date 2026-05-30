"""Auto-build paper tables from benchmark JSON artefacts.

After running `python -m tests.benchmark.run --dataset X --protocol P` for
each (dataset, protocol, pipeline) triple, this script reads the per-run
JSON in `paper/figures/results_*.json` and emits:

  paper/figures/table1_headline.md         (§7.1 main table)
  paper/figures/table2_celeba_per_type.md  (§7.2 per-spoof-type)
  paper/figures/table3_cross_dataset.md    (§7.4 generalization)
  paper/figures/table5_ablation_tracks.md  (§8.1 image vs video vs hybrid)
  paper/figures/table9_active_challenges.md (§8.5 active layer)

Sections that are not yet populated (i.e. no JSON in figures/) get an
explicit "no data" placeholder so the paper draft remains buildable.

Usage:
  python paper/figures/build_tables.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

FIG_DIR = Path(__file__).parent
RESULTS_GLOB = "results_*.json"


def load_results() -> list[dict]:
    out: list[dict] = []
    for p in sorted(FIG_DIR.glob(RESULTS_GLOB)):
        try:
            data = json.loads(p.read_text())
            data["_filename"] = p.name
            out.append(data)
        except json.JSONDecodeError:
            logger.warning("skipping malformed JSON: %s", p)
    return out


def fmt_pct(x: float | None) -> str:
    return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else "—"


def fmt_dec(x: float | None, places: int = 4) -> str:
    return f"{x:.{places}f}" if isinstance(x, (int, float)) else "—"


# In-house synthetic same-source protocols are EXCLUDED from the headline table.
# Their attacks are gamma/blur/moire re-renders of the bona-fide image and their
# ACER was computed with the threshold chosen on the test set — a leaked,
# unreproducible "0% ACER / 100%" result. They were withdrawn 2026-05-29 (see
# paper/figures/WITHDRAWN_in_house_synthetic_results.md). The headline is the
# zero-shot public-dataset evaluation only. Do NOT re-add them here.
_HEADLINE_EXCLUDED_PROTOCOLS = {
    ("in_house", "clean"),
    ("in_house", "default"),
    ("in_house", "replay_only"),
    ("in_house", "replay_only_v2"),
}


def build_table_1(results: list[dict]) -> str:
    """§7.1 — headline numbers (zero-shot public datasets; in-house synthetic excluded)."""
    lines = [
        "# Table 1 — Headline ACER / EER / AUC across datasets",
        "",
        "_In-house synthetic same-source protocols are intentionally excluded — see_",
        "_`WITHDRAWN_in_house_synthetic_results.md`. Headline = zero-shot public datasets._",
        "",
        "| Dataset | Protocol | Method | APCER | BPCER | ACER | EER | AUC | N |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not results:
        lines.append("| _no data — run `tests.benchmark.run` first_ |")
        return "\n".join(lines)
    for r in results:
        m = r["metrics"]
        if not isinstance(m, dict) or "apcer_max" not in m:
            continue
        if (r.get("dataset"), r.get("protocol")) in _HEADLINE_EXCLUDED_PROTOCOLS:
            continue
        lines.append("| {dataset} | {protocol} | {method} | {apcer} | {bpcer} | {acer} | {eer} | {auc} | {n} |".format(
            dataset=r.get("dataset", "—"),
            protocol=r.get("protocol", "—"),
            method=r.get("pipeline_name", "—"),
            apcer=fmt_pct(m.get("apcer_max")),
            bpcer=fmt_pct(m.get("bpcer")),
            acer=fmt_pct(m.get("acer")),
            eer=fmt_pct(m.get("eer")),
            auc=fmt_dec(m.get("auc")),
            n=r.get("n_samples", "—"),
        ))
    return "\n".join(lines) + "\n"


def build_table_2(results: list[dict]) -> str:
    """§7.2 — CelebA-Spoof per-spoof-type APCER for the hybrid pipeline."""
    lines = [
        "# Table 2 — CelebA-Spoof per-spoof-type APCER (hybrid pipeline)",
        "",
        "| Spoof type | APCER |",
        "|---|---:|",
    ]
    matching = [r for r in results
                if r["dataset"] == "celeba_spoof" and r["pipeline_name"] == "hybrid"]
    if not matching:
        lines.append("| _no celeba_spoof+hybrid run yet_ | — |")
        return "\n".join(lines)
    per_type = matching[0]["metrics"].get("apcer_per_type", {})
    if not per_type:
        lines.append("| _empty per-type breakdown_ | — |")
        return "\n".join(lines)
    for spoof, value in sorted(per_type.items()):
        lines.append(f"| {spoof} | {fmt_pct(value)} |")
    macro = sum(per_type.values()) / len(per_type)
    lines.append(f"| **macro avg** | **{fmt_pct(macro)}** |")
    return "\n".join(lines) + "\n"


def build_table_5_ablation(results: list[dict]) -> str:
    """§8.1 — image_only vs video_only vs hybrid on the same dataset/protocol."""
    lines = [
        "# Table 5 — Ablation: image_only vs video_only vs hybrid",
        "",
        "| Dataset | Protocol | Pipeline | APCER | BPCER | ACER | EER | AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    # Find any dataset/protocol where we have all 3 methods
    methods: dict[tuple, set] = {}
    for r in results:
        if "pipeline_name" not in r:
            continue  # e.g. active-challenge smoke runs have no pipeline_name
        methods.setdefault((r.get("dataset"), r.get("protocol")), set()).add(r["pipeline_name"])
    triples = {k for k, v in methods.items() if {"image_only", "video_only", "hybrid"} <= v}
    if not triples:
        lines.append("| _no dataset has all three pipelines run yet_ |")
        return "\n".join(lines)
    for (ds, proto) in sorted(triples):
        for pipeline in ("image_only", "video_only", "hybrid"):
            for r in results:
                if (r["dataset"], r["protocol"], r["pipeline_name"]) == (ds, proto, pipeline):
                    m = r["metrics"]
                    bold = "**" if pipeline == "hybrid" else ""
                    lines.append("| {ds} | {proto} | {bold}{pipe}{bold} | {apcer} | {bpcer} | {acer} | {eer} | {auc} |".format(
                        ds=ds, proto=proto, pipe=pipeline, bold=bold,
                        apcer=fmt_pct(m.get("apcer_max")),
                        bpcer=fmt_pct(m.get("bpcer")),
                        acer=fmt_pct(m.get("acer")),
                        eer=fmt_pct(m.get("eer")),
                        auc=fmt_dec(m.get("auc")),
                    ))
                    break
    return "\n".join(lines) + "\n"


def main():
    results = load_results()
    out_pairs = {
        "table1_headline.md": build_table_1(results),
        "table2_celeba_per_type.md": build_table_2(results),
        "table5_ablation_tracks.md": build_table_5_ablation(results),
    }
    for name, content in out_pairs.items():
        path = FIG_DIR / name
        path.write_text(content)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
