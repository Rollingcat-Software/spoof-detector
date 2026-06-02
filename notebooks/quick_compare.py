"""Per-feature LIVE vs SPOOF separability — zero-extra-deps version.

Run against a folder of `amispoof-session-*.json` dumps:

    python notebooks/quick_compare.py docs/
    python notebooks/quick_compare.py notebooks/data/

Uses only pandas + numpy (no sklearn, no seaborn, no jupyter required).
Computes per-feature AUC (Mann-Whitney U) and d-prime, prints the ranked
table. Same loader/flattening logic as `separability.ipynb` so the
results match cell-by-cell once the notebook is runnable.

Class collapse:
    LIVE                -> LIVE
    REPLAY_* / SCREEN_* /
    PRINT / MASK /
    DEEPFAKE            -> SPOOF
    everything else     -> excluded
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def flatten_frame(analyzer_name: str, payload: dict | None) -> dict:
    out: dict = {}
    if not payload:
        return out
    if "score" in payload:
        out[f"{analyzer_name}.score"] = payload["score"]
    for k, v in (payload.get("details") or {}).items():
        if isinstance(v, (int, float, bool)):
            out[f"{analyzer_name}.{k}"] = float(v)
    return out


def collapse_label(label: str) -> str:
    label = (label or "UNLABELED").upper()
    if label == "LIVE":
        return "LIVE"
    if (
        label.startswith("REPLAY")
        or label.startswith("SCREEN")
        or label in ("PRINT", "MASK", "DEEPFAKE")
    ):
        return "SPOOF"
    return "UNLABELED"


# Post-V3-veto builds. Verdict-level results from older builds came from
# obsolete veto logic and must not be mixed in. Signal-level AUC is
# build-independent, but we still filter so the dataset is consistent.
CURRENT_BUILDS = ("threat-coop", "prod-cdn-restore")


def load_folder(folder: Path, build_filter: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(folder.glob("amispoof-session-*.json"))
    if not paths:
        sys.exit(f"No amispoof-session-*.json files in {folder}")
    rows: list[dict] = []
    skipped: dict[str, int] = {}
    for path in paths:
        sess = json.loads(path.read_text(encoding="utf-8"))
        env = sess.get("environment") or {}
        version = sess.get("amispoof_version") or "unknown"
        if build_filter and not any(b in version for b in build_filter):
            skipped[version] = skipped.get(version, 0) + 1
            continue
        raw_label = env.get("capture_label") or _infer_from_filename(path)
        binary = collapse_label(raw_label)
        subject = (env.get("notes") or "").strip() or "(unlabelled)"
        frame_log = sess.get("frame_log") or []
        if not frame_log:
            frame_log = [{
                "t_sec": (sess.get("verdict") or {}).get("session_duration_sec", 0),
                "frame_id": (sess.get("verdict") or {}).get("frames_analyzed", 0),
                "analyzer_scores": sess.get("latest_analyzer_scores") or {},
                "confidence": (sess.get("verdict") or {}).get("confidence", 0),
            }]
        for f in frame_log:
            flat: dict = {
                "session": path.stem,
                "class": binary,
                "class_fine": raw_label,
                "build": version,
                "subject": subject,
                "t_sec": f.get("t_sec"),
            }
            for analyzer_name, payload in (f.get("analyzer_scores") or {}).items():
                flat.update(flatten_frame(analyzer_name, payload))
            rows.append(flat)
    if build_filter:
        kept = len({r["session"] for r in rows})
        n_sk = sum(skipped.values())
        print(f"[build filter {build_filter}] kept {kept} session(s), "
              f"excluded {n_sk} on other builds: {skipped or '{}'}")
    return pd.DataFrame(rows)


def _infer_from_filename(path: Path) -> str:
    """Pre-Phase-F dumps have no environment.capture_label. Best-effort
    inference from filename slug (the new schema bakes the class into the
    file name). Returns UNLABELED if nothing matches."""
    name = path.stem.lower()
    if "-live-" in name:
        return "LIVE"
    for k in ("phone_close", "phone_far", "phone_tilted", "laptop", "print", "mask", "deepfake"):
        if k in name:
            return "REPLAY_PHONE_CLOSE" if k == "phone_close" else k.upper()
    return "UNLABELED"


def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC = (U / (n_pos * n_neg)) where U is the Mann-Whitney U statistic.
    Returns max(auc, 1-auc) so we report magnitude regardless of direction.
    NaN-aware: drops rows where score is NaN before ranking."""
    mask = ~np.isnan(scores)
    if mask.sum() < 30:
        return float("nan")
    s, y = scores[mask], labels[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").values
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    return max(auc, 1.0 - auc)


def d_prime(live: np.ndarray, spoof: np.ndarray) -> float:
    live = live[~np.isnan(live)]
    spoof = spoof[~np.isnan(spoof)]
    if len(live) < 5 or len(spoof) < 5:
        return float("nan")
    pooled_std = math.sqrt(0.5 * (np.var(live, ddof=0) + np.var(spoof, ddof=0)))
    if pooled_std < 1e-9:
        return float("nan")
    return abs(np.mean(live) - np.mean(spoof)) / pooled_std


def rank_features(frames: pd.DataFrame) -> pd.DataFrame:
    labelled = frames[frames["class"].isin(["LIVE", "SPOOF"])].copy()
    if labelled.empty:
        sys.exit("No LIVE/SPOOF rows. Capture both classes first.")
    y = (labelled["class"] == "SPOOF").astype(int).values
    n_live = (labelled["class"] == "LIVE").sum()
    n_spoof = (labelled["class"] == "SPOOF").sum()
    skip_cols = {"session", "class", "class_fine", "build", "subject", "t_sec"}
    rows = []
    for col in labelled.columns:
        if col in skip_cols:
            continue
        x = labelled[col].values.astype(float)
        live_x = labelled.loc[labelled["class"] == "LIVE", col].dropna().values
        spoof_x = labelled.loc[labelled["class"] == "SPOOF", col].dropna().values
        cov_live = len(live_x) / max(n_live, 1)
        cov_spoof = len(spoof_x) / max(n_spoof, 1)
        if cov_live < 0.3 or cov_spoof < 0.3:
            continue
        rows.append({
            "feature": col,
            "auc": auc_mann_whitney(x, y),
            "d_prime": d_prime(live_x, spoof_x),
            "mean_live": float(np.nanmean(live_x)),
            "mean_spoof": float(np.nanmean(spoof_x)),
            "std_live": float(np.nanstd(live_x)),
            "std_spoof": float(np.nanstd(spoof_x)),
            "n_live": int(len(live_x)),
            "n_spoof": int(len(spoof_x)),
        })
    return pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last").reset_index(drop=True)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="LIVE vs SPOOF per-feature separability (build-aware)")
    ap.add_argument("folder", nargs="?", default="notebooks/data")
    ap.add_argument("--build", action="append", default=None, metavar="SUBSTR",
                    help="keep only sessions whose amispoof_version contains SUBSTR "
                         "(repeatable). e.g. --build 2026-06-01")
    ap.add_argument("--current", action="store_true",
                    help=f"shortcut: only current post-V3 builds {CURRENT_BUILDS}")
    args = ap.parse_args()
    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"Folder does not exist: {folder.resolve()}")
    build_filter = list(args.build) if args.build else []
    if args.current:
        build_filter += list(CURRENT_BUILDS)
    frames = load_folder(folder, build_filter or None)
    print(f"Loaded {len(frames)} frames from {folder.resolve()}")
    print("Class breakdown:")
    print(frames["class"].value_counts().to_string())
    print("\nBuild distribution (sessions):")
    print(frames.groupby("build")["session"].nunique().to_string())
    print("\nSubject distribution (sessions) — needed for GroupKFold:")
    print(frames.groupby("subject")["session"].nunique().to_string())
    if not build_filter:
        print("\n** WARNING: no --build filter; results MIX verdict logic across builds. "
              "Use --current for trustworthy numbers. **")
    print()
    ranking = rank_features(frames)
    if ranking.empty:
        sys.exit("No features met the >=30% coverage floor in both classes.")
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(f"Per-feature ranking — top 30 by AUC ({len(ranking)} features total):")
    print(ranking.head(30).to_string(index=False))
    print()
    strong = ranking[ranking["auc"] >= 0.95]
    if not strong.empty:
        print(f"** {len(strong)} feature(s) with AUC >= 0.95 — single-threshold candidates: **")
        print(strong[["feature", "auc", "d_prime", "mean_live", "mean_spoof"]].to_string(index=False))


if __name__ == "__main__":
    main()
