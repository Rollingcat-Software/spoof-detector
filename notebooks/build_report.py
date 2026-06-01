"""Self-contained HTML separability report.

Run against a folder of `amispoof-session-*.json` dumps:

    python notebooks/build_report.py notebooks/data
    # → writes notebooks/separability_report.html · open in any browser

Uses ONLY pandas + numpy + matplotlib (no seaborn / sklearn / jupyter).
Charts are rendered to PNGs and base64-embedded in the HTML so the
output file is a single deliverable — no asset folder, no server.

Sections produced:
  1. Dataset summary table (class breakdown, per-session frame counts)
  2. Top-30 per-feature AUC table + d-prime
  3. Distribution histograms for the top 8 features (LIVE vs SPOOF)
  4. Texture_score % below threshold per session (the veto's input)
  5. Skin_score median per session (the V3 co-signal)
  6. Correlation matrix heatmap (top-20 features, matplotlib imshow)
  7. Verdict outcome table (was-supposed-to-be vs actual)
"""
from __future__ import annotations

import base64
import io
import json
import math
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib without GUI backend so this is server/headless safe.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# === Loader: identical contract to quick_compare.py ============================
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


def collapse_label(label: str | None) -> str:
    label = (label or "UNLABELED").upper()
    if label == "LIVE":
        return "LIVE"
    if label.startswith("REPLAY") or label in ("PRINT", "MASK", "DEEPFAKE"):
        return "SPOOF"
    return "UNLABELED"


CURRENT_BUILDS = ("threat-coop", "prod-cdn-restore", "skin-typing")


def load_folder(folder: Path, build_filter: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(folder.glob("amispoof-session-*.json"))
    if not paths:
        sys.exit(f"No amispoof-session-*.json in {folder.resolve()}")
    rows: list[dict] = []
    summary: list[dict] = []
    skipped = 0
    for path in paths:
        sess = json.loads(path.read_text(encoding="utf-8"))
        env = sess.get("environment") or {}
        verdict = sess.get("verdict") or {}
        version = sess.get("amispoof_version") or "unknown"
        if build_filter and not any(b in version for b in build_filter):
            skipped += 1
            continue
        raw_label = env.get("capture_label")
        binary = collapse_label(raw_label)
        frame_log = sess.get("frame_log") or []
        if not frame_log:
            frame_log = [{
                "t_sec": verdict.get("session_duration_sec", 0),
                "frame_id": verdict.get("frames_analyzed", 0),
                "analyzer_scores": sess.get("latest_analyzer_scores") or {},
                "confidence": verdict.get("confidence", 0),
            }]
        for f in frame_log:
            flat: dict = {
                "session": path.stem,
                "class": binary,
                "class_fine": raw_label or "UNLABELED",
                "build": version,
                "ambient": env.get("ambient_label"),
                "replay_device": env.get("replay_device"),
                "notes": env.get("notes"),
                "t_sec": f.get("t_sec"),
                "frame_id": f.get("frame_id"),
            }
            for analyzer_name, payload in (f.get("analyzer_scores") or {}).items():
                flat.update(flatten_frame(analyzer_name, payload))
            rows.append(flat)
        # Per-session row for the verdict-outcomes table.
        summary.append({
            "session": path.stem,
            "build": version,
            "label_truth": binary,
            "label_fine": raw_label or "UNLABELED",
            "ambient": env.get("ambient_label"),
            "notes": env.get("notes") or "",
            "verdict_engine": "LIVE" if verdict.get("is_live") else (
                "UNCERTAIN" if verdict.get("quality_uncertain") else "SPOOF"
            ),
            "confidence": verdict.get("confidence", 0),
            "frames": len(frame_log),
            "duration_sec": verdict.get("session_duration_sec", 0),
            "incidents": len(verdict.get("incidents") or []),
        })
    return pd.DataFrame(rows), pd.DataFrame(summary)


# === Per-feature AUC + d-prime ================================================
def auc_mann_whitney(scores: np.ndarray, labels: np.ndarray) -> float:
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
        return pd.DataFrame()
    y = (labelled["class"] == "SPOOF").astype(int).values
    n_live = (labelled["class"] == "LIVE").sum()
    n_spoof = (labelled["class"] == "SPOOF").sum()
    skip = {"session", "class", "class_fine", "build", "ambient", "replay_device", "notes", "t_sec", "frame_id"}
    rows = []
    for col in labelled.columns:
        if col in skip:
            continue
        x = labelled[col].values.astype(float)
        live_x = labelled.loc[labelled["class"] == "LIVE", col].dropna().values
        spoof_x = labelled.loc[labelled["class"] == "SPOOF", col].dropna().values
        if len(live_x) / max(n_live, 1) < 0.3 or len(spoof_x) / max(n_spoof, 1) < 0.3:
            continue
        rows.append({
            "feature": col,
            "auc": auc_mann_whitney(x, y),
            "d_prime": d_prime(live_x, spoof_x),
            "mean_live": float(np.nanmean(live_x)),
            "mean_spoof": float(np.nanmean(spoof_x)),
            "std_live": float(np.nanstd(live_x)),
            "std_spoof": float(np.nanstd(spoof_x)),
            "n_live": len(live_x),
            "n_spoof": len(spoof_x),
        })
    return pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last").reset_index(drop=True)


# === Plotting helpers (matplotlib only) =======================================
def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_top_distributions(frames: pd.DataFrame, ranking: pd.DataFrame, n: int = 8) -> str:
    labelled = frames[frames["class"].isin(["LIVE", "SPOOF"])]
    top = ranking.head(n)["feature"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, top):
        live_x = labelled.loc[labelled["class"] == "LIVE", col].dropna()
        spoof_x = labelled.loc[labelled["class"] == "SPOOF", col].dropna()
        ax.hist(live_x, bins=30, alpha=0.55, label=f"LIVE (n={len(live_x)})", color="#2ea043", density=True)
        ax.hist(spoof_x, bins=30, alpha=0.55, label=f"SPOOF (n={len(spoof_x)})", color="#f85149", density=True)
        auc = ranking.loc[ranking["feature"] == col, "auc"].iloc[0]
        ax.set_title(f"{col}\nAUC={auc:.3f}", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Top-8 features — LIVE vs SPOOF distributions", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_per_session_metric(frames: pd.DataFrame, col: str, threshold: float | None, title: str) -> str:
    sessions = (
        frames.dropna(subset=[col]).groupby(["session", "class"])[col]
        .agg(["mean", "std", lambda v: (v < (threshold or float("-inf"))).mean() * 100])
        .rename(columns={"<lambda_0>": "pct_below"})
        .reset_index()
    )
    sessions = sessions.sort_values("class")
    colors = {"LIVE": "#2ea043", "SPOOF": "#f85149", "UNLABELED": "#8b949e"}
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(sessions))))
    ax.barh(sessions["session"], sessions["mean"], color=[colors.get(c, "#888") for c in sessions["class"]])
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"threshold={threshold}")
        ax.legend()
    ax.set_xlabel(col)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_correlation(frames: pd.DataFrame, ranking: pd.DataFrame, top: int = 20) -> str:
    labelled = frames[frames["class"].isin(["LIVE", "SPOOF"])]
    cols = ranking.head(top)["feature"].tolist()
    if len(cols) < 2:
        return ""
    corr = labelled[cols].corr().values
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title(f"Pearson correlation — top {top} features by AUC")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_session_map(frames: pd.DataFrame, summary: pd.DataFrame) -> str:
    """The key chart: one point per session at (texture median, skin median),
    coloured by label, with false-positives / false-negatives ringed. Bands
    show the photo (skin<8) and video (skin>=30) skin modes + the texture
    collapse threshold (25)."""
    tcol, scol = "texture.texture_score", "screen_replay.skin_score"
    if tcol not in frames.columns or scol not in frames.columns:
        return ""
    med = frames.groupby("session")[[tcol, scol]].median().reset_index()
    m = med.merge(summary[["session", "label_truth", "verdict_engine", "notes"]], on="session", how="left")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axhspan(0, 8, color="#f0883e", alpha=0.08)
    ax.axhspan(30, max(70, m[scol].max() + 5), color="#a371f7", alpha=0.08)
    ax.axvline(25, color="#8b949e", ls=":", lw=1)
    ax.text(2, 4, "PHOTO band\n(skin<8)", fontsize=8, color="#f0883e")
    ax.text(2, 33, "VIDEO band (skin≥30)", fontsize=8, color="#a371f7")
    for cls, color in [("LIVE", "#2ea043"), ("SPOOF", "#f85149"), ("UNLABELED", "#8b949e")]:
        sub = m[m["label_truth"] == cls]
        if len(sub):
            ax.scatter(sub[tcol], sub[scol], c=color, s=90, alpha=0.85,
                       edgecolors="white", linewidths=0.6, label=f"{cls} (n={len(sub)})", zorder=3)
    fp = m[(m["label_truth"] == "LIVE") & (m["verdict_engine"] == "SPOOF")]
    fn = m[(m["label_truth"] == "SPOOF") & (m["verdict_engine"] == "LIVE")]
    ax.scatter(fp[tcol], fp[scol], s=320, facecolors="none", edgecolors="#f85149",
               linewidths=2.4, label=f"FALSE-POSITIVE (n={len(fp)})", zorder=4)
    ax.scatter(fn[tcol], fn[scol], s=320, facecolors="none", edgecolors="#58a6ff",
               linewidths=2.4, marker="s", label=f"FALSE-NEGATIVE (n={len(fn)})", zorder=4)
    ax.set_xlabel("texture_score median   (← flatter / screen ·········· sharper / real face →)")
    ax.set_ylabel("skin_score median   (← photo ····· live ····· video →)")
    ax.set_title("Per-session map: texture × skin — where every session lands, and what's misclassified")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_skin_modes(frames: pd.DataFrame) -> str:
    scol = "screen_replay.skin_score"
    if scol not in frames.columns:
        return ""
    lab = frames[frames["class"].isin(["LIVE", "SPOOF"])]
    live = lab.loc[lab["class"] == "LIVE", scol].dropna()
    spoof = lab.loc[lab["class"] == "SPOOF", scol].dropna()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(live, bins=60, alpha=0.6, color="#2ea043", density=True, label=f"LIVE (n={len(live):,})")
    ax.hist(spoof, bins=60, alpha=0.6, color="#f85149", density=True, label=f"SPOOF (n={len(spoof):,})")
    ax.axvspan(0, 8, color="#f0883e", alpha=0.12)
    ax.axvline(30, color="#a371f7", ls="--", lw=1)
    ax.set_xlabel("skin_score")
    ax.set_title("skin_score is tri-modal: photo (~0, orange) · live (13-34) · video (≥30, purple line)")
    ax.legend()
    fig.tight_layout()
    return fig_to_b64(fig)


# === HTML assembly ============================================================
HTML_CSS = """
:root { --bg:#0d1117; --panel:#161b22; --border:#2a3550; --text:#c9d1d9; --muted:#8b949e;
        --live:#2ea043; --spoof:#f85149; --uncertain:#d29922; --accent:#58a6ff; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
       margin:0; padding:24px; line-height:1.5; }
h1 { color:var(--accent); margin:0 0 4px; font-size:22px; }
h2 { color:var(--text); border-bottom:1px solid var(--border); padding-bottom:8px; margin-top:32px; font-size:18px; }
h3 { color:var(--muted); font-size:14px; font-weight:500; margin:24px 0 8px; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:18px; margin:12px 0; }
table { border-collapse:collapse; width:100%; font-size:12px; }
th, td { padding:6px 10px; text-align:left; border-bottom:1px solid var(--border); }
th { background:#21262d; color:var(--muted); font-weight:600; }
tr:hover { background:#21262d; }
img { max-width:100%; height:auto; border-radius:6px; background:white; padding:6px; }
.pill { display:inline-block; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
.pill.live { background:rgba(46,160,67,0.2); color:var(--live); }
.pill.spoof { background:rgba(248,81,73,0.2); color:var(--spoof); }
.pill.uncertain { background:rgba(210,153,34,0.2); color:var(--uncertain); }
.correct { color:var(--live); } .wrong { color:var(--spoof); font-weight:600; }
.auc-strong { color:var(--live); font-weight:600; }
.auc-mid { color:var(--uncertain); }
.auc-weak { color:var(--muted); }
.muted { color:var(--muted); font-size:11px; }
.footer { color:var(--muted); font-size:11px; margin-top:32px; text-align:center; }
"""


def auc_class(auc: float) -> str:
    if math.isnan(auc):
        return "auc-weak"
    if auc >= 0.9:
        return "auc-strong"
    if auc >= 0.7:
        return "auc-mid"
    return "auc-weak"


def df_to_html(df: pd.DataFrame, transforms: dict[str, callable] | None = None) -> str:
    transforms = transforms or {}
    rows = []
    for _, r in df.iterrows():
        cells = []
        for col in df.columns:
            v = r[col]
            if col in transforms:
                cells.append(f"<td>{transforms[col](v, r)}</td>")
            elif isinstance(v, float):
                cells.append(f"<td>{v:.3f}</td>")
            else:
                cells.append(f"<td>{escape(str(v))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    header = "".join(f"<th>{escape(c)}</th>" for c in df.columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_html(folder: Path, frames: pd.DataFrame, summary: pd.DataFrame, ranking: pd.DataFrame) -> str:
    parts: list[str] = []
    parts.append(f"<h1>spoof-detector · separability report</h1>")
    parts.append(f"<div class='muted'>Source: {escape(str(folder.resolve()))} · "
                 f"{len(summary)} sessions · {len(frames):,} frames</div>")

    # 1. Dataset summary
    parts.append("<h2>1. Dataset summary</h2><div class='panel'>")
    cls_counts = summary["label_truth"].value_counts().to_dict()
    parts.append("<p><b>Class breakdown:</b> "
                 + " · ".join([f"<span class='pill {k.lower()}'>{k}</span> {v}" for k, v in cls_counts.items()])
                 + "</p>")
    parts.append(df_to_html(summary, {
        "label_truth": lambda v, r: f"<span class='pill {v.lower()}'>{v}</span>",
        "verdict_engine": lambda v, r: f"<span class='pill {v.lower()}'>{v}</span>",
        "confidence": lambda v, r: f"{float(v) * 100:.0f}%",
        "duration_sec": lambda v, r: f"{float(v):.0f}s",
    }))
    parts.append("</div>")

    # 2. Verdict outcomes
    correct = ((summary["label_truth"] == "LIVE") & (summary["verdict_engine"] == "LIVE")) | (
        (summary["label_truth"] == "SPOOF") & (summary["verdict_engine"] == "SPOOF")
    )
    parts.append("<h2>2. Verdict outcomes</h2><div class='panel'>")
    parts.append(f"<p>Correctly classified: <b>{correct.sum()} / {len(summary)}</b> "
                 f"({correct.mean() * 100:.0f}%)</p>")
    out = summary[["session", "label_truth", "verdict_engine", "confidence", "incidents", "ambient"]].copy()
    out["match"] = ["✓" if c else "✗" for c in correct]
    parts.append(df_to_html(out, {
        "label_truth": lambda v, r: f"<span class='pill {v.lower()}'>{v}</span>",
        "verdict_engine": lambda v, r: f"<span class='pill {v.lower()}'>{v}</span>",
        "confidence": lambda v, r: f"{float(v) * 100:.0f}%",
        "match": lambda v, r: f"<span class='{'correct' if v == '✓' else 'wrong'}'>{v}</span>",
    }))
    parts.append("</div>")

    # 2b. The session map + skin modes (most useful eyeball view)
    fp_n = ((summary["label_truth"] == "LIVE") & (summary["verdict_engine"] == "SPOOF")).sum()
    fn_n = ((summary["label_truth"] == "SPOOF") & (summary["verdict_engine"] == "LIVE")).sum()
    parts.append("<h2>3. Session map — texture × skin (read this first)</h2><div class='panel'>")
    parts.append(f"<p class='muted'>One dot per session at its median texture &amp; skin. "
                 f"Red rings = real face called SPOOF (<b>{fp_n}</b> false-positives); "
                 f"blue squares = spoof called LIVE (<b>{fn_n}</b> false-negatives). "
                 f"A clean detector would have green (LIVE) and red (SPOOF) dots in separate regions — "
                 f"the overlap is the whole problem.</p>")
    smap = plot_session_map(frames, summary)
    if smap:
        parts.append(f"<img src='data:image/png;base64,{smap}'>")
    skin_modes = plot_skin_modes(frames)
    if skin_modes:
        parts.append("<h3>skin_score distribution (frame-level) — the tri-modal signal</h3>")
        parts.append(f"<img src='data:image/png;base64,{skin_modes}'>")
    parts.append("</div>")

    # 4. Per-feature AUC table
    if not ranking.empty:
        parts.append("<h2>3. Per-feature separability ranking (top 30)</h2><div class='panel'>")
        parts.append("<p class='muted'>Green ≥ 0.90 · Yellow 0.70–0.90 · Grey < 0.70. "
                     "d-prime ≥ 1.5 means the LIVE/SPOOF distributions are clearly separated.</p>")
        top = ranking.head(30).copy()
        parts.append(df_to_html(top, {
            "auc": lambda v, r: f"<span class='{auc_class(v)}'>{v:.3f}</span>" if not math.isnan(v) else "—",
            "d_prime": lambda v, r: f"{v:.2f}" if not math.isnan(v) else "—",
        }))
        parts.append("</div>")

        # 4. Distribution histograms
        parts.append("<h2>4. Top-8 feature distributions (LIVE vs SPOOF)</h2><div class='panel'>")
        parts.append(f"<img src='data:image/png;base64,{plot_top_distributions(frames, ranking)}'>")
        parts.append("</div>")

        # 5. Texture veto input
        if "texture.texture_score" in frames.columns:
            parts.append("<h2>5. Texture veto input — texture_score per session</h2><div class='panel'>")
            parts.append("<p class='muted'>The veto fires when ≥30% of recent frames are below threshold (25). "
                         "Bar shows session mean.</p>")
            parts.append(f"<img src='data:image/png;base64,{plot_per_session_metric(frames, 'texture.texture_score', 25, 'texture_score (Laplacian variance) per session — lower means flatter pixels')}'>")
            parts.append("</div>")

        # 6. Skin co-signal
        if "screen_replay.skin_score" in frames.columns:
            parts.append("<h2>6. V3 co-signal — skin_score per session</h2><div class='panel'>")
            parts.append("<p class='muted'>V3 requires median skin_score ≥ 30 in the window before firing texture veto.</p>")
            parts.append(f"<img src='data:image/png;base64,{plot_per_session_metric(frames, 'screen_replay.skin_score', 30, 'screen_replay.skin_score per session — higher means more screen-like skin rendering')}'>")
            parts.append("</div>")

        # 7. Correlation matrix
        parts.append("<h2>7. Top-20 feature correlation matrix</h2><div class='panel'>")
        parts.append(f"<img src='data:image/png;base64,{plot_correlation(frames, ranking)}'>")
        parts.append("</div>")

    parts.append(f"<div class='footer'>Generated by notebooks/build_report.py · "
                 f"matplotlib + pandas + numpy · zero extra deps · spoof-detector / FIVUCSAS</div>")
    return f"<!doctype html><html><head><meta charset='utf-8'>" \
           f"<title>spoof-detector — separability report</title>" \
           f"<style>{HTML_CSS}</style></head><body>" \
           + "".join(parts) + "</body></html>"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="amispoof separability + visual report (build-aware)")
    ap.add_argument("folder", nargs="?", default="notebooks/data")
    ap.add_argument("--build", action="append", default=None, metavar="SUBSTR",
                    help="keep only sessions whose amispoof_version contains SUBSTR (repeatable)")
    ap.add_argument("--current", action="store_true",
                    help=f"shortcut: only current builds {CURRENT_BUILDS}")
    args = ap.parse_args()
    folder = Path(args.folder)
    if not folder.exists():
        sys.exit(f"Folder does not exist: {folder.resolve()}")
    bf = list(args.build) if args.build else []
    if args.current:
        bf += list(CURRENT_BUILDS)
    print(f"Loading sessions from {folder.resolve()}..." + (f" [build filter: {bf}]" if bf else ""))
    frames, summary = load_folder(folder, bf or None)
    print(f"  loaded {len(frames)} frames across {len(summary)} sessions")
    print("  builds:", summary["build"].value_counts().to_dict())
    print("Computing per-feature AUC...")
    ranking = rank_features(frames)
    print(f"  ranked {len(ranking)} features")
    print("Rendering HTML...")
    html = build_html(folder, frames, summary, ranking)
    suffix = "_current" if bf else ""
    out_path = Path("notebooks") / f"separability_report{suffix}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\nReport written: {out_path.resolve()}")
    print(f"Size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"Open in browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
