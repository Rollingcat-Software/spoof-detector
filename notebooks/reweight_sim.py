#!/usr/bin/env python3
"""Faithful offline simulator for the fuser's REAL probability (→ confidence)
under different analyzer weights, so weight changes are MEASURED, not guessed.

Why this is faithful: every row of SPOOF_SIGNAL_MAP sums to ~1.0, so the fuser's
REAL probability reduces exactly to the weight-normalized mean of analyzer scores
  REAL_prob = Σ w_a·(score_a/100) / Σ w_a
(over present, weight>0 analyzers, after the same Nyquist gate). We replay the
recorded per-frame analyzer_scores from notebooks/data/*.json through this and
report how well the per-session REAL_prob separates LIVE from SPOOF.

The point: if even the best weight set barely separates the classes, the fuser
confidence cannot be fixed by reweighting — the signal isn't in the bank, which
is the argument for the foundation-model head (tools/train_fas_adapter.py).

Run:  python notebooks/reweight_sim.py
"""
from __future__ import annotations
import glob
import json
import math

CURRENT_BUILDS = ("threat-coop", "prod-cdn-restore", "motion-typing")
NYQUIST = {"screen_flicker": 18, "micro_tremor": 20, "rppg": 10}

# Effective amispoof runtime: app.js disables these via enable*:false, so they
# never enter the fuser regardless of weight. We drop them here for fidelity.
DISABLED = {"temporal", "screen_flicker", "micro_tremor",
            "expression_dynamics", "background_grid"}

DEFAULT = {
    "minifasnet": 3.0, "planarity": 2.0, "screen_flicker": 3.0,
    "landmark_variance": 2.0, "background_grid": 1.5, "device_boundary": 0.5,
    "micro_tremor": 0.5, "rppg": 0.5, "blink": 0.5, "screen_replay": 0.5,
    "ar_filter": 0.3, "temporal": 0.3, "texture": 1.5, "moire": 0.0,
}

WEIGHT_SETS = {
    "current (default)": DEFAULT,
    "drop anti-corr screen_replay": {**DEFAULT, "screen_replay": 0.0},
    "reliability-reweighted": {
        # up-weight the few weak-but-real signals (blink d'0.44, dev_boundary
        # d'0.40), down-weight the noise (planarity, landmark_var, texture),
        # zero the anti-correlated (screen_replay). minifasnet kept as anchor.
        "minifasnet": 3.0, "blink": 2.0, "device_boundary": 1.5,
        "planarity": 0.5, "landmark_variance": 0.5, "texture": 0.5,
        "screen_replay": 0.0, "rppg": 0.5, "ar_filter": 0.3, "moire": 0.0,
    },
    "minifasnet only": {"minifasnet": 1.0},
    "blink + device_boundary only": {"blink": 1.0, "device_boundary": 1.0},
    # COMPLETE dict (every analyzer explicit → no ?? 0.5 fallback), graded by
    # measured top-line d': reliable motion+device up, noise heavies down,
    # anti-correlated zeroed. This is what ships to the amispoof config.
    "reliability v2 (shipped)": {
        "minifasnet": 2.0, "gaze": 2.0, "blink_symmetry": 2.0, "blink": 2.0,
        "behavioral_pattern": 1.5, "device_boundary": 1.5,
        "planarity": 0.5, "texture": 0.5, "landmark_variance": 0.5,
        "eyebrow_motion": 0.3,
        "rppg": 0.0, "screen_replay": 0.0, "pose_3d_consistency": 0.0,
        "moire": 0.0, "background_motion": 0.0, "ar_filter": 0.3,
        "screen_flicker": 0.0, "micro_tremor": 0.0, "temporal": 0.0,
        "background_grid": 0.0, "expression_dynamics": 0.0,
        "hand_tracking": 0.0, "voice_activity": 0.0, "audio_mouth_sync": 0.0,
    },
}


def collapse(lbl):
    lbl = (lbl or "UNLABELED").upper()
    if lbl == "LIVE":
        return "LIVE"
    if lbl.startswith(("REPLAY", "SCREEN", "PRINT", "MASK", "DEEPFAKE")):
        return "SPOOF"
    return None


def session_real_prob(frame_log, weights):
    """Mean per-frame weight-normalized score (the fuser's REAL probability)."""
    vals = []
    for fr in frame_log:
        asc = fr.get("analyzer_scores") or {}
        fps = fr.get("fps")
        num = den = 0.0
        for name, payload in asc.items():
            if name in DISABLED:
                continue
            # Faithful to MultiClassFuser: unlisted analyzers fall back to 0.5
            # (this is the footgun — ~10 uncalibrated analyzers get 0.5 each).
            w = weights.get(name)
            if w is None:
                w = 0.5
            if w <= 0:
                continue
            if name in NYQUIST and isinstance(fps, (int, float)) and 0 < fps < NYQUIST[name]:
                continue
            score = payload.get("score") if isinstance(payload, dict) else payload
            if not isinstance(score, (int, float)):
                continue
            num += w * (score / 100.0)
            den += w
        if den > 0:
            vals.append(num / den)
    return sum(vals) / len(vals) if vals else None


def auc(live, spoof):
    """P(random LIVE REAL_prob > random SPOOF REAL_prob) — 1.0 = perfect."""
    if not live or not spoof:
        return float("nan")
    wins = ties = 0
    for a in live:
        for b in spoof:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(live) * len(spoof))


def main():
    sessions = []
    for f in glob.glob("notebooks/data/*.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        ver = d.get("amispoof_version") or ""
        if not any(b in ver for b in CURRENT_BUILDS):
            continue
        cls = collapse((d.get("environment") or {}).get("capture_label"))
        if cls is None:
            continue
        fl = d.get("frame_log") or []
        if fl:
            sessions.append((cls, fl))
    n_live = sum(1 for c, _ in sessions if c == "LIVE")
    print(f"current-build sessions with frame_log: {len(sessions)} "
          f"(LIVE {n_live}, SPOOF {len(sessions) - n_live})\n")

    print(f"{'weight set':32} {'AUC':>6} {'live_m':>7} {'spoof_m':>8} {'gap':>6} {'best-thr acc':>12}")
    for name, w in WEIGHT_SETS.items():
        live, spoof = [], []
        for cls, fl in sessions:
            rp = session_real_prob(fl, w)
            if rp is None:
                continue
            (live if cls == "LIVE" else spoof).append(rp)
        a = auc(live, spoof)
        lm = sum(live) / len(live) if live else float("nan")
        sm = sum(spoof) / len(spoof) if spoof else float("nan")
        # best single-threshold accuracy
        allv = sorted(set(live + spoof))
        best = 0.0
        for t in allv:
            ok = sum(1 for x in live if x >= t) + sum(1 for x in spoof if x < t)
            best = max(best, ok / max(1, len(live) + len(spoof)))
        print(f"{name:32} {a:6.3f} {lm:7.3f} {sm:8.3f} {lm - sm:6.3f} {best:11.0%}")

    print("\nReading: AUC ~0.5 = no separation; gap = how far LIVE sits above "
          "SPOOF in REAL_prob. Single-subject data -> PROVISIONAL; confirm on "
          "multi-subject crops before trusting the absolute numbers.")


if __name__ == "__main__":
    main()
