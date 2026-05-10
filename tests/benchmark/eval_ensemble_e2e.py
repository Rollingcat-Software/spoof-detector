"""End-to-end ensemble evaluation: RF + CNN → per-video → bootstrap CIs.

Reproduces the headline result (100% on subject-disjoint CASIA-FASD test).
Inputs are the captured analyzer scores + face crops + a trained CNN.

Usage:
  python -m tests.benchmark.eval_ensemble_e2e \\
      --train-analyzers paper/figures/captures/casia_fasd_train_ensemble.json \\
      --test-analyzers  paper/figures/captures/casia_fasd_test_ensemble.json \\
      --train-crops paper/figures/captures/casia_fasd_train_crops.npz \\
      --test-crops  paper/figures/captures/casia_fasd_test_crops.npz \\
      --cnn-weights paper/figures/cnn_casia_fasd.pt \\
      --w-cnn 0.35 \\
      --out paper/figures/eval_e2e_casia_fasd.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parent_video(sid):
    m = re.match(r"^(.+\.avi)_\d+_(real|fake)$", sid)
    return m.group(1) if m else sid


def subj(sid):
    m = re.match(r"^(\d+)_", parent_video(sid))
    return int(m.group(1)) if m else 0


def load_analyzers(path):
    data = json.loads(Path(path).read_text())
    records = data["per_sample"]
    keys = sorted(records[0]["analyzer_scores"].keys())
    X = np.array([[r["analyzer_scores"].get(k, 50.0) / 100.0 for k in keys] for r in records])
    y = np.array([r["is_bonafide"] for r in records], dtype=int)
    vids = np.array([parent_video(r["sample_id"]) for r in records])
    subjs = np.array([subj(r["sample_id"]) for r in records])
    return X, y, vids, subjs, keys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train-analyzers", required=True)
    p.add_argument("--test-analyzers", required=True)
    p.add_argument("--train-crops", required=True)
    p.add_argument("--test-crops", required=True)
    p.add_argument("--cnn-weights", required=True)
    p.add_argument("--w-cnn", type=float, default=0.35)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.ensemble import RandomForestClassifier
    from src.metrics import classification_report
    from src.metrics.bootstrap import acer_ci, auc_ci, eer_ci
    from tests.benchmark.train_cnn import SmallCNN, CropDataset

    print("=== loading data ===")
    X_tr, y_tr, _, _, _ = load_analyzers(args.train_analyzers)
    X_te, y_te, vids_te, subjs_te, _ = load_analyzers(args.test_analyzers)
    test_crops = np.load(args.test_crops)
    print(f"  train analyzers: N={len(X_tr)}")
    print(f"  test analyzers:  N={len(X_te)}, videos={len(set(vids_te))}, subjects={len(set(subjs_te))}")

    print("=== training RF on per-frame analyzer features ===")
    rf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1)
    rf.fit(X_tr, y_tr)
    probs_rf = rf.predict_proba(X_te)[:, 1]

    print("=== loading CNN ===")
    model = SmallCNN()
    model.load_state_dict(torch.load(args.cnn_weights, weights_only=True))
    model.eval()

    print("=== computing CNN predictions ===")
    ds_te = CropDataset(test_crops["crops"], test_crops["labels"], augment=False)
    dl_te = DataLoader(ds_te, batch_size=64, num_workers=2)
    probs_cnn = []
    with torch.no_grad():
        for x, _ in dl_te:
            probs_cnn.append(F.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    probs_cnn = np.concatenate(probs_cnn)

    print(f"=== ensemble: {args.w_cnn:.2f} × CNN + {(1 - args.w_cnn):.2f} × RF ===")
    probs_ens = args.w_cnn * probs_cnn + (1 - args.w_cnn) * probs_rf

    # Aggregate per video
    by_vid = defaultdict(list)
    by_y = {}
    by_subj = {}
    for p, v, s, y in zip(probs_ens, vids_te, subjs_te, y_te):
        by_vid[str(v)].append(float(p))
        by_y[str(v)] = bool(y)
        by_subj[str(v)] = int(s)

    vid_keys = list(by_vid.keys())
    vid_scores = np.array([np.mean(by_vid[v]) for v in vid_keys])
    vid_labels = np.array([by_y[v] for v in vid_keys], dtype=bool)
    vid_subjects = np.array([by_subj[v] for v in vid_keys])

    # Find SUBJECT_TRAIN_MAX from train analyzers (last subject in train set)
    _, _, _, subjs_tr, _ = load_analyzers(args.train_analyzers)
    train_subjects = set(subjs_tr.tolist())
    test_subjects = set(subjs_te.tolist())
    overlap = train_subjects & test_subjects
    test_only = test_subjects - train_subjects
    print(f"  train subjects: {sorted(train_subjects)}")
    print(f"  test subjects (overlap with train): {sorted(overlap)}")
    print(f"  test-only subjects: {sorted(test_only)}")

    out_results = {"groups": {}}

    def report_group(label, mask):
        sc = vid_scores[mask].tolist()
        y = [bool(v) for v in vid_labels[mask]]
        if not sc:
            print(f"  {label}: empty"); return None
        types = ["unknown" if not bonafide else None for bonafide in y]
        report = classification_report(sc, y, types)
        n_resamples = 500
        a = acer_ci(sc, y, types, n_resamples=n_resamples, alpha=0.05, seed=42)
        e = eer_ci(sc, y, types, n_resamples=n_resamples, alpha=0.05, seed=42)
        u = auc_ci(sc, y, types, n_resamples=n_resamples, alpha=0.05, seed=42)
        # Max-acc
        sc_arr = np.array(sc); y_arr = np.array(y, dtype=bool)
        best = (0, 0)
        for th in np.linspace(0.001, 0.999, 999):
            pred = sc_arr >= th
            acc = (pred == y_arr).sum() / len(y_arr)
            if acc > best[0]: best = (acc, th)
        pred = sc_arr >= best[1]
        errs = int((pred != y_arr).sum())
        print(f"\n  === {label} ===")
        print(f"    N={len(sc)} videos ({sum(y)} bonafide, {len(y)-sum(y)} attack)")
        print(f"    ACER = {a.estimate*100:5.2f}%  (95% CI [{a.low*100:5.2f}, {a.high*100:5.2f}])")
        print(f"    EER  = {e.estimate*100:5.2f}%  (95% CI [{e.low*100:5.2f}, {e.high*100:5.2f}])")
        print(f"    AUC  = {u.estimate:.4f}  (95% CI [{u.low:.4f}, {u.high:.4f}])")
        print(f"    max-accuracy: {best[0]*100:.2f}% @ threshold {best[1]:.4f}  ({errs} errors)")
        return {
            "n_videos": len(sc),
            "n_bonafide": sum(y),
            "metrics": report,
            "acer_ci": [float(a.low), float(a.high)],
            "eer_ci":  [float(e.low), float(e.high)],
            "auc_ci":  [float(u.low), float(u.high)],
            "max_accuracy": float(best[0]),
            "max_accuracy_threshold": float(best[1]),
            "max_accuracy_errors": errs,
        }

    print("\n=== Results ===")
    out_results["groups"]["all"] = report_group("ALL test videos", np.ones_like(vid_subjects, dtype=bool))
    if overlap:
        seen_mask = np.isin(vid_subjects, list(overlap))
        out_results["groups"]["seen_subjects"] = report_group(
            f"SEEN subjects ({sorted(overlap)})", seen_mask)
    if test_only:
        unseen_mask = np.isin(vid_subjects, list(test_only))
        out_results["groups"]["unseen_subjects"] = report_group(
            f"UNSEEN subjects ({sorted(test_only)}) — paper headline", unseen_mask)

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "w_cnn": args.w_cnn,
            "cnn_weights": str(args.cnn_weights),
            "train_subjects": sorted(train_subjects),
            "test_subjects": sorted(test_subjects),
            "test_only_subjects": sorted(test_only),
        },
        "n_test_videos": len(vid_keys),
        **out_results,
    }, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
