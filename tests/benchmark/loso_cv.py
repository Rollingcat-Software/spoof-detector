"""Leave-One-Subject-Out cross-validation on CASIA-FASD.

For each subject in the training set, hold out all that subject's frames,
train the full pipeline (RF on analyzers + CNN on crops) on the remaining
subjects, evaluate on the held-out subject. Aggregate ACER/AUC across
all 20 folds.

This is the rigorous robustness check — eliminates the "lucky split"
worry from a single train/test partition.

Usage:
    python -m tests.benchmark.loso_cv \\
        --train-analyzers paper/figures/captures/casia_fasd_train_ensemble.json \\
        --test-analyzers  paper/figures/captures/casia_fasd_test_ensemble.json \\
        --train-crops     paper/figures/captures/casia_fasd_train_crops.npz \\
        --test-crops      paper/figures/captures/casia_fasd_test_crops.npz \\
        --out paper/figures/loso_cv.json \\
        --cnn-epochs 15

Run time: ~30 sec per fold × 20 folds = ~10 min total on CPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def parent_video(sid):
    m = re.match(r"^(.+\.avi)_\d+_(real|fake)$", sid)
    return m.group(1) if m else sid


def subj(sid):
    m = re.match(r"^(\d+)_", parent_video(sid))
    return int(m.group(1)) if m else 0


def load_analyzers(path):
    """Load capture JSON, return per-frame X, y, vids, subjs, sids, keys."""
    data = json.loads(Path(path).read_text())
    records = data["per_sample"]
    keys = sorted(records[0]["analyzer_scores"].keys())
    X = np.array([[r["analyzer_scores"].get(k, 50.0) / 100.0 for k in keys] for r in records])
    y = np.array([r["is_bonafide"] for r in records], dtype=int)
    vids = np.array([parent_video(r["sample_id"]) for r in records])
    subjs = np.array([subj(r["sample_id"]) for r in records])
    sids = np.array([r["sample_id"] for r in records])
    return X, y, vids, subjs, sids, keys


def evaluate_fold(
    X_an_tr, y_an_tr, X_an_te, y_an_te, vids_te,
    crops_tr, labels_tr, crops_te, labels_te, vids_te_crops,
    *, cnn_epochs: int, cnn_lr: float, cnn_seed: int,
):
    """Train RF + CNN on (tr), predict on (te), aggregate per video."""
    from sklearn.ensemble import RandomForestClassifier
    from tests.benchmark.train_cnn import SmallCNN, CropDataset

    # 1. RF on analyzer features
    rf = RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_an_tr, y_an_tr)
    probs_rf = rf.predict_proba(X_an_te)[:, 1]

    # 2. CNN on face crops
    torch.manual_seed(cnn_seed)
    np.random.seed(cnn_seed)

    # Random 90/10 train/val split (within the training fold's data)
    n_tr = len(crops_tr)
    perm = np.random.RandomState(cnn_seed).permutation(n_tr)
    val_size = max(1, n_tr // 10)
    val_idx = perm[:val_size]; tr_idx = perm[val_size:]

    ds_tr = CropDataset(crops_tr[tr_idx], labels_tr[tr_idx], augment=True)
    ds_val = CropDataset(crops_tr[val_idx], labels_tr[val_idx], augment=False)
    ds_te = CropDataset(crops_te, labels_te, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, num_workers=2)
    dl_te = DataLoader(ds_te, batch_size=64, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cnn_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cnn_epochs)

    n0 = (labels_tr[tr_idx] == 0).sum()
    n1 = (labels_tr[tr_idx] == 1).sum()
    pos_weight = max(1, n0) / max(1, n1)
    weights = torch.tensor([1.0, pos_weight], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(cnn_epochs):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # CNN test predictions
    model.eval()
    probs_cnn = []
    with torch.no_grad():
        for x, _ in dl_te:
            x = x.to(device)
            probs_cnn.append(F.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    probs_cnn = np.concatenate(probs_cnn)

    # 3. Ensemble
    probs_ens = 0.35 * probs_cnn + 0.65 * probs_rf

    # 4. Per-video aggregation
    by_vid = defaultdict(list); by_y = {}
    for p, v, label in zip(probs_ens, vids_te, y_an_te):
        by_vid[str(v)].append(float(p)); by_y[str(v)] = bool(label)
    vid_scores = np.array([np.mean(by_vid[v]) for v in by_vid])
    vid_labels = np.array([by_y[v] for v in by_vid], dtype=bool)

    # 5. Metrics: ACER + AUC + max-accuracy
    from src.metrics import classification_report
    types = ["unknown" if not bool(v) else None for v in vid_labels]
    report = classification_report(vid_scores.tolist(), [bool(v) for v in vid_labels], types)
    # max-acc threshold
    best = (0, 0)
    for th in np.linspace(0.001, 0.999, 999):
        pred = vid_scores >= th
        a = (pred == vid_labels).sum() / len(vid_labels)
        if a > best[0]: best = (a, th)
    pred = vid_scores >= best[1]
    return {
        "acer": float(report["acer"]),
        "eer": float(report["eer"]),
        "auc": float(report["auc"]),
        "max_acc": float(best[0]),
        "max_acc_threshold": float(best[1]),
        "errors": int((pred != vid_labels).sum()),
        "n_videos": int(len(vid_labels)),
        "n_bonafide": int(vid_labels.sum()),
        "n_attack": int((~vid_labels).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train-analyzers", required=True)
    p.add_argument("--test-analyzers", required=True)
    p.add_argument("--train-crops", required=True)
    p.add_argument("--test-crops", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cnn-epochs", type=int, default=15)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--cnn-seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=None,
                   help="Optional cap on number of folds (default = run on all train subjects)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose, format="%(message)s")

    # Load all data
    Xa_tr, ya_tr, vids_an_tr, subjs_an_tr, sids_an_tr, keys = load_analyzers(args.train_analyzers)
    Xa_te, ya_te, vids_an_te, subjs_an_te, sids_an_te, _ = load_analyzers(args.test_analyzers)
    crops_tr_data = np.load(args.train_crops)
    crops_te_data = np.load(args.test_crops)
    crops_tr = crops_tr_data["crops"]; labels_tr = crops_tr_data["labels"]
    crops_te = crops_te_data["crops"]; labels_te = crops_te_data["labels"]
    sids_crops_tr = crops_tr_data["sample_ids"]
    sids_crops_te = crops_te_data["sample_ids"]
    vids_crops_tr = np.array([parent_video(s) for s in sids_crops_tr])
    vids_crops_te = np.array([parent_video(s) for s in sids_crops_te])
    subjs_crops_tr = np.array([subj(s) for s in sids_crops_tr])

    # Build subject index for analyzers train
    train_subjects = sorted(set(subjs_an_tr.tolist()))
    print(f"train subjects: {train_subjects}")
    if args.folds:
        train_subjects = train_subjects[: args.folds]
        print(f"capped to first {args.folds} subjects")

    # Run LOSO
    results = []
    t0 = time.perf_counter()
    for fold_idx, held_out in enumerate(train_subjects):
        # Train on train subjects EXCEPT held_out + ALL test data (to keep ensemble realistic)
        # Actually for proper LOSO, only train on subjects-other-than-held-out
        an_tr_mask = subjs_an_tr != held_out
        crops_tr_mask = subjs_crops_tr != held_out

        X_tr_fold = Xa_tr[an_tr_mask]
        y_tr_fold = ya_tr[an_tr_mask]
        crops_tr_fold = crops_tr[crops_tr_mask]
        labels_tr_fold = labels_tr[crops_tr_mask]

        # Test on the held-out subject's frames in the train file (same person, all frames)
        an_te_mask = subjs_an_tr == held_out
        crops_held_mask = subjs_crops_tr == held_out
        # n.b. assumes order alignment between analyzer JSON and crops .npz — both come from same iter

        X_te_fold = Xa_tr[an_te_mask]
        y_te_fold = ya_tr[an_te_mask]
        vids_te_fold = vids_an_tr[an_te_mask]
        crops_te_fold = crops_tr[crops_held_mask]
        labels_te_fold = labels_tr[crops_held_mask]
        vids_crops_te_fold = vids_crops_tr[crops_held_mask]

        if len(X_te_fold) == 0:
            print(f"fold subj={held_out}: no held-out data, skip")
            continue

        n_tr = len(X_tr_fold)
        n_te = len(X_te_fold)
        bf_te = int(y_te_fold.sum())
        result = evaluate_fold(
            X_tr_fold, y_tr_fold, X_te_fold, y_te_fold, vids_te_fold,
            crops_tr_fold, labels_tr_fold, crops_te_fold, labels_te_fold, vids_crops_te_fold,
            cnn_epochs=args.cnn_epochs, cnn_lr=args.cnn_lr, cnn_seed=args.cnn_seed,
        )
        result["held_out_subject"] = int(held_out)
        result["n_train_frames"] = int(n_tr)
        result["n_test_frames"] = int(n_te)
        results.append(result)

        elapsed = time.perf_counter() - t0
        avg_per_fold = elapsed / (fold_idx + 1)
        print(f"fold {fold_idx+1}/{len(train_subjects)} subj={held_out:2d}  "
              f"train={n_tr:4d}f test={n_te:3d}f bf={bf_te:2d}  "
              f"ACER={result['acer']*100:5.2f}%  AUC={result['auc']:.4f}  "
              f"acc={result['max_acc']*100:5.2f}%  errs={result['errors']}/{result['n_videos']}  "
              f"elapsed={elapsed:.1f}s avg={avg_per_fold:.1f}s/fold")

    # Aggregate
    print(f"\n=== LOSO CV Summary (N={len(results)} folds) ===")
    if results:
        acers = [r["acer"] for r in results]
        eers = [r["eer"] for r in results]
        aucs = [r["auc"] for r in results]
        accs = [r["max_acc"] for r in results]
        errs = [r["errors"] for r in results]
        total_errs = sum(errs)
        total_videos = sum(r["n_videos"] for r in results)
        print(f"  ACER     mean ± std: {np.mean(acers)*100:.2f}% ± {np.std(acers)*100:.2f}%")
        print(f"  EER      mean ± std: {np.mean(eers)*100:.2f}% ± {np.std(eers)*100:.2f}%")
        print(f"  AUC      mean ± std: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Max-acc  mean ± std: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
        print(f"  Total errors: {total_errs}/{total_videos} = {total_errs/total_videos*100:.2f}%")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {"cnn_epochs": args.cnn_epochs, "cnn_lr": args.cnn_lr, "cnn_seed": args.cnn_seed},
        "n_folds": len(results),
        "summary": {
            "acer_mean": float(np.mean([r["acer"] for r in results])) if results else 0,
            "acer_std":  float(np.std([r["acer"] for r in results])) if results else 0,
            "auc_mean":  float(np.mean([r["auc"] for r in results])) if results else 0,
            "auc_std":   float(np.std([r["auc"] for r in results])) if results else 0,
            "total_errors": int(sum(r["errors"] for r in results)) if results else 0,
            "total_videos": int(sum(r["n_videos"] for r in results)) if results else 0,
        },
        "folds": results,
    }, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
