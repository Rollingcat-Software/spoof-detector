"""Train a small CNN classifier on extracted face crops.

A 4-layer CNN (~270K params, MiniFASNet-size) trained from scratch on
CASIA-FASD train face crops, evaluated on test face crops + per-video
aggregation.

Architecture:
  conv(3→32, 3×3) → BN → ReLU → MaxPool 2×2
  conv(32→64, 3×3) → BN → ReLU → MaxPool 2×2
  conv(64→128, 3×3) → BN → ReLU → MaxPool 2×2
  conv(128→128, 3×3) → BN → ReLU → AdaptiveAvgPool 1×1
  flatten → Linear(128→2)

Total: ~270K params.

Usage:
    python -m tests.benchmark.train_cnn \\
        --train paper/figures/captures/casia_fasd_train_crops.npz \\
        --test  paper/figures/captures/casia_fasd_test_crops.npz \\
        --out   paper/figures/cnn_casia_fasd.json \\
        --epochs 30 --batch-size 64 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class CropDataset(Dataset):
    def __init__(self, crops, labels, augment: bool = False):
        # crops: (N, H, W, 3) uint8 BGR — convert to (N, 3, H, W) float32 [0, 1] RGB-channel order ImageNet style
        self.crops = crops
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.augment = augment
        # ImageNet-ish normalization (mean/std on BGR for simplicity)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, i):
        c = self.crops[i]  # (H, W, 3) BGR uint8
        if self.augment:
            # random horizontal flip
            if np.random.random() < 0.5:
                c = c[:, ::-1].copy()
            # random brightness/contrast
            if np.random.random() < 0.5:
                alpha = 1 + 0.2 * (np.random.random() - 0.5)
                beta = 20 * (np.random.random() - 0.5)
                c = np.clip(c.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        c_rgb = c[:, :, ::-1]  # BGR → RGB
        c_chw = np.ascontiguousarray(c_rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
        x = torch.from_numpy(c_chw)
        x = (x - self.mean) / self.std
        return x, self.labels[i]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_model(model, train_loader, val_loader, *, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Class weights — inverse frequency
    train_labels = []
    for _, y in train_loader.dataset:
        if not isinstance(y, int):
            y = int(y.item()) if hasattr(y, "item") else int(y)
        train_labels.append(y)
    n_class0 = sum(1 for y in train_labels if y == 0)
    n_class1 = sum(1 for y in train_labels if y == 1)
    pos_weight = n_class0 / max(1, n_class1)
    weights = torch.tensor([1.0, pos_weight], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_auc = 0.0
    best_state = None
    history = []

    for epoch in range(epochs):
        t0 = time.perf_counter()
        # train
        model.train()
        train_loss, train_n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0); train_n += x.size(0)
        train_loss /= train_n
        scheduler.step()

        # val
        model.eval()
        val_probs, val_labels, val_loss = [], [], 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                probs = F.softmax(logits, dim=1)[:, 1]
                val_loss += loss.item() * x.size(0); val_n += x.size(0)
                val_probs.append(probs.cpu().numpy())
                val_labels.append(y.cpu().numpy())
        val_loss /= val_n
        val_probs = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        # Quick AUC
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.5

        elapsed = time.perf_counter() - t0
        logger.info(f"epoch {epoch+1:2d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc, "elapsed_sec": elapsed})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_auc


def evaluate(model, test_loader, device, vids):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return all_probs, all_labels


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--arch", default="small",
                   choices=["small", "resnet18", "mobilenetv3"],
                   help="CNN architecture. small (242K params, CPU-friendly), "
                        "resnet18 (11M params, GPU recommended), "
                        "mobilenetv3 (2M params, fast on GPU).")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose, format="%(message)s")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== loading captures ===")
    train_data = np.load(args.train)
    test_data = np.load(args.test)
    crops_tr, labels_tr, vids_tr = train_data["crops"], train_data["labels"], train_data["vids"]
    crops_te, labels_te, vids_te = test_data["crops"], test_data["labels"], test_data["vids"]
    print(f"  train: N={len(crops_tr)}, {sum(labels_tr)} bonafide, crop shape={crops_tr.shape[1:]}")
    print(f"  test:  N={len(crops_te)}, {sum(labels_te)} bonafide")

    # Split train into train+val (90/10 stratified by label, but simpler: random)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(crops_tr))
    val_size = max(1, len(perm) // 10)
    val_idx = perm[:val_size]; tr_idx = perm[val_size:]

    ds_tr = CropDataset(crops_tr[tr_idx], labels_tr[tr_idx], augment=True)
    ds_val = CropDataset(crops_tr[val_idx], labels_tr[val_idx], augment=False)
    ds_te = CropDataset(crops_te, labels_te, augment=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, num_workers=args.num_workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"  device={device}  ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
    else:
        print(f"  device={device}  (no CUDA)")

    if args.arch == "small":
        model = SmallCNN().to(device)
    elif args.arch == "resnet18":
        # Larger model for GPU. Falls back fine on CPU but slower.
        from torchvision.models import resnet18
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        model = m.to(device)
    elif args.arch == "mobilenetv3":
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
        model = m.to(device)
    else:
        raise SystemExit(f"unknown --arch: {args.arch}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {args.arch}, {n_params:,} params")
    print()

    print(f"=== training ===")
    model, history, best_val_auc = train_model(model, dl_tr, dl_val, epochs=args.epochs, lr=args.lr, device=device)
    print(f"  best val AUC: {best_val_auc:.4f}")
    print()

    print(f"=== test evaluation ===")
    test_probs, test_labels = evaluate(model, dl_te, device, vids_te)

    # Per-frame metrics
    from src.metrics import classification_report
    types = ["unknown" if not bool(l) else None for l in test_labels]
    report_frame = classification_report(test_probs.tolist(), [bool(l) for l in test_labels], types)
    print(f"  PER-FRAME: ACER={report_frame['acer']*100:.2f}% EER={report_frame['eer']*100:.2f}% AUC={report_frame['auc']:.4f}")

    # Per-video aggregation (mean)
    by_vid = defaultdict(list)
    by_vid_y = {}
    for p_score, v, l in zip(test_probs, vids_te, test_labels):
        by_vid[str(v)].append(float(p_score))
        by_vid_y[str(v)] = bool(l)
    vid_scores = np.array([np.mean(ps) for ps in by_vid.values()])
    vid_labels = np.array([by_vid_y[v] for v in by_vid.keys()], dtype=bool)
    types_vid = ["unknown" if not l else None for l in vid_labels]
    report_video = classification_report(vid_scores.tolist(), [bool(l) for l in vid_labels], types_vid)
    # Max-accuracy threshold
    best_acc, best_th = 0.0, 0.0
    for th in np.linspace(0.001, 0.999, 999):
        pred = vid_scores >= th
        acc = (pred == vid_labels).sum() / len(vid_labels)
        if acc > best_acc:
            best_acc, best_th = acc, th
    pred = vid_scores >= best_th
    errs = (pred != vid_labels).sum()
    print(f"  PER-VIDEO: ACER={report_video['acer']*100:.2f}% EER={report_video['eer']*100:.2f}% AUC={report_video['auc']:.4f}")
    print(f"  PER-VIDEO max-accuracy: {best_acc*100:.2f}% @ threshold {best_th:.4f} ({errs}/{len(vid_labels)} errors)")

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "train_capture": str(args.train),
        "test_capture": str(args.test),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_auc": best_val_auc,
        "per_frame": report_frame,
        "per_video": report_video,
        "max_accuracy_video": best_acc,
        "max_accuracy_threshold": float(best_th),
        "max_accuracy_errors": int(errs),
        "history": history,
        "n_params": n_params,
        "device": str(device),
    }, indent=2, default=str))
    # Save weights
    weights_path = out.with_suffix(".pt")
    torch.save(model.state_dict(), weights_path)
    print(f"\nwrote {out}")
    print(f"wrote {weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
