#!/usr/bin/env python3
"""Train a domain-generalizable FAS head on a FROZEN vision-foundation backbone
and export it to ONNX for the browser WebGPU execution provider.

Why this exists
---------------
The hand-tuned analyzer bank does not transfer across subjects/cameras — the
paper's own §8.2 shows MiniFASNet *alone* beats the calibrated hybrid zero-shot,
and two weighted analyzers actively harm out-of-distribution accuracy. The
2025-26 face-anti-spoofing literature's answer to domain generalization is
**vision-foundation-model features (DINOv2 / CLIP) + a small trainable head**
(S-Adapter, C-Adapter, DiVT). This script produces exactly that: a successor to
`minifasnet_v2.onnx` whose I/O contract matches `FoundationModelAnalyzer.ts`:

    face crop  ->  [1, 3, 224, 224] float32, RGB, ImageNet-normalized
               ->  backbone (frozen)  ->  head (trained)
               ->  logits [1, 2]   (index 0 = SPOOF, index 1 = REAL)

Data note
---------
The 86 amispoof captures in `notebooks/data/` are analyzer-score *telemetry*
(per-frame scores + bbox), NOT pixels — they cannot train a vision model. Use a
public FAS dataset (CASIA-FASD / CelebA-Spoof via HuggingFace, which the paper
already cites) or face crops captured locally going forward (see the README).

GTX 1650 (4 GB) friendly
------------------------
The backbone is FROZEN — only the head trains — so VRAM use is tiny and even a
CPU works. `--smoke` validates the entire train -> export -> ORT-load pipeline on
synthetic tensors with NO network downloads, so you can confirm the plumbing
before committing to the ~85 MB DINOv2 download + dataset pull.

Usage
-----
    # plumbing check, no downloads, ~5 s:
    python tools/train_fas_adapter.py --smoke --out models/fas_head_smoke.onnx

    # real run (downloads DINOv2 ~85 MB + the dataset):
    python tools/train_fas_adapter.py \
        --backbone dinov2_vits14 \
        --dataset hf:nguyenkhoa/CASIA-FASD \
        --epochs 10 --batch-size 32 \
        --out models/fas_head_dinov2.onnx --fp16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch is required
    print("ERROR: PyTorch is required. `pip install torch` (CUDA build for GPU).",
          file=sys.stderr)
    raise

# ImageNet statistics — the browser analyzer normalizes with these exact values.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224  # 14 * 16 — valid for DINOv2 /14 patch size.
NUM_CLASSES = 2   # [spoof, real]


# --------------------------------------------------------------------------- #
# Backbone + head
# --------------------------------------------------------------------------- #
def build_backbone(name: str, smoke: bool) -> tuple[nn.Module, int]:
    """Return (frozen backbone, feature_dim). Backbone outputs a [B, D] vector."""
    if smoke:
        # No-download stand-in with the same image-in / vector-out contract.
        feat_dim = 64
        backbone = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,3,224,224] -> [B,3,1,1]
            nn.Flatten(),             # -> [B,3]
            nn.Linear(3, feat_dim),   # -> [B,64]
        )
        return backbone, feat_dim

    if name.startswith("dinov2"):
        # facebookresearch/dinov2: forward(x) returns the CLS embedding [B, D].
        # vits14 -> 384, vitb14 -> 768, vitl14 -> 1024.
        backbone = torch.hub.load("facebookresearch/dinov2", name)
        feat_dim = {"dinov2_vits14": 384, "dinov2_vitb14": 768,
                    "dinov2_vitl14": 1024}.get(name, 384)
        return backbone, feat_dim

    if name.startswith("clip"):
        # Optional CLIP image-encoder path via open_clip; left as an explicit
        # error so the dependency is opt-in rather than a silent import.
        try:
            import open_clip  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "CLIP backbone needs `pip install open_clip_torch`."
            ) from exc
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
        backbone = model.visual
        feat_dim = backbone.output_dim
        return backbone, feat_dim

    raise SystemExit(f"Unknown backbone '{name}'. Use dinov2_vits14 | clip | (smoke).")


class FasHead(nn.Module):
    """Small trainable classifier head on top of frozen backbone features."""

    def __init__(self, feat_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class FasModel(nn.Module):
    """backbone (frozen) + head, exported as one graph so the browser runs
    a single ONNX session from a normalized face crop to [spoof, real] logits."""

    def __init__(self, backbone: nn.Module, head: FasHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Backbone is frozen; no grad through it even during training.
        with torch.no_grad():
            feat = self.backbone(pixel_values)
        return self.head(feat)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def synthetic_loader(n: int, batch: int, device: torch.device):
    """Random already-normalized crops + a learnable signal (brighter -> real)
    so the head's loss actually moves in --smoke mode."""
    g = torch.Generator().manual_seed(42)
    x = torch.randn(n, 3, INPUT_SIZE, INPUT_SIZE, generator=g)
    # Inject a trivial separable signal: class 1 (real) has +0.5 mean shift.
    y = (torch.rand(n, generator=g) > 0.5).long()
    x = x + (y.view(-1, 1, 1, 1).float() * 0.5)
    for i in range(0, n, batch):
        yield x[i:i + batch].to(device), y[i:i + batch].to(device)


def _pil_to_norm_tensor(img, mean, std):
    """PIL.Image -> normalized CHW float tensor. PIL + numpy only (no
    torchvision), so the folder loader needs just `pip install pillow`."""
    img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW


def folder_loader(root: str, batch: int, device: torch.device):
    """Stream a local image folder laid out as <root>/{real,spoof}/**/*.jpg —
    exactly what amispoof's '📸 Save crops' writes (real=1, spoof=0). An
    `unlabeled/` dir, if present, is ignored."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Folder datasets need `pip install pillow`.") from exc
    from pathlib import Path as _P

    base = _P(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items: list[tuple[Path, int]] = []
    for label, sub in ((1, "real"), (0, "spoof")):
        d = base / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in exts:
                items.append((p, label))
    if not items:
        raise SystemExit(f"No images under {base}/real or {base}/spoof.")
    n_real = sum(1 for _, l in items if l == 1)
    print(f"  folder dataset: {n_real} real / {len(items) - n_real} spoof "
          f"({len(items)} crops)")

    buf_x, buf_y = [], []
    for p, label in items:
        with Image.open(p) as img:
            buf_x.append(_pil_to_norm_tensor(img, IMAGENET_MEAN, IMAGENET_STD))
        buf_y.append(label)
        if len(buf_x) == batch:
            yield torch.stack(buf_x).to(device), torch.tensor(buf_y, device=device)
            buf_x, buf_y = [], []
    if buf_x:
        yield torch.stack(buf_x).to(device), torch.tensor(buf_y, device=device)


def hf_loader(hf_id, split, batch, device, max_samples=3000):
    """STREAM a HuggingFace FAS dataset (image col + binary label col), capped at
    max_samples so we don't pull the whole multi-GB set. Prefers a face-crop
    column. PIL-normalized (no torchvision). NOTE: the raw label here is the
    dataset's own convention (e.g. nguyenkhoa/antispoofing-3 = 0:live/1:spoof);
    use --flip-labels to map to this head's [spoof=0, real=1]."""
    try:
        from datasets import load_dataset
        from PIL import Image  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("HF datasets need `pip install datasets pillow`.") from exc

    IMAGE_KEY_CANDIDATES = ("cropped_image", "image", "img", "jpg")  # prefer the face crop
    LABEL_KEY_CANDIDATES = ("label", "labels", "spoof_label", "cls")
    ds = load_dataset(hf_id, split=split, streaming=True)
    feats = ds.features or {}
    image_key = next((k for k in IMAGE_KEY_CANDIDATES if k in feats), None)
    label_key = next((k for k in LABEL_KEY_CANDIDATES if k in feats), None)
    if image_key is None or label_key is None:
        raise SystemExit(
            f"Could not find image/label columns in {hf_id}. "
            f"Columns: {list(feats)}. Edit *_KEY_CANDIDATES."
        )
    print(f"  streaming {hf_id} [{split}] image='{image_key}' label='{label_key}' cap={max_samples}")
    if max_samples:
        ds = ds.take(max_samples)

    buf_x, buf_y = [], []
    for row in ds:
        img = row.get(image_key)
        if img is None:
            continue
        buf_x.append(_pil_to_norm_tensor(img, IMAGENET_MEAN, IMAGENET_STD))
        buf_y.append(int(row[label_key]))
        if len(buf_x) == batch:
            yield torch.stack(buf_x).to(device), torch.tensor(buf_y, device=device)
            buf_x, buf_y = [], []
    if buf_x:
        yield torch.stack(buf_x).to(device), torch.tensor(buf_y, device=device)


def _auc(scores, labels):
    """P(real_score > spoof_score). labels: 1=real, 0=spoof."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


# --------------------------------------------------------------------------- #
# Train + export
# --------------------------------------------------------------------------- #
def train(model: FasModel, args, device: torch.device) -> None:
    # Build the (streamed/capped) image-batch loader.
    if args.dataset.startswith("folder:"):
        gen = folder_loader(args.dataset.split("folder:", 1)[1], args.batch_size, device)
    elif args.dataset.startswith("hf:"):
        gen = hf_loader(args.dataset.split("hf:", 1)[1], args.split, args.batch_size,
                        device, args.max_samples)
    else:  # 'synthetic' default (also what --smoke uses)
        gen = synthetic_loader(args.smoke_n, args.batch_size, device)

    # Frozen backbone → extract features ONCE (one pass; only one image batch is
    # on the GPU at a time, and we keep just the [N, D] feature matrix). The head
    # then trains in milliseconds/epoch on the cached features.
    print("extracting frozen-backbone features (one pass)...")
    model.backbone.eval()
    t0 = time.time()
    Xs, Ys = [], []
    with torch.no_grad():
        for xb, yb in gen:
            Xs.append(model.backbone(xb).detach().cpu())
            Ys.append(yb.detach().cpu())
    if not Xs:
        raise SystemExit("No data materialized — check the dataset id / columns.")
    X = torch.cat(Xs)
    Y = torch.cat(Ys)
    if args.flip_labels:
        Y = 1 - Y
    n = X.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    X, Y = X[perm], Y[perm]
    nval = max(1, n // 5)
    Xval, Yval, Xtr, Ytr = X[:nval], Y[:nval], X[nval:], Y[nval:]
    print(f"  {n} samples (feat dim {X.size(1)}) in {time.time() - t0:.1f}s; "
          f"{Xtr.size(0)} train / {nval} val; "
          f"real={int((Y == 1).sum())} spoof={int((Y == 0).sum())}")

    head = model.head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    bs = args.batch_size
    for epoch in range(args.epochs):
        head.train()
        ep = torch.randperm(Xtr.size(0))
        loss_sum, correct, seen = 0.0, 0, 0
        for i in range(0, Xtr.size(0), bs):
            idx = ep[i:i + bs]
            xb = Xtr[idx].to(device)
            yb = Ytr[idx].to(device)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * xb.size(0)
            correct += int((logits.argmax(1) == yb).sum())
            seen += xb.size(0)
        head.eval()
        with torch.no_grad():
            preal = torch.softmax(head(Xval.to(device)), 1)[:, 1].cpu().tolist()
        auc = _auc(preal, Yval.tolist())
        print(f"epoch {epoch + 1}/{args.epochs}  loss={loss_sum / max(1, seen):.4f}  "
              f"train_acc={correct / max(1, seen):.3f}  val_AUC={auc:.3f}")
    model.head = head


def export_onnx(model: FasModel, out_path: Path, fp16: bool, device: torch.device) -> None:
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    # Static shape (batch=1) -> WebGPU graph-capture eligible.
    # dynamo=False uses the legacy TorchScript exporter: no onnxscript dep, and
    # the clean static graph onnxruntime-web's WASM/WebGPU EPs expect.
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["pixel_values"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamic_axes=None,
        dynamo=False,
    )
    print(f"exported ONNX -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    if fp16:
        try:
            import onnx
            from onnxconverter_common import float16
            m = onnx.load(str(out_path))
            m16 = float16.convert_float_to_float16(m, keep_io_types=True)
            fp16_path = out_path.with_suffix(".fp16.onnx")
            onnx.save(m16, str(fp16_path))
            print(f"exported fp16 ONNX -> {fp16_path}  "
                  f"({fp16_path.stat().st_size / 1e6:.1f} MB)")
        except ImportError:
            print("  (skipped --fp16: `pip install onnx onnxconverter-common`)")


def verify_onnx(out_path: Path) -> None:
    """Load the exported model in onnxruntime and run one frame — proves the
    artifact is a valid ORT graph with the contract the browser expects."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (skipped ORT verify: onnxruntime not installed)")
        return
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"ORT load OK: input {inp.name}{inp.shape} -> output {out.name}{out.shape}")
    x = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    logits = sess.run([out.name], {inp.name: x})[0]
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    print(f"  sample run -> logits {logits.ravel()}  softmax(spoof,real)={p.ravel()}")
    assert logits.shape == (1, NUM_CLASSES), logits.shape
    print("  contract OK: [1,2] logits, index 0=spoof / 1=real")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="dinov2_vits14",
                    help="dinov2_vits14 | dinov2_vitb14 | clip (default dinov2_vits14)")
    ap.add_argument("--dataset", default="synthetic",
                    help="'synthetic' | 'hf:<dataset_id>' (e.g. hf:nguyenkhoa/CASIA-FASD)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("models/fas_head.onnx"))
    ap.add_argument("--fp16", action="store_true", help="also emit a fp16 ONNX")
    ap.add_argument("--flip-labels", action="store_true",
                    help="flip if your dataset uses 1=spoof (target is 1=real)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="no-download plumbing test on synthetic data")
    ap.add_argument("--smoke-n", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=3000,
                    help="cap on streamed HF samples (avoids pulling the full set)")
    args = ap.parse_args()

    # Windows/pyarrow quirk: importing `datasets` (→ pyarrow) AFTER CUDA has been
    # initialized throws WinError 6714 during importlib's directory scan. Force
    # the import now, before any CUDA work, when we'll need it.
    if args.dataset.startswith("hf:"):
        import datasets  # noqa: F401

    device = torch.device(args.device)
    print(f"device={device}  backbone={'smoke-stub' if args.smoke else args.backbone}  "
          f"dataset={'synthetic' if args.smoke else args.dataset}")

    backbone, feat_dim = build_backbone(args.backbone, args.smoke)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    head = FasHead(feat_dim)
    model = FasModel(backbone, head).to(device)

    train(model, args, device)
    export_onnx(model, args.out, args.fp16, device)
    verify_onnx(args.out)
    print("DONE.")


if __name__ == "__main__":
    main()
