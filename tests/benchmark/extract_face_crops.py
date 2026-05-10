"""Extract face crops from a dataset for CNN training.

For each sample, runs MediaPipe face detection, crops the face with
margin, resizes to a fixed size, saves as a numpy uint8 array along with
the label.

Output:
  paper/figures/captures/<dataset>_<split>_crops.npz
    crops:  (N, H, W, 3) uint8 BGR
    labels: (N,) int8 — 1 bonafide, 0 attack
    vids:   (N,) str — parent video ID (for per-video aggregation)
    sample_ids: (N,) str

Usage:
  python -m tests.benchmark.extract_face_crops \\
      --dataset casia_fasd --root /tmp/.../extracted --split train \\
      --crop-size 96 --margin 0.2 \\
      --out paper/figures/captures/casia_fasd_train_crops.npz
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def parent_video(sample_id: str) -> str:
    m = re.match(r"^(.+\.avi)_\d+_(real|fake)$", sample_id)
    return m.group(1) if m else sample_id


def crop_face(frame: np.ndarray, bbox, size: int, margin: float) -> np.ndarray:
    """Crop face with margin, resize to (size, size, 3)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    cw, ch = x2 - x1, y2 - y1
    mx, my = int(cw * margin), int(ch * margin)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=False)
    p.add_argument("--split", default=None)
    p.add_argument("--protocol", default="extract_crops")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--crop-size", type=int, default=96)
    p.add_argument("--margin", type=float, default=0.2,
                   help="Fraction of bbox dim to add as padding (default 0.2)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector

    if args.dataset == "casia_fasd":
        from tests.benchmark.datasets.casia_fasd import iter_casia_fasd
        if args.root is None: raise SystemExit("--root required")
        samples = iter_casia_fasd(args.root, split=args.split or "test")
    else:
        from tests.benchmark.run import _load_adapter
        samples = _load_adapter(args.dataset, args.root, args.protocol)

    if args.limit:
        from itertools import islice
        samples = islice(samples, args.limit)

    detector = MediaPipeFaceDetector()

    crops, labels, vids, sample_ids = [], [], [], []
    n_skipped = 0
    n_processed = 0
    for sample in samples:
        n_processed += 1
        if n_processed % 200 == 0:
            logger.info("processed %d (kept %d, skipped %d)", n_processed, len(crops), n_skipped)
        # Read image
        if isinstance(sample.payload, bytes):
            arr = np.frombuffer(sample.payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif isinstance(sample.payload, str):
            img = cv2.imread(sample.payload)
        else:
            n_skipped += 1
            continue
        if img is None:
            n_skipped += 1
            continue

        faces = detector.detect(img)
        if not faces:
            n_skipped += 1
            continue
        # Pick largest face
        biggest = max(faces, key=lambda f: (f.bbox.x2 - f.bbox.x1) * (f.bbox.y2 - f.bbox.y1))
        crop = crop_face(img, biggest.bbox, args.crop_size, args.margin)
        crops.append(crop)
        labels.append(1 if sample.is_bonafide else 0)
        vids.append(parent_video(sample.sample_id))
        sample_ids.append(sample.sample_id)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        crops=np.array(crops, dtype=np.uint8),
        labels=np.array(labels, dtype=np.int8),
        vids=np.array(vids, dtype=str),
        sample_ids=np.array(sample_ids, dtype=str),
        crop_size=args.crop_size,
        margin=args.margin,
    )
    print(f"\n=== extracted {len(crops)}/{n_processed} ({n_processed - len(crops)} skipped) ===")
    print(f"  bonafide={sum(labels)}, attack={len(labels) - sum(labels)}")
    print(f"  unique videos: {len(set(vids))}")
    print(f"  crops shape: {np.array(crops).shape}, dtype={np.array(crops).dtype}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
