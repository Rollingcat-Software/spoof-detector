"""Synthesize plausible-physics PA samples from bona-fide face images.

This is an *in-house validation* tool — NOT a substitute for academic
benchmarks (OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof). It produces
attack samples that exhibit the *physical* artefacts real attacks
have, so the pipeline analyzers see signal:

  print_attack(img)
    Simulates a printed photograph held up to the camera:
      - Gamma compression (paper has narrower dynamic range)
      - Slight Gaussian blur (paper texture + lens slightly out of focus)
      - Paper-grain Poisson noise (simulating ink-dot pattern)
      - Slight desaturation (CMYK gamut narrower than RGB)

  replay_attack(img)
    Simulates rephotographing a phone/monitor displaying the image:
      - Per-pixel quantization to 5-6 bits (LCD bit depth)
      - Synthetic moire by overlaying a sub-pixel grid
      - Brightness shift (screen contrast vs ambient lighting)
      - Slight horizontal banding (rolling shutter beat)

  ar_filter(img)
    Simulates a beauty / face-swap AR filter:
      - Bilateral filter (skin smoothing, the primary AR-filter signature)
      - Tone-shift (warmer skin)
      - Eye / lip enhancement (mild colour saturation in those regions)

  digital_photo(img)
    Simulates a static digital image displayed on a phone (no rephotograph):
      - Effectively the original image at slightly lower resolution + JPEG round-trip

These are physics-motivated approximations, not random noise. They are
sufficient to validate the pipeline produces sensible per-class scores;
they are NOT sufficient as paper-grade evidence — the paper's headline
numbers come from the four academic benchmarks.

Usage:
    python -m tests.benchmark.synthesize_attacks \\
        --src /path/to/real/faces \\
        --out data/in_house

Generates:
    data/in_house/
        bonafide/<subject>_<idx>.jpg
        attack_print/<subject>_<idx>.jpg
        attack_replay/<subject>_<idx>.jpg
        attack_ar_filter/<subject>_<idx>.jpg
        attack_digital_photo/<subject>_<idx>.jpg
        labels.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ============================================================================
# Attack synthesisers
# ============================================================================

def print_attack(img: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Print attack: gamma compression + paper grain + slight blur + desat."""
    out = img.astype(np.float32) / 255.0
    # Gamma compression (paper has narrower dynamic range)
    gamma = float(rng.uniform(0.7, 0.85))
    out = np.clip(out ** gamma, 0, 1)
    # Slight desaturation (CMYK gamut narrower than RGB)
    hsv = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.85
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    # Slight Gaussian blur (paper texture + lens slightly out of focus)
    sigma = float(rng.uniform(0.6, 1.2))
    out = cv2.GaussianBlur(out, (0, 0), sigma)
    # Paper-grain Poisson noise — simulates ink-dot dithering
    out = out + rng.normal(0, 0.012, out.shape).astype(np.float32)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def replay_attack(img: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Replay attack: full rephotograph simulation — visible bezel,
    heavy moire, screen pixel grid, color cast, specular highlight."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # 1. STRONG LCD sub-pixel grid: every 3rd column dimmed visibly.
    grid = np.ones((h, w, 3), dtype=np.float32)
    grid[:, ::3, :]  *= 0.82  # noticeable R-column darkening
    grid[:, 1::3, :] *= 0.92
    out = out * grid

    # 2. Visible scan-lines (rolling-shutter / 60 Hz refresh beat).
    band_freq = float(rng.uniform(0.04, 0.10))
    band = 12 * np.sin(np.linspace(0, h * band_freq * 2 * np.pi, h)).astype(np.float32)
    out = out + band[:, None, None]

    # 3. Heavy moire (clearly visible Gabor-like interference).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    freq = float(rng.uniform(0.08, 0.16))  # 2x stronger than before
    angle = float(rng.uniform(0, np.pi))
    moire = 18 * np.sin(2 * np.pi * (xx * np.cos(angle) + yy * np.sin(angle)) * freq)
    out = out + moire[..., None]

    # 4. Cool color cast (LCD blue tint).
    out[..., 0] = np.clip(out[..., 0] * 1.08, 0, 255)  # B up
    out[..., 2] = np.clip(out[..., 2] * 0.94, 0, 255)  # R down

    # 5. Bit-depth quantization (6 bits/ch).
    out = np.clip(out, 0, 255)
    out = (out.astype(np.uint8) // 4) * 4

    # 6. Specular highlight from camera flash (small bright Gaussian blob).
    blob_y = int(rng.uniform(0.2, 0.5) * h)
    blob_x = int(rng.uniform(0.3, 0.7) * w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = 60.0 * np.exp(-((yy - blob_y) ** 2 + (xx - blob_x) ** 2) / (2 * (max(h, w) * 0.06) ** 2))
    out = np.clip(out.astype(np.float32) + blob[..., None], 0, 255).astype(np.uint8)

    # 7. Surrounding bezel — black border (12% of long edge).
    border = int(0.12 * max(h, w))
    framed = np.zeros((h + 2 * border, w + 2 * border, 3), dtype=np.uint8)
    framed[border:border + h, border:border + w] = out
    # Small specular reflection on the bezel itself.
    framed[border // 4:border // 2, border:-border] = 16
    return framed


def ar_filter(img: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """AR filter: bilateral skin-smoothing + warm tone-shift + saturation bump."""
    # Bilateral filter (the primary AR-filter signature: edges preserved, skin smoothed)
    out = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    # Warm tone-shift (most beauty filters add warmth)
    out = out.astype(np.float32)
    out[..., 2] = np.clip(out[..., 2] * 1.05, 0, 255)  # red up
    out[..., 0] = np.clip(out[..., 0] * 0.96, 0, 255)  # blue down slightly
    # Saturation bump (eyes/lips region — proxy: whole image)
    hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.10, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def digital_photo(img: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
    """Static digital image displayed on a phone (no rephotograph)."""
    h, w = img.shape[:2]
    # Downsample by ~10-20% then upsample (loss of detail, modest)
    scale = float(rng.uniform(0.80, 0.92))
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    # JPEG round-trip at ~70% quality
    enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 72])[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


ATTACK_FUNCS = {
    "print": print_attack,
    "replay": replay_attack,
    "ar_filter": ar_filter,
    "digital_photo": digital_photo,
}


# ============================================================================
# Main: walk source images, generate spoofs, write labels.csv
# ============================================================================

def iter_source_images(src_root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (subject, image_path) for every .jpg/.png under src_root."""
    for path in sorted(src_root.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # Subject = parent directory name
        subject = path.parent.name
        yield subject, path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--src", required=True, help="Root dir containing real face images.")
    p.add_argument("--out", required=True, help="Output dir for synthesized in-house set.")
    p.add_argument("--max-per-subject", type=int, default=10,
                   help="Cap on bona-fide images per subject (default 10).")
    p.add_argument("--min-size", type=int, default=100,
                   help="Hard reject below this size (default 100).")
    p.add_argument("--upresize-to", type=int, default=256,
                   help="Up-resize images smaller than this (default 256, preserves aspect).")
    p.add_argument("--variants-per-attack", type=int, default=1,
                   help="Stochastic variants per attack type per source (default 1).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rng = np.random.default_rng(args.seed)
    src_root = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "bonafide").mkdir(exist_ok=True)
    for k in ATTACK_FUNCS:
        (out_root / f"attack_{k}").mkdir(exist_ok=True)

    rows: list[dict] = []
    subj_counts: dict[str, int] = {}
    n_bonafide = 0
    n_attacks = 0

    for subject, path in iter_source_images(src_root):
        subj_counts.setdefault(subject, 0)
        if subj_counts[subject] >= args.max_per_subject:
            continue

        img = cv2.imread(str(path))
        if img is None:
            logger.warning("skipping unreadable: %s", path)
            continue
        h, w = img.shape[:2]
        if min(h, w) < args.min_size:
            logger.debug("skipping too-small (%dx%d): %s", w, h, path)
            continue
        # Up-resize images below `upresize_to` so face detector and MiniFASNet have
        # enough pixels (Lanczos for sharpness; preserves aspect).
        if min(h, w) < args.upresize_to:
            scale = args.upresize_to / min(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
            h, w = img.shape[:2]
        # Cap at 1024px on the long edge — pipeline analyzers down-resize anyway.
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        idx = subj_counts[subject]
        stem = f"{subject}_{idx:03d}"

        # Bona-fide
        bf_path = out_root / "bonafide" / f"{stem}.jpg"
        cv2.imwrite(str(bf_path), img)
        rows.append({
            "filename": f"bonafide/{stem}.jpg",
            "is_bonafide": "1",
            "attack_type": "",
            "subject": subject,
            "source_image": str(path.relative_to(src_root)),
        })
        n_bonafide += 1

        # Each attack variant — generate `variants_per_attack` stochastic variants per type
        for attack_name, fn in ATTACK_FUNCS.items():
            for v in range(args.variants_per_attack):
                spoof = fn(img, rng=rng)
                v_stem = f"{stem}_v{v}" if args.variants_per_attack > 1 else stem
                sp_path = out_root / f"attack_{attack_name}" / f"{v_stem}.jpg"
                cv2.imwrite(str(sp_path), spoof)
                rows.append({
                    "filename": f"attack_{attack_name}/{v_stem}.jpg",
                    "is_bonafide": "0",
                    "attack_type": attack_name,
                    "subject": subject,
                    "source_image": str(path.relative_to(src_root)),
                })
                n_attacks += 1

        subj_counts[subject] += 1

    # Write labels.csv
    labels_path = out_root / "labels.csv"
    with labels_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "is_bonafide", "attack_type", "subject", "source_image"])
        w.writeheader()
        w.writerows(rows)

    logger.info("synthesised: %d bona-fide, %d attacks (%d per type), %d subjects, → %s",
                n_bonafide, n_attacks, n_bonafide, len(subj_counts), out_root)
    logger.info("labels written to %s", labels_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
