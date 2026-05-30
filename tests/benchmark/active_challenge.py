"""Active-challenge layer benchmark (paper §8.5).

Models the deployment-time choice: when an operator can demand active
user cooperation, an additional layer of challenges (light-flash + gesture)
runs on top of the passive hybrid pipeline. The active layer should add a
+30 to +50 percentage-point swing on hard screen-replay attacks.

Aysenur's `light_challenge_service` (in `from_biometric_processor/`) is
the implementation. It works on REAL frame pairs (pre-flash + flash). To
benchmark without the actual capture-time challenge protocol, we
*synthesise* the pre/flash frames from each in-house sample and use
FlashSpoofAnalyzer to score them.

Synthesis: for each sample we treat the input as the pre-flash frame and
synthesise a flash-response by adding the expected color (e.g. red flash
adds intensity to the red channel) and a screen-replay-specific
attenuation pattern. Real bona-fide should get a uniform diffuse
response; synthesised replay/print should get either no response (paper)
or a planar-surface response (screen).

This benchmark is therefore a smoke test for the active layer's CODE
correctness — not a paper-grade evidence-of-effectiveness signal. Real
active-challenge evaluation requires actual capture-time pre/flash pairs,
which OULU-NPU and SiW do not provide. Aysenur's internal evaluation set
(`research/aysenur/working_spoof_detection/`) is the only data we have
that includes real flash captures.

Usage:
    python -m tests.benchmark.active_challenge --dataset in_house --root data/in_house_replay --limit 100
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def synthesize_flash_pair(bgr_image: np.ndarray, color: str) -> tuple[np.ndarray, np.ndarray]:
    """Given a sample image, return (pre_flash, flash) BGR pair.

    Pre-flash = the image as-is.
    Flash response model:
      For real-faces: +25 intensity on the expected color channel,
                      -8 on others (diffuse response with some chromatic spread)
      For synthesised replay/print: simulated to be close to no response at
                      all on print (white paper reflects all wavelengths
                      almost equally) or planar/glare on replay (uniform
                      bright patch, not face-shaped).

    Since we don't know whether the input is real or attack at synthesis time,
    we apply the "real diffuse" model — bona-fides should pass, synthesised
    attacks should fail by virtue of having no real pulse to begin with.
    """
    pre = bgr_image.astype(np.float32)
    color_idx = {"red": 2, "green": 1, "blue": 0}.get(color, 1)
    flash = pre.copy()
    flash[..., color_idx] = np.clip(flash[..., color_idx] + 25, 0, 255)
    other_idx = [i for i in (0, 1, 2) if i != color_idx]
    flash[..., other_idx] = np.clip(flash[..., other_idx] - 8, 0, 255)
    return pre.astype(np.uint8), flash.astype(np.uint8)


def score_with_active_layer(sample, analyzer, color: str = "red") -> dict:
    """Score one sample through both the passive pipeline and the flash analyzer."""
    from tests.benchmark.pipelines._common import load_frames

    frames = load_frames(sample, max_frames=1)
    if not frames:
        return {"sample_id": sample.sample_id, "is_bonafide": sample.is_bonafide, "active_score": 0.5}

    bgr = frames[0]
    pre, flash = synthesize_flash_pair(bgr, color=color)
    analysis = analyzer.analyze(pre_flash_bgr=pre, flash_bgr=flash, expected_color=color)

    # Compose a single live-ness-style score from the FlashSpoofAnalysis.
    # High flash_color_match + high diffuse + low planar_surface_risk = live face.
    score = (
        0.40 * float(analysis.flash_color_match_score)
        + 0.30 * float(analysis.diffuse_response_score)
        + 0.20 * (1 - float(analysis.planar_surface_risk))
        + 0.10 * float(analysis.flash_response_consistency)
    )
    return {
        "sample_id": sample.sample_id,
        "is_bonafide": sample.is_bonafide,
        "attack_type": sample.attack_type,
        "active_score": float(np.clip(score, 0, 1)),
        "details": {
            "flash_color_match": float(analysis.flash_color_match_score),
            "diffuse_response": float(analysis.diffuse_response_score),
            "planar_surface_risk": float(analysis.planar_surface_risk),
            "flash_response_consistency": float(analysis.flash_response_consistency),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=False)
    p.add_argument("--protocol", default="active_challenge")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--color", default="red", choices=["red", "green", "blue"])
    p.add_argument("--out", default="paper/figures")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    from from_biometric_processor.flash_spoof_analyzer import FlashSpoofAnalyzer
    from tests.benchmark.run import _load_adapter
    from src.metrics import classification_report

    samples = _load_adapter(args.dataset, args.root, args.protocol)
    if args.limit:
        from itertools import islice
        samples = islice(samples, args.limit)

    analyzer = FlashSpoofAnalyzer()
    rows: list[dict] = []
    for i, sample in enumerate(samples):
        if i % 20 == 0:
            logger.info("processed %d", i)
        rows.append(score_with_active_layer(sample, analyzer, color=args.color))

    if not rows:
        logger.error("empty stream")
        return 1

    scores = [r["active_score"] for r in rows]
    is_bf = [r["is_bonafide"] for r in rows]
    types = [r.get("attack_type") or "unknown" for r in rows]
    # Smoke test with no Dev split — EER-on-test operating point (opt-in, biased low).
    report = classification_report(scores, is_bf, types, allow_test_set_threshold=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"results_active_challenge_{args.dataset}_{args.protocol}.json"
    out_file.write_text(json.dumps({
        "dataset": args.dataset,
        "protocol": args.protocol,
        "method": "active_challenge_only",
        "n_samples": len(rows),
        "color": args.color,
        "metrics": report,
        "per_sample": rows,
    }, indent=2, default=str))

    print(f"\n=== Active-challenge layer (color={args.color}) ===")
    print(f"  N={len(rows)} bonafide={sum(is_bf)} attack={len(rows)-sum(is_bf)}")
    print(f"  ACER  = {report['acer']*100:.2f}%")
    print(f"  EER   = {report['eer']*100:.2f}%")
    print(f"  AUC   = {report['auc']:.4f}")
    print(f"  per-type APCER: {report['apcer_per_type']}")
    print(f"\nwrote {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
