"""Capture per-sample scores with a MiniFASNet V1SE+V2 ensemble.

Adds V1SE alongside the existing V2 by running a second MiniFASNet ONNX
session, and folds both raw scores into the per-sample analyzer dict.
The downstream fusion learns weights for `minifasnet` (V2) and
`minifasnet_v1se` separately.

This is the §8.5/§8.x ensemble experiment for the accuracy push.

Usage:
    python -m tests.benchmark.capture_ensemble \\
        --dataset casia_fasd --root /tmp/fas_datasets/akahana_casiafasd/extracted \\
        --split test --out paper/figures/captures/casia_fasd_test_ensemble.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def load_v1se_analyzer():
    """Load a MiniFASNetAnalyzer-equivalent that uses V1SE weights instead of V2."""
    from uniface.spoofing import MiniFASNet, MiniFASNetWeights

    class V1SEAnalyzer:
        name = "minifasnet_v1se"
        def __init__(self):
            self._spoofer = MiniFASNet(model_name=MiniFASNetWeights.V1SE)
            self._frame = None
        def set_frame(self, frame):
            self._frame = frame
        def analyze(self, crop, face):
            from src.domain.models import AnalyzerResult
            import time as _time
            start = _time.perf_counter()
            try:
                if self._frame is not None and face is not None and face.bbox is not None:
                    bbox = face.bbox
                    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                    result = self._spoofer.predict(self._frame, [x1, y1, x2 - x1, y2 - y1])
                else:
                    result = self._spoofer.predict(crop, [0, 0, crop.shape[1], crop.shape[0]])
                # SpoofingResult: .is_real (bool) + .confidence (0..1)
                # convert to 0..100 live-ness: real → confidence*100, attack → (1-confidence)*100
                if getattr(result, "is_real", True):
                    live_score = float(result.confidence) * 100.0
                else:
                    live_score = (1 - float(result.confidence)) * 100.0
            except Exception as e:
                logger.debug(f"V1SE failed: {e}")
                live_score = 50.0
            elapsed = (_time.perf_counter() - start) * 1000
            return AnalyzerResult(
                name=self.name,
                score=float(live_score),
                elapsed_ms=elapsed,
                details={},
            )
    return V1SEAnalyzer()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=False)
    p.add_argument("--split", default=None)
    p.add_argument("--protocol", default="capture_ensemble")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    from tests.benchmark.pipelines._common import build_hybrid_pipeline, load_frames

    pipeline, _ = build_hybrid_pipeline()
    v1se = load_v1se_analyzer()

    # Inject V1SE into the pipeline's face_analyzers
    pipeline._face_analyzers.append(v1se)
    logger.info("hybrid pipeline + V1SE ensemble armed")

    if args.dataset == "casia_fasd":
        from tests.benchmark.datasets.casia_fasd import iter_casia_fasd
        if args.root is None: raise SystemExit("--root required")
        samples_iter = iter_casia_fasd(args.root, split=args.split or "test")
    else:
        from tests.benchmark.run import _load_adapter
        samples_iter = _load_adapter(args.dataset, args.root, args.protocol)

    if args.limit:
        from itertools import islice
        samples_iter = islice(samples_iter, args.limit)

    out: list[dict] = []
    t0 = time.perf_counter()
    for i, sample in enumerate(samples_iter):
        if i % 100 == 0: logger.info("processed %d", i)
        frames = load_frames(sample, max_frames=1)
        if not frames:
            continue
        analysis = pipeline.process(frames[0])

        record = {
            "sample_id": sample.sample_id,
            "is_bonafide": sample.is_bonafide,
            "attack_type": sample.attack_type,
            "n_faces": len(analysis.faces),
            "face_detected": bool(analysis.classifications),
        }

        if not analysis.classifications:
            record["analyzer_scores"] = {}
            record["face_area_px"] = 0
            out.append(record)
            continue

        best_face_id = max(analysis.classifications.keys(),
                          key=lambda fid: analysis.classifications[fid].confidence or 0.0)
        best_cls = analysis.classifications[best_face_id]
        best_face = next((f for f in analysis.faces if f.face_id == best_face_id), None)

        record["analyzer_scores"] = {
            name: float(ar.score) for name, ar in best_cls.analyzer_results.items()
        }
        if best_face and best_face.bbox is not None:
            bbox = best_face.bbox
            record["face_area_px"] = int((bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1))
        else:
            record["face_area_px"] = 0
        out.append(record)

    elapsed = time.perf_counter() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "split": args.split or args.protocol,
        "ensemble": True,
        "n_samples": len(out),
        "elapsed_sec": elapsed,
        "per_sample": out,
    }, indent=2, default=str))

    bf = sum(1 for r in out if r["is_bonafide"])
    detected = sum(1 for r in out if r["face_detected"])
    print(f"\n=== ensemble capture: {len(out)} samples in {elapsed:.1f}s ===")
    print(f"  bonafide={bf}  attack={len(out)-bf}")
    print(f"  face_detected={detected} ({100*detected/len(out):.1f}%)")
    if out:
        keys = sorted(out[0].get("analyzer_scores", {}).keys())
        print(f"  analyzers captured: {keys}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
