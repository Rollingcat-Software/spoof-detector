"""MiniFASNet-only baseline pipeline.

The simplest possible image-level pipeline — only the MiniFASNet ONNX
model, no fuser, no other analyzers. This is the ablation row that
proves "the strong per-frame discriminator alone is X" — context for
the paper's claim that hybrid > image_only > video_only.

Use this as the method-comparison ceiling for the image track.
"""
from __future__ import annotations

from tests.benchmark.runner import Sample
from tests.benchmark.pipelines._common import load_frames


def score_sample(sample: Sample, *, max_frames: int = 30) -> float:
    """Return P(REAL) — average MiniFASNet score across decoded frames."""
    from src.infrastructure.analyzers.minifasnet_analyzer import MiniFASNetAnalyzer
    from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
    detector = MediaPipeFaceDetector()
    analyzer = MiniFASNetAnalyzer()

    frames = load_frames(sample, max_frames=max_frames)
    if not frames:
        return 0.5

    scores: list[float] = []
    for frame in frames:
        faces = detector.detect(frame)
        if not faces:
            continue
        # MiniFASNet wants the original frame + bbox.
        for face in faces:
            if hasattr(analyzer, "set_frame"):
                analyzer.set_frame(frame)
            crop = face.crop if face.crop is not None else frame
            try:
                result = analyzer.analyze(crop, face)
                # result.score is in [0, 100]; convert to [0, 1] live-ness
                scores.append(float(result.score) / 100.0)
            except Exception:
                continue

    if not scores:
        return 0.5
    return sum(scores) / len(scores)
