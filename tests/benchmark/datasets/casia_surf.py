"""CASIA-SURF dataset adapter.

CASIA-SURF (Zhang et al., CVPR Workshops 2019) — large-scale multi-modal
(RGB+depth+IR) PAD dataset. We only consume the RGB modality for parity
with our pipeline, but we record the depth/ir paths in metadata for
future cross-modal experiments.

Dataset:
    http://www.cbsr.ia.ac.cn/users/jwan/database/CASIA-SURF.html
    Access via Chinese Academy of Sciences EULA.

Layout:
    CASIA-SURF/
        Training/
            real_part/<id>/<sample>.jpg
            fake_part/<id>/<sample>.jpg
        ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tests.benchmark.runner import Sample


def iter_casia_surf(root: Path | str, split: str = "Training") -> Iterator[Sample]:
    root = Path(root) / split
    for kind, is_bonafide in (("real_part", True), ("fake_part", False)):
        for sample_dir in sorted((root / kind).iterdir() if (root / kind).exists() else []):
            if not sample_dir.is_dir():
                continue
            for path in sorted(sample_dir.glob("*.jpg")):
                yield Sample(
                    sample_id=f"{kind}/{sample_dir.name}/{path.stem}",
                    is_bonafide=is_bonafide,
                    attack_type=None if is_bonafide else "casia_surf",
                    payload=str(path),
                    metadata={"split": split, "kind": kind, "subject_dir": str(sample_dir)},
                )
