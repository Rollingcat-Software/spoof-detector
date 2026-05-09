"""CelebA-Spoof dataset adapter.

CelebA-Spoof (Zhang et al., ECCV 2020) — 625k images across 10 spoof
types and 43 attribute annotations. Most diverse FAS dataset by attack
taxonomy.

Dataset:
    https://github.com/ZhangYuanhan-AI/CelebA-Spoof
    Released under the CelebA license.

Layout:
    CelebA-Spoof/
        Data/
            train/<id>/live/*.png
            train/<id>/spoof/*.png
            test/...
        metas/intra_test/{train,test}_label.txt
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tests.benchmark.runner import Sample

# 10 spoof type codes (from CelebA-Spoof paper Table 2)
_SPOOF_TYPE_NAMES = {
    "0": "live",
    "1": "photo",
    "2": "poster",
    "3": "a4_paper",
    "4": "face_mask",
    "5": "upper_body_mask",
    "6": "region_mask",
    "7": "pc_screen",
    "8": "pad_screen",
    "9": "phone_screen",
    "10": "3d_mask",
}


def iter_celeba_spoof(root: Path | str, split: str = "test") -> Iterator[Sample]:
    """Yield CelebA-Spoof samples using the intra-test label list."""
    root = Path(root)
    label_file = root / "metas" / "intra_test" / f"{split}_label.txt"
    if not label_file.exists():
        raise FileNotFoundError(label_file)

    with label_file.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            relpath, code = parts[0], parts[-1]  # last column is spoof type id
            full = root / relpath
            is_bonafide = code == "0"
            yield Sample(
                sample_id=relpath,
                is_bonafide=is_bonafide,
                attack_type=None if is_bonafide else _SPOOF_TYPE_NAMES.get(code, f"type_{code}"),
                payload=str(full),
                metadata={"split": split, "spoof_code": code},
            )
