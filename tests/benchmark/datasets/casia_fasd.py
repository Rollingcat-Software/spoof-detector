"""CASIA-FASD adapter (open-access HF mirror).

CASIA-FASD (Zhang et al., ICB 2012) is one of the foundational FAS
benchmarks. The original release is gated through CASIA, but the
`akahana/anti-spoofing-casiafasd` HuggingFace mirror redistributes the
extracted color/depth frames as a 69 MB tarball with no EULA.

Source:
    https://huggingface.co/datasets/akahana/anti-spoofing-casiafasd

Layout after `tar xzf casiafasd.tar.gz`:
    train_img/train_img/color/<subject>_<scene>.avi_<frame>_<real|fake>.jpg
    train_img/train_img/depth/...
    test_img/test_img/color/...
    test_img/test_img/depth/...

Filename convention encodes the label:
    *_real.jpg  -> bonafide
    *_fake.jpg  -> attack

The original CASIA-FASD attack categories (warped photo, cut photo,
video replay) are NOT preserved in filenames; this mirror flattens to
real/fake only. We expose `attack_type="unknown"` for non-bonafide
samples — paper experiments that need PAI granularity should use the
EULA-bound original release instead.

Counts (verified 2026-05-09 against the akahana mirror):
    train: 1655 frames (404 real / 1251 fake)
    test:  2408 frames

Citation:
    Z. Zhang, J. Yan, S. Liu, Z. Lei, D. Yi, S. Z. Li,
    "A face antispoofing database with diverse attacks", ICB 2012.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Literal

from tests.benchmark.runner import Sample

# Filenames look like: 10_8.avi_50_fake.jpg or 10_HR_2.avi_125_real.jpg
_FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<scene>[A-Z0-9_]+)\.avi_(?P<frame>\d+)_(?P<label>real|fake)\.jpg$",
    re.IGNORECASE,
)


def iter_casia_fasd(
    root: Path | str,
    split: Literal["train", "test"] = "test",
    *,
    modality: Literal["color", "depth"] = "color",
) -> Iterator[Sample]:
    """Yield CASIA-FASD frames from the akahana HF mirror.

    Args:
        root: path to the unpacked tarball — directory containing
              `train_img/` and `test_img/`.
        split: "train" or "test".
        modality: "color" (RGB JPEGs) or "depth" (Kinect depth maps).

    Each yielded Sample has:
        sample_id     = filename stem
        is_bonafide   = filename ends with `_real`
        attack_type   = "unknown" for fake (the mirror does not preserve
                        warped vs. cut vs. replay), None for real
        payload       = absolute path to the .jpg
        metadata      = {subject, scene, frame, modality, split}
    """
    root = Path(root)
    folder = {"train": "train_img/train_img", "test": "test_img/test_img"}[split]
    leaf = root / folder / modality
    if not leaf.exists():
        raise FileNotFoundError(f"CASIA-FASD split not found: {leaf}")

    for path in sorted(leaf.iterdir()):
        m = _FILENAME_RE.match(path.name)
        if m is None:
            continue
        is_bf = m.group("label").lower() == "real"
        yield Sample(
            sample_id=path.stem,
            is_bonafide=is_bf,
            attack_type=None if is_bf else "unknown",
            payload=str(path),
            metadata={
                "subject": int(m.group("subject")),
                "scene": m.group("scene"),
                "frame": int(m.group("frame")),
                "modality": modality,
                "split": split,
                "source": "akahana_hf_mirror",
            },
        )
