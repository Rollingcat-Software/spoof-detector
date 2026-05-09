"""Kainyyy/face-anti-spoof — largeCrowd-spoof subset (HuggingFace, open).

A 3611-still-image FAS set published on HuggingFace by user `Kainyyy`.
No declared license on the dataset card; treat as research-only. The
filenames encode device + subject + scene tags but not the original
PAI taxonomy, so we expose attack_type="unknown" for spoof samples.

Source:
    https://huggingface.co/datasets/Kainyyy/face-anti-spoof

Layout:
    largeCrowd-spoof/
        live/   *.png     # 720 bonafide stills
        spoof/  *.png     # 2891 attack stills (mostly print + replay)

Sample filenames look like:
    live/AGL752VM_id147_s0_120.png            # device_subjectid_scene_frame
    spoof/FT720P_G780_REDMI4X_id0_s0_120.png  # device_chain_subjectid_scene_frame

We extract `subject` (the `id<n>` token) and `device` (everything before
`_id`) for traceability — useful for subject-disjoint protocols if a
paper needs them. We do not attempt to disentangle attack type from the
device chain since the dataset card is silent on it.

Counts (verified 2026-05-09 against the Kainyyy mirror):
    live  : 720  (719 .png + 1 .jpg)
    spoof : 2891 (2889 .png + 2 .jpg)
    total : 3611
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Literal

from tests.benchmark.runner import Sample

# Pull subject id out of e.g. "AGL752VM_id147_s0_120.png"
_ID_RE = re.compile(r"_id(\d+)_", re.IGNORECASE)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def iter_kainyyy_largecrowd(
    root: Path | str,
    split: Literal["all", "live", "spoof"] = "all",
) -> Iterator[Sample]:
    """Yield largeCrowd-spoof samples.

    Args:
        root: path to the dataset root containing `largeCrowd-spoof/`.
        split: "live", "spoof", or "all".

    Each yielded Sample has:
        sample_id     = filename stem
        is_bonafide   = file is under `live/`
        attack_type   = None for live, "unknown" for spoof
        payload       = absolute path to the .png
        metadata      = {device_chain, subject_id, source}
    """
    root = Path(root) / "largeCrowd-spoof"
    if not root.exists():
        raise FileNotFoundError(f"Kainyyy largeCrowd-spoof root missing: {root}")

    subdirs = []
    if split in ("all", "live"):
        subdirs.append(("live", True))
    if split in ("all", "spoof"):
        subdirs.append(("spoof", False))

    for subdir, is_bf in subdirs:
        folder = root / subdir
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in _IMAGE_EXTS:
                continue
            m = _ID_RE.search(path.name)
            subject_id = int(m.group(1)) if m else None
            device_chain = path.stem.split("_id")[0] if "_id" in path.stem else None
            yield Sample(
                sample_id=path.stem,
                is_bonafide=is_bf,
                attack_type=None if is_bf else "unknown",
                payload=str(path),
                metadata={
                    "device_chain": device_chain,
                    "subject_id": subject_id,
                    "source": "kainyyy_hf",
                },
            )
