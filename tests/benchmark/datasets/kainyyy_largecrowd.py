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

# Paper-wide subsampling seed (matches the bootstrap / resample convention).
_DEFAULT_SEED = 42


def _make_sample(path: Path, is_bf: bool) -> Sample:
    m = _ID_RE.search(path.name)
    subject_id = int(m.group(1)) if m else None
    device_chain = path.stem.split("_id")[0] if "_id" in path.stem else None
    return Sample(
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


def iter_kainyyy_largecrowd(
    root: Path | str,
    split: Literal["all", "live", "spoof"] = "all",
    *,
    interleave: bool = True,
    seed: int = _DEFAULT_SEED,
) -> Iterator[Sample]:
    """Yield largeCrowd-spoof samples.

    Args:
        root: path to the dataset root containing `largeCrowd-spoof/`.
        split: "live", "spoof", or "all".
        interleave: when ``split="all"``, deterministically shuffle the
            combined live+spoof stream (seeded by ``seed``) so a downstream
            ``--limit N`` truncation yields BOTH classes. Without this the
            adapter emitted all 720 ``live/`` stills before any of the 2 891
            ``spoof/`` stills, so any capped run (e.g. the n200 protocol)
            saw only bona-fide samples — producing the degenerate
            ``n_attack=0, AUC=0.0`` results stored before this fix. The
            label mapping itself was already correct (``live/``→bona-fide,
            ``spoof/``→attack); the defect was purely emission order under a
            sample cap. Set ``interleave=False`` to recover the legacy
            folder-ordered stream (e.g. for a full uncapped run).
        seed: RNG seed for the deterministic interleave (default 42, the
            paper-wide subsampling seed).

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

    paths: list[tuple[Path, bool]] = []
    for subdir, is_bf in subdirs:
        folder = root / subdir
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in _IMAGE_EXTS:
                continue
            paths.append((path, is_bf))

    # Deterministically shuffle so a downstream --limit captures both classes.
    # Only meaningful for split="all"; single-class splits are left in
    # filename order for traceability.
    if interleave and split == "all":
        import random
        random.Random(seed).shuffle(paths)

    for path, is_bf in paths:
        yield _make_sample(path, is_bf)
