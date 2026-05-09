"""AxonData attack-only video subsets (HuggingFace, CC-BY-4.0).

AxonData publishes attack-only sample subsets of their commercial
liveness datasets under CC-BY-4.0 on HuggingFace. There are no
bonafide videos in these mirrors — they are intended to be paired
with a separate live-video corpus (e.g. our in-house Marmara set, or
Kainyyy live PNGs decoded into clips) when training a binary PAD
classifier.

Sources:
    - https://huggingface.co/datasets/AxonData/Anti_Spoofing_Cut_print_attack
        Cutout 2D-mask print attacks (~15 sample videos, ~1.1 GB)
    - https://huggingface.co/datasets/AxonData/3D_paper_mask_attack_dataset_for_Liveness
        3D paper-mask attacks (~15 sample videos, ~170 MB)

License: CC-BY-4.0 (commercial use of the *full* set requires
Axon Labs licensing — these HF previews are the freely available
research subsets only).

Layout:
    <root>/
        *.mp4   *.MOV       # videos at root
        Axon Labs Cutout Sample/    # additional sample subdir on cut_print
            *.mp4 *.MOV
        Sample/                      # sometimes named just "Sample"

Counts (verified 2026-05-09):
    cut_print : 15 videos (~1.1 GB)
    3d_mask   : 15 videos (~170 MB)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

from tests.benchmark.runner import Sample

_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}

_ATTACK_TYPE_BY_KIND = {
    "cut_print": "print",
    "3d_mask": "mask",
}


def iter_axon_video(
    root: Path | str,
    kind: Literal["cut_print", "3d_mask"],
) -> Iterator[Sample]:
    """Yield videos from one Axon HF preview subset.

    Args:
        root: path to the dataset root (the directory `huggingface-cli download`
              dropped the videos into).
        kind: "cut_print" (Cutout 2D print masks) or
              "3d_mask"  (3D paper masks).

    Each yielded Sample has:
        sample_id     = filename stem
        is_bonafide   = always False — these are attack-only subsets
        attack_type   = "print" for cut_print, "mask" for 3d_mask
        payload       = absolute path to the video
        metadata      = {source, kind, subdir}
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Axon root missing: {root}")
    attack_type = _ATTACK_TYPE_BY_KIND[kind]

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _VIDEO_EXTS:
            continue
        # skip HF cache dir
        if ".cache" in path.parts:
            continue
        rel_subdir = str(path.parent.relative_to(root)) or "."
        yield Sample(
            sample_id=path.stem,
            is_bonafide=False,
            attack_type=attack_type,
            payload=str(path),
            metadata={
                "source": f"axon_hf_{kind}",
                "kind": kind,
                "subdir": rel_subdir,
            },
        )
