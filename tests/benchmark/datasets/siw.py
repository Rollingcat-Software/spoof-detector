"""SiW (Spoof in the Wild) dataset adapter.

SiW (Liu et al., CVPR 2018) — challenging videos with subject diversity
and varied lighting / pose / camera quality.

Dataset:
    http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html
    Access via Michigan State University EULA.

Layout (typical release):
    SiW/
        live/
            Live/
                <SubjectID>-<Session>-<Replay>.mp4
        spoof/
            Spoof/
                <SubjectID>-<Session>-<Type>-<Detail>.mp4
                  Type: 1=replay, 2=print

Citation:
    Y. Liu, A. Jourabloo, X. Liu,
    "Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary
    Supervision", CVPR 2018.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tests.benchmark.runner import Sample


def iter_siw(root: Path | str) -> Iterator[Sample]:
    """Yield SiW samples (live + spoof, no train/test split).

    The official protocol splits subjects 1-90 train / 91-165 test;
    apply that filter via the runner's include_subjects machinery.
    """
    root = Path(root)
    live_root = root / "live" / "Live"
    spoof_root = root / "spoof" / "Spoof"

    for path in sorted(live_root.glob("*.mp4")):
        yield Sample(
            sample_id=path.stem,
            is_bonafide=True,
            attack_type=None,
            payload=str(path),
            metadata={"split": "live"},
        )

    for path in sorted(spoof_root.glob("*.mp4")):
        # SiW spoof filenames encode the type — second-to-last char "1" replay, "2" print
        parts = path.stem.split("-")
        attack_type = "replay" if parts[2] == "1" else "print"
        yield Sample(
            sample_id=path.stem,
            is_bonafide=False,
            attack_type=attack_type,
            payload=str(path),
            metadata={"split": "spoof", "siw_type_code": parts[2]},
        )
