"""OULU-NPU dataset adapter.

OULU-NPU (Boulkenafet et al., FG 2017) is the de-facto standard FAS
benchmark — every FAS paper of the last 5 years reports on it.

Dataset:
    https://sites.google.com/site/oulunpudatabase/
    Access requires institutional EULA from CMVS / University of Oulu.

Layout (per official release):
    OULU-NPU/
        Train_files/
            <SubjectID>_<Session>_<Phone>_<PAI>.avi
        Dev_files/
            ...
        Test_files/
            ...

Filename grammar (from the dataset README):
    <SubjectID> ∈ 1..55              — 55 subjects
    <Session>   ∈ 1..3               — 3 capture sessions
    <Phone>     ∈ 1..6               — 6 phone models
    <PAI>       ∈ 1 (real)
                  2 (print, paper 1)
                  3 (print, paper 2)
                  4 (replay, monitor 1)
                  5 (replay, monitor 2)

Protocols:
    P1: train/dev/test session-disjoint
    P2: PAI-disjoint
    P3: phone-disjoint (leave-one-camera-out)
    P4: full cross — phone + session + PAI

Citation:
    Z. Boulkenafet, J. Komulainen, L. Li, X. Feng, A. Hadid,
    "OULU-NPU: A mobile face presentation attack database with real-world
    variations", IEEE FG 2017.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from tests.benchmark.runner import Sample

# Filename: <SubjectID>_<Session>_<Phone>_<PAI>.avi
_FILENAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)\.(?:avi|mov|mp4)$", re.IGNORECASE)

_PAI_TYPE = {
    1: ("bonafide", None),
    2: ("attack", "print"),
    3: ("attack", "print"),
    4: ("attack", "replay"),
    5: ("attack", "replay"),
}


def iter_oulu_npu(
    root: Path | str,
    split: str = "test",
    *,
    include_subjects: Optional[set[int]] = None,
    include_sessions: Optional[set[int]] = None,
    include_phones: Optional[set[int]] = None,
    include_pai: Optional[set[int]] = None,
) -> Iterator[Sample]:
    """Yield OULU-NPU samples.

    Args:
        root: path to the unzipped dataset root.
        split: one of "train", "dev", "test".
        include_*: filters for protocol subsets. None means "all".
                   Used to implement the 4 official protocols.

    Each yielded Sample has:
        sample_id     = original filename without extension
        is_bonafide   = PAI == 1
        attack_type   = "print" or "replay" (None for bonafide)
        payload       = absolute path to the .avi file (str)
        metadata      = {subject, session, phone, pai}
    """
    root = Path(root)
    folder = {"train": "Train_files", "dev": "Dev_files", "test": "Test_files"}[split.lower()]
    split_root = root / folder
    if not split_root.exists():
        raise FileNotFoundError(f"OULU-NPU split not found: {split_root}")

    for path in sorted(split_root.iterdir()):
        m = _FILENAME_RE.match(path.name)
        if m is None:
            continue
        subject, session, phone, pai = (int(m.group(i)) for i in (1, 2, 3, 4))
        if include_subjects and subject not in include_subjects: continue
        if include_sessions and session not in include_sessions: continue
        if include_phones and phone not in include_phones: continue
        if include_pai and pai not in include_pai: continue

        kind, attack_type = _PAI_TYPE.get(pai, ("attack", "unknown"))
        yield Sample(
            sample_id=path.stem,
            is_bonafide=(kind == "bonafide"),
            attack_type=attack_type,
            payload=str(path),
            metadata={
                "subject": subject,
                "session": session,
                "phone": phone,
                "pai": pai,
                "split": split,
            },
        )
