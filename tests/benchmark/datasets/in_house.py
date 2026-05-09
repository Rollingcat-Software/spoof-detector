"""In-house Marmara University 43-sample test set.

Used as the calibration / smoke set across all paper experiments.
The set lives in `data/in_house/` with `labels.csv` mapping filenames
to (is_bonafide, attack_type).

This adapter is the only one that can run without an external EULA —
the data was collected with consent from study participants under
KVKK / GDPR Art. 6(1)(a) and is included in the repo for reproducibility.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from tests.benchmark.runner import Sample


def iter_in_house(root: Path | str | None = None) -> Iterator[Sample]:
    """Yield in-house samples.

    Looks for `<root>/labels.csv` with columns: filename, is_bonafide, attack_type.
    Skips silently if no labels file exists (in case the data hasn't been
    populated yet — useful for CI smoke tests).
    """
    if root is None:
        root = Path(__file__).resolve().parents[3] / "data" / "in_house"
    root = Path(root)
    labels = root / "labels.csv"
    if not labels.exists():
        return  # empty stream — caller will get N=0 result

    with labels.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            full = root / row["filename"]
            is_bf = str(row["is_bonafide"]).lower() in {"1", "true", "yes"}
            yield Sample(
                sample_id=row["filename"],
                is_bonafide=is_bf,
                attack_type=None if is_bf else row.get("attack_type") or "unknown",
                payload=str(full),
                metadata={"source": "marmara_in_house"},
            )
