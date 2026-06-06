"""CelebA-Spoof TRAIN shard adapter (separate from the eval-shard adapter).

Same parquet schema as the eval shard but distinct file pattern. Used
to build a learned classifier on CelebA-Spoof train, evaluated on the
eval shard.

Source:
    huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing
    train-NNNNN-of-00093.parquet (93 shards, ~5300 samples each)

Usage in tests/benchmark/run.py:
    if dataset == "celeba_train_shard":
        from tests.benchmark.datasets.celeba_train_shard import iter_celeba_train_shard
        return iter_celeba_train_shard(parquet_paths=[root], use_cropped=True)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from tests.benchmark.runner import Sample


def iter_celeba_train_shard(
    parquet_paths,
    *,
    use_cropped: bool = True,
    yield_bytes: bool = True,
    limit: Optional[int] = None,
) -> Iterator[Sample]:
    """Stream samples from one or more CelebA-Spoof TRAIN parquet shards.

    Schema in the train shards: cropped_image, labels (int 0/1), labelNames ("live"/"spoof").
    NOTE: in the train shards, labels=1 is SPOOF and labels=0 is LIVE — confirmed via labelNames.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("celeba_train_shard adapter requires pandas + pyarrow") from e

    if not isinstance(parquet_paths, (list, tuple)):
        parquet_paths = [parquet_paths]

    n = 0
    for p in parquet_paths:
        p = Path(p)
        df = pd.read_parquet(p)
        col = "cropped_image" if use_cropped else "image"
        for i, row in df.iterrows():
            label_name = row["labelNames"]
            is_bf = label_name == "live"
            data = row[col]
            payload = data["bytes"] if yield_bytes and isinstance(data, dict) else data

            yield Sample(
                sample_id=str(row.get("image_path", f"{p.stem}_row{i}")),
                is_bonafide=is_bf,
                attack_type=None if is_bf else "unknown",
                payload=payload,
                metadata={"parquet": str(p), "row": int(i), "source": "celeba_train_shard"},
            )
            n += 1
            if limit and n >= limit:
                return
