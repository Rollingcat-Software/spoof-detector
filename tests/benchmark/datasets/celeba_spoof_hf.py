"""CelebA-Spoof evaluation slice via the nguyenkhoa HF mirror.

The original CelebA-Spoof dataset (CUHK, ECCV 2020) is gated behind
the CelebA EULA. The `nguyenkhoa/antispoofing-3` HuggingFace dataset
republishes a parquet-packed evaluation slice with `cropped_image`,
bounding boxes and `live`/`spoof` labels, downloadable without a EULA
form.

Source:
    https://huggingface.co/datasets/nguyenkhoa/antispoofing-3

For paper benchmarks we use a *single* eval parquet shard
(`data/eval-00000-of-00004.parquet`, ~440 MB, 2611 samples,
874 live / 1737 spoof). The full set is ~16 GB and exceeds our
disk budget; the runner stays parquet-shard-aware so we can extend
to the other 3 eval shards if needed.

Schema per row:
    image_path    : str   — original CelebA-Spoof relative path
    label         : int   — 0 = live, 1 = spoof
    cropped_image : bytes — face-cropped JPEG bytes
    bbox          : [x1, y1, x2, y2] floats over the *original* image
    image         : bytes — full original frame, JPEG-encoded
    labels        : int   — duplicate of `label` (dataset library convention)
    labelNames    : str   — "live" or "spoof"

The `cropped_image` and `image` cells are raw JPEG bytes — a downstream
pipeline reads them via `cv2.imdecode(np.frombuffer(...))`.

Caveat: the original CelebA-Spoof PAI taxonomy
(print / replay / 3D mask / paper-cut / ...) is NOT preserved in the
HF mirror — only live/spoof. Papers reporting on PAI-disjoint
protocols must use the full original release.

Citation:
    Y. Zhang et al., "CelebA-Spoof: Large-Scale Face Anti-Spoofing
    Dataset with Rich Annotations", ECCV 2020.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from tests.benchmark.runner import Sample


def iter_celeba_spoof_hf(
    parquet_paths: list[Path | str] | Path | str,
    *,
    use_cropped: bool = True,
    yield_bytes: bool = True,
    limit: Optional[int] = None,
) -> Iterator[Sample]:
    """Stream samples from one or more nguyenkhoa CelebA-Spoof eval parquets.

    Args:
        parquet_paths: a single parquet file or a list of them.
        use_cropped: if True, payload is the face-cropped JPEG bytes
                     (`cropped_image`); if False, the full-frame bytes
                     (`image`).
        yield_bytes: if True (default), payload is `bytes` — the JPEG
                     buffer. If False, payload is the parquet row index
                     `(parquet_path, row_idx)` — useful when the caller
                     wants to stream-decode itself.
        limit: stop after N samples (None = no cap).

    Each yielded Sample has:
        sample_id     = original `image_path` (or row idx if missing)
        is_bonafide   = labelNames == "live"  (label == 0)
        attack_type   = None for live, "unknown" for spoof
        payload       = JPEG bytes OR (parquet_path, row_idx) tuple
        metadata      = {parquet, row, bbox, original_path, source}
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "celeba_spoof_hf adapter requires pandas + pyarrow"
        ) from e

    if isinstance(parquet_paths, (str, Path)):
        parquet_paths = [parquet_paths]

    img_col = "cropped_image" if use_cropped else "image"
    n_yielded = 0

    for pq_path in parquet_paths:
        pq_path = Path(pq_path)
        df = pd.read_parquet(pq_path)
        for row_idx, row in df.iterrows():
            if limit is not None and n_yielded >= limit:
                return
            label_name = row["labelNames"]
            is_bf = label_name == "live"
            sample_id = row.get("image_path") or f"{pq_path.stem}#{row_idx}"

            if yield_bytes:
                cell = row[img_col]
                # parquet image columns can come back as raw bytes or
                # {"bytes": ..., "path": ...} dicts depending on writer.
                if isinstance(cell, dict):
                    payload = cell.get("bytes") or cell.get("path")
                else:
                    payload = cell
            else:
                payload = (str(pq_path), int(row_idx))

            yield Sample(
                sample_id=str(sample_id),
                is_bonafide=is_bf,
                attack_type=None if is_bf else "unknown",
                payload=payload,
                metadata={
                    "parquet": str(pq_path),
                    "row": int(row_idx),
                    "bbox": list(row["bbox"]) if row.get("bbox") is not None else None,
                    "original_path": row.get("image_path"),
                    "source": "nguyenkhoa_celeba_spoof_hf",
                },
            )
            n_yielded += 1
