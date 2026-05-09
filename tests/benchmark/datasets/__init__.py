"""Dataset adapters for academic FAS benchmarks.

Each module here knows how to walk one dataset's directory layout and
yield Sample(s). Datasets are NOT bundled — see each module's docstring
for the access procedure.

EULA-locked (require institutional access):
    oulu_npu     — OULU-NPU (Oulu, Finland)
    siw          — Spoof in the Wild (Michigan State)
    casia_surf   — CASIA-SURF (Chinese Academy of Sciences)
    celeba_spoof — CelebA-Spoof original release (CUHK)

Open-access HuggingFace mirrors (no EULA, downloadable today):
    casia_fasd          — CASIA-FASD (akahana mirror), 4063 frames, real/fake
    kainyyy_largecrowd  — Kainyyy/face-anti-spoof, 3611 PNGs, live/spoof
    axon_video          — AxonData CC-BY-4.0 attack-only sample videos (cut_print, 3d_mask)
    celeba_spoof_hf     — nguyenkhoa CelebA-Spoof eval shards via parquet

In-house (KVKK-consented, bundled in repo):
    in_house     — Our 43-sample Marmara University set
"""
from tests.benchmark.datasets.in_house import iter_in_house

# Other adapters declared lazily — heavy datasets need their own deps.
__all__ = ["iter_in_house"]
