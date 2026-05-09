"""Dataset adapters for academic FAS benchmarks.

Each module here knows how to walk one dataset's directory layout and
yield Sample(s). Datasets are NOT bundled — see each module's docstring
for the access procedure.

Implemented:
    oulu_npu     — OULU-NPU (Oulu, Finland)
    siw          — Spoof in the Wild (Michigan State)
    casia_surf   — CASIA-SURF (Chinese Academy of Sciences)
    celeba_spoof — CelebA-Spoof (CUHK)
    in_house     — Our 43-sample Marmara University set
"""
from tests.benchmark.datasets.in_house import iter_in_house

# Other adapters declared lazily — heavy datasets need their own deps.
__all__ = ["iter_in_house"]
