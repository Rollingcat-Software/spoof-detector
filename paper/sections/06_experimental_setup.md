# 6. Experimental Setup

## 6.1 Datasets

| Dataset | Subjects | Sessions | PA species | Modality | License |
|---|---:|---:|---|---|---|
| OULU-NPU | 55 | 3 | print (×2 paper), replay (×2 monitor) | RGB video | EULA (CMVS Oulu) |
| SiW | 165 | 1 | print, replay | RGB video | EULA (MSU) |
| CASIA-SURF | 1,000 | varies | mask cut-out | RGB+Depth+IR | EULA (CASIA) |
| CelebA-Spoof | 10,177 | varies | 10 PA species | RGB image | CelebA license |
| In-house (Marmara) | 4 | 1 | print, replay, AR filter | RGB image+video | KVKK Art. 6(1)(a), CC-BY |

The four academic datasets are NOT bundled in our release; each adapter (`tests/benchmark/datasets/<name>.py`) walks the dataset's official directory layout and yields canonical `Sample` records. The Marmara in-house set ships with the repository under explicit study-participant consent.

## 6.2 Protocols

Following the FAS literature, we report four protocols on OULU-NPU:

- **P1**: Train and test on different sessions (cross-session).
- **P2**: Train on a subset of PA species, test on the held-out species (cross-PAI).
- **P3**: Train on five phone models, test on the held-out one (cross-camera).
- **P4**: Cross-everything (sessions × PAI × phones).

For SiW we use the official 90/75 subject split. For CASIA-SURF we use the published Train/Val/Test partition and report on the RGB modality only (depth+IR ignored for parity with our pipeline). For CelebA-Spoof we use the `intra_test` protocol, with 10-class breakdown.

## 6.3 Pipelines compared

Three configurations evaluated on every protocol:

1. **`image_only`** — Ahmet's track. MiniFASNet + device-boundary + texture + moire + AR-filter. Per-frame, mean aggregation.
2. **`video_only`** — Aysenur's track. Blink + rPPG + screen-replay + micro-tremor + screen-flicker + landmark-variance + temporal. Peak-sensitive aggregation. Single-image samples score 0.5 (uncertain).
3. **`hybrid`** — Both tracks with calibrated weights and peak-sensitive aggregation. **The published method.**

All three share the same face detector, tracker, and fuser code path; only the analyzer set differs. This isolates the contribution of each track.

## 6.4 Metrics

We report the ISO/IEC 30107-3 canonical set:

- **APCER** (Attack Presentation Classification Error Rate) per attack type, then max across types.
- **BPCER** (Bona-fide Presentation Classification Error Rate).
- **ACER** (Average Classification Error Rate) = (APCER_max + BPCER) / 2.
- **EER** (Equal Error Rate) — threshold-free summary.
- **AUC** (Area Under ROC) — for cross-method comparability.

For OULU-NPU we additionally report the published-style per-protocol APCER/BPCER/ACER triplets so reviewers can directly compare against the leaderboard. For CelebA-Spoof we report per-spoof-type APCER (10 categories) plus the macro-averaged ACER.

All metrics are computed by `src/metrics/` (ISO 30107-3 reference implementation, 12 unit tests passing). The benchmark harness logs per-sample scores so the metrics can be re-computed at any threshold post-hoc.

## 6.5 Hardware and reproducibility

All evaluation runs on a single CPU thread (no GPU) on:

- Ubuntu 24.04 LTS, Linux 6.8
- Python 3.12.3
- Hetzner CX43 (16 GB RAM, AMD EPYC) — same hardware as production deployment

ONNX models are loaded via `onnxruntime` 1.18 in CPU mode. Random seeds are fixed at 42 across NumPy, PyTorch (for AR-filter training), and the dataset adapters' frame-sampling. Benchmark commands are versioned in `tests/benchmark/run.py` so any reviewer can reproduce the headline numbers with one command per dataset:

```
python -m tests.benchmark.run --dataset oulu_npu --root /data/oulu --protocol P1
python -m tests.benchmark.run --dataset siw       --root /data/siw
python -m tests.benchmark.run --dataset casia_surf --root /data/surf
python -m tests.benchmark.run --dataset celeba_spoof --root /data/celeba
python -m tests.benchmark.run --dataset in_house
```

Raw results land in `paper/figures/results_<dataset>_<protocol>_<pipeline>.{json,csv}` and are merged into the LaTeX tables of §7 by `paper/figures/build_tables.py`.
