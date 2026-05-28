# Paper-prep operator runbook

End-to-end workflow for reproducing every number in the paper from a fresh checkout. **Designed so a reviewer can re-run the entire experimental pipeline with one command per dataset and have the paper tables auto-rebuild from the JSON outputs.**

This runbook ships in the repo so reviewers can verify reproducibility.

## 0. Prerequisites

```bash
# Python 3.12+
python3 --version

# Repo + submodules cloned
git clone --recurse-submodules git@github.com:Rollingcat-Software/spoof-detector
cd spoof-detector

# Dependencies
pip install -r requirements.txt    # numpy, opencv-python, mediapipe, onnxruntime, uniface, pyarrow

# UniFace MiniFASNet model
# Will download on first MiniFASNetAnalyzer init to ~/.uniface/models/

# Verify the test suite passes (126 tests)
pytest -q
# expected: 126 passed
```

## 1. Acquire datasets (one-time setup)

The paper reports on **5 datasets**: 4 EULA-free public + 1 in-house.

### 1a. EULA-free public datasets (script-driven)

These come from HuggingFace mirrors and require no EULA. Disk: ~3 GB.

```bash
mkdir -p /tmp/fas_datasets

# CASIA-FASD (akahana mirror) — 4063 samples, ~157 MB
huggingface-cli download akahana/anti-spoofing-casiafasd \
    --repo-type dataset --local-dir /tmp/fas_datasets/akahana_casiafasd
cd /tmp/fas_datasets/akahana_casiafasd && tar xzf casiafasd.tar.gz -C extracted/
cd -

# CelebA-Spoof eval shard 0 (nguyenkhoa mirror) — 2611 samples, ~420 MB
huggingface-cli download nguyenkhoa/celeba-spoof-for-face-antispoofing \
    --repo-type dataset --local-dir /tmp/fas_datasets/nguyenkhoa_eval_shard \
    --include "data/eval-00000-of-00004.parquet"

# Kainyyy/face-anti-spoof (largeCrowd subset) — 3611 samples, ~1.0 GB
# NOTE: currently EXCLUDED from all paper results — the adapter has a known
# label-mapping bug (every record maps to bona-fide, so n_attack=0 and AUC=0.0).
# Do not cite Kainyyy numbers until the adapter is fixed (CODE CHANGE NEEDED).
huggingface-cli download Kainyyy/face-anti-spoof \
    --repo-type dataset --local-dir /tmp/fas_datasets/kainyyy_face_anti_spoof

# AxonData CC-BY-4.0 cut-print + 3D-mask — 30 attack videos, ~1.3 GB
huggingface-cli download AxonData/Anti_Spoofing_Cut_print_attack \
    --repo-type dataset --local-dir /tmp/fas_datasets/axon_cut_print
huggingface-cli download AxonData/3D_paper_mask_attack_dataset_for_Liveness \
    --repo-type dataset --local-dir /tmp/fas_datasets/axon_3d_mask
```

See `tests/benchmark/datasets/DATASETS_AVAILABLE.md` for full provenance + license details.

### 1b. In-house Marmara-University set (synthesised)

The in-house validation set ships *with* the repo (KVKK-consented bona-fide sources, MIT-licensed). Regenerate:

```bash
# From bio fixtures + practice-and-test fixtures
python -m tests.benchmark.synthesize_attacks \
    --src ../biometric-processor/tests/fixtures/images \
    --out /tmp/synth1 --max-per-subject 8 --min-size 100 \
    --upresize-to 256 --variants-per-attack 3 --seed 42

python -m tests.benchmark.synthesize_attacks \
    --src ../practice-and-test/DeepFacePractice1/images \
    --out /tmp/synth2 --max-per-subject 8 --min-size 100 \
    --upresize-to 256 --variants-per-attack 3 --seed 43

# Merge into data/in_house/
python tools/merge_synth.py /tmp/synth1 /tmp/synth2 data/in_house

# Build replay-only sub-protocol for §7.2 / §8.2
mkdir -p data/in_house_replay/{bonafide,attack_replay}
cp data/in_house/bonafide/*.jpg data/in_house_replay/bonafide/
cp data/in_house/attack_replay/*.jpg data/in_house_replay/attack_replay/
python tools/build_labels.py data/in_house_replay
```

(`tools/merge_synth.py` and `tools/build_labels.py` are simple 30-line helpers — see those files for the merge-csv logic.)

### 1c. EULA-locked academic datasets (manual)

To populate the **paper-grade headline rows** for OULU-NPU / SiW / CASIA-SURF / full-CelebA-Spoof:

| Dataset | EULA URL | Approx. wait |
|---|---|---|
| OULU-NPU | <https://sites.google.com/site/oulunpudatabase/> | 1–2 weeks |
| SiW | <http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html> | 1 week |
| CASIA-SURF | <http://www.cbsr.ia.ac.cn/users/jwan/database/CASIA-SURF.html> | 2 weeks |
| CelebA-Spoof full | <https://github.com/ZhangYuanhan-AI/CelebA-Spoof> | open license, ~75 GB |

Place each at `/data/<name>/` following the dataset's official directory layout. Adapter docstrings under `tests/benchmark/datasets/<name>.py` document each layout.

## 2. Run the benchmarks

All benchmarks are one CLI call each. Results land in `paper/figures/results_*.json` + `.csv`.

### 2a. The 3 pipelines × the 5 acquired datasets

```bash
# Note: minifasnet_only is the headline pipeline per §7 / §8.
# image_only and hybrid serve the §8.1 ablation rows.

for ds in casia_fasd in_house celeba_spoof_hf; do
  for pipeline in minifasnet_only image_only hybrid; do
    python -m tests.benchmark.run \
        --dataset $ds \
        --root /tmp/fas_datasets/akahana_casiafasd/extracted \
        --pipeline $pipeline \
        --protocol full
  done
done
```

(Adjust `--root` per dataset; see `tests/benchmark/run.py:_load_adapter` for the exact mapping.)

Approximate run times on Hetzner CX43 CPU:
- CASIA-FASD test (N=2,408): ~6 min per pipeline
- CelebA-Spoof eval (N=2,611): ~6.5 min per pipeline (extracts JPEG bytes from parquet)
- in_house (N=325 or N=100 replay subset): ~30s per pipeline

### 2b. Per-analyzer leave-one-out ablation (§8.2 / Table 8)

```bash
python -m tests.benchmark.ablation_leave_one_out \
    --dataset in_house --root data/in_house_replay --protocol replay_n100
```

Output: `paper/figures/ablation_loo_in_house_replay_n100.json`

Run time: ~10 minutes (single run of full pipeline + 13 re-fuse passes).

### 2c. Calibration sweep (§5.4)

```bash
# Sweep texture analyzer's weight from 0.0 to 1.0 in 0.05 steps
python -m tests.benchmark.calibration_sweep \
    --capture paper/figures/ablation_loo_in_house_replay_n100.json \
    --analyzer texture
```

Output: `paper/figures/calibration_sweep_texture.{json,png}`

### 2d. Latency benchmark (§7.6 / Table 4)

```bash
python -m tests.benchmark.latency --pipeline image_only --n-frames 50
python -m tests.benchmark.latency --pipeline hybrid     --n-frames 50
```

Outputs: `paper/figures/latency_image_only.json`, `paper/figures/latency_hybrid.json`

Run time: ~30s each.

### 2e. Bootstrap CIs (§7 footnotes)

The paper uses a fixed two-tier resample convention at `seed=42`:

- **n_resamples = 1500** for the small-N tiers (N ≤ 100): the in-house replay sub-protocol (§7.2, N=100), the in-house full set (§7.3, N=325), and the CASIA-FASD N=200 / N=500 subsamples (§8.7).
- **n_resamples = 100** for the full large public sets: CASIA-FASD (§7.1, N=2,408) and CelebA-Spoof (§7.1, N=2,611), where the CI is already tight (AUC width ≈ 0.02–0.03) at the lower count.

```bash
python -c "
from src.metrics import acer_ci, auc_ci, eer_ci
import json
# (path, n_resamples) per the two-tier convention above
for path, nr in [
    ('paper/figures/results_casia_fasd_test_full_minifasnet_only.json', 100),
    ('paper/figures/results_celeba_spoof_hf_eval_full_minifasnet_only.json', 100),
    ('paper/figures/results_in_house_replay_n100_minifasnet_only.json', 1500),
    ('paper/figures/results_in_house_full_n325_minifasnet_only.json', 1500),
]:
    d = json.load(open(path))
    s = d['per_sample']
    scores = [x['score'] for x in s]
    is_bf = [x['is_bonafide'] for x in s]
    types = [x['attack_type'] for x in s]
    a = acer_ci(scores, is_bf, types, n_resamples=nr, seed=42)
    u = auc_ci(scores, is_bf, types, n_resamples=nr, seed=42)
    print(f'{path.split(\"/\")[-1]}  (n_resamples={nr})')
    print(f'  ACER = {a.estimate*100:.2f}% [{a.low*100:.2f}, {a.high*100:.2f}]')
    print(f'  AUC  = {u.estimate:.4f} [{u.low:.4f}, {u.high:.4f}]')
"
```

Note: the AUC *point estimate* in the §7 tables is the full-resolution `metrics.auc` (computed at `roc_curve(..., n_points=200)`); the CI central estimate `u.estimate` above is at the bootstrap's internal `n_points=100` and may differ in the third–fourth decimal (e.g. CASIA 0.9452 vs 0.9454; in-house full N=325 0.4781 vs 0.4717). Both are reported in §7.

Approximate time: ~1-2 minutes per dataset at n_resamples=100 (large sets), ~3-5 minutes at n_resamples=1500 (small sets); most of the bootstrap cost is in EER computation.

## 3. Build paper tables + figures

After every relevant `tests/benchmark/run.py` call has produced a JSON in `paper/figures/`:

```bash
# Auto-build §7 + §8 tables from JSONs
python -m paper.figures.build_tables
# emits: paper/figures/table1_headline.md
#        paper/figures/table2_celeba_per_type.md
#        paper/figures/table5_ablation_tracks.md

# Auto-build per-(dataset,protocol) ROC PNGs
python -m paper.figures.plot_roc
# emits: paper/figures/roc_<dataset>_<protocol>.png  (one per group)
```

Both scripts read `paper/figures/results_*.json`, are idempotent, and only render groups that have JSON data — so partial runs render partial tables.

## 4. Compose the paper

The skeleton is in `paper/sections/00..10_*.md`. To assemble a single LaTeX manuscript:

```bash
cat paper/sections/{00,01,02,03,04,05,06,07,08,09,10}_*.md > /tmp/paper_assembled.md
# convert to LaTeX
pandoc /tmp/paper_assembled.md -o /tmp/paper_draft.tex \
    --listings --citeproc --biblio paper/refs/refs.bib
```

(`paper/refs/refs.bib` is generated from §2's References block — see `paper/refs/build_bib.py`.)

## 5. Common operator pitfalls

- **MediaPipe FaceLandmarker not initialising**: ensure `face_landmarker.task` is at `models/face_landmarker.task` (it ships with the repo).
- **MiniFASNet not loading**: `pip install uniface` and let the first run download the ONNX to `~/.uniface/models/` (~1.7 MB download).
- **Empty results JSON**: usually means the adapter walked the wrong root path. Check `tests/benchmark/datasets/<dataset>.py:iter_<dataset>` docstring for the expected layout.
- **bootstrap is slow**: each `eer_ci` call sorts O(N) candidates × `n_resamples` times. For the full large public sets (N>2000) the paper reports at `n_resamples=100` (CI width already ≈ 0.02–0.03 and stable); the small-N tiers (N≤100 plus the N=200/N=500 CASIA subsamples) use `n_resamples=1500`. See §2e for the exact per-tier mapping.
- **Cross-dataset numbers are *worse* than published intra-dataset numbers**: this is *expected* — the paper §9.2 disclaims that our zero-shot evaluation is a robustness signal, not a SOTA claim. Published OULU-NPU numbers below ACER 5% are intra-dataset retraining.

## 6. Adding a new dataset

1. Write an adapter at `tests/benchmark/datasets/<name>.py` that yields `Sample` per record. Pattern: copy from `casia_fasd.py` for image datasets, `axon_video.py` for video.
2. Wire into `tests/benchmark/run.py:_load_adapter` (one new branch).
3. Update the `--dataset` choices list in `run.py:main`.
4. Optionally: add a new protocol spec in `tests/benchmark/protocols.py` (currently each adapter takes free-form `protocol` strings — reserved for future structured protocols).
5. Run `tests.benchmark.run --dataset <name> ...` and confirm a non-empty JSON lands in `paper/figures/`.

## 7. Browser deployment

See `SPOOF_DETECTOR_BROWSER_READINESS.md` (438 lines) for the full port plan. Phase 1 + Phase 2 (MiniFASNet + 5 high-weight analyzers) ship in `web/`. To run the browser MVP locally:

```bash
cd web/
npm install
npm run typecheck && npm run build && npm test
# serve the demo
npx vite preview
# open http://localhost:4173/examples/demo.html in a browser with camera permissions
```
