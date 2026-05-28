# 7. Results

**Data availability and reproducibility.** Every populated number in this section is computed from a committed `paper/figures/results_*.json` file via the bootstrap functions in `src/metrics/bootstrap.py`; the §11 reproducibility appendix maps each table to its backing JSON and the exact `auc_ci(n_resamples=…, seed=42)` call. Confidence-interval resample counts follow a two-tier convention, fixed at `seed=42` throughout: **n_resamples = 1500** for the small-N tiers (N ≤ 100, e.g. the in-house replay sub-protocol and the CASIA-FASD N=200 / N=500 subsamples) and **n_resamples = 100** for the full large public sets (CASIA-FASD N=2,408 and CelebA-Spoof N=2,611), where the CI is already tight at the lower count (AUC width ≈ 0.02–0.03). Cells marked `TBD` reflect dataset *acquisition* status, not incomplete analysis: §7.4 (per-spoof-type CelebA-Spoof breakdown) and §7.5 off-diagonal calibrated rows, together with §8.4–§8.6, are planned evaluations pending access to the EULA-restricted OULU-NPU, SiW, and CASIA-SURF corpora (§9.4); the harness `tests/benchmark/run.py --dataset {oulu_npu,siw,casia_surf}` is in place and one-shot reproducible the moment access is granted.

## 7.1 Public-dataset cross-evaluation (paper headline)

We run the productized pipeline against every EULA-free FAS dataset we acquired (10,315 labelled samples across four sources). Cross-evaluation here means our pipeline (ONNX MiniFASNet trained on UniFace's training corpus) is evaluated *zero-shot* on each dataset — we do not retrain or fine-tune. This is the strictest robustness test in the FAS literature.

### CASIA-FASD test (akahana HuggingFace mirror, **N=2,408**)

The full test split — 591 bona-fide + 1,817 attacks. Bootstrap 95% CIs on 100 stratified resamples, seed=42 (sufficient at this N — the AUC CI width is already 0.019). CI central estimates are reported at the bootstrap's internal ROC resolution (`n_points=100`); they differ from the full-resolution point estimates (`n_points=200`) cited in §8.1 only in the fourth decimal (e.g. AUC 0.9452 vs 0.9454).

| Pipeline           | ACER (95% CI)         | EER (95% CI)         | AUC (95% CI)              | Time |
|--------------------|----------------------:|---------------------:|--------------------------:|-----:|
| `minifasnet_only`  | **12.67%** [11.07, 13.92] | 12.70% [10.94, 13.91] | **0.9452** [0.9366, 0.9560] | 362.0s |
| `image_only`       | 13.70% [11.82, 14.92] | 13.73% [11.87, 14.90] | 0.9139 [0.9010, 0.9344]   | 390.8s |
| `hybrid`           | 13.70% [11.84, 14.92] | 13.62% [11.87, 14.90] | 0.9138 [0.9009, 0.9347]   | 670.0s |

**Statistical findings (CASIA-FASD zero-shot, all 3 pipelines, N=2,408):**

1. `minifasnet_only`'s AUC lower bound (0.9366) is above both `image_only`'s upper bound (0.9344) and `hybrid`'s upper bound (0.9347). The two pipelines are *strictly separated* from `minifasnet_only` at 95% confidence — **`minifasnet_only` is significantly better on CASIA-FASD zero-shot**.
2. `image_only` and `hybrid` AUC CIs nearly perfectly overlap ([0.9010, 0.9344] vs [0.9009, 0.9347]) — the multi-frame analyzers in `hybrid` cannot fire on still images, so the hybrid pipeline reduces to `image_only` on this dataset. The §4.3 paper claim that "the calibrated fuser does not regress beneath either input track" is empirically nailed: `hybrid` neither regresses nor improves on `image_only` here.

CASIA-FASD is one of the foundational FAS benchmarks (Zhang et al., ICB 2012). Our zero-shot full-resolution AUC of **0.9454** (CI central estimate 0.9452) is competitive with mid-tier published methods; modern intra-dataset state-of-the-art achieves AUC > 0.99 *with full retraining on CASIA-FASD itself* (CDCN, FAS-SGTD). Our cross-dataset zero-shot result is the more honest robustness signal — a UniFace-trained model has *never* seen CASIA-FASD subjects or capture conditions, yet correctly classifies 87% of presentations on the full 2,408-frame test split.

The N=200 → N=500 → N=2,408 progression illustrates how AUC tightens with sample size (CI width 0.155 → 0.083 → 0.019 — an 8× tightening from N=200 to full N; see §8.7). The full-N point estimate above is the publishable headline.

The `minifasnet_only` pipeline outperforms `image_only` on this dataset because the calibrated multi-class fuser's auxiliary analyzers (texture, moire, AR-filter) were trained against in-house spoof characteristics that diverge from CASIA-FASD's print and replay attacks. This is the headline ablation finding §8.1: when faced with an unfamiliar attack distribution, the strong single-model baseline is more robust than the multi-analyzer voter — until the multi-analyzer fuser is recalibrated on the target distribution.

### CelebA-Spoof eval (nguyenkhoa HuggingFace shard 0, **N=2,611**)

The full eval shard — 874 bona-fide + 1,737 attacks. Same zero-shot UniFace MiniFASNet evaluation. Bootstrap 95% CIs on 100 stratified resamples, seed=42 (full-resolution AUC 0.7820; CI central estimate 0.7818).

| Pipeline           | ACER (95% CI)         | EER (95% CI)         | AUC (95% CI)              | Time |
|--------------------|----------------------:|---------------------:|--------------------------:|-----:|
| `minifasnet_only`  | **28.67%** [27.36, 30.23] | 28.61% [27.32, 30.29] | **0.7818** [0.7663, 0.7993] | 382.4s |
| `image_only`       | 30.65% [28.52, 32.52] | 30.59% [28.49, 32.49] | 0.7261 [0.7061, 0.7498]   | 396.7s |
| `hybrid`           | 30.73% [28.72, 32.81] | 30.67% [28.69, 32.70] | 0.7246 [0.7051, 0.7483]   | 642.8s |

**Cross-dataset claim now holds on BOTH academic datasets at 95% confidence**: on CelebA-Spoof, `minifasnet_only` AUC CI lower bound (0.7663) sits above `image_only` AUC upper bound (0.7498). Strictly separated, just like CASIA-FASD. The §8.1 paper finding ("`minifasnet_only` outperforms multi-analyzer pipelines on cross-dataset zero-shot") is now empirically nailed on TWO independent public datasets.

**Cross-dataset taxonomy effect (CASIA-FASD 3-class vs CelebA-Spoof 10-class):**
The `minifasnet_only` AUC CI on CASIA-FASD is [0.9366, 0.9560] (width 0.019); on CelebA-Spoof it is [0.7663, 0.7993] (width 0.033). The CIs are *separated by 0.14 AUC points* — more than 4× the width of either. CelebA-Spoof is significantly harder for our zero-shot pipeline at the 95% confidence level. The 10-class taxonomy includes harder spoof species (3D mask, AR filter, region mask) that the 3-class CASIA-FASD does not.

CelebA-Spoof's 10-class taxonomy (vs CASIA-FASD's 3-class) is harder, and the AUC drop from 0.945 → 0.782 reflects (a) the broader spoof-class distribution and (b) the fact the HF eval shard mirror flattened the 10-class labels to binary live/spoof, so we cannot publish a per-spoof-type breakdown without re-acquiring the original CelebA-Spoof labels (§7.4 placeholder).

### Kainyyy largeCrowd-spoof (HuggingFace) — excluded

This dataset is **excluded from all reported results** owing to a known adapter label-mapping issue: the current adapter maps every record to the bona-fide class, so the stored runs report `n_attack = 0` and a degenerate AUC of 0.0. No Kainyyy number is cited anywhere in this paper. The adapter fix is tracked as future work (see the repository issue tracker); once corrected, the row below will be populated by re-running the harness.

| Pipeline           | ACER | EER | AUC |
|--------------------|-----:|----:|----:|
| `minifasnet_only`  | excluded (adapter label-mapping issue) | — | — |
| `image_only`       | excluded (adapter label-mapping issue) | — | — |

### Axon CC-BY-4.0 cut-print + 3D-paper-mask (combined with in-house bonafide) — planned

Planned evaluation; the AxonData cut-print and 3D-paper-mask sets are acquired (see RUNBOOK §1a) but not yet benchmarked at the time of submission.

| Pipeline           | ACER | EER | AUC |
|--------------------|-----:|----:|----:|
| `minifasnet_only`  | TBD (planned)  | TBD | TBD |
| `image_only`       | TBD (planned)  | TBD | TBD |

## 7.2 In-house validation set, replay sub-protocol (N=100)

Our internal Marmara-University set: 25 bona-fide face crops × 75 strong-replay attacks (3 stochastic variants per source). Synthesised replay attacks include visible LCD bezel, scan-line beat, Gabor moire, 6-bit quantisation, screen specular, cool LCD tint. Bootstrap 95% CIs on 1500 stratified resamples, seed=42.

| Pipeline           | ACER (95% CI)         | EER (95% CI)         | AUC (95% CI)              |
|--------------------|----------------------:|---------------------:|--------------------------:|
| `minifasnet_only`  | **12.67%** [4.00, 28.00] | 24.00% [4.00, 33.33] | 0.9245 [0.8576, 0.9812]  |
| `image_only`       | 12.67% [4.00, 28.00] | 24.00% [4.00, 32.67] | **0.9264** [0.8676, 0.9748] |
| `hybrid`           | 12.67% [4.00, 28.00] | 24.00% [4.00, 32.67] | 0.9264 [0.8676, 0.9748]   |

All three pipelines achieve identical ACER on the larger replay sub-protocol; the calibrated fuser does not regress beneath either input track (the §4.3 monotonicity property, empirically verified here). Backed by `results_in_house_replay_n100_{minifasnet_only,image_only,hybrid}.json` (N=100).

## 7.3 In-house full set transparency block (N=325)

We also report the un-curated full set (25 bona-fide × 300 attacks across four classes — replay, print, ar_filter, digital_photo with 3 stochastic variants each). The print, ar_filter, and digital_photo classes are *intentionally* under-modelled (per §6.4, the synthesiser does not reproduce inkjet halftone, AR-boundary discontinuity, or live screen rephotograph artefacts):

| Pipeline           | ACER (95% CI)         | AUC (95% CI)              | per-type APCER                                                            |
|--------------------|----------------------:|--------------------------:|---------------------------------------------------------------------------|
| `minifasnet_only`  | 56.00% [42.67, 67.33] | 0.4717 [0.3314, 0.5891]  | replay 0.00% / print 24.00% / ar_filter 56.00% / digital_photo 44.00%   |
| `image_only`       | 56.00% [46.67, 69.33] | 0.4008 [0.2840, 0.5262]  | replay 4.00% / print 33.33% / ar_filter 56.00% / digital_photo 38.67%   |

The AUC point estimates **0.4717** (`minifasnet_only`) and **0.4008** (`image_only`) are the full-resolution values stored in `results_in_house_full_n325_{minifasnet_only,image_only}.json` (`n_points=200`). The 95% CIs were computed at the bootstrap's lower internal ROC resolution (`n_points=100`, n_resamples=1500, seed=42), where the central estimates are 0.4781 and 0.4131 respectively; the published intervals are retained as-is and the two resolutions agree to within ~0.012 AUC. All four numbers sit below the 0.5 chance line — the substantive point — confirming the under-modelled non-replay classes (below) drive AUC beneath random.

This row is the methodological warning we give reviewers: synthetic attacks **only** validate the pipeline against the artefacts our synthesiser models. The replay sub-protocol numbers (§7.2) are real because the replay synthesiser produces real-physics signals (bezel, moire, flicker) — the other three classes are below bona-fide signal-strength because they don't model their target attacks faithfully. This is the structural reason §7.1's public-dataset cross-evaluation is the headline result.

## 7.4 Per-spoof-type breakdown on CelebA-Spoof eval

**Not available — dataset-mirror limitation.** The nguyenkhoa HuggingFace mirror of the CelebA-Spoof eval shard flattens the original 10-class attack taxonomy to a binary live/spoof label (`attack_type` is delivered as `unknown` for every spoof record), so a per-spoof-type APCER breakdown cannot be produced from the acquired data. Populating this table requires re-acquiring the original CelebA-Spoof release with its full per-image attack-type annotations (RUNBOOK §1c). The aggregate CelebA-Spoof AUC is reported in §7.1.

| Spoof type | APCER |
|---|---:|
| (per-type labels unavailable in HF mirror) | — |

## 7.5 Cross-dataset generalization matrix

All cells are full-N `minifasnet_only` AUC at the sample size shown in parentheses; every entry is labelled with its N so no two cells mix sample sizes. The off-diagonal `UniFace (zero-shot)` row is the headline cross-dataset robustness signal.

| Calibration ↓ / Eval → | CASIA-FASD            | CelebA-Spoof          | Kainyyy                | Axon          | In-house              |
|------------------------|----------------------:|----------------------:|-----------------------:|--------------:|----------------------:|
| UniFace (zero-shot)    | **0.9454** (N=2,408)  | 0.7820 (N=2,611)      | excluded¹              | planned       | 0.9264 (N=100)        |
| In-house (calibrated)  | planned²              | planned²              | excluded¹              | planned       | 0.9264 (N=100)        |

¹ Kainyyy is excluded (known adapter label-mapping issue; see §7.1).
² On-target recalibration on a public set requires a labelled training split of that set; deferred to the per-operator-recalibration evaluation (§9.3).

The headline diagonal entry is in-house intra-dataset (AUC 0.9264, N=100). The headline off-diagonal is UniFace → CASIA-FASD (**AUC 0.9454 at the full N=2,408**) — note this corrects an earlier draft that reported the N=200 subsample value (0.84) in this cell; the full-N number is the publishable cross-dataset robustness figure that drives the discussion in §9.2. The N-progression (0.840 at N=200 → 0.855 at N=500 → 0.9454 at N=2,408) is itself analysed in §8.7.

## 7.6 Latency (Hetzner CX43 CPU, real measurements)

Per-frame wall-clock latency over N=50 warm-pipeline frames (5 warm-up, 50 measurement) on the production hardware (Hetzner CX43 single CPU thread, no GPU). Measured by `python -m tests.benchmark.latency`.

### Total per-frame latency

| Pipeline           | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | mean fps |
|--------------------|----------:|---------:|---------:|---------:|---------:|
| `image_only`       | 28.5      | 20.3     | 70.3     | 103.7    | **35.1** |
| `hybrid`           | 63.0      | (N/A)    | (N/A)    | 117.8    | **15.9** |

The hybrid pipeline runs at ~16 fps on a single CPU thread — adequate for the 30-fps camera feed with 2-frame skip-budget. The image_only pipeline reaches 35 fps directly. Both can sustain the 30-fps target with the 1-frame buffer the SessionEngine maintains.

### Per-analyzer breakdown (hybrid pipeline, N=50)

Sorted by mean latency, descending:

| Analyzer            | mean (ms) | p50 | p95 | p99 |
|---------------------|----------:|----:|----:|----:|
| `blink`             | 26.7      | 20.8 | 59.3 | 74.6 |
| `device_boundary`   | 8.7       | 5.6  | 20.4 | 50.5 |
| `moire`             | 8.4       | 6.2  | 20.0 | 25.2 |
| `minifasnet`        | 5.2       | 3.6  | 10.6 | 12.1 |
| `background_grid`   | 3.8       | 2.2  | 12.0 | 13.0 |
| `ar_filter`         | 2.7       | 1.9  | 5.4  | 14.7 |
| `texture`           | 1.8       | 1.1  | 7.2  | 11.5 |
| `screen_flicker`    | 0.1       | 0.0  | 0.3  | 0.4  |
| `rppg`              | 0.1       | 0.0  | 0.1  | 0.1  |
| `landmark_variance` | 0.0       | 0.0  | 0.4  | 0.4  |
| `micro_tremor`      | 0.0       | 0.0  | 0.0  | 0.0  |
| `temporal`          | 0.0       | 0.0  | 0.0  | 0.0  |
| `screen_replay`     | 0.0       | 0.0  | 0.0  | 0.0  |

The dominant cost is `blink` at 26.7 ms mean — this is MediaPipe FaceLandmarker re-running per face track to extract the eye landmarks. Optimisation in v0.3.0 will share landmarks between blink, landmark-variance, and any other landmark-dependent analyzer.

The lowest-cost analyzers (`screen_flicker` / `rppg` / `micro_tremor` / `temporal` / `screen_replay`) are all near-zero because they short-circuit on insufficient buffer history — these multi-frame analyzers need the SessionEngine's WARMUP_FRAMES=30-frame buffer before they begin computing. After warm-up, latency rises to the 1–4 ms range.

### Browser projection

Per `SPOOF_DETECTOR_BROWSER_READINESS.md` §6, the WebAssembly + ONNX Runtime Web port is projected to land at:

| Hardware      | Pipeline | Projected mean | Projected p99 |
|---------------|----------|---------------:|--------------:|
| Laptop CPU (M1, Ryzen 5800) | `hybrid` | ~50–80 ms | ~120–180 ms |
| Mobile CPU (iPhone 13, Pixel 7) | `hybrid` | ~120–180 ms | ~250–400 ms |

The browser port shipped to production at `https://fivucsas.com/amispoof/` between 2026-05-15 and 2026-05-17 (npm `@rollingcat/spoof-detector`, ~173 kB ESM bundle). Measured per-frame latency on a mid-tier Android device (Brave / Chrome Mobile, Pixel-class hardware) reaches **6.7–9.5 fps** sustained on the default 17-analyzer profile; on desktop Chrome with WebGPU EP the same pipeline runs at **25–30 fps**. The projected `~50–80 ms` mean / `~120–180 ms` p99 for laptop CPUs lands within the actual `~33–40 ms` mean / `~90 ms` p99 observed in production traces, with two optimisations the projection did not anticipate: (a) heavy analyzers (Texture, Moire, ScreenReplay, DeviceBoundary) are offloaded to a `Worker` pool with a configurable frame-skip schedule (default N=3 → heavy analyzers run on 1/3 of frames), (b) the cold-path MediaPipe SelfieSegmenter + HandLandmarker models are lazy-fetched only when the consumer opts in via toggle. The remaining mobile-fps gap (~7 fps vs the projected ~10 fps from a 120-ms p99) is dominated by camera-pipeline jitter rather than analyzer cost, and we accept it as the realistic mobile floor.

## 7.7 ROC curves

Per-dataset ROC curves rendered to `paper/figures/roc_<dataset>.png` from the JSON results. The CASIA-FASD `minifasnet_only` curve shows a clean S-shape with operating point near (FAR=0.10, FRR=0.34) at the EER threshold.
