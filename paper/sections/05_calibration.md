# 5. Calibration Methodology and the Anti-Correlation Finding

The hybrid pipeline's performance depends on the calibrated weights of the multi-class fuser (`src/infrastructure/fusion/multi_class_fuser.py:26-40`). Each analyzer's weight controls how much its score contributes to the final per-category probability distribution. This section formalises how those weights were obtained, presents the principal empirical finding (anti-correlation in Laplacian-texture and Gabor-moire), and discusses how the calibration generalises across datasets.

## 5.1 Calibration data

Calibration was performed on a 43-sample in-house labelled set captured at Marmara University in 2026-04 under KVKK Art. 6(1)(a) consent: 27 bona-fide captures from three subjects under varying lighting (daylight, fluorescent, single-source LED), and 16 attack captures comprising printed photos held at the camera, on-screen replays of those same photos on a 14-inch laptop display, and AR-filter still snapshots. The set is small but balanced across the most common in-the-wild attack types and is the same set our predecessors at FIVUCSAS used to validate face-recognition pipelines (so per-subject demographics are stable across studies).

Calibration data is *not* the in-house validation set used to populate §7.7's transparency block — that set is synthetic. The 43 calibration captures are real attacks acquired with the predicted physical artefacts. They drive the analyzer-weight assignment but are deliberately held out from §7's evaluation: any reviewer who runs `tests/benchmark/run.py` on §7's datasets gets numbers untouched by the calibration set.

## 5.2 Per-analyzer discrimination protocol

For each analyzer in `src/infrastructure/analyzers/`, we ran the Python implementation against every sample in the calibration set and recorded the analyzer's [0,100] score, where higher implies "more live-like" by the analyzer's own design (`src/domain/interfaces.py:18-27`). For every analyzer we computed:

- `μ_real` — mean score over the 27 bona-fide samples
- `μ_spoof` — mean score over the 16 attack samples
- **discrimination gap** = `μ_real − μ_spoof`

A *positive* gap is the signal we want — the analyzer scores reals higher than spoofs, so its score is informative for binary classification. A *negative* gap means the analyzer scores spoofs higher than reals, and a fuser that treats the analyzer as positive evidence is being misled.

The full table from the calibration run:

| Analyzer            | μ_real (out of 100) | μ_spoof (out of 100) | discrimination gap | implied calibrated weight |
|---------------------|--------------------:|---------------------:|-------------------:|--------------------------:|
| `minifasnet`        | 99.9                | 5.1                  | **+94.7**          | 5.0                       |
| `device_boundary`   | 34.2                | 15.0                 | +19.2              | 2.5                       |
| `screen_replay`     | 46.7                | 37.1                 | +9.6               | 0.5                       |
| `temporal`          | 90.0                | (single-frame proxy) | (neutral)          | 0.3                       |
| `texture`           | 72.1                | 78.4                 | **−6.3**           | 0.1                       |
| `moire`             | 39.1                | 44.1                 | **−5.0**           | 0.1                       |

(Multi-frame analyzers — `blink`, `rppg`, `micro_tremor`, `screen_flicker`, `landmark_variance`, `background_grid` — were calibrated separately on the 12-second video subset of the calibration set; their gaps are reported in `paper/figures/calibration_video_analyzers.csv` and informed weights 0.5 / 0.0 (rPPG temporarily disabled, see §9.2) / 2.5 / 3.0 / 2.0 / 1.5.)

## 5.3 The anti-correlation finding (principal contribution)

**Two of the five "obvious" textbook anti-spoof signals are anti-correlated on real-world data**: the Laplacian-variance texture analyzer and the Gabor-bank moire analyzer both score *spoof* captures *higher* than bona-fide ones. The discrimination gaps in the table above show this clearly: `texture` has gap −6.3, `moire` has gap −5.0.

This is a methodologically important finding because both signals are widely cited in the FAS literature [Maatta 2011; Boulkenafet 2016] and are typically deployed with a positive coefficient in fusion ensembles. Our measurement reverses the sign expected from those publications.

The mechanism behind the inversion is mundane and instructive. A high-quality screen replay rendered on a modern LCD has *more uniform texture* than a real face frame captured under uneven natural lighting:

- Real face frames in our calibration set were captured under varying lighting; even under daylight, the face skin shows thousands of micro-textural variations from pores, fine hair, and oil reflections. The Laplacian variance over the face bounding box on real frames is high because of these natural micro-textures.
- Replay attacks displayed on a 14-inch laptop screen pass through gamma + colour quantisation + LCD sub-pixel structure. The face skin's micro-textural detail is lost in the quantisation step, replaced by a uniform sub-pixel grid that the Laplacian filter sees as *less* high-frequency than the original. Thus `μ_spoof > μ_real` for `texture`.
- Similarly, the Gabor-bank `moire` analyzer expects to detect the periodic interference patterns visible when a camera photographs a screen — but the Gabor responses on natural skin are also high (skin has periodic vascular and pigment patterns). When the dataset under evaluation has high-resolution real-skin captures and modest-resolution screen captures, the Gabor signal flips the wrong way.

## 5.4 The remediation

Rather than remove the two anti-correlated analyzers, we re-weighted them to 0.1 in the fuser. The structural reason this is preferable to deletion:

1. **Interpretability** — operator-facing dashboards display per-analyzer scores so a human can audit *why* a session was flagged spoof. Removing the analyzers removes a useful interpretability column.
2. **Cross-dataset variability** — on §7.4's cross-dataset matrix the anti-correlation does flip sign for very specific phone-screen attacks. A small positive weight may turn out to be informative on those datasets without breaking calibration on others.
3. **Re-calibration without architectural change** — when re-running the calibration on a new operator's data, the weights table in `src/infrastructure/fusion/multi_class_fuser.py:26-40` is the only file that changes. No analyzers added, removed, or rewritten.

The 0.1 weight is empirically derived: we performed a 1-D sweep of the texture and moire weights from 0.0 to 1.0 in 0.05 steps on the 43-sample calibration set; ACER was minimised at the [0.05, 0.15] range with a flat plateau, and we picked 0.1 as the centre. The complete sweep is in `paper/figures/calibration_sweep.csv`.

## 5.5 Generalisation across datasets (single most important caveat)

Our calibration was performed on 43 in-house captures. Section 7's headline numbers come from running the same calibrated weights on four public datasets *zero-shot* — without recalibrating. The expected behaviour: zero-shot performance is bounded above by intra-dataset calibration. The CASIA-FASD AUC of 0.945 (full N=2,408) is bounded above by what an intra-CASIA-FASD calibration would produce, and the CelebA-Spoof AUC of 0.78 reflects the much larger taxonomy gap (10 spoof classes vs. our 7) plus the calibration gap.

Two practical consequences for §9:
- **`minifasnet_only` outperforms `image_only` zero-shot** because the auxiliary analyzers (texture, moire, ar_filter, device_boundary) carry calibration assumptions that don't transfer to a new attack distribution. The strong-discriminator baseline is more robust to distributional shift; the multi-analyzer fuser is more accurate after recalibration.
- **The recommended deployment recipe**: run the calibration sweep above on a new operator's labelled data (≥ 30 bona-fide and ≥ 15 attacks per attack class for a stable fit), then deploy with the per-operator fuser weights. Code for the sweep is in `tests/benchmark/calibration_sweep.py`.

## 5.6 What the calibration is *not*

The calibration **does not** train any model. MiniFASNet weights are frozen ONNX from UniFace's public release; MediaPipe is frozen from Google's release. The calibration only adjusts the linear coefficients in `MultiClassFuser`'s evidence-aggregation step. No backpropagation, no fine-tuning, no model surgery. This keeps the engine deployable on operator infrastructure that may have inference but not training capacity (e.g. edge deployments, browser MVP), and it makes the engine transparently auditable: the only learned numbers are 13 floats in a Python dict.
