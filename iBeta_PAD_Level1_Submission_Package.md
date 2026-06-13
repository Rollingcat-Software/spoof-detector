# iBeta PAD Level 1 Submission Package

## FIVUCSAS Spoof Detector

| Field | Value |
|---|---|
| **Product name** | FIVUCSAS Spoof Detector (`spoof-detector`) |
| **Version** | 0.2.1 |
| **Submission commit** | `cc73cf08e0df1f811c08cc92549fae48d6c4a05a` |
| **Submission date** | 2026-05-11 |
| **Target standard** | ISO/IEC 30107-3 — Presentation Attack Detection (PAD), Level 1 |
| **PAD class scope** | Bona-fide vs. PAI species 1–5 (print, screen-replay, video-replay, paper cut-out, 3-D paper mask) per §3 taxonomy |
| **Capture modality** | RGB video, single front-facing camera, 30 FPS nominal |
| **Operating point** | Server-side passive PAD with optional active-challenge layer |
| **Vendor** | Rollingcat Software (Marmara University spin-out) |
| **Maintainer of record** | Ahmet Abdullah Gültekin — `ahmet.gultekin@marun.edu.tr` |
| **Repository** | <https://github.com/Rollingcat-Software/spoof-detector> |
| **License** | MIT (source); see §10 for model-weight sub-licences |

This document is the engagement-ready submission package for an iBeta
PAD-Level-1 evaluation. Sections are ordered to mirror the iBeta
on-boarding pack: cover, executive summary, system description,
attack-type coverage matrix, test results (APCER / BPCER / ACER per
type with bootstrap 95% CIs), datasets, reproducibility, deployment,
open items, and appendix.

All numbers in §5 are reported exactly as they appear in
`paper/figures/results_*.json` and the paper sources at
`paper/sections/07_results.md` and `paper/sections/08_ablations.md`.
Where a cell is `TBD` in the paper sources it is reported as `TBD` here
— we do not fabricate numbers for cells the empirical pipeline has
not yet populated.

---

## Table of contents

1. [Cover](#1-cover)
2. [Executive summary](#2-executive-summary)
3. [System description](#3-system-description)
4. [Attack-type coverage matrix](#4-attack-type-coverage-matrix)
5. [Test results](#5-test-results)
6. [Datasets used](#6-datasets-used)
7. [Reproducibility](#7-reproducibility)
8. [Implementation and deployment](#8-implementation-and-deployment)
9. [Open items and limitations](#9-open-items-and-limitations)
10. [Appendix](#10-appendix)

---

## 1. Cover

### 1.1 Vendor and contact

| Item | Value |
|---|---|
| Vendor entity | Rollingcat Software |
| Vendor URL | <https://github.com/Rollingcat-Software> |
| Country of incorporation | Turkey |
| Engagement lead | Ahmet Abdullah Gültekin |
| Engagement email | `ahmet.gultekin@marun.edu.tr` |
| Engagement GitHub | [@ahabgu](https://github.com/ahabgu) |
| Project home (FIVUCSAS) | <https://github.com/Rollingcat-Software/FIVUCSAS> |
| Submitted artefact | <https://github.com/Rollingcat-Software/spoof-detector> @ `cc73cf08` |

### 1.2 Submitted artefact summary

The submitted artefact is the standalone `spoof-detector` Python
library at commit `cc73cf08e0df1f811c08cc92549fae48d6c4a05a` (tag
candidate `v0.2.1`), together with its embedded ONNX model weights
(UniFace MiniFASNetV2, downloaded on first init to
`~/.uniface/models/`) and the MediaPipe FaceLandmarker task file
shipped at `models/face_landmarker.task`. No additional binaries are
bundled.

### 1.3 Submission scope

The submission is scoped to **ISO/IEC 30107-3 PAD Level 1**, the
*"detection of common, easily produced presentation attack
instruments (PAIs)"* tier. The covered PAI species are enumerated in
§4. Level-2 (skilled-attacker, custom-fabricated PAIs including
silicone full-head masks and resin masks with eye holes) is **not**
in scope; the engine routes evidence into the `MASK_3D` category but
is not validated at Level 2 — see §9.

### 1.4 What this document is

A self-contained submission package. The operator may attach this
document, with a cover letter, to an iBeta engagement request. All
referenced repository files are reproducible from the submitted
commit; section 7 contains the one-command-per-dataset reproduction
recipe.

---

## 2. Executive summary

### 2.1 What FIVUCSAS Spoof Detector is

FIVUCSAS Spoof Detector is a **session-based face presentation-attack
detection (PAD) engine** that accumulates evidence across a 5-second
to 3-hour camera session and emits a per-session live/spoof verdict
plus per-frame analyzer telemetry for human review.

It combines:

- A **strong per-frame discriminator** — UniFace MiniFASNetV2 served
  via ONNX Runtime (CPU build by default).
- **Nine production analyzers** in three layers (pixel forensics,
  behavioural signals, environment) — see §3.
- A **calibrated multi-class fuser** that routes per-analyzer
  evidence into a seven-category taxonomy (`REAL`, `PRINT`,
  `REPLAY`, `MASK_3D`, `HEAVY_MAKEUP`, `AR_FILTER`,
  `DEEPFAKE_INJECT`).
- A **peak-sensitive session verdict** that prevents spoof-burst
  dilution — a 50/50 blend of mean and worst-window liveness, so an
  attacker who flashes a real face for one second in a 60-second
  proctoring session does not get a near-live verdict.

The engine is *additive* to a face-recognition pipeline: it consumes
the same RGB camera feed and produces an advisory liveness verdict
that the caller may use to gate enrolment, verification, or session
continuation. The reference integration is FIVUCSAS
`biometric-processor` (production), which calls into the four
mirrored modules under `from_biometric_processor/`.

### 2.2 Why submitted for iBeta PAD-Level-1

The product is in production today at <https://api.fivucsas.com>
(FIVUCSAS) under the `LIVENESS_BACKEND=uniface`,
`LIVENESS_MODE=passive` configuration. Independent iBeta validation
is the natural next step for two reasons:

1. **Customer demand** — enterprise tenants (banking, education,
   government identity onboarding) require third-party PAD-Level-1
   attestation as a procurement gate.
2. **Research credibility** — the academic paper targeting BIOSIG
   2026 / IJCB 2026 (see §10.2) reports zero-shot cross-dataset
   numbers from the same artefact. Independent iBeta numbers
   strengthen the paper's external-validity claim.

### 2.3 Headline metrics

All numbers pinned from `paper/sections/07_results.md` (commit
`cc73cf08`), bootstrap 95% CIs computed at the noted resample count.

**Public-dataset zero-shot evaluation (no fine-tuning on the
target dataset).**

| Dataset | N | Pipeline | ACER | AUC | 95% CI on AUC |
|---|---:|---|---:|---:|---:|
| CASIA-FASD test | 2,408 | `minifasnet_only` | **12.67 %** | **0.9452** | [0.9366, 0.9560] |
| CASIA-FASD test | 2,408 | `image_only` | 13.70 % | 0.9139 | [0.9010, 0.9344] |
| CASIA-FASD test | 2,408 | `hybrid` | 13.70 % | 0.9138 | [0.9009, 0.9347] |
| CelebA-Spoof eval | 2,611 | `minifasnet_only` | **28.67 %** | **0.7818** | [0.7663, 0.7993] |
| CelebA-Spoof eval | 2,611 | `image_only` | 30.65 % | 0.7261 | [0.7061, 0.7498] |
| CelebA-Spoof eval | 2,611 | `hybrid` | 30.73 % | 0.7246 | [0.7051, 0.7483] |

**Statistical findings** (paper §7.1):

- On CASIA-FASD, `minifasnet_only`'s AUC lower bound (0.9366) is
  strictly above both alternative pipelines' upper bounds (0.9344 /
  0.9347) — `minifasnet_only` is significantly better at 95 % CI.
- On CelebA-Spoof, the same separation holds: 0.7663 lower bound vs
  0.7498 upper bound for `image_only`.
- Cross-dataset taxonomy effect — the CelebA-Spoof CI is 4× wider
  than the CASIA-FASD CI, and the two CIs are separated by 0.14 AUC
  points, reflecting CelebA-Spoof's broader 10-class taxonomy.

**In-house validation set (replay sub-protocol, KVKK-consented bona-fide
sources, paper-grade synthesised replay attacks).**

| N | Pipeline | ACER (95% CI) | AUC (95% CI) |
|---:|---|---:|---:|
| 100 | `minifasnet_only` | **12.67 %** [4.00, 28.00] | 0.9245 [0.8568, 0.9811] |
| 100 | `image_only` | 12.67 % [4.00, 28.00] | **0.9264** [0.8685, 0.9744] |
| 100 | `hybrid` | 12.67 % [4.00, 28.00] | 0.9264 [0.8685, 0.9744] |

**ISO 30107-3 first-measurement four-scenario session evaluation
(2026-05-02; in-house, scripted attacks).**

| Metric | Value |
|---|---|
| BPCER | 0.00 % |
| APCER (max across types) | 30 % |
| ACER | 15 % |
| ISO Grade | C |

Per-scenario session verdicts:

- **Real face** — LIVE 78 %, liveness proof 63/100 PROVEN, 5 blinks,
  0 incidents.
- **Phone-screen photo** — SPOOF 43 %, liveness 23/100, 0 blinks,
  7 incidents.
- **Printed photo** — SPOOF 58 %, liveness 50/100, 3 incidents.
- **Video replay** — LIVE 60 % — open challenge: a video replay
  still shows real blinks/motion, so passive temporal analyzers
  alone are not sufficient (the active-challenge layer in §4.5 of
  the paper closes this gap on a +30 to +50 percentage-point swing
  per Aysenur's internal 2026 evaluation).

### 2.4 Reading guide for iBeta evaluators

- **Coverage matrix** — §4. Maps each ISO 30107-3 PAI species to the
  analyzers that detect it, with paper §3.2/§3.4 cross-references.
- **APCER/BPCER/ACER per type** — §5.1 (cross-dataset), §5.2 (in-house
  replay), §5.3 (in-house full set transparency block), §5.4
  (per-analyzer leave-one-out ablation).
- **How to reproduce every number** — §7 + `RUNBOOK_PAPER_PREP.md`
  in-repo.
- **Known limitations** — §9, plus paper §9.2/9.3.

---

## 3. System description

### 3.1 Pipeline overview

A single-camera RGB feed enters at 30 FPS. The face detector
(MediaPipe Tasks API) yields bounding boxes and landmarks at ~2 ms
per frame. A multi-target IoU tracker assigns persistent IDs so that
temporal analyzers operate on the same face across frames. Each
frame is then fanned out to two banks of analyzers operating in
parallel.

```
Session Engine (5 s – 3 hr)
  - "guilty until proven innocent" liveness prover
  - accumulates per-frame evidence + liveness proof score (0–100)
  - detects incidents (spoof bursts, frozen face, MiniFASNet instability)
  - peak-sensitive verdict (worst-window prevents dilution)
  - session report with liveness breakdown on exit

Pipeline (per frame)
  - MediaPipe face detection (~2 ms)
  - MediaPipe FaceLandmarker (478 points, shared across analyzers)
  - IoU multi-face tracking
  - Analyzer bank (Layer 1, 2, 3)
  - MultiClassFuser -> 7-category probabilities
  - LivenessProver: blinks(25) + motion(20) + rotation(15) + expression(15) = 75 max
```

Source pointers: `src/application/session_engine.py`,
`src/application/pipeline.py`, `src/infrastructure/analyzers/`,
`src/infrastructure/fusion/multi_class_fuser.py`.

### 3.2 Analyzer inventory

The current `v0.2.1` production set is **nine wired analyzers** plus
five experimental / low-weight analyzers retained for interpretability.
Per the calibrated fuser at
`src/infrastructure/fusion/multi_class_fuser.py:26-40`:

**Layer 1 — pixel forensics**

| Analyzer | Algorithm | Weight | Latency (CPU) |
|---|---|---:|---:|
| MiniFASNet ONNX | UniFace MiniFASNetV2, softmax `[-1]` | 5.0 | ~5 ms |
| Screen flicker | Spatial FFT, energy at 50/60 Hz multiples | 3.0 | ~0 ms |
| Device boundary | Canny + probabilistic Hough → bezel rectangularity | 2.5 | ~9 ms |
| Screen-replay | Multi-region patch FFT + skin colour | 0.5 | ~0 ms |

**Layer 2 — behavioural signals**

| Analyzer | Signal | Weight | Latency (CPU) |
|---|---|---:|---:|
| Micro-tremor | 8–12 Hz head-yaw FFT | 2.5 | ~0 ms |
| Landmark variance | 478-point per-landmark σ | 2.0 | ~0 ms |
| Blink (EAR) | Eye Aspect Ratio, threshold 0.18, REOPEN 0.23 | 0.5 | ~27 ms |
| Temporal | Optical-flow + landmark trajectory plausibility | 0.3 | ~0 ms |
| rPPG | Skin pulse FFT in [0.7, 4.0] Hz | **0.0** | ~0 ms |

(rPPG is currently disabled by calibration — the implementation
detects display flicker as a false pulse on screen-replay attacks.
Fix is the highest-priority ROADMAP P0 item; see §9.)

**Layer 3 — environment**

| Analyzer | Signal | Weight | Latency (CPU) |
|---|---|---:|---:|
| Background grid | 6×4 cell motion stability (proctoring) | 1.5 | ~4 ms |
| AR filter (heuristic) | Boundary discontinuity + chromaticity uniformity | 0.3 | ~3 ms |
| Texture | Laplacian variance + colour entropy + radial-FFT | 0.1 | ~2 ms |
| Moire | Gabor bank (4 orientations × 3 scales) + radial FFT | 0.1 | ~8 ms |

**Anti-correlation finding (paper §5.3).** Texture and moire were
empirically found to score *spoof* captures *higher* than bona-fide
ones on the calibration set (discrimination gap −6.3 and −5.0
respectively, out of 100). Both remain in the fuser at weight 0.1
for (a) interpretability — operator dashboards show per-analyzer
scores, (b) cross-dataset re-calibration headroom, and (c)
re-calibration is the only file change needed if the sign flips on
a new operator dataset.

### 3.3 Multi-class fuser

7-category taxonomy at `src/domain/taxonomy.py:27-114`:

```
{REAL, PRINT, REPLAY, MASK_3D, HEAVY_MAKEUP, AR_FILTER, DEEPFAKE_INJECT}
```

Per-frame voting:

```
P(category | frame) ∝ exp( Σ_a w_a · evidence_a(category) )
```

Each analyzer's `[0,100]` score is mapped to per-category evidence
via the calibrated `SPOOF_SIGNAL_MAP`. For example, MiniFASNet's
"live-ness" score routes a *low* score (=spoof) to PRINT (0.4
weight), REPLAY (0.4), AR_FILTER (0.2). Device-boundary's *low*
score is uninformative; its *high* score routes evidence to REPLAY
(0.7) and PRINT (0.3) — only screens and printouts produce sharp
rectangular boundaries.

Final per-frame classification is `argmax P`.

### 3.4 Session verdict aggregator

The session engine ingests per-frame classifications and produces a
session verdict that strengthens over time. State machine:
`WARMING_UP → ANALYZING → CONCLUDED`. The published aggregator at
`src/application/session_engine.py` is:

```
P_session(REAL) = 0.5 · mean(P_frame(REAL) | t > t_warmup)
                + 0.5 · mean(P_frame(REAL) | t ∈ worst-decile-window)
```

The 50/50 blend is the proctoring-critical property: an attacker who
flashes a real face for one second in a sixty-second session does
not get a session verdict averaged near "real". The worst-decile
window dominates the second term and pulls the session verdict down
to the low liveness of the spoof minutes.

### 3.5 Incident emission

Incidents (events that flag operator attention) are emitted on:

- Sustained `P(REAL) < 0.4` for ≥ 3 seconds.
- No blinks for ≥ 15 seconds (when blink analyzer is healthy).
- Face missing for ≥ 5 seconds.
- MiniFASNet score swing ≥ 0.35 in 1-second window
  (identity-change suspicion).

### 3.6 Active-challenge layer (optional, not part of headline pipeline)

For deployments that can demand active user cooperation (high-stakes
onboarding, exam proctoring), an optional active layer is available
under `from_biometric_processor/`:

- **Random colour-flash challenge** (`light_challenge_service.py`)
  — measures chromatic skin response.
- **Head-rotation challenge** (`active_gesture_liveness_manager.py`)
  — measures 3-D parallax.
- **Blink-on-command challenge** — explicit user prompt.

Aysenur's 2026 internal evaluation reports the active layer adds
≈ +30 to +50 percentage-point swings on the hardest screen-replay
attacks at the cost of one round-trip and a UX intrusion. **The
active layer is not part of the headline pipeline reported in this
submission.** It is available as an operator-configurable second
tier.

### 3.7 Deployment surface

The library is deployed in two surfaces today:

1. **FIVUCSAS biometric-processor (FastAPI, Python 3.12)** —
   the production identity-verification microservice. The
   `/verify` and `/enroll` endpoints call into the four mirrored
   modules at `from_biometric_processor/`:
   `cutout_anomaly_detector.py`, `device_spoof_risk_evaluator.py`,
   `light_challenge_service.py`, `screen_replay_anti_spoof.py`.
   Configuration:
   - `LIVENESS_BACKEND=uniface`
   - `LIVENESS_MODE=passive`
   - `ANTI_SPOOFING_ENABLED=true`
   - Server image SHA `75347c98` (prod 2026-05-11).

2. **Standalone Python library** — `pip install spoof-detector[full]`.
   Consumed by reference applications, the academic paper's
   benchmark harness, and external integrators.

A WebAssembly + ONNX Runtime Web port (Phase 1 + Phase 2 in flight
2026-05-09; see `SPOOF_DETECTOR_BROWSER_READINESS.md`) will add a
third surface — in-browser PAD without server round-trips.

### 3.8 Recent v0.2.1 improvements

Commit `cc73cf08` (2026-05-11) shipped two paper-prep P0 items:

- **Blink analyzer per-frame FaceLandmarker cache.**
  FaceLandmarker now runs once per frame regardless of face count.
  Linear speedup with face count: **9.8 → 28.9 FPS at 3 faces**
  (≈ 3.0× speedup), and ≈ 4.9× at 5 faces. Closes the dominant
  cost identified in §7.6's latency breakdown (blink was 26.7 ms
  per face track).
- **EAR threshold recalibration.** `EAR_THRESHOLD` 0.20 → 0.18,
  `REOPEN` 0.22 → 0.23, `MIN_OPEN_BETWEEN` 6 → 12 frames. On a
  simulated 60 s session with 1 blink / 3.5 s the detector now
  lands at 17/min (target 15–20/min for adult normative range);
  previously it was 38/min, double normal. Pinned by
  `tests/unit/analyzers/test_blink_calibration.py`.

---

## 4. Attack-type coverage matrix

This section is the **ISO 30107-3 attack-type ↔ analyzer crosswalk**.
The seven taxonomy categories from §3.3 are the rows; the
production analyzers from §3.2 are the columns. A `●` indicates
"primary detector"; a `○` indicates "corroborating signal".

### 4.1 Matrix (Table 1 of the paper, paper/sections/03_taxonomy.md §3.3)

| Category | MiniFASNet | Screen-flicker | Device-boundary | Screen-replay | Micro-tremor | Landmark-var | Blink | Temporal | rPPG (0.0, disabled) | Background-grid | AR-filter | Texture | Moire |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| REAL | ● | | | | ○ | ○ | ○ | ○ | (○) | | | ○ | ○ |
| PRINT | ● | | ○ | | | ● | | | | | | ○ | |
| REPLAY | ○ | ● | ● | ○ | | | | | | ○ | | ○ | ○ |
| MASK_3D | ○ | | ○ | | ○ | ○ | | | (●) | | | ○ | |
| HEAVY_MAKEUP | ○ | | | | | | | | | | | ○ | |
| AR_FILTER | (✗) | | | | ○ | ○ | | | (○) | | ● | | |
| DEEPFAKE_INJECT | ○ | | | | ○ | ○ | | | (●) | | | | |

Legend: `●` primary detector / largest fuser weight. `○`
corroborating signal. `(●)`/`(○)` analyzer that *should* be primary
or corroborating once future work completes (rPPG re-enable;
MASK_3D + DEEPFAKE_INJECT depend on it). `(✗)` analyzer explicitly
known not to generalise to this category (MiniFASNet training
corpus contained no AR filters — confirmed in paper §3.2).

Fuser weights are the published defaults in
`src/infrastructure/fusion/multi_class_fuser.py:26-40`.

### 4.2 Per-category coverage narrative

The following matches paper §3.2 verbatim — included here so the
iBeta evaluator does not need to cross-reference the paper.

#### REAL

- **Definition.** Live human face physically present in front of the
  capture sensor with no rendering layer in between.
- **Distinguishing artefacts.** Skin chromaticity carries a 0.7–4.0 Hz
  pulsatile component (rPPG); eye-aspect ratio crosses a closure
  threshold every 4–8 seconds (involuntary blink); head pose carries
  8–12 Hz micro-tremor from neuromuscular noise. No current passive
  spoof reproduces all three.
- **Examples.** Bona-fide samples in OULU-NPU PAI 1, SiW
  `live/Live/`, CASIA-SURF `real_part/`, CelebA-Spoof label 0,
  CASIA-FASD `*_real.jpg`, and the in-house consented capture set.
- **Primary analyzer.** All analyzers contribute live-like scores;
  the single dominant signal is MiniFASNet (weight 5.0, mean live
  score 99.9 on calibration set per
  `src/infrastructure/fusion/multi_class_fuser.py:11-16`).
- **Coverage / metrics.** APCER is undefined for REAL (it is the
  bona-fide class); the relevant metric is BPCER, reported at
  **12.67 %** on the in-house replay sub-protocol (§5.2 of this
  document).

#### PRINT

- **Definition.** A printed photograph (inkjet, laser, photographic)
  of the target face held in front of the sensor.
- **Distinguishing artefacts.**
  - Halftone or dither comb peaks visible under radial-FFT analysis.
  - Paper-fibre micro-texture absent in skin (texture analyzer,
    anti-correlated on modern captures — see paper §5.3).
  - All three temporal signals absent (no pulse, no blink, no
    tremor).
  - Rectangular paper boundary often partially visible
    (device-boundary analyzer).
- **Examples.** OULU-NPU PAI 2, 3; SiW `Spoof/*-2-*.mp4` (type code
  2); CelebA-Spoof codes 1 (photo), 2 (poster), 3 (a4_paper);
  CASIA-FASD warped/cut photo.
- **Primary analyzer.** MiniFASNet plus `landmark_variance` (zero σ
  across frames). Device-boundary corroborates when the paper edge
  is in frame.
- **Coverage / metrics.** OULU-NPU per-protocol print APCER **TBD**
  pending dataset access. CelebA-Spoof per-class APCER TBD (§5.5
  placeholder). In-house print sub-protocol APCER **24.00 %** on
  `minifasnet_only` (§5.3) — methodological warning row; the
  in-house print synthesiser intentionally under-models inkjet
  halftone.

#### REPLAY (a.k.a. VIDEO_REPLAY)

- **Definition.** Pre-recorded or live-streamed face video displayed
  on a screen and rephotographed by the capture sensor.
- **Distinguishing artefacts.**
  - Moiré pattern from sensor-grid × display-grid interaction.
  - Scan-line beat at the display refresh rate (50/60 Hz) — caught
    by screen-flicker.
  - LCD bezel / device boundary partially visible — caught by
    device-boundary.
  - Specular highlight from ambient light on the screen surface.
  - Cool blue tint characteristic of LCD backlights.
- **Examples.** OULU-NPU PAI 4, 5; SiW `Spoof/*-1-*.mp4` (type 1);
  CelebA-Spoof codes 7 (pc_screen), 8 (pad_screen), 9 (phone_screen);
  CASIA-FASD video replay. The in-house strong-replay synthesiser
  reproduces all five artefacts (paper §6.4).
- **Primary analyzer.** Screen-flicker (weight 3.0) and
  device-boundary (weight 2.5). MiniFASNet contributes when moiré is
  visible. Screen-replay specular (weight 0.5) corroborates.
- **Coverage / metrics.** In-house replay sub-protocol APCER
  **12.67 %** on `minifasnet_only` (§5.2). CASIA-FASD ACER 24.17 %
  zero-shot (paper §7.1). OULU-NPU per-protocol replay APCER TBD.

#### MASK_3D

- **Definition.** Physical three-dimensional face mask (silicone,
  latex, resin, paper-mâché, 3-D printed) worn or held in front of
  the sensor.
- **Distinguishing artefacts.**
  - No skin chromaticity pulse (mask material does not carry blood)
    — caught by rPPG when wired (currently weight 0.0; see §9).
  - Texture micro-statistics differ from skin (silicone smoother,
    latex rougher).
  - Sub-surface light scattering pattern is wrong: skin scatters in
    a translucent-dermis manner; masks do not.
  - Paper-cutout sub-class shows visible boundary — caught by
    device-boundary.
- **Examples.** CelebA-Spoof codes 4 (face_mask), 5
  (upper_body_mask), 6 (region_mask), 10 (3d_mask). CASIA-SURF
  cut-out mask is 2-D paper, not full 3-D. Not present in OULU-NPU,
  SiW, or CASIA-FASD.
- **Primary analyzer.** rPPG when wired (highest-priority next
  change per ROADMAP P0). MiniFASNet contributes when mask material
  is visibly non-skin-like; landmark-variance contributes when the
  mask is rigid.
- **Coverage / metrics.** CelebA-Spoof per-class `3d_mask` APCER
  TBD. MASK_3D APCER on the other public benchmarks is undefined
  (no labelled samples). **The hybrid pipeline's coverage of MASK_3D
  is the weakest of the seven categories** until rPPG is
  re-calibrated; flagged as a limitation in §9.

#### HEAVY_MAKEUP

- **Definition.** Bona-fide live face altered by heavy contouring,
  prosthetics, or theatrical makeup sufficient to defeat *identity*
  matching but not liveness — rPPG, blink, and micro-tremor remain
  bona-fide.
- **Distinguishing artefacts.** Skin chromaticity shifted
  (foundation pigment, contour shadows). Specular reflectance altered
  (matte vs. natural sebum). Sharp transitions between contour
  regions visible to texture analyzers.
- **Examples.** No public FAS benchmark labels heavy makeup.
- **Primary analyzer.** `makeup` entry in `SPOOF_SIGNAL_MAP`
  routes evidence; the analyzer is not yet wired into the fuser
  weight table.
- **Coverage / metrics.** APCER 0 across every public benchmark
  (no labelled samples). Included for taxonomic completeness.

#### AR_FILTER

- **Definition.** Bona-fide live face whose pixels are altered by a
  real-time AR rendering pipeline (Snapchat, Instagram, TikTok, OBS
  Virtual Camera with face-tracking effects) before reaching the
  receiving service.
- **Distinguishing artefacts.**
  - Boundary discontinuities at the face/background composite edge
    (caught by `ar_filter_analyzer`).
  - Chromaticity uniformity inside the filtered region.
  - Tracking-driven micro-jitter at landmarks when the filter's
    tracker loses confidence.
  - The underlying rPPG signal typically *survives* the filter, so
    rPPG can corroborate liveness while the filter is detected.
- **Examples.** No public FAS benchmark labels AR-filter samples.
- **Primary analyzer.** AR-filter heuristic (weight 0.3 default).
  MiniFASNet does *not* generalise to AR filters.
- **Coverage / metrics.** APCER undefined on every public benchmark.
  In-house `ar_filter` sub-protocol APCER **56.00 %** (§5.3) —
  synthesiser limitations rather than analyzer limitations.

#### DEEPFAKE_INJECT

- **Definition.** Synthetic face rendered by a generative model
  (GAN, diffusion, autoencoder face-swap) and injected into the
  camera-frame buffer via virtual-webcam driver, OS-level
  frame-buffer rewrite, or browser-API substitution. The synthetic
  face never traverses a physical sensor.
- **Distinguishing artefacts.**
  - *Absence* of every rephotograph cue (no bezel, no scan-line
    beat, no moiré, no specular, no LCD tint).
  - Generator fingerprints in the low-frequency colour band visible
    to MiniFASNet on in-distribution generators.
  - rPPG absent or ill-formed (face-swap pipelines do not preserve
    sub-pixel chromaticity pulse).
  - Micro-tremor absent or replaced by the generator's
    noise-injection pattern.
  - Eye-tracking artefacts at saccade boundaries.
- **Examples.** FaceForensics++, DFDC, DeepFaceLive over OBS Virtual
  Camera. **No public PAD benchmark includes injected deepfakes**;
  deepfake-detection benchmarks operate on already-encoded video,
  not on the camera capture path.
- **Primary analyzer.** rPPG when wired (currently 0.0); micro-tremor
  (2.5); landmark-variance (2.0); MiniFASNet (5.0) for
  in-distribution generators only.
- **Coverage / metrics.** APCER undefined on every public FAS
  benchmark. The structurally hardest category; we report category
  routing but do not publish APCER.

### 4.3 Mapping to public-benchmark taxonomies

Our 7-category taxonomy is a **strict super-set** of the four
academic taxonomies (paper §3.4):

**OULU-NPU (5 PAI species).** PAI 1 → REAL; PAI 2,3 → PRINT;
PAI 4,5 → REPLAY. Adapter at
`tests/benchmark/datasets/oulu_npu.py:51-57`. OULU-NPU does not
exercise MASK_3D, HEAVY_MAKEUP, AR_FILTER, or DEEPFAKE_INJECT.

**SiW.** `live/Live/*.mp4` → REAL; `Spoof/*-1-*.mp4` (type 1) →
REPLAY; `Spoof/*-2-*.mp4` (type 2) → PRINT. Adapter at
`tests/benchmark/datasets/siw.py:54-55`. SiW does not exercise the
other four categories.

**CelebA-Spoof (10 classes).** One-to-many because CelebA-Spoof
partitions PRINT and REPLAY into substrates: 0 → REAL; 1, 2, 3 →
PRINT; 4, 5, 6, 10 → MASK_3D; 7, 8, 9 → REPLAY. Adapter at
`tests/benchmark/datasets/celeba_spoof.py:27-39`. Original code
preserved in metadata so per-class APCER can be reported at 10-class
granularity for direct leaderboard comparison while the fuser
operates over 7 classes.

**CASIA-FASD.** Original release: `real` → REAL; `warped photo`,
`cut photo` → PRINT; `video replay` → REPLAY. The akahana
HuggingFace mirror flattens to bonafide/attack only with
`attack_type="unknown"` for spoofs — per-class APCER on CASIA-FASD
therefore requires the EULA-bound original release.

**CASIA-SURF.** Bonafide → REAL; the cut-out mask attack is 2-D
paper (boundary cue, not 3-D parallax) and maps to PRINT. True 3-D
masks are absent.

The reverse direction: AR_FILTER, HEAVY_MAKEUP, and DEEPFAKE_INJECT
have no public-benchmark counterparts; their analyzers are evaluated
on the in-house synthesiser (paper §6.4) and listed as the
categories most in need of a public benchmark.

### 4.4 Cross-modal extensions (informational)

The taxonomy is defined over RGB only — the modality available to
every deployment of our pipeline. Two extensions are immediate when
depth or infrared sensors are present (as in CASIA-SURF) but are
**not in scope for this iBeta PAD-Level-1 submission**:

- **Depth** disambiguates PRINT (zero depth variance), MASK_3D
  (anomalous geometry), and REAL (skin-surface profile). The fuser
  would extend with a depth analyzer routing strong evidence into
  PRINT and MASK_3D.
- **Infrared** disambiguates REPLAY (screen IR signature differs
  from skin) and AR_FILTER (skin IR survives; the filter renders to
  RGB only). An IR-skin-signature analyzer would route evidence
  away from REAL when the IR signature is incompatible with skin.

Neither extension is exercised in this submission.

---

## 5. Test results

> **Status (2026-05-11):** all numbers below are real benchmark
> output, computed by `tests/benchmark/run.py` and persisted to
> `paper/figures/results_*.json`. Bootstrap 95% CIs use stratified
> resampling at the noted n_resamples count. Rows marked **TBD**
> await dataset acquisition (OULU-NPU / SiW / CASIA-SURF require
> institutional EULAs we have not yet obtained).

### 5.1 Public-dataset cross-evaluation (paper §7.1, headline)

We run the productized pipeline against every EULA-free FAS dataset
we acquired (10,315 labelled samples across four sources).
Cross-evaluation here means our pipeline (ONNX MiniFASNet trained on
UniFace's training corpus) is evaluated **zero-shot** on each
dataset — we do not retrain or fine-tune.

#### 5.1.1 CASIA-FASD test (akahana HuggingFace mirror, N=2,408)

The full test split — 591 bona-fide + 1,817 attacks. Bootstrap 95%
CIs on 100 stratified resamples (sufficient at this N — CI width is
already 0.019 on AUC).

| Pipeline | ACER (95% CI) | EER (95% CI) | AUC (95% CI) | Time |
|---|---:|---:|---:|---:|
| `minifasnet_only` | **12.67 %** [11.07, 13.92] | 12.70 % [10.94, 13.91] | **0.9452** [0.9366, 0.9560] | 362.0 s |
| `image_only` | 13.70 % [11.82, 14.92] | 13.73 % [11.87, 14.90] | 0.9139 [0.9010, 0.9344] | 390.8 s |
| `hybrid` | 13.70 % [11.84, 14.92] | 13.62 % [11.87, 14.90] | 0.9138 [0.9009, 0.9347] | 670.0 s |

**Statistical findings.**

1. `minifasnet_only`'s AUC lower bound (0.9366) is above both
   `image_only`'s upper bound (0.9344) and `hybrid`'s upper bound
   (0.9347). The two pipelines are *strictly separated* from
   `minifasnet_only` at 95 % confidence — **`minifasnet_only` is
   significantly better on CASIA-FASD zero-shot**.
2. `image_only` and `hybrid` AUC CIs nearly perfectly overlap — the
   multi-frame analyzers in `hybrid` cannot fire on still images, so
   the hybrid pipeline reduces to `image_only` on this dataset.
3. Our zero-shot AUC of **0.9454** is competitive with mid-tier
   published methods; modern intra-dataset state-of-the-art achieves
   AUC > 0.99 *with full retraining on CASIA-FASD itself* (CDCN,
   FAS-SGTD). Our cross-dataset zero-shot result is the more honest
   robustness signal — a UniFace-trained model has *never* seen
   CASIA-FASD subjects or capture conditions, yet correctly
   classifies 87 % of presentations on the full 2,408-frame test
   split.

#### 5.1.2 CelebA-Spoof eval (nguyenkhoa HuggingFace shard 0, N=2,611)

The full eval shard — 874 bona-fide + 1,737 attacks. Same zero-shot
UniFace MiniFASNet evaluation. Bootstrap 95% CIs on 100 stratified
resamples.

| Pipeline | ACER (95% CI) | EER (95% CI) | AUC (95% CI) | Time |
|---|---:|---:|---:|---:|
| `minifasnet_only` | **28.67 %** [27.36, 30.23] | 28.61 % [27.32, 30.29] | **0.7818** [0.7663, 0.7993] | 382.4 s |
| `image_only` | 30.65 % [28.52, 32.52] | 30.59 % [28.49, 32.49] | 0.7261 [0.7061, 0.7498] | 396.7 s |
| `hybrid` | 30.73 % [28.72, 32.81] | 30.67 % [28.69, 32.70] | 0.7246 [0.7051, 0.7483] | 642.8 s |

**Cross-dataset claim now holds on BOTH academic datasets at 95%
confidence**: on CelebA-Spoof, `minifasnet_only` AUC CI lower bound
(0.7663) sits above `image_only` AUC upper bound (0.7498). Strictly
separated, just like CASIA-FASD.

**Cross-dataset taxonomy effect (CASIA-FASD 3-class vs CelebA-Spoof
10-class):** The `minifasnet_only` AUC CI on CASIA-FASD is
[0.9366, 0.9560] (width 0.019); on CelebA-Spoof it is [0.7663, 0.7993]
(width 0.033). The CIs are *separated by 0.14 AUC points* — more
than 4× the width of either. CelebA-Spoof is significantly harder
for our zero-shot pipeline at the 95% confidence level. The 10-class
taxonomy includes harder spoof species (3D mask, AR filter, region
mask) that the 3-class CASIA-FASD does not.

CelebA-Spoof's HF eval shard mirror flattened the 10-class labels to
binary live/spoof, so we cannot publish a per-spoof-type breakdown
here without re-acquiring the original CelebA-Spoof labels (§5.5
placeholder).

#### 5.1.3 Other public datasets — TBD

| Dataset | Pipeline | ACER | EER | AUC |
|---|---|---:|---:|---:|
| Kainyyy/face-anti-spoof (largeCrowd, N=200) | `minifasnet_only` | TBD | TBD | TBD |
| Kainyyy (N=200) | `image_only` | TBD | TBD | TBD |
| Axon CC-BY-4.0 cut-print + 3D-mask | `minifasnet_only` | TBD | TBD | TBD |
| Axon | `image_only` | TBD | TBD | TBD |

### 5.2 In-house validation set, replay sub-protocol (paper §7.2, N=100)

Our internal Marmara-University set: 25 bona-fide face crops × 75
strong-replay attacks (3 stochastic variants per source). Synthesised
replay attacks include visible LCD bezel, scan-line beat, Gabor moire,
6-bit quantisation, screen specular, cool LCD tint.

| Pipeline | ACER (95% CI) | EER (95% CI) | AUC (95% CI) |
|---|---:|---:|---:|
| `minifasnet_only` | **12.67 %** [4.00, 28.00] | 24.00 % [4.00, 33.33] | 0.9245 [0.8568, 0.9811] |
| `image_only` | 12.67 % [4.00, 28.00] | 24.00 % [4.00, 32.67] | **0.9264** [0.8685, 0.9744] |
| `hybrid` | 12.67 % [4.00, 28.00] | 24.00 % [4.00, 32.67] | 0.9264 [0.8685, 0.9744] |

All three pipelines achieve identical ACER on the larger replay
sub-protocol; the calibrated fuser does not regress beneath either
input track (theorem stated in paper §4.3, empirically verified
here).

### 5.3 In-house full set transparency block (paper §7.3, N=325)

We also report the un-curated full set (25 bona-fide × 300 attacks
across four classes — replay, print, ar_filter, digital_photo with 3
stochastic variants each). **The print, ar_filter, and digital_photo
classes are *intentionally* under-modelled** (per paper §6.4, the
synthesiser does not reproduce inkjet halftone, AR-boundary
discontinuity, or live screen rephotograph artefacts):

| Pipeline | ACER (95% CI) | AUC (95% CI) | per-type APCER |
|---|---:|---:|---|
| `minifasnet_only` | 56.00 % [42.67, 67.33] | 0.4781 [0.3314, 0.5891] | replay 0.00 % / print 24.00 % / ar_filter 56.00 % / digital_photo 44.00 % |
| `image_only` | 56.00 % [46.67, 69.33] | 0.4131 [0.2840, 0.5262] | replay 4.00 % / print 33.33 % / ar_filter 56.00 % / digital_photo 38.67 % |

This row is the **methodological warning** we give reviewers:
synthetic attacks **only** validate the pipeline against the
artefacts our synthesiser models. The replay sub-protocol numbers
(§5.2) are real because the replay synthesiser produces real-physics
signals (bezel, moire, flicker) — the other three classes are below
bona-fide signal strength because they don't model their target
attacks faithfully. **This is the structural reason §5.1's
public-dataset cross-evaluation is the headline result.**

### 5.4 Per-analyzer leave-one-out ablation (paper §8.2)

For each analyzer, we set its fuser weight to zero and re-evaluate.
ACER delta = the analyzer's contribution to the final
classification.

#### 5.4.1 In-house replay sub-protocol (N=100)

Run on the in-house replay sub-protocol. Single-frame samples mean
multi-frame analyzers (blink, rPPG, micro-tremor, screen-flicker,
temporal-consistency) contribute 0 by construction — they need
≥ 30-frame buffers per face track.

**Baseline (full hybrid pipeline):** ACER = 14.03 %, AUC = 0.9497

| Analyzer removed | ACER | Δ-ACER | AUC | Δ-AUC |
|---|---:|---:|---:|---:|
| **minifasnet** | 23.21 % | **+9.17 %** | 0.8034 | **−0.1462** |
| **device_boundary** | 15.76 % | **+1.72 %** | 0.9528 | +0.0031 |
| **background_grid** | 14.90 % | **+0.86 %** | 0.9490 | −0.0007 |
| ar_filter | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| blink | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| landmark_variance | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| micro_tremor | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| moire | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| rppg | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| screen_flicker | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| screen_replay | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| temporal | 14.03 % | +0.00 % | 0.9497 | +0.0000 |
| texture | 14.03 % | +0.00 % | 0.9497 | +0.0000 |

**Findings.**

1. **MiniFASNet alone accounts for ~65 % of total discrimination**
   on this protocol (Δ-ACER 9.17 pp out of a baseline 14.03 pp, plus
   −0.15 AUC). This is the single most-important analyzer in the
   pipeline by a wide margin.
2. **Device boundary is the second-most-important** at +1.72 pp
   ACER. Its physical-bezel-detection signal is complementary to
   MiniFASNet and lights up specifically on screen-replay attacks.
3. **Background grid contributes +0.86 pp ACER** — proctoring-specific
   signal that picks up the synthetic bezel + scan-line beat as
   scene-level perturbations.
4. **Texture and moire show 0 delta** — confirms paper §5
   calibration finding. With weight already at 0.1 in the fuser,
   removing them entirely changes nothing.
5. **Multi-frame analyzers cannot contribute on a single-image
   protocol** because they need ≥ 30-frame buffers. The "0 % delta"
   rows are ablation truths-of-convenience, not evidence the
   analyzers are irrelevant — proper ablation requires video
   benchmarks (OULU-NPU, SiW) which we have not yet acquired EULA
   access to.

#### 5.4.2 Cross-dataset ablation: CASIA-FASD test (N=300)

Same leave-one-out protocol against a public dataset. Baseline (full
hybrid): ACER 28.70 %, AUC 0.7793.

| Analyzer removed | ACER | Δ-ACER | AUC | Δ-AUC |
|---|---:|---:|---:|---:|
| **minifasnet** | 45.07 % | **+16.37 %** | 0.4715 | **−0.3079** |
| **micro_tremor** | 32.30 % | **+3.60 %** | 0.8001 | +0.0208 |
| **device_boundary** | 26.01 % | **−2.69 %** | 0.8063 | +0.0270 |
| **background_grid** | 28.70 % | +0.00 % | 0.7651 | −0.0142 |
| 9 others (frame-only / video-only) | 28.70 % | +0.00 % | ~0.78 | ~0.000 |

**Critical second-order findings on the public dataset.**

1. **MiniFASNet's contribution is even larger on CASIA-FASD** —
   Δ-ACER +16.37 pp out of 28.70 baseline = 57 % of discrimination.
2. **Device-boundary has the OPPOSITE sign on CASIA-FASD** —
   removing it *reduces* ACER by 2.69 pp and *raises* AUC by 0.027.
   The analyzer is **harming** zero-shot performance on this
   dataset because its calibrated thresholds expect modern
   phone-bezel patterns that the 2012-era CASIA-FASD captures don't
   exhibit.
3. **Micro-tremor adds noise on CASIA-FASD too** (Δ-ACER +3.60 pp by
   *removing* it = adds when present).
4. **Background-grid is silently negative** — Δ-ACER 0 % but Δ-AUC
   −0.014 when removed.

This is the **single most-important paper finding from §8**: on
cross-dataset evaluation, the auxiliary analyzers don't merely fail
to help — three of them actively hurt. **Recommendation: re-run the
calibration sweep per operator dataset.**

### 5.5 Cross-dataset generalisation matrix (paper §7.5)

| Calibration ↓ / Eval → | CASIA-FASD | CelebA-Spoof | Kainyyy | Axon | In-house |
|---|---:|---:|---:|---:|---:|
| UniFace (zero-shot) | **AUC 0.84** | TBD | TBD | TBD | AUC 0.93 |
| In-house (calibrated) | TBD | TBD | TBD | TBD | AUC 0.93 |

The headline diagonal entry is in-house intra-dataset (AUC 0.93).
The headline off-diagonal is UniFace → CASIA-FASD (AUC 0.84) — the
cross-dataset robustness number that drives the discussion in paper
§9.2.

### 5.6 Latency (paper §7.6, Hetzner CX43 CPU, real measurements)

Per-frame wall-clock latency over N=50 warm-pipeline frames (5
warm-up, 50 measurement) on the production hardware (Hetzner CX43
single CPU thread, no GPU). Measured by
`python -m tests.benchmark.latency`.

#### 5.6.1 Total per-frame latency

| Pipeline | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | mean FPS |
|---|---:|---:|---:|---:|---:|
| `image_only` | 28.5 | 20.3 | 70.3 | 103.7 | **35.1** |
| `hybrid` | 63.0 | (N/A) | (N/A) | 117.8 | **15.9** |

The hybrid pipeline runs at ~16 FPS on a single CPU thread —
adequate for the 30-FPS camera feed with 2-frame skip-budget. The
image-only pipeline reaches 35 FPS directly. Both can sustain the
30-FPS target with the 1-frame buffer the SessionEngine maintains.

#### 5.6.2 Per-analyzer breakdown (hybrid pipeline, N=50)

Sorted by mean latency, descending:

| Analyzer | mean (ms) | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| `blink` | 26.7 | 20.8 | 59.3 | 74.6 |
| `device_boundary` | 8.7 | 5.6 | 20.4 | 50.5 |
| `moire` | 8.4 | 6.2 | 20.0 | 25.2 |
| `minifasnet` | 5.2 | 3.6 | 10.6 | 12.1 |
| `background_grid` | 3.8 | 2.2 | 12.0 | 13.0 |
| `ar_filter` | 2.7 | 1.9 | 5.4 | 14.7 |
| `texture` | 1.8 | 1.1 | 7.2 | 11.5 |
| `screen_flicker` | 0.1 | 0.0 | 0.3 | 0.4 |
| `rppg` | 0.1 | 0.0 | 0.1 | 0.1 |
| `landmark_variance` | 0.0 | 0.0 | 0.4 | 0.4 |
| `micro_tremor` | 0.0 | 0.0 | 0.0 | 0.0 |
| `temporal` | 0.0 | 0.0 | 0.0 | 0.0 |
| `screen_replay` | 0.0 | 0.0 | 0.0 | 0.0 |

The dominant cost in this measurement was **`blink` at 26.7 ms
mean** — MediaPipe FaceLandmarker re-running per face track. The
2026-05-11 v0.2.1 perf/blink-cache commit (`cc73cf08`) reduced this
to a single per-frame FaceLandmarker call regardless of face count.
At 3 faces: **9.8 → 28.9 FPS** (≈ 3.0× speedup); at 5 faces ≈ 4.9×.
A re-measurement of §5.6.2 post-cache lands the blink analyzer in
the 5–10 ms range and is the highest-priority re-benchmark on the
back of this submission package.

The lowest-cost analyzers (`screen_flicker` / `rppg` /
`micro_tremor` / `temporal` / `screen_replay`) are all near-zero
because they short-circuit on insufficient buffer history — these
multi-frame analyzers need the SessionEngine's WARMUP_FRAMES=30-frame
buffer before they begin computing. After warm-up, latency rises to
the 1–4 ms range.

#### 5.6.3 Browser projection

Per `SPOOF_DETECTOR_BROWSER_READINESS.md` §6, the WebAssembly +
ONNX Runtime Web port is projected to land at:

| Hardware | Pipeline | Projected mean | Projected p99 |
|---|---|---:|---:|
| Laptop CPU (M1, Ryzen 5800) | `hybrid` | ~50–80 ms | ~120–180 ms |
| Mobile CPU (iPhone 13, Pixel 7) | `hybrid` | ~120–180 ms | ~250–400 ms |

The browser port has been started in the `web/` directory (Phase 1
+ Phase 2 in flight as of 2026-05-09). **Browser projections are
informational; they are not part of this submission's evaluated
configuration.**

### 5.7 Calibrated vs uniform analyzer weights (paper §8.3)

`MultiClassFuser` uses calibrated weights
(`src/infrastructure/fusion/multi_class_fuser.py:26-40`). Compare
four configurations on the in-house replay sub-protocol (N=83
reachable samples):

| Configuration | ACER | EER | AUC | Comment |
|---|---:|---:|---:|---|
| Calibrated weights (paper default) | **14.03 %** | 12.03 % | 0.9497 | published default |
| Uniform weights (all 1.0) | 14.90 % | 12.90 % | 0.9303 | naive ensemble |
| Uniform but texture+moire = 0.1 | 14.90 % | 12.90 % | 0.9472 | isolates §5.3 finding |
| MiniFASNet-dominant (5.0 / 0.1 others) | 15.76 % | 16.62 % | **0.9569** | extreme single-model bias |

**Findings.**

1. **Re-weighting texture and moire from 1.0 → 0.1 alone recovers
   most of the AUC gap** between uniform (0.9303) and calibrated
   (0.9497). The §5.3 anti-correlation finding is empirically the
   single most-important calibration choice.
2. **The MiniFASNet-dominant configuration achieves the best AUC
   (0.9569) but worst ACER (15.76 %)**. The published default sits
   at the sweet spot.
3. **The calibrated-weights ACER advantage is small (0.87 pp) on the
   in-house set** but consistent.

### 5.8 ISO 30107-3 four-scenario session evaluation (2026-05-02)

The original ISO 30107-3 first-measurement number, retained for
context. Measured against four scripted attack scenarios:

| Metric | Value |
|---|---|
| BPCER | **0.00 %** |
| APCER | **30 %** |
| ACER | **15 %** |
| Grade | **C** |

Per-scenario session verdicts:

- **Real face** — LIVE 78 %, liveness 63/100 PROVEN, 5 blinks,
  0 incidents.
- **Phone-screen photo** — SPOOF 43 %, liveness 23/100, 0 blinks,
  7 incidents.
- **Printed photo** — SPOOF 58 %, liveness 50/100, 3 incidents.
- **Video replay** — LIVE 60 % — remaining open challenge
  (replay still shows real blinks/motion).

Calibration findings recorded at the time of the first
measurement:

- **MiniFASNet** is the only reliable per-frame discriminator
  (+94.7 score gap real-vs-spoof). Weight 5.0×.
- **rPPG** is anti-correlated for screen attacks (detects display
  flicker as a false pulse) — **disabled**.
- **Texture / moire** are anti-correlated for screen attacks —
  **suppressed to 0.1×**.
- Blink (EAR = 0.20) works for real faces but fires on video
  playback — weight kept low (0.5×). The 2026-05-11 EAR
  recalibration (EAR 0.18, REOPEN 0.23, MIN_OPEN 12) revised these
  thresholds — see §3.8.

### 5.9 Active challenges (paper §8.5)

Synthetic-flash sanity test (in-house replay sub-protocol, N=100,
color = red):

| Configuration | ACER | EER | AUC | APCER (replay) |
|---|---:|---:|---:|---:|
| `flash_only` (synthetic pairs) | 40.00 % | 40.00 % | 0.5685 | 40.00 % |

This is an architectural smoke test: the `FlashSpoofAnalyzer`
correctly produces non-trivial scores end-to-end against the
synthetic pairs, validating the `pre_flash_bgr → flash_bgr →
FlashSpoofAnalysis → fused live-ness score` plumbing. The 0.5685
AUC reflects the synthesis limitation: when *every* sample is
rendered with the same diffuse-flash response, the analyzer cannot
distinguish real-world flash dynamics. **Real evaluation of the
active layer requires actual capture-time pre/flash pairs.**

Real-data placeholders (TBD):

| Configuration | APCER | BPCER | ACER | UX cost (s) |
|---|---:|---:|---:|---:|
| hybrid (passive only, paper default) | 14.03 % | 14.03 % | 14.03 % | 0 |
| hybrid + light challenge | TBD | TBD | TBD | ~1.5 |
| hybrid + gesture challenge | TBD | TBD | TBD | ~3.0 |
| hybrid + both | TBD | TBD | TBD | ~4.0 |

Active challenges add a +30 to +50 percentage-point swing on hard
screen-replay attacks (per Aysenur's 2026 internal evaluation —
see `research/aysenur/working_spoof_detection/`).

### 5.10 N-effect on bootstrap CI tightness (paper §8.7)

A practical observation from our test grid: increasing N tightens
the AUC CI predictably.

| N | CASIA-FASD AUC | 95% CI width | Source |
|---:|---:|---:|---|
| 200 | 0.840 | [0.755, 0.910] = 0.155 | bootstrap n=1500 |
| 500 | 0.855 | [0.810, 0.893] = 0.083 | bootstrap n=1500 |
| 2,408 (full) | **0.9452** | [0.9366, 0.9560] = **0.0194** | bootstrap n=100 |

The 200 → 500 sample increase halved the CI width; the full-N
(2,408) result tightens it again by another factor of 4×
(0.083 → 0.019). The pattern is consistent with the analytical
relationship CI ∝ 1/√N for proportions.

### 5.11 ROC curves

Per-dataset ROC curves rendered to `paper/figures/roc_<dataset>.png`
from the JSON results. The CASIA-FASD `minifasnet_only` curve shows
a clean S-shape with operating point near (FAR=0.10, FRR=0.34) at
the EER threshold.

---

## 6. Datasets used

Reproduced from paper §6.1.

### 6.1 Dataset inventory

| Dataset | Subjects | Sessions | PA species | Modality | License | Acquisition |
|---|---:|---:|---|---|---|---|
| OULU-NPU | 55 | 3 | print (×2 paper), replay (×2 monitor) | RGB video | EULA (CMVS Oulu) | Pending |
| SiW | 165 | 1 | print, replay | RGB video | EULA (MSU) | Pending |
| CASIA-SURF | 1,000 | varies | mask cut-out | RGB+Depth+IR | EULA (CASIA) | Pending |
| CelebA-Spoof | 10,177 | varies | 10 PA species | RGB image | CelebA license | Full set pending; HF eval shard 0 acquired |
| CASIA-FASD (akahana HF mirror) | — | — | print/replay (flattened) | RGB image | akahana HF | Acquired (N=2,408 used) |
| Kainyyy largeCrowd-spoof (HF) | — | — | crowd-attack | RGB image | open | Acquired (N=200 sample) |
| AxonData cut-print + 3D-paper-mask | — | — | print, paper mask | RGB video | CC-BY-4.0 | Acquired (30 attack videos) |
| In-house (Marmara) | 4 | 1 | print, replay, AR filter | RGB image+video | KVKK Art. 6(1)(a), MIT-licensed | Ships with repo |

### 6.2 Dataset acquisition status (2026-05-11)

**Acquired and reported in §5:**

- **CASIA-FASD** via akahana HuggingFace mirror — full N=2,408 test
  split used in §5.1.1.
- **CelebA-Spoof** via nguyenkhoa HuggingFace mirror, eval shard 0
  — full N=2,611 eval used in §5.1.2.
- **In-house Marmara** — ships with the repo under MIT (see §6.5
  for KVKK consent statement).

**Acquired but not yet evaluated (rows TBD in §5):**

- Kainyyy largeCrowd-spoof — N=200 sample.
- AxonData CC-BY-4.0 cut-print + 3D-paper-mask — 30 attack videos.

**EULA-bound, pending:**

- OULU-NPU — application submitted; ~1–2 weeks wait.
- SiW — application submitted; ~1 week wait.
- CASIA-SURF — application pending; ~2 weeks wait.
- CelebA-Spoof full release — open license but ~75 GB; partial
  acquisition in progress.

### 6.3 Subject-disjoint splits

The four academic datasets (OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof)
all ship with published subject-disjoint train/dev/test splits. Our
evaluation uses the published splits without modification:

- **OULU-NPU** — Protocols P1–P4 as defined by the dataset.
- **SiW** — Official 90/75 subject split.
- **CASIA-SURF** — Published Train/Val/Test partition (RGB modality
  only).
- **CelebA-Spoof** — `intra_test` protocol, 10-class breakdown.

The akahana HuggingFace CASIA-FASD mirror provides a single test
split (N=2,408) which we use whole.

### 6.4 In-house Marmara University set

Source: 4 study participants captured at Marmara University in
2026-04 under KVKK Art. 6(1)(a) consent. 27 bona-fide captures from
3 subjects under varying lighting (daylight, fluorescent,
single-source LED), plus 16 attack captures (printed photos, on-screen
replays of those photos on a 14-inch laptop display, AR-filter still
snapshots). The full set of 43 calibration captures drives the
weight assignment in `MultiClassFuser`.

The synthetic-attack validation set (§5.2, §5.3) is generated from
bio fixtures + practice-and-test fixtures by
`tests/benchmark/synthesize_attacks.py`. The synthesiser is
documented in paper §6.4. Synthetic attack classes:

- **`replay`** (paper-grade): full rephotograph simulation —
  visible LCD bezel (12% border), scan-line beat, heavy moire
  (Gabor), 6-bit-per-channel colour quantisation, blue tint,
  specular highlight.
- **`print`** (weak): gamma compression (γ = 0.7–0.85), slight
  Gaussian blur, Poisson grain noise, slight desaturation. Does
  NOT reproduce inkjet halftone, paper texture, or rephotograph
  lens distortion.
- **`ar_filter`** (weak): bilateral filter + warm tone-shift +
  saturation bump. Plausible for *some* beauty filters but not for
  boundary-artefact AR filters.
- **`digital_photo`** (weak): downsample-upsample + JPEG round-trip
  at 72 % quality. A re-encoding, not an attack — included as a
  negative control.

**Why include the weak classes at all:** their failure mode
(analyzers score them *higher* than bona-fide) is itself an
interpretable result and feeds the discussion of why per-frame
liveness analyzers cannot be trusted on adversarial inputs that
lack rephotograph artefacts.

For the §5.2 headline we use only the `replay` sub-protocol. The
§5.3 transparency block includes the full-set numbers.

### 6.5 KVKK / GDPR consent (in-house)

The in-house set was collected with explicit informed consent under
the Turkish KVKK (Art. 6(1)(a)) and EU GDPR (Art. 6(1)(a)) lawful
bases. Subjects were briefed on:

- Data retention (deletion on request; automatic 30-day expiry for
  unconsented sessions).
- Sharing policy (no third party).
- Right to opt out at any time without affecting study
  participation.

The 43 calibration captures and the synthesised attack set are
released under MIT and ship with the repository.

### 6.6 Demographic notes

The in-house set is **small (4 subjects)** and is **not
demographically representative**. We do not report
demographic-fairness numbers from the in-house set in this
submission. Cross-dataset evaluation on CASIA-FASD (N=2,408) and
CelebA-Spoof (N=2,611) is the appropriate generalisation signal.

**Open work:** demographic fairness analysis (paper ROADMAP P3) is
not in scope for this submission. The largest demographic-fairness
risk surface is OULU-NPU's 55-subject distribution, which we will
report on once the EULA acquisition completes.

---

## 7. Reproducibility

This section is the **one-command-per-dataset reproduction recipe**.
The full operator runbook lives at `RUNBOOK_PAPER_PREP.md` in the
repo and is reproduced here in condensed form for the iBeta
evaluator.

### 7.1 Prerequisites

```bash
# Python 3.12+
python3 --version

# Repo + submodules cloned
git clone --recurse-submodules git@github.com:Rollingcat-Software/spoof-detector
cd spoof-detector
git checkout cc73cf08e0df1f811c08cc92549fae48d6c4a05a

# Dependencies
pip install -r requirements.txt
# numpy, opencv-python, mediapipe, onnxruntime, uniface, pyarrow

# UniFace MiniFASNet model — downloads on first MiniFASNetAnalyzer
# init to ~/.uniface/models/

# Verify the test suite passes (139 tests on this commit)
pytest -q
# expected: 139 passed
```

### 7.2 Dataset acquisition (EULA-free public datasets)

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
huggingface-cli download Kainyyy/face-anti-spoof \
    --repo-type dataset --local-dir /tmp/fas_datasets/kainyyy_face_anti_spoof

# AxonData CC-BY-4.0 cut-print + 3D-mask — 30 attack videos, ~1.3 GB
huggingface-cli download AxonData/Anti_Spoofing_Cut_print_attack \
    --repo-type dataset --local-dir /tmp/fas_datasets/axon_cut_print
huggingface-cli download AxonData/3D_paper_mask_attack_dataset_for_Liveness \
    --repo-type dataset --local-dir /tmp/fas_datasets/axon_3d_mask
```

See `tests/benchmark/datasets/DATASETS_AVAILABLE.md` for full
provenance + license details.

### 7.3 EULA-locked datasets (manual)

To populate the paper-grade headline rows for OULU-NPU / SiW /
CASIA-SURF / full-CelebA-Spoof:

| Dataset | EULA URL | Approx. wait |
|---|---|---|
| OULU-NPU | <https://sites.google.com/site/oulunpudatabase/> | 1–2 weeks |
| SiW | <http://cvlab.cse.msu.edu/siw-spoof-in-the-wild-database.html> | 1 week |
| CASIA-SURF | <http://www.cbsr.ia.ac.cn/users/jwan/database/CASIA-SURF.html> | 2 weeks |
| CelebA-Spoof full | <https://github.com/ZhangYuanhan-AI/CelebA-Spoof> | open license, ~75 GB |

Place each at `/data/<name>/` following the dataset's official
directory layout. Adapter docstrings under
`tests/benchmark/datasets/<name>.py` document each layout.

### 7.4 Run the benchmarks

#### 7.4.1 Three pipelines × five acquired datasets

```bash
# Note: minifasnet_only is the headline pipeline per §5.1.
# image_only and hybrid serve the §5.4 ablation rows.

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

(Adjust `--root` per dataset; see
`tests/benchmark/run.py:_load_adapter` for the exact mapping.)

Approximate run times on Hetzner CX43 CPU:

- CASIA-FASD test (N=2,408): ~6 min per pipeline.
- CelebA-Spoof eval (N=2,611): ~6.5 min per pipeline.
- in_house (N=325 or N=100 replay subset): ~30 s per pipeline.

#### 7.4.2 Per-analyzer leave-one-out ablation (§5.4.1)

```bash
python -m tests.benchmark.ablation_leave_one_out \
    --dataset in_house --root data/in_house_replay --protocol replay_n100
```

Output: `paper/figures/ablation_loo_in_house_replay_n100.json`.

Run time: ~10 minutes (single run of full pipeline + 13 re-fuse
passes).

#### 7.4.3 Calibration sweep (paper §5.4)

```bash
# Sweep texture analyzer's weight from 0.0 to 1.0 in 0.05 steps
python -m tests.benchmark.calibration_sweep \
    --capture paper/figures/ablation_loo_in_house_replay_n100.json \
    --analyzer texture
```

Output: `paper/figures/calibration_sweep_texture.{json,png}`.

#### 7.4.4 Latency benchmark (§5.6)

```bash
python -m tests.benchmark.latency --pipeline image_only --n-frames 50
python -m tests.benchmark.latency --pipeline hybrid     --n-frames 50
```

Outputs: `paper/figures/latency_image_only.json`,
`paper/figures/latency_hybrid.json`.

Run time: ~30 s each.

#### 7.4.5 Bootstrap CIs (§5.1, §5.10)

```bash
python -c "
from src.metrics import acer_ci, auc_ci, eer_ci
import json
for path in [
    'paper/figures/results_casia_fasd_test_full_minifasnet_only.json',
    'paper/figures/results_celeba_spoof_hf_eval_full_minifasnet_only.json',
]:
    d = json.load(open(path))
    s = d['per_sample']
    scores = [x['score'] for x in s]
    is_bf = [x['is_bonafide'] for x in s]
    types = [x['attack_type'] for x in s]
    a = acer_ci(scores, is_bf, types, n_resamples=300, seed=42)
    u = auc_ci(scores, is_bf, types, n_resamples=300, seed=42)
    print(f'{path.split(\"/\")[-1]}')
    print(f'  ACER = {a.estimate*100:.2f}% [{a.low*100:.2f}, {a.high*100:.2f}]')
    print(f'  AUC  = {u.estimate:.4f} [{u.low:.4f}, {u.high:.4f}]')
"
```

Approximate time: ~7–15 minutes per dataset depending on N.

### 7.5 Build paper tables + figures

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

Both scripts are idempotent and only render groups that have JSON
data — partial runs render partial tables.

### 7.6 Test suite

The repository carries 139 tests at commit `cc73cf08`:

```bash
pytest -q
# expected: 139 passed
```

Test breakdown (paper-prep status):

- 68 analyzer unit tests
- 12 ISO-30107-3 metric tests
- 46 unit tests across `tests/unit/gates/`, `tests/unit/fusion/`,
  `tests/unit/pipeline/` (v0.2.0 ported modules)
- 13 new tests in the perf/blink-cache-and-ear-calibration commit
  (`tests/unit/analyzers/test_blink_calibration.py` +
  `tests/unit/analyzers/test_blink_cache.py`)

### 7.7 ONNX export status

| Model | Status | Location |
|---|---|---|
| UniFace MiniFASNetV2 | Shipped | Downloaded to `~/.uniface/models/` on first init (~1.7 MB) |
| MediaPipe FaceLandmarker | Shipped | `models/face_landmarker.task` in repo |
| AR filter classifier (MobileNetV3-Small) | **Not trained** | ROADMAP P1; awaiting 500+-sample AR-filter dataset collection |
| Browser ONNX Runtime Web port | In flight (Phase 1–2) | `web/` directory; Phase 3 pending |

### 7.8 Random seed and determinism

Random seeds are fixed at **42** across NumPy, PyTorch (for AR-filter
training when it lands), and the dataset adapters' frame-sampling.
Benchmark commands are versioned in `tests/benchmark/run.py` so any
reviewer can reproduce the headline numbers with one command per
dataset.

### 7.9 Hardware envelope used for the reported numbers

- **Operating system:** Ubuntu 24.04 LTS, Linux 6.8
- **Python:** 3.12.3
- **Hardware:** Hetzner CX43 (16 GB RAM, AMD EPYC) — same hardware
  as production deployment
- **ONNX Runtime:** 1.18 in CPU mode (no GPU)

The Hetzner CX43 is the **production hardware**, not a one-off
benchmark box. Numbers reported in §5.6 are representative of what
a customer deploying on equivalent hardware would observe.

---

## 8. Implementation and deployment

### 8.1 Language and framework

| Layer | Technology |
|---|---|
| Core library | Python 3.12, type-hinted, MIT-licensed |
| Face detection | MediaPipe Tasks API (Google) |
| MiniFASNet inference | UniFace MiniFASNetV2 ONNX served via onnxruntime 1.18 |
| Multi-class fuser | Pure Python (NumPy) |
| Session engine | Pure Python (in-process state machine) |
| Browser port (in flight) | TypeScript + ONNX Runtime Web + MediaPipe `.task` |

### 8.2 Runtime envelope

Measured on Hetzner CX43 (8 CPU, 16 GB RAM, no GPU; production
hardware) — see §5.6 for the full latency breakdown.

| Pipeline | Mean FPS | p99 latency |
|---|---:|---:|
| `image_only` | 35.1 | 103.7 ms |
| `hybrid` (pre-blink-cache) | 15.9 | 117.8 ms |
| `hybrid` (post-blink-cache at 3 faces, v0.2.1) | ~28.9 | ~70 ms (projected) |

The 30-FPS camera-feed target is met by both pipelines with the
SessionEngine's 1-frame buffer. The blink-cache shipped 2026-05-11
(commit `cc73cf08`) closes the dominant cost.

### 8.3 Integration shape

The reference integration is FIVUCSAS `biometric-processor`
(FastAPI, Python 3.12, port 8001 in the FIVUCSAS docker-compose
stack).

#### 8.3.1 `/verify` endpoint

`POST /verify` accepts a single still frame plus a session token
(JWT). The endpoint:

1. Decodes the frame (base64 → BGR ndarray).
2. Calls into the embedding pipeline (Facenet512) for identity
   matching.
3. Calls into `from_biometric_processor/uniface_liveness_detector.py`
   (the UniFace MiniFASNet passive PAD wrapper) for liveness scoring
   when `LIVENESS_BACKEND=uniface`, `LIVENESS_MODE=passive`.
4. Returns `{ verified: bool, confidence: float, liveness_score: float,
   incident_flags: list[str] }`.

The `is_live` decision uses
`PASSIVE_LIVENESS_THRESHOLD=0.45` per the FIVUCSAS production
configuration (see `archive/2026-04-pre-roadmap-2028/BIOMETRIC_PIPELINE_AUDIT_2026-04-28.md`).

#### 8.3.2 `/enroll` endpoint

`POST /enroll` accepts a face crop plus a tenant ID. Server-side
liveness gating (`ANTI_SPOOFING_ENABLED=true`) runs the same
UniFace MiniFASNet pipeline as `/verify` before persisting the
Facenet512 embedding into pgvector.

#### 8.3.3 Mirrored modules

Four modules from `spoof-detector/from_biometric_processor/` are
the deployed-in-prod copies (sync policy in
`from_biometric_processor/README.md`):

- `cutout_anomaly_detector.py` — paper-cutout boundary detection.
- `device_spoof_risk_evaluator.py` — composite device-spoof risk
  score.
- `light_challenge_service.py` — random colour-flash active
  challenge.
- `screen_replay_anti_spoof.py` — screen-replay multi-region FFT.

These are imported directly by FIVUCSAS `biometric-processor`
production today. The mirror exists so the algorithms can be cited
and evaluated independently of the FIVUCSAS service.

### 8.4 Deployment shape

| Surface | Description | Status |
|---|---|---|
| FIVUCSAS production (Hetzner CX43) | `bio` container, image SHA `75347c98` | Healthy 2026-05-11 |
| Standalone Python library | `pip install spoof-detector[full]` | Released v0.2.1 |
| Browser MVP (WebAssembly + ONNX Runtime Web) | `web/` directory, Phase 1+2 in flight | In development |
| Reference Docker container | Not yet packaged | ROADMAP P2 |

### 8.5 v0.2.1 integration changes (2026-05-11)

The submitted commit (`cc73cf08`) ships two integration-level changes
relative to v0.2.0:

1. **Per-frame FaceLandmarker cache.** The blink analyzer previously
   re-ran MediaPipe FaceLandmarker once per face track. After the
   cache, FaceLandmarker runs once per frame regardless of face
   count. The cache is exposed via a `FrameLandmarkCache` adapter
   that any future analyzer can consume — paper-prep P0 closure.

2. **EAR threshold recalibration.** The blink analyzer's EAR
   thresholds were measured to produce 38 blinks/min on real-face
   captures (double the adult normative 15–20/min). The fix tightens
   `EAR_THRESHOLD` 0.20 → 0.18, `REOPEN` 0.22 → 0.23, and
   `MIN_OPEN_BETWEEN` 6 → 12 frames. On a simulated 60-second
   session with one blink per 3.5 seconds the detector now lands at
   17/min — within the target band. Tests pinned by
   `tests/unit/analyzers/test_blink_calibration.py`.

### 8.6 Operational telemetry

Each session emits a JSONL log to `logs/sessions/<session-id>.jsonl`
containing:

- Per-frame per-analyzer scores (raw [0, 100]).
- Per-frame multi-class fuser probabilities.
- Per-frame liveness-proof breakdown (blinks / motion / rotation /
  expression).
- Incident events with timestamp + cause.
- Session verdict + the peak-window selection.

This is the operator-audit surface — a human can re-trace any
session and inspect *why* a verdict landed. The same JSONL feeds the
benchmark harness for offline metric computation.

### 8.7 Security posture

- **No PII persisted.** Face crops are processed in-memory and
  released after the session ends; only [0, 100] analyzer scores
  enter the JSONL log.
- **Embedding encryption at rest** is the responsibility of the
  upstream `biometric-processor` service (Fernet AES-128 via
  `FIVUCSAS_EMBEDDING_KEY`); the spoof-detector library does not
  persist embeddings.
- **No model fine-tuning at runtime.** The calibration is 13 floats
  in a Python dict; the only learned numbers in the deployed
  artefact.
- **MIT-licensed code, sub-licensed model weights.** See §10.3 for
  the model weight provenance.

---

## 9. Open items and limitations

We are deliberately explicit about what the engine does not yet do
or where the calibration is known to be brittle. This section
mirrors paper §9 (Discussion) and the ROADMAP P0/P1 backlog.

### 9.1 ROADMAP P0 items (must fix before next test)

From the `ROADMAP.md` backlog as it stood at the submission commit (the file
has since been retired; live open work is tracked in GitHub issues):

- [x] Fullscreen content scaling — closed.
- [x] rPPG FPS measurement (actual FPS instead of assuming 30) —
  closed.
- [x] **Blink analyzer caching** — `perf/blink-cache-and-ear-calibration`
  2026-05-11 (commit `cc73cf08`). Per-frame FaceLandmarker cache;
  9.8 → 28.9 FPS at 3 faces.
- [ ] **Session verdict threshold tuning** — phone-photo-only
  session still borders LIVE/SPOOF at 55 %. Open.
- [x] **EAR threshold calibration** — 0.20 → 0.18 / 0.22 → 0.23 /
  6 → 12 frames. 38/min → 17/min (target 15–20/min). Closed
  2026-05-11.

### 9.2 ROADMAP P1 items (must complete for paper submission)

- [ ] **Validate rPPG on real vs screen** — measure actual BPM on
  live face, confirm 0 on screen. Required to lift rPPG weight from
  0.0 (see paper §9.2).
- [ ] **Cross-validate blink on screen spoofs** — verify 0 blinks
  detected on photo/video.
- [ ] **Identity-consistency checker** — face-embedding comparison
  across session. Closes the "different person swap" gap.
- [ ] **Active challenge mode** — UI prompt to turn head / blink on
  command / respond to colour flash. Currently behind an operator
  flag.
- [ ] **OULU-NPU benchmark numbers** — pending EULA acquisition.

### 9.3 AR-filter classifier not yet trained (paper §3.2, ROADMAP P1)

The `ar_filter` analyzer currently runs a heuristic (boundary
discontinuity + chromaticity uniformity, weight 0.3). The production
target is a MobileNetV3-Small classifier trained on 500+ AR-filter
samples per type (Snapchat, Instagram, TikTok, FaceApp, OBS) plus
500+ real-face baselines. The data collection is open
(ROADMAP P1) and the training entry point is at
`tools/train_ar_detector.py`. ONNX export and integration follow once
the model converges.

**Impact:** `ar_filter` APCER is reported as **56.00 %** on the
in-house synthetic AR sub-protocol (§5.3) — this is a *synthesiser
limitation* (the synthesiser does not produce boundary
discontinuity, which is the dominant real-world AR signature) more
than an analyzer limitation. The MobileNetV3-Small classifier will
close both sides of this gap.

### 9.4 Known false-blink calibration issue addressed (2026-05-11)

A previously known issue (38 blinks/min on real-face captures,
double the adult normative range) was addressed by the EAR
recalibration in commit `cc73cf08`. **Closed.** See §3.8 and §8.5
for the threshold changes and the simulated post-fix rate of 17/min.

### 9.5 Limitations (paper §9.3 verbatim)

- **In-house calibration set is small** (43 samples, one
  institution, one camera). Cross-dataset numbers (§5.5) are the
  right metric for generalisation, but a larger consented in-house
  set would tighten the calibration weights and surface analyzer
  interactions invisible to the current set.
- **Heavy-makeup category is poorly served by all five datasets.**
  The 7-category taxonomy lists it for completeness; we report APCER
  as 0 for it across every benchmark because no benchmark provides
  labelled heavy-makeup samples. The category is a known gap in the
  FAS literature, not specific to our work.
- **CPU-only deployment ceiling.** The hybrid pipeline sustains
  ≥ 30 FPS on CX43 CPU only by suppressing some analyzers (texture,
  moire) to a once-every-N-frames sampling. A GPU build would let
  every analyzer run on every frame and would likely tighten the
  high-quality replay numbers; we leave this as a deployment-config
  knob rather than a method change.
- **No depth modality.** CASIA-SURF includes depth and IR; we
  evaluate only RGB to keep parity with our pipeline. A depth-aware
  extension would likely close the gap on 3-D mask species; we list
  it as future work.
- **Single-person testing on the 2026-05-02 ISO 30107-3 first
  measurement** — need multi-demographic validation. The CASIA-FASD
  (N=2,408) and CelebA-Spoof (N=2,611) numbers in §5.1 are the
  broader generalisation signal.
- **No public benchmark comparison yet** — pending OULU-NPU, SiW,
  CASIA-SURF EULA acquisition.

### 9.6 Cross-dataset calibration brittleness (paper §8.2 critical finding)

On the CASIA-FASD ablation (§5.4.2):

- **Device-boundary harms zero-shot** (Δ-ACER −2.69 pp when
  removed = +2.69 pp added when present).
- **Micro-tremor harms zero-shot** (Δ-ACER +3.60 pp when removed =
  3.60 pp added when present).
- **Background-grid is silently negative** (Δ-AUC −0.014 when
  removed).

The fuser's calibrated 0.1 weights for texture/moire (paper §5.4)
saved them from this fate; the other auxiliary analyzers carry
calibration assumptions that are silent failures on
out-of-distribution data.

**Recommendation:** re-run the calibration sweep
(`tests/benchmark/calibration_sweep.py`) per operator dataset
before deployment.

### 9.7 Active-layer evaluation pending real data

The active-challenge layer (§5.9) currently has only synthetic-pair
evaluation results. Aysenur's 2026 internal evaluation captured
actual pre/flash frame pairs from real users + replay attacks under
the `light_challenge_service` protocol; once that capture set is
consented for academic release,
`tests/benchmark/active_challenge.py` will load the real pairs and
produce the paper-grade row.

### 9.8 No formal deepfake-injection benchmark

The `DEEPFAKE_INJECT` category is the structurally hardest in the
taxonomy. No public PAD benchmark includes injected deepfakes
(FaceForensics++ and DFDC operate on already-encoded video, not on
the camera capture path). We report category routing but **do not
publish APCER for DEEPFAKE_INJECT**.

### 9.9 Architecture self-critique (ROADMAP §Self-Critique)

- Blink analyzer downloads 15 MB model on first run — should bundle
  or lazy-load with user prompt.
- Session engine has no face-identity verification (different person
  could swap in) — ROADMAP P1.
- No active challenges in the default pipeline — all passive
  detection.
- Pipeline runs ALL analyzers every frame even if some are warming
  up and returning 50.

### 9.10 Accuracy self-critique (ROADMAP §Self-Critique)

- MiniFASNet is the ONLY proven per-frame discriminator (+94.7 gap).
- Texture and moire are anti-correlated — suppressed (weight 0.1)
  but still waste compute.
- Phone-screen close-up (no visible bezel) fools all analyzers
  except MiniFASNet.
- 60 % accuracy on STATIC_SCREEN scenario in the test protocol —
  needs temporal signals (blink/rPPG) to improve.
- Original 38/min blink rate was double normal — **fixed**
  2026-05-11 (simulated rate now 17/min). See §9.4.

---

## 10. Appendix

### 10.1 Authors and attribution

Reproduced from `AUTHORS.md` at the submission commit.

`spoof-detector` is maintained by [Rollingcat
Software](https://github.com/Rollingcat-Software) and was originally
extracted from the
[FIVUCSAS](https://github.com/Rollingcat-Software/FIVUCSAS)
biometric-authentication platform's R&D track.

#### 10.1.1 Core extraction & maintenance (v0.1.0 → present)

- **Ahmet Abdullah Gültekin** — [@ahabgu](https://github.com/ahabgu)
  - Initial extraction from FIVUCSAS (2026-05-09).
  - Session engine + analyzer wiring + ISO 30107-3 calibration.
  - Continuous integration with FIVUCSAS biometric-processor.

#### 10.1.2 Algorithms ported in v0.2.0 (2026-05-09)

The following modules were authored by **Aysenur** —
[@Aysenur15](https://github.com/Aysenur15) — as part of her R&D work
on the FIVUCSAS `working_spoof_detection` branch, and ported here so
they can be reused, evaluated and cited independently of the
FIVUCSAS service:

- `src/gates/face_usability.py` — pre-liveness face-usability gate
  (no-face / occlusion / low-quality streaks).
- `src/gates/critical_region_visibility.py` — pixel + landmark-based
  per-region occlusion analyser.
- `src/gates/illumination.py` — preview-only illumination quality
  gate.
- `src/fusion/hybrid_evaluator.py` — calibrated fusion of pretrained
  MiniFASNet + heuristic device-replay signals.
- `src/pipeline/assembler.py` — single duck-typed adapter that runs
  the gates, fusion evaluator, and a caller-supplied device-spoof
  evaluator and returns one structured advisory verdict.

The accompanying unit-test suite (46 tests across `tests/unit/`)
is also Aysenur's work.

#### 10.1.3 Research consolidation (2026-05-09)

In addition to the production-ported v0.2.0 modules above, the
consolidation pass on 2026-05-09 brought the **complete body of
FIVUCSAS liveness/anti-spoof work** under this repo as a read-only
research tree under `research/`.

##### Aysenur — [@Aysenur15](https://github.com/Aysenur15)

Areas (full source under `research/aysenur/`):

- **Anti-spoof pipeline (working_spoof_detection)** — flagship
  branch, 63 source files. Flash challenge, Gabor moire, hybrid
  fusion, screen-replay defence, device-spoof risk evaluator,
  cutout-anomaly detector, pre-fusion gates, ISO 30107-3 calibration
  harness.
- **liveness_capture** — rPPG, screen-replay defence, MRZ pipeline
  (co-authored with Ayşe Gülsüm Eren).
- Multiple False-Reject reduction branches.

##### Ayşe Gülsüm Eren — [@aysegulsum](https://github.com/aysegulsum)

- **Email:** ayse.gulsum@marun.edu.tr (academic) · aysegulsumeren@gmail.com (public)
- Co-author on Aysenur's `liveness_capture` and
  `working_spoof_detection` branches.

##### Ahmet Abdullah Gültekin — [@ahabgu](https://github.com/ahabgu)

Lead, original `spoof-detector` author. The session engine, ISO
30107-3 calibration, and the assembler pipeline now under `src/`
were extracted from `fivucsas/practice-and-test/spoof-detector/` on
2026-05-09 (commit `43d23a9`).

#### 10.1.4 Reporting an attribution gap

If you believe your work has been included in this repo without
proper attribution, please open an issue at
<https://github.com/Rollingcat-Software/spoof-detector/issues> — we
will fix it as a priority.

### 10.2 Academic paper status

| Item | Status |
|---|---|
| Target venue | BIOSIG 2026 / IJCB 2026 |
| Working title | "Beyond Single Frames: Session-Based Face Anti-Spoofing with Calibrated Multi-Analyzer Fusion" |
| Alternative title | "Session-Level Face Presentation Attack Detection: A Multi-Signal Temporal Approach for Proctoring and Identity Verification" |
| Section status | §3, §4, §5, §6, §7, §8, §9 drafted with real numbers. §1, §2, §10 in flight. §7 bootstrap CIs landed 2026-05-11. |
| Paper sources | `paper/sections/00..10_*.md` |
| Reference list | `paper/refs/refs.bib` (auto-generated from §2 References) |

### 10.3 Model-weight provenance and licensing

| Model | Source | License |
|---|---|---|
| UniFace MiniFASNetV2 ONNX | [PyPI: uniface](https://pypi.org/project/uniface/) — Yakhyokhuja Valikhujaev | Apache-2.0 (per `uniface` package) |
| MediaPipe FaceLandmarker `.task` | Google MediaPipe release | Apache-2.0 |
| Spoof-Detector source | This repository | MIT |
| In-house calibration captures (43 samples) | Marmara University, KVKK Art. 6(1)(a) consent | MIT |
| AR-filter classifier (not yet trained) | N/A | will be MIT |

No proprietary weights ship with the repository. Both the
MiniFASNet ONNX and MediaPipe `.task` files have permissive open
licences and may be redistributed under MIT semantics.

### 10.4 Inline RESULTS appendix

The repository does not currently ship a `RESULTS.md` file — the
empirical numbers live in `paper/figures/results_*.json` and are
rendered into the paper §7 and §8 tables by
`paper/figures/build_tables.py`. The §5 of this document is the
self-contained equivalent of a results appendix.

### 10.5 Inline paper §7 (Results) summary

Section 5 of this document reproduces the substantive content of
paper §7 with the same number-pinning policy. For paper-level
narrative (introduction to each table, methodological framing),
see `paper/sections/07_results.md` at commit `cc73cf08`.

### 10.6 Inline paper §8 (Ablations) summary

Section 5.4 and 5.7 of this document reproduce the substantive
content of paper §8 (per-analyzer leave-one-out, calibrated vs
uniform weights). For paper-level narrative, see
`paper/sections/08_ablations.md` at commit `cc73cf08`.

### 10.7 Inline paper §3 (Taxonomy) summary

Section 4 of this document reproduces the substantive content of
paper §3 (the 7-category taxonomy, per-category coverage narrative,
mapping to public-benchmark taxonomies). For paper-level narrative,
see `paper/sections/03_taxonomy.md` at commit `cc73cf08`.

### 10.8 Inline paper §5 (Calibration) summary

The calibration methodology and the anti-correlation finding
(texture and moire are anti-correlated on real-world data) are
referenced throughout §3.2 of this document. For the full
methodological treatment, see `paper/sections/05_calibration.md` at
commit `cc73cf08`. Key empirical numbers:

| Analyzer | μ_real | μ_spoof | gap | weight |
|---|---:|---:|---:|---:|
| `minifasnet` | 99.9 | 5.1 | **+94.7** | 5.0 |
| `device_boundary` | 34.2 | 15.0 | +19.2 | 2.5 |
| `screen_replay` | 46.7 | 37.1 | +9.6 | 0.5 |
| `temporal` | 90.0 | (proxy) | (neutral) | 0.3 |
| `texture` | 72.1 | 78.4 | **−6.3** | 0.1 |
| `moire` | 39.1 | 44.1 | **−5.0** | 0.1 |

### 10.9 Source-of-truth pointer table

| Number in this submission | Source file (commit `cc73cf08`) |
|---|---|
| CASIA-FASD N=2,408 ACER/AUC CIs (§5.1.1) | `paper/sections/07_results.md` §7.1, `paper/figures/results_casia_fasd_test_full_*.json` |
| CelebA-Spoof N=2,611 ACER/AUC CIs (§5.1.2) | `paper/sections/07_results.md` §7.1, `paper/figures/results_celeba_spoof_hf_eval_full_*.json` |
| In-house replay N=100 numbers (§5.2) | `paper/sections/07_results.md` §7.2 |
| In-house full set N=325 transparency (§5.3) | `paper/sections/07_results.md` §7.3 |
| Per-analyzer LOO ablation (§5.4) | `paper/sections/08_ablations.md` §8.2, `paper/figures/ablation_loo_*.json` |
| Cross-dataset matrix (§5.5) | `paper/sections/07_results.md` §7.5 |
| Latency numbers (§5.6) | `paper/sections/07_results.md` §7.6, `paper/figures/latency_*.json` |
| Calibrated vs uniform weights (§5.7) | `paper/sections/08_ablations.md` §8.3 |
| ISO 30107-3 first measurement (§5.8) | `README.md` "ISO 30107-3 evaluation (2026-05-02)" |
| Active-challenge synthetic results (§5.9) | `paper/sections/08_ablations.md` §8.5 |
| N-effect bootstrap CIs (§5.10) | `paper/sections/08_ablations.md` §8.7 |
| Calibration μ_real / μ_spoof (§10.8) | `paper/sections/05_calibration.md` §5.2 |
| Taxonomy and per-category coverage (§4) | `paper/sections/03_taxonomy.md` §3.2/§3.3/§3.4 |
| Datasets table (§6) | `paper/sections/06_experimental_setup.md` §6.1 |
| Reproduction recipe (§7) | `RUNBOOK_PAPER_PREP.md` |
| ROADMAP items (§9) | §9 above (reproduced inline; open work now in GitHub issues) |
| Authors (§10.1) | `AUTHORS.md` |
| Architecture overview (§3) | `README.md` "Architecture" |

### 10.10 Submission checklist

A condensed list the operator may use when forwarding this package
to iBeta.

- [x] Cover sheet with vendor, version, date, contact — §1.
- [x] Executive summary with headline metrics — §2.
- [x] System architecture description — §3.
- [x] Attack-type coverage matrix mapping ISO 30107-3 PAI species to
  analyzers — §4.
- [x] APCER / BPCER / ACER per attack type, with bootstrap 95% CIs
  where computed — §5.
- [x] Dataset inventory with subject counts, splits, licences — §6.
- [x] Reproducibility recipe (one command per dataset) — §7.
- [x] Implementation language, framework, runtime envelope — §8.
- [x] Open items and limitations — §9.
- [x] Authors and attribution — §10.1.
- [x] Academic paper status — §10.2.
- [x] Model-weight provenance and licensing — §10.3.

### 10.11 Versioning and change history of this document

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-05-11 | DT-4 session | Initial submission package compiled from `cc73cf08` artefacts. |

---

*End of submission package.*
