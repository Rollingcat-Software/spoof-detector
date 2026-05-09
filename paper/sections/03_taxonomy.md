# 3. A 7-Category Spoof Taxonomy

## 3.1 Why seven categories and not two

The conventional FAS literature treats PAD as binary and reports a single APCER aggregated across whatever attack species a dataset contains. Three properties of production deployment make this aggregation lossy.

First, **per-category APCER is operationally actionable**: an operator who knows the deployment has 2% APCER on print but 35% APCER on deepfake injection can route high-stakes operations away from purely visual liveness; "12% APCER" alone supports no such decision. ISO/IEC 30107-3 defines APCER per-PAI, but most published FAS work pre-aggregates the maximum and discards the per-class structure.

Second, **different analyzers target different categories**. The fuser in `src/infrastructure/fusion/multi_class_fuser.py:43-93` routes each analyzer's evidence into the spoof category for which the analyzer was designed (`src/domain/taxonomy.py:27-114`); multi-class is the natural representation, not a post-hoc decomposition.

Third, **a multi-class taxonomy lets the engine flag absent corroboration**. A 3-D mask produces live-like scores from the moiré and screen-flicker analyzers, forcing a binary fuser to call it live; the 7-class fuser exposes that no positive *spoof* category accumulated evidence either, so the operator dashboard can default to "low-confidence, escalate".

The seven categories are defined in `src/domain/models.py:12-30` as the `SpoofCategory` enum. §3.2 gives the per-category spec; §3.3 is Table 1; §3.4 maps to public-benchmark taxonomies.

## 3.2 Per-category specification

### REAL

**Definition.** Live human face physically present in front of the capture sensor with no rendering layer in between.

**Distinguishing artefacts.** Skin chromaticity carries a 0.7–4.0 Hz pulsatile component (rPPG); eye-aspect ratio crosses a closure threshold every 4–8 seconds (involuntary blink); head pose carries 8–12 Hz micro-tremor from neuromuscular noise. No current passive spoof reproduces all three.

**Examples.** Bona-fide samples in OULU-NPU PAI 1, SiW `live/Live/`, CASIA-SURF `real_part/`, CelebA-Spoof label 0, CASIA-FASD `*_real.jpg`, and the in-house consented capture set.

**Primary analyzer.** All analyzers contribute live-like scores; the single dominant signal is MiniFASNet (weight 5.0, mean live score 99.9 on calibration set per `src/infrastructure/fusion/multi_class_fuser.py:11-16`).

**Coverage / metrics.** APCER is undefined for REAL (it is the bona-fide class); the relevant metric is BPCER, reported at 12.67% on the in-house replay sub-protocol (§7.2). Public-benchmark BPCER is TBD pending EULA acquisition.

### PRINT

**Definition.** A printed photograph (inkjet, laser, photographic) of the target face held in front of the sensor.

**Distinguishing artefacts.**
- Halftone or dither comb peaks visible under radial-FFT analysis.
- Paper-fibre micro-texture absent in skin (texture analyzer, anti-correlated on modern captures — see §1).
- All three temporal signals absent (no pulse, no blink, no tremor).
- Rectangular paper boundary often partially visible (device-boundary analyzer).

**Examples.** OULU-NPU PAI 2, 3; SiW `Spoof/*-2-*.mp4` (type code 2); CelebA-Spoof codes 1 (photo), 2 (poster), 3 (a4_paper); CASIA-FASD warped/cut photo.

**Primary analyzer.** MiniFASNet plus `landmark_variance` (zero σ across frames). Device-boundary corroborates when the paper edge is in frame.

**Coverage / metrics.** OULU-NPU per-protocol print APCER TBD pending dataset access. CelebA-Spoof per-class APCER TBD (§7.4). In-house print sub-protocol APCER 24.00% on `minifasnet_only` (§7.3); the in-house print synthesiser intentionally under-models inkjet halftone, so this is a methodological warning row.

### REPLAY (a.k.a. VIDEO_REPLAY)

**Definition.** Pre-recorded or live-streamed face video displayed on a screen and rephotographed by the capture sensor.

**Distinguishing artefacts.**
- Moiré pattern from sensor-grid × display-grid interaction (Gabor moiré analyzer; anti-correlated on modern displays — see §1).
- Scan-line beat at the display refresh rate (50/60 Hz) — caught by screen-flicker.
- LCD bezel / device boundary partially visible — caught by device-boundary.
- Specular highlight from ambient light on the screen surface.
- Cool blue tint characteristic of LCD backlights.

**Examples.** OULU-NPU PAI 4, 5; SiW `Spoof/*-1-*.mp4` (type code 1); CelebA-Spoof codes 7 (pc_screen), 8 (pad_screen), 9 (phone_screen); CASIA-FASD video replay. The in-house strong-replay synthesiser (§6.4) reproduces all five artefacts.

**Primary analyzer.** Screen-flicker (weight 3.0) and device-boundary (weight 2.5). MiniFASNet contributes when moiré is visible. Screen-replay specular (weight 0.5) corroborates.

**Coverage / metrics.** In-house replay sub-protocol APCER 12.67% on `minifasnet_only` (§7.2); CASIA-FASD ACER 24.17% zero-shot (§7.1). OULU-NPU per-protocol replay APCER TBD.

### MASK_3D

**Definition.** Physical three-dimensional face mask (silicone, latex, resin, paper-mâché, 3-D printed) worn or held in front of the sensor.

**Distinguishing artefacts.**
- No skin chromaticity pulse (mask material does not carry blood) — caught by rPPG when wired (currently weight 0.0; see §9.2).
- Texture micro-statistics differ from skin (silicone smoother, latex rougher).
- Sub-surface light scattering pattern is wrong: skin scatters in a translucent-dermis manner; masks do not.
- Paper-cutout sub-class shows visible boundary — caught by device-boundary.

**Examples.** CelebA-Spoof codes 4 (face_mask), 5 (upper_body_mask), 6 (region_mask), 10 (3d_mask). CASIA-SURF cut-out mask is 2-D paper, not full 3-D. Not present in OULU-NPU, SiW, or CASIA-FASD.

**Primary analyzer.** rPPG when wired (highest-priority next change per §9.2). MiniFASNet contributes when mask material is visibly non-skin-like; landmark-variance contributes when the mask is rigid.

**Coverage / metrics.** CelebA-Spoof per-class `3d_mask` APCER TBD (§7.4). MASK_3D APCER on the other public benchmarks is undefined (no labelled samples). The hybrid pipeline's coverage of MASK_3D is the weakest of the seven categories until rPPG is re-calibrated; flagged as a limitation in §9.3.

### HEAVY_MAKEUP

**Definition.** Bona-fide live face altered by heavy contouring, prosthetics, or theatrical makeup sufficient to defeat *identity* matching but not liveness — rPPG, blink, and micro-tremor remain bona-fide. The category exists because heavy makeup defeats the upstream task PAD protects and operators of high-stakes onboarding need a flag for human review.

**Distinguishing artefacts.**
- Skin chromaticity shifted (foundation pigment, contour shadows).
- Specular reflectance altered (matte vs. natural sebum).
- Sharp transitions between contour regions visible to texture analyzers.

**Examples.** No public FAS benchmark labels heavy makeup; stage/theatrical and cosmetics-industry sets are the natural source but are not part of standardised PAD evaluation.

**Primary analyzer.** The `makeup` entry in `SPOOF_SIGNAL_MAP` (`src/domain/taxonomy.py:68-72`) routes evidence; the analyzer is not yet wired into the fuser weight table.

**Coverage / metrics.** APCER 0 across every public benchmark (no labelled samples). Included for taxonomic completeness.

### AR_FILTER

**Definition.** Bona-fide live face whose pixels are altered by a real-time AR rendering pipeline (Snapchat, Instagram, TikTok, OBS Virtual Camera with face-tracking effects) before reaching the receiving service.

**Distinguishing artefacts.**
- Boundary discontinuities at the face/background composite edge (caught by `ar_filter_analyzer`).
- Chromaticity uniformity inside the filtered region (filters typically smooth skin chromaticity beyond what real skin exhibits).
- Tracking-driven micro-jitter at landmarks when the filter's tracker loses confidence (visible to landmark-variance over short windows).
- The underlying rPPG signal typically *survives* the filter, so rPPG can corroborate liveness while the filter is detected.

**Examples.** No public FAS benchmark labels AR-filter samples; the closest proxies are CelebA-Spoof's mask codes (physical, not AR) and our `ar_filter` synthesiser (§6.4), reported as a methodological warning row in §7.3.

**Primary analyzer.** AR-filter heuristic (weight 0.3 default; raised in AR-rejection-priority deployments). MiniFASNet does *not* generalise to AR filters (its training corpus contained none).

**Coverage / metrics.** APCER undefined on every public benchmark. In-house `ar_filter` sub-protocol APCER 56.00% on both `minifasnet_only` and `image_only` (§7.3) — synthesiser limitations rather than analyzer limitations (see §6.4, §9.3).

### DEEPFAKE_INJECTION (a.k.a. DEEPFAKE_INJECT)

**Definition.** Synthetic face rendered by a generative model (GAN, diffusion, autoencoder face-swap) and injected into the camera-frame buffer of the capture endpoint via virtual-webcam driver, OS-level frame-buffer rewrite, or browser-API substitution. The synthetic face never traverses a physical sensor.

**Distinguishing artefacts.**
- *Absence* of every rephotograph cue (no bezel, no scan-line beat, no moiré, no specular, no LCD tint) — the diagnostic signature is a pristine-looking frame with *no* rephotograph artefacts.
- Generator fingerprints in the low-frequency colour band visible to MiniFASNet on in-distribution generators (cross-generator generalisation remains open, [Dolhansky 2020]).
- rPPG absent or ill-formed (face-swap pipelines do not preserve sub-pixel chromaticity pulse).
- Micro-tremor absent or replaced by the generator's noise-injection pattern with different band statistics.
- Eye-tracking artefacts at saccade boundaries when temporal coherence is imperfect.

**Examples.** FaceForensics++ ([Rossler 2019]), DFDC ([Dolhansky 2020]), DeepFaceLive over OBS Virtual Camera. No public PAD benchmark includes injected deepfakes; the deepfake-detection benchmarks operate on already-encoded video, not on the camera capture path.

**Primary analyzer.** rPPG when wired (currently 0.0); micro-tremor (2.5); landmark-variance (2.0); MiniFASNet (5.0) for in-distribution generators only. No single analyzer is sufficient for cross-generator robustness — the structurally hardest category, underlying the open problem in §9.5.

**Coverage / metrics.** APCER undefined on every public FAS benchmark. We report category routing but do not publish APCER; a proper evaluation requires a benchmark that does not yet exist publicly.

## 3.3 Table 1 — Taxonomy summary

| Category | One-line definition | Key physical artefact | Primary analyzer (largest fuser weight) | Public dataset coverage |
|---|---|---|---|---|
| REAL | Live human face physically present | Pulse + blink + micro-tremor | MiniFASNet (5.0) | OULU-NPU PAI 1, SiW live, CASIA-SURF real, CelebA-Spoof 0, CASIA-FASD real |
| PRINT | Printed photograph held to sensor | Halftone + zero motion | MiniFASNet + landmark_variance (2.0) | OULU-NPU PAI 2,3; SiW type 2; CelebA-Spoof 1,2,3; CASIA-FASD warped/cut |
| REPLAY | Video replay on screen, rephotographed | Moiré + scan-line + bezel | screen_flicker (3.0) + device_boundary (2.5) | OULU-NPU PAI 4,5; SiW type 1; CelebA-Spoof 7,8,9; CASIA-FASD replay |
| MASK_3D | 3-D physical face mask | Absent rPPG + non-skin scatter | rPPG (0.0, disabled) + MiniFASNet | CelebA-Spoof 4,5,6,10; CASIA-SURF (cut-out) |
| HEAVY_MAKEUP | Live face with heavy cosmetic alteration | Chromaticity + specular shift | makeup-routing (un-wired, see §9.3) | None labelled |
| AR_FILTER | Live face with real-time AR rendering | Boundary discontinuity + chromaticity uniformity | ar_filter (0.3) | None labelled |
| DEEPFAKE_INJECT | Synthetic face injected via virtual webcam | *Absence* of rephotograph cues | rPPG (0.0) + micro_tremor (2.5) + landmark_variance (2.0) | None labelled in PAD benchmarks; FaceForensics++ / DFDC are video-only |

Fuser weights are the published defaults in `src/infrastructure/fusion/multi_class_fuser.py:26-40`.

## 3.4 Mapping to public-benchmark taxonomies

Our 7-category taxonomy is a strict super-set of the four academic taxonomies.

**OULU-NPU (5 PAI species).** PAI 1 → REAL; PAI 2,3 → PRINT; PAI 4,5 → REPLAY. The adapter (`tests/benchmark/datasets/oulu_npu.py:51-57`) maps PAI codes onto our `(bonafide / attack + sub-type)` tuples directly. OULU-NPU does not exercise MASK_3D, HEAVY_MAKEUP, AR_FILTER, or DEEPFAKE_INJECT.

**SiW.** `live/Live/*.mp4` → REAL; `Spoof/*-1-*.mp4` (type 1) → REPLAY; `Spoof/*-2-*.mp4` (type 2) → PRINT. Adapter at `tests/benchmark/datasets/siw.py:54-55`. SiW does not exercise the other four categories.

**CelebA-Spoof (10 classes).** One-to-many because CelebA-Spoof partitions PRINT and REPLAY into substrates: 0 → REAL; 1, 2, 3 → PRINT; 4, 5, 6, 10 → MASK_3D; 7, 8, 9 → REPLAY. The adapter (`tests/benchmark/datasets/celeba_spoof.py:27-39`) preserves the original code in metadata so per-class APCER (§7.4) can be reported at 10-class granularity for direct leaderboard comparison while the fuser operates over 7 classes.

**CASIA-FASD.** Original release: `real` → REAL; `warped photo`, `cut photo` → PRINT; `video replay` → REPLAY. The akahana HuggingFace mirror (`tests/benchmark/datasets/casia_fasd.py:1-48`) flattens to bonafide/attack only with `attack_type="unknown"` for spoofs — per-class APCER on CASIA-FASD therefore requires the EULA-bound original release.

**CASIA-SURF.** Bonafide → REAL; the cut-out mask attack is 2-D paper (boundary cue, not 3-D parallax) and maps to PRINT. True 3-D masks are absent.

The reverse direction is also explicit: AR_FILTER, HEAVY_MAKEUP, and DEEPFAKE_INJECT have no public-benchmark counterparts; their analyzers are evaluated on the in-house synthesiser (§6.4) and §9 flags them as the categories most in need of a public benchmark.

## 3.5 Cross-modal extension footnote

The taxonomy is defined over RGB only — the modality available to every deployment of our pipeline. Two extensions are immediate when depth or infrared sensors are present (as in CASIA-SURF):

- **Depth** disambiguates PRINT (zero depth variance), MASK_3D (anomalous geometry), and REAL (skin-surface profile). The fuser would extend with a depth analyzer routing strong evidence into PRINT and MASK_3D.
- **Infrared** disambiguates REPLAY (screen IR signature differs from skin) and AR_FILTER (skin IR survives; the filter renders to RGB only). An IR-skin-signature analyzer would route evidence away from REAL when the IR signature is incompatible with skin.

Neither extension is exercised in §7 (RGB-only protocol for parity with the published pipeline). They are listed because the 7-class fuser absorbs new modalities as additional analyzers in `src/infrastructure/fusion/multi_class_fuser.py:43-93` without changing the category structure — a property a binary fuser would not have.
