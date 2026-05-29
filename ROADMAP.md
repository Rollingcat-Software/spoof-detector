# Spoof Detector Roadmap

**Project**: FIVUCSAS Session-Based Face Presentation Attack Detection
**Paper Target**: BIOSIG 2026 / IJCB 2026
**Demo**: https://amispoof.fivucsas.com/ — browser-side reference implementation (migrated 2026-05-17 from fivucsas.com/amispoof/; old URL 301s)
**Last Updated**: 2026-05-17 (PRs #25–#48 — full multi-signal roadmap shipped: 19 analyzers + 15-axis passive liveness-proof scorer, subdomain migration, recorder + replay UI, browser-first paper reframe)

## Status (2026-05-17)

| Surface | State |
|---|---|
| **Browser bundle** (`@rollingcat/spoof-detector` on npm, live at amispoof.fivucsas.com) | **Production** — 19 analyzers, 15-axis proof scorer, 185-pt ceiling; 6.7–9.5 fps Pixel-class Android, 25–30 fps desktop Chrome WebGPU; 217 vitest green |
| **Python reference** (`src/`) | **Production** — calibration source-of-truth + ISO/IEC 30107-3 benchmark harness for §6–§8 of the paper; 138 pytest green |
| **Paper** (`paper/sections/`) | **Browser-first reframe** complete (PR #45); §0/§1/§10 lead with the client-side bundle, Python framed as starting-point + evaluation reference; §7.6 updated with measured browser-port perf numbers |
| **iBeta PAD-Level-1 submission package** | Scoped to the Python pipeline; re-scope to the browser bundle is a planned next iteration |

### Phases shipped this iteration (2026-05-16 / 17)

| Phase | Scope | Adds |
|---|---|---|
| **A** (PR #31) | MediaPipe blendshape + 3D matrix unlock | EyebrowAnalyzer, BlinkSymmetryAnalyzer, GazeAnalyzer, ExpressionDynamicsAnalyzer, Pose3DConsistencyAnalyzer + 5 LivenessProver axes |
| **B** (PR #32) | Behavioural temporal patterns | BehavioralPatternAnalyzer (blink-CV + saccade rate + Shannon entropy) + 1 axis |
| **C** (PR #33) | Skin colour-temperature drift | folded into TextureAnalyzer |
| **D1** (PR #34) | SelfieSegmenter background motion | BackgroundMotionAnalyzer + 1 axis (opt-in) |
| **D2** (PR #35) | HandLandmarker | HandTrackingAnalyzer + 1 axis (opt-in) |
| **D3** (PR #36) | Audio + audio-mouth sync | AudioCapture + VoiceActivityAnalyzer + AudioMouthSyncAnalyzer + 2 axes (opt-in) |
| Polish (PRs #37–#43) | Decoupling head pose from eye/mouth axes, rate-aware blink_points, tab-visibility pause + camera-recovery, mobile button labels, FileReader fallback, recorder + replay UI | — |
| Subdomain (PRs #44–#47) | amispoof.fivucsas.com cutover | sitemap.xml + robots.txt, 301 from old URL, og:image, paper §4/§6 honesty pass |

Deployment runbook: [`web/amispoof/SUBDOMAIN_MIGRATION.md`](./web/amispoof/SUBDOMAIN_MIGRATION.md).
.htaccess template: [`web/amispoof/SUBDOMAIN_HTACCESS_TEMPLATE.txt`](./web/amispoof/SUBDOMAIN_HTACCESS_TEMPLATE.txt).

## Where things live

`spoof-detector` is now the single canonical home for every
liveness / anti-spoof asset across the FIVUCSAS programme.

| Directory                  | Status     | Audience                                 |
|----------------------------|------------|------------------------------------------|
| `src/`                     | production | importable Python library; v0.2.0 curated. |
| `web/`                     | production | TypeScript port published as `@rollingcat/spoof-detector` + `/amispoof/` browser tester. |
| `tests/`                   | production | 139 Python tests, green.                |
| `tools/`                   | operator   | offline / desktop scripts.               |
| `research/aysenur/`        | reference  | Aysenur's 7 FIVUCSAS R&D branches.       |
| `research/ayse-gulsum-eren/` | reference  | attribution + commit pointers.         |
| `research/ahmet-original-spoof-detector/` | reference  | pointer; content is in `src/`. |
| `from_biometric_processor/`  | mirror     | algorithms also deployed in `bio` main; |
|                            |            | mirror-only — edit upstream first.       |

See [`research/README.md`](research/README.md) for the layout rationale and
[`from_biometric_processor/README.md`](from_biometric_processor/README.md) for
the sync policy.

## Resuming research

The most fertile starting ground is
[`research/aysenur/working_spoof_detection/`](research/aysenur/working_spoof_detection/)
— it contains:

- The flash-challenge service that became `from_biometric_processor/light_challenge_service.py`.
- The Gabor / moire detector and the screen-replay defence that became `from_biometric_processor/screen_replay_anti_spoof.py`.
- The cutout-anomaly detector that became `from_biometric_processor/cutout_anomaly_detector.py`.
- The device-spoof risk evaluator that became `from_biometric_processor/device_spoof_risk_evaluator.py`.
- The hybrid-fusion experimental harness that informed `src/fusion/hybrid_evaluator.py`.

For each of those, the curated, ISO-30107-3-calibrated v0.2.0 surface lives
under `src/`, but the experimental ground-truth, alternate-threshold sweeps,
and unit tests for in-flight ideas remain in the research tree.

## Productization checklist (research → src)

Any module graduating from `research/` (or `from_biometric_processor/`) into
`src/` must satisfy:

1. **Tests** — at least one unit test per public function; integration test
   if it crosses a gate / fusion / pipeline boundary. Co-locate under
   `tests/unit/<module-area>/`.
2. **Dependencies** — every imported package present in `requirements.txt`
   with a pinned version, and the new pin must pass Dependabot / `pip-audit`.
3. **Public API** — top-level docstring; `__all__` declared; type hints on
   public surface; no `app.*` imports left over from upstream namespaces.
4. **Provenance** — module docstring lists original author, original branch,
   and the consolidation commit it was promoted from.
5. **Attribution** — entry added to [`AUTHORS.md`](AUTHORS.md).

When a module graduates from `from_biometric_processor/`, also coordinate
with the bio team so that `biometric-processor` imports the productized
spoof-detector copy (rather than maintaining its own fork) — this is how the
mirror eventually retires.

---

## Current State (v0.2.0 / 2026-05-09)

- 9 analyzers (MiniFASNet, Device Boundary, Blink, rPPG, Screen Replay, Temporal, Texture, Moire, AR Filter)
- Session engine with incident detection and peak-sensitive verdict
- 139 tests, 23 source modules, 8 tools, ~7000 lines
- Tested accuracy: LIVE session 98% confidence, SPOOF session 63% + 13 incidents
- Blink detection working (20 blinks detected in 31s)
- rPPG pulse detection implemented (needs validation)

## Browser Port v0.1.0 / 2026-05-16 — Phases 1-4 complete

Four consecutive phases of the TypeScript port (`web/`) shipped across
2026-05-15 → 2026-05-16:

- **Phases 1-3** (PR #19): 6 base analyzers + Aysenur's 3 gates + hybrid_evaluator + assembler + 4 more analyzers (rPPG, Moire, Texture, ScreenReplay).
- **Bug-fix wave** (PRs #20-#22): verdict-lock warmup, no-blink incident wiring, blink-rate fps math, gate-panel UX, SEO surface, landmark/video alignment, stale-cache self-heal.
- **Phase 4** (PR #23): paper-calibrated weights, BackgroundGrid + Temporal analyzers, LivenessProver, Web Worker offload + frame-skip scheduler, WebGPU EP with WASM fallback, lazy bundle chunks, in-page CASIA-FASD validation harness.

Modules in `web/src/` at HEAD:

| Layer | Files | LOC | Tests |
|---|---|---|---|
| Analyzers (12) | MiniFASNet, Blink, LandmarkVariance, DeviceBoundary, MicroTremor, ScreenFlicker, Rppg, Moire, Texture, ScreenReplay, BackgroundGrid, Temporal | ~3,900 | 49 |
| Gates (Aysenur, 3) | FaceUsability, IlluminationGate, CriticalRegionVisibility | ~1,400 | 17 |
| Fusion | MultiClassFuser (paper-calibrated weights) + HybridFusionEvaluator (Aysenur) | ~450 | 28 |
| Pipeline | AntispoofPipelineAssembler (Aysenur) | ~290 | 18 |
| Session | SessionEngine + LivenessProver | ~850 | 10 |
| Workers | HeavyAnalyzerWorker + HeavyAnalyzerPool (Vite `?worker&inline`) | ~340 | 5 |
| Validation | CasiaFasdMicroBench (in-page accuracy harness) | ~190 | 5 |
| Detection | MediaPipeFaceDetector | ~150 | — |
| **Total** | 24 source files | **~7,570** | **126 vitest, all green** |

Calibrated weights (Phase 4A, per paper §5.3 + §8.3 LOO findings):
- texture 0.0, moire 0.0 (anti-correlated on high-res capture).
- device_boundary 0.5, micro_tremor 0.5 (zero-shot LOO harm).
- background_grid held at 1.5 (sole transferable positive contributor).

Deployment: https://amispoof.fivucsas.com/ — webcam-driven tester with
12-analyzer fusion, advisory face-usability gate panel, face bbox +
478-pt landmark overlay, copy button, downloadable JSON snapshot, and
a one-click `Run accuracy bench` button hitting
`runCasiaFasdMicroBench`. Main bundle 123 kB ESM / 34 kB gzip + 3 lazy
chunks (Texture 7.9 kB, Moire 10.5 kB, ScreenReplay 18.2 kB) + Worker
chunk 30 kB. Runtime `onnxruntime-web` + `@mediapipe/tasks-vision`
lazy-loaded from jsdelivr.

Performance: Web Worker offload + frame-skip (heavy N=3, gate M=5)
brings mobile Brave from ~4 fps to ~8–12 fps; PC Chrome hits ~25–30 fps.
WebGPU EP gives ~3× MiniFASNet speedup on capable browsers with silent
fallback to WASM on Brave/iOS Safari.

Mobile Brave constraints respected throughout: no SharedArrayBuffer,
no WASM threads, no hard WebGPU dep, no hard OffscreenCanvas dep.

Outstanding (browser):
- [ ] Browser-tuned per-region gate thresholds (gates still over-flag
      mobile selfies — needs a calibration sweep on captured mobile-cam
      face crops).
- [ ] Real CASIA-FASD micro-mirror samples in `web/amispoof/samples/`
      (placeholder URLs at the moment).
- [ ] `ar_filter` analyzer (no ONNX model trained yet).
- [x] **Subdomain migration** — `amispoof.fivucsas.com` live (2026-05-17, PRs #44–#47). Old URL 301s; canonical/og:url/JSON-LD all on new domain; sitemap.xml + robots.txt deployed; og:image (1200×630) rendered; v0.3.0 GitHub release tagged.
- [ ] **Stand-alone `amispoof.com` domain** (currently inherits authority from `fivucsas.com`; only worth registering if we ever spin amispoof out as an independent product line).
- [ ] iBeta PAD-Level-1 browser-bundle submission package — currently scoped to the Python pipeline; re-scope to target the browser bundle as the production system.

## Next planned — tooling & observability (2026-05-17 backlog)

These do not change the algorithm; they add the operational layer that turns a working tool into a measurable product. Roughly ranked by ROI / effort:

### Tier 1 — Privacy-first analytics across the constellation (~1 hour total)

| Surface | Tool | Why |
|---|---|---|
| `amispoof.fivucsas.com` | GoatCounter (cloud) OR Plausible OR self-hosted Umami | Measure visits, source, country, device class — without cookies or banners. ~3 KB script, GDPR/CCPA-clean by construction (no PII collected). The "no upload, no GPU" ethos of amispoof should extend to its analytics |
| Same on: fivucsas.com landing, rollingcatsoftware.com, ahmetabdullah.gultek.in, demo.fivucsas.com, all rollingcatsoftware.com sub-projects | Same | One-line script per site |

**Not recommended:** Google Analytics 4 — requires cookie banner (KVKK), 90 KB script, leaks to Google. Wrong for an open-source liveness tester whose marketing point is "no upload, no server."

### Tier 2 — JS error tracking on the camera-heavy surfaces

| Surface | Tool | Why |
|---|---|---|
| `amispoof.fivucsas.com`, `app.fivucsas.com`, `verify.fivucsas.com` | Sentry (free 5k errors/mo) OR self-hosted GlitchTip | Real users hit MediaStreamTrack failures, MediaPipe wasm load errors, AudioWorklet glitches that we'd never see otherwise. The free Sentry tier covers single-digit-thousand-users/month easily |

### Tier 3 — Cloudflare in front of TurkTicaret + Hostinger (~30 min)

DNS migration: TurkTicaret → Cloudflare nameservers (Cloudflare imports existing records automatically). Buys global CDN, free privacy-friendly Web Analytics, basic DDoS protection, modern TLS ciphers, and a free fallback HTTPS layer.

### Tier 4 — Discoverability + governance (defer / opt-in)

- **GitHub Sponsors badge** on the `spoof-detector` repo — one toggle, no downside
- **Lighthouse CI** on the `spoof-detector` repo — PageSpeed regression guard
- **`web-vitals` reporting** from amispoof → GoatCounter custom events
- **Microsoft Clarity** — heatmaps + session recording; SKIP for amispoof (privacy ethos conflict), consider for app.fivucsas.com only
- **Google Ads / AdSense** — SKIP for amispoof (technical audience, ad-block-heavy); revisit only when FIVUCSAS commercial has a paying-customer funnel and a retargeting story to test against

### What this section deliberately does NOT include

Items the page itself should *not* grow toward:
- A native mobile app (browser is the deployment target — see §10 of the paper)
- A server-side inference fallback (defeats the entire architecture)
- Closed-source proprietary analyzers (every analyzer ships under MIT)

## Priority Levels
- P0: Must fix before next test (broken/blocking)
- P1: Must complete for paper submission
- P2: Should complete for enterprise quality
- P3: Nice to have / future work

---

## P0 — Critical Fixes

- [x] Fullscreen content scaling (frame resize to screen resolution)
- [x] rPPG FPS measurement (measure actual FPS instead of assuming 30)
- [x] Blink analyzer caching — FaceLandmarker.detect() now runs once per
      frame (was once per face). Linear speedup with face count: 3 faces
      ≈ 3.0x, 5 faces ≈ 4.9x; closed by `perf/blink-cache-and-ear-calibration`
      2026-05-11.
- [ ] Session verdict threshold tuning — phone-photo-only session still borders LIVE/SPOOF at 55%
- [x] EAR threshold calibration — EAR_THRESHOLD 0.20→0.18, REOPEN
      0.22→0.23, MIN_OPEN_BETWEEN 6→12 frames. On a simulated 60 s
      session with 1 blink / 3.5 s the detector now lands at 17/min
      (target 15-20); pinned by `tests/unit/analyzers/test_blink_calibration.py`.
      Closed 2026-05-11.

## P1 — Paper Requirements

### Detection Quality
- [ ] Validate rPPG on real vs screen (measure actual BPM on live face, confirm 0 on screen)
- [ ] Cross-validate blink on screen spoofs (verify 0 blinks detected on photo/video)
- [ ] Identity consistency checker — face embedding comparison across session
- [ ] Active challenge mode — ask user to turn head / blink on command / respond to color flash
- [ ] Collect OULU-NPU benchmark results for comparison table

### AR Filter Detection (Novel Contribution)
- [ ] Collect 500+ AR filter samples (50 per filter type: Snapchat, Instagram, TikTok, FaceApp, OBS)
- [ ] Collect 500+ real face baseline samples
- [ ] Train MobileNetV3-Small classifier
- [ ] Validate cross-filter generalization
- [ ] ONNX export and integration

### Evaluation
- [ ] Run test_protocol.py with all 5 scenarios, 100+ samples total
- [ ] Compute APCER/BPCER/ACER via evaluate.py
- [ ] ROC curves per attack type
- [ ] Ablation study: contribution of each analyzer
- [ ] Session duration vs accuracy curve (5s, 10s, 30s, 1min, 5min)

### Paper Writing
- [ ] Abstract (250 words)
- [ ] Introduction + related work
- [ ] Method: session architecture + calibrated fusion + peak-sensitive verdict
- [ ] Experiments section with all metrics
- [ ] Results tables and figures
- [ ] Discussion + limitations
- [ ] Submit to BIOSIG 2026 or IJCB 2026

## P2 — Enterprise Quality

### Performance
- [ ] GPU acceleration — install onnxruntime-gpu for MiniFASNet + future AR model
- [x] Blink analyzer frame caching (avoid running FaceLandmarker N times for N faces) — shipped 2026-05-11
- [ ] Disable anti-correlated analyzers (moire, texture) — save ~10ms/frame
- [ ] Profile and optimize to sustain 30+ FPS with all analyzers

### Robustness
- [ ] Low-light testing and compensation
- [ ] Multiple face policy (proctoring: only 1 face allowed)
- [ ] Face occlusion handling (glasses, mask worn for health)
- [ ] Camera resolution auto-detection and adaptation

### Session Engine
- [ ] Identity verification via face embedding comparison across session
- [ ] Attention/gaze tracking (is the person looking at the screen?)
- [ ] Anomaly detection for behavioral patterns (long absence, repeated identity switches)
- [ ] Session pause/resume support
- [ ] Configurable session duration limits

### Integration
- [ ] REST API wrapper for amispoof.com backend
- [ ] WebSocket streaming mode for proctoring
- [ ] FIVUCSAS biometric-processor integration (feature-flagged ARFilterLivenessDetector)
- [ ] Docker container packaging
- [ ] CI/CD pipeline

## P1.5 — Next Features (User Ideas 2026-05-02)

### Grid-Based Environment Analyzer
- Divide frame into NxM grid (e.g., 8x6 cells)
- In proctoring, camera is fixed — background cells should be static
- If background cells show motion → something suspicious (screen held up, person swap)
- Per-cell analysis: abnormal reflections, screen glare, sudden brightness spikes
- Background consistency over session → detect virtual backgrounds
- Implementation: store per-cell mean/std over time, flag anomalous cells

### Pixel-Level Screen Forensics
- **Camera reflection detection**: webcam lens creates a circular reflection on screens — detect this circle shape in the face region via Hough circle transform
- **Sub-pixel pattern analysis**: LCD/OLED have RGB sub-pixel arrangements (PenTile, stripe) visible at close range — Fourier analysis detects the periodic sub-pixel grid
- **FPS mismatch / temporal aliasing**: screen refresh (60Hz) vs camera capture (30fps) creates rolling shutter artifacts, "pixel ants", and temporal beating patterns — detectable via frame-to-frame pixel difference frequency analysis
- **Color order analysis**: screens emit pure RGB primaries; real skin reflects continuous spectrum — spectral analysis of color channel ratios can detect discrete vs continuous color sources
- **Screen uniformity artifacts**: backlight bleed, IPS glow, OLED black crush — these create characteristic brightness gradients not found in natural scenes
- **Reflection completeness**: real environments have complex multi-source reflections; screens show uniform single-source reflections (the backlight)
- **Green/white block detection**: compression artifacts, screen burn-in, dead pixels — statistical outliers in pixel value distribution within face region

### Behavioral Signal Processing (Heartbeat-Style Analysis)
- Treat blink timing, head rotation angles, landmark positions as TIME SERIES
- Apply signal processing (FFT, autocorrelation, wavelet) to detect:
  - Natural blink rhythm (~0.3Hz, semi-periodic with jitter)
  - Micro-tremor frequency (8-12Hz natural oscillation)
  - Head movement frequency spectrum (natural = low-freq drift + micro-tremor)
  - Video replay: unnaturally PERFECT periodicity (looping) or fixed frequency
  - Photo: ZERO signal (flat line — no variation at all)
- Anomaly detection: establish per-person baseline, flag deviations
  - Person's normal blink rate changes suddenly → might have swapped
  - Head movement pattern changes → different person or switched to video
- This is essentially treating behavioral biometrics as a signal processing problem
- Paper angle: "Behavioral frequency analysis for continuous identity verification"

## P3 — Future Work / Research Extensions

### New Attack Detection
- [ ] Deepfake injection via virtual webcam (OBS Virtual Camera, DeepFaceLive)
- [ ] Active illumination challenge (random color flash, measure skin response)
- [ ] 3D mask detection via rPPG (mask blocks pulse)
- [ ] Voice-face consistency check (speaker verification during session)
- [ ] Audio environment analysis (detect pre-recorded audio playback)

### amispoof.com
- [ ] Static frontend (Vite + vanilla JS)
- [ ] 5-second quick test mode
- [ ] Consent framework (KVKK/GDPR)
- [ ] Ephemeral storage (30-min auto-delete)
- [ ] "Donate to research" opt-in
- [ ] Public API for third-party integration
- [ ] Domain registration and DNS setup

### Dataset & Benchmarks
- [ ] Public labeled dataset release (with consent)
- [ ] Cross-dataset evaluation (OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof)
- [ ] Demographic fairness analysis
- [ ] Adversarial robustness testing

### Proctoring Features
- [ ] Browser lockdown integration (detect tab switches, screen sharing)
- [ ] Multi-camera support (front + side)
- [ ] Exam session management (start/pause/end with timestamps)
- [ ] Incident review interface for proctors
- [ ] Webhook/callback on incident detection
- [ ] LTI integration for LMS platforms (Moodle, Canvas)

---

## Self-Critique & Known Weaknesses

### Architecture
- Blink analyzer downloads 15MB model on first run — should bundle or lazy-load with user prompt
- Session engine has no face identity verification (different person could swap in)
- No active challenges — all detection is passive
- Pipeline runs ALL analyzers every frame even if some are warming up and returning 50

### Accuracy
- MiniFASNet is the ONLY proven per-frame discriminator (+94.7 gap)
- Texture and moire are ANTI-CORRELATED (score spoofs higher than real) — suppressed but still waste compute
- Phone-screen close-up (no visible bezel) fools all analyzers except MiniFASNet
- 60% accuracy on STATIC_SCREEN scenario in test protocol — needs temporal signals (blink/rPPG) to improve
- ~~Blink rate of 38/min is double normal — EAR threshold likely too sensitive~~
  Fixed 2026-05-11 (EAR 0.18, REOPEN 0.23, MIN_OPEN 12); simulated rate now 17/min.

### Paper Gaps
- No public benchmark comparison yet
- Small test set (43 labeled samples from one person, one webcam, one environment)
- AR filter model not yet trained (need data collection)
- No cross-dataset evaluation
- Single-person testing — need multi-demographic validation

### Performance
- No GPU utilization (GTX 1650 available but unused)
- ~~FaceLandmarker runs once per face per frame (~15ms each)~~
  Fixed 2026-05-11: now runs once per frame regardless of face count.
- Total pipeline ~25-35ms (30 FPS) but could be 15ms with optimization

---

## Proposed Paper Title

**"Beyond Single Frames: Session-Based Face Anti-Spoofing with Calibrated Multi-Analyzer Fusion"**

Alternative:
**"Session-Level Face Presentation Attack Detection: A Multi-Signal Temporal Approach for Proctoring and Identity Verification"**
