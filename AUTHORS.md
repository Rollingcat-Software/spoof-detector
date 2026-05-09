# Authors

`spoof-detector` is maintained by [Rollingcat Software](https://github.com/Rollingcat-Software) and was originally extracted from the
[FIVUCSAS](https://github.com/Rollingcat-Software/FIVUCSAS) biometric-authentication
platform's R&D track.

## Contributors

The following people have authored substantive parts of this project. New
contributions are welcome — open a pull request.

### Core extraction & maintenance (v0.1.0 → present)

- **Ahmet Abdullah Gultekin** — [@ahabgu](https://github.com/ahabgu)
  - Initial extraction from FIVUCSAS (2026-05-09).
  - Session engine + analyzer wiring + ISO 30107-3 calibration.
  - Continuous integration with FIVUCSAS biometric-processor.

### Algorithms ported in v0.2.0 (2026-05-09)

The following modules were authored by **Aysenur** — [@Aysenur15](https://github.com/Aysenur15) — as part of her R&D work on the FIVUCSAS `working_spoof_detection` branch, and ported here so they can be reused, evaluated and cited independently of the FIVUCSAS service:

- `src/gates/face_usability.py` — pre-liveness face-usability gate (no-face / occlusion / low-quality streaks).
- `src/gates/critical_region_visibility.py` — pixel + landmark-based per-region occlusion analyser (eyes / nose / mouth / lower-face).
- `src/gates/illumination.py` — preview-only illumination quality gate (under/over-exposure, shadow asymmetry).
- `src/fusion/hybrid_evaluator.py` — calibrated fusion of pretrained MiniFASNet + heuristic device-replay signals (moire / flash / device-replay / rPPG sanity).
- `src/pipeline/assembler.py` — single duck-typed adapter that runs the gates, fusion evaluator, and a caller-supplied device-spoof evaluator and returns one structured advisory verdict (allow / review / block — never enforced by the assembler itself).

The accompanying unit-test suite (46 tests across `tests/unit/gates/`, `tests/unit/fusion/`, `tests/unit/pipeline/`) is also Aysenur's work, ported with import paths rewritten to match the new namespace.

If you build on or evaluate against any of these modules in a publication, please credit Aysenur.

### Research consolidation (2026-05-09)

In addition to the production-ported v0.2.0 modules above, the consolidation
pass on 2026-05-09 brought the **complete body of FIVUCSAS liveness/anti-spoof
work** under this repo as a read-only research tree. See `ROADMAP.md` and
`research/README.md`.

#### Aysenur — [@Aysenur15](https://github.com/Aysenur15)

Areas (full source under `research/aysenur/`):

- **Anti-spoof pipeline (working_spoof_detection)** — flagship branch, 63 source files. Flash challenge, Gabor moire, hybrid fusion, screen-replay defence, device-spoof risk evaluator, cutout-anomaly detector, pre-fusion gates, ISO 30107-3 calibration harness.
- **liveness_capture** — rPPG, screen-replay defence, MRZ pipeline (co-authored with Ayşe Gülsüm Eren).
- **Spoof-Detection** — early integration branch.
- **feat-anti-spoof-pipeline-local** — clean squash of the pipeline for review.
- **liveness-cascade-frr-reduction** / **liveness-p0-frr-reduction** / **liveness-p3-frr-reduction** — False-Reject reduction work, cascaded thresholds.

Four of Aysenur's modules also ship in `biometric-processor` production today
and are mirrored under `from_biometric_processor/` for unified consumption:
`cutout_anomaly_detector.py`, `device_spoof_risk_evaluator.py`,
`light_challenge_service.py`, `screen_replay_anti_spoof.py`.

#### Ayşe Gülsüm Eren

- **Email**: aysegulsumeren@gmail.com
- **GitHub**: TBD (please open an issue to add)
- **Style**: Conventional Commits

Co-author on Aysenur's `liveness_capture` and `working_spoof_detection`
branches. Specific contributions interleaved per commit; see
`research/ayse-gulsum-eren/README.md` for the `git log --author` recipe to
enumerate her commits in `biometric-processor`.

#### Ahmet Abdullah Gultekin — [@ahabgu](https://github.com/ahabgu)

Lead, original `spoof-detector` author. The session engine, ISO 30107-3
calibration, and the assembler pipeline now under `src/` were extracted from
`fivucsas/practice-and-test/spoof-detector/` on 2026-05-09 (commit `43d23a9`).
See `research/ahmet-original-spoof-detector/README.md`.

## Reporting an attribution gap

If you believe your work has been included in this repo without proper attribution, please open an issue at https://github.com/Rollingcat-Software/spoof-detector/issues — we will fix it as a priority.
