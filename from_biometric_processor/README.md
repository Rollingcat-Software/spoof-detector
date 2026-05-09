# Algorithms also deployed in `biometric-processor`

The Python modules in this directory are **verbatim copies** of files that already ship in `biometric-processor` production (`bio` main as of SHA `9d17359e18186abc317f4fe4b903ab8e82626297`, 2026-05-09). They are not imported by `src/`; they are mirrored here so that the spoof-detector repo is the single authoritative consumption point for every liveness/anti-spoof asset, regardless of which container actually runs them today.

This is the **complete** mirror — every production `app/**/*.py` that touches liveness, anti-spoof, gesture, flash, or proctoring is here.

## Algorithms (`app/application/services/` + `app/infrastructure/ml/liveness/`)

| File in this directory               | Source path in `biometric-processor`                                  |
|--------------------------------------|------------------------------------------------------------------------|
| `cutout_anomaly_detector.py`         | `app/application/services/cutout_anomaly_detector.py`                  |
| `device_boundary_detector.py`        | `app/application/services/device_boundary_detector.py`                 |
| `device_spoof_risk_evaluator.py`     | `app/application/services/device_spoof_risk_evaluator.py`              |
| `flash_spoof_analyzer.py`            | `app/application/services/flash_spoof_analyzer.py`                     |
| `light_challenge_service.py`         | `app/application/services/light_challenge_service.py`                  |
| `live_session_baseline_calibrator.py`| `app/application/services/live_session_baseline_calibrator.py`        |
| `active_liveness_detector.py`        | `app/infrastructure/ml/liveness/active_liveness_detector.py`           |
| `combined_liveness_detector.py`      | `app/infrastructure/ml/liveness/combined_liveness_detector.py`         |
| `enhanced_liveness_detector.py`      | `app/infrastructure/ml/liveness/enhanced_liveness_detector.py`         |
| `hybrid_liveness_detector.py`        | `app/infrastructure/ml/liveness/hybrid_liveness_detector.py`           |
| `moire_pattern_analysis.py`          | `app/infrastructure/ml/liveness/moire_pattern_analysis.py`             |
| `optimized_texture_liveness.py`      | `app/infrastructure/ml/liveness/optimized_texture_liveness.py`         |
| `rppg_analyzer.py`                   | `app/infrastructure/ml/liveness/rppg_analyzer.py`                      |
| `screen_replay_anti_spoof.py`        | `app/infrastructure/ml/liveness/screen_replay_anti_spoof.py`           |
| `stub_liveness_detector.py`          | `app/infrastructure/ml/liveness/stub_liveness_detector.py`             |
| `temporal_consistency_analyzer.py`   | `app/infrastructure/ml/liveness/temporal_consistency_analyzer.py`      |
| `texture_liveness_detector.py`       | `app/infrastructure/ml/liveness/texture_liveness_detector.py`          |
| `threshold_calibrator.py`            | `app/infrastructure/ml/liveness/threshold_calibrator.py`               |
| `uniface_liveness_detector.py`       | `app/infrastructure/ml/liveness/uniface_liveness_detector.py`          |

## Active-liveness orchestration (`app/application/services/`)

| File in this directory                       | Source path in `biometric-processor`                                            |
|----------------------------------------------|----------------------------------------------------------------------------------|
| `active_liveness_manager.py`                 | `app/application/services/active_liveness_manager.py`                            |
| `active_gesture_liveness_manager.py`         | `app/application/services/active_gesture_liveness_manager.py`                    |
| `active_liveness_token_service.py`           | `app/application/services/active_liveness_token_service.py`                      |
| `background_active_reaction_evaluator.py`    | `app/application/services/background_active_reaction_evaluator.py`               |

## Domain (entities + interfaces + exceptions + factory)

| File in this directory                | Source path in `biometric-processor`                                  |
|---------------------------------------|------------------------------------------------------------------------|
| `liveness_detector_interface.py`      | `app/domain/interfaces/liveness_detector.py` (renamed to avoid collision) |
| `liveness_result.py`                  | `app/domain/entities/liveness_result.py`                               |
| `liveness_report.py`                  | `app/domain/entities/liveness_report.py`                               |
| `liveness_errors.py`                  | `app/domain/exceptions/liveness_errors.py`                             |
| `liveness_factory.py`                 | `app/infrastructure/ml/factories/liveness_factory.py`                  |

**Total: 28 files mirrored.**

## Provenance and authorship

- The 4 originally-mirrored algorithms (`cutout`, `device_spoof_risk`, `light_challenge`, `screen_replay`) originate in Aysenur's (`@Aysenur15`) FIVUCSAS R&D work.
- Several of the others (rPPG, moire, hybrid, uniface, optimized texture, etc.) also evolved through Aysenur's branches before landing in bio production. See `../research/aysenur/<branch>/` for the development context and `../ATTRIBUTION.md` for credits.
- The active-liveness manager + gesture stack + threshold calibration were primarily authored by **Ahmet Abdullah Gultekin** (sole author of the spoof-detector standalone — see `../research/spoof-detector-history/PRE_EXTRACTION.md`).

## Sync policy

- These files **must not be edited here.** Edit them in `biometric-processor` and re-run the consolidation refresh to update this directory.
- The header comment in each file records the source SHA so drift is auditable with `diff <(tail -n +5 from_biometric_processor/<file>) <(git -C biometric-processor show <SHA>:<src>)`.
- This directory is **not** part of the `src/` import path, so there is no productization risk from keeping mirrors here.
- To refresh after a bio change, copy the new content and bump the `# Source SHA:` line. Use `git rev-parse HEAD` in bio for the new SHA.

## Why mirror at all?

The user's instruction (2026-05-09):

> "Have everything in spoof detector as a complete product. Copy and paste
> all works about it into spoof detector repo and make sure all work existed
> already in spoof detector. Do not remove from others, just copy paste and
> ensure all works are existed in spoof detector repo."

The first pass landed only 4 files (the 4 most directly Aysenur-authored). This pass completes the goal by mirroring every prod-deployed liveness/anti-spoof module — so the spoof-detector repo can serve as the single authoritative reference even when one container actually runs the code.
