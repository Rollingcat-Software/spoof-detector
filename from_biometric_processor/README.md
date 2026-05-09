# Algorithms also deployed in `biometric-processor`

The Python modules in this directory are **verbatim copies** of files that
already ship in `biometric-processor` production (`bio` main as of
SHA `178d2fa231daf6c1d667e652a6d390add789cd34`, 2026-05-09). They are not
imported by `src/`; they are mirrored here so that the spoof-detector repo
is the single authoritative consumption point for every liveness/anti-spoof
asset, regardless of which container actually runs them today.

| File in this directory               | Source path in `biometric-processor`                                  |
|--------------------------------------|------------------------------------------------------------------------|
| `cutout_anomaly_detector.py`         | `app/application/services/cutout_anomaly_detector.py`                  |
| `device_spoof_risk_evaluator.py`     | `app/application/services/device_spoof_risk_evaluator.py`              |
| `light_challenge_service.py`         | `app/application/services/light_challenge_service.py`                  |
| `screen_replay_anti_spoof.py`        | `app/infrastructure/ml/liveness/screen_replay_anti_spoof.py`           |

All four modules originate in Aysenur's (`@Aysenur15`) FIVUCSAS R&D work; see
`../research/aysenur/working_spoof_detection/` for the development context and
`../ATTRIBUTION.md` for credits.

## Sync policy

- These files **must not be edited here.** Edit them in `biometric-processor`
  and re-run the consolidation script to refresh this directory.
- The header comment in each file records the source SHA so drift is auditable
  with `diff` against `git -C biometric-processor show <SHA>:<src>`.
- This directory is **not** part of the `src/` import path, so there is no
  productization risk from keeping mirrors here.

## Why mirror at all?

The user's instruction (2026-05-09):

> "Have everything in spoof detector as a complete product. Copy and paste
> all works about it into spoof detector repo and make sure all work existed
> already in spoof detector. Do not remove from others, just copy paste and
> ensure all works are existed in spoof detector repo."
