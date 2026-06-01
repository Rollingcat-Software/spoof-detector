# amispoof Capture Protocol (paper-grade dataset)

**Why this exists.** Our first 43 sessions span 13 builds and ~2–3 subjects, so
they can't support a credible claim. Build-filtering to the current build leaves
only ~10 edge-heavy sessions, on which `texture_score` AUC falls from 0.87 →
0.67 — i.e. our headline number was an artifact of mixed/older data. To make
*any* defensible separability or false-positive claim we need a **systematic,
single-build, multi-subject** dataset. This protocol defines it.

## Hard rules (every capture)

1. **Current build only.** Hard-refresh `amispoof.fivucsas.com`; the version
   pill must read the latest (`2026-06-01-threat-coop` or newer). `v3_check.py`
   prints `[CURRENT]` / `[STALE]` — anything STALE is discarded.
2. **Correct dropdown label** (not "notes"): `LIVE`, `REPLAY · phone close`,
   `REPLAY · phone arm's length`, `REPLAY · laptop screen`, `PRINT · paper photo`.
   A *static photo shown on a phone screen* = `REPLAY · phone close` (screen
   medium) — note `screen-static-photo` in notes so we can split it out.
3. **Subject initials in notes**, every session (e.g. `AAG`, `MK`). Without this
   we cannot do subject-level (GroupKFold) validation — the only validation that
   counts for PAD.
4. **Condition in notes**: `<initials> <dist> <light>` → e.g. `AAG 40cm lowlight`.
5. **≥ 20 s per session**, hold steady, blink naturally, let it reach a verdict.
6. Download each report (↓ Report) into `notebooks/data/`.

## The matrix to fill (target)

Per **subject** (aim for **5 subjects**), capture these 8 cells:

| Class | Distance | Lighting | Why this cell |
|---|---|---|---|
| LIVE | ~40 cm | bright | baseline true-accept |
| LIVE | ~40 cm | **low light** | **the current false-positive regime** |
| LIVE | ~15 cm | bright | distance control (decouple pose_3d) |
| REPLAY phone | ~15 cm | bright | baseline replay |
| REPLAY phone | ~40 cm | bright | far replay (smaller screen in frame) |
| REPLAY phone | ~15 cm | **low light** | self-lit screen in the dark |
| PRINT paper photo | ~25 cm | bright | the *real* static-image class |
| REPLAY laptop | ~40 cm | bright | bigger screen, different moire |

**5 subjects × 8 cells ≈ 40 clean current-build sessions** — a defensible N with
both within- and across-subject coverage.

**Priority if time-boxed:** the **LIVE @ 40 cm low-light** cell for every subject
(that's where V3 false-fires today), then the baseline LIVE + phone-replay pair.

## After capture

```bash
python notebooks/v3_check.py                       # spot-check newest (shows build tag)
python notebooks/quick_compare.py notebooks/data --current   # build-filtered AUC + subject table
```

Then, with ≥ ~30 clean sessions over ≥ 4 subjects:
1. Recalibrate the texture/skin AND-gate at a chosen ROC point (candidate:
   `texture < 18` AND `skin ≥ 40`; verify it spares the LIVE-low-light cell and
   still catches replays — and check the low-skin "5 cm" replays).
2. Report **subject-level GroupKFold** AUC, not frame-level.
3. Binary LIVE/SPOOF only — do **not** claim spoof-*type* sub-classification.
