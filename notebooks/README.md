# Separability notebook — `separability.ipynb`

Loads `amispoof-session-*.json` dumps from the amispoof demo's **↓ Report**
button (Phase F frame-log instrumentation onwards) and computes per-feature
LIVE-vs-REPLAY separability — the input for picking thresholds vs training a
classifier.

## Quickstart

```bash
# From spoof-detector/
python -m venv .venv && . .venv/bin/activate     # or .venv\Scripts\Activate.ps1 on Windows
pip install jupyter pandas numpy scikit-learn matplotlib seaborn xgboost
jupyter lab notebooks/separability.ipynb
```

## Inputs

Drop JSON dumps into one of (the notebook checks in this order):

1. `notebooks/data/`   ← preferred once the dataset grows
2. `docs/`             ← where the bootstrap dumps already live
3. `data/`             ← legacy

Each file's ground-truth class is read from `environment.capture_label`, which the
demo's capture-bar dropdown bakes in at save time. Old (schema v1, pre-Phase-F)
dumps that have no `frame_log` are still loaded as a single-row fallback so
you can sanity-check the pipeline on the existing two dumps.

## Outputs

Per-feature AUC + d′ ranking · top-8 distribution histograms · top-20
correlation heatmap · GroupKFold-CV AUC for Logistic Regression (and XGBoost
if installed). Group keys are the session filenames, so a model can't earn
AUC by memorising a single subject's idiosyncrasies.

## Reading the numbers

| AUC | Meaning |
|---|---|
| ≥ 0.95 | Single feature is a threshold candidate — ship it as a new rule, no ML needed |
| 0.85–0.95 | Combine via logistic regression; if LogReg lands here too, try XGBoost |
| 0.55–0.85 | Useful as part of an ensemble, not a standalone rule |
| < 0.55 | Noise — drop from fusion or down-weight |

LogReg group-CV AUC is the honest headline. If it plateaus, the answer is
almost always **more sessions / more class diversity**, not a fancier model —
small-N datasets with one subject and one room are the classic over-fitting
trap that fooled cross-dataset CASIA-FASD (paper AUC dropped 0.84 → 0.71).

See `docs/SPOOF_DETECTOR_BROWSER_READINESS.md` for the broader pipeline
context.
