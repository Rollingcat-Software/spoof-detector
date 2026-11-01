# Withdrawn in-house synthetic-attack results

The following figure artefacts were **removed as unreproducible** during the
2026-05-29 results-integrity review and are intentionally absent from the repo:

- `table1_headline.md`
- `results_in_house_replay_only_{hybrid,image_only,minifasnet_only}.{csv,json}`
- `results_in_house_replay_only_v2_{hybrid,image_only,minifasnet_only}.{csv,json}`
- `results_in_house_default_{hybrid,image_only,minifasnet_only}.{csv,json}`
- `results_in_house_clean_{hybrid,image_only,minifasnet_only}.{csv,json}`

## Why

These tables reported headline numbers — including the **0.00% ACER / "100%
accuracy"** cells — produced by resubstitution on a **synthetic same-source**
attack set (the in-house attacks are gamma/blur/moire re-renders of the same
bona-fide image), with the **decision threshold chosen on the test set**. That
is test-set leakage: the result does not survive a Dev→Test split and is not
reproducible. None of these tables were cited by the current paper sections.

## What is still reported (honest, retained)

- **Zero-shot public-dataset evaluation** (the paper headline): CASIA-FASD AUC
  0.945 / ACER 12.67% (N=2,408) and CelebA-Spoof AUC 0.782 / ACER 28.67%
  (N=2,611), both with stratified bootstrap 95% CIs — see §7.1.
- The in-house **replay sub-protocol (N=100)** and **full-set transparency block
  (N=325)** are retained because §7.2–§7.3 already frame them honestly as
  synthetic-attack validation with their methodological caveats; their ACER/EER
  use an EER-on-test operating point that is now explicitly opt-in in
  `src/metrics/standard.py` (`allow_test_set_threshold=True`).

See also `paper/sections/05_calibration.md` §5.4 (fuser weights are heuristic,
not swept) and the metrics threshold policy in
`src/metrics/standard.py::classification_report`.

## Also withdrawn (2026-11-01)

- `loso_cv.json`: the 20 leave-one-subject-out folds all report the identical AUC (0.98148) while
  reporting EER and ACER of 0, which is internally inconsistent. Not cited anywhere; do not use.
- `cnn_casia_fasd.json`, `cnn_casia_fasd_seed1.json`, `cnn_casia_fasd_seed7.json`,
  `cnn_casia_fasd_seed99.json`: the operating threshold is chosen on the test split, the reported
  best validation AUC is 1.0, and the source crops (`captures/*.npz`) are not in the repository, so
  the result cannot be reproduced. Not cited anywhere; do not use.

The files are kept for provenance only.
