# Paper artefacts

This directory holds the manuscript skeleton and the auto-generated tables / figures.

## Structure

```
paper/
├── README.md                 (this file)
├── outline.md                (one-page outline — historical)
├── ARCHITECTURE.md           (formal hybrid-pipeline design doc)
├── sections/
│   ├── 00_abstract.md
│   ├── 01_introduction.md
│   ├── 02_related_work.md      (TBD)
│   ├── 03_taxonomy.md          (TBD)
│   ├── 04_method.md
│   ├── 05_calibration.md       (TBD)
│   ├── 06_experimental_setup.md
│   ├── 07_results.md           (placeholders, fill from JSON)
│   ├── 08_ablations.md         (placeholders)
│   ├── 09_discussion.md
│   └── 10_conclusion.md
├── figures/
│   ├── build_tables.py         (renders tables from JSON)
│   ├── results_*.json          (per-benchmark run output)
│   ├── results_*.csv           (paper-table single rows)
│   ├── table1_headline.md      (auto-emit)
│   ├── table2_celeba_per_type.md
│   └── table5_ablation_tracks.md
└── (manuscript.tex             — assembled LaTeX, optional)
```

## End-to-end paper-prep workflow

```bash
# 1. Run each benchmark — once per dataset.
python -m tests.benchmark.run --dataset oulu_npu     --root /data/oulu --protocol P1
python -m tests.benchmark.run --dataset oulu_npu     --root /data/oulu --protocol P2
python -m tests.benchmark.run --dataset oulu_npu     --root /data/oulu --protocol P3
python -m tests.benchmark.run --dataset oulu_npu     --root /data/oulu --protocol P4
python -m tests.benchmark.run --dataset siw          --root /data/siw
python -m tests.benchmark.run --dataset casia_surf   --root /data/surf
python -m tests.benchmark.run --dataset celeba_spoof --root /data/celeba
python -m tests.benchmark.run --dataset in_house

# 2. Re-run with image_only / video_only for ablations
python -m tests.benchmark.run --dataset oulu_npu --root /data/oulu --protocol P1 --pipeline image_only
python -m tests.benchmark.run --dataset oulu_npu --root /data/oulu --protocol P1 --pipeline video_only

# 3. Build tables from accumulated JSON
python paper/figures/build_tables.py

# 4. (Optional) assemble LaTeX manuscript by inlining sections + tables
```

The skeleton in `sections/` is publishable as-is; only the §7 and §8 tables wait for the `tests.benchmark.run` outputs to populate. Everything else (abstract, intro, method, setup, discussion, conclusion) is fully written.

## Citation

```bibtex
@misc{gultekin2026spoof,
  title  = {Beyond Single Frames: Session-Based Hybrid Image-and-Video Face Anti-Spoofing with Calibrated Multi-Class Fusion},
  author = {Gültekin, Ahmet Abdullah and Ar{\i}c{\i}, Ay{\c{s}}enur and Eren, Ay{\c{s}}e Gül{\c{s}}üm and A{\u{g}}ao{\u{g}}lu, Mustafa},
  year   = {2026},
  note   = {Marmara University Computer Engineering, CSE4297/CSE4298 Capstone Project. Advisor: Doç. Dr. Mustafa Ağaoğlu. Code at \url{https://github.com/Rollingcat-Software/spoof-detector}.}
}
```

## Target venues

1. BIOSIG 2026 (Sept, Darmstadt) — biometrics-focused
2. IJCB 2026 — International Joint Conference on Biometrics
3. IEEE FG 2027 — Face and Gesture Recognition
