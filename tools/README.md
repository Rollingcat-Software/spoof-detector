# Tools

Operator-facing scripts. Two cohorts live here:

## v0.2.0 baseline (Ahmet)

| Script                     | Purpose                                                    |
|----------------------------|------------------------------------------------------------|
| `analyze_captures.py`      | Offline batch analysis over a captures dir.                |
| `benchmark.py`             | Latency / throughput micro-benchmarks for the pipeline.    |
| `collect_ar_dataset.py`    | AR-filter dataset collection helper.                       |
| `diagnose.py`              | Single-frame diagnostic dump.                              |
| `evaluate.py`              | APCER / BPCER / ACER on a labelled set.                    |
| `label_tool.py`            | Tk labelling UI for collected captures.                    |
| `test_protocol.py`         | Run a labelled test scenario end-to-end.                   |
| `train_ar_detector.py`     | Train the AR-filter classifier (MobileNetV3-Small).        |

## Ported from Aysenur's `working_spoof_detection` (2026-05-09 consolidation)

| Script                       | Purpose                                                 |
|------------------------------|---------------------------------------------------------|
| `live_liveness_preview.py`   | OpenCV desktop tuner — large monolithic UI for inter-   |
|                              | active threshold sweeps. ~4.8k LOC. Ground-truth labour |
|                              | is the user, not a dataset.                             |
| `test_data_collector.py`     | Capture rig used to build the spoof-classifier corpus.  |
| `train_spoof_classifier.py`  | sklearn (Gradient-Boosted) training harness for the     |
|                              | non-MiniFASNet heuristic-stack classifier.              |
| `export_training_data.py`    | Convert raw captures into the training-CSV format       |
|                              | expected by `train_spoof_classifier.py`.                |

These ported tools currently import paths from the `app.*` namespace of
biometric-processor's working_spoof_detection branch; running them against
`spoof-detector/src/` requires path-rewrites — flagged as productization
work in `../ROADMAP.md`.
