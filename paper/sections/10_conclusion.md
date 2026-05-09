# 10. Conclusion

We presented a session-based hybrid image-and-video presentation-attack detection engine, calibrated against an internal 43-sample test set and evaluated against four public benchmarks (OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof) under the ISO/IEC 30107-3 metric set. Three findings stand out:

1. **Hybrid > image-only ≥ video-only** on every protocol — the two tracks address structurally different attack classes, and the calibrated multi-class fuser learns to weight them appropriately.
2. **Two textbook anti-spoof signals (Laplacian-texture, Gabor-moire) are anti-correlated on real-world data.** Re-weighting them to ≈ 0.1 in the fuser produced the largest single-step ACER improvement during calibration.
3. **Peak-sensitive session aggregation** prevents the spoof-burst dilution that pure-mean aggregation suffers in mixed-content sessions — the property that makes the engine deployment-ready for proctoring use cases.

The full algorithm corpus (390 R&D files from seven contributors, the 28-file production deployment mirror, the productized `src/` library, the four-dataset benchmark harness, and the ISO 30107-3 metrics module) is released under MIT at `github.com/Rollingcat-Software/spoof-detector` to support direct reproducibility. The paper's tables are emitted automatically by `paper/figures/build_tables.py` from the per-protocol JSON artefacts produced by `tests/benchmark/run.py`, so any reviewer can reproduce every number with one command per dataset.

Future work (beyond the open problems in §9.5) includes graduating the AR-filter MobileNetV3-Small classifier from the research tree to `src/`, retuning rPPG with a notch-filter at the local AC frequency to recover its expected discrimination, and a depth-aware extension that consumes CASIA-SURF's depth modality in the fuser.

## Acknowledgements

This work was undertaken at the Marmara University Computer Engineering Department as part of the FIVUCSAS multi-tenant biometric authentication platform. We thank the FIVUCSAS contributors @Aysenur15 (rPPG, screen-replay defence, blink, hybrid liveness, MRZ pipeline), Ayşe Gülsüm Eren (rPPG and screen-replay co-authoring), and the participants in the in-house calibration study for their consented contributions. The work is funded internally; no external grants were received.
