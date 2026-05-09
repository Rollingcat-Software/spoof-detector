# 9. Discussion

## 9.1 What works

The empirical pattern across §7 confirms the architectural thesis. A strong per-frame discriminator handles the easy attacks (print, low-quality replay) and a panel of temporal analyzers handles the harder ones (high-quality replay, AR filter, deepfake injection). Neither track alone matches the hybrid; the hybrid does not regress beneath the better individual track on any protocol. The calibrated-fusion finding is portable: anyone re-running the calibration on their own data will discover the same anti-correlation pattern in Laplacian-texture and Gabor-moire, and the same recipe (re-weight to ≈ 0.1) recovers ACER without removing the signals from interpretability output.

## 9.2 What does not work, and why

**rPPG remains uncalibrated** in the headline pipeline (weight 0.0). Despite the literature showing rPPG should provide ≈ 30 percentage-point swings on screen-replay, our implementation needs a notch filter at the local AC frequency (50 Hz in Turkey, 60 Hz in North America) to suppress the dominant flicker harmonic that otherwise saturates the FFT band of interest. The fix is in the ROADMAP P0 list and is the single highest-value next change.

**Texture and moire stay in the pipeline at 0.1 weight** rather than being deleted. Two reasons: (1) operator-facing dashboards display per-analyzer scores so a human can audit *why* a session was flagged spoof, and removing the analyzers would remove a useful interpretability column; (2) on cross-dataset evaluation (§7.4) the anti-correlation does flip sign for very specific phone-screen attacks, so a small positive weight may turn out to be informative — at the cost of being correctly anti-correlated for the dominant attacks today, the analyzer is at least available to the fuser if calibration is re-run.

**3-D mask attacks are under-represented.** OULU-NPU does not include them; CASIA-SURF does include cut-out masks but those are 2-D. Only CelebA-Spoof's `face_mask` and `3d_mask` species cover this ground, and CelebA-Spoof is image-only — so our temporal analyzers cannot help. The temporal-only ablation (8.1) will therefore be poor on 3-D mask attacks; the hybrid pipeline relies on MiniFASNet alone for them.

## 9.3 Limitations

- **In-house calibration set is small** (43 samples, one institution, one camera). Cross-dataset numbers (§7.4) are the right metric for generalization, but a larger consented in-house set would tighten the calibration weights and surface analyzer interactions invisible to the current set.
- **Heavy-makeup category is poorly served by all five datasets.** The 7-category taxonomy lists it for completeness; we report APCER as 0 for it across every benchmark because no benchmark provides labelled heavy-makeup samples. The category is a known gap in the FAS literature, not specific to our work.
- **CPU-only deployment ceiling.** The hybrid pipeline sustains ≥ 30 FPS on CX43 CPU only by suppressing some analyzers (texture, moire) to a once-every-N-frames sampling. A GPU build would let every analyzer run on every frame and would likely tighten the high-quality replay numbers; we leave this as a deployment-config knob rather than a method change.
- **No depth modality.** CASIA-SURF includes depth and IR; we evaluate only RGB to keep parity with our pipeline. A depth-aware extension would likely close the gap on 3-D mask species; we list it as future work.

## 9.4 Ethical and legal considerations

The in-house set was collected with explicit informed consent under the Turkish KVKK (Art. 6(1)(a)) and EU GDPR (Art. 6(1)(a)) lawful bases. Subjects were briefed on data retention (deletion on request, automatic 30-day expiry for unconsented sessions), shared with no third party, and given the right to opt out at any time without affecting their study participation.

The system is intended as a *defensive* PAD layer for biometric authentication, not as an offensive surveillance tool. We do not train face-recognition or identity-classification models in this work; the analyzers consume face crops only for liveness purposes.

Open-sourcing the model weights, fusion calibration, and benchmark adapters under MIT carries a dual-use risk — the same recipe can guide an attacker on which analyzers to fool. We mitigate this by ensuring (a) the active-challenge layer (§4.5) is not part of the published default and remains an operator-only knob, and (b) the multi-class taxonomy in §3 calls out the categories where a determined attacker has visible runway (deepfake injection in particular). The defensive value of an open published baseline outweighs the marginal attacker uplift, in our judgement.

## 9.5 Open problems

Three open problems we encourage the community to attack:

1. **Calibration-data adaptation.** Our calibrated weights derive from a 43-sample set. The largest published FAS benchmarks have 1k–10k subjects. A clean adaptation procedure (transfer-learn calibration weights across datasets without re-collecting calibration data) would unlock the full benchmark suite for any new analyzer added to the fuser.
2. **Active-challenge fairness.** Active challenges (gesture, light, blink-on-command) help PAD numbers but disadvantage users with motor impairments, photophobia, or visual processing differences. A literature on "FAS accessibility" does not yet exist; we encourage it.
3. **Cross-modal fusion under partial-modality drop-out.** When a deployment loses a sensor (depth-camera fails, IR LED dies), the fuser should degrade gracefully. Our current fuser does not formally model modality drop-out; this is a clean optimization problem with operational impact.
