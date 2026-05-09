# 1. Introduction

Face authentication has become one of the most widely deployed biometric modalities — present in smartphone unlock, payment authorization, banking onboarding, online proctoring, and electronic identity verification. Each of these contexts assumes that the face presented to the camera belongs to a *live human being physically present in front of the device* — an assumption that is routinely violated by a spectrum of presentation attacks (PAs), from a printed photograph held up to the lens to a sophisticated deepfake injection via a virtual webcam.

Despite a decade of academic FAS research and four widely-cited benchmarks (OULU-NPU, SiW, CASIA-SURF, CelebA-Spoof), three structural gaps persist between the published methods and the systems shipping in production today.

**Gap 1 — Per-frame bias.** Every public FAS benchmark reports per-frame accuracy. A face is presented, a single frame is classified live or attack, and the metric is recorded. But in real deployments, an attacker has minutes — sometimes hours — of camera time. They can hold a print attack steady for two seconds, then briefly reveal their real face for one frame, then swap back. Per-frame metrics report 67% accuracy and call it a day; the attacker keeps going.

**Gap 2 — Single-modality models.** Most published methods are *image* methods — a CNN, transformer, or hand-crafted descriptor on a single frame. Aysenur Akar (Marmara University, FIVUCSAS R&D) and other recent work has shown that *temporal* signals — blink rate, micro-tremor, remote photo-plethysmography (rPPG), screen flicker — carry information that no single-frame method can recover. Yet temporal and image methods are typically reported in isolation. We argue that the right architecture is a hybrid: a fast single-frame discriminator handles the obvious print/replay attacks, and a battery of temporal analyzers handles the harder ones (deepfake injection, replay-loop, AR filter).

**Gap 3 — Anti-correlated signals.** Two of the five textbook anti-spoof signals (Laplacian texture variance and Gabor-moire) are widely believed to differentiate real face frames from printed or replayed faces. We measured them, and on our calibration set they discriminate the *wrong way round*: spoofs scored higher than reals. The reason is mundane and instructive — a high-resolution screen replay actually has more uniform texture than a low-light real face. Yet the literature continues to use both as positive evidence for liveness. We reweight them to near-zero in our fuser; this single change improves ACER by 4.1 percentage points on our internal set.

## Contributions

This paper makes three contributions:

1. **A session-based hybrid PAD engine** combining one per-frame discriminator (MiniFASNet) with seven temporal analyzers, fused via a calibrated multi-class voter and aggregated by a peak-sensitive session verdict. We report ISO/IEC 30107-3 metrics on OULU-NPU (P1–P4), SiW, CASIA-SURF, and CelebA-Spoof, and demonstrate that the hybrid pipeline outperforms either modality alone.
2. **Empirical anti-correlation finding** for Laplacian-texture and Gabor-moire on real-world capture data, with calibrated re-weighting recipe.
3. **Open-source, fully reproducible release**: spoof-detector v0.2.1 — a single repository consolidating every algorithm developed across the FIVUCSAS programme (390 R&D files from seven contributors), a 28-file mirror of the production-deployed code path, an ISO 30107-3 metrics module, and a benchmark harness with adapters for all four standard datasets. MIT-licensed.

The remainder of the paper is organised as follows. §2 surveys related FAS work. §3 defines the 7-category taxonomy that drives our fusion. §4 presents the system architecture: per-frame analyzers, temporal analyzers, multi-class fusion, and the session engine. §5 details the calibration methodology and reports the anti-correlation finding. §6 describes the benchmark harness and protocols. §7 reports the headline results. §8 ablates each design choice. §9 discusses limitations and open problems. §10 concludes.
