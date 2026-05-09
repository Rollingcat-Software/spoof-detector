"""Pipeline factories for the benchmark runner.

Each module exposes `score_sample(sample) -> float` returning a [0,1]
live-ness score. Three built-ins:

    image_only  — Ahmet's image-level pipeline:
                  MiniFASNet + face usability gates + multi-class fuser.
                  Single-frame; ignores temporal signals.

    video_only  — Aysenur's temporal pipeline:
                  blink (EAR), rPPG pulse, screen replay flicker FFT,
                  micro-tremor, screen-replay anti-spoof.

    hybrid      — Image + video fused:
                  Image-level Ahmet + video-level Aysenur, evidence-weighted.
                  This is the published method.
"""
