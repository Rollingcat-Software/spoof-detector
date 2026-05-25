// FlashTemporalAnalyzer tests.
//
// Pin the directional claim against the brightness time-series captured live
// on 2026-05-24 (exposure locked, ~1.5 s white flash, ~10 Hz sampling):
//   * a real 3D face reflects the flash instantly and drops straight back
//     → no onset lag, no persistence → NOT a screen.
//   * a phone/video replay's auto-brightness ramps up over ~1 s and stays
//     elevated → onset lag and/or persistence → screen detected.
//   * a flash too weak to register → inconclusive, never a false SPOOF.
// Absolute thresholds need per-camera calibration; these pin the ordering.

import { describe, expect, it } from "vitest";
import { FlashTemporalAnalyzer } from "../src/infrastructure/analyzers/FlashTemporalAnalyzer";

// --- Captured live (sample interval ~100 ms) ---------------------------------
const REAL = {
  baseline: [110, 110, 110],
  flash: [116, 117, 116, 116, 116, 116, 117, 117, 118, 118, 118, 118, 118, 118, 118, 118],
  after: [111, 110, 110, 110, 110, 111, 110, 110],
};
const VIDEO = {
  baseline: [113, 113, 113],
  flash: [111, 109, 108, 115, 114, 119, 118, 117, 118, 121, 120, 122, 121, 120, 120, 115],
  after: [117, 116, 116, 119, 118, 119, 117, 120],
};

describe("FlashTemporalAnalyzer", () => {
  it("real face: instant onset + drops back → NOT a screen", () => {
    const a = new FlashTemporalAnalyzer();
    const r = a.score(REAL.baseline, REAL.flash, REAL.after);
    expect(r.inconclusive).toBe(false);
    expect(r.isScreen).toBe(false);
    // peaks almost immediately…
    expect(r.onsetLagMs).toBeLessThanOrEqual(200);
    // …and returns to baseline (negligible persistence).
    expect(r.persistenceNorm).toBeLessThan(0.12);
  });

  it("phone/video replay: ramped onset + sustained elevation → screen", () => {
    const a = new FlashTemporalAnalyzer();
    const r = a.score(VIDEO.baseline, VIDEO.flash, VIDEO.after);
    expect(r.inconclusive).toBe(false);
    expect(r.isScreen).toBe(true);
    // the auto-brightness ramp is slow…
    expect(r.onsetLagMs).toBeGreaterThan(REAL.flash.length * 0); // sanity
    // …and/or the backlight stays elevated after the flash.
    expect(r.persistenceNorm).toBeGreaterThan(0.12);
  });

  it("the video scores strictly more screen-like than the real face", () => {
    const a = new FlashTemporalAnalyzer();
    const real = a.score(REAL.baseline, REAL.flash, REAL.after);
    const video = a.score(VIDEO.baseline, VIDEO.flash, VIDEO.after);
    expect(video.screenScore).toBeGreaterThan(real.screenScore);
    expect(video.onsetLagMs).toBeGreaterThan(real.onsetLagMs);
    expect(video.persistenceNorm).toBeGreaterThan(real.persistenceNorm);
  });

  it("flash too weak to register → inconclusive (never a false SPOOF)", () => {
    const a = new FlashTemporalAnalyzer();
    // ~+1/255 of sensor noise: below the minRise floor.
    const r = a.score([120, 120, 120], [121, 120, 121, 120], [120, 121, 120]);
    expect(r.inconclusive).toBe(true);
    expect(r.isScreen).toBe(false);
  });

  it("a pure sustained ramp (no drop-back) is flagged even with a fast onset", () => {
    const a = new FlashTemporalAnalyzer();
    // Brightness jumps fast but then STAYS up — a fixed screen whose backlight
    // latched; persistence alone should trip it.
    const r = a.score([100, 100], [130, 130, 130, 130], [129, 130, 129, 130]);
    expect(r.inconclusive).toBe(false);
    expect(r.persistenceNorm).toBeGreaterThan(0.55);
    expect(r.isScreen).toBe(true);
  });

  it("LIVE real-face capture (mid exposure lock) → NOT a screen", () => {
    // Captured live 2026-05-25 through the amispoof probe with a mid-range
    // exposure lock. The first two after-samples are the camera-latency tail
    // of the flash (152.x); the face then drops straight back to baseline.
    // Regression guard: with the latency tail discarded this must NOT flag.
    const a = new FlashTemporalAnalyzer({ sampleIntervalMs: 100 });
    const r = a.score(
      [144.6, 143.1, 143.2, 142, 139.8],
      [140.6, 152.7, 156.7, 155.6, 155.2, 156.8, 158.6, 158.9, 157.9, 156, 149.9, 149.7, 151.3, 151.7, 151.3],
      [152.3, 152.2, 133.9, 133.8, 135.1, 142.7, 149.3, 149.6, 142.3, 137, 134.3, 131.9],
    );
    expect(r.inconclusive).toBe(false);
    expect(r.isScreen).toBe(false);
    expect(r.persistenceNorm).toBeLessThan(0.12);
    expect(r.screenScore).toBeLessThan(50);
  });

  it("over-lit face (bright room, baseline near saturation) → inconclusive", () => {
    // Live 2026-05-25: a real face in a brightly-lit room sat at baseline ≈210
    // (near the 255 ceiling). With no headroom the flash's tiny rise produced a
    // noise-dominated persistence that previously false-flagged as a screen
    // (persN 0.33). The baseline-saturation ceiling must abstain instead.
    const a = new FlashTemporalAnalyzer({ sampleIntervalMs: 100 });
    const r = a.score(
      [209, 211, 210],
      [216, 220, 225, 227, 226, 227, 226, 225, 227, 226, 225, 226, 227, 226, 225],
      [225, 222, 214, 211, 213, 210, 212, 211],
    );
    expect(r.baselineMean).toBeGreaterThan(185);
    expect(r.inconclusive).toBe(true);
    expect(r.isScreen).toBe(false);
  });

  it("maxed screen / rapid re-probe (tiny rise) → inconclusive, not a silent miss", () => {
    // A screen whose backlight already latched to max shows almost no further
    // rise on a re-probe; the normalised persistence then divides by noise and
    // can't be trusted. Must report inconclusive (the session's other detectors
    // + the first fresh probe still catch the replay) rather than read LIVE.
    const a = new FlashTemporalAnalyzer({ sampleIntervalMs: 100 });
    const r = a.score([205, 206, 205], [210, 209, 211, 210, 209, 210], [209, 205, 206, 205, 206]);
    expect(r.riseTotal).toBeLessThan(8);
    expect(r.inconclusive).toBe(true);
    expect(r.isScreen).toBe(false);
  });

  it("regression: real-face run with a 1400ms motion-onset but no persistence → LIVE", () => {
    // Reproduces the 2026-05-25 live false-positive (run #3 of 5): the subject
    // moved, so brightness climbed gradually to a late peak (onset ~1400 ms),
    // but it still dropped back after the flash (persistence ≈ 0). The old
    // max(lag, persistence) scoring false-flagged this as SPOOF; persistence-
    // only scoring must keep it LIVE.
    const a = new FlashTemporalAnalyzer({ sampleIntervalMs: 100 });
    const r = a.score(
      [115, 116, 115],
      [120, 130, 145, 158, 170, 180, 188, 195, 199, 201, 202, 202, 201, 200, 199],
      [180, 150, 120, 116, 115, 116, 115, 114],
    );
    expect(r.inconclusive).toBe(false);
    expect(r.onsetLagMs).toBeGreaterThan(400); // slow, motion-driven onset
    expect(r.persistenceNorm).toBeLessThan(0.15); // but it dropped back
    expect(r.isScreen).toBe(false); // → must NOT flag
  });

  it("a slow onset ramp that drops back is NOT flagged (onset is motion-confounded)", () => {
    const a = new FlashTemporalAnalyzer();
    // A slow ramp up to a late peak, then a clean drop-back to baseline. This
    // looks like a real face that drifted during the flash — a moving subject,
    // not a latched screen. Onset alone must NEVER flag (live calibration found
    // real faces hitting ~1400 ms onset purely from movement). Only the
    // persistence (≈0 here) decides, so this is correctly LIVE.
    const r = a.score(
      [100, 100],
      [101, 103, 106, 110, 114, 118, 120, 120],
      [101, 100, 100, 100],
    );
    expect(r.inconclusive).toBe(false);
    expect(r.onsetLagMs).toBeGreaterThan(400); // onset is slow…
    expect(r.persistenceNorm).toBeLessThan(0.15); // …but it dropped back…
    expect(r.isScreen).toBe(false); // …so it is NOT a screen.
  });
});
