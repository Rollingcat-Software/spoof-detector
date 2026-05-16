// RppgAnalyzer tests.
// Pulse detection from a procedural 64×64 face crop where each frame's
// green-channel value is either constant (no pulse → low pulse score)
// or modulated at ~1.2 Hz / 72 BPM (live → non-zero pulse).

import { describe, expect, it } from "vitest";
import { RppgAnalyzer } from "../src/infrastructure/analyzers/RppgAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

/** Build a 32×32 ImageData with given green-channel intensity. */
function makeImageData(green: number): ImageData {
  // Polyfill ImageData for node — vitest "node" env doesn't ship it.
  const w = 32;
  const h = 32;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 128; // red, ignored
    data[i + 1] = green; // green — what the analyzer reads
    data[i + 2] = 128; // blue, ignored
    data[i + 3] = 255;
  }
  return { data, width: w, height: h, colorSpace: "srgb" } as ImageData;
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(0, 0, 32, 32),
  confidence: 0.9,
};

describe("RppgAnalyzer", () => {
  it("returns warmup below MIN_FRAMES", () => {
    const a = new RppgAnalyzer();
    const r = a.analyze(makeImageData(120), face);
    expect(r.score).toBe(50.0);
    expect(r.details.warmup).toBe(true);
    expect(r.details.frames).toBe(1);
    expect(r.details.need).toBe(60);
  });

  it("constant green → no pulse → low pulse score after enough data", () => {
    // historyLen=180 so frameCount > GOOD_FRAMES=150 triggers the
    // "no pulse with sufficient data = spoof" branch.
    const a = new RppgAnalyzer({ historyLen: 180, fps: 30 });
    let last = a.analyze(makeImageData(120), face);
    for (let i = 0; i < 200; i++) {
      last = a.analyze(makeImageData(120), face);
    }
    expect(last.details.warmup).toBeUndefined();
    // Constant green → all energy lands in DC (excluded from noise). After
    // detrending the residual is numerical noise; the peak in the pulse
    // band will be tiny compared to broadband leakage so SNR stays well
    // below 1.5 and we fall through to the "spoof" branch.
    const snr = last.details.snr as number;
    expect(snr).toBeLessThan(1.5);
    expect(last.score).toBeLessThanOrEqual(30);
    expect(last.details.bpm).toBeNull();
  });

  it("synthetic 1.2 Hz / 72 BPM pulse → non-zero pulse detection", () => {
    // fps=30, N=150 → bin width = 0.2 Hz; 1.2 Hz lands exactly on bin 6,
    // safely inside the (0.75, 4.0) Hz pulse band.
    const a = new RppgAnalyzer({ historyLen: 150, fps: 30 });
    let last = a.analyze(makeImageData(120), face);
    for (let i = 0; i < 200; i++) {
      // 1.2 Hz = 72 BPM, peak-to-peak amplitude 20 on top of a 120 baseline.
      const g = Math.round(120 + 10 * Math.sin((2 * Math.PI * 1.2 * i) / 30));
      last = a.analyze(makeImageData(g), face);
    }
    expect(last.details.warmup).toBeUndefined();
    const snr = last.details.snr as number;
    const bpm = last.details.bpm as number | null;
    // Strong, isolated tone → SNR well above 2.0.
    expect(snr).toBeGreaterThan(2.0);
    // bpm should land in a sensible window around 72.
    expect(bpm).not.toBeNull();
    expect(bpm as number).toBeGreaterThan(60);
    expect(bpm as number).toBeLessThan(85);
    // Pulse-detected ⇒ score above the ambiguous floor (30).
    expect(last.score).toBeGreaterThan(50);
  });

  it("getBpm exposes the per-face estimate after a detected pulse", () => {
    const a = new RppgAnalyzer({ historyLen: 150, fps: 30 });
    a.analyze(makeImageData(120), face);
    for (let i = 0; i < 200; i++) {
      const g = Math.round(120 + 10 * Math.sin((2 * Math.PI * 1.2 * i) / 30));
      a.analyze(makeImageData(g), face);
    }
    const bpm = a.getBpm(0);
    expect(bpm).not.toBeNull();
    expect(bpm as number).toBeGreaterThan(60);
    expect(bpm as number).toBeLessThan(85);
    // Unknown face → null.
    expect(a.getBpm(999)).toBeNull();
  });
});
