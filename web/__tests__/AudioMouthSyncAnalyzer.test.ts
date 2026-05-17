import { describe, expect, it } from "vitest";
import { AudioMouthSyncAnalyzer } from "../src/infrastructure/analyzers/AudioMouthSyncAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";
import type { AudioCaptureLike } from "../src/infrastructure/audio/AudioCapture";

class StubAudio implements AudioCaptureLike {
  isActive = true;
  rmsHz = 30;
  constructor(public series: () => Float32Array) {}
  async start(): Promise<void> {}
  async stop(): Promise<void> {}
  getRecentRms(_durationSec: number): Float32Array {
    return this.series();
  }
}

function makeFace(jaw: number): FaceROI {
  return {
    face_id: 0,
    bbox: new BBox(0, 0, 100, 100),
    confidence: 0.9,
    blendshapes: new Map<string, number>([["jawOpen", jaw]]),
  };
}

describe("AudioMouthSyncAnalyzer", () => {
  it("returns 50 when audio inactive", () => {
    const audio = new StubAudio(() => new Float32Array(60));
    audio.isActive = false;
    const a = new AudioMouthSyncAnalyzer({ audio });
    const r = a.analyze(null, makeFace(0.1));
    expect(r.score).toBe(50);
  });

  it("returns 50 during warmup", () => {
    const audio = new StubAudio(() => new Float32Array(60));
    const a = new AudioMouthSyncAnalyzer({ audio, warmupFrames: 30 });
    for (let i = 0; i < 20; i++) {
      const r = a.analyze(null, makeFace(0.1 + 0.05 * (i % 2)));
      expect(r.score).toBe(50);
    }
  });

  it("scores 50 (silence) when both audio + mouth are flat", () => {
    const audio = new StubAudio(() => new Float32Array(60));
    const a = new AudioMouthSyncAnalyzer({ audio, warmupFrames: 30 });
    for (let i = 0; i < 60; i++) a.analyze(null, makeFace(0.05));
    const r = a.analyze(null, makeFace(0.05));
    expect(r.score).toBe(50);
    expect(r.details.silence).toBe(true);
  });

  it("scores high when audio and jawOpen are synchronized (real speech)", () => {
    const series = new Float32Array(60);
    let jawValues: number[] = [];
    const audio = new StubAudio(() => series);
    const a = new AudioMouthSyncAnalyzer({
      audio,
      warmupFrames: 30,
      historyLen: 60,
    });
    for (let i = 0; i < 60; i++) {
      const v = Math.abs(Math.sin(i / 4));
      series[i] = 0.05 + 0.05 * v;
      jawValues.push(0.1 + 0.4 * v);
      a.analyze(null, makeFace(jawValues[i]));
    }
    const r = a.analyze(null, makeFace(jawValues[59]));
    expect(r.details.corr).toBeGreaterThan(0.7);
    expect(r.score).toBeGreaterThan(60);
  });

  it("scores low when audio and jawOpen are anti-correlated (desync replay)", () => {
    const series = new Float32Array(60);
    let jawValues: number[] = [];
    const audio = new StubAudio(() => series);
    const a = new AudioMouthSyncAnalyzer({
      audio,
      warmupFrames: 30,
      historyLen: 60,
    });
    for (let i = 0; i < 60; i++) {
      const v = Math.abs(Math.sin(i / 4));
      series[i] = 0.05 + 0.05 * v;
      jawValues.push(0.1 + 0.4 * (1 - v)); // anti-correlated
      a.analyze(null, makeFace(jawValues[i]));
    }
    const r = a.analyze(null, makeFace(jawValues[59]));
    expect(r.score).toBeLessThan(20);
  });

  it("reset() drops per-face state", () => {
    const audio = new StubAudio(() => new Float32Array(60));
    const a = new AudioMouthSyncAnalyzer({ audio, warmupFrames: 5 });
    for (let i = 0; i < 6; i++) a.analyze(null, makeFace(0.1 + (i % 2) * 0.1));
    a.reset();
    const r = a.analyze(null, makeFace(0.1));
    expect(r.score).toBe(50);
    expect(r.details.warming).toBe(true);
  });
});
