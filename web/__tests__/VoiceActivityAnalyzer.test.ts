import { describe, expect, it } from "vitest";
import { VoiceActivityAnalyzer } from "../src/infrastructure/analyzers/VoiceActivityAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";
import type { AudioCaptureLike } from "../src/infrastructure/audio/AudioCapture";

class StubAudio implements AudioCaptureLike {
  isActive = true;
  rmsHz = 50;
  constructor(private samples: Float32Array) {}
  async start(): Promise<void> {}
  async stop(): Promise<void> {}
  getRecentRms(durationSec: number): Float32Array {
    const n = Math.min(
      this.samples.length,
      Math.ceil(durationSec * this.rmsHz),
    );
    return this.samples.subarray(this.samples.length - n);
  }
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(0, 0, 100, 100),
  confidence: 0.9,
};

describe("VoiceActivityAnalyzer", () => {
  it("returns 50 when audio inactive", () => {
    const audio = new StubAudio(new Float32Array(25));
    audio.isActive = false;
    const a = new VoiceActivityAnalyzer({ audio });
    const r = a.analyze(null, face);
    expect(r.score).toBe(50);
    expect(r.details.error).toBe("audio_inactive");
  });

  it("scores 0 on silent audio", () => {
    const audio = new StubAudio(new Float32Array(25)); // all zeros
    const a = new VoiceActivityAnalyzer({ audio, rmsThreshold: 0.01 });
    const r = a.analyze(null, face);
    expect(r.score).toBe(0);
  });

  it("scores 100 on continuous voice", () => {
    const samples = new Float32Array(25);
    for (let i = 0; i < samples.length; i++) samples[i] = 0.1;
    const audio = new StubAudio(samples);
    const a = new VoiceActivityAnalyzer({ audio, rmsThreshold: 0.01 });
    const r = a.analyze(null, face);
    expect(r.score).toBe(100);
  });

  it("scores mid-range on intermittent voice", () => {
    const samples = new Float32Array(20);
    // Half the samples above threshold.
    for (let i = 0; i < samples.length; i++) samples[i] = i % 2 ? 0.05 : 0;
    const audio = new StubAudio(samples);
    const a = new VoiceActivityAnalyzer({ audio, rmsThreshold: 0.01 });
    const r = a.analyze(null, face);
    expect(r.score).toBeGreaterThan(30);
    expect(r.score).toBeLessThan(70);
  });
});
