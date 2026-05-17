// VoiceActivityAnalyzer — Phase D3 (opt-in).
//
// Computes a per-frame voice-activity score from the AudioCapture
// rolling RMS buffer. No model — just energy-threshold detection.
// Useful as a "is anyone talking" signal that, combined with the
// AudioMouthSyncAnalyzer below, catches silent video-replay attacks
// (replay has no audio → low VAD even when the subject is "speaking"
// in the video).
//
// Per frame:
//   * Sample the last 0.5 s of RMS values
//   * voice_fraction = fraction of frames where rms > rmsThreshold
//   * score = 100 × voice_fraction
// Score 0 = silence; ~50 = intermittent speech; 100 = continuous voice.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { AudioCaptureLike } from "../audio/AudioCapture";

export interface VoiceActivityAnalyzerOptions {
  audio: AudioCaptureLike;
  /** Window length over which voice fraction is computed. Default 0.5 s. */
  windowSec?: number;
  /** RMS threshold above which a frame counts as voice. Default 0.01. */
  rmsThreshold?: number;
}

export class VoiceActivityAnalyzer implements IFaceAnalyzer {
  readonly name = "voice_activity";

  private readonly audio: AudioCaptureLike;
  private readonly windowSec: number;
  private readonly rmsThreshold: number;

  constructor(options: VoiceActivityAnalyzerOptions) {
    this.audio = options.audio;
    this.windowSec = options.windowSec ?? 0.5;
    this.rmsThreshold = options.rmsThreshold ?? 0.01;
  }

  analyze(_faceCrop: ImageData | null, _face: FaceROI): AnalyzerResult {
    const start = performance.now();
    if (!this.audio.isActive) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "audio_inactive" },
        performance.now() - start,
      );
    }
    const rms = this.audio.getRecentRms(this.windowSec);
    if (rms.length === 0) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_audio_samples" },
        performance.now() - start,
      );
    }
    let voiceFrames = 0;
    let sum = 0;
    let peak = 0;
    for (let i = 0; i < rms.length; i++) {
      sum += rms[i];
      if (rms[i] > peak) peak = rms[i];
      if (rms[i] > this.rmsThreshold) voiceFrames += 1;
    }
    const fraction = voiceFrames / rms.length;
    return makeAnalyzerResult(
      this.name,
      Math.max(0, Math.min(100, fraction * 100)),
      {
        voice_fraction: round(fraction, 3),
        rms_mean: round(sum / rms.length, 4),
        rms_peak: round(peak, 4),
        samples: rms.length,
        rms_hz: round(this.audio.rmsHz, 1),
      },
      performance.now() - start,
    );
  }

  reset(): void {
    /* stateless beyond AudioCapture */
  }
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
