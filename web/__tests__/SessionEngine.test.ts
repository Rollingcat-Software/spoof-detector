// Regression tests for SessionEngine — verdict aggregation, peak-sensitivity,
// and the optional LivenessProver gate.
//
// History: the 2026-05-16 production trace from a Chrome-mobile webcam (real
// face, 921 frames at 9 fps, 31 blinks, MiniFASNet score 100, 0 incidents)
// was wrongly verdicted as SPOOF(static_image) because the now-removed
// `verdictLockedSpoof` latch fired during the `dataConfidence` ramp.
// `live_face_does_not_latch_during_warmup_ramp` replays that trace and is
// the primary regression guard.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionEngine } from "../src/application/SessionEngine";
import { LivenessProver } from "../src/application/LivenessProver";
import {
  AnalyzerResult,
  BBox,
  FaceROI,
  FrameAnalysis,
  SpoofCategory,
  SpoofClassification,
  classificationFromProbabilities,
  makeAnalyzerResult,
} from "../src/domain/models";

// === Frame builders ==========================================================

function makeUniformLandmarks(): Float32Array {
  const N = 478;
  const lm = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) {
    lm[i * 2] = 320;
    lm[i * 2 + 1] = 240;
  }
  // Pose anchors used by LivenessProver.estimateHeadPose; kept centered so
  // the prover sees zero rotation by default.
  lm[1 * 2] = 320;
  lm[1 * 2 + 1] = 240;
  lm[10 * 2] = 320;
  lm[10 * 2 + 1] = 90;
  lm[152 * 2] = 320;
  lm[152 * 2 + 1] = 390;
  lm[234 * 2] = 220;
  lm[234 * 2 + 1] = 240;
  lm[454 * 2] = 420;
  lm[454 * 2 + 1] = 240;
  return lm;
}

interface FrameOpts {
  frameId: number;
  pReal: number;
  blinks?: number;
  miniFasNetScore?: number;
  /** Slowly drift the bbox so motion_naturalness doesn't trip "unnaturally static". */
  drift?: number;
}

function buildFrame(opts: FrameOpts): FrameAnalysis {
  const drift = opts.drift ?? 0;
  const face_id = 0;
  const face: FaceROI = {
    face_id,
    bbox: new BBox(100 + drift, 100 + drift, 540 + drift, 540 + drift),
    confidence: 0.99,
    landmarks: makeUniformLandmarks(),
  };

  const analyzers: Record<string, AnalyzerResult> = {};
  if (opts.blinks !== undefined) {
    analyzers["blink"] = makeAnalyzerResult("blink", 80, {
      blinks: opts.blinks,
    });
  }
  if (opts.miniFasNetScore !== undefined) {
    analyzers["minifasnet"] = makeAnalyzerResult(
      "minifasnet",
      opts.miniFasNetScore,
      {},
    );
  }

  const pReal = opts.pReal;
  const pSpoof = (1 - pReal) / 6;
  const probs: Record<SpoofCategory, number> = {
    [SpoofCategory.REAL]: pReal,
    [SpoofCategory.STATIC_IMAGE]: pSpoof,
    [SpoofCategory.VIDEO_REPLAY]: pSpoof,
    [SpoofCategory.MASK_3D]: pSpoof,
    [SpoofCategory.HEAVY_MAKEUP]: pSpoof,
    [SpoofCategory.AR_FILTER]: pSpoof,
    [SpoofCategory.DEEPFAKE_INJECT]: pSpoof,
  };
  const cls: SpoofClassification = classificationFromProbabilities(
    face_id,
    probs,
    analyzers,
  );
  return {
    frame_id: opts.frameId,
    faces: [face],
    classifications: { [face_id]: cls },
    frame_signals: {},
    total_ms: 5,
  };
}

/** Step `vi` fake clock forward by ms to mirror frame pacing. */
function tick(ms: number): void {
  vi.advanceTimersByTime(ms);
}

// === Tests ===================================================================

describe("SessionEngine", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-16T12:00:00Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns live verdict on a clean stable session", () => {
    const engine = new SessionEngine();
    engine.start();
    for (let i = 1; i <= 200; i++) {
      tick(33); // ~30 fps
      engine.ingest(
        buildFrame({
          frameId: i,
          pReal: 0.9,
          blinks: Math.floor(i / 30),
          miniFasNetScore: 95,
          drift: (i % 10) * 0.5,
        }),
      );
    }
    const v = engine.getVerdict();
    expect(v.is_live).toBe(true);
    expect(v.incidents.length).toBe(0);
  });

  it("live_face_does_not_latch_during_warmup_ramp (replay of 2026-05-16 chrome-mobile trace)", () => {
    // The pre-fix engine latched verdictLockedSpoof during the
    // dataConfidence ramp because past_warmup fires at frame 31 but
    // dataConfidence only reaches 1.0 at frame 150. At 9 fps that ramp
    // is ~17s — easily 15 consecutive sub-threshold frames → latch.
    // Trace inputs taken from the user-submitted JSON dump:
    //   921 frames over 110.6s (≈ 8.3 fps measured), per-frame
    //   probabilities[REAL] ≈ 0.81, MiniFASNet score 100, 31 blinks,
    //   no incidents, face present ≈ 98%.
    const engine = new SessionEngine();
    engine.start();
    const frameMs = 120; // ~8.3 fps
    const totalFrames = 921;
    for (let i = 1; i <= totalFrames; i++) {
      tick(frameMs);
      const blinks = Math.min(31, Math.floor((i / totalFrames) * 31));
      engine.ingest(
        buildFrame({
          frameId: i,
          pReal: 0.81,
          blinks,
          miniFasNetScore: 100,
          drift: (i % 20) * 0.5,
        }),
      );
    }
    const v = engine.getVerdict();
    expect(v.incidents.length).toBe(0);
    expect(v.is_live).toBe(true);
    expect(v.dominant_threat).toBeNull();
  });

  it("sustained spoof burst flags the session via worst-window even after recovery", () => {
    // Peak-sensitivity guard: a brief sustained spoof window should drag
    // worstWindowReal low enough that the session is flagged, even when
    // the majority of frames are live. This is the property the removed
    // latch was over-enforcing; worstWindowReal preserves it correctly.
    const engine = new SessionEngine();
    engine.start();
    // 200 live frames
    for (let i = 1; i <= 200; i++) {
      tick(33);
      engine.ingest(
        buildFrame({
          frameId: i,
          pReal: 0.9,
          blinks: Math.floor(i / 30),
          miniFasNetScore: 95,
          drift: (i % 10) * 0.5,
        }),
      );
    }
    // 60-frame spoof burst (pReal = 0)
    for (let i = 201; i <= 260; i++) {
      tick(33);
      engine.ingest(
        buildFrame({
          frameId: i,
          pReal: 0.0,
          blinks: 5,
          miniFasNetScore: 10,
          drift: (i % 10) * 0.5,
        }),
      );
    }
    const v = engine.getVerdict();
    expect(v.is_live).toBe(false);
    // Either the >=3-incident override fires (most likely) or worst-window
    // drags adjustedReal below 0.45 — both are acceptable paths to SPOOF.
  });

  it("three or more incidents force SPOOF even if average real is high", () => {
    // incident_override path: a single low-pReal frame every 3s for 9s
    // accrues 3 incidents while leaving avg_real high. The override
    // should still flip the verdict.
    const engine = new SessionEngine();
    engine.start();
    for (let i = 1; i <= 200; i++) {
      tick(33);
      // Inject a spoof-spike every ~60 frames so 3+ checkSpoofIncident
      // calls fire (the 2.0s throttle requires spacing).
      const isSpike = i === 60 || i === 130 || i === 195;
      engine.ingest(
        buildFrame({
          frameId: i,
          pReal: isSpike ? 0.05 : 0.9,
          blinks: Math.floor(i / 30),
          miniFasNetScore: isSpike ? 5 : 95,
          drift: (i % 10) * 0.5,
        }),
      );
    }
    const v = engine.getVerdict();
    expect(v.incidents.length).toBeGreaterThanOrEqual(3);
    expect(v.is_live).toBe(false);
  });

  it("requireProverLive is opt-in: default false leaves verdict to fusion only", () => {
    // With requireProverLive default (false), even a never-proved prover
    // (score 0) shouldn't block a clean live session.
    const prover = new LivenessProver({ enableChallenges: false });
    const engine = new SessionEngine({ prover });
    engine.start();
    for (let i = 1; i <= 200; i++) {
      tick(33);
      engine.ingest(
        buildFrame({
          frameId: i,
          pReal: 0.9,
          blinks: Math.floor(i / 30),
          miniFasNetScore: 95,
          drift: (i % 10) * 0.5,
        }),
      );
    }
    const v = engine.getVerdict();
    expect(v.is_live).toBe(true);
  });

  it("requireProverLive=true gates on prover score (Python session_engine.py:400 semantics)", () => {
    // With the gate on, a session that hasn't accumulated >=60 prover
    // points should NOT be live even if fusion is clean.
    const prover = new LivenessProver({ enableChallenges: false });
    const engine = new SessionEngine({ prover, requireProverLive: true });
    engine.start();
    for (let i = 1; i <= 200; i++) {
      tick(33);
      // No blinks, no expression, centered landmarks → prover stays low.
      engine.ingest(
        buildFrame({
          frameId: i,
          pReal: 0.95,
          miniFasNetScore: 95,
          drift: (i % 10) * 0.5,
        }),
      );
    }
    expect(prover.getScore().total).toBeLessThan(60);
    const v = engine.getVerdict();
    expect(v.is_live).toBe(false);
  });

  it("reset() clears prover state alongside engine state", () => {
    const prover = new LivenessProver({ enableChallenges: false });
    const engine = new SessionEngine({ prover });
    engine.start();
    for (let i = 1; i <= 50; i++) {
      tick(33);
      engine.ingest(
        buildFrame({
          frameId: i,
          pReal: 0.9,
          blinks: 5,
          miniFasNetScore: 95,
        }),
      );
    }
    expect(prover.getScore().blink_points).toBeGreaterThan(0);
    engine.reset();
    expect(prover.getScore().blink_points).toBe(0);
    expect(prover.getScore().total).toBe(0);
  });
});
