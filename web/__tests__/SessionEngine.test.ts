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

/**
 * A spoof frame (pReal=0.05) where `dominant` is the strict-max non-real
 * category, so checkSpoofIncident categorizes the incident as `dominant`.
 */
function buildSpoofFrame(
  frameId: number,
  dominant: SpoofCategory,
  blinks: number,
): FrameAnalysis {
  const face_id = 0;
  const face: FaceROI = {
    face_id,
    bbox: new BBox(100, 100, 540, 540),
    confidence: 0.99,
    landmarks: makeUniformLandmarks(),
  };
  const analyzers: Record<string, AnalyzerResult> = {
    blink: makeAnalyzerResult("blink", 80, { blinks }),
    minifasnet: makeAnalyzerResult("minifasnet", 5, {}),
  };
  const probs: Record<SpoofCategory, number> = {
    [SpoofCategory.REAL]: 0.05,
    [SpoofCategory.STATIC_IMAGE]: 0.05,
    [SpoofCategory.VIDEO_REPLAY]: 0.05,
    [SpoofCategory.MASK_3D]: 0.05,
    [SpoofCategory.HEAVY_MAKEUP]: 0.05,
    [SpoofCategory.AR_FILTER]: 0.05,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.05,
  };
  probs[dominant] = 0.65;
  const cls = classificationFromProbabilities(face_id, probs, analyzers);
  return {
    frame_id: frameId,
    faces: [face],
    classifications: { [face_id]: cls },
    frame_signals: {},
    total_ms: 5,
  };
}

/**
 * A frame carrying texture + screen_replay analyzer results so the
 * texture-collapse veto can be exercised. pReal stays high so the veto is the
 * ONLY spoof signal (isolates its typing). `textureScore` < 25 collapses;
 * `skinScore` selects the mode (≈0 photo / ≥30 video / 8-30 ambiguous).
 */
function buildScreenFrame(
  frameId: number,
  textureScore: number,
  skinScore: number,
  blinks: number,
  // Motion-gated typing inputs (2026-06-01). `eyeMotion` is the eyelid-
  // blendshape variance (split across left/right blink-symmetry std); `rigidVar`
  // is the raw landmark-variance (rigid-motion proxy). Defaults model a frozen,
  // rigid-still face (no eyelid motion, no rigid motion).
  eyeMotion = 0,
  rigidVar = 0,
): FrameAnalysis {
  const face_id = 0;
  const face: FaceROI = {
    face_id,
    bbox: new BBox(100, 100, 540, 540),
    confidence: 0.99,
    landmarks: makeUniformLandmarks(),
  };
  const analyzers: Record<string, AnalyzerResult> = {
    texture: makeAnalyzerResult("texture", 60, { texture_score: textureScore }),
    screen_replay: makeAnalyzerResult("screen_replay", 50, { skin_score: skinScore }),
    blink: makeAnalyzerResult("blink", 80, { blinks }),
    blink_symmetry: makeAnalyzerResult("blink_symmetry", 80, {
      std_left: eyeMotion / 2,
      std_right: eyeMotion / 2,
    }),
    landmark_variance: makeAnalyzerResult("landmark_variance", 50, {
      overall_var: rigidVar,
    }),
    minifasnet: makeAnalyzerResult("minifasnet", 95, {}),
  };
  const pReal = 0.9;
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
  const cls = classificationFromProbabilities(face_id, probs, analyzers);
  return {
    frame_id: frameId,
    faces: [face],
    classifications: { [face_id]: cls },
    frame_signals: {},
    total_ms: 5,
  };
}

/** A frame where face detection found nothing (camera dark / occluded). */
function buildMissingFrame(frameId: number): FrameAnalysis {
  return {
    frame_id: frameId,
    faces: [],
    classifications: {},
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

  it("dominant_threat follows the incident category, not the fooled fusion", () => {
    // 2026-06-01 bug: a phone replay read real≈0.82 in the fuser with the
    // residual ranking static_image > video_replay, yet 30 texture-collapse
    // VIDEO_REPLAY incidents fired. The threat label must follow the incidents
    // that actually flip the verdict, not the fusion's fooled residual.
    const engine = new SessionEngine();
    engine.start();
    for (let i = 1; i <= 200; i++) {
      tick(33);
      const isSpike = i === 60 || i === 130 || i === 195;
      engine.ingest(
        isSpike
          ? buildSpoofFrame(i, SpoofCategory.VIDEO_REPLAY, Math.floor(i / 30))
          : buildFrame({
              frameId: i,
              pReal: 0.9,
              blinks: Math.floor(i / 30),
              miniFasNetScore: 95,
              drift: (i % 10) * 0.5,
            }),
      );
    }
    const v = engine.getVerdict();
    expect(v.is_live).toBe(false);
    expect(v.incidents.length).toBeGreaterThanOrEqual(3);
    expect(v.dominant_threat).toBe(SpoofCategory.VIDEO_REPLAY);
  });

  it("texture-collapse types a static photo (skin~0) as STATIC_IMAGE", () => {
    // Photo path: flat texture + skin below a real face's range = a photo.
    const engine = new SessionEngine();
    engine.start();
    for (let i = 1; i <= 400; i++) {
      tick(33);
      engine.ingest(buildScreenFrame(i, 8, 2, 0));
    }
    const v = engine.getVerdict();
    expect(v.is_live).toBe(false);
    expect(v.dominant_threat).toBe(SpoofCategory.STATIC_IMAGE);
  });

  it("texture-collapse types a screen replay (real eyelid motion, still) as VIDEO_REPLAY", () => {
    // 2026-06-01: typing is by NON-rigid eyelid motion on rigid-still frames,
    // not by skin. A video replay held steady shows genuine eyelid-blendshape
    // variance (0.4 > 0.2 threshold) on still frames (rigidVar 5 < 30) → video.
    const engine = new SessionEngine();
    engine.start();
    for (let i = 1; i <= 400; i++) {
      tick(33);
      engine.ingest(buildScreenFrame(i, 8, 60, 5, /*eyeMotion*/ 0.4, /*rigidVar*/ 5));
    }
    const v = engine.getVerdict();
    expect(v.is_live).toBe(false);
    expect(v.dominant_threat).toBe(SpoofCategory.VIDEO_REPLAY);
  });

  it("a screen photo held still (high skin, frozen eyes) types STATIC_IMAGE not VIDEO", () => {
    // 2026-06-01: a still photo ON A SCREEN reads high skin (~60, screen glow)
    // — the old skin-mode logic mistyped it video_replay. With motion-gated
    // typing it has no eyelid motion on still frames → correctly STATIC_IMAGE.
    const engine = new SessionEngine();
    engine.start();
    for (let i = 1; i <= 400; i++) {
      tick(33);
      engine.ingest(buildScreenFrame(i, 8, 60, 0, /*eyeMotion*/ 0, /*rigidVar*/ 3));
    }
    const v = engine.getVerdict();
    expect(v.is_live).toBe(false);
    expect(v.dominant_threat).toBe(SpoofCategory.STATIC_IMAGE);
  });

  it("a hard-waved photo (jitter fakes eye-motion) stays STATIC_IMAGE (rigid frames gated out)", () => {
    // 2026-06-01: violent waving manufactures fake eyelid-blendshape variance
    // (0.4) AND huge rigid motion (rigidVar 120 > 30). Because the eyelid signal
    // is read ONLY on rigid-still frames, every frame is gated out → too few
    // still frames → defaults STATIC_IMAGE. The attack self-defeats.
    const engine = new SessionEngine();
    engine.start();
    for (let i = 1; i <= 400; i++) {
      tick(33);
      engine.ingest(
        buildScreenFrame(i, 8, 60, Math.floor(i / 20), /*eyeMotion*/ 0.4, /*rigidVar*/ 120),
      );
    }
    const v = engine.getVerdict();
    expect(v.is_live).toBe(false);
    expect(v.dominant_threat).toBe(SpoofCategory.STATIC_IMAGE);
  });

  it("texture-collapse in the ambiguous skin band [8,30) does NOT fire (spares real face at distance)", () => {
    const engine = new SessionEngine();
    engine.start();
    for (let i = 1; i <= 400; i++) {
      tick(33);
      engine.ingest(buildScreenFrame(i, 8, 20, 5));
    }
    const texInc = engine
      .getVerdict()
      .incidents.filter((inc) => /Texture collapse/.test(inc.description));
    expect(texInc.length).toBe(0);
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

  it("face-missing incident is throttled, not raised every frame (1,326-flood regression)", () => {
    // Live 2026-05-25: a pitch-dark room (0 faces) at ~50 fps raised a "Face
    // missing" incident EVERY frame — ~1,326 in 33s — instantly tripping the
    // >=3-incident override and flagging a real visitor SPOOF. The throttle
    // must cap it to roughly one incident per FACE_MISSING_ALERT_SEC.
    const engine = new SessionEngine();
    engine.start();
    // ~6.6s of a clean, present, live face (past warmup + the 5s floor).
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
    // ~10s of NO face at ~50 fps — the dark-room scenario.
    let fid = 201;
    for (let i = 0; i < 500; i++, fid++) {
      tick(20);
      engine.ingest(buildMissingFrame(fid));
    }
    const v = engine.getVerdict();
    const faceMissing = v.incidents.filter((i) =>
      (i.description ?? "").startsWith("Face missing"),
    );
    expect(faceMissing.length).toBeGreaterThanOrEqual(1); // still alerts
    expect(faceMissing.length).toBeLessThanOrEqual(5); // but throttled, not ~350
  });

  // ---- no-blink alert is fps-aware --------------------------------------

  describe("checkNoBlink fps-aware threshold", () => {
    // Helper: pick out only the static-image / no-blink incidents
    // (other paths — motion-naturalness, MFN-instability — can fire on
    // synthetic frames and aren't what this section is testing).
    function noBlinkIncidents(verdict: ReturnType<typeof firstNoBlinkVerdict>) {
      return verdict.incidents.filter((i) =>
        (i.description ?? "").startsWith("No blink for"),
      );
    }
    // Avoid TS "verdict implicitly any" — name the verdict type by example.
    function firstNoBlinkVerdict() {
      const e = new SessionEngine();
      return e.getVerdict();
    }

    it("at 30 fps, fires the no-blink incident at the documented 15s mark", () => {
      const engine = new SessionEngine();
      engine.start();
      // 30 fps × 16s = 480 frames, never a blink → expect a no-blink
      // incident after 15s. `drift` keeps motion-naturalness from firing.
      for (let i = 1; i <= 480; i++) {
        tick(33);
        engine.ingest(
          buildFrame({
            frameId: i,
            pReal: 0.9,
            blinks: 0,
            miniFasNetScore: 95,
            drift: (i % 10) * 0.5,
          }),
        );
      }
      const v = engine.getVerdict();
      const nb = noBlinkIncidents(v);
      expect(nb.length).toBeGreaterThanOrEqual(1);
      expect(nb[0].timestamp).toBeGreaterThanOrEqual(15);
    });

    it("at 7 fps, holds the alert past 15s — a slow-camera user who misses the first blink isn't flagged", () => {
      // Replay of the 2026-05-17 Chrome-mobile trace: 6.7 fps, the user
      // blinks 22 times but BlinkAnalyzer misses the first one for ~16s.
      // At 7 fps the threshold stretches to 15 × (15/7) ≈ 32s, so the
      // window from 15s → 16s (first detected blink) no longer fires.
      const engine = new SessionEngine();
      engine.start();
      for (let i = 1; i <= 112; i++) {
        tick(143); // ~7 fps × 16s ≈ 112 frames
        engine.ingest(
          buildFrame({
            frameId: i,
            pReal: 0.9,
            blinks: 0,
            miniFasNetScore: 95,
            drift: (i % 10) * 0.5,
          }),
        );
      }
      const v = engine.getVerdict();
      expect(noBlinkIncidents(v).length).toBe(0);
    });

    it("at 7 fps, still flags a true static-image attack (no blinks for >60s)", () => {
      // Sanity check that stretching the threshold doesn't disable the
      // detector — a real photo attack with no blinks for 70 seconds at
      // 7 fps must still raise the no-blink incident.
      const engine = new SessionEngine();
      engine.start();
      for (let i = 1; i <= 490; i++) {
        tick(143); // ~7 fps × 70s ≈ 490 frames
        engine.ingest(
          buildFrame({
            frameId: i,
            pReal: 0.9,
            blinks: 0,
            miniFasNetScore: 95,
            drift: (i % 10) * 0.5,
          }),
        );
      }
      const v = engine.getVerdict();
      expect(noBlinkIncidents(v).length).toBeGreaterThanOrEqual(1);
    });
  });
});
