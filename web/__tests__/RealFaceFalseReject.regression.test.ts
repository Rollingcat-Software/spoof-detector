// Regression: real people must read LIVE across camera/lighting variation
// (warm camera, dim/evening light, head-turns) WITHOUT re-opening print/replay
// spoof coverage. Grounded in the team's own measured per-session numbers
// documented in SessionEngine.ts (texture_score / skin_score tables).
//
// The bug (2026-06-03): two session vetoes feed the incidents>=3 hard latch and
// fire on real faces —
//   * texture-collapse VIDEO_REPLAY veto fires in dim light (low texture from
//     camera denoising) because its skin co-signal assumes a live face has
//     skin_score < 30, but a real mobile face measures ~49.
//   * planar-print STATIC_IMAGE veto fires on any measured planarity < 45,
//     i.e. a normal head-turn, even with MiniFASNet=100 and the user blinking.
import { describe, test, expect } from "vitest";
import {
  SessionEngine,
  MultiClassFuser,
  BBox,
  type AnalyzerResult,
  type FaceROI,
  type FrameAnalysis,
} from "../src/index";

interface FrameSpec {
  minifasnet?: number;
  textureScore?: number; // texture.details.texture_score (Laplacian sub-feature)
  skinScore?: number; // screen_replay.details.skin_score
  planarity?: { score: number; measured: boolean } | null;
  overallVar?: number; // landmark_variance.details.overall_var (rigid-motion proxy)
  eyelidStd?: number; // blink_symmetry std_left/right (per eye)
  illum?: number; // gate illuminationScore 0..1 (lit face ~0.8, dim ~0.4)
  blinks?: number; // cumulative blink count
  still?: boolean; // perfectly static bbox (for motion-static incident tests)
}

function r(name: string, score: number, details: Record<string, unknown> = {}): AnalyzerResult {
  return { name, score, details, elapsed_ms: 0 };
}

/** Replay a scripted session through the real fuser + engine; return verdict. */
function runSession(
  fps: number,
  durationSec: number,
  spec: (frameIndex: number, tSec: number) => FrameSpec,
) {
  const fuser = new MultiClassFuser();
  const engine = new SessionEngine({}); // amispoof default: no prover, requireProverLive=false
  const base = 1_000_000_000_000;
  const realNow = Date.now;
  const frames = Math.round(fps * durationSec);
  const dtMs = 1000 / fps;
  (Date as any).now = () => base; // overwritten per-frame below
  try {
    let clock = base;
    (Date as any).now = () => clock;
    engine.start();
    for (let i = 0; i < frames; i++) {
      const tSec = (i * dtMs) / 1000;
      clock = base + i * dtMs;
      const s = spec(i, tSec);
      const tex = s.textureScore ?? 100;
      const results: Record<string, AnalyzerResult> = {
        minifasnet: r("minifasnet", s.minifasnet ?? 100, { p_real: 0.99 }),
        texture: r("texture", Math.max(tex, 10), { texture_score: tex }),
        screen_replay: r("screen_replay", 75, { skin_score: s.skinScore ?? 15 }),
        landmark_variance: r("landmark_variance", 60, { overall_var: s.overallVar ?? 6 }),
        blink_symmetry: r("blink_symmetry", 80, {
          std_left: s.eyelidStd ?? 0.05,
          std_right: s.eyelidStd ?? 0.05,
        }),
        device_boundary: r("device_boundary", 92, {}),
        background_grid: r("background_grid", 80, {}),
        blink: r("blink", 90, { blinks: s.blinks ?? 0 }),
      };
      if (s.planarity) {
        results.planarity = r("planarity", s.planarity.score, {
          measured: s.planarity.measured,
          residual_norm: 0.1,
        });
      }
      const cls = fuser.fuse(0, results);
      const cx = s.still ? 320 : 320 + 6 * Math.sin(i / 5);
      const cy = s.still ? 240 : 240 + 5 * Math.cos(i / 7);
      const face: FaceROI = { face_id: 0, bbox: new BBox(cx - 60, cy - 80, cx + 60, cy + 80), confidence: 1 };
      const analysis: FrameAnalysis = {
        frame_id: i,
        faces: [face],
        classifications: { 0: cls },
        frame_signals: {},
        total_ms: 0,
        gate_result: {
          usable: true,
          blocked: false,
          reason: "",
          state: "CLEAR",
          occluded: false,
          qualityOk: true,
          occlusionScore: 0,
          illuminationScore: s.illum ?? 0.8,
          occludedRegions: [],
          underexposedRegions: [],
          overexposedRegions: [],
        },
      };
      engine.ingest(analysis);
    }
    const v = engine.getVerdict();
    return {
      is_live: v.is_live,
      quality_uncertain: v.quality_uncertain,
      word: v.is_live ? "LIVE" : v.quality_uncertain ? "UNCERTAIN" : "SPOOF",
      incidents: v.incidents.length,
    };
  } finally {
    (Date as any).now = realNow;
  }
}

// blink every ~5 s (typical human rate)
const blinkEvery5s = (i: number, t: number) => Math.floor(t / 5);

describe("real faces stay LIVE across camera/lighting variation", () => {
  test("dim/evening light real face (low texture, high skin, blinking) → LIVE", () => {
    // Code's own 'home sunset LIVE' figure: texture_score ~19. Real mobile
    // skin_score ~49. Dim illumination ~0.4.
    const v = runSession(7, 40, (i, t) => ({
      textureScore: 19,
      skinScore: 49,
      illum: 0.4,
      blinks: blinkEvery5s(i, t),
    }));
    expect(v.word).toBe("LIVE");
  });

  test("real face turning head (measured planarity dips < 45, blinking, good light) → LIVE", () => {
    const v = runSession(7, 40, (i, t) => ({
      textureScore: 100,
      skinScore: 15,
      illum: 0.8,
      planarity: { score: 30, measured: true },
      overallVar: 40, // head-turn => landmark motion
      blinks: blinkEvery5s(i, t),
    }));
    expect(v.word).toBe("LIVE");
  });
});

describe("spoof coverage preserved", () => {
  test("bright phone-screen replay (texture collapse, screen skin, well lit) → SPOOF", () => {
    // Replay signature from code's table: texture_score ~6-8, skin_score >=30,
    // good illumination (a screen is bright). Video face blinks => eyelid motion.
    const v = runSession(7, 20, (i, t) => ({
      minifasnet: 60,
      textureScore: 7,
      skinScore: 48,
      illum: 0.7,
      eyelidStd: 0.4, // moving video content
      blinks: blinkEvery5s(i, t),
    }));
    expect(v.word).toBe("SPOOF");
  });

  test("printed photo held up (flat planarity, never blinks, frozen) → SPOOF", () => {
    const v = runSession(7, 30, () => ({
      minifasnet: 100, // a sharp print fools MiniFASNet — planarity must catch it
      textureScore: 100,
      skinScore: 2, // print skin_score ~0
      illum: 0.8,
      planarity: { score: 20, measured: true },
      overallVar: 1, // frozen
      eyelidStd: 0, // no eye motion
      blinks: 0, // never blinks
    }));
    expect(v.word).toBe("SPOOF");
  });
});

describe("incident override is windowed, not an all-time latch", () => {
  test("real long session with 3 sparse stray incidents (>window apart) → LIVE", () => {
    // A real, blinking person who happens to hold very still in three brief
    // bursts spread across 90 s. Each burst raises one motion-static incident.
    // All-time count reaches 3 (old behaviour: locked SPOOF forever); but they
    // are >30 s apart, so a windowed override must NOT latch.
    const still = (t: number) =>
      (t >= 5 && t < 9) || (t >= 45 && t < 49) || (t >= 85 && t < 89);
    const v = runSession(30, 90, (i, t) => ({
      textureScore: 100,
      skinScore: 15,
      illum: 0.8,
      blinks: Math.floor(t / 5), // blinks ~every 5 s → not a photo
      still: still(t),
    }));
    expect(v.word).toBe("LIVE");
  });

  test("sustained attack firing dense incidents still latches → SPOOF", () => {
    const v = runSession(7, 20, (i, t) => ({
      minifasnet: 60,
      textureScore: 7, // dense texture-collapse incidents every 2.5 s
      skinScore: 48,
      illum: 0.7,
      eyelidStd: 0.4,
      blinks: Math.floor(t / 5),
    }));
    expect(v.word).toBe("SPOOF");
  });
});
