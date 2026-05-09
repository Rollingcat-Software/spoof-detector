// Port of TestMultiClassFuser in tests/test_analyzers.py.
// Verifies that the TS port of MultiClassFuser produces equivalent
// classifications to the Python source.

import { describe, expect, it } from "vitest";
import { MultiClassFuser } from "../src/infrastructure/fusion/MultiClassFuser";
import {
  AnalyzerResult,
  makeAnalyzerResult,
  SpoofCategory,
} from "../src/domain/models";

function results(...rs: AnalyzerResult[]): Record<string, AnalyzerResult> {
  const out: Record<string, AnalyzerResult> = {};
  for (const r of rs) out[r.name] = r;
  return out;
}

describe("MultiClassFuser", () => {
  it("high scores favor REAL (port of test_high_scores_favor_real)", () => {
    const fuser = new MultiClassFuser();
    const cls = fuser.fuse(
      1,
      results(
        makeAnalyzerResult("minifasnet", 95.0),
        makeAnalyzerResult("texture", 90.0),
        makeAnalyzerResult("moire", 85.0),
      ),
    );
    expect(cls.dominant_category).toBe(SpoofCategory.REAL);
    expect(cls.probabilities[SpoofCategory.REAL]).toBeGreaterThan(0.5);
  });

  it("low scores favor a spoof category (port of test_low_scores_favor_spoof)", () => {
    const fuser = new MultiClassFuser();
    const cls = fuser.fuse(
      1,
      results(
        makeAnalyzerResult("minifasnet", 10.0),
        makeAnalyzerResult("texture", 15.0),
        makeAnalyzerResult("moire", 5.0),
      ),
    );
    expect(cls.dominant_category).not.toBe(SpoofCategory.REAL);
  });

  it("probabilities sum to 1 (port of test_probabilities_sum_to_one)", () => {
    const fuser = new MultiClassFuser();
    const cls = fuser.fuse(
      1,
      results(
        makeAnalyzerResult("minifasnet", 60.0),
        makeAnalyzerResult("texture", 70.0),
      ),
    );
    let total = 0;
    for (const p of Object.values(cls.probabilities)) total += p;
    expect(Math.abs(total - 1.0)).toBeLessThan(0.01);
  });

  it("low moire score increases screen-attack categories (port of test_moire_low_increases_screen_categories)", () => {
    const fuser = new MultiClassFuser();
    const cls = fuser.fuse(
      1,
      results(makeAnalyzerResult("moire", 10.0)),
    );
    const screenProb =
      cls.probabilities[SpoofCategory.VIDEO_REPLAY] +
      cls.probabilities[SpoofCategory.STATIC_IMAGE];
    expect(screenProb).toBeGreaterThan(0.3);
  });

  it("MiniFASNet weight dominates (5.0 vs 0.5) — single high MiniFASNet outweighs a low blink", () => {
    const fuser = new MultiClassFuser();
    const cls = fuser.fuse(
      2,
      results(
        makeAnalyzerResult("minifasnet", 99.0),
        makeAnalyzerResult("blink", 5.0),
      ),
    );
    expect(cls.dominant_category).toBe(SpoofCategory.REAL);
  });

  it("zero-weight analyzer (rppg) does not contribute", () => {
    const fuser = new MultiClassFuser();
    const withRppg = fuser.fuse(
      1,
      results(
        makeAnalyzerResult("minifasnet", 80.0),
        makeAnalyzerResult("rppg", 5.0),
      ),
    );
    const withoutRppg = fuser.fuse(
      1,
      results(makeAnalyzerResult("minifasnet", 80.0)),
    );
    // rppg is weight 0 in DEFAULT_ANALYZER_WEIGHTS — adding it must not move probs.
    for (const cat of Object.keys(withRppg.probabilities) as SpoofCategory[]) {
      expect(
        Math.abs(withRppg.probabilities[cat] - withoutRppg.probabilities[cat]),
      ).toBeLessThan(0.001);
    }
  });

  it("custom weights override defaults", () => {
    const fuser = new MultiClassFuser({ minifasnet: 0.0, blink: 5.0 });
    const cls = fuser.fuse(
      3,
      results(
        makeAnalyzerResult("minifasnet", 95.0), // ignored
        makeAnalyzerResult("blink", 5.0),       // strong spoof signal now
      ),
    );
    expect(cls.dominant_category).not.toBe(SpoofCategory.REAL);
  });

  it("Phase-2 analyzer keys are wired (landmark_variance, device_boundary, blink, micro_tremor, screen_flicker)", () => {
    // A high-score read across all 6 wired analyzers should still
    // produce dominant=REAL — and zero analyzers should be silently
    // dropped from the fusion total.
    const fuser = new MultiClassFuser();
    const cls = fuser.fuse(
      4,
      results(
        makeAnalyzerResult("minifasnet", 95.0),
        makeAnalyzerResult("landmark_variance", 90.0),
        makeAnalyzerResult("device_boundary", 90.0),
        makeAnalyzerResult("blink", 95.0),
        makeAnalyzerResult("micro_tremor", 85.0),
        makeAnalyzerResult("screen_flicker", 92.0),
      ),
    );
    expect(cls.dominant_category).toBe(SpoofCategory.REAL);
    // All 6 analyzers must show up in the analyzer_results round-trip.
    expect(Object.keys(cls.analyzer_results).sort()).toEqual([
      "blink",
      "device_boundary",
      "landmark_variance",
      "micro_tremor",
      "minifasnet",
      "screen_flicker",
    ]);
  });

  it("low screen_flicker score (high weight 3.0) flags VIDEO_REPLAY", () => {
    // screen_flicker has weight 3.0 and the SPOOF_SIGNAL_MAP entry
    // routes 0.50 of evidence to VIDEO_REPLAY. A solo low score should
    // make VIDEO_REPLAY the dominant non-REAL category.
    const fuser = new MultiClassFuser();
    const cls = fuser.fuse(
      5,
      results(makeAnalyzerResult("screen_flicker", 5.0)),
    );
    expect(cls.dominant_category).toBe(SpoofCategory.VIDEO_REPLAY);
  });
});
