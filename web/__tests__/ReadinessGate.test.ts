// ReadinessGate tests — the pre-flight precondition that blocks session start.
// Encodes the abstain-first stance: bad captures become "fix this", never a
// false verdict. The dark-room case is the live 2026-05-25 finding that
// motivated the gate.

import { describe, expect, it } from "vitest";
import { ReadinessGate, ReadinessSignals } from "../src/application/ReadinessGate";

const good: ReadinessSignals = {
  faceCount: 1,
  faceAreaFraction: 0.18,
  faceBrightness: 130,
  occluded: false,
  occludedRegions: [],
  cameraResponsive: true,
};

const check = (r: ReturnType<ReadinessGate["evaluate"]>, id: string) =>
  r.checks.find((c) => c.id === id)!;

describe("ReadinessGate", () => {
  it("a good capture is ready (all checks pass)", () => {
    const r = new ReadinessGate().evaluate(good);
    expect(r.ready).toBe(true);
    expect(r.checks.every((c) => c.pass)).toBe(true);
  });

  it("dark room (no face detected) → not ready; face fails, rest pending", () => {
    // The live finding: pitch black → faceCount 0. Must block start, NOT verdict.
    const r = new ReadinessGate().evaluate({ ...good, faceCount: 0, faceBrightness: 0 });
    expect(r.ready).toBe(false);
    expect(check(r, "face").pass).toBe(false);
    expect(check(r, "face").message).toMatch(/no face/i);
    // dependent checks abstain until a single face is present
    expect(check(r, "lighting").pass).toBe(false);
    expect(check(r, "lighting").message).toMatch(/waiting/i);
  });

  it("over-lit face → lighting fails with a 'too bright' message", () => {
    const r = new ReadinessGate().evaluate({ ...good, faceBrightness: 220 });
    expect(r.ready).toBe(false);
    expect(check(r, "lighting").pass).toBe(false);
    expect(check(r, "lighting").message).toMatch(/too bright/i);
  });

  it("dim (but face-detected) → lighting fails with a 'too dark' message", () => {
    const r = new ReadinessGate().evaluate({ ...good, faceBrightness: 40 });
    expect(r.ready).toBe(false);
    expect(check(r, "lighting").pass).toBe(false);
    expect(check(r, "lighting").message).toMatch(/too dark/i);
  });

  it("face too far (small bbox) → distance fails", () => {
    const r = new ReadinessGate().evaluate({ ...good, faceAreaFraction: 0.02 });
    expect(r.ready).toBe(false);
    expect(check(r, "size").pass).toBe(false);
    expect(check(r, "size").message).toMatch(/closer/i);
  });

  it("multiple faces → face fails", () => {
    const r = new ReadinessGate().evaluate({ ...good, faceCount: 2 });
    expect(r.ready).toBe(false);
    expect(check(r, "face").pass).toBe(false);
    expect(check(r, "face").message).toMatch(/one person/i);
  });

  it("occluded face → visibility fails and names the region", () => {
    const r = new ReadinessGate().evaluate({
      ...good,
      occluded: true,
      occludedRegions: ["mouth"],
    });
    expect(r.ready).toBe(false);
    expect(check(r, "occlusion").pass).toBe(false);
    expect(check(r, "occlusion").message).toMatch(/uncover/i);
    expect(check(r, "occlusion").message).toMatch(/mouth/i);
  });

  it("camera not responding → camera fails", () => {
    const r = new ReadinessGate().evaluate({ ...good, cameraResponsive: false });
    expect(r.ready).toBe(false);
    expect(check(r, "camera").pass).toBe(false);
  });

  it("respects custom thresholds", () => {
    const strict = new ReadinessGate({ minFaceAreaFraction: 0.25, maxFaceBrightness: 150 });
    expect(strict.evaluate(good).ready).toBe(false); // 0.18 < 0.25
    const lenient = new ReadinessGate({ minFaceBrightness: 20, maxFaceBrightness: 250 });
    expect(lenient.evaluate({ ...good, faceBrightness: 30 }).ready).toBe(true);
  });
});
