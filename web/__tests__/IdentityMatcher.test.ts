// IdentityMatcher tests — the embedder-agnostic identity-continuity core.
// Uses synthetic embeddings: "same person" = a base direction + small noise;
// "different person" = a near-orthogonal direction. Pins enrollment, cosine
// matching, and the impersonation hysteresis (sustained mismatch, not a blip).

import { describe, expect, it } from "vitest";
import {
  IdentityMatcher,
  l2normalize,
  cosine,
  meanVector,
} from "../src/identity/IdentityMatcher";

const DIM = 16;

function vec(seed: number, jitter = 0): Float32Array {
  const v = new Float32Array(DIM);
  for (let i = 0; i < DIM; i++) {
    v[i] = Math.sin(seed * 12.9898 + i * 78.233) + jitter * Math.sin(i * 3.1 + seed);
  }
  return v;
}

const personA = vec(1);
const personAClose = vec(1, 0.05); // same person, slight variation
const personB = vec(99); // a different person

describe("IdentityMatcher", () => {
  it("starts unenrolled; match() is inert until enrolled", () => {
    const m = new IdentityMatcher();
    expect(m.getState()).toBe("unenrolled");
    const r = m.match(personA);
    expect(r.similarity).toBeNull();
    expect(r.samePerson).toBe(false);
    expect(r.impersonation).toBe(false);
  });

  it("enrolls after the required samples and matches the same person", () => {
    const m = new IdentityMatcher({ enrollSamples: 3 });
    m.beginEnroll();
    expect(m.addEnrollSample(vec(1, 0.02))).toBe(1);
    expect(m.addEnrollSample(vec(1, 0.04))).toBe(2);
    expect(m.addEnrollSample(vec(1, 0.06))).toBe(3); // auto-finalises
    expect(m.getState()).toBe("enrolled");

    const r = m.match(personAClose);
    expect(r.samePerson).toBe(true);
    expect(r.similarity!).toBeGreaterThan(0.5);
    expect(r.impersonation).toBe(false);
  });

  it("a different person is a mismatch, and a SUSTAINED mismatch → impersonation", () => {
    const m = new IdentityMatcher({ enrollSamples: 2, impostorStreak: 5, matchThreshold: 0.5 });
    m.addEnrollSample(personA);
    m.addEnrollSample(vec(1, 0.03));
    expect(m.getState()).toBe("enrolled");

    // first 4 mismatched frames: not yet impersonation (hysteresis)
    let r = m.match(personB);
    expect(r.samePerson).toBe(false);
    expect(r.similarity!).toBeLessThan(0.5);
    for (let i = 0; i < 3; i++) r = m.match(personB);
    expect(r.mismatchStreak).toBe(4);
    expect(r.impersonation).toBe(false);
    // 5th consecutive mismatch trips it
    r = m.match(personB);
    expect(r.mismatchStreak).toBe(5);
    expect(r.impersonation).toBe(true);
  });

  it("a single mismatch among matches does NOT trigger impersonation (hysteresis)", () => {
    const m = new IdentityMatcher({ enrollSamples: 1, impostorStreak: 5 });
    m.addEnrollSample(personA);
    m.match(personAClose);
    const blip = m.match(personB); // one bad frame
    expect(blip.impersonation).toBe(false);
    const back = m.match(personAClose); // back to the real person
    expect(back.samePerson).toBe(true);
    expect(back.mismatchStreak).toBe(0); // streak reset
  });

  it("loadTemplate / getTemplate round-trips a persisted identity", () => {
    const m1 = new IdentityMatcher({ enrollSamples: 1 });
    m1.addEnrollSample(personA);
    const t = m1.getTemplate()!;
    expect(t).not.toBeNull();

    const m2 = new IdentityMatcher();
    m2.loadTemplate(t);
    expect(m2.getState()).toBe("enrolled");
    expect(m2.match(personAClose).samePerson).toBe(true);
  });

  it("reset() returns to unenrolled", () => {
    const m = new IdentityMatcher({ enrollSamples: 1 });
    m.addEnrollSample(personA);
    m.reset();
    expect(m.getState()).toBe("unenrolled");
    expect(m.match(personA).similarity).toBeNull();
  });

  it("helpers: l2normalize is unit-length; cosine of a vector with itself is 1", () => {
    const u = l2normalize(personA);
    let sumSq = 0;
    for (const x of u) sumSq += x * x;
    expect(Math.sqrt(sumSq)).toBeCloseTo(1, 5);
    expect(cosine(personA, personA)).toBeCloseTo(1, 5);
    expect(cosine(personA, personB)).toBeLessThan(0.99);
  });

  it("meanVector averages element-wise", () => {
    const a = new Float32Array([0, 2, 4]);
    const b = new Float32Array([2, 4, 6]);
    const m = meanVector([a, b]);
    expect(Array.from(m)).toEqual([1, 3, 5]);
  });
});
