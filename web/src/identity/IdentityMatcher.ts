// IdentityMatcher — on-device, in-session IDENTITY CONTINUITY (research demo).
//
// A DIFFERENT axis from anti-spoofing (PAD). PAD asks "is this a real live
// face?"; this asks "is it the SAME person who enrolled?". A person-swap
// mid-session is a real live face — it passes every liveness/spoof check, so
// only identity matching catches it (an IMPERSONATION, reported distinctly from
// SPOOF). A mask trips both; a photo of the enrolled person trips only PAD.
//
// Scope (locked in PIPELINE_DESIGN.md): research-grade *continuity*, NOT
// high-assurance 1:1 authentication; 100% client-side, nothing uploaded, no
// account. This class is the embedder-agnostic decision core — the actual face
// embedding is produced by a pluggable `FaceEmbedder` (see OnnxFaceEmbedder).
//
// Decision logic:
//   * enrollment averages a few aligned-face embeddings into an L2-normalised
//     reference template.
//   * each subsequent frame's embedding is compared by cosine similarity.
//   * a SUSTAINED drop below `matchThreshold` (impostorStreak consecutive
//     mismatches) raises `impersonation` — hysteresis so a momentary head-turn
//     or a single bad crop doesn't false-trigger. A match resets the streak.

/** Produces an L2-normalised identity embedding from an aligned face crop. */
export interface FaceEmbedder {
  /** Lazy-load weights if needed. Resolves when embed() is callable. */
  ready(): Promise<void>;
  /** Embed a face crop → unit-length Float32Array, or null if it couldn't. */
  embed(faceCrop: ImageData): Promise<Float32Array | null>;
  /** Embedding dimensionality (for sanity checks). */
  readonly dim: number;
}

export type IdentityState = "unenrolled" | "enrolling" | "enrolled";

export interface IdentityMatchResult {
  state: IdentityState;
  /** Cosine similarity to the reference template in [-1, 1], or null if not comparable. */
  similarity: number | null;
  /** similarity >= matchThreshold this frame. */
  samePerson: boolean;
  /** Sustained mismatch (>= impostorStreak consecutive) — a person-swap. */
  impersonation: boolean;
  /** Consecutive below-threshold frames so far (diagnostic). */
  mismatchStreak: number;
}

export interface IdentityMatcherOptions {
  /** Cosine similarity at/above which two embeddings are the SAME person. Default 0.5. */
  matchThreshold?: number;
  /** Consecutive below-threshold frames before declaring impersonation. Default 5. */
  impostorStreak?: number;
  /** Number of enrollment samples required to finalise a template. Default 3. */
  enrollSamples?: number;
}

export class IdentityMatcher {
  private readonly matchThreshold: number;
  private readonly impostorStreak: number;
  private readonly enrollSamples: number;

  private template: Float32Array | null = null;
  private pending: Float32Array[] = [];
  private mismatchStreak = 0;
  private state: IdentityState = "unenrolled";

  constructor(options: IdentityMatcherOptions = {}) {
    this.matchThreshold = options.matchThreshold ?? 0.5;
    this.impostorStreak = Math.max(1, options.impostorStreak ?? 5);
    this.enrollSamples = Math.max(1, options.enrollSamples ?? 3);
  }

  getState(): IdentityState {
    return this.state;
  }

  /** Begin a fresh enrollment, discarding any previous template/samples. */
  beginEnroll(): void {
    this.template = null;
    this.pending = [];
    this.mismatchStreak = 0;
    this.state = "enrolling";
  }

  /**
   * Add one enrollment embedding. Returns how many samples are captured so far.
   * Once `enrollSamples` are collected the template is finalised automatically
   * and the state becomes "enrolled".
   */
  addEnrollSample(embedding: Float32Array): number {
    if (this.state !== "enrolling") this.beginEnroll();
    this.pending.push(l2normalize(embedding));
    const captured = this.pending.length;
    if (captured >= this.enrollSamples) this.finalizeEnroll(); // clears pending
    return captured;
  }

  /** Average the captured samples into the reference template. */
  finalizeEnroll(): boolean {
    if (this.pending.length === 0) return false;
    this.template = l2normalize(meanVector(this.pending));
    this.pending = [];
    this.mismatchStreak = 0;
    this.state = "enrolled";
    return true;
  }

  /** How many enrollment samples are captured so far. */
  enrollProgress(): { captured: number; required: number } {
    return { captured: this.pending.length, required: this.enrollSamples };
  }

  /** Restore an embedding previously persisted (opt-in "remember on this device"). */
  loadTemplate(template: Float32Array): void {
    this.template = l2normalize(template);
    this.pending = [];
    this.mismatchStreak = 0;
    this.state = "enrolled";
  }

  /** The current reference template (e.g. to persist), or null. */
  getTemplate(): Float32Array | null {
    return this.template ? this.template.slice() : null;
  }

  /** Compare a live embedding against the enrolled template. */
  match(embedding: Float32Array): IdentityMatchResult {
    if (this.state !== "enrolled" || !this.template) {
      return {
        state: this.state,
        similarity: null,
        samePerson: false,
        impersonation: false,
        mismatchStreak: this.mismatchStreak,
      };
    }
    const sim = cosine(l2normalize(embedding), this.template);
    const samePerson = sim >= this.matchThreshold;
    if (samePerson) this.mismatchStreak = 0;
    else this.mismatchStreak += 1;
    return {
      state: this.state,
      similarity: round(sim, 4),
      samePerson,
      impersonation: this.mismatchStreak >= this.impostorStreak,
      mismatchStreak: this.mismatchStreak,
    };
  }

  reset(): void {
    this.template = null;
    this.pending = [];
    this.mismatchStreak = 0;
    this.state = "unenrolled";
  }
}

/** Return a unit-length copy of v (no-op direction if v is all-zero). */
export function l2normalize(v: Float32Array): Float32Array {
  let sumSq = 0;
  for (let i = 0; i < v.length; i++) sumSq += v[i] * v[i];
  const norm = Math.sqrt(sumSq);
  if (norm < 1e-12) return v.slice();
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

/** Cosine similarity. If both inputs are already unit-length this is the dot product. */
export function cosine(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom < 1e-12 ? 0 : dot / denom;
}

/** Element-wise mean of a non-empty list of equal-length vectors. */
export function meanVector(vectors: Float32Array[]): Float32Array {
  const dim = vectors[0].length;
  const out = new Float32Array(dim);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) out[i] += v[i];
  }
  for (let i = 0; i < dim; i++) out[i] /= vectors.length;
  return out;
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
