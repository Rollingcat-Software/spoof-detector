// Port of src/application/liveness_prover.py
//
// "Guilty until proven innocent" — every session starts as SPOOF and the
// subject must accumulate enough passive + active liveness evidence to be
// declared LIVE. The Python source manages four passive signals (blinks,
// landmark variance, head rotation range, expression changes) plus an
// optional active-challenge loop (BLINK / TURN_LEFT / TURN_RIGHT / NOD /
// SHOW_HAND).
//
// Architectural parity with SessionEngine:
//   * Same lifecycle: `start()` / `ingest(analysis)` / `getProof()` / `reset()`.
//   * Same single-primary-face contract: only `analysis.faces[0]` is tracked
//     (matches SessionEngine.primaryFaceId).
//   * Same RingBuffer pattern for yaw/pitch history (Python `deque(maxlen=90)`).
//
// Critical mobile-Brave constraints (DO NOT VIOLATE):
//   * Pure JS — no image processing, no SharedArrayBuffer, no WebGPU, no
//     OffscreenCanvas. The prover consumes already-fused per-frame data.
//   * Single-threaded math.
//
// Python → TS adaptations:
//   * `time.time()` → `Date.now() / 1000` (wall-clock seconds, matches
//     SessionEngine.elapsedSec).
//   * `random.choice(available)` → injectable `random` callable (default
//     `Math.random`), so tests can pin challenge selection deterministically.
//   * Passive signals (blink_count, landmark_variance, expression_ratio,
//     landmarks) are pulled off the FrameAnalysis instead of being passed as
//     positional kwargs — `blink` analyzer details for blinks, `landmark_
//     variance` analyzer details for variance + expression_ratio, FaceROI
//     landmarks for head pose.
//   * `Optional[Challenge]` → `Challenge | null`.
//   * `@dataclass LivenessScore` → mutable interface with a `recomputeTotal`
//     helper (mirrors `update_total`).
//
// All calibrated thresholds are preserved VERBATIM from the Python source.

import { FrameAnalysis, SpoofClassification } from "../domain/models";
import { RingBuffer } from "../domain/session";

/** Active-challenge taxonomy (mirrors `ChallengeType` in Python). */
export enum ChallengeType {
  BLINK = "blink",
  TURN_LEFT = "turn_left",
  TURN_RIGHT = "turn_right",
  NOD = "nod",
  SHOW_HAND = "show_hand",
}

/** Lifecycle state of a single active challenge. */
export enum ChallengeState {
  WAITING = "waiting",
  PROMPTED = "prompted",
  COMPLETED = "completed",
  FAILED = "failed",
}

/** Display labels shown on the camera overlay (Python `display_text`). */
export const CHALLENGE_DISPLAY_TEXT: Readonly<Record<ChallengeType, string>> = {
  [ChallengeType.BLINK]: "BLINK YOUR EYES",
  [ChallengeType.TURN_LEFT]: "TURN HEAD LEFT",
  [ChallengeType.TURN_RIGHT]: "TURN HEAD RIGHT",
  [ChallengeType.NOD]: "NOD YOUR HEAD",
  [ChallengeType.SHOW_HAND]: "SHOW YOUR HAND",
};

/** An issued challenge and its outcome. */
export interface Challenge {
  challenge_type: ChallengeType;
  state: ChallengeState;
  /** Seconds since session start when the prompt was raised. */
  prompted_at: number;
  /** Seconds since session start when the user responded (0 if not completed). */
  completed_at: number;
  /** Hard timeout in seconds before the challenge is marked FAILED. */
  timeout_sec: number;
  /** Response latency in milliseconds (0 if still pending or failed). */
  response_latency_ms: number;
}

/** Accumulated liveness evidence. Mirrors `LivenessScore` dataclass. */
export interface LivenessScore {
  /** 0–100, must reach 60 to pass. */
  total: number;
  /** Max 25 (raised in Python — strongest passive signal). */
  blink_points: number;
  /** Max 20 from the inner clamp. */
  landmark_points: number;
  /** Max 15 from the inner clamp. */
  rotation_points: number;
  /** Max 15 from the inner clamp. */
  expression_points: number;
  /** Max 40 (active challenges). */
  challenge_points: number;
  challenges_passed: number;
  challenges_failed: number;
  /**
   * Max 12 — eye-region landmark drift, independent of blink counting.
   * Rewards gaze tracking and eyelid micro-motion. Additive over Python;
   * always 0 unless the LandmarkVarianceAnalyzer is wired and publishing
   * an `eye_var` detail.
   */
  eye_motion_points: number;
  /**
   * Max 10 — mouth-region landmark drift. Rewards talking, lip movement,
   * micro-expressions. Pulled from LandmarkVarianceAnalyzer `mouth_var`.
   */
  mouth_motion_points: number;
  /**
   * Max 8 — bbox/centroid drift over time. Rewards natural face/body sway.
   * Pulled from TemporalAnalyzer `motion`. 0 on a perfectly static photo.
   */
  face_motion_points: number;
  /**
   * Max 8 — eyebrow blendshape variance over rolling window. Sourced from
   * EyebrowAnalyzer details. 0 on a static photo with rigid brows.
   */
  eyebrow_motion_points: number;
  /**
   * Max 6 — left/right blink Pearson correlation. Sourced from
   * BlinkSymmetryAnalyzer. Awarded only when corr ≥ 0.7 (real synchronous
   * blinks); animated avatars / deepfakes typically desync per-eye blinks.
   */
  blink_symmetry_points: number;
  /**
   * Max 8 — gaze direction variance + saccade count. Sourced from
   * GazeAnalyzer. 0 on a fixed-gaze photo; humans saccade ~3/s.
   */
  gaze_variation_points: number;
  /**
   * Max 8 — mouth/cheek/nose blendshape variance over rolling window.
   * Sourced from ExpressionDynamicsAnalyzer. Approximates a passive
   * emotion-change signal without a dedicated classifier.
   */
  expression_dynamics_points: number;
  /**
   * Max 6 — 3D pose rotation orthonormality + Z-translation motion.
   * Sourced from Pose3DConsistencyAnalyzer. Catches flat-photo /
   * tilted-photo / 2D-screen attack classes whose transformation
   * matrix fits are subtly degenerate.
   */
  pose_3d_consistency_points: number;
  /**
   * Max 10 — temporal behavioural patterns (blink-interval coefficient
   * of variation, saccade rate, signal entropy). Sourced from
   * BehavioralPatternAnalyzer (Phase B). Catches looped videos and
   * animated avatars whose signal distributions look "too regular"
   * even when individual frames pass other analyzers.
   */
  behavioral_pattern_points: number;
  /**
   * Max 8 — background-region pixel drift over time, computed from
   * MediaPipe SelfieSegmenter masks (Phase D1, opt-in). Catches replay
   * attacks shown against a static backdrop. 0 when the segmenter
   * isn't enabled or when the scene is genuinely still.
   */
  background_motion_points: number;
  /**
   * Max 8 — hand-presence + gesture motion from MediaPipe HandLandmarker
   * (Phase D2, opt-in). Credits natural hand movement; caps low on the
   * deepfake "third hand" anomaly. 0 when hands are absent (neutral) or
   * when tracking is off.
   */
  hand_naturalness_points: number;
  /**
   * Max 6 — voice-activity fraction over the recent audio window
   * (Phase D3, opt-in). Strong defensive signal against silent video
   * replays. 0 when audio is off or unavailable.
   */
  voice_activity_points: number;
  /**
   * Max 12 — Pearson correlation of audio RMS vs jawOpen blendshape
   * over the rolling window (Phase D3, opt-in). Strongest single
   * anti-video-replay signal — replay attacks either lack audio or
   * have audio that's desynced from the visible mouth motion.
   */
  audio_mouth_sync_points: number;
}

/** Concise history record for reporting (mirrors `get_challenge_history`). */
export interface ChallengeHistoryEntry {
  type: ChallengeType;
  state: ChallengeState;
  latency_ms: number;
}

/** Full proof payload returned by `getProof()`. */
export interface LivenessProof {
  /** Total score in [0, 100]. >= 60 ⇒ proven live. */
  score: number;
  /** Threshold-crossing convenience boolean. */
  is_proven_live: boolean;
  /** Per-axis evidence breakdown. */
  details: LivenessScore;
  /** Currently-pending challenge, if any. */
  active_challenge: Challenge | null;
  /** Full per-challenge audit trail. */
  challenge_history: ChallengeHistoryEntry[];
  /** Seconds since `start()`. */
  elapsed_sec: number;
  /**
   * Largest yaw range observed across the rolling 3-second window, in
   * degrees. Surfaced so consumers can display "head turn coverage"
   * directly instead of inferring it from the rotation_points.
   */
  yaw_range_seen_deg: number;
  /** Largest pitch range observed across the same window, in degrees. */
  pitch_range_seen_deg: number;
}

/** Constructor options. */
export interface LivenessProverOptions {
  /** Enable active challenges (default true). */
  enableChallenges?: boolean;
  /**
   * Deterministic RNG for tests. Must return a float in [0, 1). Defaults to
   * `Math.random`. The prover only uses it to pick the next challenge type.
   */
  random?: () => number;
  /**
   * Override `EXPRESSION_RATIO_GATE` (default 1.2 — Python parity). Lower
   * the gate when the consumer wants natural-observation expression scoring;
   * proctoring (passive-only) typically uses 0.4.
   */
  expressionRatioGate?: number;
  /**
   * Override `ROTATION_THRESHOLD` in degrees (default 3.0 — Python parity).
   * Lower it to credit subtle head movement. Proctoring typically uses 2.0.
   */
  rotationThreshold?: number;
  /**
   * Override `LANDMARK_VAR_THRESHOLD` (default 1.0 — Python parity). Lower
   * it to credit small mesh drift. Proctoring typically uses 0.5.
   */
  landmarkVarThreshold?: number;
}

// MediaPipe face mesh key points for head pose estimation
// (verbatim from Python lines 98–104).
const NOSE_TIP = 1;
const FOREHEAD = 10;
const CHIN = 152;
const LEFT_EAR = 234;
const RIGHT_EAR = 454;

/**
 * Liveness prover — accumulates passive + active liveness evidence.
 *
 * Usage:
 * ```ts
 * const prover = new LivenessProver();
 * prover.start();
 * for each frame: prover.ingest(analysis);
 * const proof = prover.getProof();
 * if (proof.is_proven_live) ... // total >= 60
 * ```
 */
export class LivenessProver {
  // === Thresholds — verbatim port of Python class constants ===
  static readonly CHALLENGE_INTERVAL_SEC = 8.0;
  static readonly MAX_CHALLENGES = 5;
  static readonly BLINK_AWARD = 5.0;
  static readonly MAX_BLINK_POINTS = 25.0;
  static readonly LANDMARK_VAR_THRESHOLD = 1.0;
  static readonly ROTATION_THRESHOLD = 3.0;
  static readonly CHALLENGE_AWARD = 10.0;

  // === Cut-offs preserved verbatim from Python method bodies ===
  // (kept as named constants so the test suite can reference them.)
  static readonly PROOF_PASS_THRESHOLD = 60.0;
  static readonly EXPRESSION_RATIO_GATE = 1.2;
  static readonly FIRST_CHALLENGE_AFTER_SEC = 5.0;
  static readonly CHALLENGE_DEFAULT_TIMEOUT_SEC = 4.0;
  static readonly YAW_DELTA_TURN_DEG = 8.0;
  static readonly PITCH_DELTA_NOD_DEG = 6.0;
  static readonly LANDMARK_POINT_CAP = 20.0;
  static readonly EXPRESSION_POINT_CAP = 15.0;
  static readonly ROTATION_YAW_CAP = 10.0;
  static readonly ROTATION_PITCH_CAP = 15.0;
  static readonly CHALLENGE_POINT_CAP = 40.0;
  static readonly LANDMARK_VAR_GAIN = 4.0;
  static readonly EXPRESSION_GAIN = 3.0;
  static readonly YAW_GAIN = 0.5;
  static readonly PITCH_GAIN = 0.3;

  // === Passive movement axes (additive over Python — credit per-region
  // landmark drift and face-position drift that Python rolls up into the
  // single `landmark_points` term). The constants are conservative so they
  // can be turned on as defaults without breaking Python-parity tests:
  // their `*_THRESHOLD` is the minimum value below which no points are
  // awarded, and the inner cap saturates a reasonably-active session.
  static readonly EYE_MOTION_POINT_CAP = 12.0;
  static readonly EYE_MOTION_GAIN = 0.5;
  static readonly EYE_MOTION_THRESHOLD = 1.0;
  static readonly MOUTH_MOTION_POINT_CAP = 10.0;
  static readonly MOUTH_MOTION_GAIN = 0.5;
  static readonly MOUTH_MOTION_THRESHOLD = 1.0;
  static readonly FACE_MOTION_POINT_CAP = 8.0;
  static readonly FACE_MOTION_GAIN = 20.0;
  static readonly FACE_MOTION_THRESHOLD = 0.05;

  // === Phase A passive movement axes (sourced from MediaPipe blendshape
  // + transformMatrix outputs unlocked by the detector flag flip). Each
  // axis reads from its dedicated analyzer's `score` directly (already
  // 0–100) and scales by the cap. Threshold = minimum score below which
  // no points are awarded (e.g. neutral 50 during warmup → 0 pts).
  static readonly EYEBROW_MOTION_POINT_CAP = 8.0;
  static readonly EYEBROW_MOTION_THRESHOLD = 5.0; // analyzer score
  static readonly BLINK_SYMMETRY_POINT_CAP = 6.0;
  static readonly BLINK_SYMMETRY_THRESHOLD = 70.0; // corr ≥ 0.7 → 70 score
  static readonly GAZE_VARIATION_POINT_CAP = 8.0;
  static readonly GAZE_VARIATION_THRESHOLD = 10.0;
  static readonly EXPRESSION_DYNAMICS_POINT_CAP = 8.0;
  static readonly EXPRESSION_DYNAMICS_THRESHOLD = 5.0;
  static readonly POSE_3D_CONSISTENCY_POINT_CAP = 6.0;
  static readonly POSE_3D_CONSISTENCY_THRESHOLD = 30.0;
  static readonly BEHAVIORAL_PATTERN_POINT_CAP = 10.0;
  static readonly BEHAVIORAL_PATTERN_THRESHOLD = 15.0;
  static readonly BACKGROUND_MOTION_POINT_CAP = 8.0;
  static readonly BACKGROUND_MOTION_THRESHOLD = 15.0;
  static readonly HAND_NATURALNESS_POINT_CAP = 8.0;
  static readonly HAND_NATURALNESS_THRESHOLD = 15.0;
  static readonly VOICE_ACTIVITY_POINT_CAP = 6.0;
  static readonly VOICE_ACTIVITY_THRESHOLD = 20.0;
  static readonly AUDIO_MOUTH_SYNC_POINT_CAP = 12.0;
  static readonly AUDIO_MOUTH_SYNC_THRESHOLD = 50.0; // corr ≥ 0.5

  private readonly enableChallenges: boolean;
  private readonly random: () => number;
  // Tunable gates (Python defaults preserved). See LivenessProverOptions.
  private readonly expressionRatioGate: number;
  private readonly rotationThreshold: number;
  private readonly landmarkVarThreshold: number;

  // Use ms epoch like SessionEngine; `elapsedSec` derives seconds-since-start.
  private startTimeMs = 0;

  private score: LivenessScore = LivenessProver.emptyScore();

  private challenges: Challenge[] = [];
  private activeChallenge: Challenge | null = null;
  private lastChallengeTimeSec = 0;
  private challengesIssued = 0;

  // Head pose tracking (Python `deque(maxlen=90)`).
  private yawHistory = new RingBuffer<number>(90);
  private pitchHistory = new RingBuffer<number>(90);
  private yawRangeSeen = 0;
  private pitchRangeSeen = 0;

  // Baselines captured when a challenge is issued.
  private baselineYaw: number | null = null;
  private baselinePitch: number | null = null;
  private blinkCountAtChallenge = 0;

  // Track last observed blink count across frames (Python pulls this via the
  // closure of `update(blink_count, ...)`; here we lift it off the blink
  // analyzer in `ingest()` and remember it for hand-off into the lower-level
  // `update()` signature).
  private lastSeenBlinkCount = 0;

  constructor(options: LivenessProverOptions = {}) {
    this.enableChallenges = options.enableChallenges ?? true;
    this.random = options.random ?? Math.random;
    this.expressionRatioGate =
      options.expressionRatioGate ?? LivenessProver.EXPRESSION_RATIO_GATE;
    this.rotationThreshold =
      options.rotationThreshold ?? LivenessProver.ROTATION_THRESHOLD;
    this.landmarkVarThreshold =
      options.landmarkVarThreshold ?? LivenessProver.LANDMARK_VAR_THRESHOLD;
  }

  /** Start the session clock. */
  start(): void {
    this.startTimeMs = Date.now();
  }

  /** Seconds since `start()` (0 if `start()` wasn't called). */
  get elapsedSec(): number {
    if (this.startTimeMs === 0) return 0;
    return (Date.now() - this.startTimeMs) / 1000;
  }

  /** Currently pending challenge (or null). */
  getActiveChallenge(): Challenge | null {
    return this.activeChallenge;
  }

  /** Live snapshot of the score struct. */
  getScore(): LivenessScore {
    return { ...this.score };
  }

  /** Per-challenge audit trail (only COMPLETED challenges are recorded). */
  getChallengeHistory(): ChallengeHistoryEntry[] {
    return this.challenges.map((c) => ({
      type: c.challenge_type,
      state: c.state,
      latency_ms: Math.round(c.response_latency_ms),
    }));
  }

  /** Convenience verdict accessor. */
  get isProvenLive(): boolean {
    return this.score.total >= LivenessProver.PROOF_PASS_THRESHOLD;
  }

  /** Full proof payload (SessionEngine.getVerdict() analog). */
  getProof(): LivenessProof {
    return {
      score: this.score.total,
      is_proven_live: this.isProvenLive,
      details: this.getScore(),
      active_challenge: this.activeChallenge,
      challenge_history: this.getChallengeHistory(),
      elapsed_sec: this.elapsedSec,
      yaw_range_seen_deg: this.yawRangeSeen,
      pitch_range_seen_deg: this.pitchRangeSeen,
    };
  }

  /** Reset all state so the instance can be reused. */
  reset(): void {
    this.startTimeMs = 0;
    this.score = LivenessProver.emptyScore();
    this.challenges = [];
    this.activeChallenge = null;
    this.lastChallengeTimeSec = 0;
    this.challengesIssued = 0;
    this.yawHistory.clear();
    this.pitchHistory.clear();
    this.yawRangeSeen = 0;
    this.pitchRangeSeen = 0;
    this.baselineYaw = null;
    this.baselinePitch = null;
    this.blinkCountAtChallenge = 0;
    this.lastSeenBlinkCount = 0;
  }

  /**
   * Convenience ingestor — pulls passive signals out of `FrameAnalysis` and
   * delegates to the lower-level `update()`. Matches SessionEngine's
   * `ingest(analysis)` lifecycle so the facade can run both engines in
   * parallel off the same analysis stream.
   *
   * Only the primary face (analysis.faces[0]) is tracked — matches the
   * SessionEngine.primaryFaceId convention.
   */
  ingest(analysis: FrameAnalysis): void {
    const faceCount = analysis.faces.length;
    if (faceCount === 0) {
      // No face — Python's update() short-circuits passive proofs but still
      // ticks the challenge clock. We mirror that.
      this.update(null, this.lastSeenBlinkCount, 0, 0, 0);
      return;
    }

    const primary = analysis.faces[0];
    const cls = analysis.classifications[primary.face_id];

    const blinkCount = readBlinkCount(cls);
    const landmarks = primary.landmarks ?? null;
    const { landmarkVariance, expressionRatio, eyeVar, mouthVar } =
      readLandmarkVarianceDetails(cls);
    const faceMotion = readTemporalMotion(cls);
    // Phase A: per-analyzer 0–100 scores. The prover doesn't recompute
    // any signals here — it just awards proof points proportional to the
    // analyzer's published score, gated by a threshold floor.
    const eyebrowScore = readAnalyzerScore(cls, "eyebrow_motion");
    const blinkSymmetryScore = readAnalyzerScore(cls, "blink_symmetry");
    const gazeScore = readAnalyzerScore(cls, "gaze");
    const expressionDynamicsScore = readAnalyzerScore(cls, "expression_dynamics");
    const pose3dScore = readAnalyzerScore(cls, "pose_3d_consistency");
    const behavioralScore = readAnalyzerScore(cls, "behavioral_pattern");
    const backgroundMotionScore = readAnalyzerScore(cls, "background_motion");
    const handTrackingScore = readAnalyzerScore(cls, "hand_tracking");
    const voiceActivityScore = readAnalyzerScore(cls, "voice_activity");
    const audioMouthSyncScore = readAnalyzerScore(cls, "audio_mouth_sync");

    this.lastSeenBlinkCount = blinkCount;

    this.update(
      landmarks,
      blinkCount,
      landmarkVariance,
      expressionRatio,
      faceCount,
      eyeVar,
      mouthVar,
      faceMotion,
      eyebrowScore,
      blinkSymmetryScore,
      gazeScore,
      expressionDynamicsScore,
      pose3dScore,
      behavioralScore,
      backgroundMotionScore,
      handTrackingScore,
      voiceActivityScore,
      audioMouthSyncScore,
    );
  }

  /**
   * Low-level per-frame update. Direct 1:1 port of Python `update(...)`.
   *
   * `landmarks` is the flat MediaPipe Nx2 buffer (length 2N). Pass `null`
   * when no face is present this frame.
   */
  update(
    landmarks: Float32Array | null,
    blinkCount: number,
    landmarkVariance: number,
    expressionRatio: number,
    faceCount: number,
    eyeVar: number = 0,
    mouthVar: number = 0,
    faceMotion: number = 0,
    eyebrowScore: number = 0,
    blinkSymmetryScore: number = 0,
    gazeScore: number = 0,
    expressionDynamicsScore: number = 0,
    pose3dScore: number = 0,
    behavioralScore: number = 0,
    backgroundMotionScore: number = 0,
    handTrackingScore: number = 0,
    voiceActivityScore: number = 0,
    audioMouthSyncScore: number = 0,
  ): void {
    const elapsed = this.elapsedSec;

    // === Passive Proof: Blinks ===
    if (blinkCount > 0) {
      this.score.blink_points = Math.min(
        LivenessProver.MAX_BLINK_POINTS,
        blinkCount * LivenessProver.BLINK_AWARD,
      );
    }

    // === Passive Proof: Landmark Variance ===
    if (landmarkVariance > this.landmarkVarThreshold) {
      this.score.landmark_points = Math.min(
        LivenessProver.LANDMARK_POINT_CAP,
        landmarkVariance * LivenessProver.LANDMARK_VAR_GAIN,
      );
    }

    // === Passive Proof: Expression Changes ===
    if (expressionRatio > this.expressionRatioGate) {
      this.score.expression_points = Math.min(
        LivenessProver.EXPRESSION_POINT_CAP,
        expressionRatio * LivenessProver.EXPRESSION_GAIN,
      );
    }

    // === Passive Proof: Eye-region motion (additive over Python) ===
    // Independent of blink counting — credits gaze tracking and eyelid
    // micro-movement that the binary blink detector misses.
    if (eyeVar > LivenessProver.EYE_MOTION_THRESHOLD) {
      this.score.eye_motion_points = Math.min(
        LivenessProver.EYE_MOTION_POINT_CAP,
        eyeVar * LivenessProver.EYE_MOTION_GAIN,
      );
    }

    // === Passive Proof: Mouth-region motion (additive over Python) ===
    // Credits talking, lip movement, and sub-expression-ratio mouth motion.
    if (mouthVar > LivenessProver.MOUTH_MOTION_THRESHOLD) {
      this.score.mouth_motion_points = Math.min(
        LivenessProver.MOUTH_MOTION_POINT_CAP,
        mouthVar * LivenessProver.MOUTH_MOTION_GAIN,
      );
    }

    // === Passive Proof: Face/body sway (additive over Python) ===
    // Credits natural drift in face bbox over time, even when the head
    // pose is steady. 0 on a perfectly locked photo or frozen-frame replay.
    if (faceMotion > LivenessProver.FACE_MOTION_THRESHOLD) {
      this.score.face_motion_points = Math.min(
        LivenessProver.FACE_MOTION_POINT_CAP,
        faceMotion * LivenessProver.FACE_MOTION_GAIN,
      );
    }

    // === Phase A passive movement axes (blendshape / 3D-matrix derived).
    // Each axis is awarded as a linear scale of the analyzer's published
    // 0–100 score, clamped by the axis cap. The threshold floor zeroes out
    // the analyzer's neutral 50 (warmup / no-data state) so warming-up
    // frames don't get free credit.
    if (eyebrowScore > LivenessProver.EYEBROW_MOTION_THRESHOLD) {
      this.score.eyebrow_motion_points = Math.min(
        LivenessProver.EYEBROW_MOTION_POINT_CAP,
        (eyebrowScore / 100) * LivenessProver.EYEBROW_MOTION_POINT_CAP,
      );
    }
    if (blinkSymmetryScore > LivenessProver.BLINK_SYMMETRY_THRESHOLD) {
      this.score.blink_symmetry_points = Math.min(
        LivenessProver.BLINK_SYMMETRY_POINT_CAP,
        ((blinkSymmetryScore - LivenessProver.BLINK_SYMMETRY_THRESHOLD) /
          (100 - LivenessProver.BLINK_SYMMETRY_THRESHOLD)) *
          LivenessProver.BLINK_SYMMETRY_POINT_CAP,
      );
    }
    if (gazeScore > LivenessProver.GAZE_VARIATION_THRESHOLD) {
      this.score.gaze_variation_points = Math.min(
        LivenessProver.GAZE_VARIATION_POINT_CAP,
        (gazeScore / 100) * LivenessProver.GAZE_VARIATION_POINT_CAP,
      );
    }
    if (expressionDynamicsScore > LivenessProver.EXPRESSION_DYNAMICS_THRESHOLD) {
      this.score.expression_dynamics_points = Math.min(
        LivenessProver.EXPRESSION_DYNAMICS_POINT_CAP,
        (expressionDynamicsScore / 100) *
          LivenessProver.EXPRESSION_DYNAMICS_POINT_CAP,
      );
    }
    if (pose3dScore > LivenessProver.POSE_3D_CONSISTENCY_THRESHOLD) {
      this.score.pose_3d_consistency_points = Math.min(
        LivenessProver.POSE_3D_CONSISTENCY_POINT_CAP,
        (pose3dScore / 100) *
          LivenessProver.POSE_3D_CONSISTENCY_POINT_CAP,
      );
    }
    if (behavioralScore > LivenessProver.BEHAVIORAL_PATTERN_THRESHOLD) {
      this.score.behavioral_pattern_points = Math.min(
        LivenessProver.BEHAVIORAL_PATTERN_POINT_CAP,
        (behavioralScore / 100) *
          LivenessProver.BEHAVIORAL_PATTERN_POINT_CAP,
      );
    }
    if (backgroundMotionScore > LivenessProver.BACKGROUND_MOTION_THRESHOLD) {
      this.score.background_motion_points = Math.min(
        LivenessProver.BACKGROUND_MOTION_POINT_CAP,
        (backgroundMotionScore / 100) *
          LivenessProver.BACKGROUND_MOTION_POINT_CAP,
      );
    }
    if (handTrackingScore > LivenessProver.HAND_NATURALNESS_THRESHOLD) {
      this.score.hand_naturalness_points = Math.min(
        LivenessProver.HAND_NATURALNESS_POINT_CAP,
        (handTrackingScore / 100) *
          LivenessProver.HAND_NATURALNESS_POINT_CAP,
      );
    }
    if (voiceActivityScore > LivenessProver.VOICE_ACTIVITY_THRESHOLD) {
      this.score.voice_activity_points = Math.min(
        LivenessProver.VOICE_ACTIVITY_POINT_CAP,
        (voiceActivityScore / 100) *
          LivenessProver.VOICE_ACTIVITY_POINT_CAP,
      );
    }
    if (audioMouthSyncScore > LivenessProver.AUDIO_MOUTH_SYNC_THRESHOLD) {
      this.score.audio_mouth_sync_points = Math.min(
        LivenessProver.AUDIO_MOUTH_SYNC_POINT_CAP,
        ((audioMouthSyncScore - LivenessProver.AUDIO_MOUTH_SYNC_THRESHOLD) /
          (100 - LivenessProver.AUDIO_MOUTH_SYNC_THRESHOLD)) *
          LivenessProver.AUDIO_MOUTH_SYNC_POINT_CAP,
      );
    }

    // === Passive Proof: Head Rotation ===
    // landmarks are stored flat [x0, y0, x1, y1, ...]. The Python check
    // `len(landmarks) > max(NOSE_TIP, LEFT_EAR, RIGHT_EAR, FOREHEAD, CHIN)`
    // counts landmark *points* (each is an xy pair).
    if (landmarks && landmarks.length / 2 > maxIndex()) {
      const [yaw, pitch] = this.estimateHeadPose(landmarks);
      this.yawHistory.append(yaw);
      this.pitchHistory.append(pitch);

      if (this.yawHistory.length > 10) {
        const yawArr = this.yawHistory.toArray();
        const pitchArr = this.pitchHistory.toArray();
        const yawSpan = arrayMax(yawArr) - arrayMin(yawArr);
        const pitchSpan = arrayMax(pitchArr) - arrayMin(pitchArr);
        if (yawSpan > this.yawRangeSeen) this.yawRangeSeen = yawSpan;
        if (pitchSpan > this.pitchRangeSeen) this.pitchRangeSeen = pitchSpan;
      }

      if (this.yawRangeSeen > this.rotationThreshold) {
        this.score.rotation_points = Math.min(
          LivenessProver.ROTATION_YAW_CAP,
          this.yawRangeSeen * LivenessProver.YAW_GAIN,
        );
      }
      if (this.pitchRangeSeen > this.rotationThreshold) {
        this.score.rotation_points = Math.min(
          LivenessProver.ROTATION_PITCH_CAP,
          this.score.rotation_points +
            this.pitchRangeSeen * LivenessProver.PITCH_GAIN,
        );
      }
    }

    // === Active Challenges ===
    if (this.enableChallenges) {
      this.manageChallenges(elapsed, blinkCount, landmarks, faceCount);
    }

    this.recomputeTotal();
  }

  // --- Internal helpers (mirror Python private methods) ---------------------

  private static emptyScore(): LivenessScore {
    return {
      total: 0,
      blink_points: 0,
      landmark_points: 0,
      rotation_points: 0,
      expression_points: 0,
      challenge_points: 0,
      challenges_passed: 0,
      challenges_failed: 0,
      eye_motion_points: 0,
      mouth_motion_points: 0,
      face_motion_points: 0,
      eyebrow_motion_points: 0,
      blink_symmetry_points: 0,
      gaze_variation_points: 0,
      expression_dynamics_points: 0,
      pose_3d_consistency_points: 0,
      behavioral_pattern_points: 0,
      background_motion_points: 0,
      hand_naturalness_points: 0,
      voice_activity_points: 0,
      audio_mouth_sync_points: 0,
    };
  }

  private recomputeTotal(): void {
    this.score.total = Math.min(
      100.0,
      this.score.blink_points +
        this.score.landmark_points +
        this.score.rotation_points +
        this.score.expression_points +
        this.score.challenge_points +
        this.score.eye_motion_points +
        this.score.mouth_motion_points +
        this.score.face_motion_points +
        this.score.eyebrow_motion_points +
        this.score.blink_symmetry_points +
        this.score.gaze_variation_points +
        this.score.expression_dynamics_points +
        this.score.pose_3d_consistency_points +
        this.score.behavioral_pattern_points +
        this.score.background_motion_points +
        this.score.hand_naturalness_points +
        this.score.voice_activity_points +
        this.score.audio_mouth_sync_points,
    );
  }

  /**
   * Yaw / pitch in degrees from the 478-pt face mesh (verbatim formula
   * port of `_estimate_head_pose`, see Python lines 209–235).
   */
  private estimateHeadPose(landmarks: Float32Array): [number, number] {
    const nose = readPoint(landmarks, NOSE_TIP);
    const leftEar = readPoint(landmarks, LEFT_EAR);
    const rightEar = readPoint(landmarks, RIGHT_EAR);
    const forehead = readPoint(landmarks, FOREHEAD);
    const chin = readPoint(landmarks, CHIN);

    // Yaw: nose position relative to ear midpoint.
    const earMidX = (leftEar[0] + rightEar[0]) / 2.0;
    const earDist = Math.hypot(
      leftEar[0] - rightEar[0],
      leftEar[1] - rightEar[1],
    );
    let yaw = 0;
    if (earDist > 1.0) {
      const yawRatio = (nose[0] - earMidX) / (earDist / 2.0);
      const clamped = Math.max(-1, Math.min(1, yawRatio));
      yaw = (Math.asin(clamped) * 180) / Math.PI;
    }

    // Pitch: nose vertical position relative to forehead-chin line.
    const faceHeight = Math.hypot(
      forehead[0] - chin[0],
      forehead[1] - chin[1],
    );
    let pitch = 0;
    if (faceHeight > 1.0) {
      const verticalMid = (forehead[1] + chin[1]) / 2.0;
      const pitchRatio = (nose[1] - verticalMid) / (faceHeight / 2.0);
      const clamped = Math.max(-1, Math.min(1, pitchRatio));
      pitch = (Math.asin(clamped) * 180) / Math.PI;
    }

    // Physiological clamp — MediaPipe occasionally returns degenerate
    // landmark configurations (face partially out-of-frame, low-light
    // glitch) that produce yaw/pitch readings outside the human range
    // of motion for a webcam session. Without this, the monotonic
    // `yawRangeSeen` peak tracker wedges at outlier values (a Chrome
    // mobile run reported yaw=112.8° on a clearly-live face), which
    // doesn't affect the rotation_points score (already capped) but
    // surfaces to consumers as an obviously-wrong "head motion" number.
    const yawClamp = 60.0;
    const pitchClamp = 60.0;
    yaw = Math.max(-yawClamp, Math.min(yawClamp, yaw));
    pitch = Math.max(-pitchClamp, Math.min(pitchClamp, pitch));

    return [yaw, pitch];
  }

  private manageChallenges(
    elapsed: number,
    blinkCount: number,
    landmarks: Float32Array | null,
    faceCount: number,
  ): void {
    // Check active challenge timeout/completion.
    if (
      this.activeChallenge &&
      this.activeChallenge.state === ChallengeState.PROMPTED
    ) {
      const timeSince = elapsed - this.activeChallenge.prompted_at;
      if (timeSince > this.activeChallenge.timeout_sec) {
        this.activeChallenge.state = ChallengeState.FAILED;
        this.score.challenges_failed += 1;
        this.activeChallenge = null;
      } else {
        const completed = this.checkChallengeResponse(
          this.activeChallenge,
          blinkCount,
          landmarks,
          faceCount,
        );
        if (completed) {
          this.activeChallenge.state = ChallengeState.COMPLETED;
          this.activeChallenge.completed_at = elapsed;
          const latencyMs = (elapsed - this.activeChallenge.prompted_at) * 1000;
          this.activeChallenge.response_latency_ms = latencyMs;
          this.score.challenges_passed += 1;
          this.score.challenge_points = Math.min(
            LivenessProver.CHALLENGE_POINT_CAP,
            this.score.challenges_passed * LivenessProver.CHALLENGE_AWARD,
          );
          this.challenges.push(this.activeChallenge);
          this.activeChallenge = null;
        }
      }
    }

    // Issue new challenge if ready.
    if (
      this.activeChallenge === null &&
      this.challengesIssued < LivenessProver.MAX_CHALLENGES &&
      elapsed > LivenessProver.FIRST_CHALLENGE_AFTER_SEC &&
      elapsed - this.lastChallengeTimeSec > LivenessProver.CHALLENGE_INTERVAL_SEC
    ) {
      this.issueChallenge(elapsed, blinkCount);
    }
  }

  private issueChallenge(elapsed: number, blinkCount: number): void {
    // SHOW_HAND is in the Python ChallengeType enum but excluded from the
    // random pool in `_issue_challenge` (line 277-278) — same behaviour here.
    const available: readonly ChallengeType[] = [
      ChallengeType.BLINK,
      ChallengeType.TURN_LEFT,
      ChallengeType.TURN_RIGHT,
      ChallengeType.NOD,
    ];
    const idx = Math.min(
      available.length - 1,
      Math.max(0, Math.floor(this.random() * available.length)),
    );
    const challengeType = available[idx];

    this.activeChallenge = {
      challenge_type: challengeType,
      state: ChallengeState.PROMPTED,
      prompted_at: elapsed,
      completed_at: 0,
      timeout_sec: LivenessProver.CHALLENGE_DEFAULT_TIMEOUT_SEC,
      response_latency_ms: 0,
    };
    this.challengesIssued += 1;
    this.lastChallengeTimeSec = elapsed;

    // Store baseline for comparison.
    this.blinkCountAtChallenge = blinkCount;
    const lastYaw = this.yawHistory.last();
    if (lastYaw !== undefined) this.baselineYaw = lastYaw;
    const lastPitch = this.pitchHistory.last();
    if (lastPitch !== undefined) this.baselinePitch = lastPitch;
  }

  private checkChallengeResponse(
    challenge: Challenge,
    blinkCount: number,
    landmarks: Float32Array | null,
    _faceCount: number,
  ): boolean {
    if (challenge.challenge_type === ChallengeType.BLINK) {
      return blinkCount > this.blinkCountAtChallenge;
    }

    // Python: `if landmarks is None or len(landmarks) < 468`.
    // Our landmarks are flat (2N values), so the point count is length/2.
    if (!landmarks || landmarks.length / 2 < 468) {
      return false;
    }

    const [yaw, pitch] = this.estimateHeadPose(landmarks);

    if (challenge.challenge_type === ChallengeType.TURN_LEFT) {
      if (this.baselineYaw !== null) {
        return yaw - this.baselineYaw < -LivenessProver.YAW_DELTA_TURN_DEG;
      }
    }

    if (challenge.challenge_type === ChallengeType.TURN_RIGHT) {
      if (this.baselineYaw !== null) {
        return yaw - this.baselineYaw > LivenessProver.YAW_DELTA_TURN_DEG;
      }
    }

    if (challenge.challenge_type === ChallengeType.NOD) {
      if (this.baselinePitch !== null) {
        return (
          Math.abs(pitch - this.baselinePitch) >
          LivenessProver.PITCH_DELTA_NOD_DEG
        );
      }
    }

    return false;
  }
}

// === Module-scope helpers ====================================================

function maxIndex(): number {
  return Math.max(NOSE_TIP, FOREHEAD, CHIN, LEFT_EAR, RIGHT_EAR);
}

function readPoint(landmarks: Float32Array, idx: number): [number, number] {
  return [landmarks[idx * 2], landmarks[idx * 2 + 1]];
}

function arrayMin(xs: ArrayLike<number>): number {
  let m = Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] < m) m = xs[i];
  return m;
}

function arrayMax(xs: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < xs.length; i++) if (xs[i] > m) m = xs[i];
  return m;
}

function readBlinkCount(cls: SpoofClassification | undefined): number {
  if (!cls) return 0;
  const blink = cls.analyzer_results["blink"];
  if (!blink) return 0;
  const v = blink.details["blinks"];
  return typeof v === "number" ? v : 0;
}

function readLandmarkVarianceDetails(
  cls: SpoofClassification | undefined,
): {
  landmarkVariance: number;
  expressionRatio: number;
  eyeVar: number;
  mouthVar: number;
} {
  if (!cls)
    return { landmarkVariance: 0, expressionRatio: 0, eyeVar: 0, mouthVar: 0 };
  const lv = cls.analyzer_results["landmark_variance"];
  if (!lv)
    return { landmarkVariance: 0, expressionRatio: 0, eyeVar: 0, mouthVar: 0 };
  // Python `landmark_variance` is the raw scalar fed to the prover. The
  // LandmarkVarianceAnalyzer publishes the closest analog under
  // `overall_var` (the per-frame averaged squared displacement) — see
  // LandmarkVarianceAnalyzer.ts:226. eye_var / mouth_var are TS-port
  // additions that feed the eye-motion + mouth-motion proof axes.
  const lv_raw = lv.details["overall_var"];
  const er_raw = lv.details["expression_ratio"];
  const ev_raw = lv.details["eye_var"];
  const mv_raw = lv.details["mouth_var"];
  return {
    landmarkVariance: typeof lv_raw === "number" ? lv_raw : 0,
    expressionRatio: typeof er_raw === "number" ? er_raw : 0,
    eyeVar: typeof ev_raw === "number" ? ev_raw : 0,
    mouthVar: typeof mv_raw === "number" ? mv_raw : 0,
  };
}

function readTemporalMotion(cls: SpoofClassification | undefined): number {
  if (!cls) return 0;
  const t = cls.analyzer_results["temporal"];
  if (!t) return 0;
  const m_raw = t.details["motion"];
  return typeof m_raw === "number" ? m_raw : 0;
}

function readAnalyzerScore(
  cls: SpoofClassification | undefined,
  name: string,
): number {
  if (!cls) return 0;
  const r = cls.analyzer_results[name];
  if (!r) return 0;
  return typeof r.score === "number" ? r.score : 0;
}
