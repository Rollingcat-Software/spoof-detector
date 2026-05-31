// Port of src/application/session_engine.py
//
// Accumulates per-frame evidence into a session-level verdict that improves
// over time. Architecture:
//
//   WARMING_UP (frames 0-29) → ANALYZING (frame 30+) → CONCLUDED
//
// Peak-sensitive verdict aggregator: any sustained spoof window flags the
// session even if the majority of frames are live (critical for proctoring).
//
// Decision rule:
//
//   base = adjusted_real > 0.45 AND NOT incident_override (>=3 incidents)
//   is_live = base AND (NOT requireProverLive OR prover.isProvenLive)
//
// Python session_engine.py:400 always ANDs prover_live; we make that AND-term
// opt-in (`requireProverLive`) because the prover only reaches its 60-point
// threshold when the consumer surfaces active challenges (head-turn, nod) to
// the user. Consumers that don't wire challenge UI would otherwise reject
// every borderline live user. amispoof currently doesn't surface challenges
// → default false. verify.fivucsas.com, which does wire challenges, sets it
// to true to get the full Python "guilty until proven innocent" semantics.
//
// The prover is *always* fed (when injected) and its score contributes to
// `confidence` regardless of the gate setting — a free positive signal.
//
// Peak-sensitivity (the property that motivated the now-removed
// `verdictLockedSpoof` latch) is preserved entirely by:
//   * worstWindowReal → blendedReal → adjustedReal (sticky-low across the
//     1800-sample ring buffer; long enough for a multi-minute session)
//   * the ≥3-incident override (incidents are append-only within a session)
// Both are monotonic in evidence and survive on their own.
//
// UI flicker smoothing (the 250ms SPOOF→LIVE→SPOOF jitter) belongs at the
// display layer — apply hysteresis on the displayed verdict, not on the
// domain state machine.

import { LivenessProver } from "./LivenessProver";
import {
  ALL_SPOOF_CATEGORIES,
  FrameAnalysis,
  SpoofCategory,
  SpoofClassification,
} from "../domain/models";
import {
  buildVerdictSummary,
  Incident,
  RingBuffer,
  SessionState,
  SessionVerdict,
  Severity,
  TemporalSignals,
  uniformCategoryScores,
} from "../domain/session";

export interface SessionEngineOptions {
  sessionId?: string;
  /**
   * Optional liveness prover. When provided, the engine forwards
   * `start()`, `ingest()`, and `reset()` to it and includes its score
   * in the verdict confidence calculation.
   */
  prover?: LivenessProver | null;
  /**
   * When true, the prover's isProvenLive gate (score ≥ 60) is ANDed into
   * the live verdict — full Python session_engine.py semantics. Requires
   * the consumer to surface `detector.getProof().active_challenge` to the
   * user; otherwise borderline live sessions without spontaneous head
   * motion will be rejected. Default false — fusion-only verdict, prover
   * still feeds confidence. Has no effect when `prover` is null.
   */
  requireProverLive?: boolean;
}

export class SessionEngine {
  // === Thresholds — verbatim port of Python class constants ===
  static readonly WARMUP_FRAMES = 30;          // 1 second at 30fps
  static readonly MIN_VERDICT_FRAMES = 60;     // 2 seconds minimum for any verdict
  static readonly BLINK_EAR_THRESHOLD = 0.21;  // EAR below this = eye closed
  static readonly BLINK_CONSECUTIVE = 3;       // Frames eye must be closed
  static readonly NO_BLINK_ALERT_SEC = 15.0;
  static readonly FACE_MISSING_ALERT_SEC = 5.0;
  static readonly IDENTITY_CHANGE_THRESHOLD = 0.35;
  // Planar-print veto (2026-05-24). A planarity score below this, while the
  // analyzer is actively measuring (head/print rotating), means a flat
  // surface — printed photo or screen. Throttled so a sustained flat
  // presentation accrues the >=3 incidents that flip the verdict to SPOOF.
  // Calibrated live 2026-05-24: a tilted printed photo measures 21-31 (the
  // MediaPipe 3D-model fit keeps it off a perfect 0), a real rotating face
  // measures ~100 — so 45 catches the whole print range with wide margin
  // from a genuine face. The analyzer abstains (measured:false) below its
  // rotation gate, so a still face is never scored against this.
  static readonly PLANAR_SPOOF_SCORE = 45;
  static readonly PLANAR_MIN_ELAPSED_SEC = 3.0;
  static readonly PLANAR_INCIDENT_THROTTLE_SEC = 2.5;
  // Texture-collapse VIDEO_REPLAY veto (2026-05-31). On in-house frame-log
  // data (3 LIVE sessions / 3 SPOOF sessions including both pre-recorded
  // replay and a live video-call from a Xiaomi 14T Pro held close to the
  // camera), texture.texture_score collapsed cleanly:
  //   LIVE  mean 62.5  std 23.9
  //   SPOOF mean 13.6  std 13.1   (AUC 0.924, d-prime 2.53)
  // Threshold 25 sits >2.7 SPOOF-std above the SPOOF mean (catches >99% of
  // SPOOF frames) and ~1.6 LIVE-std below the LIVE mean (the rare LIVE
  // frame that dips below — bad lighting/blur — gets at most one incident
  // because of the throttle, never the >=3 needed to override).
  // texture.texture_score is the analyzer's Laplacian-variance sub-feature,
  // exposed via cls.analyzer_results["texture"].details["texture_score"].
  // The top-line texture.score blends color + frequency + drift so the
  // cliff (100 → 16 on SPOOF) gets averaged into a much smaller drop
  // (73 → 56) the fuser can't act on at any reasonable weight.
  static readonly TEXTURE_SCORE_SPOOF_THRESHOLD = 25;
  static readonly TEXTURE_MIN_ELAPSED_SEC = 3.0;
  static readonly TEXTURE_INCIDENT_THROTTLE_SEC = 2.5;
  // Capture-quality floor (2026-05-24). A would-be-LIVE session whose capture
  // quality is too poor (dark / occluded / no-face frames) is downgraded to
  // UNCERTAIN (prompt a re-capture) rather than confidently classified LIVE.
  // A genuine spoof (real spoof evidence) is NEVER downgraded — it stays SPOOF.
  // illuminationScore is 0-1 from the FaceUsabilityGate (normal lit face ~0.8).
  static readonly QUALITY_MIN_SAMPLES = 15;
  static readonly QUALITY_USABLE_RATIO = 0.5;
  static readonly QUALITY_ILLUM_FLOOR = 0.35;

  private readonly sessionId: string;
  private state: SessionState = SessionState.WARMING_UP;
  private startTime = 0;
  private frameCount = 0;
  private facePresentCount = 0;

  private signals: Map<number, TemporalSignals> = new Map();
  private primaryFaceId: number | null = null;

  private incidents: Incident[] = [];
  private categoryEvidence: Record<SpoofCategory, RingBuffer<number>>;

  private recentVerdicts = new RingBuffer<SpoofClassification>(300); // 10s

  private faceMissingFrames = 0;
  private consecutiveSpoofFrames = 0;
  private lastBlinkCount = 0;
  private lastBlinkObservedAt = 0;
  private lastNoBlinkIncidentAt = -Infinity;
  private lastPlanarIncidentAt = -Infinity;
  private lastTextureIncidentAt = -Infinity;
  private lastFaceMissingIncidentAt = -Infinity;
  private qualitySamples = new RingBuffer<{ usable: boolean; illum: number }>(90);

  private readonly prover: LivenessProver | null;
  private readonly requireProverLive: boolean;

  constructor(options: SessionEngineOptions = {}) {
    this.sessionId =
      options.sessionId ?? `session_${Math.floor(Date.now() / 1000)}`;
    this.prover = options.prover ?? null;
    this.requireProverLive = options.requireProverLive === true;
    this.categoryEvidence = {} as Record<SpoofCategory, RingBuffer<number>>;
    for (const cat of ALL_SPOOF_CATEGORIES) {
      // 60 seconds of evidence at 30 fps
      this.categoryEvidence[cat] = new RingBuffer<number>(1800);
    }
  }

  get currentState(): SessionState {
    return this.state;
  }

  get id(): string {
    return this.sessionId;
  }

  get elapsedSec(): number {
    if (this.startTime === 0) return 0;
    return (Date.now() - this.startTime) / 1000;
  }

  /** Start the session clock. */
  start(): void {
    this.startTime = Date.now();
    this.state = SessionState.WARMING_UP;
    this.prover?.start();
  }

  /** Reset session state but keep the engine instance reusable. */
  reset(): void {
    this.state = SessionState.WARMING_UP;
    this.startTime = 0;
    this.frameCount = 0;
    this.facePresentCount = 0;
    this.signals.clear();
    this.primaryFaceId = null;
    this.incidents = [];
    for (const cat of ALL_SPOOF_CATEGORIES) {
      this.categoryEvidence[cat].clear();
    }
    this.recentVerdicts.clear();
    this.faceMissingFrames = 0;
    this.consecutiveSpoofFrames = 0;
    this.lastBlinkCount = 0;
    this.lastBlinkObservedAt = 0;
    this.lastNoBlinkIncidentAt = -Infinity;
    this.lastPlanarIncidentAt = -Infinity;
    this.lastTextureIncidentAt = -Infinity;
    this.lastFaceMissingIncidentAt = -Infinity;
    this.qualitySamples.clear();
    this.prover?.reset();
  }

  /** Ingest a frame analysis into the session. Called every frame. */
  ingest(analysis: FrameAnalysis): void {
    this.frameCount += 1;
    this.prover?.ingest(analysis);
    const gr = analysis.gate_result;
    if (gr) {
      this.qualitySamples.append({ usable: gr.usable, illum: gr.illuminationScore });
    }
    const elapsed = this.elapsedSec;

    if (
      this.state === SessionState.WARMING_UP &&
      this.frameCount >= SessionEngine.WARMUP_FRAMES
    ) {
      this.state = SessionState.ANALYZING;
    }

    // Track face presence
    if (analysis.faces.length > 0) {
      this.facePresentCount += 1;
      this.faceMissingFrames = 0;
      // Re-arm: the next missing episode should raise its own incident.
      this.lastFaceMissingIncidentAt = -Infinity;
    } else {
      this.faceMissingFrames += 1;
      // Edge/throttled: raise at most one face-missing incident per
      // FACE_MISSING_ALERT_SEC, NOT once per frame. Without this the detector
      // fired every frame — a 30 s absence at 50 fps produced ~1,500 duplicate
      // incidents and instantly tripped the >=3-incident override (a real
      // visitor whose camera went dark was flagged SPOOF). Mirrors the
      // no-blink/planar throttles above.
      if (
        this.faceMissingFrames >
          SessionEngine.FACE_MISSING_ALERT_SEC * 30 &&
        elapsed > 5.0 &&
        elapsed - this.lastFaceMissingIncidentAt >=
          SessionEngine.FACE_MISSING_ALERT_SEC
      ) {
        this.lastFaceMissingIncidentAt = elapsed;
        this.addIncident(
          analysis.frame_id,
          Severity.MEDIUM,
          SpoofCategory.REAL,
          `Face missing for ${(this.faceMissingFrames / 30).toFixed(0)}s`,
          { missing_frames: this.faceMissingFrames },
        );
      }
    }

    for (const face of analysis.faces) {
      const fid = face.face_id;
      const cls = analysis.classifications[fid];
      if (!cls) continue;

      let signals = this.signals.get(fid);
      if (!signals) {
        signals = new TemporalSignals(fid);
        this.signals.set(fid, signals);
      }

      if (this.primaryFaceId === null) this.primaryFaceId = fid;

      signals.frame_verdicts.append(cls);

      const mfn = cls.analyzer_results["minifasnet"];
      if (mfn) signals.minifasnet_scores.append(mfn.score);

      signals.position_history.append([
        face.bbox.center[0],
        face.bbox.center[1],
        face.bbox.area,
      ]);

      for (const cat of ALL_SPOOF_CATEGORIES) {
        const p = cls.probabilities[cat] ?? 0;
        this.categoryEvidence[cat].append(p);
      }

      this.recentVerdicts.append(cls);

      // === Incident Detection ===
      this.checkSpoofIncident(cls, analysis.frame_id, elapsed);
      this.checkMotionNaturalness(signals, analysis.frame_id, elapsed);
      this.checkMiniFasNetInstability(signals, analysis.frame_id, elapsed);
      this.checkNoBlink(cls, analysis.frame_id, elapsed);
      this.checkPlanarPrint(cls, analysis.frame_id, elapsed);
      this.checkTextureCollapseReplay(cls, analysis.frame_id, elapsed);
    }
  }

  /**
   * Raise a STATIC_IMAGE incident when the planarity analyzer reports a flat
   * surface while it is actively measuring (head/print rotating). A real 3D
   * face yields a HIGH planarity score under rotation, so this never fires on
   * genuine users; the throttle lets a sustained flat presentation accrue the
   * >=3 incidents that flip getVerdict() to SPOOF. This is the lever that
   * overcomes a MiniFASNet fooled by a sharp, frame-filling printed photo —
   * the 2026-05-24 amispoof false-accept (print → LIVE 90%, MiniFASNet 100).
   */
  private checkPlanarPrint(
    cls: SpoofClassification,
    frame_id: number,
    elapsed: number,
  ): void {
    const planarity = cls.analyzer_results["planarity"];
    if (!planarity) return;
    // measured:false means there wasn't enough rotation to judge planarity —
    // never penalise those frames (keeps the false-reject rate untouched).
    if (planarity.details["measured"] !== true) return;
    if (planarity.score >= SessionEngine.PLANAR_SPOOF_SCORE) return;
    if (elapsed < SessionEngine.PLANAR_MIN_ELAPSED_SEC) return;
    if (
      elapsed - this.lastPlanarIncidentAt <
      SessionEngine.PLANAR_INCIDENT_THROTTLE_SEC
    ) {
      return;
    }

    this.lastPlanarIncidentAt = elapsed;
    this.addIncident(
      frame_id,
      Severity.HIGH,
      SpoofCategory.STATIC_IMAGE,
      `Flat surface under rotation (planarity=${Math.round(
        planarity.score,
      )}) — printed-photo / screen attack suspected`,
      {
        planarity_score: round(planarity.score, 1),
        residual_norm: planarity.details["residual_norm"] ?? null,
      },
    );
  }

  /**
   * Raise a VIDEO_REPLAY incident when the texture analyzer's Laplacian-
   * variance sub-feature (`texture_score`) collapses — the signature of a
   * face being shown on a phone or laptop screen. Catches the attack the
   * `checkPlanarPrint` veto misses: a held-close phone playing a video
   * (the user's face never rotates, so planarity stays in `measured:false`),
   * and a live video-call on a phone (no playback artifacts at all).
   *
   * Threshold is calibrated against the 2026-05-31 in-house dataset (3 LIVE
   * sessions / 3 SPOOF sessions including pre-recorded replay AND a live
   * video-call):
   *   LIVE  texture_score mean 62.5 std 23.9
   *   SPOOF texture_score mean 13.6 std 13.1   (cross-session AUC 0.924)
   * `< 25` catches >99% of SPOOF frames and only the occasional dark/blurry
   * LIVE frame; the throttle then absorbs that single incident without ever
   * crossing the >=3 threshold needed to flip getVerdict() to SPOOF.
   *
   * Unlike the top-line `texture.score` (which mixes color + frequency +
   * drift and collapses the LIVE→SPOOF gap from 49 points to 15), the
   * sub-feature exposes the actual physics-grounded signal — pixel-level
   * detail loss when a face is being rendered through an LCD's subpixel
   * grid + the camera's own Bayer mosaic.
   */
  private checkTextureCollapseReplay(
    cls: SpoofClassification,
    frame_id: number,
    elapsed: number,
  ): void {
    const texture = cls.analyzer_results["texture"];
    if (!texture) return;
    const textureScore = texture.details["texture_score"];
    if (typeof textureScore !== "number") return;
    if (textureScore >= SessionEngine.TEXTURE_SCORE_SPOOF_THRESHOLD) return;
    if (elapsed < SessionEngine.TEXTURE_MIN_ELAPSED_SEC) return;
    if (
      elapsed - this.lastTextureIncidentAt <
      SessionEngine.TEXTURE_INCIDENT_THROTTLE_SEC
    ) {
      return;
    }

    this.lastTextureIncidentAt = elapsed;
    this.addIncident(
      frame_id,
      Severity.HIGH,
      SpoofCategory.VIDEO_REPLAY,
      `Texture collapse (texture_score=${Math.round(
        textureScore,
      )}) — face rendered through a screen (replay / video-call) suspected`,
      {
        texture_score: round(textureScore, 1),
        threshold: SessionEngine.TEXTURE_SCORE_SPOOF_THRESHOLD,
      },
    );
  }

  /**
   * Raise a STATIC_IMAGE incident if the face has been present for
   * NO_BLINK_ALERT_SEC without any blink. A printed photo never blinks;
   * a live human blinks every 4–6 seconds on average. Repeats every
   * NO_BLINK_ALERT_SEC so a long-running spoof accumulates incidents
   * fast enough to trigger the >=3-incident override in getVerdict().
   */
  private checkNoBlink(
    cls: SpoofClassification,
    frame_id: number,
    elapsed: number,
  ): void {
    const blink = cls.analyzer_results["blink"];
    const currentBlinks =
      blink && typeof blink.details["blinks"] === "number"
        ? (blink.details["blinks"] as number)
        : 0;

    if (currentBlinks > this.lastBlinkCount) {
      this.lastBlinkCount = currentBlinks;
      this.lastBlinkObservedAt = elapsed;
      return;
    }

    // Liveness contract: a printed photo NEVER blinks, ever. A live person
    // produces at least one blink within ~15s under normal conditions. So
    // once we've observed ANY blink in this session, we know the subject
    // is alive — stop firing no-blink incidents on top of normal slow
    // blinking. This was a real bug: a 78-second session with 4 real
    // blinks (mobile camera at 4 fps drops some blinks) was flipping to
    // SPOOF because each gap between detected blinks crossed 15s.
    if (this.lastBlinkCount > 0) return;

    // checkNoBlink only runs when a face is in this frame (caller loop
    // iterates over analysis.faces), so face-presence is implicit. Use
    // wall-clock elapsed time directly.
    //
    // FPS-aware threshold: the BlinkAnalyzer can only catch a 100-ms blink
    // if a frame lands inside that window. At 30 fps detection is ~100%,
    // at 7 fps it drops to ~70%, and the statistical wait for the first
    // *detected* blink scales inversely with fps. NO_BLINK_ALERT_SEC is
    // calibrated for 30 fps; on slow mobile cameras we scale it up so a
    // real user doesn't accumulate a "static-image attack suspected"
    // incident just because the camera dropped frames covering their
    // first blink. Bug surfaced on a Chrome-mobile run at 6.7 fps where
    // the user blinked 22 times in 143s but had a 15-s alert anyway.
    const fpsAwareAlertSec = this.fpsAwareNoBlinkAlertSec();
    if (elapsed < fpsAwareAlertSec) return;

    const sinceBlink = elapsed - this.lastBlinkObservedAt;
    if (sinceBlink < fpsAwareAlertSec) return;

    // Throttle: once we've raised the alert, re-raise every 5s so a steady
    // print attack accumulates enough incidents to trip the >=3 override
    // around the 25-second mark (15s first alert, 20s second, 25s third).
    if (elapsed - this.lastNoBlinkIncidentAt < 5.0) return;

    this.lastNoBlinkIncidentAt = elapsed;
    this.addIncident(
      frame_id,
      Severity.HIGH,
      SpoofCategory.STATIC_IMAGE,
      `No blink for ${sinceBlink.toFixed(0)}s — printed or static-image attack suspected`,
      {
        elapsed_sec: round(elapsed, 1),
        seconds_since_blink: round(sinceBlink, 1),
      },
    );
  }

  private checkSpoofIncident(
    cls: SpoofClassification,
    frame_id: number,
    elapsed: number,
  ): void {
    const pReal = cls.probabilities[SpoofCategory.REAL] ?? 1.0;
    if (pReal < 0.45) {
      let severity: Severity;
      if (pReal < 0.2) severity = Severity.HIGH;
      else if (pReal < 0.35) severity = Severity.MEDIUM;
      else severity = Severity.LOW;

      let dominantSpoof: SpoofCategory = SpoofCategory.STATIC_IMAGE;
      let dominantP = -Infinity;
      for (const cat of ALL_SPOOF_CATEGORIES) {
        if (cat === SpoofCategory.REAL) continue;
        const p = cls.probabilities[cat] ?? 0;
        if (p > dominantP) {
          dominantP = p;
          dominantSpoof = cat;
        }
      }

      this.consecutiveSpoofFrames += 1;

      const last = this.incidents[this.incidents.length - 1];
      const shouldLog =
        this.incidents.length === 0 ||
        elapsed - (last?.timestamp ?? 0) > 2.0 ||
        (this.consecutiveSpoofFrames === 15 &&
          (this.incidents.length === 0 || last.severity !== Severity.HIGH));

      if (shouldLog) {
        const burstNote =
          this.consecutiveSpoofFrames > 10
            ? ` (burst: ${this.consecutiveSpoofFrames} frames)`
            : "";
        const pct = Math.round(pReal * 100);
        const evidence: Record<string, unknown> = {
          p_real: round(pReal, 3),
          probabilities: roundProbabilities(cls.probabilities),
          consecutive_spoof_frames: this.consecutiveSpoofFrames,
        };
        this.addIncident(
          frame_id,
          severity,
          dominantSpoof,
          `P(real)=${pct}%, dominant=${dominantSpoof}${burstNote}`,
          evidence,
        );
      }
    } else {
      this.consecutiveSpoofFrames = 0;
    }
  }

  private checkMotionNaturalness(
    signals: TemporalSignals,
    frame_id: number,
    elapsed: number,
  ): void {
    if (signals.position_history.length < 60) return;

    const recent = signals.position_history.toArray().slice(-60);
    const xs = recent.map((p) => p[0]);
    const ys = recent.map((p) => p[1]);
    const areas = recent.map((p) => p[2]);

    const meanArea =
      areas.length > 0 ? areas.reduce((a, b) => a + b, 0) / areas.length : 1;
    const varianceX = variance(xs);
    const varianceY = variance(ys);
    const posStd =
      Math.sqrt(varianceX + varianceY) / Math.max(Math.sqrt(meanArea), 1);

    const last = this.incidents[this.incidents.length - 1];
    if (posStd < 0.0001) {
      signals.motion_naturalness = 0.1;
      if (
        elapsed > 3.0 &&
        (this.incidents.length === 0 ||
          elapsed - (last?.timestamp ?? 0) > 5.0)
      ) {
        this.addIncident(
          frame_id,
          Severity.MEDIUM,
          SpoofCategory.STATIC_IMAGE,
          `Face is unnaturally static (motion_std=${posStd.toFixed(6)})`,
          { pos_std: posStd },
        );
      }
    } else if (posStd < 0.001) {
      signals.motion_naturalness = 0.4;
    } else {
      signals.motion_naturalness = Math.min(1.0, 0.5 + posStd * 100);
    }
  }

  private checkMiniFasNetInstability(
    signals: TemporalSignals,
    frame_id: number,
    elapsed: number,
  ): void {
    if (signals.minifasnet_scores.length < 20) return;

    const recent = signals.minifasnet_scores.toArray().slice(-30);
    const std = Math.sqrt(variance(recent));
    const mean = recent.reduce((a, b) => a + b, 0) / recent.length;

    const last = this.incidents[this.incidents.length - 1];
    if (std > 25 && mean < 80) {
      if (
        this.incidents.length === 0 ||
        elapsed - (last?.timestamp ?? 0) > 3.0
      ) {
        this.addIncident(
          frame_id,
          Severity.MEDIUM,
          SpoofCategory.STATIC_IMAGE,
          `MiniFASNet unstable: mean=${mean.toFixed(0)}, std=${std.toFixed(0)} (screen-like oscillation)`,
          { mfn_mean: round(mean, 1), mfn_std: round(std, 1) },
        );
      }
    }
  }

  private addIncident(
    frame_id: number,
    severity: Severity,
    category: SpoofCategory,
    description: string,
    evidence: Record<string, unknown>,
  ): void {
    this.incidents.push({
      timestamp: this.elapsedSec,
      frame_id,
      severity,
      category,
      confidence: 0.0,
      description,
      evidence,
    });
  }

  /** Get current session verdict based on accumulated evidence. */
  getVerdict(): SessionVerdict {
    const elapsed = this.elapsedSec;
    const faceRatio = this.facePresentCount / Math.max(this.frameCount, 1);

    const categoryScores = uniformCategoryScores();
    for (const cat of ALL_SPOOF_CATEGORIES) {
      const buf = this.categoryEvidence[cat];
      if (buf.length > 0) {
        const arr = buf.toArray();
        let s = 0;
        for (const x of arr) s += x;
        categoryScores[cat] = s / arr.length;
      }
    }

    const avgReal = categoryScores[SpoofCategory.REAL];

    const realEvidence = this.categoryEvidence[SpoofCategory.REAL].toArray();
    let worstWindowReal = avgReal;
    if (realEvidence.length >= 3) {
      const windowSize = Math.min(5, realEvidence.length);
      for (let i = 0; i + windowSize <= realEvidence.length; i++) {
        let s = 0;
        for (let j = 0; j < windowSize; j++) s += realEvidence[i + j];
        const windowAvg = s / windowSize;
        if (windowAvg < worstWindowReal) worstWindowReal = windowAvg;
      }
    }

    let dominantThreat: SpoofCategory | null = null;
    let dominantP = -Infinity;
    for (const cat of ALL_SPOOF_CATEGORIES) {
      if (cat === SpoofCategory.REAL) continue;
      const p = categoryScores[cat];
      if (p > dominantP) {
        dominantP = p;
        dominantThreat = cat;
      }
    }

    const dataConfidence = Math.min(1.0, this.frameCount / 150.0);
    const temporalBoost = this.computeTemporalConfidence();
    const incidentPenalty = this.computeIncidentPenalty();

    const blendedReal = 0.5 * avgReal + 0.5 * worstWindowReal;

    let adjustedReal = blendedReal * dataConfidence;
    adjustedReal = adjustedReal * (1.0 - incidentPenalty * 0.4);
    adjustedReal += temporalBoost * 0.15;

    // Verdict rule — see file header for design rationale.
    const proverScore = this.prover?.getScore();
    const proverLive =
      this.prover === null || !this.requireProverLive
        ? true
        : (proverScore?.total ?? 0) >= 60;
    const incidentOverride = this.incidents.length >= 3;
    const baseLive = adjustedReal > 0.45 && proverLive && !incidentOverride;
    // Capture-quality floor: a would-be-LIVE poor-quality capture is reported
    // UNCERTAIN (re-capture), never a confident LIVE. A genuine spoof
    // (baseLive false) is NOT downgraded — it stays SPOOF.
    const qualityOk = this.computeQualityOk();
    const qualityUncertain = baseLive && !qualityOk;
    const isLive = baseLive && qualityOk;

    const proverConfidence = proverScore ? proverScore.total / 100.0 : 0;
    let confidence = this.prover
      ? Math.min(
          1.0,
          dataConfidence *
            (0.3 * proverConfidence +
              0.3 +
              0.4 * Math.max(0, adjustedReal - 0.3)),
        )
      : Math.min(
          1.0,
          dataConfidence * (0.5 + 0.4 * Math.max(0, adjustedReal - 0.3)),
        );

    // An uncertain (poor-quality) verdict must read as low-confidence so the
    // surface prompts a re-capture rather than showing a strong number.
    if (qualityUncertain) confidence = Math.min(confidence, 0.3);

    // Blink count from blink analyzer details, if present in latest verdicts.
    let blinkCount = 0;
    let estimatedBpm: number | null = null;
    for (const cls of this.recentVerdicts.toArray()) {
      const blink = cls.analyzer_results["blink"];
      if (blink && typeof blink.details["blinks"] === "number") {
        blinkCount = blink.details["blinks"] as number;
      }
      const rppg = cls.analyzer_results["rppg"];
      if (rppg && typeof rppg.details["bpm"] === "number") {
        estimatedBpm = rppg.details["bpm"] as number;
      }
    }

    const partial: Omit<SessionVerdict, "summary"> = {
      is_live: isLive,
      confidence,
      dominant_threat: isLive || qualityUncertain ? null : dominantThreat,
      category_scores: categoryScores,
      incidents: this.incidents.slice(),
      session_duration_sec: elapsed,
      frames_analyzed: this.frameCount,
      face_detected_ratio: faceRatio,
      blink_count: blinkCount,
      estimated_bpm: estimatedBpm,
      identity_changes: 0,
      quality_uncertain: qualityUncertain,
    };
    return { ...partial, summary: buildVerdictSummary(partial) };
  }

  private computeTemporalConfidence(): number {
    if (this.signals.size === 0) return 0;
    const boosts: number[] = [];
    for (const sig of this.signals.values()) {
      if (sig.frame_verdicts.length < 30) continue;
      boosts.push(sig.motion_naturalness);
    }
    if (boosts.length === 0) return 0;
    let s = 0;
    for (const b of boosts) s += b;
    return s / boosts.length;
  }

  /**
   * Capture-quality floor. Returns false when recent frames were too poor to
   * trust a LIVE verdict (mostly unusable — dark / occluded / no-face — or
   * mean illumination below the floor). Returns true when there isn't enough
   * gate data yet, so quality only ever GATES a confident verdict, never
   * fabricates one.
   */
  private computeQualityOk(): boolean {
    const samples = this.qualitySamples.toArray();
    if (samples.length < SessionEngine.QUALITY_MIN_SAMPLES) return true;
    let usableCount = 0;
    let illumSum = 0;
    for (const s of samples) {
      if (s.usable) usableCount += 1;
      illumSum += s.illum;
    }
    const usableRatio = usableCount / samples.length;
    const meanIllum = illumSum / samples.length;
    return (
      usableRatio >= SessionEngine.QUALITY_USABLE_RATIO &&
      meanIllum >= SessionEngine.QUALITY_ILLUM_FLOOR
    );
  }

  private computeIncidentPenalty(): number {
    if (this.incidents.length === 0) return 0;

    const severityWeights: Record<Severity, number> = {
      [Severity.LOW]: 0.15,
      [Severity.MEDIUM]: 0.4,
      [Severity.HIGH]: 0.8,
      [Severity.CRITICAL]: 1.0,
    };

    let total = 0;
    for (const i of this.incidents) {
      total += severityWeights[i.severity] ?? 0;
    }

    const elapsed = this.elapsedSec;
    if (elapsed > 5.0) {
      const incidentsPerMin = this.incidents.length / (elapsed / 60.0);
      if (incidentsPerMin > 10) total *= 1.5;
      else if (incidentsPerMin > 5) total *= 1.2;
    }

    return Math.min(1.0, total / 2.0);
  }

  /**
   * Scale `NO_BLINK_ALERT_SEC` to the session's measured fps. Below the
   * 15-fps reference the threshold is stretched linearly so a slow camera
   * doesn't false-positive on missed blinks. Clamped at 4× to avoid the
   * alert never firing on a real stuck-photo attack.
   */
  private fpsAwareNoBlinkAlertSec(): number {
    const elapsed = this.elapsedSec;
    if (elapsed < 1.0) return SessionEngine.NO_BLINK_ALERT_SEC;
    const fps = this.frameCount / elapsed;
    const reference = 15.0;
    if (fps >= reference) return SessionEngine.NO_BLINK_ALERT_SEC;
    const stretch = Math.min(4.0, reference / Math.max(fps, 1.0));
    return SessionEngine.NO_BLINK_ALERT_SEC * stretch;
  }

  /** Conclude the session and return the final verdict. */
  conclude(): SessionVerdict {
    this.state = SessionState.CONCLUDED;
    return this.getVerdict();
  }

  /** Get incident timeline for reporting. */
  getTimeline(): Array<{
    time_sec: number;
    frame: number;
    severity: string;
    category: string;
    description: string;
  }> {
    return this.incidents.map((i) => ({
      time_sec: round(i.timestamp, 1),
      frame: i.frame_id,
      severity: i.severity,
      category: i.category,
      description: i.description,
    }));
  }
}

// === Helpers ===

function variance(xs: ArrayLike<number>): number {
  const n = xs.length;
  if (n === 0) return 0;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += xs[i];
  mean /= n;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const d = xs[i] - mean;
    sse += d * d;
  }
  return sse / n;
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}

function roundProbabilities(
  p: Record<SpoofCategory, number>,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const cat of ALL_SPOOF_CATEGORIES) {
    out[cat] = round(p[cat] ?? 0, 3);
  }
  return out;
}
