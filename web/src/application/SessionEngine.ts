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
// Phase 1 deviations from the Python source:
//   * LivenessProver integration is deferred to Phase 5; the boolean
//     `prover_live` is replaced by an analyzer-fusion-only decision. The
//     `_check_minifasnet_instability` and `_check_motion_naturalness`
//     incident detectors are preserved.
//   * `_pipeline_analyzers` (used to extract `_last_landmarks` from a Blink
//     instance) is omitted — landmarks now ride on FaceROI directly.
//   * np.std / np.var / np.sqrt are inlined as TS helpers below.
//
// Constants are preserved verbatim (WARMUP_FRAMES = 30, etc.).

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

  constructor(options: SessionEngineOptions = {}) {
    this.sessionId =
      options.sessionId ?? `session_${Math.floor(Date.now() / 1000)}`;
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
  }

  /** Ingest a frame analysis into the session. Called every frame. */
  ingest(analysis: FrameAnalysis): void {
    this.frameCount += 1;
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
    } else {
      this.faceMissingFrames += 1;
      if (
        this.faceMissingFrames >
          SessionEngine.FACE_MISSING_ALERT_SEC * 30 &&
        elapsed > 5.0
      ) {
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
    }
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

    // Only count "no blink" time once the face has been present long enough
    // that we *should* have seen a blink. The face-present clock is the
    // ingest-frame counter; convert to seconds at the session FPS.
    const facePresentSec = this.facePresentCount / 30.0;
    if (facePresentSec < SessionEngine.NO_BLINK_ALERT_SEC) return;

    const sinceBlink = elapsed - this.lastBlinkObservedAt;
    if (sinceBlink < SessionEngine.NO_BLINK_ALERT_SEC) return;

    // Throttle: once we've raised the alert, don't repeat for another window.
    if (
      elapsed - this.lastNoBlinkIncidentAt <
      SessionEngine.NO_BLINK_ALERT_SEC
    )
      return;

    this.lastNoBlinkIncidentAt = elapsed;
    this.addIncident(
      frame_id,
      Severity.HIGH,
      SpoofCategory.STATIC_IMAGE,
      `No blink for ${sinceBlink.toFixed(0)}s — printed or static-image attack suspected`,
      {
        elapsed_sec: round(elapsed, 1),
        face_present_sec: round(facePresentSec, 1),
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

    // Phase-1 simplified decision (no LivenessProver yet):
    //   live IFF analyzer fusion says real AND incidents < 3.
    const incidentOverride = this.incidents.length >= 3;
    const isLive = adjustedReal > 0.45 && !incidentOverride;

    const confidence = Math.min(
      1.0,
      dataConfidence * (0.5 + 0.4 * Math.max(0, adjustedReal - 0.3)),
    );

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
      dominant_threat: isLive ? null : dominantThreat,
      category_scores: categoryScores,
      incidents: this.incidents.slice(),
      session_duration_sec: elapsed,
      frames_analyzed: this.frameCount,
      face_detected_ratio: faceRatio,
      blink_count: blinkCount,
      estimated_bpm: estimatedBpm,
      identity_changes: 0,
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
