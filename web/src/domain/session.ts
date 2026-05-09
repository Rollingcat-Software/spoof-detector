// Port of src/domain/session.py
// Session-state datatypes: SessionState, Severity, Incident, SessionVerdict,
// TemporalSignals. The Python `deque(maxlen=N)` becomes a thin TS class with
// an `append()` and a fixed maximum length.

import {
  ALL_SPOOF_CATEGORIES,
  SpoofCategory,
  SpoofClassification,
} from "./models";

/** Session lifecycle states. */
export enum SessionState {
  WARMING_UP = "warming_up",
  ANALYZING = "analyzing",
  CONCLUDED = "concluded",
}

/** Spoof detection severity for incidents. */
export enum Severity {
  LOW = "low",
  MEDIUM = "medium",
  HIGH = "high",
  CRITICAL = "critical",
}

/** A detected spoofing incident within a session. */
export interface Incident {
  /** Seconds since session start. */
  timestamp: number;
  frame_id: number;
  severity: Severity;
  category: SpoofCategory;
  /** 0-1. */
  confidence: number;
  description: string;
  evidence: Record<string, unknown>;
}

/** Final or interim session verdict. */
export interface SessionVerdict {
  is_live: boolean;
  /** 0-1, how confident are we. */
  confidence: number;
  /** Most likely spoof type if not live. */
  dominant_threat: SpoofCategory | null;
  /** Accumulated evidence per category. */
  category_scores: Record<SpoofCategory, number>;
  /** Timeline of detected incidents. */
  incidents: Incident[];
  session_duration_sec: number;
  frames_analyzed: number;
  /** % of frames where face was present. */
  face_detected_ratio: number;
  blink_count: number;
  /** From rPPG, null if not enough data. */
  estimated_bpm: number | null;
  /** Number of suspected identity switches. */
  identity_changes: number;
  /** Human-readable one-line summary. */
  summary: string;
}

/** Build a one-line summary string (mirrors SessionVerdict.summary in Python). */
export function buildVerdictSummary(
  v: Omit<SessionVerdict, "summary">,
): string {
  const verdict = v.is_live ? "LIVE" : "SPOOF";
  const threat = v.dominant_threat ? ` (${v.dominant_threat})` : "";
  const conf = Math.round(v.confidence * 100);
  return (
    `${verdict}${threat} | conf=${conf}% | ` +
    `${v.session_duration_sec.toFixed(1)}s | ${v.frames_analyzed} frames | ` +
    `blinks=${v.blink_count} | incidents=${v.incidents.length}`
  );
}

/** Bounded ring-buffer (TS analog of Python `collections.deque(maxlen=N)`). */
export class RingBuffer<T> {
  private readonly buf: T[] = [];
  constructor(public readonly maxLen: number) {}

  append(value: T): void {
    if (this.buf.length >= this.maxLen) this.buf.shift();
    this.buf.push(value);
  }

  get length(): number {
    return this.buf.length;
  }

  toArray(): T[] {
    return this.buf.slice();
  }

  last(): T | undefined {
    return this.buf[this.buf.length - 1];
  }

  clear(): void {
    this.buf.length = 0;
  }
}

/** Accumulated temporal signals for a tracked face. */
export class TemporalSignals {
  blink_count = 0;
  last_blink_time = 0.0;
  ear_history = new RingBuffer<number>(90);          // 3s at 30fps
  green_channel_history = new RingBuffer<number>(300); // 10s
  estimated_bpm: number | null = null;
  pulse_confidence = 0.0;
  position_history = new RingBuffer<[number, number, number]>(150); // 5s
  motion_naturalness = 0.5;
  pose_history = new RingBuffer<[number, number, number]>(300); // 10s
  pose_variance = 0.0;
  frame_verdicts = new RingBuffer<SpoofClassification>(900); // 30s
  minifasnet_scores = new RingBuffer<number>(300);

  constructor(public readonly face_id: number) {}

  /** Estimated blink rate. Normal: 15-20/min. */
  get blink_rate_per_min(): number {
    if (this.frame_verdicts.length === 0) return 0;
    const duration_sec = this.frame_verdicts.length / 30.0;
    if (duration_sec < 2.0) return 0;
    return this.blink_count / (duration_sec / 60.0);
  }

  get avg_minifasnet(): number {
    if (this.minifasnet_scores.length === 0) return 50.0;
    const arr = this.minifasnet_scores.toArray();
    let s = 0;
    for (const x of arr) s += x;
    return s / arr.length;
  }
}

/** Convenience: a fully-uniform category-score record. */
export function uniformCategoryScores(): Record<SpoofCategory, number> {
  const u = 1.0 / ALL_SPOOF_CATEGORIES.length;
  const out: Record<SpoofCategory, number> = {} as Record<SpoofCategory, number>;
  for (const cat of ALL_SPOOF_CATEGORIES) out[cat] = u;
  return out;
}
