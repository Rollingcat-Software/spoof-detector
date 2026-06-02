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
  // Texture-collapse VIDEO_REPLAY veto (2026-05-31, revised same day).
  //
  // V1 (#74) fired one VIDEO_REPLAY incident per low-texture frame, throttled
  // at 2.5 s. False-positive on a 344 s LIVE session in lower-than-noon light
  // ("before-sunset/akşamüstü"): only 1% of frames dipped below the threshold,
  // but spread over 344 s that's still ~3 isolated incidents — enough to trip
  // the ≥3-incident SPOOF override.
  //
  // V2 (this revision) switches to a SUSTAINED-RATIO test: at most one
  // incident per evaluation window. The per-session breakdown was the
  // discriminator we missed in V1 — LIVE and SPOOF cleanly separate on the
  // FRACTION of recent frames below threshold, not the absolute count:
  //
  //   Class   tex_mean  ±std  % frames below 25  Notes
  //   LIVE     100      0     0%                 home daylight, short
  //   LIVE      66      8     0%                 room daylight
  //   LIVE      59     24    18%                 room daylight, long
  //   LIVE      56      5     1%                 home daylight
  //   LIVE      53      6     1%                 before-sunset (low light)
  //   SPOOF     38      9     8%                 home daylight (V1 underdetected)
  //   SPOOF      8      2   100%                 room daylight, prerecorded
  //   SPOOF      6      3   100%                 room daylight, videocall
  //   SPOOF     26     10    50%                 before-sunset, prerecorded
  //   SPOOF     17      6    86%                 before-sunset, prerecorded
  //   SPOOF     58     22     8%                 before-sunset, prerecorded
  //   SPOOF     19     12    89%                 before-sunset, prerecorded
  //
  // LIVE caps at 18% (long room-daylight session, still correctly classified).
  // SPOOF starts at 50% (most ≥86%). A 30% sustained-ratio over the most
  // recent ~5 s (30 frames at the typical 6-9 fps amispoof runs at) cleanly
  // separates without false-positives on long low-light LIVE sessions.
  //
  // The window is FRAME-COUNT-based, not seconds, so a slow camera doesn't
  // produce a single-frame window that fires immediately. Min 20 frames
  // before evaluating (about 2-3 s).
  //
  // texture.texture_score is the analyzer's Laplacian-variance sub-feature,
  // exposed via cls.analyzer_results["texture"].details["texture_score"].
  // The top-line texture.score blends color + frequency + drift so the
  // cliff (100 → 16 on SPOOF) gets averaged into a much smaller drop
  // (73 → 56) the fuser can't act on at any reasonable weight.
  static readonly TEXTURE_SCORE_SPOOF_THRESHOLD = 25;
  static readonly TEXTURE_MIN_ELAPSED_SEC = 3.0;
  static readonly TEXTURE_INCIDENT_THROTTLE_SEC = 2.5;
  static readonly TEXTURE_WINDOW_FRAMES = 30;
  static readonly TEXTURE_WINDOW_MIN_FRAMES = 20;
  static readonly TEXTURE_LOW_FRACTION_SPOOF = 0.30;
  // V3 CO-SIGNAL gate (2026-05-31, third revision). A LIVE session captured
  // in twilight ("home sunset") had texture_score mean = 19 — the texture
  // analyzer cannot tell that case apart from a SPOOF (which had mean = 6
  // in the same room). Per-session breakdown showed `screen_replay.skin_score`
  // still discriminates cleanly even at twilight:
  //   LIVE  skin_score range across all light conditions: 0.1 - 27.8
  //   SPOOF skin_score range across all light conditions: 31.9 - 65.5
  // So the texture veto now requires `skin_score` to ALSO look REPLAY-like
  // (median in window >= 30) before firing. A twilight LIVE has low texture
  // AND low skin_score → veto stays silent. A genuine REPLAY has low texture
  // AND high skin_score → veto fires fast. Single-feature thresholds suffer
  // from low-light camera-noise overlap; the co-signal AND-gate is the
  // anti-false-positive primitive.
  static readonly TEXTURE_COSIGNAL_SKIN_MIN = 30;
  // PHOTO path (2026-06-01). A static image — paper photo or a still photo
  // shown on a screen — has flat texture like a video replay, but its
  // skin_score sits at ~0 (no through-a-screen live-skin reflectance AND no
  // real 3-D skin), BELOW even a real face's range. Flat texture + windowed
  // skin_score median < 5 is an unambiguous STATIC_IMAGE: it can't be a real
  // face (skin too low) and isn't a video replay (skin too low for the >=30
  // video co-signal). This is what the >=30 video gate was BLIND to — photos
  // read skin~0 and slipped through as LIVE.
  // Threshold 5 (not 8) chosen by sweeping all 34 current-build sessions: at 8
  // a genuine face's transient windowed-skin dips fired (median-19.5 LIVE); 5
  // sits below any real face's SUSTAINED skin (next LIVE session up is 13.4),
  // so only true photos trip it. (The lone LIVE-labelled session that still
  // fires reads skin~0 AND texture 12 — a mislabel; real faces read skin>=13,
  // texture>=40.)
  static readonly TEXTURE_PHOTO_SKIN_MAX = 5;
  // ---- Motion-gated VIDEO-vs-STATIC typing (2026-06-01) --------------------
  // skin_score decides WHETHER a texture-collapse fires (it separates a screen
  // / photo from a real distant face), but it CANNOT type video-vs-static: a
  // glowing screen reads skin ~70 whether it plays a video or shows a photo.
  // The only true difference is FACIAL motion — but raw landmark-variance,
  // blink COUNT, and gaze.std_x are all corrupted by *rigid* motion (waving a
  // photo manufactures fake "movement"; a hard-waved photo logged 9 false
  // blinks + gaze.std_x 0.21, higher than a still video). The signal that
  // survives is NON-RIGID eyelid deformation (blink-symmetry blendshape
  // variance) measured ONLY on rigid-still frames. Empirical (clean captures,
  // 2026-06-01, SESSION doc):
  //   still photo  eyeMotion@still 0.026 | moved photo 0.058 | hard-waved ~0
  //   still video  eyeMotion@still 0.405   ← ~7x separation
  // A waved photo also self-defeats: violent motion leaves ~1% rigid-still
  // frames, so it can't even produce a measurable signal → defaults STATIC.
  // PROVISIONAL thresholds (n=1/class, validate as multi-subject data grows).
  // A frame is "rigid-still" when landmark_variance.overall_var < this.
  static readonly TYPING_RIGID_STILL_MAX = 30;
  // Median eyelid-blendshape variance on rigid-still frames >= this => the face
  // content is genuinely moving (video replay); below => frozen (static image).
  static readonly TYPING_VIDEO_EYE_MOTION_MIN = 0.2;
  // Need at least this many rigid-still frames to *confirm* a video; fewer
  // (a violently-waved photo) cannot confirm motion and defaults to STATIC.
  static readonly TYPING_MIN_STILL_FRAMES = 5;
  // Rolling window (frames) over which eyelid + rigid motion are tracked for
  // typing. Larger than the 30-frame texture window so sparse blinks are seen.
  static readonly TYPING_MOTION_WINDOW = 150;
  // Capture-quality floor (2026-05-24). A would-be-LIVE session whose capture
  // quality is too poor (dark / occluded / no-face frames) is downgraded to
  // UNCERTAIN (prompt a re-capture) rather than confidently classified LIVE.
  // A genuine spoof (real spoof evidence) is NEVER downgraded — it stays SPOOF.
  // illuminationScore is 0-1 from the FaceUsabilityGate (normal lit face ~0.8).
  static readonly QUALITY_MIN_SAMPLES = 15;
  // 2026-05-31 — relaxed from 0.5 → 0.3. The capture-quality floor is the
  // gate that pins displayed confidence to 30 % when triggered. At 0.5 the
  // floor was firing on bright-room LIVE captures whose FaceUsabilityGate
  // boolean was incorrectly flagging mouth as occluded (state breakdown
  // observed: CLEAR 47 %, OCCLUDED_* 47 %, RECOVERING 6 % — the gate's
  // mouth-region pixel threshold is calibrated for the Python pipeline and
  // misfires on browser-port landmarks at moderate distance, same root
  // cause as the readiness gate's false occlusion fix).
  //
  // At 0.3 the floor still catches genuinely poor captures (a dark or
  // unfaced session sees usable_ratio < 0.1) while letting a clearly-lit
  // face with intermittent landmark noise reach a confident LIVE. The
  // illumination floor below remains the hard gate against dim captures.
  static readonly QUALITY_USABLE_RATIO = 0.3;
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
  // Recent texture_score samples for the windowed-ratio veto (V2). Frame-count
  // based, not seconds, so a slow camera can't produce a one-frame "window".
  private recentTextureScores = new RingBuffer<number>(
    SessionEngine.TEXTURE_WINDOW_FRAMES,
  );
  // V3 co-signal: parallel ring of screen_replay.skin_score samples; the
  // texture veto checks both before firing.
  private recentSkinScores = new RingBuffer<number>(
    SessionEngine.TEXTURE_WINDOW_FRAMES,
  );
  // Motion-gated typing (2026-06-01): per-frame eyelid-blendshape variance
  // (blink_symmetry std_left+std_right) and a rigid-motion proxy
  // (landmark_variance.overall_var). Used to type a confirmed texture-collapse
  // spoof as VIDEO_REPLAY vs STATIC_IMAGE by NON-rigid eyelid motion measured
  // on rigid-still frames only. Window larger than the texture window so the
  // sparse blink signal is captured.
  private recentEyeMotion = new RingBuffer<number>(
    SessionEngine.TYPING_MOTION_WINDOW,
  );
  private recentRigidMotion = new RingBuffer<number>(
    SessionEngine.TYPING_MOTION_WINDOW,
  );
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
    this.recentTextureScores.clear();
    this.recentSkinScores.clear();
    this.recentEyeMotion.clear();
    this.recentRigidMotion.clear();
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
   * variance sub-feature (`texture_score`) collapses ACROSS A SUSTAINED
   * WINDOW — the signature of a face being shown on a phone or laptop
   * screen. Catches the attack the `checkPlanarPrint` veto misses: a held-
   * close phone playing a video (the user's face never rotates, so
   * planarity stays in `measured:false`), and a live video-call on a
   * phone (no playback artifacts at all).
   *
   * V2 windowed-ratio implementation: an incident fires only when ≥30% of
   * the last 30 frames had texture_score < 25. V1's per-frame throttle
   * mis-classified a 344 s lower-light LIVE session as SPOOF (1% isolated
   * dips over a long session still accrued ≥3 incidents).
   *
   * Sustained-ratio discrimination is clean: in-house data shows LIVE
   * sessions cap at 18% frames-below-threshold (a 13-min daylight session
   * stressed the boundary), SPOOF sessions sit at 50-100% (worst SPOOF
   * outlier 8% still reaches the 3-incident bar via other vetoes).
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
    this.recentTextureScores.append(textureScore);

    // V3: also sample the co-signal (screen_replay.skin_score). Pushed on
    // every frame the texture sample comes in so the two windows stay
    // aligned. When skin_score isn't available we push NaN so the median
    // computation is still defined (Number.isNaN filter below).
    const screenReplay = cls.analyzer_results["screen_replay"];
    const skinScore = screenReplay?.details["skin_score"];
    this.recentSkinScores.append(
      typeof skinScore === "number" ? skinScore : Number.NaN,
    );

    // Motion-gated typing samples (2026-06-01). Eyelid-blendshape variance is
    // the sum of the left/right blink-blendshape std-devs (pose-normalised by
    // MediaPipe, so rigid motion can't fake it). The rigid-motion proxy is the
    // raw landmark-variance (which DOES rise under rigid waving) — used only to
    // GATE which frames the eyelid signal is read on, never to type directly.
    const blinkSym = cls.analyzer_results["blink_symmetry"];
    const stdL = blinkSym?.details["std_left"];
    const stdR = blinkSym?.details["std_right"];
    this.recentEyeMotion.append(
      (typeof stdL === "number" ? stdL : 0) +
        (typeof stdR === "number" ? stdR : 0),
    );
    const landmarkVar = cls.analyzer_results["landmark_variance"];
    const overallVar = landmarkVar?.details["overall_var"];
    this.recentRigidMotion.append(
      typeof overallVar === "number" ? overallVar : Number.NaN,
    );

    if (elapsed < SessionEngine.TEXTURE_MIN_ELAPSED_SEC) return;
    if (this.recentTextureScores.length < SessionEngine.TEXTURE_WINDOW_MIN_FRAMES) {
      return;
    }
    if (
      elapsed - this.lastTextureIncidentAt <
      SessionEngine.TEXTURE_INCIDENT_THROTTLE_SEC
    ) {
      return;
    }

    const samples = this.recentTextureScores.toArray();
    let lowCount = 0;
    for (const s of samples) {
      if (s < SessionEngine.TEXTURE_SCORE_SPOOF_THRESHOLD) lowCount += 1;
    }
    const lowFraction = lowCount / samples.length;
    if (lowFraction < SessionEngine.TEXTURE_LOW_FRACTION_SPOOF) return;

    // 2026-06-02 motion-blur SUPPRESSOR (fixes a live false reject: a real face
    // at 50 cm MOVING was verdicted SPOOF static_image). A real moving face's
    // texture collapses from MOTION BLUR — validated: in collapse frames median
    // landmark_var was 102 vs 5 when texture was OK, and 85% of collapse frames
    // were moving — but its texture is FINE when still; a photo/screen is
    // low-texture even when still. So if we have ENOUGH rigid-still frames AND
    // their texture is mostly fine, the collapse is motion-only → suppress.
    // We only suppress on POSITIVE evidence: a violently-waved photo leaves too
    // few still frames to clear this gate, so it falls through and still FIRES
    // (no false-accept regression — the hard-waved test rigidVar=120 → 0 still
    // frames → not suppressed). Measured: SPOOF still frac<25 ~40% (won't clear,
    // fires); LIVE ~17% / this session ~1-2% (clears → suppressed).
    // Align from the most-recent end: recentRigidMotion (window 150) is larger
    // than recentTextureScores (texture window), so a start-indexed pair would
    // mismatch frames. Both buffers are appended once per frame, so the k-th
    // from the end of each is the same frame.
    const rigidSamples = this.recentRigidMotion.toArray();
    const n = Math.min(samples.length, rigidSamples.length);
    let stillLow = 0;
    let stillTotal = 0;
    for (let k = 1; k <= n; k++) {
      const rigid = rigidSamples[rigidSamples.length - k];
      if (Number.isNaN(rigid) || rigid >= SessionEngine.TYPING_RIGID_STILL_MAX) {
        continue;
      }
      stillTotal += 1;
      if (samples[samples.length - k] < SessionEngine.TEXTURE_SCORE_SPOOF_THRESHOLD) {
        stillLow += 1;
      }
    }
    if (
      stillTotal >= SessionEngine.TYPING_MIN_STILL_FRAMES &&
      stillLow / stillTotal < SessionEngine.TEXTURE_LOW_FRACTION_SPOOF
    ) {
      return; // collapse is motion-coincident on a real moving face → suppress
    }

    // V3 co-signal gate. Compute the median skin_score over the same
    // window. A genuine REPLAY has both signals collaborating (low texture
    // + high skin_score); a twilight LIVE has only the texture symptom
    // (camera noise reduction → smooth pixels → low Laplacian variance)
    // but skin_score remains low because the face is real skin not
    // through-a-screen rendering.
    const skinSamples = this.recentSkinScores
      .toArray()
      .filter((x) => !Number.isNaN(x));
    let skinMedian = Number.NaN;
    if (skinSamples.length >= SessionEngine.TEXTURE_WINDOW_MIN_FRAMES / 2) {
      const sorted = skinSamples.slice().sort((a, b) => a - b);
      skinMedian = sorted[Math.floor(sorted.length / 2)];
    }
    if (Number.isNaN(skinMedian)) {
      // No skin co-signal in the window — can't safely type the collapse.
      return;
    }

    // FIRE-OR-SUPPRESS by skin mode (unchanged anti-false-positive primitive).
    // skin_score is tri-modal: photo ~0 < real-face 5-30 < screen 30-68.
    //   skin >= 30 (screen)  OR  skin < 5 (photo/print)  → it's a spoof, FIRE.
    //   skin in [5, 30)       → real-face-at-distance / low-light band where a
    //                           genuine face's texture collapses from camera
    //                           noise. Ambiguous → SUPPRESS (no false-reject).
    const isScreenLike = skinMedian >= SessionEngine.TEXTURE_COSIGNAL_SKIN_MIN;
    let isPhotoLike = skinMedian < SessionEngine.TEXTURE_PHOTO_SKIN_MAX;
    // 2026-06-02 — a printed photo (skin<5, low texture) is held STILL. A real
    // face moving at distance also momentarily reads skin<5 + low texture from
    // MOTION BLUR: the live false reject fired this photo path at 6s/11s with
    // recent-motion median ~65. So only accept the photo path under LOW recent
    // motion; under high motion skin<5 is a blurred real face, not a still photo.
    // A genuinely waved photo is rigid+planar → still caught by the planarity
    // veto. Validated by offline replay on the labelled set: clears the false
    // reject (2 tex incidents → 0 → LIVE) with ZERO loss of spoof catches
    // (SPOOF sessions firing ≥3 incidents unchanged at 14/20).
    if (isPhotoLike) {
      const rig = this.recentRigidMotion
        .toArray()
        .filter((x) => !Number.isNaN(x));
      if (rig.length > 0) {
        const sortedRig = rig.slice().sort((a, b) => a - b);
        const motionMedian = sortedRig[Math.floor(sortedRig.length / 2)];
        if (motionMedian >= SessionEngine.TYPING_RIGID_STILL_MAX) {
          isPhotoLike = false; // high motion → blurred real face, not a still photo
        }
      }
    }
    if (!isScreenLike && !isPhotoLike) return;

    // TYPE the confirmed spoof as VIDEO_REPLAY vs STATIC_IMAGE by NON-rigid
    // eyelid motion measured ONLY on rigid-still frames (2026-06-01). This is
    // the one signal a waved photo can't fake: its eyes stay frozen open, and
    // violent waving leaves too few still frames to even measure (→ STATIC).
    const eyeArr = this.recentEyeMotion.toArray();
    const rigidArr = this.recentRigidMotion.toArray();
    const stillEye: number[] = [];
    const pairCount = Math.min(eyeArr.length, rigidArr.length);
    for (let i = 0; i < pairCount; i++) {
      const rigid = rigidArr[i];
      if (
        !Number.isNaN(rigid) &&
        rigid < SessionEngine.TYPING_RIGID_STILL_MAX &&
        eyeArr[i] > 0
      ) {
        stillEye.push(eyeArr[i]);
      }
    }
    let category: SpoofCategory;
    let detail: string;
    let stillEyeMedian: number | null = null;
    if (stillEye.length >= SessionEngine.TYPING_MIN_STILL_FRAMES) {
      const sortedEye = stillEye.slice().sort((a, b) => a - b);
      stillEyeMedian = sortedEye[Math.floor(sortedEye.length / 2)];
      if (stillEyeMedian >= SessionEngine.TYPING_VIDEO_EYE_MOTION_MIN) {
        category = SpoofCategory.VIDEO_REPLAY;
        detail =
          `non-rigid eyelid motion ${stillEyeMedian.toFixed(2)} over ${stillEye.length} ` +
          `rigid-still frames (>= ${SessionEngine.TYPING_VIDEO_EYE_MOTION_MIN}) — ` +
          `moving face content (video replay / video-call) suspected. ` +
          `skin median ${Math.round(skinMedian)}.`;
      } else {
        category = SpoofCategory.STATIC_IMAGE;
        detail =
          `eyelid motion only ${stillEyeMedian.toFixed(2)} over ${stillEye.length} ` +
          `rigid-still frames (< ${SessionEngine.TYPING_VIDEO_EYE_MOTION_MIN}) — ` +
          `frozen face (static image: printed / on-screen photo) suspected. ` +
          `skin median ${Math.round(skinMedian)}.`;
      }
    } else {
      // Too few rigid-still frames — e.g. a photo being waved to fake motion.
      // Can't confirm genuine facial motion → default to STATIC_IMAGE.
      category = SpoofCategory.STATIC_IMAGE;
      detail =
        `only ${stillEye.length} rigid-still frames (< ${SessionEngine.TYPING_MIN_STILL_FRAMES}) — ` +
        `motion too unstable to confirm a video (waved photo?); static image suspected. ` +
        `skin median ${Math.round(skinMedian)}.`;
    }

    this.lastTextureIncidentAt = elapsed;
    this.addIncident(
      frame_id,
      Severity.HIGH,
      category,
      `Texture collapse — ${Math.round(lowFraction * 100)}% of last ${samples.length} ` +
        `frames below texture threshold (${SessionEngine.TEXTURE_SCORE_SPOOF_THRESHOLD}); ${detail}`,
      {
        low_fraction: round(lowFraction, 3),
        low_count: lowCount,
        still_frames: stillTotal,
        window_frames: samples.length,
        threshold: SessionEngine.TEXTURE_SCORE_SPOOF_THRESHOLD,
        last_texture_score: round(textureScore, 1),
        skin_score_median: round(skinMedian, 1),
        still_eye_motion_median:
          stillEyeMedian === null ? null : round(stillEyeMedian, 3),
        still_frame_count: stillEye.length,
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

    // --- Dominant threat TYPE (not the live/spoof verdict — that's below) ---
    // The per-frame fusion is easily fooled on screen replays: a phone replay
    // can read real=0.82 in the fuser, with the residual split so that
    // static_image (0.08) edges out video_replay (0.06) — yet that same
    // session trips 30 texture-collapse VIDEO_REPLAY incidents. Reporting
    // "static_image" there is wrong and was user-flagged (2026-06-01): the
    // attack visibly MOVES (12 blinks, landmark motion), so it cannot be a
    // still photo. Two corrections:
    //
    //   1. Incident evidence names the threat. The analyzers that actually
    //      flip the verdict (incident_override) are a stronger threat-type
    //      signal than the fusion's fooled residual. When incidents exist,
    //      their most-frequent non-real category wins.
    //   2. Typing comes from the texture-veto incident's SKIN mode, not from
    //      motion. 2026-06-01 ultrathink: blink/landmark signals are too noisy
    //      to type with — a hand-held photo logged 5 false blinks + the highest
    //      landmark-variance in the set (moving the photo = rigid motion that
    //      MediaPipe can't tell from facial motion). The old "blinks>=1 => not
    //      static" re-label therefore MISLABELED moved photos as video_replay.
    //      checkTextureCollapseReplay now tags STATIC_IMAGE when skin<8 (photo,
    //      skin~0) and VIDEO_REPLAY when skin>=30 (through-a-screen video).
    let threatFromFusion: SpoofCategory | null = null;
    let dominantP = -Infinity;
    for (const cat of ALL_SPOOF_CATEGORIES) {
      if (cat === SpoofCategory.REAL) continue;
      const p = categoryScores[cat];
      if (p > dominantP) {
        dominantP = p;
        threatFromFusion = cat;
      }
    }

    let threatFromIncidents: SpoofCategory | null = null;
    if (this.incidents.length > 0) {
      const counts = new Map<SpoofCategory, number>();
      for (const inc of this.incidents) {
        if (inc.category === SpoofCategory.REAL) continue;
        counts.set(inc.category, (counts.get(inc.category) ?? 0) + 1);
      }
      let best = 0;
      for (const [cat, n] of counts) {
        if (n > best) {
          best = n;
          threatFromIncidents = cat;
        }
      }
    }

    const dominantThreat: SpoofCategory | null =
      threatFromIncidents ?? threatFromFusion;

    // dataConfidence ramps from 0 → 1 as we accumulate evidence. Was
    // frameCount / 150 (5 s at 30 fps). On a 5-7 fps capture that worked
    // out to ~25 s before the engine would issue a confident verdict,
    // which combined with the quality_uncertain 0.3-confidence pin made
    // the displayed confidence appear stuck at 30% for ~50 s, then jump
    // to 96 % once both gates cleared — a jarring UX.
    // Switching to elapsed-seconds means the ramp matches wall-clock time
    // regardless of fps. 5 s full ramp keeps the original "1 s warmup +
    // 4 s evidence-build" intent of the original constant.
    const dataConfidence = Math.min(1.0, this.elapsedSec / 5.0);
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
    //
    // EXCEPT when the LivenessProver has accumulated >= 60 / 100 of passive
    // movement evidence over 5+ s (its `is_proven_live` threshold). That's
    // a strong, observation-grounded LIVE signal — blink + landmark variance
    // + head rotation + eye/mouth motion all checked passively — and it's
    // perverse to display "UNCERTAIN, 34 %" on screen while the proof panel
    // simultaneously displays "PROVEN LIVE 100/100" (a real user-reported
    // inconsistency 2026-05-31). When proof has crossed the threshold AND
    // no spoof incidents are firing, we trust it over a quality-floor flag
    // that's almost always the FaceUsabilityGate's mouth-region false
    // alarm at the browser's MediaPipe landmark confidence.
    const qualityOk = this.computeQualityOk();
    // 2026-05-31 V2 — tightened. User correctly observed that
    // LivenessProver measures ACTIVITY (blinks, head movement, expressions),
    // not strict liveness — a passive real user gets a low score, while an
    // animated video replay gets a high one. So "proof_proven_live" can only
    // override quality_uncertain when there's ALSO no countervailing spoof
    // evidence: zero incidents AND at least one blink observed. That makes
    // proof-trumps a tie-breaker, not a magic override.
    //
    // (The right fix is making qualityOk accurate — the CriticalRegionVisibilityGate
    // thresholds were lowered in the same commit so quality_uncertain stops
    // firing on visible faces. This is the belt-and-braces safety net.)
    const proofProvenLive = (proverScore?.total ?? 0) >= 60;
    const noSpoofIncidents = this.incidents.length === 0;
    const observedBlinks = this.lastBlinkCount >= 1;
    const proofTrumpsQuality = proofProvenLive && noSpoofIncidents && observedBlinks;
    const qualityUncertain = baseLive && !qualityOk && !proofTrumpsQuality;
    const isLive = baseLive && (qualityOk || proofTrumpsQuality);

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
