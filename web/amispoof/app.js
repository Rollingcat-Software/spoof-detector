// amispoof — browser anti-spoof tester driver.
//
// Loads @rollingcat/spoof-detector from ./lib/, the ONNX + FaceLandmarker
// models from ./models/, and the onnxruntime-web / mediapipe-tasks-vision
// peer deps from jsdelivr via the importmap declared in index.html.

import * as ort from "onnxruntime-web";
// Cache-bust the lib bundle too — the bare ./lib/spoof-detector.js URL
// was a 7-day cached resource on mobile browsers when the .htaccess used
// max-age=604800, so users with that old cache header would load a fresh
// app.js but a stale Phase-2 lib bundle (Texture/Moire/Screen-replay/rPPG
// rows show "—", gate panel stuck on "waiting…"). The query string
// gives the import a distinct URL the browser must re-fetch.
import {
  createSpoofDetector,
  runCasiaFasdMicroBench,
  FlashReflectionAnalyzer,
  FlashTemporalAnalyzer,
  ReadinessGate,
  DEFAULT_ANALYZER_WEIGHTS,
} from "./lib/spoof-detector.js?v=2026-05-31-texture-veto-v2";

// Version handshake — checked by the inline script in index.html.
// If the user is running a stale cached app.js (no AMISPOOF_VERSION),
// the HTML triggers a one-shot reload after 4 s.
window.AMISPOOF_VERSION = "2026-05-31-texture-veto-v2";

// SessionEngine.getVerdict() returns a confidence in [0, 0.88] when the
// LivenessProver is wired (structural ceiling — see SessionEngine.ts
// confidence formula: 0.3 floor + 0.3 prover-max + 0.28 fusion-max).
// Human-facing surfaces (the verdict badge and copy-to-clipboard text)
// normalize to [0, 100] so users don't read 81% as "still uncertain".
// Machine surfaces (downloaded JSON, bench rows) keep the raw value.
const RAW_CONFIDENCE_CEILING = 0.88;
function displayConfPct(rawConfidence) {
  const normalized = (rawConfidence ?? 0) / RAW_CONFIDENCE_CEILING;
  const clamped = Math.max(0, Math.min(1, normalized));
  return Math.round(clamped * 100);
}

// Build the human-facing summary line. Used by the on-screen verdict text,
// the badge, and the Copy-to-clipboard surface — all read the SAME string
// so the page can't show a normalized 91% in one place and the engine's
// raw 80% in another. v.summary from the engine is left untouched so
// SDK consumers keep the raw scale.
function displaySummary(v) {
  // Tri-state: the engine emits quality_uncertain when the capture is too poor to
  // judge (low light / occlusion / look-away). It is NOT a spoof — painting it
  // "SPOOF" made a real person flicker red on a momentary quality dip.
  const verdictWord = v.is_live ? "LIVE" : v.quality_uncertain ? "UNCERTAIN" : "SPOOF";
  const threat = v.dominant_threat ? ` (${v.dominant_threat})` : "";
  return (
    `${verdictWord}${threat} | conf=${displayConfPct(v.confidence)}% | ` +
    `${v.session_duration_sec.toFixed(1)}s | ${v.frames_analyzed} frames | ` +
    `blinks=${v.blink_count ?? 0} | incidents=${v.incidents.length}`
  );
}

// Per-analyzer hover tooltip: static description + the most-informative
// detail fields rendered as `key=value` pairs. The full per-frame detail
// blob still ships in the downloaded JSON; this surfaces the headline
// numbers (eye/mouth/forehead variance, tremor_x/y, EAR, …) inline.
const ANALYZER_DETAIL_KEYS = {
  landmark_variance: [
    "eye_var",
    "mouth_var",
    "forehead_var",
    "expression_ratio",
  ],
  blink: ["ear", "blinks", "blink_rate_per_min"],
  micro_tremor: ["tremor_x", "tremor_y", "tremor_ratio", "data_quality"],
  rppg: ["bpm", "snr", "data_quality"],
  screen_flicker: ["dominant_freq_hz", "max_flicker_power", "measured_fps"],
  temporal: ["pos_std", "area_std", "motion"],
  background_grid: [
    "stability_ratio",
    "specular_ratio",
    "cool_ratio",
    "total_bg_cells",
  ],
  device_boundary: ["boundary_score", "line_score", "n_lines"],
  screen_replay: ["fft_score", "laplacian_score", "skin_score"],
  texture: [
    "texture_score",
    "color_score",
    "frequency_score",
    "color_drift_score",
    "color_drift_samples",
  ],
  moire: ["moire_risk", "gabor_risk", "fft_risk"],
  minifasnet: ["p_real", "p_spoof"],
  // Phase A analyzers — surface the headline details inline on hover.
  eyebrow_motion: ["activation", "std", "frames"],
  blink_symmetry: ["corr", "std_left", "std_right", "frames"],
  gaze: ["gaze_x", "gaze_y", "saccade_count", "saccade_rate_per_sec"],
  expression_dynamics: ["total", "std", "frames"],
  pose_3d_consistency: ["ortho_residual", "ortho_score", "tz", "tz_std"],
  behavioral_pattern: [
    "blink_cv",
    "saccade_rate_per_sec",
    "entropy_score",
    "fps",
  ],
  background_motion: [
    "samples",
    "bg_pixel_ratio",
    "std_r",
    "std_g",
    "std_b",
    "drift",
  ],
  hand_tracking: [
    "hand_count",
    "handedness",
    "wrist_std",
    "samples",
    "anomaly_third_hand",
  ],
  voice_activity: [
    "voice_fraction",
    "rms_mean",
    "rms_peak",
    "samples",
    "rms_hz",
  ],
  audio_mouth_sync: [
    "corr",
    "audio_std",
    "mouth_std",
    "jaw_open",
    "frames",
    "silence",
  ],
};

function formatDetailValue(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3);
  }
  return String(v);
}

function analyzerTooltip(cfg, result) {
  const keys = ANALYZER_DETAIL_KEYS[cfg.name] ?? [];
  const details = result.details ?? {};
  const pairs = keys
    .filter((k) => details[k] !== undefined)
    .map((k) => `${k}=${formatDetailValue(details[k])}`);
  if (pairs.length === 0) return cfg.desc;
  return `${cfg.desc}\n\nLive: ${pairs.join(" · ")}`;
}

// Render the LivenessProver proof panel from a getProof() snapshot.
// Updates the headline (total + proven badge), the active-challenge prompt
// (visible only while one is pending), the per-axis bars, and the trailing
// motion/challenge summary. Safe to call with `null` (placeholder state).
function renderProofPanel(proof) {
  if (!proof) {
    els.proofHeadline.className = "proof-headline pending";
    els.proofHeadline.innerHTML = "waiting for evidence…";
    els.proofChallenge.style.display = "none";
    for (const axis of PROOF_AXES) {
      const ref = proofRefs[axis.name];
      if (!ref) continue;
      ref.fill.style.width = "0%";
      ref.val.innerHTML = `0 <span class="max">/ ${axis.max}</span>`;
    }
    if (els.proofMotion) els.proofMotion.textContent = "head motion: —";
    if (els.proofChallengeStats)
      els.proofChallengeStats.textContent = "challenges: 0 passed · 0 failed";
    return;
  }

  const total = Math.round(proof.score);
  const proven = proof.is_proven_live;
  els.proofHeadline.className = `proof-headline ${proven ? "proven" : "pending"}`;
  const status = proven ? "✓ proven live" : `${60 - total} pts to proof`;
  els.proofHeadline.innerHTML = `
    <span>${proven ? "Proven live" : "Building proof"}</span>
    <span class="total">${total}<span class="max"> / 100</span> · <span style="font-size:11px">${status}</span></span>
  `;

  // Active challenge banner — only visible while one is pending AND active
  // challenges are part of this consumer's proof flow.
  const ac = proof.active_challenge;
  if (SHOW_ACTIVE_SECTION && ac && ac.state === "prompted") {
    const remain = Math.max(
      0,
      (ac.timeout_sec ?? 0) - (proof.elapsed_sec - (ac.prompted_at ?? 0)),
    );
    const label = (ac.challenge_type ?? "challenge")
      .replace(/_/g, " ")
      .toUpperCase();
    els.proofChallenge.style.display = "";
    els.proofChallenge.textContent = `${label} — ${remain.toFixed(1)}s remaining`;
  } else {
    els.proofChallenge.style.display = "none";
  }

  for (const axis of PROOF_AXES) {
    const ref = proofRefs[axis.name];
    if (!ref) continue; // axis hidden (e.g. challenges in proctoring mode)
    const v = proof.details?.[axis.name] ?? 0;
    const pct = Math.max(0, Math.min(100, (v / axis.max) * 100));
    ref.fill.style.width = `${pct.toFixed(0)}%`;
    ref.val.innerHTML = `${v.toFixed(0)} <span class="max">/ ${axis.max}</span>`;
  }

  if (els.proofMotion) {
    const yaw = proof.yaw_range_seen_deg ?? 0;
    const pitch = proof.pitch_range_seen_deg ?? 0;
    els.proofMotion.textContent = `head motion: yaw ${yaw.toFixed(1)}° · pitch ${pitch.toFixed(1)}°`;
  }
  if (els.proofChallengeStats) {
    const passed = proof.details?.challenges_passed ?? 0;
    const failed = proof.details?.challenges_failed ?? 0;
    els.proofChallengeStats.textContent = `challenges: ${passed} passed · ${failed} failed`;
  }
}

const ORT_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
const MEDIAPIPE_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

// Point ORT WASM resolver at jsdelivr so it can find the .wasm sidecar files.
// (onnxruntime-web defaults to "relative to the JS bundle", which would 404
// when our bundle is served from /amispoof/lib/.)
ort.env.wasm.wasmPaths = ORT_WASM_BASE;
ort.env.wasm.numThreads = 2;

// Grouped by what the analyzer measures over time.
// "image" = single-frame signal (works on any static input).
// "video" = needs multiple frames to produce a useful signal.
const ANALYZER_ORDER = [
  // ----- Image track -----
  {
    name: "minifasnet",
    weight: 5.0,
    label: "MiniFASNet",
    group: "image",
    desc: "Pretrained ONNX binary classifier. Catches printed photos + screen replays in a single frame.",
  },
  {
    name: "device_boundary",
    weight: 2.5,
    label: "Device boundary",
    group: "image",
    desc: "Sobel + Hough — looks for phone/tablet bezel edges around the face.",
  },
  {
    name: "texture",
    weight: 0.1,
    label: "Texture",
    group: "image",
    desc: "Laplacian variance + HSV stats + small FFT. Real skin has high-frequency detail printed paper lacks.",
  },
  {
    name: "moire",
    weight: 0.1,
    label: "Moire",
    group: "image",
    desc: "Gabor + 2D-FFT. Detects the camera × LCD pixel-grid beat of a replay attack.",
  },
  {
    name: "screen_replay",
    weight: 0.5,
    label: "Screen replay",
    group: "image",
    desc: "CLAHE + YCrCb/HSV skin-mask + specular-reflection ratio.",
  },
  // ----- Video track -----
  {
    name: "screen_flicker",
    weight: 3.0,
    label: "Screen flicker",
    group: "video",
    desc: "Per-frame mean intensity → DFT. 50/60 Hz beat from any LCD/OLED.",
  },
  {
    name: "micro_tremor",
    weight: 2.5,
    label: "Micro-tremor",
    group: "video",
    desc: "Centroid jitter FFT. Live humans involuntarily oscillate at 8–12 Hz.",
  },
  {
    name: "landmark_variance",
    weight: 2.0,
    label: "Landmark variance",
    group: "video",
    desc: "478-point landmark std over time. Zero variance = static photo.",
  },
  {
    name: "rppg",
    weight: 0.5,
    label: "rPPG (pulse)",
    group: "video",
    desc: "Green-channel forehead FFT. Detects ~1 Hz cardiac signal — absent in print/replay.",
  },
  {
    name: "blink",
    weight: 0.5,
    label: "Blink (EAR)",
    group: "video",
    desc: "Eye Aspect Ratio over time. 0 blinks for 15s ⇒ static-image incident.",
  },
  {
    name: "temporal",
    weight: 0,
    label: "Face motion",
    group: "video",
    desc: "Face position and bounding-box area variance across the rolling window. Photos and locked-frame replays score 0; live faces drift naturally.",
  },
  {
    name: "background_grid",
    weight: 1.5,
    label: "Background grid",
    group: "video",
    desc: "Per-cell stability of the scene behind the face. A replay attack on a phone screen produces specular highlights and a too-stable backdrop.",
  },
  // Phase A — blendshape / 3D matrix derived (MediaPipe free unlock).
  {
    name: "eyebrow_motion",
    weight: 0,
    label: "Eyebrow motion",
    group: "video",
    desc: "Rolling variance of 5 ARKit brow blendshapes (browInnerUp + 4 directional). 0 on a rigid-brow photo; climbs naturally during talking or thinking.",
  },
  {
    name: "blink_symmetry",
    weight: 0,
    label: "Blink symmetry",
    group: "video",
    desc: "Pearson correlation of eyeBlinkLeft vs eyeBlinkRight across a 90-frame window. Real humans blink synchronously (≥ 0.7); deepfake/AR-filter avatars often desync per-eye.",
  },
  {
    name: "gaze",
    weight: 0,
    label: "Gaze",
    group: "video",
    desc: "2D gaze vector derived from 8 eyeLook* blendshapes — rolling variance + saccade count. Fixed-gaze photos score 0; humans saccade ~3/s naturally.",
  },
  {
    name: "expression_dynamics",
    weight: 0,
    label: "Expression dynamics",
    group: "video",
    desc: "Rolling variance of 15 mouth/cheek/nose blendshapes (smile/frown/dimple/squint/sneer). Passive emotion-change proxy — no dedicated classifier needed.",
  },
  {
    name: "pose_3d_consistency",
    weight: 0,
    label: "3D pose consistency",
    group: "video",
    desc: "MediaPipe 4×4 facial transformation matrix — checks orthonormality of the rotation block + Z-translation motion. Catches tilted photos and flat-screen replays whose pose fit is degenerate.",
  },
  {
    name: "planarity",
    weight: 2.0,
    label: "Planarity (flat-surface)",
    group: "video",
    desc: "Affine landmark-reprojection residual under head rotation. A flat printed photo or screen moves as one plane (low residual → SPOOF); a real 3D face has depth parallax that breaks the affine fit (high residual → live). Camera-focus-independent, so it catches the sharp PC-focused print MiniFASNet misses. Backed by a session-level planar-print veto.",
  },
  {
    name: "behavioral_pattern",
    weight: 0,
    label: "Behavioral pattern",
    group: "video",
    desc: "Temporal-distribution check: blink-interval coefficient of variation + saccade rate + Shannon entropy of the composite jaw/brow/blink signal. Catches looped videos / animated avatars whose individual frames pass but whose distributions are too regular.",
  },
  {
    name: "background_motion",
    weight: 0,
    label: "Background motion",
    group: "video",
    desc: "MediaPipe SelfieSegmenter masks the user out; the analyzer tracks mean RGB drift of the remaining background pixels over a ~10 s window. Real environments shift subtly; a phone-screen replay holds the background constant.",
  },
  {
    name: "hand_tracking",
    weight: 0,
    label: "Hand tracking",
    group: "video",
    desc: "MediaPipe HandLandmarker tracks per-hand wrist position; rolling stddev → natural gesture credit. Flags >2 hands per frame as a deepfake artefact. Opt-in: append ?hand=1 to the URL (loads ~6 MB).",
  },
  {
    name: "voice_activity",
    weight: 0,
    label: "Voice activity",
    group: "video",
    desc: "Fraction of recent audio above the voice RMS threshold. Defensive against silent video-replay attacks. Opt-in: ?audio=1 or click 🎤 below (requires mic permission).",
  },
  {
    name: "audio_mouth_sync",
    weight: 0,
    label: "Audio-mouth sync",
    group: "video",
    desc: "Pearson correlation of audio RMS with the jawOpen blendshape over the last 2 s. Strongest single anti-replay signal: live speech correlates above 0.7; replay either has no audio or is desynced.",
  },
];

// SINGLE SOURCE OF TRUTH: the per-analyzer weight badges shown in the UI must
// reflect the weights the running fuser ACTUALLY uses. Previously each entry's
// `weight` was hardcoded here and had drifted from DEFAULT_ANALYZER_WEIGHTS in
// the lib (e.g. device_boundary/micro_tremor showed "w 2.5" while the fuser used
// 0.5), so the badges lied. Overwrite each from the lib's exported table by name.
for (const a of ANALYZER_ORDER) {
  if (Object.prototype.hasOwnProperty.call(DEFAULT_ANALYZER_WEIGHTS, a.name)) {
    a.weight = DEFAULT_ANALYZER_WEIGHTS[a.name];
  }
}

const $ = (id) => document.getElementById(id);

// Last detected face bbox — used by the opt-in flash challenge to crop the face.
let lastFaceBbox = null;

// Snapshot of the camera's settings taken when the stream starts, so the
// camera controls + on-exit can restore the webcam to how it began (rather
// than leaving it in a manual/locked state from experimentation).
let cameraDefaults = null;

// ===== Auto flash-response TEMPORAL probe (integrated video-replay detector) =====
// The validated, content-INDEPENDENT screen detector. We lock exposure, flash
// the screen white for ~1.5 s while sampling face-region brightness, then watch
// what happens after the flash. A real 3D face reflects instantly and drops
// straight back; a phone/replay's auto-brightness ramps up over ~1 s and stays
// elevated (its sensor saw our flash and re-arranged its own backlight). Runs
// automatically once past warmup, then on a cadence, and folds the result into
// the verdict. See FlashTemporalAnalyzer for the physics + discriminators.
const FLASH_PROBE_AT_FRAME = 90; // first probe ~a few seconds past warmup
const FLASH_PROBE_INTERVAL_MS = 45000; // re-probe cadence (intrusive → infrequent)
const FLASH_BASELINE_SAMPLES = 3; // pre-flash brightness samples
const FLASH_DURING_SAMPLES = 15; // ~1.5 s of flash at 100 ms spacing
const FLASH_AFTER_SAMPLES = 8; // post-flash drop-back samples
const FLASH_SAMPLE_MS = 100; // spacing between brightness samples
const flashTemporalAnalyzer = new FlashTemporalAnalyzer({ sampleIntervalMs: FLASH_SAMPLE_MS });
const flashProbe = {
  ran: false,
  running: false,
  lastAt: 0,
  result: null, // FlashTemporalResult
  reflection: null, // FlashReflectionResult (spatial, from the same flash)
  screenSuspected: false,
};

// ===== WB-cast probe (dormant diagnostic) =====
// Earlier attempt: lock white balance, measure the face colour cast. Found
// CONTENT-DEPENDENT (a screen's cast varies with the replayed scene), so it no
// longer drives the verdict — the flash-temporal probe above replaced it. Kept
// callable from the console (runWbProbe()) for diagnostics only.
const WB_PROBE_TEMP = 5000; // locked neutral white-balance temp (K)
window.WB_CAST_THRESHOLD = 0.5;
const wbProbe = {
  ran: false,
  running: false,
  lastAt: 0,
  faceRB: null,
  faceGB: null,
  cast: null,
  screenSuspected: false,
};

// ===== Pre-flight readiness gate (blocks session start) =====
// After the camera turns on we enter PREVIEW: the loop runs detection and
// evaluates capture readiness each frame, but the session verdict is NOT shown
// until the user clicks "Begin session", which is enabled only after the checks
// stay green for READINESS_STABLE_FRAMES consecutive frames. Abstain-first: a
// dark/over-lit/occluded/no-face capture yields "fix this", never a verdict.
const readinessGate = new ReadinessGate();
const READINESS_STABLE_FRAMES = 8;
let previewMode = false;
let readinessStreak = 0;

const els = {
  videoWrap: $("videoWrap"),
  video: $("video"),
  overlay: $("overlay"),
  start: $("start"),
  beginSession: $("beginSession"),
  readinessPanel: $("readinessPanel"),
  readinessHeadline: $("readinessHeadline"),
  readinessList: $("readinessList"),
  stop: $("stop"),
  reset: $("reset"),
  download: $("download"),
  bench: $("bench"),
  lightCheck: $("lightCheck"),
  flashOverlay: $("flashOverlay"),
  lightResult: $("lightResult"),
  cameraToggle: $("cameraToggle"),
  cameraPanel: $("cameraPanel"),
  benchPanel: $("benchPanel"),
  benchHeadline: $("benchHeadline"),
  benchRows: $("benchRows"),
  dot: $("dot"),
  status: $("status"),
  verdict: $("verdict"),
  verdictText: $("verdictText"),
  verdictConf: $("verdictConf"),
  state: $("state"),
  frames: $("frames"),
  fps: $("fps"),
  duration: $("duration"),
  faces: $("faces"),
  blinks: $("blinks"),
  incidents: $("incidents"),
  analyzers: $("analyzers"),
  cats: $("cats"),
  incidentList: $("incidentList"),
  details: $("details"),
  gateBanner: $("gateBanner"),
  gateBody: $("gateBody"),
  copyVerdict: $("copyVerdict"),
  proofHeadline: $("proofHeadline"),
  proofChallenge: $("proofChallenge"),
  proofRows: $("proofRows"),
  micToggle: $("micToggle"),
  handToggle: $("handToggle"),
  recordToggle: $("recordToggle"),
  replayBtn: $("replayBtn"),
  replayFile: $("replayFile"),
  replayPanel: $("replayPanel"),
  replayHeader: $("replayHeader"),
  replayChart: $("replayChart"),
  replayLegend: $("replayLegend"),
  replayClose: $("replayClose"),
  // Research/dataset capture fields (Phase F — frame_log instrumentation).
  // The dropdown bakes the ground-truth class into the JSON at save time so
  // the offline notebook can build a labelled LIVE-vs-REPLAY matrix without
  // manual relabeling. ambient_label + replay_device + notes are free-form
  // metadata the operator types before pressing ↓ Report.
  captureLabel: $("captureLabel"),
  ambientLabel: $("ambientLabel"),
  replayDevice: $("replayDevice"),
  captureNotes: $("captureNotes"),
};

const analyzerRefs = {};
function buildAnalyzerGroup(title, group) {
  const heading = document.createElement("div");
  heading.className = "analyzer-group-title";
  heading.textContent = title;
  els.analyzers.appendChild(heading);
  for (const cfg of ANALYZER_ORDER) {
    if (cfg.group !== group) continue;
    const row = document.createElement("div");
    row.className = "analyzer-row";
    row.title = cfg.desc;
    row.innerHTML = `
      <span class="name">${cfg.label} <span class="weight">w ${cfg.weight}</span></span>
      <span class="bar"><span class="fill" style="width: 50%"></span></span>
      <span class="val">—</span>
    `;
    els.analyzers.appendChild(row);
    analyzerRefs[cfg.name] = {
      row,
      fill: row.querySelector(".fill"),
      val: row.querySelector(".val"),
    };
  }
}
buildAnalyzerGroup("Image-track (single frame)", "image");
buildAnalyzerGroup("Video-track (over time)", "video");

// ---------- Liveness proof panel ----------
// LivenessProver scores accumulate every frame from passive evidence (blinks,
// landmark variance, head rotation, expression) and from active challenges
// (when surfaced). Each axis has a hard cap; the total of 60 crosses the
// proven-live threshold. The proof score feeds the engine's confidence
// calculation via the 0.3 × proverConfidence term.
// amispoof runs the prover in passive-only proctoring mode (no active
// challenges — they would disrupt an exam). The "active" section is kept
// in the registry for future non-proctoring consumers but is gated by
// SHOW_ACTIVE_SECTION; rendering and copy/download stay consistent.
const SHOW_ACTIVE_SECTION = false;
const PROOF_AXES = [
  {
    name: "blink_points",
    label: "Blink",
    max: 25,
    section: "passive",
    desc: "5 points per detected blink, capped at 25. A printed photo never blinks.",
  },
  {
    name: "landmark_points",
    label: "Landmark var",
    max: 20,
    section: "passive",
    desc: "Overall mesh drift. Capped at 20 once the 478-point face shows clear non-rigid motion.",
  },
  {
    name: "rotation_points",
    label: "Head rotation",
    max: 15,
    section: "passive",
    desc: "Combined yaw + pitch range seen across the 3-second window. Subtle natural head motion accumulates points; a fixed photo scores 0.",
  },
  {
    name: "expression_points",
    label: "Expression",
    max: 15,
    section: "passive",
    desc: "Eye/mouth/forehead variance ratio. Awarded when the face shows non-rigid expression change.",
  },
  {
    name: "eye_motion_points",
    label: "Eye motion",
    max: 12,
    section: "passive",
    desc: "Eye-region landmark drift independent of blink counting. Credits gaze tracking and eyelid micro-movement.",
  },
  {
    name: "mouth_motion_points",
    label: "Mouth motion",
    max: 10,
    section: "passive",
    desc: "Mouth-region landmark drift. Credits lip motion, talking, and sub-expression-ratio mouth movement.",
  },
  {
    name: "face_motion_points",
    label: "Face motion",
    max: 8,
    section: "passive",
    desc: "Bbox/centroid drift over time. Credits natural face/body sway; 0 on a perfectly locked photo or frozen-frame replay.",
  },
  // Phase A — blendshape / 3D matrix derived axes.
  {
    name: "eyebrow_motion_points",
    label: "Eyebrow motion",
    max: 8,
    section: "passive",
    desc: "Awarded from the EyebrowAnalyzer 0–100 score. Credits any natural brow motion (raise, furrow, asymmetric lift).",
  },
  {
    name: "blink_symmetry_points",
    label: "Blink symmetry",
    max: 6,
    section: "passive",
    desc: "Awarded when left/right blink correlation ≥ 0.7. Strong anti-deepfake / anti-AR-filter signal.",
  },
  {
    name: "gaze_variation_points",
    label: "Gaze variation",
    max: 8,
    section: "passive",
    desc: "Awarded from the GazeAnalyzer score. Credits eye-movement variability + saccade rate.",
  },
  {
    name: "expression_dynamics_points",
    label: "Expression dynamics",
    max: 8,
    section: "passive",
    desc: "Awarded from the ExpressionDynamicsAnalyzer score. Passive emotion-change proxy from 15 mouth/cheek/nose blendshapes.",
  },
  {
    name: "pose_3d_consistency_points",
    label: "3D pose",
    max: 6,
    section: "passive",
    desc: "Awarded from the Pose3DConsistencyAnalyzer score. Credits well-formed orthonormal 3D pose + natural Z-translation motion.",
  },
  {
    name: "behavioral_pattern_points",
    label: "Behavioral",
    max: 10,
    section: "passive",
    desc: "Awarded from the BehavioralPatternAnalyzer score. Credits human-like temporal distributions (irregular blink intervals, natural saccade rate, high signal entropy).",
  },
  {
    name: "background_motion_points",
    label: "Background motion",
    max: 8,
    section: "passive",
    desc: "Awarded from the BackgroundMotionAnalyzer score (Phase D1, opt-in). Credits drift in background RGB across the rolling window; 0 on a stationary replay.",
  },
  {
    name: "hand_naturalness_points",
    label: "Hand naturalness",
    max: 8,
    section: "passive",
    desc: "Awarded from the HandTrackingAnalyzer score (Phase D2, opt-in via ?hand=1). Credits natural hand gesture; caps low on the deepfake third-hand anomaly.",
  },
  {
    name: "voice_activity_points",
    label: "Voice activity",
    max: 6,
    section: "passive",
    desc: "Awarded from the VoiceActivityAnalyzer score (Phase D3, opt-in via 🎤 button or ?audio=1). Defensive against silent video replays.",
  },
  {
    name: "audio_mouth_sync_points",
    label: "Audio-mouth sync",
    max: 12,
    section: "passive",
    desc: "Awarded from the AudioMouthSyncAnalyzer score (Phase D3). Strongest single anti-video-replay signal — correlation of audio energy with jawOpen blendshape.",
  },
  {
    name: "challenge_points",
    label: "Challenges",
    max: 40,
    section: "active",
    desc: "10 points per completed active challenge (turn head, nod, blink-on-cue), capped at 40. Disabled in proctoring mode — passive axes above carry the score on their own.",
  },
];

const proofRefs = {};
function buildProofPanel() {
  const sections = SHOW_ACTIVE_SECTION ? ["passive", "active"] : ["passive"];
  const labels = {
    passive: "Passive evidence — every observed movement",
    active: "Active challenges (when surfaced)",
  };
  for (const section of sections) {
    const heading = document.createElement("div");
    heading.className = "proof-divider";
    heading.textContent = labels[section];
    els.proofRows.appendChild(heading);
    for (const axis of PROOF_AXES) {
      if (axis.section !== section) continue;
      const row = document.createElement("div");
      row.className = "proof-row";
      row.title = axis.desc;
      row.innerHTML = `
        <span class="name">${axis.label}</span>
        <span class="bar"><span class="fill" style="width: 0%"></span></span>
        <span class="val">0 <span class="max">/ ${axis.max}</span></span>
      `;
      els.proofRows.appendChild(row);
      proofRefs[axis.name] = {
        fill: row.querySelector(".fill"),
        val: row.querySelector(".val"),
        max: axis.max,
      };
    }
  }
  // Trailing summary line: yaw / pitch coverage. Challenge counts hidden
  // in proctoring mode (no challenges issued).
  const summary = document.createElement("div");
  summary.className = "proof-summary";
  const challengeSpan = SHOW_ACTIVE_SECTION
    ? `<span id="proofChallengeStats">challenges: 0 passed · 0 failed</span>`
    : `<span id="proofMode" style="color: var(--accent); opacity: 0.7">passive proctoring mode</span>`;
  summary.innerHTML = `
    <span id="proofMotion">head motion: —</span>
    ${challengeSpan}
  `;
  els.proofRows.appendChild(summary);
  els.proofMotion = $("proofMotion");
  els.proofChallengeStats = $("proofChallengeStats");
}
buildProofPanel();

const CATEGORIES = [
  "static_image",
  "video_replay",
  "mask_3d",
  "heavy_makeup",
  "ar_filter",
  "deepfake_injection",
];
const catRefs = {};
for (const cat of CATEGORIES) {
  const row = document.createElement("div");
  row.className = "cat-row";
  row.innerHTML = `
    <span>${cat.replace(/_/g, " ")}</span>
    <span class="bar"><span class="fill" style="width: 0%"></span></span>
    <span class="val">0%</span>
  `;
  els.cats.appendChild(row);
  catRefs[cat] = {
    fill: row.querySelector(".fill"),
    val: row.querySelector(".val"),
  };
}

let detector = null;
let canvas = null;
let ctx = null;
let running = false;
let smoothedFps = 0;
let lastTs = 0;
let lastVerdict = null;
let lastAnalyzerScores = null;
let lastGateResult = null;
let lastProof = null;
let knownIncidentIds = new Set();

// === Recording state (Phase E — minimal video + analytics playback) ===
// Recording is OFF by default. When the user clicks ⏺ Record session:
//   * MediaRecorder captures the camera stream to a webm in-memory buffer
//   * sessionTimeline.push({t, verdict, analyzers, proof}) on each frame
// On ⏹ Stop recording, we trigger two downloads (.webm + .json) and
// release the recorder. The two files together form a deterministic
// "session demo" — replay the video alongside the analytics log to
// audit any session frame-by-frame.
let mediaRecorder = null;
let recordedChunks = [];
// sessionTimeline now collects EVERY frame's analyzer snapshot for the entire
// session — not just when video-recording is on. Powers the ↓ Report download's
// research-grade `frame_log` time series (LIVE vs REPLAY separability work).
// Bounded at MAX_FRAME_LOG entries; on overflow we drop the oldest frame so
// the most recent session window is always retained.
let sessionTimeline = [];
const MAX_FRAME_LOG = 18000; // ~10 min at 30 fps, ~20 min at 15 fps
// Every fired auto/manual flash-temporal probe is appended here so the ↓ Report
// download exposes the probe time-series (baseline brightness, rise, post-flash
// persistence) — the discriminator that distinguishes a real face from a phone
// replay on PC. Without history every report only carries the *latest* probe.
let flashProbeHistory = [];
let recordingActive = false;
// Gate-stability smoother — the per-frame gate state oscillates between
// CLEAR and OCCLUDED_PENDING under modest landmark jitter. We keep the
// "usable" flag visible to the user only after it's held for >=5 frames,
// so transient single-frame flips don't flash the advisory banner.
const GATE_STABILITY_FRAMES = 5;
let gatePendingUsable = null;
let gatePendingFrames = 0;
let gateStableUsable = null;

function setStatus(label, kind = "live") {
  els.status.textContent = label;
  els.dot.className = `status-dot ${kind === "error" ? "error" : "live"}`;
}

async function ensureDetector() {
  if (detector) return detector;
  setStatus("loading models…");
  els.start.disabled = true;
  els.start.textContent = "loading…";

  detector = await createSpoofDetector({
    miniFasNetModelUrl: "./models/minifasnet_v2.onnx",
    faceLandmarkerTaskUrl: "./models/face_landmarker.task",
    mediaPipeWasmBaseUrl: MEDIAPIPE_WASM_BASE,
    ortExecutionProviders: ["wasm"],
    numFaces: 1,
    useGpu: true,
    // Proctoring profile — passive observation only, no mid-session
    // challenge prompts. The new passive axes (eye_motion, mouth_motion,
    // face_motion) plus the looser gates below let a natural live face
    // reach the 60-point proven-live threshold without performing.
    enableLivenessChallenges: false,
    livenessProverThresholds: {
      expressionRatioGate: 0.4,
      rotationThreshold: 2.0,
      landmarkVarThreshold: 0.5,
    },
    // Phase D1 — on by default for the demo page so the new
    // background-motion row populates. The ~250 KB SelfieSegmenter
    // model lazy-loads from MediaPipe's CDN on first frame.
    enableBackgroundSegmentation: true,
    // Phase D2 — hand tracking pulls a ~6 MB model. Off by default for
    // the demo page; enable with ?hand=1 in the URL to test.
    enableHandTracking:
      new URLSearchParams(window.location.search).get("hand") === "1",
    // Phase D3 — audio capture for VAD + audio-mouth sync. Off by
    // default for the demo page (mic permission UX); enable with
    // ?audio=1 in the URL OR click the "Enable mic" button.
    enableAudio:
      new URLSearchParams(window.location.search).get("audio") === "1",
  });
  return detector;
}

async function start() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    els.video.srcObject = stream;
    await new Promise((resolve) =>
      els.video.addEventListener("loadeddata", resolve, { once: true }),
    );
    await els.video.play();

    // Snapshot the camera's initial (auto) settings once — used by the 🎛
    // panel's "Auto (reset all)" and the on-exit restore so the webcam is
    // never stranded in a manual/locked state after experimentation.
    try {
      const vt0 = stream.getVideoTracks && stream.getVideoTracks()[0];
      if (vt0 && vt0.getSettings) cameraDefaults = vt0.getSettings();
    } catch (e) {
      /* ignore */
    }

    canvas = document.createElement("canvas");
    canvas.width = els.video.videoWidth || 640;
    canvas.height = els.video.videoHeight || 480;
    ctx = canvas.getContext("2d", { willReadFrequently: true });

    // Match overlay pixel buffer to the source canvas so bbox + landmark
    // coords drawn on the overlay land exactly on the visible face — both
    // are CSS-stretched identically by the browser, so the pixel-space
    // math stays correct.
    els.overlay.width = canvas.width;
    els.overlay.height = canvas.height;

    await ensureDetector();

    // Enter PREVIEW (readiness check) — the session does NOT start until the
    // readiness checks pass and the user clicks "Begin session".
    els.start.style.display = "none";
    if (els.beginSession) {
      els.beginSession.style.display = "";
      els.beginSession.disabled = true;
    }
    if (els.readinessPanel) els.readinessPanel.style.display = "block";
    els.stop.disabled = false;
    els.videoWrap.dataset.state = "running";
    setStatus("readiness check…", "live");

    previewMode = true;
    readinessStreak = 0;
    running = true;
    loop();

    // Auto-record (?autorec=1): kick the recorder once the camera + loop
    // are running. The first ondataavailable fires ~1 s later, so we need
    // the stream to be live before clicking. Done after running = true
    // so the loop is collecting timeline samples by the time chunks land.
    const autoRec =
      new URLSearchParams(window.location.search).get("autorec") === "1";
    if (autoRec && els.recordToggle && !recordingActive) {
      // Defer one tick so the user sees the page settle first; also lets
      // any rAF/getUserMedia plumbing finish before MediaRecorder starts.
      setTimeout(() => {
        if (running && !recordingActive) els.recordToggle.click();
      }, 200);
    }
  } catch (err) {
    console.error(err);
    setStatus(`error: ${err.message || err}`, "error");
    els.start.disabled = false;
    els.start.textContent = "Start";
  }
}

async function stop() {
  running = false;
  // If a recording is still running (auto-record mode or the user
  // forgot to click ⏹), stop it so the .webm + .json downloads fire
  // automatically. The MediaRecorder.onstop handler does the work.
  if (recordingActive && mediaRecorder && mediaRecorder.state !== "inactive") {
    try {
      mediaRecorder.stop();
    } catch (err) {
      console.error("auto-stop recording failed", err);
    }
  }
  // Restore the camera to auto + default image params BEFORE releasing the
  // stream, so stopping never leaves the webcam stuck in a manual/locked state.
  await restoreCameraDefaults();
  const stream = els.video.srcObject;
  if (stream && typeof stream.getTracks === "function") {
    for (const track of stream.getTracks()) track.stop();
  }
  els.video.srcObject = null;
  els.stop.disabled = true;
  els.start.style.display = "";
  els.start.disabled = false;
  els.start.textContent = "Start";
  // Clear any preview / readiness state so a fresh Start re-runs the check.
  previewMode = false;
  readinessStreak = 0;
  if (els.beginSession) {
    els.beginSession.style.display = "none";
    els.beginSession.disabled = true;
  }
  if (els.readinessPanel) els.readinessPanel.style.display = "none";
  els.videoWrap.dataset.state = "idle";
  setStatus("stopped", "live");
}

async function loop() {
  if (!running) return;
  // === Page-visibility guard ===
  // Mobile browsers (Chrome/Brave Android, Safari iOS) pause the
  // MediaStreamTrack when the tab is backgrounded but keep firing the
  // requestAnimationFrame callback that drives this loop. Without this
  // check the detector would run on the LAST frame the camera produced
  // before the freeze, accumulating fake "static photo" incidents and
  // decaying the proof score for what is just a tab-switch — and worse,
  // opening a brief window where an attacker could swap the user out
  // while the system kept reporting LIVE. We skip frames while hidden
  // and let the SessionEngine's incident detectors see a true gap
  // rather than synthetic static frames. The session clock keeps
  // running, so accumulated worst-window evidence stays valid.
  if (typeof document !== "undefined" && document.hidden) {
    requestAnimationFrame(loop);
    return;
  }
  // === Flash-probe guard ===
  // While an active flash probe runs (~3 s of locked exposure + a white
  // screen flash), the face washes out / drops from detection. Feeding those
  // frames to the SessionEngine spawns false "face missing" / "static image"
  // incidents that latch the peak-sensitive verdict to SPOOF — a self-inflicted
  // false reject. The probe samples the video itself (it doesn't need the
  // detector), so we PAUSE the passive analysis for its duration and let the
  // engine see a clean gap, exactly like the tab-hidden case above.
  if (flashBusy) {
    requestAnimationFrame(loop);
    return;
  }
  try {
    ctx.drawImage(els.video, 0, 0, canvas.width, canvas.height);
    const analysis = await detector.analyzeFrame(canvas);
    lastFaceBbox =
      analysis.faces && analysis.faces[0] ? analysis.faces[0].bbox : null;
    const v = detector.getVerdict();
    lastVerdict = v;

    // === Pre-flight readiness (preview) ===
    // Before the session begins we only evaluate capture readiness and gate
    // the "Begin session" button — no verdict, no active flash probe. The
    // engine still ingests frames (cheap) but that accumulation is discarded by
    // detector.reset() when the session actually begins.
    if (previewMode) {
      drawOverlay(analysis, v);
      evaluateReadiness(analysis);
      if (running) requestAnimationFrame(loop);
      return;
    }

    // Auto flash-response temporal probe — run a few seconds past warmup, then
    // on a (longer, since it's intrusive) cadence, to catch a screen/video-
    // replay swapped in mid-session. Fire-and-forget; it restores the camera.
    if (running && !flashProbe.running && !flashBusy) {
      const nowP = performance.now();
      if (
        (!flashProbe.ran && v.frames_analyzed >= FLASH_PROBE_AT_FRAME) ||
        (flashProbe.ran && nowP - flashProbe.lastAt > FLASH_PROBE_INTERVAL_MS)
      ) {
        runFlashProbe(true); // auto=true: silent, no button toggling
      }
    }
    drawOverlay(analysis, v);
    updateUI(analysis, v);
  } catch (err) {
    console.error("frame error", err);
    setStatus(`frame error: ${err.message || err}`, "error");
  }
  if (running) requestAnimationFrame(loop);
}

// ===== Pre-flight readiness evaluation (preview phase) =====
// Maps the per-frame analysis onto the SDK ReadinessGate, renders the live
// checklist, and enables "Begin session" only after the checks stay green for
// READINESS_STABLE_FRAMES consecutive frames (debounce against per-frame jitter).
function evaluateReadiness(analysis) {
  const faces = analysis.faces || [];
  const bbox = faces[0] ? faces[0].bbox : null;
  const frameArea = (canvas.width || 1) * (canvas.height || 1);
  const faceAreaFraction = bbox && bbox.area ? bbox.area / frameArea : 0;
  const gate = analysis.gate_result || null;
  const result = readinessGate.evaluate({
    faceCount: faces.length,
    faceAreaFraction,
    faceBrightness: gate ? gate.globalFaceBrightness : 0,
    occluded: gate ? !!gate.occluded : false,
    occludedRegions: gate ? gate.occludedRegions || [] : [],
    cameraResponsive: !!(els.video && els.video.readyState >= 2 && !els.video.paused),
  });
  renderReadiness(result);

  if (result.ready) readinessStreak += 1;
  else readinessStreak = 0;
  const stable = readinessStreak >= READINESS_STABLE_FRAMES;
  if (els.beginSession) els.beginSession.disabled = !stable;
  if (els.readinessHeadline) {
    els.readinessHeadline.textContent = stable
      ? "Ready — click “Begin session”"
      : result.ready
        ? "Holding steady…"
        : "Getting ready — fix the items below";
    els.readinessHeadline.className = "gate-headline " + (stable ? "ok" : "warn");
  }
  if (els.verdictText) els.verdictText.textContent = "Readiness check — see checklist";
}

function renderReadiness(result) {
  if (!els.readinessList) return;
  els.readinessList.innerHTML = result.checks
    .map((c) => {
      const color = c.pass ? "var(--green)" : "var(--amber)";
      const mark = c.pass ? "✓" : "•";
      return (
        `<div class="row"><span>${mark} ${c.label}</span>` +
        `<span style="color:${color};text-align:right">${c.message}</span></div>`
      );
    })
    .join("");
}

/** Leave PREVIEW and start the real session (discarding preview accumulation). */
function beginSession() {
  if (!previewMode || !detector) return;
  previewMode = false;
  readinessStreak = 0;
  detector.reset(); // discard frames accumulated during readiness preview
  if (els.beginSession) {
    els.beginSession.disabled = true;
    els.beginSession.style.display = "none";
  }
  if (els.readinessPanel) els.readinessPanel.style.display = "none";
  els.reset.disabled = false;
  els.download.disabled = false;
  els.videoWrap.dataset.state = "running";
  setStatus("running", "live");
}

if (els.beginSession) {
  els.beginSession.addEventListener("click", beginSession);
}

// Visibility change handler — pauses/resumes the run loop and, on
// return, recovers a *frozen camera stream*. Mobile browsers (Chrome
// / Brave Android, Safari iOS) commonly suspend the
// MediaStreamTrack while the tab is backgrounded — when the tab
// returns, the <video> element is still paused or the track is in
// "muted" state, so analyzeFrame keeps draining the same captured
// frame for many seconds. That looks like a static-photo attack to
// the engine and decays the proof score (user observed 47/100 after
// a 130s background interval). We:
//   1. Call .play() on the <video> if it's paused.
//   2. If the underlying MediaStreamTrack is "ended", re-acquire via
//      getUserMedia and swap in the new stream.
//   3. Surface the recovery status in the pill so the user knows.
async function recoverCameraStream() {
  try {
    const v = els.video;
    if (!v) return;
    if (v.paused) await v.play().catch(() => {});
    const stream = /** @type {MediaStream|null} */ (v.srcObject);
    const track = stream?.getVideoTracks?.()[0];
    const trackDead =
      !stream || !stream.active || !track || track.readyState === "ended";
    if (trackDead) {
      setStatus("camera reacquiring…", "live");
      const fresh = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user",
        },
        audio: false,
      });
      v.srcObject = fresh;
      await v.play().catch(() => {});
    }
    setStatus("running", "live");
  } catch (err) {
    console.error("camera recovery failed", err);
    setStatus(`camera recovery failed: ${err.message || err}`, "error");
  }
}

if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (!running) return;
    if (document.hidden) {
      setStatus("paused (tab hidden)", "live");
    } else {
      // Don't trust that the camera survived the background interval.
      // Asynchronously re-play / re-acquire as needed; the run loop
      // resumes on the next requestAnimationFrame tick regardless.
      void recoverCameraStream();
    }
  });
}

// Draw the detected face bbox + corner ticks + a 478-point landmark dot cloud
// onto the overlay canvas. Visual proof the ML is actually running on each
// frame; color flips with the verdict so the overlay echoes the headline.
function drawOverlay(analysis, v) {
  const octx = els.overlay.getContext("2d");
  if (!octx) return;
  octx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  if (!analysis.faces || analysis.faces.length === 0) return;

  const warming = v.frames_analyzed < 30;
  // Amber for warming OR quality-uncertain; green for live; red ONLY for a real
  // spoof verdict (not a poor-quality capture).
  const color = warming || (!v.is_live && v.quality_uncertain)
    ? "rgba(210, 153, 34, 0.95)"
    : v.is_live
      ? "rgba(63, 185, 80, 0.95)"
      : "rgba(248, 81, 73, 0.95)";

  for (const face of analysis.faces) {
    const { x1, y1, x2, y2 } = face.bbox;
    octx.lineWidth = 2;
    octx.strokeStyle = color;

    // L-shaped corner ticks (cleaner than a full box).
    const t = Math.min(28, (x2 - x1) * 0.18, (y2 - y1) * 0.18);
    octx.beginPath();
    octx.moveTo(x1, y1 + t); octx.lineTo(x1, y1); octx.lineTo(x1 + t, y1);
    octx.moveTo(x2 - t, y1); octx.lineTo(x2, y1); octx.lineTo(x2, y1 + t);
    octx.moveTo(x2, y2 - t); octx.lineTo(x2, y2); octx.lineTo(x2 - t, y2);
    octx.moveTo(x1 + t, y2); octx.lineTo(x1, y2); octx.lineTo(x1, y2 - t);
    octx.stroke();

    // 478-point landmark cloud (1-pixel dots).
    if (face.landmarks && face.landmarks.length >= 2) {
      octx.fillStyle = color.replace("0.95", "0.55");
      const lm = face.landmarks;
      for (let i = 0; i < lm.length; i += 2) {
        octx.fillRect(lm[i] - 0.5, lm[i + 1] - 0.5, 1.5, 1.5);
      }
    }
  }
}

function updateUI(analysis, v) {
  const now = performance.now();
  if (lastTs > 0) {
    const dt = now - lastTs;
    const inst = 1000 / Math.max(dt, 1);
    smoothedFps = smoothedFps === 0 ? inst : smoothedFps * 0.9 + inst * 0.1;
  }
  lastTs = now;

  els.verdictText.textContent = displaySummary(v);
  els.verdictConf.textContent = `${displayConfPct(v.confidence)}% conf`;
  const warming = v.frames_analyzed < 30;
  // UNCERTAIN (poor-quality capture) is its own state — amber, not red. Only a
  // genuine spoof verdict (is_live=false AND not quality_uncertain) goes red.
  const uncertain = !warming && !v.is_live && v.quality_uncertain;
  els.verdict.classList.toggle("live", !warming && v.is_live);
  els.verdict.classList.toggle("spoof", !warming && !v.is_live && !v.quality_uncertain);
  els.verdict.classList.toggle("uncertain", uncertain);
  els.verdict.classList.toggle("warming", warming);

  // Integrated active probe: the flash-response test. Two paths flag a screen:
  //   (1) TEMPORAL — brightness ramped up slowly and/or stayed elevated after
  //       the flash (auto-brightness latch). The primary signal.
  //   (2) OVER-LIT FALLBACK — a phone screen displaying a face is itself a
  //       bright light source → baseline saturates → the temporal probe
  //       abstains (no headroom). Live finding 2026-05-31: a video replay
  //       trips the over-lit abstain (rise ≈ 5, refl ≈ 7) and the system then
  //       reads LIVE 95% because no defense fires. Use the SPATIAL reflection
  //       score as a fallback signal.
  //
  //       BUT: live in a sun-lit office a REAL face also drives baselineMean
  //       past 185 and refl below 20 (the diffuse-reflection signal collapses
  //       when ambient light already saturates the sensor — there's no flash
  //       headroom to read). Calling SPOOF in that case false-rejects real
  //       users (2026-05-31 confirmed: user's real face flipped SPOOF in a
  //       bright office). The honest disposition is UNCERTAIN — "we can't
  //       judge in this lighting, dim the room and re-test" — never LIVE
  //       (so an over-lit-room attacker still can't authenticate) and never
  //       a confident SPOOF (so a real user isn't falsely rejected).
  //       SessionEngine already has a quality_uncertain path; we mirror its
  //       visual treatment here.
  if (flashProbe.ran && flashProbe.result) {
    const t = flashProbe.result;
    const refl = flashProbe.reflection;
    const overLitInconclusive =
      t.inconclusive &&
      t.baselineMean > 185 &&
      refl &&
      refl.score < 20;
    if (t.isScreen) {
      els.verdictText.textContent =
        `SPOOF (video-replay — flash persistence ${t.persistenceNorm} · ` +
        `screen ${t.screenScore} · onset ${t.onsetLagMs}ms)`;
      els.verdict.classList.remove("live", "warming", "uncertain");
      els.verdict.classList.add("spoof");
    } else if (overLitInconclusive) {
      els.verdictText.textContent =
        `UNCERTAIN — over-lit (baseline ${t.baselineMean.toFixed(0)}, ` +
        `refl ${refl.score}). Dim the room (close blinds / move away from windows) and re-test.`;
      els.verdict.classList.remove("live", "warming", "spoof");
      els.verdict.classList.add("uncertain");
    }
  }

  els.state.textContent = warming ? "warming_up" : "analyzing";
  els.frames.textContent = String(v.frames_analyzed);
  els.fps.textContent = smoothedFps.toFixed(1);
  els.duration.textContent = `${v.session_duration_sec.toFixed(1)}s`;
  els.faces.textContent = String(analysis.faces.length);
  els.blinks.textContent = String(v.blink_count ?? 0);
  els.incidents.textContent = String(v.incidents.length);

  // Per-analyzer scores
  const faces = Object.values(analysis.classifications);
  const analyzerResults = faces[0]?.analyzer_results ?? {};
  const snapshot = {};
  for (const cfg of ANALYZER_ORDER) {
    const r = analyzerResults[cfg.name];
    const ref = analyzerRefs[cfg.name];
    if (!ref) continue;
    if (!r) {
      ref.fill.style.width = "0%";
      ref.val.textContent = "—";
      snapshot[cfg.name] = null;
      continue;
    }
    const score = Math.max(0, Math.min(100, r.score));
    ref.fill.style.width = `${score.toFixed(0)}%`;
    ref.val.textContent = score.toFixed(0);
    // Refresh per-row tooltip with the analyzer's live detail breakdown.
    // The static description from ANALYZER_ORDER is kept as the first
    // line; per-frame numbers append after a separator so hovering any
    // row reveals the same per-region/per-axis data that ships in the
    // downloaded JSON (eye_var, mouth_var, forehead_var, tremor_x, etc.).
    ref.row.title = analyzerTooltip(cfg, r);
    snapshot[cfg.name] = {
      score: Math.round(score * 10) / 10,
      details: r.details ?? null,
    };
  }
  lastAnalyzerScores = snapshot;

  // Liveness proof — read independent of analyzers, refreshes every frame.
  const proof = detector?.getProof?.() ?? null;
  renderProofPanel(proof);
  lastProof = proof;

  // Frame log — always on (was previously gated on `recordingActive`, which
  // meant a research-grade ↓ Report JSON had no time series). Lightweight:
  // we store references to existing JS objects; no JSON serialisation in the
  // hot path. The full structure is stringified once at download time. Bounded
  // ring buffer at MAX_FRAME_LOG (~10 min at 30 fps, ~20 min at 15 fps) so
  // hour-long sessions can't OOM the browser tab.
  if (sessionTimeline.length >= MAX_FRAME_LOG) {
    sessionTimeline.shift();
  }
  const primaryFace = analysis.faces && analysis.faces[0];
  sessionTimeline.push({
    t_ms: Math.round(performance.now()),
    t_sec: Number(v.session_duration_sec.toFixed(2)),
    frame_id: analysis.frame_id,
    is_live: v.is_live,
    confidence: Number(v.confidence.toFixed(3)),
    proof_total: proof ? Math.round(proof.score) : null,
    proof_breakdown: proof ? proof.details : null,
    analyzer_scores: snapshot,
    incident_count: v.incidents.length,
    // Geometry — lets the notebook bucket scores by face size / distance.
    face_bbox: primaryFace
      ? {
          x1: Math.round(primaryFace.bbox.x1),
          y1: Math.round(primaryFace.bbox.y1),
          x2: Math.round(primaryFace.bbox.x2),
          y2: Math.round(primaryFace.bbox.y2),
          area: Math.round(primaryFace.bbox.area),
        }
      : null,
    // Gate (advisory) — slowly-changing; cheap to snapshot. Read directly
    // from the analysis since the local `gate` const is assigned later in
    // this function.
    gate: analysis.gate_result
      ? {
          usable: analysis.gate_result.usable,
          state: analysis.gate_result.state,
          occlusion: Number((analysis.gate_result.occlusionScore ?? 0).toFixed(3)),
          illumination: Number((analysis.gate_result.illuminationScore ?? 0).toFixed(3)),
        }
      : null,
    fps: Math.round(smoothedFps * 10) / 10,
  });

  // Per-category P(spoof)
  for (const cat of CATEGORIES) {
    const p = v.category_scores?.[cat] ?? 0;
    catRefs[cat].fill.style.width = `${(p * 100).toFixed(0)}%`;
    catRefs[cat].val.textContent = `${(p * 100).toFixed(0)}%`;
  }

  // Incident ledger — append only
  const newIncidents = v.incidents.filter(
    (i) => !knownIncidentIds.has(i.id ?? `${i.timestamp}-${i.type}`),
  );
  if (newIncidents.length > 0) {
    if (knownIncidentIds.size === 0) els.incidentList.innerHTML = "";
    for (const inc of newIncidents) {
      const key = inc.id ?? `${inc.timestamp}-${inc.type}`;
      knownIncidentIds.add(key);
      const row = document.createElement("div");
      row.className = "incident";
      row.innerHTML = `<b>${inc.type ?? "incident"}</b> @ ${(inc.timestamp ?? 0).toFixed(1)}s — ${
        inc.description ?? ""
      }`;
      els.incidentList.prepend(row);
    }
  }

  // Detail dump (per-analyzer .details object).
  const detailDump = Object.fromEntries(
    Object.entries(analyzerResults).map(([k, r]) => [k, r.details ?? null]),
  );
  els.details.textContent = JSON.stringify(detailDump, null, 2);

  // Aysenur's face-usability gate result. Advisory only — never blocks
  // the main verdict. On mobile cameras at low FPS the per-region pixel
  // thresholds (calibrated for the Python desktop pipeline) tend to
  // false-positive on a perfectly live face, so the labels here are
  // softened and the gate's verdict is presented as a "second opinion"
  // rather than a hard block.
  const gate = analysis.gate_result;
  lastGateResult = gate ?? null;
  if (gate && els.gateBody) {
    // Stability smoother: only flip the displayed headline after the
    // raw gate verdict holds for GATE_STABILITY_FRAMES frames.
    if (gate.usable === gatePendingUsable) {
      gatePendingFrames += 1;
    } else {
      gatePendingUsable = gate.usable;
      gatePendingFrames = 1;
    }
    if (
      gateStableUsable === null ||
      gatePendingFrames >= GATE_STABILITY_FRAMES
    ) {
      gateStableUsable = gatePendingUsable;
    }
    const displayUsable = gateStableUsable;
    const headlineText = displayUsable
      ? "Face usable"
      : `Advisory: ${gate.reason.replace(/_/g, " ")}`;
    const headlineClass = displayUsable ? "ok" : "warn";
    const lines = [
      `<div class="gate-headline ${headlineClass}">${headlineText}</div>`,
      `<div class="row"><span>State</span><span>${gate.state.replace(/_/g, " ").toLowerCase()}</span></div>`,
      `<div class="row"><span>Occlusion score</span><span>${(gate.occlusionScore * 100).toFixed(0)}%</span></div>`,
      `<div class="row"><span>Illumination score</span><span>${(gate.illuminationScore * 100).toFixed(0)}%</span></div>`,
    ];
    if (gate.occludedRegions.length > 0) {
      lines.push(
        `<div class="row"><span>Flagged regions</span><span>${gate.occludedRegions.join(", ")}</span></div>`,
      );
    }
    if (gate.underexposedRegions.length > 0) {
      lines.push(
        `<div class="row"><span>Under-exposed</span><span>${gate.underexposedRegions.join(", ")}</span></div>`,
      );
    }
    if (gate.overexposedRegions.length > 0) {
      lines.push(
        `<div class="row"><span>Over-exposed</span><span>${gate.overexposedRegions.join(", ")}</span></div>`,
      );
    }
    els.gateBody.innerHTML = lines.join("");
    if (els.gateBanner) {
      els.gateBanner.classList.toggle("warn", !displayUsable);
    }
  }
}

function reset() {
  if (!detector) return;
  detector.reset();
  smoothedFps = 0;
  lastTs = 0;
  lastProof = null;
  knownIncidentIds = new Set();
  // Clear the active probes so a fresh session re-probes from scratch.
  flashProbe.ran = false;
  flashProbe.running = false;
  flashProbe.result = null;
  flashProbe.reflection = null;
  flashProbe.screenSuspected = false;
  wbProbe.ran = false;
  wbProbe.screenSuspected = false;
  wbProbe.cast = null;
  // Drop accumulated research-mode buffers — a Reset starts a clean
  // dataset row, otherwise the next ↓ Report would mix LIVE and REPLAY
  // frames in the same JSON.
  sessionTimeline = [];
  flashProbeHistory = [];
  els.incidentList.innerHTML =
    '<div class="incident">No incidents yet.</div>';
  renderProofPanel(null);
  setStatus("running", "live");
}

function download() {
  const verdict = lastVerdict ?? detector?.getVerdict() ?? null;
  if (!verdict) return;
  // Collect the labels/notes the operator typed before pressing Record. These
  // are baked into the JSON so the offline notebook can build a labelled
  // LIVE-vs-REPLAY matrix without manual relabeling — the #1 cause of dirty
  // datasets in small-N research is misremembered post-hoc labels.
  const captureLabel = els.captureLabel?.value || "UNLABELED";
  const environment = {
    capture_label: captureLabel,
    ambient_label: (els.ambientLabel?.value || "").trim() || null,
    replay_device: (els.replayDevice?.value || "").trim() || null,
    notes: (els.captureNotes?.value || "").trim() || null,
    camera_settings: captureCameraSettingsSnapshot(),
  };
  const payload = {
    generated_at: new Date().toISOString(),
    user_agent: navigator.userAgent,
    amispoof_version: window.AMISPOOF_VERSION,
    schema_version: 2, // bumped: frame_log + environment + flash_probe_history added
    environment,
    verdict,
    latest_analyzer_scores: lastAnalyzerScores,
    latest_gate_result: lastGateResult,
    latest_liveness_proof: lastProof,
    fps_smoothed: Math.round(smoothedFps * 10) / 10,
    // Research-grade time series. The latest_* fields are the FINAL frame
    // only; frame_log is the per-frame trajectory the notebook needs to
    // compute per-feature AUC, correlation matrices, and temporal models.
    frame_log: sessionTimeline.slice(),
    frame_log_truncated: sessionTimeline.length >= MAX_FRAME_LOG,
    flash_probe_history: flashProbeHistory.slice(),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  // File name carries the capture label so a folder of dumps sorts/groups
  // by class without the user having to open each one.
  const labelSlug = captureLabel.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
  a.download = `amispoof-session-${labelSlug}-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Best-effort snapshot of the current MediaStreamTrack's getSettings().
 * Returns null silently if no camera stream is attached yet — separability
 * analysis treats null camera settings as "unknown", not a failure.
 */
function captureCameraSettingsSnapshot() {
  try {
    const stream = els.video?.srcObject;
    if (!stream) return null;
    const track = stream.getVideoTracks?.()[0];
    if (!track || typeof track.getSettings !== "function") return null;
    const s = track.getSettings();
    return {
      width: s.width ?? null,
      height: s.height ?? null,
      frame_rate: s.frameRate ?? null,
      exposure_mode: s.exposureMode ?? null,
      exposure_time: s.exposureTime ?? null,
      white_balance_mode: s.whiteBalanceMode ?? null,
      color_temperature: s.colorTemperature ?? null,
      brightness: s.brightness ?? null,
      contrast: s.contrast ?? null,
      saturation: s.saturation ?? null,
      sharpness: s.sharpness ?? null,
    };
  } catch {
    return null;
  }
}

els.start.addEventListener("click", start);
els.stop.addEventListener("click", stop);
els.reset.addEventListener("click", reset);
els.download.addEventListener("click", download);

// ---------- Phase 5E-3: in-page accuracy bench ----------
// 5 live + 5 spoof placeholder URLs under ./samples/. The actual sample
// fixtures land in a follow-up commit — until they're shipped the bench
// will surface a "failed to load" status for each row, which is intentional
// and harmless (the harness reports the error per-row, not at the top).
const BENCH_SAMPLE_URLS = [
  "./samples/live_01.jpg",
  "./samples/live_02.jpg",
  "./samples/live_03.jpg",
  "./samples/live_04.jpg",
  "./samples/live_05.jpg",
  "./samples/spoof_01.jpg",
  "./samples/spoof_02.jpg",
  "./samples/spoof_03.jpg",
  "./samples/spoof_04.jpg",
  "./samples/spoof_05.jpg",
];

async function runBench() {
  els.bench.disabled = true;
  const originalLabel = els.bench.textContent;
  els.bench.textContent = "running…";
  els.benchPanel.style.display = "block";
  els.benchHeadline.textContent = "Bench: running…";
  els.benchRows.innerHTML = "";
  try {
    // Pre-flight: confirm sample images exist before warming up the
    // detector. CASIA-FASD samples are NOT bundled in this deploy
    // (license + size) — without this check the bench fails per-row
    // with confusing fetch errors. Probing one URL is cheap.
    const probe = await fetch(BENCH_SAMPLE_URLS[0], { method: "HEAD" });
    if (!probe.ok) {
      els.benchHeadline.textContent =
        "Bench unavailable: sample images (./samples/live_*.jpg, spoof_*.jpg) are not bundled with this build. The bench harness ships in the SDK; this page deliberately omits the dataset.";
      els.benchRows.innerHTML = "";
      return;
    }
    await ensureDetector();
    const result = await runCasiaFasdMicroBench(detector, BENCH_SAMPLE_URLS);
    const pct = Math.round(result.accuracy * 100);
    els.benchHeadline.textContent = `Bench: ${result.correct}/${result.total} correct (${pct}%)`;
    els.benchRows.innerHTML = result.perSample
      .map((row) => {
        const ok = row.got === row.expected ? "✓" : "✗";
        const okColor =
          row.got === row.expected ? "var(--green)" : "var(--red)";
        const conf = (row.confidence * 100).toFixed(0);
        const fname = row.url.split("/").pop() ?? row.url;
        return `<div style="font-size:11px; padding:2px 0">
          <span style="color:${okColor}; display:inline-block; width:14px">${ok}</span>
          <code>${fname}</code> — expected <b>${row.expected}</b>, got <b>${row.got}</b>
          (${conf}% conf)
        </div>`;
      })
      .join("");
    // After the bench, the SessionEngine sits on the last sample's state —
    // reset so the user can resume normal live-camera analysis cleanly.
    detector.reset();
    smoothedFps = 0;
    lastTs = 0;
  } catch (err) {
    console.error("bench error", err);
    els.benchHeadline.textContent = `Bench error: ${err.message || err}`;
  } finally {
    els.bench.disabled = false;
    els.bench.textContent = originalLabel;
  }
}

if (els.bench) {
  els.bench.addEventListener("click", runBench);
}

// ===== Active-illumination (opt-in) flash challenge =====
// Locks camera exposure (so auto-exposure can't compensate), captures a
// baseline face crop, flashes the screen, captures the lit crop, then scores
// the photometric response with FlashReflectionAnalyzer. A real 3D face
// reflects the flash diffusely; a screen/replay emits its own light and won't.
// Opt-in only — never part of the passive proctoring verdict.
const flashAnalyzer = new FlashReflectionAnalyzer();
let flashBusy = false;
const flashSleep = (ms) => new Promise((r) => setTimeout(r, ms));

function captureFaceCrop() {
  const w = els.video.videoWidth;
  const h = els.video.videoHeight;
  if (!w || !h) return null;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const cx = c.getContext("2d");
  if (!cx) return null;
  cx.drawImage(els.video, 0, 0, w, h);
  let bx0, by0, bw, bh;
  if (lastFaceBbox) {
    bx0 = Math.max(0, Math.floor(lastFaceBbox.x1));
    by0 = Math.max(0, Math.floor(lastFaceBbox.y1));
    bw = Math.min(w - bx0, Math.floor(lastFaceBbox.x2 - lastFaceBbox.x1));
    bh = Math.min(h - by0, Math.floor(lastFaceBbox.y2 - lastFaceBbox.y1));
  } else {
    // No face yet — fall back to a centred crop where the face usually sits.
    bw = Math.floor(w * 0.5);
    bh = Math.floor(h * 0.6);
    bx0 = Math.floor(w * 0.25);
    by0 = Math.floor(h * 0.2);
  }
  if (bw <= 4 || bh <= 4) return null;
  return cx.getImageData(bx0, by0, bw, bh);
}

/** Mean per-pixel brightness (max of R,G,B) over the current face crop, 0-255. */
function brightnessOfImageData(img) {
  if (!img) return null;
  const d = img.data;
  let s = 0;
  let n = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i];
    const g = d[i + 1];
    const b = d[i + 2];
    s += r > g ? (r > b ? r : b) : g > b ? g : b;
    n++;
  }
  return n ? s / n : null;
}

/** One face-region brightness sample from the live video. */
function sampleBrightness() {
  return brightnessOfImageData(captureFaceCrop());
}

// Choose a MID-RANGE exposureTime to lock and snap it onto the camera's
// [min,max] step grid. Two camera gotchas drive this:
//   • exposureMode:'manual' ALONE silently reverts to continuous, and an
//     unaligned exposureTime is rejected ("out of range") — so we must pass
//     a step-aligned value.
//   • getSettings().exposureTime is UNRELIABLE in continuous mode (this
//     webcam reports 10000 while caps cap at 1250). Locking to that clamps to
//     MAX exposure → the sensor saturates under the flash and the webcam's
//     auto-gain stays elevated afterwards, which mimics a screen's persistence
//     (a real face then false-flags as a replay). A mid-range lock leaves
//     headroom so a real face's brightness drops straight back after the flash.
function alignedExposureTime(track) {
  const caps = track.getCapabilities ? track.getCapabilities() : null;
  if (!caps || !caps.exposureMode || !caps.exposureMode.includes("manual")) {
    return null;
  }
  const et = caps.exposureTime;
  if (!et || typeof et.min !== "number") return 156;
  // ~45 % of the range: bright enough to read a reflection, far from saturation.
  let exp = et.min + (et.max - et.min) * 0.45;
  exp = Math.max(et.min, Math.min(et.max, exp));
  exp = et.min + Math.round((exp - et.min) / et.step) * et.step;
  return exp;
}

/**
 * Active flash-response probe (opt-in). Locks exposure, flashes the screen
 * white for ~1.5 s while sampling face-region brightness, then watches the
 * drop-back. Scores BOTH:
 *   • TEMPORAL (FlashTemporalAnalyzer) — onset lag + post-flash persistence.
 *     A screen's auto-brightness ramps slowly and stays elevated; a real face
 *     reflects instantly and drops straight back. The content-independent
 *     video-replay detector — this drives the verdict override.
 *   • SPATIAL (FlashReflectionAnalyzer) — diffuse region-spread of the lit
 *     crop, reported as supporting evidence.
 * `auto=true` runs it silently from the session loop (no button toggling).
 * Always restores auto-exposure, even on error.
 */
async function runFlashProbe(auto = false) {
  if (!running || !detector || flashBusy || flashProbe.running) return;
  flashBusy = true;
  flashProbe.running = true;
  if (!auto && els.lightCheck) els.lightCheck.disabled = true;
  const track =
    els.video.srcObject && els.video.srcObject.getVideoTracks
      ? els.video.srcObject.getVideoTracks()[0]
      : null;
  let lockedExposure = false;
  const setMsg = (m) => {
    if (els.lightResult) els.lightResult.textContent = m;
  };
  try {
    setMsg("💡 Flash probe: locking exposure…");
    // 1. Lock exposure so auto-exposure can't fight the flash.
    if (track) {
      const exp = alignedExposureTime(track);
      if (exp != null) {
        await track.applyConstraints({
          advanced: [{ exposureMode: "manual", exposureTime: exp }],
        });
        lockedExposure = track.getSettings().exposureMode === "manual";
        await flashSleep(350); // let the lock settle
      }
    }

    // 2. Baseline: a crop (for the spatial test) + brightness samples (temporal).
    const baselineCrop = captureFaceCrop();
    const baselineSamples = [];
    for (let i = 0; i < FLASH_BASELINE_SAMPLES; i++) {
      const b = sampleBrightness();
      if (b != null) baselineSamples.push(b);
      await flashSleep(FLASH_SAMPLE_MS);
    }

    // 3. Flash white; sample brightness across the flash, grab a lit crop near
    //    the end (settled) for the spatial reflection score.
    setMsg("💡 Flash probe: flashing…");
    els.flashOverlay.style.background = "#ffffff";
    els.flashOverlay.style.display = "block";
    const flashSamples = [];
    let peakCrop = null;
    for (let i = 0; i < FLASH_DURING_SAMPLES; i++) {
      const b = sampleBrightness();
      if (b != null) flashSamples.push(b);
      if (i === FLASH_DURING_SAMPLES - 2) peakCrop = captureFaceCrop();
      await flashSleep(FLASH_SAMPLE_MS);
    }
    els.flashOverlay.style.display = "none";

    // 4. After: brightness samples as the flash turns off (drop-back vs persist).
    const afterSamples = [];
    for (let i = 0; i < FLASH_AFTER_SAMPLES; i++) {
      const b = sampleBrightness();
      if (b != null) afterSamples.push(b);
      await flashSleep(FLASH_SAMPLE_MS);
    }

    // 5. Restore auto-exposure.
    if (lockedExposure && track) {
      try {
        await track.applyConstraints({ advanced: [{ exposureMode: "continuous" }] });
      } catch (e) {
        /* best-effort restore */
      }
    }

    // 6. Score temporal (drives the verdict) + spatial (supporting evidence).
    const temporal = flashTemporalAnalyzer.score(
      baselineSamples,
      flashSamples,
      afterSamples,
    );
    // Quality constraint: without a genuine exposure lock the webcam's own
    // auto-gain hunts after the flash and mimics a screen's persistence — so
    // persistence can't be trusted and we must ABSTAIN rather than risk a false
    // flag (this is what would otherwise bite cameras lacking manual exposure,
    // e.g. most mobile front cameras). On PC/Brave the lock engages, so this
    // never trips. Mirrors the FaceUsabilityGate principle: don't issue a
    // confident verdict on input we couldn't control.
    if (!lockedExposure) {
      temporal.inconclusive = true;
      temporal.isScreen = false;
    }
    const reflection =
      baselineCrop && peakCrop
        ? flashAnalyzer.scoreResponse(baselineCrop, peakCrop, "white")
        : null;
    flashProbe.result = temporal;
    flashProbe.reflection = reflection;
    flashProbe.screenSuspected = temporal.isScreen;
    flashProbe.lockFailed = !lockedExposure;
    flashProbe.ran = true;
    flashProbe.lastAt = performance.now();
    // Append this probe to the research-grade history so the ↓ Report
    // download exposes the full probe trajectory (not just the most
    // recent one). Each entry is shallow — the raw per-sample arrays live
    // on window.__flashProbeRaw for live debugging.
    flashProbeHistory.push({
      t_ms: Math.round(performance.now()),
      t_sec: Number((detector?.getVerdict()?.session_duration_sec ?? 0).toFixed(2)),
      auto,
      locked_exposure: lockedExposure,
      // Mirrors the FlashTemporalResult shape exposed by the SDK (see
      // FlashTemporalAnalyzer.ts). No invented fields — keeps the JSON
      // schema stable across SDK updates.
      baseline_mean: temporal.baselineMean ?? null,
      rise_total: temporal.riseTotal ?? null,
      onset_lag_ms: temporal.onsetLagMs ?? null,
      persistence_norm: temporal.persistenceNorm ?? null,
      screen_score: temporal.screenScore ?? null,
      is_screen: temporal.isScreen === true,
      inconclusive: temporal.inconclusive === true,
      reflection_score: reflection ? (reflection.score ?? null) : null,
    });

    // Debug hook for live calibration — exposes the raw brightness time-series
    // so thresholds can be tuned against real vs screen captures.
    window.__flashProbeRaw = { baselineSamples, flashSamples, afterSamples, temporal, lockedExposure };

    const overLit = temporal.baselineMean > 185;
    const verdict = !lockedExposure
      ? "INCONCLUSIVE — couldn't lock camera exposure (active probe needs a controllable camera)"
      : temporal.inconclusive
        ? overLit
          ? "INCONCLUSIVE — face is over-lit; the flash has no headroom (dim the room for the light check)"
          : "INCONCLUSIVE — flash didn't reach the face (dim the room / move closer)"
        : temporal.isScreen
          ? "SCREEN / VIDEO-REPLAY (auto-brightness ramp/persistence)"
          : "LIVE (instant reflection, no persistence)";
    setMsg(
      `💡 Flash probe: ${verdict} · onset ${temporal.onsetLagMs}ms · ` +
        `persistence ${temporal.persistenceNorm} · rise ${temporal.riseTotal} · ` +
        `screen ${temporal.screenScore}` +
        (reflection ? ` · refl ${reflection.score}` : ""),
    );
  } catch (e) {
    els.flashOverlay.style.display = "none";
    if (lockedExposure && track) {
      try {
        await track.applyConstraints({ advanced: [{ exposureMode: "continuous" }] });
      } catch (_) {
        /* ignore */
      }
    }
    setMsg("💡 Flash probe error: " + (e && e.message ? e.message : e));
  } finally {
    flashBusy = false;
    flashProbe.running = false;
    if (!auto && els.lightCheck) els.lightCheck.disabled = false;
  }
}

if (els.lightCheck) {
  els.lightCheck.addEventListener("click", () => runFlashProbe(false));
}
// Debug hook for live calibration — fire a probe on demand from the console.
window.__runFlashProbe = () => runFlashProbe(false);

// ===== Manual camera controls (operator experimentation) =====
// Builds sliders/toggles from the live track's getCapabilities() so the
// operator can lock/adjust exposure, white balance, colour temperature, and
// the image params to probe what reveals a screen spoof (e.g. locking WB
// exposes a phone screen's colour cast that auto-WB would otherwise hide).
function camTrack() {
  return els.video.srcObject && els.video.srcObject.getVideoTracks
    ? els.video.srcObject.getVideoTracks()[0]
    : null;
}

/**
 * Restore the camera to auto (continuous exposure + white balance) and the
 * image params (brightness/contrast/saturation/sharpness) captured at session
 * start. Used by "Auto (reset all)" and on Stop / page-exit so manual locks
 * never strand the webcam. Best-effort — a driver that latches a value may
 * still need a hardware/OS reset.
 */
async function restoreCameraDefaults() {
  const track = camTrack();
  if (!track) return;
  const c = { exposureMode: "continuous", whiteBalanceMode: "continuous" };
  if (cameraDefaults) {
    for (const k of ["brightness", "contrast", "saturation", "sharpness"]) {
      if (typeof cameraDefaults[k] === "number") c[k] = cameraDefaults[k];
    }
  }
  try {
    await track.applyConstraints({ advanced: [c] });
  } catch (e) {
    /* best-effort */
  }
}

async function applyCam(constraint) {
  const track = camTrack();
  if (!track) return;
  try {
    await track.applyConstraints({ advanced: [constraint] });
  } catch (e) {
    console.warn("[camera] constraint rejected", constraint, e);
  }
  refreshCamReadout();
}

function refreshCamReadout() {
  const el = document.getElementById("camReadout");
  const track = camTrack();
  if (!el || !track) return;
  const s = track.getSettings();
  el.textContent =
    `exposure ${s.exposureMode}/${Math.round(s.exposureTime || 0)} · ` +
    `WB ${s.whiteBalanceMode}/${s.colorTemperature || "-"}K · ` +
    `bri ${s.brightness} · con ${s.contrast} · sat ${s.saturation} · shp ${s.sharpness}`;
}

function buildCameraControls() {
  const track = camTrack();
  const panel = els.cameraPanel;
  if (!panel) return;
  if (!track || !track.getCapabilities) {
    panel.textContent = "Start the camera first.";
    return;
  }
  const caps = track.getCapabilities();
  const s = track.getSettings();
  panel.innerHTML = "";
  const row = () => {
    const d = document.createElement("div");
    d.style.cssText =
      "display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px;";
    return d;
  };
  const label = (t) => {
    const l = document.createElement("label");
    l.textContent = t;
    l.style.cssText = "width:120px;flex:0 0 120px;";
    return l;
  };
  const addMode = (key, text) => {
    if (!Array.isArray(caps[key]) || caps[key].length < 2) return;
    const r = row();
    r.appendChild(label(text));
    const sel = document.createElement("select");
    caps[key].forEach((m) => {
      const o = document.createElement("option");
      o.value = m;
      o.textContent = m;
      if (m === s[key]) o.selected = true;
      sel.appendChild(o);
    });
    sel.onchange = () => applyCam({ [key]: sel.value });
    r.appendChild(sel);
    panel.appendChild(r);
  };
  const addRange = (key, text, controllingMode) => {
    const c = caps[key];
    if (!c || typeof c.min !== "number") return;
    const r = row();
    r.appendChild(label(text));
    const input = document.createElement("input");
    input.type = "range";
    input.min = c.min;
    input.max = c.max;
    input.step = c.step || 1;
    input.value = s[key] != null ? s[key] : c.min;
    input.style.flex = "1";
    const val = document.createElement("span");
    val.style.cssText = "width:64px;text-align:right;";
    val.textContent = String(Math.round(input.value));
    input.oninput = () => {
      val.textContent = String(Math.round(input.value));
    };
    input.onchange = async () => {
      // The value only applies if the controlling mode is manual.
      if (controllingMode) {
        const t = camTrack();
        if (t) {
          try {
            await t.applyConstraints({ advanced: [{ [controllingMode]: "manual" }] });
          } catch (e) {
            /* ignore */
          }
        }
      }
      applyCam({ [key]: parseFloat(input.value) });
    };
    r.appendChild(input);
    r.appendChild(val);
    panel.appendChild(r);
  };

  addMode("exposureMode", "Exposure mode");
  addRange("exposureTime", "Exposure time", "exposureMode");
  addMode("whiteBalanceMode", "White balance");
  addRange("colorTemperature", "Colour temp (K)", "whiteBalanceMode");
  addRange("brightness", "Brightness");
  addRange("contrast", "Contrast");
  addRange("saturation", "Saturation");
  addRange("sharpness", "Sharpness");

  const resetRow = row();
  const reset = document.createElement("button");
  reset.textContent = "Auto (reset all)";
  reset.className = "ghost";
  reset.style.fontSize = "12px";
  reset.onclick = async () => {
    await restoreCameraDefaults();
    buildCameraControls();
  };
  resetRow.appendChild(reset);
  panel.appendChild(resetRow);

  const ro = document.createElement("div");
  ro.id = "camReadout";
  ro.style.cssText =
    "font-size:11px;opacity:0.75;margin-top:8px;font-family:monospace;";
  panel.appendChild(ro);
  refreshCamReadout();
}

if (els.cameraToggle) {
  els.cameraToggle.addEventListener("click", () => {
    if (!els.cameraPanel) return;
    const open = els.cameraPanel.style.display !== "none";
    if (open) {
      els.cameraPanel.style.display = "none";
    } else {
      els.cameraPanel.style.display = "block";
      buildCameraControls();
    }
  });
}

// Best-effort: restore the camera to auto + default image params when the page
// is hidden/closed, so leaving never strands the webcam in a manual lock.
window.addEventListener("pagehide", () => {
  restoreCameraDefaults();
});

// Mean RGB over a fractional region of the live video frame.
function probeRegionRGB(x0f, y0f, x1f, y1f) {
  const vEl = els.video;
  const w = vEl.videoWidth;
  const h = vEl.videoHeight;
  if (!w || !h) return null;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const cx = c.getContext("2d");
  if (!cx) return null;
  cx.drawImage(vEl, 0, 0, w, h);
  const x0 = Math.max(0, Math.floor(x0f * w));
  const y0 = Math.max(0, Math.floor(y0f * h));
  const x1 = Math.min(w, Math.floor(x1f * w));
  const y1 = Math.min(h, Math.floor(y1f * h));
  if (x1 - x0 < 2 || y1 - y0 < 2) return null;
  const d = cx.getImageData(x0, y0, x1 - x0, y1 - y0).data;
  let R = 0,
    G = 0,
    B = 0,
    n = 0;
  for (let i = 0; i < d.length; i += 4) {
    R += d[i];
    G += d[i + 1];
    B += d[i + 2];
    n++;
  }
  return n ? { R: R / n, G: G / n, B: B / n } : null;
}

// Lock white balance to a fixed neutral temp, measure the face-region colour
// cast, restore auto WB. A strong cast ⇒ a screen/replay (its own white point
// isn't being corrected away). Always restores WB in `finally`.
async function runWbProbe() {
  if (!running || wbProbe.running) return;
  const track = camTrack();
  if (!track || !track.getCapabilities) return;
  const caps = track.getCapabilities();
  if (!(caps.whiteBalanceMode || []).includes("manual") || !caps.colorTemperature) {
    return;
  }
  wbProbe.running = true;
  try {
    const t = Math.max(
      caps.colorTemperature.min,
      Math.min(caps.colorTemperature.max, WB_PROBE_TEMP),
    );
    await track.applyConstraints({
      advanced: [{ whiteBalanceMode: "manual", colorTemperature: t }],
    });
    await flashSleep(450);
    let reg = null;
    if (lastFaceBbox && canvas) {
      reg = probeRegionRGB(
        lastFaceBbox.x1 / canvas.width,
        lastFaceBbox.y1 / canvas.height,
        lastFaceBbox.x2 / canvas.width,
        lastFaceBbox.y2 / canvas.height,
      );
    }
    if (!reg) reg = probeRegionRGB(0.3, 0.25, 0.7, 0.75);
    if (reg) {
      const rb = reg.R / Math.max(reg.B, 1);
      const gb = reg.G / Math.max(reg.B, 1);
      const cast = Math.abs(Math.log(rb)) + Math.abs(Math.log(gb));
      wbProbe.faceRB = +rb.toFixed(2);
      wbProbe.faceGB = +gb.toFixed(2);
      wbProbe.cast = +cast.toFixed(3);
      wbProbe.screenSuspected = cast >= (window.WB_CAST_THRESHOLD ?? 0.5);
      wbProbe.ran = true;
      wbProbe.lastAt = performance.now();
      if (els.lightResult) {
        els.lightResult.textContent =
          `🔬 Screen probe (WB-cast): R/B ${wbProbe.faceRB} · cast ${wbProbe.cast} · ` +
          (wbProbe.screenSuspected
            ? "SCREEN / VIDEO-REPLAY suspected"
            : "clear (real-scene colour)");
      }
    }
  } catch (e) {
    /* best-effort */
  } finally {
    try {
      await track.applyConstraints({ advanced: [{ whiteBalanceMode: "continuous" }] });
    } catch (e) {
      /* ignore */
    }
    wbProbe.running = false;
  }
}

// Phase D3 — wire the microphone toggle. The SDK requires audio to be
// enabled at construction time (see createSpoofDetector opts). When the
// page wasn't started with ?audio=1, clicking the button reloads with
// it set so the SDK re-initialises with audio. When it WAS, the click
// just calls detector.startAudio() to prompt for permission.
if (els.micToggle) {
  const audioPreEnabled =
    new URLSearchParams(window.location.search).get("audio") === "1";
  if (!audioPreEnabled) {
    els.micToggle.textContent = "🎤 Mic";
    els.micToggle.addEventListener("click", () => {
      const u = new URL(window.location.href);
      u.searchParams.set("audio", "1");
      window.location.href = u.toString();
    });
  } else {
    els.micToggle.addEventListener("click", async () => {
      try {
        await ensureDetector();
        if (detector.audioActive) {
          await detector.stopAudio();
          els.micToggle.textContent = "🎤 Enable mic";
        } else {
          await detector.startAudio();
          els.micToggle.textContent = "🎤 On";
        }
      } catch (e) {
        console.error("mic toggle failed", e);
        els.micToggle.textContent = "🎤 Failed";
      }
    });
  }
}

// === Hand tracking toggle (Phase D2) ===
// Same two-step UX as the mic button — the SDK needs the toggle at
// construction time, so the first click reloads with ?hand=1. After
// reload the button shows "✋ On" to acknowledge that
// the ~6 MB HandLandmarker model is wired and lazy-fetching.
if (els.handToggle) {
  const handPreEnabled =
    new URLSearchParams(window.location.search).get("hand") === "1";
  if (!handPreEnabled) {
    els.handToggle.textContent = "✋ Hand";
    els.handToggle.addEventListener("click", () => {
      const u = new URL(window.location.href);
      u.searchParams.set("hand", "1");
      window.location.href = u.toString();
    });
  } else {
    els.handToggle.textContent = "✋ On";
    els.handToggle.disabled = true;
  }
}

// === Record session (Phase E, experimental) ===
// Toggles MediaRecorder on the camera stream + a parallel per-frame
// analytics buffer. On stop, downloads two artefacts the user can
// later replay together: a .webm of the camera view and a .json log
// of every frame's scores. Either file is independently useful — the
// .json alone is a complete time-series of every analyzer + proof
// axis, enough for offline review without needing the video.
if (els.recordToggle) {
  els.recordToggle.addEventListener("click", async () => {
    try {
      if (recordingActive) {
        // Stop + download.
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
          mediaRecorder.stop();
        }
        return; // The onstop handler does the rest.
      }
      // Start. Requires that the camera session is running (else there's
      // no stream to record). The user clicks Start, *then* Record.
      const stream = els.video.srcObject;
      if (!stream) {
        els.recordToggle.textContent = "⏺ camera off";
        setTimeout(() => {
          els.recordToggle.textContent = "⏺ Rec";
        }, 2000);
        return;
      }
      recordedChunks = [];
      // sessionTimeline is no longer cleared at recording start — it now
      // tracks the entire session for the ↓ Report flow. The recording's
      // .json will still include the full timeline (recording is a superset
      // of the report use case).
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";
      mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
      };
      mediaRecorder.onstop = () => {
        recordingActive = false;
        els.recordToggle.textContent = "⏺ Rec";
        const stamp = Date.now();
        // 1) Download the webm.
        const videoBlob = new Blob(recordedChunks, { type: "video/webm" });
        downloadBlob(videoBlob, `amispoof-recording-${stamp}.webm`);
        // 2) Download the analytics timeline + a session footer.
        const finalVerdict = lastVerdict ?? detector?.getVerdict() ?? null;
        const finalProof = detector?.getProof?.() ?? null;
        const payload = {
          generated_at: new Date(stamp).toISOString(),
          user_agent: navigator.userAgent,
          amispoof_version: window.AMISPOOF_VERSION,
          recording_seconds:
            finalVerdict?.session_duration_sec ??
            (sessionTimeline.at(-1)?.t_sec ?? 0),
          frame_count: sessionTimeline.length,
          final_verdict: finalVerdict,
          final_proof: finalProof,
          timeline: sessionTimeline,
        };
        const jsonBlob = new Blob([JSON.stringify(payload, null, 2)], {
          type: "application/json",
        });
        downloadBlob(jsonBlob, `amispoof-recording-${stamp}.json`);
      };
      mediaRecorder.start(1000); // chunk every 1 s
      recordingActive = true;
      els.recordToggle.textContent = "⏹ Stop";
    } catch (err) {
      console.error("record error", err);
      els.recordToggle.textContent = "⏺ Failed";
      recordingActive = false;
      setTimeout(() => {
        els.recordToggle.textContent = "⏺ Rec";
      }, 2000);
    }
  });
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// === Session replay (Phase E) ===
// Loads a session JSON produced by the recorder and renders the final
// verdict + a tiny SVG line chart of proof_total over the session.
// Intentionally minimalist — no zoom, no scrub, no video sync — the
// goal here is "did this session ever drop below proven-live, and
// when?" not "build a full demo viewer". The recorded JSON is
// already a complete time-series; a fancier in-page viewer is a
// natural follow-up if reviewers need scrubbing.
function renderReplay(payload) {
  if (!payload || !Array.isArray(payload.timeline)) {
    els.replayHeader.textContent = "Invalid replay file: no timeline.";
    return;
  }
  const t = payload.timeline;
  const finalVerdict = payload.final_verdict ?? {};
  const live = finalVerdict.is_live ? "LIVE" : "SPOOF";
  const conf = Math.round((finalVerdict.confidence ?? 0) * 100);
  const dur = payload.recording_seconds ?? 0;
  const frames = payload.frame_count ?? t.length;
  els.replayHeader.innerHTML =
    `<div><b style="color:${
      finalVerdict.is_live ? "var(--green)" : "var(--red)"
    }">${live}</b> · final conf ${conf}% · ${dur.toFixed
      ? dur.toFixed(1)
      : dur}s · ${frames} frames · recorded ` +
    `${payload.generated_at ?? "(unknown time)"}</div>` +
    `<div style="color: var(--muted); font-size: 11px; margin-top: 4px">` +
    `amispoof version: ${payload.amispoof_version ?? "(unknown)"} · ` +
    `user agent: ${
      (payload.user_agent || "").slice(0, 60)
    }${(payload.user_agent || "").length > 60 ? "…" : ""}` +
    `</div>`;

  // Tiny SVG chart: x = frame index, y = proof_total (0..100), plus
  // a green/red dotted background indicating is_live=true/false.
  const svg = els.replayChart;
  const W = svg.clientWidth || 600;
  const H = 120;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = "";

  // Background bands per-frame: green tint when is_live, red when SPOOF.
  const bandStrip = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const stepX = W / Math.max(1, t.length);
  for (let i = 0; i < t.length; i++) {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", String(i * stepX));
    r.setAttribute("y", "0");
    r.setAttribute("width", String(stepX + 1));
    r.setAttribute("height", String(H));
    r.setAttribute(
      "fill",
      t[i].is_live ? "rgba(63,185,80,0.08)" : "rgba(248,81,73,0.18)",
    );
    bandStrip.appendChild(r);
  }
  svg.appendChild(bandStrip);

  // Proof_total line.
  let pathD = "";
  for (let i = 0; i < t.length; i++) {
    const pct = t[i].proof_total ?? 0;
    const x = i * stepX;
    const y = H - (pct / 100) * (H - 10) - 5;
    pathD += i === 0 ? `M${x},${y}` : ` L${x},${y}`;
  }
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", pathD);
  p.setAttribute("stroke", "var(--accent)");
  p.setAttribute("stroke-width", "2");
  p.setAttribute("fill", "none");
  svg.appendChild(p);

  // 60-point proven-live threshold dashed line.
  const thr = document.createElementNS("http://www.w3.org/2000/svg", "line");
  thr.setAttribute("x1", "0");
  thr.setAttribute("x2", String(W));
  thr.setAttribute("y1", String(H - (60 / 100) * (H - 10) - 5));
  thr.setAttribute("y2", String(H - (60 / 100) * (H - 10) - 5));
  thr.setAttribute("stroke", "var(--muted)");
  thr.setAttribute("stroke-dasharray", "4 4");
  thr.setAttribute("stroke-width", "1");
  svg.appendChild(thr);

  // Aggregate stats for the legend.
  const proofs = t.map((x) => x.proof_total ?? 0);
  const minProof = Math.min(...proofs);
  const maxProof = Math.max(...proofs);
  const liveFraction = t.filter((x) => x.is_live).length / Math.max(1, t.length);
  els.replayLegend.textContent =
    `proof_total: min ${minProof} · max ${maxProof} · ` +
    `verdict was LIVE for ${(liveFraction * 100).toFixed(0)}% of frames · ` +
    `green bands = LIVE, red = SPOOF, dashed line = 60-pt proven-live threshold`;
}

if (els.replayBtn && els.replayFile) {
  els.replayBtn.addEventListener("click", () => els.replayFile.click());

  // Android Chrome 148 (and some iOS Safari builds) can revoke the underlying
  // file handle the moment the picker dismisses — before ANY async read has
  // a chance to land its bytes. Mitigation has three layers:
  //   1. snapshotFile() races Blob.arrayBuffer() AND FileReader in parallel,
  //      both kicked off synchronously inside the change handler. Whichever
  //      resolves first wins; we only fail if BOTH paths reject.
  //   2. Drag-and-drop onto the replay panel uses DataTransfer.files, which
  //      on most platforms holds a stronger handle than the picker FileList.
  //   3. On hard failure, the user is offered a clipboard-paste fallback —
  //      they can paste the JSON text directly with no file-handle hop.
  els.replayFile.addEventListener("change", () => {
    const file = els.replayFile.files?.[0];
    if (!file) return;
    handleReplayPick(file);
  });

  if (els.replayPanel) {
    const setDropStyle = (active) => {
      els.replayPanel.style.outline = active ? "2px dashed #58a6ff" : "";
      els.replayPanel.style.outlineOffset = active ? "4px" : "";
    };
    els.replayPanel.addEventListener("dragover", (e) => {
      e.preventDefault();
      setDropStyle(true);
    });
    els.replayPanel.addEventListener("dragleave", () => setDropStyle(false));
    els.replayPanel.addEventListener("drop", (e) => {
      e.preventDefault();
      setDropStyle(false);
      const file = e.dataTransfer?.files?.[0];
      if (file) handleReplayPick(file);
    });
  }

  function handleReplayPick(file) {
    // Kick off BOTH readers synchronously inside the gesture/change event,
    // BEFORE any UI work or await — minimises the window in which the OS
    // can drop the handle.
    const snapshot = snapshotFile(file);
    const filename = file.name || "recording.json";
    const filesize = file.size || 0;
    finishReplayLoad(snapshot, filename, filesize);
  }

  function snapshotFile(file) {
    // Both reads invoked synchronously in this stack frame.
    const fromBlob = (
      typeof file.arrayBuffer === "function"
        ? file.arrayBuffer()
        : Promise.reject(new Error("Blob.arrayBuffer unsupported"))
    ).then(
      (buf) => ({ ok: true, buf }),
      (err) => ({ ok: false, err }),
    );
    const fromReader = new Promise((resolve) => {
      let settled = false;
      const settle = (v) => {
        if (settled) return;
        settled = true;
        resolve(v);
      };
      const reader = new FileReader();
      reader.onload = () => settle({ ok: true, buf: reader.result });
      reader.onerror = () =>
        settle({ ok: false, err: reader.error ?? new Error("FileReader failed") });
      reader.onabort = () => settle({ ok: false, err: new Error("FileReader aborted") });
      try {
        reader.readAsArrayBuffer(file);
      } catch (e) {
        settle({ ok: false, err: e });
      }
    });
    return Promise.all([fromBlob, fromReader]).then(([a, b]) => {
      if (a.ok) return a.buf;
      if (b.ok) return b.buf;
      const err = (b.err && b.err.message) ? b.err : a.err;
      throw err ?? new Error("Both file readers failed");
    });
  }

  async function finishReplayLoad(snapshot, filename, filesize) {
    els.replayPanel.style.display = "";
    els.replayHeader.textContent =
      `Loading ${filename} (${(filesize / 1024).toFixed(1)} KB)…`;
    els.replayChart.innerHTML = "";
    els.replayLegend.textContent = "";
    try {
      const buf = await snapshot;
      const text = new TextDecoder("utf-8").decode(buf);
      const payload = JSON.parse(text);
      renderReplay(payload);
    } catch (err) {
      console.error("replay parse failed", err);
      const msg = err?.message || String(err);
      const isNotFound =
        msg.includes("could not be found") ||
        err?.name === "NotFoundError" ||
        err?.code === 1 ||
        err?.code === 11;
      els.replayHeader.innerHTML = isNotFound
        ? `Replay load failed: the OS released the file handle before we could read it (common on Android Chrome). ` +
          `<strong>Fallback:</strong> drag the JSON file onto this panel, or ` +
          `<button id="replayPasteBtn" style="background:#238636;color:#fff;border:1px solid #2ea043;border-radius:4px;padding:2px 10px;font-size:12px;cursor:pointer;">paste JSON from clipboard</button>.`
        : `Replay parse failed: ${escapeReplayMsg(msg)}`;
      const pasteBtn = document.getElementById("replayPasteBtn");
      if (pasteBtn) {
        pasteBtn.addEventListener("click", pasteReplayFromClipboard);
      }
    } finally {
      els.replayFile.value = "";
    }
  }

  async function pasteReplayFromClipboard() {
    try {
      if (!navigator.clipboard || !navigator.clipboard.readText) {
        throw new Error(
          "Clipboard API unavailable in this browser — try drag-and-drop instead.",
        );
      }
      const text = await navigator.clipboard.readText();
      if (!text || !text.trim()) {
        els.replayHeader.textContent = "Clipboard is empty.";
        return;
      }
      const payload = JSON.parse(text);
      els.replayChart.innerHTML = "";
      els.replayLegend.textContent = "";
      renderReplay(payload);
    } catch (e) {
      console.error("clipboard replay parse failed", e);
      els.replayHeader.textContent = `Paste failed: ${e?.message || e}`;
    }
  }

  function escapeReplayMsg(s) {
    return String(s).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        }[c]),
    );
  }
}
if (els.replayClose) {
  els.replayClose.addEventListener("click", () => {
    els.replayPanel.style.display = "none";
  });
}

els.copyVerdict.addEventListener("click", async () => {
  const v = lastVerdict;
  if (!v) return;
  const proof = lastProof;
  const proofLines = proof
    ? [
        "",
        `Liveness proof: ${Math.round(proof.score)} / 100 (${proof.is_proven_live ? "proven live" : "not proven"})${SHOW_ACTIVE_SECTION ? "" : " [proctoring: passive-only]"}`,
        ...PROOF_AXES.filter(
          (axis) => SHOW_ACTIVE_SECTION || axis.section === "passive",
        ).map((axis) => {
          const val = proof.details?.[axis.name] ?? 0;
          return `  ${axis.label.padEnd(16)} ${String(val.toFixed(0)).padStart(3)} / ${axis.max}`;
        }),
        `  head motion        yaw ${(proof.yaw_range_seen_deg ?? 0).toFixed(1)}° · pitch ${(proof.pitch_range_seen_deg ?? 0).toFixed(1)}°`,
        ...(SHOW_ACTIVE_SECTION
          ? [
              `  challenges         ${proof.details?.challenges_passed ?? 0} passed · ${proof.details?.challenges_failed ?? 0} failed`,
            ]
          : []),
      ]
    : [];
  const text = [
    displaySummary(v),
    "",
    "Per-analyzer scores:",
    ...ANALYZER_ORDER.map((cfg) => {
      const s = lastAnalyzerScores?.[cfg.name];
      return s
        ? `  ${cfg.label.padEnd(20)} ${String(s.score).padStart(5)} (w ${cfg.weight})`
        : `  ${cfg.label.padEnd(20)} —`;
    }),
    ...proofLines,
    "",
    `Gate: ${lastGateResult ? (lastGateResult.usable ? "usable" : `advisory: ${lastGateResult.reason}`) : "—"}`,
    "",
    `Generated: ${new Date().toISOString()}`,
    `Agent: ${navigator.userAgent}`,
  ].join("\n");
  try {
    await navigator.clipboard.writeText(text);
    const original = els.copyVerdict.textContent;
    els.copyVerdict.textContent = "copied ✓";
    setTimeout(() => {
      els.copyVerdict.textContent = original;
    }, 1400);
  } catch (err) {
    console.error("clipboard write failed", err);
    els.copyVerdict.textContent = "copy failed";
  }
});
