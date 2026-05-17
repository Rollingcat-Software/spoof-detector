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
} from "./lib/spoof-detector.js?v=2026-05-17-phaseD3";

// Version handshake — checked by the inline script in index.html.
// If the user is running a stale cached app.js (no AMISPOOF_VERSION),
// the HTML triggers a one-shot reload after 4 s.
window.AMISPOOF_VERSION = "2026-05-17-phaseD3";

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
  const verdictWord = v.is_live ? "LIVE" : "SPOOF";
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

const $ = (id) => document.getElementById(id);

const els = {
  videoWrap: $("videoWrap"),
  video: $("video"),
  overlay: $("overlay"),
  start: $("start"),
  stop: $("stop"),
  reset: $("reset"),
  download: $("download"),
  bench: $("bench"),
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

    els.start.style.display = "none";
    els.stop.disabled = false;
    els.reset.disabled = false;
    els.download.disabled = false;
    els.videoWrap.dataset.state = "running";
    setStatus("running", "live");

    running = true;
    loop();
  } catch (err) {
    console.error(err);
    setStatus(`error: ${err.message || err}`, "error");
    els.start.disabled = false;
    els.start.textContent = "Start";
  }
}

function stop() {
  running = false;
  const stream = els.video.srcObject;
  if (stream && typeof stream.getTracks === "function") {
    for (const track of stream.getTracks()) track.stop();
  }
  els.video.srcObject = null;
  els.stop.disabled = true;
  els.start.style.display = "";
  els.start.disabled = false;
  els.start.textContent = "Start";
  els.videoWrap.dataset.state = "idle";
  setStatus("stopped", "live");
}

async function loop() {
  if (!running) return;
  try {
    ctx.drawImage(els.video, 0, 0, canvas.width, canvas.height);
    const analysis = await detector.analyzeFrame(canvas);
    const v = detector.getVerdict();
    lastVerdict = v;
    drawOverlay(analysis, v);
    updateUI(analysis, v);
  } catch (err) {
    console.error("frame error", err);
    setStatus(`frame error: ${err.message || err}`, "error");
  }
  if (running) requestAnimationFrame(loop);
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
  const color = warming
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
  els.verdict.classList.toggle("live", !warming && v.is_live);
  els.verdict.classList.toggle("spoof", !warming && !v.is_live);
  els.verdict.classList.toggle("warming", warming);

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
  els.incidentList.innerHTML =
    '<div class="incident">No incidents yet.</div>';
  renderProofPanel(null);
  setStatus("running", "live");
}

function download() {
  const verdict = lastVerdict ?? detector?.getVerdict() ?? null;
  if (!verdict) return;
  const blob = new Blob(
    [
      JSON.stringify(
        {
          generated_at: new Date().toISOString(),
          user_agent: navigator.userAgent,
          verdict,
          latest_analyzer_scores: lastAnalyzerScores,
          latest_gate_result: lastGateResult,
          latest_liveness_proof: lastProof,
          fps_smoothed: Math.round(smoothedFps * 10) / 10,
        },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `amispoof-session-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
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

// Phase D3 — wire the microphone toggle. The SDK requires audio to be
// enabled at construction time (see createSpoofDetector opts). When the
// page wasn't started with ?audio=1, clicking the button reloads with
// it set so the SDK re-initialises with audio. When it WAS, the click
// just calls detector.startAudio() to prompt for permission.
if (els.micToggle) {
  const audioPreEnabled =
    new URLSearchParams(window.location.search).get("audio") === "1";
  if (!audioPreEnabled) {
    els.micToggle.textContent = "🎤 Reload with mic";
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
          els.micToggle.textContent = "🎤 Mic on";
        }
      } catch (e) {
        console.error("mic toggle failed", e);
        els.micToggle.textContent = "🎤 Mic failed";
      }
    });
  }
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
