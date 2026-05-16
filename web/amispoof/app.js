// amispoof — browser anti-spoof tester driver.
//
// Loads @rollingcat/spoof-detector from ./lib/, the ONNX + FaceLandmarker
// models from ./models/, and the onnxruntime-web / mediapipe-tasks-vision
// peer deps from jsdelivr via the importmap declared in index.html.

import * as ort from "onnxruntime-web";
import { createSpoofDetector } from "./lib/spoof-detector.js";

// Version handshake — checked by the inline script in index.html.
// If the user is running a stale cached app.js (no AMISPOOF_VERSION),
// the HTML triggers a one-shot reload after 4 s.
window.AMISPOOF_VERSION = "2026-05-16e";

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
      fill: row.querySelector(".fill"),
      val: row.querySelector(".val"),
    };
  }
}
buildAnalyzerGroup("Image-track (single frame)", "image");
buildAnalyzerGroup("Video-track (over time)", "video");

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
let knownIncidentIds = new Set();

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

  els.verdictText.textContent = v.summary;
  els.verdictConf.textContent = `${(v.confidence * 100).toFixed(0)}% conf`;
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
    if (!r) {
      ref.fill.style.width = "0%";
      ref.val.textContent = "—";
      snapshot[cfg.name] = null;
      continue;
    }
    const score = Math.max(0, Math.min(100, r.score));
    ref.fill.style.width = `${score.toFixed(0)}%`;
    ref.val.textContent = score.toFixed(0);
    snapshot[cfg.name] = {
      score: Math.round(score * 10) / 10,
      details: r.details ?? null,
    };
  }
  lastAnalyzerScores = snapshot;

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
    const headlineText = gate.usable
      ? "Face usable"
      : `Advisory: ${gate.reason.replace(/_/g, " ")}`;
    const headlineClass = gate.usable ? "ok" : "warn";
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
      els.gateBanner.classList.toggle("warn", !gate.usable);
    }
  }
}

function reset() {
  if (!detector) return;
  detector.reset();
  smoothedFps = 0;
  lastTs = 0;
  knownIncidentIds = new Set();
  els.incidentList.innerHTML =
    '<div class="incident">No incidents yet.</div>';
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

els.copyVerdict.addEventListener("click", async () => {
  const v = lastVerdict;
  if (!v) return;
  const text = [
    v.summary,
    "",
    "Per-analyzer scores:",
    ...ANALYZER_ORDER.map((cfg) => {
      const s = lastAnalyzerScores?.[cfg.name];
      return s
        ? `  ${cfg.label.padEnd(20)} ${String(s.score).padStart(5)} (w ${cfg.weight})`
        : `  ${cfg.label.padEnd(20)} —`;
    }),
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
