// amispoof — browser anti-spoof tester driver.
//
// Loads @rollingcat/spoof-detector from ./lib/, the ONNX + FaceLandmarker
// models from ./models/, and the onnxruntime-web / mediapipe-tasks-vision
// peer deps from jsdelivr via the importmap declared in index.html.

import * as ort from "onnxruntime-web";
import { createSpoofDetector } from "./lib/spoof-detector.js";

const ORT_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
const MEDIAPIPE_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

// Point ORT WASM resolver at jsdelivr so it can find the .wasm sidecar files.
// (onnxruntime-web defaults to "relative to the JS bundle", which would 404
// when our bundle is served from /amispoof/lib/.)
ort.env.wasm.wasmPaths = ORT_WASM_BASE;
ort.env.wasm.numThreads = 2;

const ANALYZER_ORDER = [
  { name: "minifasnet", weight: 5.0, label: "MiniFASNet" },
  { name: "screen_flicker", weight: 3.0, label: "Screen flicker" },
  { name: "device_boundary", weight: 2.5, label: "Device boundary" },
  { name: "micro_tremor", weight: 2.5, label: "Micro-tremor" },
  { name: "landmark_variance", weight: 2.0, label: "Landmark variance" },
  { name: "blink", weight: 0.5, label: "Blink (EAR)" },
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
};

const analyzerRefs = {};
for (const cfg of ANALYZER_ORDER) {
  const row = document.createElement("div");
  row.className = "analyzer-row";
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
let knownIncidentIds = new Set();

function setStatus(label, kind = "live") {
  els.status.textContent = label;
  els.dot.className = `status-dot ${kind === "error" ? "error" : "live"}`;
}

async function ensureDetector() {
  if (detector) return detector;
  setStatus("loading models…");
  els.start.disabled = true;
  els.start.textContent = "loading models…";

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
    els.start.textContent = "Start camera";
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
  els.start.textContent = "Start camera";
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
    updateUI(analysis, v);
  } catch (err) {
    console.error("frame error", err);
    setStatus(`frame error: ${err.message || err}`, "error");
  }
  if (running) requestAnimationFrame(loop);
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
  for (const cfg of ANALYZER_ORDER) {
    const r = analyzerResults[cfg.name];
    const ref = analyzerRefs[cfg.name];
    if (!r) {
      ref.fill.style.width = "0%";
      ref.val.textContent = "—";
      continue;
    }
    const score = Math.max(0, Math.min(100, r.score));
    ref.fill.style.width = `${score.toFixed(0)}%`;
    ref.val.textContent = score.toFixed(0);
  }

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
