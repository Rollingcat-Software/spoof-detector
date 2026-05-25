# amispoof → Proctoring Pipeline — Design Doc (DRAFT for review)

**Status:** proposal, no code yet. **Author:** Claude + Ahmet. **Date:** 2026-05-25.
**Decision sought:** sign-off on the step-by-step pipeline + the client-side
identity-matching layer before implementation. Everything below is a
recommendation; flag anything to change.

---

## 1. Why change anything

Today amispoof is a **debug dashboard**: it dumps every analyzer, gate, and
proof axis on screen at once. Great for development; wrong for proctoring and
for a first-time visitor. Three concrete findings from this session drive the
redesign:

1. **No quality precondition.** In a pitch-dark room the engine reported a
   confident `SPOOF (static_image)` with **0 faces detected** and a **1,326-incident
   "face missing" flood** — it should have refused to start and said "fix your
   lighting." (Conversely a bright room over-lights the face and the active
   flash probe abstains.) The usable capture window is **dim-to-moderate**, and
   nothing enforces it.
2. **Identity is unverified.** Every liveness/anti-spoof signal passes a *real
   live face*. A **person-swap mid-exam is a real live face** → undetected. PAD
   and identity are **orthogonal axes**; proctoring needs both.
3. **It's a dump, not a flow.** The dashboard mashes "establish liveness" and
   "monitor" into one screen with no guidance, no hierarchy, no identity/brand.

## 2. The two axes (keep these distinct)

| Axis | Question | Catches | Status |
|---|---|---|---|
| **PAD / anti-spoof** | real live face vs. fake? | print, screen/replay, mask, deepfake | built (this repo) |
| **Identity** | the *same enrolled* person? | impersonation / person-swap | **new (this doc)** |

A **mask** trips both. A **person-swap** trips only identity. A **photo of the
enrolled person** trips only PAD. You need both layers for exam integrity.

## 3. Goals / non-goals (this iteration)

**Goals:** PC-first (Brave/Chrome desktop); fully client-side (no upload — keep
amispoof's "no server" promise); guided step flow; readiness gate that **blocks
Start**; continuous identity + liveness monitoring; downloadable evidence report.

**Non-goals (now):** mobile (bonus later); server round-trips; replacing the
platform's server-side Facenet512/pgvector recognition; high-assurance 1:1
identity (that stays server-side — see §6 limits).

## 4. The pipeline (step-by-step flow)

```
[0 Device] → [1 Readiness] → [2 Enroll] → [3 Liveness] → [4 Monitor] → [5 Verdict]
              (blocks Begin)   (consent)    (establish)    (continuous)   (report)
```

| Step | Purpose | Pass condition | Reuses | New |
|---|---|---|---|---|
| **0. Device & camera** | camera works, single device, resolution OK | stream live, `videoWidth ≥ 480`, frames advancing | getUserMedia plumbing | small check |
| **1. Readiness gate** | enforce usable capture before anything else | face present (exactly 1) · centered · large enough · illumination in `[floor, ceiling]` · not occluded · camera responsive | `FaceUsabilityGate` (occlusion/illumination), face bbox | gate logic + **blocking** UI + incident-dedup fix |
| **2. Identity enrollment** | bind the session to a person (consented) | N=3–5 good frames captured → reference template | face bbox + landmark alignment | **`IdentityMatcher`** (embedding model) |
| **3. Liveness establishment** | confirm real live face once | passive proof reaches threshold; optional flash challenge in dim-moderate light | SessionEngine, LivenessProver, MultiClassFuser, FlashTemporal/ReflectionAnalyzer | wiring into a step |
| **4. Continuous monitoring** | watch the exam | per-interval: identity match + liveness + occlusion + exactly-1-face | all of the above | identity-drift + multi-face/no-face incidents |
| **5. Verdict & evidence** | final, explainable result | — | SessionEngine verdict + incident ledger | timeline + downloadable JSON report |

Two UI modes kept: **"Proctoring pipeline"** (the guided flow above, default) and
**"Tester dashboard"** (today's debug view, kept for research/dev — toggle).

## 5. Identity-matching layer (the new piece)

**Where it sits:** new module `web/src/identity/IdentityMatcher.ts`, orchestrated
by the pipeline — *not* a fusion analyzer (it answers a different question).

**Model choice (decision needed — recommendation marked):**

| Option | Size | Notes |
|---|---|---|
| **✅ ONNX MobileFaceNet / SFace via onnxruntime-web** | ~4–5 MB | Reuses the runtime already loaded for MiniFASNet — one stack, one EP (WASM/WebGPU). 512-d or 128-d embedding. **Recommended.** |
| face-api.js (FaceRecognitionNet) | ~6 MB | 128-d, batteries-included, but a second ML stack + older tfjs. |
| MediaPipe FaceLandmarker geometry | 0 (already loaded) | **Not identity** — it's landmarks/blendshapes, not a recognition embedding. Insufficient alone. |

**Enrollment (step 2):** gated on readiness; capture 3–5 frames across small
pose variation; align each face crop with FaceLandmarker's 5-point landmarks;
embed; average → L2-normalized **reference template**. Store **in-memory by
default**; optional `localStorage` (explicit opt-in) for "remember me."

**Continuous match (step 4):** every ~0.5–1 s, crop the tracked face, embed,
**cosine similarity** vs. reference. `sim ≥ τ` → same person. A **sustained**
drop below `τ` (e.g. > N consecutive checks) → `IMPERSONATION` incident +
verdict, reported **separately** from SPOOF. Hysteresis + the readiness gate's
quality guarantee prevent flicker false-rejects from a head-turn.

**Privacy / consent (non-negotiable):**
- Enrollment is explicit & consented; a clear "your face is processed on this
  device and never uploaded" notice (true — no server call).
- Template clearable; ephemeral by default.
- Biometric data → KVKK/GDPR applies; mirror the platform's existing My-Profile
  consent/erase model. Never persist face *images*, only the embedding vector.

## 6. Readiness gate spec (step 1 detail)

Promote `FaceUsabilityGate` from **advisory** → **blocking precondition**.

| Check | Signal | Default | Fail message |
|---|---|---|---|
| Face present | bbox count == 1 | not 0, not >1 | "No face / multiple faces — one person, please" |
| Face size | bbox area / frame | ≥ ~12% | "Move closer" |
| Illumination | gate illumination score / mean luma | in `[~0.35, ~0.72]` (dim-moderate band; reuse the over-lit ceiling 185/255) | too low → "Add light" · too high → "Reduce/redirect light" |
| Not occluded | gate occlusion score + flagged regions | occlusion < threshold | "Uncover your eyes/nose/mouth" |
| Camera responsive | frames advancing, fps > min | live | "Camera not responding" |

"**Begin**" is disabled until **all green**, with live per-check status + fix-it
text. **Also fixes** the per-frame "face missing" incident flood: collapse to
**one incident per missing-episode** (edge-triggered, not level-triggered).

## 7. Architecture & reuse

- **Unchanged engine:** the `@rollingcat/spoof-detector` SDK stays the
  liveness/PAD core (analyzers, fusion, SessionEngine, LivenessProver,
  FlashTemporal/Reflection, FaceUsabilityGate).
- **New (browser app layer, `web/`):**
  - `identity/IdentityMatcher.ts` — embed + enroll + compare.
  - `pipeline/PipelineController.ts` — step state machine + transitions.
  - readiness-gate evaluator (thin, over existing gate signals).
  - the step-wizard UI (replaces the single amispoof dashboard layout).
- The Python `src/` reference is untouched (paper/benchmark surface).

## 8. UX / visual redesign (folded into the flow)

The pipeline *is* the redesign — it fixes the "boring dashboard" critique:
- **One focused task per step** + a progress indicator (not a wall of rows).
- **Verdict as hero** at steps 3 & 5: large state, animated confidence
  gauge, color wash + motion on LIVE/SPOOF/IMPERSONATION transitions.
- **Progressive disclosure:** an "Advanced / Why?" panel reveals the analyzer
  bank (keep the transparency that makes it credible — just demote it).
- **Identity/brand:** an `amispoof` wordmark, a type pairing (display + mono),
  a signature accent that isn't GitHub-blue, a subtle "CV scan" treatment on
  the video, a real loading/skeleton state (kills the "loading…" confusion).

## 9. Incremental build plan (one PR each, tests + verify per PR)

- **PR-A — Readiness gate + incident-dedup fix.** Highest immediate value; lands
  on the current page first (no full redesign needed to ship the safety win).
- **PR-B — Pipeline shell.** Step state machine + wizard UI; re-house today's
  dashboard as the "Tester" mode. (This is the UI redesign, done as the flow.)
- **PR-C — IdentityMatcher.** Embedding model + enroll (step 2) + continuous
  match (step 4) + `IMPERSONATION` verdict, with unit tests on the
  embed/compare math (synthetic vectors) like the analyzer tests.
- **PR-D — Monitoring + evidence + polish.** Continuous-monitoring incidents,
  evidence report, and the visual/identity/motion pass.

## 10. Honest limits / risks

- **Client-side recognition < server Facenet512.** Good for *continuity*
  (same-person drift within a session); not a high-assurance 1:1 identity
  proof — that stays server-side. We'll frame it as "session identity
  continuity," not "identity verification."
- **Identity match degrades with poor capture** — which is exactly why it sits
  *behind* the readiness gate.
- **Lighting tension is real:** passive detectors want light; the active flash
  probe wants dim. The gate enforces the dim-to-moderate overlap; the flash
  challenge remains optional and abstains outside its band.
- **PC-first.** Mobile front cameras lack manual-exposure control (flash
  abstains) and add model-download cost (identity) — deferred.
- **Model download size** (~4–5 MB identity model) on first load — lazy-load it
  only when enrollment starts.

## 11. Open decisions for you

1. **Identity model:** ONNX MobileFaceNet/SFace (recommended) vs face-api.js?
2. **Template persistence:** in-session only / `localStorage` opt-in / tie to
   the platform's server enrollment?
3. **Match strictness** `τ` + consecutive-fail count (false-reject vs. miss
   trade-off) — calibrate live like we did the flash probe.
4. **Keep the Tester dashboard** as a dev mode? (recommend: yes.)
5. **Monitoring scope:** which incidents in step 4 — identity drift, 2nd person,
   no face, look-away, phone-in-frame (hand/object detector)?
