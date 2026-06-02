# Foundation-model FAS head — training + browser integration

This is the **accuracy lever** identified in the 2026-06-02 analysis: replace the
brittle hand-tuned analyzer bank (which doesn't transfer across subjects/cameras —
paper §8.2) with a **vision-foundation backbone (DINOv2 / CLIP) + a small trained
head**, the 2025-26 SOTA approach to face-anti-spoofing domain generalization.
It runs on the client GPU via **WebGPU** — the analyzer that finally justifies the
GPU work, since the 1.7 MB MiniFASNet is too small for WebGPU to help.

## Pieces

| Piece | Path | Status |
|---|---|---|
| Trainer + ONNX exporter (Python) | `tools/train_fas_adapter.py` | ✅ pipeline verified (`--smoke`) |
| Browser analyzer (WebGPU EP) | `web/src/infrastructure/analyzers/FoundationModelAnalyzer.ts` | ✅ 267 web tests green |
| Shared I/O contract | input `pixel_values`[1,3,224,224] RGB ImageNet-norm → output `logits`[1,2] (0=spoof,1=real) | locked by the smoke round-trip |

## Data reality (read first)

The 86 captures in `notebooks/data/` are **analyzer-score telemetry, not pixels**
(per-frame `analyzer_scores` + `face_bbox`), so they **cannot train a vision
model** — they only validate the *session logic*. Training images come from:

1. **Public FAS datasets** (CASIA-FASD / CelebA-Spoof via HuggingFace — the ones
   the paper already cites). Fastest path to a first model; generalizes far better
   than the YOLOv8 dead-end (`l_version_1_300`) because DINOv2 features transfer.
2. **Your own face crops, captured going forward.** The only way to fit *your*
   cameras + multiple subjects. Requires adding a local crop-save to amispoof
   (see "Next steps") — keep crops local + gitignored; never commit face images.

## Run it

```bash
# 1. Plumbing check — no downloads, ~5 s. Proves train→export→ORT-load works.
python tools/train_fas_adapter.py --smoke --out models/fas_head_smoke.onnx

# 2. Real training. Downloads DINOv2 (~85 MB, torch.hub) + the dataset.
#    Backbone is FROZEN so this fits the GTX 1650 (4 GB) — only the head trains.
pip install datasets pillow torchvision        # one-time, for real datasets
python tools/train_fas_adapter.py \
    --backbone dinov2_vits14 \
    --dataset hf:nguyenkhoa/CASIA-FASD \
    --epochs 10 --batch-size 32 \
    --out models/fas_head_dinov2.onnx --fp16

# If AUC comes out < 0.5, the dataset uses 1=spoof — add --flip-labels.
```

Heavy/optional deps, NOT auto-installed (you're on metered data sometimes):
`onnx` (export, ~17 MB — already installed), `datasets/pillow/torchvision`
(real data), `onnxconverter-common` (`--fp16`), `open_clip_torch` (CLIP backbone),
DINOv2 weights (~85 MB on first `--backbone dinov2_*` run).

## Wire it into amispoof

1. Copy the exported ONNX to `web/amispoof/models/fas_head.onnx`.
2. Instantiate the analyzer (it defaults to WebGPU-first, WASM fallback):

   ```ts
   import { FoundationModelAnalyzer } from "@rollingcat/spoof-detector";
   const fas = new FoundationModelAnalyzer({ modelUrl: "./models/fas_head.onnx" });
   await fas.warmup();
   fas.setFrame(frame);                 // full frame for context-padded crop
   const r = await fas.analyze(null, faceROI);  // r.score 0-100, higher = live
   ```

3. It is intentionally **NOT** in `DEFAULT_ANALYZER_WEIGHTS` yet. Only add it to
   the fuser after validating it on **multi-subject** data (GroupKFold / leave-one-
   subject-out), so a new, unproven model can't decalibrate the shipped detector.

## GTX 1650 notes

- Frozen backbone → no backbone gradients → tiny VRAM; works on CPU too.
- For speed, precompute features once if you iterate on the head a lot.
- fp16 ONNX (`--fp16`) ~halves the model and is a good fit for the WebGPU EP.

## Next steps (in priority order)

1. **Multi-subject face-crop capture** in amispoof (local, gitignored) → the only
   path to a model that fits your cameras + the 3-5 subjects the paper needs.
2. Train on CASIA-FASD/CelebA-Spoof for a zero-shot baseline; measure AUC.
3. Validate with GroupKFold, then (only then) add to the fuser with a small weight.
