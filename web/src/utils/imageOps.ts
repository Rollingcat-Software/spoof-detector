// Small image utility helpers used by MiniFASNetAnalyzer.
// Mirrors the cropping math in uniface/spoofing/minifasnet.py:_crop_face
// (see /home/deploy/.local/lib/python3.12/site-packages/uniface/spoofing/minifasnet.py)
// and the bbox-padded fallback in src/infrastructure/analyzers/minifasnet_analyzer.py.

/** Image source we accept from callers. */
export type SourceImage = HTMLCanvasElement | OffscreenCanvas | ImageData;

/** Duck-type ImageData. `instanceof ImageData` blows up in node, where the
 *  global doesn't exist; canvas-likes always expose getContext(), ImageData
 *  never does, so this is unambiguous. */
function isImageDataLike(src: SourceImage): src is ImageData {
  return typeof (src as { getContext?: unknown }).getContext !== "function";
}

/** Get an ImageData snapshot from any supported input, no scaling. */
export function toImageData(src: SourceImage): ImageData {
  if (isImageDataLike(src)) return src;
  const ctx = (
    src as HTMLCanvasElement | OffscreenCanvas
  ).getContext("2d") as CanvasRenderingContext2D | null;
  if (!ctx) {
    throw new Error("toImageData: 2D context unavailable on source canvas");
  }
  return ctx.getImageData(0, 0, src.width, src.height);
}

/**
 * Compute the scale-padded crop rect around a face bbox, mirroring
 * MiniFASNet's `_crop_face`:
 *
 *   scale = min((H-1)/box_h, (W-1)/box_w, self.scale)
 *   new_w = box_w * scale; new_h = box_h * scale
 *   x1 = max(0, cx - new_w/2); y1 = max(0, cy - new_h/2)
 *   x2 = min(W-1, cx + new_w/2); y2 = min(H-1, cy + new_h/2)
 *
 * The `scale` argument matches uniface DEFAULT_SCALES (V2 = 2.7, V1SE = 4.0).
 */
export function computeMiniFasNetCropRect(
  imageWidth: number,
  imageHeight: number,
  bbox: { x1: number; y1: number; x2: number; y2: number },
  scale: number,
): { x: number; y: number; w: number; h: number } {
  const x = Math.floor(bbox.x1);
  const y = Math.floor(bbox.y1);
  const box_w = Math.max(1, Math.floor(bbox.x2 - bbox.x1));
  const box_h = Math.max(1, Math.floor(bbox.y2 - bbox.y1));

  const eff_scale = Math.min(
    (imageHeight - 1) / box_h,
    (imageWidth - 1) / box_w,
    scale,
  );
  const new_w = box_w * eff_scale;
  const new_h = box_h * eff_scale;
  const cx = x + box_w / 2;
  const cy = y + box_h / 2;

  const x1 = Math.max(0, Math.floor(cx - new_w / 2));
  const y1 = Math.max(0, Math.floor(cy - new_h / 2));
  const x2 = Math.min(imageWidth - 1, Math.floor(cx + new_w / 2));
  const y2 = Math.min(imageHeight - 1, Math.floor(cy + new_h / 2));

  return { x: x1, y: y1, w: x2 - x1 + 1, h: y2 - y1 + 1 };
}

/**
 * Crop a sub-rectangle from a SourceImage and resize it to (outW, outH).
 * Returns a fresh OffscreenCanvas containing the resized BGR-equivalent
 * pixels (in browser they're RGBA — the caller does channel swizzling).
 */
export function cropAndResize(
  src: SourceImage,
  rect: { x: number; y: number; w: number; h: number },
  outW: number,
  outH: number,
): OffscreenCanvas {
  // Stage 1: ensure source is a Canvas-backed image so drawImage() can resize.
  let stage: HTMLCanvasElement | OffscreenCanvas;
  if (isImageDataLike(src)) {
    const c = new OffscreenCanvas(src.width, src.height);
    const cctx = c.getContext("2d");
    if (!cctx) throw new Error("cropAndResize: stage canvas has no 2D ctx");
    cctx.putImageData(src, 0, 0);
    stage = c;
  } else {
    stage = src;
  }

  const out = new OffscreenCanvas(outW, outH);
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("cropAndResize: output canvas has no 2D ctx");
  // smoothing≈INTER_AREA-ish for downscale; INTER_LINEAR-ish for upscale.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(stage, rect.x, rect.y, rect.w, rect.h, 0, 0, outW, outH);
  return out;
}

/**
 * Convert a (W, H) RGBA canvas to a planar BGR Float32 NCHW tensor.
 * Matches uniface's preprocess: `face = face.astype(float32)` + HWC→CHW + expand_dims.
 * The MiniFASNet ONNX expects BGR (because the Python reads via cv2.imread).
 */
export function toBgrNchwFloat32(canvas: OffscreenCanvas): Float32Array {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("toBgrNchwFloat32: 2D ctx unavailable");
  const w = canvas.width;
  const h = canvas.height;
  const rgba = ctx.getImageData(0, 0, w, h).data; // Uint8ClampedArray length 4*w*h

  const out = new Float32Array(3 * h * w);
  const planeSize = h * w;
  // Channel order: B, G, R (NCHW with C=3)
  // out[0*plane + i] = B
  // out[1*plane + i] = G
  // out[2*plane + i] = R
  for (let i = 0, p = 0; i < planeSize; i++, p += 4) {
    const r = rgba[p];
    const g = rgba[p + 1];
    const b = rgba[p + 2];
    out[i] = b;
    out[planeSize + i] = g;
    out[2 * planeSize + i] = r;
  }
  return out;
}

/**
 * 2-class softmax, mirrors `uniface.common.softmax` for a (1, 2) logits tensor.
 * Returns [p0, p1].
 */
export function softmax2(logits: ArrayLike<number>): [number, number] {
  const l0 = logits[0];
  const l1 = logits[1];
  const m = Math.max(l0, l1);
  const e0 = Math.exp(l0 - m);
  const e1 = Math.exp(l1 - m);
  const z = e0 + e1;
  return [e0 / z, e1 / z];
}
