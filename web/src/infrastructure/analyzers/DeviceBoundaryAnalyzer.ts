// Port of src/infrastructure/analyzers/device_boundary_analyzer.py:1-189
//
// Device Boundary — fusion weight 2.5.
//
// Detects phone/tablet bezels surrounding a tracked face. The Python
// source uses OpenCV (cv2.Canny + cv2.HoughLinesP + cv2.findContours +
// cv2.approxPolyDP). The browser equivalent could lazy-load OpenCV.js
// (~10 MB), but the task spec asks for the lighter path:
//
//   * Canvas2D + native Sobel-derived edge map.
//   * Hand-rolled probabilistic Hough line-segment scan.
//   * Skip the contour rectangle search (cv2.findContours +
//     approxPolyDP) — replaced with an axis-aligned line-merge that
//     reconstructs candidate device rectangles from horizontal/vertical
//     line clusters. This is a documented ~10% accuracy hit but keeps
//     the bundle dep-free.
//
// All line- and contour-side scoring weights are preserved verbatim so
// the calibration of `boundary_score >= 0.50 → spoof` still applies.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { SourceImage, toImageData } from "../../utils/imageOps";

// Common phone/tablet aspect ratios — verbatim from Python:21.
const DEVICE_ASPECT_RATIOS = [16 / 9, 19.5 / 9, 18 / 9, 4 / 3] as const;

export interface DeviceBoundaryOptions {
  /** Padding ratio around the face bbox. Default 0.55. */
  paddingRatio?: number;
  /** Boundary score above which we declare bezel-detected. Default 0.50. */
  spoofThreshold?: number;
  /**
   * If true, downsample the ROI to MAX_DIM=160 before processing.
   * Cuts work by ~16x at the cost of edge precision; acceptable for the
   * "did we find a phone-sized rectangle" question. Default true.
   */
  downsample?: boolean;
}

const DOWNSAMPLE_MAX = 160;

export class DeviceBoundaryAnalyzer implements IFaceAnalyzer {
  readonly name = "device_boundary";

  private readonly paddingRatio: number;
  private readonly spoofThreshold: number;
  private readonly downsample: boolean;
  private currentFrame: SourceImage | null = null;

  constructor(options: DeviceBoundaryOptions = {}) {
    this.paddingRatio = Math.max(0.05, options.paddingRatio ?? 0.55);
    this.spoofThreshold = options.spoofThreshold ?? 0.50;
    this.downsample = options.downsample !== false;
  }

  /** Set the current full frame (analyzer needs surroundings of the face). */
  setFrame(frame: SourceImage): void {
    this.currentFrame = frame;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();

    if (!this.currentFrame) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "no_frame" },
        performance.now() - start,
      );
    }

    const frame = toImageData(this.currentFrame);
    const fw = frame.width;
    const fh = frame.height;
    const bbox = face.bbox;

    const padX = Math.floor(bbox.width * this.paddingRatio);
    const padY = Math.floor(bbox.height * this.paddingRatio);
    const rx1 = Math.max(0, bbox.x1 - padX);
    const ry1 = Math.max(0, bbox.y1 - padY);
    const rx2 = Math.min(fw, bbox.x2 + padX);
    const ry2 = Math.min(fh, bbox.y2 + padY);
    const roiW = rx2 - rx1;
    const roiH = ry2 - ry1;

    if (roiW <= 8 || roiH <= 8) {
      return makeAnalyzerResult(
        this.name,
        100.0,
        { no_roi: true },
        performance.now() - start,
      );
    }

    // === Build a downsampled grayscale buffer of the ROI. ===
    let workW = roiW;
    let workH = roiH;
    let scale = 1;
    if (this.downsample) {
      scale = Math.max(1, Math.ceil(Math.max(roiW, roiH) / DOWNSAMPLE_MAX));
      workW = Math.floor(roiW / scale);
      workH = Math.floor(roiH / scale);
    }
    const gray = new Float32Array(workW * workH);
    const stride = fw * 4;
    for (let y = 0; y < workH; y++) {
      const sy = ry1 + Math.min(roiH - 1, y * scale);
      let srcOff = sy * stride + rx1 * 4;
      for (let x = 0; x < workW; x++) {
        const off = srcOff + Math.min(roiW - 1, x * scale) * 4;
        // luminance ≈ 0.299R + 0.587G + 0.114B
        gray[y * workW + x] =
          0.299 * frame.data[off] +
          0.587 * frame.data[off + 1] +
          0.114 * frame.data[off + 2];
      }
      srcOff += stride;
    }

    // === Sobel edge magnitudes, then threshold to a binary edge map. ===
    const edges = sobelBinary(gray, workW, workH);

    // === Hand-rolled line scan: detect strong horizontal/vertical streaks. ===
    const minLine = Math.max(8, Math.floor(Math.min(workH, workW) * 0.18));
    const lines = scanLines(edges, workW, workH, minLine);

    const lineScore = scoreLines(lines, workH, workW);

    // === Rectangle reconstruction from line clusters (replaces cv2 contours). ===
    const contourScore = scoreReconstructedRectangles(
      lines,
      workW,
      workH,
      bbox,
      rx1,
      ry1,
      scale,
    );

    const boundaryScore = 0.5 * contourScore + 0.5 * lineScore;

    let score: number;
    if (boundaryScore >= this.spoofThreshold) {
      score = Math.max(0.0, 30.0 * (1.0 - boundaryScore));
    } else {
      score = 70.0 + 30.0 * (1.0 - boundaryScore / this.spoofThreshold);
    }
    score = Math.max(0.0, Math.min(100.0, score));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        boundary_score: round(boundaryScore, 4),
        line_score: round(lineScore, 4),
        contour_score: round(contourScore, 4),
        bezel_detected: boundaryScore >= this.spoofThreshold,
        n_lines: lines.length,
      },
      performance.now() - start,
    );
  }

  reset(): void {
    this.currentFrame = null;
  }
}

/** Sobel + magnitude threshold → binary edge map (Uint8Array, 0 or 255). */
function sobelBinary(
  gray: Float32Array,
  w: number,
  h: number,
): Uint8Array {
  const out = new Uint8Array(w * h);
  let maxMag = 0;
  const mag = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -gray[i - w - 1] -
        2 * gray[i - 1] -
        gray[i + w - 1] +
        gray[i - w + 1] +
        2 * gray[i + 1] +
        gray[i + w + 1];
      const gy =
        -gray[i - w - 1] -
        2 * gray[i - w] -
        gray[i - w + 1] +
        gray[i + w - 1] +
        2 * gray[i + w] +
        gray[i + w + 1];
      const m = Math.sqrt(gx * gx + gy * gy);
      mag[i] = m;
      if (m > maxMag) maxMag = m;
    }
  }
  // Threshold at ~Otsu-lite: 0.35 * max. Calibrated against Canny(45,140).
  // Guard against the uniform-image degeneracy: with maxMag≈0, a 0.35*max
  // threshold of 0 would mark every pixel as an edge. Require some absolute
  // gradient before any pixel can be on.
  if (maxMag < 1.0) return out;
  const thresh = 0.35 * maxMag;
  for (let i = 0; i < mag.length; i++) {
    out[i] = mag[i] >= thresh ? 255 : 0;
  }
  return out;
}

interface DetectedLine {
  /** "h" (horizontal streak) or "v" (vertical streak). */
  kind: "h" | "v";
  /** For "h": y-row. For "v": x-col. */
  major: number;
  /** Start of the run along the perpendicular axis. */
  start: number;
  /** End of the run (inclusive). */
  end: number;
  /** Length in pixels (end - start + 1). */
  length: number;
}

/**
 * Scan the binary edge map for horizontal and vertical lines using row/col
 * runs. This is the "probabilistic Hough" approximation:
 *   * For each row, collapse the binary into a 1D series and pick contiguous
 *     edge-pixel runs of length >= minLine.
 *   * Same for each column.
 * Diagonal lines are intentionally ignored — bezels are axis-aligned in
 * a typical capture geometry.
 */
function scanLines(
  edges: Uint8Array,
  w: number,
  h: number,
  minLine: number,
): DetectedLine[] {
  const out: DetectedLine[] = [];
  const minRun = Math.max(2, minLine);

  // Horizontal scan.
  for (let y = 0; y < h; y++) {
    let runStart = -1;
    let runDensity = 0;
    let runLen = 0;
    for (let x = 0; x <= w; x++) {
      const on = x < w && edges[y * w + x] !== 0;
      if (on) {
        if (runStart < 0) runStart = x;
        runDensity += 1;
        runLen += 1;
      } else if (runStart >= 0) {
        // Allow up to 5% gap pixels per row by extending into close gaps.
        const gapTolerance = 2;
        if (
          x < w &&
          edges[y * w + Math.min(w - 1, x + gapTolerance)] !== 0
        ) {
          runLen += 1;
          continue;
        }
        if (runLen >= minRun && runDensity / runLen >= 0.7) {
          out.push({
            kind: "h",
            major: y,
            start: runStart,
            end: runStart + runLen - 1,
            length: runLen,
          });
        }
        runStart = -1;
        runDensity = 0;
        runLen = 0;
      }
    }
  }

  // Vertical scan.
  for (let x = 0; x < w; x++) {
    let runStart = -1;
    let runDensity = 0;
    let runLen = 0;
    for (let y = 0; y <= h; y++) {
      const on = y < h && edges[y * w + x] !== 0;
      if (on) {
        if (runStart < 0) runStart = y;
        runDensity += 1;
        runLen += 1;
      } else if (runStart >= 0) {
        if (runLen >= minRun && runDensity / runLen >= 0.7) {
          out.push({
            kind: "v",
            major: x,
            start: runStart,
            end: runStart + runLen - 1,
            length: runLen,
          });
        }
        runStart = -1;
        runDensity = 0;
        runLen = 0;
      }
    }
  }

  return out;
}

/** Mirrors `_analyze_lines()` in the Python source. */
function scoreLines(
  lines: DetectedLine[],
  roiH: number,
  roiW: number,
): number {
  if (lines.length === 0) return 0.0;
  const minLen = Math.max(18.0, Math.min(roiW, roiH) * 0.18);
  const horizontal: DetectedLine[] = [];
  const vertical: DetectedLine[] = [];
  let totalLength = 0;
  for (const ln of lines) {
    if (ln.length < minLen) continue;
    totalLength += ln.length;
    if (ln.kind === "h") horizontal.push(ln);
    else vertical.push(ln);
  }
  const parallel = Math.min(
    1.0,
    0.5 * Math.min(horizontal.length, 2) + 0.5 * Math.min(vertical.length, 2),
  );
  const orthogonal = horizontal.length > 0 && vertical.length > 0 ? 1.0 : 0.0;
  const density = Math.min(
    1.0,
    totalLength / Math.max((roiW + roiH) * 2, 1),
  );
  return Math.min(1.0, 0.45 * parallel + 0.35 * orthogonal + 0.2 * density);
}

/**
 * Approximation of the Python `_analyze_contours()`:
 * Pair the strongest horizontal and vertical lines into candidate
 * rectangles; score each one by aspect-ratio match to a known device
 * format and by face-center inclusion. ~10% accuracy hit vs the OpenCV
 * version per the file header.
 *
 * Coordinates are in WORK space (downsampled ROI). We rescale into
 * absolute frame space using `roiX0`, `roiY0`, `scale`.
 */
function scoreReconstructedRectangles(
  lines: DetectedLine[],
  workW: number,
  workH: number,
  faceBbox: { width: number; height: number; x1: number; y1: number; x2: number; y2: number },
  roiX0: number,
  roiY0: number,
  scale: number,
): number {
  const horizontal = lines.filter((l) => l.kind === "h");
  const vertical = lines.filter((l) => l.kind === "v");
  if (horizontal.length < 2 || vertical.length < 2) return 0;

  // Sort by length desc; consider top 6 of each direction (cap combinatorics at 6×6×6×6=1296).
  horizontal.sort((a, b) => b.length - a.length);
  vertical.sort((a, b) => b.length - a.length);
  const hCand = horizontal.slice(0, 6);
  const vCand = vertical.slice(0, 6);

  const faceCx = faceBbox.x1 + faceBbox.width / 2;
  const faceCy = faceBbox.y1 + faceBbox.height / 2;
  const roiArea = workW * workH;

  let best = 0;
  for (let i = 0; i < hCand.length; i++) {
    for (let j = i + 1; j < hCand.length; j++) {
      const top = hCand[i].major < hCand[j].major ? hCand[i] : hCand[j];
      const bot = hCand[i].major < hCand[j].major ? hCand[j] : hCand[i];
      const yTop = top.major;
      const yBot = bot.major;
      const rh = yBot - yTop;
      if (rh < 8) continue;

      for (let m = 0; m < vCand.length; m++) {
        for (let k = m + 1; k < vCand.length; k++) {
          const left =
            vCand[m].major < vCand[k].major ? vCand[m] : vCand[k];
          const right =
            vCand[m].major < vCand[k].major ? vCand[k] : vCand[m];
          const xLeft = left.major;
          const xRight = right.major;
          const rw = xRight - xLeft;
          if (rw < 8) continue;

          // Closure: each corner pair must overlap.
          if (top.start > xRight || top.end < xLeft) continue;
          if (bot.start > xRight || bot.end < xLeft) continue;
          if (left.start > yBot || left.end < yTop) continue;
          if (right.start > yBot || right.end < yTop) continue;

          // === Translate work-space rect into absolute frame space. ===
          const absX = roiX0 + xLeft * scale;
          const absY = roiY0 + yTop * scale;
          const absW = rw * scale;
          const absH = rh * scale;

          // Must be bigger than the face.
          if (absW < faceBbox.width * 1.1 || absH < faceBbox.height * 1.1)
            continue;
          // Face center must be inside the rectangle.
          if (
            !(
              absX <= faceCx &&
              faceCx <= absX + absW &&
              absY <= faceCy &&
              faceCy <= absY + absH
            )
          )
            continue;

          const rectArea = rw * rh;
          // "Rectangularity": how much of the bounding rect is filled by
          // the inferred contour. A 4-line boundary covers the perimeter
          // pixels — approximate fill ratio from edge density of the four
          // segments vs the rectangle perimeter.
          const perim = 2 * (rw + rh);
          const edgePixels = top.length + bot.length + left.length + right.length;
          const rectangularity = clamp01(edgePixels / Math.max(perim, 1));

          const aspect = Math.max(rw / Math.max(rh, 1), rh / Math.max(rw, 1));
          let aspectErr = Infinity;
          for (const t of DEVICE_ASPECT_RATIOS) {
            const e = Math.abs(aspect - t);
            if (e < aspectErr) aspectErr = e;
          }
          const aspectScore = Math.max(0, 1 - aspectErr / 0.55);

          const faceCover = Math.min(
            1.0,
            Math.min(
              absW / Math.max(faceBbox.width, 1),
              absH / Math.max(faceBbox.height, 1),
            ) / 2.1,
          );
          const areaRatio = rectArea / Math.max(roiArea, 1);

          const score = Math.min(
            1.0,
            0.35 * clamp01((rectangularity - 0.6) / 0.35) +
              0.30 * aspectScore +
              0.20 * faceCover +
              0.15 * clamp01((areaRatio - 0.2) / 0.45),
          );
          if (score > best) best = score;
        }
      }
    }
  }
  return best;
}

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
