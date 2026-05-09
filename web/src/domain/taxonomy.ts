// Port of src/domain/taxonomy.py
// Maps analyzer signals to spoof category probability contributions.
//
// Key insight: different analyzers detect different attack types.
// - MiniFASNet:  general binary (catches static, replay, mask)
// - Texture:     catches static images and printed photos
// - Moire:       catches screen-based attacks (replay, static on screen)
// - Screen Replay: catches display-based attacks
// - Temporal:    catches static images (no motion)
// - rPPG:        catches static, replay, mask (no pulse)
// - AR Filter:   specifically detects AR overlays
// - Makeup:      detects heavy contouring
//
// Each entry is a partial map: only the categories an analyzer can
// indicate are listed. Missing categories implicitly contribute 0.
//
// NOTE: the Python source has duplicate "rppg" keys; the second one
// shadows the first (both are identical, so behavior is preserved).
// We keep a single entry here for clarity.

import { SpoofCategory } from "./models";

export type SpoofSignalRow = Partial<Record<SpoofCategory, number>>;

export const SPOOF_SIGNAL_MAP: Readonly<Record<string, SpoofSignalRow>> = {
  minifasnet: {
    [SpoofCategory.STATIC_IMAGE]: 0.35,
    [SpoofCategory.VIDEO_REPLAY]: 0.25,
    [SpoofCategory.MASK_3D]: 0.20,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.10,
    [SpoofCategory.AR_FILTER]: 0.05,
    [SpoofCategory.HEAVY_MAKEUP]: 0.05,
  },
  texture: {
    [SpoofCategory.STATIC_IMAGE]: 0.50,
    [SpoofCategory.VIDEO_REPLAY]: 0.30,
    [SpoofCategory.MASK_3D]: 0.10,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.10,
  },
  moire: {
    [SpoofCategory.VIDEO_REPLAY]: 0.50,
    [SpoofCategory.STATIC_IMAGE]: 0.40,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.10,
  },
  screen_replay: {
    [SpoofCategory.VIDEO_REPLAY]: 0.45,
    [SpoofCategory.STATIC_IMAGE]: 0.35,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.20,
  },
  temporal: {
    [SpoofCategory.STATIC_IMAGE]: 0.70,
    [SpoofCategory.MASK_3D]: 0.15,
    [SpoofCategory.VIDEO_REPLAY]: 0.15,
  },
  rppg: {
    [SpoofCategory.STATIC_IMAGE]: 0.30,
    [SpoofCategory.VIDEO_REPLAY]: 0.25,
    [SpoofCategory.MASK_3D]: 0.25,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.20,
  },
  ar_filter: {
    [SpoofCategory.AR_FILTER]: 0.85,
    [SpoofCategory.HEAVY_MAKEUP]: 0.10,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.05,
  },
  makeup: {
    [SpoofCategory.HEAVY_MAKEUP]: 0.70,
    [SpoofCategory.AR_FILTER]: 0.20,
    [SpoofCategory.MASK_3D]: 0.10,
  },
  device_boundary: {
    [SpoofCategory.STATIC_IMAGE]: 0.40,
    [SpoofCategory.VIDEO_REPLAY]: 0.40,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.20,
  },
  blink: {
    [SpoofCategory.STATIC_IMAGE]: 0.40,
    [SpoofCategory.VIDEO_REPLAY]: 0.20,
    [SpoofCategory.MASK_3D]: 0.15,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.15,
    [SpoofCategory.AR_FILTER]: 0.10,
  },
  landmark_variance: {
    [SpoofCategory.STATIC_IMAGE]: 0.50,
    [SpoofCategory.VIDEO_REPLAY]: 0.15,
    [SpoofCategory.MASK_3D]: 0.15,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.10,
    [SpoofCategory.AR_FILTER]: 0.10,
  },
  screen_flicker: {
    [SpoofCategory.VIDEO_REPLAY]: 0.50,
    [SpoofCategory.STATIC_IMAGE]: 0.30,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.20,
  },
  micro_tremor: {
    [SpoofCategory.STATIC_IMAGE]: 0.30,
    [SpoofCategory.VIDEO_REPLAY]: 0.30,
    [SpoofCategory.MASK_3D]: 0.20,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.20,
  },
  background_grid: {
    [SpoofCategory.STATIC_IMAGE]: 0.40,
    [SpoofCategory.VIDEO_REPLAY]: 0.40,
    [SpoofCategory.DEEPFAKE_INJECT]: 0.20,
  },
};
