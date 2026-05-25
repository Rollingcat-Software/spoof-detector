// FlashTemporalAnalyzer — active-illumination TEMPORAL liveness (OPT-IN).
//
// The strongest signal against a *video-replay on a screen*, and fully
// content-independent (unlike the WB colour-cast probe, which depends on the
// replayed scene's white point). It is the partner of FlashReflectionAnalyzer:
// where that one scores the *spatial* reflection (region spread, chroma), this
// one scores the *temporal* response of the face-region brightness to a flash.
//
// Physics / observed behaviour (validated live 2026-05-24, real face vs phone
// video replay, exposure locked, ~1.5 s white flash sampled ~10 Hz):
//
//   REAL 3D FACE — passive reflector:
//     baseline 110 → flash [116,117,116,116,116,116,117,117,118,118,118,…]
//                  → after [111,110,110,110,110,111,110,110]
//     • brightness JUMPS on the first lit frame (no lag)
//     • returns to baseline the instant the flash ends (no persistence)
//
//   SCREEN / VIDEO REPLAY — phone with auto-brightness:
//     baseline 113 → flash [111,109,108,115,114,119,118,117,118,121,120,122,…]
//                  → after [117,116,116,119,118,119,117,120]
//     • the phone's sensor SEES our flash and ramps its OWN backlight up over
//       ~1 s → a delayed, gradual rise (onset lag)
//     • the backlight stays elevated after the flash ends → persistence
//
// Two clean discriminators fall out, both content-independent:
//   1. ONSET LAG     — time to reach 80 % of the peak rise. Real ≈ 0;
//                      a screen's auto-brightness ramps over ~0.6–1.0 s.
//   2. PERSISTENCE   — (after-mean − baseline-mean) / peak-rise. Real ≈ 0;
//                      a screen stays bright because its backlight latched up.
//
// EITHER signal firing ⇒ screen. A real face fails BOTH.
//
// Caveats (must surface to the operator, never silently trust):
//   * Relies on the screen having AUTO-BRIGHTNESS ON and our flash reaching its
//     ambient-light sensor. A fixed-brightness screen won't ramp — then we lean
//     on the (separate) reflection/persistence and may report INCONCLUSIVE.
//   * If the flash never measurably reaches the face (bright room / distant
//     desktop monitor), the rise is below `minRise` ⇒ INCONCLUSIVE, never a
//     false SPOOF.
//
// INTENTIONALLY NOT part of the passive per-frame fusion — flashing the screen
// is intrusive, so it runs only when a caller explicitly drives a flash
// challenge (e.g. the amispoof "verify with light" probe). Thresholds are the
// live-calibrated defaults and remain tunable per camera / ambient light.

export interface FlashTemporalResult {
  /**
   * True when the flash produced too small a brightness rise to judge (the
   * light never reached the face — bright room / distant monitor). The caller
   * must NOT treat this as SPOOF.
   */
  inconclusive: boolean;
  /**
   * True when the temporal response looks like a screen / video-replay: a
   * delayed onset ramp and/or a backlight that stays elevated after the flash.
   * Only meaningful when `inconclusive` is false.
   */
  isScreen: boolean;
  /** 0-100, higher = more screen-like (max of the lag and persistence parts). */
  screenScore: number;

  /** Mean face-region brightness before the flash (0-255). */
  baselineMean: number;
  /** Peak face-region brightness during the flash (0-255). */
  peak: number;
  /** peak − baselineMean (0-255). The magnitude of the photometric response. */
  riseTotal: number;
  /** Samples until brightness first reached 80 % of the peak rise (-1 if none). */
  onsetLagSamples: number;
  /** onsetLagSamples × sampleIntervalMs — the onset lag in milliseconds. */
  onsetLagMs: number;
  /** afterMean − baselineMean (0-255). How elevated the face stays post-flash. */
  persistence: number;
  /** persistence / riseTotal — normalised so it's content/brightness-independent. */
  persistenceNorm: number;
}

export interface FlashTemporalOptions {
  /** Spacing between successive samples, ms. Required to report lag in ms. Default 100. */
  sampleIntervalMs?: number;
  /**
   * Minimum peak rise (0-255) for the flash to count as a usable response.
   * Below this the normalised persistence (persistence / rise) divides by a
   * tiny, noisy denominator and can't be trusted, so we report INCONCLUSIVE
   * rather than risk a silent miss or a false flag. This also covers a screen
   * whose backlight is already maxed (a rapid re-probe sees almost no further
   * rise). Default 8 (~3 % brightness). The reliable signal needs the phone at
   * rest — the session auto-probe fires on a 45 s cadence so each flash sees a
   * fresh ramp.
   */
  minRise?: number;
  /** Normalised persistence mapped to the bottom of the ramp. Default 0.15. */
  persistenceLow?: number;
  /** Normalised persistence mapped to a full screen-likeness score. Default 0.45. */
  persistenceHigh?: number;
  /** screenScore at/above which the response is judged a screen. Default 50. */
  screenThreshold?: number;
  /** Fraction of the peak rise that defines "onset reached". Default 0.8. */
  onsetFraction?: number;
  /**
   * Number of leading after-samples to DISCARD before measuring persistence.
   * A webcam delivers frames ~150-200 ms late, so the first ~2 post-flash
   * samples still contain the flash's tail and would inflate persistence on a
   * real face. Persistence is measured from the settled samples after this.
   * Default 2.
   */
  afterSkip?: number;
  /**
   * Baseline brightness (0-255) above which the face is judged OVER-LIT and the
   * probe abstains. When the room light already drives the face near sensor
   * saturation there is no headroom for the flash to register, so the tiny rise
   * yields a noise-dominated persistence (live runs in a bright room produced
   * false positives at baseline ≈210-234). The reliable discriminator is the
   * baseline, not the rise: clean reads sat at baseline ≤145. Default 185.
   */
  baselineCeiling?: number;
}

export class FlashTemporalAnalyzer {
  readonly name = "flash_temporal";

  private readonly sampleIntervalMs: number;
  private readonly minRise: number;
  private readonly persistenceLow: number;
  private readonly persistenceHigh: number;
  private readonly screenThreshold: number;
  private readonly onsetFraction: number;
  private readonly afterSkip: number;
  private readonly baselineCeiling: number;

  constructor(options: FlashTemporalOptions = {}) {
    this.sampleIntervalMs = options.sampleIntervalMs ?? 100;
    this.minRise = options.minRise ?? 8;
    this.persistenceLow = options.persistenceLow ?? 0.15;
    this.persistenceHigh = options.persistenceHigh ?? 0.45;
    this.screenThreshold = options.screenThreshold ?? 50;
    this.onsetFraction = options.onsetFraction ?? 0.8;
    this.afterSkip = options.afterSkip ?? 2;
    this.baselineCeiling = options.baselineCeiling ?? 185;
  }

  /**
   * Score the face-region brightness time-series of a flash challenge.
   *
   * @param baselineSamples brightness samples (0-255) captured before the flash
   * @param flashSamples     brightness samples captured during the flash
   * @param afterSamples     brightness samples captured after the flash ended
   */
  score(
    baselineSamples: number[],
    flashSamples: number[],
    afterSamples: number[],
  ): FlashTemporalResult {
    const baselineMean = mean(baselineSamples);
    // Persistence is measured from the SETTLED after-samples — discard the
    // leading camera-latency tail (which still holds the flash) unless that
    // would leave us with nothing to measure.
    const afterSettled =
      afterSamples.length > this.afterSkip
        ? afterSamples.slice(this.afterSkip)
        : afterSamples;
    const afterMean = mean(afterSettled);
    const peak = flashSamples.length ? Math.max(...flashSamples) : baselineMean;
    const riseTotal = peak - baselineMean;

    // Can't judge when: no samples; the face is OVER-LIT (room light already
    // near saturation → no headroom for the flash, so persistence is noise); or
    // the flash produced too small a rise (didn't reach the face / maxed screen).
    if (
      flashSamples.length === 0 ||
      baselineSamples.length === 0 ||
      baselineMean > this.baselineCeiling ||
      riseTotal < this.minRise
    ) {
      const persistenceInc = afterMean - baselineMean;
      return {
        inconclusive: true,
        isScreen: false,
        screenScore: 0,
        baselineMean: round(baselineMean, 2),
        peak: round(peak, 2),
        riseTotal: round(riseTotal, 2),
        onsetLagSamples: -1,
        onsetLagMs: -1,
        persistence: round(persistenceInc, 2),
        persistenceNorm: 0,
      };
    }

    // 1. Onset lag — samples until brightness first reaches 80 % of the peak
    //    rise. INFORMATIONAL ONLY (not part of the decision): a real face's
    //    onset is confounded by subject motion — a head that drifts while the
    //    flash ramps reads a slow onset and false-flags. Live calibration
    //    (2026-05-25) showed 1 in 5 real-face runs hitting ~1400 ms purely from
    //    movement. We keep it for diagnostics but never let it drive the score.
    const onsetTarget = baselineMean + this.onsetFraction * riseTotal;
    let onsetLagSamples = -1;
    for (let i = 0; i < flashSamples.length; i++) {
      if (flashSamples[i] >= onsetTarget) {
        onsetLagSamples = i;
        break;
      }
    }
    if (onsetLagSamples < 0) onsetLagSamples = flashSamples.length - 1;
    const onsetLagMs = onsetLagSamples * this.sampleIntervalMs;

    // 2. Persistence — how elevated the face stays after the flash ends,
    //    normalised by the rise so it's brightness-independent. THE DECISIVE
    //    SIGNAL: a passive reflector (real 3D face) drops straight back the
    //    instant the flash ends, regardless of how the subject moved; a screen
    //    whose auto-brightness latched its backlight up stays elevated. Live
    //    real-face runs clustered at ≤0.02; a phone replay sat ~0.55.
    const persistence = afterMean - baselineMean;
    const persistenceNorm = persistence / Math.max(riseTotal, 1e-6);

    const persPart = clamp01(
      (persistenceNorm - this.persistenceLow) /
        Math.max(1e-6, this.persistenceHigh - this.persistenceLow),
    );
    const screenScore = Math.round(persPart * 100);

    return {
      inconclusive: false,
      isScreen: screenScore >= this.screenThreshold,
      screenScore,
      baselineMean: round(baselineMean, 2),
      peak: round(peak, 2),
      riseTotal: round(riseTotal, 2),
      onsetLagSamples,
      onsetLagMs,
      persistence: round(persistence, 2),
      persistenceNorm: round(persistenceNorm, 3),
    };
  }
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
