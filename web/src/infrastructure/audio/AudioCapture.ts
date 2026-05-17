// AudioCapture — Phase D3 (opt-in).
//
// Wraps Web Audio API microphone capture and exposes a rolling
// per-frame RMS buffer that downstream analyzers can sample without
// poking AudioWorklet internals. Falls back gracefully on browsers
// without getUserMedia or AudioContext.
//
// Lifecycle:
//   const cap = new AudioCapture();
//   await cap.start();              // prompts mic permission
//   ... per frame, analyzers read getRecentRms(durationSec)
//   cap.stop();
//
// The RMS buffer is updated by an AudioWorkletNode (when available) or
// a ScriptProcessorNode fallback. Sample rate is whatever the browser
// gives us (typically 48 kHz); RMS frames are produced at ~50 Hz
// (every 20 ms) which is sufficient resolution for voice-activity
// + mouth-sync correlations against ~30 fps blendshape data.

export interface AudioCaptureOptions {
  /**
   * Target RMS frame rate. Browser may round to the nearest power-of-two
   * sample-frames. Default 50 Hz (20 ms frames).
   */
  rmsHz?: number;
  /**
   * Maximum history kept in the rolling buffer, in seconds. Default 5.
   * The mouth-sync analyzer correlates over ~1 s windows; voice activity
   * over ~0.3 s; 5 s leaves plenty of headroom.
   */
  historySec?: number;
}

export interface AudioCaptureLike {
  start(): Promise<void>;
  stop(): Promise<void>;
  readonly isActive: boolean;
  /** Most recent N seconds of RMS samples, newest last. */
  getRecentRms(durationSec: number): Float32Array;
  /** Effective RMS frame rate. */
  readonly rmsHz: number;
}

export class AudioCapture implements AudioCaptureLike {
  private readonly targetRmsHz: number;
  private readonly historySec: number;
  private ctx: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private buffer: Float32Array;
  private head = 0; // next write index in the circular buffer
  private actualRmsHz: number;
  private active = false;

  constructor(options: AudioCaptureOptions = {}) {
    this.targetRmsHz = options.rmsHz ?? 50;
    this.historySec = options.historySec ?? 5;
    this.actualRmsHz = this.targetRmsHz;
    this.buffer = new Float32Array(
      Math.max(16, Math.ceil(this.targetRmsHz * this.historySec)),
    );
  }

  get rmsHz(): number {
    return this.actualRmsHz;
  }

  get isActive(): boolean {
    return this.active;
  }

  async start(): Promise<void> {
    if (this.active) return;
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      throw new Error("AudioCapture: getUserMedia not available");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AudioCtx =
      (window as unknown as { AudioContext: typeof AudioContext })
        .AudioContext ??
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    if (!AudioCtx) throw new Error("AudioCapture: AudioContext not available");
    this.ctx = new AudioCtx();
    this.source = this.ctx.createMediaStreamSource(this.stream);

    // ScriptProcessor is deprecated but universally available and gives
    // us per-buffer RMS without needing an AudioWorklet bundle. Buffer
    // size 2048 @ 48 kHz ≈ 23 RMS frames/sec; we want ~50, so use 1024
    // (≈ 46 Hz at 48 kHz). The browser may negotiate a different size.
    const desired = Math.max(
      256,
      Math.min(16384, 2 ** Math.round(Math.log2(this.ctx.sampleRate / this.targetRmsHz))),
    );
    this.processor = this.ctx.createScriptProcessor(desired, 1, 1);
    this.actualRmsHz = this.ctx.sampleRate / desired;
    this.buffer = new Float32Array(
      Math.max(16, Math.ceil(this.actualRmsHz * this.historySec)),
    );
    this.head = 0;
    this.processor.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      let s = 0;
      for (let i = 0; i < input.length; i++) s += input[i] * input[i];
      const rms = Math.sqrt(s / input.length);
      this.buffer[this.head] = rms;
      this.head = (this.head + 1) % this.buffer.length;
    };
    this.source.connect(this.processor);
    // ScriptProcessor must be connected to the destination to fire; we
    // route it through a 0-gain node so the user doesn't hear feedback.
    const silentGain = this.ctx.createGain();
    silentGain.gain.value = 0;
    this.processor.connect(silentGain);
    silentGain.connect(this.ctx.destination);
    this.active = true;
  }

  async stop(): Promise<void> {
    if (!this.active) return;
    this.active = false;
    try {
      this.processor?.disconnect();
    } catch {
      /* noop */
    }
    try {
      this.source?.disconnect();
    } catch {
      /* noop */
    }
    if (this.stream) {
      for (const t of this.stream.getTracks()) t.stop();
    }
    if (this.ctx && this.ctx.state !== "closed") {
      await this.ctx.close();
    }
    this.ctx = null;
    this.source = null;
    this.processor = null;
    this.stream = null;
  }

  getRecentRms(durationSec: number): Float32Array {
    const n = Math.min(
      this.buffer.length,
      Math.max(1, Math.ceil(durationSec * this.actualRmsHz)),
    );
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      // Read backward from head; chronologically newest last.
      const src =
        (this.head - n + i + this.buffer.length * 2) % this.buffer.length;
      out[i] = this.buffer[src];
    }
    return out;
  }
}
