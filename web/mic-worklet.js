// AudioWorklet: the microphone (context rate, usually 48 kHz) -> 16 kHz mono PCM16 LE in 20 ms frames
// (320 samples = 640 bytes), posted to the page with the frame's RMS level.
// Resampler: a windowed-sinc (Blackman) low-pass at 6.8 kHz, evaluated at the two input samples around
// each output position, then linear interpolation between them (~2 x TAPS multiply-adds per sample).
const OUT_RATE = 16000;
const FRAME = 320;
const TAPS = 95;

function lowpass(cutoffHz, rate, n) {
  const h = new Float32Array(n);
  const m = (n - 1) / 2;
  const fc = cutoffHz / rate;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = i - m;
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (n - 1));
    h[i] = sinc * w;
    sum += h[i];
  }
  for (let i = 0; i < n; i++) h[i] /= sum;
  return h;
}

class Pcm16kProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / OUT_RATE; // sampleRate: AudioWorkletGlobalScope global
    this.passthrough = Math.abs(this.step - 1) < 1e-9;
    this.h = this.passthrough ? null : lowpass(Math.min(6800, OUT_RATE * 0.425), sampleRate, TAPS);
    this.buf = new Float32Array(16384);
    this.len = 0;
    this.t = TAPS; // next output position in input-buffer samples (fractional)
    this.frame = new Int16Array(FRAME);
    this.fn = 0;
    this.sumsq = 0;
  }

  filt(n) {
    const h = this.h, b = this.buf;
    let acc = 0;
    for (let j = 0; j < TAPS; j++) acc += h[j] * b[n - j];
    return acc;
  }

  emit(v) {
    const s = v > 1 ? 1 : v < -1 ? -1 : v;
    this.frame[this.fn++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    this.sumsq += s * s;
    if (this.fn === FRAME) {
      const out = this.frame.buffer.slice(0);
      this.port.postMessage({ pcm: out, rms: Math.sqrt(this.sumsq / FRAME) }, [out]);
      this.fn = 0;
      this.sumsq = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.passthrough) {
      for (let i = 0; i < ch.length; i++) this.emit(ch[i]);
      return true;
    }
    if (this.len + ch.length > this.buf.length) { this.len = 0; this.t = TAPS; } // never expected; reset, don't overflow
    this.buf.set(ch, this.len);
    this.len += ch.length;
    for (;;) {
      const i = Math.floor(this.t);
      if (i + 1 >= this.len) break;
      const f = this.t - i;
      const a = this.filt(i);
      this.emit(f > 1e-6 ? a + (this.filt(i + 1) - a) * f : a);
      this.t += this.step;
    }
    const keepFrom = Math.floor(this.t) - TAPS;
    if (keepFrom > 0) {
      this.buf.copyWithin(0, keepFrom, this.len);
      this.len -= keepFrom;
      this.t -= keepFrom;
    }
    return true;
  }
}

registerProcessor('pcm16k', Pcm16kProcessor);
