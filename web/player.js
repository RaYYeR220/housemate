// Playout of the hub's 0x02 frames: PCM16 LE mono at the rate from play.begin (24 kHz from the hub's own
// speech, any other rate from a streamed voice; the browser resamples each buffer to the output rate).
// A small jitter buffer before the first sound, back-to-back scheduling after it, a fresh lead-in after an
// underrun. done(stream_id) fires once a stream that ended has really been heard; abort never reports.

const PREBUFFER_S = 0.12; // collect this much (or the end of the stream) before starting
const LEAD_S = 0.04; // start this far ahead of "now" (first buffer, after an underrun)
const HOP_S = 0.02; // level envelope step

export class Player {
  constructor(ctx, onDone) {
    this.ctx = ctx;
    this.onDone = onDone;
    this.out = ctx.createGain();
    this.out.connect(ctx.destination);
    this.sources = new Set();
    this.underruns = 0;
    this._reset(null, 24000);
  }

  _reset(sid, rate) {
    this.sid = sid;
    this.rate = rate;
    this.pending = [];
    this.pendingS = 0;
    this.started = false;
    this.ended = false;
    this.next = 0;
    this.first = 0;
    this.carry = null;
    this.env = [];
  }

  get active() { return this.sid !== null; }

  begin(sid, rate) {
    this._silence();
    this._reset(sid, rate >= 3000 && rate <= 192000 ? rate : 24000);
  }

  push(view) {
    if (this.sid === null) return;
    let bytes = view;
    if (this.carry) { const m = new Uint8Array(bytes.length + 1); m[0] = this.carry; m.set(bytes, 1); bytes = m; this.carry = null; }
    if (bytes.length & 1) { this.carry = bytes[bytes.length - 1]; bytes = bytes.subarray(0, bytes.length - 1); }
    const n = bytes.length >> 1;
    if (!n) return;
    const pcm = new Int16Array(bytes.slice().buffer, 0, n); // copy: the frame's offset 1 is not 2-byte aligned
    const buf = this.ctx.createBuffer(1, n, this.rate), ch = buf.getChannelData(0);
    const hop = Math.max(1, Math.round(this.rate * HOP_S)), lv = new Float32Array(Math.ceil(n / hop));
    let acc = 0;
    for (let i = 0; i < n; i++) {
      const v = pcm[i] / 32768; ch[i] = v; acc += v * v;
      if ((i + 1) % hop === 0 || i === n - 1) { const k = (i / hop) | 0, cnt = i % hop + 1; lv[k] = Math.min(1, Math.sqrt(acc / cnt) * 4.5); acc = 0; }
    }
    buf.lv = lv; buf.hop = hop / this.rate;
    if (this.started) this._play(buf);
    else {
      this.pending.push(buf); this.pendingS += buf.duration;
      if (this.pendingS >= PREBUFFER_S) this._flush();
    }
  }

  end(sid) {
    if (sid !== this.sid) return;
    this.ended = true;
    if (!this.started) this._flush();
    this._check(sid);
  }

  abort(sid) {
    if (sid && sid !== this.sid) return false;
    this._silence();
    this._reset(null, this.rate);
    return true;
  }

  // 0..1 loudness of what is coming out of the speaker now
  level() {
    const t = this.ctx.currentTime, env = this.env;
    while (env.length && env[0].t1 < t) env.shift();
    for (let i = 0; i < env.length; i++) {
      const e = env[i];
      if (t >= e.t0 && t < e.t1) return e.lv[Math.min(e.lv.length - 1, ((t - e.t0) / e.hop) | 0)];
    }
    return 0;
  }

  played() { return this.first ? Math.max(0, this.ctx.currentTime - this.first) : 0; }

  _flush() {
    this.started = true;
    const list = this.pending; this.pending = []; this.pendingS = 0;
    for (let i = 0; i < list.length; i++) this._play(list[i]);
  }

  _play(buf) {
    const now = this.ctx.currentTime;
    if (this.next < now + 0.005) { if (this.first) this.underruns++; this.next = now + LEAD_S; }
    const src = this.ctx.createBufferSource(), sid = this.sid;
    src.buffer = buf; src.connect(this.out); src.start(this.next);
    if (!this.first) this.first = this.next;
    if (this.env.length < 64) this.env.push({ t0: this.next, t1: this.next + buf.duration, lv: buf.lv, hop: buf.hop });
    this.next += buf.duration;
    this.sources.add(src);
    src.onended = () => { this.sources.delete(src); this._check(sid); };
  }

  _check(sid) {
    if (sid !== null && sid === this.sid && this.ended && !this.sources.size && !this.pending.length) {
      this._reset(null, this.rate);
      this.onDone(sid);
    }
  }

  _silence() {
    for (const s of this.sources) { s.onended = null; try { s.stop(); } catch (_) { /* never started */ } }
    this.sources.clear();
  }
}
