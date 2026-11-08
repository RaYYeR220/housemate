// The round room: one Canvas 2D object. A plan of the home in cream hairlines: this room is a circle
// with a plan door; a route runs from the door to the hub square in the next room (cut by the screen's
// corner). The phone sits on the wall at the bottom; 32 square dots ride the wall either side of it.
// Listening: ink rings close onto the phone. Thinking: a mark runs the route to the hub and back.
// Speaking: cream rings leave the phone, their wall points climb the wall, the far wall reflects.
// Ringing: the wall flashes. Prompt/card: the room steps back to the top, the door swings shut.
// The app calls update/draw at a capped rate and stops calling them while the page is hidden.

const CREAM = '#fff4ea', INK = '#1b1410', TOMATO = '#c4452a', AMB_GROUND = '#2a1410';
const WALL_A = 0.46, TAU = Math.PI * 2, HP = Math.PI / 2;
const DASH = [3, 4], NODASH = [];
const cl = (v, a, b) => (v < a ? a : v > b ? b : v);
const wrap = (a) => { while (a > Math.PI) a -= TAU; while (a < -Math.PI) a += TAU; return a; };
const easeOut = (x) => 1 - Math.pow(1 - x, 3);
const THINK_P = 2.2; // one trip to the hub and back while thinking
export const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

export function createFace(cv, inner, micoff, door) {
  const ctx = cv.getContext('2d', { alpha: true });
  const root = document.documentElement;
  let W = 0, Hh = 0, dpr = 1, tr = true, lap = false, small = false;
  const G = { cx: 0, cy: 0, R: 100 }, T = { cx: 0, cy: 0, R: 100 }; // the room now, its target
  let R0 = 100, cx0 = 0, cy0 = 0;
  let ox = 0, oy = 0, oR = 100, hx = 0, hy = 0; // the next room (cut by the corner) and its hub square
  let thD = 3.9, gapD = 0.3, dX = 0, dY = 0, rX = 0, rY = 0, routeLen = 100, oGapA = -9, wallArcA = 2.5;
  let phX = 0, phY = 0, Rmax = 400, maxD = 8;
  const AX = new Float32Array(2), AY = new Float32Array(2), ABASE = new Float32Array(2);
  let st0 = 'idle';

  function targetFor(st) {
    const content = st === 'prompt' || st === 'card';
    if (tr) {
      if (content) { T.R = R0 * 0.6; T.cx = W / 2; T.cy = Hh * 0.2; } else { T.R = R0; T.cx = cx0; T.cy = cy0; }
    } else if (!lap) {
      if (st === 'card') { T.R = R0 * 0.62; T.cx = W * 0.19; T.cy = Hh / 2 + 34; }
      else if (content) { T.R = R0 * 0.68; T.cx = W * 0.215; T.cy = Hh / 2 + 30; }
      else { T.R = R0; T.cx = cx0; T.cy = cy0; }
    } else if (content) { T.R = R0 * 0.7; T.cx = W * 0.25; T.cy = Hh / 2 + 20; }
    else { T.R = R0; T.cx = cx0; T.cy = cy0; }
    small = content;
  }
  function setVar(k, v) { root.style.setProperty(k, Math.round(v) + 'px'); }
  function layout() {
    if (tr) {
      R0 = Math.min(W / 2 - 20, Hh * 0.21); cx0 = W / 2; cy0 = Hh * 0.42;
      ox = -0.1 * W; oy = -0.03 * Hh; oR = 0.44 * W; hx = 28; hy = Math.round(0.077 * Hh);
      setVar('--cap-top', cy0 + R0 + 24); setVar('--sheet-top', Hh * 0.2 + R0 * 0.6 + 26);
    } else if (!lap) {
      R0 = Hh / 2 - 34; cx0 = W * 0.4; cy0 = Hh / 2 + 8;
      ox = -50; oy = -40; oR = 190; hx = 26; hy = 56;
      setVar('--cap-left', cx0 + R0 + 34); setVar('--sheet-left', W * 0.215 + R0 * 0.68 + 24);
    } else {
      R0 = Math.min(Hh / 2 - 100, 320); cx0 = W / 2 - 190; cy0 = Hh / 2 + 10;
      ox = -60; oy = -60; oR = 300; hx = 44; hy = 80;
      setVar('--cap-left', cx0 + R0 + 48); setVar('--sheet-left', W * 0.25 + R0 * 0.7 + 40);
    }
    door.style.transform = 'translate(' + (hx + 6) + 'px,' + hy + 'px) translateY(-50%)';
  }
  function derive() {
    const { cx, cy, R } = G;
    thD = Math.atan2(hy - cy, hx - cx); if (thD < 0) thD += TAU; // the door faces the hub
    gapD = cl(54 / R, 0.22, 0.42);
    dX = cx + R * Math.cos(thD); dY = cy + R * Math.sin(thD);
    rX = hx - dX; rY = hy - dY; routeLen = Math.hypot(rX, rY) || 1;
    const ux = rX / routeLen, uy = rY / routeLen, fx = dX - ox, fy = dY - oy;
    const b = fx * ux + fy * uy, c = fx * fx + fy * fy - oR * oR, disc = b * b - c;
    oGapA = disc > 0 ? Math.atan2(dY + (-b - Math.sqrt(disc)) * uy - oy, dX + (-b - Math.sqrt(disc)) * ux - ox) : -9;
    phX = cx; phY = cy + R; Rmax = 2.5 * R;
    wallArcA = thD - HP;
    AX[0] = phX; AY[0] = phY; ABASE[0] = HP;
    AX[1] = cx + (R - 8) * Math.cos(thD); AY[1] = cy + (R - 8) * Math.sin(thD); ABASE[1] = thD;
    maxD = Math.min(R * 0.0576 * 0.82, 10);
    micoff.style.transform = 'translate(' + Math.round(phX + 16) + 'px,' + Math.round(phY - 34) + 'px)';
    inner.style.transform = 'translate(' + cx.toFixed(1) + 'px,' + cy.toFixed(1) + 'px) scale(' + (R / R0).toFixed(4) + ')';
  }
  function resize() {
    W = innerWidth; Hh = innerHeight; dpr = Math.min(devicePixelRatio || 1, 1.5); tr = Hh >= W; lap = !tr && Hh > 620;
    cv.width = Math.round(W * dpr); cv.height = Math.round(Hh * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    layout(); targetFor(st0); G.cx = T.cx; G.cy = T.cy; G.R = T.R; derive();
    root.classList.toggle('small', small); force = true;
  }

  // the 32 dots on the wall, 16 either side of the phone
  const ND = 32, HL = 17;
  const HIST = new Float32Array(HL), SZ = new Float32Array(ND).fill(3), RS = new Float32Array(ND), DA = new Float32Array(ND);
  for (let i = 0; i < ND; i++) {
    RS[i] = 0.55 + 0.45 * Math.abs(Math.sin(i * 2.399 + Math.sin(i * 0.7)));
    const k = i < 16 ? 15 - i : i - 16, side = i < 16 ? 1 : -1;
    DA[i] = HP + side * (0.16 + k * 0.0576);
  }
  let histAcc = 0;

  // rings (anchor 0 = the phone, 1 = the door)
  const NR = 32;
  const rAn = new Int8Array(NR), rRef = new Int8Array(NR);
  const rBorn = new Float64Array(NR).fill(-1e12), rStr = new Float32Array(NR), rV = new Float32Array(NR), rLife = new Float32Array(NR);
  function spawn(now, an, refl, str, v, life, delay) {
    for (let i = 0; i < NR; i++) {
      if (rBorn[i] + rLife[i] * 1000 < now && rBorn[i] <= now) {
        rAn[i] = an; rRef[i] = refl; rStr[i] = str; rV[i] = v; rLife[i] = life; rBorn[i] = now + (delay || 0) * 1000; return;
      }
    }
  }
  function ringsAlive(now) { for (let i = 0; i < NR; i++) if (rBorn[i] <= now && now - rBorn[i] < rLife[i] * 1000) return true; return false; }
  const MN = 128, MX = new Float32Array(MN), MY = new Float32Array(MN), MS = new Float32Array(MN), MA = new Float32Array(MN), MC = new Int8Array(MN);
  let mN = 0;
  function marks(base, r, al, col, s) { // where a ring from a wall point meets the round wall
    const R = G.R; if (r <= 0 || r >= 2 * R || mN >= MN - 2 || al < 0.03) return;
    const d = 2 * Math.asin(r / (2 * R));
    for (let k = -1; k <= 1; k += 2) {
      const a = base + k * d; if (Math.abs(wrap(a - thD)) < gapD / 2 + 0.03) continue;
      MX[mN] = G.cx + R * Math.cos(a); MY[mN] = G.cy + R * Math.sin(a); MS[mN] = s; MA[mN] = al; MC[mN] = col; mN++;
    }
  }

  let lvS = 0, A = 1, spawnAcc = 0, phase = 0, lastBeat = -1, prims = 0, force = true, entered = 0, frozen = false, lastBlip = -1e9;
  function onState(st, now) {
    st0 = st; entered = now; force = true;
    targetFor(st); root.classList.toggle('small', small);
    if (REDUCED) { G.cx = T.cx; G.cy = T.cy; G.R = T.R; derive(); }
    if (st === 'prompt' && !ringsAlive(now)) for (let k = 0; k < 6; k++) spawn(now, 0, 1, 0.75 - k * 0.07, Rmax / 1.9, 1.9, -(0.12 + k * 0.2));
    if (st === 'announce') { spawn(now, 1, 0, 0.95, Rmax / 1.8, 1.6, 0.4); spawn(now, 1, 0, 0.65, Rmax / 1.8, 1.6, 0.52); spawn(now, 1, 0, 0.4, Rmax / 1.8, 1.6, 0.64); }
  }

  function update(dt, now, S) {
    const st = S.state, lt = S.lt;
    if (Math.abs(G.cx - T.cx) + Math.abs(G.cy - T.cy) + Math.abs(G.R - T.R) > 0.2) {
      const k = 1 - Math.exp(-dt / 0.09);
      G.cx += (T.cx - G.cx) * k; G.cy += (T.cy - G.cy) * k; G.R += (T.R - G.R) * k;
      if (Math.abs(G.cx - T.cx) + Math.abs(G.cy - T.cy) + Math.abs(G.R - T.R) <= 0.2) { G.cx = T.cx; G.cy = T.cy; G.R = T.R; }
      derive(); force = true;
    }
    const lv = S.lv; lvS += (lv - lvS) * (1 - Math.exp(-dt / (lv > lvS ? 0.03 : 0.15)));
    const tA = st === 'prompt' ? 0.35 : st === 'offline' ? 0.5 : 1;
    A += (tA - A) * (1 - Math.exp(-dt * 6));
    const voice = st === 'listening' || st === 'followup' || st === 'speaking' || st === 'announce';
    histAcc += dt;
    while (histAcc >= 0.045) { histAcc -= 0.045; HIST.copyWithin(1, 0, HL - 1); HIST[0] = voice ? lv : 0; }
    frozen = st === 'prompt';
    if (!frozen) {
      const t = now / 1000, k = 1 - Math.exp(-dt / 0.04);
      let amp = 0;
      if (st === 'ringing') { const ph = lt % 1.2; amp = Math.max(Math.exp(-ph / 0.11), ph >= 0.18 ? Math.exp(-(ph - 0.18) / 0.11) : 0); }
      for (let i = 0; i < ND; i++) {
        const kk = i < 16 ? 15 - i : i - 16;
        let v = 0;
        if (REDUCED) v = 0;
        else if (st === 'listening' || st === 'followup') v = HIST[15 - kk] * (st === 'followup' ? 0.5 : 1.25);
        else if (st === 'speaking' || st === 'announce') v = HIST[kk] * 1.3;
        else if (st === 'ringing') v = amp * RS[i];
        else if (st === 'idle' || st === 'card' || st === 'thinking') { const w = Math.sin(t * Math.PI * 2 / 8 - kk * 0.3); v = w > 0 ? 0.16 * w * w * w * w * w * w : 0; }
        SZ[i] += ((st === 'offline' ? 2 : 3) + (maxD - 3) * cl(v, 0, 1) - SZ[i]) * k;
      }
    }
    if (REDUCED) { spawnAcc = 0; return; }
    if (st === 'speaking' || st === 'announce') {
      spawnAcc += dt;
      while (spawnAcc >= 0.18) { spawnAcc -= 0.18; if (lvS > 0.1 && (st === 'speaking' || now - entered > 700)) spawn(now, 0, 1, Math.min(1, 0.25 + lvS), Rmax / 1.9, 1.9, 0); }
    } else spawnAcc = 0;
    if (st === 'listening' || st === 'followup') phase += dt * (st === 'followup' ? 0.3 : 0.5 + 0.8 * lvS);
    if (st === 'ringing') {
      const b = Math.floor(lt / 1.2) * 2 + ((lt % 1.2) >= 0.18 ? 1 : 0);
      if (b !== lastBeat) { lastBeat = b; spawn(now, 0, 1, 1, Rmax / 1.1, 1.1, 0); }
    } else lastBeat = -1;
    if (st === 'offline' && S.blipAt > 0 && Math.abs(S.blipAt - lastBlip) > 300 && now - S.blipAt < 400) { lastBlip = S.blipAt; force = true; }
  }

  const SIG = new Float64Array(6), PSIG = new Float64Array(6);
  function same(now, S) { // nothing moved since the last frame: skip the redraw
    const st = S.state;
    if (force || (st !== 'prompt' && ringsAlive(now)) || st === 'listening' || st === 'followup' || st === 'speaking' || st === 'announce' || st === 'ringing' || st === 'thinking') return false;
    if (st === 'offline' && now - S.blipAt < 600) return false;
    let h = 0; for (let i = 0; i < ND; i++) h = (h * 31 + Math.round(SZ[i] * 4)) | 0;
    SIG[0] = S.key; SIG[1] = Math.round(A * 200); SIG[2] = (st === 'idle' || st === 'card') && !S.ambient && !REDUCED ? Math.floor((now % 10000) / 10000 * 240) : -1;
    SIG[3] = st === 'prompt' ? Math.floor(S.promptShut * 180) : -1; SIG[4] = S.ambient ? 1 : 0; SIG[5] = h;
    let eq = true; for (let i = 0; i < 6; i++) if (SIG[i] !== PSIG[i]) { eq = false; PSIG[i] = SIG[i]; }
    return eq;
  }
  const P = { x: 0, y: 0 };
  function pathAt(s) { // phone -> along the wall -> door -> route -> hub, s in 0..1
    const wl = G.R * wallArcA, tot = wl + routeLen, d = s * tot;
    if (d <= wl) { const a = HP + d / G.R; P.x = G.cx + G.R * Math.cos(a); P.y = G.cy + G.R * Math.sin(a); }
    else { const u = (d - wl) / routeLen; P.x = dX + rX * u; P.y = dY + rY * u; }
    return P;
  }
  function runner(s, al) {
    pathAt(s); ctx.globalAlpha = al; ctx.fillStyle = CREAM; ctx.fillRect(Math.round(P.x - 3), Math.round(P.y - 3), 6, 6); prims++;
  }
  function doorGlyph(shut, la, aa, dashed) { // shut 0 = open (leaf toward the centre) .. 1 = closed
    const { cx, cy, R } = G, ah = thD + gapD / 2, aj = thD - gapD / 2;
    const hX = cx + R * Math.cos(ah), hY = cy + R * Math.sin(ah), jX = cx + R * Math.cos(aj), jY = cy + R * Math.sin(aj);
    const L = Math.hypot(jX - hX, jY - hY), aOpen = Math.atan2(cy - hY, cx - hX), aShut = Math.atan2(jY - hY, jX - hX);
    const d = wrap(aShut - aOpen), aLeaf = aOpen + d * shut;
    if (dashed) ctx.setLineDash(DASH);
    ctx.globalAlpha = la; ctx.beginPath(); ctx.moveTo(hX, hY); ctx.lineTo(hX + L * Math.cos(aLeaf), hY + L * Math.sin(aLeaf)); ctx.stroke();
    if (aa > 0 && Math.abs(d * (1 - shut)) > 0.01) { ctx.globalAlpha = aa; ctx.beginPath(); ctx.arc(hX, hY, L, aLeaf, aShut, d < 0); ctx.stroke(); }
    if (dashed) ctx.setLineDash(NODASH);
    prims += 2;
  }
  // thinking: out to the hub, hold, back (an intercom goes out once and stays)
  function thinkPhase(S) {
    if (S.to) return S.lt < 0.5 ? 0 : 1;
    const u = S.lt % THINK_P;
    return u < 0.5 ? 0 : u < THINK_P - 0.5 ? 1 : 2;
  }

  function draw(now, S) {
    if (same(now, S)) return;
    force = false; prims = 0; mN = 0;
    const st = S.state, amb = S.ambient ? 0.3 : 1, grey = st === 'offline', a = A * amb, rings = !S.ambient && !REDUCED;
    const { cx, cy, R } = G;
    ctx.clearRect(0, 0, W, Hh);
    ctx.lineWidth = 1; ctx.setLineDash(NODASH); ctx.strokeStyle = CREAM;

    // the next room: its wall with a doorway where the route comes in, and the hub
    ctx.globalAlpha = a * WALL_A * (grey ? 0.5 : 0.9); ctx.beginPath();
    if (oGapA > -8) ctx.arc(ox, oy, oR, oGapA + 0.12, oGapA - 0.12 + TAU); else ctx.arc(ox, oy, oR, 0, TAU);
    ctx.stroke(); prims++;
    if (grey) ctx.setLineDash(DASH);
    ctx.globalAlpha = (grey ? 0.5 : 1) * a * 0.42; ctx.beginPath(); ctx.moveTo(dX, dY); ctx.lineTo(hx, hy); ctx.stroke(); ctx.setLineDash(NODASH); prims++;
    const tp = st === 'thinking' ? thinkPhase(S) : -1;
    const hubOn = tp === 1 || (st === 'announce' && S.lt < 0.12);
    ctx.globalAlpha = (grey ? 0.4 : 1) * amb;
    if (hubOn) { ctx.fillStyle = CREAM; ctx.fillRect(hx - 4, hy - 4, 8, 8); } else ctx.strokeRect(hx - 3.5, hy - 3.5, 7, 7);
    prims++;

    // this room's sound, clipped to its round wall
    ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.clip();
    if (rings) {
      const tR = st === 'prompt' ? entered : now, hold = st === 'prompt' ? 0.35 : 1;
      for (let i = 0; i < NR; i++) {
        const age = (tR - rBorn[i]) / 1000; if (age < 0 || age >= rLife[i]) continue;
        const f = age / rLife[i], r = 3 + rV[i] * age, al = rStr[i] * Math.pow(1 - f, 1.1) * hold;
        if (al < 0.02) continue;
        const an = rAn[i], x = AX[an], y = AY[an];
        ctx.strokeStyle = CREAM; ctx.lineWidth = 1.2 + 0.6 * rStr[i] * (1 - f);
        if (r < 2 * R + 4) { ctx.globalAlpha = al; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.stroke(); prims++; marks(ABASE[an], r, Math.min(1, al * 1.6), 0, 2 + 4 * rStr[i] * (1 - f * 0.5)); }
        if (rRef[i] && an === 0 && r > 2 * R) { ctx.globalAlpha = al * 0.35; ctx.beginPath(); ctx.arc(cx, cy - R, r - 2 * R, 0, TAU); ctx.stroke(); prims++; } // the far wall sends it back
      }
      ctx.lineWidth = 1;
      if (st === 'listening' || st === 'followup') { // your voice: ink rings close onto the phone
        const soft = st === 'followup', N = 7, lvl = soft ? 0.18 + Math.min(1, S.lv * 3) * 0.2 : Math.min(1, 0.15 + lvS * 1.4);
        ctx.strokeStyle = INK;
        for (let i = 0; i < N; i++) {
          let f = (i / N - phase) % 1; if (f < 0) f += 1;
          const al = lvl * (0.24 + 0.76 * Math.pow(1 - f, 1.5)) * (f > 0.9 ? (1 - f) * 10 : 1);
          if (al < 0.02) continue;
          const r = 4 + (2 * R - 8) * f;
          ctx.lineWidth = 1.3 + 1.1 * (1 - f) * lvl; ctx.globalAlpha = al; ctx.beginPath(); ctx.arc(phX, phY, r, 0, TAU); ctx.stroke(); prims++;
          marks(HP, r, Math.min(1, al * 1.6), 1, 3 + 4 * lvl * (1 - f * 0.5));
        }
        ctx.lineWidth = 1;
      }
      if (st === 'idle' || st === 'card') { // idle: one slow ring every 10 s, its wall points climbing the wall
        const f = (now % 10000) / 10000, al = 0.3 * Math.sin(Math.PI * Math.min(1, f * 1.25)), r = 4 + 2 * R * 0.96 * easeOut(f);
        if (al > 0.01) { ctx.strokeStyle = CREAM; ctx.globalAlpha = al * 0.7; ctx.beginPath(); ctx.arc(phX, phY, r, 0, TAU); ctx.stroke(); prims++; marks(HP, r, al * 1.4, 0, 3); }
      }
    }
    ctx.restore();

    // the wall with its door gap and jambs
    ctx.strokeStyle = CREAM;
    const wallA = a * WALL_A * (grey ? 0.5 : 1);
    ctx.globalAlpha = wallA; ctx.beginPath(); ctx.arc(cx, cy, R, thD + gapD / 2, thD - gapD / 2 + TAU); ctx.stroke(); prims++;
    if (st === 'ringing') {
      const ph = S.lt % 1.2, amp = Math.max(Math.exp(-ph / 0.12), ph >= 0.18 ? Math.exp(-(ph - 0.18) / 0.12) : 0);
      ctx.globalAlpha = amb * (0.7 + 0.3 * amp); ctx.lineWidth = 1.5 + 1.5 * amp; ctx.beginPath(); ctx.arc(cx, cy, R, thD + gapD / 2, thD - gapD / 2 + TAU); ctx.stroke(); ctx.lineWidth = 1; prims++;
    }
    const j0 = thD + gapD / 2, j1 = thD - gapD / 2;
    ctx.globalAlpha = wallA; ctx.beginPath();
    ctx.moveTo(cx + (R - 5) * Math.cos(j0), cy + (R - 5) * Math.sin(j0)); ctx.lineTo(cx + (R + 5) * Math.cos(j0), cy + (R + 5) * Math.sin(j0));
    ctx.moveTo(cx + (R - 5) * Math.cos(j1), cy + (R - 5) * Math.sin(j1)); ctx.lineTo(cx + (R + 5) * Math.cos(j1), cy + (R + 5) * Math.sin(j1));
    ctx.stroke(); prims++;
    if (grey) doorGlyph(1, 0.55 * amb, 0, true);
    else if (st === 'prompt') { ctx.lineWidth = 1.5; doorGlyph(cl(S.promptShut, 0, 1), amb, 0.5 * amb, false); ctx.lineWidth = 1; }
    for (let i = 0; i < mN; i++) { const s = MS[i]; ctx.globalAlpha = MA[i] * amb; ctx.fillStyle = MC[i] ? INK : CREAM; ctx.fillRect(Math.round((MX[i] - s / 2) * 2) / 2, Math.round((MY[i] - s / 2) * 2) / 2, s, s); }
    prims += mN;
    // the route's runner: thinking goes out and back; an announcement comes in
    if (rings && tp === 0) runner(easeOut((S.to ? S.lt : S.lt % THINK_P) / 0.5), 1);
    else if (rings && tp === 2) runner(1 - easeOut(((S.lt % THINK_P) - (THINK_P - 0.5)) / 0.5), 1);
    if (rings && st === 'announce' && S.lt < 0.45) { const u = easeOut(S.lt / 0.45), wl = R * wallArcA, tot = wl + routeLen; runner((tot - u * routeLen) / tot, amb); }
    if (grey) { const s = (now - S.blipAt) / 1000; if (s >= 0 && s < 0.5) { ctx.globalAlpha = (1 - s / 0.5) * 0.8 * amb; ctx.fillStyle = CREAM; ctx.fillRect(Math.round(dX - 3), Math.round(dY - 3), 6, 6); prims++; } }

    // the 32 dots either side of the phone, then the phone
    let col = CREAM, restA = 0.4;
    if (S.ambient) restA = 0.3;
    else if (frozen || grey) restA = 0.35;
    else if (st === 'ringing' || st === 'speaking' || st === 'announce') restA = 1;
    else if (st === 'listening' || st === 'followup') { col = INK; restA = 1; }
    else if (st === 'muted') restA = 0.25;
    ctx.fillStyle = col;
    for (let pass = 0; pass < 2; pass++) {
      ctx.globalAlpha = (pass ? 1 : restA) * amb * (frozen ? 0.6 : 1); ctx.beginPath();
      for (let i = 0; i < ND; i++) {
        const d = SZ[i];
        if ((d > 3.6) !== (pass === 1)) continue;
        const x = cx + R * Math.cos(DA[i]), y = cy + R * Math.sin(DA[i]);
        ctx.rect(Math.round((x - d / 2) * 2) / 2, Math.round((y - d / 2) * 2) / 2, d, d);
      }
      ctx.fill();
    }
    prims += ND + 2;
    const speak = st === 'speaking' || st === 'announce' || st === 'ringing', listen = st === 'listening' || st === 'followup', flat = st === 'muted' || grey;
    ctx.globalAlpha = Math.max(a, 0.35 * amb); ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(phX - 12, phY - 3, 24, 6, 1); else ctx.rect(phX - 12, phY - 3, 24, 6);
    if (flat) { ctx.fillStyle = S.ambient ? AMB_GROUND : TOMATO; ctx.fill(); ctx.strokeStyle = CREAM; ctx.stroke(); }
    else { ctx.fillStyle = listen ? INK : CREAM; if (!speak && !listen) ctx.globalAlpha *= 0.9; ctx.fill(); }
    prims += 2;
    ctx.globalAlpha = 1;
  }

  return {
    resize, update, draw, onState,
    prims: () => prims,
    invalidate() { force = true; },
    clearRings() { rBorn.fill(-1e12); force = true; },
  };
}
