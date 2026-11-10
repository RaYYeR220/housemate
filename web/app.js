// Housemate satellite: first run (name -> pairing code -> room -> tap to start), the hub connection
// (hello with the device key, reconnect with backoff), mic frames out, playout in, and the state machine
// that turns hub messages into what the face shows.
import { Player } from './player.js';
import { createFace, REDUCED } from './face.js';
import * as ui from './ui.js';
import { promptView } from './prompt.js';

const Q = new URLSearchParams(location.search);
const PERF = Q.get('perf') === '1';
const FIXED_CLOCK = /^\d{1,2}:\d{2}$/.test(Q.get('clock') || '') ? Q.get('clock') : '';
const root = document.documentElement;
const el = ui.el;
const BACKOFF = [1, 2, 4, 8, 16, 30]; // seconds between reconnect attempts
const AMBIENT_AFTER = 180000; // quiet this long -> the dark ambient face
const SHIFT_EVERY = 180000; // burn-in guard: move the whole face by a pixel or two
const SHIFTS = [[0, 0], [2, 1], [-1, 2], [-2, -1], [1, -2]];
const STATE_ID = { idle: 0, listening: 1, thinking: 2, speaking: 3, followup: 4, announce: 5, prompt: 6, card: 7, ringing: 8, muted: 9, offline: 10 };
const HUB_STATES = ['idle', 'listening', 'thinking', 'speaking', 'ringing'];
const DENY_WAIT = 600000; // after "Deny" on the PC, ask again only after 10 minutes

// ---------- storage (private windows can throw) ----------
const store = {
  get(k) { try { return localStorage.getItem('housemate.' + k); } catch (_) { return null; } },
  set(k, v) { try { if (v == null) localStorage.removeItem('housemate.' + k); else localStorage.setItem('housemate.' + k, String(v)); } catch (_) { /* not kept */ } },
};
function deviceId() {
  let id = store.get('device_id');
  if (!id || !/^[A-Za-z0-9_.:-]{4,128}$/.test(id)) {
    const b = new Uint8Array(8); crypto.getRandomValues(b);
    id = 'sat-' + Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
    store.set('device_id', id);
  }
  return id;
}

// ---------- state ----------
const H = { // the hub and this device
  id: deviceId(), name: store.get('name') || '', key: store.get('device_key'), wantRoom: store.get('want_room') || '',
  deniedUntil: Number(store.get('denied_until')) || 0,
  ws: null, trusted: false, everTrusted: false, attempt: 0, retryAt: 0, timer: 0, parked: false,
  hubState: 'idle', prevHubState: 'idle', followup: false, room: store.get('room'), roomName: store.get('room_name') || '',
  rooms: [], wake: store.get('wake') || '', conversation: store.get('conversation') || '', hubName: store.get('hub_name') || '',
  streamKind: '', streamEndedAt: -1e9, abortAt: -1e9, stopUntil: 0, intercomTo: '', intercomSlug: '', intercomAt: -1e9, pickReturn: '',
};
const A = { ctx: null, player: null, node: null, stream: null, chime: null, started: false, micRms: 0, lock: null, frames: 0, dropped: 0 };
const S = { state: '', lt: 0, lv: 0, ambient: false, to: '', promptShut: 0, blipAt: -1e9, key: 0, enteredAt: 0, turnAt: -1e9 };
const CAP = { you: '', youAt: -1e9, say: '', sayAt: -1e9, label: '', notice: '', noticeAt: -1e9, words: 0 };
const timers = [];
const prompts = new Map();
let muted = store.get('muted') === '1';
let held = false;
let card = null, cardHidden = false;
let lastActivity = performance.now();
const LOG = []; // last control messages (types only), for checks from the dev tools
function log(dir, m) { LOG.push([Math.round(performance.now()), dir, m.type, m.stream_id || m.state || m.kind || '']); if (LOG.length > 300) LOG.shift(); }
window.__hm = { S, H, A, CAP, LOG, timers, prompts };

const face = createFace(el.canvas, el.inner, el.micoff, el.door);

// ---------- the hub connection ----------
function send(msg) {
  const ws = H.ws;
  if (!ws || ws.readyState !== 1) return false;
  ws.send(JSON.stringify(msg)); log('>', msg);
  return true;
}

function connect() {
  clearTimeout(H.timer); H.timer = 0;
  if (H.ws || H.parked || !H.name) return;
  const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
  let ws;
  try { ws = new WebSocket(url); } catch (_) { scheduleReconnect(); return; }
  ws.binaryType = 'arraybuffer';
  H.ws = ws; S.blipAt = performance.now();
  ws.onopen = () => {
    const hello = { type: 'hello', device_id: H.id, name: H.name.slice(0, 64), ua: navigator.userAgent.slice(0, 512) };
    if (H.key) hello.device_key = H.key;
    else if (H.wantRoom) hello.room = H.wantRoom.slice(0, 64); // shown on the PC next to the pairing code
    send(hello);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') onControl(ev.data);
    else if (A.player) { const u = new Uint8Array(ev.data); if (u[0] === 2) A.player.push(u.subarray(1)); }
  };
  ws.onclose = (ev) => onClose(ws, ev.code);
}

function scheduleReconnect(delayS) {
  clearTimeout(H.timer);
  const d = delayS != null ? delayS : BACKOFF[Math.min(H.attempt, BACKOFF.length - 1)];
  H.attempt++;
  H.retryAt = performance.now() + d * 1000;
  H.timer = setTimeout(connect, d * 1000);
}

function onClose(ws, code) {
  if (H.ws !== ws) return;
  H.ws = null; H.trusted = false;
  if (A.player) A.player.abort();
  held = false; el.talk.classList.remove('held');
  ui.setConnected(false);
  if (code === 4401) { // the hub forgot this device (revoked, or a new hub): pair again
    H.key = null; store.set('device_key', null); H.attempt = 0;
    scheduleReconnect(0.3); return;
  }
  if (code === 4409) { // the same device connected from another tab
    H.parked = true;
    ui.showError('Housemate', 'Open somewhere else', 'This speaker is now running in another tab or window.', 'Use it here', () => {
      H.parked = false; H.attempt = 0; route(); connect();
    });
    return;
  }
  if (code === 4403) { // the owner declined this pairing on the PC: ask again only later, by hand
    if (H.deniedUntil < Date.now()) { H.deniedUntil = Date.now() + DENY_WAIT; store.set('denied_until', H.deniedUntil); }
    showDenied();
    return;
  }
  if (code === 4408 && ui.currentScreen() === 'pair') { H.attempt = 0; scheduleReconnect(0.5); return; }
  if (code === 4429) H.attempt = BACKOFF.length - 1;
  if (!H.everTrusted && !H.key && ui.currentScreen() !== 'setup') ui.showScreen('connect');
  el.connectDetail.textContent = "Can't reach the hub at " + location.host + '. Retrying.';
  scheduleReconnect();
}

function onControl(text) {
  let m;
  try { m = JSON.parse(text); } catch (_) { return; }
  if (!m || typeof m.type !== 'string') return;
  log('<', m);
  const now = performance.now();
  switch (m.type) {
    case 'pair.code':
      H.trusted = false;
      ui.showPairCode(String(m.code || ''));
      H.codeUntil = now + 600000;
      ui.showScreen('pair');
      break;
    case 'paired':
      H.trusted = true; H.everTrusted = true; H.attempt = 0;
      H.key = m.device_key; store.set('device_key', m.device_key);
      H.rooms = Array.isArray(m.rooms) ? m.rooms.filter((r) => r && r.slug && r.name) : [];
      store.set('rooms', JSON.stringify(H.rooms.map((r) => String(r.name)).slice(0, 12)));
      setRoom(m.room || null, m.room_name || '');
      if (m.wake) { H.wake = String(m.wake); store.set('wake', H.wake); }
      if (m.conversation) { H.conversation = String(m.conversation); store.set('conversation', H.conversation); }
      if (m.hub_name) { H.hubName = String(m.hub_name); store.set('hub_name', H.hubName); }
      if (!H.room && H.wantRoom && send({ type: 'room.set', room: H.wantRoom })) setRoom(slugOf(H.wantRoom), H.wantRoom);
      H.deniedUntil = 0; store.set('denied_until', null);
      H.hubState = 'idle'; H.followup = false;
      prompts.clear(); // the hub sends the room's open prompts again right after this
      ui.setConnected(true);
      updateDoor();
      if (muted) send({ type: 'mute', on: true });
      route();
      break;
    case 'state':
      if (m.room) setRoom(m.room, m.room_name || '');
      if (HUB_STATES.includes(m.state)) onHubState(m.state, now);
      break;
    case 'caption': onCaption(m, now); break;
    case 'card':
      card = m; cardHidden = false; ui.renderCard(m); activity(now);
      break;
    case 'timers': setTimers(Array.isArray(m.items) ? m.items : [], now); break;
    case 'prompt':
      prompts.delete(m.id); prompts.set(m.id, Object.assign({ at: now }, m));
      showPrompt(); activity(now);
      break;
    case 'prompt.clear':
      if (prompts.delete(m.id)) showPrompt();
      break;
    case 'chime': playChime(m.kind); activity(now); break;
    case 'play.begin':
      H.streamKind = String(m.stream_id).split('-')[0];
      if (A.player) A.player.begin(m.stream_id, m.sample_rate | 0);
      activity(now);
      break;
    case 'play.end':
      if (A.player && A.player.sid === m.stream_id) A.player.end(m.stream_id);
      else if (!A.player) send({ type: 'playback.done', stream_id: m.stream_id }); // nothing can play here yet
      break;
    case 'play.abort':
      if (A.player) A.player.abort(m.stream_id);
      H.abortAt = now;
      break;
    case 'error':
      if (m.code === 'pairing_denied') { H.deniedUntil = Date.now() + DENY_WAIT; store.set('denied_until', H.deniedUntil); }
      else if (m.code === 'pairing_expired') ui.showPairCode('······');
      else if (m.code === 'rate_limited') H.attempt = BACKOFF.length - 1;
      else if (m.code === 'bad_room') notice('That room name needs letters or digits', now);
      break;
  }
}

function onHubState(state, now) {
  const prev = H.hubState;
  if (state === 'listening' && H.intercomTo && now - H.intercomAt > 2500) H.intercomTo = ''; // not from the tap
  if (state === 'listening' && prev !== 'listening') {
    // listening right after an answer (no barge-in) is the follow-up window; after a wake it is a new turn
    H.followup = !held && !H.intercomTo && (prev === 'speaking' || H.streamEndedAt > now - 1500) && now - H.abortAt > 400;
  }
  if (prev === 'speaking' && state !== 'speaking') H.streamEndedAt = now;
  H.prevHubState = prev; H.hubState = state;
  if (state !== 'ringing') H.stopUntil = 0;
  if (state === 'idle' && H.intercomTo && now - H.intercomAt > 1500) H.intercomTo = '';
  if (state !== 'idle') activity(now);
}

const slugOf = (name) => String(name).trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
function setRoom(slug, name) {
  const known = slug ? H.rooms.find((r) => r.slug === slug) : null;
  H.room = slug; H.roomName = name || (known && known.name) || (slug ? cap1(slug.replace(/-/g, ' ')) : '');
  if (slug && !known) H.rooms.push({ slug, name: H.roomName }); // a room made on the PC (Rooms page)
  store.set('room', slug); store.set('room_name', H.roomName || null);
  ui.setRoom(H.roomName);
  updateDoor();
  if (slug && ui.currentScreen() === 'room') route();
}

// the other room in the corner: the first room that is not this one (tap it to talk to it)
function updateDoor() {
  const other = H.rooms.find((r) => r.slug !== H.room);
  H.intercomSlug = other ? other.slug : '';
  ui.setDoor(other ? other.name : H.hubName || 'Hub', !!other);
}

function showDenied() {
  H.parked = true;
  clearTimeout(H.timer);
  const left = H.deniedUntil - Date.now();
  ui.showError('Pairing', 'Pairing declined', left > 0
    ? 'This speaker was not approved on the PC. You can ask again in ' + Math.ceil(left / 60000) + ' min.'
    : 'This speaker was not approved on the PC. You can ask again now.', 'Ask again', () => {
    if (H.deniedUntil > Date.now()) { showDenied(); return; }
    H.deniedUntil = 0; store.set('denied_until', null);
    H.parked = false; H.attempt = 0;
    ui.showScreen('connect'); connect();
  });
}

// which screen: name -> code -> room -> tap to start -> face
function route() {
  if (H.deniedUntil > Date.now() && !H.key) { showDenied(); return; }
  if (H.parked) return;
  if (!H.name) { ui.showScreen('setup'); return; }
  if (!H.trusted) {
    if (H.everTrusted || H.key) { ui.showScreen(A.started ? 'face' : 'start'); return; }
    if (ui.currentScreen() !== 'pair') ui.showScreen('connect');
    return;
  }
  if (!H.room || H.pickReturn) { openRoomPicker(); return; }
  ui.showScreen(A.started ? 'face' : 'start');
}

function openRoomPicker() {
  ui.renderRooms(H.rooms, H.room, pickRoom);
  el.roomNew.value = '';
  ui.showScreen('room');
}
function pickRoom(name) {
  name = String(name || '').trim().slice(0, 64);
  if (!name) return;
  if (!send({ type: 'room.set', room: name })) return;
  H.pickReturn = '';
  if (!H.rooms.some((r) => r.name.toLowerCase() === name.toLowerCase())) H.rooms.push({ slug: slugOf(name), name });
  ui.showScreen(A.started ? 'face' : 'start');
}

// ---------- captions, notices, timers, prompts ----------
function activity(now) {
  lastActivity = now;
  if (S.ambient) setAmbient(false);
}
function notice(text, now) { CAP.notice = text; CAP.noticeAt = now; }

function onCaption(m, now) {
  const text = String(m.text || '').trim();
  if (!text) return;
  switch (m.role) {
    case 'user': CAP.you = text; CAP.youAt = now; break;
    case 'status': notice(text.replace(/\.$/, ''), now); break;
    case 'announcement': CAP.say = text; CAP.sayAt = now; CAP.label = 'Announcement'; break;
    case 'intercom': CAP.say = 'Voice message'; CAP.sayAt = now; CAP.label = 'From the ' + text; break;
    default: CAP.say = text; CAP.sayAt = now; CAP.label = '';
  }
  CAP.words = CAP.say.split(/\s+/).filter(Boolean).length;
  activity(now);
}

const cap1 = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
function setTimers(items, now) {
  timers.length = 0;
  for (const t of items) {
    if (!t || typeof t !== 'object') continue;
    const alarm = t.kind === 'alarm';
    timers.push({
      id: String(t.id || ''), alarm, state: String(t.state || 'pending'), missed: !!t.missed,
      label: cap1(String(t.label || '').trim()) || (alarm ? 'Alarm' : 'Timer'),
      end: now + (Number(t.remaining_s) || 0) * 1000, duration: Number(t.duration_s) || 0, due: Number(t.due_at) || 0,
    });
  }
}
function ringingTimer() { return timers.find((t) => t.state === 'ringing') || null; }
function pendingTimers(now) {
  const out = [];
  for (const t of timers) if (t.state === 'pending') out.push({ label: t.label, left: (t.end - now) / 1000 });
  return out.sort((a, b) => a.left - b.left);
}
function ringInfo() {
  const t = ringingTimer();
  if (!t) return { label: 'Timer', sub: 'done' };
  let sub = t.alarm ? new Date(t.due * 1000).toTimeString().slice(0, 5) + ' alarm' : (t.duration ? ui.mmss(t.duration) + ' · done' : 'done');
  if (t.missed) sub = 'missed while offline';
  return { label: t.label, sub };
}

function activePrompt() { let p = null; for (const v of prompts.values()) p = v; return p; }
function showPrompt() {
  const p = activePrompt();
  if (!p) return;
  ui.renderPrompt(promptView(p), (value) => {
    if (!send({ type: 'ui.action', kind: 'prompt', id: p.id, value })) notice("Can't reach the hub", performance.now());
  });
}

// ---------- audio: unlock, mic, playout, chime ----------
async function startAudio() {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) { ui.showError('Housemate', 'This browser has no audio', 'Open Housemate in Chrome, Edge, Firefox or Safari.', 'Try again', () => route()); return false; }
  if (!A.ctx) A.ctx = new AC({ latencyHint: 'interactive' });
  try { // inside the tap: a one-sample silent buffer unlocks output on iOS
    const b = A.ctx.createBuffer(1, 1, 22050), s = A.ctx.createBufferSource();
    s.buffer = b; s.connect(A.ctx.destination); s.start(0);
  } catch (_) { /* fine */ }
  const resumed = A.ctx.resume().catch(() => {});
  if (!A.player) A.player = new Player(A.ctx, (sid) => send({ type: 'playback.done', stream_id: sid }));
  loadChime();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.isSecureContext) {
    ui.showError('Microphone', 'Open this page over https', 'Browsers only allow the microphone on a secure page. Use the https:// address the hub printed.', 'Try again', () => route());
    return false;
  }
  try {
    A.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
  } catch (e) {
    const denied = e && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
    ui.showError('Microphone', denied ? 'The microphone is blocked' : 'No microphone found',
      denied ? "Allow the microphone for this page in the browser's site settings, then try again." : 'Connect a microphone, then try again.',
      'Try again', () => ui.showScreen('start'));
    return false;
  }
  try {
    await resumed;
    await A.ctx.audioWorklet.addModule('mic-worklet.js');
    A.node = new AudioWorkletNode(A.ctx, 'pcm16k', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: 'explicit' });
    A.node.port.onmessage = (e) => onMicFrame(e.data);
    const src = A.ctx.createMediaStreamSource(A.stream), sink = A.ctx.createGain();
    sink.gain.value = 0;
    src.connect(A.node); A.node.connect(sink); sink.connect(A.ctx.destination);
  } catch (e) {
    ui.showError('Microphone', 'Audio could not start', String((e && e.message) || e).slice(0, 200), 'Try again', () => ui.showScreen('start'));
    return false;
  }
  for (const tr of A.stream.getAudioTracks()) tr.addEventListener('ended', () => { A.started = false; if (A.node) A.node.port.onmessage = null; route(); });
  A.started = true;
  wakeLock();
  return true;
}

function onMicFrame(d) {
  A.micRms = d.rms;
  const ws = H.ws;
  if (!H.trusted || !H.room || muted || !ws || ws.readyState !== 1) return;
  if (ws.bufferedAmount > 32000) { A.dropped++; return; } // a stalled link: drop, never queue minutes of audio
  const out = new Uint8Array(641);
  out[0] = 1; out.set(new Uint8Array(d.pcm), 1);
  ws.send(out); A.frames++;
}
const micLevel = () => Math.max(0, Math.min(1, (20 * Math.log10(A.micRms + 1e-9) + 55) / 40));

function loadChime() {
  if (A.chime || !A.ctx) return;
  A.chime = 'loading';
  fetch('chime.wav').then((r) => r.arrayBuffer()).then((b) => new Promise((ok, no) => A.ctx.decodeAudioData(b, ok, no)))
    .then((buf) => { A.chime = buf; }).catch(() => { A.chime = null; });
}
function playChime(kind) {
  if (!A.ctx || !A.chime || typeof A.chime === 'string') return;
  const s = A.ctx.createBufferSource(), g = A.ctx.createGain();
  s.buffer = A.chime; s.playbackRate.value = kind === 'announce' ? 1.12 : kind === 'prompt' ? 0.94 : 1;
  g.gain.value = 0.8; s.connect(g); g.connect(A.ctx.destination); s.start();
}

async function wakeLock() {
  if (!('wakeLock' in navigator) || document.hidden) return;
  try { A.lock = await navigator.wakeLock.request('screen'); } catch (_) { A.lock = null; }
}

// ---------- what the face shows ----------
function visualState(now) {
  if (!H.trusted) return 'offline';
  const hs = H.hubState;
  if (prompts.size) return 'prompt';
  if ((hs === 'ringing' || (hs === 'speaking' && ringingTimer())) && now > H.stopUntil) return 'ringing';
  if (hs === 'speaking') return H.streamKind === 'ann' ? 'announce' : 'speaking';
  if (held) return 'listening';
  if (hs === 'listening') return H.followup ? 'followup' : 'listening';
  if (hs === 'thinking') return 'thinking';
  if (muted) return 'muted';
  if (card && !cardHidden) return 'card';
  return 'idle';
}

let bodyCls = '';
function enterState(st, now) {
  const prev = S.state;
  S.state = st; S.enteredAt = now; S.key++;
  if (st === 'listening' || st === 'followup') S.turnAt = now;
  const cls = 's-' + st;
  if (cls !== bodyCls) { if (bodyCls) root.classList.remove(bodyCls); root.classList.add(cls); bodyCls = cls; }
  if (st !== 'idle' && st !== 'card' && st !== 'muted') activity(now);
  face.onState(st, now, prev);
  el.talk.disabled = st === 'offline' || muted;
  capKey = '';
}

function compute(now) {
  const st = visualState(now);
  if (st !== S.state) enterState(st, now);
  S.lt = (now - S.enteredAt) / 1000;
  S.to = H.intercomTo && (st === 'listening' || st === 'thinking') ? H.intercomTo : '';
  S.lv = st === 'listening' || st === 'followup' ? micLevel()
    : (st === 'speaking' || st === 'announce') && A.player ? A.player.level() : 0;
  const p = st === 'prompt' ? activePrompt() : null;
  S.promptShut = p ? Math.min(1, (now - p.at) / 300000) : 0; // the door swings shut over Hermes's default 5 min
}

// caption model -> ui (rebuilt only when it changes)
const CM = { mode: 'none', text: '', on: -1, label: '' };
let capKey = '', capOn = -2;
function captionModel(st, now) {
  CM.on = -1; CM.label = ''; CM.mode = 'none'; CM.text = '';
  const freshYou = CAP.youAt >= S.turnAt - 200;
  if (st === 'listening') {
    if (!held && freshYou) { CM.mode = 'you'; CM.text = CAP.you; } else { CM.mode = 'live'; CM.text = 'Listening…'; }
    if (S.to) CM.label = 'To the ' + S.to;
  } else if (st === 'followup') {
    if (CAP.say && now - CAP.sayAt < 60000) { CM.mode = 'say'; CM.text = CAP.say; CM.on = 999; CM.label = CAP.label; }
  } else if (st === 'thinking') {
    if (freshYou) { CM.mode = 'you'; CM.text = CAP.you; }
    if (S.to) CM.label = 'To the ' + S.to;
  } else if (st === 'speaking' || st === 'announce') {
    if (CAP.say && (CAP.sayAt >= S.turnAt - 200 || (st === 'announce' && now - CAP.sayAt < 30000))) {
      CM.mode = 'say'; CM.text = CAP.say; CM.label = CAP.label;
      let target = CAP.words;
      if (A.player && A.player.active && st === 'speaking') target = Math.min(target, 1 + Math.floor(A.player.played() / 0.32));
      CM.on = Math.min(target, 1 + Math.floor((now - CAP.sayAt) / 60));
    } else if (freshYou) { CM.mode = 'you'; CM.text = CAP.you; }
  } else if (st === 'idle' && CAP.say && now - CAP.sayAt < 120000) { CM.mode = 'dim'; CM.text = CAP.say; CM.on = 999; CM.label = CAP.label; }
}
function renderCaption(now) {
  captionModel(S.state, now);
  const key = CM.mode + '\u0001' + CM.text + '\u0001' + CM.label;
  if (key !== capKey) { capKey = key; capOn = CM.on; ui.renderCaption(CM, true); }
  else if (CM.on !== capOn) { capOn = CM.on; ui.renderCaption(CM, false); }
}

function proof(st, now) {
  const mode = H.conversation === 'per_room' ? 'Room by room' : 'Household';
  const hub = H.hubName ? ' · hub ' + H.hubName : '';
  const quiet = st === 'idle' && capKey.startsWith('none') && !pendingTimers(now).length;
  return (quiet ? '' : wakeText() + ' · ') + mode + hub;
}
const wakeText = () => (H.wake ? 'Say "' + H.wake + '"' : 'Say the wake word');
function statusText(st, now) {
  if (CAP.notice && now - CAP.noticeAt < 4000 && st !== 'ringing') return [CAP.notice, true];
  switch (st) {
    case 'thinking': return [S.to ? 'Sending to the ' + S.to : 'Working on it'];
    case 'followup': return ['Still listening'];
    case 'listening': return [S.to ? 'Speak now · it plays in the ' + S.to : held ? 'Listening · let go to send' : proof(st, now)];
    case 'ringing': return ['Say "stop" or tap anywhere'];
    case 'muted': return ['Mic off · tap the mic to listen again'];
    case 'offline': {
      if (!H.ws && H.retryAt > now + 50) return ["Can't reach the hub · retrying in " + Math.ceil((H.retryAt - now) / 1000) + ' s'];
      return [H.everTrusted ? 'Reconnecting to the hub' : 'Connecting to the hub'];
    }
    default: return [proof(st, now)];
  }
}

// ---------- frame loop (capped rate, perf ring) ----------
const capFor = (st) => (st === 'listening' || st === 'speaking' || st === 'ringing' || st === 'announce' || st === 'followup') ? 60 : st === 'offline' ? 10 : 30;
const FT = new Float32Array(300), FS = new Float32Array(300);
let ftI = 0, ftN = 0, raf = 0, lastTick = 0, prevRender = 0, fpsN = 0, fpsT = 0, fps = 0, hudT = 0, secT = 0, cap = 30;
const perf = { avg: 0, p95: 0, fps: 0, cap: 30, frames: 0, state: '' };
window.__perf = perf;
window.__perfReset = () => { ftN = 0; ftI = 0; };

function frame(now) {
  raf = requestAnimationFrame(frame);
  cap = REDUCED ? 4 : capFor(S.state);
  const interval = 1000 / cap, elapsed = now - lastTick;
  if (elapsed < interval - 1.5) return;
  lastTick = elapsed > interval * 2 ? now : lastTick + interval;
  const dt = Math.min(0.1, (now - (prevRender || now)) / 1000); prevRender = now;
  const a = performance.now();
  compute(now);
  face.update(dt, now, S);
  face.draw(now, S);
  renderCaption(now);
  if (now - secT >= 250) { secT = now; slowTick(now); }
  const ms = performance.now() - a;
  FT[ftI] = ms; ftI = (ftI + 1) % 300; if (ftN < 300) ftN++;
  fpsN++; if (now - fpsT >= 1000) { fps = fpsN * 1000 / (now - fpsT); fpsN = 0; fpsT = now; }
  if (now - hudT >= 500) { hudT = now; perfStats(); }
}
function perfStats() {
  let s = 0; for (let i = 0; i < ftN; i++) { s += FT[i]; FS[i] = FT[i]; }
  const sub = FS.subarray(0, ftN); sub.sort();
  perf.avg = ftN ? s / ftN : 0; perf.p95 = ftN ? sub[Math.min(ftN - 1, Math.floor(ftN * 0.95))] : 0;
  perf.fps = fps; perf.cap = cap; perf.frames = ftN; perf.state = S.state;
  if (PERF) el.hud.textContent = 'js ' + perf.avg.toFixed(2) + ' ms avg · ' + perf.p95.toFixed(2) + ' p95\n' + fps.toFixed(0) + ' fps · cap ' + cap + ' · ' + S.state + ' · ' + face.prims() + ' prims';
}
function loop(on) {
  if (on && !raf && !document.hidden) { lastTick = 0; prevRender = 0; raf = requestAnimationFrame(frame); }
  else if (!on && raf) { cancelAnimationFrame(raf); raf = 0; }
}

let shiftT = performance.now(), shiftK = 0;
function slowTick(now) {
  ui.renderClock(FIXED_CLOCK);
  ui.setHint(wakeText());
  const st = S.state;
  ui.renderTimers(pendingTimers(now), st === 'ringing' ? ringInfo() : null);
  const [text, badge] = statusText(st, now);
  ui.setStatus(text, badge);
  if (H.intercomTo && now - H.intercomAt > 30000) H.intercomTo = '';
  if (!S.ambient && (st === 'idle' || st === 'card') && now - lastActivity > AMBIENT_AFTER) setAmbient(true);
  if (now - shiftT > SHIFT_EVERY) {
    shiftT = now; shiftK = (shiftK + 1) % SHIFTS.length;
    el.face.style.transform = 'translate(' + SHIFTS[shiftK][0] + 'px,' + SHIFTS[shiftK][1] + 'px)';
    face.invalidate();
  }
}
function setAmbient(on) {
  S.ambient = on; root.classList.toggle('ambient', on); face.invalidate();
  const meta = document.querySelector('meta[name=theme-color]'); if (meta) meta.content = on ? '#2a1410' : '#c4452a';
}

// ---------- input ----------
function holdBegin(e) {
  if (held || !H.trusted || muted || S.state === 'offline' || S.state === 'prompt') return;
  if (e && e.preventDefault) e.preventDefault();
  if (A.ctx && A.ctx.state !== 'running') A.ctx.resume().catch(() => {});
  held = true; el.talk.classList.add('held'); H.intercomTo = '';
  if (A.player) A.player.abort(); // barge-in: stop talking at once, the hub aborts the stream too
  send({ type: 'ptt', on: true });
  activity(performance.now());
}
function holdEnd() {
  if (!held) return;
  held = false; el.talk.classList.remove('held');
  send({ type: 'ptt', on: false });
}
el.talk.addEventListener('pointerdown', (e) => { try { el.talk.setPointerCapture(e.pointerId); } catch (_) { /* fine */ } holdBegin(e); });
el.talk.addEventListener('pointerup', holdEnd);
el.talk.addEventListener('pointercancel', holdEnd);
el.talk.addEventListener('lostpointercapture', holdEnd);
el.talk.addEventListener('contextmenu', (e) => e.preventDefault());
el.talk.addEventListener('keydown', (e) => { if ((e.key === ' ' || e.key === 'Enter') && !e.repeat) holdBegin(e); });
el.talk.addEventListener('keyup', (e) => { if (e.key === ' ' || e.key === 'Enter') holdEnd(); });

el.mute.setAttribute('aria-pressed', muted ? 'true' : 'false');
el.mute.addEventListener('click', () => {
  muted = !muted; store.set('muted', muted ? '1' : '0');
  el.mute.setAttribute('aria-pressed', muted ? 'true' : 'false');
  el.mute.setAttribute('aria-label', muted ? 'Unmute the microphone' : 'Mute the microphone');
  if (muted) holdEnd();
  el.talk.disabled = muted || S.state === 'offline';
  send({ type: 'mute', on: muted });
  activity(performance.now());
});

el.door.addEventListener('click', () => {
  const now = performance.now();
  activity(now);
  const st = S.state;
  if (!H.intercomSlug) return notice('Add a speaker in another room to talk to it', now);
  if (st === 'offline') return notice("Can't reach the hub", now);
  if (muted) return notice('Mic off', now);
  if (st === 'prompt' || st === 'ringing' || st === 'thinking') return notice('Busy right now', now);
  const name = el.door.textContent;
  if (send({ type: 'ui.action', kind: 'intercom', id: H.intercomSlug, value: name })) {
    if (A.player) A.player.abort();
    H.intercomTo = name; H.intercomAt = now;
  }
});

el.cardClose.addEventListener('click', () => { cardHidden = true; activity(performance.now()); });
el.rname.addEventListener('click', () => { if (H.trusted) { H.pickReturn = 'face'; openRoomPicker(); } });

document.addEventListener('pointerdown', (e) => {
  const now = performance.now();
  activity(now);
  if (A.ctx && A.ctx.state !== 'running' && A.started) A.ctx.resume().catch(() => {});
  if (S.state === 'ringing' && ui.currentScreen() === 'face' && !e.target.closest('.ctl')) {
    e.stopPropagation();
    send({ type: 'ui.action', kind: 'stop_timer', id: '' });
    H.stopUntil = now + 4000;
    swallowClick = true; // the tap stops the ring and does nothing else
  }
}, true);
let swallowClick = false;
document.addEventListener('click', (e) => { if (swallowClick) { swallowClick = false; e.stopPropagation(); e.preventDefault(); } }, true);
document.addEventListener('pointerup', () => setTimeout(() => { swallowClick = false; }, 400), true);

el.setup.addEventListener('submit', (e) => {
  e.preventDefault();
  const name = el.name.value.trim().slice(0, 64);
  if (!name) return;
  H.name = name; store.set('name', name);
  H.wantRoom = el.wantRoom.value.trim().slice(0, 64); store.set('want_room', H.wantRoom || null);
  ui.showScreen('connect'); el.connectDetail.textContent = '';
  connect();
});
el.roomForm.addEventListener('submit', (e) => { e.preventDefault(); pickRoom(el.roomNew.value); });
el.start.addEventListener('click', async () => {
  el.start.disabled = true;
  const ok = await startAudio();
  el.start.disabled = false;
  if (ok) { ui.showScreen('face'); start(); }
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) { loop(false); holdEnd(); return; }
  if (A.ctx && A.started) A.ctx.resume().catch(() => {});
  if (A.started) wakeLock();
  if (!H.ws && H.name && !H.parked) { H.attempt = 0; connect(); }
  if (ui.currentScreen() === 'face') loop(true);
});
addEventListener('online', () => { if (!H.ws && H.name && !H.parked) { H.attempt = 0; connect(); } });
addEventListener('resize', () => { face.resize(); capKey = ''; });

// ---------- boot ----------
function start() {
  face.resize();
  const ready = () => { face.resize(); root.classList.add('ready'); capKey = ''; };
  (document.fonts && document.fonts.ready ? Promise.race([document.fonts.ready, new Promise((r) => setTimeout(r, 1500))]) : Promise.resolve()).then(ready);
  loop(true);
}
if (PERF) el.hud.hidden = false;
ui.setRoom(H.roomName);
updateDoor();
el.name.value = H.name || (/iPhone|Android.*Mobile/.test(navigator.userAgent) ? 'Phone' : /iPad|Android|Tablet/.test(navigator.userAgent) ? 'Tablet' : 'Laptop');
el.wantRoom.value = H.wantRoom;
let seen = [];
try { seen = JSON.parse(store.get('rooms') || '[]'); } catch (_) { /* not kept */ }
const known = [...new Set([H.roomName, ...(Array.isArray(seen) ? seen : []), 'Kitchen', 'Living room', 'Bedroom', 'Office'].filter((n) => typeof n === 'string' && n))].slice(0, 6);
ui.renderRooms(known.map((n) => ({ slug: '', name: n })), '', (n) => { el.wantRoom.value = n; }, el.roomChips);
route();
if (H.name && !H.parked) connect();
setInterval(() => { // the pairing code's time left
  if (ui.currentScreen() !== 'pair' || !H.codeUntil) return;
  const left = (H.codeUntil - performance.now()) / 1000;
  el.codeLeft.textContent = left > 0 ? 'The code works for ' + ui.mmss(left) + ' more.' : 'Getting a new code…';
}, 1000);
