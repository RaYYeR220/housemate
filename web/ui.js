// DOM side of the satellite: screens, the clock inside the room, timers, captions, status line,
// the approval / question card and content cards. Everything here is driven by app.js.

const $ = (id) => document.getElementById(id);
const root = document.documentElement;

export const el = {
  face: $('face'), canvas: $('obj'), inner: $('inner'), micoff: $('micoff'), door: $('door'),
  rname: $('rname'), room: $('room-name'), digits: $('t-digits'), date: $('date'),
  tbLabel: $('tb-label'), tbDigits: $('tb-digits'), tbMore: $('tb-more'), rbLabel: $('rb-label'), rbSub: $('rb-sub'),
  capwrap: $('capwrap'), capLabel: $('cap-label'), caption: $('caption'), hint: $('hint'), status: $('status'),
  pTitle: $('p-title'), pDetails: $('p-details'), pBtns: $('p-btns'), pFoot: $('p-foot'),
  cTitle: $('c-title'), cBody: $('c-body'), cardClose: $('card-close'),
  mute: $('mute'), talk: $('talk'), hud: $('hud'),
  name: $('name'), wantRoom: $('want-room'), roomChips: $('room-chips'), setup: $('s-setup'), code: $('code'), codeCmd: $('code-cmd'), codeLeft: $('code-left'),
  roomForm: $('s-room'), rooms: $('rooms'), roomNew: $('room-new'), start: $('start'), startText: $('start-text'),
  connectDetail: $('connect-detail'), errKicker: $('err-kicker'), errTitle: $('err-title'), errText: $('err-text'), errBtn: $('err-btn'),
};

let screen = '';
export function showScreen(name) {
  if (name === screen) return;
  if (screen) root.classList.remove('scr-' + screen);
  screen = name;
  root.classList.add('scr-' + name);
  const first = { setup: el.name, room: el.rooms.querySelector('button'), start: el.start, error: el.errBtn }[name];
  if (first) setTimeout(() => { try { first.focus({ preventScroll: true }); } catch (_) { /* gone */ } }, 50);
}
export const currentScreen = () => screen;

export const mmss = (s) => {
  s = Math.max(0, Math.ceil(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + String(r).padStart(2, '0');
};
const slots = (str) => str.replace(/[0-9]/g, '<span class="d">$&</span>').replace(/:/g, '<span class="c">:</span>');
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };

let lastClock = '', lastMinute = -1;
const DATE_FMT = new Intl.DateTimeFormat(undefined, { weekday: 'long', day: 'numeric', month: 'long' });
export function renderClock(fixed) {
  const d = new Date(), minute = d.getHours() * 60 + d.getMinutes();
  if (minute === lastMinute) return; // the clock and the date only change once a minute
  lastMinute = minute;
  const hm = fixed || String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  if (hm !== lastClock) { lastClock = hm; el.digits.innerHTML = slots(hm); }
  el.date.textContent = DATE_FMT.format(d).replace(',', '');
}

export function setRoom(name) {
  el.room.textContent = name || 'No room yet';
  el.rname.setAttribute('aria-label', (name ? 'This is the ' + name : 'No room yet') + '. Change the room');
  document.title = name ? 'Housemate · ' + name : 'Housemate';
}
export function setDoor(label, enabled) {
  el.door.textContent = label || '';
  el.door.setAttribute('aria-label', enabled ? 'Talk to the ' + label : label + ' (the hub)');
}
export function setConnected(on) { el.rname.dataset.conn = on ? '1' : '0'; }

// timers: the soonest one inside the room; a ringing one fills the room
let lastTimers = '';
export function renderTimers(list, ringing) {
  root.classList.toggle('has-timer', list.length > 0);
  let key = ringing ? 'R' + ringing.label + ringing.sub : '';
  for (let i = 0; i < list.length && i < 3; i++) key += '|' + list[i].label + mmss(list[i].left);
  if (key === lastTimers) return;
  lastTimers = key;
  if (ringing) { el.rbLabel.textContent = ringing.label; el.rbSub.textContent = ringing.sub; }
  if (!list.length) return;
  el.tbLabel.textContent = list[0].label;
  el.tbDigits.innerHTML = slots(mmss(list[0].left));
  el.tbMore.textContent = list.slice(1, 3).map((t) => t.label + ' ' + mmss(t.left)).join('   ');
}

// caption model -> DOM (word spans revealed as they are spoken)
export function renderCaption(m, rebuild) {
  const c = el.caption;
  if (rebuild) {
    el.capLabel.textContent = m.label || '';
    el.capwrap.classList.toggle('none', m.mode === 'none');
    if (m.mode === 'none') { c.className = ''; c.textContent = ''; return; }
    if (m.mode === 'live') { c.className = 'live'; c.textContent = m.text; return; }
    c.className = m.mode === 'you' ? 'you' : m.mode === 'dim' ? 'dim' : '';
    clear(c);
    const ws = m.text.split(/\s+/).filter(Boolean).slice(0, 120);
    for (let i = 0; i < ws.length; i++) {
      const s = document.createElement('span'); s.className = 'w'; s.textContent = ws[i]; c.appendChild(s);
      if (i < ws.length - 1) c.appendChild(document.createTextNode(' '));
    }
  }
  if (m.mode === 'say' || m.mode === 'dim') {
    const ws = c.children, on = m.on == null ? 999 : m.on;
    for (let i = 0; i < ws.length; i++) ws[i].classList.toggle('on', i < on);
  }
}

let lastStatus = '';
export function setStatus(text, badge) {
  if (text === lastStatus) return;
  lastStatus = text;
  el.status.textContent = text;
  el.status.classList.toggle('badge', !!badge);
}
export function setHint(text) { if (el.hint.textContent !== text) el.hint.textContent = text; }

// the approval / question card; onPick(value) per button
export function renderPrompt(p, onPick) {
  el.pTitle.textContent = p.title;
  clear(el.pDetails);
  for (const d of p.details) {
    const s = document.createElement('span');
    if (d.code) { const code = document.createElement('code'); code.textContent = d.text; s.appendChild(code); } else s.textContent = d.text;
    el.pDetails.appendChild(s);
  }
  clear(el.pBtns);
  el.pBtns.classList.toggle('many', p.buttons.length > 2);
  for (const b of p.buttons) {
    const btn = document.createElement('button');
    btn.type = 'button'; btn.textContent = b.label; if (b.primary) btn.className = 'primary';
    btn.addEventListener('click', () => {
      for (const x of el.pBtns.children) x.disabled = true;
      onPick(b.value);
    });
    el.pBtns.appendChild(btn);
  }
  el.pFoot.textContent = p.foot;
}

// a content card: plain text; numbered or bulleted lines become a list
export function renderCard(card) {
  el.cTitle.textContent = card.title || 'Card';
  clear(el.cBody);
  let list = null;
  for (const raw of String(card.body || '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) { list = null; continue; }
    const num = /^(\d+)[.)]\s+(.*)$/.exec(line), dot = /^[-*•]\s+(.*)$/.exec(line);
    if (num || dot) {
      const tag = num ? 'OL' : 'UL';
      if (!list || list.tagName !== tag) { list = document.createElement(tag); el.cBody.appendChild(list); }
      const li = document.createElement('li'); li.textContent = num ? num[2] : dot[1]; list.appendChild(li);
    } else {
      list = null;
      const p = document.createElement('p'); p.textContent = line; el.cBody.appendChild(p);
    }
  }
}

export function renderRooms(rooms, current, onPick, box = el.rooms) {
  clear(box);
  for (const r of rooms) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = r.name; b.setAttribute('aria-pressed', r.slug === current ? 'true' : 'false');
    b.addEventListener('click', () => onPick(r.name));
    box.appendChild(b);
  }
}

export function showPairCode(code) {
  el.code.textContent = code.length === 6 ? code.slice(0, 3) + ' ' + code.slice(3) : code;
  el.code.setAttribute('aria-label', 'Pairing code ' + code.split('').join(' '));
  el.codeCmd.textContent = 'hermes housemate pair ' + code;
}

export function showError(kicker, title, text, button, onClick) {
  el.errKicker.textContent = kicker;
  el.errTitle.textContent = title;
  el.errText.textContent = text;
  el.errBtn.textContent = button;
  el.errBtn.onclick = onClick;
  showScreen('error');
}
