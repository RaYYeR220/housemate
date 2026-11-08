// Housemate · Rooms. The desktop half of the housemate plugin (Hermes Desktop Plugin SDK).
//
// A live plan of the home: every room is a zone, every paired device a dot with its earshot drawn
// round it. Devices that can hear each other (same room, or rooms in one earshot group) share an
// outline; when the wake word reaches them, both rings flash and the hub's arbitration decides:
// one answers, the others defer. Timers, the last exchange, announcements, pairing and the
// conversation mode sit on the same page; a status-bar chip counts rooms and timers.
//
// Data: plugin RPC `plugin.housemate.rooms.*` where Hermes has it, else the plugin's REST routes
// (`/api/plugins/housemate/*`, Hermes 0.23.0). Live: the `plugin.housemate.rooms.changed` event;
// polling when no event arrives. Plain ESM, no build: only the SDK and React are imported.

import {
  Button,
  Codicon,
  ConfirmDialog,
  CopyButton,
  Kbd,
  Loader,
  PALETTE_AREA,
  ROUTES_AREA,
  SIDEBAR_NAV_AREA,
  STATUSBAR_AREAS,
  SegmentedControl,
  StatusDot,
  Textarea,
  WorkspacePageHeaderControl,
  atom,
  createBudgetedLoop,
  haptic,
  host,
  relativeTime,
  useTheme,
  useValue
} from '@hermes/plugin-sdk'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Fragment, jsx, jsxs } from 'react/jsx-runtime'

const PLUGIN_ID = 'housemate'
const ROUTE = '/rooms'
const EVENT = 'plugin.housemate.rooms.changed'
const RPC_PREFIX = 'plugin.housemate.rooms.'
const MODES = [
  { id: 'household', label: 'Household' },
  { id: 'per_room', label: 'Per room' }
]
const ACTIVE = { listening: 1, thinking: 1, speaking: 1, ringing: 1 }
const TAU = Math.PI * 2
const PAD = 12
const TRAY = 46
const TOP = 20

// ─────────────────────────────────────────────────────────────── store

let store = null

function errText(err) {
  if (!err) return 'unknown error'
  if (typeof err === 'string') return err
  return String(err.message || err.error || err)
}

function missingMethod(err) {
  return !!err && (err.code === -32601 || /method not found|unknown method|no such method/i.test(errText(err)))
}

function createStore(ctx) {
  const $data = atom(null)
  const $status = atom({ loading: true, error: null, transport: null, live: false })
  const $pulses = atom([])
  let transport = null
  let lastFetch = 0
  let gotEvent = false
  let inflight = null
  let pageViews = 0
  const rang = new Set()

  const setStatus = patch => $status.set({ ...$status.get(), ...patch })

  function accept(state) {
    if (!state || typeof state !== 'object' || !Array.isArray(state.rooms)) return
    const prev = $data.get()
    let next = state
    if (!next.qr && prev && prev.qr && prev.hub && next.hub && prev.hub.url === next.hub.url) next = { ...next, qr: prev.qr }
    next = { ...next, offset: typeof next.now === 'number' ? next.now - Date.now() / 1000 : 0 }
    $data.set(next)
    setStatus({ loading: false, error: null })
    for (const t of next.timers || []) {
      if (t.state !== 'ringing' || rang.has(t.id)) continue
      rang.add(t.id)
      const room = roomName(next, t.room)
      host.notify({ kind: 'info', title: 'Housemate', message: `${t.label ? cap(t.label) + ' timer' : 'A timer'} is ringing in ${room}.` })
    }
  }

  async function request(name, params) {
    if (transport !== 'rest') {
      try {
        const result = await host.request(RPC_PREFIX + name, params || {})
        transport = 'rpc'
        setStatus({ transport })
        return result
      } catch (err) {
        if (!missingMethod(err)) throw err
        transport = 'rest'
        setStatus({ transport })
      }
    }
    return name === 'state'
      ? ctx.rest('/state')
      : ctx.rest('/' + name, { method: 'POST', body: params || {}, timeoutMs: name === 'announce' ? 95000 : 25000 })
  }

  function refresh() {
    if (inflight) return inflight
    inflight = (async () => {
      try {
        const result = await request('state', { qr: true })
        lastFetch = Date.now()
        if (result && result.ok) accept(result)
        else setStatus({ loading: false, error: (result && result.error) || 'The hub did not answer.' })
      } catch (err) {
        lastFetch = Date.now()
        setStatus({ loading: false, error: errText(err) })
      } finally {
        inflight = null
      }
    })()
    return inflight
  }

  async function act(name, params) {
    try {
      const result = await request(name, params)
      if (!result || result.ok === false) {
        host.notify({ kind: 'error', title: 'Housemate', message: (result && result.error) || 'The hub did not answer.' })
        return result || { ok: false }
      }
      refresh()
      return result
    } catch (err) {
      host.notifyError(err, 'Housemate: the hub did not answer')
      return { ok: false, error: errText(err) }
    }
  }

  function pulse(targets) {
    const start = performance.now() / 1000
    $pulses.set([...$pulses.get().filter(p => start - p.start < 2), ...targets.map(to => ({ to, start, dur: 0.9 }))])
  }

  ctx.onEvent(EVENT, event => {
    const payload = event && event.payload
    if (!payload || !payload.state) return
    gotEvent = true
    setStatus({ live: true })
    accept(payload.state)
  })
  // Poll: every 2 s on the page until an event shows the feed works, then a slow safety net.
  ctx.setInterval(() => {
    const every = pageViews > 0 ? (gotEvent ? 15000 : 2000) : gotEvent ? 30000 : 10000
    if (Date.now() - lastFetch >= every - 100) refresh()
  }, 1000)
  if (host.state && host.state.gateway && typeof host.state.gateway.listen === 'function') {
    ctx.onDispose(host.state.gateway.listen(v => v === 'open' && refresh()))
  }
  refresh()

  return {
    $data,
    $status,
    $pulses,
    refresh,
    act,
    pulse,
    approvedOnce: false,
    pageOpen() {
      pageViews += 1
      refresh()
      return () => { pageViews -= 1 }
    }
  }
}

// ─────────────────────────────────────────────────────────────── helpers

function cap(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s
}

function roomName(data, slug) {
  const r = data && data.rooms ? data.rooms.find(x => x.room === slug) : null
  return r ? r.name : slug ? cap(String(slug).replace(/-/g, ' ')) : 'no room'
}

function nowS(data) {
  return Date.now() / 1000 + (data && data.offset ? data.offset : 0)
}

function mmss(sec) {
  sec = Math.max(0, Math.ceil(sec))
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = sec % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`
}

function prettyWake(model) {
  if (!model) return null
  return String(model)
    .replace(/_v[\d.]+$/i, '')
    .replace(/[_-]+/g, ' ')
    .split(' ')
    .map(cap)
    .join(' ')
}

function shortFingerprint(fp) {
  if (!fp) return ''
  const parts = String(fp).split(':')
  return parts.length > 10 ? `${parts.slice(0, 8).join(':')} … ${parts.slice(-2).join(':')}` : fp
}

function hostOf(url) {
  try {
    return new URL(url).host
  } catch {
    return url || ''
  }
}

function earshotKey(data, dev) {
  const room = (data.rooms || []).find(r => r.room === dev.room)
  return room && room.earshot_group ? 'group:' + room.earshot_group : 'room:' + dev.room
}

// Arbitration tags: "answered" for the winner, "deferred" for the others in that round, and
// "suppressed" while a device in the same earshot speaks (the echo guard: no follow-up there).
function tagsFor(data, t) {
  const tags = {}
  const recent = (data.wakes || []).filter(w => t - w.at < 3.5).sort((a, b) => a.at - b.at)
  for (const w of recent) {
    if (t - w.at < 2.5) tags[w.winner] = { tag: 'answered', at: w.at }
    for (const id of w.deferred || []) tags[id] = { tag: 'deferred', at: w.at }
  }
  const placed = (data.devices || []).filter(d => d.room && d.online)
  for (const d of placed) {
    if (tags[d.device_id] || d.state !== 'idle') continue
    const key = earshotKey(data, d)
    const loud = placed.find(o => o.device_id !== d.device_id && earshotKey(data, o) === key && o.state === 'speaking')
    if (loud) tags[d.device_id] = { tag: 'suppressed', at: t }
  }
  return tags
}

const STATE_WORD = {
  idle: 'idle',
  listening: 'listening',
  thinking: 'thinking',
  speaking: 'speaking',
  ringing: 'ringing',
  offline: 'offline'
}

// ─────────────────────────────────────────────────────────────── geometry

function orderRooms(rooms) {
  // Rooms in one earshot group sit next to each other, so their devices' circles can overlap.
  const out = []
  const seen = new Set()
  for (const r of rooms) {
    if (seen.has(r.room)) continue
    out.push(r)
    seen.add(r.room)
    if (!r.earshot_group) continue
    for (const o of rooms) {
      if (!seen.has(o.room) && o.earshot_group === r.earshot_group) {
        out.push(o)
        seen.add(o.room)
      }
    }
  }
  return out
}

function zoneRects(rooms, x0, y0, w, h, g) {
  const Z = {}
  const n = rooms.length
  const wide = w >= h * 0.85
  if (!n) return Z
  if (n === 1) {
    Z[rooms[0].room] = { x: x0, y: y0, w, h }
  } else if (n === 2) {
    if (wide) {
      const kw = Math.round((w - g) / 2)
      Z[rooms[0].room] = { x: x0, y: y0, w: kw, h }
      Z[rooms[1].room] = { x: x0 + kw + g, y: y0, w: w - kw - g, h }
    } else {
      const kh = Math.round((h - g) / 2)
      Z[rooms[0].room] = { x: x0, y: y0, w, h: kh }
      Z[rooms[1].room] = { x: x0, y: y0 + kh + g, w, h: h - kh - g }
    }
  } else if (n === 3) {
    if (wide) {
      const kw = Math.round((w - g) * 0.5)
      const oh = Math.round((h - g) * 0.52)
      Z[rooms[0].room] = { x: x0, y: y0, w: kw, h }
      Z[rooms[1].room] = { x: x0 + kw + g, y: y0, w: w - kw - g, h: oh }
      Z[rooms[2].room] = { x: x0 + kw + g, y: y0 + oh + g, w: w - kw - g, h: h - oh - g }
    } else {
      const kh = Math.round((h - g) * 0.5)
      const ow = Math.round((w - g) * 0.5)
      Z[rooms[0].room] = { x: x0, y: y0, w, h: kh }
      Z[rooms[1].room] = { x: x0, y: y0 + kh + g, w: ow, h: h - kh - g }
      Z[rooms[2].room] = { x: x0 + ow + g, y: y0 + kh + g, w: w - ow - g, h: h - kh - g }
    }
  } else {
    const cols = Math.max(2, Math.min(n, Math.round(Math.sqrt(n * Math.max(0.6, w / h)))))
    const rows = Math.ceil(n / cols)
    const ch = (h - g * (rows - 1)) / rows
    rooms.forEach((r, i) => {
      const row = Math.floor(i / cols)
      const inRow = row === rows - 1 ? n - row * cols : cols
      const col = i - row * cols
      const cw = (w - g * (inRow - 1)) / inRow
      Z[r.room] = { x: Math.round(x0 + col * (cw + g)), y: Math.round(y0 + row * (ch + g)), w: Math.round(cw), h: Math.round(ch) }
    })
  }
  return Z
}

function geometry(data, W, H, e) {
  const rooms = orderRooms(data.rooms || [])
  const g = 10 + 16 * e
  const x0 = PAD
  const y0 = PAD + TOP
  const w = Math.max(120, W - 2 * PAD)
  const h = Math.max(120, H - y0 - PAD - TRAY - 10)
  const Z = zoneRects(rooms, x0, y0, w, h, g)
  const S = Math.min(w, h)
  const R0 = Math.round(S * 0.26)
  const D = {}
  const center = z => ({ x: z.x + z.w / 2, y: z.y + z.h * 0.52 })
  for (const r of rooms) {
    const z = Z[r.room]
    if (!z) continue
    const members = (data.devices || []).filter(d => d.room === r.room)
    if (!members.length) continue
    const c = center(z)
    const peers = r.earshot_group ? rooms.filter(o => o.room !== r.room && o.earshot_group === r.earshot_group && Z[o.room]) : []
    let dir = { x: 0, y: 0 }
    if (peers.length) {
      const px = peers.reduce((s, o) => s + center(Z[o.room]).x, 0) / peers.length
      const py = peers.reduce((s, o) => s + center(Z[o.room]).y, 0) / peers.length
      const len = Math.hypot(px - c.x, py - c.y) || 1
      dir = { x: (px - c.x) / len, y: (py - c.y) / len }
    }
    const pull = peers.length ? Math.min(z.w, z.h) * 0.28 : 0
    const base = { x: c.x + dir.x * pull, y: c.y + dir.y * pull }
    const perp = peers.length ? { x: -dir.y, y: dir.x } : { x: 1, y: 0 }
    const step = Math.min(64, (Math.abs(perp.x) > 0.5 ? z.w : z.h) / (members.length + 1))
    members.forEach((d, i) => {
      const off = (i - (members.length - 1) / 2) * step
      const x = Math.max(z.x + 30, Math.min(z.x + z.w - 30, base.x + perp.x * off))
      const y = Math.max(z.y + 64, Math.min(z.y + z.h - 64, base.y + perp.y * off))
      const r0 = peers.length ? R0 : Math.max(26, Math.min(R0, Math.min(z.w, z.h) / 2 - 10))
      D[d.device_id] = { x: Math.round(x), y: Math.round(y), r: Math.round(r0), room: r.room }
    })
  }
  // Devices in one earshot but different rooms: make sure their circles really overlap.
  const placed = (data.devices || []).filter(d => D[d.device_id])
  for (let i = 0; i < placed.length; i++) {
    for (let j = i + 1; j < placed.length; j++) {
      const a = placed[i]
      const b = placed[j]
      if (a.room === b.room || earshotKey(data, a) !== earshotKey(data, b)) continue
      const da = D[a.device_id]
      const db = D[b.device_id]
      const need = (Math.hypot(da.x - db.x, da.y - db.y) + 24) / 2
      da.r = Math.max(da.r, Math.round(need))
      db.r = Math.max(db.r, Math.round(need))
    }
  }
  const groups = {}
  for (const d of placed) (groups[earshotKey(data, d)] = groups[earshotKey(data, d)] || []).push(d.device_id)
  const zs = Object.values(Z)
  const house = zs.length
    ? {
        x: Math.min(...zs.map(z => z.x)) - 6,
        y: Math.min(...zs.map(z => z.y)) - 6,
        x1: Math.max(...zs.map(z => z.x + z.w)) + 6,
        y1: Math.max(...zs.map(z => z.y + z.h)) + 6
      }
    : { x: x0, y: y0, x1: x0 + w, y1: y0 + h }
  return { Z, D, groups, house, hub: { x: house.x1 - 28, y: house.y }, tray: { x: PAD, y: H - PAD - TRAY, w: W - 2 * PAD, h: TRAY }, W, H }
}

function zoneAt(geo, x, y) {
  for (const [room, z] of Object.entries(geo.Z)) if (x >= z.x && x <= z.x + z.w && y >= z.y && y <= z.y + z.h) return room
  return null
}

// ─────────────────────────────────────────────────────────────── theme colours (canvas)

// Theme tokens can be expressions (color-mix(), var()) a canvas does not parse. The page's own
// hidden probe element resolves each one through CSS; its computed colour is concrete.
function parseColor(v) {
  v = (v || '').trim()
  let m = v.match(/^#([0-9a-f]{3,8})$/i)
  if (m) {
    let hex = m[1]
    if (hex.length <= 4) hex = hex.split('').map(c => c + c).join('')
    const a = hex.length >= 8 ? parseInt(hex.slice(6, 8), 16) / 255 : 1
    return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16), a]
  }
  m = v.match(/^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)(?:[\s,/]+([\d.]+%?))?\s*\)$/i)
  if (m) return [+m[1], +m[2], +m[3], alphaOf(m[4])]
  m = v.match(/^color\(\s*srgb\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)(?:\s*\/\s*([\d.]+%?))?\s*\)$/i)
  if (m) return [+m[1] * 255, +m[2] * 255, +m[3] * 255, alphaOf(m[4])]
  return null
}

function alphaOf(a) {
  if (a === undefined || a === '') return 1
  return a.endsWith('%') ? parseFloat(a) / 100 : +a
}

function cssColor(rgba) {
  const [r, g, b, a] = rgba.map((x, i) => (i < 3 ? Math.round(Math.max(0, Math.min(255, x))) : Math.max(0, Math.min(1, x))))
  return `rgba(${r},${g},${b},${a})`
}

function luminance(rgb) {
  const f = c => {
    c /= 255
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
  }
  return 0.2126 * f(rgb[0]) + 0.7152 * f(rgb[1]) + 0.0722 * f(rgb[2])
}

const TOKENS = {
  accent: '--ui-accent',
  s1: '--ui-stroke-primary',
  s2: '--ui-stroke-secondary',
  s3: '--ui-stroke-tertiary',
  s4: '--ui-stroke-quaternary',
  t1: '--ui-text-primary',
  t2: '--ui-text-secondary',
  t3: '--ui-text-tertiary',
  t4: '--ui-text-quaternary',
  green: '--ui-green',
  yellow: '--ui-yellow',
  hover: '--ui-row-hover-background',
  card: '--ui-bg-card',
  bg: '--ui-bg-chrome'
}

function readColors(probe) {
  const c = {}
  const raw = {}
  for (const [key, name] of Object.entries(TOKENS)) {
    probe.style.color = `var(${name})`
    const value = getComputedStyle(probe).color
    const rgba = parseColor(value)
    raw[key] = rgba
    c[key] = rgba ? cssColor(rgba) : value
  }
  probe.style.color = ''
  // A blue accent on a blue ground (nous-alt dark) all but disappears: draw "live" in the
  // primary text colour when the accent is too close to the page.
  c.live = c.accent
  if (raw.accent && raw.bg) {
    const la = luminance(raw.accent)
    const lb = luminance(raw.bg)
    if ((Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05) < 2.6) c.live = c.t1
  }
  return c
}

// ─────────────────────────────────────────────────────────────── canvas

const easeIO = x => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2)
const lerp = (a, b, k) => a + (b - a) * k

function drawScene(cv, model, now) {
  const { geo, data, colors: C, drop, dragAt, reduced, pulses, e } = model
  if (!geo || !C) return
  const ctx = cv.getContext('2d')
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const W = geo.W
  const H = geo.H
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
    cv.width = Math.round(W * dpr)
    cv.height = Math.round(H * dpr)
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, W, H)
  ctx.lineWidth = 1
  ctx.setLineDash([])
  const circle = (x, y, r) => {
    ctx.beginPath()
    ctx.arc(x, y, Math.max(0.5, r), 0, TAU)
    ctx.stroke()
  }
  const t = nowS(data)

  // household: one house outline round every zone
  if (e < 1) {
    const hs = geo.house
    ctx.globalAlpha = 1 - e
    ctx.strokeStyle = C.s2
    ctx.beginPath()
    ctx.roundRect(hs.x + 0.5, hs.y + 0.5, hs.x1 - hs.x, hs.y1 - hs.y, 8)
    ctx.stroke()
    ctx.globalAlpha = 1
  }
  // zones
  for (const r of data.rooms || []) {
    const z = geo.Z[r.room]
    if (!z) continue
    const has = (data.devices || []).some(d => d.room === r.room)
    const hot = drop === r.room
    ctx.fillStyle = hot ? C.hover : C.card
    ctx.globalAlpha = hot ? 1 : 0.3
    ctx.beginPath()
    ctx.roundRect(z.x + 0.5, z.y + 0.5, z.w - 1, z.h - 1, 6)
    ctx.fill()
    ctx.globalAlpha = 1
    ctx.strokeStyle = hot ? C.live : e > 0.5 ? C.s2 : C.s3
    if (!has && !hot) ctx.setLineDash([3, 3])
    ctx.stroke()
    ctx.setLineDash([])
  }
  // the tray at the edge (devices not placed yet)
  const tr = geo.tray
  ctx.strokeStyle = C.s3
  ctx.setLineDash([3, 3])
  ctx.beginPath()
  ctx.roundRect(tr.x + 0.5, tr.y + 0.5, tr.w - 1, tr.h - 1, 4)
  ctx.stroke()
  ctx.setLineDash([])

  const devices = (data.devices || []).filter(d => geo.D[d.device_id])
  // earshot circles
  ctx.strokeStyle = C.s3
  for (const d of devices) {
    const p = geo.D[d.device_id]
    ctx.globalAlpha = d.online ? 1 : 0.4
    circle(p.x, p.y, p.r)
  }
  ctx.globalAlpha = 1
  // shared outline of every earshot with more than one device: each circle's arcs outside the others
  ctx.strokeStyle = C.s1
  ctx.setLineDash([4, 3])
  ctx.beginPath()
  for (const ids of Object.values(geo.groups)) {
    if (ids.length < 2) continue
    for (const id of ids) {
      const c = geo.D[id]
      let pen = false
      for (let k = 0; k <= 144; k++) {
        const a = (k / 144) * TAU
        const x = c.x + Math.cos(a) * c.r
        const y = c.y + Math.sin(a) * c.r
        let inside = false
        for (const o of ids) {
          if (o !== id && Math.hypot(x - geo.D[o].x, y - geo.D[o].y) < geo.D[o].r - 0.5) {
            inside = true
            break
          }
        }
        if (inside) {
          pen = false
          continue
        }
        if (!pen) {
          ctx.moveTo(x, y)
          pen = true
        } else ctx.lineTo(x, y)
      }
    }
  }
  ctx.stroke()
  ctx.setLineDash([])
  // a dragged device previews its earshot
  if (dragAt) {
    ctx.strokeStyle = drop ? C.live : C.t4
    ctx.setLineDash([3, 3])
    circle(dragAt.x, dragAt.y, Math.round(Math.min(W, H) * 0.2))
    ctx.setLineDash([])
  }

  const tags = tagsFor(data, t)
  const hub = geo.hub
  for (const d of devices) {
    const p = geo.D[d.device_id]
    const s = d.online ? d.state : 'offline'
    const R = p.r
    const tg = tags[d.device_id]
    // live state, the sound clipped to the device's own earshot
    ctx.save()
    ctx.beginPath()
    ctx.arc(p.x, p.y, R, 0, TAU)
    ctx.clip()
    ctx.strokeStyle = C.live
    ctx.lineWidth = 1.2
    if (!reduced && s === 'listening') {
      for (let i = 0; i < 4; i++) {
        const f = (now * 0.55 + i / 4) % 1
        ctx.globalAlpha = Math.pow(f, 1.3) * (1 - Math.pow(f, 10)) * 0.75
        circle(p.x, p.y, R - f * (R - 8))
      }
    }
    if (!reduced && s === 'speaking') {
      for (let i = 0; i < 4; i++) {
        const f = (now * 0.6 + i / 4) % 1
        ctx.globalAlpha = Math.pow(1 - f, 1.5) * Math.min(1, f * 10) * 0.8
        circle(p.x, p.y, 8 + f * (R - 8))
      }
    }
    if (s === 'ringing') {
      ctx.strokeStyle = C.yellow
      ctx.lineWidth = 1.5
      if (reduced) {
        ctx.globalAlpha = 0.8
        circle(p.x, p.y, 14)
      } else {
        const cyc = now % 1.2
        for (const b of [0, 0.18]) {
          const age = cyc - b
          if (age < 0 || age > 0.9) continue
          const k = age / 0.9
          ctx.globalAlpha = Math.pow(1 - k, 1.5)
          circle(p.x, p.y, 9 + k * R)
        }
      }
    }
    ctx.restore()
    ctx.globalAlpha = 1
    ctx.lineWidth = 1
    // arbitration: both rings flash for the 400 ms window, then the winner locks
    if (tg && tg.tag !== 'suppressed') {
      const age = t - tg.at
      if (age >= 0 && age < 0.45 && !reduced) {
        ctx.strokeStyle = C.live
        ctx.lineWidth = 1.5
        ctx.globalAlpha = (1 - age / 0.45) * (0.5 + 0.5 * Math.abs(Math.sin(age * 22)))
        circle(p.x, p.y, R)
        ctx.globalAlpha = 1
      } else if (tg.tag === 'answered') {
        ctx.strokeStyle = C.live
        ctx.lineWidth = 1.5
        circle(p.x, p.y, 10)
        ctx.globalAlpha = 0.55
        ctx.lineWidth = 1
        circle(p.x, p.y, R)
        ctx.globalAlpha = 1
      }
      ctx.lineWidth = 1
    }
    // thinking: a bead runs to the hub and back
    if (s === 'thinking') {
      ctx.strokeStyle = C.live
      ctx.globalAlpha = 0.45
      ctx.beginPath()
      ctx.moveTo(p.x, p.y)
      ctx.lineTo(hub.x, hub.y)
      ctx.stroke()
      ctx.globalAlpha = 1
      if (!reduced) {
        const ph = (now % 1.1) / 1.1
        const f = easeIO(ph < 0.5 ? ph * 2 : (1 - ph) * 2)
        ctx.fillStyle = C.live
        ctx.beginPath()
        ctx.arc(lerp(p.x, hub.x, f), lerp(p.y, hub.y, f), 3, 0, TAU)
        ctx.fill()
      }
    }
    if (s === 'offline' && !reduced) {
      const k = (now % 4) / 0.6
      if (k < 1) {
        ctx.strokeStyle = C.t4
        ctx.globalAlpha = 1 - k
        circle(p.x, p.y, 6 + k * 14)
        ctx.globalAlpha = 1
      }
    }
    // the device dot
    const dim = tg && tg.tag !== 'answered' && !ACTIVE[s] ? 0.4 : 1
    ctx.globalAlpha = dim
    const col = s === 'offline' ? C.t4 : s === 'ringing' ? C.yellow : ACTIVE[s] ? C.live : C.green
    if (tg && tg.tag === 'suppressed' && !ACTIVE[s]) {
      ctx.strokeStyle = C.t3
      ctx.lineWidth = 1.25
      circle(p.x, p.y, 4)
      ctx.lineWidth = 1
    } else {
      ctx.fillStyle = col
      ctx.beginPath()
      ctx.arc(p.x, p.y, 4.5, 0, TAU)
      ctx.fill()
      if (s === 'thinking') {
        ctx.fillStyle = C.bg
        ctx.beginPath()
        ctx.arc(p.x, p.y, 2, 0, TAU)
        ctx.fill()
      }
    }
    ctx.globalAlpha = 1
  }
  // the hub: a small square on the house outline
  ctx.fillStyle = C.bg
  ctx.strokeStyle = C.t2
  ctx.fillRect(Math.round(hub.x) - 4.5, Math.round(hub.y) - 4.5, 9, 9)
  ctx.strokeRect(Math.round(hub.x) - 4.5, Math.round(hub.y) - 4.5, 9, 9)
  // announcements in flight: from the hub to each target device
  const ts = performance.now() / 1000
  for (const pu of pulses) {
    const b = geo.D[pu.to]
    if (!b) continue
    const f = easeIO(Math.max(0, Math.min(1, (ts - pu.start) / pu.dur)))
    if (ts - pu.start > pu.dur + 0.3) continue
    ctx.strokeStyle = C.live
    ctx.globalAlpha = 0.4
    ctx.beginPath()
    ctx.moveTo(hub.x, hub.y)
    ctx.lineTo(b.x, b.y)
    ctx.stroke()
    ctx.globalAlpha = 1
    ctx.fillStyle = C.live
    ctx.beginPath()
    ctx.arc(lerp(hub.x, b.x, f), lerp(hub.y, b.y, f), 3.5, 0, TAU)
    ctx.fill()
  }
}

function animating(model) {
  const { data, pulses, dragAt, e, eTarget } = model
  if (!data) return false
  if (dragAt || Math.abs(e - eTarget) > 0.001) return true
  const ts = performance.now() / 1000
  if (pulses.some(p => ts - p.start < p.dur + 0.4)) return true
  const t = nowS(data)
  if ((data.wakes || []).some(w => t - w.at < 3.6)) return true
  return (data.devices || []).some(d => d.room && (ACTIVE[d.online ? d.state : 'offline'] || !d.online))
}

// ─────────────────────────────────────────────────────────────── styles

const CSS = `
.hm-page{display:flex;flex-direction:column;height:100%;min-height:0;overflow:auto;color:var(--ui-text-primary);padding:18px clamp(1.25rem,4vw,4rem) 24px;box-sizing:border-box}
.hm-head{display:flex;align-items:center;gap:12px;min-height:32px}
.hm-title{font-size:.9375rem;font-weight:600;letter-spacing:-.01em}
.hm-sub{font-size:12px;color:var(--ui-text-tertiary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.hm-grow{flex:1 1 auto}
.hm-cap{font-size:12px;color:var(--ui-text-tertiary);margin:6px 0 4px;line-height:1.45}
.hm-body{display:grid;grid-template-columns:minmax(0,1fr) 272px;gap:28px;align-items:start;max-width:75rem;width:100%}
.hm-narrow .hm-body{grid-template-columns:minmax(0,1fr)}
.hm-ear{position:relative;height:clamp(440px,calc(100vh - 190px),760px);user-select:none}
.hm-narrow .hm-ear{height:520px}
.hm-ear canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.hm-layer{position:absolute;inset:0;pointer-events:none}
.hm-probe{position:absolute;width:0;height:0;overflow:hidden;visibility:hidden}
.hm-layer > *{pointer-events:auto}
.hm-zn{position:absolute;pointer-events:none!important;transition:left .5s,top .5s,width .5s,height .5s}
.hm-zn-head{position:absolute;left:12px;top:10px;right:12px}
.hm-zn-name{font-size:.9375rem;font-weight:600;letter-spacing:-.01em;line-height:1.3}
.hm-own{font-size:12px;font-weight:400;letter-spacing:0;color:var(--ui-text-tertiary);transition:opacity .5s}
.hm-timers{margin-top:2px;display:inline-flex;flex-direction:column;pointer-events:auto}
.hm-tmr{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--ui-text-secondary);white-space:nowrap;height:22px}
.hm-tmr .hm-mono{color:var(--ui-text-primary)}
.hm-tmr.is-ringing,.hm-tmr.is-ringing .hm-mono{color:var(--ui-yellow);font-weight:600}
.hm-mono{font-family:var(--font-mono,ui-monospace,monospace);font-size:12px;font-variant-numeric:tabular-nums}
.hm-last{position:absolute;left:12px;right:12px;bottom:9px;font-size:12px;line-height:1.4;color:var(--ui-text-tertiary)}
.hm-last div{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hm-ago{color:var(--ui-text-quaternary)}
.hm-from{color:var(--ui-text-secondary)}
.hm-empty{position:absolute;left:12px;right:12px;text-align:center;font-size:12px;color:var(--ui-text-tertiary)}
.hm-dv{position:absolute;transform:translateX(-50%);text-align:center;white-space:nowrap;pointer-events:none!important;line-height:1.35;transition:left .5s,top .5s}
.hm-dv .hm-n{font-size:11px;color:var(--ui-text-tertiary)}
.hm-dv .hm-s{font-size:11px;font-weight:500;color:var(--ui-text-secondary)}
.hm-s.is-live{color:var(--hm-live)}
.hm-s.is-warn{color:var(--ui-yellow);font-weight:600}
.hm-s.is-muted,.hm-s.is-deferred,.hm-s.is-suppressed{color:var(--ui-text-quaternary)}
.hm-s.is-answered{color:var(--hm-live);font-weight:600}
.hm-grp{position:absolute;transform:translate(-50%,-50%);font-size:11px;color:var(--ui-text-tertiary);background:var(--ui-bg-chrome);padding:0 6px;white-space:nowrap;pointer-events:none!important}
.hm-legend{position:absolute;font-size:.7rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--ui-text-tertiary);background:var(--ui-bg-chrome);padding:0 6px;transform:translateY(-50%);pointer-events:none!important;white-space:nowrap;transition:opacity .5s}
.hm-hublbl{position:absolute;transform:translate(-100%,-50%);font-size:11px;color:var(--ui-text-tertiary);background:var(--ui-bg-chrome);padding:0 8px 0 6px;pointer-events:none!important;white-space:nowrap}
.hm-tray{position:absolute;display:flex;align-items:center;gap:12px;padding:0 12px;font-size:12px;color:var(--ui-text-tertiary);overflow:hidden}
.hm-chip{display:flex;align-items:center;gap:8px;min-width:0}
.hm-grab{width:22px;height:22px;display:grid;place-items:center;cursor:grab;border-radius:4px;flex:none;touch-action:none}
.hm-grab:hover{background:var(--chrome-action-hover)}
.hm-grab i{position:relative;width:8px;height:8px;border-radius:50%;background:var(--ui-yellow);display:block}
.hm-grab i::after{content:'';position:absolute;inset:-1px;border-radius:50%;border:1px solid var(--ui-yellow);animation:hm-wait 2s ease-out infinite}
.hm-grab.is-placed i::after{border-color:var(--ui-green)}
@keyframes hm-wait{from{transform:scale(1);opacity:.7}to{transform:scale(3.2);opacity:0}}
.hm-grab.is-placed i{background:var(--ui-green)}
.hm-chip .hm-nm{color:var(--ui-text-primary);font-weight:500;white-space:nowrap;cursor:grab}
.hm-hint{min-width:0;color:var(--ui-text-quaternary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hm-dragdot{position:fixed;z-index:65;width:10px;height:10px;margin:-5px 0 0 -5px;border-radius:50%;background:var(--ui-yellow);pointer-events:none}
.hm-dragcover{position:fixed;inset:0;z-index:64;cursor:grabbing}
.hm-col{display:flex;flex-direction:column;gap:30px;min-width:0}
.hm-sec-title{font-size:.7rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--ui-text-tertiary);margin:0 0 8px}
.hm-targets{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 6px}
.hm-to{font-size:12px;color:var(--ui-text-tertiary)}
.hm-to b{color:var(--ui-text-primary);font-weight:600}
.hm-row{display:flex;align-items:center;gap:8px}
.hm-between{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:10px}
.hm-qr{display:flex;gap:14px;align-items:flex-start}
.hm-qr svg{flex:none;display:block}
.hm-small{font-size:12px;line-height:1.45;color:var(--ui-text-secondary)}
.hm-label{font-size:12px;color:var(--ui-text-tertiary);margin:12px 0 2px}
.hm-req{padding:10px 0;border-bottom:1px solid var(--ui-stroke-quaternary)}
.hm-req:last-child{border-bottom:0}
.hm-req-name{display:flex;align-items:center;gap:8px;font-size:.875rem;font-weight:600}
.hm-req-meta{font-size:12px;color:var(--ui-text-tertiary);margin:2px 0 8px 14px}
.hm-req-meta .hm-mono{color:var(--ui-text-primary)}
.hm-facts{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;font-size:12px}
.hm-facts dt{color:var(--ui-text-tertiary)}
.hm-facts dd{margin:0;color:var(--ui-text-primary);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hm-note{font-size:12px;color:var(--ui-text-tertiary);line-height:1.45;margin-top:8px}
.hm-center{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;min-height:360px;color:var(--ui-text-tertiary);font-size:13px;text-align:center}
@media (prefers-reduced-motion: reduce){.hm-zn,.hm-dv,.hm-own,.hm-legend{transition:none}.hm-grab i::after{animation:none;opacity:0}}
`

// ─────────────────────────────────────────────────────────────── components

function useTick(ms) {
  const [, set] = useState(0)
  useEffect(() => {
    const id = setInterval(() => set(n => n + 1), ms)
    return () => clearInterval(id)
  }, [ms])
}

function useSize(ref) {
  const [size, setSize] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return undefined
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      setSize(s => (Math.round(r.width) === s.w && Math.round(r.height) === s.h ? s : { w: Math.round(r.width), h: Math.round(r.height) }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return size
}

function SectionTitle({ children }) {
  return jsx('h4', { className: 'hm-sec-title', children })
}

function TimerRow({ timer, data }) {
  const left = timer.due_at - nowS(data)
  const ringing = timer.state === 'ringing'
  const label = timer.label ? cap(timer.label) : timer.kind === 'alarm' ? 'Alarm' : 'Timer'
  return jsxs('div', {
    className: 'hm-tmr' + (ringing ? ' is-ringing' : ''),
    children: [
      jsx('span', { children: label }),
      jsx('span', { className: 'hm-mono', children: ringing ? 'ringing' : mmss(left) }),
      jsx(Button, {
        variant: 'ghost',
        size: 'icon-xs',
        'aria-label': ringing ? `Stop the ${label} timer` : `Cancel the ${label} timer`,
        title: ringing ? 'Stop' : 'Cancel timer',
        onClick: () => {
          haptic('tap')
          store.act('timer_cancel', { timer_id: timer.id })
        },
        children: jsx(Codicon, { name: 'close', size: 12 })
      })
    ]
  })
}

function DeviceLabel({ dev, pos, tag }) {
  const s = dev.online ? dev.state : 'offline'
  let word = STATE_WORD[s] || s
  let cls = ACTIVE[s] ? (s === 'ringing' ? 'is-warn' : 'is-live') : s === 'offline' ? 'is-muted' : ''
  if (tag && (tag.tag !== 'answered' ? s === 'idle' : ACTIVE[s] || s === 'idle')) {
    word = tag.tag
    cls = 'is-' + tag.tag
  }
  return jsxs('div', {
    className: 'hm-dv',
    style: { left: pos.x + 'px', top: pos.y + 11 + 'px' },
    children: [jsx('div', { className: 'hm-n', children: dev.name }), jsx('div', { className: 'hm-s ' + cls, children: word })]
  })
}

function Zone({ room, z, data, e, pendingFor }) {
  const devs = (data.devices || []).filter(d => d.room === room.room)
  const timers = (data.timers || []).filter(t => t.room === room.room)
  const x = (data.exchanges || {})[room.room]
  const t = nowS(data)
  let last = null
  if (devs.length) {
    last = x
      ? jsxs('div', {
          className: 'hm-last',
          children: [
            jsx('div', {
              children: x.from ? jsx('span', { className: 'hm-from', children: 'Announcement' }) : x.you ? `“${x.you}”` : ''
            }),
            jsxs('div', {
              children: [
                jsx('span', { children: x.say || '…' }),
                ' ',
                jsx('span', { className: 'hm-ago', children: `· ${relativeTime(x.at * 1000, t * 1000)}` })
              ]
            })
          ]
        })
      : jsx('div', { className: 'hm-last', children: jsx('div', { children: 'No conversation yet' }) })
  }
  return jsxs('div', {
    className: 'hm-zn',
    style: { left: z.x + 'px', top: z.y + 'px', width: z.w + 'px', height: z.h + 'px' },
    children: [
      jsxs('div', {
        className: 'hm-zn-head',
        children: [
          jsxs('div', {
            className: 'hm-zn-name',
            children: [jsx('span', { children: room.name }), jsx('span', { className: 'hm-own', style: { opacity: e }, children: ' · own conversation' })]
          }),
          timers.length ? jsx('div', { className: 'hm-timers', children: timers.map(tm => jsx(TimerRow, { timer: tm, data }, tm.id)) }) : null
        ]
      }),
      last,
      devs.length
        ? null
        : jsx('div', {
            className: 'hm-empty',
            style: { top: z.h * 0.5 - 8 + 'px' },
            children: pendingFor ? `${pendingFor} wants this room. Drag it in from below.` : 'No device yet. Drag one in from below.'
          })
    ]
  })
}

function Tray({ geo, data, onGrab }) {
  const pending = data.pending || []
  const unplaced = (data.devices || []).filter(d => !d.room)
  const items = [
    ...pending.map(p => ({ kind: 'pending', key: 'p' + p.code, p })),
    ...unplaced.map(d => ({ kind: 'device', key: 'd' + d.device_id, d }))
  ]
  const tr = geo.tray
  const children = [jsx(SectionTitle, { children: 'Not placed' }, 'title')]
  if (!items.length) {
    children.push(jsx('span', { children: 'No devices waiting. Open the hub address on a phone or laptop to add one.' }, 'none'))
  }
  items.slice(0, 2).forEach((it, i) => {
    const name = it.kind === 'pending' ? it.p.name : it.d.name
    const grab = ev => onGrab(ev, it)
    children.push(
      jsxs(
        'div',
        {
          className: 'hm-chip',
          children: [
            jsx('span', {
              className: 'hm-grab' + (it.kind === 'device' ? ' is-placed' : ''),
              role: 'button',
              tabIndex: 0,
              'aria-label': `Drag ${name} into a room`,
              title: 'Drag into a room',
              onPointerDown: grab,
              children: jsx('i', {})
            }),
            jsx('span', { className: 'hm-nm', onPointerDown: grab, children: name }),
            it.kind === 'pending' ? jsx('span', { className: 'hm-mono', children: fmtCode(it.p.code) }) : null,
            it.kind === 'pending'
              ? jsx(Button, { variant: 'secondary', size: 'xs', onClick: () => approve(it.p, it.p.wants || null, data), children: 'Approve' })
              : null,
            it.kind === 'pending'
              ? jsx(Button, { variant: 'ghost', size: 'xs', onClick: () => deny(it.p), children: 'Deny' })
              : null,
            i === Math.min(items.length, 2) - 1 ? jsx('span', { className: 'hm-hint', children: 'or drag it into a room' }) : null
          ]
        },
        it.key
      )
    )
  })
  if (items.length > 2) children.push(jsx('span', { className: 'hm-hint', children: `+${items.length - 2} more on the right` }, 'more'))
  return jsx('div', { className: 'hm-tray', style: { left: tr.x + 'px', top: tr.y + 'px', width: tr.w + 'px', height: tr.h + 'px' }, children })
}

function fmtCode(code) {
  const c = String(code || '')
  return c.length === 6 ? `${c.slice(0, 3)} ${c.slice(3)}` : c
}

let confirmSetter = null

function approve(p, room, data) {
  const go = () => store.act('pair_approve', room ? { code: p.code, room } : { code: p.code })
  if (store.approvedOnce || !confirmSetter) {
    go()
    return
  }
  const where = room ? roomName(data, room) || room : null
  confirmSetter({
    title: where ? `Add ${p.name} to ${where}?` : `Add ${p.name}?`,
    description: `Check that the device shows ${fmtCode(p.code)}. It will listen for the wake word${where ? ` and answer in ${where}` : ''}.`,
    confirmLabel: 'Add device',
    onConfirm: async () => {
      store.approvedOnce = true
      await go()
    }
  })
}

function deny(p) {
  store.act('pair_deny', { code: p.code })
}

function Earshot({ data, mode }) {
  const boxRef = useRef(null)
  const cvRef = useRef(null)
  const probeRef = useRef(null)
  const size = useSize(boxRef)
  const { themeName, resolvedMode } = useTheme()
  const pulses = useValue(store.$pulses)
  const [drag, setDrag] = useState(null)
  const eTarget = mode === 'per_room' ? 1 : 0
  const model = useRef({ e: eTarget, eTarget, data, geo: null, colors: null, pulses: [], drop: null, dragAt: null, reduced: false })
  useTick(500)

  const reduced = useMemo(() => {
    try {
      return window.matchMedia('(prefers-reduced-motion: reduce)').matches
    } catch {
      return false
    }
  }, [])
  const geo = useMemo(() => (size.w > 0 ? geometry(data, size.w, size.h, eTarget) : null), [data, size.w, size.h, eTarget])
  const m = model.current
  m.data = data
  m.eTarget = eTarget
  m.pulses = pulses
  m.reduced = reduced
  m.drop = drag ? drag.zone : null
  m.dragAt = drag ? { x: drag.x, y: drag.y } : null

  const loopRef = useRef(null)
  useEffect(() => {
    const cv = cvRef.current
    if (!cv) return undefined
    if (probeRef.current) m.colors = readColors(probeRef.current)
    const draw = now => {
      const mm = model.current
      const tNow = now / 1000
      if (Math.abs(mm.e - mm.eTarget) > 0.001) mm.e = mm.reduced ? mm.eTarget : lerp(mm.e, mm.eTarget, 0.18)
      else mm.e = mm.eTarget
      if (mm.geoTarget && mm.data) mm.geo = Math.abs(mm.e - mm.eTarget) > 0.001 ? geometry(mm.data, mm.geoTarget.W, mm.geoTarget.H, mm.e) : mm.geoTarget
      drawScene(cv, mm, tNow)
    }
    const loop = createBudgetedLoop(draw, { fps: 30, idleWhen: () => !animating(model.current) })
    loopRef.current = loop
    return () => {
      loop.dispose()
      loopRef.current = null
    }
  }, [])
  useEffect(() => {
    // The provider repaints the tokens in its own effect (after ours): read again a beat later.
    const read = () => {
      if (probeRef.current) model.current.colors = readColors(probeRef.current)
      if (loopRef.current) loopRef.current.wake()
    }
    read()
    const id = setTimeout(read, 80)
    return () => clearTimeout(id)
  }, [themeName, resolvedMode])
  useEffect(() => {
    m.geoTarget = geo
    if (!m.geo || Math.abs(m.e - eTarget) <= 0.001) m.geo = geo
    if (loopRef.current) loopRef.current.wake()
  })

  const onGrab = useCallback(
    (ev, item) => {
      if (!geo) return
      ev.preventDefault()
      const box = boxRef.current
      const r0 = box.getBoundingClientRect()
      const at = e2 => ({ x: e2.clientX - r0.left, y: e2.clientY - r0.top, cx: e2.clientX, cy: e2.clientY })
      const first = at(ev)
      let cur = { ...first, zone: zoneAt(geo, first.x, first.y), item }
      setDrag(cur)
      const move = e2 => {
        const p = at(e2)
        cur = { ...p, zone: zoneAt(geo, p.x, p.y), item }
        setDrag(cur)
      }
      const up = () => {
        window.removeEventListener('pointermove', move)
        window.removeEventListener('pointerup', up)
        window.removeEventListener('pointercancel', up)
        setDrag(null)
        const zone = cur.zone
        if (!zone) return
        haptic('tap')
        if (item.kind === 'pending') approve(item.p, zone, data)
        else store.act('assign', { device_id: item.d.device_id, room: zone })
      }
      window.addEventListener('pointermove', move)
      window.addEventListener('pointerup', up)
      window.addEventListener('pointercancel', up)
    },
    [geo, data]
  )

  const layer = []
  if (geo) {
    const t = nowS(data)
    const tags = tagsFor(data, t)
    const wants = {}
    for (const p of data.pending || []) if (p.wants) wants[slugish(p.wants)] = p.name
    for (const r of data.rooms || []) {
      const z = geo.Z[r.room]
      if (z) layer.push(jsx(Zone, { room: r, z, data, e: eTarget, pendingFor: wants[r.room] }, 'z' + r.room))
    }
    for (const d of data.devices || []) {
      const p = geo.D[d.device_id]
      if (p) layer.push(jsx(DeviceLabel, { dev: d, pos: p, tag: tags[d.device_id] }, 'd' + d.device_id))
    }
    // "within earshot" labels for groups that span rooms
    for (const ids of Object.values(geo.groups)) {
      const roomsIn = [...new Set(ids.map(id => geo.D[id].room))]
      if (roomsIn.length < 2) continue
      let lead = ids[0]
      for (const id of ids) if (geo.D[id].x < geo.D[lead].x) lead = id
      const c = geo.D[lead]
      // On the lead circle, where it stays clearest of every device's name label.
      const labels = Object.values(geo.D).map(d => ({ x: d.x, y: d.y + 20 }))
      let lx = 0
      let ly = 0
      let best = -1
      for (const deg of [115, 140, 90, 165, 65, 195]) {
        const ang = (deg * Math.PI) / 180
        const x = Math.max(90, Math.min(geo.W - 90, c.x + Math.cos(ang) * c.r))
        const y = Math.min(geo.tray.y - 20, c.y + Math.sin(ang) * c.r)
        const room = Math.min(...labels.map(l => Math.hypot((l.x - x) / 2.2, l.y - y)))
        if (room > best + 4) {
          best = room
          lx = x
          ly = y
        }
      }
      layer.push(
        jsx('div', { className: 'hm-grp', style: { left: lx + 'px', top: ly + 'px' }, children: `Within earshot · ${roomsIn.map(r => roomName(data, r)).join(' + ')}` }, 'g' + ids.join())
      )
    }
    layer.push(
      jsx('div', { className: 'hm-legend', style: { left: geo.house.x + 10 + 'px', top: geo.house.y + 'px', opacity: 1 - eTarget }, children: 'Household · one conversation' }, 'legend')
    )
    layer.push(
      jsx('div', { className: 'hm-hublbl', style: { left: geo.hub.x - 6 + 'px', top: geo.hub.y + 'px' }, children: `hub · ${hostOf(data.hub && data.hub.url)}` }, 'hub')
    )
    layer.push(jsx(Tray, { geo, data, onGrab }, 'tray'))
  }
  return jsxs('div', {
    className: 'hm-ear',
    ref: boxRef,
    style: { '--hm-live': (m.colors && m.colors.live) || 'var(--ui-accent)' },
    children: [
      jsx('canvas', { ref: cvRef, 'aria-hidden': true }),
      jsx('span', { ref: probeRef, 'aria-hidden': true, className: 'hm-probe' }),
      jsx('div', { className: 'hm-layer', children: layer }),
      drag ? jsx('div', { className: 'hm-dragcover' }) : null,
      drag ? jsx('div', { className: 'hm-dragdot', style: { left: drag.cx + 'px', top: drag.cy + 'px' } }) : null
    ]
  })
}

function slugish(name) {
  return String(name || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function Announce({ data }) {
  const [text, setText] = useState('')
  const [picked, setPicked] = useState(null)
  const [busy, setBusy] = useState(false)
  const live = (data.rooms || []).filter(r => r.online).map(r => r.room)
  const targets = picked || live
  const toggle = room => {
    const cur = new Set(targets)
    cur.has(room) ? cur.delete(room) : cur.add(room)
    setPicked([...cur])
  }
  const send = async () => {
    const body = text.trim()
    if (!body || busy || !targets.length) return
    setBusy(true)
    const all = targets.length === (data.rooms || []).length
    const result = await store.act('announce', { text: body, rooms: all ? 'all' : targets })
    setBusy(false)
    if (result && result.ok) {
      setText('')
      const to = new Set(result.rooms || targets)
      store.pulse((data.devices || []).filter(d => to.has(d.room) && d.online).map(d => d.device_id))
      const where = [...to].map(r => roomName(data, r)).join(', ')
      const off = (result.offline || []).map(r => roomName(data, r)).join(', ')
      host.notify({ kind: 'success', title: 'Housemate', message: `Announced in ${where}.${off ? ` No device online in ${off}.` : ''}` })
    }
  }
  return jsxs('section', {
    children: [
      jsx(SectionTitle, { children: 'Announce' }),
      jsx(Textarea, {
        value: text,
        rows: 2,
        placeholder: 'Dinner’s ready.',
        'aria-label': 'Announcement',
        onChange: ev => setText(ev.target.value),
        onKeyDown: ev => {
          if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) {
            ev.preventDefault()
            send()
          }
        }
      }),
      jsx('div', {
        className: 'hm-targets',
        children: (data.rooms || []).map(r => {
          const on = targets.includes(r.room)
          return jsx(
            Button,
            {
              variant: on ? 'secondary' : 'outline',
              size: 'xs',
              'aria-pressed': on,
              onClick: () => toggle(r.room),
              children: [on ? jsx(Codicon, { name: 'check', size: 11 }, 'i') : null, r.name]
            },
            r.room
          )
        })
      }),
      jsxs('div', {
        className: 'hm-to',
        children: ['To: ', jsx('b', { children: targets.length ? targets.map(r => roomName(data, r)).join(', ') : 'pick a room' })]
      }),
      jsxs('div', {
        className: 'hm-between',
        children: [
          jsxs('span', { className: 'hm-row', children: [jsx(Kbd, { children: 'Ctrl' }), jsx(Kbd, { children: '↵' })] }),
          jsx(Button, { onClick: send, disabled: busy || !text.trim() || !targets.length, children: busy ? 'Announcing…' : 'Announce' })
        ]
      })
    ]
  })
}

function QrCode({ qr }) {
  const { resolvedMode } = useTheme()
  if (!qr || !qr.path) return null
  // Always dark modules on a light ground, in theme colours: phone cameras read that reliably.
  const ink = resolvedMode === 'dark' ? 'var(--ui-bg-chrome)' : 'var(--ui-text-primary)'
  const paper = resolvedMode === 'dark' ? 'var(--ui-text-primary)' : 'var(--ui-bg-chrome)'
  return jsxs('svg', {
    width: 88,
    height: 88,
    viewBox: `0 0 ${qr.size} ${qr.size}`,
    shapeRendering: 'crispEdges',
    role: 'img',
    'aria-label': `QR code of ${qr.text}`,
    children: [jsx('rect', { width: qr.size, height: qr.size, style: { fill: paper } }), jsx('path', { d: qr.path, style: { fill: ink } })]
  })
}

function AddDevice({ data }) {
  const hub = data.hub || {}
  const pending = data.pending || []
  const unplaced = (data.devices || []).filter(d => !d.room)
  return jsxs('section', {
    children: [
      jsx(SectionTitle, { children: 'Add a device' }),
      jsxs('div', {
        className: 'hm-qr',
        children: [
          jsx(QrCode, { qr: data.qr }),
          jsx('p', {
            className: 'hm-small',
            style: { margin: 0 },
            children: 'Scan it with the old phone or laptop, or type the address. Check the certificate matches, then approve the code it shows.'
          })
        ]
      }),
      hub.url
        ? jsxs(Fragment, {
            children: [
              jsx('div', { className: 'hm-label', children: 'Hub address' }),
              jsxs('div', {
                className: 'hm-row',
                children: [jsx('span', { className: 'hm-mono', children: hub.url }), jsx(CopyButton, { text: hub.url, label: 'Copy address', appearance: 'icon', buttonSize: 'icon-xs', buttonVariant: 'ghost' })]
              })
            ]
          })
        : null,
      hub.fingerprint
        ? jsxs(Fragment, {
            children: [
              jsx('div', { className: 'hm-label', children: 'Certificate · SHA-256' }),
              jsxs('div', {
                className: 'hm-row',
                children: [
                  jsx('span', { className: 'hm-mono', title: hub.fingerprint, children: shortFingerprint(hub.fingerprint) }),
                  jsx(CopyButton, { text: hub.fingerprint, label: 'Copy fingerprint', appearance: 'icon', buttonSize: 'icon-xs', buttonVariant: 'ghost' })
                ]
              })
            ]
          })
        : null,
      jsx('div', { style: { height: 18 } }),
      jsx(SectionTitle, { children: `Pending requests (${pending.length})` }),
      pending.length
        ? pending.map(p =>
            jsxs(
              'div',
              {
                className: 'hm-req',
                children: [
                  jsxs('div', { className: 'hm-req-name', children: [jsx(StatusDot, { tone: 'warn' }), p.name] }),
                  jsxs('div', {
                    className: 'hm-req-meta',
                    children: [
                      jsx('span', { className: 'hm-mono', children: fmtCode(p.code) }),
                      p.wants ? ` · wants ${p.wants}` : '',
                      ` · ${p.ip} · expires in ${mmss(p.expires_in_s)}`
                    ]
                  }),
                  jsxs('div', {
                    className: 'hm-row',
                    style: { marginLeft: 14 },
                    children: [
                      jsx(Button, { variant: 'secondary', size: 'sm', onClick: () => approve(p, p.wants || null, data), children: 'Approve' }),
                      jsx(Button, { variant: 'ghost', size: 'sm', onClick: () => deny(p), children: 'Deny' })
                    ]
                  })
                ]
              },
              p.code
            )
          )
        : jsx('div', { className: 'hm-note', style: { marginTop: 0 }, children: 'No device is waiting. A new one shows a 6-digit code; it appears here.' }),
      unplaced.length
        ? jsx('div', { className: 'hm-note', children: `${unplaced.map(d => d.name).join(', ')} paired but not in a room yet: drag ${unplaced.length > 1 ? 'them' : 'it'} into one.` })
        : null
    ]
  })
}

function HubFacts({ data }) {
  const hub = data.hub || {}
  const online = (data.devices || []).filter(d => d.online).length
  const rows = [
    ['Host', hostOf(hub.url)],
    ['Wake word', prettyWake(hub.wake_word) ? `“${prettyWake(hub.wake_word)}”` : '—'],
    ['Conversation', data.mode === 'per_room' ? 'Per room' : 'Household'],
    ['Devices', `${online} online · ${(data.devices || []).length} paired`]
  ]
  return jsxs('section', {
    children: [
      jsx(SectionTitle, { children: 'Hub' }),
      jsx('dl', { className: 'hm-facts', style: { margin: 0 }, children: rows.flatMap(([k, v]) => [jsx('dt', { children: k }, k + 't'), jsx('dd', { children: v }, k + 'd')]) }),
      jsx('div', {
        className: 'hm-note',
        children:
          data.mode === 'per_room'
            ? 'Each room keeps its own conversation with Hermes.'
            : 'One conversation across every room: ask in the Office about a timer set in the Kitchen.'
      })
    ]
  })
}

function ModeSwitch({ data, pending, onChange }) {
  const value = pending || data.mode || 'household'
  return jsx(WorkspacePageHeaderControl, {
    id: 'housemate-mode',
    children: jsx(SegmentedControl, { options: MODES, value, onChange, sizing: 'content' })
  })
}

function RoomsPage() {
  const data = useValue(store.$data)
  const status = useValue(store.$status)
  const rootRef = useRef(null)
  const size = useSize(rootRef)
  const [confirm, setConfirm] = useState(null)
  const [modeBusy, setModeBusy] = useState(null)
  const [modeNote, setModeNote] = useState(null)
  useEffect(() => store.pageOpen(), [])
  useEffect(() => {
    confirmSetter = setConfirm
    return () => {
      if (confirmSetter === setConfirm) confirmSetter = null
    }
  }, [])
  useTick(1000)

  const changeMode = async next => {
    if (!data || next === (data.mode || 'household') || modeBusy) return
    haptic('tap')
    setModeBusy(next)
    const result = await store.act('mode', { mode: next })
    setModeBusy(null)
    if (result && result.ok) {
      const label = next === 'per_room' ? 'Per room' : 'Household'
      const saved = result.persisted
        ? 'Saved in config.yaml, so the next gateway start keeps it.'
        : `Not saved (${result.persist_error || 'config.yaml not writable'}): the next gateway start goes back to the configured mode.`
      setModeNote({
        text: `${label} mode applies from the next question; a running answer finishes as it was. ${saved}`,
        at: Date.now(),
        sticky: !result.persisted
      })
    }
  }

  const narrow = size.w > 0 && size.w < 1000
  const style = jsx('style', { children: CSS })
  if (!data) {
    const err = status.error
    return jsxs('div', {
      className: 'hm-page',
      ref: rootRef,
      children: [
        style,
        jsxs('div', { className: 'hm-head', children: [jsx('span', { className: 'hm-title', children: 'Rooms' })] }),
        jsx('div', {
          className: 'hm-center',
          children: err
            ? [
                jsx('div', { children: `Can’t reach the Housemate hub: ${err}` }, 'm'),
                jsx('div', { className: 'hm-note', children: 'Start the gateway with the housemate platform enabled (and the plugin in plugins.enabled), then try again.' }, 'n'),
                jsx(Button, { variant: 'outline', size: 'sm', onClick: () => store.refresh(), children: 'Try again' }, 'b')
              ]
            : jsx(Loader, { label: 'Loading rooms' })
        })
      ]
    })
  }
  const onlineRooms = (data.rooms || []).filter(r => r.online).length
  const wake = prettyWake(data.hub && data.hub.wake_word)
  const sub = [hostOf(data.hub && data.hub.url), wake ? `“${wake}”` : null, `${onlineRooms} room${onlineRooms === 1 ? '' : 's'} live`]
    .filter(Boolean)
    .join(' · ')
  return jsxs('div', {
    className: 'hm-page' + (narrow ? ' hm-narrow' : ''),
    ref: rootRef,
    children: [
      style,
      jsxs('div', {
        className: 'hm-head',
        children: [
          jsx('span', { className: 'hm-title', children: 'Rooms' }),
          jsx('span', { className: 'hm-sub', children: sub }),
          jsx('span', { className: 'hm-grow' }),
          status.error ? jsx('span', { className: 'hm-sub', style: { color: 'var(--ui-yellow)' }, children: 'hub not answering · showing the last state' }) : null,
          jsx(ModeSwitch, { data, pending: modeBusy, onChange: changeMode })
        ]
      }),
      jsx('p', {
        className: 'hm-cap',
        children:
          modeNote && (modeNote.sticky || Date.now() - modeNote.at < 12000)
            ? modeNote.text
            : 'Circles show how far each device hears. Overlapping devices settle who answers.'
      }),
      jsxs('div', {
        className: 'hm-body',
        children: [
          jsx(Earshot, { data, mode: modeBusy || data.mode || 'household' }),
          jsxs('div', { className: 'hm-col', children: [jsx(Announce, { data }), jsx(AddDevice, { data }), jsx(HubFacts, { data })] })
        ]
      }),
      confirm
        ? jsx(ConfirmDialog, {
            open: true,
            onClose: () => setConfirm(null),
            onConfirm: async () => {
              await confirm.onConfirm()
            },
            title: confirm.title,
            description: confirm.description,
            confirmLabel: confirm.confirmLabel,
            dismissOnConfirm: true
          })
        : null
    ]
  })
}

function StatusChip() {
  const data = useValue(store.$data)
  const status = useValue(store.$status)
  useTick(1000)
  if (!data) return null
  const ringing = (data.timers || []).find(t => t.state === 'ringing')
  const rooms = (data.rooms || []).filter(r => r.online).length
  const timers = (data.timers || []).length
  let text = `${rooms} room${rooms === 1 ? '' : 's'} · ${timers} timer${timers === 1 ? '' : 's'}`
  let tone = rooms ? 'good' : 'muted'
  if (ringing) {
    text = `${ringing.label ? cap(ringing.label) : 'Timer'} ringing · ${roomName(data, ringing.room)}`
    tone = 'warn'
  } else if (status.error) {
    text = 'Rooms · hub offline'
    tone = 'muted'
  }
  return jsxs('button', {
    type: 'button',
    className: 'hm-chipbar' + (ringing ? ' is-warn' : ''),
    title: 'Open Rooms',
    onClick: () => host.navigate(ROUTE),
    children: [jsx('style', { children: '.hm-chipbar{display:flex;align-items:center;gap:6px;height:100%;padding:0 6px;font-size:.6875rem;color:var(--ui-text-tertiary);background:transparent;border:0;cursor:pointer;white-space:nowrap}.hm-chipbar:hover{color:var(--ui-text-secondary)}.hm-chipbar.is-warn{color:var(--ui-yellow);font-weight:600}' }), jsx(StatusDot, { tone }), text]
  })
}

// ─────────────────────────────────────────────────────────────── plugin

export default {
  id: PLUGIN_ID,
  name: 'Housemate',
  register(ctx) {
    store = createStore(ctx)
    ctx.onDispose(() => {
      store = null
      confirmSetter = null
    })
    ctx.registerMany([
      { id: 'page', area: ROUTES_AREA, data: { path: ROUTE }, render: () => jsx(RoomsPage, {}) },
      { id: 'nav', area: SIDEBAR_NAV_AREA, data: { path: ROUTE, label: 'Rooms', codicon: 'home' } },
      { id: 'chip', area: STATUSBAR_AREAS.right, order: 125, render: () => jsx(StatusChip, {}) },
      {
        id: 'open',
        area: PALETTE_AREA,
        data: { id: 'housemate.rooms.open', label: 'Open Rooms', keywords: ['housemate', 'rooms', 'speakers', 'timers', 'announce'], run: () => host.navigate(ROUTE) }
      }
    ])
  }
}
