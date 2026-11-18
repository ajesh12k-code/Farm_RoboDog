// Pilot panels: safety banner, mission, battery, camera, node health, run/KPIs.
// Every input comes from an optional package (safety / mission / perception / telemetry), so every
// renderer accepts null / missing fields and shows 'n/a' instead of failing.
import { t } from './i18n.js';

const $ = (id) => document.getElementById(id);
const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const pick = (obj, ...keys) => {
  if (!obj || typeof obj !== 'object') return undefined;
  for (const k of keys) if (obj[k] !== undefined && obj[k] !== null) return obj[k];
  return undefined;
};

function setText(id, text, cls) {
  const el = $(id);
  if (!el) return;
  if (el.textContent !== text) el.textContent = text;
  if (cls !== undefined && el.className !== cls) el.className = cls;
}

function setBar(id, frac) {
  const el = $(id);
  if (!el) return;
  const f = frac === null ? 0 : Math.max(0, Math.min(1, frac));
  el.style.width = `${(f * 100).toFixed(0)}%`;
  el.className = frac === null ? '' : f < 0.2 ? 'bad' : f < 0.4 ? 'amber' : 'good';
}

/** Latched safety banner: safety_status from the supervisor, else the console's own latch. */
export function renderSafety(safety, consoleLatch) {
  const supLatched = Boolean(pick(safety, 'estop', 'latched', 'estopped', 'estop_latched'));
  const latched = supLatched || Boolean(consoleLatch?.latched);
  $('safety-banner').hidden = !latched;
  document.body.classList.toggle('estopped', latched);
  if (!latched) return latched;
  const reason = pick(safety, 'reason', 'estop_reason', 'cause') ?? consoleLatch?.message ?? '';
  setText('safety-reason', String(reason));
  const st = num(pick(safety, 'stop_t_s', 'stop_time_s', 'stop_t'));
  const sd = num(pick(safety, 'stop_dist_m', 'stop_distance_m', 'stop_dist'));
  const parts = [];
  parts.push(`${t('stop_t')} ${st === null ? t('na') : `${st.toFixed(2)} s`}`);
  parts.push(`${t('stop_dist')} ${sd === null ? t('na') : `${sd.toFixed(2)} m`}`);
  if (!safety) parts.push('(no safety supervisor)');
  setText('safety-stop', parts.join(' · '));
  return latched;
}

export function renderBattery(ext) {
  const b = ext?.battery;
  let pct = num(pick(b, 'percent', 'battery_pct', 'soc_pct'));
  const frac = num(pick(b, 'percentage'));  // sensor_msgs/BatteryState convention: 0..1
  if (pct === null && frac !== null) pct = frac <= 1 ? frac * 100 : frac;
  if (pct === null) pct = num(pick(ext?.safety, 'battery_pct', 'battery_percent'));
  if (pct === null) pct = num(pick(ext?.telemetry, 'battery_pct', 'battery_percent'));
  setBar('battery-bar', pct === null ? null : pct / 100);
  setText('battery-text', pct === null ? t('na') : `${Math.round(pct)}%`,
    pct === null ? 'muted' : pct < 20 ? 'bad' : pct < 40 ? 'amber' : '');
}

export function renderMission(mission, game) {
  const state = pick(mission, 'state', 'status', 'phase');
  setText('mission-state', state === undefined ? t('na') : String(state), state === undefined ? 'muted' : '');
  let prog = num(pick(mission, 'progress', 'progress_frac'));
  if (prog !== null && prog > 1) prog /= 100;
  setBar('mission-bar', prog);
  setText('mission-progress', prog === null ? t('na') : `${Math.round(prog * 100)}%`);
  const picked = num(pick(mission, 'picked', 'n_picked'));
  const attempted = num(pick(mission, 'attempted', 'n_attempted'));
  setText('mission-picked', picked === null && attempted === null ? t('na')
    : `${picked ?? t('na')} / ${attempted ?? t('na')}`);
  const basket = pick(mission, 'basket');
  let n = num(pick(basket, 'n', 'count')) ?? num(pick(mission, 'basket_n'));
  const cap = num(pick(basket, 'capacity', 'cap')) ?? num(pick(mission, 'basket_capacity'));
  if (n === null && game) n = num(game.in_basket);
  setText('mission-basket', n === null && cap === null ? t('na') : `${n ?? t('na')} / ${cap ?? t('na')}`);
}

export function renderRun(ext) {
  const tel = ext?.telemetry;
  const runId = pick(tel, 'run_id') ?? pick(ext?.mission, 'run_id');
  setText('run-id', runId === undefined ? t('na') : String(runId).slice(0, 32), runId === undefined ? 'muted' : '');
  const kpis = pick(tel, 'kpis', 'counters');
  const box = $('kpis');
  const entries = kpis && typeof kpis === 'object' ? Object.entries(kpis).slice(0, 8) : [];
  const html = entries.map(([k, v]) => `${k}: ${typeof v === 'number' ? +v.toFixed(3) : v}`).join('  ·  ');
  if (box.textContent !== html) box.textContent = html;
  const dets = ext?.detections?.items;
  setText('det-count', Array.isArray(dets) ? String(dets.length) : t('na'), Array.isArray(dets) ? '' : 'muted');
}

const HEALTH_ORDER = ['sim', 'loco', 'harvest', 'game', 'safety', 'mission', 'detections', 'telemetry', 'battery', 'camera'];

/** Coloured chips per node / topic: ok = green, warn = amber, fail = red, n/a = grey. */
export function renderHealth(health) {
  const box = $('health');
  const h = health && typeof health === 'object' ? health : {};
  const names = [...HEALTH_ORDER.filter((k) => k in h), ...Object.keys(h).filter((k) => !HEALTH_ORDER.includes(k))];
  const key = names.map((n) => `${n}:${h[n]}`).join('|');
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.replaceChildren(...names.map((n) => {
    const chip = document.createElement('span');
    const level = ['ok', 'warn', 'fail'].includes(h[n]) ? h[n] : 'na';
    chip.className = `chip ${level}`;
    chip.textContent = n;
    chip.title = `${n}: ${h[n] ?? 'n/a'}`;
    return chip;
  }));
}

let lastCamAt = 0;
export function renderCamera(msg) {
  const img = $('cam-img');
  if (msg && typeof msg.data === 'string' && (msg.format === 'jpeg' || msg.format === 'png')) {
    img.src = `data:image/${msg.format};base64,${msg.data}`;
    img.hidden = false;
    $('cam-na').hidden = true;
    lastCamAt = performance.now();
  }
}

/** Grey the thumbnail out when frames stop (camera node gone). */
export function ageCamera() {
  const stale = performance.now() - lastCamAt > 5000;
  $('cam-img').classList.toggle('stale', stale);
  if (stale && lastCamAt === 0) $('cam-na').hidden = false;
}
