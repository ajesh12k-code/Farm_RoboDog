// Atharv Nandi browser console: wires the bridge connection, 3D view, orchard console (WP10), HUD and operator input.
import { BridgeConnection, bridgeUrl, loadToken, saveToken } from './net.js';
import { SceneView } from './scene3d.js';
import { DriveInput, addSlider, bindDpad, bindEstop, bindKeyboard } from './controls.js';
import { HI_REVIEW_NOTE, applyI18n, getLang, t, toggleLang } from './i18n.js';
import { ageCamera, renderBattery, renderCamera, renderHealth, renderMission, renderRun, renderSafety } from './panels.js';
import { OrchardConsole } from './orchard.js';

const $ = (id) => document.getElementById(id);
const ARM_SLIDERS = [  // [label, index into the 7-DOF arm vector, step, digits]
  ['Gantry', 0, 0.01, 2], ['Yaw', 1, 0.05, 2], ['Shoulder', 2, 0.05, 2],
  ['Elbow', 3, 0.05, 2], ['Wrist', 4, 0.05, 2], ['Gripper', 5, 0.001, 3],
];
const GAIT_LABELS = ['Speed', 'Step height', 'Freq (Hz)', 'Balance'];

const view = new SceneView($('view'));
const ui = { role: null, authRequired: false, devMode: false, token: loadToken(), scene: null, armHome: null, armValues: null, gait: [], gaitValues: null, envelopeShown: false, rttMs: null };

const conn = new BridgeConnection(bridgeUrl(), {
  getToken: () => ui.token,
  onOpen: () => { $('banner').hidden = true; },
  onClose: () => {
    ui.helloSeen = false;
    $('banner').hidden = false;
    drive.releaseAll();  // the bridge already released the walk when the socket dropped
  },
  onMessage: handleMessage,
});
const drive = new DriveInput((cmd) => conn.send({ type: 'cmd', ...cmd }));
const orchard = new OrchardConsole((msg) => {
  if (!conn.send(msg)) toast(t('not_connected'), false);
}, (text, ok) => toast(text, ok));

// ---------------------------------------------------------------- views (tabs)

function setTab(tab) {
  document.body.dataset.tab = tab;
  for (const b of document.querySelectorAll('#tabs [data-tab]')) b.classList.toggle('on', b.dataset.tab === tab);
  if (tab === 'map') requestAnimationFrame(() => { orchard._fitMap(); orchard.drawMap(); });
}
for (const b of document.querySelectorAll('#tabs [data-tab]')) b.addEventListener('click', () => setTab(b.dataset.tab));
document.addEventListener('nandi-tab', (e) => setTab(e.detail));

// ---------------------------------------------------------------- messages

function handleMessage(msg) {
  if (orchard.onMessage(msg)) return;
  switch (msg.type) {
    case 'hello': onHello(msg); break;
    case 'auth': onAuth(msg); break;
    case 'camera': renderCamera(msg); break;
    case 'styles': view.setStyles(msg.data); break;
    case 'scene': onScene(msg); break;
    case 'state': onState(msg); break;
    case 'envelope':
      view.setEnvelope(msg.data);
      ui.envelopeShown = Boolean(msg.data);
      $('btn-envelope').classList.toggle('on', ui.envelopeShown);
      break;
    case 'result': toast(msg.message || `${msg.name}: ${msg.success ? 'ok' : 'failed'}`, msg.success); break;
    case 'error': toast(`Bridge rejected a request: ${msg.message}`, false); break;
    case 'pong': ui.rttMs = Math.max(0, performance.now() - msg.t); break;
    default: break;
  }
}

function onHello(h) {
  ui.helloSeen = true;
  ui.authRequired = Boolean(h.auth_required);
  ui.devMode = Boolean(h.dev_mode);
  setRole(h.role ?? null);
  if (ui.authRequired && !h.role && !ui.token) askToken(t('auth_prompt'));
}

function onAuth(a) {
  if (a.ok) {
    setRole(a.role);
    const dlg = $('auth-dialog');
    if (dlg.open) dlg.close();
    return;
  }
  ui.token = null;
  saveToken(null);
  setRole(null);
  askToken(t('auth_bad'));
}

function setRole(role) {
  ui.role = role;
  const badge = $('auth-badge');
  const label = role === 'dev' ? t('dev_badge') : role === 'operator' ? t('operator')
    : role === 'viewer' ? t('viewer') : '-';
  badge.textContent = label;
  badge.className = `badge ${role ?? 'none'}`;
  const readOnly = role === 'viewer' || role === null;
  document.body.classList.toggle('viewer', readOnly);
  for (const el of document.querySelectorAll('[data-op]')) el.disabled = readOnly;
  orchard.setReadOnly(readOnly);
  if (readOnly) drive.releaseAll();
}

function askToken(message) {
  const dlg = $('auth-dialog');
  $('auth-msg').textContent = message;
  if (!dlg.open) dlg.showModal();
  $('auth-token').focus();
}

$('auth-form').addEventListener('submit', () => {
  ui.token = $('auth-token').value.trim() || null;
  $('auth-token').value = '';
  saveToken(ui.token);
  if (ui.token) conn.send({ type: 'auth', token: ui.token });
});

function onScene(scene) {
  if (!ui.helloSeen) setRole('dev');  // pre-auth bridge (no hello): legacy dev behaviour
  ui.scene = scene;
  view.build(scene);
  view.setFollow(view.follow);

  const arm = scene.arm;
  ui.armHome = arm.home.slice();
  ui.armValues = arm.home.slice();
  const armGrid = $('arm-sliders');
  armGrid.replaceChildren();
  ui.armSliders = ARM_SLIDERS.map(([label, i, step, digits]) => addSlider(armGrid, {
    label, min: arm.low[i], max: arm.high[i], step, value: arm.home[i], digits,
    onInput: (v) => {
      ui.armValues[i] = v;
      if (i === 5) ui.armValues[6] = v;  // both fingers close together
      conn.send({ type: 'manual_arm', values: ui.armValues });
    },
  }));

  const g = scene.gait;
  ui.gaitValues = g.default.slice();
  const gaitGrid = $('gait-sliders');
  gaitGrid.replaceChildren();
  ui.gait = GAIT_LABELS.map((label, i) => addSlider(gaitGrid, {
    label, min: g.low[i], max: g.high[i], step: 0.05, value: g.default[i],
    onInput: (v) => {
      ui.gaitValues[i] = v;
      conn.send({ type: 'gait_params', values: ui.gaitValues });
    },
  }));
}

function onState(s) {
  if (s.poses) view.applyPoses(s.poses);
  view.setHidden(s.hidden);
  if (s.styles) view.setStyles(s.styles);
  orchard.onState(s);
  const game = s.game; const loco = s.loco; const harv = s.harvest; const tel = s.telemetry;

  if (game) {
    view.markBasket(game.in_basket_names);
    setText('hud-crops', `${game.in_basket ?? 0} / ${game.crops_total ?? '?'} in basket`,
      game.crops_total && game.in_basket === game.crops_total ? 'good' : '');
    setText('hud-falls', String(game.falls ?? 0), game.falls ? 'amber' : '');
    const rtf = Number.isFinite(game.rtf) ? `  (RTF ${game.rtf.toFixed(2)})` : '';
    setText('hud-time', `${clock(game.elapsed_s)}${rtf}`);
    $('fallen').hidden = !game.fallen;
  } else {
    setText('hud-crops', 'no game status');
  }

  if (loco) {
    const mode = loco.mode === 'rl' ? 'RL balance' : 'Tripod';
    const policy = loco.policy_loaded ? '' : ' (no RL policy)';
    const moving = loco.walking ? (s.hold ? ' - walking (held)' : ' - walking') : '';
    setText('hud-mode', `${mode}${policy}${moving}`, loco.walking ? 'good' : '');
    if (Array.isArray(loco.gait_params) && ui.gaitValues) {
      loco.gait_params.forEach((v, i) => {
        if (!ui.gait[i] || !Number.isFinite(v)) return;
        if (ui.gait[i].sync(v)) ui.gaitValues[i] = v;
      });
    }
  } else {
    setText('hud-mode', 'locomotion offline', 'amber');
  }

  if (harv) {
    const label = { rl_balance: 'RL balance (auto)', ik_harvest: 'IK harvest', manual: 'Manual' }[harv.mode] ?? harv.mode;
    setText('hud-arm', `${label}${harv.phase ? ` - ${harv.phase}` : ''}`, harv.mode === 'ik_harvest' ? 'good' : '');
    const parts = [];
    if (harv.target) parts.push(`Target ${harv.target}`);
    if (harv.last_result) parts.push(`Last: ${harv.last_result}`);
    setText('hud-harvest', parts.join('  |  '), 'muted');
  } else {
    setText('hud-arm', 'harvest node offline', 'amber');
  }

  if (tel) {
    setText('hud-pos', `x ${signed(tel.x)}  y ${signed(tel.y)}  hdg ${Math.round(tel.heading_deg)}°`);
    setText('hud-tilt', `${tel.tilt_deg.toFixed(1)}°  height ${tel.z.toFixed(2)} m`,
      tel.tilt_deg > 25 ? 'bad' : tel.tilt_deg > 12 ? 'amber' : '');
  }

  const rtt = ui.rttMs === null ? '' : `${Math.round(ui.rttMs)} ms`;
  if (s.sim_age_s === null || s.sim_age_s === undefined) setText('hud-link', `${rtt}  waiting for sim`, 'amber');
  else if (s.sim_age_s > 1.0) setText('hud-link', `${rtt}  sim stalled (${s.sim_age_s.toFixed(0)} s)`, 'bad');
  else setText('hud-link', rtt || 'connected', 'good');

  // Pilot panels: every source is optional and renders n/a until its package lands.
  const ext = s.ext ?? {};
  renderSafety(ext.safety ?? null, s.estop ?? null);
  renderBattery(ext);
  renderMission(ext.mission ?? null, game);
  renderRun(ext);
  renderHealth(s.health);
  view.setDetections(ext.detections?.items ?? []);
}

// ---------------------------------------------------------------- actions

function onAction(name) {
  switch (name) {
    case 'estop':
      estop();
      return;
    case 'estop_reset':
      if (confirm(t('estop_confirm'))) conn.send({ type: 'service', name: 'estop_reset' });
      break;
    case 'mission_start':
      drive.releaseAll();
      conn.send({ type: 'mission_start', zone: 'zone_a' });
      break;
    case 'mission_abort':
      drive.releaseAll();
      conn.send({ type: 'service', name: 'mission_abort' });
      break;
    case 'stop':
      drive.releaseAll();  // otherwise a held key would restart the walk
      if (orchard.status?.active) conn.send({ type: 'preset_pause' });  // X also pauses a running preset
      conn.send({ type: 'service', name: 'stop' });
      conn.send({ type: 'service', name: 'release_arm' });  // same as the Tk STOP: gait stops, arm back to balance
      break;
    case 'envelope':
      conn.send({ type: 'envelope', show: !ui.envelopeShown });
      if (!ui.envelopeShown) toast('Computing work envelope...', true);
      break;
    case 'follow':
      view.setFollow(!view.follow);
      $('btn-follow').classList.toggle('on', view.follow);
      break;
    case 'release_arm':
      conn.send({ type: 'service', name });
      if (ui.armSliders) {  // the arm goes back to rest: show that on the sliders
        ui.armValues = ui.armHome.slice();
        ui.armSliders.forEach((sl, k) => sl.set(ui.armHome[ARM_SLIDERS[k][1]]));
      }
      break;
    case 'auto_harvest':
    case 'reset_world':
      drive.releaseAll();  // like the Tk panel: never walk during a pick or straight after a reset
      conn.send({ type: 'service', name });
      break;
    default:  // toggle_mode
      conn.send({ type: 'service', name });
  }
  if (!conn.connected) toast('Not connected to the sim', false);
}

/** E-STOP: drop every held drive source locally, then ask the bridge (allowed for every role). */
function estop() {
  drive.releaseAll();
  document.body.classList.add('estopped');
  if (!ui.helloSeen) conn.send({ type: 'service', name: 'stop' });  // pre-console bridge has no estop
  if (!conn.send({ type: 'service', name: 'estop' })) toast(`${t('estop')}: ${t('not_connected')}`, false);
}

for (const btn of document.querySelectorAll('[data-action]')) {
  btn.addEventListener('click', () => { onAction(btn.dataset.action); btn.blur(); });
}
$('btn-envelope').addEventListener('click', (e) => { onAction('envelope'); e.currentTarget.blur(); });
$('btn-follow').addEventListener('click', (e) => { onAction('follow'); e.currentTarget.blur(); });
bindEstop($('btn-estop'), estop);
$('btn-estop-reset').addEventListener('click', (e) => { onAction('estop_reset'); e.currentTarget.blur(); });
$('btn-mission-start').addEventListener('click', (e) => { onAction('mission_start'); e.currentTarget.blur(); });
$('btn-mission-abort').addEventListener('click', (e) => { onAction('mission_abort'); e.currentTarget.blur(); });
$('btn-lang').addEventListener('click', (e) => { toggleLang(); syncLang(); e.currentTarget.blur(); });

function syncLang() {
  $('btn-lang').textContent = getLang() === 'hi' ? 'EN' : 'हि';
  $('i18n-note').textContent = HI_REVIEW_NOTE;
  setRole(ui.role);  // re-label the role badge
  orchard.relabel();
}
applyI18n();
syncLang();

bindKeyboard(drive, onAction);
bindDpad($('drive'), drive);

// ---------------------------------------------------------------- helpers

function setText(id, text, cls = '') {
  const el = $(id);
  if (el.textContent !== text) el.textContent = text;
  if (el.className !== cls) el.className = cls;
}

function clock(sec) {
  if (!Number.isFinite(sec)) return '-';
  const s = Math.max(0, Math.floor(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

const signed = (v) => (v >= 0 ? '+' : '') + v.toFixed(2);

function toast(text, ok) {
  const el = document.createElement('div');
  el.className = ok ? 'toast' : 'toast fail';
  el.textContent = text;
  const box = $('toasts');
  box.append(el);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => el.remove(), ok ? 3500 : 6000);
}

setInterval(() => conn.send({ type: 'ping', t: performance.now() }), 2000);
setInterval(ageCamera, 1000);
conn.connect();
