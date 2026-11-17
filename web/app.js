// Atharv Nandi browser console: wires the bridge connection, 3D view, HUD and operator input.
import { BridgeConnection, bridgeUrl } from './net.js';
import { SceneView } from './scene3d.js';
import { DriveInput, addSlider, bindDpad, bindKeyboard } from './controls.js';

const $ = (id) => document.getElementById(id);
const ARM_SLIDERS = [  // [label, index into the 7-DOF arm vector, step, digits]
  ['Gantry', 0, 0.01, 2], ['Yaw', 1, 0.05, 2], ['Shoulder', 2, 0.05, 2],
  ['Elbow', 3, 0.05, 2], ['Wrist', 4, 0.05, 2], ['Gripper', 5, 0.001, 3],
];
const GAIT_LABELS = ['Speed', 'Step height', 'Freq (Hz)', 'Balance'];

const view = new SceneView($('view'));
const ui = { scene: null, armHome: null, armValues: null, gait: [], gaitValues: null, envelopeShown: false, rttMs: null };

const conn = new BridgeConnection(bridgeUrl(), {
  onOpen: () => { $('banner').hidden = true; },
  onClose: () => {
    $('banner').hidden = false;
    drive.releaseAll();  // the bridge already released the walk when the socket dropped
  },
  onMessage: handleMessage,
});
const drive = new DriveInput((cmd) => conn.send({ type: 'cmd', ...cmd }));

// ---------------------------------------------------------------- messages

function handleMessage(msg) {
  switch (msg.type) {
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

function onScene(scene) {
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
  const game = s.game; const loco = s.loco; const harv = s.harvest; const t = s.telemetry;

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

  if (t) {
    setText('hud-pos', `x ${signed(t.x)}  y ${signed(t.y)}  hdg ${Math.round(t.heading_deg)}°`);
    setText('hud-tilt', `${t.tilt_deg.toFixed(1)}°  height ${t.z.toFixed(2)} m`,
      t.tilt_deg > 25 ? 'bad' : t.tilt_deg > 12 ? 'amber' : '');
  }

  const rtt = ui.rttMs === null ? '' : `${Math.round(ui.rttMs)} ms`;
  if (s.sim_age_s === null || s.sim_age_s === undefined) setText('hud-link', `${rtt}  waiting for sim`, 'amber');
  else if (s.sim_age_s > 1.0) setText('hud-link', `${rtt}  sim stalled (${s.sim_age_s.toFixed(0)} s)`, 'bad');
  else setText('hud-link', rtt || 'connected', 'good');
}

// ---------------------------------------------------------------- actions

function onAction(name) {
  switch (name) {
    case 'stop':
      drive.releaseAll();  // otherwise a held key would restart the walk
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

for (const btn of document.querySelectorAll('[data-action]')) {
  btn.addEventListener('click', () => { onAction(btn.dataset.action); btn.blur(); });
}
$('btn-envelope').addEventListener('click', (e) => { onAction('envelope'); e.currentTarget.blur(); });
$('btn-follow').addEventListener('click', (e) => { onAction('follow'); e.currentTarget.blur(); });
$('side-toggle').addEventListener('click', () => {
  const side = $('side');
  side.classList.toggle('collapsed');
  $('side-toggle').setAttribute('aria-expanded', String(!side.classList.contains('collapsed')));
});
if (window.matchMedia('(max-width: 760px)').matches) $('side').classList.add('collapsed');

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
conn.connect();
