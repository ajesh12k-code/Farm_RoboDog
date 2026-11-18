// Operator input: keyboard, on-screen D-pad, sliders. Pure UI -> message callbacks; no networking.

const HOLD_REFRESH_MS = 200;  // contract: senders re-publish a held command at >= 5 Hz

// key code -> [vx, vy, yaw]   (vy + = left, yaw + = turn left / CCW)
const DRIVE_KEYS = {
  KeyW: [1, 0, 0], ArrowUp: [1, 0, 0],
  KeyS: [-1, 0, 0], ArrowDown: [-1, 0, 0],
  KeyA: [0, 1, 0],
  KeyD: [0, -1, 0],
  KeyQ: [0, 0, 1], ArrowLeft: [0, 0, 1],
  KeyE: [0, 0, -1], ArrowRight: [0, 0, -1],
};

const ACTION_KEYS = {
  Space: 'estop', KeyX: 'stop', KeyH: 'auto_harvest', KeyR: 'reset_world', KeyG: 'toggle_mode',
  Escape: 'release_arm', KeyV: 'envelope', KeyF: 'follow',
};

const clamp1 = (v) => Math.max(-1, Math.min(1, v));

/**
 * Combines every active drive source (keys, D-pad pointers) into one held cmd_vel.
 * Any source pressed => {"hold": true} re-sent every 200 ms; last source released => release.
 */
export class DriveInput {
  constructor(sendCmd) {
    this.sendCmd = sendCmd;
    this.sources = new Map();
    this.timer = null;
  }

  press(id, axes) {
    this.sources.set(id, axes);
    this._emit();
  }

  release(id) {
    if (this.sources.delete(id)) this._emit();
  }

  releaseAll() {
    if (this.sources.size === 0 && this.timer === null) return;
    this.sources.clear();
    this._emit();
  }

  get active() { return this.sources.size > 0; }

  _current() {
    let vx = 0; let vy = 0; let yaw = 0;
    for (const [x, y, w] of this.sources.values()) { vx += x; vy += y; yaw += w; }
    return { vx: clamp1(vx), vy: clamp1(vy), yaw: clamp1(yaw) };
  }

  _emit() {
    if (this.sources.size === 0) {
      clearInterval(this.timer);
      this.timer = null;
      this.sendCmd({ vx: 0, vy: 0, yaw: 0, hold: false });  // release: finish the cycle, stop
      return;
    }
    this.sendCmd({ ...this._current(), hold: true });
    if (this.timer === null) {
      this.timer = setInterval(() => this.sendCmd({ ...this._current(), hold: true }), HOLD_REFRESH_MS);
    }
  }
}

const isTyping = (el) => el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable);
// Text entry (the token dialog): no robot hotkeys at all, so typing a token never drives the robot.
const isTextEntry = (el) => el && (el.tagName === 'TEXTAREA' || el.isContentEditable
  || (el.tagName === 'INPUT' && !['range', 'checkbox', 'button'].includes(el.type)));

/** Keyboard: drive keys hold while down; action keys fire once per press. */
export function bindKeyboard(drive, onAction) {
  window.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (isTextEntry(e.target)) return;
    if (isTyping(e.target) && e.code.startsWith('Arrow')) return;  // let a focused slider use arrows
    if (DRIVE_KEYS[e.code]) {
      e.preventDefault();
      if (!e.repeat) drive.press(`key:${e.code}`, DRIVE_KEYS[e.code]);
    } else if (ACTION_KEYS[e.code]) {
      e.preventDefault();
      if (!e.repeat) onAction(ACTION_KEYS[e.code]);
    }
  });
  window.addEventListener('keyup', (e) => drive.release(`key:${e.code}`));
  // A key released while the tab is unfocused never fires keyup: never leave the robot walking.
  window.addEventListener('blur', () => drive.releaseAll());
  document.addEventListener('visibilitychange', () => { if (document.hidden) drive.releaseAll(); });
}

/** D-pad buttons with data-drive="vx,vy,yaw": pointer down = hold, up/cancel/leave = release. */
export function bindDpad(root, drive) {
  for (const btn of root.querySelectorAll('[data-drive]')) {
    const axes = btn.dataset.drive.split(',').map(Number);
    const id = `pad:${btn.dataset.drive}`;
    const up = () => { btn.classList.remove('pressed'); drive.release(id); };
    btn.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      if (btn.disabled) return;  // viewer role
      btn.setPointerCapture(e.pointerId);
      btn.classList.add('pressed');
      drive.press(id, axes);
    });
    btn.addEventListener('pointerup', up);
    btn.addEventListener('pointercancel', up);
    btn.addEventListener('lostpointercapture', up);
    btn.addEventListener('contextmenu', (e) => e.preventDefault());
  }
}

/**
 * A labelled range slider row inside a .sliders grid. onInput(value) is throttled to ~10 Hz
 * with a trailing call, so the final position is always sent.
 */
export function addSlider(grid, { label, min, max, step, value, digits = 2, onInput }) {
  const lab = document.createElement('label');
  lab.textContent = label;
  const input = document.createElement('input');
  Object.assign(input, { type: 'range', min, max, step, value });
  const out = document.createElement('output');
  out.textContent = Number(value).toFixed(digits);
  grid.append(lab, input, out);

  let last = 0;
  let pending = null;
  const fire = () => { last = performance.now(); pending = null; onInput(Number(input.value)); };
  input.addEventListener('input', () => {
    out.textContent = Number(input.value).toFixed(digits);
    if (pending) return;
    const wait = Math.max(0, 100 - (performance.now() - last));
    pending = setTimeout(fire, wait);
  });

  let touchedAt = -Infinity;
  input.addEventListener('pointerdown', () => { touchedAt = performance.now(); });
  input.addEventListener('input', () => { touchedAt = performance.now(); });
  return {
    input,
    /** Set from remote state unless the operator touched the slider in the last 2 s. */
    sync(v) {
      if (performance.now() - touchedAt < 2000 || !Number.isFinite(v)) return false;
      input.value = v;
      out.textContent = Number(input.value).toFixed(digits);
      return true;
    },
    set(v) {
      input.value = v;
      out.textContent = Number(input.value).toFixed(digits);
    },
  };
}

/**
 * E-STOP button: fires on pointerdown (first touch, no 300 ms click delay, works with gloves on
 * a resistive screen as a tap) and on keyboard activation; never twice for one press.
 */
export function bindEstop(btn, fire) {
  let downAt = -Infinity;
  btn.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    downAt = performance.now();
    btn.classList.add('pressed');
    fire();
  });
  const up = () => btn.classList.remove('pressed');
  btn.addEventListener('pointerup', up);
  btn.addEventListener('pointercancel', up);
  btn.addEventListener('pointerleave', up);
  btn.addEventListener('click', () => {
    if (performance.now() - downAt > 600) fire();  // Enter / assistive tech
    btn.blur();
  });
  btn.addEventListener('contextmenu', (e) => e.preventDefault());
}
