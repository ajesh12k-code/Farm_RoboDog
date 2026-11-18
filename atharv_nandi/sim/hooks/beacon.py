"""WP1 beacon and handheld e-stop input (MVP-SPEC 2.5; ORCHARD-MVP-SPEC 1.1, 2.5; RTH-DOCKING E.11).

Two control-rate hooks (20 Hz each): ``handheld_estop`` (the handheld input) and ``beacon`` (``mutates_model``, the
lamp). They are separate hooks so a fault in one (the registry disables a hook that raises) never takes the other
with it. The logic is pure Python and ROS-free, unit-tested in tests/test_robot_additions.py.

Handheld e-stop input
    The wireless handheld e-stop of the field supervisor, as a console-independent input. The handheld (or its
    bridge) publishes ``/nandi/handheld_estop`` (std_msgs/String JSON) at >= 2 Hz::

        {"device": "hh-1", "boot": "a1f3", "seq": 17, "pressed": false}

    ``seq`` must increase per (device, boot); a repeated or older ``seq`` is not a new message (the bus keeps
    the newest message, so the same one is seen every tick). A handheld that restarts sends a new ``boot``.
    After a world reset the last message is NOT taken as fresh: the handheld counts as not heard (``seen``
    false) until a new message arrives. The mushroom latches on the device (twist to release), so ``pressed``
    is the device state, not an edge. A device that has been heard and then is silent for more than
    ``LINK_TIMEOUT_S`` (0.5 s, MISSION-PRESETS 8 "local safety link loss") is reported as ``link_ok: false``.
    The silence is the larger of the sim-time and the wall-time age, so a sim running slower than real time
    does not report a lost link late in wall time (a faster one is merely conservative).
    The first device name heard is the paired one: other device names are refused (counted in ``foreign``). A
    paired handheld that reboots (new ``boot``) is accepted, but a press is only released by the boot that
    reported it: after a reboot the device must report pressed, then released. So neither a second device nor a
    rebooted handheld can clear a press. A malformed message is counted once in ``rejected``, not every tick.
    The hook publishes ``/nandi/handheld_estop/state`` on every change and at 2 Hz and leaves the same dict in
    ``bus.values["handheld_estop"]``. Latching the robot's
    e-stop on it is the supervisor's job (change request to WP6: ``safety_supervisor`` subscribes this topic
    and latches on ``estop_request``; local-link loss goes to WP28). WP1 provides the input and the receiver
    site ``handheld_estop_rx``.

Beacon lamp
    Picks one operator state per tick, by priority, from (1) the runner's safe hold (``bus.values["hold"]``)
    and (2) the newest ``/nandi/beacon/request`` (``{"state": "working", "source":
    "mission"}``; any source may request a non-safety state, nobody may request ``estop`` or ``fault``).
    Red (``estop`` / ``fault``) comes ONLY from the hold: the handheld input does not colour the lamp, because
    until the supervisor latches on ``estop_request`` (WP6) a red lamp would tell the field supervisor the robot
    stopped while it keeps moving. Once WP6 latches, its hold turns the lamp red.
    It writes ``geom_rgba`` of the ``beacon`` geom (render-only, no physics effect), reports it through the
    sim-state mutation provider ``beacon`` (so renderers that rebuild the world show the same colour), and
    publishes ``/nandi/beacon`` on every change of (state, source) and at 1 Hz::

        {"state": "estop", "pattern": {"rgb": [...], "mode": "steady", "hz": 0.0}, "rgba": [...], "on": true,
         "source": "hold", "reason": "estop", "t": 12.34}

    Patterns per operator state (RTH-DOCKING E.11, MVP-SPEC 2.5; colours ASSUMED) come from
    ``DEFAULT_PATTERNS`` unless ``config/beacon.yaml`` (or ``$NANDI_BEACON_CONFIG``) overrides them under
    ``patterns:`` (the WP43 change request: patterns read from config).
"""
import math
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import mujoco

from atharv_nandi import paths
from atharv_nandi.core import sim_state

BEACON_GEOM = "beacon"
PROVIDER_KEY = "beacon"
PROVIDER_VERSION = 1
HANDHELD_TOPIC = "/nandi/handheld_estop"
HANDHELD_STATE_TOPIC = "/nandi/handheld_estop/state"
REQUEST_TOPIC = "/nandi/beacon/request"
BEACON_TOPIC = "/nandi/beacon"
BEACON_RATE_HZ = 20.0
HANDHELD_RATE_HZ = 20.0
BEACON_PUBLISH_PERIOD_S = 1.0
HANDHELD_PUBLISH_PERIOD_S = 0.5
LINK_TIMEOUT_S = 0.5            # MISSION-PRESETS 8: local safety link heartbeat > 0.5 s -> link lost
DIM = 0.15                      # lamp "off" brightness, so the housing stays visible
CONFIG_ENV = "NANDI_BEACON_CONFIG"
CONFIG_PATH = paths.config_path("beacon.yaml")

AMBER, GREEN, BLUE, RED, CYAN = (1.0, 0.62, 0.05), (0.05, 0.85, 0.2), (0.1, 0.35, 1.0), (0.9, 0.05, 0.05), \
    (0.05, 0.85, 0.9)
MODES = ("steady", "flash", "pulse", "breathe")

# Operator state -> pattern. RTH-DOCKING E.11 (Working green steady, Going home green slow pulse, You are driving
# blue steady, Stopped amber steady, red for e-stop or fault, Charging cyan breathing, pre-motion warning amber
# fast flash) plus MVP-SPEC 2.5 (blue for a tool change, flashing amber for an assist request, amber at rest).
DEFAULT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "estop": {"rgb": RED, "mode": "steady", "hz": 0.0},
    "fault": {"rgb": RED, "mode": "steady", "hz": 0.0},
    "premotion": {"rgb": AMBER, "mode": "flash", "hz": 4.0},
    "assist": {"rgb": AMBER, "mode": "flash", "hz": 1.0},
    "stopped": {"rgb": AMBER, "mode": "steady", "hz": 0.0},
    "charging": {"rgb": CYAN, "mode": "breathe", "hz": 0.25},
    "driving": {"rgb": BLUE, "mode": "steady", "hz": 0.0},
    "tool_change": {"rgb": BLUE, "mode": "steady", "hz": 0.0},
    "going_home": {"rgb": GREEN, "mode": "pulse", "hz": 0.5},
    "working": {"rgb": GREEN, "mode": "steady", "hz": 0.0},
    "idle": {"rgb": AMBER, "mode": "steady", "hz": 0.0},
}
# Highest priority first. estop / fault only come from the runner hold, never from a request or the handheld.
PRIORITY = ("estop", "fault", "premotion", "assist", "stopped", "charging", "driving", "tool_change",
            "going_home", "working", "idle")
SAFETY_STATES = ("estop", "fault")
# Runner hold reasons that are faults rather than an operator e-stop (sim_runner / hooks registry).
FAULT_HOLD_REASONS = ("hook_fault", "supervisor_lost")


# ---------------------------------------------------------------------- patterns
def _num(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def validate_pattern(p: Dict[str, Any]) -> Dict[str, Any]:
    """A pattern as ``{"rgb": (r, g, b) in [0, 1], "mode": one of MODES, "hz": >= 0 (> 0 unless steady)}``."""
    if not isinstance(p, dict):
        raise ValueError(f"pattern must be a mapping, got {type(p).__name__}")
    rgb = p.get("rgb")
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        raise ValueError(f"pattern rgb must be 3 numbers, got {rgb!r}")
    rgb = tuple(_num(c, "rgb") for c in rgb)
    if any(c < 0.0 or c > 1.0 for c in rgb):
        raise ValueError(f"pattern rgb out of [0, 1]: {rgb}")
    mode = p.get("mode", "steady")
    if mode not in MODES:
        raise ValueError(f"pattern mode {mode!r} not in {MODES}")
    hz = _num(p.get("hz", 0.0), "hz")
    if hz < 0.0 or (mode != "steady" and hz <= 0.0):
        raise ValueError(f"pattern hz {hz} invalid for mode {mode}")
    return {"rgb": rgb, "mode": mode, "hz": hz}


def intensity(pattern: Dict[str, Any], t: float) -> float:
    """Lamp brightness in [DIM, 1] at time t (s). flash: 50 % duty; pulse: 30 % duty; breathe: raised cosine."""
    mode, hz = pattern["mode"], float(pattern["hz"])
    if mode == "steady":
        return 1.0
    phase = (t * hz) % 1.0
    if mode == "flash":
        return 1.0 if phase < 0.5 else DIM
    if mode == "pulse":
        return 1.0 if phase < 0.3 else DIM
    return DIM + (1.0 - DIM) * (0.5 - 0.5 * math.cos(2.0 * math.pi * phase))   # breathe


def rgba_at(pattern: Dict[str, Any], t: float) -> Tuple[float, float, float, float]:
    k = intensity(pattern, t)
    r, g, b = pattern["rgb"]
    return (r * k, g * k, b * k, 1.0)


def load_patterns(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """DEFAULT_PATTERNS, overridden per state by ``patterns:`` in config/beacon.yaml (or $NANDI_BEACON_CONFIG).
    A missing file means the defaults; a malformed one raises (fail loudly at hook registration)."""
    pats = {k: validate_pattern(v) for k, v in DEFAULT_PATTERNS.items()}
    path = path or os.environ.get(CONFIG_ENV) or CONFIG_PATH
    if not os.path.isfile(path):
        return pats
    from atharv_nandi.core.safety_logic import parse_simple_yaml  # the repo's nested key: value subset parser
    with open(path) as f:
        doc = parse_simple_yaml(f.read())
    over = (doc or {}).get("patterns") or {}
    if not isinstance(over, dict):
        raise ValueError(f"{path}: patterns must be a mapping")
    for state, p in over.items():
        if state not in DEFAULT_PATTERNS:
            raise ValueError(f"{path}: unknown beacon state {state!r} (known: {sorted(DEFAULT_PATTERNS)})")
        pats[state] = validate_pattern(dict(DEFAULT_PATTERNS[state], **p))
    return pats


# ---------------------------------------------------------------------- handheld e-stop
class HandheldEstop:
    """Decoder of the handheld e-stop messages (pure logic; sim time in seconds, any monotonic base).

    ``wall_clock`` (e.g. ``time.monotonic``) adds a wall-time age: the link age is the larger of the two, so a
    sim slower than real time still reports a silent handheld within LINK_TIMEOUT_S of wall time."""

    def __init__(self, timeout_s: float = LINK_TIMEOUT_S, wall_clock: Optional[Callable[[], float]] = None):
        self.timeout_s = float(timeout_s)
        self.wall_clock = wall_clock
        self.reset()

    def reset(self, keep_seq: bool = False):
        """Forget the link. keep_seq: remember the pairing and the last (device, boot, seq), so the stale newest
        message the bus still holds after a world reset is not taken as a fresh heartbeat."""
        if not keep_seq:
            self.paired: Optional[str] = None     # device name of the first handheld heard
            self.device: Optional[str] = None     # "name/boot"
            self.seq: Optional[int] = None
        self.pressed = False
        self.press_boot: Optional[str] = None     # boot that reported the current press
        self.last_rx: Optional[float] = None
        self.last_rx_wall: Optional[float] = None
        self.rejected = 0
        self.foreign = 0
        self._last_bad: Any = None

    def _bad(self, msg: Any, counter: str) -> bool:
        if msg != self._last_bad:                  # the bus keeps the newest message: count it once
            setattr(self, counter, getattr(self, counter) + 1)
            self._last_bad = msg
        return False

    def update(self, msg: Any, now: float) -> bool:
        """Feed the newest message seen on the topic. Returns True if it was a new, valid message."""
        if msg is None:
            return False
        if not isinstance(msg, dict) or not isinstance(msg.get("pressed"), bool) \
                or isinstance(msg.get("seq"), bool) or not isinstance(msg.get("seq"), int):
            return self._bad(msg, "rejected")
        name, boot = str(msg.get("device", "handheld")), str(msg.get("boot", ""))
        if self.paired is not None and name != self.paired:
            return self._bad(msg, "foreign")
        device = f"{name}/{boot}"
        seq = int(msg["seq"])
        if device == self.device and self.seq is not None and seq <= self.seq:
            return False                    # same message again (the bus keeps the newest) or a replay
        self.paired, self.device, self.seq, self.last_rx = name, device, seq, float(now)
        self.last_rx_wall = self.wall_clock() if self.wall_clock is not None else None
        if msg["pressed"]:
            self.pressed, self.press_boot = True, boot
        elif not self.pressed or boot == self.press_boot:
            self.pressed, self.press_boot = False, None
        # else: a rebooted handheld reports released while a press from an earlier boot stands: keep the press
        return True

    def age(self, now: float) -> Optional[float]:
        if self.last_rx is None:
            return None
        age = max(0.0, float(now) - self.last_rx)
        if self.wall_clock is not None and self.last_rx_wall is not None:
            age = max(age, self.wall_clock() - self.last_rx_wall)
        return age

    def state(self, now: float) -> Dict[str, Any]:
        seen = self.last_rx is not None
        age = self.age(now)
        link_ok = seen and age <= self.timeout_s
        return {
            "seen": seen,
            "device": self.device,
            "seq": self.seq,
            "pressed": bool(self.pressed) if seen else False,
            "link_ok": bool(link_ok),
            "age_s": None if age is None else round(age, 3),
            # what the supervisor should do with it (WP6 / WP28 change requests)
            "estop_request": bool(seen and self.pressed),
            "local_link_lost": bool(seen and not link_ok),
            "rejected": self.rejected,
            "foreign": self.foreign,
        }


# ---------------------------------------------------------------------- operator state
def choose_state(hold: Optional[dict], request: Any) -> Tuple[str, str, str]:
    """(state, source, reason) by PRIORITY from the runner hold and the newest request.

    The handheld input is deliberately not an input here: red / amber "stopped" from an e-stop that nothing has
    latched yet would tell the field supervisor the robot stopped while it keeps moving. A malformed request
    (non-string state, any JSON shape) is ignored, never raised."""
    cands = []
    if isinstance(hold, dict):
        reason = str(hold.get("reason") or "estop")
        cands.append(("fault" if reason in FAULT_HOLD_REASONS else "estop", "hold", reason))
    if isinstance(request, dict):
        st = request.get("state")
        if isinstance(st, str) and st in DEFAULT_PATTERNS and st not in SAFETY_STATES:
            cands.append((st, str(request.get("source", "request")), str(request.get("reason", ""))))
    if not cands:
        return "idle", "default", ""
    return min(cands, key=lambda c: PRIORITY.index(c[0]))


# ---------------------------------------------------------------------- model access + provider
def beacon_geom(m) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, BEACON_GEOM)


def _pack(m) -> dict:
    g = beacon_geom(m)
    if g < 0:
        return {}
    return {"rgba": [float(v) for v in m.geom_rgba[g]]}


def _unpack(m, data: dict) -> None:
    if not data:
        return
    rgba = data.get("rgba")
    if not isinstance(rgba, (list, tuple)) or len(rgba) != 4:
        raise ValueError(f"beacon rgba must be 4 numbers, got {rgba!r}")
    vals = [_num(v, "rgba") for v in rgba]
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise ValueError(f"beacon rgba out of [0, 1]: {vals}")
    g = beacon_geom(m)
    if g < 0:
        raise ValueError("state carries a beacon colour but this model has no 'beacon' geom")
    m.geom_rgba[g] = vals


sim_state.register_provider(PROVIDER_KEY, _pack, _unpack, version=PROVIDER_VERSION)


# ---------------------------------------------------------------------- hooks
class BeaconHooks:
    """Holds the state of both hooks; ``register`` wires ``handheld_tick`` and ``beacon_tick`` into a HookRegistry
    as two hooks, so an exception in one disables only that one."""

    def __init__(self, patterns: Optional[Dict[str, Dict[str, Any]]] = None,
                 wall_clock: Optional[Callable[[], float]] = time.monotonic):
        self.patterns = patterns if patterns is not None else load_patterns()
        self.handheld = HandheldEstop(wall_clock=wall_clock)
        self._rest_rgba: Optional[list] = None
        self._state: Optional[Tuple[str, str]] = None   # (state, source) last published
        self._next_beacon_pub = 0.0
        self._hh_last: Optional[dict] = None
        self._next_hh_pub = 0.0

    def tick(self, m, d, t, bus):
        """Both parts in order (tests and single-hook callers); the registry runs them as two hooks."""
        self.handheld_tick(m, d, t, bus)
        self.beacon_tick(m, d, t, bus)

    # handheld e-stop input
    def handheld_tick(self, m, d, t, bus):
        self.handheld.update(bus.latest(HANDHELD_TOPIC), t)
        st = self.handheld.state(t)
        bus.values["handheld_estop"] = st
        # change key: the decoded link state only (age and the bad-message counters ride on the 2 Hz publish)
        key = {k: v for k, v in st.items() if k not in ("age_s", "rejected", "foreign")}
        if key != self._hh_last or t + 1e-9 >= self._next_hh_pub:
            if st["seen"]:
                bus.publish(HANDHELD_STATE_TOPIC, dict(st, t=round(float(t), 4)))
            self._hh_last = key
            self._next_hh_pub = t + HANDHELD_PUBLISH_PERIOD_S

    # beacon lamp
    def beacon_tick(self, m, d, t, bus):
        g = beacon_geom(m)
        state, source, reason = choose_state(bus.values.get("hold"), bus.latest(REQUEST_TOPIC))
        pat = self.patterns[state]
        rgba = rgba_at(pat, t)
        if g >= 0:
            if self._rest_rgba is None:
                self._rest_rgba = [float(v) for v in m.geom_rgba[g]]
            m.geom_rgba[g] = rgba
        if (state, source) != self._state or t + 1e-9 >= self._next_beacon_pub:
            bus.publish(BEACON_TOPIC, {
                "state": state, "source": source, "reason": reason,
                "pattern": {"rgb": list(pat["rgb"]), "mode": pat["mode"], "hz": pat["hz"]},
                "rgba": [round(v, 4) for v in rgba], "on": intensity(pat, t) >= 0.999,
                "t": round(float(t), 4)})
            self._state = (state, source)
            self._next_beacon_pub = t + BEACON_PUBLISH_PERIOD_S

    def handheld_reset(self, m, d):
        self.handheld.reset(keep_seq=True)
        self._hh_last, self._next_hh_pub = None, 0.0

    def beacon_reset(self, m, d):
        g = beacon_geom(m) if m is not None else -1
        if g >= 0 and self._rest_rgba is not None:
            m.geom_rgba[g] = self._rest_rgba
        self._state, self._next_beacon_pub = None, 0.0

    def reset(self, m, d):
        self.handheld_reset(m, d)
        self.beacon_reset(m, d)


def register(hooks):
    """Two hooks: ``handheld_estop`` (input; publishes the handheld state) and ``beacon`` (lamp, mutates_model).
    Separate so a fault in one (e.g. a malformed request) never disables the other."""
    b = BeaconHooks()
    hooks.register("handheld_estop", b.handheld_tick, rate_hz=HANDHELD_RATE_HZ,
                   topics_in=(HANDHELD_TOPIC,), reset=b.handheld_reset)
    hooks.register("beacon", b.beacon_tick, rate_hz=BEACON_RATE_HZ, mutates_model=True,
                   topics_in=(REQUEST_TOPIC,), reset=b.beacon_reset)
    return b
