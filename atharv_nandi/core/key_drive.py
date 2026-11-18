"""Pure operator-input logic for the control panel: keys -> /nandi/cmd_vel, plus gait sliders.

No Tkinter / ROS imports here so the mapping, the X11 autorepeat debounce and the 5 Hz
hold re-publish can be unit-tested with an injected clock (tests/test_key_drive.py).

cmd_vel contract (see README / ROS contract):
    linear.x = vx [-1, 1] (forward +), linear.y = vy [-1, 1] (left +), angular.z = yaw [-1, 1] (CCW +),
    linear.z = 1.0 hold (keep walking while commands keep arriving) | 0.0 single step.
    vx = vy = yaw = 0 with linear.z = 0 => release (finish current gait cycle, then stop).
Locomotion drops a hold after 0.5 s without cmd_vel (dead-man), so held commands are re-sent at 5 Hz.
"""
import time
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# Drive keys -> unit (vx, vy, yaw) contribution. Q/E turn left (CCW, +yaw) / right.
DRIVE_KEYS: Dict[str, Tuple[float, float, float]] = {
    "w": (1.0, 0.0, 0.0),
    "s": (-1.0, 0.0, 0.0),
    "a": (0.0, 1.0, 0.0),
    "d": (0.0, -1.0, 0.0),
    "q": (0.0, 0.0, 1.0),
    "e": (0.0, 0.0, -1.0),
}
# One-shot action keys (Tk keysyms, lower-cased) -> GUI action name.
ACTION_KEYS: Dict[str, str] = {
    "space": "stop",
    "h": "harvest",
    "r": "reset",
    "g": "gait",
    "escape": "release",
}
# Prefix for on-screen D-pad buttons so a mouse press never cancels a held keyboard key.
PAD_PREFIX = "pad:"

DEBOUNCE_S = 0.06  # X11 autorepeat sends KeyRelease+KeyPress pairs a few ms apart
REPUBLISH_S = 0.2  # 5 Hz re-publish while held (locomotion dead-man is 0.5 s)

KEY_HELP = "Keys: W/S fwd/back  A/D strafe  Q/E turn  Space stop  H harvest  R reset  G gait  Esc release arm"


class DriveCommand(NamedTuple):
    """One /nandi/cmd_vel message: velocities in [-1, 1] and the hold flag (linear.z)."""
    vx: float
    vy: float
    yaw: float
    hold: bool


RELEASE = DriveCommand(0.0, 0.0, 0.0, False)


def normalize_key(keysym: str) -> str:
    """Lower-case a Tk keysym so Caps Lock / Shift ('W') maps like 'w'; None/'' -> ''."""
    return (keysym or "").lower()


def drive_axis(key: str) -> Optional[Tuple[float, float, float]]:
    """Return the (vx, vy, yaw) unit vector of a drive key or D-pad key ('pad:w'), else None."""
    if key.startswith(PAD_PREFIX):
        key = key[len(PAD_PREFIX):]
    return DRIVE_KEYS.get(key)


def action_for_key(keysym: str) -> Optional[str]:
    """Return the one-shot action name bound to a keysym, or None."""
    return ACTION_KEYS.get(normalize_key(keysym))


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def combine_keys(keys: Iterable[str]) -> DriveCommand:
    """Sum the held drive keys into one command (W+Q = forward arc left; W+S cancels).

    Each axis is clamped to [-1, 1] (W and pad:w together is still 1.0). If every axis
    cancels to zero the result is RELEASE, never an ambiguous zero-velocity hold.
    """
    vx = vy = yaw = 0.0
    seen = set()
    for key in keys:
        axis = drive_axis(key)
        name = key[len(PAD_PREFIX):] if key.startswith(PAD_PREFIX) else key
        if axis is None or name in seen:
            continue
        seen.add(name)
        vx, vy, yaw = vx + axis[0], vy + axis[1], yaw + axis[2]
    vx, vy, yaw = _clamp(vx), _clamp(vy), _clamp(yaw)
    if vx == 0.0 and vy == 0.0 and yaw == 0.0:
        return RELEASE
    return DriveCommand(vx, vy, yaw, True)


class KeyDrive:
    """Debounced key-state tracker that decides when to publish cmd_vel.

    Usage from the GUI: ``press(keysym)`` on KeyPress (returns True only for a real, non-autorepeat
    press, so one-shot actions fire once), ``release(keysym)`` on KeyRelease, and ``poll()`` every
    ~50 ms; publish whatever ``poll()`` returns (None = nothing to send).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 debounce_s: float = DEBOUNCE_S, republish_s: float = REPUBLISH_S):
        self._clock = clock
        self._debounce_s = debounce_s
        self._republish_s = republish_s
        self._down = set()  # keys currently considered held
        self._pending: Dict[str, float] = {}  # key -> time its debounced release takes effect
        self._last_sent = RELEASE  # robot starts idle: nothing to release at startup
        self._last_sent_t = float("-inf")

    def _settle(self, now: float):
        """Apply debounced releases whose deadline has passed."""
        for key, deadline in list(self._pending.items()):
            if now >= deadline:
                self._down.discard(key)
                del self._pending[key]

    def press(self, keysym: str) -> bool:
        """Register a key press; True if fresh, False if it is an autorepeat of a held key."""
        key = normalize_key(keysym)
        if not key:
            return False
        self._settle(self._clock())
        if key in self._pending:  # release+press within the debounce window = autorepeat
            del self._pending[key]
            return False
        if key in self._down:  # repeat press without a release (some platforms)
            return False
        self._down.add(key)
        return True

    def release(self, keysym: str, immediate: bool = False):
        """Register a key release; debounced unless ``immediate`` (mouse buttons do not autorepeat)."""
        key = normalize_key(keysym)
        if key not in self._down:
            return
        if immediate:
            self._down.discard(key)
            self._pending.pop(key, None)
        else:
            self._pending[key] = self._clock() + self._debounce_s

    def clear(self):
        """Forget every held key (focus loss: KeyRelease events will not arrive any more)."""
        self._down.clear()
        self._pending.clear()

    def stop_driving(self):
        """Drop held drive keys only (STOP/harvest/reset); the next poll() publishes RELEASE if walking.

        Action keys stay tracked so holding Space/R/H autorepeats are still swallowed and the
        action does not fire again ~30 times a second.
        """
        for key in [k for k in self._down if drive_axis(k)]:
            self._down.discard(key)
            self._pending.pop(key, None)

    def held_keys(self) -> List[str]:
        """Sorted list of keys currently held (after applying expired debounced releases)."""
        self._settle(self._clock())
        return sorted(self._down)

    def command(self) -> DriveCommand:
        """Current command implied by the held drive keys."""
        return combine_keys(self.held_keys())

    def poll(self) -> Optional[DriveCommand]:
        """Return the cmd_vel to publish now, or None.

        Publishes on every change (including the final RELEASE) and re-publishes an unchanged
        hold every ``republish_s`` so the locomotion dead-man stays satisfied.
        """
        now = self._clock()
        cmd = self.command()
        changed = cmd != self._last_sent
        if not changed and not (cmd.hold and now - self._last_sent_t >= self._republish_s):
            return None
        self._last_sent, self._last_sent_t = cmd, now
        return cmd


# ---- Gait parameter sliders (/nandi/gait_params Float64MultiArray) --------------------------
class GaitParam(NamedTuple):
    label: str
    lo: float
    hi: float
    default: float
    resolution: float


GAIT_PARAMS: Tuple[GaitParam, ...] = (
    GaitParam("Speed", 0.2, 1.5, 1.0, 0.05),
    GaitParam("Step ht", 0.5, 2.0, 1.0, 0.05),
    GaitParam("Freq Hz", 0.8, 3.0, 1.8, 0.1),
    GaitParam("Balance", 0.0, 2.0, 1.0, 0.05),
)
GAIT_DEFAULTS: List[float] = [p.default for p in GAIT_PARAMS]


def clamp_gait_params(values: Sequence) -> List[float]:
    """Return 4 gait params clamped to their ranges; missing/non-numeric entries use defaults."""
    out = []
    for i, spec in enumerate(GAIT_PARAMS):
        try:
            v = float(values[i])
        except (IndexError, TypeError, ValueError):
            v = spec.default
        if v != v:  # NaN
            v = spec.default
        out.append(max(spec.lo, min(spec.hi, v)))
    return out
