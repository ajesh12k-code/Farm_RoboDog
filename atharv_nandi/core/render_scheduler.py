"""GL render scheduler for the perception process (WP5; MVP-SPEC 2.4, 3.7, 3.8; ORCHARD-MVP-SPEC 3.7;
change request from WP36).

``register(camera, priority, modes)`` adds a camera. Priority is an int (lower = first) or a role name; the role
order is **wrist > head > ring > crown > belly > rear > pano** (MVP-SPEC 3.8 "wrist > head > ring > pano", WP36
adds belly and rear after ring; the crown, a TREE 360 camera served by capture requests, streams after the ring).
LiDAR is not GL and is not scheduled here (it runs in the sim process). Periodic renders below the head priority
keep a budget reserve (see Budget).

Modes (``set_mode``):

``drive``    default: head at 2 Hz, the ring round-robin at 2 cameras/s (the full ring every 2 s), crown at
             0.5 Hz, pano composed at <= 1 Hz from the newest ring frames (no GL pass of its own).
``grip``     GRIP / CUT / TOOL_CHANGE: **the ring is paused**; wrist 2 Hz, head 2 Hz.
``tree360``  TREE 360 station: **only capture requests** for the crown camera and the tree-side ring camera are
             served (RGB + depth + seg per capture, 3 passes budgeted); the other ring cameras, head, wrist and
             pano are paused (OS 3.7).
``paused``   nothing renders (e-stop, fallen, viewer-only).

A camera registered for a mode renders periodically at its rate; ``request(camera, tag)`` queues a one-shot
capture (TREE 360) that is served before any periodic render, in request order, while the camera is enabled in the
current mode.

Budget: a ``camera_rig.RenderBudget``-style token bucket caps the share of wall time spent in GL
(``budget_frac`` 0.35, burst 0.3 s; MVP-SPEC 3.8 raised 0.20 -> 0.35). An over-budget **periodic** render is
skipped (the caller falls back to geometry-only detection); an over-budget **capture** is deferred, never dropped
(the station waits for its ack). Periodic renders below the head priority start only with ``RESERVE_S`` (0.1 s) of
budget left, so when GL time is scarce the head (and wrist) camera keeps its rate. ``stats`` reports the measured render fraction (GL seconds / wall seconds since
the mode began), skips and deferrals per camera; the TREE 360 acceptance reads it (<= 0.35).
"""
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

ROLE_ORDER = ("wrist", "head", "ring", "crown", "belly", "rear", "pano")
ROLE_PRIORITY = {r: 10 * (k + 1) for k, r in enumerate(ROLE_ORDER)}
MODES = ("drive", "grip", "tree360", "paused")
GRIP_STATES = ("GRIP", "CUT", "TOOL_CHANGE")
BUDGET_FRAC = 0.35
BURST_S = 0.3
RING_CAMERAS_PER_S = 2.0
RESERVE_S = 0.1                  # budget kept for the head / wrist camera (periodic renders of lower priority)


def role_of(camera: str) -> str:
    c = camera.lower()
    for r in ROLE_ORDER:
        if c.startswith(r):
            return r
    return "pano" if "pano" in c else "rear"


@dataclass
class Slot:
    camera: str
    priority: int
    modes: Tuple[str, ...]
    role: str
    rate_hz: float
    passes: int = 1
    next_t: float = 0.0
    renders: int = 0
    skips: int = 0
    deferred: int = 0
    spent_s: float = 0.0


@dataclass
class Job:
    camera: str
    kind: str                 # "periodic" | "capture"
    tag: Optional[dict] = None
    passes: int = 1


class TokenBucket:
    """GL seconds accrue at ``frac`` per wall second up to ``burst_s``; a render may start while positive."""

    def __init__(self, frac: float = BUDGET_FRAC, burst_s: float = BURST_S):
        self.frac, self.burst_s = max(0.0, float(frac)), float(burst_s)
        self.tokens = self.burst_s
        self._last: Optional[float] = None

    def refill(self, now: float):
        if self._last is not None:
            self.tokens = min(self.burst_s, self.tokens + self.frac * max(0.0, now - self._last))
        self._last = now

    def ok(self, reserve: float = 0.0) -> bool:
        return self.frac > 0.0 and self.tokens > float(reserve)

    def charge(self, seconds: float):
        self.tokens -= float(seconds)


class RenderScheduler:
    def __init__(self, budget_frac: float = BUDGET_FRAC, burst_s: float = BURST_S,
                 ring_per_s: float = RING_CAMERAS_PER_S, max_renders_per_tick: int = 1):
        self.bucket = TokenBucket(budget_frac, burst_s)
        self.slots: Dict[str, Slot] = {}
        self.mode = "drive"
        self.mode_reason = ""
        self.ring_period = 1.0 / ring_per_s if ring_per_s > 0 else math.inf
        self._ring_next = 0.0
        self._ring_idx = 0
        self.max_per_tick = int(max_renders_per_tick)
        self.reserve_s = RESERVE_S
        self.queue: List[Job] = []
        self.only: Optional[set] = None
        self._window_t0: Optional[float] = None
        self._window_spent = 0.0
        self.total_spent = 0.0

    # ------------------------------------------------------------------ registration
    def register(self, camera: str, priority: Union[int, str, None] = None, modes: Iterable[str] = ("drive",),
                 rate_hz: float = 1.0, passes: int = 1, role: Optional[str] = None) -> Slot:
        if camera in self.slots:
            raise ValueError(f"camera {camera!r} registered twice")
        ms = tuple(modes)
        bad = [m for m in ms if m not in MODES]
        if bad:
            raise ValueError(f"unknown modes {bad}; known {MODES}")
        r = role or role_of(camera)
        if priority is None:
            priority = ROLE_PRIORITY.get(r, 99)
        elif isinstance(priority, str):
            if priority not in ROLE_PRIORITY:
                raise ValueError(f"unknown role {priority!r}")
            r = role or priority
            priority = ROLE_PRIORITY[priority]
        s = Slot(camera, int(priority), ms, r, float(rate_hz), int(passes))
        self.slots[camera] = s
        return s

    def order(self) -> List[str]:
        return [s.camera for s in sorted(self.slots.values(), key=lambda s: (s.priority, s.camera))]

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode: str, now: float = 0.0, reason: str = "", cameras: Optional[Iterable[str]] = None):
        """Switch mode. For ``tree360``, ``cameras`` names the station cameras (crown + the tree-side ring camera);
        every other camera is paused."""
        if mode not in MODES:
            raise ValueError(f"unknown render mode {mode!r}")
        self.only = set(cameras) if (mode == "tree360" and cameras is not None) else None
        if mode != self.mode:
            self.mode, self.mode_reason = mode, reason
            self._window_t0, self._window_spent = now, 0.0
            if mode != "tree360":
                self.queue = [j for j in self.queue if self.enabled(j.camera)]

    def mode_for_task(self, task_state: Optional[str], tree360: bool = False, paused: bool = False) -> str:
        """The mode a task state implies (GRIP/CUT/TOOL_CHANGE -> grip)."""
        if paused:
            return "paused"
        if tree360:
            return "tree360"
        if task_state and task_state.upper() in GRIP_STATES:
            return "grip"
        return "drive"

    def enabled(self, camera: str) -> bool:
        s = self.slots.get(camera)
        if s is None or self.mode not in s.modes or self.mode == "paused":
            return False
        return self.only is None or camera in self.only

    # ------------------------------------------------------------------ jobs
    def request(self, camera: str, tag: Optional[dict] = None, passes: int = 3) -> bool:
        """Queue a one-shot capture. False when the camera is unknown or not enabled in this mode."""
        if not self.enabled(camera):
            return False
        self.queue.append(Job(camera, "capture", dict(tag or {}), int(passes)))
        return True

    def due(self, now: float) -> List[Job]:
        """Jobs to render now, highest priority first (captures first). Charge each with ``charge``."""
        self.bucket.refill(now)
        if self._window_t0 is None:
            self._window_t0 = now
        out: List[Job] = []
        if self.mode == "paused":
            return out
        if self.queue:
            if self.bucket.ok():
                out.append(self.queue.pop(0))
            else:
                self.slots[self.queue[0].camera].deferred += 1
            return out[: self.max_per_tick]
        if self.mode == "tree360":
            return out
        cand: List[Slot] = []
        rings = [s for s in sorted(self.slots.values(), key=lambda s: s.camera)
                 if s.role == "ring" and self.mode in s.modes]
        for s in sorted(self.slots.values(), key=lambda s: (s.priority, s.camera)):
            if self.mode not in s.modes or s.role in ("ring", "pano") or s.rate_hz <= 0:
                continue
            if now + 1e-9 >= s.next_t:
                cand.append(s)
        ring_due = bool(rings) and now + 1e-9 >= self._ring_next
        picked: List[Job] = []
        for s in cand:
            if len(picked) >= self.max_per_tick:
                break
            if ring_due and ROLE_PRIORITY["ring"] < s.priority:
                break
            if self.bucket.ok(self._reserve(s)):
                picked.append(Job(s.camera, "periodic", None, s.passes))
            else:
                s.skips += 1
            s.next_t = max(s.next_t + 1.0 / s.rate_hz, now + 0.5 / s.rate_hz)
        if ring_due and len(picked) < self.max_per_tick:
            s = rings[self._ring_idx % len(rings)]
            self._ring_idx += 1
            if self.bucket.ok(self._reserve(s)):
                picked.append(Job(s.camera, "periodic", None, s.passes))
            else:
                s.skips += 1
            self._ring_next = max(self._ring_next + self.ring_period, now + 0.5 * self.ring_period)
        return picked

    def _reserve(self, s: Slot) -> float:
        """Periodic renders below the head priority start only with ``reserve_s`` of budget left, so when GL time
        is scarce the head (and wrist) camera keeps its rate and the ring, crown and others give way."""
        return 0.0 if s.priority <= ROLE_PRIORITY["head"] else self.reserve_s

    def charge(self, camera: str, seconds: float):
        self.bucket.charge(seconds)
        s = self.slots.get(camera)
        if s is not None:
            s.renders += 1
            s.spent_s += float(seconds)
        self._window_spent += float(seconds)
        self.total_spent += float(seconds)

    # ------------------------------------------------------------------ report
    def fraction(self, now: float) -> float:
        if self._window_t0 is None:
            return 0.0
        wall = max(1e-9, now - self._window_t0)
        return self._window_spent / wall

    def stats(self, now: float) -> dict:
        return {"mode": self.mode, "reason": self.mode_reason, "render_fraction": round(self.fraction(now), 4),
                "budget_frac": self.bucket.frac, "queue": len(self.queue),
                "cameras": {s.camera: {"renders": s.renders, "skips": s.skips, "deferred": s.deferred,
                                       "mean_ms": round(1e3 * s.spent_s / s.renders, 2) if s.renders else None}
                            for s in self.slots.values()}}


def default_scheduler(cfg: Optional[dict] = None, cameras: Sequence[str] = ()) -> RenderScheduler:
    """The perception node's default registration (``config/perception.yaml`` ``scheduler:``)."""
    c = dict(cfg or {})
    sch = RenderScheduler(float(c.get("budget_frac", BUDGET_FRAC)), float(c.get("burst_s", BURST_S)),
                          float(c.get("ring_cameras_per_s", RING_CAMERAS_PER_S)))
    rates = {"wrist": 2.0, "head": 2.0, "crown": 0.5, "ring": 0.5, "belly": 0.5, "rear": 0.5}
    rates.update({k: float(v) for k, v in (c.get("rates_hz") or {}).items()})
    modes = {"wrist": ("grip",), "head": ("drive", "grip"), "crown": ("drive", "tree360"),
             "ring": ("drive", "tree360"), "belly": ("drive",), "rear": ("drive",)}
    passes = {"wrist": 2, "head": 2, "crown": 2, "ring": 1, "belly": 1, "rear": 1}
    for cam in cameras:
        r = role_of(cam)
        sch.register(cam, r, modes.get(r, ("drive",)), rates.get(r, 0.5), passes.get(r, 1), role=r)
    return sch
