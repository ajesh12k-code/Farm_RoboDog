"""Crown mast runtime logic: rate-limited extension, the command arbiter, swept clearance (WP24).

ORCHARD-MVP-SPEC 2.3 (crown envelope rules, mast command arbitration) and 3.5 (mast runtime). Pure Python and
numpy, no MuJoCo and no ROS, so every rule is unit-tested in ``tests/test_mast.py``. ``atharv_nandi/sim/hooks/mast.py`` binds
it to a model: it writes the interpolated section ``body_pos`` every physics step, runs the geometric strike
check, owns the virtual gimbal and the e-stop brake, and publishes ``/nandi/mast/state``.

Arbiter (``MastArbiter``; implements ``interfaces.MastControl``). The arbiter is the ONLY writer of the mast
setpoint. Priority (``interfaces.MAST_PRIORITY``): e-stop brake > safety retract (WP6 gate decisions, strike,
wind, tilt) > recovery (WP28) > preset (WP25 / WP27). Rules:

    * Every request is ``MastRequest {source, target_ext, reason, gimbal?}``; the newest request of a source
      replaces that source's older one. The winning request is the highest-priority active one.
    * A request from a source of LOWER priority than the active winner is refused (``crown_gate``,
      ``preempted``): it is never queued behind it.
    * When a request wins over lower-priority ones, those are DROPPED (logged in ``preempted``), so a mast that
      was retracted by safety or braked by the e-stop never re-raises by itself: the requester must submit again
      and pass the gates again.
    * A raise (target above the current extension) passes ``check(ext_from, ext_to, request)`` first: the hook's
      gate chain (e-stop, standing, cross-slope, wind, WP6 gates, swept clearance). A retract is never gated.
    * ``brake()`` (e-stop): the sections hold where they are, every other request is dropped and new ones are
      refused until ``release_brake()``. An explicit brake latches (``brake_latched``): the end of the runner hold
      releases only the hook's own hold brake. A retract during e-stop recovery is a new recovery request (WP28) after
      the hold has cleared.
    * A ``safety`` request, and a ``recovery`` request that was a retract when it was accepted (target at or below
      the extension), complete (are removed) once the extension reaches the target, so a finished recovery retract
      never blocks a later preset raise. A recovery raise and every preset request are setpoints and stay until
      replaced or released.
    * ``safety`` requests only retract. A safety request with a target above the current extension is refused
      (``crown_gate``, ``safety_raise``). This stops any publisher from using the safety level to raise past the
      gates or to switch off the hook backstops (OS 2.3 defines the level as "safety retract").
    * ``step(dt, hold_raise, hold_lower)`` implements the hook's rule 3, "distance < 0.05 m stops motion of the
      mast". ``hold_raise`` stops upward motion. ``hold_lower`` also stops downward motion; the hook sets it when
      lowering would close on the nearest obstacle. A ``safety`` retract (strike, wind, tilt) always runs.
    * ``raise_hold`` (a non-empty reason, set by the hook's per-tick re-check of the raise gates) also stops
      upward motion. ``abort_raise`` drops a raise whose gates kept failing; the requester must submit again.

Rate limit: ``RATE_MPS`` 0.2 m/s in [0, ``EXT_MAX_M``], applied per physics step (``step(dt)``): at dt 0.002 the
setpoint moves at most 0.4 mm per step and a full 1.60 m stroke takes 8.0 s. ``split(ext)`` gives the three
section offsets (ext / 3 each, nested, so section k's world offset is k * ext / 3).

Swept clearance (rule 1). ``swept_clearance(obstacles, c0, c1, r_head)``: the head moving from c0 to c1 sweeps a
capsule of radius ``r_head`` (the head's bounding sphere); the distance from it to every obstacle (``Obstacle``:
capsule, cylinder as capsule, sphere, ellipsoid, box) must be at least ``CLEARANCE_M`` 0.10 m. Capsule-capsule is
exact (segment-segment distance); ellipsoid, box and sphere sample the swept segment every ``SAMPLE_M`` and use
an exact point-box distance and the first-order point-ellipsoid distance ``k0 (k0 - 1) / k1`` (k0 = |p / r|,
k1 = |p / r^2|; exact on spheres, within a few mm of the true distance near the surface of the WP23 canopy
ellipsoids, see the tests). ``geom_distance`` is the same math for a single head pose: the analytic fallback of
the per-step strike check when ``mujoco.mj_geomDistance`` is missing.

``alley_clear(head_xy, head_top_z, alleys)``: rule 1 / MISSION-PRESETS 0.1 ``alley_clear_height_m`` per alley side.
ASSUMED reading: the clear height applies under the tree's overhang on that side, i.e. where the head is within
``canopy_r_m + ALLEY_ZONE_MARGIN_M`` (0.3 m) of the trunk axis, horizontally, in the side's half-plane (``-y``,
``+y``, ``-x``, ``+x`` or ``all``). The head top plus ``CLEARANCE_M`` must stay at or below the clear height above
the tree base.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi.core.interfaces import MAST_EXT_MAX_M, MAST_PRIORITY, GateDecision, MastRequest

EXT_MAX_M = MAST_EXT_MAX_M          # 1.60 m stroke (OS 2.3)
RATE_MPS = 0.2                      # rate limit (OS 2.3): 8 s full stroke
N_SECTIONS = 3                      # mast_s1 > mast_s2 > mast_s3, ext / 3 each
LENS_STOWED_M = 1.10                # crown lens world z at the home keyframe, stowed (WP1, MEASURED 1.10)
CLEARANCE_M = 0.10                  # rule 1: swept-cylinder clearance
NEAR_M = 0.05                       # rule 3: distance < 0.05 m stops mast motion
STRIKE_M = 0.0                      # rule 3: distance <= 0 is a strike event
SAMPLE_M = 0.01                     # swept-segment sampling for non-capsule obstacles
ALLEY_ZONE_MARGIN_M = 0.3           # ASSUMED: alley clear height applies within canopy_r + 0.3 m of the trunk
PAN_LIMIT_RAD = math.pi             # gimbal pan +-180 deg (OS 2.3)
TILT_MIN_RAD = math.radians(-60.0)  # gimbal tilt -60 .. +45 deg (OS 2.3)
TILT_MAX_RAD = math.radians(45.0)
GATE = "crown_gate"                 # MISSION-PRESETS 0.5 reason codes used below
R_CLEARANCE, R_WIND, R_SLOPE, R_STABILITY = "crown_clearance", "wind", "slope", "stability"
_EPS = 1e-9


def clamp_ext(ext: float) -> float:
    return min(EXT_MAX_M, max(0.0, float(ext)))


def split(ext: float) -> Tuple[float, float, float]:
    """``body_pos`` z offset of each section relative to its parent: ext / 3 each (OS 2.3)."""
    e = clamp_ext(ext) / N_SECTIONS
    return (e, e, e)


def lens_z(ext: float, stowed: float = LENS_STOWED_M) -> float:
    """Nominal crown lens height on the home stance: 1.10 m stowed, 2.70 m at 1.60."""
    return stowed + clamp_ext(ext)


def clamp_gimbal(pan: float, tilt: float) -> Tuple[float, float]:
    """Wrap pan into [-pi, pi], clamp tilt into [-60, +45] deg (``MastArbiter.aim`` flags a clamp in
    ``aim_clamped`` / state ``gimbal_clamped``)."""
    p = math.atan2(math.sin(float(pan)), math.cos(float(pan)))
    return p, min(TILT_MAX_RAD, max(TILT_MIN_RAD, float(tilt)))


def rate_step(ext: float, target: float, dt: float, rate: float = RATE_MPS) -> float:
    """One rate-limited step from ``ext`` toward ``target`` (both clamped to the stroke)."""
    lim = rate * dt
    tgt = clamp_ext(target)
    new = clamp_ext(ext + max(-lim, min(lim, tgt - ext)))
    return tgt if abs(tgt - new) <= 1e-9 else new     # no extra step for float round-off


# ---------------------------------------------------------------------- arbiter
@dataclass
class Decision:
    """The last ``submit`` outcome, as published on ``/nandi/mast/state`` (``last_decision``)."""
    allowed: bool
    source: str
    target_ext: Optional[float]
    reason: str
    detail: str = ""
    t: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"allowed": self.allowed, "source": self.source, "target_ext": self.target_ext,
                "reason": self.reason, "detail": self.detail, "t": round(float(self.t), 4)}


CheckFn = Callable[[float, float, MastRequest], GateDecision]


class MastArbiter:
    """Sole writer of the mast setpoint (``MastControl``). See the module docstring for the rules."""

    def __init__(self, rate: float = RATE_MPS, check: Optional[CheckFn] = None, ext: float = 0.0):
        self.rate = float(rate)
        self.check = check
        self.ext = clamp_ext(ext)
        self.requests: Dict[str, MastRequest] = {}
        self.braked = False
        self.gimbal: Tuple[float, float] = (0.0, 0.0)
        self.last: Optional[Decision] = None
        self.preempted: List[Tuple[str, str]] = []      # (dropped source, by source) for the state / tests
        self.completed: List[str] = []                  # reasons of completed safety requests
        self.held_by: str = ""                          # why the last step did not move toward the target
        self.raise_hold: str = ""                       # set by the hook's per-tick raise re-gate ("" = clear)
        self.aborted: List[Tuple[str, str]] = []        # (source, detail) of raises dropped by the re-gate
        self.brake_latched = False                      # an explicit brake (brake() / estop request): only
                                                        # release_brake() ends it, never the end of a runner hold
        self.aim_clamped = False                        # the last aim was outside the gimbal limits and clamped
        self._retracts: set = set()                     # sources whose accepted request is a retract
        self.t = 0.0

    # -- requests
    def winner(self) -> Optional[MastRequest]:
        if not self.requests:
            return None
        return min(self.requests.values(), key=lambda r: (r.priority, -r.t))

    def _drop_below(self, priority: int, by: str):
        for s in [s for s, r in self.requests.items() if r.priority > priority]:
            self.requests.pop(s)
            self.preempted.append((s, by))

    def _decide(self, ok: bool, req: MastRequest, reason: str = "", detail: str = "") -> GateDecision:
        self.last = Decision(ok, req.source, req.target_ext, reason, detail, req.t)
        if ok:
            return GateDecision.ok(GATE, target_ext=float(req.target_ext if req.target_ext is not None else self.ext))
        return GateDecision.refuse(GATE, reason, target_ext=float(req.target_ext or 0.0), ext_m=self.ext)

    def submit(self, request: MastRequest) -> GateDecision:
        """Accept or refuse one request (refusal carries a MISSION-PRESETS 0.5 reason code)."""
        if request.source == "estop":
            self.brake()
            return self._decide(True, request)
        if self.braked:
            return self._decide(False, request, GATE, "estop")
        w = self.winner()
        if w is not None and w.priority < request.priority:
            return self._decide(False, request, GATE, f"preempted:{w.source}:{w.reason}")
        target = request.target_ext
        if request.source == "safety" and (target is None or target > self.ext + _EPS):
            return self._decide(False, request, GATE, "safety_raise")
        if target is not None and target > self.ext + _EPS and self.check is not None:
            try:
                dec = self.check(self.ext, float(target), request)
            except Exception as exc:  # noqa: BLE001 - a broken gate fails safe (refuse), never raises
                dec = GateDecision.refuse(GATE, R_STABILITY, error=0.0)
                self.last = Decision(False, request.source, target, R_STABILITY, f"gate_error:{exc}", request.t)
                return dec
            if not dec.allowed:
                detail = str(dec.values.get("detail", "")) if isinstance(dec.values, dict) else ""
                self.last = Decision(False, request.source, target, dec.reason or GATE, detail or dec.gate,
                                     request.t)
                return dec
        self.requests[request.source] = request
        if target is not None and target <= self.ext + _EPS:
            self._retracts.add(request.source)
        else:
            self._retracts.discard(request.source)
        self._drop_below(request.priority, request.source)
        if request.gimbal is not None:
            self.aim(*request.gimbal)
        return self._decide(True, request)

    def release(self, source: str) -> bool:
        """Remove ``source``'s request (``estop``: release the brake). True if something was removed."""
        if source == "estop":
            was = self.braked
            self.release_brake()
            return was
        return self.requests.pop(source, None) is not None

    def brake(self, latch: bool = True):
        """E-stop: hold in place, drop every request, refuse new ones until ``release_brake``. ``latch`` False is
        the hook's brake from the runner hold, which ends with the hold; an explicit brake (the default, also an
        ``estop`` request) latches, also when it arrives while a hold brake is already on."""
        if not self.braked:
            self._drop_below(-1, "estop")
        self.braked = True
        self.brake_latched = self.brake_latched or bool(latch)

    def release_brake(self):
        self.braked = False
        self.brake_latched = False

    def aim(self, pan: float, tilt: float):
        self.gimbal = clamp_gimbal(pan, tilt)
        self.aim_clamped = abs(self.gimbal[1] - float(tilt)) > 1e-9   # tilt outside -60 .. +45 deg (pan wraps)

    def reset(self, ext: float = 0.0):
        self.ext = clamp_ext(ext)
        self.requests.clear()
        self.braked = False
        self.brake_latched = False
        self.aim_clamped = False
        self._retracts.clear()
        self.gimbal = (0.0, 0.0)
        self.last = None
        self.preempted.clear()
        self.completed.clear()
        self.held_by = ""
        self.raise_hold = ""
        self.aborted.clear()

    def raising(self) -> Optional[MastRequest]:
        """The winning request when it is a raise in progress (target above the extension), else None."""
        w = self.winner()
        if self.braked or w is None or w.target_ext is None or clamp_ext(w.target_ext) <= self.ext + _EPS:
            return None
        return w

    def abort_raise(self, reason: str, detail: str, t: float = 0.0) -> bool:
        """Drop the raise in progress (its gates failed during the motion); the mast stays where it is."""
        w = self.raising()
        if w is None:
            return False
        self.requests.pop(w.source, None)
        self.aborted.append((w.source, detail))
        self.raise_hold = ""
        self.last = Decision(False, w.source, w.target_ext, reason, f"regate:{detail}", t)
        return True

    # -- motion
    def target(self) -> float:
        w = self.winner()
        if self.braked or w is None or w.target_ext is None:
            return self.ext
        return clamp_ext(w.target_ext)

    def step(self, dt: float, hold_raise: bool = False, hold_lower: bool = False) -> float:
        """Advance the setpoint by one physics step. Returns the new extension."""
        self.held_by = ""
        if self.braked:
            self.held_by = "estop"
            return self.ext
        w = self.winner()
        if w is None:
            return self.ext
        tgt = self.target()
        if tgt > self.ext + _EPS and (hold_raise or self.raise_hold):
            self.held_by = "near_obstacle" if hold_raise else f"gate:{self.raise_hold}"
            return self.ext
        if hold_lower and tgt < self.ext - _EPS and w.source != "safety":
            self.held_by = "near_obstacle"
            return self.ext
        self.ext = rate_step(self.ext, tgt, dt, self.rate)
        if abs(self.ext - tgt) <= _EPS and (w.source == "safety"
                                            or (w.source == "recovery" and "recovery" in self._retracts)):
            self.requests.pop(w.source, None)       # a safety request or a recovery retract completes
            self._retracts.discard(w.source)
            self.completed.append(w.reason)
        return self.ext

    @property
    def moving(self) -> bool:
        return not self.braked and abs(self.target() - self.ext) > _EPS

    def state(self) -> Dict[str, Any]:
        w = self.winner()
        last = self.last
        gated = bool(last is not None and not last.allowed)
        return {"ext_m": self.ext, "target_ext": self.target(), "gimbal": list(self.gimbal),
                "gated": gated, "reason": last.reason if gated else "",
                "arbiter_source": "estop" if self.braked else (w.source if w else None),
                "arbiter_reason": "estop_brake" if self.braked else (w.reason if w else ""),
                "braked": self.braked, "brake_latched": self.brake_latched, "moving": self.moving,
                "held_by": self.held_by, "gimbal_clamped": self.aim_clamped,
                "last_decision": last.as_dict() if last else None}


def priority_table() -> List[Tuple[str, int]]:
    return sorted(MAST_PRIORITY.items(), key=lambda kv: kv[1])


# ---------------------------------------------------------------------- geometry
CAPSULE, CYLINDER, SPHERE, ELLIPSOID, BOX = "capsule", "cylinder", "sphere", "ellipsoid", "box"
KINDS = ("wood", "envelope", "rope")


@dataclass
class Obstacle:
    """One obstacle in world coordinates. ``size`` follows MuJoCo: capsule / cylinder (r, half-length),
    sphere (r), ellipsoid (rx, ry, rz), box (half extents). ``mat`` is the 3x3 world rotation."""
    name: str
    kind: str                    # wood | envelope | rope
    gtype: str
    pos: np.ndarray
    mat: np.ndarray
    size: np.ndarray
    rbound: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def segment(self) -> Tuple[np.ndarray, np.ndarray]:
        half = float(self.size[1])
        ax = self.mat[:, 2]
        return self.pos - half * ax, self.pos + half * ax


def seg_seg(p1, q1, p2, q2) -> float:
    """Minimum distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    p1, q1, p2, q2 = (np.asarray(v, float) for v in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= _EPS and e <= _EPS:
        return float(np.linalg.norm(r))
    if a <= _EPS:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = float(d1 @ r)
        if e <= _EPS:
            t, s = 0.0, min(1.0, max(0.0, -c / a))
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / den)) if den > _EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def points_ellipsoid(P: np.ndarray, c, mat, radii) -> np.ndarray:
    """Signed distance of points P (n, 3) to an ellipsoid; first-order ``k0 (k0 - 1) / k1``."""
    q = (np.asarray(P, float) - np.asarray(c, float)) @ np.asarray(mat, float)   # into the ellipsoid frame
    r = np.asarray(radii, float)
    k0 = np.linalg.norm(q / r, axis=1)
    k1 = np.linalg.norm(q / (r * r), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(k1 > _EPS, k0 * (k0 - 1.0) / np.maximum(k1, _EPS), -float(np.min(r)))
    return dist


def points_box(P: np.ndarray, c, mat, half) -> np.ndarray:
    q = np.abs((np.asarray(P, float) - np.asarray(c, float)) @ np.asarray(mat, float)) - np.asarray(half, float)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def points_segment(P: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)
    if L2 <= _EPS:
        return np.linalg.norm(np.asarray(P, float) - a, axis=1)
    t = np.clip(((np.asarray(P, float) - a) @ ab) / L2, 0.0, 1.0)
    return np.linalg.norm(np.asarray(P, float) - (a + t[:, None] * ab), axis=1)


def capsule_distance(c0, c1, r: float, ob: Obstacle) -> float:
    """Signed distance from the capsule (segment c0-c1, radius r) to ``ob``."""
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    if ob.gtype in (CAPSULE, CYLINDER):
        a, b = ob.segment()
        return seg_seg(c0, c1, a, b) - float(ob.size[0]) - r
    L = float(np.linalg.norm(c1 - c0))
    n = max(2, int(math.ceil(L / SAMPLE_M)) + 1)
    P = c0 + np.linspace(0.0, 1.0, n)[:, None] * (c1 - c0)
    if ob.gtype == SPHERE:
        return float(np.min(np.linalg.norm(P - ob.pos, axis=1))) - float(ob.size[0]) - r
    if ob.gtype == ELLIPSOID:
        return float(np.min(points_ellipsoid(P, ob.pos, ob.mat, ob.size[:3]))) - r
    if ob.gtype == BOX:
        return float(np.min(points_box(P, ob.pos, ob.mat, ob.size[:3]))) - r
    # unknown type: its bounding sphere (conservative)
    return float(np.min(np.linalg.norm(P - ob.pos, axis=1))) - float(ob.rbound) - r


def geom_distance(c0, c1, r: float, ob: Obstacle) -> float:
    """Analytic fallback of ``mj_geomDistance`` for a head / section modelled as a capsule (c0 = c1: a sphere)."""
    return capsule_distance(c0, c1, r, ob)


def swept_clearance(obstacles: Sequence[Obstacle], c0, c1, r_head: float,
                    clearance: float = CLEARANCE_M) -> Tuple[float, Optional[Obstacle]]:
    """(min distance, nearest obstacle) of the swept head capsule; (inf, None) with no obstacles."""
    best, who = math.inf, None
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    lo, hi = np.minimum(c0, c1) - r_head - clearance, np.maximum(c0, c1) + r_head + clearance
    for ob in obstacles:
        # bounding-sphere reject: cheaper than the exact distance and never wrong
        if np.any(ob.pos + ob.rbound < lo) or np.any(ob.pos - ob.rbound > hi):
            continue
        dist = capsule_distance(c0, c1, r_head, ob)
        if dist < best:
            best, who = dist, ob
    return best, who


@dataclass(frozen=True)
class AlleyRule:
    """One alley side of one tree: ``alley_clear_height_m`` above ``base_z`` within the overhang zone."""
    tree_id: str
    x: float
    y: float
    base_z: float
    canopy_r: float
    side: str                     # -y | +y | -x | +x | all
    clear_h: float


def alley_rules(entry: dict, base: Sequence[float]) -> List[AlleyRule]:
    """Rules of one registry tree entry (``alley_sides`` and ``alley_clear_height_m`` are parallel lists)."""
    sides = entry.get("alley_sides") or []
    hs = entry.get("alley_clear_height_m") or []
    cr = entry.get("canopy_r_m")
    if not isinstance(sides, list) or not isinstance(hs, list) or not isinstance(cr, (int, float)):
        return []
    out = []
    for s, h in zip(sides, hs):
        if isinstance(h, (int, float)) and not isinstance(h, bool) and math.isfinite(h) and isinstance(s, str):
            out.append(AlleyRule(str(entry.get("tree_id", "")), float(base[0]), float(base[1]), float(base[2]),
                                 float(cr), s, float(h)))
    return out


def _in_side(side: str, dx: float, dy: float) -> bool:
    return {"all": True, "-y": dy < 0.0, "+y": dy > 0.0, "-x": dx < 0.0, "+x": dx > 0.0}.get(side, False)


def alley_clear(head_xy: Sequence[float], head_top_z: float, rules: Sequence[AlleyRule],
                clearance: float = CLEARANCE_M) -> Tuple[float, Optional[AlleyRule]]:
    """(margin, violated rule or None): margin = clear height - (head top + clearance) of the tightest rule."""
    best, who = math.inf, None
    for r in rules:
        dx, dy = float(head_xy[0]) - r.x, float(head_xy[1]) - r.y
        if math.hypot(dx, dy) > r.canopy_r + ALLEY_ZONE_MARGIN_M or not _in_side(r.side, dx, dy):
            continue
        margin = (r.base_z + r.clear_h) - (float(head_top_z) + clearance)
        if margin < best:
            best, who = margin, r
    return best, (who if best < 0.0 else None)
