"""Logical cut rule (CLIPPER pedicel / stalk clip, PRUNER prune and tidy) and kinematic debris (WP2;
MVP-SPEC 3.4, ORCHARD-MVP-SPEC 3.4). Pure Python + numpy, no MuJoCo, no ROS.

The cut fires (``CutRule.update``, control rate) only when ALL hold (MVP-SPEC 3.4):

    1. closure     blade ctrl >= 0.95 sustained >= 30 ms, AND the blade joint >= 90 % closed
    2. capacity    commanded capacity forcerange / lever >= F_cut (crop table; prune: green-wood shear of the
                   diameter). A property of the actuator and command, not of how the position error decays
    3. trap        distance from the cut point (``cut_k`` / prune cut point) to the ``blade_a``-``blade_b``
                   segment < 6 mm (demo, the default) or < 4 mm (physics mode)
    4. angle       stem-blade angle < 30 deg: the stem (pedicel) line against the blade hinge axis (the scissor
                   cuts in the plane normal to it), as an unsigned line angle 0..90 deg

A near miss (trap or angle failing at full closure) never fires; ``CutRule`` reports which condition held.
``pedicel_cut_point`` gives where the blades cross the pedicel (the closest point on the stem_end -> anchor
segment to the blade segment); the pedicel error is |cut point - stem end| - pedicel_leave in the fruit frame.

Prune (OS 3.4, MP 6): ``prune_gate`` refuses a stem over the PRUNER capacity (12 mm, ``over_capacity`` with
``paste_required`` for > 12 mm, MISSION-PRESETS 6) or one the truth marks as keep wood. ``Debris`` is the
kinematic drop of the cut piece (a mocap body, 0 DOF): free fall from the cut pose to the ground, then rest.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi.core.interfaces import GateDecision

CTRL_CLOSED = 0.95
SUSTAIN_S = 0.030
CLOSED_FRAC = 0.90
WINDOW_MM = {"demo": 6.0, "physics": 4.0}
MAX_ANGLE_DEG = 30.0
DAMAGE_MM = 5.0                 # a blade within 5 mm of any fruit geom at closure = damage (report)
PRUNE_CAPACITY_MM = 12.0        # (assumption, spec) green wood
PASTE_REQUIRED_MM = 12.0        # cut > 12 mm -> Bordeaux paste MARK (MISSION-PRESETS 6)
G = 9.81


# ---------------------------------------------------------------------------------------------- geometry
def point_segment(p, a, b) -> Tuple[float, np.ndarray, float]:
    """(distance, closest point, parameter s in [0, 1]) from point p to segment a-b."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    L2 = float(ab @ ab)
    s = 0.0 if L2 < 1e-18 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    c = a + s * ab
    return float(np.linalg.norm(p - c)), c, s


def segment_segment(p1, q1, p2, q2) -> Tuple[np.ndarray, np.ndarray, float]:
    """Closest points (on segment 1, on segment 2) and their distance (Ericson, Real-Time Collision Detection)."""
    p1, q1, p2, q2 = (np.asarray(v, float) for v in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-18
    if a <= eps and e <= eps:
        return p1, p2, float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))
    c1, c2 = p1 + s * d1, p2 + t * d2
    return c1, c2, float(np.linalg.norm(c1 - c2))


def line_angle_deg(u, v) -> float:
    """Unsigned angle between two lines (0..90 deg)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 90.0
    c = abs(float(u @ v)) / (nu * nv)
    return math.degrees(math.acos(min(1.0, c)))


def pedicel_cut_point(stem_end, anchor, blade_a, blade_b) -> np.ndarray:
    """Where the blade segment crosses the pedicel: the closest point on stem_end -> anchor to blade_a-blade_b."""
    c1, _c2, _d = segment_segment(stem_end, anchor, blade_a, blade_b)
    return c1


def pedicel_error_mm(cut_point, stem_end, leave_m: float) -> float:
    """|distance(cut point, stem end) - pedicel_leave| in mm (frame-free: both points move with the fruit)."""
    return 1000.0 * abs(float(np.linalg.norm(np.asarray(cut_point, float) - np.asarray(stem_end, float)))
                        - float(leave_m))


def capacity_n(forcerange_nm: float, lever_m: float) -> float:
    return float(forcerange_nm) / float(lever_m) if lever_m > 0 else 0.0


# ---------------------------------------------------------------------------------------------- cut rule
@dataclass
class CutInputs:
    t: float
    blade_ctrl: float              # 0..1, 1 = closed
    closed_frac: float             # blade joint / full closing travel
    capacity_n: float              # forcerange / lever
    f_cut_n: float                 # crop / diameter cut force
    dist_mm: float                 # cut point -> blade segment
    angle_deg: float               # stem line vs blade hinge axis


@dataclass
class CutDecision:
    fire: bool
    conditions: Dict[str, bool]
    sustained_s: float


@dataclass
class CutRule:
    mode: str = "demo"             # demo (6 mm) | physics (4 mm)
    ctrl_closed: float = CTRL_CLOSED
    sustain_s: float = SUSTAIN_S
    closed_frac: float = CLOSED_FRAC
    max_angle_deg: float = MAX_ANGLE_DEG
    window_mm: Optional[float] = None
    _since: Optional[float] = field(default=None, repr=False)
    fired: bool = False

    def __post_init__(self):
        if self.window_mm is None:
            if self.mode not in WINDOW_MM:
                raise ValueError(f"cut mode {self.mode!r} not in {sorted(WINDOW_MM)}")
            self.window_mm = WINDOW_MM[self.mode]

    def reset(self):
        self._since, self.fired = None, False

    def update(self, x: CutInputs) -> CutDecision:
        """One control tick. Fires at most once until ``reset`` (one cut per target)."""
        if x.blade_ctrl >= self.ctrl_closed:
            if self._since is None:
                self._since = float(x.t)
        else:
            self._since = None
        sustained = 0.0 if self._since is None else float(x.t) - self._since
        cond = {"ctrl_sustained": sustained + 1e-9 >= self.sustain_s,
                "closed": x.closed_frac >= self.closed_frac,
                "capacity": x.capacity_n >= x.f_cut_n,
                "trap": x.dist_mm < self.window_mm,
                "angle": x.angle_deg < self.max_angle_deg}
        fire = all(cond.values()) and not self.fired
        if fire:
            self.fired = True
        return CutDecision(fire, cond, sustained)


# ---------------------------------------------------------------------------------------------- prune
def prune_gate(diameter_mm: float, keep: bool = False, capacity_mm: float = PRUNE_CAPACITY_MM,
               in_band: bool = True) -> GateDecision:
    """May the PRUNER cut this stem? Refuses keep wood, out-of-band targets and stems over capacity (> 12 mm:
    MARK ``over_capacity`` with ``paste_required``)."""
    d = float(diameter_mm)
    vals = {"diameter_mm": d, "capacity_mm": float(capacity_mm), "paste_required": float(d > PASTE_REQUIRED_MM)}
    if keep:
        return GateDecision.refuse("prune", "keep_wood", **vals)
    if not in_band:
        return GateDecision.refuse("prune", "out_of_reach_marked", **vals)
    if not math.isfinite(d) or d <= 0.0:
        return GateDecision.refuse("prune", "bad_diameter", **vals)
    if d > capacity_mm + 1e-9:
        return GateDecision.refuse("prune", "over_capacity", **vals)
    return GateDecision.ok("prune", **vals)


# ---------------------------------------------------------------------------------------------- debris
@dataclass
class Debris:
    """Kinematic free fall of a cut piece from ``pos0`` to rest on ``ground_z`` (+ its radius): z(t) = z0 - g t^2/2."""
    pos0: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]
    t0: float
    rest_z: float
    source: str = ""
    landed: bool = False

    def pos(self, t: float) -> Tuple[float, float, float]:
        tau = max(0.0, float(t) - self.t0)
        z = self.pos0[2] - 0.5 * G * tau * tau
        if z <= self.rest_z:
            z, self.landed = self.rest_z, True
        return (self.pos0[0], self.pos0[1], z)

    def impact_speed(self) -> float:
        return math.sqrt(max(0.0, 2.0 * G * (self.pos0[2] - self.rest_z)))
