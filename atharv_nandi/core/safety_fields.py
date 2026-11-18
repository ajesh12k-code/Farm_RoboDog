"""Hull-relative safety fields, crown scaling, orbit person rule, animal stop, bunch exclusion (WP6).

MVP-SPEC 5.3 and ORCHARD-MVP-SPEC 5.2 / 5.3. Pure Python + numpy, no MuJoCo, no ROS: ``atharv_nandi/sim/hooks/stability.py``
casts the safety rays and feeds ``Return`` points in; everything here is unit-tested in tests/test_safety_fields.py.
L0 hard safety: every number below is a code constant (WORKFLOWS-SPEC L0, "code only, never YAML").

Frames. Every point is in the **robot base frame**: x forward, y left, origin at the trunk centre projected on the
ground plane. Fields are polygons **offset from the robot hull**, not radii around a sensor, and a distance is
always **person (return) to hull**.

Hull (``robot_hull``). 2D convex hull of the trunk box, the feet at their current stance, the **current arm reach
envelope** (arm FK points + 0.1 m) and the crown head footprint. During an orbit the hull is swept along the
planned arc (``OrbitArc``) with a 0.3 m corridor (OS 5.2: "the planned arc +-0.3 m").

Fields (``field_offsets``), metres beyond the hull, by sector of the return's bearing from the hull centre
(front |bearing| <= 45 deg, rear >= 135 deg, sides between):

    mode      protective (stop + latch)                         warning
    WALK      front d_stop(v) + 0.25, sides 0.25, rear 0.30       front 2.5, sides 1.0, rear 1.5
    WORK      0.30 all round                                     1.0 all round
    (STOPPED uses WORK; ORBIT uses WALK on the swept hull)
    crown up  protective >= 0.5 x lens height, warning >= lens height, all round (fall envelope, ASSUMED)

    d_stop(v) = v * 0.1 s + v^2 / (2 * 0.6 m/s^2)   (ASSUMED)

Responses (``FieldMonitor.update``):

    protective intrusion (any unmasked return)           stop + latch (``protective``); arm stop
    animal in the protective or warning field            stop + latch (``animal``): animals are never person logic
    bumper contact                                       stop + latch (``bumper``)
    warning intrusion, WALK                              speed cap 30 %, beacon flash
    warning intrusion, ORBIT                             orbit pause (not latched; resumes when clear)
    warning intrusion, WORK / STOPPED                    arm pause, blade close and share lowering blocked
    hand at arm height inside the WORK envelope          inside the WORK protective field -> stop + latch

The latch clears only with ``clear()`` while nothing is in the protective field and no animal in the warning
field (``protective_clear``). There is no muting: plants are masked from the map, never muted.

Classification (``classify_geom``). Returns from **known static map geoms** (ground, terrain, tree wood /
foliage / fruit, pool fruit, props, ropes, laterals, crate, dock, mud patches, stakes, legacy crop items: listed
from the model by name at build time) are masked (``assisted: map_masked_lidar``). Every other return is an
obstacle: conservative geometry, **no person truth label**. Animal actor geoms are the one labelled class
(``animal``, assisted): they stop the robot from the warning field and are never treated by person logic.

Bunch exclusion (``bunch_exclusion``): >= 3 m from any MARKed bunch while a human cut is flagged in progress.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi.core.interfaces import GateDecision
from atharv_nandi.core.stability import convex_hull

MODE_WALK, MODE_WORK, MODE_STOPPED, MODE_ORBIT = "WALK", "WORK", "STOPPED", "ORBIT"
MODES = (MODE_WALK, MODE_WORK, MODE_STOPPED, MODE_ORBIT)
FIELD_PROTECTIVE, FIELD_WARNING = "protective", "warning"
CLS_OBSTACLE, CLS_ANIMAL = "obstacle", "animal"

# MVP-SPEC 5.3 field table (ASSUMED)
WALK_PROTECTIVE = {"front": 0.25, "side": 0.25, "rear": 0.30}      # + d_stop(v) in the direction of travel
WALK_WARNING = {"front": 2.5, "side": 1.0, "rear": 1.5}
WORK_PROTECTIVE = {"front": 0.30, "side": 0.30, "rear": 0.30}
WORK_WARNING = {"front": 1.0, "side": 1.0, "rear": 1.0}
STOP_REACTION_S = 0.1
STOP_DECEL_MPS2 = 0.6
WALK_SPEED_CAP_MPS = 0.3
WARNING_SPEED_SCALE = 0.3                     # warning field: speed cap 30 %
CROWN_PROTECTIVE_K = 0.5                      # OS 5.2: protective >= hull + 0.5 x lens height while ext > 0
CROWN_WARNING_K = 1.0                         # warning >= hull + lens height
CROWN_UP_EXT_M = 0.02
ARM_ENVELOPE_PAD_M = 0.10                     # hull includes the arm FK + 0.1 m
ORBIT_CORRIDOR_M = 0.30                       # OS 5.2: planned arc +-0.3 m
FRONT_HALF_DEG = 45.0
REAR_FROM_DEG = 135.0
BUNCH_EXCLUSION_M = 3.0                       # OS 5.2
CLEAR_HOLD_S = 0.5                            # the field must stay clear this long before the latch can clear
TRUNK_HALF_XY = (0.33, 0.15)                  # WP1 trunk box half sizes (m)

R_PROTECTIVE, R_ANIMAL, R_BUMPER = "protective", "animal", "bumper"
# Static map geoms (masked). Matched on the geom name, or on the body name for anonymous geoms.
MAP_PREFIXES = ("floor", "orchard_ground", "terrain", "hfield", "ground", "tree_", "pool_", "crate_", "dock_",
                "prop_", "rope_", "lateral_", "patch_", "stake", "ridge_", "basin", "bund", "crop_", "plant",
                "stem", "row_", "fixture_", "trunk_stub", "canopy_stub", "mulch", "water_",
                "debris_")                    # WP2 prune debris: robot-made, known, falls inside the WORK field
ACTOR_PREFIX = "actor_"


def d_stop(v: float) -> float:
    """Stopping distance (m) at speed v (m/s): reaction 0.1 s + braking at 0.6 m/s^2 (ASSUMED)."""
    v = max(0.0, float(v))
    return v * STOP_REACTION_S + v * v / (2.0 * STOP_DECEL_MPS2)


def classify_geom(name: str, body_name: str = "") -> Optional[str]:
    """None = masked static map geom; ``animal`` for animal actor geoms; ``obstacle`` for everything else
    (persons, their ladders and poles, unknown objects)."""
    n = name or ""
    if n.startswith(ACTOR_PREFIX):
        return CLS_ANIMAL if n.startswith("actor_animal_") else CLS_OBSTACLE
    key = n or (body_name or "")
    if key.startswith(ACTOR_PREFIX):
        return CLS_ANIMAL if key.startswith("actor_animal_") else CLS_OBSTACLE
    if any(key.startswith(p) for p in MAP_PREFIXES):
        return None
    return CLS_OBSTACLE


@dataclass(frozen=True)
class Return:
    """One safety return in the base frame (x, y, z) with its class and source (``mast_*``, ``low_*``)."""
    x: float
    y: float
    z: float = 0.0
    cls: str = CLS_OBSTACLE
    geom: str = ""
    source: str = ""


# ---------------------------------------------------------------------------------------- geometry
def robot_hull(feet_xy: Iterable[Sequence[float]] = (), arm_xy: Iterable[Sequence[float]] = (),
               head_xy: Optional[Sequence[float]] = None, head_half: float = 0.06,
               trunk_half: Sequence[float] = TRUNK_HALF_XY, arm_pad: float = ARM_ENVELOPE_PAD_M) -> List[Tuple]:
    """Convex hull (CCW, base frame) of trunk box + feet + arm points padded by ``arm_pad`` + head footprint."""
    hx, hy = float(trunk_half[0]), float(trunk_half[1])
    pts: List[Tuple[float, float]] = [(hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)]
    pts += [(float(p[0]), float(p[1])) for p in feet_xy]
    for p in arm_xy:
        for k in range(8):
            a = k * math.pi / 4.0
            pts.append((float(p[0]) + arm_pad * math.cos(a), float(p[1]) + arm_pad * math.sin(a)))
    if head_xy is not None:
        for sx in (-1, 1):
            for sy in (-1, 1):
                pts.append((float(head_xy[0]) + sx * head_half, float(head_xy[1]) + sy * head_half))
    return convex_hull(pts)


def _seg_dist(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def hull_distance(p: Sequence[float], hull: Sequence[Sequence[float]]) -> float:
    """Distance (m) from point p to the convex hull: 0 inside (or on it), > 0 outside."""
    n = len(hull)
    if n == 0:
        return math.hypot(p[0], p[1])
    if n < 3:
        return min(_seg_dist(p, hull[i], hull[(i + 1) % n]) for i in range(n))
    inside = True
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
            inside = False
            break
    if inside:
        return 0.0
    return min(_seg_dist(p, hull[i], hull[(i + 1) % n]) for i in range(n))


def hull_distances(P: np.ndarray, hull: Sequence[Sequence[float]]) -> np.ndarray:
    """``hull_distance`` for N points (N x 2) at once (vectorised; same result)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 2)
    n = len(hull)
    if n == 0:
        return np.hypot(P[:, 0], P[:, 1])
    A = np.asarray(hull, dtype=np.float64)
    B = np.roll(A, -1, axis=0)
    E = B - A                                                   # (n, 2)
    D = P[:, None, :] - A[None, :, :]                           # (N, n, 2)
    L2 = np.maximum((E * E).sum(1), 1e-300)
    t = np.clip((D * E[None]).sum(2) / L2[None], 0.0, 1.0)
    Q = D - t[:, :, None] * E[None]
    dist = np.sqrt((Q * Q).sum(2)).min(1)
    if n >= 3:
        cross = E[None, :, 0] * D[:, :, 1] - E[None, :, 1] * D[:, :, 0]
        dist = np.where((cross >= -1e-12).all(1), 0.0, dist)
    return dist


def hull_centre(hull: Sequence[Sequence[float]]) -> Tuple[float, float]:
    if not hull:
        return (0.0, 0.0)
    return (sum(p[0] for p in hull) / len(hull), sum(p[1] for p in hull) / len(hull))


def sector(p: Sequence[float], centre: Sequence[float] = (0.0, 0.0)) -> str:
    b = abs(math.degrees(math.atan2(p[1] - centre[1], p[0] - centre[0])))
    if b <= FRONT_HALF_DEG:
        return "front"
    if b >= REAR_FROM_DEG:
        return "rear"
    return "side"


def field_offsets(mode: str, v: float = 0.0, lens_z: Optional[float] = None, ext: float = 0.0) -> Dict[str, dict]:
    """{protective: {front, side, rear}, warning: {...}} in metres beyond the hull (module table). ``v`` is the signed
    forward speed: the d_stop(v) term goes to the sector in the direction of travel (rear when walking in reverse,
    e.g. climbing nose-up > 5 deg); ``v=None`` assumes the speed cap, forward."""
    if mode not in MODES:
        raise ValueError(f"field mode {mode!r} not in {MODES}")
    if v is not None and math.isnan(v):
        v = None          # unknown speed = the speed cap, forward (NaN used to drop the d_stop term: WPCOV review)
    if mode in (MODE_WALK, MODE_ORBIT):
        prot = dict(WALK_PROTECTIVE)
        sec = "rear" if (v is not None and v < 0.0) else "front"
        prot[sec] += d_stop(min(abs(v), WALK_SPEED_CAP_MPS) if v is not None else WALK_SPEED_CAP_MPS)
        warn = dict(WALK_WARNING)
    else:
        prot, warn = dict(WORK_PROTECTIVE), dict(WORK_WARNING)
    if ext > CROWN_UP_EXT_M and lens_z is not None and lens_z > 0:
        for k in prot:
            prot[k] = max(prot[k], CROWN_PROTECTIVE_K * lens_z)
            warn[k] = max(warn[k], CROWN_WARNING_K * lens_z)
    return {FIELD_PROTECTIVE: prot, FIELD_WARNING: warn}


def field_polygon(hull: Sequence[Sequence[float]], offsets: dict, n: int = 48) -> List[List[float]]:
    """Display polygon of one field: for n bearings from the hull centre, the hull boundary + the sector offset."""
    c = hull_centre(hull)
    out = []
    A = np.asarray(hull, dtype=np.float64)
    E = np.roll(A, -1, axis=0) - A
    for k in range(n):
        a = 2.0 * math.pi * k / n
        u = (math.cos(a), math.sin(a))
        # exit point of the ray c + s u through the convex hull: solve c + s u = A_i + w E_i, 0 <= w <= 1
        lo = 0.0
        for (ax, ay), (ex, ey) in zip(A, E):
            den = u[0] * (-ey) - u[1] * (-ex)
            if abs(den) < 1e-12:
                continue
            bx, by = ax - c[0], ay - c[1]
            s = (bx * (-ey) - by * (-ex)) / den
            w = (u[0] * by - u[1] * bx) / den
            if s > lo and -1e-9 <= w <= 1.0 + 1e-9:
                lo = s
        p = (c[0] + lo * u[0], c[1] + lo * u[1])
        off = offsets[sector((p[0] + u[0], p[1] + u[1]), c)]
        out.append([round(p[0] + off * u[0], 3), round(p[1] + off * u[1], 3)])
    return out


@dataclass
class OrbitArc:
    """Planned orbit arc as robot poses (x, y, yaw) in the base frame of the current pose; the hull is swept along
    it with a ``corridor`` (OS 5.2)."""
    poses: List[Tuple[float, float, float]]
    corridor: float = ORBIT_CORRIDOR_M

    def distance(self, p: Sequence[float], hull: Sequence[Sequence[float]]) -> float:
        best = hull_distance(p, hull)
        for x, y, yaw in self.poses:
            c, s = math.cos(yaw), math.sin(yaw)
            q = ((p[0] - x) * c + (p[1] - y) * s, -(p[0] - x) * s + (p[1] - y) * c)
            best = min(best, hull_distance(q, hull))
        return max(0.0, best - self.corridor) if self.poses else best

    def distances(self, P: np.ndarray, hull: Sequence[Sequence[float]]) -> np.ndarray:
        """``distance`` for N points at once."""
        best = hull_distances(P, hull)
        for x, y, yaw in self.poses:
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = P[:, 0] - x, P[:, 1] - y
            best = np.minimum(best, hull_distances(np.stack([dx * c + dy * s, -dx * s + dy * c], 1), hull))
        return np.maximum(0.0, best - self.corridor) if self.poses else best


# ---------------------------------------------------------------------------------------- monitor
@dataclass
class FieldHit:
    field: str
    dist_m: float
    cls: str
    sector: str
    ret: Return


@dataclass
class FieldState:
    t: float
    mode: str
    hull: List[Tuple[float, float]]
    offsets: Dict[str, dict]
    protective: List[FieldHit] = field(default_factory=list)
    warning: List[FieldHit] = field(default_factory=list)
    latched: bool = False
    latch_reason: str = ""
    speed_scale: float = 1.0
    orbit_pause: bool = False
    arm_stop: bool = False
    blade_block: bool = False
    beacon: str = ""
    nearest_m: Optional[float] = None
    nearest_cls: str = ""

    @property
    def stop(self) -> bool:
        return self.latched

    def as_dict(self, polygons: bool = True) -> dict:
        d = {"t": round(self.t, 4), "mode": self.mode, "latched": self.latched, "latch_reason": self.latch_reason,
             "stop": self.stop, "speed_scale": self.speed_scale, "orbit_pause": self.orbit_pause,
             "arm_stop": self.arm_stop, "blade_block": self.blade_block, "beacon": self.beacon,
             "n_protective": len(self.protective), "n_warning": len(self.warning),
             "nearest_m": None if self.nearest_m is None else round(self.nearest_m, 3), "nearest_cls": self.nearest_cls,
             "offsets": {k: {s: round(v, 3) for s, v in o.items()} for k, o in self.offsets.items()},
             "assisted": ["map_masked_lidar"]}
        if polygons:
            d["hull"] = [[round(p[0], 3), round(p[1], 3)] for p in self.hull]
            d["polygons"] = {k: field_polygon(self.hull, o) for k, o in self.offsets.items()}
        return d


def animal_hits(hits: Iterable[FieldHit]) -> List[FieldHit]:
    """OS 5.2 animal stop: the hits (protective or warning field) of animal returns. ``FieldMonitor.update`` latches
    on any of them (never the person warning-field logic). One function so the interlock mutation check
    (``tests/interlock_mutants.py`` ``animal_stop``) disables the rule the live monitor runs."""
    return [h for h in hits if h.cls == CLS_ANIMAL]


class FieldMonitor:
    """Stateful field logic: classify every return into the protective / warning field and keep the latch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.latched = False
        self.latch_reason = ""
        self.latch_t: Optional[float] = None
        self.clear_since: Optional[float] = None
        self.events: List[dict] = []
        self.last: Optional[FieldState] = None

    def classify(self, returns: Iterable[Return], hull, offsets: Dict[str, dict],
                 arc: Optional[OrbitArc] = None) -> Tuple[List[FieldHit], List[FieldHit]]:
        prot: List[FieldHit] = []
        warn: List[FieldHit] = []
        rets = list(returns)
        if not rets:
            return prot, warn
        c = hull_centre(hull)
        P = np.array([(r.x, r.y) for r in rets], dtype=np.float64)
        dist = arc.distances(P, hull) if arc is not None else hull_distances(P, hull)
        b = np.degrees(np.abs(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])))
        secs = np.where(b <= FRONT_HALF_DEG, 0, np.where(b >= REAR_FROM_DEG, 2, 1))
        names = ("front", "side", "rear")
        lim_p = np.array([offsets[FIELD_PROTECTIVE][n] for n in names])[secs]
        lim_w = np.array([offsets[FIELD_WARNING][n] for n in names])[secs]
        for i in np.nonzero(dist <= lim_w)[0]:
            r, sec = rets[i], names[secs[i]]
            if dist[i] <= lim_p[i]:
                prot.append(FieldHit(FIELD_PROTECTIVE, float(dist[i]), r.cls, sec, r))
            else:
                warn.append(FieldHit(FIELD_WARNING, float(dist[i]), r.cls, sec, r))
        return prot, warn

    def update(self, t: float, mode: str, returns: Iterable[Return], hull, v: float = 0.0,
               lens_z: Optional[float] = None, ext: float = 0.0, arc: Optional[OrbitArc] = None,
               bumper: bool = False) -> FieldState:
        if mode == MODE_ORBIT and arc is None:
            arc = OrbitArc([])
        offsets = field_offsets(mode, v, lens_z, ext)
        prot, warn = self.classify(returns, hull, offsets, arc if mode == MODE_ORBIT else None)
        animal = animal_hits(prot + warn)
        reason = ""
        if bumper:
            reason = R_BUMPER
        elif animal:
            reason = R_ANIMAL
        elif prot:
            reason = R_PROTECTIVE
        if reason and not self.latched:
            self.latched, self.latch_reason, self.latch_t = True, reason, float(t)
            hit = (animal or prot or [None])[0]
            self.events.append({"t": round(float(t), 4), "reason": reason, "mode": mode,
                                "dist_m": None if hit is None else round(hit.dist_m, 3),
                                "source": "" if hit is None else hit.ret.source})
        clear_now = not prot and not animal and not bumper
        if clear_now:
            if self.clear_since is None:
                self.clear_since = float(t)
        else:
            self.clear_since = None
        st = FieldState(t=float(t), mode=mode, hull=list(hull), offsets=offsets, protective=prot, warning=warn,
                        latched=self.latched, latch_reason=self.latch_reason)
        person_warn = [h for h in warn if h.cls != CLS_ANIMAL]
        if self.latched:
            st.speed_scale, st.arm_stop, st.blade_block, st.beacon = 0.0, True, True, "fault"
            st.orbit_pause = mode == MODE_ORBIT
        elif person_warn:
            if mode == MODE_ORBIT:
                st.orbit_pause, st.speed_scale = True, 0.0
            elif mode == MODE_WALK:
                st.speed_scale = WARNING_SPEED_SCALE
            else:
                st.arm_stop, st.blade_block = True, True
            st.beacon = "warning_flash"
        hits = prot + warn
        if hits:
            h = min(hits, key=lambda x: x.dist_m)
            st.nearest_m, st.nearest_cls = h.dist_m, h.cls
        self.last = st
        return st

    def protective_clear(self, t: float) -> bool:
        return self.clear_since is not None and t - self.clear_since >= CLEAR_HOLD_S - 1e-9

    def clear(self, t: float) -> Tuple[bool, str]:
        """Operator clear of the protective latch: only while the field stayed clear for CLEAR_HOLD_S."""
        if not self.latched:
            return True, "not latched"
        if not self.protective_clear(t):
            return False, f"refused: {self.latch_reason} field not clear"
        reason = self.latch_reason
        self.latched, self.latch_reason, self.latch_t = False, "", None
        return True, f"protective latch cleared (was {reason})"


# ---------------------------------------------------------------------------------------- bunch exclusion
def bunch_exclusion(robot_xy: Sequence[float], marked_bunches: Iterable[Sequence[float]],
                    human_cut_in_progress: bool, hull_r: float = 0.0) -> GateDecision:
    """OS 5.2: while a human cut is flagged in progress the robot stays >= 3 m from every MARKed bunch."""
    pts = [tuple(b) for b in marked_bunches]
    if not human_cut_in_progress or not pts:
        return GateDecision.ok("bunch_exclusion")
    d = min(math.hypot(robot_xy[0] - b[0], robot_xy[1] - b[1]) for b in pts) - hull_r
    if d < BUNCH_EXCLUSION_M:
        return GateDecision.refuse("bunch_exclusion", "bunch_exclusion", dist_m=round(d, 3),
                                   limit_m=BUNCH_EXCLUSION_M)
    return GateDecision.ok("bunch_exclusion", dist_m=round(d, 3))


def to_base_frame(points_w: np.ndarray, base_xy: Sequence[float], yaw: float) -> np.ndarray:
    """World points (N x 3) into the base frame (x forward, y left, z unchanged)."""
    P = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = P[:, 0] - base_xy[0], P[:, 1] - base_xy[1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, P[:, 2]], axis=1)
