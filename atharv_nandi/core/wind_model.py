"""Scripted wind, gusts, anemometer and aerodynamic drag for Atharv Nandi (WP7).

ORCHARD-MVP-SPEC 3.6: scripted wind and gusts as ``xfrc_applied`` drag on the trunk and the mast sections,
F = 1/2 rho v^2 Cd A along the relative air velocity, Cd 1.2 for tubes (cylinder, capsule) and 1.5 for boxes
(ASSUMED; sphere 0.5). A is the exact silhouette of each geom perpendicular to the relative wind:

    box (half sizes a, b, c; axes e1..e3)  A = 4 (|u.e1| b c + |u.e2| a c + |u.e3| a b)
    cylinder (r, half length h, axis e3)   A = 4 r h sin(theta) + pi r^2 |cos(theta)|
    capsule                                A = 4 r h sin(theta) + pi r^2
    sphere                                 A = pi r^2

Nested coaxial tubes (the 3-section crown mast stowed inside its outer tube) are shadowed: the part of an inner
tube's length that lies inside a wider coaxial tube is not exposed, and its force acts at the exposed centre.
Each geom's force is applied at its (exposed) centre: ``xfrc_applied`` gets F and (p - body CoM) x F, so the
tipping moment of a raised mast is in physics. The hook keeps its own contribution and replaces it each tick, so
other writers of ``xfrc_applied`` (a WP6 test harness) are not overwritten.

Anemometer (crown head site ``anemometer`` from WP1; the hook uses the scripted wind at the head height): first
order lag 0.05 s (sonic, ASSUMED), 10 s running mean (EMA). **Gust flag**: filtered speed >= mean + 5 m/s
(WMO-style 10 kt peak-over-mean, ASSUMED) -> set on the first 50 Hz sample past it (<= 0.1 s after the true wind
crosses the same criterion); clears after 3 s below mean + 4 m/s. Gate thresholds of ORCHARD-MVP-SPEC 2.3 (raise
< 10 m/s and no gust, crown-up walk < 8 m/s, standing crown-up >= 10 m/s or gust -> retract) are WP6's decisions;
the state carries the inputs (``speed_mps``, ``gust``, ``mean_mps``) and the limits for display.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

RHO_AIR = 1.225           # kg/m^3, ISA sea level (ASSUMED; North India (UP) plain ~100 m, 25-40 C gives 1.12-1.18)
CD_TUBE = 1.2             # ASSUMED (ORCHARD-MVP-SPEC 3.6)
CD_BOX = 1.5              # ASSUMED
CD_SPHERE = 0.5           # ASSUMED
GUST_DELTA_MPS = 5.0      # ASSUMED gust criterion: filtered speed - mean >= 5 m/s
GUST_CLEAR_DELTA_MPS = 4.0
GUST_HOLD_S = 3.0
ANEMO_TAU_S = 0.05        # ASSUMED sonic anemometer response
MEAN_TAU_S = 10.0
# ORCHARD-MVP-SPEC 2.3 limits (WP6 decides; carried for the console)
WIND_RAISE_LIMIT_MPS = 10.0
WIND_WALK_CROWN_LIMIT_MPS = 8.0
WIND_RETRACT_STANDING_MPS = 10.0
DRAG_BODIES = ("trunk", "mast_base", "mast_outer", "mast_drive", "mast_s1", "mast_s2", "mast_s3",
               "crown_head", "mast_mock", "mast_head")
SKIP_GROUPS = (3,)        # v1 mass_only internal geoms (motors, battery) sit inside the shell: no drag
# atharv_nandi.xml (WP1) keeps its mass_only geoms in the robot group 2 with alpha 0: invisible geoms get no drag

# mjtGeom ids (stable across MuJoCo 3.x): plane 0, hfield 1, sphere 2, capsule 3, ellipsoid 4, cylinder 5, box 6
G_SPHERE, G_CAPSULE, G_ELLIPSOID, G_CYLINDER, G_BOX = 2, 3, 4, 5, 6


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _num(v) -> float:
    """A JSON number as float (bool, str and non-finite values raise ValueError)."""
    if not _finite(v):
        raise ValueError(f"not a finite number: {v!r}")
    return float(v)


def dynamic_pressure(v: float, rho: float = RHO_AIR) -> float:
    return 0.5 * rho * v * v


def drag_force_n(v: float, cd: float, area: float, rho: float = RHO_AIR) -> float:
    """|F| = 1/2 rho v^2 Cd A."""
    return dynamic_pressure(v, rho) * cd * area


def _col(xmat: Sequence[float], j: int) -> Tuple[float, float, float]:
    return (xmat[j], xmat[3 + j], xmat[6 + j])   # row-major 3 x 3, column j = body axis j in world


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cd_for(gtype: int) -> float:
    if gtype == G_BOX:
        return CD_BOX
    if gtype in (G_CYLINDER, G_CAPSULE):
        return CD_TUBE
    return CD_SPHERE


def projected_area(gtype: int, size: Sequence[float], xmat: Sequence[float], u: Sequence[float],
                   exposed_frac: float = 1.0) -> float:
    """Silhouette area of a geom perpendicular to unit vector ``u`` (world). ``exposed_frac`` scales a tube's side."""
    if gtype == G_BOX:
        a, b, c = size[0], size[1], size[2]
        e1, e2, e3 = _col(xmat, 0), _col(xmat, 1), _col(xmat, 2)
        return 4.0 * (abs(_dot(u, e1)) * b * c + abs(_dot(u, e2)) * a * c + abs(_dot(u, e3)) * a * b)
    if gtype in (G_CYLINDER, G_CAPSULE):
        r, h = size[0], size[1]
        ct = max(-1.0, min(1.0, _dot(u, _col(xmat, 2))))
        st = math.sqrt(max(0.0, 1.0 - ct * ct))
        side = 4.0 * r * h * st * exposed_frac
        end = math.pi * r * r * (1.0 if gtype == G_CAPSULE else abs(ct))
        return side + end
    if gtype == G_SPHERE:
        return math.pi * size[0] * size[0]
    if gtype == G_ELLIPSOID:   # sphere of the mean radius (approximation, ASSUMED)
        r = (size[0] + size[1] + size[2]) / 3.0
        return math.pi * r * r
    return 0.0


@dataclass
class TubeSpan:
    """A vertical tube for the coaxial shadowing rule."""
    key: int
    x: float
    y: float
    r: float
    z0: float
    z1: float


def exposed_spans(tubes: Sequence[TubeSpan], axis_tol: float = 0.01) -> Dict[int, Tuple[float, float, float]]:
    """key -> (exposed fraction, exposed centre z, length): a tube's length inside a wider coaxial tube is hidden."""
    out = {}
    for t in tubes:
        cover = []
        for o in tubes:
            if o.key == t.key or o.r <= t.r + 1e-9:
                continue
            if math.hypot(o.x - t.x, o.y - t.y) > axis_tol + (o.r - t.r):
                continue
            lo, hi = max(t.z0, o.z0), min(t.z1, o.z1)
            if hi > lo:
                cover.append((lo, hi))
        cover.sort()
        merged: List[List[float]] = []
        for lo, hi in cover:
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        # exposed intervals = [z0, z1] minus merged
        exp, cur = [], t.z0
        for lo, hi in merged:
            if lo > cur:
                exp.append((cur, lo))
            cur = max(cur, hi)
        if cur < t.z1:
            exp.append((cur, t.z1))
        length = t.z1 - t.z0
        e_len = sum(hi - lo for lo, hi in exp)
        zc = (sum((hi * hi - lo * lo) / 2.0 for lo, hi in exp) / e_len) if e_len > 1e-12 else (t.z0 + t.z1) / 2
        out[t.key] = (e_len / length if length > 0 else 1.0, zc, length)
    return out


# ------------------------------------------------------------------ scripted wind
@dataclass(frozen=True)
class Gust:
    t0: float
    peak_mps: float
    rise_s: float = 0.0
    hold_s: float = 2.0
    fall_s: float = 1.0
    dir_deg: Optional[float] = None    # None = the base direction

    def __post_init__(self):
        for k in ("t0", "peak_mps", "rise_s", "hold_s", "fall_s"):
            v = getattr(self, k)
            if not _finite(v) or (k != "t0" and v < 0):
                raise ValueError(f"Gust.{k} must be finite (and >= 0)")
        if self.dir_deg is not None and not _finite(self.dir_deg):
            raise ValueError("Gust.dir_deg must be finite (or None)")

    def factor(self, t: float) -> float:
        """0..1 envelope: linear rise, hold, linear fall."""
        a = t - self.t0
        if a < 0:
            return 0.0
        if a < self.rise_s:
            return a / self.rise_s
        a -= self.rise_s
        if a < self.hold_s:
            return 1.0
        a -= self.hold_s
        if a < self.fall_s:
            return 1.0 - a / self.fall_s
        return 0.0


@dataclass
class WindScript:
    """Base wind (speed at z_ref, direction the wind blows TOWARD, deg from +x / East, CCW) + gusts + optional
    seeded turbulence (Ornstein-Uhlenbeck on the speed, intensity ``turb_sigma_mps``, off by default) + optional
    power-law shear (alpha 0 = uniform with height, default)."""
    speed_mps: float = 0.0
    dir_deg: float = 90.0
    gusts: List[Gust] = field(default_factory=list)
    turb_sigma_mps: float = 0.0
    turb_tau_s: float = 2.0
    shear_alpha: float = 0.0
    z_ref_m: float = 2.0
    seed: int = 7

    def __post_init__(self):
        if not (_finite(self.speed_mps) and self.speed_mps >= 0):
            raise ValueError("wind speed must be finite and >= 0")
        if not _finite(self.dir_deg):
            raise ValueError("wind dir_deg must be finite")
        if not (_finite(self.turb_sigma_mps) and self.turb_sigma_mps >= 0):
            raise ValueError("turb_sigma_mps must be finite and >= 0")
        if not (_finite(self.turb_tau_s) and self.turb_tau_s > 0):
            raise ValueError("turb_tau_s must be finite and > 0")
        if not (_finite(self.shear_alpha) and 0.0 <= self.shear_alpha <= 1.0):
            raise ValueError("shear_alpha must be in [0, 1]")
        if not (_finite(self.z_ref_m) and self.z_ref_m > 0):
            raise ValueError("z_ref_m must be finite and > 0")
        if not all(isinstance(g, Gust) for g in self.gusts):
            raise ValueError("gusts must be Gust objects")
        self._rng = random.Random(self.seed + 303)
        self._turb, self._turb_t = 0.0, None

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "WindScript":
        """Strict parse (a bad field raises ValueError / KeyError / TypeError; the caller rejects the command)."""
        if d is not None and not isinstance(d, dict):
            raise TypeError("wind script must be a JSON object")
        d = dict(d or {})
        raw = d.pop("gusts", None) or []
        if not isinstance(raw, list) or not all(isinstance(g, dict) for g in raw):
            raise TypeError("gusts must be a list of objects")
        gusts = [Gust(_num(g["t0"]), _num(g["peak_mps"]), _num(g.get("rise_s", 0.0)), _num(g.get("hold_s", 2.0)),
                      _num(g.get("fall_s", 1.0)), None if g.get("dir_deg") is None else _num(g["dir_deg"]))
                 for g in raw]
        kw = {k: (int(v) if k == "seed" else _num(v)) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(gusts=gusts, **kw)

    @property
    def calm(self) -> bool:
        return self.speed_mps == 0.0 and not self.gusts and self.turb_sigma_mps == 0.0

    def reset(self):
        self._rng = random.Random(self.seed + 303)
        self._turb, self._turb_t = 0.0, None

    def _turbulence(self, t: float) -> float:
        if self.turb_sigma_mps <= 0:
            return 0.0
        if self._turb_t is None or t <= self._turb_t:
            self._turb_t = t
            return self._turb
        dt = t - self._turb_t
        a = math.exp(-dt / self.turb_tau_s)
        self._turb = a * self._turb + self.turb_sigma_mps * math.sqrt(1 - a * a) * self._rng.gauss(0.0, 1.0)
        self._turb_t = t
        return self._turb

    def velocity(self, t: float, z: Optional[float] = None) -> Tuple[float, float, float]:
        """World wind velocity (vx, vy, 0) at time t and height z (gusts add to the base speed)."""
        base = max(0.0, self.speed_mps + self._turbulence(t))
        th = math.radians(self.dir_deg)
        vx, vy = base * math.cos(th), base * math.sin(th)
        for g in self.gusts:
            f = g.factor(t)
            if f > 0:
                # extra speed e along the gust direction u so that |v + e u| = peak_mps at the envelope top
                # (same direction as the base: e = peak - base, as before; a crosswind gust reaches its peak too)
                gd = math.radians(self.dir_deg if g.dir_deg is None else g.dir_deg)
                ux, uy = math.cos(gd), math.sin(gd)
                bu = vx * ux + vy * uy
                v2 = vx * vx + vy * vy
                # peak at or below the current speed: no extra (a gust never slows or reverses the wind)
                p2 = g.peak_mps * g.peak_mps
                extra = (-bu + math.sqrt(bu * bu - v2 + p2)) * f if p2 > v2 else 0.0
                vx += extra * ux
                vy += extra * uy
        if self.shear_alpha and z is not None:
            s = (max(z, 0.05) / self.z_ref_m) ** self.shear_alpha
            vx, vy = vx * s, vy * s
        return (vx, vy, 0.0)

    def speed(self, t: float, z: Optional[float] = None) -> float:
        vx, vy, _ = self.velocity(t, z)
        return math.hypot(vx, vy)


class Anemometer:
    """First-order lagged speed + running mean + gust flag (sampled by the hook at 50 Hz)."""

    def __init__(self, tau_s: float = ANEMO_TAU_S, mean_tau_s: float = MEAN_TAU_S, gust_delta: float = GUST_DELTA_MPS,
                 clear_delta: float = GUST_CLEAR_DELTA_MPS, hold_s: float = GUST_HOLD_S):
        self.tau_s, self.mean_tau_s = float(tau_s), float(mean_tau_s)
        self.gust_delta, self.clear_delta, self.hold_s = float(gust_delta), float(clear_delta), float(hold_s)
        self.reset()

    def reset(self):
        self.speed = 0.0
        self.mean = 0.0
        self.peak = 0.0
        self.gust = False
        self.gust_onset_t: Optional[float] = None
        self.dir_deg = 0.0
        self._below_since: Optional[float] = None
        self._t: Optional[float] = None
        self.gust_count = 0

    def update(self, t: float, wind_xy: Sequence[float]) -> dict:
        v = math.hypot(wind_xy[0], wind_xy[1])
        if self._t is None:
            self.speed = self.mean = v
        elif t > self._t:
            dt = t - self._t
            self.speed += (1.0 - math.exp(-dt / self.tau_s)) * (v - self.speed) if self.tau_s > 0 else v - self.speed
            # 10 s EMA of the filtered speed (a 2 s gust lifts it by ~18 % of the excess; sustained wind clears)
            self.mean += (1.0 - math.exp(-dt / self.mean_tau_s)) * (self.speed - self.mean)
        if self._t is None or t > self._t:
            self._t = t
        if v > 1e-6:
            self.dir_deg = math.degrees(math.atan2(wind_xy[1], wind_xy[0])) % 360.0
        self.peak = max(self.peak, self.speed)
        if not self.gust and self.speed >= self.mean + self.gust_delta:
            self.gust, self.gust_onset_t, self._below_since = True, t, None
            self.gust_count += 1
        elif self.gust:
            if self.speed < self.mean + self.clear_delta:
                if self._below_since is None:
                    self._below_since = t
                elif t - self._below_since >= self.hold_s - 1e-9:
                    self.gust, self._below_since = False, None
            else:
                self._below_since = None
        return self.state(t)

    def state(self, t: Optional[float] = None) -> dict:
        return {"t": None if t is None else round(t, 4), "speed_mps": round(self.speed, 3),
                "mean_mps": round(self.mean, 3), "peak_mps": round(self.peak, 3), "dir_deg": round(self.dir_deg, 1),
                "gust": self.gust, "gust_onset_t": self.gust_onset_t, "gust_count": self.gust_count,
                "limits": {"raise_mps": WIND_RAISE_LIMIT_MPS, "walk_crown_up_mps": WIND_WALK_CROWN_LIMIT_MPS,
                           "retract_standing_mps": WIND_RETRACT_STANDING_MPS},
                "source": "scripted", "assumed": True}


def gust_criterion_onset(script: WindScript, t_from: float, t1: float, dt: float = 1e-3,
                         delta: float = GUST_DELTA_MPS, mean_tau_s: float = MEAN_TAU_S) -> Optional[float]:
    """Truth onset of a gust: the first t in [t_from, t1] at which the TRUE scripted speed reaches its own running
    mean (the same 10 s EMA the anemometer uses, started at t_from) + delta. The flag latency of the anemometer
    against this is its response lag plus the 50 Hz sampling."""
    mean = script.speed(t_from)
    n = int(math.ceil((t1 - t_from) / dt))
    a = 1.0 - math.exp(-dt / mean_tau_s)
    for i in range(n + 1):
        t = t_from + i * dt
        v = script.speed(t)
        if v >= mean + delta:
            return t
        mean += a * (v - mean)
    return None


# ------------------------------------------------------------------ MuJoCo drag application
@dataclass
class DragElement:
    geom: int
    body: int
    gtype: int
    cd: float


class WindDrag:
    """Resolves the drag geoms of a model by body name and writes their drag into ``d.xfrc_applied``."""

    def __init__(self, m, bodies: Sequence[str] = DRAG_BODIES, rho: float = RHO_AIR):
        import mujoco  # noqa: WPS433 - optional at import time of the pure helpers
        self._mj = mujoco
        self.rho = float(rho)
        self.elements: List[DragElement] = []
        self.body_names = []
        for name in bodies:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                continue
            self.body_names.append(name)
            for g in range(m.ngeom):
                if (m.geom_bodyid[g] != bid or int(m.geom_group[g]) in SKIP_GROUPS
                        or float(m.geom_rgba[g][3]) <= 0.0):
                    continue
                gt = int(m.geom_type[g])
                if gt in (G_SPHERE, G_CAPSULE, G_ELLIPSOID, G_CYLINDER, G_BOX):
                    self.elements.append(DragElement(g, bid, gt, cd_for(gt)))
        self._applied: Dict[int, List[float]] = {}
        self.last_force_n = 0.0
        self.last_forces: Dict[int, Tuple[float, float, float]] = {}
        self.last_areas: Dict[int, float] = {}

    def clear(self, d=None):
        """Remove this model's contribution (d given) and forget it (after mj_resetData, d is None)."""
        if d is not None:
            for b, w in self._applied.items():
                for i in range(6):
                    d.xfrc_applied[b, i] -= w[i]
        self._applied = {}
        self.last_force_n = 0.0
        self.last_forces, self.last_areas = {}, {}

    def _tubes(self, m, d) -> Dict[int, Tuple[float, float, float]]:
        spans = []
        for e in self.elements:
            if e.gtype not in (G_CYLINDER, G_CAPSULE):
                continue
            xm = d.geom_xmat[e.geom]
            if abs(xm[8]) < 0.99:     # axis not vertical: no shadowing
                continue
            p = d.geom_xpos[e.geom]
            h = float(m.geom_size[e.geom][1])
            spans.append(TubeSpan(e.geom, float(p[0]), float(p[1]), float(m.geom_size[e.geom][0]),
                                  float(p[2]) - h, float(p[2]) + h))
        return exposed_spans(spans) if len(spans) > 1 else {}

    def apply(self, m, d, script: WindScript, t: float) -> float:
        """Replace this hook's drag in d.xfrc_applied; returns the total |F| (N)."""
        mj = self._mj
        new: Dict[int, List[float]] = {}
        shadow = self._tubes(m, d)
        vel_arr = _np_zeros6()
        total = [0.0, 0.0, 0.0]
        self.last_forces, self.last_areas = {}, {}
        for e in self.elements:
            p = [float(v) for v in d.geom_xpos[e.geom]]
            frac = 1.0
            if e.geom in shadow:
                frac, zc, _ = shadow[e.geom]
                p[2] = zc
                if frac <= 1e-9 and e.gtype == G_CYLINDER:
                    continue   # fully inside a wider tube (ends hidden too)
            wx, wy, wz = script.velocity(t, p[2])
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_GEOM, e.geom, vel_arr, 0)
            rel = (wx - vel_arr[3], wy - vel_arr[4], wz - vel_arr[5])
            v = math.sqrt(_dot(rel, rel))
            if v < 1e-9:
                continue
            u = (rel[0] / v, rel[1] / v, rel[2] / v)
            area = projected_area(e.gtype, m.geom_size[e.geom], d.geom_xmat[e.geom], u, frac)
            f = drag_force_n(v, e.cd, area, self.rho)
            fv = (f * u[0], f * u[1], f * u[2])
            c = d.xipos[e.body]
            r = (p[0] - c[0], p[1] - c[1], p[2] - c[2])
            tq = (r[1] * fv[2] - r[2] * fv[1], r[2] * fv[0] - r[0] * fv[2], r[0] * fv[1] - r[1] * fv[0])
            w = new.setdefault(e.body, [0.0] * 6)
            for i in range(3):
                w[i] += fv[i]
                w[3 + i] += tq[i]
                total[i] += fv[i]
            self.last_forces[e.geom] = fv
            self.last_areas[e.geom] = area
        if not all(math.isfinite(x) for w in new.values() for x in w):
            self.clear(d)
            raise ValueError("non-finite wind drag (own contribution removed)")
        for b, w in self._applied.items():
            for i in range(6):
                d.xfrc_applied[b, i] -= w[i]
        for b, w in new.items():
            for i in range(6):
                d.xfrc_applied[b, i] += w[i]
        self._applied = new
        self.last_force_n = math.sqrt(_dot(total, total))
        return self.last_force_n


def _np_zeros6():
    import numpy as np
    return np.zeros(6)
