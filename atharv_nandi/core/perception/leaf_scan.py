"""WP26 LEAF SCAN: leaf selection from the WP23 ``leaf_k`` targets, clearance-keeping upper and no-contact underside
views, colony/symptom-level leads, ``LeafRecord`` rows and the TreeRecord ``leaf`` block (MISSION-PRESETS 3;
ORCHARD-MVP-SPEC 2.5, 3.3).

ROS-free, MuJoCo-free. Geometry is in the world frame (tree plan positions + the plan's ``pos``).

* **Selection** (:func:`select_leaves`): in-band leaves (z ≤ 0.8 m) stratified over 4 azimuth quadrants × leaf
  kind (young flush / mature), each imaged on both sides; leaves above the band get a crown view labelled
  ``not_close_up`` (crown at 1.10 m, tilt ≤ +45°). Papaya: ``n/a_crown_only`` (no leaves in the pick band);
  banana: only in-band dead or drooping blades, else ``n/a_crown_only``. Selection never reads decal truth.
* **Views** (:func:`plan_view`): upper side from above along the leaf normal, underside from below looking up
  (axis within 35° of the normal and z ≥ 0.2, the WP0c sweep rule), stand-off 0.12-0.20 m (0.15 preferred).
  A candidate is kept only when a wrist proxy (capsule behind the lens) keeps **≥ 2 cm** from every leaf quad,
  wood capsule, prune target and fruit of the tree along the whole straight approach (10 cm along the axis).
* **Runtime guard** (:class:`ClearanceGuard`): latches a stop when a measured clearance falls below 2 cm, the
  caller retracts and the side is logged ``no_clearance``; base drift > 2 cm pauses the arm for a re-approach; a NaN
  clearance or drift reading stops (``sensor_invalid``).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi.core.perception import pest_detect as pdet
from atharv_nandi.sim.world import trees as gen

N_A = "n/a_crown_only"
LABEL = "colony_symptom_level"       # OS 2.5: sim LEAF SCAN is colony- or symptom-level evidence, not insect level


# ---------------------------------------------------------------------------------------------- distances
def seg_seg_dist(p1, q1, p2, q2) -> float:
    """Closest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    p1, q1, p2, q2 = (np.asarray(x, float) for x in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e <= 1e-12:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def point_box_dist(p, c, R, half) -> float:
    q = R.T @ (np.asarray(p, float) - np.asarray(c, float))
    d = np.maximum(np.abs(q) - np.asarray(half, float), 0.0)
    return float(np.linalg.norm(d))


@dataclass
class Obstacle:
    kind: str            # capsule | box | sphere
    name: str
    role: str            # leaf | wood | prune | fruit | bunch
    a: np.ndarray = None
    b: np.ndarray = None
    r: float = 0.0
    c: np.ndarray = None
    R: np.ndarray = None
    half: np.ndarray = None

    def bound(self) -> Tuple[np.ndarray, float]:
        """Bounding sphere (centre, radius) for culling."""
        if self.kind == "capsule":
            return 0.5 * (self.a + self.b), 0.5 * float(np.linalg.norm(self.b - self.a)) + self.r
        if self.kind == "sphere":
            return self.c, self.r
        return self.c, float(np.linalg.norm(self.half))

    def dist_to_capsule(self, a, b, r) -> float:
        if self.kind == "capsule":
            return seg_seg_dist(a, b, self.a, self.b) - r - self.r
        if self.kind == "sphere":
            return seg_seg_dist(a, b, self.c, self.c) - r - self.r
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / 0.002)) + 1)
        P = a[None, :] + (b - a)[None, :] * np.linspace(0.0, 1.0, n)[:, None]
        q = (P - self.c[None, :]) @ self.R
        return float(np.linalg.norm(np.maximum(np.abs(q) - self.half[None, :], 0.0), axis=1).min()) - r


def tree_obstacles(plan: dict, roles: Sequence[str] = ("leaf", "wood", "prune", "fruit", "bag", "bunch", "decal")) -> "ObstacleSet":
    """Collision-relevant geoms of a tree plan in the world frame: leaf quads (boxes), keep wood and prune targets
    (capsules / boxes), fruit and bags (bounding spheres, conservative), the banana bunch, decals (enclosing boxes).
    Foliage meshes (group 3 visual) are not obstacles; the foliage envelope is a query volume for the crown, not the wrist."""
    base = np.asarray(plan["pos"], float)
    out: List[Obstacle] = []
    for g in plan["geoms"]:
        k = g["kind"]
        if k not in roles:
            continue
        if g["type"] == "capsule" and "fromto" in g:
            ft = g["fromto"]
            out.append(Obstacle("capsule", g["name"], k, a=base + ft[:3], b=base + ft[3:], r=float(g["size"][0])))
        elif g["type"] == "box":
            out.append(Obstacle("box", g["name"], k, c=base + g["pos"], R=pdet.quat_mat(g.get("quat", [1, 0, 0, 0])),
                                half=np.asarray(g["size"], float)))
        elif g["type"] == "cylinder":      # decals: the enclosing box (conservative)
            out.append(Obstacle("box", g["name"], k, c=base + g["pos"], R=pdet.quat_mat(g.get("quat", [1, 0, 0, 0])),
                                half=np.array([g["size"][0], g["size"][0], g["size"][1]], float)))
        elif g["type"] in ("ellipsoid", "sphere"):
            out.append(Obstacle("sphere", g["name"], k, c=base + g["pos"], r=float(max(g["size"]))))
    return ObstacleSet(out)


class ObstacleSet:
    """Obstacles with vectorised bounding-sphere culling: exact distances only for those whose bound could beat
    the current minimum (boxes: the proxy axis sampled at <= 2 mm, exact for the thin leaf quads to ~1e-4 m)."""

    def __init__(self, obstacles: Sequence[Obstacle]):
        self.items = list(obstacles)
        if self.items:
            b = [o.bound() for o in self.items]
            self.bc = np.array([x[0] for x in b], float)
            self.br = np.array([x[1] for x in b], float)
        else:
            self.bc, self.br = np.zeros((0, 3)), np.zeros(0)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def min_dist(self, a, b, r) -> Tuple[float, str]:
        if not self.items:
            return float("inf"), ""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L2 = float(d @ d)
        t = np.clip(((self.bc - a) @ d) / max(L2, 1e-12), 0.0, 1.0)
        lower = np.linalg.norm(self.bc - (a[None, :] + t[:, None] * d[None, :]), axis=1) - self.br - r
        best, who = float("inf"), ""
        for i in np.argsort(lower):
            if lower[i] >= best:
                break
            v = self.items[i].dist_to_capsule(a, b, r)
            if v < best:
                best, who = v, self.items[i].name
        return best, who


def _as_set(obstacles) -> ObstacleSet:
    return obstacles if isinstance(obstacles, ObstacleSet) else ObstacleSet(obstacles)


@dataclass
class WristProxy:
    """Wrist camera head: a capsule from the lens back along the optical axis."""
    r_m: float = 0.025
    len_m: float = 0.10

    def segment(self, lens, fwd) -> Tuple[np.ndarray, np.ndarray]:
        lens = np.asarray(lens, float)
        return lens, lens - pdet.unit(fwd) * self.len_m


def clearance(obstacles, proxy: WristProxy, lens, fwd) -> Tuple[float, str]:
    a, b = proxy.segment(lens, fwd)
    return _as_set(obstacles).min_dist(a, b, proxy.r_m)


def approach_clearance(obstacles, proxy: WristProxy, lens, fwd, approach_m: float, samples: int = 0
                       ) -> Tuple[float, str]:
    """Min clearance over the straight approach (lens from ``lens - approach·fwd`` to ``lens``). The proxy moves
    along its own axis, so the swept volume is one capsule from the lens back ``approach + len`` (exact)."""
    lens, fwd = np.asarray(lens, float), pdet.unit(fwd)
    return _as_set(obstacles).min_dist(lens, lens - fwd * (approach_m + proxy.len_m), proxy.r_m)


# ---------------------------------------------------------------------------------------------- sensor values
def _num(x) -> Optional[float]:
    """``float(x)`` or None when ``x`` is not a number (bool, str, None, ...)."""
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _finite(x) -> bool:
    """A usable sensor reading: a finite number (NaN / ±inf / non-numeric are refused, never passed as clean)."""
    v = _num(x)
    return v is not None and math.isfinite(v)


# ---------------------------------------------------------------------------------------------- guard
class ClearanceGuard:
    """Runtime wrist-leaf clearance interlock (MP 3): below ``min_m`` → latched ``stop`` (retract, log)."""

    def __init__(self, min_m: float = 0.02, drift_max_m: float = 0.02):
        self.min_m, self.drift_max_m = float(min_m), float(drift_max_m)
        self.latched = False
        self.min_seen = float("inf")
        self.log: List[dict] = []

    def update(self, clearance_m: float, t: float = 0.0, who: str = "") -> str:
        clearance_m = _num(clearance_m)
        if clearance_m is None or math.isnan(clearance_m):          # an unreadable clearance is a violation
            if not self.latched:
                self.log.append({"t": t, "event": "sensor_invalid", "what": "clearance_m", "geom": who,
                                 "action": "retract"})
            self.latched = True
            return "stop"
        self.min_seen = min(self.min_seen, clearance_m)
        if clearance_m < self.min_m:
            if not self.latched:
                self.log.append({"t": t, "event": "clearance_violation", "clearance_m": round(float(clearance_m), 4),
                                 "geom": who, "action": "retract"})
            self.latched = True
        return "stop" if self.latched else "ok"

    def base_drift(self, drift_m: float, t: float = 0.0) -> str:
        if not _finite(drift_m):
            self.log.append({"t": t, "event": "sensor_invalid", "what": "drift_m", "action": "stop"})
            return "sensor_invalid"
        drift_m = float(drift_m)
        if drift_m > self.drift_max_m:
            self.log.append({"t": t, "event": "base_drift", "drift_m": round(float(drift_m), 4),
                             "action": "pause_reapproach"})
            return "pause_reapproach"
        return "ok"

    def reset(self):
        self.latched = False


# ---------------------------------------------------------------------------------------------- selection
def _quadrant(plan: dict, p_world) -> int:
    local = np.asarray(p_world, float) - np.asarray(plan["pos"], float)
    ax, ay = pdet._alley_dir(plan.get("entry"))
    ref = math.atan2(-ay, -ax)        # alley side first
    ang = (math.atan2(local[1], local[0]) - ref + math.pi / 4.0) % (2 * math.pi)
    return int(ang // (math.pi / 2.0))


def leaf_applicability(plan: dict) -> Tuple[bool, str]:
    """(True, '') when the crop has leaves in the band; else (False, 'n/a_crown_only') (MP 3 preconditions)."""
    crop = plan["crop"]
    leaves = plan["truth"].get("leaves", [])
    if crop == "papaya":
        return False, N_A
    if crop == "banana":
        ok = any(lf["in_band"] and (not lf.get("functional", True) or "droop" in str(lf.get("kind", "")))
                 for lf in leaves)
        return (True, "") if ok else (False, N_A)
    if not any(lf["in_band"] for lf in leaves):
        return False, N_A
    return True, ""


def select_leaves(plan: dict, n_leaves: Optional[int] = None, demo: bool = False, branch: Optional[dict] = None,
                  cfg: Optional[dict] = None) -> dict:
    """Leaf targets for one tree: ``{status, close_up[], not_close_up[]}``. ``branch`` ``{clock, h_m}`` (from a
    flag) puts the leaves nearest that pin first. Never reads decal truth."""
    c = (cfg or pdet.load_config())["leaf_scan"]
    ok, why = leaf_applicability(plan)
    if not ok:
        return {"status": why, "close_up": [], "not_close_up": []}
    leaves = list(plan["truth"]["leaves"])
    if plan["crop"] == "banana":
        leaves = [lf for lf in leaves if not lf.get("functional", True) or "droop" in str(lf.get("kind", ""))]
    band = float(c["pick_band_z_m"])
    inb = [lf for lf in leaves if lf["in_band"] and float(lf["world"][2]) <= band + 1e-9]
    high = [lf for lf in leaves if lf not in inb]
    if demo:
        n_close, n_high = int(c["demo_n_leaves"]), int(c["demo_high_n"])
    else:
        n_total = int(n_leaves if n_leaves is not None else c["n_leaves"])
        n_close = min(n_total, len(inb))
        n_high = max(0, n_total - n_close)

    def key(lf):
        if branch:
            clk = abs(((int(lf.get("clock", pdet.clock_of_world(plan, lf["world"]))) - int(branch.get("clock", 6)) + 6)
                       % 12) - 6)
            return (clk + abs(float(lf["world"][2]) - float(branch.get("h_m", 0.5))) * 10.0, lf["k"])
        return (0, lf["k"])

    # stratify: round-robin over (quadrant, kind) cells, each cell in key order
    cells: Dict[Tuple[int, str], List[dict]] = {}
    for lf in sorted(inb, key=key):
        cells.setdefault((_quadrant(plan, lf["world"]), str(lf.get("kind"))), []).append(lf)
    order = sorted(cells, key=lambda ck: (key(cells[ck][0]), ck))
    picked: List[dict] = []
    while len(picked) < min(n_close, len(inb)):
        for ck in order:
            if cells[ck] and len(picked) < n_close:
                picked.append(cells[ck].pop(0))
    hi = sorted(high, key=key)[:n_high]
    return {"status": "ok", "close_up": picked, "not_close_up": hi}


# ---------------------------------------------------------------------------------------------- view planning
def cone_axes(axis, cone_deg: float, steps: int) -> List[np.ndarray]:
    """``axis`` first, then rings at cone/2 and cone (``steps`` azimuths each)."""
    a = pdet.unit(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = pdet.unit(np.cross(a, ref))
    v = np.cross(a, u)
    out = [a]
    for th in (cone_deg / 2.0, cone_deg):
        t = math.radians(th)
        for k in range(steps):
            ph = 2 * math.pi * k / steps
            out.append(pdet.unit(math.cos(t) * a + math.sin(t) * (math.cos(ph) * u + math.sin(ph) * v)))
    return out


@dataclass
class LeafView:
    leaf_id: str
    side: str                  # upper | under | crown
    view: pdet.View
    standoff_m: float
    clearance_m: float
    clear_geom: str
    mm_per_px: float
    ok: bool = True
    reason: str = ""

    def row(self) -> dict:
        return {"leaf_id": self.leaf_id, "side": self.side, "standoff_m": round(self.standoff_m, 3),
                "clearance_m": None if not math.isfinite(self.clearance_m) else round(self.clearance_m, 4),
                "clear_geom": self.clear_geom, "mm_per_px": round(self.mm_per_px, 3), "ok": self.ok,
                "reason": self.reason, "pose": [round(x, 4) for x in self.view.pose7()]}


def plan_view(leaf: dict, side: str, obstacles: Sequence[Obstacle], cfg: Optional[dict] = None) -> LeafView:
    """Best clearance-keeping wrist view of one leaf side; ``ok`` False with ``reason`` ``no_clearance`` when no
    candidate keeps ``min_clearance_m`` over the approach."""
    full = cfg or pdet.load_config()
    c = full["leaf_scan"]
    cam = full["cameras"]["wrist"]
    proxy = WristProxy(**c["wrist_proxy"])
    p = np.asarray(leaf["world"], float)
    n = pdet.unit(leaf["n"])
    n_up = n if n[2] >= 0 else -n
    s_pref = float(c["standoff_m"])
    lo, hi = c["standoff_range_m"]
    standoffs = sorted({round(s_pref, 3), round(lo, 3), round(hi, 3), round(0.5 * (lo + s_pref), 3)},
                       key=lambda s: abs(s - s_pref))
    min_c, min_z = float(c["min_clearance_m"]), float(c["min_z_m"])
    best: Optional[LeafView] = None
    best_fail: Optional[LeafView] = None
    for s in standoffs:
        if s < min_z:
            continue
        for a in cone_axes(n_up, float(c["view_cone_deg"]), int(c["view_cone_steps"])):
            if side == "under":
                if a[2] < float(c["underside_min_up"]):
                    continue
                lens, fwd = p - a * s, a
            else:
                lens, fwd = p + a * s, -a
            if lens[2] < 0.05:
                continue
            clr, who = approach_clearance(obstacles, proxy, lens, fwd, float(c["approach_m"]),
                                          int(c["approach_samples"]))
            v = pdet.View(pos=lens, fwd=fwd, fovy_deg=cam["fovy_deg"], width=cam["width"], height=cam["height"],
                          camera="wrist_cam", band="low", tag={"leaf": leaf["id"], "side": side})
            lv = LeafView(leaf["id"], side, v, s, clr, who, v.mm_per_px(s))
            if clr >= min_c:
                if best is None or clr > best.clearance_m + 0.005:
                    best = lv
            elif best_fail is None or clr > best_fail.clearance_m:
                best_fail = lv
        if best is not None:
            return best
    if best_fail is None:
        v = pdet.View(pos=p - n_up * s_pref, fwd=n_up, camera="wrist_cam")
        best_fail = LeafView(leaf["id"], side, v, s_pref, float("-inf"), "", v.mm_per_px(s_pref))
    best_fail.ok, best_fail.reason = False, "no_clearance"
    return best_fail


def crown_view(plan: dict, leaf: dict, cfg: Optional[dict] = None) -> LeafView:
    """Leaves above the band: crown camera at 1.10 m on the alley side, tilted up ≤ +45°, ``not_close_up``."""
    full = cfg or pdet.load_config()
    c, cam = full["leaf_scan"], full["cameras"]["crown"]
    p = np.asarray(leaf["world"], float)
    base = np.asarray(plan["pos"], float)
    radial = pdet.unit(np.array([p[0] - base[0], p[1] - base[1], 0.0]))
    if not np.any(radial):
        radial = np.array([0.0, -1.0, 0.0])
    cr = float((plan.get("entry") or {}).get("canopy_r_m") or 1.0)
    lens = base + radial * (cr + 0.9)
    lens[2] = float(c["crown_lens_m"])
    fwd = pdet.unit(p - lens)
    tilt = math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
    tmax = float(c["crown_tilt_max_deg"])
    if tilt > tmax:
        h = pdet.unit(np.array([fwd[0], fwd[1], 0.0]))
        fwd = pdet.unit(h * math.cos(math.radians(tmax)) + np.array([0, 0, math.sin(math.radians(tmax))]))
    v = pdet.View(pos=lens, fwd=fwd, fovy_deg=cam["fovy_deg"], width=cam["width"], height=cam["height"],
                  camera="crown_cam", band="mid", tag={"leaf": leaf["id"], "side": "crown"})
    dist = float(np.linalg.norm(p - lens))
    return LeafView(leaf["id"], "crown", v, dist, float("inf"), "", v.mm_per_px(dist), True, "not_close_up")




# ---------------------------------------------------------------------------------------------- preconditions
PRECONDITION_REASONS = ("dwell_slope", "wet_foliage", "work_fields_not_clear", "wrist_camera_fault",
                        "blade_not_locked_open", "low_light", "sensor_invalid", N_A)


def check_preconditions(ctx: Optional[dict] = None, plan: Optional[dict] = None, cfg: Optional[dict] = None
                        ) -> List[str]:
    """MP 3 / 3b scan preconditions and interlocks as reason codes (empty = go). ``ctx`` keys, all optional
    (absent = not checked here; the WP27 Ready check owns the common ones): ``cross_slope_deg`` (WP6 dwell gate,
    ≤ 3° for dwells to 60 s), ``wet_foliage``, ``work_fields_clear`` (incl. hand-in-envelope), ``wrist_camera_ok``,
    ``tool`` (mounted tool id) + ``blade_locked_open`` (CLIPPER T2 / PRUNER T5 blade locked open), ``light_lux`` +
    ``dawn_mode`` / ``hour``. A supplied numeric reading that is NaN, inf or not a number gives ``sensor_invalid``
    (fails closed; NaN would pass every comparison). With ``plan``: ``n/a_crown_only`` when the crop has no leaves in band (LEAF SCAN only;
    TRUNK SCAN passes no plan)."""
    full = cfg or pdet.load_config()
    c = full["leaf_scan"]
    ctx = ctx or {}
    out: List[str] = []
    # a supplied numeric reading that is NaN / inf / not a number fails closed (NaN compares False everywhere)
    bad = [k for k in ("cross_slope_deg", "light_lux", "hour") if ctx.get(k) is not None and not _finite(ctx[k])]
    if bad:
        out.append("sensor_invalid")
    if ctx.get("cross_slope_deg") is not None and "cross_slope_deg" not in bad:
        from atharv_nandi.core import stability
        if not stability.gate_dwell(abs(float(ctx["cross_slope_deg"])), float(c["dwell_s"])).allowed:
            out.append("dwell_slope")
    if ctx.get("wet_foliage"):
        out.append("wet_foliage")
    if ctx.get("work_fields_clear") is False:
        out.append("work_fields_not_clear")
    if ctx.get("wrist_camera_ok") is False:
        out.append("wrist_camera_fault")
    if ctx.get("tool") in tuple(c["blade_tools"]) and not ctx.get("blade_locked_open", False):
        out.append("blade_not_locked_open")
    if ctx.get("light_lux") is not None and "light_lux" not in bad:
        dawn = bool(ctx.get("dawn_mode")) or (ctx.get("hour") is not None and "hour" not in bad
                                              and float(ctx["hour"]) < float(c["dawn_before_h"]))
        thr = float(c["dawn_min_light_lux"] if dawn else c["min_light_lux"])
        if float(ctx["light_lux"]) < thr:
            out.append("low_light")
    if plan is not None:
        ok, why = leaf_applicability(plan)
        if not ok:
            out.append(why)
    return out


def backed_off(lv: "LeafView", obstacles: Sequence["Obstacle"], cfg: dict) -> "LeafView":
    """``depth < min-Z`` → back off ``backoff_m`` (3 cm) along the view axis. The new pose is re-checked like a
    :func:`plan_view` candidate: stand-off within ``standoff_range_m`` (0.12-0.20 m), footprint ≤
    ``max_footprint_mm_px`` (O-08) and a 12 mm decal ≥ ``min_decal_px`` px, else ``min_z`` (no stand-off left that
    is both deep enough and sharp enough); lens ≥ 0.05 m above the ground and the clearance rule over its own
    straight approach, else ``no_clearance``."""
    c, cam = cfg["leaf_scan"], cfg["cameras"]["wrist"]
    proxy = WristProxy(**c["wrist_proxy"])
    lens = lv.view.pos - lv.view.fwd * float(c["backoff_m"])
    s = lv.standoff_m + float(c["backoff_m"])
    v = pdet.View(pos=lens, fwd=lv.view.fwd, fovy_deg=cam["fovy_deg"], width=cam["width"], height=cam["height"],
                  camera=lv.view.camera, band=lv.view.band, tag=dict(lv.view.tag))
    mmpx = v.mm_per_px(s)
    eps = 1e-9
    if s > float(c["standoff_range_m"][1]) + eps or mmpx > float(c["max_footprint_mm_px"]) + eps \
            or 12.0 / mmpx < float(c["min_decal_px"]) - eps:
        return LeafView(lv.leaf_id, lv.side, v, s, lv.clearance_m, lv.clear_geom, mmpx, False, "min_z")
    if lens[2] < 0.05:
        return LeafView(lv.leaf_id, lv.side, v, s, float("-inf"), "ground", mmpx, False, "no_clearance")
    clr, who = approach_clearance(obstacles, proxy, lens, lv.view.fwd, float(c["approach_m"]), int(c["approach_samples"]))
    nv = LeafView(lv.leaf_id, lv.side, v, s, clr, who, mmpx)
    if not (clr >= float(c["min_clearance_m"])):
        nv.ok, nv.reason = False, "no_clearance"
    return nv


def capture_side(lv: "LeafView", obstacles: Sequence["Obstacle"], guard: "ClearanceGuard", cfg: dict,
                 capture_feed=None) -> Tuple["LeafView", dict]:
    """MP 3 failure / recovery for one side: ``capture_feed(leaf_view, attempt) -> {blurred, depth_m, drift_m}``
    (a physics or console caller; absent = a clean capture). Base drift > 2 cm → pause, re-approach (≤
    ``reapproach_max``, then ``base_drift``); depth < min-Z → back off 3 cm (≤ ``backoff_max``, then ``min_z``);
    blur → retry (≤ ``blur_retry``, then ``blurred``). Returns the final view and ``{retries, backoffs,
    reapproaches, extra_s}``; extra time goes into the time model."""
    c = cfg["leaf_scan"]
    st = {"retries": 0, "backoffs": 0, "reapproaches": 0, "extra_s": 0.0}
    if capture_feed is None or not lv.ok:
        return lv, st
    frames_s = int(c["frames_per_side"]) * float(c["frame_s"])
    approach_s = float(c["approach_m"]) / float(c["approach_speed_m_s"]) + float(c["settle_s"])
    for attempt in range(1 + int(c["blur_retry"]) + int(c["backoff_max"]) + int(c["reapproach_max"])):
        f = capture_feed(lv, attempt) or {}
        bad = [k for k in ("drift_m", "depth_m") if f.get(k) is not None and not _finite(f[k])]
        if bad:                                       # NaN / inf reading: never a clean capture, stop the side
            guard.log.append({"event": "sensor_invalid", "leaf": lv.leaf_id, "side": lv.side, "what": bad,
                              "action": "stop"})
            lv.ok, lv.reason = False, "sensor_invalid"
            break
        drift = guard.base_drift(float(f.get("drift_m") or 0.0))
        if drift != "ok":
            st["reapproaches"] += 1
            st["extra_s"] += approach_s
            if st["reapproaches"] > int(c["reapproach_max"]):
                lv.ok, lv.reason = False, "base_drift"
                break
            continue
        if f.get("depth_m") is not None and float(f["depth_m"]) < float(c["min_z_m"]):
            st["backoffs"] += 1
            if st["backoffs"] > int(c["backoff_max"]):
                lv.ok, lv.reason = False, "min_z"
                break
            lv = backed_off(lv, obstacles, cfg)
            guard.log.append({"event": "backoff", "leaf": lv.leaf_id, "side": lv.side,
                              "standoff_m": round(lv.standoff_m, 3), "clearance_m": round(lv.clearance_m, 4)})
            st["extra_s"] += float(c["backoff_m"]) / float(c["approach_speed_m_s"]) + float(c["settle_s"])
            if not lv.ok:
                break
            continue
        if f.get("blurred"):
            st["retries"] += 1
            st["extra_s"] += frames_s
            guard.log.append({"event": "blur_retry", "leaf": lv.leaf_id, "side": lv.side, "n": st["retries"]})
            if st["retries"] > int(c["blur_retry"]):
                lv.ok, lv.reason = False, "blurred"
                break
            continue
        break
    else:
        lv.ok, lv.reason = False, lv.reason or "capture_failed"
    st["extra_s"] = round(st["extra_s"], 2)
    return lv, st


# ---------------------------------------------------------------------------------------------- run
@dataclass
class LeafScanResult:
    tree_id: str
    status: str
    records: List[dict] = field(default_factory=list)
    leads: List[dict] = field(default_factory=list)
    views: List[LeafView] = field(default_factory=list)
    log: List[dict] = field(default_factory=list)
    leaf_block: dict = field(default_factory=dict)
    marks: List[dict] = field(default_factory=list)
    revisit: List[dict] = field(default_factory=list)     # reachable leaves with no imaged side (not a MARK)


def leaf_time_s(prev_lens, views: Sequence[LeafView], cfg: dict) -> float:
    """Time model for one leaf (O-08 median ≤ 10 s/leaf): move from the previous lens pose, straight approach,
    settle and frames per side, a side switch between upper and under."""
    c = cfg["leaf_scan"]
    v_arm, v_app = float(c["transit_speed_m_s"]), float(c["approach_speed_m_s"])
    t, here = 0.0, None if prev_lens is None else np.asarray(prev_lens, float)
    for i, lv in enumerate(views):
        if not lv.ok:
            continue
        start = lv.view.pos - lv.view.fwd * float(c["approach_m"])
        if here is not None:
            t += float(np.linalg.norm(start - here)) / v_arm
        t += float(c["approach_m"]) / v_app + float(c["settle_s"]) + int(c["frames_per_side"]) * float(c["frame_s"])
        if i:
            t += float(c["side_switch_s"])
        here = lv.view.pos
    return t


def order_leaves(leaves: Sequence[dict]) -> List[dict]:
    """Greedy nearest-neighbour visiting order, starting at the lowest leaf (shorter arm transits)."""
    left = sorted(leaves, key=lambda lf: (float(lf["world"][2]), lf["k"]))
    out: List[dict] = []
    while left:
        if not out:
            out.append(left.pop(0))
            continue
        here = np.asarray(out[-1]["world"], float)
        i = min(range(len(left)), key=lambda j: (float(np.linalg.norm(np.asarray(left[j]["world"], float) - here)),
                                                  left[j]["k"]))
        out.append(left.pop(i))
    return out


def run_leaf_scan(plan: dict, profile: dict, detector: Optional[pdet.SimTruthLeadDetector] = None,
                  n_leaves: Optional[int] = None, demo: bool = False, branch: Optional[dict] = None,
                  guard_feed=None, cfg: Optional[dict] = None, ctx: Optional[dict] = None, capture_feed=None,
                  dawn_mode: bool = False) -> LeafScanResult:
    """LEAF SCAN of one tree (MP 3 steps 2-7) with the SimTruth lead detector (assisted, labelled).

    ``ctx``: preconditions / interlocks (:func:`check_preconditions`); any reason → ``status: refused`` with
    ``reasons`` and no motion. ``capture_feed``: per-side capture outcomes for the MP 3 failure / recovery rows
    (:func:`capture_side`). ``dawn_mode`` (before 08:00) lowers the light threshold; the LED is on.

    ``guard_feed(leaf_view) -> [clearance samples]`` lets a physics caller feed measured clearances (MuJoCo
    ``mj_geomDistance``) into the guard; by default the planner's own approach clearances are fed."""
    full = cfg or pdet.load_config()
    c = full["leaf_scan"]
    sel = select_leaves(plan, n_leaves, demo, branch, full)
    res = LeafScanResult(plan["tree_id"], sel["status"])
    if ctx is not None:
        why = check_preconditions(dict(ctx, dawn_mode=ctx.get("dawn_mode", dawn_mode)), None, full)
        if why and sel["status"] == "ok":
            res.status = "refused"
            res.leaf_block = {"status": "refused", "reasons": why}
            return res
    if sel["status"] != "ok":
        res.leaf_block = {"status": sel["status"]}
        return res
    det = detector or pdet.SimTruthLeadDetector(full)
    classes = [k for k in pdet.active_classes(profile, plan["season"], cfg=full) if not k["distance_only"]]
    decals = pdet.truth_decals(plan, profile, full)
    obs = tree_obstacles(plan)
    guard = ClearanceGuard(float(c["min_clearance_m"]), float(c["drift_max_m"]))
    times, prev = [], None
    for lf in order_leaves(sel["close_up"]):
        views = [plan_view(lf, side, obs, full) for side in ("upper", "under")]
        extra_s = 0.0
        rec = {"leaf_id": lf["id"], "k": lf["k"], "kind": lf.get("kind"), "h_m": round(float(lf["world"][2]), 3),
               "h_class": lf.get("h_class"), "close_up": True, "label": LABEL, "sides": [], "findings": []}
        for i in range(len(views)):
            lv = views[i]
            if lv.ok:
                samples = guard_feed(lv) if guard_feed else [lv.clearance_m]
                guard.reset()
                state = "ok"
                for s in samples:
                    state = guard.update(float(s), who=lv.clear_geom)
                if state == "stop":
                    lv.ok, lv.reason = False, "no_clearance"
            cap = {"retries": 0, "backoffs": 0, "reapproaches": 0, "extra_s": 0.0}
            if lv.ok and capture_feed is not None:
                lv, cap = capture_side(lv, obs, guard, full, capture_feed)
            extra_s += cap["extra_s"]
            row = lv.row()
            if capture_feed is not None:
                row.update({k: cap[k] for k in ("retries", "backoffs", "reapproaches")})
            if lv.ok:
                row["frames"] = int(c["frames_per_side"])
                found = det.detect(lv.view, decals, classes, plan)
                for ld in found:
                    ld["label"] = LABEL
                    ld["leaf"] = lf["id"]
                    ld["side"] = lv.side
                res.leads.extend(found)
                row["leads_n"] = len(found)
            rec["sides"].append(row)
            res.views.append(lv)
            views[i] = lv
        if not any(lv.ok for lv in views):
            side_reasons = {lv.side: lv.reason for lv in views}
            if all(r == "no_clearance" for r in side_reasons.values()):
                # MP 3 "unreachable -> MARK out_of_reach_marked": only when no side has a clear view
                res.marks.append({"schema": "nandi.mark/1", "tree_id": plan["tree_id"],
                                  "physical_tag": (plan.get("entry") or {}).get("physical_tag"), "kind": "leaf",
                                  "leaf_id": lf["id"], "h_class": lf.get("h_class"), "h_m": rec["h_m"],
                                  "reason": "out_of_reach_marked", "side_reasons": side_reasons})
            else:
                # reachable but not imaged (blurred, base_drift, min_z, sensor_invalid, capture_failed): a robot
                # revisit, not a human job sheet row
                res.revisit.append({"leaf_id": lf["id"], "h_class": lf.get("h_class"), "h_m": rec["h_m"],
                                    "side_reasons": side_reasons})
        t = leaf_time_s(prev, views, full) + extra_s
        if any(lv.ok for lv in views):
            times.append(t)
            prev = [lv for lv in views if lv.ok][-1].view.pos
        rec["time_s"] = round(t, 2)
        res.records.append(rec)
    for lf in sel["not_close_up"]:
        lv = crown_view(plan, lf, full)
        found = det.detect(lv.view, decals, classes, plan)
        for ld in found:
            ld["label"] = "not_close_up"
            ld["leaf"] = lf["id"]
            ld["side"] = "crown"
        res.leads.extend(found)
        res.views.append(lv)
        res.records.append({"leaf_id": lf["id"], "k": lf["k"], "kind": lf.get("kind"),
                            "h_m": round(float(lf["world"][2]), 3), "h_class": lf.get("h_class"), "close_up": False,
                            "label": "not_close_up", "sides": [dict(lv.row(), frames=1, leads_n=len(found))]})
    res.log = guard.log
    sym = pdet.dedupe_leads(res.leads)
    for r in res.records:
        r["findings"] = [s["class"] for s in sym if s.get("leaf") == r["leaf_id"]]
    planned = 2 * len(sel["close_up"]) + len(sel["not_close_up"])
    ok_views = [lv for lv in res.views if lv.ok]
    close_ok = [lv for lv in ok_views if lv.side in ("upper", "under")]
    imaged_leaves = sum(1 for r in res.records if r["close_up"] and any(s["ok"] for s in r["sides"]))
    under_n = sum(1 for lv in ok_views if lv.side == "under")
    res.leaf_block = {
        "status": "ok", "n": imaged_leaves, "underside_n": under_n, "label": LABEL,
        "findings": [{k: s[k] for k in ("class", "band", "clock", "h_class", "h_m", "unit", "count", "score",
                                         "stage", "detector")} | {"leaf": s.get("leaf"), "side": s.get("side")}
                     for s in sym],
        "planned_views": planned, "imaged_views": len(ok_views),
        "imaged_frac": round(len(ok_views) / planned, 3) if planned else None,
        "underside_frac": round(under_n / max(1, len(sel["close_up"])), 3),
        "not_close_up_n": len(sel["not_close_up"]),
        "footprint_mm_px_max": round(max((lv.mm_per_px for lv in close_ok), default=0.0), 3),
        "min_clearance_m": round(min((lv.clearance_m for lv in close_ok), default=float("inf")), 4)
        if close_ok else None,
        "median_s_per_leaf": round(statistics.median(times), 2) if times else None,
        "led": str(c["led"]), "dawn_mode": bool(dawn_mode or (ctx or {}).get("dawn_mode")),
        "not_imaged": {r: sum(1 for lv in res.views if not lv.ok and lv.reason == r)
                       for r in sorted({lv.reason for lv in res.views if not lv.ok})},
        "revisit_leaves": [r["leaf_id"] for r in res.revisit],
    }
    return res


def leaf_truth_ids(plan: dict, profile: dict, cfg: Optional[dict] = None, leaf_ids: Optional[Sequence[str]] = None
                   ) -> Dict[str, str]:
    """Active leaf-decal truth ``{decal id: class}`` on the given leaves (default: every in-band leaf)."""
    full = cfg or pdet.load_config()
    act = {k["class"] for k in pdet.active_classes(profile, plan["season"], cfg=full) if not k["distance_only"]}
    ids = set(leaf_ids) if leaf_ids is not None else {lf["id"] for lf in plan["truth"]["leaves"] if lf["in_band"]}
    return {d["id"]: d["class"] for d in plan["truth"]["decals"] if d.get("leaf") in ids and d["class"] in act}


__all__ = ["ClearanceGuard", "LeafView", "LeafScanResult", "Obstacle", "WristProxy", "approach_clearance",
           "backed_off", "capture_side", "check_preconditions", "PRECONDITION_REASONS",
           "clearance", "cone_axes", "crown_view", "leaf_applicability", "leaf_time_s", "leaf_truth_ids",
           "plan_view", "run_leaf_scan", "select_leaves", "seg_seg_dist", "tree_obstacles", "N_A", "LABEL", "gen"]
