"""TREE 360 band planner and measured coverage (WP25; ORCHARD-MVP-SPEC 1.3 O-01, 2.3 "Bands", 2.5 "Ray groups").

Pure Python + numpy. MuJoCo appears only behind ``mujoco_ray_fn`` (the coverage ray caster) and
``canopy_from_model`` (reads a tree's envelope from a compiled model); every rule is unit-tested without them.

Band planner (OS 2.3). A TREE 360 band is one camera at one lens height looking at the canopy surface from the
horizontal stand-off ``d`` (lens to canopy surface). On a vertical "planar screen" at distance ``d`` a camera at lens
height ``z`` with tilt ``t`` (positive up) and vertical field of view ``fovy`` covers

    z_lo = z + d * tan(t - fovy / 2)        z_hi = z + d * tan(t + fovy / 2)        (z_lo clipped at 0)

    low   tree-side ring camera, lens 0.98 m, fixed tilt -25 deg, fovy 83.6 deg (includes the trunk/base band)
    mid   crown camera, lens 1.70 m (ext 0.60), default tilt -10 deg, fovy 60 deg
    high  crown camera, lens 2.70 m (ext 1.60), default tilt -20 deg, fovy 60 deg

``plan_bands(d)`` keeps the default tilts where they already give >= ``overlap_m`` (0.30 m) between neighbouring
bands and reach ``top_m`` (2.8 m); otherwise it solves the crown tilt per band: the mid tilt is lowered until its
bottom sits ``overlap_m + OVERLAP_MARGIN_M`` under the ring top; the high tilt is kept inside the window where the
band both reaches the top and overlaps the mid band by the same amount (clamped to the gimbal's -60..+45 deg). The
result lists every band's interval, the overlaps and whether 0..top is covered. At the spec stand-offs (computed,
``tests/test_coverage.py``): d 0.9 -> low 0-1.25, mid 0.94-2.03, high 1.63-2.86 (overlaps 0.31 / 0.40); d 1.2 and
1.5 keep the default tilts too.

Coverage (KPI O-01). The canopy surface of one tree is sampled at ``N_AZ`` (24) azimuths x ``LEVELS`` (3) heights per
band = **72 points per band**, at 1/6, 3/6, 5/6 of the band's planned height interval (clipped to the tree). A point
sits on the foliage envelope (the WP23 group-5 ellipsoid: radius from its height), or on the trunk below the canopy.
A frame of that band **sees** a point when

    1. the point projects into the image (pinhole, optical frame x right / y down / z forward) in front of the lens,
    2. the footprint at the point (mm/px = 1000 * range * 2 tan(fovy / 2) / height_px) is at most the band's target
       (``FOOTPRINT_MAX_MM``: ring 20, crown 12 mm/px at 320 x 240, ASSUMED targets for canopy-level evidence),
    3. the coverage ray from the lens to the point is not blocked: ``ray_fn(origin, unit_dir, range)`` returns the
       first hit distance against wood + fruit + foliage (geom groups {0, 1, 3}, OS 2.5 coverage mask; the robot's
       group 2 and the query-only group 5 are not in it) and a hit nearer than ``range - OCCLUSION_TOL_M`` occludes,
    4. the pixel is not masked by the privacy filter (``FilteredFrame.masked``: person, detector, parcel masks).

A point that passes 1-3 but only ever fails 4 (or whose frame was dropped whole) is **masked**, never seen:
``masked_pct`` = masked points / points (O-01 "masked share reported separately"). Coverage is never extrapolated:
a point counts only from a frame that saw it. ``az_deg`` per band is the union of the 15 deg azimuth sectors with at
least one seen point; ``sides`` gives the same per alley side (a point belongs to the side its azimuth faces,
``side_of_az``), ``by_arc`` per flown arc. Points sit on the analytic envelope (``surface_samples``) or, with a
model, on the tree's first rendered surface along a horizontal ray (``surface_samples_model``). The WP5 parcel mask
hides a point only where the pixel's depth shows that point; without a ray caster a back-face test on the radial
normal hides the far side.
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

BANDS = ("low", "mid", "high")
# camera geometry (OS 2.3 / 2.5): ring 4:3 fovy 83.6 (HFOV 100), crown fovy 60 (HFOV 75.2)
RING_LENS_M, RING_TILT_DEG, RING_FOVY_DEG = 0.98, -25.0, 83.6
CROWN_FOVY_DEG = 60.0
MID_LENS_M, HIGH_LENS_M = 1.70, 2.70
MID_EXT_M, HIGH_EXT_M = 0.60, 1.60
DEFAULT_TILT_DEG = {"mid": -10.0, "high": -20.0}
GIMBAL_TILT_MIN_DEG, GIMBAL_TILT_MAX_DEG = -60.0, 45.0
OVERLAP_M = 0.30                 # OS 2.3: >= 0.3 m between neighbouring bands
OVERLAP_MARGIN_M = 0.02          # the solver aims 2 cm above the bar (pose noise of a settled station)
TOP_M = 2.8                      # OS 6 band test: 0-2.8 m covered
STANDOFFS_M = (0.9, 1.2, 1.5)    # OS 2.3 table / MISSION-PRESETS 0.8 standoff_m range
N_AZ = 24                        # 15 deg azimuth sectors
LEVELS = 3                       # heights per band -> 72 points per band
FOOTPRINT_MAX_MM = {"low": 20.0, "mid": 12.0, "high": 12.0}   # ASSUMED canopy-level targets (sim 320 x 240)
OCCLUSION_TOL_M = 0.05           # a hit within 5 cm of the point is the point's own surface
PARCEL_DEPTH_TOL_M = 0.30       # a parcel-masked pixel hides a sample point only if it shows that point
COVERAGE_GROUPS = (0, 1, 3)      # OS 2.5 coverage ray mask: wood/ground, fruit, foliage
FULL_MIN_PCT = 90.0              # O-01 full orbit: >= 90 % per band
TWO_SIDED_MIN_DEG = 120.0        # O-01 two-sided: >= 2 x 120 deg


# ====================================================================== band planner
def band_interval(lens_z: float, tilt_deg: float, fovy_deg: float, d: float) -> Tuple[float, float]:
    """(z_lo, z_hi) covered on a vertical plane at horizontal distance ``d`` (z_lo clipped at 0)."""
    h = math.radians(fovy_deg) / 2.0
    t = math.radians(tilt_deg)
    lo = lens_z + d * math.tan(t - h) if t - h > -math.pi / 2 + 1e-9 else -math.inf
    hi = lens_z + d * math.tan(t + h) if t + h < math.pi / 2 - 1e-9 else math.inf
    return max(0.0, lo), hi


def _tilt_for_bottom(lens_z: float, z_bottom: float, fovy_deg: float, d: float) -> float:
    """Tilt (deg) that puts the bottom edge of the band at ``z_bottom``."""
    return math.degrees(math.atan2(z_bottom - lens_z, d)) + fovy_deg / 2.0


def _tilt_for_top(lens_z: float, z_top: float, fovy_deg: float, d: float) -> float:
    return math.degrees(math.atan2(z_top - lens_z, d)) - fovy_deg / 2.0


def _clamp_tilt(t: float) -> float:
    return min(GIMBAL_TILT_MAX_DEG, max(GIMBAL_TILT_MIN_DEG, t))


@dataclass
class Band:
    name: str
    camera: str          # "ring" (tree-side ring camera) or "crown"
    lens_z: float
    ext_m: float         # crown extension for this band (0 for the ring band)
    tilt_deg: float
    fovy_deg: float
    z_lo: float
    z_hi: float
    solved: bool = False  # True when the tilt was changed from the default

    def as_dict(self) -> dict:
        return {"band": self.name, "camera": self.camera, "lens_z": round(self.lens_z, 3), "ext_m": self.ext_m,
                "tilt_deg": round(self.tilt_deg, 2), "z_lo": round(self.z_lo, 3), "z_hi": round(self.z_hi, 3),
                "solved": self.solved}


@dataclass
class BandPlan:
    d: float
    top_m: float
    overlap_m: float
    bands: Dict[str, Band]

    @property
    def overlaps(self) -> Dict[str, float]:
        b = self.bands
        return {"low_mid": b["low"].z_hi - b["mid"].z_lo, "mid_high": b["mid"].z_hi - b["high"].z_lo}

    @property
    def covered(self) -> bool:
        """0..top covered with >= overlap_m between neighbouring bands (OS 6 band coverage test)."""
        b, ov = self.bands, self.overlaps
        return (b["low"].z_lo <= 1e-9 and b["high"].z_hi >= self.top_m - 1e-9
                and ov["low_mid"] >= self.overlap_m - 1e-9 and ov["mid_high"] >= self.overlap_m - 1e-9)

    def tilt(self, band: str) -> float:
        return self.bands[band].tilt_deg

    def as_dict(self) -> dict:
        return {"d_m": round(self.d, 3), "top_m": self.top_m, "covered": self.covered,
                "overlaps": {k: round(v, 3) for k, v in self.overlaps.items()},
                "bands": {k: v.as_dict() for k, v in self.bands.items()}}


def plan_bands(d: float, top_m: float = TOP_M, overlap_m: float = OVERLAP_M,
               mid_lens_m: float = MID_LENS_M, high_lens_m: float = HIGH_LENS_M,
               tilts: Optional[Dict[str, float]] = None) -> BandPlan:
    """The per-tree band plan from the stand-off ``d`` (lens to canopy surface, m). See the module docstring."""
    if not (d > 0 and math.isfinite(d)):
        raise ValueError(f"stand-off d must be > 0, got {d!r}")
    tilts = dict(DEFAULT_TILT_DEG, **(tilts or {}))
    want = overlap_m + OVERLAP_MARGIN_M
    lo_lo, lo_hi = band_interval(RING_LENS_M, RING_TILT_DEG, RING_FOVY_DEG, d)
    low = Band("low", "ring", RING_LENS_M, 0.0, RING_TILT_DEG, RING_FOVY_DEG, lo_lo, lo_hi)
    # mid: keep the default tilt unless the overlap with the ring is short; then lower it
    t_mid = tilts["mid"]
    m_lo, m_hi = band_interval(mid_lens_m, t_mid, CROWN_FOVY_DEG, d)
    solved_mid = False
    if lo_hi - m_lo < overlap_m:
        t_mid = _clamp_tilt(_tilt_for_bottom(mid_lens_m, lo_hi - want, CROWN_FOVY_DEG, d))
        m_lo, m_hi = band_interval(mid_lens_m, t_mid, CROWN_FOVY_DEG, d)
        solved_mid = True
    mid = Band("mid", "crown", mid_lens_m, MID_EXT_M, t_mid, CROWN_FOVY_DEG, m_lo, m_hi, solved_mid)
    # high: the window [t_top, t_ov] reaches the top and overlaps mid; keep the default inside it
    t_high = tilts["high"]
    t_top = _tilt_for_top(high_lens_m, top_m, CROWN_FOVY_DEG, d)
    t_ov = _tilt_for_bottom(high_lens_m, m_hi - want, CROWN_FOVY_DEG, d)
    h_lo, h_hi = band_interval(high_lens_m, t_high, CROWN_FOVY_DEG, d)
    solved_high = False
    if m_hi - h_lo < overlap_m or h_hi < top_m:
        if t_top <= t_ov:
            t_high = min(max(t_high, t_top), t_ov)
        else:                      # cannot have both: keep the overlap (no gap), report the top shortfall
            t_high = t_ov
        t_high = _clamp_tilt(t_high)
        h_lo, h_hi = band_interval(high_lens_m, t_high, CROWN_FOVY_DEG, d)
        solved_high = True
    high = Band("high", "crown", high_lens_m, HIGH_EXT_M, t_high, CROWN_FOVY_DEG, h_lo, h_hi, solved_high)
    return BandPlan(float(d), float(top_m), float(overlap_m), {"low": low, "mid": mid, "high": high})


# ====================================================================== canopy model and sample points
@dataclass
class Canopy:
    """One tree's canopy surface: trunk axis at ``xy`` on ground ``ground_z``; foliage ellipsoids
    ``(centre_xyz_world, (a, b, c))`` (axis-aligned, WP23 envelopes); trunk radius below the canopy."""
    tree_id: str
    xy: Tuple[float, float]
    ground_z: float
    height_m: float
    trunk_r: float
    ellipsoids: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]

    def radius(self, z: float, az: float) -> float:
        """Horizontal distance of the canopy surface from the trunk axis at world height ``z``, azimuth ``az``."""
        best = 0.0
        ca, sa = math.cos(az), math.sin(az)
        for (cx, cy, cz), (a, b, c) in self.ellipsoids:
            u = (z - cz) / c
            if abs(u) >= 1.0:
                continue
            k = math.sqrt(1.0 - u * u)
            ox, oy = cx - self.xy[0], cy - self.xy[1]
            # ray from the axis along az against the ellipse (x - ox)^2/(ak)^2 + (y - oy)^2/(bk)^2 = 1
            A = (ca / (a * k)) ** 2 + (sa / (b * k)) ** 2
            B = -2.0 * (ox * ca / (a * k) ** 2 + oy * sa / (b * k) ** 2)
            C = (ox / (a * k)) ** 2 + (oy / (b * k)) ** 2 - 1.0
            disc = B * B - 4 * A * C
            if disc < 0:
                continue
            s = (-B + math.sqrt(disc)) / (2 * A)
            best = max(best, s)
        return max(best, self.trunk_r)


def canopy_from_entry(entry: dict, ground_z: float = 0.0) -> Canopy:
    """Analytic canopy from a registry entry (WP23 ``canopy`` archetype: trunk_h * 0.75 .. H ellipsoid)."""
    H = float(entry["height_m"])
    cr = float(entry["canopy_r_m"])
    trunk_h = float(entry.get("trunk_h_m") or 0.35 * H)
    z0, z1 = trunk_h * 0.75, H
    x, y = float(entry["geo"]["x"]), float(entry["geo"]["y"])
    return Canopy(str(entry["tree_id"]), (x, y), float(ground_z), H, float(entry.get("trunk_r_m") or 0.06),
                  [((x, y, ground_z + (z0 + z1) / 2.0), (cr, cr, (z1 - z0) / 2.0))])


def canopy_from_model(m, d, tree_id: str) -> Optional[Canopy]:
    """The canopy of ``tree_id`` from the compiled model: its body's group-5 ellipsoid envelopes and trunk capsule."""
    import mujoco
    from atharv_nandi.sim.world import trees as trees_mod
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, trees_mod.body_name(tree_id))
    if bid < 0:
        return None
    ell, trunk_r, top = [], 0.0, 0.0
    base = d.xpos[bid]
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != bid:
            continue
        gt, grp = int(m.geom_type[g]), int(m.geom_group[g])
        pos, size = d.geom_xpos[g], m.geom_size[g]
        if grp == trees_mod.GRP_QUERY and gt == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            ell.append((tuple(float(v) for v in pos), tuple(float(v) for v in size[:3])))
            top = max(top, float(pos[2] + size[2] - base[2]))
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if grp == trees_mod.GRP_WOOD and "trunk" in name and gt == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            trunk_r = max(trunk_r, float(size[0]))
    if not ell:
        return None
    return Canopy(tree_id, (float(base[0]), float(base[1])), float(base[2]), top, trunk_r or 0.06, ell)


@dataclass
class Samples:
    """Sample points of one band: ``points`` (N, 3) world, ``az`` (N,) rad, ``z`` (N,) height above ground."""
    band: str
    points: np.ndarray
    az: np.ndarray
    z: np.ndarray


def band_levels(z_lo: float, z_hi: float, height_m: float, levels: int = LEVELS) -> List[float]:
    """Sample heights of a band: the centres of ``levels`` equal slices of its interval clipped to the tree."""
    lo, hi = max(0.05, z_lo), min(height_m - 0.05, z_hi)
    if hi <= lo:
        hi = lo = min(max(0.05, (z_lo + z_hi) / 2.0), height_m - 0.05)
    return [lo + (hi - lo) * (2 * k + 1) / (2.0 * levels) for k in range(levels)]


def surface_samples(canopy: Canopy, plan: BandPlan, n_az: int = N_AZ, levels: int = LEVELS) -> Dict[str, Samples]:
    """72 (24 x 3) canopy-surface points per band (module docstring)."""
    out = {}
    azs = [2.0 * math.pi * k / n_az for k in range(n_az)]
    for name in BANDS:
        b = plan.bands[name]
        pts, az_l, z_l = [], [], []
        for z in band_levels(b.z_lo, b.z_hi, canopy.height_m, levels):
            for a in azs:
                r = canopy.radius(canopy.ground_z + z, a)
                pts.append((canopy.xy[0] + r * math.cos(a), canopy.xy[1] + r * math.sin(a), canopy.ground_z + z))
                az_l.append(a)
                z_l.append(z)
        out[name] = Samples(name, np.asarray(pts, float), np.asarray(az_l, float), np.asarray(z_l, float))
    return out


def surface_samples_model(canopy: Canopy, plan: BandPlan, ray_fn: Callable, body_of: Callable[[int], int],
                          body_id: int, n_az: int = N_AZ, levels: int = LEVELS,
                          max_hops: int = 4) -> Dict[str, Samples]:
    """The same 72 points per band, placed on the tree's **rendered** surface: a horizontal ray from outside
    (envelope radius + 0.3 m) toward the trunk axis at each sample height and azimuth; the point is its first hit
    on a geom of the tree's own body (coverage groups {0, 1, 3}: wood, fruit, foliage). A hit on another body (a
    neighbour's canopy in a hedgerow) is stepped past (``max_hops``). A ray that only finds the axis (a gap through
    the canopy) keeps the analytic envelope point. ``ray_fn(origin, dir, max_dist) -> (dist, geom_id) | None``;
    ``body_of(geom_id) -> body id``."""
    analytic = surface_samples(canopy, plan, n_az, levels)
    out = {}
    for name, s in analytic.items():
        pts = s.points.copy()
        for i in range(len(pts)):
            a, z = float(s.az[i]), float(canopy.ground_z + s.z[i])
            u = np.array([math.cos(a), math.sin(a), 0.0])
            r0 = canopy.radius(z, a) + 0.3
            origin = np.array([canopy.xy[0], canopy.xy[1], z]) + r0 * u
            d_left = r0
            for _ in range(max_hops):
                hit = ray_fn(origin, -u, d_left)
                if hit is None:
                    break
                dist, gid = hit
                if body_of(int(gid)) == int(body_id):
                    pts[i] = origin - u * dist
                    break
                step = dist + 0.01
                origin = origin - u * step
                d_left -= step
                if d_left <= 0:
                    break
        out[name] = Samples(name, pts, s.az.copy(), s.z.copy())
    return out


def mujoco_surface_ray_fn(m, d, groups: Sequence[int] = COVERAGE_GROUPS) -> Callable:
    """``ray_fn`` for ``surface_samples_model``: (distance, geom id) of the first hit on ``groups``, or None."""
    import mujoco
    grp = np.zeros(6, dtype=np.uint8)
    for g in groups:
        grp[int(g)] = 1
    gid = np.zeros(1, dtype=np.int32)

    def ray(origin, direction, max_dist):
        dist = mujoco.mj_ray(m, d, np.asarray(origin, float), np.asarray(direction, float), grp, 1, -1, gid)
        return None if dist < 0 or dist > max_dist else (float(dist), int(gid[0]))
    return ray


# ====================================================================== alley sides
_SIDE_DIR = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}


def side_of_az(az: float, sides: Sequence[str]) -> str:
    """The alley side (``+x``/``+y``/``-x``/``-y`` from the registry ``alley_sides``) an azimuth (rad, 0 = +x)
    faces: the nearest listed side direction. ``all`` or no usable side -> the +-y half plane."""
    usable = [s for s in sides if s in _SIDE_DIR] or ["-y", "+y"]
    deg = math.degrees(az) % 360.0
    return min(usable, key=lambda s: abs((deg - _SIDE_DIR[s] + 180.0) % 360.0 - 180.0))


# ====================================================================== visibility of one frame
def footprint_mm(range_m: np.ndarray, fovy_deg: float, height_px: int) -> np.ndarray:
    return 1e3 * np.asarray(range_m) * 2.0 * math.tan(math.radians(fovy_deg) / 2.0) / float(height_px)


def mujoco_ray_fn(m, d, groups: Sequence[int] = COVERAGE_GROUPS, exclude_body: int = -1) -> Callable:
    """``ray_fn(origin, unit_dir, max_dist) -> first hit distance or None`` on geom groups ``groups`` (OS 2.5)."""
    import mujoco
    grp = np.zeros(6, dtype=np.uint8)
    for g in groups:
        grp[int(g)] = 1
    gid = np.zeros(1, dtype=np.int32)

    def ray(origin, direction, max_dist):
        dist = mujoco.mj_ray(m, d, np.asarray(origin, float), np.asarray(direction, float), grp, 1,
                             int(exclude_body), gid)
        return None if dist < 0 or dist > max_dist else float(dist)
    return ray


def frame_visibility(points: np.ndarray, pose, intr, fovy_deg: float, footprint_max_mm: float,
                     ray_fn: Optional[Callable], mask: Optional[np.ndarray] = None,
                     tol_m: float = OCCLUSION_TOL_M, parcel: Optional[np.ndarray] = None,
                     depth: Optional[np.ndarray] = None,
                     normals: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """(seen (N,), masked (N,), counts) of one frame. ``pose`` maps world -> optical (``detector_model.Pose``);
    ``intr`` has fx, fy, cx, cy, width, height; ``mask`` is the frame's privacy mask (H, W) or None.
    ``masked`` marks points that pass the geometry (rules 1-3) but sit on a masked pixel. ``parcel`` (the WP5
    parcel-boundary mask) counts only where the pixel shows the point (raw ``depth`` within ``PARCEL_DEPTH_TOL_M``
    of its optical z): a parcel pixel is content beyond the parcel, so at a thin-foliage edge where the pixel shows the
    background the point is not lost to the parcel mask (it was checked visible by its coverage ray)."""
    P = pose.apply(np.asarray(points, float))
    z = P[:, 2]
    n = len(points)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.cx + intr.fx * P[:, 0] / z
        v = intr.cy + intr.fy * P[:, 1] / z
    in_img = (z > 1e-6) & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    rng = np.linalg.norm(P, axis=1)
    fp_ok = footprint_mm(rng, fovy_deg, intr.height) <= footprint_max_mm
    geo = in_img & fp_ok
    origin = pose.inverse_apply(np.zeros((1, 3)))[0]
    if ray_fn is None and normals is not None:
        # analytic mode (no ray caster): a convex surface hides its far side (back-face test on the radial normal)
        geo &= np.einsum("ij,ij->i", np.asarray(normals, float), origin[None, :] - np.asarray(points, float)) > 0
    unocc = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(geo):
        dvec = np.asarray(points[i], float) - origin
        dist = float(np.linalg.norm(dvec))
        if ray_fn is None:
            unocc[i] = True
            continue
        hit = ray_fn(origin, dvec / dist, dist)
        unocc[i] = hit is None or hit >= dist - tol_m
    vis = geo & unocc
    masked = np.zeros(n, dtype=bool)
    if mask is not None and vis.any():
        mk = np.asarray(mask, bool)
        ui = np.clip(u[vis].astype(int), 0, mk.shape[1] - 1)
        vi = np.clip(v[vis].astype(int), 0, mk.shape[0] - 1)
        masked[np.flatnonzero(vis)] = mk[vi, ui]
        if parcel is not None and depth is not None:
            pk, dz = np.asarray(parcel, bool), np.asarray(depth, float)
            idx = np.flatnonzero(vis)
            same = np.abs(dz[vi, ui] - z[idx]) <= PARCEL_DEPTH_TOL_M
            masked[idx] |= pk[vi, ui] & same
    seen = vis & ~masked
    counts = {"in_image": int(in_img.sum()), "footprint_ok": int(geo.sum()), "unoccluded": int(vis.sum()),
              "occluded": int((geo & ~unocc).sum()), "masked": int(masked.sum()), "seen": int(seen.sum())}
    return seen, masked, counts


# ====================================================================== accumulator
@dataclass
class _BandAcc:
    samples: Samples
    seen: np.ndarray
    masked: np.ndarray
    normals: Optional[np.ndarray] = None
    frames: int = 0
    dropped: int = 0
    side_seen: Dict[str, np.ndarray] = field(default_factory=dict)


class CoverageAccumulator:
    """O-01 per band over every frame of one TREE 360 visit (module docstring)."""

    def __init__(self, canopy: Canopy, plan: BandPlan, n_az: int = N_AZ, levels: int = LEVELS,
                 footprint_max_mm: Optional[Dict[str, float]] = None,
                 samples: Optional[Dict[str, Samples]] = None, alley_sides: Sequence[str] = ("-y", "+y")):
        self.canopy, self.plan, self.n_az = canopy, plan, int(n_az)
        self.footprint = dict(FOOTPRINT_MAX_MM, **(footprint_max_mm or {}))
        self.alley_sides = [s for s in alley_sides if s in _SIDE_DIR] or ["-y", "+y"]
        smp = samples if samples is not None else surface_samples(canopy, plan, n_az, levels)
        self.acc = {b: _BandAcc(s, np.zeros(len(s.points), bool), np.zeros(len(s.points), bool),
                                np.stack([np.cos(s.az), np.sin(s.az), np.zeros_like(s.az)], axis=1))
                    for b, s in smp.items()}
        self.log: List[dict] = []

    def add_frame(self, band: str, pose, intr, fovy_deg: float, ray_fn: Optional[Callable],
                  mask: Optional[np.ndarray] = None, dropped: bool = False, side: str = "",
                  seq: Optional[int] = None, parcel: Optional[np.ndarray] = None,
                  depth: Optional[np.ndarray] = None) -> dict:
        """Count one frame of ``band``. ``dropped``: the privacy filter dropped the whole frame (every point it would
        have seen counts as masked)."""
        a = self.acc[band]
        a.frames += 1
        full = np.ones((intr.height, intr.width), bool) if dropped else mask
        seen, masked, counts = frame_visibility(a.samples.points, pose, intr, fovy_deg, self.footprint[band],
                                                ray_fn, full, parcel=None if dropped else parcel,
                                                depth=None if dropped else depth, normals=a.normals)
        if dropped:
            a.dropped += 1
        a.seen |= seen
        a.masked |= masked
        if side:
            a.side_seen.setdefault(side, np.zeros(len(a.seen), bool))
            a.side_seen[side] |= seen
        rec = dict(counts, band=band, seq=seq, side=side, dropped=bool(dropped))
        self.log.append(rec)
        return rec

    def _az_deg(self, seen: np.ndarray, az: np.ndarray) -> float:
        if not seen.any():
            return 0.0
        sector = np.round(az[seen] / (2 * math.pi / self.n_az)).astype(int) % self.n_az
        return 360.0 * len(set(sector.tolist())) / self.n_az

    def _alley_az(self, seen: np.ndarray, az: np.ndarray) -> Dict[str, float]:
        """Seen azimuth (deg, 15 deg sectors) per alley side: a point belongs to the side its azimuth faces."""
        side = np.array([side_of_az(float(a), self.alley_sides) for a in az])
        return {s: self._az_deg(seen & (side == s), az) for s in self.alley_sides}

    def summary(self) -> dict:
        bands, total_seen, total_n, total_masked = {}, 0, 0, 0
        for b, a in self.acc.items():
            n = len(a.seen)
            masked_only = a.masked & ~a.seen
            bands[b] = {"n": n, "seen": int(a.seen.sum()), "pct": round(100.0 * a.seen.sum() / n, 2),
                        "masked_pct": round(100.0 * masked_only.sum() / n, 2), "frames": a.frames,
                        "dropped": a.dropped, "az_deg": self._az_deg(a.seen, a.samples.az),
                        "sides": self._alley_az(a.seen, a.samples.az),
                        "by_arc": {s: self._az_deg(v, a.samples.az) for s, v in sorted(a.side_seen.items())}}
            total_seen += int(a.seen.sum())
            total_n += n
            total_masked += int(masked_only.sum())
        return {"bands": bands, "seen": total_seen, "n": total_n,
                "masked_pct": round(100.0 * total_masked / max(1, total_n), 2),
                "az_deg_by_band": {b: v["az_deg"] for b, v in bands.items()},
                "pct_by_band": {b: v["pct"] for b, v in bands.items()}}


def coverage_pass(summary: dict, mode: str) -> Tuple[bool, str]:
    """O-01 gate: full >= 90 % per band; two-sided >= 120 deg of azimuth on each of two sides, every band."""
    bands = summary["bands"]
    if mode == "full":
        bad = [b for b, v in bands.items() if v["pct"] < FULL_MIN_PCT]
        return (not bad), ("" if not bad else "full_below_90:" + ",".join(bad))
    bad = []
    for b, v in bands.items():
        sides = sorted(v["sides"].values(), reverse=True)
        if len(sides) < 2 or sides[1] < TWO_SIDED_MIN_DEG:
            bad.append(b)
    return (not bad), ("" if not bad else "two_sided_below_2x120:" + ",".join(bad))
