"""Static stability of Atharv Nandi: support polygon, CoM margin, tip-over angles, mode limits.

Pure numpy (no MuJoCo, no ROS). stability_probe.py feeds it numbers taken from a MuJoCo model;
atharv_nandi/nodes/stability_monitor.py publishes the result on /nandi/stability. Everything here is unit-tested
with closed-form cases in tests/test_stability.py.

Definitions (alignment spec v1, section 6, 2026-11-17):
    margin       signed horizontal distance (m) from the gravity-projected CoM to the nearest edge
                 of the support polygon (convex hull of the feet in contact); + inside, - outside.
                 Fewer than 3 contacts (or collinear contacts) -> -inf and mode UNSUPPORTED, unless
                 the stance tripod is within STANCE_NEAR_M of the ground (tripod-gait transient, F3).
    tip angle    per polygon edge, the extra rotation about that edge (deg) before the CoM is
                 directly above it: atan2(d_perp, h_CoM_above_edge) on flat ground; the general
                 form here works for any gravity direction (tilted floor / slope). Reported as the
                 minimum over edges.
    head push    horizontal force (N) at the mast head that tips the robot about an edge:
                 F = m_total * g * d_perp / (z_head - z_edge); reported as the minimum over edges.
    slope        terrain gradient in the robot's heading frame: pitch_deg > 0 = nose up, roll_deg > 0
                 = left (+y) side up. From a plane fit of foot contact points or from the IMU
                 quaternion; both derive pitch/roll from a unit "up" normal the same way, so they agree.

Status is ADVISORY in this phase (spec G5): classify() returns inhibit flags and reasons; nothing in
the stack enforces them yet. Enforcement in safety_logic / mission_logic is a follow-up package.

Constant tags: MEASURED (audit ref), CITED, DERIVED (arithmetic shown), ASSUMED.
"""
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi import paths
from atharv_nandi.core.safety_logic import geofence_margin

G = 9.81  # m/s^2, the MJCF <option gravity="0 0 -9.81">

# --- modes ---------------------------------------------------------------------------------------
MODE_STAND = "STAND"
MODE_TRIPOD_WALK = "TRIPOD_WALK"
MODE_SCAN_WIDE = "SCAN_WIDE"
MODE_MAST_UP = "MAST_UP"
MODE_UNSUPPORTED = "UNSUPPORTED"
MODES = (MODE_STAND, MODE_TRIPOD_WALK, MODE_SCAN_WIDE, MODE_MAST_UP, MODE_UNSUPPORTED)

# --- section 6 limits ------------------------------------------------------------------------------
# Contact: a foot counts as "in contact" when its geom-to-world contact distance <= this (m). ASSUMED
# (spec v1 section 6): 2 mm, about 3x the 0.70 mm standing penetration MEASURED by the contact audit.
CONTACT_DIST_M = 0.002

# STAND (6 feet). ASSUMED design targets (spec v1 section 6); the home stance measures 0.20+ m.
STAND_MIN_MARGIN_M = 0.10
STAND_MIN_TIP_DEG = 15.0

# TRIPOD_WALK (mast stowed or <= 1.0 m). Quasi-static tripod margin ASSUMED 0.05 m; tip angle 9.6 deg
# MEASURED (mounting-stability audit 25efcff, 1.0 m mast case).
TRIPOD_MIN_MARGIN_M = 0.05
TRIPOD_MIN_TIP_DEG = 9.6
MAST_STOWED_MAX_H_M = 1.0          # above this the mast counts as "up" (spec v1 section 6)
# Dynamic tripod support (review F3, 2026-11-17). The tripod walk spends most of its time with only
# 2 feet inside CONTACT_DIST_M (MEASURED on the aligned model, flat, vx 1: 69.7 % of 300 samples at
# 0 kg, 46.0 % at 10 kg; scratch_f3.py). Those instants are not "unsupported": the stance tripod is a
# few mm above the ground and lands within a few control ticks. When fewer than 3 feet touch, the probe
# takes the tripod (A = FR, ML, RR or B = FL, MR, RL) whose highest foot is lowest; if that foot is
# within STANCE_NEAR_M of the ground (ray down along gravity) the tripod feet count as the support set
# (support "stance_tripod") and the mode stays TRIPOD_WALK. A lifted, falling or toppled robot has no
# tripod that close to the ground and stays UNSUPPORTED.
# ASSUMED from MEASURED clearances (docs/hld.md 3.4): stance p50 3.8-8.5 mm, swing p50 54-60 mm.
STANCE_NEAR_M = 0.015
SUPPORT_CONTACT = "contact"
SUPPORT_STANCE_TRIPOD = "stance_tripod"
SUPPORT_NONE = "none"

# Fraction of walking time with < 3 feet in contact: reported only. 21.6 % MEASURED (mounting audit
# 25efcff); the < 2 % target belongs to the gait-overlap follow-up.
UNDER3_TARGET_FRAC = 0.02

# SCAN_WIDE (hip splay 0.45 rad, crouch -0.06 m, 6 feet): required for TREE 360 / LEAF SCAN / PICK.
SCAN_WIDE_HIP_RAD = 0.45           # ASSUMED stance definition (spec v1 section 6)
SCAN_WIDE_DETECT_HIP_RAD = 0.40    # DERIVED: |hip| >= 0.45 - 0.05 tolerance counts as the wide stance
SCAN_WIDE_CROUCH_M = -0.06         # ASSUMED stance definition (spec v1 section 6)
SCAN_WIDE_MIN_MARGIN_M = 0.10

# MAST_UP (mast > 1.0 m): stance only, all 6 feet down.
MAST_UP_MAX_SLOPE_DEG = 5.0        # MEASURED basis: mounting audit F/G/H (25efcff)
MAST_UP_MAX_BASKET_KG = 10.0       # = basket walking rating (load_model.BASKET_RATED_WALK_KG)
MAST_UP_INHIBIT_MARGIN_M = 0.08    # inhibit below this margin (spec v1 section 6)
MAST_UP_MIN_HEAD_PUSH_N = 15.0     # head-push-to-tip must stay >= this
# The rigid-body head push (m g d / lever) over-predicts: the legs are position servos (kp 120-300) and
# the stance yields before a foot lifts. MEASURED (review s7, 5 N/s ramp at the mast head, aligned model,
# 2026-11-17): rigid 34.2 N vs sim lift-off 28.6 N (3 m mast, 1.5 kg head, 10 kg basket; ratio 0.84),
# 26.8 vs 22.6 N at 5 deg roll (0.84), 44.0 vs 38.4 N with a 2.2 m mast (0.87). The probe reports
# rigid x HEAD_PUSH_COMPLIANCE (0.80 = below the worst measured ratio) and keeps the rigid value as
# head_push_rigid_n.
HEAD_PUSH_COMPLIANCE = 0.80

# Debounce (review F3): a single tripod-gait sample can show 2 feet down or the CoM outside the 3 feet
# that happen to touch; those last less than one stance phase (1 / (2 x 1.8 Hz) = 0.28 s at the default
# gait frequency, locomotion_logic.GAIT_PARAM_DEFAULTS). MonitorCore publishes an inhibit flag (and the
# UNSUPPORTED mode) only after it has been raised continuously for INHIBIT_DEBOUNCE_S. DERIVED: 0.30 s
# > 0.28 s. mast_trip is never debounced. Slower gaits (freq_hz < 1.67) have longer transients.
INHIBIT_DEBOUNCE_S = 0.30
UNDEBOUNCED_INHIBITS = ("mast_trip",)
MAST_TRIP_HEAD_FORCE_N = 15.0      # trip (stop + stow) at this measured head force
MAST_TRIP_BASE_MOMENT_NM = 40.0    # trip at this mast_base moment
MAST_HEAD_RATED_KG = 1.5           # = load_model mast_head_ring rating
MAST_HEAD_WIDE_KG = 3.0            # only in the SCAN_WIDE stance on slopes <= 5 deg

# Slopes (deg). MEASURED basis: mounting audit F/G/H (25efcff) at leg kp 240-300.
WALK_MAX_PITCH_DEG = 10.0          # nose-up/down, mast stowed
WALK_MAX_ROLL_DEG = 10.0           # cross-slope, mast stowed
CLIMB_REVERSE_ABOVE_DEG = 5.0      # above this nose-up the robot climbs in reverse
PARK_NOSE_DOWNHILL_ABOVE_DEG = 5.0  # safe-park / e-stop on > 5 deg: point the nose downhill

# Fall: the existing safety_logic.TiltFallDetector (35 deg) is kept; repeated here for the report.
FALL_TILT_DEG = 35.0

# Inhibit flags published on /nandi/stability (advisory). Consumers map them to commands later.
INHIBIT_WALK = "walk"                        # no walking at all
INHIBIT_WALK_FORWARD_UPHILL = "walk_forward_uphill"  # climb this slope in reverse
INHIBIT_MAST_UP = "mast_up"                  # do not raise (or keep raised) the mast
INHIBIT_ARM_PICK = "arm_pick"                # no pick / clip with the arm
INHIBIT_BASKET_LOAD = "basket_load"          # do not add produce to the basket
INHIBIT_MAST_TRIP = "mast_trip"              # trip: stop and stow the mast now
INHIBITS = (INHIBIT_WALK, INHIBIT_WALK_FORWARD_UPHILL, INHIBIT_MAST_UP, INHIBIT_ARM_PICK,
            INHIBIT_BASKET_LOAD, INHIBIT_MAST_TRIP)

_EPS = 1e-12


# --- geometry -----------------------------------------------------------------------------------
def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        raise ValueError("zero-length vector")
    return v / n


def convex_hull(points: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    """2D convex hull, counter-clockwise, no repeated or collinear vertices (Andrew's monotone chain).

    Returns fewer than 3 points when the input is degenerate (all coincident or collinear).
    """
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= _EPS:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= _EPS:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull


def support_polygon(contact_xy: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    """CCW support polygon of the contact points (x, y). < 3 vertices means no support area."""
    return convex_hull(contact_xy)


def com_margin(com_xy: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    """Signed distance (m) of the projected CoM to the polygon boundary: + inside, - outside.

    A polygon with fewer than 3 vertices has no area: -inf.
    """
    if len(polygon) < 3:
        return -math.inf
    return float(geofence_margin(com_xy, polygon))


def gravity_frame(gravity_dir: Sequence[float] = (0.0, 0.0, -1.0)) -> np.ndarray:
    """3x3 rotation whose rows are (e1, e2, up) with up = -gravity; e1 is world x made horizontal.

    Coordinates p @ R.T give (horizontal_1, horizontal_2, height above the origin along -g).
    """
    gx, gy, gz = (float(v) for v in gravity_dir)          # plain floats: called at 50 Hz by the WP6 hook
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n < _EPS:
        raise ValueError("zero-length vector")
    ux, uy, uz = -gx / n, -gy / n, -gz / n
    rx, ry, rz = (1.0, 0.0, 0.0) if abs(ux) < 0.9 else (0.0, 1.0, 0.0)
    k = rx * ux + ry * uy + rz * uz
    ax, ay, az = rx - k * ux, ry - k * uy, rz - k * uz
    m = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / m, ay / m, az / m
    return np.array([[ax, ay, az],
                     [uy * az - uz * ay, uz * ax - ux * az, ux * ay - uy * ax],
                     [ux, uy, uz]])


def support_polygon_3d(points: Iterable[Sequence[float]],
                       gravity_dir: Sequence[float] = (0.0, 0.0, -1.0)) -> List[np.ndarray]:
    """Support polygon of 3D contact points: the hull in the plane normal to gravity, CCW seen from
    above (from -g), returned as the original 3D points (the highest one where two project together)."""
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    if not pts:
        return []
    R = gravity_frame(gravity_dir)
    by_xy: Dict[Tuple[float, float], np.ndarray] = {}
    for p in pts:
        q = R @ p
        key = (float(q[0]), float(q[1]))
        if key not in by_xy or (R @ by_xy[key])[2] < q[2]:
            by_xy[key] = p
    return [by_xy[k] for k in convex_hull(by_xy.keys())]


def edge_tip_angles(com: Sequence[float], polygon_pts_3d: Sequence[Sequence[float]],
                    gravity_dir: Sequence[float] = (0.0, 0.0, -1.0)) -> List[dict]:
    """Per edge i (vertex i -> i+1 of a CCW polygon seen from above):
        d_perp_m    horizontal distance of the CoM inside the edge line (+ inside)
        h_m         CoM height above the edge line, along -g
        tip_deg     atan2(d_perp, h): rotation about the edge that brings the CoM over it
        outward     unit horizontal outward normal of the edge (the downhill direction it tips to)
        z_edge_m    edge midpoint height along -g (for head_push_to_tip)
    """
    n = len(polygon_pts_3d)
    if n < 3:
        return []
    R = gravity_frame(gravity_dir)
    up = R[2]
    c = np.asarray(com, dtype=np.float64)
    out = []
    for i in range(n):
        a = np.asarray(polygon_pts_3d[i], dtype=np.float64)
        b = np.asarray(polygon_pts_3d[(i + 1) % n], dtype=np.float64)
        u = _unit(b - a)
        # horizontal inward normal: up x u_h, where u_h is the edge direction made horizontal
        u_h = _unit(u - np.dot(u, up) * up)
        n_in = np.cross(up, u_h)
        r = c - a
        r = r - np.dot(r, u) * u          # vector from the edge line to the CoM, normal to the edge
        # basis of the plane normal to the edge axis: up_p (up made normal to u) and n_p (inward)
        up_p = _unit(up - np.dot(up, u) * u)
        n_p = np.cross(up_p, u)
        d = float(np.dot(r, n_p))
        h = float(np.dot(r, up_p))
        out.append({
            "d_perp_m": d,
            "h_m": h,
            "tip_deg": math.degrees(math.atan2(d, h)),
            "outward": (-n_in).tolist(),
            "z_edge_m": float(np.dot(0.5 * (a + b), up)),
        })
    return out


def tip_angles(com: Sequence[float], polygon_pts_3d: Sequence[Sequence[float]],
               gravity_dir: Sequence[float] = (0.0, 0.0, -1.0)) -> List[float]:
    """tip_deg of every edge (see edge_tip_angles); the stance tip angle is the minimum."""
    return [e["tip_deg"] for e in edge_tip_angles(com, polygon_pts_3d, gravity_dir)]


def min_tip_angle(com, polygon_pts_3d, gravity_dir=(0.0, 0.0, -1.0)) -> float:
    a = tip_angles(com, polygon_pts_3d, gravity_dir)
    return min(a) if a else -math.inf


def head_push_to_tip(m_total: float, d_perp: float, z_head: float, z_edge: float, g: float = G) -> float:
    """Horizontal force (N) at height z_head that tips m_total about an edge at z_edge whose CoM
    distance is d_perp: F = m g d / (z_head - z_edge). 0 when the CoM is already outside."""
    lever = z_head - z_edge
    if lever <= _EPS:
        return math.inf
    return max(0.0, m_total * g * d_perp / lever)


def min_head_push_to_tip(m_total: float, com, polygon_pts_3d, head_pos,
                         gravity_dir=(0.0, 0.0, -1.0), g: float = G) -> float:
    """Smallest head_push_to_tip over all polygon edges (N); -inf without a support polygon."""
    edges = edge_tip_angles(com, polygon_pts_3d, gravity_dir)
    if not edges:
        return -math.inf
    z_head = float(np.dot(np.asarray(head_pos, dtype=np.float64), gravity_frame(gravity_dir)[2]))
    return min(head_push_to_tip(m_total, e["d_perp_m"], z_head, e["z_edge_m"], g) for e in edges)


def _pitch_roll_from_up(up: np.ndarray, yaw: float) -> Tuple[float, float]:
    """(pitch_deg, roll_deg) of a unit up-normal expressed in the heading frame (yaw about world z).

    pitch > 0 nose up, roll > 0 left side up: a plane z = a x + b y (heading frame) has normal
    ~(-a, -b, 1), so pitch = atan2(-n_x, n_z), roll = atan2(-n_y, n_z).
    """
    c, s = math.cos(yaw), math.sin(yaw)
    nx = c * up[0] + s * up[1]
    ny = -s * up[0] + c * up[1]
    nz = up[2]
    return math.degrees(math.atan2(-nx, nz)), math.degrees(math.atan2(-ny, nz))


def slope_from_points(points: Sequence[Sequence[float]], yaw: float = 0.0) -> Optional[Tuple[float, float]]:
    """Least-squares plane through >= 3 non-collinear 3D points (e.g. the 6 foot contacts).

    Returns (pitch_deg, roll_deg) in the heading frame of `yaw` (rad), or None if degenerate.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] < 3:
        return None
    Q = P - P.mean(axis=0)
    _, sv, vt = np.linalg.svd(Q)
    if sv[1] < 1e-9:  # collinear
        return None
    n = vt[2]
    if n[2] < 0:
        n = -n
    return _pitch_roll_from_up(n / np.linalg.norm(n), yaw)


def quat_to_mat(quat: Sequence[float]) -> np.ndarray:
    """3x3 rotation (body -> world) of a MuJoCo (w, x, y, z) quaternion; normalises first."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), _EPS)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_from_quat(quat: Sequence[float]) -> float:
    """Heading (rad) of the body x axis projected on the world xy plane."""
    R = quat_to_mat(quat)
    return math.atan2(R[1, 0], R[0, 0])


def slope_from_mat(R: np.ndarray) -> Tuple[float, float]:
    """(pitch_deg, roll_deg) of a body rotation matrix (body -> frame whose z is up)."""
    R = np.asarray(R, dtype=np.float64)
    return _pitch_roll_from_up(R[:, 2], math.atan2(R[1, 0], R[0, 0]))


def slope_from_quat(quat: Sequence[float]) -> Tuple[float, float]:
    """(pitch_deg, roll_deg) of the body (IMU) z axis in its own heading frame; same convention as
    slope_from_points, so a body standing square on a plane reports the plane's slope."""
    return slope_from_mat(quat_to_mat(quat))


def cop(points: Sequence[Sequence[float]], normal_forces: Sequence[float]) -> Optional[np.ndarray]:
    """Centre of pressure: normal-force-weighted mean of the contact points (None if no load)."""
    P = np.asarray(points, dtype=np.float64)
    f = np.abs(np.asarray(normal_forces, dtype=np.float64))
    if P.size == 0 or float(f.sum()) <= _EPS:
        return None
    return (P * f[:, None]).sum(axis=0) / float(f.sum())


# --- modes and limits -------------------------------------------------------------------------------
def infer_mode(n_contacts: int, mast_h_m: Optional[float], hip_abs_rad: Sequence[float]) -> str:
    """Stance mode from what the robot is doing: UNSUPPORTED (< 3 feet), MAST_UP (mast > 1.0 m),
    SCAN_WIDE (6 feet, every |hip| >= SCAN_WIDE_DETECT_HIP_RAD), STAND (6 feet), else TRIPOD_WALK."""
    if n_contacts < 3:
        return MODE_UNSUPPORTED
    if mast_h_m is not None and mast_h_m > MAST_STOWED_MAX_H_M:
        return MODE_MAST_UP
    hips = [abs(float(h)) for h in hip_abs_rad]
    if n_contacts >= 6 and hips and min(hips) >= SCAN_WIDE_DETECT_HIP_RAD:
        return MODE_SCAN_WIDE
    if n_contacts >= 6:
        return MODE_STAND
    return MODE_TRIPOD_WALK


def mode_limits(mode: str) -> dict:
    """Minimum margin (m) and tip angle (deg) for a mode (None = not a requirement of that mode)."""
    return {
        MODE_STAND: {"min_margin_m": STAND_MIN_MARGIN_M, "min_tip_deg": STAND_MIN_TIP_DEG},
        MODE_TRIPOD_WALK: {"min_margin_m": TRIPOD_MIN_MARGIN_M, "min_tip_deg": TRIPOD_MIN_TIP_DEG},
        MODE_SCAN_WIDE: {"min_margin_m": SCAN_WIDE_MIN_MARGIN_M, "min_tip_deg": None},
        MODE_MAST_UP: {"min_margin_m": MAST_UP_INHIBIT_MARGIN_M, "min_tip_deg": None},
        MODE_UNSUPPORTED: {"min_margin_m": None, "min_tip_deg": None},
    }[mode]


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def classify(mode: str, inputs: dict) -> dict:
    """Check the section 6 limits for `mode`. Advisory: returns {ok, inhibit, reasons}.

    inputs (all optional except where a check needs them; missing values skip that check):
        n_contacts, margin_m, tip_angle_deg, pitch_deg, roll_deg, basket_kg, basket_limit_kg,
        mast_h_m, head_kg, head_push_to_tip_n, head_force_n, mast_base_moment_nm, stance_wide (bool)
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    inhibit: List[str] = []
    reasons: List[str] = []

    def add(flag: str, why: str):
        if flag not in inhibit:
            inhibit.append(flag)
        if why not in reasons:
            reasons.append(why)

    g = inputs.get
    n = int(g("n_contacts", 0) or 0)
    margin, tip = g("margin_m"), g("tip_angle_deg")
    pitch, roll = g("pitch_deg"), g("roll_deg")
    lim = mode_limits(mode)

    if mode == MODE_UNSUPPORTED:
        inhibit.extend([INHIBIT_WALK, INHIBIT_MAST_UP, INHIBIT_ARM_PICK, INHIBIT_BASKET_LOAD])
        reasons.append(f"{n} feet in contact (< 3): no support polygon")
        return {"ok": False, "inhibit": inhibit, "reasons": reasons}

    if lim["min_margin_m"] is not None and _finite(margin) and margin < lim["min_margin_m"]:
        why = f"{mode}: margin {margin:.3f} m < {lim['min_margin_m']:.3f} m"
        if mode == MODE_MAST_UP:
            add(INHIBIT_MAST_UP, why)
        elif mode == MODE_TRIPOD_WALK:
            add(INHIBIT_WALK, why)
        else:
            add(INHIBIT_ARM_PICK, why)
            add(INHIBIT_MAST_UP, why)
    if lim["min_tip_deg"] is not None and _finite(tip) and tip < lim["min_tip_deg"]:
        why = f"{mode}: tip angle {tip:.1f} deg < {lim['min_tip_deg']:.1f} deg"
        add(INHIBIT_WALK if mode == MODE_TRIPOD_WALK else INHIBIT_ARM_PICK, why)
        add(INHIBIT_MAST_UP, why)

    # slopes
    if _finite(pitch) and abs(pitch) > WALK_MAX_PITCH_DEG:
        add(INHIBIT_WALK, f"pitch {pitch:.1f} deg beyond +-{WALK_MAX_PITCH_DEG:.0f} deg")
    if _finite(roll) and abs(roll) > WALK_MAX_ROLL_DEG:
        add(INHIBIT_WALK, f"cross-slope {roll:.1f} deg beyond +-{WALK_MAX_ROLL_DEG:.0f} deg")
    if _finite(pitch) and pitch > CLIMB_REVERSE_ABOVE_DEG:
        add(INHIBIT_WALK_FORWARD_UPHILL, f"nose-up {pitch:.1f} deg > {CLIMB_REVERSE_ABOVE_DEG:.0f} deg: climb in reverse")
    slope = max(abs(pitch) if _finite(pitch) else 0.0, abs(roll) if _finite(roll) else 0.0)

    # basket
    bkg, blim = g("basket_kg"), g("basket_limit_kg")
    if _finite(bkg) and _finite(blim) and bkg >= blim:
        add(INHIBIT_BASKET_LOAD, f"basket {bkg:.2f} kg >= limit {blim:.2f} kg")

    # mast
    mast_h = g("mast_h_m")
    head_kg = g("head_kg")
    wide = bool(g("stance_wide", mode == MODE_SCAN_WIDE))
    head_rated = MAST_HEAD_WIDE_KG if (wide and slope <= MAST_UP_MAX_SLOPE_DEG) else MAST_HEAD_RATED_KG
    if mode == MODE_MAST_UP:
        add(INHIBIT_WALK, f"mast up ({mast_h:.2f} m > {MAST_STOWED_MAX_H_M:.1f} m): stance only"
            if _finite(mast_h) else "mast up: stance only")
        if n < 6:
            add(INHIBIT_MAST_UP, f"MAST_UP needs all 6 feet down ({n})")
        if slope > MAST_UP_MAX_SLOPE_DEG:
            add(INHIBIT_MAST_UP, f"slope {slope:.1f} deg > {MAST_UP_MAX_SLOPE_DEG:.0f} deg with the mast up")
        if _finite(bkg) and bkg > MAST_UP_MAX_BASKET_KG:
            add(INHIBIT_MAST_UP, f"basket {bkg:.1f} kg > {MAST_UP_MAX_BASKET_KG:.0f} kg with the mast up")
        hp = g("head_push_to_tip_n")
        if _finite(hp) and hp < MAST_UP_MIN_HEAD_PUSH_N:
            add(INHIBIT_MAST_UP, f"head push to tip {hp:.1f} N < {MAST_UP_MIN_HEAD_PUSH_N:.0f} N")
        hf, mb = g("head_force_n"), g("mast_base_moment_nm")
        if _finite(hf) and hf >= MAST_TRIP_HEAD_FORCE_N:
            add(INHIBIT_MAST_TRIP, f"mast head force {hf:.1f} N >= {MAST_TRIP_HEAD_FORCE_N:.0f} N")
        if _finite(mb) and mb >= MAST_TRIP_BASE_MOMENT_NM:
            add(INHIBIT_MAST_TRIP, f"mast base moment {mb:.1f} Nm >= {MAST_TRIP_BASE_MOMENT_NM:.0f} Nm")
    elif _finite(mast_h) and mast_h > MAST_STOWED_MAX_H_M:
        add(INHIBIT_WALK, f"mast {mast_h:.2f} m above the {MAST_STOWED_MAX_H_M:.1f} m stowed height")
    if _finite(head_kg) and head_kg > head_rated:
        add(INHIBIT_MAST_UP, f"mast head {head_kg:.2f} kg > rated {head_rated:.1f} kg")

    return {"ok": not inhibit, "inhibit": inhibit, "reasons": reasons}


# --- /nandi/stability JSON contract ----------------------------------------------------------------
STATUS_KEYS = (
    "t", "mode", "n_contacts", "com", "margin_m", "tip_angle_deg", "slope_deg", "head_push_to_tip_n",
    "basket_kg", "basket_limit_kg", "basket_binding", "arm_reach_m", "arm_limit_kg", "leg_rms_nm_max",
    "inhibit", "reasons", "advisory", "enforced",
)


def _num(x, nd: int = 4):
    """Round a number for the wire; non-finite and None become None (JSON null)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def make_status(t: float, mode: str, n_contacts: int, com: Optional[Sequence[float]], margin_m,
                tip_angle_deg, pitch_deg, roll_deg, head_push_to_tip_n, basket_kg, basket_limit_kg,
                basket_binding: Optional[str], arm_reach_m, arm_limit_kg, leg_rms_nm_max,
                inhibit: Sequence[str], reasons: Sequence[str]) -> dict:
    """The /nandi/stability payload (spec v1 section 9) with every number finite or null."""
    return {
        "t": _num(t, 3),
        "mode": str(mode),
        "n_contacts": int(n_contacts),
        "com": None if com is None else [_num(v) for v in com],
        "margin_m": _num(margin_m),
        "tip_angle_deg": _num(tip_angle_deg, 2),
        "slope_deg": {"pitch": _num(pitch_deg, 2), "roll": _num(roll_deg, 2)},
        "head_push_to_tip_n": _num(head_push_to_tip_n, 2),
        "basket_kg": _num(basket_kg, 3),
        "basket_limit_kg": _num(basket_limit_kg, 3),
        "basket_binding": basket_binding,
        "arm_reach_m": _num(arm_reach_m),
        "arm_limit_kg": _num(arm_limit_kg, 3),
        "leg_rms_nm_max": _num(leg_rms_nm_max, 2),
        "inhibit": list(inhibit),
        "reasons": list(reasons),
        "advisory": True,
        "enforced": False,
    }


def _check_numbers(obj) -> bool:
    if obj is None or isinstance(obj, (str, bool)):
        return True
    if isinstance(obj, (int, float)):
        return math.isfinite(float(obj))
    if isinstance(obj, list):
        return all(_check_numbers(v) for v in obj)
    if isinstance(obj, dict):
        return all(_check_numbers(v) for v in obj.values())
    return False


def build_stability_status(status: dict) -> str:
    """Serialise a make_status() dict; refuses missing keys and non-finite numbers."""
    missing = [k for k in STATUS_KEYS if k not in status]
    if missing:
        raise ValueError(f"/nandi/stability payload missing {missing}")
    if not _check_numbers(status):
        raise ValueError("/nandi/stability payload has a non-finite number")
    return json.dumps(status, allow_nan=False, sort_keys=True)


def parse_stability_status(text: str) -> Optional[dict]:
    """Inverse of build_stability_status; None if the text is not a valid payload."""
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(d, dict) or any(k not in d for k in STATUS_KEYS):
        return None
    if d.get("mode") not in MODES or not _check_numbers(d):
        return None
    return d


# =================================================================================================
# WP6 crown and stance rules (ORCHARD-MVP-SPEC 2.3, 5.2, 5.3; MVP-SPEC 5.3). L0 hard safety: every limit
# below is a code constant (WORKFLOWS-SPEC L0: "code only, never YAML"). Pure numpy; the MuJoCo side is
# atharv_nandi/sim/hooks/stability.py, the matrix that measures the table is tools/crown_gate_matrix.py.
# =================================================================================================
from atharv_nandi.core.interfaces import GateDecision  # noqa: E402  (pure dataclass, no ROS)

LEG_NAMES = ("FR", "FL", "MR", "ML", "RR", "RL")
TRIPOD_A = ("FR", "ML", "RR")          # = tripod_gait.TRIPOD_PHASE / stability_probe.TRIPODS
TRIPOD_B = ("FL", "MR", "RL")
RACK_SIDE_Y = -1.0                     # WP1: rack cradles on the right rail (trunk y -0.18)

# OS 2.3 rule 1 (raise while standing)
RAISE_MIN_HEXA_M = 0.13                # 6-foot margin at the target extension
RAISE_MAX_CROSS_DEG = 3.0              # dwell limit (OS 2.1; WP0c: 3 deg for 60 s dwells)
WIND_RAISE_MAX_MPS = 10.0              # raise needs wind < 10 m/s and no gust
# OS 2.3 rule 2 (walk with the crown up)
WALK_START_TRIPOD_M = 0.04
WALK_CONTINUE_TRIPOD_M = 0.03          # 0.01 hysteresis
MARGIN_FILTER_S = 0.3                  # the continue gate fails only after 0.3 s below it
CROWN_WALK_MAX_MPS = 0.15
WIND_CROWN_WALK_MAX_MPS = 8.0          # ASSUMED until the tripod tipping measurement (config/crown_gates.json)
# Crown-up walking slope cap. The WP6 matrix margin alone would allow more (config/crown_gates.json envelope
# >= 0.04 m to 5 deg), but the dynamic walk test (OS 6 "crown walk") validates 0 and 2 deg only, and OS 5.2 states
# "crown-up walking per the gate table (~ <= 2 deg)". Policy, not physics: raise it only with a new walk test.
CROWN_WALK_MAX_SLOPE_DEG = 2.0
# OS 2.3 rule 4 / RTH-DOCKING: standing with the crown up, wind >= 10 or a gust -> retract
WIND_RETRACT_STANDING_MPS = 10.0
STANCE_WITHIN_S = 0.5                  # rule 4 (i): the gait stops into a 6-foot stance within 0.5 s
# OS 2.3 rule 5
FILTERED_TILT_RETRACT_DEG = 6.0        # 0.5 s mean of |roll|, |pitch| relative to the local slope estimate
TILT_FILTER_S = 0.5
INSTANT_TILT_RETRACT_DEG = 12.0
# OS 5.2 slope rules
WALK_MAX_CROSS_DEG = 5.0               # walking any preset <= 5 deg cross-slope; > 5 deg refused
WALK_MAX_TILT_DEG = 5.0                # WP0c: pitch binds (rear-down worst); the 5 deg cap applies to any fall line
DWELL_SHORT_MAX_DEG = 5.0              # 20 s pick cycle (WP0c creep, feasibility report 6)
DWELL_LONG_MAX_DEG = 3.0               # dwells up to 60 s (LEAF SCAN, TREE 360 station, WORK assist)
DWELL_SHORT_S = 20.0
# MVP-SPEC 5.3 gating
PLAN_MIN_STATIC_M = 0.03               # planner / IK / tool change / share lowering refused below this
LATCH_MARGIN_M = 0.01                  # latch (STOPPED / WORK only) below this for 0.3 s
LATCH_SUSTAIN_S = 0.3
STANCE_HYST_S = 0.06                   # a foot leaves the stance set after 60 ms without flag and contact
CROWN_UP_EXT_M = 0.02                  # above this extension the crown counts as "up"
RERAISE_AFTER_S = 30.0                 # OS 5.3: re-raise once after 30 s if the gates are green
# Measured base-robot tripod margin. OS 2.1 / MVP-SPEC 5.3 quote 0.083 m (lumped probe, pre-alignment); the WP0c
# matrix on the aligned model measured 0.114 m (feasibility report 7). Kept for the record only: gates use the
# live margin and the WP6 matrix (config/crown_gates.json), never this number.
BASE_TRIPOD_MARGIN_SPEC_M = 0.083
BASE_TRIPOD_MARGIN_WP0C_M = 0.114

GATE_RAISE, GATE_WALK, GATE_SLOPE, GATE_DWELL, GATE_PLAN = (
    "crown_raise", "crown_walk", "slope", "dwell", "static_margin")
R_STABILITY, R_SLOPE, R_WIND, R_SPEED, R_STANCE, R_ESTOP = (
    "stability", "slope", "wind", "speed", "not_standing", "estop")

CROWN_GATES_JSON = paths.config_path("crown_gates.json")

# OS 2.3 computed table (lumped probe, 29.23 kg): the fallback when config/crown_gates.json is missing.
SPEC_GATE_TABLE = {"0": {"0": 0.050, "2": 0.044, "3": 0.041, "5": 0.035, "10": 0.020},
                   "0.6": {"0": 0.049, "2": 0.042, "3": 0.039, "5": 0.033, "10": 0.017},
                   "1.6": {"0": 0.047, "2": 0.040, "3": 0.037, "5": 0.030, "10": 0.012}}


def _proj_frame(g: Sequence[float]) -> np.ndarray:
    return gravity_frame(g)


def predicted_tripod_margin(com_b: Sequence[float], feet_b: Dict[str, Sequence[float]],
                            gravity_b: Sequence[float] = (0.0, 0.0, -1.0), rack_side_y: float = RACK_SIDE_Y,
                            worst_case: bool = True, com_shift_b: Optional[Sequence[float]] = None,
                            terrain_roll_deg: Optional[float] = None) -> dict:
    """OS 5.2 ``predicted_tripod_margin``: min over tripod A and B of the CoM margin, projected along gravity.

    Everything is in the trunk (body) frame: ``com_b`` the whole-robot CoM (it already holds the current basket
    mass and payload), ``feet_b`` {leg: foot centre}, ``gravity_b`` gravity seen from the body (slope). With
    ``worst_case`` the cross-slope is evaluated both ways, as measured and mirrored, and the worse one is kept: the
    rack side downhill (the default worst case: the rack hangs on the right rail) is always one of the two, and a
    load on the other side (an arm reaching left with a fruit) can never make the prediction more optimistic than
    the measured stance. The along-x component (pitch) is kept as measured. ``com_shift_b`` moves the CoM first
    (e.g. arm at PARK instead of its current pose: m_arm / M x (c_arm_park - c_arm_now)).
    Returns {margin_m, A, B, gravity_b (the binding one), worst_case, side: rack_downhill / rack_uphill / level}.

    ``terrain_roll_deg`` (the local cross-slope, roll > 0 = left side up = rack side down): when given, only the
    TERRAIN part of the cross-slope is mirrored: gravity is turned about the body x axis by 2 roll, and the trunk's
    own transient roll (gait, a basin rim) stays as it is. Without it (a static stance square on its plane, the
    matrix) the body-frame y component is mirrored.
    """
    g0 = np.asarray(_unit(gravity_b), dtype=np.float64)
    c = np.asarray(com_b, dtype=np.float64)
    if com_shift_b is not None:
        c = c + np.asarray(com_shift_b, dtype=np.float64)

    def one(g):
        R = _proj_frame(g)
        cp = (R @ c)[:2]
        out = {}
        for name, legs in (("A", TRIPOD_A), ("B", TRIPOD_B)):
            pts = [(R @ np.asarray(feet_b[k], dtype=np.float64))[:2] for k in legs if k in feet_b]
            out[name] = com_margin(cp, convex_hull(pts)) if len(pts) == 3 else -math.inf
        return out

    cands = [g0]
    if worst_case:
        if terrain_roll_deg is None:
            g1 = g0.copy()
            g1[1] = -g1[1]
        else:
            phi = math.radians(2.0 * float(terrain_roll_deg))
            cs, sn = math.cos(phi), math.sin(phi)
            g1 = np.array([g0[0], cs * g0[1] - sn * g0[2], sn * g0[1] + cs * g0[2]])
        # rack-downhill candidate first: it wins a tie (the documented default worst case)
        rack_up = ((g0[1] * rack_side_y < 0) if terrain_roll_deg is None
                   else (terrain_roll_deg * rack_side_y > 0))
        cands = [g1, g0] if (rack_side_y and rack_up) else [g0, g1]
    best = None
    for g in cands:
        r = one(g)
        m = min(r["A"], r["B"])
        if best is None or m < best[0]:
            best = (m, r, g)
    m, r, g = best
    side = "level"
    if rack_side_y and abs(g[1]) > 1e-9:
        side = "rack_downhill" if g[1] * rack_side_y > 0 else "rack_uphill"
    return {"margin_m": m, "A": r["A"], "B": r["B"], "gravity_b": g.tolist(), "worst_case": bool(worst_case),
            "side": side}


def hexa_margin(com_b: Sequence[float], feet_b: Dict[str, Sequence[float]],
                gravity_b: Sequence[float] = (0.0, 0.0, -1.0), com_shift_b: Optional[Sequence[float]] = None) -> float:
    """6-foot margin (all feet, projected along gravity, body frame)."""
    c = np.asarray(com_b, dtype=np.float64)
    if com_shift_b is not None:
        c = c + np.asarray(com_shift_b, dtype=np.float64)
    R = _proj_frame(gravity_b)
    pts = [(R @ np.asarray(p, dtype=np.float64))[:2] for p in feet_b.values()]
    return com_margin((R @ c)[:2], convex_hull(pts))


def mast_com_shift(dext: float, section_kg: Sequence[float], head_kg: float, total_kg: float,
                   axis_b: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """CoM shift (body frame) when the extension changes by ``dext``: section k (1-based, nested) moves k/3 of it,
    the head the whole of it (OS 2.3 ``ext / 3`` split)."""
    n = len(section_kg)
    moved = sum(kg * (k + 1) / n for k, kg in enumerate(section_kg)) + head_kg
    return np.asarray(axis_b, dtype=np.float64) * (moved * float(dext) / max(total_kg, _EPS))


def slope_tilt_deg(roll_deg: float, pitch_deg: float) -> float:
    """Tilt of the plane (deg) from its heading-frame pitch and roll."""
    tr, tp = math.tan(math.radians(roll_deg)), math.tan(math.radians(pitch_deg))
    return math.degrees(math.atan(math.hypot(tr, tp)))


# ---------------------------------------------------------------------------------------- gate table
class CrownGateTable:
    """The WP6 matrix (config/crown_gates.json ``matrix.envelope``): worst tripod margin over every fall line with
    tilt <= T, per arm pose and extension. ``margin(ext, tilt)`` interpolates linearly in tilt and extension (and
    is conservative beyond the measured tilts: the last two points extrapolate). The arm default is the worse of
    PARK and T4 at each point."""

    def __init__(self, envelope: Dict[str, Dict[str, Dict[str, float]]], run_id: str = "", source: str = ""):
        self.env = {arm: {float(e): {float(t): float(v) for t, v in row.items()} for e, row in ext.items()}
                    for arm, ext in envelope.items()}
        self.run_id, self.source = run_id, source

    @classmethod
    def load(cls, path: Optional[str] = None) -> "CrownGateTable":
        path = path or CROWN_GATES_JSON
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            mx = doc["matrix"]
            return cls(mx["envelope"], mx.get("meta", {}).get("run_id", ""), path)
        except (OSError, KeyError, ValueError, TypeError):
            return cls({"PARK": SPEC_GATE_TABLE}, "", "ORCHARD-MVP-SPEC 2.3 computed table (fallback)")

    @staticmethod
    def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
        if len(xs) == 1:
            return ys[0]
        if x <= xs[0]:
            return ys[0]
        for i in range(1, len(xs)):
            if x <= xs[i]:
                a = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                return ys[i - 1] + a * (ys[i] - ys[i - 1])
        a = (x - xs[-2]) / (xs[-1] - xs[-2])   # extrapolate (falling margins keep falling)
        return ys[-2] + a * (ys[-1] - ys[-2])

    def _arm(self, arm: str, ext: float, tilt: float) -> float:
        rows = self.env[arm]
        exts = sorted(rows)
        vals = []
        for e in exts:
            ts = sorted(rows[e])
            vals.append(self._interp(ts, [rows[e][t] for t in ts], max(0.0, tilt)))
        return self._interp(exts, vals, min(max(ext, exts[0]), exts[-1]))

    def margin(self, ext: float, tilt_deg: float, arm: Optional[str] = None) -> float:
        arms = [arm] if arm is not None else list(self.env)
        return min(self._arm(a, float(ext), float(tilt_deg)) for a in arms if a in self.env)

    def max_tilt(self, ext: float, gate_m: float, arm: Optional[str] = None, step: float = 0.1) -> float:
        """Largest tilt (deg, 0.1 steps up to 15) at which the table margin stays >= gate_m."""
        t = 0.0
        while t <= 15.0 and self.margin(ext, t + step, arm) >= gate_m:
            t += step
        return round(t, 1) if self.margin(ext, 0.0, arm) >= gate_m else -1.0


# ---------------------------------------------------------------------------------------- pure gates
def gate_raise(ext_from: float, ext_to: float, hexa_at_target_m: float, cross_deg: float, wind_mps: float,
               gust: bool, standing: bool, n_stance: int, estop: bool = False,
               tilt_deg: float = 0.0) -> GateDecision:
    """OS 2.3 rule 1, WP6 part (the swept clearance is WP24's): 6-foot stance and gait stopped, 6-foot margin
    >= 0.13 m at the target extension, local cross-slope <= 3 deg, tilt <= the 5 deg walking cap, wind < 10 m/s
    and no gust, e-stop clear. A lower target is never gated (retracts always run)."""
    v = {"ext_from": float(ext_from), "ext_to": float(ext_to), "hexa_m": float(hexa_at_target_m),
         "cross_deg": float(cross_deg), "wind_mps": float(wind_mps), "n_stance": float(n_stance)}
    lim = {"hexa_min_m": RAISE_MIN_HEXA_M, "cross_max_deg": RAISE_MAX_CROSS_DEG, "wind_max_mps": WIND_RAISE_MAX_MPS}
    if ext_to <= ext_from + 1e-9:
        return GateDecision(True, GATE_RAISE, "", v, lim)
    if estop:
        return GateDecision(False, GATE_RAISE, R_ESTOP, v, lim)
    if not standing or n_stance < 6:
        return GateDecision(False, GATE_RAISE, R_STANCE, v, lim)
    if abs(cross_deg) > RAISE_MAX_CROSS_DEG or abs(tilt_deg) > WALK_MAX_TILT_DEG:
        return GateDecision(False, GATE_RAISE, R_SLOPE, v, lim)
    if gust or wind_mps >= WIND_RAISE_MAX_MPS:
        return GateDecision(False, GATE_RAISE, R_WIND, v, lim)
    if not (math.isfinite(hexa_at_target_m) and hexa_at_target_m >= RAISE_MIN_HEXA_M):
        return GateDecision(False, GATE_RAISE, R_STABILITY, v, lim)
    return GateDecision(True, GATE_RAISE, "", v, lim)


def gate_crown_walk(ext: float, tripod_m: float, speed_mps: float, wind_mps: float, gust: bool,
                    slope_deg: float, starting: bool = True) -> GateDecision:
    """OS 2.3 rule 2 (stateless form; ``CrownWalkGate`` adds the 0.3 s filter): with the crown up the predicted
    worst tripod margin must be >= 0.04 m to start (0.03 m to continue), speed <= 0.15 m/s, wind < 8 m/s and no
    gust, slope <= CROWN_WALK_MAX_SLOPE_DEG. A stowed crown passes (the slope rule still applies separately)."""
    gate_m = WALK_START_TRIPOD_M if starting else WALK_CONTINUE_TRIPOD_M
    v = {"ext": float(ext), "tripod_m": float(tripod_m), "speed_mps": float(speed_mps), "wind_mps": float(wind_mps),
         "slope_deg": float(slope_deg)}
    lim = {"tripod_min_m": gate_m, "speed_max_mps": CROWN_WALK_MAX_MPS, "wind_max_mps": WIND_CROWN_WALK_MAX_MPS,
           "slope_max_deg": CROWN_WALK_MAX_SLOPE_DEG}
    if ext <= CROWN_UP_EXT_M:
        return GateDecision(True, GATE_WALK, "", v, lim)
    if speed_mps > CROWN_WALK_MAX_MPS + 1e-9:
        return GateDecision(False, GATE_WALK, R_SPEED, v, lim)
    if gust or wind_mps >= WIND_CROWN_WALK_MAX_MPS:
        return GateDecision(False, GATE_WALK, R_WIND, v, lim)
    if abs(slope_deg) > CROWN_WALK_MAX_SLOPE_DEG + 1e-9:
        return GateDecision(False, GATE_WALK, R_SLOPE, v, lim)
    if not (math.isfinite(tripod_m) and tripod_m >= gate_m):
        return GateDecision(False, GATE_WALK, R_STABILITY, v, lim)
    return GateDecision(True, GATE_WALK, "", v, lim)


def gate_walk_slope(cross_deg: float, pitch_deg: float = 0.0) -> GateDecision:
    """OS 5.2: walking any preset <= 5 deg cross-slope (and, from the WP0c matrix, <= 5 deg tilt on any fall line,
    since rear-down pitch binds); > 5 deg refused (the 10 deg patch demo)."""
    tilt = slope_tilt_deg(cross_deg, pitch_deg)
    v = {"cross_deg": float(cross_deg), "pitch_deg": float(pitch_deg), "tilt_deg": tilt}
    lim = {"cross_max_deg": WALK_MAX_CROSS_DEG, "tilt_max_deg": WALK_MAX_TILT_DEG}
    if abs(cross_deg) > WALK_MAX_CROSS_DEG + 1e-9 or tilt > WALK_MAX_TILT_DEG + 1e-9:
        return GateDecision(False, GATE_SLOPE, R_SLOPE, v, lim)
    return GateDecision(True, GATE_SLOPE, "", v, lim)


def gate_dwell(slope_deg: float, dwell_s: float) -> GateDecision:
    """Dwell presets (WP0c creep): <= 5 deg for a <= 20 s pick cycle, <= 3 deg for longer dwells (to 60 s)."""
    lim_deg = DWELL_SHORT_MAX_DEG if dwell_s <= DWELL_SHORT_S else DWELL_LONG_MAX_DEG
    v = {"slope_deg": float(slope_deg), "dwell_s": float(dwell_s)}
    if abs(slope_deg) > lim_deg + 1e-9:
        return GateDecision(False, GATE_DWELL, R_SLOPE, v, {"slope_max_deg": lim_deg})
    return GateDecision(True, GATE_DWELL, "", v, {"slope_max_deg": lim_deg})


def gate_static_margin(predicted_static_m: float, what: str = "plan") -> GateDecision:
    """MVP-SPEC 5.3: planner / IK targets, tool change and share lowering refused below 0.03 m predicted static."""
    v = {"margin_m": float(predicted_static_m)}
    if not (math.isfinite(predicted_static_m) and predicted_static_m >= PLAN_MIN_STATIC_M):
        return GateDecision(False, f"{GATE_PLAN}:{what}", R_STABILITY, v, {"margin_min_m": PLAN_MIN_STATIC_M})
    return GateDecision(True, f"{GATE_PLAN}:{what}", "", v, {"margin_min_m": PLAN_MIN_STATIC_M})


def check_step(ext: float, speed_mps: float, cross_deg: float, pitch_deg: float = 0.0, wind_mps: float = 0.0,
               walking: bool = True, dwell_s: float = 0.0, table: Optional[CrownGateTable] = None,
               raise_to: Optional[float] = None) -> List[GateDecision]:
    """Planning check of one scripted step / preset default (WP27 tour gates): every failing decision, [] = ok.
    Uses the WP6 matrix table for the margin (no live CoM at plan time)."""
    table = table or CrownGateTable.load()
    tilt = slope_tilt_deg(cross_deg, pitch_deg)
    bad: List[GateDecision] = []
    if walking:
        for dec in (gate_walk_slope(cross_deg, pitch_deg),
                    gate_crown_walk(ext, table.margin(ext, tilt), speed_mps, wind_mps, False, tilt, starting=True)):
            if not dec:
                bad.append(dec)
    elif dwell_s > 0:
        dec = gate_dwell(tilt, dwell_s)
        if not dec:
            bad.append(dec)
    if raise_to is not None and raise_to > ext:
        hexa = table_hexa_margin(raise_to, tilt)
        dec = gate_raise(ext, raise_to, hexa, cross_deg, wind_mps, False, not walking, 6 if not walking else 3,
                         tilt_deg=tilt)
        if not dec:
            bad.append(dec)
    return bad


def table_hexa_margin(ext: float, tilt_deg: float, path: Optional[str] = None) -> float:
    """Worst 6-foot margin of the WP6 matrix up to tilt (deg); the spec's 0.142 m floor when no table exists."""
    try:
        with open(path or CROWN_GATES_JSON, encoding="utf-8") as fh:
            rows = json.load(fh)["matrix"]["rows"]
        sub = [r for r in rows if abs(r["ext"] - ext) < 0.31 or r["ext"] >= ext] or rows
        inb = [r["hexa"] for r in sub if r["tilt_deg"] <= tilt_deg + 0.05]
        return min(inb) if inb else min(r["hexa"] for r in sub)
    except (OSError, KeyError, ValueError, TypeError):
        return 0.142


# ---------------------------------------------------------------------------------------- stateful filters
class StanceSet:
    """MVP-SPEC 5.3: stance set = gait stance flags U foot contacts, with 60 ms hysteresis (a foot leaves the set
    only after 60 ms without both its flag and a contact)."""

    def __init__(self, legs: Sequence[str] = LEG_NAMES, hyst_s: float = STANCE_HYST_S):
        self.legs = tuple(legs)
        self.hyst_s = float(hyst_s)
        self.last_on: Dict[str, float] = {k: -math.inf for k in self.legs}
        self.members: set = set()

    def update(self, t: float, contacts: Iterable[str], flags: Iterable[str] = ()) -> set:
        on = set(contacts) | set(flags)
        for k in self.legs:
            if k in on:
                self.last_on[k] = float(t)
        self.members = {k for k in self.legs if t - self.last_on[k] <= self.hyst_s + 1e-9}
        return set(self.members)

    def reset(self):
        self.last_on = {k: -math.inf for k in self.legs}
        self.members = set()


class SustainFilter:
    """``bad`` must hold continuously for ``sustain_s`` before ``tripped`` turns on; any good sample clears it."""

    def __init__(self, sustain_s: float):
        self.sustain_s = float(sustain_s)
        self.since: Optional[float] = None

    def update(self, t: float, bad: bool) -> bool:
        if not bad:
            self.since = None
            return False
        if self.since is None:
            self.since = float(t)
        return t - self.since >= self.sustain_s - 1e-9

    def reset(self):
        self.since = None


class CrownWalkGate:
    """OS 2.3 rule 2 with state: walking with the crown up starts only at >= 0.04 m and fails only after the
    margin stayed < 0.03 m for 0.3 s (0.01 m hysteresis, 0.3 s filter)."""

    def __init__(self):
        self.active = False
        self.low = SustainFilter(MARGIN_FILTER_S)

    def start(self, tripod_m: float) -> bool:
        self.active = math.isfinite(tripod_m) and tripod_m >= WALK_START_TRIPOD_M
        self.low.reset()
        return self.active

    def update(self, t: float, tripod_m: float) -> bool:
        """True while the continue gate holds (call every tick while walking crown-up)."""
        if not self.active:
            return False
        bad = not (math.isfinite(tripod_m) and tripod_m >= WALK_CONTINUE_TRIPOD_M)
        if self.low.update(t, bad):
            self.active = False
        return self.active

    def stop(self):
        self.active = False
        self.low.reset()


class TiltFilter:
    """OS 2.3 rule 5: 0.5 s running mean of |roll| and |pitch| relative to the local slope estimate (deg)."""

    def __init__(self, window_s: float = TILT_FILTER_S):
        self.window_s = float(window_s)
        self.buf: List[Tuple[float, float, float]] = []

    def update(self, t: float, roll_deg: float, pitch_deg: float, slope_roll_deg: float = 0.0,
               slope_pitch_deg: float = 0.0) -> float:
        self.buf.append((float(t), abs(roll_deg - slope_roll_deg), abs(pitch_deg - slope_pitch_deg)))
        while self.buf and self.buf[0][0] < t - self.window_s + 1e-9:
            self.buf.pop(0)
        if not self.buf:
            return 0.0
        r = sum(b[1] for b in self.buf) / len(self.buf)
        p = sum(b[2] for b in self.buf) / len(self.buf)
        return max(r, p)

    def full(self, t: float) -> bool:
        return bool(self.buf) and t - self.buf[0][0] >= self.window_s - 0.03

    def reset(self):
        self.buf = []


# ---------------------------------------------------------------------------------------- crown supervisor
W_IDLE, W_STANCE, W_RETRACT = "idle", "stance", "retract"


class CrownSupervisor:
    """Per-tick crown and stance decisions (OS 2.3 rules 2, 4, 5; OS 5.3 crown row). Pure: the hook feeds it the
    measured state and turns its outputs into a ``safety`` retract request (WP24 arbiter), a stance request (the
    gait stops into a 6-foot stance) and walk inhibits.

    Ordered wind / gust response (rule 4): (i) stance request at once, (ii) retract once the robot stands in a
    6-foot stance or ``STANCE_WITHIN_S`` after the request (WP24's own backstop retracts at the same 0.5 s),
    (iii) no walking until the crown is stowed. Standing crown-up: wind >= 10 m/s or a gust retracts at once.
    Walking crown-up: wind >= 8 m/s or a gust starts the ordered response.

    Other retracts (rule 5): filtered tilt > 6 deg (once the 0.5 s window is full), instantaneous tilt > 12 deg,
    the walk margin gate failing (predicted tripod < 0.03 m for 0.3 s while walking crown-up), local safety-link
    loss, low-battery return. Each retract is reported once per event with its reason.
    """

    def __init__(self):
        self.walk_gate = CrownWalkGate()
        self.tilt = TiltFilter()
        self.reset()

    def reset(self):
        self.wind_state = W_IDLE
        self.wind_since: Optional[float] = None
        self.wind_reason = "wind"
        self.retract_reason: Optional[str] = None
        self.retracts: List[dict] = []
        self.walking_crown_up = False
        self.last_retract: Optional[dict] = None
        self.reraised = False
        self.walk_gate.stop()
        self.tilt.reset()

    def update(self, t: float, ext: float, walking: bool, six_foot_stance: bool, tripod_m: float,
               roll_deg: float, pitch_deg: float, slope_roll_deg: float = 0.0, slope_pitch_deg: float = 0.0,
               wind_mps: float = 0.0, gust: bool = False, local_link_ok: bool = True,
               low_battery_return: bool = False, retracting: bool = False) -> dict:
        crown_up = ext > CROWN_UP_EXT_M
        filt = self.tilt.update(t, roll_deg, pitch_deg, slope_roll_deg, slope_pitch_deg)
        retract: Optional[str] = None
        stance_request = False
        windy_walk = gust or wind_mps >= WIND_CROWN_WALK_MAX_MPS
        windy_stand = gust or wind_mps >= WIND_RETRACT_STANDING_MPS
        # rule 4: ordered wind response
        if crown_up and self.wind_state == W_IDLE and ((walking and windy_walk) or (not walking and windy_stand)):
            self.wind_state, self.wind_since = W_STANCE, t
        if self.wind_state == W_STANCE:
            stance_request = True
            if six_foot_stance and not walking or t - self.wind_since >= STANCE_WITHIN_S - 1e-9:
                self.wind_state = W_RETRACT
                self.wind_reason = "gust" if gust else "wind"
        if self.wind_state == W_RETRACT:
            stance_request = True
            if not crown_up:
                self.wind_state, self.wind_since = W_IDLE, None
            else:
                # re-fires every tick until a safety retract runs (as the margin / tilt retracts): a retract
                # that could not be sent (e-stop hold, refused) is sent again on the next tick that can
                retract = self.wind_reason
        # rule 2: margin gate while walking crown-up
        if crown_up and walking:
            if not self.walking_crown_up:
                self.walking_crown_up = True
                ok = self.walk_gate.start(tripod_m)
            else:
                ok = self.walk_gate.update(t, tripod_m)
            if not ok and retract is None:
                retract = "margin"
        else:
            self.walking_crown_up = False
            self.walk_gate.stop()
        # rule 5
        if crown_up and retract is None:
            if max(abs(roll_deg), abs(pitch_deg)) > INSTANT_TILT_RETRACT_DEG:
                retract = "tilt"
            elif self.tilt.full(t) and filt > FILTERED_TILT_RETRACT_DEG:
                retract = "tilt_filtered"
            elif not local_link_ok:
                retract = "local_link_lost"
            elif low_battery_return:
                retract = "low_battery_return"
        new_event = False
        if retract is not None and crown_up and not retracting:
            new_event = True
            ev = {"t": round(float(t), 3), "reason": retract, "ext_m": round(float(ext), 4)}
            self.retracts.append(ev)
            self.last_retract = ev
            self.reraised = False
        walk_inhibit = []
        if self.wind_state != W_IDLE:
            walk_inhibit.append("wind_stance")
        return {"retract": retract if new_event else None, "retract_reason": retract, "stance_request": stance_request,
                "wind_state": self.wind_state, "walk_inhibit": walk_inhibit, "tilt_filtered_deg": round(filt, 3),
                "walk_gate_active": self.walk_gate.active}

    def reraise_due(self, t: float, standing: bool, gates_green: bool) -> Optional[float]:
        """OS 5.3: re-raise once after 30 s if the gates are green -> the extension to go back to, else None."""
        ev = self.last_retract
        if (ev is None or self.reraised or not standing or not gates_green or self.wind_state != W_IDLE
                or ev["reason"] in ("local_link_lost", "low_battery_return") or t - ev["t"] < RERAISE_AFTER_S):
            return None
        self.reraised = True
        return float(ev["ext_m"])


class StabilityLatch:
    """MVP-SPEC 5.3: latch only in the STOPPED and WORK states, when the margin is < 0.01 m for 0.3 s. While
    walking the margin is reported, not latched (the tilt-fall latch is the walking protection). Degenerate
    stances (< 3 feet in the set) are reported and not gated."""

    def __init__(self):
        self.low = SustainFilter(LATCH_SUSTAIN_S)
        self.latched = False
        self.reason = ""

    def update(self, t: float, state: str, margin_m: float, n_stance: int) -> bool:
        if self.latched:
            return True
        if state not in ("STOPPED", "WORK") or n_stance < 3:
            self.low.reset()
            return False
        if self.low.update(t, not (math.isfinite(margin_m) and margin_m >= LATCH_MARGIN_M)):
            self.latched, self.reason = True, f"stability margin < {LATCH_MARGIN_M} m for {LATCH_SUSTAIN_S} s"
        return self.latched

    def reset(self, margin_ok: bool) -> bool:
        if self.latched and margin_ok:
            self.latched, self.reason = False, ""
            self.low.reset()
        return not self.latched


# ---------------------------------------------------------------------------------------- basin rims (WP6 walk test)
# MEASURED (tests/test_crown_gates.py, crown walk, orchard_lite seeds 7/11/23): walking crown-up ACROSS a trunk-basin
# rim (0.10 m crest, 30 deg flanks) pitches the trunk 9-14 deg and drops the predicted tripod margin to -0.02..-0.14 m
# (ext 1.6) and 0.005..0.03 m (ext 0.6): the margin gate retracts, as it must. Crown-up paths therefore keep the hull
# outside the rim footprint; a rim is crossed stowed (or the station is reached stop-and-go, rule 7).
BASIN_RIM_H_M = 0.10
BASIN_RIM_CREST_M = 0.15
BASIN_EDGE_DEG = 30.0
CROWN_PATH_HULL_R_M = 0.40             # robot hull radius for the path check (trunk half-diagonal + feet)


def basin_keepout_radius(basin_r_m: float, rim_h_m: float = BASIN_RIM_H_M, crest_m: float = BASIN_RIM_CREST_M,
                         edge_deg: float = BASIN_EDGE_DEG) -> float:
    """Outer radius of a basin rim footprint (crest + half-cosine flank, world/terrain_orchard geometry)."""
    flank = math.pi * rim_h_m / (2.0 * math.tan(math.radians(edge_deg)))
    return float(basin_r_m) + float(crest_m) + flank


def gate_crown_path(ext: float, path_xy: Sequence[Sequence[float]], basins: Sequence[Tuple[float, float, float]],
                    hull_r: float = CROWN_PATH_HULL_R_M) -> GateDecision:
    """A crown-up path (ext > CROWN_UP_EXT_M) must keep the robot hull outside every basin rim footprint.
    ``basins`` = [(x, y, basin_r_m)]; the path is a polyline of robot centre points."""
    if ext <= CROWN_UP_EXT_M or not basins or not path_xy:
        return GateDecision.ok("crown_path")
    worst = math.inf
    for p in path_xy:
        for bx, by, br in basins:
            worst = min(worst, math.hypot(p[0] - bx, p[1] - by) - basin_keepout_radius(br) - hull_r)
    if worst < 0.0:
        return GateDecision.refuse("crown_path", "basin_rim", clearance_m=round(worst, 3))
    return GateDecision.ok("crown_path", clearance_m=round(worst, 3))
