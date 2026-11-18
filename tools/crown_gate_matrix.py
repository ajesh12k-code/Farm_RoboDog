#!/usr/bin/env python3
"""WP6 crown gate matrix, e-stop sit-down and tripod wind tipping on the WP1 full layout (ORCHARD-MVP-SPEC 2.3, 6).

The WP0c sweep (``tools/feasibility_sweep.py matrix``) measured the matrix on the frozen pre-WP1 robot with a
lumped mast it added itself. This tool re-runs it on the SHIPPED WP1 robot (``atharv_nandi.xml``: sectioned mast
``mast_s1..s3``, crown head, rack cradles, tray, bumper, belly rests), plus what WP1 does not carry yet:

    head            the crown head is raised to the 0.6 kg hard cap (WP1 ships 0.40 kg; +0.2 kg at the head centre)
    rack tools      T2 / T4 / T5 masses (WP0c ``TOOLS``, assumptions until WP2) hanging under the WP1 cradles
                    ``rack_cradle_0..2`` (right rail, y -0.18) in the WP0c frozen order
    basket          3 kg payload box at ``basket_payload``
    arm             PARK (home keyframe) or the worst T4 pose with T4 mounted and a 1.8 kg fruit (WP0c
                    ``worst_t4_pose``; T4 is then not on the rack)

Sub-commands (each <= 180 s on the capped 2-CPU runner):

    matrix [--write]   ext {0, 0.6, 1.6} x roll {0, +-2, +-3, +-5} x pitch {0, +-5, +-10} x arm {PARK, T4}
                       (210 rows; roll > 0 = rack side downhill, pitch < 0 = rear / mast downhill), plus a heading
                       sweep (tilt 2..10 deg x 24 fall lines, ext x arm). Writes config/crown_gates.json.
    sitdown [--write]  e-stop at ext 1.6 on 5 deg cross-slope (rack downhill and uphill): the crown is braked
                       high, the legs lose power (kp x 0, power off, and kp x 0.08, a soft sit) and the trunk sinks
                       until the folded legs or the belly rests carry it. Records whether it fell, which geoms
                       touch the ground and the CoM margin to the support polygon of every contact after 5 s.
    tipwind [--write]  tripod stance (tripod B lifted mid-swing, held), ext 1.6, flat: WP7 drag model; tipping
                       speed per wind direction from the static moment (v_tip = v_ref sqrt(m g d / M(v_ref))),
                       confirmed dynamically at 0.9 and 1.1 v_tip (stance foot lift-off / tilt > 10 deg).

Static margins: whole-robot CoM projected along gravity onto the 6-foot hull and tripods A / B after a 1.5 s
settle (WP0c ``margins``). Each run is tagged ``wp6-<kind>-<utc>-<sha>``.
"""
import argparse
import datetime
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from atharv_nandi import paths  # noqa: E402
from tools import feasibility_sweep as fs  # noqa: E402  (helpers only: slopes, margins, T4 pose, tool masses)
from atharv_nandi.core import stability as st  # noqa: E402
from atharv_nandi.core import wind_model as wm  # noqa: E402
from atharv_nandi.core.tripod_gait import TripodGaitGenerator  # noqa: E402

ROBOT_XML = os.path.join(paths.ROBOT_MODEL_DIR, "atharv_nandi.xml")
OUT_JSON = os.path.join(REPO, "config", "crown_gates.json")
HEAD_CAP_KG = 0.6
SECTIONS = ("mast_s1", "mast_s2", "mast_s3")
MATRIX_EXT = (0.0, 0.6, 1.6)
MATRIX_ROLL = (0.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0)
MATRIX_PITCH = (-10.0, -5.0, 0.0, 5.0, 10.0)
HEADING_TILTS = (2.0, 3.0, 5.0, 7.0, 10.0)
HEADING_DIRS = tuple(float(a) for a in range(0, 360, 15))
TILT_BOUNDS = (0.0, 2.0, 3.0, 5.0, 7.0, 10.0)
SETTLE_S = 1.5
SITDOWN_KP_SCALE = 0.08
O = mujoco.mjtObj


def run_meta(kind: str) -> dict:
    sha = os.environ.get("NANDI_GIT_SHA", "").strip()
    if not sha:
        try:
            sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
        except Exception:
            sha = ""
    sha = sha or "unknown"
    now = datetime.datetime.now(datetime.timezone.utc)
    return {"run_id": f"wp6-{kind}-{now.strftime('%Y%m%dT%H%M%SZ')}-{sha}",
            "device_id": os.environ.get("NANDI_DEVICE_ID", f"capped-runner-{platform.machine()}-2cpu-2gb"),
            "git_sha": sha, "mujoco": mujoco.__version__, "utc": now.isoformat(timespec="seconds")}


def _id(m, t, name):
    return mujoco.mj_name2id(m, t, name)


def _frozen() -> dict:
    with open(fs.CANDIDATES_JSON, encoding="utf-8") as fh:
        return json.load(fh).get("frozen") or {}


def build_full_layout(mounted: Optional[str] = None, fruit_kg: float = 0.0, basket_kg: float = 3.0,
                      head_kg: float = HEAD_CAP_KG) -> mujoco.MjModel:
    """WP1 robot + rack tools + basket + head cap (+ T4 and fruit on the wrist). Additions are contype 0."""
    spec = mujoco.MjSpec.from_file(ROBOT_XML)
    for n in ("ridge_1", "ridge_2"):
        g = spec.geom(n)
        if g is not None:
            spec.delete(g)
    nc = dict(contype=0, conaffinity=0, group=5)
    head = spec.body("crown_head")
    shipped = sum(g.mass for g in head.geoms) if head is not None else 0.0
    if head is not None and head_kg > shipped + 1e-6:
        head.add_geom(name="wp6_head_cap", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
                      mass=head_kg - shipped, **nc)
    order = [c["tool"] for c in (_frozen().get("rack") or {}).get("cradles", [])] or ["T2", "T4", "T5"]
    trunk = spec.body("trunk")
    for i, tool in enumerate(order):
        cg = spec.geom(f"rack_cradle_{i}")
        if cg is None or tool == mounted or tool not in fs.TOOLS:
            continue
        p = list(cg.pos)
        t = fs.TOOLS[tool]
        trunk.add_geom(name=f"wp6_racked_{tool}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=[p[0], p[1], p[2] - 0.01, p[0], p[1], p[2] - t["len"]], size=[max(t["r"], 0.02), 0, 0],
                       mass=t["mass"], **nc)
    if basket_kg > 0:
        spec.body("dorsal_basket").add_geom(name="wp6_payload", type=mujoco.mjtGeom.mjGEOM_BOX,
                                            pos=[0, 0, 0.041 + 0.03], size=[0.15, 0.10, 0.03], mass=basket_kg, **nc)
    if mounted:
        t = fs.TOOLS[mounted]
        wrist = spec.body("arm_wrist")
        wrist.add_geom(name=f"wp6_tool_{mounted}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                       fromto=[fs.FLANGE_X + 0.01, 0, 0, t["tcp"][0] - 0.04, 0, 0], size=[t["r"], 0, 0],
                       mass=t["mass"], **nc)
        if fruit_kg > 0:
            wrist.add_geom(name="wp6_fruit", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=list(t["tcp"]),
                           size=[0.06, 0, 0], mass=fruit_kg, **nc)
    return spec.compile()


def set_ext(m, ext: float, rest: Optional[List[float]] = None) -> List[float]:
    ids = [_id(m, O.mjOBJ_BODY, n) for n in SECTIONS]
    rest = rest if rest is not None else [float(m.body_pos[b, 2]) for b in ids]
    for b, z0 in zip(ids, rest):
        m.body_pos[b, 2] = z0 + ext / 3.0
    return rest


def _start(m, arm_q=None) -> mujoco.MjData:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_ctrl[0]
    if arm_q is not None:
        d.qpos[7:12] = arm_q
        d.ctrl[0:5] = np.asarray(arm_q) + fs.gravity_offsets(m, d)
    return d


def predicted(m, d) -> float:
    """controllers.stability.predicted_tripod_margin on this state (worst of the measured and mirrored cross-slope)."""
    feet = {f[:2]: d.geom_xpos[_id(m, O.mjOBJ_GEOM, f)].copy() for f in fs.FEET}
    R = d.xmat[1].reshape(3, 3)
    p0 = d.xpos[1]
    com_b = R.T @ (d.subtree_com[1] - p0)
    feet_b = {k: R.T @ (v - p0) for k, v in feet.items()}
    g_b = R.T @ np.asarray(m.opt.gravity, float)
    return st.predicted_tripod_margin(com_b, feet_b, g_b)["margin_m"]


def matrix_case(models: dict, rest: dict, ext, arm, q, roll=0.0, pitch=0.0, tilt=None, dh=None) -> dict:
    m = models[arm]
    set_ext(m, ext, rest[arm])
    fs._apply_slope(m, roll, pitch, tilt, dh)
    tilt_deg, fall = fs.tilt_of(m)
    d = _start(m, q if arm == "T4" else None)
    fell = False
    for _ in range(int(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.12:
            fell = True
            break
    mg = fs.margins(m, d)
    rack = (("downhill" if roll > 0 else "uphill" if roll < 0 else "level") if tilt is None
            else fs.rack_side(fall, tilt_deg))
    return {"ext": ext, "arm": arm, "roll_deg": roll if tilt is None else None,
            "pitch_deg": pitch if tilt is None else None, "tilt_deg": tilt_deg, "downhill_deg": fall, "rack": rack,
            **mg, "predicted": round(predicted(m, d), 4), "fell": fell,
            "mass_kg": round(float(m.body_subtreemass[1]), 3)}


def _models():
    q, _ = fs.worst_t4_pose(_frozen(), 1.8)
    models = {"PARK": build_full_layout()}
    if q is not None:
        models["T4"] = build_full_layout(mounted="T4", fruit_kg=1.8)
    rest = {k: set_ext(mm, 0.0) for k, mm in models.items()}
    return models, rest, q


def envelope(rows: List[dict]) -> dict:
    """env[arm][ext][T] = min tripod over every row with tilt <= T (any fall line)."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm in sorted({r["arm"] for r in rows}):
        out[arm] = {}
        for ext in sorted({r["ext"] for r in rows}):
            sub = [r for r in rows if r["arm"] == arm and r["ext"] == ext]
            out[arm][f"{ext:g}"] = {f"{T:g}": round(min(r["tripod"] for r in sub if r["tilt_deg"] <= T + 0.05), 4)
                                    for T in TILT_BOUNDS if any(r["tilt_deg"] <= T + 0.05 for r in sub)}
    return out


def cmd_matrix(args) -> dict:
    t0 = time.time()
    models, rest, q = _models()
    rows, heading = [], []
    for arm in models:
        for ext in MATRIX_EXT:
            for pitch in MATRIX_PITCH:
                for roll in MATRIX_ROLL:
                    rows.append(matrix_case(models, rest, ext, arm, q, roll, pitch))
            if not args.grid_only:
                for tilt in HEADING_TILTS:
                    for dh in HEADING_DIRS:
                        heading.append(matrix_case(models, rest, ext, arm, q, tilt=tilt, dh=dh))
    every = rows + heading
    # cross-slope table by extension, worst of PARK / T4, rack downhill vs uphill, pitch 0 and worst pitch
    cross = {}
    for ext in MATRIX_EXT:
        c = {}
        for roll in (0.0, 2.0, 3.0, 5.0):
            for side, sgn in (("down", 1.0), ("up", -1.0)):
                sub = [r for r in rows if r["ext"] == ext and r["roll_deg"] == sgn * roll]
                c[f"{roll:g}_{side}"] = {"pitch0": round(min(r["tripod"] for r in sub if r["pitch_deg"] == 0), 4),
                                         "any_pitch": round(min(r["tripod"] for r in sub), 4),
                                         "hexa_min": round(min(r["hexa"] for r in sub), 4)}
        cross[f"{ext:g}"] = c
    # predicted (worst of the measured cross-slope and its mirror) against the measured stance: never optimistic
    pred_err = [r["predicted"] - r["tripod"] for r in every]
    # ... and against the worse of the measured row and its measured mirror row (roll -> -roll, same arm / ext /
    # pitch): the residual is the settle asymmetry the body-frame mirror does not model
    by_key = {(r["arm"], r["ext"], r["roll_deg"], r["pitch_deg"]): r for r in rows}
    mirror_err: Dict[str, List[float]] = {}
    for r in rows:
        o = by_key.get((r["arm"], r["ext"], -r["roll_deg"], r["pitch_deg"]))
        if o is None:
            continue
        band = "pitch_le_5" if abs(r["pitch_deg"]) <= 5.0 else "pitch_10"
        mirror_err.setdefault(f"{r['arm']}_{band}", []).append(r["predicted"] - min(r["tripod"], o["tripod"]))
    out = {"schema": "nandi.crown_gates/1", "owner": "WP6 (tools/crown_gate_matrix.py)", "meta": run_meta("matrix"),
           "layout": {"robot": "sim/models/AtharvNandi/atharv_nandi.xml (WP1)", "head_kg": HEAD_CAP_KG,
                      "basket_kg": 3.0, "rack": "T2/T4/T5 masses under rack_cradle_0..2 (right rail)",
                      "arms": {"PARK": "home keyframe", "T4": "worst T4 pose (WP0c), T4 mounted + 1.8 kg fruit"},
                      "mass_kg": {k: round(float(mm.body_subtreemass[1]), 3) for k, mm in models.items()}},
           "convention": "roll > 0 = rack (right) side downhill; pitch > 0 nose downhill, < 0 rear / mast downhill; "
                         "downhill_deg = fall line in the robot frame (0 nose, 90 left, 180 rear, 270 rack)",
           "envelope": envelope(every), "cross_table": cross,
           "min_hexa": round(min(r["hexa"] for r in every), 4), "any_fell": any(r["fell"] for r in every),
           "unstable_tripod_rows": sum(1 for r in every if r["tripod"] < 0),
           "flat_crown_up_min_tripod": min(r["tripod"] for r in rows if r["ext"] > 0 and r["roll_deg"] == 0
                                           and r["pitch_deg"] == 0),
           "predicted_vs_measured_m": {"n": len(pred_err), "min": round(min(pred_err), 4),
                                       "max": round(max(pred_err), 4)},
           "predicted_vs_mirror_m": {k: {"n": len(v), "min": round(min(v), 4), "max": round(max(v), 4)}
                                     for k, v in sorted(mirror_err.items())},
           "wall_s": round(time.time() - t0, 1), "rows": rows, "heading_rows": heading}
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "heading_rows")}, indent=1))
    if args.write:
        _merge_write({"matrix": out})
    return out


def _merge_write(block: dict):
    doc = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as fh:
            doc = json.load(fh)
    doc.update({"schema": "nandi.crown_gates/1", "classification": "Public",
                "owner": "WP6 (tools/crown_gate_matrix.py)"})
    doc.update(block)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        fh.write(_dumps(doc) + "\n")


def _dumps(obj, indent: int = 0) -> str:
    """Indented JSON whose row lists (rows, heading_rows, cases) keep one compact row per line."""
    pad = " " * indent
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            if k in ("rows", "heading_rows") and isinstance(v, list):
                inner = ",\n".join(pad + "  " + json.dumps(r, separators=(",", ":")) for r in v)
                items.append(f"{pad} {json.dumps(k)}: [\n{inner}\n{pad} ]")
            else:
                items.append(f"{pad} {json.dumps(k)}: {_dumps(v, indent + 1).lstrip()}")
        return pad + "{\n" + ",\n".join(items) + "\n" + pad + "}" if items else pad + "{}"
    if isinstance(obj, list) and any(isinstance(x, (dict, list)) for x in obj):
        return pad + "[\n" + ",\n".join(_dumps(x, indent + 1) for x in obj) + "\n" + pad + "]"
    return pad + json.dumps(obj)


# ------------------------------------------------------------------------------------------------ sit-down
BELLY = ("belly_rest_fl", "belly_rest_fr", "belly_rest_rl", "belly_rest_rr")


def support_margin(m, d) -> dict:
    """CoM margin (along gravity) to the hull of every robot-ground contact point, and which geoms touch."""
    gh = fs.unit(m.opt.gravity)
    x = np.array([1.0, 0, 0])
    u = fs.unit(x - np.dot(x, gh) * gh)
    v = fs.unit(np.cross(-gh, u))
    P = lambda p: (float(np.dot(p, u)), float(np.dot(p, v)))  # noqa: E731
    pts, names = [], set()
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            if m.body_rootid[m.geom_bodyid[g]] == 1 and m.geom_bodyid[g] != 0:
                other = c.geom2 if g == c.geom1 else c.geom1
                if m.geom_bodyid[other] == 0:
                    pts.append(P(c.pos))
                    names.add(mujoco.mj_id2name(m, O.mjOBJ_GEOM, g) or str(g))
    hull = st.convex_hull(pts)
    return {"margin_m": round(st.com_margin(P(d.subtree_com[1]), hull), 4) if len(hull) >= 3 else None,
            "contacts": sorted(names), "belly_rests_down": sorted(n for n in names if n in BELLY)}


def sitdown_case(roll: float, ext: float = 1.6, arm: str = "PARK", kp_scale: float = SITDOWN_KP_SCALE,
                 seconds: float = 3.0) -> dict:
    models, rest, q = _models()
    m = models[arm]
    set_ext(m, ext, rest[arm])
    fs.set_slope(m, roll, 0.0)
    d = _start(m, q if arm == "T4" else None)
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    z0 = float(d.qpos[2])
    before = fs.margins(m, d)
    # e-stop: crown braked (body_pos frozen at ext), legs de-powered -> sit down on the belly rests
    hold = d.qpos[[int(m.jnt_qposadr[m.actuator_trnid[a, 0]]) for a in range(25)]].copy()
    d.ctrl[:25] = hold
    kp0 = m.actuator_gainprm[:, 0].copy()
    bp1 = m.actuator_biasprm[:, 1].copy()
    for a in range(7, 25):
        m.actuator_gainprm[a, 0] = kp0[a] * kp_scale
        m.actuator_biasprm[a, 1] = bp1[a] * kp_scale
    max_tilt, fell = 0.0, False
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        R = d.xmat[1].reshape(3, 3)
        up = -fs.unit(m.opt.gravity)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(R[:, 2], up))))))
        max_tilt = max(max_tilt, tilt)
        if tilt > 35.0:
            fell = True
    m.actuator_gainprm[:, 0] = kp0
    m.actuator_biasprm[:, 1] = bp1
    sm = support_margin(m, d)
    return {"roll_deg": roll, "rack": "downhill" if roll > 0 else "uphill", "ext": ext, "arm": arm,
            "trunk_z_before_m": round(z0, 4), "trunk_z_after_m": round(float(d.qpos[2]), 4),
            "tripod_before_m": before["tripod"], "hexa_before_m": before["hexa"],
            "max_tilt_rel_gravity_deg": round(max_tilt, 2), "fell": fell, **sm}


def cmd_sitdown(args) -> dict:
    t0 = time.time()
    cases = [dict(sitdown_case(r, kp_scale=k, seconds=5.0), kp_scale=k) for r in (5.0, -5.0) for k in (0.0, SITDOWN_KP_SCALE)]
    out = {"meta": run_meta("sitdown"), "kp_scales": [0.0, SITDOWN_KP_SCALE], "cases": cases,
           "belly_rests_touch": any(c["belly_rests_down"] for c in cases),
           "no_fall": not any(c["fell"] for c in cases),
           "min_belly_polygon_margin_m": min((c["margin_m"] for c in cases if c["margin_m"] is not None), default=None),
           "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    if args.write:
        _merge_write({"sitdown": out})
    return out


# ------------------------------------------------------------------------------------------------ wind tipping
def tripod_hold_targets(m) -> np.ndarray:
    """Leg targets with tripod B at the top of its swing (A in stance), from the real gait generator."""
    g = TripodGaitGenerator(0.02)
    g.set_command(1.0, 0.0, 0.0, hold=True)
    best, best_lift = None, -1.0
    for _ in range(200):
        t = g.step(0.0, 0.0)
        # B = FL, MR, RL; leg order FR FL MR ML RR RL, 3 joints each; lift is on the calf (index 2)
        lift = sum(abs(t[3 * i + 2] - home) for i, home in zip((1, 2, 5), [m.key_ctrl[0][7 + 3 * i + 2]
                                                                         for i in (1, 2, 5)]))
        if lift > best_lift and g.amp >= 1.0:
            best, best_lift = list(t), lift
    return np.asarray(best)


def tipwind_case(m, rest, legs, dir_deg: float, v: float, seconds: float = 3.0) -> dict:
    set_ext(m, 1.6, rest)
    fs.set_slope(m, 0.0, 0.0)
    d = _start(m)
    d.ctrl[7:25] = legs
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drag = wm.WindDrag(m)
    script = wm.WindScript(speed_mps=v, dir_deg=dir_deg)
    stance = [_id(m, O.mjOBJ_GEOM, f) for f in ("FR_foot", "ML_foot", "RR_foot")]
    z_feet0 = [float(d.geom_xpos[g][2]) for g in stance]
    lifted, max_tilt = False, 0.0
    for k in range(int(seconds / m.opt.timestep)):
        if k % 10 == 0:
            drag.apply(m, d, script, float(d.time))
        mujoco.mj_step(m, d)
        R = d.xmat[1].reshape(3, 3)
        max_tilt = max(max_tilt, math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))))
        if any(float(d.geom_xpos[g][2]) - z0 > 0.02 for g, z0 in zip(stance, z_feet0)):
            lifted = True
    return {"v": v, "tipped": lifted or max_tilt > 10.0, "foot_lift": lifted, "max_tilt_deg": round(max_tilt, 2)}


def static_tip_speed(m, rest, legs, dir_deg: float, v_ref: float = 10.0) -> dict:
    """Quasi-static: the wind moment at v_ref shifts the effective CoM by M_h / (m g); v_tip where it equals the
    tripod margin toward the downwind edge."""
    set_ext(m, 1.6, rest)
    d = _start(m)
    d.ctrl[7:25] = legs
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drag = wm.WindDrag(m)
    drag.apply(m, d, wm.WindScript(speed_mps=v_ref, dir_deg=dir_deg), 0.0)
    feet = [d.geom_xpos[_id(m, O.mjOBJ_GEOM, f)][:2].copy() for f in ("FR_foot", "ML_foot", "RR_foot")]
    hull = st.convex_hull(feet)
    mass = float(m.body_subtreemass[1])
    z_ground = float(np.mean([d.geom_xpos[_id(m, O.mjOBJ_GEOM, f)][2] for f in ("FR_foot", "ML_foot", "RR_foot")]))
    fx = fy = mom_x = mom_y = 0.0
    for gid, f in drag.last_forces.items():
        h = float(d.geom_xpos[gid][2]) - z_ground
        fx += f[0]
        fy += f[1]
        mom_x += f[0] * h
        mom_y += f[1] * h
    drag.clear(d)
    com = d.subtree_com[1][:2]
    shift = np.array([mom_x, mom_y]) / (mass * 9.81)
    m0 = st.com_margin(com, hull)
    hi = 1.0
    while st.com_margin(com + hi * shift, hull) > 0 and hi < 1e4:   # margin(com + s shift) = 0, s = (v / v_ref)^2
        hi *= 2.0
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if st.com_margin(com + mid * shift, hull) > 0:
            lo = mid
        else:
            hi = mid
    s_tip = 0.5 * (lo + hi)
    return {"dir_deg": dir_deg, "force_n_at_ref": round(math.hypot(fx, fy), 3), "v_ref": v_ref,
            "tripod_margin_m": round(m0, 4), "v_tip_static_mps": round(v_ref * math.sqrt(s_tip), 2)}


def cmd_tipwind(args) -> dict:
    t0 = time.time()
    m = build_full_layout()
    rest = set_ext(m, 0.0)
    legs = tripod_hold_targets(m)
    rows = []
    for dir_deg in (0.0, 90.0, 180.0, 270.0):
        s = static_tip_speed(m, rest, legs, dir_deg)
        lo, hi = 4.0, 40.0          # dynamic: step wind (no ramp), bisection to 0.25 m/s
        while hi - lo > 0.25:
            mid = 0.5 * (lo + hi)
            if tipwind_case(m, rest, legs, dir_deg, mid)["tipped"]:
                hi = mid
            else:
                lo = mid
        rows.append({**s, "v_tip_dynamic_step_mps": round(lo, 2),
                     "check_0p9": tipwind_case(m, rest, legs, dir_deg, round(0.9 * lo, 2))})
    worst = min(rows, key=lambda r: min(r["v_tip_static_mps"], r["v_tip_dynamic_step_mps"]))
    out = {"meta": run_meta("tipwind"), "stance": "tripod A (FR, ML, RR) down, B at mid-swing top, held",
           "ext": 1.6, "layout_mass_kg": round(float(m.body_subtreemass[1]), 3), "rows": rows,
           "tripod_tip_speed_min_mps": min(worst["v_tip_static_mps"], worst["v_tip_dynamic_step_mps"]),
           "worst_dir_deg": worst["dir_deg"], "tipped_rule": "a stance foot lifts > 2 cm or tilt > 10 deg within 3 s",
           "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(out, indent=1))
    if args.write:
        _merge_write({"tipwind": out})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("matrix", "sitdown", "tipwind"))
    ap.add_argument("--write", action="store_true", help="merge the result into config/crown_gates.json")
    ap.add_argument("--grid-only", action="store_true", help="matrix: skip the heading sweep")
    args = ap.parse_args(argv)
    {"matrix": cmd_matrix, "sitdown": cmd_sitdown, "tipwind": cmd_tipwind}[args.cmd](args)


if __name__ == "__main__":
    main()
