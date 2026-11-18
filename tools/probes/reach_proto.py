"""Prototype: joint-free telescoping crown mast + (comparison) lifted arm pedestal on Atharv Nandi.
Static stability (CoM projection along gravity, CoP from contact forces), slope via rotated gravity,
wind via xfrc_applied on the trunk, and a short tripod walk. Prints JSON lines.
"""
import json, math, sys
import numpy as np
import mujoco
from atharv_nandi.core.tripod_gait import TripodGaitGenerator

BASE = open("atharv_nandi/sim/models/AtharvNandi/atharv_nandi.xml").read()
# flat ground: drop the test ridges and the loose crop
for tag in ['<geom name="ridge_1"', '<geom name="ridge_2"']:
    i = BASE.index(tag); j = BASE.index("/>", i) + 2; BASE = BASE[:i] + BASE[j:]
i = BASE.index('<body name="crop_item_1"'); j = BASE.index("</body>", i) + 7; BASE = BASE[:i] + BASE[j:]
BASE = BASE.replace(' 0.62 0.0 0.035 1 0 0 0"', '"', 1)

RING_Z = 0.59      # trunk-frame z of the ring (lens ~0.98 m world at home), as MVP-SPEC 2.3
STROKE = 1.60      # m of crown travel
MAST_X = -0.36

def mast_xml(stage_kg=0.5, head_kg=0.6):
    return f"""
      <body name="mast_base" pos="{MAST_X} 0 0.05">
        <geom name="mast_drive" type="box" pos="0 0 0.03" size="0.04 0.04 0.03" mass="0.40" rgba="0.3 0.3 0.3 1"/>
        <geom name="mast_outer" type="capsule" fromto="0 0 0.06 0 0 {RING_Z}" size="0.02" mass="0.35" rgba="0.5 0.5 0.5 1"/>
        <geom name="mast_ring" type="cylinder" pos="0 0 {RING_Z}" size="0.06 0.04" mass="0.50" rgba="0.1 0.1 0.1 1"/>
        <body name="mast_stage" pos="0 0 0">
          <geom name="mast_inner" type="capsule" fromto="0 0 {RING_Z + 0.10 - STROKE} 0 0 {RING_Z + 0.06}" size="0.015" mass="{stage_kg}" rgba="0.7 0.7 0.7 1" contype="0" conaffinity="0"/>
          <geom name="crown_head" type="box" pos="0 0 {RING_Z + 0.12}" size="0.06 0.06 0.06" mass="{head_kg}" rgba="0.9 0.2 0.2 1"/>
          <camera name="crown_cam" pos="0.06 0 {RING_Z + 0.12}" xyaxes="0 -1 0 0 0 1" fovy="60"/>
        </body>
      </body>
"""

def build(variant="mast", lift=0.0, tip_kg=0.0, head_kg=0.6):
    x = BASE
    if variant in ("mast", "combo"):
        k = x.index("    </body>\n\n  </worldbody>") if "    </body>\n\n  </worldbody>" in x else x.rindex("    </body>\n")
        x = x[:k] + mast_xml(head_kg=head_kg) + x[k:]
    if variant in ("lift", "combo") and lift > 0:
        # fixed pedestal: raise the arm carriage (Phase-D stand-in; a real lift is a joint in the arm chain)
        x = x.replace('<body name="arm_carriage" pos="0.05 0 0.155">',
                      f'<geom name="lift_column" type="box" pos="0.05 0 {0.155 + lift/2}" size="0.04 0.04 {lift/2}" mass="1.5" rgba="0.4 0.4 0.4 1" contype="0" conaffinity="0"/>'
                      f'<body name="arm_carriage" pos="0.05 0 {0.155 + lift}">')
    if tip_kg > 0:
        x = x.replace('<site name="gripper_tip"',
                      f'<geom name="payload" type="sphere" pos="0.105 0 0" size="0.05" mass="{tip_kg}" contype="0" conaffinity="0" rgba="1 0.6 0 1"/><site name="gripper_tip"')
    return mujoco.MjModel.from_xml_string(x)

FEET = ["FR_foot", "FL_foot", "MR_foot", "ML_foot", "RR_foot", "RL_foot"]
TRI = {"A": ["FR_foot", "ML_foot", "RR_foot"], "B": ["FL_foot", "MR_foot", "RL_foot"]}

def hull(pts):
    pts = sorted(map(tuple, pts))
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]

def margin(h, p):
    """signed distance from p to the CCW hull boundary (+ inside)."""
    d = 1e9
    for i in range(len(h)):
        a, b = np.array(h[i]), np.array(h[(i+1) % len(h)])
        e = b - a; n = np.array([-e[1], e[0]]) / np.linalg.norm(e)  # inward normal for CCW
        d = min(d, float(np.dot(p - a, n)))
    return d

def basis(g):
    gh = g / np.linalg.norm(g)
    x = np.array([1.0, 0, 0]); u = x - np.dot(x, gh) * gh; u /= np.linalg.norm(u)
    v = np.cross(-gh, u); v /= np.linalg.norm(v)
    return gh, u, v

def proj(p, u, v): return np.array([np.dot(p, u), np.dot(p, v)])

def set_slope(m, deg, axis):
    a = math.radians(deg); g = 9.81
    if axis == "roll":   # cross-slope, downhill = -y (right side)
        m.opt.gravity[:] = [0, -g * math.sin(a), -g * math.cos(a)]
    else:                # pitch, downhill = +x (nose down)
        m.opt.gravity[:] = [g * math.sin(a), 0, -g * math.cos(a)]

def wind_elements(m, d, ext):
    """(world point, area m2, Cd) list; areas lateral (assumptions, see notes)."""
    tz = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")]
    el = [(tz + [0, 0, -0.17], 0.10, 1.2), (tz + [0, 0, 0.03], 0.116, 1.5)]
    try:
        mb = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mast_base")]
        el += [(mb + [0, 0, RING_Z/2], 0.04 * RING_Z, 1.2), (mb + [0, 0, RING_Z], 0.0096, 1.5)]
        if ext > 0.01: el.append((mb + [0, 0, RING_Z + ext/2], 0.03 * ext, 1.2))
        el.append((mb + [0, 0, RING_Z + 0.12 + ext], 0.0144, 1.5))
    except Exception:
        pass
    return el

def apply_wind(m, d, ext, v, dirn):
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    d.xfrc_applied[:] = 0
    if v <= 0: return 0.0, 0.0
    q = 0.5 * 1.2 * v * v
    F = np.zeros(3); T = np.zeros(3)
    for p, A, cd in wind_elements(m, d, ext):
        f = q * cd * A * dirn; F += f; T += np.cross(p - d.xipos[tid], f)
    d.xfrc_applied[tid, :3] = F; d.xfrc_applied[tid, 3:] = T
    return float(np.linalg.norm(F)), float(np.linalg.norm(T))

def cop(m, d, u, v):
    tot = 0.0; acc = np.zeros(2); f6 = np.zeros(6)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for i in range(d.ncon):
        c = d.contact[i]
        if floor not in (c.geom1, c.geom2): continue
        mujoco.mj_contactForce(m, d, i, f6)
        fn = abs(f6[0]); tot += fn; acc += fn * proj(c.pos, u, v)
    return (acc / tot if tot > 0 else None), tot

def set_arm(m, d, ctrl_arm):
    d.ctrl[0:7] = ctrl_arm

def run(variant="mast", ext=0.0, slope=0.0, axis="roll", wind=0.0, lift=0.0, tip_kg=0.0,
        arm=None, head_kg=0.6, walk=0.0, secs=2.5, ramp_ext=None):
    m = build(variant, lift, tip_kg, head_kg)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mast_stage")
    if sid >= 0: m.body_pos[sid, 2] = ext
    if arm is not None:
        d.ctrl[0:5] = arm; d.qpos[7:12] = arm
    set_slope(m, slope, axis)
    gh, u, v = basis(m.opt.gravity.copy())
    dirn = gh * 0  # wind ⟂ gravity, along the downhill direction (lateral for roll, fwd for pitch)
    dirn = np.array([0, -1.0, 0]) if axis == "roll" else np.array([1.0, 0, 0])
    dirn = dirn - np.dot(dirn, gh) * gh; dirn /= np.linalg.norm(dirn)
    gait = TripodGaitGenerator(0.02) if walk > 0 else None
    if gait: gait.set_command(walk, 0, 0, hold=True)
    n = int(secs / m.opt.timestep); res = {"min_cop": 1e9, "min_com": 1e9, "max_tilt": 0.0, "fell": False}
    ftot = 0
    for k in range(n):
        if ramp_ext is not None and sid >= 0:
            m.body_pos[sid, 2] = min(ramp_ext[1], ramp_ext[0] * k * m.opt.timestep)
            ext = m.body_pos[sid, 2]
        if gait and k % 10 == 0:
            qw, qx, qy, qz = d.qpos[3:7]
            roll = math.atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy)); pitch = math.asin(max(-1, min(1, 2*(qw*qy-qz*qx))))
            d.ctrl[7:25] = gait.step(roll, pitch)
        F, T = apply_wind(m, d, ext, wind, dirn)
        mujoco.mj_step(m, d)
        up = d.xmat[1].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(max(-1, min(1, float(np.dot(up, -gh))))))
        res["max_tilt"] = max(res["max_tilt"], tilt)
        if tilt > 35 or d.qpos[2] < 0.12: res["fell"] = True
        if k > n // 2:
            feet = [d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f)] for f in FEET]
            h = hull([proj(p, u, v) for p in feet])
            com = d.subtree_com[1]
            cp, ftot = cop(m, d, u, v)
            if not gait:
                res["min_com"] = min(res["min_com"], margin(h, proj(com, u, v)))
            if cp is not None: res["min_cop"] = min(res["min_cop"], margin(hull(_contact_pts(m, d, u, v)) if len(_contact_pts(m, d, u, v)) >= 3 else h, cp))
    # final static numbers
    feet = {f: d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f)].copy() for f in FEET}
    com = d.subtree_com[1].copy()
    hpts = [proj(p, u, v) for p in feet.values()]
    h = hull(hpts)
    # CoM height above the support plane, along -g
    foot_low = np.mean([np.dot(p, gh) for p in feet.values()]) + 0.03
    hc = foot_low - float(np.dot(com, gh))
    out = dict(variant=variant, ext=round(ext, 3), crown_lens_z=round(0.39 + RING_Z + 0.12 + ext, 3) if sid >= 0 else None,
               lift=lift, tip_kg=tip_kg, head_kg=head_kg, slope=slope, axis=axis, wind=wind, walk=walk,
               mass=round(float(m.body_subtreemass[1]), 2), com_h=round(hc, 3),
               com_xy_trunk=[round(float(x), 3) for x in (com - d.xpos[1])[:2]],
               hexa_com_margin=round(margin(h, proj(com, u, v)), 3),
               tri_A=round(margin(hull([proj(feet[f], u, v) for f in TRI["A"]]), proj(com, u, v)), 3),
               tri_B=round(margin(hull([proj(feet[f], u, v) for f in TRI["B"]]), proj(com, u, v)), 3),
               min_cop_margin=round(res["min_cop"], 3), min_com_margin_run=round(res["min_com"], 3) if res["min_com"] < 1e8 else None,
               max_tilt=round(res["max_tilt"], 2), fell=res["fell"], wind_F=round(F, 2), wind_T=round(T, 2),
               trunk_disp=[round(float(x), 3) for x in d.qpos[0:2]])
    if slope == 0:
        # static tip-over angles about the lateral and rear edges (flat ground geometry)
        lat = min(abs(p[1]) for p in hpts) - abs(out["com_xy_trunk"][1])
        out["tip_deg_lateral"] = round(math.degrees(math.atan2(lat, hc)), 1)
    return out

def _contact_pts(m, d, u, v):
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return [proj(d.contact[i].pos, u, v) for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2)]

def emit(**kw):
    try:
        print(json.dumps(run(**kw)), flush=True)
    except Exception as e:
        print(json.dumps({"err": repr(e), **{k: str(v) for k, v in kw.items()}}), flush=True)

def main():
    global MAST_X
    which = sys.argv[1] if len(sys.argv) > 1 else "static"
    if which == "static":
        emit(variant="none")
        for e in (0.0, 0.8, 1.6):
            for s, ax in ((0, "roll"), (10, "roll"), (15, "roll"), (10, "pitch"), (15, "pitch")):
                emit(variant="mast", ext=e, slope=s, axis=ax)
    elif which == "wind":
        for e in (0.0, 1.6):
            for w in (8, 12, 15, 20):
                emit(variant="mast", ext=e, wind=w)
            emit(variant="mast", ext=e, wind=12, slope=10)
    elif which == "lift":
        # arm sideways (yaw +pi/2), upper arm vertical, forearm horizontal: lateral worst case
        arm = [0.0, 1.5708, 0.0, 0.0, 0.0]
        for L in (0.0, 0.4, 0.8):
            for kg in (0.5, 1.5):
                for s in (0, 10):
                    emit(variant="combo", ext=1.6 if L == 0 else 0.0, lift=L, tip_kg=kg, arm=arm, slope=s)
    elif which == "walk":
        for e in (0.0, 0.8, 1.6):
            for s in (0, 10):
                emit(variant="mast", ext=e, slope=s, walk=0.6, secs=6.0)
    elif which == "sens":
        for mx in (-0.36, -0.30):
            MAST_X = mx
            for hk in (0.3, 0.6, 1.0):
                r = run(variant="mast", ext=1.6, head_kg=hk); r2 = run(variant="mast", ext=1.6, head_kg=hk, slope=10)
                print(json.dumps({"mast_x": mx, "head_kg": hk, "flat_tri": min(r["tri_A"], r["tri_B"]), "flat_hexa": r["hexa_com_margin"], "roll10_tri": min(r2["tri_A"], r2["tri_B"]), "com_h": r["com_h"], "tip_deg": r.get("tip_deg_lateral")}), flush=True)
    elif which == "ramp":
        emit(variant="mast", ext=0.0, slope=10, ramp_ext=(0.1, 1.6), secs=18.0)

if __name__ == '__main__':
    main()
