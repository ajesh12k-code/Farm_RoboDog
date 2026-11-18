#!/usr/bin/env python3
"""treesim feasibility prototype: procedural orchard trees (mango/guava/papaya/banana) added to
the live Nandi farm spec via MjSpec, joint-free trees, fruit hung by connect/weld equalities.
Measures compile, bodies/geoms/nv/neq, contacts, ms/step, detach, and render ms/pass.
Scratch only. Usage: python3 treesim_proto.py [phys|render|detach|all]
"""
import json, math, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NANDI_TERRAIN", "flat")
import numpy as np
import mujoco
from atharv_nandi.core import farm_scene

G = mujoco.mjtGeom
VIS_LEAF, VIS_PEST = 3, 4          # render groups: 3 foliage, 4 pest/disease markers

def rot_z(a):
    return [math.cos(a / 2), 0, 0, math.sin(a / 2)]

def quat_from_to(v):
    v = np.asarray(v, float); v /= np.linalg.norm(v)
    z = np.array([0, 0, 1.0]); c = float(np.dot(z, v))
    if c > 0.9999: return [1, 0, 0, 0]
    ax = np.cross(z, v); s = np.linalg.norm(ax); ax /= s
    ang = math.atan2(s, c)
    return [math.cos(ang / 2), *(ax * math.sin(ang / 2))]

def wood(body, p0, p1, r, rgba=(0.45, 0.3, 0.18, 1)):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    body.add_geom(type=G.mjGEOM_CAPSULE, fromto=[*p0, *p1], size=[r, 0, 0], rgba=list(rgba),
                  contype=1, conaffinity=0, group=1)

ACC = None   # when a list, foliage is merged into one mesh per tree (one draw call)

def _quat_mat(q):
    m = np.zeros(9); mujoco.mju_quat2Mat(m, np.asarray(q, float)); return m.reshape(3, 3)

def foliage(body, pos, size, rgba=(0.15, 0.45, 0.15, 1), quat=(1, 0, 0, 0), kind=G.mjGEOM_ELLIPSOID):
    if ACC is not None:
        R = _quat_mat(quat); c = np.asarray(pos, float); hs = np.asarray(size, float)
        if kind == G.mjGEOM_ELLIPSOID:   # flattened octahedron, 6 verts 8 faces
            V = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float) * hs
            F = [[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]]
        else:                            # box, 8 verts 12 faces
            V = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)], float) * hs
            F = [[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]
        base = len(ACC[0]) // 3
        ACC[0].extend((V @ R.T + c).ravel().tolist()); ACC[1].extend((np.asarray(F) + base).ravel().tolist())
        return
    body.add_geom(type=kind, pos=list(pos), size=list(size), quat=list(quat), rgba=list(rgba),
                  contype=0, conaffinity=0, group=VIS_LEAF, mass=0)

def pest(body, pos, rng, n=6, rgba=(0.95, 0.95, 0.9, 1)):
    for _ in range(n):
        body.add_geom(type=G.mjGEOM_SPHERE, pos=list(np.asarray(pos) + rng.normal(0, 0.01, 3)),
                      size=[0.003, 0, 0], rgba=list(rgba), contype=0, conaffinity=0,
                      group=VIS_PEST, mass=0)

class Builder:
    def __init__(self, spec, opts):
        self.spec, self.o = spec, opts
        self.n_dyn, self.n_static_fruit, self.eq = 0, 0, []

    def fruit(self, tree, name, anchor_w, centre_w, size, mass, rgba, mode, kind=G.mjGEOM_ELLIPSOID):
        """mode connect|weld|static. anchor_w = pedicel point (world)."""
        if mode == "static" or self.n_dyn_tree >= self.o["dyn_per_tree"]:
            tree.add_geom(type=kind, pos=list(np.asarray(centre_w) - self.tree_pos), size=list(size),
                          rgba=list(rgba), contype=0, conaffinity=0, group=1, mass=0)
            self.n_static_fruit += 1
            return
        b = self.spec.worldbody.add_body(name=name, pos=list(centre_w))
        b.add_freejoint(name=name + "_fj")
        b.add_geom(type=kind, size=list(size), mass=mass, rgba=list(rgba), contype=1, conaffinity=0,
                   friction=[1.0, 0.02, 0.002], group=1)
        b.add_site(name=name + "_ped", pos=list(np.asarray(anchor_w) - np.asarray(centre_w)), size=[0.004, 0, 0])
        e = self.spec.add_equality(name="attach_" + name, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1=name, name2=tree.name)
        rel = list(np.asarray(anchor_w) - np.asarray(centre_w))
        if mode == "connect":
            e.type = mujoco.mjtEq.mjEQ_CONNECT
            e.data[0:3] = rel
        else:
            e.type = mujoco.mjtEq.mjEQ_WELD
            e.data[0:3] = list(np.asarray(anchor_w) - self.tree_pos)   # weld anchor is in body2 (tree) frame
            e.data[10] = 1.0  # torquescale
        e.solref = [0.006, 1]
        self.eq.append((name, mode))
        self.n_dyn += 1
        self.n_dyn_tree += 1

    def tree(self, species, pos, seed, height):
        rng = np.random.default_rng(seed)
        pos = np.asarray([pos[0], pos[1], 0.0]); self.tree_pos = pos
        t = self.spec.worldbody.add_body(name=f"tree_{species}_{seed}", pos=list(pos))
        self.n_dyn_tree = 0
        o = self.o
        global ACC
        ACC = ([], []) if o.get("foliage_mesh") else None
        getattr(self, "_" + species)(t, pos, rng, height, o)
        if ACC is not None and ACC[0]:
            mn = f"canopy_{self.seed_tag}"
            self.spec.add_mesh(name=mn, uservert=ACC[0], userface=ACC[1], inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            t.add_geom(type=G.mjGEOM_MESH, meshname=mn, rgba=[0.15, 0.45, 0.15, 1], contype=0, conaffinity=0,
                       group=VIS_LEAF, mass=0)
        ACC = None
        return t

    def _canopy_tree(self, t, pos, rng, H, o, fruit_size, fruit_mass, fruit_rgb, mode, n_fruit, sp):
        trunk_h = 0.35 * H
        wood(t, [0, 0, 0], [0, 0, trunk_h], 0.05 + 0.01 * H)
        crown_r = 0.45 * H                       # canopy radius ~ 0.45 H (assumption)
        n_sc = int(rng.integers(3, 5))
        tips = []
        for i in range(n_sc):
            a = 2 * math.pi * i / n_sc + rng.uniform(-0.3, 0.3)
            el = rng.uniform(0.6, 0.9)
            L = 0.55 * H
            p1 = np.array([math.cos(a) * L * math.cos(el), math.sin(a) * L * math.cos(el), trunk_h + L * math.sin(el)])
            wood(t, [0, 0, trunk_h], p1, 0.03)
            for j in range(2):
                a2 = a + rng.uniform(-0.7, 0.7)
                p2 = p1 + 0.3 * H * np.array([math.cos(a2) * 0.7, math.sin(a2) * 0.7, rng.uniform(0.1, 0.6)])
                wood(t, p1, p2, 0.015)
                tips.append(p2)
        # foliage clusters on a canopy ellipsoid shell
        cz = trunk_h + 0.4 * H
        for k in range(o["leaf_clusters"]):
            th = rng.uniform(0, 2 * math.pi); ph = rng.uniform(-0.3, 1.3)
            rr = crown_r * rng.uniform(0.7, 1.0)
            p = [rr * math.cos(th) * math.cos(ph), rr * math.sin(th) * math.cos(ph), cz + 0.6 * crown_r * math.sin(ph)]
            foliage(t, p, [0.16, 0.16, 0.05], quat=rot_z(th))
        # individual leaf cards in the reachable band 0.4-1.4 m (LEAF SCAN targets), with markers
        for k in range(o["leaf_cards"]):
            th = rng.uniform(0, 2 * math.pi)
            p = [crown_r * 0.9 * math.cos(th), crown_r * 0.9 * math.sin(th), rng.uniform(0.5, 1.4)]
            foliage(t, p, [0.06, 0.025, 0.002], quat=rot_z(th), kind=G.mjGEOM_BOX, rgba=(0.2, 0.5, 0.2, 1))
            if k < o["pest_patches"]:
                # disease spot (top) and insect cluster (underside)
                t.add_geom(type=G.mjGEOM_CYLINDER, pos=[p[0], p[1], p[2] + 0.0025], size=[0.012, 0.0005, 0],
                           rgba=[0.35, 0.2, 0.05, 1], contype=0, conaffinity=0, group=VIS_PEST, mass=0)
                pest(t, [p[0], p[1], p[2] - 0.006], rng, n=o["insects_per_patch"])
        # fruit: hang below scaffold tips and on outer canopy
        for k in range(n_fruit):
            th = rng.uniform(0, 2 * math.pi)
            r = crown_r * rng.uniform(0.6, 0.95)
            z = rng.uniform(0.5, min(1.6, cz + 0.3))
            anchor = pos + np.array([r * math.cos(th), r * math.sin(th), z])
            centre = anchor - np.array([0, 0, fruit_size[2] + 0.02])
            ripe = rng.random() < 0.4
            rgba = fruit_rgb[1] if ripe else fruit_rgb[0]
            self.fruit(t, f"{sp}{self.seed_tag}_{k}", anchor, centre, fruit_size, fruit_mass, rgba,
                       mode if ripe else "static")

    def _mango(self, t, pos, rng, H, o):
        self._canopy_tree(t, pos, rng, H, o, [0.04, 0.035, 0.06], 0.3,
                          [(0.3, 0.6, 0.2, 1), (0.95, 0.75, 0.1, 1)], "connect", o["fruit_mango"], "m")

    def _guava(self, t, pos, rng, H, o):
        self._canopy_tree(t, pos, rng, H, o, [0.035, 0.035, 0.04], 0.15,
                          [(0.4, 0.7, 0.3, 1), (0.85, 0.9, 0.4, 1)], "weld", o["fruit_guava"], "g")

    def _papaya(self, t, pos, rng, H, o):
        wood(t, [0, 0, 0], [0, 0, H], 0.08, rgba=(0.55, 0.55, 0.45, 1))
        n_leaf = 14
        for k in range(n_leaf):
            a = 2 * math.pi * k / n_leaf * 2.4
            d = np.array([math.cos(a), math.sin(a), rng.uniform(0.2, 0.8)]); d /= np.linalg.norm(d)
            p0 = np.array([0, 0, H - 0.05]); p1 = p0 + 0.6 * d
            t.add_geom(type=G.mjGEOM_CAPSULE, fromto=[*p0, *p1], size=[0.008, 0, 0], rgba=[0.5, 0.7, 0.3, 1],
                       contype=0, conaffinity=0, group=VIS_LEAF, mass=0)
            foliage(t, p1 + 0.15 * d, [0.25, 0.25, 0.01], quat=rot_z(a))
        # fruit column on the trunk below the crown
        for k in range(o["fruit_papaya"]):
            a = rng.uniform(0, 2 * math.pi); z = H - 0.15 - 0.06 * k
            anchor = pos + np.array([0.08 * math.cos(a), 0.08 * math.sin(a), z])
            centre = anchor + np.array([0.09 * math.cos(a), 0.09 * math.sin(a), -0.05])
            ripe = k < 3
            self.fruit(t, f"p{self.seed_tag}_{k}", anchor, centre, [0.07, 0.07, 0.11], 1.0,
                       (0.95, 0.6, 0.1, 1) if ripe else (0.3, 0.6, 0.2, 1), "weld" if ripe else "static")
        if o["pest_patches"]:
            pest(t, [0.1, 0, H - 0.4], rng, n=o["insects_per_patch"] * 2)   # mealybug colony on trunk

    def _banana(self, t, pos, rng, H, o):
        t.add_geom(type=G.mjGEOM_CYLINDER, pos=[0, 0, H / 2], size=[0.11, H / 2, 0], rgba=[0.45, 0.6, 0.3, 1],
                   contype=1, conaffinity=0, group=1)
        for k in range(9):
            a = 2 * math.pi * k / 9 * 1.6 + rng.uniform(-0.2, 0.2)
            d = np.array([math.cos(a), math.sin(a), rng.uniform(0.0, 0.9)]); d /= np.linalg.norm(d)
            c = np.array([0, 0, H]) + 0.8 * d
            q = quat_from_to(d)
            t.add_geom(type=G.mjGEOM_BOX, pos=list(c), size=[0.22, 0.004, 0.75], quat=q,
                       rgba=[0.2, 0.55, 0.2, 1] if k > 1 else [0.55, 0.45, 0.2, 1],   # 2 senescent (deleaf)
                       contype=0, conaffinity=0, group=VIS_LEAF, mass=0)
        # bunch (static, assist/mark): peduncle + 6 hands of fingers
        t.add_geom(type=G.mjGEOM_CAPSULE, fromto=[0.12, 0, H - 0.1, 0.25, 0, H - 0.9], size=[0.025, 0, 0],
                   rgba=[0.4, 0.5, 0.2, 1], contype=0, conaffinity=0, group=1, mass=0)
        for h in range(6):
            z = H - 0.25 - 0.1 * h
            t.add_geom(type=G.mjGEOM_ELLIPSOID, pos=[0.2, 0, z], size=[0.14, 0.14, 0.05], rgba=[0.4, 0.65, 0.2, 1],
                       contype=1, conaffinity=0, group=1)
        t.add_site(name=f"bunch_cut_{self.seed_tag}", pos=[0.13, 0, H - 0.12], size=[0.01, 0, 0])
        for s in range(2):                                     # suckers (desucker targets)
            a = rng.uniform(0, 2 * math.pi)
            t.add_geom(type=G.mjGEOM_CAPSULE, fromto=[0.3 * math.cos(a), 0.3 * math.sin(a), 0,
                                                       0.3 * math.cos(a), 0.3 * math.sin(a), rng.uniform(0.3, 0.8)],
                       size=[0.04, 0, 0], rgba=[0.4, 0.6, 0.3, 1], contype=1, conaffinity=0, group=1)
        if o["pest_patches"]:
            pest(t, [0, 0.11, 0.5], rng, n=o["insects_per_patch"])

def build(layout, opts):
    spec = farm_scene._build_spec()
    b = Builder(spec, opts)
    for i, (sp, x, y, H) in enumerate(layout):
        b.seed_tag = i
        b.tree(sp, (x, y), 1000 + i, H)
    t0 = time.perf_counter(); m = spec.compile(); tc = time.perf_counter() - t0
    return m, b, tc

def orchard(species, rows, per_row, row_sp, tree_sp, H, x0=1.2):
    out = []
    for r in range(rows):
        y = (r - (rows - 1) / 2) * row_sp
        if rows == 1: y = row_sp / 2
        for k in range(per_row):
            out.append((species, x0 + k * tree_sp, y, H))
    return out

DEF = dict(dyn_per_tree=4, leaf_clusters=40, leaf_cards=30, pest_patches=4, insects_per_patch=6,
           fruit_mango=20, fruit_guava=25, fruit_papaya=12)

def reset(m, d):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[74:] = m.qpos0[74:]      # keyframe covers robot + legacy crops only (nq 74)

def bench(m, steps=1500, drop_all=False, push=0.0):
    d = mujoco.MjData(m)
    reset(m, d)
    if drop_all: d.eq_active[:] = 0
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    # keyframe only covers robot+legacy crops; tree fruit free joints start at qpos0 (their body pos)
    for _ in range(300): mujoco.mj_step(m, d)   # settle
    ncon, nefc = [], []
    t0 = time.perf_counter()
    for _ in range(steps):
        if push: d.xfrc_applied[tb, 0] = push
        mujoco.mj_step(m, d); ncon.append(d.ncon); nefc.append(d.nefc)
    dt = (time.perf_counter() - t0) / steps
    tree_b = {i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("tree_")}
    wood_con = sum(1 for i in range(d.ncon) if m.geom_bodyid[d.contact[i].geom1] in tree_b or m.geom_bodyid[d.contact[i].geom2] in tree_b)
    return dict(ms_step=round(dt * 1e3, 4), rtf_physics_only=round(m.opt.timestep / dt, 2),
                ncon_mean=round(float(np.mean(ncon)), 1), ncon_max=int(max(ncon)),
                nefc_mean=round(float(np.mean(nefc)), 1), trunk_z=round(float(d.qpos[2]), 3), trunk_x=round(float(d.qpos[0]), 3),
                badqacc=int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number), tree_contacts_end=wood_con)

def stats(m, b, tc):
    return dict(nbody=m.nbody, ngeom=m.ngeom, nv=m.nv, nq=m.nq, neq=m.neq,
                dyn_fruit=b.n_dyn, static_fruit=b.n_static_fruit, compile_s=round(tc, 3),
                geom_vis_only=int(np.sum((m.geom_contype == 0) & (m.geom_conaffinity == 0))))

def phys():
    res = {}
    H = dict(mango=2.4, guava=2.2, papaya=2.6, banana=2.6)
    sp = dict(mango=(2.5, 2.5), guava=(3.0, 1.5), papaya=(1.8, 1.8), banana=(1.8, 1.8))  # row, tree (assumption)
    m, b, tc = build([], DEF); res["robot_only"] = {**stats(m, b, tc), **bench(m)}
    mix = [("mango", 1.4, 1.0, 2.4), ("guava", 3.2, 1.0, 2.2), ("papaya", 1.4, -1.0, 2.6), ("banana", 3.2, -1.0, 2.6)]
    m, b, tc = build(mix, DEF); res["mixed_4"] = {**stats(m, b, tc), **bench(m)}
    for s in ("mango", "guava", "papaya", "banana"):
        lay = orchard(s, 2, 8, *sp[s], H[s])
        m, b, tc = build(lay, DEF); res[f"{s}_2x8"] = {**stats(m, b, tc), **bench(m)}
    lay = orchard("mango", 4, 8, 2.5, 2.5, 2.4)
    m, b, tc = build(lay, DEF); res["mango_4x8"] = {**stats(m, b, tc), **bench(m)}
    m, b, tc = build(lay, {**DEF, "dyn_per_tree": 0}); res["mango_4x8_nodyn"] = {**stats(m, b, tc), **bench(m)}
    m, b, tc = build(lay, {**DEF, "dyn_per_tree": 99}); res["mango_4x8_alldyn_ripe"] = {**stats(m, b, tc), **bench(m, 800)}
    m, b, tc = build(lay, {**DEF, "leaf_clusters": 150, "leaf_cards": 80, "pest_patches": 20})
    res["mango_4x8_denseleaf"] = {**stats(m, b, tc), **bench(m)}
    m, b, tc = build(lay, DEF); res["mango_4x8_all_dropped"] = {**stats(m, b, tc), **bench(m, drop_all=True)}
    # worst case: robot shoved (40 N) into a mango trunk/scaffolds at x 0.75
    m, b, tc = build([("mango", 0.60, 0.0, 2.4)], DEF); res["robot_pushed_into_tree"] = {**stats(m, b, tc), **bench(m, push=150.0)}
    m, b, tc = build([("banana", 0.62, 0.0, 2.6)], DEF); res["robot_pushed_into_banana"] = {**stats(m, b, tc), **bench(m, push=150.0)}
    return res

def detach():
    """Ramp a pull on one connect (mango) and a twist on one weld (guava); release on threshold."""
    m, b, _ = build([("mango", 1.4, 1.0, 2.4), ("guava", 3.2, 1.0, 2.2)], {**DEF, "dyn_per_tree": 1})
    d = mujoco.MjData(m); reset(m, d)
    out = {}
    for name, mode in b.eq:
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "attach_" + name)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        thr = 12.0 if mode == "connect" else 0.15   # N for pull, N*m for twist (assumptions)
        reset(m, d)
        for _ in range(250): mujoco.mj_step(m, d)
        t_rel, peak = None, 0.0
        for i in range(3000):
            ramp = i * m.opt.timestep
            d.xfrc_applied[bid] = 0
            if mode == "connect":
                d.xfrc_applied[bid, :3] = [0, 0, -8.0 * ramp]        # pull down, 8 N/s
            else:
                d.xfrc_applied[bid, 3:] = [0, 0, 0.1 * ramp]         # twist about z, 0.1 N*m/s
            mujoco.mj_step(m, d)
            rows = np.where((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == eid))[0]
            f = d.efc_force[rows]
            val = float(np.linalg.norm(f[:3])) if mode == "connect" else float(np.linalg.norm(f[3:6]))
            peak = max(peak, val)
            if d.eq_active[eid] and val > thr:
                d.eq_active[eid] = 0; t_rel = round(ramp, 3)
                break
        for _ in range(500): mujoco.mj_step(m, d)
        out[name] = dict(mode=mode, rows=len(rows), thr=thr, released_at_s=t_rel, peak=round(peak, 3),
                         fruit_z_after=round(float(d.xpos[bid, 2]), 3))
    return out

def render():
    lay = orchard("mango", 4, 8, 2.5, 2.5, 2.4)
    res = {}
    for label, layout, o in (("1_tree", [("mango", 1.4, 1.25, 2.4)], DEF), ("mango_4x8", lay, DEF),
                             ("mango_4x8_denseleaf", lay, {**DEF, "leaf_clusters": 150, "leaf_cards": 80, "pest_patches": 20})):
        m, b, _ = build(layout, o)
        d = mujoco.MjData(m); reset(m, d); mujoco.mj_forward(m, d)
        for (w, h) in ((160, 120), (320, 240), (640, 480)) if label != 'mango_4x8_denseleaf' else ((320, 240),):
            r = mujoco.Renderer(m, h, w, max_geom=20000)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [1.4, 1.25, 1.0]; cam.distance = 2.0; cam.elevation = -10
            opt = mujoco.MjvOption()
            for grp_on in (True, False):
                opt.geomgroup[VIS_LEAF] = grp_on; opt.geomgroup[VIS_PEST] = grp_on
                ts = []
                for k in range(24):                      # 24 frames = 15 deg orbit steps
                    cam.azimuth = k * 15
                    t0 = time.perf_counter(); r.update_scene(d, cam, opt); r.render(); ts.append(time.perf_counter() - t0)
                res[f"{label}_{w}x{h}_{'leaves' if grp_on else 'noleaves'}"] = dict(
                    ms_mean=round(1e3 * float(np.mean(ts[2:])), 1), ngeom_scene=r.scene.ngeom)
            # depth pass
            r.enable_depth_rendering(); ts = []
            for k in range(12):
                cam.azimuth = k * 30; t0 = time.perf_counter(); r.update_scene(d, cam); r.render(); ts.append(time.perf_counter() - t0)
            r.disable_depth_rendering()
            res[f"{label}_{w}x{h}_depth"] = dict(ms_mean=round(1e3 * float(np.mean(ts[2:])), 1))
            r.close()
    return res

def render2():
    """Render cost for a 360 orbit: primitives vs merged foliage mesh, shadows on/off, scene size."""
    lay = orchard("mango", 4, 8, 2.5, 2.5, 2.4)
    res = {}
    cases = [("1tree_prim", [("mango", 1.4, 1.25, 2.4)], DEF),
             ("1tree_mesh", [("mango", 1.4, 1.25, 2.4)], {**DEF, "foliage_mesh": True}),
             ("3tree_mesh", orchard("mango", 1, 3, 2.5, 2.5, 2.4), {**DEF, "foliage_mesh": True}),
             ("32tree_prim", lay, DEF),
             ("32tree_mesh", lay, {**DEF, "foliage_mesh": True}),
             ("32tree_mesh_dense", lay, {**DEF, "foliage_mesh": True, "leaf_clusters": 150, "leaf_cards": 80}),
             ("16banana_prim", orchard("banana", 2, 8, 1.8, 1.8, 2.6), DEF),
             ("16papaya_mesh", orchard("papaya", 2, 8, 1.8, 1.8, 2.6), {**DEF, "foliage_mesh": True})]
    for label, layout, o in cases:
        t0 = time.perf_counter(); m, b, _ = build(layout, o); tc = time.perf_counter() - t0
        d = mujoco.MjData(m); reset(m, d); mujoco.mj_forward(m, d)
        for (w, h) in ((160, 120), (320, 240), (640, 480)):
            r = mujoco.Renderer(m, h, w, max_geom=20000)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [layout[0][1], layout[0][2], 1.0]; cam.distance = 2.0; cam.elevation = -10
            opt = mujoco.MjvOption(); opt.geomgroup[VIS_LEAF] = 1; opt.geomgroup[VIS_PEST] = 1
            for shadow in ((0, 1) if w == 160 else (0,)):
                ts = []
                for k in range(14):
                    cam.azimuth = k * 26
                    t1 = time.perf_counter(); r.update_scene(d, cam, opt)
                    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
                    r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
                    r.render(); ts.append(time.perf_counter() - t1)
                res[f"{label}_{w}x{h}_sh{shadow}"] = dict(ms=round(1e3 * float(np.median(ts[2:])), 1),
                                                         scene_ngeom=r.scene.ngeom, model_ngeom=m.ngeom, build_s=round(tc, 3))
            r.close()
    return res

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "phys"
    out = dict(mujoco=mujoco.__version__, cpus=os.cpu_count())
    if what in ("phys", "all"): out["phys"] = phys()
    if what in ("detach", "all"): out["detach"] = detach()
    if what in ("render", "all"): out["render"] = render()
    if what == "render2": out["render2"] = render2()
    print(json.dumps(out, indent=1))
