"""WP4 compose modules ``fixtures`` (order 41) and ``dock`` (order 80). Orchard worlds only.

``fixtures`` (ORCHARD-MVP-SPEC 3.6, 4.2), all world-static, no joints:
    drip laterals   ``lateral_<id>_q<k>_<j>``: query capsules (group 5, contype 0: invisible to safety and
                    coverage rays, cameras and the contact solver) and their visual twins ``lateral_<id>_v<k>_<j>``
                    (group 4, contype 0), both on the ground surface, one piece per ``LATERAL_PIECE_M`` of
                    segment k
    plastic mulch   ``mulch_<id>``: 6 mm boxes (group 0) with friction 0.5 and priority 2 (the mulch friction
                    reaches the foot contact)
    prop ropes      ``rope_<id>`` query capsule (group 5, contype 0; listed in ``rope_lines[]`` for the crown
                    clearance check) + ``rope_<id>_v`` visual twin (group 4)
    props           ``prop_<id>`` bamboo poles: wood-like (group 0, contype 1 / conaffinity 0: the robot touches
                    them, other statics never pair)
    stub trees      only while no ``trees`` module (WP23) is registered: body ``tree_stub_<tree_id>`` with a wood
                    trunk capsule (group 0, contype 1 / conaffinity 0) and a canopy ellipsoid (group 3, visual), so
                    the orchard is drivable before WP23 lands. The fingerprint records it.
``dock`` (order 80): dock plate (visual, flush) + backboard; the crate as an open box (floor + 4 walls,
group 0, stiff contact) with a site ``crate_aabb`` whose size is the inner half extents (WP3 unload check).
"""
import math

import mujoco
import numpy as np

from atharv_nandi.sim.world import compose, field_spec, terrain_orchard

ORDER_FIXTURES = compose.ORDER["fixtures"] + 1     # 41: after legacy_terrain (40) has set its ground solref
ORDER_DOCK = compose.ORDER["dock"]                 # 80
VERSION = "1"
GRP_GROUND, GRP_FOLIAGE, GRP_DECAL, GRP_QUERY = 0, 3, 4, 5
LATERAL_PIECE_M = 0.5
LATERAL_Z_TOL_M = 0.01                             # lateral pieces follow the ground within 1 cm
LATERAL_RGBA = [0.05, 0.05, 0.05, 1.0]
ROPE_RGBA = [0.85, 0.80, 0.55, 1.0]
MULCH_RGBA = [0.12, 0.12, 0.13, 1.0]
BAMBOO_RGBA = [0.70, 0.62, 0.35, 1.0]
CRATE_RGBA = [0.15, 0.35, 0.75, 1.0]
CRATE_SOLREF = (0.005, 1.0)
MULCH_FRICTION = 0.5
MULCH_H = 0.003                                    # half height: 6 mm film on its bed
G = mujoco.mjtGeom


def _capsule(spec, name, a, b, r, group, rgba, contype=0, conaffinity=0, body=None):
    (body or spec.worldbody).add_geom(name=name, type=G.mjGEOM_CAPSULE, fromto=[*a, *b], size=[r, 0, 0],
                                      group=group, contype=contype, conaffinity=conaffinity, rgba=rgba)


def ground_pieces(grid, a, b, piece_m: float = LATERAL_PIECE_M, tol_m: float = LATERAL_Z_TOL_M,
                  min_m: float = 0.05):
    """Split segment a→b into straight pieces (at most ``piece_m`` long) whose chord stays within ``tol_m`` of
    the ground surface at the 1/4, 1/2 and 3/4 points (halved down to ``min_m`` over basin rims and slopes)."""
    a, b = np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    L = float(np.linalg.norm(b - a))
    n = max(1, int(math.ceil(L / piece_m)))
    todo = [(a + (b - a) * j / n, a + (b - a) * (j + 1) / n) for j in range(n)]
    out = []
    while todo:
        p, q = todo.pop(0)
        zp, zq = grid.surface_at(p[0], p[1]), grid.surface_at(q[0], q[1])
        off = max(abs(zp + f * (zq - zp) - grid.surface_at(*(p + f * (q - p)))) for f in (0.25, 0.5, 0.75))
        if off > tol_m and float(np.linalg.norm(q - p)) > 2 * min_m:
            m = (p + q) / 2
            todo[:0] = [(p, m), (m, q)]
        else:
            out.append((p, q))
    return out


def _laterals(spec, field, grid):
    for lat in field_spec.laterals(field):
        r = lat["r_m"]
        pts = lat["points"]
        for k in range(len(pts) - 1):
            a, b = np.asarray(pts[k], float), np.asarray(pts[k + 1], float)
            # query and visual pieces share the same ground-following axis (the query geom must sit where the
            # lateral is: a group-5 ray or geom-pose consumer reads it, OS 3.6 / WP21 / WP6)
            for j, (p, q) in enumerate(ground_pieces(grid, a, b)):
                zp = grid.surface_at(p[0], p[1]) + r
                zq = grid.surface_at(q[0], q[1]) + r
                _capsule(spec, f"{lat['id']}_q{k}_{j}", [p[0], p[1], zp], [q[0], q[1], zq], r, GRP_QUERY, LATERAL_RGBA)
                _capsule(spec, f"{lat['id']}_v{k}_{j}", [p[0], p[1], zp], [q[0], q[1], zq], r, GRP_DECAL, LATERAL_RGBA)


def _mulch(spec, field, grid):
    for m in (field.get("fixtures") or {}).get("mulch", []):
        x0, x1 = (float(v) for v in m["x"])
        y0, y1 = (float(v) for v in m["y"])
        xs = np.linspace(x0, x1, 9)
        ys = np.linspace(y0, y1, 5)
        base = max(grid.surface_at(x, y) for x in xs for y in ys)
        spec.worldbody.add_geom(name=f"mulch_{m['id']}", type=G.mjGEOM_BOX,
                                pos=[(x0 + x1) / 2, (y0 + y1) / 2, base + MULCH_H],
                                size=[(x1 - x0) / 2, (y1 - y0) / 2, MULCH_H],
                                group=GRP_GROUND, friction=[float(m.get("friction", MULCH_FRICTION)), 0.02, 0.01],
                                priority=2, solref=[0.01, 1.0], rgba=MULCH_RGBA)


def _ropes_and_props(spec, field):
    for rope in field_spec.rope_lines(field):
        _capsule(spec, f"rope_{rope['id']}", rope["a"], rope["b"], rope["r_m"], GRP_QUERY, ROPE_RGBA)
        _capsule(spec, f"rope_{rope['id']}_v", rope["a"], rope["b"], rope["r_m"], GRP_DECAL, ROPE_RGBA)
    for p in field_spec.props(field):
        _capsule(spec, f"prop_{p['id']}", p["a"], p["b"], p["r_m"], GRP_GROUND, BAMBOO_RGBA, contype=1, conaffinity=0)


def stub_tree_dims(t: dict):
    """(trunk top z, canopy centre z, canopy radii) of the placeholder tree (ASSUMED shapes, OS 3.3 table)."""
    H, cr = float(t["height_m"]), float(t["canopy_r_m"])
    crop = t["crop"]
    if crop in ("papaya", "banana"):
        trunk = float(t.get("trunk_h_m") or (0.80 if crop == "papaya" else 0.70) * H)
        trunk = min(trunk, H - 0.2)
        crown_h = max(H - trunk, 0.3)
        return trunk, trunk + crown_h / 2.0, (cr, cr, crown_h / 2.0 + 0.1)
    trunk = float(t.get("trunk_h_m") or 0.35 * H)
    crown_h = H - trunk
    return trunk, trunk + crown_h / 2.0, (cr, cr, crown_h / 2.0)


def _stub_trees(spec, field, grid):
    for t in field_spec.trees(field):
        x, y = field_spec.tree_xy(t)
        z0 = grid.surface_at(x, y)
        body = spec.worldbody.add_body(name=f"tree_stub_{t['tree_id']}", pos=[x, y, z0])
        top, cz, radii = stub_tree_dims(t)
        _capsule(spec, f"tree_stub_{t['tree_id']}_trunk", [0, 0, 0], [0, 0, top], float(t["trunk_r_m"]),
                 GRP_GROUND, [0.36, 0.25, 0.14, 1.0], contype=1, conaffinity=0, body=body)
        body.add_geom(name=f"tree_stub_{t['tree_id']}_canopy", type=G.mjGEOM_ELLIPSOID, pos=[0, 0, cz],
                      size=list(radii), group=GRP_FOLIAGE, contype=0, conaffinity=0, rgba=[0.20, 0.45, 0.18, 0.9])


def trees_module_present() -> bool:
    return any(m.name == "trees" for m in compose.modules())


def attach_fixtures(spec, field, ctx):
    if not field_spec.is_orchard(field):
        return
    grid = terrain_orchard.grid_for(field, ctx.seed)
    _laterals(spec, field, grid)
    _mulch(spec, field, grid)
    _ropes_and_props(spec, field)
    if not trees_module_present():
        _stub_trees(spec, field, grid)


def attach_dock(spec, field, ctx):
    if not field_spec.is_orchard(field):
        return
    grid = terrain_orchard.grid_for(field, ctx.seed)
    dk = field["dock"]
    dx, dy = (float(v) for v in dk["xy"])
    sx, sy = (float(v) for v in dk.get("size_m", [1.0, 0.8]))
    yaw = math.radians(float(dk.get("yaw_deg", 0.0)))
    quat = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    body = spec.worldbody.add_body(name="dock", pos=[dx, dy, grid.surface_at(dx, dy)], quat=quat)
    body.add_geom(name="dock_plate", type=G.mjGEOM_BOX, pos=[0, 0, 0.002], size=[sx / 2, sy / 2, 0.002],
                  group=GRP_GROUND, contype=0, conaffinity=0, rgba=[0.55, 0.55, 0.58, 1.0])
    body.add_geom(name="dock_backboard", type=G.mjGEOM_BOX, pos=[-sx / 2 - 0.45, 0, 0.35], size=[0.03, sy / 2, 0.35],
                  group=GRP_GROUND, rgba=[0.9, 0.6, 0.1, 1.0])
    body.add_site(name="dock_ref", pos=[0, 0, 0.004], size=[0.02, 0, 0], group=GRP_QUERY, rgba=[1, 1, 1, 0])

    cr = field["crate"]
    cx, cy = (float(v) for v in cr["xy"])
    lx, ly, lz = (float(v) for v in cr.get("size_m", [0.6, 0.4, 0.3]))
    w = float(cr.get("wall_m", 0.015))
    crate = spec.worldbody.add_body(name="crate", pos=[cx, cy, grid.surface_at(cx, cy)])
    kw = dict(type=G.mjGEOM_BOX, group=GRP_GROUND, solref=list(CRATE_SOLREF), rgba=CRATE_RGBA)
    crate.add_geom(name="crate_floor", pos=[0, 0, w / 2], size=[lx / 2, ly / 2, w / 2], **kw)
    crate.add_geom(name="crate_wall_px", pos=[lx / 2 - w / 2, 0, lz / 2], size=[w / 2, ly / 2, lz / 2], **kw)
    crate.add_geom(name="crate_wall_nx", pos=[-lx / 2 + w / 2, 0, lz / 2], size=[w / 2, ly / 2, lz / 2], **kw)
    crate.add_geom(name="crate_wall_py", pos=[0, ly / 2 - w / 2, lz / 2], size=[lx / 2 - w, w / 2, lz / 2], **kw)
    crate.add_geom(name="crate_wall_ny", pos=[0, -ly / 2 + w / 2, lz / 2], size=[lx / 2 - w, w / 2, lz / 2], **kw)
    crate.add_site(name="crate_aabb", pos=[0, 0, w + (lz - w) / 2], size=[lx / 2 - w, ly / 2 - w, (lz - w) / 2],
                   type=mujoco.mjtGeom.mjGEOM_BOX, group=GRP_QUERY, rgba=[1, 1, 1, 0])


def _fp_fixtures(field, ctx) -> str:
    if not field_spec.is_orchard(field):
        return ""
    import hashlib
    import json
    h = hashlib.sha1(json.dumps(field.get("fixtures"), sort_keys=True).encode()).hexdigest()[:10]
    return f"f={h},stub_trees={int(not trees_module_present())}"


def _fp_dock(field, ctx) -> str:
    if not field_spec.is_orchard(field):
        return ""
    return json_compact([field.get("dock"), field.get("crate")])


def json_compact(obj) -> str:
    import hashlib
    import json
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]


compose.register_module("fixtures", ORDER_FIXTURES, attach_fixtures, version=VERSION, fingerprint=_fp_fixtures)
compose.register_module("dock", ORDER_DOCK, attach_dock, version=VERSION, fingerprint=_fp_dock)
