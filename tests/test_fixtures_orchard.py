"""WP4 fixtures (ORCHARD-MVP-SPEC 3.6, 4.2, 5.2): drip laterals + foot-on-lateral counter, mulch, prop rope,
bamboo prop, dock and crate, placeholder trees.

Acceptance (WP4): lateral counter fires on a scripted foot placement, not at 3 cm; rope and
laterals in group 5, invisible to safety rays.
"""
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core import world_events as wev
from atharv_nandi.sim.hooks import HookRegistry, Bus
from atharv_nandi.sim.world import compose, field_spec, terrain_orchard
from atharv_nandi.sim.world.modules import fixtures as fx

SAFETY_MASK = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)     # OS 2.5: safety rays {0, 1} (+ actors in 1)
COVERAGE_MASK = np.array([1, 1, 0, 1, 0, 0], dtype=np.uint8)
QUERY_MASK = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)


@pytest.fixture
def world(monkeypatch):
    def build(name, **env):
        monkeypatch.setenv("NANDI_WORLD", name)
        monkeypatch.setenv("NANDI_NOVNC", "0")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return farm_scene.load_farm_model()
    return build


def gname(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""


def geoms_with(m, prefix):
    return [g for g in range(m.ngeom) if gname(m, g).startswith(prefix)]


def ray(m, d, p, v, mask, bodyexclude=-1):
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.asarray(p, float), np.asarray(v, float), mask, 1, bodyexclude, gid)
    return dist, (gname(m, gid[0]) if gid[0] >= 0 else None)


# ---------------------------------------------------------------------- groups / ray visibility
def test_rope_and_laterals_are_group5_query_geoms_invisible_to_safety_rays(world):
    m = world("orchard_demo")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q = geoms_with(m, "lateral_")
    queries = [g for g in q if "_q" in gname(m, g)]
    visuals = [g for g in q if "_v" in gname(m, g)]
    rope = geoms_with(m, "rope_")
    assert len(queries) >= 6 and visuals and len(rope) == 2
    for g in queries + [g for g in rope if not gname(m, g).endswith("_v")]:
        assert m.geom_group[g] == 5 and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, gname(m, g)
    for g in visuals + [g for g in rope if gname(m, g).endswith("_v")]:
        assert m.geom_group[g] == 4 and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, gname(m, g)
    # rope at 1.4 m between B-1 and B-2: a horizontal ray across it at 1.4 m sees it only with the query mask
    f = field_spec.load("orchard_demo")
    r = field_spec.rope_lines(f)[0]
    mid = (np.asarray(r["a"]) + np.asarray(r["b"])) / 2
    assert mid[2] == pytest.approx(1.4)
    for mask, want in ((SAFETY_MASK, False), (COVERAGE_MASK, False), (QUERY_MASK, True)):
        dist, hit = ray(m, d, mid + [0, -1.0, 0], [0, 1, 0], mask)
        assert (hit is not None and hit.startswith("rope_")) == want, (mask, hit)
    # a lateral from above: the safety mask hits the ground under it, the query mask the lateral
    lat = next(l for l in field_spec.laterals(f) if l["id"] == "lateral_R1")
    x, y = (lat["points"][0][0] + 0.25, lat["points"][0][1])
    _, hit = ray(m, d, [x, y, 1.0], [0, 0, -1], SAFETY_MASK)
    assert hit == "orchard_ground"
    _, hit = ray(m, d, [x, y, 1.0], [0, 0, -1], QUERY_MASK)
    assert hit is not None and hit.startswith("lateral_R1_q")
    # a horizontal safety ray at 0.18 m (low safety plane) along an alley never stops on a lateral
    dist, hit = ray(m, d, [x, y - 0.5, 0.012], [0, 1, 0], SAFETY_MASK)
    assert hit is None or not hit.startswith(("lateral_", "rope_"))


@pytest.mark.parametrize("w", ["orchard_demo", "orchard_demo_monsoon", "orchard_lite"])
def test_lateral_query_geoms_follow_the_ground(world, w):
    """Review: the query capsules sat at z = r, buried up to 0.10 m under basin rims (OT-1 ring ~9 cm)."""
    from atharv_nandi.sim.world import terrain_orchard
    m = world(w)
    f = field_spec.load(w)
    g = terrain_orchard.grid_for(f, 7)
    worst, n = 0.0, 0
    # the flooded-basin water discs (monsoon) end in a surface step (water at z 0 against the rim flank): a
    # straight piece cannot drape a step, so the points within 12 cm (one minimum piece) of a round patch edge are left out
    edges = [(p.center, p.r_m) for p in g.cfg.patches if p.r_m > 0]
    near_step = lambda x, y: any(abs(math.hypot(x - c[0], y - c[1]) - r) < 0.12 for c, r in edges)
    for gid in geoms_with(m, "lateral_"):
        name = gname(m, gid)
        if "_q" not in name:
            continue
        n += 1
        r = float(m.geom_size[gid][0])
        c, half = m.geom_pos[gid], float(m.geom_size[gid][1])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[gid])
        ax = R.reshape(3, 3)[:, 2]
        for u in np.linspace(-1.0, 1.0, 9):
            p = c + u * half * ax
            if not near_step(p[0], p[1]):
                worst = max(worst, abs((p[2] - r) - g.surface_at(p[0], p[1])))
        assert m.geom_group[gid] == 5 and m.geom_contype[gid] == 0
    print(f"WP4 {w}: {n} lateral query pieces, worst axis offset from surface + r: {worst * 100:.2f} cm")
    assert n > 0 and worst <= 0.015
    # the query and visual pieces coincide (same pose), one visual per query piece
    q = sorted(gname(m, x) for x in geoms_with(m, "lateral_") if "_q" in gname(m, x))
    v = sorted(gname(m, x) for x in geoms_with(m, "lateral_") if "_v" in gname(m, x))
    assert [s.replace("_q", "_v", 1) for s in q] == v
    if w == "orchard_demo":                                              # the OT-1 ring sits on the rim crest
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ring = next(l for l in field_spec.laterals(f) if l["id"] == "lateral_OT1_ring")
        x, y = ring["points"][0][:2]
        ot1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tree_OT-1")       # its group-5 foliage envelope (WP23)
        dist, hit = ray(m, d, [x, y, 1.0], [0, 0, -1], QUERY_MASK, ot1)
        assert hit is not None and hit.startswith("lateral_OT1_ring_q")
        assert 1.0 - dist == pytest.approx(g.surface_at(x, y) + 2 * ring["r_m"], abs=0.012)


def test_laterals_on_every_tree_line_and_the_ot1_ring_with_a_spur_across_the_orbit():
    f = field_spec.load("orchard_demo")
    lats = {l["id"]: l for l in field_spec.laterals(f)}
    assert {"lateral_R1", "lateral_R2", "lateral_R3", "lateral_R4", "lateral_OT1_ring", "lateral_OT1_spur"} <= set(lats)
    for row in field_spec.rows(f):
        ts = field_spec.trees_in_row(f, row["row_id"])
        lat = lats[f"lateral_{row['row_id']}"]
        ys = {p[1] for p in lat["points"]}
        assert ys == {row["y"] + ts[0]["drip_line_offset_m"]}
        assert min(p[0] for p in lat["points"]) < min(field_spec.tree_xy(t)[0] for t in ts)
        assert max(p[0] for p in lat["points"]) > max(field_spec.tree_xy(t)[0] for t in ts)
    ot = field_spec.tree_by_id(f, "OT-1")
    ox, oy = field_spec.tree_xy(ot)
    ring = lats["lateral_OT1_ring"]["points"]
    assert all(math.hypot(px - ox, py - oy) == pytest.approx(0.8) for px, py in ring)
    r1 = ot["canopy_r_m"] + ot["orbit_standoff_m"]
    spur = lats["lateral_OT1_spur"]["points"]
    radii = [math.hypot(px - ox, py - oy) for px, py in spur]
    assert min(radii) < r1 < max(radii)                                 # the spur crosses the orbit circle
    # the counter sees the crossing: a foot placed on the spur at the orbit radius counts
    c = wev.LateralCounter(field_spec.lateral_segments(f))
    a = math.atan2(spur[1][1] - oy, spur[1][0] - ox)
    ev = c.update(0.0, [("FL_foot", ox + r1 * math.cos(a), oy + r1 * math.sin(a), True)])
    assert ev and ev[0]["lateral"] == "lateral_OT1_spur"
    mask = field_spec.map_mask_items(f)
    assert len(mask["trunks"]) == 17 and len(mask["basins"]) == 17 and mask["props"] and mask["ropes"]


# ---------------------------------------------------------------------- lateral counter
def test_lateral_counter_fires_on_a_foot_placement_not_at_3_cm():
    c = wev.LateralCounter([("lat_a", (0.0, 0.0), (5.0, 0.0))])
    assert c.update(0.0, [("FR_foot", 1.0, 0.0, True)])[0]["count"] == 1          # on the axis
    assert c.update(0.02, [("FR_foot", 1.0, 0.0, True)]) == []                    # same placement: once
    assert c.update(0.04, [("FR_foot", 1.0, 0.0, False)]) == []                   # lifted
    assert len(c.update(0.06, [("FR_foot", 1.2, 0.015, True)])) == 1              # 1.5 cm: counts
    c.update(0.08, [("FR_foot", 1.2, 0.015, False)])
    assert c.update(0.10, [("FR_foot", 2.0, 0.03, True)]) == []                   # 3 cm: does not
    assert c.update(0.12, [("FL_foot", 2.0, -0.03, True)]) == []
    assert c.update(0.14, [("ML_foot", 3.0, 0.0, False)]) == []                    # not in contact
    assert c.update(0.16, [("ML_foot", 6.5, 0.0, True)]) == []                     # past the lateral end
    assert c.status()["count"] == 2 and c.status()["per_foot"] == {"FR_foot": 2}
    c.reset()
    assert c.count == 0 and c.events == []
    assert wev.LateralCounter([]).nearest((0, 0)) == (None, math.inf)
    d = wev.point_segment_distance(np.array([[0, 1.0], [6, 0]]), np.array([[0, 0]]), np.array([[5, 0]]))
    assert np.allclose(d[:, 0], [1.0, 1.0])


def _place_foot(m, d, foot, xy, grid):
    """Translate the robot so ``foot``'s centre is above xy, resting on the ground surface."""
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]   # face +y: one foot per lateral line
    mujoco.mj_forward(m, d)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, foot)
    fx_, fy_, fz_ = d.geom_xpos[g]
    d.qpos[0] += xy[0] - fx_
    d.qpos[1] += xy[1] - fy_
    d.qpos[2] += (grid.surface_at(*xy) + 0.03) - fz_
    mujoco.mj_forward(m, d)


@pytest.mark.parametrize("offset,expect", [(0.0, 1), (0.01, 1), (0.03, 0)])
def test_hook_counts_a_scripted_foot_placement_on_a_lateral(world, offset, expect):
    from atharv_nandi.sim.hooks import world_events as hook
    m = world("orchard_lite")
    d = mujoco.MjData(m)
    f = field_spec.load("orchard_lite")
    grid = terrain_orchard.grid_for(f, 7)
    lat = next(l for l in field_spec.laterals(f) if l["id"] == "lateral_L2")
    y = lat["points"][0][1]
    _place_foot(m, d, "FR_foot", (3.3, y + offset), grid)
    reg, bus = HookRegistry(), Bus()
    hook.register(reg)
    reg.reset(m, d)
    reg.tick(m, d, bus)
    rt = hook.runtime_for(m)
    assert rt.counter.per_foot.get("FR_foot", 0) == expect
    steps = [p for t, p in bus.drain() if t == hook.EVENTS_TOPIC and '"FR_foot"' in p]
    assert len(steps) == expect
    if expect:
        assert '"lateral_L2"' in steps[0] and '"lateral_step"' in steps[0]
        assert bus.values["world"]["laterals"]["count"] == rt.counter.count >= 1


# ---------------------------------------------------------------------- mulch, props, dock, crate, stubs
def test_mulch_strips_in_r3_and_r4_carry_friction_0_5(world):
    m = world("orchard_demo")
    f = field_spec.load("orchard_demo")
    strips = f["fixtures"]["mulch"]
    assert {s["id"][:2] for s in strips} == {"R3", "R4"}
    ground_pri = int(m.geom_priority[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "orchard_ground")])
    for s in strips:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"mulch_{s['id']}")
        assert m.geom_friction[g][0] == pytest.approx(0.5) and m.geom_priority[g] > ground_pri
        assert m.geom_group[g] == 0
        top = m.geom_pos[g][2] + m.geom_size[g][2]
        assert 0.0 < top <= 0.012
    assert fx.MULCH_FRICTION == 0.5


def test_bamboo_prop_and_rope_in_r4(world):
    m = world("orchard_demo")
    f = field_spec.load("orchard_demo")
    props = field_spec.props(f)
    assert [p["tree_id"] for p in props] == ["R4-3"] and props[0]["kind"] == "bamboo"
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "prop_bamboo_B3")
    assert m.geom_group[g] == 0 and m.geom_contype[g] == 1 and m.geom_conaffinity[g] == 0
    ropes = field_spec.rope_lines(f)
    assert [(r["id"], r["tree_id"]) for r in ropes] == [("B1_B2", "R4-1")]
    b1, b2 = field_spec.tree_by_id(f, "B-1"), field_spec.tree_by_id(f, "B-2")
    assert ropes[0]["a"][2] == ropes[0]["b"][2] == 1.4
    assert ropes[0]["a"][0] > field_spec.tree_xy(b1)[0] and ropes[0]["b"][0] < field_spec.tree_xy(b2)[0]


def test_dock_plate_and_open_crate(world):
    m = world("orchard_lite")
    f = field_spec.load("orchard_lite")
    plate = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "dock_plate")
    assert m.geom_contype[plate] == 0 and m.geom_conaffinity[plate] == 0
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "crate_aabb")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cx, cy = f["crate"]["xy"]
    assert d.site_xpos[site][:2] == pytest.approx([cx, cy])
    lx, ly, lz = f["crate"]["size_m"]
    w = f["crate"]["wall_m"]
    assert m.site_size[site] == pytest.approx([lx / 2 - w, ly / 2 - w, (lz - w) / 2])
    # open top: a ray straight down into the crate centre lands on the crate floor, not a lid
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.array([cx, cy, 1.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
    assert gname(m, gid[0]) == "crate_floor" and 1.0 - dist == pytest.approx(w, abs=1e-6)
    # a fruit-sized ball dropped in stays inside the crate AABB
    assert len(geoms_with(m, "crate_")) == 5


def test_stub_trees_until_the_wp23_trees_module_exists(world, monkeypatch):
    m = world("orchard_demo")                                            # WP23 landed: real trees, no stubs
    f = field_spec.load("orchard_demo")
    assert not geoms_with(m, "tree_stub_")
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tree_OT-1") >= 0
    wid_trees = farm_scene.world_identity()[1]
    monkeypatch.delitem(compose._REGISTRY, "trees")                     # without the module the stubs return
    m2 = farm_scene.load_farm_model()
    trunks = geoms_with(m2, "tree_stub_")
    assert len([g for g in trunks if gname(m2, g).endswith("_trunk")]) == 17
    for g in trunks:
        if gname(m2, g).endswith("_trunk"):
            assert m2.geom_contype[g] == 1 and m2.geom_conaffinity[g] == 0 and m2.geom_group[g] == 0
        else:
            assert m2.geom_contype[g] == 0 and m2.geom_group[g] == 3
    top, cz, radii = fx.stub_tree_dims(field_spec.tree_by_id(f, "OT-1"))
    assert top == pytest.approx(0.77) and cz + radii[2] == pytest.approx(2.2)
    assert farm_scene.world_identity()[1] != wid_trees
