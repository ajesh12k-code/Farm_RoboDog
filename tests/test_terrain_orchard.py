"""WP4 terrain and field registry (ORCHARD-MVP-SPEC 3.6, 4.1, 4.2; MISSION-PRESETS 0.1; OQ-02 B).

Acceptance (WP4): ``legacy`` identical; terrain within ±5 % of config; R1 alley ≤ 1°. Plus the
field files validate, the trees sit on the WP0c frozen candidates (NHB 5 x 2.5 m), the compiled hfield matches
the pure grid, the monsoon variant and the soft-soil sink (3-5 cm, assumption).
"""
import json
import math
import os

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene, model_layout as mlay
from atharv_nandi.core.sim_driver import SimDriver
from atharv_nandi.sim.world import compose, field_spec, terrain_orchard
from atharv_nandi.sim.world.modules import fruit_pool as pool_mod
from atharv_nandi.sim.world.modules import terrain as terrain_mod
from atharv_nandi.sim.world.modules import tools as tools_mod

WP4_MODULES = ("terrain", "fixtures", "dock", "actors")
FEET = ("FR_foot", "FL_foot", "MR_foot", "ML_foot", "RR_foot", "RL_foot")


@pytest.fixture
def world(monkeypatch):
    def build(name, **env):
        monkeypatch.setenv("NANDI_WORLD", name)
        monkeypatch.setenv("NANDI_NOVNC", "0")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return farm_scene.load_farm_model()
    return build


def model_bytes(m) -> bytes:
    buf = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return buf.tobytes()


def hfield_z(m, x, y):
    """Ground height of the compiled hfield at the nearest grid node (independent of the Python grid)."""
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, terrain_mod.GROUND_GEOM)
    hid = int(m.geom_dataid[gid])
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    sx, sy, sz, _ = m.hfield_size[hid]
    cx, cy, cz = m.geom_pos[gid]
    data = m.hfield_data[m.hfield_adr[hid]:m.hfield_adr[hid] + nrow * ncol].reshape(nrow, ncol)
    j = int(round((x - (cx - sx)) / (2 * sx) * (ncol - 1)))
    i = int(round((y - (cy - sy)) / (2 * sy) * (nrow - 1)))
    return float(cz + sz * data[i, j])


def hfield_grid(m):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, terrain_mod.GROUND_GEOM)
    hid = int(m.geom_dataid[gid])
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    sx, sy, sz, _ = m.hfield_size[hid]
    cx, cy, cz = m.geom_pos[gid]
    z = cz + sz * m.hfield_data[m.hfield_adr[hid]:m.hfield_adr[hid] + nrow * ncol].reshape(nrow, ncol)
    return z, (cx - sx, cy - sy), (2 * sx / (ncol - 1))


def max_slope_deg(m, rect, baseline=0.2):
    z, (x0, y0), res = hfield_grid(m)
    k = max(1, int(round(baseline / res)))
    xs = x0 + res * np.arange(z.shape[1])
    ys = y0 + res * np.arange(z.shape[0])
    sub = z[np.ix_((ys >= rect[2]) & (ys <= rect[3]), (xs >= rect[0]) & (xs <= rect[1]))]
    if sub.shape[0] <= k or sub.shape[1] <= k:
        return 0.0
    gx, gy = terrain_orchard.square_gradient(sub, k, res)
    return float(np.degrees(np.arctan(np.hypot(gx, gy).max())))


def max_slope_deg_corner(m, rect, baseline=0.2):
    """The pre-review corner stencil (x difference along the bottom edge, y along the left edge), reported."""
    z, (x0, y0), res = hfield_grid(m)
    k = max(1, int(round(baseline / res)))
    xs = x0 + res * np.arange(z.shape[1])
    ys = y0 + res * np.arange(z.shape[0])
    sub = z[np.ix_((ys >= rect[2]) & (ys <= rect[3]), (xs >= rect[0]) & (xs <= rect[1]))]
    gx = (sub[:, k:] - sub[:, :-k]) / (k * res)
    gy = (sub[k:, :] - sub[:-k, :]) / (k * res)
    return float(np.degrees(np.arctan(np.hypot(gx[:-k, :], gy[:, :-k]).max())))


# ---------------------------------------------------------------------- field files
def test_every_field_file_validates_and_orchard_trees_follow_mp01():
    worlds = list(field_spec.iter_worlds())
    assert worlds == ["legacy", "orchard_lite", "orchard_demo", "orchard_demo_monsoon"]
    for w in worlds:
        f = field_spec.load(w)
        assert field_spec.validate(f) == [], w
        assert f["world"] == w
    demo = field_spec.load("orchard_demo")
    s = field_spec.summary(demo)
    assert s["n_trees"] == 17 and s["by_crop"] == {"mango": 5, "guava": 4, "papaya": 4, "banana": 4}
    assert field_spec.summary(field_spec.load("orchard_lite"))["by_crop"] == {"mango": 3, "guava": 1, "papaya": 1,
                                                                               "banana": 1}
    for t in field_spec.trees(demo):
        for k in field_spec.TREE_REQUIRED:
            assert k in t, (t["tree_id"], k)
        assert 1.2 <= t["height_m"] <= 3.0
    assert field_spec.tree_by_id(demo, "B-2")["tree_id"] == "R4-2"      # physical tag lookup
    assert field_spec.tree_by_id(demo, "OT-1")["orbit_clear"] is True
    assert {field_spec.tree_by_id(demo, f"R4-{k}")["alley_clear_height_m"][0] for k in range(1, 5)} == {1.3}
    with pytest.raises(KeyError):
        field_spec.tree_by_id(demo, "nope")
    with pytest.raises(field_spec.FieldError):
        field_spec.load("no_such_world")


def test_demo_trees_sit_on_the_wp0c_frozen_candidates_with_nhb_spacing():
    cand = json.load(open(os.path.join(field_spec.FIELDS_DIR, "orchard_candidates.json")))
    demo = field_spec.load("orchard_demo")
    for c in cand["candidates"]["trees"]:
        t = field_spec.tree_by_id(demo, c["id"])
        assert field_spec.tree_xy(t) == (c["x"], c["y"])
        assert t["height_m"] == pytest.approx(c["H"]) and t["canopy_r_m"] == pytest.approx(c["canopy_r"], abs=1e-3)
        assert t["seed"] == c["seed"]
    r1 = [field_spec.tree_xy(t)[0] for t in field_spec.trees_in_row(demo, "R1")]
    assert np.allclose(np.diff(r1), 2.5)                             # NHB 5 x 2.5 m in-row (OQ-02 B)
    assert field_spec.check_spacing(demo) == []
    # the frozen spacing is enforced: R1 at 2.0 m in-row or a 4 m row pitch are refused
    bad = json.loads(json.dumps(demo))
    for t in bad["trees"]:
        if t["row_id"] == "R1":
            t["geo"]["x"] = 2.0 + 2.0 * (int(t["tree_id"][-1]) - 1)
    assert any("in-row" in e for e in field_spec.check_spacing(bad))
    bad2 = json.loads(json.dumps(demo))
    for r in bad2["rows"]:
        if r["row_id"] == "R2":
            r["y"] = 6.5
    assert any("pitch" in e for e in field_spec.check_spacing(bad2))
    # 5 x 3 m is the other frozen option
    alt = json.loads(json.dumps(demo))
    for t in alt["trees"]:
        if t["row_id"] == "R1":
            t["geo"]["x"] = 2.0 + 3.0 * (int(t["tree_id"][-1]) - 1)
    assert field_spec.check_spacing(alt) == []
    # free alleys after canopy (OS 4.2 claims ~2.3 / 2.0 / 1.6 m with the widened layout)
    assert field_spec.alley_free_width(demo, "R1", "R2") == pytest.approx(5.0 - 1.1 - 0.7)
    assert field_spec.alley_free_width(demo, "R3", "R4") == pytest.approx(3.5 - 0.9 - 1.4)


def test_validation_catches_schema_errors():
    demo = field_spec.load("orchard_demo")
    assert field_spec.validate({"schema": "x"})
    assert field_spec.validate([]) == ["field is not an object"]
    broken = json.loads(json.dumps(demo))
    broken["trees"][1]["physical_tag"] = broken["trees"][2]["physical_tag"]
    broken["trees"][3]["crop"] = "apple"
    broken["trees"][4]["basin_r_m"] = 0.3
    broken["trees"][5]["graft_union_h_m"] = None                     # grafted guava without a union height
    del broken["trees"][6]["drip_line_offset_m"]
    errs = field_spec.validate(broken)
    assert any("duplicate physical_tag" in e for e in errs)
    assert any("apple" in e for e in errs)
    assert any("basin_r_m" in e for e in errs)
    assert any("graft_union_h_m" in e for e in errs)
    assert any("drip_line_offset_m" in e for e in errs)
    leg = field_spec.load("legacy")
    assert field_spec.validate(dict(leg, trees=[{}]))                 # legacy must stay tree-free
    close = json.loads(json.dumps(demo))
    ot = next(t for t in close["trees"] if t["tree_id"] == "OT-1")
    ot["geo"]["y"] = 0.5                                               # annulus reaches R1 canopies
    assert any("orbit annulus" in e for e in field_spec.check_clearances(close))


# ---------------------------------------------------------------------- legacy identical
def test_legacy_world_compiles_byte_identical_with_and_without_wp4(world, monkeypatch):
    m_with = world("legacy")
    saved = {n: m for n, m in ((m.name, m) for m in compose.modules()) if n in WP4_MODULES}
    assert set(saved) == set(WP4_MODULES)
    try:
        for n in WP4_MODULES:
            compose.unregister_module(n)
        m_without = farm_scene.load_farm_model()
    finally:
        for n, mod in saved.items():
            compose.register_module(n, mod.order, mod.attach_fn, mod.version, mod.fingerprint)
    assert model_bytes(m_with) == model_bytes(m_without)
    assert mlay.Layout.from_model(m_with).is_legacy_shape
    assert terrain_mod.parse_world_text(m_with) is None
    # the legacy field file exists but carries nothing that changes the world
    leg = field_spec.load("legacy")
    assert leg["kind"] == "legacy" and "trees" not in leg and compose.registry_hash(compose.load_field("legacy", 7)) == ""


def test_legacy_run_is_byte_identical_with_the_world_events_hook(world):
    from atharv_nandi.sim.hooks import load_all
    m = world("legacy")
    a = SimDriver(m, load_hooks=False)
    b = SimDriver(m, hooks=load_all(names=["world_events"]))
    for drv in (a, b):
        drv.walk(0.6, 0.0, 0.3, seconds=0.6)
    assert a.d.qpos.tobytes() == b.d.qpos.tobytes()
    assert not [p for p in b.published if p[1].startswith("/nandi/world/")]


# ---------------------------------------------------------------------- terrain vs config (±5 %)
def test_compiled_hfield_matches_the_python_grid_and_is_oriented(world):
    m = world("orchard_demo")
    f = field_spec.load("orchard_demo")
    g = terrain_orchard.grid_for(f, 7)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(3)
    pts = [(12.9, 5.0), (11.9, 14.5), (6.0, -3.5), (2.0, 2.5 + 0.875)] + [
        (float(x), float(y)) for x, y in zip(rng.uniform(-6, 16, 40), rng.uniform(-6, 17, 40))]
    gid = np.array([-1], dtype=np.int32)
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, terrain_mod.GROUND_GEOM)
    n = 0
    for x, y in pts:
        dist = mujoco.mj_ray(m, d, np.array([x, y, 3.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
        if gid[0] != ground:
            continue
        n += 1
        assert 3.0 - dist == pytest.approx(g.height_at(x, y), abs=0.002), (x, y)
    assert n >= 30


def test_basins_bund_and_slopes_within_5_percent_of_config(world):
    m = world("orchard_demo")
    f = field_spec.load("orchard_demo")
    tc = f["terrain"]
    h = tc["basins"]["rim_h_m"]
    crest = tc["basins"]["rim_crest_m"]
    g = terrain_orchard.grid_for(f, 7)
    for t in field_spec.trees(f):
        x, y = field_spec.tree_xy(t)
        r = t["basin_r_m"]
        assert 0.6 <= r <= 0.9                                           # OS 3.6 basin radius range
        # rim crest (depth of the basin) on the four compass points; the basin floor at the trunk
        for a in (0, 90, 180, 270):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            zc = hfield_z(m, x + (r + crest / 2) * ca, y + (r + crest / 2) * sa) - g.layers["noise"][
                int(round((y + (r + crest / 2) * sa - g.y0) / g.res)), int(round((x + (r + crest / 2) * ca - g.x0) / g.res))]
            assert zc == pytest.approx(h, rel=0.05), (t["tree_id"], a)
        assert abs(hfield_z(m, x + 0.3 * r, y)) <= 0.011                 # floor: noise only
    b = tc["bunds"][0]
    (bx0, by), (bx1, _) = b["from"], b["to"]
    mid = (bx0 + bx1) / 2
    for dy in (-0.10, 0.0, 0.10):                                         # flat 0.3 m crest
        zb = hfield_z(m, mid, by + dy) - g.layers["noise"][int(round((by + dy - g.y0) / g.res)), int(round((mid - g.x0) / g.res))]
        assert zb == pytest.approx(b["h_m"], rel=0.05)
    assert terrain_orchard.flank_run(0.10, 30.0) == pytest.approx(math.pi * 0.10 / (2 * math.tan(math.radians(30))))
    for s in tc["slopes"]:
        (x0, x1), (y0, y1) = s["core"]["x"], s["core"]["y"]
        meas = []
        for yy in np.linspace(y0 + 0.2, y1 - 0.2, 7):
            dz = hfield_z(m, x1 - 0.2, yy) - hfield_z(m, x0 + 0.2, yy)
            meas.append(math.degrees(math.atan(dz / (x1 - x0 - 0.4))))
        assert np.allclose(meas, s["deg"], rtol=0.05), (s["id"], meas)
        # the full-slope run along the walk line covers the configured core (5° strip: 4 m); the side faces are
        # cones of the same slope around the high edge, so the run grows a little past the core ends
        along = [yy for yy in np.arange(y0 - 1.0, y1 + 1.0, 0.05)
                 if abs(g.cross_slope_deg(x0 + 0.5, yy, math.pi / 2) + s["deg"]) <= 0.01 * s["deg"]]
        assert y1 - y0 - 0.15 <= max(along) - min(along) <= y1 - y0 + 0.6, (s["id"], min(along), max(along))
    hs = next(s for s in tc["slopes"] if s["id"] == "headland_5deg")
    assert hs["deg"] == 5.0 and hs["core"]["y"][1] - hs["core"]["y"][0] == 4.0
    assert next(s for s in tc["slopes"] if s["id"] == "patch_10deg")["deg"] == 10.0
    assert g.slope_deg_at(11.0, 14.5) == pytest.approx(10.0, rel=0.05)


@pytest.mark.parametrize("w", ["orchard_demo", "orchard_demo_monsoon"])
def test_slope_features_are_never_steeper_than_nominal_over_their_whole_footprint(world, w):
    """Review: the cosine fall / side-blend faces reached 15-16° (5° strip) and 24-34° (10° patch)."""
    m = world(w)
    f = field_spec.load(w)
    g = terrain_orchard.grid_for(f, 7)
    out = {}
    for sd in f["terrain"]["slopes"]:
        s = terrain_orchard.SlopeSpec.from_dict(sd)
        x0, x1, y0, y1 = s.outer()
        rect = (x0 - 0.3, x1 + 0.3, y0 - 0.3, y1 + 0.3)                   # the footprint plus a margin
        layer = g._slope_one(s)                                           # the feature alone, every face
        jm = (g.X[0] >= rect[0]) & (g.X[0] <= rect[1])
        im = (g.Y[:, 0] >= rect[2]) & (g.Y[:, 0] <= rect[3])
        sub = layer[np.ix_(im, jm)]
        k = 4                                                             # 0.2 m baseline
        gx, gy = terrain_orchard.square_gradient(sub, k, g.res)
        feat = float(np.degrees(np.arctan(np.hypot(gx, gy).max())))
        # along any single axis (no stencil pairing at all) the faces are at most nominal too
        axis = max(float(np.abs(np.diff(sub, axis=1)).max()), float(np.abs(np.diff(sub, axis=0)).max())) / g.res
        assert math.degrees(math.atan(axis)) <= s.deg * 1.01, (s.id, math.degrees(math.atan(axis)))
        # outside the footprint the feature is exactly 0
        mask = ~((g.X >= x0) & (g.X <= x1) & (g.Y >= y0) & (g.Y <= y1))
        assert float(np.abs(layer[mask]).max()) == 0.0, s.id
        # the compiled ground over the footprint (basins / noise excluded by staying off the tree lines)
        comp = max_slope_deg(m, rect)
        out[s.id] = {"feature_deg": round(feat, 2), "compiled_deg": round(comp, 2), "grid_deg": round(
            g.max_slope_deg(rect), 2), "corner_stencil_deg": round(max_slope_deg_corner(m, rect), 2),
            "footprint": [round(v, 2) for v in (x0, x1, y0, y1)]}
        assert feat <= s.deg * 1.01, out
        assert g.max_slope_deg(rect) <= s.deg * 1.05, out
        assert comp <= s.deg * 1.05, out
        # the walk lane of the 5° strip is the whole core width, not a 0.4 m lane around walk_x
        if "walk_x" in sd:
            lane = (s.core[0], s.core[1], s.core[2] - 0.5, s.core[3] + 0.5)
            assert g.max_slope_deg(lane) <= s.deg * 1.05, out
    print(f"WP4 {w} slope features, max slope over the whole footprint (0.2 m baseline): {out}")


def test_r1_alleys_are_flat_within_1_degree(world):
    m = world("orchard_demo")
    f = field_spec.load("orchard_demo")
    zones = {z["id"]: z for z in f["terrain"]["noise"]["gentle_zones"] if "id" in z}
    worst = {}
    for zid in ("R1_alley_s", "R1_alley_n"):
        z = zones[zid]
        rect = (z["x"][0] + 0.5, z["x"][1] - 0.5, z["y"][0] + 0.3, z["y"][1] - 0.3)   # alley floor (basin rims excluded)
        worst[zid] = max_slope_deg(m, rect)
        assert worst[zid] <= 1.0, worst
    print(f"WP4 R1 alley max slope (compiled hfield, 0.2 m baseline): {worst}")
    # rest of the field is rougher (noise 1 cm) but still walkable; home pad and OT-1 annulus are flat
    g = terrain_orchard.grid_for(f, 7)
    assert g.max_slope_deg((-2.0, 4.0, -3.0, 0.8)) <= 0.05            # home pad (the patch pits excluded)
    ot = field_spec.tree_by_id(f, "OT-1")
    ox, oy = field_spec.tree_xy(ot)
    r1 = ot["canopy_r_m"] + ot["orbit_standoff_m"]                       # 1.9 m orbit radius
    for r in (r1 - 0.5, r1, r1 + 0.5):                                   # OT-1 orbit annulus is flat
        for a in np.radians(np.arange(0, 360, 5)):
            assert g.slope_deg_at(ox + r * math.cos(a), oy + r * math.sin(a)) <= 0.05


def test_mud_sink_and_patch_geoms(world):
    m = world("orchard_demo")
    f = field_spec.load("orchard_demo")
    for p in f["terrain"]["patches"]:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"patch_{p['id']}")
        assert gid >= 0
        assert m.geom_friction[gid][0] == pytest.approx(p["friction"])
        assert int(m.geom_priority[gid]) > int(m.geom_priority[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_GEOM, terrain_mod.GROUND_GEOM)])          # patch friction reaches the foot
        top = m.geom_pos[gid][2] + m.geom_size[gid][2]
        assert top == pytest.approx(0.0, abs=1e-9)
        assert hfield_z(m, *p["center"]) == pytest.approx(-p["depth_m"], abs=1e-6)
        assert math.hypot(*p["center"]) >= 2.5                             # near home, off the dock
    mud = next(p for p in f["terrain"]["patches"] if p["kind"] == "mud")
    assert mud["friction"] == 0.3 and mud["center"] == [-3.0, -2.0]
    # mud and sink on flat ground near home, away from the 10° patch
    g = terrain_orchard.grid_for(f, 7)
    for p in f["terrain"]["patches"]:
        cx, cy = p["center"]
        hx, hy = p["half"]
        ring = [(cx + sx * (hx + 0.3), cy + sy * (hy + 0.3)) for sx in (-1, 0, 1) for sy in (-1, 0, 1) if sx or sy]
        assert all(g.slope_deg_at(x, y) <= 0.05 for x, y in ring)
        assert math.dist(p["center"], (11.0, 14.5)) > 10.0


def _stand_at(m, xy, secs=2.0):
    drv = SimDriver(m, load_hooks=False)
    drv.d.qpos[0:2] = xy
    tools_mod.seat_tools(m, drv.d)                 # WP2: racked tools follow a teleported robot
    mujoco.mj_forward(m, drv.d)
    drv.wait(secs)
    fids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FEET]
    return [-(float(drv.d.geom_xpos[g][2]) - 0.03) for g in fids], drv


def test_sink_patch_takes_3_to_5_cm_of_foot_penetration_and_mud_does_not(world):
    m = world("orchard_demo")
    pen_sink, drv = _stand_at(m, (-5.0, -2.0))
    assert all(0.03 <= p <= 0.05 for p in pen_sink), pen_sink
    assert max(abs(v) for v in drv.roll_pitch()) < math.radians(1.0)
    pen_mud, _ = _stand_at(m, (-3.0, -2.0))
    assert all(p <= 0.005 for p in pen_mud), pen_mud
    pen_flat, _ = _stand_at(m, (-1.5, -2.0))
    assert all(p <= 0.002 for p in pen_flat), pen_flat
    print(f"WP4 sink: 6-foot stance penetration {min(pen_sink) * 100:.2f}-{max(pen_sink) * 100:.2f} cm; "
          f"mud {max(pen_mud) * 1000:.1f} mm; flat {max(pen_flat) * 1000:.1f} mm")


def test_monsoon_variant_floods_basins_and_softens_alleys(world):
    m = world("orchard_demo_monsoon")
    f = field_spec.load("orchard_demo_monsoon")
    ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, terrain_mod.GROUND_GEOM)
    assert m.geom_friction[ground][0] == pytest.approx(0.4)
    floods = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("patch_flood_")]
    assert len(floods) == 17
    for g in floods:
        assert m.geom_friction[g][0] == pytest.approx(0.3)
        assert m.geom_solref[g][0] == pytest.approx(terrain_mod.SINK_SOLREF[0])
    muds = [p for p in f["terrain"]["patches"] if p["kind"] == "mud"]
    assert len(muds) == 3 * len([p for p in field_spec.load("orchard_demo")["terrain"]["patches"] if p["kind"] == "mud"])
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "water_R1-1") >= 0
    assert any(e["type"] == "rain" and e["trigger"]["mode"] == "always" for e in f["events"])
    dry = world("orchard_demo")
    assert dry.geom_friction[mujoco.mj_name2id(dry, mujoco.mjtObj.mjOBJ_GEOM, terrain_mod.GROUND_GEOM)][0] == pytest.approx(0.8)


def test_orchard_worlds_keep_the_frozen_layout_and_add_no_dof(world):
    nv = {}
    for w in ("legacy", "orchard_lite", "orchard_demo", "orchard_demo_monsoon"):
        m = world(w)
        lay = mlay.current()
        assert lay.world == w and not mlay.frozen_violations(lay)
        nv[w] = (m.nq, m.nv, m.nu)
    orch = {v for w, v in nv.items() if w != "legacy"}
    assert len(orch) == 1, nv                                         # terrain/fixtures/actors/trees are joint-free
    nq0, nv0, nu0 = nv["legacy"]                                      # only the WP23 fruit pool adds dof: N free slots
    # ... and the 3 WP2 tools: free joint + roll + blade/jaw each (7 + 2 qpos, 6 + 2 dof, 2 actuators)
    assert orch == {(nq0 + 7 * pool_mod.N_SLOTS + 3 * 9, nv0 + 6 * pool_mod.N_SLOTS + 3 * 8, nu0 + 6)}, nv
    m = world("orchard_demo")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    assert np.all(np.isfinite(d.qpos)) and d.qpos[2] == pytest.approx(0.34)
    assert terrain_mod.parse_world_text(m) == ("orchard_demo", 7)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert m.geom_contype[floor] == 0 and m.geom_conaffinity[floor] == 0
    for name in ("ridge_1", "ridge_2", "clod_0"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name) == -1


def test_orchard_world_refuses_the_legacy_tilted_terrain(world):
    with pytest.raises(ValueError, match="NANDI_TERRAIN=orchard"):
        world("orchard_lite", NANDI_TERRAIN="orchard")


def test_world_id_tracks_terrain_and_seed(world, monkeypatch):
    monkeypatch.setenv("NANDI_NOVNC", "0")
    monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
    a = farm_scene.world_identity()[1]
    monkeypatch.setenv("NANDI_WORLD", "orchard_demo_monsoon")
    b = farm_scene.world_identity()[1]
    monkeypatch.setenv("NANDI_SEED", "11")
    c = farm_scene.world_identity()[1]
    assert len({a, b, c}) == 3
    g7 = terrain_orchard.grid_for(field_spec.load("orchard_demo"), 7)
    g11 = terrain_orchard.grid_for(field_spec.load("orchard_demo"), 11)
    assert not np.array_equal(g7.z, g11.z) and np.array_equal(g7.z, terrain_orchard.TerrainGrid(g7.cfg, 7).z)


def test_terrain_grid_queries():
    f = field_spec.load("orchard_lite")
    g = terrain_orchard.grid_for(f, 7)
    assert g.height_at(100.0, 0.0) == 0.0 and g.surface_at(-100.0, 0.0) == 0.0
    assert g.surface_at(-3.0, -2.0) == 0.0 and g.height_at(-3.0, -2.0) == pytest.approx(-0.06)
    data, rng, base = g.hfield_data()
    assert data.min() == 0.0 and data.max() == pytest.approx(1.0) and base == pytest.approx(g.zmin)
    assert g.summary()["n_basins"] == 6
    assert g.cross_slope_deg(0.0, 0.0, 0.0) == pytest.approx(0.0, abs=0.01)
    assert g.max_slope_deg((0.0, 0.01, 0.0, 0.01)) == 0.0
    cfg = terrain_orchard.TerrainConfig.from_field({"terrain": {"slopes": [
        {"id": "s", "deg": 8.0, "core": {"x": [0, 2], "y": [0, 2]}, "rise": "-y"}], "extent": {"x": [-3, 3], "y": [-3, 3]}}})
    gg = terrain_orchard.TerrainGrid(cfg, 1)
    assert gg.slope_deg_at(1.0, 1.0) == pytest.approx(8.0, rel=0.02)
    assert gg.height_at(1.0, 0.0) > gg.height_at(1.0, 2.0)
