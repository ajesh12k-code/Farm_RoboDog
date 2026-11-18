"""WP23 tree generator, foliage envelope, leaf targets, registry and orchard worlds (ORCHARD-MVP-SPEC 3.3, 4.1, 4.2).

Headless, ROS-free; every test well under 10 s (an orchard_demo compile is ~0.35 s on the 2-CPU sim runner).
"""
import contextlib
import copy
import json
import os
import shutil

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core import sim_state
from atharv_nandi.sim.world import compose, crops
from atharv_nandi.sim.world import trees as gen
from atharv_nandi.sim.world.modules import trees as mt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CANDIDATES = gen.load_candidates()
# Golden plan hashes at seed 7 (default season demo_all_fruiting): change only with a deliberate generator /
# config change, and say so in the commit (the hash is folded into nothing else; world_id uses the configs).
GOLDEN = {"OT-1": "d2cc8fab5b", "R2-1": "21daad7af3", "R3-1": "2651fa75a3", "R4-1": "17a9dcc998"}


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build(world, fields_dir=compose.FIELDS_DIR, **kw):
    """(model, plans, composed) without touching model_layout.current()."""
    with env(NANDI_WORLD=world, **kw):
        cfg = farm_scene.world_config()
        w = compose.build(cfg["world"], cfg["seed"], farm_scene.ROBOT_XML, cfg, fields_dir=fields_dir)
        m = w.spec.compile()
        return m, mt.current_plans() if w.field.get("trees") or gen.registry(w.field) else [], w


@pytest.fixture(scope="module")
def demo():
    return build("orchard_demo")


@pytest.fixture(scope="module")
def lite():
    return build("orchard_lite")


def tree_geoms(m, body):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]


def gname(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)


# ---------------------------------------------------------------------- configs
def test_species_configs_validate():
    sp = gen.load_all_species()
    assert set(sp) >= {"mango", "guava", "papaya", "banana"}
    assert {s["archetype"] for s in sp.values()} == {"canopy", "papaya", "banana"}
    for s in sp.values():
        assert gen.validate_species(s) == []
    bad = copy.deepcopy(sp["mango"])
    bad["archetype"] = "cactus"
    assert gen.validate_species(bad)
    bad = copy.deepcopy(sp["mango"])
    bad["leaves"]["n"] = 13
    assert any("leaves.n" in e for e in gen.validate_species(bad))


# ---------------------------------------------------------------------- determinism
def test_seeded_hash_stable():
    species, profiles = gen.load_all_species(), crops.load_profiles()
    reg = {e["tree_id"]: e for e in gen.default_registry("orchard_demo")}
    for tid, h in GOLDEN.items():
        e = reg[tid]
        a = gen.generate(e, species[e["crop"]], profiles[e["crop"]], seed=7)
        b = gen.generate(copy.deepcopy(e), species[e["crop"]], profiles[e["crop"]], seed=7)
        assert a["hash"] == b["hash"] == gen.plan_hash(a)
        assert a["hash"].startswith(h), (tid, a["hash"])
        assert gen.generate(e, species[e["crop"]], profiles[e["crop"]], seed=11)["hash"] != a["hash"]
        assert json.loads(json.dumps(a)) == a                       # plain JSON throughout


def test_plans_identical_across_worlds_and_builds(demo):
    # WP4's orchard_lite has its own trees (L1-1..L2-4); the monsoon variant shares the 17 demo trees
    d = {p["tree_id"]: p["hash"] for p in demo[1]}
    mon = {p["tree_id"]: p["hash"] for p in build("orchard_demo_monsoon")[1]}
    assert mon == d
    assert [p["hash"] for p in build("orchard_demo")[1]] == [p["hash"] for p in demo[1]]


# ---------------------------------------------------------------------- budgets and groups
@pytest.mark.parametrize("season", gen.SEASONS)
def test_render_geoms_per_tree_at_most_70_incl_leaf_quads(season):
    with env(NANDI_SEASON=season):
        plans = gen.world_plans(compose.load_field("orchard_demo", 7), 7)
    assert len(plans) == 17
    for p in plans:
        n = sum(1 for g in p["geoms"] if g["group"] <= gen.RENDER_GROUP_MAX)
        assert n == p["render_geoms"] <= gen.RENDER_GEOMS_PER_TREE_MAX, (p["tree_id"], season, n)
        assert p["leaf_quads"] <= gen.LEAF_QUADS_MAX


def test_compiled_tree_geoms_match_the_plan_and_groups(demo):
    m, plans, _ = demo
    for p in plans:
        gids = tree_geoms(m, p["body"])
        assert len(gids) == len(p["geoms"])
        render = [g for g in gids if m.geom_group[g] <= gen.RENDER_GROUP_MAX]
        assert len(render) == p["render_geoms"] <= 70
        for g in gids:
            name, grp = gname(m, g), int(m.geom_group[g])
            assert m.geom_conaffinity[g] == 0, name                  # no tree geom ever has conaffinity
            if "_wood_" in name:
                assert grp == gen.GRP_WOOD and m.geom_contype[g] == 1
            elif "_prune_" in name:
                assert grp in (gen.GRP_WOOD, gen.GRP_FOLIAGE) and m.geom_contype[g] == 0
            elif "_foliage_" in name or "_leaf_" in name or "_panicle_" in name:
                assert grp == gen.GRP_FOLIAGE and m.geom_contype[g] == 0
            elif "_envelope_" in name:
                assert grp == gen.GRP_QUERY and m.geom_contype[g] == 0
            elif "_decal_" in name:
                assert grp == gen.GRP_DECAL and m.geom_contype[g] == 0
            elif "_fruit_" in name or "_bag_" in name or "_ribbon_" in name:
                assert grp == gen.GRP_FRUIT and m.geom_contype[g] == 0
                assert m.geom_rgba[g, 3] == 1.0                     # hideable: compiled alpha is exactly 1
            elif "_bunch_" in name:
                assert grp == gen.GRP_FRUIT and m.geom_contype[g] == 1
            else:
                raise AssertionError(f"unclassified tree geom {name}")
        # foliage is merged: at most 2 mesh geoms per tree
        assert sum(1 for g in gids if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH) <= 2
        # every tree has a query-only foliage envelope
        assert any("_envelope_" in gname(m, g) for g in gids)


def test_trees_are_static_and_jointless(demo):
    m, plans, _ = demo
    for p in plans:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p["body"])
        assert m.body_jntnum[b] == 0 and m.body_parentid[b] == 0 and m.body_weldid[b] == 0


def test_leaf_quads_with_leaf_k_sites(demo):
    m, plans, _ = demo
    for p in plans:
        leaves = p["truth"]["leaves"]
        assert len(leaves) == p["leaf_quads"] <= 12
        for lf in leaves:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, lf["site"])
            assert sid >= 0 and lf["site"].endswith(f"_leaf_k_{lf['k']}")
            np.testing.assert_allclose(m.site_pos[sid], lf["p"], atol=1e-6)
            assert abs(np.linalg.norm(lf["n"]) - 1.0) < 1e-6
        if p["crop"] in ("mango", "guava"):
            assert len(leaves) == 12 and sum(lf["in_band"] for lf in leaves) >= 6
            assert sum(1 for lf in leaves if lf["decal"]) == 2       # 8-15 mm colony/symptom-level decals
        if p["crop"] == "papaya":
            assert leaves == [] and p["truth"]["leaf_scan"] == "n/a: crown-only"
        if p["crop"] == "banana":
            assert all(lf["kind"] == "dead_blade" and lf["in_band"] for lf in leaves)


def test_papaya_has_no_foliage_in_the_pick_band(demo):
    for p in demo[1]:
        if p["crop"] != "papaya":
            continue
        for mesh in p["meshes"]:
            z = np.asarray(mesh["vert"]).reshape(-1, 3)[:, 2]
            assert z.min() > gen.PICK_BAND_Z_M, (p["tree_id"], z.min())


def test_foliage_envelope_matches_the_wp0c_canopy_model(demo):
    cand = {t["id"]: t for t in CANDIDATES["candidates"]["trees"]}
    for p in demo[1]:
        t = cand[p["tree_id"]]
        env_ = p["truth"]["envelope"]
        H, cr = t["H"], t["canopy_r"]
        for z in (0.6, 1.0, 1.5, 2.0):
            c, s = np.asarray(env_["c"]), np.asarray(env_["size"])
            hw = s[0] * np.sqrt(max(0.0, 1.0 - ((z - c[2]) / s[2]) ** 2)) if abs(z - c[2]) < s[2] else 0.0
            want = gen.canopy_halfwidth(p["archetype"], H, cr, t["trunk_h"], z, t.get("alley_clear_height_m"))
            assert abs(hw - want) < 1e-6, (p["tree_id"], z)


# ---------------------------------------------------------------------- WP0c alignment and demo truth
def test_in_band_fruit_are_the_wp0c_frozen_points(demo):
    plans = {p["tree_id"]: p for p in demo[1]}
    st = {t["id"]: t for t in CANDIDATES["frozen"]["targets"]}
    n = 0
    for tg in CANDIDATES["candidates"]["targets"]:
        if tg["task"] not in ("mango_pedicel_cut", "guava_clip", "papaya_cradle"):
            continue
        fr = [f for f in plans[tg["tree"]]["truth"]["fruit"] if (f.get("wp0c") or {}).get("id") == tg["id"]]
        assert len(fr) == 1, tg["id"]
        f = fr[0]
        key = "centre" if tg["task"] == "papaya_cradle" else "cut_k"
        np.testing.assert_allclose(f[key], tg["p"], atol=1e-6)
        frozen = st[tg["id"]]["status"] == "FROZEN"
        if st[tg["id"]].get("reason") == "above_pick_band":
            assert not f["in_band"]
        if frozen:
            assert f["in_band"]
        n += 1
    assert n == 40 + 28 + 20


def test_prune_targets_are_non_colliding_and_wp0c_aligned(demo):
    m, plans, _ = demo
    plans = {p["tree_id"]: p for p in plans}
    for tg in CANDIDATES["candidates"]["targets"]:
        if not tg["task"].startswith("prune_") and tg["task"] != "banana_tidy":
            continue
        pr = [t for t in plans[tg["tree"]]["truth"]["prune"] if (t.get("wp0c") or {}).get("id") == tg["id"]]
        assert len(pr) == 1, tg["id"]
        t = pr[0]
        np.testing.assert_allclose(t["cut_p"], tg["p"], atol=1e-6)
        assert t["diameter_mm"] == tg["stem_mm"] and t["action"] == "cut" and not t["keep"]
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, t["geom"])
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    for p in plans.values():
        for t in p["truth"]["prune"]:
            if t["job"] == "rootstock_sprout":
                assert t["below_graft_union"] and t["a"][2] < p["truth"]["graft_union_h_m"]   # shoot origin
        for w in p["truth"]["wood"]:
            assert w["keep"]


def test_demo_fruit_truth_os_1_2(demo):
    plans = {p["tree_id"]: p for p in demo[1]}
    mature_in = lambda p: [f for f in p["truth"]["fruit"] if f["in_band"] and f["harvest_stage"]]  # noqa: E731
    assert len(mature_in(plans["OT-1"])) == 3                       # step 5: 3 of 3 in-band mature mangoes
    out = [f for f in plans["OT-1"]["truth"]["fruit"] if not f["in_band"] and f["harvest_stage"]]
    assert out, "OT-1 needs out-of-band mature fruit for MARK pins"
    assert len(mature_in(plans["R2-2"])) == 2                       # step 6: 2 guavas on G-2
    for tid in ("R3-1", "R3-2", "R3-3", "R3-4"):                    # papaya: lowest in-band ripe
        fr = sorted(plans[tid]["truth"]["fruit"], key=lambda f: f["centre"][2])
        ripe = [f for f in fr if f["harvest_stage"]]
        assert 1 <= len(ripe) <= 3 and ripe == fr[:len(ripe)]
        assert all(1.0 <= f["mass_kg"] <= 1.8 for f in fr)
    r33 = {f["wp0c"]["id"]: f for f in plans["R3-3"]["truth"]["fruit"] if f.get("wp0c")}
    assert r33["R3-3:papaya_cradle:1"]["clearance_mm"] < 15       # WP0c's no_clearance papaya, from geometry
    for p in plans.values():
        for f in p["truth"]["fruit"]:
            if p["crop"] == "mango":
                assert 0.15 <= f["mass_kg"] <= 0.35 and 70 <= f["length_mm"] <= 100
                assert f["pool_kind"] == "connect"
                assert abs(f["cut_k"][2] - f["stem_end"][2] - 0.02) < 1e-9    # cut_k 2.0 cm above the stem end
                assert f["pedicel_len_m"] > f["pedicel_leave_m"]
            if p["crop"] == "guava":
                assert 0.136 <= f["mass_kg"] <= 0.164 and f["pool_kind"] == "weld"


def _spheroid_overlap(c1, r1, h1, c2, r2, h2):
    """True overlap test for two same-shape upright spheroids (exact after the z scaling)."""
    assert abs(r1 / h1 - r2 / h2) < 1e-4                    # same shape (sizes are rounded to 1 um)
    k = min(r1 / h1, r2 / h2)
    d = np.asarray(c2) - np.asarray(c1)
    return float(np.sqrt(d[0] ** 2 + d[1] ** 2 + (k * d[2]) ** 2)) - r1 - r2


@pytest.mark.parametrize("world", ["orchard_demo", "orchard_lite"])
def test_papaya_fruit_never_interpenetrate_and_jaw_clearance_os_1_2_step_8(world, demo, lite):
    plans = (demo if world == "orchard_demo" else lite)[1]
    pf = crops.load_profile("papaya")
    for p in plans:
        if p["crop"] != "papaya":
            continue
        fr = p["truth"]["fruit"]
        for a in range(len(fr)):
            for b in range(a + 1, len(fr)):
                fa, fb = fr[a], fr[b]
                g = _spheroid_overlap(fa["centre"], fa["size_m"][0], fa["size_m"][2],
                                      fb["centre"], fb["size_m"][0], fb["size_m"][2])
                assert g >= 0.002 - 1e-9, (fa["fruit_id"], fb["fruit_id"], g)
        for f in fr:
            assert 2 * f["size_m"][0] <= 0.16 + 1e-9                 # fits the open T4 jaw
            assert f["fruit_gap_mm"] >= 2.0 - 1e-6
            # the jaw-clearance truth includes the trunk: never above the finger-to-trunk distance
            c = np.asarray(f["centre"])
            az = np.arctan2(c[1], c[0])
            t = np.array([-np.sin(az), np.cos(az), 0.0]) * max(0.08, f["size_m"][0])
            trunk_r = p["entry"]["trunk_r_m"]
            finger_trunk_mm = 1000 * (min(np.hypot(*(c + t)[:2]), np.hypot(*(c - t)[:2])) - trunk_r)
            assert f["clearance_mm"] <= finger_trunk_mm + 0.051       # 0.1 mm rounding
        if p["tree_id"] == "R3-1":                                   # P-1 (OS 1.2 step 8)
            ripe = sorted((f for f in fr if f["harvest_stage"] and f["in_band"]), key=lambda f: f["centre"][2])
            assert len(ripe) == 2
            assert crops.harvest_decision(pf, ripe[0]) == ("pick", "")
            assert crops.harvest_decision(pf, ripe[1]) == ("mark", "no_clearance")
            assert ripe[0]["clearance_mm"] >= 15 > ripe[1]["clearance_mm"]


def test_jaw_clearance_sees_trunk_and_neighbours():
    c = np.array([0.0, -0.15, 0.7])
    r, h = 0.07, 0.11
    free = gen.papaya_jaw_clearance_m(c, r, h, [], trunk_r=0.07, trunk_h=2.0)
    # alone: limited by the trunk, finger at (+-0.08, -0.15) from the axis
    assert abs(free - (np.hypot(0.08, 0.15) - 0.07)) < 1e-9
    near = gen.papaya_jaw_clearance_m(c, r, h, [(np.array([0.08 + 0.005 + r, -0.15, 0.7]), r, h)], 0.07, 2.0)
    assert abs(near - 0.005) < 2e-4                              # 5 mm from the finger line to the surface
    assert gen.ellipsoid_gap_m(c, r, h, c + [0.0, 0.0, 2 * h + 0.01], r, h) == pytest.approx(0.01 * r / h)


def test_mango_skin_colour_carries_no_stage(demo):
    for crop, n_colours in (("mango", 1), ("guava", 2), ("papaya", 2)):
        cols = {tuple(f["rgba"]) for p in demo[1] if p["crop"] == crop for f in p["truth"]["fruit"]}
        stages = {f["stage"] for p in demo[1] if p["crop"] == crop for f in p["truth"]["fruit"]}
        assert len(stages) == 2 and len(cols) == n_colours, crop
    mango = [f for p in demo[1] if p["crop"] == "mango" for f in p["truth"]["fruit"]]
    for f in mango:
        lo, hi = (2.0, 5.0) if f["stage"] == "mature" else (0.0, 1.0)
        assert lo <= f["cues"]["shoulder_rise_mm"] <= hi
        assert (f["days_since_fruit_set"] >= 84) == (f["stage"] == "mature")


def test_banana_truth_os_3_3(demo):
    m, plans, _ = demo
    for p in plans:
        if p["crop"] != "banana":
            continue
        t = p["truth"]
        assert 9 <= t["leaves_total"] <= 12
        dead = [b for b in t["blades"] if b["dead"]]
        assert len(dead) == 2
        tidy = [x for x in t["prune"] if x["job"] == "banana_tidy"]
        assert len(tidy) == 2 and all(x["in_band"] and x["action"] == "cut" for x in tidy)
        assert 1 <= sum(1 for b in t["blades"] if b["sigatoka_pct"] > 0) <= 2
        b = t["bunch"]
        assert 1.2 <= b["h_m"] <= 1.8 and b["ribbon_colour"] in gen.RIBBON_RGBA and b["tool_hint"] == "human_cut"
        assert b["shooting_date"] and b["harvest_window"]["basis"] == "days_since_shooting"
        assert b["exclusion_m"] == 3.0
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, b["geom"])
        assert m.geom_contype[g] == 1                               # the bunch is collidable
        suck = [x for x in t["prune"] if x["job"] == "sucker"]
        assert len(suck) == 2 and sum(x.get("follower", False) for x in suck) == 1
        assert all(x["action"] in ("mark", "skip") for x in suck)
        stem = [w for w in t["wood"] if w["role"] == "pseudostem"][0]
        assert abs(stem["girth_cm"] - 74.1) < 0.5                   # girth 74 cm, r 0.118
        weevil = [d for d in t["decals"] if d["class"] == "weevil_gum"]
        assert all(0.2 <= d["h_m"] <= 1.5 for d in weevil)
    droop = plans[[p["tree_id"] for p in plans].index("R4-2")]["truth"]["drooping_leaf"]
    assert abs(droop["tip_z_m"] - 1.6) < 1e-6
    r42 = [p for p in plans if p["tree_id"] == "R4-2"][0]
    assert sum(1 for g in r42["geoms"] if g["kind"] == "envelope") == 2    # ellipsoid + drooping-leaf capsule


def test_guava_prune_marks_os_1_2_step_7(demo):
    for p in demo[1]:
        if p["crop"] != "guava":
            continue
        pr = p["truth"]["prune"]
        thick = [t for t in pr if t["job"] == "skirt_lift" and t["diameter_mm"] > 12]
        assert thick and all(t["action"] == "mark" and t["paste_required"] for t in thick)
        ws = [t for t in pr if t["job"] == "water_shoot"]
        assert ws and all(t["action"] == "mark" for t in ws)
        cuts = [t for t in pr if t["action"] == "cut"]
        assert all(t["in_band"] and t["diameter_mm"] <= 12 for t in cuts)


def test_ot1_registry_flags_and_decals(demo):
    p = [p for p in demo[1] if p["tree_id"] == "OT-1"][0]
    assert p["entry"]["orbit_clear"] is True
    gum = [d for d in p["truth"]["decals"] if d["class"] == "gummosis"]
    assert gum and abs(gum[0]["h_m"] - 0.3) < 1e-6 and gum[0]["band"] == "low"
    assert {lf["decal"] for lf in p["truth"]["leaves"] if lf["decal"]} == {"anthracnose", "mealybug"}
    assert all(d["diameter_mm"] >= 50 for d in p["truth"]["decals"] if d.get("leaf") is None)


def test_registry_carries_mission_presets_0_1_fields():
    need = ("tree_id", "physical_tag", "block_id", "row_id", "crop", "cultivar", "form", "height_m", "canopy_r_m",
            "trunk_r_m", "graft_union_h_m", "orbit_clear", "alley_sides", "alley_clear_height_m",
            "drip_line_offset_m", "rope_lines", "props", "fruit_set_date", "bagging_date", "ribbon_colour", "flags")
    reg = gen.default_registry("orchard_demo")
    assert len(reg) == 17 and len({e["physical_tag"] for e in reg}) == 17
    for e in reg:
        for k in need:
            assert k in e, (e["tree_id"], k)
    r = {e["tree_id"]: e for e in reg}
    assert r["R4-1"]["rope_lines"][0]["h_m"] == 1.4 and r["R4-3"]["props"][0]["kind"] == "bamboo"
    assert r["R4-1"]["alley_clear_height_m"] == 1.3
    # OQ-02 B: mango rows 5 x 2.5 m
    xs = sorted(e["x"] for e in reg if e["row_id"] == "R1")
    assert np.allclose(np.diff(xs), 2.5)
    assert gen.default_registry("legacy") == []


# ---------------------------------------------------------------------- seasons
def test_seasons_change_fruit_bags_and_panicles():
    field = compose.load_field("orchard_demo", 7)
    by = {}
    for s in gen.SEASONS:
        with env(NANDI_SEASON=s):
            by[s] = {p["tree_id"]: p for p in gen.world_plans(field, 7)}
    ot = lambda s: by[s]["OT-1"]["truth"]  # noqa: E731
    assert ot("panicle")["fruit"] == [] and len(ot("panicle")["panicles"]) >= 8
    assert sum(pn["malformed"] for pn in ot("panicle")["panicles"]) >= 1
    assert ot("post_harvest")["fruit"] == []
    bagged = ot("bagged")["fruit"]
    assert bagged and all(f["bagged"] and f["static_kind"] == "bag" for f in bagged) and ot("bagged")["bags_n"]
    prof = crops.load_profile("mango")
    assert crops.harvest_decision(prof, bagged[0]) == ("skip", "bagged")
    fs = ot("fruit_set")["fruit"]
    assert fs and all(not f["harvest_stage"] for f in fs)
    assert max(f["length_mm"] for f in fs) < min(f["length_mm"] for f in ot("mature")["fruit"])
    # papaya and banana fruit in every season (non-seasonal in the sim); decals carry the season
    for s in gen.SEASONS:
        assert by[s]["R3-1"]["truth"]["fruit"] and by[s]["R4-1"]["truth"]["bunch"]
        assert all(d["season"] == s for d in by[s]["OT-1"]["truth"]["decals"])
    with env(NANDI_SEASON="monsoon"), pytest.raises(gen.TreeConfigError):
        gen.season_name({})


# ---------------------------------------------------------------------- worlds, nv invariance, world_id
def test_legacy_world_has_no_trees_or_pool():
    m, _, w = build("legacy")
    assert m.nq == 32 + 7 * 6 and m.nv == 31 + 6 * 6 and m.neq == 0
    assert not any((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("tree_", "pool_"))
                   for b in range(m.nbody))
    sim_state.discover_providers()
    muts = sim_state.pack_mutations(m)["providers"]
    assert "pool" not in muts and "hidden_geoms" not in muts      # legacy sim_state stays byte-identical
    om = build("orchard_lite")[0]
    assert {"pool", "hidden_geoms"} <= set(sim_state.pack_mutations(om)["providers"])
    build("legacy")
    assert "pool" not in sim_state.pack_mutations(m)["providers"]


def test_providers_and_plans_follow_the_model_not_the_last_build():
    """Composing a legacy (or another orchard) world after an orchard one must not change what the orchard model
    packs or which plans its pool sees (providers are per model via sim_state ``applies``; plans per model)."""
    from atharv_nandi.core import fruit_pool as fp
    om, oplans, _ = build("orchard_demo")
    lm, _, _ = build("legacy")
    sim_state.discover_providers()
    assert {"pool", "hidden_geoms"} <= set(sim_state.pack_mutations(om)["providers"])
    assert not {"pool", "hidden_geoms"} & set(sim_state.pack_mutations(lm)["providers"])
    assert mt.plans_for_model(om) == oplans and mt.plans_for_model(lm) == []
    pool = fp.FruitPool(om)                                    # default plans: the model's own
    assert len(pool.fruit) == sum(len(p["truth"]["fruit"]) for p in oplans) > 0
    lite_m, lite_plans, _ = build("orchard_lite")             # same tree ids (OT-1, R2-1...), other world
    assert mt.plans_for_model(om) == oplans and mt.plans_for_model(lite_m) == lite_plans
    assert fp.FruitPool(lite_m).fruit.keys() == fp.tree_gen.fruit_table(lite_plans).keys()
    # a strict apply on the legacy model needs no WP23 blocks; a WP23 block sent to it is refused
    d = mujoco.MjData(lm)
    msg = sim_state.pack(lm, d, "w")
    assert sim_state.apply(lm, d, msg, "w")[0]
    msg["mutations"]["providers"]["pool"] = {"v": 1, "data": {}}
    ok, why = sim_state.apply(lm, d, msg, "w")
    assert not ok and "does not apply" in why


def test_nv_invariant_17_vs_32_trees(tmp_path, demo):
    m17 = demo[0]
    reg = gen.default_registry("orchard_demo")
    more = []
    for k, e in enumerate(reg[1:16]):                     # 15 more trees, copies shifted to a far block
        e = copy.deepcopy(e)
        e["tree_id"] = f"Z{k + 1}"
        e["physical_tag"] = f"Z{k + 1}"
        e["y"] = e["y"] + 20.0
        for key in ("wood", "targets", "leaves"):
            e.pop(key, None)
        more.append(e)
    (tmp_path / "orchard_demo.json").write_text(json.dumps({"trees": reg + more}))
    m32, plans32, w = build("orchard_demo", fields_dir=str(tmp_path))
    assert len(plans32) == 32
    assert (m32.nq, m32.nv, m32.nu, m32.neq, m32.njnt) == (m17.nq, m17.nv, m17.nu, m17.neq, m17.njnt)
    assert m17.nv == 31 + 36 + 60 + 24                      # robot + 6 legacy crops + 10 pool bodies + 3 WP2 tools
    assert m32.ngeom > m17.ngeom
    assert w.world_id != demo[2].world_id                   # the field registry is in world_id


def test_field_registry_merges_wp0c_anchors_and_needs_positions(tmp_path):
    (tmp_path / "orchard_lite.json").write_text(json.dumps({"trees": [
        {"tree_id": "OT-1", "crop": "mango", "x": 6.0, "y": -3.5, "height_m": 2.2, "canopy_r_m": 1.0}]}))
    field = compose.load_field("orchard_lite", 7, str(tmp_path))
    reg = gen.registry(field)
    assert reg[0]["targets"] and reg[0]["wood"]             # merged from WP0c by tree id
    with pytest.raises(gen.TreeConfigError):
        gen.normalize_entry({"tree_id": "Q", "crop": "mango"})


def test_orchard_lite_composition(lite):
    m, plans, _ = lite
    ids = [p["tree_id"] for p in plans]
    assert ids == ["L1-1", "L1-2", "L2-1", "L2-2", "L2-3", "L2-4"]        # WP4 config/fields/orchard_lite.json
    assert [p["crop"] for p in plans[2:]] == ["mango", "guava", "papaya", "banana"]
    hr = [p for p in plans if p["tree_id"].startswith("L1-")]
    assert all(p["entry"]["form"] == "hedgerow" for p in hr)
    assert abs(hr[1]["pos"][0] - hr[0]["pos"][0]) == 2.0     # 3 x 2 m UHDP pair
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))


def test_world_id_tracks_season_and_tree_config(tmp_path):
    _, _, w0 = build("orchard_lite")
    _, _, w1 = build("orchard_lite", NANDI_SEASON="panicle")
    assert w1.world_id != w0.world_id
    shutil.copytree(gen.TREES_DIR, tmp_path / "trees")
    p = tmp_path / "trees" / "mango.yaml"
    p.write_text(p.read_text().replace("blobs: 40", "blobs: 38"))
    with env(NANDI_TREES_CONFIG=str(tmp_path / "trees")):
        wid = compose.identify("orchard_lite", 7, farm_scene.world_config())[0]
    assert wid != w0.world_id


def test_fifth_crop_by_config_builds_orchard_lite(tmp_path, lite):
    import yaml
    tdir, cdir = tmp_path / "trees", tmp_path / "crops"
    shutil.copytree(gen.TREES_DIR, tdir)
    shutil.copytree(crops.CROPS_DIR, cdir)
    sp = yaml.safe_load((tdir / "guava.yaml").read_text())
    sp.update(species="lychee", label="Lychee (Shahi), by config only", height_m=[1.8, 2.6])
    (tdir / "lychee.yaml").write_text(yaml.safe_dump(sp))
    pf = yaml.safe_load((cdir / "guava.yaml").read_text())
    pf.update(crop="lychee", cultivars=["Shahi"], default_cultivar="Shahi", mvp_scope="scout_mark",
              calendar_up={"Shahi": {"harvest": ["05-20", "06-15"]}}, mass_kg=[0.018, 0.025])
    (cdir / "lychee.yaml").write_text(yaml.safe_dump(pf))
    m, plans, w = build("orchard_lite", NANDI_TREES_CONFIG=str(tdir), NANDI_CROPS_CONFIG=str(cdir))
    ids = [p["tree_id"] for p in plans]
    assert "X1-1" in ids and len(ids) == 7
    x = plans[ids.index("X1-1")]
    assert x["crop"] == "lychee" and x["truth"]["fruit"] and x["render_geoms"] <= 70
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tree_X1-1") >= 0
    assert (m.nq, m.nv) == (lite[0].nq, lite[0].nv) and w.world_id != lite[2].world_id
    # a crop with a tree config but no crop profile fails loudly
    (cdir / "lychee.yaml").unlink()
    with pytest.raises(gen.TreeConfigError):
        build("orchard_lite", NANDI_TREES_CONFIG=str(tdir), NANDI_CROPS_CONFIG=str(cdir))
