"""WP26 TRUNK/BASE SCAN (MISSION-PRESETS 3b): azimuths, wrist close-ups at 0.2/0.5/0.8 m + ring low frame, trunk
leads per crop and season, TreeRecord trunk{}, rootstock-sprout PRUNE flags, clearance. ROS-free, < 10 s each."""
import math
import time

import numpy as np
import pytest

from atharv_nandi.core.perception import pest_detect as pd
from atharv_nandi.core.perception import trunk_scan as ts
from atharv_nandi.sim.world import compose, crops
from atharv_nandi.sim.world import trees as gen

CFG = pd.load_config()
PROFILES = crops.load_profiles()


@pytest.fixture(scope="module")
def demo():
    return {p["tree_id"]: p for p in gen.world_plans(compose.load_field("orchard_demo", 7), 7)}


@pytest.fixture(scope="module")
def lite():
    return {p["tree_id"]: p for p in gen.world_plans(compose.load_field("orchard_lite", 7), 7)}


def test_four_azimuths_full_two_in_hedgerow(demo, lite):
    azs, mode = ts.azimuths(demo["OT-1"])
    assert mode == "full" and len(azs) == 4
    assert np.allclose(np.diff(azs), math.pi / 2)
    for tid in ("L1-1", "L1-2"):                       # WP4 3 x 2 m mango hedgerow pair
        assert ts.is_hedgerow(lite[tid])
        azs, mode = ts.azimuths(lite[tid])
        assert mode == "two_sided" and len(azs) == 2
        assert abs(abs(azs[1] - azs[0]) - math.pi) < 1e-9


def test_views_tile_the_base_band_and_keep_2cm(demo, lite):
    t0 = time.time()
    for p in list(demo.values()) + list(lite.values()):
        views, mode = ts.plan_views(p)
        wrist = [v for v in views if v.kind == "wrist"]
        ring = [v for v in views if v.kind == "ring_low"]
        n_az = 4 if mode == "full" else 2
        assert len(ring) == n_az
        hs = sorted({round(v.h_m, 3) for v in wrist})
        assert hs[0] == pytest.approx(0.2) and max(hs) <= 0.8 + 1e-9 and len(hs) >= 2
        for v in wrist:
            assert v.ok and v.clearance_m >= 0.02, (p["tree_id"], v.row())
            # portrait wrist frame at the stand-off: vertical half-cover >= half the 0.3 m height spacing
            depth = float(np.linalg.norm(v.view.pos - v.view.tag["aim"])) - ts.trunk_geometry(p)[0]
            half = depth * math.tan(math.radians(v.view.fovy_deg) / 2.0)
            assert half >= 0.15 - 1e-3
            assert 0.15 - 1e-9 <= depth <= 0.25 + 1e-9                # MP 3b: wrist at 0.15-0.25 m
    assert time.time() - t0 < 10.0


@pytest.mark.parametrize("world", ["orchard_demo", "orchard_lite"])
def test_trunk_lead_recall(world):
    """MP 3b KPI: trunk-class lead recall >= 0.9 (SimTruth) over the decals the planned views can image; the
    all-sides recall (hedgerow in-row faces included) is reported."""
    rep = pd.kpi_report(world, scout_seeds=())["trunk_scan"]
    assert rep["lead_recall_accessible"] >= 0.9 and rep["accessible_n"] >= 10
    assert all(n == (4 if mode == "full" else 2) for _, mode, n in rep["azimuths"])
    print(f"{world} TRUNK SCAN: recall accessible {rep['lead_recall_accessible']} (n {rep['accessible_n']}), "
          f"all sides {rep['lead_recall_all_sides']} (n {rep['all_n']})")


def test_trunk_record_keys_units_and_stage(demo):
    want = {"mango": {"gum", "frass", "mealybug"}, "guava": {"wilt", "frass"}, "papaya": {"collar_rot"},
            "banana": {"weevil_holes"}}
    seen = {c: set() for c in want}
    for p in demo.values():
        r = ts.run_trunk_scan(p, PROFILES[p["crop"]])
        assert set(r.trunk) >= {"frass", "gum", "collar_rot", "mud_tubes", "weevil_holes", "band_ok"}
        assert r.trunk["stage"] == "lead" and r.trunk["band_ok"] is None      # no band decal class in WP23 yet
        for s in r.symptoms:
            assert s["unit"] == "trunk" and s["stage"] == "lead" and s["h_m"] <= 1.6     # ring frame: up to 1.5 m weevil gum
        seen[p["crop"]] |= {k for k in want[p["crop"]] if r.trunk[k]}
        assert r.time_s > 0
    for c in want:
        assert seen[c] == want[c], (c, seen[c])


def test_rootstock_sprouts_below_the_graft_union_flag_prune(demo, lite):
    n = 0
    for p in list(demo.values()) + list(lite.values()):
        r = ts.run_trunk_scan(p, PROFILES[p["crop"]])
        gu = p["truth"].get("graft_union_h_m") or p["entry"].get("graft_union_h_m")
        cand = ts.sprouts_below_graft(p)
        for t in cand:
            assert min(t["a"][2], t["b"][2]) < gu
        flagged = {f["prune_id"] for f in r.prune_flags}
        assert flagged <= {t["id"] for t in cand}
        assert len(flagged) == len(cand), (p["tree_id"], flagged, [t["id"] for t in cand])
        for f in r.prune_flags:
            assert f["flag"] == "prune" and f["kind"] == "rootstock_sprout" and f["h_class"] == "neeche"
        n += len(flagged)
    assert n >= 8


def test_season_changes_the_trunk_classes():
    mango = PROFILES["mango"]
    cls = lambda s: {c["class"] for c in pd.active_classes(mango, s, scale="trunk")}   # noqa: E731
    assert "mealybug" in cls("panicle") and "mealybug" in cls("fruit_set")        # Dec-Jan trunk crawl
    assert "mealybug" not in cls("post_harvest")
    assert {"stem_borer_frass", "gummosis"} <= cls("post_harvest")                 # all seasons
    p = gen.world_plans(compose.load_field("orchard_demo", 7), 7, season="post_harvest")[0]
    r = ts.run_trunk_scan(p, mango)
    assert all(s["class"] != "mealybug" for s in r.symptoms)
    assert not r.trunk["mealybug"]


def test_blocked_wrist_pose_is_dropped_with_no_clearance(demo):
    p = dict(demo["OT-1"])
    p["geoms"] = list(p["geoms"]) + [{"name": "tree_OT-1_fruit_99", "kind": "fruit", "type": "sphere", "group": 1,
                                      "pos": [0.0, -0.06 - 0.20, 0.5], "size": [0.20, 0, 0]}]
    views, _ = ts.plan_views(p)
    bad = [v for v in views if v.kind == "wrist" and not v.ok]
    assert bad and all(v.reason == "no_clearance" for v in bad)
    r = ts.run_trunk_scan(p, PROFILES["mango"])
    assert r.azimuths_imaged >= 3


def test_trunk_scan_interlocks_refuse_with_reasons(demo):
    r = ts.run_trunk_scan(demo["OT-1"], PROFILES["mango"], ctx={"cross_slope_deg": 5.0, "tool": "T2"})
    assert r.mode == "refused" and r.trunk == {"status": "refused", "reasons": ["dwell_slope", "blade_not_locked_open"]}
    assert r.views == [] and r.leads == []
    # papaya has no leaves in band, but TRUNK SCAN still applies (collar rot)
    ok = ts.run_trunk_scan(demo["R3-1"], PROFILES["papaya"], ctx={"cross_slope_deg": 2.0})
    assert ok.mode in ("full", "two_sided") and ok.azimuths_imaged >= 2


def test_block_sample_trunk_scan_stratified_per_crop(demo):
    plans = list(demo.values())
    out = ts.run_trunk_block(plans, PROFILES)
    agg = CFG["aggregation"]["block_sample_n"]
    for crop, sp in out["sampling"].items():
        n_crop = sum(1 for p in plans if p["crop"] == crop)
        assert sp["n"] == min(n_crop, agg.get(crop, agg["default"])) and sp["basis"] == "stratified_by_direction"
        assert out["block"][crop]["trees_n"] == sp["n"] and out["block"][crop]["stage"] == "lead"
        assert out["block"][crop]["sampling_plan"] is sp
        for v in out["block"][crop]["pct_trees_affected"].values():
            assert 0.0 < v <= 100.0
    assert set(out["trees"]) == {t for sp in out["sampling"].values() for t in sp["tree_ids"]}
    assert all(t["stage"] == "lead" for t in out["trees"].values())
    assert out == ts.run_trunk_block(plans, PROFILES)                        # deterministic
    two = ts.run_trunk_block(plans, PROFILES, n=2)
    assert all(sp["n"] <= 2 for sp in two["sampling"].values())


def test_low_leaf_in_the_wrist_path_is_an_obstacle_as_leaf_scan(demo):
    """MP 3b interlocks 'as LEAF SCAN': wrist-to-leaf >= 2 cm. A low-skirted tree with a leaf quad on the
    straight wrist approach at 0.5 m forces the stand-off / yaw search or drops the view; no kept wrist view comes within
    2 cm of it."""
    from atharv_nandi.core.perception import leaf_scan as lsc
    base_views, _ = ts.plan_views(demo["OT-1"])
    tgt = next(v for v in base_views if v.kind == "wrist" and v.ok and abs(v.h_m - 0.5) < 1e-6)
    mid = tgt.view.pos - tgt.view.fwd * 0.06 - np.asarray(demo["OT-1"]["pos"], float)
    p = dict(demo["OT-1"])
    leaf = {"name": "tree_OT-1_leaf_low_99", "kind": "leaf", "type": "box", "group": 1,
            "pos": [float(x) for x in mid], "size": [0.04, 0.04, 0.002]}
    p["geoms"] = list(p["geoms"]) + [leaf]
    views, _ = ts.plan_views(p)
    obs = lsc.tree_obstacles(p)
    assert any(o.name == leaf["name"] for o in obs)
    proxy = lsc.WristProxy(**CFG["leaf_scan"]["wrist_proxy"])
    for v in views:
        if v.kind == "wrist" and v.ok:
            clr, _ = lsc.approach_clearance(obs, proxy, v.view.pos, v.view.fwd, CFG["leaf_scan"]["approach_m"])
            assert clr >= 0.02 - 1e-6, (v.azimuth_deg, v.h_m, clr)
    same = next(v for v in views if v.kind == "wrist" and v.azimuth_deg == tgt.azimuth_deg and v.h_m == tgt.h_m)
    assert not (same.ok and np.allclose(same.view.pos, tgt.view.pos))      # the blocked pose moved or was dropped
