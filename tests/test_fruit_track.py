"""WP26 FRUIT SCOUT (MISSION-PRESETS 4; ORCHARD-MVP-SPEC 4.4): tree-frame tracking and dedupe over 8 views, stage per
crop profile (mango without skin colour), size, reachable vs MARK, harvest window = truth, bagged season.
ROS-free, < 10 s each."""
import copy
import time

import numpy as np
import pytest

from atharv_nandi.core.perception import fruit_track as ft
from atharv_nandi.core.perception import pest_detect as pd
from atharv_nandi.core.perception import ripeness as rp
from atharv_nandi.sim.world import compose, crops
from atharv_nandi.sim.world import trees as gen

CFG = pd.load_config()
PROFILES = crops.load_profiles()
TODAY = gen.DEFAULT_SIM_DATE


@pytest.fixture(scope="module")
def demo():
    return {p["tree_id"]: p for p in gen.world_plans(compose.load_field("orchard_demo", 7), 7)}


def _det(frame, p, ray_from, kind="fruit", sd=0.012, truth="f"):
    o = np.asarray(ray_from, float)
    d = ft.FruitDetection(frame, kind, np.asarray(p, float), np.asarray(p, float), 80.0, "low", "low", 0.8, {}, [0, 0, 0],
                          truth)
    d.origin, d.ray = o, pd.unit(d.p_tree - o)
    d.depth_m, d.sigma_depth_m, d.sigma_lat_m = float(np.linalg.norm(d.p_tree - o)), sd, 0.006
    return d


def test_one_fruit_seen_in_8_views_counts_once_and_triangulates():
    rng = np.random.default_rng(3)
    truth = np.array([0.3, -0.5, 0.6])
    other = np.array([0.3, -0.5, 0.75])                  # a neighbour 15 cm above
    trk = ft.TreeFrameTracker(0.04, 0.04, 2)
    for k in range(8):
        a = 2 * np.pi * k / 8
        o = np.array([2.0 * np.cos(a), 2.0 * np.sin(a), 1.0])
        dets = []
        for p, tid in ((truth, "a"), (other, "b")):
            ray = pd.unit(p - o)
            noisy = p + ray * rng.normal(0, 0.012) + rng.normal(0, 0.004, 3)
            dets.append(_det(k, noisy, o, truth=tid))
        trk.update(dets)
    tracks = trk.finalize()
    assert len(tracks) == 2
    a = next(t for t in tracks if t.truth() == "a")
    assert a.hits == 8 and np.linalg.norm(a.mean - truth) < 0.01
    # bags and fruit never associate
    trk2 = ft.TreeFrameTracker()
    trk2.update([_det(0, truth, [2, 0, 1])])
    trk2.update([_det(1, truth, [0, 2, 1], kind="bag")])
    assert len(trk2.tracks) == 2


def test_single_low_confidence_hits_are_dropped_confident_ones_kept():
    trk = ft.TreeFrameTracker(0.04, 0.04, 2, single_hit_conf_min=0.6)
    fp = _det(0, [0.5, 0.5, 1.0], [2, 0, 1])
    fp.conf = 0.4
    good = _det(1, [-0.5, 0.2, 0.7], [0, -2, 1])
    good.conf = 0.8
    trk.update([fp])
    trk.update([good])
    trk.update([])
    assert [t.dets[0] is good for t in trk.finalize()] == [True]


@pytest.mark.parametrize("world", ["orchard_demo", "orchard_lite"])
def test_dedupe_count_error_over_8_views(world):
    """MP 4 / O-09: visible count error <= 10 % over 8 views (aggregate over every fruiting tree and 4 seeds);
    per-tree runs over 10 % are reported (small trees: one fruit of 8 is 12.5 %)."""
    t0 = time.time()
    rep = pd.kpi_report(world, scout_seeds=(7, 11, 23, 26))["fruit_scout"]
    assert rep["count_error_aggregate"] <= 0.10
    assert rep["trees_over_10pct"] <= 0.05 * rep["tree_runs"] + 1e-9
    assert rep["stage_acc"] >= 0.9                           # SimTruth
    assert rep["size_mae_mm_low_mean"] <= 5.0
    assert rep["window_equal"].split("/")[0] == rep["window_equal"].split("/")[1]
    for crop, v in rep["stage_acc_by_crop"].items():
        assert v["simtruth"] >= 0.9, (crop, v)
    print(f"{world} FRUIT SCOUT: {rep}")
    assert time.time() - t0 < 10.0


def test_mango_stage_never_uses_skin_colour():
    m = PROFILES["mango"]
    base = {"shoulder_rise_mm": 3.2, "stem_end_colour": "yellowing"}
    st = rp.classify(m, base, 95)
    for junk in ({"skin_colour": "green"}, {"skin": "dark_green"}, {"peel_color": "red"}, {"rgba": [1, 0, 0, 1]},
                 {"skin_score": 0.0}, {"colour": "yellow"}):
        assert rp.classify(m, {**base, **junk}, 95) == st
    assert st[0] == "mature"
    assert rp.classify(m, {"shoulder_rise_mm": 0.4, "stem_end_colour": "green"}, 95)[0] == "immature"
    # one flipped cue alone does not flip the vote
    assert rp.classify(m, {"shoulder_rise_mm": 0.4, "stem_end_colour": "yellowing"}, 95)[0] == "immature"
    assert rp.classify(m, {"shoulder_rise_mm": 3.0, "stem_end_colour": "green"}, 95)[0] == "mature"
    # observations of a mango never carry a colour key
    rng = np.random.default_rng(0)
    f = {"cues": {"shoulder_rise_mm": 3.0, "stem_end_colour": "yellowing", "skin": "green"}}
    assert set(rp.observe_cues(f, m, rng)) == {"shoulder_rise_mm", "stem_end_colour"}
    # ColorBlob baseline: identical stage colours, so it cannot stage a mango
    assert rp.colorblob_stage(m, m["stages"][1]["rgba"]) == m["stages"][0]["id"]


def test_stage_rules_for_guava_papaya_banana():
    g, pp, b = PROFILES["guava"], PROFILES["papaya"], PROFILES["banana"]
    assert rp.classify(g, {"skin_score": 0.8})[0] == "mature" and rp.classify(g, {"skin_score": 0.2})[0] == "immature"
    assert rp.classify(pp, {"orange_streaks": 3})[0] == "colour_break"
    assert rp.classify(pp, {"orange_streaks": 0})[0] == "immature"
    assert rp.classify(b, {"fingers": "angular"}, 100)[2] == "ribbon_age"
    assert rp.classify(b, {"fingers": "round"})[0] == "mature"


def test_harvest_window_equals_truth_for_every_tree(demo):
    for p in demo.values():
        r = ft.scout(p, PROFILES[p["crop"]], TODAY, crown_gate_ok=True)
        truth = p["truth"]["harvest_window"]
        assert rp.window_equals(r.fruit["harvest_window"], truth), (p["tree_id"], r.fruit["harvest_window"], truth)
    blk = ft.scout_block(list(demo.values()), PROFILES, TODAY, crown_gate_ok=True)
    assert blk["harvest_window"]["from"] == min(p["truth"]["harvest_window"]["from"] for p in demo.values())
    assert blk["harvest_window"]["trees_open_n"] == 5            # the 5 mango trees, Dashehari mid-June
    assert "estimate" in blk["block_yield_label"] and blk["block_yield_kg"] > 0


def test_banana_window_from_the_ribbon_rota_when_the_registry_has_no_date(demo):
    p = copy.deepcopy(demo["R4-1"])
    sd = p["entry"]["shooting_date"]
    colour = p["truth"]["bunch"]["ribbon_colour"]
    p["entry"]["shooting_date"] = None
    no = ft.scout(p, PROFILES["banana"], TODAY, crown_gate_ok=True)
    assert (no.fruit["harvest_window"] or {}).get("basis") != "days_since_shooting"   # no date, no rota: unknown
    yes = ft.scout(p, PROFILES["banana"], TODAY, crown_gate_ok=True, rota={colour: sd})
    assert rp.window_equals(yes.fruit["harvest_window"], p["truth"]["bunch"]["harvest_window"])
    assert yes.marks[0]["kind"] == "bunch" and yes.marks[0]["tool_hint"] == "human_cut"


def test_window_without_registry_dates_falls_back_to_stage_then_calendar():
    m = PROFILES["mango"]
    w = rp.tree_window(m, {"cultivar": "Dashehari"}, {"mature": 3}, TODAY)
    assert w["basis"] == "stage_distribution" and w["open"] and w["from"] == TODAY
    w2 = rp.tree_window(m, {"cultivar": "Dashehari"}, {"immature": 3}, TODAY)
    assert w2["basis"] == "calendar_up" and w2["from"] == "2026-06-08"
    w3 = rp.tree_window(m, {"bagging_date": "2026-04-25"}, None, TODAY)
    assert w3["basis"] == "calendar"


def test_bagged_season_counts_bags_and_refuses_harvest():
    plans = gen.world_plans(compose.load_field("orchard_demo", 7), 7, season="bagged")
    p = next(x for x in plans if x["crop"] == "mango" and x["truth"]["bags_n"])
    r = ft.scout(p, PROFILES["mango"], TODAY, crown_gate_ok=True)
    assert r.fruit["bags_n"] >= 0.8 * r.assisted["truth_visible_n"] > 0
    assert r.fruit["harvest"] == {"refused": True, "reason": "bagged"}
    assert all(row["stage_basis"] == "calendar" for row in r.assisted["rows"] if row["kind"] == "bag")
    assert rp.harvest_refusal({"bagged": True}) == "bagged" and rp.harvest_refusal({}) is None
    assert not [m for m in r.marks if m["kind"] == "fruit"]          # no pick or MARK on bagged fruit


def test_reachable_vs_mark_and_record_shape(demo):
    p = demo["OT-1"]
    r = ft.scout(p, PROFILES["mango"], TODAY, crown_gate_ok=True)
    f = r.fruit
    for k in ("visible_by_stage", "bags_n", "size_mm_hist", "est_kg", "reachable_n", "visible_n", "method",
              "visible_fraction", "harvest_window"):
        assert k in f
    assert sum(f["visible_by_stage"].values()) == f["visible_n"]
    assert sum(f["size_mm_hist"]["counts"]) == f["visible_n"]
    truth_reach = sum(1 for x in p["truth"]["fruit"] if x["in_band"] and x["stage"] == "mature")
    reach_mature = sum(1 for row in r.assisted["rows"] if row["reachable"] and row["stage"] == "mature")
    assert reach_mature == truth_reach == 3                     # OS 1.2: OT-1 has 3 mature in-band mangoes
    assert r.marks and all(m["reason"] == "out_of_reach_marked" and m["tool_hint"] == "pole_picker"
                           and m["stage"] == "mature" and m["h_class"] in ("beech", "upar") for m in r.marks)
    assert f["est_kg"] > 0 and f["method"].startswith("simtruth_track")


def test_orbit_views_are_8_stations_by_3_bands(demo):
    vs = ft.orbit_views(demo["OT-1"])
    assert len(vs) == 24
    assert sorted({round(v.pos[2], 2) for v in vs}) == [0.98, 1.7, 2.7]
    assert len({v.tag["station"] for v in vs}) == 8


def test_cue_fusion_and_fallbacks():
    assert rp.fuse_cues([{"a": 1.0, "s": "x"}, {"a": 2.0, "s": "y"}, {"s": "y"}]) == {"a": 1.5, "s": "y"}
    m = PROFILES["mango"]
    assert rp.classify(m, {}) == ("immature", 0.0, "none")
    assert rp.classify(m, {}, 95)[0] == "immature"                  # days alone (weight 0.2) never makes mature
    assert rp.days_since({"bagging_date": "2026-04-25"}, m, TODAY) == 84.0
    assert rp.days_since({}, m, TODAY) is None and rp.days_since({}, m, None) is None
    assert rp.classify(PROFILES["banana"], {})[2] == "none"
    other = copy.deepcopy(PROFILES["guava"])
    other["crop"] = "litchi"                                         # a 5th crop by config: profile cues
    assert rp.classify(other, {"skin": "light_green_yellow"})[0] == "mature"
    assert rp.classify(other, {}, 120)[0] == "mature"
    rng = np.random.default_rng(1)
    assert rp.colorblob_stage(PROFILES["papaya"], PROFILES["papaya"]["stages"][1]["rgba"], rng) == "colour_break"
    obs = rp.observe_cues({"cues": {"orange_streaks": [3, 5]}}, PROFILES["papaya"], rng)
    assert 0 <= obs["orange_streaks"] <= 4
    assert rp.block_window([None]) is None and rp.window_equals(None, None) and not rp.window_equals({}, None)


def test_frames_precondition_stale_frames_and_crown_gate(demo):
    assert ft.frames_precondition("reuse_tree360", 10.0) == ("reuse_tree360", None)
    assert ft.frames_precondition("reuse_tree360", 30.0, crown_gate_ok=True) == ("one_lap", None)   # stale -> lap
    assert ft.frames_precondition("reuse_tree360", 30.0, crown_gate_ok=False) == ("reuse_tree360", "frames_stale")
    assert ft.frames_precondition("one_lap", None, crown_gate_ok=False) == ("one_lap", "crown_gate")
    assert ft.frames_precondition("one_lap", None, crown_gate_ok=True) == ("one_lap", None)
    assert ft.frames_precondition("lidar", None)[1] == "bad_source"
    # fail closed: an unknown crown gate is not green, an unknown / NaN / inf frame age is not fresh
    assert ft.frames_precondition("one_lap", None, None) == ("one_lap", "crown_gate_unknown")
    assert ft.frames_precondition("reuse_tree360", 48.0, None) == ("reuse_tree360", "frames_stale")
    for age in (None, float("nan"), float("inf"), "x", -1.0):
        why = ft.frames_precondition("reuse_tree360", age, False)[1]
        assert why in ("frames_age_unknown", "frames_stale"), (age, why)
        assert ft.frames_precondition("scan_bursts", age, None)[1] is not None
        assert ft.frames_precondition("reuse_tree360", age, True) == ("one_lap", None)
    p = demo["OT-1"]
    r = ft.scout(p, PROFILES["mango"], TODAY)                     # caller left the gate out: refused, no frames
    assert r.fruit == {"status": "refused", "reason": "crown_gate_unknown"} and r.frames == 0
    r = ft.scout(p, PROFILES["mango"], TODAY, frames_age_h=float("nan"), source="reuse_tree360")
    assert r.fruit == {"status": "refused", "reason": "frames_age_unknown"} and r.frames == 0
    r = ft.scout(p, PROFILES["mango"], TODAY, frames_age_h=30.0, crown_gate_ok=False, source="reuse_tree360")
    assert r.fruit == {"status": "refused", "reason": "frames_stale"} and r.frames == 0
    r = ft.scout(p, PROFILES["mango"], TODAY, frames_age_h=30.0, crown_gate_ok=True, source="reuse_tree360")
    assert r.fruit["method"] == "simtruth_track:one_lap" and r.fruit["count_label"] == "estimate"
    r = ft.scout(p, PROFILES["mango"], TODAY, frames_age_h=2.0, source="reuse_tree360")
    assert r.fruit["method"] == "simtruth_track:reuse_tree360"
    blk = ft.scout_block([demo["OT-1"], demo["R1-1"]], PROFILES, TODAY, crown_gate_ok=False)
    assert blk["refused"] == {"OT-1": "crown_gate", "R1-1": "crown_gate"} and blk["harvest_window"] is None
