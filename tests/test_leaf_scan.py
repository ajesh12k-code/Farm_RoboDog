"""WP26 LEAF SCAN (MISSION-PRESETS 3; ORCHARD-MVP-SPEC 2.5, 3.3): leaf selection, clearance-keeping upper and
no-contact underside views, footprint, leads, KPIs. ROS-free; unit tests well under 10 s each; the ``gl`` tests
render the composed orchard_demo world through a wrist-camera proxy and skip cleanly without a GL backend.
"""
import contextlib
import copy
import math
import os
import time

import mujoco
import numpy as np
import pytest

from atharv_nandi.core.perception import leaf_scan as ls
from atharv_nandi.core.perception import pest_detect as pd
from atharv_nandi.sim.world import compose, crops
from atharv_nandi.sim.world import trees as gen

CFG = pd.load_config()
PROFILES = crops.load_profiles()


@pytest.fixture(scope="module")
def demo_plans():
    return {p["tree_id"]: p for p in gen.world_plans(compose.load_field("orchard_demo", 7), 7)}


@pytest.fixture(scope="module")
def lite_plans():
    return {p["tree_id"]: p for p in gen.world_plans(compose.load_field("orchard_lite", 7), 7)}


# ---------------------------------------------------------------------------------------------- config
def test_config_validates_and_refuses_a_clearance_below_2cm(tmp_path):
    assert pd.validate_config(CFG) == []
    bad = copy.deepcopy({k: v for k, v in CFG.items() if k != "_path"})
    bad["leaf_scan"]["min_clearance_m"] = 0.015
    import yaml
    p = tmp_path / "po.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(pd.PerceptionConfigError, match="min_clearance_m"):
        pd.load_config(str(p))
    bad2 = copy.deepcopy(bad)
    bad2["leaf_scan"]["min_clearance_m"] = 0.02
    bad2["classes"]["sooty_mould"]["unit"] = "per_tree"
    p.write_text(yaml.safe_dump(bad2))
    with pytest.raises(pd.PerceptionConfigError, match="unit"):
        pd.load_config(str(p))


@pytest.mark.parametrize("sec,key,val,match", [
    ("trunk_scan", "min_clearance_m", 0.0, "trunk_scan.min_clearance_m"),
    ("trunk_scan", "min_clearance_m", None, "trunk_scan.min_clearance_m"),
    ("leaf_scan", "min_clearance_m", float("nan"), "leaf_scan.min_clearance_m"),
    ("leaf_scan", "drift_max_m", 0.5, "drift_max_m"),
    ("leaf_scan", "drift_max_m", 0.0, "drift_max_m"),
    ("leaf_scan", "blade_tools", [], "blade_tools"),
    ("leaf_scan", "blade_tools", ["T2"], "blade_tools"),
    ("leaf_scan", "max_footprint_mm_px", 2.0, "max_footprint_mm_px"),
    ("fruit_track", "frames_max_age_h", 72, "frames_max_age_h"),
    ("fruit_track", "frames_max_age_h", float("inf"), "frames_max_age_h"),
])
def test_config_refuses_an_override_that_loosens_an_interlock(sec, key, val, match):
    """An override (``NANDI_PERCEPTION_ORCHARD``) may not switch off the TRUNK SCAN 2 cm rule, the drift pause,
    the blade interlock, the O-08 footprint or the MP 4 24 h frame age."""
    bad = copy.deepcopy({k: v for k, v in CFG.items() if k != "_path"})
    bad[sec][key] = val
    errs = pd.validate_config(bad)
    assert any(match in e for e in errs), errs


# ---------------------------------------------------------------------------------------------- selection
def test_papaya_is_n_a_crown_only_and_banana_only_in_band_dead_blades(demo_plans, lite_plans):
    for tid in ("R3-1", "R3-2", "R3-3", "R3-4"):
        r = ls.run_leaf_scan(demo_plans[tid], PROFILES["papaya"])
        assert r.status == ls.N_A == "n/a_crown_only" and r.leaf_block == {"status": "n/a_crown_only"}
        assert r.records == [] and r.leads == []
    for tid in ("R4-1", "R4-2", "R4-3", "R4-4"):
        sel = ls.select_leaves(demo_plans[tid])
        assert sel["status"] == "ok" and sel["close_up"]
        assert all(not lf["functional"] and lf["in_band"] for lf in sel["close_up"])
    # WP4's orchard_lite banana has no in-band dead blade: n/a (MP 3 precondition)
    assert ls.select_leaves(lite_plans["L2-4"])["status"] == "n/a_crown_only"


def test_mango_selection_is_stratified_capped_and_blind_to_decal_truth(demo_plans):
    p = demo_plans["OT-1"]
    sel = ls.select_leaves(p)
    inb = [lf for lf in p["truth"]["leaves"] if lf["in_band"]]
    assert len(sel["close_up"]) == len(inb) == 8
    assert len(sel["close_up"]) + len(sel["not_close_up"]) <= CFG["leaf_scan"]["n_leaves"] == 12
    assert all(lf["world"][2] <= 0.8 for lf in sel["close_up"])
    assert all(lf["world"][2] > 0.8 or not lf["in_band"] for lf in sel["not_close_up"])
    quads = {ls._quadrant(p, lf["world"]) for lf in sel["close_up"][:4]}
    assert len(quads) >= 3, quads                     # the first picks spread over azimuth quadrants
    kinds = {lf["kind"] for lf in sel["close_up"]}
    assert kinds == {"young", "mature"}
    demo = ls.select_leaves(p, demo=True)
    assert len(demo["close_up"]) == 6 and len(demo["not_close_up"]) == 1
    # decal truth never steers the selection
    q = copy.deepcopy(p)
    for lf in q["truth"]["leaves"]:
        lf["decal"] = None
    q["truth"]["decals"] = []
    assert [lf["id"] for lf in ls.select_leaves(q)["close_up"]] == [lf["id"] for lf in sel["close_up"]]
    # a flagged branch pin puts the nearest leaf first
    target = sel["close_up"][-1]
    br = ls.select_leaves(p, n_leaves=2, branch={"clock": pd.clock_of_world(p, target["world"]), "h_m": target["world"][2]})
    assert br["close_up"][0]["id"] == target["id"]


# ---------------------------------------------------------------------------------------------- geometry
def test_segment_distance_and_box_distance():
    assert ls.seg_seg_dist([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]) == pytest.approx(1.0)
    assert ls.seg_seg_dist([0, 0, 0], [1, 0, 0], [0.5, -1, 1], [0.5, 1, 1]) == pytest.approx(1.0)
    assert ls.seg_seg_dist([0, 0, 0], [0, 0, 0], [3, 4, 0], [3, 4, 0]) == pytest.approx(5.0)
    assert ls.seg_seg_dist([0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]) == pytest.approx(1.0)
    box = ls.Obstacle("box", "b", "leaf", c=np.zeros(3), R=np.eye(3), half=np.array([0.05, 0.02, 0.0015]))
    assert box.dist_to_capsule([0, 0, 0.1], [0, 0, 0.2], 0.025) == pytest.approx(0.1 - 0.0015 - 0.025, abs=1e-6)
    s = ls.ObstacleSet([box, ls.Obstacle("sphere", "s", "fruit", c=np.array([0, 0, 0.5]), r=0.04)])
    d, who = s.min_dist([0, 0, 0.3], [0, 0, 0.35], 0.0)
    assert who == "s" and d == pytest.approx(0.11)


def test_footprint_is_0_72_mm_per_px_at_0_15_m():
    v = pd.View(pos=[0, 0, 0], fwd=[0, 0, -1], fovy_deg=60.0, width=320, height=240)
    assert v.mm_per_px(0.15) == pytest.approx(0.7217, abs=1e-3)       # OS 2.5: 0.72 mm/px (computed)
    assert pd.disc_px(v, [0, 0, -0.15], [0, 0, 1], 0.012) == pytest.approx(16.6, abs=0.1)
    assert pd.disc_px(v, [0, 0, -0.15], [0, 0, -1], 0.012) == 0.0       # back side: hidden by its own leaf
    u, vv, z = v.project([0.01, 0.02, -0.15])
    assert np.allclose(v.unproject(u, vv, z), [0.01, 0.02, -0.15])


def test_views_keep_2cm_and_underside_looks_up_from_below(demo_plans):
    t0 = time.time()
    n = 0
    for tid in ("OT-1", "R1-1", "R2-1", "R4-1"):
        p = demo_plans[tid]
        obs = ls.tree_obstacles(p)
        for lf in ls.select_leaves(p)["close_up"]:
            nup = pd.unit(lf["n"]) * (1 if lf["n"][2] >= 0 else -1)
            for side in ("upper", "under"):
                lv = ls.plan_view(lf, side, obs)
                assert lv.ok, (lf["id"], side, lv.reason, lv.clearance_m)
                assert lv.clearance_m >= 0.02
                lo, hi = CFG["leaf_scan"]["standoff_range_m"]
                assert lo - 1e-9 <= lv.standoff_m <= hi + 1e-9
                assert lv.mm_per_px <= CFG["leaf_scan"]["max_footprint_mm_px"]
                rel = np.asarray(lf["world"]) - lv.view.pos
                if side == "under":
                    assert lv.view.fwd[2] >= 0.2 and float(rel @ nup) > 0      # below the leaf, looking up
                    ang = math.degrees(math.acos(min(1.0, float(lv.view.fwd @ nup))))
                    assert ang <= CFG["leaf_scan"]["view_cone_deg"] + 1e-6
                else:
                    assert float(rel @ nup) < 0
                n += 1
    assert n >= 2 * (8 + 8 + 8 + 2)
    assert time.time() - t0 < 10.0


def test_blocked_view_moves_off_axis_or_marks_no_clearance(demo_plans):
    p = demo_plans["OT-1"]
    lf = ls.select_leaves(p)["close_up"][0]
    obs = ls.tree_obstacles(p)
    free = ls.plan_view(lf, "under", obs)
    # a fruit-sized sphere exactly on the preferred axis: the planner must pick another axis / stand-off
    nup = pd.unit(lf["n"]) * (1 if lf["n"][2] >= 0 else -1)
    blocker = ls.Obstacle("sphere", "blocker", "fruit", c=np.asarray(lf["world"]) - nup * 0.15, r=0.03)
    alt = ls.plan_view(lf, "under", ls.ObstacleSet(list(obs) + [blocker]))
    assert alt.ok and alt.clearance_m >= 0.02
    assert not np.allclose(alt.view.pos, free.view.pos)
    # a shell of blockers around the leaf: nothing keeps 2 cm -> no_clearance, the leaf becomes a MARK
    shell = [ls.Obstacle("sphere", "shell", "fruit", c=np.asarray(lf["world"]), r=0.30)]
    bad = ls.plan_view(lf, "under", ls.ObstacleSet(list(obs) + shell))
    assert not bad.ok and bad.reason == "no_clearance"


def test_clearance_guard_latches_and_drift_pauses(demo_plans):
    g = ls.ClearanceGuard(0.02)
    assert g.update(0.05) == "ok" and g.update(0.03) == "ok"
    assert g.update(0.019, t=1.0, who="tree_OT-1_leaf_3") == "stop"
    assert g.update(0.05) == "stop"                       # latched until the caller retracts and resets
    assert g.log[0]["event"] == "clearance_violation" and g.log[0]["action"] == "retract"
    assert g.base_drift(0.025) == "pause_reapproach" and g.base_drift(0.01) == "ok"
    g.reset()
    assert g.update(0.05) == "ok"
    # a physics caller feeding a measured violation turns that side into no_clearance (MP 3 failure handling)
    p = demo_plans["OT-1"]
    r = ls.run_leaf_scan(p, PROFILES["mango"], guard_feed=lambda lv: [0.05, 0.015] if lv.side == "under" else [0.05])
    assert all(s["ok"] == (s["side"] == "upper") for rec in r.records if rec["close_up"] for s in rec["sides"])
    assert r.leaf_block["underside_n"] == 0
    assert any(e["event"] == "clearance_violation" for e in r.log)


# ---------------------------------------------------------------------------------------------- KPIs
@pytest.mark.parametrize("world", ["orchard_demo", "orchard_lite"])
def test_leaf_scan_kpis(world):
    """O-08: imaged/planned >= 90 %, underside >= 40 %, footprint <= 1.0 mm/px, median <= 10 s/leaf,
    min clearance >= 2 cm, lead recall >= 0.9 (SimTruth, colony/symptom level)."""
    t0 = time.time()
    rep = pd.kpi_report(world, scout_seeds=())["leaf_scan"]
    assert rep["imaged_frac"] >= 0.90 and rep["underside_frac"] >= 0.40
    assert rep["footprint_mm_px_max"] <= 1.0
    assert rep["median_s_per_leaf"] <= 10.0
    assert rep["min_clearance_m"] >= 0.02
    assert rep["lead_recall"] >= 0.9 and rep["leads_truth_n"] >= 8
    print(f"{world} LEAF SCAN: {rep}")
    assert time.time() - t0 < 10.0


def test_leaf_record_shape_and_labels(demo_plans):
    r = ls.run_leaf_scan(demo_plans["OT-1"], PROFILES["mango"], demo=True)
    b = r.leaf_block
    assert b["status"] == "ok" and b["label"] == "colony_symptom_level"
    assert b["n"] == 6 and b["not_close_up_n"] == 1 and b["underside_n"] == 6
    assert all(f["stage"] == "lead" for f in b["findings"])
    hi = [rec for rec in r.records if not rec["close_up"]]
    assert len(hi) == 1 and hi[0]["label"] == "not_close_up" and hi[0]["sides"][0]["side"] == "crown"
    for ld in r.leads:
        assert ld["stage"] == "lead" and ld["label"] in ("colony_symptom_level", "not_close_up")
    frag = pd.record_fragment(demo_plans["OT-1"], pd.dedupe_leads(r.leads), leaf=b, did=["leaf_scan"])
    assert frag["leaf"]["underside_n"] == 6
    assert all(a in pd.NEXT_ACTIONS or a.startswith("mark_jobs:") for a in frag["next_actions"])


# ---------------------------------------------------------------------------------------------- physics cross-check
@contextlib.contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update(kw)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def demo_world():
    """orchard_demo composed with a mocap wrist proxy: camera (fovy 60) + a swept capsule (contype 0) for
    ``mj_geomDistance``. The model's near plane is set to 2 cm (the perception model's own setting)."""
    from atharv_nandi.core import farm_scene
    from atharv_nandi.sim.world.modules import trees as mt
    with _env(NANDI_WORLD="orchard_demo"):
        cfg = farm_scene.world_config()
        w = compose.build(cfg["world"], cfg["seed"], farm_scene.ROBOT_XML, cfg)
        plans = {p["tree_id"]: p for p in mt.current_plans()}
        b = w.spec.worldbody.add_body(name="wp26_wrist", mocap=True, pos=[0, 0, -5])
        b.add_camera(name="wp26_wrist_cam", fovy=60.0)
        half = (CFG["leaf_scan"]["approach_m"] + CFG["leaf_scan"]["wrist_proxy"]["len_m"]) / 2.0
        b.add_geom(name="wp26_swept", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   size=[CFG["leaf_scan"]["wrist_proxy"]["r_m"], half, 0], pos=[0, 0, half],
                   contype=0, conaffinity=0, group=5, rgba=[1, 0, 1, 0])
        m = w.spec.compile()
    m.vis.map.znear = 0.02 / m.stat.extent
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    return m, d, plans


def _place(m, d, view):
    mid = int(m.body_mocapid[m.body("wp26_wrist").id])
    d.mocap_pos[mid] = view.pos
    d.mocap_quat[mid] = view.quat()
    mujoco.mj_forward(m, d)


def test_planned_views_keep_2cm_in_mujoco_geom_distance(demo_world):
    """The analytic clearance (leaf boxes, wood / prune capsules, fruit bounding spheres) against MuJoCo's
    ``mj_geomDistance`` of the swept wrist capsule (approach + housing) to every geom of the tree in groups 0, 1,
    3 (leaf quads; the merged foliage mesh is visual and excluded) and 4: >= 2 cm everywhere and the analytic value
    is never above MuJoCo's by more than 1 mm (conservative)."""
    m, d, plans = demo_world
    t0 = time.time()
    worst = []
    for tid in ("OT-1", "R1-2", "R2-1", "R4-2"):
        p = plans[tid]
        obs = ls.tree_obstacles(p)
        body = m.body(p["body"]).id
        gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_group[g] in (0, 1, 3, 4)
                and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
        sw = m.geom("wp26_swept").id
        fromto = np.zeros(6)
        for lf in ls.select_leaves(p)["close_up"]:
            for side in ("upper", "under"):
                lv = ls.plan_view(lf, side, obs)
                _place(m, d, lv.view)
                dmin = min(mujoco.mj_geomDistance(m, d, sw, g, 0.5, fromto) for g in gids)
                assert dmin >= 0.02 - 1e-4, (lf["id"], side, dmin)
                assert lv.clearance_m <= dmin + 1e-3, (lf["id"], side, lv.clearance_m, dmin)
                worst.append(dmin)
    print(f"mj_geomDistance clearance: min {min(worst):.4f} m over {len(worst)} planned views")
    assert time.time() - t0 < 10.0


def _gl_ok():
    from test_pano import gl_available
    return gl_available()


@pytest.mark.gl
def test_wrist_footprint_on_a_12mm_decal_and_colorblob_reported(demo_world):
    """WP26 ``gl``: the rendered wrist frame (320 x 240, fovy 60) of each in-band leaf decal from its planned
    upper view spans >= 12 px for a 12 mm decal (segmentation pixels, equivalent diameter scaled to 12 mm);
    the analytic prediction agrees within 15 %; ColorBlob on the same RGB frames is reported, not gated.
    Underside frames see the leaf, never the decal (WP23 decals sit on the upper face)."""
    if not _gl_ok():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    m, d, plans = demo_world
    W, H = CFG["cameras"]["wrist"]["width"], CFG["cameras"]["wrist"]["height"]
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    for g in range(6):
        opt.geomgroup[g] = 1 if g <= 4 else 0
    cid = m.camera("wp26_wrist_cam").id
    cb = pd.ColorBlobLeadDetector()
    rows, cb_hits, n = [], 0, 0
    try:
        for tid in ("OT-1", "R1-1", "R1-2", "R1-3", "R1-4", "R2-1", "R2-2"):
            p = plans[tid]
            obs = ls.tree_obstacles(p)
            classes = [k for k in pd.active_classes(PROFILES[p["crop"]], p["season"]) if not k["distance_only"]]
            normals = pd.decal_normals(p)
            for lf in p["truth"]["leaves"]:
                if not lf["decal"] or not lf["in_band"]:
                    continue
                dec = next(x for x in p["truth"]["decals"] if x.get("leaf") == lf["id"])
                gid = m.geom(dec["geom"]).id
                for side in ("upper", "under"):
                    lv = ls.plan_view(lf, side, obs)
                    _place(m, d, lv.view)
                    r.update_scene(d, camera=cid, scene_option=opt)
                    r.enable_segmentation_rendering()
                    seg = r.render()
                    r.disable_segmentation_rendering()
                    mask = (seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                    leaf_px = int((seg[..., 0] == m.geom(lf["geom"]).id).sum())
                    if side == "under":
                        assert mask.sum() == 0 and leaf_px > 1000
                        continue
                    eq = 2.0 * math.sqrt(mask.sum() / math.pi)
                    px12 = eq * 12.0 / dec["diameter_mm"]
                    pred = pd.disc_px(lv.view, dec["world"], normals[dec["geom"]], dec["diameter_mm"] / 1000.0)
                    assert px12 >= CFG["leaf_scan"]["min_decal_px"], (lf["id"], px12)
                    assert abs(eq - pred) <= 0.15 * pred, (lf["id"], eq, pred)
                    rgb = r.render()
                    blobs = cb.detect(rgb, classes, background_rgb=[[0.18, 0.50, 0.18], [0.45, 0.30, 0.18]])
                    ys, xs = np.nonzero(mask)
                    hit = any(b["class"] == dec["class"] and xs.min() - 2 <= b["u"] <= xs.max() + 2
                              and ys.min() - 2 <= b["v"] <= ys.max() + 2 for b in blobs)
                    cb_hits += int(hit)
                    n += 1
                    rows.append((lf["id"], dec["class"], round(dec["diameter_mm"], 1), round(eq, 1), round(px12, 1)))
    finally:
        r.close()
    assert n >= 10
    print(f"wrist decal footprint (id, class, mm, px, px@12mm): {rows}")
    print(f"ColorBlob (reported, not gated): class-correct recall {cb_hits}/{n} = {cb_hits / n:.2f}")


# ---------------------------------------------------------------------------------------------- preconditions, recovery
def test_preconditions_and_interlocks_give_reason_codes(demo_plans):
    ot1 = demo_plans["OT-1"]
    assert ls.check_preconditions({}, ot1) == []
    assert ls.check_preconditions({"cross_slope_deg": 3.0}) == []           # WP6 dwell gate: <= 3 deg for 60 s
    assert ls.check_preconditions({"cross_slope_deg": -3.5}) == ["dwell_slope"]
    assert ls.check_preconditions({"wet_foliage": True}) == ["wet_foliage"]
    assert ls.check_preconditions({"work_fields_clear": False}) == ["work_fields_not_clear"]
    assert ls.check_preconditions({"wrist_camera_ok": False}) == ["wrist_camera_fault"]
    for tool in ("T2", "T5"):                                                # CLIPPER / PRUNER blade locked open
        assert ls.check_preconditions({"tool": tool}) == ["blade_not_locked_open"]
        assert ls.check_preconditions({"tool": tool, "blade_locked_open": True}) == []
    assert ls.check_preconditions({"tool": "T4"}) == []                      # TWIST has no blade
    assert ls.check_preconditions({"light_lux": 500}) == ["low_light"]
    assert ls.check_preconditions({"light_lux": 500, "dawn_mode": True}) == []
    assert ls.check_preconditions({"light_lux": 500, "hour": 6.5}) == []     # before 08:00 -> dawn threshold
    assert ls.check_preconditions({"light_lux": 100, "dawn_mode": True}) == ["low_light"]
    assert ls.check_preconditions({}, demo_plans["R3-1"]) == [ls.N_A]        # papaya
    assert set(ls.check_preconditions({"wet_foliage": True, "tool": "T2"})) <= set(ls.PRECONDITION_REASONS)
    r = ls.run_leaf_scan(ot1, PROFILES["mango"], ctx={"wet_foliage": True, "cross_slope_deg": 4.0})
    assert r.status == "refused" and r.leaf_block == {"status": "refused", "reasons": ["dwell_slope", "wet_foliage"]}
    assert r.views == [] and r.leads == []
    ok = ls.run_leaf_scan(ot1, PROFILES["mango"], ctx={"cross_slope_deg": 1.0, "tool": "T5",
                                                       "blade_locked_open": True}, dawn_mode=True)
    assert ok.status == "ok" and ok.leaf_block["led"] == "on" and ok.leaf_block["dawn_mode"] is True


def test_blur_retry_backoff_and_drift_reapproach(demo_plans):
    """MP 3 failure / recovery: blur -> retry 2; depth < min-Z -> back off 3 cm (clearance re-checked); base drift >
    2 cm -> pause and re-approach. Past the limits the side is not imaged and the reason is logged."""
    ot1 = demo_plans["OT-1"]
    base = ls.run_leaf_scan(ot1, PROFILES["mango"])
    first = {}

    def close_ok(res):
        return sum(1 for lv in res.views if lv.ok and lv.side in ("upper", "under"))
    n_close = close_ok(base)
    assert n_close >= 8 and base.leaf_block["not_close_up_n"] >= 1

    def blur_twice(lv, attempt):
        return {"blurred": attempt < 2}
    r = ls.run_leaf_scan(ot1, PROFILES["mango"], capture_feed=blur_twice)
    assert close_ok(r) == n_close
    sides = [s for rec in r.records if rec["close_up"] for s in rec["sides"] if s["ok"]]
    assert sides and all(s["retries"] == 2 for s in sides)
    assert sum(1 for e in r.log if e["event"] == "blur_retry") == 2 * len(sides)
    assert r.leaf_block["median_s_per_leaf"] > base.leaf_block["median_s_per_leaf"]

    r = ls.run_leaf_scan(ot1, PROFILES["mango"], capture_feed=lambda lv, a: {"blurred": True})
    assert close_ok(r) == 0 and r.leaf_block["not_imaged"]["blurred"] == n_close
    assert all(ld["side"] == "crown" for ld in r.leads)
    # blurred is reachable: a robot revisit with the side reason, never an out_of_reach_marked human job
    blurred_leaves = [x["leaf_id"] for x in r.records if x["close_up"] and all(s["reason"] == "blurred" for s in x["sides"])]
    assert blurred_leaves and not [m for m in r.marks if m["leaf_id"] in blurred_leaves]
    assert {v["leaf_id"] for v in r.revisit} >= set(blurred_leaves) and r.leaf_block["revisit_leaves"]
    assert all("blurred" in v["side_reasons"].values() for v in r.revisit if v["leaf_id"] in blurred_leaves)

    def too_close_once(lv, attempt):
        key = (lv.leaf_id, lv.side)
        first.setdefault(key, lv.standoff_m)
        return {"depth_m": 0.08 if attempt == 0 else lv.standoff_m}
    r = ls.run_leaf_scan(ot1, PROFILES["mango"], capture_feed=too_close_once)
    backs = [e for e in r.log if e["event"] == "backoff"]
    assert len(backs) == n_close
    for lv in r.views:
        if lv.side in ("upper", "under") and lv.ok:
            assert abs(lv.standoff_m - first[(lv.leaf_id, lv.side)] - 0.03) < 1e-9
            assert lv.clearance_m >= 0.02
    assert all(lv.reason in ("", "no_clearance") for lv in r.views if lv.side != "crown")

    r = ls.run_leaf_scan(ot1, PROFILES["mango"], capture_feed=lambda lv, a: {"depth_m": 0.05})
    assert close_ok(r) == 0 and set(r.leaf_block["not_imaged"]) <= {"min_z", "no_clearance"}

    r = ls.run_leaf_scan(ot1, PROFILES["mango"], capture_feed=lambda lv, a: {"drift_m": 0.03 if a == 0 else 0.0})
    assert close_ok(r) == n_close
    assert sum(1 for e in r.log if e["event"] == "base_drift") == n_close
    r = ls.run_leaf_scan(ot1, PROFILES["mango"], capture_feed=lambda lv, a: {"drift_m": 0.03})
    assert close_ok(r) == 0 and set(r.leaf_block["not_imaged"]) == {"base_drift"}


NAN = float("nan")


def test_invalid_sensor_values_fail_closed(demo_plans):
    """NaN compares False everywhere: a NaN / inf slope, lux or hour refuses the scan, and a NaN depth, drift or
    clearance never counts as a clean capture."""
    for ctx in ({"cross_slope_deg": NAN}, {"cross_slope_deg": float("inf")}, {"light_lux": NAN},
                {"light_lux": "bright"}, {"light_lux": 5000, "hour": NAN}):
        assert "sensor_invalid" in ls.check_preconditions(ctx), ctx
    assert ls.check_preconditions({"cross_slope_deg": 3.5}) == ["dwell_slope"]
    assert ls.check_preconditions({"light_lux": 5000, "hour": 6.0}) == []
    ot1 = demo_plans["OT-1"]
    r = ls.run_leaf_scan(ot1, PROFILES["mango"], ctx={"cross_slope_deg": NAN})
    assert r.status == "refused" and r.leaf_block["reasons"] == ["sensor_invalid"] and r.views == []
    for feed in ({"depth_m": NAN}, {"drift_m": NAN}, {"depth_m": float("-inf")}, {"drift_m": float("inf")}):
        r = ls.run_leaf_scan(ot1, PROFILES["mango"], capture_feed=lambda lv, a, f=feed: f)
        close = [lv for lv in r.views if lv.side in ("upper", "under")]
        assert close and not any(lv.ok for lv in close), feed
        assert set(r.leaf_block["not_imaged"]) <= {"sensor_invalid", "no_clearance"}, feed
        assert not [ld for ld in r.leads if ld["side"] != "crown"]
    g = ls.ClearanceGuard()
    assert g.update(NAN) == "stop" and g.latched and g.log[-1]["event"] == "sensor_invalid"
    assert ls.ClearanceGuard().base_drift(NAN) == "sensor_invalid"
    r = ls.run_leaf_scan(ot1, PROFILES["mango"], guard_feed=lambda lv: [0.05, NAN])
    assert not any(lv.ok for lv in r.views if lv.side != "crown")


def test_backoff_keeps_standoff_footprint_and_ground(demo_plans, lite_plans):
    """A back-off past the 0.20 m stand-off (footprint > 1.0 mm/px, a 12 mm decal < 12 px) or below the 0.05 m lens
    floor is not imaged: ``min_z`` / ``no_clearance``, no leads from that side."""
    c = CFG["leaf_scan"]
    lo, hi = c["standoff_range_m"]
    for plans in (demo_plans, lite_plans):
        for p in plans.values():
            prof = PROFILES[p["crop"]]
            r = ls.run_leaf_scan(p, prof, capture_feed=lambda lv, a: {"depth_m": 0.05 if a < 2 else 1.0})
            for lv in r.views:
                if lv.side in ("upper", "under") and lv.ok:
                    assert lo - 1e-9 <= lv.standoff_m <= hi + 1e-9, (p["tree_id"], lv.leaf_id, lv.standoff_m)
                    assert lv.mm_per_px <= c["max_footprint_mm_px"] + 1e-9
                    assert 12.0 / lv.mm_per_px >= c["min_decal_px"] - 1e-9
                    assert lv.view.pos[2] >= 0.05
            bad = {lv.leaf_id + lv.side for lv in r.views if not lv.ok}
            assert not [ld for ld in r.leads if ld["side"] != "crown" and ld["leaf"] + ld["side"] in bad]
    ot1 = demo_plans["OT-1"]
    obs = ls.tree_obstacles(ot1)
    lf = ls.select_leaves(ot1, cfg=CFG)["close_up"][0]
    lv = ls.plan_view(lf, "upper", obs, CFG)
    assert lv.ok
    lv.standoff_m = hi                                   # already at the far end: no stand-off left
    nv = ls.backed_off(lv, obs, CFG)
    assert not nv.ok and nv.reason == "min_z" and nv.mm_per_px > c["max_footprint_mm_px"]
    under = ls.plan_view(lf, "under", obs, CFG)          # underside back-off moves the lens down
    under.view.pos = under.view.pos.copy()
    under.view.pos[2] = 0.051
    under.standoff_m = lo
    nv = ls.backed_off(under, obs, CFG)
    assert under.view.fwd[2] > 0 and nv.view.pos[2] < 0.05 and not nv.ok and nv.reason == "no_clearance"


def test_unreachable_leaf_marks_and_other_failures_carry_the_side_reason(demo_plans):
    ot1 = demo_plans["OT-1"]
    r = ls.run_leaf_scan(ot1, PROFILES["mango"], guard_feed=lambda lv: [0.01])      # every side blocked
    close = [x for x in r.records if x["close_up"]]
    assert len(r.marks) == len(close) and not r.revisit
    assert all(m["reason"] == "out_of_reach_marked" and set(m["side_reasons"].values()) == {"no_clearance"}
               for m in r.marks)
    for feed, why in (({"drift_m": 0.03}, "base_drift"), ({"depth_m": 0.05}, "min_z"),
                      ({"depth_m": NAN}, "sensor_invalid")):
        r = ls.run_leaf_scan(ot1, PROFILES["mango"], capture_feed=lambda lv, a, f=feed: f)
        assert not [m for m in r.marks if why in m["side_reasons"].values()], why
        assert any(why in v["side_reasons"].values() for v in r.revisit), why
