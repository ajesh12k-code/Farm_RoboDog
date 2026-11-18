"""WP0c acceptance: orchard reach, load and feasibility sweep (ROS-free, headless MuJoCo).

Fast tests check the recorded blocks of config/fields/orchard_candidates.json against the WP0c acceptance list
and re-verify every frozen pose on the live model. Tests marked `slow` re-run the
measurements themselves; they are skipped unless NANDI_SLOW=1 or `-m slow` is given.
"""
import json
import math
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "tools"))

from tools import feasibility_sweep as fsw  # noqa: E402

DOC = fsw.load_json()
FROZEN = DOC["frozen"]
TREES = {t["id"]: t for t in DOC["candidates"]["trees"]}
TARGETS = {t["id"]: t for t in DOC["candidates"]["targets"]}
FZ = [t for t in FROZEN["targets"] if t["status"] == "FROZEN"]
CONTACT_TOOLS = ("T2", "T4", "T5")


def _slow_on(request) -> bool:
    return os.environ.get("NANDI_SLOW") == "1" or "slow" in (request.config.getoption("markexpr") or "")


@pytest.fixture
def slow_only(request):
    if not _slow_on(request):
        pytest.skip("slow: set NANDI_SLOW=1 or run with -m slow")


@pytest.fixture(scope="module")
def sw():
    cradles = [{"tool": c["tool"], "pos": c["pos"]} for c in FROZEN["rack"]["cradles"]]
    return fsw.ArmSweep(cradles=cradles)


# --------------------------------------------------------------------------------------------- inputs
def test_candidates_regenerate_identically():
    gen = json.loads(json.dumps(fsw.generate_candidates()))
    assert gen == DOC["candidates"], "orchard_candidates.json is stale: run `feasibility_sweep.py candidates --write`"


def test_layout_uses_frozen_nhb_spacing():
    """OQ-02 B: mango rows 5 x 2.5 m; OT-1 isolated; 17 trees of the OS 4.2 demo."""
    trees = DOC["candidates"]["trees"]
    assert len(trees) == 17
    by = {}
    for t in trees:
        by.setdefault(t["species"], []).append(t)
    assert {k: len(v) for k, v in by.items()} == {"mango": 5, "guava": 4, "papaya": 4, "banana": 4}
    r1 = sorted((t for t in trees if t["id"].startswith("R1-")), key=lambda t: t["x"])
    assert all(b["x"] - a["x"] == pytest.approx(2.5) for a, b in zip(r1, r1[1:]))
    rows = DOC["layout"]["rows_y"]
    assert rows["R2"] - rows["R1"] == pytest.approx(5.0)
    assert DOC["layout"]["mango_5x3_in_row_m"] == 3.0
    var = DOC["layout"]["variants"]["mango_5x3"]
    assert var["in_row_m"] == 3.0 and all(b - a == pytest.approx(3.0) for a, b in zip(var["x"], var["x"][1:]))
    assert all(1.2 <= t["H"] <= 3.0 for t in trees)


def test_variant_5x3_trees_respaces_only_r1():
    trees = DOC["candidates"]["trees"]
    var = {t["id"]: t for t in fsw.variant_5x3_trees(trees)}
    for t in trees:
        v = var[t["id"]]
        assert v["y"] == t["y"] and v["wood"] == t["wood"] and v["canopy_r"] == t["canopy_r"]
        if not t["id"].startswith("R1-"):
            assert v["x"] == t["x"]
    r1 = sorted((v for v in var.values() if v["id"].startswith("R1-")), key=lambda t: t["x"])
    assert len(r1) == 4 and all(b["x"] - a["x"] == pytest.approx(3.0) for a, b in zip(r1, r1[1:]))


def test_variant_5x3_swept_for_every_r1_target():
    """OQ-02 B: the 5 x 3 m mango spacing is swept, not only recorded as a constant."""
    v = FROZEN["variant_5x3"]
    r1 = [t for t in DOC["candidates"]["targets"] if t["tree"].startswith("R1-")]
    assert v["in_row_m"] == 3.0 and v["x"] == DOC["layout"]["variants"]["mango_5x3"]["x"]
    assert v["targets"] == len(r1) == len(v["targets_5x3"])
    assert {t["id"] for t in v["targets_5x3"]} == {t["id"] for t in r1}
    assert v["frozen_5x3"] == sum(1 for t in v["targets_5x3"] if t["status"] == "FROZEN")
    base = {t["id"]: t["status"] for t in FROZEN["targets"]}
    assert v["frozen_5x2.5"] == sum(1 for t in r1 if base[t["id"]] == "FROZEN")
    assert {d["id"] for d in v["status_changes"]} == {t["id"] for t in v["targets_5x3"] if t["status"] != base[t["id"]]}
    per_tree = {}
    for t in v["targets_5x3"]:
        if t["status"] == "FROZEN" and t["id"].split(":")[1] == "mango_pedicel_cut":
            per_tree[t["id"].split(":")[0]] = per_tree.get(t["id"].split(":")[0], 0) + 1
    assert max(per_tree.values(), default=0) >= 3                   # OS 3.4 demo need holds on 5 x 3 m too


# --------------------------------------------------------------------------------------------- geometry units
def test_closed_form_ik_matches_mujoco_fk(sw):
    rng = np.random.default_rng(7)
    n_checked = 0
    for _ in range(25):
        p = np.array([rng.uniform(-0.1, 0.6), rng.uniform(0.15, 0.6), rng.uniform(0.15, 0.8)])
        axis = fsw.unit([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        Q, A = sw.candidates(p, fsw.TOOLS["T4"]["tcp"], axis, 60.0, max_n=8)
        for q5, a in zip(Q, A):
            sw.fk(q5)
            tcp, tax = sw.tcp_of(fsw.TOOLS["T4"]["tcp"])
            assert np.linalg.norm(tcp - sw.robot_to_sim(p)) < 1e-6
            assert np.dot(sw.R_robot.T @ tax, axis) >= math.cos(math.radians(60.0)) - 1e-6
            assert np.all(q5 >= sw.limits[:, 0] - 1e-9) and np.all(q5 <= sw.limits[:, 1] + 1e-9)
            n_checked += 1
    assert n_checked > 50


def test_reach_lookup_interpolates_and_wraps():
    rm = {"az_deg": [-180, -90, 0, 90], "z": [0.0, 1.0], "rho": [[1, 1], [2, 2], [3, 5], [4, 4]]}
    assert fsw.reach_lookup(rm, 0.0, 0.0) == pytest.approx(3.0)
    assert fsw.reach_lookup(rm, 0.0, 0.5) == pytest.approx(4.0)
    assert fsw.reach_lookup(rm, math.radians(45), 0.0) == pytest.approx(3.5)
    assert fsw.reach_lookup(rm, math.radians(135), 0.0) == pytest.approx(2.5)   # 90 -> 180 (= -180) wraps
    assert fsw.reach_lookup(rm, 0.0, 9.0) == pytest.approx(5.0)                  # z clamps


def test_station_check_rejects_wood_and_foliage():
    t = TREES["OT-1"]
    trees = [t]
    ok, why = fsw.station_check(np.array([t["x"], t["y"] - 0.2]), 0.0, trees)
    assert not ok and why.startswith(("feet_wood", "body_wood"))
    ok, why = fsw.station_check(np.array([t["x"], t["y"] - 0.75]), 0.0, trees)
    assert not ok and why.startswith("crown_foliage")           # stowed head inside the mango skirt
    ok, why = fsw.station_check(np.array([t["x"], t["y"] - 1.4]), 0.0, trees)
    assert ok, why


def test_margin_of_uses_thresholds():
    chk = {"robot": 0.03, "wood": 0.25, "ground": 0.05, "keepout": 0.2, "carriage": 0.0, "jaw": 0.02}
    assert fsw.margin_of(chk) == pytest.approx(0.005)            # jaw 0.02 - 0.015


# --------------------------------------------------------------------------------------------- frozen targets
def test_every_frozen_target_in_band_and_within_80pct_reach():
    assert FZ, "nothing frozen"
    for t in FZ:
        z = t["p_robot"][2]
        assert z <= fsw.PICK_BAND_Z + 1e-9, t["id"]
        rm = FROZEN["reach_maps"]["T1" if t["tool"] == "T1" else ("CAM" if t["tool"] == "CAM" else "T4")]
        lim = fsw.REACH_FRACTION * fsw.reach_lookup(rm, math.atan2(t["p_robot"][1], t["p_robot"][0]), z)
        assert t["rho"] == pytest.approx(math.hypot(*t["p_robot"][:2]), abs=1e-3)
        assert t["rho"] <= lim + 1e-3, t["id"]
        assert t["margin"] >= 0.0, t["id"]


def test_contact_targets_have_a_5cm_exit():
    for t in FZ:
        if t["tool"] in CONTACT_TOOLS:
            assert t["exit"] is not None and t["exit"]["margin"] >= 0.0, t["id"]


def test_frozen_poses_reverify_on_the_live_model(sw):
    """Every frozen joint vector puts the TCP on its target with all clearances (no arm-trunk/mast/rail/basket/
    leg/wood collision, keep-out, jaw clearance over the twist roll stroke) and so does the exit end pose."""
    for t in FZ:
        tg, tree = TARGETS[t["id"]], TREES[t["tree"]]
        st, h = np.array(t["station"]["xy"]), math.radians(t["station"]["heading_deg"])
        fsw.load_world(sw, tree, st, h, DOC["candidates"]["trees"], tg)
        tcp = fsw.TOOLS[t["tool"]]["tcp"]
        sw.fk(np.array(t["q"]))
        pos, _axis = sw.tcp_of(tcp)
        assert np.linalg.norm(pos - sw.robot_to_sim(t["p_robot"])) < 2e-3, t["id"]
        chk = sw.check_pose(np.array(t["q"]), t["tool"], jaw_check=t["tool"] == "T4")
        assert chk["ok"], (t["id"], chk)
        if t["tool"] == "T4":           # the recorded twist directions hold over the stroke at target and exit end
            dirs = t["twist"]["dirs_ok"]
            assert dirs and set(dirs) <= set(fsw.TWIST_DIRS), t["id"]
            assert t["twist"]["preferred"] in dirs
            assert sw.check_pose(np.array(t["q"]), "T4", jaw_check=True, twist_dirs=dirs)["ok"], t["id"]
            assert sw.check_pose(np.array(t["exit"]["q_end"]), "T4", jaw_check=True, twist_dirs=dirs)["ok"], t["id"]
        if t["exit"]:
            chk = sw.check_pose(np.array(t["exit"]["q_end"]), t["tool"], jaw_check=t["tool"] == "T4")
            assert chk["ok"], (t["id"], "exit", chk)
            sw.fk(np.array(t["exit"]["q_end"]))
            pos_end, _ = sw.tcp_of(tcp)
            assert np.linalg.norm(pos_end - pos) == pytest.approx(fsw.EXIT_M, abs=3e-3), t["id"]


def test_marks_have_reasons_and_rules_hold():
    allowed = {"above_pick_band", "no_station", "out_of_reach_80", "ik_or_collision", "no_clearance",
               "human_cut_only (OQ-03)"}
    for t in FROZEN["targets"]:
        tg = TARGETS[t["id"]]
        if tg["p"][2] > fsw.PICK_BAND_Z:
            assert t["status"] == "MARK", t["id"]
        if t["status"] == "MARK":
            assert t["reason"] in allowed, t
    by_id = {t["id"]: t for t in FROZEN["targets"]}
    assert all(by_id[f"R4-{i}:banana_bunch:1"]["status"] == "MARK" for i in range(1, 5))
    seeded = by_id["R3-3:papaya_cradle:1"]                       # neighbour fruit seeded inside 15 mm
    assert seeded["status"] == "MARK" and seeded["reason"] == "no_clearance"


def test_demo_needs_are_covered():
    """OS 3.4: the demo needs 3 mango + 2 guava + 1 guava + 1 papaya per visit."""
    per = {}
    for t in FZ:
        per.setdefault((t["tree"], t["task"]), []).append(t)
    mango = max(len(per.get((f"R1-{i}", "mango_pedicel_cut"), [])) for i in range(1, 5))
    guava = max(len(per.get((f"R2-{i}", "guava_clip"), [])) + len(per.get((f"R2-{i}", "guava_cradle"), []))
                for i in range(1, 5))
    papaya = [t for t in FZ if t["task"] == "papaya_cradle"]
    assert mango >= 3 and guava >= 3 and len(papaya) >= 1
    assert all(t["clear"]["jaw"] >= fsw.JAW_CLEAR_M for t in papaya)
    assert len(per.get(("OT-1", "mango_pedicel_cut"), [])) >= 3


def test_leaf_underside_fraction_and_wrist_pitch_range():
    in_band = [lf for lf in FROZEN["leaves"] if lf["in_band"]]
    under = [lf for lf in in_band if lf["underside"]]
    assert in_band
    assert len(under) / len(in_band) >= fsw.UNDERSIDE_MIN_FRACTION
    pitches = [lf["pose"]["wrist_pitch"] for lf in under]
    assert all(abs(p) <= 2.0 + 1e-6 for p in pitches)
    assert max(abs(p) for p in pitches) > 1.5                     # the sweep uses the +-2.0 rad range
    for lf in under:
        assert lf["pose"]["clear"]["leaf"] >= fsw.LEAF_CLEAR_M
        assert lf["pose"]["axis_r"][2] > 0.2                       # looking up at the underside


def test_rack_three_cradles_reachable_on_one_rail():
    rack = FROZEN["rack"]
    assert rack["ok"]
    cr = rack["cradles"]
    assert [c["tool"] for c in cr] == ["T2", "T4", "T5"]
    assert all(c["reachable"] and c["margin"] >= 0 for c in cr)
    assert len({tuple(c["pos"][1:]) for c in cr}) == 1             # same y, z: one rail
    xs = sorted(c["pos"][0] for c in cr)
    assert min(b - a for a, b in zip(xs, xs[1:])) >= 0.12 - 1e-9
    assert all(c["pos"][1] < -0.12 for c in cr)                    # right side, outside the basket wall
    assert min(rack["leg_clear_by_right_hip_rad"].values()) >= fsw.ROBOT_CLEAR_M


def test_crate_side_drop_points():
    cr = FROZEN["crate"]["frozen"]
    assert cr["n_ok"] == cr["n"] > 0


def test_mast_keepout_at_x_minus_030():
    m = {tuple(e["mast_xy"]): e for e in FROZEN["mast"]}
    e = m[(-0.30, 0.0)]
    assert e["frozen_poses_keepout_margin"] >= 0.0
    g_min = min(t["q"][0] for t in FZ)
    assert g_min >= e["gantry_min_for_keepout"] - 1e-9
    assert set(e["static_clear"]) >= {"gantry_rail", "gantry_pillar_rear", "basket_rear"}   # reported, see report


def test_sweep_wall_time_within_budget():
    assert FROZEN["meta"]["wall_s"] <= fsw.SWEEP_BUDGET_S


def test_blocks_carry_run_ids():
    for k in ("frozen", "load", "creep", "matrix"):
        meta = DOC[k]["meta"]
        assert meta["run_id"].startswith(f"wp0c-{'sweep' if k == 'frozen' else k}-")
        assert meta["device_id"] and meta["git_sha"] and meta["git_sha"] != "unknown"


# --------------------------------------------------------------------------------------------- load
def test_recorded_t4_load_gates():
    load = DOC["load"]
    assert load["pass_spec_gate"]                                  # WP0c acceptance as written (59.5 N.m, sag)
    st = load["frozen_t4_static"]
    assert load["pass_model_headroom_by_fruit_kg"] == {k: v["poses_over_70pct_of_model_forcerange"] == 0
                                                       for k, v in st.items()}
    assert load["pass"] == (load["pass_spec_gate"] and all(load["pass_model_headroom_by_fruit_kg"].values()))
    # 1.8 kg runs elbow / wrist above 70 % of the shipped forcerange on frozen poses: the overall flag must say so
    assert not load["pass_model_headroom_by_fruit_kg"]["1.8"] and not load["pass"]
    assert max(st["1.8"]["max_pct_of_model_forcerange"].values()) > fsw.MODEL_HEADROOM
    for c in load["ff_mismatch_cases"]:                            # FF that ignores the fruit sags measurably
        ideal = next(x for x in load["cases"] if x["ff"] and x["fruit_kg"] == c["fruit_kg"])
        assert c["ff_mode"] == "fruit_0kg" and ideal["ff_mode"] == "ideal"
        assert c["sag_m"] > 10 * ideal["sag_m"]
    for c in load["cases"]:
        if c["ff"]:
            assert c["sag_m"] < fsw.SAG_GATE_M
            assert c["static_torque_nm"]["shoulder"] <= fsw.SHOULDER_GATE_NM
            assert not c["saturated"]
        else:
            assert c["sag_m"] > fsw.SAG_GATE_M                     # feed-forward is mandatory for T4
    assert {c["fruit_kg"] for c in load["cases"]} == {1.2, 1.8}


def test_t4_load_feedforward_live():
    q, tid = fsw.worst_t4_pose(FROZEN, 1.8)
    assert q is not None and tid == DOC["load"]["pose_from"]
    cradles = [{"tool": c["tool"], "pos": c["pos"]} for c in FROZEN["rack"]["cradles"]]
    for kg in (1.2, 1.8):
        c = fsw.load_case(q, kg, ff=True, seconds=2.0, cradles=cradles)
        assert c["sag_m"] < fsw.SAG_GATE_M, c
        assert c["static_torque_nm"]["shoulder"] <= fsw.SHOULDER_GATE_NM, c


# --------------------------------------------------------------------------------------------- creep + matrix
def test_creep_recorded_and_dwell_limit_set():
    cr = DOC["creep"]
    rolls = {(c["roll_deg"], c["arm"]) for c in cr["cases"]}
    for r in (3.0, -3.0, 5.0, -5.0):
        assert (r, "PARK") in rolls and (r, "T4_worst") in rolls
    assert cr["dwell_limit_deg"] == min(fsw.dwell_limit(cr["cases"]), fsw.WALK_SLOPE_LIMIT_DEG)
    assert cr["dwell_limit_deg"] >= 3.0
    pitches = {c["pitch_deg"] for c in cr["cases"] if c["pitch_deg"] is not None}
    assert {-5.0, -3.0, 3.0, 5.0} <= pitches                       # rear (mast) downhill measured too
    assert {round(c["downhill_deg"]) for c in cr["cases"] if c["tilt_deg"] > 0} >= {0, 45, 90, 135, 180, 225, 270, 315}


def test_creep_dwell_duration_is_stated_and_uses_long_runs():
    cr = DOC["creep"]
    assert cr["dwell_window_s"] == fsw.CREEP_WINDOW_S and "pick cycle" in cr["rule"]
    assert cr["time_to_2cm_s_by_tilt"] == fsw.dwell_time_by_tilt(cr["long_cases"])
    assert cr["time_to_2cm_s_at_dwell_limit"] == cr["time_to_2cm_s_by_tilt"][f"{cr['dwell_limit_deg']:g}"]
    assert cr["long_dwell_limit_deg"] == fsw.long_dwell_limit(cr["long_cases"])
    # 5 deg creeps past 2 cm inside a 60 s dwell, so the long-dwell limit is below the pick-cycle limit
    t5 = cr["time_to_2cm_s_by_tilt"]["5"]
    assert t5 is not None and t5 < fsw.LONG_DWELL_S
    assert cr["long_dwell_limit_deg"] is not None and cr["long_dwell_limit_deg"] < cr["dwell_limit_deg"]
    assert all(c["window_s"] == fsw.CREEP_LONG_WINDOW_S for c in cr["long_cases"])


def test_dwell_helpers_on_synthetic_cases():
    mk = lambda tilt, creep, t=None: {"tilt_deg": tilt, "creep_m": creep, "time_to_2cm_s": t}  # noqa: E731
    cases = [mk(0, 0.001), mk(3, 0.005), mk(3, 0.006), mk(5, 0.009), mk(10, 0.025), mk(12, 0.01)]
    assert fsw.dwell_limit(cases) == 5.0                            # stops at the first failing tilt
    longs = [mk(3, 0.015, None), mk(3, 0.016, 90.0), mk(5, 0.024, 51.0), mk(5, 0.02, 70.0)]
    assert fsw.dwell_time_by_tilt(longs) == {"3": 90.0, "5": 51.0}
    assert fsw.long_dwell_limit(longs) == 3.0
    assert fsw.long_dwell_limit(longs, dwell_s=50.0) == 5.0


def test_matrix_recorded_six_foot_margin_and_no_fall():
    mx = DOC["matrix"]
    rows = mx["rows"]
    assert len(rows) == 2 * len(fsw.MATRIX_EXT) * len(fsw.MATRIX_PITCH_DEG) * len(fsw.MATRIX_ROLL_DEG)
    assert len(mx["heading_rows"]) == 2 * len(fsw.MATRIX_EXT) * len(fsw.MATRIX_TILTS_DEG) * len(fsw.MATRIX_DOWNHILL_DEG)
    assert not mx["any_fell"]
    assert all(r["hexa"] > 0.13 for r in rows + mx["heading_rows"])   # OS 6 crown gate matrix: 6-foot > 0.13 m
    assert {r["rack"] for r in rows} == {"downhill", "uphill", "level"}
    assert mx["crown_up_walking"].startswith("allowed") == (mx["flat_crown_up_min_tripod"] >= 0.04)


def test_matrix_covers_rear_downhill_pitch_and_every_heading():
    """The mast sits at the rear (x -0.30): rear-downhill pitch (pitch < 0) is the worse case and must be swept."""
    mx = DOC["matrix"]
    assert {r["pitch_deg"] for r in mx["rows"]} == set(fsw.MATRIX_PITCH_DEG)
    assert min(fsw.MATRIX_PITCH_DEG) <= -10.0
    hd = {r["downhill_deg"] for r in mx["heading_rows"]}
    assert hd == set(fsw.MATRIX_DOWNHILL_DEG)


def test_matrix_summary_matches_rows():
    mx = DOC["matrix"]
    every = mx["rows"] + mx["heading_rows"]
    s = fsw.matrix_summary(every)
    for k in ("worst_row", "unstable_tripod_rows", "rows_below_start_gate", "envelope", "start_gate_max_tilt_deg",
              "continue_gate_max_tilt_deg", "start_gate_max_tilt_park_deg", "continue_gate_max_tilt_park_deg"):
        assert mx[k] == json.loads(json.dumps(s[k])), k
    assert mx["worst_row"]["tripod"] == min(r["tripod"] for r in every)
    # the start gate is not met at the worst slope swept: crown-up walking cannot be allowed on every slope
    assert mx["start_gate_max_tilt_park_deg"] is not None
    assert mx["start_gate_max_tilt_park_deg"] < max(fsw.MATRIX_TILTS_DEG)


def test_matrix_summary_on_synthetic_rows():
    mk = lambda arm, ext, tilt, trip, dh=0.0: {"arm": arm, "ext": ext, "roll_deg": None, "pitch_deg": None,  # noqa: E731
                                               "tilt_deg": tilt, "downhill_deg": dh, "rack": "level",
                                               "tripod": trip, "hexa": 0.2}
    rows = [mk("PARK", 0.0, 0.0, 0.09), mk("PARK", 0.0, 2.0, 0.05, 180.0), mk("PARK", 0.0, 2.0, 0.07, 0.0),
            mk("PARK", 0.0, 3.0, 0.035, 180.0), mk("PARK", 0.0, 5.0, 0.05), mk("PARK", 0.0, 10.0, -0.01)]
    s = fsw.matrix_summary(rows)
    assert s["worst_row"]["tripod"] == -0.01 and s["unstable_tripod_rows"] == 1
    assert s["envelope"]["PARK"]["0"]["2"] == {"min_tripod": 0.05, "downhill_deg": 180.0, "tilt_deg": 2.0}
    assert s["envelope"]["PARK"]["0"]["5"]["min_tripod"] == 0.035   # running minimum over all tilts <= 5
    assert s["start_gate_max_tilt_deg"]["PARK"]["0"] == 2.0          # 3 deg drops below 0.04
    assert s["continue_gate_max_tilt_deg"]["PARK"]["0"] == 8.0       # 0.035 >= 0.03 until the 10 deg row
    assert s["start_gate_max_tilt_park_deg"] == 2.0


def test_slope_gravity_convention():
    m = fsw.build_layout()
    fsw.set_slope(m, 5.0, 0.0)
    assert m.opt.gravity[1] < 0 and m.opt.gravity[0] == pytest.approx(0.0)   # roll > 0: right (rack) downhill
    assert np.linalg.norm(m.opt.gravity) == pytest.approx(9.81)
    assert fsw.tilt_of(m) == (pytest.approx(5.0, abs=1e-3), 270.0)
    g_roll = m.opt.gravity.copy()
    fsw.set_tilt(m, 5.0, 270.0)                                     # same plane as roll +5
    assert np.allclose(m.opt.gravity, g_roll, atol=1e-9)
    fsw.set_slope(m, 0.0, -10.0)                                    # pitch < 0: rear (mast) downhill
    assert m.opt.gravity[0] < 0 and fsw.tilt_of(m)[1] == 180.0
    assert fsw.rack_side(270.0, 5.0) == "downhill" and fsw.rack_side(90.0, 5.0) == "uphill"
    assert fsw.rack_side(0.0, 5.0) == "level" and fsw.rack_side(270.0, 0.0) == "level"


# --------------------------------------------------------------------------------------------- T4 twist roll
def test_twist_roll_stroke_covers_pre_roll_release_and_overshoot():
    for k, s in fsw.TWIST_DIRS.items():
        rolls = fsw.twist_rolls_deg(k)
        assert 0.0 in rolls and all(str(r) != "-0.0" for r in rolls)
        assert -s * fsw.TWIST_PRE_ROLL_DEG in rolls                      # twist_pick pre-rolls against the twist
        assert max(s * r for r in rolls) >= fsw.TWIST_MIN_RELEASE_DEG + 15.0 - 1e-9
    from atharv_nandi.core.tasks import twist_pick
    assert twist_pick.PRE_ROLL_DEG == fsw.TWIST_PRE_ROLL_DEG


def _box_surface(m, d, g, n=15):
    s, R, c = m.geom_size[g], d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    u = np.linspace(-1.0, 1.0, n)
    A, B = np.meshgrid(u, u)
    out = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        for sg in (-1.0, 1.0):
            V = np.zeros((A.size, 3))
            V[:, ax], V[:, o[0]], V[:, o[1]] = sg, A.ravel(), B.ravel()
            out.append(c + (V * s) @ R.T)
    return np.concatenate(out)


def test_geom_distance_never_overstates_a_box_box_clearance(sw):
    """MuJoCo 3.3.7 returns an exact 0.0 for some separated box-box pairs (finger pad vs leaf quad); the guard
    must not answer with the fromto length (it overstated by up to 7 mm). Sampled surface points bound the true
    distance from above, so the guarded value may never exceed them."""
    import mujoco
    m, d = sw.m, sw.d
    pads = [g for g in sw.arm_core if sw.gname(g) in ("finger_left_pad", "finger_right_pad")]
    rng = np.random.default_rng(0)
    ft = np.zeros(6)
    glitches = 0
    for i in range(160):
        g = pads[i % 2]
        sw.fk(sw.q_base[sw.qadr])
        c = d.geom_xpos[g] + rng.uniform(-0.05, 0.05, 3)
        sw.set_leaf(sw.R_robot.T @ (c - sw.ground_origin), fsw.unit(rng.normal(size=3)))
        sw.fk(sw.q_base[sw.qadr])
        raw = mujoco.mj_geomDistance(m, d, g, sw.obs_leaf, 0.25, ft)
        glitches += int(raw == 0.0 and np.linalg.norm(ft[3:] - ft[:3]) > 1e-6)
        upper = np.linalg.norm(_box_surface(m, d, g)[:, None] - _box_surface(m, d, sw.obs_leaf)[None], axis=2).min()
        if upper > 0.002:
            assert fsw.geom_distance(m, d, g, sw.obs_leaf, 0.25, np.zeros(6)) <= upper + 1e-4, i
    assert glitches > 0                        # the sample exercises the guard
    sw.clear_obstacles()


def test_pooled_obstacles_keep_their_runtime_orientation(sw):
    """The pooled obstacles are compiled axis-aligned; MuJoCo would treat them as sharing the world frame
    (geom_sameframe) and drop the geom_quat written at run time, standing every oblique branch upright."""
    sw.clear_obstacles()
    for g in sw.obs_cap + sw.obs_sph + sw.obs_box + [sw.obs_leaf]:
        assert sw.m.geom_sameframe[g] == 0
    sw.add_capsule([0.5, -0.15, 0.3], [0.5, 0.15, 0.3], 0.02)            # horizontal branch along robot y
    sw.set_leaf([0.4, 0.0, 0.5], [1.0, 0.0, 0.0])                        # leaf normal along robot x
    sw.fk(sw.q_base[sw.qadr])
    zc = sw.d.geom_xmat[sw.obs_cap[0]].reshape(3, 3)[:, 2]
    zl = sw.d.geom_xmat[sw.obs_leaf].reshape(3, 3)[:, 2]
    assert abs(zc @ (sw.R_robot @ np.array([0.0, 1.0, 0.0]))) == pytest.approx(1.0, abs=1e-6)
    assert abs(zl @ (sw.R_robot @ np.array([1.0, 0.0, 0.0]))) == pytest.approx(1.0, abs=1e-6)
    sw.clear_obstacles()


def test_jaw_roll_rotates_about_the_tool_axis_and_restores(sw):
    sw.clear_obstacles()
    q = np.array(FZ[0]["q"])
    sw.fk(q)
    jaws = sw.jaw_geoms["T4"]
    before = [(sw.d.geom_xpos[g].copy(), sw.d.geom_xmat[g].copy()) for g in jaws]
    sw.add_sphere([0.0, 0.0, 3.0], 0.01, kind="jaw")                   # far away: only the frame matters
    obs = [o[0] for o in sw.active_obs["jaw"]]
    d0 = sw.dist(jaws, obs)[0]
    assert sw.jaw_dist_rolled("T4", obs, 0.0) == pytest.approx(d0, abs=1e-9)
    assert sw.jaw_dist_rolled("T4", obs, 180.0) == pytest.approx(d0, abs=2e-3)   # the two jaws swap
    for g, (p, R) in zip(jaws, before):
        assert np.allclose(sw.d.geom_xpos[g], p) and np.allclose(sw.d.geom_xmat[g], R)
    sw.clear_obstacles()


def test_jaw_check_picks_the_twist_direction_that_clears(sw):
    """An obstacle where the left jaw sits after +45 deg of roll: level jaws clear it, so a roll-0 check passes,
    but the + twist (pre-roll -20, roll to +45) hits it; the mirrored - twist clears it."""
    sw.clear_obstacles()
    t = next(t for t in FZ if t["tool"] == "T4")
    q = np.array(t["q"])
    sw.fk(q)
    g = next(g for g in sw.jaw_geoms["T4"] if sw.gname(g).endswith("_l"))
    p_loc, _R = sw._jaw_local[g]
    Rw, pw = sw.d.xmat[sw.wrist_id].reshape(3, 3), sw.d.xpos[sw.wrist_id]
    c_sim = pw + Rw @ (fsw.rotx(math.radians(45.0)) @ p_loc)
    sw.add_sphere(sw.R_robot.T @ (c_sim - sw.ground_origin), 0.012, kind="jaw", label="seeded")
    chk = sw.check_pose(q, "T4", jaw_check=True)
    tw = chk["twist"]
    assert tw["by_roll_deg"][0.0] >= fsw.JAW_CLEAR_M                    # invisible to a level-jaw check
    assert tw["by_roll_deg"][45.0] < 0.0
    assert tw["dirs_ok"] == ["-"] and chk["ok"]
    assert chk["jaw"] == pytest.approx(tw["min_by_dir"]["-"])
    assert not sw.check_pose(q, "T4", jaw_check=True, twist_dirs=["+"])["ok"]
    sw.clear_obstacles()


def test_frozen_t4_records_carry_the_twist_stroke():
    t4 = [t for t in FZ if t["tool"] == "T4"]
    assert t4
    for t in t4:
        tw = t["twist"]
        assert tw["rolls_deg"] == {k: list(fsw.twist_rolls_deg(k)) for k in fsw.TWIST_DIRS}
        for k in tw["dirs_ok"]:
            assert min(tw["jaw_by_roll_deg"][f"{r:g}"] for r in fsw.twist_rolls_deg(k)) >= fsw.JAW_CLEAR_M, t["id"]
            assert tw["exit_jaw_min_by_dir"][k] >= fsw.JAW_CLEAR_M, t["id"]
        assert t["clear"]["jaw"] == pytest.approx(max(tw["jaw_min_by_dir"][k] for k in tw["dirs_ok"]), abs=1e-4)
        assert t["clear"]["jaw_roll0"] == tw["jaw_by_roll_deg"]["0"]


# --------------------------------------------------------------------------------------------- live re-verification
def test_leaf_underside_poses_reverify_on_the_live_model(sw):
    n = 0
    leaves = {lf["id"]: lf for lf in DOC["candidates"]["leaves"]}
    for lf in FROZEN["leaves"]:
        if not lf.get("underside"):
            continue
        pose, cand, tree = lf["pose"], leaves[lf["id"]], TREES[lf["tree"]]
        st, h = np.array(pose["station"]["xy"]), math.radians(pose["station"]["heading_deg"])
        fsw.load_world(sw, tree, st, h, DOC["candidates"]["trees"])
        R = fsw.rotz(h)
        sw.set_leaf(fsw.to_robot(tree, st, h, np.array(cand["p"])), R.T @ fsw.unit(cand["n"]))
        chk = sw.check_pose(np.array(pose["q"]), "CAM", leaf_check=True)
        assert chk["ok"] and chk["leaf"] >= fsw.LEAF_CLEAR_M, (lf["id"], chk)
        pos, axis = sw.tcp_of(fsw.TOOLS["CAM"]["tcp"])
        assert np.linalg.norm(pos - sw.robot_to_sim(pose["cam_robot"])) < 2e-3, lf["id"]
        assert (sw.R_robot.T @ axis)[2] > 0.2, lf["id"]                  # looking up at the underside
        n += 1
    assert n == FROZEN["summary"]["leaf_underside"]
    sw.clear_obstacles()


def test_rack_cradle_poses_reverify_on_the_live_model(sw):
    for i, c in enumerate(FROZEN["rack"]["cradles"]):
        sw.clear_obstacles()
        own = [g for n, g in sw.rack_geoms.items() if n in (f"racked_{c['tool']}", f"rack_cradle_{i}")]
        chk = sw.check_pose(np.array(c["q"]), c["tool"], exclude_robot=own)
        assert chk["ok"], (c["tool"], chk)
        pos, axis = sw.tcp_of(fsw.RACK_TCP)
        assert np.linalg.norm(pos - (sw.trunk_pos + sw.trunk_R @ np.array(c["pos"]))) < 2e-3, c["tool"]
        assert axis @ (sw.trunk_R @ np.array([0.0, 0.0, -1.0])) >= math.cos(math.radians(8.0)) - 1e-6   # tool down


def test_crate_drop_poses_reverify_on_the_live_model(sw):
    fz = FROZEN["crate"]["frozen"]
    opt = next(o for o in FROZEN["crate"]["options"] if o["gap_m"] == fz["gap_m"])
    y_near, _hz = fsw.crate_world(sw, fz["gap_m"])
    assert y_near == pytest.approx(fz["crate_near_wall_y"], abs=1e-3)
    n = 0
    for p in opt["points"]:
        if not p["ok"]:
            continue
        for q in (p["q"], p["exit_q_end"]):
            chk = sw.check_pose(np.array(q), "T2")
            assert chk["ok"], (p["p"], chk)
        sw.fk(np.array(p["q"]))
        pos, _ = sw.tcp_of(fsw.TOOLS["T2"]["tcp"])
        assert np.linalg.norm(pos - sw.robot_to_sim(p["p"])) < 2e-3
        n += 1
    assert n == fz["n_ok"] == fz["n"]
    sw.clear_obstacles()


def test_candidate_basins_match_the_pilot_registry():
    reg = fsw.load_json(os.path.join(fsw.REPO, "config", "fields", "orchard_demo.json"))
    by = {t["tree_id"]: t for t in reg["trees"]}
    for t in DOC["candidates"]["trees"]:
        assert t["basin_r"] == pytest.approx(by[t["id"]]["basin_r_m"]), t["id"]


# --------------------------------------------------------------------------------------------- JSON hygiene
def test_committed_blocks_are_not_stale():
    assert fsw.stale_blocks(DOC) == [] and DOC.get("stale_blocks") == []
    assert DOC["frozen"]["inputs"] == {"candidates_digest": fsw.candidates_digest(DOC["candidates"])}
    for k in ("load", "creep", "matrix"):
        assert DOC[k]["inputs"] == {"frozen_run_id": FROZEN["meta"]["run_id"]}, k


def test_stale_blocks_flag_regenerated_candidates_and_resweeps():
    doc = json.loads(json.dumps({k: DOC[k] for k in ("candidates", "frozen", "load", "creep", "matrix")}))
    doc["candidates"]["trees"][0]["H"] += 0.1
    assert fsw.stale_blocks(doc) == ["frozen"]
    doc["frozen"]["meta"]["run_id"] = "wp0c-sweep-new"
    assert fsw.stale_blocks(doc) == ["frozen", "load", "creep", "matrix"]


def test_save_json_is_atomic(tmp_path, monkeypatch):
    path = str(tmp_path / "c.json")
    fsw.save_json({"a": 1}, path)

    def boom(*a, **k):
        raise KeyboardInterrupt("killed mid-write")
    monkeypatch.setattr(fsw.json, "dump", boom)
    with pytest.raises(KeyboardInterrupt):
        fsw.save_json({"a": 2}, path)
    monkeypatch.undo()
    assert fsw.load_json(path) == {"a": 1}
    assert os.listdir(tmp_path) == ["c.json"]                           # no temp file left behind


def _writer(path, key, n):
    for i in range(n):
        fsw.update_json({key: i}, path)


def test_update_json_keeps_concurrent_blocks(tmp_path):
    import multiprocessing as mp
    path = str(tmp_path / "c.json")
    fsw.save_json({"base": True}, path)
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=_writer, args=(path, f"k{j}", 15)) for j in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(60)
        assert p.exitcode == 0
    doc = fsw.load_json(path)
    assert doc["base"] is True and all(doc[f"k{j}"] == 14 for j in range(4))
    assert sorted(os.listdir(tmp_path)) == ["c.json"]                  # lock released, no temp files


def test_json_lock_breaks_a_stale_lock(tmp_path):
    path = str(tmp_path / "c.json")
    lock = str(tmp_path / ".c.json.lock")
    os.mkdir(lock)
    old = time.time() - fsw.JSON_LOCK_STALE_S - 5
    os.utime(lock, (old, old))
    fsw.update_json({"x": 1}, path)
    assert fsw.load_json(path)["x"] == 1 and not os.path.exists(lock)


# --------------------------------------------------------------------------------------------- status + budget
def test_load_summary_states_the_vacuous_sag_gate():
    s = DOC["load"]["load_summary"]
    assert s == json.loads(json.dumps(fsw.load_summary(DOC["load"])))
    assert s["spec_gate"] == "pass" and s["overall"] == "fail"
    assert s["model_headroom_gate"] == {"1.2": "pass", "1.8": "fail"}
    assert s["sag_gate"].startswith("vacuous")
    assert s["sag_gate_with_fruit_blind_ff"] == "fail"
    assert all(v > 5 * fsw.SAG_GATE_M for v in s["sag_fruit_blind_ff_m"].values())
    assert "vacuous" in s["text"] and "fails at 1.8 kg" in s["text"]


def test_matrix_records_mast_mass_against_the_budget():
    b = DOC["matrix"]["layout_vs_budget"]
    assert b == json.loads(json.dumps(fsw.mast_budget()))
    assert b["mast_system_kg"] == pytest.approx(2.6, abs=1e-3) and b["mast_moving_kg"] == pytest.approx(1.0, abs=1e-3)
    assert b["mast_system_kg"] > fsw.MAST_SYSTEM_BUDGET_KG and b["mast_moving_kg"] > fsw.MAST_MOVING_BUDGET_KG
    assert b["within_budget"] is False
    assert "crown_gates.json" in b["superseded_by"] and "pre-WP1" in b["robot"]


def test_creep_counts_from_arrival():
    cr = DOC["creep"]
    assert cr["ref_s"] == fsw.CREEP_REF_S <= 0.1 and "after arrival" in cr["rule"]
    for c in cr["cases"] + cr["long_cases"]:
        assert c["ref_s"] == fsw.CREEP_REF_S and 0.0 <= c["drift_first_1s_m"] <= c["creep_m"] + 0.005


# --------------------------------------------------------------------------------------------- slow re-runs
@pytest.mark.slow
def test_creep_rerun_matches_record(slow_only):
    rec = {(c["roll_deg"], c["pitch_deg"], c["arm"]): c for c in DOC["creep"]["cases"]}
    cradles = [{"tool": c["tool"], "pos": c["pos"]} for c in FROZEN["rack"]["cradles"]]
    for roll, pitch in ((3.0, 0.0), (5.0, 0.0), (0.0, -5.0)):
        c = fsw.creep_case(roll, pitch, cradles=cradles)
        assert c["creep_m"] == pytest.approx(rec[(roll, pitch, "PARK")]["creep_m"], abs=1e-3)


@pytest.mark.slow
def test_matrix_rerun_matches_record(slow_only):
    rec = {(r["arm"], r["ext"], r["pitch_deg"], r["roll_deg"]): r for r in DOC["matrix"]["rows"]}
    cradles = [{"tool": c["tool"], "pos": c["pos"]} for c in FROZEN["rack"]["cradles"]]
    for ext, roll, pitch in ((1.6, 0.0, 0.0), (1.6, 5.0, 0.0), (0.0, -5.0, 0.0), (1.6, 5.0, -10.0)):
        r = fsw.matrix_case(ext, roll, pitch, cradles=cradles)
        assert r["tripod"] == pytest.approx(rec[("PARK", ext, pitch, roll)]["tripod"], abs=2e-3)
    hrec = {(r["arm"], r["ext"], r["tilt_deg"], r["downhill_deg"]): r for r in DOC["matrix"]["heading_rows"]}
    r = fsw.matrix_case(1.6, 0.0, 0.0, cradles=cradles, tilt_deg=5.0, downhill_deg=210.0)
    assert r["tripod"] == pytest.approx(hrec[("PARK", 1.6, 5.0, 210.0)]["tripod"], abs=2e-3)


@pytest.mark.slow
def test_full_sweep_reproduces_frozen(slow_only):
    res = fsw.run_sweep(DOC, progress=False)
    assert res["meta"]["wall_s"] <= fsw.SWEEP_BUDGET_S
    got = {t["id"]: t["status"] for t in res["targets"]}
    want = {t["id"]: t["status"] for t in FROZEN["targets"]}
    assert got == want
    assert res["summary"]["underside_fraction"] == FROZEN["summary"]["underside_fraction"]
    vgot = {t["id"]: t["status"] for t in res["variant_5x3"]["targets_5x3"]}
    vwant = {t["id"]: t["status"] for t in FROZEN["variant_5x3"]["targets_5x3"]}
    assert vgot == vwant
