"""WP27 Ready check (MISSION-PRESETS 0.2, ORCHARD-MVP-SPEC 4.5): every failure gives its reason code; season
matrix, rain, wet foliage, heat, link state, crown gates (WP6 table), chips and adjustments. ROS-free."""
import copy
import json
import os

import pytest

from atharv_nandi.core import presets as ps
from atharv_nandi.core import ready_check as rc
from atharv_nandi.core import weather_stub as ws

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIELD = json.load(open(os.path.join(REPO, "config", "fields", "orchard_demo.json"), encoding="utf-8"))
MP_05 = ("out_of_reach_marked orbit_blocked range_lost low_light disk_low payload_full tool_unavailable rtk "
         "energy_reserve stability crown_gate crown_clearance wind slope geofence person_hold animal_hold "
         "local_link_lost cloud_link_lost stuck sink over_capacity paste_required twist_failed no_clearance bagged "
         "leaf_minimum season rain wet_foliage heat self_right_failed registry_mismatch not_in_mvp").split()


@pytest.fixture(scope="module")
def reg():
    r = ps.Registry.load()
    assert r.errors == []
    return r


def wx(**kw):
    """A weather_stub state (WP7) with overrides."""
    s = ws.WeatherStub(ws.WeatherScript(month=kw.pop("month", 11), start_hour=kw.pop("hour", 9.0))).state(0.0)
    s.update(kw)
    return s


def run(reg, pid, targets=None, params=None, field=FIELD, **ctx):
    base = dict(season_profile="demo_all_fruiting", month=11, frames_age_h=0.1)
    base.update(ctx)
    return rc.check(reg.get(pid), rc.ReadyContext.nominal(**base), targets, params, field)


OT1 = {"tree_ids": ["OT-1"]}
R1 = {"row_ids": ["R1"]}


def test_nominal_tour_presets_are_green(reg):
    for pid, tg, prm in (("scan", R1, None), ("tree360", OT1, None), ("leaf_scan", OT1, {"n_leaves": 6}),
                         ("trunk_scan", OT1, None), ("fruit_scout", OT1, None), ("harvest_mango", OT1, None),
                         ("harvest_guava", {"tree_ids": ["G-2"]}, None), ("harvest_papaya", {"tree_ids": ["P-1"]}, None),
                         ("prune", {"tree_ids": ["G-2"]}, None), ("prune_banana", {"tree_ids": ["B-2"]}, None),
                         ("harvest_banana", {"tree_ids": ["B-2"]}, None), ("dock_unload", None, None),
                         ("return_home", None, None), ("safe_park", None, None), ("stuck_recover", None, None)):
        r = run(reg, pid, tg, prm)
        assert r.level == rc.GO and r.reasons == [] and r.startable(), (pid, r.to_dict())
        assert r.estimate.get("t_s", 0) > 0, pid


# (preset, targets, params, ctx overrides, expected reason, expected level)
FAILURES = [
    ("scan", R1, None, {"rtk": "NONE"}, "rtk", "no_go"),
    ("scan", R1, None, {"rtk": "FLOAT"}, "rtk", "no_go"),                    # FLOAT only RETURN HOME / SAFE PARK
    ("scan", R1, None, {"soc": 0.12}, "energy_reserve", "no_go"),
    ("tree360", OT1, None, {"disk_free_mb": 20.0}, "disk_low", "no_go"),      # 1.5 x 26 MB
    ("scan", R1, None, {"estop": True}, "estop", "no_go"),
    ("scan", R1, None, {"local_link": "unknown"}, "local_link_lost", "no_go"),  # WP7: unknown refuses
    ("scan", R1, None, {"local_link": "lost"}, "local_link_lost", "no_go"),
    ("scan", R1, None, {"cloud_link": "lost"}, "cloud_link_lost", "no_go"),
    ("scan", R1, None, {"stability_ok": False}, "stability", "no_go"),
    ("scan", R1, None, {"cross_slope_deg": 6.0}, "slope", "no_go"),          # walking > 5 deg refused
    ("tree360", OT1, None, {"cross_slope_deg": 4.0}, "slope", "no_go"),       # dwell <= 3 deg
    ("leaf_scan", OT1, None, {"pitch_deg": 4.0}, "slope", "no_go"),           # any fall line
    ("harvest_mango", OT1, None, {"tools_available": ("T4", "T5")}, "tool_unavailable", "no_go"),
    ("harvest_mango", OT1, None, {"holding": True, "tool_mounted": "T4"}, "tool_unavailable", "no_go"),
    ("harvest_papaya", {"tree_ids": ["P-1"]}, None, {"payload_kg": 1.5}, "payload_full", "no_go"),   # + 1.8 kg > 3
    ("harvest_papaya", {"tree_ids": ["P-1"]}, None, {"cradle_free": False}, "payload_full", "no_go"),
    ("harvest_mango", OT1, None, {"tray_cells_free": 0}, "payload_full", "no_go"),
    ("leaf_scan", OT1, None, {"weather": wx(wet_foliage=True, wet_cause="dew")}, "wet_foliage", "no_go"),
    ("harvest_mango", OT1, None, {"weather": wx(wet_foliage=True, wet_cause="rain")}, "wet_foliage", "no_go"),
    ("prune", {"tree_ids": ["G-2"]}, None, {"weather": wx(wet_foliage=True, wet_cause="rain")}, "wet_foliage", "no_go"),
    ("scan", R1, None, {"weather": wx(rain=True, rain_mm_h=8.0, wet_foliage=True)}, "rain", "no_go"),
    ("tree360", OT1, None, {"weather": wx(temp_c=43.0)}, "heat", "no_go"),
    ("tree360", OT1, None, {"weather": wx(temp_c=39.5)}, "heat", "degraded"),
    ("tree360", OT1, None, {"light_lux": 300.0}, "low_light", "no_go"),
    ("leaf_scan", OT1, {"dawn_mode": True}, {"light_lux": 600.0, "weather": wx(hour=10.0)}, "low_light", "no_go"),
    ("tree360", OT1, None, {"wind_mps": 10.5}, "wind", "no_go"),             # raise: wind < 10
    ("tree360", OT1, None, {"gust": True}, "wind", "no_go"),
    ("harvest_mango", OT1, None, {"season_profile": "up", "month": 1}, "season", "no_go"),
    ("harvest_guava", {"tree_ids": ["G-2"]}, None, {"season_profile": "up", "month": 5}, "season", "no_go"),
    ("prune", {"tree_ids": ["OT-1"]}, None, {"season_profile": "up", "month": 1}, "season", "degraded"),
    ("tree360", OT1, None, {"season_profile": "up", "month": 5}, "wind", "degraded"),
    ("scan", R1, None, {"season_profile": "up", "month": 8}, "season", "degraded"),
    ("harvest_papaya", {"tree_ids": ["P-1"]}, None, {"season_profile": "up"}, "not_in_mvp", "no_go"),
    ("scan", R1, None, {"novnc": True, "actors": True}, "novnc_actors", "no_go"),
    ("tree360", {"tree_ids": ["R1-2"]}, {"arc_mode": "full"}, {}, "orbit_blocked", "no_go"),
    ("fruit_scout", OT1, None, {"frames_age_h": None, "wind_mps": 12.0}, "crown_gate", "no_go"),
    ("work_lead", R1, None, {}, "job_sheet_missing", "no_go"),
    ("work_follow", None, None, {}, "tag_missing", "no_go"),
    ("work_tote", None, None, {}, "tool_unavailable", "no_go"),
    ("dock_unload", None, None, {"ext_m": 0.6}, "crown_gate", "no_go"),
    ("estop_recovery", None, None, {}, "estop_not_engaged", "no_go"),
    ("estop_recovery", None, None, {"estop": True, "estop_confirmed": (True, False)}, "confirm_required", "no_go"),
    ("self_right_spike", None, None, {"person_clear_confirmed": True}, "not_fallen", "no_go"),
    ("self_right_spike", None, None, {"fallen": True}, "confirm_required", "no_go"),
    ("leaf_scan", OT1, None, {"work_field_clear": False}, "person_hold", "no_go"),
    ("trunk_scan", OT1, None, {"wrist_camera_ok": False}, "sensor_fault", "no_go"),
    ("work_spray", None, None, {}, "not_in_mvp", "no_go"),
    ("tree360", {"tree_ids": ["XX-1"]}, None, {}, "registry_mismatch", "no_go"),
    ("tree360", OT1, {"stations": 99}, {}, "param_invalid", "no_go"),
]


@pytest.mark.parametrize("pid, targets, params, ctx, reason, level", FAILURES,
                         ids=[f"{f[0]}-{f[4]}-{i}" for i, f in enumerate(FAILURES)])
def test_each_ready_failure_gives_its_reason_code(reg, pid, targets, params, ctx, reason, level):
    r = run(reg, pid, targets, params, **ctx)
    assert r.level == level, r.to_dict()
    assert reason in r.reasons, r.to_dict()
    assert reason in MP_05 or reason in rc.WP27_REASONS
    if level == rc.NO_GO:
        assert not r.startable(True)
        row = next(i for i in r.items if i["reason"] == reason and i["level"] == rc.NO_GO)
        assert row["check"] and row["detail"]                     # the tile shows the reason
    else:
        assert r.needs_confirm and not r.startable() and r.startable(confirmed=True)


def test_every_reason_code_is_in_the_enum_or_a_documented_extension():
    from atharv_nandi.core.workflow import reasons as wr
    assert set(MP_05) <= set(wr.MP_05)
    missing = [c for c in rc.WP27_REASONS if c not in wr.REASONS]
    # the WP27 extensions outside the WP34 enum, documented as a change request to the WP34 enum
    documented = {"novnc_actors", "param_invalid", "preset_unknown", "job_sheet_missing", "tag_missing",
                  "sensor_fault", "confirm_required", "estop_not_engaged", "not_fallen", "queue_full", "not_checked"}
    assert set(missing) <= documented, sorted(set(missing) - documented)


def test_float_rtk_is_enough_for_return_home_and_safe_park(reg):
    for pid in ("return_home", "safe_park"):
        assert run(reg, pid, rtk="FLOAT").level == rc.GO, pid
    assert run(reg, "return_home", rtk="FLOAT", soc=0.2).level == rc.GO      # low SoC return is never blocked


def test_recovery_presets_ignore_season_weather_energy(reg):
    storm = wx(rain=True, rain_mm_h=20.0, wet_foliage=True, temp_c=44.0)
    for pid in ("safe_park", "return_home", "stuck_recover"):
        r = run(reg, pid, weather=storm, soc=0.05, season_profile="up", month=7, cloud_link="lost")
        assert r.level == rc.GO, (pid, r.to_dict())


def test_safe_park_without_local_link_parks_in_place(reg):
    r = run(reg, "safe_park", local_link="lost")
    assert r.level == rc.GO and r.adjust == {"move_max_m": 0.0}
    assert any("park in place" in c for c in r.chips)
    assert r.estimate["t_s"] == pytest.approx(11.5)                         # no move


def test_crown_walk_gate_falls_back_to_stowed_or_stop_and_go(reg):
    # wind 8-10 m/s: raising is allowed, walking crown-up is not (WP6: < 8 m/s)
    s = run(reg, "scan", R1, None, wind_mps=9.0)
    assert s.level == rc.GO and s.adjust.get("crown_band") == "stowed"
    assert any(c.startswith("crown_gate") for c in s.chips)
    t = run(reg, "tree360", OT1, None, wind_mps=9.0)
    assert t.level == rc.GO and t.adjust == {"crown_walk": "stop_and_go"}
    assert t.estimate["t_s"] == pytest.approx(231.6, abs=0.5)               # 29 s / station (OS 2.3 rule 7)
    # 2.5 deg: crown-up walking is <= 2 deg (WP6 policy), the dwell <= 3 deg still holds
    t2 = run(reg, "tree360", OT1, None, cross_slope_deg=2.5)
    assert t2.level == rc.GO and t2.adjust == {"crown_walk": "stop_and_go"}
    s2 = run(reg, "scan", R1, None, cross_slope_deg=4.0)
    assert s2.level == rc.GO and s2.adjust.get("crown_band") == "stowed"    # stowed walking <= 5 deg


def test_alley_clearance_runs_scan_stowed_under_the_banana_rope(reg):
    r = run(reg, "scan", {"row_ids": ["R4"]})
    assert r.level == rc.GO and r.adjust.get("crown_band") == "stowed"
    assert any(c.startswith("crown_clearance") for c in r.chips)
    low = copy.deepcopy(FIELD)
    for t in low["trees"]:
        if t["row_id"] == "R4":
            t["alley_clear_height_m"] = [1.0, 1.0]                          # below the stowed head (1.16 m)
    r2 = run(reg, "scan", {"row_ids": ["R4"]}, field=low)
    assert r2.level == rc.NO_GO and "crown_clearance" in r2.reasons


def test_tree360_two_sided_when_the_annulus_is_not_clear(reg):
    r = run(reg, "tree360", {"tree_ids": ["R1-2"]})
    assert r.level == rc.GO and r.adjust.get("arc_mode") == "two_sided"
    lidar = run(reg, "tree360", OT1, None, annulus_clear={"OT-1": False})
    assert lidar.adjust.get("arc_mode") == "two_sided"                       # LiDAR wins over the registry


def test_fruit_scout_one_lap_when_frames_are_stale(reg):
    r = run(reg, "fruit_scout", OT1, None, frames_age_h=30.0)
    assert r.level == rc.GO and r.adjust == {"source": "one_lap"} and r.estimate["t_s"] == pytest.approx(160.0)


def test_dawn_mode_lowers_the_light_threshold_before_8(reg):
    r = run(reg, "leaf_scan", OT1, {"dawn_mode": True}, light_lux=600.0, weather=wx(hour=6.5))
    assert r.level == rc.GO
    assert run(reg, "leaf_scan", OT1, None, light_lux=600.0, weather=wx(hour=6.5)).reasons == ["low_light"]


def test_bagged_fruit_refuses_harvest_and_missing_registry_fields(reg):
    f = copy.deepcopy(FIELD)
    t = next(t for t in f["trees"] if t["tree_id"] == "OT-1")
    t["bagging_date"] = "2026-04-25"
    assert "bagged" in run(reg, "harvest_mango", OT1, field=f).reasons
    t["canopy_r_m"] = None
    assert "registry_mismatch" in run(reg, "tree360", OT1, {"r1_m": 1.9}, field=f).reasons
    g = copy.deepcopy(FIELD)
    next(t for t in g["trees"] if t["tree_id"] == "R2-2")["graft_union_h_m"] = None
    assert "registry_mismatch" in run(reg, "prune", {"tree_ids": ["G-2"]}, field=g).reasons


def test_geofence_contains_the_whole_path_incl_orbit_annuli(reg):
    f = copy.deepcopy(FIELD)
    f["geofence"] = [[-6.5, -4.0], [16.8, -4.0], [16.8, 17.5], [-6.5, 17.5]]   # cuts the OT-1 annulus (y -6)
    r = run(reg, "tree360", OT1, field=f)
    assert r.level == rc.NO_GO and "geofence" in r.reasons
    assert run(reg, "scan", R1, field=f).level == rc.GO


def test_estop_recovery_needs_both_confirmations(reg):
    r = run(reg, "estop_recovery", estop=True, estop_confirmed=(True, True))
    assert r.level == rc.GO                                                  # e-stop engaged is its precondition
    for conf in ((False, False), (True, False), (False, True)):
        assert not run(reg, "estop_recovery", estop=True, estop_confirmed=conf).startable(True)


def test_tile_state_without_targets(reg):
    r = run(reg, "tree360")
    assert any(i["check"] == "targets" and i["level"] == rc.INFO for i in r.items)
    tiles = ps.console_tiles(reg, rc.check_all(reg, rc.ReadyContext.nominal(season_profile="demo_all_fruiting"), FIELD))
    by = {t["id"]: t for t in tiles}
    assert by["work_spray"]["enabled"] is False and by["work_spray"]["reason"] == "not_in_mvp"
    assert by["work_follow"]["enabled"] is False and by["work_follow"]["reason"] == "tag_missing"
    assert by["safe_park"]["enabled"] is True and by["scan"]["hi"] == "सर्वे"
    assert [t["id"] for t in tiles][0] == "scan"                             # console order
    assert ps.console_tiles(reg)[0]["reason"] == "not_checked"


# ------------------------------------------------------------------ season matrix (OS 4.5)
def test_season_file_covers_every_preset_key_and_month(reg):
    s = rc.season_table()
    assert s.validate() == []
    for p in reg:
        for crop in p.crops:
            assert p.season_key(crop) in s.keys, (p.id, crop)
    for k in s.keys:
        for mo in range(1, 13):
            assert s.verdict(k, mo)["level"] in (rc.GO, rc.DEGRADED, rc.NO_GO)


@pytest.mark.parametrize("key, month, level, reason", [
    ("scan", 1, "go", ""), ("scan", 5, "degraded", "heat"), ("scan", 8, "degraded", "season"), ("scan", 11, "go", ""),
    ("tree360", 4, "go", ""), ("tree360", 6, "degraded", "wind"), ("tree360", 9, "degraded", "season"),
    ("leaf_scan", 2, "go", ""), ("leaf_scan", 7, "degraded", "season"),
    ("harvest_mango", 6, "go", ""), ("harvest_mango", 8, "degraded", "season"), ("harvest_mango", 9, "no_go", "season"),
    ("harvest_mango", 11, "no_go", "season"),
    ("harvest_guava", 1, "go", ""), ("harvest_guava", 4, "no_go", "season"), ("harvest_guava", 8, "degraded", "season"),
    ("harvest_guava", 12, "go", ""),
    ("prune_mango", 8, "go", ""), ("prune_mango", 2, "degraded", "season"),
    ("prune_guava", 5, "go", ""), ("prune_guava", 6, "degraded", "season"), ("prune_guava", 1, "go", ""),
    ("recovery", 7, "go", ""), ("dock_unload", 7, "go", ""),
])
def test_season_matrix_matches_os_45(key, month, level, reason):
    v = rc.season_table().verdict(key, month, "up")
    assert (v["level"], v["reason"]) == (level, reason), v


def test_demo_profile_lifts_season_but_not_weather_and_says_so():
    s = rc.season_table()
    v = s.verdict("harvest_mango", 11, "demo_all_fruiting")
    assert v["level"] == rc.GO and "not agronomically realistic" in v["label"]
    assert s.verdict("tree360", 6, "demo_all_fruiting")["reason"] == "wind"   # weather entries kept
    assert s.verdict("nope", 1)["level"] == rc.NO_GO


def test_bad_season_files_are_refused(tmp_path):
    import yaml
    doc = yaml.safe_load(open(rc.SEASON_YAML, encoding="utf-8"))
    bad = copy.deepcopy(doc)
    bad["keys"]["scan"]["oct_dec"] = {"level": "no_go"}                         # no reason
    with pytest.raises(ValueError, match="needs a reason"):
        rc.Season(bad)
    bad = copy.deepcopy(doc)
    del bad["keys"]["scan"]["jan_feb"]
    with pytest.raises(ValueError, match="missing"):
        rc.Season(bad)
    bad = copy.deepcopy(doc)
    bad["groups"]["oct_dec"] = [10, 11]
    with pytest.raises(ValueError, match="cover months"):
        rc.Season(bad)


def test_no_tool_change_when_the_tool_is_mounted(reg):
    a = run(reg, "harvest_guava", {"tree_ids": ["G-2"]}, {"expected_n": 2})
    b = run(reg, "harvest_guava", {"tree_ids": ["G-2"]}, {"expected_n": 2}, tool_mounted="T2")
    assert a.estimate["t_s"] - b.estimate["t_s"] == pytest.approx(9.0)
    c = run(reg, "harvest_guava", {"tree_ids": ["G-2"]}, {"expected_n": 2, "method": "twist"}, tool_mounted="T2")
    assert c.estimate["t_s"] == pytest.approx(9.0 + 5.0 + 2 * 22.0)          # TWIST needs the change


def test_tree360_on_ot1_walks_its_arcs_at_mid_under_the_registry_clearance(reg):
    """CR-WP27-03: OT-1 alley_clear_height_m 2.5 m < the high-band head 2.76 m + 0.10 m: the OS 1.2 snake (arcs at
    high) is not allowed; the Ready check runs snake_mid (arcs at mid, high only standing), ~190 s > O-06 165 s."""
    r = run(reg, "tree360", OT1)
    assert r.level == rc.GO and r.adjust == {"crown_walk": "snake_mid"}
    assert r.estimate["t_s"] == pytest.approx(189.6, abs=0.5)
    f = copy.deepcopy(FIELD)
    next(t for t in f["trees"] if t["tree_id"] == "OT-1")["alley_clear_height_m"] = [3.0, 3.0]
    r2 = run(reg, "tree360", OT1, field=f)
    assert r2.adjust == {} and r2.estimate["t_s"] == pytest.approx(149.6, abs=0.5)
    # a gate failure (slope 2.5 deg) wins over the clearance fallback: arcs stowed, not at mid
    r3 = run(reg, "tree360", OT1, cross_slope_deg=2.5)
    assert r3.adjust == {"crown_walk": "stop_and_go"}


# ------------------------------------------------------------------ review fixes (WP27 fixer)
def test_tree360_blocked_annulus_keeps_the_alley_clearance_fallback(reg):
    """A blocked annulus (two_sided) must not swallow the clearance fallback: R1-2 alley 2.5 m < snake head."""
    for tid in ("R1-1", "R1-2"):
        r = run(reg, "tree360", {"tree_ids": [tid]})
        assert r.level == rc.GO and r.adjust == {"arc_mode": "two_sided", "crown_walk": "snake_mid"}, r.to_dict()
        assert r.params["crown_walk"] == "snake_mid" and r.params["arc_mode"] == "two_sided"
        walks = [g for g in reg.get("tree360").gate_profile(r.params) if g["kind"] == "walk"]
        assert max(float(g["ext_m"]) for g in walks) <= 0.6 + 1e-9          # no crown-up walk at high ext 1.6 m


def test_tree360_blocked_annulus_on_a_slope_walks_the_arcs_stowed(reg):
    """WP6 refuses crown-up walking above 2 deg: blocked annulus + 2.5 deg must give stop_and_go, never snake."""
    r = run(reg, "tree360", {"tree_ids": ["R1-2"]}, cross_slope_deg=2.5)
    assert r.adjust == {"arc_mode": "two_sided", "crown_walk": "stop_and_go"}, r.to_dict()
    assert r.params["crown_walk"] == "stop_and_go"
    walks = [g for g in reg.get("tree360").gate_profile(r.params) if g["kind"] == "walk"]
    assert all(float(g["ext_m"]) == 0.0 for g in walks)
    ot = run(reg, "tree360", OT1, cross_slope_deg=2.5)                    # same answer with a clear annulus
    assert ot.params["crown_walk"] == "stop_and_go"


def test_merge_keeps_the_more_conservative_fallback():
    it = rc._Items()
    it.merge({"crown_walk": "stop_and_go"})
    assert it.merge({"crown_walk": "snake_mid"}) == {"crown_walk": "stop_and_go"}
    it.merge({"crown_band": "stowed"})
    it.merge({"crown_band": "low"})
    assert it.adjust["crown_band"] == "stowed"


@pytest.mark.parametrize("field", ["cross_slope_deg", "pitch_deg", "wind_mps", "soc", "disk_free_mb", "light_lux"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_context_fails_closed(reg, field, bad):
    r = run(reg, "tree360", OT1, **{field: bad})
    assert r.level == rc.NO_GO and r.reasons == ["sensor_fault"], r.to_dict()


def test_non_finite_payload_and_position_fail_closed(reg):
    assert run(reg, "harvest_mango", OT1, payload_kg=float("nan")).reasons == ["sensor_fault"]
    assert run(reg, "scan", R1, robot_xy=(float("nan"), 0.0)).reasons == ["sensor_fault"]
    # SAFE PARK relaxes slope / energy / geofence and reads none of these: a dropout does not strand the robot
    assert run(reg, "safe_park", None, cross_slope_deg=float("nan"), soc=float("nan")).level == rc.GO


@pytest.mark.parametrize("month", [0, 13, -1, float("nan")])
def test_out_of_range_month_gives_a_reason_not_an_exception(reg, month):
    r = run(reg, "scan", R1, month=month)
    assert r.level == rc.NO_GO and r.reasons == ["season"]
    tiles = rc.check_all(reg, rc.ReadyContext.nominal(month=month), FIELD)
    assert tiles["scan"].reasons == ["season"]


def test_duplicate_target_trees_are_refused(reg):
    r = run(reg, "harvest_mango", {"tree_ids": ["OT-1", "OT-1", "OT-1"]})
    assert r.level == rc.NO_GO and r.reasons == ["param_invalid"]
    assert run(reg, "scan", {"row_ids": ["R1", "R1"]}).reasons == ["param_invalid"]


def test_multi_tree_estimate_and_geofence_use_each_trees_params(reg):
    one = [run(reg, "tree360", {"tree_ids": [t]}) for t in ("R1-1", "R1-3")]
    both = run(reg, "tree360", {"tree_ids": ["R1-1", "R1-3"]})
    assert both.estimate["t_s"] == pytest.approx(one[0].estimate["t_s"] + one[1].estimate["t_s"], abs=0.05)
    assert both.estimate["disk_mb"] == pytest.approx(52.0)
    pts = rc.path_points(reg.get("tree360"), [FIELD_TREES["R1-1"], FIELD_TREES["R1-3"]], {"r1_m": 9.0},
                         rc.ReadyContext.nominal(), FIELD, [{"r1_m": 1.71}, {"r1_m": 2.0}])
    x3, y3 = FIELD_TREES["R1-3"]["geo"]["x"], FIELD_TREES["R1-3"]["geo"]["y"]
    assert max(((p[0] - x3) ** 2 + (p[1] - y3) ** 2) ** 0.5 for p in pts[17:33]) == pytest.approx(2.6)


FIELD_TREES = {t["tree_id"]: t for t in FIELD["trees"]}
