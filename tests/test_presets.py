"""WP27 mission presets registry (MISSION-PRESETS 0.1, 0.5, 0.7, 0.8; WP27). ROS-free.

Acceptance covered here: every preset validates; bad params refused with a reason; ``cmd/mission`` mapping (one
internal type, ``plan.source``); legacy modes unchanged; the console queue with ETA, SoC and disk; ETA within
+-20 % of the reference times (measured sim runs where they exist, the OS 1.2 computed budget otherwise), and two
``slow`` real-run checks of the ETA model (walking speed, mango one round).
"""
import copy
import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from atharv_nandi.core import mission_logic as ml
from atharv_nandi.core import presets as ps
from atharv_nandi.core import ready_check as rc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIELD = json.load(open(os.path.join(REPO, "config", "fields", "orchard_demo.json"), encoding="utf-8"))
NOW = datetime(2026, 11, 18, 4, 30, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def reg():
    r = ps.Registry.load()
    assert r.errors == [], r.errors
    return r


def _doc(pid):
    with open(os.path.join(ps.PRESETS_DIR, f"{pid}.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _errs(doc):
    return ps.validate_doc(doc, ml.MODES, rc.PRECONDITIONS, rc.season_table().keys)


# ------------------------------------------------------------------ registry and schema
def test_every_preset_file_validates_and_loads(reg):
    files = sorted(f[:-5] for f in os.listdir(ps.PRESETS_DIR) if f.endswith(".yaml"))
    assert sorted(p.id for p in reg) == files
    assert len(reg) >= 21
    for p in reg:
        assert _errs(p.doc) == [], p.id
        for k in ps.REQUIRED:                                  # MP 0.1 fields
            assert k in p.doc, (p.id, k)
        for k in ("tile", "hi", "en", "icon"):
            assert p.console.get(k), (p.id, k)


def test_every_mp_mode_has_a_preset_and_legacy_modes_stay(reg):
    assert ml.LEGACY_MODES == ("harvest_zone", "patrol")
    assert set(ml.MODES) == set(ml.LEGACY_MODES) | set(ml.PRESET_MODES)
    assert len(ml.PRESET_MODES) == 16                          # MP 0.1 list
    used = {p.mode for p in reg if p.enabled}
    assert used == set(ml.PRESET_MODES), set(ml.PRESET_MODES) ^ used


def test_catalogue_covers_the_mp_presets(reg):
    for pid in ("scan", "tree360", "leaf_scan", "trunk_scan", "fruit_scout", "harvest_mango", "harvest_guava",
                "harvest_papaya", "harvest_banana", "prune", "prune_banana", "work_tote", "work_lead", "work_follow",
                "work_spray", "safe_park", "return_home", "stuck_recover", "estop_recovery", "self_right_spike",
                "dock_unload"):
        assert pid in reg, pid
    spray = reg.get("work_spray")
    assert not spray.enabled and spray.disabled_reason == "not_in_mvp"
    # per-crop methods (MP 5) and tools (WP2 ids)
    assert reg.get("harvest_mango").tool_for() == "T2"
    assert reg.get("harvest_guava").tool_for({"method": "twist"}) == "T4"
    assert reg.get("harvest_papaya").tool_for() == "T4" and reg.get("harvest_papaya").pilot_scope == "demo_only"
    assert reg.get("harvest_banana").tool_for() is None and reg.get("harvest_banana").doc["method"] == "mark"
    assert reg.get("prune").tool_for() == "T5"
    assert reg.get("leaf_scan").crops == ("mango", "guava", "banana")   # papaya: n/a crown-only


def test_example_file_semantics_mp_08(reg):
    p = reg.get("tree360")
    ot1 = next(t for t in FIELD["trees"] if t["tree_id"] == "OT-1")
    r = p.resolve({}, ot1)
    assert r["r1_m"] == pytest.approx(1.9)                      # clamp(1.0 + 0.9, 1.2, 3.0)
    assert r["r2_m"] == pytest.approx(2.7)                      # clamp(r1 + 0.8, ...)
    assert p.resolve({"standoff_m": 1.5}, ot1)["r1_m"] == pytest.approx(2.5)
    assert p.resolve({}, None)["r1_m"] is None                  # needs the tree
    assert p.crown == "required" and p.walks_crown_up() and p.max_head_m() == pytest.approx(2.76)
    sg = p.gate_profile(p.resolve({"crown_walk": "stop_and_go"}, ot1))
    assert not any(g["kind"] == "walk" and g["ext_m"] > 0 for g in sg)      # stop-and-go: arcs stowed


def test_registry_reports_invalid_files(tmp_path, reg):
    good = _doc("scan")
    (tmp_path / "scan.yaml").write_text(yaml.safe_dump(good))
    bad = copy.deepcopy(good)
    bad["id"] = "wrong_name"
    (tmp_path / "other.yaml").write_text(yaml.safe_dump(bad))
    (tmp_path / "broken.yaml").write_text("id: [unclosed\n")
    r = ps.Registry.load(str(tmp_path))
    assert list(r.presets) == ["scan"]
    assert any("file name must be <id>.yaml" in e for e in r.errors)
    assert any(e.startswith("broken.yaml") for e in r.errors)


@pytest.mark.parametrize("mutate, needle", [
    (lambda d: d.update(mode="harvest_tomato"), "not in mission_logic.MODES"),
    (lambda d: d.pop("season"), "missing field 'season'"),
    (lambda d: d.update(colour="red"), "unknown field 'colour'"),
    (lambda d: d.update(version=0), "version must be an integer"),
    (lambda d: d.update(season="no_such_key"), "not in config/season/up.yaml"),
    (lambda d: d["preconditions"].append("vibes_ok"), "has no Ready check"),
    (lambda d: d.update(slope_limit_deg=8.0), "slope_limit_deg must be in [0, 5]"),
    (lambda d: d["gates"].append({"kind": "walk", "ext_m": "$nope", "v_mps": 0.1}), "names no param"),
    (lambda d: d["gates"].append({"kind": "hover", "ext_m": 0.0}), "kind not in"),
    (lambda d: d["gates"].append({"kind": "walk", "ext_m": 0.0}), "walk gate needs v_mps"),
    (lambda d: d["gates"].append({"kind": "walk", "ext_m": 0.0, "v_mps": 0.1, "when": {"sides": "up"}}), "enum param"),
    (lambda d: d["estimate"].update(eta_s=3), "estimate key 'eta_s'"),
    (lambda d: d["estimate"].update(fixed_s="__import__('os')"), "not allowed"),
    (lambda d: d["estimate"].update(per_tree_s={"by": "sides", "values": {"L": 1}}), "values must cover"),
    (lambda d: d["params"].update(v_mps={"default": 0.5, "max": 0.15}), "above max"),
    (lambda d: d["params"].update(v_mps={"min": 0.1}), "needs a mapping with 'default'"),
    (lambda d: d.update(crown="sometimes"), "crown 'sometimes'"),
    (lambda d: d.update(weather="snow"), "weather 'snow'"),
    (lambda d: d.update(tool="T9"), "tool 'T9'"),
    (lambda d: d["console"].pop("hi"), "console.hi missing"),
    (lambda d: d["console"].update(show=["ghost"]), "unknown param 'ghost'"),
    (lambda d: d.update(crops=["tomato"]), "crop 'tomato'"),
    (lambda d: d.update(targets=["lasso"]), "target form 'lasso'"),
    (lambda d: d["ready"].update(relax=["gravity"]) if "ready" in d else d.update(ready={"relax": ["gravity"]}),
     "ready.relax 'gravity'"),
])
def test_invalid_preset_documents_are_refused_with_the_reason(mutate, needle):
    d = _doc("scan")
    mutate(d)
    errs = _errs(d)
    assert any(needle in e for e in errs), errs


def test_expressions_cannot_run_code():
    for bad in ("os.system('x')", "__import__('os')", "(lambda: 1)()", "[1][0]", "a.b", "open('x')"):
        with pytest.raises(ValueError):
            ps.Expr(bad)
    assert ps.Expr("clamp(canopy_r+standoff_m,1.2,3.0)").eval({"canopy_r_m": 2.5, "standoff_m": 0.9}) == 3.0


# ------------------------------------------------------------------ params and targets refused with a reason
@pytest.mark.parametrize("pid, params, param", [
    ("scan", {"v_mps": 0.20}, "v_mps"),                       # crown up <= 0.15 m/s
    ("scan", {"v_mps": "fast"}, "v_mps"),
    ("scan", {"v_mps": True}, "v_mps"),
    ("scan", {"v_mps": float("nan")}, "v_mps"),
    ("scan", {"sides": "both_and_up"}, "sides"),
    ("scan", {"warp": 9}, "warp"),
    ("tree360", {"stations": 2}, "stations"),
    ("tree360", {"bands": ["low", "ultra"]}, "bands"),
    ("tree360", {"bands": []}, "bands"),
    ("tree360", {"standoff_m": 0.5}, "standoff_m"),
    ("harvest_mango", {"pedicel_leave_cm": 6.0}, "pedicel_leave_cm"),
    ("harvest_mango", {"method": "twist"}, "method"),          # mango is CLIP only
    ("harvest_papaya", {"max_items": 2}, "max_items"),         # one papaya per trip
    ("prune", {"max_stem_mm": 20.0}, "max_stem_mm"),           # PRUNER capacity 12 mm
    ("prune", {"jobs": ["fell_tree"]}, "jobs"),
    ("work_tote", {"tote_kg": 5.0}, "tote_kg"),                # <= 3 kg (not a field crate)
    ("work_follow", {"v_mps": 0.3}, "v_mps"),
    ("leaf_scan", {"min_clearance_m": 0.01}, "min_clearance_m"),
])
def test_bad_params_refused_with_reason(reg, pid, params, param):
    with pytest.raises(ps.PresetError) as ei:
        reg.get(pid).resolve(params)
    assert ei.value.reason == "param_invalid" and ei.value.param == param
    assert ei.value.to_dict()["param"] == param


@pytest.mark.parametrize("pid, targets, reason", [
    ("tree360", {"tree_ids": ["XX-9"]}, "registry_mismatch"),
    ("scan", {"row_ids": ["R9"]}, "registry_mismatch"),
    ("scan", {"block_id": "B9"}, "registry_mismatch"),
    ("harvest_mango", {"tree_ids": ["G-2"]}, "not_in_mvp"),       # guava tree for the mango preset
    ("leaf_scan", {"tree_ids": ["P-1"]}, "not_in_mvp"),           # papaya: n/a crown-only
    ("scan", {"row_ids": ["R1"], "tree_ids": ["OT-1"]}, "param_invalid"),
    ("tree360", {"row_ids": ["R1"]}, "param_invalid"),            # form not accepted
    ("scan", {"block_sample": {"block_id": "B1-mango-hdp", "n": 9}}, "param_invalid"),
    ("safe_park", {"tree_ids": ["OT-1"]}, "param_invalid"),       # takes no targets
])
def test_bad_targets_refused_with_reason(reg, pid, targets, reason):
    with pytest.raises(ps.PresetError) as ei:
        ps.resolve_targets(reg.get(pid), FIELD, targets)
    assert ei.value.reason == reason


def test_targets_resolve_by_id_tag_row_block_and_sample(reg):
    scan = reg.get("scan")
    assert [t["tree_id"] for t in ps.resolve_targets(scan, FIELD, {"row_ids": ["R1"]})] == ["R1-1", "R1-2", "R1-3", "R1-4"]
    assert [t["tree_id"] for t in ps.resolve_targets(scan, FIELD, {"tree_ids": ["M-2"]})] == ["R1-2"]   # physical tag
    blk = ps.resolve_targets(scan, FIELD, {"block_id": "B2-guava"})
    assert len(blk) == 4 and {t["crop"] for t in blk} == {"guava"}
    s = ps.resolve_targets(scan, FIELD, {"block_sample": {"block_id": "B1-mango-hdp", "n": 2}})
    assert len(s) == 2 and len({t["tree_id"] for t in s}) == 2
    assert ps.route_m(ps.resolve_targets(scan, FIELD, {"row_ids": ["R1"]}), FIELD) == pytest.approx(10.0)


# ------------------------------------------------------------------ cmd/mission mapping
def test_preset_start_maps_to_one_cmd_mission_type(reg):
    for src in ("console", "zone_alert", "edge_hub"):
        cmd = ml.from_preset_start({"preset": "tree360", "targets": {"tree_ids": ["OT-1"]}, "params": {"stations": 8}},
                                   reg, FIELD, source=src, now=NOW)
        assert cmd["mode"] == "tree360" and cmd["mission_id"] == "nandi-sim-20261118T0430Z"
        assert cmd["plan"]["source"] == src and cmd["plan"]["preset"] == "tree360" and cmd["plan"]["version"] == 2
        assert cmd["plan"]["tree_ids"] == ["OT-1"] and cmd["plan"]["resolved"]["r1_m"] == pytest.approx(1.9)
        mc = ml.parse_command(json.dumps(cmd), NOW)
        assert mc.mode == "tree360" and mc.zone is None and mc.plan["preset"] == "tree360"
    every = {p.mode for p in reg if p.enabled}
    for mode in every:                                         # all modes parse through the same type
        mc = ml.parse_command({"mission_id": "m", "mode": mode, "not_after": (NOW + timedelta(minutes=5)).isoformat(),
                               "plan": {"source": "console", "preset": "x"}}, NOW)
        assert mc.mode == mode


@pytest.mark.parametrize("msg, reason", [
    ({"preset": "nope"}, "preset_unknown"),
    ({"preset": "scan", "targets": {"row_ids": ["R1"]}, "params": {"v_mps": 0.3}}, "param_invalid"),
    ({"preset": "harvest_mango", "targets": {"tree_ids": ["P-1"]}}, "not_in_mvp"),
    ({"preset": "work_spray"}, "not_in_mvp"),
    ({"preset": 3}, "command_invalid"),
    ("{not json", "command_invalid"),
])
def test_preset_start_refusals_carry_the_reason(reg, msg, reason):
    with pytest.raises(ml.Rejection) as ei:
        ml.from_preset_start(msg, reg, FIELD, now=NOW)
    assert ei.value.reason == reason


@pytest.mark.parametrize("plan, detail", [
    (None, "needs a plan"),
    ({"source": "whatsapp", "preset": "scan"}, "plan.source"),
    ({"source": "console"}, "plan.preset"),
    ({"source": "console", "preset": "scan", "params": [1]}, "plan.params"),
])
def test_parse_command_refuses_bad_plans(plan, detail):
    raw = {"mission_id": "m", "mode": "scan_rows", "not_after": (NOW + timedelta(minutes=5)).isoformat(), "plan": plan}
    with pytest.raises(ml.Rejection) as ei:
        ml.parse_command(raw, NOW)
    assert ei.value.reason == "command_invalid" and detail in ei.value.detail


def _legacy_controller():
    from atharv_nandi.core import geo_frame as gf
    origin, cfg = gf.load_site()
    return ml.MissionController(ml.config_from_site(origin, cfg, None))


def test_legacy_executor_unchanged_and_refuses_preset_modes(reg):
    c = _legacy_controller()
    obs = ml.Observation(t=0.0, utc=NOW, pose=(0.0, 0.0, 0.0), in_basket=0)
    cmd = ml.from_preset_start({"preset": "scan", "targets": {"row_ids": ["R1"]}}, reg, FIELD, now=NOW)
    ok, reason, detail = c.submit(json.dumps(cmd), obs)
    assert not ok and reason == "command_invalid" and "PresetQueue" in detail
    # a legacy harvest_zone command still goes through the legacy path (here refused for its own legacy reason)
    leg = {"mission_id": "m1", "mode": "harvest_zone", "zone": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
           "not_after": (NOW - timedelta(minutes=1)).isoformat()}
    ok, reason, _ = c.submit(json.dumps(leg), obs)
    assert not ok and reason == "expired"


# ------------------------------------------------------------------ queue with ETA, SoC, disk
def _ready(reg, pid, targets=None, params=None, **ctx):
    c = rc.ReadyContext.nominal(season_profile="demo_all_fruiting", month=11, frames_age_h=0.1, **ctx)
    return rc.check(reg.get(pid), c, targets, params, FIELD)


def _enqueue(q, reg, pid, targets=None, params=None, **ctx):
    r = _ready(reg, pid, targets, params, **ctx)
    cmd = ps.to_mission_command(reg.get(pid), targets, params, now=NOW, field_doc=FIELD)
    trees = ps.resolve_targets(reg.get(pid), FIELD, targets) if targets else []
    xy = (trees[0]["geo"]["x"], trees[0]["geo"]["y"]) if trees else (0.0, 0.0)
    end = (trees[-1]["geo"]["x"], trees[-1]["geo"]["y"]) if trees else xy
    return q.add(cmd, r.estimate, xy, end, r.level, r.reasons)


def test_queue_projects_eta_soc_disk_and_reorders(reg):
    q = ml.PresetQueue()
    a = _enqueue(q, reg, "scan", {"row_ids": ["R1"]})
    b = _enqueue(q, reg, "tree360", {"tree_ids": ["OT-1"]})
    c = _enqueue(q, reg, "harvest_mango", {"tree_ids": ["OT-1"]}, {"expected_n": 3})
    pr = q.projection(soc=1.0, disk_free_mb=4096.0, robot_xy=(0.0, 0.0))
    rows = pr["items"]
    assert [r["qid"] for r in rows] == [a, b, c]
    assert rows[0]["eta_start_s"] < rows[0]["eta_end_s"] <= rows[1]["eta_start_s"] < rows[2]["eta_end_s"]
    assert rows[0]["soc_end"] > rows[1]["soc_end"] > rows[2]["soc_end"] > 0.9
    assert rows[1]["disk_free_end_mb"] == pytest.approx(4096.0 - 2.0 * 4 - 26.0, abs=0.2)
    assert pr["blocked_at"] is None and pr["eta_s"] == pytest.approx(rows[-1]["eta_end_s"])
    q.reorder([c, a, b])
    assert [i.qid for i in q.items] == [c, a, b]
    with pytest.raises(ml.Rejection):
        q.reorder([a, b])
    assert q.remove(a) and len(q) == 2 and q.pop(1.0, 4096.0, (0.0, 0.0)).qid == c


def test_queue_blocks_on_energy_reserve_and_disk(reg):
    q = ml.PresetQueue()
    _enqueue(q, reg, "scan", {"row_ids": ["R1"]})
    _enqueue(q, reg, "tree360", {"tree_ids": ["OT-1"]})
    low = q.projection(soc=0.17, disk_free_mb=4096.0, robot_xy=(0.0, 0.0))
    assert "energy_reserve" in low["items"][-1]["blocked"] and low["blocked_at"] is not None
    disk = q.projection(soc=1.0, disk_free_mb=30.0, robot_xy=(0.0, 0.0))
    assert disk["items"][0]["blocked"] == [] and disk["items"][1]["blocked"] == ["disk_low"]   # 26 MB x 1.5 > 22


def test_queue_refuses_no_go_and_overflow(reg):
    q = ml.PresetQueue(max_len=2)
    with pytest.raises(ml.Rejection) as ei:
        _enqueue(q, reg, "scan", {"row_ids": ["R1"]}, cross_slope_deg=10.0)
    assert ei.value.reason == "slope"
    _enqueue(q, reg, "scan", {"row_ids": ["R1"]})
    _enqueue(q, reg, "scan", {"row_ids": ["R1"]})
    assert q.items[0].cmd["mission_id"] != q.items[1].cmd["mission_id"]
    with pytest.raises(ml.Rejection) as ei:
        _enqueue(q, reg, "scan", {"row_ids": ["R1"]})
    assert ei.value.reason == "queue_full"
    assert q.status(1.0, 4096.0, (0.0, 0.0))["n"] == 2


# ------------------------------------------------------------------ ETA +- 20 %
# (preset, targets, params, reference s, source). Measured references first; the rest are the OS 1.2 computed
# budget rows minus their walking share (the walk between presets is timed by the queue / tour, not the preset).
ETA_REFS = [
    ("harvest_mango", {"tree_ids": ["OT-1"]}, {"expected_n": 3}, 74.2,
     "measured: WP8 TreeRunner OT-1, mount + 3 picks, 74.2 s sim (runner 2026-11-18; the slow test below re-runs it)"),
    ("dock_unload", None, {"items": 3}, 100.8, "measured: WP13 slow DOCK/UNLOAD, 3 items, 100.8 s sim"),
    ("scan", {"row_ids": ["R1"]}, {"crown_band": "mid"}, 85.0, "computed: OS 1.2 SCAN R1"),
    ("tree360", {"tree_ids": ["OT-1"]}, {}, 150.0, "computed: OS 1.2 TREE 360 1 radius"),
    ("tree360", {"tree_ids": ["OT-1"]}, {"crown_walk": "stop_and_go"}, 232.0, "computed: OS 1.2 stop-and-go"),
    ("leaf_scan", {"tree_ids": ["OT-1"]}, {"n_leaves": 6, "n_high": 1}, 90.0, "computed: OS 1.2 LEAF SCAN"),
    ("fruit_scout", {"tree_ids": ["OT-1"]}, {}, 10.0, "computed: OS 1.2 FRUIT SCOUT reuse"),
    ("harvest_mango", {"tree_ids": ["OT-1"]}, {"expected_n": 3}, 66.0, "computed: OS 1.2 tool change + 3 clips"),
    ("harvest_guava", {"tree_ids": ["G-2"]}, {"expected_n": 2, "tool_change_s": 0.0}, 89.0 - 8.0 / 0.18,
     "computed: OS 1.2 R2 row - walk (CLIPPER stays)"),
    ("prune", {"tree_ids": ["G-2"]}, {"expected_n": 2}, 46.0, "computed: OS 1.2 tool change + 2 prunes"),
    ("harvest_guava", {"tree_ids": ["G-3"]}, {"method": "twist", "expected_n": 1}, 31.0, "computed: OS 1.2 twist"),
    ("harvest_papaya", {"tree_ids": ["P-1"]}, {"expected_n": 1}, 63.0 - 6.0 / 0.18, "computed: OS 1.2 R3 - walk"),
    ("prune_banana", {"tree_ids": ["B-2"]}, {"expected_n": 1}, 79.0 - 6.0 / 0.18, "computed: OS 1.2 R4 - walk"),
    ("estop_recovery", None, {"confirm_s": 5.0}, 50.0 - 11.0 - 10.0, "computed: OS 1.2 RECOVERY (e)"),
]


@pytest.mark.parametrize("pid, targets, params, ref_s, source", ETA_REFS, ids=[f"{r[0]}-{r[3]:.0f}" for r in ETA_REFS])
def test_eta_within_20_percent_of_reference(reg, pid, targets, params, ref_s, source):
    p = reg.get(pid)
    trees = ps.resolve_targets(p, FIELD, targets) if targets else []
    est = p.estimate(p.resolve(params, trees[0] if trees else None), n_trees=max(1, len(trees)),
                     route_m=rc.route_for(p, trees, rc.ReadyContext(), FIELD), tree=trees[0] if trees else None)
    print(f"\n[eta] {pid} {params}: estimate {est['t_s']:.1f} s vs {ref_s:.1f} s ({source})")
    assert abs(est["t_s"] - ref_s) <= 0.2 * ref_s, (est, ref_s, source)


def test_estimate_needs_the_tree_for_tree_expressions(reg):
    with pytest.raises(ps.PresetError) as ei:
        reg.get("tree360").estimate(None, tree=None)
    assert ei.value.reason == "param_invalid"


@pytest.mark.slow
def test_eta_walking_speed_matches_a_real_walk(monkeypatch):
    """The queue / tour transit model (0.18 m/s stowed) against a 4 m straight walk of the real gait on
    orchard_lite at the transit command (ml.TRANSIT_CMD_VX)."""
    monkeypatch.setenv("NANDI_WORLD", "orchard_lite")
    monkeypatch.setenv("NANDI_SEED", "7")
    from atharv_nandi.core import farm_scene
    from atharv_nandi.core.sim_driver import SimDriver
    d = SimDriver(farm_scene.load_farm_model())
    d.wait(0.5)
    t0 = time.perf_counter()
    r = d.walk_distance(4.0, vx=ml.TRANSIT_CMD_VX, timeout_s=60.0)
    eta = 4.0 * 1.0 / ml.TRANSIT_V_MPS
    print(f"\n[eta walk] cmd {ml.TRANSIT_CMD_VX}: {r['dist_m']:.3f} m in {r['sim_s']:.2f} s sim "
          f"({r['dist_m'] / r['sim_s']:.3f} m/s), ETA {eta:.2f} s, wall {time.perf_counter() - t0:.1f} s")
    assert r["ok"]
    assert abs(r["sim_s"] - eta) <= 0.2 * r["sim_s"]


@pytest.mark.slow
def test_eta_mango_one_round_matches_the_wp8_run(monkeypatch, reg):
    """HARVEST mango on OT-1 through the WP8 TreeRunner (assisted station_teleport, so no walking): sim time from
    the tool mount to the last placed fruit vs the preset estimate for the delivered count."""
    import atharv_nandi.sim.hooks as hook_registry
    from atharv_nandi.core import farm_scene
    from atharv_nandi.core.sim_driver import SimDriver
    from atharv_nandi.core.task_retry import RetryPolicy
    from atharv_nandi.core.tasks import sim_executor as sx, tree_task
    from atharv_nandi.sim.hooks import tools as ht
    from atharv_nandi.sim.world import crops
    from atharv_nandi.sim.world.modules import trees as trees_mod
    monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
    monkeypatch.setenv("NANDI_SEED", "7")
    m = farm_scene.load_farm_model()
    drv = SimDriver(m, hooks=hook_registry.load_all(names=["transfer"]))
    rt = ht.install(drv)
    drv.wait(0.3)
    ex = sx.SimTaskExecutor(drv, rt, publish=lambda topic, payload: None)
    runner = sx.TreeRunner(ex, policy=RetryPolicy(per_target=1))
    cand = json.load(open(os.path.join(REPO, "config", "fields", "orchard_candidates.json")))
    plans = {p["tree_id"]: p for p in trees_mod.plans_for_model(m)}
    job = tree_task.harvest_job(plans["OT-1"], crops.load_profile("mango"), stations=tree_task.frozen_stations(cand))
    t0, w0 = drv.t, time.perf_counter()
    res = runner.run_tree(job)
    sim_s = drv.t - t0
    picked = [r for r in res["results"] if r["outcome"] == "picked"]
    p = reg.get("harvest_mango")
    est = p.estimate(p.resolve({"expected_n": len(picked)}), n_trees=1)
    print(f"\n[eta mango] picked {len(picked)}/{len(res['results'])} in {sim_s:.1f} s sim "
          f"(cycles {[r['cycle_s'] for r in picked]}), ETA {est['t_s']:.1f} s, wall {time.perf_counter() - w0:.1f} s")
    assert len(picked) >= 2
    assert abs(est["t_s"] - sim_s) <= 0.2 * sim_s


def test_queue_needs_a_confirm_for_degraded_items(reg):
    q = ml.PresetQueue()
    r = _ready(reg, "tree360", {"tree_ids": ["OT-1"]}, weather={**rc.ReadyContext().weather_state(), "temp_c": 39.5})
    assert r.level == "degraded"
    cmd = ps.to_mission_command(reg.get("tree360"), {"tree_ids": ["OT-1"]}, now=NOW, field_doc=FIELD)
    with pytest.raises(ml.Rejection) as ei:
        q.add(cmd, r.estimate, (6.0, -3.5), (6.0, -3.5), r.level, r.reasons)
    assert ei.value.reason == "confirm_required"
    assert q.add(cmd, r.estimate, (6.0, -3.5), (6.0, -3.5), r.level, r.reasons, confirmed=True) == "q1"


# ------------------------------------------------------------------ review fixes (WP27 fixer)
def test_preset_start_without_a_registry_refuses_targets(reg):
    for msg in ({"preset": "harvest_mango", "targets": {"tree_ids": ["NOPE-99"]}},
                {"preset": "harvest_mango", "targets": {"tree_ids": ["OT-1"]}}):
        with pytest.raises(ml.Rejection) as ei:
            ml.from_preset_start(msg, reg, now=NOW)                      # field_doc None
        assert ei.value.reason == "registry_mismatch"
    with pytest.raises(ml.Rejection) as ei:
        ml.from_preset_start({"preset": "harvest_mango", "targets": {"tree_ids": ["NOPE-99"]}}, reg, FIELD, now=NOW)
    assert ei.value.reason == "registry_mismatch"
    with pytest.raises(ml.Rejection) as ei:
        ml.from_preset_start({"preset": "harvest_mango", "targets": {"tree_ids": ["B-2"]}}, reg, FIELD, now=NOW)
    assert ei.value.reason == "not_in_mvp"
    cmd = ml.from_preset_start({"preset": "return_home"}, reg, now=NOW)  # no targets: no registry needed
    assert cmd["plan"]["tree_ids"] == []


def test_preset_start_refuses_duplicate_trees_and_resolves_per_tree(reg):
    with pytest.raises(ml.Rejection) as ei:
        ml.from_preset_start({"preset": "harvest_mango", "targets": {"tree_ids": ["OT-1", "OT-1"]}}, reg, FIELD,
                             now=NOW)
    assert ei.value.reason == "param_invalid"
    cmd = ml.from_preset_start({"preset": "tree360", "targets": {"tree_ids": ["R1-1", "R1-3"]}}, reg, FIELD, now=NOW)
    by = cmd["plan"]["resolved_by_tree"]
    assert by["R1-1"]["r1_m"] == pytest.approx(1.71) and by["R1-3"]["r1_m"] == pytest.approx(2.0)
    assert cmd["plan"]["resolved"] == by["R1-1"]


@pytest.mark.parametrize("expr", ["9**9**9", "2**10"])
def test_expressions_have_no_power_operator(expr):
    with pytest.raises(ValueError):
        ps.Expr(expr)
    doc = copy.deepcopy(yaml.safe_load(open(os.path.join(REPO, "config", "presets", "scan.yaml"), encoding="utf-8")))
    doc.setdefault("estimate", {})["fixed_s"] = expr
    assert any("not allowed" in e for e in ps.validate_doc(doc, ml.MODES, rc.PRECONDITIONS, rc.season_table().keys))


def test_expression_errors_are_param_invalid_not_crashes(reg):
    assert ps.Expr("1 / (x - 1)").names == {"x"}
    with pytest.raises(ValueError):
        ps.Expr("1 / (x - 1)").eval({"x": 1.0})
    with pytest.raises(ps.PresetError) as ei:
        ps.Preset._num("1 / (x - 1)", {"x": 1.0}, {}, "t")
    assert ei.value.reason == "param_invalid"


def test_queue_add_needs_a_ready_level(reg):
    q = ml.PresetQueue()
    with pytest.raises(ml.Rejection) as ei:
        q.add({"mission_id": "x"}, {}, (0, 0), (0, 0))
    assert ei.value.reason == "not_checked" and len(q) == 0
    with pytest.raises(ml.Rejection):
        q.add({"mission_id": "x"}, {}, (0, 0), (0, 0), "green")


def test_queue_pop_refuses_a_blocked_head_item(reg):
    q = ml.PresetQueue()
    q.add({"mission_id": "a", "plan": {"preset": "scan"}}, {"t_s": 100, "walk_m": 10, "disk_mb": 5000}, (0, 0),
          (10, 0), "go")
    pr = q.projection(0.1, 100.0, (0.0, 0.0))
    assert pr["blocked_at"] == "q1" and pr["items"][0]["blocked"] == ["energy_reserve", "disk_low"]
    with pytest.raises(ml.Rejection) as ei:
        q.pop(0.1, 100.0, (0.0, 0.0))
    assert ei.value.reason == "energy_reserve" and len(q) == 1      # stays queued
    with pytest.raises(ml.Rejection) as ei:
        q.pop(1.0, 100.0, (0.0, 0.0))
    assert ei.value.reason == "disk_low"
    with pytest.raises(ml.Rejection) as ei:                          # NaN SoC fails closed
        q.pop(float("nan"), 1e6, (0.0, 0.0))
    assert ei.value.reason == "energy_reserve"
    with pytest.raises(ml.Rejection) as ei:                          # Ready check re-run at dispatch
        q.pop(1.0, 1e6, (0.0, 0.0), recheck=lambda item: ("no_go", ["wind"]))
    assert ei.value.reason == "wind"
    with pytest.raises(ml.Rejection) as ei:
        q.pop(1.0, 1e6, (0.0, 0.0), recheck=lambda item: ("degraded", ["heat"]))
    assert ei.value.reason == "confirm_required"
    assert q.pop(1.0, 1e6, (0.0, 0.0), recheck=lambda item: ("go", [])).qid == "q1"
    assert q.pop(1.0, 1e6, (0.0, 0.0)) is None


def test_queue_ready_recheck_with_a_ready_result(reg):
    q = ml.PresetQueue()
    _enqueue(q, reg, "scan", {"row_ids": ["R1"]})
    with pytest.raises(ml.Rejection) as ei:
        q.pop(1.0, 4096.0, (0.0, 0.0), recheck=lambda item: _ready(reg, "scan", {"row_ids": ["R1"]}, rtk="NONE"))
    assert ei.value.reason == "rtk"


def test_queue_is_thread_safe():
    import threading
    q = ml.PresetQueue(max_len=10_000)

    def worker():
        for k in range(500):
            q.add({"mission_id": ""}, {}, (0, 0), (0, 0), "go")
            if k % 100 == 0:
                q.status(1.0, 1e6, (0.0, 0.0))

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ids = [i.qid for i in q.items]
    assert len(ids) == 2000 and len(set(ids)) == 2000
