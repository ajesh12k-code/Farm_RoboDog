"""WP27 tour gates (ORCHARD-MVP-SPEC 0 rule 14 and 6 "tour gates"; WP27): every scripted step of
config/missions/orchard_tour.yaml (and recovery_demo.yaml) and every preset default satisfies the WP6 gate table
(config/crown_gates.json via controllers.stability), the speed caps and the slope rules; the scripted slopes are
checked against the orchard_demo terrain (six-foot plane fit); a seeded contradiction fails. ROS-free, no MuJoCo.
"""
import copy
import json
import os

import pytest

from atharv_nandi.core import mission_logic as ml
from atharv_nandi.core import presets as ps
from atharv_nandi.core import stability as st

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIELD = json.load(open(os.path.join(REPO, "config", "fields", "orchard_demo.json"), encoding="utf-8"))


@pytest.fixture(scope="module")
def reg():
    r = ps.Registry.load()
    assert r.errors == []
    return r


@pytest.fixture(scope="module")
def grid():
    from atharv_nandi.sim.world import terrain_orchard as to
    return to.grid_for(FIELD)


@pytest.fixture(scope="module")
def table():
    t = st.CrownGateTable.load()
    assert t.run_id, "the WP6 matrix (config/crown_gates.json) must be loaded, not the OS 2.3 fallback"
    return t


@pytest.fixture(scope="module", params=["orchard_tour", "recovery_demo"])
def mission(request):
    return ml.load_mission(request.param)


def test_missions_validate(reg, mission):
    assert ml.validate_mission(mission, reg) == []
    assert mission["world"] == FIELD["world"] and mission["season"] == FIELD["season"]


def test_tour_follows_the_os_12_script(reg):
    tour = ml.load_mission("orchard_tour")
    presets = [s.get("preset") for s in tour["steps"] if s.get("preset")]
    assert presets == ["scan", "tree360", "leaf_scan", "fruit_scout", "harvest_mango", "harvest_guava", "prune",
                       "harvest_guava", "harvest_papaya", "prune_banana", "scan", "estop_recovery", "dock_unload"]
    steps = {s["id"]: s for s in tour["steps"]}
    assert steps["s02_tree360"]["targets"] == {"tree_ids": ["OT-1"]}
    assert steps["s08_harvest_guava_twist"]["params"]["method"] == "twist"
    # 1b: the 5 deg strip is crossed stowed; 10a: gust; 10b: 10 deg patch refusal (the gates must trip)
    assert any(g["kind"] == "walk" and g["slope_deg"] == 5.0 and g["ext_m"] == 0.0 for g in steps["s01b_headland"]["motion"])
    assert {g.get("expect_refused") for g in steps["s10a_gust"]["motion"] if g["kind"] == "event"} == {"wind"}
    assert {g.get("expect_refused") for g in steps["s10b_patch_refusal"]["motion"] if g["kind"] == "event"} == {"slope"}


def test_every_scripted_step_satisfies_the_gate_table(mission, grid, table):
    """Rule 14: no scripted step contradicts the gate table. Segments tagged with a listed known issue are checked
    separately below (they DO contradict it and are tracked as a change request)."""
    viol = ml.tour_gate_violations(mission, FIELD, grid, table)
    open_ = [v for v in viol if not v["known_issue"]]
    for v in viol:
        print(f"  {v['step']}[{v['segment']}] {v['gate']}: {v['reason']} {v['detail']} known={v['known_issue']}")
    assert open_ == [], open_


def test_known_issue_segments_really_violate(mission, grid, table):
    """A known-issue marker must not hide a segment that is fine (a stale marker fails)."""
    tagged = [(s["id"], i) for s in mission["steps"] for i, g in enumerate(s.get("motion") or []) if g.get("known_issue")]
    viol = {(v["step"], v["segment"]) for v in ml.tour_gate_violations(mission, FIELD, grid, table) if v["known_issue"]}
    assert set(tagged) == viol, sorted(set(tagged) ^ viol)


@pytest.mark.xfail(strict=True, reason="CR-WP27-01 (to WP0c / WP8 / WP4): WP0c frozen stations sit on or inside "
                                       "the 0.10 m basin rims: foot-plane tilt 4.1-11.7 deg > the 3 / 5 deg dwell and "
                                       "5 deg walk limits")
def test_known_issue_cr_wp27_01_is_fixed(grid, table):
    tour = ml.load_mission("orchard_tour")
    viol = [v for v in ml.tour_gate_violations(tour, FIELD, grid, table) if v["known_issue"] == "CR-WP27-01"]
    worst = max((float(v["detail"].get("slope_deg", 0)) for v in viol if isinstance(v["detail"], dict)), default=0.0)
    print(f"\n[CR-WP27-01] {len(viol)} violations, worst station dwell tilt {worst:.2f} deg")
    assert viol == []


def test_every_preset_default_satisfies_the_gate_table(reg, table):
    bad = {p.id: ml.preset_gate_violations(p, table) for p in reg}
    assert {k: v for k, v in bad.items() if v} == {}


def test_every_tour_step_is_ready_as_scripted(reg, mission):
    for row in ml.tour_steps(mission, reg, FIELD):
        if not row["preset"]:
            continue
        exp = row["expect"]
        assert row["level"] == exp.get("level", "go"), row
        if exp.get("reason"):
            assert exp["reason"] in row["reasons"], row


def test_tour_crown_up_walking_only_on_validated_ground(grid):
    """Every crown-up walk of the tour: measured terrain <= 2 deg (WP6 policy) and the hull outside every basin rim."""
    tour = ml.load_mission("orchard_tour")
    n = 0
    for s in tour["steps"]:
        for g in s.get("motion") or []:
            if g["kind"] == "walk" and g["ext_m"] > st.CROWN_UP_EXT_M:
                n += 1
                cr, pi = ml.segment_slope(grid, g)
                assert st.slope_tilt_deg(cr, pi) <= st.CROWN_WALK_MAX_SLOPE_DEG, (s["id"], cr, pi)
                assert g["v_mps"] <= st.CROWN_WALK_MAX_MPS
                assert st.gate_crown_path(g["ext_m"], g["path"], ml._basins(FIELD))
    assert n == 3                                     # SCAN R1 (mid), TREE 360 orbit (high), gust walk (mid)


# ------------------------------------------------------------------ seeded contradictions fail
def _walk(**kw):
    g = {"kind": "walk", "ext_m": 0.6, "v_mps": 0.15, "slope_deg": 0.0, "path": [[0.5, 5.0], [3.0, 5.0]]}
    g.update(kw)
    return g


@pytest.mark.parametrize("seg, reason", [
    (_walk(slope_deg=5.0), "slope"),                                        # crown-up walk on 5 deg (WP27 acceptance)
    (_walk(v_mps=0.20), "speed"),                                           # 0.20 m/s crown-up (WP27 acceptance)
    (_walk(ext_m=1.6, v_mps=0.16), "speed"),
    (_walk(slope_deg=2.5), "slope"),                                        # > 2 deg crown-up policy
    (_walk(ext_m=0.6, wind_mps=8.5), "wind"),
    (_walk(ext_m=0.0, v_mps=0.25), "speed"),                                # stowed cap 0.18 m/s
    (_walk(ext_m=0.0, v_mps=0.35), "speed"),                                # cap 0.3 m/s
    (_walk(ext_m=0.0, v_mps=0.18, slope_deg=6.0), "slope"),                 # walking > 5 deg
    (_walk(ext_m=0.6, path=[[4.5, 1.0], [4.5, 4.0]]), "basin_rim"),         # crown-up across the R1-2 basin rim
    (_walk(ext_m=0.0, v_mps=0.18, slope_deg=1.0, path=[[11.5, 5.0], [11.5, 3.0]]), "slope_understated"),  # 5 deg strip
    ({"kind": "dwell", "ext_m": 0.0, "dwell_s": 60.0, "at": [0.0, 0.0], "slope_deg": 4.0}, "slope"),
    ({"kind": "dwell", "ext_m": 0.0, "dwell_s": 15.0, "at": [0.0, 0.0], "slope_deg": 5.5}, "slope"),
    ({"kind": "raise", "ext_m": 0.0, "to_m": 1.6, "at": [0.0, 0.0], "slope_deg": 3.5}, "slope"),
    ({"kind": "raise", "ext_m": 0.0, "to_m": 1.6, "at": [0.0, 0.0], "slope_deg": 0.0, "wind_mps": 10.0}, "wind"),
    ({"kind": "raise", "ext_m": 0.0, "to_m": 0.6, "at": [11.0, 14.5], "slope_deg": 0.0}, "slope_understated"),
    ({"kind": "event", "expect_refused": "wind", "ext_m": 0.6, "v_mps": 0.15, "wind_mps": 5.0, "slope_deg": 0.0},
     "not_refused"),                                                        # a scripted gust that would not trip
    ({"kind": "event", "expect_refused": "slope", "ext_m": 0.6, "v_mps": 0.15, "slope_deg": 1.0}, "not_refused"),
])
def test_seeded_contradiction_fails(seg, reason, grid, table):
    viol = ml.check_segment(seg, grid, FIELD, table)
    assert reason in {v["reason"] for v in viol}, viol


def test_seeded_contradiction_in_the_tour_file_fails(grid, table):
    tour = ml.load_mission("orchard_tour")
    assert [v for v in ml.tour_gate_violations(tour, FIELD, grid, table) if not v["known_issue"]] == []
    for mutate, reason in (
            (lambda s: s["motion"][2].update(slope_deg=5.0), "slope"),          # SCAN R1 crown-up on 5 deg
            (lambda s: s["motion"][2].update(v_mps=0.20), "speed"),             # 0.20 m/s crown-up
            (lambda s: s["motion"][2].update(path=[[0.5, 5.0], [11.5, 5.0], [11.5, 3.0]]), "slope"),  # onto the strip
    ):
        bad = copy.deepcopy(tour)
        mutate(next(s for s in bad["steps"] if s["id"] == "s01_scan_r1"))
        viol = [v for v in ml.tour_gate_violations(bad, FIELD, grid, table) if not v["known_issue"]]
        assert reason in {v["reason"] for v in viol}, viol


def test_seeded_contradiction_in_a_preset_default_fails(reg, table):
    scan = reg.get("scan")
    for gates, reason in (
            ([{"kind": "walk", "ext_m": 0.6, "v_mps": 0.15, "slope_deg": 5.0}], "slope"),
            ([{"kind": "walk", "ext_m": 0.6, "v_mps": 0.20}], "speed"),          # 0.20 m/s crown-up at 2 deg
            ([{"kind": "walk", "ext_m": "$crown_ext_m", "v_mps": "$v_mps"}], "slope"),   # no slope cap: 5 deg limit
            ([{"kind": "dwell", "ext_m": 0.0, "dwell_s": 60.0}], "slope"),       # long dwell at the 5 deg limit
    ):
        doc = copy.deepcopy(scan.doc)
        doc["gates"] = gates
        bad = ps.from_doc(doc)
        assert reason in {v["reason"] for v in ml.preset_gate_violations(bad, table)}, gates


# ------------------------------------------------------------------ time
def test_tour_eta_matches_the_budget_outside_the_walking(reg):
    """ETA +- 20 %: the tour estimate without its transit walks against the OS 1.2 budget without its walking rows
    (the walks are routed on the frozen layout here and are longer, see the next test)."""
    tour = ml.load_mission("orchard_tour")
    est = ml.tour_estimate(tour, reg, FIELD)
    rows = est["steps"]
    work = sum(r["t_s"] - r["transit_s"] for r in rows)
    ref = tour["budget_s"] - tour["budget_walk_s"]
    walk_m = sum(r["transit_m"] for r in rows)
    print(f"\n[tour eta] total {est['total_s']:.0f} s (work {work:.0f} s + transit {est['total_s'] - work:.0f} s, "
          f"{walk_m:.1f} m) vs budget {tour['budget_s']} s (work {ref:.0f} s + walking {tour['budget_walk_s']} s)")
    for r in rows:
        print(f"  {r['id']:24s} {r['t_s']:7.1f} s  transit {r['transit_s']:6.1f} s  budget {r['budget_s']}")
    assert abs(work - ref) <= 0.2 * ref


@pytest.mark.xfail(strict=True, reason="CR-WP27-02 (project decision, OS 1.2): routed on the frozen orchard_demo "
                                       "layout the tour walks ~110 m (OS budget ~41 m) and the ETA is ~24 min > the "
                                       "18 min gate; the gate is re-set by a project decision, not silently")
def test_tour_eta_fits_the_18_min_gate(reg):
    tour = ml.load_mission("orchard_tour")
    est = ml.tour_estimate(tour, reg, FIELD)
    print(f"\n[CR-WP27-02] tour ETA {est['total_s']:.0f} s vs gate {tour['gate_s']} s")
    assert est["total_s"] <= tour["gate_s"]


def test_recovery_demo_about_four_minutes(reg):
    demo = ml.load_mission("recovery_demo")
    est = ml.tour_estimate(demo, reg, FIELD)
    print(f"\n[recovery_demo eta] {est['total_s']:.0f} s vs ~{demo['budget_s']} s (assumption)")
    assert 0.5 * demo["budget_s"] <= est["total_s"] <= 1.5 * demo["budget_s"]
