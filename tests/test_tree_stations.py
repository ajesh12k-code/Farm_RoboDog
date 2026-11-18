"""WP8 tree stations (``slow``): the per-crop task planners end to end on orchard_demo, seed 7, assisted.

Each station builds its own world, mounts the tool through the WP2 rack swap, assigns the tree's fruit to the WP23 pool,
puts the robot at each target's station (assisted ``station_teleport``), runs the task state machine through the
MP 0.4 target loop (WPCAP OMPL transits and parks, 50 Hz TCP servo on the truth target, base pose loop) and lets WP3
place the fruit stalk-down in a cell or the papaya in the cradle. Between trees the tray is emptied by the assisted
``empty_tray`` stand-in for DOCK / UNLOAD. Acceptance (WP8):

* mango one round >= 5/6, pedicel error <= 5 mm; guava clip >= 5/6; twists >= 4/5; papaya at 1.8 kg >= 4/5;
* prune >= 4/5 with 0 keep-wood cuts; banana tidy 0 green cuts;
* no-clearance papaya -> MARK; bagged fruit refused (unit tier: tests/test_tasks_unit.py);
* each station <= 120 s wall (also capped below: 2 CPU / 2 GB runner).
"""
import json
import os
import time

import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core.sim_driver import SimDriver

pytestmark = pytest.mark.slow

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WALL_LIMIT_S = 120.0


@pytest.fixture
def station(monkeypatch):
    """(executor, runner, plans, stations) on a fresh orchard_demo world with WP2 + WP3 in the driver."""
    import atharv_nandi.sim.hooks as hook_registry
    from atharv_nandi.core.task_retry import RetryPolicy
    from atharv_nandi.core.tasks import sim_executor as sx, tree_task
    from atharv_nandi.sim.hooks import tools as ht
    from atharv_nandi.sim.world.modules import trees as trees_mod

    def build(policy=None, **runner_kw):
        monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
        monkeypatch.setenv("NANDI_SEED", "7")
        m = farm_scene.load_farm_model()
        drv = SimDriver(m, hooks=hook_registry.load_all(names=["transfer"]))
        rt = ht.install(drv)
        drv.wait(0.3)
        published = []
        ex = sx.SimTaskExecutor(drv, rt, publish=lambda topic, payload: published.append((topic, payload)))
        runner = sx.TreeRunner(ex, policy=policy or RetryPolicy(per_target=1), **runner_kw)
        cand = json.load(open(os.path.join(REPO, "config", "fields", "orchard_candidates.json")))
        plans = {p["tree_id"]: p for p in trees_mod.plans_for_model(m)}
        return ex, runner, plans, tree_task.frozen_stations(cand), published
    return build


def _report(name, results, extra=""):
    rows = [f"{r['target_id']}:{r['outcome']}:{r['reason']}:att{r['attempts']}:re{r['reapproaches']}" for r in results]
    print(f"\n[{name}] {extra}\n  " + "\n  ".join(rows))


def _harvest(ex, runner, plans, stations, trees, method=None, **job_kw):
    from atharv_nandi.core.tasks import sim_executor as sx, tree_task
    from atharv_nandi.sim.world import crops
    results, delivered, marks = [], [], []
    for tid in trees:
        p = plans[tid]
        job = tree_task.harvest_job(p, crops.load_profile(p["crop"]), method=method, stations=stations,
                                    exclude=ex.rt.pool.harvested, **job_kw)
        res = runner.run_tree(job)
        results.extend(res["results"])
        marks.extend(res["marks"])
        delivered.append(sx.empty_tray(ex))
    return results, delivered, marks


def _final(results):
    """Last attempt per target (a target loop retry runs the target again)."""
    out = {}
    for r in results:
        out[r["target_id"]] = r
    return list(out.values())


def test_mango_one_round_clip(station):
    t0 = time.perf_counter()
    ex, runner, plans, stations, published = station()
    results, delivered, marks = _harvest(ex, runner, plans, stations, ["OT-1", "R1-1"])
    wall = time.perf_counter() - t0
    fin = _final(results)
    picked = [r for r in fin if r["outcome"] == "picked"]
    errs = [abs(r["record"]["pedicel_error_mm"]) for r in picked]
    _report("mango", results, f"picked {len(picked)}/{len(fin)} pedicel err max {max(errs or [0]):.2f} mm "
                              f"wall {wall:.1f} s cycles {[r['cycle_s'] for r in picked]}")
    assert len(fin) == 6                                     # OT-1 3 + R1-1 3 mature in-band mangoes (one round)
    assert len(picked) >= 5
    assert all(e <= 5.0 for e in errs)
    assert all(r["record"]["method"] == "clip" and r["record"]["cell"] is not None for r in picked)
    assert sum(d["n"] for d in delivered) == len(picked)    # every picked mango was seated in its own cell
    # mature fruit above the band are MARKed (pole picker), and the MARKs were published
    up = [mk for mk in marks if mk["kind"] == "fruit" and mk["reason"] == "out_of_reach_marked"]
    assert len(up) >= 10 and all(mk["tool_hint"] == "pole_picker" and mk["h_class"] in ("beech", "upar") for mk in up)
    assert sum(1 for t, _p in published if t == "/nandi/mark") >= len(up)
    # every transit / park was planned by the WPCAP planner (OMPL when the wheel is importable)
    from atharv_nandi.core.planning import ompl_planner
    backend = "ompl" if ompl_planner.ompl_available() else "rrtconnect"
    assert len(ex.plan_stats) >= 12 and all(p["backend"] == backend for p in ex.plan_stats)
    assert sum(p["ok"] for p in ex.plan_stats) >= 12
    print(f"  plans {len(ex.plan_stats)} backend {backend} max {max(p['plan_s'] for p in ex.plan_stats):.3f} s")
    assert wall <= WALL_LIMIT_S


def test_guava_clip(station):
    t0 = time.perf_counter()
    ex, runner, plans, stations, _pub = station()
    results, delivered, _marks = _harvest(ex, runner, plans, stations, ["R2-1", "R2-2", "R2-3"])
    wall = time.perf_counter() - t0
    fin = _final(results)
    picked = [r for r in fin if r["outcome"] == "picked"]
    _report("guava clip", results, f"picked {len(picked)}/{len(fin)} wall {wall:.1f} s")
    assert len(fin) == 6 and len(picked) >= 5
    assert all(abs(r["record"]["pedicel_error_mm"]) <= 5.0 for r in picked)
    assert wall <= WALL_LIMIT_S


def test_guava_twist(station):
    t0 = time.perf_counter()
    ex, runner, plans, stations, _pub = station()
    results, _delivered, _marks = _harvest(ex, runner, plans, stations, ["R2-1", "R2-2", "R2-3"], method="twist",
                                           max_items=2)
    fin = _final(results)[:5]
    wall = time.perf_counter() - t0
    picked = [r for r in fin if r["outcome"] == "picked"]
    _report("guava twist", results, f"picked {len(picked)}/{len(fin)} wall {wall:.1f} s "
                                    f"tau {[r['record'].get('tau_peak_nm') for r in picked]}")
    assert len(fin) == 5 and len(picked) >= 4
    assert all(r["record"]["method"] == "twist" for r in picked)
    assert wall <= WALL_LIMIT_S


def test_papaya_twist_at_1p8kg_and_no_clearance_mark(station):
    """Five papayas at 1.8 kg (one per trip; the cradle is emptied between trips), the second ripe fruit of P-1 (R3-1,
    crowded, WP23 truth clearance -4.4 mm) is MARKed ``no_clearance`` and never touched."""
    from atharv_nandi.core.tasks import sim_executor as sx, tree_task
    from atharv_nandi.sim.world import crops
    t0 = time.perf_counter()
    ex, runner, plans, stations, _pub = station(fruit_mass_kg={"papaya": 1.8})
    results, marks = [], []
    for tid in ("R3-1", "R3-2", "R3-4", "R3-2", "R3-4"):
        p = plans[tid]
        job = tree_task.harvest_job(p, crops.load_profile("papaya"), stations=stations, exclude=ex.rt.pool.harvested)
        res = runner.run_tree(job)
        results.extend(res["results"])
        marks.extend(res["marks"])
        sx.empty_tray(ex)
    wall = time.perf_counter() - t0
    fin = _final(results)
    picked = [r for r in fin if r["outcome"] == "picked"]
    speeds = [max(r["tcp_mps_by_state"].get(s, 0.0) for s in ("carry", "park")) for r in picked]
    _report("papaya 1.8 kg", results, f"picked {len(picked)}/{len(fin)} carry TCP max {max(speeds or [0]):.3f} m/s "
                                      f"torque {[r['precheck'].get('torque_nm') for r in fin]} wall {wall:.1f} s")
    assert len(fin) == 5 and len(picked) >= 4
    assert all(r["record"]["mass_kg"] == pytest.approx(1.8) and r["record"]["cell"] == "cradle" for r in picked)
    assert all(r["precheck"]["torque_nm"] <= 59.5 for r in fin)
    assert all(v <= 0.5 + 0.05 for v in speeds)              # carry and park with the papaya at <= 0.5 m/s TCP
    nc = [mk for mk in marks if mk["reason"] == "no_clearance"]
    assert any(mk["target_id"] == "R3-1:f02" for mk in nc)
    assert all(r["target_id"] != "R3-1:f02" for r in results)
    assert wall <= WALL_LIMIT_S


def test_prune_jobs_no_keep_wood_cuts(station):
    from atharv_nandi.core.tasks import tree_task
    from atharv_nandi.sim.world import crops
    t0 = time.perf_counter()
    ex, runner, plans, stations, _pub = station()
    results, marks = [], []
    for tid in ("OT-1", "R2-1"):
        p = plans[tid]
        job = tree_task.prune_job(p, crops.load_profile(p["crop"]), stations=stations)
        if tid == "R2-1":
            job.targets = job.targets[:1]
        res = runner.run_tree(job)
        results.extend(res["results"])
        marks.extend(res["marks"])
    wall = time.perf_counter() - t0
    fin = _final(results)
    cut = [r for r in fin if r["outcome"] == "cut"]
    jobs = {r["record"].get("job") for r in cut}
    cut_events = [e for e in ex.rt.events if e.get("event") == "cut" and e.get("tool") == "T5"]
    keep = {p["id"] for pl in plans.values() for p in pl["truth"].get("prune", []) if p.get("keep")}
    _report("prune", results, f"cut {len(cut)}/{len(fin)} jobs {sorted(jobs)} marks {len(marks)} wall {wall:.1f} s")
    assert len(fin) == 5 and len(cut) >= 4
    assert {"rootstock_sprout", "skirt_lift"} <= jobs
    assert not [e for e in cut_events if e["target"] in keep]          # 0 keep-wood cuts
    big = [mk for mk in marks if (mk.get("diameter_mm") or 0) > 12.0]
    assert big and all(mk["paste_required"] for mk in big)              # > 12 mm MARKed with paste_required
    assert wall <= WALL_LIMIT_S


def test_banana_tidy_no_green_cuts(station):
    from atharv_nandi.core.tasks import tree_task
    from atharv_nandi.sim.world import crops
    t0 = time.perf_counter()
    ex, runner, plans, stations, _pub = station()
    results, marks, cleaned = [], [], []
    for tid in ("R4-1", "R4-2"):
        p = plans[tid]
        res = runner.run_tree(tree_task.prune_job(p, crops.load_profile("banana"), stations=stations))
        results.extend(res["results"])
        marks.extend(res["marks"])
        cleaned.append(res.get("blade_clean", False))
    wall = time.perf_counter() - t0
    fin = _final(results)
    cut = [r for r in fin if r["outcome"] == "cut"]
    green = {p["id"] for pl in plans.values() for p in pl["truth"].get("prune", []) if p.get("green")}
    green |= {b["id"] for pl in plans.values() for b in pl["truth"].get("blades", []) if b.get("functional")}
    cut_events = [e for e in ex.rt.events if e.get("event") == "cut" and e.get("tool") == "T5"]
    _report("banana tidy", results, f"cut {len(cut)}/{len(fin)} marks {len(marks)} blade_clean {cleaned} "
                                    f"wall {wall:.1f} s")
    assert len(cut) >= 3
    assert not [e for e in cut_events if e["target"] in green]           # 0 green-leaf cuts
    assert cleaned == [False, True]                                     # blade_clean between banana plants
    kinds = {mk["kind"] for mk in marks}
    assert {"sanitation_deleaf", "sucker"} <= kinds
    assert wall <= WALL_LIMIT_S
