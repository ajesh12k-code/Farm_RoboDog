"""WP8 executor interlocks on orchard_demo, seed 7, assisted (``slow``): the person-in-WORK-field pause during the
OMPL transit and the re-approach from the pre-pose after it, the T4 slip response (lower before open) and the
executor timeout cleanup. Each test builds its own world, mounts the tool through the WP2 rack swap and runs one
target through ``SimTaskExecutor.run`` (review round 1 of WP8)."""
import json
import os

import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core.sim_driver import SimDriver

pytestmark = pytest.mark.slow

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def world(monkeypatch):
    import atharv_nandi.sim.hooks as hook_registry
    from atharv_nandi.core.tasks import sim_executor as sx, tree_task
    from atharv_nandi.sim.hooks import tools as ht
    from atharv_nandi.sim.world.modules import trees as trees_mod

    def build(**runner_kw):
        monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
        monkeypatch.setenv("NANDI_SEED", "7")
        m = farm_scene.load_farm_model()
        drv = SimDriver(m, hooks=hook_registry.load_all(names=["transfer"]))
        rt = ht.install(drv)
        drv.wait(0.3)
        ex = sx.SimTaskExecutor(drv, rt)
        runner = sx.TreeRunner(ex, **runner_kw)
        cand = json.load(open(os.path.join(REPO, "config", "fields", "orchard_candidates.json")))
        plans = {p["tree_id"]: p for p in trees_mod.plans_for_model(m)}
        return ex, runner, plans, tree_task.frozen_stations(cand)
    return build


def _first_target(ex, runner, plans, stations, tree_id, crop):
    """Mount, assign and station the tree's first harvest target; returns it."""
    from atharv_nandi.core.tasks import tree_task
    from atharv_nandi.sim.world import crops
    job = tree_task.harvest_job(plans[tree_id], crops.load_profile(crop), stations=stations,
                                exclude=ex.rt.pool.harvested)
    tg = job.targets[0]
    assert ex.mount(tg.tool)
    runner.assign(job)
    assert runner.station_fn(tg) is not None
    ex._cur_state = ""
    return tg


def _arm_q(ex):
    return np.array(ex.d.qpos[ex.ik.qadr], float)


def test_person_during_the_transit_pauses_the_arm_and_replans(world):
    """A person enters the WORK field 0.1 s into the OMPL transit and stays 2.3 s: the transit stops, the SM pauses
    (person_hold), the arm stays put, and once the field clears it re-plans and picks."""
    ex, runner, plans, stations = world()
    tg = _first_target(ex, runner, plans, stations, "OT-1", "mango")
    win = {"t0": None}
    seen = []

    def person():
        if win["t0"] is None and ex._cur_state == "plan":
            win["t0"] = ex.drv.t
        on = win["t0"] is not None and win["t0"] + 0.1 <= ex.drv.t <= win["t0"] + 2.4
        if on:
            seen.append((ex.drv.t, _arm_q(ex), ex._cur_state))
        return on
    ex.person_fn = person
    n_plans = len(ex.plan_stats)
    out = ex.run(runner._sm(tg))
    hist = [(h[1], h[2]) for h in out["history"]]
    print(f"\n[person transit] outcome {out['outcome']}:{out['reason']} ticks {len(seen)} history {hist}")
    assert seen and ("plan", "person_hold") in hist and ("person_hold", "plan") in hist
    settle = [q for t, q, _s in seen if t >= seen[0][0] + 0.3]          # 0.3 s for the PD to stop the arm
    moved = max(float(np.max(np.abs(q - settle[0]))) for q in settle)
    print(f"  arm moved {moved:.4f} rad while the person was in the field")
    assert moved < 0.05
    assert len(ex.plan_stats) - n_plans >= 3            # transit, re-planned transit, park
    assert out["outcome"] == "picked"


def test_person_at_the_fruit_reapproaches_from_the_pre_pose(world):
    """A person for 1 s on entering ``align``: after the hold the SM re-plans (a new OMPL transit back to the 10 cm
    pre-pose) and approaches again, instead of closing in place."""
    ex, runner, plans, stations = world()
    tg = _first_target(ex, runner, plans, stations, "OT-1", "mango")
    win = {"t0": None}

    def person():
        if win["t0"] is None and ex._cur_state == "align":
            win["t0"] = ex.drv.t
        return win["t0"] is not None and win["t0"] <= ex.drv.t <= win["t0"] + 1.0
    ex.person_fn = person
    n_plans = len(ex.plan_stats)
    out = ex.run(runner._sm(tg))
    hist = out["history"]
    names = [h[2] for h in hist]
    print(f"\n[person align] outcome {out['outcome']}:{out['reason']} history {[(h[0], h[1], h[2]) for h in hist]}")
    k = names.index("person_hold")
    assert names[k + 1:k + 4] == ["plan", "roll_align", "approach"]
    t_plan, t_roll = hist[k + 1][0], hist[k + 2][0]
    assert t_roll - t_plan >= 0.3                      # a real transit ran (it used to finish in one tick)
    assert len(ex.plan_stats) - n_plans >= 3            # first transit, re-plan, park
    assert out["outcome"] == "picked"


def test_t4_slip_lowers_the_papaya_before_opening(world, monkeypatch):
    """A forced slip 0.3 s into the papaya carry (1.8 kg): the executor lowers the TCP before the jaws open."""
    from atharv_nandi.core.tasks import sim_executor as sx
    from atharv_nandi.core.tasks.twist_pick import TwistPickSM
    ex, runner, plans, stations = world(fruit_mass_kg={"papaya": 1.8})
    rec = {}

    class SlipSM(TwistPickSM):
        def slipped(self, obs):
            if self.state == "carry" and self.elapsed(obs) >= 0.3:
                rec.setdefault("z_slip", float(ex.tcp("T4")[2]))
                return True
            return super().slipped(obs)
    monkeypatch.setitem(sx.SM_CLASSES, "twist_pick", SlipSM)
    tg = _first_target(ex, runner, plans, stations, "R3-2", "papaya")
    opens = []
    orig = ex.tool_open

    def tool_open():
        opens.append((ex._cur_state, float(ex.tcp("T4")[2])))
        orig()
    ex.tool_open = tool_open
    out = ex.run(runner._sm(tg))
    print(f"\n[slip] outcome {out['outcome']}:{out['reason']} z_slip {rec.get('z_slip')} opens {opens[-2:]}")
    assert (out["outcome"], out["reason"]) == ("dropped", "dropped")
    z_open = opens[-1][1]
    tcp = ex.tcp("T4")
    floor = sx.ground_z(ex.m, ex.d, float(tcp[0]), float(tcp[1]), z0=z_open - sx.LOWER_RAY_BELOW_M)
    assert opens[-1][0] == "done"
    assert z_open <= rec["z_slip"] - 0.10 or z_open <= floor + sx.LOWER_CLEAR_M + 0.03


def test_executor_timeout_opens_parks_and_reports(world):
    """The SM does not finish in time (here 3 s, mid-transit): the executor opens the tool, parks the arm and the
    target ends ``failed`` / ``timeout``; the target loop skips it (reason timeout, not retried)."""
    from atharv_nandi.core.task_retry import TargetLoop
    from atharv_nandi.core.tasks import sim_executor as sx
    ex, runner, plans, stations = world()
    tg = _first_target(ex, runner, plans, stations, "OT-1", "mango")
    out = ex.run(runner._sm(tg), timeout_s=3.0)
    print(f"\n[timeout] {out['outcome']}:{out['reason']} cleanup {out.get('cleanup')}")
    assert out["timeout"] and (out["outcome"], out["reason"]) == ("failed", "timeout")
    assert out["cleanup"]["parked"] and not out["cleanup"]["caught"]
    assert float(np.max(np.abs(_arm_q(ex) - sx.REST_Q))) < 0.15
    d = TargetLoop([tg.target_id]).report(tg.target_id, out["outcome"], out["reason"], committed=out["committed"])
    assert d["decision"] == "halt_tree"                 # 1 of 1 skipped (> 50 %): the tree halts
    assert d["target_id"] == tg.target_id


def test_person_during_the_park_freezes_the_arm_with_the_fruit(world):
    """After the commit the park is atomic, but a person in the WORK field still freezes it (the fruit stays in the
    catcher, trajectory time stops) and it resumes when the field clears."""
    ex, runner, plans, stations = world()
    tg = _first_target(ex, runner, plans, stations, "OT-1", "mango")
    win = {"t0": None}
    seen = []

    def person():
        if win["t0"] is None and ex._cur_state == "park":
            win["t0"] = ex.drv.t
        on = win["t0"] is not None and win["t0"] + 0.3 <= ex.drv.t <= win["t0"] + 1.5
        if on:
            seen.append(_arm_q(ex))
        return on
    ex.person_fn = person
    out = ex.run(runner._sm(tg))
    print(f"\n[person park] outcome {out['outcome']} paused {ex.paused_s:.2f} s ticks {len(seen)}")
    assert ex.paused_s >= 1.0 and len(seen) >= 50
    settle = seen[15:]
    assert max(float(np.max(np.abs(q - settle[0]))) for q in settle) < 0.05
    assert "person_hold" not in [h[2] for h in out["history"]]        # the SM block stays atomic
    assert out["outcome"] == "picked"
