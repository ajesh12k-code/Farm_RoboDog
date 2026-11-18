"""WP34 engine: py_trees 2.2.3 runtime, L0/L1 rules, operator commands, active time, tick cost.

Acceptance rows covered here: a workflow with no person guard still pauses (L0/L1); pause / resume / abort with
active-time timeouts; pause between ``twist`` success and ``place`` -> place completes, then paused;
<= 2 ms per tick at 10 Hz for a 100-node tree with the debug streams off (measured in the capped runner; the
CI-runner number is taken by the CI run). Skips cleanly without py_trees.
"""
import statistics
import time

import pytest

py_trees = pytest.importorskip("py_trees")

from atharv_nandi.core.workflow import blackboard, compiler, engine as E, reasons  # noqa: E402
from atharv_nandi.core.workflow.primitives import registry as reg, stubs  # noqa: E402


def A(do, **kw):
    body = {"do": do}
    body.update(kw)
    if reg.PRIMITIVES[do].running and "timeout_s" not in body:
        body["timeout_s"] = reg.PRIMITIVES[do].default_timeout_s
    return {"action": body}


def doc(steps, **extra):
    d = {"schema": "nandi.workflow/1", "id": "t", "version": 1, "kind": "preset", "primitives_api": 1,
         "interlocks": ["common"], "root": {"sequence": [A("start_ack")] + list(steps)},
         "on_pause": {"ref": "finaliser_default@1"}, "on_abort": {"ref": "finaliser_default@1"}}
    d.update(extra)
    return d


HARVEST_BODY = [A("enter_work"),
                {"atomic": [A("twist", args={"rule": "stem_axis"}), A("hold_verify"), A("place")]},
                A("arm_park"), A("exit_work")]
HARVEST = doc([A("move_to", args={"target": "tree", "v_max": 0.15}),
               {"foreach": {"set": "fruit", "labels": {"en": "fruit", "hi": "फल"},
                            "body": {"sequence": HARVEST_BODY}}},
               A("stow_mast")], console={"en": "HARVEST", "hi": "तुड़ाई"})


_RUN = [0]


def make(d, **world_kw):
    comp = compiler.compile_doc(d)
    world_kw.setdefault("targets", {"fruit": ["f1", "f2", "f3"]})
    w = stubs.StubWorld(**world_kw)
    _RUN[0] += 1
    eng = E.Engine(comp, run_id=f"eng{_RUN[0]}", world=w, checkpoint_store=False)
    return eng, w


def starts(w, name):
    return [c for c in w.calls if c[0] == "start" and c[2] == name]


def kinds(eng):
    return [e["kind"] for e in eng.events]


@pytest.fixture(autouse=True)
def _stubs():
    stubs.bind_all()
    yield


def test_harvest_runs_to_success_on_stubs():
    eng, w = make(HARVEST)
    assert eng.run() == "succeeded"
    assert w.placed == 3 and not w.holding and not w.violations
    assert [e["target"] for e in eng.events_of("target_done")] == ["f1", "f2", "f3"]
    assert eng.events[-1]["on_complete"] == "rth"
    assert eng.done
    eng.close()


def test_motion_waits_for_start_ack_and_local_link():
    eng, w = make(HARVEST)
    eng.start()
    for _ in range(30):
        eng.tick()
    assert eng.state == "running" and not starts(w, "move_to")            # no ack yet: no motion
    w.fix_signal("local_link", False)
    eng.ack("handheld")
    for _ in range(10):
        eng.tick()
    assert not starts(w, "move_to")                                         # ack but local link down
    w.fix_signal("local_link", True)
    assert eng.run() == "succeeded"
    eng.close()
    eng, w = make(HARVEST, script={"start_ack": "hang"})                    # no ack within 600 s active
    eng.start()
    for _ in range(6200):
        eng.tick()
        if eng.done:
            break
    assert eng.state == "failed" and eng.reason == "timeout" and not starts(w, "move_to")
    eng.close()


def test_pause_between_twist_success_and_place_completes_place_first():
    def on_hold_verify(engine):
        if not engine.events_of("request"):
            engine.pause()                                # the twist has already released the fruit
    eng, w = make(HARVEST, on_start={"hold_verify": on_hold_verify}, targets={"fruit": ["f1", "f2"]})
    assert eng.run() == "paused"
    req = eng.events_of("request")[0]
    assert req["queued"] is True and req["request"] == "pause"
    order = [(c[0], c[2]) for c in w.calls]
    i_place = order.index(("success", "place"))
    i_twist = order.index(("success", "twist"))
    fin = kinds(eng).index("finaliser_start")
    assert i_twist < i_place
    assert w.placed == 1 and not w.holding                                 # place completed, then paused
    assert eng.events[fin]["which"] == "pause"
    assert ("cancel", "place") not in order
    eng.resume()
    assert eng.run() == "succeeded" and w.placed == 2 and not w.violations
    eng.close()


def test_rth_inside_atomic_and_estop_inside_atomic():
    eng, w = make(HARVEST, on_start={"hold_verify": lambda e: e.rth("rth_battery")})
    assert eng.run() == "rth" and eng.reason == "rth_battery"
    assert w.placed == 1 and not w.holding
    eng.close()

    def estop(engine):
        engine.world.fix_signal("estop_clear", False)
    eng, w = make(HARVEST, on_start={"hold_verify": estop})
    assert eng.run() == "estopped"
    assert eng.outcomes == [{"node": eng.outcomes[0]["node"], "outcome": "dropped"}]
    assert w.dropped == 1 and w.placed == 0
    assert not eng.events_of("finaliser_start")                              # L0 owns the item now
    eng.close()
    eng, w = make(HARVEST, on_start={"hold_verify": estop}, estop_drops=False)
    assert eng.run() == "estopped" and eng.outcomes[0]["outcome"] == "held_at_estop"
    eng.close()


def test_person_pause_without_any_guard():
    """A.0 rule 1 mutation: no YAML guard anywhere; a person in the warning field still pauses the run."""
    assert not [n for n in compiler.compile_doc(HARVEST).root.walk() if n.type == "guard"]
    eng, w = make(HARVEST)
    eng.start()
    eng.ack()
    for _ in range(12):
        eng.tick()
    w.fix_signal("person_warning_clear", False)
    for _ in range(60):
        eng.tick()
    # on_pause itself is held while the person is in the field: no arm or mast motion (A.0 rule 1)
    assert eng.state == "pausing" and eng.reason == "person_hold" and eng.events_of("finaliser_person_hold")
    n = len(w.calls)
    for _ in range(30):
        eng.tick()
    assert not [c for c in w.calls[n:] if c[0] == "start"]
    w.fix_signal("person_warning_clear", True)
    for _ in range(60):
        eng.tick()
        if eng.state == "paused":
            break
    assert eng.state == "paused" and eng.reason == "person_hold"
    assert eng.last_interrupt["source"] == "L1"
    assert [e["which"] for e in eng.events_of("finaliser_end")] == ["pause"]
    eng.resume()
    assert eng.run() == "succeeded" and not w.violations
    eng.close()


def test_person_pause_inside_atomic_waits_for_the_block():
    def person(engine):
        engine.world.fix_signal("person_warning_clear", False)
    eng, w = make(HARVEST, on_start={"twist": person}, targets={"fruit": ["f1"]})
    assert eng.run(max_ticks=200) == "pausing" and eng.reason == "person_hold"   # on_pause held: person present
    assert w.placed == 1 and not w.holding
    w.fix_signal("person_warning_clear", True)
    assert eng.run() == "paused" and eng.reason == "person_hold"


def test_protective_stop_and_local_link_loss_freeze_then_pause():
    for sig, reason in (("protective_clear", "protective_stop"), ("local_link", "local_link_lost")):
        eng, w = make(HARVEST)
        eng.start()
        eng.ack()
        for _ in range(8):
            eng.tick()
        w.fix_signal(sig, False)
        eng.tick()
        assert eng.state == "l0_hold" and eng.l0_reason == reason
        active = eng.clock.active
        for _ in range(50):
            eng.tick()
        assert eng.clock.active == active                                    # frozen: no active time passes
        w.fix_signal(sig, True)
        assert eng.run() == "paused" and eng.reason == reason
        eng.resume()
        assert eng.run() == "succeeded" and not w.violations
        eng.close()


def test_protective_stop_inside_atomic_breaks_it_and_skips_the_target():
    def stop(engine):
        if not engine.outcomes:
            engine.world.fix_signal("protective_clear", False)
    eng, w = make(HARVEST, on_start={"hold_verify": stop}, targets={"fruit": ["f1", "f2"]})
    eng.start()
    eng.ack()
    for _ in range(200):
        eng.tick()
        if eng.state == "l0_hold":
            break
    assert eng.state == "l0_hold" and eng.outcomes[0]["outcome"] == "held_at_estop"
    w.fix_signal("protective_clear", True)
    assert eng.run() == "paused"
    assert not w.holding                                                      # release_safe placed it
    eng.resume()
    assert eng.run() == "succeeded"
    skipped = eng.events_of("target_skipped")
    assert skipped and skipped[0]["reason"] == "held_at_estop"               # nothing resumes mid-atomic
    eng.close()


def test_abort_runs_on_abort_and_waits_for_atomic():
    eng, w = make(HARVEST, on_start={"hold_verify": lambda e: e.abort()})
    assert eng.run() == "aborted" and eng.reason == "operator_abort"
    assert w.placed == 1 and not w.holding
    assert [e["which"] for e in eng.events_of("finaliser_end")] == ["abort"]
    assert w.arm == "parked" and w.mast == "stowed"
    eng.close()


def test_active_time_timeouts_ignore_pauses():
    # continue policy: the stopwatch keeps the active time from before the pause
    d = doc([A("settle", args={"s": 9}, timeout_s=1.0, resumable="continue")])
    eng, w = make(d, script={"settle": "hang"})
    eng.start()
    eng.ack()
    for _ in range(6):
        eng.tick()
    eng.pause()
    while eng.state != "paused":
        eng.tick()
    for _ in range(3000):                                                    # 300 s paused
        eng.tick()
    assert not eng.events_of("timeout")
    eng.resume()
    eng.ack()
    n = 0
    while not eng.events_of("timeout"):
        eng.tick()
        n += 1
        assert n < 100
    to = eng.events_of("timeout")[0]
    assert to["after_s"] == pytest.approx(1.0, abs=0.11)                    # active time, not wall / sim
    assert eng.clock.total - eng.clock.active > 299
    eng.run()
    assert eng.state == "failed" and eng.reason == "timeout"
    eng.close()
    # restart policy: the step restarts its timer on re-entry, so a long pause never times it out at once
    d = doc([A("settle", args={"s": 9}, timeout_s=1.0)])
    eng, w = make(d, durations={"settle": 8})
    eng.start()
    eng.ack()
    for _ in range(7):
        eng.tick()
    eng.pause()
    eng.run(auto_ack=False)
    for _ in range(10000):                                                   # > 15 min paused
        eng.tick()
    eng.resume()
    assert eng.status()["state"] == "resuming"
    for _ in range(5):
        eng.tick()
    assert eng.state == "resuming"                                           # long pause: needs an ack
    assert eng.run() == "succeeded" and not eng.events_of("timeout")
    assert [c[0] for c in w.calls if c[2] == "settle"].count("start") == 2
    eng.close()


def test_resume_before_start_ack_and_bad_commands():
    eng, w = make(HARVEST)
    with pytest.raises(RuntimeError):
        eng.resume()
    eng.start()
    with pytest.raises(RuntimeError):
        eng.start()
    eng.pause()
    eng.tick()
    while eng.state != "paused":
        eng.tick()
    eng.resume()
    assert eng.run() == "succeeded" and not w.violations
    with pytest.raises(ValueError):
        eng._request("teleport", "rtk", "x")
    with pytest.raises(ValueError):
        eng._request("pause", "bored", "x")
    eng.close()


def test_retry_vary_backoff_and_reason_on_fail():
    d = doc([A("move_to", args={"target": "x", "v_max": 0.15}, retry={"n": 2, "backoff_s": 0.3,
                                                                       "vary": {"v_max": [-0.05]}})])
    eng, w = make(d, script={"move_to": ["fail", "fail", "success"]})
    assert eng.run() == "succeeded"
    assert [e["attempt"] for e in eng.events_of("retry")] == [1, 2]
    assert len(starts(w, "move_to")) == 3
    eng.close()
    d = doc([A("move_to", args={"target": "x"}, retry={"n": 1}, reason_on_fail="geofence")])
    eng, w = make(d, script={"move_to": "fail"})
    assert eng.run() == "failed" and eng.reason == "geofence"
    assert [e["which"] for e in eng.events_of("finaliser_end")] == ["abort"]
    eng.close()


def test_recover_rounds_skip_and_skip_ratio_rth():
    body = {"recover": {"body": A("capture"), "recovery": [A("backup", args={"d": 0.2})], "max_rounds": 2,
                        "then": "skip(orbit_blocked)"}}
    d = doc([{"foreach": {"set": "trees", "skip_ratio_halt": 0.5, "body": body}}])
    eng, w = make(d, targets={"trees": ["a", "b", "c"]}, script={"capture": ["fail", "success", "fail", "fail",
                                                                                 "fail", "fail"]})
    assert eng.run() == "rth" and eng.reason == "rth_skip_ratio"
    assert [e["target"] for e in eng.events_of("target_done")] == ["a"]
    assert [(e["target"], e["reason"]) for e in eng.events_of("target_skipped")] == [("b", "orbit_blocked"),
                                                                                    ("c", "orbit_blocked")]
    assert len(starts(w, "backup")) == 3
    eng.close()
    cont = doc([{"recover": {"body": A("capture"), "max_rounds": 2, "then": "continue"}}, A("stow_mast")])
    eng, w = make(cont, script={"capture": "fail"})
    assert eng.run() == "succeeded" and len(starts(w, "capture")) == 2
    eng.close()
    fail = doc([{"recover": {"body": A("capture"), "max_rounds": 1, "then": "fail"}}])
    eng, w = make(fail, script={"capture": "fail"})
    assert eng.run() == "failed"
    eng.close()


def test_target_loop_template_assist_replies():
    tpl = {"template": {"id": "target_loop", "set": "trees", "rounds": 2, "reason": "out_of_reach_marked",
                        "body": A("capture")}}
    for reply, state, skips in (("skip", "rth", 2), ("take_over", "succeeded", 0), ("halt", "halted", 0)):
        eng, w = make(doc([tpl]), targets={"trees": ["a", "b", "c"]}, script={"capture": "fail"},
                      assist_reply=reply)
        assert eng.run() == state, reply
        assert len(eng.events_of("target_skipped")) == skips
        if reply == "halt":
            assert eng.reason == "operator_halt"
        eng.close()
    eng, w = make(doc([tpl]), targets={"trees": ["a", "b", "c"]}, script={"capture": ["fail", "fail"]},
                  durations={"assist_request": 4000})
    assert eng.run() == "succeeded"                                          # assist timed out -> skip(reason)
    assert eng.events_of("target_skipped")[0]["reason"] == "out_of_reach_marked"
    eng.close()


def test_operator_skip_target():
    d = doc([{"foreach": {"set": "trees", "body": A("capture", timeout_s=60)}}])
    eng, w = make(d, targets={"trees": ["a", "b"]}, durations={"capture": 20},
                  on_start={"capture": lambda e: e.skip_target() if not e.events_of("target_skipped") else None})
    assert eng.run() == "succeeded"
    assert [e["target"] for e in eng.events_of("target_skipped")] == ["a"]
    assert ("cancel", "/root/sequence/1/foreach/body/action", "capture") in w.calls
    eng.close()


@pytest.mark.parametrize("how,state,reason", [
    ("pause", "paused", "wind"), ("rth", "rth", "wind"), ("safe_park", "paused", "wind"),
    ("halt_latch", "halted", "wind"), ("skip_target", "succeeded", "wind"), ("stow_continue", "succeeded", None)])
def test_guard_violations(how, state, reason):
    child = {"foreach": {"set": "trees", "body": {"sequence": [A("capture", timeout_s=60), A("capture")]}}}
    g = {"guard": {"cond": "wind_lt_crown", "on_violation": how, "child": child}}
    if how == "skip_target":
        g = {"foreach": {"set": "trees", "body": {"guard": {"cond": "wind_lt_crown", "on_violation": how,
                                                            "child": A("capture", timeout_s=60)}}}}
    flips = []

    def gust(engine):
        if not flips:
            flips.append(1)
            engine.world.fix_signal("wind_lt_crown", False)
    eng, w = make(doc([g]), targets={"trees": ["a", "b"]}, on_start={"capture": gust})
    got = eng.run()
    if how == "stow_continue":
        assert got == "succeeded" and starts(w, "stow_mast")
    elif how == "skip_target":
        assert got == "succeeded" and eng.events_of("target_skipped")[0]["reason"] == reason
    else:
        assert got == state and eng.reason == reason
    assert eng.events_of("guard_violation")
    eng.close()


def test_checks_wait_for_selector_parallel_and_assist_else():
    d = doc([{"check": {"cond": "rain_dry", "else": "fail", "reason": "rain"}}])
    eng, w = make(d, signals={"rain_dry": False})
    assert eng.run() == "failed" and eng.reason == "rain"
    eng.close()
    d = doc([{"wait_for": {"cond": "rain_dry", "timeout_s": 1.0}}])
    eng, w = make(d, signals={"rain_dry": False})
    assert eng.run() == "failed" and eng.reason == "timeout"
    eng.close()
    d = doc([{"foreach": {"set": "t", "body": {"check": {"cond": "tag_seen", "else": "assist"}}}}])
    eng, w = make(d, targets={"t": ["a"]}, signals={"tag_seen": False}, assist_reply="skip")
    assert eng.run() == "succeeded" and eng.events_of("target_skipped")
    eng.close()
    d = doc([{"selector": {"memory": True, "children": [A("capture"), A("lidar_profile")]}},
             {"parallel": {"policy": "all", "children": [A("capture"), A("detect")]}},
             {"parallel": {"policy": "one", "children": [A("capture"), A("follow_tag")]}}])
    eng, w = make(d, script={"capture": ["fail"]}, durations={"follow_tag": 50})
    assert eng.run() == "succeeded" and starts(w, "lidar_profile")
    eng.close()


def test_flow_primitives_and_blackboard_set():
    d = doc([A("set", args={"key": "note", "value": 7}), A("request", args={"what": "safe_park"}), A("capture")])
    eng, w = make(d)
    assert eng.run() == "paused" and eng.reason == "safe_park"
    assert eng.bb.get("note") == 7
    eng.close()
    eng, w = make(doc([A("request_rth", args={"reason": "rth_done"}), A("capture")]))
    assert eng.run() == "rth" and eng.reason == "rth_done"
    eng.close()
    eng, w = make(doc([A("halt", args={"reason": "slope"})]))
    assert eng.run() == "halted" and eng.reason == "slope"
    eng.close()
    eng, w = make(doc([A("wait_operator", args={"ack": "power lines checked"}), A("beacon"), A("lower")]))
    assert eng.run() == "succeeded"
    eng.close()


def test_rollback_policy_on_resume():
    d = doc([A("enter_work"), A("arm_approach", args={"pose": "p"}), A("arm_park"), A("exit_work")])
    eng, w = make(d, durations={"arm_approach": 10})
    eng.start()
    eng.ack()
    while not starts(w, "arm_approach"):
        eng.tick()
    eng.pause()
    eng.run()
    eng.resume()
    assert eng.run() == "succeeded"
    assert eng.events_of("rollback")[0]["to"] == "arm_park"
    assert len(starts(w, "arm_approach")) == 2
    eng.close()


def test_params_within_min_max_only_and_status_breadcrumb():
    d = dict(HARVEST, params={"v": {"default": 0.1, "min": 0.05, "max": 0.15}, "m": {"default": "a",
                                                                                     "enum": ["a", "b"]}})
    comp = compiler.compile_doc(d)
    with pytest.raises(ValueError):
        E.Engine(comp, run_id="p1", params={"v": 0.5}, checkpoint_store=False)
    with pytest.raises(ValueError):
        E.Engine(comp, run_id="p2", params={"m": "z"}, checkpoint_store=False)
    with pytest.raises(ValueError):
        E.Engine(comp, run_id="p3", params={"nope": 1}, checkpoint_store=False)
    eng, w = make(HARVEST, durations={"twist": 30})
    eng.start()
    eng.ack()
    while not starts(w, "twist"):
        eng.tick()
    eng.tick()
    st = eng.status()
    assert st["breadcrumb"]["en"] == "HARVEST > fruit 1/3 > twist"
    assert st["breadcrumb"]["hi"].startswith("तुड़ाई > फल 1/3")
    assert st["target"] == {"i": 0, "n": 3} and st["atomic_open"] is True
    seen = []
    eng.on_event = seen.append
    eng.pause()
    assert seen and seen[0]["kind"] == "request"
    eng.close()


def test_hook_ticks_attached_engine_and_maps_holds():
    from atharv_nandi.sim import hooks
    import atharv_nandi.sim.hooks.workflow as hw
    reg_ = hooks.HookRegistry()
    hw.register(reg_)
    bus = hooks.Bus()

    class D:
        time = 0.0
    d = D()
    reg_.tick(None, d, bus)                                                  # nothing attached: no-op
    assert not bus.outbox
    eng, w = make(HARVEST, durations={"twist": 200})
    eng.start()
    eng.ack()
    hw.attach(bus, eng)
    for i in range(1, 31):
        d.time = i * 0.1
        reg_.tick(None, d, bus)
    assert eng.ticks == 30
    pubs = [p for t, p in bus.drain() if t == hw.STATUS_TOPIC]
    assert 5 <= len(pubs) <= 7                                               # <= 2 Hz over 3 s
    bus.values["hold"] = {"reason": "hook_fault", "since": 3.0}
    d.time = 3.1
    reg_.tick(None, d, bus)
    assert eng.state == "l0_hold"
    bus.values["hold"] = {"reason": "estop", "since": 3.2}
    d.time = 3.2
    reg_.tick(None, d, bus)
    assert eng.state == "estopped"
    hw.detach(bus)
    reg_.reset(None, None)
    eng.close()


def _wide_tree(n_nodes):
    """A 100-node tree: guard > parallel of sequences of long-running actions + checks (every leaf ticked)."""
    seqs, count = [], 3
    while count + 4 <= n_nodes:
        seqs.append({"sequence": {"memory": False, "children": [{"check": {"cond": "rain_dry"}}, A("capture", timeout_s=3600),
                                  {"check": {"cond": "stability"}}]}})
        count += 4
    root = {"guard": {"cond": "wind_lt_crown", "on_violation": "pause",
                      "child": {"parallel": {"policy": "all", "children": seqs}}}}
    d = doc([root])
    return d


def test_tick_cost_100_nodes_debug_off():
    d = _wide_tree(100)
    comp = compiler.compile_doc(d)
    assert 95 <= comp.size() <= 100
    w = stubs.StubWorld(script={"capture": "hang"})
    eng = E.Engine(comp, run_id="bench", world=w, checkpoint_store=False)
    assert py_trees.blackboard.Blackboard.activity_stream is None
    eng.start()
    eng.ack()
    for _ in range(20):
        eng.tick()
    assert len([b for b in eng.root.iterate() if b.status == py_trees.common.Status.RUNNING]) >= 50
    times = []
    for _ in range(600):
        t0 = time.perf_counter()
        eng.tick()
        times.append(time.perf_counter() - t0)
    times.sort()
    mean_ms = 1e3 * statistics.mean(times)
    p99_ms = 1e3 * times[int(0.99 * len(times))]
    print(f"\nWP34 tick cost, {comp.size()} nodes, all leaves RUNNING: mean {mean_ms:.3f} ms, "
          f"p50 {1e3 * times[len(times) // 2]:.3f} ms, p99 {p99_ms:.3f} ms, max {1e3 * times[-1]:.3f} ms")
    assert mean_ms <= 2.0
    eng.close()
    # the debug stream is opt-in
    eng2 = E.Engine(comp, run_id="bench-debug", world=stubs.StubWorld(), debug=True, checkpoint_store=False)
    assert py_trees.blackboard.Blackboard.activity_stream is not None
    eng2.close()
    blackboard.set_debug(False)


def test_reason_codes_raised_are_in_the_enum():
    eng, w = make(HARVEST, script={"twist": "fail"})
    eng.run()
    for e in eng.events:
        if "reason" in e and e["reason"] is not None:
            assert e["reason"] in reasons.REASONS, e
    eng.close()


# ---------------------------------------------------------------------------------------------- review round 1
def _pause(eng, ticks=12, reason="operator_pause"):
    eng.start()
    eng.ack()
    for _ in range(ticks):
        eng.tick()
    eng.pause(reason)
    for _ in range(80):
        eng.tick()
        if eng.state == "paused":
            return
    raise AssertionError(eng.state)


def _drive(eng, n=200):
    for _ in range(n):
        if eng._waiting_ack:
            eng.ack("t")
        eng.tick()
        if eng.done or eng.state == "paused":
            break
    return eng.state


def test_abort_and_rth_while_paused_never_drive_back_into_the_orchard():
    for cmd, terminal, which in (("abort", "aborted", "abort"), ("rth", "rth", "pause")):
        eng, w = make(HARVEST)
        _pause(eng)
        n = len(w.calls)
        getattr(eng, cmd)()
        assert _drive(eng) == terminal
        later = [c[2] for c in w.calls[n:] if c[0] == "start"]
        assert "return_to_checkpoint" not in later and "twist" not in later
        assert eng.events_of("finaliser_end")[-1]["which"] == which and not w.violations
        with pytest.raises(RuntimeError):
            eng.resume()
        eng.close()


def test_abort_while_resuming_cancels_the_resume_check():
    eng, w = make(HARVEST, durations={"return_to_checkpoint": 30})
    _pause(eng)
    eng.resume()
    for _ in range(5):
        eng.tick()
    assert eng.state == "resuming" and starts(w, "return_to_checkpoint")
    eng.abort()
    assert _drive(eng) == "aborted"
    assert [c for c in w.calls if c[0] == "cancel" and c[2] == "return_to_checkpoint"
            and c[1].startswith("/resume_check")]
    assert not [c for c in w.calls if c[0] == "success" and c[2] == "return_to_checkpoint"]
    assert eng.events_of("resume_cancelled")[0]["request"] == "abort"


def test_abort_or_rth_while_pausing_take_effect_after_on_pause():
    for cmd, terminal, whiches in (("abort", "aborted", ["pause", "abort"]), ("rth", "rth", ["pause"])):
        eng, w = make(HARVEST, durations={"release_safe": 6})
        eng.start()
        eng.ack()
        for _ in range(12):
            eng.tick()
        eng.pause()
        for _ in range(3):
            eng.tick()
        assert eng.state == "pausing"
        getattr(eng, cmd)()
        assert _drive(eng) == terminal
        assert [e["which"] for e in eng.events_of("finaliser_end")] == whiches
        eng.close()


def test_person_pause_covers_the_resume_check():
    eng, w = make(HARVEST, durations={"return_to_checkpoint": 30})
    _pause(eng)
    eng.resume()
    for _ in range(3):
        eng.tick()
    assert starts(w, "return_to_checkpoint")
    w.fix_signal("person_warning_clear", False)
    eng.tick()
    assert eng.state == "pausing" and eng.reason == "person_hold"
    n = len(w.calls)
    for _ in range(60):
        eng.tick()
    assert not [c for c in w.calls[n:] if c[0] in ("start", "success")]       # nothing moves with a person
    assert not [c for c in w.calls if c[0] == "success" and c[2] == "return_to_checkpoint"]
    w.fix_signal("person_warning_clear", True)
    assert _drive(eng) == "paused" and eng.reason == "person_hold"
    eng.resume()
    assert eng.run() == "succeeded" and not w.violations
    eng.close()


def test_person_present_when_resume_is_requested_never_starts_the_check():
    eng, w = make(HARVEST)
    _pause(eng)
    w.fix_signal("person_warning_clear", False)
    eng.resume()
    for _ in range(20):
        eng.tick()
    assert eng.state == "paused" and eng.reason == "person_hold"
    assert not starts(w, "return_to_checkpoint")
    eng.close()


def test_person_holds_the_abort_finaliser():
    eng, w = make(HARVEST, durations={"release_safe": 6})
    eng.start()
    eng.ack()
    for _ in range(12):
        eng.tick()
    eng.abort()
    eng.tick()
    eng.tick()
    w.fix_signal("person_warning_clear", False)
    eng.tick()
    n = len(w.calls)
    for _ in range(50):
        eng.tick()
    assert eng.state == "aborting" and eng.events_of("finaliser_person_hold")
    assert not [c for c in w.calls[n:] if c[0] in ("start", "success")]
    w.fix_signal("person_warning_clear", True)
    assert _drive(eng) == "aborted" and eng.events_of("finaliser_person_clear")
    eng.close()


def test_protective_stop_during_a_terminal_finaliser_keeps_the_terminal_state():
    def rth_cmd(e):
        e.rth()
    cases = (("abort", lambda e: e.abort(), "aborted", "operator_abort"),
             ("fail", None, "failed", "primitive_failed"),
             ("rth", rth_cmd, "rth", "rth_operator"))
    for name, cmd, terminal, reason in cases:
        kw = {"durations": {"release_safe": 10}}
        if name == "fail":
            kw["script"] = {"exit_work": "fail"}
        eng, w = make(HARVEST, **kw)
        eng.start()
        eng.ack()
        for _ in range(12):
            eng.tick()
        if cmd is not None:
            cmd(eng)
        for _ in range(300):
            eng.tick()
            if eng.state in ("aborting", "finalising", "pausing"):
                break
        eng.tick()
        w.fix_signal("protective_clear", False)
        for _ in range(3):
            eng.tick()
        assert eng.state == "l0_hold", name
        w.fix_signal("protective_clear", True)
        assert _drive(eng, 300) == terminal, (name, eng.state)
        assert eng.reason == reason, name
        eng.close()


def test_estop_while_paused_latches():
    eng, w = make(HARVEST)
    _pause(eng)
    w.fix_signal("estop_clear", False)
    eng.tick()
    assert eng.state == "estopped" and eng.reason == "estop"
    w.fix_signal("estop_clear", True)
    with pytest.raises(RuntimeError):
        eng.resume()
    eng.close()


def test_unbounded_param_values_are_refused_at_run_time():
    from atharv_nandi.core.workflow import validate
    d = doc([A("move_to", args={"target": "tree", "v_max": "$v"})], params={"v": {"default": 0.15}})
    assert "param_unbounded" in {i.code for i in validate.validate_doc(d)}
    comp = compiler.compile_doc(d)
    for bad in (5.0, "fast", True, 0.31):
        with pytest.raises(ValueError):
            E.Engine(comp, run_id="pv", params={"v": bad}, checkpoint_store=False)
    E.Engine(comp, run_id="pv-ok", params={"v": 0.1}, checkpoint_store=False).close()
    bounded = doc([A("move_to", args={"target": "tree", "v_max": "$v"})],
                  params={"v": {"default": 0.15, "min": 0.05, "max": 0.18}})
    assert not {i.code for i in validate.validate_doc(bounded)}
    with pytest.raises(ValueError):
        E.Engine(compiler.compile_doc(bounded), run_id="pv2", params={"v": 0.25}, checkpoint_store=False)


TREE360ISH = doc([A("move_to", args={"target": "tree", "v_max": 0.15}), A("raise_mast", args={"band": "mid"}),
                  {"foreach": {"set": "fruit", "body": {"sequence": [A("capture", args={"cams": ["crown"]})]}}},
                  A("stow_mast")], preconditions=["crown_gate"])


def test_resume_restores_the_mast_before_the_loop_continues():
    eng, w = make(TREE360ISH, targets={"fruit": [f"f{i}" for i in range(8)]}, durations={"capture": 5})
    _pause(eng, ticks=16)
    assert w.mast == "stowed"
    eng.resume()
    seen, n = [], len(w.calls)
    for _ in range(400):
        if eng._waiting_ack:
            eng.ack()
        eng.tick()
        seen += [w.mast for c in w.calls[n:] if c[0] == "start" and c[2] == "capture"]
        n = len(w.calls)
        if eng.done:
            break
    assert eng.state == "succeeded" and seen and set(seen) == {"up"}
    assert len(starts(w, "raise_mast")) == 2
    assert [c for c in w.calls if c[0] == "start" and c[1].startswith("/resume_check")][-1][2] == "raise_mast"
    eng.close()


def test_resume_restores_tool_and_work_zone_entered_before_the_loop():
    d = doc([A("tool_change", args={"tool": "clip"}), A("enter_work"),
             {"foreach": {"set": "fruit", "body": {"sequence": [
                 {"atomic": [A("twist", args={"rule": "stem_axis"}), A("hold_verify"), A("place")]},
                 A("arm_park")]}}}, A("exit_work")])
    eng, w = make(d, targets={"fruit": ["f1", "f2", "f3", "f4"]})
    _pause(eng, ticks=14)
    assert not w.work                                   # tools_safe left the work zone
    eng.resume()
    assert eng.run() == "succeeded" and not w.violations, w.violations
    assert len(starts(w, "enter_work")) == 2
    eng.close()


def test_fault_matrix_catches_a_resume_without_the_posture_restore(monkeypatch):
    d = doc([A("enter_work"), {"foreach": {"set": "fruit", "body": {"sequence": [
        {"atomic": [A("twist", args={"rule": "stem_axis"}), A("hold_verify"), A("place")]}, A("arm_park")]}}},
        A("exit_work")])
    comp = compiler.compile_doc(d)
    node = [n.id for n in comp.root.walk() if n.type == "action" and n.do == "arm_park"][0]
    case = {"node": node, "do": "arm_park", "mode": "interrupt"}
    ok = E.run_fault_case(comp, case, world_kw={"targets": {"fruit": ["f1", "f2"]}})
    assert ok["state"] == "succeeded" and not ok["violations"]
    monkeypatch.setattr(E.Engine, "_restore_steps", lambda self: [])
    bad = E.run_fault_case(comp, case, world_kw={"targets": {"fruit": ["f1", "f2"]}})
    assert any(v.startswith("blade_outside_work") for v in bad["violations"])
