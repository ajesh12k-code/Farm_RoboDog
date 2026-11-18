"""WP34 checkpoints (WORKFLOWS-SPEC A.4): the record, the ``workflow_checkpoint`` sim-state provider round trip
through ``set_state`` / reset, and resume from a checkpoint (version check, return_to_checkpoint, drift gate,
re-planned target set, node path)."""
import copy

import pytest

from atharv_nandi.core.workflow import checkpoint as C

CP = {"schema": C.SCHEMA, "workflow": "t@1#abcd1234", "run_id": "r1", "node_path": [3], "foreach": {},
      "completed_targets": ["a"], "last_safe": [1.0, 2.0, 90.0], "wf": {"note": 1}, "params": {}, "reason": "rth_wind",
      "supervisor": "rth", "clock": {"active": 3.0, "total": 9.0}, "seq": 2,
      "posture": {"mast": None, "work": True, "tool": "twist"}, "paused_at": 8.0}


def test_record_validation():
    cp = C.Checkpoint.from_dict(CP)
    assert cp.to_dict() == CP
    for bad in ({**CP, "schema": "x"}, {**CP, "extra": 1}, {**CP, "workflow": ""}, {**CP, "workflow": "t1"},
                {**CP, "node_path": ["a"]}, {**CP, "last_safe": [1]}, {**CP, "wf": {"x": float("nan")}}, 5):
        with pytest.raises(C.CheckpointError):
            C.Checkpoint.from_dict(bad)
    with pytest.raises(C.CheckpointError):
        C.Checkpoint(**{**CP, "wf": {"x": object()}}).to_dict()


def test_store_and_provider_pack_unpack():
    C.clear()
    assert C.pack() == {} and C.latest() is None
    C.save(C.Checkpoint.from_dict(CP))
    C.save(C.Checkpoint.from_dict({**CP, "run_id": "r2"}))
    assert C.latest().run_id == "r2" and C.latest("r1").seq == 2
    data = C.pack()
    C.reset_hook()
    assert C.STORE == {}
    C.unpack(None, data)
    assert set(C.STORE) == {"r1", "r2"}
    before = copy.deepcopy(C.STORE)
    for bad in (5, {"runs": 1}, {"other": {}}, {"runs": {"r9": {**CP, "run_id": "r1"}}}, {"runs": {"r1": {}}}):
        with pytest.raises(C.CheckpointError):
            C.unpack(None, bad)
        assert C.STORE == before                                    # nothing written on a bad block
    C.unpack(None, {})
    assert C.STORE == {}
    C.clear("nope")
    C.save(C.Checkpoint.from_dict(CP))
    C.clear("r1")
    assert C.STORE == {}


def test_round_trip_through_sim_state_and_world_reset():
    mujoco = pytest.importorskip("mujoco")
    from atharv_nandi.core import sim_state
    from atharv_nandi.sim import hooks
    keys = sim_state.discover_providers()
    assert C.PROVIDER_KEY in keys                                   # registered by atharv_nandi/sim/hooks/workflow.py at import
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='.1'/></body>"
                                       "</worldbody></mujoco>")
    d = mujoco.MjData(m)
    C.clear()
    st = sim_state.decode(sim_state.encode(sim_state.pack(m, d, "w")))
    assert st["mutations"]["providers"][C.PROVIDER_KEY] == {"v": 1, "data": {}}
    C.save(C.Checkpoint.from_dict(CP))
    st = sim_state.decode(sim_state.encode(sim_state.pack(m, d, "w")))
    expect = copy.deepcopy(C.STORE)
    reg = hooks.HookRegistry()
    import atharv_nandi.sim.hooks.workflow as hw
    hw.register(reg)
    reg.reset(m, d)                                                 # world reset clears the store
    assert C.STORE == {}
    assert sim_state.apply(m, d, st, "w") == (True, "ok")           # set_state restores it
    assert C.STORE == expect
    bad = copy.deepcopy(st)
    bad["mutations"]["providers"][C.PROVIDER_KEY]["data"] = {"runs": {"x": {"schema": "no"}}}
    ok, why = sim_state.apply(m, d, bad, "w")
    assert not ok and C.STORE == expect


# ---------------------------------------------------------------------------------------------- resume
from atharv_nandi.core.workflow import compiler  # noqa: E402
from atharv_nandi.core.workflow.primitives import registry as reg_, stubs  # noqa: E402

try:
    import py_trees  # noqa: F401
    from atharv_nandi.core.workflow import engine as E
except ImportError:                                                     # the record tests above still run
    E = None
needs_py_trees = pytest.mark.skipif(E is None, reason="py_trees not installed (requirements-workflow.txt)")


def A(do, **kw):
    body = {"do": do}
    body.update(kw)
    if reg_.PRIMITIVES[do].running and "timeout_s" not in body:
        body["timeout_s"] = reg_.PRIMITIVES[do].default_timeout_s
    return {"action": body}


SURVEY = {"schema": "nandi.workflow/1", "id": "survey", "version": 1, "kind": "preset", "primitives_api": 1,
          "interlocks": ["common"],
          "root": {"sequence": [A("start_ack"), A("move_to", args={"target": "row_start"}),
                                {"foreach": {"set": "trees", "body": {"sequence": [
                                    A("move_to", args={"target": "tree", "v_max": 0.15}), A("capture")]}}},
                                A("capture"), A("stow_mast")]},
          "on_pause": {"ref": "finaliser_default@1"}, "on_abort": {"ref": "finaliser_default@1"}}


def lib_with(doc):
    lib = compiler.Library()
    entry = lib.add(doc, area="presets")
    return lib, entry


def run_until_rth(lib, entry, after_targets=2, **world_kw):
    comp = compiler.compile_doc(entry.doc(), lib, entry.data())
    w = stubs.StubWorld(targets={"trees": ["a", "b", "c", "d"]}, **world_kw)
    eng = E.Engine(comp, run_id="cp-run", world=w, library=lib)

    def maybe_rth(e):
        if len(e.events_of("target_done")) >= after_targets and not e.events_of("request"):
            e.rth("rth_battery")
    w.on_start["capture"] = maybe_rth
    w.on_start["move_to"] = lambda e: w.pose.__setitem__(0, w.pose[0] + 1.0)
    assert eng.run() == "rth"
    cp = C.latest("cp-run")
    eng.close()
    return cp, w


@pytest.fixture
def _stubs():
    stubs.bind_all()
    C.clear()
    yield
    C.clear()


@needs_py_trees
def test_rth_checkpoint_then_resume_in_a_fresh_engine(_stubs):
    lib, entry = lib_with(SURVEY)
    cp, w0 = run_until_rth(lib, entry)
    assert cp.workflow == entry.ident and cp.supervisor == "rth" and cp.reason == "rth_battery"
    assert cp.completed_targets == ["a", "b"] and cp.node_path == [2]
    assert cp.last_safe is not None and cp.clock["active"] > 0
    w = stubs.StubWorld(targets={"trees": ["d", "c", "b", "a", "e"]})        # re-plan from the new pose
    w.pose = [-50.0, 3.0, 10.0]                                              # at dock staging
    eng = E.Engine.from_checkpoint(C.Checkpoint.from_dict(cp.to_dict()), lib, world=w)
    assert eng.state == "paused"
    eng.resume()
    assert eng.run() == "succeeded", (eng.reason, eng.events[-6:])
    assert [c for c in w.calls if c[2] == "return_to_checkpoint" and c[0] == "success"]
    done = [e["target"] for e in eng.events_of("target_done")]
    assert done == ["d", "c", "e"]                                           # a, b not redone
    moves = [c for c in w.calls if c[0] == "start" and c[2] == "move_to"]
    assert all(c[1] != "/root/sequence/1/action" for c in moves)            # node path: row_start not re-run
    assert not w.violations
    eng.close()


@needs_py_trees
def test_resume_refused_when_the_workflow_changed(_stubs):
    lib, entry = lib_with(SURVEY)
    cp, _ = run_until_rth(lib, entry)
    lib.add(dict(SURVEY, version=2), area="presets")                          # bumped during the charge
    with pytest.raises(E.ResumeRefused) as ei:
        E.Engine.from_checkpoint(cp, lib)
    assert ei.value.reason == "workflow_changed"
    lib2, _ = lib_with(dict(SURVEY, interlocks=["common", "arm_park"]))      # same version, other bytes
    with pytest.raises(E.ResumeRefused):
        E.Engine.from_checkpoint(cp, lib2)
    with pytest.raises(E.ResumeRefused):
        E.Engine.from_checkpoint(C.Checkpoint.from_dict({**cp.to_dict(), "workflow": "ghost@1#00000000"}), lib)


@needs_py_trees
def test_drift_gate_and_ready_check_refuse_resume(_stubs):
    lib, entry = lib_with(SURVEY)
    cp, _ = run_until_rth(lib, entry)
    eng = E.Engine.from_checkpoint(cp, lib, world=stubs.StubWorld(drift_m=0.5))
    eng.resume()
    assert eng.run() == "paused" and eng.reason == "drift_gate"
    assert eng.events_of("resume_refused")
    eng.close()
    eng = E.Engine.from_checkpoint(cp, lib, world=stubs.StubWorld(signals={"ready_common": False}))
    eng.resume()
    assert eng.run() == "paused" and eng.reason == "ready_check"
    assert not [e for e in eng.events if e["kind"] == "resumed"]
    eng.close()


@needs_py_trees
def test_checkpoint_after_every_target_and_on_pause(_stubs):
    lib, entry = lib_with(SURVEY)
    comp = compiler.compile_doc(entry.doc(), lib, entry.data())
    w = stubs.StubWorld(targets={"trees": ["a", "b", "c"]})
    eng = E.Engine(comp, run_id="cp-every", world=w, library=lib)
    assert eng.run() == "succeeded"
    cps = eng.events_of("checkpoint")
    assert [c["supervisor"] for c in cps] == ["running"] * 3
    assert C.latest("cp-every").foreach["/root/sequence/2/foreach"]["completed"] == ["a", "b", "c"]
    eng.close()
    eng = E.Engine(comp, run_id="cp-pause", world=stubs.StubWorld(targets={"trees": ["a", "b"]}), library=lib)
    eng.start()
    eng.ack()
    for _ in range(4):
        eng.tick()
    eng.pause()
    eng.run()
    eng.bb.set("progress", 0.5)
    cp = eng.write_checkpoint("operator_pause", "pause")
    assert cp.wf == {"progress": 0.5} or cp.wf.get("progress") == 0.5
    assert C.latest("cp-pause").reason == "operator_pause"
    eng.close()


def test_corrupt_fields_are_refused_and_nothing_is_written():
    fe = "/root/sequence/2/foreach"
    bads = [{"foreach": {fe: {"completed": "f1", "skipped": {}, "total": 1}}},
            {"foreach": {fe: {"completed": [], "skipped": [], "total": 1}}},
            {"foreach": {fe: {"completed": [], "skipped": {}, "total": "x"}}},
            {"foreach": {fe: {"completed": [], "skipped": {}, "total": 1, "bogus": 1}}},
            {"foreach": []}, {"clock": {"active": "1"}}, {"clock": {"active": -1.0}}, {"clock": [1]},
            {"params": [1]}, {"wf": "x"}, {"wf": {"../other/x": 1}}, {"completed_targets": "ab"},
            {"seq": "2"}, {"reason": 5}, {"node_path": [-1]}, {"node_path": [True]}, {"last_safe": ["a", 1, 2]},
            {"posture": {"mast": "up"}}, {"posture": {"work": "yes"}}, {"posture": {"extra": 1}},
            {"paused_at": "later"}]
    C.clear()
    C.save(C.Checkpoint.from_dict(CP))
    before = copy.deepcopy(C.STORE)
    for patch in bads:
        with pytest.raises(C.CheckpointError):
            C.Checkpoint.from_dict({**CP, **patch})
        with pytest.raises(C.CheckpointError):
            C.unpack(None, {"runs": {"r1": {**CP, **patch}}})
        assert C.STORE == before, patch
    C.clear()


@needs_py_trees
def test_resume_from_a_checkpoint_needs_the_long_pause_ack(_stubs):
    lib, entry = lib_with(SURVEY)
    cp, _ = run_until_rth(lib, entry)
    assert cp.paused_at is not None and cp.paused_at <= cp.clock["total"]
    eng = E.Engine.from_checkpoint(cp, lib)                       # pause length unknown: counts as long
    eng.resume()
    assert eng._waiting_ack and eng.events_of("resume_requested")[0]["needs_ack"]
    for _ in range(20):
        eng.tick()
    assert eng.state == "resuming" and not [c for c in eng.world.calls if c[0] == "start"]
    eng.ack()
    assert eng.run() == "succeeded"
    eng.close()
    eng = E.Engine.from_checkpoint(cp, lib, elapsed_s=3 * 3600.0)  # RTH, dock, charge
    eng.resume()
    assert eng._waiting_ack
    eng.close()
    eng = E.Engine.from_checkpoint(cp, lib, elapsed_s=60.0)
    eng.resume()
    assert not eng._waiting_ack
    eng.close()
    eng = E.Engine.from_checkpoint(cp, lib, elapsed_s=3 * 3600.0, auto_continue=True)
    eng.resume()
    assert not eng._waiting_ack
    eng.close()


@needs_py_trees
def test_checkpoint_that_does_not_fit_the_workflow_is_refused(_stubs):
    lib, entry = lib_with(SURVEY)
    cp, _ = run_until_rth(lib, entry)
    for patch in ({"params": {"ghost": 1}}, {"wf": {"bad key!": 1}}):
        bad = C.Checkpoint(**{**cp.to_dict(), **patch})
        with pytest.raises(E.ResumeRefused) as ei:
            E.Engine.from_checkpoint(bad, lib)
        assert ei.value.reason == "not_resumable"
    bad = C.Checkpoint(**cp.to_dict())
    bad.foreach = {"/root/sequence/2/foreach": {"completed": "a", "total": "x"}}  # mutated after parsing
    with pytest.raises(E.ResumeRefused) as ei:
        E.Engine.from_checkpoint(bad, lib)
    assert ei.value.reason == "not_resumable"


@needs_py_trees
def test_checkpoint_restore_brings_the_mast_back(_stubs):
    doc = dict(SURVEY, id="survey_up", preconditions=["crown_gate"])
    doc["root"] = {"sequence": [A("start_ack"), A("raise_mast", args={"band": "high"}),
                                {"foreach": {"set": "trees", "body": {"sequence": [A("capture")]}}},
                                A("stow_mast")]}
    lib, entry = lib_with(doc)
    cp, _ = run_until_rth(lib, entry)
    assert cp.posture["mast"] == {"do": "raise_mast", "args": {"band": "high"}}
    w = stubs.StubWorld(targets={"trees": ["a", "b", "c", "d"]})
    eng = E.Engine.from_checkpoint(cp, lib, world=w, elapsed_s=30.0)
    eng.resume()
    seen, n = [], 0
    for _ in range(400):
        eng.tick()
        seen += [w.mast for c in w.calls[n:] if c[0] == "start" and c[2] == "capture"]
        n = len(w.calls)
        if eng.done:
            break
    assert eng.state == "succeeded" and seen and set(seen) == {"up"}
    eng.close()
