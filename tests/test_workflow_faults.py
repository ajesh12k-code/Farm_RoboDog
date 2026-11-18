"""WP34 fault matrix (WORKFLOWS-SPEC A.6.4), generated from the YAML.

Every action node of each workflow below is forced to {fail, timeout, interrupt} on the stub primitives and
the run is checked for: the recovery branch taken (recover / selector / skip), a reason code from the enum, the
finaliser, the terminal state, and the run-time invariant monitor (no commit outside an atomic, no arm_park while
holding without release_safe, ...). Interrupt = an operator pause injected when the node starts: outside an
atomic the run pauses at once; inside one (``hold_verify`` / ``place`` after ``twist``) the block completes
first. After the pause the run resumes through ``_lib/resume_check@1`` and must finish.
"""
import pytest

py_trees = pytest.importorskip("py_trees")

from atharv_nandi.core.workflow import compiler, engine as E, reasons  # noqa: E402
from atharv_nandi.core.workflow.primitives import stubs  # noqa: E402

HARVEST = """
schema: nandi.workflow/1
id: harvest_guava_twist
version: 1
kind: preset
primitives_api: 1
preconditions: [ready_common, crown_gate]
interlocks: [common, arm_park]
root:
  sequence:
  - action: {do: start_ack, timeout_s: 600}
  - action: {do: move_to, args: {target: tree_entry, v_max: 0.15}, timeout_s: 60}
  - action: {do: tool_change, args: {tool: twist}, timeout_s: 30}
  - foreach:
      set: fruit
      body:
        sequence:
        - action: {do: enter_work, timeout_s: 5}
        - recover:
            body:
              sequence:
              - action: {do: arm_approach, args: {pose: pre_grasp}, timeout_s: 15}
              - action: {do: trap, timeout_s: 8}
            recovery: [{do: arm_park, timeout_s: 10}]
            max_rounds: 2
            then: skip(twist_failed)
        - atomic:
          - action: {do: twist, args: {rule: stem_axis}, timeout_s: 8}
          - action: {do: hold_verify, args: {window_s: 1.0}, timeout_s: 5}
          - action: {do: place, args: {into: cell}, timeout_s: 15}
        - action: {do: arm_park, timeout_s: 10}
        - action: {do: exit_work, timeout_s: 5}
  - action: {do: stow_mast, timeout_s: 9}
on_pause: {ref: finaliser_default@1}
on_abort: {ref: finaliser_default@1}
"""

TREE360 = """
schema: nandi.workflow/1
id: tree360
version: 3
kind: preset
primitives_api: 1
preconditions: [annulus_clear_or_two_sided, disk_ok, crown_gate]
interlocks: [common, crown_walk_cap]
approach_envelope: {min_standoff_m: 0.9, mast_max_under_skirt: stowed, enter_annulus: never}
root:
  sequence:
  - action: {do: start_ack, timeout_s: 600}
  - action: {do: move_to, args: {target: entry_tangent, v_max: 0.15}, timeout_s: 60}
  - action: {do: raise_mast, args: {band: mid}, timeout_s: 12}
  - foreach:
      set: stations
      body:
        recover:
          body: {template: capture_station}
          recovery: [{do: reenter_station, retry: {n: 2}, timeout_s: 30}]
          then: skip(arc_blocked)
  - action: {do: stow_mast, timeout_s: 9}
on_pause: {ref: finaliser_default@1}
on_abort: {ref: finaliser_default@1}
"""

CHORE = """
schema: nandi.workflow/1
id: band_check_lite
version: 1
kind: chore
primitives_api: 1
interlocks: [common]
root:
  sequence:
  - action: {do: start_ack, timeout_s: 600}
  - action: {do: plan_route, args: {pattern: block_sample}, timeout_s: 10}
  - template:
      id: target_loop
      set: trees
      rounds: 2
      reason: coverage_partial
      body:
        sequence:
        - action: {do: move_to, args: {target: alley_view, v_max: 0.15}, timeout_s: 60}
        - action: {do: capture, args: {cams: [ring, belly]}, timeout_s: 20}
        - action: {do: detect, args: {classes: [band_present, torn]}, timeout_s: 10}
        - action: {do: update_record, args: {kind: trunk.band}}
on_pause: {ref: finaliser_default@1}
on_abort: {ref: finaliser_default@1}
"""

WORKFLOWS = {"harvest": HARVEST, "tree360": TREE360, "chore": CHORE}
TARGETS = {"fruit": ["f1", "f2"], "stations": ["s1", "s2", "s3"], "trees": ["t1", "t2", "t3", "t4"]}


def _cases():
    out = []
    for name, text in WORKFLOWS.items():
        comp = compiler.compile_text(text)
        for c in E.fault_cases(comp):
            out.append(pytest.param(name, c, id=f"{name}:{c['do']}@{c['node'].split('/', 2)[-1]}:{c['mode']}"))
    return out


CASES = _cases()


def test_matrix_is_generated_from_the_yaml():
    for name, text in WORKFLOWS.items():
        comp = compiler.compile_text(text)
        actions = [n for n in comp.root.walk() if n.type == "action"]
        assert len(E.fault_cases(comp)) == 3 * len(actions)
    assert len(CASES) >= 60


@pytest.fixture(autouse=True)
def _stubs():
    stubs.bind_all()


@pytest.mark.parametrize("name,case", CASES)
def test_fault_case(name, case):
    comp = compiler.compile_text(WORKFLOWS[name])
    r = E.run_fault_case(comp, case, world_kw={"targets": TARGETS}, max_ticks=20000)
    assert not r["violations"], r["violations"]
    assert r["state"] in reasons.TERMINAL, (r["state"], r["first_stop"])
    assert not r["holding"] or r["released"]
    anc = E.enclosing(comp, case["node"])
    types = [a.type for a in anc]
    in_atomic = "atomic" in types
    started = any(c[0] == "start" and c[1] == case["node"] for c in r["calls"])
    if not started:                                         # e.g. a recovery step never needed
        assert r["state"] == "succeeded" and not r["paused"]
        return
    if case["mode"] == "interrupt":
        assert r["paused"] and r["first_stop"] == "paused"
        assert r["finalisers"][0]["which"] == "pause"
        assert r["resumed_state"] == "succeeded", r["events"][-5:]
        if in_atomic:                                       # the atomic completes before the pause applies
            atomic = [a for a in anc if a.type == "atomic"][-1]
            last = atomic.children[-1].id
            done = [i for i, c in enumerate(r["calls"]) if c[0] == "success" and c[1] == last]
            fin = [i for i, c in enumerate(r["calls"]) if c[2] == "release_safe" and c[0] == "start"]
            assert done and fin and done[0] < fin[0]
            assert not any(c[0] == "cancel" and c[1] in {ch.id for ch in atomic.children} for c in r["calls"])
        return
    from atharv_nandi.core.workflow.primitives import registry as reg
    if case["mode"] == "timeout" and reg.PRIMITIVES[case["do"]].running:
        assert any(e["kind"] == "timeout" and e["node"] == case["node"] for e in r["events"])
    elif case["mode"] == "timeout":                         # a non-RUNNING primitive that hangs is a contract breach
        assert any(e["kind"] == "contract_violation" and e["node"] == case["node"] for e in r["events"])
    reason = r["reason"]
    if r["state"] == "succeeded":
        # the failure was absorbed by a recovery branch: recover / selector / skip inside a foreach
        assert {"recover", "selector", "foreach"} & set(types), types
        absorbed = [e for e in r["events"] if e["kind"] in ("recover_round", "target_skipped", "retry")]
        assert absorbed or "selector" in types
        recs = [a for a in anc if a.type == "recover" and a.children[1].children]
        if recs and case["node"].startswith(recs[-1].children[0].id) and case["mode"] == "fail":
            rec_ids = {n.id for n in recs[-1].children[1].walk()}
            assert any(c[0] == "start" and c[1] in rec_ids for c in r["calls"]), "recovery branch not taken"
        for e in r["events"]:
            if e["kind"] == "target_skipped":
                assert e["reason"] in reasons.REASONS
    elif r["state"] == "rth":
        assert reason == "rth_skip_ratio"
        assert r["finalisers"][-1]["which"] == "pause"
    else:
        assert r["state"] in ("failed", "halted"), r["state"]
        assert reason in reasons.REASONS
        assert r["finalisers"] and r["finalisers"][-1]["which"] == "abort"
    if case["mode"] == "fail" and r["state"] == "failed":
        assert reason in ("primitive_failed",) or reason in reasons.REASONS


def test_commit_point_cases_explicitly():
    comp = compiler.compile_text(HARVEST)
    hv = [n for n in comp.root.walk() if n.do == "hold_verify"][0]
    r = E.run_fault_case(comp, {"node": hv.id, "do": "hold_verify", "mode": "interrupt"},
                         world_kw={"targets": TARGETS})
    order = [(c[0], c[2]) for c in r["calls"]]
    assert order.index(("success", "twist")) < order.index(("success", "place")) < order.index(("start",
                                                                                                "release_safe"))
    assert r["resumed_state"] == "succeeded"
    # e-stop inside the atomic -> dropped, no finaliser
    w = stubs.StubWorld(targets=TARGETS, on_start={hv.id: lambda e: e.world.fix_signal("estop_clear", False)})
    eng = E.Engine(comp, run_id="fx-estop", world=w, checkpoint_store=False)
    assert eng.run() == "estopped" and eng.outcomes[0]["outcome"] == "dropped"
    eng.close()
    # RTH requested inside the atomic -> place completes, then the run goes home
    w = stubs.StubWorld(targets=TARGETS, on_start={hv.id: lambda e: e.rth("rth_wind")})
    eng = E.Engine(comp, run_id="fx-rth", world=w, checkpoint_store=False)
    assert eng.run() == "rth" and w.placed == 1 and not w.holding
    eng.close()
    # release_safe with no reachable place keeps holding and raises the holding assist
    tw = [n for n in comp.root.walk() if n.do == "place"][0]
    r = E.run_fault_case(comp, {"node": tw.id, "do": "place", "mode": "fail"},
                         world_kw={"targets": TARGETS, "place_reachable": False})
    assert r["state"] == "failed" and r["holding"] and r["released"]
    assert any(e["kind"] == "assist" and e["reason"] == "holding_item" for e in r["events"])
    assert not r["violations"]
