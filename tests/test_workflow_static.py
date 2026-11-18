"""WP34 static abstract interpretation (WORKFLOWS-SPEC A.6.3, A.2 commit points, CHORES D.0 envelope).

Each test builds a small workflow and checks the issue codes ``static_check`` proves or refuses. No py_trees.
"""
import pytest

from atharv_nandi.core.workflow import compiler, schema, static_check, validate


def A(do, **kw):
    body = {"do": do}
    body.update(kw)
    from atharv_nandi.core.workflow.primitives import registry as reg
    if reg.PRIMITIVES.get(do) is not None and reg.PRIMITIVES[do].running and "timeout_s" not in body:
        body["timeout_s"] = reg.PRIMITIVES[do].default_timeout_s
    return {"action": body}


def doc(steps, **extra):
    d = {"schema": "nandi.workflow/1", "id": "t", "version": 1, "kind": "preset", "primitives_api": 1,
         "interlocks": ["common"], "preconditions": ["ready_common"],
         "root": {"sequence": [A("start_ack")] + list(steps)},
         "on_pause": {"ref": "finaliser_default@1"}, "on_abort": {"ref": "finaliser_default@1"}}
    d.update(extra)
    return d


def codes(d):
    return {i.code for i in schema.errors(validate.validate_doc(d))}


HARVEST = [A("enter_work"), {"atomic": [A("clip", args={"cut_k": 0.5}), A("hold_verify"), A("place")]},
           A("arm_park"), A("exit_work")]


def test_minimal_harvest_is_clean():
    assert codes(doc(HARVEST)) == set()


def test_speed_caps_follow_the_mast_state():
    gate = {"preconditions": ["ready_common", "crown_gate"]}
    assert codes(doc([A("move_to", args={"target": "x", "v_max": 0.18})])) == set()
    assert "speed_cap" in codes(doc([A("move_to", args={"target": "x", "v_max": 0.2})]))
    up = [A("raise_mast", args={"band": "mid"}), A("move_to", args={"target": "x", "v_max": 0.16})]
    assert "speed_cap" in codes(doc(up, **gate))
    up[1]["action"]["args"]["v_max"] = 0.15
    assert codes(doc(up, **gate)) == set()
    p = doc([A("follow_rows", args={"v_max": "$v"})], params={"v": {"default": 0.1, "min": 0.0, "max": 0.25}})
    assert "speed_cap" in codes(p)                         # a param's max is the worst case


def test_approach_envelope_and_annulus():
    env = {"approach_envelope": {"min_standoff_m": 0.6, "enter_annulus": "never"}}
    mv = lambda so: A("move_to", args={"target": "tree", "standoff_m": so})          # noqa: E731
    assert codes(doc([mv(0.9)], **env)) == set()
    assert "approach_envelope" in codes(doc([mv(0.3)], **env))
    assert "approach_envelope" in codes(doc([mv(-0.2)], **dict(env, approach_envelope={"enter_annulus": "never"})))
    stowed_only = {"approach_envelope": {"enter_annulus": "stowed_only"}}
    assert codes(doc([mv(-0.2)], **stowed_only)) == set()
    gate = ["ready_common", "crown_gate"]
    up = [A("raise_mast", args={"band": "low"}), mv(0.1)]
    got = codes(doc(up, preconditions=gate, approach_envelope={"enter_annulus": "allowed"}))
    assert got == {"annulus_mast_up"}
    up = [A("raise_mast", args={"band": "low"}), mv(-0.1)]
    assert "approach_envelope" in codes(doc(up, preconditions=gate, **stowed_only))
    p = doc([A("move_to", args={"target": "t", "standoff_m": "$so"})], params={"so": {"default": 1.0, "min": 0.1, "max": 5.0}},
            **env)
    assert "approach_envelope" in codes(p)                 # a param's min is the worst case
    assert codes(doc([A("move_to", args={"target": "t", "standoff_m": "$nope_param"})], **env)) == {"unknown_param"}


def test_tool_change_needs_mast_stowed_and_empty_jaws():
    gate = ["ready_common", "crown_gate"]
    assert codes(doc([A("tool_change", args={"tool": "clipper"})])) == set()
    up = [A("raise_mast", args={"band": "mid"}), A("tool_change", args={"tool": "clipper"})]
    assert "tool_change_unsafe" in codes(doc(up, preconditions=gate))


def test_atomic_rules():
    assert "atomic_without_commit" in codes(doc([A("enter_work"), {"atomic": [A("hold_verify"), A("place")]}]))
    assert "atomic_not_closed" in codes(doc([A("enter_work"), {"atomic": [A("clip"), A("hold_verify")]},
                                             A("release_safe"), A("arm_park")]))
    moving = [A("enter_work"), {"atomic": [A("twist"), A("move_to", args={"target": "crate"}), A("place")]}]
    assert "atomic_motion" in codes(doc(moving))
    nested = [A("enter_work"), {"atomic": [A("cut"), {"sequence": [A("backup", args={"d": 0.2})]}, A("place")]}]
    assert "atomic_motion" in codes(doc(nested))
    waiting = [A("enter_work"), {"atomic": [A("ground_pick"), {"wait_for": {"cond": "target_tracked",
                                                                           "timeout_s": 30}}, A("place")]}]
    assert "atomic_wait" in codes(doc(waiting))
    waiting[1]["atomic"][1]["wait_for"]["timeout_s"] = 5
    assert codes(doc(waiting)) == set()
    arm_ok = [A("enter_work"), {"atomic": [A("twist"), A("arm_approach"), A("lower"), A("place")]}, A("arm_park")]
    assert codes(doc(arm_ok)) == set()
    bridge = [A("enter_work"), {"atomic": [A("pick", args={"method": "clip"})]}, A("arm_park")]
    assert codes(doc(bridge)) == set()                     # the bridge pick places itself
    assert "commit_outside_atomic" in codes(doc([A("enter_work"), A("pick", args={"method": "clip"})]))
    second = [A("enter_work"), {"atomic": [A("clip"), A("twist"), A("place")]}]
    assert "commit_outside_atomic" in codes(doc(second))


def test_release_safe_end_then_park_is_allowed():
    rel = [A("enter_work"), {"atomic": [A("clip"), A("hold_verify"), A("release_safe")]}, A("arm_park")]
    assert codes(doc(rel)) == set()


def test_unsafe_terminal_and_finaliser_checks():
    # release_safe may keep holding: a run that then ends successfully without placing is flagged
    leftover = [A("enter_work"), {"atomic": [A("clip"), A("hold_verify"), A("release_safe")]}]
    got = static_check.check(compiler.compile_doc(doc(leftover)))
    assert {i.code for i in got} == set()                   # released: an assist is raised, not a silent drop
    bad_fin = {"sequence": [A("release_safe"), A("arm_park"), A("safe_hold")]}         # never stows the mast
    gate = ["ready_common", "crown_gate"]
    d = doc([A("raise_mast", args={"band": "mid"}), A("stow_mast")], preconditions=gate, on_pause=bad_fin)
    assert "finaliser_unsafe" in codes(d)
    good_fin = {"sequence": [A("release_safe"), A("arm_park"), A("stow_mast"), A("safe_hold")]}
    assert codes(doc(HARVEST, on_abort=good_fin)) == set()


def test_raise_mast_gates_and_stowed_only():
    assert "raise_mast_ungated" in codes(doc([A("raise_mast", args={"band": "mid"})]))
    gated = [{"check": {"cond": "crown_gate"}}, A("raise_mast", args={"band": "mid"}), A("stow_mast")]
    assert codes(doc(gated)) == set()
    guarded = [{"guard": {"cond": "wind_lt_crown", "on_violation": "rth",
                          "child": {"sequence": [A("raise_mast", args={"band": "mid"}), A("stow_mast")]}}}]
    assert codes(doc(guarded)) == set()
    waited = [{"wait_for": {"cond": "crown_gate", "timeout_s": 30}}, A("mast_snake", args={"band": "high"}),
              A("stow_mast")]
    assert codes(doc(waited)) == set()
    storm = doc(gated, mast="stowed_only")
    assert "raise_mast_stowed_only" in codes(storm)


def test_motion_before_ack_inside_composites():
    d = doc([])
    d["root"] = {"selector": [{"check": {"cond": "rain_dry"}}, A("move_to", args={"target": "x"})]}
    assert "motion_before_start_ack" in codes(d)
    d["root"] = {"sequence": [A("stop"), A("sit_belly"), A("start_ack"), A("move_to", args={"target": "x"})]}
    assert codes(d) == set()                                # stopping and sitting are safe directions
    par = doc([{"parallel": {"policy": "one", "children": [A("capture"), A("follow_rows")]}}])
    assert codes(par) == set()


def test_foreach_skip_paths_and_recover():
    loop = {"foreach": {"set": "fruit", "body": {"sequence": [
        A("enter_work"),
        {"recover": {"body": {"atomic": [A("clip"), A("hold_verify"), A("place")]},
                     "recovery": [A("arm_park")], "max_rounds": 2, "then": "skip(out_of_reach_marked)"}},
        A("arm_park"), A("exit_work")]}}}
    # a skipped target may still hold the clipped fruit (atomic failed after the commit): the next target's
    # arm_park is reachable while holding -> refused; and the recovery arm_park after a failed atomic too
    got = codes(doc([loop]))
    assert "arm_park_while_holding" in got
    safe = {"foreach": {"set": "fruit", "body": {"sequence": [
        A("enter_work"),
        {"recover": {"body": {"sequence": [A("arm_approach"), A("trap")]}, "recovery": [A("arm_park")],
                     "max_rounds": 2, "then": "skip(out_of_reach_marked)"}},
        {"atomic": [A("clip"), A("hold_verify"), A("place")]}, A("arm_park"), A("exit_work")]}}}
    assert codes(doc([safe])) == set()
    cont = {"recover": {"body": A("capture"), "recovery": [], "max_rounds": 1, "then": "continue"}}
    assert codes(doc([cont])) == set()
    tpl = {"template": {"id": "target_loop", "set": "trees", "body": {"sequence": [
        A("move_to", args={"target": "tree", "v_max": 0.15}), A("capture")]}}}
    assert codes(doc([tpl])) == set()
    sk = {"foreach": {"set": "t", "body": {"selector": {"memory": True, "children": [
        {"check": {"cond": "target_tracked", "else": "skip(range_lost)"}}, A("capture")]}}}}
    assert codes(doc([sk])) == set()
    gd = {"foreach": {"set": "t", "body": {"guard": {"cond": "visibility_ok", "on_violation": "skip_target",
                                                       "child": A("capture")}}}}
    assert codes(doc([gd])) == set()


def test_subtree_kind_assumes_ack_and_a_person_guard_is_not_required():
    sub = {"schema": "nandi.workflow/1", "id": "s", "version": 1, "kind": "subtree", "primitives_api": 1,
           "root": {"sequence": [A("move_to", args={"target": "x"})]}}
    assert codes(sub) == set()
    # A.0 rule 1: there is no person guard anywhere in these workflows and they are valid; the L0/L1 pause is
    # proven at run time (test_workflow_engine.py::test_person_pause_without_any_guard)
    assert codes(doc(HARVEST)) == set()
    st = static_check.initial_states(compiler.compile_doc(sub))
    assert all(s.ack for s in st)
