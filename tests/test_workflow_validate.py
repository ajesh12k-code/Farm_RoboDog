"""WP34 validator: structure, semantics and versioning (WORKFLOWS-SPEC A.6.1, A.6.2, A.6.6).

The seeded bad files of the WP34 acceptance row are mutations of one good preset (``GOOD``): unknown
primitive, missing timeout, unbounded retry, missing on_abort, tool_change while holding, blade outside
enter_work, commit primitive outside atomic, arm_park while holding without release_safe, motion before
start_ack, set() outside /wf/, a YAML that clears the e-stop. Each must be refused with its own issue code.
No py_trees needed.
"""
import copy
import os

import pytest
import yaml

from atharv_nandi.core.workflow import compiler, reasons, schema, validate
from atharv_nandi.core.workflow.primitives import registry as reg

GOOD = """
schema: nandi.workflow/1
id: harvest_guava_twist
version: 1
kind: preset
primitives_api: 1
preconditions: [ready_common, crown_gate]
interlocks: [common, arm_park]
approach_envelope: {min_standoff_m: 0.6, mast_max_under_skirt: stowed, enter_annulus: never}
params:
  v_mps: {default: 0.15, min: 0.05, max: 0.18}
root:
  sequence:
  - action: {do: start_ack, timeout_s: 600}
  - action: {do: move_to, args: {target: tree_entry, v_max: $v_mps, standoff_m: 0.9}, timeout_s: 60}
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


def good():
    return yaml.safe_load(GOOD)


def body(doc):
    """The foreach body sequence list of GOOD."""
    return doc["root"]["sequence"][3]["foreach"]["body"]["sequence"]


def codes(doc, lib=None):
    return {i.code for i in schema.errors(validate.validate_doc(doc, lib))}


def test_good_preset_is_valid():
    issues = validate.validate_doc(good())
    assert schema.errors(issues) == [], issues


# ------------------------------------------------------------------ the seeded bad files (acceptance)
def _unknown_primitive(d):
    d["root"]["sequence"][4] = {"action": {"do": "teleport", "timeout_s": 9}}


def _missing_timeout(d):
    del body(d)[1]["recover"]["body"]["sequence"][1]["action"]["timeout_s"]


def _unbounded_retry(d):
    body(d)[1]["recover"]["body"]["sequence"][1]["action"]["retry"] = {"n": 10}


def _missing_on_abort(d):
    del d["on_abort"]


def _tool_change_while_holding(d):
    b = body(d)
    b[2]["atomic"][2] = {"action": {"do": "release_safe", "timeout_s": 20}}   # may keep holding
    b.insert(4, {"action": {"do": "tool_change", "args": {"tool": "clipper"}, "timeout_s": 30}})


def _blade_outside_enter_work(d):
    del body(d)[0]


def _commit_outside_atomic(d):
    b = body(d)
    b[2:3] = b[2]["atomic"]


def _arm_park_holding_without_release_safe(d):
    d["on_abort"] = {"sequence": [{"action": {"do": "arm_park", "timeout_s": 10}},
                                  {"action": {"do": "stow_mast", "timeout_s": 9}},
                                  {"action": {"do": "safe_hold"}}]}


def _motion_before_start_ack(d):
    s = d["root"]["sequence"]
    s[0], s[1] = s[1], s[0]


def _set_outside_wf(d):
    body(d).append({"action": {"do": "set", "args": {"key": "/sensors/rtk_fixed", "value": True}}})


def _clears_estop(d):
    body(d).append({"action": {"do": "set", "args": {"key": "/safety/estop_clear", "value": True}}})


def _estop_reset_primitive(d):
    d["root"]["sequence"].insert(1, {"action": {"do": "estop_reset"}})


def _retry_on_commit(d):
    body(d)[2]["atomic"][0]["action"]["retry"] = {"n": 3}


SEEDED = {
    "unknown_primitive": (_unknown_primitive, "unknown_primitive"),
    "missing_timeout": (_missing_timeout, "missing_timeout"),
    "unbounded_retry": (_unbounded_retry, "unbounded_retry"),
    "missing_on_abort": (_missing_on_abort, "missing_on_abort"),
    "tool_change_while_holding": (_tool_change_while_holding, "tool_change_unsafe"),
    "blade_outside_enter_work": (_blade_outside_enter_work, "blade_outside_work"),
    "commit_outside_atomic": (_commit_outside_atomic, "commit_outside_atomic"),
    "arm_park_holding_no_release_safe": (_arm_park_holding_without_release_safe, "arm_park_while_holding"),
    "motion_before_start_ack": (_motion_before_start_ack, "motion_before_start_ack"),
    "set_outside_wf": (_set_outside_wf, "set_outside_wf"),
    "yaml_clears_estop": (_clears_estop, "estop_clear_attempt"),
    "estop_reset_primitive": (_estop_reset_primitive, "estop_clear_attempt"),
    "retry_on_commit": (_retry_on_commit, "retry_on_commit"),
}


@pytest.mark.parametrize("name", sorted(SEEDED))
def test_seeded_bad_file_is_refused(name):
    mutate, code = SEEDED[name]
    d = good()
    mutate(d)
    got = codes(d)
    assert code in got, (name, got)


def test_seeded_files_also_fail_through_validate_text():
    d = good()
    _clears_estop(d)
    issues = validate.validate_text(yaml.safe_dump(d))
    assert "estop_clear_attempt" in {i.code for i in issues}
    assert validate.validate_text("a: [")[0].code == "yaml_error"


# ------------------------------------------------------------------ more semantics
def test_set_inside_wf_is_accepted_and_every_condition_source_is_refused():
    d = good()
    body(d).append({"action": {"do": "set", "args": {"key": "seen", "value": 1}}})
    body(d).append({"action": {"do": "set", "args": {"key": "/wf/notes/last", "value": "x"}}})
    assert codes(d) == set()
    for cid, c in reg.CONDITIONS.items():                # mutation: YAML set() on every condition source
        assert not c.source.startswith("/wf/")
        d = good()
        body(d).append({"action": {"do": "set", "args": {"key": c.source, "value": True}}})
        assert "set_outside_wf" in codes(d), cid


@pytest.mark.parametrize("key", ["", "/", "../x", "a//b", 5])
def test_bad_set_keys(key):
    d = good()
    body(d).append({"action": {"do": "set", "args": {"key": key, "value": 1}}})
    assert "set_outside_wf" in codes(d)


def test_forbidden_primitives_say_why():
    for name in reg.FORBIDDEN:
        d = good()
        d["root"]["sequence"].insert(1, {"action": {"do": name}})
        got = codes(d)
        assert got & {"estop_clear_attempt", "forbidden_primitive"}, name


def test_args_reasons_conditions_and_params():
    cases = {
        "unknown_arg": lambda d: d["root"]["sequence"][1]["action"]["args"].update(warp=9),
        "missing_arg": lambda d: d["root"]["sequence"][1]["action"]["args"].pop("target"),
        "out_of_range": lambda d: d["root"]["sequence"][1]["action"]["args"].update(v_max=0.5),
        "bad_arg_type": lambda d: d["root"]["sequence"][1]["action"]["args"].update(target=3),
        "bad_arg_value": lambda d: body(d)[2]["atomic"][2]["action"]["args"].update(into="ground"),
        "unknown_param": lambda d: d["root"]["sequence"][1]["action"]["args"].update(v_max="$nope"),
        "unknown_reason": lambda d: body(d)[1]["recover"].update(then="skip(bored)"),
        "unknown_condition": lambda d: d["preconditions"].append("vibes"),
        "unknown_interlock": lambda d: d["interlocks"].append("vibes"),
        "missing_common_interlock": lambda d: d.update(interlocks=["arm_park"]),
        "missing_on_pause": lambda d: d.pop("on_pause"),
        "primitives_api_mismatch": lambda d: d.update(primitives_api=2),
        "bad_timeout": lambda d: d["root"]["sequence"][1]["action"].update(timeout_s=-1),
    }
    for code, fn in cases.items():
        d = good()
        fn(d)
        assert code in codes(d), code
    d = good()                                            # a param whose max breaks the arg range
    d["params"]["v_mps"]["max"] = 0.9
    assert "out_of_range" in codes(d)


def test_retry_and_rounds_bounds():
    d = good()
    a = body(d)[1]["recover"]["body"]["sequence"][1]["action"]
    a["retry"] = {"n": 3, "backoff_s": 0.5, "vary": {"warp": [0.1]}}
    assert "unknown_arg" in codes(d)
    a["retry"] = {"n": 3, "backoff_s": -1}
    assert "bad_value" in codes(d)
    a["retry"] = {"n": 3}
    assert codes(d) == set()
    body(d)[1]["recover"]["max_rounds"] = 6
    assert "unbounded_retry" in codes(d)
    d = good()                                            # nested recovers: 3 x 2 = 6 rounds per target
    inner = body(d)[1]
    inner["recover"]["max_rounds"] = 3
    body(d)[1] = {"recover": {"body": inner, "max_rounds": 2, "then": "skip(twist_failed)"}}
    assert "unbounded_retry" in codes(d)
    d = good()
    a = d["root"]["sequence"][1]["action"]
    a["retry"] = {"n": 1, "vary": {"v_max": "big"}}
    assert "bad_value" in codes(d)


def test_skip_outside_foreach_and_guard_checks():
    d = good()
    d["root"]["sequence"].insert(1, {"action": {"do": "skip", "args": {"reason": "rtk"}}})
    assert "skip_outside_foreach" in codes(d)
    d = good()
    d["root"]["sequence"][1] = {"guard": {"cond": "wind_lt_crown", "on_violation": "skip_target",
                                          "child": d["root"]["sequence"][1]}}
    assert "skip_outside_foreach" in codes(d)
    d = good()
    d["root"]["sequence"][1] = {"guard": {"cond": "nope", "on_violation": "ignore", "reason": "bored",
                                          "child": d["root"]["sequence"][1]}}
    assert {"unknown_condition", "bad_on_violation", "unknown_reason"} <= codes(d)
    d = good()
    d["root"]["sequence"].insert(1, {"wait_for": {"cond": "rain_dry"}})
    assert "missing_timeout" in codes(d)
    d = good()
    d["root"]["sequence"].insert(1, {"wait_for": {"cond": "rain_dry", "timeout_s": 0, "on_timeout": "skip(rain)"}})
    assert {"bad_timeout", "skip_outside_foreach"} <= codes(d)
    d = good()
    d["root"]["sequence"].insert(1, {"check": {"cond": "rain_dry", "else": "explode"}})
    assert "bad_value" in codes(d)
    d = good()
    body(d).insert(0, {"check": {"cond": "rain_dry", "else": "skip(nope)", "reason": "nah"}})
    assert "unknown_reason" in codes(d)
    d = good()
    body(d)[1]["recover"]["then"] = "explode"
    assert "bad_value" in codes(d)
    d = good()
    d["root"]["sequence"][1]["action"]["resumable"] = "rollback(move_to)"
    assert "bad_value" in codes(d)
    d = good()
    d["root"]["sequence"][1]["action"]["resumable"] = "sometimes"
    assert "bad_value" in codes(d)
    d = good()
    d["root"]["sequence"][1]["action"]["reason_on_fail"] = "nah"
    assert "unknown_reason" in codes(d)


# ------------------------------------------------------------------ structure
@pytest.mark.parametrize("mutate,code", [
    (lambda d: d.update(colour="red"), "unknown_field"),
    (lambda d: d.pop("root"), "missing_field"),
    (lambda d: d.update(schema="nandi.workflow/2"), "bad_schema"),
    (lambda d: d.update(version="3"), "bad_type"),
    (lambda d: d.update(kind="party"), "bad_value"),
    (lambda d: d.update(preconditions="ready_common"), "bad_type"),
    (lambda d: d.update(on_complete="dance"), "bad_value"),
    (lambda d: d.update(mast="tall"), "bad_value"),
    (lambda d: d.update(trigger="whenever"), "bad_value"),
    (lambda d: d.update(id=""), "bad_type"),
    (lambda d: d.update(approach_envelope="close"), "bad_type"),
    (lambda d: d["approach_envelope"].update(enter_annulus="sometimes"), "bad_value"),
    (lambda d: d["approach_envelope"].update(width=1), "unknown_field"),
    (lambda d: d["approach_envelope"].update(min_standoff_m="far"), "bad_type"),
    (lambda d: d.update(params=[1]), "bad_type"),
    (lambda d: d["params"].update(x=3), "bad_type"),
    (lambda d: d["params"].update(x={"default": 1, "colour": 2}), "unknown_field"),
    (lambda d: d["params"].update(x={"min": 3, "max": 1}), "bad_value"),
    (lambda d: d["params"].update(x={"default": 0, "min": 1}), "out_of_range"),
    (lambda d: d["params"].update(x={"default": 9, "max": 1}), "out_of_range"),
    (lambda d: d["root"].update(dance=[]), "bad_node"),
    (lambda d: d.update(root={"dance": []}), "unknown_node_type"),
    (lambda d: d.update(root={"sequence": []}), "empty_composite"),
    (lambda d: d.update(root={"sequence": 3}), "bad_type"),
    (lambda d: d.update(root={"sequence": {"children": 3}}), "bad_type"),
    (lambda d: d.update(root={"sequence": {"memory": 1, "children": [{"check": {"cond": "rain_dry"}}]}}), "bad_type"),
    (lambda d: d.update(root={"selector": {"memory": True}}), "missing_field"),
    (lambda d: d.update(root={"parallel": [1]}), "bad_type"),
    (lambda d: d.update(root={"parallel": {"policy": "most", "children": [{"check": {"cond": "rain_dry"}}]}}),
     "bad_value"),
    (lambda d: d.update(root={"action": "start_ack"}), "bad_type"),
    (lambda d: d.update(root={"action": {"do": 3}}), "bad_type"),
    (lambda d: d.update(root={"action": {"do": "stop", "args": [1]}}), "bad_type"),
    (lambda d: d.update(root={"action": {"do": "stop", "timeout_s": "x"}}), "bad_type"),
    (lambda d: d.update(root={"action": {"do": "stop", "retry": 3}}), "bad_type"),
    (lambda d: d.update(root={"action": {"do": "stop", "retry": {"n": 1, "forever": 1}}}), "unknown_field"),
    (lambda d: d.update(root={"action": {"do": "stop", "retry": {"vary": 1}}}), "bad_type"),
    (lambda d: d.update(root={"action": {"do": "stop", "colour": 1}}), "unknown_field"),
    (lambda d: d.update(root={"foreach": {"set": 1, "body": {"check": {"cond": "x"}}, "skip_ratio_halt": 2}}),
     "bad_type"),
    (lambda d: d.update(root={"foreach": {"set": "t", "body": {"check": {"cond": "x"}}, "skip_ratio_halt": 2}}),
     "bad_value"),
    (lambda d: d.update(root={"recover": {"body": {"check": {"cond": "x"}}, "recovery": 1}}), "bad_type"),
    (lambda d: d.update(root={"recover": {"body": {"check": {"cond": "x"}}, "max_rounds": "2"}}), "bad_type"),
    (lambda d: d.update(root={"recover": {"body": {"check": {"cond": "x"}}, "then": 1}}), "bad_type"),
    (lambda d: d.update(root={"recover": {"body": {"check": {"cond": "x"}},
                                          "recovery": [{"do": "stop", "colour": 1}]}}), "unknown_field"),
    (lambda d: d.update(root={"recover": {"body": {"check": {"cond": "x"}}, "recovery": [{"dance": 1}]}}),
     "unknown_node_type"),
    (lambda d: d.update(root={"guard": {"cond": "x", "on_violation": "pause"}}), "missing_field"),
    (lambda d: d.update(root={"wait_for": {"cond": "x", "timeout_s": "x"}}), "bad_type"),
    (lambda d: d.update(root={"subtree": {"ref": "noversion"}}), "bad_ref"),
    (lambda d: d.update(root={"template": {"name": 1}}), "bad_type"),
    (lambda d: d.update(root={"template": {"id": "x", "body": {"dance": 1}}}), "unknown_node_type"),
    (lambda d: d.update(root={"slot": 1}), "bad_type"),
    (lambda d: d.update(root={"check": [1]}), "bad_type"),
    (lambda d: d.update(on_pause={"ref": 3}), "bad_ref"),
])
def test_structure_issues(mutate, code):
    d = good()
    mutate(d)
    got = {i.code for i in schema.validate_structure(d)}
    assert code in got, got


def test_structure_basics():
    assert schema.validate_structure([1])[0].code == "not_a_mapping"
    assert schema.node_type([1]) == (None, None)
    assert schema.children_of("sequence", 3) == []
    assert "error unknown_field at /x: y" == str(schema.Issue("unknown_field", "/x", "y"))
    for t in ("manual", "schedule(0 6 * * *)", "event(storm_over)", {"schedule": "x"}):
        d = good()
        d["trigger"] = t
        assert not schema.validate_structure(d)


# ------------------------------------------------------------------ refs, templates, library
def lib_with(*docs):
    lib = compiler.Library()
    for d in docs:
        lib.add(d)
    return lib


def subtree_doc(wid, root, version=1, **extra):
    d = {"schema": "nandi.workflow/1", "id": wid, "version": version, "kind": "subtree", "primitives_api": 1,
         "root": root}
    d.update(extra)
    return d


def test_subtree_refs_resolve_and_cycles_are_refused():
    a = subtree_doc("a", {"subtree": {"ref": "b@1"}})
    b = subtree_doc("b", {"subtree": {"ref": "a@1"}})
    lib = lib_with(a, b)
    d = good()
    d["root"]["sequence"].insert(1, {"subtree": {"ref": "a@1"}})
    assert "subtree_cycle" in codes(d, lib)
    d = good()
    d["root"]["sequence"].insert(1, {"subtree": {"ref": "ghost@1"}})
    assert "unresolved_ref" in codes(d)
    d = good()
    d["root"]["sequence"].insert(1, {"template": "ghost"})
    assert "unresolved_ref" in codes(d)
    pres = dict(good(), id="other")
    lib = lib_with(pres)
    d = good()
    d["root"]["sequence"].insert(1, {"subtree": {"ref": "other@1"}})
    assert "bad_ref" in codes(d, lib)
    d = good()
    d["root"]["sequence"].insert(1, {"template": {"id": "capture_station", "radius": 2}})
    assert "bad_template_arg" in codes(d)
    lib = lib_with(subtree_doc("holey", {"sequence": [{"slot": "body"}]}))
    d = good()
    d["root"]["sequence"].insert(1, {"subtree": {"ref": "holey@1"}})
    assert "unfilled_slot" in codes(d, lib)
    d = good()
    d["root"]["sequence"].insert(1, {"slot": "body"})
    assert "slot_outside_template" in codes(d)
    d = good()
    d["root"]["sequence"][4]["action"]["resumable"] = "roll back"
    assert "bad_value" in codes(d)


def test_spec_example_tree360_validates():
    """WORKFLOWS-SPEC A.2 example (abridged), with the crown_gate precondition its raise_mast needs."""
    text = """
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
    assert schema.errors(validate.validate_text(text)) == []
    no_gate = text.replace(", crown_gate]", "]")
    assert "raise_mast_ungated" in {i.code for i in validate.validate_text(no_gate)}


def test_lib_area_is_valid_and_indexed():
    out = validate.validate_library()
    assert {k for k in out if k.startswith("_lib/")} >= {"_lib/finaliser_default@1", "_lib/resume_check@1",
                                                       "_lib/target_loop@1", "_lib/capture_station@1"}
    bad = {k: [str(i) for i in v] for k, v in out.items() if schema.errors(v)}
    assert not bad, bad
    assert validate.main([compiler.WORKFLOWS_DIR]) == 0


def _area(tmp_path, doc, index_extra=None, area="presets"):
    d = tmp_path / area
    d.mkdir(parents=True, exist_ok=True)
    (d / "w.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    entry = {"id": doc["id"], "version": doc["version"], "file": "w.yaml"}
    entry.update(index_extra or {})
    (d / "INDEX.yaml").write_text(yaml.safe_dump({"entries": [entry]}))
    return d


def test_versioning_edit_without_bump_is_refused(tmp_path):
    lib_dir = tmp_path / "wf"
    shutil_copy_lib(lib_dir)
    area = _area(lib_dir, good())
    assert validate.reindex(str(area)) == ["harvest_guava_twist@1"]
    assert validate.reindex(str(area)) == []
    lib = compiler.Library(str(lib_dir))
    res = validate.validate_library(lib)
    assert schema.errors(res["presets/harvest_guava_twist@1"]) == []
    d = good()                                            # edit without a bump
    d["root"]["sequence"][4]["action"]["timeout_s"] = 8
    (area / "w.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    res = validate.validate_library(compiler.Library(str(lib_dir)))
    assert "version_not_bumped" in {i.code for i in res["presets/harvest_guava_twist@1"]}
    with pytest.raises(ValueError, match="without a version bump"):
        validate.reindex(str(area))
    assert validate.main([str(lib_dir)]) == 1
    d["version"] = 2                                      # bumped but INDEX not updated
    (area / "w.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    res = validate.validate_library(compiler.Library(str(lib_dir)))
    assert "index_stale" in {i.code for i in res["presets/harvest_guava_twist@1"]}
    assert validate.main(["--reindex", str(lib_dir)]) == 0
    lib = compiler.Library(str(lib_dir))
    assert lib.get("harvest_guava_twist").version == 2
    assert lib.get("harvest_guava_twist@2").ident.startswith("harvest_guava_twist@2#")


def shutil_copy_lib(dst):
    import shutil
    shutil.copytree(os.path.join(compiler.WORKFLOWS_DIR, "_lib"), str(dst / "_lib"))


def test_index_problems(tmp_path):
    lib_dir = tmp_path / "wf"
    shutil_copy_lib(lib_dir)
    area = _area(lib_dir, good(), {"primitives_api": 7})
    res = validate.validate_library(compiler.Library(str(lib_dir)))
    got = {i.code for i in res["presets/harvest_guava_twist@1"]}
    assert {"index_stale", "index_mismatch"} <= got                     # no sha256; api differs
    (area / "INDEX.yaml").write_text(yaml.safe_dump({"entries": [{"id": "x"}, {"id": "harvest_guava_twist",
                                     "version": 1, "file": "w.yaml"}, {"id": "harvest_guava_twist", "version": 1,
                                                                       "file": "w.yaml"}]}))
    lib = compiler.Library(str(lib_dir))
    assert {i.code for i in lib.index_issues} == {"bad_index", "duplicate_entry"}
    assert "INDEX" in validate.validate_library(lib)
    (area / "INDEX.yaml").write_text(yaml.safe_dump({"entries": [{"id": "renamed", "version": 1, "file": "w.yaml"}]}))
    res = validate.validate_library(compiler.Library(str(lib_dir)))
    assert "index_mismatch" in {i.code for i in res["presets/renamed@1"]}
    (area / "w.yaml").write_text("- 1\n")
    res = validate.validate_library(compiler.Library(str(lib_dir)))
    assert res["presets/renamed@1"][0].code == "not_a_mapping"
    (area / "w.yaml").write_text("a: [\n")
    res = validate.validate_library(compiler.Library(str(lib_dir)))
    assert res["presets/renamed@1"][0].code == "unreadable"
    with pytest.raises(KeyError):
        compiler.Library(str(lib_dir)).get("nothing@1")
    with pytest.raises(KeyError):
        compiler.Library(str(lib_dir)).latest("nothing")
    assert compiler.Library(str(lib_dir)).area("presets")[0].id == "renamed"


def test_reasons_enum():
    assert {"out_of_reach_marked", "band_missing", "workflow_changed", "rth_skip_ratio", "arc_blocked",
            "dropped", "held_at_estop"} <= reasons.REASONS
    assert reasons.check("rtk") == "rtk"
    with pytest.raises(ValueError):
        reasons.check("bored")
    assert len(reasons.REASONS) == len(set(reasons.MP_05 + reasons.CHORES_D0 + reasons.RTH_E8 + reasons.ENGINE))


def test_registry_catalogue_covers_spec_groups():
    must = {"start_ack", "wait_operator", "plan_route", "move_to", "follow_rows", "orbit_arc", "settle", "stop",
            "backup", "alt_heading", "sit_belly", "stand", "return_to_checkpoint", "raise_mast", "stow_mast",
            "gimbal_aim", "mast_snake", "capture", "lidar_profile", "terrain_patch", "detect", "track_dedupe",
            "stage_estimate", "coverage_update", "read_trap", "visibility_check", "update_record", "triage", "mark",
            "jobsheet_line", "deepscan_request", "manifest_close", "log_kpi", "enter_work", "exit_work",
            "arm_approach", "arm_park", "tool_change", "roll_align", "trap", "clip", "twist", "cut", "ground_pick",
            "hold_verify", "place", "lower", "release_safe", "pick", "lead_stop", "follow_tag", "assist_request",
            "request_rth", "dock_stage", "dock_detect", "dock_approach", "dock_final_align", "under_belly_clear",
            "dock_seat", "verify_contacts", "charge", "dock_stand", "undock", "unload_transfer", "sync_stub",
            "tools_safe", "safe_hold", "beacon", "request", "skip", "halt", "set"}
    assert must <= set(reg.PRIMITIVES)
    assert {p for p, s in reg.PRIMITIVES.items() if s.commit} == {"clip", "twist", "cut", "ground_pick", "pick"}
    assert set(reg.READY_COMMON) <= set(reg.CONDITIONS)
    with pytest.raises(KeyError):
        reg.spec("teleport")
    with pytest.raises(KeyError):
        reg.bind("teleport", lambda s: None)
    assert compiler.parse_call("skip(arc_blocked)") == ("skip", "arc_blocked")
    with pytest.raises(compiler.CompileError):
        compiler.parse_call("skip(a)(b)")
    with pytest.raises(compiler.CompileError):
        compiler.parse_call(3)


def test_compiler_ir_defaults_and_identity():
    comp = compiler.compile_text(GOOD)
    assert comp.ident.startswith("harvest_guava_twist@1#") and len(comp.ident.split("#")[1]) == 8
    assert comp.kind == "preset" and comp.version == 1 and comp.id == "harvest_guava_twist"
    assert comp.root.f["memory"] is True
    a = comp.root.children[1]
    assert a.do == "move_to" and a.f["resumable"] == ("restart", None)
    approach = [n for n in comp.root.walk() if n.do == "arm_approach"][0]
    assert approach.f["resumable"] == ("rollback", "arm_park")
    assert comp.on_pause.type == "subtree" and comp.on_pause.f["ref"] == "finaliser_default@1"
    assert comp.size() > 15 and len(list(comp.nodes())) > comp.size()
    sel = compiler.compile_node({"selector": [{"check": {"cond": "rain_dry"}}]})
    assert sel.f["memory"] is False
    d = good()
    d.pop("on_pause")
    assert compiler.compile_doc(d).on_pause.f["ref"] == "finaliser_default@1"   # the default finaliser
    assert compiler.substitute({"a": "${x}-y", "b": "$x", "c": ["$z"]}, {"x": 2}, {}) == {"a": "2-y", "b": 2,
                                                                                          "c": ["$z"]}


def test_check_value_against_the_arg_spec():
    from atharv_nandi.core.workflow.primitives import registry as R
    ok = [(R.ArgSpec("num", min=0.0, max=0.3), 0.2), (R.ArgSpec("int"), 3), (R.ArgSpec("enum", choices=("a",)), "a"),
          (R.ArgSpec("str"), "x"), (R.ArgSpec("bool"), False), (R.ArgSpec("list"), []),
          (R.ArgSpec("reason"), "wind"), (R.ArgSpec("any"), object())]
    bad = [(R.ArgSpec("num", min=0.0, max=0.3), 0.31), (R.ArgSpec("num"), "fast"), (R.ArgSpec("num"), True),
           (R.ArgSpec("num"), float("nan")), (R.ArgSpec("int"), 1.5), (R.ArgSpec("enum", choices=("a",)), "b"),
           (R.ArgSpec("str"), 1), (R.ArgSpec("bool"), 1), (R.ArgSpec("list"), "x"), (R.ArgSpec("reason"), "nope")]
    assert all(R.check_value(s, v) is None for s, v in ok)
    assert all(R.check_value(s, v) for s, v in bad)
