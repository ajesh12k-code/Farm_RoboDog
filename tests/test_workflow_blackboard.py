"""WP34 blackboard namespaces and key access control (WORKFLOWS-SPEC A.3).

Validate time (no py_trees): only relative or ``/wf/`` keys are YAML-writable. Run time (py_trees key access
control): condition clients are READ-only, every source key has one EXCLUSIVE writer (the L0 mirror /
adapters), a run writes only ``/wf/<run_id>/*``; for every condition id a runtime write attempt raises.
"""
import pytest

from atharv_nandi.core.workflow import blackboard as B
from atharv_nandi.core.workflow.primitives import registry as reg


@pytest.mark.parametrize("key,ok,canon", [
    ("note", True, "note"), ("/wf/note", True, "note"), ("sets/trees", True, "sets/trees"),
    ("/safety/estop_clear", False, None), ("/sensors/rtk_fixed", False, None), ("/energy/x", False, None),
    ("/elsewhere", False, None), ("", False, None), ("a/../b", False, None), (None, False, None)])
def test_yaml_keys(key, ok, canon):
    good, out = B.yaml_key(key)
    assert good is ok
    if ok:
        assert out == canon
        assert B.run_key("r1", out) == f"/wf/r1/{canon}"


def test_every_condition_source_is_outside_wf():
    for cid, c in reg.CONDITIONS.items():
        assert c.source.startswith(B.SOURCE_NAMESPACES), cid
        assert B.yaml_key(c.source)[0] is False
        assert B.source_key(cid) == c.source
    assert set(B.NAMESPACES) == set(B.SOURCE_NAMESPACES) | {"/wf/"}


py_trees = pytest.importorskip("py_trees")


def test_runtime_access_control():
    bb = B.RunBlackboard("bbrun")
    B.mirror_values({"estop_clear": True, "/sensors/rtk_fixed": False})
    assert bb.condition("estop_clear") is True and bb.condition("rtk_fixed") is False
    assert bb.read_source("/safety/estop_clear") is True
    for cid, c in reg.CONDITIONS.items():                         # mutation: a behaviour tries to spoof it
        with pytest.raises(AttributeError):
            bb.try_write_source(c.source, True)
    with pytest.raises(AttributeError):                           # a second writer cannot even register
        rogue = py_trees.blackboard.Client(name="rogue")
        rogue.register_key(key="/safety/estop_clear", access=py_trees.common.Access.WRITE)
    with pytest.raises(PermissionError):
        bb.set("/safety/estop_clear", True)
    with pytest.raises(PermissionError):
        bb.get("/safety/estop_clear")
    with pytest.raises(PermissionError):
        B.mirror().write("/wf/bbrun/x", 1)
    B.mirror().write("/sensors/new_sensor", 3)                    # a new source key is claimed exclusively
    assert py_trees.blackboard.Blackboard.get("/sensors/new_sensor") == 3
    full = bb.set("note", 5)
    assert full == "/wf/bbrun/note" and bb.get("/wf/note") == 5 and bb.get("missing", 9) == 9
    bb.set("note", 6)
    assert bb.snapshot() == {"note": 6}
    other = B.RunBlackboard("bbother")
    assert other.get("note") is None                              # runs are isolated by prefix
    other.restore({"a": 1, "b": [1, 2]})
    assert other.snapshot() == {"a": 1, "b": [1, 2]}
    bb.close()
    other.close()
    assert not py_trees.blackboard.Blackboard.exists("/wf/bbrun/note")


def test_condition_default_when_never_mirrored():
    bb = B.RunBlackboard("bbdef")
    key = reg.CONDITIONS["tag_seen"].source
    py_trees.blackboard.Blackboard.unset(key)
    B.mirror().last.pop(key, None)
    assert bb.condition("tag_seen", default=False) is False
    assert bb.read_source(key, "none") == "none"
    B.mirror_values({"tag_seen": True})
    assert bb.condition("tag_seen") is True
    bb.close()
    bb = B.RunBlackboard("bbgone")
    bb.set("x", 1)
    py_trees.blackboard.Blackboard.unset("/wf/bbgone/x")
    assert bb.get("x", "gone") == "gone" and bb.snapshot() == {}
    bb.close()


def test_debug_stream_is_opt_in():
    B.set_debug(False)
    assert py_trees.blackboard.Blackboard.activity_stream is None
    bb = B.RunBlackboard("bbdebug", debug=True)
    assert py_trees.blackboard.Blackboard.activity_stream is not None
    bb.close()
    B.set_debug(False)
