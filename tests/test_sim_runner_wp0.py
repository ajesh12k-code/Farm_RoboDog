"""WP0 sim_runner plug-in rules (ROS-free: rclpy and the message packages are stubbed; headless).

sim_runner is frozen after WP0, so the rules later work packages rely on are pinned here:
non-joint actuators (site, body / adhesion) neither crash the runner nor alias a robot joint, and stay out
of the hold and /nandi/joint_states; hooks see the safe hold in bus.values["hold"] and cannot drive the
robot actuators 0..24; a hook that raises is disabled and puts the runner into safe hold instead of ending
the node; hook resets run before mj_forward; the world_id / layout / hooks parameters are read-only.
"""
import importlib
import sys
import time
import types

import mujoco
import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import model_layout as mlay
from atharv_nandi.sim.world import compose


# ---------------------------------------------------------------------- ROS stubs
class _Rec:
    """Message / descriptor stand-in: keeps its keyword arguments as attributes."""

    def __init__(self, *args, **kw):
        self.__dict__.update(kw)


class _JointState(_Rec):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.header = types.SimpleNamespace(stamp=None)


class _Pub:
    def __init__(self, topic):
        self.topic, self.sent = topic, []

    def publish(self, msg):
        self.sent.append(msg)


class _Log:
    def __init__(self):
        self.lines = []

    def __getattr__(self, level):
        return lambda text: self.lines.append((level, text))


class _Node:
    def __init__(self, name):
        self.pubs, self.params, self._log = {}, {}, _Log()

    def create_publisher(self, _type, topic, _depth):
        return self.pubs.setdefault(topic, _Pub(topic))

    def create_subscription(self, *a, **k):
        return None

    def create_service(self, *a, **k):
        return None

    def create_client(self, *a, **k):
        return types.SimpleNamespace(service_is_ready=lambda: False)

    def declare_parameter(self, name, value, descriptor=None):
        self.params[name] = (value, descriptor)

    def get_logger(self):
        return self._log

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_msg=lambda: None))

    def destroy_node(self):
        pass


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


@pytest.fixture
def runner(monkeypatch):
    """(sim_runner module, ok-deadline setter) imported against stubbed rclpy / msgs."""
    deadline = {"t": 0.0}
    stubs = {
        "rclpy": _module("rclpy", ok=lambda: time.monotonic() < deadline["t"], init=lambda *a: None,
                         spin=lambda *a: None, shutdown=lambda: None),
        "rclpy.node": _module("rclpy.node", Node=_Node),
        "rcl_interfaces": _module("rcl_interfaces"),
        "rcl_interfaces.msg": _module("rcl_interfaces.msg", ParameterDescriptor=_Rec),
        "sensor_msgs": _module("sensor_msgs"), "sensor_msgs.msg": _module("sensor_msgs.msg", JointState=_JointState),
        "std_msgs": _module("std_msgs"),
        "std_msgs.msg": _module("std_msgs.msg", Float64MultiArray=_Rec, String=_Rec),
        "std_srvs": _module("std_srvs"), "std_srvs.srv": _module("std_srvs.srv", Trigger=_Rec),
    }
    for k, v in stubs.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.delitem(sys.modules, "atharv_nandi.nodes.sim_runner", raising=False)
    monkeypatch.setenv("NANDI_HEADLESS", "1")
    mod = importlib.import_module("atharv_nandi.nodes.sim_runner")

    def run_for(node, seconds):
        deadline["t"] = time.monotonic() + seconds
        node.run_sim()

    yield mod, run_for
    sys.modules.pop("atharv_nandi.nodes.sim_runner", None)


def _attach_odd_actuators(spec, field, ctx):
    """A tool with a site thrust actuator, a body adhesion actuator and one slide joint actuator."""
    b = spec.worldbody.add_body(name="t_tool", pos=[0.5, 0.5, 0.1])
    b.add_freejoint(name="t_tool_free")
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.02], mass=0.1)
    b.add_site(name="t_tool_site")
    spec.add_actuator(name="t_thrust", trntype=mujoco.mjtTrn.mjTRN_SITE, target="t_tool_site")
    adh = spec.add_actuator(name="t_adhesion", trntype=mujoco.mjtTrn.mjTRN_BODY, target="t_tool")
    adh.set_to_adhesion(gain=1.0)
    adh.ctrlrange, adh.ctrllimited = [0, 1], 1
    j = b.add_body(name="t_jaw", pos=[0.03, 0, 0])
    j.add_joint(name="t_jaw", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[-0.01, 0.01])
    j.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.005, 0.005, 0.005], mass=0.01)
    spec.add_actuator(name="t_jaw", target="t_jaw", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    # a joint-less body a hook moves (model mutation), restored by the hook's reset
    spec.worldbody.add_body(name="t_marker", pos=[1.0, 0.0, 0.5]).add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0)


def _make(mod, monkeypatch, reg=None):
    monkeypatch.setattr(hook_registry, "load_all", lambda *a, **k: reg if reg is not None else hook_registry.HookRegistry())
    with compose.temporary_module("t_odd_tools", compose.ORDER["tools"], _attach_odd_actuators):
        return mod.NandiSimRunner()


def _act(node, name):
    return mujoco.mj_name2id(node.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


# ---------------------------------------------------------------------- non-joint actuators
def test_site_and_body_actuators_do_not_crash_or_alias_a_robot_joint(runner, monkeypatch):
    mod, _ = runner
    node = _make(mod, monkeypatch)
    lay = mlay.current()
    assert node.model.nu == mlay.ROBOT_NU + 3
    # joint_states: 25 robot joints + the slide jaw; the site / body actuators are left out
    assert node.joint_names == [lay.actuator_joint[i] for i in range(mlay.ROBOT_NU)] + ["t_jaw"]
    assert len(set(node.joint_names)) == len(node.joint_names)
    assert node.act_qadr[-1] == lay.qpos_adr("t_jaw") and node.act_dadr[-1] == lay.dof_adr("t_jaw")
    node._publish_joint_states()
    msg = node.pubs["/nandi/joint_states"].sent[-1]
    assert len(msg.name) == len(msg.position) == len(msg.velocity) == len(msg.effort) == mlay.ROBOT_NU + 1


def test_safe_hold_holds_and_brakes_only_the_robot_actuators(runner, monkeypatch):
    mod, _ = runner
    node = _make(mod, monkeypatch)
    thrust, adh, jaw = _act(node, "t_thrust"), _act(node, "t_adhesion"), _act(node, "t_jaw")
    node.data.ctrl[[thrust, adh, jaw]] = [0.3, 0.7, 0.004]
    gain0 = node.model.actuator_gainprm.copy()
    node._enter_hold("operator", time.time())
    assert node.hold_ctrl.shape == (mlay.ROBOT_NU,)
    # appended actuators keep their ctrl and their gains (their hooks see bus.values["hold"])
    assert np.allclose(node.data.ctrl[[thrust, adh, jaw]], [0.3, 0.7, 0.004])
    assert np.array_equal(node.model.actuator_gainprm[mlay.ROBOT_NU:], gain0[mlay.ROBOT_NU:])
    if node.hold_kp_scale > 1.0:
        assert node.model.actuator_gainprm[node.hold_idx[0], 0] > gain0[node.hold_idx[0], 0]
    assert node.bus.values["hold"]["reason"] == "operator" and node.bus.holding()
    node._leave_hold()
    assert node.bus.values["hold"] is None and np.array_equal(node.model.actuator_gainprm, gain0)


def test_safe_hold_never_resets_model_fields_a_hook_set_on_appended_actuators(runner, monkeypatch):
    """A hook may re-gain its own actuator or brake its own dof (WP24 mast e-stop brake, WP2 tool swap):
    entering or leaving safe hold restores only the robot actuators and robot dofs to their base values."""
    mod, _ = runner
    node = _make(mod, monkeypatch)
    jaw = _act(node, "t_jaw")
    jaw_dof = mlay.current().dof_adr("t_jaw")
    node.model.actuator_gainprm[jaw, 0] = 7.0
    node.model.actuator_biasprm[jaw, 1] = -7.0
    node.model.dof_frictionloss[jaw_dof] = 0.3
    node._enter_hold("operator", time.time())
    assert node.model.actuator_gainprm[jaw, 0] == 7.0 and node.model.actuator_biasprm[jaw, 1] == -7.0
    assert node.model.dof_frictionloss[jaw_dof] == 0.3
    if node.hold_brake > 0.0:
        assert node.model.dof_frictionloss[node.robot_dadr[node.hold_idx[-1]]] > 0.0
    node._leave_hold()
    assert node.model.actuator_gainprm[jaw, 0] == 7.0 and node.model.dof_frictionloss[jaw_dof] == 0.3
    n = mlay.ROBOT_NU
    assert np.array_equal(node.model.actuator_gainprm[:n], node.base_gainprm[:n])
    assert np.array_equal(node.model.dof_frictionloss[node.robot_dadr], node.base_frictionloss[node.robot_dadr])


# ---------------------------------------------------------------------- hooks
def test_hooks_see_the_hold_and_cannot_drive_robot_actuators(runner, monkeypatch):
    mod, run_for = runner
    seen = {"hold": [], "ctrl0_in_step": [], "ctrl_jaw": []}
    reg = hook_registry.HookRegistry()

    def meddle(m, d, t, bus):
        seen["hold"].append(bus.values["hold"])
        d.ctrl[0] = 0.99                      # robot actuator: must never reach mj_step
        d.ctrl[m.nu - 1] = 0.006              # appended jaw: the hook's own

    def watch(m, d, t, bus):
        seen["ctrl0_in_step"].append(float(d.ctrl[0]))
        seen["ctrl_jaw"].append(float(d.ctrl[m.nu - 1]))

    reg.register("meddle", meddle, every_step=True)
    reg.register("watch", watch, rate_hz=500)
    node = _make(mod, monkeypatch, reg)
    node._enter_hold("operator", time.time())
    run_for(node, 0.25)
    assert len(seen["hold"]) > 20
    assert all(h is not None and h["reason"] == "operator" for h in seen["hold"])
    assert all(abs(v - node.hold_ctrl[0]) < 1e-12 for v in seen["ctrl0_in_step"])
    assert all(abs(v - 0.006) < 1e-12 for v in seen["ctrl_jaw"])


def test_a_raising_hook_is_disabled_and_puts_the_runner_into_safe_hold(runner, monkeypatch):
    mod, run_for = runner
    reg = hook_registry.HookRegistry()
    calls = {"good": 0, "bad": 0}

    def good(m, d, t, bus):
        calls["good"] += 1
        bus.publish("/nandi/t_good", {"x": np.float64(1.5), "v": np.arange(2)})

    def bad(m, d, t, bus):
        calls["bad"] += 1
        if d.time > 0.05:
            raise RuntimeError("boom")

    reg.register("good", good, rate_hz=50)
    reg.register("bad", bad, rate_hz=50)
    node = _make(mod, monkeypatch, reg)
    run_for(node, 0.4)                                   # the node survives: run_sim returns normally
    assert set(node.hook_faults) == {"bad"} and "boom" in node.hook_faults["bad"]
    assert node.hold_ctrl is not None and node.hold_reason == hook_registry.REASON_HOOK_FAULT
    assert node.bus.values["hold"]["reason"] == hook_registry.REASON_HOOK_FAULT
    bad_calls = calls["bad"]
    run_for(node, 0.1)
    assert calls["bad"] == bad_calls and calls["good"] > 10   # disabled; the others keep running
    assert node.pubs["/nandi/t_good"].sent[-1].data == '{"x":1.5,"v":[0,1]}'
    hb = node.pubs["/nandi/heartbeat"].sent[-1].data
    assert '"hook_faults"' in hb and "boom" in hb


def test_a_non_json_bus_payload_faults_only_its_hook():
    reg = hook_registry.HookRegistry()
    reg.register("nan", lambda m, d, t, bus: bus.publish("/nandi/t", {"v": float("nan")}))
    reg.register("obj", lambda m, d, t, bus: bus.publish("/nandi/t", object()))
    reg.register("ok", lambda m, d, t, bus: bus.publish("/nandi/t", [1]))
    bus, d = hook_registry.Bus(), types.SimpleNamespace(time=0.0)
    reg.tick(None, d, bus)
    assert sorted(n for n, _ in reg.drain_faults()) == ["nan", "obj"]
    assert bus.drain() == [("/nandi/t", "[1]")]
    assert set(reg.faulted()) == {"nan", "obj"}


def test_hook_reset_restores_the_model_before_kinematics(runner, monkeypatch):
    mod, _ = runner
    reg = hook_registry.HookRegistry()
    home = {}

    def reset(m, d):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "t_marker")
        m.body_pos[b] = home["pos"]

    reg.register("marker", lambda *a: None, mutates_model=True, reset=reset)
    node = _make(mod, monkeypatch, reg)
    b = mujoco.mj_name2id(node.model, mujoco.mjtObj.mjOBJ_BODY, "t_marker")
    home["pos"] = node.model.body_pos[b].copy()
    node.model.body_pos[b] = home["pos"] + [0.0, 0.0, 0.6]        # the hook's mutation
    mujoco.mj_forward(node.model, node.data)
    node._apply_reset()
    assert np.allclose(node.data.xpos[b], home["pos"])           # no stale mutated pose until the next step


def test_world_parameters_are_read_only_and_fast_hooks_are_flagged(runner, monkeypatch):
    mod, _ = runner
    reg = hook_registry.HookRegistry()
    reg.register("fast", lambda *a: None, rate_hz=1000)
    reg.register("ok50", lambda *a: None, rate_hz=50)
    node = _make(mod, monkeypatch, reg)
    for name in ("world_id", "layout", "hooks"):
        value, desc = node.params[name]
        assert desc is not None and desc.read_only is True, name
    assert node.params["world_id"][0] == node.world_id
    dt = node.model.opt.timestep
    assert reg.rate_capped(dt) == ["fast"]
    assert any(level == "warning" and "'fast'" in text for level, text in node.get_logger().lines)


def test_sim_state_publish_errors_never_end_the_node(runner, monkeypatch):
    mod, _ = runner
    from atharv_nandi.core import sim_state
    node = _make(mod, monkeypatch)
    sim_state.register_provider("t_broken", lambda m: 1 / 0, lambda m, p: None)
    try:
        node._publish_sim_state()
    finally:
        sim_state.unregister_provider("t_broken")
    assert "/nandi/sim_state" not in node.pubs or not node.pubs["/nandi/sim_state"].sent
    assert any(level == "error" and "sim_state" in text for level, text in node.get_logger().lines)
    node._publish_sim_state()
    assert node.pubs["/nandi/sim_state"].sent
