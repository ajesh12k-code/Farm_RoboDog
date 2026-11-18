"""WPCAP HAL: ros2_control-style interfaces, the MuJoCo sim backend and the stub hardware backend
(docs/cap-libraries.md 2). Headless, ROS-free; the MuJoCo tests use the legacy world (fast to compile)."""
import copy
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core.hal import (
    HW_IF_EFFORT, HW_IF_POSITION, HW_IF_VELOCITY, HalError, LifecycleState, ReturnType, backend_name,
    build_resource_manager, joint_limits, load_config, parse_hardware,
)
from atharv_nandi.core.hal.controllers import (
    ControllerManager, ForwardCommandController, JointStateBroadcaster, JointTrajectoryController,
)
from atharv_nandi.core.planning import ArmPlanner, time_parameterize
from atharv_nandi.core.planning.collision_scene import ARM_JOINTS

DOC = load_config()


# ---------------------------------------------------------------------- config
def test_config_parses_every_robot_joint():
    hw = parse_hardware(DOC)
    assert [h.name for h in hw] == ["nandi_arm", "nandi_legs"]
    joints = [j for h in hw for j in h.joint_names]
    assert len(joints) == 25 == len(set(joints))
    assert joints[:5] == list(ARM_JOINTS)
    sh = hw[0].joint("arm_shoulder")
    assert {c.name for c in sh.command_interfaces} == {HW_IF_POSITION, HW_IF_VELOCITY, HW_IF_EFFORT}
    assert sh.command(HW_IF_POSITION).min == -1.6 and sh.command(HW_IF_POSITION).max == 2.4
    assert {s.name for s in hw[1].sensors} == {"imu"} and {s.name for s in hw[0].sensors} == {"tool_flange_ft"}
    v, a = joint_limits(DOC, ARM_JOINTS)
    assert v == [0.15, 1.2, 1.0, 1.2, 1.5] and all(x > 0 for x in a)


def test_config_against_the_model():
    """Every hal joint exists in the MJCF and its position clamp equals the joint range."""
    m = farm_scene.load_farm_model()
    for h in parse_hardware(DOC):
        for j in h.joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j.name)
            assert jid >= 0, j.name
            lo, hi = m.jnt_range[jid]
            pos = j.command(HW_IF_POSITION)
            assert (pos.min, pos.max) == pytest.approx((lo, hi)), j.name


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.update(schema="nandi.hal/0"), "schema"),
    (lambda d: d["hardware"][0]["joints"].append(copy.deepcopy(d["hardware"][0]["joints"][0])), "twice"),
    (lambda d: d["hardware"][0]["joints"][0]["command_interfaces"].update(torque={}), "unknown command"),
    (lambda d: d["hardware"][0]["joints"][0].update(command_interfaces={}), "no command interface"),
    (lambda d: d.update(hardware=[]), "no hardware"),
])
def test_config_errors(mutate, msg):
    doc = copy.deepcopy(DOC)
    mutate(doc)
    with pytest.raises(HalError, match=msg):
        parse_hardware(doc)


def test_backend_selection(monkeypatch):
    assert backend_name(DOC) == "mujoco"
    monkeypatch.setenv("NANDI_HAL_BACKEND", "stub")
    assert backend_name(DOC) == "stub"
    with pytest.raises(HalError, match="no plugin"):
        build_resource_manager(DOC, backend="canbus")


# ---------------------------------------------------------------------- stub backend
def stub_rm(**kw):
    rm = build_resource_manager(DOC, backend="stub", factory_kwargs=kw)
    rm.activate_all()
    return rm


def test_lifecycle_transitions():
    rm = build_resource_manager(DOC, backend="stub")
    arm = rm.components["nandi_arm"]
    assert arm.state == LifecycleState.UNCONFIGURED
    with pytest.raises(HalError, match="cannot activate"):
        rm.transition("nandi_arm", "activate")
    assert rm.transition("nandi_arm", "configure") == LifecycleState.INACTIVE
    assert not rm.is_available("arm_yaw/position")
    with pytest.raises(HalError, match="not an available"):
        rm.claim(["arm_yaw/position"], "c")
    assert rm.transition("nandi_arm", "activate") == LifecycleState.ACTIVE
    rm.claim(["arm_yaw/position"], "c")
    rm.transition("nandi_arm", "deactivate")
    assert rm.claimed() == {}                                  # deactivation releases the claims
    assert rm.transition("nandi_arm", "shutdown") == LifecycleState.FINALIZED
    st = rm.component_states()
    assert st["nandi_arm"]["state"] == "finalized" and st["nandi_legs"]["backend"] == "stub"
    assert len(rm.state_interface_names()) == 25 * 3 + 6 + 10
    assert len(rm.command_interface_names()) == 5 * 3 + 2 * 2 + 18 * 3


def test_exclusive_claims_and_mode_switch():
    rm = stub_rm()
    rm.claim(["arm_yaw/position", "arm_elbow/position"], "a")
    with pytest.raises(HalError, match="already claimed"):
        rm.claim(["arm_yaw/position"], "b")
    with pytest.raises(HalError, match="already commanded"):
        rm.claim(["arm_yaw/velocity"], "b")
    with pytest.raises(HalError, match="refused"):
        rm.claim(["arm_wrist_pitch/position", "arm_wrist_pitch/velocity"], "b")
    assert "arm_wrist_pitch/position" not in rm.claimed()    # all or none
    rm.release("a")
    rm.claim(["arm_yaw/velocity"], "b")
    assert rm.components["nandi_arm"].command_mode["arm_yaw"] == HW_IF_VELOCITY
    with pytest.raises(HalError, match="already owned"):
        from atharv_nandi.core.hal.stub_backend import StubSystem
        rm.add(StubSystem(), parse_hardware(DOC)[0].__class__("dup", joints=parse_hardware(DOC)[0].joints))


def test_command_clamped_to_limits():
    rm = stub_rm()
    (h,) = rm.claim(["arm_shoulder/position"], "c")
    h.set_value(9.0)
    assert h.get_value() == 2.4
    h.set_value(-9.0)
    assert h.get_value() == -1.6


def test_stub_trajectory_execution():
    rm = stub_rm()
    cm = ControllerManager(rm)
    jsb = cm.add(JointStateBroadcaster())
    jtc = cm.add(JointTrajectoryController("arm_jtc", ARM_JOINTS, goal_tolerance=0.01))
    cm.activate("joint_state_broadcaster")
    cm.activate("arm_jtc")
    v, a = joint_limits(DOC, ARM_JOINTS)
    path = np.array([[0, 0, 0, 0, 0], [0.1, 0.8, 0.5, -0.6, 0.4], [0.0, 1.2, 0.2, -0.2, 0.9]], dtype=float)
    tr = time_parameterize(path, v, a)
    jtc.follow(tr)
    dt, t = 0.01, 0.0
    while t < tr.duration + 0.5:
        cm.update(t, dt)
        t += dt
    assert jtc.done
    q = np.array(jsb.msg["position"][:5])
    assert np.max(np.abs(q - path[-1])) < 0.01
    assert jtc.max_error < 0.1
    assert len(jsb.msg["name"]) == 25 and jsb.msg["name"][:5] == list(ARM_JOINTS)
    print(f"stub JTC: duration {tr.duration:.2f} s, max tracking error {jtc.max_error:.4f}, final "
          f"{np.max(np.abs(q - path[-1])):.5f}")


def test_stub_velocity_and_effort_modes():
    rm = stub_rm()
    cm = ControllerManager(rm)
    vel = cm.add(ForwardCommandController("yaw_vel", ["arm_yaw"], HW_IF_VELOCITY))
    eff = cm.add(ForwardCommandController("wrist_eff", ["arm_wrist_pitch"], HW_IF_EFFORT))
    cm.activate("yaw_vel")
    cm.activate("wrist_eff")
    vel.set_command([0.5])
    eff.set_command([0.2])
    for k in range(100):
        cm.update(k * 0.01, 0.01)
    assert rm.state("arm_yaw/position") == pytest.approx(0.5, abs=0.03)
    assert rm.state("arm_yaw/velocity") == pytest.approx(0.5, abs=0.01)
    assert rm.state("arm_wrist_pitch/effort") == pytest.approx(0.2)
    assert rm.state("arm_wrist_pitch/velocity") > 0.0
    vel.set_command([9.0])                                   # clamped to the interface max 1.2
    for k in range(100):
        cm.update(1.0 + k * 0.01, 0.01)
    assert rm.state("arm_yaw/velocity") == pytest.approx(1.2, abs=0.01)


def test_stub_fault_deactivates_controllers_and_recovers():
    rm = stub_rm()
    cm = ControllerManager(rm)
    fwd = cm.add(ForwardCommandController("arm_pos", ["arm_yaw"], HW_IF_POSITION))
    legs = cm.add(ForwardCommandController("leg_pos", ["FR_hip_joint"], HW_IF_POSITION))
    cm.activate("arm_pos")
    cm.activate("leg_pos")
    rm.components["nandi_arm"].inject_fault("comm_timeout")
    out = cm.update(0.0, 0.01)
    assert out["read"]["nandi_arm"] == ReturnType.ERROR
    assert rm.components["nandi_arm"].state == LifecycleState.UNCONFIGURED
    assert not fwd.active and legs.active                     # only the failed component's controllers
    assert "arm_yaw/position" not in rm.claimed() and "FR_hip_joint/position" in rm.claimed()
    assert rm.component_states()["nandi_arm"]["faults"] == ["comm_timeout"]
    assert any("arm_pos" in e for e in cm.events)
    rm.transition("nandi_arm", "configure")
    rm.transition("nandi_arm", "activate")
    cm.activate("arm_pos")
    assert cm.update(0.01, 0.01)["read"]["nandi_arm"] == ReturnType.OK
    rm.components["nandi_legs"].inject_fault("overcurrent:FR_hip_joint")
    assert cm.update(0.02, 0.01)["write"]["nandi_legs"] == ReturnType.ERROR and not legs.active


def test_stub_delay_and_noise_deterministic():
    def run(seed):
        rm = stub_rm(delay_steps=3, noise_std=0.002, seed=seed)
        (h,) = rm.claim(["arm_elbow/position"], "c")
        h.set_value(0.5)
        out = []
        for k in range(10):
            rm.write(k * 0.01, 0.01)
            rm.read(k * 0.01, 0.01)
            out.append(rm.state("arm_elbow/position"))
        return out
    a, b = run(1), run(1)
    assert a == b and a != run(2)
    assert max(abs(x) for x in a[:3]) < 0.01                  # 3-step transport delay: no motion yet
    assert 0.05 < a[-1] < 1.2 * 0.07 + 0.01                   # 7 moving steps at max_velocity 1.2 rad/s


# ---------------------------------------------------------------------- MuJoCo backend
@pytest.fixture()
def sim():
    m = farm_scene.load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    rm = build_resource_manager(DOC, backend="mujoco", factory_kwargs={"model": m, "data": d})
    rm.activate_all()
    return m, d, rm


def run_sim(m, d, cm, seconds, t0=0.0, rate_hz=100.0):
    n_sub = int(round(1.0 / rate_hz / m.opt.timestep))
    t = t0
    for _ in range(int(seconds * rate_hz)):
        cm.update(t, 1.0 / rate_hz)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        t += 1.0 / rate_hz
    return t


def test_mujoco_states_match_data(sim):
    m, d, rm = sim
    cm = ControllerManager(rm)
    jsb = cm.add(JointStateBroadcaster())
    cm.activate("joint_state_broadcaster")
    run_sim(m, d, cm, 0.2)
    cm.update(0.2, 0.01)
    for name, q, v in zip(jsb.msg["name"], jsb.msg["position"], jsb.msg["velocity"]):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert q == pytest.approx(d.qpos[m.jnt_qposadr[jid]]) and v == pytest.approx(d.qvel[m.jnt_dofadr[jid]])
    quat = [rm.state(f"imu/orientation.{a}") for a in "wxyz"]
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-6)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
    assert rm.state("imu/linear_acceleration.z") == pytest.approx(d.sensordata[m.sensor_adr[sid] + 2])
    assert rm.state("tool_flange_ft/force.z") == rm.state("tool_flange_ft/force.z")   # finite, not NaN


def test_mujoco_activation_holds_pose(sim):
    m, d, rm = sim
    cm = ControllerManager(rm)
    q0 = d.qpos.copy()
    run_sim(m, d, cm, 1.0)
    assert d.qpos[2] > 0.25                                    # still standing on the commanded hold
    assert np.max(np.abs(d.qpos[7:12] - q0[7:12])) < 0.03


def test_mujoco_executes_planned_arm_trajectory(sim):
    m, d, rm = sim
    cm = ControllerManager(rm)
    cm.add(JointStateBroadcaster())
    jtc = cm.add(JointTrajectoryController("arm_jtc", ARM_JOINTS, goal_tolerance=0.02))
    cm.activate("arm_jtc")
    planner = ArmPlanner.from_config(m)
    adr = planner.scene.qadr
    goal = d.qpos[adr] + np.array([0.08, 0.9, 0.5, 0.6, -0.5])
    res = planner.plan(d.qpos, goal)
    assert res.ok, res.reason
    jtc.follow(res.trajectory)
    t = run_sim(m, d, cm, res.trajectory.duration + 1.0)
    assert jtc.done, np.abs(d.qpos[adr] - goal)
    err = np.max(np.abs(d.qpos[adr] - goal))
    print(f"MuJoCo JTC: duration {res.trajectory.duration:.2f} s, max tracking error {jtc.max_error:.4f}, "
          f"final error {err:.4f}, backend {res.backend}, plan {res.plan_time_s:.3f} s")
    assert err < 0.02 and jtc.max_error < 0.15
    assert d.qpos[2] > 0.25


def test_mujoco_velocity_mode(sim):
    m, d, rm = sim
    cm = ControllerManager(rm)
    yaw = cm.add(ForwardCommandController("yaw_vel", ["arm_yaw"], HW_IF_VELOCITY))
    cm.activate("yaw_vel")
    y0 = rm.state("arm_yaw/position") if not math.isnan(rm.state("arm_yaw/position")) else 0.0
    yaw.set_command([0.5])
    run_sim(m, d, cm, 1.0)
    yaw.set_command([0.0])
    run_sim(m, d, cm, 0.5, t0=1.0)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_yaw")
    assert d.qpos[m.jnt_qposadr[jid]] - y0 == pytest.approx(0.5, abs=0.05)


def test_mujoco_effort_mode_switch_restores_servo(sim):
    m, d, rm = sim
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_wrist_pitch")
    gain, bias = m.actuator_gainprm[a].copy(), m.actuator_biasprm[a].copy()
    cm = ControllerManager(rm)
    eff = cm.add(ForwardCommandController("wrist_eff", ["arm_wrist_pitch"], HW_IF_EFFORT))
    pos = cm.add(ForwardCommandController("wrist_pos", ["arm_wrist_pitch"], HW_IF_POSITION))
    cm.activate("wrist_eff")
    assert m.actuator_gainprm[a, 0] == 0.0
    dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_wrist_pitch")]
    eff.set_command([9.0])                                   # clamped to the 5 N.m forcerange
    run_sim(m, d, cm, 0.05)
    assert d.qfrc_applied[dof] == pytest.approx(5.0)
    assert rm.state("arm_wrist_pitch/effort") == pytest.approx(5.0, abs=0.5)
    cm.switch(stop=["wrist_eff"], start=["wrist_pos"])
    assert np.array_equal(m.actuator_gainprm[a], gain) and np.array_equal(m.actuator_biasprm[a], bias)
    assert d.qfrc_applied[dof] == 0.0
    q_now = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_wrist_pitch")]]
    pos.set_command([q_now])
    run_sim(m, d, cm, 0.5, t0=0.05)
    rm.transition("nandi_arm", "deactivate")
    assert rm.component_states()["nandi_arm"]["effort_mode_actuators"] == []


def _wrist(m, d):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_wrist_pitch")
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]


def test_mujoco_deactivate_velocity_holds_position(sim):
    """Safety interlock: a stopped velocity controller must not keep integrating its last command."""
    m, d, rm = sim
    cm = ControllerManager(rm)
    vel = cm.add(ForwardCommandController("wrist_vel", ["arm_wrist_pitch"], HW_IF_VELOCITY))
    cm.activate("wrist_vel")
    vel.set_command([0.5])
    t = run_sim(m, d, cm, 0.5)
    qadr, _ = _wrist(m, d)
    cm.deactivate("wrist_vel")
    q_stop = float(d.qpos[qadr])
    assert rm.claimed() == {}
    st = rm.component_states()["nandi_arm"]
    assert st["command_mode"]["arm_wrist_pitch"] == HW_IF_POSITION
    assert rm.command_value("arm_wrist_pitch/velocity") == 0.0
    assert rm.command_value("arm_wrist_pitch/position") == pytest.approx(q_stop)
    run_sim(m, d, cm, 1.0, t0=t)
    assert abs(d.qpos[qadr] - q_stop) < 0.05, (q_stop, d.qpos[qadr])
    cm.activate("wrist_vel")                                 # re-activation starts without the stale 0.5
    run_sim(m, d, cm, 0.5, t0=t + 1.0)
    assert abs(d.qpos[qadr] - q_stop) < 0.05


def test_mujoco_deactivate_effort_restores_servo_and_zero_torque(sim):
    m, d, rm = sim
    a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm_wrist_pitch")
    gain = m.actuator_gainprm[a].copy()
    cm = ControllerManager(rm)
    eff = cm.add(ForwardCommandController("wrist_eff", ["arm_wrist_pitch"], HW_IF_EFFORT))
    cm.activate("wrist_eff")
    eff.set_command([5.0])
    t = run_sim(m, d, cm, 0.05)
    qadr, dof = _wrist(m, d)
    assert d.qfrc_applied[dof] == pytest.approx(5.0)
    cm.deactivate("wrist_eff")
    assert np.array_equal(m.actuator_gainprm[a], gain) and d.qfrc_applied[dof] == 0.0
    assert rm.component_states()["nandi_arm"]["command_mode"]["arm_wrist_pitch"] == HW_IF_POSITION
    assert rm.command_value("arm_wrist_pitch/effort") == 0.0
    q_stop = float(d.qpos[qadr])
    run_sim(m, d, cm, 1.0, t0=t)
    assert d.qfrc_applied[dof] == 0.0
    assert abs(d.qpos[qadr] - q_stop) < 0.05, (q_stop, d.qpos[qadr])


def test_stub_deactivate_velocity_stops_joint():
    rm = stub_rm()
    cm = ControllerManager(rm)
    vel = cm.add(ForwardCommandController("yaw_vel", ["arm_yaw"], HW_IF_VELOCITY))
    cm.activate("yaw_vel")
    vel.set_command([0.5])
    for k in range(50):
        cm.update(k * 0.01, 0.01)
    cm.deactivate("yaw_vel")
    for k in range(20):
        cm.update(0.5 + k * 0.01, 0.01)
    y = rm.state("arm_yaw/position")
    for k in range(100):
        cm.update(0.7 + k * 0.01, 0.01)
    assert rm.state("arm_yaw/position") == pytest.approx(y, abs=0.01)
    assert abs(rm.state("arm_yaw/velocity")) < 0.01


def test_jtc_refuses_trajectory_not_starting_at_state():
    rm = stub_rm()
    cm = ControllerManager(rm)
    cm.add(JointStateBroadcaster())
    jtc = cm.add(JointTrajectoryController("arm_jtc", ARM_JOINTS))
    cm.activate("arm_jtc")
    cm.update(0.0, 0.01)
    v, a = joint_limits(DOC, ARM_JOINTS)
    q = jtc.positions()
    far = time_parameterize(np.vstack([q + [0, 0.5, 0, 0, 0], q + [0, 0.8, 0, 0, 0]]), v, a)
    with pytest.raises(HalError, match="start_tolerance"):
        jtc.follow(far)
    assert jtc.done and jtc.trajectory is None
    near = time_parameterize(np.vstack([q + [0, 0.01, 0, 0, 0], q + [0, 0.3, 0, 0, 0]]), v, a)
    jtc.follow(near)
    assert not jtc.done


# ---------------------------------------------------------------------- edge cases of the interface layer
def test_interface_edge_cases(tmp_path):
    from atharv_nandi.core.hal import interfaces as hi
    from atharv_nandi.core.hal.stub_backend import StubSystem
    from atharv_nandi.core.hal.sim_backend import MujocoSystem
    doc = {"schema": hi.SCHEMA, "hardware": [{"name": "x", "plugins": {"stub": "atharv_nandi.core.hal.stub_backend.StubSystem"},
                                              "joints": [{"name": "j1", "command_interfaces": ["position", "velocity"],
                                                          "state_interfaces": {"position": {}, "velocity": {}}}]}]}
    (info,) = parse_hardware(doc)
    assert [c.name for c in info.joint("j1").command_interfaces] == ["position", "velocity"]
    with pytest.raises(KeyError):
        info.joint("nope")
    for bad, msg in (({"schema": hi.SCHEMA, "hardware": [{"joints": []}]}, "without a name"),
                     ({"schema": hi.SCHEMA, "hardware": [{"name": "y", "joints": [{"command_interfaces": ["position"]}]}]},
                      "entry without a name")):
        with pytest.raises(HalError, match=msg):
            parse_hardware(bad)
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(HalError, match="not a mapping"):
        load_config(str(p))
    with pytest.raises(HalError, match="no limits"):
        joint_limits(doc, ["j9"])
    # base class defaults
    base = hi.SystemInterface()
    for cb in ("on_configure", "on_activate", "on_deactivate", "on_cleanup", "on_shutdown", "on_error"):
        assert getattr(base, cb)() == ReturnType.OK
    rm = build_resource_manager(doc, backend="stub")
    with pytest.raises(HalError, match="already loaded"):
        rm.add(StubSystem(), info)
    m = farm_scene.load_farm_model()
    bogus = hi.HardwareInfo("m", joints=[hi.ComponentInfo("no_such_joint", command_interfaces=[hi.InterfaceInfo("position")])])
    with pytest.raises(HalError, match="on_init failed"):
        rm.add(MujocoSystem(model=m, data=mujoco.MjData(m)), bogus)
    with pytest.raises(ValueError):
        MujocoSystem()
    rm.configure_all()
    rm.activate_all()
    c = rm.components["x"]
    assert len(c.export_state_interfaces()) == 2 and len(c.export_command_interfaces()) == 2
    assert not rm.is_available("zz/position")
    (h,) = rm.claim(["j1/velocity"], "o")
    h.set_value(float("nan"))
    assert math.isnan(rm.command_value("j1/velocity"))
    assert rm.transition("x", "shutdown") == LifecycleState.FINALIZED       # active -> deactivate -> finalized
    only = build_resource_manager(DOC, backend="stub", hardware=["nandi_legs"])
    assert list(only.components) == ["nandi_legs"]

    class Flaky(StubSystem):
        def on_activate(self):
            return ReturnType.ERROR

    rm2 = hi.ResourceManager()
    rm2.add(Flaky(), parse_hardware(doc)[0])
    rm2.transition("x", "configure")
    with pytest.raises(HalError, match="on_activate failed"):
        rm2.transition("x", "activate")
    assert rm2.components["x"].state == LifecycleState.UNCONFIGURED
    assert rm2.components["x"].prepare_command_mode_switch(["zz/position"], []) == ReturnType.ERROR
