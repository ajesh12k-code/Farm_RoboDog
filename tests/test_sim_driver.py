"""WP0b: ROS-free headless driver (atharv_nandi/core/sim_driver.py) and headless_eval on it.

Acceptance (WP0b; ORCHARD-MVP-SPEC 3.5, 6; MVP-SPEC 5.1):
  * legacy headless_eval unchanged: the driver's step loop is byte-identical (qpos) to the pre-WP0b
    headless_eval loop on seeds 7 / 11 / 23 (the full-run comparison is in the WP0B contract notes)
  * unit: walk 1 m with a stub renderer, <= 10 s wall
  * gl + slow: walk 1 m + one orbit station incl. crown move and capture (about 20 s sim); the same
    station with the stub renderer runs at unit tier
  * hooks at their declared rates; the substep callback is called before every physics step
"""
import json
import math
import os
import subprocess
import sys
import time

import mujoco
import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import farm_scene
from atharv_nandi.core import model_layout as mlay
from atharv_nandi.core import sim_driver as sd
from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch
from atharv_nandi.core.tripod_gait import TripodGaitGenerator

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORLD_ENV = ("NANDI_WORLD", "NANDI_TERRAIN", "NANDI_SEED", "NANDI_CROPS", "NANDI_MAST_H", "NANDI_MAST_HEAD_KG",
             "NANDI_DT", "NANDI_PAYLOAD_KG", "NANDI_BUND_H")


def _build(**env):
    """Compile a world with exactly these NANDI_* knobs (others unset), restoring the environment after."""
    saved = {k: os.environ.get(k) for k in WORLD_ENV}
    try:
        for k in WORLD_ENV:
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = str(v)
        return farm_scene.load_farm_model()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def flat7():
    return _build(NANDI_TERRAIN="flat", NANDI_SEED=7)


def _driver(model, **kw):
    kw.setdefault("hooks", hook_registry.HookRegistry())
    return sd.SimDriver(model, **kw)


def _upright(d) -> bool:
    roll, pitch = quat_to_roll_pitch(*d.qpos[3:7])
    return abs(roll) < math.radians(20) and abs(pitch) < math.radians(20) and d.qpos[2] > 0.15


# ---------------------------------------------------------------------- ROS-free
def test_driver_and_headless_eval_import_without_ros():
    code = ("import sys; import atharv_nandi.core.sim_driver, tools.headless_eval; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('rclpy', 'std_msgs', 'sensor_msgs')); "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


# ---------------------------------------------------------------------- legacy parity
def _legacy_steps(model, n, plan):
    """The pre-WP0b tools/headless_eval.py HeadlessWorld.step loop, verbatim in substance."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    gait = TripodGaitGenerator(0.02)
    legs, arm = list(model.key_ctrl[0][7:]), list(model.key_ctrl[0][:7])
    gait_every = max(1, int(round(0.02 / model.opt.timestep)))
    for k in range(n):
        if k in plan:
            kind, val = plan[k]
            if kind == "walk":
                gait.set_command(*val, hold=True)
            elif kind == "release":
                gait.release()
            else:
                arm = list(val)
        if k % gait_every == 0:
            roll, pitch = quat_to_roll_pitch(*d.qpos[3:7])
            legs = gait.step(roll, pitch)
        d.ctrl[:7] = arm
        d.ctrl[7:] = legs
        mujoco.mj_step(model, d)
    return d.qpos.copy()


def _driver_steps(model, n, plan):
    drv = _driver(model)
    for k in range(n):
        if k in plan:
            kind, val = plan[k]
            if kind == "walk":
                drv.gait.set_command(*val, hold=True)
            elif kind == "release":
                drv.gait.release()
            else:
                drv.command(val)
        drv.step()
    return drv.qpos()


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_step_loop_is_byte_identical_to_legacy_headless_eval(seed):
    model = _build(NANDI_TERRAIN="rough", NANDI_SEED=seed)
    arm2 = list(model.key_ctrl[0][:7])
    arm2[2] += 0.3
    plan = {0: ("walk", (1.0, 0.0, 0.2)), 700: ("arm", arm2), 900: ("walk", (0.0, 0.0, 1.0)), 1200: ("release", None)}
    a = _legacy_steps(model, 1500, plan)
    b = _driver_steps(model, 1500, plan)
    assert a.tobytes() == b.tobytes()
    assert np.linalg.norm(a[0:2] - model.key_qpos[0][0:2]) > 0.05  # the robot actually moved


# ---------------------------------------------------------------------- walk 1 m (unit, stub renderer)
def test_walk_one_metre_with_stub_renderer(flat7):
    w0 = time.perf_counter()
    drv = _driver(flat7, renderer=sd.StubRenderer(32, 24))
    drv.stand(0.5)
    calls = {"n": 0}

    def sub(m, d, t, bus):
        calls["n"] += 1

    drv.add_substep(sub)
    steps0 = drv.steps
    res = drv.walk_distance(1.0, timeout_s=25.0)
    cap = drv.capture(sd.View.look(drv.d.qpos[0:3] + [0.0, 0.0, 0.5], drv.d.qpos[0:3] + [1.0, 0.0, 0.0]))
    wall = time.perf_counter() - w0
    print(f"walk 1 m: {res}, wall {wall:.2f} s, steps {drv.steps}")
    assert res["ok"] and res["dist_m"] >= 1.0
    assert abs(res["heading_change_deg"]) < 15.0
    assert _upright(drv.d)
    assert calls["n"] == drv.steps - steps0  # substep before every physics step
    assert cap.rgb.shape == (24, 32, 3) and cap.renderer == "stub" and drv.captures == [cap]
    assert wall <= 10.0


# ---------------------------------------------------------------------- hooks, substep, rates
def test_hooks_run_at_declared_rates_and_substep_every_step(flat7):
    reg = hook_registry.HookRegistry()
    log = []
    reg.register("r50", lambda m, d, t, bus: log.append(("r50", t)))
    reg.register("r10", lambda m, d, t, bus: None, rate_hz=10)
    reg.register("r2k", lambda m, d, t, bus: None, rate_hz=2000)   # above 1/dt = 500 Hz: once per step
    reg.register("sub", lambda m, d, t, bus: log.append(("sub", t)), every_step=True)
    drv = _driver(flat7, hooks=reg)
    order = []
    drv.add_substep(lambda m, d, t, bus: order.append(("drv_sub", t, d.time)), name="drv_sub")
    drv.add_task("task5", lambda dr: order.append(("task", dr.t)), rate_hz=5)
    assert drv.rate_capped == ["r2k"]
    drv.wait(2.0)
    n = drv.steps
    assert n == 1000
    calls = drv.hook_calls()
    assert calls["sub"] == n and calls["r2k"] == n
    assert abs(calls["r50"] - 100) <= 1 and abs(calls["r10"] - 20) <= 1
    assert drv.task_calls()["task5"] == 11  # t = 0.002, 0.2, 0.4 .. 2.0 (first tick after step 1)
    assert sum(1 for o in order if o[0] == "drv_sub") == n and drv.substep_calls == n
    dt = flat7.opt.timestep
    subs = [t for k, t in log if k == "sub"]
    assert np.allclose(subs, np.arange(n) * dt)               # substep: BEFORE step k (t = k dt)
    ticks = [t for k, t in log if k == "r50"]
    assert ticks[0] == pytest.approx(dt)                       # tick: AFTER the first step
    assert np.allclose(np.diff(ticks[1:]), 0.02, atol=1e-9)   # then on the 50 Hz sim-time grid
    # order within a step: hook substep, then driver substep, then mj_step, then tick
    assert log[0] == ("sub", 0.0) and order[0][1] == 0.0


def test_substep_cannot_drive_robot_actuators_and_owns_appended_ones(flat7):
    reg = hook_registry.HookRegistry()
    seen = []

    def grab(m, d, t, bus):
        d.ctrl[:mlay.ROBOT_NU] = 0.7

    reg.register("grab", grab, every_step=True)
    drv = _driver(flat7, hooks=reg)
    drv.add_substep(lambda m, d, t, bus: seen.append(d.ctrl[:mlay.ROBOT_NU].copy()), name="look")
    drv.run_steps(5)
    arm = np.asarray(drv.arm)
    assert np.allclose(drv.d.ctrl[:7], arm) and np.allclose(drv.d.ctrl[7:25], drv.legs)
    assert np.allclose(seen[-1], 0.7)  # visible inside the substep point, overwritten before mj_step


def test_hook_fault_disables_hook_and_enters_safe_hold(flat7):
    reg = hook_registry.HookRegistry()
    n = {"bad": 0}

    def bad(m, d, t, bus):
        n["bad"] += 1
        if t > 0.11:
            raise RuntimeError("boom")

    reg.register("bad", bad, rate_hz=50)
    reg.register("watch", lambda m, d, t, bus: bus.publish("/nandi/test/hold", {"hold": bus.holding()}), rate_hz=10)
    drv = _driver(flat7, hooks=reg)
    drv.walk(1.0)
    drv.wait(0.5)
    assert "bad" in drv.hook_faults and "boom" in drv.hook_faults["bad"]
    assert drv.holding and drv.bus.values["hold"]["reason"] == hook_registry.REASON_HOOK_FAULT
    assert n["bad"] == 7  # t = 0.002, 0.02 .. 0.12: disabled after the raise
    assert not drv.gait.is_walking
    held = drv.d.ctrl[:25].copy()
    drv.wait(0.2)
    assert np.array_equal(drv.d.ctrl[:25], held)
    msgs = [p for _, topic, p in drv.published if topic == "/nandi/test/hold"]
    assert msgs[0] == '{"hold":false}' and msgs[-1] == '{"hold":true}'
    assert drv.published_count["/nandi/test/hold"] == len(msgs)
    drv.leave_hold()
    assert drv.bus.values["hold"] is None and not drv.holding
    drv.enter_hold("estop")
    drv.enter_hold("other")
    assert drv.bus.values["hold"]["reason"] == "estop"


def test_bus_inbox_and_publish_callback(flat7):
    reg = hook_registry.HookRegistry()
    reg.register("echo", lambda m, d, t, bus: bus.publish("/nandi/test/echo", bus.latest("/nandi/test/in", {})),
                 rate_hz=10, topics_in=["/nandi/test/in"])
    drv = _driver(flat7, hooks=reg)
    got = []
    drv.on_publish = lambda topic, text: got.append((topic, text))
    drv.deliver("/nandi/test/in", {"k": 1})
    drv.wait(0.2)
    assert got and got[-1] == ("/nandi/test/echo", '{"k":1}')


def test_reset_restores_hook_mutations_before_forward(flat7):
    m = flat7
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crop_item_2")
    home = m.body_pos[bid].copy()
    reg = hook_registry.HookRegistry()
    seen = {}

    def mutate(mm, d, t, bus):
        mm.body_pos[bid, 2] = home[2] + 0.5

    def restore(mm, d):
        seen["time_at_reset"] = float(d.time)
        mm.body_pos[bid] = home

    reg.register("mut", mutate, rate_hz=50, mutates_model=True, reset=restore)
    drv = _driver(m, hooks=reg)
    try:
        drv.wait(0.1)
        assert m.body_pos[bid, 2] == pytest.approx(home[2] + 0.5)
        drv.walk(1.0)
        drv.enter_hold("estop")
        drv.reset()
        assert seen["time_at_reset"] == 0.0 and drv.t == 0.0 and drv.steps == 0
        assert np.allclose(m.body_pos[bid], home)
        assert drv.hook_calls()["mut"] == 0 and not drv.gait.is_walking
        assert np.array_equal(drv.d.qpos, m.key_qpos[0])
        # a reset never re-arms (sim_runner._apply_reset): still held, at the home pose, same reason
        assert drv.holding and drv.bus.values["hold"]["reason"] == "estop"
        drv.step()
        assert np.array_equal(drv.d.ctrl[:mlay.ROBOT_NU], drv.home_ctrl[:mlay.ROBOT_NU])
    finally:
        m.body_pos[bid] = home


def test_reset_keeps_hook_fault_hold_and_faults(flat7):
    reg = hook_registry.HookRegistry()

    def bad(m, d, t, bus):
        raise RuntimeError("boom")

    reg.register("bad", bad, rate_hz=50)
    drv = _driver(flat7, hooks=reg)
    drv.wait(0.05)
    assert drv.holding and drv.bus.values["hold"]["reason"] == hook_registry.REASON_HOOK_FAULT
    since = drv.bus.values["hold"]["since"]
    drv.reset()
    assert drv.holding and drv.bus.values["hold"] == {"reason": hook_registry.REASON_HOOK_FAULT, "since": since}
    assert "bad" in drv.hook_faults
    drv.walk(1.0)
    drv.wait(0.2)
    assert np.array_equal(drv.d.ctrl[:mlay.ROBOT_NU], drv.home_ctrl[:mlay.ROBOT_NU]) and drv.holding


def test_bus_drained_without_hook_files(flat7):
    drv = _driver(flat7)  # empty HookRegistry: the normal task-layer setup
    got = []
    drv.on_publish = lambda topic, text: got.append(topic)
    drv.add_substep(lambda m, d, t, bus: bus.publish("/nandi/test/sub", 1) if d.time < 0.01 else None, name="s")
    drv.add_task("t", lambda dr: dr.bus.publish("/nandi/test/task", 2), rate_hz=10)
    drv.wait(0.5)
    assert drv.bus.outbox == []
    assert drv.published_count["/nandi/test/sub"] == 5 and drv.published_count["/nandi/test/task"] == 6
    assert len(got) == len(drv.published) == 11


def test_observer_timecap_still_drains_bus_and_faults(flat7):
    reg = hook_registry.HookRegistry()
    reg.register("pub", lambda m, d, t, bus: bus.publish("/nandi/test/p", 1), every_step=True)

    def bad(m, d, t, bus):
        if d.time > 0.009:  # the 5th step's tick
            raise RuntimeError("late")

    reg.register("bad", bad, rate_hz=1000)  # rate-capped: once per step
    drv = _driver(flat7, hooks=reg)

    def obs(dr):
        if dr.steps == 5:
            raise sd.TimeCap("observer")

    drv.add_observer(obs)
    with pytest.raises(sd.TimeCap, match="observer"):
        drv.run_steps(10)
    assert drv.steps == 5 and drv.published_count["/nandi/test/p"] == 5 and drv.bus.outbox == []
    assert "bad" in drv.hook_faults and drv.holding


def test_walk_distance_backwards_holds_heading(flat7):
    drv = _driver(flat7)
    drv.stand(0.5)
    res = drv.walk_distance(1.0, vx=-1.0, timeout_s=25.0)
    print(f"walk 1 m backwards: {res}")
    assert res["ok"] and res["dist_m"] >= 1.0
    assert abs(res["heading_change_deg"]) < 15.0
    assert _upright(drv.d)


def test_divergence_raises_and_blocks_stepping_until_reset(flat7):
    drv = _driver(flat7)
    drv.wait(0.05)
    hit = {"n": 0}

    def nan(m, d, t, bus):
        if t > 0.06 and not hit["n"]:
            hit["n"] = 1
            d.qvel[0] = float("nan")

    drv.add_substep(nan, name="nan")
    with pytest.raises(sd.Diverged):
        drv.wait(0.1)
    assert isinstance(sd.Diverged("x"), sd.TimeCap)
    assert drv.diverged is not None and drv.diverged.startswith("BADQVEL")
    steps = drv.steps
    with pytest.raises(sd.Diverged):
        drv.step()
    assert drv.steps == steps  # a teleported robot is never stepped on
    drv.reset()
    assert drv.diverged is None
    drv.wait(0.05)
    assert drv.t == pytest.approx(0.05)


def test_time_cap_is_exact_and_nonpositive_wait_does_not_step(flat7):
    drv = _driver(flat7, max_sim_s=0.05)
    with pytest.raises(sd.TimeCap, match="time_cap"):
        drv.run_until(lambda dr: False, 1.0)
    assert drv.t == pytest.approx(0.05) and drv.steps == 25
    with pytest.raises(sd.TimeCap):
        drv.wait(0.1)  # already at the cap: no step
    assert drv.steps == 25
    drv = _driver(flat7)
    drv.wait(0)
    drv.wait(-1)
    assert drv.steps == 0
    drv.wait(1e-6)  # a positive wait shorter than dt still steps once
    assert drv.steps == 1


def test_leave_hold_blends_and_drops_gait_commands_given_during_hold(flat7):
    drv = _driver(flat7)
    drv.stand(0.3)
    drv.enter_hold("estop")
    drv.add_task("c", lambda dr: dr.command([1.0] * 7), rate_hz=10)  # a task keeps commanding during the hold
    drv.walk(1.0)
    drv.wait(0.3)
    prev = drv.d.ctrl[:mlay.ROBOT_NU].copy()
    drv.leave_hold()
    dur = drv.rearm[2]
    assert drv.rearm_cfg[1] <= dur <= drv.rearm_cfg[2] and not drv.gait.is_walking
    worst = 0.0
    for _ in range(int(round(dur / drv.dt)) + 2):
        drv.step()
        cur = drv.d.ctrl[:mlay.ROBOT_NU].copy()
        worst = max(worst, float(np.max(np.abs(cur - prev))))
        prev = cur
    assert worst < 0.05  # rad per 2 ms step (an unblended re-arm jumps ~2.8 rad here)
    assert drv.rearm is None and not drv.gait.is_walking
    assert np.allclose(drv.d.ctrl[:7], 1.0) and np.allclose(drv.d.ctrl[7:25], drv.legs)


def test_headless_eval_reports_hook_fault_hold(monkeypatch, tmp_path):
    from tools import headless_eval as he
    import atharv_nandi.nodes.telemetry_bridge as tb

    def bad(m, d, t, bus):
        if t > 0.5:
            raise RuntimeError("boom")

    def load_all():  # stands in for a WP hook file that faults
        reg = hook_registry.HookRegistry()
        reg.register("bad", bad, rate_hz=10)
        return reg

    monkeypatch.setattr(hook_registry, "load_all", load_all)
    saved = {k: os.environ.get(k) for k in WORLD_ENV}
    try:
        row = he.run_one("flat", 7, 20.0, str(tmp_path), None, None, tb.load_config(he.CONFIG))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert row["end_reason"] == hook_registry.REASON_HOOK_FAULT and "boom" in row["hook_faults"]["bad"]
    assert row["sim_s"] < 2.0 and row["diverged"] is None
    run_dir = os.path.join(REPO, row["run_dir"]) if not os.path.isabs(row["run_dir"]) else row["run_dir"]
    events = [json.loads(line) for line in open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8")]
    kinds = {e["kind"]: e for e in events}
    assert kinds["hold"]["reason"] == hook_registry.REASON_HOOK_FAULT and "bad" in kinds["hold"]["hook_faults"]
    assert kinds["run_end"]["reason"] == hook_registry.REASON_HOOK_FAULT


def test_time_cap_stop_and_errors(flat7):
    drv = _driver(flat7, max_sim_s=0.1)
    with pytest.raises(sd.TimeCap):
        drv.wait(0.2)
    drv = _driver(flat7)
    drv.add_observer(lambda dr: dr.stop("custom") if dr.steps == 3 else None)
    with pytest.raises(sd.TimeCap, match="custom"):
        drv.wait(0.1)
    assert drv.steps == 50  # the wait finishes its steps, then stops
    with pytest.raises(ValueError):
        drv.add_task("x", lambda dr: None, rate_hz=0)
    drv.add_task("x", lambda dr: None)
    with pytest.raises(ValueError):
        drv.add_task("x", lambda dr: None)
    drv.add_substep(lambda *a: None, name="s")
    with pytest.raises(ValueError):
        drv.add_substep(lambda *a: None, name="s")
    drv.remove_substep("s")
    drv.remove_task("x")
    drv.add_task("boom", lambda dr: 1 / 0, rate_hz=50)
    with pytest.raises(ZeroDivisionError):  # task bugs propagate (hooks are the isolated layer)
        drv.step()
    phases = []
    drv.on_phase = phases.append
    drv.phase("approach")
    assert phases == ["approach"] and drv.phase_name == "approach"


def test_view_look_round_trip():
    v = sd.View.look([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert np.allclose(v.eye(), [1.0, 2.0, 3.0])
    assert v.elevation_deg == pytest.approx(-45.0) and v.azimuth_deg == pytest.approx(0.0)
    cam = v.mjv_camera()
    assert cam.distance == pytest.approx(math.sqrt(2.0))
    with pytest.raises(ValueError):
        sd.View.look([0, 0, 0], [0, 0, 0])


def test_sim_state_payload(flat7):
    drv = _driver(flat7)
    drv.run_steps(3)
    st = drv.sim_state()
    assert st["nq"] == flat7.nq and st["t"] == pytest.approx(3 * flat7.opt.timestep)
    assert st["world_id"] == mlay.current().world_id


def test_default_construction_loads_world_and_hook_files(monkeypatch):
    monkeypatch.setenv("NANDI_TERRAIN", "flat")
    drv = sd.SimDriver()
    # a hook file may register several hooks (platform.py registers 3), so each file adds at least one
    assert len(drv.hooks) >= len(hook_registry.hook_files())
    assert drv.ik is drv.ik  # built once
    drv.close()


# ---------------------------------------------------------------------- orbit station (crown move + capture)
CROWN_RATE_MPS = 0.2       # WP24 rate limit (ORCHARD-MVP-SPEC 2.3)
CROWN_EXT_M = 0.6
MAST_H = 0.5
STANDOFF_M = 1.5           # minimum stand-off from the trunk
ROBOT_HALF_M = 0.45        # trunk centre to the outer feet, ASSUMED clearance from the bund toe


class CrownRamp:
    """Test stand-in for the WP24 mast runtime: a substep callback writing the interpolated ``body_pos`` of
    the mast head every physics step (rate-limited), the mechanism ORCHARD-MVP-SPEC 3.5 specifies."""

    def __init__(self, model):
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mast_head")
        assert self.bid >= 0
        self.z0 = float(model.body_pos[self.bid, 2])
        self.ext, self.target = 0.0, 0.0
        self.deltas, self.calls = [], 0

    def __call__(self, m, d, t, bus):
        self.calls += 1
        step = CROWN_RATE_MPS * m.opt.timestep
        new = self.ext + max(-step, min(step, self.target - self.ext))
        self.deltas.append(abs(new - self.ext))
        self.ext = new
        m.body_pos[self.bid, 2] = self.z0 + self.ext

    def restore(self, m):
        m.body_pos[self.bid, 2] = self.z0


def _orbit_station(model, renderer):
    """Walk 1 m, go to a stand-off station of the nearest tree, face it, raise the crown, capture, lower."""
    w0 = time.perf_counter()
    drv = _driver(model, renderer=renderer)
    crown = CrownRamp(model)
    drv.add_substep(crown, name="crown")
    tilt = []
    drv.add_observer(lambda dr: tilt.append(quat_to_roll_pitch(*dr.d.qpos[3:7])) if dr.steps % 10 == 0 else None)
    try:
        drv.stand(0.5)
        walk = drv.walk_distance(1.0, timeout_s=20.0)
        x, y = drv.trunk_xy()
        tree = min(farm_scene.orchard_layout(7), key=lambda t: math.hypot(t["x"] - x, t["y"] - y))
        tx, ty = tree["x"], tree["y"]
        # Stand clear of the basin bund (the crown is not raised on a bund: WP6 gates), in the alley.
        standoff = min(abs(ty), max(STANDOFF_M, tree["basin_r"] + farm_scene.bund_half_width(tree["bund_h"])
                                    + ROBOT_HALF_M))
        station = (tx, ty - math.copysign(standoff, ty))  # alley side of the trunk
        reached = drv.go_to(station, tol_m=0.15, timeout_s=25.0)
        faced = drv.face((tx, ty), tol_deg=10.0)
        drv.stand(0.6)
        head = drv.d.xpos[crown.bid]
        target = np.array([tx, ty, farm_scene.orchard_ground_z(ty) + 1.2])
        low = drv.capture(sd.View.look(head.copy(), target, name="crown"))
        z_low = float(drv.d.xpos[crown.bid][2])
        crown.target = CROWN_EXT_M
        drv.run_until(lambda dr: crown.ext >= CROWN_EXT_M - 1e-9, timeout_s=6.0)
        drv.stand(0.3)
        z_high = float(drv.d.xpos[crown.bid][2])
        high = drv.capture(sd.View.look(drv.d.xpos[crown.bid].copy(), target, name="crown"))
        crown.target = 0.0
        drv.run_until(lambda dr: crown.ext <= 1e-9, timeout_s=6.0)
        drv.stand(0.3)
        res = {"walk": walk, "reached": reached, "faced": faced, "sim_s": drv.t, "steps": drv.steps,
               "crown_calls": crown.calls, "max_delta_mm": 1000 * max(crown.deltas), "lens_rise_m": z_high - z_low,
               "max_tilt_deg": math.degrees(max(max(abs(r), abs(p)) for r, p in tilt)),
               "standoff_m": standoff, "dist_to_tree_m": math.hypot(drv.trunk_xy()[0] - tx, drv.trunk_xy()[1] - ty),
               "upright": _upright(drv.d), "captures": (low, high), "wall_s": time.perf_counter() - w0}
    finally:
        crown.restore(model)
        drv.close()
    print({k: v for k, v in res.items() if k != "captures"})
    return res


@pytest.fixture(scope="module")
def orchard():
    return _build(NANDI_TERRAIN="orchard", NANDI_SEED=7, NANDI_MAST_H=MAST_H)


def _check_station(res):
    assert res["walk"]["ok"] and res["reached"] and res["faced"] and res["upright"]
    assert 12.0 <= res["sim_s"] <= 30.0
    assert res["crown_calls"] == res["steps"]                     # substep every physics step
    assert res["max_delta_mm"] <= 0.4 + 1e-9                     # per-step interpolated setpoint
    assert res["lens_rise_m"] == pytest.approx(CROWN_EXT_M, abs=0.02)
    assert res["max_tilt_deg"] < 12.0
    assert res["dist_to_tree_m"] == pytest.approx(res["standoff_m"], abs=0.35)
    low, high = res["captures"]
    assert high.view.eye()[2] - low.view.eye()[2] == pytest.approx(CROWN_EXT_M, abs=0.02)


def test_orbit_station_with_stub_renderer(orchard):
    res = _orbit_station(orchard, sd.StubRenderer(32, 24))
    _check_station(res)
    assert res["wall_s"] <= 10.0


def _gl_available() -> bool:
    code = "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); r = mujoco.Renderer(m, 8, 8); r.close()"
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


@pytest.mark.gl
@pytest.mark.slow
def test_orbit_station_with_mujoco_renderer(orchard):
    if not _gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    res = _orbit_station(orchard, sd.MjRenderer(orchard, 96, 72))
    _check_station(res)
    low, high = res["captures"]
    assert low.renderer == high.renderer == "mujoco" and low.rgb.shape == (72, 96, 3)
    assert low.rgb.std() > 1.0 and high.rgb.std() > 1.0           # not a blank frame
    assert np.abs(low.rgb.astype(int) - high.rgb.astype(int)).mean() > 1.0  # the crown moved the lens


# ---------------------------------------------------------------------- orbit station on the real WP24 mast
# Since WP0b landed, the robot model carries the WP1 crown mast (mast_s1..s3, crown_head, crown_cam) and WP24
# drives it from the driver substep point (``mast_step``, every_step hook). The same station as above, with the
# real runtime and its raise gates instead of the test stand-in, and the capture from the crown camera itself.
def _orbit_station_wp24(model, renderer):
    from atharv_nandi.sim.hooks import mast as hm
    from atharv_nandi.core.interfaces import MastRequest

    w0 = time.perf_counter()
    reg = hook_registry.HookRegistry()
    rt = hm.register(reg)
    drv = sd.SimDriver(model, hooks=reg, renderer=renderer)
    s1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mast_s1")
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "crown_cam")
    assert s1 >= 0 and cam >= 0
    z_s1, deltas, tilt = [], [], []

    def watch(dr):
        z_s1.append(3.0 * float(model.body_pos[s1, 2]))       # ext (3 equal sections)
        if len(z_s1) > 1:
            deltas.append(abs(z_s1[-1] - z_s1[-2]))
        if dr.steps % 10 == 0:
            tilt.append(quat_to_roll_pitch(*dr.d.qpos[3:7]))

    drv.add_observer(watch)
    view = sd.View.of_camera("crown_cam")
    try:
        drv.stand(0.5)
        walk = drv.walk_distance(1.0, timeout_s=20.0)
        x, y = drv.trunk_xy()
        tree = min(farm_scene.orchard_layout(7), key=lambda t: math.hypot(t["x"] - x, t["y"] - y))
        tx, ty = tree["x"], tree["y"]
        standoff = min(abs(ty), max(STANDOFF_M, tree["basin_r"] + farm_scene.bund_half_width(tree["bund_h"])
                                    + ROBOT_HALF_M))
        station = (tx, ty - math.copysign(standoff, ty))
        reached = drv.go_to(station, tol_m=0.15, timeout_s=25.0)
        faced = drv.face((tx, ty), tol_deg=10.0)
        drv.stand(0.6)                                       # standing gate: trunk speed <= 0.05 m/s
        low = drv.capture(view)
        dec = rt.submit(MastRequest("preset", CROWN_EXT_M, "wp0b_orbit_station", t=drv.t))
        drv.run_until(lambda dr: rt.arb.ext >= CROWN_EXT_M - 1e-9, timeout_s=6.0)
        drv.stand(0.3)
        high = drv.capture(view)
        st_high = rt.state()
        rt.submit(MastRequest("preset", 0.0, "wp0b_orbit_stow", t=drv.t))
        drv.run_until(lambda dr: rt.arb.ext <= 1e-9, timeout_s=6.0)
        drv.stand(0.3)
        res = {"walk": walk, "reached": reached, "faced": faced, "raise_ok": bool(dec),
               "raise_reason": getattr(dec, "reason", ""), "sim_s": drv.t, "steps": drv.steps,
               "mast_step_calls": drv.hook_calls().get("mast_step"), "mast_calls": drv.hook_calls().get("mast"),
               "max_delta_mm": 1000 * max(deltas), "ext_high": st_high["ext_m"], "ext_end": rt.arb.ext,
               "lens_rise_m": float(high.eye[2] - low.eye[2]), "strikes": st_high["strike_count"],
               "max_tilt_deg": math.degrees(max(max(abs(r), abs(p)) for r, p in tilt)),
               "upright": _upright(drv.d), "hook_faults": dict(drv.hook_faults), "holding": drv.holding,
               "captures": (low, high), "wall_s": time.perf_counter() - w0}
    finally:
        drv.reset()                                          # hooks reset: mast stowed in the shared model
        drv.close()
    print({k: v for k, v in res.items() if k != "captures"})
    return res


@pytest.fixture(scope="module")
def orchard_wp1():
    return _build(NANDI_TERRAIN="orchard", NANDI_SEED=7)


def _check_station_wp24(res):
    assert res["walk"]["ok"] and res["reached"] and res["faced"] and res["upright"]
    assert res["raise_ok"], res["raise_reason"]
    assert not res["hook_faults"] and not res["holding"] and res["strikes"] == 0
    assert 12.0 <= res["sim_s"] <= 30.0
    assert res["mast_step_calls"] == res["steps"]                 # WP24 substep before every physics step
    assert res["mast_calls"] == pytest.approx(res["sim_s"] * 50, abs=2)   # 50 Hz rate hook on the sim grid
    assert res["max_delta_mm"] <= 0.4 + 1e-6                      # per-step interpolated (0.2 m/s x 2 ms)
    assert res["ext_high"] == pytest.approx(CROWN_EXT_M, abs=1e-6) and res["ext_end"] == 0.0
    assert res["lens_rise_m"] == pytest.approx(CROWN_EXT_M, abs=0.03)    # crown_cam rode up with the head
    assert res["max_tilt_deg"] < 12.0


def test_orbit_station_on_wp24_mast_with_stub_renderer(orchard_wp1):
    res = _orbit_station_wp24(orchard_wp1, sd.StubRenderer(32, 24))
    _check_station_wp24(res)
    low, high = res["captures"]
    assert low.view.camera == "crown_cam" and low.renderer == "stub"
    assert res["wall_s"] <= 10.0


@pytest.mark.gl
@pytest.mark.slow
def test_orbit_station_on_wp24_mast_with_mujoco_renderer(orchard_wp1):
    if not _gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    res = _orbit_station_wp24(orchard_wp1, sd.MjRenderer(orchard_wp1, 96, 72))
    _check_station_wp24(res)
    low, high = res["captures"]
    assert low.renderer == high.renderer == "mujoco" and low.rgb.shape == (72, 96, 3)
    assert low.rgb.std() > 1.0 and high.rgb.std() > 1.0           # not a blank frame
    assert np.abs(low.rgb.astype(int) - high.rgb.astype(int)).mean() > 1.0  # the crown camera view changed


def test_model_camera_view(flat7):
    drv = _driver(flat7)
    try:
        drv.stand(0.1)
        cap = drv.capture(sd.View.of_camera("crown_cam"))
        cid = mujoco.mj_name2id(flat7, mujoco.mjtObj.mjOBJ_CAMERA, "crown_cam")
        assert np.allclose(cap.eye, drv.d.cam_xpos[cid]) and cap.view.name == "crown_cam"
        free = drv.capture(sd.View.look((0, 0, 2), (1, 0, 0)))
        assert np.allclose(free.eye, (0, 0, 2))
        with pytest.raises(ValueError):
            drv.capture(sd.View.of_camera("no_such_cam"))
        with pytest.raises(ValueError):
            sd.View.of_camera("crown_cam").eye()
        with pytest.raises(ValueError):
            sd.View.of_camera("crown_cam").mjv_camera()
        with pytest.raises(ValueError):
            sd.View.of_camera("")
        mc = sd.View.of_camera("crown_cam").mjv_camera(flat7)
        assert mc.type == mujoco.mjtCamera.mjCAMERA_FIXED and mc.fixedcamid == cid
    finally:
        drv.close()
