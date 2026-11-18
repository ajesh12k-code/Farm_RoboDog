"""WP7 platform: battery integration and schema, GNSS states / dropouts / RTK rule, IMU noise, pose estimate,
link heartbeats, and the platform hooks driven on the real model (ROS-free, headless, each <= 10 s)."""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

import mujoco  # noqa: E402

import atharv_nandi.sim.hooks as hook_registry  # noqa: E402
from atharv_nandi.core import energy_model as em  # noqa: E402
from atharv_nandi.core import geo_frame as gf  # noqa: E402
from atharv_nandi.core import gnss_model as gm  # noqa: E402
from atharv_nandi.core import imu_model as im  # noqa: E402
from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch  # noqa: E402
from atharv_nandi.core.tripod_gait import TripodGaitGenerator  # noqa: E402
from atharv_nandi.sim.hooks import platform as hp  # noqa: E402

from tools.physics import measure_baseline as mb  # noqa: E402

ORIGIN = gf.SiteOrigin(12.5, -30.0, 0.0)


# ------------------------------------------------------------------ harness
@pytest.fixture(scope="module")
def model():
    return mb.build(NANDI_TERRAIN="flat", NANDI_CROPS=1)


class Run:
    """Real model + gait + the WP7 hooks only (substep -> mj_step -> tick, like sim_runner)."""

    def __init__(self, m, hooks=True):
        self.m = m
        self.d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, self.d, 0)
        hp._STATE.pop(id(m), None)
        self.reg = hook_registry.HookRegistry()
        if hooks:
            hp.register(self.reg)
        self.reg.reset(m, self.d)
        mujoco.mj_forward(m, self.d)
        self.bus = hook_registry.Bus()
        self.gait = TripodGaitGenerator(mb.CTRL_DT)
        self.sub = int(round(mb.CTRL_DT / m.opt.timestep))
        self.out = []
        self.on_tick = None

    @property
    def p(self) -> hp.Platform:
        return hp._STATE[id(self.m)]

    def run(self, seconds, vx=0.0):
        if vx:
            self.gait.set_command(vx, 0.0, 0.0, hold=True)
        else:
            self.gait.set_command(0.0, 0.0, 0.0, hold=False)
        for _ in range(int(round(seconds / mb.CTRL_DT))):
            r, p = quat_to_roll_pitch(*self.d.qpos[3:7])
            self.d.ctrl[mb.LEG_CTRL] = self.gait.step(r, p)
            for _ in range(self.sub):
                self.reg.substep(self.m, self.d, self.bus)
                mujoco.mj_step(self.m, self.d)
                self.reg.tick(self.m, self.d, self.bus)
            if self.on_tick:
                self.on_tick(self)
            self.out.extend(self.bus.drain())
        assert self.reg.faulted() == {}

    def msgs(self, topic):
        import json
        return [json.loads(p) for t, p in self.out if t == topic]


# ------------------------------------------------------------------ battery
def test_constant_power_integration_is_analytic():
    """The every-step energy integral and Battery.consume equal P t / 3600 for a constant mechanical power."""
    params = em.EnergyParams()
    m = SimpleNamespace(nu=4, opt=SimpleNamespace(timestep=0.002))
    d = SimpleNamespace(actuator_force=np.array([2.0, -3.0, 0.5, 0.0]),
                        actuator_velocity=np.array([5.0, 2.0, -4.0, 9.0]))
    p_mech = 10.0 + 6.0 + 2.0
    p_el = p_mech / params.eta + params.p_idle_w
    fake = SimpleNamespace(params=params, e_step_wh=0.0, p_last_w=0.0)
    for _ in range(50 * 3600):   # 180,000 physics steps of 2 ms = 360 s
        hp.Platform.energy_step(fake, m, d)
    t = 50 * 3600 * 0.002
    assert fake.e_step_wh == pytest.approx(p_el * t / 3600.0, rel=1e-9)
    b = em.Battery(params)
    for k in range(1, 361):
        b.consume(float(k), p_el / 3600.0, p_el)
    assert b.energy_wh == pytest.approx(params.capacity_wh - p_el * 360 / 3600.0, rel=1e-12)
    assert b.est_remaining_min() > 0


def test_soc_never_rises():
    b = em.Battery(em.EnergyParams())
    socs = [b.soc]
    for k, e in enumerate([0.1, -5.0, float("nan"), 0.0, 2.0, float("inf"), 0.3]):
        socs.append(b.consume(float(k), e, 50.0))
    assert all(s2 <= s1 for s1, s2 in zip(socs, socs[1:]))
    assert b.energy_wh == pytest.approx(em.EnergyParams().capacity_wh - 2.4)
    b.consume(10.0, 1e9)
    assert b.soc == 0.0


def test_battery_state_schema_and_electrics():
    p = em.EnergyParams()
    b = em.Battery(p)
    st = b.status()
    assert em.validate_battery_state(st) == []
    assert st["power_supply_status"] == em.POWER_SUPPLY_STATUS_FULL
    assert st["projected_end_soc"] is None and st["reserve"] is None
    b.consume(1.0, 100.0, 200.0)
    b.set_projection(0.41, 0.19)
    st = b.status()
    assert em.validate_battery_state(st) == []
    assert st["power_supply_status"] == em.POWER_SUPPLY_STATUS_DISCHARGING
    assert (st["projected_end_soc"], st["reserve"]) == (0.41, 0.19)
    # 8S LFP: plateau 26.2-26.6 V at 80 % SoC; P = V I within 0.1 %
    assert 26.0 < st["voltage"] < 27.0
    assert -st["current"] * st["voltage"] == pytest.approx(200.0, rel=1e-3)
    assert st["capacity"] == pytest.approx(500.0 / 25.6, abs=1e-4)
    assert st["charge"] == pytest.approx(st["capacity"] * 0.8, rel=1e-4)
    assert len(st["cell_voltage"]) == 8
    # the legacy keys stay
    assert set(st) >= {"soc", "power_w", "energy_wh", "est_remaining_min", "source", "assumed"}
    # schema violations are caught
    bad = dict(st, percentage=1.5)
    assert "percentage outside [0, 1]" in em.validate_battery_state(bad)
    assert em.validate_battery_state({k: v for k, v in st.items() if k != "voltage"}) == ["missing voltage"]
    assert em.validate_battery_state(dict(st, current=3.0))
    assert em.validate_battery_state(dict(st, power_supply_status=True))
    assert em.validate_battery_state("x") == ["not an object"]
    # saturation of the Thevenin solve and the r = 0 case
    assert em.pack_current_a(1e9, 25.6, 0.05) == pytest.approx(25.6 / 0.1)
    assert em.pack_current_a(256.0, 25.6, 0.0) == pytest.approx(10.0)
    assert em.lfp_ocv_cell_v(0.0) == 2.5 and em.lfp_ocv_cell_v(1.0) == 3.45 and em.lfp_ocv_cell_v(2.0) == 3.45


# ------------------------------------------------------------------ GNSS
def test_gnss_states_sigmas_and_navsatfix_schema():
    s = gm.GnssScript(dropouts=[gm.Dropout(20.0, 10.0, gm.NONE), gm.Dropout(40.0, 5.0, gm.FLOAT)])
    assert [s.state_at(t) for t in (0.0, 4.9, 5.0, 19.9, 20.0, 29.9, 30.0, 34.9, 35.0, 41.0, 46.0, 50.1)] == \
        ["FLOAT", "FLOAT", "FIXED", "FIXED", "NONE", "NONE", "FLOAT", "FLOAT", "FIXED", "FLOAT", "FLOAT", "FIXED"]
    g = gm.GnssModel(ORIGIN, gm.GnssScript(start_state=gm.FIXED), seed=3)
    errs = []
    for k in range(3000):
        f = g.sample(k * 0.1, 1.0, 2.0, 0.5)
        assert gm.validate_navsatfix(f) == []
        x, y, _ = ORIGIN.to_local(f["latitude"], f["longitude"], f["altitude"])
        errs.append((x - 1.0, y - 2.0))
    sd = np.std(np.array(errs), axis=0)
    assert np.all(np.abs(sd - 0.02) < 0.002)          # FIXED 0.02 m per axis
    gf_ = gm.GnssModel(ORIGIN, gm.GnssScript(start_state=gm.NONE, dropouts=[gm.Dropout(0.0, 1e6, gm.FLOAT)]))
    e = [gf_.sample(0.0, 0, 0, 0)["local"][0] for _ in range(3000)]
    assert abs(np.std(e) - 0.5) < 0.05                 # FLOAT 0.5 m
    none = gm.GnssModel(ORIGIN, gm.GnssScript(start_state=gm.NONE)).sample(1.0, 0, 0, 0)
    assert none["status"]["status"] == gm.NAVSAT_NO_FIX and none["latitude"] is None
    assert gm.validate_navsatfix(none) == []
    assert gm.validate_navsatfix({"rtk": "X"})
    assert gm.GnssScript(start_state=gm.FIXED).state_at(0.0) == gm.FIXED
    assert gm.GnssScript.from_dict({"dropouts": [{"t0": 1, "dur": 2, "state": "float"}]}).state_at(1.5) == gm.FLOAT
    with pytest.raises(ValueError):
        gm.Dropout(0.0, 1.0, gm.FIXED)
    with pytest.raises(ValueError):
        gm.GnssScript.from_dict({"start_state": "rtk"})


def test_rtk_start_gate_and_mid_row_rule():
    assert gm.rtk_start_gate(gm.FIXED, "scan") == (True, "")
    assert gm.rtk_start_gate(gm.FLOAT, "scan") == (False, "rtk")
    assert gm.rtk_start_gate(gm.NONE, "tree360") == (False, "rtk")
    assert gm.rtk_start_gate(gm.FLOAT, "return_home") == (True, "")
    assert gm.rtk_start_gate(gm.NONE, "return_home") == (False, "rtk")
    sup = gm.RtkSupervisor()
    assert sup.update(0.0, gm.FLOAT) == gm.A_CONTINUE          # not armed before the first FIXED
    sup.waypoint((3.0, 0.0), gm.FIXED)
    sup.waypoint((9.0, 9.0), gm.FLOAT)                           # a FLOAT waypoint is not a return target
    assert sup.update(1.0, gm.FIXED) == gm.A_CONTINUE
    assert sup.update(2.0, gm.NONE, item_active=True) == gm.A_FINISH_ITEM
    assert sup.update(5.0, gm.NONE, item_active=True) == gm.A_FINISH_ITEM
    assert sup.update(6.0, gm.NONE, item_active=False) == gm.A_PAUSE
    assert sup.update(35.9, gm.FLOAT) == gm.A_PAUSE
    assert sup.update(36.0, gm.FLOAT) == gm.A_RETURN_ODOM       # 30 s after the pause
    assert sup.return_target == (3.0, 0.0)
    st = sup.status(40.0)
    assert st["reason"] == "rtk" and st["waited_s"] == 34.0
    assert sup.update(41.0, gm.FIXED) == gm.A_CONTINUE
    assert [e["event"] for e in sup.events] == ["rtk_lost", "finish_item", "pause", "return_odometry", "rtk_recovered"]
    sup2 = gm.RtkSupervisor()
    sup2.update(0.0, gm.FIXED)
    sup2.observe((1.5, 2.5), gm.FIXED)
    assert sup2.return_target == (1.5, 2.5)


def test_pose_estimator_drift_and_fix():
    pe = gm.PoseEstimator(seed=11)
    pe.predict((0.0, 0.0))
    pe.correct((0.0, 0.0), 0.02, gm.FIXED)
    for k in range(1, 501):                       # 10 m along x on odometry only
        pe.predict((k * 0.02, 0.0))
    pe.correct(None, None, gm.NONE)
    err = math.hypot(pe.x - 10.0, pe.y)
    est = pe.estimate(1.0, 0.0)
    assert est["source"] == "odometry" and est["dist_since_fix_m"] == pytest.approx(10.0)
    assert 0.0 < err < 0.6                        # 2 % scale + 1 deg heading (1 sigma) -> ~0.2-0.3 m
    k = math.hypot(0.02, math.radians(1.0))
    assert est["sigma_xy_m"] == pytest.approx(math.sqrt(0.02 ** 2 + (k * 10.0) ** 2), abs=2e-4)
    assert err < 2.0 * est["sigma_xy_m"]                 # the reported sigma covers the real drift
    pe.correct((10.0, 0.0), 0.02, gm.FIXED)       # a FIXED fix pulls it back
    assert math.hypot(pe.x - 10.0, pe.y) < 0.02 * err + 0.01
    assert pe.estimate(2.0, 0.0)["dist_since_fix_m"] == 0.0


# ------------------------------------------------------------------ IMU
def test_imu_noise_layer_statistics_and_ideal_mode():
    q = (0.9238795, 0.0, 0.0, 0.3826834)
    imu = im.ImuModel(seed=5)
    g, a, ang = [], [], []
    for k in range(4000):
        s = imu.sample(k * 0.02, q, (0.1, 0.0, 0.0), (0.0, 0.0, 9.81))
        assert k > 0 or im.validate_imu(s) == []
        g.append(s["angular_velocity"]["x"])
        a.append(s["linear_acceleration"]["z"])
        o = s["orientation"]
        ang.append(im.quat_angle_deg(q, (o["w"], o["x"], o["y"], o["z"])))
    assert abs(np.std(g) - 0.01) < 0.002 and abs(np.mean(g) - 0.1) < 0.005
    assert abs(np.std(a) - 0.05) < 0.01
    assert 0.1 < np.mean(ang) < 0.6                # 0.2 deg per axis -> ~0.32 deg mean angle
    assert any(abs(b) > 0 for b in imu.gyro_bias)  # bias random walk moved
    ideal = im.ImuModel(enabled=False).sample(0.0, q, (0.1, 0.2, 0.3), (1, 2, 3))
    assert ideal["noise"] is False and ideal["angular_velocity"] == {"x": 0.1, "y": 0.2, "z": 0.3}
    assert im.validate_imu(ideal) == []
    assert im.validate_imu({"header": {}}) != []
    with pytest.raises(ValueError):
        im.ImuNoise(gyro_sigma=-1.0)


# ------------------------------------------------------------------ links
def test_link_monitor_timeouts_and_confirm():
    lk = hp.LinkMonitor("local", 0.5, "stop_hold", "local_link_lost", "local_link_lost_hold")
    assert lk.update(10.0) is None and lk.state == "unknown"     # no heartbeat source yet
    lk.beat(0.0)
    assert lk.update(0.3) is None and lk.state == "up"
    assert lk.update(0.5) is None                                # not > 0.5 s
    assert lk.update(0.52) == "lost"
    st = lk.status(0.52)
    assert (st["action"], st["reason"], st["needs_confirm"]) == ("stop_hold", "local_link_lost", True)
    lk.beat(1.0)
    assert lk.update(1.0) == "restored" and lk.needs_confirm
    lk.ack()
    assert not lk.status(1.0)["needs_confirm"]
    lk.drop(2.0, 1.0)
    lk.beat(2.5)                                                  # injected loss: beat ignored
    assert lk.update(2.6) == "lost" and lk.status(2.6)["injected_drop"]


def test_hook_link_loss_local_and_cloud(model):
    """Heartbeats at 10 Hz (local) and 1 Hz (cloud); then both stop: local lost <= 0.52 s, cloud <= 5.02 s."""
    r = Run(model)
    beats = {"on": True, "n": 0}

    def hb(run):
        if beats["on"]:
            beats["n"] += 1
            run.bus.deliver(hp.TOPIC_HB["local"], {"seq": beats["n"]})
            if beats["n"] % 10 == 0:
                run.bus.deliver(hp.TOPIC_HB["cloud"], {"seq": beats["n"]})

    r.on_tick = hb
    r.run(3.0)
    assert r.bus.values["link"]["local"]["state"] == "up" and r.bus.values["link"]["cloud"]["state"] == "up"
    beats["on"] = False
    t_stop = r.d.time
    r.run(6.0)
    alerts = {a["id"]: a for a in r.msgs("/nandi/platform/alert")}
    assert alerts["local_link_lost_hold"]["action"] == "stop_hold"
    assert alerts["cloud_link_lost_hold"]["action"] == "safe_park_hold"
    # last beat was at the last delivery before t_stop
    # a beat delivered after a tick is registered at the next 50 Hz tick (+0.02 s); detection <= 1 tick late
    t_beat = t_stop + 0.02
    assert 0.5 < alerts["local_link_lost_hold"]["t"] - t_beat <= 0.5 + 0.021
    assert 5.0 - 1.0 < alerts["cloud_link_lost_hold"]["t"] - t_beat <= 5.0 + 0.021
    states = r.msgs("/nandi/link/local")
    assert states[-1]["state"] == "lost" and states[-1]["needs_confirm"]


# ------------------------------------------------------------------ hooks on the real model
def test_hooks_do_not_change_legacy_physics_when_calm(model):
    a, b = Run(model, hooks=True), Run(model, hooks=False)
    a.run(4.0, vx=0.5)
    qa = a.d.qpos.copy()
    b.run(4.0, vx=0.5)
    assert np.array_equal(qa, b.d.qpos)            # byte-identical: calm wind writes no xfrc
    assert not np.any(a.d.xfrc_applied)


def test_hooks_publish_rates_battery_and_pose(model):
    r = Run(model)
    r.run(2.0)
    r.run(8.0, vx=0.5)
    sim_t = r.d.time
    counts = {tp: len(r.msgs(tp)) for tp in ("/nandi/imu", "/nandi/gnss/fix", "/nandi/pose_estimate",
                                             "/nandi/platform/battery", "/nandi/energy", "/nandi/wind",
                                             "/nandi/weather", "/nandi/link/local")}
    assert abs(counts["/nandi/imu"] - 50 * sim_t) <= 2
    assert abs(counts["/nandi/gnss/fix"] - 10 * sim_t) <= 2 and abs(counts["/nandi/pose_estimate"] - 10 * sim_t) <= 2
    assert abs(counts["/nandi/platform/battery"] - sim_t) <= 1 and abs(counts["/nandi/weather"] - sim_t) <= 1
    bats = r.msgs("/nandi/platform/battery")
    assert all(em.validate_battery_state(b) == [] for b in bats)
    socs = [b["soc"] for b in bats]
    assert all(s2 <= s1 for s1, s2 in zip(socs, socs[1:])) and socs[-1] < 1.0
    # battery drain == integrated energy (same number), and > idle-only
    e = r.msgs("/nandi/energy")[-1]
    assert r.p.battery.energy_wh == pytest.approx(500.0 - r.p.e_total_wh, abs=1e-9)
    assert r.p.e_total_wh > 15.0 * sim_t / 3600.0 * 1.5
    assert e["route_m"] > 0.8 and bats[-1]["reserve"] is not None and bats[-1]["reserve"] >= 0.15
    fixes = r.msgs("/nandi/gnss/fix")
    assert fixes[0]["rtk"] == "FLOAT" and fixes[-1]["rtk"] == "FIXED"       # badge FLOAT -> FIXED
    pose = r.msgs("/nandi/pose_estimate")[-1]
    assert pose["source"] == "rtk_fixed"
    assert math.hypot(pose["x"] - r.d.qpos[0], pose["y"] - r.d.qpos[1]) < 0.06
    assert all(im.validate_imu(m) == [] for m in r.msgs("/nandi/imu")[:20])
    assert r.bus.values["platform"]["rtk"] == "FIXED"


def test_hook_gnss_dropout_pause_wait_return_and_rtk_rejection(model):
    """FIXED -> NONE mid-row: rtk_lost alert, start gate rejects (rtk), pause, after 30 s return on odometry."""
    r = Run(model)
    r.run(6.0, vx=0.5)
    assert r.p.rtk_state == "FIXED"
    r.bus.deliver("/nandi/gnss/cmd", {"dropout": {"dur": 40.0, "state": "NONE"}})
    r.run(1.0, vx=0.5)
    assert gm.rtk_start_gate(r.bus.values["platform"]["rtk"], "scan") == (False, "rtk")
    assert r.bus.values["platform"]["rtk_action"] == gm.A_PAUSE
    r.run(31.0)
    ids = [a["id"] for a in r.msgs("/nandi/platform/alert")]
    assert "rtk_lost" in ids and "rtk_return_odometry" in ids
    ret = next(a for a in r.msgs("/nandi/platform/alert") if a["id"] == "rtk_return_odometry")
    lost = next(a for a in r.msgs("/nandi/platform/alert") if a["id"] == "rtk_lost")
    assert ret["t"] - lost["t"] == pytest.approx(30.0, abs=0.05) and ret["target"] is not None
    assert r.msgs("/nandi/pose_estimate")[-1]["source"] == "odometry"
    r.run(16.0)                                   # dropout over + 5 s convergence -> FIXED again
    assert r.p.rtk_state == "FIXED" and r.bus.values["platform"]["rtk_action"] == gm.A_CONTINUE
    assert "rtk_recovered" in [a["id"] for a in r.msgs("/nandi/platform/alert")]


def test_hook_reset_restores_full_battery(model):
    r = Run(model)
    r.run(3.0, vx=0.5)
    assert r.p.battery.soc < 1.0
    mujoco.mj_resetDataKeyframe(model, r.d, 0)
    r.reg.reset(model, r.d)
    assert r.p.battery.soc == 1.0 and r.p.e_total_wh == 0.0 and r.p.route_m == 0.0


def test_env_json_reads_text_and_files(tmp_path, monkeypatch):
    monkeypatch.delenv("NANDI_X", raising=False)
    assert hp._env_json("NANDI_X") is None
    monkeypatch.setenv("NANDI_X", '{"a": 1}')
    assert hp._env_json("NANDI_X") == {"a": 1}
    f = tmp_path / "w.json"
    f.write_text('{"speed_mps": 3}')
    monkeypatch.setenv("NANDI_X", str(f))
    assert hp._env_json("NANDI_X") == {"speed_mps": 3}


# ------------------------------------------------------------------ review fixes (WP7 fixer)
BAD_COMMANDS = [
    ("/nandi/platform/plan", {"route_m": 10, "items": "3.5"}),
    ("/nandi/platform/plan", {"route_m": float("nan")}),
    ("/nandi/platform/plan", {"waypoint": ["a", 1]}),
    ("/nandi/link/cmd", {"drop": {"link": "local", "dur": "x"}}),
    ("/nandi/link/cmd", {"drop": "local"}),
    ("/nandi/link/cmd", {"ack": "wifi"}),
    ("/nandi/wind/cmd", {"gusts": [{"t0": 0}]}),
    ("/nandi/wind/cmd", {"speed_mps": -1}),
    ("/nandi/wind/cmd", {"gusts": "abc"}),
    ("/nandi/wind/cmd", {"speed_mps": 6, "dir_deg": float("nan")}),
    ("/nandi/wind/cmd", {"speed_mps": 8, "turb_sigma_mps": 1, "turb_tau_s": 0}),
    ("/nandi/wind/cmd", {"shear_alpha": 0.2, "z_ref_m": 0}),
    ("/nandi/weather/cmd", {"rain": {"dur": "long"}}),
    ("/nandi/weather/cmd", {"temp_c": "hot"}),
    ("/nandi/gnss/cmd", {"dropout": {"dur": "x"}}),
    ("/nandi/gnss/cmd", {"dropout": {"dur": 5, "state": "RTK"}}),
    ("/nandi/gnss/cmd", {"dropout": {"dur": -1}}),
    ("/nandi/platform/plan", "not an object"),
]


def test_malformed_commands_are_rejected_and_watchdogs_keep_running(model):
    """Every bad command is rejected (alert cmd_rejected, no hook fault); the local link is still detected lost,
    and bus.values stay current (stamped with the sim time)."""
    r = Run(model)
    beats = {"on": True}
    r.on_tick = lambda run: beats["on"] and run.bus.deliver(hp.TOPIC_HB["local"], {})
    r.run(1.0)
    wind0 = r.p.wind_script
    for topic, payload in BAD_COMMANDS:
        r.bus.deliver(topic, payload)
        r.run(0.04)                                   # Run.run asserts reg.faulted() == {}
    rejected = [a for a in r.msgs("/nandi/platform/alert") if a["id"] == "cmd_rejected"]
    assert len(rejected) == len(BAD_COMMANDS)
    assert r.p.wind_script is wind0 and r.p.plan is None and not np.any(r.d.xfrc_applied)
    assert r.bus.values["link"]["local"]["state"] == "up"
    beats["on"] = False
    r.run(1.0)
    assert r.reg.faulted() == {}
    assert r.bus.values["link"]["local"]["state"] == "lost"
    assert r.bus.values["link"]["local"]["t"] == pytest.approx(r.d.time, abs=0.021)
    assert r.bus.values["platform"]["t"] == pytest.approx(r.d.time, abs=0.021)


@pytest.mark.parametrize("payload", [1, True, None, {}, "hb", 0, False])
def test_any_heartbeat_payload_counts_as_a_beat(model, payload):
    r = Run(model)
    n = {"k": 0}

    def hb(run):
        n["k"] += 1
        if n["k"] % 5 == 0:                           # every 0.1 s
            run.bus.deliver(hp.TOPIC_HB["local"], payload)
    r.on_tick = hb
    r.run(3.0)
    st = r.bus.values["link"]["local"]
    assert st["state"] == "up" and st["losses"] == 0
    assert not [a for a in r.msgs("/nandi/platform/alert") if a["id"] == "local_link_lost_hold"]


def test_bus_arrival_count():
    b = hook_registry.Bus()
    assert b.arrival_count("/nandi/x") == 0
    b.deliver("/nandi/x", 1)
    b.deliver("/nandi/x", 1)
    assert b.arrival_count("/nandi/x") == 2 and b.latest("/nandi/x") == 1


def test_link_restore_awaits_confirm_until_ack():
    lk = hp.LinkMonitor("local", 0.5, "stop_hold", "local_link_lost", "local_link_lost_hold")
    lk.beat(0.0)
    lk.update(0.1)
    assert lk.update(0.7) == "lost" and lk.status(0.7)["action"] == "stop_hold"
    lk.beat(1.0)
    assert lk.update(1.0) == "restored"
    st = lk.status(1.0)
    assert (st["state"], st["action"], st["needs_confirm"], st["reason"]) == \
        ("up", "await_confirm", True, "local_link_lost")
    lk.ack()
    assert lk.status(1.0)["action"] == "none" and lk.status(1.0)["reason"] == ""


def test_link_wall_gate_ignores_fast_sim_and_still_detects():
    """clock sim_and_wall: at RTF 6 a 1 Hz wall heartbeat (6 sim-s apart) is not a loss; a real stop is."""
    lk = hp.LinkMonitor("cloud", 5.0, "safe_park_hold", "cloud_link_lost", "cloud_link_lost_hold", "sim_and_wall")
    plain = hp.LinkMonitor("cloud", 5.0, "safe_park_hold", "cloud_link_lost", "cloud_link_lost_hold")
    lost = {"gate": 0, "plain": 0}
    for k in range(1000):                              # 20 sim-s at 50 Hz, wall = sim / 6
        t = k * 0.02
        w = t / 6.0
        if k % 300 == 0:
            lk.beat(t, w)
            plain.beat(t, w)
        lost["gate"] += lk.update(t, w) == "lost"
        lost["plain"] += plain.update(t, w) == "lost"
    assert lost == {"gate": 0, "plain": 3}
    t_last = 18.0
    assert lk.update(t_last + 40.0, t_last / 6.0 + 5.1) == "lost"   # beats stopped: wall age > 5 s
    with pytest.raises(ValueError):
        hp.LinkMonitor("x", 1.0, "a", "b", "c", clock="wall")


def test_no_plan_low_soc_triggers_return_and_dock_charges(model):
    r = Run(model)
    r.run(6.0, vx=0.5)                                 # away from home, no plan
    p = r.p
    assert math.hypot(r.d.qpos[0] - p.home[0], r.d.qpos[1] - p.home[1]) > hp.DOCK_RADIUS_M
    p.battery.energy_wh = 0.01 * p.params.capacity_wh
    r.run(3.0)
    alerts = [a for a in r.msgs("/nandi/platform/alert") if a["id"] == "low_battery_return"]
    assert len(alerts) == 1 and alerts[0]["plan"] is False
    assert r.bus.values["platform"]["energy_action"] == "return_and_wait" and not p.docked
    # parked at home with no plan for DOCK_DWELL_S -> charger: full, latch cleared (same rule as mission_executor)
    h = Run(model)
    h.run(1.0)
    h.p.battery.energy_wh = 0.5 * h.p.params.capacity_wh
    h.run(hp.DOCK_DWELL_S + 0.5)
    assert h.p.docked and h.p.charges == 1 and h.p.battery.soc == pytest.approx(1.0)
    assert [a["id"] for a in h.msgs("/nandi/platform/alert")].count("docked_charged") == 1
    assert h.msgs("/nandi/platform/battery")[-1]["docked"] is True
    h.run(2.0)
    assert h.p.charges == 1 and h.p.battery.soc == pytest.approx(1.0)


def test_wind_cmd_does_not_mutate_bus_payload_and_reset_drops_injections(model):
    r = Run(model)
    r.run(0.2)
    payload = {"speed_mps": 2.0, "gusts": [{"t0": 0.0, "peak_mps": 8.0}]}
    r.bus.deliver("/nandi/wind/cmd", payload)
    r.bus.deliver("/nandi/weather/cmd", {"temp_c": 45.0})
    r.run(0.1)
    assert payload["gusts"][0]["t0"] == 0.0
    assert r.p.wind_script.gusts[0].t0 == pytest.approx(0.22, abs=0.021)
    assert r.p.weather.script.temp_c == 45.0
    mujoco.mj_resetDataKeyframe(model, r.d, 0)
    r.reg.reset(model, r.d)
    assert r.p.wind_script.calm and r.p.weather.script.temp_c is None


def test_wind_fault_removes_own_drag(model):
    r = Run(model)
    r.bus.deliver("/nandi/wind/cmd", {"speed_mps": 10.0})
    r.run(0.5)
    tr = r.p.trunk
    assert abs(r.d.xfrc_applied[tr, 1]) > 1.0

    def boom(*a, **k):
        raise RuntimeError("boom")
    r.p.anemo.update = boom
    for _ in range(30):
        mujoco.mj_step(model, r.d)
        r.reg.tick(model, r.d, r.bus)
    assert "wind" in r.reg.faulted() and "platform" not in r.reg.faulted()
    assert np.allclose(r.d.xfrc_applied, 0.0)


def test_wind_bookkeeping_survives_time_going_back(model):
    """MuJoCo's own auto-reset zeroes xfrc_applied and time without hooks.reset: the drag is rebuilt, not ~0."""
    r = Run(model)
    r.bus.deliver("/nandi/wind/cmd", {"speed_mps": 10.0})
    r.run(1.0)
    tr = r.p.trunk
    f0 = float(r.d.xfrc_applied[tr, 1])
    r.d.xfrc_applied[:] = 0.0
    r.d.time = 0.0
    r.p.wind_tick(model, r.d, 0.0, r.bus)
    assert float(r.d.xfrc_applied[tr, 1]) == pytest.approx(f0, rel=0.2)
