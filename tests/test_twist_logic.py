"""WP2 twist release (T4 TWIST; ORCHARD-MVP-SPEC 3.4). Pure-logic tests of the release statistic, then headless
MuJoCo with the WP23 fruit pool (weld slot at the CoM, torquescale 1) and the real T4 tool (WP2):

    papaya hanging 10 s untouched never releases; full jaw closure without roll never releases;
    release above tau with >= 30 deg roll, not at 0.8 tau.
"""
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core.sim_driver import SimDriver
from atharv_nandi.core.tools import grasp_sense, registry as reg, twist_logic
from atharv_nandi.core.tools.tool_changer import tcp_closed_form
from atharv_nandi.core.tools.twist_logic import TwistRelease
import atharv_nandi.sim.hooks.tools as ht

DT = 0.02
PAPAYA = "L2-3:f01"


# ---------------------------------------------------------------------------------------------- logic
def _run(stat, torque_fn, roll_fn, seconds):
    t = 0.0
    while t < seconds:
        if stat.update(torque_fn(t), roll_fn(t)):
            return t
        t += DT
    return None


def test_hanging_untouched_never_releases():
    s = TwistRelease(0.3, DT)
    s.capture_baseline(0.0)
    assert _run(s, lambda t: 1e-4 * math.sin(t), lambda t: None, 10.0) is None and s.peak < 1e-3


def test_full_closure_without_roll_never_releases():
    s = TwistRelease(0.3, DT)
    s.capture_baseline(0.0)
    s.arm_roll(0.0)
    assert _run(s, lambda t: 1.0, lambda t: 0.0, 5.0) is None          # torque well above tau, no roll
    assert s.above > 100 and s.roll_deg == 0.0


@pytest.mark.parametrize("roll_deg, fires", [(29.9, False), (30.0, True), (45.0, True)])
def test_roll_gate(roll_deg, fires):
    s = TwistRelease(0.3, DT)
    s.capture_baseline(0.0)
    s.arm_roll(0.2)
    assert (_run(s, lambda t: 0.6, lambda t: 0.2 + math.radians(roll_deg), 2.0) is not None) == fires


def test_release_above_tau_not_at_0p8_tau():
    tau = 0.3
    lo = TwistRelease(tau, DT)
    lo.capture_baseline(0.0)
    lo.arm_roll(0.0)
    assert _run(lo, lambda t: 0.8 * tau, lambda t: math.radians(60), 10.0) is None
    hi = TwistRelease(tau, DT)
    hi.capture_baseline(0.0)
    hi.arm_roll(0.0)
    t = _run(hi, lambda t: 1.2 * tau, lambda t: math.radians(60), 10.0)
    # 40 ms LPF from 0 crosses tau / (1.2 tau) after ln(6)/(a/dt) ticks, then 3 ticks above
    assert t is not None and 0.04 <= t <= 0.2


def test_baseline_is_subtracted_and_released_latches():
    s = TwistRelease(0.3, DT)
    s.capture_baseline(0.5)                              # a constant rest moment is not a twist
    s.arm_roll(0.0)
    assert _run(s, lambda t: 0.5, lambda t: math.radians(90), 2.0) is None
    assert _run(s, lambda t: 0.9, lambda t: math.radians(90), 2.0) is not None and s.released
    assert s.update(0.0, 0.0)                            # latched
    t = TwistRelease(0.3, DT)
    t.arm_roll(0.0)
    t.update(0.2, 0.0)
    assert t.baseline == 0.2                             # first reading when not captured


def test_axis_helpers_and_validation():
    ax = twist_logic.pedicel_axis([0, 0, 0], [0.07, 0, 0.01])
    assert np.linalg.norm(ax) == pytest.approx(1.0) and ax[0] > 0.98
    assert twist_logic.axis_torque([0.2, 0.0, 5.0], [1, 0, 0]) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        twist_logic.pedicel_axis([0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        TwistRelease(0.0, DT)
    with pytest.raises(ValueError):
        TwistRelease(0.3, 0.0)


# ---------------------------------------------------------------------------------------------- MuJoCo
@pytest.fixture
def world(monkeypatch):
    def build():
        monkeypatch.setenv("NANDI_WORLD", "orchard_lite")
        m = farm_scene.load_farm_model()
        drv = SimDriver(m, load_hooks=False)
        rt = ht.install(drv)
        drv.wait(0.3)
        return m, drv, rt
    return build


def stem_torque(pool, d, k):
    s = pool.slots[k]
    axis = twist_logic.pedicel_axis(d.site_xpos[s.stem_end], d.site_xpos[s.anchor])
    return twist_logic.axis_torque(pool.reaction(d, k)[1], axis)


def test_papaya_hanging_10s_untouched_never_releases(world):
    m, drv, rt = world()
    pool, d = rt.pool, drv.d
    k = pool.assign(d, PAPAYA)
    s = pool.slots[k]
    stat = TwistRelease(0.3, DT)
    stat.capture_baseline(stem_torque(pool, d, k))
    stat.arm_roll(0.0)                                    # even with the roll armed: no roll, no torque
    for _ in range(int(10.0 / DT)):
        drv.wait(DT)
        assert not stat.update(stem_torque(pool, d, k), 0.0)
    assert stat.peak < 0.01 and pool.is_attached(d, k) and pool.anchor_error_m(d, k) < 1e-3
    assert m.eq_data[s.eq][10] == 1.0                     # torquescale 1: rows 3-5 are N.m


@pytest.mark.parametrize("scale, fires", [(0.8, False), (1.2, True)])
def test_applied_twist_about_the_pedicel(world, scale, fires):
    """The real weld reaction under an external torque about the pedicel (a stand-in jaw) at 0.8 / 1.2 tau."""
    m, drv, rt = world()
    pool, d = rt.pool, drv.d
    k = pool.assign(d, PAPAYA)
    s = pool.slots[k]
    tau = 0.3
    stat = TwistRelease(tau, DT)
    stat.capture_baseline(stem_torque(pool, d, k))
    stat.arm_roll(0.0)
    axis = twist_logic.pedicel_axis(d.site_xpos[s.stem_end], d.site_xpos[s.anchor])
    released = None
    for i in range(int(3.0 / DT)):
        d.xfrc_applied[s.body, 3:6] = scale * tau * min(1.0, i * DT / 0.5) * axis
        drv.wait(DT)
        if stat.update(stem_torque(pool, d, k), math.radians(45.0)):
            released = i * DT
            break
    d.xfrc_applied[s.body] = 0.0
    assert (released is not None) == fires
    if not fires:
        assert stat.peak == pytest.approx(0.8 * tau, rel=0.05)   # the measurement chain reads the applied twist


def stage_papaya(m, drv, rt):
    """Mount T4, move the papaya in front of the robot with its pedicel along the tool axis (the WP0c cradle
    approach: radial into the trunk), offset sideways so the fixed jaw meets its side, approach with open jaws."""
    pool, d = rt.pool, drv.d
    rt.request_change("T4")
    assert drv.run_until(lambda dr: not rt.changer.busy, 8.0) and rt.changer.mounted == "T4"
    t = reg.get("T4")
    k = pool.assign(d, PAPAYA)
    s = pool.slots[k]
    r = float(m.geom_size[s.fruit][0])
    tp, tR = drv.ik.trunk_pose(d.qpos)
    tcp_target = tp + tR @ np.array([0.66, 0.0, 0.30])
    _q, ax = tcp_closed_form(drv.ik, d.qpos, tcp_target, t.tcp_from_wrist_m)
    y = np.cross([0.0, 0.0, 1.0], ax)
    y /= np.linalg.norm(y)
    centre = tcp_target + y * (reg.JAW_FIXED_Y - r - 0.002)
    # turn the fruit 180 deg about z (its pedicel pointed at its own trunk, -x): weld relquat = conj(q)
    mujoco.mj_forward(m, d)
    delta = centre - d.xpos[s.body]
    m.eq_data[s.eq][0:3] += delta
    d.qpos[s.qadr:s.qadr + 3] += delta
    d.qpos[s.qadr + 3:s.qadr + 7] = [0.0, 0.0, 0.0, 1.0]
    m.eq_data[s.eq][6:10] = [0.0, 0.0, 0.0, -1.0]
    d.qvel[s.dadr:s.dadr + 6] = 0.0
    mujoco.mj_forward(m, d)
    pedicel = twist_logic.pedicel_axis(d.site_xpos[s.stem_end], d.site_xpos[s.anchor])
    assert twist_logic.axis_torque(pedicel, ax) > 0.98                       # roll axis ~ pedicel
    arm = ht.DriverArm(drv, rt)
    arm.tcp_to(tcp_target - 0.18 * ax, 1.5)
    assert arm.align(lambda: tcp_target) < 0.003
    ids = rt.h["T4"]
    assert max(grasp_sense.touch(d, a) for a in ids.touch_adr) < 0.5          # open jaws around it, no squeeze
    return k, s, ids


def roll_until(drv, rt, pool, k, deg, ramp_s, timeout_s):
    t0 = drv.t
    drv.add_task("twist_roll", lambda dr: rt.actuate({"roll": min(1.0, (dr.t - t0) / ramp_s) * math.radians(deg)}),
                 50.0)
    try:
        drv.run_until(lambda dr: not pool.is_attached(dr.d, k), timeout_s)
    finally:
        drv.remove_task("twist_roll")


def test_t4_twist_releases_with_roll_and_not_on_closure(world):
    m, drv, rt = world()
    pool, d = rt.pool, drv.d
    k, s, ids = stage_papaya(m, drv, rt)
    n0 = len(rt.events)
    rt.actuate({"value": 0.0})                                               # close the jaws
    drv.wait(2.0)
    ta, tb = (grasp_sense.touch(d, a) for a in ids.touch_adr)
    assert ta > 1.0 and tb > 1.0                                             # both pads press the papaya
    assert pool.is_attached(d, k) and rt.twist[k].roll0 is not None          # closed, roll armed: no release
    assert not rt.request_change(None) and rt.events[-1]["reason"] == "holding_payload"
    roll_until(drv, rt, pool, k, 90.0, 2.0, 6.0)
    rel = [e for e in rt.events[n0:] if e["event"] == "twist_release"]
    assert len(rel) == 1 and not pool.is_attached(d, k)
    e = rel[0]
    assert e["roll_deg"] >= 30.0 and e["tau_peak_nm"] > e["tau_detach_nm"] == 0.3
    assert e["caught"] and rt.caught["slot"] == k


def test_t4_twist_below_tau_never_releases(world):
    """tau set above the roll drive's torque: the pads turn >= 30 deg but the stalk moment stays <= 0.8 tau."""
    m, drv, rt = world()
    pool, d = rt.pool, drv.d
    rt._tau_cache = {"papaya": 2.6}                    # roll cap 2 N.m = 0.77 tau
    k, s, ids = stage_papaya(m, drv, rt)
    rt.actuate({"value": 0.0})
    drv.wait(1.0)
    roll_until(drv, rt, pool, k, 90.0, 2.0, 5.0)
    stat = rt.twist[k]
    assert pool.is_attached(d, k) and stat.roll_deg >= 30.0
    assert stat.peak <= 0.8 * 2.6
