"""WP7 wind: drag = 1/2 rho v^2 Cd A within 1 % (forces and the resulting accelerations), silhouettes, nested-tube
shadowing, gust flag latency <= 0.1 s, and the wind hook on the real robot (ROS-free, headless, each <= 10 s)."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

import mujoco  # noqa: E402

from atharv_nandi.core import wind_model as wm  # noqa: E402
from test_platform import Run, mb  # noqa: E402

EYE = (1, 0, 0, 0, 1, 0, 0, 0, 1)

TEST_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 0" timestep="0.001"/>
  <worldbody>
    <body name="trunk" pos="0 0 1">
      <freejoint/>
      <geom name="box" type="box" size="0.34 0.14 0.05" mass="10" euler="0 0 {yaw}"/>
    </body>
    <body name="mast_s1" pos="3 0 1">
      <freejoint/>
      <geom name="tube" type="cylinder" size="0.025 0.4" mass="2"/>
    </body>
    <body name="crown_head" pos="6 0 1">
      <freejoint/>
      <geom name="cap" type="capsule" size="0.05 0.1" mass="1" euler="{tilt} 0 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _xmat_rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c, -s, 0, s, c, 0, 0, 0, 1)


def test_silhouette_areas():
    a, b, c = 0.34, 0.14, 0.05
    assert wm.projected_area(wm.G_BOX, (a, b, c), EYE, (0, 1, 0)) == pytest.approx(4 * a * c)
    assert wm.projected_area(wm.G_BOX, (a, b, c), EYE, (1, 0, 0)) == pytest.approx(4 * b * c)
    u = (math.cos(0.3), math.sin(0.3), 0.0)
    assert wm.projected_area(wm.G_BOX, (a, b, c), EYE, u) == pytest.approx(4 * c * (b * u[0] + a * u[1]))
    # a box rotated 90 deg about z shows its long face to x-wind
    assert wm.projected_area(wm.G_BOX, (a, b, c), _xmat_rz(math.pi / 2), (1, 0, 0)) == pytest.approx(4 * a * c)
    r, h = 0.025, 0.4
    assert wm.projected_area(wm.G_CYLINDER, (r, h, 0), EYE, (1, 0, 0)) == pytest.approx(4 * r * h)
    assert wm.projected_area(wm.G_CYLINDER, (r, h, 0), EYE, (0, 0, 1)) == pytest.approx(math.pi * r * r)
    th = math.radians(30)
    u = (math.cos(th), 0.0, math.sin(th))     # 60 deg from the axis
    assert wm.projected_area(wm.G_CYLINDER, (r, h, 0), EYE, u) == pytest.approx(
        4 * r * h * math.cos(th) + math.pi * r * r * math.sin(th))
    assert wm.projected_area(wm.G_CAPSULE, (r, h, 0), EYE, (1, 0, 0)) == pytest.approx(4 * r * h + math.pi * r * r)
    assert wm.projected_area(wm.G_SPHERE, (0.06, 0, 0), EYE, (1, 0, 0)) == pytest.approx(math.pi * 0.0036)
    assert wm.projected_area(wm.G_ELLIPSOID, (0.06, 0.06, 0.06), EYE, (1, 0, 0)) == pytest.approx(math.pi * 0.0036)
    assert wm.projected_area(0, (1, 1, 1), EYE, (1, 0, 0)) == 0.0
    assert (wm.cd_for(wm.G_BOX), wm.cd_for(wm.G_CYLINDER), wm.cd_for(wm.G_CAPSULE), wm.cd_for(wm.G_SPHERE)) == \
        (1.5, 1.2, 1.2, 0.5)
    assert wm.drag_force_n(10.0, 1.2, 0.1) == pytest.approx(0.5 * 1.225 * 100 * 1.2 * 0.1)


def test_nested_tube_shadowing():
    """Stowed: inner sections inside the outer tube are hidden; extended: only the part above it is exposed."""
    outer = wm.TubeSpan(1, 0.0, 0.0, 0.03, 0.4, 1.0)
    s1 = wm.TubeSpan(2, 0.0, 0.0, 0.025, 0.45, 0.95)          # fully inside
    s2 = wm.TubeSpan(3, 0.001, 0.0, 0.02, 0.8, 1.6)           # 0.2 m inside, 0.6 m exposed
    other = wm.TubeSpan(4, 0.5, 0.0, 0.01, 0.4, 1.0)          # not coaxial
    ex = wm.exposed_spans([outer, s1, s2, other])
    assert ex[1][0] == pytest.approx(1.0) and ex[4][0] == pytest.approx(1.0)
    assert ex[2][0] == pytest.approx(0.0)
    assert ex[3][0] == pytest.approx(0.6 / 0.8) and ex[3][1] == pytest.approx(1.3)
    # s2 is shadowed by both the outer tube and s1 where they overlap (union, not double counted)
    s1b = wm.TubeSpan(2, 0.0, 0.0, 0.025, 0.45, 1.2)
    ex = wm.exposed_spans([outer, s1b, s2])
    assert ex[3][0] == pytest.approx(0.4 / 0.8) and ex[3][1] == pytest.approx(1.4)
    assert ex[2][0] == pytest.approx(0.2 / 0.75)


@pytest.mark.parametrize("v,dir_deg,yaw", [(10.0, 90.0, 0.0), (11.0, 0.0, 0.0), (15.0, 30.0, 0.4), (5.0, 200.0, 1.2)])
def test_drag_equals_half_rho_v2_cd_a_within_1_percent(v, dir_deg, yaw):
    """Force written to xfrc_applied AND the free-body acceleration it produces match 1/2 rho v^2 Cd A to 1 %."""
    m = mujoco.MjModel.from_xml_string(TEST_XML.format(yaw=yaw, tilt=0.7))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    drag = wm.WindDrag(m)
    assert drag.body_names == ["trunk", "mast_s1", "crown_head"]
    script = wm.WindScript(speed_mps=v, dir_deg=dir_deg)
    drag.apply(m, d, script, 0.0)
    th = math.radians(dir_deg)
    u = (math.cos(th), math.sin(th), 0.0)
    q = 0.5 * wm.RHO_AIR * v * v
    expect = {
        "trunk": q * 1.5 * 4 * 0.05 * (0.14 * abs(math.cos(th - yaw)) + 0.34 * abs(math.sin(th - yaw))),
        "mast_s1": q * 1.2 * 4 * 0.025 * 0.4,
        "crown_head": q * 1.2 * (4 * 0.05 * 0.1 * math.sqrt(1 - (math.sin(0.7) * u[1]) ** 2) + math.pi * 0.05 ** 2),
    }
    mujoco.mj_forward(m, d)
    for name, f_exp in expect.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        f = d.xfrc_applied[bid, :3]
        assert np.linalg.norm(f) == pytest.approx(f_exp, rel=0.01)
        assert np.dot(f, u) / np.linalg.norm(f) == pytest.approx(1.0, abs=1e-9)    # along the wind
        acc = d.qacc[m.jnt_dofadr[m.body_jntadr[bid]]:][:3]
        assert np.linalg.norm(acc) * m.body_mass[bid] == pytest.approx(f_exp, rel=0.01)
    # relative airspeed: a body moving with the wind feels less
    d.qvel[0:3] = [0.5 * v * u[0], 0.5 * v * u[1], 0.0]
    mujoco.mj_forward(m, d)
    drag.apply(m, d, script, 0.0)
    assert np.linalg.norm(d.xfrc_applied[1, :3]) == pytest.approx(0.25 * expect["trunk"], rel=0.01)


def test_drag_torque_is_lever_times_force_and_bookkeeping_keeps_other_writers():
    xml = """<mujoco><option gravity="0 0 0"/><worldbody><body name="trunk" pos="0 0 0.3"><freejoint/>
      <geom type="box" size="0.3 0.1 0.05" mass="5"/>
      <geom type="cylinder" size="0.025 0.5" pos="0 0 0.6" mass="0.01"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    d.xfrc_applied[1, :] = [1, 2, 3, 4, 5, 6]          # another writer (a WP6 harness)
    drag = wm.WindDrag(m)
    script = wm.WindScript(speed_mps=12.0, dir_deg=90.0)
    drag.apply(m, d, script, 0.0)
    drag.apply(m, d, script, 0.02)                     # replaced, not accumulated
    q = 0.5 * wm.RHO_AIR * 144
    f_box, f_tube = q * 1.5 * 4 * 0.3 * 0.05, q * 1.2 * 4 * 0.025 * 0.5
    own = d.xfrc_applied[1] - np.array([1, 2, 3, 4, 5, 6])
    assert own[1] == pytest.approx(f_box + f_tube, rel=1e-9) and abs(own[0]) < 1e-9
    com_z = d.xipos[1][2]
    tx = -((0.3 - com_z) * f_box + (0.9 - com_z) * f_tube)   # r x F, F along +y: tau_x = -r_z F_y
    assert own[3] == pytest.approx(tx, rel=1e-6)
    drag.clear(d)
    assert np.allclose(d.xfrc_applied[1], [1, 2, 3, 4, 5, 6])


def test_wind_script_gusts_turbulence_and_parse():
    s = wm.WindScript.from_dict({"speed_mps": 3.0, "dir_deg": 0.0, "gusts": [
        {"t0": 10.0, "peak_mps": 11.0, "rise_s": 1.0, "hold_s": 2.0, "fall_s": 1.0}]})
    assert s.speed(0.0) == pytest.approx(3.0) and s.speed(10.5) == pytest.approx(7.0)
    assert s.speed(12.0) == pytest.approx(11.0) and s.speed(13.5) == pytest.approx(7.0) and s.speed(20) == 3.0
    assert not s.calm and wm.WindScript().calm
    sh = wm.WindScript(speed_mps=5.0, shear_alpha=0.2, z_ref_m=2.0)
    assert sh.speed(0.0, 2.0) == pytest.approx(5.0) and sh.speed(0.0, 1.0) == pytest.approx(5.0 * 0.5 ** 0.2)
    tu = wm.WindScript(speed_mps=6.0, turb_sigma_mps=1.0, seed=3)
    v = [tu.speed(k * 0.02) for k in range(20000)]
    assert abs(np.mean(v) - 6.0) < 0.3 and 0.7 < np.std(v) < 1.3
    tu.reset()
    assert tu.speed(0.0) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        wm.WindScript(speed_mps=-1.0)
    with pytest.raises(ValueError):
        wm.Gust(0.0, 5.0, rise_s=-1.0)


def _flag_time(script, t_end, dt=0.02):
    an = wm.Anemometer()
    k = 0
    while k * dt <= t_end:
        t = k * dt
        st = an.update(t, script.velocity(t)[:2])
        if st["gust"]:
            return t, an
        k += 1
    return None, an


GUST_CASES = [(0.0, 11.0, 0.0), (3.0, 11.0, 0.0), (4.0, 15.0, 0.0), (3.0, 11.0, 1.0), (5.0, 12.0, 2.0)]


@pytest.mark.parametrize("base,peak,rise", GUST_CASES)
def test_gust_flag_within_0p1_s_of_onset(base, peak, rise):
    t0 = 20.0
    s = wm.WindScript(speed_mps=base, gusts=[wm.Gust(t0, peak, rise_s=rise, hold_s=3.0, fall_s=1.0)])
    onset = wm.gust_criterion_onset(s, 0.0, t0 + rise + 1.0)   # truth: true speed >= its running mean + 5
    assert onset is not None
    t_flag, an = _flag_time(s, t0 + rise + 3.0)
    assert t_flag is not None and 0.0 <= t_flag - onset <= 0.1, (onset, t_flag)
    if rise == 0.0:
        assert t_flag - t0 <= 0.1
    st = an.state(t_flag)
    assert st["gust"] and st["gust_onset_t"] == t_flag and st["gust_count"] == 1


def test_no_false_gust_and_flag_clears():
    for s in (wm.WindScript(speed_mps=9.0), wm.WindScript(speed_mps=7.0, turb_sigma_mps=0.8, seed=5),
              wm.WindScript(gusts=[wm.Gust(5.0, 12.0, rise_s=60.0, hold_s=10.0)])):   # 0.2 m/s^2 build-up
        t, _ = _flag_time(s, 70.0)
        assert t is None
    s = wm.WindScript(speed_mps=2.0, gusts=[wm.Gust(10.0, 12.0, hold_s=2.0, fall_s=0.0)])
    an = wm.Anemometer()
    flags = [(k * 0.02, an.update(k * 0.02, s.velocity(k * 0.02)[:2])["gust"]) for k in range(1000)]
    on = [t for t, g in flags if g]
    assert on[0] == pytest.approx(10.02, abs=0.021) and on[-1] == pytest.approx(12.0 + 3.0, abs=0.05)
    assert an.state()["gust"] is False


def test_wind_hook_on_robot_trunk_and_mast(monkeypatch):
    """Scripted 11 m/s side gust on the standing robot with the mast mock: hook force = sum of element drags,
    the gust is flagged <= 0.1 s, the hook publishes /nandi/wind, and the robot does not fall (6-foot stance)."""
    monkeypatch.setenv("NANDI_WIND", '{"speed_mps": 2.0, "dir_deg": 90, "gusts": '
                                     '[{"t0": 3.0, "peak_mps": 11.0, "rise_s": 0.0, "hold_s": 3.0, "fall_s": 1.0}]}')
    m = mb.build(NANDI_TERRAIN="flat", NANDI_CROPS=1, NANDI_MAST_H=1.6, NANDI_MAST_HEAD_KG=0.6)
    r = Run(m)
    r.run(4.0)
    p = r.p
    # the NANDI_MAST_H mock (mast_mock, mast_head) plus the WP1 crown mast (stowed) of atharv_nandi.xml
    assert set(p.drag.body_names) == {"trunk", "mast_mock", "mast_head", "mast_base", "mast_s1", "mast_s2", "mast_s3",
                                      "crown_head"}
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, e.geom) for e in p.drag.elements}
    assert {"mast_outer", "mast_ring", "mast_drive", "crown_head_g"} <= names
    assert not any(m.geom_rgba[e.geom][3] <= 0 for e in p.drag.elements)    # hidden mass-only geoms: no drag
    w = r.bus.values["wind"]
    assert w["gust"] and 3.0 <= w["gust_onset_t"] <= 3.1
    q = 0.5 * wm.RHO_AIR * 11.0 ** 2
    f_trunk = q * 1.5 * 4 * 0.34 * 0.05       # trunk shell side face
    f_tube = q * 1.2 * 4 * 0.025 * 0.8         # 1.6 m mast tube (r 0.025)
    f_head = q * 0.5 * math.pi * 0.06 ** 2
    # WP1 stowed mast, side-on: outer tube 0.044 x 0.61, ring 0.12 x 0.08 (tubes), drive 0.08 x 0.06, head 0.12^2
    f_wp1 = q * (wm.CD_TUBE * (0.044 * 0.61 + 0.12 * 0.08) + wm.CD_BOX * (0.08 * 0.06 + 0.12 * 0.12))
    assert p.drag.last_force_n > f_trunk + f_tube + f_head + f_wp1      # + gantry, rack, bumper on the trunk
    assert p.drag.last_force_n == pytest.approx(f_trunk + f_tube + f_head + f_wp1, rel=0.35)
    pubs = r.msgs("/nandi/wind")
    assert any(x["gust"] for x in pubs) and pubs[-1]["force_n"] > 0
    assert any(a["id"] == "gust" for a in r.msgs("/nandi/platform/alert"))
    tilt = math.degrees(2 * math.asin(min(1.0, math.hypot(r.d.qpos[4], r.d.qpos[5]))))
    assert tilt < 10.0
    r.run(4.0)                                  # gust over: forces back to the 2 m/s base
    assert p.drag.last_force_n < 0.1 * (f_trunk + f_tube)
    mujoco.mj_resetDataKeyframe(m, r.d, 0)
    r.reg.reset(m, r.d)
    assert p.drag._applied == {} and p.anemo.gust is False


def test_wind_script_rejects_bad_fields():
    for bad in ({"dir_deg": float("nan")}, {"turb_tau_s": 0}, {"turb_tau_s": -1}, {"z_ref_m": 0},
                {"shear_alpha": float("inf")}, {"shear_alpha": -0.1}, {"turb_sigma_mps": float("nan")},
                {"speed_mps": "5"}, {"speed_mps": True}, {"gusts": [{"t0": 0}]},
                {"gusts": [{"t0": 0, "peak_mps": 9, "dir_deg": float("nan")}]}, {"gusts": "abc"}):
        with pytest.raises((ValueError, TypeError, KeyError)):
            wm.WindScript.from_dict(bad)
    with pytest.raises(ValueError):
        wm.Gust(0.0, 9.0, dir_deg=float("inf"))


def test_crosswind_gust_reaches_its_peak_speed():
    s = wm.WindScript(speed_mps=6.0, dir_deg=0.0, gusts=[wm.Gust(0.0, 11.0, dir_deg=90.0)])
    assert s.speed(0.5) == pytest.approx(11.0)
    same = wm.WindScript(speed_mps=6.0, dir_deg=0.0, gusts=[wm.Gust(0.0, 11.0)])
    assert same.speed(0.5) == pytest.approx(11.0)
    weak = wm.WindScript(speed_mps=6.0, dir_deg=0.0, gusts=[wm.Gust(0.0, 4.0, dir_deg=180.0)])
    assert weak.speed(0.5) == pytest.approx(6.0)                  # peak below the base: no extra


def test_non_finite_drag_is_refused_and_removed():
    class _S:
        calm = False

        def velocity(self, t, z=None):
            return (float("nan"), 0.0, 0.0)
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="trunk"><freejoint/><geom type="box" size=".2 .1 .05"/></body>'
        '</worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    drag = wm.WindDrag(m)
    drag.apply(m, d, wm.WindScript(speed_mps=5.0), 0.0)
    assert np.any(d.xfrc_applied)
    with pytest.raises(ValueError):
        drag.apply(m, d, _S(), 0.02)
    assert np.allclose(d.xfrc_applied, 0.0)
