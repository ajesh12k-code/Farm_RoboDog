"""BT4 of the physics alignment spec v1: the stability probe on the live farm_scene model.

Thresholds are model-agnostic (they hold before and after the Package A re-mass / re-gain):
the home stance must have a 6-foot margin >= 0.10 m and tip angle >= 15 deg, and the probe's
predicted tip angle must match a sim tilt-to-tip experiment within 3 deg.

Tilt-to-tip experiment: tilting gravity is the same as tilting the floor (the whole problem rotates
rigidly), so the test rotates model.opt.gravity stepwise by 0.5 deg toward one side, lets the robot
settle 0.4 s per step with the legs held at the home ctrl (balance off, no mast mock), and records the
first tilt at which the trunk has rotated > 10 deg relative to the floor. The geometric prediction
assumes a rigid robot that does not slide, so the test copy of the model gets (a) stiff legs (kp 3000)
and (b) friction 2.0 with the elliptic cone, impratio 10. MEASURED 2026-11-17 on this model (scratch run):
with the as-built kp 120 legs and the pyramidal cone the robot creeps downhill (0.2-0.9 m) and tips at
16-33 deg against a 35-40 deg rigid prediction; with kp 3000 + elliptic cone the error is <= 0.5 deg.
test_report_compliant_tip_angle prints the as-built numbers (not asserted).
"""
import math
import time

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene as fs
from atharv_nandi.core import stability as st
from atharv_nandi.core.stability_probe import FOOT_GEOMS, StabilityProbe, assess, basket_load

TILT_STEP_DEG = 0.5
TILT_SETTLE_S = 0.4
TIPPED_DEG = 10.0


def load(monkeypatch, terrain="flat", crops=1, **env):
    monkeypatch.setenv("NANDI_TERRAIN", terrain)
    monkeypatch.setenv("NANDI_CROPS", str(crops))
    monkeypatch.delenv("NANDI_SEED", raising=False)
    for k in ("NANDI_PAYLOAD_KG", "NANDI_MAST_H", "NANDI_MAST_HEAD_KG"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return fs.load_farm_model()


def settled(m, seconds=2.0):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[fs.crop_qpos_slice(1)][:3] = (6.0, 6.0, fs.CROP_RADIUS)  # crop out of the way
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d


def trunk_tilt_deg(q):
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (q[4] ** 2 + q[5] ** 2)))))


def tilt_to_tip(m, d0, direction_xy, max_deg=60.0):
    """Smallest gravity tilt (deg, toward direction_xy) at which the trunk rotates > TIPPED_DEG."""
    d = mujoco.MjData(m)
    d.qpos[:] = d0.qpos
    d.ctrl[:] = d0.ctrl
    tilt0 = trunk_tilt_deg(d.qpos)
    g0 = m.opt.gravity.copy()
    n_settle = int(TILT_SETTLE_S / m.opt.timestep)
    try:
        for k in range(int(max_deg / TILT_STEP_DEG) + 1):
            th = math.radians(k * TILT_STEP_DEG)
            m.opt.gravity[:] = 9.81 * np.array([direction_xy[0] * math.sin(th), direction_xy[1] * math.sin(th),
                                                -math.cos(th)])
            for _ in range(n_settle):
                mujoco.mj_step(m, d)
            if trunk_tilt_deg(d.qpos) - tilt0 > TIPPED_DEG:
                return k * TILT_STEP_DEG
    finally:
        m.opt.gravity[:] = g0
    return None


def stiff_no_slip(m, kp=3000.0):
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.impratio = 10.0
    m.geom_friction[:, 0] = 2.0
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.endswith(("_hip_joint", "_thigh_joint", "_calf_joint")):
            m.actuator_gainprm[i, 0] = kp
            m.actuator_biasprm[i, 1] = -kp
            m.actuator_biasprm[i, 2] = -2.0 * math.sqrt(kp)
            m.actuator_forcerange[i] = (-2000.0, 2000.0)


# --- BT4 -------------------------------------------------------------------------------------------
def test_home_keyframe_stand_margin_and_tip(monkeypatch):
    m = load(monkeypatch)
    p = StabilityProbe(m)
    r = p.probe(m.key_qpos[0])
    assert r["n_contacts"] == 6 and sorted(r["feet_in_contact"]) == sorted(FOOT_GEOMS)
    assert r["margin_m"] >= st.STAND_MIN_MARGIN_M
    assert r["tip_angle_deg"] >= st.STAND_MIN_TIP_DEG
    assert r["mode"] == st.MODE_STAND
    assert r["mass_kg"] == pytest.approx(float(m.body_subtreemass[m.body("trunk").id]))
    v = st.classify(r["mode"], {"n_contacts": 6, "margin_m": r["margin_m"], "tip_angle_deg": r["tip_angle_deg"]})
    assert v["ok"]
    print(f"\nBT4 home: margin {r['margin_m']:.3f} m, tip {r['tip_angle_deg']:.2f} deg, CoM {np.round(r['com'], 4)}, "
          f"mass {r['mass_kg']:.2f} kg")


@pytest.mark.parametrize("name,direction", [("rear", (-1.0, 0.0)), ("left", (0.0, 1.0))])
def test_predicted_tip_angle_matches_tilt_experiment(monkeypatch, name, direction):
    m = load(monkeypatch)
    stiff_no_slip(m)
    d = settled(m)
    p = StabilityProbe(m)
    r = p.probe(d.qpos)
    assert r["n_contacts"] == 6
    pred = p.tip_angle_toward(r, direction)
    measured = tilt_to_tip(m, d, direction)
    print(f"\nBT4 tilt {name}: predicted {pred:.2f} deg, measured {measured} deg")
    assert measured is not None
    assert abs(measured - pred) <= 3.0


def test_report_compliant_tip_angle(monkeypatch):
    """REPORT ONLY: the as-built model (its own gains, cone, friction) on the same tilt ramp."""
    m = load(monkeypatch)
    d = settled(m)
    p = StabilityProbe(m)
    r = p.probe(d.qpos)
    out = {}
    for name, direction in (("rear", (-1.0, 0.0)), ("left", (0.0, 1.0))):
        out[name] = (round(p.tip_angle_toward(r, direction), 2), tilt_to_tip(m, d, direction))
    print(f"\nBT4 as-built (predicted rigid, measured compliant) deg: {out}")


def test_probe_never_steps_and_is_cheap(monkeypatch):
    m = load(monkeypatch, terrain="rough", crops=6)
    p = StabilityProbe(m)
    q = m.key_qpos[0].copy()
    r1 = p.probe(q)
    assert p.data.time == 0.0  # mj_forward only
    t0 = time.perf_counter()
    n = 100
    for _ in range(n):
        p.probe(q)
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"\nprobe cost {ms:.3f} ms (rough, 6 crops)")
    assert ms < 2.0
    assert p.probe(q)["margin_m"] == r1["margin_m"]  # deterministic, no state carried over
    with pytest.raises(ValueError):
        p.probe(q[:-1])


def test_unsupported_when_lifted(monkeypatch):
    m = load(monkeypatch)
    q = m.key_qpos[0].copy()
    q[2] += 0.2  # all feet in the air
    r = StabilityProbe(m).probe(q)
    assert r["n_contacts"] == 0 and r["margin_m"] == -math.inf and r["mode"] == st.MODE_UNSUPPORTED
    s = assess(r)
    assert s["margin_m"] is None and st.INHIBIT_WALK in s["inhibit"]
    assert st.parse_stability_status(st.build_stability_status(s)) == s


def test_basket_crop_shifts_com(monkeypatch):
    m = load(monkeypatch)
    q = m.key_qpos[0].copy()
    sl = fs.crop_qpos_slice(1)
    q[sl.start:sl.start + 3] = q[0:3] + np.array([-0.2, 0.0, 0.096])  # resting on the basket floor
    kg, loads = basket_load(m, q, ["crop_item_1", "crop_item_99"], payload_kg=0.0)
    assert kg == pytest.approx(0.15) and len(loads) == 1
    p = StabilityProbe(m)
    r0, r1 = p.probe(q), p.probe(q, extra_loads=loads)
    m0 = r0["mass_kg"]
    assert r1["mass_kg"] == pytest.approx(m0 + 0.15)
    expect_x = (m0 * r0["com"][0] + 0.15 * (q[0] - 0.2)) / (m0 + 0.15)
    assert r1["com"][0] == pytest.approx(expect_x, abs=1e-12)
    s = assess(r1, basket_kg=kg)
    assert s["basket_kg"] == pytest.approx(0.15) and s["basket_binding"] == "volume"


def test_slope_from_feet_matches_imu_on_tilted_stand(monkeypatch):
    """Gravity tilted 6 deg toward +y = the floor rolled left (+y) side down. Slopes are relative to
    gravity, so the foot plane fit and the IMU must both report roll -6 deg, pitch 0."""
    m = load(monkeypatch)
    q = m.key_qpos[0].copy()
    ang = math.radians(6.0)
    g0 = m.opt.gravity.copy()
    try:
        m.opt.gravity[:] = 9.81 * np.array([0.0, math.sin(ang), -math.cos(ang)])
        r = StabilityProbe(m).probe(q)
    finally:
        m.opt.gravity[:] = g0
    assert r["slope_imu_deg"][1] == pytest.approx(-6.0, abs=1e-6)
    assert r["slope_points_deg"][1] == pytest.approx(-6.0, abs=0.5)
    assert r["slope_points_deg"][0] == pytest.approx(r["slope_imu_deg"][0], abs=0.5)
    left = max(r["edges"], key=lambda e: e["outward"][1])    # downhill edge
    right = min(r["edges"], key=lambda e: e["outward"][1])
    assert left["tip_deg"] < right["tip_deg"]
    s = assess(r)
    assert s["slope_deg"]["roll"] == pytest.approx(-6.0, abs=0.5)


def _mast_model(height_m, head_kg):
    """A welded mast + head on the trunk (contype 0), built here so the test does not need Package A."""
    spec = mujoco.MjSpec.from_file(fs.ROBOT_XML)
    trunk = spec.body("trunk")
    base = trunk.add_body(name="test_mast", pos=[0.03, 0.0, 0.05])
    base.add_site(name="test_mast_base_site", pos=[0, 0, 0.01], size=[0.006, 0, 0])
    length = height_m - 0.34 - 0.05
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, 0, 0, length], size=[0.02, 0, 0],
                  mass=1.0 * length, contype=0, conaffinity=0)
    head = base.add_body(name="test_mast_head", pos=[0, 0, length])
    head.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0], mass=head_kg, contype=0, conaffinity=0)
    head.add_site(name="mast_head_ring", size=[0.006, 0, 0])
    spec.add_sensor(name="mast_base_f", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="test_mast_base_site")
    spec.add_sensor(name="mast_base_t", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                    objname="test_mast_base_site")
    return spec.compile()


def test_mast_head_push_and_mode():
    m = _mast_model(2.2, 1.5)
    assert m.nq == fs.CROP_QPOS_START + 7
    p = StabilityProbe(m)
    r = p.probe(m.key_qpos[0])
    assert r["mode"] == st.MODE_MAST_UP
    assert r["mast_h_m"] == pytest.approx(2.2 - np.mean([c[2] for c in r["contact_points"]]), abs=0.02)
    z_head = r["mast_head"][2]
    expect = min(r["mass_kg"] * 9.81 * e["d_perp_m"] / (z_head - e["z_edge_m"]) for e in r["edges"])
    assert r["head_push_rigid_n"] == pytest.approx(expect, rel=1e-9)
    # review F8: the reported value is the rigid one x the measured compliance factor (conservative)
    assert r["head_push_to_tip_n"] == pytest.approx(expect * st.HEAD_PUSH_COMPLIANCE, rel=1e-9)
    assert 0.0 < st.HEAD_PUSH_COMPLIANCE < 0.84  # below the worst MEASURED sim / rigid ratio (0.84)
    assert r["mast_base_force_n"] > 0 and r["mast_base_moment_nm"] is not None
    s = assess(r, head_kg=1.5)
    assert st.INHIBIT_WALK in s["inhibit"]
    assert s["head_push_to_tip_n"] is not None and s["head_push_to_tip_n"] >= st.MAST_UP_MIN_HEAD_PUSH_N
    print(f"\nmast 2.2 m / 1.5 kg mock at home: head push to tip {r['head_push_to_tip_n']:.1f} N, "
          f"margin {r['margin_m']:.3f} m")


def test_package_a_mocks_are_read_when_present(monkeypatch):
    """After Package A: NANDI_MAST_H / NANDI_PAYLOAD_KG mocks and the section 4 sites. Skipped before."""
    m = load(monkeypatch, NANDI_MAST_H=2.2, NANDI_PAYLOAD_KG=10)
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mast_head_ring") < 0:
        pytest.skip("Package A mast mock not merged yet")
    r = StabilityProbe(m).probe(m.key_qpos[0])
    assert r["mode"] == st.MODE_MAST_UP and r["mast_h_m"] > 1.0
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu") >= 0


def test_monitor_core_status_is_valid_json(monkeypatch):
    from atharv_nandi.core.stability_probe import MonitorCore
    m = load(monkeypatch)
    core = MonitorCore(m, payload_kg=2.0, head_kg=1.5)
    assert core.status(0.0) is None  # nothing received yet
    assert not core.on_qpos([0.0] * (m.nq - 1)) and core.bad_nq == 1
    q = m.key_qpos[0].copy()
    sl = fs.crop_qpos_slice(1)
    q[sl.start:sl.start + 3] = q[0:3] + np.array([0.0, 0.0, 0.096])
    assert core.on_qpos(q)
    core.on_game({"in_basket_names": ["crop_item_1"], "in_basket": 1})
    core.on_joint_states(0.0, ["gantry_slide", "FR_hip_joint", "FR_calf_joint"], [70.0, 3.0, -4.0])
    s = core.status(1.0)
    assert s["basket_kg"] == pytest.approx(2.15) and s["leg_rms_nm_max"] == pytest.approx(4.0)
    assert s["mode"] == st.MODE_STAND and s["inhibit"] == [] and s["advisory"] and not s["enforced"]
    assert core.head_kg is None  # no mast in this model
    assert st.parse_stability_status(st.build_stability_status(s)) == s


# --- review F3 (2026-11-17): tripod-gait transients must not read as "unsupported" -----------------
def test_stance_tripod_counts_when_just_above_ground(monkeypatch):
    """All feet 8 mm up (none within CONTACT_DIST_M): the stance tripod supports, mode TRIPOD_WALK.
    50 mm up: nothing is near the ground, UNSUPPORTED as before."""
    m = load(monkeypatch)
    p = StabilityProbe(m)
    q = m.key_qpos[0].copy()
    q[2] += 0.008
    r = p.probe(q)
    assert r["n_contacts"] == 0 and r["support"] == st.SUPPORT_STANCE_TRIPOD and r["n_support"] == 3
    assert 0.005 < r["stance_clearance_m"] <= st.STANCE_NEAR_M
    assert r["mode"] == st.MODE_TRIPOD_WALK and r["margin_m"] > 0 and len(r["polygon"]) == 3
    assert {f[:2] for f in r["support_feet"]} in ({"FR", "ML", "RR"}, {"FL", "MR", "RL"})
    q[2] += 0.042
    r = p.probe(q)
    assert r["support"] == st.SUPPORT_NONE and r["mode"] == st.MODE_UNSUPPORTED
    assert r["stance_clearance_m"] > st.STANCE_NEAR_M
    home = p.probe(m.key_qpos[0])
    assert home["support"] == st.SUPPORT_CONTACT and home["stance_clearance_m"] is None


def test_monitor_debounces_unsupported_and_walk_inhibit(monkeypatch):
    from atharv_nandi.core.stability_probe import MonitorCore
    m = load(monkeypatch)
    core = MonitorCore(m)
    q_ok = m.key_qpos[0].copy()
    q_up = q_ok.copy()
    q_up[2] += 0.2
    core.on_qpos(q_ok)
    assert core.status(0.0)["mode"] == st.MODE_STAND
    seen = []
    for k in range(1, 6):  # lifted from t = 0.1 s on, 10 Hz like the node
        core.on_qpos(q_up)
        s = core.status(0.1 * k)
        assert s["mode_raw"] == st.MODE_UNSUPPORTED and st.INHIBIT_WALK in s["inhibit_raw"]
        seen.append((s["mode"], st.INHIBIT_WALK in s["inhibit"]))
        assert st.parse_stability_status(st.build_stability_status(s)) == s
    # held for < INHIBIT_DEBOUNCE_S (0.1, 0.2, 0.3 s after onset is >= 0.3 at the 4th sample)
    assert seen[:3] == [(st.MODE_STAND, False)] * 3
    assert seen[3:] == [(st.MODE_UNSUPPORTED, True)] * 2
    core.on_qpos(q_ok)
    s = core.status(0.6)
    assert s["mode"] == st.MODE_STAND and s["inhibit"] == []


def test_mast_trip_is_not_debounced(monkeypatch):
    from atharv_nandi.core.stability_probe import MonitorCore
    core = MonitorCore(load(monkeypatch))
    assert st.UNDEBOUNCED_INHIBITS == (st.INHIBIT_MAST_TRIP,)
    s = core._debounce(5.0, {"mode": st.MODE_MAST_UP, "inhibit": [st.INHIBIT_MAST_TRIP, st.INHIBIT_WALK]})
    assert s["inhibit"] == [st.INHIBIT_MAST_TRIP] and s["inhibit_raw"] == [st.INHIBIT_MAST_TRIP, st.INHIBIT_WALK]


WALK_S = 3.0
WALK_MONITOR_HZ = 10


@pytest.mark.parametrize("payload_kg", [0, 10])
def test_monitor_does_not_block_its_own_tripod_walk(monkeypatch, payload_kg):
    """Review F3: closed-loop tripod walk, vx 1 on flat ground, monitor sampled at 10 Hz like the node.
    MEASURED on the aligned model before this fix (tools/physics/monitor_walk_probe.py, 6 s): the probe
    said UNSUPPORTED in 69.7 % (0 kg) / 46 % (10 kg) of samples and the monitor inhibited walking in
    76 % / 65 % (50 Hz sampling). After: the
    debounced walk inhibit is 0 % in every walk measured (flat 0 / 10 kg, turn, rough seeds 1 / 5, ridges).
    Needs the aligned (Package A) model: on the pre-alignment gait the tripod margin stays under 0.05 m for
    longer than a stance phase and the monitor (correctly) inhibits 33 % of samples (MEASURED, this test).
    The pre-alignment model has no "imu" site, which is the marker used here."""
    from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch
    from atharv_nandi.core.stability_probe import MonitorCore
    from atharv_nandi.core.tripod_gait import TripodGaitGenerator
    env = {"NANDI_PAYLOAD_KG": payload_kg} if payload_kg else {}
    m = load(monkeypatch, **env)
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu") < 0:
        pytest.skip("pre-alignment model (Package A not merged)")
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[fs.crop_qpos_slice(1)][:3] = (6.0, 6.0, fs.CROP_RADIUS)
    core = MonitorCore(m, payload_kg=payload_kg)
    gait = TripodGaitGenerator(0.02)
    sub = int(round(0.02 / m.opt.timestep))
    every = int(round(1.0 / (WALK_MONITOR_HZ * 0.02)))
    raw_unsup = blocked = n = 0
    for k in range(int((WALK_S + 1.0) / 0.02)):
        if k == 50:
            gait.set_command(1.0, 0.0, 0.0, hold=True)
        roll, pitch = quat_to_roll_pitch(*d.qpos[3:7])
        d.ctrl[7:25] = gait.step(roll, pitch)
        for _ in range(sub):
            mujoco.mj_step(m, d)
        if k < 50 or k % every:
            continue
        core.on_qpos(d.qpos)
        s = core.status(k * 0.02)
        n += 1
        raw_unsup += s["mode_raw"] == st.MODE_UNSUPPORTED
        blocked += st.INHIBIT_WALK in s["inhibit"]
    print(f"\nwalk {payload_kg} kg: raw UNSUPPORTED {100 * raw_unsup / n:.1f} %, walk inhibited "
          f"{100 * blocked / n:.1f} % of {n} samples")
    assert n >= 25 and blocked == 0
    assert raw_unsup / n < 0.35  # stance-tripod support (raw, no debounce) cuts the 46-70 % baseline
