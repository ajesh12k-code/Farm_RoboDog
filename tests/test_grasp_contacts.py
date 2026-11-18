"""WPCAP contact / grasp physics (docs/cap-libraries.md 3): per-pair params for finger-fruit, fruit-tray and
foot-soil, documented in atharv_nandi/sim/world/modules/contact_pairs.py and checked here in the composed orchard_demo world.

Scenes are set up kinematically (no IK walk) so each test is a few hundred ms: the robot stands at home on its
servos, a pool fruit is freed from its equality (what FruitPool.assign does to the collision fields: geom and
body contype), placed between the open pads (gravcomp while the jaws close) and then lifted / swung, or placed in a
tray cell and left to settle. Thresholds: lift slip <= 2 mm and creep <= 1 mm/s (the harvest grasp check allows
6 mm over 0.3 s, GRASP_MAX_SLIP_M); tray: no jitter = drift <= 0.05 mm and relative speed <= 1 mm/s.
"""
import contextlib
import math
import os

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.sim.world import compose
from atharv_nandi.sim.world.modules import contact_pairs as cp

SLIP_MAX_M = 0.002
CREEP_MAX_M_S = 0.001
TRAY_DRIFT_MAX_M = 5e-5
TRAY_SPEED_MAX_M_S = 0.001


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def build(pairs: bool = True, world: str = "orchard_demo"):
    with env(NANDI_WORLD=world):
        if pairs:
            return farm_scene.load_farm_model()
        with compose.temporary_module("contact_pairs", cp.ORDER, lambda *a: None):
            return farm_scene.load_farm_model()


@pytest.fixture(scope="module")
def tuned():
    return build(True)


@pytest.fixture(scope="module")
def untuned():
    return build(False)


def G(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)


def A(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)


@contextlib.contextmanager
def free_fruit(m, slot, mass=None, contype=1):
    """Fresh MjData standing at home; pool slot freed and collidable. Model fields are restored afterwards."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"pool_{slot}")
    fg = G(m, f"pool_{slot}_fruit")
    saved = (int(m.geom_contype[fg]), int(m.body_contype[b]), float(m.body_mass[b]), float(m.body_gravcomp[b]))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for a in range(m.nu):
        d.ctrl[a] = d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]
    for _ in range(250):
        mujoco.mj_step(m, d)
    d.eq_active[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"pool_eq_{slot}")] = 0
    m.geom_contype[fg], m.body_contype[b] = contype, contype
    if mass:
        m.body_mass[b] = mass
    try:
        yield d, b, fg
    finally:
        m.geom_contype[fg], m.body_contype[b], m.body_mass[b], m.body_gravcomp[b] = saved


def _wrist(m, d):
    w = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm_wrist")
    return d.xpos[w].copy(), d.xmat[w].reshape(3, 3).copy()


def _in_wrist(m, d, p):
    wp, wr = _wrist(m, d)
    return wr.T @ (p - wp)


def close_on(m, d, b, offset=0.0):
    """Fruit between the open pads (offset m along the tool axis), jaws closed with the fruit weightless."""
    jnt = m.body_jntadr[b]
    adr, dof = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
    _, wr = _wrist(m, d)
    mid = 0.5 * (d.geom_xpos[G(m, "finger_left_pad")] + d.geom_xpos[G(m, "finger_right_pad")]) + wr[:, 0] * offset
    d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7], d.qvel[dof:dof + 6] = mid, [1, 0, 0, 0], 0.0
    m.body_gravcomp[b] = 1.0
    d.ctrl[A(m, "finger_left")] = d.ctrl[A(m, "finger_right")] = 0.04
    for _ in range(250):
        mujoco.mj_step(m, d)
    m.body_gravcomp[b] = 0.0


# arm at a WP0c side-pick pose (OT-1 mango_pedicel_cut 1, robot frame) for the lift test
PICK_Q = (-0.02, 1.5223, 0.2778, 0.8951, -0.9634)
ARM = ("gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch")


def lift(m, slot, mass=None, dshoulder=-0.3, steps=500):
    """Grasp at a pick pose, lift ~10 cm by raising the shoulder over 1 s, hold 0.3 s. Slip in the wrist frame."""
    with free_fruit(m, slot, mass) as (d, b, fg):
        for n, q in zip(ARM, PICK_Q):
            d.ctrl[A(m, n)] = q
        for _ in range(750):
            mujoco.mj_step(m, d)
        close_on(m, d, b)
        for _ in range(50):
            mujoco.mj_step(m, d)
        p0, z0 = _in_wrist(m, d, d.xpos[b]), d.xpos[b][2]
        sh0 = d.ctrl[A(m, "arm_shoulder")]
        slip = []
        for k in range(steps + 150):
            d.ctrl[A(m, "arm_shoulder")] = sh0 + dshoulder * min(1.0, (k + 1) / steps)
            mujoco.mj_step(m, d)
            slip.append(np.linalg.norm(_in_wrist(m, d, d.xpos[b]) - p0))
        slip = np.array(slip)
        creep = float(np.max(np.abs(np.diff(slip[-100:])))) / m.opt.timestep
        return {"slip": float(slip.max()), "creep": creep, "rise": float(d.xpos[b][2] - z0)}


def offset_swing(m, slot, mass=0.4, offset=0.015, dpitch=-0.6):
    """Fruit gripped 1.5 cm off the pad line, wrist pitched 0.6 rad: rotation of the fruit in the grip (deg)."""
    with free_fruit(m, slot, mass) as (d, b, fg):
        close_on(m, d, b, offset)
        _, wr = _wrist(m, d)
        r0 = wr.T @ d.xmat[b].reshape(3, 3)
        w0 = d.ctrl[A(m, "arm_wrist_pitch")]
        for k in range(750):
            d.ctrl[A(m, "arm_wrist_pitch")] = w0 + dpitch * min(1.0, (k + 1) / 500)
            mujoco.mj_step(m, d)
        _, wr = _wrist(m, d)
        r1 = wr.T @ d.xmat[b].reshape(3, 3)
        return math.degrees(math.acos(np.clip((np.trace(r0.T @ r1) - 1.0) / 2.0, -1.0, 1.0)))


def tray_settle(m, slot, drop=0.01, mass=None):
    """Fruit dropped `drop` m into front-left tray cell 0 (WP3 tray bit set), 0.5 s settle, 1 s watched."""
    with free_fruit(m, slot, mass, contype=1 | 4) as (d, b, fg):
        jnt = m.body_jntadr[b]
        adr, dof = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
        c = 0.5 * (d.geom_xpos[G(m, "tray_div_l0")] + d.geom_xpos[G(m, "tray_div_l1")])
        c[1] = 0.5 * (d.geom_xpos[G(m, "tray_lane_wall_l")][1] + d.geom_xpos[G(m, "basket_left_wall")][1])
        floor = d.geom_xpos[G(m, "basket_floor")][2] + m.geom_size[G(m, "basket_floor")][2]
        c[2] = floor + m.geom_size[fg][2] + drop
        d.qpos[adr:adr + 3], d.qpos[adr + 3:adr + 7], d.qvel[dof:dof + 6] = c, [1, 0, 0, 0], 0.0
        bk = m.geom_bodyid[G(m, "basket_floor")]

        def rel():
            return d.xmat[bk].reshape(3, 3).T @ (d.xpos[b] - d.xpos[bk])

        for _ in range(250):
            mujoco.mj_step(m, d)
        r0, ps = rel(), []
        for _ in range(500):
            mujoco.mj_step(m, d)
            ps.append(rel())
        ps = np.array(ps)
        speed = float(np.max(np.linalg.norm(np.diff(ps, axis=0), axis=1))) / m.opt.timestep
        cons = [c for c in d.contact[:d.ncon] if fg in (c.geom1, c.geom2)]
        return {"drift": float(np.linalg.norm(ps[-1] - r0)), "speed": speed,
                "contacts": [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2 if c.geom1 == fg else c.geom1),
                              int(c.dim), [float(v) for v in c.solref]) for c in cons]}


# ---------------------------------------------------------------------- compiled parameters
def test_pairs_compiled_in_orchard_worlds(tuned):
    m = tuned
    assert m.npair == 2 * 10
    ff = cp.PAIRS["finger_fruit"]
    pads = {G(m, p) for p in cp.PAD_GEOMS}
    for i in range(m.npair):
        g1, g2 = int(m.pair_geom1[i]), int(m.pair_geom2[i])
        assert g1 in pads and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2).endswith("_fruit")
        assert m.pair_dim[i] == ff["condim"]
        assert np.allclose(m.pair_friction[i], ff["friction"]) and np.allclose(m.pair_solref[i], ff["solref"])
        assert np.allclose(m.pair_solimp[i], ff["solimp"])
    ft = cp.PAIRS["fruit_tray"]
    for n in cp.TRAY_GEOMS:
        g = G(m, n)
        assert m.geom_priority[g] == ft["priority"] and m.geom_condim[g] == ft["condim"], n
        assert np.allclose(m.geom_solref[g], ft["solref"]) and np.allclose(m.geom_friction[g], ft["friction"])


def test_legacy_world_unchanged():
    m = build(True, world="legacy")
    assert m.npair == 0
    assert all(m.geom_priority[G(m, n)] == 0 for n in cp.TRAY_GEOMS)


def test_explicit_pair_replaces_dynamic_contact(tuned):
    m = tuned
    with free_fruit(m, 3) as (d, b, fg):
        close_on(m, d, b)
        pads = [c for c in d.contact[:d.ncon] if fg in (c.geom1, c.geom2)
                and {c.geom1, c.geom2} & {G(m, p) for p in cp.PAD_GEOMS}]
    assert len(pads) == 2                                   # one per pad, no duplicate dynamic contact
    for c in pads:
        assert c.dim == 4 and np.allclose(c.friction[:3], cp.PAIRS["finger_fruit"]["friction"][:3])
        assert np.allclose(c.solref, cp.PAIRS["finger_fruit"]["solref"])


# ---------------------------------------------------------------------- finger-fruit
@pytest.mark.parametrize("slot, mass", [(0, None), (1, 0.6), (5, 0.25)])
def test_slip_below_threshold_during_lift(tuned, slot, mass):
    r = lift(tuned, slot, mass)
    print(f"lift slot {slot} mass {mass or 'template'}: max slip {1000 * r['slip']:.2f} mm, "
          f"creep {1000 * r['creep']:.3f} mm/s, rise {100 * r['rise']:.1f} cm")
    assert r["rise"] > 0.08
    assert r["slip"] <= SLIP_MAX_M and r["creep"] <= CREEP_MAX_M_S


def test_torsional_friction_holds_off_axis_grip(tuned, untuned):
    tuned_deg, base_deg = offset_swing(tuned, 4), offset_swing(untuned, 4)
    print(f"off-axis grip under a 0.6 rad wrist pitch: fruit turns {tuned_deg:.2f} deg with the pair "
          f"(condim 4), {base_deg:.2f} deg with geom mixing (condim 3)")
    assert tuned_deg <= 3.0 and base_deg >= 15.0


# ---------------------------------------------------------------------- fruit-tray
@pytest.mark.parametrize("slot, drop, mass", [(5, 0.01, None), (6, 0.02, 0.3), (7, 0.02, 0.15)])
def test_no_jitter_in_tray(tuned, slot, drop, mass):
    r = tray_settle(tuned, slot, drop, mass)
    print(f"tray slot {slot} drop {100 * drop:.0f} cm: drift {1e6 * r['drift']:.1f} um, "
          f"max relative speed {1000 * r['speed']:.3f} mm/s, contacts {r['contacts']}")
    assert r["contacts"], "fruit not resting on the tray"
    for name, dim, solref in r["contacts"]:
        assert name in cp.TRAY_GEOMS and dim == cp.PAIRS["fruit_tray"]["condim"]
        assert np.allclose(solref, cp.PAIRS["fruit_tray"]["solref"])
    assert r["drift"] <= TRAY_DRIFT_MAX_M and r["speed"] <= TRAY_SPEED_MAX_M_S


# ---------------------------------------------------------------------- foot-soil
def test_foot_soil_params_pinned_and_live(tuned):
    m = tuned
    eff = cp.foot_soil_effective(m)
    assert eff["uniform"] and eff["ground"] == "orchard_ground"
    for k, v in cp.FOOT_SOIL_EXPECTED.items():
        assert np.allclose(eff[k], v), (k, eff[k])
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for a in range(m.nu):
        d.ctrl[a] = d.qpos[m.jnt_qposadr[m.actuator_trnid[a, 0]]]
    for _ in range(500):
        mujoco.mj_step(m, d)
    feet = {G(m, f): f for f in cp.FOOT_GEOMS}
    pos0 = {g: d.geom_xpos[g].copy() for g in feet}
    pen, n_live = 0.0, 0
    for _ in range(1000):
        mujoco.mj_step(m, d)
        for c in d.contact[:d.ncon]:
            if c.geom1 in feet or c.geom2 in feet:
                n_live += 1
                pen = max(pen, -float(c.dist))
                assert c.dim == eff["condim"] and np.allclose(c.solref, eff["solref"])
                assert np.allclose(c.friction, eff["friction"])
    slip = max(np.linalg.norm((d.geom_xpos[g] - pos0[g])[:2]) for g in feet)
    print(f"foot-soil: {n_live} live contacts checked, max penetration {1000 * pen:.2f} mm, "
          f"stance foot slip over 2 s {1000 * slip:.3f} mm")
    assert n_live > 0 and pen <= 0.001 and slip <= 0.001


def test_mixing_helper_matches_live_contacts(untuned):
    """mixed_params reproduces MuJoCo's geom mixing (priority pad vs fruit; equal priority on the legacy floor)."""
    m = untuned
    pad, fruit = G(m, "finger_left_pad"), G(m, "pool_0_fruit")
    mix = cp.mixed_params(m, pad, fruit)
    assert mix["condim"] == 3 and np.allclose(mix["solref"], m.geom_solref[pad])      # the pad has priority 1
    assert mix["friction"][0] == pytest.approx(m.geom_friction[pad][0])
    legacy = build(True, world="legacy")
    eff = cp.foot_soil_effective(legacy)
    assert eff["ground"] == "floor" and eff["uniform"] and eff["condim"] == 3
