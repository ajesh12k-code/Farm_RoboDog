"""WP23 fixed-slot fruit pool (ORCHARD-MVP-SPEC 3.4): slot kinds, anchor frames, qpos0 fill, damping and settle,
cut_k on the fruit, contact pairs, assign / release / recycle, the pool and hidden_geoms sim-state providers, and
the twist-release reference statistic. Headless, ROS-free, each test well under 10 s on the 2-CPU sim runner."""
import contextlib
import os

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core import fruit_pool as fp
from atharv_nandi.core import sim_state
from atharv_nandi.sim.world import compose
from atharv_nandi.sim.world.modules import fruit_pool as pool_mod
from atharv_nandi.sim.world.modules import trees as mt

G = 9.81


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        os.environ.update({k: str(v) for k, v in kw.items()})
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_lite():
    # OT-1 (WP0c anchors) is not in WP4's orchard_lite (L1-1..L2-4), so the pool tests compose orchard_demo
    with env(NANDI_WORLD="orchard_demo"):
        cfg = farm_scene.world_config()
        w = compose.build(cfg["world"], cfg["seed"], farm_scene.ROBOT_XML, cfg)
        return w.spec.compile(), mt.current_plans(), w


@pytest.fixture(scope="module")
def world():
    m, plans, w = build_lite()
    return m, plans, w


@pytest.fixture()
def pool(world):
    m, plans, _ = world
    p = fp.FruitPool(m, plans)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    p.reset(d)
    yield p, d
    p.reset(d)


def pick(p, crop, tree=None, mature=True, in_band=True):
    return next(f for f in p.fruit.values() if f["crop"] == crop and (tree is None or f["tree_id"] == tree)
                and f["harvest_stage"] == mature and f["in_band"] == in_band)


def step(m, d, seconds):
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)


# ---------------------------------------------------------------------- layout
def test_ten_fixed_slots_5_connect_5_weld(world):
    m = world[0]
    for k in range(10):
        n = pool_mod.names(k)
        eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n["eq"])
        want = mujoco.mjtEq.mjEQ_CONNECT if k < 5 else mujoco.mjtEq.mjEQ_WELD
        assert m.eq_type[eq] == want and m.eq_obj2id[eq] == 0          # to the world
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n["body"])
        assert m.body_parentid[b] == 0 and m.body_jntnum[b] == 1
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n["joint"])
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        da = m.jnt_dofadr[j]
        assert np.allclose(m.dof_damping[da:da + 6], pool_mod.POOL_DAMPING)
        assert np.allclose(m.dof_armature[da:da + 6], pool_mod.POOL_ARMATURE)
        for site in ("cut_k", "stem_end", "anchor"):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n[site])
            assert m.site_bodyid[sid] == b                              # pedicel sites live on the fruit body
    # WP2 adds its own tool welds (rack / grip / catch) to orchard worlds: count the pool's equalities only
    pool_eqs = [e for e in range(m.neq) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "").startswith("pool_eq_")]
    assert len(pool_eqs) == 10
    # equality rows at rest: 5 connect x 3 + 5 weld x 6 = 45 (parked slots stay held)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    assert int(np.sum((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(d.efc_id, pool_eqs))) == 45


def test_qpos0_fill_no_nan_after_500_steps(world):
    m = world[0]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for k in range(10):
        s = pool_mod.SlotIds(m, k)
        np.testing.assert_allclose(m.key_qpos[0][s.qadr:s.qadr + 7], pool_mod.park_pos(k) + [1, 0, 0, 0])
    p0 = d.qpos.copy()
    step(m, d, 1.0)                                      # 500 steps at dt 0.002
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
    for k in range(10):
        s = pool_mod.SlotIds(m, k)
        assert np.linalg.norm(d.qpos[s.qadr:s.qadr + 3] - p0[s.qadr:s.qadr + 3]) < 1e-3   # parked, held


# ---------------------------------------------------------------------- anchor frames
@pytest.mark.parametrize("crop", ["mango", "guava", "papaya"])
def test_anchor_frames_no_false_torque_at_t0(pool, crop):
    p, d = pool
    f = pick(p, crop)
    k = p.assign(d, f["fruit_id"])
    assert p.anchor_error_m(d, k) < 1e-9
    s = p.slots[k]
    assert np.linalg.norm(d.qfrc_constraint[s.dadr + 3:s.dadr + 6]) < 1e-6    # no moment about the CoM
    F, tau = p.reaction(d, k)
    mg = f["mass_kg"] * G
    assert F[2] > 0.8 * mg and np.linalg.norm(F[:2]) < 1e-6           # soft constraint ramps to m g
    if crop == "mango":
        assert tau is None and p.slots[k].kind == "connect"
    else:
        assert np.linalg.norm(tau) < 1e-6 and p.slots[k].kind == "weld"
    step(p.m, d, 1.0)
    F, tau = p.reaction(d, k)
    assert abs(F[2] - mg) < 0.01 * mg and p.anchor_error_m(d, k) < 2e-4
    if tau is not None:
        assert np.linalg.norm(tau) < 1e-6


def test_wrong_anchor_frames_give_a_false_torque(pool):
    """The gotcha the frame rule prevents (OS 3.4, treesim probe: false 19.5 N.m at t = 0).

    Weld: the CoM written in the fruit (body1) frame, i.e. (0, 0, 0), lands at the WORLD origin when read as a
    body2 (world) anchor; with a consistent relpose the weld holds, but gravity now has a lever of metres about
    the anchor. Connect: a world-frame anchor read in the body1 frame puts the attachment point metres off the
    fruit. Both show as a large moment at t = 0; the correct frames give none (test above)."""
    p, d = pool
    f = pick(p, "papaya")
    k = p.assign(d, f["fruit_id"])
    s = p.slots[k]
    e = p.m.eq_data[s.eq].copy()
    centre = np.asarray(f["centre_world"])
    p.m.eq_data[s.eq][0:3] = [0.0, 0.0, 0.0]
    p.m.eq_data[s.eq][3:6] = -centre                  # relpose consistent with the (wrong) world anchor
    try:
        mujoco.mj_forward(p.m, d)
        step(p.m, d, 0.5)                             # once the soft weld carries the load (t0 is a ramp)
        _, tau = p.reaction(d, k)
        assert np.linalg.norm(tau) > 10.0             # gravity lever about the far anchor: ~34 N.m measured
        assert np.linalg.norm(d.xpos[s.body] - centre) > 0.05     # and the fruit sags off its tree
    finally:
        p.m.eq_data[s.eq] = e
    fm = pick(p, "mango", "OT-1")
    km = p.assign(d, fm["fruit_id"])
    sm = p.slots[km]
    good = np.array(d.qfrc_constraint[sm.dadr + 3:sm.dadr + 6])
    assert np.linalg.norm(good) < 1e-6                # correct connect: no moment about the CoM
    em = p.m.eq_data[sm.eq].copy()
    p.m.eq_data[sm.eq][0:3] = fm["anchor_top_world"]  # world anchor written where the body1-frame one belongs
    try:
        mujoco.mj_forward(p.m, d)
        step(p.m, d, 0.5)
        assert np.linalg.norm(d.xpos[sm.body] - fm["centre_world"]) > 0.05    # yanked around a far pivot
    finally:
        p.m.eq_data[sm.eq] = em


def test_weld_torque_readout_scale(pool):
    p, d = pool
    f = pick(p, "guava")
    k = p.assign(d, f["fruit_id"])
    s = p.slots[k]
    d.xfrc_applied[s.body, 3:6] = [0.0, 0.0, 0.4]       # 0.4 N.m about the stem axis (world z)
    step(p.m, d, 1.0)
    assert abs(p.stem_axis_torque(d, k) + 0.4) < 0.01  # the stem holds against it
    d.xfrc_applied[s.body] = 0.0
    with pytest.raises(fp.PoolError):
        p.stem_axis_torque(d, p.assign(d, pick(p, "mango")["fruit_id"]))


# ---------------------------------------------------------------------- bumped mango, cut_k
def test_bumped_mango_settles_within_1s_and_cut_k_moves_with_the_fruit(pool):
    p, d = pool
    f = pick(p, "mango", "OT-1")
    k = p.assign(d, f["fruit_id"])
    s = p.slots[k]
    step(p.m, d, 0.5)
    local = np.array(p.m.site_pos[s.cut_k])
    c0 = p.cut_point(d, k)
    np.testing.assert_allclose(c0, f["cut_k_world"], atol=1e-3)
    seen = []
    p.bump(d, k, [2.5, 0.0, 0.0], 0.02, steps_cb=lambda dd: seen.append(p.cut_point(dd, k)))  # 0.05 N.s
    rates, moved = [], 0.0
    for _ in range(int(1.5 / p.m.opt.timestep)):
        mujoco.mj_step(p.m, d)
        rates.append(p.angular_rate_dps(d, k))
        c = p.cut_point(d, k)
        moved = max(moved, float(np.linalg.norm(c - c0)))
        R = np.array(d.xmat[s.body]).reshape(3, 3)
        np.testing.assert_allclose(c, d.xpos[s.body] + R @ local, atol=1e-9)   # cut_k rides on the fruit body
        assert p.anchor_error_m(d, k) < 1e-3                                   # connect keeps the anchor
    assert max(rates) > 50.0                              # it really swung
    n1 = int(1.0 / p.m.opt.timestep)
    assert max(rates[n1:]) < 5.0, max(rates[n1:])        # < 5 deg/s from 1 s after the bump on, and it stays
    assert moved > 1e-3                                   # cut_k moved with the swinging fruit
    # the pedicel error is measured in the fruit frame: a cut exactly at cut_k is 0 mm, 5 mm above is 5 mm
    assert p.pedicel_error_mm(d, k, p.cut_point(d, k)) < 1e-6
    R = np.array(d.xmat[s.body]).reshape(3, 3)
    assert abs(p.pedicel_error_mm(d, k, p.cut_point(d, k) + R @ np.array([0, 0, 0.005])) - 5.0) < 1e-6


def _demo_harvest_mangoes(p):
    return [f for f in p.fruit.values() if f["crop"] == "mango" and f["in_band"] and f["harvest_stage"]
            and not f["bagged"]]


@pytest.mark.parametrize("impulse_ns", [0.05, 0.2])
def test_every_demo_mango_settles_within_1s(impulse_ns):
    """OS 3.4 / WP23 gate over every in-band harvest-stage mango of orchard_demo (not one fruit): after a lateral
    bump the angular rate drops below 5 deg/s within 1 s and stays there (settle_s = last time >= 5 deg/s)."""
    with env(NANDI_WORLD="orchard_demo"):
        cfg = farm_scene.world_config()
        w = compose.build(cfg["world"], cfg["seed"], farm_scene.ROBOT_XML, cfg)
        m, plans = w.spec.compile(), mt.current_plans()
    p = fp.FruitPool(m, plans)
    d = mujoco.MjData(m)
    mangoes = _demo_harvest_mangoes(p)
    assert len(mangoes) == 15                              # OT-1 3 + R1-1..R1-4 3 each
    worst = {}
    for f in mangoes:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        p.reset(d)
        k = p.assign(d, f["fruit_id"])
        step(m, d, 0.5)
        r = fp.settle_time_after_bump(p, d, k, impulse_ns=impulse_ns, horizon_s=1.5)
        assert r["peak_dps"] > 50.0, f["fruit_id"]            # it really swung
        assert r["max_anchor_err_m"] < 1e-3, f["fruit_id"]
        worst[f["fruit_id"]] = r["settle_s"]
    bad = {k: v for k, v in worst.items() if v >= 1.0}
    assert not bad, bad


def test_spec_damping_0_002_would_not_settle(pool):
    """Why POOL_DAMPING is 0.02, not the OS 3.4 assumption 0.002 (recorded in the WP23 contract notes)."""
    p, d = pool
    k = p.assign(d, pick(p, "mango", "OT-1")["fruit_id"])
    s = p.slots[k]
    keep = p.m.dof_damping[s.dadr:s.dadr + 6].copy()
    try:
        p.m.dof_damping[s.dadr + 3:s.dadr + 6] = 0.002
        step(p.m, d, 0.5)
        r = fp.settle_time_after_bump(p, d, k, impulse_ns=0.05, horizon_s=1.0)
        assert r["rate_at_1s_dps"] > 5.0
    finally:
        p.m.dof_damping[s.dadr:s.dadr + 6] = keep


# ---------------------------------------------------------------------- contacts
def test_fruit_wood_pairs_never_collide_robot_fruit_do(pool):
    p, d = pool
    m = p.m
    tree_geoms = [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                  m.geom_bodyid[g]) or "").startswith("tree_")]
    f = pick(p, "mango", "OT-1")
    k = p.assign(d, f["fruit_id"])
    fg = p.slots[k].fruit
    for g in tree_geoms:                                   # bitmask filter: (ct1 & ca2) | (ct2 & ca1) == 0
        assert (m.geom_contype[fg] & m.geom_conaffinity[g]) | (m.geom_contype[g] & m.geom_conaffinity[fg]) == 0
    # teleport the (released) fruit into the OT-1 trunk capsule: no contact
    p.release(d, k)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "tree_OT-1_wood_0")
    s = p.slots[k]
    d.qpos[s.qadr:s.qadr + 3] = d.geom_xpos[trunk] if d.geom_xpos[trunk].any() else [6.0, -3.5, 0.4]
    mujoco.mj_forward(m, d)
    pairs = {(int(c.geom1), int(c.geom2)) for c in d.contact[:d.ncon]}
    assert not any(fg in pr and (set(pr) - {fg}) & set(tree_geoms) for pr in pairs)
    # into the robot trunk: contact
    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    rgeoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == rb and m.geom_contype[g] and m.geom_conaffinity[g]]
    assert rgeoms
    top = np.array(d.geom_xpos[rgeoms[0]]) + [0.0, 0.0, float(m.geom_size[rgeoms[0]][2]) + 0.02]
    d.qpos[s.qadr:s.qadr + 3] = top                       # resting into the top face of the robot trunk box
    mujoco.mj_forward(m, d)
    pairs = {(int(c.geom1), int(c.geom2)) for c in d.contact[:d.ncon]}
    assert any(fg in pr for pr in pairs), "robot-fruit must collide"


def test_released_mango_falls_and_parked_slots_never_touch(pool):
    p, d = pool
    f = pick(p, "mango", "OT-1")
    k = p.assign(d, f["fruit_id"])
    step(p.m, d, 0.2)
    p.release(d, k)                                        # the logical cut
    assert not p.is_attached(d, k)
    z0 = float(d.xpos[p.slots[k].body][2])
    step(p.m, d, 0.2)
    assert float(d.xpos[p.slots[k].body][2]) < z0 - 0.1
    for s in p.slots:
        if s.k != k:
            assert p.m.geom_contype[s.fruit] == 0 and p.m.geom_rgba[s.fruit, 3] == 0.0


@pytest.mark.parametrize("crop", ["mango", "guava", "papaya"])
def test_assigned_fruit_has_its_own_mass_matrix_and_falls_at_g(pool, crop):
    """Pool bodies are MuJoCo simple bodies: the mass matrix is the compiled dof_M0, so assign must write it (WP3
    change request: a released 1.54 kg papaya fell at 100.8 m/s^2 with the 0.15 kg template)."""
    p, d = pool
    m = p.m
    f = pick(p, crop, "OT-1" if crop == "mango" else None)
    k = p.assign(d, f["fruit_id"])
    s = p.slots[k]
    assert m.body_simple[s.body] == 1
    arm = m.dof_armature[s.dadr:s.dadr + 6]
    np.testing.assert_allclose(m.dof_M0[s.dadr:s.dadr + 3], f["mass_kg"] + arm[:3], rtol=1e-12)
    np.testing.assert_allclose(m.dof_M0[s.dadr + 3:s.dadr + 6], m.body_inertia[s.body] + arm[3:], rtol=1e-12)
    p.release(d, k)
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    np.testing.assert_allclose(np.diag(M)[s.dadr:s.dadr + 6], m.dof_M0[s.dadr:s.dadr + 6], rtol=1e-9)
    assert abs(float(d.qacc[s.dadr + 2]) + G) < 0.05          # free fall at g (damping and armature aside)
    tmpl = p._park[k]["body"]["dof_M0"]
    p.recycle(d, k)
    np.testing.assert_array_equal(m.dof_M0[s.dadr:s.dadr + 6], tmpl)    # parked template restored


def test_pool_payload_without_dof_m0_still_applies(pool):
    """A v1 pool payload packed before dof_M0 was added is accepted and the mass matrix derived from mass/inertia;
    a non-positive dof_M0 is refused before anything is written."""
    p, d = pool
    m = p.m
    k = p.assign(d, pick(p, "papaya")["fruit_id"])
    a = p.slots[k].dadr
    good = pool_mod._pack_pool(m)
    want = m.dof_M0[a:a + 6].copy()
    import copy as _copy
    old = _copy.deepcopy(good)
    for item in old["slots"]:
        item["fields"]["body"].pop("dof_M0")
    m.dof_M0[a:a + 6] = 1.0
    pool_mod._unpack_pool(m, old)
    np.testing.assert_allclose(m.dof_M0[a:a + 6], want, rtol=1e-12)
    bad = _copy.deepcopy(good)
    bad["slots"][k]["fields"]["body"]["dof_M0"] = [0.0] * 6
    m.dof_M0[a:a + 6] = 2.0
    with pytest.raises(ValueError):
        pool_mod._unpack_pool(m, bad)
    np.testing.assert_array_equal(m.dof_M0[a:a + 6], 2.0)
    pool_mod._unpack_pool(m, good)
    np.testing.assert_array_equal(m.dof_M0[a:a + 6], want)


# ---------------------------------------------------------------------- assign / release / recycle
def _fields(m):
    return {k: np.array(getattr(m, k)).copy() for k in ("geom_rgba", "geom_contype", "geom_conaffinity", "geom_size",
            "geom_pos", "geom_quat", "geom_rbound", "geom_aabb", "eq_data", "site_pos", "body_mass", "body_inertia",
            "body_subtreemass", "body_invweight0", "dof_invweight0", "dof_M0", "body_contype", "body_conaffinity")}


def test_assign_release_recycle_restores_statics(pool):
    p, d = pool
    before = _fields(p.m)
    ids = [pick(p, "mango", "OT-1")["fruit_id"], pick(p, "guava")["fruit_id"], pick(p, "papaya")["fruit_id"]]
    slots = [p.assign(d, fid) for fid in ids]
    for fid in ids:
        assert p.m.geom_rgba[p.static_geom(fid), 3] == 0.0 and p.static_geom(fid) in mt.hidden(p.m)
    step(p.m, d, 0.1)
    for k in slots:
        p.release(d, k)
    step(p.m, d, 0.1)
    for k in slots:
        p.recycle(d, k, harvested=False)
    after = _fields(p.m)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)
    assert mt.hidden(p.m) == [] and p.assignments == {}
    # harvested: the static stays hidden after recycle, and reset brings it back
    k = p.assign(d, ids[0])
    p.recycle(d, k, harvested=True)
    assert p.m.geom_rgba[p.static_geom(ids[0]), 3] == 0.0 and ids[0] in p.harvested
    with pytest.raises(fp.PoolError):
        p.assign(d, ids[0])
    p.reset(d)
    np.testing.assert_array_equal(_fields(p.m)["geom_rgba"], before["geom_rgba"])


def test_assignment_rules(pool):
    p, d = pool
    mango = [f["fruit_id"] for f in p.fruit.values() if f["crop"] == "mango"]
    for fid in mango[:5]:
        assert p.slots[p.assign(d, fid, forward=False)].kind == "connect"
    with pytest.raises(fp.PoolError, match="no free connect slot"):
        p.assign(d, mango[5])
    p.reset(d)
    with pytest.raises(fp.PoolError, match="already"):
        k = p.assign(d, mango[0])
        p.assign(d, mango[0])
    banana_fruit = [f for f in p.fruit.values() if f["crop"] == "banana"]
    assert banana_fruit == []                              # the bunch is static truth, never a pool fruit
    with pytest.raises(fp.PoolError):
        p.assign(d, "nope:f01")
    bagged = dict(p.fruit[mango[1]], bagged=True)
    p.fruit["bag:f01"] = bagged
    p._static_gid["bag:f01"] = p._static_gid[mango[1]]
    with pytest.raises(fp.PoolError, match="bagged"):
        p.assign(d, "bag:f01")
    del k


def test_assign_tree_takes_all_mature_in_band_fruit(pool):
    p, d = pool
    got = p.assign_tree(d, "OT-1")
    want = sorted(f["fruit_id"] for f in p.fruit.values()
                  if f["tree_id"] == "OT-1" and f["in_band"] and f["harvest_stage"])
    assert sorted(got) == want and len(want) == 3            # OS 1.2 step 5
    assert sorted(p.slots[k].kind for k in got.values()) == ["connect"] * 3


def test_assign_tree_is_all_or_nothing(pool):
    p, d = pool
    m = p.m
    mango = [f["fruit_id"] for f in p.fruit.values() if f["crop"] == "mango" and not f["bagged"]]
    before = (m.geom_rgba.copy(), m.eq_data.copy(), d.qpos.copy())
    with pytest.raises(fp.PoolError, match="no free connect slot"):
        p.assign_tree(d, None, mango[:6])                      # a 6th mango: refused up front
    assert p.assignments == {} and mt.hidden(m) == []
    np.testing.assert_array_equal(m.geom_rgba, before[0])
    np.testing.assert_array_equal(m.eq_data, before[1])
    np.testing.assert_array_equal(d.qpos, before[2])
    with pytest.raises(fp.PoolError, match="twice"):
        p.assign_tree(d, None, [mango[0], mango[0]])
    p.recycle(d, p.assign(d, mango[2]), harvested=True)          # a harvested fruit in the list
    with pytest.raises(fp.PoolError, match="already"):
        p.assign_tree(d, None, mango[:2] + [mango[2]])
    assert p.assignments == {} and mt.hidden(m) == [p.static_geom(mango[2])]
    # a failure inside assign (after the checks) rolls back what was already assigned, and runs mj_forward
    real = p.assign
    calls = []

    def flaky(dd, fid, forward=True):
        calls.append(fid)
        if len(calls) == 2:
            raise fp.PoolError("injected")
        return real(dd, fid, forward=forward)
    p.assign = flaky
    try:
        with pytest.raises(fp.PoolError, match="injected"):
            p.assign_tree(d, None, mango[:2])
    finally:
        del p.assign
    assert p.assignments == {} and mt.hidden(m) == [p.static_geom(mango[2])]
    s0 = p.slots[0]
    np.testing.assert_allclose(d.xpos[s0.body], pool_mod.park_pos(0), atol=1e-9)   # forward ran after the rollback
    got = p.assign_tree(d, None, mango[:2])
    assert sorted(got) == sorted(mango[:2]) and len(p.assignments) == 2


# ---------------------------------------------------------------------- sim-state providers
def test_pool_and_hidden_geoms_round_trip_through_sim_state(pool):
    p, d = pool
    sim_state.discover_providers()
    ids = [pick(p, "mango", "OT-1")["fruit_id"], pick(p, "papaya")["fruit_id"]]
    for fid in ids:
        p.assign(d, fid)
    step(p.m, d, 0.05)
    wid = "test-wp23"
    msg = sim_state.decode(sim_state.encode(sim_state.pack(p.m, d, wid)))
    m2 = build_lite()[0]                                    # a renderer that built the same world
    d2 = mujoco.MjData(m2)
    ok, why = sim_state.apply(m2, d2, msg, wid)
    assert ok, why
    mujoco.mj_kinematics(m2, d2)
    a, b = _fields(p.m), _fields(m2)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    np.testing.assert_array_equal(d.qpos, d2.qpos)
    assert mt.hidden(m2) == mt.hidden(p.m) and len(mt.hidden(m2)) == 2
    mujoco.mj_kinematics(p.m, d)
    p2 = fp.FruitPool(m2, p.plans)
    assert p2.assignments == p.assignments
    np.testing.assert_allclose(p2.cut_point(d2, p2.slot_of(ids[0])), p.cut_point(d, p.slot_of(ids[0])), atol=1e-12)
    # back to parked: the receiver un-hides and re-parks too
    p.reset(d)
    ok, why = sim_state.apply(m2, d2, sim_state.pack(p.m, d, wid), wid)
    assert ok, why and mt.hidden(m2) == []
    np.testing.assert_array_equal(_fields(p.m)["geom_rgba"], _fields(m2)["geom_rgba"])
    # malformed payloads are refused without writing
    bad = sim_state.pack(p.m, d, wid)
    bad["mutations"]["providers"]["hidden_geoms"]["data"] = {"geoms": [0]}     # geom 0 is the floor
    ok, why = sim_state.apply(m2, d2, bad, wid)
    assert not ok and "hidden_geoms" in why
    bad = sim_state.pack(p.m, d, wid)
    bad["mutations"]["providers"]["pool"]["data"]["slots"] = bad["mutations"]["providers"]["pool"]["data"]["slots"][:3]
    ok, why = sim_state.apply(m2, d2, bad, wid)
    assert not ok and "pool" in why


def test_unpack_pool_validates_before_writing(pool):
    """A malformed pool payload raises in _unpack_pool itself (not only via sim_state's rollback), writes nothing."""
    p, d = pool
    m = p.m
    p.assign(d, pick(p, "mango", "OT-1")["fruit_id"])
    good = pool_mod._pack_pool(m)
    snap = (m.eq_data.copy(), m.geom_size.copy(), dict(p.assignments), list(p.harvested))
    import copy as _copy

    def broken(fn):
        x = _copy.deepcopy(good)
        fn(x)
        return x
    cases = {
        "harvested string": broken(lambda x: x.update(harvested="OT-1:f01")),
        "harvested ints": broken(lambda x: x.update(harvested=[1, 2])),
        "missing field key": broken(lambda x: x["slots"][7]["fields"]["fruit"].pop("rbound")),
        "short eq_data": broken(lambda x: x["slots"][9]["fields"].update(eq_data=[0.0] * 3)),
        "nan size": broken(lambda x: x["slots"][8]["fields"]["fruit"].update(size=[float("nan")] * 3)),
        "fields not a dict": broken(lambda x: x["slots"][6].update(fields=[1, 2])),
        "repeated slot": broken(lambda x: x["slots"][1].update(slot=0, eq_id=x["slots"][0]["eq_id"])),
        "bool slot": broken(lambda x: x["slots"][0].update(slot=True)),
        "fruit_id not str": broken(lambda x: x["slots"][2].update(fruit_id=5)),
    }
    for name, payload in cases.items():
        with pytest.raises(ValueError):
            pool_mod._unpack_pool(m, payload)
        np.testing.assert_array_equal(m.eq_data, snap[0], err_msg=name)
        np.testing.assert_array_equal(m.geom_size, snap[1], err_msg=name)
        assert p.assignments == snap[2] and p.harvested == snap[3], name
    pool_mod._unpack_pool(m, good)                              # the good one still applies
    assert p.assignments == snap[2]


# ---------------------------------------------------------------------- twist release reference statistic
def test_hanging_papaya_never_releases_and_no_roll_never_releases(pool):
    p, d = pool
    f = pick(p, "papaya")
    k = p.assign(d, f["fruit_id"])
    tau = 0.3
    stat = fp.TwistReleaseStat(tau_detach=tau, dt=p.m.opt.timestep)
    for _ in range(int(10.0 / p.m.opt.timestep)):          # 10 s untouched
        mujoco.mj_step(p.m, d)
        assert not stat.update(p.stem_axis_torque(d, k), 0.0)
    assert stat.peak < 1e-3
    # a torque far above tau about the stem axis but no relative roll (jaws closed, no roll): never releases
    stat = fp.TwistReleaseStat(tau_detach=tau, dt=p.m.opt.timestep)
    stat.update(p.stem_axis_torque(d, k), 0.0)
    d.xfrc_applied[p.slots[k].body, 3:6] = [0.0, 0.0, 1.0]
    fired = False
    for _ in range(500):
        mujoco.mj_step(p.m, d)
        fired |= stat.update(p.stem_axis_torque(d, k), 0.0)
    assert not fired and stat.above >= 3                    # above tau, held back only by the roll gate
    assert stat.update(p.stem_axis_torque(d, k), 35.0)      # with >= 30 deg roll it fires
    d.xfrc_applied[p.slots[k].body] = 0.0


def test_tau_sweep_short_run(world):
    out = fp.tau_sweep(taus=(0.1, 0.3), hold_s=0.5, m=world[0])
    crops_ = {r["crop"]: r for r in out["fruit"]}
    assert set(crops_) == {"guava", "papaya"}
    for r in crops_.values():
        assert r["hold_peak_nm"] < 1e-3
        assert r["twist_fire_s"]["0.1"] == pytest.approx(1.0 + 0.048, abs=0.01)   # ramp 0.1 N.m/s + 40 ms LPF
    assert crops_["papaya"]["bump_peak_nm"] > crops_["guava"]["bump_peak_nm"]     # bigger radius, bigger moment
    assert crops_["papaya"]["recommended_tau_nm"] == 0.3
