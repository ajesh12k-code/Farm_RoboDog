"""WPCAP arm motion planning: MuJoCo collision scene + OMPL / pure-Python RRT-Connect (docs/cap-libraries.md 1).

Headless, ROS-free. The OMPL backend tests skip when ``import ompl`` fails (CI venv without the wheel); the
``rrtconnect`` fallback runs everywhere. Pick scenes are the WP0c frozen targets (config/fields/orchard_candidates.json)
in orchard_demo, with a 6 cm "branch" (mocap sphere, wood class) parked on the straight joint-space path so the
planner has to detour.
"""
import contextlib
import importlib
import json
import math
import os
import sys
import time

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core.planning import ArmPlanner, CollisionScene, ompl_planner, time_parameterize
from atharv_nandi.sim.world import compose

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROBE = "tree_wpcap_probe"
PICK_TASKS = ("mango_pedicel_cut", "guava_clip", "guava_cradle", "papaya_cradle")
BACKENDS = [pytest.param("ompl", marks=pytest.mark.skipif(not ompl_planner.ompl_available(),
                                                          reason="ompl not importable")), "rrtconnect"]


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _probe_module(spec, field, ctx):
    b = spec.worldbody.add_body(name=PROBE, mocap=True, pos=[30.0, 30.0, 0.5])
    b.add_geom(name=PROBE + "_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.06, 0, 0], contype=1, conaffinity=0,
               group=0)


@pytest.fixture(scope="module")
def world():
    with env(NANDI_WORLD="orchard_demo"), compose.temporary_module("wpcap_probe", 90, _probe_module):
        m = farm_scene.load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d, d.qpos.copy()


def timing_factor() -> float:
    """Wall-clock slack for plan-time asserts: ``NANDI_TIMING_FACTOR`` (default 1), x3 under coverage / tracing.

    The 1 s budget is the WPCAP spec on an idle machine; a loaded shared runner or pytest-cov tracing slows the
    pure-Python checker, so the assert and the planner budget scale together instead of flaking."""
    f = float(os.environ.get("NANDI_TIMING_FACTOR", "1") or 1.0)
    if sys.gettrace() is not None or "coverage" in sys.modules:
        f *= 3.0
    return max(1.0, f)


def pick_targets(per_pair: int):
    cand = json.load(open(os.path.join(REPO, "config", "fields", "orchard_candidates.json")))
    out = [t for t in cand["frozen"]["targets"] if t["status"] == "FROZEN" and t["task"] in PICK_TASKS]
    seen, picked = {}, []
    for t in out:
        k = (t["tree"], t["task"])
        if seen.get(k, 0) < per_pair:
            seen[k] = seen.get(k, 0) + 1
            picked.append(t)
    return picked


@pytest.fixture(scope="module")
def targets():
    """One frozen pick target per (tree, pick task): every pick tree and crop of orchard_demo (17)."""
    return pick_targets(1)


def station_qpos(q0, st):
    q = q0.copy()
    h = math.radians(st["heading_deg"])
    q[0], q[1] = st["xy"]
    q[3:7] = [math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)]
    return q


def mocap_index(m):
    return int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PROBE)])


def place_probe(m, d, sc, start, goal):
    """Probe sphere at the tip position halfway along the straight joint path (blocks the direct move)."""
    mi = mocap_index(m)
    d.mocap_pos[mi] = [30.0, 30.0, 0.5]
    sc.sync(m, d)
    d.mocap_pos[mi] = sc.tip(0.5 * (start + goal))
    sc.sync(m, d)


def park_probe(m, d):
    d.mocap_pos[mocap_index(m)] = [30.0, 30.0, 0.5]


# ---------------------------------------------------------------------- collision scene
def test_scene_classes_and_home_valid(world):
    m, d, q0 = world
    sc = CollisionScene(m)
    sc.sync(m, d)
    counts = sc.obstacle_counts()
    for cls in ("wood", "robot", "keepout", "ground", "carriage"):
        assert counts.get(cls, 0) > 0, counts
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in sc.moving_geoms}
    assert {"gripper_palm", "finger_left_pad", "finger_right_pad"} <= names and len(sc.moving_geoms) == 6
    home = q0[sc.qadr]
    assert sc.is_valid(home), sc.contacts(home)
    assert sc.d.ncon == 0                     # legs on the ground etc. are never generated
    cl = sc.clearance(home)
    assert cl["carriage"] > 0.0 and cl["robot"] >= sc.margins["robot"]


def test_scene_detects_self_keepout_and_ground(world):
    m, d, q0 = world
    sc = CollisionScene(m)
    sc.sync(m, d)
    rng = np.random.default_rng(3)
    lo, hi = sc.limits[:, 0], sc.limits[:, 1]
    found = set()
    for _ in range(3000):
        q = lo + (hi - lo) * rng.random(5)
        if not sc.is_valid(q):
            found.update(c["class"] for c in sc.contacts(q))
        if {"robot", "keepout", "ground", "carriage"} <= found:
            break
    assert {"robot", "keepout", "ground", "carriage"} <= found, found


def test_live_model_untouched(world):
    m, d, q0 = world
    before = (m.geom_contype.copy(), m.geom_conaffinity.copy(), m.geom_margin.copy(), m.body_contype.copy(),
              m.pair_margin.copy())
    sc = CollisionScene(m)
    sc.sync(m, d)
    sc.is_valid(q0[sc.qadr])
    after = (m.geom_contype, m.geom_conaffinity, m.geom_margin, m.body_contype, m.pair_margin)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


# ---------------------------------------------------------------------- planning
def _pick_campaign(world, targets, backend):
    m, d, q0 = world
    budget = 1.0 * timing_factor()
    planner = ArmPlanner(m, backend=backend, time_limit_s=budget)
    sc = planner.scene
    # independent checker (own copy, own data): physical contact (all margins 0) at half the verify step
    checker = CollisionScene(m, margins={k: 0.0 for k in planner.scene.margins})
    times, waypoints, clear = [], [], []
    for t in targets:
        d.qpos[:] = station_qpos(q0, t["station"])
        start, goal = d.qpos[sc.qadr].copy(), np.array(t["q"])
        place_probe(m, d, sc, start, goal)
        assert not sc.motion_valid(start, goal, 0.005), t["id"]   # the direct move is blocked
        res = planner.plan(None, goal, data=d)
        assert res.ok, (t["id"], res.reason, res.contacts)
        assert res.backend == backend
        assert np.allclose(res.path[0], start) and np.allclose(res.path[-1], goal)
        checker.sync(m, d)
        assert all(checker.motion_valid(a, b, 0.0025) for a, b in zip(res.path[:-1], res.path[1:])), t["id"]
        assert res.min_clearance["wood"] >= planner.scene.margins["wood"] - 2e-3    # margin, sampled every 0.02
        times.append(res.plan_time_s)
        waypoints.append(len(res.path))
        clear.append(res.min_clearance["wood"])
    park_probe(m, d)
    print(f"{backend}: {len(times)} picks, plan time p50 {np.median(times):.3f} s, p95 {np.percentile(times, 95):.3f} s, "
          f"max {max(times):.3f} s; waypoints mean {np.mean(waypoints):.1f}; min wood clearance {min(clear):.4f} m")
    assert max(times) <= budget, (max(times), budget)


@pytest.mark.parametrize("backend", BACKENDS)
def test_typical_pick_plans_collision_free_under_1s(world, targets, backend):
    assert len(targets) == 17
    _pick_campaign(world, targets, backend)


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
def test_pick_campaign_two_per_tree_task(world, backend):
    targets = pick_targets(2)
    assert len(targets) == 34
    _pick_campaign(world, targets, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_free_move_is_direct(world, targets, backend):
    m, d, q0 = world
    park_probe(m, d)
    planner = ArmPlanner(m, backend=backend)
    t = targets[0]
    d.qpos[:] = station_qpos(q0, t["station"])
    res = planner.plan(None, np.array(t["q"]), data=d)
    assert res.ok and len(res.path) == 2 and res.trajectory.duration > 0
    assert res.to_dict()["waypoints"] == 2


def test_goal_in_collision_refused(world, targets):
    m, d, q0 = world
    planner = ArmPlanner(m, backend="rrtconnect")
    t = targets[0]
    d.qpos[:] = station_qpos(q0, t["station"])
    planner.scene.sync(m, d)
    d.mocap_pos[mocap_index(m)] = planner.scene.tip(np.array(t["q"]))
    res = planner.plan(None, np.array(t["q"]), data=d)
    park_probe(m, d)
    assert not res.ok and res.reason == "goal_invalid"
    assert any(c["class"] == "wood" and c["obstacle"] == PROBE + "_g" for c in res.contacts)
    out = planner.plan(None, np.array(t["q"]) + np.array([0, 0, 10.0, 0, 0]), data=d)
    assert out.reason == "goal_invalid" and out.contacts[0]["class"] == "limits"


def _assign_fruit(m, d, slot, pos):
    """What FruitPool.assign does to the collision fields (contype on geom and body, pose, equality off)."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"pool_{slot}")
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"pool_{slot}_fruit")
    saved = (int(m.geom_contype[g]), int(m.body_contype[b]))
    m.geom_contype[g], m.body_contype[b] = 1, 1
    adr = m.jnt_qposadr[m.body_jntadr[b]]
    d.qpos[adr:adr + 3] = pos
    d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    return b, g, saved


def test_pool_fruit_target_must_be_ignored(world, targets):
    m, d, q0 = world
    park_probe(m, d)
    planner = ArmPlanner(m, backend="rrtconnect")
    t = targets[0]
    d.qpos[:] = station_qpos(q0, t["station"])
    planner.scene.sync(m, d)
    goal = np.array(t["q"])
    b, g, saved = _assign_fruit(m, d, 0, planner.scene.tip(goal))
    try:
        res = planner.plan(None, goal, data=d)
        assert res.reason == "goal_invalid" and res.contacts[0]["class"] == "fruit"
        res = planner.plan(None, goal, data=d, ignore=["pool_0"])
        assert res.ok, res.reason
        res = planner.plan(None, goal, data=d, ignore_classes=["fruit"])
        assert res.ok
    finally:
        m.geom_contype[g], m.body_contype[b] = saved


def test_attached_fruit_is_carried_and_checked(world):
    m, d, q0 = world
    park_probe(m, d)
    planner = ArmPlanner(m, backend="rrtconnect")
    sc = planner.scene
    d.qpos[:] = q0
    sc.sync(m, d)
    home = q0[sc.qadr]
    b, g, saved = _assign_fruit(m, d, 1, sc.tip(home))     # held at the tip
    adr = m.jnt_qposadr[m.body_jntadr[b]]
    try:
        sc.sync(m, d)
        sc.attach("pool_1")
        assert g in sc.moving_geoms
        q2 = home + np.array([0.0, 0.6, 0.0, 0.0, 0.0])
        sc.is_valid(q2)
        carried = sc.d.qpos[adr:adr + 3].copy()
        assert np.linalg.norm(carried - sc.tip(q2)) < 1e-3          # the fruit followed the wrist
        d.mocap_pos[mocap_index(m)] = carried + np.array([0.0, 0.0, 0.09])
        sc.sync(m, d)
        hits = sc.contacts(q2)
        assert any(c["arm"] == "pool_1_fruit" and c["obstacle"] == PROBE + "_g" for c in hits), hits
        sc.detach_all()
        assert not any(c["arm"] == "pool_1_fruit" for c in sc.contacts(q2))
        park_probe(m, d)
        res = planner.plan(None, home + np.array([0.0, 0.3, 0.0, 0.0, 0.0]), data=d, attach=["pool_1"],
                           ignore_classes=["fruit"])
        assert res.ok, (res.reason, res.contacts)
    finally:
        m.geom_contype[g], m.body_contype[b] = saved
        park_probe(m, d)


def _fruit_in_left_pad(m, d, sc, slot, depth=0.004):
    """Assign a pool slot and place it between the pads, ``depth`` m into finger_left_pad (pads closed on it)."""
    gp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "finger_left_pad")
    b, g, saved = _assign_fruit(m, d, slot, [0.0, 0.0, 0.0])
    mujoco.mj_forward(m, d)
    p = d.geom_xpos[gp].copy()
    p[1] -= m.geom_size[gp][1] + m.geom_size[g][1] - depth
    adr = m.jnt_qposadr[m.body_jntadr[b]]
    d.qpos[adr:adr + 3] = p
    mujoco.mj_forward(m, d)
    assert mujoco.mj_geomDistance(m, d, g, gp, 0.1, np.zeros(6)) == pytest.approx(-depth, abs=5e-4)
    return b, g, saved


def test_explicit_finger_fruit_pairs_follow_ignore_and_attach(world):
    """The contact_pairs finger x fruit <pair>s bypass contype masks: held / ignored fruit must not invalidate."""
    m, d, q0 = world
    assert m.npair > 0
    park_probe(m, d)
    planner = ArmPlanner(m, backend="rrtconnect")
    sc = planner.scene
    d.qpos[:] = q0
    sc.sync(m, d)
    home = q0[sc.qadr]
    goal = home + np.array([0.0, 0.3, 0.0, 0.0, 0.0])
    b, g, saved = _fruit_in_left_pad(m, d, sc, 1)
    try:
        # WP8 carry: held fruit attached (with or without ignoring the fruit class)
        for kw in ({"attach": ["pool_1"]}, {"attach": ["pool_1"], "ignore_classes": ["fruit"]},
                   {"attach": ["pool_1"], "ignore": ["pool_1"]}):
            res = planner.plan(None, goal, data=d, **kw)
            assert res.ok, (kw, res.reason, res.contacts)
        # WP8 approach / grasp: target ignored while the pads touch it
        for kw in ({"ignore": ["pool_1"]}, {"ignore_classes": ["fruit"]}):
            sc.detach_all()
            sc.sync(m, d)
            sc.set_ignore(**({"geoms": kw["ignore"]} if "ignore" in kw else {"classes": kw["ignore_classes"]}))
            assert sc.is_valid(home), (kw, sc.contacts(home))
        # not ignored, not attached: a real fruit obstacle, reported with its class and the explicit-pair margin
        sc.detach_all()
        sc.sync(m, d)
        sc.set_ignore()
        hits = sc.contacts(home)
        assert hits and all(c["class"] == "fruit" and c["obstacle"] == "pool_1_fruit" for c in hits), hits
        assert any(c["arm"] == "finger_left_pad" for c in hits)
        res = planner.plan(None, goal, data=d)
        assert res.reason == "start_invalid" and res.contacts[0]["class"] == "fruit"
        # parked / released slot (contype 0) overlapping a pad is not an obstacle
        m.geom_contype[g], m.body_contype[b] = saved
        sc.sync(m, d)
        sc.set_ignore()
        assert sc.is_valid(home), sc.contacts(home)
    finally:
        m.geom_contype[g], m.body_contype[b] = saved
        park_probe(m, d)


def test_rrtconnect_deterministic(world, targets):
    m, d, q0 = world
    t = targets[1]
    paths = []
    for _ in range(2):
        planner = ArmPlanner(m, backend="rrtconnect", seed=11)
        d.qpos[:] = station_qpos(q0, t["station"])
        start, goal = d.qpos[planner.scene.qadr].copy(), np.array(t["q"])
        place_probe(m, d, planner.scene, start, goal)
        paths.append(planner.plan(None, goal, data=d).path)
    park_probe(m, d)
    assert paths[0].shape == paths[1].shape and np.allclose(paths[0], paths[1])


def test_time_parameterization_respects_limits():
    path = np.array([[0, 0, 0, 0, 0], [0.2, 1.0, -0.5, 0.3, 0.0], [0.2, 1.2, 0.4, -0.6, 1.0]], dtype=float)
    vmax, amax = np.array([0.15, 1.2, 1.0, 1.2, 1.5]), np.array([0.5, 3.0, 2.5, 3.0, 4.0])
    tr = time_parameterize(path, vmax, amax)
    dt = 0.0005
    ts = np.arange(0.0, tr.duration + dt, dt)
    q = np.array([tr.sample(t) for t in ts])
    v = np.diff(q, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    assert np.all(np.max(np.abs(v), axis=0) <= vmax * 1.01)
    assert np.all(np.max(np.abs(a), axis=0) <= amax * 1.05)
    assert np.allclose(tr.sample(0.0), path[0]) and np.allclose(tr.sample(tr.duration), path[-1])
    for k, tk in enumerate(tr.t):
        assert np.allclose(tr.sample(tk), path[k], atol=1e-9)
    assert len(tr.samples(0.05)) >= int(tr.duration / 0.05)


def test_from_config_uses_hal_yaml(world):
    m, _, _ = world
    p = ArmPlanner.from_config(m)
    assert p.time_limit_s == 1.0 and p.scene.margins["ground"] == 0.02
    assert np.allclose(p.vmax, [0.15, 1.2, 1.0, 1.2, 1.5])
    assert p.backend == ("ompl" if ompl_planner.ompl_available() else "rrtconnect")


def test_ompl_is_optional(world, monkeypatch):
    """requirements-ci without the wheel: the module imports, auto falls back, an explicit ompl request fails."""
    m, _, _ = world
    for k in [k for k in sys.modules if k == "ompl" or k.startswith("ompl.")]:
        monkeypatch.delitem(sys.modules, k)
    monkeypatch.setitem(sys.modules, "ompl", None)
    try:
        mod = importlib.reload(ompl_planner)
        assert not mod.ompl_available() and mod.ompl_import_error()
        assert mod.ArmPlanner(m, backend="auto").backend == "rrtconnect"
        with pytest.raises(RuntimeError):
            mod.ArmPlanner(m, backend="ompl")
        with pytest.raises(ValueError):
            mod.ArmPlanner(m, backend="moveit")
    finally:
        monkeypatch.undo()
        importlib.reload(ompl_planner)


# ---------------------------------------------------------------------- attached subtree (WP8 change request)
TOOL = "wpcap_probe_tool"


def _tool_module(spec, field, ctx):
    """A free tool root (handle) with a hinged blade under it, like the WP2 tools; contype 0 and gravcomp 1, so it
    only exists for the planner once attached."""
    t = spec.worldbody.add_body(name=TOOL, pos=[25.0, 25.0, 0.5], gravcomp=1.0)
    t.add_freejoint()
    t.add_geom(name=TOOL + "_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.015, 0.015, 0.03], contype=0,
               conaffinity=0, mass=0.05)
    blade = t.add_body(name=TOOL + "_blade", pos=[0.0, 0.0, 0.08], gravcomp=1.0)
    blade.add_joint(name=TOOL + "_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
    blade.add_geom(name=TOOL + "_blade_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.04], contype=0,
                   conaffinity=0, mass=0.02)


@pytest.fixture(scope="module")
def tool_world():
    with env(NANDI_WORLD="orchard_demo"), compose.temporary_module("wpcap_probe", 90, _probe_module), \
            compose.temporary_module("wpcap_probe_tool", 91, _tool_module):
        m = farm_scene.load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    if m.nkey:
        with contextlib.suppress(Exception):
            mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d, d.qpos.copy()


def test_attach_carries_the_whole_subtree(tool_world):
    """``attach(root)`` moves the root's free joint with the wrist and brings every body below it (blade on its own
    hinge) into the moving set; detaching drops them again. Replaces WP8's ``subtree_attach`` wrapper."""
    m, d, q0 = tool_world
    sc = CollisionScene(m)
    d.qpos[:] = q0
    sc.sync(m, d)
    home = q0[sc.qadr]
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TOOL)
    adr = m.jnt_qposadr[m.body_jntadr[root]]
    d.qpos[adr:adr + 3] = sc.tip(home)
    d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    gh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, TOOL + "_handle")
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, TOOL + "_blade_g")
    sc.sync(m, d)
    sc.attach(TOOL)
    assert gh in sc.moving_geoms and gb in sc.moving_geoms
    assert gb not in sc.obstacle_geoms
    q2 = home + np.array([0.0, 0.6, 0.0, 0.0, 0.0])
    sc.is_valid(q2)
    blade_at = sc.d.geom_xpos[gb].copy()
    assert np.linalg.norm(sc.d.qpos[adr:adr + 3] - sc.tip(q2)) < 1e-3
    # the blade sits 8 cm above the handle in the tool frame, wherever the wrist took it
    sc.is_valid(q2)
    assert abs(np.linalg.norm(sc.d.geom_xpos[gb] - sc.d.xpos[root]) - 0.08) < 1e-6
    # the probe on the blade (not the handle) is a violation reported against the blade geom
    mi = mocap_index(m)
    d.mocap_pos[mi] = blade_at + np.array([0.0, 0.0, 0.075])
    sc.sync(m, d)
    hits = sc.contacts(q2)
    assert any(c["arm"] == TOOL + "_blade_g" and c["obstacle"] == PROBE + "_g" for c in hits), hits
    assert not sc.is_valid(q2)
    sc.detach_all()
    assert gb not in sc.moving_geoms and gh not in sc.moving_geoms
    assert not any(c["arm"].startswith(TOOL) for c in sc.contacts(q2))
    park_probe(m, d)
