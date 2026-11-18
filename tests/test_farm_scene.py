"""World builder checks: every terrain compiles, crop knob, stable qpos prefix, walkable terrain."""
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core.farm_scene import (
    CROP_RADIUS,
    RIDGES,
    ROBOT_XML,
    crop_body_names,
    crop_qpos_slice,
    load_farm_model,
)

FOOT_RADIUS = 0.03  # *_foot spheres in the robot MJCF


def _load(monkeypatch, terrain="rough", crops=None, seed=None):
    monkeypatch.setenv("NANDI_TERRAIN", terrain)
    if crops is None:
        monkeypatch.delenv("NANDI_CROPS", raising=False)
    else:
        monkeypatch.setenv("NANDI_CROPS", str(crops))
    if seed is None:
        monkeypatch.delenv("NANDI_SEED", raising=False)
    else:
        monkeypatch.setenv("NANDI_SEED", str(seed))
    return load_farm_model()


def _geom_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def _clod_ids(m):
    return [i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("clod_")]


@pytest.mark.parametrize("terrain", ["flat", "ridges", "rough"])
def test_every_terrain_compiles_with_default_world(monkeypatch, terrain):
    m = _load(monkeypatch, terrain)
    assert crop_body_names(m) == [f"crop_item_{k}" for k in range(1, 7)]
    assert m.nq == 74 and m.nu == 25
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    has_ridges = _geom_id(m, "ridge_1") != -1
    assert has_ridges is (terrain != "flat")
    assert bool(_clod_ids(m)) is (terrain == "rough")


def test_unknown_terrain_falls_back_to_rough(monkeypatch):
    m = _load(monkeypatch, "lava")
    assert _clod_ids(m)


@pytest.mark.parametrize("crops, expected", [(1, 1), (3, 3), (9, 9), (0, 1), (-4, 1)])
def test_crop_count_knob(monkeypatch, crops, expected):
    m = _load(monkeypatch, crops=crops)
    names = crop_body_names(m)
    assert names == [f"crop_item_{k}" for k in range(1, expected + 1)]
    assert m.nq == 32 + 7 * expected
    assert m.key_qpos.shape[1] == m.nq
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for k, name in enumerate(names, start=1):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert np.allclose(d.xpos[bid], d.qpos[crop_qpos_slice(k)][:3])
        assert d.qpos[crop_qpos_slice(k)][2] == pytest.approx(CROP_RADIUS)


@pytest.mark.parametrize("terrain", ["flat", "ridges", "rough"])
def test_qpos_prefix_identical_to_raw_robot_mjcf(monkeypatch, terrain):
    raw = mujoco.MjModel.from_xml_path(ROBOT_XML)
    m = _load(monkeypatch, terrain)
    assert raw.nq == 39 and raw.nu == m.nu
    assert np.array_equal(m.key_qpos[0][: raw.nq], raw.key_qpos[0])
    assert np.array_equal(m.key_ctrl[0], raw.key_ctrl[0])
    for j in range(raw.njnt):
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) == mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert m.jnt_qposadr[j] == raw.jnt_qposadr[j]
    for a in range(raw.nu):
        assert mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) == mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_ACTUATOR, a)


def test_layout_is_deterministic_per_seed(monkeypatch):
    a = _load(monkeypatch, seed=7).key_qpos[0].copy()
    b = _load(monkeypatch, seed=7).key_qpos[0].copy()
    c = _load(monkeypatch, seed=8).key_qpos[0].copy()
    assert np.array_equal(a, b)
    assert not np.array_equal(a[39:], c[39:])


def test_ridges_are_low_smooth_mounds(monkeypatch):
    m = _load(monkeypatch, "ridges")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for name, x, h, w in RIDGES:
        g = _geom_id(m, name)
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
        radius = m.geom_size[g][0]
        crest = d.geom_xpos[g][2] + radius
        assert crest == pytest.approx(h)
        assert crest <= 0.045
        assert d.geom_xpos[g][0] == pytest.approx(x)
        axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
        assert np.allclose(np.abs(axis), [0, 1, 0], atol=1e-9)  # lies across the robot's path
        # mound meets the ground at its footprint half-width, with a slope the foot can roll up
        assert math.sqrt(radius ** 2 - (radius - h) ** 2) == pytest.approx(w)
        edge_slope = math.degrees(math.acos((radius - h) / radius))
        assert edge_slope < 30.0


def test_clods_are_below_foot_radius_and_off_the_crops(monkeypatch):
    m = _load(monkeypatch, "rough", crops=12)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    clods = _clod_ids(m)
    assert 1 <= len(clods) <= 40  # crops 6..12 reach into the clod patch, so some clods are skipped
    assert len(_clod_ids(_load(monkeypatch, "rough"))) >= 30  # default 6 crops barely touch it
    for g in clods:
        top = d.geom_xpos[g][2] + m.geom_size[g][2]
        assert d.geom_xpos[g][2] == pytest.approx(m.geom_size[g][2])  # sits on the floor
        assert top < FOOT_RADIUS
    # every crop rests on the floor only: no initial contact with ridges, clods or the robot
    floor = _geom_id(m, "floor")
    crop_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in crop_body_names(m)}
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if b1 in crop_bodies or b2 in crop_bodies:
            assert floor in (c.geom1, c.geom2)
