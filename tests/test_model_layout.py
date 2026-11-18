"""WP0 name-based layout (MVP-SPEC 0 rules 1-2, 3.2): frozen robot table, v1 parity, RL observation parity.

ROS-free, headless, each test well under 10 s.
"""
import os

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.core import model_layout as mlay
from atharv_nandi.core.rl_balance_policy import RLBalanceController

V1_XML = os.path.join(os.path.dirname(farm_scene.ROBOT_XML), "atharv_nandi_v1.xml")


def test_frozen_table_shape():
    assert len(mlay.ROBOT_JOINTS) == 26 and len(mlay.ROBOT_ACTUATORS) == mlay.ROBOT_NU == 25
    assert mlay.ROBOT_JOINTS[0] == ("root", 0) and mlay.ROBOT_JOINTS[-1] == ("RL_calf_joint", 31)
    assert [a for _, a in mlay.ROBOT_JOINTS[1:]] == list(range(7, 32))


@pytest.mark.parametrize("terrain", ["rough", "flat", "ridges", "orchard"])
def test_actuators_0_24_and_qpos_0_32_match_the_frozen_table(monkeypatch, terrain):
    monkeypatch.setenv("NANDI_TERRAIN", terrain)
    m = farm_scene.load_farm_model()
    lay = mlay.Layout.from_model(m)
    assert mlay.frozen_violations(lay) == []
    for name, adr in mlay.ROBOT_JOINTS:
        assert lay.qpos_adr(name) == adr, name
    assert lay.actuators[:25] == mlay.ROBOT_ACTUATORS
    for i, name in enumerate(mlay.ROBOT_ACTUATORS):
        assert lay.act_id(name) == i
    # ctrl order == qpos[7:32] order (sim_runner and the RL code rely on it)
    assert [lay.qpos_adr(j) for j in lay.actuator_joint[:25]] == list(range(7, 32))


def test_v1_copy_matches_the_live_robot():
    v1 = mlay.Layout.from_model(mujoco.MjModel.from_xml_path(V1_XML))
    live = mlay.Layout.from_model(mujoco.MjModel.from_xml_path(farm_scene.ROBOT_XML))
    assert mlay.frozen_violations(v1) == [] and mlay.frozen_violations(live) == []
    robot = dict(mlay.ROBOT_JOINTS)
    assert {n: v for n, v in v1.joints.items() if n in robot} == {n: v for n, v in live.joints.items() if n in robot}
    assert v1.actuators[:25] == live.actuators[:25]
    assert v1.crops[0] == live.crops[0] == ("crop_item_1", 32)


def test_rl_literal_slices_match_the_layout():
    """rl_balance_policy.py and rl_training keep literal slices; this guards them (design rule 2)."""
    lay = mlay.Layout.from_model(farm_scene.load_farm_model())
    arm5 = ("gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch")
    assert [lay.qpos_adr(j) for j in arm5] == list(range(7, 12))        # qpos[7:12]
    assert [lay.dof_adr(j) for j in arm5] == list(range(6, 11))         # qvel[6:11]
    legs = [f"{leg}_{p}_joint" for leg in ("FR", "FL", "MR", "ML", "RR", "RL") for p in ("hip", "thigh", "calf")]
    assert [lay.qpos_adr(j) for j in legs] == list(range(14, 32))       # qpos[14:32]
    assert [lay.dof_adr(j) for j in legs] == list(range(13, 31))        # qvel[13:31]
    assert [lay.act_id(a) for a in mlay.ROBOT_ACTUATORS[7:]] == list(range(7, 25))  # ctrl[7:25]
    assert lay.qpos_slice("root") == mlay.ROOT_QPOS and lay.dof_slice("root") == slice(0, 6)


def test_rl_observation_identical_on_v1_and_current_world():
    v1 = RLBalanceController(model=mujoco.MjModel.from_xml_path(V1_XML), load_policy=False)
    cur = RLBalanceController(model=farm_scene.load_farm_model(), load_policy=False)
    rng = np.random.default_rng(7)
    for _ in range(5):
        prefix0 = np.array(v1.model.key_qpos[0][:32])
        prefix0[0:3] += rng.uniform(-0.05, 0.05, 3)
        prefix0[7:32] += rng.uniform(-0.1, 0.1, 25)
        prefix1 = prefix0.copy()
        prefix1[7:32] += rng.uniform(-0.01, 0.01, 25)
        q_v1 = [np.concatenate([p, v1.model.key_qpos[0][32:]]) for p in (prefix0, prefix1)]
        q_cur = [np.concatenate([p, cur.model.key_qpos[0][32:]]) for p in (prefix0, prefix1)]
        a, b = v1.build_obs(*q_v1), cur.build_obs(*q_cur)
        assert a.shape == (84,)
        np.testing.assert_array_equal(a, b)


def test_layout_lookups_and_json_round_trip():
    m = farm_scene.load_farm_model()
    lay = mlay.current()
    assert lay is not None and lay.nq == m.nq and len(lay.world_id) == 12
    back = mlay.Layout.from_json(lay.to_json())
    assert back == lay and back.to_json() == lay.to_json()
    assert back.crop_qpos(2) == slice(39, 46) and back.crop_slice("crop_item_2") == slice(39, 46)
    for bad in (lambda: lay.qpos_adr("nope"), lambda: lay.dof_adr("nope"), lambda: lay.qpos_slice("nope"),
                lambda: lay.dof_slice("nope"), lambda: lay.act_id("nope"), lambda: lay.eq_id("nope"),
                lambda: lay.mocap_id("nope"), lambda: lay.crop_qpos(0), lambda: lay.crop_qpos(7),
                lambda: lay.crop_slice("crop_item_9"), lambda: lay.crop_positions([0.0] * 3)):
        with pytest.raises(mlay.LayoutError):
            bad()
    with pytest.raises(mlay.LayoutError):
        mlay.Layout.from_dict(dict(lay.to_dict(), v=99))


def test_frozen_violations_are_reported():
    lay = mlay.current() or mlay.Layout.from_model(farm_scene.load_farm_model())
    d = lay.to_dict()
    d["joints"]["arm_yaw"] = [9, 7, 3]
    d["actuators"] = ["x"] + d["actuators"][1:]
    d["crops"][1] = ["crop_item_2", 50]
    d["joints"]["intruder"] = [33, 33, 3]
    bad = mlay.frozen_violations(mlay.Layout.from_dict(d))
    text = " | ".join(bad)
    assert "arm_yaw" in text and "actuators 0..24" in text and "crop_item_2" in text and "intruder" in text
    with pytest.raises(mlay.LayoutError):
        mlay.assert_frozen(mlay.Layout.from_dict(d))


def test_legacy_crop_count_fallback_without_a_layout():
    saved = mlay.current()
    try:
        mlay.set_current(None)
        assert mlay.n_legacy_crops_for(32 + 7 * 4) == 4
        assert mlay.n_legacy_crops_for(10) == 0
        assert list(mlay.legacy_crop_positions([0.0] * 32 + [1, 2, 3, 1, 0, 0, 0])) == ["crop_item_1"]
    finally:
        mlay.set_current(saved)
