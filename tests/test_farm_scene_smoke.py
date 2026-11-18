import mujoco

from atharv_nandi.core.farm_scene import crop_body_names, crop_qpos_slice, load_farm_model


def test_qpos_prefix_and_crops():
    m = load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    names = crop_body_names(m)
    assert names[0] == "crop_item_1"
    assert m.nq == 32 + 7 * len(names)
    for k, name in enumerate(names, start=1):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        mujoco.mj_forward(m, d)
        assert abs(d.xpos[bid][0] - d.qpos[crop_qpos_slice(k)][0]) < 1e-9
