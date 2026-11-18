"""Perception camera model: cameras added on top of the shared farm world (no ROS)."""
import math
import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import camera_rig
from atharv_nandi.core.detector_model import DetectorModel, load_config, project
from atharv_nandi.core.farm_scene import load_farm_model


@pytest.fixture(scope="module")
def world():
    model = camera_rig.build_perception_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def test_camera_names_exist_and_qpos_layout_unchanged(world):
    model, _ = world
    for name in camera_rig.CAMERA_NAMES:
        assert camera_rig.camera_id(model, name) >= 0
    phys = load_farm_model()
    assert model.nq == phys.nq and model.nv == phys.nv and model.nu == phys.nu
    assert model.ncam == phys.ncam + 2
    with pytest.raises(KeyError):
        camera_rig.camera_id(model, "no_such_cam")


def test_cameras_are_attached_to_trunk_and_gripper(world):
    model, _ = world
    head, wrist = (camera_rig.camera_id(model, n) for n in camera_rig.CAMERA_NAMES)
    body = lambda b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)  # noqa: E731
    assert model.cam_bodyid[head] == body("trunk")
    assert model.cam_bodyid[wrist] == body("arm_wrist")
    assert model.cam_fovy[head] == pytest.approx(70.0)


def test_head_cam_looks_forward_pitched_down_35deg(world):
    model, data = world
    cam = camera_rig.camera_pose(data, camera_rig.camera_id(model, camera_rig.HEAD_CAM))
    trunk = camera_rig.body_pose(model, data)
    axis_world = cam.R[2]                      # optical +z in world coordinates
    axis_trunk = trunk.R @ axis_world
    p = math.radians(35.0)
    assert axis_trunk == pytest.approx([math.cos(p), 0.0, -math.sin(p)], abs=1e-6)
    assert trunk.R @ cam.R[0] == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)  # image right = robot right


def test_crop_item_1_projects_into_head_image_at_home(world):
    model, data = world
    names, truth = camera_rig.crop_truth(model, data)
    assert names[0] == "crop_item_1"
    cam = camera_rig.camera_pose(data, camera_rig.camera_id(model, camera_rig.HEAD_CAM))
    intr = camera_rig.intrinsics(model, camera_rig.HEAD_CAM, 160, 120)
    uv, z = project(cam.apply(truth[:1]), intr)
    assert abs(uv[0, 0] - 80.0) < 2.0           # straight ahead
    assert 60.0 < uv[0, 1] < 120.0              # lower half of the image
    assert 0.3 < z[0] < 0.6


def test_geometry_only_detection_error_at_home(world):
    """Same budget as the live acceptance check: mean error for crop_item_1 over 20 frames < 4 cm."""
    model, data = world
    cfg = load_config()
    det = DetectorModel(cfg["detector"])
    names, truth = camera_rig.crop_truth(model, data)
    cam = camera_rig.camera_pose(data, camera_rig.camera_id(model, camera_rig.HEAD_CAM))
    trunk = camera_rig.body_pose(model, data)
    intr = camera_rig.intrinsics(model, camera_rig.HEAD_CAM, 160, 120)
    want = trunk.apply(truth[:1])[0]
    errs = []
    for _ in range(20):
        hits = [d for d in det.detect(truth, names, cam, trunk, intr) if d["_truth"] == "crop_item_1"]
        errs += [np.linalg.norm(np.array(h["xyz_trunk"]) - want) for h in hits]
    assert len(errs) >= 15
    assert float(np.mean(errs)) < 0.04


def test_set_qpos_rejects_wrong_length(world):
    model, _ = world
    data = mujoco.MjData(model)
    assert not camera_rig.set_qpos(model, data, [0.0] * (model.nq - 1))
    q = np.array(model.key_qpos[0])
    q[0] += 0.5
    assert camera_rig.set_qpos(model, data, q)
    assert data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")][0] == pytest.approx(q[0])


def test_same_view():
    from atharv_nandi.core.detector_model import Pose
    a = Pose.from_frame_in_world(np.zeros(3), np.eye(3))
    b = Pose.from_frame_in_world(np.array([0.005, 0, 0]), np.eye(3))
    c = Pose.from_frame_in_world(np.array([0.05, 0, 0]), np.eye(3))
    ang = math.radians(3.0)
    rot = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
    d = Pose.from_frame_in_world(np.zeros(3), rot)
    assert camera_rig.same_view(a, b) and not camera_rig.same_view(a, c) and not camera_rig.same_view(a, d)
    assert not camera_rig.same_view(a, None)


def test_same_scene_sees_arm_and_crop_motion_but_not_trunk_motion(world):
    model, _ = world
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ref = camera_rig.scene_state(data)
    assert ref.shape == (model.nq - 7,)
    q = data.qpos.copy()
    q[0] += 0.5  # trunk moves: that is same_view's job, not same_scene's
    assert camera_rig.set_qpos(model, data, q)
    assert camera_rig.same_scene(camera_rig.scene_state(data), ref)
    for idx, delta in ((9, 0.2), (12, 0.01), (34, 0.05)):  # shoulder, finger_l, crop_item_1 z
        q2 = q.copy()
        q2[idx] += delta
        assert camera_rig.set_qpos(model, data, q2)
        assert not camera_rig.same_scene(camera_rig.scene_state(data), ref), idx
    assert not camera_rig.same_scene(ref, None)


def test_render_budget_caps_cpu_share():
    b = camera_rig.RenderBudget(0.2, burst_s=0.1)
    t, rendered = 0.0, 0
    for _ in range(100):          # 10 s of 10 Hz ticks, each render costs 80 ms
        if b.allow(t):
            b.charge(0.08)
            rendered += 1
        t += 0.1
    assert b.spent_s <= 0.2 * 10 + 0.1 + 0.08 + 1e-9
    assert rendered >= 20 and b.skips > 0
    assert not camera_rig.RenderBudget(0.0).allow(0.0)


def _gl_available() -> bool:
    """Probe in a subprocess: a broken GL stack can abort the interpreter instead of raising."""
    code = (
        "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); "
        "r = mujoco.Renderer(m, 8, 8); r.close()"
    )
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


def test_offscreen_render_is_not_constant(world):
    if not _gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    model, data = world
    cam = camera_rig.CameraRenderer(model, camera_rig.HEAD_CAM, 64, 48)
    try:
        img = cam.rgb(data)
        rgb, depth = cam.rgbd(data)
        ref_depth = cam.depth(data)
    finally:
        cam.close()
    assert img.shape == (48, 64, 3) and img.dtype == np.uint8
    assert img.std() > 1.0
    assert np.array_equal(rgb, img)
    near = ref_depth < 3.0
    assert near.any() and np.median(np.abs(depth[near] - ref_depth[near])) < 0.01
