"""Unit tests for the simulated crop detector (pure numpy, no ROS, no GL)."""
import hashlib
import json
import math

import numpy as np
import pytest

from atharv_nandi.core import detector_model as dm
from atharv_nandi.core.detector_model import (
    CONFIG_PATH, DetectorModel, Pose, backproject, build_message, config_sha256,
    encode_message, intrinsics_from_fovy, load_config, parse_simple_yaml, project, validate_message,
)

IDENTITY = Pose(np.eye(3), np.zeros(3))
INTR90 = intrinsics_from_fovy(90.0, 160, 120)  # f = 60 px exactly


def cfg(**over):
    c = dict(load_config()["detector"])
    c.update(over)
    return c


def clean_cfg(**over):
    """No misses, no false positives, no conf filtering: isolates one effect at a time."""
    base = dict(miss_rate=0.0, fp_rate_per_frame=0.0, conf_threshold=0.0)
    base.update(over)
    return cfg(**base)


# ------------------------------------------------------------------ config
def test_config_defaults_match_contract():
    c = load_config()
    d = c["detector"]
    assert d["name"] == "sim_truth_noise"
    assert d["max_range_m"] == pytest.approx(2.5)
    assert d["pos_sigma_base_m"] == pytest.approx(0.015)
    assert d["pos_sigma_range_frac"] == pytest.approx(0.01)
    assert d["miss_rate"] == pytest.approx(0.05)
    assert d["fp_rate_per_frame"] == pytest.approx(0.05)
    assert d["oracle_ids"] is False
    assert isinstance(d["seed"], int)
    assert c["camera"]["width"] == 160 and c["camera"]["height"] == 120
    assert c["camera"]["head_fovy_deg"] == pytest.approx(70.0)
    assert c["camera"]["head_pitch_down_deg"] == pytest.approx(35.0)
    assert c["node"]["detection_rate_hz"] <= 5.0
    assert c["node"]["image_rate_hz"] == pytest.approx(2.0)


def test_simple_yaml_parser_matches_pyyaml():
    text = open(CONFIG_PATH, encoding="utf-8").read()
    ours = parse_simple_yaml(text)
    yaml = pytest.importorskip("yaml")
    assert ours == yaml.safe_load(text)


def test_simple_yaml_parser_scalars_and_lists():
    out = parse_simple_yaml("a:\n  b: 1  # c\n  c: [0.5, 2, x]\n  d: false\n  e: 'q # r'\nf: 1.5e-2\n")
    assert out == {"a": {"b": 1, "c": [0.5, 2, "x"], "d": False, "e": "q # r"}, "f": 0.015}


def test_config_sha256_is_file_hash():
    sha = config_sha256()
    assert sha == hashlib.sha256(open(CONFIG_PATH, "rb").read()).hexdigest()
    assert len(sha) == 64


# ------------------------------------------------------------------ pinhole
def test_intrinsics_from_fovy():
    assert INTR90.fy == pytest.approx(60.0)
    assert INTR90.fx == INTR90.fy
    assert (INTR90.cx, INTR90.cy) == (80.0, 60.0)
    i70 = intrinsics_from_fovy(70.0, 160, 120)
    assert i70.fy == pytest.approx(60.0 / math.tan(math.radians(35.0)))


def test_projection_of_known_point_is_exact_pixel():
    uv, z = project(np.array([[0.5, 0.25, 1.0], [0.0, 0.0, 2.0]]), INTR90)
    assert uv[0] == pytest.approx([110.0, 75.0])
    assert uv[1] == pytest.approx([80.0, 60.0])
    assert z == pytest.approx([1.0, 2.0])


def test_projection_behind_camera_is_nan_and_backproject_inverts():
    uv, _ = project(np.array([[0.1, 0.1, -1.0]]), INTR90)
    assert np.isnan(uv).all()
    p = np.array([[0.3, -0.2, 1.7]])
    uv, z = project(p, INTR90)
    assert backproject(uv, z, INTR90) == pytest.approx(p)


def test_pose_round_trip():
    ang = 0.7
    rot = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
    pose = Pose.from_frame_in_world(np.array([1.0, 2.0, 0.5]), rot)
    p_world = np.array([[1.5, 2.0, 0.5]])
    p_local = pose.apply(p_world)
    assert p_local[0] == pytest.approx([0.5 * math.cos(ang), -0.5 * math.sin(ang), 0.0])
    assert pose.inverse_apply(p_local) == pytest.approx(p_world)


# ------------------------------------------------------------------ visibility
def test_fov_cut():
    det = DetectorModel(clean_cfg())
    inside = [0.9, 0.0, 1.0]     # u = 80 + 54 = 134 < 160
    outside = [1.4, 0.0, 1.0]    # u = 164 > 160
    above = [0.0, -1.05, 1.0]    # v = -3 < 0
    ok, uv, _, _ = det.visible(np.array([inside, outside, above]), INTR90)
    assert ok.tolist() == [True, False, False]


def test_range_cut():
    det = DetectorModel(clean_cfg())
    ok, _, _, rng = det.visible(np.array([[0.0, 0.0, 2.4], [0.0, 0.0, 2.6], [0.0, 0.0, 0.05]]), INTR90)
    assert ok.tolist() == [True, False, False]
    assert rng == pytest.approx([2.4, 2.6, 0.05])


def test_occlusion_by_depth_buffer():
    det = DetectorModel(clean_cfg())
    p = np.array([[0.0, 0.0, 1.0]])
    clear = np.full((120, 160), 5.0, dtype=np.float32)
    clear[60, 80] = 1.0 - 0.035          # the crop's own front surface
    blocked = clear.copy()
    blocked[60, 80] = 0.6                 # something 40 cm in front of it
    assert det.visible(p, INTR90, clear)[0].tolist() == [True]
    assert det.visible(p, INTR90, blocked)[0].tolist() == [False]
    assert det.visible(p, INTR90, None)[0].tolist() == [True]


# ------------------------------------------------------------------ noise model
def test_position_noise_statistics_match_sigma():
    det = DetectorModel(clean_cfg(), seed=123)
    truth = np.array([[0.0, 0.0, 1.5]])
    xs = []
    for _ in range(2000):
        out = det.detect(truth, ["crop_item_1"], IDENTITY, IDENTITY, INTR90)
        assert len(out) == 1
        xs.append(out[0]["xyz_trunk"])
    err = np.asarray(xs) - truth
    sigma = 0.015 + 0.01 * 1.5
    assert det.sigma_at(1.5) == pytest.approx(sigma)
    for axis in range(3):
        assert abs(err[:, axis].std() - sigma) < 0.10 * sigma
        assert abs(err[:, axis].mean()) < 0.1 * sigma


def test_seeded_determinism():
    truth = np.array([[0.1, 0.0, 1.0], [-0.3, 0.1, 1.8]])
    names = ["crop_item_1", "crop_item_2"]

    def run(seed):
        det = DetectorModel(cfg(), seed=seed)
        return [det.detect(truth, names, IDENTITY, IDENTITY, INTR90) for _ in range(50)]

    assert run(5) == run(5)
    assert run(5) != run(6)
    # the config seed is used when none is given
    a, b = DetectorModel(cfg()), DetectorModel(cfg())
    assert a.detect(truth, names, IDENTITY, IDENTITY, INTR90) == b.detect(truth, names, IDENTITY, IDENTITY, INTR90)


def test_false_positive_rate_within_tolerance():
    det = DetectorModel(cfg(conf_threshold=0.0), seed=11)
    frames = 6000
    n_fp = sum(len(det.detect(np.zeros((0, 3)), [], IDENTITY, IDENTITY, INTR90)) for _ in range(frames))
    rate = n_fp / frames
    assert abs(rate - 0.05) < 0.015, rate


def test_false_positives_are_in_view_and_range():
    det = DetectorModel(cfg(conf_threshold=0.0, fp_rate_per_frame=1.0), seed=3)
    for _ in range(200):
        for d in det.detect(np.zeros((0, 3)), [], IDENTITY, IDENTITY, INTR90):
            assert d["_truth"] is None
            uv, z = project(np.array([d["xyz_trunk"]]), INTR90)
            assert 0.4 - 1e-3 <= z[0] <= 2.5 + 1e-3
            assert -0.5 <= uv[0, 0] <= 160.5 and -0.5 <= uv[0, 1] <= 120.5


def test_miss_rate_within_tolerance():
    det = DetectorModel(cfg(fp_rate_per_frame=0.0, conf_threshold=0.0), seed=4)
    frames = 6000
    hits = sum(len(det.detect(np.array([[0.0, 0.0, 1.0]]), ["c"], IDENTITY, IDENTITY, INTR90)) for _ in range(frames))
    assert abs((1 - hits / frames) - 0.05) < 0.015


def test_confidence_bounded_and_falls_with_range():
    det = DetectorModel(clean_cfg(), seed=9)

    def mean_conf(z):
        c = [det.detect(np.array([[0.0, 0.0, z]]), ["c"], IDENTITY, IDENTITY, INTR90)[0]["conf"] for _ in range(300)]
        assert min(c) >= 0.0 and max(c) <= 1.0
        return float(np.mean(c))

    near, far = mean_conf(0.5), mean_conf(2.3)
    assert near > far + 0.1


def test_bbox_contains_projection_and_scales_with_range():
    det = DetectorModel(clean_cfg(bbox_sigma_px=0.0))
    near = det.detect(np.array([[0.0, 0.0, 0.5]]), ["c"], IDENTITY, IDENTITY, INTR90)[0]["bbox_px"]
    far = det.detect(np.array([[0.0, 0.0, 2.0]]), ["c"], IDENTITY, IDENTITY, INTR90)[0]["bbox_px"]
    assert near[0] <= 80 <= near[2] and near[1] <= 60 <= near[3]
    assert (near[2] - near[0]) > 3 * (far[2] - far[0])
    assert near[2] - near[0] == pytest.approx(2 * 60 * 0.035 / 0.5, abs=1.0)


# ------------------------------------------------------------------ ids
def test_track_ids_hide_names_and_are_stable():
    det = DetectorModel(clean_cfg(), seed=1)
    truth = np.array([[0.2, 0.0, 1.0], [-0.4, 0.1, 1.5]])
    names = ["crop_item_1", "crop_item_2"]
    seen = {}
    for _ in range(100):
        for d in det.detect(truth, names, IDENTITY, IDENTITY, INTR90):
            assert d["id_hint"].startswith("trk_")
            seen.setdefault(d["_truth"], set()).add(d["id_hint"])
    assert {k: len(v) for k, v in seen.items()} == {"crop_item_1": 1, "crop_item_2": 1}
    assert seen["crop_item_1"] != seen["crop_item_2"]


def test_oracle_ids_use_body_names():
    det = DetectorModel(clean_cfg(oracle_ids=True))
    out = det.detect(np.array([[0.0, 0.0, 1.0]]), ["crop_item_7"], IDENTITY, IDENTITY, INTR90)
    assert out[0]["id_hint"] == "crop_item_7"


def test_trunk_frame_output():
    # Trunk 1 m behind the camera along the optical axis: trunk-frame z = camera z + 1.
    trunk = Pose(np.eye(3), np.array([0.0, 0.0, 1.0]))
    det = DetectorModel(clean_cfg(pos_sigma_base_m=0.0, pos_sigma_range_frac=0.0))
    out = det.detect(np.array([[0.1, 0.0, 1.0]]), ["c"], IDENTITY, trunk, INTR90)
    assert out[0]["xyz_trunk"] == pytest.approx([0.1, 0.0, 2.0])


# ------------------------------------------------------------------ message
def test_message_schema_and_size_under_16k():
    det = DetectorModel(clean_cfg(max_detections=32), seed=2)
    xs = np.linspace(-0.6, 0.6, 8)
    zs = np.linspace(0.8, 2.2, 4)
    truth = np.array([[x, 0.0, z] for x in xs for z in zs])
    dets = det.detect(truth, [f"crop_item_{i}" for i in range(len(truth))], IDENTITY, IDENTITY, INTR90)
    assert len(dets) > 20
    msg = build_message(12.3456, "head_cam", "sim_truth_noise", config_sha256(), dets)
    s = encode_message(msg)
    assert len(s.encode()) < 16 * 1024
    decoded = json.loads(s)
    assert validate_message(decoded) == []
    assert set(decoded) == {"t", "frame", "camera", "model", "detections"}
    assert decoded["frame"] == "trunk" and decoded["model"]["name"] == "sim_truth_noise"
    assert all(set(d) == {"id_hint", "xyz_trunk", "conf", "bbox_px"} for d in decoded["detections"])


def test_validate_message_rejects_bad_payloads():
    good = build_message(1.0, "head_cam", "sim_truth_noise", "0" * 64,
                         [{"id_hint": "trk_1", "xyz_trunk": [0.1, 0.2, 0.3], "conf": 0.9, "bbox_px": [1, 2, 3, 4]}])
    assert validate_message(good) == []
    bad = json.loads(json.dumps(good))
    bad["detections"][0]["conf"] = 1.5
    bad["detections"][0]["bbox_px"] = [5, 2, 3, 4]
    bad["frame"] = "world"
    errs = validate_message(bad)
    assert any("conf" in e for e in errs) and any("bbox" in e for e in errs) and any("frame" in e for e in errs)
    assert validate_message({"t": 1.0}) != []


def test_encode_message_trims_to_16k():
    dets = [{"id_hint": f"trk_{i}", "xyz_trunk": [0.123456] * 3, "conf": 0.5, "bbox_px": [1, 2, 3, 4]}
            for i in range(400)]
    s = encode_message(build_message(0.0, "head_cam", "sim_truth_noise", "0" * 64, dets))
    assert len(s.encode()) <= dm.MAX_MESSAGE_BYTES
    assert 0 < len(json.loads(s)["detections"]) < 400


# ------------------------------------------------------------------ ego-motion (robot turning)
def _turning_rig(yaw: float, cam_h: float = 0.40, pitch_deg: float = 35.0):
    """World poses of a trunk yawed by ``yaw`` and its head camera (pitched down, at ``cam_h``)."""
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), -math.sin(p)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    cam = Pose.from_frame_in_world(np.array([0.0, 0.0, cam_h]), np.stack([right, down, fwd], axis=1))
    c, s = math.cos(yaw), math.sin(yaw)
    trunk = Pose.from_frame_in_world(np.array([-0.35, 0.0, 0.34]),
                                     np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))
    return cam, trunk


@pytest.mark.parametrize("yaw_rate", [0.2, 0.4, 0.8])
def test_track_ids_survive_turning_in_place(yaw_rate):
    # Crops 0.3 m apart on the ground; the robot yaws back and forth at 5 Hz so crops leave and
    # re-enter the view. Association must use the world frame, or a stale trunk-frame track is
    # handed to the neighbouring crop.
    det = DetectorModel(cfg(fp_rate_per_frame=0.0, conf_threshold=0.0), seed=5)
    truth = np.array([[0.6, y, 0.0] for y in (-0.6, -0.3, 0.0, 0.3, 0.6)])
    names = [f"crop_item_{i}" for i in range(len(truth))]
    intr = intrinsics_from_fovy(70.0, 160, 120)
    owners, ids, seen_at = {}, {}, {}
    yaw, direction = 0.0, 1.0
    for frame in range(100):
        yaw += direction * yaw_rate * 0.2
        if abs(yaw) > 0.6:
            direction = -direction
        cam, trunk = _turning_rig(yaw)
        for d in det.detect(truth, names, cam, trunk, intr):
            owners.setdefault(d["id_hint"], set()).add(d["_truth"])
            ids.setdefault(d["_truth"], set()).add(d["id_hint"])
            seen_at.setdefault(d["_truth"], []).append(frame)
    assert all(len(v) == 1 for v in owners.values()), owners  # no id ever covers two crops
    # A crop only gets a new id after it was out of view for longer than track_max_age frames.
    max_age = int(cfg()["track_max_age"])
    for name, frames in seen_at.items():
        long_gaps = sum(1 for a, b in zip(frames, frames[1:]) if b - a - 1 > max_age)
        assert len(ids[name]) <= 1 + long_gaps, (name, ids[name], frames)


def test_false_positives_lie_on_or_above_the_ground():
    det = DetectorModel(cfg(conf_threshold=0.0, fp_rate_per_frame=1.0), seed=8)
    intr = intrinsics_from_fovy(70.0, 160, 120)
    cam, trunk = _turning_rig(0.0)
    zs = []
    for _ in range(500):
        for d in det.detect(np.zeros((0, 3)), [], cam, trunk, intr):
            zs.append(float(trunk.inverse_apply(np.array(d["xyz_trunk"]))[0, 2]))
    assert len(zs) > 300
    assert min(zs) >= -1e-3, min(zs)  # xyz_trunk is rounded to 0.1 mm
