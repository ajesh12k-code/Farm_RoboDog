"""ROS-free tests for the browser bridge's pure logic (sim/controllers/web_scene.py)."""
import json
import math
import os

import mujoco
import numpy as np
import pytest

from controllers import web_scene as ws
from controllers.farm_scene import crop_body_names, crop_qpos_slice, load_farm_model

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))


@pytest.fixture(scope="module")
def model():
    return load_farm_model()


@pytest.fixture(scope="module")
def scene(model):
    return ws.build_scene(model)


def _home_qpos(model):
    return np.array(model.key_qpos[0], dtype=np.float64)


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geoms_of(scene, body_id):
    return [g for g in scene["geoms"] if g["body"] == body_id]


# ---------------------------------------------------------------- scene

def test_scene_is_strict_json(scene):
    text = ws.encode_message(scene)
    assert json.loads(text)["type"] == "scene"
    json.dumps(scene, allow_nan=False)  # raises on NaN/inf


def test_scene_geoms_use_supported_types_and_shapes(scene):
    assert scene["geoms"]
    for g in scene["geoms"]:
        assert g["type"] in ws.GEOM_TYPES.values()
        assert len(g["size"]) == 3 and len(g["pos"]) == 3 and len(g["rgba"]) == 4
        assert abs(math.sqrt(sum(v * v for v in g["quat"])) - 1.0) < 1e-4


def test_scene_lists_moving_bodies_only(model, scene):
    ids = [b["id"] for b in scene["bodies"]]
    assert 0 not in ids
    assert ids == sorted(set(ids))
    assert scene["trunk"] == _body_id(model, "trunk") and scene["trunk"] in ids
    for name in crop_body_names(model):
        assert _body_id(model, name) in ids
    for b in scene["bodies"]:
        assert _geoms_of(scene, b["id"]), f"streamed body {b['name']} has no drawable geom"
    assert scene["crops"] == crop_body_names(model)
    assert scene["nq"] == model.nq


def test_floor_plane_is_static_and_sized(scene):
    floor = [g for g in scene["geoms"] if g["name"] == "floor"]
    assert len(floor) == 1
    assert floor[0]["type"] == "plane" and floor[0]["body"] == 0
    assert floor[0]["size"][:2] == [25.0, 25.0]


def test_capsule_from_fromto_uses_half_length_and_midpoint(model, scene):
    # FR thigh: <geom type="capsule" fromto="0 0 0 0 0 -0.23" size="0.026"/>
    caps = [g for g in _geoms_of(scene, _body_id(model, "FR_thigh")) if g["type"] == "capsule"]
    assert len(caps) == 1
    r, half, _ = caps[0]["size"]
    assert r == pytest.approx(0.026) and half == pytest.approx(0.115)
    assert caps[0]["pos"] == pytest.approx([0.0, 0.0, -0.115], abs=1e-6)
    # local z of the capsule must point along -z of the body (fromto direction), i.e. a 180 deg flip
    w, x, y, z = caps[0]["quat"]
    zz = 1.0 - 2.0 * (x * x + y * y)  # R[2][2]
    assert zz == pytest.approx(-1.0, abs=1e-5)
    # FR hip: <geom type="cylinder" fromto="0 0 0 0 -0.065 0" size="0.035"/>
    cyl = [g for g in _geoms_of(scene, _body_id(model, "FR_hip")) if g["type"] == "cylinder"]
    assert len(cyl) == 1
    assert cyl[0]["size"][:2] == pytest.approx([0.035, 0.0325])
    assert cyl[0]["pos"] == pytest.approx([0.0, -0.0325, 0.0], abs=1e-6)


def test_crop_geom_is_red_sphere(model, scene):
    g = _geoms_of(scene, _body_id(model, "crop_item_1"))
    assert len(g) == 1 and g[0]["type"] == "sphere"
    assert g[0]["size"][0] == pytest.approx(ws.CROP_RADIUS)
    assert g[0]["rgba"] == pytest.approx([0.9, 0.15, 0.15, 1.0])


def test_arm_spec_matches_qpos_layout(model, scene):
    arm = scene["arm"]
    assert arm["names"] == list(ws.ARM_JOINTS)
    for i, name in enumerate(ws.ARM_JOINTS):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert int(model.jnt_qposadr[j]) == 7 + i
        assert arm["low"][i] < arm["high"][i]
    assert arm["home"] == pytest.approx([0.0, 0.0, 0.1, -1.8, 1.4, 0.0, 0.0])


# ---------------------------------------------------------------- poses / telemetry

def test_pose_frame_at_home(model, scene):
    data = mujoco.MjData(model)
    ids = [b["id"] for b in scene["bodies"]]
    frame = ws.pose_frame(model, data, ids, _home_qpos(model))
    assert len(frame) == 7 * len(ids)
    t = ids.index(scene["trunk"])
    assert frame[7 * t:7 * t + 7] == pytest.approx([0, 0, 0.34, 1, 0, 0, 0], abs=1e-4)
    c = ids.index(_body_id(model, "crop_item_1"))
    assert frame[7 * c:7 * c + 3] == pytest.approx([0.62, 0.0, 0.035], abs=1e-4)
    assert all(isinstance(v, float) for v in frame)


def test_pose_frame_tracks_qpos(model, scene):
    data = mujoco.MjData(model)
    ids = [b["id"] for b in scene["bodies"]]
    q = _home_qpos(model)
    q[0:3] = [1.5, -0.25, 0.30]
    half = math.radians(90) / 2
    q[3:7] = [math.cos(half), 0, 0, math.sin(half)]  # yaw +90 deg
    k = len(crop_body_names(model))
    q[crop_qpos_slice(k)][:3] = [2.0, 0.5, 0.2]
    frame = ws.pose_frame(model, data, ids, q)
    t = ids.index(scene["trunk"])
    assert frame[7 * t:7 * t + 7] == pytest.approx([1.5, -0.25, 0.30, math.cos(half), 0, 0, math.sin(half)], abs=1e-4)
    c = ids.index(_body_id(model, f"crop_item_{k}"))
    assert frame[7 * c:7 * c + 3] == pytest.approx([2.0, 0.5, 0.2], abs=1e-4)
    # Basket rides 0.055 above the trunk origin, rotated with it.
    b = ids.index(_body_id(model, "dorsal_basket"))
    assert frame[7 * b:7 * b + 3] == pytest.approx([1.5, -0.25, 0.355], abs=1e-4)


def test_pose_frame_rejects_bad_qpos(model, scene):
    data = mujoco.MjData(model)
    ids = [b["id"] for b in scene["bodies"]]
    assert ws.pose_frame(model, data, ids, _home_qpos(model)[:-1]) is None
    q = _home_qpos(model)
    q[5] = float("nan")
    assert ws.pose_frame(model, data, ids, q) is None
    assert ws.pose_frame(model, data, ids, ["x"] * model.nq) is None


def test_telemetry_heading_and_tilt():
    half = math.radians(90) / 2
    t = ws.telemetry([1.0, 2.0, 0.3, math.cos(half), 0.0, 0.0, math.sin(half)])
    assert t["x"] == 1.0 and t["y"] == 2.0 and t["z"] == 0.3
    assert t["heading_deg"] == pytest.approx(90.0)
    assert t["tilt_deg"] == pytest.approx(0.0)
    roll = math.radians(30) / 2
    assert ws.telemetry([0, 0, 0.3, math.cos(roll), math.sin(roll), 0, 0])["tilt_deg"] == pytest.approx(30.0, abs=0.1)
    assert ws.telemetry([0, 0, 0.3]) is None
    assert ws.telemetry([0, 0, float("nan"), 1, 0, 0, 0]) is None


# ---------------------------------------------------------------- JSON helpers

def test_encode_message_nulls_non_finite():
    out = json.loads(ws.encode_message({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": -float("inf")}}))
    assert out == {"a": None, "b": [1.0, None], "c": {"d": None}}


def test_parse_status():
    assert ws.parse_status('{"mode":"tripod","walking":false}') == {"mode": "tripod", "walking": False}
    assert ws.parse_status("[1,2]") is None
    assert ws.parse_status("not json") is None
    assert ws.parse_status("") is None


def test_parse_envelope():
    seg = [[0, 0, 0], [0.1, 0.2, 0.3]]
    env = ws.parse_envelope(json.dumps({"canopy_segments": [seg, [[0, 0], [1, 1]]], "floor_segments": [seg],
                                        "joints_discovered": ["a"]}))
    assert env == {"canopy_segments": [seg], "floor_segments": [seg]}
    assert ws.parse_envelope("") is None
    assert ws.parse_envelope("{bad") is None
    assert ws.parse_envelope('{"canopy_segments": 3}') is None
    assert ws.parse_envelope('{"canopy_segments": [], "floor_segments": []}') is None


# ---------------------------------------------------------------- client messages

LOW, HIGH = [-1.0] * 7, [1.0] * 7


def _parse(obj):
    return ws.parse_client_message(json.dumps(obj) if not isinstance(obj, str) else obj, LOW, HIGH)


def test_cmd_is_clamped_and_defaulted():
    assert _parse({"type": "cmd", "vx": 3, "vy": -0.5, "yaw": -9, "hold": True}) == {
        "type": "cmd", "vx": 1.0, "vy": -0.5, "yaw": -1.0, "hold": True}
    assert _parse({"type": "cmd", "vx": 0.5}) == {"type": "cmd", "vx": 0.5, "vy": 0.0, "yaw": 0.0, "hold": False}


@pytest.mark.parametrize("raw", [
    "not json", "[1]", "null", "{}", '{"type":"nope"}', "[" * 3000,
    '{"type":"cmd","vx":NaN}', '{"type":"cmd","vx":Infinity}', '{"type":"cmd","vx":"1"}',
    '{"type":"cmd","vx":true}', '{"type":"cmd","hold":1}',
    '{"type":"service","name":"self_destruct"}', '{"type":"service"}',
    '{"type":"gait_params","values":[1,1,1]}', '{"type":"gait_params","values":"1,1,1,1"}',
    '{"type":"manual_arm","values":[0,0,0,0,0,0,null]}',
    '{"type":"envelope","show":"yes"}', '{"type":"ping","t":"now"}',
])
def test_bad_messages_raise_bad_message_only(raw):
    with pytest.raises(ws.BadMessage):
        ws.parse_client_message(raw, LOW, HIGH)


def test_binary_and_oversized_frames_rejected():
    with pytest.raises(ws.BadMessage):
        ws.parse_client_message(b'{"type":"ping"}', LOW, HIGH)
    with pytest.raises(ws.BadMessage):
        ws.parse_client_message('{"type":"ping","pad":"' + "x" * ws.MAX_MSG_CHARS + '"}', LOW, HIGH)


def test_service_names_cover_contract():
    assert set(ws.SERVICES) == {"stop", "release_arm", "auto_harvest", "toggle_mode", "reset_world", "work_envelope"}
    for name in ws.SERVICES:
        assert _parse({"type": "service", "name": name}) == {"type": "service", "name": name}
    assert ws.SERVICE_TOPICS["toggle_mode"] == "/nandi/toggle_locomotion_mode"
    assert ws.SERVICE_TOPICS["work_envelope"] == "/compute_work_envelope"


def test_gait_params_clamped_to_contract():
    assert _parse({"type": "gait_params", "values": [9, 0, 1.8, -1]})["values"] == [1.5, 0.5, 1.8, 0.0]


def test_manual_arm_clamped_to_model_limits(model, scene):
    arm = scene["arm"]
    raw = json.dumps({"type": "manual_arm", "values": [5, -9, 0.1, -1.8, 1.4, 0.5, -0.5]})
    vals = ws.parse_client_message(raw, arm["low"], arm["high"])["values"]
    assert vals[0] == pytest.approx(arm["high"][0]) and vals[1] == pytest.approx(arm["low"][1])
    assert vals[2:5] == pytest.approx([0.1, -1.8, 1.4])
    assert vals[5] == pytest.approx(arm["high"][5]) and vals[6] == pytest.approx(arm["low"][6])


def test_envelope_and_ping():
    assert _parse({"type": "envelope", "show": False}) == {"type": "envelope", "show": False}
    assert _parse({"type": "ping", "t": 12.5}) == {"type": "ping", "t": 12.5}


# ---------------------------------------------------------------- cmd_vel / hold

def test_cmd_to_twist_contract():
    assert ws.cmd_to_twist(0.5, -0.25, 0.75, True) == (0.5, -0.25, 1.0, 0.75)
    assert ws.cmd_to_twist(1, 0, 0, False) == (1.0, 0.0, 0.0, 0.0)


def test_hold_tracker_hold_refresh_and_release():
    h, a, b = ws.HoldTracker(1.0), object(), object()
    assert h.tick(0.0) is None
    assert h.on_cmd(a, 1.0, 0.0, 0.2, True, now=0.0) == (1.0, 0.0, 0.2, True)
    assert h.active and h.tick(0.5) == (1.0, 0.0, 0.2, True)
    assert h.on_cmd(a, 1.0, 0.0, 0.2, True, now=0.9) == (1.0, 0.0, 0.2, True)  # refresh
    assert h.tick(1.8) == (1.0, 0.0, 0.2, True)
    assert h.on_disconnect(b) is None and h.active  # someone else leaving changes nothing
    assert h.on_cmd(a, 0.0, 0.0, 0.0, False, now=2.0) == ws.RELEASE
    assert not h.active and h.tick(2.1) is None


def test_hold_tracker_dead_man_expiry():
    h, a = ws.HoldTracker(1.0), object()
    h.on_cmd(a, 0.0, 1.0, 0.0, True, now=10.0)
    assert h.tick(11.0) == (0.0, 1.0, 0.0, True)
    assert h.tick(11.01) == ws.RELEASE
    assert not h.active and h.tick(11.1) is None


def test_hold_tracker_disconnect_and_single_step():
    h, a = ws.HoldTracker(1.0), object()
    h.on_cmd(a, -1.0, 0.0, 0.0, True, now=0.0)
    assert h.on_disconnect(a) == ws.RELEASE and not h.active
    h.on_cmd(a, -1.0, 0.0, 0.0, True, now=0.0)
    assert h.on_cmd(a, 0.0, 0.0, 1.0, False, now=0.1) == (0.0, 0.0, 1.0, False)  # single step clears hold
    assert not h.active
    assert h.on_cmd(a, 0.0, 0.0, 0.0, True, now=0.2) == ws.RELEASE  # zero 'hold' is a release
    h.on_cmd(a, 1.0, 0.0, 0.0, True, now=0.3)
    assert h.release() == ws.RELEASE and not h.active


def test_hold_tracker_default_matches_contract_dead_man():
    # Contract: a hold with no cmd_vel for 0.5 s is released; the bridge must not stretch that.
    h, a = ws.HoldTracker(), object()
    h.on_cmd(a, 1.0, 0.0, 0.0, True, now=0.0)
    assert h.tick(0.5) == (1.0, 0.0, 0.0, True)
    assert h.tick(0.51) == ws.RELEASE


# ---------------------------------------------------------------- WebSocket origin

@pytest.mark.parametrize("origin, host, expected", [
    (None, "localhost:8080", True),                       # non-browser client
    ("", "localhost:8080", True),
    ("http://localhost:8080", "localhost:8080", True),    # the bridge's own page
    ("http://LOCALHOST:8080", "localhost:8080", True),
    ("http://127.0.0.1:8080", "localhost:8080", False),   # different origin string
    ("https://evil.example", "localhost:8080", False),    # a page on another site
    ("http://localhost:3000", "localhost:8080", False),
    ("null", "localhost:8080", False),                    # file:// / sandboxed page
    ("http://localhost:8080", None, False),
])
def test_origin_allowed(origin, host, expected):
    assert ws.origin_allowed(origin, host) is expected


def test_origin_allowed_refuses_dns_rebinding_unless_allowlisted():
    # evil.example resolving to 127.0.0.1: Origin and Host match, but the hostname is not allowed.
    assert ws.origin_allowed("http://evil.example:8080", "evil.example:8080") is False
    allowed = ws.allowed_hostnames("farmbox.lan, Evil.Example")
    assert ws.origin_allowed("http://evil.example:8080", "evil.example:8080", allowed) is True
    assert ws.origin_allowed("http://farmbox.lan:8080", "farmbox.lan:8080", allowed) is True
    assert ws.origin_allowed("http://[::1]:8080", "[::1]:8080") is True


def test_allowed_hostnames_always_includes_loopback():
    assert ws.allowed_hostnames(None) == ws.LOCAL_HOSTNAMES
    assert ws.allowed_hostnames(" , ") == ws.LOCAL_HOSTNAMES
    assert "farmbox.lan" in ws.allowed_hostnames("farmbox.lan")


# ---------------------------------------------------------------- static files

@pytest.fixture()
def web_root(tmp_path):
    root = tmp_path / "web"
    (root / "sub").mkdir(parents=True)
    (root / "index.html").write_text("<html></html>")
    (root / "app.js").write_text("//")
    (root / "sub" / "x.css").write_text("")
    (root / "secret.py").write_text("")
    (tmp_path / "outside.js").write_text("//")
    return str(root)


def test_static_resolution(web_root):
    real = os.path.realpath(web_root)
    assert ws.resolve_static_path(web_root, "/") == os.path.join(real, "index.html")
    assert ws.resolve_static_path(web_root, "/app.js?v=3#x") == os.path.join(real, "app.js")
    assert ws.resolve_static_path(web_root, "/sub/x.css") == os.path.join(real, "sub", "x.css")
    assert ws.resolve_static_path(web_root, "/sub/../app.js") == os.path.join(real, "app.js")
    assert ws.mime_type("a/b.js").startswith("text/javascript")


@pytest.mark.parametrize("path", [
    "/../outside.js", "/%2e%2e/outside.js", "/sub/../../outside.js", "//etc/passwd", "/..%2Foutside.js",
    "/%2e%2e%5coutside.js", "/app.js%00.html", "/missing.js", "/sub", "/secret.py", "/sub/",
])
def test_static_resolution_rejects(web_root, path):
    assert ws.resolve_static_path(web_root, path) is None


def test_static_symlink_escape_rejected(web_root, tmp_path):
    link = os.path.join(web_root, "link.js")
    try:
        os.symlink(str(tmp_path / "outside.js"), link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert ws.resolve_static_path(web_root, "/link.js") is None


def test_repo_web_client_is_servable():
    for name in ("index.html", "app.js", "scene3d.js", "controls.js", "net.js", "style.css"):
        assert ws.resolve_static_path(WEB_DIR, "/" + name) is not None, name
