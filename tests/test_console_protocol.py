"""controllers.console_protocol: console message types, delegation to web_scene, missions, health."""
import base64
import json

import pytest

from atharv_nandi.core import console_protocol as cp
from atharv_nandi.core import web_scene
from atharv_nandi.core import web_auth

LOW = [-0.26, -3.14, -1.6, -2.6, -2.0, 0.0, 0.0]
HIGH = [0.26, 3.14, 2.4, 2.6, 2.0, 0.04, 0.04]


def parse(obj):
    return cp.parse_client_message(json.dumps(obj), LOW, HIGH)


# ---------------------------------------------------------------- client messages

def test_extra_services_contract():
    assert cp.EXTRA_SERVICES == {"estop": "/nandi/estop", "estop_reset": "/nandi/estop_reset",
                                 "mission_abort": "/nandi/mission/abort"}
    for name, topic in web_scene.SERVICE_TOPICS.items():
        assert cp.SERVICE_TOPICS[name] == topic  # base services unchanged


@pytest.mark.parametrize("name", ["estop", "estop_reset", "mission_abort"])
def test_extra_services_parse(name):
    assert parse({"type": "service", "name": name}) == {"type": "service", "name": name}


def test_mission_start_parses_with_zone_and_default():
    assert parse({"type": "mission_start", "zone": "zone_a"}) == {"type": "mission_start", "zone": "zone_a"}
    assert parse({"type": "mission_start"}) == {"type": "mission_start", "zone": "zone_a"}


@pytest.mark.parametrize("zone", ["../etc/passwd", "Zone A", "", 3, "a" * 40, "zone/a"])
def test_mission_start_rejects_bad_zones(zone):
    with pytest.raises(cp.BadMessage):
        parse({"type": "mission_start", "zone": zone})


def test_auth_message_parses_and_requires_token():
    assert parse({"type": "auth", "token": "abc"}) == {"type": "auth", "token": "abc"}
    for bad in ({"type": "auth"}, {"type": "auth", "token": ""}, {"type": "auth", "token": 5}):
        with pytest.raises(cp.BadMessage):
            parse(bad)


@pytest.mark.parametrize("msg", [
    {"type": "cmd", "vx": 2.0, "vy": -0.5, "yaw": 0.1, "hold": True},
    {"type": "service", "name": "auto_harvest"},
    {"type": "gait_params", "values": [1.0, 1.0, 1.8, 1.0]},
    {"type": "manual_arm", "values": [0, 0, 0, 0, 0, 0.02, 0.02]},
    {"type": "envelope", "show": True},
    {"type": "ping", "t": 12.5},
])
def test_known_base_types_are_delegated_unchanged(msg):
    raw = json.dumps(msg)
    assert cp.parse_client_message(raw, LOW, HIGH) == web_scene.parse_client_message(raw, LOW, HIGH)


def test_unknown_types_are_delegated_to_web_scene(monkeypatch):
    seen = []

    def fake(raw, lo, hi):
        seen.append(raw)
        return {"type": "delegated"}

    monkeypatch.setattr(cp.web_scene, "parse_client_message", fake)
    assert parse({"type": "something_new"}) == {"type": "delegated"}
    assert len(seen) == 1


def test_unknown_type_still_rejected_by_delegate():
    with pytest.raises(cp.BadMessage, match="unknown message type"):
        parse({"type": "launch_missiles"})


def test_unknown_service_lists_console_services():
    with pytest.raises(cp.BadMessage, match="estop"):
        parse({"type": "service", "name": "self_destruct"})


@pytest.mark.parametrize("raw", [b"\x00", "x" * 5000, "{", "[]"])
def test_malformed_frames_raise_bad_message(raw):
    with pytest.raises(cp.BadMessage):
        cp.parse_client_message(raw, LOW, HIGH)


def test_estop_parsed_message_is_permitted_for_viewer():
    msg = parse({"type": "service", "name": "estop"})
    assert web_auth.permit(web_auth.ROLE_VIEWER, msg["type"], msg["name"])[0]
    msg = parse({"type": "mission_start", "zone": "zone_a"})
    assert not web_auth.permit(web_auth.ROLE_VIEWER, msg["type"], msg.get("name"))[0]


def test_extend_scene_adds_services_without_mutating_base():
    base = {"type": "scene", "services": list(web_scene.SERVICES)}
    out = cp.extend_scene(base)
    assert "estop" in out["services"] and "estop" not in base["services"]
    assert out["console"]["camera_topic"] == "/nandi/camera/head/compressed"


# ---------------------------------------------------------------- E-STOP latch

def test_estop_latch_blocks_motion_only():
    assert cp.blocked_by_estop({"type": "cmd", "vx": 0.5, "vy": 0, "yaw": 0, "hold": True})
    assert not cp.blocked_by_estop({"type": "cmd", "vx": 0, "vy": 0, "yaw": 0, "hold": False})  # release ok
    assert cp.blocked_by_estop({"type": "manual_arm", "values": []})
    assert cp.blocked_by_estop({"type": "mission_start", "zone": "zone_a"})
    assert cp.blocked_by_estop({"type": "service", "name": "auto_harvest"})
    for name in ("stop", "estop", "estop_reset", "release_arm", "reset_world", "mission_abort"):
        assert not cp.blocked_by_estop({"type": "service", "name": name})


# ---------------------------------------------------------------- missions

FEATURES = {"type": "FeatureCollection", "properties": {"zone": "zone_a"},
            "features": [{"type": "Feature", "properties": {"zone": "zone_a"},
                          "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}]}


def test_mission_command_from_zone_file(tmp_path):
    d = tmp_path / "config" / "missions"
    d.mkdir(parents=True)
    (d / "zone_a.geojson").write_text(json.dumps(FEATURES))
    text, message = cp.mission_command("zone_a", str(d), str(tmp_path))
    cmd = json.loads(text)
    assert cmd["action"] == "start" and cmd["zone"] == "zone_a"
    assert cmd["source"] == "config/missions/zone_a.geojson"
    assert cmd["mission"] == FEATURES
    assert "zone_a" in message
    assert cp.list_zones(str(d)) == ["zone_a"]


def test_mission_command_finds_zone_by_properties(tmp_path):
    (tmp_path / "pilot_field.geojson").write_text(json.dumps(FEATURES))
    text, _ = cp.mission_command("zone_a", str(tmp_path))
    assert json.loads(text)["zone"] == "zone_a"


def test_mission_command_missing_package_or_zone(tmp_path):
    text, message = cp.mission_command("zone_a", str(tmp_path / "nope"))
    assert text is None and "mission package" in message
    (tmp_path / "zone_b.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    (tmp_path / "broken.geojson").write_text("{not json")
    text, message = cp.mission_command("zone_a", str(tmp_path))
    assert text is None and "zone_a" in message
    assert cp.mission_command("../x", str(tmp_path))[0] is None


# ---------------------------------------------------------------- optional topics

def test_parse_ext_status_and_garbage():
    assert cp.parse_ext("safety", '{"estop": true, "reason": "bumper"}') == {"estop": True, "reason": "bumper"}
    assert cp.parse_ext("safety", "not json") is None
    assert cp.parse_ext("mission", "[1, 2]") is None


def test_parse_detections_normalises_shapes():
    text = json.dumps({"frame": "world", "detections": [
        {"x": 1, "y": 2, "z": 0.1, "label": "tomato", "confidence": 0.91234},
        {"position": [0.5, -0.5, 0.0], "score": 0.5},
        {"pos": {"x": 0.1, "y": 0.2}},
        {"x": "bad"}, "junk",
    ]})
    out = cp.parse_ext("detections", text)
    assert out["frame"] == "world"
    assert [d["xyz"] for d in out["items"]] == [[1.0, 2.0, 0.1], [0.5, -0.5, 0.0], [0.1, 0.2, 0.0]]
    assert out["items"][0]["label"] == "tomato" and out["items"][0]["confidence"] == 0.912
    bare = cp.parse_ext("detections", json.dumps([{"x": 0, "y": 0}] * 500))
    assert len(bare["items"]) == cp.MAX_DETECTIONS


def test_camera_frame_and_rate_limit():
    f = cp.camera_frame("jpeg", b"\xff\xd8abc", 12.0)
    assert f["type"] == "camera" and f["format"] == "jpeg"
    assert base64.b64decode(f["data"]) == b"\xff\xd8abc"
    assert cp.camera_frame("rgb8; png compressed", b"x", 0)["format"] == "png"
    assert cp.camera_frame("tiff", b"x", 0) is None
    assert cp.camera_frame("jpeg", b"x" * (cp.CAMERA_MAX_BYTES + 1), 0) is None
    rl = cp.RateLimiter(cp.CAMERA_MAX_HZ)
    accepted = sum(rl.accept(i * 0.05) for i in range(200))  # 20 Hz for 10 s
    assert accepted <= 21  # <= 2 Hz


# ---------------------------------------------------------------- health

def test_age_levels():
    assert [cp.age_level(a) for a in (None, 0.1, 2.5, 9.0)] == ["n/a", "ok", "warn", "fail"]


def test_node_health_prefers_supervisor_verdicts():
    ages = {"sim": 0.02, "loco": 3.0, "safety": 0.3, "camera": None}
    safety = {"nodes": {"loco": "ok", "perception": {"age_s": 9.0}, "mission": False, "x": {"state": "warn"}}}
    h = cp.node_health(ages, safety)
    assert h == {"sim": "ok", "loco": "ok", "safety": "ok", "camera": "n/a",
                 "perception": "fail", "mission": "fail", "x": "warn"}


def test_health_report_ok_and_stale():
    code, body = cp.health_report(0.05, {"sim": 0.05}, None)
    assert code == 200 and body["ok"] and body["last_qpos_age_s"] == 0.05 and body["nodes"]["sim"] == "ok"
    code, body = cp.health_report(2.5, {"sim": 2.5}, None)
    assert code == 503 and not body["ok"]
    code, body = cp.health_report(None, {"sim": None}, None)
    assert code == 503 and body["last_qpos_age_s"] is None
    json.dumps(body)  # serialisable


def test_node_health_ignores_stale_supervisor_verdicts():
    safety = {"nodes": {"sim": "ok", "perception": "ok", "safety": "ok"}}
    h = cp.node_health({"sim": 0.01, "safety": 120.0, "perception": None}, safety)
    assert h == {"sim": "ok", "safety": "fail", "perception": "n/a"}
    # Fresh supervisor: its verdicts apply, except about itself.
    h = cp.node_health({"sim": 0.01, "safety": 0.2, "perception": None}, {"nodes": {"perception": "ok", "safety": "fail"}})
    assert h == {"sim": "ok", "safety": "ok", "perception": "ok"}


def test_fresh_ext_drops_stale_and_unseen_payloads():
    ext = {"safety": {"estop": True}, "mission": {"state": "run"}, "battery": None}
    out = cp.fresh_ext(ext, {"safety": cp.EXT_STALE_S + 1, "mission": 0.5, "battery": None})
    assert out == {"safety": None, "mission": {"state": "run"}, "battery": None}
    assert ext["safety"] == {"estop": True}  # input untouched


# ---------------------------------------------------------------- persisted E-STOP latch

def test_estop_latch_survives_restart(tmp_path):
    path = str(tmp_path / "runs" / "console_estop.json")
    latch = cp.EstopLatch(path)
    assert not latch.latched
    seq = latch.latch("c1@1.2.3.4:5", 100.0)
    assert latch.set_message(seq, "E-STOP: stopped")
    again = cp.EstopLatch(path)  # bridge restart
    assert again.latched and again.state["by"] == "c1@1.2.3.4:5" and again.state["t"] == 100.0
    assert "before the bridge restarted" in again.state["message"]
    assert again.clear(again.seq)
    assert not cp.EstopLatch(path).latched


def test_estop_latch_corrupt_file_fails_safe(tmp_path):
    path = tmp_path / "console_estop.json"
    path.write_text("{not json")
    latch = cp.EstopLatch(str(path))
    assert latch.latched and "unreadable" in latch.state["message"]


def test_estop_pressed_during_reset_stays_latched(tmp_path):
    latch = cp.EstopLatch(str(tmp_path / "e.json"))
    first = latch.latch("c1", 1.0)
    reset_started_at = latch.seq          # reset begins, awaits /nandi/estop_reset ...
    second = latch.latch("c2", 2.0)       # ... E-STOP pressed meanwhile
    assert not latch.clear(reset_started_at)
    assert latch.latched and latch.state["by"] == "c2"
    assert not latch.set_message(first, "stale outcome")  # older press cannot overwrite the newer one
    assert latch.set_message(second, "E-STOP: ok")
    assert latch.clear(latch.seq) and not latch.latched
    assert not latch.set_message(second, "late outcome after reset")
    assert not latch.latched


def test_estop_latch_without_path_is_memory_only():
    latch = cp.EstopLatch(None)
    latch.latch("c1", 1.0)
    assert latch.latched and latch.errors == 0


def test_audit_values_per_type():
    assert cp.audit_values({"type": "cmd", "vx": 0.5, "vy": 0.0, "yaw": 0.0, "hold": True}) == [0.5, 0.0, 0.0, True]
    assert cp.audit_values({"type": "manual_arm", "values": [0.1, 0.2]}) == [0.1, 0.2]
    assert cp.audit_values({"type": "mission_start", "zone": "zone_a"}) == "zone_a"
    assert cp.audit_values({"type": "envelope", "show": True}) is True
    assert cp.audit_values({"type": "service", "name": "stop"}) is None


# ================================================================ WP10 orchard console (MP 0.7, OS 5.5)

import os  # noqa: E402

import yaml  # noqa: E402

from atharv_nandi.core import records as rec  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _field(world="orchard_demo"):
    with open(os.path.join(REPO, "config", "fields", f"{world}.json"), encoding="utf-8") as fh:
        f = json.load(fh)
    f["world"] = world
    return f


def _tour():
    with open(os.path.join(REPO, "config", "missions", "orchard_tour.yaml"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


NOMINAL_IN = {"link_local": {"state": "up"}, "link_cloud": {"state": "up"}, "gnss": {"rtk": "FIXED"},
              "battery": {"soc": 0.9}}


@pytest.fixture()
def hub():
    h = cp.ConsoleHub(_field(), sim_local_link=True, tour_doc=_tour())
    for k, v in NOMINAL_IN.items():
        h.ingest(k, json.dumps(v), 0.0)
    return h


def _ctx(h, now=0.5, **kw):
    for k, v in NOMINAL_IN.items():   # keep the platform inputs fresh
        h.ingest(k, json.dumps(v), now)
    return h.ready_context(now, (0.0, 0.0), **kw)


def _cmd(obj):
    return cp.parse_any(json.dumps(obj), LOW, HIGH)


def _start(h, preset, now, targets=None, confirm=False, estop=False):
    ctx = _ctx(h, now)
    msg = _cmd({"type": "preset_start", "preset": preset, "targets": targets or {}, "confirm": confirm})
    return h.handle(msg, now, ctx, estop)


def _out(h, key=None):
    out = h.take_outbox()
    return [p for k, p in out if key is None or k == key]


# ---------------------------------------------------------------- client schemas (MP 0.7)

GOOD_CONSOLE = [
    {"type": "preset_start", "preset": "tree360", "targets": {"tree_ids": ["OT-1"]}, "params": {"stations": 8}},
    {"type": "preset_start", "preset": "scan", "targets": {"row_ids": ["R1"]}},
    {"type": "preset_start", "preset": "fruit_scout", "targets": {"block_id": "B1-mango-hdp"}},
    {"type": "preset_start", "preset": "safe_park"},
    {"type": "preset_pause"}, {"type": "preset_resume"}, {"type": "preset_abort"}, {"type": "preset_abort", "qid": "q3"},
    {"type": "tree_mark", "tree_id": "R1-2", "kind": "fruit", "clock": 3, "h_class": "upar"},
    {"type": "tree_mark", "tree_id": "R1-2", "kind": "prune", "h_m": 1.1, "side": "E", "tool_hint": "lopper"},
    {"type": "assist_reply", "reply": "take_over"}, {"type": "assist_reply", "reply": "skip"},
    {"type": "assist_reply", "reply": "halt"},
    {"type": "queue_reorder", "order": ["q2", "q1"]},
    {"type": "jobsheet_request", "row_id": "R1"}, {"type": "jobsheet_request", "row_id": "R1", "lang": "en"},
    {"type": "recovery", "op": "run_safe_park"}, {"type": "recovery", "op": "check", "item": "two_people"},
    {"type": "crown", "band": "mid"}, {"type": "link_test", "link": "local", "dur_s": 6},
]


@pytest.mark.parametrize("msg", GOOD_CONSOLE, ids=lambda m: m["type"])
def test_console_messages_parse(msg):
    out = _cmd(msg)
    assert out["type"] == msg["type"]
    json.dumps(out, allow_nan=False)


BAD_CONSOLE = [
    {"type": "preset_start"},                                              # no preset
    {"type": "preset_start", "preset": "../etc"},                           # not an id
    {"type": "preset_start", "preset": "tree360", "targets": ["OT-1"]},     # targets not an object
    {"type": "preset_start", "preset": "tree360", "targets": {"tree_ids": []}},
    {"type": "preset_start", "preset": "tree360", "targets": {"tree_ids": ["OT-1"], "row_ids": ["R1"]}},
    {"type": "preset_start", "preset": "tree360", "targets": {"trees": ["OT-1"]}},
    {"type": "preset_start", "preset": "tree360", "params": {"x": {"a": {"b": 1}}}},
    {"type": "preset_start", "preset": "tree360", "params": {"x": float("nan")}},
    {"type": "preset_start", "preset": "tree360", "confirm": "yes"},
    {"type": "preset_abort", "qid": 3},
    {"type": "tree_mark", "tree_id": "R1-2", "kind": "spray", "h_class": "upar"},
    {"type": "tree_mark", "tree_id": "R1-2", "kind": "fruit", "clock": 13, "h_class": "upar"},
    {"type": "tree_mark", "tree_id": "R1-2", "kind": "fruit"},              # no height
    {"type": "tree_mark", "tree_id": "R1-2", "kind": "fruit", "h_m": 9.0},
    {"type": "tree_mark", "tree_id": "R1-2", "kind": "fruit", "h_class": "upar", "side": "NE"},
    {"type": "assist_reply", "reply": "retry"},
    {"type": "queue_reorder", "order": "q1"},
    {"type": "jobsheet_request", "row_id": "R1", "lang": "fr"},
    {"type": "recovery", "op": "confirm_local"},                            # never from the console (WP28)
    {"type": "recovery", "op": "check", "item": "nothing"},
    {"type": "crown", "band": "max"},
    {"type": "link_test", "dur_s": 600},
]


@pytest.mark.parametrize("msg", BAD_CONSOLE)
def test_bad_console_messages_raise_bad_message(msg):
    with pytest.raises(cp.BadMessage):
        cp.parse_any(json.dumps(msg, allow_nan=True), LOW, HIGH)


def test_console_parse_leaves_other_types_to_the_base_protocol():
    assert cp.parse_any(json.dumps({"type": "ping", "t": 1.0}), LOW, HIGH)["type"] == "ping"
    assert cp.parse_any(json.dumps({"type": "service", "name": "estop"}), LOW, HIGH)["name"] == "estop"
    with pytest.raises(cp.BadMessage):
        cp.parse_any("x" * (web_scene.MAX_MSG_CHARS + 1), LOW, HIGH)


@pytest.mark.parametrize("kind", cp.CONSOLE_TYPES)
def test_viewer_refused_every_console_command_operator_allowed(kind):
    ok, why = web_auth.permit(web_auth.ROLE_VIEWER, kind)
    assert not ok and "read-only" in why
    assert web_auth.permit(web_auth.ROLE_OPERATOR, kind)[0]
    assert web_auth.permit(web_auth.ROLE_DEV, kind)[0]
    assert not web_auth.permit(None, kind)[0]
    assert web_auth.permit(web_auth.ROLE_VIEWER, "service", "estop")[0]   # the E-STOP is never blocked


def test_console_commands_are_audit_logged(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = cp.ConsoleAuditLog(str(path))
    for i, m in enumerate(GOOD_CONSOLE):
        msg = _cmd(m)
        assert log.record(100.0 + 2 * i, "c1", "operator", msg["type"], msg.get("name"), cp.console_audit_values(msg))
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == len(GOOD_CONSOLE)
    assert {r["type"] for r in rows} == {m["type"] for m in GOOD_CONSOLE}
    first = rows[0]
    assert first["role"] == "operator" and first["values"][0] == "tree360" and "OT-1" in first["values"][1]
    assert log.record(200.0, "c1", "operator", "preset_pause", None, None)
    assert not log.record(200.5, "c1", "operator", "preset_pause", None, None)    # 1 s dedupe
    assert log.record(201.6, "c1", "operator", "preset_pause", None, None)
    assert log.record(201.7, "c1", "operator", "service", "estop", None)          # every E-STOP still logged


# ---------------------------------------------------------------- rate and byte budget

def test_stream_limiter_caps_each_stream_at_2_hz():
    lim = cp.StreamLimiter()
    sent = {"a": 0, "b": 0}
    for k in range(1000):            # 10 s offered at 100 Hz, payload changing every time
        now = k * 0.01
        for s in sent:
            sent[s] += lim.offer(s, f"{s}{k}", now)
    assert all(15 <= v <= 21 for v in sent.values()), sent
    lim2 = cp.StreamLimiter()        # unchanged payload: once, then the 5 s keep-alive only
    n = sum(lim2.offer("x", "same", k * 0.01) for k in range(1000))
    assert n == 2


def test_hub_streams_never_exceed_2_hz(hub):
    counts = {}
    for k in range(200):             # 20 s of 10 Hz ticks with the coverage / mast inputs changing each tick
        now = 1.0 + k * 0.1
        hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "arc", "station": k % 8, "stations": 8,
                                          "coverage_pct": {"low": k % 100}}), now)
        hub.ingest("mast", json.dumps({"ext_m": (k % 16) / 10.0, "arbiter_source": "preset"}), now)
        ctx = _ctx(hub, now)
        hub.tick(now, ctx, False, True)
        for m in hub.streams(now, ctx):
            assert cp.validate_server_message(m) == [], m["type"]
            counts[m["type"]] = counts.get(m["type"], 0) + 1
    assert counts["coverage"] <= 41 and counts["mast"] <= 41, counts
    assert all(v <= 41 for v in counts.values())


def test_byte_budget_holds_1_mib_per_s():
    b = cp.ByteBudget()
    frame = 30_000                    # an oversized 30 kB state frame at 60 Hz = 1.8 MB/s offered
    sent = []
    for k in range(600):
        now = k / 60.0
        if b.allow(frame, now):
            sent.append((now, frame))
    for k in range(600):              # every 1 s window, wherever it starts
        t0 = k / 60.0
        assert sum(n for t, n in sent if t0 <= t < t0 + 1.0) <= cp.BUDGET_BYTES_PER_S
    assert b.dropped > 0 and len(sent) > 300
    assert b.allow(600_000, 11.0, must=True)    # events (scene, E-STOP results) always pass ...
    assert not b.allow(frame * 20, 11.1)        # ... and droppable frames give way after them


# ---------------------------------------------------------------- server schemas

def test_server_messages_match_schemas(hub):
    ctx = _ctx(hub)
    msgs = hub.snapshot(0.5, ctx)
    assert {m["type"] for m in msgs} >= {"map", "preset_status", "coverage", "mast", "recovery"}
    for m in msgs:
        assert cp.validate_server_message(m) == [], (m["type"], cp.validate_server_message(m))
    assert cp.validate_server_message({"type": "mast"})                      # missing keys are reported
    assert cp.validate_server_message({"type": "nope"}) == ["unknown type 'nope'"]


def test_map_has_every_tree_tag_lateral_and_annulus(hub):
    f = _field()
    m = hub.map
    assert {t["tree_id"] for t in m["trees"]} == {t["tree_id"] for t in f["trees"]}
    for t in m["trees"]:
        src = next(x for x in f["trees"] if x["tree_id"] == t["tree_id"])
        assert t["tag"] == src["physical_tag"] and t["xy"] == [src["geo"]["x"], src["geo"]["y"]]
        assert 1.2 <= m["annuli"][t["tree_id"]]["r1"] <= 3.0
    assert len(m["laterals"]) == len(f["fixtures"]["laterals"]) and all(len(x["points"]) >= 2 for x in m["laterals"])
    assert m["geofence"] and m["home"] == [0.0, 0.0]
    assert {r["row_id"] for r in m["rows"]} == {r["row_id"] for r in f["rows"]}
    assert m["annuli"]["OT-1"]["r1"] == 1.9                                  # canopy 1.0 + standoff 0.9


def test_legacy_world_has_an_empty_map_and_refuses_tree_presets():
    h = cp.ConsoleHub({"world": "legacy"})
    assert h.map["trees"] == [] and h.map["world"] == "legacy"
    ok, why, _ = _start(h, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    assert not ok and why.startswith("registry_mismatch")


# ---------------------------------------------------------------- preset bar and Ready check

def test_tiles_carry_reasons_season_and_executor(hub):
    tiles = {t["id"]: t for t in hub.preset_status(0.5, _ctx(hub))["tiles"]}
    assert len(tiles) == 21
    assert tiles["tree360"]["enabled"] and tiles["tree360"]["executor"] == "tree360" and tiles["tree360"]["live"]
    assert tiles["work_spray"]["enabled"] is False and tiles["work_spray"]["reason"] == "not_in_mvp"
    assert tiles["estop_recovery"]["reason"] == "estop_not_engaged"
    assert "demo_all_fruiting" in (tiles["harvest_mango"]["season"] or {}).get("label", "")
    assert tiles["harvest_mango"]["executor"] == "demo" and not tiles["harvest_mango"]["live"]
    for t in tiles.values():
        assert t["hi"] and t["en"] and t["level"] in ("go", "degraded", "no_go", None)


def test_ready_check_refuses_with_the_reason_code():
    h = cp.ConsoleHub(_field())
    ctx = h.ready_context(1.0, (0.0, 0.0))                    # no link / gnss messages yet: unknown, NONE
    ok, why, _ = h.handle(_cmd({"type": "preset_start", "preset": "tree360", "targets": {"tree_ids": ["OT-1"]}}),
                          1.0, ctx)
    assert not ok and "rtk" in why and "local_link_lost" in why
    tiles = {t["id"]: t for t in h.preset_status(1.0, ctx)["tiles"]}
    assert not tiles["tree360"]["enabled"] and tiles["tree360"]["reason"] in ("rtk", "local_link_lost")


def test_degraded_start_needs_the_operator_confirm(hub):
    hub.ingest("weather", json.dumps({"month": 11, "hour": 10, "rain": False, "rain_mm_h": 0.0, "wet_foliage": False,
                                      "temp_c": 39.5}), 0.5)          # > 38 C: degraded (heat)
    ok, why, _ = _start(hub, "tree360", 0.6, {"tree_ids": ["OT-1"]})
    assert not ok and why.startswith("confirm_required") and "heat" in why
    ok, why, _ = _start(hub, "tree360", 0.7, {"tree_ids": ["OT-1"]}, confirm=True)
    assert ok, why


def test_estop_blocks_starts_except_estop_recovery(hub):
    ok, why, _ = _start(hub, "tree360", 0.6, {"tree_ids": ["OT-1"]}, estop=True)
    assert not ok and why.startswith("estop")


# ---------------------------------------------------------------- TREE 360 through the WP25 hook + drive relay

def test_tree360_runs_through_the_hook_with_pause_resume_and_live_drive(hub):
    ok, why, _ = _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    assert ok and "queued as q1" in why
    hub.tick(1.1, _ctx(hub, 1.1), False, True)
    out = hub.take_outbox()
    start = [p for k, p in out if k == "tree360_cmd"]
    assert start and start[0]["cmd"] == "start" and start[0]["tree_id"] == "OT-1"
    assert set(start[0]["params"]) <= {"arc_mode", "standoff_m", "stations", "bands", "settle_s", "capture_s", "v_mps",
                                       "crown_walk", "run_id", "reason"}
    # OT-1 alley clear height < the high band: WP27 adjusts to snake_mid; TREE 360 has no such mode -> stop_and_go
    assert start[0]["params"]["crown_walk"] == "stop_and_go"
    assert any(k == "cloud_beat" for k, _ in out) and any(k == "local_beat" for k, _ in out)
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "approach", "stations": 8, "station": 0}), 1.5)
    hub.ingest("drive_tree360", json.dumps({"vx": 0.6, "yaw": -0.2}), 1.5)
    hub.tick(1.6, _ctx(hub, 1.6))
    assert hub.drive(1.6) == (0.6, 0.0, -0.2, True) and hub.relaying
    assert hub.drive(1.7, estop=True) == web_scene.RELEASE      # the E-STOP releases the relay at once
    assert hub.drive(1.8, estop=True) is None
    ok, _, _ = hub.handle(_cmd({"type": "preset_pause"}), 2.0, _ctx(hub, 2.0))
    assert ok and _out(hub, "tree360_cmd") == [{"cmd": "pause"}]
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "arc", "paused": True}), 2.2)
    assert hub.drive(2.3) is None                                 # paused run: nothing relayed
    ok, _, _ = hub.handle(_cmd({"type": "preset_resume"}), 2.5, _ctx(hub, 2.5))
    assert ok and _out(hub, "tree360_cmd") == [{"cmd": "resume"}]
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "arc", "paused": False, "station": 3,
                                      "stations": 8, "coverage_pct": {"low": 40, "mid": 20, "high": 0}}), 3.0)
    assert hub.drive(3.0) is None                                 # the 1.5 s drive is stale (> 0.5 s): not replayed
    hub.ingest("drive_tree360", json.dumps({"vx": 0.6, "yaw": -0.2}), 3.0)
    assert hub.drive(3.0) == (0.6, 0.0, -0.2, True)
    cov = hub.coverage(3.0)
    assert cv_ok(cov) and cov["coverage_pct"] == {"low": 40.0, "mid": 20.0, "high": 0.0}
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "done", "station": 8, "stations": 8}), 9.0)
    hub.tick(9.1, _ctx(hub, 9.1))
    assert hub.active is None and hub.history[0]["status"] == "done"
    assert hub.drive(9.2) == web_scene.RELEASE and hub.drive(9.3) is None


def cv_ok(msg):
    return cp.validate_server_message(msg) == []


def test_tree360_without_the_hook_is_reported_not_silently_lost(hub):
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    hub.tick(1.1, _ctx(hub, 1.1))
    hub.tick(1.1 + cp.TREE360_START_TIMEOUT_S + 0.5, _ctx(hub, 8.0))
    assert hub.active is None and hub.history[0]["status"] == "no_live_executor"


def test_tree360_hook_error_refuses_the_item(hub):
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    hub.tick(1.1, _ctx(hub, 1.1))
    hub.ingest("tree360", json.dumps({"error": "tree360: run already active"}), 1.3)
    hub.tick(1.4, _ctx(hub, 1.4))
    assert hub.history[0]["status"] == "refused" and "already active" in hub.history[0]["reason"]


def test_drive_relay_priority_and_stale_release(hub):
    hub.ingest("drive_work", json.dumps({"cmd": [0.4, 0.0, 0.1], "stop": False}), 1.0)
    assert hub.drive(1.1) == (0.4, 0.0, 0.1, True)
    hub.ingest("drive_recovery", json.dumps({"vx": -0.3, "vy": 0.0, "yaw": 0.0, "release": False}), 1.2)
    assert hub.drive(1.3) == (-0.3, 0.0, 0.0, True)                  # recovery beats work assist
    hub.ingest("drive_recovery", json.dumps({"vx": 0.0, "release": True, "lock": True}), 1.35)
    assert hub.drive(1.38) == web_scene.RELEASE                      # recovery holds the robot locked: no work drive
    hub.ingest("drive_recovery", json.dumps({"vx": 0.0, "release": True}), 1.4)
    assert hub.drive(1.45) == (0.4, 0.0, 0.1, True)                  # released, unlocked: work drive (fresh)
    assert hub.drive(2.0) == web_scene.RELEASE                       # every drive stale -> one release
    assert hub.drive(2.1) is None


def test_estop_aborts_the_running_item_and_pauses_the_queue(hub):
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    _start(hub, "tree360", 1.0, {"tree_ids": ["R1-1"]})
    hub.tick(1.1, _ctx(hub, 1.1))
    hub.take_outbox()
    hub.tick(2.0, _ctx(hub, 2.0, estop=True), estop=True)
    assert {"cmd": "abort", "reason": "operator"} in _out(hub, "tree360_cmd")
    assert hub.paused and hub.paused_reason == "estop"
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "aborted", "abort_reason": "estop"}), 2.5)
    hub.tick(3.0, _ctx(hub, 3.0), False)
    assert hub.active is None and hub.history[0]["status"] == "aborted"
    assert len(hub.queue.items) == 1                                  # nothing dispatched while paused
    ok, _, _ = hub.handle(_cmd({"type": "preset_resume"}), 4.0, _ctx(hub, 4.0))
    assert ok and not hub.paused


def test_queue_reorder_and_remove(hub):
    for t in ("OT-1", "R1-1", "R1-2"):
        hub.paused = True
        _start(hub, "tree360", 1.0, {"tree_ids": [t]})
    ids = [i.qid for i in hub.queue.items]
    ok, _, _ = hub.handle(_cmd({"type": "queue_reorder", "order": list(reversed(ids))}), 1.1, _ctx(hub, 1.1))
    assert ok and [i.qid for i in hub.queue.items] == list(reversed(ids))
    ok, why, _ = hub.handle(_cmd({"type": "queue_reorder", "order": ids[:1]}), 1.2, _ctx(hub, 1.2))
    assert not ok and why.startswith("command_invalid")
    ok, _, _ = hub.handle(_cmd({"type": "preset_abort", "qid": ids[1]}), 1.3, _ctx(hub, 1.3))
    assert ok and ids[1] not in [i.qid for i in hub.queue.items]
    st = hub.preset_status(1.4, _ctx(hub, 1.4))
    assert st["queue"]["n"] == 2 and all(cp.validate_server_message(st) == [] for _ in [0])
    for it in st["queue"]["items"]:
        assert it["eta_end_s"] > it["eta_start_s"] >= 0 and 0 < it["soc_end"] <= 1 and it["disk_free_end_mb"] > 0


def test_assist_reply_take_over_skip_halt(hub):
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    hub.tick(1.1, _ctx(hub, 1.1))
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "arc", "assist": {"reason": "person_hold"}}), 1.5)
    assert hub.preset_status(1.5, _ctx(hub, 1.5))["assist"]["reason"] == "person_hold"
    hub.take_outbox()
    ok, _, _ = hub.handle(_cmd({"type": "assist_reply", "reply": "take_over"}), 1.6, _ctx(hub, 1.6))
    assert ok and _out(hub, "tree360_cmd") == [{"cmd": "pause"}] and hub.active["paused"]
    ok, _, _ = hub.handle(_cmd({"type": "assist_reply", "reply": "halt"}), 1.7, _ctx(hub, 1.7))
    assert ok and {"cmd": "abort", "reason": "operator"} in _out(hub, "tree360_cmd") and hub.paused


# ---------------------------------------------------------------- recovery menu (WP28)

def test_recovery_arms_first_then_runs_safe_park(hub):
    ok, _, _ = hub.handle(_cmd({"type": "recovery", "op": "run_safe_park"}), 1.0, _ctx(hub, 1.0))
    assert ok and _out(hub, "recovery_cmd") == [{"arm": True}]        # one message per sim loop (live inbox)
    hub.tick(1.2, _ctx(hub, 1.2))
    assert _out(hub, "recovery_cmd") == []                            # not armed yet: wait
    hub.ingest("recovery", json.dumps({"armed": True, "active": None}), 1.3)
    hub.tick(1.4, _ctx(hub, 1.4))
    assert _out(hub, "recovery_cmd") == [{"run": "safe_park"}]
    for op, payload in (("ack_resume", {"ack": "resume"}), ("ack_halt", {"ack": "halt"}),
                        ("confirm_operator", {"confirm": "operator"}), ("abort", {"abort": True})):
        hub.handle(_cmd({"type": "recovery", "op": op}), 2.0, _ctx(hub, 2.0))
        assert _out(hub, "recovery_cmd") == [payload]
    assert all(k != "recovery_local" for k in cp.HUB_TOPICS_OUT)
    assert "/nandi/recovery/local" not in cp.HUB_TOPICS_OUT.values()   # the console never sends the local confirm


def test_safe_park_preset_follows_the_recovery_state(hub):
    hub.ingest("recovery", json.dumps({"armed": True, "active": None}), 0.9)
    _start(hub, "safe_park", 1.0)
    hub.tick(1.1, _ctx(hub, 1.1))
    assert _out(hub, "recovery_cmd") == [{"run": "safe_park"}]
    hub.ingest("recovery", json.dumps({"armed": True, "active": "safe_park", "behaviour": {"step": "move"}}), 2.0)
    hub.tick(2.1, _ctx(hub, 2.1))
    assert hub.active["state"] == "running"
    rs = hub.recovery(2.1)
    assert cv_ok(rs) and rs["active"] == "safe_park" and rs["links"]["local"]["sim"] is True
    hub.ingest("recovery", json.dumps({"armed": True, "active": None, "history": [{"outcome": "resumed"}]}), 9.0)
    hub.tick(9.1, _ctx(hub, 9.1))
    assert hub.history[0]["status"] == "done"


def test_link_test_is_a_sim_tool_only():
    h = cp.ConsoleHub(_field(), sim_tools=False)
    ok, why, _ = h.handle(_cmd({"type": "link_test", "link": "local", "dur_s": 6}), 1.0, h.ready_context(1.0))
    assert not ok and "sim tool" in why
    h2 = cp.ConsoleHub(_field(), sim_tools=True)
    ok, _, _ = h2.handle(_cmd({"type": "link_test", "link": "local", "dur_s": 6}), 1.0, h2.ready_context(1.0))
    assert ok and _out(h2, "link_cmd") == [{"drop": {"link": "local", "dur": 6.0}}]


def test_crown_requests_go_to_the_arbiter_but_not_during_tree360_or_estop(hub):
    ok, _, _ = hub.handle(_cmd({"type": "crown", "band": "mid"}), 1.0, _ctx(hub, 1.0))
    assert ok and _out(hub, "mast_cmd") == [{"source": "preset", "band": "mid", "reason": "operator"}]
    ok, why, _ = hub.handle(_cmd({"type": "crown", "band": "high"}), 1.0, _ctx(hub, 1.0), True)
    assert not ok and why.startswith("estop")
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    hub.tick(1.1, _ctx(hub, 1.1))
    ok, why, _ = hub.handle(_cmd({"type": "crown", "band": "stow"}), 1.2, _ctx(hub, 1.2))
    assert not ok and why.startswith("crown_gate")


def test_mast_gauge_shows_extension_gate_and_arbiter_source(hub):
    hub.ingest("mast", json.dumps({"ext_m": 0.6, "target_ext": 1.6, "gated": True, "reason": "wind",
                                   "detail": "wind:9.1", "arbiter_source": "safety", "has_mast": True}), 1.0)
    hub.ingest("wind", json.dumps({"speed_mps": 9.1, "mean_mps": 8.4, "gust": True}), 1.0)
    m = hub.mast(1.1)
    assert cv_ok(m) and m["band"] == "mid" and m["gated"] and m["arbiter_source"] == "safety"
    assert m["wind"]["gust"] and m["reason"] == "wind"


# ---------------------------------------------------------------- MARKs, records, job sheet

def test_tree_mark_makes_a_valid_mark_and_the_jobsheet_lists_it_hindi_first(hub):
    ok, why, _ = hub.handle(_cmd({"type": "tree_mark", "tree_id": "R1-2", "kind": "fruit", "clock": 3,
                                  "h_class": "upar"}), 1.0, _ctx(hub, 1.0))
    assert ok, why
    marks = _out(hub, "mark")
    assert len(marks) == 1 and rec.validate_mark(marks[0]) == []
    assert marks[0]["side"] == "E" and marks[0]["h_class"] == "upar" and marks[0]["physical_tag"] == "M-2"
    ev = hub.take_events()
    assert ev[0]["type"] == "mark" and cv_ok(ev[0])
    hub.ingest("mark", json.dumps(marks[0]), 1.1)                    # our own echo from /nandi/mark: de-duplicated
    assert len(hub.marks) == 1
    ok, _, replies = hub.handle(_cmd({"type": "jobsheet_request", "row_id": "R1"}), 2.0, _ctx(hub, 2.0))
    assert ok and len(replies) == 1
    js_msg = replies[0]
    assert cv_ok(js_msg) and js_msg["missing"] == []                  # O-23: every MARK appears
    assert js_msg["sheet"]["lang"] == ["hi", "en"] and js_msg["sheet"]["n_marks"] == 1
    assert js_msg["html"].index("कतार R1") < js_msg["html"].index("Row R1")      # Hindi first
    assert js_msg["voice"][0].startswith("कतार R1")
    ok, why, _ = hub.handle(_cmd({"type": "jobsheet_request", "row_id": "R9"}), 2.0, _ctx(hub, 2.0))
    assert not ok


def test_tree_record_summary_media_urls_and_media_path_guard(hub, tmp_path):
    record = {"schema": "nandi.tree/1", "run_id": "run7", "tree_id": "OT-1", "physical_tag": "OT-1", "crop": "mango",
              "visit_t": "2026-11-18T10:00Z", "presets": ["tree360"], "coverage": {"mode": "full", "masked_pct": 1.0},
              "symptoms": [{"class": "anthracnose", "band": "mid", "clock": 3, "stage": "lead", "count": 2}],
              "media": {"manifest_root": "ab", "keyframe_paths": ["low/000001.jpg", "../../etc/passwd"]},
              "next_actions": ["deep_scan"], "geo": {"x": 6.0, "y": -3.5}}
    hub.ingest("tree_record", json.dumps(record), 1.0)
    ev = hub.take_events()[0]
    assert cv_ok(ev) and "geo" not in ev["record"] and ev["record"]["leads"][0]["stage"] == "lead"
    media = ev["record"]["media"]
    k = hub.media_key
    assert media["keyframes"] == [f"/media/run7/tree360/OT-1/low/000001.jpg?k={k}"]   # traversal path dropped
    assert media["strips"][0]["url"] == f"/media/run7/tree360/OT-1/strip_low.jpg?k={k}"
    d = tmp_path / "run7" / "tree360" / "OT-1"
    (d / "low").mkdir(parents=True)
    (d / "strip_low.jpg").write_bytes(b"\xff\xd8jpeg")
    (d / "low" / "000001.jpg").write_bytes(b"\xff\xd8jpeg")
    (d / "manifest.json").write_text("{}")
    ok_path = cp.resolve_media_path(str(tmp_path), "/media/run7/tree360/OT-1/strip_low.jpg", hub.media_allowed)
    assert ok_path and ok_path.endswith("strip_low.jpg")
    assert cp.resolve_media_path(str(tmp_path), "/media/run7/tree360/OT-1/low/000001.jpg", hub.media_allowed)
    for bad in ("/media/run7/tree360/OT-1/manifest.json", "/media/run7/tree360/R1-1/strip_low.jpg",
                "/media/run8/tree360/OT-1/strip_low.jpg", "/media/run7/tree360/OT-1/../OT-1/strip_low.jpg",
                "/media/run7/tree360/OT-1/strip_low.jpg/..", "/media/../tree360/OT-1/strip_low.jpg"):
        assert cp.resolve_media_path(str(tmp_path), bad, hub.media_allowed) is None, bad


def test_world_state_person_positions_are_never_kept(hub):
    hub.ingest("world", json.dumps({"light": {"lux": 30000}, "actors": [{"id": "w1", "class": "person",
                                                                          "xy": [3.0, 4.0]}]}), 1.0)
    assert "actors" not in hub.data["world"] and hub.data["world"]["actors_n"] == 1
    assert hub.ready_context(1.0).light_lux == 30000


def test_presets_without_a_live_executor_are_reported(hub):
    ok, why, _ = _start(hub, "harvest_mango", 1.0, {"tree_ids": ["OT-1"]})
    assert ok and "not live" in why
    hub.tick(1.1, _ctx(hub, 1.1))
    assert _out(hub, "preset_run")[0]["plan"]["preset"] == "harvest_mango"   # WP27 cmd/mission for an executor
    assert hub.history[0]["status"] == "no_live_executor"


def test_demo_tour_executor_runs_a_matching_step_when_the_node_is_up(hub):
    hub.ingest("demo", json.dumps({"state": "idle"}), 0.9)
    _start(hub, "harvest_mango", 1.0, {"tree_ids": ["OT-1"]})
    hub.tick(1.1, _ctx(hub, 1.1))
    run = _out(hub, "demo_run")
    assert run == [{"steps": ["s05_harvest_mango"], "program": "orchard_tour", "transit": "teleport"}]
    hub.ingest("demo", json.dumps({"state": "running"}), 2.0)
    hub.tick(2.1, _ctx(hub, 2.1))
    hub.ingest("demo", json.dumps({"state": "done", "steps": [{"id": "s05_harvest_mango", "status": "done"}]}), 5.0)
    hub.tick(5.1, _ctx(hub, 5.1))
    assert hub.history[0]["status"] == "done"


# ---------------------------------------------------------------- WP10 review fixes: STOP, interlocks, freshness

def _tree360_walking(hub, t=1.5):
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    hub.tick(1.1, _ctx(hub, 1.1), False, True)
    hub.take_outbox()
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "approach", "stations": 8, "station": 0}), t)
    hub.ingest("drive_tree360", json.dumps({"vx": 0.6, "yaw": -0.2}), t)
    assert hub.drive(t + 0.05) == (0.6, 0.0, -0.2, True) and hub.relaying


def test_pilot_stop_holds_the_relay_and_pauses_tree360_until_resume(hub):
    _tree360_walking(hub)
    note = hub.stop_motion(1.6, "stop")
    assert note == "tree360 paused" and hub.drive_hold == "stop" and hub.active["paused"]
    assert _out(hub, "tree360_cmd") == [{"cmd": "pause"}]
    hub.ingest("drive_tree360", json.dumps({"vx": 0.6, "yaw": -0.2}), 1.65)   # the hook has not seen the pause yet
    assert hub.drive(1.7) == web_scene.RELEASE and not hub.relaying          # one release, then nothing
    for t in (1.8, 1.9, 2.0):
        hub.ingest("drive_tree360", json.dumps({"vx": 0.6, "yaw": -0.2}), t)
        assert hub.drive(t) is None
    st = hub.preset_status(2.0, _ctx(hub, 2.0))
    assert st["relay_hold"] == "stop" and cv_ok(st)
    ok, why, _ = hub.handle(_cmd({"type": "preset_resume"}), 2.1, _ctx(hub, 2.1))
    assert ok and hub.drive_hold is None and not hub.paused and _out(hub, "tree360_cmd") == [{"cmd": "resume"}]
    hub.ingest("drive_tree360", json.dumps({"vx": 0.6, "yaw": -0.2}), 2.2)
    assert hub.drive(2.2) == (0.6, 0.0, -0.2, True)


def test_any_stop_motion_call_seen_in_locomotion_status_holds_the_relay(hub):
    assert hub.observe_stop_seq(4, 0.5) is None                   # first sample is the baseline
    _tree360_walking(hub)
    assert hub.observe_stop_seq(4, 1.6) is None                   # unchanged
    assert hub.observe_stop_seq(5, 1.6) == "tree360 paused"       # Tk panel / API / another client pressed STOP
    hub.ingest("drive_tree360", json.dumps({"vx": 0.6, "yaw": -0.2}), 1.7)
    assert hub.drive(1.7) == web_scene.RELEASE and hub.drive(1.8) is None
    assert hub.observe_stop_seq(True, 1.9) is None and hub.observe_stop_seq("6", 1.9) is None


def test_recovery_relay_from_the_menu_can_be_paused_and_stays_stopped(hub):
    hub.ingest("recovery", json.dumps({"armed": True, "active": "return_home"}), 1.0)
    hub.ingest("drive_recovery", json.dumps({"vx": -0.3, "vy": 0.0, "yaw": 0.0, "release": False}), 1.0)
    assert hub.drive(1.0) == (-0.3, 0.0, 0.0, True) and hub.active is None
    ok, why, _ = hub.handle(_cmd({"type": "preset_pause"}), 1.1, _ctx(hub, 1.1))
    assert ok and "recovery motion stopped" in why and _out(hub, "recovery_cmd") == [{"abort": True}]
    hub.ingest("drive_recovery", json.dumps({"vx": -0.3, "vy": 0.0, "yaw": 0.0, "release": False}), 1.15)
    assert hub.drive(1.2) == web_scene.RELEASE and hub.drive(1.3) is None
    hub.ingest("recovery", json.dumps({"armed": True, "active": "return_home"}), 1.3)
    hub.tick(1.3, _ctx(hub, 1.3))
    assert hub.drive_hold == "pause"                             # the same behaviour still winding down: held
    # a recovery run from the menu is an explicit operator motion request
    ok, _, _ = hub.handle(_cmd({"type": "recovery", "op": "run_return_home"}), 1.4, _ctx(hub, 1.4))
    hub.ingest("drive_recovery", json.dumps({"vx": -0.3, "vy": 0.0, "yaw": 0.0, "release": False}), 1.45)
    assert ok and hub.drive_hold is None and hub.drive(1.5) == (-0.3, 0.0, 0.0, True)


def test_a_new_recovery_behaviour_after_the_stop_may_move_the_robot(hub):
    hub.ingest("recovery", json.dumps({"armed": True, "active": "return_home"}), 1.0)
    hub.ingest("drive_recovery", json.dumps({"vx": -0.3, "release": False}), 1.0)
    hub.drive(1.0)
    hub.stop_motion(1.1, "stop")
    hub.ingest("recovery", json.dumps({"armed": True, "active": None}), 1.5)
    hub.tick(1.5, _ctx(hub, 1.5))
    assert hub.drive_hold == "stop"
    hub.ingest("recovery", json.dumps({"armed": True, "active": "cloud_link"}), 2.0)   # WP28 safety behaviour
    hub.tick(2.0, _ctx(hub, 2.0))
    assert hub.drive_hold is None


def test_stop_with_nothing_running_is_a_no_op_for_the_hub(hub):
    assert hub.stop_motion(1.0, "stop") is None and hub.drive_hold is None and not hub.paused


def test_stop_pauses_the_queue_and_a_start_or_resume_ends_it(hub):
    hub.paused, hub.paused_reason = True, ""
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    assert hub.paused                                             # a plain queue pause stays...
    hub.paused, hub.paused_reason = True, "stop"
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1"]})
    assert not hub.paused                                         # ...an explicit start ends a STOP pause...
    hub.queue.items.clear()
    hub.paused, hub.paused_reason = True, "estop"
    _start(hub, "tree360", 1.0, {"tree_ids": ["R1-1"]})
    assert hub.paused and hub.paused_reason == "estop"            # ...never the E-STOP pause
    hub.paused, hub.paused_reason = False, ""
    assert hub.stop_motion(1.2, "stop") == "relay stopped" and hub.paused and hub.paused_reason == "stop"
    hub.tick(1.3, _ctx(hub, 1.3))
    assert hub.active is None                                     # nothing dispatched after the STOP
    ok, _, _ = hub.handle(_cmd({"type": "preset_resume"}), 1.4, _ctx(hub, 1.4))
    assert ok and not hub.paused and hub.drive_hold is None


def test_resume_is_refused_while_the_estop_is_latched_even_with_nothing_running(hub):
    hub.tick(1.0, _ctx(hub, 1.0, estop=True), estop=True)
    assert hub.paused and hub.paused_reason == "estop"
    ok, why, _ = hub.handle(_cmd({"type": "preset_resume"}), 1.1, _ctx(hub, 1.1, estop=True), estop=True)
    assert not ok and why.startswith("estop") and hub.paused
    ok, _, _ = hub.handle(_cmd({"type": "preset_resume"}), 1.2, _ctx(hub, 1.2))
    assert ok and not hub.paused


@pytest.mark.parametrize("msg,blocked", [
    ({"type": "manual_arm", "values": [0.2, 1.0, 1.0, 0.5, 0.5, 0.02, 0.02]}, True),
    ({"type": "gait_params", "values": [1.0, 1.0, 1.8, 1.0]}, True),
    ({"type": "mission_start", "zone": "zone_a"}, True),
    ({"type": "service", "name": "auto_harvest"}, True),
    ({"type": "cmd", "vx": 0.5, "vy": 0.0, "yaw": 0.0, "hold": True}, True),
    ({"type": "cmd", "vx": 0.0, "vy": 0.0, "yaw": 0.0, "hold": False}, False),
    ({"type": "service", "name": "stop"}, False),
    ({"type": "service", "name": "estop"}, False),
    ({"type": "service", "name": "release_arm"}, False),
    ({"type": "preset_pause"}, False),
])
def test_relay_interlock_blocks_arm_gait_and_second_motion_sources(msg, blocked):
    assert cp.blocked_by_relay(msg) is blocked


def test_sim_local_link_is_never_on_in_field_mode():
    assert cp.sim_local_link_on(None, False) and cp.sim_local_link_on("", False)
    assert cp.sim_local_link_on("1", False) and not cp.sim_local_link_on("0", False)
    for raw in (None, "", "1", "true", "on"):
        assert not cp.sim_local_link_on(raw, True), raw
    h = cp.ConsoleHub(_field(), sim_local_link=True, sim_tools=False)   # field mode (sim tools off)
    h.tick(1.0, h.ready_context(1.0))
    assert not h.sim_local_link and not any(k == "local_beat" for k, _ in h.take_outbox())


def test_tree360_drive_has_its_own_dead_man_and_never_crosses_trees(hub):
    _start(hub, "tree360", 1.0, {"tree_ids": ["OT-1", "R1-1"]})
    hub.tick(1.1, _ctx(hub, 1.1))
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "approach"}), 1.2)
    hub.ingest("drive_tree360", json.dumps({"vx": 0.7, "yaw": 0.3}), 1.3)
    assert hub.drive(1.4) == (0.7, 0.0, 0.3, True)
    for t in (2.0, 2.5, 3.0):                                     # progress stays fresh, the drive does not
        hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "arc"}), t)
    assert hub.drive(3.0) == web_scene.RELEASE and hub.drive(3.1) is None
    hub.ingest("tree360", json.dumps({"tree_id": "OT-1", "phase": "done"}), 5.0)
    hub.tick(5.1, _ctx(hub, 5.1))
    assert hub.active["tree"] == "R1-1"
    hub.ingest("drive_tree360", json.dumps({"vx": 0.7, "yaw": 0.3}), 5.05)   # OT-1's last drive, before R1-1 started
    hub.ingest("tree360", json.dumps({"tree_id": "R1-1", "phase": "approach"}), 5.3)
    assert hub.drive(5.4) is None                                 # not replayed for R1-1
    hub.ingest("drive_tree360", json.dumps({"vx": 0.5, "yaw": 0.0}), 5.5)
    assert hub.drive(5.5) == (0.5, 0.0, 0.0, True)


def test_media_needs_the_per_process_key(hub, tmp_path):
    d = tmp_path / "run7" / "tree360" / "OT-1"
    d.mkdir(parents=True)
    (d / "strip_low.jpg").write_bytes(b"\xff\xd8jpeg")
    hub.media_allowed.add(("run7", "OT-1"))
    url = "/media/run7/tree360/OT-1/strip_low.jpg"
    k = hub.media_key
    assert len(k) >= 20 and cp.ConsoleHub(_field()).media_key != k
    assert cp.resolve_media_path(str(tmp_path), f"{url}?k={k}", hub.media_allowed, k)
    for bad in (url, f"{url}?k=", f"{url}?k={k[:-1]}", f"{url}?token={k}", f"{url}?k=x&k={k}"):
        assert cp.resolve_media_path(str(tmp_path), bad, hub.media_allowed, k) is None, bad
    assert not cp.media_key_ok(f"{url}?k=", "")


@pytest.mark.parametrize("key,payload", [
    ("tree360", {"tree_id": "OT-1", "phase": "arc", "assist": True}),
    ("tree360", {"tree_id": "OT-1", "phase": "arc", "assist": "x"}),
    ("tree360", {"tree_id": "OT-1", "phase": "arc", "assist": [1]}),
    ("work", {"tag": "issued"}),
    ("stability", {"slope_deg": {"roll": "steep"}, "latched": False}),
])
def test_malformed_hub_inputs_never_stop_the_tick_or_the_beats(hub, key, payload):
    hub.ingest(key, json.dumps(payload), 1.0)
    ctx = hub.ready_context(1.0)
    hub.tick(1.0, ctx, False, True)
    out = hub.take_outbox()
    assert any(k == "local_beat" for k, _ in out) and any(k == "cloud_beat" for k, _ in out)
    assert cv_ok(hub.preset_status(1.0, ctx))


def test_unusable_status_fails_closed_with_sensor_fault(hub, monkeypatch):
    real = hub._ready_context

    def boom(*a, ignore=False):
        if not ignore:
            raise TypeError("bad payload")
        return real(*a, ignore=ignore)

    monkeypatch.setattr(hub, "_ready_context", boom)
    ctx = hub.ready_context(1.0)
    assert ctx.rtk == "NONE" and not ctx.stability_ok and hub.context_fault.startswith("sensor_fault")
    chips = {c["id"] for c in hub.preset_status(1.0, ctx)["chips"]}
    assert "sensor_fault" in chips


def test_tree360_hook_keeps_the_live_drive_fresh_and_drops_the_previous_runs_drive():
    from types import SimpleNamespace

    from atharv_nandi.sim.hooks import Bus
    from atharv_nandi.sim.hooks import tree360 as h3

    class FakeRun:
        def __init__(self):
            self.io = SimpleNamespace(gait=None, close=lambda: None)
            self.orbit_value, self.assist, self.phase, self.finished, self.result = {}, None, "approach", False, None

        def tick(self):
            pass

        def progress(self):
            return {"phase": self.phase}

    bus = Bus()
    h3.attach(bus, FakeRun())
    bus.values["tree360_drive"] = {"vx": 0.6, "yaw": 0.0, "t": 0.0}   # set once (a stand / drive is not repeated)
    sent = []
    for i in range(51):                                                # 1 s at 50 Hz
        h3._tick_inner(None, None, i * 0.02, bus)
        if any(topic == h3.DRIVE_TOPIC for topic, _ in bus.drain()):
            sent.append(i * 0.02)
    assert len(sent) >= 10 and max(b - a for a, b in zip(sent, sent[1:])) <= h3.DRIVE_REPUBLISH_S + 0.021
    h3.detach(bus)
    h3.attach(bus, FakeRun())                                          # next tree of a multi-tree item
    assert "tree360_drive" not in bus.values
    h3._tick_inner(None, None, 2.0, bus)
    assert not [p for topic, p in bus.drain() if topic == h3.DRIVE_TOPIC]
