"""kis/v1 mapper (atharv_nandi/core/kis_mapper.py) and the ROS-free TelemetryBridge core."""
import json
import math
import re

import pytest

from atharv_nandi.core import kis_mapper as km

TOPIC_RE = re.compile(r"^kis/v1/[a-z0-9][a-z0-9-]{0,39}/[a-z0-9][a-z0-9-]{0,39}/[a-z0-9][a-z0-9-]{0,63}"
                      r"/(tele/[a-z0-9][a-z0-9_-]{0,31}|state|status)$")
HOME = [0.0, 0.0, 0.34, 1.0, 0.0, 0.0, 0.0]


def mapper(**kw):
    args = dict(tenant="sim-nandi", site="sim-field-01", device="nandi-sim-01", boot=17,
                origin={"lat": 0.0, "lon": 0.0, "alt": 0.0}, sw="atharv-nandi abc1234")
    args.update(kw)
    return km.KisMapper(**args)


def test_topic_format_for_every_stream_and_state_status():
    m = mapper()
    topics = [m.tele_topic(s) for s in km.STREAMS] + [m.state_topic(), m.status_topic()]
    assert topics[0] == "kis/v1/sim-nandi/sim-field-01/nandi-sim-01/tele/pose"
    for t in topics:
        assert TOPIC_RE.match(t), t


def test_tenant_must_be_sim():
    for bad in ("kisaanu", "acme-farms", "simx", "prod-sim-1"):
        with pytest.raises(km.TenantRefused):
            mapper(tenant=bad)
    assert mapper(tenant="sim-x").tenant == "sim-x"


@pytest.mark.parametrize("kw", [dict(tenant="sim_bad"), dict(tenant="Sim-a"), dict(site="a/b"),
                                dict(device="nandi.sim"), dict(device="")])
def test_bad_slugs_refused(kw):
    with pytest.raises(km.BadSlug):
        mapper(**kw)


def test_bad_stream_slug_refused():
    with pytest.raises(km.BadSlug):
        mapper().tele_topic("pose/x")


def test_envelope_fields_on_every_payload():
    m = mapper()
    payloads = [m.pose(1000, HOME)[1], m.batt(1000, {"soc": 0.9})[1],
                m.mission(1000, {"state": "idle"})[1], m.detect(1000, [])[1],
                m.harvest(1000, {"attempt": 1})[1], m.state(1000, "idle")[1]]
    for p in payloads:
        assert p["v"] == 1 and p["ts"] == 1000 and p["ts_src"] == "sim" and p["boot"] == 17
        assert p["data_provenance"] == "simulated" and isinstance(p["seq"], int)


def test_seq_monotonic_per_stream_and_independent():
    m = mapper()
    seqs = [m.pose(i, HOME)[1]["seq"] for i in range(5)]
    assert seqs == [0, 1, 2, 3, 4]
    assert m.mission(0, {})[1]["seq"] == 0  # its own counter
    assert m.pose(9, HOME)[1]["seq"] == 5


def test_seq_resets_on_new_boot():
    m = mapper()
    for i in range(3):
        m.pose(i, HOME)
    m.new_boot(18)
    p = m.pose(10, HOME)[1]
    assert p["seq"] == 0 and p["boot"] == 18


def test_tele_payload_16kib_limit_raises():
    m = mapper()
    small = [{"id": i, "cls": "tomato", "score": 0.9} for i in range(10)]
    _, p = m.detect(1, small)
    assert len(km.encode(p)) <= km.MAX_TELE_BYTES
    big = [{"id": i, "cls": "tomato" * 10, "score": 0.9} for i in range(400)]
    with pytest.raises(km.PayloadTooLarge):
        m.detect(1, big)


def test_payload_exactly_at_limit_passes_one_byte_over_fails():
    assert km.check_size("x" * km.MAX_TELE_BYTES, km.MAX_TELE_BYTES) == km.MAX_TELE_BYTES
    with pytest.raises(km.PayloadTooLarge):
        km.check_size("x" * (km.MAX_TELE_BYTES + 1), km.MAX_TELE_BYTES)


def test_state_limit_is_8kib():
    m = mapper()
    with pytest.raises(km.PayloadTooLarge):
        m.state(1, "idle", {"blob": "y" * 9000})


def test_pose_enu_and_latlon_from_origin():
    m = mapper(origin={"lat": 0.0, "lon": 0.0, "alt": 100.0})
    yaw = math.pi / 2
    q = [111.32, -55.66, 0.34, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    _, p = m.pose(5, q, sim_t_ms=1_794_000_001_250)
    assert p["enu"] == {"e": 111.32, "n": -55.66, "u": 0.34}
    assert p["lat"] == pytest.approx(math.degrees(-55.66 / km.EARTH_RADIUS_M), abs=1e-9)
    assert p["lon"] == pytest.approx(math.degrees(111.32 / km.EARTH_RADIUS_M), abs=1e-9)
    assert p["alt"] == pytest.approx(100.34)
    assert p["yaw_deg"] == pytest.approx(90.0)
    assert p["sim"] == {"t": 1_794_000_001_250}


def test_enu_to_latlon_scales_longitude_with_latitude():
    lat, lon = km.enu_to_latlon(1000.0, 0.0, 60.0, 10.0)
    assert lat == pytest.approx(60.0)
    assert lon - 10.0 == pytest.approx(2 * math.degrees(1000.0 / km.EARTH_RADIUS_M), rel=1e-6)


def test_status_is_plain_string_and_will_is_retained_offline():
    m = mapper()
    assert m.status(True) == ("kis/v1/sim-nandi/sim-field-01/nandi-sim-01/status", "online")
    assert m.status(False)[1] == "offline"
    w = m.will()
    assert w == {"topic": m.status_topic(), "payload": "offline", "qos": 1, "retain": True, "will_delay_s": 10}


def test_retain_only_state_and_status():
    m = mapper()
    assert km.retain_for(m.state_topic()) and km.retain_for(m.status_topic())
    assert not any(km.retain_for(m.tele_topic(s)) for s in km.STREAMS)


def test_every_robot_stream_is_flagged_proposed():
    assert set(km.STREAMS) == {"pose", "batt", "mission", "detect", "harvest"}
    assert all(status == "PROPOSED" for status, _ in km.STREAMS.values())
    assert km.STREAMS["harvest"][1] != "robot-stream-design"


def test_batt_passes_only_finite_numbers():
    _, p = mapper().batt(1, {"soc": 0.5, "voltage_v": float("nan"), "power_w": True, "energy_wh": 12.5, "x": 1})
    assert p["soc"] == 0.5 and p["energy_wh"] == 12.5
    assert "voltage_v" not in p and "power_w" not in p and "x" not in p


def test_mission_summary_from_status_dicts():
    h = {"mode": "ik_harvest", "phase": "carry", "target": "crop_item_1", "last_result": ""}
    g = {"in_basket": 2, "crops_total": 6, "falls": 1, "fallen": False}
    s = km.mission_summary(h, g, run_id="r1")
    assert s == {"state": "harvesting", "arm_mode": "ik_harvest", "phase": "carry", "target": "crop_item_1",
                 "last_result": "", "picked": 2, "crops_total": 6, "falls": 1, "fallen": False, "run_id": "r1"}
    s2 = km.mission_summary({"mode": "rl_balance"}, g, {"state": "paused", "mission_id": "m-1"})
    assert s2["state"] == "paused" and s2["mission_id"] == "m-1"


def test_jsonl_lines_carry_topic_and_payload():
    m = mapper()
    lines = km.lines_for([m.pose(1, HOME), m.status(True)])
    a, b = (json.loads(x) for x in lines)
    assert a["topic"].endswith("/tele/pose") and a["retain"] is False and a["payload"]["seq"] == 0
    assert b == {"topic": m.status_topic(), "retain": True, "payload": "online"}


# ---------------------------------------------------------------------- bridge core (no ROS)
class ListSink:
    def __init__(self):
        self.msgs = []
        self.count = 0

    def send(self, topic, payload, retain):
        self.msgs.append((topic, payload, retain))
        self.count += 1

    def close(self):
        pass


def test_bridge_pose_rate_and_status_lifecycle():
    import atharv_nandi.nodes.telemetry_bridge as tb
    sink = ListSink()
    clock = {"ms": 0}
    br = tb.TelemetryBridge(mapper(), [sink], {"rates_hz": {"pose": 2.0, "mission": 1.0}}, "run-1",
                            wall_ms=lambda: clock["ms"], log=lambda s: None)
    br.start()
    br.on_qpos(HOME)
    for i in range(21):  # 5 s at 4 Hz ticks (exact binary fractions)
        clock["ms"] = i * 250
        br.tick(i * 0.25)
    br.stop()
    topics = [t for t, _, _ in sink.msgs]
    assert topics[0].endswith("/status") and sink.msgs[0][1] == "online" and sink.msgs[0][2] is True
    assert topics[-1].endswith("/status") and sink.msgs[-1][1] == "offline"
    poses = [p for t, p, _ in sink.msgs if t.endswith("/tele/pose")]
    assert len(poses) == 11  # 2 Hz over 5 s, first at t=0
    assert [p["seq"] for p in poses] == list(range(11))
    assert all(b["ts"] > a["ts"] for a, b in zip(poses, poses[1:]))


def test_bridge_refuses_oversize_without_crashing():
    import atharv_nandi.nodes.telemetry_bridge as tb
    sink = ListSink()
    br = tb.TelemetryBridge(mapper(), [sink], {}, "r", log=lambda s: None)
    br.on_detections({"detections": [{"cls": "x" * 100} for _ in range(400)]})
    assert br.dropped == 1 and not sink.msgs
    br.on_detections({"detections": [{"cls": "tomato"}], "model": "crop@1.0.0"})
    assert sink.msgs[-1][0].endswith("/tele/detect") and sink.msgs[-1][1]["model"] == "crop@1.0.0"


def test_bridge_mission_on_change():
    import atharv_nandi.nodes.telemetry_bridge as tb
    sink = ListSink()
    br = tb.TelemetryBridge(mapper(), [sink], {}, "r", log=lambda s: None)
    h = {"mode": "ik_harvest", "phase": "approach", "target": "crop_item_1", "last_result": ""}
    br.on_harvest(h, 0.0)
    br.on_harvest(dict(h), 0.01)           # unchanged: no new mission message
    br.on_harvest(dict(h, phase="descend"), 0.02)
    missions = [p for t, p, _ in sink.msgs if t.endswith("/tele/mission")]
    assert [m["phase"] for m in missions] == ["approach", "descend"]
    assert missions[0]["state"] == "harvesting" and missions[0]["run_id"] == "r"


def test_mqtt_sink_is_off_by_default_and_paho_not_imported(tmp_path):
    import sys
    import atharv_nandi.nodes.telemetry_bridge as tb
    cfg = tb.load_config()
    assert cfg["sinks"]["mqtt"]["enabled"] is False
    assert km.SIM_TENANT_RE.match(cfg["identity"]["tenant"])
    sinks = tb.make_sinks(cfg, str(tmp_path), mapper())
    assert [type(s).__name__ for s in sinks] == ["FileSink"]
    assert "paho" not in sys.modules
    text = open(tb.DEFAULT_CONFIG, encoding="utf-8").read()
    assert "kisaanu.com" not in text and "mqtt://" not in text and "wss://" not in text


def test_mini_yaml_matches_pyyaml_on_the_shipped_config():
    import atharv_nandi.nodes.telemetry_bridge as tb
    text = open(tb.DEFAULT_CONFIG, encoding="utf-8").read()
    cfg = tb.mini_yaml(text)
    assert cfg["identity"]["tenant"] == "sim-nandi" and cfg["rates_hz"]["pose"] == 2.0
    assert cfg["sinks"]["mqtt"]["enabled"] is False and cfg["site_origin"]["lat"] == 0.0
    yaml = pytest.importorskip("yaml")
    assert cfg == yaml.safe_load(text)


# ---------------------------------------------------------------------- review fixes (seq, envelope, NaN, sim.t, state rate)
def test_refused_payload_does_not_consume_seq():
    m = mapper()
    with pytest.raises(km.PayloadTooLarge):
        m.detect(0, [{"x": "a" * 20000}])
    assert m.detect(0, [])[1]["seq"] == 0
    with pytest.raises(km.PayloadTooLarge):
        m.state(0, "idle", {"blob": "y" * 9000})
    assert m.state(0, "idle")[1]["seq"] == 0


def test_body_cannot_override_envelope_or_provenance():
    m = mapper()
    _, p = m.mission(0, {"seq": -5, "data_provenance": "measured", "ts_src": "ntp", "v": 9, "boot": 1, "ts": 7})
    assert (p["seq"], p["data_provenance"], p["ts_src"], p["v"], p["boot"], p["ts"]) == (0, "simulated", "sim", 1, 17, 0)
    _, s = m.state(0, "auto", {"data_provenance": "measured", "seq": 99, "mode": "x", "fw": "fake"})
    assert s["seq"] == 0 and s["data_provenance"] == "simulated" and s["mode"] == "auto" and s["fw"] == m.sw


def test_non_finite_pose_and_payloads_refused():
    m = mapper()
    with pytest.raises(km.NonFinite):
        m.pose(0, [float("nan")] * 7)
    with pytest.raises(km.NonFinite):
        m.mission(0, {"x": float("inf")})
    assert m.pose(0, HOME)[1]["seq"] == 0  # refusals left no seq gap
    with pytest.raises(ValueError):
        km.encode({"a": float("nan")})


def test_bridge_pose_keeps_sim_instant_in_epoch_ms_across_qpos_and_reset():
    import atharv_nandi.nodes.telemetry_bridge as tb
    sink = ListSink()
    clock = {"ms": 1_794_000_000_000}
    br = tb.TelemetryBridge(mapper(), [sink], {}, "r", wall_ms=lambda: clock["ms"], log=lambda s: None)
    br.on_game({"elapsed_s": 10.0}, 0.0)            # anchor = wall - 10 s
    for i in range(4):
        clock["ms"] += 500
        br.on_qpos(HOME)                            # must not wipe sim.t
        br.tick(i * 0.5)
    poses = [p for t, p, _ in sink.msgs if t.endswith("/tele/pose")]
    assert len(poses) == 4 and all("sim" in p for p in poses)
    assert poses[0]["sim"]["t"] == 1_794_000_000_000  # epoch ms: anchor + 10 s elapsed
    clock["ms"] += 500
    br.on_game({"elapsed_s": 0.2}, 2.0)             # reset_world: elapsed went backwards
    br.on_qpos(HOME)
    br.tick(2.0)
    last = [p for t, p, _ in sink.msgs if t.endswith("/tele/pose")][-1]
    assert last["sim"]["t"] == clock["ms"] and last["sim"]["t"] > poses[-1]["sim"]["t"]  # re-anchored, monotonic


def test_bridge_nan_qpos_dropped_not_written(tmp_path):
    import atharv_nandi.nodes.telemetry_bridge as tb
    fs = tb.FileSink(str(tmp_path / "kis_v1.jsonl"))
    br = tb.TelemetryBridge(mapper(), [fs], {}, "r", log=lambda s: None)
    br.on_qpos([float("nan")] * 7)
    br.tick(0.0)
    fs.close()
    assert br.dropped == 1
    assert "NaN" not in (tmp_path / "kis_v1.jsonl").read_text()


def test_bridge_state_ignores_drifting_safety_values_and_is_spaced():
    import atharv_nandi.nodes.telemetry_bridge as tb
    sink = ListSink()
    clock = {"ms": 0}
    br = tb.TelemetryBridge(mapper(), [sink], {}, "r", wall_ms=lambda: clock["ms"], log=lambda s: None)
    br.start()
    states = lambda: [p for t, p, _ in sink.msgs if t.endswith("/state")]  # noqa: E731
    n0 = len(states())
    for i in range(50):  # 5 s of a 10 Hz safety topic with a moving tilt and timestamp
        clock["ms"] += 100
        br.on_safety({"estop": False, "tilt_deg": 0.1 * i, "stamp": clock["ms"], "state": "ok"})
        br.tick(clock["ms"] / 1000.0)
    assert len(states()) == n0 + 1  # one send: the safety block appeared; the drift is not state
    assert states()[-1]["safety"] == {"estop": False, "state": "ok"}
    clock["ms"] += 100
    br.on_safety({"estop": True, "tilt_deg": 9.0, "stamp": clock["ms"], "state": "estop"})
    br.on_safety({"estop": False, "tilt_deg": 9.0, "stamp": clock["ms"], "state": "ok"})
    br.on_safety({"estop": True, "tilt_deg": 9.0, "stamp": clock["ms"], "state": "estop"})
    clock["ms"] += 1000
    br.tick(clock["ms"] / 1000.0)              # change inside the 1 s window goes out from tick
    assert states()[-1]["safety"]["estop"] is True
    assert len(states()) <= n0 + 3
