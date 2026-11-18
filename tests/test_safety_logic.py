"""ROS-free tests for atharv_nandi/core/safety_logic.py (latch, geofence, speed cap, stale, fall, stop metrics)."""
import json
import math

import pytest

from atharv_nandi.core.safety_logic import (
    DEFAULT_CONFIG, EstopLatch, LivenessTracker, SafetyCore, SpeedCap, StaleMonitor, StopMetrics,
    TiltFallDetector, build_safety_status, default_config_path, distance_to_edge, geofence_margin,
    heartbeat_json, load_safety_config, parse_heartbeat, parse_safety_status, parse_simple_yaml,
    point_in_polygon,
)

SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
L_SHAPE = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]


# --- latch ---------------------------------------------------------------------------------
def test_latch_engage_is_latched_and_first_reason_sticks():
    latch = EstopLatch()
    assert latch.engage("geofence", 1.0) is True
    assert latch.engage("operator", 2.0) is False
    assert latch.estopped and latch.reason == "geofence" and latch.engaged_t == 1.0


def test_latch_cannot_reset_while_cause_present():
    latch = EstopLatch()
    latch.engage("geofence", 0.0)
    assert latch.can_reset(cause_clear=False) is False
    ok, msg = latch.reset(cause_clear=False)
    assert ok is False and "geofence" in msg and latch.estopped


def test_latch_reset_when_cause_clear_rearms():
    latch = EstopLatch()
    latch.engage("operator", 0.0)
    assert latch.can_reset(True)
    ok, _ = latch.reset(True)
    assert ok and not latch.estopped and latch.reason is None
    assert latch.engage("fall", 3.0) is True  # can latch again after re-arming


def test_latch_reset_when_not_estopped_is_noop_success():
    ok, msg = EstopLatch().reset(cause_clear=False)
    assert ok and "Not" in msg


# --- geofence ------------------------------------------------------------------------------
@pytest.mark.parametrize("pt,inside", [((1, 1), True), ((3, 1), False), ((-0.01, 1), False), ((1, 2.5), False)])
def test_point_in_square(pt, inside):
    assert point_in_polygon(pt, SQUARE) is inside


def test_point_on_edge_and_vertex_counts_inside():
    assert point_in_polygon((2.0, 1.0), SQUARE)
    assert point_in_polygon((0.0, 0.0), SQUARE)
    assert point_in_polygon((1.0, 0.0), SQUARE)


def test_concave_polygon_notch_is_outside():
    assert point_in_polygon((0.5, 2.5), L_SHAPE)
    assert not point_in_polygon((2.0, 2.0), L_SHAPE)   # in the notch
    assert point_in_polygon((2.5, 0.5), L_SHAPE)


def test_distance_to_edge_and_signed_margin():
    assert distance_to_edge((1.0, 0.5), SQUARE) == pytest.approx(0.5)
    assert distance_to_edge((3.0, 3.0), SQUARE) == pytest.approx(math.sqrt(2))
    assert geofence_margin((1.0, 0.25), SQUARE) == pytest.approx(0.25)
    assert geofence_margin((2.5, 1.0), SQUARE) == pytest.approx(-0.5)


def test_default_geofence_contains_home_and_crops_but_not_far_field():
    poly = DEFAULT_CONFIG["geofence"]["polygon"]
    for p in [(0, 0), (1.0, 0.45), (3.2, -0.45), (6.5, 1.2)]:
        assert point_in_polygon(p, poly), p
    assert not point_in_polygon((8.0, 0.0), poly)
    assert not point_in_polygon((0.0, -2.5), poly)


# --- speed cap -----------------------------------------------------------------------------
def test_speed_cap_clamps_planar_magnitude_preserving_direction():
    cap = SpeedCap(max_speed_scale=1.0, max_cmd=0.5)
    vx, vy, yaw = cap.clamp_cmd(0.8, 0.6, 0.0)
    assert math.hypot(vx, vy) == pytest.approx(0.5)
    assert vy / vx == pytest.approx(0.75)
    assert cap.clamp_cmd(0.3, 0.0, -0.9) == (0.3, 0.0, -0.5)


def test_speed_cap_speed_scale():
    cap = SpeedCap(max_speed_scale=1.0)
    assert cap.clamp_speed_scale(1.5) == 1.0
    assert cap.clamp_speed_scale(0.7) == 0.7


# --- stale ---------------------------------------------------------------------------------
def test_stale_monitor_with_injected_clock():
    mon = StaleMonitor(max_age_s=0.5)
    mon.touch("/nandi/sim_qpos", 10.0)
    mon.touch("/nandi/leg_targets", 10.2)
    assert mon.stale(10.4) == []
    assert mon.stale(10.6) == ["/nandi/sim_qpos"]
    assert mon.stale(10.8) == ["/nandi/leg_targets", "/nandi/sim_qpos"]
    mon.touch("/nandi/sim_qpos", 10.8)
    assert mon.stale(10.9) == ["/nandi/leg_targets"]
    assert mon.age("/nandi/sim_qpos", 10.9) == pytest.approx(0.1)


def test_stale_monitor_never_seen_vs_expected():
    mon = StaleMonitor(0.5)
    assert mon.stale(100.0, ["/a"]) == []         # never seen, not expected: not stale
    mon.expect("/a", 100.0)
    assert mon.stale(100.4, ["/a"]) == []         # grace period
    assert mon.stale(100.6, ["/a"]) == ["/a"]


# --- fall ----------------------------------------------------------------------------------
def test_tilt_must_be_sustained():
    det = TiltFallDetector(tilt_deg=35.0, sustain_s=0.3)
    assert det.update(40.0, False, 0.0) is False
    assert det.update(40.0, False, 0.2) is False
    assert det.update(20.0, False, 0.25) is False  # dipped back: timer restarts
    assert det.update(40.0, False, 0.3) is False
    assert det.update(40.0, False, 0.61) is True


def test_fallen_flag_is_immediate():
    assert TiltFallDetector().update(0.0, True, 0.0) is True


# --- stop metrics --------------------------------------------------------------------------
def test_stop_metrics_on_synthetic_deceleration():
    """v(t) = 0.2 - 0.5 t (m/s) along x, sampled at 1 kHz: stops below 0.02 m/s at t = 0.36 s."""
    sm = StopMetrics()
    dt, x, t = 0.001, 0.0, 0.0
    sm.start(t, (x, 0.0))
    res = None
    while res is None and t < 2.0:
        v = max(0.0, 0.2 - 0.5 * t)
        x += v * dt
        t += dt
        res = sm.update(t, (x, 0.0), max(0.0, 0.2 - 0.5 * t))
    assert res is not None and res == sm.last
    assert res["t_s"] == pytest.approx(0.36, abs=0.002)
    # integral of v over [0, 0.36] = 0.2*0.36 - 0.25*0.36^2 = 0.0396
    assert res["dist_m"] == pytest.approx(0.0396, abs=5e-4)
    assert not sm.measuring
    assert sm.update(t + 1.0, (5.0, 0.0), 1.0) is None  # idle until the next start


def test_stop_metrics_cancel_keeps_previous_result():
    sm = StopMetrics()
    sm.start(0.0, (0, 0))
    sm.update(0.1, (0.01, 0), 0.0)
    first = sm.last
    sm.start(1.0, (0, 0))
    sm.cancel()
    assert sm.last == first and not sm.measuring


# --- supervisor core -----------------------------------------------------------------------
def _core(**over):
    cfg = load_safety_config("/nonexistent.yaml")
    cfg["geofence"]["polygon"] = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    for k, v in over.items():
        cfg[k].update(v)
    return SafetyCore(cfg)


def test_core_geofence_latches_and_reset_refused_until_inside():
    core = _core()
    assert core.update_pose(0.0, (0.5, 0.0), 0.0, False) is None
    assert core.update_pose(0.1, (1.05, 0.0), 0.0, False) == "geofence"
    assert core.latch.estopped and core.latch.reason == "geofence" and not core.geofence_ok
    ok, _ = core.reset()
    assert not ok and core.latch.estopped
    core.update_pose(0.15, (0.95, 0.0), 0.0, False)  # swayed back 5 cm inside: not enough (0.1 m)
    assert core.geofence_ok and not core.reset()[0]
    core.update_pose(0.2, (0.0, 0.0), 0.0, False)   # e.g. after /nandi/reset_world
    ok, _ = core.reset()
    assert ok and not core.latch.estopped
    assert core.events[0]["reason"] == "geofence"


def test_core_fall_latches_and_reset_refused_while_tilted():
    core = _core()
    core.update_pose(0.0, (0, 0), 40.0, False)
    assert core.update_pose(0.35, (0, 0), 40.0, False) == "fall"
    assert not core.reset()[0]
    core.update_pose(0.4, (0, 0), 3.0, False)
    assert core.reset()[0]


def test_core_operator_estop_resets_any_time():
    core = _core()
    assert core.engage_operator(1.0)
    assert core.latch.reason == "operator"
    assert core.reset()[0]


class _Wall:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_core_operator_reset_refused_while_outside_geofence():
    core = _core()
    core.engage_operator(0.0)
    assert core.update_pose(0.1, (8.0, 0.0), 0.0, False) is None  # already latched by the operator
    ok, msg = core.reset()
    assert not ok and core.latch.estopped and "geofence" in msg
    core.update_pose(0.2, (0.95, 0.0), 0.0, False)  # inside: enough after an operator latch
    assert core.reset()[0]


def test_core_operator_reset_refused_while_tilted_or_fallen():
    core = _core()
    core.engage_operator(0.0)
    core.update_pose(0.0, (0, 0), 40.0, False)
    core.update_pose(0.4, (0, 0), 40.0, True)
    ok, msg = core.reset()
    assert not ok and core.latch.estopped and "fall" in msg
    core.update_pose(0.5, (0, 0), 40.0, False)  # still tilted, no sustain needed to refuse
    assert not core.reset()[0]
    core.update_pose(0.6, (0, 0), 2.0, False)
    assert core.reset()[0] and not core.latch.estopped


def test_core_reset_does_not_restart_the_tilt_timer():
    core = _core(fall={"sustain_s": 0.3})
    core.update_pose(0.0, (0, 0), 40.0, False)   # tilted, not latched yet
    core.reset()                                  # a no-op reset must not clear the tilt timer
    assert core.update_pose(0.31, (0, 0), 40.0, False) == "fall"  # still latches on time


def test_core_rearm_stamps_and_persistence_round_trip(tmp_path):
    from atharv_nandi.core.safety_logic import load_latch_state, save_latch_state
    wall = _Wall(1000.0)
    core = SafetyCore(_core().cfg, wall_clock=wall)
    assert core.engage_operator(0.0) and core.engaged_at == 1000.0
    path = str(tmp_path / "latch.json")
    save_latch_state(path, core.state())
    # supervisor restart: a fresh core restores the latch, with its original stamp
    core2 = SafetyCore(_core().cfg, wall_clock=_Wall(2000.0))
    assert core2.restore(load_latch_state(path), 0.0)
    assert core2.latch.estopped and core2.latch.reason == "operator" and core2.engaged_at == 1000.0
    core2.wall_clock.t = 2001.0
    assert core2.reset()[0] and core2.rearmed_at == 2001.0
    save_latch_state(path, core2.state())
    core3 = SafetyCore(_core().cfg)
    assert core3.restore(load_latch_state(path), 0.0) is False
    assert not core3.latch.estopped and core3.rearmed_at == 2001.0
    assert load_latch_state(str(tmp_path / "missing.json")) is None
    assert SafetyCore(_core().cfg).restore(None, 0.0) is False


def test_core_supervisor_lost_latch_clears_only_on_reset():
    core = _core()
    assert core.engage_supervisor_lost(0.0)
    assert core.latch.reason == "supervisor_lost"
    core.update_pose(1.0, (0, 0), 0.0, False)
    assert core.latch.estopped            # never clears by itself
    assert core.reset()[0]


def test_hold_release_needs_a_rearm_after_the_hold_started():
    from atharv_nandi.core.safety_logic import hold_should_release
    assert not hold_should_release(True, False, 50.0, 10.0)          # still e-stopped
    assert not hold_should_release(False, True, 50.0, 10.0)          # supervisor silent
    assert not hold_should_release(False, False, None, 10.0)         # restarted supervisor, never reset
    assert not hold_should_release(False, False, 5.0, 10.0)          # reset predates this hold
    assert hold_should_release(False, False, 10.0, 10.0)
    assert hold_should_release(False, False, 11.0, 10.0)


def test_rearm_duration_and_smoothstep():
    from atharv_nandi.core.safety_logic import rearm_duration, smoothstep
    assert rearm_duration([0.0, 0.0], [1.0, 1.0], 0.25, 0.5, 3.0) == 0.5
    assert rearm_duration([0.35, 0.0], [1.0, 1.0], 0.25, 0.5, 3.0) == pytest.approx(1.4)
    assert rearm_duration([6.0], [1.0], 0.25, 0.5, 3.0) == 3.0
    assert rearm_duration([1.0], [0.0], 0.25, 0.5, 3.0) == 0.5   # zero-width range ignored
    assert smoothstep(-1) == 0.0 and smoothstep(0.5) == 0.5 and smoothstep(2) == 1.0


def test_core_geofence_disabled():
    core = _core(geofence={"enabled": False})
    assert core.update_pose(0.0, (50.0, 0.0), 0.0, False) is None and core.geofence_ok


# --- status, heartbeat, config ---------------------------------------------------------------
def test_safety_status_contract_fields_and_round_trip():
    latch = EstopLatch()
    latch.engage("operator", 0.0)
    live = LivenessTracker(max_age_s=2.0)
    live.beat("sim_runner", 0.0)
    live.beat("locomotion", 7.0, ok=True)
    st = build_safety_status(latch, 1.0, True, False, ["/nandi/sim_qpos"], {"t_s": 0.4, "dist_m": 0.05},
                             live.nodes(8.0), "safe_hold")
    for key in ("estopped", "latched_reason", "speed_cap", "geofence_ok", "fallen", "stale", "last_stop", "nodes"):
        assert key in st
    assert st["nodes"]["sim_runner"] == {"age_s": 8.0, "ok": False}
    assert st["nodes"]["locomotion"]["ok"] is True
    parsed = parse_safety_status(json.dumps(st))
    assert parsed == {"estopped": True, "latched_reason": "operator", "engaged_at": None, "rearmed_at": None}
    st2 = build_safety_status(latch, 1.0, True, False, [], None, {}, engaged_at=100.5, rearmed_at=90.25)
    p2 = parse_safety_status(json.dumps(st2))
    assert p2["engaged_at"] == 100.5 and p2["rearmed_at"] == 90.25
    assert parse_safety_status("not json") is None


def test_heartbeat_round_trip_and_rejects_garbage():
    hb = parse_heartbeat(heartbeat_json("locomotion", 1.23456, True, {"walking": True}))
    assert hb == {"node": "locomotion", "ok": True, "info": {"walking": True}}
    assert parse_heartbeat("{}") is None and parse_heartbeat("[1]") is None and parse_heartbeat("x") is None


def test_repo_config_loads_and_matches_simple_parser():
    cfg = load_safety_config(default_config_path())
    assert cfg["speed"]["max_speed_scale"] <= 1.0
    assert len(cfg["geofence"]["polygon"]) >= 3
    with open(default_config_path(), encoding="utf-8") as fh:
        simple = parse_simple_yaml(fh.read())
    assert simple["geofence"]["polygon"] == cfg["geofence"]["polygon"]
    assert simple["fall"] == {"tilt_deg": 35.0, "sustain_s": 0.3}
    assert simple["geofence"]["enabled"] is True
    assert simple["estop"]["hold_damping_scale"] == cfg["estop"]["hold_damping_scale"] >= 1.0
    assert simple["stale"]["topics"] == ["/nandi/sim_qpos", "/nandi/leg_targets"]


def test_config_env_override(tmp_path, monkeypatch):
    p = tmp_path / "s.yaml"
    p.write_text("speed:\n  max_speed_scale: 0.5\n")
    monkeypatch.setenv("NANDI_SAFETY_CONFIG", str(p))
    cfg = load_safety_config()
    assert cfg["speed"]["max_speed_scale"] == 0.5
    assert cfg["speed"]["max_cmd"] == 1.0          # default kept
    assert cfg["fall"]["tilt_deg"] == 35.0
