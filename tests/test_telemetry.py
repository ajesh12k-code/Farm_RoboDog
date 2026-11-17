import json
import math

import pytest

from controllers.telemetry import (
    BAD_RED, OK_GREEN, WARN_AMBER, format_clock, format_game, format_harvest, format_mode,
    format_telemetry, parse_game_status, parse_harvest_status, parse_locomotion_status,
    parse_qpos_telemetry, shorten,
)

BAD_PAYLOADS = [None, "", "not json", "[1, 2]", "42", '"str"', b"{}", "{"]


# ---- qpos telemetry ----------------------------------------------------------------------------
def test_qpos_level_and_short():
    assert parse_qpos_telemetry([]) == (0.0, 0.0, 0.0)
    x, z, tilt = parse_qpos_telemetry([0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0] + [0.0] * 67)
    assert (x, z) == (0.5, 0.3)
    assert tilt == pytest.approx(0.0)


def test_qpos_tilt_90deg_roll():
    s = math.sin(math.pi / 4)
    _, _, tilt = parse_qpos_telemetry([0, 0, 0.3, s, s, 0.0, 0.0])
    assert tilt == pytest.approx(90.0)


def test_format_telemetry_colors():
    assert format_telemetry((0.0, 0.3, 5.0))[1] == "#38bdf8"
    assert format_telemetry((0.0, 0.3, 20.0))[1] == "#fbbf24"
    assert format_telemetry((0.0, 0.3, 30.0))[1] == "#f87171"


# ---- locomotion_status -------------------------------------------------------------------------
@pytest.mark.parametrize("payload", BAD_PAYLOADS)
def test_loco_bad_payload_defaults(payload):
    s = parse_locomotion_status(payload)
    assert s["ok"] is False
    assert s["mode"] == "tripod" and s["policy_loaded"] is False and s["walking"] is False
    assert s["gait_params"] == [1.0, 1.0, 1.8, 1.0]


def test_loco_full_payload():
    s = parse_locomotion_status(json.dumps({"mode": "rl", "policy_loaded": True, "policy_path": "/p.zip",
                                            "gait_params": [1.2, 0.8, 2.0, 1.5], "walking": True}))
    assert s == {"mode": "rl", "policy_loaded": True, "policy_path": "/p.zip",
                 "gait_params": [1.2, 0.8, 2.0, 1.5], "walking": True, "ok": True}


def test_loco_bad_fields_fall_back():
    s = parse_locomotion_status(json.dumps({"mode": "jetpack", "policy_loaded": "yes", "policy_path": 3,
                                            "gait_params": [1, "x", None, 2], "walking": 1}))
    assert s["ok"] is True
    assert s["mode"] == "tripod" and s["policy_loaded"] is False and s["policy_path"] is None
    assert s["gait_params"] == [1.0, 1.0, 1.8, 2.0]
    assert s["walking"] is False
    assert parse_locomotion_status('{"gait_params": [1, 2]}')["gait_params"] == [1.0, 1.0, 1.8, 1.0]


def test_loco_defaults_not_shared_between_calls():
    a = parse_locomotion_status(None)
    a["gait_params"][0] = 99.0
    assert parse_locomotion_status(None)["gait_params"][0] == 1.0


def test_format_mode_truthful():
    assert format_mode(parse_locomotion_status(None)).toggle_enabled is False
    no_policy = format_mode(parse_locomotion_status('{"mode": "tripod", "policy_loaded": false}'))
    assert no_policy.label == "Tripod Gait (no RL policy)"
    assert no_policy.toggle_enabled is False and no_policy.color == WARN_AMBER
    # Even a (buggy) mode=rl without a policy is shown as tripod: there is nothing else it can run
    assert format_mode(parse_locomotion_status('{"mode": "rl", "policy_loaded": false}')).label.startswith("Tripod")
    rl = format_mode(parse_locomotion_status('{"mode": "rl", "policy_loaded": true, "walking": true}'))
    assert rl.label == "Mode: RL Balance Policy - walking" and rl.toggle_enabled and rl.color == OK_GREEN
    tri = format_mode(parse_locomotion_status('{"mode": "tripod", "policy_loaded": true}'))
    assert tri.label == "Mode: Tripod Gait" and tri.toggle_enabled


# ---- harvest_status ----------------------------------------------------------------------------
@pytest.mark.parametrize("payload", BAD_PAYLOADS)
def test_harvest_bad_payload_defaults(payload):
    h = parse_harvest_status(payload)
    assert h == {"mode": "rl_balance", "phase": "idle", "target": None, "last_result": "", "ok": False}
    assert format_harvest(h)[0].startswith("Arm: waiting")


def test_harvest_payload_and_format():
    h = parse_harvest_status(json.dumps({"mode": "ik_harvest", "phase": "lift", "target": "crop_item_3",
                                         "last_result": "picked crop_item_1"}))
    assert h["ok"] and h["mode"] == "ik_harvest" and h["target"] == "crop_item_3"
    line, last, color = format_harvest(h)
    assert line == "Arm: IK harvest  |  lift  |  crop_item_3"
    assert last == "Last: picked crop_item_1"
    assert color == WARN_AMBER


def test_harvest_bad_fields():
    h = parse_harvest_status('{"mode": 5, "phase": null, "target": 7, "last_result": ["x"]}')
    assert h["ok"] is True
    assert (h["mode"], h["phase"], h["target"], h["last_result"]) == ("rl_balance", "idle", None, "")
    line, last, _ = format_harvest(h)
    assert line == "Arm: RL balance  |  idle" and last == ""


# ---- game_status -------------------------------------------------------------------------------
@pytest.mark.parametrize("payload", BAD_PAYLOADS)
def test_game_bad_payload_defaults(payload):
    g = parse_game_status(payload)
    assert g["ok"] is False and g["crops_total"] == 0 and g["in_basket_names"] == []
    assert format_game(g)[0].startswith("Score: waiting")


def test_game_payload_and_format():
    g = parse_game_status(json.dumps({"crops_total": 6, "in_basket": 2,
                                      "in_basket_names": ["crop_item_1", "crop_item_4"],
                                      "falls": 1, "fallen": False, "elapsed_s": 83.7, "rtf": 0.9764}))
    assert g["in_basket_names"] == ["crop_item_1", "crop_item_4"]
    text, color = format_game(g)
    assert text == "Crops 2/6 | Falls 1 | 01:23 | RTF 0.98"
    assert color not in (BAD_RED, OK_GREEN)


def test_game_fallen_and_complete_colors():
    fallen = parse_game_status('{"crops_total": 6, "in_basket": 0, "falls": 1, "fallen": true}')
    text, color = format_game(fallen)
    assert text.endswith("FALLEN") and color == BAD_RED
    done = parse_game_status('{"crops_total": 3, "in_basket": 3}')
    assert format_game(done)[1] == OK_GREEN


def test_game_bad_fields():
    g = parse_game_status('{"crops_total": "6", "in_basket": -2, "in_basket_names": ["a", 3], '
                          '"falls": true, "fallen": "no", "elapsed_s": NaN, "rtf": Infinity}')
    assert g["ok"] is True
    assert (g["crops_total"], g["in_basket"], g["falls"], g["fallen"]) == (0, 0, 0, False)
    assert g["in_basket_names"] == ["a"]
    assert g["elapsed_s"] == 0.0 and g["rtf"] == 0.0


def test_format_clock_and_shorten():
    assert format_clock(0) == "00:00"
    assert format_clock(59.9) == "00:59"
    assert format_clock(3725) == "62:05"
    assert format_clock(-4) == "00:00"
    assert shorten("abc", 5) == "abc"
    assert shorten("abcdefgh", 6) == "abc..."
    assert shorten(None) == ""
