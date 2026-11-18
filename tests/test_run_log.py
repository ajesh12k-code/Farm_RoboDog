"""Run evidence logic (atharv_nandi/core/run_log.py) and the ROS-free Recorder in atharv_nandi/nodes/run_recorder.py."""
import datetime as dt
import json
import os

import pytest

from atharv_nandi.core import run_log as rl

H_IDLE = {"mode": "rl_balance", "phase": "idle", "target": None, "last_result": ""}


def H(phase, target="crop_item_1"):
    return {"mode": "ik_harvest", "phase": phase, "target": target, "last_result": ""}


def G(n, names=(), elapsed=0.0, falls=0):
    return {"crops_total": 6, "in_basket": n, "in_basket_names": list(names), "falls": falls,
            "fallen": False, "elapsed_s": elapsed, "rtf": 1.0}


def test_run_id_format():
    t = dt.datetime(2026, 11, 17, 9, 5, 3, tzinfo=dt.timezone.utc)
    assert rl.utc_stamp(t) == "20261117T090503Z"
    assert rl.make_run_id(rl.utc_stamp(t), "ABC1234", 7) == "20261117T090503Z-abc1234-7"
    assert rl.make_run_id("20261117T090503Z", "a/b..c", "-3") == "20261117T090503Z-abc-3"


def test_sw_version_string():
    assert rl.sw_version("abcdef0123", False) == "atharv-nandi abcdef0"
    assert rl.sw_version("abcdef0123", True) == "atharv-nandi abcdef0-dirty"
    assert rl.sw_version(None, None) == "atharv-nandi unknown"


def test_git_sha_reads_loose_and_packed_refs(tmp_path, monkeypatch):
    monkeypatch.delenv("NANDI_GIT_SHA", raising=False)
    g = tmp_path / ".git"
    (g / "refs" / "heads").mkdir(parents=True)
    (g / "HEAD").write_text("ref: refs/heads/feat/x\n")
    (g / "packed-refs").write_text("# pack-refs\n" + "1" * 40 + " refs/heads/feat/x\n")
    assert rl.git_sha(str(tmp_path)) == "1" * 40
    (g / "refs" / "heads" / "feat").mkdir()
    (g / "refs" / "heads" / "feat" / "x").write_text("2" * 40 + "\n")
    assert rl.git_sha(str(tmp_path)) == "2" * 40
    (g / "HEAD").write_text("3" * 40 + "\n")  # detached
    assert rl.git_sha(str(tmp_path)) == "3" * 40
    monkeypatch.setenv("NANDI_GIT_SHA", "f" * 40)
    assert rl.git_sha(str(tmp_path)) == "f" * 40


def test_build_meta_fields_and_config_hash(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("a: 1\n")
    meta = rl.build_meta("r1", str(tmp_path), [str(cfg)], env={"NANDI_TERRAIN": "flat", "NANDI_SEED": "9",
                                                               "NANDI_GIT_SHA": "x"})
    for k in ("run_id", "device_id", "git_sha", "dirty", "config_sha256", "NANDI_TERRAIN", "NANDI_SEED",
              "start_utc", "sw"):
        assert k in meta
    assert meta["device_id"] == "nandi-sim-01" and meta["NANDI_TERRAIN"] == "flat" and meta["NANDI_SEED"] == "9"
    assert meta["config_sha256"]["c.yaml"] == rl.sha256_file(str(cfg))


def run_attempt(tr, t0, end_result, phases=("approach", "descend", "grasp", "lift", "verify_grasp", "carry",
                                            "release", "verify_drop", "return_rest")):
    for i, ph in enumerate(phases):
        tr.harvest_status(t0 + i, H(ph))
    tr.harvest_status(t0 + len(phases), dict(H_IDLE, last_result=end_result))
    return t0 + len(phases)


def test_attempt_success_from_basket_truth_and_phase_timings():
    tr = rl.AttemptTracker(sw="s", run_id="r")
    tr.crop_positions({"crop_item_1": [0.62, 0.0, 0.035]})
    tr.game_status(0.0, G(0))
    t_end = run_attempt(tr, 1.0, "Harvested crop_item_1 into basket")
    assert tr.poll(t_end) == []                      # waiting SETTLE_S for basket truth
    tr.game_status(t_end + 2.0, G(1, ["crop_item_1"], elapsed=12.0))
    (a,) = tr.poll(t_end + 2.0)
    assert a["success"] is True and a["reported_success"] is True and a["agrees"] is True
    assert a["crop_id"] == "crop_item_1" and a["crop_xyz"] == [0.62, 0.0, 0.035]
    assert a["duration_s"] == 9.0 and a["phase_durations_s"]["carry"] == 1.0
    assert sum(a["phase_durations_s"].values()) == pytest.approx(9.0)
    assert a["basket_delta"] == 1 and a["outcome"] == "picked"


def test_node_says_harvested_but_basket_disagrees():
    tr = rl.AttemptTracker()
    tr.game_status(0.0, G(0))
    t = run_attempt(tr, 1.0, "Harvested crop_item_1 into basket")
    tr.game_status(t + 2.0, G(0))
    (a,) = tr.poll(t + 2.0)
    assert a["success"] is False and a["reported_success"] is True and a["agrees"] is False


def test_basket_delta_from_another_crop_is_not_success():
    tr = rl.AttemptTracker()
    tr.game_status(0.0, G(0))
    t = run_attempt(tr, 1.0, "Drop missed: crop_item_1 not in basket")
    tr.game_status(t + 2.0, G(1, ["crop_item_2"]))  # something else fell in
    (a,) = tr.poll(t + 2.0)
    assert a["success"] is False and a["agrees"] is True


def test_lost_after_grasp_vs_false_grasp_report():
    # Lifted 0.4 m, then fell back during carry: the live "Drop missed" pattern.
    tr = rl.AttemptTracker()
    tr.game_status(0.0, G(0))
    tr.crop_positions({"crop_item_1": [0.62, 0.0, 0.035]})
    for i, ph in enumerate(("approach", "descend", "grasp", "lift", "verify_grasp", "carry")):
        tr.harvest_status(1.0 + i, H(ph))
        z = 0.44 if ph in ("verify_grasp",) else 0.035
        tr.crop_positions({"crop_item_1": [0.6, 0.0, z]})
    tr.harvest_status(8.0, dict(H_IDLE, last_result="Drop missed: crop_item_1 not in basket"))
    tr.game_status(10.0, G(0))
    (a,) = tr.poll(10.0)
    assert a["outcome"] == "lost_after_grasp" and a["lost_in_phase"] == "carry"
    assert a["crop_max_rise_m"] == pytest.approx(0.405)
    # Never lifted, but the node still carried: a false grasp report.
    tr2 = rl.AttemptTracker()
    tr2.game_status(0.0, G(0))
    tr2.crop_positions({"crop_item_1": [0.62, 0.0, 0.035]})
    run_attempt(tr2, 1.0, "Drop missed: crop_item_1 not in basket")
    tr2.game_status(20.0, G(0))
    (b,) = tr2.poll(20.0)
    assert b["outcome"] == "false_grasp_report" and b["grasp_reported_ok"] is True


def test_world_reset_finalises_pending_with_pre_reset_truth():
    tr = rl.AttemptTracker()
    tr.game_status(0.0, G(0, elapsed=5.0))
    t = run_attempt(tr, 6.0, "Harvested crop_item_1 into basket")
    tr.game_status(t + 0.1, G(1, ["crop_item_1"], elapsed=15.5))
    tr.game_status(t + 0.5, G(0, elapsed=0.0))  # reset: count back to 0
    (a,) = tr.poll(t + 0.5)
    assert a["success"] is True and a["truth_source"] == "world_reset"


def test_flush_ends_running_attempt():
    tr = rl.AttemptTracker()
    tr.game_status(0.0, G(0))
    tr.harvest_status(1.0, H("approach"))
    (a,) = tr.flush(3.0)
    assert a["result"] == "Run ended during harvest" and a["success"] is False and a["truth_source"] == "run_end"


def test_two_attempts_counted_separately():
    tr = rl.AttemptTracker()
    tr.game_status(0.0, G(0))
    t = run_attempt(tr, 1.0, "Grasp failed on crop_item_1 (2 attempts)", phases=("approach", "regrasp"))
    t = run_attempt(tr, t + 1, "Harvested crop_item_1 into basket", phases=("approach",))
    tr.game_status(t + 3, G(1, ["crop_item_1"]))
    out = tr.poll(t + 3)
    assert [a["attempt"] for a in out] == [1, 2] and [a["success"] for a in out] == [False, True]


def test_interventions_only_while_mission_runs_and_debounced():
    det = rl.InterventionDetector(gap_s=1.0)
    assert det.manual(0.0, "cmd_vel") is False  # no mission
    det.mission_active = True
    assert det.manual(1.0, "cmd_vel") is True
    assert det.manual(1.2, "cmd_vel") is False   # same held command
    assert det.manual(1.3, "manual_arm") is True
    assert det.manual(3.0, "cmd_vel") is True
    assert det.manual(5.0, "cmd_vel", nonzero=False) is False
    assert det.count == 3


def test_edge_counter():
    e = rl.EdgeCounter()
    assert [e.update(v) for v in (False, True, True, False, True)] == [False, True, False, False, True]
    assert e.count == 2


def test_effort_stats_peak_rms_and_pct_of_limit():
    s = rl.EffortStats()
    s.add(["a", "b"], [3.0, -1.0], [0.5, 2.0])
    s.add(["a", "b"], [-4.0, 1.0], [-1.5, 0.0])
    out = s.summary({"a": 8.0}, source="test")
    a, b = out["actuators"]["a"], out["actuators"]["b"]
    assert a["peak_abs_effort"] == 4.0 and a["rms_abs_effort"] == pytest.approx(3.5355, abs=1e-4)
    assert a["peak_pct_of_limit"] == 50.0 and a["rms_pct_of_limit"] == pytest.approx(44.19, abs=0.01)
    assert a["peak_abs_velocity"] == 1.5 and b["force_limit"] is None and b["peak_pct_of_limit"] is None
    assert out["max_peak_pct_of_limit"] == 50.0 and out["samples"] == 2


def test_mjcf_force_limits_from_live_model():
    from atharv_nandi.core.farm_scene import load_farm_model
    lim = rl.mjcf_force_limits(load_farm_model())
    assert lim["gantry_slide"] == 80.0 and lim["finger_left"] == 25.0
    assert lim["FR_hip"] == 100.0 and lim["FR_hip_joint"] == 100.0  # leg_motor class, by joint name too


def test_recorder_writes_run_directory(tmp_path):
    import atharv_nandi.nodes.run_recorder as rr
    meta = {"run_id": "r1", "device_id": "nandi-sim-01", "sw": "atharv-nandi x", "start_utc": "2026-11-17T00:00:00"}
    rec = rr.Recorder(str(tmp_path), meta, bridge=None, force_limits={"FR_hip_joint": 85.0}, summary_period_s=1.0)
    hk = rec.hooks()
    q = [0.0] * 32 + [0.62, 0.0, 0.035, 1, 0, 0, 0]
    hk["qpos"](0.0, q)
    hk["game"](0.0, G(0))
    hk["harvest"](1.0, H("approach"))
    hk["cmd_vel"](1.5, True)                           # operator drives during the harvest
    hk["joint_states"](1.6, ["FR_hip_joint"], [42.5], [1.0])
    hk["harvest"](2.0, H("carry"))
    hk["harvest"](3.0, dict(H_IDLE, last_result="Harvested crop_item_1 into basket"))
    hk["game"](5.0, G(1, ["crop_item_1"], elapsed=5.0, falls=1))
    hk["tick"](5.0)
    rec.close(6.0)
    attempts = rl.read_jsonl(str(tmp_path / "harvest_attempts.jsonl"))
    assert len(attempts) == 1 and attempts[0]["success"] is True
    kinds = [e["kind"] for e in rl.read_jsonl(str(tmp_path / "events.jsonl"))]
    assert kinds == ["intervention", "fall", "run_end"]
    eff = json.load(open(tmp_path / "joint_effort_summary.json"))
    assert eff["actuators"]["FR_hip_joint"]["peak_pct_of_limit"] == 50.0
    man = json.load(open(tmp_path / "manifest.json"))
    assert man["final"] is True and {c["path"] for c in man["chunks"]} >= {"harvest_attempts.jsonl", "events.jsonl"}
    assert rec.live_status(6.0)["picks"] == 1


def test_recorder_without_joint_states_says_so(tmp_path):
    import atharv_nandi.nodes.run_recorder as rr
    rec = rr.Recorder(str(tmp_path), {"run_id": "r2"}, bridge=None)
    rec.write_summaries()
    eff = json.load(open(tmp_path / "joint_effort_summary.json"))
    assert eff["samples"] == 0 and "joint_states" in eff["note"]
    assert os.path.isfile(tmp_path / "manifest.json")


def test_estop_detection_variants():
    import atharv_nandi.nodes.run_recorder as rr
    assert rr.estop_engaged({"estop": True}) and rr.estop_engaged({"state": "ESTOP"})
    assert not rr.estop_engaged({"estop": False, "state": "estop"}) and not rr.estop_engaged({})
