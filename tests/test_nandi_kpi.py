"""KPI maths of tools/nandi_kpi.py on a hand-built fixture run directory."""
import json
import os
import sys

import pytest

from tools import nandi_kpi  # noqa: E402

RUN = "20261117T090000Z-abc1234-7"
T0 = 1_794_000_000_000  # epoch ms


def attempt(n, success, total, phases, outcome=None, agrees=True, grasp=True):
    return {"attempt": n, "success": success, "duration_s": total, "phase_durations_s": phases,
            "outcome": outcome or ("picked" if success else "not_picked"), "agrees": agrees,
            "grasp_reported_ok": grasp}


def pose(seq, e, n, ts):
    return {"topic": "kis/v1/sim-nandi/sim-field-01/nandi-sim-01/tele/pose", "retain": False,
            "payload": {"v": 1, "ts": ts, "ts_src": "sim", "boot": 1, "seq": seq, "enu": {"e": e, "n": n, "u": 0.3}}}


def batt(seq, ts, wh=None, w=None):
    p = {"v": 1, "ts": ts, "ts_src": "sim", "boot": 1, "seq": seq}
    if wh is not None:
        p["energy_wh"] = wh
    if w is not None:
        p["power_w"] = w
    return {"topic": "kis/v1/sim-nandi/sim-field-01/nandi-sim-01/tele/batt", "retain": False, "payload": p}


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / RUN
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"run_id": RUN, "device_id": "nandi-sim-01",
                                             "start_utc": "2026-11-17T09:00:00.000+00:00",
                                             "start_utc_ms": T0, "sw": "atharv-nandi abc1234"}))
    attempts = [
        attempt(1, True, 10.0, {"approach": 2.0, "carry": 4.0}),
        attempt(2, False, 20.0, {"approach": 4.0, "carry": 6.0}, outcome="lost_after_grasp"),
        attempt(3, True, 30.0, {"approach": 6.0, "carry": 8.0}),
        attempt(4, False, 40.0, {"approach": 8.0}, outcome="false_grasp_report", agrees=False),
    ]
    (d / "harvest_attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts))
    events = [{"kind": "fall"}, {"kind": "intervention"}, {"kind": "intervention"}, {"kind": "intervention"},
              {"kind": "estop"}, {"kind": "world_reset"}]
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    # 30 min of poses: 5 m in 10 min, a 10 m jump in 0.5 s (a reset: ignored), then 1 m east.
    kis = [pose(0, 0.0, 0.0, T0), pose(1, 3.0, 4.0, T0 + 600_000), pose(2, 13.0, 4.0, T0 + 600_500),
           pose(3, 14.0, 4.0, T0 + 1_800_000), batt(0, T0, wh=100.0), batt(1, T0 + 1_800_000, wh=90.0)]
    (d / "kis_v1.jsonl").write_text("".join(json.dumps(k) + "\n" for k in kis) + '{"topic": "half')  # torn line
    return d


def kpis(d):
    return {r["metric"]: r for r in nandi_kpi.compute(str(d))["rows"]}


def test_counts_and_rates(run_dir):
    k = kpis(run_dir)
    assert k["run_hours"]["value"] == pytest.approx(0.5)
    assert k["attempts"]["value"] == 4 and k["picks"]["value"] == 2
    assert k["pick_success_rate"]["value"] == pytest.approx(0.5)
    assert k["picks_per_hour"]["value"] == pytest.approx(4.0)
    assert k["drops"]["value"] == 1 and k["false_grasp_reports"]["value"] == 1
    assert k["verdict_disagreements"]["value"] == 1
    assert k["falls"]["value"] == 1 and k["estops"]["value"] == 1 and k["interventions"]["value"] == 3
    assert k["interventions_per_hour"]["value"] == pytest.approx(6.0)


def test_distance_skips_reset_jumps(run_dir):
    assert kpis(run_dir)["distance_walked_m"]["value"] == pytest.approx(6.0)


def test_energy_per_pick_from_battery(run_dir):
    assert kpis(run_dir)["energy_per_pick_wh"]["value"] == pytest.approx(5.0)


def test_energy_from_power_integral():
    b = [batt(0, 0, w=100.0)["payload"], batt(1, 3_600_000, w=100.0)["payload"]]
    assert nandi_kpi.battery_energy_wh(b) == pytest.approx(100.0)
    assert nandi_kpi.battery_energy_wh([]) is None


def test_cycle_time_percentiles(run_dir):
    k = kpis(run_dir)
    assert k["cycle_total_p50_s"]["value"] == pytest.approx(25.0)
    assert k["cycle_total_p90_s"]["value"] == pytest.approx(37.0)
    assert k["cycle_approach_p50_s"]["value"] == pytest.approx(5.0)
    assert k["cycle_carry_p90_s"]["value"] == pytest.approx(7.6) and k["cycle_carry_p90_s"]["n"] == 3


def test_percentile_matches_numpy_linear():
    np = pytest.importorskip("numpy")
    vals = [3.0, 1.0, 7.0, 2.5, 9.0, 4.0]
    for q in (0, 10, 50, 90, 100):
        assert nandi_kpi.percentile(vals, q) == pytest.approx(float(np.percentile(vals, q)))
    assert nandi_kpi.percentile([], 50) is None and nandi_kpi.percentile([2.0], 90) == 2.0


def test_every_number_carries_identity(run_dir):
    for r in nandi_kpi.compute(str(run_dir))["rows"]:
        assert r["device_id"] == "nandi-sim-01" and r["run_id"] == RUN and r["date"] == "2026-11-17"
        assert r["data_provenance"] == "simulated"


def test_no_battery_and_no_attempts_gives_nulls(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"run_id": "r", "duration_s": 1800.0, "start_utc": "2026-11-17"}))
    k = kpis(d)
    assert k["attempts"]["value"] == 0 and k["pick_success_rate"]["value"] is None
    assert k["picks_per_hour"]["value"] == 0.0 and k["energy_per_pick_wh"]["value"] is None
    assert "no tele/batt" in k["energy_per_pick_wh"]["note"] and k["run_hours"]["note"] == "sim time"


def test_cli_prints_table_and_writes_kpi_and_manifest(run_dir, capsys):
    assert nandi_kpi.main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "picks_per_hour" in out and RUN in out and "SIMULATED" in out
    kpi = json.load(open(run_dir / "kpi.json"))
    assert kpi["run_id"] == RUN and len(kpi["rows"]) > 10
    man = json.load(open(run_dir / "manifest.json"))
    assert "kpi.json" in [c["path"] for c in man["chunks"]]


def test_distance_excludes_small_reset_teleport_from_world_reset_events():
    # 19 poses walking 0.9 m east at 2 Hz, then reset_world puts the trunk back at e=0 (0.9 m jump in 0.5 s
    # is under the 2 m/s speed filter); the recorder logs world_reset a little after the teleport.
    poses = [{"ts": T0 + i * 500, "seq": i, "enu": {"e": 0.05 * i, "n": 0.0}} for i in range(19)]
    poses.append({"ts": T0 + 19 * 500, "seq": 19, "enu": {"e": 0.0, "n": 0.0}})
    assert nandi_kpi.distance_walked(poses) == pytest.approx(1.8)          # without events: over-reports
    assert nandi_kpi.distance_walked(poses, [T0 + 19 * 500 + 400]) == pytest.approx(0.9)
    assert nandi_kpi.distance_walked(poses, [T0 + 60_000]) == pytest.approx(1.8)  # unrelated reset: no effect


def test_distance_reset_already_caught_by_speed_filter_not_double_counted(run_dir):
    # the fixture's 10 m jump is a reset dropped by speed; a world_reset event next to it must not drop more
    poses = nandi_kpi.kis_payloads(nandi_kpi.rl.read_jsonl(str(run_dir / "kis_v1.jsonl")), "pose")
    assert nandi_kpi.distance_walked(poses, [T0 + 600_600]) == pytest.approx(6.0)
