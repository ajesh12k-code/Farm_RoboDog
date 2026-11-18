"""Review F3 probe: does the stability monitor block the robot's own tripod walk?

Closed-loop tripod walk (same loop as measure_baseline.walk) with the StabilityProbe on every 20 ms
control tick and MonitorCore sampled like the node (default 10 Hz). Prints one JSON line per case:
fraction of samples with < 3 feet in contact, the per-sample (raw) and debounced walk inhibit, and the
stance / swing tripod clearances that set stability.STANCE_NEAR_M.

Needs tools/physics/measure_baseline.py (physics-alignment package A). Run inside the sim image:
    nandi-sim <wt> python3 tools/physics/monitor_walk_probe.py '[{"payload": 10}, {"terrain": "rough", "seed": 5}]'
Case keys: terrain, seed, payload (kg), vx, yaw, secs, monitor_hz.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tools.physics import measure_baseline as mb  # noqa: E402
from atharv_nandi.core import stability as st  # noqa: E402
from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch  # noqa: E402
from atharv_nandi.core.stability_probe import TRIPODS, MonitorCore, StabilityProbe  # noqa: E402
from atharv_nandi.core.tripod_gait import TripodGaitGenerator  # noqa: E402

CTRL_DT = 0.02
SETTLE_TICKS = 50


def pct(xs, qs):
    return [round(1000 * float(np.percentile(xs, q)), 1) for q in qs] if xs else None


def run(terrain="flat", seed=7, payload=None, vx=1.0, yaw=0.0, secs=6.0, monitor_hz=10.0):
    m = mb.build(NANDI_TERRAIN=terrain, NANDI_SEED=seed, NANDI_CROPS=6, NANDI_PAYLOAD_KG=payload)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if terrain == "rough":
        d.qpos[0] += mb.ROUGH_START_X
    mujoco.mj_forward(m, d)
    probe = StabilityProbe(m)
    core = MonitorCore(m, payload_kg=payload or 0.0)
    gait = TripodGaitGenerator(CTRL_DT)
    sub = int(round(CTRL_DT / m.opt.timestep))
    every = max(1, int(round(1.0 / (monitor_hz * CTRL_DT))))
    n = lt3 = unsup = 0
    feet_hist = [0] * 7
    stance_clear, swing_clear = [], []
    mon_n = raw_walk = deb_walk = 0
    for k in range(int((secs + 1.0) / CTRL_DT)):
        if k == SETTLE_TICKS:
            gait.set_command(vx, 0.0, yaw, hold=True)
        roll, pitch = quat_to_roll_pitch(*d.qpos[3:7])
        d.ctrl[7:25] = gait.step(roll, pitch)
        for _ in range(sub):
            mujoco.mj_step(m, d)
        if k < SETTLE_TICKS:
            continue
        r = probe.probe(d.qpos)
        n += 1
        feet_hist[r["n_contacts"]] += 1
        lt3 += r["n_contacts"] < 3
        unsup += r["mode"] == st.MODE_UNSUPPORTED
        if r["stance_clearance_m"] is not None:
            stance_clear.append(r["stance_clearance_m"])
            swing_clear.append(max(max(probe.foot_clearance(f"{leg}_foot")[0] for leg in t) for t in TRIPODS))
        if k % every == 0:
            core.on_qpos(d.qpos)
            s = core.status(k * CTRL_DT)
            mon_n += 1
            raw_walk += st.INHIBIT_WALK in s["inhibit_raw"]
            deb_walk += st.INHIBIT_WALK in s["inhibit"]
    print(json.dumps({
        "terrain": terrain, "seed": seed, "payload": payload, "vx": vx, "yaw": yaw, "samples": n,
        "feet_in_contact_pct_0to6": [round(100 * x / n, 1) for x in feet_hist],
        "lt3_frac": round(lt3 / n, 3), "probe_unsupported_frac": round(unsup / n, 3),
        "monitor_hz": monitor_hz, "walk_inhibit_raw_frac": round(raw_walk / mon_n, 3),
        "walk_inhibit_debounced_frac": round(deb_walk / mon_n, 3),
        "stance_clear_mm_p50_p95_max": pct(stance_clear, (50, 95, 100)),
        "swing_clear_mm_p5_p50": pct(swing_clear, (5, 50)),
    }))


if __name__ == "__main__":
    for case in json.loads(sys.argv[1] if len(sys.argv) > 1 else '[{}, {"payload": 10}]'):
        run(**case)
