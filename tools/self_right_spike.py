#!/usr/bin/env python3
"""WP28 SELF-RIGHT measurement spike (MISSION-PRESETS 8 row SELF-RIGHT; OPEN-QUESTIONS OQ-19 A). Report only, no gate.

The robot starts rolled 60 or 90 deg onto its right side (crown stowed, arm folded at PARK, WP2 tools on the rack),
settles 1.5 s with the legs at the stand pose, then a scripted leg push runs open loop:

    push    the down-side (right) legs swing their feet out under the body (hip toward -0.7) and extend (calf toward
            -0.5); the up-side (left) legs reach toward the ground (hip toward +0.7, calf folded) to catch the roll-back
    catch   every leg back to the stand pose (the WP0b home targets), so the six feet land
    hold    stand pose for the rest of the case

Measured per case: success (tilt < 15 deg at the end and the trunk above 0.20 m), time to tilt < 30 deg, max tilt,
peak |actuator_force| / forcerange per leg joint (hip, thigh, calf; the worst leg), and whether any joint saturated.
The person-clear + confirmation gate (``recovery_behaviours.self_right_gate``) is checked first and recorded; in the
sim both are given (``--no-confirm`` shows the refusal). Writes a JSON report (``--out``) and prints a table.

Run (capped runner, ~10-40 s wall):
    nandi-sim <worktree> python3 tools/self_right_spike.py --out runs/self_right_spike.json
    nandi-sim <worktree> python3 tools/self_right_spike.py --roll 60 --quick --out /tmp/x.json
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from atharv_nandi.core import recovery_behaviours as rb  # noqa: E402
from atharv_nandi.core.tripod_gait import JOINT_LIMITS, SIDE, home_leg_targets  # noqa: E402

LEGS = ("FR", "FL", "MR", "ML", "RR", "RL")
SETTLE_S = 1.5
SUCCESS_TILT_DEG = 15.0
UPRIGHT_TILT_DEG = 30.0


def tilt_deg(q) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q[4] ** 2 + q[5] ** 2)))))


def push_targets(push: dict, t_rel: float):
    """18 leg targets (hip, thigh, calf per leg) of the scripted push at ``t_rel`` s after the settle."""
    home = home_leg_targets()
    if t_rel < 0.0 or t_rel >= push["push_s"] + push["catch_s"]:
        return home
    out = list(home)
    k = min(1.0, t_rel / push["ramp_s"]) if t_rel < push["push_s"] else max(
        0.0, 1.0 - (t_rel - push["push_s"]) / push["catch_s"])
    for i, side in enumerate(SIDE):
        h, th, ca = home[3 * i:3 * i + 3]
        if side < 0:                                   # right legs: down side (roll > 0 puts +y up)
            tgt = (-push["hip_down"], th + push["thigh_down"], push["calf_down"])
        else:
            tgt = (push["hip_up"], th, push["calf_up"])
        for j, (a, b) in enumerate(zip((h, th, ca), tgt)):
            lo, hi = JOINT_LIMITS[j]
            out[3 * i + j] = max(lo, min(hi, a + k * (b - a)))
    return out


DEFAULT_PUSH = {"hip_down": 0.7, "thigh_down": 0.0, "calf_down": -0.5, "hip_up": 0.7, "calf_up": -2.2,
                "ramp_s": 0.6, "push_s": 1.2, "catch_s": 0.8}
VARIANTS = {
    "default": {},
    "thigh_fwd": {"thigh_down": -0.6},
    "fast": {"ramp_s": 0.25, "push_s": 0.8},
}


def run_case(m, roll_deg: float, push: dict, case_s: float, person_clear=True, confirmed=True) -> dict:
    from atharv_nandi.core.sim_driver import SimDriver
    from atharv_nandi.sim.world.modules.tools import seat_tools
    ok, why = rb.self_right_gate(person_clear, confirmed, crown_stowed=True, arm_folded=True)
    res = {"roll0_deg": roll_deg, "push": push, "gate": {"allowed": ok, "reason": why}}
    if not ok:
        res.update(success=False, time_s=None, max_tilt_deg=None, torque_vs_forcerange=None, refused=True)
        return res
    drv = SimDriver(m)
    d = drv.d
    r = math.radians(roll_deg)
    d.qpos[0], d.qpos[1] = 1.6, -3.6                     # flat ground east of home (orchard_demo)
    d.qpos[2] = 0.30 if roll_deg < 75 else 0.26
    d.qpos[3:7] = [math.cos(r / 2), math.sin(r / 2), 0.0, 0.0]
    seat_tools(m, d)
    mujoco.mj_forward(m, d)
    drv.gait.stop()
    t_settle = SETTLE_S
    drv.gait.step = lambda roll, pitch: push_targets(push, drv.t - t_settle)   # scripted legs, no gait
    leg_idx = list(range(7, 7 + 18))
    frc = np.abs(m.actuator_forcerange[leg_idx]).max(axis=1)
    frc[frc <= 0] = np.inf
    peak = np.zeros(18)
    sat = 0
    tilts = []
    t_up = None
    w0 = time.perf_counter()
    while drv.t < case_s:
        drv.step()
        f = np.abs(d.actuator_force[leg_idx])
        peak = np.maximum(peak, f / frc)
        sat += int(np.any(f >= 0.999 * frc))
        tl = tilt_deg(d.qpos)
        tilts.append(tl)
        if t_up is None and drv.t > t_settle and tl < UPRIGHT_TILT_DEG:
            t_up = drv.t - t_settle
    end_tilt = tilts[-1]
    pk = peak.reshape(6, 3)
    res.update(
        success=bool(end_tilt < SUCCESS_TILT_DEG and d.qpos[2] > 0.20),
        time_s=None if t_up is None else round(t_up, 3),
        settled_tilt_deg=round(float(np.mean(tilts[int(0.9 * t_settle / m.opt.timestep):int(t_settle / m.opt.timestep)]))
                               if len(tilts) > int(t_settle / m.opt.timestep) else end_tilt, 1),
        end_tilt_deg=round(end_tilt, 1), max_tilt_deg=round(max(tilts), 1), trunk_z_end_m=round(float(d.qpos[2]), 3),
        torque_vs_forcerange={"hip": round(float(pk[:, 0].max()), 3), "thigh": round(float(pk[:, 1].max()), 3),
                              "calf": round(float(pk[:, 2].max()), 3),
                              "worst_leg": LEGS[int(np.argmax(pk.max(axis=1)))]},
        saturated_steps=sat, sim_s=round(drv.t, 2), wall_s=round(time.perf_counter() - w0, 2))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roll", type=float, action="append", help="initial roll deg (repeat; default 60 and 90)")
    ap.add_argument("--variant", action="append", choices=sorted(VARIANTS), help="push variants (default all)")
    ap.add_argument("--quick", action="store_true", help="default push only, 4 s per case")
    ap.add_argument("--case-s", type=float, default=6.0)
    ap.add_argument("--no-confirm", action="store_true", help="show the gate refusal")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    os.environ.setdefault("NANDI_WORLD", "orchard_demo")
    from atharv_nandi.core import farm_scene
    m = farm_scene.load_farm_model()
    rolls = a.roll or [60.0, 90.0]
    variants = ["default"] if a.quick else (a.variant or sorted(VARIANTS))
    case_s = 4.0 if a.quick else a.case_s
    cases = []
    for roll in rolls:
        for v in variants:
            push = {**DEFAULT_PUSH, **VARIANTS[v]}
            c = run_case(m, roll, push, case_s, confirmed=not a.no_confirm)
            c["variant"] = v
            cases.append(c)
            tq = c.get("torque_vs_forcerange") or {}
            print(f"roll {roll:4.0f}  {v:10s} success {c['success']!s:5}  t_up {c.get('time_s')}  max tilt "
                  f"{c.get('max_tilt_deg')}  end tilt {c.get('end_tilt_deg')}  torque/forcerange hip {tq.get('hip')} "
                  f"thigh {tq.get('thigh')} calf {tq.get('calf')} ({tq.get('worst_leg')})  sat steps "
                  f"{c.get('saturated_steps')}  gate {c['gate']}", flush=True)
    rep = {"schema": "nandi.self_right_spike/1", "gate": "none (report only)", "world": os.environ["NANDI_WORLD"],
           "assumed": ["open_loop_scripted_push", "flat_ground", "tools_on_rack", "crown_stowed"],
           "mujoco": mujoco.__version__, "cases": cases}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
