#!/usr/bin/env python3
"""Closed-loop mission check in bare MuJoCo (no ROS): the real MissionController, TripodGaitGenerator,
HarvestSequence and energy model drive the live farm world exactly as the ROS nodes would.

    python3 tools/mission_sim_check.py [--zone config/missions/zone_a.geojson] [--abort-at 20] [--max-sim 600]

Prints a JSON summary: state transitions with sim times, picked vs game-status in_basket, falls,
SoC trace monotonicity, and (with --abort-at) the trunk speed 1 s after the abort.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from atharv_nandi.core import harvest_logic as hl  # noqa: E402
from atharv_nandi.core import mission_logic as ml  # noqa: E402
from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver  # noqa: E402
from atharv_nandi.core.energy_model import Battery, EnergyParams, electrical_power_w  # noqa: E402
from atharv_nandi.core.farm_scene import RIDGES, crop_body_names, crop_qpos_slice, load_farm_model, scene_config  # noqa: E402
from atharv_nandi.core.game_status import FallCounter, is_fallen  # noqa: E402
from atharv_nandi.core.geo_frame import load_site  # noqa: E402
from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch, quat_to_yaw  # noqa: E402
from atharv_nandi.core.nav_logic import CmdVelHold  # noqa: E402
from atharv_nandi.core.tripod_gait import TripodGaitGenerator  # noqa: E402

GAIT_DT = 0.02
CTRL_DT = 0.1


class World:
    def __init__(self):
        self.m = load_farm_model()
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_forward(self.m, self.d)
        self.names = crop_body_names(self.m)
        self.gait = TripodGaitGenerator(GAIT_DT)
        self.arm = list(self.m.key_ctrl[0][:7])
        self.legs = list(self.m.key_ctrl[0][7:])
        self.falls = FallCounter()
        self.n_sub = int(round(GAIT_DT / self.m.opt.timestep))
        self.trace = []  # (t, x, y)
        # actuated joint dof addresses for the power model
        self.dofs = [int(self.m.jnt_dofadr[self.m.actuator_trnid[i, 0]]) for i in range(self.m.nu)]

    def tick_gait(self):
        roll, pitch = quat_to_roll_pitch(*self.d.qpos[3:7])
        self.legs = self.gait.step(roll, pitch)
        self.d.ctrl[:7] = self.arm
        self.d.ctrl[7:] = self.legs
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)
        self.falls.update(is_fallen(self.d.qpos[0:3], self.d.qpos[3:7]))

    def power_w(self, params):
        tau = self.d.actuator_force
        qd = [self.d.qvel[a] for a in self.dofs]
        return electrical_power_w(tau, qd, params)

    def crops(self):
        return {n: tuple(self.d.qpos[crop_qpos_slice(k)][:3]) for k, n in enumerate(self.names, 1)}

    def in_basket(self):
        return hl.basket_contents(self.d.qpos, self.names)

    def pose(self):
        q = self.d.qpos
        return float(q[0]), float(q[1]), quat_to_yaw(*q[3:7])

    def speed(self):
        return float(np.linalg.norm(self.d.qvel[0:2]))


class HarvestIO:
    """HarvestSequence I/O: arm ctrl from the sequence, legs held, gait not advanced."""

    def __init__(self, w: World):
        self.w = w

    def qpos(self):
        return self.w.d.qpos.copy()

    def command(self, arm7):
        self.w.arm = list(arm7)

    def wait(self, seconds):
        for _ in range(max(1, int(round(seconds / GAIT_DT)))):
            self.w.tick_gait()

    def phase(self, name):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default=os.path.join(ROOT, "config", "missions", "zone_a.geojson"))
    ap.add_argument("--max-sim", type=float, default=600.0)
    ap.add_argument("--abort-at", type=float, default=None)
    ap.add_argument("--expired", action="store_true")
    ap.add_argument("--no-harvest", action="store_true", help="arm never grasps (counting-rule check)")
    args = ap.parse_args()

    origin, cfg = load_site()
    eparams = EnergyParams.from_dict(cfg.get("energy"))
    mcfg = ml.config_from_site(origin, cfg, geofence=None)
    mcfg.keep_out_x = ml.ridge_keep_out(RIDGES, scene_config()[1])
    ctl = ml.MissionController(mcfg)
    w = World()
    ik = DorsalArmIKSolver(w.m)
    batt = Battery(eparams)
    hold = CmdVelHold()
    for _ in range(int(1.0 / GAIT_DT)):
        w.tick_gait()

    with open(args.zone) as fh:
        zone = json.load(fh)
    utc0 = datetime(2026, 11, 17, 6, 0, tzinfo=timezone.utc)
    not_after = utc0 + (timedelta(seconds=-5) if args.expired else timedelta(minutes=30))
    cmd = {"mission_id": "sim-check-1", "mode": "harvest_zone", "zone": zone,
           "not_after": not_after.isoformat().replace("+00:00", "Z")}

    harvest = {"mode": "rl_balance", "result": ""}

    def observe():
        t = float(w.d.time)
        crops = w.crops()
        names = w.in_basket()
        return ml.Observation(
            t=t, utc=utc0 + timedelta(seconds=t), pose=w.pose(), trunk_speed=w.speed(),
            walking=w.gait.is_walking, crops=crops, crop_source="truth", in_basket=len(names),
            in_basket_names=names, fallen=w.falls.fallen, harvest_mode=harvest["mode"],
            harvest_result=harvest["result"], soc=batt.soc)

    t_submit = float(w.d.time)
    ok, reason, detail = ctl.submit(json.dumps(cmd), observe())
    transitions = [(round(float(w.d.time), 2), ctl.state, ctl.reason)]
    plan = [(round(p.x, 2), round(p.y, 2), p.kind, p.crop) for p in ctl.waypoints]
    socs, aborted_t, speed_after_abort = [], None, None
    last_state = ctl.state
    max_speed_cmd = 0.0
    while float(w.d.time) - t_submit < args.max_sim:
        obs = observe()
        act = ctl.step(obs)
        if ctl.state != last_state:
            transitions.append((round(obs.t, 2), ctl.state, ctl.reason or ctl.return_reason))
            last_state = ctl.state
        out = hold.tick(act.cmd)
        if act.force_release:
            hold.force_release()
            out = hold.tick(None)
        if "stop_motion" in act.calls:
            w.gait.stop()
        if out is not None:
            vx, vy, yaw, h = out
            w.gait.set_command(vx, vy, yaw, h >= 0.5)
            max_speed_cmd = max(max_speed_cmd, abs(vx))
        if "auto_harvest" in act.calls:
            target, msg = hl.select_target(w.d.qpos, w.names)
            if target:
                ct = hl.to_trunk(hl.crop_pos(w.d.qpos, target), w.d.qpos)
                tilt = [round(math.degrees(a), 1) for a in quat_to_roll_pitch(*w.d.qpos[3:7])]
                transitions.append((round(float(w.d.time), 2), "harvest_call",
                                    f"{target} trunk-frame {np.round(ct, 3).tolist()} roll/pitch {tilt} pose {np.round(w.pose(), 3).tolist()}"))
            if target is None:
                ctl.on_harvest_reply(False, msg)
            else:
                ctl.on_harvest_reply(True, msg)
                harvest["mode"] = "ik_harvest"
                ctl.step(observe())  # the controller sees ik_harvest at least once
                if args.no_harvest:
                    HarvestIO(w).wait(5.0)
                    res = "Grasp failed (no-harvest mode)"
                else:
                    okh, res = hl.HarvestSequence(ik, HarvestIO(w)).run(target, list(w.arm))
                w.arm = list(hl.REST_ARM)
                harvest.update(mode="rl_balance", result=res)
                transitions.append((round(float(w.d.time), 2), "harvest", f"{target}: {res}"))
        if args.abort_at is not None and aborted_t is None and obs.t - t_submit >= args.abort_at \
                and ctl.state in ml.ACTIVE_STATES:
            ctl.abort()
            aborted_t = obs.t
            # the node publishes the zero release and calls stop_motion right away
            w.gait.set_command(0.0, 0.0, 0.0, False)
            w.gait.stop()
            transitions.append((round(obs.t, 2), "abort", f"speed {w.speed():.3f}"))
        for _ in range(int(round(CTRL_DT / GAIT_DT))):
            w.tick_gait()
            batt.update(float(w.d.time), w.power_w(eparams))
        socs.append(batt.soc)
        if aborted_t is not None and speed_after_abort is None and float(w.d.time) - aborted_t >= 1.0:
            speed_after_abort = w.speed()
        if ctl.state in ("done", "failed", "aborted", "rejected") and (aborted_t is None or speed_after_abort is not None):
            # run the exit actions once more
            ctl.step(observe())
            break
    st = ctl.status()
    summary = {
        "accepted": ok, "reject_reason": reason, "detail": detail,
        "transitions": transitions, "plan": plan,
        "final": {k: st[k] for k in ("state", "reason_code", "attempted", "picked", "basket_count",
                                     "basket_capacity", "soc", "progress", "results")},
        "game_status_in_basket": w.in_basket(),
        "falls": w.falls.falls,
        "sim_s": round(float(w.d.time) - t_submit, 1),
        "soc_start_end": [round(socs[0], 5) if socs else None, round(socs[-1], 5) if socs else None],
        "soc_monotonic": all(b <= a + 1e-12 for a, b in zip(socs, socs[1:])),
        "battery": batt.status(),
        "final_pose": [round(v, 3) for v in w.pose()],
        "speed_1s_after_abort": None if speed_after_abort is None else round(speed_after_abort, 4),
        "events": ctl.events,
    }
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
