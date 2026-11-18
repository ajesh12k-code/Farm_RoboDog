#!/usr/bin/env python3
"""Headless physics measurements for Atharv Nandi (MuJoCo only, no ROS, no viewer).

    python3 tools/physics/measure_baseline.py                 # print the baseline JSON
    python3 tools/physics/measure_baseline.py --write PATH    # also write it (baseline_pre_alignment.json)
    python3 tools/physics/measure_baseline.py --only walk     # one group: walk | turn | harvest | cost

Imported by tests/test_physics_alignment.py, so the acceptance tests and the recorded baseline use exactly
the same harness. Every metric is measured on the live world built by atharv_nandi/core/farm_scene.load_farm_model()
with the env knobs given per call (NANDI_TERRAIN, NANDI_SEED, NANDI_CROPS, NANDI_PAYLOAD_KG, NANDI_MAST_H ...).

Harness conventions (shared with tests/test_tripod_gait.py and tests/test_harvest_logic.py):
  * gait at 50 Hz (TripodGaitGenerator, default GaitParams incl. balance_gain 1), physics at the model timestep;
  * a run starts at the home keyframe, settles `settle_s` with the gait idle, then walks `seconds`;
  * velocities are trunk displacement / walk time; yaw rate is the unwrapped heading change / walk time.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time
from typing import Dict, Iterable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from atharv_nandi.core import harvest_logic as hl  # noqa: E402
from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver  # noqa: E402
from atharv_nandi.core.farm_scene import load_farm_model  # noqa: E402
from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch, quat_to_yaw  # noqa: E402
from atharv_nandi.core.tripod_gait import LEG_NAMES, TripodGaitGenerator  # noqa: E402

CTRL_DT = 0.02
LEG_CTRL = slice(7, 25)
FALL_TILT_DEG = 35.0      # safety_logic TiltFallDetector threshold (spec section 6)
FALL_MIN_Z = 0.12         # game_status.FALL_MIN_TRUNK_Z
SAT_FRACTION = 0.98       # |tau| >= 0.98 * forcerange counts as saturated
ROUGH_START_X = 3.3     # rough walks start just before the clod patch (farm_scene._add_clods: x 3.6-6.5)
ENV_KEYS = ("NANDI_TERRAIN", "NANDI_SEED", "NANDI_CROPS", "NANDI_PAYLOAD_KG", "NANDI_MAST_H",
            "NANDI_MAST_HEAD_KG", "NANDI_DT")


@contextlib.contextmanager
def scene_env(**env):
    """Temporarily set NANDI_* env vars (None = unset) for load_farm_model()."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    try:
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in env.items():
            if v is not None:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build(**env) -> mujoco.MjModel:
    with scene_env(**env):
        return load_farm_model()


def _tilt_deg(q) -> float:
    qx, qy = q[1], q[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))))


class Probe:
    """Geom/actuator index sets used by the metrics."""

    def __init__(self, m: mujoco.MjModel):
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.feet = np.array([gid(f"{n}_foot") for n in LEG_NAMES])
        leg_bodies = {bid(f"{n}_{p}") for n in LEG_NAMES for p in ("thigh", "calf")}
        self.leg_capsules = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in leg_bodies
                                      and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE])
        self.floor = gid("floor")
        self.world_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == 0])
        self.leg_fr = np.abs(m.actuator_forcerange[LEG_CTRL, 1]).copy()
        self.calf_act = [7 + 3 * i + 2 for i in range(6)]
        self.leg_dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(7, 25)])

    def contacts(self, d: mujoco.MjData):
        n = d.ncon
        if n == 0:
            return np.zeros((0, 2), dtype=int), np.zeros(0)
        return np.asarray(d.contact.geom[:n]), np.asarray(d.contact.dist[:n])

    def foot_penetration(self, d) -> float:
        """Deepest foot-vs-world-geom penetration (m, >= 0)."""
        g, dist = self.contacts(d)
        if len(dist) == 0:
            return 0.0
        foot = np.isin(g[:, 0], self.feet) | np.isin(g[:, 1], self.feet)
        world = np.isin(g[:, 0], self.world_geoms) | np.isin(g[:, 1], self.world_geoms)
        sel = dist[foot & world]
        return float(max(0.0, -sel.min())) if len(sel) else 0.0

    def feet_in_contact(self, d, max_dist: float = 0.002) -> int:
        g, dist = self.contacts(d)
        n = 0
        for f in self.feet:
            hit = ((g[:, 0] == f) & np.isin(g[:, 1], self.world_geoms)) | \
                  ((g[:, 1] == f) & np.isin(g[:, 0], self.world_geoms))
            n += bool(np.any(hit & (dist <= max_dist)))
        return n

    def capsule_floor_contact(self, d) -> bool:
        g, _ = self.contacts(d)
        if len(g) == 0:
            return False
        a = np.isin(g[:, 0], self.leg_capsules) & np.isin(g[:, 1], self.world_geoms)
        b = np.isin(g[:, 1], self.leg_capsules) & np.isin(g[:, 0], self.world_geoms)
        return bool(np.any(a | b))


def walk(vx=1.0, vy=0.0, yaw=0.0, seconds=6.0, settle_s=1.0, terrain="flat", seed=7, crops=6,
         payload_kg: Optional[float] = None, model: Optional[mujoco.MjModel] = None,
         start_x: Optional[float] = None, **env) -> Dict:
    """Closed-loop tripod walk; returns speed, tilt, fall and contact/torque statistics.

    start_x: trunk x at the start (default: ROUGH_START_X on rough terrain so the walk crosses the clod
    patch at x 3.6-6.5 m, else the keyframe x = 0, which crosses both ridges)."""
    if start_x is None:
        start_x = ROUGH_START_X if terrain == "rough" else 0.0
    m = model if model is not None else build(NANDI_TERRAIN=terrain, NANDI_SEED=seed, NANDI_CROPS=crops,
                                              NANDI_PAYLOAD_KG=payload_kg, **env)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[0] += start_x
    mujoco.mj_forward(m, d)
    pr = Probe(m)
    substeps = int(round(CTRL_DT / m.opt.timestep))
    gait = TripodGaitGenerator(CTRL_DT)
    st = dict(max_tilt=0.0, min_z=math.inf, fell=False, cap=0, pen=0.0, sat=0, nsat=0, cmd_speed=0.0,
              lt3=0, ticks=0, pmech=0.0, nfeet=[0] * 7, qd_max=0.0)
    prev_ctrl = np.array(m.key_ctrl[0][LEG_CTRL])
    walking = [False]

    def tick():
        roll, pitch = quat_to_roll_pitch(*d.qpos[3:7])
        ctrl = np.array(gait.step(roll, pitch))
        d.ctrl[LEG_CTRL] = ctrl
        if walking[0]:
            st["cmd_speed"] = max(st["cmd_speed"], float(np.max(np.abs(ctrl - prev_ctrl))) / CTRL_DT)
        prev_ctrl[:] = ctrl
        for _ in range(substeps):
            mujoco.mj_step(m, d)
            if walking[0]:
                tau = np.abs(d.actuator_force[LEG_CTRL])
                st["sat"] += int(np.sum(tau >= SAT_FRACTION * pr.leg_fr))
                st["nsat"] += 18
                st["pmech"] += float(np.sum(np.abs(d.actuator_force * d.actuator_velocity))) / substeps
                st["qd_max"] = max(st["qd_max"], float(np.max(np.abs(d.qvel[pr.leg_dof]))))
        if walking[0]:
            st["ticks"] += 1
            tilt = _tilt_deg(d.qpos[3:7])
            st["max_tilt"] = max(st["max_tilt"], tilt)
            st["min_z"] = min(st["min_z"], float(d.qpos[2]))
            st["fell"] |= tilt > FALL_TILT_DEG or d.qpos[2] < FALL_MIN_Z
            st["cap"] += pr.capsule_floor_contact(d)
            st["pen"] = max(st["pen"], pr.foot_penetration(d))
            nf = pr.feet_in_contact(d)
            st["lt3"] += nf < 3
            st["nfeet"][nf] += 1

    for _ in range(int(round(settle_s / CTRL_DT))):
        tick()
    walking[0] = True
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    yaw_prev, yaw_acc = quat_to_yaw(*d.qpos[3:7]), 0.0
    gait.set_command(vx, vy, yaw, hold=True)
    n = int(round(seconds / CTRL_DT))
    for _ in range(n):
        tick()
        y_now = quat_to_yaw(*d.qpos[3:7])
        yaw_acc += (y_now - yaw_prev + math.pi) % (2 * math.pi) - math.pi
        yaw_prev = y_now
    t = n * CTRL_DT
    dx, dy = float(d.qpos[0]) - x0, float(d.qpos[1]) - y0
    return {
        "vx_mps": dx / t, "vy_mps": dy / t, "dist_m": math.hypot(dx, dy), "xy_drift_m": math.hypot(dx, dy),
        "yaw_dps": math.degrees(yaw_acc) / t, "max_tilt_deg": st["max_tilt"], "min_z": st["min_z"],
        "fell": bool(st["fell"]), "capsule_floor_ticks": st["cap"], "foot_pen_max_m": st["pen"],
        "leg_sat_frac": st["sat"] / max(1, st["nsat"]), "peak_cmd_speed_rps": st["cmd_speed"],
        "lt3_feet_frac": st["lt3"] / max(1, st["ticks"]),
        "feet_hist": [k / max(1, st["ticks"]) for k in st["nfeet"]],
        "mech_power_w": st["pmech"] / max(1, st["ticks"]),
        "peak_leg_qvel_rps": st["qd_max"],
    }


def stand(seconds=2.0, settle_s=2.0, terrain="flat", payload_kg: Optional[float] = None,
          model: Optional[mujoco.MjModel] = None, **env) -> Dict:
    """Stand at the home ctrl: trunk height, foot penetration, jitter and front/rear calf torque."""
    m = model if model is not None else build(NANDI_TERRAIN=terrain, NANDI_CROPS=1,
                                              NANDI_PAYLOAD_KG=payload_kg, **env)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    pr = Probe(m)
    d.ctrl[:] = m.key_ctrl[0]
    for _ in range(int(round(settle_s / m.opt.timestep))):
        mujoco.mj_step(m, d)
    n = int(round(seconds / m.opt.timestep))
    z, v2, qv2, pen, calf = [], 0.0, 0.0, 0.0, np.zeros(6)
    fell = False
    for k in range(n):
        mujoco.mj_step(m, d)
        z.append(float(d.qpos[2]))
        v2 += float(np.dot(d.qvel[0:3], d.qvel[0:3]))
        qv2 += float(np.mean(d.qvel[pr.leg_dof] ** 2))
        calf += np.abs(d.actuator_force[pr.calf_act])
        fell |= _tilt_deg(d.qpos[3:7]) > FALL_TILT_DEG or d.qpos[2] < FALL_MIN_Z
        if k % 10 == 0:
            pen = max(pen, pr.foot_penetration(d))
    calf /= n
    front, rear = float(np.mean(calf[0:2])), float(np.mean(calf[4:6]))
    return {"trunk_z": float(np.mean(z)), "fell": bool(fell), "foot_pen_max_m": pen,
            "trunk_speed_rms": math.sqrt(v2 / n), "leg_qvel_rms": math.sqrt(qv2 / n),
            "calf_tau_front": front, "calf_tau_mid": float(np.mean(calf[2:4])), "calf_tau_rear": rear,
            "calf_front_rear_rel": abs(front - rear) / max(front, rear),
            "n_feet": pr.feet_in_contact(d)}


class SimIO:
    """Same I/O as tests/test_harvest_logic.SimIO: 50 Hz qpos samples, slewed arm targets, legs at home."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.legs = model.key_ctrl[0][7:].copy()
        self.goal = list(model.key_ctrl[0][:7])
        self.out = list(self.goal)
        self.sample = data.qpos.copy()
        self.every = int(round(hl.PUBLISH_DT / model.opt.timestep))
        self.phases = []

    def qpos(self):
        return self.sample.copy()

    def command(self, arm7):
        self.goal = list(arm7)

    def wait(self, seconds):
        for _ in range(max(1, int(round(seconds / hl.PUBLISH_DT)))):
            self.out = hl.slew(self.out, self.goal, hl.SLEW_PER_TICK)
            self.d.ctrl[:7] = self.out
            self.d.ctrl[7:] = self.legs
            for _ in range(self.every):
                mujoco.mj_step(self.m, self.d)
            self.sample = self.d.qpos.copy()

    def phase(self, name):
        self.phases.append(name)


def harvest(seed=7, payload_kg: Optional[float] = None, target="crop_item_1", terrain=None,
            model: Optional[mujoco.MjModel] = None, still_s: float = 0.0, crop_jitter_m: float = 0.05,
            **env) -> Dict:
    """One HarvestSequence of `target` from the home pose; returns ok/msg/in_basket (+ basket rest stats).

    NANDI_SEED does not move crop_item_1 in the scene, so the crop is shifted here by up to +-crop_jitter_m in x and
    y (rng 100 + seed) before the run: different seeds are different runs (review F4). 0 = keyframe position."""
    m = model if model is not None else build(NANDI_SEED=seed, NANDI_TERRAIN=terrain,
                                              NANDI_PAYLOAD_KG=payload_kg, **env)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    k_t = int(target.rsplit("_", 1)[1])
    dxy = np.zeros(2)
    if crop_jitter_m > 0.0:
        dxy = np.random.default_rng(100 + seed).uniform(-crop_jitter_m, crop_jitter_m, 2)
        d.qpos[32 + 7 * (k_t - 1):32 + 7 * (k_t - 1) + 2] += dxy
    mujoco.mj_forward(m, d)
    io = SimIO(m, d)
    io.wait(1.0)
    ik = DorsalArmIKSolver(m)
    t0 = d.time
    ok, msg = hl.HarvestSequence(ik, io).run(target, list(m.key_ctrl[0][:7]))
    io.wait(1.0)
    out = {"ok": bool(ok), "msg": msg, "crop_shift_m": [float(v) for v in dxy],
           "in_basket": bool(hl.crop_in_basket(d.qpos, target)), "sim_s": float(d.time - t0), "phases": list(io.phases)}
    if still_s > 0.0 and out["in_basket"]:
        k = int(target.rsplit("_", 1)[1])
        sl = slice(32 + 7 * (k - 1), 32 + 7 * (k - 1) + 3)
        p0 = hl.to_trunk(d.qpos[sl], d.qpos)
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body]
        pen = 0.0
        for k2 in range(int(round(still_s / hl.PUBLISH_DT))):
            io.wait(hl.PUBLISH_DT)
            for i in range(d.ncon):
                c = d.contact[i]
                if c.geom1 in geoms or c.geom2 in geoms:
                    pen = max(pen, -float(c.dist))
        out["basket_drift_m"] = float(np.linalg.norm(hl.to_trunk(d.qpos[sl], d.qpos) - p0))
        out["basket_pen_m"] = pen
    return out


def cost(steps=3000, terrain="rough", seed=7, crops=6, model: Optional[mujoco.MjModel] = None, **env) -> Dict:
    """Mean wall-clock microseconds per mj_step while walking (gait time excluded)."""
    m = model if model is not None else build(NANDI_TERRAIN=terrain, NANDI_SEED=seed, NANDI_CROPS=crops, **env)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    gait = TripodGaitGenerator(CTRL_DT)
    gait.set_command(1.0, 0.0, 0.0, hold=True)
    sub = int(round(CTRL_DT / m.opt.timestep))
    spent, n = 0.0, 0
    while n < steps:
        roll, pitch = quat_to_roll_pitch(*d.qpos[3:7])
        d.ctrl[LEG_CTRL] = gait.step(roll, pitch)
        t = time.perf_counter()
        for _ in range(sub):
            mujoco.mj_step(m, d)
        spent += time.perf_counter() - t
        n += sub
    return {"us_per_step": 1e6 * spent / n, "timestep": float(m.opt.timestep),
            "rtf": m.opt.timestep / (spent / n)}


# ---------------------------------------------------------------------------------- baseline
def baseline(groups: Iterable[str]) -> Dict:
    groups = set(groups)
    out = {"mujoco": mujoco.__version__, "harness": "tools/physics/measure_baseline.py"}
    if "walk" in groups:
        w = {"flat": walk(terrain="flat")}
        for terr in ("ridges", "rough"):
            for s in (1, 2, 3):
                w[f"{terr}_seed{s}"] = walk(terrain=terr, seed=s)
        out["walk_vx1_6s"] = w
    if "turn" in groups:
        out["turn_6s"] = {"left": walk(vx=0.0, yaw=1.0, terrain="flat"),
                          "right": walk(vx=0.0, yaw=-1.0, terrain="flat")}
    if "harvest" in groups:
        runs = [harvest(seed=s) for s in range(1, 11)]
        out["harvest_crop1"] = {"success": sum(r["ok"] and r["in_basket"] for r in runs), "runs": len(runs),
                                "msgs": [r["msg"] for r in runs]}
    if "cost" in groups:
        out["cost_rough_6crops"] = min((cost() for _ in range(3)), key=lambda r: r["us_per_step"])
    if "stand" in groups:
        out["stand_flat"] = stand()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="walk,turn,harvest,cost,stand")
    ap.add_argument("--write")
    a = ap.parse_args(argv)
    res = baseline(a.only.split(","))
    txt = json.dumps(res, indent=1, sort_keys=True)
    print(txt)
    if a.write:
        with open(a.write, "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
