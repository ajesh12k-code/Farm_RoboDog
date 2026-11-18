#!/usr/bin/env python3
"""ROS-free, in-process Atharv Nandi harvest evaluation (no viewer, no ROS, no wall-clock pacing).

    python3 tools/headless_eval.py --seeds 1 --terrain flat --max-sim-s 60
    python3 tools/headless_eval.py --seeds 3 --terrain flat,ridges,rough --max-sim-s 300   # heavy: CI runner
    python3 tools/headless_eval.py --seed-list 7,11,23 --terrain flat --max-sim-s 60       # explicit seeds

For every (terrain, seed) it builds the live world (atharv_nandi/core/farm_scene, NANDI_TERRAIN/NANDI_SEED),
then repeats until --max-sim-s of SIM time: pick the nearest crop with atharv_nandi/core/harvest_logic
(walk toward it with atharv_nandi/core/tripod_gait when none is in reach) and run the real HarvestSequence.
All three modules are imported read-only. Physics, gait, hooks and the HarvestSequence I/O run on
atharv_nandi/core/sim_driver.SimDriver (WP0b; one MjData, per-physics-step substep point); with no hook files the
step sequence is the pre-WP0b one, so the results on a seed are unchanged. Each run writes a normal run directory (runs/<run_id>/:
meta.json, kis_v1.jsonl pose/mission/harvest, harvest_attempts.jsonl, events.jsonl,
joint_effort_summary.json from data.actuator_force vs the MJCF forcerange, kpi.json, manifest.json)
and prints one I-EAR-05-style results row (JSON) per run; --out appends the rows to a JSONL file.

Timestamps in a headless run are simulated instants (start wall ms + sim time), ts_src "sim".
"""
import argparse
import datetime as _dt
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

import mujoco  # noqa: E402

from atharv_nandi.core import harvest_logic as hl  # noqa: E402
from atharv_nandi.core import kis_mapper as km  # noqa: E402
from atharv_nandi.core import manifest as mf  # noqa: E402
from atharv_nandi.core import run_log as rl  # noqa: E402
from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver  # noqa: E402
from atharv_nandi.core.farm_scene import ROBOT_XML, crop_body_names, load_farm_model  # noqa: E402
from atharv_nandi.core.game_status import FallCounter, build_game_status, is_fallen  # noqa: E402
from atharv_nandi.core.sim_driver import SimDriver, TimeCap  # noqa: E402
from tools import nandi_kpi  # noqa: E402

CONFIG = os.path.join(REPO, "config", "telemetry.yaml")
GAIT_DT = 0.02
EFFORT_EVERY = 5          # sample actuator force every 5 physics steps (100 Hz)
STATUS_PERIOD_S = 0.5     # game status (basket truth) at 2 Hz, like sim_runner
POSE_PERIOD_S = 0.5
WALK_CHUNK_S = 0.5
MAX_FAILS_PER_CROP = 2
# Walk until the target is this close (horizontal, trunk centre): the 1.08 m contract reach is the
# selector limit, but the arm IK often cannot hover at 0.95-1.0 m, so the eval stops nearer.
STANDOFF_M = 0.75


class HeadlessWorld(SimDriver):
    """The run's world on ``controllers.sim_driver.SimDriver`` (one MjData; gait, hooks, HarvestSequence I/O),
    plus this tool's per-step observer: actuator effort (100 Hz), game status (2 Hz) and pose (2 Hz)."""

    def __init__(self, model, max_sim_s: float, tracker: rl.AttemptTracker, mapper: km.KisMapper,
                 kis: rl.JsonlWriter, events: rl.JsonlWriter, start_ms: int, run_id: str):
        super().__init__(model, gait_dt=GAIT_DT, max_sim_s=max_sim_s)
        self.crops = crop_body_names(model)
        self.tracker, self.mapper, self.kis, self.events = tracker, mapper, kis, events
        self.start_ms, self.run_id = start_ms, run_id
        self.falls = FallCounter()
        self.effort = rl.EffortStats()
        self.act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        self.act_dof = [model.jnt_dofadr[model.actuator_trnid[i][0]] for i in range(model.nu)]
        self.next_status = 0.0
        self.next_pose = 0.0
        self.harvest = {"mode": "rl_balance", "phase": "idle", "target": None, "last_result": ""}
        self.fallen_stop = False
        self.pose_nonfinite = False  # non-finite qpos in a pose row: the run stops (no NaN in kis_v1.jsonl)
        self.hold_stop: Optional[str] = None  # the driver entered safe hold (a atharv_nandi/sim/hooks file faulted): run stops
        self.add_observer(HeadlessWorld._observe)

    def ts(self) -> int:
        return self.start_ms + int(self.t * 1000)

    # -- HarvestSequence I/O (qpos / command / wait come from SimDriver)
    def phase(self, name: str):
        super().phase(name)
        self.harvest = dict(self.harvest, mode="ik_harvest", phase=name)
        self.tracker.harvest_status(self.t, self.harvest)

    # -- per-step observer (after mj_step, same order as the pre-WP0b loop)
    def _observe(self):
        d = self.d
        if self.holding and self.hold_stop is None:
            # Nothing in this tool enters hold: a atharv_nandi/sim/hooks file faulted (hook_fault). Say so and end the run
            # instead of standing frozen until --max-sim-s.
            self.hold_stop = (self.bus.values.get("hold") or {}).get("reason") or "hold"
            self.events.write({"t": round(self.t, 3), "utc_ms": self.ts(), "kind": "hold", "reason": self.hold_stop,
                               "hook_faults": dict(self.hook_faults), "run_id": self.run_id,
                               "device_id": self.mapper.device})
            self.stop(self.hold_stop)
        if self.steps % EFFORT_EVERY == 0:
            self.effort.add(self.act_names, d.actuator_force, [d.qvel[j] for j in self.act_dof])
        if self.t >= self.next_status:
            self.next_status += STATUS_PERIOD_S
            self.publish_status()
        if self.t >= self.next_pose:
            self.next_pose += POSE_PERIOD_S
            try:  # sim.t = the simulated instant in epoch ms (here the same clock as ts: ts_src "sim")
                self.emit(self.mapper.pose(self.ts(), d.qpos, self.ts()))
            except km.NonFinite:
                self.pose_nonfinite = True
                raise TimeCap("diverged")

    def emit(self, tp):
        topic, payload = tp
        self.kis.write({"topic": topic, "retain": km.retain_for(topic), "payload": payload})

    def publish_status(self):
        q = self.d.qpos
        fell = self.falls.update(is_fallen(q[0:3], q[3:7]))
        if fell:
            self.events.write({"t": round(self.t, 3), "utc_ms": self.ts(), "kind": "fall", "run_id": self.run_id,
                               "device_id": self.mapper.device})
            self.fallen_stop = True
            self.stop("fallen")
        crops = {n: q[32 + 7 * (k - 1): 32 + 7 * (k - 1) + 3] for k, n in enumerate(self.crops, start=1)}
        g = build_game_status(crops, q[0:3], q[3:7], self.falls.falls, self.falls.fallen, self.t, 0.0)
        self.tracker.crop_positions(crops)
        self.tracker.game_status(self.t, g)
        self.game = g
        for a in self.tracker.poll(self.t):
            self.on_attempt(a)

    def on_attempt(self, a):  # replaced by the runner
        pass

    # -- behaviours
    def walk_toward(self, name: str, seconds: float):
        p = hl.to_trunk(hl.crop_pos(self.d.qpos, name), self.d.qpos)
        ang = math.atan2(p[1], p[0])
        if abs(ang) > 0.35:
            self.gait.set_command(0.0, 0.0, 1.0 if ang > 0 else -1.0, hold=True)
        else:
            self.gait.set_command(1.0, 0.0, max(-1.0, min(1.0, 1.5 * ang)), hold=True)
        self.wait(seconds)


def run_one(terrain: str, seed: int, max_sim_s: float, runs_dir: str, sha: Optional[str], dirty: Optional[bool],
            cfg: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["NANDI_TERRAIN"], os.environ["NANDI_SEED"] = terrain, str(seed)
    run_id = rl.make_run_id(rl.utc_stamp(), (sha or "nogit")[:7], seed)
    while os.path.exists(os.path.join(runs_dir, run_id)):
        time.sleep(0.2)
        run_id = rl.make_run_id(rl.utc_stamp(), (sha or "nogit")[:7], seed)
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir)
    sw = rl.sw_version(sha, dirty)
    start_ms = int(time.time() * 1000)
    ident = cfg.get("identity", {}) or {}
    mapper = km.KisMapper(ident.get("tenant", "sim-nandi"), ident.get("site", "sim-field-01"),
                          ident.get("device", rl.DEVICE_ID), boot=start_ms & 0xFFFFFFFF,
                          origin=cfg.get("site_origin"), sw=sw)
    meta = rl.build_meta(run_id, REPO, [CONFIG], device_id=mapper.device,
                         extra={"tenant": mapper.tenant, "site": mapper.site, "start_utc_ms": start_ms,
                                "boot": mapper.boot, "mode": "headless_eval", "clock": "sim",
                                "max_sim_s": max_sim_s})
    rl.write_json(os.path.join(run_dir, "meta.json"), meta)

    model = load_farm_model()
    kis = rl.JsonlWriter(os.path.join(run_dir, "kis_v1.jsonl"))
    events = rl.JsonlWriter(os.path.join(run_dir, "events.jsonl"))
    attempts_log = rl.JsonlWriter(os.path.join(run_dir, "harvest_attempts.jsonl"))
    tracker = rl.AttemptTracker(sw=sw, run_id=run_id, device_id=mapper.device)
    w = HeadlessWorld(model, max_sim_s, tracker, mapper, kis, events, start_ms, run_id)
    tracker.wall_ms = w.ts
    attempts: List[Dict[str, Any]] = []

    def on_attempt(a):
        attempts.append(a)
        attempts_log.write(a)
        w.emit(mapper.harvest(w.ts(), {k: v for k, v in a.items() if k != "phases"}))
    w.on_attempt = on_attempt

    kis.write({"topic": mapper.status_topic(), "retain": True, "payload": "online"})
    w.emit(mapper.state(w.ts(), "auto", {"run_id": run_id, "gait": "tripod"}))
    ik = DorsalArmIKSolver(model)
    fails: Dict[str, int] = {}
    wall0 = time.perf_counter()
    end_reason = "time_cap"
    try:
        w.stand(1.0)  # settle on the legs
        while True:
            names = [n for n in w.crops if fails.get(n, 0) < MAX_FAILS_PER_CROP]
            free = [n for n in names if not hl.crop_in_basket(w.d.qpos, n)]
            if not free:
                end_reason = "no_crops_left"
                break
            target, _ = hl.select_target(w.d.qpos, names, reach=STANDOFF_M)
            if target is None:
                nearest = min(free, key=lambda n: hl.horizontal_distance(w.d.qpos, n))
                if hl.crop_under_body(w.d.qpos, nearest):
                    w.gait.set_command(-1.0, 0.0, 0.0, hold=True)  # back off a crop under the body
                    w.wait(WALK_CHUNK_S)
                else:
                    w.walk_toward(nearest, WALK_CHUNK_S)
                continue
            w.gait.stop()
            w.stand(0.6)
            w.harvest = {"mode": "rl_balance", "phase": "idle", "target": target, "last_result": ""}
            seq = hl.HarvestSequence(ik, w)
            ok, msg = seq.run(target, w.arm)
            w.harvest = {"mode": "rl_balance", "phase": "idle", "target": None, "last_result": msg}
            tracker.harvest_status(w.t, w.harvest)
            if not ok:
                fails[target] = fails.get(target, 0) + 1
            w.wait(0.5)
    except TimeCap:
        if w.pose_nonfinite or w.diverged is not None:
            end_reason = "diverged"
        elif w.fallen_stop:
            end_reason = "fallen"
        elif w.hold_stop is not None:
            end_reason = w.hold_stop
        else:
            end_reason = "time_cap"
        if w.harvest.get("mode") == "ik_harvest":
            w.harvest = {"mode": "rl_balance", "phase": "idle", "target": None,
                         "last_result": f"Run ended ({end_reason}) during harvest"}
            tracker.harvest_status(w.t, w.harvest)
    wall_s = time.perf_counter() - wall0
    w.publish_status()
    for a in tracker.flush(w.t):
        on_attempt(a)
    events.write({"t": round(w.t, 3), "utc_ms": w.ts(), "kind": "run_end", "reason": end_reason,
                  "hook_faults": dict(w.hook_faults), "diverged": w.diverged, "run_id": run_id, "device_id": mapper.device})
    w.emit(mapper.mission(w.ts(), km.mission_summary(w.harvest, w.game, None, run_id)))
    kis.write({"topic": mapper.status_topic(), "retain": True, "payload": "offline"})
    for f in (kis, events, attempts_log):
        f.close()

    limits = rl.mjcf_force_limits(model)
    effort = w.effort.summary(limits, source="headless_eval data.actuator_force (100 Hz)")
    effort.update({"run_id": run_id, "device_id": mapper.device, "date": meta["start_utc"][:10]})
    rl.write_json(os.path.join(run_dir, "joint_effort_summary.json"), effort)
    meta.update({"duration_s": round(w.t, 3), "wall_s": round(wall_s, 3), "end_reason": end_reason,
                 "hook_faults": dict(w.hook_faults), "diverged": w.diverged,
                 "end_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")})
    rl.write_json(os.path.join(run_dir, "meta.json"), meta)
    mf.write_manifest(run_dir, meta, final=True, closed_ms=w.ts())
    kpi = nandi_kpi.compute(run_dir)
    nandi_kpi.write(run_dir, kpi)
    k = {r["metric"]: r["value"] for r in kpi["rows"]}

    import platform
    return {  # I-EAR-05 edge-bench-result style row (+ harvest fields); simulated
        "schema": "edge-bench-result (I-EAR-05 style, PROPOSED harvest fields)",
        "device_id": mapper.device, "run_id": run_id, "date": meta["start_utc"][:10],
        "model_sha256": rl.sha256_file(ROBOT_XML),
        "runtime": f"mujoco {mujoco.__version__} / python {platform.python_version()} / headless_eval",
        "latency_ms": round(1000.0 * w.wall_in_step / max(1, w.steps), 4),
        "fps": round(w.steps / wall_s, 1) if wall_s > 0 else None,
        "watts": None,
        "accuracy": k.get("pick_success_rate"),
        "terrain": terrain, "seed": seed, "sim_s": round(w.t, 2), "wall_s": round(wall_s, 2),
        "rtf": round(w.t / wall_s, 3) if wall_s > 0 else None, "end_reason": end_reason,
        "attempts": k.get("attempts"), "picks": k.get("picks"), "picks_per_hour": k.get("picks_per_hour"),
        "drops": k.get("drops"), "false_grasp_reports": k.get("false_grasp_reports"),
        "verdict_disagreements": k.get("verdict_disagreements"), "falls": k.get("falls"),
        "distance_walked_m": k.get("distance_walked_m"),
        "hook_faults": dict(w.hook_faults), "diverged": w.diverged,
        "max_peak_pct_of_force_limit": effort.get("max_peak_pct_of_limit"),
        "sw": sw, "data_provenance": "simulated", "run_dir": os.path.relpath(run_dir, REPO),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROS-free Atharv Nandi harvest evaluation")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (7, 8, ...); default 1")
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--seed-list", default=None, help="comma list of seeds (e.g. 7,11,23); overrides --seeds/--seed0")
    ap.add_argument("--terrain", default="flat", help="comma list of flat,ridges,rough")
    ap.add_argument("--max-sim-s", type=float, default=60.0)
    ap.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--out", default=None, help="append result rows to this JSONL file")
    args = ap.parse_args(argv)
    import atharv_nandi.nodes.telemetry_bridge as tb
    cfg = tb.load_config(CONFIG)
    sha, dirty = rl.git_sha(REPO), rl.git_dirty(REPO)
    os.makedirs(args.runs_dir, exist_ok=True)
    rows = []
    seeds = ([int(s) for s in args.seed_list.split(",") if s.strip()] if args.seed_list
             else list(range(args.seed0, args.seed0 + args.seeds)))
    for terrain in [t.strip() for t in args.terrain.split(",") if t.strip()]:
        for seed in seeds:
            row = run_one(terrain, seed, args.max_sim_s, args.runs_dir, sha, dirty, cfg)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if args.out:
                with open(args.out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
