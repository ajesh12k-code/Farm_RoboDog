#!/usr/bin/env python3
"""WP11 budget and evidence: the orchard tour benchmark with run ids (ORCHARD-MVP-SPEC 1.1, 1.2, 3.7; MVP-SPEC 3.8).

Acceptance (WP11): **RTF >= 0.9 over the tour on the CI runner; <= 20 min wall.** CI-runner runs are not
authorised yet, so every section runs in the capped local runner (2 CPU / 2 GB, 180 s per run) and the
CI-runner column stays ``pending`` until an approved CI run re-runs the same commands with ``NANDI_DEVICE_ID``.

Sections (one per runner call; each writes one JSON with ``identity.run_id = wp11-<section>-<utc>-<sha12>[-dirty]``,
``device_id``, ``git``, ``tool_blob``):

    world      the real composed world (``NANDI_WORLD=orchard_demo`` / ``orchard_lite``), no GL, no ROS: nq / nv / eq
               rows at rest / mocap, render geoms per tree, raw physics ms/step over N steps (robot holding the home
               keyframe), the full SimDriver step with every hook loaded (what the tour pays per step), BADQACC,
               scene geoms after the 4 m cull at the TREE 360 stations and the alleys, LiDAR 1,620-ray sweep cost.
    capture    GL: culled RGB + depth + seg captures at 320 x 240 at the OT-1 TREE 360 stations (low = ring facing
               the tree, mid / high = crown) on the perception model, p90, scene geoms, and the
               ``camera_rig.RenderBudget`` replay of WP11a (gate E).
    tour       one tour segment (or a step list) headless on SimBackend (the code path ``/nandi/demo/run`` uses),
               instrumented: per step sim s / wall s / CPU s / physics wall, a 10 s sim RTF timeline, the tour pass.
               Unpaced (default) it measures the **headroom** sim_s / wall_s: a window >= 0.9 means this host can run
               that part of the tour at RTF 0.9. ``--pace 0.9`` wall-paces the driver like the demo and measures the
               realised RTF per window.
    live       the live stack (``launch/launch_nandi.sh``, ``NANDI_HEADLESS=1``, the chosen world) plus
               ``atharv_nandi/nodes/demo_tour_node.py``; posts ``/nandi/demo/run``, samples ``/nandi/game_status`` RTF (the paced
               sim_runner sharing the CPUs), ``/nandi/demo/state`` and CPU per process from ``/proc``.
    summary    pure: merges tour / live JSONs into the tour verdict (sim s, wall s, min window RTF, projected wall
               at RTF 0.9) next to the gates, CI-runner column pending.

Run (host computes the sha; the container cannot read a worktree ``.git``)::

    R=<nandi-sim>; W=<worktree>; SHA=$(git -C $W rev-parse --short=12 HEAD)$(git -C $W diff --quiet HEAD || echo -dirty)
    $R $W python3 tools/spot_tour_bench.py world   --git $SHA --world orchard_demo --out runs/wp11/world_demo.json
    $R $W python3 tools/spot_tour_bench.py capture --git $SHA --out runs/wp11/capture.json
    $R $W python3 tools/spot_tour_bench.py tour    --git $SHA --segment B1 --out runs/wp11/tour_B1.json [--pace 0.9]
    $R $W python3 tools/spot_tour_bench.py live    --git $SHA --program tour_lite --seconds 110 --out runs/wp11/live.json
    $R $W python3 tools/spot_tour_bench.py summary --git $SHA runs/wp11/tour_*.json --out runs/wp11/summary.json

On the CI runner the full tour runs in one ``tour --program orchard_tour`` call (no 180 s cap); locally it is split into
the WP9 segments (``SEGMENTS``, same as ``tests/test_tour_demo.py``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ------------------------------------------------------------------ gates (spec) and constants
RTF_TARGET = 0.9                 # ORCHARD-MVP-SPEC 1.1 / WP11: RTF >= 0.9 over the tour
RTF_TOLERANCE = 0.02             # paced runs: realised RTF >= target - tolerance (as WP11a gate C)
WALL_GATE_S = 20 * 60.0          # <= 20 min wall
SIM_GATE_S = 18 * 60.0           # <= 18 min sim (OS 1.2)
WINDOW_SIM_S = 10.0              # RTF timeline window (sim seconds)
PACE_HZ = 50.0                   # pacing task rate (sim Hz)
CI_PENDING = "pending: CI-runner runs not authorised yet (2026-11-17)"
CULL_DIST_M = 4.0
STANDOFF_M = 0.9                 # TREE 360 lens-to-canopy stand-off (OS 1.2 step 2)
STATIONS = 8
CAPTURE_W, CAPTURE_H = 320, 240
CROWN_MID = (1.70, -10.0)        # lens height (m), gimbal pitch (deg)
CROWN_HIGH = (2.70, -20.0)
ACK_WINDOW_SIM_S = 0.5           # 3.7: perception ack budget per capture window (sim s)
PASSES_BUDGET = 3                # 2 GL passes per capture, budgeted as 3
ORBIT_SIM_S = 165.0              # O-06

# WP9 segments: each a fresh world, <= 180 s wall on the 2 CPU runner; flag = actors on (tests/test_tour_demo.py)
SEGMENTS: Dict[str, Tuple[List[str], bool]] = {
    "A": (["s00_ready", "s01_scan_r1", "s01b_headland", "s02_tree360", "s03_leaf_scan", "s04_fruit_scout"], True),
    "B1": (["s05_harvest_mango"], False),
    "B2": (["s06_harvest_guava", "s07_prune_guava", "s08_harvest_guava_twist"], True),
    "C1": (["s08b_harvest_papaya", "s09_prune_banana", "s10b_patch_refusal"], False),
    "C2": (["s10a_gust", "s10e_estop", "s11_dock_unload", "s11b_allowance", "s12_report"], False),
}

_GIT_ARG_RE = r"^[0-9a-f]{7,40}(-dirty)?$"
RUN_ID_RE = r"^wp11-[a-z]+(-[A-Za-z0-9_]+)?-\d{8}T\d{6}Z-([0-9a-f]{7,40}(-dirty)?|nogit)$"


# ------------------------------------------------------------------ identity (run-identity rule)
def check_git_arg(value: str) -> str:
    v = (value or "").strip()
    if not re.match(_GIT_ARG_RE, v):
        raise SystemExit(f"--git {value!r}: expected <hex sha>[-dirty], e.g. "
                         "$(git rev-parse --short=12 HEAD)$(git diff --quiet HEAD || echo -dirty)")
    return v


def tool_blob_sha(path: str = __file__) -> str:
    """git blob id of this file as run (``git rev-parse <commit>:tools/spot_tour_bench.py`` must equal it)."""
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def cgroup_cpus() -> Optional[float]:
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else round(int(q) / int(p), 2)
    except Exception:
        return None


def device_id() -> str:
    return os.environ.get("NANDI_DEVICE_ID") or f"mac-capped-runner-{cgroup_cpus() or os.cpu_count()}cpu"


def _git(sha: Optional[str]) -> str:
    if sha:
        return check_git_arg(sha)
    if os.environ.get("NANDI_GIT_SHA"):
        return check_git_arg(os.environ["NANDI_GIT_SHA"])
    try:
        from atharv_nandi.core import run_log
        s = run_log.git_sha(REPO) or "nogit"
        return s[:12] + ("-dirty" if run_log.git_dirty(REPO) else "")
    except Exception:
        return "nogit"


def make_run_id(section: str, sha: str, utc: Optional[str] = None, tag: str = "") -> str:
    utc = utc or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = re.sub(r"[^A-Za-z0-9_]", "_", tag) if tag else ""
    return f"wp11-{section}{'-' + tag if tag else ''}-{utc}-{sha}"


def identity(section: str, sha: Optional[str] = None, tag: str = "") -> dict:
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    g = _git(sha)
    out = {"run_id": make_run_id(section, g, utc, tag), "device_id": device_id(), "git": g,
           "tool_blob": tool_blob_sha(), "utc": utc, "python": platform.python_version(),
           "machine": platform.machine(), "cgroup_cpus": cgroup_cpus(), "os_cpu_count": os.cpu_count(),
           "env": {k: os.environ.get(k) for k in ("LP_NUM_THREADS", "MUJOCO_GL", "NANDI_WORLD", "NANDI_SEED",
                                                   "NANDI_TERRAIN", "NANDI_ACTORS")},
           "ci_runner": CI_PENDING, "data_provenance": "simulated"}
    try:
        import mujoco
        out["mujoco"] = mujoco.__version__
    except Exception:
        pass
    return out


# ------------------------------------------------------------------ pure helpers (unit-tested, no MuJoCo)
def stats(v: Sequence[float], nd: int = 3) -> dict:
    a = sorted(float(x) for x in v)
    if not a:
        return {"n": 0}

    def pct(p):
        k = (len(a) - 1) * p / 100.0
        lo, hi = math.floor(k), math.ceil(k)
        return a[lo] + (a[hi] - a[lo]) * (k - lo)
    return {"n": len(a), "median": round(pct(50), nd), "mean": round(sum(a) / len(a), nd), "p90": round(pct(90), nd),
            "max": round(a[-1], nd), "min": round(a[0], nd)}


def pace_sleep_s(sim_elapsed: float, wall_elapsed: float, rtf: float) -> float:
    """Wall seconds to sleep so that sim time does not run ahead of ``rtf`` x wall time (0 when behind)."""
    if rtf <= 0:
        return 0.0
    return max(0.0, sim_elapsed / rtf - wall_elapsed)


def rtf_windows(samples: Sequence[Tuple[float, float]], window_s: float = WINDOW_SIM_S,
                exclude: Sequence[Tuple[float, float]] = ()) -> List[dict]:
    """``samples`` = (sim t, wall t), increasing. Cut into consecutive windows of >= ``window_s`` sim seconds;
    RTF of a window = d_sim / d_wall. Wall intervals in ``exclude`` (operator pauses, world builds) are taken out of
    d_wall. A trailing remainder shorter than half a window is merged into the last window."""
    out: List[dict] = []
    if len(samples) < 2:
        return out

    def excl(w0, w1):
        return sum(max(0.0, min(w1, b) - max(w0, a)) for a, b in exclude)
    i0 = 0
    for i in range(1, len(samples)):
        if samples[i][0] - samples[i0][0] >= window_s - 1e-9:
            out.append((i0, i))
            i0 = i
    if i0 < len(samples) - 1:
        if out and samples[-1][0] - samples[i0][0] < 0.5 * window_s:
            out[-1] = (out[-1][0], len(samples) - 1)
        else:
            out.append((i0, len(samples) - 1))
    rows = []
    for a, b in out:
        ds = samples[b][0] - samples[a][0]
        dw = samples[b][1] - samples[a][1] - excl(samples[a][1], samples[b][1])
        rows.append({"sim_t0": round(samples[a][0], 2), "sim_s": round(ds, 3), "wall_s": round(dw, 3),
                     "rtf": round(ds / dw, 3) if dw > 1e-9 else None})
    return rows


def window_summary(windows: Sequence[dict], rtf_min: float) -> dict:
    r = [w["rtf"] for w in windows if w.get("rtf") is not None]
    below = [w for w in windows if w.get("rtf") is not None and w["rtf"] < rtf_min]
    return {"n": len(r), "min": round(min(r), 3) if r else None, "median": stats(r)["median"] if r else None,
            "below_n": len(below), "below_sim_s": round(sum(w["sim_s"] for w in below), 1),
            "worst": sorted(below, key=lambda w: w["rtf"])[:5]}


def tour_verdict(sim_s: float, wall_s: float, windows_min_rtf: Optional[float], paced: bool,
                 rtf_target: float = RTF_TARGET, tol: float = RTF_TOLERANCE, live_rtf_min: Optional[float] = None,
                 complete: bool = True, pace: Optional[float] = None, tour_pass: Optional[bool] = None,
                 live_failed: Sequence[str] = ()) -> dict:
    """The WP11 gates for one tour (or a set of segments that together cover the tour).

    * ``rtf``: unpaced, every window's headroom >= ``rtf_target`` (this host can hold RTF 0.9 everywhere); paced,
      every realised window RTF >= pace - tol (``pace`` defaults to ``rtf_target``; the same threshold
      ``window_summary`` counts against). A live-stack measurement (sim_runner RTF while the tour runs) must also
      hold >= target - tol when given; a failed live run (``live_failed``: run not sent, a process died, no
      sim_runner window) makes ``rtf_ok`` False, never silently green.
    * ``wall_20min``: the wall time an observer waits for the tour **at the demo pace**: ``sim_s / rtf_target``
      (an unpaced run finishes sooner, but the demo is watched at RTF ~0.9, so the projection is the gate); the
      measured unpaced wall is reported next to it.
    * ``sim_18min``: OS 1.2.
    * ``tour_pass``: the WP9 tour pass (every segment), reported as its own gate row; it does not change ``rtf_ok``.
    ``complete`` False (a segment, not the whole tour) keeps the time gates informative only (``None``)."""
    paced_min = (pace if pace else rtf_target) - tol
    ok_rtf = None
    if windows_min_rtf is not None:
        ok_rtf = windows_min_rtf >= (paced_min if paced else rtf_target)
        if live_rtf_min is not None:
            ok_rtf = ok_rtf and live_rtf_min >= rtf_target - tol
    if live_failed:
        ok_rtf = False
    wall_demo = sim_s / rtf_target
    return {"rtf_ok": ok_rtf, "window_rtf_min": windows_min_rtf, "live_rtf_min": live_rtf_min,
            "live_failed": list(live_failed), "paced": paced, "pace": pace if paced else None,
            "paced_rtf_min": round(paced_min, 3) if paced else None, "tour_pass": tour_pass,
            "sim_s": round(sim_s, 1), "wall_s_measured": round(wall_s, 1),
            "wall_s_at_rtf_target": round(wall_demo, 1),
            "sim_18min_ok": (sim_s <= SIM_GATE_S) if complete else None,
            "wall_20min_ok": (wall_demo <= WALL_GATE_S) if complete else None,
            "wall_20min_ok_unpaced": (wall_s <= WALL_GATE_S) if complete else None,
            "complete_tour": complete, "ci_runner": CI_PENDING}


def live_failures(res: dict) -> List[str]:
    """Why a ``live`` section result cannot count as a measurement (empty = usable)."""
    why = []
    if res.get("run_sent_wall_s") is None:
        why.append("run_not_sent")
    if res.get("stack_alive") is False:
        why.append("stack_died")
    if res.get("tour_node_alive") is False:
        why.append("tour_node_died")
    if not res.get("sim_runner_windows") and (res.get("sim_runner_window_rtf") or {}).get("min") is None:
        why.append("no_sim_runner_windows")
    last = res.get("tour_last_state") or {}
    if last.get("state") == "error":
        why.append("tour_error")
    return why


def merge_tours(docs: Sequence[dict]) -> dict:
    """Sum the ``tour`` section JSONs of segments that together are one tour (each step id once)."""
    steps, seen, dup = [], set(), []
    sim_s = wall_s = build_s = 0.0
    wins: List[dict] = []
    passed, paced, paces = True, set(), set()
    runs = []
    for d in docs:
        runs.append(d["identity"]["run_id"])
        r = d["result"]
        paced.add(bool(r.get("pace")))
        if r.get("pace"):
            paces.add(float(r["pace"]))
        for s in r["steps"]:
            if s["id"] in seen:
                dup.append(s["id"])
            seen.add(s["id"])
            steps.append(s)
        sim_s += r["sim_s"]
        wall_s += r["wall_s"]
        build_s += r.get("build_wall_s", 0.0)
        wins += r["windows"]
        passed = passed and bool(r["tour_pass"])
    return {"runs": runs, "steps": steps, "duplicate_steps": dup, "sim_s": round(sim_s, 1),
            "wall_s": round(wall_s, 1), "build_wall_s": round(build_s, 1), "windows": wins,
            "tour_pass_all_segments": passed, "paced": paced == {True}, "mixed_pacing": len(paced) > 1,
            "pace": next(iter(paces)) if len(paces) == 1 else None, "mixed_pace": len(paces) > 1}


# ------------------------------------------------------------------ world (no GL, no ROS)
def _set_world_env(world: str, seed: int, actors: bool = True):
    os.environ["NANDI_WORLD"] = world
    os.environ["NANDI_SEED"] = str(int(seed))
    os.environ.setdefault("NANDI_NOVNC_UP", "0")
    os.environ["NANDI_ACTORS"] = "1" if actors else "0"
    os.environ.setdefault("LP_NUM_THREADS", "1")


def load_world(world: str, seed: int = 7):
    _set_world_env(world, seed)
    from atharv_nandi.core import farm_scene
    return farm_scene.load_farm_model()


def field_doc(world: str) -> dict:
    return json.load(open(os.path.join(REPO, "config", "fields", f"{world}.json")))


def tree360_stations(tree: dict, n: int = STATIONS, standoff: float = STANDOFF_M) -> List[Tuple[float, float, float]]:
    """(x, y, yaw) of the robot at the n orbit stations, tree on the left (CCW orbit), r = canopy_r + d."""
    cx, cy = tree["geo"]["x"], tree["geo"]["y"]
    r = float(tree.get("canopy_r_m", 1.0)) + float(tree.get("orbit_standoff_m", standoff) or standoff)
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        out.append((x, y, a + math.pi / 2))            # heading along the CCW tangent: the tree is on the left
    return out


def alley_poses(doc: dict, step_m: float = 2.0) -> List[Tuple[float, float, float]]:
    """Robot poses along every row's south alley (between rows), heading +x."""
    rows = doc.get("rows") or []
    trees = doc.get("trees") or []
    poses = []
    for i, r in enumerate(rows):
        xs = [t["geo"]["x"] for t in trees if t.get("row_id") == r["row_id"]]
        if not xs:
            continue
        y = r["y"] - 0.5 * ((r["y"] - rows[i - 1]["y"]) if i > 0 else 2.5)
        x = min(xs) - 1.0
        while x <= max(xs) + 1.0 + 1e-9:
            poses.append((round(x, 2), round(y, 2), 0.0))
            x += step_m
    return poses


def _home(m, d):
    import mujoco
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)


def _pose(m, d, q0, pose):
    """Teleport the robot trunk (free joint 0) to (x, y, yaw); everything else as in q0."""
    import mujoco
    x, y, yaw = pose
    d.qpos[:] = q0
    d.qpos[0], d.qpos[1] = x, y
    d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)


def eq_rows_at_rest(m, d) -> int:
    import mujoco
    import numpy as np
    mujoco.mj_forward(m, d)
    return int(np.sum(np.asarray(d.efc_type[:d.nefc]) == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)))


def scene_cull_counts(m, poses, cameras=("ring_0", "ring_1", "ring_2", "ring_3", "crown_cam"),
                      dist: float = CULL_DIST_M) -> dict:
    """mjvScene geom count before / after the 4 m post-filter (``controllers.scene_cull``) at the given poses."""
    import mujoco
    from atharv_nandi.core.camera_rig import RENDER_GROUPS
    from atharv_nandi.core.scene_cull import SceneCuller
    d = mujoco.MjData(m)
    _home(m, d)
    q0 = d.qpos.copy()
    scn = mujoco.MjvScene(m, maxgeom=20000)
    opt = mujoco.MjvOption()
    for g, on in enumerate(RENDER_GROUPS):
        opt.geomgroup[g] = on
    culler = SceneCuller(m, scn, dist)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in cameras]
    before, after = [], []
    for p in poses:
        _pose(m, d, q0, p)
        for cid in ids:
            if cid < 0:
                continue
            cam.fixedcamid = cid
            mujoco.mjv_updateScene(m, d, opt, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            b, a = culler.cull(d.cam_xpos[cid])
            before.append(b)
            after.append(a)
    return {"poses": len(poses), "cameras": [c for c, i in zip(cameras, ids) if i >= 0],
            "before": stats(before, 1), "after": stats(after, 1)}


def physics_bench(m, steps: int = 2000, settle: int = 300) -> dict:
    """Raw ``mj_step`` with the robot holding the home keyframe (no gait, no hooks): the WP11a physics row."""
    import mujoco
    d = mujoco.MjData(m)
    _home(m, d)
    if m.nkey:
        d.ctrl[:] = m.key_ctrl[0]
    for _ in range(settle):
        mujoco.mj_step(m, d)
    t0 = time.perf_counter()
    for _ in range(steps):
        mujoco.mj_step(m, d)
    wall = time.perf_counter() - t0
    ms = wall / steps * 1e3
    return {"steps": steps, "ms_per_step": round(ms, 4), "rtf_physics_only": round(m.opt.timestep / (ms / 1e3), 1),
            "core_frac_at_rtf_0_9": round(RTF_TARGET * ms / 1e3 / m.opt.timestep, 4),
            "badqacc": int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number),
            "nefc_end": int(d.nefc), "ncon_end": int(d.ncon), "trunk_z_end": round(float(d.qpos[2]), 3)}


def driver_bench(m, seconds: float = 4.0) -> dict:
    """The WP0b ``SimDriver`` standing with every ``atharv_nandi/sim/hooks`` file loaded: the per-step cost a tour pays."""
    from atharv_nandi.core.sim_driver import SimDriver
    drv = SimDriver(m)
    try:
        drv.stand(0.5)
        w_in0, n0 = drv.wall_in_step, drv.steps
        t0 = time.perf_counter()
        drv.stand(float(seconds))
        wall = time.perf_counter() - t0
        n = drv.steps - n0
        phys = drv.wall_in_step - w_in0
        return {"sim_s": seconds, "steps": n, "ms_per_step": round(wall / n * 1e3, 4),
                "physics_ms_per_step": round(phys / n * 1e3, 4), "headroom_rtf": round(seconds / wall, 2),
                "core_frac_at_rtf_0_9": round(RTF_TARGET * wall / seconds, 3),
                "hooks": sorted(getattr(drv.hooks, "_hooks", {}).keys())}
    finally:
        drv.close()


def lidar_bench(m, poses, reps: int = 1) -> dict:
    import mujoco
    from atharv_nandi.core.lidar_model import LidarModel, SAFETY_GROUPS, COVERAGE_GROUPS
    d = mujoco.MjData(m)
    _home(m, d)
    q0 = d.qpos.copy()
    lid = LidarModel(m)
    t_s, t_c = [], []
    for _ in range(reps):
        for p in poses:
            _pose(m, d, q0, p)
            t0 = time.perf_counter()
            lid.sweep(d, SAFETY_GROUPS)
            t1 = time.perf_counter()
            lid.sweep(d, COVERAGE_GROUPS)
            t_s.append((t1 - t0) * 1e3)
            t_c.append((time.perf_counter() - t1) * 1e3)
    return {"rays": int(lid.n_rays),
            "safety_ms": stats(t_s), "coverage_ms": stats(t_c)}


def render_geoms_per_tree(m) -> dict:
    from atharv_nandi.sim.world.modules import trees as trees_mod
    plans = trees_mod.plans_for_model(m)
    per = {p["tree_id"]: int(p["render_geoms"]) for p in plans}
    return {"n_trees": len(per), "max": max(per.values()) if per else 0, "per_tree": per}


def sec_world(args) -> dict:
    import mujoco
    m = load_world(args.world, args.seed)
    d = mujoco.MjData(m)
    _home(m, d)
    doc = field_doc(args.world)
    trees = {t["tree_id"]: t for t in doc.get("trees") or []}
    orbit_tree = trees.get(args.tree) or next(iter(trees.values()))
    stations = tree360_stations(orbit_tree)
    alleys = alley_poses(doc)
    out = {"world": args.world, "seed": args.seed,
           "counts": {"nq": m.nq, "nv": m.nv, "nu": m.nu, "nbody": m.nbody, "ngeom": m.ngeom, "neq": m.neq,
                      "nmocap": m.nmocap, "eq_rows_at_rest": eq_rows_at_rest(m, d), "timestep": m.opt.timestep},
           "render_geoms_per_tree": render_geoms_per_tree(m),
           "physics": physics_bench(m, args.steps),
           "scene_cull_stations": dict(scene_cull_counts(m, stations), tree=orbit_tree["tree_id"]),
           "scene_cull_alleys": scene_cull_counts(m, alleys),
           "lidar": lidar_bench(m, stations[:4] + alleys[::3])}
    if not args.no_driver:
        out["driver"] = driver_bench(m, args.driver_s)
    return out


# ------------------------------------------------------------------ capture (GL)
def ensure_display():
    """GL for the capture / live sections (as WP11a): start Xvfb when MUJOCO_GL is glfw or unset and no X display
    answers; returns the process (or None)."""
    import shutil
    gl = os.environ.get("MUJOCO_GL", "")
    if gl not in ("", "glfw") or sys.platform == "darwin":
        return None
    disp = os.environ.get("DISPLAY")
    if disp and shutil.which("xdpyinfo"):
        if subprocess.run(["xdpyinfo", "-display", disp], capture_output=True).returncode == 0:
            return None
    if not shutil.which("Xvfb"):
        return None
    os.environ["DISPLAY"] = ":97"
    os.environ["MUJOCO_GL"] = "glfw"
    p = subprocess.Popen(["Xvfb", ":97", "-screen", "0", "1280x800x24"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    return p


def token_bucket_skips(capture_ms: Sequence[float], window_wall_s: float, frac: float = 0.35,
                       burst_s: float = 0.3) -> int:
    """Replay ``camera_rig.RenderBudget`` (WP11a gate E) from a full bucket: captures one ``window_wall_s`` apart,
    each charged. Call it per station (the ~10 s arc between stations refills the bucket to its cap)."""
    from atharv_nandi.core.camera_rig import RenderBudget
    b = RenderBudget(frac, burst_s)
    now, skips = 0.0, 0
    for c in capture_ms:
        if b.allow(now):
            b.charge(c / 1e3)
        else:
            skips += 1
        now += window_wall_s
    return skips


def burst_for_zero_skips(capture_ms: Sequence[float], window_wall_s: float, frac: float = 0.35) -> float:
    """Smallest RenderBudget burst (1 ms steps, rounded up) for which the replay skips nothing (monotone)."""
    lo, hi = 0.0, sum(capture_ms) / 1e3 + 1e-3
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if token_bucket_skips(capture_ms, window_wall_s, frac, mid) == 0:
            hi = mid
        else:
            lo = mid
    need = math.ceil(hi * 1e3) / 1e3
    return need if token_bucket_skips(capture_ms, window_wall_s, frac, need) == 0 else need + 1e-3


def sec_capture(args) -> dict:
    xv = ensure_display()
    try:
        import mujoco
        _set_world_env(args.world, args.seed)
        from atharv_nandi.core import camera_rig
        m = camera_rig.build_perception_model()
        d = mujoco.MjData(m)
        _home(m, d)
        q0 = d.qpos.copy()
        doc = field_doc(args.world)
        tree = next(t for t in doc["trees"] if t["tree_id"] == args.tree)
        cam = camera_rig.CameraRenderer(m, camera_rig.CROWN_CAM, CAPTURE_W, CAPTURE_H,
                                        cull_dist=None if args.no_cull else CULL_DIST_M)
        crown = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_rig.CROWN_CAM)
        cam_pos0 = m.cam_pos[crown].copy()
        cam_quat0 = m.cam_quat[crown].copy()
        rows = []
        try:
            _pose(m, d, q0, tree360_stations(tree)[0])
            cam.capture(d)                                  # warm-up (first render builds GL state), not counted
            for _ in range(args.reps):
                for si, st in enumerate(tree360_stations(tree)):
                    _pose(m, d, q0, st)
                    ring = camera_rig.ring_facing(m, d, (tree["geo"]["x"], tree["geo"]["y"]))
                    station = []
                    for band, cname, cfg in (("low", ring, None), ("mid", camera_rig.CROWN_CAM, CROWN_MID),
                                             ("high", camera_rig.CROWN_CAM, CROWN_HIGH)):
                        aim_err = None
                        if cfg is not None:
                            # crown lens height above ground and the gimbal aimed at the trunk as TREE 360 does
                            # (tree360._aim: pan = trunk bearing - head heading, the band tilt; hooks/mast
                            # gimbal_quat). The WP24 provider writes these in the live model; here directly on the
                            # perception model's camera, restored after.
                            body_z = float(d.xpos[m.cam_bodyid[crown]][2])
                            m.cam_pos[crown][2] = cam_pos0[2] + (cfg[0] - (body_z + cam_pos0[2]))
                            mujoco.mj_camlight(m, d)
                            m.cam_quat[crown] = crown_gimbal_quat(
                                d.cam_xpos[crown][:2], head_heading(d.xmat[m.cam_bodyid[crown]]),
                                (tree["geo"]["x"], tree["geo"]["y"]), cfg[1])
                            mujoco.mj_camlight(m, d)
                            aim_err = round(look_bearing_err_deg(d.cam_xmat[crown], d.cam_xpos[crown][:2],
                                                                 (tree["geo"]["x"], tree["geo"]["y"])), 2)
                        cap = cam.use(cname).capture(d)
                        station.append(cap["ms"]["total"])
                        rows.append({"station": si, "band": band, "camera": cname, "ms": round(cap["ms"]["total"], 2),
                                     "rgbd_ms": round(cap["ms"]["rgbd"], 2), "seg_ms": round(cap["ms"]["seg"], 2),
                                     "ngeom": list(cap["ngeom"]), "aim_err_deg": aim_err})
                    m.cam_pos[crown] = cam_pos0
                    m.cam_quat[crown] = cam_quat0
        finally:
            cam.close()
            m.cam_pos[crown] = cam_pos0
            m.cam_quat[crown] = cam_quat0
        ms = [r["ms"] for r in rows]
        p90 = stats(ms)["p90"]
        # 3-pass budget: a 2-pass capture x 1.5; one capture per 0.5 s sim window at RTF 0.9 = 0.556 s wall
        win = ACK_WINDOW_SIM_S / RTF_TARGET
        budget_ms = [x * PASSES_BUDGET / 2.0 for x in ms]
        return {"world": args.world, "tree": args.tree, "cull": not args.no_cull, "size": [CAPTURE_W, CAPTURE_H],
                "captures": len(rows), "capture_ms": stats(ms, 1),
                "crown_aim_err_deg_max": max((r["aim_err_deg"] for r in rows if r["aim_err_deg"] is not None),
                                             default=None),
                "by_band": {b: stats([r["ms"] for r in rows if r["band"] == b], 1) for b in ("low", "mid", "high")},
                "scene_geoms_after": stats([r["ngeom"][1] for r in rows], 1),
                "scene_geoms_before": stats([r["ngeom"][0] for r in rows], 1),
                "station_3pass_s_p90": round(3 * p90 * 1.5 / 1e3, 3),
                "render_budget_replay_skips": sum(token_bucket_skips(budget_ms[i:i + 3], win, 0.35, args.burst)
                                                 for i in range(0, len(budget_ms), 3)),
                "burst_s_for_zero_skips": round(max(burst_for_zero_skips(budget_ms[i:i + 3], win)
                                                    for i in range(0, len(budget_ms), 3)), 3),
                "render_budget_replay_skips_p90_station": token_bucket_skips([p90 * PASSES_BUDGET / 2.0] * 3, win,
                                                                            0.35, args.burst),
                "render_budget": {"frac": 0.35, "burst_s": args.burst, "window_wall_s": round(win, 3)},
                "rows": rows}
    finally:
        if xv:
            xv.terminate()


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def crown_pan(cam_xy, heading: float, tree_xy) -> float:
    """Gimbal pan (head frame, rad) that points the crown lens at the trunk: as ``tree360._aim``."""
    return _wrap(math.atan2(tree_xy[1] - cam_xy[1], tree_xy[0] - cam_xy[0]) - heading)


def head_heading(xmat) -> float:
    """Yaw of a body frame from its world rotation matrix (row-major 9 floats): the bearing of its +x axis."""
    r = [float(v) for v in list(xmat)[:9]]
    return math.atan2(r[3], r[0])


def crown_gimbal_quat(cam_xy, heading: float, tree_xy, tilt_deg: float):
    """``cam_quat`` of the crown lens aimed at the trunk with the band tilt (``hooks/mast.gimbal_quat``)."""
    from atharv_nandi.sim.hooks import mast as hm
    return hm.gimbal_quat(crown_pan(cam_xy, heading, tree_xy), math.radians(tilt_deg))


def look_bearing_err_deg(cam_xmat, cam_xy, tree_xy) -> float:
    """Horizontal angle (deg) between a camera's view direction (-z) and the bearing to the trunk."""
    r = [float(v) for v in list(cam_xmat)[:9]]
    fx, fy = -r[2], -r[5]
    return abs(math.degrees(_wrap(math.atan2(fy, fx) - math.atan2(tree_xy[1] - cam_xy[1], tree_xy[0] - cam_xy[0]))))


# ------------------------------------------------------------------ tour (SimBackend, headless)
class TourProbe:
    """Instrumentation for one headless tour: a 1 Hz sim-time sampler, an optional pacing task, per-step rows from
    the runner's ``/nandi/demo/events``."""

    def __init__(self, drv, pace: Optional[float] = None, sample_hz: float = 1.0):
        self.drv, self.pace = drv, pace
        self.samples: List[Tuple[float, float]] = []
        self.cpu: List[Tuple[float, float]] = []
        self.steps: Dict[str, dict] = {}
        self.slept = 0.0
        self.s0 = float(drv.t)
        self.w0 = time.perf_counter()
        self.c0 = time.process_time()
        self.phys0 = drv.wall_in_step
        self.pauses: List[Tuple[float, float]] = []
        drv.add_task("wp11_sample", self._sample, sample_hz)
        if pace:
            drv.add_task("wp11_pace", self._pace, PACE_HZ)
        self._sample(drv)

    def _sample(self, drv):
        self.samples.append((float(drv.t) - self.s0, time.perf_counter() - self.w0))
        self.cpu.append((float(drv.t) - self.s0, time.process_time() - self.c0))

    def _pace(self, drv):
        s = pace_sleep_s(float(drv.t) - self.s0, time.perf_counter() - self.w0 - self._paused_s(), self.pace)
        if s > 0:
            time.sleep(s)
            self.slept += s

    def _paused_s(self) -> float:
        return sum(b - a for a, b in self.pauses)

    def on_publish(self, topic: str, payload):
        if topic != "/nandi/demo/events" or not isinstance(payload, dict):
            return
        ev, sid = payload.get("event"), payload.get("step")
        now = (float(self.drv.t) - self.s0, time.perf_counter() - self.w0, time.process_time() - self.c0,
               self.drv.wall_in_step - self.phys0)
        if ev == "step_start" and sid:
            self.steps[sid] = {"start": now}
        elif ev == "step_end" and sid and sid in self.steps:
            s = self.steps[sid]["start"]
            ds, dw, dc, dp = (now[i] - s[i] for i in range(4))
            self.steps[sid].update({"sim_s": round(ds, 2), "wall_s": round(dw, 2), "cpu_s": round(dc, 2),
                                    "physics_wall_s": round(dp, 2), "status": payload.get("status"),
                                    "rtf": round(ds / dw, 3) if dw > 1e-6 else None})


def sec_tour(args) -> dict:
    from atharv_nandi.core import demo_tour as dt
    if args.segment:
        steps, actors = SEGMENTS[args.segment]
        actors = actors and not args.no_actors
    else:
        steps = [s for s in (args.steps or "").split(",") if s] or None
        actors = not args.no_actors
    prog = dt.load_program(args.program).select(steps)
    w_build = time.perf_counter()
    be = dt.SimBackend(prog.world, args.seed, transit=args.transit, capture=args.capture, actors=actors)
    build_wall = time.perf_counter() - w_build
    probe = TourProbe(be.drv, pace=args.pace)
    runs_dir = args.runs_dir or os.path.join(REPO, "runs", "wp11", "tours")
    try:
        runner = dt.TourRunner(prog, be, seed=args.seed, runs_dir=runs_dir, run_id=args.run_tag,
                               publish=probe.on_publish)
        res = runner.run()
        probe._sample(be.drv)
    finally:
        be.close()
    t = res["tour"]
    sim_s, wall_s = probe.samples[-1]
    cpu_s = probe.cpu[-1][1]
    wins = rtf_windows(probe.samples, args.window_s)
    ws = window_summary(wins, (args.pace - RTF_TOLERANCE) if args.pace else RTF_TARGET)
    rows = []
    for r in res["steps"]:
        p = probe.steps.get(r["id"], {})
        rows.append({"id": r["id"], "status": r["status"], "reason": r["reason"][:200], "sim_s": r["sim_s"],
                     "budget_s": r.get("budget_s"), "wall_s": p.get("wall_s"), "cpu_s": p.get("cpu_s"),
                     "physics_wall_s": p.get("physics_wall_s"), "rtf": p.get("rtf")})
    phys = be.drv.wall_in_step - probe.phys0
    return {"program": args.program, "segment": args.segment, "seed": args.seed, "world": prog.world,
            "transit": args.transit, "capture": args.capture, "actors": actors, "pace": args.pace,
            "build_wall_s": round(build_wall, 2), "sim_s": round(sim_s, 1), "wall_s": round(wall_s, 1),
            "cpu_s": round(cpu_s, 1), "physics_wall_s": round(phys, 1), "paced_sleep_s": round(probe.slept, 1),
            "headroom_rtf": round(sim_s / wall_s, 3) if wall_s else None,
            "rtf_realised": round(sim_s / wall_s, 3) if (args.pace and wall_s) else None,
            "cpu_cores_mean": round(cpu_s / wall_s, 3) if wall_s else None,
            "tour_pass": bool(t["pass"]), "fail_reasons": t.get("fail_reasons"), "safety": t.get("safety"),
            "tour_sim_s": t.get("sim_s"), "steps": rows, "windows": wins, "window_rtf": ws,
            "verdict": tour_verdict(sim_s, wall_s, ws["min"], bool(args.pace), pace=args.pace,
                                    tour_pass=bool(t["pass"]), complete=args.segment is None
                                    and steps is None and args.program == "orchard_tour")}


# ------------------------------------------------------------------ live (ROS stack + demo_tour_node)
def _proc_cpu() -> Dict[int, Tuple[str, float]]:
    hz = os.sysconf("SC_CLK_TCK")
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"/proc/{pid}/stat").read()
            cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()
        except OSError:
            continue
        f = stat[stat.rfind(")") + 2:].split()
        out[int(pid)] = (cmd, (int(f[11]) + int(f[12])) / hz)
    return out


def proc_name(cmd: str) -> str:
    """Process label: the script basename, or `<node>.py` for `python3 -m atharv_nandi.nodes <node>` (WPR).
    Same rule as atharv_nandi.launch_plan.proc_label (also the installed `lib/atharv_nandi/<node>`)."""
    from atharv_nandi.launch_plan import proc_label
    return proc_label(cmd)


def cpu_by_process(s0: dict, s1: dict, wall: float, self_pid: int) -> List[dict]:
    rows = []
    for pid, (cmd, cpu) in s1.items():
        if pid == self_pid or not cmd:
            continue
        cores = (cpu - s0.get(pid, (cmd, 0.0))[1]) / wall
        if cores >= 0.005:
            rows.append({"pid": pid, "proc": proc_name(cmd), "cores": round(cores, 3)})
    rows.sort(key=lambda r: -r["cores"])
    return rows


def sec_live(args) -> dict:
    xv = ensure_display()
    env = dict(os.environ, NANDI_HEADLESS="1", NANDI_WORLD=args.world, NANDI_SEED=str(args.seed),
               NANDI_NOVNC_UP="0", LP_NUM_THREADS="1")
    stack = subprocess.Popen(["bash", os.path.join(REPO, "launch", "launch_nandi.sh")], env=env, cwd=REPO,
                             stdout=open(os.path.join(args.log_dir, "live_stack.log"), "w"),
                             stderr=subprocess.STDOUT, start_new_session=True)
    tour = subprocess.Popen([sys.executable, "-m", "atharv_nandi.nodes", "demo_tour_node"], env=env, cwd=REPO,
                            stdout=open(os.path.join(args.log_dir, "live_tour.log"), "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    w_start = time.perf_counter()
    gs: List[Tuple[float, float, float]] = []      # (wall, sim t, rtf) from /nandi/game_status
    demo: List[dict] = []
    try:
        import rclpy
        from std_msgs.msg import String
        rclpy.init()
        node = rclpy.create_node("wp11_bench")

        def on_gs(msg):
            try:
                j = json.loads(msg.data)
            except ValueError:
                return
            gs.append((time.perf_counter() - w_start, float(j.get("elapsed_s") or 0.0),
                       float(j.get("rtf") or 0.0)))

        def on_demo(msg):
            try:
                j = json.loads(msg.data)
            except ValueError:
                return
            demo.append({"wall": round(time.perf_counter() - w_start, 2), "state": j.get("state"),
                         "current": j.get("current"), "t_sim": j.get("t_sim"), "pass": j.get("pass")})

        node.create_subscription(String, "/nandi/game_status", on_gs, 10)
        node.create_subscription(String, "/nandi/demo/state", on_demo, 10)
        pub = node.create_publisher(String, "/nandi/demo/run", 10)
        # wait for the paced sim (first game_status) and the tour node (its subscription on /nandi/demo/run)
        t_ready = None
        while time.perf_counter() - w_start < args.warmup_s:
            rclpy.spin_once(node, timeout_sec=0.2)
            if gs and pub.get_subscription_count() > 0:
                t_ready = time.perf_counter() - w_start
                break
        run = {"program": args.program, "seed": args.seed, "transit": "walk",
               "steps": [s for s in args.steps.split(",") if s] if args.steps else "full"}
        sent = None
        if t_ready is not None:
            time.sleep(1.0)
            pub.publish(String(data=json.dumps(run)))
            sent = time.perf_counter() - w_start
        s0, c0 = _proc_cpu(), time.perf_counter()
        while time.perf_counter() - w_start < (sent or args.warmup_s) + args.seconds:
            rclpy.spin_once(node, timeout_sec=0.2)
            if demo and demo[-1]["state"] in ("done", "error") and sent is not None:
                break
        s1, c1 = _proc_cpu(), time.perf_counter()
        node.destroy_node()
        rclpy.shutdown()
        procs = cpu_by_process(s0, s1, c1 - c0, os.getpid())
        after = [g for g in gs if sent is not None and g[0] >= sent]
        rtf_series = [g[2] for g in after]
        # tour progress: demo state t_sim vs wall since the run was sent
        prog = [(d["t_sim"], d["wall"]) for d in demo if d.get("t_sim") is not None and sent and d["wall"] >= sent]
        tour_headroom = None
        if len(prog) >= 2 and prog[-1][1] > prog[0][1]:
            tour_headroom = round((prog[-1][0] - prog[0][0]) / (prog[-1][1] - prog[0][1]), 3)
        sim_wins = rtf_windows([(g[1], g[0]) for g in after], args.window_s) if after else []
        res = {"world": args.world, "program": args.program, "steps": run["steps"], "warmup_wall_s": t_ready,
                "run_sent_wall_s": sent, "window_wall_s": round(c1 - c0, 1), "processes": procs,
                "total_cores": round(sum(r["cores"] for r in procs), 3),
                "game_status_rtf": stats(rtf_series), "sim_runner_windows": sim_wins,
                "sim_runner_window_rtf": window_summary(sim_wins, RTF_TARGET - RTF_TOLERANCE),
                "tour_progress": prog[-1] if prog else None, "tour_headroom_rtf": tour_headroom,
                "tour_last_state": demo[-1] if demo else None, "stack_alive": stack.poll() is None,
                "tour_node_alive": tour.poll() is None}
        # a live run that never started or lost a process is a failed section (non-zero exit), not empty data
        res["failures"] = live_failures(res)
        res["ok"] = not res["failures"]
        return res
    finally:
        for p in (tour, stack):
            try:
                os.killpg(p.pid, 15)
            except Exception:
                pass
        if xv:
            xv.terminate()


# ------------------------------------------------------------------ summary (pure)
def sec_summary(args) -> dict:
    docs = [json.load(open(p)) for f in args.inputs for p in sorted(glob.glob(f))]
    tours = [d for d in docs if d.get("section") == "tour"]
    lives = [d for d in docs if d.get("section") == "live"]
    out = {"inputs": [d["identity"]["run_id"] for d in docs]}
    if tours:
        mt = merge_tours(tours)
        ws = window_summary(mt["windows"],
                            ((mt["pace"] or RTF_TARGET) - RTF_TOLERANCE) if mt["paced"] else RTF_TARGET)
        covered = {s["id"] for s in mt["steps"]}
        try:
            from atharv_nandi.core import demo_tour as dt
            want = [s.id for s in dt.load_program("orchard_tour").steps]
        except Exception:
            want = sorted(covered)
        complete = set(want) <= covered and not mt["duplicate_steps"]
        live_min = None
        # every live doc must be a real measurement: a failed one fails the RTF gate instead of being dropped
        failed = [f"{x['identity']['run_id']}:{'+'.join(live_failures(x['result']))}" for x in lives
                  if live_failures(x["result"])]
        lv = [(x["result"].get("sim_runner_window_rtf") or {}).get("min") for x in lives]
        lv = [v for v in lv if v is not None]
        if lv:
            live_min = min(lv)
        out["tour"] = {k: v for k, v in mt.items() if k != "windows"}
        out["tour"]["missing_steps"] = [s for s in want if s not in covered]
        out["tour"]["window_rtf"] = ws
        out["verdict"] = tour_verdict(mt["sim_s"], mt["wall_s"], ws["min"], mt["paced"], live_rtf_min=live_min,
                                      complete=complete, pace=mt["pace"], tour_pass=mt["tour_pass_all_segments"],
                                      live_failed=failed)
    if lives:
        out["live"] = [{"run_id": x["identity"]["run_id"], "total_cores": x["result"]["total_cores"],
                        "sim_runner_window_rtf": x["result"].get("sim_runner_window_rtf"),
                        "tour_headroom_rtf": x["result"].get("tour_headroom_rtf"),
                        "failures": live_failures(x["result"])} for x in lives]
    return out


SECTIONS = {"world": sec_world, "capture": sec_capture, "tour": sec_tour, "live": sec_live, "summary": sec_summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Atharv Nandi WP11 orchard tour benchmark (run ids, run-identity rule)")
    ap.add_argument("section", choices=sorted(SECTIONS))
    ap.add_argument("inputs", nargs="*", help="summary: tour / live JSON files or globs")
    ap.add_argument("--out", help="also write the JSON here")
    ap.add_argument("--git", help="<hex sha>[-dirty] for the run id (the container cannot read a worktree .git)")
    ap.add_argument("--world", default="orchard_demo")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tree", default="OT-1", help="world / capture: the TREE 360 tree")
    ap.add_argument("--steps", default="", help="world: physics steps (int); tour / live: comma step ids")
    ap.add_argument("--driver-s", type=float, default=4.0, help="world: SimDriver standing seconds")
    ap.add_argument("--no-driver", action="store_true")
    ap.add_argument("--reps", type=int, default=1, help="capture: orbits")
    ap.add_argument("--no-cull", action="store_true")
    ap.add_argument("--burst", type=float, default=0.3, help="capture: RenderBudget burst (s)")
    ap.add_argument("--program", default="orchard_tour", help="tour / live: tour_lite | orchard_tour")
    ap.add_argument("--segment", choices=sorted(SEGMENTS), help="tour: one WP9 segment of orchard_tour")
    ap.add_argument("--pace", type=float, default=None, help="tour: wall-pace the driver at this RTF (e.g. 0.9)")
    ap.add_argument("--transit", default="walk", choices=["walk", "teleport"])
    ap.add_argument("--capture", default="geometry", choices=["geometry", "gl", "auto"])
    ap.add_argument("--no-actors", action="store_true")
    ap.add_argument("--window-s", type=float, default=WINDOW_SIM_S)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--seconds", type=float, default=100.0, help="live: measurement window after the run is sent")
    ap.add_argument("--warmup-s", type=float, default=60.0, help="live: max start-up wait")
    a = ap.parse_intermixed_args(argv)
    if a.section == "world":
        a.steps = int(a.steps or 2000)
    if a.git:
        check_git_arg(a.git)
    tag = a.segment or (a.world if a.section in ("world", "capture") else "")
    ident = identity(a.section, a.git, tag)
    a.run_tag = ident["run_id"]
    a.log_dir = os.path.dirname(os.path.abspath(a.out)) if a.out else os.path.join(REPO, "runs", "wp11")
    os.makedirs(a.log_dir, exist_ok=True)
    w0 = time.perf_counter()
    res = SECTIONS[a.section](a)
    doc = {"identity": ident, "section": a.section, "wall_s": round(time.perf_counter() - w0, 1), "result": res}
    txt = json.dumps(doc, indent=1, default=str)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(txt)
    brief = {k: v for k, v in res.items() if k not in ("rows", "windows", "steps", "sim_runner_windows")}
    print(json.dumps({"run_id": ident["run_id"], "wall_s": doc["wall_s"], **brief}, indent=1, default=str))
    if a.section == "live" and not res.get("ok"):
        print(f"live: FAILED {res.get('failures')} (JSON written; it fails the summary RTF gate)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
