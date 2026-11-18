"""In-process executor of the WP8 task state machines on a ``SimDriver`` (headless, ROS-free; MuJoCo).

It binds the pure SMs (``clip_pedicel``, ``twist_pick``, ``prune_cut``, ``banana_tidy``) to

* WP2 ``hooks.tools`` ``ToolRuntime`` (mount, blade / jaw / roll actuation, the cut rule and the twist release, events),
* WP3 ``hooks.transfer`` ``TransferController`` (stalk-down cell / papaya cradle placement from the PARK catcher),
* WPCAP ``controllers.planning.ArmPlanner`` (OMPL RRT-Connect, collision-checked transits and parks with the mounted
  tool and the held fruit carried by the wrist),
* ``arm_ik_solver`` (tool-aware TCP closed form, mast keep-out, gravity feed-forward with the tool and fruit masses at
  their own CoMs, shoulder torque, predicted CoM),

and closes two loops the spec asks for: a **50 Hz TCP servo** on the tracked target (the wrist-camera stand-in;
assisted: the target pose is truth) and the **base pose loop** (OS 2.1: drift > 2 cm during a dwell pauses the arm
and re-approaches). Driven by ``TreeRunner.run_tree`` (one tree, one preset) through the MP 0.4 ``TargetLoop``.

Assisted register entries this executor adds (``ASSISTED``): ``truth_target_pose`` (servo target and stem estimate
from the pool / tree truth), ``truth_base_pose`` (base pose loop on the true trunk pose), ``station_teleport``
(``teleport_station``: the robot is put at the WP0c station instead of walking there; WP9 / WP25 own the walk).
"""
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import mujoco
import numpy as np

from atharv_nandi.core import harvest_logic as hl
from atharv_nandi.core.task_retry import RetryPolicy, TargetLoop
from atharv_nandi.core.tasks import banana_tidy
from atharv_nandi.core.tasks.clip_pedicel import ClipPedicelSM
from atharv_nandi.core.tasks.common import BLADE_CLEAN_S, Cmd, Obs, Target
from atharv_nandi.core.tasks.prune_cut import PruneCutSM
from atharv_nandi.core.tasks.twist_pick import TwistPickSM
from atharv_nandi.core.tools import tool_changer
from atharv_nandi.sim.world.modules import fruit_pool as pool_mod

ASSISTED = ("truth_target_pose", "truth_base_pose")
CTRL_DT = 0.02
REST_Q = np.array(hl.REST_ARM[:5], dtype=float)
SERVO_RATES = np.array([0.25, 1.2, 1.2, 1.2, 1.5])       # rad/s (gantry m/s) servo slew limits (assumption)
APPROACH_MPS = 0.08                                       # along-axis approach / retreat speed
SERVO_GAIN = 0.06                                         # per-tick integral gain on the TCP error
SERVO_CORR_MAX = 0.03
BASE_SETTLE_MPS = 0.004
JAW_CLEAR_M = 0.002
LOWER_MPS = 0.10                                          # slip response: lower speed (assumption)
LOWER_DZ_M = 0.35                                         # lower at most this far
LOWER_CLEAR_M = 0.20                                      # stop this high above the ground / hull under the TCP
LOWER_RAY_BELOW_M = 0.15                                  # the floor ray starts this far below the TCP
STABILITY_MIN_MARGIN_M = tool_changer.MIN_MARGIN_M        # the WP2 blade-close gate margin (one source)
SM_CLASSES = {"clip_pedicel": ClipPedicelSM, "twist_pick": TwistPickSM, "prune_cut": PruneCutSM,
              "banana_tidy": banana_tidy.BananaTidySM}


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


class SimTaskExecutor:
    """Runs one task SM to completion on a SimDriver with WP2 installed (``hooks.tools.install``) and the WP3
    transfer hooks loaded. ``publish(topic, payload)`` (optional) receives ``/nandi/mark`` and ``/nandi/task/state``."""

    def __init__(self, drv, rt, transfer=None, planner=None, publish: Optional[Callable[[str, Any], None]] = None,
                 values: Optional[dict] = None):
        from atharv_nandi.sim.hooks import transfer as transfer_hook
        self.drv, self.rt, self.m, self.d = drv, rt, drv.m, drv.d
        self.ik = drv.ik
        self.tc = transfer if transfer is not None else transfer_hook.controller_for(self.m)
        self._transfer_rt = transfer_hook.runtime_for(self.m)
        if planner is None:
            from atharv_nandi.core.planning import ArmPlanner
            planner = ArmPlanner.from_config(self.m)
        self.planner = planner
        self.publish = publish
        self.values = values if values is not None else drv.bus.values
        self.q_cmd = np.array(self.d.qpos[self.ik.qadr], dtype=float)
        self.events_seen = len(rt.events)
        self.log: List[dict] = []
        self.plan_stats: List[dict] = []
        self.max_tcp_mps = 0.0
        self._tcp_speeds: List[float] = []
        self._speed_by_state: Dict[str, float] = {}
        self._cur_state = ""
        self.ref_pose: Optional[np.ndarray] = None
        self.person_fn: Callable[[], bool] = lambda: bool((self.values.get("work_field") or {}).get("person"))
        self.estop_fn: Callable[[], bool] = lambda: bool(drv.holding)
        self.min_margin_m = 0.03
        self.paused_s = 0.0                            # sim time trajectories stood frozen for a person (park, carry)
        self.trace: Optional[List[dict]] = None        # set to [] to record one row per tick (debugging, probes)

    # ================================================================== geometry / state
    def trunk(self):
        return self.ik.trunk_pose(self.d.qpos)

    def tcp_site(self, tool: str) -> int:
        return self.rt.h[tool].tcp

    def tcp(self, tool: str) -> np.ndarray:
        return np.array(self.d.site_xpos[self.tcp_site(tool)])

    def base_pose(self) -> np.ndarray:
        q = self.d.qpos
        yaw = math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2))
        return np.array([q[0], q[1], yaw])

    def base_drift(self) -> float:
        if self.ref_pose is None:
            return 0.0
        p = self.base_pose()
        dyaw = (p[2] - self.ref_pose[2] + math.pi) % (2 * math.pi) - math.pi
        return float(math.hypot(p[0] - self.ref_pose[0], p[1] - self.ref_pose[1]) + 0.3 * abs(dyaw))

    def base_speed(self) -> float:
        return float(np.linalg.norm(self.d.qvel[0:2]))

    def loads(self, tool: Optional[str]) -> list:
        """[(kg, point in the wrist frame)]: the mounted tool at its subtree CoM, the caught fruit at its body."""
        if tool is None:
            return []
        m, d = self.m, self.d
        wb = self.ik.wrist_id
        wp, wr = np.array(d.xpos[wb]), np.array(d.xmat[wb]).reshape(3, 3)
        out = []
        body = self.rt.h[tool].body
        out.append((float(m.body_subtreemass[body]), wr.T @ (np.array(d.subtree_com[body]) - wp)))
        if self.rt.caught:
            s = self.rt.pool.slots[self.rt.caught["slot"]]
            out.append((float(m.body_mass[s.body]), wr.T @ (np.array(d.xpos[s.body]) - wp)))
        return out

    def expected_loads(self, tool: str, fruit_kg: float) -> list:
        """Loads for the precheck: the tool now plus ``fruit_kg`` at the TCP."""
        out = self.loads(tool)[:1]
        if fruit_kg > 0:
            out.append((float(fruit_kg), np.array([self.ik.tool_tcp_m(tool), 0.0, 0.0])))
        return out

    # ================================================================== arm output
    def _send(self, q5, tool: Optional[str]):
        q5 = np.clip(np.asarray(q5, float), self.ik.limits[:, 0], self.ik.limits[:, 1])
        self.q_cmd = q5
        off = self.ik.gravity_offsets(self.d.qpos, q5, 0.0, loads=self.loads(tool))
        self.drv.arm = list(q5 + off) + list(self.drv.arm[5:7])

    def wait_tick(self, tool: Optional[str], prev_tcp=None):
        self.drv.wait(CTRL_DT)
        if tool is not None and prev_tcp is not None:
            v = float(np.linalg.norm(self.tcp(tool) - prev_tcp)) / CTRL_DT
            self._tcp_speeds.append(v)
            self._speed_by_state[self._cur_state] = max(self._speed_by_state.get(self._cur_state, 0.0), v)

    # ================================================================== tool helpers
    def actuate(self, req: dict) -> bool:
        return self.rt.actuate(req, hold=bool(self.drv.holding), values=self.values)

    def tool_open(self):
        self.actuate({"open": True})

    def tool_close(self, tool: str):
        self.actuate({"value": 0.0 if tool == "T4" else 1.0})

    def mount(self, tool: str, timeout_s: float = 12.0) -> bool:
        if self.rt.changer.mounted == tool:
            return True
        if not self.rt.request_change(tool, hold=bool(self.drv.holding), values=self.values):
            return False
        self.drv.run_until(lambda dr: not self.rt.changer.busy, timeout_s)
        self.q_cmd = np.array(self.rt.plant.q_home, float)
        return self.rt.changer.mounted == tool

    def blade_clean(self, seconds: float = BLADE_CLEAN_S):
        """OS 2.4 tool hygiene step (5 s, budgeted): tool open, arm parked."""
        self.tool_open()
        self.drv.wait(seconds)

    # ================================================================== stem / roll
    def hinge_axis(self, tool: str) -> np.ndarray:
        ids = self.rt.h[tool]
        jid = int(self.m.dof_jntid[ids.act_dadr])
        return _unit(self.d.xaxis[jid])

    def roll_axis(self, tool: str) -> np.ndarray:
        jid = int(self.m.dof_jntid[self.rt.h[tool].roll_dadr])
        return _unit(self.d.xaxis[jid])

    def roll_now(self, tool: str) -> float:
        return float(self.d.qpos[self.rt.h[tool].roll_qadr])

    def aligned_roll(self, tool: str, stem_dir) -> float:
        """Tool roll (rad) that lays the blade hinge along ``stem_dir`` (cut rule: stem line vs hinge < 30 deg)."""
        a = self.roll_axis(tool)
        h = self.hinge_axis(tool)
        s = _unit(np.asarray(stem_dir, float))
        s = _unit(s - a * float(a @ s))
        if float(np.linalg.norm(s)) < 1e-6:
            return self.roll_now(tool)
        delta = math.atan2(float(np.cross(h, s) @ a), float(h @ s))
        if delta > math.pi / 2:
            delta -= math.pi
        elif delta < -math.pi / 2:
            delta += math.pi
        lo, hi = self.rt.h[tool].spec.roll_range
        return float(np.clip(self.roll_now(tool) + delta, lo, hi))

    # ================================================================== targets
    def target_fns(self, tg: Target):
        """(aim_fn, stem_dir_fn, body_name) for the servo: the TCP aim point and the stem line in the world."""
        rt, d = self.rt, self.d
        if tg.task in ("clip_pedicel", "twist_pick"):
            k = rt.pool.slot_of(tg.meta["fruit_id"])
            s = rt.pool.slots[k]
            body = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, s.body)
            stem = lambda: np.array(d.site_xpos[s.anchor]) - np.array(d.site_xpos[s.stem_end])  # noqa: E731
            if tg.task == "clip_pedicel":
                return (lambda: np.array(d.site_xpos[s.cut_k])), stem, body
            head = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "twist_head")
            from atharv_nandi.core.tools import registry as reg

            def aim():
                # the fixed jaw's inner face JAW_CLEAR_M off the fruit's side: the fruit's support half-width along the
                # jaw normal (head y) at the live roll (ellipsoid support function: a rolled jaw meets the long axis)
                y = np.array(d.xmat[head]).reshape(3, 3)[:, 1]
                u = np.array(d.xmat[s.body]).reshape(3, 3).T @ y
                hw = float(np.sqrt(np.sum((np.asarray(self.m.geom_size[s.fruit], float) * u) ** 2)))
                return np.array(d.xpos[s.body]) - y * (reg.JAW_FIXED_Y - hw - JAW_CLEAR_M)
            return aim, stem, body
        a, b = np.asarray(tg.meta.get("a"), float), np.asarray(tg.meta.get("b"), float)
        off = np.asarray(tg.meta["tree"]["pos"], float)
        cut = np.asarray(tg.point, float)
        return (lambda: cut), (lambda: (b + off) - (a + off)), tg.meta.get("geom")

    def approach_pitch(self, tg: Target, stem_dir) -> float:
        """Tool axis pitch below horizontal. T4: along the pedicel (twist about the stalk: papaya radial and level,
        guava hanging -> tool down); T2 / T5: the WP0c frozen approach pitch."""
        if tg.task == "twist_pick":
            p = _unit(stem_dir)
            return math.pi / 2 if abs(p[2]) > 0.7 else 0.0
        return self.cut_pitch(tg, stem_dir)

    CUT_AXIS_COS_MAX = math.sin(math.radians(20.0))     # tool axis within 20 deg of perpendicular to the stem

    def cut_pitch(self, tg: Target, stem_dir) -> float:
        """T2 / T5: the blade hinge is perpendicular to the tool axis, so the roll can lay it along the stem only if
        the tool axis is near perpendicular to the stem (cut rule < 30 deg). The arm is planar: the axis yaw is the
        shoulder -> target bearing and only the pitch is free. Keep the WP0c frozen pitch when it is perpendicular
        enough (pedicels hang vertical: a level approach), else the nearest pitch in [0, 90 deg] that is (a radial
        skirt shoot or a sprout: from above)."""
        _p, R = self.trunk()
        t_pos = _p
        s0 = t_pos + R @ self.ik.shoulder0
        u = np.asarray(tg.point, float)[:2] - s0[:2]
        u = _unit(u)
        s = _unit(stem_dir)

        def cosang(p):
            ax = np.array([u[0] * math.cos(p), u[1] * math.cos(p), -math.sin(p)])
            return abs(float(ax @ s))

        p0 = float(tg.pitch)
        if cosang(p0) <= self.CUT_AXIS_COS_MAX:
            return p0
        cands = [math.radians(a) for a in range(0, 91, 5)]
        ok = [p for p in cands if cosang(p) <= self.CUT_AXIS_COS_MAX]
        if ok:
            return min(ok, key=lambda p: abs(p - p0))
        return min(cands, key=cosang)

    def reachable(self, tg: Target) -> bool:
        """The target's at-pose and pre-pose have a closed-form solution from the live base pose (station check)."""
        try:
            point = np.asarray(tg.point, float)
            if tg.task in ("clip_pedicel", "twist_pick") and self.rt.pool.slot_of(tg.meta["fruit_id"]) is not None:
                aim_fn, stem_fn, _b = self.target_fns(tg)
                point, stem = aim_fn(), stem_fn()
            else:
                stem = (np.asarray(tg.meta.get("b"), float) - np.asarray(tg.meta.get("a"), float)) \
                    if tg.meta.get("a") is not None else np.array(tg.meta.get("pedicel") or (0.0, 0.0, 1.0), float)
            pitch = self.approach_pitch(tg, stem)
            q, ax = self.ik.tcp_solve(self.d.qpos, point, tg.tool, pitch, gantry=None)
            if q is None:
                return False
            pre = 0.15 if tg.tool == "T4" else 0.10
            q2, _ = self.ik.tcp_solve(self.d.qpos, point - pre * ax, tg.tool, pitch, gantry=float(q[0]))
            return q2 is not None
        except Exception:  # noqa: BLE001 - a malformed target is simply not reachable
            return False

    # ================================================================== motions
    def _plan_to(self, q_goal, tool: str, ignore=(), attach_fruit: bool = False, max_tcp_mps: Optional[float] = None):
        attach = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.rt.h[tool].body)]
        if attach_fruit and self.rt.caught:
            attach.append(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                            self.rt.pool.slots[self.rt.caught["slot"]].body))
        q_start = np.array(self.d.qpos[self.ik.qadr], float)
        res = self.planner.plan(None, q_goal, start_q=q_start, ignore=[i for i in ignore if i], attach=attach,
                                model=self.m, data=self.d)
        if not res.ok and res.reason == "start_invalid":
            # the arm is already touching something (a fruit at the trap): plan from a start that ignores fruit
            res = self.planner.plan(None, q_goal, start_q=q_start, ignore=[i for i in ignore if i],
                                    ignore_classes=("fruit",), attach=attach, model=self.m, data=self.d)
        self.plan_stats.append({"ok": res.ok, "reason": res.reason, "backend": res.backend,
                                "plan_s": round(res.plan_time_s, 3), "n": 0 if res.path is None else len(res.path),
                                "clearance": res.min_clearance})
        if not res.ok:
            return None
        traj = res.trajectory
        scale = 1.0
        if max_tcp_mps:
            tips = [self.tool_tcp_at(traj.sample(t), tool) for t in np.arange(0.0, traj.duration + 1e-9, 0.02)]
            v = max([float(np.linalg.norm(b - a)) / 0.02 for a, b in zip(tips, tips[1:])] or [0.0])
            scale = max(1.0, v / (0.8 * max_tcp_mps))
        return traj, scale

    def tool_tcp_at(self, q5, tool: str) -> np.ndarray:
        """World TCP of the mounted tool with the arm at q5 (live base pose)."""
        self.ik._fk(self.d.qpos, q5)
        wp, wr = self.ik._wrist_frame()
        return wp + wr[:, 0] * self.ik.tool_tcp_m(tool)

    def _run_traj(self, traj, scale, tool, abort_on_person: bool = False, deadline: Optional[float] = None) -> str:
        """Play a planned trajectory at 50 Hz. Returns "" when it ran to the end, else why it stopped: ``estop``
        (always), ``person`` (``abort_on_person``: the pre-commit transit; the SM pauses and re-plans), ``timeout``
        (``deadline``). Without ``abort_on_person`` a person in the WORK field **freezes** the trajectory where it is
        (the arm holds, trajectory time stops) until the field clears (MP 0.3 / WORKFLOWS A.0, WP2 FREEZE_INTERLOCKS):
        the park and the carry keep the fruit, only the motion pauses."""
        t0 = self.drv.t
        dur = traj.duration * scale
        prev = self.tcp(tool)
        paused = 0.0
        while True:
            s = min(dur, self.drv.t - t0 - paused)
            self._send(traj.sample(s / scale), tool)
            self.wait_tick(tool, prev)
            prev = self.tcp(tool)
            if self.estop_fn():
                return "estop"
            if self.person_fn():
                if abort_on_person:
                    return "person"                     # q_cmd = the last sample: the caller holds the arm there
                while self.person_fn() and not self.estop_fn() and (deadline is None or self.drv.t < deadline):
                    t_hold = self.drv.t
                    self._send(self.q_cmd, tool)            # frozen: trajectory time stops
                    self.wait_tick(tool)
                    paused += self.drv.t - t_hold
                    self.paused_s += self.drv.t - t_hold
                prev = self.tcp(tool)
                if self.estop_fn():
                    return "estop"
            if deadline is not None and self.drv.t >= deadline:
                return "timeout"
            if s >= dur:
                break
        for _ in range(int(0.3 / CTRL_DT)):
            self._send(traj.sample(traj.duration), tool)
            self.wait_tick(tool)
        return ""

    # ================================================================== the SM loop
    def run(self, sm, timeout_s: float = 90.0) -> dict:
        """Run one task SM to its end. Returns ``sm.summary()`` plus executor measurements."""
        tg: Target = sm.target
        tool = tg.tool
        rt, d = self.rt, self.d
        self._tcp_speeds: List[float] = []
        self._speed_by_state: Dict[str, float] = {}
        live_aim, stem_fn, body = self.target_fns(tg)
        frozen = {"p_t": None}

        def aim_fn():
            """The tracked target until the commit point; after the cut / release the fruit rides on the tool, so the
            aim is frozen where it was (trunk frame: it still follows the base)."""
            tp, tR = self.trunk()
            if frozen["p_t"] is not None:
                return tp + tR @ frozen["p_t"]
            return live_aim()

        def freeze():
            if frozen["p_t"] is None:
                tp, tR = self.trunk()
                frozen["p_t"] = tR.T @ (live_aim() - tp)
        ignore = [body] if body else []
        if tg.task in ("clip_pedicel", "twist_pick"):
            k = rt.pool.slot_of(tg.meta["fruit_id"])
            ignore.append(mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, rt.pool.slots[k].pedicel))
        pitch = self.approach_pitch(tg, stem_fn())
        self.ref_pose = self.base_pose()
        t_end = self.drv.t + timeout_s
        mode, s_cur, s_goal, corr, axis_t = "none", 0.0, 0.0, np.zeros(3), None
        aligned_roll = self.roll_now(tool)
        roll_target, roll_rate = aligned_roll, None
        roll_travel, roll_prev, gripping = 0.0, self.roll_now(tool), False
        arm_done, arm_failed = False, ""
        placed: Optional[str] = None
        place_fid = None
        cell = None
        precheck = self._precheck(tg, aim_fn(), pitch)
        cmd = Cmd()
        settle_t = None
        gantry = 0.0
        person_seen = False           # a transit was aborted for a person: the SM sees it on the next tick
        lower = None                  # arm="lower": {"xy", "z", "z_goal", "pitch", "t_settle"}
        wall0 = time.perf_counter()
        prev_tcp = self.tcp(tool)
        while not sm.done and self.drv.t < t_end:
            events = rt.events[self.events_seen:]
            self.events_seen = len(rt.events)
            ta, tb = (0.0, 0.0)
            if tool == "T4":
                from atharv_nandi.core.tools import grasp_sense
                ta, tb = (grasp_sense.touch(d, a) for a in rt.h["T4"].touch_adr)
            touch = min(ta, tb)
            r_now = self.roll_now(tool)
            if tool == "T4":
                if touch >= 0.5 and not gripping:
                    gripping, roll_prev = True, r_now
                if gripping:
                    roll_travel += abs(math.degrees(r_now - roll_prev))
                    roll_prev = r_now
            tcp = self.tcp(tool)
            err = math.inf
            if axis_t is not None:
                _p, R = self.trunk()
                aim = aim_fn() + s_cur * (R @ axis_t)
                err = float(np.linalg.norm(aim - tcp))
            if placed is None and place_fid is not None and not self.tc.busy():
                it = self.tc.items.get(place_fid)
                placed = "seated" if it is not None and it.get("state") == "seated" else "refused:place_failed"
                cell = it.get("where") if it else None
            if cell is not None:
                precheck["cell"] = cell
            person_now = self.person_fn() or person_seen
            person_seen = False
            obs = Obs(t=self.drv.t, arm_done=arm_done, arm_failed=arm_failed, tcp_err_m=err,
                      base_drift_m=self.base_drift(), base_settled=self.base_speed() < BASE_SETTLE_MPS,
                      estop=self.estop_fn(), person=person_now, stability_ok=self.stability_ok(),
                      events=list(events), caught=bool(rt.caught),
                      touch_n=touch, touch_max_n=max(ta, tb), roll_travel_deg=roll_travel, roll_deg=math.degrees(r_now), placed=placed,
                      precheck=precheck)
            if sm.committed or rt.caught:
                freeze()
            cmd = sm.step(obs)
            self._cur_state = sm.state
            if sm.committed:
                freeze()
            if self.trace is not None:
                self.trace.append(self._trace_row(tg, obs, sm, aim_fn))
            # ---- tool
            if cmd.tool == "open":
                self.tool_open()
                gripping, roll_travel = False, 0.0
            elif cmd.tool == "close":
                self.tool_close(tool)
            elif cmd.tool == "stop":
                rt.act.stop()
                rt.act.resume()
            if cmd.roll_deg is not None:
                if tool in ("T2", "T5"):
                    roll_target = aligned_roll + math.radians(cmd.roll_deg)
                else:
                    roll_target = math.radians(cmd.roll_deg)
                roll_rate = cmd.arm_args.get("roll_rate_dps") if cmd.arm_args else None
            if roll_rate:
                step = math.radians(roll_rate) * CTRL_DT
                cur = float(d.ctrl[rt.h[tool].roll_act])
                self.actuate({"roll": cur + float(np.clip(roll_target - cur, -step, step))})
            elif cmd.roll_deg is not None:
                self.actuate({"roll": roll_target})
            if cmd.rebase:
                self.ref_pose = self.base_pose()
            if cmd.place:
                place_fid, placed = self._start_place(), None
                if place_fid is None:
                    placed = "refused:" + (self._place_refusal or "place_failed")
            if cmd.done:
                break
            # ---- arm
            arm_moved = cmd.arm in ("plan_pre", "park")
            if cmd.arm is not None and cmd.arm != "none":
                arm_done, arm_failed = False, ""
                if cmd.arm == "plan_pre":
                    standoff = float(cmd.arm_args.get("standoff_m", 0.1))
                    q_at, ax = self.ik.tcp_solve(d.qpos, aim_fn(), tool, pitch, gantry=None)
                    if q_at is None:
                        arm_failed = "unreachable"
                    else:
                        gantry = float(q_at[0])
                        _p, R = self.trunk()
                        axis_t = R.T @ ax
                        q_pre, _ = self.ik.tcp_solve(d.qpos, aim_fn() - standoff * ax, tool, pitch, gantry=gantry)
                        out = None if q_pre is None else self._plan_to(q_pre, tool, ignore=ignore)
                        if out is None:
                            arm_failed = "plan_failed"
                        else:
                            why = self._run_traj(out[0], out[1], tool, abort_on_person=True, deadline=t_end)
                            s_cur = s_goal = -standoff
                            corr = np.zeros(3)
                            if tool in ("T2", "T5"):
                                aligned_roll = self.aligned_roll(tool, stem_fn())
                                roll_target = aligned_roll
                            arm_done = not why
                            arm_failed = "" if why in ("person", "timeout") else why
                            if why:
                                # stopped short of the pre-pose: hold there; a person -> the SM pauses (person_hold)
                                # and re-plans from wherever the arm stopped once the field is clear
                                person_seen = why == "person"
                                axis_t = None
                    mode = "servo" if axis_t is not None and arm_done else "hold"
                    if tool in ("T2", "T5") and cmd.roll_deg is None:
                        roll_target = aligned_roll
                elif cmd.arm in ("approach", "retreat"):
                    s_goal, settle_t = float(cmd.arm_args.get("s", 0.0)), None
                    mode = "servo"
                    if axis_t is None:
                        # no approach axis yet (the transit never reached the pre-pose): nothing to back out of
                        mode, arm_done = "hold", True
                    self._servo_v = APPROACH_MPS if cmd.arm == "approach" else min(
                        APPROACH_MPS, float(cmd.arm_args.get("max_tcp_mps") or APPROACH_MPS))
                elif cmd.arm == "servo":
                    s_goal, mode = 0.0, "servo"
                    arm_done = True
                elif cmd.arm == "park":
                    q_park = REST_Q.copy()
                    out = self._plan_to(q_park, tool, attach_fruit=True, max_tcp_mps=cmd.arm_args.get("max_tcp_mps"))
                    if out is None:
                        out = self._direct_traj(q_park)
                    why = self._run_traj(out[0], out[1], tool, deadline=t_end)   # a person freezes, not aborts
                    mode, axis_t = "hold", None
                    arm_done, arm_failed = not why, ("" if why == "timeout" else why)
                elif cmd.arm == "lower":
                    lower = self._lower_start(tool)
                    mode = "lower"
                    arm_done = lower is None
                elif cmd.arm == "hold":
                    mode = "hold"
                    arm_done = True
            # ---- servo tick
            if mode in ("servo", "lower") and self.person_fn():
                # a person in the WORK field freezes every arm motion the SM does not pause itself (the carry
                # retreat after the commit, the skip retreat, the slip lower); the motion resumes once it clears
                self._send(self.q_cmd, tool)
                settle_t = None
                if lower is not None:
                    lower["t_settle"] = None
            elif mode == "lower" and lower is not None:
                if self._lower_tick(tool, lower):
                    arm_done, mode, lower = True, "hold", None
            elif mode == "servo" and axis_t is not None:
                v = getattr(self, "_servo_v", APPROACH_MPS)
                ds = float(np.clip(s_goal - s_cur, -v * CTRL_DT, v * CTRL_DT))
                s_cur += ds
                _p, R = self.trunk()
                ax_w = R @ axis_t
                aim = aim_fn() + s_cur * ax_w
                e = aim - self.tcp(tool)
                if abs(s_goal - s_cur) < 1e-9:
                    corr = np.clip(corr + SERVO_GAIN * e, -SERVO_CORR_MAX, SERVO_CORR_MAX)
                q_goal, _ = self.ik.tcp_solve(d.qpos, aim + corr, tool, pitch, gantry=gantry)
                if q_goal is not None:
                    step = SERVO_RATES * CTRL_DT
                    self._send(self.q_cmd + np.clip(q_goal - self.q_cmd, -step, step), tool)
                if abs(s_goal - s_cur) < 1e-9:
                    if settle_t is None:
                        settle_t = self.drv.t
                    if not arm_done and (float(np.linalg.norm(e)) < 0.004 or self.drv.t - settle_t > 1.0):
                        arm_done = True
                else:
                    settle_t = None
            elif mode == "hold":
                self._send(self.q_cmd, tool)
            if arm_moved:
                prev_tcp = self.tcp(tool)
            self.wait_tick(tool, prev_tcp)
            prev_tcp = self.tcp(tool)
        timed_out = not sm.done
        cleanup = self._timeout_cleanup(sm, tool) if timed_out else None
        out = sm.summary()
        out.update({"wall_s": round(time.perf_counter() - wall0, 2), "sim_s": round(self.drv.t - (sm.t0 or 0.0), 2),
                    "max_tcp_mps": round(max(self._tcp_speeds or [0.0]), 3),
                    "tcp_mps_by_state": {k: round(v, 3) for k, v in self._speed_by_state.items()}, "precheck": precheck,
                    "timeout": timed_out, "committed": bool(sm.committed)})
        if cleanup is not None:
            out["cleanup"] = cleanup
        for mk in out["marks"]:
            self._emit_mark(mk)
        self.log.append(out)
        return out

    # ================================================================== safety helpers
    def stability_ok(self) -> bool:
        """The live stability interlock, from the same source as the WP2 blade-close gate (``values["stability"]``
        ``{ok, latched, degenerate, margin_m}``, the stability monitor's output): see ``tool_changer.stability_ok`` (a
        latched or degenerate monitor, a margin below 0.03 m or a margin published as None = non-finite refuse)."""
        return tool_changer.stability_ok(self.values)       # one rule with the WP2 close gate (WPCOV review)

    def _lower_start(self, tool: str) -> Optional[dict]:
        """OS 5.3 slip response: lower the TCP straight down (tool pitch and gantry kept) to LOWER_CLEAR_M above
        what is under it (ground, or the hull at PARK), at most LOWER_DZ_M. None if already that low."""
        tcp = self.tcp(tool)
        ax = np.array(self.d.xmat[self.ik.wrist_id]).reshape(3, 3)[:, 0]          # tool axis = wrist x
        pitch = -math.asin(max(-1.0, min(1.0, float(ax[2]))))
        # cast from below the held fruit (papaya half-length ~0.12 m) so the ray does not hit the tool or the fruit
        floor = ground_z(self.m, self.d, float(tcp[0]), float(tcp[1]), z0=float(tcp[2]) - LOWER_RAY_BELOW_M)
        z_goal = max(float(tcp[2]) - LOWER_DZ_M, floor + LOWER_CLEAR_M)
        if float(tcp[2]) <= z_goal + 0.005:
            return None
        return {"xy": tcp[:2].copy(), "z": float(tcp[2]), "z_goal": z_goal, "pitch": pitch,
                "gantry": float(self.d.qpos[self.ik.qadr[0]]), "t_settle": None}

    def _lower_tick(self, tool: str, lw: dict) -> bool:
        """One 20 ms step of the lower; True when done (reached, or the arm cannot go lower)."""
        lw["z"] = max(lw["z_goal"], lw["z"] - LOWER_MPS * CTRL_DT)
        aim = np.array([lw["xy"][0], lw["xy"][1], lw["z"]])
        q_goal, _ = self.ik.tcp_solve(self.d.qpos, aim, tool, lw["pitch"], gantry=lw["gantry"])
        if q_goal is None:
            self._send(self.q_cmd, tool)
            return True                                   # as low as the arm reaches here
        step = SERVO_RATES * CTRL_DT
        self._send(self.q_cmd + np.clip(q_goal - self.q_cmd, -step, step), tool)
        if lw["z"] <= lw["z_goal"] + 1e-9:
            if lw["t_settle"] is None:
                lw["t_settle"] = self.drv.t
            return abs(float(self.tcp(tool)[2]) - lw["z_goal"]) < 0.01 or self.drv.t - lw["t_settle"] > 1.0
        return False

    def _timeout_cleanup(self, sm, tool: str, budget_s: float = 30.0) -> dict:
        """The SM did not finish in ``timeout_s``: make the arm safe before the next target / station. A caught
        fruit is parked and placed (release_safe: into the tray / cradle, never opened over the ground); otherwise
        the tool opens and the arm parks. The SM ends ``failed`` / ``timeout`` (never retried after the commit)."""
        rt = self.rt
        rep: Dict[str, Any] = {"state": sm.state, "caught": bool(rt.caught)}
        if self.estop_fn():
            rep["estop"] = True
            sm.finish(self.drv.t, "halted", "estop")
            return rep
        deadline = self.drv.t + budget_s
        if not rt.caught:
            self.tool_open()
        out = self._plan_to(REST_Q.copy(), tool, attach_fruit=True)
        if out is None:
            out = self._direct_traj(REST_Q.copy())
        why = self._run_traj(out[0], out[1], tool, deadline=deadline)
        rep["parked"] = not why
        if rt.caught and not why:
            fid = self._start_place()
            if fid is None:
                rep["placed"] = "refused:" + (self._place_refusal or "place_failed")
            else:
                self.drv.run_until(lambda dr: not self.tc.busy(), max(0.5, deadline - self.drv.t))
                it = self.tc.items.get(fid) or {}
                rep["placed"] = "seated" if it.get("state") == "seated" else "refused:place_failed"
            if rep["placed"] == "seated":
                self.tool_open()
        sm.finish(self.drv.t, "failed", "timeout")
        return rep

    _place_refusal = ""

    def _start_place(self) -> Optional[str]:
        """Release the catch at PARK and start the WP3 transfer; returns the fruit id or None (refused)."""
        rt = self.rt
        self._place_refusal = ""
        if not rt.caught:
            self._place_refusal = "nothing_caught"
            return None
        slot = rt.caught["slot"]
        fid = rt.pool.assignments[slot]["fruit_id"]
        inputs = self._transfer_rt.inputs(self.d, self.drv.bus)
        if not (inputs.arm_parked and inputs.gait_stopped):
            # wait a little for the PARK servo to settle (FF + servo lag) and for the base to stop rocking after the
            # park swing (the WP3 interlock refuses the hand-over with gait_moving otherwise: a dropped fruit)
            for _ in range(50):
                self._send(self.q_cmd, rt.changer.mounted)
                self.drv.wait(CTRL_DT)
                inputs = self._transfer_rt.inputs(self.d, self.drv.bus)
                if inputs.arm_parked and inputs.gait_stopped:
                    break
        rt.release_catch()
        g = self.tc.place(self.d, fid, "auto", inputs)
        if not g:
            self._place_refusal = g.reason
            return None
        return fid

    def _direct_traj(self, q_goal):
        from atharv_nandi.core.planning import time_parameterize
        path = np.vstack([np.array(self.d.qpos[self.ik.qadr], float), np.asarray(q_goal, float)])
        return time_parameterize(path, self.planner.vmax, self.planner.amax), 1.0

    def _precheck(self, tg: Target, aim, pitch) -> dict:
        """MP 5 preconditions at the target pose: total shoulder torque <= 59.5 N.m with the expected fruit mass
        (``Target.mass_kg``; papaya 1.8 kg per the spec when the station test sets it), predicted CoM margin >= 0.03 m,
        payload room."""
        from atharv_nandi.core.arm_ik_solver import SHOULDER_TORQUE_LIMIT_NM
        from atharv_nandi.core import stability
        out: Dict[str, Any] = {}
        q, _ax = self.ik.tcp_solve(self.d.qpos, aim, tg.tool, pitch, gantry=None)
        if q is None:
            return out
        loads = self.expected_loads(tg.tool, tg.mass_kg if tg.task in ("clip_pedicel", "twist_pick") else 0.0)
        tq = self.ik.shoulder_torque_nm(self.d.qpos, q, loads)
        out["torque_nm"] = round(tq, 2)
        out["torque_ok"] = tq <= SHOULDER_TORQUE_LIMIT_NM
        margin = self.predicted_margin(q, loads)
        if margin is not None:
            out["margin_m"] = round(margin, 4)
            # MVP-SPEC 5.3 static margin rule (stability.gate_static_margin; NaN refuses) plus this executor's own floor
            out["margin_ok"] = bool(stability.gate_static_margin(margin, "ik")) and margin >= self.min_margin_m
        if tg.task in ("clip_pedicel", "twist_pick") and self.tc is not None:
            from atharv_nandi.core.tools.kinematic_transfer import PAYLOAD_CAP_KG
            room = self.tc.payload_kg() + tg.mass_kg <= PAYLOAD_CAP_KG + 1e-9
            slot_free = (self.tc.cradle is None) if tg.crop == "papaya" else bool(self.tc.free_cells())
            out["payload_ok"] = bool(room and slot_free)
        return out

    def predicted_margin(self, q5, loads) -> Optional[float]:
        """Static CoM margin over the feet in contact with the arm at q5 and the loads (MP 5 precondition)."""
        from atharv_nandi.core import stability
        m, d = self.m, self.d
        pts = {}
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
                if name.endswith("_calf"):
                    pts.setdefault(name, (float(c.pos[0]), float(c.pos[1])))
        if len(pts) < 3:
            return None
        com = self.ik.predicted_com(d.qpos, q5, loads)
        return float(stability.com_margin(com[:2], stability.support_polygon(list(pts.values()))))

    def _emit_mark(self, mk: dict):
        if self.publish is not None:
            self.publish("/nandi/mark", mk)


# ====================================================================== stations
def ground_z(m, d, x: float, y: float, z0: float = 3.0, exclude_body: int = -1) -> float:
    """World z of the first geom-group-0 surface below (x, y), ray cast down from ``z0``; 0 when nothing is hit."""
    gid = np.zeros(1, dtype=np.int32)
    grp = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    dist = mujoco.mj_ray(m, d, np.array([x, y, z0]), np.array([0.0, 0.0, -1.0]), grp, 1, int(exclude_body), gid)
    return z0 - dist if dist >= 0 else 0.0


def teleport_station(drv, station: dict, settle_s: float = 1.5) -> dict:
    """Assisted (``station_teleport``): put the robot at a WP0c station (xy, heading), seat the tools, settle with
    the arm at PARK. Returns {xy_err_m, tilt_deg, creep_mps}. WP9 / WP25 replace this with the walk."""
    from atharv_nandi.sim.world.modules import tools as tools_mod
    m, d = drv.m, drv.d
    h = math.radians(float(station["heading_deg"]))
    gid = np.zeros(1, dtype=np.int32)
    grp = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    def ground(x, y):
        dist = mujoco.mj_ray(m, d, np.array([x, y, 3.0]), np.array([0.0, 0.0, -1.0]), grp, 1, -1, gid)
        return 3.0 - dist if dist >= 0 else 0.0

    z0 = float(d.qpos[2]) - ground(d.qpos[0], d.qpos[1])
    x, y = (float(v) for v in station["xy"])
    d.qpos[0:3] = [x, y, ground(x, y) + z0 + 0.005]
    d.qpos[3:7] = [math.cos(h / 2), 0.0, 0.0, math.sin(h / 2)]
    d.qvel[0:6] = 0.0
    mujoco.mj_forward(m, d)
    tools_mod.seat_tools(m, d)
    drv.wait(settle_s)
    t_end = drv.t + 3.0
    while drv.t < t_end and float(np.linalg.norm(d.qvel[0:2])) > 0.005:
        drv.wait(0.1)
    p0 = d.qpos[0:2].copy()
    drv.wait(0.5)
    from atharv_nandi.core.stability import slope_from_quat
    pitch, roll = slope_from_quat(d.qpos[3:7])
    return {"xy_err_m": round(float(np.linalg.norm(d.qpos[0:2] - np.asarray(station["xy"]))), 4),
            "pitch_deg": round(pitch, 2), "roll_deg": round(roll, 2),
            "creep_mps": round(float(np.linalg.norm(d.qpos[0:2] - p0)) / 0.5, 4)}


def station_for(target: Target, tree_xy, standoff_m: float = 0.42, along_m: float = 0.05) -> dict:
    """A side-mode station for a target without a WP0c one: the robot beside the target, the target at robot-frame
    (along_m, standoff_m) (the WP0c convention), facing tangentially around the trunk."""
    p = np.asarray(target.point[:2], float)
    r = _unit(p - np.asarray(tree_xy[:2], float))       # radial, trunk -> target
    yaw = math.atan2(r[0], -r[1])                         # robot left (+y) = -r: the target is on its left
    xy = p + r * standoff_m - np.array([math.cos(yaw), math.sin(yaw)]) * along_m
    return {"xy": xy.tolist(), "heading_deg": math.degrees(yaw)}


STATION_BACKOFF_M = {"T4": 0.10}
STATION_SHIFTS_M = (0.0, 0.10, -0.10, 0.20, -0.20)
STATION_SHIFTS_T5_M = (0.0, -0.05, 0.05, -0.10)    # T5: along the arm bearing (keeps the cut-angle geometry)
DWELL_TILT_DEG = 3.0                 # OS 2.1: dwell presets <= 3 deg until the WP0c creep data
# target positions in the robot frame tried by ``cut_station`` (forward first: the body stays clear of the trunk)
CUT_STATION_SLOTS = ((0.70, -0.10), (0.70, 0.10), (0.62, 0.0), (0.05, 0.42), (0.05, -0.42))
SHOULDER_XY = (0.05, 0.0)            # arm_ik_solver shoulder0 (trunk frame) at gantry 0
ROBOT_CLEAR_TRUNK_M = 0.55           # robot centre to any trunk axis (legs reach ~0.45 m from the centre)


def best_cut_cos(stem_dir, bearing_xy) -> float:
    """Smallest |cos(tool axis, stem)| over pitches 0..90 deg for a tool axis yawed along ``bearing_xy``."""
    s = _unit(stem_dir)
    u = _unit(np.asarray(bearing_xy, float)[:2])
    return min(abs(float(np.array([u[0] * math.cos(p), u[1] * math.cos(p), -math.sin(p)]) @ s))
               for p in np.radians(np.arange(0, 91, 5)))


def cut_station(tg: Target, trunks_xy: Sequence[Sequence[float]], cos_max: float = math.sin(math.radians(20.0))):
    """A station from which the planar arm can lay the blade hinge along the shoot (cut rule < 30 deg): the
    shoulder -> target bearing near perpendicular to the shoot's horizontal direction. Tries the WP0c station first,
    then CUT_STATION_SLOTS with the bearing on either side of the shoot; rejects robot centres closer than
    ROBOT_CLEAR_TRUNK_M to any trunk. Returns (station, best |cos|) or (None, inf)."""
    a = np.asarray(tg.meta.get("a"), float)
    b = np.asarray(tg.meta.get("b"), float)
    s = b - a
    p = np.asarray(tg.point[:2], float)

    def ok_xy(xy):
        return all(math.hypot(xy[0] - t[0], xy[1] - t[1]) >= ROBOT_CLEAR_TRUNK_M for t in trunks_xy)

    def bearing_of(st):
        h = math.radians(st["heading_deg"])
        c, sn = math.cos(h), math.sin(h)
        sh = np.asarray(st["xy"], float) + np.array([c * SHOULDER_XY[0] - sn * SHOULDER_XY[1],
                                                       sn * SHOULDER_XY[0] + c * SHOULDER_XY[1]])
        return p - sh

    if tg.station is not None:
        c0 = best_cut_cos(s, bearing_of(tg.station))
        if c0 <= cos_max:
            return dict(tg.station), c0
    sxy = s[:2]
    if float(np.linalg.norm(sxy)) < 1e-6:
        sxy = np.array([1.0, 0.0])
    perp = _unit(np.array([-sxy[1], sxy[0]]))
    best = (None, math.inf)
    for fx, fy in CUT_STATION_SLOTS:
        for sign in (1.0, -1.0):
            want = sign * perp                                    # world bearing shoulder -> target
            rb = math.atan2(fy - SHOULDER_XY[1], fx - SHOULDER_XY[0])
            h = math.atan2(want[1], want[0]) - rb
            c, sn = math.cos(h), math.sin(h)
            xy = p - np.array([c * fx - sn * fy, sn * fx + c * fy])
            if not ok_xy(xy):
                continue
            st = {"xy": xy.tolist(), "heading_deg": math.degrees(h), "source": "wp8_cut_station"}
            cc = best_cut_cos(s, bearing_of(st))
            if cc < best[1] - 1e-9:
                best = (st, cc)
            if cc <= cos_max:
                return st, cc
    return best


TWIST_STATION_SLOTS = ((0.05, 0.52), (0.05, -0.52), (0.62, 0.0))
TWIST_BEARING_TOL_DEG = 15.0


def _bearing_of(st: dict, p_xy) -> np.ndarray:
    h = math.radians(st["heading_deg"])
    c, sn = math.cos(h), math.sin(h)
    sh = np.asarray(st["xy"], float) + np.array([c * SHOULDER_XY[0] - sn * SHOULDER_XY[1],
                                                   sn * SHOULDER_XY[0] + c * SHOULDER_XY[1]])
    return np.asarray(p_xy, float) - sh


def twist_station(tg: Target, trunks_xy: Sequence[Sequence[float]]):
    """T4 with a level pedicel (papaya: radial into the trunk): a station whose shoulder -> fruit bearing runs along the
    pedicel, so the roll axis is the stalk axis the WP2 release statistic reads. The WP0c (backed-off) station when it
    already is within TWIST_BEARING_TOL_DEG, else TWIST_STATION_SLOTS. A hanging (vertical) pedicel: the tool comes
    down from above and any bearing works. Returns (station, bearing error deg)."""
    ped = np.asarray(tg.meta.get("pedicel") or (0.0, 0.0, 1.0), float)
    p = np.asarray(tg.point[:2], float)
    base = backed_off_station(tg) if tg.station is not None else None
    if abs(_unit(ped)[2]) > 0.7 or float(np.linalg.norm(ped[:2])) < 1e-6:
        return base, 0.0
    want = _unit(ped[:2])

    def err_deg(st):
        b = _unit(_bearing_of(st, p))
        return math.degrees(math.acos(float(np.clip(b @ want, -1.0, 1.0))))

    if base is not None and err_deg(base) <= TWIST_BEARING_TOL_DEG:
        return base, err_deg(base)
    best = (base, err_deg(base) if base is not None else math.inf)
    for fx, fy in TWIST_STATION_SLOTS:
        rb = math.atan2(fy - SHOULDER_XY[1], fx - SHOULDER_XY[0])
        h = math.atan2(want[1], want[0]) - rb
        c, sn = math.cos(h), math.sin(h)
        xy = p - np.array([c * fx - sn * fy, sn * fx + c * fy])
        if not all(math.hypot(xy[0] - t[0], xy[1] - t[1]) >= ROBOT_CLEAR_TRUNK_M for t in trunks_xy):
            continue
        st = {"xy": xy.tolist(), "heading_deg": math.degrees(h), "source": "wp8_twist_station"}
        e = err_deg(st)
        if e < best[1]:
            best = (st, e)
        if e <= TWIST_BEARING_TOL_DEG:
            return st, e
    return best


def backed_off_station(tg: Target) -> dict:
    """The WP0c station of a target, backed off away from it for tools whose TCP is longer than WP0c assumed:
    the T4 jaw centre is 0.25 m from the wrist (WP0c 0.20) and the radial cradle approach needs a 0.12 m pre-pose,
    so at the WP0c 0.42 m stand-off the wrist would pass behind the shoulder axis. Backing off 0.10 m keeps the target
    within 80 % of the reach (WP0c rho 0.41-0.47 m vs reach80 0.75 m). Change request WP0c: re-run the T4 rows."""
    st = dict(tg.station)
    off = STATION_BACKOFF_M.get(tg.tool, 0.0)
    if off:
        v = np.asarray(tg.point[:2], float) - np.asarray(st["xy"], float)
        st["xy"] = (np.asarray(st["xy"], float) - off * _unit(v)).tolist()
    return st


# ====================================================================== tree runner
class TreeRunner:
    """One tree, one preset: mount the tool, assign the fruit to the pool, then the MP 0.4 target loop over the
    job's targets (station per target, SM, retry / skip / halt), MARKs collected. ``station_fn(target)`` puts the
    robot at its station (default ``teleport_station`` with the target's WP0c station)."""

    def __init__(self, ex: SimTaskExecutor, policy: Optional[RetryPolicy] = None,
                 station_fn: Optional[Callable[[Target], Optional[dict]]] = None,
                 fruit_mass_kg: Optional[Dict[str, float]] = None):
        self.ex = ex
        self.fruit_mass_kg = dict(fruit_mass_kg or {})   # crop -> kg override (the 1.8 kg papaya check, MP 5)
        self.policy = policy or RetryPolicy()
        self.station_fn = station_fn or self.default_station
        try:
            from atharv_nandi.sim.world.modules import trees as trees_mod
            self.trunks_xy = [tuple(pl["pos"][:2]) for pl in trees_mod.plans_for_model(ex.m)]
        except Exception:  # noqa: BLE001 - a world without trees
            self.trunks_xy = []
        self.last_crop: Optional[str] = None
        self.last_tree: Optional[str] = None
        self.results: List[dict] = []
        self.marks: List[dict] = []

    def default_station(self, tg: Target) -> Optional[dict]:
        """Assisted station placement (``station_teleport``): T2 / T4 the WP0c station (T4 backed off); T5 a station
        whose arm bearing lets the blade hinge lie along the shoot (``cut_station``). Then the dwell rule (OS 2.1:
        <= 3 deg until WP0c creep data): if the robot settles tilted more than DWELL_TILT_DEG, the station is shifted
        along the robot's x (T5: along the arm bearing) by STATION_SHIFTS_M and the flattest one is kept."""
        extra = {}
        if tg.tool == "T5" and tg.meta.get("a") is not None:
            st, cc = cut_station(tg, self.trunks_xy)
            if st is None:
                return None
            extra = {"station_source": st.get("source", "wp0c"), "cut_cos": round(cc, 3)}
        elif tg.tool == "T4":
            st, e = twist_station(tg, self.trunks_xy)
            if st is None:
                return None
            extra = {"station_source": st.get("source", "wp0c"), "bearing_err_deg": round(e, 1)}
        elif tg.station is not None:
            st = backed_off_station(tg)
        else:
            return None
        h = math.radians(float(st["heading_deg"]))
        if tg.tool == "T5":
            d = _unit(np.asarray(tg.point[:2], float) - np.asarray(st["xy"], float))
        else:
            d = np.array([math.cos(h), math.sin(h)])
        best = None
        tries = []
        for k, off in enumerate(STATION_SHIFTS_T5_M if tg.tool == "T5" else STATION_SHIFTS_M):
            cand = dict(st, xy=(np.asarray(st["xy"], float) + off * d).tolist())
            out = teleport_station(self.ex.drv, cand)
            tilt = max(abs(out["pitch_deg"]), abs(out["roll_deg"]))
            reach = self.ex.reachable(tg)
            tries.append({"shift_m": off, "tilt_deg": tilt, "creep_mps": out["creep_mps"], "reachable": reach})
            score = tilt + 100.0 * max(0.0, out["creep_mps"] - 0.005) + (0.0 if reach else 1000.0)
            if best is None or score < best[0]:
                best = (score, cand, out)
            if tilt <= DWELL_TILT_DEG and out["creep_mps"] <= 0.005 and reach:
                break
        if best[2] is not out:
            out = teleport_station(self.ex.drv, best[1])
        out.update(extra)
        out["shift_m"] = round(float(np.dot(np.asarray(best[1]["xy"]) - np.asarray(st["xy"]), d)), 3)
        out["tries"] = tries
        out["dwell_ok"] = max(abs(out["pitch_deg"]), abs(out["roll_deg"])) <= DWELL_TILT_DEG
        return out

    def _sm(self, tg: Target, **kw):
        return SM_CLASSES[tg.task](tg, **kw)

    def assign(self, job) -> Dict[str, int]:
        ids = [t.meta["fruit_id"] for t in job.targets if t.task in ("clip_pedicel", "twist_pick")]
        if not ids:
            return {}
        out = self.ex.rt.pool.assign_tree(self.ex.d, job.tree_id, ids)
        kg = self.fruit_mass_kg.get(job.crop)
        if kg is not None:              # assign_tree already synced dof_M0 for the pool mass (WP23)
            for k in out.values():
                set_fruit_mass(self.ex.m, self.ex.rt.pool, k, kg)
        if kg is not None:
            for t in job.targets:
                t.mass_kg = float(kg)
        mujoco.mj_forward(self.ex.m, self.ex.d)
        return out

    def run_tree(self, job, sm_kwargs: Optional[dict] = None) -> dict:
        ex = self.ex
        sm_kwargs = sm_kwargs or {}
        for mk in job.marks:
            self.marks.append(mk)
            ex._emit_mark(mk)
        tools = {t.tool for t in job.targets}
        res = {"tree_id": job.tree_id, "crop": job.crop, "preset": job.preset, "results": [], "marks": list(job.marks),
               "skipped": list(job.skipped), "deferred": list(job.deferred), "stations": []}
        if not job.targets:
            res["loop"] = TargetLoop([], self.policy, job.tree_id).summary()
            return res
        if banana_tidy.needs_blade_clean(self.last_crop, job.crop, self.last_tree, job.tree_id):
            ex.blade_clean()
            res["blade_clean"] = True
        self.last_crop, self.last_tree = job.crop, job.tree_id
        tool = next(iter(tools))
        if not ex.mount(tool):
            res["error"] = "tool_unavailable"
            return res
        self.assign(job)
        loop = TargetLoop([t.target_id for t in job.targets], self.policy, job.tree_id)
        by_id = {t.target_id: t for t in job.targets}
        while not loop.finished:
            tid = loop.next_target()
            if tid is None:
                break
            tg = by_id[tid]
            st = self.station_fn(tg)
            if st is not None:
                res["stations"].append({"target_id": tid, **st})
            out = ex.run(self._sm(tg, **sm_kwargs.get(tg.task, {})))
            res["results"].append(out)
            self.marks.extend(out["marks"])
            dec = loop.report(tid, out["outcome"] or "failed", out["reason"],
                              committed=out["outcome"] in ("picked", "dropped", "cut") or bool(out.get("committed")))
            out["decision"] = dec["decision"]
        res["released"] = release_tree_slots(ex, job.tree_id)
        res["loop"] = loop.summary()
        res["marks"] = res["marks"] + [mk for r in res["results"] for mk in r["marks"]]
        return res


def set_fruit_mass(m, pool, slot: int, kg: float):
    """Scale an assigned pool fruit to ``kg`` (inertia with it): the MP 5 / OS 2.2 check at the 1.8 kg papaya."""
    s = pool.slots[slot]
    old = float(m.body_mass[s.body])
    f = float(kg) / old
    m.body_mass[s.body] = float(kg)
    m.body_subtreemass[s.body] = float(kg)
    m.body_inertia[s.body] = m.body_inertia[s.body] * f
    m.body_invweight0[s.body] = m.body_invweight0[s.body] / f
    m.dof_invweight0[s.dadr:s.dadr + 6] = m.dof_invweight0[s.dadr:s.dadr + 6] / f
    pool_mod.sync_dof_m0(m, s)       # simple body: MuJoCo's mass matrix is dof_M0, not body_mass


def release_tree_slots(ex: SimTaskExecutor, tree_id: str) -> List[str]:
    """End of a tree: give back the pool slots of its fruit that the transfer does not hold (still attached on the
    tree -> static fruit shown again; released but not in the tray, e.g. dropped -> harvested)."""
    pool, d = ex.rt.pool, ex.d
    held = {it["slot"] for it in (ex.tc.items.values() if ex.tc is not None else []) if not it.get("recycled")}
    out = []
    for k, a in pool.assignments.items():
        if a["tree_id"] != tree_id or k in held or (ex.rt.caught and ex.rt.caught.get("slot") == k):
            continue
        pool.recycle(d, k, harvested=not pool.is_attached(d, k), forward=False)
        out.append(a["fruit_id"])
    if out:
        mujoco.mj_forward(ex.m, d)
    return out


def empty_tray(ex: SimTaskExecutor) -> dict:
    """Assisted stand-in for DOCK / UNLOAD between station runs (``tray_teleport_unload``; WP13 / WP27 own the real
    unload): the tray and cradle contents are counted, the transfer is reset and their pool slots are recycled as
    harvested. Returns {n, kg, per_tree}."""
    tc, pool, d = ex.tc, ex.rt.pool, ex.d
    per: Dict[str, Dict[str, float]] = {}
    slots = []
    for it in tc.items.values():
        if it.get("recycled") or it["state"] not in ("seated", "moving"):
            continue
        p = per.setdefault(it["tree_id"], {"n": 0, "kg": 0.0})
        p["n"] += 1
        p["kg"] = round(p["kg"] + it["mass_kg"], 4)
        slots.append(it["slot"])
    tc.reset(d)
    for k in slots:
        if k in pool.assignments:
            pool.recycle(d, k, harvested=True, forward=False)
    mujoco.mj_forward(ex.m, d)
    return {"n": len(slots), "kg": round(sum(p["kg"] for p in per.values()), 4), "per_tree": per}


def _trace_row_impl(ex, tg, obs, sm, aim_fn) -> dict:
    row = {"t": round(obs.t, 3), "state": sm.state, "err_mm": round(1000 * obs.tcp_err_m, 2) if math.isfinite(obs.tcp_err_m)
           else None, "drift_mm": round(1000 * obs.base_drift_m, 1), "touch": round(obs.touch_n, 2),
           "roll": round(obs.roll_deg, 1), "travel": round(obs.roll_travel_deg, 1), "caught": obs.caught,
           "ev": [e.get("event") for e in obs.events]}
    if tg.tool == "T4":
        head = mujoco.mj_name2id(ex.m, mujoco.mjtObj.mjOBJ_BODY, "twist_head")
        k = ex.rt.pool.slot_of(tg.meta["fruit_id"])
        if k is not None:
            s = ex.rt.pool.slots[k]
            R = np.array(ex.d.xmat[head]).reshape(3, 3)
            row["fruit_head"] = np.round(R.T @ (np.array(ex.d.xpos[s.body]) - np.array(ex.d.xpos[head])), 4).tolist()
            row["jaw"] = round(float(ex.d.qpos[ex.rt.h["T4"].act_qadr]), 4)
    if obs.tcp_err_m > 0.02 and math.isfinite(obs.tcp_err_m):
        tb = set(ex.rt.h[tg.tool].bodies) | set(ex.planner.scene.arm_bodies)
        cons = []
        for i in range(ex.d.ncon):
            c = ex.d.contact[i]
            b1, b2 = int(ex.m.geom_bodyid[c.geom1]), int(ex.m.geom_bodyid[c.geom2])
            if b1 in tb or b2 in tb:
                cons.append((mujoco.mj_id2name(ex.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(ex.m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), round(c.dist, 4)))
        row["cons"] = cons[:6]
        row["q_cmd"] = np.round(ex.q_cmd, 3).tolist()
        row["q"] = np.round(ex.d.qpos[ex.ik.qadr], 3).tolist()
    return row


SimTaskExecutor._trace_row = lambda self, tg, obs, sm, aim_fn: _trace_row_impl(self, tg, obs, sm, aim_fn)
