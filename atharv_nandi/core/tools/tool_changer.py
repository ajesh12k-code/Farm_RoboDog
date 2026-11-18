"""Rack tool change: the MVP-SPEC 2.3 ToolChanger state machine for T2 / T4 / T5 (WP2; ORCHARD-MVP-SPEC 2.4).

``ToolChanger`` is pure Python (no MuJoCo, no ROS). One job moves one tool between its rack cradle and the arm:

    mount  idle -> open_gripper -> transit -> rack_pre -> rack_grip -> close -> check_pose -> lock -> verify
           -> lift -> park -> done
    dock   idle -> transit -> rack_pre -> rack_grip -> check_pose -> lock -> verify -> open_gripper -> lift
           -> park -> done
    any    -> fault -> safe_park -> fault (assist request, MVP-SPEC 5.6)

A change A -> B is ``dock A`` then ``mount B`` (``ToolChanger.change`` queues both). Rules (MVP-SPEC 2.3):

    rack_grip   the flange (mount) or the tool's handle_ref (dock) within ``grip_tol_mm`` (5) of its target
                within ``grip_timeout_s`` after the move, else fault ``grip_align``
    check_pose  mount: X_meas = inv(X_wrist) . X_tool against the design relpose X_nom; dock: inv(X_trunk) .
                X_tool against X_rack. Error > 2 mm or > 2 deg -> fault ``pose_check`` (a 4 mm offset is refused)
    lock        mount: write X_nom (never X_meas) into the grip weld, grip on, rack off, tool contacts on;
                dock: write X_rack into the rack weld, rack on, grip off, tool contacts parked. One atomic op
    verify      after ``verify_s`` (0.1 s): residual of the active weld < 1 mm, the other weld off, and presence:
                mount = the handle on the flange (``tool_present``, <= 1 mm), dock = the handle in its cradle
                (``cradle_occupied``, <= 5 mm); at the end of the lift the other side must be empty (mount: the
                cradle, dock: the flange), else fault ``verify_presence``
    interlocks  a job is refused while a payload is held, e-stopped, the crown is up (ext > 5 mm), the predicted
                CoM margin is < 0.03 m or a person is in the WORK field (``swap_gate``). The same interlocks
                (except holding) are re-checked every tick while a job runs (``SwapObs.interlock``): an e-stop, a
                person in the field, the crown going up or a low margin during a job ends it in ``fault`` with
                that reason. E-stop and person freeze the arm (``safe_park`` waits until they clear); after the
                crown / margin faults ``safe_park`` runs at once (folding the arm raises the margin)

Arm motions are joint-space smoothstep moves between the poses of a ``SwapPlan`` (the plant computes them:
WP0c tool-down docking at the frozen gantry value). ``MjToolPlant`` is the MuJoCo side: plans, observations
and the weld / contact ops on a composed model.
"""
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from atharv_nandi.core import stability as st
from atharv_nandi.core.interfaces import GateDecision
from atharv_nandi.core.tools import registry as reg

POSE_TOL_MM = 2.0
POSE_TOL_DEG = 2.0
RESIDUAL_TOL_MM = 1.0
GRIP_TOL_MM = 5.0
CROWN_STOWED_M = 0.005
MIN_MARGIN_M = st.PLAN_MIN_STATIC_M        # 0.03 m (MVP-SPEC 5.3); the rule itself is st.gate_static_margin
FINGER_OPEN = 0.0
FREEZE_INTERLOCKS = ("estop", "person_in_work_field")   # the arm does not move (no safe_park) while these hold
FINGER_ON_HANDLE = 0.034          # finger_* joint: pads 1 mm off the 20 mm handle (q 0 open, 0.04 closed)

DURATIONS = {"open_gripper": 0.2, "transit": 1.0, "rack_pre": 0.9, "rack_grip": 0.6, "close": 0.25,
             "verify": 0.1, "lift": 0.5, "park": 1.0, "safe_park": 1.2}
GRIP_TIMEOUT_S = 0.8

MOUNT_PHASES = ("open_gripper", "transit", "rack_pre", "rack_grip", "close", "check_pose", "lock", "verify", "lift",
                "park")
DOCK_PHASES = ("transit", "rack_pre", "rack_grip", "check_pose", "lock", "verify", "open_gripper", "lift", "park")
MOVES = {"transit": "q_transit", "rack_pre": "q_pre", "rack_grip": "q_dock", "lift": "q_pre", "park": "q_park"}


@dataclass
class Interlocks:
    estop: bool = False
    holding: bool = False                 # a payload is held (catch weld, jaw touch or fingers on a crop)
    crown_ext_m: float = 0.0
    margin_m: Optional[float] = None      # predicted CoM margin (m); None = unknown (not gated)
    person_in_field: bool = False


def swap_gate(il: Interlocks) -> GateDecision:
    """May a tool change start now? Refusal reasons (MVP-SPEC 2.3 interlocks + OS 2.3 rule 5 crown retract)."""
    vals = {"crown_ext_m": float(il.crown_ext_m), "margin_m": -1.0 if il.margin_m is None else float(il.margin_m)}
    if il.estop:
        return GateDecision.refuse("tool_change", "estop", **vals)
    if il.holding:
        return GateDecision.refuse("tool_change", "holding_payload", **vals)
    if not il.crown_ext_m <= CROWN_STOWED_M:          # NaN extension = not known stowed (fail closed)
        return GateDecision.refuse("tool_change", "crown_up", **vals)
    if il.margin_m is not None and not _margin_ok(il.margin_m, "tool_change"):   # NaN margin refuses (WPCOV review)
        return GateDecision.refuse("tool_change", "low_margin", **vals)
    if il.person_in_field:
        return GateDecision.refuse("tool_change", "person_in_work_field", **vals)
    return GateDecision.ok("tool_change", **vals)


JAW_OPEN_CTRL_M = 0.15         # twist_jaw ctrl above this = jaws commanded open (disarms the roll count)
CLOSE_TARGET_M = 0.15          # blade-close interlock: a target within this of the TCP (MVP-SPEC 2.3)
BLADE_CLOSE_CTRL = 0.5         # blade value at or above this = a closing command (the cut rule re-arms below it)


STABILITY_STOP_FLAGS = ("latched", "degenerate")   # monitor flags that refuse whatever the margin says


def _margin_ok(margin: Any, what: str) -> bool:
    """The static CoM margin rule (``stability.gate_static_margin``, looked up at call time so the interlock mutation
    check disables the live rule). A bool, non-real or non-finite margin refuses; numpy scalars are real numbers."""
    if isinstance(margin, bool) or not isinstance(margin, numbers.Real):
        return False
    m = float(margin)
    return math.isfinite(m) and bool(st.gate_static_margin(m, what))


def stability_ok(values: Optional[dict]) -> bool:
    """The live stability interlock shared by the blade / jaw close gate and the WP8 executor: ``values["stability"]``
    (the stability monitor's output, ``sim/hooks/stability.py``). False when ``ok`` is False, the monitor is
    ``latched`` or the stance is ``degenerate``, or the margin fails ``stability.gate_static_margin`` (below
    ``MIN_MARGIN_M``, non-finite or non-numeric). A ``margin_m`` key published as None is the monitor's encoding of a
    non-finite margin (NaN, or -inf for a degenerate tripod) and refuses too (fail closed; WPCOV review). No
    published stability dict, or one without a ``margin_m`` key -> True (the precheck and the plan gates stay the
    backstop)."""
    stab = (values or {}).get("stability") or {}
    if not isinstance(stab, dict):
        return True
    if stab.get("ok") is False:
        return False
    if any(bool(stab.get(k)) for k in STABILITY_STOP_FLAGS):
        return False
    if "margin_m" not in stab:
        return True
    return _margin_ok(stab["margin_m"], "close")


def close_gate(kind: str, value: Any, values: Optional[dict], nearest_target_m) -> Optional[str]:
    """MVP-SPEC 2.3 blade / jaw close interlock (pure; the WP2 tools hook calls it from ``actuate``).

    ``kind`` is the mounted tool's kind (``twist`` closes below ``JAW_OPEN_CTRL_M``, blades close at or above
    ``BLADE_CLOSE_CTRL``); ``values`` the bus values (``work_field.person``, ``stability.ok / margin_m``);
    ``nearest_target_m`` the TCP -> nearest target distance, a float or a zero-argument callable (evaluated only for
    a closing command that passed the other checks). Returns the refusal reason or None (not a close, or allowed).
    A non-finite or non-numeric stability margin refuses (fail closed; it used to pass as ``nan < 0.03`` is False)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None                                   # the hook's value check refuses it
    closing = float(value) < JAW_OPEN_CTRL_M if kind == "twist" else float(value) >= BLADE_CLOSE_CTRL
    if not closing:
        return None
    values = values or {}
    work = values.get("work_field") or {}
    if isinstance(work, dict) and work.get("person"):
        return "person_in_work_field"
    if not stability_ok(values):
        return "stability"
    dist = nearest_target_m() if callable(nearest_target_m) else nearest_target_m
    if not float(dist) <= CLOSE_TARGET_M:         # NaN distance = no target
        return "no_target"
    return None


@dataclass
class SwapPlan:
    tool_id: str
    direction: str                        # mount | dock
    q_transit: Tuple[float, ...]
    q_pre: Tuple[float, ...]
    q_dock: Tuple[float, ...]
    q_park: Tuple[float, ...]
    tool_mass_kg: float = 0.0


@dataclass
class SwapObs:
    t: float
    flange_handle_mm: float = math.inf    # mount: flange -> handle_ref; dock: handle_ref -> cradle point
    pose_err_mm: float = math.inf         # X_meas vs X_nom (mount) or X_rack (dock)
    pose_err_deg: float = math.inf
    residual_mm: float = math.inf         # active weld residual (grip after a mount lock, rack after a dock lock)
    other_weld_active: bool = True        # the weld that must be off after the lock (rack for mount, grip for dock)
    estop: bool = False
    interlock: str = ""                   # a swap_gate reason that holds now (person / crown_up / low_margin)
    tool_present: Optional[bool] = None   # handle_ref on the flange (<= 1 mm); None = not measured
    cradle_occupied: Optional[bool] = None   # handle_ref in its cradle (<= 5 mm); None = not measured


@dataclass
class SwapCmd:
    phase: str
    q5: Optional[Tuple[float, ...]] = None
    fingers: Tuple[float, float] = (FINGER_OPEN, FINGER_OPEN)
    payload_kg: float = 0.0               # mass hanging on the wrist beyond the arm (tool), for feed-forward
    ops: List[Tuple[str, str]] = field(default_factory=list)   # ("lock_grip" | "lock_rack", tool_id)
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class _Job:
    plan: SwapPlan
    phases: Tuple[str, ...]
    i: int = 0
    t0: float = 0.0
    q_from: Optional[Tuple[float, ...]] = None
    locked: bool = False
    pose_err: Tuple[float, float] = (math.nan, math.nan)
    t_start: float = 0.0
    wait_t0: Optional[float] = None


def _smooth(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


class ToolChanger:
    """Rack swap state machine. ``request``/``change`` start jobs; ``step(obs)`` advances one control tick."""

    def __init__(self, mounted: Optional[str] = None, durations: Optional[Dict[str, float]] = None):
        self.mounted = mounted            # tool id on the arm, or None (T1 picker)
        self.durations = dict(DURATIONS, **(durations or {}))
        self.phase = "idle"
        self.reason = ""
        self.queue: List[SwapPlan] = []
        self.job: Optional[_Job] = None
        self.q_cmd: Optional[Tuple[float, ...]] = None
        self.fingers: Tuple[float, float] = (FINGER_OPEN, FINGER_OPEN)
        self.last_result: Optional[Dict[str, Any]] = None
        self._safe_from: Optional[Tuple[float, ...]] = None
        self._safe_t0 = 0.0
        self._q_park: Optional[Tuple[float, ...]] = None

    # ------------------------------------------------------------------ queries
    @property
    def busy(self) -> bool:
        return self.job is not None or bool(self.queue) or self.phase == "safe_park"

    @property
    def faulted(self) -> bool:
        return self.phase in ("fault", "safe_park")

    def state(self) -> Dict[str, Any]:
        return {"mounted": self.mounted, "phase": self.phase, "reason": self.reason, "busy": self.busy,
                "job": None if self.job is None else {"tool": self.job.plan.tool_id,
                                                      "direction": self.job.plan.direction},
                "queued": [(p.tool_id, p.direction) for p in self.queue], "last": self.last_result}

    # ------------------------------------------------------------------ requests
    def request(self, plans: Sequence[SwapPlan], il: Interlocks, q_now: Sequence[float]) -> GateDecision:
        """Queue one or more jobs (a dock and / or a mount) after the interlock gate. Refused while busy."""
        if self.busy:
            return GateDecision.refuse("tool_change", "busy")
        if self.phase == "fault":
            return GateDecision.refuse("tool_change", "fault_needs_assist")
        gate = swap_gate(il)
        if not gate:
            self.last_result = {"ok": False, "reason": gate.reason, "tool": plans[0].tool_id if plans else None}
            return gate
        mounted = self.mounted
        for p in plans:
            if p.direction == "mount" and mounted is not None:
                return GateDecision.refuse("tool_change", "already_mounted")
            if p.direction == "dock" and mounted != p.tool_id:
                return GateDecision.refuse("tool_change", "not_mounted")
            if p.direction not in ("mount", "dock"):
                return GateDecision.refuse("tool_change", "bad_direction")
            mounted = p.tool_id if p.direction == "mount" else None
        self.queue = list(plans)
        self.q_cmd = tuple(float(v) for v in q_now)
        return gate

    def clear_fault(self):
        """Operator assist acknowledged (after safe_park): back to idle; the mount state stays as it is."""
        if self.phase == "fault":
            self.phase, self.reason = "idle", ""

    # ------------------------------------------------------------------ stepping
    def _start_next(self, t: float):
        plan = self.queue.pop(0)
        phases = MOUNT_PHASES if plan.direction == "mount" else DOCK_PHASES
        self.job = _Job(plan, phases, 0, t, self.q_cmd, t_start=t)
        self._q_park = plan.q_park
        self._enter(t)

    def _enter(self, t: float):
        j = self.job
        j.t0, j.q_from, j.wait_t0 = t, self.q_cmd, None
        self.phase = j.phases[j.i]

    def _advance(self, t: float, cmd: SwapCmd):
        j = self.job
        j.i += 1
        if j.i >= len(j.phases):
            res = {"ok": True, "tool": j.plan.tool_id, "direction": j.plan.direction, "reason": "",
                   "pose_err_mm": j.pose_err[0], "pose_err_deg": j.pose_err[1], "duration_s": t - j.t_start}
            self.last_result = res
            cmd.events.append({"event": "tool_change_done", **res})
            self.job = None
            self.phase = "idle"
            if self.queue:
                self._start_next(t)
            return
        self._enter(t)

    def _fault(self, t: float, reason: str, cmd: SwapCmd, **extra):
        j = self.job
        res = {"ok": False, "tool": j.plan.tool_id if j else None, "direction": j.plan.direction if j else None,
               "reason": reason, **extra}
        self.last_result = res
        cmd.events.append({"event": "tool_change_fault", **res})
        self.job, self.queue = None, []
        self.phase, self.reason = "safe_park", reason
        self._safe_from, self._safe_t0 = self.q_cmd, t

    def _fault_queued(self, t: float, reason: str, cmd: SwapCmd):
        """A queued (not started) job meets an interlock: the whole change faults before the arm moves."""
        p = self.queue[0]
        self.job = _Job(p, MOUNT_PHASES if p.direction == "mount" else DOCK_PHASES, 0, t, self.q_cmd, t_start=t)
        self._q_park = p.q_park
        self._fault(t, reason, cmd)

    def _payload(self) -> float:
        return 0.0 if self.mounted is None else reg.get(self.mounted).mass_kg

    def step(self, obs: SwapObs) -> SwapCmd:
        t = float(obs.t)
        cmd = SwapCmd(phase=self.phase)
        stop = "estop" if obs.estop else (obs.interlock or "")
        if self.job is None and self.queue and stop:
            self._fault_queued(t, stop, cmd)
        if self.job is None and self.queue:
            self._start_next(t)
        if self.phase == "safe_park":
            if stop in FREEZE_INTERLOCKS:
                self._safe_t0 = t                       # frozen while e-stopped; restarts from where it stopped
                self._safe_from = self.q_cmd
            elif self._q_park is not None and self._safe_from is not None:
                s = _smooth((t - self._safe_t0) / self.durations["safe_park"])
                self.q_cmd = tuple(a + s * (b - a) for a, b in zip(self._safe_from, self._q_park))
                if s >= 1.0:
                    self.phase = "fault"
            else:
                self.phase = "fault"
        elif self.job is not None:
            if stop:
                self._fault(t, stop, cmd)
            else:
                self._step_job(t, obs, cmd)
        cmd.phase = self.phase
        cmd.q5 = self.q_cmd
        cmd.fingers = self.fingers
        cmd.payload_kg = self._payload()
        return cmd

    def _step_job(self, t: float, obs: SwapObs, cmd: SwapCmd):
        j = self.job
        ph = self.phase
        el = t - j.t0
        p = j.plan
        if ph in MOVES:
            goal = getattr(p, MOVES[ph])
            dur = self.durations[ph]
            s = _smooth(el / dur)
            self.q_cmd = tuple(a + s * (b - a) for a, b in zip(j.q_from, goal))
            if s < 1.0:
                return
            if ph == "lift":
                mount = p.direction == "mount"
                # presence part 2: mount = tool on the hand AND the cradle empty; dock = cradle full AND hand empty
                bad = (obs.tool_present is not None and obs.tool_present != mount) or \
                      (obs.cradle_occupied is not None and obs.cradle_occupied == mount)
                if bad:
                    self._fault(t, "verify_presence", cmd, tool_present=obs.tool_present,
                                cradle_occupied=obs.cradle_occupied)
                    return
            if ph == "rack_grip":
                if obs.flange_handle_mm > GRIP_TOL_MM:
                    if j.wait_t0 is None:
                        j.wait_t0 = t
                    if t - j.wait_t0 > GRIP_TIMEOUT_S:
                        self._fault(t, "grip_align", cmd, flange_handle_mm=obs.flange_handle_mm)
                    return
            self._advance(t, cmd)
        elif ph == "open_gripper":
            self.fingers = (FINGER_OPEN, FINGER_OPEN)
            if el >= self.durations[ph]:
                self._advance(t, cmd)
        elif ph == "close":
            self.fingers = (FINGER_ON_HANDLE, FINGER_ON_HANDLE)
            if el >= self.durations[ph]:
                self._advance(t, cmd)
        elif ph == "check_pose":
            j.pose_err = (float(obs.pose_err_mm), float(obs.pose_err_deg))
            if not (obs.pose_err_mm <= POSE_TOL_MM and obs.pose_err_deg <= POSE_TOL_DEG):
                self._fault(t, "pose_check", cmd, pose_err_mm=obs.pose_err_mm, pose_err_deg=obs.pose_err_deg)
                return
            self._advance(t, cmd)
        elif ph == "lock":
            if p.direction == "mount":
                cmd.ops.append(("lock_grip", p.tool_id))
                self.mounted = p.tool_id
            else:
                cmd.ops.append(("lock_rack", p.tool_id))
                self.mounted = None
            j.locked = True
            self._advance(t, cmd)
        elif ph == "verify":
            if el < self.durations[ph]:
                return
            # presence (MVP-SPEC 2.3 step 8), part 1 (the arm is still at the cradle): mount = the tool on the hand,
            # dock = the tool in its cradle. Part 2 (cradle empty / hand empty) runs at the end of the lift
            mount = p.direction == "mount"
            seen = obs.tool_present if mount else obs.cradle_occupied
            if obs.residual_mm >= RESIDUAL_TOL_MM or obs.other_weld_active or seen is False:
                self._fault(t, "verify", cmd, residual_mm=obs.residual_mm, other_weld_active=obs.other_weld_active,
                            tool_present=obs.tool_present, cradle_occupied=obs.cradle_occupied)
                return
            cmd.events.append({"event": "tool_locked", "tool": p.tool_id, "direction": p.direction,
                               "residual_mm": obs.residual_mm})
            self._advance(t, cmd)
        else:  # pragma: no cover - phase tables are closed
            self._fault(t, f"unknown_phase:{ph}", cmd)


# ---------------------------------------------------------------------------------------------- MuJoCo plant
def _rot_err_deg(Ra, Rb) -> float:
    import numpy as np
    R = np.asarray(Ra).T @ np.asarray(Rb)
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.degrees(math.acos(c))


class RigidMount(ValueError):
    """A change requested on a rigid build (``NANDI_TOOL_MOUNT=rigid:<tool>``): refused ``rigid_mount``."""


class MjToolPlant:
    """Plans, observations and weld / contact ops for ``ToolChanger`` on a composed world (MjModel + MjData)."""

    def __init__(self, m, d, ik=None):
        import numpy as np
        from atharv_nandi.sim.world.modules import tools as tools_mod
        from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver
        self.m, self.d = m, d
        self.h = tools_mod.handles(m)
        if self.h is None:
            raise ValueError("this world has no tools (compose with NANDI_TOOLS=on or an orchard world)")
        self.ik = ik if ik is not None else DorsalArmIKSolver(m)
        self._np = np
        # home arm pose from the keyframe (the park pose)
        self.q_home = tuple(float(v) for v in (m.key_ctrl[0][:5] if m.nkey else np.zeros(5)))

    # ---------------------------------------------------------------- state
    def mounted(self) -> Optional[str]:
        if self.h.rigid is not None:
            return self.h.rigid                 # NANDI_TOOL_MOUNT=rigid:<tool>: mounted for the whole run
        on = [k for k, ids in self.h.tools.items() if self.d.eq_active[ids.grip_eq]]
        return on[0] if on else None

    def weld_counts(self) -> Dict[str, int]:
        d = self.d
        return {"rack_active": sum(int(d.eq_active[i.rack_eq]) for i in self.h.tools.values() if not i.rigid),
                "grip_active": sum(int(d.eq_active[i.grip_eq]) for i in self.h.tools.values())}

    # ---------------------------------------------------------------- planning
    def dock_poses(self, tool_id: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """(q_dock, q_pre): tool-down at the cradle (flange on the cradle point) and 5 cm above it."""
        np = self._np
        t = reg.get(tool_id)
        cradle = np.asarray(self.m.site_pos[self.h[tool_id].cradle], float)
        q_dock = self.ik.planar_solution(cradle + np.array([0.0, 0.0, reg.FLANGE_X_M]), t.cradle_gantry)
        q_pre = self.ik.planar_solution(cradle + np.array([0.0, 0.0, reg.FLANGE_X_M + 0.05]), t.cradle_gantry)
        if q_dock is None or q_pre is None:
            raise ValueError(f"{t.title}: cradle not reachable tool-down (WP0c frozen poses changed?)")
        return tuple(float(v) for v in q_dock), tuple(float(v) for v in q_pre)

    def plan(self, tool_id: str, direction: str) -> SwapPlan:
        t = reg.get(tool_id)
        q_dock, q_pre = self.dock_poses(tool_id)
        h = self.q_home
        q_transit = (t.cradle_gantry, q_dock[1], h[2], h[3], h[4])
        return SwapPlan(tool_id, direction, q_transit, q_pre, q_dock, tuple(h), t.mass_kg)

    def plans_for_change(self, target: Optional[str]) -> List[SwapPlan]:
        """Jobs that leave ``target`` mounted (None / 'T1' = the picker: dock whatever is mounted)."""
        target = None if target in (None, "", reg.PICKER) else reg.get(target).tool_id
        cur = self.mounted()
        out = []
        if cur == target:
            return out
        if self.h.rigid is not None:
            raise RigidMount(f"rigid build: {self.h.rigid} is mounted for the whole run (NANDI_TOOL_MOUNT)")
        if cur is not None:
            out.append(self.plan(cur, "dock"))
        if target is not None:
            out.append(self.plan(target, "mount"))
        return out

    # ---------------------------------------------------------------- observation
    def _pose(self, body: int):
        np = self._np
        return np.array(self.d.xpos[body]), np.array(self.d.xmat[body]).reshape(3, 3)

    def rel_tool(self, tool_id: str, frame_body: int):
        """(pos, R) of the tool body in ``frame_body``'s frame."""
        pf, Rf = self._pose(frame_body)
        pt, Rt = self._pose(self.h[tool_id].body)
        return Rf.T @ (pt - pf), Rf.T @ Rt

    def eq_relpose(self, eq: int):
        np = self._np
        e = np.asarray(self.m.eq_data[eq], float)
        R = np.zeros(9)
        import mujoco
        mujoco.mju_quat2Mat(R, e[6:10])
        return e[3:6].copy(), R.reshape(3, 3)

    def weld_error(self, tool_id: str, which: str) -> Tuple[float, float]:
        """(mm, deg) between the tool's pose and the relpose of its ``rack`` or ``grip`` weld (active or not)."""
        np = self._np
        ids = self.h[tool_id]
        eq, frame = (ids.rack_eq, self.h.trunk) if which == "rack" else (ids.grip_eq, self.h.wrist)
        p, R = self.rel_tool(tool_id, frame)
        p0, R0 = self.eq_relpose(eq)
        return 1000.0 * float(np.linalg.norm(p - p0)), _rot_err_deg(R0, R)

    def observe(self, tool_id: Optional[str], direction: Optional[str], estop: bool = False) -> SwapObs:
        np = self._np
        from atharv_nandi.core.tools import grasp_sense
        obs = SwapObs(t=float(self.d.time), estop=bool(estop))
        if tool_id is None:
            return obs
        ids = self.h[tool_id]
        d = self.d
        obs.tool_present = grasp_sense.tool_present(d.site_xpos[self.h.flange], d.site_xpos[ids.handle])
        obs.cradle_occupied = grasp_sense.cradle_occupied(d.site_xpos[ids.handle], d.site_xpos[ids.cradle])
        if direction == "mount":
            obs.flange_handle_mm = 1000.0 * float(np.linalg.norm(d.site_xpos[self.h.flange] - d.site_xpos[ids.handle]))
            obs.pose_err_mm, obs.pose_err_deg = self.weld_error(tool_id, "grip")
            obs.residual_mm = obs.pose_err_mm
            obs.other_weld_active = bool(d.eq_active[ids.rack_eq])
        else:
            obs.flange_handle_mm = 1000.0 * float(np.linalg.norm(d.site_xpos[ids.handle] - d.site_xpos[ids.cradle]))
            obs.pose_err_mm, obs.pose_err_deg = self.weld_error(tool_id, "rack")
            obs.residual_mm = obs.pose_err_mm
            obs.other_weld_active = bool(d.eq_active[ids.grip_eq])
        return obs

    # ---------------------------------------------------------------- ops
    def apply_ops(self, ops: Sequence[Tuple[str, str]]):
        from atharv_nandi.sim.world.modules import tools as tools_mod
        for op, tool_id in ops:
            ids = self.h[tool_id]
            if op == "lock_grip":
                self.m.eq_data[ids.grip_eq][3:10] = list(reg.X_NOM_POS) + list(reg.X_NOM_QUAT)   # X_nom, not X_meas
                self.d.eq_active[ids.grip_eq] = 1
                self.d.eq_active[ids.rack_eq] = 0
                tools_mod.set_contact(self.m, tool_id, True)
            elif op == "lock_rack":
                self.d.eq_active[ids.rack_eq] = 1
                self.d.eq_active[ids.grip_eq] = 0
                tools_mod.set_contact(self.m, tool_id, False)
            else:
                raise ValueError(f"unknown tool op {op!r}")

    # ---------------------------------------------------------------- interlocks
    def crown_ext_m(self) -> float:
        import mujoco
        ext = 0.0
        for n in ("mast_s1", "mast_s2", "mast_s3"):
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
            if b >= 0:
                ext += float(self.m.body_pos[b][2])
        return ext

    def margin_m(self) -> Optional[float]:
        """Static CoM margin over the feet in ground contact (``stability.com_margin``); None if < 3 feet."""
        import mujoco
        from atharv_nandi.core import stability
        m, d = self.m, self.d
        feet = []
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                b = int(m.geom_bodyid[g])
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if name.endswith("_calf"):
                    feet.append((name, float(c.pos[0]), float(c.pos[1])))
        pts = {}
        for name, x, y in feet:
            pts.setdefault(name, (x, y))
        if len(pts) < 3:
            return None
        poly = stability.support_polygon(list(pts.values()))
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        return float(stability.com_margin(d.subtree_com[root][:2], poly))


# ---------------------------------------------------------------------------------------------- tool TCP IK
def tcp_ik(ik, qpos, target, axis, tcp_local=(0.20, 0.0, 0.0), q_seed=None, iters: int = 120, tol_m: float = 2e-4,
           axis_w: float = 0.3, damp: float = 1e-3):
    """Arm joints (5) putting a point fixed on ``arm_wrist`` (``tcp_local``, e.g. the mounted tool's TCP at
    (tcp_from_wrist, 0, 0)) at ``target`` with the wrist x axis (= tool axis) along ``axis`` (world).

    Damped least squares on the ``DorsalArmIKSolver`` model; returns (q5, pos_err_m, axis_err_deg). A reference
    for WP2 tests and the WP8 tool-aware planner (the solver's own IK targets ``gripper_tip`` only)."""
    import mujoco
    import numpy as np
    m, d = ik.model, ik.data
    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, reg.WRIST_BODY)
    q = np.array(q_seed if q_seed is not None else qpos[ik.qadr], dtype=np.float64)
    tgt, ax = np.asarray(target, float), np.asarray(axis, float) / np.linalg.norm(axis)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    loc = np.asarray(tcp_local, float)
    for _ in range(iters):
        ik._fk(qpos, q)
        R = d.xmat[wrist].reshape(3, 3)
        p = d.xpos[wrist] + R @ loc
        x = R[:, 0]
        e_p = tgt - p
        e_r = np.cross(x, ax)
        if np.linalg.norm(e_p) < tol_m and np.linalg.norm(e_r) < 1e-3:
            break
        mujoco.mj_jac(m, d, jacp, jacr, p, wrist)
        J = np.vstack([jacp[:, ik.dadr], axis_w * jacr[:, ik.dadr]])
        e = np.concatenate([e_p, axis_w * e_r])
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
        q = np.clip(q + dq, ik.limits[:, 0], ik.limits[:, 1])
    ik._fk(qpos, q)
    R = d.xmat[wrist].reshape(3, 3)
    p = d.xpos[wrist] + R @ loc
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(R[:, 0] @ ax)))))
    return q, float(np.linalg.norm(tgt - p)), ang


def tcp_closed_form(ik, qpos, target, tcp_from_wrist: float, pitch: float = 0.0, gantry: float = 0.0):
    """Closed-form arm joints (elbow up) putting the tool TCP at ``target`` (world) with the tool axis in the arm
    plane, pitched ``pitch`` rad below horizontal (0 = level, pi/2 = tool down). The arm is planar, so the axis
    yaw is the shoulder -> target bearing. Returns (q5, axis_world) or (None, None) when out of reach."""
    import numpy as np
    t_pos, t_rot = ik.trunk_pose(qpos)
    p = t_rot.T @ (np.asarray(target, float) - t_pos)
    s0 = ik.shoulder0 + np.array([gantry, 0.0, 0.0])
    u = p[:2] - s0[:2]
    if float(np.linalg.norm(u)) < 1e-6:
        return None, None
    u = u / np.linalg.norm(u)
    axis = np.array([u[0] * math.cos(pitch), u[1] * math.cos(pitch), -math.sin(pitch)])
    wrist = p - tcp_from_wrist * axis
    q = ik.planar_solution(wrist, gantry)
    if q is None:
        return None, None
    q = np.array(q, float)
    q[4] = q[4] - math.pi / 2.0 + pitch          # planar_solution aims the tool down (sum pi/2)
    if not (ik.limits[4, 0] <= q[4] <= ik.limits[4, 1]):
        return None, None
    return q, t_rot @ axis
