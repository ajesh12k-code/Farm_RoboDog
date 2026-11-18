"""WP8 shared task-layer pieces (pure Python, no MuJoCo, no ROS): reason codes, MARK and harvest records, the pick
band rule, papaya ``jaw_clearance``, the observation / command types the per-crop state machines exchange with an
executor, and the ``TaskSM`` base class.

Spec: ORCHARD-MVP-SPEC 2.2 (pick band), 2.4 (tools, jaw clearance), 3.4 (cut / twist rules), 5.3 (failure modes);
MISSION-PRESETS 0.4 (failure semantics), 0.5 (reason codes), 0.6 (MarkRecord ``nandi.mark/1``), 5 (HARVEST / PICK),
6 (CLIP / PRUNE). Labels as in the specs: every threshold here is an (assumption) unless the spec freezes it.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------- constants (spec values)
PICK_BAND_Z_M = 0.8              # OS 2.2: contact targets at z <= 0.8 m
REACH_FRACTION = 0.8             # OS 2.2: and within 80 % of the measured reach in their direction
JAW_OPENING_M = 0.16             # OS 2.4: T4 jaw opening
JAW_CLEARANCE_MM = 15.0          # OS 2.4: + 15 mm per side clear of trunk and neighbours
PASTE_REQUIRED_MM = 12.0         # MP 0.6 / 6: cut > 12 mm -> Bordeaux paste
PRUNER_MAX_STEM_MM = 12.0        # MP 6: max_stem_mm = PRUNER capacity (assumption)
PAPAYA_PLACE_MPS = 0.5           # MP 5 / OS 2.4: papaya lowered at <= 0.5 m/s
PEDICEL_TOL_MM = 5.0             # WP8 acceptance: pedicel error <= 5 mm (out of tolerance -> deliver and report)
BASE_DRIFT_M = 0.02              # OS 2.1: base moved > 2 cm during a dwell -> pause the arm and re-approach
TWIST_FAIL_DEG = 540.0           # MP 5 / OS 5.3: no release at relative roll >= 540 deg -> open, MARK twist_failed
SLIP_TOUCH_N = 1.0               # OS 5.3: jaw touch < 1 N during the carry = slip in the cradle
BANANA_EXCLUSION_M = 3.0         # OS 2.4 / 5.2: keep >= 3 m from a MARKed bunch while a human cut is in progress
BLADE_CLEAN_S = 5.0              # OS 2.4: blade_clean time step between banana plants and on any crop change

# MISSION-PRESETS 0.5 (the subset the task layer emits) + WP8 additions (see the WP8 contract notes)
REASONS = ("out_of_reach_marked", "paste_required", "twist_failed", "no_clearance", "bagged", "over_capacity",
           "payload_full", "tool_unavailable", "stability", "person_hold", "not_in_mvp", "human_cut",
           # WP8 additions (change request WP34: atharv_nandi/core/workflow/reasons.py)
           "cut_failed", "dropped", "plan_failed", "base_moved", "keep_wood", "immature", "torque_limit", "estop",
           "pedicel_out_of_tolerance", "grip_failed", "place_refused", "tree_halt", "operator_skip", "timeout")
MARK_KINDS = ("fruit", "prune", "rootstock_sprout", "skirt_lift", "malformed_panicle", "sanitation_deleaf", "sucker",
              "bunch")
TOOL_HINTS = ("pole_picker", "lopper", "sickle", "saw", "human_cut")
OUTCOMES = ("picked", "cut", "marked", "skipped", "dropped", "failed", "halted")


# ---------------------------------------------------------------------- geometry helpers
def h_class(h_m: float) -> str:
    """MP 0.6: neeche < 0.8 m | beech 0.8-1.8 m | upar > 1.8 m."""
    h = float(h_m)
    return "neeche" if h < 0.8 else ("beech" if h <= 1.8 else "upar")


def side_of(tree_xy: Sequence[float], p_xy: Sequence[float]) -> str:
    """Compass side (N/E/S/W) of a point around its trunk; +y is north, +x east (geo frame of the field JSON)."""
    dx, dy = float(p_xy[0]) - float(tree_xy[0]), float(p_xy[1]) - float(tree_xy[1])
    az = math.degrees(math.atan2(dx, dy)) % 360.0      # 0 = N, 90 = E
    return "NESW"[int(((az + 45.0) % 360.0) // 90.0)]


def clock_of(tree_xy: Sequence[float], p_xy: Sequence[float], view_from_xy: Optional[Sequence[float]] = None) -> int:
    """Clock position 1-12 seen from the alley: 12 = the far side of the trunk, 6 = the near side (towards the
    viewer), 3 = right, 9 = left. ``view_from_xy`` = where the operator stands (default: due south)."""
    tx, ty = float(tree_xy[0]), float(tree_xy[1])
    vx, vy = (tx, ty - 1.0) if view_from_xy is None else (float(view_from_xy[0]), float(view_from_xy[1]))
    away = math.atan2(ty - vy, tx - vx)                  # direction viewer -> trunk
    ang = math.atan2(float(p_xy[1]) - ty, float(p_xy[0]) - tx) - away
    cw = (-math.degrees(ang)) % 360.0                    # clockwise from "far side"
    k = int(round(cw / 30.0)) % 12
    return 12 if k == 0 else k


def in_pick_band(z_m: float, reach_frac: Optional[float] = None) -> bool:
    """OS 2.2 rule: z <= 0.8 m and (when known) within 80 % of the reach in that direction."""
    return float(z_m) <= PICK_BAND_Z_M + 1e-9 and (reach_frac is None or float(reach_frac) <= REACH_FRACTION + 1e-9)


def jaw_clearance_ok(clearance_mm: Optional[float], need_mm: float = JAW_CLEARANCE_MM) -> bool:
    """OS 2.4 ``jaw_clearance``: the free space beside the open 0.16 m jaws must be >= 15 mm per side (WP23 truth
    ``clearance_mm``, the wrist-depth stand-in). Unknown clearance passes (the T4 contact check is the backstop)."""
    return clearance_mm is None or float(clearance_mm) >= float(need_mm)


def paste_required(diameter_mm: Optional[float], limit_mm: float = PASTE_REQUIRED_MM) -> bool:
    return diameter_mm is not None and float(diameter_mm) > float(limit_mm)


# ---------------------------------------------------------------------- records
def make_mark(tree: dict, kind: str, point: Sequence[float], reason: str, *, stage: Optional[str] = None,
              diameter_mm: Optional[float] = None, tool_hint: Optional[str] = None, target_id: Optional[str] = None,
              clock: Optional[int] = None, h_m: Optional[float] = None, extra: Optional[dict] = None) -> dict:
    """MarkRecord ``nandi.mark/1`` (MP 0.6), one per human job. ``tree`` = {tree_id, physical_tag?, pos (x, y)}."""
    if kind not in MARK_KINDS:
        raise ValueError(f"mark kind {kind!r} not in {MARK_KINDS}")
    if tool_hint is not None and tool_hint not in TOOL_HINTS:
        raise ValueError(f"tool_hint {tool_hint!r} not in {TOOL_HINTS}")
    p = [float(v) for v in point]
    txy = tree.get("pos", (0.0, 0.0))[:2]
    h = float(p[2]) if h_m is None else float(h_m)
    rec = {"schema": "nandi.mark/1", "tree_id": tree["tree_id"], "physical_tag": tree.get("physical_tag"),
           "kind": kind, "target_id": target_id,
           "clock": int(clock) if clock is not None else clock_of(txy, p[:2]),
           "side": side_of(txy, p[:2]), "h_class": h_class(h), "h_m": round(h, 3), "stage": stage,
           "reason": reason, "diameter_mm": None if diameter_mm is None else round(float(diameter_mm), 1),
           "paste_required": paste_required(diameter_mm), "tool_hint": tool_hint}
    if extra:
        rec.update(extra)
    return rec


def harvest_record(tree_id: str, target_id: str, crop: str, method: str, outcome: str, *,
                   pedicel_mm: Optional[float] = None, pedicel_error_mm: Optional[float] = None,
                   cell: Any = None, reason: str = "", attempts: int = 1, cycle_s: Optional[float] = None,
                   assisted: Sequence[str] = ()) -> dict:
    """``HarvestRecord`` per item (MP 5 outputs): tree_id, method, pedicel_mm, outcome, cell id."""
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome {outcome!r} not in {OUTCOMES}")
    return {"schema": "nandi.harvest/1", "tree_id": tree_id, "target_id": target_id, "crop": crop, "method": method,
            "outcome": outcome, "reason": reason,
            "pedicel_mm": None if pedicel_mm is None else round(float(pedicel_mm), 2),
            "pedicel_error_mm": None if pedicel_error_mm is None else round(float(pedicel_error_mm), 2),
            "cell": cell, "attempts": int(attempts), "cycle_s": None if cycle_s is None else round(float(cycle_s), 2),
            "assisted": list(assisted)}


# ---------------------------------------------------------------------- SM I/O
@dataclass
class Target:
    """One contact target for a task state machine (built by ``tree_task.plan_*`` from WP23 truth)."""
    target_id: str
    tree_id: str
    crop: str
    task: str                         # clip_pedicel | twist_pick | prune_cut | banana_tidy
    tool: str                         # T2 | T4 | T5
    point: Tuple[float, float, float]  # world: cut point (clip / prune) or fruit centre (twist)
    job: Optional[str] = None         # prune job (rootstock_sprout, skirt_lift, ..., banana_tidy)
    stage: Optional[str] = None
    diameter_mm: Optional[float] = None
    mass_kg: float = 0.0
    clearance_mm: Optional[float] = None
    bagged: bool = False
    keep: bool = False
    in_band: bool = True
    green: bool = False               # banana: a functional (green) leaf; never cut
    pedicel_leave_m: float = 0.0
    station: Optional[dict] = None    # WP0c station {xy, heading_deg} if known
    pitch: float = 0.0                # tool axis pitch below horizontal at the target (rad)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Obs:
    """What the executor reports to a state machine every control tick (20 ms)."""
    t: float = 0.0
    arm_done: bool = False            # the current arm motion (plan / approach / retreat / park) has finished
    arm_failed: str = ""              # the motion could not be planned / executed (reason)
    tcp_err_m: float = math.inf       # |aim - TCP| of the servo (stand-in for the wrist-camera track)
    base_drift_m: float = 0.0         # base pose loop: |pose - dwell reference| (OS 2.1)
    base_settled: bool = True         # base speed below the settle threshold
    estop: bool = False
    person: bool = False              # person in the WORK field (hand-in-envelope stop)
    stability_ok: bool = True
    events: List[dict] = field(default_factory=list)   # tool events since the last tick (/nandi/tool/event shape)
    caught: bool = False              # the mounted tool holds a released fruit (catch weld)
    touch_n: float = 0.0              # T4: min of the two jaw pad forces (N): both pads press (grip)
    touch_max_n: float = 0.0          # T4: max of the two pad forces (N): the jaws still hold something (slip rule)
    roll_travel_deg: float = 0.0      # T4: accumulated |roll| travel since the jaws gripped
    roll_deg: float = 0.0             # tool roll joint (deg)
    placed: Optional[str] = None      # transfer result: None (running / not asked) | "seated" | "refused:<reason>"
    precheck: Dict[str, Any] = field(default_factory=dict)   # {torque_nm, margin_m} at the target pose


@dataclass
class Cmd:
    """What a state machine asks the executor to do this tick. ``arm`` changes start a new motion."""
    arm: Optional[str] = None         # plan_pre | approach | servo | retreat | park | hold | lower | none
    arm_args: Dict[str, Any] = field(default_factory=dict)
    tool: Optional[str] = None        # open | close | stop
    roll_deg: Optional[float] = None  # tool roll target (deg)
    place: bool = False               # release the catch and start the transfer into the tray / cradle
    rebase: bool = False              # take the current base pose as the dwell reference (after a re-approach)
    done: bool = False


class TaskSM:
    """Tick-driven state machine base: ``step(obs) -> Cmd``. Subclasses define ``s_<state>(obs)`` handlers that
    return ``(next_state, Cmd)``. Common rules (MP 0.3 / 0.4, OS 2.1):

    * an e-stop latches ``halted`` (tool stop, arm hold) from any state;
    * a person in the WORK field pauses (arm hold, tool open when before the commit point) until it clears, then the
      target is re-approached from the pre-pose: a fresh ``plan_pre`` (new IK, new OMPL transit to the stand-off,
      re-referenced base pose). The executor also aborts a running transit for the person, and freezes any arm motion
      the SM does not own (park, carry, skip retreat, lower) until the field is clear;
    * before the commit point, a base drift > 2 cm pauses the arm, waits for the base to settle, re-references the
      base pose and re-plans from the pre-pose (``max_reapproach``, then skip ``base_moved``);
    * before the commit point, a stability loss (``Obs.stability_ok`` False, or the WP2 blade-close gate refusing
      with ``stability``) opens the tool, backs out and skips ``stability`` (a retry reason: a new station);
    * after the commit (cut fired / twist released) the hold-verify -> place block is atomic: only the e-stop stops
      it (a person only freezes the arm in the executor).
    """

    pre_m = 0.10                      # pre-pose stand-off along the tool axis (subclasses set their PRE_M)

    PRE_COMMIT = ("roll_align", "approach", "align", "close", "grip", "twist", "realign")
    MOVING = ("plan", "roll_align", "approach", "align", "close", "grip", "twist", "realign")

    def __init__(self, target: Target, retries: int = 1, max_reapproach: int = 2, align_tol_m: float = 0.001,
                 align_timeout_s: float = 3.0):
        self.target = target
        self.retries = int(retries)
        self.max_reapproach = int(max_reapproach)
        self.align_tol_m = float(align_tol_m)
        self.align_timeout_s = float(align_timeout_s)
        self.state = "start"
        self.t_state = 0.0
        self.history: List[Tuple[float, str, str, str]] = []   # (t, from, to, why)
        self.attempts = 1
        self.reapproaches = 0
        self.committed = False
        self.outcome: Optional[str] = None
        self.reason = ""
        self.marks: List[dict] = []
        self.record: Dict[str, Any] = {}
        self.t0: Optional[float] = None
        self.t_end: Optional[float] = None
        self._resume_after_pause = "plan"
        self._ok_since: Optional[float] = None

    # -------------------------------------------------------------- driver
    @property
    def done(self) -> bool:
        return self.outcome is not None

    def go(self, t: float, state: str, why: str = ""):
        if state != self.state:
            self.history.append((round(t, 3), self.state, state, why))
            self.state, self.t_state = state, t
            self._ok_since = None

    def finish(self, t: float, outcome: str, reason: str = "") -> Cmd:
        self.outcome, self.reason, self.t_end = outcome, reason, t
        self.go(t, "done", f"{outcome}:{reason}")
        return Cmd(arm="none", done=True)

    def elapsed(self, obs: Obs) -> float:
        return obs.t - self.t_state

    def step(self, obs: Obs) -> Cmd:
        if self.t0 is None:
            self.t0 = obs.t
            self.t_state = obs.t
        if self.done:
            return Cmd(arm="none", done=True)
        if obs.estop and self.state != "halted":
            self.go(obs.t, "halted", "estop")
            self.outcome, self.reason, self.t_end = "halted", "estop", obs.t
            return Cmd(arm="hold", tool="stop", done=True)
        if not self.committed:
            if obs.person and self.state in self.MOVING:
                self._resume_after_pause = "plan"
                self.go(obs.t, "person_hold", "person_in_work_field")
                return Cmd(arm="hold", tool="open")
            if self.state in self.PRE_COMMIT and obs.base_drift_m > BASE_DRIFT_M:
                self.go(obs.t, "base_pause", f"drift {obs.base_drift_m * 1000:.0f} mm")
                return Cmd(arm="hold", tool="open")
            if self.state in self.MOVING and (not obs.stability_ok or close_refused(obs.events) == "stability"):
                return self._skip_retreat(obs, "stability")
        handler = getattr(self, f"s_{self.state}")
        nxt, cmd = handler(obs)
        if nxt is not None and nxt != self.state:
            self.go(obs.t, nxt)
        return cmd

    # -------------------------------------------------------------- common states
    def plan_roll_deg(self) -> Optional[float]:
        """Tool roll for the transit to the pre-pose (T2 / T5: the hinge-along-the-stem roll; T4 overrides)."""
        return getattr(self, "roll", 0.0)

    def replan_cmd(self, rebase: bool = False) -> Cmd:
        """Re-approach from the pre-pose: a new ``plan_pre`` (IK at the live target, OMPL transit to the stand-off),
        tool open. Used after a person hold and a base pause."""
        return Cmd(arm="plan_pre", arm_args={"standoff_m": self.pre_m}, tool="open", roll_deg=self.plan_roll_deg(),
                   rebase=rebase)

    def s_person_hold(self, obs: Obs):
        if obs.person:
            return None, Cmd(arm="hold")
        return self._resume_after_pause, self.replan_cmd(rebase=True)

    def s_base_pause(self, obs: Obs):
        if not obs.base_settled and self.elapsed(obs) < 3.0:
            return None, Cmd(arm="hold")
        self.reapproaches += 1
        if self.reapproaches > self.max_reapproach:
            return None, self._skip_retreat(obs, "base_moved")
        return "plan", self.replan_cmd(rebase=True)

    def s_halted(self, obs: Obs):
        return None, Cmd(arm="hold", tool="stop", done=True)

    def s_retreat_skip(self, obs: Obs):
        if obs.arm_done or obs.arm_failed:
            return "park_skip", Cmd(arm="park")
        return None, Cmd()

    def s_park_skip(self, obs: Obs):
        if obs.arm_done or obs.arm_failed:
            outcome = self._pending_outcome or "skipped"
            return None, self.finish(obs.t, outcome, self._pending_reason)
        return None, Cmd()

    _pending_reason = ""
    _pending_outcome: Optional[str] = None

    def _skip_retreat(self, obs: Obs, reason: str, outcome: str = "skipped", mark: Optional[dict] = None) -> Cmd:
        """Open the tool, back out along the approach axis, park, then finish ``outcome`` with ``reason``."""
        if mark is not None:
            self.marks.append(mark)
        self._pending_reason, self._pending_outcome = reason, outcome
        self.go(obs.t, "retreat_skip", reason)
        return Cmd(arm="retreat", arm_args={"s": -self.pre_m}, tool="open")

    def aligned(self, obs: Obs, hold_s: float = 0.2) -> bool:
        """TCP within the alignment tolerance for ``hold_s`` (the wrist-camera servo has converged)."""
        if obs.tcp_err_m <= self.align_tol_m:
            if self._ok_since is None:
                self._ok_since = obs.t
            return obs.t - self._ok_since >= hold_s
        self._ok_since = None
        return False

    def summary(self) -> dict:
        return {"target_id": self.target.target_id, "tree_id": self.target.tree_id, "task": self.target.task,
                "outcome": self.outcome, "reason": self.reason, "attempts": self.attempts,
                "reapproaches": self.reapproaches, "cycle_s": None if self.t_end is None or self.t0 is None
                else round(self.t_end - self.t0, 2), "marks": list(self.marks), "record": dict(self.record),
                "history": list(self.history)}


def close_refused(events: Sequence[dict]) -> Optional[str]:
    """The reason the WP2 blade / jaw close gate refused a close (``actuate_refused``: ``stability``,
    ``person_in_work_field``, ``no_target``, ...), or None."""
    e = first_event(events, "actuate_refused")
    return None if e is None else str(e.get("reason") or "")


def first_event(events: Sequence[dict], *names: str) -> Optional[dict]:
    for e in events:
        if e.get("event") in names:
            return e
    return None
