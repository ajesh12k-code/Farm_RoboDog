"""T2 CLIPPER pedicel clip, mango (one round) and guava (default method): MISSION-PRESETS 5, ORCHARD-MVP-SPEC 3.4.

Per target: select -> stem estimate -> roll align -> trap -> close -> cut event -> hold verify -> place -> rest.

States (``ClipPedicelSM.state``):

    start        preconditions: bagged -> skip ``bagged``; immature -> skip; out of the pick band -> MARK
                 ``out_of_reach_marked`` (tool_hint pole_picker); torque / margin precheck at the target pose
                 (``Obs.precheck``) -> skip ``torque_limit`` / ``stability``
    plan         collision-free transit to the pre-pose 10 cm before the cut point (WPCAP OMPL planner), blades open
    roll_align   clipper roll so the blade hinge lies along the pedicel (cut rule: stem vs hinge < 30 deg)
    approach     straight in along the tool axis until the pedicel is in the V-trap
    align        closed-loop TCP servo on ``cut_k`` (wrist-camera stand-in) to <= 1 mm
    close        blade closed; wait for the WP2 ``cut`` event (commit point) or a ``cut_miss`` / 1 s timeout
    realign      MP 5 retry: open, roll +-10 deg, re-trap (``retries``, default 1)
    hold_verify  the catcher holds the fruit (catch weld) -> retreat, else outcome ``dropped``
    retreat      back out 10 cm along the axis with the fruit
    park         arm to PARK (planned, held fruit attached)
    place        release the catch, WP3 transfer stalk-down into the next free cell; wait ``seated``
    done         outcome picked | skipped | marked | dropped | halted; ``record`` = HarvestRecord fields

A pedicel out of tolerance (> 5 mm) is delivered and reported (OS 5.3: retries 0). No pull release, ever.
"""
from typing import Optional

from atharv_nandi.core.tasks.common import (PEDICEL_TOL_MM, Cmd, Obs, Target, TaskSM, first_event, harvest_record,
                                      in_pick_band, make_mark)

PRE_M = 0.10            # pre-pose stand-off before the cut point along the tool axis
RETREAT_M = 0.10
CUT_WAIT_S = 1.0        # no cut event this long after the close -> treat as a miss (MP 6 "no cut 1 s after close")
REALIGN_DEG = 10.0      # MP 5: re-align +-10 deg, re-trap (1)
ROLL_SETTLE_S = 0.3
OPEN_S = 0.2


class ClipPedicelSM(TaskSM):
    method = "clip"
    pre_m = PRE_M

    def __init__(self, target: Target, roll_deg: float = 0.0, retries: int = 1, **kw):
        super().__init__(target, retries=retries, **kw)
        self.roll0 = float(roll_deg)        # roll that puts the hinge along the stem (from the stem estimate)
        self.roll = self.roll0
        self.cut: Optional[dict] = None
        self.tray_cell = None

    # -------------------------------------------------------------- states
    def s_start(self, obs: Obs):
        tg = self.target
        if tg.bagged:
            return None, self.finish(obs.t, "skipped", "bagged")
        if tg.stage is not None and tg.meta.get("harvest_stage") is False:
            return None, self.finish(obs.t, "skipped", "immature")
        if not tg.in_band or not in_pick_band(tg.point[2]):
            self.marks.append(make_mark(tg.meta.get("tree", {"tree_id": tg.tree_id}), "fruit", tg.point,
                                        "out_of_reach_marked", stage=tg.stage, tool_hint="pole_picker",
                                        target_id=tg.target_id))
            return None, self.finish(obs.t, "marked", "out_of_reach_marked")
        pre = obs.precheck or {}
        if pre.get("torque_ok") is False:
            return None, self.finish(obs.t, "skipped", "torque_limit")
        if pre.get("margin_ok") is False:
            return None, self.finish(obs.t, "skipped", "stability")
        return "plan", Cmd(arm="plan_pre", arm_args={"standoff_m": PRE_M}, tool="open", roll_deg=self.roll)

    def s_plan(self, obs: Obs):
        if obs.arm_failed:
            return None, self.finish(obs.t, "skipped", "plan_failed")
        if obs.arm_done:
            return "roll_align", Cmd(tool="open", roll_deg=self.roll)
        return None, Cmd()

    def s_roll_align(self, obs: Obs):
        if self.elapsed(obs) >= ROLL_SETTLE_S:
            return "approach", Cmd(arm="approach", arm_args={"s": 0.0})
        return None, Cmd(roll_deg=self.roll)

    def s_approach(self, obs: Obs):
        if obs.arm_failed:
            return None, self._skip_retreat(obs, "plan_failed")
        if obs.arm_done:
            return "align", Cmd(arm="servo")
        return None, Cmd()

    def s_align(self, obs: Obs):
        if self.aligned(obs):
            return "close", Cmd(tool="close")
        if self.elapsed(obs) >= self.align_timeout_s:
            return "close", Cmd(tool="close")      # best effort: the cut rule's trap window decides
        return None, Cmd()

    def s_close(self, obs: Obs):
        cut = first_event(obs.events, "cut")
        if cut is not None:
            self.cut, self.committed = cut, True
            return "hold_verify", Cmd()
        if first_event(obs.events, "cut_refused") is not None:
            return None, self._skip_retreat(obs, "not_in_mvp")
        miss = first_event(obs.events, "cut_miss")
        if miss is not None or self.elapsed(obs) >= CUT_WAIT_S:
            if self.attempts <= self.retries:
                self.attempts += 1
                sign = 1.0 if self.attempts % 2 == 0 else -1.0
                self.roll = self.roll0 + sign * REALIGN_DEG
                return "realign", Cmd(tool="open", roll_deg=self.roll)
            return None, self._skip_retreat(obs, "cut_failed")
        return None, Cmd()

    def s_realign(self, obs: Obs):
        if self.elapsed(obs) >= max(OPEN_S, ROLL_SETTLE_S):
            return "align", Cmd(arm="servo")
        return None, Cmd(tool="open", roll_deg=self.roll)

    def s_hold_verify(self, obs: Obs):
        if obs.caught:
            return "retreat", Cmd(arm="retreat", arm_args={"s": -RETREAT_M})
        if self.elapsed(obs) >= 0.3:
            return None, self._finish_record(obs, "dropped", "dropped")
        return None, Cmd()

    def s_retreat(self, obs: Obs):
        if obs.arm_done or obs.arm_failed:
            return "park", Cmd(arm="park")
        return None, Cmd()

    def s_park(self, obs: Obs):
        if not obs.caught:
            return None, self._finish_record(obs, "dropped", "dropped")
        if obs.arm_done or obs.arm_failed:
            return "place", Cmd(place=True)
        return None, Cmd()

    def s_place(self, obs: Obs):
        if obs.placed is None:
            return None, Cmd()
        if obs.placed == "seated":
            self.tray_cell = obs.precheck.get("cell") if obs.precheck else None
            err = self.cut.get("pedicel_error_mm") if self.cut else None
            reason = "pedicel_out_of_tolerance" if err is not None and abs(err) > PEDICEL_TOL_MM else ""
            return None, self._finish_record(obs, "picked", reason, tool="open")
        return None, self._finish_record(obs, "dropped", obs.placed.replace("refused:", "place_refused:"),
                                         tool="open")

    # -------------------------------------------------------------- record
    def _finish_record(self, obs: Obs, outcome: str, reason: str, tool: Optional[str] = None) -> Cmd:
        tg, c = self.target, self.cut or {}
        self.record = harvest_record(tg.tree_id, tg.target_id, tg.crop, self.method, outcome,
                                     pedicel_mm=c.get("pedicel_mm"), pedicel_error_mm=c.get("pedicel_error_mm"),
                                     cell=obs.precheck.get("cell") if obs.precheck else None, reason=reason,
                                     attempts=self.attempts, cycle_s=None if self.t0 is None else obs.t - self.t0,
                                     assisted=c.get("assisted", ()))
        cmd = self.finish(obs.t, outcome, reason)
        cmd.tool = tool
        return cmd
