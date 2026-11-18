"""T4 TWIST pick: papaya (one per trip, ``jaw_clearance``) and the guava twist option. MISSION-PRESETS 5,
ORCHARD-MVP-SPEC 2.4, 3.4, 5.3.

States (``TwistPickSM.state``):

    start        preconditions: bagged -> skip; out of the pick band -> MARK ``out_of_reach_marked``; papaya
                 ``jaw_clearance`` (0.16 m jaw + 15 mm per side clear of trunk and neighbours) fails -> MARK
                 ``no_clearance`` (retries 0); payload / cradle full -> skip ``payload_full``; torque (total shoulder
                 torque <= 59.5 N.m with the expected fruit mass, 1.8 kg for papaya) or margin -> skip
    plan         transit to the pre-pose 15 cm before the fruit centre, jaws open (WPCAP planner)
    roll_align   pre-roll the open jaws (see pre-roll) clear of the fruit
    approach     straight in along the tool axis (the pedicel runs radially into the trunk: the roll axis)
    align        closed-loop TCP servo on the fruit centre (+ fixed-jaw offset) to <= 2 mm
    grip         jaws closed; both soft pads > 1 N within 1.5 s, else re-grip once then skip ``grip_failed``
    pre-roll     the jaws close with the tool rolled PRE_ROLL_DEG against the twist direction: rigid pads on an elongated
                 papaya cam open at about 25-28 deg of roll from the narrow section (0.16 m jaw, 0.14 x 0.23 m fruit),
                 short of the 30 deg the WP2 release rule counts from the grip; gripping at -20 deg gives ~45 deg of
                 usable twist through the narrow section. The retry grips at +20 deg and twists the other way
    twist        roll ramp towards +270 deg (back to -270 deg at the end stop) with the jaws closed until the WP2 ``twist_release``
                 event (commit point); accumulated relative roll >= 540 deg without a release -> open, MARK
                 ``twist_failed``; one re-grip retry (OS 5.3 retries 1)
    hold_verify  caught and pads > 1 N -> carry
    carry        retreat + park at <= 0.5 m/s TCP (papaya bruising); a pad below 1 N during the carry = slip:
                 stop, lower, open, outcome ``dropped``
    place        release the catch; WP3 lays the papaya in the rear cradle (guava: a cell); wait ``seated``
"""
from typing import Optional

from atharv_nandi.core.tasks.common import (PAPAYA_PLACE_MPS, SLIP_TOUCH_N, TWIST_FAIL_DEG, Cmd, Obs, Target, TaskSM,
                                      first_event, harvest_record, in_pick_band, jaw_clearance_ok, make_mark)

PRE_M = 0.15          # jaw tips >= 2 cm before the fruit at the pre-pose (tips 4 cm past the TCP + fruit radius)
PRE_ROLL_S = 0.8
RETREAT_M = 0.12
GRIP_N = 1.0
GRIP_WAIT_S = 1.5
ROLL_STROKE_DEG = 270.0       # T4 twist_roll range is +-270 deg
PRE_ROLL_DEG = 20.0           # grip with the jaws pre-rolled against the twist direction (see below)
TWIST_RATE_DPS = 60.0         # roll ramp (deg/s); the WP2 release fired at 30 deg after ~4.7 s on the papaya cam
TWIST_TIMEOUT_S = 25.0
OPEN_S = 0.4
SLIP_LOWER_TIMEOUT_S = 8.0   # the lower takes <= 0.35 m at 0.1 m/s; a stuck lower still opens (and reports dropped)


class TwistPickSM(TaskSM):
    method = "twist"
    pre_m = PRE_M
    PRE_COMMIT = ("roll_align", "approach", "align", "grip", "twist", "realign", "regrip")
    MOVING = ("plan", "roll_align", "approach", "align", "grip", "twist", "regrip")

    def __init__(self, target: Target, retries: int = 1, carry_mps: Optional[float] = None, **kw):
        kw.setdefault("align_tol_m", 0.002)
        super().__init__(target, retries=retries, **kw)
        self.carry_mps = carry_mps if carry_mps is not None else (PAPAYA_PLACE_MPS if target.crop == "papaya" else 0.45)
        self.release: Optional[dict] = None
        self.roll_goal = ROLL_STROKE_DEG
        self.pre_roll = -PRE_ROLL_DEG
        self.watch_pads = False
        self._slip_since: Optional[float] = None

    def plan_roll_deg(self) -> Optional[float]:
        return 0.0                                          # jaws level for the transit; pre-roll at the pre-pose

    def s_start(self, obs: Obs):
        tg = self.target
        tree = tg.meta.get("tree", {"tree_id": tg.tree_id})
        if tg.bagged:
            return None, self.finish(obs.t, "skipped", "bagged")
        if not tg.in_band or not in_pick_band(tg.point[2]):
            self.marks.append(make_mark(tree, "fruit", tg.point, "out_of_reach_marked", stage=tg.stage,
                                        tool_hint="pole_picker", target_id=tg.target_id))
            return None, self.finish(obs.t, "marked", "out_of_reach_marked")
        if not jaw_clearance_ok(tg.clearance_mm):
            self.marks.append(make_mark(tree, "fruit", tg.point, "no_clearance", stage=tg.stage,
                                        tool_hint="pole_picker", target_id=tg.target_id,
                                        extra={"clearance_mm": tg.clearance_mm}))
            return None, self.finish(obs.t, "marked", "no_clearance")
        pre = obs.precheck or {}
        if pre.get("payload_ok") is False:
            return None, self.finish(obs.t, "skipped", "payload_full")
        if pre.get("torque_ok") is False:
            return None, self.finish(obs.t, "skipped", "torque_limit")
        if pre.get("margin_ok") is False:
            return None, self.finish(obs.t, "skipped", "stability")
        return "plan", Cmd(arm="plan_pre", arm_args={"standoff_m": PRE_M}, tool="open", roll_deg=0.0)

    def s_plan(self, obs: Obs):
        if obs.arm_failed:
            return None, self.finish(obs.t, "skipped", "plan_failed")
        if obs.arm_done:
            return "roll_align", Cmd(tool="open", roll_deg=self.pre_roll)
        return None, Cmd()

    def s_roll_align(self, obs: Obs):
        # pre-roll the open jaws at the pre-pose (clear of the fruit), then go in
        if self.elapsed(obs) >= PRE_ROLL_S:
            return "approach", Cmd(arm="approach", arm_args={"s": 0.0})
        return None, Cmd(roll_deg=self.pre_roll)

    def s_approach(self, obs: Obs):
        if obs.arm_failed:
            return None, self._skip_retreat(obs, "plan_failed")
        if obs.arm_done:
            return "align", Cmd(arm="servo")
        return None, Cmd()

    def s_align(self, obs: Obs):
        if self.aligned(obs) or self.elapsed(obs) >= self.align_timeout_s:
            return "grip", Cmd(tool="close")
        return None, Cmd()

    def s_grip(self, obs: Obs):
        if obs.touch_n >= GRIP_N and self.elapsed(obs) >= 0.3:
            self.roll_goal = ROLL_STROKE_DEG if self.pre_roll <= 0 else -ROLL_STROKE_DEG
            return "twist", Cmd(roll_deg=self.roll_goal, arm_args={"roll_rate_dps": TWIST_RATE_DPS})
        if self.elapsed(obs) >= GRIP_WAIT_S:
            if self.attempts <= self.retries:
                self.attempts += 1
                return "regrip", Cmd(tool="open", roll_deg=self.pre_roll)
            return None, self._skip_retreat(obs, "grip_failed")
        return None, Cmd()

    def s_regrip(self, obs: Obs):
        if self.elapsed(obs) >= OPEN_S:
            return "align", Cmd(arm="servo")
        return None, Cmd(tool="open", roll_deg=self.pre_roll)

    def s_twist(self, obs: Obs):
        rel = first_event(obs.events, "twist_release")
        if rel is not None:
            self.release, self.committed = rel, True
            return "hold_verify", Cmd(roll_deg=obs.roll_deg)      # stop rolling where it is
        if obs.roll_travel_deg >= TWIST_FAIL_DEG or self.elapsed(obs) >= TWIST_TIMEOUT_S:
            if self.attempts <= self.retries:
                self.attempts += 1
                self.pre_roll = -self.pre_roll                     # grip the other way round, twist the other way
                return "regrip", Cmd(tool="open", roll_deg=self.pre_roll)
            mark = make_mark(self.target.meta.get("tree", {"tree_id": self.target.tree_id}), "fruit", self.target.point,
                             "twist_failed", stage=self.target.stage, tool_hint="pole_picker",
                             target_id=self.target.target_id, extra={"roll_deg": round(obs.roll_travel_deg, 1)})
            return None, self._skip_retreat(obs, "twist_failed", outcome="marked", mark=mark)
        if self.roll_goal > 0 and obs.roll_deg >= ROLL_STROKE_DEG - 2.0:
            self.roll_goal = -ROLL_STROKE_DEG                    # twist back (the stroke is +-270 deg)
        elif self.roll_goal < 0 and obs.roll_deg <= -ROLL_STROKE_DEG + 2.0:
            self.roll_goal = ROLL_STROKE_DEG
        return None, Cmd(roll_deg=self.roll_goal, arm_args={"roll_rate_dps": TWIST_RATE_DPS})

    def s_hold_verify(self, obs: Obs):
        # pads pressing (>= 1 N) -> the carry watches the pads (OS 5.3 slip rule); after a cam-over the pads may read
        # 0 while the assisted catch weld holds the fruit -> the carry watches the catch instead (weld_grasp)
        if obs.caught and self.elapsed(obs) >= 0.2:
            self.watch_pads = obs.touch_max_n >= SLIP_TOUCH_N
            return "carry", Cmd(arm="retreat", arm_args={"s": -RETREAT_M, "max_tcp_mps": self.carry_mps})
        if self.elapsed(obs) >= 0.5:
            return "slip", Cmd(arm="lower", tool=None)
        return None, Cmd()

    def slipped(self, obs: Obs) -> bool:
        if not obs.caught:
            return True
        if not self.watch_pads:
            return False
        if obs.touch_max_n < SLIP_TOUCH_N:
            if self._slip_since is None:
                self._slip_since = obs.t
            return obs.t - self._slip_since >= 0.2
        self._slip_since = None
        return False

    def s_carry(self, obs: Obs):
        if self.slipped(obs):
            return "slip", Cmd(arm="lower")
        if obs.arm_done or obs.arm_failed:
            return "park", Cmd(arm="park", arm_args={"max_tcp_mps": self.carry_mps})
        return None, Cmd()

    def s_park(self, obs: Obs):
        if self.slipped(obs):
            return "slip", Cmd(arm="lower")
        if obs.arm_done or obs.arm_failed:
            return "place", Cmd(place=True)
        return None, Cmd()

    def s_slip(self, obs: Obs):
        # OS 5.3: stop, lower, open, log dropped (retries 0). The executor lowers the TCP (``arm="lower"``: straight
        # down at LOWER_MPS to LOWER_CLEAR_M above the ground, or as low as the arm reaches) and only then reports
        # arm_done, so the jaws open near the ground (release_safe), not at carry height
        if obs.arm_done or obs.arm_failed or self.elapsed(obs) >= SLIP_LOWER_TIMEOUT_S:
            return None, self._finish_record(obs, "dropped", "dropped", tool="open")
        return None, Cmd()

    def s_place(self, obs: Obs):
        if obs.placed is None:
            return None, Cmd()
        if obs.placed == "seated":
            return None, self._finish_record(obs, "picked", "", tool="open")
        return None, self._finish_record(obs, "dropped", obs.placed.replace("refused:", "place_refused:"), tool="open")

    def _finish_record(self, obs: Obs, outcome: str, reason: str, tool: Optional[str] = None) -> Cmd:
        tg, r = self.target, self.release or {}
        self.record = harvest_record(tg.tree_id, tg.target_id, tg.crop, self.method, outcome,
                                     cell=obs.precheck.get("cell") if obs.precheck else None, reason=reason,
                                     attempts=self.attempts, cycle_s=None if self.t0 is None else obs.t - self.t0,
                                     assisted=r.get("assisted", ()))
        self.record.update({"tau_peak_nm": r.get("tau_peak_nm"), "roll_deg": r.get("roll_deg"),
                            "mass_kg": tg.mass_kg})
        cmd = self.finish(obs.t, outcome, reason)
        cmd.tool = tool
        return cmd
