"""T5 PRUNER cut for the CLIP / PRUNE jobs (MISSION-PRESETS 6, ORCHARD-MVP-SPEC 3.4, 5.3).

Jobs: ``rootstock_sprout`` (any shoot below ``graft_union_h_m``), ``skirt_lift`` (branches below 0.5-0.75 m),
``dead_twig``, ``malformed_panicle``, ``water_shoot`` (mostly > 12 mm: MARK), ``tip_back_20cm`` (mango post-harvest),
``banana_tidy`` (dead blade dangling into the band, cut at the band; ``banana_tidy.py``).

States (``PruneCutSM.state``):

    start        keep wood / green leaf -> skip ``keep_wood`` (never cut: O-13 / O-14); out of the band -> MARK
                 ``out_of_reach_marked`` (tool_hint lopper / saw); > ``max_stem_mm`` (12) -> MARK ``over_capacity``
                 with ``diameter_mm`` and ``paste_required`` (> 12 mm, Bordeaux paste)
    plan         transit to the pre-pose 10 cm before the cut point (WPCAP planner), blade open
    roll_align   pruner roll so the hinge lies along the shoot (cut rule < 30 deg)
    approach     straight in along the tool axis (targets are contype 0 in the sim, so the jaws never push)
    align        closed-loop TCP servo on the cut point to <= 1 mm
    close        blade closed; WP2 ``cut`` (commit: the shoot is hidden, debris drops) or ``prune_refused``
                 (over capacity -> open, MARK) or no cut 1 s after the close -> re-align once (MP 6), then skip
                 ``cut_failed``
    retreat      blade open, back out 10 cm, park; outcome ``cut``
"""
from typing import Optional

from atharv_nandi.core.tasks.common import (PRUNER_MAX_STEM_MM, Cmd, Obs, Target, TaskSM, first_event, in_pick_band,
                                      make_mark, paste_required)

PRE_M = 0.10
RETREAT_M = 0.10
CUT_WAIT_S = 1.0
REALIGN_DEG = 10.0
ROLL_SETTLE_S = 0.3

MARK_KIND = {"rootstock_sprout": "rootstock_sprout", "skirt_lift": "skirt_lift", "malformed_panicle": "malformed_panicle",
             "banana_tidy": "sanitation_deleaf", "sucker": "sucker"}


def tool_hint_for(diameter_mm: Optional[float], z_m: float) -> str:
    """MARK tool hint: sickle for thin high shoots, lopper up to 25 mm, saw above (assumption)."""
    d = float(diameter_mm or 0.0)
    if d > 25.0:
        return "saw"
    if z_m > 1.8 and d <= 12.0:
        return "pole_picker"
    return "lopper"


class PruneCutSM(TaskSM):
    method = "prune"
    pre_m = PRE_M

    def __init__(self, target: Target, roll_deg: float = 0.0, retries: int = 1,
                 max_stem_mm: float = PRUNER_MAX_STEM_MM, **kw):
        super().__init__(target, retries=retries, **kw)
        self.roll0 = self.roll = float(roll_deg)
        self.max_stem_mm = float(max_stem_mm)
        self.cut: Optional[dict] = None

    def _mark(self, reason: str, hint: Optional[str] = None, extra: Optional[dict] = None) -> dict:
        tg = self.target
        kind = MARK_KIND.get(tg.job or "", "prune")
        return make_mark(tg.meta.get("tree", {"tree_id": tg.tree_id}), kind, tg.point, reason,
                         diameter_mm=tg.diameter_mm, tool_hint=hint or tool_hint_for(tg.diameter_mm, tg.point[2]),
                         target_id=tg.target_id, extra={"job": tg.job, **(extra or {})})

    def s_start(self, obs: Obs):
        tg = self.target
        if tg.keep or tg.green:
            return None, self.finish(obs.t, "skipped", "keep_wood")
        if not tg.in_band or not in_pick_band(tg.point[2]):
            self.marks.append(self._mark("out_of_reach_marked"))
            return None, self.finish(obs.t, "marked", "out_of_reach_marked")
        if tg.diameter_mm is not None and tg.diameter_mm > self.max_stem_mm:
            self.marks.append(self._mark("over_capacity"))          # paste_required is the MARK flag (> 12 mm)
            return None, self.finish(obs.t, "marked", "over_capacity")
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
        if self.aligned(obs) or self.elapsed(obs) >= self.align_timeout_s:
            return "close", Cmd(tool="close")
        return None, Cmd()

    def s_close(self, obs: Obs):
        cut = first_event(obs.events, "cut")
        if cut is not None:
            self.cut, self.committed = cut, True
            self.record = {"target_id": self.target.target_id, "job": self.target.job,
                           "diameter_mm": cut.get("diameter_mm"), "debris": cut.get("debris"),
                           "cut_target": cut.get("target")}
            return "retreat", Cmd(arm="retreat", arm_args={"s": -RETREAT_M}, tool="open")
        ref = first_event(obs.events, "prune_refused")
        if ref is not None:
            paste = bool(ref.get("paste_required")) or paste_required(self.target.diameter_mm)
            self.marks.append(self._mark("over_capacity", extra={"paste_required": paste}))
            return None, self._skip_retreat(obs, "over_capacity", outcome="marked")
        if first_event(obs.events, "cut_miss") is not None or self.elapsed(obs) >= CUT_WAIT_S:
            if self.attempts <= self.retries:
                self.attempts += 1
                self.roll = self.roll0 + (REALIGN_DEG if self.attempts % 2 == 0 else -REALIGN_DEG)
                return "realign", Cmd(tool="open", roll_deg=self.roll)
            return None, self._skip_retreat(obs, "cut_failed")
        return None, Cmd()

    def s_realign(self, obs: Obs):
        if self.elapsed(obs) >= ROLL_SETTLE_S:
            return "align", Cmd(arm="servo")
        return None, Cmd(tool="open", roll_deg=self.roll)

    def s_retreat(self, obs: Obs):
        if obs.arm_done or obs.arm_failed:
            return "park", Cmd(arm="park")
        return None, Cmd()

    def s_park(self, obs: Obs):
        if obs.arm_done or obs.arm_failed:
            return None, self.finish(obs.t, "cut")
        return None, Cmd()
