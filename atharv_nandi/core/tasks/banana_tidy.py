"""Banana: MARK-only harvest and the tidy-up cut (MISSION-PRESETS 5 banana row, 6 steps 4-6; ORCHARD-MVP-SPEC 2.4).

* ``BananaTidySM``: T5 cut of a **dead** blade dangling into the pick band, at the band (tidy-up only; the petiole cut
  is a human job). A green (functional) leaf is never a target: the SM refuses it (``keep_wood``, O-14: 0 green-leaf
  cuts) even if a caller hands one over.
* ``banana_marks``: the bunch MARK (ribbon colour -> shooting date -> harvest window, height, hands-count lead,
  ``tool_hint: human_cut``), the sanitation de-leaf MARKs (leaf ids with Sigatoka %, never leaving fewer than
  ``min_functional_leaves`` green leaves; WP23 truth ``sanitation_deleaf`` already respects the floor) and the
  de-sucker MARKs (peepers / swords; the follower is kept, Nandi never cuts at ground level).
* ``BLADE_CLEAN_S`` (5 s) between banana plants and on a crop change; ``exclusion_m`` (3 m) from a MARKed bunch while
  a human cut is flagged in progress (``bunch_exclusion_ok``). No "hold position" assist.
"""
import math
from typing import Dict, List, Sequence

from atharv_nandi.core.tasks.common import BANANA_EXCLUSION_M, BLADE_CLEAN_S, Obs, Target, make_mark
from atharv_nandi.core.tasks.prune_cut import PruneCutSM

__all__ = ["BananaTidySM", "banana_marks", "bunch_exclusion_ok", "needs_blade_clean", "BLADE_CLEAN_S"]


class BananaTidySM(PruneCutSM):
    method = "banana_tidy"

    def s_start(self, obs: Obs):
        tg = self.target
        if tg.job != "banana_tidy" or tg.green or tg.keep:
            return None, self.finish(obs.t, "skipped", "keep_wood")
        return super().s_start(obs)


def banana_marks(plan: dict) -> List[dict]:
    """Every banana MARK of one plant from its WP23 plan (``truth`` + ``pos``)."""
    tr = plan["truth"]
    pos = list(plan.get("pos", (0.0, 0.0, 0.0)))
    tree = {"tree_id": plan["tree_id"], "physical_tag": tr.get("physical_tag"), "pos": pos}
    out: List[dict] = []
    b = tr.get("bunch")
    if b:
        out.append(make_mark(tree, "bunch", b["world"], "human_cut", stage=b.get("stage"), tool_hint="human_cut",
                             target_id=b.get("geom"), h_m=b.get("h_m"),
                             extra={"ribbon_colour": b.get("ribbon_colour"), "shooting_date": b.get("shooting_date"),
                                    "harvest_window": b.get("harvest_window"), "hands_n_lead": b.get("hands_n"),
                                    "exclusion_m": b.get("exclusion_m", BANANA_EXCLUSION_M)}))
    blades = {bl["id"]: bl for bl in tr.get("blades", [])}
    prune = {p["id"]: p for p in tr.get("prune", [])}
    for lid in tr.get("sanitation_deleaf", []):
        bl = blades.get(lid, {})
        p = prune.get(bl.get("prune", ""), {})
        pt = p.get("cut_world") or [pos[0], pos[1], 1.2]
        out.append(make_mark(tree, "sanitation_deleaf", pt, "human_cut", tool_hint="sickle", target_id=lid,
                             extra={"sigatoka_pct": bl.get("sigatoka_pct"), "functional": bl.get("functional"),
                                    "cut_at": "petiole"}))
    for sid in tr.get("suckers", []):
        p = prune.get(sid)
        if not p or p.get("follower"):
            continue                                   # the follower (tallest sword) is kept
        out.append(make_mark(tree, "sucker", p["cut_world"], "human_cut", diameter_mm=p.get("diameter_mm"),
                             tool_hint="human_cut", target_id=sid, h_m=p.get("h_m"),
                             extra={"sucker_type": p.get("sucker_type"), "gouge": True}))
    return out


def tidy_targets(plan: dict) -> List[Target]:
    """T5 tidy targets of one plant: dead, in-band ``banana_tidy`` blades only."""
    tr = plan["truth"]
    pos = list(plan.get("pos", (0.0, 0.0, 0.0)))
    tree = {"tree_id": plan["tree_id"], "physical_tag": tr.get("physical_tag"), "pos": pos}
    out = []
    for p in tr.get("prune", []):
        if p.get("job") != "banana_tidy":
            continue
        out.append(Target(target_id=p["id"], tree_id=plan["tree_id"], crop="banana", task="banana_tidy", tool="T5",
                          point=tuple(p["cut_world"]), job="banana_tidy", diameter_mm=p.get("diameter_mm"),
                          keep=bool(p.get("keep")), in_band=bool(p.get("in_band", True)),
                          green=bool(p.get("green", False)),
                          meta={"tree": tree, "geom": p.get("geom"), "wp0c": (p.get("wp0c") or {}).get("id"),
                                "a": p.get("a"), "b": p.get("b")}))
    return out


def bunch_exclusion_ok(robot_xy: Sequence[float], marks: Sequence[dict], bunches_world: Dict[str, Sequence[float]],
                       in_progress: Sequence[str]) -> bool:
    """>= 3 m from every MARKed bunch whose human cut is flagged in progress (``in_progress`` = tree ids)."""
    for m in marks:
        if m.get("kind") != "bunch" or m.get("tree_id") not in in_progress:
            continue
        p = bunches_world.get(m["tree_id"])
        if p is not None and math.hypot(robot_xy[0] - p[0], robot_xy[1] - p[1]) < m.get("exclusion_m", BANANA_EXCLUSION_M):
            return False
    return True


def needs_blade_clean(prev_crop, crop, prev_tree, tree) -> bool:
    """OS 2.4 tool hygiene: a blade_clean step between banana plants and on any crop change."""
    if prev_crop is None:
        return False
    return prev_crop != crop or (crop == "banana" and prev_tree != tree)
