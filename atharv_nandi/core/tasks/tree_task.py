"""Per-tree job planning for HARVEST / PICK and CLIP / PRUNE (MISSION-PRESETS 5, 6). Pure: reads a WP23 tree plan
(``world.trees`` output: ``truth`` + ``pos``) and the crop profile (``world.crops``), returns the contact targets in
execution order and the MARKs / skips that need no arm move.

* **Mango, one round:** every mature, in-band, unbagged fruit of the tree is a CLIP target (the pool's
  ``assign_tree`` default); mature fruit above the band are MARKed ``out_of_reach_marked`` (tool_hint pole_picker);
  bagged fruit are refused (``bagged``, no pool slot); immature fruit are left alone.
* **Guava:** CLIP (default) or TWIST (``method="twist"``), one per cell.
* **Papaya:** TWIST, one per trip: the lowest colour-break in-band fruit that passes ``jaw_clearance``; the rest of the
  ripe in-band fruit that pass wait for the next trip (``payload_full``); no-clearance fruit are MARKed.
* **Banana:** MARK only (bunch, de-leaf, de-sucker) plus the T5 tidy of dead in-band blades (``banana_tidy.py``).
* **Prune jobs:** in-band candidates <= ``max_stem_mm`` are cut, the rest MARKed with ``diameter_mm`` and
  ``paste_required``; keep wood is never a target. Order: low to high (debris falls clear of the next target).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from atharv_nandi.core.tasks import banana_tidy
from atharv_nandi.core.tasks.common import (PICK_BAND_Z_M, Target, in_pick_band, jaw_clearance_ok, make_mark)
from atharv_nandi.core.tasks.prune_cut import tool_hint_for

PRUNE_JOBS = ("rootstock_sprout", "skirt_lift", "dead_twig", "malformed_panicle", "water_shoot", "tip_back_20cm",
              "banana_tidy", "banana_sanitation_deleaf", "banana_desucker")
TASK_OF_METHOD = {"clip": ("clip_pedicel", "T2"), "twist": ("twist_pick", "T4")}


@dataclass
class TreeJob:
    tree_id: str
    crop: str
    preset: str                                     # harvest | prune
    method: str
    targets: List[Target] = field(default_factory=list)
    marks: List[dict] = field(default_factory=list)
    skipped: List[dict] = field(default_factory=list)   # {target_id, reason} with no arm move
    deferred: List[dict] = field(default_factory=list)  # next trip (papaya one per trip)

    def summary(self) -> dict:
        return {"tree_id": self.tree_id, "crop": self.crop, "preset": self.preset, "method": self.method,
                "targets": [t.target_id for t in self.targets], "marks": len(self.marks),
                "skipped": list(self.skipped), "deferred": list(self.deferred)}


def _tree(plan: dict) -> dict:
    return {"tree_id": plan["tree_id"], "physical_tag": plan["truth"].get("physical_tag"),
            "pos": list(plan.get("pos", (0.0, 0.0, 0.0)))}


def _pitch_from_axis(axis) -> float:
    if not axis:
        return 0.0
    return -math.asin(max(-1.0, min(1.0, float(axis[2]))))


def _station(stations: Optional[Dict[str, dict]], wp0c_id: Optional[str]) -> Optional[dict]:
    if not stations or not wp0c_id:
        return None
    return stations.get(wp0c_id)


def harvest_job(plan: dict, profile: dict, method: Optional[str] = None, max_items: Optional[int] = None,
                stations: Optional[Dict[str, dict]] = None, payload_free_cells: int = 8,
                cradle_free: bool = True, exclude: Sequence[str] = ()) -> TreeJob:
    """HARVEST / PICK targets for one tree (MP 5). ``stations`` maps WP0c target ids to ``{station, axis_r, ...}``
    (WP0c frozen targets) so each target carries its stand-off station and approach pitch. ``exclude`` = fruit
    already harvested (the pool's ``harvested`` list): a later round or trip never targets them again."""
    from atharv_nandi.sim.world import crops as crops_mod
    crop = plan["crop"]
    m = method or profile["harvest_method"]
    tree = _tree(plan)
    job = TreeJob(plan["tree_id"], crop, "harvest", m)
    if m == "mark" or crop == "banana":
        job.marks.extend(banana_tidy.banana_marks(plan) if crop == "banana" else [])
        return job
    if m not in profile["methods"]:
        job.skipped.append({"target_id": plan["tree_id"], "reason": "not_in_mvp"})
        return job
    task, tool = TASK_OF_METHOD[m]
    fruit = sorted(plan["truth"].get("fruit", []), key=lambda f: (f.get("target_z_m", f["centre"][2]), f["fruit_id"]))
    picks: List[Target] = []
    gone = set(exclude)
    for f in fruit:
        if f["fruit_id"] in gone:
            continue
        decision, reason = crops_mod.harvest_decision(profile, f, m)
        if f.get("bagged"):
            job.skipped.append({"target_id": f["fruit_id"], "reason": "bagged"})
            continue
        if decision == "skip":
            if reason:
                job.skipped.append({"target_id": f["fruit_id"], "reason": reason})
            continue
        point = f["cut_k_world"] if task == "clip_pedicel" else f["centre_world"]
        if decision == "mark":
            job.marks.append(make_mark(tree, "fruit", point, reason, stage=f.get("stage"),
                                       tool_hint="pole_picker" if reason == "out_of_reach_marked" else "human_cut",
                                       target_id=f["fruit_id"], clock=f.get("clock"),
                                       extra={"clearance_mm": f.get("clearance_mm")} if reason == "no_clearance" else None))
            continue
        wp0c = (f.get("wp0c") or {}).get("id")
        st = _station(stations, wp0c)
        picks.append(Target(
            target_id=f["fruit_id"], tree_id=plan["tree_id"], crop=crop, task=task, tool=tool, point=tuple(point),
            stage=f.get("stage"), mass_kg=float(f.get("mass_kg", 0.0)), clearance_mm=f.get("clearance_mm"),
            bagged=bool(f.get("bagged")), in_band=bool(f.get("in_band", True)),
            pedicel_leave_m=float(f.get("pedicel_leave_m", 0.0)),
            station=(st or {}).get("station"), pitch=_pitch_from_axis((st or {}).get("axis_r")),
            meta={"tree": tree, "fruit_id": f["fruit_id"], "harvest_stage": bool(f.get("harvest_stage")),
                  "pedicel": [float(a) - float(b) for a, b in zip(f["anchor_top_world"], f["stem_end_world"])],
                  "wp0c": wp0c, "pool_kind": f.get("pool_kind"), "clock": f.get("clock")}))
    if crop == "papaya":
        # one per trip: the lowest passing fruit now, the rest of the passing ones on the next trip
        ok = [t for t in picks if jaw_clearance_ok(t.clearance_mm, profile.get("jaw_clearance_mm", 15))]
        if not cradle_free:
            job.deferred.extend({"target_id": t.target_id, "reason": "payload_full"} for t in ok)
            ok = []
        job.targets = ok[:1]
        job.deferred.extend({"target_id": t.target_id, "reason": "payload_full"} for t in ok[1:])
        return job
    room = max(0, int(payload_free_cells))
    if max_items is not None:
        room = min(room, int(max_items))
    job.targets = picks[:room]
    job.deferred.extend({"target_id": t.target_id, "reason": "payload_full"} for t in picks[room:])
    return job


def prune_job(plan: dict, profile: dict, jobs: Optional[Sequence[str]] = None,
              stations: Optional[Dict[str, dict]] = None) -> TreeJob:
    """CLIP / PRUNE targets for one tree (MP 6). ``jobs`` = the operator's job checkboxes (default: the profile's)."""
    from atharv_nandi.sim.world import crops as crops_mod
    crop = plan["crop"]
    tree = _tree(plan)
    job = TreeJob(plan["tree_id"], crop, "prune", "prune")
    if crop == "banana":
        for t in banana_tidy.tidy_targets(plan):
            if t.green or t.keep:
                job.skipped.append({"target_id": t.target_id, "reason": "keep_wood"})
                continue
            st = _station(stations, t.meta.get("wp0c"))
            t.station = (st or {}).get("station")
            t.pitch = _pitch_from_axis((st or {}).get("axis_r"))
            if not in_pick_band(t.point[2]):
                job.skipped.append({"target_id": t.target_id, "reason": "out_of_reach_marked"})
                continue
            job.targets.append(t)
        job.marks.extend(m for m in banana_tidy.banana_marks(plan) if m["kind"] != "bunch")
        return job
    wanted = set(jobs if jobs is not None else profile["prune"].get("jobs", []))
    max_mm = float(profile["prune"].get("max_stem_mm", 12))
    cands = sorted(plan["truth"].get("prune", []), key=lambda p: (p.get("z_m", p["cut_world"][2]), p["id"]))
    for p in cands:
        if p.get("job") not in wanted:
            continue
        if p.get("keep"):
            job.skipped.append({"target_id": p["id"], "reason": "keep_wood"})
            continue
        z = float(p.get("z_m", p["cut_world"][2]))
        decision, reason = crops_mod.prune_decision(profile, p["job"], float(p["diameter_mm"]), z, keep=bool(p.get("keep")))
        if decision == "skip":
            job.skipped.append({"target_id": p["id"], "reason": reason or "keep_wood"})
            continue
        kind = p["job"] if p["job"] in ("rootstock_sprout", "skirt_lift", "malformed_panicle") else "prune"
        if decision == "mark" or z > PICK_BAND_Z_M or float(p["diameter_mm"]) > max_mm:
            # OS 5.3 / MP 6: over the PRUNER capacity -> MARK reason ``over_capacity``; ``paste_required`` (> 12 mm)
            # is the MARK flag make_mark sets from diameter_mm, never the reason
            if z > PICK_BAND_Z_M:
                reason = "out_of_reach_marked"
            elif reason in ("", "paste_required"):
                reason = "over_capacity"
            job.marks.append(make_mark(tree, kind, p["cut_world"], reason,
                                       diameter_mm=p["diameter_mm"], tool_hint=tool_hint_for(p["diameter_mm"], z),
                                       target_id=p["id"], clock=p.get("clock"), extra={"job": p["job"]}))
            continue
        wp0c = (p.get("wp0c") or {}).get("id")
        st = _station(stations, wp0c)
        job.targets.append(Target(
            target_id=p["id"], tree_id=plan["tree_id"], crop=crop, task="prune_cut", tool="T5",
            point=tuple(p["cut_world"]), job=p["job"], diameter_mm=float(p["diameter_mm"]), keep=False,
            in_band=bool(p.get("in_band", True)), station=(st or {}).get("station"),
            pitch=_pitch_from_axis((st or {}).get("axis_r")),
            meta={"tree": tree, "geom": p.get("geom"), "wp0c": wp0c, "a": p.get("a"), "b": p.get("b"),
                  "below_graft_union": p.get("below_graft_union")}))
    return job


def frozen_stations(candidates: dict) -> Dict[str, dict]:
    """WP0c ``orchard_candidates.json`` frozen targets by id (those with a station)."""
    return {t["id"]: t for t in candidates.get("frozen", {}).get("targets", []) if t.get("station")}
