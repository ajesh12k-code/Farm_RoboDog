"""WP26 TRUNK/BASE SCAN: 4 azimuth stand-offs (2 alley-side ones in hedgerows), the ring low frame plus wrist
close-ups at 0.2 / 0.5 / 0.8 m, trunk-class leads per crop and season, the TreeRecord ``trunk{}`` block, and PRUNE
flags for rootstock sprouts below ``graft_union_h_m`` (MISSION-PRESETS 3b; ORCHARD-MVP-SPEC 2.5, 3.3).

ROS-free, MuJoCo-free. Leads use the SimTruth detector (assisted, labelled) on views built here; the wrist proxy
keeps the same ≥ 2 cm clearance as LEAF SCAN (MP 3b interlocks "as LEAF SCAN") from wood, sprouts and fruit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi.core.perception import leaf_scan as lsc
from atharv_nandi.core.perception import pest_detect as pdet
from atharv_nandi.sim.world import crops as crop_profiles

LEAD_UNIT = "trunk"


def trunk_geometry(plan: dict) -> Tuple[float, float]:
    """(trunk radius m, clear trunk height m): the trunk / pseudostem keep-wood capsule of the plan."""
    geoms = {g["name"]: g for g in plan["geoms"]}
    for w in plan["truth"].get("wood", []):
        if w.get("role") in ("trunk", "pseudostem"):
            g = geoms.get(w.get("geom"), {})
            if "a" in w and "b" in w:
                top = max(float(w["a"][2]), float(w["b"][2]))
            elif "fromto" in g:
                top = max(float(g["fromto"][2]), float(g["fromto"][5]))
            else:
                top = float((plan.get("entry") or {}).get("trunk_h_m") or 0.8)
            r = float(g["size"][0]) if g.get("size") else float(w.get("diameter_mm", 120.0)) / 2000.0
            return r, top
    e = plan.get("entry") or {}
    return float(e.get("trunk_r_m") or 0.06), float(e.get("trunk_h_m") or 0.8)


def is_hedgerow(plan: dict, cfg: Optional[dict] = None) -> bool:
    c = (cfg or pdet.load_config())["trunk_scan"]
    e = plan.get("entry") or {}
    return str(e.get("form")) in c["hedgerow_forms"] or "hedgerow" in str(e.get("block_id", "")).lower()


def azimuths(plan: dict, cfg: Optional[dict] = None) -> Tuple[List[float], str]:
    """View azimuths (rad, world) around the trunk and the mode ``full`` | ``two_sided``. Azimuth 0 faces the
    first alley side (``alley_sides[0]``); hedgerow trees get only the 2 alley-side azimuths."""
    c = (cfg or pdet.load_config())["trunk_scan"]
    ax, ay = pdet._alley_dir(plan.get("entry"))
    a0 = math.atan2(ay, ax)
    if is_hedgerow(plan, cfg):
        return [a0, a0 + math.pi], "two_sided"
    n = int(c["azimuths"])
    return [a0 + 2 * math.pi * k / n for k in range(n)], "full"


def wrist_view(lens, fwd, cam: dict, portrait: bool, aim) -> pdet.View:
    """Wrist camera view; ``portrait`` rolls it 90° (the landscape horizontal FOV becomes the vertical one, same
    mm/px), so one close-up covers ±0.154 m of bark height at 0.20 m."""
    w, h, fovy = int(cam["width"]), int(cam["height"]), float(cam["fovy_deg"])
    if portrait:
        fovx = 2.0 * math.degrees(math.atan(math.tan(math.radians(fovy) / 2.0) * w / h))
        w, h, fovy = h, w, fovx
    return pdet.View(pos=lens, fwd=fwd, fovy_deg=fovy, width=w, height=h, camera="wrist_cam", band="low",
                     tag={"aim": np.asarray(aim, float), "portrait": portrait})


@dataclass
class TrunkView:
    azimuth_deg: float
    h_m: float
    kind: str                   # wrist | ring_low
    view: pdet.View
    clearance_m: float
    clear_geom: str
    ok: bool = True
    reason: str = ""

    def row(self) -> dict:
        return {"azimuth_deg": round(self.azimuth_deg, 1), "h_m": round(self.h_m, 3), "kind": self.kind,
                "clearance_m": None if not math.isfinite(self.clearance_m) else round(self.clearance_m, 4),
                "clear_geom": self.clear_geom, "ok": self.ok, "reason": self.reason,
                "mm_per_px": round(self.view.mm_per_px(max(0.05, float(np.linalg.norm(self.view.pos - self.view.tag.get(
                    "aim", self.view.pos + self.view.fwd))))), 3)}


def plan_views(plan: dict, cfg: Optional[dict] = None) -> Tuple[List[TrunkView], str]:
    """Ring low frame + wrist close-ups per azimuth. Each wrist pose looks horizontally at the trunk axis from
    ``standoff_m`` off the bark; when the proxy would come closer than ``min_clearance_m`` to a leaf, wood, a sprout or a
    fruit (approach included), the stand-off grows within ``standoff_range_m`` and the view yaws ±20° before the
    view is dropped with ``no_clearance``."""
    full = cfg or pdet.load_config()
    c = full["trunk_scan"]
    wc, rc = full["cameras"]["wrist"], full["cameras"]["ring"]
    base = np.asarray(plan["pos"], float)
    r_t, h_t = trunk_geometry(plan)
    e = plan.get("entry") or {}
    canopy_r = float(e.get("canopy_r_m") or 1.0)
    obs = lsc.tree_obstacles(plan)          # MP 3b interlocks "as LEAF SCAN": leaves included (wrist-leaf >= 2 cm)
    trunk_name = next((w["geom"] for w in plan["truth"].get("wood", []) if w.get("role") in ("trunk", "pseudostem")),
                      None)
    proxy = lsc.WristProxy(**full["leaf_scan"]["wrist_proxy"])
    min_c = float(c["min_clearance_m"])
    approach = float(full["leaf_scan"]["approach_m"])
    hs = sorted({round(min(h, max(0.2, h_t - 0.02), 1.0), 3) for h in c["heights_m"]})
    azs, mode = azimuths(plan, full)
    out: List[TrunkView] = []
    for az in azs:
        d = np.array([math.cos(az), math.sin(az), 0.0])
        lens_z = float(c["ring_low"]["lens_z_m"])
        rl = base + d * (canopy_r + float(c["ring_low"]["standoff_m"]))
        rl[2] = lens_z
        aim = base + np.array([0.0, 0.0, 0.5])
        v = pdet.View(pos=rl, fwd=aim - rl, fovy_deg=rc["fovy_deg"], width=rc["width"], height=rc["height"],
                      camera="ring_low", band="low", tag={"aim": aim})
        out.append(TrunkView(math.degrees(az) % 360.0, 0.5, "ring_low", v, float("inf"), ""))
        for h in hs:
            chosen, fail = None, None
            for s in (float(c["standoff_m"]), float(c["standoff_range_m"][1]), float(c["standoff_range_m"][0])):
                for yaw in (0.0, math.radians(20.0), -math.radians(20.0)):
                    dd = np.array([math.cos(az + yaw), math.sin(az + yaw), 0.0])
                    aim = base + np.array([0.0, 0.0, h])
                    lens = aim + dd * (r_t + s)
                    fwd = -dd
                    clr, who = lsc.approach_clearance(obs, proxy, lens, fwd, approach)
                    v = wrist_view(lens, fwd, wc, bool(c.get("portrait", False)), aim)
                    tv = TrunkView(math.degrees(az) % 360.0, h, "wrist", v, clr, who)
                    if clr >= min_c:
                        chosen = tv
                        break
                    if fail is None or clr > fail.clearance_m:
                        fail = tv
                if chosen:
                    break
            if chosen is None:
                fail.ok, fail.reason = False, "no_clearance"
                chosen = fail
            out.append(chosen)
    _ = trunk_name
    return out, mode


def _covered(view: TrunkView, d: dict, cfg: dict) -> bool:
    """A bark decal is imaged by a view when its normal is within ``view_half_angle_deg`` of the view direction,
    and its centre lies in the frame."""
    c = cfg["trunk_scan"]
    ray = pdet.unit(view.view.pos - d["p_world"])
    if float(np.dot(pdet.unit(d["n_world"]), ray)) < math.cos(math.radians(float(c["view_half_angle_deg"]))):
        return False
    if not view.view.in_frame(d["p_world"]):
        return False
    return True


@dataclass
class TrunkScanResult:
    tree_id: str
    mode: str
    views: List[TrunkView] = field(default_factory=list)
    leads: List[dict] = field(default_factory=list)
    symptoms: List[dict] = field(default_factory=list)
    trunk: dict = field(default_factory=dict)
    prune_flags: List[dict] = field(default_factory=list)
    azimuths_imaged: int = 0
    time_s: float = 0.0
    log: List[dict] = field(default_factory=list)


def sprouts_below_graft(plan: dict) -> List[dict]:
    """Truth rootstock sprouts whose origin is below ``graft_union_h_m`` (the PRUNE flag candidates)."""
    gu = plan["truth"].get("graft_union_h_m") or (plan.get("entry") or {}).get("graft_union_h_m")
    out = []
    for t in plan["truth"].get("prune", []):
        if t.get("job") != "rootstock_sprout":
            continue
        z0 = min(float(t["a"][2]), float(t["b"][2]))
        if gu is None or z0 < float(gu):
            out.append(t)
    return out


def run_trunk_scan(plan: dict, profile: dict, detector: Optional[pdet.SimTruthLeadDetector] = None,
                   cfg: Optional[dict] = None, ctx: Optional[dict] = None) -> TrunkScanResult:
    """TRUNK/BASE SCAN of one tree (MP 3b steps 1-4). ``ctx``: interlocks "as LEAF SCAN"
    (:func:`leaf_scan.check_preconditions` without the leaves-in-band rule); any reason → ``mode: refused``."""
    full = cfg or pdet.load_config()
    if ctx is not None:
        why = lsc.check_preconditions(ctx, None, full)
        if why:
            return TrunkScanResult(plan["tree_id"], "refused", trunk={"status": "refused", "reasons": why})
    c = full["trunk_scan"]
    det = detector or pdet.SimTruthLeadDetector(full)
    classes = pdet.active_classes(profile, plan["season"], scale="trunk", cfg=full)
    decals = [d for d in pdet.truth_decals(plan, profile, full) if d["scale"] == "trunk"]
    views, mode = plan_views(plan, full)
    res = TrunkScanResult(plan["tree_id"], mode, views)
    guard = lsc.ClearanceGuard(float(c["min_clearance_m"]))
    az_ok: Dict[float, bool] = {}
    for tv in views:
        if tv.kind == "wrist":
            guard.reset()
            if guard.update(tv.clearance_m, who=tv.clear_geom) == "stop":
                tv.ok, tv.reason = False, "no_clearance"
        if not tv.ok:
            continue
        az_ok[round(tv.azimuth_deg, 1)] = az_ok.get(round(tv.azimuth_deg, 1), False) or tv.kind == "wrist"
        seen = [d for d in decals if _covered(tv, d, full)]
        found = det.detect(tv.view, seen, classes, plan)
        for ld in found:
            ld["azimuth_deg"] = round(tv.azimuth_deg, 1)
            ld["view_kind"] = tv.kind
        res.leads.extend(found)
    res.log = guard.log
    res.azimuths_imaged = sum(1 for v in az_ok.values() if v)
    res.symptoms = pdet.dedupe_leads(res.leads)
    keys = {k["class"]: k.get("trunk_key") for k in classes}
    trunk = {"frass": False, "gum": False, "collar_rot": False, "mud_tubes": False, "weevil_holes": False,
             "band_ok": None, "mealybug": False, "wilt": False}
    for s in res.symptoms:
        k = keys.get(s["class"])
        if k and k != "band_ok":
            trunk[k] = True
    trunk.update({"mode": mode, "azimuths_imaged": res.azimuths_imaged,
                  "azimuths_planned": len({round(v.azimuth_deg, 1) for v in views}),
                  "classes_checked": sorted(keys), "stage": "lead"})
    res.trunk = trunk
    # rootstock sprouts below the graft union, visible in an ok wrist view -> PRUNE flag (MP 3b step 4)
    base = np.asarray(plan["pos"], float)
    for t in sprouts_below_graft(plan):
        p = base + np.asarray(t["cut_p"], float)
        vis = any(v.ok and v.kind == "wrist" and v.view.in_frame(p) for v in views)
        if not vis:
            continue
        res.prune_flags.append({"tree_id": plan["tree_id"], "kind": "rootstock_sprout", "prune_id": t["id"],
                                "h_m": round(float(p[2]), 3), "h_class": crop_profiles.h_class(float(p[2])),
                                "clock": pdet.clock_of_world(plan, p), "diameter_mm": t.get("diameter_mm"),
                                "action": t.get("action"), "reason": t.get("reason"), "flag": "prune"})
    n_wrist = sum(1 for v in views if v.kind == "wrist" and v.ok)
    n_ring = sum(1 for v in views if v.kind == "ring_low" and v.ok)
    res.time_s = round(n_wrist * float(c["time_per_view_s"]) + n_ring * float(c["ring_frame_s"]), 2)
    return res


def trunk_truth_ids(plan: dict, profile: dict, cfg: Optional[dict] = None, accessible_only: bool = False
                    ) -> Dict[str, str]:
    """Active trunk-scale decal truth ``{id: class}`` in the 0-1.0 m base band. ``accessible_only``: only decals
    inside the coverage of a planned, clear view (frustum + bark angle; a hedgerow's in-row faces are not
    accessible from the alleys). The KPI reports both."""
    full = cfg or pdet.load_config()
    act = {k["class"] for k in pdet.active_classes(profile, plan["season"], scale="trunk", cfg=full)}
    out = {d["id"]: d for d in pdet.truth_decals(plan, profile, full)
           if d["scale"] == "trunk" and d["class"] in act and float(d["p_world"][2]) <= 1.0}
    if accessible_only:
        views, _ = plan_views(plan, full)
        out = {i: d for i, d in out.items() if any(v.ok and _covered(v, d, full) for v in views)}
    return {i: d["class"] for i, d in out.items()}



def run_trunk_block(plans: Sequence[dict], profiles: Dict[str, dict], n: Optional[int] = None, seed: int = 26,
                    cfg: Optional[dict] = None, ctx: Optional[dict] = None) -> dict:
    """TRUNK/BASE SCAN with ``block_sample`` (MP 3b params; OS 5.6): per crop of the block, N trees stratified by
    direction (``pest_detect.block_sampling_plan``, N from the crop unless ``n``), each scanned, then % trees
    affected per class over the trees actually scanned. Leads only; not an every-tree claim."""
    full = cfg or pdet.load_config()
    by_id = {p["tree_id"]: p for p in plans}
    entries = [{"tree_id": p["tree_id"], "crop": p["crop"], "x": float(p["pos"][0]), "y": float(p["pos"][1])}
               for p in plans]
    out = {"sampling": {}, "trees": {}, "block": {}, "prune_flags": [], "stage": "lead", "basis": "block_sample"}
    for crop in sorted({e["crop"] for e in entries}):
        sp = pdet.block_sampling_plan(entries, crop=crop, n=n, seed=seed, cfg=full)
        rows = []
        for tid in sp["tree_ids"]:
            r = run_trunk_scan(by_id[tid], profiles[crop], cfg=full, ctx=ctx)
            out["trees"][tid] = r.trunk
            out["prune_flags"].extend(r.prune_flags)
            rows.append({"tree_id": tid, "symptoms": r.symptoms})
        out["sampling"][crop] = sp
        out["block"][crop] = pdet.block_summary(rows, sp)
    return out


__all__ = ["TrunkScanResult", "TrunkView", "azimuths", "is_hedgerow", "plan_views", "run_trunk_block", "run_trunk_scan",
           "sprouts_below_graft", "trunk_geometry", "trunk_truth_ids", "Sequence"]
