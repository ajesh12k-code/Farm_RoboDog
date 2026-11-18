"""WP26 FRUIT SCOUT: fruit and bag detections lifted into the tree frame, tracked across views and counted once,
stage per crop profile, size, reachable vs MARK, est. kg and the harvest window (MISSION-PRESETS 4;
ORCHARD-MVP-SPEC 4.4, 5.6).

ROS-free, MuJoCo-free.

* **Views** (:func:`orbit_views`): the TREE 360 stations (8 by default, CCW at r1 = canopy_r + 0.9 m, clamped
  to 1.2-3.0 m) with the low (ring 0.98 m), mid (crown 1.70 m) and high (crown 2.70 m) band frames aimed at the
  tree; ``source: one_lap`` / ``reuse_tree360`` use the same geometry.
* **Detector** (:class:`SimTruthFruitDetector`, assisted, labelled): a static fruit or bag is detectable in a
  frame when it is in the frustum, within range, faces the camera (outward from the trunk axis), spans ≥ 10 px
  and is not occluded (seeded canopy self-occlusion); then reported with ``p_detect``, a 1 px bbox-centre error
  and a range-dependent depth error, plus false positives. Detections are lifted with depth into the world, then
  into the **tree frame** (world minus the tree position; trees are not rotated).
* **Tracker** (:class:`TreeFrameTracker`): greedy one-to-one association per frame to the nearest track mean of
  the same kind within ``dedupe_radius_m`` (0.04 m); tracks closer than ``merge_radius_m`` merge at the end; a
  track counts once with ≥ ``min_hits`` hits (single-view tracks only when the run had < 3 frames).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi.core.perception import pest_detect as pdet
from atharv_nandi.core.perception import ripeness as rp
from atharv_nandi.sim.world import crops as cp
from atharv_nandi.sim.world import trees as gen

BANDS = (("low", 0.98, 0.6), ("mid", 1.70, None), ("high", 2.70, None))   # (band, lens z, aim z or None = auto)


def _fcfg(cfg: Optional[dict]) -> dict:
    return (cfg or pdet.load_config())["fruit_track"]


# ---------------------------------------------------------------------------------------------- views
def orbit_views(plan: dict, n_views: Optional[int] = None, cfg: Optional[dict] = None,
                bands: Sequence[str] = ("low", "mid", "high")) -> List[pdet.View]:
    """Station frames around the tree: ``n_views`` stations × the requested bands."""
    full = cfg or pdet.load_config()
    n = int(n_views if n_views is not None else full["fruit_track"]["n_views"])
    base = np.asarray(plan["pos"], float)
    e = plan.get("entry") or {}
    cr = float(e.get("canopy_r_m") or 1.0)
    H = float(e.get("height_m") or 2.2)
    r1 = min(3.0, max(1.2, cr + 0.9))
    ax, ay = pdet._alley_dir(e)
    a0 = math.atan2(ay, ax)
    out = []
    for k in range(n):
        az = a0 + 2.0 * math.pi * k / n
        d = np.array([math.cos(az), math.sin(az), 0.0])
        for name, lens_z, aim_z in BANDS:
            if name not in bands:
                continue
            cam = full["cameras"]["ring" if name == "low" else "crown"]
            lens = base + d * r1
            lens[2] = lens_z
            az_aim = aim_z if aim_z is not None else min(lens_z - 0.25, 0.75 * H) if name == "mid" else min(lens_z - 0.6, H - 0.3)
            aim = base + np.array([0.0, 0.0, az_aim])
            out.append(pdet.View(pos=lens, fwd=aim - lens, fovy_deg=cam["fovy_deg"], width=cam["width"],
                                 height=cam["height"], camera="ring" if name == "low" else "crown_cam", band=name,
                                 tag={"station": k, "azimuth_deg": round(math.degrees(az) % 360.0, 1)}))
    return out


# ---------------------------------------------------------------------------------------------- detector
def _hash_u(*parts) -> float:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") / float(1 << 64)


@dataclass
class FruitDetection:
    frame: int
    kind: str                     # fruit | bag
    p_tree: np.ndarray
    p_world: np.ndarray
    length_mm: float
    band: str
    view_band: str
    conf: float
    cues: dict
    rgb: List[float]
    truth: Optional[str] = None   # assisted (scoring only)
    origin: Optional[np.ndarray] = None   # camera centre, tree frame
    ray: Optional[np.ndarray] = None      # unit viewing ray through the bbox centre
    depth_m: float = 0.0                  # measured range along the ray
    sigma_lat_m: float = 0.01
    sigma_depth_m: float = 0.02


class SimTruthFruitDetector:
    name = "simtruth"

    def __init__(self, profile: dict, cfg: Optional[dict] = None, seed: Optional[int] = None):
        self.full = cfg or pdet.load_config()
        self.c = self.full["fruit_track"]
        self.profile = profile
        self.seed = int(self.c["seed"] if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

    def geometric_visible(self, view: pdet.View, f: dict, plan: dict, frame: int) -> bool:
        p = np.asarray(f["centre_world"], float)
        if not view.in_frame(p):
            return False
        rng_m = float(np.linalg.norm(view.pos - p))
        if rng_m > float(self.c["max_range_m"]):
            return False
        base = np.asarray(plan["pos"], float)
        out = np.array([p[0] - base[0], p[1] - base[1], 0.0])
        out = pdet.unit(out) if np.linalg.norm(out) > 1e-6 else -view.fwd
        if float(np.dot(out, pdet.unit(view.pos - p))) < float(self.c["facing_cos_min"]):
            return False
        px = 1000.0 * float(f["length_mm"]) / 1000.0 / view.mm_per_px(rng_m)
        if px < float(self.c["min_px"]):
            return False
        return _hash_u(self.seed, f["fruit_id"], view.tag.get("station"), view.band, "occ") >= float(self.c["occlusion_p"])

    def detect(self, view: pdet.View, plan: dict, frame: int) -> List[FruitDetection]:
        c, rng = self.c, self.rng
        base = np.asarray(plan["pos"], float)
        out: List[FruitDetection] = []
        hidden = set(plan.get("_hidden_fruit", ()))
        for f in plan["truth"]["fruit"]:
            if f["fruit_id"] in hidden or not self.geometric_visible(view, f, plan, frame):
                continue
            if rng.uniform() > float(c["p_detect"]):
                continue
            p = np.asarray(f["centre_world"], float)
            u, v, z = view.project(p)
            u += rng.normal(0, c["lateral_sigma_px"])
            v += rng.normal(0, c["lateral_sigma_px"])
            rng_m = float(np.linalg.norm(view.pos - p))
            z_obs = z + rng.normal(0, c["depth_sigma_m"][0] + c["depth_sigma_m"][1] * rng_m)
            pw = view.unproject(u, v, z_obs)
            sig_d = float(c["depth_sigma_m"][0] + c["depth_sigma_m"][1] * rng_m)
            low = f["band"] == "low" and view.band == "low"
            sig = c["size_sigma_mm_low"] if low else c["size_sigma_mm_coarse"]
            out.append(FruitDetection(frame, "bag" if f.get("bagged") else "fruit", pw - base, pw,
                                      float(f["length_mm"]) + float(rng.normal(0, sig)), gen.band_of(float(pw[2])),
                                      view.band, float(np.clip(0.9 - 0.1 * rng_m + rng.normal(0, 0.03), 0.05, 1.0)),
                                      {} if f.get("bagged") else rp.observe_cues(f, self.profile, rng, self.full),
                                      list(f.get("rgba", [0, 0, 0, 1]))[:3], f["fruit_id"]))
            _attach_ray(out[-1], view, base, sig_d, float(c["lateral_sigma_px"]))
        for _ in range(int(rng.poisson(float(c["fp_per_view"])))):
            u, v = rng.uniform(0, view.width), rng.uniform(0, view.height)
            depth = float(np.linalg.norm(view.pos - base)) - float(rng.uniform(0.2, 1.0))
            pw = view.unproject(u, v, max(0.3, depth))
            out.append(FruitDetection(frame, "fruit", pw - base, pw, float(rng.uniform(60, 110)),
                                      gen.band_of(float(pw[2])), view.band, float(rng.uniform(0.2, 0.5)), {},
                                      [0.4, 0.5, 0.2], None))
            _attach_ray(out[-1], view, base, 0.02, float(c["lateral_sigma_px"]))
        return out


# ---------------------------------------------------------------------------------------------- tracker
def _attach_ray(d: FruitDetection, view: pdet.View, base: np.ndarray, sigma_depth_m: float, sigma_px: float):
    o = view.pos - base
    ray = pdet.unit(d.p_tree - o)
    d.origin, d.ray = o, ray
    d.depth_m = float(np.linalg.norm(d.p_tree - o))
    d.sigma_depth_m = float(sigma_depth_m)
    d.sigma_lat_m = max(1e-4, sigma_px * view.mm_per_px(d.depth_m) / 1000.0)


def det_cov(d: FruitDetection) -> np.ndarray:
    if d.ray is None:
        return np.eye(3) * d.sigma_depth_m ** 2
    uu = np.outer(d.ray, d.ray)
    return d.sigma_lat_m ** 2 * (np.eye(3) - uu) + d.sigma_depth_m ** 2 * uu


def triangulate(dets: Sequence[FruitDetection], with_cov: bool = False):
    """Least-squares point from rays with anisotropic noise: lateral σ (pixel error × range) across each ray,
    depth σ along it (information-weighted). One detection → its own lifted point. ``with_cov``: also the 3×3
    covariance of the estimate."""
    if any(d.ray is None for d in dets):
        m = np.mean([d.p_tree for d in dets], axis=0)
        return (m, np.eye(3) * 0.02 ** 2) if with_cov else m
    A, b = np.zeros((3, 3)), np.zeros(3)
    for d in dets:
        uu = np.outer(d.ray, d.ray)
        P = np.eye(3) - uu
        wl, wd = 1.0 / d.sigma_lat_m ** 2, 1.0 / d.sigma_depth_m ** 2
        A += wl * P + wd * uu
        b += wl * (P @ d.origin) + wd * (uu @ (d.origin + d.depth_m * d.ray))
    m = np.linalg.solve(A, b)
    return (m, np.linalg.inv(A)) if with_cov else m


@dataclass
class Track:
    tid: int
    kind: str
    dets: List[FruitDetection] = field(default_factory=list)
    _mean: Optional[np.ndarray] = None
    _cov: Optional[np.ndarray] = None

    @property
    def hits(self) -> int:
        return len(self.dets)

    @property
    def frames(self) -> List[int]:
        return [d.frame for d in self.dets]

    @property
    def mean(self) -> np.ndarray:
        if self._mean is None:
            self._mean, self._cov = triangulate(self.dets, with_cov=True)
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        _ = self.mean
        return self._cov

    @property
    def conf(self) -> float:
        return max(d.conf for d in self.dets)

    @property
    def cues(self) -> List[dict]:
        return [d.cues for d in self.dets if d.cues]

    @property
    def rgbs(self) -> List[List[float]]:
        return [d.rgb for d in self.dets]

    def add(self, d: FruitDetection):
        self.dets.append(d)
        self._mean = self._cov = None

    def merge(self, o: "Track"):
        self.dets += o.dets
        self._mean = self._cov = None

    def truth(self) -> Optional[str]:
        c: Dict[str, int] = {}
        for d in self.dets:
            if d.truth:
                c[d.truth] = c.get(d.truth, 0) + 1
        return max(c, key=lambda k: c[k]) if c else None

    def length_mm(self) -> Tuple[float, str]:
        low = [d.length_mm for d in self.dets if d.view_band == "low" and d.band == "low"]
        if low:
            return float(np.median(low)), "depth"
        return float(np.median([d.length_mm for d in self.dets])), "coarse"

    def gate(self, d: FruitDetection, radius: float, chi2: float = 16.27) -> Optional[float]:
        """Association cost of ``d`` to this track (squared Mahalanobis distance of the innovation under track +
        detection covariance), or None outside the gate. Inside the gate: Euclidean distance ≤ ``radius`` (the
        MP 4 ``dedupe_radius_m``) **or** Mahalanobis² ≤ ``chi2`` (99.9 %, 3 dof), so the depth error of a single
        far view (σ ≈ 1.4 cm at 1.7 m) does not split one fruit into two tracks."""
        m = self.mean
        innov = m - d.p_tree
        S = self.cov + det_cov(d)
        m2 = float(innov @ np.linalg.solve(S, innov))
        if float(np.linalg.norm(innov)) <= radius or m2 <= chi2:
            return m2
        return None


class TreeFrameTracker:
    """Tree-frame multi-view fruit tracker (count once): greedy one-to-one association per frame against
    triangulated track estimates, end-of-tree merge of tracks with disjoint frames, ``min_hits`` filter."""

    def __init__(self, dedupe_radius_m: float = 0.04, merge_radius_m: float = 0.04, min_hits: int = 2,
                 single_hit_conf_min: float = 1.01):
        self.r, self.mr, self.min_hits = float(dedupe_radius_m), float(merge_radius_m), int(min_hits)
        self.single_conf = float(single_hit_conf_min)
        self.tracks: List[Track] = []
        self.frames = 0

    def update(self, dets: Sequence[FruitDetection]):
        self.frames += 1
        pairs = []
        for i, d in enumerate(dets):
            for j, t in enumerate(self.tracks):
                if t.kind != d.kind:
                    continue
                cost = t.gate(d, self.r)
                if cost is not None:
                    pairs.append((cost, i, j))
        used_d, used_t = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_d or j in used_t:
                continue
            self.tracks[j].add(dets[i])
            used_d.add(i)
            used_t.add(j)
        for i, d in enumerate(dets):
            if i not in used_d:
                self.tracks.append(Track(len(self.tracks), d.kind, [d]))

    def _same(self, ta: Track, tb: Track, chi2: float = 16.27) -> bool:
        """Merge test: estimates within ``merge_radius_m`` or Mahalanobis² ≤ chi2 under both covariances."""
        dv = ta.mean - tb.mean
        if float(np.linalg.norm(dv)) <= self.mr:
            return True
        return float(dv @ np.linalg.solve(ta.cov + tb.cov, dv)) <= chi2

    def finalize(self) -> List[Track]:
        tracks = list(self.tracks)
        merged = True
        while merged:
            merged = False
            for a in range(len(tracks)):
                for b in range(a + 1, len(tracks)):
                    ta, tb = tracks[a], tracks[b]
                    if ta.kind == tb.kind and not set(ta.frames) & set(tb.frames) and self._same(ta, tb):
                        ta.merge(tb)
                        tracks.pop(b)
                        merged = True
                        break
                if merged:
                    break
        need = self.min_hits if self.frames >= 3 else 1
        return [t for t in tracks if t.hits >= need or t.conf >= self.single_conf]


# ---------------------------------------------------------------------------------------------- scout
@dataclass
class ScoutResult:
    tree_id: str
    fruit: dict
    tracks: List[Track]
    marks: List[dict]
    assisted: dict
    frames: int


def _est_kg(profile: dict, length_mm: float, mid_len_mm: float) -> float:
    lo, hi = profile.get("mass_kg", [0.2, 0.3])
    return float((lo + hi) / 2.0) * (length_mm / max(mid_len_mm, 1.0)) ** 3


def visible_truth(plan: dict, views: Sequence[pdet.View], det: SimTruthFruitDetector) -> List[str]:
    """Truth fruit / bags geometrically visible (facing, unoccluded, ≥ min px) in at least one frame."""
    return [f["fruit_id"] for f in plan["truth"]["fruit"]
            if any(det.geometric_visible(v, f, plan, i) for i, v in enumerate(views))]


def scout_tree(plan: dict, profile: dict, today=None, n_views: Optional[int] = None, seed: Optional[int] = None,
               source: str = "one_lap", rota: Optional[Dict[str, str]] = None, cfg: Optional[dict] = None) -> ScoutResult:
    """FRUIT SCOUT of one tree (MP 4 steps 1-7)."""
    full = cfg or pdet.load_config()
    c = full["fruit_track"]
    entry = plan.get("entry") or {}
    det = SimTruthFruitDetector(profile, full, seed)
    views = orbit_views(plan, n_views, full)
    trk = TreeFrameTracker(c["dedupe_radius_m"], c["merge_radius_m"], c["min_hits"], c.get("single_hit_conf_min", 1.01))
    for i, v in enumerate(views):
        trk.update(det.detect(v, plan, i))
    tracks = trk.finalize()
    hs = set(cp.harvest_stages(profile))
    days = rp.days_since(entry, profile, today)
    edges = list(c["size_hist_edges_mm"])
    lens = [f["length_mm"] for f in plan["truth"]["fruit"]] or [80.0]
    mid_len = float(np.median(lens))
    by_stage: Dict[str, int] = {}
    hist = [0] * (len(edges) - 1)
    reach_n, kg, bags_n = 0, 0.0, 0
    marks: List[dict] = []
    stage_hits = stage_n = cb_hits = 0
    size_err: List[float] = []
    truth_by_id = {f["fruit_id"]: f for f in plan["truth"]["fruit"]}
    leave_m = float((profile.get("pedicel_leave_cm") or {}).get("default", 2.0)) / 100.0 \
        if isinstance(profile.get("pedicel_leave_cm"), dict) else 0.02
    rows = []
    for t in tracks:
        L, size_basis = t.length_mm()
        p = t.mean
        if t.kind == "bag":
            bags_n += 1
            st = cp.stage_for_age(profile, days) if days is not None else sorted(hs)[0] if hs else profile["stages"][0]["id"]
            basis = "calendar"
        else:
            st, _, basis = rp.classify(profile, rp.fuse_cues(t.cues), days, full)
        by_stage[st] = by_stage.get(st, 0) + 1
        for b in range(len(edges) - 1):
            if edges[b] <= L < edges[b + 1]:
                hist[b] += 1
        target_z = float(p[2]) + (L / 2000.0 + leave_m if profile["harvest_method"] == "clip" else 0.0)
        reachable = target_z <= cp.PICK_BAND_Z_M and t.kind == "fruit"
        reach_n += int(reachable)
        kg += _est_kg(profile, L, mid_len)
        tf = truth_by_id.get(t.truth() or "")
        if tf is not None and t.kind == "fruit":
            stage_n += 1
            stage_hits += int(st == tf["stage"])
            cb_hits += int(rp.colorblob_stage(profile, np.mean(t.rgbs, axis=0), det.rng, full) == tf["stage"])
            if size_basis == "depth":
                size_err.append(abs(L - float(tf["length_mm"])))
        row = {"track": t.tid, "kind": t.kind, "p_tree": [round(float(x), 3) for x in p], "stage": st,
               "stage_basis": basis, "length_mm": round(L, 1), "size_basis": size_basis, "hits": t.hits,
               "reachable": reachable, "conf": round(t.conf, 3)}
        rows.append(row)
        if t.kind == "fruit" and st in hs and not reachable and profile["mvp_scope"] in ("harvest", "scout_mark"):
            pw = np.asarray(plan["pos"], float) + p
            marks.append({"schema": "nandi.mark/1", "tree_id": plan["tree_id"],
                          "physical_tag": entry.get("physical_tag"), "kind": "fruit",
                          "clock": pdet.clock_of_world(plan, pw), "side": _side(plan, pw),
                          "h_class": cp.h_class(float(pw[2])), "h_m": round(float(pw[2]), 3), "stage": st,
                          "reason": "out_of_reach_marked", "diameter_mm": None, "paste_required": False,
                          "tool_hint": "pole_picker"})
    stage_counts = dict(by_stage)
    window = rp.tree_window(profile, entry, stage_counts, today)
    vis = visible_truth(plan, views, det)
    visible_n = sum(1 for t in tracks)
    fruit = {"visible_by_stage": stage_counts, "bags_n": bags_n,
             "size_mm_hist": {"edges": edges, "counts": hist}, "est_kg": round(kg, 2),
             "est_kg_label": "estimate", "reachable_n": reach_n, "visible_n": visible_n,
             "method": f"simtruth_track:{source}", "visible_fraction": round(len(vis) / max(1, len(plan["truth"]["fruit"])), 3),
             "count_label": "estimate",   # MP 4: canopy occlusion -> the visible count is labelled an estimate
             "harvest_window": window}
    if bags_n:
        fruit["harvest"] = {"refused": True, "reason": rp.HARVEST_REFUSED_BAGGED}
    n_vis = len(vis)
    assisted = {"truth_visible_n": n_vis, "truth_total_n": len(plan["truth"]["fruit"]),
                "count_error": round(abs(visible_n - n_vis) / n_vis, 4) if n_vis else (0.0 if visible_n == 0 else 1.0),
                "stage_acc": round(stage_hits / stage_n, 4) if stage_n else None, "stage_n": stage_n,
                "colorblob_stage_acc": round(cb_hits / stage_n, 4) if stage_n else None,
                "size_mae_mm_low": round(float(np.mean(size_err)), 2) if size_err else None,
                "size_n_low": len(size_err), "window_truth": plan["truth"].get("harvest_window"),
                "window_equal": rp.window_equals(window, plan["truth"].get("harvest_window")), "rows": rows,
                "label": "assisted (SimTruth)"}
    return ScoutResult(plan["tree_id"], fruit, tracks, marks, assisted, trk.frames)


def _side(plan: dict, pw) -> str:
    d = np.asarray(pw, float) - np.asarray(plan["pos"], float)
    ang = math.degrees(math.atan2(d[1], d[0])) % 360.0
    return "E" if ang < 45 or ang >= 315 else "N" if ang < 135 else "W" if ang < 225 else "S"


def scout_banana(plan: dict, profile: dict, today=None, rota: Optional[Dict[str, str]] = None,
                 seed: int = 26, cfg: Optional[dict] = None) -> ScoutResult:
    """Banana FRUIT SCOUT: the static bunch; age from the ribbon (registry shooting date or the farm rota), finger
    angularity secondary; MARK only (human cut, 3 m exclusion)."""
    full = cfg or pdet.load_config()
    b = plan["truth"].get("bunch") or {}
    entry = rp.banana_entry(plan.get("entry") or {}, {"ribbon_colour": b.get("ribbon_colour")}, rota)
    rng = np.random.default_rng(seed)
    days = rp.days_since(entry, profile, today)
    st_truth = b.get("stage")
    fingers = (cp.stage(profile, st_truth).get("cues") or {}).get("fingers") if st_truth else None
    obs = rp.observe_cues({"cues": {"fingers": fingers} if fingers else {}}, profile, rng, full)
    st, conf, basis = rp.classify(profile, obs, days, full)
    window = rp.tree_window(profile, entry, {st: 1}, today)
    fruit = {"visible_by_stage": {st: 1} if b else {}, "bags_n": 0, "size_mm_hist": None,
             "est_kg": b.get("mass_kg"), "est_kg_label": "estimate", "reachable_n": 0, "visible_n": 1 if b else 0,
             "method": "bunch_ribbon", "visible_fraction": 1.0 if b else 0.0, "count_label": "count",
             "harvest_window": window}
    marks = []
    if b:
        pw = np.asarray(b["world"], float)
        marks.append({"schema": "nandi.mark/1", "tree_id": plan["tree_id"], "physical_tag": entry.get("physical_tag"),
                      "kind": "bunch", "clock": pdet.clock_of_world(plan, pw), "side": _side(plan, pw),
                      "h_class": cp.h_class(float(pw[2])), "h_m": round(float(pw[2]), 3), "stage": st,
                      "reason": "human_cut", "tool_hint": "human_cut", "exclusion_m": b.get("exclusion_m", 3.0)})
    assisted = {"stage_truth": st_truth, "stage_ok": st == st_truth, "window_truth": b.get("harvest_window"),
                "window_equal": rp.window_equals(window, b.get("harvest_window")), "label": "assisted (SimTruth)",
                "stage_basis": basis}
    return ScoutResult(plan["tree_id"], fruit, [], marks, assisted, 0)


SOURCES = ("reuse_tree360", "one_lap", "scan_bursts")


def frames_precondition(source: str = "one_lap", frames_age_h: Optional[float] = None,
                        crown_gate_ok: Optional[bool] = None, cfg: Optional[dict] = None) -> Tuple[str, Optional[str]]:
    """MP 4 precondition: reused frames ≤ ``frames_max_age_h`` (24 h) old, or the crown gate green for one lap.
    Returns ``(source to use, refusal reason or None)``: stale reused frames fall back to ``one_lap`` when the
    crown gate is green, else ``frames_stale``; a lap with the crown gate red is ``crown_gate``.

    Fail closed: only ``crown_gate_ok is True`` is green (``None`` = unknown → ``crown_gate_unknown``), and a
    reused-frame age must be a finite number ≤ the limit (``None`` / NaN / inf = unknown → treated as stale)."""
    c = _fcfg(cfg)
    if source not in SOURCES:
        return source, "bad_source"
    gate_green = crown_gate_ok is True
    gate_reason = "crown_gate" if crown_gate_ok is False else "crown_gate_unknown"
    if source in ("reuse_tree360", "scan_bursts"):
        try:
            age = float(frames_age_h) if frames_age_h is not None and not isinstance(frames_age_h, bool) else None
        except (TypeError, ValueError):
            age = None
        fresh = age is not None and math.isfinite(age) and 0.0 <= age <= float(c["frames_max_age_h"])
        if fresh:
            return source, None
        if not gate_green:
            return source, ("frames_stale" if age is not None and math.isfinite(age) else "frames_age_unknown")
        return "one_lap", None
    if not gate_green:
        return source, gate_reason
    return source, None


def scout(plan: dict, profile: dict, today=None, frames_age_h: Optional[float] = None,
          crown_gate_ok: Optional[bool] = None, **kw) -> ScoutResult:
    """FRUIT SCOUT of one tree with the MP 4 frame precondition (:func:`frames_precondition`); a refusal returns
    ``fruit{status: refused, reason}`` and no frames. ``crown_gate_ok`` must be given as True for a lap (the
    default None is unknown and refuses with ``crown_gate_unknown``); reused frames need a finite ``frames_age_h``."""
    src, why = frames_precondition(kw.get("source", "one_lap"), frames_age_h, crown_gate_ok, kw.get("cfg"))
    if why:
        return ScoutResult(plan["tree_id"], {"status": "refused", "reason": why}, [], [], {}, 0)
    kw["source"] = src
    if profile["crop"] == "banana":
        return scout_banana(plan, profile, today, rota=kw.get("rota"), cfg=kw.get("cfg"))
    return scout_tree(plan, profile, today, **kw)


def scout_block(plans: Sequence[dict], profiles: Dict[str, dict], today=None, **kw) -> dict:
    """FRUIT SCOUT over a block: per-tree fruit blocks, the block yield estimate (labelled estimate; commercially
    sensitive, OQ-24) and the block harvest window."""
    res = [scout(p, profiles[p["crop"]], today, **kw) for p in plans]
    kg = sum(float(r.fruit.get("est_kg") or 0.0) for r in res)
    return {"trees": {r.tree_id: r.fruit for r in res}, "marks": [m for r in res for m in r.marks],
            "block_yield_kg": round(kg, 1), "block_yield_label": "estimate (commercially sensitive, OQ-24)",
            "harvest_window": rp.block_window([r.fruit.get("harvest_window") for r in res], today),
            "refused": {r.tree_id: r.fruit["reason"] for r in res if r.fruit.get("status") == "refused"},
            "assisted": {r.tree_id: {k: v for k, v in r.assisted.items() if k != "rows"} for r in res}}


__all__ = ["FruitDetection", "SOURCES", "ScoutResult", "SimTruthFruitDetector", "Track", "TreeFrameTracker", "frames_precondition", "orbit_views",
           "scout", "scout_banana", "scout_block", "scout_tree", "visible_truth"]
