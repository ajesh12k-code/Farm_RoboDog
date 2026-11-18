"""WP26 symptom / pest leads: seasonal decal classes, view geometry, SimTruth and ColorBlob lead detectors,
per-unit aggregation, block sampling plans and the enumerated next actions (ORCHARD-MVP-SPEC 2.5, 3.3, 4.4, 5.6;
MISSION-PRESETS 1, 3, 3b).

ROS-free and MuJoCo-free (numpy + PyYAML; ``world.trees`` / ``world.crops`` are pure Python). The other WP26
modules (``leaf_scan``, ``trunk_scan``, ``fruit_track``, ``ripeness``) share the config loader and the view
geometry here.

Rules kept here, for every caller:
* **Leads stay leads** (OS 5.6): every symptom this module emits has ``stage: "lead"``; ``confirmed`` only comes
  from the hub. :func:`lead` refuses anything else.
* ``next_actions`` come only from :data:`NEXT_ACTIONS` (``mark_jobs:<n>`` with an integer); no spray or treatment
  advice exists in this module.
* The SimTruth detector is **assisted** (it reads the decal truth through a view model) and says so in every lead
  (``detector: simtruth``); the ColorBlob baseline only reads RGB pixels and is reported, never gated.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi import paths
from atharv_nandi.sim.world import crops as crop_profiles
from atharv_nandi.sim.world import trees as gen

REPO = paths.REPO
CONFIG_PATH = os.path.join(REPO, "config", "perception_orchard.yaml")
SCHEMA = "nandi.perception_orchard/1"
NEXT_ACTIONS = ("revisit", "tree360", "deep_scan", "leaf_scan", "trunk_scan", "human_inspect")   # + mark_jobs:<n>
UNITS = ("per_panicle", "per_shoot", "per_leaf", "trunk")
TRUNK_KEYS = ("frass", "gum", "collar_rot", "mud_tubes", "weevil_holes", "band_ok", "mealybug", "wilt")
STRATA = ("N", "E", "S", "W")

_CFG: Optional[dict] = None


class PerceptionConfigError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------- config
def load_config(path: Optional[str] = None, refresh: bool = False) -> dict:
    """``config/perception_orchard.yaml`` (``NANDI_PERCEPTION_ORCHARD`` overrides the path). Cached."""
    global _CFG
    p = path or os.environ.get("NANDI_PERCEPTION_ORCHARD", "").strip() or CONFIG_PATH
    if _CFG is not None and not refresh and path is None and _CFG.get("_path") == p:
        return _CFG
    import yaml
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    errs = validate_config(cfg)
    if errs:
        raise PerceptionConfigError(f"{p}: " + "; ".join(errs))
    cfg["_path"] = p
    if path is None:
        _CFG = cfg
    return cfg


def validate_config(cfg) -> List[str]:
    if not isinstance(cfg, dict):
        return ["not a mapping"]
    errs = []
    if cfg.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA}")
    for sec in ("cameras", "leaf_scan", "trunk_scan", "classes", "simtruth", "colorblob", "aggregation",
                "fruit_track", "ripeness"):
        if not isinstance(cfg.get(sec), dict):
            errs.append(f"missing section {sec}")
    for name, c in (cfg.get("classes") or {}).items():
        if c.get("unit") not in UNITS:
            errs.append(f"class {name}: unit {c.get('unit')!r} not in {UNITS}")
        if c.get("trunk_key") is not None and c["trunk_key"] not in TRUNK_KEYS:
            errs.append(f"class {name}: trunk_key {c['trunk_key']!r} not in {TRUNK_KEYS}")
        rgb = c.get("rgb")
        if not (isinstance(rgb, list) and len(rgb) == 3):
            errs.append(f"class {name}: rgb must be [r, g, b]")
    # safety interlocks: an override (NANDI_PERCEPTION_ORCHARD) may tighten them, never loosen or switch them off
    def num(sec, key):
        try:
            v = float((cfg.get(sec) or {}).get(key))
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    ls = cfg.get("leaf_scan") or {}
    ts = cfg.get("trunk_scan") or {}
    fr = cfg.get("fruit_track") or {}
    for sec, d in (("leaf_scan", ls), ("trunk_scan", ts)):
        v = num(sec, "min_clearance_m")
        if d and (v is None or v < 0.02):
            errs.append(f"{sec}.min_clearance_m must be a number >= 0.02 (MISSION-PRESETS 3 / 3b)")
    if ls:
        v = num("leaf_scan", "drift_max_m")
        if v is None or not (0.0 < v <= 0.02):
            errs.append("leaf_scan.drift_max_m must be in (0, 0.02] (MISSION-PRESETS 3: base drift > 2 cm pauses)")
        blades = ls.get("blade_tools")
        if not isinstance(blades, list) or not {"T2", "T5"} <= set(map(str, blades)):
            errs.append("leaf_scan.blade_tools must list at least T2 and T5 (MISSION-PRESETS 3 blade interlock)")
        v = num("leaf_scan", "max_footprint_mm_px")
        if v is None or not (0.0 < v <= 1.0):
            errs.append("leaf_scan.max_footprint_mm_px must be in (0, 1.0] (O-08)")
    if fr:
        v = num("fruit_track", "frames_max_age_h")
        if v is None or not (0.0 < v <= 24.0):
            errs.append("fruit_track.frames_max_age_h must be in (0, 24] (MISSION-PRESETS 4)")
    return errs


# ---------------------------------------------------------------------------------------------- classes
def class_catalogue(cfg: Optional[dict] = None) -> Dict[str, dict]:
    return dict((cfg or load_config())["classes"])


def active_classes(profile: dict, season: str, scale: Optional[str] = None, cfg: Optional[dict] = None) -> List[dict]:
    """Symptom classes of a crop that are in season (``demo_all_fruiting``: every class, labelled, as WP23 does),
    merged with the catalogue (unit, trunk key, colour signature). Catalogue-only classes for the crop
    (e.g. mango ``shoot_gall``) join when their own ``seasons`` match."""
    cat = class_catalogue(cfg)
    if season == "demo_all_fruiting":
        rows = [dict(c) for c in profile["symptoms"]]
    else:
        rows = [dict(c) for c in crop_profiles.symptom_classes(profile, season)]
    have = {c["class"] for c in profile["symptoms"]}          # catalogue-only = not in the crop profile at all
    s = "mature" if season == "demo_all_fruiting" else season
    for name, c in cat.items():
        if name in have or profile["crop"] not in c.get("crops", []):
            continue
        seasons = c.get("seasons", ["all"])
        if season == "demo_all_fruiting" or "all" in seasons or s in seasons:
            rows.append({"class": name, "unit": c["unit"], "scale": c["scale"], "seasons": seasons,
                         "bands": c.get("bands", ["low", "mid", "high"])})
    out = []
    for r in rows:
        c = cat.get(r["class"], {})
        r.setdefault("unit", c.get("unit"))
        r["trunk_key"] = c.get("trunk_key")
        r["rgb"] = c.get("rgb", gen.DECAL_DEFAULT_RGBA[:3])
        r["distance_only"] = bool(c.get("distance_only", False))
        r["graded"] = bool(c.get("graded", False))
        if scale is None or r.get("scale") == scale:
            out.append(r)
    return out


# ---------------------------------------------------------------------------------------------- view geometry
def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def quat_z_axis(q) -> np.ndarray:
    """Local z axis of a (w, x, y, z) quaternion in the parent frame."""
    w, x, y, z = (float(c) for c in q)
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def quat_mat(q) -> np.ndarray:
    w, x, y, z = (float(c) for c in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_quat(R: np.ndarray) -> List[float]:
    return [float(c) for c in gen.mat2quat(np.asarray(R, float))]


@dataclass
class View:
    """A pinhole camera pose in the world. ``fwd`` is the optical axis (MuJoCo cameras look along local −z)."""
    pos: np.ndarray
    fwd: np.ndarray
    fovy_deg: float = 60.0
    width: int = 320
    height: int = 240
    camera: str = "wrist_cam"
    band: str = "low"
    tag: dict = field(default_factory=dict)
    up_hint: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.fwd = unit(self.fwd)
        up = np.asarray(self.up_hint, float)
        if abs(float(np.dot(up, self.fwd))) > 0.98:
            up = np.array([1.0, 0.0, 0.0]) if abs(self.fwd[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = unit(np.cross(self.fwd, up))
        self.right = right
        self.up = np.cross(right, self.fwd)

    @property
    def f_px(self) -> float:
        return 0.5 * self.height / math.tan(math.radians(self.fovy_deg) / 2.0)

    def rot(self) -> np.ndarray:
        """Camera-to-world rotation with MuJoCo's convention (x right, y up, z back)."""
        return np.stack([self.right, self.up, -self.fwd], axis=1)

    def quat(self) -> List[float]:
        return mat_quat(self.rot())

    def pose7(self) -> Tuple[float, ...]:
        return tuple(float(c) for c in list(self.pos) + self.quat())

    def depth(self, p) -> float:
        return float(np.dot(np.asarray(p, float) - self.pos, self.fwd))

    def project(self, p) -> Optional[Tuple[float, float, float]]:
        """(u, v, depth) in pixels (u right, v down, origin top-left), or None behind the camera."""
        d = np.asarray(p, float) - self.pos
        z = float(np.dot(d, self.fwd))
        if z <= 1e-6:
            return None
        u = self.width / 2.0 + self.f_px * float(np.dot(d, self.right)) / z
        v = self.height / 2.0 - self.f_px * float(np.dot(d, self.up)) / z
        return u, v, z

    def in_frame(self, p, margin_px: float = 0.0) -> bool:
        pr = self.project(p)
        if pr is None:
            return False
        u, v, _ = pr
        return margin_px <= u <= self.width - margin_px and margin_px <= v <= self.height - margin_px

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        x = (u - self.width / 2.0) * depth / self.f_px
        y = -(v - self.height / 2.0) * depth / self.f_px
        return self.pos + depth * self.fwd + x * self.right + y * self.up

    def mm_per_px(self, depth_m: float) -> float:
        return footprint_mm_per_px(self.fovy_deg, self.height, depth_m)


def footprint_mm_per_px(fovy_deg: float, height_px: int, dist_m: float) -> float:
    """Object-space size of one pixel at ``dist_m`` (square pixels; same formula as ``camera_rig``)."""
    return 1000.0 * 2.0 * dist_m * math.tan(math.radians(fovy_deg) / 2.0) / float(height_px)


def look_at(pos, target, **kw) -> View:
    return View(pos=np.asarray(pos, float), fwd=np.asarray(target, float) - np.asarray(pos, float), **kw)


def disc_px(view: View, p, n, diameter_m: float) -> float:
    """Equivalent-area diameter in pixels of a flat disc (centre ``p``, normal ``n``) seen by ``view``;
    0 when behind the camera or seen from its back side."""
    pr = view.project(p)
    if pr is None:
        return 0.0
    ray = unit(view.pos - np.asarray(p, float))
    c = float(np.dot(unit(n), ray))
    if c <= 0.0:
        return 0.0
    dist = float(np.linalg.norm(view.pos - np.asarray(p, float)))
    return 1000.0 * diameter_m / view.mm_per_px(dist) * math.sqrt(c)


# ---------------------------------------------------------------------------------------------- truth helpers
def decal_normals(plan: dict) -> Dict[str, np.ndarray]:
    """Decal geom name -> world normal (the decal cylinder's z axis; trees are not rotated)."""
    return {g["name"]: quat_z_axis(g["quat"]) for g in plan["geoms"] if g["kind"] == "decal" and "quat" in g}


def truth_decals(plan: dict, profile: Optional[dict] = None, cfg: Optional[dict] = None) -> List[dict]:
    """The plan's decals with world normal, diameter (m) and the catalogue scale; ``active`` = in season."""
    normals = decal_normals(plan)
    cat = class_catalogue(cfg)
    act = None
    if profile is not None:
        act = {c["class"] for c in active_classes(profile, plan["season"], cfg=cfg)}
    out = []
    for d in plan["truth"]["decals"]:
        row = dict(d)
        row["n_world"] = normals.get(d["geom"], np.array([0.0, -1.0, 0.0]))
        row["p_world"] = np.asarray(d["world"], float)
        row["diameter_m"] = float(d["diameter_mm"]) / 1000.0
        row["scale"] = "leaf" if d.get("leaf") else (cat.get(d["class"], {}).get("scale") or "canopy")
        if row["scale"] == "leaf" and d.get("unit") != "per_leaf":
            row["scale"] = "leaf"
        row["active"] = True if act is None else d["class"] in act
        out.append(row)
    return out


def lead(cls: str, unit_: str, p_world, h_m: float, band: str, clock: int, score: float, detector: str,
         stage: str = "lead", **extra) -> dict:
    """One symptom lead (TreeRecord ``symptoms[]`` row). Only ``stage: lead`` is ever written by Nandi."""
    if stage != "lead":
        raise ValueError("Nandi writes only stage 'lead' (OS 5.6: confirmed comes from the hub)")
    if unit_ not in UNITS:
        raise ValueError(f"unit {unit_!r} not in {UNITS}")
    row = {"class": cls, "band": band, "clock": int(clock), "h_class": crop_profiles.h_class(float(h_m)),
           "h_m": round(float(h_m), 3), "unit": unit_, "count": 1, "score": round(float(score), 3),
           "stage": "lead", "detector": detector, "p": [round(float(c), 4) for c in p_world]}
    row.update(extra)
    return row


def _alley_dir(entry: dict) -> Tuple[int, int]:
    alley = (entry or {}).get("alley_sides") or ["-y"]
    return {"-y": (0, -1), "+y": (0, 1), "-x": (-1, 0), "+x": (1, 0)}.get(alley[0], (0, -1))


def clock_of_world(plan: dict, p_world) -> int:
    local = np.asarray(p_world, float) - np.asarray(plan["pos"], float)
    return gen.clock_of(local, _alley_dir(plan.get("entry")))


# ---------------------------------------------------------------------------------------------- SimTruth detector
class SimTruthLeadDetector:
    """Assisted lead detector: reads decal truth through a view model (frustum, facing, footprint), then
    misses / confuses / adds false positives with a seeded RNG. Labelled ``simtruth`` in every lead."""

    name = "simtruth"

    def __init__(self, cfg: Optional[dict] = None, seed: Optional[int] = None):
        full = cfg or load_config()
        self.cfg = full["simtruth"]
        self.rng = np.random.default_rng(int(self.cfg["seed"] if seed is None else seed))

    def visible(self, view: View, d: dict) -> Tuple[bool, float]:
        p, n = d["p_world"], d["n_world"]
        if not view.in_frame(p):
            return False, 0.0
        ray = unit(view.pos - p)
        if float(np.dot(unit(n), ray)) < math.cos(math.radians(float(self.cfg["max_view_angle_deg"]))):
            return False, 0.0
        px = disc_px(view, p, n, d["diameter_m"])
        return px >= float(self.cfg["min_px"]), px

    def detect(self, view: View, decals: Sequence[dict], classes: Sequence[dict], plan: dict) -> List[dict]:
        c = self.cfg
        by_class = {k["class"]: k for k in classes}
        names = sorted(by_class)
        out = []
        for d in decals:
            if not d.get("active", True) or d["class"] not in by_class:
                continue
            ok, px = self.visible(view, d)
            if not ok or self.rng.uniform() > float(c["p_detect"]):
                continue
            cls = d["class"]
            if len(names) > 1 and self.rng.uniform() < float(c["p_confuse"]):
                cls = names[int(self.rng.integers(0, len(names)))]
            k = by_class[cls]
            score = float(self.rng.uniform(*c["score"]))
            out.append(lead(cls, k["unit"], d["p_world"], float(d["p_world"][2]), gen.band_of(float(d["p_world"][2])),
                            clock_of_world(plan, d["p_world"]), score, self.name, px=round(px, 1),
                            truth=d["id"], camera=view.camera, view_band=view.band))
        n_fp = int(self.rng.poisson(float(c["fp_per_frame"])))
        for _ in range(n_fp):
            if not names:
                break
            k = by_class[names[int(self.rng.integers(0, len(names)))]]
            u, v = self.rng.uniform(0, view.width), self.rng.uniform(0, view.height)
            p = view.unproject(u, v, float(self.rng.uniform(0.12, 0.4)))
            out.append(lead(k["class"], k["unit"], p, float(p[2]), gen.band_of(float(p[2])), clock_of_world(plan, p),
                            float(self.rng.uniform(*c["fp_score"])), self.name, truth=None, camera=view.camera,
                            view_band=view.band))
        thr = float(c["score_threshold"])
        return [x for x in out if x["score"] >= thr]


# ---------------------------------------------------------------------------------------------- ColorBlob baseline
class ColorBlobLeadDetector:
    """RGB-only baseline: pixels close (in chromaticity and brightness relative to the frame) to a class colour
    signature form blobs; blob class = nearest signature among the active classes. Classes that share a colour
    are not separable: their per-class recall is what it is, and it is reported, never gated."""

    name = "colorblob"

    def __init__(self, cfg: Optional[dict] = None):
        full = cfg or load_config()
        self.cfg = full["colorblob"]

    @staticmethod
    def _feat(rgb: np.ndarray) -> np.ndarray:
        f = np.asarray(rgb, np.float32)
        if f.max() > 1.5:
            f = f / 255.0
        s = f.sum(axis=-1, keepdims=True) + 1e-6
        chroma = f / s
        return np.concatenate([chroma[..., :2], s / 3.0], axis=-1)

    def detect(self, rgb: np.ndarray, classes: Sequence[dict], depth: Optional[np.ndarray] = None,
               background_rgb: Optional[Sequence[Sequence[float]]] = None) -> List[dict]:
        from atharv_nandi.nodes.perception_node import label_components   # lazy: pure numpy labeller (WP5)
        img = self._feat(rgb[..., :3])
        gain = 1.0
        if background_rgb:
            # light gain: the frame's median brightness vs the brightest expected background colour
            bg = np.asarray(background_rgb, float)
            exp_v = float(np.max(bg.sum(axis=-1) / 3.0))
            gain = float(np.median(img[..., 2])) / max(exp_v, 1e-3)
            gain = float(np.clip(gain, 0.2, 3.0))
        sigs = []
        for k in classes:
            f = self._feat(np.asarray(k["rgb"], np.float32)[None, None, :])[0, 0]
            f[2] *= gain
            sigs.append(f)
        if not sigs:
            return []
        S = np.stack(sigs)                                           # (K, 3)
        dist = np.linalg.norm(img[:, :, None, :] - S[None, None, :, :], axis=-1)   # (H, W, K)
        if background_rgb:
            B = np.stack([self._feat(np.asarray(b, np.float32)[None, None, :])[0, 0] for b in background_rgb])
            B[:, 2] *= gain
            dbg = np.linalg.norm(img[:, :, None, :] - B[None, None, :, :], axis=-1).min(axis=-1)
        else:
            dbg = np.full(img.shape[:2], np.inf)
        best = dist.argmin(axis=-1)
        bd = dist.min(axis=-1)
        cand = (bd <= float(self.cfg["max_rgb_dist"])) & (bd < dbg)
        out = []
        for ki, k in enumerate(classes):
            lab, n = label_components(cand & (best == ki))
            for j in range(1, n + 1):
                ys, xs = np.nonzero(lab == j)
                if len(xs) < int(self.cfg["min_area_px"]):
                    continue
                det = {"class": k["class"], "unit": k["unit"], "u": float(xs.mean()), "v": float(ys.mean()),
                       "area_px": int(len(xs)), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                       "detector": self.name, "stage": "lead"}
                if depth is not None:
                    z = depth[ys, xs]
                    z = z[np.isfinite(z)]
                    det["z"] = float(np.median(z)) if len(z) else None
                out.append(det)
        return out


# ---------------------------------------------------------------------------------------------- aggregation
def dedupe_leads(leads: Sequence[dict], radius_m: float = 0.05) -> List[dict]:
    """One symptom per (class, place): leads of one class within ``radius_m`` merge (views of the same decal);
    ``count`` = number of distinct instances (1 per merged row), ``views`` = frames it was seen in,
    ``score`` = max."""
    rows: List[dict] = []
    for ld in sorted(leads, key=lambda x: -x["score"]):
        p = np.asarray(ld["p"], float)
        hit = None
        for r in rows:
            if r["class"] == ld["class"] and float(np.linalg.norm(np.asarray(r["p"], float) - p)) <= radius_m:
                hit = r
                break
        if hit is None:
            r = dict(ld)
            r["views"] = 1
            r["truth_ids"] = [ld.get("truth")] if ld.get("truth") else []
            rows.append(r)
        else:
            hit["views"] += 1
            if ld.get("truth") and ld["truth"] not in hit["truth_ids"]:
                hit["truth_ids"].append(ld["truth"])
    for r in rows:
        r.pop("truth", None)
    return rows


def unit_summary(symptoms: Sequence[dict], n_units: Optional[Dict[str, int]] = None) -> Dict[str, dict]:
    """Per class, in its standard unit (OS 5.6): ``n_affected`` instances, ``n_units`` sampled units when known
    (leaves imaged, panicles seen, shoots sampled, 1 trunk), ``pct`` = 100 · affected / units, ``per_unit`` for
    counts per panicle. Leads only."""
    n_units = n_units or {}
    out: Dict[str, dict] = {}
    for s in symptoms:
        r = out.setdefault(s["class"], {"unit": s["unit"], "n_affected": 0, "score_max": 0.0, "stage": "lead"})
        r["n_affected"] += int(s.get("count", 1))
        r["score_max"] = max(r["score_max"], float(s["score"]))
    for cls, r in out.items():
        u = r["unit"]
        nu = 1 if u == "trunk" else n_units.get(u)
        r["n_units"] = nu
        if u == "trunk":
            r["present"] = r["n_affected"] > 0
        elif nu:
            r["pct"] = round(100.0 * min(r["n_affected"], nu) / nu, 1)
            if u == "per_panicle":
                r["per_unit"] = round(r["n_affected"] / nu, 3)
        else:
            r["pct"] = None
    return out


def block_summary(tree_rows: Sequence[dict], plan_used: Optional[dict] = None) -> dict:
    """% trees affected per class over the trees actually scanned (``tree_rows``: ``{tree_id, symptoms[]}``),
    with the sampling plan used. Leads only."""
    n = len(tree_rows)
    per: Dict[str, int] = {}
    for t in tree_rows:
        for cls in {s["class"] for s in t.get("symptoms", [])}:
            per[cls] = per.get(cls, 0) + 1
    return {"trees_n": n, "stage": "lead",
            "pct_trees_affected": {c: round(100.0 * k / n, 1) for c, k in sorted(per.items())} if n else {},
            "sampling_plan": plan_used}


def stratum_of(xy, centroid) -> str:
    dx, dy = float(xy[0]) - float(centroid[0]), float(xy[1]) - float(centroid[1])
    ang = math.degrees(math.atan2(dy, dx)) % 360.0           # 0 = +x = E (fictitious local frame: +y = N)
    if 45.0 <= ang < 135.0:
        return "N"
    if 135.0 <= ang < 225.0:
        return "W"
    if 225.0 <= ang < 315.0:
        return "S"
    return "E"


def block_sampling_plan(entries: Sequence[dict], crop: Optional[str] = None, n: Optional[int] = None,
                        seed: int = 26, cfg: Optional[dict] = None) -> dict:
    """N trees stratified by direction (N/E/S/W of the block centroid; OS 5.6). N from the crop
    (``aggregation.block_sample_n``, assumption for KVK review), never more than the block. Deterministic in
    (entries, crop, n, seed): strata are filled round-robin, a stratum's order is a seeded shuffle."""
    agg = (cfg or load_config())["aggregation"]
    rows = [e for e in entries if crop is None or e.get("crop") == crop]
    if not rows:
        return {"n": 0, "crop": crop, "strata": {}, "tree_ids": [], "basis": "stratified_by_direction"}
    want = int(n if n is not None else agg["block_sample_n"].get(crop or "default", agg["block_sample_n"]["default"]))
    xy = np.array([[float(e.get("x", (e.get("geo") or {}).get("x", 0.0))),
                    float(e.get("y", (e.get("geo") or {}).get("y", 0.0)))] for e in rows])
    c = xy.mean(axis=0)
    strata: Dict[str, List[str]] = {s: [] for s in STRATA}
    for e, p in zip(rows, xy):
        strata[stratum_of(p, c)].append(e["tree_id"])
    rng = np.random.default_rng(seed)
    for s in STRATA:
        ids = sorted(strata[s])
        rng.shuffle(ids)
        strata[s] = ids
    picked, k = [], 0
    while len(picked) < min(want, len(rows)):
        s = STRATA[k % 4]
        if strata[s]:
            picked.append(strata[s].pop(0))
        k += 1
    used: Dict[str, List[str]] = {s: [] for s in STRATA}
    for e, p in zip(rows, xy):
        if e["tree_id"] in picked:
            used[stratum_of(p, c)].append(e["tree_id"])
    return {"n": len(picked), "crop": crop, "strata": {s: sorted(v) for s, v in used.items()}, "tree_ids": picked,
            "basis": "stratified_by_direction", "requested_n": want}


def next_actions(symptoms: Sequence[dict] = (), trunk_leads: int = 0, leaf_leads: int = 0, marks_n: int = 0,
                 coverage_partial: bool = False, did: Iterable[str] = (), triage: float = 0.5) -> List[str]:
    """The enumerated, non-advisory ``next_actions[]`` (OS 5.6, MP 0.6). Never treatment advice."""
    did = set(did)
    out: List[str] = []
    strong = [s for s in symptoms if float(s.get("score", 0)) >= triage]
    if coverage_partial:
        out.append("revisit")
    if any(s["unit"] in ("per_panicle", "per_shoot") for s in strong) and "tree360" not in did:
        out.append("tree360")
    if strong and "tree360" in did:
        out.append("deep_scan")
    if any(s["unit"] == "per_leaf" for s in strong) and "leaf_scan" not in did:
        out.append("leaf_scan")
    if any(s["unit"] == "trunk" for s in strong) and "trunk_scan" not in did:
        out.append("trunk_scan")
    if trunk_leads or leaf_leads:
        out.append("human_inspect")
    if marks_n:
        out.append(f"mark_jobs:{int(marks_n)}")
    seen = []
    for a in out:
        if a not in seen:
            seen.append(a)
    for a in seen:
        assert a in NEXT_ACTIONS or a.startswith("mark_jobs:"), a
    return seen


def recall(leads: Sequence[dict], truth_ids: Iterable[str], by_class: bool = True,
           truth_class: Optional[Dict[str, str]] = None) -> Tuple[float, int, int]:
    """(recall, hits, n) of truth decals found by (deduped or raw) SimTruth leads. With ``by_class`` a hit needs
    the right class (``truth_class`` maps id -> class)."""
    want = list(dict.fromkeys(truth_ids))
    if not want:
        return 1.0, 0, 0
    found = set()
    for ld in leads:
        ids = ld.get("truth_ids") or ([ld["truth"]] if ld.get("truth") else [])
        for i in ids:
            if not by_class or truth_class is None or truth_class.get(i) == ld["class"]:
                found.add(i)
    hits = sum(1 for i in want if i in found)
    return hits / len(want), hits, len(want)


# ---------------------------------------------------------------------------------------------- canopy leads
def canopy_leads(plan: dict, profile: dict, views: Sequence[View], detector: Optional[SimTruthLeadDetector] = None,
                 cfg: Optional[dict] = None) -> List[dict]:
    """Canopy- and trunk-band leads from SCAN / TREE 360 frames (≥ 5 cm decals; OS 2.5). Hopper evidence is
    taken from these distance frames only (``distance_only`` classes), reported per panicle."""
    full = cfg or load_config()
    det = detector or SimTruthLeadDetector(full)
    classes = active_classes(profile, plan["season"], cfg=full)
    decals = [d for d in truth_decals(plan, profile, full) if d["scale"] != "leaf"]
    out: List[dict] = []
    for v in views:
        found = det.detect(v, decals, classes, plan)
        for ld in found:
            ld["label"] = "canopy_level"
        out.extend(found)
    return out


def canopy_truth_ids(plan: dict, profile: dict, views: Sequence[View], cfg: Optional[dict] = None) -> Dict[str, str]:
    """Active non-leaf decal truth ``{id: class}`` that at least one of ``views`` can image (frustum, facing,
    ≥ min px): the denominator of the canopy-lead recall."""
    full = cfg or load_config()
    det = SimTruthLeadDetector(full)
    act = {c["class"] for c in active_classes(profile, plan["season"], cfg=full)}
    return {d["id"]: d["class"] for d in truth_decals(plan, profile, full)
            if d["scale"] != "leaf" and d["class"] in act and any(det.visible(v, d)[0] for v in views)}


# ---------------------------------------------------------------------------------------------- record fragment
def record_fragment(plan: dict, symptoms: Sequence[dict] = (), leaf: Optional[dict] = None,
                    trunk: Optional[dict] = None, fruit: Optional[dict] = None, marks_n: int = 0,
                    did: Iterable[str] = (), n_units: Optional[Dict[str, int]] = None,
                    coverage_partial: bool = False, triage: Optional[float] = None) -> dict:
    """The WP26 fields of ``TreeRecord nandi.tree/1`` (MP 0.6): ``symptoms[]`` (leads only, no truth ids, no
    pixels), ``units``, ``trunk``, ``leaf``, ``fruit`` and ``next_actions[]``. No images, no personal fields."""
    keep = ("class", "band", "clock", "h_class", "h_m", "unit", "count", "score", "stage")
    rows = [{k: s[k] for k in keep} for s in symptoms]
    for r in rows:
        assert r["stage"] == "lead"
    thr = float(triage if triage is not None else load_config()["aggregation"]["triage_threshold"])
    out = {"tree_id": plan["tree_id"], "crop": plan["crop"], "symptoms": rows,
           "units": unit_summary(symptoms, n_units),
           "detector": sorted({s.get("detector", "simtruth") for s in symptoms}) or ["simtruth"],
           "assisted": ["simtruth_leads"] if any(s.get("detector") == "simtruth" for s in symptoms) else []}
    if trunk is not None:
        out["trunk"] = trunk
    if leaf is not None:
        out["leaf"] = leaf if leaf.get("status") == "ok" else {"status": leaf.get("status")}
    if fruit is not None:
        out["fruit"] = fruit
    out["next_actions"] = next_actions(symptoms, trunk_leads=sum(1 for s in symptoms if s["unit"] == "trunk"),
                                       leaf_leads=sum(1 for s in symptoms if s["unit"] == "per_leaf"),
                                       marks_n=marks_n, coverage_partial=coverage_partial, did=did, triage=thr)
    return out


# ---------------------------------------------------------------------------------------------- KPI report
def kpi_report(world: str = "orchard_demo", seed: int = 7, scout_seeds: Sequence[int] = (26,)) -> dict:
    """WP26 KPI numbers on a composed world's tree plans (no MuJoCo, no GL): leaf scan, trunk scan, canopy
    leads, fruit scout. Used by the tests and ``python3 -m atharv_nandi.core.perception.pest_detect --report``."""
    import statistics
    from atharv_nandi.sim.world import compose
    from atharv_nandi.core.perception import fruit_track as ft, leaf_scan as ls, trunk_scan as ts
    field_ = compose.load_field(world, seed)
    plans = gen.world_plans(field_, seed)
    profiles = crop_profiles.load_profiles()
    today = field_.get("sim_date") or gen.DEFAULT_SIM_DATE
    rep: Dict[str, object] = {"world": world, "seed": seed, "trees": len(plans), "today": today}
    lh = ln = 0
    planned = imaged = under = close = 0
    times, clears, foot = [], [], []
    na = []
    for p in plans:
        r = ls.run_leaf_scan(p, profiles[p["crop"]])
        b = r.leaf_block
        if b.get("status") != "ok":
            na.append((p["tree_id"], b.get("status")))
            continue
        truth = ls.leaf_truth_ids(p, profiles[p["crop"]], leaf_ids=[x["leaf_id"] for x in r.records if x["close_up"]])
        rc = recall(dedupe_leads(r.leads), truth, True, truth)
        lh, ln = lh + rc[1], ln + rc[2]
        planned += b["planned_views"]
        imaged += b["imaged_views"]
        under += b["underside_n"]
        close += sum(1 for x in r.records if x["close_up"])
        times += [x["time_s"] for x in r.records if x["close_up"]]
        if b["min_clearance_m"] is not None:
            clears.append(b["min_clearance_m"])
        foot.append(b["footprint_mm_px_max"])
    rep["leaf_scan"] = {"lead_recall": round(lh / ln, 4) if ln else None, "leads_truth_n": ln,
                        "imaged_frac": round(imaged / planned, 4) if planned else None,
                        "underside_frac": round(under / close, 4) if close else None,
                        "median_s_per_leaf": round(statistics.median(times), 2) if times else None,
                        "min_clearance_m": min(clears) if clears else None,
                        "footprint_mm_px_max": max(foot) if foot else None, "n_a": na}
    th = tn = th_all = tn_all = 0
    az_full = []
    for p in plans:
        r = ts.run_trunk_scan(p, profiles[p["crop"]])
        acc = ts.trunk_truth_ids(p, profiles[p["crop"]], accessible_only=True)
        allt = ts.trunk_truth_ids(p, profiles[p["crop"]])
        a = recall(r.symptoms, acc, True, acc)
        b = recall(r.symptoms, allt, True, allt)
        th, tn, th_all, tn_all = th + a[1], tn + a[2], th_all + b[1], tn_all + b[2]
        az_full.append((p["tree_id"], r.mode, r.azimuths_imaged))
    rep["trunk_scan"] = {"lead_recall_accessible": round(th / tn, 4) if tn else None, "accessible_n": tn,
                         "lead_recall_all_sides": round(th_all / tn_all, 4) if tn_all else None, "all_n": tn_all,
                         "azimuths": az_full}
    ch = cn = 0
    for p in plans:
        views = ft.orbit_views(p)
        leads = dedupe_leads(canopy_leads(p, profiles[p["crop"]], views))
        truth = canopy_truth_ids(p, profiles[p["crop"]], views)
        rc = recall(leads, truth, True, truth)
        ch, cn = ch + rc[1], cn + rc[2]
    rep["canopy_leads"] = {"lead_recall": round(ch / cn, 4) if cn else None, "truth_n": cn}
    agg = {"abs": 0, "truth": 0, "trees": 0, "over10": 0, "max_err": 0.0, "stage_hits": 0, "stage_n": 0,
           "cb_hits": 0, "window_eq": 0, "windows": 0, "size_err": [], "per_crop": {}}
    for s in scout_seeds:
        for p in plans:
            r = ft.scout(p, profiles[p["crop"]], today, seed=s, crown_gate_ok=True) if p["crop"] != "banana" else \
                ft.scout(p, profiles[p["crop"]], today, crown_gate_ok=True)
            a = r.assisted
            agg["windows"] += 1
            agg["window_eq"] += int(a["window_equal"])
            if p["crop"] == "banana":
                continue
            agg["abs"] += abs(r.fruit["visible_n"] - a["truth_visible_n"])
            agg["truth"] += a["truth_visible_n"]
            agg["trees"] += 1
            agg["over10"] += int(a["count_error"] > 0.10)
            agg["max_err"] = max(agg["max_err"], a["count_error"])
            pc = agg["per_crop"].setdefault(p["crop"], {"stage_hits": 0, "stage_n": 0, "cb_hits": 0})
            if a["stage_n"]:
                h = round(a["stage_acc"] * a["stage_n"])
                cbh = round(a["colorblob_stage_acc"] * a["stage_n"])
                agg["stage_hits"] += h
                agg["stage_n"] += a["stage_n"]
                agg["cb_hits"] += cbh
                pc["stage_hits"] += h
                pc["stage_n"] += a["stage_n"]
                pc["cb_hits"] += cbh
            if a["size_mae_mm_low"] is not None:
                agg["size_err"].append(a["size_mae_mm_low"])
    rep["fruit_scout"] = {
        "count_error_aggregate": round(agg["abs"] / agg["truth"], 4) if agg["truth"] else None,
        "trees_over_10pct": agg["over10"], "tree_runs": agg["trees"], "count_error_max": agg["max_err"],
        "stage_acc": round(agg["stage_hits"] / agg["stage_n"], 4) if agg["stage_n"] else None,
        "colorblob_stage_acc": round(agg["cb_hits"] / agg["stage_n"], 4) if agg["stage_n"] else None,
        "stage_acc_by_crop": {c: {"simtruth": round(v["stage_hits"] / v["stage_n"], 4),
                                  "colorblob": round(v["cb_hits"] / v["stage_n"], 4), "n": v["stage_n"]}
                              for c, v in agg["per_crop"].items() if v["stage_n"]},
        "size_mae_mm_low_mean": round(float(np.mean(agg["size_err"])), 2) if agg["size_err"] else None,
        "window_equal": f"{agg['window_eq']}/{agg['windows']}", "seeds": list(scout_seeds)}
    return rep


if __name__ == "__main__":     # pragma: no cover
    import json
    import sys
    args = sys.argv[1:]
    w = args[args.index("--report") + 1] if "--report" in args and len(args) > args.index("--report") + 1 else "orchard_demo"
    print(json.dumps(kpi_report(w, scout_seeds=(7, 11, 23, 26)), indent=1, default=str))
