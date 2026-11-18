"""Procedural orchard trees (WP23; ORCHARD-MVP-SPEC 3.3, 4.1, 4.2; MISSION-PRESETS 0.1).

Pure Python (numpy + PyYAML): ``generate(entry, species, profile, season, seed)`` turns one tree-registry entry
into a JSON-able **tree plan**: every geom, mesh and site of the tree's single static body plus the truth record
(fruit with stage / mass / in-band, bags, decals with season, prune candidates with keep-or-cut and diameter,
suckers, leaves with functional / Sigatoka state, bunch with ribbon and shooting date). ``atharv_nandi/sim/world/modules/
trees.py`` writes plans into the ``MjSpec``; ``atharv_nandi/sim/world/modules/fruit_pool.py`` and ``atharv_nandi/core/fruit_pool.py``
use the fruit table of the same plans. Nothing here imports MuJoCo, so plans and hashes are testable anywhere.

Geom groups (OS 2.5 "Ray groups")::

    0 wood (keep wood contype 1 / conaffinity 0; prune candidates contype 0)   1 fruit (static: visual only)
    3 foliage (merged meshes) and leaf quads      4 decals (truth in the plan)      5 query-only: foliage envelope
Contact rule: tree geoms never have conaffinity, so two tree geoms or a tree geom and a pool fruit (contype 1 /
conaffinity 0) never collide ((1&0)|(1&0) = 0); the robot (1/1) collides with keep wood (OS 3.4).

Frames: tree frame = trunk axis at the origin on the ground, world axes (the WP0c convention); a plan's ``pos`` is
the world position of that origin. Every number is an (assumption) inside the cited ranges of OS 3.3.

Registry: ``config/fields/<world>.json`` ``trees`` (WP4 owns those files). Until WP4 lands, or for an entry
without geometry, the orchard worlds use ``default_registry()``: the WP0c candidate trees of
``config/fields/orchard_candidates.json`` (positions, wood skeleton, frozen fruit / prune / leaf points), so the
generated in-band targets are exactly the points WP0c swept. ``NANDI_TREES_CONFIG`` / ``NANDI_CROPS_CONFIG``
override the config directories (a 5th crop by config); ``NANDI_SEASON`` the season.
"""
import copy
import datetime as _dt
import glob
import hashlib
import itertools
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi import paths
from atharv_nandi.sim.world import crops

REPO = paths.REPO
TREES_DIR = os.path.join(REPO, "config", "trees")
CANDIDATES_JSON = os.path.join(REPO, "config", "fields", "orchard_candidates.json")
SPECIES_SCHEMA = "nandi.tree_species/1"
PLAN_SCHEMA = "nandi.tree_plan/1"

GRP_WOOD, GRP_FRUIT, GRP_ROBOT, GRP_FOLIAGE, GRP_DECAL, GRP_QUERY = 0, 1, 2, 3, 4, 5
RENDER_GROUP_MAX = GRP_DECAL               # groups 0..4 are rendered; 5 is query-only
RENDER_GEOMS_PER_TREE_MAX = 70             # OS 3.7 / WP23 acceptance
LEAF_QUADS_MAX = 12                        # OS 3.3
PICK_BAND_Z_M = crops.PICK_BAND_Z_M
ARCHETYPES = ("canopy", "papaya", "banana")
SEASONS = crops.SEASONS + ("demo_all_fruiting",)
DEFAULT_SEASON = "demo_all_fruiting"       # OS 1.1: all four species fruit at once (labelled, not realistic)
DEFAULT_SIM_DATE = "2026-06-20"            # Dashehari harvest window (2nd week June - 1st week July)
ORCHARD_WORLDS = ("orchard_lite", "orchard_demo", "orchard_demo_monsoon")
# WP23 demo registry facts that WP4 field trees do not carry (OS 1.2 / 3.3 / 3.6 demo details)
REGISTRY_FACT_KEYS = ("decals", "leaf_decals", "drooping_leaf", "crowded_ripe_k")
# WP4 field ``season`` labels (climate windows) -> WP23 tree season. ``NANDI_SEASON`` still overrides.
FIELD_SEASON_ALIASES = {"monsoon_jul_sep": DEFAULT_SEASON}
BAND_EDGES = (("low", PICK_BAND_Z_M), ("mid", 1.8), ("high", float("inf")))
HIDEABLE_KINDS = ("fruit", "bag", "prune")  # static geoms the pool / cut rule may hide (alpha 0, contype 0)

WOOD_RGBA = [0.45, 0.30, 0.18, 1.0]
SPROUT_RGBA = [0.40, 0.55, 0.22, 1.0]
DEAD_RGBA = [0.50, 0.44, 0.36, 1.0]
BAG_RGBA = [0.86, 0.80, 0.62, 1.0]
ENVELOPE_RGBA = [0.0, 1.0, 0.0, 0.08]
RIBBON_RGBA = {"red": [0.85, 0.1, 0.1, 1.0], "blue": [0.1, 0.25, 0.85, 1.0], "yellow": [0.95, 0.85, 0.1, 1.0],
               "white": [0.95, 0.95, 0.95, 1.0], "green": [0.1, 0.7, 0.2, 1.0], "black": [0.05, 0.05, 0.05, 1.0]}
DECAL_RGBA = {"sooty_mould": [0.08, 0.08, 0.08, 1.0], "gummosis": [0.55, 0.30, 0.05, 1.0],
              "weevil_gum": [0.60, 0.40, 0.10, 1.0], "mealybug": [0.95, 0.95, 0.92, 1.0],
              "anthracnose": [0.20, 0.12, 0.06, 1.0], "sigatoka": [0.45, 0.35, 0.10, 1.0]}
DECAL_DEFAULT_RGBA = [0.30, 0.22, 0.10, 1.0]


class TreeConfigError(ValueError):
    pass


# ---------------------------------------------------------------------- small numeric helpers
def _r(x, nd: int = 6):
    """Round floats (recursively) so a plan's canonical JSON, and hence its hash, is platform-stable."""
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return [_r(float(v), nd) for v in x.reshape(-1)]
    if isinstance(x, (float, np.floating)):
        v = round(float(x), nd)
        return 0.0 if v == 0 else v
    if isinstance(x, np.integer):
        return int(x)
    return x


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else np.array([0.0, 0.0, 1.0])


def mat2quat(R: np.ndarray) -> List[float]:
    """Rotation matrix -> unit quaternion (w, x, y, z), w >= 0 (Shepperd)."""
    R = np.asarray(R, dtype=float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q.tolist()


def frame_quat(z_axis, x_hint) -> List[float]:
    """Quaternion of a frame whose local z is ``z_axis`` and local x is ``x_hint`` made orthogonal to it."""
    z = _unit(z_axis)
    x = np.asarray(x_hint, dtype=float) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = _unit(x)
    y = np.cross(z, x)
    return mat2quat(np.stack([x, y, z], axis=1))


def capsule_frame(a, b) -> Tuple[List[float], List[float], float]:
    """(pos, quat, half_length) of a capsule / cylinder from ``a`` to ``b`` (MuJoCo: axis = local z)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    half = float(np.linalg.norm(d)) / 2.0
    return ((a + b) / 2.0).tolist(), frame_quat(d if half > 0 else [0, 0, 1], [1.0, 0.0, 0.0]), half


def band_of(z: float) -> str:
    for name, top in BAND_EDGES:
        if z < top or (name == "low" and z <= top):
            return name
    return "high"


def clock_of(p_local, alley_dir) -> int:
    """Clock position 1-12 of a tree-frame point as seen from the alley (12 up, 3 right, 6 down, 9 left)."""
    view = -_unit([alley_dir[0], alley_dir[1], 0.0])          # looking from the alley toward the trunk
    right = np.array([view[1], -view[0], 0.0])                 # viewer's right on the tree face
    lat = float(np.dot(np.asarray(p_local, float), right))
    up = float(p_local[2]) - 1.0                               # centre of the face at 1.0 m (assumption)
    ang = math.degrees(math.atan2(lat, up)) % 360.0            # 0 = up, clockwise
    c = int(round(ang / 30.0)) % 12
    return 12 if c == 0 else c


# ---------------------------------------------------------------------- config
def trees_dir() -> str:
    return os.environ.get("NANDI_TREES_CONFIG", "").strip() or TREES_DIR


def _yaml(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TreeConfigError(f"{path}: not a mapping")
    return data


def validate_species(sp: dict) -> List[str]:
    errs = []
    for k in ("schema", "species", "archetype", "height_m", "foliage"):
        if k not in sp:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if sp["schema"] != SPECIES_SCHEMA:
        errs.append(f"schema {sp['schema']!r} != {SPECIES_SCHEMA}")
    if sp["archetype"] not in ARCHETYPES:
        errs.append(f"archetype {sp['archetype']!r} not in {ARCHETYPES}")
    lo, hi = sp["height_m"]
    if not 0.5 <= lo <= hi <= 4.0:
        errs.append("height_m must be within 0.5..4.0 m")
    if sp["archetype"] == "canopy":
        for k in ("trunk_frac", "trunk_r_m", "canopy_r_frac", "scaffolds", "fruit", "leaves", "decals"):
            if k not in sp:
                errs.append(f"canopy archetype needs {k}")
        if sp.get("leaves") and int(sp["leaves"]["n"]) > LEAF_QUADS_MAX:
            errs.append(f"leaves.n > {LEAF_QUADS_MAX}")
    elif sp["archetype"] == "papaya":
        for k in ("trunk_r_m", "petioles", "fruit"):
            if k not in sp:
                errs.append(f"papaya archetype needs {k}")
    else:
        for k in ("pseudostem_r_m", "leaves_n", "bunch", "suckers"):
            if k not in sp:
                errs.append(f"banana archetype needs {k}")
    return errs


def load_species(name: str, directory: Optional[str] = None) -> dict:
    path = os.path.join(directory or trees_dir(), f"{name}.yaml")
    if not os.path.isfile(path):
        raise TreeConfigError(f"no tree species config {path}")
    sp = _yaml(path)
    errs = validate_species(sp)
    if errs:
        raise TreeConfigError(f"{path}: " + "; ".join(errs))
    if sp["species"] != name:
        raise TreeConfigError(f"{path}: species {sp['species']!r} != file name {name!r}")
    return sp


def load_all_species(directory: Optional[str] = None) -> Dict[str, dict]:
    d = directory or trees_dir()
    names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(d, "*.yaml")))
    return {n: load_species(n, d) for n in names}


def config_digest(directory: Optional[str] = None) -> str:
    d = directory or trees_dir()
    h = hashlib.sha1()
    for f in sorted(glob.glob(os.path.join(d, "*.yaml"))):
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def season_name(field: Optional[dict] = None) -> str:
    s = os.environ.get("NANDI_SEASON", "").strip()
    if not s:
        s = (field or {}).get("season") or DEFAULT_SEASON
        s = FIELD_SEASON_ALIASES.get(s, s)
    if s not in SEASONS:
        raise TreeConfigError(f"season {s!r} not in {SEASONS}")
    return s


# ---------------------------------------------------------------------- registry
_CANDIDATES_CACHE: Dict[str, dict] = {}


def load_candidates(path: str = CANDIDATES_JSON) -> dict:
    """WP0c ``orchard_candidates.json`` (read-only here): candidate trees, targets with frozen status, leaves."""
    if path not in _CANDIDATES_CACHE:
        if not os.path.isfile(path):
            _CANDIDATES_CACHE[path] = {}
        else:
            with open(path, encoding="utf-8") as fh:
                _CANDIDATES_CACHE[path] = json.load(fh)
    return _CANDIDATES_CACHE[path]


def _wp0c_anchors(cand: dict) -> Dict[str, dict]:
    """tree_id -> {wood, targets[], leaves[]} from the WP0c file, with each target's frozen status."""
    if not cand:
        return {}
    status = {t["id"]: t for t in (cand.get("frozen", {}).get("targets") or [])}
    out: Dict[str, dict] = {}
    for t in cand.get("candidates", {}).get("trees", []):
        out[t["id"]] = {"wood": copy.deepcopy(t.get("wood") or []), "targets": [], "leaves": [],
                        "trunk_h_m": t.get("trunk_h"), "seed": t.get("seed")}
    for tg in cand.get("candidates", {}).get("targets", []):
        if tg["tree"] not in out:
            continue
        s = status.get(tg["id"], {})
        e = {k: tg[k] for k in ("id", "task", "p") if k in tg}
        for k in ("stem_mm", "fruit_kg", "neighbours", "neighbour_r"):
            if tg.get(k) is not None:
                e[k] = copy.deepcopy(tg[k])
        e["status"] = s.get("status", "CANDIDATE")
        if s.get("reason"):
            e["reason"] = s["reason"]
        out[tg["tree"]]["targets"].append(e)
    for lf in cand.get("candidates", {}).get("leaves", []):
        if lf["tree"] in out:
            out[lf["tree"]]["leaves"].append({k: copy.deepcopy(lf[k]) for k in ("id", "p", "n", "kind")})
    return out


CULTIVAR = {"mango": "Dashehari", "guava": "Allahabad Safeda", "papaya": "Red Lady", "banana": "Grand Naine"}
FORM = {"mango": "dwarf", "guava": "grafted", "papaya": "dwarf", "banana": "dwarf"}
GRAFT_H = {"mango": 0.3, "guava": 0.3}     # above the WP0c sprout bases (0.18-0.24 m); OS 3.3 range 0.2-0.3
RIBBONS = ("red", "blue", "yellow", "white")


def _demo_entries(cand: dict, sim_date: str) -> List[dict]:
    anchors = _wp0c_anchors(cand)
    today = _dt.date.fromisoformat(sim_date)
    entries = []
    for n, t in enumerate(cand.get("candidates", {}).get("trees", []), start=1):
        sp, tid = t["species"], t["id"]
        row = "OT" if tid.startswith("OT") else tid.split("-")[0]
        e = {"tree_id": tid, "physical_tag": f"{n:02d}", "block_id": "B1", "row_id": row, "crop": sp,
             "cultivar": CULTIVAR.get(sp, ""), "form": FORM.get(sp, "dwarf"), "x": t["x"], "y": t["y"],
             "height_m": t["H"], "canopy_r_m": round(float(t["canopy_r"]), 4), "trunk_r_m": t["trunk_r"],
             "trunk_h_m": t.get("trunk_h"), "graft_union_h_m": GRAFT_H.get(sp),
             "orbit_clear": t.get("alley_sides") == ["all"], "alley_sides": t.get("alley_sides"),
             "alley_clear_height_m": t.get("alley_clear_height_m"), "drip_line_offset_m": 0.4,
             "rope_lines": [], "props": [], "fruit_set_date": None, "bagging_date": None, "ribbon_colour": None,
             "last_visit": None, "flags": [], "seed": t.get("seed", n)}
        if sp in ("mango", "guava"):
            e["fruit_set_date"] = (today - _dt.timedelta(days=95 if sp == "mango" else 120)).isoformat()
        if sp == "papaya":
            e["fruit_set_date"] = (today - _dt.timedelta(days=160)).isoformat()
        if sp == "banana":
            k = int(tid.split("-")[1]) - 1
            e["ribbon_colour"] = RIBBONS[k % len(RIBBONS)]
            e["shooting_date"] = (today - _dt.timedelta(days=(100, 70, 105, 50)[k % 4])).isoformat()
        e.update(anchors.get(tid, {}))
        entries.append(e)
    by_id = {e["tree_id"]: e for e in entries}
    # demo details of OS 1.2 / 3.3 / 3.6 (positions of WP4 fixtures stay WP4's; these are registry facts)
    if "OT-1" in by_id:
        by_id["OT-1"]["decals"] = [{"class": "gummosis", "p": [0.0, -0.06, 0.30]},
                                   {"class": "sooty_mould", "p": [-0.55, -0.62, 1.30]}]
        by_id["OT-1"]["leaf_decals"] = ["anthracnose", "mealybug"]
        by_id["OT-1"]["flags"] = ["tree360", "leaf_scan", "trunk_scan", "harvest_window", "prune"]
    if "R4-1" in by_id and "R4-2" in by_id:
        a, b = by_id["R4-1"], by_id["R4-2"]
        a["rope_lines"] = [{"to": "R4-2", "a": [a["x"], a["y"], 1.4], "b": [b["x"], b["y"], 1.4], "h_m": 1.4}]
    if "R4-2" in by_id:
        by_id["R4-2"]["drooping_leaf"] = {"tip_z_m": 1.6, "reach_m": 1.15}
    if "R4-3" in by_id:
        by_id["R4-3"]["props"] = [{"kind": "bamboo", "under": "bunch"}]
    if "R3-1" in by_id:              # OS 1.2 step 8, papaya P-1: the second ripe fruit is crowded (no_clearance)
        by_id["R3-1"]["crowded_ripe_k"] = 2
    return entries


def default_registry(world: str, species: Optional[Sequence[str]] = None, sim_date: str = DEFAULT_SIM_DATE,
                     candidates_path: str = CANDIDATES_JSON) -> List[dict]:
    """The tree registry an orchard world uses until WP4's ``config/fields/<world>.json`` carries ``trees``.

    ``orchard_demo`` / ``orchard_demo_monsoon``: the 17 WP0c candidate trees (OT-1, R1 mango 5 x 2.5 m, R2 guava,
    R3 papaya, R4 banana). ``orchard_lite`` (OS 4.1): one tree per configured species (OT-1, R2-1, R3-1, R4-1
    for the four MVP crops, then one generated tree for every further species in ``config/trees``) plus a
    3 x 2 m mango hedgerow pair (HR-1, HR-2). Other worlds: ``[]`` (no trees: the legacy world is unchanged)."""
    if world not in ORCHARD_WORLDS:
        return []
    cand = load_candidates(candidates_path)
    demo = _demo_entries(cand, sim_date)
    if world in ("orchard_demo", "orchard_demo_monsoon"):
        return demo
    species = list(species) if species is not None else sorted(load_all_species())
    by_id = {e["tree_id"]: e for e in demo}
    out = []
    for sp, tid in (("mango", "OT-1"), ("guava", "R2-1"), ("papaya", "R3-1"), ("banana", "R4-1")):
        if sp in species and tid in by_id:
            out.append(copy.deepcopy(by_id[tid]))
    for k, x in enumerate((2.0, 4.0)):        # UHDP 3 x 2 m hedgerow pair, generated (no WP0c anchors)
        out.append({"tree_id": f"HR-{k + 1}", "physical_tag": f"H{k + 1}", "block_id": "B2", "row_id": "HR",
                    "crop": "mango", "cultivar": "Amrapali", "form": "hedgerow", "x": x, "y": -8.0,
                    "height_m": 1.8, "canopy_r_m": 0.75, "trunk_r_m": 0.05, "graft_union_h_m": 0.25,
                    "orbit_clear": False, "alley_sides": ["-y", "+y"], "drip_line_offset_m": 0.4,
                    "fruit_set_date": (_dt.date.fromisoformat(sim_date) - _dt.timedelta(days=95)).isoformat(),
                    "flags": [], "seed": 50 + k})
    extra = [s for s in species if s not in ("mango", "guava", "papaya", "banana")]
    for k, sp in enumerate(extra):            # a 5th (6th...) crop by config only
        cfg = load_species(sp)
        h = float(sum(cfg["height_m"]) / 2.0)
        out.append({"tree_id": f"X{k + 1}-1", "physical_tag": f"X{k + 1}", "block_id": "B3", "row_id": f"X{k + 1}",
                    "crop": sp, "cultivar": "", "form": "grafted", "x": 10.0 + 3.0 * k, "y": -8.0, "height_m": h,
                    "alley_sides": ["-y", "+y"], "drip_line_offset_m": 0.4,
                    "fruit_set_date": (_dt.date.fromisoformat(sim_date) - _dt.timedelta(days=120)).isoformat(),
                    "flags": [], "seed": 60 + k})
    return out


def registry(field: dict, species: Optional[Sequence[str]] = None) -> List[dict]:
    """Registry entries for this field: ``field['trees']`` when present (WP4). Each entry is merged with its WP0c
    anchors (wood, targets, leaves) when the tree id matches a candidate and the entry has none, and with the
    WP23 demo registry facts WP4 does not carry (``REGISTRY_FACT_KEYS``; flags are unioned). On
    ``orchard_lite`` one generated tree per further configured species (``X<k>-1``) is appended. Without
    ``field['trees']``: the WP23 default registry."""
    sim_date = field.get("sim_date") or DEFAULT_SIM_DATE
    trees = field.get("trees")
    if not trees:
        return default_registry(field.get("world", ""), species, sim_date)
    anchors = _wp0c_anchors(load_candidates())
    facts = {e["tree_id"]: e for e in _demo_entries(load_candidates(), sim_date)}
    out = []
    for e in trees:
        e = copy.deepcopy(e)
        a = anchors.get(e.get("tree_id"))
        if a:
            for k, v in a.items():
                e.setdefault(k, copy.deepcopy(v))
        f = facts.get(e.get("tree_id"))
        if f:
            for k in REGISTRY_FACT_KEYS:
                if k in f:
                    e.setdefault(k, copy.deepcopy(f[k]))
            e["flags"] = list(e.get("flags") or []) + [x for x in f.get("flags") or [] if x not in (e.get("flags") or [])]
        out.append(e)
    if field.get("world") == "orchard_lite":
        have = {e.get("crop") for e in out}
        species = list(species) if species is not None else sorted(load_all_species())
        out += [x for x in default_registry("orchard_lite", species, sim_date)
                if x["tree_id"].startswith("X") and x["crop"] not in have]
    return out


def normalize_entry(e: dict) -> dict:
    """Required registry fields with defaults (MISSION-PRESETS 0.1). Raises on a missing id, crop or position."""
    for k in ("tree_id", "crop"):
        if not e.get(k):
            raise TreeConfigError(f"registry entry needs {k}: {e!r}")
    out = copy.deepcopy(e)
    if "x" not in out or "y" not in out:
        geo = out.get("geo") or {}
        pos = out.get("pos") or geo.get("local_xy")
        if not pos and "x" in geo and "y" in geo:      # WP4 field_spec tree: geo {frame, x, y}
            pos = (geo["x"], geo["y"])
        if not pos:
            raise TreeConfigError(f"registry entry {e['tree_id']} has no x/y position")
        out["x"], out["y"] = float(pos[0]), float(pos[1])
    out.setdefault("physical_tag", out["tree_id"])
    out.setdefault("alley_sides", ["-y", "+y"])
    out.setdefault("flags", [])
    return out


def body_name(tree_id: str) -> str:
    return "tree_" + "".join(c if c.isalnum() or c in "-_" else "_" for c in str(tree_id))


# ---------------------------------------------------------------------- generator
def _rng(seed: int, tree_id: str, entry_seed=None) -> np.random.Generator:
    h = int(hashlib.sha1(f"{int(seed)}:{tree_id}:{entry_seed}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


def _u(rng, rng_pair) -> float:
    lo, hi = rng_pair
    return float(lo) if lo == hi else float(rng.uniform(lo, hi))


def _ui(rng, rng_pair) -> int:
    lo, hi = int(rng_pair[0]), int(rng_pair[1])
    return lo if lo == hi else int(rng.integers(lo, hi + 1))


class _Plan:
    """Accumulates one tree's geoms / meshes / sites / truth; ``done()`` returns the JSON-able plan."""

    def __init__(self, entry: dict, sp: dict, profile: dict, season: str, seed: int):
        self.e, self.sp, self.pf, self.season, self.seed = entry, sp, profile, season, int(seed)
        self.tid = entry["tree_id"]
        self.body = body_name(self.tid)
        self.pos = [float(entry["x"]), float(entry["y"]), float(entry.get("z", 0.0))]
        self.rng = _rng(seed, self.tid, entry.get("seed"))
        self.geoms: List[dict] = []
        self.meshes: List[dict] = []
        self.sites: List[dict] = []
        self.counts: Dict[str, int] = {}
        self.fruit: List[dict] = []
        self.prune: List[dict] = []
        self.wood: List[dict] = []
        self.leaves: List[dict] = []
        self.decals: List[dict] = []
        self.extras: Dict[str, object] = {}
        self._mesh: Dict[str, Tuple[List[float], List[int]]] = {}
        alley = entry.get("alley_sides") or ["-y"]
        self.alley_dir = {"-y": (0, -1), "+y": (0, 1), "-x": (-1, 0), "+x": (1, 0)}.get(alley[0], (0, -1))

    # -- primitives
    def _name(self, kind: str) -> str:
        k = self.counts.get(kind, 0)
        self.counts[kind] = k + 1
        return f"{self.body}_{kind}_{k}"

    def geom(self, kind: str, gtype: str, group: int, rgba, contype: int = 0, **kw) -> dict:
        g = {"name": self._name(kind), "kind": kind, "type": gtype, "group": int(group), "contype": int(contype),
             "conaffinity": 0, "rgba": _r(list(rgba), 4), "hideable": kind in HIDEABLE_KINDS}
        for k, v in kw.items():
            g[k] = _r(v)
        self.geoms.append(g)
        return g

    def capsule(self, kind, a, b, r, group, rgba, contype=0) -> dict:
        return self.geom(kind, "capsule", group, rgba, contype, fromto=list(a) + list(b), size=[r, 0.0, 0.0])

    def site(self, kind: str, p, **meta) -> dict:
        s = {"name": f"{self.body}_{kind}", "pos": _r(list(p))}
        s.update(meta)
        self.sites.append(s)
        return s

    def blob(self, mesh: str, c, hs, quat_R: Optional[np.ndarray] = None):
        """Add a flattened octahedron (6 verts, 8 faces) to merged mesh ``mesh``."""
        V = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * np.asarray(hs)
        if quat_R is not None:
            V = V @ np.asarray(quat_R).T
        V = V + np.asarray(c, float)
        F = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
        verts, faces = self._mesh.setdefault(mesh, ([], []))
        base = len(verts) // 3
        verts.extend(V.reshape(-1).tolist())
        faces.extend((F + base).reshape(-1).tolist())

    def stick(self, mesh: str, a, b, w: float):
        """A thin elongated octahedron from a to b (twigs, petioles) in a merged mesh."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return
        q = frame_quat(d, [0, 0, 1])
        R = _quat_mat(q)
        self.blob(mesh, (a + b) / 2.0, [w, w, L / 2.0], R)

    def flush_mesh(self, mesh: str, kind: str, group: int, rgba):
        if mesh not in self._mesh or not self._mesh[mesh][0]:
            return
        verts, faces = self._mesh.pop(mesh)
        name = f"{self.body}_{mesh}"
        self.meshes.append({"name": name, "vert": _r(verts, 5), "face": [int(f) for f in faces]})
        self.geom(kind, "mesh", group, rgba, 0, mesh=name)

    def world(self, p) -> List[float]:
        return _r((np.asarray(p, float) + np.asarray(self.pos)).tolist())

    # -- shared pieces
    def envelope_ellipsoid(self, c, size):
        self.geom("envelope", "ellipsoid", GRP_QUERY, ENVELOPE_RGBA, 0, pos=list(c), size=list(size))

    def envelope_capsule(self, a, b, r):
        self.capsule("envelope", a, b, r, GRP_QUERY, ENVELOPE_RGBA, 0)

    def keep_wood(self, role, a, b, r, rgba=WOOD_RGBA):
        g = self.capsule("wood", a, b, r, GRP_WOOD, rgba, contype=1)
        self.wood.append({"geom": g["name"], "role": role, "keep": True, "diameter_mm": _r(2000.0 * r, 1),
                          "a": _r(list(a)), "b": _r(list(b))})
        return g

    def prune_target(self, job, a, b, diameter_mm, cut_p, rgba=SPROUT_RGBA, group=GRP_WOOD, wp0c=None, gtype=None,
                     extra=None):
        r = float(diameter_mm) / 2000.0
        if gtype == "box":
            pos, quat, half = capsule_frame(a, b)
            g = self.geom("prune", "box", group, rgba, 0, pos=pos, quat=quat, size=[r * 9.0, 0.002, half])
        else:
            g = self.capsule("prune", a, b, max(r, 0.002), group, rgba, 0)
        action, reason = crops.prune_decision(self.pf, job, float(diameter_mm), float(cut_p[2]))
        t = {"id": f"{self.tid}:prune:{len(self.prune) + 1}", "geom": g["name"], "job": job, "keep": False,
             "diameter_mm": _r(float(diameter_mm), 1), "cut_p": _r(list(cut_p)), "cut_world": self.world(cut_p),
             "a": _r(list(a)), "b": _r(list(b)), "z_m": _r(float(cut_p[2]), 3), "in_band": cut_p[2] <= PICK_BAND_Z_M,
             "action": action, "reason": reason, "paste_required": crops.paste_required(self.pf, diameter_mm),
             "h_class": crops.h_class(float(cut_p[2])), "clock": clock_of(cut_p, self.alley_dir)}
        if wp0c:
            t["wp0c"] = wp0c
        if extra:
            t.update(extra)
        self.prune.append(t)
        return t

    def decal_class_pool(self, scale: str) -> List[dict]:
        if self.season == "demo_all_fruiting":
            return [c for c in self.pf["symptoms"] if c.get("scale") == scale]
        return crops.symptom_classes(self.pf, self.season, scale)

    def decal(self, cls: dict, p, n, r, unit=None):
        n = _unit(n)
        q = frame_quat(n, [0, 0, 1])
        g = self.geom("decal", "cylinder", GRP_DECAL, DECAL_RGBA.get(cls["class"], DECAL_DEFAULT_RGBA), 0,
                      pos=(np.asarray(p, float) + 0.002 * n).tolist(), quat=q, size=[r, 0.0008, 0.0])
        d = {"id": f"{self.tid}:decal:{len(self.decals) + 1}", "geom": g["name"], "class": cls["class"],
             "unit": unit or cls.get("unit"), "seasons": list(cls.get("seasons", [])),
             "season": self.season, "band": band_of(float(p[2])), "h_m": _r(float(p[2]), 3),
             "h_class": crops.h_class(float(p[2])), "clock": clock_of(p, self.alley_dir), "p": _r(list(p)),
             "world": self.world(p), "diameter_mm": _r(2000.0 * r, 1), "stage": "lead"}
        self.decals.append(d)
        return d

    def leaf(self, k: int, p, n, kind: str, size, decal_cls: Optional[dict] = None, functional: bool = True):
        p, n = np.asarray(p, float), _unit(n)
        radial = np.array([p[0], p[1], 0.0])
        q = frame_quat(n, radial if np.linalg.norm(radial) > 1e-6 else [1, 0, 0])
        g = self.geom("leaf", "box", GRP_FOLIAGE, [0.18, 0.50, 0.18, 1.0] if functional else DEAD_RGBA, 0,
                      pos=p.tolist(), quat=q, size=[size[0] / 2.0, size[1] / 2.0, 0.0015])
        s = self.site(f"leaf_k_{k}", p)
        lf = {"id": f"{self.tid}:leaf:{k + 1}", "k": k, "geom": g["name"], "site": s["name"], "p": _r(p.tolist()),
              "world": self.world(p), "n": _r(n.tolist()), "kind": kind, "in_band": bool(p[2] <= PICK_BAND_Z_M),
              "h_class": crops.h_class(float(p[2])), "functional": functional, "decal": None}
        if decal_cls is not None:
            r = _u(self.rng, (self.sp.get("leaf_decals") or {}).get("r_m", [0.004, 0.0075]))
            d = self.decal(decal_cls, p + 0.001 * n, n, r, unit="per_leaf")
            d["leaf"] = lf["id"]
            lf["decal"] = d["class"]
        self.leaves.append(lf)
        return lf

    # -- fruit
    def add_fruit(self, kind_hint: str, centre, length: float, shape, mass: float, stage_id: str, axis,
                  stem_end, anchor_top, leave_m: float, days: Optional[float], wp0c=None, bagged=False,
                  target_p=None, extra=None):
        st = crops.stage(self.pf, stage_id)
        size = [shape[0] * length, shape[1] * length, shape[2] * length]
        k = len(self.fruit)
        if bagged:
            g = self.geom("bag", "box", GRP_FRUIT, BAG_RGBA, 0, pos=list(centre),
                          size=[size[0] * 1.15, size[1] * 1.15, size[2] * 1.2])
        else:
            g = self.geom("fruit", "ellipsoid", GRP_FRUIT, st["rgba"], 0, pos=list(centre), size=size)
        cues = {}
        for ck, cv in (st.get("cues") or {}).items():
            cues[ck] = _r(_u(self.rng, cv), 2) if isinstance(cv, list) else cv
        axis = _unit(axis)
        cut = np.asarray(stem_end, float) + leave_m * axis
        tz = float((target_p if target_p is not None else (cut if self.pf["harvest_method"] == "clip" else centre))[2])
        f = {"fruit_id": f"{self.tid}:f{k + 1:02d}", "k": k, "tree_id": self.tid, "crop": self.pf["crop"],
             "geom": g["name"], "static_kind": "bag" if bagged else "fruit", "stage": stage_id,
             "harvest_stage": bool(st.get("harvest")), "bagged": bool(bagged), "mass_kg": _r(mass, 4),
             "size_m": _r(size), "length_mm": _r(1000.0 * length, 1), "rgba": _r(list(st["rgba"]), 4),
             "centre": _r(list(centre)), "centre_world": self.world(centre), "stem_axis": _r(axis.tolist()),
             "stem_end": _r(list(stem_end)), "anchor_top": _r(list(anchor_top)), "cut_k": _r(cut.tolist()),
             "cut_k_world": self.world(cut), "stem_end_world": self.world(stem_end),
             "anchor_top_world": self.world(anchor_top), "pedicel_leave_m": _r(leave_m),
             "pedicel_len_m": _r(float(np.linalg.norm(np.asarray(anchor_top) - np.asarray(stem_end)))),
             "pool_kind": self.pf["pool_kind"], "target_z_m": _r(tz, 3), "in_band": bool(tz <= PICK_BAND_Z_M + 1e-9),
             "band": band_of(float(centre[2])), "h_class": crops.h_class(float(centre[2])),
             "clock": clock_of(centre, self.alley_dir), "days_since_fruit_set": None if days is None else int(days),
             "cues": cues}
        if wp0c:
            f["wp0c"] = wp0c
        if extra:
            f.update(extra)
        self.fruit.append(f)
        return f

    def stage_list(self, n_in_band: Sequence[bool], mature_in_band: int, mature_frac_out: float) -> List[str]:
        hs = crops.harvest_stages(self.pf)
        imm = [s["id"] for s in self.pf["stages"] if not s.get("harvest")]
        ripe, green = hs[0], (imm[0] if imm else hs[0])
        out, left = [], int(mature_in_band)
        for inb in n_in_band:
            if inb:
                out.append(ripe if left > 0 else green)
                left -= 1
            else:
                out.append(ripe if self.rng.uniform() < mature_frac_out else green)
        return out

    def days_for(self, stage_id: str) -> Optional[int]:
        key = "days_since_shooting" if self.pf["harvest_window"]["basis"] == "days_since_shooting" \
            else "days_since_fruit_set"
        rng_ = crops.stage(self.pf, stage_id).get(key)
        if not rng_:
            return None
        lo, hi = rng_
        return int(self.rng.integers(int(lo), int(min(hi, lo + 20)) + 1))

    # -- result
    def done(self) -> dict:
        render = sum(1 for g in self.geoms if g["group"] <= RENDER_GROUP_MAX)
        e = self.e
        truth = {"tree_id": self.tid, "physical_tag": e.get("physical_tag"), "crop": self.pf["crop"],
                 "cultivar": e.get("cultivar"), "form": e.get("form"), "season": self.season,
                 "fruit": self.fruit, "bags_n": sum(1 for f in self.fruit if f["bagged"]),
                 "decals": self.decals, "wood": self.wood, "prune": self.prune, "leaves": self.leaves}
        truth.update(self.extras)
        plan = {"schema": PLAN_SCHEMA, "tree_id": self.tid, "body": self.body, "crop": self.pf["crop"],
                "archetype": self.sp["archetype"], "season": self.season, "seed": self.seed,
                "pos": _r(self.pos), "entry": _r(_jsonable(e)), "geoms": self.geoms, "meshes": self.meshes,
                "sites": self.sites, "render_geoms": render,
                "leaf_quads": sum(1 for g in self.geoms if g["kind"] == "leaf"), "truth": truth}
        plan = _jsonable(plan)
        plan["hash"] = plan_hash(plan)
        return plan


def _np_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _jsonable(x):
    """Plain JSON types only (numpy scalars and arrays converted), so plans hash and serialise anywhere."""
    return json.loads(json.dumps(x, default=_np_default))


def _quat_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def canopy_halfwidth(archetype: str, H: float, canopy_r: float, trunk_h: float, z: float,
                     clear_h: Optional[float] = None) -> float:
    """Horizontal radius of the foliage envelope at height z (the WP0c ``canopy_halfwidth`` model)."""
    if archetype == "papaya":
        z0, z1 = H - 0.6, H + 0.1
    elif archetype == "banana":
        z0, z1 = (clear_h or 1.3), H + 0.4
    else:
        z0, z1 = trunk_h * 0.75, H
    if not (z0 < z < z1):
        return 0.0
    zc, hz = (z0 + z1) / 2.0, (z1 - z0) / 2.0
    return canopy_r * math.sqrt(max(0.0, 1.0 - ((z - zc) / hz) ** 2))


def _season_has_fruit(season: str, crop: str) -> bool:
    if crop in ("papaya", "banana"):          # non-seasonal in the sim: always fruiting
        return True
    return season in ("fruit_set", "bagged", "mature", "demo_all_fruiting")


def _gen_canopy(P: _Plan):
    e, sp, pf, rng = P.e, P.sp, P.pf, P.rng
    H = float(e.get("height_m") or _u(rng, sp["height_m"]))
    cr = float(e.get("canopy_r_m") or min(sp["canopy_r_frac"] * H, 1.1))
    trunk_r = float(e.get("trunk_r_m") or _u(rng, sp["trunk_r_m"]))
    trunk_h = float(e.get("trunk_h_m") or round(sp["trunk_frac"] * H, 3))
    targets = e.get("targets") or []
    # -- keep wood: trunk + scaffolds (WP0c skeleton when given) + branches
    skel = e.get("wood") or []
    if skel:
        tr = skel[0]
        P.keep_wood("trunk", tr["a"], tr["b"], tr["r"])
        scaff = [(np.asarray(w["a"], float), np.asarray(w["b"], float), float(w["r"])) for w in skel[1:]]
        trunk_h = float(tr["b"][2])
    else:
        P.keep_wood("trunk", [0, 0, 0], [0, 0, trunk_h], trunk_r)
        n_sc = _ui(rng, sp["scaffolds"]["n"])
        L = 0.55 * H / 2.2
        scaff = []
        a0 = float(rng.uniform(0, 2 * math.pi))
        for i in range(n_sc):
            a = a0 + 2 * math.pi * i / n_sc + float(rng.uniform(-0.25, 0.25))
            b = np.array([0.7 * L * math.cos(a), 0.7 * L * math.sin(a), trunk_h + 0.7 * L])
            scaff.append((np.array([0, 0, trunk_h]), b, float(sp["scaffolds"]["r_m"])))
    for a, b, r in scaff:
        P.keep_wood("scaffold", a, b, r)
    br_r = float(sp["scaffolds"].get("branch_r_m", 0.015))
    tips = []
    for a, b, r in scaff:
        az = math.atan2(b[1], b[0])
        for j in range(int(sp["scaffolds"].get("branches_each", 2))):
            a2 = az + (-0.55 if j == 0 else 0.55) + float(rng.uniform(-0.15, 0.15))
            L2 = 0.28 * H
            p2 = b + L2 * np.array([0.6 * math.cos(a2), 0.6 * math.sin(a2), float(rng.uniform(0.55, 0.85))])
            p2[2] = min(p2[2], H - 0.12)
            P.keep_wood("branch", b, p2, br_r)
            tips.append(p2)
    gu = e.get("graft_union_h_m")
    if gu is None and sp.get("graft_union"):
        gu = round(_u(rng, sp["graft_union"]["h_m"]), 3)
    if gu:
        g = P.geom("wood", "ellipsoid", GRP_WOOD, [0.40, 0.27, 0.16, 1.0], 1, pos=[0, 0, gu],
                   size=[trunk_r * 1.3, trunk_r * 1.3, 0.025])
        P.wood.append({"geom": g["name"], "role": "graft_union", "keep": True, "h_m": _r(gu, 3),
                       "diameter_mm": _r(2600.0 * trunk_r, 1)})
    P.extras["graft_union_h_m"] = _r(gu, 3) if gu else None
    # -- envelope (query-only) and merged foliage
    z0, z1 = trunk_h * 0.75, H
    zc, hz = (z0 + z1) / 2.0, (z1 - z0) / 2.0
    P.envelope_ellipsoid([0, 0, zc], [cr, cr, hz])
    P.extras["envelope"] = {"type": "ellipsoid", "c": _r([0, 0, zc]), "size": _r([cr, cr, hz])}
    fol = sp["foliage"]
    for _ in range(int(fol["blobs"])):
        th, ph = float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(-0.9, 1.2))
        rr = float(rng.uniform(0.72, 0.98))
        c = [rr * cr * math.cos(th) * math.cos(ph), rr * cr * math.sin(th) * math.cos(ph), zc + rr * hz * math.sin(ph)]
        P.blob("canopy", c, fol["blob_m"], _quat_mat(frame_quat([0, 0, 1], [math.cos(th), math.sin(th), 0])))
    # -- fruit
    fr = sp["fruit"]
    leave_m = float(pf["pedicel_leave_cm"]["default"]) / 100.0
    fruits = []    # (cut_or_stem_end_point, wp0c, target_kind)
    task = "mango_pedicel_cut" if pf["crop"] == "mango" else "guava_clip"
    for t in targets:
        if t.get("task") == task:
            fruits.append((np.asarray(t["p"], float), {"id": t["id"], "status": t.get("status"),
                                                       "reason": t.get("reason")}))
    has_fruit = _season_has_fruit(P.season, pf["crop"])
    n_total = _ui(rng, fr["n"])
    placed = [f[0] for f in fruits]
    tries = 0
    while len(fruits) < n_total and tries < 400:
        tries += 1
        z = float(rng.uniform(max(fr["shell_z_m"][0], 0.85 if targets else fr["shell_z_m"][0]),
                              min(fr["shell_z_m"][1], H - 0.15)))
        hw = canopy_halfwidth("canopy", H, cr, trunk_h, z)
        rr = (hw if hw > 0 else 0.8 * cr) * float(rng.uniform(0.85, 1.0))
        th = float(rng.uniform(0, 2 * math.pi))
        p = np.array([rr * math.cos(th), rr * math.sin(th), z])
        if any(np.linalg.norm(p - q) < fr["min_gap_m"] for q in placed):
            continue
        placed.append(p)
        fruits.append((p, None))
    if not has_fruit:
        fruits = []
    in_band = [bool(p[2] <= PICK_BAND_Z_M + 1e-9) for p, _ in fruits]
    mib = int(e.get("mature_in_band_n", fr.get("demo_mature_in_band", 0)))
    stages = P.stage_list(in_band, mib, float(fr.get("mature_frac_out_of_band", 0.5)))
    if P.season == "fruit_set":
        stages = [s for s in stages]
        stages = [next(s["id"] for s in pf["stages"] if not s.get("harvest"))] * len(fruits)
    bag = P.season == "bagged" and bool(sp.get("bags")) and pf.get("bagging", {}).get("used")
    scale = 0.4 if P.season == "fruit_set" else 1.0
    mass_rng = pf.get("mass_kg") or [0.15, 0.3]
    for (p, wp0c), st in zip(fruits, stages):
        length = _u(rng, fr["length_m"]) * scale
        hzf = fr["shape"][2] * length
        ped = max(_u(rng, fr["pedicel_m"]), leave_m + 0.02)
        cut = p                                   # the WP0c point is the cut point (cut_k)
        stem_end = cut - np.array([0, 0, leave_m])
        anchor = stem_end + np.array([0, 0, ped])
        centre = stem_end - np.array([0, 0, hzf])
        f = P.add_fruit("fruit", centre, length, fr["shape"], _u(rng, mass_rng) * (scale ** 3), st, [0, 0, 1],
                        stem_end, anchor, leave_m, P.days_for(st), wp0c=wp0c, bagged=bag, target_p=cut)
        # fruiting twig (visual, merged into the foliage mesh): anchor -> into the canopy
        inward = np.array([-anchor[0], -anchor[1], 0.0])
        tip = anchor + 0.25 * _unit(inward + np.array([0, 0, 1.2 * np.linalg.norm(inward) + 0.2]))
        P.stick("canopy", anchor, tip, 0.006)
        f["twig"] = _r([anchor.tolist(), tip.tolist()])
    P.flush_mesh("canopy", "foliage", GRP_FOLIAGE, fol["rgba"])
    # -- panicles (panicle season)
    pan = sp.get("panicles")
    P.extras["panicles"] = []
    if P.season == "panicle" and pan and tips:
        n_p, n_m = _ui(rng, pan["n"]), _ui(rng, pan["malformed"])
        for i in range(n_p):
            b = tips[i % len(tips)] + np.array([float(rng.uniform(-0.1, 0.1)), float(rng.uniform(-0.1, 0.1)), 0.05])
            malformed = i < n_m
            if malformed:
                g = P.geom("panicle", "sphere", GRP_FOLIAGE, [0.55, 0.45, 0.30, 1.0], 0, pos=b.tolist(),
                           size=[0.06, 0, 0])
            else:
                g = P.capsule("panicle", b, b + np.array([0, 0, 0.2]), 0.025, GRP_FOLIAGE, [0.85, 0.75, 0.45, 1.0])
            P.extras["panicles"].append({"geom": g["name"], "malformed": malformed, "p": _r(b.tolist()),
                                         "h_m": _r(float(b[2]), 3), "in_band": bool(b[2] <= PICK_BAND_Z_M)})
    # -- prune candidates (contype 0): rootstock sprouts, skirt shoots, a thick skirt branch, water shoot, dead twig
    for t in targets:
        if t.get("task") not in ("prune_rootstock_sprout", "prune_skirt_shoot"):
            continue
        p = np.asarray(t["p"], float)
        az = math.atan2(p[1], p[0])
        rad = np.array([math.cos(az), math.sin(az), 0.0])
        wp = {"id": t["id"], "status": t.get("status"), "reason": t.get("reason")}
        if t["task"] == "prune_rootstock_sprout":
            a = trunk_r * rad + np.array([0, 0, p[2] - 0.04])
            d = _unit(p - a)
            P.prune_target("rootstock_sprout", a, p + 0.14 * d, t.get("stem_mm", 6), p, wp0c=wp,
                           extra={"below_graft_union": bool(gu and a[2] < gu) or not gu})
        else:
            a = trunk_r * rad + np.array([0, 0, max(0.25, p[2] - 0.12)])
            d = _unit(p - a)
            P.prune_target("skirt_lift", a, p + 0.15 * d, t.get("stem_mm", 10), p, rgba=WOOD_RGBA, wp0c=wp)
    if not targets:
        gsp = sp.get("graft_union") or {}
        for _ in range(_ui(rng, gsp.get("rootstock_sprouts", [0, 0])) if gu else 0):
            az = float(rng.uniform(0, 2 * math.pi))
            rad = np.array([math.cos(az), math.sin(az), 0.0])
            z = float(rng.uniform(0.08, max(0.1, gu - 0.04)))
            a = trunk_r * rad + np.array([0, 0, z])
            p = a + 0.06 * _unit(rad + np.array([0, 0, 0.7]))
            P.prune_target("rootstock_sprout", a, p + 0.12 * _unit(p - a), _u(rng, gsp["sprout_mm"]), p,
                           extra={"below_graft_union": True})
        sk = sp.get("skirt") or {}
        for _ in range(_ui(rng, sk.get("n", [0, 0]))):
            az = float(rng.uniform(0, 2 * math.pi))
            z = _u(rng, sk["z_m"])
            rad = np.array([math.cos(az), math.sin(az), 0.0])
            a = trunk_r * rad + np.array([0, 0, max(0.25, z - 0.12)])
            p = 0.75 * cr * rad + np.array([0, 0, z])
            P.prune_target("skirt_lift", a, p + 0.15 * _unit(p - a), _u(rng, sk["diam_mm"]), p, rgba=WOOD_RGBA)
    used_az = [math.atan2(f["centre"][1], f["centre"][0]) for f in P.fruit]

    def free_az(base: float) -> float:
        for _ in range(40):
            az = base + float(rng.uniform(-1.0, 1.0))
            if all(abs((az - u + math.pi) % (2 * math.pi) - math.pi) > 0.35 for u in used_az):
                return az
        return base + math.pi
    alley_az = math.atan2(P.alley_dir[1], P.alley_dir[0])
    skm = sp.get("skirt_mark") or {}
    for _ in range(_ui(rng, skm.get("n", [0, 0]))):
        az = free_az(alley_az + 0.9)
        rad = np.array([math.cos(az), math.sin(az), 0.0])
        z = _u(rng, skm["z_m"])
        a = trunk_r * rad + np.array([0, 0, max(0.25, z - 0.1)])
        p = 0.7 * cr * rad + np.array([0, 0, z])
        P.prune_target("skirt_lift", a, p + 0.15 * _unit(p - a), _u(rng, skm["diam_mm"]), p, rgba=WOOD_RGBA)
    ws = sp.get("water_shoots") or {}
    for _ in range(_ui(rng, ws.get("n", [0, 0]))):
        if not scaff:
            break
        a_, b_, _r_ = scaff[int(rng.integers(0, len(scaff)))]
        z = _u(rng, ws["z_m"])
        base = a_ + 0.6 * (b_ - a_)
        top = np.array([base[0] * 1.1, base[1] * 1.1, max(z, float(base[2]) + 0.2)])
        P.prune_target("water_shoot", base, top + np.array([0, 0, 0.15]), _u(rng, ws["diam_mm"]), top)
    dt = sp.get("dead_twigs") or {}
    for _ in range(_ui(rng, dt.get("n", [0, 0]))):
        if not tips:
            break
        b = tips[int(rng.integers(0, len(tips)))]
        p = b + np.array([0.1, 0.0, -0.05])
        P.prune_target("dead_twig", b, p + np.array([0.1, 0, -0.05]), _u(rng, dt["diam_mm"]), p, rgba=DEAD_RGBA)
    # -- leaves (<= 12 quads + leaf_k sites)
    lv = sp.get("leaves") or {}
    leaf_src = e.get("leaves") or []
    leaf_cls = P.decal_class_pool("leaf")
    named = [c for c in e.get("leaf_decals", [])]
    if not leaf_src and lv:
        for k in range(min(int(lv["n"]), LEAF_QUADS_MAX)):
            az = math.radians((-90.0, -45.0, -135.0, 90.0)[k % 4] + float(rng.uniform(-10, 10))) + alley_az + math.pi / 2
            z = (0.55, 0.65, 0.75, 1.10, 0.60, 1.5)[k % 6]
            rr = cr * (0.85 + 0.1 * (k % 2))
            tilt = math.radians(35.0 + 10.0 * (k % 3))
            leaf_src.append({"p": [rr * math.cos(az), rr * math.sin(az), z],
                             "n": [math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)],
                             "kind": ("young", "mature")[k % 2]})
    n_leaf_decals = int((sp.get("leaf_decals") or {}).get("n", 0))
    in_band_idx = [i for i, lf in enumerate(leaf_src) if lf["p"][2] <= PICK_BAND_Z_M]
    for k, lf in enumerate(leaf_src[:LEAF_QUADS_MAX]):
        cls = None
        if k in in_band_idx[:n_leaf_decals]:
            j = in_band_idx.index(k)
            if j < len(named):
                cls = next((c for c in pf["symptoms"] if c["class"] == named[j]), {"class": named[j],
                           "unit": "per_leaf", "seasons": ["all"]})
            elif leaf_cls:
                cls = leaf_cls[j % len(leaf_cls)]
        P.leaf(k, lf["p"], lf["n"], lf.get("kind", "mature"), lv.get("size_m", [0.1, 0.04]), cls)
    # -- decals: registry overrides first, then seeded canopy / trunk classes
    _decals_common(P, H, cr, trunk_r, trunk_h, zc, hz)


def _decals_common(P: _Plan, H, cr, trunk_r, trunk_h, zc, hz, trunk_axis_r=None):
    rng, sp, e = P.rng, P.sp, P.e
    dc = sp.get("decals") or {}
    for d in e.get("decals", []):
        cls = next((c for c in P.pf["symptoms"] if c["class"] == d["class"]),
                   {"class": d["class"], "unit": d.get("unit", "trunk"), "seasons": ["all"]})
        p = np.asarray(d["p"], float)
        n = _unit([p[0], p[1], 0.0]) if np.linalg.norm(p[:2]) > 1e-6 else np.array([0.0, -1.0, 0.0])
        P.decal(cls, p, n, 0.03)
    n_extra = max(0, int(dc.get("n", 0)) - len(e.get("decals", [])))
    pools = [P.decal_class_pool("trunk"), P.decal_class_pool("canopy")]
    for i in range(n_extra):
        pool = pools[i % 2] or pools[(i + 1) % 2]
        if not pool:
            break
        cls = pool[int(rng.integers(0, len(pool)))]
        r = _u(rng, dc.get("r_m", [0.025, 0.035]))
        az = float(rng.uniform(0, 2 * math.pi))
        rad = np.array([math.cos(az), math.sin(az), 0.0])
        if cls.get("scale") == "trunk" or P.sp["archetype"] != "canopy" and pool is pools[0]:
            zr = dc.get("z_m", [0.2, min(0.8, trunk_h - 0.05)])
            z = _u(rng, [zr[0], max(zr[0], min(zr[1], trunk_h - 0.05))])
            p = (trunk_axis_r or trunk_r) * rad + np.array([0, 0, z])
        else:
            ph = float(rng.uniform(-0.3, 0.8))
            p = np.array([cr * math.cos(az) * math.cos(ph), cr * math.sin(az) * math.cos(ph), zc + hz * math.sin(ph)])
        P.decal(cls, p, rad, r)


PAPAYA_JAW_OPEN_M = 0.16          # T4 cradle jaw opening (OS 2.x / MISSION-PRESETS 5)
PAPAYA_MIN_GAP_M = 0.002          # static papaya surfaces never closer than this (no visual interpenetration)
PAPAYA_CROWD_GAP_M = 0.004        # the demo crowding fruit sits this close to its ripe neighbour
_ELLIPSE_N = 721


def ellipsoid_gap_m(c1, r1: float, h1: float, c2, r2: float, h2: float) -> float:
    """Surface gap of two upright prolate spheroids (semi-axes r, r, h with h >= r), never above the true gap.

    z is scaled by k = the smaller r / h of the pair. That affine map turns each spheroid into one with semi-axes
    (r, r, k h) <= r, i.e. inside the sphere of radius r, and it only shrinks distances; so the sphere gap in the
    scaled space is <= the true gap (exact, sign included, when both have the same shape, as papaya fruit do)."""
    k = min(r1 / h1, r2 / h2)
    d = np.asarray(c2, float) - np.asarray(c1, float)
    return math.sqrt(d[0] ** 2 + d[1] ** 2 + (k * d[2]) ** 2) - r1 - r2


def papaya_jaw_clearance_m(c, rx: float, hz: float, others, trunk_r: float, trunk_h: float) -> float:
    """``jaw_clearance`` truth (OS 4.4 papaya): the free space around the open T4 jaws.

    The cradle approaches radially from the alley side with the jaws open to 0.16 m (or the fruit width if
    that is larger), so the two finger lines run at +-half the opening in the tangent direction, from the
    fruit's centre plane out to its outer surface + 5 cm, over the middle half of its height. The result is the
    smallest distance from those lines to a neighbouring fruit (spheroid) or the trunk (cylinder); the pick
    passes when it is >= ``jaw_clearance_mm`` (15 mm per side). Distances to a spheroid are measured in its
    meridian plane against 721 points of the ellipse (error < 0.1 mm at papaya sizes)."""
    c = np.asarray(c, float)
    az = math.atan2(c[1], c[0])
    rad = np.array([math.cos(az), math.sin(az), 0.0])
    tan = np.array([-math.sin(az), math.cos(az), 0.0])
    half = max(PAPAYA_JAW_OPEN_M / 2.0, rx)
    q = np.array([c + side * half * tan + s * rad + np.array([0.0, 0.0, dz])
                  for side in (-1.0, 1.0) for s in np.linspace(0.0, rx + 0.05, 8)
                  for dz in (-0.5 * hz, 0.0, 0.5 * hz)])
    best = 1.0
    below = q[:, 2] <= trunk_h
    if below.any():
        best = min(best, float(np.min(np.hypot(q[below, 0], q[below, 1]))) - trunk_r)
    th = np.linspace(-math.pi / 2.0, math.pi / 2.0, _ELLIPSE_N)
    for o, ro, ho in others:
        d = q - np.asarray(o, float)
        rho, z = np.hypot(d[:, 0], d[:, 1]), d[:, 2]
        er, ez = ro * np.cos(th), ho * np.sin(th)
        dist = np.sqrt(np.min((rho[:, None] - er[None, :]) ** 2 + (z[:, None] - ez[None, :]) ** 2, axis=1))
        inside = (rho / ro) ** 2 + (z / ho) ** 2 < 1.0
        best = min(best, float(np.min(np.where(inside, -dist, dist))))
    return best


def _rot_z(p, ang: float) -> np.ndarray:
    ca, sa = math.cos(ang), math.sin(ang)
    return np.array([ca * p[0] - sa * p[1], sa * p[0] + ca * p[1], p[2]])


def _papaya_place(pts, lengths, fr, trunk_r: float, crowded_k: int, n_ripe: int):
    """(centres, lengths, wp0c) with no two static papayas closer than ``PAPAYA_MIN_GAP_M``.

    Frozen WP0c targets (``wp`` set) keep their points; a frozen fruit's length is capped (not below 80 % of
    the configured minimum) to clear the frozen fruit placed before it. Every other point (WP0c neighbours,
    the helix) keeps its height and radius and turns about the trunk by the smallest angle that clears all
    placed fruit; a point that clears nowhere is dropped. ``crowded_k`` > 0 (registry ``crowded_ripe_k``,
    OS 1.2 step 8 on P-1): one extra fruit is placed just clear of the k-th lowest in-band ripe fruit, inside
    its jaw path, so that fruit fails ``jaw_clearance`` and is MARKed ``no_clearance``."""
    sx, sz = float(fr["shape"][0]), float(fr["shape"][2])
    lo_len = 0.8 * float(fr["length_m"][0])
    placed = []          # (centre, length, wp)

    def gap_to_all(c, L, skip=None):
        return min((ellipsoid_gap_m(c, sx * L, sz * L, q, sx * Lq, sz * Lq)
                    for j, (q, Lq, _) in enumerate(placed) if j != skip), default=1.0)
    order = sorted(range(len(pts)), key=lambda i: (pts[i][1] is None, pts[i][0][2]))
    for i in order:
        c, wp = np.asarray(pts[i][0], float), pts[i][1]
        L = float(lengths[i])
        if wp is not None:
            while L > lo_len and gap_to_all(c, L) < PAPAYA_MIN_GAP_M:
                L = max(lo_len, L - 0.002)
            placed.append((c, L, wp))
            continue
        for step in range(0, 181):
            hit = None
            for sgn in ((1.0,) if step == 0 else (1.0, -1.0)):
                q = _rot_z(c, sgn * math.radians(step))
                if gap_to_all(q, L) >= PAPAYA_MIN_GAP_M:
                    hit = q
                    break
            if hit is not None:
                placed.append((hit, L, None))
                break
    placed.sort(key=lambda t: t[0][2])
    if crowded_k > 0:
        ripe = [j for j, t in enumerate(placed) if t[0][2] <= PICK_BAND_Z_M][:n_ripe]
        if len(ripe) >= crowded_k:
            tc, tL, _ = placed[ripe[crowded_k - 1]]
            L = float(fr["length_m"][0])
            # a little higher than the target (so it is not one of the lowest ripe), pushed out from the trunk
            # by up to 30 % (crowded fruit bulge outward) and turned to the azimuth (1 deg steps) that leaves the
            # target the least jaw clearance without touching any fruit
            others = [(q, sx * Lq, sz * Lq) for q, Lq, _ in placed if q is not tc]
            best = None
            for scale, deg in itertools.product((1.0, 1.1, 1.2, 1.3), range(-120, 121)):
                q = _rot_z(np.array([tc[0] * scale, tc[1] * scale, tc[2] + 0.02]), math.radians(deg))
                if ellipsoid_gap_m(tc, sx * tL, sz * tL, q, sx * L, sz * L) < PAPAYA_CROWD_GAP_M \
                        or gap_to_all(q, L) < PAPAYA_MIN_GAP_M or q[2] > PICK_BAND_Z_M:
                    continue
                jaw = papaya_jaw_clearance_m(tc, sx * tL, sz * tL, others + [(q, sx * L, sz * L)], trunk_r,
                                             float("inf"))
                if best is None or jaw < best[0] - 1e-9:
                    best = (jaw, q)
            if best is not None:
                placed.append((best[1], L, None))
            placed.sort(key=lambda t: t[0][2])
    return [t[0] for t in placed], [t[1] for t in placed], [t[2] for t in placed]


def _gen_papaya(P: _Plan):
    e, sp, pf, rng = P.e, P.sp, P.pf, P.rng
    H = float(e.get("height_m") or _u(rng, sp["height_m"]))
    cr = float(e.get("canopy_r_m") or 0.9)
    trunk_r = float(e.get("trunk_r_m") or _u(rng, sp["trunk_r_m"]))
    trunk_h = float(e.get("trunk_h_m") or (H - float(sp.get("crown_drop_m", 0.3))))
    skel = e.get("wood") or []
    if skel:
        trunk_r, trunk_h = float(skel[0]["r"]), float(skel[0]["b"][2])
    P.keep_wood("trunk", [0, 0, 0], [0, 0, trunk_h], trunk_r, rgba=[0.55, 0.55, 0.45, 1.0])
    top = np.array([0, 0, trunk_h])
    pet = sp["petioles"]
    for k in range(int(pet["n"])):
        a = 2 * math.pi * k / pet["n"] * 2.4 + float(rng.uniform(-0.1, 0.1))
        d = _unit([math.cos(a), math.sin(a), float(rng.uniform(0.25, 0.85))])
        L = _u(rng, pet["len_m"])
        p1 = top + L * d
        P.stick("crown", top, p1, 0.008)
        dm = _u(rng, pet["disc_m"])
        P.blob("crown", p1 + 0.1 * d, [dm, dm, 0.012],
               _quat_mat(frame_quat([0, 0, 1], [math.cos(a), math.sin(a), 0])))
    P.flush_mesh("crown", "foliage", GRP_FOLIAGE, sp["foliage"]["rgba"])
    z0, z1 = H - 0.6, H + 0.1
    P.envelope_ellipsoid([0, 0, (z0 + z1) / 2.0], [cr, cr, (z1 - z0) / 2.0])
    P.extras["envelope"] = {"type": "ellipsoid", "c": _r([0, 0, (z0 + z1) / 2.0]), "size": _r([cr, cr, (z1 - z0) / 2])}
    # fruit column: WP0c cradle points (+ their neighbours), then a helix up to below the crown
    fr = sp["fruit"]
    pts = []
    for t in e.get("targets") or []:
        if t.get("task") != "papaya_cradle":
            continue
        cand = [(np.asarray(t["p"], float), {"id": t["id"], "status": t.get("status"), "reason": t.get("reason")})]
        cand += [(np.asarray(q, float), None) for q in t.get("neighbours", [])]
        for p, wp in cand:
            dup = next((i for i, (q, _) in enumerate(pts) if np.linalg.norm(q - p) < 0.01), None)
            if dup is None:
                pts.append((p, wp))
            elif wp and pts[dup][1] is None:
                pts[dup] = (pts[dup][0], wp)
    length0 = float(np.mean(fr["length_m"]))
    r_col = trunk_r + fr["shape"][0] * length0 + 0.005
    n_total = _ui(rng, fr["n"])
    z = max([p[2] for p, _ in pts] + [fr["column_z_m"][0]]) + 0.12
    k = 0
    while len(pts) < n_total and z < min(fr["column_z_m"][1], trunk_h - 0.12):
        a = 2.39996 * k + float(rng.uniform(-0.1, 0.1))
        pts.append((np.array([r_col * math.cos(a), r_col * math.sin(a), z]), None))
        z += 0.11
        k += 1
    pts.sort(key=lambda t: t[0][2])
    n_ripe = int(e.get("ripe_in_band_n", fr.get("demo_ripe_in_band", _ui(rng, fr["ripe_lowest"]))))
    # sizes first, then placement without interpenetration (frozen WP0c points never move: their lengths are
    # capped; every other point turns about the trunk until it clears), then the demo crowding of OS 1.2 step 8
    lengths = [_u(rng, fr["length_m"]) for _ in pts]
    centres, lengths, wps = _papaya_place(pts, lengths, fr, trunk_r, int(e.get("crowded_ripe_k") or 0), n_ripe)
    lowest = [i for i, c in enumerate(centres) if c[2] <= PICK_BAND_Z_M]
    ripe_ids = set(lowest[:n_ripe]) if lowest else set(range(min(n_ripe, len(centres))))
    hs, imm = crops.harvest_stages(pf)[0], next(s["id"] for s in pf["stages"] if not s.get("harvest"))
    sx, sz = float(fr["shape"][0]), float(fr["shape"][2])
    for i, (c, length, wp) in enumerate(zip(centres, lengths, wps)):
        st = hs if i in ripe_ids else imm
        hzf = sz * length
        az = math.atan2(c[1], c[0])
        rad = np.array([math.cos(az), math.sin(az), 0.0])
        stem_end = c + np.array([0, 0, hzf])
        anchor = trunk_r * rad + np.array([0, 0, stem_end[2] + 0.01])
        others = [(q, sx * L, sz * L) for j, (q, L) in enumerate(zip(centres, lengths)) if j != i]
        jaw = papaya_jaw_clearance_m(c, sx * length, hzf, others, trunk_r, trunk_h)
        gap = min((ellipsoid_gap_m(c, sx * length, hzf, q, rq, hq) for q, rq, hq in others), default=1.0)
        extra = {"clearance_mm": _r(1000.0 * jaw, 1), "fruit_gap_mm": _r(1000.0 * gap, 1)}
        P.add_fruit("fruit", c, length, fr["shape"], _u(rng, fr["mass_kg"]), st, [0, 0, 1], stem_end, anchor, 0.0,
                    P.days_for(st), wp0c=wp, target_p=c, extra=extra)
    P.extras["leaf_scan"] = "n/a: crown-only"
    _decals_common(P, H, cr, trunk_r, trunk_h, (z0 + z1) / 2.0, (z1 - z0) / 2.0)


def _gen_banana(P: _Plan):
    e, sp, pf, rng = P.e, P.sp, P.pf, P.rng
    H = float(e.get("height_m") or _u(rng, sp["height_m"]))
    cr = float(e.get("canopy_r_m") or 1.4)
    R = float(e.get("trunk_r_m") or sp["pseudostem_r_m"])
    g = P.geom("wood", "cylinder", GRP_WOOD, [0.45, 0.60, 0.30, 1.0], 1, pos=[0, 0, H / 2.0], size=[R, H / 2.0, 0])
    P.wood.append({"geom": g["name"], "role": "pseudostem", "keep": True, "diameter_mm": _r(2000.0 * R, 1),
                   "girth_cm": _r(200.0 * math.pi * R, 1)})
    top = np.array([0, 0, H])
    n_leaves = _ui(rng, sp["leaves_n"])
    n_dead = int(sp.get("dead_blades", 2))
    n_sig = _ui(rng, sp["sigatoka_leaves"])
    leaves_truth = []
    tidy = [t for t in (e.get("targets") or []) if t.get("task") == "banana_tidy"]
    droop = e.get("drooping_leaf")
    alley_az = math.atan2(P.alley_dir[1], P.alley_dir[0])
    bl, bw = sp["blade_m"]
    for k in range(n_leaves):
        lid = f"{P.tid}:blade:{k + 1}"
        if k < n_dead:
            # dead blade hanging into the pick band (banana_tidy candidate, cut at the band)
            if k < len(tidy):
                cp = np.asarray(tidy[k]["p"], float)
                wp = {"id": tidy[k]["id"], "status": tidy[k].get("status"), "reason": tidy[k].get("reason")}
                diam = float(tidy[k].get("stem_mm", 8))
            else:
                az = alley_az + float(rng.uniform(-0.5, 0.5))
                cp = np.array([0.6 * math.cos(az), 0.6 * math.sin(az), 0.75])
                wp, diam = None, 8.0
            az = math.atan2(cp[1], cp[0])
            a = np.array([(R + 0.02) * math.cos(az), (R + 0.02) * math.sin(az), H - 0.45])
            b = cp + 0.35 * _unit(cp - a)
            t = P.prune_target("dead_blade", a, b, diam, cp, rgba=sp["foliage"].get("dead_rgba", DEAD_RGBA),
                               group=GRP_FOLIAGE, wp0c=wp, gtype="box",
                               extra={"leaf": lid, "tidy_only": True, "green": False})
            # banana_tidy is a band tidy-up cut, not the petiole cut: classify it as a cut when in band
            t["job"] = "banana_tidy"
            t["action"], t["reason"] = ("cut", "") if t["in_band"] else ("mark", "out_of_reach_marked")
            leaves_truth.append({"id": lid, "functional": False, "dead": True, "sigatoka_pct": 0.0,
                                 "prune": t["id"]})
            continue
        if droop and k == n_dead:
            az = alley_az
            d = _unit([math.cos(az) * droop.get("reach_m", 1.15), math.sin(az) * droop.get("reach_m", 1.15),
                       droop.get("tip_z_m", 1.6) - (H - 0.1)])
            tip = np.array([0, 0, H - 0.1]) + d * math.hypot(droop.get("reach_m", 1.15), H - 0.1 - droop["tip_z_m"])
            tip[2] = droop["tip_z_m"]
            P.stick("blades", [0, 0, H - 0.1], tip, bw / 2.0)
            P.envelope_capsule([0, 0, H - 0.1], tip.tolist(), bw / 2.0)
            P.extras["drooping_leaf"] = {"leaf": lid, "tip": _r(tip.tolist()), "tip_world": P.world(tip),
                                         "tip_z_m": _r(float(tip[2]), 3)}
        else:
            a = 2 * math.pi * k / n_leaves * 1.6 + float(rng.uniform(-0.2, 0.2))
            d = _unit([math.cos(a), math.sin(a), float(rng.uniform(0.15, 0.9))])
            P.stick("blades", top - np.array([0, 0, 0.1]), top + bl * d, bw / 2.0)
        sig = 0.0
        if k - n_dead < n_sig:
            sig = round(float(rng.uniform(15.0, 60.0)), 1)
        leaves_truth.append({"id": lid, "functional": sig < 50.0, "dead": False, "sigatoka_pct": sig})
    P.flush_mesh("blades", "foliage", GRP_FOLIAGE, sp["foliage"]["rgba"])
    ach = e.get("alley_clear_height_m") or 1.3
    clear_h = float(min(ach)) if isinstance(ach, (list, tuple)) else float(ach)   # WP4: per side [L, R]
    z0, z1 = clear_h, H + 0.4
    P.envelope_ellipsoid([0, 0, (z0 + z1) / 2.0], [cr, cr, (z1 - z0) / 2.0])
    P.extras["envelope"] = {"type": "ellipsoid", "c": _r([0, 0, (z0 + z1) / 2.0]), "size": _r([cr, cr, (z1 - z0) / 2])}
    # Sigatoka streaks: one decal per affected blade (on the blade's outer half; truth carries the leaf id)
    sig_cls = next((c for c in pf["symptoms"] if c["class"] == "sigatoka"), None)
    for lf in leaves_truth:
        if lf["sigatoka_pct"] > 0 and sig_cls:
            a = float(rng.uniform(0, 2 * math.pi))
            p = top + np.array([0.6 * math.cos(a), 0.6 * math.sin(a), 0.25])
            d = P.decal(sig_cls, p, [0, 0, 1], 0.03, unit="per_leaf")
            d["leaf"] = lf["id"]
    # bunch (static, collidable like wood), peduncle, ribbon
    bt = next((t for t in (e.get("targets") or []) if t.get("task") == "banana_bunch"), None)
    bz = _u(rng, sp["bunch"]["z_m"])
    bp = np.asarray(bt["p"], float) if bt else np.array([0.3 * math.cos(alley_az), 0.3 * math.sin(alley_az), bz])
    blen, br = float(sp["bunch"]["len_m"]), float(sp["bunch"]["r_m"])
    P.capsule("wood", [R * math.cos(math.atan2(bp[1], bp[0])), R * math.sin(math.atan2(bp[1], bp[0])), H - 0.15],
              (bp + np.array([0, 0, blen / 2.0])).tolist(), 0.025, GRP_WOOD, [0.40, 0.50, 0.20, 1.0], contype=1)
    P.wood.append({"geom": P.geoms[-1]["name"], "role": "peduncle", "keep": True, "diameter_mm": 50.0})
    stage_rgba = pf["stages"][0]["rgba"]
    gb = P.geom("bunch", "ellipsoid", GRP_FRUIT, stage_rgba, 1, pos=bp.tolist(), size=[br, br, blen / 2.0])
    colour = e.get("ribbon_colour") or RIBBONS[0]
    gr = P.geom("ribbon", "box", GRP_FRUIT, RIBBON_RGBA.get(colour, [1, 0, 1, 1]), 0,
                pos=(bp + np.array([0, 0, blen / 2.0 + 0.03])).tolist(), size=[0.03, 0.03, 0.012])
    shoot = e.get("shooting_date")
    sim_date = e.get("_sim_date") or DEFAULT_SIM_DATE
    age = None if not shoot else (_dt.date.fromisoformat(sim_date) - _dt.date.fromisoformat(shoot)).days
    stage_id = crops.stage_for_age(pf, age) if age is not None else pf["stages"][0]["id"]
    P.extras["bunch"] = {"geom": gb["name"], "ribbon_geom": gr["name"], "ribbon_colour": colour,
                         "shooting_date": shoot, "age_days": age, "stage": stage_id,
                         "harvest_window": crops.harvest_window(pf, e), "h_m": _r(float(bp[2]), 3),
                         "p": _r(bp.tolist()), "world": P.world(bp),
                         "hands_n": _ui(rng, sp["bunch"]["hands"]), "mass_kg": _r(_u(rng, sp["bunch"]["mass_kg"]), 2),
                         "tool_hint": "human_cut", "exclusion_m": pf.get("bunch_exclusion_m", 3.0),
                         "wp0c": {"id": bt["id"], "status": bt.get("status"), "reason": bt.get("reason")} if bt else None}
    # suckers: peeper (< 0.5 m) / sword; the tallest sword is the follower (keep), the rest desucker MARK
    su = sp["suckers"]
    sk = []
    for k in range(int(su["n"])):
        az = alley_az + math.pi / 2 + k * math.pi + float(rng.uniform(-0.4, 0.4))
        dist = _u(rng, su["dist_m"])
        h = _u(rng, su["h_m"])
        rr = _u(rng, su["r_m"])
        b0 = np.array([dist * math.cos(az), dist * math.sin(az), 0.0])
        sk.append((b0, h, rr))
    follower = max(range(len(sk)), key=lambda i: sk[i][1]) if sk else None
    P.extras["suckers"] = []
    for i, (b0, h, rr) in enumerate(sk):
        t = P.prune_target("sucker", b0, b0 + np.array([0, 0, h]), 2000.0 * rr, b0 + np.array([0, 0, 0.1]),
                           rgba=[0.42, 0.62, 0.30, 1.0], extra={"sucker_type": "peeper" if h < 0.5 else "sword",
                                                               "follower": i == follower, "h_m": _r(h, 3)})
        if i == follower:
            t["keep"], t["action"], t["reason"] = True, "skip", "follower"
        P.extras["suckers"].append(t["id"])
    P.extras["leaves_total"] = n_leaves
    P.extras["functional_leaves"] = sum(1 for lf in leaves_truth if lf["functional"])
    P.extras["blades"] = leaves_truth
    P.extras["sanitation_deleaf"] = crops.deleaf_plan(pf, leaves_truth, "flowering")
    # leaf targets: only the in-band dead blades (WP0c leaves), else n/a crown-only
    for k, lf in enumerate((e.get("leaves") or [])[:LEAF_QUADS_MAX]):
        P.leaf(k, lf["p"], lf["n"], lf.get("kind", "dead_blade"), [0.10, 0.05], None, functional=False)
    if not P.leaves:
        P.extras["leaf_scan"] = "n/a: crown-only"
    _decals_common(P, H, cr, R, H, 1.0, 0.5)


GENERATORS = {"canopy": _gen_canopy, "papaya": _gen_papaya, "banana": _gen_banana}


def generate(entry: dict, sp: dict, profile: dict, season: str = DEFAULT_SEASON, seed: int = 7,
             sim_date: str = DEFAULT_SIM_DATE) -> dict:
    """One registry entry -> tree plan (geoms, meshes, sites, truth, render-geom count, hash)."""
    if season not in SEASONS:
        raise TreeConfigError(f"season {season!r} not in {SEASONS}")
    e = normalize_entry(entry)
    e["_sim_date"] = sim_date
    if sp["species"] != e["crop"] or profile["crop"] != e["crop"]:
        raise TreeConfigError(f"{e['tree_id']}: species/profile mismatch for crop {e['crop']!r}")
    P = _Plan(e, sp, profile, season, seed)
    GENERATORS[sp["archetype"]](P)
    P.extras["harvest_window"] = crops.harvest_window(profile, e, sim_date) if profile["crop"] != "banana" else \
        P.extras.get("bunch", {}).get("harvest_window")
    P.e = {k: v for k, v in e.items() if k != "_sim_date"}
    return P.done()


def plan_hash(plan: dict) -> str:
    """sha1 of the plan's canonical JSON without its own ``hash`` field (stable for a seed and registry)."""
    body = {k: v for k, v in plan.items() if k != "hash"}
    return hashlib.sha1(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def world_plans(field: dict, seed: int, season: Optional[str] = None, trees_directory: Optional[str] = None,
                crops_directory: Optional[str] = None) -> List[dict]:
    """Every tree plan of a field (``[]`` for a world without trees). Deterministic in (field, seed, configs)."""
    species = load_all_species(trees_directory)
    entries = registry(field, sorted(species))
    if not entries:
        return []
    profiles = crops.load_profiles(crops_directory)
    s = season or season_name(field)
    sim_date = field.get("sim_date") or DEFAULT_SIM_DATE
    out = []
    for e in entries:
        crop = e.get("crop")
        if crop not in species or crop not in profiles:
            raise TreeConfigError(f"tree {e.get('tree_id')}: crop {crop!r} needs config/trees/{crop}.yaml and "
                                  f"config/crops/{crop}.yaml")
        out.append(generate(e, species[crop], profiles[crop], s, seed, sim_date))
    return out


def fruit_table(plans: Sequence[dict]) -> Dict[str, dict]:
    """fruit_id -> fruit dict over every plan (the pool assigns from this table)."""
    return {f["fruit_id"]: f for p in plans for f in p["truth"]["fruit"]}


def truth_records(plans: Sequence[dict]) -> List[dict]:
    return [p["truth"] for p in plans]
