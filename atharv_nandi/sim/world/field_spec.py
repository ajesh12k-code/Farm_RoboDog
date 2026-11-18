"""Field registry: ``config/fields/<world>.json`` (WP4; ORCHARD-MVP-SPEC 3.3, 3.6, 4; MISSION-PRESETS 0.1).

One JSON file per world is the single source for the orchard layout: the **tree registry** (MP 0.1, the
targets every preset resolves to), the terrain parameters (``terrain_orchard``), the field fixtures
(laterals, mulch, ropes, props, dock, crate), the actors (``actors``), the lighting default and the
scripted world events (``controllers.world_events``). Pure Python (json + math), no MuJoCo: the web
console, the mission layer and the tests read it without building a model.

Schema ``nandi.field/1`` (top level)::

    schema, world, kind ("legacy" | "orchard"), label, classification, season, season_note,
    frame {name, origin_note}, spacing {basis, ...}, soil_type,
    home {xy, yaw_deg}, dock {xy, yaw_deg, size_m}, crate {xy, size_m, wall_m},
    geofence [[x, y], ...]   (polygon, CCW),
    rows [{row_id, block_id, crop, y, in_row_m, alley_flat}],
    trees [<tree entry>],          # MP 0.1, below
    terrain {...},                 # terrain_orchard.TerrainConfig.from_field
    fixtures {laterals_auto, laterals [...], mulch [...], bunds [...]},
    actors [...],                  # world.actors
    lighting {default},            # world.lighting presets
    events [...]                   # controllers.world_events

Tree entry (MISSION-PRESETS 0.1 field names; extra keys allowed)::

    identity  tree_id, physical_tag, block_id, row_id, crop, cultivar, form (dwarf|grafted|hedgerow), geo {x, y}
    geometry  height_m, canopy_r_m, trunk_r_m, trunk_h_m, graft_union_h_m (null = own-rooted), basin_r_m,
              orbit_clear, alley_sides [L, R], alley_clear_height_m [L, R], skirt_h_m, skirted_up
    obstacles drip_line_offset_m, rope_lines [{id, a [x,y,z], b [x,y,z], r_m}], props [{id, kind, a, b, r_m}]
    agronomy  fruit_set_date, bagging_date, ribbon_colour, last_visit, flags []
    sim       seed (generator seed for WP23)

The legacy world's file has ``kind: legacy`` and no trees: every WP4 module is a no-op there, so the legacy
model compiles byte-identical (``tests/test_terrain_orchard.py``).
"""
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from atharv_nandi import paths

FIELD_SCHEMA = "nandi.field/1"
REPO = paths.REPO
FIELDS_DIR = os.path.join(REPO, "config", "fields")
ORCHARD_WORLDS = ("orchard_lite", "orchard_demo", "orchard_demo_monsoon")
CROPS = ("mango", "guava", "papaya", "banana")
FORMS = ("dwarf", "grafted", "hedgerow", "own_rooted", "sucker")

# NHB high-density mango spacing frozen for the project (OQ-02 B, 2026-11-17).
NHB_MANGO_SPACINGS_M = ((5.0, 2.5), (5.0, 3.0))   # (row pitch, in-row)
# ORCHARD-MVP-SPEC 3.6: trunk basins r 0.6-0.9 m.
BASIN_R_RANGE_M = (0.6, 0.9)

TREE_REQUIRED = ("tree_id", "physical_tag", "block_id", "row_id", "crop", "cultivar", "form", "geo",
                 "height_m", "canopy_r_m", "trunk_r_m", "graft_union_h_m", "orbit_clear", "alley_sides",
                 "alley_clear_height_m", "drip_line_offset_m", "rope_lines", "props", "fruit_set_date",
                 "bagging_date", "ribbon_colour", "last_visit", "flags")


class FieldError(ValueError):
    """A field file that breaks the ``nandi.field/1`` schema."""


# ---------------------------------------------------------------------- loading
def field_path(world: str, fields_dir: str = FIELDS_DIR) -> str:
    return os.path.join(fields_dir, f"{world}.json")


def load(world: str, fields_dir: str = FIELDS_DIR, validate_field: bool = True) -> dict:
    """The parsed field of ``world`` (``FieldError`` if missing or invalid)."""
    path = field_path(world, fields_dir)
    if not os.path.isfile(path):
        raise FieldError(f"no field file for world {world!r} ({path})")
    with open(path, encoding="utf-8") as fh:
        field = json.load(fh)
    if validate_field:
        errs = validate(field)
        if errs:
            raise FieldError(f"{path}: " + "; ".join(errs[:8]))
    return field


def is_orchard(field: Optional[dict]) -> bool:
    return bool(field) and field.get("kind") == "orchard"


# ---------------------------------------------------------------------- accessors
def trees(field: dict) -> List[dict]:
    return list(field.get("trees") or [])


def tree_xy(tree: dict) -> Tuple[float, float]:
    g = tree["geo"]
    return float(g["x"]), float(g["y"])


def tree_by_id(field: dict, tree_id: str) -> dict:
    for t in trees(field):
        if t["tree_id"] == tree_id or t.get("physical_tag") == tree_id:
            return t
    raise KeyError(f"tree {tree_id!r} not in field {field.get('world')!r}")


def trees_in_row(field: dict, row_id: str) -> List[dict]:
    return sorted((t for t in trees(field) if t.get("row_id") == row_id), key=lambda t: tree_xy(t)[0])


def rows(field: dict) -> List[dict]:
    return list(field.get("rows") or [])


def rope_lines(field: dict) -> List[dict]:
    """Every rope line (per-tree ``rope_lines[]``), de-duplicated by id: ``{id, a, b, r_m, tree_id}``."""
    out, seen = [], set()
    for t in trees(field):
        for r in t.get("rope_lines") or []:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            out.append({"id": r["id"], "a": [float(v) for v in r["a"]], "b": [float(v) for v in r["b"]],
                        "r_m": float(r.get("r_m", 0.006)), "tree_id": t["tree_id"]})
    return out


def props(field: dict) -> List[dict]:
    out = []
    for t in trees(field):
        for p in t.get("props") or []:
            out.append({"id": p["id"], "kind": p.get("kind", "bamboo"), "a": [float(v) for v in p["a"]],
                        "b": [float(v) for v in p["b"]], "r_m": float(p.get("r_m", 0.025)),
                        "tree_id": t["tree_id"]})
    return out


def laterals(field: dict) -> List[dict]:
    """Drip laterals as polylines ``{id, kind, points [[x, y], ...], r_m, tree_id|row_id}``.

    ``fixtures.laterals_auto`` (default true) generates one straight lateral per tree line at the row's
    ``drip_line_offset_m`` (signed, + = toward +y), from the first tree − ``lateral_overhang_m`` to the last
    tree + overhang. ``fixtures.laterals`` adds explicit ones: ``{id, ring: {tree_id, r_m, n}}`` (a ring around
    a trunk) or ``{id, points}`` (a spur)."""
    fx = field.get("fixtures") or {}
    r = float(fx.get("lateral_r_m", 0.008))
    over = float(fx.get("lateral_overhang_m", 1.0))
    out = []
    if fx.get("laterals_auto", True):
        for row in rows(field):
            ts = [t for t in trees_in_row(field, row["row_id"]) if not t.get("orbit_clear")]
            if not ts:
                continue
            off = float(ts[0].get("drip_line_offset_m", 0.0))
            xs = [tree_xy(t)[0] for t in ts]
            y = tree_xy(ts[0])[1] + off
            out.append({"id": f"lateral_{row['row_id']}", "kind": "row", "row_id": row["row_id"], "r_m": r,
                        "points": [[min(xs) - over, y], [max(xs) + over, y]]})
    for lat in fx.get("laterals") or []:
        if "ring" in lat:
            ring = lat["ring"]
            cx, cy = tree_xy(tree_by_id(field, ring["tree_id"]))
            n = int(ring.get("n", 16))
            rr = float(ring["r_m"])
            pts = [[cx + rr * math.cos(2 * math.pi * k / n), cy + rr * math.sin(2 * math.pi * k / n)]
                   for k in range(n + 1)]
            out.append({"id": lat["id"], "kind": "ring", "tree_id": ring["tree_id"], "r_m": r, "points": pts})
        else:
            out.append({"id": lat["id"], "kind": lat.get("kind", "spur"), "r_m": float(lat.get("r_m", r)),
                        "points": [[float(p[0]), float(p[1])] for p in lat["points"]]})
    return out


def lateral_segments(field: dict) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
    """(lateral_id, a, b) for every straight piece of every lateral."""
    segs = []
    for lat in laterals(field):
        pts = lat["points"]
        for k in range(len(pts) - 1):
            segs.append((lat["id"], (pts[k][0], pts[k][1]), (pts[k + 1][0], pts[k + 1][1])))
    return segs


def map_mask_items(field: dict) -> Dict[str, list]:
    """Static obstacles the safety layer masks from the map (ORCHARD-MVP-SPEC 5.2 ``map_masked_lidar``):
    tree trunks, basins, laterals, props and ropes, from the registry (truth)."""
    return {
        "trunks": [{"tree_id": t["tree_id"], "xy": list(tree_xy(t)), "r_m": float(t["trunk_r_m"])} for t in trees(field)],
        "basins": [{"tree_id": t["tree_id"], "xy": list(tree_xy(t)), "r_m": float(t.get("basin_r_m", 0.0))}
                   for t in trees(field) if t.get("basin_r_m")],
        "laterals": laterals(field),
        "props": props(field),
        "ropes": rope_lines(field),
    }


def alley_free_width(field: dict, row_a: str, row_b: str) -> float:
    """Free alley width (m) between two adjacent rows after canopy: pitch − max canopy_r of each row."""
    ta, tb = trees_in_row(field, row_a), trees_in_row(field, row_b)
    if not ta or not tb:
        raise KeyError(f"rows {row_a!r}/{row_b!r} not both in the field")
    pitch = abs(tree_xy(tb[0])[1] - tree_xy(ta[0])[1])
    return pitch - max(float(t["canopy_r_m"]) for t in ta) - max(float(t["canopy_r_m"]) for t in tb)


def point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def in_geofence(field: dict, x: float, y: float) -> bool:
    fence = field.get("geofence")
    return True if not fence else point_in_polygon(x, y, fence)


# ---------------------------------------------------------------------- validation
def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _vec(v, n) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == n and all(_num(x) for x in v)


def validate_tree(t: dict) -> List[str]:
    tid = t.get("tree_id", "?")
    errs = [f"tree {tid}: missing {k}" for k in TREE_REQUIRED if k not in t]
    if errs:
        return errs
    if t["crop"] not in CROPS:
        errs.append(f"tree {tid}: crop {t['crop']!r} not in {CROPS}")
    if t["form"] not in FORMS:
        errs.append(f"tree {tid}: form {t['form']!r} not in {FORMS}")
    g = t["geo"]
    if not (isinstance(g, dict) and _num(g.get("x")) and _num(g.get("y"))):
        errs.append(f"tree {tid}: geo needs numeric x, y")
    for k in ("height_m", "canopy_r_m", "trunk_r_m"):
        if not (_num(t[k]) and t[k] > 0):
            errs.append(f"tree {tid}: {k} must be > 0")
    if _num(t.get("height_m")) and not 1.2 <= t["height_m"] <= 3.0:
        errs.append(f"tree {tid}: height_m {t['height_m']} outside the 1.2-3.0 m target (4-10 ft)")
    gu = t["graft_union_h_m"]
    if gu is not None and not (_num(gu) and 0.0 < gu < 1.0):
        errs.append(f"tree {tid}: graft_union_h_m must be null or in (0, 1) m")
    if t["form"] == "grafted" and gu is None:
        errs.append(f"tree {tid}: grafted tree needs graft_union_h_m")
    for k in ("alley_sides", "alley_clear_height_m"):
        if not (isinstance(t[k], list) and len(t[k]) in (1, 2)):
            errs.append(f"tree {tid}: {k} must be a list [L, R]")
    if "basin_r_m" in t and t["basin_r_m"] is not None and not (
            _num(t["basin_r_m"]) and BASIN_R_RANGE_M[0] - 1e-9 <= t["basin_r_m"] <= BASIN_R_RANGE_M[1] + 1e-9):
        errs.append(f"tree {tid}: basin_r_m {t['basin_r_m']} outside {BASIN_R_RANGE_M} (OS 3.6)")
    for r in t["rope_lines"]:
        if not (isinstance(r, dict) and r.get("id") and _vec(r.get("a"), 3) and _vec(r.get("b"), 3)):
            errs.append(f"tree {tid}: rope line needs id, a [x,y,z], b [x,y,z]")
    for p in t["props"]:
        if not (isinstance(p, dict) and p.get("id") and _vec(p.get("a"), 3) and _vec(p.get("b"), 3)):
            errs.append(f"tree {tid}: prop needs id, a [x,y,z], b [x,y,z]")
    if t["crop"] == "banana" and not t.get("ribbon_colour"):
        errs.append(f"tree {tid}: banana needs ribbon_colour (bunch age, OS 4.4)")
    if not isinstance(t["flags"], list):
        errs.append(f"tree {tid}: flags must be a list")
    return errs


def validate(field: dict) -> List[str]:
    """Schema errors of a field dict (empty = valid)."""
    if not isinstance(field, dict):
        return ["field is not an object"]
    errs = []
    if field.get("schema") != FIELD_SCHEMA:
        errs.append(f"schema {field.get('schema')!r} != {FIELD_SCHEMA}")
    kind = field.get("kind")
    if kind not in ("legacy", "orchard"):
        errs.append(f"kind {kind!r} not legacy|orchard")
    if kind == "legacy":
        if field.get("trees"):
            errs.append("legacy field must not carry trees (the legacy world stays byte-identical)")
        return errs
    ids, tags = set(), set()
    for t in trees(field):
        errs += validate_tree(t)
        tid, tag = t.get("tree_id"), t.get("physical_tag")
        if tid in ids:
            errs.append(f"duplicate tree_id {tid}")
        if tag in tags:
            errs.append(f"duplicate physical_tag {tag}")
        ids.add(tid)
        tags.add(tag)
    if not trees(field):
        errs.append("orchard field has no trees")
    row_ids = {r.get("row_id") for r in rows(field)}
    for t in trees(field):
        if t.get("row_id") not in row_ids and not t.get("orbit_clear"):
            errs.append(f"tree {t.get('tree_id')}: row {t.get('row_id')!r} not in rows[]")
    for k in ("home", "dock", "crate"):
        if not (isinstance(field.get(k), dict) and _vec(field[k].get("xy"), 2)):
            errs.append(f"{k}.xy missing")
    errs += _validate_events(field)
    if not errs:
        errs += check_spacing(field)
        errs += check_clearances(field)
    return errs


def _validate_events(field: dict) -> List[str]:
    """``events[]`` errors (``controllers.world_events.validate_events``: types, trigger, light preset, fog density,
    actor ids), so a bad authored event fails at load, not as a hook fault during a tick."""
    from atharv_nandi.core import world_events as wev
    acts = field.get("actors") or []
    ids = [a.get("id") for a in acts if isinstance(a, dict)] if isinstance(acts, list) else []
    return wev.validate_events(field.get("events"), ids)


def check_spacing(field: dict) -> List[str]:
    """Mango rows use the frozen NHB HDP spacing 5 x 2.5 or 5 x 3 m (OQ-02 B)."""
    errs = []
    ys = {r["row_id"]: float(r["y"]) for r in rows(field)}
    for row in rows(field):
        if row.get("crop") != "mango" or row.get("form") == "hedgerow":
            continue
        ts = trees_in_row(field, row["row_id"])
        xs = [tree_xy(t)[0] for t in ts]
        gaps = [round(b - a, 3) for a, b in zip(xs, xs[1:])]
        pitches = [abs(ys[row["row_id"]] - y) for rid, y in ys.items() if rid != row["row_id"]]
        pitch = min(pitches) if pitches else None
        ok_in = {s[1] for s in NHB_MANGO_SPACINGS_M}
        if any(g not in ok_in for g in gaps) or len(set(gaps)) > 1:
            errs.append(f"row {row['row_id']}: mango in-row gaps {gaps} not one of {sorted(ok_in)} (NHB, OQ-02 B)")
        if pitch is not None and abs(pitch - NHB_MANGO_SPACINGS_M[0][0]) > 1e-6:
            errs.append(f"row {row['row_id']}: mango row pitch {pitch} m != 5.0 m (NHB, OQ-02 B)")
    return errs


def check_clearances(field: dict) -> List[str]:
    """``orbit_clear`` trees need a clear annulus of r1 + 0.6 m to any other canopy (OS 4.2, OT-1)."""
    errs = []
    for t in trees(field):
        if not t.get("orbit_clear"):
            continue
        x, y = tree_xy(t)
        r1 = float(t["canopy_r_m"]) + float(t.get("orbit_standoff_m", 0.9))
        need = r1 + 0.6
        for o in trees(field):
            if o is t:
                continue
            ox, oy = tree_xy(o)
            gap = math.hypot(ox - x, oy - y) - float(o["canopy_r_m"])
            if gap < need - 1e-9:
                errs.append(f"tree {t['tree_id']}: orbit annulus {need:.2f} m reaches {o['tree_id']} canopy ({gap:.2f} m)")
    return errs


def summary(field: dict) -> dict:
    """Counts the console shows on the map (crop counts, rows, fixtures)."""
    by_crop: Dict[str, int] = {}
    for t in trees(field):
        by_crop[t["crop"]] = by_crop.get(t["crop"], 0) + 1
    return {"world": field.get("world"), "kind": field.get("kind"), "n_trees": len(trees(field)),
            "by_crop": by_crop, "rows": [r["row_id"] for r in rows(field)],
            "n_laterals": len(laterals(field)) if is_orchard(field) else 0,
            "n_ropes": len(rope_lines(field)), "n_props": len(props(field)),
            "n_actors": len(field.get("actors") or []), "n_events": len(field.get("events") or [])}


def iter_worlds(fields_dir: str = FIELDS_DIR) -> Iterable[str]:
    for w in ("legacy",) + ORCHARD_WORLDS:
        if os.path.isfile(field_path(w, fields_dir)):
            yield w
