"""Crop profiles for the orchard MVP (WP23; ORCHARD-MVP-SPEC 4.4, MISSION-PRESETS 4-6).

``config/crops/<crop>.yaml`` (schema ``nandi.crop_profile/1``) holds what a task layer needs to know about a
crop and nothing about geometry (that is ``config/trees/<species>.yaml``, read by ``world.trees``):

    stage model and maturity cues   (mango: shoulder rise, stem-end colour, days since fruit set; never skin colour)
    harvest_method / methods / tool / pool_kind   (mango clip on a connect slot; guava clip or twist and papaya
                                                    twist on weld slots; banana MARK only)
    pedicel_leave_cm, tau_detach_nm, placement_speed_max_m_s
    prune jobs, capacity (max_stem_mm) and paste_required_mm
    min_functional_leaves (banana MARK planner), symptom classes with band and season, UP calendar windows

Pure Python (PyYAML only), ROS- and MuJoCo-free. Every function returns plain data; nothing here decides a
diagnosis or a treatment (design rule 15): the outputs are leads, MARK reasons and harvest-window estimates.

The directory is ``config/crops`` unless ``NANDI_CROPS_CONFIG`` names another one (tests, a 5th crop).
"""
import datetime as _dt
import glob
import hashlib
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from atharv_nandi import paths

REPO = paths.REPO
CROPS_DIR = os.path.join(REPO, "config", "crops")
SCHEMA = "nandi.crop_profile/1"

PICK_BAND_Z_M = 0.8                 # OS 2.2: contact targets at z <= 0.8 m
HEIGHT_CLASSES = (("neeche", 0.8), ("beech", 1.8), ("upar", float("inf")))   # MISSION-PRESETS 0.6
HARVEST_METHODS = ("clip", "twist", "mark")
POOL_KINDS = ("connect", "weld", "none")
MVP_SCOPES = ("harvest", "scout_mark")
SEASONS = ("panicle", "fruit_set", "bagged", "mature", "post_harvest")
# Crops whose maturity must never be read from skin colour (OS 4.4: mango is harvested mature-green).
NO_SKIN_COLOUR_CROPS = ("mango",)
_SKIN_PARTS = {"skin", "peel", "rind", "surface", "blush"}
_COLOUR_WORDS = {"colour", "color", "hue", "tint", "rgb", "rgba", "blush"}
_NON_SKIN_PARTS = {"stem", "pedicel", "flesh", "pulp", "seed", "stone", "shoulder", "latex", "sap"}
REQUIRED = ("schema", "crop", "cultivars", "default_cultivar", "mvp_scope", "harvest_method", "methods",
            "pool_kind", "pedicel_leave_cm", "stages", "harvest_window", "prune", "symptoms", "skin_colour_cue",
            "maturity_cues")


class CropProfileError(ValueError):
    pass


def crops_dir() -> str:
    return os.environ.get("NANDI_CROPS_CONFIG", "").strip() or CROPS_DIR


def _yaml_load(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise CropProfileError(f"{path}: not a mapping")
    return data


def is_skin_colour_cue(name) -> bool:
    """True for a cue that reads the fruit's skin colour: any skin / peel / rind word, or a bare colour word
    (``colour``, ``peel_color``, ``skin_hue``...) that names no other part. ``stem_end_colour`` (a stem-end
    cue) is not a skin cue."""
    toks = {t for t in str(name).lower().replace("-", "_").replace(" ", "_").split("_") if t}
    if toks & _SKIN_PARTS:
        return True
    return bool(toks & _COLOUR_WORDS) and not (toks & _NON_SKIN_PARTS)


def validate(p: dict) -> List[str]:
    """Every problem with a profile, as text ('' list = valid)."""
    errs = []
    for k in REQUIRED:
        if k not in p:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if p["schema"] != SCHEMA:
        errs.append(f"schema {p['schema']!r} != {SCHEMA}")
    if p["harvest_method"] not in HARVEST_METHODS:
        errs.append(f"harvest_method {p['harvest_method']!r} not in {HARVEST_METHODS}")
    if any(m not in HARVEST_METHODS for m in p["methods"]) or p["harvest_method"] not in p["methods"]:
        errs.append("methods must be a subset of clip|twist|mark and contain harvest_method")
    if p["pool_kind"] not in POOL_KINDS:
        errs.append(f"pool_kind {p['pool_kind']!r} not in {POOL_KINDS}")
    if p["harvest_method"] == "mark" and p["pool_kind"] != "none":
        errs.append("a MARK-only crop has pool_kind none")
    if p["harvest_method"] == "clip" and p["pool_kind"] == "none":
        errs.append("a clipped crop needs a pool slot kind")
    if p["mvp_scope"] not in MVP_SCOPES:
        errs.append(f"mvp_scope {p['mvp_scope']!r} not in {MVP_SCOPES}")
    if p["default_cultivar"] not in p["cultivars"]:
        errs.append("default_cultivar not in cultivars")
    ped = p["pedicel_leave_cm"]
    if not (isinstance(ped, dict) and ped.get("min", 0) <= ped.get("default", 0) <= ped.get("max", 0)):
        errs.append("pedicel_leave_cm needs min <= default <= max")
    if "twist" in p["methods"]:
        tau = p.get("tau_detach_nm")
        if not isinstance(tau, (int, float)) or not 0.05 <= tau <= 2.0:
            errs.append("a twist crop needs tau_detach_nm in [0.05, 2.0] N.m")
    stages = p["stages"]
    if not isinstance(stages, list) or len(stages) < 2:
        errs.append("stages needs >= 2 entries")
    else:
        ids = [s.get("id") for s in stages]
        if len(set(ids)) != len(ids):
            errs.append("stage ids repeat")
        if not any(s.get("harvest") for s in stages):
            errs.append("no stage is harvestable")
        for s in stages:
            if len(s.get("rgba", [])) != 4:
                errs.append(f"stage {s.get('id')!r}: rgba needs 4 values")
        no_skin = p["crop"] in NO_SKIN_COLOUR_CROPS
        if no_skin and p["skin_colour_cue"] is not False:
            errs.append(f"{p['crop']}: skin_colour_cue must be false (OS 4.4: never a maturity cue)")
        if no_skin or not p["skin_colour_cue"]:
            # The rule that matters for mango (OS 4.4): skin colour must carry no stage information.
            if len({tuple(s["rgba"]) for s in stages if len(s.get("rgba", [])) == 4}) > 1:
                errs.append("skin colour is not a cue but stage colours differ")
            listed = [c for c in p["maturity_cues"] if is_skin_colour_cue(c)]
            listed += [k for s in stages for k in (s.get("cues") or {}) if is_skin_colour_cue(k)]
            if listed:
                errs.append(f"skin colour is not a cue but a skin cue is listed: {sorted(set(listed))}")
        key = "days_since_shooting" if (p["harvest_window"] or {}).get("basis") == "days_since_shooting" \
            else "days_since_fruit_set"
        spans = [s.get(key) for s in stages if s.get(key) is not None]
        if spans:
            if len(spans) != len(stages) or any(len(r) != 2 or r[0] > r[1] for r in spans):
                errs.append(f"every stage needs a [lo, hi] {key} range")
            elif any(int(b[0]) != int(a[1]) + 1 for a, b in zip(spans, spans[1:])):
                errs.append(f"stage {key} ranges must tile whole days (next lo = hi + 1)")
    prune = p["prune"]
    if not isinstance(prune, dict) or "paste_required_mm" not in prune or "max_stem_mm" not in prune:
        errs.append("prune needs max_stem_mm and paste_required_mm")
    for s in p["symptoms"]:
        if not {"class", "unit", "bands", "seasons"} <= set(s):
            errs.append(f"symptom {s!r} needs class, unit, bands, seasons")
        elif any(x not in SEASONS + ("all",) for x in s["seasons"]):
            errs.append(f"symptom {s['class']}: unknown season")
    return errs


def load_profile(crop: str, directory: Optional[str] = None) -> dict:
    path = os.path.join(directory or crops_dir(), f"{crop}.yaml")
    if not os.path.isfile(path):
        raise CropProfileError(f"no crop profile {path}")
    p = _yaml_load(path)
    errs = validate(p)
    if errs:
        raise CropProfileError(f"{path}: " + "; ".join(errs))
    if p["crop"] != crop:
        raise CropProfileError(f"{path}: crop {p['crop']!r} != file name {crop!r}")
    return p


def load_profiles(directory: Optional[str] = None) -> Dict[str, dict]:
    d = directory or crops_dir()
    names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(d, "*.yaml")))
    return {n: load_profile(n, d) for n in names}


def config_digest(directory: Optional[str] = None) -> str:
    """sha1 over the profile files (name + bytes): folded into the trees module fingerprint / world_id."""
    d = directory or crops_dir()
    h = hashlib.sha1()
    for f in sorted(glob.glob(os.path.join(d, "*.yaml"))):
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


# ---------------------------------------------------------------------- stage and windows
def stage(p: dict, stage_id: str) -> dict:
    for s in p["stages"]:
        if s["id"] == stage_id:
            return s
    raise KeyError(f"{p['crop']}: no stage {stage_id!r}")


def harvest_stages(p: dict) -> List[str]:
    return [s["id"] for s in p["stages"] if s.get("harvest")]


def _age_key(p: dict) -> str:
    return "days_since_shooting" if p["harvest_window"]["basis"] == "days_since_shooting" else "days_since_fruit_set"


def stage_for_age(p: dict, days: float) -> str:
    """Stage from the age (days since fruit set, or since shooting for banana): the registry / calendar basis."""
    key = _age_key(p)
    d = math.floor(float(days))       # completed days: 83.5 d since fruit set is day 83 (ranges are whole days)
    for s in p["stages"]:
        lo, hi = s.get(key, [None, None])
        if lo is not None and lo <= d <= hi:
            return s["id"]
    # before the first range: the first stage; past the last range: the last stage (an over-age fruit is still
    # on the tree and still in its last stage; harvest_window says whether the window has closed)
    return p["stages"][-1]["id"] if d > 0 else p["stages"][0]["id"]


def stage_from_cues(p: dict, cues: dict) -> Optional[str]:
    """The stage whose cue ranges contain ``cues`` (SimTruth / ColorBlob output), or None if ambiguous/empty.

    Numeric cue ranges are [lo, hi]; strings must match. Skin colour is never consulted when the profile says
    ``skin_colour_cue: false`` or the crop is in ``NO_SKIN_COLOUR_CROPS`` (any skin or bare colour cue in the input,
    see ``is_skin_colour_cue``, is ignored for mango)."""
    hits = []
    for s in p["stages"]:
        want = s.get("cues") or {}
        ok, used = True, 0
        for k, v in want.items():
            if is_skin_colour_cue(k) and (not p["skin_colour_cue"] or p["crop"] in NO_SKIN_COLOUR_CROPS):
                continue
            if k not in cues:
                continue
            used += 1
            got = cues[k]
            if isinstance(v, list):
                ok &= isinstance(got, (int, float)) and v[0] <= got <= v[1]
            else:
                ok &= got == v
        if ok and used:
            hits.append(s["id"])
    return hits[0] if len(hits) == 1 else None


def parse_date(text) -> Optional[_dt.date]:
    if not text:
        return None
    if isinstance(text, _dt.date):
        return text
    return _dt.date.fromisoformat(str(text))


def harvest_window(p: dict, entry: dict, today=None) -> Optional[dict]:
    """{from, to, basis} for a registry entry (OS 4.4 / MISSION-PRESETS 4): from the fruit-set date (mango,
    guava, papaya; mango bagging date minus 28 d when only that is known) or the banana shooting date
    (registry ``shooting_date``, or the ribbon colour mapped by the caller). None if the registry lacks a date."""
    win = p["harvest_window"]
    lo, hi = win["days"]
    basis = win["basis"]
    if basis == "days_since_shooting":
        start = parse_date(entry.get("shooting_date"))
    else:
        start = parse_date(entry.get("fruit_set_date"))
        if start is None and entry.get("bagging_date") and p.get("bagging", {}).get("used"):
            start = parse_date(entry["bagging_date"]) - _dt.timedelta(days=p["bagging"]["days_after_fruit_set"][0])
            basis = "calendar"
    if start is None:
        return None
    out = {"from": (start + _dt.timedelta(days=lo)).isoformat(), "to": (start + _dt.timedelta(days=hi)).isoformat(),
           "basis": basis}
    if today is not None:
        t = parse_date(today)
        out["days_to_open"] = (parse_date(out["from"]) - t).days
        out["open"] = parse_date(out["from"]) <= t <= parse_date(out["to"])
    return out


def calendar_window(p: dict, cultivar: Optional[str] = None, year: int = 2026) -> Optional[Tuple[str, str]]:
    """UP calendar harvest window (ISO dates) for the cultivar; a window across New Year ends next year."""
    cal = p.get("calendar_up") or {}
    c = cal.get(cultivar or p["default_cultivar"])
    if not c or "harvest" not in c:
        return None
    a, b = c["harvest"]
    start = _dt.date.fromisoformat(f"{year}-{a}")
    end = _dt.date.fromisoformat(f"{year}-{b}")
    if end < start:
        end = _dt.date.fromisoformat(f"{year + 1}-{b}")
    return start.isoformat(), end.isoformat()


def h_class(z: float) -> str:
    for name, top in HEIGHT_CLASSES:
        if z < top:
            return name
    return "upar"


# ---------------------------------------------------------------------- decisions (advisory, reason-coded)
def harvest_decision(p: dict, fruit: dict, method: Optional[str] = None) -> Tuple[str, str]:
    """('pick'|'mark'|'skip', reason) for one truth fruit (``world.trees`` fruit dict) under this profile.

    Reason codes are MISSION-PRESETS 0.5 ones: bagged, out_of_reach_marked, not_in_mvp (MARK-only crop),
    no_clearance (papaya jaw clearance), '' when picking. Immature fruit are skipped ('', not a MARK)."""
    m = method or p["harvest_method"]
    if m not in p["methods"]:
        return "skip", "not_in_mvp"
    if fruit.get("bagged"):
        return "skip", "bagged"
    if fruit.get("stage") not in harvest_stages(p):
        return "skip", ""
    if p["harvest_method"] == "mark" or m == "mark":
        return "mark", "human_cut"
    if not fruit.get("in_band", fruit.get("stem_end", [0, 0, 9])[2] <= PICK_BAND_Z_M):
        return "mark", "out_of_reach_marked"
    if m == "twist" and p.get("jaw_clearance_mm") is not None:
        gap = fruit.get("clearance_mm")
        if gap is not None and gap < p["jaw_clearance_mm"]:
            return "mark", "no_clearance"
    return "pick", ""


def paste_required(p: dict, diameter_mm: float) -> bool:
    return float(diameter_mm) > float(p["prune"]["paste_required_mm"])


def prune_decision(p: dict, job: str, diameter_mm: float, z: float, keep: bool = False) -> Tuple[str, str]:
    """('cut'|'mark'|'skip', reason) for one prune candidate. Never cuts keep wood (O-13: 0 keep-wood cuts)."""
    if keep:
        return "skip", "keep"
    if job not in p["prune"].get("jobs", []) and job not in ("dead_blade", "sucker"):
        return "skip", "not_in_mvp"
    if job in ("banana_sanitation_deleaf", "banana_desucker", "sucker"):
        return "mark", "human_cut"
    if z > PICK_BAND_Z_M:
        return "mark", "out_of_reach_marked"
    if diameter_mm > p["prune"]["max_stem_mm"]:
        return "mark", "paste_required" if paste_required(p, diameter_mm) else "over_capacity"
    return "cut", ""


def min_functional_leaves(p: dict, phase: str) -> Optional[int]:
    """Banana MARK planner floor (default 9 at flowering, >= 4 at harvest, assumption); None for other crops."""
    m = p.get("min_functional_leaves")
    return None if not m else int(m[phase])


def deleaf_plan(p: dict, leaves: Sequence[dict], phase: str = "flowering") -> List[str]:
    """Leaf ids a human may remove (sanitation deleaf MARK) without going below the functional-leaf floor.

    Dead leaves always go; Sigatoka leaves go worst first while the functional count stays >= the floor."""
    floor = min_functional_leaves(p, phase)
    if floor is None:
        return []
    dead = [lf["id"] for lf in leaves if not lf.get("functional", True)]
    functional = [lf for lf in leaves if lf.get("functional", True)]
    sick = sorted((lf for lf in functional if lf.get("sigatoka_pct", 0) > 0), key=lambda lf: -lf["sigatoka_pct"])
    out, n = list(dead), len(functional)
    for lf in sick:
        if n - 1 < floor:
            break
        out.append(lf["id"])
        n -= 1
    return out


def symptom_classes(p: dict, season: str, scale: Optional[str] = None) -> List[dict]:
    """Symptom classes active in ``season`` (``demo_all_fruiting`` counts as ``mature``)."""
    s = "mature" if season == "demo_all_fruiting" else season
    return [c for c in p["symptoms"] if ("all" in c["seasons"] or s in c["seasons"])
            and (scale is None or c.get("scale") == scale)]


def ribbon_shooting_date(colour: str, rota: Dict[str, str]) -> Optional[str]:
    """Shooting date (ISO) from a bunch ribbon colour via the farm's ribbon rota {colour: shooting date}.

    The rota is a field-registry fact (one colour per shooting week, reused after the cycle), so the colour alone
    is ambiguous across cycles; None when the colour is not in the rota."""
    d = rota.get(colour) if rota else None
    return parse_date(d).isoformat() if d else None
