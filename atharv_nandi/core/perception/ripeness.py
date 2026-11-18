"""WP26 maturity per crop profile and the harvest window per tree and block (ORCHARD-MVP-SPEC 4.4; MISSION-PRESETS 4).

ROS-free, MuJoCo-free. Two layers:

* **Cue observation** (:func:`observe_cues`, SimTruth, assisted and labelled): the WP23 truth cues of one fruit
  seen through a noise model (mango shoulder rise mm + stem-end colour; guava skin score; papaya orange streak
  count; banana finger angularity). The ColorBlob baseline (:func:`colorblob_stage`) only sees the rendered
  colour and is reported, never gated.
* **Stage decision** (:func:`classify`): per crop from the observed cues plus the registry date.
  **Mango never uses skin colour** (OS 4.4: Dashehari / Langra are harvested mature-green): any skin or bare
  colour key in the observation is dropped before the vote (``world.crops.is_skin_colour_cue``), and the vote is
  shoulder rise + stem-end colour + days since ``fruit_set_date``.

Harvest window (:func:`tree_window`): from the registry date when the registry has one (mango / guava / papaya
``fruit_set_date``, mango ``bagging_date`` → basis ``calendar``, banana ``shooting_date`` or the ribbon rota), else
from the observed stage distribution (harvest-stage fruit seen → open now, basis ``stage_distribution``), else the
UP calendar for the cultivar (basis ``calendar_up``). :func:`block_window` spans the trees.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi.core.perception import pest_detect as pdet
from atharv_nandi.sim.world import crops as cp

HARVEST_REFUSED_BAGGED = "bagged"


def _cfg(cfg: Optional[dict]) -> dict:
    return (cfg or pdet.load_config())["ripeness"]


# ---------------------------------------------------------------------------------------------- observation
def observe_cues(fruit: dict, profile: dict, rng: np.random.Generator, cfg: Optional[dict] = None) -> dict:
    """Noisy cue observation of one truth fruit (SimTruth). Mango: no skin colour key is ever produced."""
    c = _cfg(cfg)
    crop = profile["crop"]
    cues = fruit.get("cues") or {}
    out: Dict[str, object] = {}
    if crop == "mango":
        m = c["mango"]
        if "shoulder_rise_mm" in cues:
            out["shoulder_rise_mm"] = round(float(cues["shoulder_rise_mm"]) + float(rng.normal(0, m["shoulder_sigma_mm"])), 2)
        if "stem_end_colour" in cues:
            col = cues["stem_end_colour"]
            if rng.uniform() < float(m["stem_colour_flip_p"]):
                col = "green" if col == "yellowing" else "yellowing"
            out["stem_end_colour"] = col
    elif crop == "guava":
        base = {"dark_green": 0.25, "light_green_yellow": 0.75}.get(cues.get("skin"), 0.5)
        out["skin_score"] = round(float(np.clip(base + rng.normal(0, c["guava"]["hue_sigma"]), 0, 1)), 3)
    elif crop == "papaya":
        n = cues.get("orange_streaks", 0)
        n = int(round(float(np.mean(n)))) if isinstance(n, list) else int(n)
        seen = int(sum(1 for _ in range(n) if rng.uniform() >= float(c["papaya"]["streak_miss_p"])))
        out["orange_streaks"] = seen
    elif crop == "banana":
        f = cues.get("fingers")
        if f is not None and rng.uniform() < float(c["banana"]["angularity_flip_p"]):
            f = "round" if f == "angular" else "angular"
        if f is not None:
            out["fingers"] = f
    return out


def fuse_cues(obs: Sequence[dict]) -> dict:
    """Fuse per-view cue observations of one tracked fruit: numeric mean, string majority (ties: first seen)."""
    out: Dict[str, object] = {}
    keys = []
    for o in obs:
        for k in o:
            if k not in keys:
                keys.append(k)
    for k in keys:
        vals = [o[k] for o in obs if k in o]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = round(float(np.mean(vals)), 3)
        else:
            counts: Dict[object, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            out[k] = max(counts, key=lambda v: (counts[v], -vals.index(v)))
    return out


# ---------------------------------------------------------------------------------------------- stage decision
def days_since(entry: dict, profile: dict, today) -> Optional[float]:
    """Days since the registry fruit set (or shooting for banana; mango bagging date − 28 d when only that)."""
    t = cp.parse_date(today)
    if t is None:
        return None
    if profile["harvest_window"]["basis"] == "days_since_shooting":
        s = cp.parse_date(entry.get("shooting_date"))
    else:
        s = cp.parse_date(entry.get("fruit_set_date"))
        if s is None and entry.get("bagging_date") and (profile.get("bagging") or {}).get("used"):
            s = cp.parse_date(entry["bagging_date"]) - _dt.timedelta(days=profile["bagging"]["days_after_fruit_set"][0])
    return None if s is None else float((t - s).days)


def _no_skin(profile: dict, obs: dict) -> dict:
    if profile["crop"] in cp.NO_SKIN_COLOUR_CROPS or not profile.get("skin_colour_cue", False):
        return {k: v for k, v in obs.items() if not cp.is_skin_colour_cue(k) and k not in ("skin_score", "rgb", "rgba")}
    return dict(obs)


def classify(profile: dict, obs: dict, days: Optional[float] = None, cfg: Optional[dict] = None
             ) -> Tuple[str, float, str]:
    """(stage id, confidence 0..1, basis) for one fruit from observed cues (+ registry age)."""
    c = _cfg(cfg)
    crop = profile["crop"]
    hs = cp.harvest_stages(profile)
    imm = [s["id"] for s in profile["stages"] if not s.get("harvest")]
    ripe, green = hs[0], (imm[0] if imm else hs[0])
    obs = _no_skin(profile, obs)
    if crop == "mango":
        w = c["mango"]["weights"]
        votes, wsum = 0.0, 0.0
        if "shoulder_rise_mm" in obs:
            x = float(obs["shoulder_rise_mm"])
            votes += w["shoulder_rise"] / (1.0 + math.exp(-(x - 1.5) / 0.25))
            wsum += w["shoulder_rise"]
        if "stem_end_colour" in obs:
            votes += w["stem_end_colour"] * (1.0 if obs["stem_end_colour"] == "yellowing" else 0.0)
            wsum += w["stem_end_colour"]
        if days is not None:
            votes += w["days_since_fruit_set"] * (1.0 if cp.stage_for_age(profile, days) in hs else 0.0)
            wsum += w["days_since_fruit_set"]
        if wsum <= 0:
            return green, 0.0, "none"
        score = votes / (w["shoulder_rise"] + w["stem_end_colour"] + w["days_since_fruit_set"])
        stage = ripe if score >= 0.6 else green
        conf = (score - 0.6) / 0.4 if stage == ripe else (0.6 - score) / 0.6
        return stage, round(min(1.0, max(0.0, conf)), 3), "shoulder_rise+stem_end_colour+days_since_fruit_set"
    if crop == "guava":
        s = float(obs.get("skin_score", 0.5))
        return (ripe if s >= 0.5 else green), round(min(1.0, abs(s - 0.5) * 2.0), 3), "skin_colour"
    if crop == "papaya":
        n = int(obs.get("orange_streaks", 0))
        return (ripe if n >= 2 else green), (1.0 if n >= 3 or n == 0 else 0.6), "colour_break_streaks"
    if crop == "banana":
        if days is not None:
            return cp.stage_for_age(profile, days), 0.9, "ribbon_age"
        f = obs.get("fingers")
        if f is None:
            return green, 0.0, "none"
        return (ripe if f == "round" else green), 0.6, "finger_angularity"
    st = cp.stage_from_cues(profile, obs)
    if st is None and days is not None:
        st = cp.stage_for_age(profile, days)
    return (st or green), 0.5, "profile_cues"


def colorblob_stage(profile: dict, rgb, rng: Optional[np.random.Generator] = None, cfg: Optional[dict] = None) -> str:
    """ColorBlob baseline: the stage whose rgba is nearest the observed colour (ties: the first stage). For mango
    the stage colours are identical on purpose, so this baseline cannot separate stages (reported, never gated)."""
    x = np.asarray(rgb, float)[:3]
    if rng is not None:
        x = x + rng.normal(0, _cfg(cfg)["colorblob_stage"]["chroma_sigma"], 3)
    best, bd = profile["stages"][0]["id"], float("inf")
    for s in profile["stages"]:
        d = float(np.linalg.norm(np.asarray(s["rgba"], float)[:3] - x))
        if d < bd - 1e-9:
            best, bd = s["id"], d
    return best


def harvest_refusal(fruit: dict) -> Optional[str]:
    """HARVEST is refused on bagged fruit (OQ-15 A): reason code ``bagged``."""
    return HARVEST_REFUSED_BAGGED if fruit.get("bagged") else None


# ---------------------------------------------------------------------------------------------- windows
def _with_today(w: dict, today) -> dict:
    out = dict(w)
    if today is not None:
        t = cp.parse_date(today)
        out["days_to_open"] = (cp.parse_date(out["from"]) - t).days
        out["open"] = cp.parse_date(out["from"]) <= t <= cp.parse_date(out["to"])
    return out


def banana_entry(entry: dict, bunch_obs: Optional[dict] = None, rota: Optional[Dict[str, str]] = None) -> dict:
    """Registry entry with ``shooting_date`` resolved from the ribbon colour via the farm rota when missing."""
    e = dict(entry)
    if not e.get("shooting_date"):
        colour = (bunch_obs or {}).get("ribbon_colour") or e.get("ribbon_colour")
        sd = cp.ribbon_shooting_date(colour, rota or {}) if colour else None
        if sd:
            e["shooting_date"] = sd
    return e


def tree_window(profile: dict, entry: dict, stage_counts: Optional[Dict[str, int]] = None, today=None,
                cultivar: Optional[str] = None) -> Optional[dict]:
    """Harvest window ``{from, to, basis}`` (+ ``days_to_open``, ``open`` with ``today``) for one tree."""
    w = cp.harvest_window(profile, entry)
    if w is not None:
        return _with_today(w, today)
    lo, hi = profile["harvest_window"]["days"]
    hs = set(cp.harvest_stages(profile))
    t = cp.parse_date(today)
    if stage_counts and t is not None and sum(n for s, n in stage_counts.items() if s in hs) > 0:
        return _with_today({"from": t.isoformat(), "to": (t + _dt.timedelta(days=int(hi - lo))).isoformat(),
                            "basis": "stage_distribution"}, today)
    cal = cp.calendar_window(profile, cultivar or entry.get("cultivar"), (t or _dt.date(2026, 1, 1)).year)
    if cal:
        return _with_today({"from": cal[0], "to": cal[1], "basis": "calendar_up"}, today)
    return None


def block_window(windows: Sequence[Optional[dict]], today=None) -> Optional[dict]:
    """Block harvest window: earliest ``from`` to latest ``to`` over the trees, with the count of trees open."""
    ws = [w for w in windows if w]
    if not ws:
        return None
    out = {"from": min(w["from"] for w in ws), "to": max(w["to"] for w in ws),
           "basis": "trees:" + "+".join(sorted({w["basis"] for w in ws})), "trees_n": len(ws)}
    if today is not None:
        out = _with_today(out, today)
        out["trees_open_n"] = sum(1 for w in ws if _with_today(w, today)["open"])
    return out


def window_equals(a: Optional[dict], b: Optional[dict]) -> bool:
    """Window identity on (from, to, basis): the acceptance check "window = truth"."""
    if a is None or b is None:
        return a is b
    return all(a.get(k) == b.get(k) for k in ("from", "to", "basis"))


__all__ = ["banana_entry", "block_window", "classify", "colorblob_stage", "days_since", "fuse_cues",
           "harvest_refusal", "observe_cues", "tree_window", "window_equals", "List"]
