"""Ready check for the orchard mission presets (WP27; pure Python, no ROS, no MuJoCo).

MISSION-PRESETS 0.2 and ORCHARD-MVP-SPEC 4.5: a preset tile stays disabled, showing its reason, until every check
passes. ``check(preset, ctx, targets, params, field_doc)`` returns a ``ReadyResult``:

    level        go | degraded | no_go   (degraded = warn, the start needs an operator confirm)
    reasons      reason codes, no_go first (MISSION-PRESETS 0.5 plus the WP27 extensions below)
    items        one row per check: {check, level, reason, detail, values}; level go | info | degraded | no_go
    chips        info rows (``crown_gate: slope``, ``two_sided``...): the run goes ahead with ``adjust``
    adjust       param changes the run makes because of a chip (crown_band: stowed, arc_mode: two_sided, ...)
    estimate     the preset estimate for these targets (presets.Preset.estimate) + the energy plan
    season       {key, month, level, reason, note, label}

Common checks (MP 0.2): RTK FIXED (FLOAT only for presets with ``float_ok``: RETURN HOME, SAFE PARK); projected end
SoC >= reserve (WP7 route_energy: return-to-home energy + 15 % of the pack); free disk >= 1.5 x the estimate;
geofence contains the whole path incl. orbit annuli; e-stop clear; local safety link up (WP7: unknown refuses);
cloud link not lost; stability OK; local cross-slope within the preset limit and the WP6 slope / dwell gates;
season matrix (config/season/up.yaml); weather (WP7 weather_stub: rain no-go, wet foliage no-go for LEAF SCAN and
contact presets, > 42 C no-go, > 38 C degraded); noVNC off while actors are enabled. Preset preconditions add the
crown gates (WP6 ``gate_raise`` / ``gate_crown_walk`` on the WP6 matrix table), wind, alley clearance, tool in the
rack, payload room, light (dawn mode for LEAF SCAN), bagged fruit, registry fields, job sheet, tag, confirmations.
A preset skips the common checks listed in its ``ready.relax``.

WP27 reason-code extensions (not in MISSION-PRESETS 0.5; change request to WP34 ``atharv_nandi/core/workflow/reasons.py``):
``estop`` (already in the WP34 enum), ``novnc_actors``, ``param_invalid``, ``preset_unknown``, ``job_sheet_missing``,
``tag_missing``, ``sensor_fault``, ``confirm_required``, ``estop_not_engaged``, ``not_fallen``, ``queue_full``,
``not_checked``.
"""
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from atharv_nandi.core import presets as ps
from atharv_nandi.core import stability as st
from atharv_nandi.core import weather_stub as ws
from atharv_nandi.core.energy_model import EnergyParams
from atharv_nandi.core import gnss_model as gm
from atharv_nandi.core.gnss_model import FIXED, FLOAT
from atharv_nandi.core.route_energy import Coefficients, RoutePlan, ready_check_energy

ROOT = ps.ROOT
SEASON_YAML = os.path.join(ROOT, "config", "season", "up.yaml")
CROPS_DIR = os.path.join(ROOT, "config", "crops")

GO, INFO, DEGRADED, NO_GO = "go", "info", "degraded", "no_go"
_RANK = {GO: 0, INFO: 1, DEGRADED: 2, NO_GO: 3}

DISK_FACTOR = 1.5                 # free disk >= 1.5 x the preset estimate (disk headroom rule)
PAYLOAD_MAX_KG = 3.0              # arm payload / basket cap (MP 5 interlocks)
DETOUR_FACTOR = 1.3               # straight-line transit x this for the energy plan (assumption)
ORBIT_ANNULUS_M = 0.6             # annulus R +- 0.6 m clear (MP 2)
CLEARANCE_MARGIN_M = 0.10         # swept-clearance margin (OS 2.3 rule 1, used by rule 2 for the route)
DEMO_PROFILES = ("demo_all_fruiting",)
NOMINAL_LUX = 20000.0             # sim lighting "morning" (assumption)
LIGHT_PRESET_LUX = {"noon": 80000.0, "morning": 20000.0, "overcast": 8000.0, "dawn": 800.0, "dusk": 400.0}  # assumption
WEATHER_REPRESENTATIVE = {"exempt": "recovery", "leaf": "leaf_scan", "contact": "harvest", "non_contact": "scan"}

# WP27 reason codes outside MISSION-PRESETS 0.5 (the WP27 contract notes)
WP27_REASONS = ("estop", "novnc_actors", "param_invalid", "preset_unknown", "job_sheet_missing", "tag_missing",
                "sensor_fault", "confirm_required", "estop_not_engaged", "not_fallen", "queue_full", "not_checked")

COMMON = ("rtk", "energy", "disk", "geofence", "estop", "local_link", "cloud_link", "stability", "slope", "season",
          "weather", "novnc")

# Every precondition name a preset file may list (presets.validate_doc refuses others).
PRECONDITIONS = (
    "ready_common", "registry_present", "canopy_r_known", "slope_walk", "dwell_slope", "slope_crown_walk",
    "crown_walk_or_stowed", "crown_gate", "alley_clearance", "annulus_clear_or_two_sided", "light", "no_wet_foliage",
    "work_field_clear", "wrist_camera_ok", "crop_leaves_in_band", "graft_union_known", "frames_fresh_or_crown_gate",
    "tool_in_rack", "payload_room", "crop_profile", "scout_fresh_or_inline", "fruit_not_bagged", "crop_in_mvp",
    "season_window", "job_sheet_exists", "tag_paired", "tote_on_deck", "crown_stowed", "local_link_for_move",
    "estop_engaged", "both_confirmations", "fallen_latched", "person_clear_confirm",
)


# ------------------------------------------------------------------ season matrix
class Season:
    """config/season/up.yaml: per key and month go / degraded / no_go with the reason (OS 4.5)."""

    def __init__(self, doc: dict, path: str = ""):
        self.doc, self.path = doc, path
        self.groups = {g: [int(m) for m in ms] for g, ms in (doc.get("groups") or {}).items()}
        self.table: Dict[str, Dict[int, dict]] = {}
        for key, rows in (doc.get("keys") or {}).items():
            by_month: Dict[int, dict] = {}
            singles = {}
            for k, v in rows.items():
                if isinstance(k, int) or str(k).isdigit():
                    singles[int(k)] = v
                else:
                    for mo in self.groups.get(k, []):
                        by_month[mo] = v
            by_month.update(singles)
            self.table[key] = by_month
        self.profiles = doc.get("profiles") or {}
        errs = self.validate()
        if errs:
            raise ValueError("; ".join(errs))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Season":
        path = path or SEASON_YAML
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh), path)

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self.table)

    def validate(self) -> List[str]:
        errs = []
        if self.doc.get("schema") != "nandi.season/1":
            errs.append("season schema must be nandi.season/1")
        months = sorted(m for ms in self.groups.values() for m in ms)
        if months != list(range(1, 13)):
            errs.append(f"season groups must cover months 1..12 once, got {months}")
        for key, rows in self.table.items():
            missing = [m for m in range(1, 13) if m not in rows]
            if missing:
                errs.append(f"season key {key!r}: months {missing} missing")
            for m, v in rows.items():
                if not isinstance(v, dict) or v.get("level") not in (GO, DEGRADED, NO_GO):
                    errs.append(f"season key {key!r} month {m}: level must be go / degraded / no_go")
                elif v["level"] != GO and not v.get("reason"):
                    errs.append(f"season key {key!r} month {m}: a {v['level']} entry needs a reason")
        for name, prof in self.profiles.items():
            for k in (prof or {}).get("lift_season") or []:
                if k not in self.table:
                    errs.append(f"season profile {name!r}: unknown key {k!r}")
        return errs

    def verdict(self, key: str, month: int, profile: str = "up") -> dict:
        if key not in self.table:
            return {"key": key, "month": int(month), "level": NO_GO, "reason": "season",
                    "note": f"no season entry {key!r}", "label": ""}
        row = dict(self.table[key][int(month)])
        out = {"key": key, "month": int(month), "level": row["level"], "reason": row.get("reason", ""),
               "note": row.get("note", ""), "label": ""}
        prof = self.profiles.get(profile) or {}
        if prof:
            out["label"] = prof.get("label", "")
            if key in (prof.get("lift_season") or []) and out["reason"] == "season" and out["level"] != GO:
                out.update(level=GO, reason="", note=f"{profile}: {out['note']}".strip())
        return out


_SEASON: Optional[Season] = None


def season_table() -> Season:
    global _SEASON
    if _SEASON is None:
        _SEASON = Season.load()
    return _SEASON


def crop_profile(crop: str) -> Optional[dict]:
    path = os.path.join(CROPS_DIR, f"{crop}.yaml")
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except OSError:
        return None


# ------------------------------------------------------------------ context and result
@dataclass
class ReadyContext:
    """Everything the Ready check reads, as plain values (the hook / console fills it from the bus)."""
    rtk: str = "NONE"                         # WP7 /nandi/gnss/fix rtk
    soc: float = 1.0
    capacity_wh: float = EnergyParams().capacity_wh
    coeffs: Optional[Coefficients] = None     # WP7 measured coefficients; None = seed
    disk_free_mb: float = 0.0
    robot_xy: Tuple[float, float] = (0.0, 0.0)
    home_xy: Optional[Tuple[float, float]] = None   # None = field home
    estop: bool = False
    local_link: str = "unknown"               # WP7 /nandi/link/local state: unknown / up / lost
    cloud_link: str = "unknown"
    stability_ok: bool = True
    cross_slope_deg: float = 0.0              # local cross-slope (WP6 foot plane / terrain)
    pitch_deg: float = 0.0
    wind_mps: float = 0.0                     # WP7 /nandi/wind
    gust: bool = False
    weather: Optional[dict] = None            # WP7 weather_stub state; None = WeatherStub(month).state(0)
    month: Optional[int] = None               # None = weather month (default 11, the demo month)
    season_profile: str = "up"                # field "season" (demo_all_fruiting in orchard_demo)
    tools_available: Tuple[str, ...] = ("T2", "T4", "T5")   # in the rack
    tool_mounted: Optional[str] = None
    holding: bool = False
    payload_kg: float = 0.0
    tray_cells_free: int = 8
    cradle_free: bool = True
    light_lux: float = NOMINAL_LUX
    novnc: bool = False
    actors: bool = True
    frames_age_h: Optional[float] = None      # newest TREE 360 / SCAN frames of the target (FRUIT SCOUT)
    marks_n: int = 0                          # MARKs on the job sheet (LEAD-TO-MARKS)
    tag_paired: bool = False                  # FOLLOW-ME UWB / BLE tag
    tote_on_deck: bool = False
    ext_m: float = 0.0                        # crown extension now
    wrist_camera_ok: bool = True
    work_field_clear: bool = True
    fallen: bool = False
    person_clear_confirmed: bool = False
    estop_confirmed: Tuple[bool, bool] = (False, False)   # (local, operator)
    annulus_clear: Dict[str, bool] = field(default_factory=dict)   # tree_id -> LiDAR annulus clear (else registry)
    confirmed: bool = False                   # operator confirmed the degraded warnings

    @classmethod
    def nominal(cls, **kw) -> "ReadyContext":
        """A healthy robot at home: FIXED, full pack, 4 GB free, links up, calm, dry, morning light."""
        base = dict(rtk=FIXED, soc=1.0, disk_free_mb=4096.0, local_link="up", cloud_link="up")
        base.update(kw)
        return cls(**base)

    def weather_state(self) -> dict:
        if self.weather is not None:
            return dict(self.weather)
        return ws.WeatherStub(ws.WeatherScript(month=self.month or 11)).state(0.0)

    def month_now(self) -> int:
        if self.month is not None:
            return int(self.month)
        return int(self.weather_state().get("month", 11))


@dataclass
class ReadyResult:
    preset: str
    level: str
    reasons: List[str]
    items: List[dict]
    adjust: Dict[str, Any]
    estimate: dict
    season: dict
    params: Dict[str, Any] = field(default_factory=dict)
    tree_ids: List[str] = field(default_factory=list)

    @property
    def needs_confirm(self) -> bool:
        return self.level == DEGRADED

    @property
    def chips(self) -> List[str]:
        return [f"{i['reason']}: {i['detail']}" if i["detail"] else i["reason"] for i in self.items
                if i["level"] == INFO]

    def startable(self, confirmed: bool = False) -> bool:
        return self.level == GO or (self.level == DEGRADED and confirmed)

    def to_dict(self) -> dict:
        return {"preset": self.preset, "level": self.level, "reasons": list(self.reasons),
                "needs_confirm": self.needs_confirm, "chips": self.chips, "adjust": dict(self.adjust),
                "items": [dict(i) for i in self.items if i["level"] != GO], "season": dict(self.season),
                "estimate": dict(self.estimate), "tree_ids": list(self.tree_ids)}


class _Items:
    def __init__(self):
        self.rows: List[dict] = []
        self.adjust: Dict[str, Any] = {}

    def add(self, check: str, level: str, reason: str = "", detail: str = "", **values):
        self.rows.append({"check": check, "level": level, "reason": reason, "detail": detail,
                          "values": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in values.items()}})

    def ok(self, check: str, **values):
        self.add(check, GO, **values)

    def merge(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Fold a fallback into ``adjust`` (every gate's fallback is kept, not only the first one). On the same
        key the more conservative value wins: ``crown_walk`` snake < snake_mid < stop_and_go, ``crown_band``
        stowed, ``crown_ext_m`` / ``move_max_m`` the lower. Returns what was applied."""
        applied = {}
        for k, v in changes.items():
            cur = self.adjust.get(k)
            if k in self.adjust and cur != v:
                if k == "crown_walk":
                    v = max(cur, v, key=lambda m: _CROWN_WALK_RANK.get(m, 0))
                elif k == "crown_band":
                    v = "stowed" if "stowed" in (cur, v) else v
                elif k in ("crown_ext_m", "move_max_m") and ps._is_num(cur) and ps._is_num(v):
                    v = min(cur, v)
            self.adjust[k] = v
            applied[k] = v
        return applied


_CROWN_WALK_RANK = {"snake": 0, "snake_mid": 1, "stop_and_go": 2}
# ReadyContext numbers that feed a gate; a non-finite value (sensor dropout, bad bus value) refuses the start
# (fail closed) with sensor_fault instead of reading as "within limits".
# field -> (common checks that read it, preconditions that read it); a field is validated when the preset uses it.
_CROWN_PRE = ("slope_crown_walk", "crown_walk_or_stowed", "crown_gate", "frames_fresh_or_crown_gate")
_SLOPE_PRE = ("slope_walk", "dwell_slope") + _CROWN_PRE
_FINITE_FIELDS = {
    "soc": (("energy",), ()), "capacity_wh": (("energy",), ()), "disk_free_mb": (("disk",), ()),
    "cross_slope_deg": (("slope",), _SLOPE_PRE), "pitch_deg": (("slope",), _SLOPE_PRE),
    "wind_mps": ((), _CROWN_PRE), "payload_kg": ((), ("payload_room",)), "light_lux": ((), ("light",)),
    "ext_m": ((), _CROWN_PRE + ("crown_stowed",)), "robot_xy": (("geofence", "energy"), ()),
}


def _finite(v) -> bool:
    if isinstance(v, (tuple, list)):
        return len(v) == 2 and all(_finite(x) for x in v)
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def nonfinite_fields(preset, ctx: "ReadyContext", common: bool) -> List[str]:
    """The ReadyContext numbers this preset reads that are NaN / inf / not numbers (fail closed: sensor_fault)."""
    relax, wants = set(preset.relax), set(preset.preconditions)
    out = []
    for name, (checks, pres) in _FINITE_FIELDS.items():
        used = (common and any(c not in relax for c in checks)) or bool(wants & set(pres))
        if used and not _finite(getattr(ctx, name)):
            out.append(name)
    return out


# ------------------------------------------------------------------ geometry helpers
def _dist(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _in_poly(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _xy(t: dict) -> Tuple[float, float]:
    return float(t["geo"]["x"]), float(t["geo"]["y"])


def home_of(field_doc: Optional[dict], ctx: ReadyContext) -> Tuple[float, float]:
    if ctx.home_xy is not None:
        return tuple(ctx.home_xy)
    h = (field_doc or {}).get("home") or {}
    return tuple(h.get("xy", (0.0, 0.0)))


def path_points(preset: ps.Preset, trees: Sequence[dict], params: dict, ctx: ReadyContext,
                field_doc: Optional[dict], per_tree: Optional[Sequence[dict]] = None) -> List[Tuple[float, float]]:
    """Points the run visits (robot, each target incl. its orbit annulus, home) for the geofence check.
    ``per_tree`` = resolved params per target (each tree its own orbit radius); default ``params`` for all."""
    pts = [tuple(ctx.robot_xy)]
    for i, t in enumerate(trees):
        x, y = _xy(t)
        q = per_tree[i] if per_tree and i < len(per_tree) else params
        r = float(q.get("r1_m") or 0.0) if preset.mode == "tree360" else 0.0
        if r > 0:
            rr = r + ORBIT_ANNULUS_M
            pts += [(x + rr * math.cos(a), y + rr * math.sin(a)) for a in [k * math.pi / 8 for k in range(16)]]
        else:
            reach = float(t.get("canopy_r_m") or 0.0) + 0.6
            pts += [(x + reach, y), (x - reach, y), (x, y + reach), (x, y - reach)]
    pts.append(home_of(field_doc, ctx))
    return pts


def energy_plan(preset: ps.Preset, est: dict, trees: Sequence[dict], params: dict, ctx: ReadyContext,
                field_doc: Optional[dict]) -> RoutePlan:
    """Remaining route for the WP7 reserve check: transit robot -> first target, the preset's own walking, return
    from the last target to home (x DETOUR_FACTOR)."""
    home = home_of(field_doc, ctx)
    if trees:
        first, last = _xy(trees[0]), _xy(trees[-1])
        transit = _dist(ctx.robot_xy, first) * DETOUR_FACTOR
        ret = _dist(last, home) * DETOUR_FACTOR
    else:
        transit = 0.0
        ret = 0.0 if preset.mode in ("return_home", "dock_unload") else _dist(ctx.robot_xy, home) * DETOUR_FACTOR
    t_work = est["t_s"] + transit / ps.DEFAULT_V_MPS
    return RoutePlan(route_m=transit + est["walk_m"], items=int(est["items"]), t_work_s=t_work,
                     crown_m=est["crown_m"], return_m=ret)


def route_for(preset: ps.Preset, trees: Sequence[dict], ctx: ReadyContext, field_doc: Optional[dict]) -> float:
    """``route_m`` handed to the estimate: rows for SCAN-type runs, the way home / to the dock for RETURN HOME and
    DOCK/UNLOAD, the tree-to-tree walk otherwise."""
    if preset.mode in ("return_home", "dock_unload") and not trees:
        tgt = home_of(field_doc, ctx)
        if preset.mode == "dock_unload":
            tgt = tuple(((field_doc or {}).get("dock") or {}).get("xy", tgt))
        return _dist(ctx.robot_xy, tgt) * DETOUR_FACTOR
    if preset.mode in ("scan_rows",):
        return ps.route_m(trees, field_doc)
    return sum(_dist(_xy(a), _xy(b)) for a, b in zip(trees, trees[1:])) * DETOUR_FACTOR


# ------------------------------------------------------------------ the check
def check(preset: ps.Preset, ctx: ReadyContext, targets: Optional[dict] = None, params: Optional[dict] = None,
          field_doc: Optional[dict] = None, season: Optional[Season] = None,
          table: Optional[st.CrownGateTable] = None) -> ReadyResult:
    """The Ready check of one preset start (MP 0.2). ``targets`` None = tile state without a selection (target
    checks are skipped and an info row asks for targets)."""
    season = season or season_table()
    table = table or st.CrownGateTable.load()
    it = _Items()
    bad_month = None
    try:
        month = ctx.month_now()
        if not 1 <= month <= 12:
            raise ValueError(month)
        weather = ctx.weather_state()
    except (TypeError, ValueError, OverflowError):
        bad_month, month, weather = ctx.month, 0, {}
    empty_season = {"key": preset.season_key(), "month": month, "level": GO, "reason": "", "note": "", "label": ""}

    def done(est=None, p=None, trees=(), season_v=None) -> ReadyResult:
        level = GO
        for r in it.rows:
            if r["level"] in (DEGRADED, NO_GO) and _RANK[r["level"]] > _RANK[level]:
                level = r["level"]
        reasons = [r["reason"] for r in it.rows if r["level"] == NO_GO]
        reasons += [r["reason"] for r in it.rows if r["level"] == DEGRADED and r["reason"] not in reasons]
        return ReadyResult(preset.id, level, list(dict.fromkeys(reasons)), it.rows, it.adjust, est or {},
                           season_v or empty_season, dict(p or {}), [t["tree_id"] for t in trees])

    if not preset.enabled:
        it.add("enabled", NO_GO, preset.disabled_reason, f"{preset.id} is disabled")
        return done()
    relax = set(preset.relax)
    wants = set(preset.preconditions)
    common = "ready_common" in wants or preset.kind in ("recovery", "dock")
    uses_season = "season" not in relax and (common or "season_window" in wants)
    if bad_month is not None and (uses_season or ("weather" not in relax and common)):
        it.add("season", NO_GO, "season", f"month {bad_month!r} not in 1..12", month=str(bad_month))
        return done()
    bad = nonfinite_fields(preset, ctx, common)
    if bad:
        it.add("context", NO_GO, "sensor_fault", f"not a finite number: {bad} (fail closed)", fields=bad)
        return done()

    # ---- targets and params
    trees: List[dict] = []
    if preset.targets == ("none",) or targets is not None:
        try:
            trees = ps.resolve_targets(preset, field_doc, targets)
        except ps.PresetError as exc:
            it.add("targets", NO_GO, exc.reason, exc.detail)
            return done()
    else:
        it.add("targets", INFO, "not_checked", "select targets")
    try:
        per_tree = _resolve_each(preset, ctx, params or {}, trees)
    except ps.PresetError as exc:
        it.add("params", NO_GO, exc.reason, exc.detail, param=exc.param)
        return done()
    p = per_tree[0]

    # ---- estimate
    est: dict = {}
    if trees or preset.targets == ("none",):
        try:
            est = estimate_for(preset, per_tree, trees, route_for(preset, trees, ctx, field_doc))
        except ps.PresetError as exc:
            it.add("estimate", NO_GO, exc.reason, exc.detail)
            return done(p=p, trees=trees)

    # ---- season
    sv = (season.verdict(preset.season_key(trees[0]["crop"] if trees else None), month, ctx.season_profile)
          if bad_month is None else dict(empty_season))
    if uses_season:
        if sv["level"] == NO_GO:
            it.add("season", NO_GO, sv["reason"] or "season", sv["note"], key=sv["key"], month=month)
        elif sv["level"] == DEGRADED:
            it.add("season", DEGRADED, sv["reason"] or "season", sv["note"], key=sv["key"], month=month)
        else:
            it.ok("season", key=sv["key"], month=month)
    if preset.pilot_scope == "demo_only" and ctx.season_profile not in DEMO_PROFILES:
        it.add("pilot_scope", NO_GO, "not_in_mvp", f"{preset.id} runs only in the demo (OQ-01 A)")

    if common:
        _common(preset, ctx, it, relax, est, trees, p, field_doc, weather, per_tree)
    for name in preset.preconditions:
        fn = _PRE.get(name)
        if fn is not None:                   # each gate judges the plan as the earlier gates adjusted it
            fn(preset, ctx, it, trees, {**p, **it.adjust}, field_doc, table, weather)
    if it.adjust:                            # re-estimate with the adjusted params (stowed scan, stop-and-go orbit)
        try:
            per_tree = _resolve_each(preset, ctx, {**(params or {}), **it.adjust}, trees, params)
            p = per_tree[0]
            if trees or preset.targets == ("none",):
                est = estimate_for(preset, per_tree, trees, route_for(preset, trees, ctx, field_doc))
        except ps.PresetError as exc:        # an adjustment the preset cannot run: refuse, never run unadjusted
            it.add("adjust", NO_GO, exc.reason, f"adjusted plan {it.adjust}: {exc.detail}", param=exc.param)
    if est and "energy" not in relax and common:
        plan = energy_plan(preset, est, trees, p, ctx, field_doc)
        ec = ready_check_energy(ctx.soc, plan, ctx.coeffs or Coefficients.seed(), ctx.capacity_wh)
        est = dict(est, energy_wh=round(ec.need_wh, 2), projected_end_soc=round(ec.projected_end_soc, 4),
                   reserve_soc=round(ec.reserve, 4), route_m=round(plan.route_m, 2), return_m=round(plan.return_m, 2))
        if ec.ok:
            it.ok("energy", projected_end_soc=ec.projected_end_soc, reserve=ec.reserve)
        else:
            it.add("energy", NO_GO, ec.reason, ec.detail, projected_end_soc=ec.projected_end_soc,
                   reserve=ec.reserve)
    return done(est, p, trees, sv)


def _resolve_each(preset, ctx, params: dict, trees: Sequence[dict], given: Optional[dict] = None) -> List[dict]:
    """Resolved params per target tree (expression params such as the orbit radius from canopy_r_m differ per
    tree); one entry (no tree) when there are no targets."""
    given = params if given is None else given
    return [_tool_mounted(preset, ctx, preset.resolve(params, t), given) for t in (list(trees) or [None])]


def estimate_for(preset: ps.Preset, per_tree: Sequence[dict], trees: Sequence[dict], route_m: float) -> dict:
    """The preset estimate for these targets. Equal per-tree params: ``estimate(n_trees=n)`` as before. Params
    that differ per tree: the first tree's run + every further tree's marginal share (its n_trees 1 - 0 delta)."""
    if len(trees) <= 1 or all(q == per_tree[0] for q in per_tree[1:]):
        return preset.estimate(per_tree[0], n_trees=max(1, len(trees)), route_m=route_m,
                               tree=trees[0] if trees else None)
    out = dict(preset.estimate(per_tree[0], n_trees=1, route_m=route_m, tree=trees[0]))
    for t, q in zip(trees[1:], per_tree[1:]):
        one = preset.estimate(q, n_trees=1, route_m=0.0, tree=t)
        zero = preset.estimate(q, n_trees=0, route_m=0.0, tree=t)
        for k in ("t_s", "walk_m", "walk_s", "crown_m", "disk_mb", "items"):
            out[k] = out[k] + one[k] - zero[k]
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def _tool_mounted(preset, ctx, p, params):
    """No tool change when the preset's tool is already on the wrist (the CLIPPER stays from mango to guava)."""
    tool = preset.tool_for(p)
    if "tool_change_s" in p and tool and ctx.tool_mounted == tool and "tool_change_s" not in (params or {}):
        return dict(p, tool_change_s=0.0)
    return p


def _common(preset, ctx, it, relax, est, trees, p, field_doc, weather, per_tree=None):
    if "rtk" not in relax:
        if gm.rtk_start_gate(ctx.rtk, preset.id, float_ok=preset.float_ok)[0]:    # one rule (MP 0.2), not a copy
            it.ok("rtk", rtk=ctx.rtk)
        else:
            it.add("rtk", NO_GO, "rtk", f"RTK {ctx.rtk}: FIXED needed to start"
                   + (" (FLOAT is enough)" if preset.float_ok else ""))
    if "disk" not in relax and est:
        need = DISK_FACTOR * est.get("disk_mb", 0.0)
        if ctx.disk_free_mb >= need:
            it.ok("disk", free_mb=ctx.disk_free_mb, need_mb=need)
        else:
            it.add("disk", NO_GO, "disk_low", f"free {ctx.disk_free_mb:.0f} MB < 1.5 x {est.get('disk_mb', 0.0)} MB",
                   free_mb=ctx.disk_free_mb, need_mb=need)
    if "geofence" not in relax:
        fence = (field_doc or {}).get("geofence")
        if fence:
            out = [pt for pt in path_points(preset, trees, p, ctx, field_doc, per_tree) if not _in_poly(pt[0], pt[1], fence)]
            if out:
                it.add("geofence", NO_GO, "geofence", f"{len(out)} path point(s) outside the geofence",
                       first=[round(out[0][0], 2), round(out[0][1], 2)])
            else:
                it.ok("geofence")
        else:
            it.add("geofence", INFO, "geofence", "no geofence in the field file")
    if "estop" not in relax:
        if ctx.estop:
            it.add("estop", NO_GO, "estop", "e-stop engaged")
        else:
            it.ok("estop")
    if "local_link" not in relax:
        if ctx.local_link == "up":
            it.ok("local_link")
        else:
            it.add("local_link", NO_GO, "local_link_lost", f"local safety link {ctx.local_link}")
    if "cloud_link" not in relax:
        if ctx.cloud_link == "lost":
            it.add("cloud_link", NO_GO, "cloud_link_lost", "console heartbeat lost: no new targets")
        elif ctx.cloud_link != "up":
            it.add("cloud_link", INFO, "cloud_link_lost", f"console heartbeat {ctx.cloud_link}")
        else:
            it.ok("cloud_link")
    if "stability" not in relax:
        if ctx.stability_ok:
            it.ok("stability")
        else:
            it.add("stability", NO_GO, "stability", "stability not OK (latched or margin low)")
    if "slope" not in relax:
        tilt = st.slope_tilt_deg(ctx.cross_slope_deg, ctx.pitch_deg)
        if abs(ctx.cross_slope_deg) > preset.slope_limit_deg + 1e-9 or tilt > max(preset.slope_limit_deg,
                                                                                    st.WALK_MAX_TILT_DEG) + 1e-9:
            it.add("slope", NO_GO, "slope", f"local cross-slope {ctx.cross_slope_deg:.1f} deg > {preset.slope_limit_deg} "
                   f"deg ({preset.id})", cross_deg=ctx.cross_slope_deg, tilt_deg=tilt)
        else:
            it.ok("slope", cross_deg=ctx.cross_slope_deg)
    if "weather" not in relax:
        v = ws.preset_verdict(weather, WEATHER_REPRESENTATIVE[preset.weather_class])
        if not v["reasons"]:
            it.ok("weather", temp_c=weather.get("temp_c"))
        for r in v["reasons"]:
            lvl = NO_GO if r["level"] == ws.NO_GO else DEGRADED
            it.add("weather", lvl, r["code"], {"rain": f"rain {weather.get('rain_mm_h', 0)} mm/h",
                                               "wet_foliage": f"wet foliage ({weather.get('wet_cause')})",
                                               "heat": f"{weather.get('temp_c')} C"}.get(r["code"], ""),
                   temp_c=weather.get("temp_c"))
    if "novnc" not in relax:
        if novnc_actor_conflict(ctx.novnc, ctx.actors):
            it.add("novnc", NO_GO, "novnc_actors", "noVNC must be off while actors are enabled")
        else:
            it.ok("novnc")


def novnc_actor_conflict(novnc: bool, actors: bool) -> bool:
    """MVP-SPEC 3.6 / DP-13: noVNC must be off while person actors are composed (a no_go). Pure, so the interlock
    mutation check (tests/test_interlock_mutation.py) can disable it."""
    return bool(novnc) and bool(actors)


# ------------------------------------------------------------------ preconditions
def _registry_present(preset, ctx, it, trees, p, field_doc, table, weather):
    if not (field_doc or {}).get("trees"):
        it.add("registry_present", NO_GO, "registry_mismatch", "no tree registry (config/fields/<world>.json)")
    else:
        it.ok("registry_present")


def _canopy_r_known(preset, ctx, it, trees, p, field_doc, table, weather):
    bad = [t["tree_id"] for t in trees if not ps._is_num(t.get("canopy_r_m"))]
    if bad:
        it.add("canopy_r_known", NO_GO, "registry_mismatch", f"canopy_r_m missing for {bad}")
    else:
        it.ok("canopy_r_known")


def _walking(preset, p) -> bool:
    return any(g["kind"] == "walk" for g in preset.gate_profile(p))


def _slope_walk(preset, ctx, it, trees, p, field_doc, table, weather):
    dec = st.gate_walk_slope(ctx.cross_slope_deg, ctx.pitch_deg)
    if dec:
        it.ok("slope_walk")
    else:
        it.add("slope_walk", NO_GO, "slope", f"walking refused at {dec.values.get('tilt_deg', 0):.1f} deg "
               f"(> {st.WALK_MAX_CROSS_DEG} deg, WP6)", **dec.values)


def _dwell_slope(preset, ctx, it, trees, p, field_doc, table, weather):
    dwell = max([float(g.get("dwell_s", 0.0)) for g in preset.gate_profile(p) if g["kind"] == "dwell"] + [0.0])
    tilt = st.slope_tilt_deg(ctx.cross_slope_deg, ctx.pitch_deg)
    dec = st.gate_dwell(tilt, dwell)
    if dec:
        it.ok("dwell_slope", dwell_s=dwell)
    else:
        it.add("dwell_slope", NO_GO, "slope", f"dwell {dwell:.0f} s refused at {tilt:.1f} deg "
               f"(limit {dec.limits['slope_max_deg']} deg, WP6)", tilt_deg=tilt, dwell_s=dwell)


def _fallback(preset, p, reason: str = "") -> Optional[Dict[str, Any]]:
    """The param change that runs this preset without the failing crown-up walk (None = no fallback): SCAN runs
    stowed; TREE 360 walks its arcs at mid when only the head clearance fails (``snake_mid``), else stowed
    (``stop_and_go``, OS 2.3 rule 7)."""
    if "crown_band" in preset.params and "stowed" in (preset.params["crown_band"].enum or ()):
        return {"crown_band": "stowed", **({"crown_ext_m": 0.0} if "crown_ext_m" in preset.params else {})}
    enum = (preset.params["crown_walk"].enum or ()) if "crown_walk" in preset.params else ()
    if reason == "crown_clearance" and p.get("crown_walk") == "snake" and "snake_mid" in enum:
        return {"crown_walk": "snake_mid"}
    if "stop_and_go" in enum and p.get("crown_walk") != "stop_and_go":
        return {"crown_walk": "stop_and_go"}
    return None


def _crown_walk(preset, ctx, it, trees, p, field_doc, table, weather, name="crown_walk"):
    if "crown_band" in p and p.get("crown_band") == "stowed":
        it.ok(name, mode="stowed")
        return
    walks = [g for g in preset.gate_profile(p) if g["kind"] == "walk" and float(g.get("ext_m", 0.0)) > st.CROWN_UP_EXT_M]
    if not walks:
        it.ok(name)
        return
    tilt = st.slope_tilt_deg(ctx.cross_slope_deg, ctx.pitch_deg)
    bad = None
    for g in walks:
        ext = float(g["ext_m"])
        dec = st.gate_crown_walk(ext, table.margin(ext, tilt), float(g.get("v_mps", 0.0)), ctx.wind_mps, ctx.gust,
                                 tilt, starting=True)
        if not dec:
            bad = dec
            break
    if bad is None:
        it.ok(name)
        return
    fb = _fallback(preset, p)
    if fb is not None:                       # merged with the other gates' fallbacks (annulus, clearance)
        applied = it.merge(fb)
        it.add(name, INFO, "crown_gate", f"{bad.reason}: run with {applied}", **bad.values)
    else:
        it.add(name, NO_GO, bad.reason if bad.reason in ("slope", "wind", "speed", "stability") else "crown_gate",
               f"crown-up walking refused: {bad.reason} (WP6 gate table)", **bad.values)


def _crown_gate(preset, ctx, it, trees, p, field_doc, table, weather):
    """OS 2.3 rule 1 (raise while standing) for the highest raise of the profile, WP6 table margins."""
    raises = [g for g in preset.gate_profile(p) if g["kind"] == "raise" and float(g.get("to_m", 0.0)) > float(g.get("ext_m", 0.0))]
    if not raises or preset.crown == "none":
        it.ok("crown_gate")
        return
    g = max(raises, key=lambda g: float(g["to_m"]))
    tilt = st.slope_tilt_deg(ctx.cross_slope_deg, ctx.pitch_deg)
    hexa = st.table_hexa_margin(float(g["to_m"]), tilt)
    dec = st.gate_raise(float(ctx.ext_m), float(g["to_m"]), hexa, ctx.cross_slope_deg, ctx.wind_mps, ctx.gust,
                        True, 6, ctx.estop, tilt_deg=tilt)
    if dec:
        it.ok("crown_gate", hexa_m=hexa)
        return
    reason = dec.reason if dec.reason in ("slope", "wind", "stability", "estop") else "crown_gate"
    if preset.crown == "optional":
        fb = _fallback(preset, p)
        if fb is not None:
            it.merge(fb)
        it.add("crown_gate", INFO, "crown_gate", f"{dec.reason}: run stowed", **dec.values)
    else:
        it.add("crown_gate", NO_GO, reason, f"crown raise refused: {dec.reason} (WP6 gate table)", **dec.values)


def _alley_clearance(preset, ctx, it, trees, p, field_doc, table, weather):
    """``alley_clear_height_m`` >= planned head height + the 0.10 m swept margin (OS 2.3 rules 1-2) for crown-up
    walking; >= the stowed head + margin for any walk. A crown-up walk that does not fit falls back (stowed scan,
    TREE 360 arcs at mid or stowed); with no fallback the preset is refused ``crown_clearance``."""
    if not trees:
        return
    clear = min(min(t.get("alley_clear_height_m") or [9.9]) for t in trees)
    p = {**p, **it.adjust}                   # judge the plan the gates already adjusted (stowed, stop-and-go)
    stowed_need = ps.STOWED_LENS_M + ps.CROWN_HEAD_HALF_M + CLEARANCE_MARGIN_M
    need = preset.walk_head_m(p) + CLEARANCE_MARGIN_M if p.get("crown_band") != "stowed" else stowed_need
    if clear < stowed_need - 1e-9:
        it.add("alley_clearance", NO_GO, "crown_clearance", f"alley clear height {clear} m < stowed head + margin "
               f"{stowed_need:.2f} m", clear_m=clear)
        return
    if clear >= need - 1e-9:
        it.ok("alley_clearance", clear_m=clear, need_m=need)
        return
    p2 = dict(p)
    for _ in range(2):                       # TREE 360: snake -> snake_mid -> stop_and_go
        fb = _fallback(preset, p2, "crown_clearance")
        if fb is None:
            break
        p2.update(fb)
        need2 = (preset.walk_head_m(p2) + CLEARANCE_MARGIN_M if p2.get("crown_band") != "stowed" else stowed_need)
        if clear >= need2 - 1e-9:
            changes = it.merge({k: v for k, v in p2.items() if p.get(k) != v})
            it.add("alley_clearance", INFO, "crown_clearance", f"alley clear height {clear} m < head + margin "
                   f"{need:.2f} m: run with {changes}", clear_m=clear, need_m=need)
            return
    it.add("alley_clearance", NO_GO, "crown_clearance", f"alley clear height {clear} m < head + margin {need:.2f} m",
           clear_m=clear, need_m=need)


def _annulus(preset, ctx, it, trees, p, field_doc, table, weather):
    blocked = [t["tree_id"] for t in trees if not ctx.annulus_clear.get(t["tree_id"], bool(t.get("orbit_clear")))]
    if not blocked:
        it.ok("annulus")
    elif p.get("arc_mode") == "full":
        it.add("annulus", NO_GO, "orbit_blocked", f"annulus not clear for {blocked} (arc_mode full)")
    else:
        if p.get("arc_mode") != "two_sided":
            it.adjust.setdefault("arc_mode", "two_sided")
        it.add("annulus", INFO, "orbit_blocked", f"two-sided for {blocked}")


def _light(preset, ctx, it, trees, p, field_doc, table, weather):
    need = float(p.get("min_light_lux", 0.0) or 0.0)
    mode = "normal"
    if p.get("dawn_mode"):
        if weather.get("dawn_mode_window"):
            need, mode = float(p.get("dawn_light_lux", need)), "dawn"
        else:
            it.add("dawn_mode", INFO, "low_light", "dawn mode only before 08:00: normal threshold")
    if ctx.light_lux >= need:
        it.ok("light", lux=ctx.light_lux, need=need, mode=mode)
    else:
        it.add("light", NO_GO, "low_light", f"light {ctx.light_lux:.0f} lux < {need:.0f} lux ({mode})",
               lux=ctx.light_lux, need=need)


def _wet(preset, ctx, it, trees, p, field_doc, table, weather):
    if weather.get("wet_foliage"):
        if not any(r["reason"] == "wet_foliage" for r in it.rows):
            it.add("no_wet_foliage", NO_GO, "wet_foliage", f"wet foliage ({weather.get('wet_cause')})")
    else:
        it.ok("no_wet_foliage")


def _flag(attr, reason, detail, check_name):
    def fn(preset, ctx, it, trees, p, field_doc, table, weather):
        if getattr(ctx, attr):
            it.ok(check_name)
        else:
            it.add(check_name, NO_GO, reason, detail)
    return fn


def _person_clear_confirm(preset, ctx, it, trees, p, field_doc, table, weather):
    """MP 8 SELF-RIGHT person-clear + confirmation gate, the live Ready-check copy of
    ``recovery_behaviours.self_right_gate``'s first two rules (the spike tool calls that one)."""
    if ctx.person_clear_confirmed:
        it.ok("person_clear_confirm")
    else:
        it.add("person_clear_confirm", NO_GO, "confirm_required", "person-clear not confirmed")


def _leaves_in_band(preset, ctx, it, trees, p, field_doc, table, weather):
    ban = [t["tree_id"] for t in trees if t.get("crop") == "banana"]
    if ban:
        it.add("crop_leaves_in_band", INFO, "leaf_minimum", f"banana {ban}: in-band dead / drooping blades only")
    else:
        it.ok("crop_leaves_in_band")


def _graft_union(preset, ctx, it, trees, p, field_doc, table, weather):
    bad = [t["tree_id"] for t in trees if not ps._is_num(t.get("graft_union_h_m"))]
    needs = preset.mode == "prune_tree" and "rootstock_sprout" in (p.get("jobs") or [])
    if bad and needs:
        it.add("graft_union_known", NO_GO, "registry_mismatch", f"graft_union_h_m missing for {bad} (rootstock_sprout)")
    elif bad:
        it.add("graft_union_known", INFO, "registry_mismatch", f"graft_union_h_m missing for {bad}: no sprout flags")
    else:
        it.ok("graft_union_known")


def _frames_fresh(preset, ctx, it, trees, p, field_doc, table, weather):
    if p.get("source") == "one_lap":
        _crown_gate(preset, ctx, it, trees, p, field_doc, table, weather)
        return
    if ctx.frames_age_h is not None and ctx.frames_age_h <= float(p.get("max_frame_age_h", 24.0)):
        it.ok("frames_fresh", age_h=ctx.frames_age_h)
        return
    age = "none" if ctx.frames_age_h is None else f"{ctx.frames_age_h:.1f} h old"
    # one lap needs the crown raise gate green (OS 2.3 rule 1 at the high band)
    tilt = st.slope_tilt_deg(ctx.cross_slope_deg, ctx.pitch_deg)
    dec = st.gate_raise(0.0, 1.6, st.table_hexa_margin(1.6, tilt), ctx.cross_slope_deg, ctx.wind_mps, ctx.gust, True,
                        6, ctx.estop, tilt_deg=tilt)
    if dec:
        it.adjust.setdefault("source", "one_lap")
        it.add("frames_fresh", INFO, "crown_gate", f"frames {age}: one lap")
    else:
        it.add("frames_fresh", NO_GO, "crown_gate", f"frames {age} and the crown gate is red ({dec.reason})")


def _tool(preset, ctx, it, trees, p, field_doc, table, weather):
    tool = preset.tool_for(p)
    if tool is None:
        it.ok("tool_in_rack")
    elif ctx.tool_mounted == tool:
        it.ok("tool_in_rack", tool=tool, mounted=True)
    elif ctx.holding:
        it.add("tool_in_rack", NO_GO, "tool_unavailable", f"holding an item: no tool change to {tool}")
    elif tool in ctx.tools_available:
        it.ok("tool_in_rack", tool=tool)
    else:
        it.add("tool_in_rack", NO_GO, "tool_unavailable", f"{tool} not in the rack")


def _payload(preset, ctx, it, trees, p, field_doc, table, weather):
    pl = preset.payload(p)
    if not pl:
        it.ok("payload_room")
        return
    n = int(p.get("expected_n") or 0) or 1
    n = min(n, int(p.get("max_items") or n))
    kg = ctx.payload_kg + n * float(pl.get("per_item_kg", 0.0))
    room = ctx.cradle_free if pl.get("cradle") == "rear" else ctx.tray_cells_free >= 1
    if kg > PAYLOAD_MAX_KG + 1e-9:
        it.add("payload_room", NO_GO, "payload_full", f"payload {kg:.2f} kg > {PAYLOAD_MAX_KG} kg", kg=kg)
    elif not room:
        it.add("payload_room", NO_GO, "payload_full", "no free tray cell / cradle: unload first")
    else:
        it.ok("payload_room", kg=kg)


def _crop_profile(preset, ctx, it, trees, p, field_doc, table, weather):
    bad = sorted({t["crop"] for t in trees if crop_profile(t["crop"]) is None})
    if bad:
        it.add("crop_profile", NO_GO, "registry_mismatch", f"no crop profile for {bad}")
    else:
        it.ok("crop_profile")


def _crop_in_mvp(preset, ctx, it, trees, p, field_doc, table, weather):
    """Project decision OQ-01 A: papaya and banana are scout + MARK only in the MVP; a contact preset on a scout_mark crop
    runs only in a demo season profile (the demo papaya twist, OQ-11 A)."""
    if preset.kind != "contact":
        return
    bad = sorted({t["crop"] for t in trees if (crop_profile(t["crop"]) or {}).get("mvp_scope") == "scout_mark"})
    if bad and ctx.season_profile not in DEMO_PROFILES and preset.id != "prune_banana":
        it.add("crop_in_mvp", NO_GO, "not_in_mvp", f"{bad}: scout + MARK only in the MVP (OQ-01 A)")
    elif bad and preset.id != "prune_banana":
        it.add("crop_in_mvp", INFO, "not_in_mvp", f"{bad}: demo only ({ctx.season_profile})")
    else:
        it.ok("crop_in_mvp")


def _not_bagged(preset, ctx, it, trees, p, field_doc, table, weather):
    bagged = [t["tree_id"] for t in trees if t.get("bagging_date")]
    if bagged:
        it.add("fruit_not_bagged", NO_GO, "bagged", f"fruit bagged on {bagged}: count bags, HARVEST refused")
    else:
        it.ok("fruit_not_bagged")


def _scout_inline(preset, ctx, it, trees, p, field_doc, table, weather):
    if ctx.frames_age_h is not None and ctx.frames_age_h <= 24.0:
        it.ok("scout_fresh_or_inline")
    else:
        it.add("scout_fresh_or_inline", INFO, "not_checked", "no FRUIT SCOUT <= 24 h: inline scout")


def _job_sheet(preset, ctx, it, trees, p, field_doc, table, weather):
    if ctx.marks_n > 0:
        it.ok("job_sheet_exists", marks=ctx.marks_n)
    else:
        it.add("job_sheet_exists", NO_GO, "job_sheet_missing", "no MARK on the job sheet")


def _crown_stowed(preset, ctx, it, trees, p, field_doc, table, weather):
    if ctx.ext_m <= st.CROWN_UP_EXT_M:
        it.ok("crown_stowed")
    else:
        it.add("crown_stowed", NO_GO, "crown_gate", f"crown at {ctx.ext_m:.2f} m: stow first")


def _link_for_move(preset, ctx, it, trees, p, field_doc, table, weather):
    if ctx.local_link == "up":
        it.ok("local_link_for_move")
    else:
        it.adjust.setdefault("move_max_m", 0.0)
        it.add("local_link_for_move", INFO, "local_link_lost", "park in place (no move without the local link)")


def _estop_engaged(preset, ctx, it, trees, p, field_doc, table, weather):
    if ctx.estop:
        it.ok("estop_engaged")
    else:
        it.add("estop_engaged", NO_GO, "estop_not_engaged", "no e-stop to recover from")


def _both_confirm(preset, ctx, it, trees, p, field_doc, table, weather):
    if all(ctx.estop_confirmed) and len(ctx.estop_confirmed) == 2:
        it.ok("both_confirmations")
    else:
        missing = [n for n, ok in zip(("local", "operator"), ctx.estop_confirmed) if not ok]
        it.add("both_confirmations", NO_GO, "confirm_required", f"missing {missing} confirmation")


_PRE = {
    "registry_present": _registry_present,
    "canopy_r_known": _canopy_r_known,
    "slope_walk": _slope_walk,
    "dwell_slope": _dwell_slope,
    "slope_crown_walk": lambda *a: _crown_walk(*a, name="slope_crown_walk"),
    "crown_walk_or_stowed": _crown_walk,
    "crown_gate": _crown_gate,
    "alley_clearance": _alley_clearance,
    "annulus_clear_or_two_sided": _annulus,
    "light": _light,
    "no_wet_foliage": _wet,
    "work_field_clear": _flag("work_field_clear", "person_hold", "a person in the WORK field", "work_field_clear"),
    "wrist_camera_ok": _flag("wrist_camera_ok", "sensor_fault", "wrist camera not OK", "wrist_camera_ok"),
    "crop_leaves_in_band": _leaves_in_band,
    "graft_union_known": _graft_union,
    "frames_fresh_or_crown_gate": _frames_fresh,
    "tool_in_rack": _tool,
    "payload_room": _payload,
    "crop_profile": _crop_profile,
    "scout_fresh_or_inline": _scout_inline,
    "fruit_not_bagged": _not_bagged,
    "crop_in_mvp": _crop_in_mvp,
    "job_sheet_exists": _job_sheet,
    "tag_paired": _flag("tag_paired", "tag_missing", "no paired operator tag (adult, UWB / BLE)", "tag_paired"),
    "tote_on_deck": _flag("tote_on_deck", "tool_unavailable", "no tote on the deck", "tote_on_deck"),
    "crown_stowed": _crown_stowed,
    "local_link_for_move": _link_for_move,
    "estop_engaged": _estop_engaged,
    "both_confirmations": _both_confirm,
    "fallen_latched": _flag("fallen", "not_fallen", "no fall latch", "fallen_latched"),
    "person_clear_confirm": lambda *a: _person_clear_confirm(*a),     # looked up at call time (interlock mutants)
}
assert set(_PRE) | {"ready_common", "season_window"} == set(PRECONDITIONS), "every precondition needs a check"


def check_all(registry: ps.Registry, ctx: ReadyContext, field_doc: Optional[dict] = None,
              targets: Optional[Dict[str, dict]] = None) -> Dict[str, ReadyResult]:
    """Ready check of every preset (console tiles). ``targets`` maps preset id -> its selection."""
    season = season_table()
    table = st.CrownGateTable.load()
    return {p.id: check(p, ctx, (targets or {}).get(p.id), None, field_doc, season, table) for p in registry}
