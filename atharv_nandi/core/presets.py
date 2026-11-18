"""Mission presets registry for Atharv Nandi (WP27; pure Python, no ROS, no MuJoCo).

MISSION-PRESETS 0.1 / 0.8: one YAML file per preset in ``config/presets/<preset>.yaml``. Fields:

    id, mode, version, params (default / min / max / enum / items), preconditions[], steps[], interlocks[],
    outputs[], kpis[], retry, season (key into config/season/up.yaml), console (tile, icon, hi / en labels,
    shown and Advanced params)

plus the WP27 planning blocks every preset carries:

    kind            survey | camera | contact | assist | recovery | dock
    crops[]         crops the preset may run on (registry ``crop``); a tree of another crop is refused ``not_in_mvp``
    targets[]       target forms accepted: tree_ids | row_ids | block_id | block_sample | none
    crown           required | optional | none: whether the preset needs the crown up (optional = it runs stowed
                    with a ``crown_gate`` chip when a crown gate fails)
    weather         exempt | leaf | contact | non_contact: the weather_stub (WP7) class of the preset
    tool            the WP2 tool id (T2 / T4 / T5), or {by: <enum param>, values: {...}}
    payload         {per_item_kg, cells, cradle}: payload-room check of contact presets
    slope_limit_deg the local cross-slope limit of the Ready check (OS 2.3 / 5.2)
    gates[]         the default motion profile, checked against the WP6 gate table (``tests/test_tour_gates.py``):
                    {kind: walk | dwell | raise, ext_m, v_mps, dwell_s, to_m, slope_deg, when}; values may name a
                    param with ``$name`` (its default); ``when: {param: value}`` keeps a segment only for that value
    estimate        ETA / energy / disk model: {fixed_s, per_tree_s, per_item_s, walk_m, walk_s, v_mps, crown_m,
                    items, disk_mb, disk_mb_per_tree}; numbers, ``$param``, expressions over the params and the
                    target context (n_trees, n_items, route_m, canopy_r, pi), or {by: <enum param>, values: {...}}
    ready           {relax: [...]}: common Ready checks this preset skips (recovery and dock presets)
    enabled         false = shown greyed with ``disabled_reason`` (SPOT SPRAY: not_in_mvp)
    pilot_scope     demo_only: runs only in a demo season profile (papaya twist, OQ-01 A / OQ-11 A)

``Preset.resolve(overrides, tree)`` validates the operator's params (refusal reason ``param_invalid`` with the
param name); ``resolve_targets`` turns ``{tree_ids | row_ids | block_id | block_sample}`` into registry trees
(``registry_mismatch`` for an unknown id, ``not_in_mvp`` for a tree of a crop the preset does not run on);
``to_mission_command`` builds the one internal ``cmd/mission`` type (``plan.source`` in console / zone_alert / edge_hub
until the command-naming question is settled).
"""
import ast
import glob
import math
import operator
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from atharv_nandi import paths

ROOT = paths.REPO
PRESETS_DIR = os.path.join(ROOT, "config", "presets")

SCHEMA = "nandi.preset/1"
REQUIRED = ("id", "mode", "version", "params", "preconditions", "steps", "interlocks", "outputs", "kpis",
            "retry", "season", "console")
OPTIONAL = ("schema", "kind", "crops", "targets", "slope_limit_deg", "gates", "estimate", "ready", "enabled",
            "disabled_reason", "title", "method", "pilot_scope", "notes", "float_ok", "crown", "weather", "tool",
            "payload")
KINDS = ("survey", "camera", "contact", "assist", "recovery", "dock")
CROPS = ("mango", "guava", "papaya", "banana")
TARGET_FORMS = ("tree_ids", "row_ids", "block_id", "block_sample", "none")
GATE_KINDS = ("walk", "dwell", "raise")
GATE_KEYS = ("kind", "ext_m", "v_mps", "dwell_s", "to_m", "slope_deg", "when")
CROWN_USE = ("required", "optional", "none")
WEATHER_CLASSES = ("exempt", "leaf", "contact", "non_contact")
TOOL_IDS = ("T2", "T4", "T5")                   # WP2 registry: CLIPPER, TWIST, PRUNER
PILOT_SCOPES = ("mvp", "demo_only")
SOURCES = ("console", "zone_alert", "edge_hub")       # plan.source until the command naming is settled
ESTIMATE_KEYS = ("fixed_s", "per_tree_s", "per_item_s", "walk_m", "walk_s", "v_mps", "crown_m", "items", "disk_mb",
                 "disk_mb_per_tree")
DEFAULT_V_MPS = 0.18                            # stowed walking (speed_scale 0.5, measured, probe)
RELAXABLE = ("energy", "rtk", "estop", "local_link", "cloud_link", "stability", "season", "weather", "novnc", "geofence",
             "disk", "slope")

# Reason codes this module returns (MISSION-PRESETS 0.5 plus the WP27 extensions listed in the WP27 contract notes)
R_PARAM = "param_invalid"
R_PRESET = "preset_unknown"
R_REGISTRY = "registry_mismatch"
R_NOT_IN_MVP = "not_in_mvp"

STOWED_LENS_M = 1.10      # OS 2.3: lens height with the crown stowed
CROWN_HEAD_HALF_M = 0.06  # crown_head box half-size (head top = lens + 0.06)
CROWN_RATE_MPS = 0.2      # OS 2.3: mast rate limit (s per metre of travel = 5)


class PresetError(Exception):
    """A refused preset request: ``reason`` is a MISSION-PRESETS 0.5 code (or a WP27 extension)."""

    def __init__(self, reason: str, detail: str = "", param: str = ""):
        super().__init__(detail or reason)
        self.reason, self.detail, self.param = reason, detail or reason, param

    def to_dict(self) -> dict:
        out = {"reason": self.reason, "detail": self.detail}
        if self.param:
            out["param"] = self.param
        return out


# ------------------------------------------------------------------ safe expressions
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Mod: operator.mod}           # no Pow: an unbounded 9**9**9 would hang the estimate (and the tiles)
_FUN = {"min": min, "max": max, "abs": abs, "round": round, "sqrt": math.sqrt, "ceil": math.ceil,
        "floor": math.floor, "clamp": lambda x, lo, hi: max(lo, min(hi, x))}
_CONST = {"pi": math.pi}


class Expr:
    """Arithmetic over names, numbers and min / max / abs / round / sqrt / ceil / floor / clamp. No attribute
    access, no calls of anything else: a preset file cannot run code."""

    def __init__(self, text: str):
        self.text = str(text)
        try:
            self.tree = ast.parse(self.text, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"expression {self.text!r}: {exc.msg}") from None
        self.names = set()
        self._check(self.tree.body)

    def _check(self, n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return
        if isinstance(n, ast.Name):
            if n.id not in _CONST:
                self.names.add(n.id)
            return
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            self._check(n.left)
            self._check(n.right)
            return
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            self._check(n.operand)
            return
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN and not n.keywords):
            for a in n.args:
                self._check(a)
            return
        raise ValueError(f"expression {self.text!r}: {type(n).__name__} not allowed")

    def eval(self, env: Dict[str, float]) -> float:
        def ev(n):
            if isinstance(n, ast.Constant):
                return n.value
            if isinstance(n, ast.Name):
                if n.id in _CONST:
                    return _CONST[n.id]
                v = lookup(env, n.id)
                if v is None:
                    raise KeyError(n.id)
                return v
            if isinstance(n, ast.BinOp):
                return _BIN[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, ast.UnaryOp):
                v = ev(n.operand)
                return -v if isinstance(n.op, ast.USub) else v
            return _FUN[n.func.id](*[ev(a) for a in n.args])
        try:
            v = ev(self.tree.body)
        except (ZeroDivisionError, OverflowError, TypeError) as exc:
            raise ValueError(f"expression {self.text!r}: {exc}") from None
        if not (isinstance(v, (int, float)) and math.isfinite(v)):
            raise ValueError(f"expression {self.text!r} is not a finite number")
        return float(v)


def lookup(env: Dict[str, Any], name: str):
    """A name, or its unit-suffixed param (``r1`` finds ``r1_m``, ``canopy_r`` finds ``canopy_r_m``)."""
    if name in env:
        return env[name]
    for suf in ("_m", "_s", "_mps", "_deg"):
        if name + suf in env:
            return env[name + suf]
    return None


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ------------------------------------------------------------------ params
@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    enum: Optional[Tuple[Any, ...]] = None
    items: Optional[Tuple[Any, ...]] = None   # allowed members of a list-valued param
    expr: Optional[Expr] = None               # default given as an expression ("clamp(canopy_r+standoff_m,1.2,3.0)")
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d) -> "ParamSpec":
        if not isinstance(d, dict) or "default" not in d:
            raise ValueError(f"param {name!r}: needs a mapping with 'default'")
        default = d["default"]
        expr = None
        if isinstance(default, str) and any(c in default for c in "()+-*/") and not d.get("enum"):
            expr = Expr(default)
        lo, hi = d.get("min"), d.get("max")
        for k, v in (("min", lo), ("max", hi)):
            if v is not None and not _is_num(v):
                raise ValueError(f"param {name!r}: {k} must be a number")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"param {name!r}: min > max")
        enum = tuple(d["enum"]) if d.get("enum") is not None else None
        items = tuple(d["items"]) if d.get("items") is not None else None
        p = cls(name, default, lo, hi, enum, items, expr, dict(d))
        if expr is None:
            err = p.check(default)
            if err:
                raise ValueError(f"param {name!r}: default {default!r} {err}")
        return p

    @property
    def numeric(self) -> bool:
        return self.expr is not None or _is_num(self.default)

    def check(self, v) -> str:
        """'' when v is a valid value of this param, else what is wrong."""
        if self.enum is not None:
            return "" if v in self.enum else f"not in {list(self.enum)}"
        if self.items is not None:
            if not isinstance(v, list) or not v:
                return "must be a non-empty list"
            bad = [x for x in v if x not in self.items]
            return f"has {bad} not in {list(self.items)}" if bad else ""
        if self.numeric:
            if not _is_num(v):
                return "must be a finite number"
            if self.min is not None and v < self.min - 1e-12:
                return f"below min {self.min}"
            if self.max is not None and v > self.max + 1e-12:
                return f"above max {self.max}"
            return ""
        if isinstance(self.default, bool):
            return "" if isinstance(v, bool) else "must be true or false"
        if isinstance(self.default, dict):
            return "" if isinstance(v, dict) else "must be a mapping"
        if isinstance(self.default, list):
            return "" if isinstance(v, list) else "must be a list"
        if isinstance(self.default, str):
            return "" if isinstance(v, str) and v.strip() else "must be a non-empty string"
        if self.default is None:
            return ""
        return "" if type(v) is type(self.default) else f"must be {type(self.default).__name__}"


# ------------------------------------------------------------------ preset
@dataclass
class Preset:
    id: str
    mode: Optional[str]
    version: int
    params: Dict[str, ParamSpec]
    doc: Dict[str, Any]
    path: str = ""

    # ---- simple accessors
    @property
    def kind(self) -> str:
        return self.doc.get("kind", "survey")

    @property
    def enabled(self) -> bool:
        return bool(self.doc.get("enabled", True))

    @property
    def disabled_reason(self) -> str:
        return self.doc.get("disabled_reason", R_NOT_IN_MVP) if not self.enabled else ""

    @property
    def crops(self) -> Tuple[str, ...]:
        return tuple(self.doc.get("crops") or CROPS)

    @property
    def targets(self) -> Tuple[str, ...]:
        return tuple(self.doc.get("targets") or ("tree_ids",))

    @property
    def preconditions(self) -> Tuple[str, ...]:
        return tuple(self.doc.get("preconditions") or ())

    @property
    def interlocks(self) -> Tuple[str, ...]:
        return tuple(self.doc.get("interlocks") or ())

    @property
    def slope_limit_deg(self) -> float:
        return float(self.doc.get("slope_limit_deg", 5.0))

    @property
    def relax(self) -> Tuple[str, ...]:
        return tuple((self.doc.get("ready") or {}).get("relax") or ())

    @property
    def console(self) -> dict:
        return dict(self.doc.get("console") or {})

    def season_key(self, crop: Optional[str] = None) -> str:
        key = str(self.doc.get("season", self.id))
        if "{crop}" in key:
            key = key.replace("{crop}", crop or (self.crops[0] if self.crops else ""))
        return key

    @property
    def crown(self) -> str:
        """required | optional | none (default: required when the default profile raises the crown)."""
        c = self.doc.get("crown")
        if c in CROWN_USE:
            return c
        return "required" if self.uses_crown() else "none"

    @property
    def weather_class(self) -> str:
        w = self.doc.get("weather")
        if w in WEATHER_CLASSES:
            return w
        return {"recovery": "exempt", "dock": "exempt", "contact": "contact", "assist": "contact"}.get(
            self.kind, "non_contact")

    @property
    def pilot_scope(self) -> str:
        return str(self.doc.get("pilot_scope", "mvp"))

    @property
    def float_ok(self) -> bool:
        return bool(self.doc.get("float_ok", False))

    def tool_for(self, params: Optional[dict] = None) -> Optional[str]:
        """The WP2 tool id this run needs (None = no tool)."""
        t = self.doc.get("tool")
        if isinstance(t, dict):
            p = params if params is not None else self.resolve({})
            return (t.get("values") or {}).get(p.get(t.get("by")))
        return t

    def payload(self, params: Optional[dict] = None) -> dict:
        """{per_item_kg, cells, cradle} with ``$param`` values filled in ({} = no payload)."""
        pl = dict(self.doc.get("payload") or {})
        if not pl:
            return {}
        p = params if params is not None else self.resolve({})
        return {k: self.subst(v, p) for k, v in pl.items()}

    def uses_crown(self, params: Optional[dict] = None) -> bool:
        return any(float(g.get("ext_m", 0.0) or 0.0) > 0.02 or float(g.get("to_m", 0.0) or 0.0) > 0.02
                   for g in self.gate_profile(params))

    def walks_crown_up(self, params: Optional[dict] = None) -> bool:
        return any(g["kind"] == "walk" and float(g.get("ext_m", 0.0)) > 0.02 for g in self.gate_profile(params))

    def dwells(self, params: Optional[dict] = None) -> bool:
        return any(g["kind"] == "dwell" for g in self.gate_profile(params))

    def max_head_m(self, params: Optional[dict] = None) -> float:
        """Highest planned head top (m above ground) of the default profile."""
        ext = max([float(g.get("ext_m", 0.0) or 0.0) for g in self.gate_profile(params)]
                  + [float(g.get("to_m", 0.0) or 0.0) for g in self.gate_profile(params)] + [0.0])
        return STOWED_LENS_M + ext + CROWN_HEAD_HALF_M

    def walk_head_m(self, params: Optional[dict] = None) -> float:
        """Head top while walking: the highest crown-up walk segment, else the stowed head."""
        ext = max([float(g.get("ext_m", 0.0)) for g in self.gate_profile(params) if g["kind"] == "walk"] + [0.0])
        return STOWED_LENS_M + ext + CROWN_HEAD_HALF_M

    # ---- params
    def defaults(self, tree: Optional[dict] = None) -> Dict[str, Any]:
        return self.resolve({}, tree)

    def resolve(self, overrides: Optional[dict] = None, tree: Optional[dict] = None) -> Dict[str, Any]:
        """Defaults + operator overrides, validated. Expression defaults are evaluated against the other params
        and the tree context (``canopy_r`` = ``canopy_r_m`` etc.); without a tree they stay unresolved (None)
        unless overridden. Raises PresetError('param_invalid', ..., param)."""
        overrides = dict(overrides or {})
        for k in overrides:
            if k not in self.params:
                raise PresetError(R_PARAM, f"{self.id}: unknown param {k!r}", k)
        out: Dict[str, Any] = {}
        for name, spec in self.params.items():
            if name in overrides:
                v = overrides[name]
                if spec.numeric and isinstance(v, bool):
                    raise PresetError(R_PARAM, f"{self.id}.{name}: must be a finite number", name)
                if spec.numeric and _is_num(v) and isinstance(v, int):
                    v = float(v)
                err = spec.check(v)
                if err:
                    raise PresetError(R_PARAM, f"{self.id}.{name} = {v!r}: {err}", name)
                out[name] = v
            elif spec.expr is None:
                out[name] = spec.default
        env = dict(tree_context(tree))
        env.update({k: v for k, v in out.items() if _is_num(v)})
        pending = [n for n, s in self.params.items() if n not in out]
        for _ in range(len(pending) + 1):          # expressions may chain (r2 from r1)
            progressed = False
            for name in list(pending):
                spec = self.params[name]
                try:
                    v = spec.expr.eval(env)
                except KeyError:
                    continue
                except ValueError as exc:
                    raise PresetError(R_PARAM, f"{self.id}: param {name!r}: {exc}", name) from None
                if spec.min is not None:
                    v = max(v, spec.min)
                if spec.max is not None:
                    v = min(v, spec.max)
                out[name] = env[name] = round(v, 6)
                pending.remove(name)
                progressed = True
            if not progressed:
                break
        for name in pending:
            out[name] = None
        return {k: out[k] for k in self.params}

    def subst(self, v, params: Optional[dict] = None):
        """``$name`` -> the param value (default when params is None)."""
        if isinstance(v, str) and v.startswith("$"):
            p = params if params is not None else self.resolve({})
            name = v[1:]
            if name not in p:
                raise KeyError(f"{self.id}: gate value {v!r} names no param")
            return p[name]
        return v

    def gate_profile(self, params: Optional[dict] = None) -> List[dict]:
        """The motion profile for these params (defaults when None): ``$param`` values filled in, segments whose
        ``when`` does not match the params dropped."""
        p = params if params is not None else self.resolve({})
        out = []
        for g in self.doc.get("gates") or []:
            when = g.get("when") or {}
            if any(p.get(k) != v for k, v in when.items()):
                continue
            gg = {k: self.subst(v, p) for k, v in g.items() if k != "when"}
            gg.setdefault("ext_m", 0.0)
            out.append(gg)
        return out

    # ---- estimate
    def estimate(self, params: Optional[dict] = None, n_trees: int = 1, n_items: Optional[int] = None,
                 route_m: float = 0.0, tree: Optional[dict] = None) -> dict:
        """ETA / energy / disk of one run of this preset (transit to the first target excluded):
        {t_s, walk_m, walk_s, items, crown_m, disk_mb}. ``route_m`` is the planned route (rows, laps) from the
        caller. Walking time is ``walk_s`` when the preset gives it, else ``walk_m / v_mps``. An expression that
        needs a tree number the caller did not give (``canopy_r`` without a tree) raises PresetError(param_invalid)."""
        p = params if params is not None else self.resolve({}, tree)
        est = self.doc.get("estimate") or {}
        env = dict(tree_context(tree))
        env.update({k: v for k, v in p.items() if _is_num(v)})
        env.update(n_trees=float(n_trees), route_m=float(route_m))
        env["n_items"] = float(n_items if n_items is not None else self._num(est.get("items", 0), env, p, self.id))
        val = {}
        for k in ESTIMATE_KEYS:
            if k == "walk_s" and "walk_s" not in est:
                continue
            default = DEFAULT_V_MPS if k == "v_mps" else 0.0
            val[k] = self._num(est.get(k, default), env, p, self.id)
        v = val["v_mps"] if val["v_mps"] > 0 else DEFAULT_V_MPS
        walk_s = val["walk_s"] if "walk_s" in val else val["walk_m"] / v
        items = int(round(env["n_items"]))
        t = val["fixed_s"] + val["per_tree_s"] * n_trees + val["per_item_s"] * items + walk_s
        return {"t_s": round(t, 2), "walk_m": round(val["walk_m"], 3), "walk_s": round(walk_s, 2), "items": items,
                "crown_m": round(val["crown_m"], 3),
                "disk_mb": round(val["disk_mb"] + val["disk_mb_per_tree"] * n_trees, 2)}

    @staticmethod
    def _num(v, env, params: Optional[dict] = None, pid: str = "") -> float:
        if isinstance(v, dict):                      # {by: <enum param>, values: {value: number | expr}}
            key = (params or {}).get(v.get("by"))
            if key not in (v.get("values") or {}):
                raise PresetError(R_PARAM, f"{pid}: estimate has no value for {v.get('by')} = {key!r}",
                                  str(v.get("by")))
            return Preset._num(v["values"][key], env, params, pid)
        if _is_num(v):
            return float(v)
        if isinstance(v, str):
            name = v[1:] if v.startswith("$") else None
            if name is not None:
                r = lookup(env, name)
                if not _is_num(r):
                    raise PresetError(R_PARAM, f"{pid}: estimate needs {name!r}", name)
                return float(r)
            try:
                return Expr(v).eval(env)
            except KeyError as exc:
                raise PresetError(R_PARAM, f"{pid}: estimate needs {exc.args[0]!r} (give the target tree)",
                                  str(exc.args[0])) from None
            except ValueError as exc:
                raise PresetError(R_PARAM, f"{pid}: {exc}") from None
        return 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "mode": self.mode, "version": self.version, "kind": self.kind,
                "enabled": self.enabled, "season": self.doc.get("season"), "crops": list(self.crops)}


def tree_context(tree: Optional[dict]) -> Dict[str, float]:
    """Numbers of a registry tree a param expression may use (both ``canopy_r`` and ``canopy_r_m``)."""
    if not tree:
        return {}
    out = {}
    for k in ("height_m", "canopy_r_m", "trunk_r_m", "graft_union_h_m", "basin_r_m", "orbit_standoff_m"):
        if _is_num(tree.get(k)):
            out[k] = float(tree[k])
    return out


# ------------------------------------------------------------------ validation and loading
def validate_doc(doc: Any, modes: Sequence[str], preconditions: Optional[Sequence[str]] = None,
                 season_keys: Optional[Sequence[str]] = None) -> List[str]:
    """Every problem of a preset document ([] = valid)."""
    if not isinstance(doc, dict):
        return ["preset file is not a mapping"]
    errs = []
    pid = doc.get("id", "?")
    for k in REQUIRED:
        if k not in doc:
            errs.append(f"{pid}: missing field {k!r}")
    for k in doc:
        if k not in REQUIRED and k not in OPTIONAL:
            errs.append(f"{pid}: unknown field {k!r}")
    if doc.get("schema", SCHEMA) != SCHEMA:
        errs.append(f"{pid}: schema {doc.get('schema')!r} != {SCHEMA}")
    if not isinstance(doc.get("id"), str) or not doc.get("id", "").replace("_", "").isalnum():
        errs.append(f"{pid}: id must be a snake_case string")
    enabled = doc.get("enabled", True)
    if not isinstance(enabled, bool):
        errs.append(f"{pid}: enabled must be true / false")
    mode = doc.get("mode")
    if enabled is not False and mode not in modes:
        errs.append(f"{pid}: mode {mode!r} not in mission_logic.MODES")
    if enabled is False and mode is not None and mode not in modes:
        errs.append(f"{pid}: mode {mode!r} not in mission_logic.MODES")
    if not (isinstance(doc.get("version"), int) and not isinstance(doc.get("version"), bool)
            and doc.get("version") >= 1):
        errs.append(f"{pid}: version must be an integer >= 1")
    if doc.get("kind", "survey") not in KINDS:
        errs.append(f"{pid}: kind {doc.get('kind')!r} not in {KINDS}")
    for c in doc.get("crops") or []:
        if c not in CROPS:
            errs.append(f"{pid}: crop {c!r} not in {CROPS}")
    for t in doc.get("targets") or []:
        if t not in TARGET_FORMS:
            errs.append(f"{pid}: target form {t!r} not in {TARGET_FORMS}")
    params = doc.get("params")
    specs: Dict[str, ParamSpec] = {}
    if not isinstance(params, dict):
        errs.append(f"{pid}: params must be a mapping")
    else:
        for name, d in params.items():
            try:
                specs[name] = ParamSpec.from_dict(name, d)
            except ValueError as exc:
                errs.append(f"{pid}: {exc}")
        known = set(specs) | {"canopy_r", "canopy_r_m", "height", "height_m", "trunk_r", "trunk_r_m",
                              "graft_union_h", "graft_union_h_m", "basin_r", "basin_r_m", "orbit_standoff",
                              "orbit_standoff_m"}
        for name, s in specs.items():
            if s.expr is not None:
                bad = [n for n in s.expr.names if lookup({k: 1 for k in known}, n) is None]
                if bad:
                    errs.append(f"{pid}: param {name!r} expression names unknown {bad}")
    for k in ("preconditions", "steps", "interlocks", "outputs", "kpis"):
        v = doc.get(k)
        if k in doc and not (isinstance(v, list) and all(isinstance(x, str) and x for x in v)):
            errs.append(f"{pid}: {k} must be a list of strings")
    if isinstance(doc.get("steps"), list) and not doc["steps"]:
        errs.append(f"{pid}: steps must not be empty")
    if preconditions is not None:
        for pc in doc.get("preconditions") or []:
            if pc not in preconditions:
                errs.append(f"{pid}: precondition {pc!r} has no Ready check")
    if "retry" in doc and not isinstance(doc.get("retry"), dict):
        errs.append(f"{pid}: retry must be a mapping")
    season = doc.get("season")
    if not isinstance(season, str) or not season:
        errs.append(f"{pid}: season must be a key string")
    elif season_keys is not None:
        keys = ([season.replace("{crop}", c) for c in (doc.get("crops") or CROPS)] if "{crop}" in season
                else [season])
        for key in keys:
            if key not in season_keys:
                errs.append(f"{pid}: season key {key!r} not in config/season/up.yaml")
    con = doc.get("console")
    if not isinstance(con, dict):
        errs.append(f"{pid}: console must be a mapping")
    else:
        for k in ("tile", "hi", "en", "icon"):
            if not isinstance(con.get(k), str) or not con.get(k):
                errs.append(f"{pid}: console.{k} missing")
        for k in ("show", "advanced"):
            for n in con.get(k) or []:
                if isinstance(params, dict) and n not in params:
                    errs.append(f"{pid}: console.{k} names unknown param {n!r}")
    sl = doc.get("slope_limit_deg", 5.0)
    if not _is_num(sl) or not 0 <= sl <= 5.0:
        errs.append(f"{pid}: slope_limit_deg must be in [0, 5] (OS 5.2: > 5 deg refused)")
    def enum_of(name):
        d = params.get(name) if isinstance(params, dict) else None
        return d.get("enum") if isinstance(d, dict) else None

    def check_value(where, v, exprs_ok):
        if isinstance(v, dict):
            by, values = v.get("by"), v.get("values")
            if set(v) != {"by", "values"} or not isinstance(values, dict) or not values:
                errs.append(f"{pid}: {where}: a mapping needs exactly {{by, values}}")
                return
            enum = enum_of(by)
            if enum is None:
                errs.append(f"{pid}: {where}: by {by!r} is not an enum param")
                return
            missing = [e for e in enum if e not in values]
            extra = [k for k in values if k not in enum]
            if missing or extra:
                errs.append(f"{pid}: {where}: values must cover {list(enum)} exactly (missing {missing}, extra {extra})")
            for k, x in values.items():
                check_value(f"{where}.{k}", x, exprs_ok)
            return
        if isinstance(v, str) and v.startswith("$"):
            if not isinstance(params, dict) or v[1:] not in params:
                errs.append(f"{pid}: {where} {v!r} names no param")
            return
        if isinstance(v, str) and exprs_ok:
            try:
                Expr(v)
            except ValueError as exc:
                errs.append(f"{pid}: {where}: {exc}")
            return
        if not _is_num(v):
            errs.append(f"{pid}: {where} = {v!r} must be a number{' or an expression' if exprs_ok else ''} or $param")

    for g in doc.get("gates") or []:
        if not isinstance(g, dict) or g.get("kind") not in GATE_KINDS:
            errs.append(f"{pid}: gate {g!r}: kind not in {GATE_KINDS}")
            continue
        for k, v in g.items():
            if k == "kind":
                continue
            if k not in GATE_KEYS:
                errs.append(f"{pid}: gate key {k!r} not in {GATE_KEYS}")
            elif k == "when":
                if not isinstance(v, dict) or not v:
                    errs.append(f"{pid}: gate when must be a non-empty mapping")
                    continue
                for wk, wv in v.items():
                    enum = enum_of(wk)
                    if enum is None or wv not in enum:
                        errs.append(f"{pid}: gate when {wk}: {wv!r} is not a value of an enum param")
            else:
                check_value(f"gate {k}", v, exprs_ok=False)
        if g.get("kind") == "walk" and "v_mps" not in g:
            errs.append(f"{pid}: walk gate needs v_mps")
        if g.get("kind") == "dwell" and "dwell_s" not in g:
            errs.append(f"{pid}: dwell gate needs dwell_s")
        if g.get("kind") == "raise" and "to_m" not in g:
            errs.append(f"{pid}: raise gate needs to_m")
    est = doc.get("estimate") or {}
    if not isinstance(est, dict):
        errs.append(f"{pid}: estimate must be a mapping")
    else:
        for k, v in est.items():
            if k not in ESTIMATE_KEYS:
                errs.append(f"{pid}: estimate key {k!r} not in {ESTIMATE_KEYS}")
            else:
                check_value(f"estimate {k}", v, exprs_ok=True)
    if doc.get("crown") is not None and doc.get("crown") not in CROWN_USE:
        errs.append(f"{pid}: crown {doc.get('crown')!r} not in {CROWN_USE}")
    if doc.get("weather") is not None and doc.get("weather") not in WEATHER_CLASSES:
        errs.append(f"{pid}: weather {doc.get('weather')!r} not in {WEATHER_CLASSES}")
    if doc.get("pilot_scope") is not None and doc.get("pilot_scope") not in PILOT_SCOPES:
        errs.append(f"{pid}: pilot_scope {doc.get('pilot_scope')!r} not in {PILOT_SCOPES}")
    tool = doc.get("tool")
    if tool is not None:
        if isinstance(tool, dict):
            enum = enum_of(tool.get("by"))
            vals = tool.get("values") if isinstance(tool.get("values"), dict) else {}
            if enum is None or sorted(map(str, vals)) != sorted(map(str, enum)):
                errs.append(f"{pid}: tool {{by, values}} must map every value of an enum param")
            bad = [t for t in vals.values() if t not in TOOL_IDS]
            if bad:
                errs.append(f"{pid}: tool ids {bad} not in {TOOL_IDS}")
        elif tool not in TOOL_IDS:
            errs.append(f"{pid}: tool {tool!r} not in {TOOL_IDS}")
    pl = doc.get("payload")
    if pl is not None:
        if not isinstance(pl, dict) or "per_item_kg" not in pl:
            errs.append(f"{pid}: payload needs per_item_kg")
        else:
            for k, v in pl.items():
                if k == "cradle":
                    if v not in ("rear",):
                        errs.append(f"{pid}: payload cradle {v!r} not in ['rear']")
                elif k in ("per_item_kg", "cells"):
                    check_value(f"payload {k}", v, exprs_ok=False)
                else:
                    errs.append(f"{pid}: payload key {k!r} not in [per_item_kg, cells, cradle]")
    if not isinstance(doc.get("float_ok", False), bool):
        errs.append(f"{pid}: float_ok must be true / false")
    for r in (doc.get("ready") or {}).get("relax") or []:
        if r not in RELAXABLE:
            errs.append(f"{pid}: ready.relax {r!r} not in {RELAXABLE}")
    return errs


def from_doc(doc: dict, path: str = "") -> Preset:
    params = {n: ParamSpec.from_dict(n, d) for n, d in (doc.get("params") or {}).items()}
    return Preset(doc["id"], doc.get("mode"), int(doc["version"]), params, doc, path)


class Registry:
    """All presets of a directory, validated on load. ``errors`` lists every invalid file (never loaded)."""

    def __init__(self, presets: Dict[str, Preset], errors: List[str], directory: str = ""):
        self.presets, self.errors, self.directory = presets, errors, directory

    @classmethod
    def load(cls, directory: Optional[str] = None, season_keys: Optional[Sequence[str]] = None) -> "Registry":
        from atharv_nandi.core import mission_logic as ml
        from atharv_nandi.core import ready_check as rc
        directory = directory or PRESETS_DIR
        if season_keys is None:
            try:
                season_keys = list(rc.Season.load().keys)
            except (OSError, ValueError):
                season_keys = None
        presets, errors = {}, []
        for path in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh)
            except (OSError, yaml.YAMLError) as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")
                continue
            errs = validate_doc(doc, ml.MODES, rc.PRECONDITIONS, season_keys)
            if isinstance(doc, dict) and doc.get("id") and os.path.basename(path) != f"{doc['id']}.yaml":
                errs.append(f"{os.path.basename(path)}: file name must be <id>.yaml")
            if isinstance(doc, dict) and doc.get("id") in presets:
                errs.append(f"{doc['id']}: duplicate id")
            if errs:
                errors.extend(errs)
                continue
            presets[doc["id"]] = from_doc(doc, path)
        return cls(presets, errors, directory)

    def __contains__(self, pid: str) -> bool:
        return pid in self.presets

    def __iter__(self):
        return iter(self.presets.values())

    def __len__(self) -> int:
        return len(self.presets)

    def get(self, pid: str) -> Preset:
        if pid not in self.presets:
            raise PresetError(R_PRESET, f"preset {pid!r} not in the registry")
        return self.presets[pid]

    def by_mode(self, mode: str) -> List[Preset]:
        return [p for p in self.presets.values() if p.mode == mode]


# ------------------------------------------------------------------ targets
def resolve_targets(preset: Preset, field_doc: Optional[dict], targets: Optional[dict]) -> List[dict]:
    """``{tree_ids | row_ids | block_id | block_sample}`` -> registry trees (in the given / row order).
    Presets with target form ``none`` (recovery, dock) take no trees. Raises PresetError(registry_mismatch)."""
    targets = dict(targets or {})
    forms = preset.targets
    if forms == ("none",):
        if any(v for v in targets.values()):
            raise PresetError(R_PARAM, f"{preset.id} takes no targets", "targets")
        return []
    if field_doc is None or not field_doc.get("trees"):
        raise PresetError(R_REGISTRY, "no tree registry (config/fields/<world>.json)")
    given = [k for k, v in targets.items() if v not in (None, [], "")]
    if len(given) != 1:
        raise PresetError(R_PARAM, f"{preset.id}: give exactly one of {list(forms)}", "targets")
    form = given[0]
    if form not in forms:
        raise PresetError(R_PARAM, f"{preset.id}: target form {form!r} not in {list(forms)}", "targets")
    trees = field_doc["trees"]
    by_id = {t["tree_id"]: t for t in trees}
    by_tag = {t.get("physical_tag"): t for t in trees if t.get("physical_tag")}
    val = targets[form]
    if form == "tree_ids":
        ids = [val] if isinstance(val, str) else list(val)
        out = []
        for i in ids:
            t = by_id.get(i) or by_tag.get(i)
            if t is None:
                raise PresetError(R_REGISTRY, f"tree {i!r} not in the registry")
            out.append(t)
    elif form == "row_ids":
        ids = [val] if isinstance(val, str) else list(val)
        out = []
        for r in ids:
            row = sorted([t for t in trees if t.get("row_id") == r], key=lambda t: t["geo"]["x"])
            if not row:
                raise PresetError(R_REGISTRY, f"row {r!r} not in the registry")
            out += row
    elif form == "block_id":
        out = sorted([t for t in trees if t.get("block_id") == val], key=lambda t: (t["geo"]["y"], t["geo"]["x"]))
        if not out:
            raise PresetError(R_REGISTRY, f"block {val!r} not in the registry")
    else:  # block_sample: {block_id, n}
        if not isinstance(val, dict) or "block_id" not in val:
            raise PresetError(R_PARAM, "block_sample needs {block_id, n}", "targets")
        block = [t for t in trees if t.get("block_id") == val["block_id"]]
        n = val.get("n", 1)
        if not block:
            raise PresetError(R_REGISTRY, f"block {val['block_id']!r} not in the registry")
        if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(block)):
            raise PresetError(R_PARAM, f"block_sample n must be 1..{len(block)}", "targets")
        out = block_sample(block, n)
    ids = [t["tree_id"] for t in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:                                  # a tree twice would count twice in the estimate and the Ready check
        raise PresetError(R_PARAM, f"{preset.id}: target trees {dup} given more than once", "targets")
    for t in out:
        if t.get("crop") not in preset.crops:
            raise PresetError(R_NOT_IN_MVP, f"{preset.id} runs on {list(preset.crops)}; {t['tree_id']} is "
                                          f"{t.get('crop')!r}")
    return out


def block_sample(block: Sequence[dict], n: int) -> List[dict]:
    """N trees stratified by direction (OS 5.6): round-robin over the four quadrants around the block centre."""
    cx = sum(t["geo"]["x"] for t in block) / len(block)
    cy = sum(t["geo"]["y"] for t in block) / len(block)
    quads: Dict[int, List[dict]] = {0: [], 1: [], 2: [], 3: []}
    for t in sorted(block, key=lambda t: t["tree_id"]):
        a = math.atan2(t["geo"]["y"] - cy, t["geo"]["x"] - cx)
        quads[int(((a + math.pi) / (math.pi / 2))) % 4].append(t)
    out: List[dict] = []
    while len(out) < n:
        for q in range(4):
            if quads[q] and len(out) < n:
                out.append(quads[q].pop(0))
    return out


def route_m(trees: Sequence[dict], field_doc: Optional[dict] = None) -> float:
    """Planned walking length of a row-based run: per row, the tree span + one in-row spacing."""
    rows: Dict[str, List[dict]] = {}
    for t in trees:
        rows.setdefault(t.get("row_id", ""), []).append(t)
    pitch = {r.get("row_id"): float(r.get("in_row_m", 2.5)) for r in (field_doc or {}).get("rows", [])}
    total = 0.0
    for rid, ts in rows.items():
        xs = [t["geo"]["x"] for t in ts]
        total += (max(xs) - min(xs)) + pitch.get(rid, 2.5) if len(ts) > 1 else 0.0
    return total


# ------------------------------------------------------------------ cmd/mission
def mission_id(device: str = "nandi-sim", now: Optional[datetime] = None) -> str:
    """Platform mission id ``{device}-{UTC yyyymmddThhmmZ}``."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{device}-{now.strftime('%Y%m%dT%H%MZ')}"


def to_mission_command(preset: Preset, targets: Optional[dict], params: Optional[dict] = None,
                       source: str = "console", now: Optional[datetime] = None, device: str = "nandi-sim",
                       ttl_s: float = 1800.0, field_doc: Optional[dict] = None) -> dict:
    """The one internal ``cmd/mission`` type for a preset start (MP 0.1): {mission_id, mode, not_after,
    plan {source, preset, version, targets, tree_ids, params, resolved, resolved_by_tree?}}. Validates params and
    targets first: a preset with targets needs the tree registry (no field_doc = PresetError registry_mismatch).
    ``resolved`` is for the first tree; a multi-tree run also carries ``resolved_by_tree`` {tree_id: params}
    (per-tree values such as the orbit radius from canopy_r_m)."""
    if source not in SOURCES:
        raise PresetError(R_PARAM, f"plan.source {source!r} not in {SOURCES}", "source")
    if not preset.enabled:
        raise PresetError(preset.disabled_reason, f"{preset.id} is disabled ({preset.disabled_reason})")
    trees = resolve_targets(preset, field_doc, targets)
    by_tree = {t["tree_id"]: preset.resolve(params or {}, t) for t in trees}
    resolved = by_tree[trees[0]["tree_id"]] if trees else preset.resolve(params or {}, None)
    now = now or datetime.now(timezone.utc)
    return {"mission_id": mission_id(device, now), "mode": preset.mode,
            "not_after": (now + timedelta(seconds=ttl_s)).isoformat(),
            "plan": {"source": source, "preset": preset.id, "version": preset.version,
                     "targets": dict(targets or {}), "tree_ids": [t["tree_id"] for t in trees],
                     "params": dict(params or {}), "resolved": resolved,
                     **({"resolved_by_tree": by_tree} if len(trees) > 1 else {})}}


# ------------------------------------------------------------------ console (WP10 reads this)
def console_tiles(registry: Registry, ready: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Preset bar entries: {id, tile, hi, en, icon, mode, enabled, reason, level, season, show, advanced}.
    ``ready`` maps preset id -> ready_check.ReadyResult (or its dict); a tile without a result is disabled with
    reason ``not_checked``."""
    out = []
    for p in sorted(registry, key=lambda p: (p.console.get("order", 99), p.id)):
        c = p.console
        tile = {"id": p.id, "tile": c.get("tile"), "hi": c.get("hi"), "en": c.get("en"), "icon": c.get("icon"),
                "group": c.get("group", p.kind), "mode": p.mode, "show": list(c.get("show") or []),
                "advanced": list(c.get("advanced") or [])}
        if not p.enabled:
            tile.update(enabled=False, reason=p.disabled_reason, level="no_go", season=None)
        else:
            r = (ready or {}).get(p.id)
            r = r.to_dict() if hasattr(r, "to_dict") else r
            if r is None:
                tile.update(enabled=False, reason="not_checked", level=None, season=None)
            else:
                tile.update(enabled=r["level"] != "no_go", reason=(r["reasons"][0] if r["reasons"] else ""),
                            level=r["level"], needs_confirm=r["needs_confirm"], season=r.get("season"))
        out.append(tile)
    return out
