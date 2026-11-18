"""Primitive catalogue v1 (WP34; WORKFLOWS-SPEC A.3) and the adapter binding point.

Every primitive declares typed args with ranges, blackboard reads and writes, whether it can return RUNNING,
a default timeout, a resumable policy, a commit flag, its motion class and the owning adapter package. The
validator and the static check read ONLY this table; the engine asks ``adapter(name)`` for an implementation.
``stubs.py`` (WP34) binds a stub for every primitive so validation and the fault matrix run before any
controller exists; WP35 / WP38 adapters replace them with ``bind(name, factory)``.

Adapter protocol (one instance per action node, built by ``factory(spec)``)::

    start(args: dict, ctx) -> None      the node was (re-)entered; ``ctx`` is the engine RunContext
    update(ctx) -> "running" | "success" | "failure"
    cancel(ctx) -> None                 the node was stopped while RUNNING (pause, abort, L0)
    resume(ctx) -> None                 optional: ``resumable: continue`` re-entry after a pause
    reason -> str | None                optional: failure reason in the reasons.py enum

Conditions are a separate table: id -> blackboard source key. A condition is evaluated by reading its source
key through a READ-only blackboard client (blackboard.py); no source key lives under ``/wf/``, so a YAML
``set()`` can never spoof one (validator) and a runtime write attempt raises (py_trees key access control).
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

PRIMITIVES_API = 1          # major; a workflow with another ``primitives_api`` is refused

RESUMABLE = ("restart", "continue", "rollback")


@dataclass(frozen=True)
class ArgSpec:
    kind: str                                  # num | int | str | bool | enum | list | any | key | reason | ref
    required: bool = False
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Tuple[str, ...] = ()
    default: Any = None


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    group: str
    owner: str                                 # adapter package (WP34 stubs bind them all until then)
    args: Dict[str, ArgSpec] = field(default_factory=dict)
    running: bool = True                       # may return RUNNING -> ``timeout_s`` is mandatory
    default_timeout_s: Optional[float] = None
    max_timeout_s: float = 3600.0
    resumable: str = "restart"                 # restart | continue | rollback(arm_park)
    commit: bool = False                       # commit point (A.2): must be the first child of an atomic
    self_placing: bool = False                 # bridge ``pick``: the whole call is one atomic incl. the place
    motion: Optional[str] = None               # base | mast | arm | None
    safe_direction: bool = False               # stopping / parking motion allowed without start_ack
    blade: bool = False                        # blade or jaw primitive: only inside enter_work
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    request_only: bool = False                 # safety-facing request that L0 may refuse


def check_value(s: ArgSpec, v: Any) -> Optional[str]:
    """Why ``v`` does not satisfy ``s`` (type, catalogue range, enum), or None. The engine checks every
    resolved ``$param`` arg with it before a run starts."""
    if s.kind in ("num", "int"):
        if not isinstance(v, (int, float)) or isinstance(v, bool) or (s.kind == "int" and not isinstance(v, int)):
            return f"expected a number, got {v!r}"
        if v != v or (s.min is not None and v < s.min) or (s.max is not None and v > s.max):
            return f"{v} outside the catalogue range [{s.min}, {s.max}]"
    elif s.kind == "enum" and v not in s.choices:
        return f"{v!r} not in {list(s.choices)}"
    elif s.kind in ("str", "key", "ref") and not isinstance(v, str):
        return f"expected a string, got {v!r}"
    elif s.kind == "bool" and not isinstance(v, bool):
        return f"expected a bool, got {v!r}"
    elif s.kind == "list" and not isinstance(v, list):
        return f"expected a list, got {v!r}"
    elif s.kind == "reason":
        from .. import reasons
        if not reasons.is_reason(v):
            return f"reason {v!r} is not in atharv_nandi/core/workflow/reasons.py"
    return None


def _n(lo=None, hi=None, req=False, default=None):
    return ArgSpec("num", req, lo, hi, (), default)


def _e(*choices, req=False, default=None):
    return ArgSpec("enum", req, None, None, tuple(choices), default)


_S = ArgSpec("str")
_SR = ArgSpec("str", True)
_L = ArgSpec("list")
_B = ArgSpec("bool")

BANDS = ("low", "mid", "high", "ring_low", "ext")
CAMS = ("ring", "crown", "wrist", "head", "belly", "rear")


def _spec(name, group, owner, **kw) -> PrimitiveSpec:
    return PrimitiveSpec(name=name, group=group, owner=owner, **kw)


_CATALOGUE = [
    # ------------------------------------------------------------ start and supervision (WP34 -> WP38)
    _spec("start_ack", "supervision", "WP34", default_timeout_s=600, resumable="restart",
          reads=("/safety/local_link",)),
    _spec("wait_operator", "supervision", "WP34", args={"ack": _SR}, default_timeout_s=600),
    # ------------------------------------------------------------ motion (WP35; WP6 caps in the adapter)
    _spec("plan_route", "motion", "WP35", running=True, default_timeout_s=10,
          args={"pattern": _S, "constraints": _L, "set": _S}),
    _spec("move_to", "motion", "WP35", motion="base", default_timeout_s=60,
          args={"target": _SR, "v_max": _n(0.0, 0.30), "fence": _S, "standoff_m": _n(-5.0, 10.0),
                "max_slope_deg": _n(0.0, 15.0)}),
    _spec("follow_rows", "motion", "WP35", motion="base", default_timeout_s=600,
          args={"mode": _e("rtk", "odometry", default="rtk"), "v_max": _n(0.0, 0.30)}),
    _spec("orbit_arc", "motion", "WP35", motion="base", default_timeout_s=120,
          args={"tree": _S, "r": _n(0.5, 3.0), "dtheta": _n(-6.3, 6.3), "v_max": _n(0.0, 0.30)}),
    _spec("settle", "motion", "WP35", default_timeout_s=10, resumable="restart", args={"s": _n(0.0, 30.0)}),
    _spec("stop", "motion", "WP35", running=False, motion="base", safe_direction=True),
    _spec("backup", "motion", "WP35", motion="base", default_timeout_s=15, args={"d": _n(0.0, 0.3, req=True)}),
    _spec("alt_heading", "motion", "WP35", motion="base", default_timeout_s=15, args={"deg": _n(-30.0, 30.0, req=True)}),
    _spec("sit_belly", "motion", "WP35", motion="base", safe_direction=True, default_timeout_s=20),
    _spec("stand", "motion", "WP35", motion="base", default_timeout_s=20),
    _spec("reenter_station", "motion", "WP35", motion="base", default_timeout_s=30,   # A.2 tree360 example
          args={"station": _S, "v_max": _n(0.0, 0.30)}),
    _spec("return_to_checkpoint", "motion", "WP35", motion="base", default_timeout_s=1800, resumable="restart",
          args={"v_max": _n(0.0, 0.30)}),
    # ------------------------------------------------------------ mast and gimbal (WP35 -> WP24 arbiter)
    _spec("raise_mast", "mast", "WP35", motion="mast", default_timeout_s=12, request_only=True,
          args={"band": _e(*BANDS), "ext": _n(0.0, 1.60)}),
    _spec("stow_mast", "mast", "WP35", motion="mast", safe_direction=True, default_timeout_s=9, request_only=True),
    _spec("gimbal_aim", "mast", "WP35", default_timeout_s=5, args={"bearing": _n(-180, 180), "tilt": _n(-90, 30)}),
    _spec("mast_snake", "mast", "WP35", motion="mast", default_timeout_s=60, request_only=True, args={"band": _e(*BANDS)}),
    # ------------------------------------------------------------ sensing (WP35)
    _spec("capture", "sensing", "WP35", default_timeout_s=20,
          args={"cams": _L, "bands": _L, "window": _n(0.0, 30.0), "mode": _e("normal", "close_range", default="normal")}),
    _spec("lidar_profile", "sensing", "WP35", default_timeout_s=10),
    _spec("terrain_patch", "sensing", "WP35", default_timeout_s=10),
    _spec("detect", "sensing", "WP35", default_timeout_s=10, args={"model": _S, "classes": _L}),
    _spec("track_dedupe", "sensing", "WP35", running=False, args={"r": _n(0.0, 2.0)}),
    _spec("stage_estimate", "sensing", "WP35", running=False),
    _spec("coverage_update", "sensing", "WP35", running=False),
    _spec("read_trap", "sensing", "WP35", default_timeout_s=10),
    _spec("visibility_check", "sensing", "WP35", running=False),
    # ------------------------------------------------------------ records (WP35)
    _spec("update_record", "records", "WP35", running=False, args={"kind": _S}),
    _spec("triage", "records", "WP35", running=False),
    _spec("mark", "records", "WP35", running=False, args={"kind": _SR, "reason": ArgSpec("reason")}),
    _spec("jobsheet_line", "records", "WP35", running=False),
    _spec("deepscan_request", "records", "WP35", running=False),
    _spec("manifest_close", "records", "WP35", running=False),
    _spec("log_kpi", "records", "WP35", running=False, args={"kpi": _S, "value": ArgSpec("any")}),
    # ------------------------------------------------------------ work zone and arm (WP35)
    _spec("enter_work", "arm", "WP35", default_timeout_s=5),
    _spec("exit_work", "arm", "WP35", default_timeout_s=5),
    _spec("arm_approach", "arm", "WP35", motion="arm", default_timeout_s=15, resumable="rollback",
          args={"pose": _S, "h": _n(0.0, 1.0)}),
    _spec("arm_park", "arm", "WP35", motion="arm", safe_direction=True, default_timeout_s=10),
    _spec("tool_change", "arm", "WP35", motion="arm", default_timeout_s=30, args={"tool": _SR}),
    _spec("roll_align", "arm", "WP35", motion="arm", default_timeout_s=5, resumable="rollback"),
    _spec("trap", "arm", "WP35", motion="arm", blade=True, default_timeout_s=8, resumable="rollback"),
    _spec("clip", "arm", "WP35", motion="arm", blade=True, commit=True, default_timeout_s=5,
          args={"cut_k": _n(0.0, 1.0)}),
    _spec("twist", "arm", "WP35", motion="arm", blade=True, commit=True, default_timeout_s=8, args={"rule": _S}),
    _spec("cut", "arm", "WP35", motion="arm", blade=True, commit=True, default_timeout_s=5),
    _spec("ground_pick", "arm", "WP35", motion="arm", blade=True, commit=True, default_timeout_s=15,
          args={"target": _S}),
    _spec("hold_verify", "arm", "WP35", default_timeout_s=5, args={"window_s": _n(0.0, 5.0)}),
    _spec("place", "arm", "WP35", motion="arm", default_timeout_s=15,
          args={"into": _e("cell", "cradle", "sanitation_bin", default="cell")}),
    _spec("lower", "arm", "WP35", motion="arm", default_timeout_s=10, args={"v_max": _n(0.0, 0.5)}),
    _spec("release_safe", "arm", "WP35", motion="arm", safe_direction=True, default_timeout_s=20),
    _spec("pick", "arm", "WP35", motion="arm", blade=True, commit=True, self_placing=True, default_timeout_s=60,
          args={"method": _e("clip", "twist", "suction")}),
    # ------------------------------------------------------------ assist (WP35)
    _spec("lead_stop", "assist", "WP35", motion="base", default_timeout_s=120, args={"mark": _S}),
    _spec("follow_tag", "assist", "WP35", motion="base", default_timeout_s=1800),
    _spec("assist_request", "assist", "WP35", default_timeout_s=300,
          args={"options": ArgSpec("list", default=["take_over", "skip", "halt"]), "reason": ArgSpec("reason")}),
    # ------------------------------------------------------------ dock and RTH (WP38)
    _spec("request_rth", "dock", "WP38", running=False, args={"reason": ArgSpec("reason", True)}),
    *[_spec(n, "dock", "WP38", motion=("base" if n in ("dock_approach", "dock_final_align", "dock_seat",
                                                       "dock_stand", "undock") else None), default_timeout_s=t)
      for n, t in (("dock_stage", 60), ("dock_detect", 30), ("dock_approach", 120), ("dock_final_align", 60),
                   ("under_belly_clear", 10), ("dock_seat", 30), ("verify_contacts", 10), ("dock_stand", 30),
                   ("undock", 60), ("unload_transfer", 120), ("sync_stub", 60))],
    _spec("charge", "dock", "WP38", default_timeout_s=36000, max_timeout_s=86400, resumable="continue",
          args={"until": _S}),
    # ------------------------------------------------------------ safety-facing, request only (WP34)
    _spec("tools_safe", "safety", "WP34", safe_direction=True, default_timeout_s=10, request_only=True),
    _spec("safe_hold", "safety", "WP34", running=False, safe_direction=True, request_only=True),
    _spec("beacon", "safety", "WP34", running=False, args={"pattern": _S}),
    _spec("request", "safety", "WP34", running=False, request_only=True, args={"what": _e("safe_park", req=True)}),
    # ------------------------------------------------------------ flow (WP34)
    _spec("skip", "flow", "WP34", running=False, args={"reason": ArgSpec("reason", True)}),
    _spec("halt", "flow", "WP34", running=False, args={"reason": ArgSpec("reason", True)}),
    _spec("set", "flow", "WP34", running=False, args={"key": ArgSpec("key", True), "value": ArgSpec("any", True)}),
]

PRIMITIVES: Dict[str, PrimitiveSpec] = {p.name: p for p in _CATALOGUE}

# ``wait_for`` and ``check`` are node types in the YAML (A.2) and flow entries in the catalogue (A.3)
FLOW_NODES = ("wait_for", "check")

# primitive names a YAML file must never name: they would clear an e-stop, bypass the WP24 arbiter or remove
# the L0/L1 person pause (A.6.3). Listed so the validator can say WHY instead of "unknown primitive".
FORBIDDEN: Dict[str, str] = {
    "estop_reset": "clears the e-stop (L0 only)", "clear_estop": "clears the e-stop (L0 only)",
    "estop_clear": "clears the e-stop (L0 only)", "reset_estop": "clears the e-stop (L0 only)",
    "disable_person_pause": "removes the L0/L1 person pause", "person_pause_off": "removes the L0/L1 person pause",
    "disable_safety": "disables L0", "mast_direct": "bypasses the WP24 mast arbiter",
    "arbiter_override": "bypasses the WP24 mast arbiter", "set_mast_ext": "bypasses the WP24 mast arbiter",
}


# ---------------------------------------------------------------------------------------------- conditions
@dataclass(frozen=True)
class ConditionSpec:
    id: str
    source: str            # blackboard key the condition reads (never under /wf/)
    owner: str
    crown_gate: bool = False


def _c(cid, ns, owner="WP35", crown_gate=False):
    return ConditionSpec(cid, f"/{ns}/{cid}", owner, crown_gate)


READY_COMMON = ("rtk_fixed", "soc_ge_reserve", "disk_ok", "geofence_contains", "estop_clear", "local_link",
                "stability", "cross_slope_le", "light_ge", "wind_lt_crown", "rain_dry", "heat_ok", "season_ok",
                "wet_foliage", "tool_in_rack", "payload_room")

_NS = {
    "rtk_fixed": "sensors", "soc_ge_reserve": "energy", "disk_ok": "sensors", "geofence_contains": "sensors",
    "estop_clear": "safety", "local_link": "safety", "stability": "safety", "cross_slope_le": "sensors",
    "light_ge": "sensors", "wind_lt_crown": "weather", "rain_dry": "weather", "heat_ok": "weather",
    "season_ok": "sensors", "wet_foliage": "weather", "tool_in_rack": "sensors", "payload_room": "sensors",
    "person_warning_clear": "safety", "protective_clear": "safety", "hand_in_envelope_clear": "safety",
    "annulus_clear": "sensors", "alley_clear_height": "sensors", "target_tracked": "sensors",
    "base_drift_lt_2cm": "sensors", "tag_seen": "sensors", "tote_kg_lt": "sensors", "cloud_link_up": "rth",
    "visibility_ok": "sensors", "trafficable": "sensors", "holding": "sensors", "crown_gate": "safety",
    "ready_common": "safety", "annulus_clear_or_two_sided": "sensors", "canopy_r_known": "sensors",
    "slope_crown_walk": "safety", "drift_ok": "sensors",
}

CONDITIONS: Dict[str, ConditionSpec] = {
    cid: _c(cid, ns, owner="WP34" if cid in ("ready_common", "drift_ok") else "WP35",
            crown_gate=cid in ("crown_gate", "wind_lt_crown", "slope_crown_walk"))
    for cid, ns in _NS.items()
}

# conditions a YAML ``guard`` is not needed for: L0/L1 enforce them on every workflow (A.0 rule 1)
L0_L1_ENFORCED: FrozenSet[str] = frozenset({"person_warning_clear", "protective_clear", "estop_clear",
                                            "local_link", "hand_in_envelope_clear"})

# precondition / interlock ids accepted in headers (Ready-check ids are condition ids; interlocks per MP 0.3)
INTERLOCKS: FrozenSet[str] = frozenset({"common", "person_pause", "crown_scaled_fields", "crown_auto_retract",
                                        "arm_park", "crown_walk_cap", "work_fields", "speed_caps"})

SPEED_CAP_STOWED = 0.18     # m/s, MP 0.3 (measured, probe)
SPEED_CAP_CROWN_UP = 0.15   # m/s, MP 0.3
SPEED_CAP_MAX = 0.30        # m/s, MP 0.3


# ---------------------------------------------------------------------------------------------- adapters
_ADAPTERS: Dict[str, Callable[[PrimitiveSpec], Any]] = {}


def spec(name: str) -> PrimitiveSpec:
    try:
        return PRIMITIVES[name]
    except KeyError:
        raise KeyError(f"unknown primitive {name!r}") from None


def bind(name: str, factory: Callable[[PrimitiveSpec], Any]) -> None:
    """Bind the adapter factory for primitive ``name`` (WP35 / WP38 adapters; stubs.py binds every one)."""
    spec(name)
    _ADAPTERS[name] = factory


def unbind(name: str) -> None:
    _ADAPTERS.pop(name, None)


def adapter(name: str):
    """A fresh adapter instance for ``name``; the stub when nothing else is bound."""
    if name not in _ADAPTERS:
        from . import stubs  # noqa: F401  binds every stub at import
        stubs.bind_all(only_missing=True)
    return _ADAPTERS[name](spec(name))


def bound() -> Dict[str, str]:
    """name -> adapter factory qualname (introspection, contract tests)."""
    return {n: getattr(f, "__qualname__", repr(f)) for n, f in sorted(_ADAPTERS.items())}
