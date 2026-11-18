"""Pilot-grade interface stubs shared between work packages (WP0; MVP-SPEC 5.1, ORCHARD-MVP-SPEC 2.3).

Each Protocol is the contract a later package implements; each ``Null*`` class is a safe, inert stand-in
so callers (WP6 e-stop, WP25/WP27 presets, WP28 recovery) can be written and tested before the
implementation lands. Plain data, no ROS, no MuJoCo.

    MastControl     WP24 implements (sole writer of the mast setpoint, command arbiter)
    MastRequest     what WP6 / WP25 / WP27 / WP28 submit to the arbiter: {source, target_ext, reason}
    GateDecision    WP6 gate functions return it (crown, slope, wind, speed gates)
    TreeObservation WP5/WP25/WP26 produce it per capture, keyed by tree and band
    PresetStep      WP27 preset step (one row of ``config/presets/<preset>.yaml`` ``steps[]``)
    ToolActuation   WP2 implements (blade/jaw); WP6 codes the e-stop against ``open_blade``/``stop``
    TransferControl WP3 implements (tray / cradle / unload transfers)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

# ---------------------------------------------------------------------- mast
MAST_EXT_MAX_M = 1.60          # ORCHARD-MVP-SPEC 2.3: extension range [0, 1.60]
MAST_SOURCES = ("estop", "safety", "recovery", "preset")
# Arbiter priority: lower number wins (e-stop brake > safety retract > recovery > preset).
MAST_PRIORITY = {s: i for i, s in enumerate(MAST_SOURCES)}


@dataclass(frozen=True)
class MastRequest:
    """One request to the WP24 arbiter. ``target_ext`` None = brake / hold in place (e-stop)."""
    source: str
    target_ext: Optional[float]
    reason: str
    gimbal: Optional[Tuple[float, float]] = None   # (pan, tilt) rad of the virtual gimbal, None = keep
    t: float = 0.0

    def __post_init__(self):
        if self.source not in MAST_PRIORITY:
            raise ValueError(f"mast source {self.source!r} not in {MAST_SOURCES}")
        if self.target_ext is not None and not 0.0 <= float(self.target_ext) <= MAST_EXT_MAX_M:
            raise ValueError(f"target_ext {self.target_ext} outside [0, {MAST_EXT_MAX_M}]")
        if not self.reason:
            raise ValueError("a mast request needs a reason")

    @property
    def priority(self) -> int:
        return MAST_PRIORITY[self.source]


def arbitrate(requests: Sequence[MastRequest]) -> Optional[MastRequest]:
    """Reference priority rule: the highest-priority request wins; within a source the newest wins."""
    if not requests:
        return None
    return min(requests, key=lambda r: (r.priority, -r.t))


@runtime_checkable
class MastControl(Protocol):
    def submit(self, request: MastRequest) -> "GateDecision": ...
    def brake(self) -> None: ...
    def state(self) -> Dict[str, Any]: ...          # {ext_m, lens_z, gimbal, gated, reason, arbiter_source}


# ---------------------------------------------------------------------- gates
@dataclass(frozen=True)
class GateDecision:
    """Result of one gate (WP6 pure functions). ``allowed`` False carries a reason code (MISSION-PRESETS 0.5)."""
    allowed: bool
    gate: str
    reason: str = ""
    values: Dict[str, float] = field(default_factory=dict)   # the numbers the decision used (margins, slope...)
    limits: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.gate:
            raise ValueError("a gate decision needs the gate name")
        if not self.allowed and not self.reason:
            raise ValueError(f"gate {self.gate!r} refused without a reason code")

    def __bool__(self):
        return self.allowed

    @classmethod
    def ok(cls, gate: str, **values) -> "GateDecision":
        return cls(True, gate, "", dict(values))

    @classmethod
    def refuse(cls, gate: str, reason: str, **values) -> "GateDecision":
        return cls(False, gate, reason, dict(values))


# ---------------------------------------------------------------------- trees
BANDS = ("low", "mid", "high")


@dataclass
class TreeObservation:
    """One pose-tagged observation of a tree (no image bytes, no personal fields)."""
    tree_id: str
    band: str
    t: float
    camera: str
    pose: Tuple[float, ...]                      # camera pose in the world (x, y, z, qw, qx, qy, qz)
    azimuth_deg: float = 0.0                     # around the tree, 0 = +x, CCW
    side: str = ""                               # "L" | "R" | "" (full orbit)
    frame_ref: str = ""                          # recorder key of the stored frame
    masked_pct: float = 0.0
    detections: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.band not in BANDS:
            raise ValueError(f"band {self.band!r} not in {BANDS}")
        if len(self.pose) != 7:
            raise ValueError("pose is (x, y, z, qw, qx, qy, qz)")


# ---------------------------------------------------------------------- presets
@dataclass(frozen=True)
class PresetStep:
    preset: str
    step: str
    params: Dict[str, Any] = field(default_factory=dict)
    preconditions: Tuple[str, ...] = ()
    interlocks: Tuple[str, ...] = ()
    timeout_s: Optional[float] = None


# ---------------------------------------------------------------------- tools
@runtime_checkable
class ToolActuation(Protocol):
    def open_blade(self) -> None: ...
    def stop(self) -> None: ...
    def set(self, value: float) -> None: ...
    def state(self) -> Dict[str, Any]: ...


@runtime_checkable
class TransferControl(Protocol):
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def busy(self) -> bool: ...


class NullToolActuation:
    """No tool mounted: every command is accepted and recorded, nothing moves."""

    def __init__(self):
        self.value = 0.0
        self.stopped = False
        self.log: List[str] = []

    def open_blade(self):
        self.value = 0.0
        self.log.append("open_blade")

    def stop(self):
        self.stopped = True
        self.log.append("stop")

    def set(self, value: float):
        if self.stopped:
            self.log.append("set refused (stopped)")
            return
        self.value = float(value)
        self.log.append(f"set {self.value:g}")

    def state(self):
        return {"mounted": None, "value": self.value, "stopped": self.stopped, "stub": True}


class NullTransferControl:
    def __init__(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def busy(self) -> bool:
        return False


class NullMastControl:
    """No mast: every request is refused with ``not_in_mvp`` except the brake; state reports a stowed crown."""

    def __init__(self):
        self.braked = False
        self.requests: List[MastRequest] = []

    def submit(self, request: MastRequest) -> GateDecision:
        self.requests.append(request)
        return GateDecision.refuse("mast", "not_in_mvp", target_ext=request.target_ext or 0.0)

    def brake(self):
        self.braked = True

    def state(self):
        return {"ext_m": 0.0, "lens_z": None, "gimbal": (0.0, 0.0), "gated": True, "reason": "not_in_mvp",
                "arbiter_source": None, "braked": self.braked}
