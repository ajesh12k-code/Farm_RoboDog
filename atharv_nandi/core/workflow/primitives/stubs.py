"""Stub adapters for every primitive (WP34; WORKFLOWS-SPEC A.3, A.6.4).

They let validation, the engine tests and the fault matrix run before any controller adapter exists (WP35 /
WP38 bind the real ones with ``registry.bind``). A ``StubWorld`` is the scripted environment of one run:

* ``script``: node id or primitive name -> outcome ``success`` | ``fail`` | ``hang`` (never finishes: the active
  timeout fires) or a list of outcomes consumed one per start (``["fail", "success"]`` exercises a retry);
* ``durations``: primitive -> ticks until it finishes (default 2 for RUNNING primitives, 1 otherwise);
* ``on_start``: node id or primitive -> ``fn(engine)`` called when that node starts (fault injection);
* ``signals``: condition id -> value mirrored into ``/sensors``, ``/safety``, ... every tick (all pass by
  default; ``holding`` and ``drift_ok`` follow the stub robot state);
* abstract robot state (``holding``, ``mast``, ``arm``, ``work``, ``tool``, ``pose``) and an **invariant
  monitor**: a commit outside an atomic block, ``arm_park`` while holding without ``release_safe``, a tool change
  while holding or with the mast up, a blade outside ``enter_work`` or motion before ``start_ack`` is appended to
  ``violations`` (the static check proves the same properties on the YAML; the fault matrix checks them at run
  time).
"""
import math
from typing import Any, Callable, Dict, List, Optional

from . import registry as reg

RUNNING, SUCCESS, FAILURE = "running", "success", "failure"


class StubWorld:
    def __init__(self, targets: Optional[Dict[str, List[str]]] = None, script: Optional[Dict[str, Any]] = None,
                 durations: Optional[Dict[str, int]] = None, signals: Optional[Dict[str, Any]] = None,
                 assist_reply: str = "skip", place_reachable: bool = True, estop_drops: bool = True,
                 on_start: Optional[Dict[str, Callable]] = None, drift_m: float = 0.0):
        self.targets = dict(targets or {})
        self.default_targets = ["t1", "t2", "t3"]
        self.script = {k: (list(v) if isinstance(v, list) else v) for k, v in (script or {}).items()}
        self.durations = dict(durations or {})
        self.signals: Dict[str, Any] = {cid: True for cid in reg.CONDITIONS}
        self.signals.update(signals or {})
        self.assist_reply = assist_reply
        self.place_reachable = place_reachable
        self.estop_drops = estop_drops
        self.on_start = dict(on_start or {})
        self.drift_m = drift_m
        self.holding = False
        self.released = False
        self.mast = "stowed"
        self.arm = "parked"
        self.work = False
        self.tool: Optional[str] = None
        self.pose = [0.0, 0.0, 0.0]
        self.calls: List[tuple] = []          # (event, node id, primitive)
        self.violations: List[str] = []
        self.placed = 0
        self.dropped = 0

    # ---------------------------------------------------------------- script
    def outcome_for(self, node_id: str, name: str) -> str:
        for key in (node_id, name):
            if key in self.script:
                v = self.script[key]
                if isinstance(v, list):
                    return v.pop(0) if v else SUCCESS
                return v
        return SUCCESS

    def duration(self, spec: reg.PrimitiveSpec) -> int:
        return int(self.durations.get(spec.name, 2 if spec.running else 1))

    def plan(self, set_name: str) -> List[str]:
        return list(self.targets.get(set_name, self.default_targets))

    def signal_values(self, engine=None) -> Dict[str, Any]:
        out = dict(self.signals)
        if "holding" not in self._overrides():
            out["holding"] = self.holding
        if "drift_ok" not in self._overrides():
            out["drift_ok"] = self._drift_ok(engine)
        return out

    def _overrides(self):
        return getattr(self, "_fixed", set())

    def fix_signal(self, cid: str, value: Any) -> None:
        """Pin a signal (also ``holding`` / ``drift_ok``, which otherwise follow the stub state)."""
        self.signals[cid] = value
        self._fixed = self._overrides() | {cid}

    def _drift_ok(self, engine) -> bool:
        target = getattr(engine, "resume_target", None) if engine is not None else None
        if target is None:
            return True
        dx, dy = self.pose[0] - target[0], self.pose[1] - target[1]
        dyaw = abs((self.pose[2] - target[2] + 180.0) % 360.0 - 180.0)
        return math.hypot(dx, dy) <= 0.10 and dyaw <= 5.0

    # ---------------------------------------------------------------- invariant monitor
    def check_start(self, spec: reg.PrimitiveSpec, ctx) -> None:
        n = spec.name
        where = getattr(ctx, "current_node", "")
        if spec.commit and getattr(ctx, "atomic_depth", 0) == 0:
            self.violations.append(f"commit_outside_atomic {n} at {where}")
        if n == "arm_park" and self.holding and not self.released:
            self.violations.append(f"arm_park_while_holding at {where}")
        if n == "tool_change" and (self.holding or self.mast != "stowed"):
            self.violations.append(f"tool_change_unsafe at {where}")
        if spec.blade and not self.work:
            self.violations.append(f"blade_outside_work {n} at {where}")
        if spec.motion and not spec.safe_direction and not getattr(ctx, "started", True):
            self.violations.append(f"motion_before_start_ack {n} at {where}")

    def apply_success(self, spec: reg.PrimitiveSpec, args: dict) -> None:
        n = spec.name
        if n in ("raise_mast", "mast_snake"):
            self.mast = "up"
        elif n == "stow_mast":
            self.mast = "stowed"
        elif n in ("arm_approach", "roll_align", "trap", "lower"):
            self.arm = "out"
        elif n == "arm_park":
            self.arm = "parked"
        elif n == "enter_work":
            self.work = True
        elif n in ("exit_work", "tools_safe"):          # tools made safe: enter_work again before a blade
            self.work = False
        elif n == "tool_change":
            self.tool = args.get("tool")
        elif spec.commit and not spec.self_placing:
            self.holding, self.released, self.arm = True, False, "out"
        elif n == "place":
            if self.holding:
                self.placed += 1
            self.holding = self.released = False
        elif n == "release_safe" and self.holding:
            if self.place_reachable:
                self.placed += 1
                self.holding = self.released = False
            else:
                self.released = True
        elif n == "pick":
            self.placed += 1


class StubPrimitive:
    """Generic stub: RUNNING for ``duration`` ticks, then the scripted outcome; success applies the effect."""

    def __init__(self, spec: reg.PrimitiveSpec):
        self.spec = spec
        self.left = 0
        self.mode = SUCCESS
        self.args: Dict[str, Any] = {}
        self.reason: Optional[str] = None
        self.active = False

    def world(self, ctx) -> StubWorld:
        return ctx.world

    def start(self, args: dict, ctx) -> None:
        w = self.world(ctx)
        self.args = dict(args)
        self.reason = None
        node = getattr(ctx, "current_node", "")
        self.mode = w.outcome_for(node, self.spec.name)
        self.left = w.duration(self.spec)
        self.active = True
        w.calls.append(("start", node, self.spec.name))
        w.check_start(self.spec, ctx)
        for key in (node, self.spec.name):
            fn = w.on_start.get(key)
            if fn is not None:
                fn(ctx.engine)

    def resume(self, ctx) -> None:
        self.active = True
        self.world(ctx).calls.append(("resume", getattr(ctx, "current_node", ""), self.spec.name))

    def cancel(self, ctx) -> None:
        if self.active:
            self.active = False
            self.world(ctx).calls.append(("cancel", getattr(ctx, "current_node", ""), self.spec.name))

    def update(self, ctx) -> str:
        if self.mode == "hang":
            return RUNNING
        self.left -= 1
        if self.left > 0:
            return RUNNING
        return self.finish(ctx)

    def finish(self, ctx) -> str:
        w = self.world(ctx)
        self.active = False
        if self.mode == "fail":
            w.calls.append(("fail", getattr(ctx, "current_node", ""), self.spec.name))
            return FAILURE
        out = self.effect(ctx)
        w.calls.append(("success" if out == SUCCESS else out, getattr(ctx, "current_node", ""), self.spec.name))
        return out

    def effect(self, ctx) -> str:
        self.world(ctx).apply_success(self.spec, self.args)
        return SUCCESS


class _StartAck(StubPrimitive):
    def update(self, ctx) -> str:
        if self.mode == "fail":
            return self.finish(ctx)
        if self.mode == "hang":
            return RUNNING
        if ctx.engine.acked and ctx.cond("local_link"):
            ctx.started = True
            return self.finish(ctx)
        return RUNNING


class _WaitOperator(StubPrimitive):
    def update(self, ctx) -> str:
        if self.mode == "fail":
            return self.finish(ctx)
        return self.finish(ctx) if ctx.engine.acked and self.mode != "hang" else RUNNING


class _Assist(StubPrimitive):
    def effect(self, ctx) -> str:
        reply = self.world(ctx).assist_reply
        self.world(ctx).calls.append(("assist_reply", getattr(ctx, "current_node", ""), reply))
        if reply == "take_over":
            return SUCCESS
        if reply == "halt":
            ctx.halt("operator_halt")
            return FAILURE
        ctx.skip_target(self.args.get("reason") or ctx.last_reason or "operator_skip")
        return FAILURE


class _Skip(StubPrimitive):
    def effect(self, ctx) -> str:
        ctx.skip_target(self.args["reason"])
        return FAILURE


class _Halt(StubPrimitive):
    def effect(self, ctx) -> str:
        ctx.halt(self.args["reason"])
        return FAILURE


class _Set(StubPrimitive):
    def effect(self, ctx) -> str:
        ctx.bb.set(self.args["key"], self.args["value"])
        return SUCCESS


class _RequestRth(StubPrimitive):
    def effect(self, ctx) -> str:
        ctx.request("rth", self.args.get("reason", "rth_operator"), source="workflow")
        return SUCCESS


class _Request(StubPrimitive):
    def effect(self, ctx) -> str:
        ctx.request(self.args.get("what", "safe_park"), "safe_park", source="workflow")
        return SUCCESS


class _HoldVerify(StubPrimitive):
    def effect(self, ctx) -> str:
        return SUCCESS if self.world(ctx).holding else FAILURE


class _ReleaseSafe(StubPrimitive):
    def effect(self, ctx) -> str:
        w = self.world(ctx)
        was = w.holding
        w.apply_success(self.spec, self.args)
        if was and w.holding:
            ctx.event("assist", reason="holding_item")
        return SUCCESS


class _ReturnToCheckpoint(StubPrimitive):
    def effect(self, ctx) -> str:
        w = self.world(ctx)
        target = getattr(ctx.engine, "resume_target", None)
        if target is not None:
            w.pose = [target[0] + w.drift_m, target[1], target[2]]
        return SUCCESS


SPECIAL = {
    "start_ack": _StartAck, "wait_operator": _WaitOperator, "assist_request": _Assist, "skip": _Skip,
    "halt": _Halt, "set": _Set, "request_rth": _RequestRth, "request": _Request, "hold_verify": _HoldVerify,
    "release_safe": _ReleaseSafe, "return_to_checkpoint": _ReturnToCheckpoint,
}


def factory(spec: reg.PrimitiveSpec) -> StubPrimitive:
    return SPECIAL.get(spec.name, StubPrimitive)(spec)


def bind_all(only_missing: bool = False) -> None:
    """Bind the stub for every catalogue primitive (``only_missing``: keep adapters WP35 / WP38 bound)."""
    current = reg.bound()
    for name in reg.PRIMITIVES:
        if only_missing and name in current:
            continue
        reg.bind(name, factory)
