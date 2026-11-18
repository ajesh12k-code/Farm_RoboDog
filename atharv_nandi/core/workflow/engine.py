"""Workflow engine: py_trees 2.2.3 runtime for compiled ``nandi.workflow/1`` trees (WP34; WORKFLOWS-SPEC A.0-A.4).

Layering (A.0). The engine is **L2**: it ticks one compiled tree at 10 Hz and owns ``/wf/<run_id>/*``. The
**L0 mirror** (``/safety/*``) and the **mandatory L1 rules** are applied by ``Engine.tick`` BEFORE the tree is
ticked and do not depend on any YAML node, so deleting a ``guard`` can never remove them:

    estop_clear false            L0 e-stop, in every state but idle (paused included): stop everything NOW (an
                                 open atomic too: outcome ``dropped`` or ``held_at_estop``); terminal
                                 ``estopped``; no finaliser (the e-stop recovery checklist owns the item)
    protective_clear / local_link false
                                 L0 freeze NOW (local-link stop and hold); an open atomic is broken
                                 (logged); when clear again a running or resuming run goes through the normal
                                 pause (``on_pause``) and waits for an operator resume, while an abort, failure,
                                 halt or RTH finaliser restarts and still ends in its terminal state
    person_warning_clear false   L1 pause, reason ``person_hold``: running -> queued until any open ``atomic``
                                 ends; resuming -> the resume check is cancelled, then ``on_pause``; a finaliser
                                 is held (running leaves cancelled) until the field is clear

Operator commands (``pause``, ``resume``, ``abort``, ``skip_target``, ``rth``, ``ack``) are requests: the WP38
run supervisor is the only caller in production (the console never ticks the tree). Requests raised while an
``atomic`` block is open are queued and applied when it ends (A.2), so a pause between ``twist`` success and
``place`` lets ``place`` complete and then pauses. Paused: abort / halt run ``on_abort``, rth / safe_park run
``on_pause`` again (RTH checkpoint), nothing re-enters the tree. Resuming: abort / rth / pause cancel the resume
check first. Pausing: ``on_pause`` finishes, then an abort chains ``on_abort`` and an rth ends in ``rth``.

Resume restores the posture recorded at the pause (A.4 step 5) after the drift gate: the tool, the mast (behind
a ``crown_gate`` check) and the work zone, which ``on_pause`` (``tools_safe``, ``stow_mast``) took away.

Timeouts count ACTIVE time (``clock.ActiveClock``): the main tree's clock is frozen while it is not ticked.
Checkpoints (``checkpoint.py``) are written after every completed target, on pause and on RTH.

Production ticks run with py_trees debug logging, snapshot visitors and the blackboard activity stream off
(the 2 ms per tick gate, A.1); ``debug=True`` turns the activity stream on for tests and console snapshots.
"""
import copy
import itertools
from typing import Callable, Dict, List, Optional

import py_trees
from py_trees.common import Status

from . import blackboard, checkpoint, compiler, reasons
from .clock import ActiveClock, Stopwatch
from .primitives import registry as reg
from .primitives import stubs

DT = 0.1                         # 10 Hz tick (assumption, A.0)
ACK_AFTER_PAUSE_S = 900.0        # a pause longer than 15 min needs an operator ack on resume (A.4 step 3)
REQUEST_PRIORITY = {"abort": 5, "rth": 4, "safe_park": 3, "halt": 3, "pause": 2, "skip": 1}
ACTIVE_STATES = ("running", "pausing", "resuming", "aborting", "finalising")

_ST = {stubs.RUNNING: Status.RUNNING, stubs.SUCCESS: Status.SUCCESS, stubs.FAILURE: Status.FAILURE}


class ResumeRefused(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


# ---------------------------------------------------------------------------------------------- context
class RunContext:
    """What behaviours and adapters see of the run."""

    def __init__(self, engine: "Engine"):
        self.engine = engine
        self.world = engine.world
        self.bb = engine.bb
        self.atomic_depth = 0
        self.skip_pending: Optional[str] = None
        self.halt_pending: Optional[str] = None
        self.last_reason: Optional[str] = None
        self.current_node = ""
        self.started = False
        self.frames: List["Foreach"] = []
        self.clock: ActiveClock = engine.clock

    def cond(self, cid: str) -> bool:
        return self.bb.condition(cid)

    def resolve(self, args: dict) -> dict:
        out = {}
        for k, v in args.items():
            if isinstance(v, str) and v.startswith("$") and v[1:] in self.engine.params:
                v = self.engine.params[v[1:]]
            out[k] = v
        return out

    def skip_target(self, reason: str) -> None:
        self.skip_pending = reasons.check(reason)
        self.last_reason = reason
        self.event("skip_requested", reason=reason)

    def halt(self, reason: str) -> None:
        self.halt_pending = reasons.check(reason)
        self.last_reason = reason
        self.event("halt", reason=reason)

    def fail(self, reason: str) -> None:
        self.last_reason = reasons.check(reason)

    def request(self, kind: str, reason: str, source: str = "workflow") -> None:
        self.engine._request(kind, reason, source)

    def event(self, kind: str, **kw) -> None:
        self.engine.event(kind, **kw)

    @property
    def short_circuit(self) -> bool:
        return self.skip_pending is not None or self.halt_pending is not None

    def plan_targets(self, set_name: str) -> List[str]:
        override = self.bb.get(f"sets/{set_name}")
        if isinstance(override, list):
            return [str(t) for t in override]
        return [str(t) for t in self.world.plan(set_name)]


# ---------------------------------------------------------------------------------------------- behaviours
class _Wf:
    """Mixin: the IR node and the run context."""
    node: compiler.Node
    ctx: RunContext


class Action(py_trees.behaviour.Behaviour, _Wf):
    def __init__(self, node: compiler.Node, ctx: RunContext):
        super().__init__(name=node.label or node.f["do"])
        self.node, self.ctx = node, ctx
        self.do = node.f["do"]
        self.spec = reg.spec(self.do)
        self.adapter = None
        self.attempt = 0
        self.watch: Optional[Stopwatch] = None
        self.backoff: Optional[Stopwatch] = None
        self.interrupted = False
        self.rollback = None
        self.suppressed = False
        self.reason: Optional[str] = None
        self.cur_args: dict = {}

    # -------------------------------------------------------------- lifecycle
    def initialise(self):
        self.attempt = 0
        self.interrupted = False
        self.rollback = None
        self.backoff = None
        self.reason = None
        self.suppressed = self.ctx.short_circuit
        if not self.suppressed:
            self._begin()

    def _args(self) -> dict:
        args = self.ctx.resolve(self.node.f["args"])
        for a, deltas in (self.node.f["retry"].get("vary") or {}).items():
            if self.attempt and a in args and isinstance(args[a], (int, float)):
                args[a] = args[a] + deltas[(self.attempt - 1) % len(deltas)]
        return args

    def _begin(self):
        self.ctx.current_node = self.node.id
        self.adapter = reg.adapter(self.do)
        self.cur_args = self._args()
        self.adapter.start(self.cur_args, self.ctx)
        self.watch = Stopwatch(self.ctx.clock)

    def _cancel(self):
        if self.adapter is not None:
            self.ctx.current_node = self.node.id
            self.adapter.cancel(self.ctx)

    def interrupt(self):
        """Pause / L0 freeze while RUNNING: cancel the adapter, re-enter by the resumable policy later."""
        if self.status == Status.RUNNING and not self.interrupted:
            self._cancel()
            self.interrupted = True

    def _reenter(self):
        self.interrupted = False
        kind, to = self.node.f["resumable"]
        if kind == "continue" and hasattr(self.adapter, "resume"):
            self.ctx.current_node = self.node.id
            self.adapter.resume(self.ctx)
            return
        if kind == "rollback":
            self.ctx.current_node = self.node.id
            self.rollback = reg.adapter(to)
            self.rollback.start({}, self.ctx)
            self.ctx.event("rollback", node=self.node.id, to=to)
            return
        self._begin()

    def update(self) -> Status:
        ctx = self.ctx
        if self.suppressed:
            return Status.FAILURE
        if ctx.short_circuit:
            self._cancel()
            return Status.FAILURE
        if self.interrupted:
            self._reenter()
        ctx.current_node = self.node.id
        if self.rollback is not None:
            st = self.rollback.update(ctx)
            if st == stubs.RUNNING:
                return Status.RUNNING
            self.rollback = None
            if st == stubs.FAILURE:
                return self._failed("primitive_failed")
            self._begin()
        if self.backoff is not None:
            if not self.backoff.expired(self.node.f["retry"].get("backoff_s", 0.0)):
                return Status.RUNNING
            self.backoff = None
            self._begin()
        st = self.adapter.update(ctx)
        if st == stubs.SUCCESS:
            ctx.engine._note_success(self.do, self.cur_args)
            return Status.SUCCESS
        if st == stubs.RUNNING and not self.spec.running:
            self._cancel()
            ctx.event("contract_violation", node=self.node.id, do=self.do, detail="RUNNING from a non-RUNNING primitive")
            return self._retry_or_fail("primitive_failed")
        if st == stubs.RUNNING:
            if self.watch.expired(self.node.f.get("timeout_s")):
                self._cancel()
                ctx.event("timeout", node=self.node.id, do=self.do, after_s=round(self.watch.elapsed(), 3))
                return self._retry_or_fail("timeout")
            return Status.RUNNING
        if ctx.short_circuit:          # the adapter itself skipped / halted
            return Status.FAILURE
        return self._retry_or_fail(getattr(self.adapter, "reason", None) or "primitive_failed")

    def _retry_or_fail(self, why: str) -> Status:
        n = self.node.f["retry"].get("n", 0) or 0
        if self.attempt < n:
            self.attempt += 1
            self.ctx.event("retry", node=self.node.id, attempt=self.attempt, why=why)
            self.backoff = Stopwatch(self.ctx.clock)
            if (self.node.f["retry"].get("backoff_s", 0.0) or 0.0) <= 0:
                self.backoff = None
                self._begin()
            return Status.RUNNING
        return self._failed(why)

    def _failed(self, why: str) -> Status:
        reason = self.node.f.get("reason_on_fail") or (why if reasons.is_reason(why) else "primitive_failed")
        self.reason = reason
        self.ctx.fail(reason)
        self.ctx.event("action_failed", node=self.node.id, do=self.do, reason=reason, why=why)
        return Status.FAILURE

    def terminate(self, new_status: Status):
        if new_status == Status.INVALID and self.status == Status.RUNNING and not self.interrupted:
            self._cancel()


class Check(py_trees.behaviour.Behaviour, _Wf):
    """``check`` (instant) and ``wait_for`` (RUNNING until the condition holds or the active timeout)."""

    def __init__(self, node: compiler.Node, ctx: RunContext):
        super().__init__(name=node.label)
        self.node, self.ctx = node, ctx
        self.wait = node.type == "wait_for"
        self.watch: Optional[Stopwatch] = None
        self.assist = None
        self.interrupted = False

    def initialise(self):
        self.watch = Stopwatch(self.ctx.clock)
        self.assist = None

    def interrupt(self):
        if self.status == Status.RUNNING and self.assist is not None:
            self.assist.cancel(self.ctx)
            self.assist = None
        self.interrupted = True

    def update(self) -> Status:
        ctx = self.ctx
        if ctx.short_circuit:
            return Status.FAILURE
        if self.interrupted:
            self.interrupted = False
            self.watch = Stopwatch(ctx.clock)
        if self.assist is not None:
            ctx.current_node = self.node.id
            st = self.assist.update(ctx)
            if st == stubs.RUNNING:
                return Status.RUNNING
            return _ST[st]
        if ctx.cond(self.node.f["cond"]):
            return Status.SUCCESS
        if self.wait and not self.watch.expired(self.node.f.get("timeout_s")):
            return Status.RUNNING
        kind, arg = self.node.f["else" if not self.wait else "on_timeout"]
        reason = self.node.f.get("reason") or ("timeout" if self.wait else "check_failed")
        if kind == "skip":
            ctx.skip_target(arg)
            return Status.FAILURE
        if kind == "assist":
            ctx.current_node = self.node.id
            ctx.last_reason = reason
            self.assist = reg.adapter("assist_request")
            self.assist.start({"reason": reason}, ctx)
            return Status.RUNNING
        ctx.fail(reason)
        ctx.event("check_failed", node=self.node.id, cond=self.node.f["cond"], reason=reason)
        return Status.FAILURE


class Atomic(py_trees.composites.Sequence, _Wf):
    """Non-interruptible completion block (A.2): requests are queued while it is open; only L0 breaks it."""

    def __init__(self, node, ctx, children):
        super().__init__(name=node.label or "atomic", memory=True, children=children)
        self.node, self.ctx = node, ctx
        self.open = False
        self.broken: Optional[str] = None

    def initialise(self):
        if not self.open:
            self.open = True
            self.ctx.atomic_depth += 1

    def _close(self):
        if self.open:
            self.open = False
            self.ctx.atomic_depth -= 1

    def terminate(self, new_status):
        self._close()

    def tick(self):
        if self.broken is not None:
            outcome, self.broken = self.broken, None
            self._close()
            for c in self.children:
                if c.status != Status.INVALID:
                    c.stop(Status.INVALID)
            if self.ctx.frames:
                self.ctx.skip_target(outcome)
            else:
                self.ctx.fail(outcome)
            self.stop(Status.FAILURE)
            yield self
            return
        yield from super().tick()


class Guard(py_trees.decorators.Decorator, _Wf):
    """Reactive guard (A.2): the condition is evaluated every tick BEFORE the child; only tightens L0/L1."""

    def __init__(self, node, ctx, child):
        super().__init__(name=node.label, child=child)
        self.node, self.ctx = node, ctx
        self.violated = False
        self.stow = None

    def initialise(self):
        self.violated = False
        self.stow = None

    def update(self):
        return self.decorated.status

    def tick(self):
        ctx = self.ctx
        if self.status != Status.RUNNING:
            self.initialise()
        ok = ctx.cond(self.node.f["cond"])
        if not ok and not self.violated:
            self.violated = True
            how = self.node.f["on_violation"]
            reason = self.node.f.get("reason") or _COND_REASON.get(self.node.f["cond"], "guard_violation")
            ctx.event("guard_violation", node=self.node.id, cond=self.node.f["cond"], on_violation=how)
            if how == "skip_target" and not ctx.atomic_depth:
                ctx.skip_target(reason)
                self.stop(Status.FAILURE)
                yield self
                return
            if how == "stow_continue":
                ctx.current_node = self.node.id
                self.stow = reg.adapter("stow_mast")
                self.stow.start({}, ctx)
            else:
                ctx.request({"halt_latch": "halt", "skip_target": "skip"}.get(how, how), reason, source="guard")
        elif ok:
            self.violated = False
        if self.stow is not None:
            ctx.current_node = self.node.id
            stow_st = self.stow.update(ctx)
            if stow_st == stubs.RUNNING:
                self.status = Status.RUNNING
                yield self
                return
            self.stow = None
            if stow_st == stubs.SUCCESS:
                ctx.engine._note_success("stow_mast", {})
        for n in self.decorated.tick():
            yield n
        new = self.decorated.status
        if new != Status.RUNNING:
            self.stop(new)
        self.status = new
        yield self


_COND_REASON = {"person_warning_clear": "person_hold", "wind_lt_crown": "wind", "rain_dry": "rain",
                "visibility_ok": "low_visibility", "trafficable": "not_trafficable", "crown_gate": "crown_gate",
                "cloud_link_up": "cloud_link_lost", "heat_ok": "heat", "stability": "stability"}


class _Composite(py_trees.composites.Composite, _Wf):
    def stop(self, new_status=Status.INVALID):
        for c in self.children:
            if c.status == Status.RUNNING:
                c.stop(Status.INVALID)
        py_trees.composites.Composite.stop(self, new_status)


class Foreach(_Composite):
    """Iterate a target set (A.2): index, completed / skipped ids, checkpoint per target, re-plan on resume."""

    def __init__(self, node, ctx, body):
        super().__init__(name=node.label or node.f["set"], children=[body])
        self.node, self.ctx = node, ctx
        self.body = body
        self.total = 0
        self.pending: List[str] = []
        self.completed: List[str] = []
        self.skipped: Dict[str, str] = {}
        self.current: Optional[str] = None
        self.replan = False
        self.restored = False
        self.halting = False

    def _plan(self, keep_current: bool = False):
        """(Re-)plan the target set, dropping completed and skipped ids. On an in-process resume the target in
        progress keeps its place and its body re-enters by each node's resumable policy (A.4 step 7); if the
        re-plan no longer contains it, it is dropped and the body restarts on the next target."""
        planned = self.ctx.plan_targets(self.node.f["set"])
        done = set(self.completed) | set(self.skipped)
        if not (keep_current and self.current in planned):
            if self.current is not None and self.body.status != Status.INVALID:
                self.body.stop(Status.INVALID)
            self.current = None
        self.pending = [t for t in planned if t not in done and t != self.current]
        if not self.restored or not self.total:
            self.total = len(planned)

    def state(self) -> dict:
        return {"index": len(self.completed) + len(self.skipped), "total": self.total, "current": self.current,
                "completed": list(self.completed), "skipped": dict(self.skipped)}

    def restore(self, st: dict):
        self.completed = list(st.get("completed", []))
        self.skipped = dict(st.get("skipped", {}))
        self.total = int(st.get("total", 0))
        self.restored = True
        self.replan = True

    def tick(self):
        ctx = self.ctx
        if self.status != Status.RUNNING:
            if not self.restored:
                self.completed, self.skipped = [], {}
            self.halting = False
            self._plan()
            self.restored = False
            self.replan = False
            self.initialise()
        if self.replan:
            self._plan(keep_current=True)
            self.replan = False
            ctx.event("replan", node=self.node.id, pending=list(self.pending))
        if self.halting:
            self.status = Status.RUNNING
            yield self
            return
        if self.current is None:
            if not self.pending:
                self.stop(Status.SUCCESS)
                yield self
                return
            self.current = self.pending.pop(0)
            ctx.bb.set("target", self.current)
            ctx.bb.set("target_index", len(self.completed) + len(self.skipped))
            if self.body.status != Status.INVALID:
                self.body.stop(Status.INVALID)
        ctx.frames.append(self)
        try:
            for n in self.body.tick():
                yield n
        finally:
            ctx.frames.pop()
        st = self.body.status
        if st == Status.RUNNING:
            self.status = Status.RUNNING
            yield self
            return
        t, self.current = self.current, None
        if ctx.halt_pending:
            self.stop(Status.FAILURE)
            yield self
            return
        if ctx.skip_pending is not None:
            reason, ctx.skip_pending = ctx.skip_pending, None
            self.skipped[t] = reason
            ctx.event("target_skipped", node=self.node.id, target=t, reason=reason)
            ratio = self.node.f.get("skip_ratio_halt")
            if ratio and self.total and len(self.skipped) > ratio * self.total:
                ctx.event("skip_ratio_halt", node=self.node.id, skipped=len(self.skipped), total=self.total)
                ctx.request("rth", "rth_skip_ratio", source="workflow")
                self.halting = True
            self.ctx.engine._target_done(self)
        elif st == Status.SUCCESS:
            self.completed.append(t)
            ctx.event("target_done", node=self.node.id, target=t)
            self.ctx.engine._target_done(self)
        else:
            self.stop(Status.FAILURE)
            yield self
            return
        self.status = Status.RUNNING
        yield self


class Recover(_Composite):
    """``recover {body, recovery[], max_rounds, then}`` (A.2)."""

    def __init__(self, node, ctx, body, rec):
        super().__init__(name=node.label or "recover", children=[body, rec])
        self.node, self.ctx = node, ctx
        self.body, self.rec = body, rec
        self.round = 1
        self.phase = "body"

    def tick(self):
        ctx = self.ctx
        if self.status != Status.RUNNING:
            for c in self.children:
                if c.status != Status.INVALID:
                    c.stop(Status.INVALID)
            self.round, self.phase = 1, "body"
            self.initialise()
        child = self.body if self.phase == "body" else self.rec
        for n in child.tick():
            yield n
        st = child.status
        if st == Status.RUNNING:
            self.status = Status.RUNNING
            yield self
            return
        if self.phase == "body":
            if st == Status.SUCCESS:
                self.stop(Status.SUCCESS)
                yield self
                return
            if ctx.short_circuit:
                self.stop(Status.FAILURE)
                yield self
                return
            if self.round < int(self.node.f.get("max_rounds", 2)):
                self.round += 1
                ctx.event("recover_round", node=self.node.id, round=self.round)
                if self.rec.children:
                    self.phase = "rec"
                    self.rec.stop(Status.INVALID)
                else:
                    self.body.stop(Status.INVALID)
                self.status = Status.RUNNING
                yield self
                return
            kind, arg = self.node.f["then"]
            if kind == "continue":
                self.stop(Status.SUCCESS)
            elif kind == "skip":
                ctx.skip_target(arg)
                self.stop(Status.FAILURE)
            else:
                self.stop(Status.FAILURE)
            yield self
            return
        if ctx.short_circuit:
            self.stop(Status.FAILURE)
            yield self
            return
        self.phase = "body"
        self.body.stop(Status.INVALID)
        self.status = Status.RUNNING
        yield self


class Subtree(py_trees.decorators.PassThrough, _Wf):
    def __init__(self, node, ctx, child):
        super().__init__(name=node.label or node.f["ref"], child=child)
        self.node, self.ctx = node, ctx


class _Seq(py_trees.composites.Sequence, _Wf):
    pass


class _Sel(py_trees.composites.Selector, _Wf):
    pass


class _Par(py_trees.composites.Parallel, _Wf):
    pass


def build(node: compiler.Node, ctx: RunContext):
    """IR -> py_trees behaviour tree."""
    t = node.type
    if t == "action":
        b = Action(node, ctx)
    elif t in ("check", "wait_for"):
        b = Check(node, ctx)
    else:
        kids = [build(c, ctx) for c in node.children]
        if t == "sequence":
            b = _Seq(name=node.label or "sequence", memory=node.f.get("memory", True), children=kids)
        elif t == "selector":
            b = _Sel(name=node.label or "selector", memory=node.f.get("memory", False), children=kids)
        elif t == "parallel":
            pol = (py_trees.common.ParallelPolicy.SuccessOnOne() if node.f.get("policy") == "one"
                   else py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=True))
            b = _Par(name=node.label or "parallel", policy=pol, children=kids)
        elif t == "atomic":
            b = Atomic(node, ctx, kids)
        elif t == "guard":
            b = Guard(node, ctx, kids[0])
        elif t == "foreach":
            b = Foreach(node, ctx, kids[0])
        elif t == "recover":
            b = Recover(node, ctx, kids[0], kids[1])
        elif t == "subtree":
            b = Subtree(node, ctx, kids[0])
        else:                                                          # pragma: no cover
            raise ValueError(f"unknown IR node {t!r}")
        b.node, b.ctx = node, ctx
    return b


def _leaves(root):
    for b in root.iterate():
        if isinstance(b, (Action, Check)):
            yield b


# ---------------------------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, comp: compiler.Compiled, run_id: str = "run1", params: Optional[dict] = None,
                 world: Optional[stubs.StubWorld] = None, library: Optional[compiler.Library] = None,
                 dt: float = DT, debug: bool = False, auto_continue: bool = False,
                 ack_after_pause_s: float = ACK_AFTER_PAUSE_S, checkpoint_store: bool = True):
        self.comp = comp
        self.library = library or compiler.default_library()
        self.run_id = str(run_id)
        self.dt = float(dt)
        self.debug = debug
        self.auto_continue = auto_continue
        self.ack_after_pause_s = ack_after_pause_s
        self.use_store = checkpoint_store
        self.world = world if world is not None else stubs.StubWorld()
        self.params = self._params(params or {})
        self.bb = blackboard.RunBlackboard(self.run_id, debug=debug)
        self.clock = ActiveClock()
        self.ctx = RunContext(self)
        self.root = build(comp.root, self.ctx)
        self.state = "idle"
        self.reason: Optional[str] = None
        self.acked = False
        self.requests: List[dict] = []
        self.events: List[dict] = []
        self.on_event: Optional[Callable[[dict], None]] = None
        self.fin = None
        self.fin_clock = ActiveClock()
        self.fin_next = ""
        self.fin_kind = ""
        self.paused_at: Optional[float] = None
        self.last_interrupt: Optional[dict] = None
        self.resume_target: Optional[List[float]] = None
        self.last_safe: Optional[List[float]] = list(self.world.pose)
        self.cp_seq = 0
        self.ticks = 0
        self.outcomes: List[dict] = []
        self.l0_reason: Optional[str] = None
        self.l0_prev: Optional[tuple] = None          # (state, fin_kind, fin_next, reason) when L0 froze the run
        self.fin_chain: Optional[tuple] = None        # (reason, next) of an abort / halt raised while pausing
        self.fin_held = False                         # finaliser held by the L1 person pause
        self.person_present = False
        self._resume_ticked = False
        self._gap_unknown = False                     # restored from a checkpoint without a known pause length
        # posture the engine saw primitives establish (A.4 step 5): mast {do, args} | None, work zone, tool
        self.posture: Dict[str, object] = {"mast": None, "work": False, "tool": None}
        self.posture_at_pause: Optional[dict] = None
        self._waiting_ack = False
        self._check_param_uses()

    # -------------------------------------------------------------- params (console: within min/max only)
    def _params(self, given: dict) -> dict:
        out = dict(self.comp.params)
        for k, v in given.items():
            spec = self.comp.param_specs.get(k)
            if spec is None:
                raise ValueError(f"{self.comp.id} has no param {k!r}")
            num = isinstance(v, (int, float)) and not isinstance(v, bool)
            default = spec.get("default")
            numeric_param = (spec.get("min") is not None or spec.get("max") is not None
                             or (isinstance(default, (int, float)) and not isinstance(default, bool)))
            if numeric_param and not num:
                raise ValueError(f"param {k}={v!r} must be a number")
            if not numeric_param and default is not None and spec.get("enum") is None \
                    and type(v) is not type(default):
                raise ValueError(f"param {k}={v!r} must be a {type(default).__name__} like its default")
            if num:
                if spec.get("min") is not None and v < spec["min"] or spec.get("max") is not None and v > spec["max"]:
                    raise ValueError(f"param {k}={v} outside [{spec.get('min')}, {spec.get('max')}]")
            if spec.get("enum") is not None and v not in spec["enum"]:
                raise ValueError(f"param {k}={v!r} not in {spec['enum']}")
            out[k] = v
        return out

    def _check_param_uses(self) -> None:
        """Every ``$param`` an action uses must, with its value for this run, satisfy the primitive's ArgSpec
        (type, catalogue range, enum): the console may set params within min / max only (A.7), and a param
        without bounds must not carry a value past the catalogue range into an adapter."""
        for top in (self.comp.root, self.comp.on_pause, self.comp.on_abort):
            if top is None:
                continue
            for n in top.walk():
                if n.type != "action" or n.f["do"] not in reg.PRIMITIVES:
                    continue
                spec = reg.PRIMITIVES[n.f["do"]]
                for a, v in n.f["args"].items():
                    if not (isinstance(v, str) and v.startswith("$") and v[1:] in self.params) or a not in spec.args:
                        continue
                    why = reg.check_value(spec.args[a], self.params[v[1:]])
                    if why is not None:
                        raise ValueError(f"param {v[1:]}={self.params[v[1:]]!r} for {n.f['do']}.{a} at {n.id}: {why}")

    # -------------------------------------------------------------- posture (A.4 step 5)
    def _note_success(self, do: str, args: dict) -> None:
        p = self.posture
        if do in ("raise_mast", "mast_snake"):
            p["mast"] = {"do": do, "args": {k: v for k, v in args.items() if k in reg.PRIMITIVES["raise_mast"].args}}
        elif do == "stow_mast":
            p["mast"] = None
        elif do == "enter_work":
            p["work"] = True
        elif do in ("exit_work", "tools_safe"):
            p["work"] = False
        elif do == "tool_change":
            p["tool"] = args.get("tool")

    def _restore_steps(self) -> list:
        """YAML steps that bring the posture back to the one recorded at the pause (after the drift gate):
        tool first (only while stowed and empty), then the mast behind a crown gate check, then the work zone.
        The arm is not restored: pre-commit contact primitives roll back to ``arm_park`` (A.4 step 7)."""
        want, cur = self.posture_at_pause, self.posture
        if not want:
            return []
        steps = []
        if want.get("tool") is not None and cur.get("tool") != want["tool"]:
            steps.append({"action": {"do": "tool_change", "args": {"tool": want["tool"]}, "timeout_s": 30,
                                     "labels": {"en": "restore tool", "hi": "औज़ार वापस"}}})
        if want.get("mast") is not None and cur.get("mast") is None:
            steps.append({"check": {"cond": "crown_gate", "else": "fail", "reason": "crown_gate",
                                    "labels": {"en": "crown gate", "hi": "क्राउन जाँच"}}})
            steps.append({"action": {"do": "raise_mast", "args": dict(want["mast"].get("args") or {}),
                                     "timeout_s": 12, "labels": {"en": "restore mast", "hi": "मस्तूल वापस"}}})
        if want.get("work") and not cur.get("work"):
            steps.append({"action": {"do": "enter_work", "timeout_s": 5,
                                     "labels": {"en": "restore work zone", "hi": "कार्य क्षेत्र वापस"}}})
        return steps

    # -------------------------------------------------------------- events
    def event(self, kind: str, **kw) -> None:
        e = {"t": round(self.clock.total, 3), "active": round(self.clock.active, 3), "kind": kind, **kw}
        self.events.append(e)
        if self.on_event is not None:
            self.on_event(e)

    def events_of(self, kind: str) -> List[dict]:
        return [e for e in self.events if e["kind"] == kind]

    # -------------------------------------------------------------- operator commands (via the WP38 supervisor)
    def start(self) -> None:
        if self.state != "idle":
            raise RuntimeError(f"start in state {self.state}")
        self.state = "running"
        self.event("start", workflow=self.comp.ident, params=self.params)

    def ack(self, source: str = "handheld") -> None:
        self.acked = True
        self._waiting_ack = False
        self.event("ack", source=source)

    def pause(self, reason: str = "operator_pause") -> None:
        self._request("pause", reason, "operator")

    def abort(self, reason: str = "operator_abort") -> None:
        self._request("abort", reason, "operator")

    def rth(self, reason: str = "rth_operator") -> None:
        self._request("rth", reason, "operator")

    def skip_target(self, reason: str = "operator_skip") -> None:
        self._request("skip", reason, "operator")

    def resume(self) -> None:
        if self.state != "paused":
            raise RuntimeError(f"resume in state {self.state}")
        # a restored checkpoint with no known pause length counts as long: RTH, dock and charge take > 15 min
        long_pause = self._gap_unknown or (
            self.paused_at is not None and self.clock.total - self.paused_at > self.ack_after_pause_s)
        self.acked = self.acked and not long_pause
        self._waiting_ack = long_pause and not self.auto_continue
        self.resume_target = self.last_safe
        self._resume_ticked = False
        if not self.ctx.started:                        # paused before start_ack: nothing moved, nothing to check
            self.fin, self.fin_kind, self.fin_next = None, "resume", "running"
            self.state = "resuming"
            self.event("resume_requested", long_pause=long_pause, needs_ack=self._waiting_ack)
            return
        restore = self._restore_steps()
        if restore:
            node = compiler.compile_node({"sequence": [{"subtree": {"ref": compiler.RESUME_CHECK}}] + restore},
                                         self.library, "/resume_check")
        else:
            node = compiler.compile_node({"ref": compiler.RESUME_CHECK}, self.library, "/resume_check")
        self.fin = build(node, self.ctx)
        self.fin_clock = ActiveClock()
        self.fin_kind, self.fin_next = "resume", "running"
        self.state = "resuming"
        self.event("resume_requested", long_pause=long_pause, needs_ack=self._waiting_ack)

    def _request(self, kind: str, reason: str, source: str) -> None:
        reasons.check(reason)
        if kind not in REQUEST_PRIORITY:
            raise ValueError(f"unknown request {kind!r}")
        self.requests.append({"kind": kind, "reason": reason, "source": source})
        self.event("request", request=kind, reason=reason, source=source, queued=self.ctx.atomic_depth > 0)

    # -------------------------------------------------------------- L0 / L1 (never YAML)
    def _mirror(self, signals: Optional[dict]) -> None:
        values = self.world.signal_values(self) if signals is None else signals
        blackboard.mirror_values(values)

    def _l0_l1(self) -> bool:
        """Apply the e-stop, protective / local-link freeze and the person pause. False = do not tick the tree.

        The e-stop latches in every non-terminal state but ``idle``, ``paused`` included (the recovery
        checklist is never skipped). The person pause covers every tree the engine ticks (A.0 rule 1): in
        ``running`` it queues a ``person_hold`` pause; in ``resuming`` it cancels the resume check (the base
        does not drive back with a person in the warning field) and runs ``on_pause``; a finaliser
        (``pausing``, ``aborting``, ``finalising``) is held, its running leaves cancelled, until the field is
        clear, then re-enters by each leaf's resumable policy."""
        c = self.ctx.cond
        if self.state == "idle" or self.state in reasons.TERMINAL:
            return True
        if not c("estop_clear"):
            self._estop()
            return False
        self.person_present = not c("person_warning_clear")
        if self.state == "paused":                    # stationary: only the e-stop matters until a command
            return True
        frozen = None
        if not c("protective_clear"):
            frozen = "protective_stop"
        elif not c("local_link") and self.ctx.started:
            frozen = "local_link_lost"
        if frozen is not None:
            if self.state != "l0_hold":
                self._freeze(frozen)
            return False
        if self.state == "l0_hold":
            self._l0_clear()
            return False
        if self.person_present and self.state == "running":
            if not any(r["reason"] == "person_hold" for r in self.requests):
                self._request("pause", "person_hold", "L1")
        elif self.person_present and self.state == "resuming":
            self.event("person_hold", state="resuming")
            self._cancel_resume("pause", "person_hold")
        return True

    def _break_atomics(self, outcome_held: str) -> None:
        for b in self.root.iterate():
            if isinstance(b, Atomic) and b.open:
                outcome = "dropped" if (self.world.holding and self.world.estop_drops and outcome_held == "estop") \
                    else ("held_at_estop" if self.world.holding else "dropped")
                b.broken = outcome
                self.outcomes.append({"node": b.node.id, "outcome": outcome})
                self.event("atomic_interrupted_l0", node=b.node.id, outcome=outcome)
                if outcome == "dropped" and self.world.holding:
                    self.world.holding = False
                    self.world.dropped += 1

    def _estop(self) -> None:
        self.event("estop", state=self.state)
        self._break_atomics("estop")
        self._interrupt_leaves()
        self.root.stop(Status.INVALID)
        if self.fin is not None:
            self.fin.stop(Status.INVALID)
            self.fin = None
        self.requests = []
        self._end("estopped", "estop")

    def _freeze(self, reason: str) -> None:
        self.event("l0_hold", reason=reason, state=self.state)
        # what the run was doing: an abort / failure / RTH finaliser resumes as such after the hold, it never
        # becomes a resumable ``paused``
        self.l0_prev = (self.state, self.fin_kind, self.fin_next, self.reason)
        self._break_atomics("protective")
        self._interrupt_leaves()
        self._stop_fin()
        self.l0_reason = reason
        self.clock.freeze()
        self.state = "l0_hold"

    def _l0_clear(self) -> None:
        self.event("l0_clear", reason=self.l0_reason)
        prev, kind, nxt, reason = self.l0_prev or ("running", "", "", None)
        self.l0_prev = None
        if prev in ("aborting", "finalising"):
            self._begin_finaliser("abort", reason, nxt)
        elif prev == "pausing" and nxt != "paused":
            self._begin_finaliser("pause", reason, nxt, supervisor=nxt)
        else:                                         # running, resuming, pausing -> paused
            self._begin_finaliser("pause", self.l0_reason, "paused", snapshot=prev == "running")

    def _stop_fin(self) -> None:
        if self.fin is not None:
            for leaf in _leaves(self.fin):
                if leaf.status == Status.RUNNING:
                    leaf.interrupt()
            self.fin.stop(Status.INVALID)
            self.fin = None
        self.fin_held = False

    # -------------------------------------------------------------- requests
    def _take_request(self, kinds) -> Optional[dict]:
        """Pop the highest-priority request among ``kinds``; others of those kinds are dropped, the rest stay."""
        mine = [r for r in self.requests if r["kind"] in kinds]
        if not mine:
            return None
        req = max(mine, key=lambda r: REQUEST_PRIORITY[r["kind"]])
        self.requests = [r for r in self.requests if r["kind"] not in kinds]
        self.last_interrupt = dict(req)
        self.event("apply_request", request=req["kind"], reason=req["reason"], source=req["source"],
                   state=self.state)
        return req

    def _apply_requests(self) -> None:
        if not self.requests:
            return
        if self.state == "running":
            self._apply_running()
        elif self.state == "paused":
            self._apply_paused()
        elif self.state == "resuming":
            self._apply_resuming()
        elif self.state == "pausing":
            self._apply_pausing()
        elif self.state in ("aborting", "finalising"):
            for r in self.requests:
                self.event("request_dropped", request=r["kind"], reason=r["reason"], state=self.state)
            self.requests = []

    def _apply_running(self) -> None:
        if self.ctx.atomic_depth > 0:
            return
        req = self._take_request(REQUEST_PRIORITY)
        kind, reason = req["kind"], req["reason"]
        if kind == "skip":
            if self.ctx.frames or self._innermost_foreach() is not None:
                self.ctx.skip_target(reason)
            return
        if kind in ("abort", "halt"):
            self._interrupt_leaves()
            self.root.stop(Status.INVALID)
            self._begin_finaliser("abort", reason, "aborted" if kind == "abort" else "halted")
            return
        # pause, rth, safe_park: checkpoint, on_pause finaliser, then paused (rth: terminal rth for L1)
        self._interrupt_leaves()
        nxt = {"pause": "paused", "safe_park": "paused", "rth": "rth"}[kind]
        self._begin_finaliser("pause", reason, nxt, supervisor=kind, snapshot=True)

    def _apply_paused(self) -> None:
        """Paused (on_pause done, nothing moving): abort / halt run on_abort; rth / safe_park run on_pause
        again (idempotent) with an RTH checkpoint; a pause is a no-op; a skip waits for the resume."""
        req = self._take_request(("abort", "halt", "rth", "safe_park", "pause"))
        if req is None:
            return
        kind, reason = req["kind"], req["reason"]
        if kind in ("abort", "halt"):
            self.root.stop(Status.INVALID)
            self._begin_finaliser("abort", reason, "aborted" if kind == "abort" else "halted")
        elif kind in ("rth", "safe_park"):
            self._begin_finaliser("pause", reason, "rth" if kind == "rth" else "paused", supervisor=kind)

    def _apply_resuming(self) -> None:
        req = self._take_request(("abort", "halt", "rth", "safe_park", "pause"))
        if req is not None:
            self._cancel_resume(req["kind"], req["reason"])

    def _cancel_resume(self, kind: str, reason: str) -> None:
        """Stop the resume check (and any posture restore) and do ``kind`` instead."""
        began = self.fin is not None and self._resume_ticked
        self._stop_fin()
        self._waiting_ack = False
        self.event("resume_cancelled", request=kind, reason=reason)
        if kind in ("abort", "halt"):
            self.root.stop(Status.INVALID)
            self._begin_finaliser("abort", reason, "aborted" if kind == "abort" else "halted")
        elif kind == "rth":
            self._begin_finaliser("pause", reason, "rth", supervisor="rth")
        elif began:                                   # the base or the mast may have moved: make safe again
            self._begin_finaliser("pause", reason, "paused", supervisor=kind)
        else:                                         # nothing moved since on_pause
            self.state = "paused"
            self.reason = reason
            self.clock.freeze()

    def _apply_pausing(self) -> None:
        """on_pause is running: let it finish (it is the safe finaliser), then abort / halt chain on_abort;
        rth turns the end state into terminal ``rth`` (with an RTH checkpoint); pause / safe_park are no-ops."""
        req = self._take_request(("abort", "halt", "rth", "safe_park", "pause"))
        if req is None:
            return
        kind, reason = req["kind"], req["reason"]
        if kind in ("abort", "halt"):
            self.fin_chain = (reason, "aborted" if kind == "abort" else "halted")
        elif kind == "rth" and self.fin_next == "paused":
            self.fin_next = "rth"
            self.reason = reason
            self.write_checkpoint(reason, "rth")

    def _innermost_foreach(self):
        for b in self.root.iterate():
            if isinstance(b, Foreach) and b.status == Status.RUNNING and b.current is not None:
                return b
        return None

    def _interrupt_leaves(self) -> None:
        for leaf in _leaves(self.root):
            if leaf.status == Status.RUNNING:
                leaf.interrupt()

    def _begin_finaliser(self, which: str, reason: Optional[str], nxt: str, supervisor: Optional[str] = None,
                         snapshot: bool = False):
        self.reason = reason
        if which == "pause":
            if snapshot or self.posture_at_pause is None:
                self.posture_at_pause = copy.deepcopy(self.posture)
            self.write_checkpoint(reason, supervisor or "pause")
        else:
            self.fin_chain = None
        top = self.comp.on_pause if which == "pause" else self.comp.on_abort
        self.clock.freeze()
        self.fin = build(top, self.ctx) if top is not None else None
        self.fin_held = False
        self.fin_clock = ActiveClock()
        self.fin_kind, self.fin_next = which, nxt
        self.state = "pausing" if which == "pause" else ("aborting" if nxt == "aborted" else "finalising")
        self.event("finaliser_start", which=which, reason=reason, next=nxt)

    # -------------------------------------------------------------- tick
    def tick(self, signals: Optional[dict] = None, l0: Optional[str] = None) -> str:
        """One 10 Hz tick. ``signals`` (cond id or source key -> value) replaces the stub world mirror; ``l0`` is
        the runner / driver safe-hold reason (``bus.values["hold"]``): ``estop`` latches the e-stop, any other
        reason freezes like a protective stop."""
        self.ticks += 1
        values = dict(self.world.signal_values(self) if signals is None else signals)
        if l0 is not None:
            values["estop_clear" if l0 == "estop" else "protective_clear"] = False
        self._mirror(values)
        tree_running = self.state == "running"
        self.clock.advance(self.dt, active=tree_running)
        if not self._l0_l1():
            return self.state
        self._apply_requests()
        if self.state == "running":
            self.clock.unfreeze()
            self.ctx.clock = self.clock
            self.root.tick_once()
            st = self.root.status
            if self.ctx.halt_pending and st != Status.RUNNING:
                reason, self.ctx.halt_pending = self.ctx.halt_pending, None
                self._begin_finaliser("abort", reason, "halted")
            elif st == Status.SUCCESS:
                self._end("succeeded", None)
            elif st == Status.FAILURE:
                self._begin_finaliser("abort", self.ctx.last_reason or "primitive_failed", "failed")
            else:
                self._apply_requests()
        elif self.state in ("pausing", "aborting", "finalising", "resuming"):
            self._tick_finaliser()
        return self.state

    def _tick_finaliser(self) -> None:
        if self.state == "resuming" and self._waiting_ack:
            return
        if self.fin is None:
            self._finish_finaliser(Status.SUCCESS)
            return
        if self.person_present:                       # L1 person pause: hold the finaliser, no arm / mast motion
            if not self.fin_held:
                self.fin_held = True
                for leaf in _leaves(self.fin):
                    if leaf.status == Status.RUNNING:
                        leaf.interrupt()
                self.event("finaliser_person_hold", which=self.fin_kind)
            return
        if self.fin_held:
            self.fin_held = False
            self.event("finaliser_person_clear", which=self.fin_kind)
        if self.fin_kind == "resume":
            self._resume_ticked = True
        self.fin_clock.advance(self.dt)
        self.ctx.clock = self.fin_clock
        saved_skip, saved_halt = self.ctx.skip_pending, self.ctx.halt_pending
        self.ctx.skip_pending = self.ctx.halt_pending = None
        try:
            self.fin.tick_once()
        finally:
            if self.fin_kind != "resume":
                self.ctx.skip_pending, self.ctx.halt_pending = saved_skip, saved_halt
            self.ctx.clock = self.clock
        if self.fin.status != Status.RUNNING:
            self._finish_finaliser(self.fin.status)

    def _finish_finaliser(self, st: Status) -> None:
        self.event("finaliser_end", which=self.fin_kind, status=st.name.lower(), next=self.fin_next)
        self.fin = None
        if self.fin_kind == "resume":
            if st == Status.SUCCESS:
                self.state = "running"
                self.reason = None
                self._gap_unknown = False
                self.posture_at_pause = None
                self.clock.unfreeze()
                for b in self.root.iterate():
                    if isinstance(b, Foreach):
                        b.replan = True
                self.event("resumed")
            else:
                reason = self.ctx.last_reason or "not_resumable"
                self.event("resume_refused", reason=reason)
                if self._resume_ticked:               # the check may have moved the base or the mast
                    self._begin_finaliser("pause", reason, "paused", supervisor="pause")
                else:
                    self.state = "paused"
                    self.reason = reason
            return
        if self.fin_kind == "pause" and self.fin_chain is not None:
            reason, nxt = self.fin_chain
            self.fin_chain = None
            self.root.stop(Status.INVALID)
            self._begin_finaliser("abort", reason, nxt)
            return
        if self.fin_next == "paused":
            self.state = "paused"
            self.paused_at = self.clock.total
            return
        self._end(self.fin_next, self.reason)

    def _end(self, terminal: str, reason: Optional[str]) -> None:
        self.state = terminal
        self.reason = reason
        self.event("end", state=terminal, reason=reason, on_complete=self.comp.doc.get("on_complete", "rth"))

    # -------------------------------------------------------------- run helpers
    @property
    def done(self) -> bool:
        return self.state in reasons.TERMINAL

    def run(self, max_ticks: int = 10000, until: Optional[Callable[["Engine"], bool]] = None,
            auto_ack: bool = True) -> str:
        if self.state == "idle":
            self.start()
        for _ in range(max_ticks):
            if auto_ack and not self.acked and (self.state == "running" or self._waiting_ack):
                self.ack("test")
            self.tick()
            if self.done or self.state == "paused" or (until is not None and until(self)):
                break
        return self.state

    # -------------------------------------------------------------- checkpoints (A.4)
    def _target_done(self, fe: Foreach) -> None:
        w = self.world
        if not w.holding and w.arm == "parked":
            self.last_safe = list(w.pose)
        self.write_checkpoint(None, "running")

    def _node_path(self) -> List[int]:
        path, b = [], self.root
        while isinstance(b, py_trees.composites.Sequence) and not isinstance(b, Atomic) and b.status == Status.RUNNING:
            if b.current_child is None:
                break
            i = b.children.index(b.current_child)
            path.append(i)
            b = b.current_child
            while isinstance(b, Subtree):
                b = b.decorated
        return path

    def foreach_nodes(self) -> Dict[str, Foreach]:
        return {b.node.id: b for b in self.root.iterate() if isinstance(b, Foreach)}

    def write_checkpoint(self, reason: Optional[str], supervisor: str) -> checkpoint.Checkpoint:
        self.cp_seq += 1
        fe = self.foreach_nodes()
        cp = checkpoint.Checkpoint(
            workflow=self.comp.ident, run_id=self.run_id, node_path=self._node_path(),
            foreach={k: v.state() for k, v in fe.items()},
            completed_targets=sorted(itertools.chain.from_iterable(v.completed for v in fe.values())),
            last_safe=[float(x) for x in self.last_safe] if self.last_safe is not None else None,
            wf=self.bb.snapshot(), params=dict(self.params), reason=reason, supervisor=supervisor,
            clock=self.clock.to_dict(), seq=self.cp_seq,
            posture=copy.deepcopy(self.posture_at_pause if supervisor != "running" and self.posture_at_pause
                                  else self.posture),
            paused_at=None if supervisor == "running" else self.clock.total)
        if self.use_store:
            checkpoint.save(cp)
        self.event("checkpoint", seq=self.cp_seq, reason=reason, supervisor=supervisor)
        return cp

    @classmethod
    def from_checkpoint(cls, cp: checkpoint.Checkpoint, library: Optional[compiler.Library] = None,
                        run_id: Optional[str] = None, elapsed_s: Optional[float] = None, **kw) -> "Engine":
        """A paused engine restored from ``cp``; refuses ``workflow_changed`` if the INDEX entry moved on and
        ``not_resumable`` if the checkpoint does not fit the compiled workflow.

        ``elapsed_s``: seconds since the checkpoint's pause, if the caller (WP38) knows them. Unknown (the
        default) counts as a long pause, so ``resume`` waits for an operator ack unless ``auto_continue``."""
        if not isinstance(cp, checkpoint.Checkpoint):
            raise ResumeRefused("not_resumable", "not a checkpoint")
        try:
            checkpoint.Checkpoint.from_dict(cp.to_dict())
        except checkpoint.CheckpointError as exc:
            raise ResumeRefused("not_resumable", str(exc)) from None
        lib = library or compiler.default_library()
        ref = cp.workflow.split("#", 1)[0]
        try:
            entry = lib.get(ref)
        except KeyError:
            raise ResumeRefused("workflow_changed", f"{ref} is no longer in the library") from None
        latest = lib.latest(entry.id)
        if entry.ident != cp.workflow or latest.version != entry.version:
            raise ResumeRefused("workflow_changed", f"checkpoint {cp.workflow} != current {latest.ident}: "
                                                    "offer restart from the next target")
        comp = compiler.compile_doc(entry.doc(), lib, entry.data())
        try:
            eng = cls(comp, run_id=run_id or cp.run_id, params=cp.params, library=lib, **kw)
        except ValueError as exc:
            raise ResumeRefused("not_resumable", f"checkpoint params: {exc}") from None
        try:
            eng.bb.restore(cp.wf)
            fe = eng.foreach_nodes()
            for nid, st in cp.foreach.items():
                if nid in fe:
                    fe[nid].restore(st)
        except (ValueError, TypeError, KeyError, PermissionError) as exc:
            eng.close()
            raise ResumeRefused("not_resumable", f"checkpoint does not fit {comp.ident}: {exc}") from None
        b = eng.root
        for i in cp.node_path:
            if not isinstance(b, py_trees.composites.Sequence) or i >= len(b.children):
                break
            b.status = Status.RUNNING
            b.current_child = b.children[i]
            b = b.children[i]
            while isinstance(b, Subtree):
                b.status = Status.RUNNING
                b = b.decorated
        eng.clock = ActiveClock.from_dict(cp.clock)
        eng.ctx.clock = eng.clock
        eng.last_safe = cp.last_safe
        eng.cp_seq = cp.seq
        eng.reason = cp.reason
        eng.state = "paused"
        eng.paused_at = cp.paused_at if cp.paused_at is not None else eng.clock.total
        if elapsed_s is None:
            eng._gap_unknown = True
        else:
            eng.clock.total = eng.paused_at + max(0.0, float(elapsed_s))
        # posture as recorded; the current one is unknown after RTH / dock: assume stowed, out of work, no tool
        eng.posture_at_pause = copy.deepcopy(cp.posture) if cp.posture else None
        eng.ctx.started = True
        eng.event("restored", workflow=cp.workflow, seq=cp.seq, node_path=cp.node_path)
        return eng

    # -------------------------------------------------------------- console (WP43 reads this; <= 2 Hz)
    def breadcrumb(self, lang: str = "en") -> str:
        parts = [self.comp.doc.get("console", {}).get(lang) or self.comp.doc.get("console", {}).get("tile")
                 or self.comp.id] if isinstance(self.comp.doc.get("console"), dict) else [self.comp.id]
        b = self.root
        while b is not None:
            node = getattr(b, "node", None)
            if isinstance(b, Foreach) and b.current is not None:
                parts.append(f"{_label(node, lang)} {len(b.completed) + len(b.skipped) + 1}/{b.total}")
            elif isinstance(b, (Action, Check)) and b.status == Status.RUNNING:
                parts.append(_label(node, lang))
            nxt = None
            for c in b.children:
                if c.status == Status.RUNNING:
                    nxt = c
                    break
            b = nxt
        return " > ".join(p for p in parts if p)

    def status(self) -> dict:
        fe = [b for b in self.foreach_nodes().values() if b.status == Status.RUNNING]
        top = fe[0] if fe else None
        guards = [b.node.f["cond"] for b in self.root.iterate() if isinstance(b, Guard) and b.status == Status.RUNNING]
        return {"workflow": self.comp.ident, "run_id": self.run_id, "state": self.state, "reason": self.reason,
                "breadcrumb": {"en": self.breadcrumb("en"), "hi": self.breadcrumb("hi")},
                "target": None if top is None else {"i": len(top.completed) + len(top.skipped), "n": top.total},
                "skips": sum(len(b.skipped) for b in self.foreach_nodes().values()),
                "active_guards": guards, "last_interrupt": self.last_interrupt, "atomic_open": self.ctx.atomic_depth > 0}

    def close(self) -> None:
        self.bb.close()


def _label(node, lang) -> str:
    if node is None:
        return ""
    return (node.labels or {}).get(lang) or (node.labels or {}).get("en") or node.label


# ---------------------------------------------------------------------------------------------- fault matrix
FAULT_MODES = ("fail", "timeout", "interrupt")


def fault_cases(comp: compiler.Compiled) -> List[dict]:
    """Every action node of the main tree x {fail, timeout, interrupt} (A.6.4)."""
    out = []
    for n in comp.root.walk():
        if n.type == "action":
            for mode in FAULT_MODES:
                out.append({"node": n.id, "do": n.do, "mode": mode})
    return out


def enclosing(comp: compiler.Compiled, node_id: str) -> List[compiler.Node]:
    """The ancestors of ``node_id`` (root first)."""
    def rec(n, acc):
        if n.id == node_id:
            return acc
        for c in n.children:
            r = rec(c, acc + [n])
            if r is not None:
                return r
        return None
    return rec(comp.root, []) or []


def run_fault_case(comp: compiler.Compiled, case: dict, library: Optional[compiler.Library] = None,
                   world_kw: Optional[dict] = None, max_ticks: int = 4000) -> dict:
    """Run ``comp`` on stubs with one forced fault; returns what happened (the test asserts it)."""
    world_kw = dict(world_kw or {})
    script = dict(world_kw.pop("script", {}) or {})
    on_start = {}
    if case["mode"] == "fail":
        script[case["node"]] = "fail"
    elif case["mode"] == "timeout":
        script[case["node"]] = "hang"
    else:
        fired = []

        def inject(engine, _fired=fired):
            if not _fired:
                _fired.append(True)
                engine.pause("operator_pause")
        on_start[case["node"]] = inject
    world = stubs.StubWorld(script=script, on_start=on_start, **world_kw)
    run_id = f"fault-{abs(hash((case['node'], case['mode']))) % 10**8}"
    eng = Engine(comp, run_id=run_id, world=world, library=library, checkpoint_store=False)
    try:
        state = eng.run(max_ticks=max_ticks)
        paused = state == "paused"
        resumed_state = None
        if paused:
            eng.resume()
            resumed_state = eng.run(max_ticks=max_ticks)
        return {"case": case, "state": eng.state, "first_stop": state, "paused": paused,
                "resumed_state": resumed_state, "reason": eng.reason,
                "finalisers": [e for e in eng.events if e["kind"] == "finaliser_end"],
                "calls": list(world.calls), "violations": list(world.violations), "events": list(eng.events),
                "holding": world.holding, "released": world.released, "ticks": eng.ticks}
    finally:
        eng.close()
