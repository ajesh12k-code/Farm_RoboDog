"""Static abstract interpretation of a compiled workflow (WP34; WORKFLOWS-SPEC A.6.3, A.2 commit points).

Tracks, along every path, a SET of abstract robot states ``S(ack, mast, arm, holding, released, work, tool,
crown_ok)`` and proves:

* ``start_ack`` dominates every motion-capable node (stopping / parking primitives excepted);
* ``move_to`` / ``follow_rows`` / ``orbit_arc`` respect the speed cap for the mast state (0.18 m/s stowed,
  0.15 m/s crown up, MP 0.3) and the ``approach_envelope`` (``standoff_m`` >= ``min_standoff_m``; never inside
  ``canopy_r_m`` + 0.2 m with the mast up; under the skirt only with ``enter_annulus: stowed_only`` and the mast
  stowed, never with ``never``);
* ``tool_change`` only with the mast stowed and nothing held;
* every blade or jaw primitive sits inside ``enter_work``;
* every commit primitive (``clip``, ``twist``, ``cut``, ``ground_pick``, bridge ``pick``) is the FIRST child of an
  ``atomic`` that ends in ``place`` / ``release_safe`` (bridge ``pick`` places itself); an ``atomic`` holds no
  motion other than ``arm_*``, ``lower``, ``place`` (and ``release_safe``) and no ``wait_for`` over 10 s;
* no path reaches ``arm_park`` with ``holding`` set except through ``release_safe``;
* ``raise_mast`` / ``mast_snake`` only after a crown gate (``crown_gate`` precondition, or a ``check`` /
  ``wait_for`` / ``guard`` on a crown-gate condition); unreachable at all in a ``mast: stowed_only`` workflow;
* every path ends in a terminal plus the finaliser: ``on_pause`` and ``on_abort`` are analysed from EVERY state
  an interrupt can see (any point outside an ``atomic``, and every end state) and must leave the arm parked,
  the mast stowed and nothing held unless ``release_safe`` raised the holding assist; a successful end of the
  root may not leave an item held.

E-stop clearing, arbiter bypass and ``set()`` on ``/safety/*`` are refused by the semantic pass (validate.py);
the L0/L1 person pause does not depend on any YAML node (engine.py, mutation-tested).
"""
from dataclasses import dataclass, replace
from typing import FrozenSet, List, Optional, Set, Tuple

from . import compiler, schema
from .primitives import registry as reg

Issue = schema.Issue
ANNULUS_MARGIN_M = 0.2           # CHORES D.0: inside canopy_r_m + 0.2 m with the mast up is refused
ATOMIC_WAIT_MAX_S = 10.0
FIXPOINT_MAX = 12
BASE_MOTION_CAPPED = ("move_to", "follow_rows", "orbit_arc", "reenter_station", "return_to_checkpoint")
ATOMIC_MOTION_OK = ("lower", "place", "release_safe")


@dataclass(frozen=True)
class S:
    ack: bool = False
    mast: str = "stowed"          # stowed | up
    arm: str = "parked"           # parked | out
    holding: bool = False
    released: bool = False        # release_safe passed while holding (keep holding + assist)
    work: bool = False            # inside enter_work .. exit_work
    tool: Optional[str] = None
    crown_ok: bool = False
    skipping: bool = False        # a failure that is a skipped target (consumed by the enclosing foreach)


States = FrozenSet[S]


class _Ctx:
    def __init__(self, comp: compiler.Compiled):
        self.comp = comp
        doc = comp.doc
        self.env = doc.get("approach_envelope") or {}
        self.stowed_only = doc.get("mast") == "stowed_only"
        self.params = comp.param_specs
        self.issues: List[Issue] = []
        self._seen: Set[Tuple[str, str]] = set()
        self.interrupt: Set[S] = set()
        self.in_atomic = 0

    def err(self, code: str, path: str, msg: str):
        if (code, path) not in self._seen:
            self._seen.add((code, path))
            self.issues.append(Issue(code, path, msg))

    def num(self, v) -> Optional[float]:
        """A literal number, or the worst case of a ``$param`` (its max, else its default)."""
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, str) and v.startswith("$"):
            p = self.params.get(v[1:]) or {}
            for k in ("max", "default"):
                if isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool):
                    return float(p[k])
        return None

    def min_num(self, v) -> Optional[float]:
        if isinstance(v, str) and v.startswith("$"):
            p = self.params.get(v[1:]) or {}
            for k in ("min", "default"):
                if isinstance(p.get(k), (int, float)) and not isinstance(p.get(k), bool):
                    return float(p[k])
            return None
        return self.num(v)


def initial_states(comp: compiler.Compiled) -> States:
    crown = "crown_gate" in (comp.doc.get("preconditions") or [])
    if comp.kind == "subtree":
        return frozenset({S(ack=True, crown_ok=crown)})
    return frozenset({S(crown_ok=crown)})


def check(comp: compiler.Compiled) -> List[Issue]:
    ctx = _Ctx(comp)
    start = initial_states(comp)
    succ, fail = _run(comp.root, start, ctx)
    ends = set(succ) | set(fail)
    ctx.interrupt |= ends
    for s in succ:
        if s.holding and not s.released and comp.kind != "subtree":
            ctx.err("unsafe_terminal", comp.root.id, "a successful end of the root still holds an item")
    if comp.kind != "subtree":
        for top in (comp.on_pause, comp.on_abort):
            if top is None:
                continue
            fin_in = frozenset(replace(s, ack=True, skipping=False) for s in ctx.interrupt) or start
            f_succ, f_fail = _run(top, fin_in, ctx)
            for s in f_succ:
                if s.mast != "stowed" or s.arm != "parked" or (s.holding and not s.released):
                    ctx.err("finaliser_unsafe", top.id, f"the finaliser can end with mast {s.mast}, arm {s.arm}, "
                                                        f"holding {s.holding}")
                    break
    return ctx.issues


# ---------------------------------------------------------------------------------------------- transfer
def _skipping(states) -> States:
    return frozenset(replace(s, skipping=True) for s in states)


def _split(fails):
    skip = {s for s in fails if s.skipping}
    return skip, set(fails) - skip


def _run(n: compiler.Node, states: States, ctx: _Ctx) -> Tuple[States, States]:
    """(success states, failure states). A failure state with ``skipping`` set is a skipped target: it short-
    circuits selectors and recovers (the engine's skip_pending) and is consumed by the enclosing foreach."""
    t = n.type
    if t == "action":
        s, f = _action(n, states, ctx, first_in_atomic=False)
        if n.do in ("skip", "assist_request"):
            f = _skipping(f)
        return s, f
    if t == "sequence" or t == "parallel":
        cur, fails = states, set()
        for c in n.children:
            s, f = _run(c, cur, ctx)
            fails |= f
            cur = s
        return frozenset(cur), frozenset(fails)
    if t == "selector":
        cur, succ, skipped = states, set(), set()
        for c in n.children:
            s, f = _run(c, cur, ctx)
            succ |= s
            sk, plain = _split(f)
            skipped |= sk
            cur = frozenset(plain)
        return frozenset(succ), frozenset(set(cur) | skipped)
    if t == "atomic":
        return _atomic(n, states, ctx)
    if t == "subtree":
        return _run(n.children[0], states, ctx)
    if t == "guard":
        inner = states
        if reg.CONDITIONS.get(n.f["cond"]) is not None and reg.CONDITIONS[n.f["cond"]].crown_gate:
            inner = frozenset(replace(s, crown_ok=True) for s in states)
        s, f = _run(n.children[0], inner, ctx)
        violated = _skipping(states) if n.f["on_violation"] == "skip_target" else states
        return s, frozenset(set(f) | set(violated))
    if t in ("check", "wait_for"):
        cond = reg.CONDITIONS.get(n.f["cond"])
        if t == "wait_for" and ctx.in_atomic:
            to = n.f.get("timeout_s")
            if to is None or not isinstance(to, (int, float)) or to > ATOMIC_WAIT_MAX_S:
                ctx.err("atomic_wait", n.id, f"wait_for inside an atomic needs timeout_s <= {ATOMIC_WAIT_MAX_S}")
        succ = states
        if cond is not None and cond.crown_gate:
            succ = frozenset(replace(s, crown_ok=True) for s in states)
        if not ctx.in_atomic:
            ctx.interrupt |= states
        kind = n.f["else" if t == "check" else "on_timeout"][0]
        return succ, (_skipping(states) if kind in ("skip", "assist") else states)
    if t == "foreach":
        entry = set(states)
        fail_all: Set[S] = set()
        for _ in range(FIXPOINT_MAX):
            s, f = _run(n.children[0], frozenset(entry), ctx)
            sk, plain = _split(f)
            fail_all |= plain
            new = entry | set(s) | {replace(x, skipping=False) for x in sk}
            if new == entry:
                break
            entry = new
        return frozenset(entry), frozenset(fail_all)
    if t == "recover":
        body, rec = n.children
        rounds = n.f.get("max_rounds") if isinstance(n.f.get("max_rounds"), int) else 2
        rounds = max(1, min(rounds, 5))
        cur = states
        succ_all: Set[S] = set()
        fail_all: Set[S] = set()
        skipped: Set[S] = set()
        for r in range(rounds):
            s, f = _run(body, cur, ctx)
            succ_all |= s
            sk, plain = _split(f)
            skipped |= sk
            if r == rounds - 1 or not plain:
                fail_all |= plain
                break
            rs, rf = _run(rec, frozenset(plain), ctx)
            rsk, rplain = _split(rf)
            skipped |= rsk
            cur = frozenset(set(rs) | rplain)
        then = n.f["then"][0]
        if then == "continue":
            return frozenset(succ_all | fail_all), frozenset(skipped)
        if then == "skip":
            return frozenset(succ_all), frozenset(skipped | set(_skipping(fail_all)))
        return frozenset(succ_all), frozenset(skipped | fail_all)
    raise ValueError(f"static check: unknown node type {t!r}")         # pragma: no cover


def _atomic(n: compiler.Node, states: States, ctx: _Ctx) -> Tuple[States, States]:
    kids = n.children
    first = kids[0] if kids else None
    if first is None or first.type != "action" or not reg.PRIMITIVES.get(first.do or "", reg.PRIMITIVES["stop"]).commit:
        ctx.err("atomic_without_commit", n.id, "an atomic block starts with its commit primitive")
    last = kids[-1] if kids else None
    self_placing = first is not None and first.type == "action" and \
        getattr(reg.PRIMITIVES.get(first.do or ""), "self_placing", False)
    if not self_placing and (last is None or last.type != "action" or last.do not in ("place", "release_safe")):
        ctx.err("atomic_not_closed", n.id, "an atomic block ends in place or release_safe")
    ctx.in_atomic += 1
    try:
        cur, fails = states, set()
        for i, c in enumerate(kids):
            if c.type == "action":
                s, f = _action(c, cur, ctx, first_in_atomic=(i == 0))
            else:
                s, f = _run(c, cur, ctx)
            fails |= f
            cur = s
    finally:
        ctx.in_atomic -= 1
    return frozenset(cur), frozenset(fails)


def _motion_not_ok_in_atomic(do: Optional[str]) -> bool:
    spec = reg.PRIMITIVES.get(do or "")
    if spec is None or not spec.motion:
        return False
    return not (do.startswith("arm_") or do in ATOMIC_MOTION_OK or spec.commit)


def _action(n: compiler.Node, states: States, ctx: _Ctx, first_in_atomic: bool) -> Tuple[States, States]:
    do = n.do
    spec = reg.PRIMITIVES.get(do or "")
    if spec is None:
        return states, states
    if not ctx.in_atomic:
        ctx.interrupt |= states
    args = n.f.get("args") or {}
    if spec.commit and not first_in_atomic:
        ctx.err("commit_outside_atomic", n.id, f"{do} is a commit point: it must be the first child of an atomic")
    if ctx.in_atomic and _motion_not_ok_in_atomic(do):
        ctx.err("atomic_motion", n.id, f"{do} is motion: not allowed inside an atomic")
    if do in ("raise_mast", "mast_snake") and ctx.stowed_only:
        ctx.err("raise_mast_stowed_only", n.id, "this workflow declares mast: stowed_only")
    succ: Set[S] = set()
    fail: Set[S] = set()
    for s in states:
        if spec.motion and not spec.safe_direction and not s.ack:
            ctx.err("motion_before_start_ack", n.id, f"{do} can move before start_ack succeeded")
        if do == "tool_change" and (s.holding or s.mast != "stowed"):
            ctx.err("tool_change_unsafe", n.id, "tool_change needs the mast stowed and nothing held")
        if spec.blade and not s.work:
            ctx.err("blade_outside_work", n.id, f"{do} is a blade / jaw primitive outside enter_work")
        if do == "arm_park" and s.holding and not s.released:
            ctx.err("arm_park_while_holding", n.id, "arm_park reachable while holding, without release_safe")
        if do in ("raise_mast", "mast_snake") and not s.crown_ok:
            ctx.err("raise_mast_ungated", n.id, "raise_mast before the crown gates are checked")
        if do in BASE_MOTION_CAPPED:
            _speed_and_envelope(n, s, args, ctx)
        succ |= _effect(spec, s, args)
        fail.add(s)
        if spec.commit and not spec.self_placing:
            fail.add(replace(s, holding=True, released=False, arm="out"))
    return frozenset(succ), frozenset(fail)


def _speed_and_envelope(n, s: S, args, ctx: _Ctx):
    v = ctx.num(args.get("v_max"))
    cap = reg.SPEED_CAP_CROWN_UP if s.mast == "up" else reg.SPEED_CAP_STOWED
    if v is not None and v > cap + 1e-9:
        ctx.err("speed_cap", n.id, f"v_max {v} > {cap} m/s with the mast {s.mast}")
    if n.do != "move_to":
        return
    so = ctx.min_num(args.get("standoff_m"))
    if so is None:
        return
    env = ctx.env
    min_so = env.get("min_standoff_m")
    if isinstance(min_so, (int, float)) and so < min_so - 1e-9:
        ctx.err("approach_envelope", n.id, f"standoff_m {so} < approach_envelope.min_standoff_m {min_so}")
    if s.mast == "up" and so < ANNULUS_MARGIN_M:
        ctx.err("annulus_mast_up", n.id, f"move_to inside canopy_r_m + {ANNULUS_MARGIN_M} m with the mast up")
    if so < 0:
        mode = env.get("enter_annulus", "never")
        if mode == "never" or (mode == "stowed_only" and s.mast != "stowed"):
            ctx.err("approach_envelope", n.id, f"move_to under the skirt with enter_annulus {mode} and mast {s.mast}")


def _effect(spec: reg.PrimitiveSpec, s: S, args) -> Set[S]:
    n = spec.name
    if n == "start_ack":
        return {replace(s, ack=True)}
    if n in ("raise_mast", "mast_snake"):
        return {replace(s, mast="up")}
    if n == "stow_mast":
        return {replace(s, mast="stowed")}
    if n in ("arm_approach", "roll_align", "trap", "lower"):
        return {replace(s, arm="out")}
    if n == "arm_park":
        return {replace(s, arm="parked")}
    if n == "enter_work":
        return {replace(s, work=True)}
    if n == "exit_work":
        return {replace(s, work=False)}
    if n == "tool_change":
        return {replace(s, tool=str(args.get("tool")))}
    if spec.commit and not spec.self_placing:
        return {replace(s, holding=True, released=False, arm="out")}
    if n == "place":
        return {replace(s, holding=False, released=False)}
    if n == "release_safe":
        if s.holding:
            return {replace(s, holding=False, released=False), replace(s, released=True)}
        return {s}
    return {s}
