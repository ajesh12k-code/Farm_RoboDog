"""Per-target retry / skip / assist / halt policy for contact presets (MISSION-PRESETS 0.4; renamed from the
vegetable ``recovery.py``). Pure Python, no MuJoCo, no ROS.

MISSION-PRESETS 0.4, in order:

1. retry a failed target up to the per-target limit (``RetryPolicy.per_target``; the task SMs also retry inside a
   target: MP 5 re-trap once, T4 re-grip once);
2. skip with a reason code;
3. assist request: Take over / Skip / Halt (``assist`` reasons; the loop waits for ``assist_reply``);
4. more than 50 % of a tree's targets skipped -> the preset halts and the robot returns home (``halt_tree`` with
   ``next: return_home``);
5. a safety event (e-stop, protective stop) always halts and latches (``latch``; only ``clear_latch`` re-arms).

Commit point (WORKFLOWS-SPEC A.0): before the cut / release a failure rolls back to ``arm_park`` and may retry; after
it the hold-verify -> place block is atomic, so an outcome after the commit (``dropped``, ``picked``) is never retried.
Outcomes that are final by nature (MARKs, ``bagged``, ``keep_wood``, ``no_clearance``, ``twist_failed`` already retried
inside the SM) are never retried here either.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

FINAL_OUTCOMES = ("picked", "cut", "marked", "dropped", "halted")
COMMITTED_OUTCOMES = ("picked", "cut", "dropped", "halted")   # after the commit point: never an assist
NO_RETRY_REASONS = ("bagged", "keep_wood", "no_clearance", "twist_failed", "immature", "out_of_reach_marked",
                    "over_capacity", "paste_required", "payload_full", "not_in_mvp", "torque_limit", "estop",
                    "operator_skip", "human_cut")
RETRY_REASONS = ("cut_failed", "grip_failed", "plan_failed", "base_moved", "stability", "place_refused")
ASSIST_REASONS = ("twist_failed", "grip_failed", "base_moved")     # OS 5.3 rows with "assist" (twist_failed: final
#                                                                   but still prompts; the others once retries run out)
SAFETY_REASONS = ("estop", "protective_stop", "fall", "local_link_lost")
ASSIST_REPLIES = ("take_over", "skip", "halt")


@dataclass
class RetryPolicy:
    per_target: int = 1                 # extra whole-target attempts after the first (MP 0.4 step 1)
    halt_skip_fraction: float = 0.5     # > 50 % of a tree's targets skipped -> halt (MP 0.4 step 4)
    assist_on: Sequence[str] = ASSIST_REASONS
    assist_auto_reply: Optional[str] = "skip"   # headless runs: no operator -> the assist times out to Skip


@dataclass
class TargetState:
    target_id: str
    attempts: int = 0
    outcome: Optional[str] = None
    reason: str = ""
    history: List[dict] = field(default_factory=list)


class TargetLoop:
    """The MP 0.4 target loop over one tree's targets. Call ``next_target()``, run it, ``report(...)`` the result;
    the decision says what happens next (``retry`` / ``next`` / ``assist`` / ``halt_tree`` / ``halt_safety``)."""

    def __init__(self, target_ids: Sequence[str], policy: Optional[RetryPolicy] = None, tree_id: str = ""):
        self.policy = policy or RetryPolicy()
        self.tree_id = tree_id
        self.order = list(target_ids)
        self.state: Dict[str, TargetState] = {t: TargetState(t) for t in self.order}
        self.queue: List[str] = list(self.order)
        self.latched: Optional[str] = None
        self.halted: Optional[str] = None
        self.pending_assist: Optional[dict] = None
        self.log: List[dict] = []

    # -------------------------------------------------------------- queries
    @property
    def finished(self) -> bool:
        return self.halted is not None or self.latched is not None or (not self.queue and self.pending_assist is None)

    def skipped(self) -> List[str]:
        return [t for t, s in self.state.items() if s.outcome == "skipped"]

    def skip_fraction(self) -> float:
        return len(self.skipped()) / max(1, len(self.order))

    def next_target(self) -> Optional[str]:
        if self.latched or self.halted or self.pending_assist is not None or not self.queue:
            return None
        return self.queue[0]

    # -------------------------------------------------------------- results
    def report(self, target_id: str, outcome: str, reason: str = "", committed: bool = False) -> dict:
        """One attempt of ``target_id`` ended with ``outcome`` / ``reason``. Returns the decision dict."""
        st = self.state[target_id]
        st.attempts += 1
        st.history.append({"outcome": outcome, "reason": reason, "committed": committed})
        if outcome == "halted" or reason in SAFETY_REASONS:
            st.outcome, st.reason = "halted", reason or "estop"
            self.latched = st.reason
            return self._decide(target_id, "halt_safety", reason=st.reason)
        if reason in self.policy.assist_on and not committed and outcome not in COMMITTED_OUTCOMES:
            retryable = (reason in RETRY_REASONS and outcome not in FINAL_OUTCOMES
                         and st.attempts <= self.policy.per_target)
            if not retryable:
                # OS 5.3 "assist" rows: a final result that still needs the operator (T4 no release: the SM already
                # re-gripped once and MARKed ``twist_failed``) and a retryable one whose retries are spent
                return self._assist(target_id, reason, outcome if outcome in FINAL_OUTCOMES else "skipped")
        if outcome in FINAL_OUTCOMES or committed or reason in NO_RETRY_REASONS:
            st.outcome, st.reason = ("skipped" if outcome not in FINAL_OUTCOMES else outcome), reason
            self._pop(target_id)
            return self._after_final(target_id)
        # a pre-commit failure: retry while attempts remain, else skip (assist reasons were handled above)
        if st.attempts <= self.policy.per_target and reason in RETRY_REASONS:
            return self._decide(target_id, "retry", reason=reason, attempt=st.attempts + 1)
        st.outcome, st.reason = "skipped", reason
        self._pop(target_id)
        return self._after_final(target_id)

    def _assist(self, target_id: str, reason: str, outcome: str) -> dict:
        """Raise the operator assist (Take over / Skip / Halt). ``outcome`` = what the target counts as on Skip
        (``marked`` for a MARKed twist_failed, else ``skipped``)."""
        self.pending_assist = {"target_id": target_id, "reason": reason, "outcome": outcome,
                               "options": list(ASSIST_REPLIES)}
        if self.policy.assist_auto_reply is not None:
            return self.assist_reply(self.policy.assist_auto_reply, auto=True)
        return self._decide(target_id, "assist", reason=reason, options=list(ASSIST_REPLIES))

    def assist_reply(self, reply: str, auto: bool = False) -> dict:
        """Operator answer to an assist request: take_over (the human does it: counted as skipped by the robot,
        reason ``operator_take_over``), skip, or halt."""
        if self.pending_assist is None:
            return {"decision": "ignored", "reason": "no_assist_pending"}
        if reply not in ASSIST_REPLIES:
            raise ValueError(f"assist reply {reply!r} not in {ASSIST_REPLIES}")
        a, self.pending_assist = self.pending_assist, None
        tid = a["target_id"]
        st = self.state[tid]
        if reply == "halt":
            st.outcome, st.reason = a.get("outcome") or "skipped", a["reason"]
            self._pop(tid)
            self.halted = "operator_halt"
            return self._decide(tid, "halt_tree", reason="operator_halt", next="return_home", assist=reply, auto=auto)
        st.outcome = a.get("outcome") or "skipped"
        st.reason = a["reason"] if reply == "skip" else "operator_take_over"
        self._pop(tid)
        d = self._after_final(tid)
        d.update({"assist": reply, "auto": auto})
        return d

    def latch(self, reason: str) -> dict:
        """A safety event outside a target (e-stop between targets): halt and latch."""
        self.latched = reason
        return self._decide(None, "halt_safety", reason=reason)

    def clear_latch(self):
        self.latched = None

    # -------------------------------------------------------------- internals
    def _pop(self, target_id: str):
        if target_id in self.queue:
            self.queue.remove(target_id)

    def _after_final(self, target_id: str) -> dict:
        if self.skip_fraction() > self.policy.halt_skip_fraction:
            self.halted = "tree_halt"
            return self._decide(target_id, "halt_tree", reason="tree_halt", next="return_home",
                                skipped=len(self.skipped()), of=len(self.order))
        return self._decide(target_id, "next" if self.queue else "tree_done")

    def _decide(self, target_id: Optional[str], decision: str, **kw) -> dict:
        d = {"tree_id": self.tree_id, "target_id": target_id, "decision": decision, **kw}
        self.log.append(d)
        return d

    def summary(self) -> dict:
        counts: Dict[str, int] = {}
        for s in self.state.values():
            counts[s.outcome or "pending"] = counts.get(s.outcome or "pending", 0) + 1
        return {"tree_id": self.tree_id, "counts": counts, "skip_fraction": round(self.skip_fraction(), 3),
                "halted": self.halted, "latched": self.latched,
                "targets": {t: {"outcome": s.outcome, "reason": s.reason, "attempts": s.attempts}
                            for t, s in self.state.items()}}
