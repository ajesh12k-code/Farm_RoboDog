"""Workflow checkpoints (WP34; WORKFLOWS-SPEC A.4) and the ``workflow_checkpoint`` sim-state provider.

Written after every completed target, on pause and on RTH. Fields (A.4): ``workflow`` = ``id@version#sha8``,
the node path, foreach index / completed / skipped target ids per foreach node, ``last_safe`` pose, the
``/wf/<run_id>`` whitelist, params, interruption reason and the run-supervisor state (plus the active clock).

``STORE`` keeps the newest checkpoint per run in this process. The store is registered with WP0's mutation
provider registry as ``workflow_checkpoint`` (``atharv_nandi/sim/hooks/workflow.py`` calls ``register_provider()`` at
import, so ``sim_state.discover_providers()`` finds it in every process): ``set_state(get_state())``
round-trips it, and a world reset (``hooks.reset``) clears it. ``pack`` returns ``{}`` while no run has a
checkpoint, so a world without workflows carries an empty block.
"""
import copy
import math
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

PROVIDER_KEY = "workflow_checkpoint"
PROVIDER_VERSION = 1
SCHEMA = "nandi.wf_checkpoint/1"


class CheckpointError(ValueError):
    pass


@dataclass
class Checkpoint:
    workflow: str                                   # id@version#sha8
    run_id: str
    node_path: List[int] = field(default_factory=list)
    foreach: Dict[str, dict] = field(default_factory=dict)  # node id -> {index, completed[], skipped{id: reason}}
    completed_targets: List[str] = field(default_factory=list)
    last_safe: Optional[List[float]] = None         # [x, y, yaw_deg] of the last safe alley pose
    wf: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    supervisor: str = "running"
    clock: Dict[str, float] = field(default_factory=dict)
    seq: int = 0                                    # monotonically increasing per run
    posture: Dict[str, Any] = field(default_factory=dict)  # {mast: {do, args} | None, work, tool} to restore (A.4.5)
    paused_at: Optional[float] = None               # clock total when the pause / RTH began (None: target done)
    schema: str = SCHEMA

    def to_dict(self) -> dict:
        d = asdict(self)
        check_json(d)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        if not isinstance(d, dict) or d.get("schema") != SCHEMA:
            raise CheckpointError(f"not a {SCHEMA} checkpoint")
        check_json(d)
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise CheckpointError(f"unknown checkpoint fields {sorted(unknown)}")
        for k in ("workflow", "run_id"):
            if not isinstance(d.get(k), str) or not d[k]:
                raise CheckpointError(f"checkpoint {k} missing")
        if "#" not in d["workflow"] or "@" not in d["workflow"]:
            raise CheckpointError(f"checkpoint workflow {d['workflow']!r} is not id@version#sha8")
        if not isinstance(d.get("node_path", []), list) or not all(isinstance(i, int) for i in d.get("node_path", [])):
            raise CheckpointError("node_path must be a list of ints")
        if any(isinstance(i, bool) or i < 0 for i in d.get("node_path", [])):
            raise CheckpointError("node_path indices must be ints >= 0")
        ls = d.get("last_safe")
        if ls is not None and (not isinstance(ls, list) or len(ls) != 3 or not all(_num(x) for x in ls)):
            raise CheckpointError("last_safe must be [x, y, yaw_deg]")
        _check_types(d)
        return cls(**copy.deepcopy(d))


def check_json(obj: Any, where: str = "checkpoint") -> None:
    """Strict JSON (sim_state.encode refuses NaN / inf): raise ``CheckpointError`` otherwise."""
    try:
        json.dumps(obj, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"{where} is not strict JSON: {exc}") from None


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _str_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _check_types(d: dict) -> None:
    """Field types (A.4): a bad block is refused here, before anything is written or restored."""
    fe = d.get("foreach", {})
    if not isinstance(fe, dict):
        raise CheckpointError("foreach must be {node_id: state}")
    for nid, st in fe.items():
        where = f"foreach[{nid!r}]"
        if not isinstance(nid, str) or not isinstance(st, dict):
            raise CheckpointError(f"{where} must be a node id -> state mapping")
        unknown = set(st) - {"index", "total", "current", "completed", "skipped"}
        if unknown:
            raise CheckpointError(f"{where} has unknown fields {sorted(unknown)}")
        for k in ("index", "total"):
            v = st.get(k, 0)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise CheckpointError(f"{where}.{k} must be an int >= 0")
        if st.get("current") is not None and not isinstance(st["current"], str):
            raise CheckpointError(f"{where}.current must be a target id or null")
        if not _str_list(st.get("completed", [])):
            raise CheckpointError(f"{where}.completed must be a list of target ids")
        sk = st.get("skipped", {})
        if not isinstance(sk, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in sk.items()):
            raise CheckpointError(f"{where}.skipped must be {{target id: reason}}")
    if not _str_list(d.get("completed_targets", [])):
        raise CheckpointError("completed_targets must be a list of target ids")
    for k in ("wf", "params"):
        v = d.get(k, {})
        if not isinstance(v, dict) or not all(isinstance(x, str) for x in v):
            raise CheckpointError(f"{k} must be a mapping with string keys")
    if any(".." in k or k.startswith("/") for k in d.get("wf", {})):
        raise CheckpointError("wf keys are relative to /wf/<run_id>")
    clock = d.get("clock", {})
    if not isinstance(clock, dict) or set(clock) - {"active", "total"} \
            or not all(_num(v) and v >= 0 for v in clock.values()):
        raise CheckpointError("clock must be {active, total} with finite seconds >= 0")
    if d.get("reason") is not None and not isinstance(d["reason"], str):
        raise CheckpointError("reason must be a reason code or null")
    if not isinstance(d.get("supervisor", "running"), str):
        raise CheckpointError("supervisor must be a string")
    seq = d.get("seq", 0)
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise CheckpointError("seq must be an int >= 0")
    pa = d.get("paused_at")
    if pa is not None and not (_num(pa) and pa >= 0):
        raise CheckpointError("paused_at must be finite seconds >= 0 or null")
    po = d.get("posture", {})
    if not isinstance(po, dict) or set(po) - {"mast", "work", "tool"}:
        raise CheckpointError("posture must be {mast, work, tool}")
    mast = po.get("mast")
    if mast is not None and not (isinstance(mast, dict) and isinstance(mast.get("do"), str)
                                 and isinstance(mast.get("args", {}), dict)):
        raise CheckpointError("posture.mast must be {do, args} or null")
    if not isinstance(po.get("work", False), bool):
        raise CheckpointError("posture.work must be a bool")
    if po.get("tool") is not None and not isinstance(po["tool"], str):
        raise CheckpointError("posture.tool must be a tool id or null")


# ---------------------------------------------------------------------------------------------- store
STORE: Dict[str, dict] = {}


def save(cp: Checkpoint) -> dict:
    d = cp.to_dict()
    STORE[cp.run_id] = d
    return d


def latest(run_id: Optional[str] = None) -> Optional[Checkpoint]:
    if run_id is None:
        if not STORE:
            return None
        run_id = sorted(STORE)[-1]
    d = STORE.get(run_id)
    return Checkpoint.from_dict(d) if d is not None else None


def clear(run_id: Optional[str] = None) -> None:
    if run_id is None:
        STORE.clear()
    else:
        STORE.pop(run_id, None)


# ---------------------------------------------------------------------------------------------- provider
def pack(model=None) -> dict:
    if not STORE:
        return {}
    return {"runs": copy.deepcopy(STORE)}


def unpack(model, data: dict) -> None:
    """Replace the store with ``data`` (validated first: nothing is written if any checkpoint is bad)."""
    if not isinstance(data, dict):
        raise CheckpointError("workflow_checkpoint data must be a dict")
    if not data:
        STORE.clear()
        return
    if set(data) != {"runs"} or not isinstance(data["runs"], dict):
        raise CheckpointError("workflow_checkpoint data must be {runs: {run_id: checkpoint}}")
    parsed = {}
    for rid, d in data["runs"].items():
        cp = Checkpoint.from_dict(d)
        if cp.run_id != rid:
            raise CheckpointError(f"checkpoint run_id {cp.run_id!r} filed under {rid!r}")
        parsed[rid] = cp.to_dict()
    STORE.clear()
    STORE.update(parsed)


def reset_hook(m=None, d=None) -> None:
    """World reset: the checkpoint store goes with the world (``hooks.reset``)."""
    STORE.clear()


def register_provider() -> None:
    from atharv_nandi.core import sim_state
    sim_state.register_provider(PROVIDER_KEY, pack, unpack, version=PROVIDER_VERSION)
