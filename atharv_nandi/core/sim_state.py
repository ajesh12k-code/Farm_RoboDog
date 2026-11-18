"""Full sim state envelope + mutation-provider registry (WP0; MVP-SPEC 3.2 / 5.7, ORCHARD-MVP-SPEC 3.2).

``/nandi/sim_state`` carries everything a renderer, the recorder or a scorer needs to rebuild the
picture of the live world: qpos, mocap poses, ``eq_active`` and a versioned ``mutations`` block for
state that lives in MODEL fields (mast section ``body_pos``, pool ``eq_data``, hidden geoms, debris)
and is therefore not in qpos.

WP0 owns only the envelope and the registry. A module registers its own provider:

    sim_state.register_provider("mast", pack=lambda m: {...}, unpack=lambda m, payload: ..., version=1)

``pack(m) -> dict`` (JSON-serialisable) reads the module's model fields; ``unpack(m, payload)`` writes
them back (``pack`` must return ``{}`` in a world that lacks the module's bodies). An optional
``applies(m) -> bool`` limits a provider to the models that hold its bodies: for any other model it is left
out of ``pack`` and not required by a strict ``apply`` (so a world without the module keeps a byte-identical
message even in a process that also built a world with it). Register a provider at
IMPORT time of the package's world module ``atharv_nandi/sim/world/modules/<name>.py`` or of its hook file
``atharv_nandi/sim/hooks/<name>.py`` (WP24 mast has only the latter). ``discover_providers()`` imports both directories,
so the physics node and every renderer hold the same provider set; ``camera_rig.set_state`` calls it once
before its first apply. ``apply`` validates the whole message first, then unpacks the providers (rolled
back if one raises), then writes qpos / mocap / ``eq_active``; ``camera_rig.set_state`` runs ``mj_kinematics``.

Envelope (JSON object, ``SIM_STATE_SCHEMA``)::

    {"schema": "nandi.sim_state/1", "world_id": str, "t": float,
     "nq": int, "qpos": [nq], "nmocap": int, "mocap_pos": [3*nmocap], "mocap_quat": [4*nmocap],
     "neq": int, "eq_active": [neq of 0/1],
     "mutations": {"v": 1, "providers": {<key>: {"v": <provider version>, "data": {...}}}}}

Floats go through ``json`` with ``repr`` precision, so a round trip is bit-exact.
"""
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

SIM_STATE_SCHEMA = "nandi.sim_state/1"
MUTATIONS_VERSION = 1
SIM_STATE_TOPIC = "/nandi/sim_state"
SIM_STATE_PERIOD_S = 1.0 / 30.0   # 30 Hz of sim time (MVP-SPEC 5.7)


class SimStateError(ValueError):
    pass


@dataclass(frozen=True)
class Provider:
    key: str
    pack: Callable[[Any], dict]
    unpack: Callable[[Any, dict], None]
    version: int = 1
    applies: Optional[Callable[[Any], bool]] = None

    def applies_to(self, model) -> bool:
        return self.applies is None or bool(self.applies(model))


_PROVIDERS: Dict[str, Provider] = {}


def register_provider(key: str, pack: Callable, unpack: Callable, version: int = 1,
                      applies: Optional[Callable] = None) -> Provider:
    """Register (or replace, same key) a model-mutation provider. Keys are module names (``mast``, ``pool``...).

    ``applies(model) -> bool`` (optional): the models this provider covers; default all."""
    if not key or not isinstance(key, str):
        raise ValueError("provider key must be a non-empty string")
    p = Provider(key, pack, unpack, int(version), applies)
    _PROVIDERS[key] = p
    return p


def unregister_provider(key: str) -> None:
    _PROVIDERS.pop(key, None)


def providers() -> List[Provider]:
    return [_PROVIDERS[k] for k in sorted(_PROVIDERS)]


# ---------------------------------------------------------------------- pack
def pack_mutations(model) -> dict:
    out = {}
    for p in providers():
        if not p.applies_to(model):
            continue
        data = p.pack(model)
        if not isinstance(data, dict):
            raise SimStateError(f"provider {p.key!r}: pack() returned {type(data).__name__}, not dict")
        out[p.key] = {"v": p.version, "data": data}
    return {"v": MUTATIONS_VERSION, "providers": out}


def pack(model, data, world_id: str) -> dict:
    """The full state of (model, data) as a JSON-ready dict."""
    return {
        "schema": SIM_STATE_SCHEMA,
        "world_id": str(world_id),
        "t": float(data.time),
        "nq": int(model.nq),
        "qpos": [float(v) for v in data.qpos],
        "nmocap": int(model.nmocap),
        "mocap_pos": [float(v) for v in np.asarray(data.mocap_pos).reshape(-1)],
        "mocap_quat": [float(v) for v in np.asarray(data.mocap_quat).reshape(-1)],
        "neq": int(model.neq),
        "eq_active": [int(v) for v in np.asarray(data.eq_active).reshape(-1)],
        "mutations": pack_mutations(model),
    }


def _reject_constant(name):
    raise SimStateError(f"non-finite number {name} in sim_state")


def encode(state: dict) -> str:
    """Strict JSON (no NaN / Infinity, which browsers' ``JSON.parse`` rejects): raises ``SimStateError``."""
    try:
        return json.dumps(state, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SimStateError(f"sim_state not encodable: {exc}") from None


def decode(text) -> dict:
    try:
        state = json.loads(text, parse_constant=_reject_constant)
    except SimStateError:
        raise
    except (TypeError, ValueError) as exc:
        raise SimStateError(f"not JSON: {exc}") from None
    if not isinstance(state, dict) or state.get("schema") != SIM_STATE_SCHEMA:
        raise SimStateError(f"not a {SIM_STATE_SCHEMA} message")
    return state


_DISCOVERED = False


def discover_providers(force: bool = False) -> List[str]:
    """Import every world module and every hook file once so their ``register_provider`` calls run.

    Hook files are imported, not registered (``hooks.import_all``). Returns the registered provider keys."""
    global _DISCOVERED
    if force or not _DISCOVERED:
        from atharv_nandi.sim.world import compose  # noqa: deferred: compose imports controllers
        from atharv_nandi.sim import hooks
        compose.discover(force=force)
        hooks.import_all()
        _DISCOVERED = True
    return sorted(_PROVIDERS)


# ---------------------------------------------------------------------- apply
def _array(state: dict, key: str, n: int) -> np.ndarray:
    raw = state.get(key, [])
    if not isinstance(raw, (list, tuple)):
        raise SimStateError(f"{key} is {type(raw).__name__}, not a list")
    if len(raw) != n:
        raise SimStateError(f"{key} has {len(raw)} entries, model needs {n}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
        raise SimStateError(f"{key} has a non-number entry")
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise SimStateError(f"{key} has a non-finite entry")
    return arr


def _validate(model, state, world_id: Optional[str], strict_providers: bool) -> dict:
    """Every check and conversion; raises SimStateError. Returns the converted arrays + provider blocks."""
    if not isinstance(state, dict) or state.get("schema") != SIM_STATE_SCHEMA:
        raise SimStateError(f"schema is not {SIM_STATE_SCHEMA}")
    if world_id is not None and state.get("world_id") != world_id:
        raise SimStateError(f"world_id mismatch: state {state.get('world_id')!r} != model {world_id!r}")
    t = state.get("t", 0.0)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t):
        raise SimStateError(f"t is not a finite number: {t!r}")
    out = {"t": float(t), "qpos": _array(state, "qpos", model.nq),
           "mocap_pos": _array(state, "mocap_pos", 3 * model.nmocap),
           "mocap_quat": _array(state, "mocap_quat", 4 * model.nmocap)}
    eq = state.get("eq_active", [])
    if not isinstance(eq, (list, tuple)) or len(eq) != model.neq:
        n = len(eq) if isinstance(eq, (list, tuple)) else type(eq).__name__
        raise SimStateError(f"eq_active has {n} entries, model needs {model.neq}")
    if any(v not in (0, 1) or isinstance(v, float) for v in eq):
        raise SimStateError("eq_active entries must be 0 or 1")
    out["eq_active"] = np.asarray([int(v) for v in eq], dtype=np.uint8)
    muts = state.get("mutations")
    if muts is None:
        muts = {"v": MUTATIONS_VERSION, "providers": {}}
    if not isinstance(muts, dict):
        raise SimStateError(f"mutations is {type(muts).__name__}, not an object")
    if muts.get("v") != MUTATIONS_VERSION or isinstance(muts.get("v"), bool):
        raise SimStateError(f"mutations version {muts.get('v')!r} != {MUTATIONS_VERSION}")
    sent = muts.get("providers", {})
    if not isinstance(sent, dict):
        raise SimStateError(f"mutations.providers is {type(sent).__name__}, not an object")
    blocks = []
    for key in sorted(sent):
        block = sent[key]
        p = _PROVIDERS.get(key)
        if p is None:
            raise SimStateError(f"no provider registered for mutations[{key!r}]")
        if not p.applies_to(model):
            raise SimStateError(f"provider {key!r} does not apply to this model")
        if not isinstance(block, dict):
            raise SimStateError(f"mutations[{key!r}] is {type(block).__name__}, not an object")
        if block.get("v") != p.version or isinstance(block.get("v"), bool):
            raise SimStateError(f"provider {key!r} version {block.get('v')!r} != {p.version}")
        payload = block.get("data", {})
        if not isinstance(payload, dict):
            raise SimStateError(f"mutations[{key!r}].data is {type(payload).__name__}, not an object")
        blocks.append((p, payload))
    if strict_providers:
        missing = sorted(k for k, p in _PROVIDERS.items() if k not in sent and p.applies_to(model))
        if missing:
            raise SimStateError(f"state lacks mutations for registered providers {missing}")
    out["blocks"] = blocks
    return out


def check(model, state: dict, world_id: Optional[str], strict_providers: bool = True) -> Tuple[bool, str]:
    """(ok, reason): may ``state`` be applied to ``model``? Nothing is written; never raises on bad input."""
    try:
        _validate(model, state, world_id, strict_providers)
    except SimStateError as exc:
        return False, str(exc)
    return True, "ok"


def apply(model, data, state: dict, world_id: Optional[str], strict_providers: bool = True) -> Tuple[bool, str]:
    """Write every provider's mutations, then qpos, mocap and eq_active. No kinematics (the caller runs it).

    Refuses (``(False, reason)``, writes nothing) on a schema, world_id, type, size, non-finite or provider
    mismatch: the whole message is validated and converted before the first write. A provider whose
    ``unpack`` raises refuses the message too; the providers already unpacked are rolled back to the
    ``pack`` taken before. ``world_id=None`` skips the id check (only for tests and tools that built the
    model themselves). Never raises on bad input."""
    try:
        v = _validate(model, state, world_id, strict_providers)
    except SimStateError as exc:
        return False, str(exc)
    done = []
    for p, payload in v["blocks"]:
        try:
            before = p.pack(model)
        except Exception as exc:  # noqa: BLE001 - provider bug: refuse, nothing written yet for p
            _rollback(model, done)
            return False, f"provider {p.key!r}: pack() before unpack failed: {exc}"
        try:
            p.unpack(model, payload)
        except Exception as exc:  # noqa: BLE001
            _rollback(model, done + [(p, before)])
            return False, f"provider {p.key!r}: unpack failed: {type(exc).__name__}: {exc}"
        done.append((p, before))
    data.time = v["t"]
    data.qpos[:] = v["qpos"]
    if model.nmocap:
        data.mocap_pos[:] = v["mocap_pos"].reshape(model.nmocap, 3)
        data.mocap_quat[:] = v["mocap_quat"].reshape(model.nmocap, 4)
    if model.neq:
        data.eq_active[:] = v["eq_active"]
    return True, "ok"


def _rollback(model, done):
    for p, before in reversed(done):
        try:
            p.unpack(model, before)
        except Exception:  # noqa: BLE001 - best effort; the refusal is already reported
            pass
