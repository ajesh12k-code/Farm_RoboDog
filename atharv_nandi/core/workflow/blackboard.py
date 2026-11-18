"""Blackboard namespaces and key access control (WP34; WORKFLOWS-SPEC A.3 "Blackboard namespaces").

| Namespace                          | Written by                                   | Read by       |
|------------------------------------|----------------------------------------------|---------------|
| ``/sensors/*``                     | adapters fed by WP36 / WP7 hooks only        | conditions    |
| ``/safety/*``                      | the L0 mirror (WP6, WP24, WP28) only         | conditions    |
| ``/energy/*`` ``/weather/*`` ``/rth/*`` | WP37 / WP38 adapters only               | conditions    |
| ``/wf/<run_id>/*``                 | the workflow (``set``, primitives)           | the workflow  |

Two layers enforce it:

* **validate time** (pure Python, no py_trees): ``yaml_key(key)`` accepts a YAML ``set()`` key only when it is
  relative (``foo`` -> ``/wf/<run_id>/foo``) or under ``/wf/``; anything else is refused (``set_outside_wf``).
* **run time** (py_trees 2.2.3 key access control): every source namespace is written through ONE mirror client
  that holds ``EXCLUSIVE_WRITE`` on its keys, conditions get ``READ`` clients, and a run's ``set`` gets ``WRITE``
  only on ``/wf/<run_id>/<key>``. A behaviour that tries to write a condition source raises ``AttributeError``
  (py_trees: no write access), and a second writer cannot even register (exclusive-write conflict).

py_trees keeps one process-global blackboard; the mirror clients are created once per process (``mirror()``),
runs are isolated by their ``/wf/<run_id>/`` prefix and ``RunBlackboard.close()`` deletes that prefix.
The activity stream (a debug stream) stays OFF unless ``debug=True`` (the 2 ms tick gate, A.1).
"""
import re
from typing import Any, Dict, Tuple

from .primitives import registry as reg

WF = "/wf/"
SOURCE_NAMESPACES = ("/sensors/", "/safety/", "/energy/", "/weather/", "/rth/")
NAMESPACES = {
    "/sensors/": "adapters fed by WP36/WP7 hooks only",
    "/safety/": "L0 mirror (WP6, WP24, WP28) only",
    "/energy/": "WP37/WP38 adapters only",
    "/weather/": "WP37/WP38 adapters only",
    "/rth/": "WP37/WP38 adapters only",
    "/wf/": "the workflow (set, primitives); the only YAML-writable keys",
}
_SEG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")


def yaml_key(key: Any) -> Tuple[bool, str]:
    """(ok, canonical-or-reason) for a YAML ``set()`` key. Canonical keys are relative to ``/wf/<run_id>/``."""
    if not isinstance(key, str) or not key.strip():
        return False, "key must be a non-empty string"
    k = key.strip()
    if k.startswith("/"):
        if not k.startswith(WF):
            ns = "/" + k.split("/")[1] + "/" if k.count("/") >= 2 else k
            owner = NAMESPACES.get(ns, "not a workflow namespace")
            return False, f"{k!r} is outside /wf/ ({ns}: {owner})"
        k = k[len(WF):]
    parts = k.split("/")
    if any(not _SEG.match(p) for p in parts):
        return False, f"bad key segment in {key!r}"
    return True, k


def run_key(run_id: str, rel: str) -> str:
    return f"{WF}{run_id}/{rel}"


def source_key(cond_id: str) -> str:
    return reg.CONDITIONS[cond_id].source


# ---------------------------------------------------------------------------------------------- run time
_MIRROR = None


class _Mirror:
    """The single writer of every condition source key (one py_trees client per namespace, EXCLUSIVE_WRITE).

    Writes only values that changed (a 10 Hz mirror of ~36 keys stays cheap)."""

    def __init__(self):
        import py_trees
        self.py_trees = py_trees
        self.last: Dict[str, Any] = {}
        self.clients: Dict[str, Any] = {}
        self.key_client: Dict[str, Any] = {}
        for c in reg.CONDITIONS.values():
            self._claim(c.source)

    def _claim(self, key: str):
        ns = "/" + key.split("/")[1] + "/"
        cl = self.clients.get(ns)
        if cl is None:
            name = {"/safety/": "L0 mirror", "/sensors/": "sensor adapters"}.get(ns, "energy/weather/rth adapters")
            cl = self.py_trees.blackboard.Client(name=f"{name} {ns}")
            self.clients[ns] = cl
        cl.register_key(key=key, access=self.py_trees.common.Access.EXCLUSIVE_WRITE)
        self.key_client[key] = cl

    def write(self, key: str, value: Any):
        if key not in self.key_client:
            if not key.startswith(SOURCE_NAMESPACES):
                raise PermissionError(f"the mirror writes source namespaces only, not {key!r}")
            self._claim(key)
        storage = self.py_trees.blackboard.Blackboard.storage
        if key in storage and key in self.last and self.last[key] == value and type(self.last[key]) is type(value):
            return
        self.key_client[key].set(key, value)
        self.last[key] = value


def mirror() -> _Mirror:
    """The process-wide source-key writer (created on first use; needs py_trees)."""
    global _MIRROR
    if _MIRROR is None:
        _MIRROR = _Mirror()
    return _MIRROR


def set_debug(enabled: bool) -> None:
    import py_trees
    if enabled:
        py_trees.blackboard.Blackboard.enable_activity_stream(maximum_size=2000)
    else:
        py_trees.blackboard.Blackboard.disable_activity_stream()


class RunBlackboard:
    """The blackboard view of one run: READ on condition sources, WRITE on ``/wf/<run_id>/*`` only."""

    def __init__(self, run_id: str, debug: bool = False):
        import py_trees
        self.py_trees = py_trees
        self.run_id = str(run_id)
        self.prefix = f"{WF}{self.run_id}/"
        set_debug(debug)
        self.mirror = mirror()
        self.cond = py_trees.blackboard.Client(name=f"conditions {self.run_id}")
        for c in reg.CONDITIONS.values():
            self.cond.register_key(key=c.source, access=py_trees.common.Access.READ)
        self.wf = py_trees.blackboard.Client(name=f"workflow {self.run_id}")
        self._wf_keys = set()

    # ---------------------------------------------------------------- conditions (READ)
    def condition(self, cond_id: str, default: bool = False) -> bool:
        key = source_key(cond_id)
        try:
            return bool(self.cond.get(key))
        except KeyError:
            return default

    def read_source(self, key: str, default: Any = None) -> Any:
        try:
            return self.cond.get(key)
        except KeyError:
            return default

    def try_write_source(self, key: str, value: Any):
        """What a behaviour that tried to spoof a condition would do: py_trees raises ``AttributeError``."""
        self.cond.set(key, value)

    # ---------------------------------------------------------------- the run's own namespace (WRITE)
    def set(self, key: str, value: Any) -> str:
        ok, rel = yaml_key(key)
        if not ok:
            raise PermissionError(f"set() refused: {rel}")
        full = run_key(self.run_id, rel)
        if full not in self._wf_keys:
            self.wf.register_key(key=full, access=self.py_trees.common.Access.WRITE)
            self._wf_keys.add(full)
        self.wf.set(full, value)
        return full

    def get(self, key: str, default: Any = None) -> Any:
        ok, rel = yaml_key(key)
        if not ok:
            raise PermissionError(f"get() refused: {rel}")
        full = run_key(self.run_id, rel)
        if full not in self._wf_keys:
            return default
        try:
            return self.wf.get(full)
        except KeyError:
            return default

    def snapshot(self) -> Dict[str, Any]:
        """The ``/wf/<run_id>`` whitelist for a checkpoint (relative keys)."""
        out = {}
        for full in sorted(self._wf_keys):
            try:
                out[full[len(self.prefix):]] = self.wf.get(full)
            except KeyError:
                continue
        return out

    def restore(self, values: Dict[str, Any]) -> None:
        for k, v in (values or {}).items():
            self.set(k, v)

    def close(self) -> None:
        bb = self.py_trees.blackboard.Blackboard
        for full in list(self._wf_keys):
            bb.unset(full)
        self.wf.unregister(clear=False)
        self.cond.unregister(clear=False)
        self._wf_keys.clear()


def mirror_values(values: Dict[str, Any]) -> None:
    """Write condition sources: ``{cond_id or /ns/key: value}`` through the mirror (adapters / L0 mirror)."""
    m = mirror()
    for k, v in values.items():
        m.write(k if k.startswith("/") else source_key(k), v)
