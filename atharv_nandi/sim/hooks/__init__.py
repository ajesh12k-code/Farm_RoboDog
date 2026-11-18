"""Control-rate hook registry (WP0; MVP-SPEC 5.1, design rule 9; ORCHARD-MVP-SPEC 3.5).

A work package adds per-step behaviour by dropping one file ``atharv_nandi/sim/hooks/<name>.py`` that defines
``register(hooks)`` and calls ``hooks.register(...)`` in it. ``sim_runner`` (frozen after WP0) and the
WP0b ``sim_driver`` load every hook file with ``load_all()`` and drive the returned ``HookRegistry``:

    registry.substep(m, d, bus)   before EVERY ``mj_step``: hooks registered with ``every_step=True``
                                  (the per-physics-step substep point, e.g. WP24 mast interpolation)
    registry.tick(m, d, bus)      after every ``mj_step``: each rate hook whose period has elapsed in sim
                                  time (default 50 Hz), at most once per call
    registry.reset(m, d)          after a world reset (sim time back to 0) and BEFORE ``mj_forward``: a hook's
                                  ``reset`` restores the model fields it mutated, then kinematics run on them

Hook signature: ``fn(m, d, t, bus)``; ``t`` is ``d.time``. ``mutates_model=True`` declares that the hook
writes model fields (body_pos, eq_data, contype, ...): those writes must be undone by the hook's reset
and reported through a sim-state mutation provider (``controllers.sim_state``).

``Bus`` is the in-process exchange between hooks and the node: ``bus.publish(topic, payload)`` queues a
JSON payload that ``sim_runner`` publishes on ``topic`` (std_msgs/String) and ``bus.latest(topic)`` reads
the newest JSON message received on a topic a hook declared in ``topics_in``. ``bus.arrival_count(topic)`` counts the
deliveries (a new message is a new count, even when its value repeats). ``bus.since(topic, count)`` returns the
messages delivered after a hook's last count, oldest first, from a window of the newest ``HISTORY_N`` per topic,
plus how many fell out of that window, so a hook that ticks slower than the deliveries still sees each one
(``sim_runner`` itself still hands over only the newest message per topic per sim batch). ``bus.publish`` encodes the
payload at once (numpy scalars and arrays become JSON numbers and lists; NaN / inf and anything else that is
not JSON raise ``ValueError`` inside the publishing hook).

Rules ``sim_runner`` enforces (frozen after WP0):
    ctrl          hooks own the appended actuators (index >= 25) only. The runner re-writes the robot
                  actuators 0..24 before every ``mj_step``, so a hook write to them never reaches physics.
    hold          ``bus.values["hold"]`` is ``None`` while the robot runs and ``{"reason", "since"}``
                  while the runner safe-holds (e-stop, supervisor lost, hook fault). The runner holds the robot
                  actuators itself; a hook that drives an appended actuator must brake or refuse on it (WP24
                  mast e-stop brake, WP2 swap refused while e-stopped). ``bus.holding()`` is the shortcut.
    faults        an exception in a hook (``fn`` or ``reset``) disables that hook for the rest of the process,
                  is logged and reported in the heartbeat (``hook_faults``), and puts the runner into safe hold
                  with reason ``hook_fault`` (left by an ``/nandi/estop_reset`` like every hold). It never ends
                  the physics node.
    rate          a rate above the physics rate (1 / timestep) runs once per step; ``rate_capped(dt)`` lists
                  those hooks and the runner warns about them at start.
"""
import collections
import glob
import importlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

HOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RATE_HZ = 50.0
REASON_HOOK_FAULT = "hook_fault"   # safe-hold reason when a hook raised
_EPS = 1e-9
HISTORY_N = 32                     # Bus.since window per topic


def _json_default(obj):
    """numpy scalars / arrays (anything with ``tolist`` or ``item``) as plain JSON values."""
    for attr in ("tolist", "item"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return fn()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def encode_payload(payload: Any) -> str:
    """A bus payload as the String data published for it: ``str`` as is, anything else strict JSON."""
    if isinstance(payload, str):
        return payload
    try:
        return json.dumps(payload, default=_json_default, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bus payload is not JSON: {exc}") from None


class Bus:
    """Hook <-> node exchange. Outbox entries are (topic, payload); the inbox keeps the newest payload per topic."""

    def __init__(self):
        self.outbox: List[Tuple[str, Any]] = []
        self.inbox: Dict[str, Any] = {}
        self.arrivals: Dict[str, int] = {}     # topic -> number of deliveries (a new message, whatever its value)
        self.history: Dict[str, Any] = {}       # topic -> deque of the newest HISTORY_N payloads (``since``)
        # Shared state between hooks and the runner. The runner owns "hold" (None, or {"reason", "since"}
        # while it safe-holds); other keys are free for hooks.
        self.values: Dict[str, Any] = {"hold": None}

    def publish(self, topic: str, payload: Any):
        """Queue ``payload`` for ``topic``; encoded now (``encode_payload``), so a bad payload fails in the hook."""
        if not isinstance(topic, str) or not topic.startswith("/nandi/"):
            raise ValueError(f"bus topic {topic!r} must start with /nandi/")
        self.outbox.append((topic, encode_payload(payload)))

    def holding(self) -> bool:
        """True while sim_runner safe-holds the robot (e-stop, supervisor lost, hook fault)."""
        return self.values.get("hold") is not None

    def drain(self) -> List[Tuple[str, Any]]:
        out, self.outbox = self.outbox, []
        return out

    def deliver(self, topic: str, payload: Any):
        self.inbox[topic] = payload
        self.arrivals[topic] = self.arrivals.get(topic, 0) + 1
        self.history.setdefault(topic, collections.deque(maxlen=HISTORY_N)).append(payload)

    def arrival_count(self, topic: str) -> int:
        """How many messages were delivered on ``topic`` (a hook compares it with its last count to see a new
        message, including a repeat of the same value, ``null`` or ``{}``)."""
        return self.arrivals.get(topic, 0)

    def latest(self, topic: str, default: Any = None) -> Any:
        return self.inbox.get(topic, default)

    def since(self, topic: str, count: int) -> Tuple[List[Any], int]:
        """(messages delivered on ``topic`` after arrival number ``count``, oldest first; how many of those are no
        longer in the ``HISTORY_N`` window and were lost)."""
        n = self.arrivals.get(topic, 0)
        want = max(0, n - int(count))
        hist = self.history.get(topic, ())
        got = list(hist)[len(hist) - min(want, len(hist)):] if want else []
        return got, want - len(got)


@dataclass
class Hook:
    name: str
    fn: Callable
    rate_hz: float = DEFAULT_RATE_HZ
    mutates_model: bool = False
    every_step: bool = False
    topics_in: Tuple[str, ...] = ()
    reset_fn: Optional[Callable] = None
    calls: int = 0
    error: Optional[str] = None        # set (and the hook disabled) when it raised
    _next_t: float = field(default=0.0, repr=False)


class HookRegistry:
    def __init__(self):
        self._hooks: Dict[str, Hook] = {}
        self._faults: List[Tuple[str, str]] = []

    # ------------------------------------------------------------------ registration
    def register(self, name: str, fn: Callable, rate_hz: float = DEFAULT_RATE_HZ, mutates_model: bool = False,
                 every_step: bool = False, topics_in: Iterable[str] = (), reset: Optional[Callable] = None) -> Hook:
        """Add a hook. ``every_step=True`` runs it before every physics step (rate_hz ignored)."""
        if name in self._hooks:
            raise ValueError(f"hook {name!r} registered twice")
        if not every_step and not (rate_hz > 0 and math.isfinite(rate_hz)):
            raise ValueError(f"hook {name!r}: rate_hz must be > 0")
        topics = tuple(topics_in)
        for t in topics:
            if not t.startswith("/nandi/"):
                raise ValueError(f"hook {name!r}: topic {t!r} must start with /nandi/")
        h = Hook(name, fn, float(rate_hz), bool(mutates_model), bool(every_step), topics, reset)
        self._hooks[name] = h
        return h

    def unregister(self, name: str):
        self._hooks.pop(name, None)

    def hooks(self) -> List[Hook]:
        return [self._hooks[n] for n in sorted(self._hooks)]

    def __len__(self):
        return len(self._hooks)

    def topics_in(self) -> List[str]:
        return sorted({t for h in self._hooks.values() for t in h.topics_in})

    def model_mutating(self) -> List[str]:
        return [h.name for h in self.hooks() if h.mutates_model]

    def rate_capped(self, dt: float) -> List[str]:
        """Rate hooks declared faster than the physics rate 1/dt (they run once per step)."""
        return [h.name for h in self.hooks() if not h.every_step and h.rate_hz * dt > 1.0 + 1e-9]

    def faulted(self) -> Dict[str, str]:
        """name -> error of every hook disabled by an exception."""
        return {h.name: h.error for h in self.hooks() if h.error is not None}

    def drain_faults(self) -> List[Tuple[str, str]]:
        """(name, error) of the hooks that faulted since the last call."""
        out, self._faults = self._faults, []
        return out

    def _fault(self, h: Hook, where: str, exc: BaseException):
        h.error = f"{where}: {type(exc).__name__}: {exc}"
        self._faults.append((h.name, h.error))

    @property
    def has_substep(self) -> bool:
        return any(h.every_step for h in self._hooks.values())

    def table(self) -> List[dict]:
        """The documented hook table (name, rate, every_step, mutates_model): published with the layout."""
        return [{"name": h.name, "rate_hz": None if h.every_step else h.rate_hz, "every_step": h.every_step,
                 "mutates_model": h.mutates_model, "topics_in": list(h.topics_in)} for h in self.hooks()]

    # ------------------------------------------------------------------ driving
    # Every hook call is isolated: an exception disables that hook (``error``) and is queued for
    # ``drain_faults``; it never propagates into the physics loop.
    def substep(self, m, d, bus: Bus):
        for h in self.hooks():
            if h.every_step and h.error is None:
                try:
                    h.fn(m, d, float(d.time), bus)
                except Exception as exc:  # noqa: BLE001 - isolate any hook bug
                    self._fault(h, "substep", exc)
                    continue
                h.calls += 1

    def tick(self, m, d, bus: Bus):
        t = float(d.time)
        for h in self.hooks():
            if h.every_step or h.error is not None:
                continue
            if t + _EPS >= h._next_t:
                period = 1.0 / h.rate_hz
                # next due time on the period grid strictly after t (no burst after a stall)
                h._next_t = (math.floor((t + _EPS) / period) + 1) * period
                try:
                    h.fn(m, d, t, bus)
                except Exception as exc:  # noqa: BLE001
                    self._fault(h, "tick", exc)
                    continue
                h.calls += 1

    def reset(self, m=None, d=None):
        """Rebase every hook and call its reset (faulted hooks too: they may have mutated the model)."""
        for h in self.hooks():
            h._next_t = 0.0
            h.calls = 0
            if h.reset_fn is not None:
                try:
                    h.reset_fn(m, d)
                except Exception as exc:  # noqa: BLE001
                    if h.error is None:
                        self._fault(h, "reset", exc)


def hook_files(hooks_dir: str = HOOKS_DIR) -> List[str]:
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(hooks_dir, "*.py")))
    return [n for n in names if not n.startswith("_")]


def import_all(hooks_dir: Optional[str] = None) -> List[str]:
    """Import every hook file WITHOUT registering its hooks: runs its import-time declarations (such as
    ``sim_state.register_provider``) in processes that do not drive hooks (renderers, recorder)."""
    names = hook_files(hooks_dir or HOOKS_DIR)
    for n in names:
        importlib.import_module(f"{__name__}.{n}")
    return names


def load_all(registry: Optional[HookRegistry] = None, names: Optional[Iterable[str]] = None) -> HookRegistry:
    """Import every ``atharv_nandi/sim/hooks/<name>.py`` (or just ``names``) and call its ``register(registry)``."""
    reg = registry if registry is not None else HookRegistry()
    for n in (list(names) if names is not None else hook_files()):
        mod = importlib.import_module(f"{__name__}.{n}")
        fn = getattr(mod, "register", None)
        if not callable(fn):
            raise AttributeError(f"atharv_nandi/sim/hooks/{n}.py has no register(hooks)")
        fn(reg)
    return reg
