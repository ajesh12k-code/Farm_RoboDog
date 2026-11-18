"""Workflow engine hook (WP34; WORKFLOWS-SPEC A.0 L2 ticked at 10 Hz from a WP0b driver hook, A.4 checkpoints).

* registers the ``workflow_checkpoint`` sim-state mutation provider at import (``sim_state.discover_providers``
  imports every hook file), so ``set_state(get_state())`` round-trips the checkpoint store;
* ``register(hooks)`` adds the ``workflow`` rate hook (10 Hz). It is a no-op until a run supervisor attaches an
  engine with ``attach(bus, engine)`` (``bus.values["workflow"]``). While attached it ticks the engine, passes the
  runner / driver safe hold (``bus.values["hold"]``: ``estop`` latches the e-stop, any other reason freezes like
  a protective stop) and publishes the console status on ``/nandi/workflow/status`` at most every 0.5 s;
* a world reset clears the checkpoint store (the store goes with the world).

Importing this file does not import py_trees: processes without the workflow runtime (renderer, recorder, the
live image before the py_trees pin) load it unchanged. Physics is untouched (no ctrl, no model writes).
"""
from atharv_nandi.core.workflow import checkpoint

HOOK_NAME = "workflow"
RATE_HZ = 10.0
STATUS_TOPIC = "/nandi/workflow/status"
STATUS_PERIOD_S = 0.5            # <= 2 Hz console stream (A.4, WP43)
BUS_KEY = "workflow"

checkpoint.register_provider()


def attach(bus, engine) -> None:
    bus.values[BUS_KEY] = engine
    bus.values["workflow_status_t"] = None


def detach(bus) -> None:
    bus.values.pop(BUS_KEY, None)


def _tick(m, d, t, bus):
    eng = bus.values.get(BUS_KEY)
    if eng is None:
        return
    hold = bus.values.get("hold")
    eng.tick(l0=(hold or {}).get("reason") if hold else None)
    last = bus.values.get("workflow_status_t")
    if last is None or t - last >= STATUS_PERIOD_S - 1e-9:
        bus.values["workflow_status_t"] = t
        bus.publish(STATUS_TOPIC, eng.status())


def _reset(m, d):
    checkpoint.reset_hook(m, d)


def register(hooks):
    hooks.register(HOOK_NAME, _tick, rate_hz=RATE_HZ, reset=_reset)
