"""WP3 hooks: kinematic transfers into the desap tray / papaya cradle and the crate unload (ORCHARD-MVP-SPEC 2.4,
1.2 step 11; MISSION-PRESETS 9). Logic in ``atharv_nandi/core/tools/kinematic_transfer.py``.

    transfer_kinematics   every physics step (substep point, before ``mj_step``): interlocks (runner hold, arm PARK,
                          gait stopped), the carried item's pose, the falling item's impact and settle check, the
                          riders' pose + velocity every 5 steps (100 Hz)
    transfer              50 Hz: commands from ``/nandi/transfer/cmd``; ``/nandi/transfer/state`` at 2 Hz and on
                          change, ``/nandi/transfer/events`` per event; ``bus.values["transfer"]`` = the state

Only in models with the WP1 tray and the WP23 fruit pool (orchard worlds); elsewhere every call returns at once and
nothing is published (the legacy world stays byte-identical). Commands (JSON object, ``seq`` echoed in events):

    {"op": "place", "fruit_id": "OT-1:f01", "target": "auto" | 0..7 | "cradle"}
    {"op": "unload"}   {"op": "pause"}   {"op": "resume"}   {"op": "abort"}   {"op": "release_slots"}
    [cmd, cmd, ...]    a batch (at most MAX_BATCH), applied in order in one tick

The hook inbox keeps only the newest message per topic (WP0 ``Bus``; sim_runner also coalesces between control
ticks), so commands sent within one 20 ms tick collapse into the last one: send a batch, or wait for the ``cmd``
event with your ``seq``. A lost message seen on the bus gives a ``cmd_lost`` event. A malformed command gives a
``cmd_refused`` event; it never faults the hook. Model mutations: the ballast fields
(body_mass / ipos / inertia / iquat, ancestors' subtreemass) and, on pool slots, contact flags and the trimmed pedicel
(those two are packed by WP23's ``pool`` provider) and the pool slots' ``dof_M0``. Sim-state provider ``transfer``
(v1): ``{ballast, state, m0}`` (``m0`` optional on unpack).
"""
import math
import weakref
from typing import Any, Optional

from atharv_nandi.core import sim_state
from atharv_nandi.core.tools import kinematic_transfer as kt

STATE_TOPIC = "/nandi/transfer/state"
EVENTS_TOPIC = "/nandi/transfer/events"
CMD_TOPIC = "/nandi/transfer/cmd"
RATE_HZ = 50.0
STATE_PERIOD_S = 0.5
OPS = ("place", "unload", "pause", "resume", "abort", "release_slots")
MAX_BATCH = 16


class Runtime:
    """Per-model hook state: the controller (pool attached when the fruit pool of this model is known)."""

    def __init__(self, m):
        self.m = m
        pool = None
        try:
            from atharv_nandi.core import fruit_pool as fp
            pool = fp.FruitPool(m)
        except Exception:  # noqa: BLE001 - no pool / plans unknown in this process: ballast-only (renderers)
            pool = None
        self.ctrl = kt.TransferController(m, pool)
        self.published_seq = self.ctrl.event_seq
        self.next_state_t = 0.0
        self.last_key = None
        self.arrivals_seen = None      # bus arrival count of CMD_TOPIC already accounted for

    # -------------------------------------------------------------- inputs
    def inputs(self, d, bus) -> kt.GateInputs:
        L = self.ctrl.L
        return kt.GateInputs(arm_parked=L.arm_parked(d), gait_stopped=L.gait_stopped(d), holding=bus.holding(),
                             crown_stowed=crown_stowed(bus, self.m))

    # -------------------------------------------------------------- hooks
    def substep(self, d, bus):
        c = self.ctrl
        if c.pool is None or (not c.items):
            return
        c.substep(d, self.inputs(d, bus) if c.busy() else kt.GateInputs(holding=bus.holding()))

    def command(self, raw: Any, d, bus):
        c = self.ctrl
        t = float(d.time)
        seq = raw.get("seq") if isinstance(raw, dict) else None
        why = check_command(raw)
        if why:
            c._event(t, "cmd_refused", reason=why, cmd_seq=seq)
            return
        op = raw["op"]
        if op == "place":
            res = c.place(d, raw["fruit_id"], raw.get("target", "auto"), self.inputs(d, bus))
        elif op == "unload":
            res = c.unload(d, self.inputs(d, bus))
        elif op == "pause":
            c.pause("operator")
            res = None
        elif op == "resume":
            c.resume()
            res = None
        elif op == "abort":
            res = c.abort(d)
        else:
            res = c.release_crate_slots(d)
        c._event(t, "cmd", op=op, cmd_seq=seq, ok=bool(res) if res is not None and not isinstance(res, list) else True,
                 reason=getattr(res, "reason", ""))

    def tick(self, d, t: float, bus):
        c = self.ctrl
        had = CMD_TOPIC in bus.inbox
        raw = bus.inbox.pop(CMD_TOPIC, None)
        n = bus.arrival_count(CMD_TOPIC)
        if self.arrivals_seen is None:
            self.arrivals_seen = n - (1 if had else 0)
        lost = n - self.arrivals_seen - (1 if had else 0)
        self.arrivals_seen = n
        if lost > 0:
            c._event(float(d.time), "cmd_lost", n=lost)
        if isinstance(raw, list):
            if not raw or len(raw) > MAX_BATCH:
                c._event(float(d.time), "cmd_refused", reason=f"batch must hold 1..{MAX_BATCH} commands",
                         cmd_seq=None)
            else:
                for cmd in raw:
                    self.command(cmd, d, bus)
        elif raw is not None:
            self.command(raw, d, bus)
        new = [e for e in c.events if e["seq"] > self.published_seq]
        for e in new:
            bus.publish(EVENTS_TOPIC, e)
        if new:
            self.published_seq = new[-1]["seq"]
        # cheap change key first; the full status is built only when it is published
        key = (c.busy(), c.paused, tuple(c.cells), c.cradle, c.job["phase"] if c.job else None, c.event_seq)
        if key != self.last_key or t + 1e-9 >= self.next_state_t:
            st = c.status()
            st["t"] = round(float(t), 3)
            bus.values["transfer"] = st
            bus.publish(STATE_TOPIC, st)
            self.next_state_t = t + STATE_PERIOD_S
            self.last_key = key


def crown_stowed(bus, m=None) -> bool:
    """MISSION-PRESETS 9 precondition for the unload, from WP24's ``bus.values["mast"]`` (50 Hz): stowed when the
    extension is at most ``kt.CROWN_STOWED_EXT_M`` (= WP13's ``dock_logic.CROWN_STOWED_EXT_M``). Fails closed like
    ``dock_logic.crown_stowed``: without a mast state (mast hook not loaded or not ticked yet) it is stowed only in a
    model without the mast (``m`` given; no model -> stowed); a malformed extension counts as not stowed."""
    st = getattr(bus, "values", {}).get("mast")
    if not isinstance(st, dict):
        if m is None:
            return True
        try:
            from atharv_nandi.sim.hooks import mast as hm
            return not hm.has_mast(m)
        except Exception:  # noqa: BLE001 - fail safe
            return False
    if st.get("has_mast") is False:
        return True
    ext = st.get("ext_m")
    if isinstance(ext, bool) or not isinstance(ext, (int, float)) or not math.isfinite(ext):
        return False
    return float(ext) <= kt.CROWN_STOWED_EXT_M


def _mast_gate(ext_from, ext_to, ctx):
    """WP24 raise gate: no crown raise while an unload runs (MISSION-PRESETS 9, crown stowed: no arm swing over the
    crate with the crown up). Lowering is always allowed."""
    from atharv_nandi.core.interfaces import GateDecision
    m = ctx.get("m") if isinstance(ctx, dict) else None
    rt = _RUNTIMES.get(m) if m is not None else None
    if rt is not None and rt.ctrl.unloading and rt.ctrl.busy() and ext_to > ext_from + 1e-9:
        return GateDecision.refuse("transfer", "transfer_unloading")
    return GateDecision.ok("transfer")


def check_command(raw: Any) -> str:
    """'' for a well-formed command, else the reason it is refused."""
    if not isinstance(raw, dict):
        return "not an object"
    op = raw.get("op")
    if op not in OPS:
        return f"op must be one of {OPS}"
    if op == "place":
        if not isinstance(raw.get("fruit_id"), str) or not raw["fruit_id"]:
            return "place needs fruit_id"
        tgt = raw.get("target", "auto")
        if not (tgt in ("auto", "cradle") or (isinstance(tgt, int) and not isinstance(tgt, bool)
                                               and 0 <= tgt < kt.N_CELLS)):
            return "target must be auto, cradle or a cell 0..7"
    return ""


_RUNTIMES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_NOT_APPLICABLE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def applies(m) -> bool:
    """Models with the WP1 tray and the WP23 fruit pool (orchard worlds)."""
    if m in _RUNTIMES:
        return True
    if m in _NOT_APPLICABLE:
        return False
    import mujoco
    from atharv_nandi.sim.world.modules import fruit_pool as pool_mod
    ok = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, kt.CELL_SITES[0]) >= 0 and pool_mod.model_state(m) is not None
    if not ok:
        _NOT_APPLICABLE[m] = True
    return ok


def runtime_for(m) -> Optional[Runtime]:
    rt = _RUNTIMES.get(m)
    if rt is None and applies(m):
        rt = _RUNTIMES[m] = Runtime(m)
    return rt


def controller_for(m) -> Optional[kt.TransferController]:
    """The transfer controller of model m (None outside orchard worlds). Task layers (WP8, WP13, WP27) call it."""
    rt = runtime_for(m)
    return None if rt is None else rt.ctrl


# ---------------------------------------------------------------------- hook functions
def _substep(m, d, t, bus):
    rt = runtime_for(m)
    if rt is not None:
        rt.substep(d, bus)


def _tick(m, d, t, bus):
    rt = runtime_for(m)
    if rt is not None:
        rt.tick(d, t, bus)


def _reset(m, d):
    if m is None:
        return
    rt = runtime_for(m)
    if rt is not None:
        rt.ctrl.reset(d)
        rt.published_seq = rt.ctrl.event_seq
        rt.last_key = None
        rt.next_state_t = 0.0


def register(hooks):
    from atharv_nandi.sim.hooks import mast as mast_hook
    mast_hook.register_gate(_mast_gate)
    hooks.register("transfer_kinematics", _substep, every_step=True, mutates_model=True, reset=_reset)
    hooks.register("transfer", _tick, rate_hz=RATE_HZ, topics_in=(CMD_TOPIC,))


# ---------------------------------------------------------------------- sim-state provider
def _finite_list(v, n, name):
    if not isinstance(v, list) or len(v) != n or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                       or not math.isfinite(x) for x in v):
        raise ValueError(f"transfer {name} must be {n} finite numbers")


def _pack(m) -> dict:
    rt = runtime_for(m)
    if rt is None:
        return {}
    return {"ballast": rt.ctrl.ballast_fields(), "state": rt.ctrl.to_dict(), "m0": rt.ctrl.m0_fields()}


def _unpack(m, payload: dict) -> None:
    rt = runtime_for(m)
    if not payload:
        return
    if rt is None:
        raise ValueError("transfer mutations for a world without the tray and fruit pool")
    if not isinstance(payload, dict) or set(payload) not in ({"ballast", "state"}, {"ballast", "state", "m0"}):
        raise ValueError("transfer payload needs {ballast, state} and optionally m0")
    b = payload["ballast"]
    if not isinstance(b, dict) or set(b) != {"mass", "ipos", "inertia", "iquat"}:
        raise ValueError("transfer ballast needs {mass, ipos, inertia, iquat}")
    mass = b["mass"]
    base = rt.ctrl._ballast0["mass"]
    if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(mass) \
            or not base - 1e-12 <= mass <= base + kt.PAYLOAD_CAP_KG + 1e-6:
        raise ValueError(f"transfer ballast.mass {mass!r} outside [{base}, {base + kt.PAYLOAD_CAP_KG}]")
    _finite_list(b["ipos"], 3, "ballast.ipos")
    _finite_list(b["inertia"], 3, "ballast.inertia")
    _finite_list(b["iquat"], 4, "ballast.iquat")
    if min(b["inertia"]) <= 0:
        raise ValueError("transfer ballast.inertia must be > 0")
    pool = rt.ctrl.pool
    kt.check_state_dict(payload["state"], n_slots=None if pool is None else len(pool.slots))
    m0 = payload.get("m0")
    if m0 is not None:
        n = 0 if pool is None else len(pool.slots)
        if not isinstance(m0, list) or len(m0) != n:
            raise ValueError(f"transfer m0 must list {n} slots")
        for row in m0:
            _finite_list(row, 6, "m0[]")
            if min(row) <= 0:
                raise ValueError("transfer m0 must be > 0")
    rt.ctrl._write_ballast(mass, b["ipos"], b["inertia"], b["iquat"])
    rt.ctrl.load_dict(payload["state"])
    if m0 is not None:
        rt.ctrl.load_m0(m0)


sim_state.register_provider("transfer", _pack, _unpack, version=1, applies=applies)
