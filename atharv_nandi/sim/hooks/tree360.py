"""TREE 360 hook (WP25; ORCHARD-MVP-SPEC 5.4, MISSION-PRESETS 2): ticks one ``controllers.tree360.Tree360Run``.

``register(hooks)`` adds the ``tree360`` rate hook (50 Hz, ``topics_in`` = ``/nandi/tree360/cmd``). It is a no-op
until a run is attached (``bus.values["tree360"]``):

    in-process     ``start(bus, m, d, tree_id, params, gait=driver.gait)`` (SimDriver tests, tools, WP27 presets)
    topic          ``/nandi/tree360/cmd`` ``{cmd: start, tree_id, params?}`` | ``{cmd: abort}`` |
                   ``{cmd: pause | resume}`` (console Pause / Resume); ``params.out_root`` is refused (outputs
                   go under NANDI_RUNS_DIR). The run is built from the world's field registry; without a gait the
                   drive commands go to ``bus.values["tree360_drive"]`` and ``/nandi/tree360/drive`` (bridge to
                   ``/nandi/cmd_vel``: change request, the WP25 contract notes)

While a run is attached the hook ticks it (after the ``lidar`` hook: it reads the fresh trunk fit), shares
``bus.values["orbit"]`` (``{active, tree_id, arc: [[x, y, yaw]] base frame}``, read by the WP6 safety fields for the
orbit person rule) and publishes ``/nandi/tree360/progress`` at 2 Hz and on a phase change. When the run ends it
publishes ``/nandi/tree/record`` (the ``nandi.tree/1`` fragment) and ``/nandi/deepscan/request``
(``nandi.deepscan/1``, only when the visit is done and its coverage passes) once, keeps the result in
``bus.values["tree360_result"]`` and detaches. A malformed command is answered on ``/nandi/tree360/progress``
with ``{error}``; the hook never raises (a fault inside the run becomes an ordered abort ``internal_error:<type>``,
a hook fault would safe-hold the robot).
Physics is untouched: no ctrl, no model field writes (the mast moves only through WP24 requests; an assisted
``entry: teleport`` writes qpos, station tier only).
"""
import math
from typing import Optional

HOOK_NAME = "tree360"
RATE_HZ = 50.0
CMD_TOPIC = "/nandi/tree360/cmd"
PROGRESS_TOPIC = "/nandi/tree360/progress"
RECORD_TOPIC = "/nandi/tree/record"
DEEPSCAN_TOPIC = "/nandi/deepscan/request"
DRIVE_TOPIC = "/nandi/tree360/drive"
ASSIST_TOPIC = "/nandi/assist"
PROGRESS_PERIOD_S = 0.5
DRIVE_REPUBLISH_S = 0.1    # the console relay drops a drive older than 0.5 s: keep the latest one fresh
BUS_KEY = "tree360"
RESULT_KEY = "tree360_result"


class _State:
    def __init__(self):
        self.cmd_count = 0
        self.last_progress = -1.0
        self.last_phase = None
        self.last_drive = None
        self.last_drive_t = -1.0
        self.assist_sent = False


def _st(bus) -> _State:
    st = bus.values.get("_tree360_hook")
    if st is None:
        st = bus.values["_tree360_hook"] = _State()
    return st


def _field_and_registry(m):
    from atharv_nandi.core import model_layout
    from atharv_nandi.sim.world import compose, trees
    import os
    lay = model_layout.current()
    world = (lay.world if lay is not None and getattr(lay, "world", None) else compose.world_name())
    seed = int(os.environ.get("NANDI_SEED", "7"))
    field = compose.load_field(world, seed)
    return field, trees.registry(field)


def start(bus, m, d, tree_id: str, params: Optional[dict] = None, gait=None, registry=None, field=None,
          canopy=None):
    """Build and attach a ``Tree360Run`` for ``tree_id`` (returns it). Raises on a bad tree id or params."""
    from atharv_nandi.core import coverage as cv
    from atharv_nandi.core import tree360 as t3
    cfg = t3.Tree360Config.from_params(params)
    if registry is None or field is None:
        f2, r2 = _field_and_registry(m)
        field = field if field is not None else f2
        registry = registry if registry is not None else r2
    entry = next((e for e in registry if e.get("tree_id") == tree_id), None)
    if entry is None:
        raise ValueError(f"tree360: tree {tree_id!r} not in the registry")
    geofence = field.get("geofence")
    capture = cfg.capture
    if capture == "auto":
        capture = "gl" if gait is not None and _gl_ok() else ("perception" if gait is None else "geometry")
    io = t3.SimIO(m, d, bus, gait=gait, capture=capture, privacy=cfg.privacy, geofence=geofence,
                  width=cfg.width, height=cfg.height)
    if canopy is None:
        canopy = cv.canopy_from_model(m, d, tree_id)
    lid = bus.values.get("lidar") or {}
    clusters = lid.get("clusters") or [] if isinstance(lid, dict) else []
    run = t3.Tree360Run(entry, registry, cfg, io, geofence=geofence, clusters=clusters, canopy=canopy)
    attach(bus, run)
    return run


_GL = None


def _gl_ok() -> bool:
    """GL probe in a subprocess (a broken GL stack can abort the interpreter instead of raising)."""
    global _GL
    if _GL is None:
        import subprocess
        import sys
        code = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); "
                "r = mujoco.Renderer(m, 8, 8); r.close()")
        try:
            _GL = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode == 0
        except Exception:     # noqa: BLE001
            _GL = False
    return _GL


def attach(bus, run) -> None:
    bus.values[BUS_KEY] = run
    bus.values.pop(RESULT_KEY, None)
    bus.values.pop("tree360_drive", None)   # never republish the previous run's last drive for this one
    st = _st(bus)
    st.last_phase, st.last_progress = None, -1.0
    st.last_drive, st.last_drive_t = None, -1.0


def detach(bus) -> None:
    run = bus.values.pop(BUS_KEY, None)
    bus.values["orbit"] = {"active": False}
    if run is not None:
        try:
            run.io.close()
        except Exception:     # noqa: BLE001
            pass


def _commands(m, d, t, bus):
    _ST = _st(bus)
    msgs, _lost = bus.since(CMD_TOPIC, _ST.cmd_count)
    _ST.cmd_count = bus.arrival_count(CMD_TOPIC)
    for msg in msgs:
        try:
            import json
            if isinstance(msg, str):
                msg = json.loads(msg)
            if not isinstance(msg, dict):
                raise ValueError("command must be a JSON object")
            cmd = msg.get("cmd")
            run = bus.values.get(BUS_KEY)
            if cmd == "start":
                if run is not None and not run.finished:
                    raise ValueError("a TREE 360 run is active")
                params = msg.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("params must be a JSON object")
                if "out_root" in params:
                    # the output root is the process's NANDI_RUNS_DIR: a console command never picks a path
                    raise ValueError("out_root is not accepted on /nandi/tree360/cmd (NANDI_RUNS_DIR)")
                start(bus, m, d, str(msg.get("tree_id")), params, gait=bus.values.get("tree360_gait"))
            elif cmd == "abort":
                if run is not None:
                    run.abort(str(msg.get("reason") or "operator_abort")[:64])
            elif cmd in ("pause", "resume"):
                if run is None or run.finished:
                    raise ValueError("no TREE 360 run is active")
                run.pause(cmd == "pause")
            else:
                raise ValueError(f"unknown cmd {cmd!r}")
        except Exception as exc:          # noqa: BLE001 - a bad command never faults the hook
            bus.publish(PROGRESS_TOPIC, {"error": f"{type(exc).__name__}: {exc}", "t": round(float(t), 3)})


def _tick(m, d, t, bus):
    """Never raises (a hook fault puts sim_runner into hook_fault safe hold): a fault outside the run detaches it."""
    try:
        _tick_inner(m, d, t, bus)
    except Exception as exc:              # noqa: BLE001
        try:
            bus.publish(PROGRESS_TOPIC, {"error": f"hook: {type(exc).__name__}: {exc}", "t": round(float(t), 3)})
            run = bus.values.get(BUS_KEY)
            if run is not None:
                bus.values[RESULT_KEY] = run.result or {"status": "aborted", "tree_id": getattr(run, "tree_id", None),
                                                        "abort_reason": f"internal_error:{type(exc).__name__}"}
            detach(bus)
        except Exception:                 # noqa: BLE001
            pass


def _tick_inner(m, d, t, bus):
    _ST = _st(bus)
    if bus.arrival_count(CMD_TOPIC) != _ST.cmd_count:
        _commands(m, d, t, bus)
    run = bus.values.get(BUS_KEY)
    if run is None:
        return
    try:
        run.tick()                         # Tree360Run.tick never raises; this is the last line of defence
    except Exception as exc:              # noqa: BLE001 - a hook fault would safe-hold the robot
        bus.publish(PROGRESS_TOPIC, {"error": f"tick: {type(exc).__name__}: {exc}", "t": round(float(t), 3)})
        try:
            run._fail(exc)
        except Exception:                 # noqa: BLE001
            pass
        if not run.finished:
            run.phase, run.status = "aborted", "aborted"
            run.result = run.result or {"status": "aborted", "abort_reason": f"internal_error:{type(exc).__name__}",
                                        "tree_id": run.tree_id, "record": None, "deepscan": None}
    bus.values["orbit"] = run.orbit_value
    drive = bus.values.get("tree360_drive")
    if run.io.gait is None and drive is not None and (drive is not _ST.last_drive
                                                      or t - _ST.last_drive_t >= DRIVE_REPUBLISH_S - 1e-9
                                                      or t < _ST.last_drive_t):
        _ST.last_drive, _ST.last_drive_t = drive, t
        bus.publish(DRIVE_TOPIC, drive)
    if run.assist and not _ST.assist_sent:
        _ST.assist_sent = True
        bus.publish(ASSIST_TOPIC, dict(run.assist, source="tree360", tree_id=run.tree_id))
    elif not run.assist:
        _ST.assist_sent = False
    if run.phase != _ST.last_phase or t - _ST.last_progress >= PROGRESS_PERIOD_S - 1e-9 or run.finished:
        _ST.last_phase, _ST.last_progress = run.phase, t
        bus.publish(PROGRESS_TOPIC, run.progress())
    if run.finished:
        res = run.result or {}
        if res.get("record"):
            bus.publish(RECORD_TOPIC, res["record"])
        if res.get("deepscan") and res.get("deepscan_requested"):
            bus.publish(DEEPSCAN_TOPIC, res["deepscan"])
        bus.values[RESULT_KEY] = res
        detach(bus)


def _reset(m, d):
    """Nothing in the model to restore. The hook has no bus here: an attached run sees the sim time go backwards on
    its next tick and ends with an ordered abort ``world_reset`` (records written, then detached)."""


def register(hooks):
    hooks.register(HOOK_NAME, _tick, rate_hz=RATE_HZ, topics_in=(CMD_TOPIC,), reset=_reset)
