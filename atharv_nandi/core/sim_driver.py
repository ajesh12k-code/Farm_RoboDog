"""ROS-free, in-process driver for one Atharv Nandi world (WP0b; MVP-SPEC 5.1, ORCHARD-MVP-SPEC 3.5, 6).

One ``MjData``. ``SimDriver.step()`` advances ONE physics step in the same order ``sim_runner`` uses, plus
the parts that run in other nodes when ROS is up (gait, arm targets, task state machines), so task layers
(WP8, WP9, WP25, WP27-29) and ``tools/headless_eval.py`` test the real stack without ROS, a viewer or
wall-clock pacing:

    1. gait        every ``gait_every`` steps (GAIT_DT): ``TripodGaitGenerator.step(roll, pitch)`` -> leg targets
    2. ctrl        robot actuators 0..24 <- arm (7) + legs (18), or the safe-hold pose while holding, or the
                   re-arm blend (hold pose -> targets, smoothstep) after ``leave_hold``
    3. substep     ``hooks.substep(m, d, bus)`` (``every_step=True`` hooks), then every driver substep callback
                   ``fn(m, d, t, bus)`` (``add_substep``): the per-physics-step substep point (WP24 writes the
                   interpolated mast ``body_pos`` here)
    4. ctrl again  robot actuators re-written: a hook or substep callback never drives them (WP0 rule 1)
    5. mj_step     then the divergence check: a new ``mjWARN_BADQPOS/BADQVEL/BADQACC`` (MuJoCo silently reset
                   MjData to qpos0) sets ``driver.diverged`` and raises ``Diverged`` (a ``TimeCap``); every
                   later ``step()`` raises it again until ``reset()``
    6. tick        ``hooks.tick(m, d, bus)`` (rate hooks on the sim-time grid), then the driver's task state
                   machines (``add_task(name, fn, rate_hz)``, ``fn(driver)``, same grid rule), then the step
                   observers (``add_observer(fn)``, ``fn(driver)``, every step)
    7. bus         ALWAYS, also when 6 raised: queued ``bus.publish`` messages (from hooks, substep callbacks
                   or tasks) are moved to ``driver.published`` (and ``on_publish``); hook faults are drained:
                   the hook stays disabled, the driver enters safe hold ``hook_fault``

With no hook files, no substep callbacks and no tasks, steps 1, 2 and 5 are byte-for-byte the pre-WP0b
``tools/headless_eval.py`` loop (``legacy headless_eval unchanged``, tests/test_sim_driver.py).

Safe hold and reset follow ``sim_runner``: hold freezes the robot actuators at the current joint positions
(fingers keep their ctrl) and stops the gait; ``leave_hold`` drops any gait command given during the hold
(locomotion_service refuses cmd_vel while e-stopped) and blends from the hold pose to the arm / leg targets
over ``safety_logic.rearm_duration`` (config/safety.yaml ``rearm_*``) with ``smoothstep``; ``reset()`` while
holding stays held (hold pose = home pose, same reason: a reset never re-arms). Hook faults,
``bus.values["hold"]``, the reset order (``mj_resetDataKeyframe`` -> ``hooks.reset`` -> ``mj_forward``) and
rate capping are the same. Differences from ``sim_runner`` (documented, not hidden): the robot targets are
re-evaluated every step (``sim_runner`` holds one ROS target per catch-up batch); hold does not scale gains or
add the joint brake.

Rendering is pluggable: ``StubRenderer`` (unit tier: records the view, returns a blank frame, no GL) or
``MjRenderer`` (``mujoco.Renderer`` on the driver's own model and data, so model mutations made in a substep,
such as mast ``body_pos``, are in the frame; tests mark it ``gl``).
"""
import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import model_layout as mlay
from atharv_nandi.core.locomotion_logic import MotionInhibitGate, quat_to_roll_pitch
from atharv_nandi.core.safety_logic import load_safety_config, rearm_duration, smoothstep
from atharv_nandi.core.tripod_gait import TripodGaitGenerator

GAIT_DT = 0.02          # gait generator period (s), as locomotion_service / headless_eval
PUBLISHED_KEEP = 10000  # bus messages kept in ``driver.published``
_EPS = 1e-9
# MuJoCo resets MjData to qpos0 (time ~0) on these and only bumps d.warning[...].number
_BAD_STATE_WARNINGS = (
    ("BADQPOS", int(mujoco.mjtWarning.mjWARN_BADQPOS)),
    ("BADQVEL", int(mujoco.mjtWarning.mjWARN_BADQVEL)),
    ("BADQACC", int(mujoco.mjtWarning.mjWARN_BADQACC)),
)


class TimeCap(Exception):
    """``wait`` / ``run`` stopped: sim-time cap reached or ``driver.stop(reason)`` was called."""


class Diverged(TimeCap):
    """Physics blew up (non-finite / huge qpos, qvel or qacc): MuJoCo reset MjData behind the driver's back."""


# ---------------------------------------------------------------------------------------------- rendering
@dataclass
class View:
    """A free camera: look at ``lookat`` from ``distance`` along azimuth / elevation (MuJoCo mjvCamera), or,
    with ``camera`` set (``View.of_camera``), a camera of the model such as ``crown_cam``, which follows the
    mast sections and the gimbal that WP24 writes in the substep (the free-camera fields are then unused)."""
    lookat: Tuple[float, float, float]
    distance: float
    azimuth_deg: float
    elevation_deg: float
    name: str = "free"
    camera: Optional[str] = None

    @classmethod
    def of_camera(cls, camera: str) -> "View":
        """The view of the model camera ``camera`` (``mjCAMERA_FIXED``; its pose comes from ``MjData``)."""
        if not camera:
            raise ValueError("camera name required")
        return cls((0.0, 0.0, 0.0), 1.0, 0.0, 0.0, name=camera, camera=camera)

    def camera_id(self, m) -> int:
        """Model camera id of ``camera`` (``ValueError`` if the model has no such camera); -1 for a free view."""
        if self.camera is None:
            return -1
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
        if cid < 0:
            raise ValueError(f"model has no camera {self.camera!r}")
        return int(cid)

    @classmethod
    def look(cls, eye: Sequence[float], target: Sequence[float], name: str = "free") -> "View":
        """The view from point ``eye`` at point ``target`` (world frame)."""
        e, t = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
        v = t - e  # viewing direction
        dist = float(np.linalg.norm(v))
        if dist < 1e-6:
            raise ValueError("eye and target coincide")
        az = math.degrees(math.atan2(v[1], v[0]))
        el = math.degrees(math.asin(max(-1.0, min(1.0, v[2] / dist))))
        return cls(tuple(float(x) for x in t), dist, az, el, name)

    def eye(self) -> np.ndarray:
        """Camera position implied by a free view (matches mjvCamera's free-camera convention). A model-camera
        view has no fixed eye: use ``Capture.eye`` (the camera position when it was rendered)."""
        if self.camera is not None:
            raise ValueError(f"view of model camera {self.camera!r}: its eye is in MjData (Capture.eye)")
        az, el = math.radians(self.azimuth_deg), math.radians(self.elevation_deg)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return np.asarray(self.lookat) - self.distance * fwd

    def mjv_camera(self, m=None) -> "mujoco.MjvCamera":
        cam = mujoco.MjvCamera()
        if self.camera is not None:
            if m is None:
                raise ValueError("a model-camera view needs the model")
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = self.camera_id(m)
            return cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.lookat
        cam.distance = self.distance
        cam.azimuth = self.azimuth_deg
        cam.elevation = self.elevation_deg
        return cam


@dataclass
class Capture:
    t: float
    step: int
    view: View
    renderer: str
    rgb: np.ndarray
    eye: Optional[np.ndarray] = None    # world camera position at capture (free view: View.eye(); model: cam_xpos)


class StubRenderer:
    """No GL: records every capture and returns a blank RGB frame of the requested size (unit tier)."""
    name = "stub"

    def __init__(self, width: int = 64, height: int = 48):
        self.width, self.height = int(width), int(height)

    def render(self, m, d, view: View) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
        pass


class MjRenderer:
    """Offscreen ``mujoco.Renderer`` on the driver's own model (needs a GL backend: MUJOCO_GL / DISPLAY)."""
    name = "mujoco"

    def __init__(self, model, width: int = 160, height: int = 120):
        self.width, self.height = int(width), int(height)
        self._r = mujoco.Renderer(model, self.height, self.width)

    def render(self, m, d, view: View) -> np.ndarray:
        from atharv_nandi.sim.world import lighting      # orchard fog (WP4): no-op in models without a ``sun`` light
        lighting.sync_renderer(self._r)
        self._r.update_scene(d, camera=view.mjv_camera(m))
        return self._r.render().copy()

    def close(self):
        if self._r is not None:
            self._r.close()
            self._r = None


# ---------------------------------------------------------------------------------------------- tasks
@dataclass
class _Task:
    name: str
    fn: Callable
    rate_hz: float
    calls: int = 0
    _next_t: float = field(default=0.0, repr=False)


class SimDriver:
    """One world, one MjData, no ROS. See the module docstring for the per-step order."""

    def __init__(self, model=None, *, hooks: Optional["hook_registry.HookRegistry"] = None, load_hooks: bool = True,
                 bus: Optional["hook_registry.Bus"] = None, renderer=None, gait_dt: float = GAIT_DT,
                 max_sim_s: float = math.inf):
        if model is None:
            from atharv_nandi.core.farm_scene import load_farm_model
            model = load_farm_model()
        self.m = model
        self.d = mujoco.MjData(model)
        if hooks is None:
            hooks = hook_registry.load_all() if load_hooks else hook_registry.HookRegistry()
        self.hooks = hooks
        self.bus = bus if bus is not None else hook_registry.Bus()
        self.renderer = renderer
        self.gait_dt = float(gait_dt)
        self.dt = float(model.opt.timestep)
        self.gait_every = max(1, int(round(self.gait_dt / self.dt)))
        self.max_sim_s = float(max_sim_s)
        self.rate_capped = self.hooks.rate_capped(self.dt)  # hooks declared above 1/dt: run once per step

        n = mlay.ROBOT_NU
        self.home_ctrl = np.array(model.key_ctrl[0][:n], dtype=np.float64) if model.nkey else np.zeros(n)
        trn = model.actuator_trnid[:n, 0]
        self._robot_qadr = np.array([model.jnt_qposadr[j] for j in trn], dtype=np.int64)
        self._ctrl_lo = model.actuator_ctrlrange[:n, 0].copy()
        self._ctrl_hi = model.actuator_ctrlrange[:n, 1].copy()
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(n)]
        self._finger_idx = np.array([i for i in range(n) if names[i].startswith("finger")], dtype=np.int64)
        est = load_safety_config()["estop"]
        self.rearm_cfg = (float(est.get("rearm_rate", 0.25)), float(est.get("rearm_min_s", 0.5)),
                          float(est.get("rearm_max_s", 3.0)))
        self._ctrl_span = self._ctrl_hi - self._ctrl_lo
        self.rearm: Optional[Tuple[np.ndarray, float, float]] = None  # (from_ctrl, t0_sim, duration_s)
        self.diverged: Optional[str] = None  # e.g. "BADQVEL at t=0.012" once physics blew up
        self._warn_seen = [0] * len(_BAD_STATE_WARNINGS)

        self._substeps: List[Tuple[str, Callable]] = []
        self._tasks: Dict[str, _Task] = {}
        self._observers: List[Callable] = []
        self.on_publish: Optional[Callable[[str, str], None]] = None
        self.on_phase: Optional[Callable[[str], None]] = None
        self.published: Deque[Tuple[float, str, str]] = deque(maxlen=PUBLISHED_KEEP)
        self.published_count: Counter = Counter()
        self.hook_faults: Dict[str, str] = {}
        self.captures: List[Capture] = []
        self._ik = None
        self.wall_in_step = 0.0
        self.substep_calls = 0
        self.hold_ctrl: Optional[np.ndarray] = None
        self.bus.values["hold"] = None
        self._reset_state()
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, self.d, 0)
        mujoco.mj_forward(model, self.d)

    # ------------------------------------------------------------------ state
    def _reset_state(self):
        self.gait = TripodGaitGenerator(self.gait_dt)
        self.inhibit = MotionInhibitGate()
        self._cmd_raw = self._cmd_sent = (0.0, 0.0, 0.0)
        self.arm: List[float] = list(self.home_ctrl[mlay.ARM_CTRL])
        self.legs: List[float] = list(self.home_ctrl[mlay.LEG_CTRL])
        self.steps = 0
        self.rearm = None  # teleported home with home targets: nothing to blend
        self.diverged = None
        self._warn_seen = [int(self.d.warning[i].number) for _, i in _BAD_STATE_WARNINGS]
        self.stop_reason: Optional[str] = None
        self.phase_name = "idle"
        for t in self._tasks.values():
            t.calls, t._next_t = 0, 0.0

    def reset(self):
        """World reset in ``sim_runner`` order: keyframe 0, hooks restore their model fields, then mj_forward.
        A reset never re-arms: while holding (e-stop, hook fault, ...) the robot stays held at the home pose
        with the same ``bus.values["hold"]``, as ``sim_runner._apply_reset``. Faulted hooks stay disabled."""
        if self.m.nkey:
            mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        else:
            mujoco.mj_resetData(self.m, self.d)
        self.hooks.reset(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self._reset_state()
        if self.hold_ctrl is not None:
            self.hold_ctrl = self.home_ctrl[:mlay.ROBOT_NU].copy()  # still held: hold the home pose
        self._drain_bus()
        self._drain_faults()  # a reset_fn that raised

    @property
    def t(self) -> float:
        return float(self.d.time)

    @property
    def ik(self):
        """The closed-form arm IK on this model (built on first use)."""
        if self._ik is None:
            from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver
            self._ik = DorsalArmIKSolver(self.m)
        return self._ik

    def trunk_xy(self) -> np.ndarray:
        return self.d.qpos[0:2].copy()

    def heading(self) -> float:
        """Trunk yaw (rad) from the free-joint quaternion."""
        w, x, y, z = self.d.qpos[3:7]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def roll_pitch(self) -> Tuple[float, float]:
        return quat_to_roll_pitch(*self.d.qpos[3:7])

    def sim_state(self, world_id: Optional[str] = None) -> dict:
        """``/nandi/sim_state`` payload of this world (qpos, mocap, eq_active, provider mutations)."""
        from atharv_nandi.core import sim_state
        if world_id is None:
            lay = mlay.current()
            world_id = lay.world_id if lay is not None else ""
        return sim_state.pack(self.m, self.d, world_id)

    # ------------------------------------------------------------------ plug-ins
    def add_substep(self, fn: Callable, name: Optional[str] = None) -> str:
        """``fn(m, d, t, bus)`` before EVERY mj_step, after the ``every_step`` hooks. Returns its name."""
        name = name or getattr(fn, "__name__", "substep")
        if any(n == name for n, _ in self._substeps):
            raise ValueError(f"substep {name!r} added twice")
        self._substeps.append((name, fn))
        return name

    def remove_substep(self, name: str):
        self._substeps = [(n, f) for n, f in self._substeps if n != name]

    def add_task(self, name: str, fn: Callable, rate_hz: float = 50.0) -> _Task:
        """A task state machine: ``fn(driver)`` after mj_step on the sim-time grid at ``rate_hz`` (capped at
        once per step). Exceptions propagate (a task bug fails its test; hooks are the isolated layer)."""
        if name in self._tasks:
            raise ValueError(f"task {name!r} added twice")
        if not (rate_hz > 0 and math.isfinite(rate_hz)):
            raise ValueError(f"task {name!r}: rate_hz must be > 0")
        task = _Task(name, fn, float(rate_hz))
        self._tasks[name] = task
        return task

    def remove_task(self, name: str):
        self._tasks.pop(name, None)

    def add_observer(self, fn: Callable):
        """``fn(driver)`` after every step (logging, KPIs). It may raise ``TimeCap`` to end the run."""
        self._observers.append(fn)

    def deliver(self, topic: str, payload: Any):
        """A message on a hook input topic (``topics_in``), as sim_runner hands it over from ROS."""
        self.bus.deliver(topic, payload)

    # ------------------------------------------------------------------ safe hold
    @property
    def holding(self) -> bool:
        return self.hold_ctrl is not None

    def enter_hold(self, reason: str = "estop"):
        """Safe hold: robot actuators frozen at the current joint positions (fingers keep their ctrl), gait
        stopped, ``bus.values["hold"] = {reason, since}`` for the hooks. A second call keeps the first reason."""
        if self.hold_ctrl is not None:
            return
        n = mlay.ROBOT_NU
        hold = np.clip(self.d.qpos[self._robot_qadr], self._ctrl_lo, self._ctrl_hi)
        hold[self._finger_idx] = self.d.ctrl[self._finger_idx]
        self.hold_ctrl = hold[:n]
        self.rearm = None
        self.gait.stop()
        self.bus.values["hold"] = {"reason": reason or "estop", "since": self.t}

    def leave_hold(self):
        """Re-arm as ``sim_runner``: gait commands given during the hold are dropped (the gait stays stopped
        until the next ``walk`` / ``set_command``), and the robot actuators blend from the hold pose to the
        arm / leg targets over ``rearm_duration`` (smoothstep, sim time). Faulted hooks stay disabled."""
        if self.hold_ctrl is None:
            return
        self.gait.stop()
        start = self.hold_ctrl[:mlay.ROBOT_NU].copy()
        dur = rearm_duration(self._targets() - start, self._ctrl_span, *self.rearm_cfg)
        self.rearm = (start, float(self.d.time), dur)
        self.hold_ctrl = None
        self.bus.values["hold"] = None

    # ------------------------------------------------------------------ stepping
    def _targets(self) -> np.ndarray:
        return np.concatenate([np.asarray(self.arm, dtype=np.float64), np.asarray(self.legs, dtype=np.float64)])

    def _robot_ctrl(self):
        d = self.d
        if self.hold_ctrl is not None:
            d.ctrl[:mlay.ROBOT_NU] = self.hold_ctrl
        elif self.rearm is not None:
            start, t0, dur = self.rearm
            a = smoothstep((float(d.time) - t0) / dur)
            if a >= 1.0:
                self.rearm = None
                d.ctrl[:7] = self.arm
                d.ctrl[7:mlay.ROBOT_NU] = self.legs
            else:
                d.ctrl[:mlay.ROBOT_NU] = start + a * (self._targets() - start)
        else:
            d.ctrl[:7] = self.arm
            d.ctrl[7:mlay.ROBOT_NU] = self.legs

    def _apply_inhibit(self):
        """WP6 ``bus.values["motion_inhibit"]`` on the gait, as ``locomotion_service`` does with
        ``/nandi/motion_inhibit``: a stance / walk / stop inhibit stops the gait (and keeps stopping a new command
        while set); the warning-field ``speed_scale`` scales the command (the unscaled command is kept, so the
        scale does not compound and lifts when the field clears). Without the WP6 hook nothing changes."""
        mi = self.bus.values.get("motion_inhibit")
        if mi is None:
            return
        g = self.gait
        self.inhibit.update(mi)
        if self.inhibit.halt:
            if g.is_active:
                g.stop()
            self._cmd_raw = self._cmd_sent = tuple(g.cmd)
            return
        cmd = tuple(g.cmd)
        if cmd != self._cmd_sent:                       # a new command since the last tick
            self._cmd_raw = cmd
        want = tuple(c * self.inhibit.scale for c in self._cmd_raw)
        if want != cmd:
            g.cmd = want
        self._cmd_sent = want

    def step(self):
        """One physics step (module docstring order)."""
        if self.diverged is not None:
            raise Diverged(f"diverged: {self.diverged}")
        m, d = self.m, self.d
        if self.hold_ctrl is None and self.steps % self.gait_every == 0:
            self._apply_inhibit()
            roll, pitch = quat_to_roll_pitch(*d.qpos[3:7])
            self.legs = self.gait.step(roll, pitch)
        self._robot_ctrl()
        hooks = self.hooks
        has_hooks = len(hooks) > 0
        if has_hooks and hooks.has_substep:
            hooks.substep(m, d, self.bus)
        if self._substeps:
            t = float(d.time)
            for _, fn in self._substeps:
                fn(m, d, t, self.bus)
            self.substep_calls += 1
        if has_hooks or self._substeps:
            self._robot_ctrl()  # a hook / substep never drives the robot actuators
        t_before = float(d.time)
        t0 = time.perf_counter()
        mujoco.mj_step(m, d)
        self.wall_in_step += time.perf_counter() - t0
        self.steps += 1
        try:
            self._check_divergence(t_before)
            if has_hooks:
                hooks.tick(m, d, self.bus)
            if self._tasks:
                self._tick_tasks()
            for fn in self._observers:
                fn(self)
        finally:  # also when an observer / task raised (TimeCap, Diverged): nothing queued is lost
            if self.bus.outbox:
                self._drain_bus()
            if has_hooks:
                self._drain_faults()

    def _check_divergence(self, t_before: float):
        """A new bad-state warning means MuJoCo reset MjData to qpos0 during this mj_step: stop the run."""
        w = self.d.warning
        for k, (label, i) in enumerate(_BAD_STATE_WARNINGS):
            n = int(w[i].number)
            if n > self._warn_seen[k]:
                self._warn_seen[k] = n
                self.diverged = f"{label} at t={t_before:.4f}"
        if self.diverged is not None:
            raise Diverged(f"diverged: {self.diverged}")

    def _tick_tasks(self):
        t = float(self.d.time)
        for task in list(self._tasks.values()):
            if t + _EPS >= task._next_t:
                period = 1.0 / task.rate_hz
                task._next_t = (math.floor((t + _EPS) / period) + 1) * period
                task.calls += 1
                task.fn(self)

    def _drain_bus(self):
        for topic, text in self.bus.drain():
            self.published.append((self.t, topic, text))
            self.published_count[topic] += 1
            if self.on_publish is not None:
                self.on_publish(topic, text)

    def _drain_faults(self):
        faults = self.hooks.drain_faults()
        for name, err in faults:
            self.hook_faults[name] = err
        if faults:
            self.enter_hold(hook_registry.REASON_HOOK_FAULT)

    def stop(self, reason: str):
        """Ask the running ``wait`` to end with ``TimeCap`` once it has done its steps (first reason wins)."""
        if self.stop_reason is None:
            self.stop_reason = reason

    def run_steps(self, n: int):
        for _ in range(int(n)):
            self.step()

    def _at_cap(self) -> bool:
        return self.t >= self.max_sim_s - 0.5 * self.dt  # half a step: float sums of dt land a hair short

    def wait(self, seconds: float):
        """Step ``seconds`` of sim time (``seconds > 0``: at least one step; ``<= 0``: none). ``TimeCap`` exactly at
        the sim-time cap (the wait is cut at ``max_sim_s``, never overshoots it) or, after its steps, on ``stop``."""
        n = max(1, int(round(seconds / self.dt))) if seconds > 0 else 0
        for _ in range(n):
            if self._at_cap():
                break
            self.step()
        if self._at_cap():
            raise TimeCap("time_cap")
        if self.stop_reason is not None:
            raise TimeCap(self.stop_reason)

    def run_until(self, pred: Callable[["SimDriver"], bool], timeout_s: float, chunk_s: float = GAIT_DT) -> bool:
        """Step in ``chunk_s`` slices until ``pred(driver)``; False after ``timeout_s`` of sim time."""
        t_end = self.t + timeout_s
        while self.t < t_end - _EPS:
            self.wait(min(chunk_s, max(self.dt, t_end - self.t)))
            if pred(self):
                return True
        return bool(pred(self))

    # ------------------------------------------------------------------ HarvestSequence I/O
    def qpos(self) -> np.ndarray:
        return self.d.qpos.copy()

    def command(self, arm7: Sequence[float]):
        self.arm = list(arm7)

    def phase(self, name: str):
        self.phase_name = name
        if self.on_phase is not None:
            self.on_phase(name)

    # ------------------------------------------------------------------ behaviours
    def walk(self, vx: float, vy: float = 0.0, yaw: float = 0.0, seconds: Optional[float] = None):
        """Hold a gait command (normalised -1..1, as /nandi/cmd_vel); step ``seconds`` if given."""
        self.gait.set_command(vx, vy, yaw, hold=True)
        if seconds is not None:
            self.wait(seconds)

    def stand(self, seconds: float):
        """Finish the gait cycle and stand for ``seconds``."""
        self.gait.release()
        self.wait(seconds)

    def walk_distance(self, dist_m: float, vx: float = 1.0, timeout_s: float = 30.0, settle_s: float = 0.6) -> dict:
        """Walk straight (heading held by a yaw P-term) until the trunk moved ``dist_m`` horizontally, then stand.
        Returns ``{ok, dist_m, sim_s, heading_change_deg}``."""
        p0, h0, t0 = self.trunk_xy(), self.heading(), self.t

        def moved(_):
            return float(np.linalg.norm(self.trunk_xy() - p0)) >= dist_m

        def steer(_):
            # +yaw turns the trunk CCW whatever the travel direction (tripod_gait.leg_strides): no sign(vx)
            err = _wrap(h0 - self.heading())
            self.gait.set_command(vx, 0.0, max(-1.0, min(1.0, 2.0 * err)), hold=True)
            return moved(_)

        self.walk(vx)
        ok = self.run_until(steer, timeout_s)
        sim_s = self.t - t0
        self.stand(settle_s)
        return {"ok": ok, "dist_m": float(np.linalg.norm(self.trunk_xy() - p0)), "sim_s": sim_s,
                "heading_change_deg": math.degrees(_wrap(self.heading() - h0))}

    def face(self, target_xy: Sequence[float], tol_deg: float = 8.0, timeout_s: float = 20.0) -> bool:
        """Turn in place until the trunk heads at ``target_xy`` within ``tol_deg``, then stand."""
        tx, ty = float(target_xy[0]), float(target_xy[1])

        def turn(_):
            x, y = self.trunk_xy()
            err = _wrap(math.atan2(ty - y, tx - x) - self.heading())
            if abs(err) <= math.radians(tol_deg):
                return True
            self.gait.set_command(0.0, 0.0, 1.0 if err > 0 else -1.0, hold=True)
            return False

        ok = self.run_until(turn, timeout_s)
        self.gait.release()
        return ok

    def go_to(self, target_xy: Sequence[float], tol_m: float = 0.15, timeout_s: float = 60.0) -> bool:
        """Turn toward ``target_xy`` and walk until within ``tol_m`` (same steering as headless_eval)."""
        tx, ty = float(target_xy[0]), float(target_xy[1])

        def drive(_):
            x, y = self.trunk_xy()
            if math.hypot(tx - x, ty - y) <= tol_m:
                return True
            ang = _wrap(math.atan2(ty - y, tx - x) - self.heading())
            if abs(ang) > 0.35:
                self.gait.set_command(0.0, 0.0, 1.0 if ang > 0 else -1.0, hold=True)
            else:
                self.gait.set_command(1.0, 0.0, max(-1.0, min(1.0, 1.5 * ang)), hold=True)
            return False

        ok = self.run_until(drive, timeout_s)
        self.gait.release()
        return ok

    def capture(self, view: View) -> Capture:
        """Render ``view`` from the current state with the driver's renderer (a ``StubRenderer`` if none)."""
        cid = view.camera_id(self.m)       # an unknown model camera fails here, also with the stub renderer
        eye = self.d.cam_xpos[cid].copy() if cid >= 0 else view.eye()
        if self.renderer is None:
            self.renderer = StubRenderer()
        rgb = self.renderer.render(self.m, self.d, view)
        cap = Capture(self.t, self.steps, view, self.renderer.name, rgb, eye)
        self.captures.append(cap)
        return cap

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    # ------------------------------------------------------------------ report
    def hook_calls(self) -> Dict[str, int]:
        return {h.name: h.calls for h in self.hooks.hooks()}

    def task_calls(self) -> Dict[str, int]:
        return {t.name: t.calls for t in self._tasks.values()}


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
