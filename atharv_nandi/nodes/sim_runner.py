#!/usr/bin/env python3
"""Atharv Nandi physics + MuJoCo passive viewer, paced to wall-clock.

Threads:
    rclpy spin thread  - subscriptions + services; only touches the small shared state below,
                         always under self._lock.
    sim loop (main)    - owns MjModel/MjData exclusively: steps, publishes, renders, resets.

Safety (README, "Safety"): while /nandi/safety_status says estopped, the loop ignores
/nandi/arm_targets and /nandi/leg_targets and holds every actuator at the joint angles of the
moment of engage ("safe_hold"; zeroing torque would collapse the hexapod), with kp / kv scaled and
a joint brake (estop.hold_* in config/safety.yaml) so it settles fast. The fingers keep their last
commanded ctrl so a held crop keeps its grip force. It measures the stop (time + trunk travel until
trunk speed < 0.02 m/s) and reports mode + metrics on /nandi/heartbeat for the supervisor. If the
supervisor was seen and then goes silent for SUPERVISOR_TIMEOUT_S, the loop also safe-holds
(fail-safe). A hold is left only for a re-arm (safety_status.rearmed_at) stamped after the hold
started, never just because a status says estopped=false; leaving it blends from the hold pose to
the live targets instead of snapping.

NANDI_HEADLESS=1 runs without the MuJoCo viewer (tests / CI smoke runs).

WP0 plug-in points (this file is frozen after WP0; work packages extend it only through them):
    hooks       every atharv_nandi/sim/hooks/<name>.py is loaded (hooks.load_all): ``every_step`` hooks run before each mj_step
                (substep point), rate hooks after it at their declared sim-time rate (default 50 Hz). Their
                bus.publish(topic, payload) goes out as std_msgs/String JSON; topics they declare in
                ``topics_in`` are subscribed (String JSON) and delivered to bus.latest(topic).
    sim_state   /nandi/sim_state (String JSON, atharv_nandi/core/sim_state.py) every 1/30 s of sim time: world_id,
                qpos, mocap, eq_active and every registered provider's model mutations.
    parameters  world_id, layout (model_layout.Layout JSON) and hooks (the hook table), declared once.
/nandi/sim_qpos keeps its shape (the full qpos; byte-identical to pre-WP0 in the legacy world). Only the
robot actuators 0..24 are written from arm/leg targets (and re-written before every mj_step, so a hook
cannot drive them); appended actuators (>= 25) belong to their hooks. Safe hold holds and brakes the robot
actuators only and tells the hooks through bus.values["hold"]; a hook that raises is disabled and puts the
runner into safe hold (reason hook_fault), it never ends the node. /nandi/joint_states lists every actuator
with a joint transmission (the 25 robot joints, then appended module joints); site / body / tendon
actuators are left out of it and out of the hold.
"""
import contextlib
import json
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi import node_params  # noqa: E402
from atharv_nandi.core import model_layout as mlay
from atharv_nandi.core import sim_state
from atharv_nandi.core.farm_scene import crop_body_names, crop_qpos_slice, load_farm_model
from atharv_nandi.core.game_status import FallCounter, RtfMeter, build_game_status, is_fallen, plan_catch_up
from atharv_nandi.core.safety_logic import (
    HOLD_MODE_NONE, HOLD_MODE_SAFE, REASON_SUPERVISOR_LOST, StopMetrics, heartbeat_json, hold_should_release,
    load_safety_config, parse_safety_status, rearm_duration, smoothstep,
)

QPOS_PUB_PERIOD_S = 0.02     # /nandi/sim_qpos at 50 Hz of sim time
VIEWER_PERIOD_S = 1.0 / 30   # viewer.sync at most ~30 Hz wall
STATUS_PERIOD_S = 0.5        # /nandi/game_status at 2 Hz wall
MAX_CATCHUP_S = 0.05         # at most 50 ms of sim per loop pass; older debt is dropped
RESET_TIMEOUT_S = 2.0
# After a reset, ignore /nandi/arm_targets this long: the harvest node keeps publishing its
# pre-reset (slewed) arm command until it sees the reset in /nandi/sim_qpos and snaps to rest.
ARM_RESET_HOLD_S = 0.3
JOINT_STATES_EVERY = 1        # /nandi/joint_states on every sim_qpos publish (50 Hz sim time)
HEARTBEAT_PERIOD_S = 0.5      # /nandi/heartbeat at 2 Hz wall, plus on every hold-mode change / stop
SUPERVISOR_TIMEOUT_S = 1.0    # safety_status silent this long after it was seen => fail-safe hold


class _NoViewer:
    """Stand-in for the passive viewer when NANDI_HEADLESS=1."""

    user_scn = None

    def is_running(self):
        return True

    def sync(self):
        pass


class NandiSimRunner(Node):
    """MuJoCo physics & 3D renderer subscribing to modular service outputs."""

    def __init__(self):
        super().__init__("nandi_sim_runner")
        node_params.declare(self, "nandi_sim_runner")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        self.model = load_farm_model()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.layout = mlay.current()  # set by load_farm_model (frozen robot table checked there)
        self.world_id = self.layout.world_id
        self.crop_names = crop_body_names(self.model)
        self.trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.home_ctrl = np.array(self.model.key_ctrl[0], dtype=np.float64)
        # Actuator -> joint maps from the layout (actuator_joint is "" for site / body / tendon transmissions).
        # Robot actuators 0..24 (frozen) all drive joints; ctrl order == qpos[7:32] order, but derive it.
        lay = self.layout
        self.robot_qadr, self.robot_dadr = self._joint_adrs(lay, range(mlay.ROBOT_NU))
        # /nandi/joint_states: every actuator with a joint transmission, in actuator order.
        self.js_act = np.array([i for i in range(self.model.nu) if lay.actuator_joint[i]], dtype=np.int64)
        self.joint_names = [lay.actuator_joint[i] for i in self.js_act]
        self.act_qadr, self.act_dadr = self._joint_adrs(lay, self.js_act)
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.base_biasprm = self.model.actuator_biasprm.copy()
        self.base_gainprm = self.model.actuator_gainprm.copy()
        self.base_frictionloss = self.model.dof_frictionloss.copy()
        act_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        # Fingers keep their commanded ctrl in safe hold (grip force); the other robot actuators are held +
        # braked. Appended actuators (>= 25) are their hooks' (bus.values["hold"]): never held or re-gained here.
        robot = range(mlay.ROBOT_NU)
        self.finger_idx = np.array([i for i in robot if act_names[i].startswith("finger")], dtype=np.int64)
        self.hold_idx = np.array([i for i in robot if i not in set(self.finger_idx)], dtype=np.int64)
        est = load_safety_config()["estop"]
        self.hold_kp_scale = max(1.0, float(est.get("hold_stiffness_scale", 1.0)))
        self.hold_kv_scale = max(1.0, float(est.get("hold_damping_scale", 1.0)))
        self.hold_brake = max(0.0, float(est.get("hold_brake_nm", 0.0)))
        self.rearm_cfg = (float(est.get("rearm_rate", 0.25)), float(est.get("rearm_min_s", 0.5)),
                          float(est.get("rearm_max_s", 3.0)))
        self.ctrl_span = self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        self.headless = os.environ.get("NANDI_HEADLESS", "0") not in ("", "0", "false", "no")

        # Shared with the spin thread (guarded by _lock).
        self._lock = threading.Lock()
        self._arm_targets = self.home_ctrl[mlay.ARM_CTRL].copy()
        self._arm_hold_until = 0.0        # time.monotonic() before which arm_targets are ignored
        self._leg_targets = self.home_ctrl[mlay.LEG_CTRL].copy()
        self._hook_inbox = {}             # topic -> newest JSON payload for the hooks (topics_in)
        self._wireframe = None            # (canopy_segs, floor_segs) or None
        self._reset_requests = []         # threading.Event per pending /reset_world call
        self._estopped = False            # from /nandi/safety_status
        self._estop_reason = None
        self._supervisor_seen = None      # time.monotonic() of the last safety_status, None = never
        self._engaged_at = None           # safety_status.engaged_at / rearmed_at (wall clock)
        self._rearmed_at = None

        # Owned by the sim loop: safe hold.
        self.hold_ctrl = None             # np.ndarray(ROBOT_NU) while holding, else None
        self.hold_reason = None
        self.hold_since = None            # wall time the hold counts from (latch engage or loss)
        self.rearm = None                 # (from_ctrl, t0_sim, duration_s) while blending out of a hold
        self.stop_metrics = StopMetrics()
        self._hb_dirty = False

        # Owned by the sim loop.
        self.falls = FallCounter()
        self.rtf = RtfMeter()

        self.create_subscription(Float64MultiArray, "/nandi/arm_targets", self.on_arm, 10)
        self.create_subscription(Float64MultiArray, "/nandi/leg_targets", self.on_legs, 10)
        self.create_subscription(String, "/nandi/workspace_wireframe", self.on_wireframe, 10)
        self.create_subscription(String, "/nandi/safety_status", self.on_safety_status, 10)
        self.qpos_pub = self.create_publisher(Float64MultiArray, "/nandi/sim_qpos", 10)
        self.js_pub = self.create_publisher(JointState, "/nandi/joint_states", 10)
        self.hb_pub = self.create_publisher(String, "/nandi/heartbeat", 10)
        self.status_pub = self.create_publisher(String, "/nandi/game_status", 10)
        # Reset clears the envelope for every viewer (Tk panel, browsers), not just this one.
        self.wf_pub = self.create_publisher(String, "/nandi/workspace_wireframe", 10)
        self.create_service(Trigger, "/nandi/reset_world", self.on_reset_world)
        # WP0: sim_state, layout + world_id parameters, hook registry and its bus.
        self.state_pub = self.create_publisher(String, sim_state.SIM_STATE_TOPIC, 10)
        self.hooks = hook_registry.load_all()
        sim_state.discover_providers()  # the same provider set every renderer loads
        self.bus = hook_registry.Bus()
        self.hook_faults = {}             # name -> error of every hook disabled by an exception
        self._state_err = None            # last /nandi/sim_state publish error (logged once per distinct one)
        self._bus_pubs = {}
        for topic in self.hooks.topics_in():
            self.create_subscription(String, topic, lambda msg, t=topic: self.on_hook_topic(t, msg), 10)
        # Read-only: the world_id stamped in /nandi/sim_state is fixed when the world is composed.
        for pname, value, desc in (
                ("world_id", self.world_id, "world_id stamped in /nandi/sim_state"),
                ("layout", self.layout.to_json(), "model_layout.Layout JSON of the composed world"),
                ("hooks", json.dumps(self.hooks.table()), "hook table (name, rate_hz, every_step, ...)")):
            self.declare_parameter(pname, value, ParameterDescriptor(description=desc, read_only=True))
        for name in self.hooks.rate_capped(self.model.opt.timestep):
            self.get_logger().warning(
                f"hook {name!r} is declared faster than the physics rate {1.0 / self.model.opt.timestep:.0f} Hz:"
                " it runs once per step")
        # After a reset, ask the controllers to stand down so the robot stays at home.
        self.stop_motion_cli = self.create_client(Trigger, "/nandi/stop_motion")
        self.release_arm_cli = self.create_client(Trigger, "/nandi/release_arm")
        self.get_logger().info(
            f"World: nq={self.model.nq}, {len(self.crop_names)} crops, dt={self.model.opt.timestep}"
            f", world {self.layout.world} id {self.world_id}, {len(self.hooks)} hooks"
        )

    @staticmethod
    def _joint_adrs(lay, idx):
        """(qpos adr, dof adr) of the joint each actuator in ``idx`` drives; every one must be a joint actuator."""
        names = [lay.actuator_joint[i] for i in idx]
        bad = [lay.actuators[i] for i, n in zip(idx, names) if not n]
        if bad:
            raise mlay.LayoutError(f"actuators without a joint transmission: {bad}")
        return (np.array([lay.qpos_adr(n) for n in names], dtype=np.int64),
                np.array([lay.dof_adr(n) for n in names], dtype=np.int64))

    # ---------------- spin-thread callbacks ----------------
    def on_arm(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            with self._lock:
                if time.monotonic() < self._arm_hold_until:
                    return  # stale pre-reset command
                self._arm_targets = np.array(msg.data, dtype=np.float64)

    def on_legs(self, msg: Float64MultiArray):
        if len(msg.data) == 18:
            with self._lock:
                self._leg_targets = np.array(msg.data, dtype=np.float64)

    def on_hook_topic(self, topic: str, msg: String):
        try:
            payload = json.loads(msg.data) if msg.data else None
        except ValueError:
            self.get_logger().warn(f"Ignoring non-JSON message on {topic} (hook input)")
            return
        with self._lock:
            self._hook_inbox[topic] = payload

    def on_safety_status(self, msg: String):
        st = parse_safety_status(msg.data)
        if st is None:
            return
        with self._lock:
            self._estopped = st["estopped"]
            self._estop_reason = st["latched_reason"]
            self._engaged_at = st["engaged_at"]
            self._rearmed_at = st["rearmed_at"]
            self._supervisor_seen = time.monotonic()

    def on_wireframe(self, msg: String):
        wf = None
        if msg.data:
            try:
                data = json.loads(msg.data)
                wf = (
                    np.array(data["canopy_segments"], dtype=np.float64).reshape(-1, 2, 3),
                    np.array(data["floor_segments"], dtype=np.float64).reshape(-1, 2, 3),
                )
            except (ValueError, KeyError, TypeError) as exc:
                self.get_logger().warn(f"Ignoring malformed workspace_wireframe: {exc}")
                return
        with self._lock:
            self._wireframe = wf

    def on_reset_world(self, request, response):
        done = threading.Event()
        with self._lock:
            self._reset_requests.append(done)
        if not done.wait(RESET_TIMEOUT_S):
            with self._lock:
                if done in self._reset_requests:
                    self._reset_requests.remove(done)
            response.success = False
            response.message = "Sim loop not responding - world not reset"
            return response
        self.wf_pub.publish(String(data=""))
        for cli in (self.stop_motion_cli, self.release_arm_cli):
            if cli.service_is_ready():
                cli.call_async(Trigger.Request())  # fire-and-forget; never block the executor
        response.success = True
        response.message = f"World reset: robot + {len(self.crop_names)} crops at home, falls/timer cleared"
        return response

    # ---------------- sim-loop helpers ----------------
    def _apply_reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Hooks restore the model fields they mutated (mast body_pos, ...) BEFORE kinematics run on them.
        self.hooks.reset(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        with self._lock:
            self._arm_targets = self.home_ctrl[mlay.ARM_CTRL].copy()
            self._arm_hold_until = time.monotonic() + ARM_RESET_HOLD_S
            self._leg_targets = self.home_ctrl[mlay.LEG_CTRL].copy()
            self._wireframe = None
        if self.hold_ctrl is not None:
            self.hold_ctrl = self.home_ctrl[:mlay.ROBOT_NU].copy()  # still e-stopped: hold the home pose
            self.stop_metrics.cancel()
        self.rearm = None  # teleported home with home targets: nothing to blend
        self.falls.reset()
        self.rtf.rebase()
        self._hook_faults()  # a reset_fn that raised

    def _game_status(self) -> dict:
        qpos = self.data.qpos
        crops = {name: qpos[crop_qpos_slice(k)][:3] for k, name in enumerate(self.crop_names, start=1)}
        return build_game_status(
            crops, qpos[0:3], qpos[3:7], self.falls.falls, self.falls.fallen,
            self.data.time, self.rtf.rtf,
        )

    def _publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self._game_status())))

    def _publish_qpos(self):
        msg = Float64MultiArray()
        msg.data = self.data.qpos.tolist()
        self.qpos_pub.publish(msg)

    def _publish_sim_state(self):
        # A provider's pack() bug or a non-finite state skips this message; it never ends the node.
        try:
            text = sim_state.encode(sim_state.pack(self.model, self.data, self.world_id))
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            if err != self._state_err:
                self.get_logger().error(f"/nandi/sim_state not published: {err}")
            self._state_err = err
            return
        self._state_err = None
        self.state_pub.publish(String(data=text))

    def _hooks_in(self):
        """Hand the newest hook-input messages to the bus (sim loop only)."""
        with self._lock:
            inbox, self._hook_inbox = self._hook_inbox, {}
        for topic, payload in inbox.items():
            self.bus.deliver(topic, payload)

    def _hooks_out(self):
        """Publish what the hooks queued on the bus (String JSON, one lazily created publisher per topic)."""
        for topic, payload in self.bus.drain():
            try:
                text = hook_registry.encode_payload(payload)  # bus.publish already encoded it
            except ValueError as exc:
                self.get_logger().error(f"dropped hook message on {topic}: {exc}")
                continue
            pub = self._bus_pubs.get(topic)
            if pub is None:
                pub = self._bus_pubs[topic] = self.create_publisher(String, topic, 10)
            pub.publish(String(data=text))

    def _hook_faults(self):
        """Log + report every hook that faulted; enter safe hold (reason hook_fault) if not holding already."""
        faults = self.hooks.drain_faults()
        if not faults:
            return
        for name, err in faults:
            self.hook_faults[name] = err
            self.get_logger().error(f"hook {name!r} disabled after an exception: {err}")
        self._hb_dirty = True
        if self.hold_ctrl is None:
            self._enter_hold(hook_registry.REASON_HOOK_FAULT, time.time())
        self._sync_bus_hold()

    def _sync_bus_hold(self):
        """bus.values["hold"]: None, or what the runner is holding for (read by hooks)."""
        self.bus.values["hold"] = (None if self.hold_ctrl is None else
                                   {"reason": self.hold_reason, "since": self.hold_since})

    def _publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.data.qpos[self.act_qadr].tolist()
        msg.velocity = self.data.qvel[self.act_dadr].tolist()
        msg.effort = self.data.actuator_force[self.js_act].tolist()
        self.js_pub.publish(msg)

    def _publish_heartbeat(self):
        info = {
            "hold_mode": HOLD_MODE_SAFE if self.hold_ctrl is not None else HOLD_MODE_NONE,
            "hold_reason": self.hold_reason,
            "hold_since": self.hold_since,
            "rearming": self.rearm is not None,
            "last_stop": self.stop_metrics.last,
            "stop_measuring": self.stop_metrics.measuring,
            "rtf": round(float(self.rtf.rtf), 3),
            "hook_faults": dict(self.hook_faults),
        }
        self.hb_pub.publish(String(data=heartbeat_json("sim_runner", self.data.time, True, info)))
        self._hb_dirty = False

    def _update_hold(self):
        """Enter / leave safe hold from the latest safety_status (and the supervisor watchdog)."""
        with self._lock:
            estopped, reason, seen = self._estopped, self._estop_reason, self._supervisor_seen
            engaged_at, rearmed_at = self._engaged_at, self._rearmed_at
        lost = seen is not None and time.monotonic() - seen > SUPERVISOR_TIMEOUT_S
        if self.hold_ctrl is None:
            if estopped or lost:
                # The hold counts from the latch's engage stamp (or now, if we hold on our own): only
                # an estop_reset stamped after that releases it.
                since = engaged_at if (estopped and engaged_at is not None) else time.time()
                self._enter_hold(reason if estopped else REASON_SUPERVISOR_LOST, since)
        elif hold_should_release(estopped, lost, rearmed_at, self.hold_since):
            self._leave_hold()

    def _set_hold_physics(self, on: bool):
        m, i = self.model, self.hold_idx
        # Restore only the robot actuators 0..24 and their dofs: appended actuators / dofs are their
        # hooks' (a mast e-stop brake or a tool re-gain must survive entering or leaving the hold).
        n, rd = mlay.ROBOT_NU, self.robot_dadr
        m.actuator_gainprm[:n] = self.base_gainprm[:n]
        m.actuator_biasprm[:n] = self.base_biasprm[:n]
        m.dof_frictionloss[rd] = self.base_frictionloss[rd]
        if not on:
            return
        m.actuator_gainprm[i, 0] = self.base_gainprm[i, 0] * self.hold_kp_scale
        m.actuator_biasprm[i, 1] = self.base_biasprm[i, 1] * self.hold_kp_scale
        m.actuator_biasprm[i, 2] = self.base_biasprm[i, 2] * self.hold_kv_scale
        dofs = self.robot_dadr[i]
        brake = np.where(i < 7, 0.5 * self.hold_brake, self.hold_brake)  # arm joints: half
        m.dof_frictionloss[dofs] = self.base_frictionloss[dofs] + brake

    def _enter_hold(self, reason, since):
        n = mlay.ROBOT_NU  # appended actuators keep their ctrl: their hooks see bus.values["hold"]
        hold = np.clip(self.data.qpos[self.robot_qadr], self.ctrl_lo[:n], self.ctrl_hi[:n])
        # Fingers keep the ctrl being applied now (the commanded grip): holding them at their
        # current opening would zero the squeeze on a held crop.
        hold[self.finger_idx] = self.data.ctrl[self.finger_idx]
        self.hold_ctrl, self.hold_reason, self.hold_since = hold, reason or "estop", since
        self.rearm = None
        self._set_hold_physics(True)
        self.stop_metrics.start(self.data.time, self.data.qpos[0:2])
        self._hb_dirty = True
        self._sync_bus_hold()
        self.get_logger().warning(f"E-STOP ({self.hold_reason}): safe hold, arm/leg targets ignored")

    def _leave_hold(self):
        with self._lock:
            target = np.concatenate([self._arm_targets, self._leg_targets])
        start = self.hold_ctrl[:mlay.ROBOT_NU].copy()  # only the robot actuators blend back to targets
        dur = rearm_duration(target - start, self.ctrl_span[:mlay.ROBOT_NU], *self.rearm_cfg)
        self.rearm = (start, self.data.time, dur)
        self.hold_ctrl, self.hold_reason, self.hold_since = None, None, None
        self._set_hold_physics(False)
        self.stop_metrics.cancel()
        self._hb_dirty = True
        self._sync_bus_hold()
        self.get_logger().info(f"E-stop reset: blending to arm/leg targets over {dur:.2f} s")

    def _live_ctrl(self) -> np.ndarray:
        """Targets for the robot actuators 0..24 now: arm + legs, faded in from the hold pose after a re-arm."""
        with self._lock:
            target = np.concatenate([self._arm_targets, self._leg_targets])
        if self.rearm is None:
            return target
        start, t0, dur = self.rearm
        a = smoothstep((self.data.time - t0) / dur)
        if a >= 1.0:
            self.rearm = None
            self._hb_dirty = True
            return target
        return start + a * (target - start)

    def _render_wireframe(self, viewer, wireframe):
        scn = viewer.user_scn
        if scn is None:
            return
        scn.ngeom = 0
        if wireframe is None:
            return
        canopy, floor = wireframe

        t_pos = self.data.xpos[self.trunk_id]
        t_rot = self.data.xmat[self.trunk_id].reshape(3, 3)
        c1 = (canopy[:, 0, :] @ t_rot.T) + t_pos
        c2 = (canopy[:, 1, :] @ t_rot.T) + t_pos
        f1 = (floor[:, 0, :] @ t_rot.T) + t_pos
        f2 = (floor[:, 1, :] @ t_rot.T) + t_pos

        cyan = np.array([0.15, 0.85, 0.95, 0.26], dtype=np.float32)
        red = np.array([1.0, 0.22, 0.22, 0.55], dtype=np.float32)
        z3, eye = np.zeros(3), np.eye(3).flatten()
        idx = 0
        for p1, p2 in zip(c1, c2):
            if idx >= scn.maxgeom - 10:
                break
            mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE, z3, z3, eye, cyan)
            mujoco.mjv_connector(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025, p1, p2)
            idx += 1
        for p1, p2 in zip(f1, f2):
            if idx >= scn.maxgeom - 10:
                break
            mujoco.mjv_initGeom(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE, z3, z3, eye, red)
            mujoco.mjv_connector(scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0035, p1, p2)
            idx += 1
        scn.ngeom = idx

    # ---------------- sim loop ----------------
    def run_sim(self):
        dt = self.model.opt.timestep
        qpos_every = max(1, int(round(QPOS_PUB_PERIOD_S / dt)))
        state_every = max(1, int(round(sim_state.SIM_STATE_PERIOD_S / dt)))
        hooks, bus = self.hooks, self.bus
        has_hooks, has_substep = len(hooks) > 0, hooks.has_substep
        max_steps = max(1, int(round(MAX_CATCHUP_S / dt)))

        steps = 0                     # physics steps since the last reset (sim time = steps * dt)
        anchor = time.perf_counter()  # wall time at which `steps` was 0
        last_view = last_status = last_hb = -1e9

        viewer_cm = (contextlib.nullcontext(_NoViewer()) if self.headless
                     else mujoco.viewer.launch_passive(self.model, self.data))
        with viewer_cm as viewer:
            while viewer.is_running() and rclpy.ok():
                with self._lock:
                    pending, self._reset_requests = self._reset_requests, []
                if pending:
                    self._apply_reset()
                    steps, anchor = 0, time.perf_counter()
                    for ev in pending:
                        ev.set()
                    last_status = -1e9  # publish the fresh status right away

                now = time.perf_counter()
                n, dropped = plan_catch_up((now - anchor) - steps * dt, dt, max_steps)
                anchor += dropped  # forgive debt we cannot pay back

                self._update_hold()
                if n:
                    # Robot actuators 0..24 only: safe hold ignores targets; appended ones are the hooks'.
                    robot_ctrl = self.hold_ctrl if self.hold_ctrl is not None else self._live_ctrl()
                    self.data.ctrl[:mlay.ROBOT_NU] = robot_ctrl
                    if has_hooks:
                        self._hooks_in()
                    for _ in range(n):
                        if has_substep:
                            hooks.substep(self.model, self.data, bus)  # per-physics-step substep point
                        if has_hooks:
                            self.data.ctrl[:mlay.ROBOT_NU] = robot_ctrl  # a hook never drives the robot
                        mujoco.mj_step(self.model, self.data)
                        steps += 1
                        if has_hooks:
                            hooks.tick(self.model, self.data, bus)
                        if self.stop_metrics.measuring:
                            qv = self.data.qvel
                            speed = float(np.sqrt(qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2]))
                            if self.stop_metrics.update(self.data.time, self.data.qpos[0:2], speed):
                                self._hb_dirty = True
                                self.get_logger().info(f"Stopped after e-stop: {self.stop_metrics.last}")
                        if steps % qpos_every == 0:  # steps restarts at 0 on reset, like sim time
                            self._publish_qpos()
                            if (steps // qpos_every) % JOINT_STATES_EVERY == 0:
                                self._publish_joint_states()
                        if steps % state_every == 0:
                            self._publish_sim_state()
                    if has_hooks:
                        self._hooks_out()
                        self._hook_faults()
                    qpos = self.data.qpos
                    self.falls.update(is_fallen(qpos[0:3], qpos[3:7]))

                now = time.perf_counter()
                self.rtf.update(self.data.time, now)
                if now - last_view >= VIEWER_PERIOD_S:
                    with self._lock:
                        wireframe = self._wireframe
                    self._render_wireframe(viewer, wireframe)
                    viewer.sync()
                    last_view = now
                if now - last_status >= STATUS_PERIOD_S:
                    self._publish_status()
                    last_status = now
                if self._hb_dirty or now - last_hb >= HEARTBEAT_PERIOD_S:
                    self._publish_heartbeat()
                    last_hb = now

                if n == 0:
                    # Not yet due: sleep until the next physics step (or viewer frame) is.
                    behind = (time.perf_counter() - anchor) - steps * dt
                    time.sleep(max(0.0, min(dt - behind, VIEWER_PERIOD_S)))


def main():
    rclpy.init()
    node = NandiSimRunner()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run_sim()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
