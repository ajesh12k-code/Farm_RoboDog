#!/usr/bin/env python3
"""Atharv Nandi browser bridge: ROS 2 <-> WebSocket, plus the static web UI, on one port.

    http://<host>:8080/          Three.js operator console (web/, Three.js vendored: works offline)
    http://<host>:8080/healthz   JSON {ok, nodes, last_qpos_age_s}; 503 when sim_qpos is > 2 s old
    http://<host>:8080/media/<run_id>/tree360/<tree_id>/strip_<band>.jpg   TREE 360 strips / keyframes named by a
                                 TreeRecord (FilteredFrames from the WP25 recorder only)
    ws://<host>:8080/ws          JSON protocol (atharv_nandi/core/web_scene.py + atharv_nandi/core/console_protocol.py)

WP10 orchard console (atharv_nandi/core/console_protocol.ConsoleHub, MISSION-PRESETS 0.7): preset bar with the WP27 Ready
check, queue, TREE 360 / recovery / work / dock executors through their hooks' topics, the live gait relay of
/nandi/tree360/drive, /nandi/recovery/motion and /nandi/work/drive onto /nandi/cmd_vel (E-STOP and the console latch
win), the console link heartbeat (/nandi/link/cloud/heartbeat while a browser is connected), records, MARKs and the
Hindi-first job sheets. NANDI_SIM_LOCAL_LINK=1 (default outside field mode) also beats /nandi/link/local/heartbeat
as a stand-in for the WP1 handheld, labelled "sim handheld" in the UI; it never touches /nandi/recovery/local.

Access control (atharv_nandi/core/web_auth.py): NANDI_OPERATOR_TOKEN / NANDI_VIEWER_TOKEN; with neither
set (and NANDI_FIELD_MODE unset) everyone may drive, as before, and the browser shows 'DEV: no auth'.

ROS spins in a background thread; the websockets 13 asyncio server owns the main thread.
ROS callbacks only store the latest values (under a lock); everything client-facing,
including MuJoCo forward kinematics, runs on the asyncio loop.
"""
import asyncio
import collections
import email.utils
import http
import itertools
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.http11 import Response

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi import paths
from atharv_nandi.core import console_protocol as cp
from atharv_nandi.core import model_layout
from atharv_nandi.core import sim_state as sim_state_mod
from atharv_nandi.core import web_auth
from atharv_nandi.core import web_scene as proto
from atharv_nandi.core.farm_scene import load_farm_model, world_config
from atharv_nandi.sim.world import compose

REPO_ROOT = paths.REPO
WEB_DIR = paths.WEB_DIR
MISSIONS_DIR = os.environ.get("NANDI_MISSIONS_DIR", os.path.join(REPO_ROOT, "config", "missions"))
AUDIT_LOG = os.environ.get("NANDI_AUDIT_LOG", os.path.join(REPO_ROOT, "runs", "console_audit.jsonl"))
# The console E-STOP latch survives bridge restarts (crash, pkill, launcher supervise loop).
ESTOP_STATE = os.environ.get("NANDI_ESTOP_STATE", os.path.join(REPO_ROOT, "runs", "console_estop.json"))
ALLOWED_HOSTS = proto.allowed_hostnames(os.environ.get("NANDI_WEB_ALLOWED_HOSTS"))
WS_PATH = "/ws"
HEALTH_PATH = "/healthz"
STREAM_HZ = 30.0
HOLD_REPUBLISH_HZ = 10.0
# Browser re-sends a held cmd at 5 Hz; 0.5 s without one = released (the contract's cmd_vel dead-man,
# which the 10 Hz republish below would otherwise mask).
HOLD_REFRESH_TIMEOUT_S = 0.5
SERVICE_TIMEOUT_S = 5.0
ENVELOPE_TIMEOUT_S = 20.0
WS_POLICY_CLOSE = 1008
HUB_HZ = 10.0
RUNS_DIR = os.environ.get("NANDI_RUNS_DIR") or os.path.join(REPO_ROOT, "runs")
TOUR_FILE = os.path.join(REPO_ROOT, "config", "missions", "orchard_tour.yaml")


class WebBridgeNode(Node):
    """ROS side only: publishers, latest-value subscriptions, Trigger clients."""

    def __init__(self, nq: int):
        super().__init__("nandi_web_bridge")
        node_params.declare(self, "nandi_web_bridge")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
        self._nq = nq
        self._lock = threading.Lock()
        self._qpos, self._qpos_seq, self._qpos_stamp = None, 0, None
        self.statuses = {"game": None, "harvest": None, "loco": None}
        self.ext = {key: None for key in cp.EXT_TOPICS}
        self.stamps = {}  # topic key -> time.monotonic() of the last message
        self.camera = None  # latest camera event dict (rate-limited)
        self._camera_rate = cp.RateLimiter(cp.CAMERA_MAX_HZ)
        self.on_wireframe = None  # set by the asyncio side; called from the ROS thread
        self._sim_state = None    # newest /nandi/sim_state text (WP0: qpos + mocap + provider mutations)
        self._sim_state_seq = 0
        self.hub_inbox = collections.deque(maxlen=512)  # (key, text, monotonic) for the ConsoleHub

        self.cmd_pub = self.create_publisher(Twist, "/nandi/cmd_vel", 10)
        self.gait_pub = self.create_publisher(Float64MultiArray, "/nandi/gait_params", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/nandi/manual_arm", 10)
        self.wf_pub = self.create_publisher(String, "/nandi/workspace_wireframe", 10)
        self.mission_pub = self.create_publisher(String, cp.MISSION_COMMAND_TOPIC, 10)

        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self._on_qpos, 10)
        for key, topic in (("game", "/nandi/game_status"), ("harvest", "/nandi/harvest_status"),
                           ("loco", "/nandi/locomotion_status")):
            self.create_subscription(String, topic, lambda msg, k=key: self._on_status(k, msg), 10)
        self.create_subscription(String, "/nandi/workspace_wireframe", self._on_wireframe, 10)
        # Optional topics from the safety / mission / perception / telemetry packages: absent = 'n/a'.
        for key, topic in cp.EXT_TOPICS.items():
            self.create_subscription(String, topic, lambda msg, k=key: self._on_ext(k, msg), 10)
        self.create_subscription(CompressedImage, cp.CAMERA_TOPIC, self._on_camera, 2)
        self.create_subscription(String, "/nandi/sim_state", self._on_sim_state, 2)
        for key, topic in cp.HUB_TOPICS_IN.items():
            self.create_subscription(String, topic, lambda msg, k=key: self._on_hub(k, msg), 20)
        self.hub_pubs = {key: self.create_publisher(String, topic, 10) for key, topic in cp.HUB_TOPICS_OUT.items()}

        # NB: Node already has a `clients` property, hence `srv_clients`.
        self.srv_clients = {name: self.create_client(Trigger, srv) for name, srv in cp.SERVICE_TOPICS.items()}

    # ---- ROS-thread callbacks
    def _on_qpos(self, msg: Float64MultiArray):
        if len(msg.data) != self._nq:
            self.get_logger().warn(
                f"sim_qpos has {len(msg.data)} values, farm model expects nq={self._nq}; ignoring "
                "(is every node building its model via load_farm_model()?)", throttle_duration_sec=5.0)
            return
        q = np.array(msg.data, dtype=np.float64)
        with self._lock:
            self._qpos, self._qpos_seq, self._qpos_stamp = q, self._qpos_seq + 1, time.monotonic()

    def _on_status(self, key: str, msg: String):
        self.statuses[key] = proto.parse_status(msg.data)
        self.stamps[key] = time.monotonic()

    def _on_ext(self, key: str, msg: String):
        self.ext[key] = cp.parse_ext(key, msg.data)
        self.stamps[key] = time.monotonic()

    def _on_camera(self, msg: CompressedImage):
        now = time.monotonic()
        self.stamps["camera"] = now
        if not self._camera_rate.accept(now):
            return
        frame = cp.camera_frame(msg.format, bytes(msg.data), time.time())
        if frame is not None:
            self.camera = frame

    def _on_sim_state(self, msg: String):
        with self._lock:
            self._sim_state, self._sim_state_seq = msg.data, self._sim_state_seq + 1

    def _on_hub(self, key: str, msg: String):
        self.hub_inbox.append((key, msg.data, time.monotonic()))

    def latest_sim_state(self):
        with self._lock:
            return self._sim_state_seq, self._sim_state

    def publish_hub(self, key: str, payload):
        self.hub_pubs[key].publish(String(data=json.dumps(payload, separators=(",", ":"))))

    def _on_wireframe(self, msg: String):
        cb = self.on_wireframe
        if cb is not None:
            cb(msg.data)

    # ---- called from the asyncio thread
    def latest_qpos(self):
        with self._lock:
            return self._qpos_seq, self._qpos, self._qpos_stamp

    def topic_ages(self, now: float):
        """Seconds since each known topic last arrived (None = never seen)."""
        _, _, qstamp = self.latest_qpos()
        ages = {"sim": None if qstamp is None else round(now - qstamp, 3)}
        for key in ("game", "harvest", "loco", *cp.EXT_TOPICS, "camera"):
            stamp = self.stamps.get(key)
            ages[key] = None if stamp is None else round(now - stamp, 2)
        return ages

    def publish_cmd(self, vx: float, vy: float, yaw: float, hold: bool):
        lx, ly, lz, az = proto.cmd_to_twist(vx, vy, yaw, hold)
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z = lx, ly, lz, az
        self.cmd_pub.publish(msg)

    def service_ready(self, name: str) -> bool:
        return self.srv_clients[name].service_is_ready()

    async def call_trigger(self, name: str, timeout: float = SERVICE_TIMEOUT_S):
        """Await a Trigger service from asyncio -> (success, message). Never raises."""
        client = self.srv_clients[name]
        if not client.service_is_ready():
            return False, f"Service {cp.SERVICE_TOPICS[name]} is offline"
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def _set(fut):
            if not done.done():
                done.set_result(fut)

        ros_future = client.call_async(Trigger.Request())
        # rclpy completes the future on the executor thread; hop back onto the asyncio loop.
        ros_future.add_done_callback(lambda fut: loop.call_soon_threadsafe(_set, fut))
        try:
            fut = await asyncio.wait_for(done, timeout)
        except asyncio.TimeoutError:
            ros_future.cancel()
            return False, f"{cp.SERVICE_TOPICS[name]} timed out after {timeout:.0f} s"
        try:
            res = fut.result()
        except Exception as exc:  # noqa: BLE001 - surface any rclpy failure to the browser
            return False, f"{cp.SERVICE_TOPICS[name]} failed: {exc}"
        if res is None:
            return False, f"{cp.SERVICE_TOPICS[name]} returned no response"
        return bool(res.success), str(res.message)


class _Client:
    """One browser. Events are queued in order; state/camera frames keep only the newest."""

    _ids = itertools.count(1)

    def __init__(self, conn, auth: web_auth.ClientAuth):
        self.conn = conn
        self.auth = auth
        peer = conn.remote_address
        self.id = f"c{next(self._ids)}@{peer[0]}:{peer[1]}" if peer else f"c{next(self._ids)}"
        self.events = collections.deque(maxlen=64)
        self.state = None
        self.camera = None
        self.camera_t = None
        self.wake = asyncio.Event()
        self.budget = cp.ByteBudget()   # MP 0.7: <= 1 MiB/s per client; events always pass
        self.last_seen = time.monotonic()

    def push_event(self, text: str):
        self.events.append(text)
        self.wake.set()

    def push_state(self, text: str):
        self.state = text
        self.wake.set()

    def push_camera(self, text: str):
        self.camera = text
        self.wake.set()

    async def run_sender(self):
        try:
            while True:
                await self.wake.wait()
                self.wake.clear()
                while self.events:
                    text = self.events.popleft()
                    self.budget.allow(len(text), time.monotonic(), must=True)
                    await self.conn.send(text)
                if not self.auth.authenticated:
                    continue  # nothing but hello/auth replies before authentication
                if self.camera is not None:
                    text, self.camera = self.camera, None
                    if self.budget.allow(len(text), time.monotonic()):
                        await self.conn.send(text)
                if self.state is not None:
                    text, self.state = self.state, None
                    if self.budget.allow(len(text), time.monotonic()):  # over budget: skip, the next frame wins
                        await self.conn.send(text)
        except ConnectionClosed:
            pass


class WebBridge:
    """asyncio side: HTTP static files + /healthz, WebSocket clients, 30 Hz state stream, held-cmd republish."""

    def __init__(self, node: WebBridgeNode, model: mujoco.MjModel, auth: web_auth.AuthConfig):
        self.node, self.model, self.auth = node, model, auth
        self.audit = cp.ConsoleAuditLog(AUDIT_LOG)
        self.data = mujoco.MjData(model)
        scene = proto.build_scene(model)
        self.scene = cp.extend_scene(scene)
        self.body_ids = [b["id"] for b in scene["bodies"]]
        self.scene_geom_ids = [g["id"] for g in scene["geoms"]]
        self.styled_ids = list(scene.get("styled") or [])
        self._styles, self._styles_new = proto.geom_styles(model, self.styled_ids), False
        lay = model_layout.current()
        self.world_id = lay.world_id if lay is not None else None
        sim_state_mod.discover_providers()
        self._state_seen, self._state_errors, self._hidden = -1, 0, []
        cfg = world_config()
        field = compose.load_field(cfg["world"], cfg["seed"])
        tour = None
        try:
            import yaml
            with open(TOUR_FILE, encoding="utf-8") as fh:
                tour = yaml.safe_load(fh)
        except (OSError, ValueError, ImportError):
            tour = None
        local_link = cp.sim_local_link_on(os.environ.get("NANDI_SIM_LOCAL_LINK"), auth.field_mode)
        if auth.field_mode and os.environ.get("NANDI_SIM_LOCAL_LINK", "").strip():
            node.get_logger().warn("NANDI_SIM_LOCAL_LINK ignored in field mode (the sim handheld beat is never faked)")
        self.hub = cp.ConsoleHub(field, sim_local_link=local_link,
                                 sim_tools=not auth.field_mode, novnc=os.environ.get("NANDI_NOVNC_UP") == "1",
                                 tour_doc=tour)
        self._ctx = None
        self._robot_xy = (0.0, 0.0)
        self.arm_low, self.arm_high = scene["arm"]["low"], scene["arm"]["high"]
        self.clients = set()
        self.hold = proto.HoldTracker(HOLD_REFRESH_TIMEOUT_S)
        self.envelope_msg = proto.encode_message({"type": "envelope", "data": None})
        self.envelope_cache = None  # last good /compute_work_envelope JSON
        self.estop = cp.EstopLatch(ESTOP_STATE)  # console-side latch, persisted across restarts
        self.auth_throttle = web_auth.AuthThrottle()
        if self.estop.latched:
            node.get_logger().warn(f"E-STOP latch restored from {ESTOP_STATE}: {self.estop.state['message']}")
        self._seen_seq, self._poses, self._telem = -1, None, None
        self._camera_seen = None
        self._tasks = set()
        self.loop = None
        self.stop_signal = None  # signal number that ended run(), if any

    # ---- helpers
    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _broadcast_event(self, text: str):
        """Event for every authenticated client (sockets still inside their auth window get nothing)."""
        for c in list(self.clients):
            if c.auth.authenticated:
                c.push_event(text)

    @staticmethod
    def _send(client: _Client, msg):
        client.push_event(proto.encode_message(msg))

    def _result(self, client: _Client, name: str, ok: bool, message: str):
        self._send(client, {"type": "result", "name": name, "success": ok, "message": message})

    def _publish(self, cmd):
        if cmd is not None:
            self.node.publish_cmd(*cmd)

    # ---- HTTP
    def _health(self):
        now = time.monotonic()
        ages = self.node.topic_ages(now)
        return cp.health_report(ages["sim"], ages, cp.fresh_ext(self.node.ext, ages).get("safety"))

    def _process_request(self, connection, request):
        path = request.path.split("?", 1)[0]
        if path == WS_PATH:
            peer = connection.remote_address[0] if connection.remote_address else None
            if self.auth.auth_required and self.auth_throttle.blocked(peer, time.monotonic()):
                return connection.respond(http.HTTPStatus.TOO_MANY_REQUESTS,
                                          "Too many failed authentications; try again later\n")
            if not proto.origin_allowed(request.headers.get("Origin"), request.headers.get("Host"),
                                        ALLOWED_HOSTS):
                return connection.respond(http.HTTPStatus.FORBIDDEN, "Cross-origin WebSocket refused\n")
            return None  # continue the WebSocket handshake
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return connection.respond(http.HTTPStatus.NOT_FOUND, f"WebSocket endpoint is {WS_PATH}\n")
        if path == HEALTH_PATH:
            status, body = self._health()
            return self._response(status, json.dumps(body, separators=(",", ":")).encode(), "application/json")
        if path.startswith("/media/"):
            full = cp.resolve_media_path(RUNS_DIR, request.path, self.hub.media_allowed, self.hub.media_key)
            if full is None:
                return connection.respond(http.HTTPStatus.NOT_FOUND, "Not found\n")
            with open(full, "rb") as f:
                body = f.read()
            return self._response(200, body, "image/jpeg")
        full = proto.resolve_static_path(WEB_DIR, request.path)
        if full is None:
            return connection.respond(http.HTTPStatus.NOT_FOUND, "Not found\n")
        with open(full, "rb") as f:
            body = f.read()
        return self._response(200, body, proto.mime_type(full))

    @staticmethod
    def _response(status: int, body: bytes, ctype: str):
        headers = Headers()
        headers["Date"] = email.utils.formatdate(usegmt=True)
        headers["Connection"] = "close"
        headers["Content-Type"] = ctype
        headers["Content-Length"] = str(len(body))
        headers["Cache-Control"] = "no-cache"
        return Response(status, http.HTTPStatus(status).phrase, headers, body)

    # ---- WebSocket
    async def _handler(self, conn):
        path = conn.request.path if conn.request is not None else ""
        auth = web_auth.ClientAuth(self.auth, time.monotonic(), web_auth.token_from_path(path))
        client = _Client(conn, auth)
        self.clients.add(client)
        sender = asyncio.create_task(client.run_sender())
        self._send(client, self._hello(client))
        if auth.failures:
            self.auth_throttle.record_failure(self._peer(client), time.monotonic())
        if auth.authenticated:
            self._welcome(client)
        else:
            self._spawn(self._auth_deadline(client))
        try:
            async for raw in conn:
                self._on_message(client, raw)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(client)
            self.audit.forget(client.id)
            sender.cancel()
            self._publish(self.hold.on_disconnect(client))  # never leave the robot walking

    @staticmethod
    def _peer(client: _Client):
        addr = client.conn.remote_address
        return addr[0] if addr else None

    def _hello(self, client: _Client):
        return {"type": "hello", "auth_required": self.auth.auth_required, "dev_mode": self.auth.dev_mode,
                "field_mode": self.auth.field_mode, "role": client.auth.role, "client": client.id}

    def _welcome(self, client: _Client):
        scene = dict(self.scene)
        scene["console"] = {**scene["console"], "mission_zones": cp.list_zones(MISSIONS_DIR),
                            "role": client.auth.role}
        client.push_event(proto.encode_message(scene))
        client.push_event(self.envelope_msg)
        client.push_event(proto.encode_message({"type": "styles", "data": self._styles}))
        ctx = self._ctx or self._context(time.monotonic())
        for msg in self.hub.snapshot(time.monotonic(), ctx):
            client.push_event(proto.encode_message(msg))

    async def _auth_deadline(self, client: _Client):
        await asyncio.sleep(web_auth.AUTH_TIMEOUT_S)
        if client in self.clients and client.auth.expired(time.monotonic()):
            self._send(client, {"type": "auth", "ok": False, "role": None,
                                "message": f"no valid token within {web_auth.AUTH_TIMEOUT_S:.0f} s"})
            await asyncio.sleep(0.1)  # let the sender flush the reason
            await client.conn.close(WS_POLICY_CLOSE, "authentication required")

    def _on_message(self, client: _Client, raw):
        client.last_seen = time.monotonic()
        try:
            msg = cp.parse_any(raw, self.arm_low, self.arm_high)
        except proto.BadMessage as exc:
            self._send(client, {"type": "error", "message": str(exc)})
            return
        kind, name = msg["type"], msg.get("name")
        if kind == "auth":
            self._on_auth(client, msg["token"])
            return
        allowed, reason = web_auth.permit(client.auth.role, kind, name)
        if not allowed:
            self._send(client, {"type": "error", "message": reason})
            return
        if self.estop.latched and cp.blocked_by_estop(msg):
            self._result(client, name or kind, False, "E-STOP is latched: press RESET E-STOP first")
            return
        if self.hub.relaying and cp.blocked_by_relay(msg):
            # MP 0.3: arm PARKed and no second motion source while a preset walks the robot
            self._result(client, name or kind, False, "A preset is driving the robot: Pause it (or Take over) first")
            return
        self.audit.record(time.time(), client.id, client.auth.role, kind, name, cp.console_audit_values(msg))
        try:
            self._dispatch(client, msg)
        except Exception as exc:  # noqa: BLE001 - one bad request must not drop the connection
            self.node.get_logger().error(f"web client request {kind} failed: {exc}")
            self._send(client, {"type": "error", "message": "internal error"})

    def _on_auth(self, client: _Client, token: str):
        was = client.auth.authenticated
        ok = client.auth.try_token(token)
        role = client.auth.role
        if not ok:
            self.auth_throttle.record_failure(self._peer(client), time.monotonic())
        self._send(client, {"type": "auth", "ok": ok, "role": role,
                            "message": f"authenticated as {role}" if ok else "invalid token"})
        if ok and not was:
            self._welcome(client)
        elif client.auth.locked_out:
            self._spawn(self._close_after_flush(client, "too many invalid tokens"))

    async def _close_after_flush(self, client: _Client, reason: str):
        await asyncio.sleep(0.1)  # let the sender flush the auth reply
        await client.conn.close(WS_POLICY_CLOSE, reason)

    def _dispatch(self, client: _Client, msg):
        kind = msg["type"]
        if kind == "cmd":
            self._publish(self.hold.on_cmd(client, msg["vx"], msg["vy"], msg["yaw"], msg["hold"], time.monotonic()))
        elif kind == "service":
            if msg["name"] in cp.STOPPING_SERVICES:
                # Stand still first (like the Tk panel), or the 10 Hz republish would restart the walk
                # right after stop_motion: during the pick, or straight after the reset. The console relay
                # holds too (pauses the running preset), until an operator Resume / start.
                self._publish(self.hold.release())
                now = time.monotonic()
                note = self.hub.stop_motion(now, msg["name"])
                if note is not None:
                    self._publish(self.hub.drive(now, self._estopped()))
                    self._flush_hub(now)
            self._spawn(self._run_service(client, msg["name"]))
        elif kind == "mission_start":
            text, message = cp.mission_command(msg["zone"], MISSIONS_DIR, REPO_ROOT)
            if text is not None:
                self.node.mission_pub.publish(String(data=text))
            self._result(client, "mission_start", text is not None, message)
        elif kind == "gait_params":
            self.node.gait_pub.publish(Float64MultiArray(data=msg["values"]))
        elif kind == "manual_arm":
            self.node.arm_pub.publish(Float64MultiArray(data=msg["values"]))
        elif kind == "envelope":
            self._spawn(self._show_envelope(client, msg["show"]))
        elif kind == "ping":
            self._send(client, {"type": "pong", "t": msg["t"]})
        elif kind in cp.CONSOLE_TYPES:
            now = time.monotonic()
            ok, message, replies = self.hub.handle(msg, now, self._ctx or self._context(now), self._estopped())
            for r in replies:
                self._send(client, r)
            self._result(client, kind, ok, message)
            self._flush_hub(now)

    async def _run_service(self, client: _Client, name: str):
        if name == "estop":
            await self._estop(client)
            return
        if name == "estop_reset":
            await self._estop_reset(client)
            return
        if name == "work_envelope":
            self.envelope_cache = None  # explicit request = recompute
            await self._show_envelope(client, True)
            return
        if name == "auto_harvest":
            await self.node.call_trigger("stop")  # same as the Tk panel: stand still before picking
        ok, message = await self.node.call_trigger(name)
        if name == "reset_world" and ok:
            self.node.wf_pub.publish(String(data=""))  # keep every viewer's envelope in sync with the reset
        self._result(client, name, ok, message)

    async def _estop(self, client: _Client):
        """E-STOP: latch the console, stop the gait now, and tell the safety supervisor.

        /nandi/mission/abort is always called too, so a mission executor stops even when it runs
        without a supervisor. Without a supervisor (/nandi/estop offline) the console latch still
        blocks motion commands and a running harvest is aborted via release_arm.
        """
        seq = self.estop.latch(client.id, time.time())
        self.node.get_logger().warn(f"E-STOP from {client.id} ({client.auth.role})")
        (ok, message), (stop_ok, stop_msg), (abort_ok, abort_msg) = await asyncio.gather(
            self.node.call_trigger("estop"), self.node.call_trigger("stop"),
            self.node.call_trigger("mission_abort"))
        mission = f"mission_abort ({abort_msg})" if self.node.service_ready("mission_abort") else "no mission executor"
        if not self.node.service_ready("estop"):
            arm_ok, arm_msg = await self.node.call_trigger("release_arm")
            ok = stop_ok
            message = (f"E-STOP latched in console; /nandi/estop offline, fell back to stop_motion "
                       f"({stop_msg}) + release_arm ({arm_msg}) + {mission}")
        else:
            message = f"E-STOP: {message}; {mission}"
        self.estop.set_message(seq, message)
        self._broadcast_event(proto.encode_message(
            {"type": "result", "name": "estop", "success": ok, "message": message}))

    async def _estop_reset(self, client: _Client):
        seq = self.estop.seq  # an E-STOP pressed while the reset is in flight must win
        if self.node.service_ready("estop_reset"):
            ok, message = await self.node.call_trigger("estop_reset")
        else:
            ok, message = True, "/nandi/estop_reset offline: console latch cleared (no safety supervisor)"
        if ok and not self.estop.clear(seq):
            ok, message = False, "E-STOP pressed again during the reset: still latched, reset again"
        self.node.get_logger().warn(f"E-STOP reset from {client.id} ({client.auth.role}): {ok} {message}")
        self._broadcast_event(proto.encode_message(
            {"type": "result", "name": "estop_reset", "success": ok, "message": message}))

    async def _show_envelope(self, client: _Client, show: bool):
        if not show:
            self.node.wf_pub.publish(String(data=""))
            self._result(client, "envelope", True, "Work envelope hidden")
            return
        if self.envelope_cache is None:
            ok, text = await self.node.call_trigger("work_envelope", ENVELOPE_TIMEOUT_S)
            if not ok:
                self._result(client, "envelope", False, text)
                return
            env = proto.parse_envelope(text)
            if env is None:
                self._result(client, "envelope", False, "Work envelope service returned no usable segments")
                return
            self.envelope_cache = text
        else:
            env = proto.parse_envelope(self.envelope_cache)
        # Publishing (not just sending) also shows it in the MuJoCo viewer and the other browsers;
        # our own subscription forwards it to every client via _on_wireframe_text.
        self.node.wf_pub.publish(String(data=self.envelope_cache))
        self._result(client, "envelope", True,
                     f"Work envelope: {len(env['canopy_segments'])} canopy + {len(env['floor_segments'])} floor segments")

    def _on_wireframe_text(self, text: str):
        self.envelope_msg = proto.encode_message({"type": "envelope", "data": proto.parse_envelope(text)})
        self._broadcast_event(self.envelope_msg)

    def _wireframe_from_ros(self, text: str):
        """ROS thread -> asyncio loop."""
        try:
            self.loop.call_soon_threadsafe(self._on_wireframe_text, text)
        except RuntimeError:
            pass  # loop already closed during shutdown

    # ---- WP10 console hub
    def _apply_sim_state(self, text: str):
        try:
            st = sim_state_mod.decode(text)
        except sim_state_mod.SimStateError as exc:
            return False, str(exc)
        ok, why = sim_state_mod.apply(self.model, self.data, st, self.world_id or None)
        if ok:
            mujoco.mj_kinematics(self.model, self.data)
        return ok, why

    def _estopped(self) -> bool:
        safety = cp.fresh_ext(self.node.ext, self.node.topic_ages(time.monotonic())).get("safety") or {}
        return self.estop.latched or bool(safety.get("estop") or safety.get("latched") or safety.get("estopped"))

    def _context(self, now: float):
        tel = self._telem or {}
        if not self.clients:
            # _stream_tick (which keeps _telem current) idles without browsers; the queue still dispatches, so the
            # Ready check and the dispatch projection (reserve, ETA, distance home) need the live pose anyway
            _, qpos, _ = self.node.latest_qpos()
            tel = (proto.telemetry(qpos) if qpos is not None else None) or tel
        if tel:
            self._robot_xy = (float(tel.get("x", 0.0)), float(tel.get("y", 0.0)))
        try:
            free_mb = shutil.disk_usage(RUNS_DIR if os.path.isdir(RUNS_DIR) else REPO_ROOT).free / 1e6
        except OSError:
            free_mb = 0.0
        game = self.node.statuses.get("game") or {}
        self._ctx = self.hub.ready_context(now, self._robot_xy, self._estopped(), disk_free_mb=free_mb,
                                           fallen=bool(game.get("fallen")))
        return self._ctx

    def _flush_hub(self, now: float):
        for key, payload in self.hub.take_outbox():
            self.node.publish_hub(key, payload)
        for ev in self.hub.take_events():
            self._broadcast_event(proto.encode_message(ev))

    def _hub_step(self, what: str, fn):
        """One stage of the hub tick: a fault is logged and the later stages (heartbeats, relay) still run."""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            self.node.get_logger().error(f"hub {what} failed: {type(exc).__name__}: {exc}", throttle_duration_sec=5.0)
            return None

    def _hub_tick(self):
        now = time.monotonic()
        inbox = self.node.hub_inbox
        while inbox:
            key, text, t = inbox.popleft()
            self._hub_step(f"ingest {key}", lambda k=key, x=text, tt=t: self.hub.ingest(k, x, tt))
        loco = self.node.statuses.get("loco") or {}
        if isinstance(loco, dict):  # any /nandi/stop_motion (Tk panel, API, another client) holds the relay
            self._hub_step("stop_seq", lambda: self.hub.observe_stop_seq(loco.get("stop_seq"), now))
        estop = self._estopped()
        ctx = self._hub_step("context", lambda: self._context(now)) or self._ctx
        alive = any(c.auth.authenticated and now - c.last_seen < cp.CLIENT_ALIVE_S for c in self.clients)
        if ctx is not None:
            self._hub_step("tick", lambda: self.hub.tick(now, ctx, estop, alive))
        cmd = self._hub_step("drive", lambda: self.hub.drive(now, estop))
        if cmd is not None and not self.hold.active:
            self.node.publish_cmd(*cmd)
        for msg in (self._hub_step("streams", lambda: self.hub.streams(now, ctx)) if ctx is not None else None) or []:
            if msg["type"] == "preset_status":
                self.node.publish_hub("preset_state", msg)
            if self.clients:
                self._broadcast_event(proto.encode_message(msg))
        self._flush_hub(now)

    # ---- periodic loops
    def _stream_tick(self):
        if not self.clients:
            return
        now = time.monotonic()
        seq, qpos, stamp = self.node.latest_qpos()
        sseq, stext = self.node.latest_sim_state()
        if stext is not None and sseq != self._state_seen and self._state_errors < 50:
            # /nandi/sim_state also carries mocap (actors) and provider mutations (crown mast sections, fruit pool,
            # hidden fruit / cut shoots), which /nandi/sim_qpos alone cannot show.
            self._state_seen = sseq
            ok, why = self._apply_sim_state(stext)
            if ok:
                self._poses = proto.poses_of(self.data, self.body_ids)
                self._telem = proto.telemetry(self.data.qpos)
                self._hidden = proto.hidden_geoms(self.model, self.scene_geom_ids)
                styles = proto.geom_styles(self.model, self.styled_ids)
                if styles != self._styles:
                    self._styles, self._styles_new = styles, True
                self._seen_seq = seq
            else:
                self._state_errors += 1
                self.node.get_logger().warn(f"sim_state refused ({why}); using sim_qpos", throttle_duration_sec=10.0)
        if qpos is not None and seq != self._seen_seq and (stext is None or self._state_errors >= 50):
            self._seen_seq = seq
            self._poses = proto.pose_frame(self.model, self.data, self.body_ids, qpos)
            self._telem = proto.telemetry(qpos)
        st = self.node.statuses
        ages = self.node.topic_ages(now)
        ext = cp.fresh_ext(self.node.ext, ages)  # a dead publisher's last message must not freeze the UI
        text = proto.encode_message({
            "type": "state", "poses": self._poses, "telemetry": self._telem,
            "game": st["game"], "harvest": st["harvest"], "loco": st["loco"],
            "sim_age_s": round(now - stamp, 2) if stamp is not None else None,
            "hold": self.hold.active,
            "ext": ext,
            "ages": ages,
            "health": cp.node_health(ages, ext.get("safety")),
            "estop": self.estop.as_dict(),
            "hidden": self._hidden,
            "styles": self._styles if self._styles_new else None,  # only when a provider changed them
            "relay": self.hub.relaying,
        })
        self._styles_new = False
        cam = self.node.camera
        cam_text = None
        if cam is not None and cam is not self._camera_seen:
            self._camera_seen = cam
            cam_text = proto.encode_message(cam)
        for c in list(self.clients):
            c.push_state(text)
            if cam_text is not None:
                c.push_camera(cam_text)

    def _hold_tick(self):
        owner = self.hold.owner
        cmd = self.hold.tick(time.monotonic())
        self._publish(cmd)
        if cmd is not None and not cmd[3] and owner in self.clients:
            self._result(owner, "cmd", False, "Walk released: browser stopped refreshing the held command")

    async def _periodic(self, hz: float, tick):
        """Run tick() at a fixed rate without bursting after a stall; log, never die, on errors."""
        period = 1.0 / hz
        next_t = self.loop.time()
        while True:
            next_t += period
            now = self.loop.time()
            if next_t < now - period:
                next_t = now
            await asyncio.sleep(max(0.0, next_t - now))
            try:
                tick()
            except Exception as exc:  # noqa: BLE001
                self.node.get_logger().error(f"{tick.__name__} failed: {exc}", throttle_duration_sec=5.0)

    async def run(self, host: str, port: int):
        self.loop = asyncio.get_running_loop()
        self.node.on_wireframe = self._wireframe_from_ros
        async with serve(self._handler, host, port, process_request=self._process_request,
                         max_size=2 ** 16, ping_interval=20, ping_timeout=20) as server:
            mode = "DEV: no auth" if self.auth.dev_mode else ("field mode, tokens required" if self.auth.field_mode
                                                              else "tokens required")
            self.node.get_logger().info(
                f"Atharv Nandi web UI on http://{host}:{port}/ (WebSocket {WS_PATH}, {HEALTH_PATH}; {mode})")
            self._spawn(self._periodic(STREAM_HZ, self._stream_tick))
            self._spawn(self._periodic(HOLD_REPUBLISH_HZ, self._hold_tick))
            self._spawn(self._periodic(HUB_HZ, self._hub_tick))
            # rclpy turns SIGTERM into a context shutdown that would leave this loop serving forever:
            # stop cleanly instead, so `pkill -f "atharv_nandi.nodes web_bridge"` (or the launcher) really ends the process.
            stop = asyncio.Event()
            for sig in (signal.SIGTERM, signal.SIGINT):
                self.loop.add_signal_handler(sig, lambda s=sig: (setattr(self, "stop_signal", s), stop.set()))
            await stop.wait()


class _QuietStaticFilter(logging.Filter):
    """websockets reports every plain-HTTP reply (our static files, 404s) as a failed handshake."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        for _ in range(4):
            if exc is None:
                return True
            if isinstance(exc, InvalidStatus):
                return False
            exc = exc.__cause__ or exc.__context__
        return True


def main():
    logging.basicConfig(level=logging.WARNING, format="[web_bridge] %(levelname)s %(name)s: %(message)s")
    logging.getLogger("websockets.server").addFilter(_QuietStaticFilter())
    # Loopback by default. The Docker image sets NANDI_WEB_HOST=0.0.0.0 and publishes the port on
    # 127.0.0.1 only. Without tokens anyone who can reach the port can drive (DEV: no auth).
    host = os.environ.get("NANDI_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("NANDI_WEB_PORT", "8080"))
    try:
        auth = web_auth.AuthConfig.from_env()
    except web_auth.AuthConfigError as exc:
        print(f"[web_bridge] refusing to start: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)
    warning = auth.startup_warning(host)
    if warning:
        print(f"[web_bridge] WARNING: {warning}", file=sys.stderr, flush=True)

    model = load_farm_model()
    rclpy.init()
    node = WebBridgeNode(model.nq)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    bridge = WebBridge(node, model, auth)
    try:
        asyncio.run(bridge.run(host, port))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if bridge.stop_signal is not None:
        # 128 + signal, like a shell: launch_nandi.sh restarts a TERMinated bridge (handy after edits).
        sys.exit(128 + int(bridge.stop_signal))


if __name__ == "__main__":
    main()
