#!/usr/bin/env python3
"""Atharv Nandi browser bridge: ROS 2 <-> WebSocket, plus the static web UI, on one port.

    http://<host>:8080/     Three.js client (web/)
    ws://<host>:8080/ws     JSON protocol (see controllers/web_scene.py)

ROS spins in a background thread; the websockets 13 asyncio server owns the main thread.
ROS callbacks only store the latest values (under a lock); everything client-facing,
including MuJoCo forward kinematics, runs on the asyncio loop.
"""
import asyncio
import collections
import email.utils
import http
import logging
import os
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.http11 import Response

sys.path.insert(0, os.path.dirname(__file__))
from controllers import web_scene as proto
from controllers.farm_scene import load_farm_model

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
ALLOWED_HOSTS = proto.allowed_hostnames(os.environ.get("NANDI_WEB_ALLOWED_HOSTS"))
WS_PATH = "/ws"
STREAM_HZ = 30.0
HOLD_REPUBLISH_HZ = 10.0
# Browser re-sends a held cmd at 5 Hz; 0.5 s without one = released (the contract's cmd_vel dead-man,
# which the 10 Hz republish below would otherwise mask).
HOLD_REFRESH_TIMEOUT_S = 0.5
SERVICE_TIMEOUT_S = 5.0
ENVELOPE_TIMEOUT_S = 20.0


class WebBridgeNode(Node):
    """ROS side only: publishers, latest-value subscriptions, Trigger clients."""

    def __init__(self, nq: int):
        super().__init__("nandi_web_bridge")
        self._nq = nq
        self._lock = threading.Lock()
        self._qpos, self._qpos_seq, self._qpos_stamp = None, 0, None
        self.statuses = {"game": None, "harvest": None, "loco": None}
        self.on_wireframe = None  # set by the asyncio side; called from the ROS thread

        self.cmd_pub = self.create_publisher(Twist, "/nandi/cmd_vel", 10)
        self.gait_pub = self.create_publisher(Float64MultiArray, "/nandi/gait_params", 10)
        self.arm_pub = self.create_publisher(Float64MultiArray, "/nandi/manual_arm", 10)
        self.wf_pub = self.create_publisher(String, "/nandi/workspace_wireframe", 10)

        self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self._on_qpos, 10)
        for key, topic in (("game", "/nandi/game_status"), ("harvest", "/nandi/harvest_status"),
                           ("loco", "/nandi/locomotion_status")):
            self.create_subscription(String, topic, lambda msg, k=key: self._on_status(k, msg), 10)
        self.create_subscription(String, "/nandi/workspace_wireframe", self._on_wireframe, 10)

        # NB: Node already has a `clients` property, hence `srv_clients`.
        self.srv_clients = {name: self.create_client(Trigger, srv) for name, srv in proto.SERVICE_TOPICS.items()}

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

    def _on_wireframe(self, msg: String):
        cb = self.on_wireframe
        if cb is not None:
            cb(msg.data)

    # ---- called from the asyncio thread
    def latest_qpos(self):
        with self._lock:
            return self._qpos_seq, self._qpos, self._qpos_stamp

    def publish_cmd(self, vx: float, vy: float, yaw: float, hold: bool):
        lx, ly, lz, az = proto.cmd_to_twist(vx, vy, yaw, hold)
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z = lx, ly, lz, az
        self.cmd_pub.publish(msg)

    async def call_trigger(self, name: str, timeout: float = SERVICE_TIMEOUT_S):
        """Await a Trigger service from asyncio -> (success, message). Never raises."""
        client = self.srv_clients[name]
        if not client.service_is_ready():
            return False, f"Service {proto.SERVICE_TOPICS[name]} is offline"
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
            return False, f"{proto.SERVICE_TOPICS[name]} timed out after {timeout:.0f} s"
        try:
            res = fut.result()
        except Exception as exc:  # noqa: BLE001 - surface any rclpy failure to the browser
            return False, f"{proto.SERVICE_TOPICS[name]} failed: {exc}"
        if res is None:
            return False, f"{proto.SERVICE_TOPICS[name]} returned no response"
        return bool(res.success), str(res.message)


class _Client:
    """One browser. Events are queued in order; state frames keep only the newest (no lag build-up)."""

    def __init__(self, conn):
        self.conn = conn
        self.events = collections.deque(maxlen=64)
        self.state = None
        self.wake = asyncio.Event()

    def push_event(self, text: str):
        self.events.append(text)
        self.wake.set()

    def push_state(self, text: str):
        self.state = text
        self.wake.set()

    async def run_sender(self):
        try:
            while True:
                await self.wake.wait()
                self.wake.clear()
                while self.events:
                    await self.conn.send(self.events.popleft())
                if self.state is not None:
                    text, self.state = self.state, None
                    await self.conn.send(text)
        except ConnectionClosed:
            pass


class WebBridge:
    """asyncio side: HTTP static files, WebSocket clients, 30 Hz state stream, held-cmd republish."""

    def __init__(self, node: WebBridgeNode, model: mujoco.MjModel):
        self.node, self.model = node, model
        self.data = mujoco.MjData(model)
        scene = proto.build_scene(model)
        self.scene_msg = proto.encode_message(scene)
        self.body_ids = [b["id"] for b in scene["bodies"]]
        self.arm_low, self.arm_high = scene["arm"]["low"], scene["arm"]["high"]
        self.clients = set()
        self.hold = proto.HoldTracker(HOLD_REFRESH_TIMEOUT_S)
        self.envelope_msg = proto.encode_message({"type": "envelope", "data": None})
        self.envelope_cache = None  # last good /compute_work_envelope JSON
        self._seen_seq, self._poses, self._telem = -1, None, None
        self._tasks = set()
        self.loop = None

    # ---- helpers
    def _spawn(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _broadcast_event(self, text: str):
        for c in list(self.clients):
            c.push_event(text)

    @staticmethod
    def _result(client: _Client, name: str, ok: bool, message: str):
        client.push_event(proto.encode_message({"type": "result", "name": name, "success": ok, "message": message}))

    def _publish(self, cmd):
        if cmd is not None:
            self.node.publish_cmd(*cmd)

    # ---- HTTP
    def _process_request(self, connection, request):
        path = request.path.split("?", 1)[0]
        if path == WS_PATH:
            if not proto.origin_allowed(request.headers.get("Origin"), request.headers.get("Host"),
                                        ALLOWED_HOSTS):
                return connection.respond(http.HTTPStatus.FORBIDDEN, "Cross-origin WebSocket refused\n")
            return None  # continue the WebSocket handshake
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return connection.respond(http.HTTPStatus.NOT_FOUND, f"WebSocket endpoint is {WS_PATH}\n")
        full = proto.resolve_static_path(WEB_DIR, request.path)
        if full is None:
            return connection.respond(http.HTTPStatus.NOT_FOUND, "Not found\n")
        with open(full, "rb") as f:
            body = f.read()
        headers = Headers()
        headers["Date"] = email.utils.formatdate(usegmt=True)
        headers["Connection"] = "close"
        headers["Content-Type"] = proto.mime_type(full)
        headers["Content-Length"] = str(len(body))
        headers["Cache-Control"] = "no-cache"
        return Response(200, "OK", headers, body)

    # ---- WebSocket
    async def _handler(self, conn):
        client = _Client(conn)
        self.clients.add(client)
        sender = asyncio.create_task(client.run_sender())
        client.push_event(self.scene_msg)
        client.push_event(self.envelope_msg)
        try:
            async for raw in conn:
                self._on_message(client, raw)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(client)
            sender.cancel()
            self._publish(self.hold.on_disconnect(client))  # never leave the robot walking

    def _on_message(self, client: _Client, raw):
        try:
            msg = proto.parse_client_message(raw, self.arm_low, self.arm_high)
        except proto.BadMessage as exc:
            client.push_event(proto.encode_message({"type": "error", "message": str(exc)}))
            return
        try:
            self._dispatch(client, msg)
        except Exception as exc:  # noqa: BLE001 - one bad request must not drop the connection
            self.node.get_logger().error(f"web client request {msg.get('type')} failed: {exc}")
            client.push_event(proto.encode_message({"type": "error", "message": "internal error"}))

    def _dispatch(self, client: _Client, msg):
        kind = msg["type"]
        if kind == "cmd":
            self._publish(self.hold.on_cmd(client, msg["vx"], msg["vy"], msg["yaw"], msg["hold"], time.monotonic()))
        elif kind == "service":
            if msg["name"] in ("stop", "auto_harvest", "reset_world"):
                # Stand still first (like the Tk panel), or the 10 Hz republish would restart the walk
                # right after stop_motion: during the pick, or straight after the reset.
                self._publish(self.hold.release())
            self._spawn(self._run_service(client, msg["name"]))
        elif kind == "gait_params":
            self.node.gait_pub.publish(Float64MultiArray(data=msg["values"]))
        elif kind == "manual_arm":
            self.node.arm_pub.publish(Float64MultiArray(data=msg["values"]))
        elif kind == "envelope":
            self._spawn(self._show_envelope(client, msg["show"]))
        elif kind == "ping":
            client.push_event(proto.encode_message({"type": "pong", "t": msg["t"]}))

    async def _run_service(self, client: _Client, name: str):
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

    # ---- periodic loops
    def _stream_tick(self):
        if not self.clients:
            return
        seq, qpos, stamp = self.node.latest_qpos()
        if qpos is not None and seq != self._seen_seq:
            self._seen_seq = seq
            self._poses = proto.pose_frame(self.model, self.data, self.body_ids, qpos)
            self._telem = proto.telemetry(qpos)
        st = self.node.statuses
        text = proto.encode_message({
            "type": "state", "poses": self._poses, "telemetry": self._telem,
            "game": st["game"], "harvest": st["harvest"], "loco": st["loco"],
            "sim_age_s": round(time.monotonic() - stamp, 2) if stamp is not None else None,
            "hold": self.hold.active,
        })
        for c in list(self.clients):
            c.push_state(text)

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
            self.node.get_logger().info(f"Atharv Nandi web UI on http://{host}:{port}/ (WebSocket {WS_PATH})")
            self._spawn(self._periodic(STREAM_HZ, self._stream_tick))
            self._spawn(self._periodic(HOLD_REPUBLISH_HZ, self._hold_tick))
            await server.serve_forever()


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
    # Loopback by default: anyone who can reach the port can drive the robot (no auth). The Docker
    # image sets NANDI_WEB_HOST=0.0.0.0 and publishes the port on 127.0.0.1 only.
    host = os.environ.get("NANDI_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("NANDI_WEB_PORT", "8080"))

    model = load_farm_model()
    rclpy.init()
    node = WebBridgeNode(model.nq)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        asyncio.run(WebBridge(node, model).run(host, port))
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
