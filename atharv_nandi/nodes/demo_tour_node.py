#!/usr/bin/env python3
"""Atharv Nandi demo tour node (WP9): ``/nandi/demo/run`` -> a headless orchard tour -> records, job sheets, KPIs.

    in   /nandi/demo/run    (std_msgs/String JSON) {"steps": "full" | [step ids], "program": "tour_lite" |
                            "orchard_tour", "seed": 7, "transit": "walk" | "teleport"}   or   {"cmd": "abort"}
                            | {"cmd": "pause"} | {"cmd": "resume"}
    out  /nandi/demo/state  (String JSON, 2 Hz + on change)  TourRunner.status(): steps with status / sim_s, pass
         /nandi/demo/events (String JSON, one per event)     step_start / step_end / tour_end / refused / queued / tour_error ...
         /nandi/tree/record, /nandi/mark, /nandi/records/harvest, /nandi/jobsheet, /nandi/kpi, /nandi/kpi/scopes

The tour runs on its **own** SimDriver world in this process (``controllers.demo_tour.SimBackend``), not on the
live ``sim_runner`` world: the live stack has no hook that runs the WP8 task state machines or the WP13 dock session
(change requests to WP8, WP13, WP27 and WP9). The 3D view therefore does not move during a
``/nandi/demo/run``; the console shows the state, records and job sheets. Not started by ``launch/launch_nandi.sh``
(not a WP9 file): ``python3 -m atharv_nandi.nodes demo_tour_node`` next to the stack.
"""
import json
import threading
from typing import Any, List, Optional, Tuple

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi.core import demo_tour as dt  # noqa: E402

PROGRAMS = ("tour_lite", "orchard_tour")
TOPICS_OUT = ("/nandi/demo/state", "/nandi/demo/events", "/nandi/tree/record", "/nandi/mark",
              "/nandi/records/harvest", "/nandi/jobsheet", "/nandi/kpi", "/nandi/kpi/scopes")


def parse_run(text: str) -> Tuple[str, Any]:
    """``/nandi/demo/run`` payload -> ("run", kwargs) | ("abort" | "pause" | "resume", None); ValueError if bad."""
    try:
        d = json.loads(text)
    except (TypeError, ValueError):
        raise ValueError("demo/run: not JSON")
    if not isinstance(d, dict):
        raise ValueError("demo/run: expected an object")
    if d.get("cmd") in ("abort", "pause", "resume"):
        return d["cmd"], None
    program = d.get("program", "tour_lite")
    if program not in PROGRAMS:
        raise ValueError(f"demo/run: program {program!r} not in {PROGRAMS}")
    steps = d.get("steps", "full")
    if steps != "full" and not (isinstance(steps, list) and steps and all(isinstance(s, str) for s in steps)):
        raise ValueError("demo/run: steps must be 'full' or a list of step ids")
    seed = d.get("seed", 7)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 31:
        raise ValueError("demo/run: seed must be a non-negative int")
    transit = d.get("transit", "walk")
    if transit not in ("walk", "teleport"):
        raise ValueError("demo/run: transit must be walk or teleport")
    prog = dt.load_program(program)
    if steps != "full":
        prog.select(steps)                  # raises on an unknown step id
    return "run", {"program": program, "steps": None if steps == "full" else steps, "seed": seed, "transit": transit}


class TourJob:
    """One tour at a time on a worker thread (the ROS executor keeps spinning for state and abort).

    ``start`` clears the previous runner at once; while the new world loads (``state: loading``, no runner yet) a
    pause / resume / abort is queued and applied to the runner the moment it exists, so no command is dropped and
    none reaches a finished tour. Any error of the worker (world or model load included) is a ``tour_error`` event.
    """

    def __init__(self, publish, backend_factory=None):
        self.publish = publish
        self.backend_factory = backend_factory or (lambda world, seed, transit: dt.SimBackend(world, seed,
                                                                                              transit=transit))
        self.thread: Optional[threading.Thread] = None
        self.runner: Optional[dt.TourRunner] = None
        self.last: Optional[dict] = None
        self.loading = False
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._queued: List[str] = []

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, kw: dict) -> bool:
        if self.busy:
            return False
        with self._lock:
            self.runner = None              # never let a command reach the previous, finished tour
            self.loading, self.error, self._queued = True, None, []

        def work():
            be = None
            try:
                prog = dt.load_program(kw["program"]).select(kw["steps"])
                be = self.backend_factory(prog.world, kw["seed"], kw["transit"])
                runner = dt.TourRunner(prog, be, seed=kw["seed"], publish=self.publish)
                with self._lock:
                    for cmd in self._queued:
                        self._apply(runner, cmd)
                    self._queued = []
                    self.runner, self.loading = runner, False
                self.last = runner.run()
            except Exception as exc:  # noqa: BLE001 - report, never kill the node
                self.error = f"{type(exc).__name__}: {exc}"
                self.publish("/nandi/demo/events", {"event": "tour_error", "error": self.error})
            finally:
                self.loading = False
                if be is not None:
                    try:
                        be.close()
                    except Exception:  # noqa: BLE001
                        pass

        self.thread = threading.Thread(target=work, name="wp9_tour", daemon=True)
        self.thread.start()
        return True

    @staticmethod
    def _apply(runner, cmd: str):
        if cmd == "abort":
            runner.abort("operator")
        elif cmd == "pause":
            runner.pause()
        elif cmd == "resume":
            runner.resume()

    def control(self, cmd: str) -> Tuple[bool, str]:
        """pause / resume / abort -> (applied or queued, reason when refused)."""
        with self._lock:
            r = self.runner
            if r is None and self.loading:
                self._queued.append(cmd)
                return True, "queued"
            if r is None or not self.busy or r.state == "done":
                return False, "no tour running"
            self._apply(r, cmd)
            return True, ""

    def state(self) -> dict:
        if self.runner is not None:
            return self.runner.status()
        st = "loading" if self.loading else ("error" if self.error else "idle")
        out = {"schema": dt.STATE_SCHEMA, "state": st, "steps": [], "data_provenance": "simulated"}
        if self.error and not self.loading:
            out["error"] = self.error
        if self._queued:
            out["queued"] = list(self._queued)
        return out


def main():  # pragma: no cover - needs rclpy and a ROS graph
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    class DemoTourNode(Node):
        def __init__(self):
            super().__init__("nandi_demo_tour")
            node_params.declare(self, "nandi_demo_tour")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
            self.pubs = {t: self.create_publisher(String, t, 10) for t in TOPICS_OUT}
            self.job = TourJob(self.out)
            self.create_subscription(String, "/nandi/demo/run", self.on_run, 10)
            self.create_timer(0.5, lambda: self.out("/nandi/demo/state", self.job.state()))
            self.get_logger().info("demo tour node ready: /nandi/demo/run {steps: full | [ids], program, seed}")

        def out(self, topic: str, payload: Any):
            pub = self.pubs.get(topic)
            if pub is not None:
                msg = String()
                msg.data = json.dumps(payload, ensure_ascii=False, default=str)
                pub.publish(msg)

        def on_run(self, msg):
            try:
                kind, kw = parse_run(msg.data)
            except ValueError as exc:
                self.out("/nandi/demo/events", {"event": "refused", "reason": str(exc)})
                return
            if kind in ("abort", "pause", "resume"):
                ok, why = self.job.control(kind)
                if not ok:
                    self.out("/nandi/demo/events", {"event": "refused", "cmd": kind, "reason": why})
                elif why == "queued":
                    self.out("/nandi/demo/events", {"event": "queued", "cmd": kind, "reason": "world loading"})
            elif kind == "run" and not self.job.start(kw):
                self.out("/nandi/demo/events", {"event": "refused", "reason": "busy"})

    from rclpy.executors import ExternalShutdownException
    rclpy.init()
    node = DemoTourNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
