"""The Atharv Nandi bringup as data (WPR; issue #104): which nodes start, in which order, and the restart policy.

Both launchers follow this table and policy:

* ``launch/nandi.launch.py`` (``ros2 launch atharv_nandi nandi.launch.py``) reads it directly;
* ``launch/launch_nandi.sh`` (the default of ``docker/start.sh``) carries the same order and the same policy in
  bash; ``tests/test_package_layout.py`` checks the two agree.

Order (stages ``STAGE_DELAY_S`` apart): the services and pilot nodes, then ``sim_runner`` and ``web_bridge``,
then the Tk console. The stack lives as long as ``sim_runner`` (the physics): when it exits, everything stops.
Every other node is supervised: a crash (non-zero exit) is logged and the node restarted after
``RESTART_DELAY_S``, up to ``NANDI_NODE_RESTARTS`` (3) times; a node that ran ``NANDI_NODE_STABLE_S`` (60 s)
before crashing gets its budget back; a clean exit (status 0, e.g. closing the Tk window) is not restarted.
A fresh stack deletes the e-stop latch file first (``NANDI_ESTOP_STATE``, else ``estop_state`` in the params
file, default ``/tmp/nandi_estop_latch.json``; bash: ``python3 -m atharv_nandi.launch_plan clear-estop <params>``);
a supervisor restart inside a running stack keeps it.

ROS-free.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

STAGE_DELAY_S = 1.0
RESTART_DELAY_S = 2.0
DEFAULT_MAX_RESTARTS = 3
DEFAULT_STABLE_S = 60.0
DEFAULT_ESTOP_STATE = "/tmp/nandi_estop_latch.json"
LOG_PREFIX = "[launch_nandi]"


@dataclass(frozen=True)
class NodeSpec:
    executable: str                       # console entry point / module in atharv_nandi.nodes
    stage: int                            # start after stage * STAGE_DELAY_S
    supervised: bool = True               # False: the stack's lifetime (sim_runner)
    env: Tuple[Tuple[str, str], ...] = ()  # extra environment for this node only


NODES: Tuple[NodeSpec, ...] = (
    NodeSpec("work_envelope_service", 0),
    NodeSpec("locomotion_service", 0),
    NodeSpec("harvest_ik_service", 0),
    NodeSpec("stability_monitor", 0),
    NodeSpec("safety_supervisor", 0),
    NodeSpec("mission_executor", 0),
    NodeSpec("perception_node", 0),
    NodeSpec("telemetry_bridge", 0),
    NodeSpec("run_recorder", 0),
    NodeSpec("sim_runner", 1, supervised=False, env=(("LIBGL_ALWAYS_SOFTWARE", "1"),)),
    NodeSpec("web_bridge", 1),
    NodeSpec("control_panel_gui", 2),
)


def node(executable: str) -> NodeSpec:
    return next(n for n in NODES if n.executable == executable)


def start_delay_s(spec: NodeSpec) -> float:
    return spec.stage * STAGE_DELAY_S


def estop_state_path(environ: Mapping[str, str] = os.environ, params_file: Optional[str] = None) -> str:
    """The latch file the safety_supervisor will use: a set ``NANDI_ESTOP_STATE`` wins, else ``estop_state``
    from ``params_file`` (as node_params resolves it for the supervisor), else the default."""
    if "NANDI_ESTOP_STATE" in environ:
        return environ["NANDI_ESTOP_STATE"] or DEFAULT_ESTOP_STATE
    if params_file:
        from atharv_nandi import node_params
        vals = node_params.file_values("nandi_safety_supervisor", ["-", "--ros-args", "--params-file", params_file])
        v = node_params.env_string(vals.get("estop_state"), "estop_state")
        if v:
            return v
    return DEFAULT_ESTOP_STATE


def clear_estop_latch(environ: Mapping[str, str] = os.environ, params_file: Optional[str] = None) -> Optional[str]:
    """A fresh stack starts un-latched: delete the latch file. Returns the path removed (None if absent)."""
    path = estop_state_path(environ, params_file)
    try:
        os.remove(path)
        return path
    except FileNotFoundError:
        return None


def _int_env(environ: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(environ.get(name, "") or default)
    except ValueError:
        return default


def _float_env(environ: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class RestartPolicy:
    """Restart decisions of one supervised node (the bash ``supervise`` loop as an object)."""
    name: str
    max_restarts: int = DEFAULT_MAX_RESTARTS
    stable_s: float = DEFAULT_STABLE_S
    restarts: int = 0
    log: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, name: str, environ: Mapping[str, str] = os.environ) -> "RestartPolicy":
        return cls(name, _int_env(environ, "NANDI_NODE_RESTARTS", DEFAULT_MAX_RESTARTS),
                   _float_env(environ, "NANDI_NODE_STABLE_S", DEFAULT_STABLE_S))

    def on_exit(self, returncode: Optional[int], ran_s: float) -> str:
        """"clean" | "restart" | "give_up" for an exit after ``ran_s`` seconds; the log line is appended."""
        rc = 0 if returncode is None else int(returncode)
        if rc == 0:
            self.log.append(f"{LOG_PREFIX} {self.name} exited cleanly; not restarting")
            return "clean"
        if ran_s >= self.stable_s:
            self.restarts = 0
        if self.restarts >= self.max_restarts:
            self.log.append(f"{LOG_PREFIX} {self.name} exited with status {rc}; giving up after "
                            f"{self.max_restarts} restarts")
            return "give_up"
        self.restarts += 1
        self.log.append(f"{LOG_PREFIX} {self.name} exited with status {rc}; restart {self.restarts}/"
                        f"{self.max_restarts} in {RESTART_DELAY_S:g} s")
        return "restart"


def policies(environ: Mapping[str, str] = os.environ) -> Dict[str, RestartPolicy]:
    return {n.executable: RestartPolicy.from_env(n.executable, environ) for n in NODES if n.supervised}


def proc_label(cmd: str) -> str:
    """A short label for a process command line (CPU tables in tools/spot_tour_bench.py, tools/spike_demo_budget.py):
    ``<node>.py`` for a node run as ``python3 -m atharv_nandi.nodes <node>`` / ``-m atharv_nandi.nodes.<node>``
    (bash launcher, nandi.launch.py from source) or as the installed ``.../lib/atharv_nandi/<node>`` (ros2 launch);
    else the first ``*.py`` / ``Xvfb*`` basename; else the first 40 characters."""
    toks = cmd.split()
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t == "-m" and nxt == "atharv_nandi.nodes" and i + 2 < len(toks):
            return toks[i + 2] + ".py"
        if t == "-m" and nxt.startswith("atharv_nandi.nodes."):
            return nxt.rsplit(".", 1)[1] + ".py"
        if "/lib/atharv_nandi/" in t:
            return os.path.basename(t) + ".py"
    return next((os.path.basename(t) for t in toks if t.endswith(".py") or t.startswith("Xvfb")), cmd[:40])


if __name__ == "__main__":   # launch/launch_nandi.sh: python3 -m atharv_nandi.launch_plan clear-estop <params file>
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "clear-estop":
        removed = clear_estop_latch(os.environ, sys.argv[2] if len(sys.argv) > 2 else None)
        if removed:
            print(f"{LOG_PREFIX} cleared e-stop latch {removed}", file=sys.stderr)
    else:
        sys.exit("usage: python3 -m atharv_nandi.launch_plan clear-estop [params file]")
