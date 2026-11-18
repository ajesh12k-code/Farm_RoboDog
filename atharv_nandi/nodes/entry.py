"""Console entry points for every node (WPR): params bootstrap, then the node module's ``main()``.

``setup.py`` maps each executable (``ros2 run atharv_nandi <node>``) to a function here; from a source tree
``python3 -m atharv_nandi.nodes <node> [--ros-args ...]`` does the same (``launch/launch_nandi.sh``).
The bootstrap (``atharv_nandi.node_params``) runs before the node module is imported, because some node
modules read their ``NANDI_*`` env vars at import time.
"""
import importlib
import sys
from typing import Dict, List, Optional, Sequence

# executable (= module in atharv_nandi.nodes) -> ROS node name it creates
NODES: Dict[str, str] = {
    "sim_runner": "nandi_sim_runner",
    "web_bridge": "nandi_web_bridge",
    "locomotion_service": "nandi_locomotion_service",
    "harvest_ik_service": "nandi_harvest_ik_service",
    "stability_monitor": "nandi_stability_monitor",
    "safety_supervisor": "nandi_safety_supervisor",
    "mission_executor": "nandi_mission_executor",
    "perception_node": "nandi_perception",
    "telemetry_bridge": "nandi_telemetry_bridge",
    "run_recorder": "nandi_run_recorder",
    "work_envelope_service": "work_envelope_service",
    "control_panel_gui": "nandi_control_panel",
    "demo_tour_node": "nandi_demo_tour",
}


def run(executable: str, argv: Optional[Sequence[str]] = None):
    """Bootstrap the params of ``executable`` from ``argv`` (default ``sys.argv``) and call its ``main()``."""
    from atharv_nandi import node_params
    node_params.bootstrap(NODES[executable], sys.argv if argv is None else argv)
    mod = importlib.import_module(f"atharv_nandi.nodes.{executable}")
    return mod.main()


def cli(argv: Optional[List[str]] = None) -> int:
    """``python3 -m atharv_nandi.nodes <executable> [args...]``: argv[1:] become the node's ``sys.argv[1:]``."""
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2 or argv[1] not in NODES:
        sys.stderr.write("usage: python3 -m atharv_nandi.nodes <node> [--ros-args ...]\nnodes: "
                         + ", ".join(sorted(NODES)) + "\n")
        return 2
    exe = argv[1]
    sys.argv = [argv[0]] + argv[2:]
    rc = run(exe, sys.argv)
    return rc if isinstance(rc, int) else 0


def _make(exe: str):
    def entry():
        return run(exe)
    entry.__name__ = exe
    entry.__doc__ = f"ros2 run atharv_nandi {exe}"
    return entry


sim_runner = _make("sim_runner")
web_bridge = _make("web_bridge")
locomotion_service = _make("locomotion_service")
harvest_ik_service = _make("harvest_ik_service")
stability_monitor = _make("stability_monitor")
safety_supervisor = _make("safety_supervisor")
mission_executor = _make("mission_executor")
perception_node = _make("perception_node")
telemetry_bridge = _make("telemetry_bridge")
run_recorder = _make("run_recorder")
work_envelope_service = _make("work_envelope_service")
control_panel_gui = _make("control_panel_gui")
demo_tour_node = _make("demo_tour_node")
