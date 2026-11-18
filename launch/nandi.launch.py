"""Atharv Nandi bringup for ``ros2 launch`` (WPR; issue #104), equivalent to ``launch/launch_nandi.sh``.

    ros2 launch atharv_nandi nandi.launch.py                    # after colcon build + source install/setup.bash
    ros2 launch launch/nandi.launch.py                          # straight from a source tree (PYTHONPATH=<repo>)
    ros2 launch atharv_nandi nandi.launch.py params_file:=/path/to/nandi.yaml gui:=false

Node table, stage order and restart policy come from ``atharv_nandi.launch_plan`` (the same as the bash
launcher): stages 1 s apart, ``sim_runner`` is the stack's lifetime (its exit shuts everything down), every
other node is restarted after a crash up to ``NANDI_NODE_RESTARTS`` (3) times with the budget reset after
``NANDI_NODE_STABLE_S`` (60 s) of uptime, a clean exit is not restarted. A fresh launch deletes the e-stop
latch file; a ``safety_supervisor`` restart inside the launch keeps it (the supervisor reloads it).

Each node gets ``--ros-args --params-file <params_file>`` (default ``config/nandi.yaml``): the declared
parameters; a set ``NANDI_*`` env var still wins (``atharv_nandi.node_params``). With the package installed
the nodes run as ``Node(package="atharv_nandi", executable=...)``; without it (source tree) as
``python3 -m atharv_nandi.nodes <node>``.
"""
import os
import sys
import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo, OpaqueFunction, \
    RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit, OnProcessStart, OnShutdown
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

from atharv_nandi import launch_plan as lp
from atharv_nandi import paths

GUI_NODES = ("control_panel_gui",)


def _installed() -> bool:
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = get_package_prefix(paths.PACKAGE_NAME)
    except Exception:
        return False
    return os.path.isfile(os.path.join(prefix, "lib", paths.PACKAGE_NAME, "sim_runner"))


def _action(spec: lp.NodeSpec, params_file: str, installed: bool):
    env = dict(spec.env)
    if installed:
        from launch_ros.actions import Node
        return Node(package=paths.PACKAGE_NAME, executable=spec.executable, parameters=[params_file],
                    output="screen", emulate_tty=True, additional_env=env or None)
    src = paths.SOURCE_ROOT
    pp = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + pp if pp else "")
    return ExecuteProcess(cmd=[sys.executable, "-m", "atharv_nandi.nodes", spec.executable,
                               "--ros-args", "--params-file", params_file],
                          name=spec.executable, cwd=src, output="screen", emulate_tty=True, additional_env=env)


def _setup(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower() not in ("0", "false", "no", "off")
    installed = _installed()
    removed = lp.clear_estop_latch(os.environ, params_file)
    out = [LogInfo(msg=f"{lp.LOG_PREFIX} params {params_file}; nodes "
                       f"{'installed (ros2 run)' if installed else 'from source (python3 -m)'}"
                       + (f"; cleared e-stop latch {removed}" if removed else ""))]
    pol = lp.policies()
    owner = {}                       # id(action) -> (spec, action); the action is kept alive here
    stopping = []                    # non-empty once the launch shuts down: no more restarts
    started = {}                     # id(action) -> monotonic start

    def launch_one(spec):
        act = _action(spec, params_file, installed)
        owner[id(act)] = (spec, act)
        return act

    def on_start(event, ctx):
        started[id(event.action)] = time.monotonic()

    def on_exit(event, ctx):
        spec, _ = owner.pop(id(event.action), (None, None))
        if spec is None or stopping:
            return None
        ran = time.monotonic() - started.pop(id(event.action), time.monotonic())
        if not spec.supervised:
            stopping.append(spec.executable)
            return [LogInfo(msg=f"{lp.LOG_PREFIX} {spec.executable} exited with status {event.returncode}; "
                                "stopping the stack"),
                    EmitEvent(event=Shutdown(reason=f"{spec.executable} exited"))]
        p = pol[spec.executable]
        verdict = p.on_exit(event.returncode, ran)
        msgs = [LogInfo(msg=p.log[-1])]
        if verdict == "restart":
            return msgs + [TimerAction(period=lp.RESTART_DELAY_S, actions=[launch_one(spec)])]
        return msgs

    out.append(RegisterEventHandler(OnProcessStart(on_start=on_start)))
    out.append(RegisterEventHandler(OnProcessExit(on_exit=on_exit)))
    out.append(RegisterEventHandler(OnShutdown(on_shutdown=lambda event, ctx: stopping.append("shutdown"))))
    for spec in lp.NODES:
        if spec.executable in GUI_NODES and not gui:
            continue
        act = launch_one(spec)
        delay = lp.start_delay_s(spec)
        out.append(TimerAction(period=delay, actions=[act]) if delay > 0 else act)
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=paths.config_path("nandi.yaml"),
                              description="ROS 2 params file for every node (NANDI_* env vars still win)"),
        DeclareLaunchArgument("gui", default_value="true", description="start the Tk control panel (needs X)"),
        OpaqueFunction(function=_setup),
    ])
