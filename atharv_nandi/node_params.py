"""Declared ROS parameters for every Atharv Nandi node, loaded from ``config/nandi.yaml`` (WPR; issue #103).

Each parameter mirrors one ``NANDI_*`` environment variable, which stays the override:

1. an env var that is **set** (even to an empty string) wins, exactly as before;
2. else a non-empty value from the params file(s) / ``-p name:=value`` on the command line is exported into
   the process environment under that env var, so the code that reads ``os.environ`` sees it;
3. else nothing is set and the code default applies (an empty YAML value means "unset").

``bootstrap`` runs before the node module is imported (``atharv_nandi.nodes.entry``), because some modules read
their env vars at import time. ``declare`` then declares the same names on the rclpy node so ``ros2 param list``
/ ``ros2 param get`` show the effective values.

Secrets are never parameters: ``NANDI_OPERATOR_TOKEN``, ``NANDI_VIEWER_TOKEN``, ``NANDI_MQTT_USERNAME`` and
``NANDI_MQTT_PASSWORD`` stay env-only with no default (project rule "a secret never has a default"). So does
``NANDI_NOVNC_UP``, the privacy lock-out decision docker/start.sh exports (``ENV_ONLY``). A YAML mapping or list
for a JSON env var (``wind``, ``weather``, ``gnss``, ``link``) is exported as JSON.

Pure Python apart from ``declare`` (which takes the rclpy node); ``bootstrap`` and the parsers are ROS-free.
"""
import json
import os
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

# Parameters every node declares: the world, the robot build and the shared files. They must agree across
# processes (sim_runner, web_bridge, perception, ... all compose the same world), so they sit under ``/**``.
SHARED: Dict[str, str] = {
    "world": "NANDI_WORLD",
    "seed": "NANDI_SEED",
    "crops": "NANDI_CROPS",
    "terrain": "NANDI_TERRAIN",
    "bund_h": "NANDI_BUND_H",
    "dt": "NANDI_DT",
    "actors": "NANDI_ACTORS",
    "season": "NANDI_SEASON",
    "trees_config": "NANDI_TREES_CONFIG",
    "crops_config": "NANDI_CROPS_CONFIG",
    "tools": "NANDI_TOOLS",
    "tool_mount": "NANDI_TOOL_MOUNT",
    "cut": "NANDI_CUT",
    "mast_h": "NANDI_MAST_H",
    "wind": "NANDI_WIND",
    "weather": "NANDI_WEATHER",
    "gnss": "NANDI_GNSS",
    "link": "NANDI_LINK",
    "imu_noise": "NANDI_IMU_NOISE",
    "energy_card": "NANDI_ENERGY_CARD",
    "recovery": "NANDI_RECOVERY",
    "dock_charge_x": "NANDI_DOCK_CHARGE_X",
    "hal_backend": "NANDI_HAL_BACKEND",
    "hal_config": "NANDI_HAL_CONFIG",
    "site_file": "NANDI_SITE_FILE",
    "safety_config": "NANDI_SAFETY_CONFIG",
    "perception_orchard": "NANDI_PERCEPTION_ORCHARD",
    "beacon_config": "NANDI_BEACON_CONFIG",
    "runs_dir": "NANDI_RUNS_DIR",
    "estop_state": "NANDI_ESTOP_STATE",
    "field_mode": "NANDI_FIELD_MODE",
    "novnc": "NANDI_NOVNC",
}

# Node name (as the node calls ``Node.__init__``) -> its own parameters on top of SHARED.
NODE_PARAMS: Dict[str, Dict[str, str]] = {
    "nandi_sim_runner": {"headless": "NANDI_HEADLESS"},
    "nandi_web_bridge": {
        "web_host": "NANDI_WEB_HOST",
        "web_port": "NANDI_WEB_PORT",
        "web_allowed_hosts": "NANDI_WEB_ALLOWED_HOSTS",
        "missions_dir": "NANDI_MISSIONS_DIR",
        "audit_log": "NANDI_AUDIT_LOG",
        "sim_local_link": "NANDI_SIM_LOCAL_LINK",
    },
    "nandi_locomotion_service": {},
    "nandi_harvest_ik_service": {},
    "nandi_stability_monitor": {"payload_kg": "NANDI_PAYLOAD_KG", "mast_head_kg": "NANDI_MAST_HEAD_KG"},
    "nandi_safety_supervisor": {},
    "nandi_mission_executor": {},
    "nandi_perception": {},
    "nandi_telemetry_bridge": {"telemetry_config": "NANDI_TELEMETRY_CONFIG", "mqtt_url": "NANDI_MQTT_URL"},
    "nandi_run_recorder": {"telemetry_config": "NANDI_TELEMETRY_CONFIG", "mqtt_url": "NANDI_MQTT_URL"},
    "work_envelope_service": {},
    "nandi_control_panel": {},
    "nandi_demo_tour": {},
}

SECRET_ENV = ("NANDI_OPERATOR_TOKEN", "NANDI_VIEWER_TOKEN", "NANDI_MQTT_USERNAME", "NANDI_MQTT_PASSWORD")
# Env-only on purpose (never a parameter): ``NANDI_NOVNC_UP`` is the privacy lock-out decision docker/start.sh
# exports (actors.novnc_up); a params file must not be able to claim "noVNC is down" and let person actors in.
ENV_ONLY = SECRET_ENV + ("NANDI_NOVNC_UP",)
# Parameters whose env var is parsed as JSON (sim/hooks/platform.py::_env_json): a YAML mapping / list is
# exported as JSON, not as a Python repr or a comma list.
JSON_PARAMS = frozenset(("wind", "weather", "gnss", "link"))
WILDCARDS = ("/**", "**")


def spec(node_name: str) -> Dict[str, str]:
    """param -> env var for ``node_name`` (SHARED plus its own). KeyError for an unknown node."""
    out = dict(SHARED)
    out.update(NODE_PARAMS[node_name])
    return out


def env_string(value, param: Optional[str] = None) -> str:
    """A YAML / ``-p`` value as the env var string the code expects (bools as 1 / 0, None as unset).

    A mapping, a list holding a mapping / list, or any list / mapping of a JSON parameter (``JSON_PARAMS``) is
    exported as JSON; a flat list otherwise as a comma list (``crops: [mango, guava]`` -> ``mango,guava``)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (list, tuple)):
        if param in JSON_PARAMS or any(isinstance(v, (Mapping, list, tuple)) for v in value):
            return json.dumps(list(value))
        return ",".join(env_string(v) for v in value)
    return str(value)


def _section(doc: Mapping, node_name: str) -> Dict[str, object]:
    """Merged ``ros__parameters`` for ``node_name`` from one params-file document: wildcard first, then exact."""
    out: Dict[str, object] = {}
    if not isinstance(doc, Mapping):
        return out
    keys = list(WILDCARDS) + [node_name, "/" + node_name, "/**/" + node_name]
    for k in keys:
        body = doc.get(k)
        if isinstance(body, Mapping) and isinstance(body.get("ros__parameters"), Mapping):
            out.update(body["ros__parameters"])
    return out


def load_params_file(path: str, node_name: str) -> Dict[str, object]:
    """The parameters a ROS 2 params file gives ``node_name`` (``/**`` wildcard, then the node's own section)."""
    with open(path, encoding="utf-8") as fh:
        return _section(yaml.safe_load(fh) or {}, node_name)


def parse_ros_args(argv: Sequence[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """(params files, [(name, value)] from ``-p name:=value``) inside every ``--ros-args ... [--]`` block."""
    files: List[str] = []
    overrides: List[Tuple[str, str]] = []
    in_ros = False
    i = 0
    args = list(argv)
    while i < len(args):
        a = args[i]
        if a == "--ros-args":
            in_ros = True
        elif a == "--" and in_ros:
            in_ros = False
        elif in_ros and a == "--params-file" and i + 1 < len(args):
            files.append(args[i + 1])
            i += 1
        elif in_ros and a in ("-p", "--param") and i + 1 < len(args):
            name, sep, value = args[i + 1].partition(":=")
            if sep:
                overrides.append((name, value))
            i += 1
        i += 1
    return files, overrides


def file_values(node_name: str, argv: Sequence[str]) -> Dict[str, object]:
    """Values for ``node_name`` from the params files and ``-p`` overrides in ``argv`` (later wins)."""
    files, overrides = parse_ros_args(argv)
    vals: Dict[str, object] = {}
    for f in files:
        vals.update(load_params_file(f, node_name))
    for name, value in overrides:
        try:
            vals[name] = yaml.safe_load(value)
        except yaml.YAMLError:
            vals[name] = value
    return vals


def resolve(node_name: str, values: Mapping[str, object], environ: Mapping[str, str]) -> Dict[str, Tuple[str, str]]:
    """param -> (effective string value, source) with source "env" | "params" | "default" (default = unset)."""
    out: Dict[str, Tuple[str, str]] = {}
    for param, var in spec(node_name).items():
        if var in environ:
            out[param] = (environ[var], "env")
        else:
            v = env_string(values.get(param), param)
            out[param] = (v, "params") if v != "" else ("", "default")
    return out


def bootstrap(node_name: str, argv: Optional[Sequence[str]] = None,
              environ: Optional[MutableMapping[str, str]] = None) -> Dict[str, Tuple[str, str]]:
    """Export the params-file values of ``node_name`` into ``environ`` for every env var that is not set.

    Call before importing the node module. Returns ``resolve``'s table."""
    import sys
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    table = resolve(node_name, file_values(node_name, argv), environ)
    sp = spec(node_name)
    for param, (value, source) in table.items():
        if source == "params":
            environ[sp[param]] = value
    return table


def declare(node, node_name: str, environ: Optional[Mapping[str, str]] = None) -> List[str]:
    """Declare every parameter of ``node_name`` on the rclpy ``node`` with its effective value (env wins).

    String parameters with dynamic typing (a YAML int or bool is accepted). The params-file value rclpy applies
    on declare is replaced by the env value when the env var is set. Returns the declared names (none for an
    object without ``declare_parameter``, such as a test stand-in)."""
    if not callable(getattr(node, "declare_parameter", None)):
        return []
    from rcl_interfaces.msg import ParameterDescriptor

    environ = os.environ if environ is None else environ
    has = getattr(node, "has_parameter", None)
    declared = []
    for param, var in spec(node_name).items():
        if callable(has) and has(param):
            continue
        desc = ParameterDescriptor(description=f"mirrors env {var} (a set env var wins; empty = code default)",
                                   dynamic_typing=True)
        node.declare_parameter(param, environ.get(var, ""), desc)
        get = getattr(node, "get_parameter", None)
        if var in environ and callable(get) and env_string(get(param).value, param) != environ[var]:
            from rclpy.parameter import Parameter
            node.set_parameters([Parameter(param, Parameter.Type.STRING, environ[var])])
        declared.append(param)
    return declared


def all_env_vars() -> Iterable[str]:
    """Every env var some node maps to a parameter."""
    seen = set(SHARED.values())
    for p in NODE_PARAMS.values():
        seen |= set(p.values())
    return sorted(seen)
