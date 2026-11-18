"""WPR: the repo is one ament_python package (``atharv_nandi``) with real imports, entry points, launch files and
declared parameters from ``config/nandi.yaml``. ROS-free (no rclpy, no colcon): the colcon build and the
``ros2 launch`` smoke run through ``tools/ros_launch_smoke.sh`` in the image.
"""
import ast
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

from atharv_nandi import launch_plan as lp
from atharv_nandi import node_params as npar
from atharv_nandi import paths
from atharv_nandi.nodes import entry

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(REPO, "atharv_nandi")
BASH_LAUNCHER = os.path.join(REPO, "launch", "launch_nandi.sh")
ROS_LAUNCH = os.path.join(REPO, "launch", "nandi.launch.py")
PARAMS_YAML = os.path.join(REPO, "config", "nandi.yaml")
CODE_DIRS = ("atharv_nandi", "tools", "tests", "launch")
# NANDI_* vars that are deliberately not parameters: secrets (no default, env only), launcher knobs, run ids.
NOT_PARAMS = set(npar.ENV_ONLY) | {"NANDI_RUN_ID", "NANDI_GIT_SHA", "NANDI_GIT_DIRTY", "NANDI_NODE_RESTARTS",
                                     "NANDI_NODE_STABLE_S", "NANDI_PARAMS_FILE", "NANDI_LAUNCH"}


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def py_files(*dirs):
    for d in dirs:
        for dirpath, dirnames, files in os.walk(os.path.join(REPO, d)):
            dirnames[:] = [x for x in dirnames if x not in ("__pycache__", "vendor")]
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(dirpath, f)


def node_modules():
    return sorted(f[:-3] for f in os.listdir(os.path.join(PKG, "nodes"))
                  if f.endswith(".py") and f not in ("__init__.py", "__main__.py", "entry.py"))


# ---------------------------------------------------------------------- layout and imports
def test_no_sys_path_edits_anywhere():
    bad = []
    for p in py_files(*CODE_DIRS, "rl_training"):
        if os.path.basename(p) == os.path.basename(__file__):
            continue
        for i, line in enumerate(read(p).splitlines(), 1):
            if re.search(r"\bsys\.path\s*(\.insert|\.append|\[|\+?=)", line):
                bad.append(f"{os.path.relpath(p, REPO)}:{i}: {line.strip()}")
    assert not bad, "sys.path edits:\n" + "\n".join(bad)


def test_no_old_top_level_imports():
    old = "controllers|world|hooks|workflow|" + "|".join(node_modules())
    rx = re.compile(r"^\s*(from|import) (%s)(\.|\s|$)" % old)
    bad = [f"{os.path.relpath(p, REPO)}:{i}" for p in py_files(*CODE_DIRS)
           for i, line in enumerate(read(p).splitlines(), 1) if rx.match(line)]
    assert not bad, bad


def test_package_dirs_are_regular_packages():
    for d in ("", "core", "core/workflow", "core/hal", "core/perception", "core/planning", "core/tasks",
              "core/tools", "nodes", "sim", "sim/world", "sim/world/modules", "sim/hooks"):
        assert os.path.isfile(os.path.join(PKG, d, "__init__.py")), d
    assert not os.path.isdir(os.path.join(REPO, "sim", "controllers"))
    assert os.listdir(os.path.join(REPO, "sim")) == ["launch_nandi.sh"]     # old-image compatibility shim only


def test_paths_resolve_in_the_source_tree():
    assert paths.REPO == REPO and paths.CONFIG_DIR == os.path.join(REPO, "config")
    assert os.path.isfile(os.path.join(paths.ROBOT_MODEL_DIR, "atharv_nandi.xml"))
    assert os.path.isfile(os.path.join(paths.WEB_DIR, "index.html"))
    assert paths.config_path("nandi.yaml") == PARAMS_YAML


def test_paths_fall_back_to_the_ament_share_dir(tmp_path):
    # an installed copy has no config/ next to the package: the share dir is asked (here: not installed)
    r = paths.data_root(str(tmp_path))
    assert r == str(tmp_path) or os.path.isdir(os.path.join(r, "config"))
    (tmp_path / "config").mkdir()
    assert paths.data_root(str(tmp_path)) == str(tmp_path)


# ---------------------------------------------------------------------- ament package metadata
def test_package_xml_and_setup_agree():
    root = ET.parse(os.path.join(REPO, "package.xml")).getroot()
    assert root.get("format") == "3" and root.findtext("name") == paths.PACKAGE_NAME
    assert root.find("export").findtext("build_type") == "ament_python"
    deps = {e.text for e in root.findall("exec_depend")}
    assert {"rclpy", "std_msgs", "std_srvs", "sensor_msgs", "geometry_msgs", "launch", "launch_ros"} <= deps
    assert os.path.isfile(os.path.join(REPO, "resource", paths.PACKAGE_NAME))
    cfg = read(os.path.join(REPO, "setup.cfg"))
    assert "$base/lib/atharv_nandi" in cfg
    setup = ast.parse(read(os.path.join(REPO, "setup.py")))
    nodes = next(ast.literal_eval(n.value) for n in ast.walk(setup)
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "NODES")
    assert sorted(nodes) == sorted(entry.NODES) == node_modules()


def test_every_node_has_an_entry_point_and_its_node_name():
    for exe, node_name in entry.NODES.items():
        assert callable(getattr(entry, exe)), exe
        src = read(os.path.join(PKG, "nodes", f"{exe}.py"))
        if exe == "control_panel_gui":
            src = read(os.path.join(PKG, "core", "ros_client.py"))
        if exe == "telemetry_bridge":
            assert f'name: str = "{node_name}"' in src
        elif exe == "run_recorder":
            assert f'name="{node_name}"' in src
        else:
            assert f'super().__init__("{node_name}")' in src, exe
        assert "def main(" in read(os.path.join(PKG, "nodes", f"{exe}.py")), exe


def test_every_node_declares_its_parameters():
    for exe, node_name in entry.NODES.items():
        path = os.path.join(PKG, "core", "ros_client.py") if exe == "control_panel_gui" else \
            os.path.join(PKG, "nodes", f"{exe}.py")
        if exe == "run_recorder":       # the telemetry BridgeNode, named nandi_run_recorder
            path = os.path.join(PKG, "nodes", "telemetry_bridge.py")
        src = read(path)
        assert "node_params.declare(self, " in src, exe
    assert set(npar.NODE_PARAMS) == set(entry.NODES.values())


def test_entry_cli_usage_and_argv(monkeypatch):
    assert entry.cli(["x"]) == 2 and entry.cli(["x", "nope"]) == 2
    seen = {}

    def fake_run(exe, argv):
        seen["exe"], seen["argv"] = exe, list(argv)
        return None

    monkeypatch.setattr(entry, "run", fake_run)
    import sys
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    assert entry.cli(["m", "web_bridge", "--ros-args", "-p", "web_port:=9"]) == 0
    assert seen == {"exe": "web_bridge", "argv": ["m", "--ros-args", "-p", "web_port:=9"]}


# ---------------------------------------------------------------------- launch: both launchers agree
def bash_plan():
    """[(executable, stage, supervised)] in the order launch_nandi.sh starts them; `sleep 1` ends a stage."""
    out, stage = [], 0
    for line in read(BASH_LAUNCHER).splitlines():
        s = line.strip()
        if s == "sleep 1":
            stage += 1
        m = re.match(r"^supervise (\w+)$", s)
        if m:
            out.append((m.group(1), stage, True))
        m = re.match(r"^(?:\w+=\S+ )*run_node (\w+) &$", s)
        if m:
            out.append((m.group(1), stage, False))
    return out


def test_bash_launcher_matches_the_launch_plan():
    assert bash_plan() == [(n.executable, n.stage, n.supervised) for n in lp.NODES]
    src = read(BASH_LAUNCHER)
    assert 'python3 -m atharv_nandi.launch_plan clear-estop "$PARAMS"' in src
    assert 'MAX_RESTARTS="${NANDI_NODE_RESTARTS:-%d}"' % lp.DEFAULT_MAX_RESTARTS in src
    assert 'STABLE_S="${NANDI_NODE_STABLE_S:-%d}"' % lp.DEFAULT_STABLE_S in src
    assert "--params-file" in src and "config/nandi.yaml" in src
    assert "LIBGL_ALWAYS_SOFTWARE=1 run_node sim_runner &" in src and 'wait "$SIM_PID"' in src


def test_launch_plan_covers_the_live_nodes():
    assert {n.executable for n in lp.NODES} == set(entry.NODES) - {"demo_tour_node"}  # demo tour: started by hand
    assert [n.executable for n in lp.NODES if not n.supervised] == ["sim_runner"]
    assert lp.start_delay_s(lp.node("sim_runner")) == lp.STAGE_DELAY_S
    assert lp.start_delay_s(lp.node("control_panel_gui")) == 2 * lp.STAGE_DELAY_S


def test_ros_launch_file_uses_the_plan():
    src = read(ROS_LAUNCH)
    tree = ast.parse(src)
    assert any(isinstance(n, ast.FunctionDef) and n.name == "generate_launch_description" for n in tree.body)
    for needle in ("lp.NODES", "lp.policies()", "lp.clear_estop_latch(os.environ, params_file)", "Shutdown(", "OnProcessExit",
                   'paths.config_path("nandi.yaml")', "lp.RESTART_DELAY_S", "OnShutdown"):
        assert needle in src, needle


def test_restart_policy_matches_the_bash_supervisor():
    p = lp.RestartPolicy("web_bridge")
    assert [p.on_exit(143, 1.0) for _ in range(4)] == ["restart", "restart", "restart", "give_up"]
    assert p.log == ["[launch_nandi] web_bridge exited with status 143; restart 1/3 in 2 s",
                     "[launch_nandi] web_bridge exited with status 143; restart 2/3 in 2 s",
                     "[launch_nandi] web_bridge exited with status 143; restart 3/3 in 2 s",
                     "[launch_nandi] web_bridge exited with status 143; giving up after 3 restarts"]
    # the same wording as the bash loop
    src = read(BASH_LAUNCHER)
    assert 'log "$name exited with status $rc; restart $restarts/$MAX_RESTARTS in 2 s"' in src
    assert 'log "$name exited with status $rc; giving up after $MAX_RESTARTS restarts"' in src
    assert 'log "$name exited cleanly; not restarting"' in src


def test_restart_policy_clean_exit_and_stable_reset():
    p = lp.RestartPolicy("x", max_restarts=1, stable_s=60.0)
    assert p.on_exit(0, 1.0) == "clean" and p.log[-1] == "[launch_nandi] x exited cleanly; not restarting"
    assert p.on_exit(None, 1.0) == "clean"
    assert p.on_exit(1, 1.0) == "restart" and p.on_exit(1, 1.0) == "give_up"
    q = lp.RestartPolicy("y", max_restarts=1, stable_s=60.0)
    assert q.on_exit(1, 1.0) == "restart" and q.on_exit(1, 61.0) == "restart"      # ran 60 s: budget back


def test_restart_policy_from_env():
    p = lp.RestartPolicy.from_env("x", {"NANDI_NODE_RESTARTS": "5", "NANDI_NODE_STABLE_S": "10"})
    assert (p.max_restarts, p.stable_s) == (5, 10.0)
    p = lp.RestartPolicy.from_env("x", {"NANDI_NODE_RESTARTS": "bad"})
    assert (p.max_restarts, p.stable_s) == (3, 60.0)
    assert set(lp.policies({})) == {n.executable for n in lp.NODES} - {"sim_runner"}


def test_fresh_launch_clears_the_estop_latch(tmp_path):
    f = tmp_path / "latch.json"
    f.write_text("{}")
    env = {"NANDI_ESTOP_STATE": str(f)}
    assert lp.estop_state_path(env) == str(f) and lp.estop_state_path({}) == lp.DEFAULT_ESTOP_STATE
    assert lp.clear_estop_latch(env) == str(f) and not f.exists()
    assert lp.clear_estop_latch(env) is None


def test_fresh_launch_clears_the_latch_named_in_the_params_file(tmp_path):
    """Review: estop_state set only in the params YAML must be the file both launchers delete."""
    f = tmp_path / "yaml_latch.json"
    f.write_text("{}")
    pf = tmp_path / "p.yaml"
    pf.write_text("/**:\n  ros__parameters:\n    estop_state: %s\n" % f)
    assert lp.estop_state_path({}, str(pf)) == str(f)
    assert lp.estop_state_path({"NANDI_ESTOP_STATE": "/x"}, str(pf)) == "/x"          # a set env var wins
    assert lp.estop_state_path({}, PARAMS_YAML) == lp.DEFAULT_ESTOP_STATE           # shipped file: default
    env = {k: v for k, v in os.environ.items() if k != "NANDI_ESTOP_STATE"}
    env["PYTHONPATH"] = REPO
    r = subprocess.run([sys.executable, "-m", "atharv_nandi.launch_plan", "clear-estop", str(pf)], env=env,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and not f.exists(), r.stderr


# ---------------------------------------------------------------------- parameters
def test_params_yaml_lists_every_declared_parameter_and_changes_nothing():
    doc = yaml.safe_load(read(PARAMS_YAML))
    assert set(doc["/**"]["ros__parameters"]) == set(npar.SHARED)
    for node_name, own in npar.NODE_PARAMS.items():
        if own:
            assert set(doc[node_name]["ros__parameters"]) == set(own), node_name
    for sec in doc.values():
        # empty = the code default: shipping this file changes no behaviour against the env-only stack
        assert all(v == "" for v in sec["ros__parameters"].values())
    for sec in doc:
        assert sec == "/**" or sec in npar.NODE_PARAMS, sec


def test_no_secret_is_a_parameter():
    for node_name in npar.NODE_PARAMS:
        assert not set(npar.spec(node_name).values()) & set(npar.SECRET_ENV)
    assert not re.search(r"TOKEN|PASSWORD|USERNAME", read(PARAMS_YAML).split("\n/**:")[1])


def test_every_nandi_env_var_in_the_package_is_a_parameter_or_listed():
    used = set()
    for p in py_files("atharv_nandi"):
        used |= set(re.findall(r"[\"'](NANDI_[A-Z0-9_]+)[\"']", read(p)))
    missing = sorted(used - set(npar.all_env_vars()) - NOT_PARAMS)
    assert not missing, f"NANDI_* env vars without a parameter in node_params.py: {missing}"


def test_parse_ros_args():
    files, ov = npar.parse_ros_args(["prog", "x", "--ros-args", "--params-file", "/a.yaml", "-p", "seed:=11",
                                     "--param", "world:=orchard_lite", "-r", "__ns:=/x", "--", "-p", "no:=1"])
    assert files == ["/a.yaml"] and ov == [("seed", "11"), ("world", "orchard_lite")]
    assert npar.parse_ros_args(["prog", "-p", "seed:=3"]) == ([], [])


def test_params_file_wildcard_then_node_section(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text('/**:\n  ros__parameters:\n    world: orchard_lite\n    seed: 7\n'
                 'nandi_web_bridge:\n  ros__parameters:\n    seed: 11\n    web_port: 9090\n'
                 '/nandi_sim_runner:\n  ros__parameters:\n    headless: true\n')
    assert npar.load_params_file(str(f), "nandi_web_bridge") == {"world": "orchard_lite", "seed": 11,
                                                                 "web_port": 9090}
    assert npar.load_params_file(str(f), "nandi_sim_runner") == {"world": "orchard_lite", "seed": 7,
                                                                 "headless": True}
    vals = npar.file_values("nandi_web_bridge", ["p", "--ros-args", "--params-file", str(f), "-p", "seed:=12"])
    assert vals["seed"] == 12 and vals["web_port"] == 9090


def test_env_wins_then_params_then_default(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text('/**:\n  ros__parameters:\n    world: orchard_lite\n    seed: 11\n    terrain: ""\n'
                 'nandi_sim_runner:\n  ros__parameters:\n    headless: true\n')
    argv = ["p", "--ros-args", "--params-file", str(f)]
    env = {"NANDI_SEED": "23", "NANDI_TERRAIN": ""}
    table = npar.bootstrap("nandi_sim_runner", argv, env)
    assert table["seed"] == ("23", "env") and env["NANDI_SEED"] == "23"             # env wins
    assert table["world"] == ("orchard_lite", "params") and env["NANDI_WORLD"] == "orchard_lite"
    assert table["headless"] == ("1", "params") and env["NANDI_HEADLESS"] == "1"     # bool -> 1
    assert table["terrain"] == ("", "env") and env["NANDI_TERRAIN"] == ""            # set-but-empty stays set
    assert table["crops"] == ("", "default") and "NANDI_CROPS" not in env            # unset stays unset
    assert "NANDI_WEB_PORT" not in env                                               # not this node's param


def test_shipped_params_file_exports_nothing():
    env = {}
    npar.bootstrap("nandi_web_bridge", ["p", "--ros-args", "--params-file", PARAMS_YAML], env)
    assert env == {}


@pytest.mark.parametrize("value,expected", [(True, "1"), (False, "0"), (None, ""), (7, "7"), (0.5, "0.5"),
                                            ("x", "x"), ([1, 2], "1,2")])
def test_env_string(value, expected):
    assert npar.env_string(value) == expected


def test_env_string_exports_mappings_and_json_params_as_json():
    """Review: a YAML mapping became a Python repr, which platform._env_json (json.loads) rejects."""
    assert json.loads(npar.env_string({"speed_mps": 3})) == {"speed_mps": 3}
    assert json.loads(npar.env_string([{"t": 0}, {"t": 5}])) == [{"t": 0}, {"t": 5}]
    assert json.loads(npar.env_string([1, 2], "wind")) == [1, 2]
    assert npar.env_string(["mango", "guava"], "crops") == "mango,guava"
    for p in sorted(npar.JSON_PARAMS):
        assert p in npar.SHARED, p


def test_bootstrap_exports_json_for_p_and_params_file(tmp_path):
    env = {}
    npar.bootstrap("nandi_sim_runner", ["x", "--ros-args", "-p", 'wind:={"speed_mps": 3}'], env)
    assert json.loads(env["NANDI_WIND"]) == {"speed_mps": 3}
    f = tmp_path / "p.yaml"
    f.write_text('/**:\n  ros__parameters:\n    gnss: {"drop_s": [10, 20]}\n')
    env = {}
    npar.bootstrap("nandi_sim_runner", ["x", "--ros-args", "--params-file", str(f)], env)
    assert json.loads(env["NANDI_GNSS"]) == {"drop_s": [10, 20]}


def test_proc_label_names_nodes_in_every_launch_mode():
    """Review: tools/spike_demo_budget.py labelled `python3 -m atharv_nandi.nodes <node>` by its first 40 chars."""
    assert lp.proc_label("python3 -m atharv_nandi.nodes locomotion_service --ros-args") == "locomotion_service.py"
    assert lp.proc_label("python3 -m atharv_nandi.nodes.web_bridge") == "web_bridge.py"
    assert lp.proc_label("/ws/.colcon/install/atharv_nandi/lib/atharv_nandi/safety_supervisor --ros-args "
                         "--params-file /x.yaml") == "safety_supervisor.py"
    assert lp.proc_label("python3 /ws/tools/x.py") == "x.py" and lp.proc_label("Xvfb :99") == "Xvfb"
    src = read(os.path.join(REPO, "tools", "spike_demo_budget.py"))
    assert "proc_label(cmd)" in src


def test_novnc_up_is_env_only():
    """Review: the privacy lock-out decision docker/start.sh exports must not be settable from a params file."""
    for node_name in npar.NODE_PARAMS:
        assert not set(npar.spec(node_name).values()) & set(npar.ENV_ONLY), node_name
    env = {}
    npar.bootstrap("nandi_sim_runner", ["x", "--ros-args", "-p", "novnc_up:=0"], env)
    assert "NANDI_NOVNC_UP" not in env
    assert "novnc_up" not in yaml.safe_load(read(PARAMS_YAML))["/**"]["ros__parameters"]


class _Param:
    def __init__(self, value):
        self.value = value


class _FakeNode:
    """The rclpy surface ``declare`` uses; a params-file override is applied on declare like rclpy does."""

    def __init__(self, overrides=None):
        self.params, self.overrides, self.set_calls = {}, dict(overrides or {}), []

    def has_parameter(self, name):
        return name in self.params

    def declare_parameter(self, name, value, descriptor=None):
        self.params[name] = self.overrides.get(name, value)

    def get_parameter(self, name):
        return _Param(self.params[name])

    def set_parameters(self, ps):
        for p in ps:
            self.params[p.name] = p.value
            self.set_calls.append(p.name)


def test_declare_shows_the_effective_values():
    pytest.importorskip("rcl_interfaces.msg")
    pytest.importorskip("rclpy.parameter")
    node = _FakeNode({"seed": 11, "world": "orchard_lite"})
    env = {"NANDI_SEED": "23", "NANDI_WORLD": "orchard_lite", "NANDI_WEB_PORT": "9090"}
    names = npar.declare(node, "nandi_web_bridge", env)
    assert set(names) == set(npar.spec("nandi_web_bridge"))
    assert node.params["seed"] == "23" and node.set_calls == ["seed"]                # env wins over the file
    assert node.params["world"] == "orchard_lite" and node.params["web_port"] == "9090"
    assert node.params["crops"] == ""
    assert npar.declare(node, "nandi_web_bridge", env) == []                          # already declared
    assert npar.declare(object(), "nandi_web_bridge", env) == []                      # no rclpy surface


def test_spec_unknown_node():
    with pytest.raises(KeyError):
        npar.spec("nandi_nope")
