"""ament_python package atharv_nandi (WPR). ``colcon build`` installs:

* the Python package (``atharv_nandi.*``) with its package data: MJCF models (``sim/models/``) and the browser
  console (``web/``);
* one console script per node in ``lib/atharv_nandi/`` (``ros2 run atharv_nandi <node>``; ``nodes/entry.py``);
* ``share/atharv_nandi/``: ``package.xml``, ``launch/``, ``config/`` (the params file and every data file the
  code reads through ``atharv_nandi.paths``).
"""
import os

from setuptools import find_packages, setup

PACKAGE = "atharv_nandi"
HERE = os.path.dirname(os.path.abspath(__file__))


def tree(src: str, dest: str):
    """data_files entries for every file under ``src`` (relative to the repo root), mirrored under ``dest``."""
    out = []
    for dirpath, dirnames, files in os.walk(os.path.join(HERE, src)):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        rel = os.path.relpath(dirpath, HERE)
        fs = sorted(os.path.join(rel, f) for f in files if not f.endswith((".pyc", ".pyo")))
        if fs:
            out.append((os.path.join(dest, os.path.relpath(dirpath, os.path.join(HERE, src))).rstrip("/."), fs))
    return out


def package_data(*subdirs: str):
    """Every file under ``atharv_nandi/<subdir>`` as package data (setuptools 59 has no ``**`` globs)."""
    out = []
    root = os.path.join(HERE, PACKAGE)
    for sub in subdirs:
        for dirpath, dirnames, files in os.walk(os.path.join(root, sub)):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            out += [os.path.relpath(os.path.join(dirpath, f), root) for f in files if not f.endswith(".py")]
    return sorted(out)


def share(sub: str) -> str:
    return os.path.join("share", PACKAGE, sub)


NODES = ("sim_runner", "web_bridge", "locomotion_service", "harvest_ik_service", "stability_monitor",
         "safety_supervisor", "mission_executor", "perception_node", "telemetry_bridge", "run_recorder",
         "work_envelope_service", "control_panel_gui", "demo_tour_node")

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(include=[PACKAGE, PACKAGE + ".*"]),
    package_data={PACKAGE: package_data("sim/models", "web")},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE]),
        (share(""), ["package.xml"]),
        (share("launch"), ["launch/nandi.launch.py", "launch/launch_nandi.sh"]),
    ] + [(d if d else share("config"), fs) for d, fs in tree("config", share("config"))],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Kisaanu",
    maintainer_email="noreply@kisaanu.com",
    description="Atharv Nandi orchard robot simulator (MuJoCo + ROS 2 Humble)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [f"{n} = {PACKAGE}.nodes.entry:{n}" for n in NODES]},
)
