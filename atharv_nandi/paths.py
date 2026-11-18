"""Where Atharv Nandi finds its data (WPR): one place instead of ``__file__``-relative joins in every module.

* ``PACKAGE_DIR``: the ``atharv_nandi`` Python package. Package data that ships inside it: the MJCF models
  (``atharv_nandi/sim/models/``) and the browser console (``web/``).
* ``REPO``: the data root holding ``config/`` (and ``runs/``, ``rl_training/``, ``docs/`` in a source tree).
  In a source checkout (the dev container mounts it at ``/ws``) it is the checkout; in a ``colcon build``
  install without the source tree it is the ament share directory ``share/atharv_nandi`` (``config/`` is
  installed there by ``setup.py``).

Pure Python, no ROS import unless the source tree is absent (then ``ament_index_python`` is asked).
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_ROOT = os.path.dirname(PACKAGE_DIR)
PACKAGE_NAME = "atharv_nandi"


def data_root(source_root: str = SOURCE_ROOT) -> str:
    """The source checkout when it has ``config/``, else the installed ament share dir (fallback: the source)."""
    if os.path.isdir(os.path.join(source_root, "config")):
        return source_root
    try:
        from ament_index_python.packages import get_package_share_directory
        return get_package_share_directory(PACKAGE_NAME)
    except Exception:  # ament index missing or package not installed
        return source_root


REPO = data_root()
CONFIG_DIR = os.path.join(REPO, "config")
RUNS_DIR = os.path.join(REPO, "runs")
MODELS_DIR = os.path.join(PACKAGE_DIR, "sim", "models")
ROBOT_MODEL_DIR = os.path.join(MODELS_DIR, "AtharvNandi")
WEB_DIR = os.path.join(PACKAGE_DIR, "web")


def config_path(*parts: str) -> str:
    """``config/<parts...>`` under the data root."""
    return os.path.join(CONFIG_DIR, *parts)
