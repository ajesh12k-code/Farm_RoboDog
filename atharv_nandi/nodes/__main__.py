"""``python3 -m atharv_nandi.nodes <node> [--ros-args ...]`` (see ``entry.py``)."""
import sys

from atharv_nandi.nodes.entry import cli

sys.exit(cli())
