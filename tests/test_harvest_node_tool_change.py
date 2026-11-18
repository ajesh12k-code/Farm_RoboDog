"""WP8 harvest node ``tool_change`` mode (unit tier, ROS-free): rclpy / std_msgs / std_srvs are stubbed, the node
class runs on a compiled farm model. Covers the review findings: an auto-harvest is refused while a WP2 rack swap
owns the arm, and the swap forwarding never touches the harvest worker's IK solver (shared MjData)."""
import importlib
import json
import sys
import types

import pytest


class _Clock:
    def __init__(self):
        self.t = 100.0

    def now(self):
        return types.SimpleNamespace(nanoseconds=int(self.t * 1e9))


class _Node:
    def __init__(self, name):
        self._clock = _Clock()
        self.published = []

    def create_subscription(self, *a, **k):
        return None

    def create_publisher(self, _t, topic, _q):
        return types.SimpleNamespace(publish=lambda msg, topic=topic: self.published.append((topic, msg)))

    def create_service(self, *a, **k):
        return None

    def create_timer(self, *a, **k):
        return None

    def get_logger(self):
        return types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None)

    def get_clock(self):
        return self._clock


class _Msg:
    def __init__(self, data=None):
        self.data = data


@pytest.fixture
def node(monkeypatch):
    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _Node
    rclpy.node = rclpy_node
    std_msgs, std_msgs_msg = types.ModuleType("std_msgs"), types.ModuleType("std_msgs.msg")
    std_msgs_msg.Float64MultiArray = std_msgs_msg.String = _Msg
    std_srvs, std_srvs_srv = types.ModuleType("std_srvs"), types.ModuleType("std_srvs.srv")
    std_srvs_srv.Trigger = object
    for name, mod in (("rclpy", rclpy), ("rclpy.node", rclpy_node), ("std_msgs", std_msgs),
                      ("std_msgs.msg", std_msgs_msg), ("std_srvs", std_srvs), ("std_srvs.srv", std_srvs_srv)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "atharv_nandi.nodes.harvest_ik_service", raising=False)
    svc = importlib.import_module("atharv_nandi.nodes.harvest_ik_service")
    n = svc.HarvestIKServiceNode()
    n.have_qpos = True
    yield n
    sys.modules.pop("atharv_nandi.nodes.harvest_ik_service", None)


def _swap_msg(q5=(0.0, 0.3, -0.6, 0.2, 0.0)):
    return _Msg(json.dumps({"q5": list(q5), "payload_kg": 0.4, "phase": "dock"}))


def test_auto_harvest_is_refused_while_a_rack_swap_owns_the_arm(node):
    node.on_tool_arm(_swap_msg())
    assert node.mode == "tool_change"
    resp = node.on_auto_harvest(None, types.SimpleNamespace(success=None, message=""))
    assert resp.success is False and "Tool change" in resp.message
    assert node.mode == "tool_change" and node.worker is None        # no harvest thread took the arm mid-dock


def test_swap_forwarding_uses_its_own_ik_solver(node):
    assert node.tool_ik is not node.ik and node.tool_ik.data is not node.ik.data

    def boom(*a, **k):
        raise AssertionError("the swap forwarding touched the harvest worker's solver")
    node.ik.gravity_offsets = boom
    node.ik._fk = boom
    node.on_tool_arm(_swap_msg())
    assert node.mode == "tool_change" and len(node.arm_goal) == 7
    # a manual override still wins over the swap; release returns to RL balance
    node.on_manual(_Msg([0.0] * 7))
    assert node.mode == "manual"
    node.on_tool_arm(_swap_msg())
    assert node.mode == "manual"
