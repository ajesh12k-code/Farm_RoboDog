"""Navigation: pose-to-goal controller on a kinematic toy model, turn in place, cmd_vel hold."""
import math

import pytest

from atharv_nandi.core import nav_logic as nav
from atharv_nandi.core.locomotion_logic import HOLD_TIMEOUT_S


def drive(pose, goal, steps=3000, dt=0.1):
    for i in range(steps):
        c = nav.goto(pose, goal)
        if c.arrived:
            return pose, i
        assert -1.0 <= c.vx <= 1.0 and -1.0 <= c.yaw <= 1.0
        pose = nav.unicycle_step(pose, c.vx, c.yaw, dt)
    return pose, None


@pytest.mark.parametrize("start,goal", [
    ((0.0, 0.0, 0.0), (3.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 2.0)),       # 90 deg left
    ((0.0, 0.0, 0.0), (-2.0, -0.5)),     # behind: turn in place first
    ((1.0, 1.0, 2.5), (4.0, -1.0)),
])
def test_arrives_within_tolerance_on_toy_model(start, goal):
    pose, steps = drive(start, goal)
    assert steps is not None
    assert math.hypot(pose[0] - goal[0], pose[1] - goal[1]) <= nav.ARRIVE_TOL_M


def test_turns_in_place_when_goal_is_behind():
    c = nav.goto((0.0, 0.0, 0.0), (-1.0, 0.1))
    assert c.vx == 0.0 and abs(c.yaw) == 1.0
    c = nav.goto((0.0, 0.0, 0.0), (1.0, 0.3))   # small error: walk and steer left
    assert c.vx > 0.3 and c.yaw > 0


def test_heading_p_gain_and_saturation():
    c = nav.goto((0.0, 0.0, 0.0), (5.0, 5.0 * math.tan(0.2)))
    assert c.yaw == pytest.approx(nav.K_HEADING * 0.2)
    assert nav.goto((0.0, 0.0, 0.0), (0.0, -3.0)).yaw == -1.0


def test_wrap_angle_and_turn_to():
    assert nav.wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert nav.turn_to(3.0, -3.0).yaw > 0  # shortest way across +-pi is CCW
    assert nav.turn_to(0.05, 0.0).arrived


def test_cmd_vel_hold_semantics():
    h = nav.CmdVelHold()
    out = [h.tick((0.6, -0.2)) for _ in range(5)]
    assert all(o == (0.6, 0.0, -0.2, 1.0) for o in out)  # hold=1 each 0.1 s tick < 0.5 s dead-man
    assert 1.0 / nav.CMD_RATE_HZ < HOLD_TIMEOUT_S
    rel = [h.tick(None) for _ in range(5)]
    assert rel[:3] == [(0.0, 0.0, 0.0, 0.0)] * 3 and rel[3:] == [None, None]
    h.force_release()
    assert h.tick((0.0, 0.0)) == (0.0, 0.0, 0.0, 0.0)
