"""Harvest decision logic, sequence control flow (fake world) and a closed-loop MuJoCo harvest (ROS-free)."""
import json
import math

import mujoco
import numpy as np
import pytest

from controllers import harvest_logic as hl
from controllers.arm_ik_solver import DorsalArmIKSolver, IKResult
from controllers.farm_scene import CROP_RADIUS, crop_body_names, crop_qpos_slice, load_farm_model

NAMES = [f"crop_item_{k}" for k in range(1, 7)]


def world_qpos(trunk=(0.0, 0.0, 0.34), heading=0.0, crops=None):
    """Minimal 74-long qpos: trunk pose + crop centres (arm/legs zero; logic never reads them)."""
    q = np.zeros(32 + 7 * len(NAMES))
    q[0:3] = trunk
    q[3:7] = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
    crops = crops or {}
    for k, name in enumerate(NAMES, start=1):
        s = crop_qpos_slice(k)
        q[s] = [*crops.get(name, (10.0 + k, 0.0, CROP_RADIUS)), 1.0, 0.0, 0.0, 0.0]
    return q


# ---------------------------------------------------------------------- pure helpers
def test_quat_to_mat_matches_mujoco():
    rng = np.random.default_rng(3)
    for _ in range(20):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        ref = np.zeros(9)
        mujoco.mju_quat2Mat(ref, quat)
        assert np.allclose(hl.quat_to_mat(quat), ref.reshape(3, 3))


def test_basket_footprint_in_trunk_frame():
    assert hl.in_basket([0.0, 0.065, 0.096])
    assert not hl.in_basket([0.0, 0.065, 0.20])   # held above the basket (> trunk-frame z 0.12)
    assert not hl.in_basket([0.0, 0.065, 0.16])   # still falling in, above the wall tops (0.125)
    assert not hl.in_basket([0.1, 0.0, 0.18])     # lying on the gantry rail
    assert not hl.in_basket([0.33, 0.0, 0.09])    # past the front wall
    assert not hl.in_basket([0.0, 0.12, 0.09])    # outside the side wall
    q = world_qpos(trunk=(2.0, -1.0, 0.34), heading=1.2)
    q[crop_qpos_slice(3)] = [*hl.to_world([0.1, -0.06, 0.096], q), 1, 0, 0, 0]
    assert hl.crop_in_basket(q, "crop_item_3")
    assert hl.basket_contents(q, NAMES) == ["crop_item_3"]


def test_select_target_nearest_not_in_basket():
    crops = {"crop_item_1": (0.62, 0.0, CROP_RADIUS), "crop_item_2": (0.5, 0.6, CROP_RADIUS)}
    q = world_qpos(crops=crops)
    assert hl.select_target(q, NAMES) == ("crop_item_1", "Harvesting crop_item_1 (0.62 m)")
    # crop_item_1 now in the basket -> next nearest
    q[crop_qpos_slice(1)] = [*hl.to_world([0.0, 0.065, 0.096], q), 1, 0, 0, 0]
    name, msg = hl.select_target(q, NAMES)
    assert name == "crop_item_2" and msg.startswith("Harvesting crop_item_2")


def test_select_target_out_of_reach_message():
    q = world_qpos(crops={"crop_item_4": (1.5, 0.0, CROP_RADIUS)})
    assert hl.select_target(q, NAMES) == (None, "No crop in reach (nearest crop_item_4 1.50 m)")


def test_select_target_skips_crop_under_the_body():
    # Walked ~0.6 m forward: crop_item_1 (0.62, 0) is now under the trunk, crop_item_2 is clear of it.
    crops = {"crop_item_1": (0.62, 0.0, CROP_RADIUS), "crop_item_2": (1.3, 0.5, CROP_RADIUS)}
    q = world_qpos(trunk=(0.6, 0.0, 0.34), crops=crops)
    assert hl.crop_under_body(q, "crop_item_1") and not hl.crop_under_body(q, "crop_item_2")
    name, msg = hl.select_target(q, NAMES)
    assert name == "crop_item_2" and msg.startswith("Harvesting crop_item_2")
    # Nothing else in reach: tell the operator to back up instead of driving the arm into the body.
    q = world_qpos(trunk=(0.6, 0.0, 0.34), crops={"crop_item_1": (0.62, 0.0, CROP_RADIUS)})
    assert hl.select_target(q, NAMES) == (
        None, "No crop in reach (nearest crop_item_1 0.02 m, under the body - back up)")
    # A crop beside the body (trunk-frame |y| >= 0.175) is still a target.
    q = world_qpos(trunk=(0.62, 0.0, 0.34), crops={"crop_item_1": (0.62, 0.2, CROP_RADIUS)})
    assert hl.select_target(q, NAMES)[0] == "crop_item_1"


def test_world_reset_detected_only_on_impossible_jumps():
    a = world_qpos(trunk=(0.8, 0.1, 0.33), crops={"crop_item_1": (0.62, 0.0, CROP_RADIUS)})
    a[7:12] = [0.26, 1.2, 1.8, -0.4, 0.9]          # arm out on a grasp
    b = a.copy()
    b[0] += 0.006                                  # 0.3 m/s walking, one 20 ms sample later
    b[7:12] += [0.01, 0.06, 0.06, -0.06, 0.06]     # slew-limited arm motion
    b[crop_qpos_slice(1)][2] += 0.04               # crop falling at 2 m/s
    assert not hl.world_reset_detected(a, b)

    home = world_qpos(crops={"crop_item_1": (0.62, 0.0, CROP_RADIUS)})
    home[7:12] = hl.REST_ARM[:5]
    assert hl.world_reset_detected(a, home)        # trunk + arm jump back to the keyframe

    still = home.copy()                            # robot never moved, arm out, crop in the basket
    still[7:12] = [0.0, 0.0, 0.9, -1.8, 1.4]
    assert hl.world_reset_detected(still, home)    # arm (shoulder 0.8 rad) snapped back
    held = home.copy()
    held[crop_qpos_slice(1)] = [*hl.to_world([0.0, 0.065, 0.096], held), 1, 0, 0, 0]
    assert hl.world_reset_detected(held, home)     # crop teleported from the basket to the field
    assert not hl.world_reset_detected(home, home.copy())
    assert not hl.world_reset_detected(home, home[:40])  # mismatched sizes are ignored


def test_select_target_all_in_basket():
    q = world_qpos()
    for k, name in enumerate(NAMES, start=1):
        q[crop_qpos_slice(k)] = [*hl.to_world([-0.25 + 0.08 * (k - 1), 0.065, 0.096], q), 1, 0, 0, 0]
    assert hl.select_target(q, NAMES) == (None, "No crop in reach (all crops already in basket)")


def test_grasp_check():
    before, tip = np.array([0.6, 0.0, 0.035]), np.array([0.6, 0.0, 0.26])
    assert hl.grasp_ok(before, tip - [0, 0, 0.025], tip)
    assert not hl.grasp_ok(before, before, tip)                       # never left the ground
    assert not hl.grasp_ok(before, before + [0, 0, 0.06], tip)        # rose a bit but fell away
    assert not hl.grasp_ok(before, tip - [0.07, 0, 0.025], tip)       # dangling far from the tip


def test_status_json_contract():
    s = json.loads(hl.status_json("ik_harvest", "lift", "crop_item_2", ""))
    assert s == {"mode": "ik_harvest", "phase": "lift", "target": "crop_item_2", "last_result": ""}
    assert json.loads(hl.status_json("rl_balance", "idle", None, "x"))["target"] is None


def test_slew_limits_each_joint():
    out = hl.slew([0.0, 0.0, 1.0], [1.0, -1.0, 1.01], [0.1, 0.2, 0.05])
    assert out == pytest.approx([0.1, -0.2, 1.01])


# ---------------------------------------------------------------------- sequence control flow (fake world)
class FakeIK:
    grip_open, grip_closed = 0.0, 0.022
    shoulder0 = np.array([0.05, 0.0, 0.195])

    def __init__(self, reachable=True):
        self.reachable, self.last_target = reachable, np.array([0.3, 0.0, 0.8])

    def solve(self, qpos, target, seed_q=None, tool_down=True, gantry=None, max_iters=60):
        if not self.reachable:
            return IKResult(np.zeros(5), 0.3, 0.0, False)
        self.last_target = np.asarray(target, dtype=np.float64)
        return IKResult(np.array([0.2, 0.0, 1.0, -0.3, 0.8]), 0.0, 0.0, True)

    def tip_pos(self, qpos, q5=None):
        return self.last_target.copy()  # perfect tracking of the last IK target


class FakeIO:
    """Crop sticks to the tip while the gripper is closed (if grippy), falls straight down when released."""

    def __init__(self, ik, grippy=True, abort_after=None):
        self.ik, self.grippy, self.abort_after = ik, grippy, abort_after
        self.q = world_qpos(crops={"crop_item_1": (0.62, 0.0, CROP_RADIUS)})
        self.commands, self.phases, self.waits, self.holding = [], [], 0, False

    def qpos(self):
        if self.holding:
            self.q[crop_qpos_slice(1)] = [*(self.ik.tip_pos(self.q) - [0, 0, hl.GRASP_TIP_DZ]), 1, 0, 0, 0]
        return self.q.copy()

    def command(self, arm7):
        self.commands.append(list(arm7))
        grip = arm7[5]
        if self.grippy and grip >= 0.9 * self.ik.grip_closed:
            self.holding = True
        elif self.holding and grip < 0.5 * self.ik.grip_closed:
            self.holding = False
            p = hl.to_trunk(self.ik.tip_pos(self.q) - [0, 0, hl.GRASP_TIP_DZ], self.q)
            if abs(p[0]) < hl.BASKET_HALF_X and abs(p[1]) < hl.BASKET_HALF_Y:
                p[2] = 0.096  # lands on the basket floor
                w = hl.to_world(p, self.q)
            else:
                w = hl.to_world(p, self.q)
                w[2] = CROP_RADIUS
            self.q[crop_qpos_slice(1)] = [*w, 1, 0, 0, 0]

    def wait(self, seconds):
        self.waits += 1
        if self.abort_after is not None and self.waits > self.abort_after:
            raise hl.HarvestAborted()

    def phase(self, name):
        self.phases.append(name)


def test_sequence_success_path_drops_into_basket():
    ik = FakeIK()
    io = FakeIO(ik)
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert (ok, msg) == (True, "Harvested crop_item_1 into basket")
    assert hl.crop_in_basket(io.q, "crop_item_1")
    assert io.phases[0] == "approach" and io.phases[-1] == "return_rest" and "regrasp" not in io.phases
    assert io.commands[-1] == pytest.approx(list(hl.REST_ARM))


def test_sequence_regrasps_once_then_reports_failure_and_rests():
    ik = FakeIK()
    io = FakeIO(ik, grippy=False)
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert (ok, msg) == (False, "Grasp failed on crop_item_1 (2 attempts)")
    assert io.phases.count("verify_grasp") == 2 and "regrasp" in io.phases and "carry" not in io.phases
    assert io.commands[-1] == pytest.approx(list(hl.REST_ARM))


def test_sequence_unreachable_crop_is_reported_and_arm_rests():
    ik = FakeIK(reachable=False)
    io = FakeIO(ik)
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert not ok and msg == "crop_item_1 out of arm reach (0.62 m), move closer"
    assert io.phases[-1] == "return_rest"


def test_sequence_abort_stops_without_touching_the_arm_again():
    ik = FakeIK()
    io = FakeIO(ik, abort_after=30)
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert (ok, msg) == (False, "Aborted harvest of crop_item_1")
    assert "return_rest" not in io.phases
    assert len(io.commands) == 31  # each command is followed by one wait; the 31st wait aborted


# ---------------------------------------------------------------------- closed loop in bare MuJoCo
class SimIO:
    """Drives ctrl exactly like sim_runner: arm ctrl[0:7] from the sequence, legs held at home ctrl."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.legs = model.key_ctrl[0][7:].copy()
        self.commands, self.phases = [], []

    def qpos(self):
        return self.d.qpos.copy()

    def command(self, arm7):
        self.d.ctrl[:7] = arm7
        self.commands.append(list(arm7))

    def wait(self, seconds):
        for _ in range(max(1, int(round(seconds / self.m.opt.timestep)))):
            self.d.ctrl[7:] = self.legs
            mujoco.mj_step(self.m, self.d)

    def phase(self, name):
        self.phases.append(name)


_MODEL = load_farm_model()
_OPENING = DorsalArmIKSolver(_MODEL).gripper_opening()
GRIPPER_TOO_SMALL = _OPENING < 2 * CROP_RADIUS


@pytest.fixture(scope="module")
def closed_loop():
    model = _MODEL
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    io = SimIO(model, data)
    io.wait(1.0)  # settle on the legs
    ik = DorsalArmIKSolver(model)
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", list(model.key_ctrl[0][:7]))
    io.wait(1.0)  # let the arm reach rest
    return {"model": model, "data": data, "io": io, "ok": ok, "msg": msg}


def test_closed_loop_sequence_completes_and_arm_returns_to_rest(closed_loop):
    io, d = closed_loop["io"], closed_loop["data"]
    assert closed_loop["msg"]
    assert io.phases[0] == "approach" and io.phases[-1] == "return_rest"
    assert io.commands[-1] == pytest.approx(list(hl.REST_ARM))
    assert np.allclose(d.qpos[7:12], hl.REST_ARM[:5], atol=0.15)
    assert d.qpos[2] > 0.2  # robot still standing


@pytest.mark.xfail(
    GRIPPER_TOO_SMALL,
    reason=(
        f"MJCF gripper opens only {_OPENING * 100:.1f} cm but crops are {2 * CROP_RADIUS * 100:.1f} cm wide: "
        "pads land on top of the crop and cannot pinch it (sequence correctly reports the grasp failure)"
    ),
    raises=AssertionError,
    strict=False,
)
def test_closed_loop_harvest_puts_crop_item_1_in_basket(closed_loop):
    assert closed_loop["ok"], closed_loop["msg"]
    assert hl.crop_in_basket(closed_loop["data"].qpos, "crop_item_1")
    assert crop_body_names(closed_loop["model"])[0] == "crop_item_1"
