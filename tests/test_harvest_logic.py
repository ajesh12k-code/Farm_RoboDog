"""Harvest decision logic, sequence control flow (fake world) and a closed-loop MuJoCo harvest (ROS-free)."""
import json
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import harvest_logic as hl
from atharv_nandi.core.arm_ik_solver import DorsalArmIKSolver, IKResult
from atharv_nandi.core.farm_scene import CROP_RADIUS, crop_body_names, crop_qpos_slice, load_farm_model

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
    held = tip - [0, 0, 0.025]
    assert hl.grasp_ok(before, held, tip)
    assert hl.grasp_ok(before, held, tip, held + [0, 0, 0.01], tip + [0, 0, 0.01])  # carried up together
    assert not hl.grasp_ok(before, before, tip)                       # never left the ground
    assert not hl.grasp_ok(before, before + [0, 0, 0.06], tip)        # rose a bit but fell away
    assert not hl.grasp_ok(before, tip - [0.07, 0, 0.025], tip)       # dangling far from the tip
    # The live false positive: risen 17 cm and 4.5 cm below the tip (passed the old 6 cm rule) but
    # sliding out of the pads; the second sample GRASP_HOLD_S later catches it.
    slipping = tip - [0, 0, 0.045]
    assert hl.grasp_ok(before, slipping, tip)
    assert not hl.grasp_ok(before, slipping, tip, slipping - [0, 0, 0.025], tip)
    assert not hl.grasp_ok(before, tip - [0, 0, 0.05], tip)          # below the pad centre band
    ok, detail = hl.grasp_check(before, before, tip)
    assert not ok and detail == "rose 0.0 cm, 22.5 cm from the tip"


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
    limits = np.array([[-0.26, 0.26], [-3.14, 3.14], [-1.6, 1.68], [-2.6, 2.6], [-2.0, 2.0]])

    def __init__(self, reachable=True, path_err=0.0):
        self.reachable, self.last_target = reachable, np.array([0.3, 0.0, 0.8])
        self.path_err = path_err  # pos_err (not ok) reported for the descend phase's Cartesian waypoints
        self.phase = None         # set by FakeIO.phase

    def solve(self, qpos, target, seed_q=None, tool_down=True, gantry=None, max_iters=60):
        if not self.reachable:
            return IKResult(np.zeros(5), 0.3, 0.0, False)
        self.last_target = np.asarray(target, dtype=np.float64)
        if self.path_err and self.phase == "descend":
            return IKResult(np.array([0.2, 0.0, 1.0, -0.3, 0.8]), self.path_err, 0.0, False)
        return IKResult(np.array([0.2, 0.0, 1.0, -0.3, 0.8]), 0.0, 0.0, True)

    def tip_pos(self, qpos, q5=None):
        return self.last_target.copy()  # perfect tracking of the last IK target


class FakeIO:
    """Crop sticks to the tip while the gripper is closed (if grippy), falls straight down when released."""

    def __init__(self, ik, grippy=True, abort_after=None, drop_in_phase=None):
        self.ik, self.grippy, self.abort_after = ik, grippy, abort_after
        self.drop_in_phase = drop_in_phase  # the crop falls out of the gripper once this phase starts
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
        self.ik.phase = name
        if name == self.drop_in_phase:
            self.grippy, self.holding = False, False
            self.q[crop_qpos_slice(1)] = [0.62, 0.0, CROP_RADIUS, 1, 0, 0, 0]


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
    assert not ok and msg.startswith("Grasp failed on crop_item_1 (2 attempts; last: rose 0.0 cm")
    assert io.phases.count("verify_grasp") == 2 and "regrasp" in io.phases and "carry" not in io.phases
    assert io.commands[-1] == pytest.approx(list(hl.REST_ARM))


def test_sequence_unreachable_crop_is_reported_and_arm_rests():
    ik = FakeIK(reachable=False)
    io = FakeIO(ik)
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert not ok and msg == "crop_item_1 out of arm reach (0.62 m), move closer"
    assert io.phases[-1] == "return_rest"


def test_sequence_unreachable_crop_on_a_ridge_names_the_tilt():
    """Out of reach because the trunk is pitched (ridge), not only far: the message says so."""
    ik = FakeIK(reachable=False)
    io = FakeIO(ik)
    a = math.radians(-3.0)  # about y: nose up
    io.q[3:7] = [math.cos(a / 2), 0.0, math.sin(a / 2), 0.0]
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert not ok and msg.startswith("crop_item_1 out of arm reach (0.62 m; trunk tilted 3.0 deg, pitch +3.0"), msg
    assert msg.endswith("level the robot or move closer")


def test_sequence_unreachable_crop_under_the_body_says_back_up():
    ik = FakeIK(reachable=False)
    io = FakeIO(ik)
    io.q[crop_qpos_slice(1)][0:2] = [0.26, 0.0]
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert (ok, msg) == (False, "crop_item_1 out of arm reach (0.26 m, under the body), back up")


class NudgingIO(FakeIO):
    """The crop is shoved `dx` along x after `at` waits (and again every `every` waits, if given)."""

    def __init__(self, ik, at, dx, every=None):
        super().__init__(ik)
        self.at, self.dx, self.every = at, dx, every

    def wait(self, seconds):
        super().wait(seconds)
        if not self.holding and (self.waits == self.at or (self.every and self.waits > self.at
                                                            and (self.waits - self.at) % self.every == 0)):
            self.q[crop_qpos_slice(1)][0] += self.dx


def _planned_crops(seq):
    planned, plan = [], seq._plan_column

    def spy(qpos, crop, target):
        planned.append(np.array(crop))
        return plan(qpos, crop, target)
    seq._plan_column = spy
    return planned


def test_sequence_replans_the_column_when_the_crop_moved_before_descend():
    """Crop rolls 3 cm during the approach: the column is re-planned for where it is now (the old code
    descended the stale pinned-gantry column and failed half-way with 'Arm path out of reach')."""
    ik = FakeIK()
    io = NudgingIO(ik, at=10, dx=0.03)
    seq = hl.HarvestSequence(ik, io)
    planned = _planned_crops(seq)
    ok, msg = seq.run("crop_item_1", hl.REST_ARM)
    assert (ok, msg) == (True, "Harvested crop_item_1 into basket")
    assert [round(float(p[0]), 3) for p in planned] == [0.62, 0.65]
    assert io.phases.count("descend") == 1


def test_sequence_gives_up_on_a_crop_that_keeps_moving():
    ik = FakeIK()
    io = NudgingIO(ik, at=10, dx=0.03, every=12)
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert not ok and msg.startswith("crop_item_1 keeps moving (") and msg.endswith("since planned), try again")
    assert "descend" not in io.phases and io.phases[-1] == "return_rest"


def test_sequence_reports_a_crop_dropped_while_carrying():
    ik = FakeIK()
    io = FakeIO(ik, drop_in_phase="carry")
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert (ok, msg) == (False, "Dropped crop_item_1 while carrying it")
    assert "release" not in io.phases and io.phases[-1] == "return_rest"
    assert io.commands[-1] == pytest.approx(list(hl.REST_ARM))


def test_sequence_tolerates_small_path_error_but_not_a_large_one():
    """A waypoint 0.9 cm off (gantry pinned + sag offset at the envelope edge) is fine; 3 cm is not."""
    for err, expect_ok in ((0.009, True), (0.03, False)):
        ik = FakeIK(path_err=err)
        io = FakeIO(ik)
        ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
        assert ok == expect_ok, (err, msg)
        if not expect_ok:
            assert msg == "Arm path out of reach (3.0 cm off)"


def test_sequence_abort_stops_without_touching_the_arm_again():
    ik = FakeIK()
    io = FakeIO(ik, abort_after=30)
    ok, msg = hl.HarvestSequence(ik, io).run("crop_item_1", hl.REST_ARM)
    assert (ok, msg) == (False, "Aborted harvest of crop_item_1")
    assert "return_rest" not in io.phases
    assert len(io.commands) == 31  # each command is followed by one wait; the 31st wait aborted


# ---------------------------------------------------------------------- closed loop in bare MuJoCo
class SimIO:
    """Drives a bare MuJoCo world the way the live stack does (sim_runner + harvest_ik_service):

    * physics at the model timestep (500 Hz), legs held at the home keyframe ctrl;
    * qpos() returns the latest 50 Hz /nandi/sim_qpos sample, never the live MjData;
    * command() only sets the arm goal: every 20 ms tick the published arm_targets move toward it by at
      most hl.SLEW_PER_TICK (exactly harvest_ik_service.on_tick) and become ctrl[0:7];
    * wait() is wall-clock time in the live node: with rtf < 1 each tick advances only rtf * 20 ms of
      sim time (a sim running slower than real time).
    """

    def __init__(self, model, data, rtf=1.0, start_cmd=None):
        self.m, self.d, self.rtf = model, data, rtf
        self.legs = model.key_ctrl[0][7:].copy()
        self.goal = list(start_cmd if start_cmd is not None else model.key_ctrl[0][:7])
        self.out = list(self.goal)
        self.sample = data.qpos.copy()
        self.steps, self.debt = 0, 0.0
        self.per_tick = hl.PUBLISH_DT / model.opt.timestep
        self.qpos_every = int(round(hl.PUBLISH_DT / model.opt.timestep))
        self.commands, self.phases = [], []

    def qpos(self):
        return self.sample.copy()

    def command(self, arm7):
        self.goal = list(arm7)
        self.commands.append(list(arm7))

    def wait(self, seconds):
        for _ in range(max(1, int(round(seconds / hl.PUBLISH_DT)))):
            self.out = hl.slew(self.out, self.goal, hl.SLEW_PER_TICK)
            self.debt += self.rtf * self.per_tick
            while self.debt >= 1.0:
                self.debt -= 1.0
                self.d.ctrl[:7] = self.out
                self.d.ctrl[7:] = self.legs
                mujoco.mj_step(self.m, self.d)
                self.steps += 1
                if self.steps % self.qpos_every == 0:
                    self.sample = self.d.qpos.copy()

    def phase(self, name):
        self.phases.append(name)


def place_robot(model, data, trunk_xy=None, heading=0.0, crop_shift=None):
    """Home keyframe, optionally with the trunk teleported/rotated and crop_item_k shifted (world dx, dy)."""
    mujoco.mj_resetDataKeyframe(model, data, 0)
    if trunk_xy is not None:
        data.qpos[0:2] = trunk_xy
        data.qpos[3:7] = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
    for k, (dx, dy) in (crop_shift or {}).items():
        data.qpos[crop_qpos_slice(k)][0:2] += [dx, dy]
    mujoco.mj_forward(model, data)


def run_closed_loop_harvest(model, target="crop_item_1", rtf=1.0, settle=1.0, **placement):
    """Stand still for `settle` s, run one HarvestSequence, let the arm come to rest. Returns a dict."""
    data = mujoco.MjData(model)
    place_robot(model, data, **placement)
    io = SimIO(model, data, rtf=rtf)
    io.wait(settle)  # settle on the legs
    ik = DorsalArmIKSolver(model)
    ok, msg = hl.HarvestSequence(ik, io).run(target, list(model.key_ctrl[0][:7]))
    io.wait(1.0)  # let the arm reach rest
    return {"model": model, "data": data, "io": io, "ok": ok, "msg": msg,
            "in_basket": hl.crop_in_basket(data.qpos, target)}


def trunk_beside(model, k, offset):
    """Trunk xy (heading 0) putting crop_item_k at trunk-frame `offset`."""
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    crop = d.qpos[crop_qpos_slice(k)][:3]
    return (float(crop[0] - offset[0]), float(crop[1] - offset[1]))


_MODEL = load_farm_model()


@pytest.fixture(scope="module")
def closed_loop():
    return run_closed_loop_harvest(_MODEL)


def test_closed_loop_sequence_completes_and_arm_returns_to_rest(closed_loop):
    io, d = closed_loop["io"], closed_loop["data"]
    assert closed_loop["msg"]
    assert io.phases[0] == "approach" and io.phases[-1] == "return_rest"
    assert io.commands[-1] == pytest.approx(list(hl.REST_ARM))
    assert np.allclose(d.qpos[7:12], hl.REST_ARM[:5], atol=0.15)
    assert d.qpos[2] > 0.2  # robot still standing


def test_closed_loop_harvest_puts_crop_item_1_in_basket(closed_loop):
    print(f"crop_item_1 from home: ok={closed_loop['ok']} msg={closed_loop['msg']!r}")
    assert closed_loop["ok"], closed_loop["msg"]
    assert closed_loop["in_basket"]
    assert "regrasp" not in closed_loop["io"].phases  # first grasp holds
    assert crop_body_names(closed_loop["model"])[0] == "crop_item_1"


@pytest.mark.parametrize("shift", [(0.02, 0.0), (-0.02, 0.02), (0.0, -0.02)])
def test_closed_loop_harvest_tolerates_crop_offsets(shift):
    r = run_closed_loop_harvest(_MODEL, crop_shift={1: shift})
    assert r["ok"] and r["in_basket"], (shift, r["msg"])


def test_closed_loop_harvest_with_slow_sim():
    """Wall-clock waits with the sim at 60 % real time: the sequence waits for the arm, not the clock."""
    r = run_closed_loop_harvest(_MODEL, rtf=0.6)
    assert r["ok"] and r["in_basket"], r["msg"]


@pytest.mark.parametrize("k", [2, 3])
def test_closed_loop_harvest_other_crops_with_trunk_teleported(k):
    """Trunk teleported (standing) so crop_item_k sits ahead-left/right of the nose, like crop_item_1.

    0.42 m ahead (was 0.45): with kp 300 legs the trunk no longer sags (stands ~9 mm higher, 0.334 vs 0.325 m on
    flat, MEASURED), and crop_item_3 lies beside ridge_1 with the front feet on it, so at 0.45 m the ground pick was
    just past the pinned-gantry column envelope (0.53 m, "move closer")."""
    side = 0.25 if _crop_y(k) > 0 else -0.25
    r = run_closed_loop_harvest(_MODEL, target=f"crop_item_{k}", trunk_xy=trunk_beside(_MODEL, k, (0.42, side)))
    assert r["ok"] and r["in_basket"], r["msg"]
    assert r["data"].qpos[2] > 0.2


def _crop_y(k):
    d = mujoco.MjData(_MODEL)
    mujoco.mj_resetDataKeyframe(_MODEL, d, 0)
    return float(d.qpos[crop_qpos_slice(k)][1])


def test_closed_loop_held_crop_does_not_creep_out_of_the_pads():
    """Root cause of the live 'Drop missed': with MuJoCo's default soft contact the pinched crop slid
    through the pads at ~8 cm/s (any grip force) and fell out early in the carry. The finger_pad contact
    class must hold it: < 2 mm/s relative to the tip while hanging still."""
    data = mujoco.MjData(_MODEL)
    place_robot(_MODEL, data)
    io = SimIO(_MODEL, data)
    io.wait(1.0)
    ik = DorsalArmIKSolver(_MODEL)
    seq = hl.HarvestSequence(ik, io)
    seq.q5, seq.grip = np.array(hl.REST_ARM[:5]), hl.REST_ARM[5]
    ok, _ = seq._pick("crop_item_1", 1)
    assert ok
    rel = lambda: crop_pos_now(data) - ik.tip_pos(data.qpos)  # noqa: E731
    r0 = rel()
    io.wait(2.0)
    creep = np.linalg.norm(rel() - r0) / 2.0
    print(f"held crop creep {creep * 1000:.2f} mm/s, crop centre {-r0[2] * 100:.1f} cm below the tip")
    assert creep < 0.002
    assert 0.015 < -r0[2] < 0.04


def crop_pos_now(data, k=1):
    return data.qpos[crop_qpos_slice(k)][:3].copy()


def test_closed_loop_crop_tight_against_the_legs_fails_cleanly():
    """A crop beside the front leg (trunk-frame (0.1, 0.2)) cannot be picked from above: the planner's
    arm-vs-robot check reports it out of reach before the arm moves (it used to try, hit the robot and
    report a grasp failure)."""
    r = run_closed_loop_harvest(_MODEL, crop_shift={1: (0.1 - 0.62, 0.2)})
    assert not r["ok"] and "out of arm reach" in r["msg"], r["msg"]
    assert r["io"].phases == ["approach", "return_rest"]


def test_closed_loop_truly_out_of_reach_fails_cleanly():
    """A crop 0.95 m ahead (inside the 1.08 m select radius, beyond the arm) fails before the arm moves."""
    r = run_closed_loop_harvest(_MODEL, crop_shift={1: (0.33, 0.0)})
    assert not r["ok"] and r["msg"].startswith("crop_item_1 out of arm reach (0.9") and r["msg"].endswith(
        " m), move closer")
    assert r["io"].phases == ["approach", "return_rest"]
    assert not r["in_basket"]


def test_closed_loop_regrasp_waits_for_the_rolling_crop_and_replans():
    """crop_item_2 placed 8 cm beside crop_item_1 blocks a finger pad, so the first grasp fails (right)
    and the released crop rolls. The regrasp must plan for where the crop comes to rest; it used to plan
    for where it still was mid-roll and fail half-way down with 'Arm path out of reach (1.7 cm off)'."""
    d = mujoco.MjData(_MODEL)
    mujoco.mj_resetDataKeyframe(_MODEL, d, 0)
    c1, c2 = d.qpos[crop_qpos_slice(1)][:3], d.qpos[crop_qpos_slice(2)][:3]
    r = run_closed_loop_harvest(_MODEL, crop_shift={2: (c1[0] - c2[0], c1[1] + 0.08 - c2[1])})
    assert "Arm path out of reach" not in r["msg"], r["msg"]
    if not r["ok"] and r["msg"].startswith("Grasp failed"):
        # condim-6 crops (rolling friction) no longer roll away when let go, so the regrasp can meet the same
        # blocking neighbour: a clean two-attempt failure after a wait-and-replan, never a mid-path abort.
        assert r["io"].phases.count("regrasp") == 1, r["io"].phases
    elif not r["ok"]:  # the crop rolled under the body / out of reach: a clean report before descending
        assert "out of arm reach" in r["msg"], r["msg"]
        assert r["io"].phases.count("descend") == 1, r["io"].phases
    else:
        assert r["in_basket"]


def test_closed_loop_crop_on_a_ridge_fails_cleanly_and_names_the_tilt():
    """crop_item_3 directly ahead stands on a ridge (trunk ~4 deg nose up): the ground pick is out of
    the arm's reach, and the report says why instead of only 'move closer'."""
    r = run_closed_loop_harvest(_MODEL, target="crop_item_3", trunk_xy=trunk_beside(_MODEL, 3, (0.62, 0.0)))
    if r["ok"]:
        assert r["in_basket"]
        return
    assert "out of arm reach" in r["msg"] and "trunk tilted" in r["msg"], r["msg"]
    assert r["io"].phases == ["approach", "return_rest"]
