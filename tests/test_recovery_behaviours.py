"""WP28 recovery behaviours (ORCHARD-MVP-SPEC 5.3; MISSION-PRESETS 8; WP28 acceptance):

    local-link loss: stopped + tools safe <= 3 s, crown <= 8.5 s, 0 motion
    cloud-link loss: <= 3 m to the alley edge, then hold; no new targets
    stuck (mud) and sink >= 80 % over 5 seeds (``slow``)
    low-SoC return on RTK FLOAT <= 0.5 m from home
    mutation test: no motion without both e-stop confirmations
    spike report (no gate): tools/self_right_spike.py (``slow``)
    unit <= 10 s

Pure logic on a kinematic plant (no MuJoCo), the hook on the bus, then the real tripod gait on orchard_demo
(SimDriver, every atharv_nandi/sim/hooks file loaded). Headless, ROS-free."""
import contextlib
import itertools
import json
import math
import os
import random
import time

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import recovery_behaviours as rb
from atharv_nandi.core.interfaces import MastRequest

UNIT_WALL_S = 10.0
# real-gait unit tests: the WP13 dock tests use 20 s on the shared 2-CPU runner (WP5 lidar costs ~4-5 ms per sweep)
GAIT_WALL_S = 20.0


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        os.environ.update({k: str(v) for k, v in kw.items()})
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def demo_field():
    from atharv_nandi.sim.world import field_spec
    return field_spec.load("orchard_demo")


@pytest.fixture(scope="module")
def fmap():
    return rb.FieldMap.from_field(demo_field())


# ---------------------------------------------------------------------- kinematic plant (pure)
class Plant:
    """Kinematic stand-in: gait velocities per unit command near the MEASURED ones (vx 0.25 m/s, yaw 0.55 rad/s),
    a 0.3 s stop ramp, the crown at 0.2 m/s toward the newest recovery request, tools safe 0.2 s after an open."""
    VX, WZ, DT = 0.25, 0.55, 0.02

    def __init__(self, x=0.0, y=0.0, yaw=0.0, ext=0.0, **links):
        self.x, self.y, self.yaw, self.t = float(x), float(y), float(yaw), 0.0
        self.ext = float(ext)
        self.crown_target = None
        self.v = 0.0
        self.w = 0.0
        self.walking = False
        self.holding = False
        self.hold_reason = ""
        self.tools_safe = False
        self._tools_t = None
        self.link_local = links.get("local", "up")
        self.link_cloud = links.get("cloud", "up")
        self.rtk = "FIXED"
        self.energy = "continue"
        self.tilt = 0.0
        self.blocked = None          # callable(x, y) -> True: no progress there (stall zone)
        self.cmd = (0.0, 0.0)
        self.mission = None          # (vx, yaw) a mission commands while recovery does not own the gait
        self.moved_while_locked = 0.0

    def obs(self, **kw):
        return rb.Obs(self.t, self.x, self.y, self.yaw, speed=abs(self.v), walking=self.walking,
                      holding=self.holding, hold_reason=self.hold_reason, ext_m=self.ext, tools_safe=self.tools_safe,
                      link_local=self.link_local, link_cloud=self.link_cloud, rtk=self.rtk, energy_action=self.energy,
                      tilt_deg=self.tilt, cmd_vx=self.cmd[0], cmd_vy=0.0, **kw)

    def apply(self, a: rb.Actions):
        if a.tools_open and not self.tools_safe:
            self._tools_t = self._tools_t if self._tools_t is not None else self.t
            if self.t - self._tools_t >= 0.2:
                self.tools_safe = True
        if a.crown is not None:
            self.crown_target = a.crown
        if a.power_off:
            self.holding, self.hold_reason = True, "fall"
        if a.stand_up:
            self.holding = False
        mo = a.motion
        vx, wz = 0.0, 0.0
        if mo is not None and mo.moving:
            vx, wz = mo.vx, mo.yaw
        elif mo is None and not a.lock and self.mission is not None:
            vx, wz = self.mission
        if self.holding:
            vx = wz = 0.0
            self.v = 0.0                    # the hold freezes the joints (the WP6 stop timing is measured in sim)
            self.tools_safe = True          # WP6 EstopActions opens every blade / jaw on the latch edge
        self.cmd = (vx, 0.0)
        tv = self.VX * vx
        k = min(1.0, self.DT / 0.15)
        self.v += k * (tv - self.v)
        self.w = self.WZ * wz
        if mo is not None and mo.stop:
            self.v = 0.0 if abs(self.v) < 0.02 else self.v * 0.6
        self.walking = abs(tv) > 1e-6 or abs(wz) > 1e-6 or abs(self.v) > 0.01
        step = self.v * self.DT
        if self.blocked is not None and self.blocked(self.x + math.cos(self.yaw) * step,
                                                     self.y + math.sin(self.yaw) * step, self):
            step = 0.0
        if a.lock and (mo is None or not mo.moving):
            self.moved_while_locked += abs(step)
        self.x += math.cos(self.yaw) * step
        self.y += math.sin(self.yaw) * step
        self.yaw = rb.wrap(self.yaw + self.w * self.DT)
        if self.crown_target is not None and not self.holding:
            d = self.crown_target - self.ext
            self.ext += max(-0.2 * self.DT, min(0.2 * self.DT, d))
        self.t += self.DT


def run(mgr, plant, until, limit_s=200.0, each=None, **obs_kw):
    while plant.t < limit_s:
        a = mgr.update(plant.obs(**obs_kw))
        plant.apply(a)
        if each is not None:
            each(plant, mgr, a)
        if until(plant, mgr, a):
            return True
    return False


# ---------------------------------------------------------------------- field map (pure)
def test_park_target_alley_edge_within_3m_clear_and_out_of_basins(fmap):
    """MP 8 SAFE PARK target: from anywhere in the four orchard_demo alleys the nearest alley edge or headland is
    <= 3 m away, clear of every canopy / basin rim footprint (+ hull) and inside the geofence."""
    t0 = time.perf_counter()
    fld = demo_field()
    rows = sorted(r["y"] for r in fld["rows"])
    n = 0
    for ya, yb in zip(rows, rows[1:]):
        for x in np.arange(2.0, 7.01, 0.5):
            for y in np.linspace(ya + 1.5, yb - 1.5, 4):
                tgt, kind = fmap.park_target((x, y))
                assert tgt is not None, (x, y, kind)
                assert math.hypot(tgt[0] - x, tgt[1] - y) <= rb.SAFE_PARK_MAX_M + 1e-9
                assert fmap.clear(tgt) and kind in ("alley_edge", "headland")
                n += 1
    # centre of the R1-R2 alley: the edge, not the lane centre; out of the lane by the canopy edge + hull
    tgt, kind = fmap.park_target((5.0, 5.0))
    assert kind == "alley_edge" and abs(tgt[1] - 5.0) > 0.5
    # a robot inside a basin rim is moved out of it; headland ground stays where it is
    tgt, kind = fmap.park_target((6.0, -3.5 + 0.9))          # OT-1 basin
    assert tgt is not None and fmap.clear(tgt)
    tgt, kind = fmap.park_target((0.6, 9.0))
    assert kind in ("headland", "alley_edge") and fmap.clear(tgt)
    print(f"park targets checked: {n}")
    assert time.perf_counter() - t0 < UNIT_WALL_S


def test_route_home_is_clear_of_canopies_and_ends_at_home(fmap):
    for p in ((6.0, 5.0), (8.0, 9.25), (4.0, 12.75), (9.0, 0.3), (3.0, 16.0)):
        route = fmap.route_home(p)
        assert route[-1] == fmap.home
        cur = p
        for w in route:
            assert fmap.segment_clear(cur, w), (p, route)
            cur = w
    assert fmap.route_home((1.0, -0.5)) == [fmap.home]                 # open ground: straight


# ---------------------------------------------------------------------- behaviours on the plant (pure)
def test_local_link_loss_stops_tools_crown_and_no_motion_until_ack():
    """OS 5.3 / MP 8 LOCAL SAFETY LINK LOSS on the plant: walking crown-up (1.6 m) with a mission commanding forward
    the whole time; the link drops: stopped + tools safe <= 3 s, crown stowed <= 8.5 s, then 0 motion while lost and
    after the restore until the operator answers; "resume" hands motion back."""
    p = Plant(5.0, 5.0, 0.0, ext=1.6)
    p.mission = (0.6, 0.0)
    mgr = rb.RecoveryManager()
    run(mgr, p, lambda pl, *_: pl.t >= 2.0)
    assert p.walking and mgr.active is None
    t_loss = p.t
    p.link_local = "lost"
    run(mgr, p, lambda pl, m, a: isinstance(m.active, rb.LocalLinkLoss) and m.active.step == "hold", 30.0)
    b = mgr.active
    st = b.status()
    print("local link loss:", json.dumps({k: st[k] for k in ("stopped_s", "tools_safe_s", "crown")}))
    assert st["stopped_s"] <= rb.TOOLS_SAFE_S and st["tools_safe_s"] <= rb.TOOLS_SAFE_S
    assert p.ext <= rb.CROWN_STOWED_M and st["crown"]["stow_s"] <= rb.CROWN_STOW_S
    xy = (p.x, p.y)
    p.moved_while_locked = 0.0
    run(mgr, p, lambda pl, *_: pl.t >= t_loss + 20.0)
    p.link_local = "up"
    run(mgr, p, lambda pl, *_: pl.t >= t_loss + 25.0)
    assert b.step == "await_confirm" and mgr.last.prompt.startswith("Link was lost")
    assert math.hypot(p.x - xy[0], p.y - xy[1]) < 1e-6 and p.moved_while_locked == 0.0
    ok, _ = mgr.command(p.t, {"ack": "maybe"})
    assert not ok
    assert mgr.command(p.t, {"ack": "resume"}) == (True, "resumed")
    run(mgr, p, lambda pl, *_: pl.t >= t_loss + 27.0)
    assert mgr.active is None and math.hypot(p.x - xy[0], p.y - xy[1]) > 0.1      # the mission moves again


def test_local_link_ack_refused_while_lost_and_halt_stays_parked():
    """Halt is not Resume: with a mission commanding forward the whole time, the robot stays parked and locked after
    "Halt" (queue paused, no new targets) until a later Resume; a new link loss still takes over."""
    p = Plant(1.0, 1.0)
    p.mission = (0.6, 0.0)
    mgr = rb.RecoveryManager()
    p.link_local = "lost"
    run(mgr, p, lambda pl, m, a: m.active is not None and m.active.step == "hold", 20.0)
    ok, msg = mgr.command(p.t, {"ack": "resume"})
    assert not ok and "not restored" in msg
    p.link_local = "up"
    run(mgr, p, lambda pl, m, a: m.active.step == "await_confirm", 25.0)
    assert mgr.command(p.t, {"ack": "halt"}) == (True, "halted")
    assert mgr.pending_ack_link == ["local"]
    xy, t1 = (p.x, p.y), p.t
    run(mgr, p, lambda pl, *_: pl.t >= t1 + 5.0)
    assert math.hypot(p.x - xy[0], p.y - xy[1]) < 1e-6                     # stays parked
    assert mgr.history[-1]["outcome"] == "halted" and len(mgr.history) == 1
    assert mgr.active is not None and mgr.active.outcome == "halted"
    assert mgr.last.lock and mgr.last.queue_pause and not mgr.last.new_targets and mgr.last.prompt.startswith("Halted")
    assert not mgr.command(p.t, {"run": "safe_park"}, p.obs())[0]          # the halt waits for the operator
    assert mgr.command(p.t, {"ack": "halt"}) == (True, "halted")
    assert not mgr.command(p.t, {"ack": "maybe"})[0]
    p.link_local = "lost"                                                  # safety still takes over
    run(mgr, p, lambda pl, *_: pl.t >= t1 + 6.0)
    assert isinstance(mgr.active, rb.LocalLinkLoss) and mgr.active.active
    p.link_local = "up"
    run(mgr, p, lambda pl, m, a: m.active.step == "await_confirm", t1 + 20.0)
    assert mgr.command(p.t, {"ack": "halt"}) == (True, "halted")
    run(mgr, p, lambda pl, *_: pl.t >= t1 + 21.0)
    assert mgr.command(p.t, {"ack": "resume"}) == (True, "resumed after halt") and mgr.active is None
    xy = (p.x, p.y)
    run(mgr, p, lambda pl, *_: pl.t >= t1 + 23.0)
    assert math.hypot(p.x - xy[0], p.y - xy[1]) > 0.1                      # the mission moves again


def test_halt_on_cloud_link_loss_and_safe_park_hold_stays_parked(fmap):
    for case in ("cloud", "safe_park"):
        p = Plant(5.0, 5.0)
        mgr = rb.RecoveryManager(fmap)
        run(mgr, p, lambda pl, *_: pl.t >= 0.2)
        if case == "cloud":
            p.link_cloud = "lost"
            run(mgr, p, lambda pl, m, a: isinstance(m.active, rb.CloudLinkLoss) and m.active.step == "hold", 60.0)
            p.link_cloud = "up"
            run(mgr, p, lambda pl, m, a: m.active.step == "await_confirm", 65.0)
        else:
            assert mgr.command(p.t, {"run": "safe_park"}, p.obs())[0]
            run(mgr, p, lambda pl, m, a: m.active.step == "hold", 60.0)
        p.mission = (0.6, 0.0)
        assert mgr.command(p.t, {"ack": "halt"}) == (True, "halted")
        xy, t1 = (p.x, p.y), p.t
        run(mgr, p, lambda pl, *_: pl.t >= t1 + 3.0)
        assert math.hypot(p.x - xy[0], p.y - xy[1]) < 0.005, case             # (the park stand ramp settles)
        assert mgr.last.lock and mgr.last.queue_pause and not mgr.last.new_targets, case
        assert mgr.command(p.t, {"ack": "resume"}) == (True, "resumed after halt")
        run(mgr, p, lambda pl, *_: pl.t >= t1 + 5.0)
        assert math.hypot(p.x - xy[0], p.y - xy[1]) > 0.1, case


def test_cloud_link_loss_parks_at_the_alley_edge_holds_and_starts_no_targets(fmap):
    """MP 8 CLOUD LINK LOSS: SAFE PARK <= 3 m to the alley edge, hold, queue paused, 0 new targets while lost; a local
    supervisor ack resumes while the cloud is still lost (no re-trigger), a console ack after the restore too."""
    p = Plant(5.0, 5.0, 0.3)
    mgr = rb.RecoveryManager(fmap)
    starts = {"n": 0}

    def mission(pl, m, a):                     # a mission layer that starts a target whenever it may
        if a.new_targets and m.active is None:
            starts["n"] += 1

    run(mgr, p, lambda pl, *_: pl.t >= 0.5)
    n0 = starts["n"]
    p.link_cloud = "lost"
    xy0 = (p.x, p.y)
    run(mgr, p, lambda pl, m, a: isinstance(m.active, rb.CloudLinkLoss) and m.active.step == "hold", 60.0, each=mission)
    park = mgr.active.park.status()
    d = math.hypot(p.x - xy0[0], p.y - xy0[1])
    print("cloud park:", park["target_kind"], park["target"], f"moved {d:.2f} m")
    assert park["target_kind"] == "alley_edge" and d <= rb.SAFE_PARK_MAX_M
    assert fmap.clear((p.x, p.y))
    run(mgr, p, lambda pl, *_: pl.t >= 40.0, each=mission)
    assert starts["n"] == n0                       # 0 new targets while disconnected
    assert mgr.last.queue_pause and not mgr.last.new_targets and mgr.last.lock
    ok, _ = mgr.command(p.t, {"ack": "resume"})            # console ack while lost: refused
    assert not ok
    ok, msg = mgr.command(p.t, {"ack": "resume", "local": True})             # self-asserted on the console: refused
    assert not ok and "local safety link" in msg
    assert mgr.command(p.t, {"ack": "resume"}, source="local") == (True, "resumed")
    run(mgr, p, lambda pl, *_: pl.t >= 42.0)
    assert mgr.active is None                        # no re-trigger while the cloud stays lost after a local resume
    p.link_cloud = "up"
    run(mgr, p, lambda pl, *_: pl.t >= 43.0)
    p.link_cloud = "lost"
    run(mgr, p, lambda pl, m, a: isinstance(m.active, rb.CloudLinkLoss), 50.0)
    assert isinstance(mgr.active, rb.CloudLinkLoss)   # a new loss after a restore triggers again


def test_cloud_loss_without_the_local_link_sits_in_place(fmap):
    p = Plant(5.0, 5.0)
    mgr = rb.RecoveryManager(fmap)
    p.link_local = "lost"
    p.link_cloud = "lost"
    run(mgr, p, lambda pl, m, a: m.active is not None and m.active.step == "hold", 20.0)
    assert isinstance(mgr.active, rb.LocalLinkLoss)                  # local loss wins: no motion at all
    sp = rb.SafePark(fmap, reason="operator")
    o = p.obs()
    sp.start(o)
    for _ in range(50):
        sp.update(o)
    assert sp.step in ("sit", "hold") and "local_link" in sp.log[-1][2] or sp.target is None


def test_stall_detector_progress_along_the_command_sink_and_turning():
    det = rb.StallDetector()
    t = 0.0
    # forward command, 1 cm in 5 s -> stuck at 5 s
    hit = None
    while t < 6.0 and hit is None:
        hit = det.update(rb.Obs(t, 0.002 * t, 0.0, 0.0, walking=True, cmd_vx=1.0, prog_x=0.002 * t, prog_y=0.0))
        t += 0.02
    assert hit == rb.R_STUCK and 4.95 <= t <= 5.1
    # rocking +-5 cm while commanded forward: still stuck (progress is along the command)
    det.reset()
    hit, t = None, 0.0
    while t < 6.0 and hit is None:
        x = 0.05 * math.sin(3 * t) - 0.01 * t
        hit = det.update(rb.Obs(t, x, 0.0, 0.0, walking=True, cmd_vx=1.0, prog_x=x, prog_y=0.0))
        t += 0.02
    assert hit == rb.R_STUCK
    # progressing 0.2 m/s: never; turning in place (no translation command): never; holding: never
    for kw in ({"cmd_vx": 1.0}, {"cmd_vx": 0.0}, {"cmd_vx": 1.0, "holding": True}):
        det.reset()
        for i in range(400):
            tt = i * 0.02
            x = 0.2 * tt if kw.get("cmd_vx") else 0.0
            assert det.update(rb.Obs(tt, x, 0.0, 0.3 * tt * (kw["cmd_vx"] == 0), walking=True, prog_x=x, prog_y=0.0,
                                     **kw)) is None
    # sink: feet deeper than SINK_DEPTH_M for 1 s while walking
    det.reset()
    hit = None
    for i in range(80):
        tt = i * 0.02
        hit = hit or det.update(rb.Obs(tt, 0.1 * tt, 0, 0, walking=True, cmd_vx=1.0, sink_m=0.06, prog_x=0.1 * tt,
                                       prog_y=0))
    assert hit == rb.R_SINK


def _stall_zone(y_line, passable_step=None):
    """Plant stall: no progress past ``y_line`` heading +y unless the gait's step height >= passable_step."""
    def blocked(x, y, pl):
        return y > y_line and pl.y <= y_line + 1e-9 and (passable_step is None or (pl.step_h or 1.0) < passable_step)
    return blocked


def test_stuck_ladder_recovers_with_the_higher_step_and_goes_to_assist_after_two_retries():
    """MP 8 STUCK: stop -> back up 0.2 m -> retry (step x 1.5) -> alternative heading -> assist; 2 retries."""
    for passable, want in ((1.5, "recovered"), (None, "assist")):
        p = Plant(0.0, 0.0, math.pi / 2)
        p.step_h = 1.0
        p.mission = (1.0, 0.0)
        p.blocked = _stall_zone(1.0, passable)
        mgr = rb.RecoveryManager()

        def each(pl, m, a):
            pl.step_h = (a.motion.step_height if a.motion is not None and a.motion.step_height else 1.0)

        run(mgr, p, lambda pl, m, a: (m.history and m.history[-1]["behaviour"] == "stuck") or (
            m.active is not None and m.active.outcome == "assist"), 150.0, each=each, prog_x=None)
        b = mgr.active if mgr.active is not None else None
        res = b.outcome if b is not None else mgr.history[-1]["outcome"]
        st = b.status() if b is not None else mgr.history[-1]["status"]
        print(want, json.dumps(st["attempts"]))
        assert res == want
        assert all(abs(a_["backup_m"] - rb.BACKUP_M) < 0.03 for a_ in st["attempts"])
        if want == "assist":
            assert len(st["attempts"]) == rb.STUCK_RETRIES and mgr.last.lock and mgr.last.beacon == "assist"
            assert abs(st["heading_deg"] - (90.0 + rb.ALT_HEADING_DEG)) < 1.0       # rung 2: +30 deg
            ok, _ = mgr.command(p.t, {"ack": "halt"})
            assert ok and mgr.active is None
        else:
            assert len(st["attempts"]) == 1 and p.y > 1.0


def test_return_home_rtk_and_odometry_row_following(fmap):
    """MP 8 RETURN HOME on the plant from the R1-R2 alley: FIXED at 0.18 m/s; odometry (RTK NONE, the estimate drifts
    2 % + 1 deg) at 0.10 m/s, corrected by registry-trunk observations: arrives <= 0.5 m from home (truth)."""
    t0 = time.perf_counter()
    for rtk in ("FIXED", "NONE"):
        p = Plant(6.0, 5.0, math.pi)
        p.rtk = rtk
        mgr = rb.RecoveryManager(fmap)
        assert mgr.command(0.0, {"run": "return_home"}, p.obs()) == (True, "return_home started")
        est = {"x": p.x, "y": p.y, "px": p.x, "py": p.y}
        rng = random.Random(3)
        bias = math.radians(1.0)
        speeds = []

        def obs_kw():
            if rtk == "FIXED":
                return {}
            lms = []
            for t_ in fmap.nearest_trees((p.x, p.y)):
                c, s = math.cos(p.yaw), math.sin(p.yaw)
                dx, dy = t_.x - p.x, t_.y - p.y
                lms.append((t_.x, t_.y, c * dx + s * dy + rng.gauss(0, 0.03), -s * dx + c * dy + rng.gauss(0, 0.03)))
            return {"landmarks": tuple(lms)}

        while p.t < 400.0 and mgr.active is not None:
            dx, dy = p.x - est["px"], p.y - est["py"]
            est["px"], est["py"] = p.x, p.y
            c, s = math.cos(bias), math.sin(bias)
            est["x"] += 1.02 * (c * dx - s * dy)
            est["y"] += 1.02 * (s * dx + c * dy)
            o = p.obs(**obs_kw()) if rtk == "NONE" else p.obs()
            if rtk == "NONE":
                o.x, o.y = est["x"], est["y"]
            a = mgr.update(o)
            p.apply(a)
            if a.motion is not None and a.motion.moving and abs(a.motion.yaw) < 0.3:
                speeds.append(abs(p.v))
        h = mgr.history[-1]
        err = math.hypot(p.x - fmap.home[0], p.y - fmap.home[1])
        print(rtk, h["outcome"], f"err {err:.3f} m", "v_max", round(max(speeds), 3), h["status"]["mode_s"],
              "landmarks", h["status"]["landmarks"])
        assert h["outcome"] == "home" and err <= rb.HOME_TOL_M
        cap = rb.RTH_SPEED_RTK_MPS if rtk == "FIXED" else rb.RTH_SPEED_ODOM_MPS
        assert max(speeds) <= cap + 0.01
        if rtk == "NONE":
            assert h["status"]["landmarks"] > 50
    assert time.perf_counter() - t0 < UNIT_WALL_S


def _estop_run(events, link_up=True):
    """Hold from t = 1 s; ``events`` = [(t, kind)], kind in local / operator / early_local (before the hold) /
    external_release (somebody re-arms the runner without the recovery). Returns (log, plant, mgr)."""
    p = Plant(1.0, 1.0)
    p.mission = (1.0, 0.0)
    p.link_local = "up" if link_up else "lost"
    mgr = rb.RecoveryManager()
    evs = sorted(events)
    log = {"stand_up": None, "moved_before": 0.0, "results": []}
    started = False
    while p.t < 12.0:
        if not started and p.t >= 1.0:
            p.holding, p.hold_reason, started = True, "operator", True
        while evs and p.t >= evs[0][0]:
            _, kind = evs.pop(0)
            if kind == "external_release":
                p.holding = False
            else:
                k = "local" if kind == "early_local" else kind
                log["results"].append((kind, mgr.command(p.t, {"confirm": k}, p.obs(),
                                                           source="local" if k == "local" else "console")))
        x0 = p.x
        a = mgr.update(p.obs())
        if a.stand_up and log["stand_up"] is None:
            log["stand_up"] = p.t
        if log.get("ready") is None and any(h["behaviour"] == "estop_recover" and h["outcome"] == "ready"
                                            for h in mgr.history):
            log["ready"] = p.t
        p.apply(a)
        if started and log.get("ready") is None:
            log["moved_before"] += abs(p.x - x0)
    return log, p, mgr


def test_estop_mutation_no_motion_without_both_confirmations():
    """MP 8 E-STOP RECOVERY interlock, mutation test over event sequences: missing, swapped, duplicated, early
    (before the e-stop), local while the local link is lost, an external re-arm. Motion (or a re-arm) only after a
    valid local confirm followed by a valid operator confirm; then the robot stands up and the mission moves again."""
    t0 = time.perf_counter()
    kinds = ["local", "operator", "early_local", "external_release"]
    n = n_ok = 0
    for r in range(0, 4):
        for seq in itertools.permutations(kinds + ["operator"], r):
            for link_up in (True, False):
                events = []
                for i, k in enumerate(seq):
                    events.append((0.5 if k == "early_local" else 1.5 + 0.5 * i, k))
                log, p, mgr = _estop_run(events, link_up)
                valid = False
                seen_local = False
                for _, k in sorted(events):
                    if k == "local" and link_up:
                        seen_local = seen_local or True
                    if k == "operator" and seen_local:
                        valid = True
                        break
                n += 1
                external = "external_release" in seq
                if valid:
                    n_ok += 1
                    assert log.get("ready") is not None, (seq, link_up, log)
                    assert external or log["stand_up"] is not None, (seq, link_up, log)
                    assert p.x > 1.05, (seq, link_up)                          # resumed and moved
                else:
                    assert log["stand_up"] is None and log.get("ready") is None, (seq, link_up, log)
                assert log["moved_before"] == 0.0, (seq, link_up, log)          # never moved before both
    print(f"e-stop mutation sequences: {n}, valid {n_ok}")
    assert n > 60 and n_ok > 5
    assert time.perf_counter() - t0 < UNIT_WALL_S


def test_estop_checklist_crown_retract_after_rearm_and_resume_point():
    p = Plant(1.0, 1.0, ext=0.6)
    mgr = rb.RecoveryManager()
    run(mgr, p, lambda pl, *_: pl.t >= 0.2)
    p.holding, p.hold_reason = True, "handheld"
    run(mgr, p, lambda pl, *_: pl.t >= 0.5, item_active=True, current_item="R1-2", last_completed="R1-1")
    b = mgr.active
    assert isinstance(b, rb.EstopRecovery) and b.aborted_item == "R1-2" and b.resume_from == "R1-1"
    mgr.command(p.t, {"confirm": "local"}, p.obs(), source="local")
    run(mgr, p, lambda pl, *_: pl.t >= 0.6)
    mgr.command(p.t, {"confirm": "operator"}, p.obs())
    # fields not clear: the checklist holds the re-arm
    run(mgr, p, lambda pl, *_: pl.t >= 2.0, fields_clear=False)
    assert b.step == "checklist" and p.holding and not b.checklist["fields_clear"]
    run(mgr, p, lambda pl, m, a: m.active is None, 30.0)
    h = mgr.history[-1]
    assert h["outcome"] == "ready" and p.ext <= rb.CROWN_STOWED_M and not p.holding
    log_steps = [s for _, s, _ in h["log"]]
    assert log_steps.index("rearm") < log_steps.index("crown_retract") < log_steps.index("stand")


def test_fallen_powers_off_locks_and_needs_the_full_checklist():
    p = Plant(1.0, 1.0)
    mgr = rb.RecoveryManager()
    p.tilt = 40.0
    run(mgr, p, lambda pl, m, a: isinstance(m.active, rb.Fallen), 2.0)
    assert 0.29 <= p.t <= 0.36                                       # tilt > 35 deg for 0.3 s
    run(mgr, p, lambda pl, *_: pl.t >= 1.0)
    assert p.holding and mgr.last.lock and mgr.last.prompt == "Fallen: manual recovery"
    assert not mgr.command(p.t, {"check": "person_clear"})[0]          # person clear is last
    for item in rb.FALLEN_CHECKLIST:
        assert mgr.command(p.t, {"check": item})[0]
    p.tilt = 0.0
    run(mgr, p, lambda pl, *_: pl.t >= 1.2)
    assert mgr.history[-1]["outcome"] == "manual" and p.holding         # never stands up by itself
    # no FALLEN restart while people handle it in the still-latched fall hold
    run(mgr, p, lambda pl, *_: pl.t >= 4.0)
    assert mgr.active is None and [h["behaviour"] for h in mgr.history] == ["fallen"]
    assert not mgr.last.power_off and p.holding
    # the supervisor resets the hold: no power_off puts it back, it stays out of the hold
    p.holding, p.hold_reason = False, ""
    run(mgr, p, lambda pl, *_: pl.t >= 6.0)
    assert not p.holding and mgr.active is None and len(mgr.history) == 1
    # a new fall is a new FALLEN
    p.tilt = 50.0
    run(mgr, p, lambda pl, m, a: isinstance(m.active, rb.Fallen), 7.0)
    assert isinstance(mgr.active, rb.Fallen) and p.holding


def test_fallen_manual_while_tilted_then_righted_then_tips_again():
    p = Plant(1.0, 1.0)
    mgr = rb.RecoveryManager()
    p.tilt = 40.0
    run(mgr, p, lambda pl, m, a: isinstance(m.active, rb.Fallen), 2.0)
    for item in rb.FALLEN_CHECKLIST:
        mgr.command(p.t, {"check": item})
    run(mgr, p, lambda pl, *_: pl.t >= 2.0)                               # still tilted, people lifting: no restart
    assert mgr.active is None and mgr.fall_manual == "tilted"
    p.tilt = 5.0
    run(mgr, p, lambda pl, *_: pl.t >= 2.5)
    assert mgr.fall_manual == "righted" and mgr.active is None
    p.tilt = 45.0                                                          # tipped over again in the hold
    run(mgr, p, lambda pl, *_: pl.t >= 2.6)
    assert isinstance(mgr.active, rb.Fallen) and mgr.fall_manual is None


def test_manager_priorities_and_operator_menu(fmap):
    p = Plant(5.0, 5.0)
    mgr = rb.RecoveryManager(fmap)
    assert mgr.command(0.0, {"run": "safe_park"}, p.obs())[0]
    run(mgr, p, lambda pl, *_: pl.t >= 0.5)
    p.energy = "return_and_wait"                                          # RTH outranks SAFE PARK
    run(mgr, p, lambda pl, *_: pl.t >= 1.0)
    assert isinstance(mgr.active, rb.ReturnHome)
    p.link_cloud = "lost"                                                 # cloud loss does not stop a battery RTH
    run(mgr, p, lambda pl, *_: pl.t >= 2.0)
    assert isinstance(mgr.active, rb.ReturnHome)
    p.link_local = "lost"                                                 # local loss stops everything
    run(mgr, p, lambda pl, *_: pl.t >= 2.5)
    assert isinstance(mgr.active, rb.LocalLinkLoss)
    assert not mgr.command(p.t, {"run": "return_home"}, p.obs())[0]
    p.holding, p.hold_reason = True, "operator"
    run(mgr, p, lambda pl, *_: pl.t >= 3.0)
    assert isinstance(mgr.active, rb.EstopRecovery)
    p.tilt = 50.0
    run(mgr, p, lambda pl, *_: pl.t >= 3.5)
    assert isinstance(mgr.active, rb.Fallen)
    assert not mgr.command(p.t, {"bogus": 1})[0] and not mgr.command(p.t, "x")[0]
    assert not mgr.command(p.t, {"run": "fly"})[0]


def test_self_right_gate():
    assert rb.self_right_gate(True, True, True, True) == (True, "")
    assert rb.self_right_gate(False, True, True, True) == (False, "person_hold")
    assert rb.self_right_gate(True, False, True, True) == (False, "needs_confirmation")
    assert rb.self_right_gate(True, True, False, True) == (False, "crown_gate")
    assert rb.self_right_gate(True, True, True, False) == (False, "arm_not_folded")


# ---------------------------------------------------------------------- hook (bus, MuJoCo)
@pytest.fixture
def demo_model():
    """A fresh orchard_demo model per test: the WP7 platform runtime (pose estimate, battery) is per model, and a
    teleported start would otherwise inherit the previous test's estimate."""
    from atharv_nandi.core import farm_scene
    with env(NANDI_WORLD="orchard_demo"):
        return farm_scene.load_farm_model()


def test_hook_unarmed_is_byte_neutral_and_publishes_state(demo_model):
    """Every other package's tests load this hook: unarmed it must not change physics (qpos bytes equal with and
    without the hook over 1.5 s of walking) and acts on nothing, but still publishes its state."""
    import atharv_nandi.sim.hooks as hk
    from atharv_nandi.core.sim_driver import SimDriver
    t0 = time.perf_counter()
    names = hk.hook_files()
    assert "recovery" in names
    out = []
    for use in (True, False):
        reg = hk.load_all(names=[n for n in names if use or n != "recovery"])
        drv = SimDriver(demo_model, hooks=reg)
        drv.walk(1.0)
        drv.wait(1.5)
        out.append(drv.d.qpos.tobytes())
        if use:
            rec = drv.bus.values["recovery"]
            assert rec["armed"] is False and rec["actions"]["motion"] is None
            assert drv.published_count["/nandi/recovery/state"] >= 3
    assert out[0] == out[1]
    assert time.perf_counter() - t0 < GAIT_WALL_S


def test_hook_commands_results_and_malformed_input(demo_model):
    from atharv_nandi.core.sim_driver import SimDriver
    drv = SimDriver(demo_model)
    drv.wait(0.1)
    for c in ({"arm": True}, {"run": "safe_park"}, {"confirm": "local"}, "not json", {"ack": 3}, [1, 2]):
        drv.deliver("/nandi/recovery/cmd", c)
    drv.wait(0.1)
    res = [json.loads(t)for _, tp, t in drv.published if tp == "/nandi/recovery/alert"]
    res = [r for r in res if r.get("id") == "cmd_result"]
    assert [r["ok"] for r in res] == [True, True, False, False, False, False]
    assert drv.hook_faults == {} and drv.bus.values["recovery"]["active"] == "safe_park"


# ---------------------------------------------------------------------- real gait (SimDriver, orchard_demo)
def _place(drv, x, y, yaw):
    from atharv_nandi.sim.world.modules.tools import seat_tools
    d = drv.d
    d.qpos[0], d.qpos[1] = x, y
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    seat_tools(drv.m, d)
    mujoco.mj_forward(drv.m, d)


def _driver(m, xy=None, yaw=0.0, links=("local", "cloud")):
    """SimDriver + RecoveryRunner + a heartbeat task (10 Hz) for the links still in ``beats``."""
    from atharv_nandi.core.sim_driver import SimDriver
    drv = SimDriver(m)
    if xy is not None:
        _place(drv, xy[0], xy[1], yaw)
    runner = rb.RecoveryRunner(drv)
    beats = {k: True for k in links}

    def hb(dv):
        for k, on in beats.items():
            if on:
                dv.deliver(f"/nandi/link/{k}/heartbeat", 1)
    drv.add_task("heartbeat", hb, rate_hz=10)
    return drv, runner, beats


def _rec(drv):
    return drv.bus.values.get("recovery") or {}


def test_sim_local_link_loss_crown_up_walking(demo_model):
    """Acceptance on the real gait: walking crown-up (1.6 m, ~0.15 m/s) with a mission task commanding forward every
    tick; the local link drops (WP7 injection): stopped + tools safe <= 3 s and crown stowed <= 8.5 s after the
    LOSS (heartbeat timeout included), 0 motion while disconnected (truth), and none after the restore until the
    operator acks."""
    t0 = time.perf_counter()
    drv, runner, beats = _driver(demo_model, (1.6, -3.6), 0.0)
    drv.stand(1.0)
    assert {k: v["state"] for k, v in drv.bus.values["link"].items()} == {"local": "up", "cloud": "up"}
    mc = drv.bus.values["mast_control"]
    assert mc.submit(MastRequest("preset", 1.6, "test", t=drv.t))
    drv.run_until(lambda dv: dv.bus.values["mast"]["ext_m"] >= 1.59, 12.0)
    mission = {"on": True}
    drv.add_task("mission", lambda dv: mission["on"] and dv.gait.set_command(0.6, 0.0, 0.0, hold=True), rate_hz=50)
    drv.wait(2.0)
    v_walk = float(np.hypot(*drv.d.qvel[:2]))
    t_loss = drv.t
    drv.deliver("/nandi/link/cmd", {"drop": {"link": "local", "dur": 25.0}})
    drv.run_until(lambda dv: (_rec(dv).get("behaviour") or {}).get("step") == "hold", 14.0, chunk_s=0.1)
    b = _rec(drv)["behaviour"]
    lost_since = drv.bus.values["link"]["local"]["lost_since"]
    stopped = b["t0"] + b["stopped_s"] - t_loss
    tools = b["t0"] + b["tools_safe_s"] - t_loss
    crown = b["t0"] + b["crown"]["stow_s"] - t_loss
    xy = drv.trunk_xy()
    drv.run_until(lambda dv: dv.t >= t_loss + 24.0, 30.0, chunk_s=0.2)
    drift = float(np.linalg.norm(drv.trunk_xy() - xy))
    drv.run_until(lambda dv: (_rec(dv).get("behaviour") or {}).get("step") == "await_confirm", 4.0, chunk_s=0.1)
    drv.wait(1.0)
    drift2 = float(np.linalg.norm(drv.trunk_xy() - xy))
    kpi = _rec(drv)["kpi"]
    print(f"local link loss: walking {v_walk:.3f} m/s, detect {lost_since - t_loss:.2f} s, stopped {stopped:.2f} s, "
          f"tools {tools:.2f} s, crown stowed {crown:.2f} s after the loss; drift while lost {drift * 1000:.1f} mm, "
          f"after restore {drift2 * 1000:.1f} mm; lock stops {runner.lock_stops}; kpi {kpi}")
    assert 0.1 < v_walk and stopped <= rb.TOOLS_SAFE_S and tools <= rb.TOOLS_SAFE_S
    assert crown <= rb.CROWN_STOW_S
    assert drift < 0.01 and drift2 < 0.01 and kpi["local_drift_truth_m"] < 0.01
    assert _rec(drv)["prompt"].startswith("Link was lost")
    drv.deliver("/nandi/recovery/cmd", {"ack": "resume"})
    drv.wait(2.0)
    assert _rec(drv)["active"] is None and float(np.linalg.norm(drv.trunk_xy() - xy)) > 0.05
    assert drv.bus.values["link"]["local"]["needs_confirm"] is False             # the WP7 ack was forwarded
    assert drv.hook_faults == {}
    assert time.perf_counter() - t0 < GAIT_WALL_S


def test_sim_cloud_link_loss_parks_at_the_alley_edge(demo_model):
    """Acceptance on the real gait: in the R1-R2 alley, the console heartbeat stops: SAFE PARK moves <= 3 m to the
    alley edge (outside the canopy / basin footprint), then holds standing; 0 new targets while disconnected."""
    t0 = time.perf_counter()
    drv, runner, beats = _driver(demo_model, (4.0, 5.0), 0.0)
    drv.stand(1.0)
    starts = {"n": 0, "lost": 0}

    def mission(dv):
        r = _rec(dv)
        if r.get("new_targets", True) and r.get("active") is None:
            starts["n"] += 1
        elif dv.bus.values["link"]["cloud"]["state"] == "lost" and r.get("new_targets"):
            starts["lost"] += 1
    drv.add_task("mission", mission, rate_hz=5)
    xy0 = drv.trunk_xy()
    beats["cloud"] = False
    t_off = drv.t
    ok = drv.run_until(lambda dv: (_rec(dv).get("behaviour") or {}).get("step") == "hold", 45.0, chunk_s=0.1)
    n_start = starts["n"]
    park = _rec(drv)["behaviour"]["park"]
    xy = drv.trunk_xy()
    d = float(np.linalg.norm(xy - xy0))
    fm = rb.FieldMap.from_field(demo_field())
    drv.wait(3.0)
    hold_drift = float(np.linalg.norm(drv.trunk_xy() - xy))
    print(f"cloud loss: detect {drv.bus.values['link']['cloud']['lost_since'] - t_off:.2f} s, park "
          f"{park['target_kind']} {park['target']}, moved {d:.2f} m (truth), at {xy.round(2).tolist()}, "
          f"hold drift {hold_drift * 1000:.1f} mm, park done {drv.t - 3.0 - t_off:.1f} s after the loss")
    assert ok and park["target_kind"] == "alley_edge" and d <= rb.SAFE_PARK_MAX_M
    assert math.hypot(xy[0] - park["target"][0], xy[1] - park["target"][1]) < 0.3
    assert fm.clear(xy, hull=rb.HULL_HALF_M - 0.15)
    assert starts["n"] == n_start and starts["lost"] == 0 and hold_drift < 0.02
    assert time.perf_counter() - t0 < GAIT_WALL_S


def test_sim_estop_recovery_no_motion_without_both_confirmations(demo_model):
    """The mutation interlock on the real gait: e-stop while walking; a mission keeps commanding forward; somebody
    re-arms the runner from outside (as an /nandi/estop_reset would): the lock keeps the robot still. Only local ->
    operator confirms re-arm and hand the gait back."""
    t0 = time.perf_counter()
    drv, runner, beats = _driver(demo_model, (1.6, -3.6), 0.0)
    drv.stand(0.5)
    drv.add_task("mission", lambda dv: dv.gait.set_command(1.0, 0.0, 0.0, hold=True), rate_hz=50)
    drv.wait(1.5)
    drv.enter_hold("operator")
    drv.wait(1.0)
    xy = drv.trunk_xy()
    drv.deliver("/nandi/recovery/cmd", {"confirm": "operator"})          # wrong order: refused
    drv.wait(0.5)
    drv.leave_hold()                                                      # external re-arm, no confirmations
    drv.wait(2.0)
    moved_ext = float(np.linalg.norm(drv.trunk_xy() - xy))
    drv.enter_hold("operator")
    drv.wait(0.5)
    drv.deliver("/nandi/recovery/local", {"confirm": "local"})
    drv.wait(0.5)
    assert drv.holding and runner.stand_ups == 0
    drv.deliver("/nandi/recovery/cmd", {"confirm": "operator"})
    ok = drv.run_until(lambda dv: _rec(dv).get("active") is None, 10.0, chunk_s=0.1)
    moved_before = float(np.linalg.norm(drv.trunk_xy() - xy))
    drv.wait(2.0)
    moved_after = float(np.linalg.norm(drv.trunk_xy() - xy))
    res = [json.loads(t) for _, tp, t in drv.published if tp == "/nandi/recovery/alert"]
    oks = [(r["cmd"]["confirm"], r["ok"]) for r in res if r.get("id") == "cmd_result"]
    print(f"estop: moved after an external re-arm {moved_ext * 1000:.1f} mm, before ready {moved_before * 1000:.1f} mm,"
          f" after ready {moved_after:.2f} m; confirms {oks}; stand-ups {runner.stand_ups}")
    assert oks == [("operator", False), ("local", True), ("operator", True)]
    assert ok and moved_ext < 0.02 and moved_before < 0.03 and moved_after > 0.2 and runner.stand_ups == 1
    assert time.perf_counter() - t0 < GAIT_WALL_S


def test_sim_fallen_powers_off_and_locks(demo_model):
    """A robot rolled to 60 deg: fall latch after 0.3 s -> hold ``fall``, lock, the Fallen checklist alert."""
    drv, runner, beats = _driver(demo_model, (1.6, -3.6), 0.0)
    d = drv.d
    r = math.radians(60.0)
    d.qpos[2] = 0.45
    d.qpos[3:7] = [math.cos(r / 2), math.sin(r / 2), 0, 0]
    mujoco.mj_forward(drv.m, d)
    drv.run_until(lambda dv: dv.holding, 3.0)
    alerts = [json.loads(t) for _, tp, t in drv.published if tp == "/nandi/recovery/alert"]
    fall = [a for a in alerts if a.get("id") == "fallen"]
    assert drv.holding and drv.bus.values["hold"]["reason"] == "fall" and fall
    assert fall[0]["checklist"] == list(rb.FALLEN_CHECKLIST)
    assert _rec(drv)["prompt"] == "Fallen: manual recovery" and _rec(drv)["lock"]


def _battery_low(drv, soc=0.12):
    from atharv_nandi.core import dock_logic as dl
    drv.wait(0.05)
    bat = dl.platform_battery(drv.m)
    bat.energy_wh = soc * bat.p.capacity_wh


@pytest.mark.slow
@pytest.mark.parametrize("gnss,start", [("FLOAT", (4.0, 5.0)), ("NONE", (4.0, 5.0))])
def test_sim_low_soc_return_home(demo_model, gnss, start):
    """Acceptance: low SoC (WP7 reserve monitor -> return_and_wait) in the R1-R2 alley, RTK FLOAT for the whole run
    (WP7 GNSS script) -> RETURN HOME on the WP7 pose estimate at 0.18 m/s, <= 0.5 m from home (truth). NONE: odometry
    only, row-following against the registry trunks at 0.10 m/s."""
    t0 = time.perf_counter()
    # FLOAT for the whole run; NONE: FIXED for the first 2 s (the robot knew where it was), then no fix at all
    script = ({"start_state": "FLOAT", "dropouts": [{"t0": 0.0, "dur": 1e6, "state": "FLOAT"}]} if gnss == "FLOAT"
              else {"start_state": "FIXED", "dropouts": [{"t0": 2.0, "dur": 1e6, "state": "NONE"}]})
    with env(NANDI_GNSS=json.dumps(script)):
        drv, runner, beats = _driver(demo_model, start, math.pi)
        _battery_low(drv)
        samples = []                         # progression speed: trunk xy at 2 Hz while driving (no gait sway)

        def sample(dv):
            if (_rec(dv).get("behaviour") or {}).get("step") == "drive":
                samples.append((dv.t, *dv.trunk_xy()))
        drv.add_task("speed_probe", sample, rate_hz=2)
        ok = drv.run_until(lambda dv: any(h["behaviour"] == "return_home" for h in _rec(dv).get("history", [])),
                           150.0, chunk_s=0.2)
    h = [x for x in runner_history(drv) if x["behaviour"] == "return_home"][-1]
    st = h["status"]
    home = (0.0, 0.0)
    err = float(math.hypot(drv.d.qpos[0] - home[0], drv.d.qpos[1] - home[1]))
    v = [math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) for a, b in zip(samples, samples[1:])
         if 0.4 < b[0] - a[0] < 0.6]
    v_mean = float(np.mean(v)) if v else 0.0
    print(f"RTH {gnss}: outcome {h['outcome']}, err {err:.3f} m (truth), est err {st['est_err_m']}, "
          f"{st['home_s']} s, mode_s {st['mode_s']}, mean speed {v_mean:.3f} m/s, route {st['route']}, "
          f"landmarks {st['landmarks']}, wall {time.perf_counter() - t0:.1f} s")
    assert ok and h["outcome"] == "home" and err <= rb.HOME_TOL_M
    assert drv.bus.values["platform"]["rtk"] == gnss
    cap = rb.RTH_SPEED_RTK_MPS if gnss != "NONE" else rb.RTH_SPEED_ODOM_MPS
    assert v_mean <= cap * 1.15


def runner_history(drv):
    from atharv_nandi.sim.hooks import recovery as hr
    return hr.runtime_for(drv.m).mgr.history


def root_line(m, patch, h, at=0.3, w=0.05):
    """Sim fixture (ASSUMED obstacle, in-memory, not in the WP4 field): a root / clod line across +y travel, ``at`` m
    past the patch centre, ``w`` wide, ``h`` above the patch top: hfield cells raised (they poke through the flush
    patch box). WP4's patches alone never stall the gait (MEASURED: mud crossed at 0.27 m/s, sink_1 at 0.18 m/s, even
    at friction 0.03), so the stall comes from a real contact obstacle that the nominal step catches."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, patch)
    cx, cy, hx = m.geom_pos[g][0], m.geom_pos[g][1], m.geom_size[g][0]
    hg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "orchard_ground")
    hid = int(m.geom_dataid[hg])
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    rx, ry, zt, _ = m.hfield_size[hid]
    gx, gy, gz = m.geom_pos[hg]
    data = m.hfield_data[int(m.hfield_adr[hid]):int(m.hfield_adr[hid]) + nrow * ncol].reshape(nrow, ncol)
    xs = gx - rx + 2 * rx * np.arange(ncol) / (ncol - 1)
    ys = gy - ry + 2 * ry * np.arange(nrow) / (nrow - 1)
    n = 0
    for i, y in enumerate(ys):
        if abs(y - (cy + at)) <= w / 2 + 1e-9:
            for j, x in enumerate(xs):
                if abs(x - cx) <= hx:
                    data[i, j] = max(data[i, j], (h - gz) / zt)
                    n += 1
    return n


# MEASURED fixture heights (2026-11-18): the nominal step (lift ~7 cm) stalls, x 1.5 (~10.5 cm) clears. On the soft patch
# the feet sink ~3.6 cm (WP4), so its root is lower for about the same height above the sunk feet.
ROOT_H = {"mud": ("patch_mud_1", -3.0, 0.05), "sink": ("patch_sink_1", -5.0, 0.035)}


def stuck_trial(patch_kind, seed, recovery=True, limit_s=60.0, h=None):
    from atharv_nandi.core import farm_scene
    from atharv_nandi.core.sim_driver import SimDriver
    patch, cx, h0 = ROOT_H[patch_kind]
    h = h0 if h is None else h
    rng = random.Random(seed)
    with env(NANDI_WORLD="orchard_demo"):
        m = farm_scene.load_farm_model()
    root_line(m, patch, h)
    drv = SimDriver(m)
    dx, hd = rng.uniform(-0.25, 0.25), math.radians(rng.uniform(-5, 5))
    _place(drv, cx + dx, -3.4, math.pi / 2 + hd)
    if recovery:
        rb.RecoveryRunner(drv)
    drv.stand(1.0 + rng.uniform(0, 0.5))
    goal = (cx + dx, -0.3)

    def mission(dv):                       # the mission walks to a goal beyond the patch (recovery overrides it)
        x, y = dv.trunk_xy()
        ang = rb.wrap(math.atan2(goal[1] - y, goal[0] - x) - dv.heading())
        dv.gait.set_command(1.0 if abs(ang) < 0.6 else 0.0, 0.0, max(-1, min(1, 1.5 * ang)), hold=True)
    drv.add_task("mission", mission, rate_hz=50)
    tilt = [0.0]
    drv.add_observer(lambda dv: tilt.__setitem__(0, max(tilt[0], math.degrees(math.acos(
        max(-1.0, min(1.0, 1 - 2 * (dv.d.qpos[4] ** 2 + dv.d.qpos[5] ** 2))))))))

    def done(dv):
        r = _rec(dv)
        return dv.trunk_xy()[1] >= goal[1] - 0.05 or (r.get("behaviour") or {}).get("outcome") == "assist"
    drv.run_until(done, limit_s, chunk_s=0.1)
    from atharv_nandi.sim.hooks import recovery as hr
    mgr = hr.runtime_for(m).mgr
    inc = [(x["behaviour"], x["outcome"], x["status"].get("attempts")) for x in mgr.history]
    if mgr.active is not None:
        inc.append((mgr.active.name, mgr.active.outcome, getattr(mgr.active, "attempts", None)))
    return {"reached": bool(drv.trunk_xy()[1] >= goal[1] - 0.05), "y": float(drv.trunk_xy()[1]), "t": drv.t,
            "max_tilt": tilt[0], "incidents": inc}


@pytest.mark.slow
@pytest.mark.parametrize("patch_kind", ["mud", "sink"])
def test_sim_stuck_and_sink_recovered_over_5_seeds(patch_kind):
    """Acceptance (``slow``): the mission walks across the mud / soft-soil patch over a root line; the nominal step
    stalls on it (progress < 2 cm in 5 s along the command), the ladder runs on the real gait; >= 80 % of 5 seeds
    reach the goal beyond the patch with no assist, and a control run without recovery stays stalled."""
    t0 = time.perf_counter()
    res = [stuck_trial(patch_kind, s) for s in (1, 2, 3, 4, 5)]
    for s, r in zip((1, 2, 3, 4, 5), res):
        print(patch_kind, s, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
    stalls = sum(1 for r in res if any(i[0] == "stuck" for i in r["incidents"]))
    recovered = sum(1 for r in res if r["reached"] and not any(i[1] == "assist" for i in r["incidents"]))
    print(f"{patch_kind}: stalled {stalls}/5, recovered {recovered}/5, wall {time.perf_counter() - t0:.1f} s")
    assert stalls == 5                               # every seed actually stalled first (the KPI is not vacuous)
    assert recovered >= 4                            # >= 80 %
    assert all(r["max_tilt"] < rb.FALL_TILT_DEG for r in res)


@pytest.mark.slow
def test_sim_stuck_control_without_recovery_stays_stalled():
    r = stuck_trial("sink", 1, recovery=False, limit_s=40.0)
    print("control (no recovery):", json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
    assert not r["reached"]


@pytest.mark.slow
def test_self_right_spike_runs_and_reports(tmp_path):
    """MP 8 SELF-RIGHT spike: report only, no gate. The tool runs one case and writes its JSON report."""
    import subprocess
    import sys
    tool = os.path.join(os.path.dirname(__file__), "..", "tools", "self_right_spike.py")
    out = tmp_path / "spike.json"
    r = subprocess.run([sys.executable, tool, "--roll", "60", "--quick", "--out", str(out)], capture_output=True,
                       text=True, timeout=170)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads(out.read_text())
    case = rep["cases"][0]
    for k in ("roll0_deg", "success", "time_s", "max_tilt_deg", "torque_vs_forcerange", "gate"):
        assert k in case
    assert rep["gate"] == "none (report only)"


# ---------------------------------------------------------------------- branch coverage (pure + hook glue)
def run_for(mgr, plant, until, dur):
    return run(mgr, plant, until, plant.t + dur)

def test_field_map_without_geofence_and_route_fallbacks():
    fm = rb.FieldMap([{"tree_id": "a", "x": 2.0, "y": 0.0, "canopy_r": 1.0, "basin_r": 0.5, "row_id": "R"},
                      {"tree_id": "b", "x": 4.0, "y": 0.0, "canopy_r": 1.0, "basin_r": 0.5, "row_id": "R"}],
                     [{"row_id": "R", "y": 0.0}, {"row_id": "X", "y": 9.0}], home=(0.0, 0.0))
    assert fm.in_geofence((100.0, 100.0)) and fm.alley_of((50.0, 0.0)) is None
    r = fm.route_home((3.0, 3.0))
    assert r[-1] == (0.0, 0.0)
    # home inside a canopy: the grid start / goal cells are forced free, the path still ends there
    assert fm.route_home((3.0, 3.0), home=(2.0, 0.0))[-1] == (2.0, 0.0)
    # the alley-centreline fallback (grid disabled)
    fm._grid_path = lambda p, h, res=0.25: []
    alt = fm.route_home((3.0, 3.0))
    assert alt[-1] == (0.0, 0.0) and len(alt) >= 1
    assert fm.route_home((3.0, -3.0))[-1] == (0.0, 0.0)
    assert fm.park_target((20.0, 20.0))[1] in ("headland", "open")
    empty = rb.FieldMap([], [])
    assert empty.park_target((1.0, 1.0)) == ((1.0, 1.0), "open")


def test_safe_park_branches(fmap):
    def run_sp(sp, o_kw, n=200, **plant_kw):
        p = Plant(5.0, 5.0, **plant_kw)
        o = p.obs(**o_kw)
        sp.start(o)
        for i in range(n):
            o.t = i * 0.02
            sp.update(o)
        return sp
    assert run_sp(rb.SafePark(fmap, move=False), {}).log[-2][2] == "no_move"
    assert run_sp(rb.SafePark(None), {}).log[-2][2] == "no_field_map"
    assert run_sp(rb.SafePark(fmap), {"walk_ok": False}).log[-2][2] == "gates"
    sp = rb.SafePark(fmap)
    p = Plant(5.0, 5.0, local="lost")
    o = p.obs()
    o.tools_safe = None
    sp.start(o)
    sp.update(o)
    assert any(w == "local_link_not_up" for _, _, w in sp.log)
    # tools never become safe: tool_fault after 3 s, then the park goes on
    sp = rb.SafePark(None)
    o = Plant(5.0, 5.0).obs()
    o.tools_safe = False
    sp.start(o)
    for i in range(200):
        o.t = i * 0.02
        sp.update(o)
    assert any(a["id"] == "tool_fault" for a in sp.alerts) and sp.status()["tools_safe_s"] is None
    # e-stop during SAFE PARK: halt
    o.holding = True
    assert sp.update(o).motion.stop
    # already at the target
    tgt, _ = fmap.park_target((5.0, 5.0))
    sp = rb.SafePark(fmap)
    o = Plant(*tgt).obs()
    o.tools_safe = None
    sp.start(o)
    sp.update(o)
    assert any(w.startswith("already_") for _, _, w in sp.log)


def test_return_home_item_hold_timeout_and_reaim(fmap):
    p = Plant(1.0, -0.5)
    rh = rb.ReturnHome(fmap, reason="energy_reserve")
    o = p.obs(item_active=True)
    rh.start(o)
    assert rh.step == "finish_item" and rh.alerts[0]["id"] == "low_battery_return"
    for i in range(int(rb.ITEM_FINISH_S / 0.02) + 5):
        o.t = i * 0.02
        rh.update(o)
    assert any(w == "item_abandoned" for _, _, w in rh.log)
    o.holding = True
    assert rh.update(o).motion.stop
    o.holding = False
    o.t = rb.RTH_TIMEOUT_S + 1.0
    a = rh.update(o)                    # stow (crown already stowed) -> drive -> past the timeout: assist
    assert rh.outcome == "assist" and a.lock and a.beacon == "assist"
    # a settle that lands off the aim re-aims
    rh2 = rb.ReturnHome(None, home=(0.0, 0.0))
    o2 = Plant(0.5, 0.0).obs()
    rh2.start(o2)
    rh2.route, rh2.wp = [(0.0, 0.0)], 0
    rh2._to("settle", 0.0)
    o2.t = rb.ARRIVE_SETTLE_S + 0.1
    rh2.update(o2)
    assert rh2.step == "drive" and "re-aim" in rh2.log[-1][2]
    # the manager: an RTH assist is acknowledged
    mgr = rb.RecoveryManager(fmap)
    mgr.active = rh
    assert mgr.command(o.t, {"ack": "halt"}) == (True, "assist acknowledged") and mgr.active is None


def test_links_lost_again_and_cloud_halt(fmap):
    p = Plant(5.0, 5.0)
    mgr = rb.RecoveryManager(fmap)
    p.link_local = "lost"
    run_for(mgr, p, lambda pl, m, a: m.active.step == "hold", 20.0)
    p.link_local = "up"
    run_for(mgr, p, lambda pl, m, a: m.active.step == "await_confirm", 5.0)
    p.link_local = "lost"
    run_for(mgr, p, lambda pl, m, a: m.active.step == "hold", 2.0)
    assert mgr.active.log[-1][2] == "lost_again"
    p.link_local = "up"
    run_for(mgr, p, lambda pl, m, a: m.active.step == "await_confirm", 5.0)
    mgr.command(p.t, {"ack": "resume"})
    run_for(mgr, p, lambda pl, m, a: m.active is None, 2.0)
    p.link_cloud = "lost"
    run_for(mgr, p, lambda pl, m, a: isinstance(m.active, rb.CloudLinkLoss) and m.active.step == "hold", 60.0)
    assert not mgr.command(p.t, {"ack": "maybe"})[0]
    p.link_cloud = "up"
    run_for(mgr, p, lambda pl, m, a: m.active.step == "await_confirm", 2.0)
    p.link_cloud = "lost"
    run_for(mgr, p, lambda pl, m, a: m.active.step == "hold", 2.0)
    p.link_cloud = "up"
    run_for(mgr, p, lambda pl, m, a: m.active.step == "await_confirm", 2.0)
    assert mgr.command(p.t, {"ack": "halt"}) == (True, "halted")
    run_for(mgr, p, lambda pl, m, a: False, 0.5)
    assert mgr.active is not None and mgr.active.outcome == "halted" and mgr.last.lock
    assert mgr.command(p.t, {"ack": "resume"}) == (True, "resumed after halt")
    # a SAFE PARK hold is released by an ack; abort; nothing to ack / abort
    run_for(mgr, p, lambda pl, m, a: m.active is None, 2.0)
    mgr.command(p.t, {"run": "safe_park"}, p.obs())
    run_for(mgr, p, lambda pl, m, a: m.active.step == "hold", 30.0)
    assert mgr.command(p.t, {"ack": "resume"})[0]
    run_for(mgr, p, lambda pl, m, a: m.active is None, 2.0)
    assert not mgr.command(p.t, {"ack": "resume"})[0] and not mgr.command(p.t, {"abort": True})[0]
    mgr.command(p.t, {"run": "return_home"}, p.obs())
    assert mgr.command(p.t, {"abort": True}) == (True, "aborted") and mgr.active is None
    assert not mgr.command(p.t, {"confirm": "local"})[0] and not mgr.command(p.t, {"check": "x"})[0]
    assert rb.RecoveryManager().command(0.0, {"run": "safe_park"}) == (False, "no observation yet")


def test_estop_new_hold_during_retract_and_bad_confirms():
    p = Plant(1.0, 1.0, ext=0.6)
    mgr = rb.RecoveryManager()
    run(mgr, p, lambda pl, *_: pl.t >= 0.1)
    p.holding, p.hold_reason = True, "operator"
    run(mgr, p, lambda pl, *_: pl.t >= 0.2)
    b = mgr.active
    assert not b.confirm(p.t, "remote")[0] and not b.confirm(0.0, "local")[0]
    mgr.command(p.t, {"confirm": "local"}, p.obs(), source="local")
    assert not b.confirm(p.t, "local")[0]
    run(mgr, p, lambda pl, *_: pl.t >= 0.4)
    mgr.command(p.t, {"confirm": "operator"}, p.obs())
    run(mgr, p, lambda pl, m, a: b.step == "crown_retract", 5.0)
    p.holding = True
    run(mgr, p, lambda pl, *_: True)
    assert b.step == "await_local" and b.log[-1][2] == "estop_again" and not b.confirms
    assert b.update(p.obs()).lock
    assert not rb.EstopRecovery().confirm(0.0, "local")[0]       # not started


def test_stuck_turn_timeout_and_stand_up_seq():
    sr = rb.StuckRecover(attempt=1, heading=0.0)
    o = Plant(0.0, 0.0).obs()
    sr.start(o)
    sr._to("backup", 0.0)
    sr.bk_from = (0.0, 0.0)
    o.t = rb.BACKUP_TIMEOUT_S + 0.1
    sr.update(o)
    assert sr.step == "turn"
    o.t += 16.0
    sr.update(o)
    assert sr.step == "retry" and sr.log[-1][2] == "turn_timeout"
    o.holding = True
    assert sr.update(o).motion.stop


class _FakeMast:
    def __init__(self):
        self.requests = []

    def submit(self, r):
        self.requests.append(r)
        return True


def test_hook_glue_crown_tools_beacon(demo_model):
    """The armed hook's actuation glue on a bus: a recovery retract only when no higher-priority retract is already
    going down (rate-limited), tools open re-sent every 1 s while not safe, beacon requests on change."""
    from atharv_nandi.sim import hooks
    from atharv_nandi.sim.hooks import recovery as hr
    d = mujoco.MjData(demo_model)
    mujoco.mj_forward(demo_model, d)
    rt = hr.RecoveryRuntime(demo_model, d)
    bus = hooks.Bus()
    mc = _FakeMast()
    bus.values.update(mast_control=mc, mast={"ext_m": 1.2, "target_ext": 1.2, "arbiter_source": "preset"},
                      tools={"mounted": "T2", "value": 0.9, "busy": False})
    o = rt.obs(demo_model, d, 0.0, bus)
    assert o.tools_safe is False and o.ext_m == 1.2
    a = rb.Actions(crown=0.0, crown_reason="t", tools_open=True, beacon="stopped")
    rt._apply(0.0, bus, o, a)
    rt._apply(0.2, bus, o, a)                                             # rate limit: one request, one open
    assert len(mc.requests) == 1 and mc.requests[0].source == "recovery"
    assert bus.arrival_count("/nandi/tool/actuate") == 1 and bus.latest("/nandi/beacon/request")["state"] == "stopped"
    rt._apply(1.1, bus, o, a)
    assert bus.arrival_count("/nandi/tool/actuate") == 2 and len(mc.requests) == 2
    bus.values["mast"] = {"ext_m": 1.0, "target_ext": 0.0, "arbiter_source": "safety"}
    rt._apply(2.0, bus, o, a)
    assert len(mc.requests) == 2                                         # a safety retract is already going down
    bus.values["mast"] = {"ext_m": 0.01, "target_ext": 0.0, "arbiter_source": None}
    rt._apply(3.0, bus, o, rb.Actions())
    assert len(mc.requests) == 2 and bus.latest("/nandi/beacon/request")["state"] == "idle"
    bus.values["mast"] = {"has_mast": False}
    rt._ensure_crown(4.0, bus, mc, 0.0, "x")
    bus.values["mast"] = {"ext_m": "bad"}
    rt._ensure_crown(5.0, bus, mc, 0.0, "x")
    assert len(mc.requests) == 2
    bus.values["tools"] = {"mounted": "T4", "value": 0.16, "busy": False}
    assert rt.obs(demo_model, d, 0.0, bus).tools_safe is True
    bus.values["tools"] = {"mounted": "T2", "value": 0.1, "busy": True}
    assert rt.obs(demo_model, d, 0.0, bus).tools_safe is False
    bus.deliver("/nandi/recovery/cmd", '{"arm": true}')
    rt._commands(0.0, bus, o)
    assert rt.armed and rt.results[-1]["ok"]
    rt.reset()
    assert rt.armed is False and hr._field_for(None) is None


# ---------------------------------------------------------------------- review fixes (WP28 fixer)
def test_estop_confirm_sources_one_sender_cannot_give_both():
    """The local confirm only on the local channel, the operator confirm only on the console: one console client
    sending both is refused; a self-asserted ``local: true`` ack on the console is refused (cloud-link test)."""
    p = Plant(1.0, 1.0)
    p.holding, p.hold_reason = True, "estop"
    mgr = rb.RecoveryManager()
    run(mgr, p, lambda pl, *_: pl.t >= 0.5)
    ok, msg = mgr.command(p.t, {"confirm": "local"})
    assert not ok and "local safety link" in msg
    p.t += 0.02
    assert not mgr.command(p.t, {"confirm": "operator"})[0]
    run(mgr, p, lambda pl, *_: pl.t >= 3.0)
    assert p.holding and mgr.active.step == "await_local"
    assert not mgr.command(p.t, {"confirm": "operator"}, source="local")[0]
    assert not mgr.command(p.t, {"run": "safe_park"}, p.obs(), source="local")[0]
    assert not mgr.command(p.t, {"confirm": "local"}, source="radio")[0]
    assert mgr.command(p.t, {"confirm": "local"}, source="local")[0]
    assert not mgr.command(p.t + 0.1, {"confirm": "operator"}, source="local")[0]
    assert mgr.command(p.t + 0.1, {"confirm": "operator"})[0]
    run(mgr, p, lambda pl, m, a: m.active is None, 10.0)
    assert not p.holding and mgr.history[-1]["outcome"] == "ready"


def test_estop_hold_at_t0_recovers():
    p = Plant(1.0, 1.0)
    p.holding, p.hold_reason = True, "estop"
    mgr = rb.RecoveryManager()
    a = mgr.update(p.obs())                                                # hold seen at t = 0.0
    p.apply(a)
    assert mgr.active.t_hold == 0.0
    run(mgr, p, lambda pl, *_: pl.t >= 0.5)
    assert mgr.command(p.t, {"confirm": "local"}, source="local")[0]
    run(mgr, p, lambda pl, *_: pl.t >= 0.6)
    assert mgr.command(p.t, {"confirm": "operator"})[0]
    assert run(mgr, p, lambda pl, m, a: m.active is None, 4.0)
    assert mgr.history[-1]["outcome"] == "ready" and not p.holding


def test_stuck_stows_the_crown_before_any_recovery_motion():
    p = Plant(5.0, 5.0, ext=1.5)
    p.mission = (0.6, 0.0)
    p.blocked = lambda x, y, pl: x > 5.3
    mgr = rb.RecoveryManager()
    moved_up = []

    def each(pl, m, a):
        if isinstance(m.active, rb.StuckRecover) and a.motion is not None and a.motion.moving and pl.ext > 0.03:
            moved_up.append(pl.t)

    assert run(mgr, p, lambda pl, m, a: isinstance(m.active, rb.StuckRecover) and m.active.step == "retry", 80.0,
               each=each)
    steps = [s for _, s, _ in mgr.active.log]
    assert moved_up == [] and p.ext <= rb.CROWN_STOWED_M + 0.01
    assert steps.index("stow") < steps.index("backup") and mgr.last.crown == 0.0


def test_aborted_low_soc_return_home_is_not_restarted(fmap):
    p = Plant(4.0, 5.0)
    p.energy = "return_and_wait"
    mgr = rb.RecoveryManager(fmap)
    run(mgr, p, lambda pl, *_: pl.t >= 1.0)
    assert isinstance(mgr.active, rb.ReturnHome)
    assert mgr.command(p.t, {"abort": True}) == (True, "aborted")
    run(mgr, p, lambda pl, *_: pl.t >= 2.0)
    assert mgr.active is None
    assert mgr.command(p.t, {"run": "return_home"}, p.obs())[0]           # the operator can still start it
    assert mgr.command(p.t, {"abort": True})[0]
    p.energy = "continue"
    run(mgr, p, lambda pl, *_: pl.t >= 2.1)
    p.energy = "return_and_wait"                                           # a new reserve breach: restarts
    run(mgr, p, lambda pl, *_: pl.t >= 2.2)
    assert isinstance(mgr.active, rb.ReturnHome)


def test_return_home_and_safe_park_moves_detect_a_stall(fmap):
    # RETURN HOME stuck in mud: assist (locked, beacon assist) instead of the 600 s timeout
    p = Plant(4.0, 5.0)
    p.blocked = lambda x, y, pl: pl.t > 3.0 and isinstance(mgr.active, rb.ReturnHome) and mgr.active.step == "drive"
    mgr = rb.RecoveryManager(fmap)
    assert mgr.command(0.0, {"run": "return_home"}, p.obs())[0]
    alerts = []
    assert run(mgr, p, lambda pl, m, a: m.active.outcome == "assist", 60.0,
               each=lambda pl, m, a: alerts.extend(a.alerts))
    rh = mgr.active
    assert rh.log[-1][2] == "stuck" and p.t < 20.0
    assert {"stuck", "rth_failed"} <= {al["id"] for al in alerts}
    t1 = p.t
    run(mgr, p, lambda pl, *_: pl.t >= t1 + 0.5)
    assert mgr.last.lock and mgr.last.beacon == "assist" and mgr.last.motion == rb.STAND
    assert mgr.command(p.t, {"ack": "halt"}) == (True, "assist acknowledged")
    # SAFE PARK move stuck: sits where it is
    p = Plant(5.0, 5.0, 0.3)
    mgr = rb.RecoveryManager(fmap)
    assert mgr.command(0.0, {"run": "safe_park"}, p.obs())[0]
    run(mgr, p, lambda pl, m, a: m.active.step == "move", 20.0)
    p.blocked = lambda x, y, pl: True
    assert run(mgr, p, lambda pl, m, a: m.active.step == "hold", 40.0)
    assert any(why == "move_stopped:stuck" for _, _, why in mgr.active.log)


def test_hook_disarm_refused_while_a_behaviour_holds_the_robot(demo_model):
    from atharv_nandi.sim import hooks
    from atharv_nandi.sim.hooks import recovery as hr
    d = mujoco.MjData(demo_model)
    mujoco.mj_forward(demo_model, d)
    rt = hr.RecoveryRuntime(demo_model, d)
    bus = hooks.Bus()
    bus.deliver("/nandi/recovery/cmd", {"arm": True})
    rt.tick(demo_model, d, 0.0, bus)
    bus.values["link"] = {"local": {"state": "lost"}, "cloud": {"state": "up"}}
    rt.tick(demo_model, d, 0.02, bus)
    assert isinstance(rt.mgr.active, rb.LocalLinkLoss) and bus.values["motion_inhibit"]["recovery"]["stop"]
    bus.deliver("/nandi/recovery/cmd", {"arm": False})
    rt.tick(demo_model, d, 0.04, bus)
    assert not rt.results[-1]["ok"] and "holds the robot" in rt.results[-1]["detail"]
    assert rt.armed and bus.values["motion_inhibit"]["recovery"] == {"stop": True, "walk": True,
                                                                     "reasons": ["local_link"]}
    # idle: a disarm is accepted
    rt.mgr.active = None
    bus.values["link"] = {"local": {"state": "up"}, "cloud": {"state": "up"}}
    bus.deliver("/nandi/recovery/cmd", {"arm": False})
    rt.tick(demo_model, d, 0.06, bus)
    assert rt.results[-1]["ok"] and not rt.armed


def test_hook_local_channel_and_paired_handheld(demo_model):
    from atharv_nandi.sim import hooks
    from atharv_nandi.sim.hooks import recovery as hr
    d = mujoco.MjData(demo_model)
    mujoco.mj_forward(demo_model, d)
    rt = hr.RecoveryRuntime(demo_model, d)
    bus = hooks.Bus()
    bus.values["recovery_arm"] = True
    bus.values["hold"] = {"reason": "operator"}
    bus.values["link"] = {"local": {"state": "up"}, "cloud": {"state": "up"}}
    rt.tick(demo_model, d, 0.0, bus)
    assert isinstance(rt.mgr.active, rb.EstopRecovery)
    bus.deliver("/nandi/recovery/cmd", {"confirm": "local"})               # console: refused
    rt.tick(demo_model, d, 0.1, bus)
    assert not rt.results[-1]["ok"] and rt.results[-1]["source"] == "console"
    bus.values["handheld_estop"] = {"seen": True, "device": "hh-1"}
    bus.deliver("/nandi/recovery/local", {"confirm": "local"})             # no device: refused
    rt.tick(demo_model, d, 0.2, bus)
    assert not rt.results[-1]["ok"] and "paired handheld" in rt.results[-1]["detail"]
    bus.deliver("/nandi/recovery/local", {"confirm": "local", "device": "hh-2"})
    rt.tick(demo_model, d, 0.3, bus)
    assert not rt.results[-1]["ok"]
    bus.deliver("/nandi/recovery/local", {"confirm": "local", "device": "hh-1"})
    rt.tick(demo_model, d, 0.4, bus)
    assert rt.results[-1]["ok"] and rt.results[-1]["source"] == "local"
    assert rt.mgr.active.step == "await_operator"
    bus.deliver("/nandi/recovery/local", {"confirm": "operator", "device": "hh-1"})
    rt.tick(demo_model, d, 0.5, bus)
    assert not rt.results[-1]["ok"]
    bus.deliver("/nandi/recovery/cmd", {"confirm": "operator"})
    rt.tick(demo_model, d, 0.6, bus)
    assert rt.results[-1]["ok"] and rt.mgr.active.step in ("checklist", "rearm")


@pytest.mark.slow
def test_sim_stuck_full_ladder_to_assist_on_the_real_gait():
    """The rungs the recoverable fixture never reaches, on the real gait (3 seeds): a 10 cm root line across the whole
    mud patch (ASSUMED; x 1.5 does not clear it, and the +30 deg heading meets it too): stall -> back up -> rung 1
    (x 1.5) stalls -> back up -> rung 2 (+30 deg, x 1.5) stalls -> assist (locked, beacon assist), no fall. A rung
    that frees the robot only for a new stall within 1.5 m / 60 s continues the same ladder (straight to assist)."""
    t0 = time.perf_counter()
    for seed in (1, 2, 3):
        r = stuck_trial("mud", seed, h=0.10, limit_s=90.0)
        print("full ladder", seed, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}))
        assert not r["reached"] and r["max_tilt"] < rb.FALL_TILT_DEG, seed
        stuck = [i for i in r["incidents"] if i[0] == "stuck"]
        assert stuck and stuck[-1][1] == "assist" and all(i[1] in ("recovered", "assist") for i in stuck), seed
        atts = [a for i in stuck for a in (i[2] or [])]
        assert [a["attempt"] for a in atts] == [1, 2], seed                       # 2 retries, then assist
        assert atts[0]["result"] in ("stuck", "sink", "timeout"), seed
    print(f"full ladder wall {time.perf_counter() - t0:.1f} s")
