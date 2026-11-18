"""WP3 transfers: desap cell tray, papaya cradle, ballast and crate unload (ORCHARD-MVP-SPEC 1.2 step 11, 2.4;
MISSION-PRESETS 9; WP3 acceptance):

    items in the crate AABB, impact < 2 m/s (papaya <= 0.5)
    0 fruit-on-fruit contacts in the tray; one fruit per cell, stalk-down
    ballast = delivered sum; cap 3 kg; one papaya per trip
    transfer only with arm PARK and gait stopped
    <= 10 s (each test, wall, on the 2-CPU capped runner)

Headless, ROS-free; the orchard_demo world (OS 1.2 demo load: OT-1 3 mango, R2 3 guava, R3-1 1 papaya)."""
import contextlib
import json
import math
import os
import time

import mujoco
import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import farm_scene, sim_state
from atharv_nandi.core.interfaces import TransferControl
from atharv_nandi.core.sim_driver import SimDriver
from atharv_nandi.core.tools import kinematic_transfer as kt
from atharv_nandi.sim.hooks import transfer as ht

DEMO = (("OT-1", 3), ("R2-1", 2), ("R2-3", 1), ("R3-1", 1))   # 3 mango + 3 guava + 1 papaya (OS 1.2 steps 5-8)
PARK_OFFSET_Y = 0.53     # trunk axis to crate centre when parked alongside (feet |y| 0.25 + r 0.03 + gap + crate 0.2)
CATCH_DROP_M = 0.10      # the fruit CoM hangs this far below the PARK TCP in the catcher


def hold_at_catcher(ctrl, m, d, slot, offset=(0.0, 0.0, -CATCH_DROP_M)):
    """Put a released pool fruit where WP2 / WP8 hand it over: held in the catcher at the PARK TCP."""
    mujoco.mj_forward(m, d)
    s = ctrl.pool.slots[slot]
    d.qpos[s.qadr:s.qadr + 3] = ctrl.L.handoff_point(d) + np.asarray(offset, float)
    d.qvel[s.dadr:s.dadr + 6] = 0.0
    mujoco.mj_forward(m, d)


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


@pytest.fixture(scope="module")
def model():
    with env(NANDI_WORLD="orchard_demo"):
        return farm_scene.load_farm_model()


def demo_ids(pool):
    out = []
    for tree, n in DEMO:
        out += sorted(f for f, v in pool.fruit.items() if v["tree_id"] == tree and v["in_band"] and v["harvest_stage"]
                      and not v["bagged"])[:n]
    return out


class Ctx:
    def __init__(self, m):
        self.m = m
        self.d = mujoco.MjData(m)
        self.reg = hook_registry.load_all(names=["transfer"])
        self.bus = hook_registry.Bus()
        self.bus.values["mast"] = {"ext_m": 0.0, "has_mast": True}   # WP24 crown stowed (mast hook not loaded)
        self.ctrl = ht.controller_for(m)
        self.pool = self.ctrl.pool
        mujoco.mj_resetDataKeyframe(m, self.d, 0)
        self.pool.reset(self.d)
        self.reg.reset(m, self.d)
        mujoco.mj_forward(m, self.d)

    def step(self, n=1, cb=None):
        m, d = self.m, self.d
        for _ in range(n):
            d.ctrl[:25] = m.key_ctrl[0][:25]
            self.reg.substep(m, d, self.bus)
            mujoco.mj_step(m, d)
            self.reg.tick(m, d, self.bus)
            if cb:
                cb(self)

    def inputs(self):
        return ht.runtime_for(self.m).inputs(self.d, self.bus)

    def until_idle(self, limit_s=40.0, cb=None):
        n = 0
        while self.ctrl.busy() and n * self.m.opt.timestep < limit_s:
            self.step(1, cb)
            n += 1
        assert not self.ctrl.busy(), "transfer did not finish"
        return n * self.m.opt.timestep

    def handoff(self, fid, offset=(0.0, 0.0, -CATCH_DROP_M)):
        """Assign + release the fruit (logical cut / twist release) and hold it at the catcher next to the PARK TCP
        (the arm at PARK): what WP2 / WP8 hand over."""
        slot = self.pool.assign(self.d, fid)
        self.pool.release(self.d, slot)
        hold_at_catcher(self.ctrl, self.m, self.d, slot, offset)
        return slot

    def demo_ids(self):
        return demo_ids(self.pool)

    def place_all(self, ids):
        times = []
        for fid in ids:
            self.handoff(fid)
            g = self.ctrl.place(self.d, fid, "auto", self.inputs())
            assert g, (fid, g.reason, g.values)
            times.append(self.until_idle())
        return times

    def park_by_crate(self):
        c, _, _ = self.ctrl.crate_frame(self.d)
        self.d.qpos[0:2] = [c[0], c[1] + PARK_OFFSET_Y]
        mujoco.mj_forward(self.m, self.d)
        self.step(300)


@pytest.fixture()
def ctx(model):
    c = Ctx(model)
    yield c
    c.ctrl.reset(c.d)
    c.pool.reset(c.d)


def angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))))


# ---------------------------------------------------------------------- pure logic
def test_interlock_order_and_reasons():
    assert kt.interlock(kt.GateInputs())
    assert kt.interlock(kt.GateInputs(holding=True, arm_parked=False, gait_stopped=False)).reason == "estop"
    assert kt.interlock(kt.GateInputs(arm_parked=False, gait_stopped=False)).reason == "arm_not_parked"
    assert kt.interlock(kt.GateInputs(gait_stopped=False)).reason == "gait_moving"


def test_path_speeds_ends_and_round_trip():
    q0 = [1.0, 0.0, 0.0, 0.0]
    q1 = kt._q_from_to([0, 0, 1], [0, 0, -1])
    p = kt.Path([[0, 0, 0], [0, 0, 0.2], [0.3, 0, 0.2], [0.3, 0, 0.05]], [q0, q0, q1, q1], [0.45, 0.45, 0.3])
    assert p.duration == pytest.approx(0.2 / 0.45 + 0.3 / 0.45 + 0.15 / 0.3)
    assert p.max_speed() == pytest.approx(0.45) and p.last_speed() == pytest.approx(0.3)
    pe, qe = p.pose(p.duration + 1.0)
    assert np.allclose(pe, [0.3, 0, 0.05]) and abs(abs(float(qe @ q1)) - 1.0) < 1e-12
    v, w = p.vel(p.duration)
    assert not v.any() and not w.any()
    v, _ = p.vel(0.1)
    assert np.allclose(v, [0, 0, 0.45])
    # numerical angular velocity matches during the rotating carry segment
    s, h = 0.7, 1e-5
    _, w = p.vel(s)
    qa, qb = p.pose(s)[1], p.pose(s + h)[1]
    ang, ax = kt._q_angle_axis(qa, qb)
    assert np.allclose(kt._qmat(qa) @ (ax * ang / h), w, atol=1e-3)
    p2 = kt.Path.from_dict(json.loads(json.dumps(p.to_dict())))
    assert p2.duration == p.duration and np.array_equal(p2.p, p.p)
    with pytest.raises(kt.TransferError):
        kt.Path([[0, 0, 0]], [q0], [])
    with pytest.raises(kt.TransferError):
        kt.Path([[0, 0, 0], [1, 0, 0]], [q0, q0], [0.0])


def test_quaternion_helpers():
    for a, b in (([0, 0, 1], [0, 0, -1]), ([1, 0, 0], [1, 0, 0]), ([0, 1, 0], [0, 0, 1]), ([1, 0, 0], [-1, 0, 0])):
        q = kt._q_from_to(a, b)
        assert np.allclose(kt._qmat(q) @ np.asarray(a, float), b, atol=1e-12)
    R = kt._qmat(kt._q_from_to([0, 0, 1], [1, 0, 0]))
    assert np.allclose(kt.ellipsoid_half_extents(R, [0.07, 0.07, 0.11]), [0.11, 0.07, 0.07])


def test_command_validation():
    assert ht.check_command({"op": "place", "fruit_id": "OT-1:f01"}) == ""
    assert ht.check_command({"op": "place", "fruit_id": "x", "target": 7}) == ""
    assert ht.check_command({"op": "unload"}) == ""
    for bad in (None, [], {"op": "jump"}, {"op": "place"}, {"op": "place", "fruit_id": "x", "target": 8},
                {"op": "place", "fruit_id": "x", "target": True}):
        assert ht.check_command(bad), bad


# ---------------------------------------------------------------------- model layout
def test_layout_cells_fit_the_demo_fruit(ctx):
    L, pool = ctx.ctrl.L, ctx.pool
    assert len(L.cells) == kt.N_CELLS == 8
    assert L.floor_top == pytest.approx(0.006)
    # every cell is bounded by real walls: lane 84 mm, cell ~85 mm (WP1 geometry)
    for h in L.cell_half:
        assert 0.040 <= h[0] <= 0.045 and 0.040 <= h[1] <= 0.045
    # every in-band harvest-stage mango and guava of orchard_demo fits a cell stalk-down
    for fid, f in pool.fruit.items():
        if f["crop"] in ("mango", "guava") and f["in_band"] and f["harvest_stage"]:
            q = kt._q_from_to(f["stem_axis"], [0, 0, -1])
            h = kt.ellipsoid_half_extents(kt._qmat(q), f["size_m"])
            assert h[0] <= L.cell_half[0][0] and h[1] <= L.cell_half[0][1], (fid, h)
    assert L.rail_half_gap == pytest.approx(0.021) and L.rail_top == pytest.approx(0.030)
    assert isinstance(ctx.ctrl, TransferControl)


# ---------------------------------------------------------------------- tray placement
def test_demo_load_into_tray_and_cradle(ctx):
    """OS 1.2 steps 5-8: 3 mango + 3 guava into cells, 1 papaya into the cradle; one per cell, stalk-down, 0
    fruit-on-fruit contacts, ballast = delivered sum, papaya placement <= 0.5 m/s."""
    t0 = time.perf_counter()
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    trunk = c.L.trunk
    sub0 = float(m.body_subtreemass[trunk])
    ids = ctx.demo_ids()
    assert len(ids) == 7
    times = ctx.place_all(ids)
    ctx.step(100)
    # one fruit per cell, stalk-down, papaya along x in the cradle
    assert c.cells[:6] == ids[:6] and c.cells[6:] == [None, None] and c.cradle == ids[6]
    bp, bq = c.L.basket_pose(d)
    Rb = kt._qmat(bq)
    for it in c.tray_items():
        s = ctx.pool.slots[it["slot"]]
        axis_w = np.array(d.xmat[s.body]).reshape(3, 3) @ np.array(it["stem_axis_body"])
        axis_b = Rb.T @ axis_w
        if it["where"] == "cradle":
            assert angle_deg(axis_b, [1, 0, 0]) < 1.0
            assert it["place_speed_mps"] <= kt.PAPAYA_V_MAX_MPS
        else:
            assert angle_deg(axis_b, [0, 0, -1]) < 1.0           # stalk-down
            loc = Rb.T @ (np.array(d.xpos[s.body]) - bp)
            cell = c.L.cell_pos[it["where"]]
            assert np.allclose(loc[:2], cell[:2], atol=2e-3), (it["fruit_id"], loc, cell)
            # nothing below it: the fruit bottom sits on the floor of its own cell
            hz = kt.ellipsoid_half_extents(kt._qmat(c._seat_pose(it)[1]), m.geom_size[s.fruit])[2]
            assert loc[2] - hz == pytest.approx(c.L.floor_top + kt.SEAT_GAP_M, abs=2e-3)
    # truth contact check: 0 fruit-on-fruit contacts, min gap reported
    assert c.fruit_contacts == 0 and c.tray_contact_pairs(d) == []
    gap = c.min_tray_gap_m(d)
    assert gap > 0.005, gap
    # ballast = delivered sum (+ the 1 g base), CoM at the seats, subtree mass follows
    delivered = sum(ctx.pool.fruit[f]["mass_kg"] for f in ids)
    assert c.payload_kg() == pytest.approx(delivered, abs=1e-6) and delivered <= kt.PAYLOAD_CAP_KG
    assert float(m.body_mass[c.L.ballast]) == pytest.approx(0.001 + delivered, abs=1e-6)
    assert float(m.body_subtreemass[trunk]) == pytest.approx(sub0 + delivered, abs=1e-6)
    com = sum(it["mass_kg"] * (c._seat_pose(it)[0] - c.L.ballast_pos) for it in c.tray_items()) / (0.001 + delivered)
    assert np.allclose(m.body_ipos[c.L.ballast], com, atol=1e-6)
    assert np.all(m.body_inertia[c.L.ballast] > 0)
    assert max(times) <= 2.5                        # each placement <= 2.5 s sim (from the PARK TCP, ~0.75 m up)
    assert c.max_papaya_speed_mps <= kt.PAPAYA_V_MAX_MPS and c.max_speed_mps <= kt.V_CARRY_MPS + 1e-9
    print(f"WP3 tray: 7 items, per-item {min(times):.2f}-{max(times):.2f} s sim, payload {delivered:.4f} kg, "
          f"min fruit-fruit gap {1000 * gap:.1f} mm, contacts {c.fruit_contacts}, wall {time.perf_counter() - t0:.1f} s")
    assert time.perf_counter() - t0 < 10.0


def test_one_papaya_per_trip_and_3kg_cap(ctx):
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    ids = ctx.demo_ids()
    ctx.place_all(ids)
    # a second papaya: refused (one per trip)
    fr = ctx.pool.fruit
    pap2 = next(f for f, v in fr.items() if v["crop"] == "papaya" and v["tree_id"] == "R3-2" and v["in_band"]
                and v["harvest_stage"])
    ctx.handoff(pap2)
    g = c.place(d, pap2, "auto", ctx.inputs())
    assert not g and g.reason == "payload_full"
    # one more mango would exceed 3 kg (2.99 kg aboard): refused, nothing written
    mango = next(f for f, v in fr.items() if v["crop"] == "mango" and v["tree_id"] == "R1-1" and v["in_band"]
                 and v["harvest_stage"])
    ctx.handoff(mango)
    before = float(m.body_mass[c.L.ballast])
    g = c.place(d, mango, "auto", ctx.inputs())
    assert not g and g.reason == "payload_full" and c.payload_kg() + fr[mango]["mass_kg"] > kt.PAYLOAD_CAP_KG
    assert float(m.body_mass[c.L.ballast]) == before and mango not in c.items
    # an occupied cell is refused
    g = c.place(d, mango, 0, ctx.inputs())
    assert not g and g.reason == "payload_full" and "cell 0" in c.events[-1]["detail"]


def test_refusals_attached_unknown_busy(ctx):
    d, c = ctx.d, ctx.ctrl
    fid = ctx.demo_ids()[0]
    slot = ctx.pool.assign(d, fid)                     # still on the tree (equality on)
    g = c.place(d, fid, "auto", ctx.inputs())
    assert not g and g.reason == "still_attached"
    g = c.place(d, "OT-1:f99", "auto", ctx.inputs())
    assert not g and g.reason == "unknown_item"
    ctx.pool.release(d, slot)
    hold_at_catcher(c, ctx.m, d, slot)
    assert c.place(d, fid, "auto", ctx.inputs())
    other = ctx.demo_ids()[1]
    ctx.handoff(other)
    g = c.place(d, other, "auto", ctx.inputs())
    assert not g and g.reason == "busy"
    assert c.refusals == 3
    g = c.unload(d, ctx.inputs())
    assert not g and g.reason == "busy"


def test_transfer_only_with_arm_park_and_gait_stopped(ctx):
    """MISSION-PRESETS 9 interlock: refused unless the arm is at PARK and the gait stopped; mid-transfer the
    arm leaving PARK or the trunk moving pauses the item in place, and it resumes by itself."""
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    fid = ctx.demo_ids()[0]
    ctx.handoff(fid)
    L = c.L
    # arm off PARK (shoulder +0.2 rad): refused
    qs = L.park_qadr[2]
    d.qpos[qs] += 0.2
    assert not L.arm_parked(d)
    g = c.place(d, fid, "auto", ctx.inputs())
    assert not g and g.reason == "arm_not_parked"
    d.qpos[qs] -= 0.2
    # trunk moving at 0.1 m/s: refused
    d.qvel[L.trunk_dadr] = 0.1
    assert not L.gait_stopped(d)
    g = c.place(d, fid, "auto", ctx.inputs())
    assert not g and g.reason == "gait_moving"
    d.qvel[L.trunk_dadr] = 0.0
    assert c.place(d, fid, "auto", ctx.inputs())
    ctx.step(100)
    # the gait starts mid-transfer: paused, path progress frozen
    s0 = c.job["s"]
    ctx.bus.values["hold"] = None

    def push(cx):
        cx.d.qvel[L.trunk_dadr] = 0.1
    ctx.step(50, push)
    assert c.paused == "gait_moving"
    assert c.job["s"] - s0 <= 1.5 * m.opt.timestep    # interlocks every physics step: at most one step on
    s1 = c.job["s"]
    ctx.step(50, push)
    assert c.job["s"] == s1
    ctx.step(20)                                       # stopped again: resumes by itself
    assert c.paused is None and c.job["s"] > s1
    ctx.until_idle()
    assert c.items[fid]["state"] == "seated"
    ev = [e["event"] for e in c.events]
    assert "paused" in ev and "resumed" in ev


def test_estop_freezes_item_until_resume(ctx):
    d, c = ctx.d, ctx.ctrl
    fid = ctx.demo_ids()[6]                            # the papaya
    ctx.handoff(fid)
    assert c.place(d, fid, "auto", ctx.inputs())
    ctx.step(200)
    ctx.bus.values["hold"] = {"reason": "estop", "since": float(d.time)}
    ctx.step(1)
    assert c.paused == "estop"
    s = ctx.pool.slots[c.items[fid]["slot"]]
    p0 = np.array(d.qpos[s.qadr:s.qadr + 3])
    ctx.step(250)                                      # 0.5 s held: item stays in the air, not dropped
    assert np.linalg.norm(np.array(d.qpos[s.qadr:s.qadr + 3]) - p0) < 2e-3
    ctx.bus.values["hold"] = None
    ctx.step(50)
    assert c.paused == "estop"                         # latched: needs an explicit resume
    c.resume()
    ctx.until_idle()
    assert c.items[fid]["state"] == "seated" and c.cradle == fid


def test_operator_pause_resume_and_place_abort_is_atomic(ctx):
    d, c = ctx.d, ctx.ctrl
    fid = ctx.demo_ids()[0]
    ctx.handoff(fid)
    assert c.place(d, fid, "auto", ctx.inputs())
    ctx.step(50)
    c.pause()
    s = c.job["s"]
    ctx.step(100)
    assert c.job["s"] == s and c.busy()
    c.resume()
    c.abort(d)                                          # after the commit a place completes
    ctx.until_idle()
    assert c.items[fid]["state"] == "seated"


# ---------------------------------------------------------------------- unload
def test_unload_into_crate_with_per_tree_attribution(ctx):
    """OS 1.2 step 11 / O-20: every item in the crate AABB, impact < 2 m/s (papaya <= 0.5), per-tree kg; cell by
    cell, then the cradle; the ballast empties; <= 4 s sim per item (OS 1.2 budget 7 x 4 s)."""
    t0 = time.perf_counter()
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    ids = ctx.demo_ids()
    ctx.place_all(ids)
    # at home the crate is out of reach: refused
    g = c.unload(d, ctx.inputs())
    assert not g and g.reason == "crate_out_of_reach"
    ctx.park_by_crate()
    g = c.unload(d, ctx.inputs())
    assert g, (g.reason, g.values)
    assert g.values["reach_m"] <= kt.REACH_MAX_M
    sim_s = ctx.until_idle()
    summary = c.crate_summary()
    assert summary["n"] == 7 and summary["dropped"] == 0
    for fid in ids:
        it = c.items[fid]
        s = ctx.pool.slots[it["slot"]]
        assert it["state"] == "crate" and it["outcome"] == "delivered"
        assert c.in_crate(d, d.xpos[s.body])
        lim = kt.PAPAYA_IMPACT_MAX_MPS if it["crop"] == "papaya" else kt.IMPACT_MAX_MPS
        assert 0.0 < it["impact_mps"] <= lim and it["impact_ok"], it
        assert it["assisted"] == [kt.ASSISTED_TAG]
    # cell by cell, then the cradle
    order = [e["fruit_id"] for e in c.events if e["event"] == "delivered"]
    assert order == ids
    # per-tree attribution = the trees and masses the fruit came from
    want = {}
    for fid in ids:
        f = ctx.pool.fruit[fid]
        w = want.setdefault(f["tree_id"], {"n": 0, "kg": 0.0})
        w["n"] += 1
        w["kg"] += f["mass_kg"]
    got = summary["per_tree"]
    assert set(got) == set(want)
    for t, w in want.items():
        assert got[t]["n"] == w["n"] and got[t]["kg"] == pytest.approx(w["kg"], abs=1e-4)
    assert float(m.body_mass[c.L.ballast]) == pytest.approx(0.001, abs=1e-12)
    assert c.payload_kg() == 0.0 and c.cells == [None] * 8 and c.cradle is None
    assert sim_s / 7 <= 4.0, sim_s
    pap = c.items[ids[6]]
    print(f"WP3 unload: 7 items in {sim_s:.2f} s sim ({sim_s / 7:.2f} s/item), reach {g.values['reach_m']:.3f} m, "
          f"impact max {summary['max_impact_mps']:.3f} m/s (papaya {pap['impact_mps']:.3f}), "
          f"{summary['kg']:.4f} kg, wall {time.perf_counter() - t0:.1f} s")
    assert time.perf_counter() - t0 < 10.0


def test_unload_item_outside_crate_is_dropped_and_slots_recycle(ctx):
    d, c = ctx.d, ctx.ctrl
    ids = ctx.demo_ids()[:2]
    ctx.place_all(ids)
    ctx.park_by_crate()
    real = c.plan_crate

    def plan_outside(dd, order):
        spots = real(dd, order)
        _, _, h = c.crate_frame(dd)
        spots[order[1]] = [float(-h[0] - 0.10), 0.0, float(-h[2])]    # 10 cm beyond the crate's -x wall
        return spots
    c.plan_crate = plan_outside
    try:
        assert c.unload(d, ctx.inputs())
        ctx.until_idle()
    finally:
        del c.plan_crate
    assert c.items[ids[0]]["outcome"] == "delivered" and c.items[ids[1]]["outcome"] == "dropped"
    assert c.crate_summary()["n"] == 1 and c.crate_summary()["dropped"] == 1
    assert c.per_tree()["OT-1"]["dropped_n"] == 1
    # unloaded bodies give their pool slots back; attribution stays
    free0 = len(ctx.pool.free_slots("connect"))
    out = c.release_crate_slots(d)
    assert sorted(out) == sorted(ids) and len(ctx.pool.free_slots("connect")) == free0 + 2
    assert all(f in ctx.pool.harvested for f in ids)
    assert c.crate_summary()["n"] == 1 and c.release_crate_slots(d) == []


def test_unload_abort_returns_item_to_its_cell(ctx):
    d, c = ctx.d, ctx.ctrl
    ids = ctx.demo_ids()[:2]
    ctx.place_all(ids)
    ctx.park_by_crate()
    assert c.unload(d, ctx.inputs())
    ctx.step(250)                                     # first item in the air
    assert c.job["op"] == "unload" and c.job["phase"] == "carry"
    c.abort(d)
    ctx.until_idle()
    assert c.cells[:2] == ids and all(c.items[f]["state"] == "seated" for f in ids)
    assert c.crate_summary()["n"] == 0 and c.fruit_contacts == 0
    # the aborted plan holds no crate spot: a crate with exactly as many spots as tray items takes the retry
    assert c.crate_spots == {}
    real = c.crate_spots_local

    def two_spots(dd, side=1.0):
        small, papaya = real(dd, side)
        return small[:2], papaya
    c.crate_spots_local = two_spots
    try:
        g = c.unload(d, ctx.inputs())
    finally:
        del c.crate_spots_local
    assert g, (g.reason, g.values)


def test_unload_refusals(ctx):
    d, c = ctx.d, ctx.ctrl
    g = c.unload(d, ctx.inputs())
    assert not g and g.reason == "empty"
    g = c.unload(d, kt.GateInputs(gait_stopped=False))
    assert not g and g.reason == "gait_moving"


def test_crown_stowed_reads_the_mast_state(model):
    class B:
        def __init__(self, mast):
            self.values = {"hold": None} if mast is None else {"hold": None, "mast": mast}
    assert ht.crown_stowed(B(None))                                  # no mast hook loaded
    assert ht.crown_stowed(B({"ext_m": 0.0, "has_mast": False}))     # model without the mast
    assert ht.crown_stowed(B({"ext_m": 0.0, "has_mast": True}))
    assert ht.crown_stowed(B({"ext_m": kt.CROWN_STOWED_EXT_M, "has_mast": True}))
    assert not ht.crown_stowed(B({"ext_m": 0.60, "has_mast": True}))  # mid band
    for bad in (None, "0", float("nan"), True):                       # malformed: fail safe
        assert not ht.crown_stowed(B({"ext_m": bad, "has_mast": True}))

    # fail closed without a mast state (mast hook not ticked yet): stowed only in a model without the mast
    m_plain = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    assert not ht.crown_stowed(B(None), model)
    assert ht.crown_stowed(B(None), m_plain)


def test_crown_threshold_agrees_with_wp13(model):
    """One crown-stowed threshold with WP13: a crown WP13 counts as stowed (it parks and calls unload) is never
    refused ``crown_gate`` here, and the other way round (the 5-20 mm band of the old 5 mm threshold included)."""
    from atharv_nandi.core import dock_logic

    class Drv:
        def __init__(self, ext):
            self.m = model
            self.bus = hook_registry.Bus()
            self.bus.values["mast"] = {"ext_m": ext, "has_mast": True}
    assert kt.CROWN_STOWED_EXT_M == dock_logic.CROWN_STOWED_EXT_M
    for ext in (0.0, 0.005, 0.012, dock_logic.CROWN_STOWED_EXT_M, 0.021, 0.60):
        drv = Drv(ext)
        assert ht.crown_stowed(drv.bus, model) == dock_logic.crown_stowed(drv), ext
    drv = Drv(0.0)
    del drv.bus.values["mast"]
    assert ht.crown_stowed(drv.bus, model) == dock_logic.crown_stowed(drv) is False


def test_unload_needs_the_crown_stowed(ctx):
    """MISSION-PRESETS 9 precondition: crown stowed. With the crown at the mid band the unload is refused
    ``crown_gate`` and nothing changes; stowed, the same unload runs and delivers."""
    d, c = ctx.d, ctx.ctrl
    fid = ctx.demo_ids()[0]
    ctx.place_all([fid])
    ctx.park_by_crate()
    ctx.bus.values["mast"] = {"ext_m": 0.60, "has_mast": True}
    inp = ctx.inputs()
    n_ref = c.refusals
    assert not inp.crown_stowed
    g = c.unload(d, inp)
    assert not g and g.reason == "crown_gate"
    assert not c.busy() and not c.unloading and c.queue == [] and c.crate_spots == {}
    assert c.items[fid]["state"] == "seated" and c.refusals == n_ref + 1
    ctx.bus.values["mast"] = {"ext_m": 0.0, "has_mast": True}
    assert c.unload(d, ctx.inputs())
    ctx.until_idle()
    assert c.items[fid]["state"] == "crate" and c.crate_summary()["n"] == 1


def test_crown_raised_mid_unload_pauses_until_stowed(ctx):
    """MISSION-PRESETS 9: no arm swing over the crate with the crown up. A crown raised during the unload pauses it
    (``crown_gate``, the item frozen in the air, never dropped); stowed again, it resumes by itself and delivers."""
    d, c = ctx.d, ctx.ctrl
    ids = ctx.demo_ids()[:2]
    ctx.place_all(ids)
    ctx.park_by_crate()
    assert c.unload(d, ctx.inputs())
    ctx.step(250)                                     # first item in the air
    assert c.job and c.job["op"] == "unload" and c.job["phase"] == "carry" and c.paused is None
    ctx.bus.values["mast"] = {"ext_m": 0.60, "has_mast": True}
    ctx.step(1)
    assert c.paused == "crown_gate"
    s0 = c.job["s"]
    ctx.step(200)
    assert c.paused == "crown_gate" and c.job["s"] == s0 and c.crate_summary()["n"] == 0
    kt.check_state_dict(c.to_dict(), n_slots=len(ctx.pool.slots))      # the pause survives a snapshot
    ctx.bus.values["mast"] = {"ext_m": 0.0, "has_mast": True}
    ctx.step(1)
    assert c.paused is None
    ctx.until_idle()
    assert [c.items[f]["state"] for f in ids] == ["crate", "crate"]
    assert c.crate_summary()["n"] == 2 and c.fruit_contacts == 0
    kinds = [(e["event"], e.get("reason")) for e in c.events]
    assert ("paused", "crown_gate") in kinds and ("resumed", "crown_gate") in kinds


def test_mast_raise_gate_while_unloading(ctx):
    """WP24 raise gate registered by the transfer hook: raises refused while an unload runs, lowering always
    allowed, raises allowed again when the unload is done."""
    from atharv_nandi.sim.hooks import mast as hm
    d, c, m = ctx.d, ctx.ctrl, ctx.m
    assert ht._mast_gate in hm._GATES
    assert ht._mast_gate(0.0, 0.60, {"m": m})                          # idle
    ctx.place_all(ctx.demo_ids()[:1])
    ctx.park_by_crate()
    assert c.unload(d, ctx.inputs())
    g = ht._mast_gate(0.0, 0.60, {"m": m})
    assert not g and g.reason == "transfer_unloading"
    assert ht._mast_gate(0.60, 0.0, {"m": m})                          # lowering
    ctx.until_idle()
    assert ht._mast_gate(0.0, 0.60, {"m": m})


def test_mass_matrix_follows_the_released_fruit(ctx):
    """WP23 change request worked around: the falling papaya accelerates at g, not at m_fruit / m_template x g."""
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    fid = ctx.demo_ids()[6]
    ctx.place_all([fid])
    ctx.park_by_crate()
    assert c.unload(d, ctx.inputs())
    acc = []

    def watch(cx):
        j = cx.ctrl.job
        if j and j["phase"] == "fall" and not j["touched"]:
            acc.append(float(cx.d.qacc[ctx.pool.slots[c.items[fid]["slot"]].dadr + 2]))
    ctx.until_idle(cb=watch)
    assert acc and abs(acc[0] + 9.81) < 0.05, acc[:3]


# ---------------------------------------------------------------------- walking, hook, state
def test_riders_stay_in_their_cells_while_walking(model):
    """Seated fruit ride the tray while the robot walks 0.4 m (SimDriver gait, the transfer hooks loaded)."""
    t0 = time.perf_counter()
    drv = SimDriver(model, hooks=hook_registry.load_all(names=["transfer"]))
    c = ht.controller_for(model)
    c.reset(drv.d)
    c.pool.reset(drv.d)
    try:
        drv.stand(0.5)
        ids = demo_ids(c.pool)[:3]
        for fid in ids:
            slot = c.pool.assign(drv.d, fid)
            c.pool.release(drv.d, slot)
            hold_at_catcher(c, model, drv.d, slot)
            assert c.place(drv.d, fid, "auto", kt.GateInputs())
            drv.run_until(lambda dr: not c.busy(), 5.0)
        err = []

        def check(dr):
            if c.steps % kt.RIDER_EVERY == 0:
                bp, bq = c.L.basket_pose(dr.d)
                for it in c.tray_items():
                    s = c.pool.slots[it["slot"]]
                    want = bp + kt._qmat(bq) @ c._seat_pose(it)[0]
                    err.append(float(np.linalg.norm(np.array(dr.d.qpos[s.qadr:s.qadr + 3]) - want)))
        drv.add_observer(check)
        x0 = float(drv.d.qpos[0])
        drv.walk(0.5, seconds=3.0)
        drv.stand(0.5)
        moved = float(drv.d.qpos[0]) - x0
        assert moved > 0.2, moved
        assert max(err) < 0.003, max(err)
        assert c.fruit_contacts == 0 and c.tray_contact_pairs(drv.d) == []
        assert float(drv.d.qpos[2]) > 0.25                         # standing, not fallen
        print(f"WP3 riders: walked {moved:.2f} m, max rider error {1000 * max(err):.2f} mm, "
              f"wall {time.perf_counter() - t0:.1f} s")
    finally:
        c.reset(drv.d)
        c.pool.reset(drv.d)
    assert time.perf_counter() - t0 < 10.0


def test_hook_commands_state_and_events(ctx):
    d, c, bus = ctx.d, ctx.ctrl, ctx.bus
    ctx.step(300)                                     # settled after the reset: no gait interlock
    fid = ctx.demo_ids()[0]
    ctx.handoff(fid)
    bus.deliver(ht.CMD_TOPIC, {"op": "place", "fruit_id": fid, "seq": 4})
    ctx.step(10)
    assert c.busy()
    bus.deliver(ht.CMD_TOPIC, {"op": "jump", "seq": 5})        # malformed: refused, never a fault
    ctx.step(10)
    ctx.until_idle()
    ctx.step(30)
    assert ctx.reg.faulted() == {}
    pub = bus.drain()
    states = [json.loads(p) for t, p in pub if t == ht.STATE_TOPIC]
    events = [json.loads(p) for t, p in pub if t == ht.EVENTS_TOPIC]
    assert states and states[-1]["cells"][0] == fid and states[-1]["payload_kg"] > 0
    assert bus.values["transfer"]["cells"][0] == fid
    kinds = [e["event"] for e in events]
    assert "place_start" in kinds and "seated" in kinds and "cmd_refused" in kinds
    assert any(e["event"] == "cmd" and e["cmd_seq"] == 4 and e["ok"] for e in events)
    assert any(e["event"] == "cmd_refused" and e["cmd_seq"] == 5 for e in events)
    assert [e["seq"] for e in events] == sorted({e["seq"] for e in events})   # each once, in order


def test_sim_state_provider_round_trip(ctx):
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    ids = ctx.demo_ids()
    ctx.place_all(ids[:2])
    ctx.handoff(ids[2])
    assert c.place(d, ids[2], "auto", ctx.inputs())
    ctx.step(200)                                     # mid-transfer
    sim_state.discover_providers()
    st = sim_state.pack(m, d, "w")
    msg = json.loads(sim_state.encode(st))
    blk = msg["mutations"]["providers"]["transfer"]
    assert blk["v"] == 1 and blk["data"]["state"]["job"]["phase"] == "carry"
    before = (json.dumps(c.to_dict(), sort_keys=True), json.dumps(c.ballast_fields()))
    # disturb, then restore
    c.reset(d)
    assert float(m.body_mass[c.L.ballast]) == pytest.approx(0.001)
    d2 = mujoco.MjData(m)
    ok, why = sim_state.apply(m, d2, msg, "w")
    assert ok, why
    assert (json.dumps(c.to_dict(), sort_keys=True), json.dumps(c.ballast_fields())) == before
    # a bad block is refused without writing
    bad = json.loads(json.dumps(msg))
    bad["mutations"]["providers"]["transfer"]["data"]["ballast"]["mass"] = 9.0
    ok, why = sim_state.apply(m, d2, bad, "w")
    assert not ok and "transfer" in why
    bad = json.loads(json.dumps(msg))
    bad["mutations"]["providers"]["transfer"]["data"]["state"]["cells"] = ["nope"] * 8
    assert not sim_state.apply(m, d2, bad, "w")[0]
    assert (json.dumps(c.to_dict(), sort_keys=True), json.dumps(c.ballast_fields())) == before
    # the restored controller finishes the transfer on the restored data
    d.qpos[:] = d2.qpos
    mujoco.mj_forward(m, d)
    ctx.until_idle()
    assert c.items[ids[2]]["state"] == "seated" and c.fruit_contacts == 0


def test_reset_restores_model_fields(ctx):
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    b = c.L.ballast
    base = (float(m.body_mass[b]), m.body_ipos[b].copy(), m.body_inertia[b].copy(), m.body_iquat[b].copy())
    sub = {a: float(m.body_subtreemass[a]) for a in c.L.ancestors}
    m0 = m.dof_M0.copy()
    ids = ctx.demo_ids()
    ctx.place_all([ids[0], ids[6]])
    ctx.park_by_crate()
    assert c.unload(d, ctx.inputs())
    ctx.until_idle()
    ctx.reg.reset(m, d)
    assert float(m.body_mass[b]) == base[0] and np.array_equal(m.body_ipos[b], base[1])
    assert np.array_equal(m.body_inertia[b], base[2]) and np.array_equal(m.body_iquat[b], base[3])
    assert all(float(m.body_subtreemass[a]) == v for a, v in sub.items())
    assert np.array_equal(m.dof_M0, m0)
    assert c.items == {} and not c.busy()


def test_legacy_world_is_untouched():
    with env(NANDI_WORLD="legacy"):
        m = farm_scene.load_farm_model()
    assert not ht.applies(m) and ht.controller_for(m) is None
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    reg = hook_registry.load_all(names=["transfer"])
    bus = hook_registry.Bus()
    q = d.qpos.copy()
    for _ in range(20):
        reg.substep(m, d, bus)
        reg.tick(m, d, bus)
    assert np.array_equal(d.qpos, q) and bus.drain() == [] and reg.faulted() == {}
    sim_state.discover_providers()
    assert "transfer" not in sim_state.pack_mutations(m)["providers"]


def test_hook_unload_pause_resume_abort_and_release_commands(ctx):
    d, c, bus = ctx.d, ctx.ctrl, ctx.bus
    ids = ctx.demo_ids()[:2]
    ctx.place_all(ids)
    ctx.park_by_crate()

    def cmd(op, n=2):
        bus.deliver(ht.CMD_TOPIC, {"op": op})
        ctx.step(n * 10)
    cmd("unload")
    assert c.busy() and c.unloading
    cmd("pause")
    s = c.job["s"]
    ctx.step(50)
    assert c.paused == "operator" and c.job["s"] == s
    cmd("resume")
    assert c.paused is None and c.job["s"] > s
    ctx.until_idle()
    assert c.crate_summary()["n"] == 2
    cmd("release_slots")
    assert all(c.items[f].get("recycled") for f in ids)
    cmd("abort")
    ev = [json.loads(p) for t, p in bus.drain() if t == ht.EVENTS_TOPIC]
    assert [e["op"] for e in ev if e["event"] == "cmd"] == ["unload", "pause", "resume", "release_slots", "abort"]
    ht._reset(None, None)                              # a reset without a model is a no-op


BAD_STATES = [
    lambda s: s.pop("v"),
    lambda s: s.update(v=2),
    lambda s: s.update(items=[]),
    lambda s: s["items"].update(x={"fruit_id": "y", "state": "seated", "slot": 0, "mass_kg": 0.1}),
    lambda s: s["items"].update(x={"fruit_id": "x", "state": "seated", "slot": True, "mass_kg": 0.1}),
    lambda s: s["items"].update(x={"fruit_id": "x", "state": "seated", "slot": 0, "mass_kg": -1}),
    lambda s: s.update(cells=[None] * 7),
    lambda s: s.update(cradle="nope"),
    lambda s: s.update(job={"fruit_id": "nope", "phase": "carry"}),
    lambda s: s.update(queue=["nope"]),
    lambda s: s.update(paused="sleepy"),
    lambda s: s.update(steps=-1),
    lambda s: s.update(max_speed_mps=float("nan")),
    lambda s: s.update(crate_spots=[]),
    lambda s: s.update(crate_spots={"nope": [0, 0, 0]}),
]


@pytest.mark.parametrize("k", range(len(BAD_STATES)))
def test_malformed_state_is_refused(ctx, k):
    st = json.loads(json.dumps(ctx.ctrl.to_dict()))
    BAD_STATES[k](st)
    with pytest.raises(ValueError):
        kt.check_state_dict(st)


def test_malformed_ballast_blocks_are_refused(ctx):
    m = ctx.m
    good = ht._pack(m)
    assert set(good) == {"ballast", "state", "m0"}
    for mutate in (lambda p: p.pop("state"), lambda p: p["ballast"].pop("iquat"),
                   lambda p: p["ballast"].update(mass=True), lambda p: p["ballast"].update(ipos=[0, 0]),
                   lambda p: p["ballast"].update(inertia=[0.0, 1e-7, 1e-7]),
                   lambda p: p["ballast"].update(iquat=[1, 0, 0, float("inf")]),
                   lambda p: p.update(m0=p["m0"][:-1]), lambda p: p["m0"][0].__setitem__(0, 0.0),
                   lambda p: p.update(extra=1)):
        p = json.loads(json.dumps(good))
        mutate(p)
        with pytest.raises(ValueError):
            ht._unpack(m, p)
    ht._unpack(m, {})                                 # empty block: nothing to do
    with env(NANDI_WORLD="legacy"):
        legacy = farm_scene.load_farm_model()
    assert ht._pack(legacy) == {}
    with pytest.raises(ValueError):
        ht._unpack(legacy, good)


def test_controller_without_pool_is_inert(model):
    c = kt.TransferController(model, None)
    d = mujoco.MjData(model)
    g = c.place(d, "OT-1:f01")
    assert not g and g.reason == "unknown_item"
    c.substep(d)
    assert c.release_crate_slots(d) == [] and not c.busy()


# ---------------------------------------------------------------------- review fixes
def test_place_refuses_fruit_not_at_the_catcher(ctx):
    """OS 1.2 step 11 (never faked): only a fruit held at the catcher next to the PARK TCP is carried; a released fruit
    lying anywhere else (on the tree 6.7 m away, beside the robot) is refused and nothing is written."""
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    fid = ctx.demo_ids()[0]
    slot = ctx.pool.assign(d, fid)
    ctx.pool.release(d, slot)                          # released, still where it hung
    mujoco.mj_forward(m, d)
    s = ctx.pool.slots[slot]
    flags = (int(m.geom_contype[s.fruit]), int(m.body_contype[s.body]))
    g = c.place(d, fid, "auto", ctx.inputs())
    assert not g and g.reason == "not_at_catcher" and g.values["dist_m"] > 1.0
    assert fid not in c.items and not c.busy() and (int(m.geom_contype[s.fruit]), int(m.body_contype[s.body])) == flags
    tcp = c.L.handoff_point(d)
    for off, ok in (((0.0, 0.0, -(kt.HANDOFF_RADIUS_M + 0.02)), False), ((0.0, 0.4, 0.0), False),
                    ((0.0, 0.0, -(kt.HANDOFF_RADIUS_M - 0.02)), True)):
        hold_at_catcher(c, m, d, slot, off)
        assert np.allclose(c.L.handoff_point(d), tcp)
        g = c.place(d, fid, "auto", ctx.inputs())
        assert bool(g) == ok, (off, g.reason, g.values)
    ctx.until_idle()
    assert c.items[fid]["state"] == "seated"


def test_wrong_target_has_its_own_reason(ctx):
    d, c = ctx.d, ctx.ctrl
    ids = ctx.demo_ids()
    mango, papaya = ids[0], ids[6]
    ctx.handoff(mango)
    g = c.place(d, mango, "cradle", ctx.inputs())
    assert not g and g.reason == "wrong_target"          # empty tray: not payload_full
    ctx.pool.reset(d)
    ctx.handoff(papaya)
    g = c.place(d, papaya, 3, ctx.inputs())
    assert not g and g.reason == "wrong_target"
    g = c.place(d, papaya, "basket", ctx.inputs())
    assert not g and g.reason == "wrong_target"
    assert "wrong_target" in kt.REASONS and "not_at_catcher" in kt.REASONS and c.items == {}


def test_hook_command_batch_and_lost_commands(ctx):
    """The hook inbox keeps the newest message per topic: a batch is applied in order in one tick, and two messages
    within one tick give a ``cmd_lost`` event instead of a silent loss."""
    c, bus = ctx.ctrl, ctx.bus
    ctx.step(1)
    bus.deliver(ht.CMD_TOPIC, [{"op": "pause", "seq": 1}, {"op": "resume", "seq": 2}, {"op": "jump", "seq": 3}])
    ctx.step(10)
    cmds = [(e["op"], e["cmd_seq"]) for e in c.events if e["event"] == "cmd"]
    assert cmds == [("pause", 1), ("resume", 2)]
    assert any(e["event"] == "cmd_refused" and e["cmd_seq"] == 3 for e in c.events)
    assert not any(e["event"] == "cmd_lost" for e in c.events)
    bus.deliver(ht.CMD_TOPIC, {"op": "pause", "seq": 4})
    bus.deliver(ht.CMD_TOPIC, {"op": "resume", "seq": 5})
    ctx.step(10)
    lost = [e for e in c.events if e["event"] == "cmd_lost"]
    assert len(lost) == 1 and lost[0]["n"] == 1
    assert [e["cmd_seq"] for e in c.events if e["event"] == "cmd"][-1] == 5
    for bad in ([], [{"op": "pause"}] * (ht.MAX_BATCH + 1)):
        bus.deliver(ht.CMD_TOPIC, bad)
        ctx.step(10)
        assert c.events[-1]["event"] == "cmd_refused"
    assert ctx.reg.faulted() == {} and c.paused is None


def _mid_transfer_state(ctx):
    ids = ctx.demo_ids()
    ctx.place_all(ids[:1])
    ctx.handoff(ids[1])
    assert ctx.ctrl.place(ctx.d, ids[1], "auto", ctx.inputs())
    ctx.step(100)
    return ids, json.loads(json.dumps(ctx.ctrl.to_dict()))


def test_unpack_refuses_states_that_would_fault_the_hooks(ctx):
    """The provider validates every field the hooks read: the review's payload (job without op / s, slot 99) and the
    other inconsistent shapes are refused, and nothing faults or is written."""
    m, c = ctx.m, ctx.ctrl
    ids, good = _mid_transfer_state(ctx)
    a, b = ids[0], ids[1]
    kt.check_state_dict(good, n_slots=len(ctx.pool.slots))
    payload = ht._pack(m)
    before = json.dumps(c.to_dict(), sort_keys=True)

    def job_without_op_and_s(s):
        s["job"] = {"fruit_id": b, "phase": "carry", "path": s["job"]["path"]}
    mutations = [
        job_without_op_and_s,
        lambda s: s["items"][a].update(slot=99),
        lambda s: s["job"].update(op="fly"),
        lambda s: s["job"].pop("s"),
        lambda s: s["job"].update(s=1e9),
        lambda s: s["job"].update(s=float("nan")),
        lambda s: s["job"].update(phase="fall"),
        lambda s: s["job"]["path"].update(v=[0.0, 0.0, 0.0]),
        lambda s: s["job"]["path"].update(p=[0, 0]),
        lambda s: s["items"][a].update(where=9),
        lambda s: s["items"][a].update(where="cradle"),
        lambda s: s["items"][a].update(crop=None),
        lambda s: s["items"][a].update(tree_id=3),
        lambda s: s["items"][a].update(stem_axis_body=[0, 0]),
        lambda s: s["items"][a].update(stem_axis_body=[0, 0, 0]),
        lambda s: s["items"][a].update(impact_mps="fast"),
        lambda s: s["items"][a].pop("stem_axis_body"),
        lambda s: s["items"][a].update(state="moving"),
        lambda s: s["items"][a].update(state="falling"),
        lambda s: s["items"][a].update(slot=s["items"][b]["slot"]),
        lambda s: s["items"][b].update(state="seated"),
        lambda s: s.update(cells=[b, a] + [None] * 6),
        lambda s: s.update(cells=[a, a] + [None] * 6),
        lambda s: s.update(cradle=a),
        lambda s: s.update(job=None),
        lambda s: s.update(queue=[b]),
        lambda s: s.update(queue=[a]),                   # queued for the crate without a crate spot
        lambda s: s.update(unloading="yes"),
        lambda s: s.update(contact_pairs=[[a]]),
    ]
    for k, mutate in enumerate(mutations):
        st = json.loads(json.dumps(good))
        mutate(st)
        with pytest.raises(ValueError):
            kt.check_state_dict(st, n_slots=len(ctx.pool.slots))
        p = json.loads(json.dumps(payload))
        p["state"] = st
        with pytest.raises(ValueError):
            ht._unpack(m, p)
        assert json.dumps(c.to_dict(), sort_keys=True) == before, k
    ctx.step(30)
    assert ctx.reg.faulted() == {}
    ctx.until_idle()
    assert c.items[b]["state"] == "seated"


def test_set_state_restores_the_pool_mass_matrix(ctx):
    """dof_M0 written for a released fruit is packed by the transfer provider: applying a snapshot taken before brings
    the slot's mass matrix back, so the replay is bit-exact."""
    m, d, c = ctx.m, ctx.d, ctx.ctrl
    pap = ctx.demo_ids()[6]
    slot = ctx.pool.assign(d, pap)                     # hanging (weld)
    mujoco.mj_forward(m, d)
    sim_state.discover_providers()
    msg = json.loads(sim_state.encode(sim_state.pack(m, d, "w")))
    blk = msg["mutations"]["providers"]["transfer"]["data"]
    assert len(blk["m0"]) == len(ctx.pool.slots)
    m0 = m.dof_M0.copy()

    def run(dd, n=400):
        for _ in range(n):
            dd.ctrl[:25] = m.key_ctrl[0][:25]
            ctx.reg.substep(m, dd, ctx.bus)
            mujoco.mj_step(m, dd)
            ctx.reg.tick(m, dd, ctx.bus)
        return dd.qpos.copy()
    da = mujoco.MjData(m)
    assert sim_state.apply(m, da, msg, "w")[0]
    traj_a = run(da)
    c._restore_m0(slot)       # perturb: the parked template (WP23 assign now writes the true dof_M0 itself)
    assert not np.array_equal(m.dof_M0, m0)
    db = mujoco.MjData(m)
    assert sim_state.apply(m, db, msg, "w")[0]
    assert np.array_equal(m.dof_M0, m0)
    assert np.array_equal(run(db), traj_a)
    # a block without m0 (older snapshot): parked slots get the template, assigned ones keep their true dof_M0
    c._restore_m0(slot)
    assert not np.array_equal(m.dof_M0, m0)
    p = json.loads(json.dumps(blk))
    p.pop("m0")
    ht._unpack(m, p)
    assert np.array_equal(m.dof_M0, m0)
