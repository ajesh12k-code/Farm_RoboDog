"""WP25 orbit planner and closed-loop side-mode controller (ORCHARD-MVP-SPEC 1.2 step 2, 5.1, 5.3; MISSION-PRESETS 2).

Unit tier: planner geometry (full / two-sided / blocked), the 0.3 m neighbour rule, transit around a row, the
controller on a kinematic hexapod model (lagging, biased gait, mast 0.30 m behind the trunk). ``slow``: the real
closed loop on the orchard_demo OT-1 orbit with the ``ot1_orbit`` flat zone removed (rough noise 1 cm + basins),
WP5 low-plane trunk fit, seeds {7, 11, 23}: radius RMS (truth) <= 0.10 m (O-02).
"""
import json
import math
import os

import pytest

from atharv_nandi.core import orbit as ob

OT1 = {"tree_id": "OT-1", "xy": (6.0, -3.5), "canopy_r_m": 1.0, "orbit_clear": True, "alley_sides": ["all"]}


def _row(n=4, x0=2.0, dx=2.5, y=2.5, cr=1.0):
    return [{"tree_id": f"R-{k}", "xy": (x0 + dx * k, y), "canopy_r_m": cr} for k in range(n)]


# ------------------------------------------------------------------------------------------- planner
def test_orbit_radius_clamp():
    assert ob.orbit_radius(1.0, 0.9) == pytest.approx(1.9)       # OS 1.2 step 2: r1 = 1.9 m on OT-1
    assert ob.orbit_radius(0.1, 0.9) == pytest.approx(1.2)
    assert ob.orbit_radius(2.0, 1.5) == pytest.approx(3.0)


def test_full_orbit_plan():
    p = ob.plan_orbit(OT1, [], 0.9, 8)
    assert p.mode == "full" and p.gap_deg == 0.0
    sts = p.stations[0]
    assert len(sts) == 9 and p.n_capture == 8 and not sts[-1].capture        # 8 stations + the closing point
    for a, b in zip(sts, sts[1:]):
        assert b.theta - a.theta == pytest.approx(math.radians(45))
    for s in sts:
        assert math.hypot(s.xy[0] - 6.0, s.xy[1] + 3.5) == pytest.approx(1.9)
        # CCW with the tree on the left: heading = tangent turned in by the crab atan(0.30 / R)
        assert ob.wrap(s.heading - (s.theta + math.pi / 2)) == pytest.approx(math.atan2(0.30, 1.9), abs=1e-9)
    assert p.arc_length_m() == pytest.approx(2 * math.pi * 1.9)
    e = ob.entry_pose(p)
    assert math.hypot(e["xy"][0] - sts[0].trunk_xy()[0], e["xy"][1] - sts[0].trunk_xy()[1]) < 1e-9


def test_two_sided_hedgerow_arcs_keep_0p3_m_from_neighbours():
    """OS 5.3 "arc blocked": the arcs stop where a neighbour canopy comes within 0.3 m of the hull."""
    row = _row()
    tree = dict(row[1], alley_sides=["-y", "+y"], orbit_clear=False)
    p = ob.plan_orbit(tree, [row[0], row[2]], 0.9, 8)
    assert p.mode == "two_sided" and p.reason == "orbit_blocked" and len(p.arcs) == 2
    assert sorted(p.blocked_by) == ["R-0", "R-2"]
    assert {s.side for arc in p.stations for s in arc} == {"-y", "+y"}
    for a, b in p.arcs:
        for i in range(int(math.degrees(b - a)) + 1):
            th = a + math.radians(i)
            x, y = ob.point_on_orbit(p.centre, p.R, th)
            for nb in (row[0], row[2]):
                gap = math.hypot(x - nb["xy"][0], y - nb["xy"][1]) - nb["canopy_r_m"] - ob.HULL_R_M
                assert gap >= ob.ARC_CLEAR_M - 1e-6, (math.degrees(th), gap)
    assert 0.0 < p.gap_deg < 360.0
    assert p.gap_deg == pytest.approx(360.0 - sum(math.degrees(b - a) for a, b in p.arcs))


def test_two_sided_explicit_on_a_clear_tree_and_full_refused_when_blocked():
    p = ob.plan_orbit(dict(OT1, alley_sides=["-y", "+y"]), [], 0.9, 8, arc_mode="two_sided")
    assert p.mode == "two_sided" and len(p.arcs) == 2 and p.reason == ""
    row = _row()
    blocked = ob.plan_orbit(dict(row[1], orbit_clear=True), [row[0], row[2]], 0.9, 8, arc_mode="full")
    assert blocked.mode == "two_sided"                     # coverage is measured, never assumed (OS rule 12)


def test_blocked_everywhere_and_geofence():
    ring = [{"tree_id": f"N{k}", "xy": (2.0 * math.cos(k * math.pi / 3), 2.0 * math.sin(k * math.pi / 3)),
             "canopy_r_m": 1.0} for k in range(6)]
    p = ob.plan_orbit({"tree_id": "C", "xy": (0.0, 0.0), "canopy_r_m": 1.0}, ring, 0.9, 8)
    assert p.mode == "blocked" and p.reason == "orbit_blocked" and p.gap_deg == 360.0
    fence = [[-10, -10], [10, -10], [10, 1.0], [-10, 1.0]]          # cuts the +y side of the annulus
    q = ob.plan_orbit({"tree_id": "C", "xy": (0.0, 0.0), "canopy_r_m": 1.0, "orbit_clear": True}, [], 0.9, 8,
                      geofence=fence)
    assert q.mode == "two_sided" and "geofence" in q.blocked_by
    for arc in q.stations:
        for s in arc:
            assert s.xy[1] < 1.0


def test_param_ranges():
    with pytest.raises(ValueError):
        ob.plan_orbit(OT1, [], 0.9, 3)
    with pytest.raises(ValueError):
        ob.plan_orbit(OT1, [], 0.9, 8, arc_mode="spiral")


def test_free_arcs_and_side_arcs():
    blocked = [False] * 360
    for i in list(range(0, 40)) + list(range(170, 200)):
        blocked[i] = True
    arcs = ob.free_arcs(blocked)
    assert arcs[0][1] - arcs[0][0] > arcs[1][1] - arcs[1][0]
    sa = ob.side_arcs(blocked, ["-y", "+y"])
    assert [s for *_, s in sa] == ["+y", "-y"]
    a, b, _ = sa[0]
    assert a >= 40 and b <= 169


def test_registry_helpers_and_lidar_annulus():
    reg = [{"tree_id": "A", "x": 0.0, "y": 0.0, "canopy_r_m": 1.0, "basin_r_m": 0.8},
           {"tree_id": "B", "geo": {"x": 2.5, "y": 0.0}, "canopy_r_m": 0.9},
           {"tree_id": "C", "x": 30.0, "y": 0.0, "canopy_r_m": 0.9}]
    assert [n["tree_id"] for n in ob.neighbours_of(reg[0], reg)] == ["B"]
    t = ob.tree_of(reg[1])
    assert t["xy"] == (2.5, 0.0) and t["alley_sides"] == ["-y", "+y"]
    assert ob.basins_of(reg) == [(0.0, 0.0, 0.8)]
    cl = [{"lo": [1.8, -0.1, 0.0], "hi": [2.0, 0.1, 1.0]}, {"lo": [8.0, 8.0, 0], "hi": [8.2, 8.2, 1]}, {"bad": 1}]
    near = ob.clusters_as_neighbours(cl, (0.0, 0.0), 1.9)
    assert len(near) == 1 and near[0]["tree_id"] == "lidar_0"
    p = ob.plan_orbit({"tree_id": "A", "xy": (0, 0), "canopy_r_m": 1.0, "orbit_clear": True}, near, 0.9, 8)
    assert p.mode == "two_sided"                          # MP 2: annulus clear in map AND LiDAR


def test_crown_path_basin_gate():
    p = ob.plan_orbit(OT1, [], 0.9, 8)
    assert ob.crown_path_ok(p, 1.6, [(6.0, -3.5, 0.8)]).allowed        # OT-1: hull 1.9 m out, rim 1.22 m
    assert not ob.crown_path_ok(p, 1.6, [(6.0, -3.5, 1.6)]).allowed    # a wide basin: stop-and-go
    assert ob.crown_path_ok(p, 0.0, [(6.0, -3.5, 1.6)]).allowed        # stowed walking is not gated


def test_transit_path_goes_around_the_row():
    row = _row(n=4)
    obst = [(t["xy"][0], t["xy"][1], t["canopy_r_m"]) for t in row]
    start, goal = (4.5, 2.5 + 1.9), (4.5, 2.5 - 1.9)
    fence = [[-6, -6], [16, -6], [16, 12], [-6, 12]]
    path = ob.transit_path(start, goal, obst, fence)
    assert path is not None and path[-1] == goal
    pts = [start] + path
    for p, q in zip(pts, pts[1:]):
        for i in range(21):
            x, y = p[0] + (q[0] - p[0]) * i / 20, p[1] + (q[1] - p[1]) * i / 20
            for ox, oy, r in obst:
                assert math.hypot(x - ox, y - oy) >= r + ob.HULL_R_M + 0.10 - 0.02
    L = ob.path_length(start, path)
    assert 7.0 < L < 25.0
    assert ob.transit_path((0, 5), (0, 6), obst, fence) == [(0.0, 6.0)]           # straight when clear
    assert ob.transit_path(start, goal, obst, [[3, 3], [6, 3], [6, 6], [3, 6]]) is None   # fenced in


def test_arc_poses_base_frame():
    p = ob.plan_orbit(OT1, [], 0.9, 8)
    s0, s1 = p.stations[0][0], p.stations[0][1]
    arc = ob.arc_poses_base(s0.xy, s0.heading, p.centre, p.R, s0.theta, s1.theta)
    assert arc[0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    assert arc[-1][1] > 0.2 and arc[-1][2] == pytest.approx(math.radians(45), abs=1e-3)   # curving left


# ------------------------------------------------------------------------------------------- controller
def test_speed_rules():
    assert ob.speed_for(0.05) == ob.V_ORBIT_MPS
    assert ob.speed_for(0.2) == pytest.approx(0.10)        # > 0.15 m: slow to 0.10 m/s
    assert ob.speed_for(-0.31) == 0.0                      # > 0.30 m: stop and re-enter
    assert abs(ob.omega_cmd(0.15, 1.9, 5.0, 0.0)) == ob.OMEGA_MAX


def _kinematic_orbit(R0=1.9, start_err=0.0, v_bias=0.85, w_bias=0.9, w_drift=-0.007, lag=0.3, T=85.0):
    """Hexapod stand-in: unicycle trunk, mast reference 0.30 m behind, first-order lag on the commands, the gait
    under-delivering speed and yaw by the given factors plus a yaw drift (the MEASURED -0.007 rad/s)."""
    p = ob.plan_orbit(OT1, [], 0.9, 8)
    s0 = p.stations[0][0]
    h = s0.heading
    tx, ty = ob.trunk_from_ref(ob.point_on_orbit(p.centre, p.R + start_err, s0.theta), h)
    ctl = ob.OrbitController(p.centre, p.R)
    ctl.start(s0.theta, s0.theta + 2 * math.pi)
    v = w = 0.0
    dt, t = 0.02, 0.0
    steps = []
    while t < T:
        ref = ob.ref_point((tx, ty), h)
        st = ctl.step(ref, h, dt, v_meas=v, w_meas=w)
        steps.append(st)
        if st.done or st.stop:
            break
        v_t = st.vx * ob.V_PER_CMD * v_bias
        w_t = st.yaw * ob.W_PER_CMD * w_bias + w_drift
        v += (v_t - v) * dt / lag
        w += (w_t - w) * dt / lag
        tx += v * math.cos(h) * dt
        ty += v * math.sin(h) * dt
        h = ob.wrap(h + w * dt)
        t += dt
    return ctl, steps, t


def test_closed_loop_holds_radius_on_a_kinematic_model():
    ctl, steps, t = _kinematic_orbit()
    errs = [s.e_r for s in steps]
    assert steps[-1].done
    assert ob.rms(errs[250:]) < 0.03            # integrators absorb the 15 % speed / 10 % yaw shortfall
    assert max(abs(e) for e in errs) < 0.10
    assert 75.0 < t < 100.0                     # ~2 pi 1.9 / 0.15 = 80 s
    assert ctl.gait.iv > 0 and ctl.gait.iw > 0


def test_closed_loop_recovers_from_an_offset_and_slows():
    ctl, steps, _ = _kinematic_orbit(start_err=0.2, T=40.0)
    assert steps[0].v == pytest.approx(0.10)    # > 0.15 m out: 0.10 m/s
    assert abs(steps[-1].e_r) < 0.05
    _, steps2, _ = _kinematic_orbit(start_err=0.35, T=5.0)
    assert steps2[0].stop and steps2[0].vx == 0.0


def test_crab_feedforward_removes_the_bias():
    """Without the atan(mast_x / R) crab term the 0.30 m mast offset biases the radius ~+0.05-0.07 m (MEASURED
    0.05 m on orchard_demo before the fix); with it the steady error is ~0."""
    ctl, steps, _ = _kinematic_orbit(v_bias=1.0, w_bias=1.0, w_drift=0.0)
    steady = [s.e_r for s in steps[1500:]]
    assert abs(sum(steady) / len(steady)) < 0.01


def test_nearest_station():
    p = ob.plan_orbit(OT1, [], 0.9, 8)
    sts = p.all_stations()
    assert ob.nearest_station(sts, sts[3].theta + 0.1).k == 3
    assert ob.nearest_station(sts, sts[3].theta + 0.1, after_k=3).k == 4


# ------------------------------------------------------------------------------------------- slow: real loop
def rough_world(seed: int, tmp_path):
    """orchard_demo with the ``ot1_orbit`` flat zone removed: the OT-1 annulus gets the 1 cm rough noise."""
    from atharv_nandi.core import farm_scene
    from atharv_nandi.sim.world import compose, trees
    field = json.load(open(os.path.join(compose.FIELDS_DIR, "orchard_demo.json")))
    zones = field["terrain"]["noise"]["flat_zones"]
    field["terrain"]["noise"]["flat_zones"] = [z for z in zones if z.get("id") != "ot1_orbit"]
    fdir = str(tmp_path / "fields")
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, "orchard_demo.json"), "w") as fh:
        json.dump(field, fh)
    cfg = farm_scene.world_config()
    w = compose.build("orchard_demo", seed, farm_scene.ROBOT_XML, cfg, fields_dir=fdir)
    f2 = compose.load_field("orchard_demo", seed, fdir)
    return w.spec.compile(), f2, trees.registry(f2)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [7, 11, 23])
def test_orbit_radius_rms_rough_basins(seed, tmp_path, monkeypatch):
    """O-02: radius RMS (truth) <= 0.10 m over a full OT-1 orbit on rough ground with basins, physics mode (R_meas
    from the WP5 low-plane trunk fit), 0 falls. MEASURED 2026-11-18 on the capped runner: 0.027 / 0.027 / 0.027 m."""
    monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
    monkeypatch.setenv("NANDI_ACTORS", "0")
    monkeypatch.setenv("NANDI_SEED", str(seed))
    import atharv_nandi.sim.hooks as hr
    from atharv_nandi.core import sim_driver as sd
    from atharv_nandi.sim.hooks import tree360 as h3
    m, field, reg = rough_world(seed, tmp_path)
    drv = sd.SimDriver(m, hooks=hr.load_all(names=["mast", "stability", "lidar", "tree360"]))
    drv.wait(0.3)
    run = h3.start(drv.bus, drv.m, drv.d, "OT-1",
                   {"entry": "teleport", "capture": "geometry", "measure": "physics", "out_root": str(tmp_path),
                    "run_id": f"rough-{seed}", "mp4": False}, gait=drv.gait, registry=reg, field=field)
    while not run.finished and drv.t < 240.0:
        drv.wait(1.0)
    res = run.result
    k = res["kpi"]
    print(f"seed {seed}: " + json.dumps({x: k[x] for x in ("radius_rms_truth_m", "radius_max_truth_m",
                                                            "radius_rms_ctrl_m", "orbit_s", "fit_methods")}))
    assert res["status"] == "done", res["abort_reason"]
    assert not drv.hook_faults
    assert k["radius_rms_truth_m"] <= 0.10
    assert k["falls"] == 0 and k["max_tilt_deg"] < 10.0
    assert k["fit_methods"].get("registry_rtk", 0) == 0          # the trunk fit carried the whole orbit
