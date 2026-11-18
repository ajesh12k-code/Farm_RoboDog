"""WP5 mast LiDAR, low safety planes, ray masks, trunk fit, clusters and the lidar hook (MVP-SPEC 2.4, 5.4;
ORCHARD-MVP-SPEC 2.5, 6). ROS-free, no GL."""
import json
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene, lidar_model as lm
from test_pano import place_person, place_robot, world_env

TREE = "tree_L1-1"          # orchard_lite mango, trunk r 0.055 m, trunk height 0.7 m, scaffolds from ~0.6 m


@pytest.fixture(scope="module")
def lite():
    with world_env("orchard_lite"):
        m = farm_scene.load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


# ------------------------------------------------------------------------------------------- ray accuracy, masks
def _box_world():
    xml = """<mujoco><worldbody>
      <geom name="floor" type="plane" size="20 20 0.1" group="0"/>
      <geom name="wall" type="box" pos="3 0 1" size="0.05 4 1" group="0"/>
      <geom name="leaf" type="box" pos="1.5 0 1" size="0.02 0.5 1" group="3"/>
      <geom name="robotish" type="box" pos="-1.5 0 1" size="0.05 4 1" group="2"/>
      <body name="trunk" pos="0 0 0.34"><freejoint/>
        <geom type="box" size="0.34 0.14 0.05" group="2" mass="5"/>
        <site name="mast_lidar" pos="-0.30 0 0.645"/>
      </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_multiray_range_within_1cm_and_robot_excluded():
    m, d = _box_world()
    L = lm.LidarModel(m)
    assert L.n_rays == 3 * 360 + 2 * 270              # 1,620 rays (MVP-SPEC 3.8)
    sc = L.sweep(d, lm.SAFETY_GROUPS, noise=False)
    s = sc["mast"]
    org = s.origin
    k = int(np.argmin(np.abs(s.angles)) )             # 0 deg azimuth, 0 deg beam: straight at the wall (x 2.95)
    sel = np.nonzero((s.beams == 0.0))[0]
    k = sel[int(np.argmin(np.abs(s.angles[sel])))]
    assert abs(s.ranges[k] - (2.95 - org[0])) < 0.01
    # the leaf (group 3) is invisible to the safety mask, hit by the coverage mask
    cov = L.sweep(d, lm.COVERAGE_GROUPS, noise=False)["mast"]
    assert abs(cov.ranges[k] - (1.48 - org[0])) < 0.01
    assert m.geom_group[cov.geom[k]] == 3 and m.geom_group[s.geom[k]] == 0
    # group 2 (robot) is never hit: the rear ray passes the robot-ish wall at x -1.5 and finds nothing
    k180 = sel[int(np.argmin(np.abs(np.abs(s.angles[sel]) - math.pi)))]
    assert not np.isfinite(s.ranges[k180])
    hit = s.geom[s.valid]
    assert not np.any(m.geom_group[hit] == 2)
    # the -30 deg beam meets the ground at about lens_z / tan(30)
    sel30 = np.nonzero(s.beams == -30.0)[0]
    k30 = sel30[int(np.argmin(np.abs(np.abs(s.angles[sel30]) - math.pi)))]
    assert abs(s.ranges[k30] * math.cos(math.radians(30)) - org[2] / math.tan(math.radians(30))) < 0.01
    # low planes at 0.18 m world height
    for name in ("low_fl", "low_rr"):
        assert abs(sc[name].origin[2] - lm.LOW_PLANE_Z_M) < 1e-9


def test_noise_and_min_range():
    m, d = _box_world()
    L = lm.LidarModel(m, seed=11)
    a = L.sweep(d, noise=False)["mast"].ranges
    b = L.sweep(d, noise=True)["mast"].ranges
    fin = np.isfinite(a) & np.isfinite(b)
    err = b[fin] - a[fin]
    assert 0.012 < float(np.std(err)) < 0.03          # sigma 2 cm
    assert np.all(~np.isfinite(b) | (b >= lm.RANGE_MIN_M))


def test_robot_geoms_never_hit_in_the_orchard(lite):
    m, d = lite
    L = lm.LidarModel(m)
    for groups in (lm.SAFETY_GROUPS, lm.COVERAGE_GROUPS):
        sc = L.sweep(d, groups, noise=False)
        for s in sc.values():
            g = s.hit_geoms()
            assert not np.any(np.isin(m.geom_group[g], (2, 5))), s.name
    # safety rays never see foliage, coverage rays do somewhere near a tree
    tc = d.xpos[m.body(TREE).id]
    place_robot(m, d, tc[0], tc[1] - 1.2)
    saf = np.concatenate([s.hit_geoms() for s in L.sweep(d, lm.SAFETY_GROUPS, noise=False).values()])
    cov = np.concatenate([s.hit_geoms() for s in L.sweep(d, lm.COVERAGE_GROUPS, noise=False).values()])
    assert not np.any(m.geom_group[saf] == 3) and np.any(m.geom_group[cov] == 3)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


# ------------------------------------------------------------------------------------------- trunk fit
@pytest.mark.parametrize("seed", [7, 11, 23])
def test_trunk_fit_within_2cm_at_1_to_3m_with_scaffolds(lite, seed):
    """OS 2.5 / WP5: circle fit on the 0.18 m low planes, mango with scaffolds from ~0.6 m, noise sigma 2 cm,
    10 scans; the robot in the alley on either side of the tree."""
    m, d = lite
    tb = m.body(TREE).id
    tc = d.xpos[tb][:2].copy()
    L = lm.LidarModel(m, seed=seed)
    errs = []
    for dist in (1.0, 1.5, 2.0, 2.5, 3.0):
        for side in (-1.0, 1.0):
            place_robot(m, d, tc[0], tc[1] + side * dist)
            f = lm.TrunkFitter(window=10, seed=seed)
            for _ in range(10):
                f.add_scans(L.sweep(d, lm.SAFETY_GROUPS, noise=True))
            prior = tc + np.random.default_rng(seed).uniform(-0.15, 0.15, 2)      # registry / odometry offset
            r = f.fit(prior, prior_r=0.055)
            assert r.ok, (dist, side, r.reason)
            e = float(np.linalg.norm(r.centre - tc))
            errs.append(e)
            assert e <= 0.02, (dist, side, e, r.method, r.n)
            R_meas = float(np.linalg.norm(r.centre - d.xpos[m.body("trunk").id][:2]))
            assert abs(R_meas - dist) <= 0.02
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"trunk fit seed {seed}: max err {max(errs) * 100:.2f} cm, mean {np.mean(errs) * 100:.2f} cm")


def test_trunk_fit_exact_without_noise_and_free_fit_without_prior(lite):
    m, d = lite
    tc = d.xpos[m.body(TREE).id][:2].copy()
    L = lm.LidarModel(m)
    place_robot(m, d, tc[0], tc[1] - 1.0)
    f = lm.TrunkFitter()
    f.add_scans(L.sweep(d, noise=False))
    fixed = f.fit(tc, prior_r=0.055)
    assert fixed.method == "low_plane_circle_fixed_r" and np.linalg.norm(fixed.centre - tc) < 1e-3
    free = f.fit(tc)                                   # free radius, noise-free, 1 m: exact
    assert free.ok and free.method == "low_plane_circle" and abs(free.radius - 0.055) < 2e-3
    assert np.linalg.norm(free.centre - tc) < 2e-3
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def test_trunk_fit_failure_and_canopy_fallback(lite):
    m, d = lite
    tc = d.xpos[m.body(TREE).id][:2].copy()
    L = lm.LidarModel(m)
    f = lm.TrunkFitter()
    assert f.fit(tc, 0.055).reason == "too_few_points"            # no scans yet
    place_robot(m, d, tc[0], tc[1] - 1.2)
    f.add_scans(L.sweep(d, noise=False))
    far = f.fit(tc + np.array([3.0, 0.0]), 0.055)                 # wrong prior: nothing within the gate
    assert not far.ok and far.reason == "too_few_points"
    cov = L.sweep(d, lm.COVERAGE_GROUPS, noise=False, low=False)["mast"].points()
    cf = lm.fit_canopy_edge(cov, tc, canopy_r=0.85)
    assert cf.ok and cf.method in ("canopy_edge_ellipse", "canopy_edge_circle")
    assert np.linalg.norm(cf.centre - tc) < 0.35                    # coarse: a fallback, labelled as such
    assert not lm.fit_canopy_edge(np.zeros((0, 3)), tc, 0.85).ok
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def test_circle_helpers():
    a = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    pts = np.stack([1.0 + 0.3 * np.cos(a), -2.0 + 0.3 * np.sin(a)], axis=1)
    c, r = lm.fit_circle_algebraic(pts)
    assert np.allclose(c, [1.0, -2.0]) and abs(r - 0.3) < 1e-9
    c2, r2, rms = lm.fit_circle_geometric(pts[:10], [1.1, -1.9], 0.25)
    assert np.allclose(c2, [1.0, -2.0], atol=1e-6) and abs(r2 - 0.3) < 1e-6 and rms < 1e-9
    assert lm.arc_span_deg(pts[:10], [1.0, -2.0]) == pytest.approx(81.0, abs=1.0)


# ------------------------------------------------------------------------------------------- clusters
def test_nonmap_clusters_find_the_person_not_trees(lite):
    m, d = lite
    L = lm.LidarModel(m)
    place_robot(m, d, 0.0, 0.0)
    place_person(m, d, (2.5, 0.3))
    cl = lm.nonmap_clusters(L.sweep(d, noise=False).values(), L.map_geom, d.xpos[L.trunk][:2])
    near = [c for c in cl if np.hypot(c.centroid[0] - 2.5, c.centroid[1] - 0.3) < 0.6]
    assert len(near) == 1
    c = near[0]
    assert c.lo[2] <= 0.0 and c.hi[2] >= lm.PERSON_TOP_M                 # widened to a person height
    js = c.as_dict()
    back = lm.Cluster.from_dict(json.loads(json.dumps(js)))
    assert np.allclose(back.lo, c.lo, atol=1e-3) and back.corners().shape == (8, 3)
    # map geoms: trees, ground; actors are not map
    assert L.map_geom[m.geom("floor").id] if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0 else True
    tree_geoms = np.nonzero(np.asarray(m.geom_bodyid) == m.body(TREE).id)[0]
    assert L.map_geom[tree_geoms].all()
    place_person(m, d, (-30.0, -30.0))
    far = lm.nonmap_clusters(L.sweep(d, noise=False).values(), L.map_geom, d.xpos[L.trunk][:2])
    assert not [c for c in far if np.hypot(c.centroid[0] - 2.5, c.centroid[1] - 0.3) < 0.6]
    assert lm.cluster_xy(np.zeros((0, 3))) == []
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def test_laser_scan_json():
    m, d = _box_world()
    s = lm.LidarModel(m).sweep(d, noise=False)["low_fl"]
    js = lm.laser_scan(s, "low_fl")
    text = json.dumps(js, allow_nan=False)
    back = json.loads(text)
    assert len(back["ranges"]) == 270 and back["range_max"] == lm.RANGE_MAX_M
    assert abs(back["angle_max"] - back["angle_min"] - math.radians(270)) < 1e-5
    assert None in back["ranges"] or all(isinstance(v, float) for v in back["ranges"])


# ------------------------------------------------------------------------------------------- hook
def test_lidar_hook_publishes_scans_clusters_and_fit(lite):
    from atharv_nandi.sim.hooks import Bus, HookRegistry
    from atharv_nandi.sim.hooks import lidar as lidar_hook
    m, d = lite
    reg = HookRegistry()
    lidar_hook.register(reg)
    h = reg.hooks()[0]
    assert h.name == "lidar" and h.rate_hz == 10.0 and not h.mutates_model
    tc = d.xpos[m.body(TREE).id][:2]
    place_robot(m, d, tc[0], tc[1] - 1.5)
    bus = Bus()
    bus.deliver(lidar_hook.FIT_REQUEST_TOPIC, {"tree_id": "L1-1", "prior_xy": [float(tc[0]), float(tc[1])],
                                               "trunk_r_m": 0.055, "canopy_r_m": 0.85})
    for k in range(10):
        d.time = 0.1 * k
        reg.tick(m, d, bus)
    assert not reg.faulted()
    out = {}
    for topic, payload in bus.drain():
        out.setdefault(topic, []).append(json.loads(payload))
    assert len(out["/nandi/scan"]) == 10 and len(out["/nandi/scan"][0]["beams"]) == 3
    assert len(out["/nandi/scan_low"][0]["planes"]) == 2
    assert "clusters" in out["/nandi/lidar/clusters"][0]
    fit = out["/nandi/lidar/trunk_fit"][-1]
    assert fit["ok"] and fit["tree_id"] == "L1-1" and abs(fit["R_meas"] - 1.5) < 0.02
    assert bus.values["lidar"]["n_rays"] == 1620
    reg.reset(m, d)
    assert lidar_hook._HOOK.request is None
    d.time = 0.0
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def test_lidar_sweep_cost_recorded(lite):
    import time
    m, d = lite
    L = lm.LidarModel(m)
    t0 = time.perf_counter()
    for _ in range(20):
        L.sweep(d)
    ms = 1e3 * (time.perf_counter() - t0) / 20
    print(f"lidar sweep 1620 rays: {ms:.2f} ms ({ms * 10:.1f} ms/s at 10 Hz)")
    assert ms < 20.0                                  # MVP-SPEC 3.8 estimate 5-20 ms/s at 10 Hz -> < 2 ms/sweep typ.


# ------------------------------------------------------------------------------------------- edge cases
def test_lidar_edge_cases():
    with pytest.raises(RuntimeError):
        lm.LidarModel(mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1 1 .1' type='plane'/>"
                                                    "</worldbody></mujoco>"))
    assert lm.map_geoms(mujoco.MjModel.from_xml_string("<mujoco/>")).size == 0
    assert lm.nonmap_clusters([], np.zeros(0, bool)) == []
    assert lm.arc_span_deg(np.array([[1.0, 0.0]]), [0.0, 0.0]) == 0.0
    f = lm.TrunkFitter(window=2)
    f.add(np.zeros((0, 3)))
    f.add([[0.0, 0.1, 0.18], [0.03, 0.08, 0.18]])
    f.add([[0.06, 0.1, 0.18]])                                 # window 2: the oldest is dropped
    assert len(f.points_near([0.0, 0.0])) == 3
    r = f.fit([0.03, 0.15])                                    # no radius prior, too few for a free fit
    assert not r.ok and r.reason == "no_free_fit_no_radius_prior"
    # scattered points: the fixed-radius residual is too high
    g = lm.TrunkFitter(cluster_gap_m=1.0, max_rms_m=0.01)
    g.add(np.random.default_rng(1).uniform(-0.2, 0.2, (40, 2)))
    bad = g.fit([0.0, 0.0], prior_r=0.05)
    assert not bad.ok and bad.reason == "residual_too_high"
    # only 2 points in the nearest cluster
    h = lm.TrunkFitter(min_points=3)
    h.add([[0.0, 0.0], [0.01, 0.0], [0.4, 0.0], [0.41, 0.0], [0.42, 0.0]])
    assert h.fit([-0.05, 0.0], 0.05).reason == "too_few_points"
    # free fit through RANSAC on a clean circle with an outlier, no prior
    a = np.linspace(0, math.pi * 1.2, 30)
    pts = np.stack([0.1 * np.cos(a), 0.1 * np.sin(a)], axis=1)
    k = lm.TrunkFitter(cluster_gap_m=0.2)
    k.add(np.vstack([pts, [[0.0, 0.0]]]))
    fr = k.fit([0.0, 0.0])
    assert fr.ok and fr.method == "low_plane_circle" and abs(fr.radius - 0.1) < 0.01
    # ellipse fit: a clean ellipse, a degenerate set
    t = np.linspace(0, 2 * math.pi, 50, endpoint=False)
    e = lm.fit_ellipse(np.stack([2 + 0.8 * np.cos(t), 1 + 0.5 * np.sin(t)], axis=1))
    assert e is not None and np.allclose(e[0], [2, 1], atol=1e-6) and np.allclose(e[1], [0.8, 0.5], atol=1e-6)
    assert lm.fit_ellipse(np.stack([t, 2 * t], axis=1)) is None
    # canopy fallback: a narrow arc uses the fixed registry radius (circle)
    arc = np.stack([3 + 0.85 * np.cos(t[:6] * 0.2), 0.85 * np.sin(t[:6] * 0.2), np.full(6, 1.0)], axis=1)
    cc = lm.fit_canopy_edge(arc, [3.0, 0.0], 0.85)
    assert cc.ok and cc.method == "canopy_edge_circle" and np.linalg.norm(cc.centre - [3.0, 0.0]) < 0.05


def test_lidar_hook_config_and_canopy_fallback(lite, monkeypatch):
    from atharv_nandi.sim.hooks import Bus
    from atharv_nandi.sim.hooks import lidar as lidar_hook
    import atharv_nandi.core.detector_model as dm
    monkeypatch.setattr(dm, "load_config", lambda *a: (_ for _ in ()).throw(OSError("no config")))
    assert lidar_hook.load_lidar_config() == {}
    monkeypatch.undo()
    assert lidar_hook.load_lidar_config()["rate_hz"] == 10.0
    m, d = lite
    h = lidar_hook.LidarHook({"publish_scans": False})
    tc = d.xpos[m.body(TREE).id][:2]
    place_robot(m, d, tc[0], tc[1] - 1.2)
    bus = Bus()
    # a prior far from any trunk with a canopy radius: the low-plane fit fails, the canopy fallback reports
    bus.deliver(lidar_hook.FIT_REQUEST_TOPIC, {"tree_id": "x", "prior_xy": [float(tc[0]), float(tc[1]) + 0.6],
                                               "trunk_r_m": 0.055, "canopy_r_m": 0.85})
    h.tick(m, d, 0.0, bus)
    out = [json.loads(p) for t, p in bus.drain() if t == lidar_hook.FIT_TOPIC]
    assert out and out[0]["method"].startswith("canopy_edge") and "low_plane:" in out[0]["reason"]
    assert not any(t in (lidar_hook.SCAN_TOPIC, lidar_hook.SCAN_LOW_TOPIC) for t, _ in bus.drain())
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


# ------------------------------------------------------------------------------------------- review round 1
def test_pool_fruit_on_the_tree_is_not_a_person_candidate(lite):
    """WP5 review: WP23 pool fruit (free bodies, group 1) hanging on the tree during HARVEST / PICK formed 0-2 m
    'person' clusters (detector mask over the pick target, wrist frames dropped). They are registry fruit now."""
    from atharv_nandi.core import fruit_pool
    m, d = lite
    pool = fruit_pool.FruitPool(m)
    pool.assign_tree(d, "L1-1")
    pool.assign_tree(d, "L1-2")
    try:
        L = lm.LidarModel(m, seed=7)
        pg = np.array([g for g in range(m.ngeom)
                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "").startswith("pool_")])
        assert len(pg) and L.map_geom[pg].all()
        fruit_xy = [d.xpos[pool.slots[k].body][:2].copy() for k in pool.assignments]
        assert fruit_xy
        poses_with_hits = 0
        for x in np.linspace(-1.0, 6.0, 12):
            for y in np.linspace(-1.0, 5.0, 12):
                place_robot(m, d, x, y)
                sc = L.sweep(d, lm.SAFETY_GROUPS, noise=False)
                if not any(np.isin(s.geom[s.valid], pg).any() for s in sc.values()):
                    continue
                poses_with_hits += 1
                for c in lm.nonmap_clusters(sc.values(), L.map_geom, d.xpos[L.trunk][:2]):
                    for fx, fy in fruit_xy:
                        assert not (c.lo[0] - 0.05 <= fx <= c.hi[0] + 0.05 and c.lo[1] - 0.05 <= fy <= c.hi[1] + 0.05), \
                            (x, y, c.as_dict())
        assert poses_with_hits > 10                            # the rays did hit the pooled fruit
        print(f"pool fruit hit from {poses_with_hits} of 144 poses, no cluster")
    finally:
        pool.reset(d)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)


@pytest.mark.parametrize("req, reason", [
    ({"prior_xy": [2.0]}, "prior_xy_not_2_numbers"),
    ({"prior_xy": "ab"}, "prior_xy_not_2_numbers"),
    ({"prior_xy": [2.0, float("nan")]}, "prior_xy_not_2_numbers"),
    ({"prior_xy": [2, 2.5], "trunk_r_m": "x"}, "trunk_r_m_not_positive_number"),
    ({"prior_xy": [2, 2.5], "trunk_r_m": -0.05}, "trunk_r_m_not_positive_number"),
    ({"prior_xy": [2, 2.5], "canopy_r_m": "big"}, "canopy_r_m_not_positive_number"),
    (["not", "a", "dict"], "not_an_object"),
])
def test_malformed_trunk_fit_request_is_refused_not_raised(lite, req, reason):
    """WP5 review: a malformed request raised in the hook, which disables the LiDAR and safe-holds the runner."""
    from atharv_nandi.sim.hooks import Bus, HookRegistry
    from atharv_nandi.sim.hooks import lidar as lidar_hook
    m, d = lite
    reg = HookRegistry()
    lidar_hook.register(reg)
    bus = Bus()
    bus.deliver(lidar_hook.FIT_REQUEST_TOPIC, req)
    for k in range(3):
        d.time = 0.1 * k
        reg.tick(m, d, bus)
    assert not reg.faulted()
    out = {}
    for topic, payload in bus.drain():
        out.setdefault(topic, []).append(json.loads(payload))
    assert len(out["/nandi/lidar/clusters"]) == 3              # the LiDAR keeps running
    fits = out["/nandi/lidar/trunk_fit"]
    assert len(fits) == 1 and fits[0]["ok"] is False and fits[0]["reason"] == f"bad_request:{reason}"
    assert lidar_hook._HOOK.request is None
    with pytest.raises(ValueError):
        lidar_hook._HOOK.fit(d, req if isinstance(req, dict) else {})
    # a request without prior_xy is a cancel (no reply), as before
    bus.deliver(lidar_hook.FIT_REQUEST_TOPIC, {"tree_id": "L1-1"})
    reg.tick(m, d, bus)
    assert not any(t == lidar_hook.FIT_TOPIC for t, _ in bus.drain())
    d.time = 0.0


# ------------------------------------------------------------------------------------- terrain (hfield) rays
def test_hfield_rays_match_mj_multiray_and_stay_cheap_on_orchard_demo():
    """Integrate fix: MuJoCo's ray / hfield test walks every cell of the ray's xy box, so on the orchard_demo
    terrain (501 x 481 cells) one sweep took about 35-67 ms. ``_HfieldCaster`` marches the terrain itself; it must
    give the same returns as plain ``mj_multiRay`` and must not write the live model."""
    import time
    with world_env("orchard_demo"):
        m = farm_scene.load_farm_model()
    d = mujoco.MjData(m)
    fast = lm.LidarModel(m)
    assert len(fast._hf) == 1
    ref = lm.LidarModel(m)
    ref._hf = []                                        # plain mj_multiRay on the live model
    groups0 = m.geom_group.copy()
    rng = np.random.default_rng(3)
    qa = m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")]]
    n_rays = n_bad = 0
    worst, t_fast = 0.0, 0.0
    for _ in range(12):
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[qa:qa + 2] = rng.uniform(-3.0, 13.0, 2)
        yaw = rng.uniform(-math.pi, math.pi)
        d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        for groups in (lm.SAFETY_GROUPS, lm.COVERAGE_GROUPS):
            t = time.perf_counter()
            a = fast.sweep(d, groups=groups, noise=False)
            t_fast += time.perf_counter() - t
            b = ref.sweep(d, groups=groups, noise=False)
            for k in a:
                fa, fb = np.isfinite(a[k].ranges), np.isfinite(b[k].ranges)
                both = fa & fb
                n_rays += len(fa)
                n_bad += int((fa != fb).sum() + (a[k].geom[both] != b[k].geom[both]).sum())
                if both.any():
                    worst = max(worst, float(np.abs(a[k].ranges[both] - b[k].ranges[both]).max()))
    print(f"hfield rays: {n_rays} rays, {n_bad} hit/geom mismatches, worst {worst * 1000:.3f} mm, "
          f"{t_fast / 24 * 1000:.2f} ms/sweep")
    assert n_bad <= n_rays * 1e-3 and worst < 0.005
    assert np.array_equal(m.geom_group, groups0)       # the live model is never written
    assert t_fast / 24 < 0.025                         # was 35-67 ms per sweep with the hfield in mj_multiRay


# ------------------------------------------------------------------------------------- ring-depth band cross-check
@pytest.mark.parametrize("seed", [7, 11, 23])
def test_ring_depth_band_cross_check_agrees_with_low_plane_fit(lite, seed):
    """OS 2.5: the low-plane fit is cross-checked on a ring-depth band at 0.2-0.5 m. Mango with scaffolds from
    ~0.6 m and pruning sprouts on the trunk base (``tree_L1-1_prune_*``, which touch the trunk in the band), depth
    noise sigma 1 cm, both alley sides, 1.0-2.5 m: band centre <= 2 cm from the truth and ``agree`` with the
    low-plane fit."""
    m, d = lite
    tc = d.xpos[m.body(TREE).id][:2].copy()
    L = lm.LidarModel(m, seed=seed)
    B = lm.RingDepthBand(L, seed=seed)
    assert [c[0] for c in B.cams] == list(lm.RING_CAMS)
    errs, deltas = [], []
    for dist in (1.0, 1.5, 2.0, 2.5):
        for side in (-1.0, 1.0):
            place_robot(m, d, tc[0], tc[1] + side * dist)
            gz = float(d.xpos[L.trunk][2]) - lm.NOMINAL_TRUNK_Z_M
            prior = tc + np.random.default_rng(seed).uniform(-0.15, 0.15, 2)
            f = lm.TrunkFitter(window=10, seed=seed)
            for _ in range(10):
                f.add_scans(L.sweep(d, lm.SAFETY_GROUPS, noise=True))
            prim = f.fit(prior, prior_r=0.055)
            P, O, cams = B.points(d, prior, gz)
            assert len(P) and np.all((P[:, 2] - gz >= 0.2) & (P[:, 2] - gz <= 0.5))
            cc = lm.band_cross_check(P, O, prior, 0.055, prim)
            assert cc["ok"], (dist, side, cc)
            e = float(np.linalg.norm(np.asarray(cc["centre"]) - tc))
            errs.append(e)
            deltas.append(cc["delta_m"])
            assert e <= 0.02 and cc["agree"] is True, (dist, side, e, cc)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"ring band seed {seed}: max err {max(errs) * 100:.2f} cm, max delta {max(deltas) * 100:.2f} cm")


def test_band_cross_check_flags_disagreement_and_edge_cases():
    rng = np.random.default_rng(1)
    c = np.array([2.0, 1.0])
    a = rng.uniform(-1.2, 1.2, 60) - math.pi / 2               # the arc facing a camera at (2, -0.5)
    z = rng.uniform(0.2, 0.5, 60)
    P = np.column_stack([c[0] + 0.055 * np.cos(a), c[1] + 0.055 * np.sin(a), z])
    O = np.array([2.0, -0.5, 0.93])
    # a sprout touching the trunk (one cluster): trimmed out
    s = np.column_stack([np.full(15, c[0] + 0.07), np.linspace(c[1] - 0.06, c[1] - 0.16, 15), np.full(15, 0.3)])
    good = lm.TrunkFit(True, "low_plane_circle_fixed_r", c + [0.01, 0.0], 0.055, 0.01, 20, 90.0)
    out = lm.band_cross_check(np.vstack([P, s]), O, c + [0.1, 0.05], 0.055, good)
    assert out["ok"] and out["agree"] is True and np.linalg.norm(np.asarray(out["centre"]) - c) < 0.005
    assert out["inliers"] <= 60 + 2
    bad = lm.TrunkFit(True, "low_plane_circle_fixed_r", c + [0.06, 0.0], 0.055, 0.01, 20, 90.0)
    assert lm.band_cross_check(P, O, c, 0.055, bad)["agree"] is False
    # a canopy fallback is not compared; nothing near the prior; free fit without a radius
    canopy = lm.TrunkFit(True, "canopy_edge_circle", c, 0.85, 0.1, 20, 90.0)
    assert lm.band_cross_check(P, O, c, 0.055, canopy)["agree"] is None
    far = lm.band_cross_check(P, O, c + [3.0, 0.0], 0.055, good)
    assert not far["ok"] and far["reason"] == "too_few_points" and far["agree"] is None
    assert lm.band_cross_check(np.zeros((0, 3)), None, c, None)["reason"] == "too_few_points"
    free = lm.band_cross_check(P, None, c, None, good)
    assert free["fit"] == "free" and free["ok"] and abs(free["radius"] - 0.055) < 0.005
    line = np.column_stack([np.linspace(1.9, 2.1, 30), np.full(30, 1.0), np.full(30, 0.3)])
    assert not lm.band_cross_check(line, None, c, None)["ok"]   # a straight edge (basin rim) is no trunk
    # back-projection: the optical axis of a camera at the origin looking along +x
    R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])     # optical axes (x right, y down, z fwd)
    q = lm.backproject(np.array([79.5]), np.array([59.5]), np.array([2.0]), np.zeros(3), R, 100, 100, 80, 60)
    assert np.allclose(q, [[2.0, 0.0, 0.0]])


def test_lidar_hook_reports_the_ring_band_cross_check(lite):
    from atharv_nandi.sim.hooks import Bus
    from atharv_nandi.sim.hooks import lidar as hl

    def run(h, n, t0=0.0, req=None, bus=None):
        bus = bus or Bus()
        if req is not None:
            bus.deliver(hl.FIT_REQUEST_TOPIC, req)
        for k in range(n):
            h.tick(m, d, t0 + 0.1 * k, bus)
        return [json.loads(p) for t, p in bus.drain() if t == hl.FIT_TOPIC]

    m, d = lite
    tc = d.xpos[m.body(TREE).id][:2].copy()
    req = {"tree_id": TREE, "prior_xy": [float(tc[0]), float(tc[1])], "trunk_r_m": 0.055}
    place_robot(m, d, tc[0], tc[1] - 1.2)
    h = hl.LidarHook({"ring_band": {"max_range_m": 2.5}})
    hb = Bus()
    fit = run(h, 10, req=req, bus=hb)[-1]
    cc = fit["cross_check"]
    assert fit["ok"] and cc["ok"] and cc["agree"] is True and cc["method"] == "ring_depth_band", fit
    assert cc["cams"] and cc["rays"] > 0 and cc["delta_m"] <= 0.03 and abs(cc["range_m"] - 1.2) < 0.05
    place_robot(m, d, tc[0], tc[1] - 3.0)
    assert run(h, 1, 1.0, bus=hb)[-1]["cross_check"]["reason"] == "out_of_range"   # the request is still active
    off = hl.LidarHook({"ring_band": {"enabled": False}})
    assert "cross_check" not in run(off, 1, req=req)[-1]
    # a failing cross-check never costs the primary fit
    place_robot(m, d, tc[0], tc[1] - 1.2)
    boom = hl.LidarHook()
    boom._bind(m)
    boom.band.points = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    out = run(boom, 10, req=req)[-1]
    assert out["ok"] and out["cross_check"]["reason"] == "error:RuntimeError"
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def test_ring_band_points_are_cached_at_the_band_rate(lite):
    from atharv_nandi.sim.hooks import lidar as hl
    m, d = lite
    tc = d.xpos[m.body(TREE).id][:2].copy()
    place_robot(m, d, tc[0], tc[1] - 1.2)
    h = hl.LidarHook()
    h._bind(m)
    for _ in range(10):
        h.fitter.add_scans(h.lidar.sweep(d, lm.SAFETY_GROUPS, noise=True))
    calls = []
    real = h.band.points
    h.band.points = lambda *a, **k: calls.append(1) or real(*a, **k)
    req = {"tree_id": TREE, "prior_xy": [float(tc[0]), float(tc[1])], "trunk_r_m": 0.055}
    for k in range(10):                                     # 1 s at 10 Hz, band at 2 Hz
        d.time = 0.1 * k
        assert h.fit(d, req)["cross_check"]["agree"] is True
    assert len(calls) == 2
    h.fit(d, dict(req, prior_xy=[float(tc[0]) + 0.01, float(tc[1])]))   # a new prior re-casts at once
    assert len(calls) == 3
    h.reset()
    assert h._band_cache is None
    d.time = 0.0
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------- WP5 review fixes: close range, basin rim, radius
@pytest.mark.parametrize("seed", [7, 11])
def test_band_cross_check_close_beside_the_tree_never_locks_onto_a_sprout(lite, seed):
    """TRUNK/BASE SCAN, PRUNE and PICK stand 0.45-0.65 m beside the tree. At yaw 0 (tree on the left) the tree-side
    ring camera cannot see 0.2-0.5 m that close and the front camera sees only ``tree_L1-1_prune_1``: the band
    fit there used to be ``ok`` 28-38 cm off the trunk and report ``agree`` False against a good low-plane fit. Now
    such a fit is ``off_trunk`` (or ``too_few_inliers``) with ``agree`` None; every ``ok`` band fit is on the
    trunk."""
    m, d = lite
    tc = d.xpos[m.body(TREE).id][:2].copy()
    L = lm.LidarModel(m, seed=seed)
    B = lm.RingDepthBand(L, seed=seed)
    rejected = []
    for dist in (0.45, 0.55, 0.65):
        for yaw in (0.0, math.pi / 4, math.pi / 2, math.pi):
            place_robot(m, d, tc[0], tc[1] - dist, yaw)
            gz = float(d.xpos[L.trunk][2]) - lm.NOMINAL_TRUNK_Z_M
            f = lm.TrunkFitter(window=10, seed=seed)
            for _ in range(10):
                f.add_scans(L.sweep(d, lm.SAFETY_GROUPS, noise=True))
            prim = f.fit(list(tc), prior_r=0.055)
            assert prim.ok and np.linalg.norm(prim.centre - tc) <= 0.02, (dist, yaw, prim)
            P, O, _ = B.points(d, list(tc), gz)
            cc = lm.band_cross_check(P, O, list(tc), 0.055, prim)
            if cc["ok"]:
                assert np.linalg.norm(np.asarray(cc["centre"]) - tc) <= 0.02 and cc["agree"] is True, (dist, yaw, cc)
            else:
                assert cc["reason"] in ("off_trunk", "too_few_inliers") and cc["agree"] is None, (dist, yaw, cc)
                rejected.append((dist, yaw, cc["reason"]))
    assert (0.45, 0.0, "off_trunk") in rejected and (0.55, 0.0, "off_trunk") in rejected
    assert all(y == 0.0 for _, y, _ in rejected)            # yaw 45-180 see the trunk in the band
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def test_band_compare_off_trunk_gate_keeps_a_band_that_catches_a_wrong_primary():
    c = np.array([2.0, 1.0])
    fit = lm.band_result(fit="fixed_r", ok=True, n=40, inliers=30, centre=[2.0, 1.0], radius=0.055)
    bad_prim = lm.TrunkFit(True, "low_plane_circle_fixed_r", c + [0.2, 0.0], 0.055, 0.01, 20, 90.0)
    out = lm.band_compare(fit, c, bad_prim)                 # band on the prior, primary 20 cm off: disagreement
    assert out["ok"] and out["agree"] is False and abs(out["delta_m"] - 0.2) < 1e-6
    sprout = dict(fit, centre=[2.25, 0.8])                  # 32 cm from the prior and from a good primary
    good = lm.TrunkFit(True, "low_plane_circle_fixed_r", c, 0.055, 0.01, 20, 90.0)
    out = lm.band_compare(sprout, c, good)
    assert not out["ok"] and out["reason"] == "off_trunk" and out["agree"] is None and out["delta_m"] is None
    assert lm.band_compare(sprout, c, None)["reason"] == "off_trunk"
    # free radius: judged only on a wide free-fit arc
    wide = dict(fit, free_radius=0.09, radius_delta_m=0.035, free_arc_deg=150.0)
    out = lm.band_compare(wide, c, good)
    assert out["radius_mismatch"] is True and out["agree"] is False and out["delta_m"] == 0.0
    narrow = dict(wide, free_arc_deg=60.0)
    out = lm.band_compare(narrow, c, good)
    assert out["radius_mismatch"] is None and out["agree"] is True


@pytest.fixture(scope="module")
def demo():
    with world_env("orchard_demo"):
        m = farm_scene.load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    yield m, d
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def _place_on_terrain(m, d, x, y, yaw=0.0):
    """The robot level at the terrain height under it (ground + nominal trunk height), not the keyframe height:
    at 1.0 m from an orchard_demo tree it stands on the ~9 cm basin rim."""
    g = np.zeros(1, np.int32)
    r = mujoco.mj_ray(m, d, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]),
                      np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, g)
    qa = m.jnt_qposadr[m.body_jntadr[m.body("trunk").id]]
    d.qpos[qa:qa + 3] = [x, y, 5.0 - r + lm.NOMINAL_TRUNK_Z_M]
    d.qpos[qa + 3:qa + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    return 5.0 - r


def _demo_fit(m, d, tree, dist, side, seed, r_req=None):
    g = m.geom(tree + "_wood_0").id
    tc, tr = d.geom_xpos[g][:2].copy(), float(m.geom_size[g][0])
    _place_on_terrain(m, d, tc[0], tc[1] + side * dist)
    L = lm.LidarModel(m, seed=seed)
    f = lm.TrunkFitter(window=10, seed=seed)
    for _ in range(10):
        f.add_scans(L.sweep(d, lm.SAFETY_GROUPS, noise=True))
    r = tr if r_req is None else r_req
    prim = f.fit(list(tc), prior_r=r)
    P, O, _ = lm.RingDepthBand(L, seed=seed).points(d, list(tc), float(d.xpos[L.trunk][2]) - lm.NOMINAL_TRUNK_Z_M)
    return tc, tr, prim, lm.band_cross_check(P, O, list(tc), r, prim)


@pytest.mark.parametrize("tree", ["tree_R1-3", "tree_OT-1"])
def test_trunk_fit_within_2cm_on_the_orchard_demo_basin_rim(demo, tree):
    """WP5 'trunk fit <= 2 cm at 1-3 m' on orchard_demo terrain: at 1.0 m the robot stands on the basin rim and the
    0.18 m planes cut the pruning sprout roots that touch the trunk; the plain fixed-radius fit was 3.0-3.9 cm off
    (or residual_too_high). Seeded from a sensor-facing fixed-radius RANSAC and trimmed (4 cm), it is <= 2 cm."""
    m, d = demo
    errs = []
    for dist in (0.9, 1.0, 1.2):
        for side in (-1.0, 1.0):
            for seed in (7, 31):
                tc, _, prim, cc = _demo_fit(m, d, tree, dist, side, seed)
                assert prim.ok, (dist, side, seed, prim)
                e = float(np.linalg.norm(prim.centre - tc))
                errs.append(e)
                assert e <= 0.02, (dist, side, seed, e)
                assert cc["ok"] and cc["agree"] is True, (dist, side, seed, cc)
    print(f"{tree} basin rim: max low-plane error {max(errs) * 100:.2f} cm")


def test_band_cross_check_flags_a_wrong_registry_radius(demo):
    """A wrong registry ``trunk_r_m`` shifts the fixed-radius band centre the same way as the fixed-radius
    low-plane centre (R1-2, true r 0.055, requested 0.09: primary 4.7-5.5 cm off, centres 1.2-2.9 cm apart). The
    free-radius band fit catches it: ``radius_mismatch`` True, ``agree`` False."""
    m, d = demo
    for side in (-1.0, 1.0):
        tc, tr, prim, cc = _demo_fit(m, d, "tree_R1-2", 1.2, side, 7, r_req=0.09)
        assert cc["ok"] and cc["radius_mismatch"] is True and cc["agree"] is False, cc
        assert abs(cc["free_radius"] - tr) <= 0.01
        _, _, prim, cc = _demo_fit(m, d, "tree_R1-2", 1.2, side, 7)
        assert cc["radius_mismatch"] is False and cc["agree"] is True, cc


def test_cross_check_payload_always_carries_every_contract_field(lite):
    from atharv_nandi.sim.hooks import Bus
    from atharv_nandi.sim.hooks import lidar as hl
    m, d = lite
    tc = d.xpos[m.body(TREE).id][:2].copy()
    req = {"tree_id": TREE, "prior_xy": [float(tc[0]), float(tc[1])], "trunk_r_m": 0.055}
    keys = set(lm.BAND_FIELDS) | {"cams", "rays", "range_m", "band_t"}
    h = hl.LidarHook()
    h._bind(m)
    place_robot(m, d, tc[0], tc[1] - 1.2)
    for _ in range(10):
        h.fitter.add_scans(h.lidar.sweep(d, lm.SAFETY_GROUPS, noise=True))
    ok = h.fit(d, req)["cross_check"]
    assert keys <= set(ok) and ok["agree"] is True and ok["radius_mismatch"] is False
    place_robot(m, d, tc[0], tc[1] - 3.0)
    far = h.fit(d, req)["cross_check"]
    assert keys <= set(far) and far["reason"] == "out_of_range" and far["cams"] == [] and far["n"] == 0
    place_robot(m, d, tc[0], tc[1] - 1.2)
    h._band_cache = None
    h.band.points = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    err = h.fit(d, req)["cross_check"]
    assert keys <= set(err) and err["reason"] == "error:RuntimeError" and abs(err["range_m"] - 1.2) < 0.05
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def test_band_fit_is_cached_with_the_band_cast(lite, monkeypatch):
    """The band circle fit (fixed-radius RANSAC + trims, ~3 ms) is deterministic for the cached points, so it runs
    at the 2 Hz band rate, not at every 10 Hz tick; only delta / agree are recomputed against the live primary."""
    from atharv_nandi.sim.hooks import lidar as hl
    m, d = lite
    tc = d.xpos[m.body(TREE).id][:2].copy()
    place_robot(m, d, tc[0], tc[1] - 1.2)
    h = hl.LidarHook()
    h._bind(m)
    for _ in range(10):
        h.fitter.add_scans(h.lidar.sweep(d, lm.SAFETY_GROUPS, noise=True))
    calls = []
    real = lm.band_fit
    monkeypatch.setattr(lm, "band_fit", lambda *a, **k: calls.append(1) or real(*a, **k))
    req = {"tree_id": TREE, "prior_xy": [float(tc[0]), float(tc[1])], "trunk_r_m": 0.055}
    for k in range(10):
        d.time = 0.1 * k
        assert h.fit(d, req)["cross_check"]["agree"] is True
    assert len(calls) == 2
    d.time = 0.0
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
