"""WP25 band planner and TREE 360 coverage (ORCHARD-MVP-SPEC 1.3 O-01, 2.3 "Bands", 2.5 "Ray groups", 6).

Unit tier, ROS-free: the band planner is pure arithmetic; the occlusion tests use a tiny MuJoCo model so the
coverage ray mask ({0, 1, 3}: wood, fruit, foliage; never the robot's group 2 or the query-only group 5) is the real
``mj_ray`` filter.
"""
import math

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import coverage as cv
from atharv_nandi.core.detector_model import Pose, intrinsics_from_fovy


# ------------------------------------------------------------------------------------------- band planner
@pytest.mark.parametrize("d", cv.STANDOFFS_M)
def test_band_coverage_at_standoffs(d):
    """OS 6 "band coverage": 0-2.8 m covered with >= 0.3 m overlap at d in {0.9, 1.2, 1.5}."""
    bp = cv.plan_bands(d)
    b = bp.bands
    assert bp.covered, bp.as_dict()
    assert b["low"].z_lo == 0.0
    assert b["high"].z_hi >= 2.8
    assert bp.overlaps["low_mid"] >= 0.30 - 1e-9
    assert bp.overlaps["mid_high"] >= 0.30 - 1e-9
    assert b["mid"].lens_z == pytest.approx(1.70) and b["high"].lens_z == pytest.approx(2.70)
    assert b["mid"].ext_m == pytest.approx(0.60) and b["high"].ext_m == pytest.approx(1.60)
    for name in ("mid", "high"):
        assert cv.GIMBAL_TILT_MIN_DEG <= b[name].tilt_deg <= cv.GIMBAL_TILT_MAX_DEG


def test_band_table_matches_spec():
    """The OS 2.3 table (computed, planar screen) with the default tilts mid -10, high -20 deg."""
    want = {0.9: ((0, 1.25), (0.95, 2.03), (1.63, 2.86), 0.31, 0.40),
            1.2: ((0, 1.34), (0.69, 2.14), (1.27, 2.91), 0.65, 0.87),
            1.5: ((0, 1.43), (0.44, 2.25), (0.91, 2.96), 0.99, 1.34)}
    for d, (low, mid, high, ov1, ov2) in want.items():
        bp = cv.plan_bands(d)
        b = bp.bands
        assert not b["mid"].solved and not b["high"].solved      # the default tilts already pass at these d
        for band, (lo, hi) in (("low", low), ("mid", mid), ("high", high)):
            assert b[band].z_lo == pytest.approx(lo, abs=0.011), (d, band)
            assert b[band].z_hi == pytest.approx(hi, abs=0.011), (d, band)
        assert bp.overlaps["low_mid"] == pytest.approx(ov1, abs=0.011)
        assert bp.overlaps["mid_high"] == pytest.approx(ov2, abs=0.011)


def test_band_solver_fixes_a_gap():
    """The previous plan (mid at 1.90 m, untilted) left a gap above the ring at d 0.9; the solver re-tilts it."""
    bp_bad = cv.band_interval(1.90, 0.0, cv.CROWN_FOVY_DEG, 0.9)
    ring_top = cv.band_interval(cv.RING_LENS_M, cv.RING_TILT_DEG, cv.RING_FOVY_DEG, 0.9)[1]
    assert bp_bad[0] > ring_top                                   # 1.38 > 1.25: the old gap
    bp = cv.plan_bands(0.9, mid_lens_m=1.90, tilts={"mid": 0.0})
    assert bp.bands["mid"].solved
    assert bp.covered and bp.overlaps["low_mid"] >= 0.30


def test_band_solver_short_standoff_and_bad_input():
    bp = cv.plan_bands(0.6)
    assert bp.overlaps["low_mid"] >= 0.30 - 1e-9 and bp.overlaps["mid_high"] >= 0.30 - 1e-9
    with pytest.raises(ValueError):
        cv.plan_bands(0.0)
    with pytest.raises(ValueError):
        cv.plan_bands(float("nan"))


# ------------------------------------------------------------------------------------------- samples
def _canopy():
    return cv.Canopy("T", (0.0, 0.0), 0.0, 2.2, 0.06, [((0.0, 0.0, 1.39), (1.0, 1.0, 0.81))])


def test_samples_72_per_band_on_the_surface():
    can, bp = _canopy(), cv.plan_bands(0.9)
    smp = cv.surface_samples(can, bp)
    for b in cv.BANDS:
        s = smp[b]
        assert len(s.points) == cv.N_AZ * cv.LEVELS == 72
        r = np.hypot(s.points[:, 0], s.points[:, 1])
        for p, rr, z in zip(s.points, r, s.z):
            want = can.radius(z, math.atan2(p[1], p[0]))
            assert rr == pytest.approx(want, abs=1e-6)
    low = smp["low"]
    assert low.z.min() < 0.58 and np.hypot(*low.points[np.argmin(low.z), :2]) == pytest.approx(0.06)  # trunk


def test_side_of_az():
    assert cv.side_of_az(math.radians(90), ["-y", "+y"]) == "+y"
    assert cv.side_of_az(math.radians(260), ["-y", "+y"]) == "-y"
    assert cv.side_of_az(math.radians(10), ["+x", "-x"]) == "+x"
    assert cv.side_of_az(0.3, ["all"]) in ("-y", "+y")


# ------------------------------------------------------------------------------------------- visibility
def _look_at(eye, target):
    """World -> optical pose of a camera at ``eye`` looking at ``target`` (x right, y down, z forward)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Pose.from_frame_in_world(eye, np.stack([x, y, z], axis=1))


BLOCKER_XML = """
<mujoco>
  <worldbody>
    <geom name="target" type="box" size="0.02 0.3 0.3" pos="0 0 1" group="3"/>
    <geom name="blocker" type="box" size="0.05 0.1 0.1" pos="-0.6 0 1" group="{grp}"/>
  </worldbody>
</mujoco>
"""


def _model(grp):
    m = mujoco.MjModel.from_xml_string(BLOCKER_XML.format(grp=grp))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


@pytest.mark.parametrize("grp,occluded", [(3, True), (0, True), (1, True), (2, False), (5, False)])
def test_point_behind_foliage_not_counted(grp, occluded):
    """OS 6 "coverage occlusion": a sample point behind foliage (group 3; also wood 0 / fruit 1) is not counted;
    robot (2) and query-only (5) geoms are not in the coverage mask."""
    m, d = _model(grp)
    ray = cv.mujoco_ray_fn(m, d)
    pts = np.array([[-0.02, 0.0, 1.0],          # on the target face, behind the blocker
                    [-0.02, 0.25, 1.25]])       # on the target face, clear of the blocker
    pose = _look_at([-1.2, 0.0, 1.0], [0.0, 0.0, 1.0])
    intr = intrinsics_from_fovy(60.0, 320, 240)
    seen, masked, cnt = cv.frame_visibility(pts, pose, intr, 60.0, 50.0, ray)
    assert bool(seen[1])
    assert bool(seen[0]) is (not occluded)
    assert cnt["occluded"] == (1 if occluded else 0)
    assert not masked.any()


def test_masked_and_dropped_points_are_masked_not_seen():
    can, bp = _canopy(), cv.plan_bands(0.9)
    acc = cv.CoverageAccumulator(can, bp)
    pose = _look_at([1.9, 0.0, 0.98], [0.0, 0.0, 0.7])
    intr = intrinsics_from_fovy(83.6, 320, 240)
    full_mask = np.ones((240, 320), bool)
    acc.add_frame("low", pose, intr, 83.6, None, mask=full_mask)
    s = acc.summary()
    assert s["bands"]["low"]["seen"] == 0 and s["bands"]["low"]["masked_pct"] > 0
    acc.add_frame("low", pose, intr, 83.6, None, mask=None, dropped=True)
    assert acc.summary()["bands"]["low"]["dropped"] == 1
    acc.add_frame("low", pose, intr, 83.6, None, mask=np.zeros((240, 320), bool))
    s2 = acc.summary()["bands"]["low"]
    assert s2["seen"] > 0
    assert s2["masked_pct"] < s["bands"]["low"]["masked_pct"]       # a later clean frame clears the mask share


def test_parcel_mask_counts_only_where_it_shows_the_point():
    can, bp = _canopy(), cv.plan_bands(0.9)
    smp = cv.surface_samples(can, bp)["low"].points
    pose = _look_at([1.9, 0.0, 0.98], [0.0, 0.0, 0.7])
    intr = intrinsics_from_fovy(83.6, 320, 240)
    parcel = np.ones((240, 320), bool)
    far = np.full((240, 320), 50.0)              # the pixels show background beyond the parcel (thin-foliage edge)
    seen, masked, _ = cv.frame_visibility(smp, pose, intr, 83.6, 20.0, None, np.zeros((240, 320), bool),
                                          parcel=parcel, depth=far)
    assert seen.sum() > 0 and not masked.any()
    z = pose.apply(smp)[:, 2]
    near = np.full((240, 320), float(np.median(z)))                 # the pixels show the points themselves
    seen2, masked2, _ = cv.frame_visibility(smp, pose, intr, 83.6, 20.0, None, np.zeros((240, 320), bool),
                                            parcel=parcel, depth=near)
    assert masked2.sum() > 0


def test_footprint_limit():
    can, bp = _canopy(), cv.plan_bands(0.9)
    pts = cv.surface_samples(can, bp)["mid"].points
    intr = intrinsics_from_fovy(60.0, 320, 240)
    near = _look_at([1.9, 0.0, 1.7], [0.0, 0.0, 1.5])
    far = _look_at([4.0, 0.0, 1.7], [0.0, 0.0, 1.5])
    s_near, _, _ = cv.frame_visibility(pts, near, intr, 60.0, 12.0, None)
    s_far, _, c_far = cv.frame_visibility(pts, far, intr, 60.0, 12.0, None)
    assert s_near.sum() > 0 and s_far.sum() == 0 and c_far["in_image"] > 0     # 12 mm/px reached at ~2.5 m


def test_full_orbit_of_frames_covers_and_pass_rules():
    can, bp = _canopy(), cv.plan_bands(0.9)
    acc = cv.CoverageAccumulator(can, bp)
    for k in range(8):
        th = 2 * math.pi * k / 8
        eye = np.array([1.9 * math.cos(th), 1.9 * math.sin(th)])
        acc.add_frame("low", _look_at([*eye, 0.98], [0, 0, 0.98 - 1.9 * math.tan(math.radians(25))]),
                      intrinsics_from_fovy(83.6, 320, 240), 83.6, None, side="full")
        for band in ("mid", "high"):
            b = bp.bands[band]
            tz = b.lens_z + 1.9 * math.tan(math.radians(b.tilt_deg))
            acc.add_frame(band, _look_at([*eye, b.lens_z], [0, 0, tz]), intrinsics_from_fovy(60.0, 320, 240),
                          60.0, None, side="full")
    s = acc.summary()
    for band in cv.BANDS:
        assert s["bands"][band]["pct"] >= 90.0, (band, s["bands"][band])
        assert s["bands"][band]["az_deg"] == 360.0
    ok, why = cv.coverage_pass(s, "full")
    assert ok, why
    # two-sided rule: >= 120 deg on each alley side, every band (a full orbit passes it too)
    ok2, _ = cv.coverage_pass(s, "two_sided")
    assert ok2
    bad = {"bands": {"low": {"pct": 50.0, "sides": {"-y": 180.0, "+y": 90.0}}}}
    assert cv.coverage_pass(bad, "two_sided") == (False, "two_sided_below_2x120:low")
    assert cv.coverage_pass(bad, "full") == (False, "full_below_90:low")


SURF_XML = """
<mujoco>
  <worldbody>
    <body name="tree_T" pos="0 0 0">
      <geom name="trunk" type="capsule" fromto="0 0 0 0 0 0.6" size="0.06" group="0"/>
      <geom name="blob" type="ellipsoid" size="0.5 0.5 0.4" pos="0 0 1.2" group="3"/>
    </body>
    <body name="tree_N" pos="1.2 0 0">
      <geom name="nblob" type="sphere" size="0.4" pos="0 0 1.2" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_surface_samples_model_first_hit_on_own_body():
    m = mujoco.MjModel.from_xml_string(SURF_XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    can = cv.Canopy("T", (0.0, 0.0), 0.0, 1.6, 0.06, [((0.0, 0.0, 1.2), (0.7, 0.7, 0.4))])
    bp = cv.plan_bands(0.9)
    bid = m.body("tree_T").id
    smp = cv.surface_samples_model(can, bp, cv.mujoco_surface_ray_fn(m, d), lambda g: int(m.geom_bodyid[g]), bid)
    mid = smp["mid"]
    i = int(np.argmin(np.abs(mid.az - 0.0) + np.abs(mid.z - 1.2)))  # az 0 faces the neighbour at x = 1.2
    r = math.hypot(*mid.points[i, :2])
    k = math.sqrt(max(0.0, 1 - ((mid.z[i] - 1.2) / 0.4) ** 2))
    assert r == pytest.approx(0.5 * k, abs=0.01)          # stepped past the neighbour's blob onto its own
    low = smp["low"]
    j = int(np.argmin(low.z))
    assert math.hypot(*low.points[j, :2]) == pytest.approx(0.06, abs=0.005)   # the trunk below the canopy
