"""WP5 ring pano, self-masks and the crown camera (MVP-SPEC 2.4, 3.7; ORCHARD-MVP-SPEC 2.3, 2.5, 6)."""
import contextlib
import math
import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import camera_rig, pano, self_mask


# ------------------------------------------------------------------------------------------- shared WP5 helpers
# (test_lidar.py, test_privacy.py and test_detectors.py import these from here)
_GL = None


def gl_available() -> bool:
    """Probe in a subprocess: a broken GL stack can abort the interpreter instead of raising."""
    global _GL
    if _GL is None:
        code = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); "
                "r = mujoco.Renderer(m, 8, 8); r.close()")
        try:
            _GL = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode == 0
        except Exception:
            _GL = False
    return _GL


@contextlib.contextmanager
def world_env(world: str = "orchard_lite", **extra):
    """Compose ``world`` with person actors allowed (noVNC down), restoring the environment afterwards."""
    from atharv_nandi.core import camera_rig
    keys = {"NANDI_WORLD": world, "NANDI_NOVNC_UP": "0", **extra}
    old = {k: os.environ.get(k) for k in keys}
    old_wid = camera_rig._WORLD_ID
    os.environ.update(keys)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        camera_rig._WORLD_ID = old_wid


def set_ext(m, ext: float, tilt_deg: float = 0.0, pan_deg: float = 0.0):
    """Crown mast extension + gimbal written the way the WP24 provider writes them."""
    from atharv_nandi.core import mast as mc
    from atharv_nandi.sim.hooks import mast as mh
    for b, z in zip([m.body(s).id for s in mh.SECTIONS], mc.split(ext)):
        m.body_pos[b, 2] = z
    m.cam_quat[m.camera("crown_cam").id] = mh.gimbal_quat(math.radians(pan_deg), math.radians(tilt_deg))


def place_robot(m, d, x: float, y: float, yaw: float = 0.0):
    d.qpos[0], d.qpos[1] = x, y
    d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)


def place_person(m, d, xy, yaw: float = 0.0, body: str = "actor_person_worker_1"):
    mid = int(m.body_mocapid[m.body(body).id])
    d.mocap_pos[mid] = [xy[0], xy[1], 0.0]
    d.mocap_quat[mid] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)


def cam_forward_xy(d, cid: int) -> np.ndarray:
    f = -d.cam_xmat[cid].reshape(3, 3)[:, 2]
    v = np.array([f[0], f[1]])
    return v / max(1e-9, np.linalg.norm(v))



# ------------------------------------------------------------------------------------------- LUT geometry
def test_every_pano_pixel_covered_and_seams_overlap():
    lut = pano.PanoLUT(pano.nominal_ring(160, 120))
    assert lut.uncovered() == 0                       # every pixel seen by >= 1 camera
    ov = lut.seam_overlap_deg()                       # (4 seams, H rows)
    el = lut.el
    low = el <= -1.0
    assert ov.shape == (4, pano.PANO_H)
    assert ov[:, low].min() >= 5.0                    # >= 5 deg at each seam on every row at or below -1 deg
    assert np.mean(low) >= 0.89                       # that is about 90 % of the strip rows (computed)
    assert ov.min() > 1.0                             # and the top rows still overlap (no seam gap anywhere)
    # on the centre row the neighbours overlap by about 10 deg (HFOV 100 at 90 deg spacing)
    centre = int(np.argmin(np.abs(el + 25.0)))
    assert ov[:, centre].min() >= 10.0


def test_spec_band_minus67_plus17_would_leave_corners_uncovered():
    """Why the strip uses -56..+5: the spec's -67..+17 is the centre-column FOV (computed, documented)."""
    lut = pano.PanoLUT(pano.nominal_ring(160, 120), el_min=-67.0, el_max=17.0)
    assert lut.uncovered() > 1000


def test_project_backproject_round_trip():
    rng = np.random.default_rng(3)
    for cam in pano.nominal_ring(160, 120) + [pano.RingCamera("crown", 33.0, -10.0, 60.0, 320, 240)]:
        u = rng.uniform(0, cam.width, 500)
        v = rng.uniform(0, cam.height, 500)
        d = cam.backproject(u, v)
        u2, v2, front = cam.project(d)
        assert front.all()
        assert np.max(np.abs(u2 - u)) < 1e-9 and np.max(np.abs(v2 - v)) < 1e-9
        d2 = cam.backproject(u2, v2)
        assert np.max(np.abs(d2 - d)) < 1e-9
    assert abs(pano.nominal_ring()[0].hfov_deg - 100.0) < 0.05


def test_ring_from_model_matches_mjcf():
    with world_env("orchard_lite"):
        m = camera_rig.build_perception_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ring = pano.ring_from_model(m, d, 160, 120)
    for k, cam in enumerate(ring):
        yaw = (cam.yaw_deg - 90.0 * k + 180.0) % 360.0 - 180.0
        assert abs(yaw) < 0.5 and abs(cam.pitch_down_deg - 25.0) < 0.5 and abs(cam.fovy_deg - 83.6) < 1e-6


def test_stitch_feather_self_mask_fill_and_grey():
    ring = pano.nominal_ring(160, 120)
    lut = pano.PanoLUT(ring)
    colours = {"ring_0": (200, 0, 0), "ring_1": (0, 200, 0), "ring_2": (0, 0, 200), "ring_3": (200, 200, 0)}
    imgs = {n: np.full((120, 160, 3), c, dtype=np.uint8) for n, c in colours.items()}
    out, filled = lut.stitch(imgs)
    assert filled.all()
    W = pano.PANO_W
    row = int(np.argmin(np.abs(lut.el + 25.0)))
    col0 = W // 2                                     # straight ahead: ring_0 only
    assert tuple(out[row, col0]) == colours["ring_0"]
    # seam at azimuth +45 deg (ring_0 / ring_1): a blend, neither pure colour
    c45 = int(round((180.0 - 45.0) / 360.0 * W - 0.5))
    px = out[row, c45].astype(int)
    assert px[0] > 20 and px[1] > 20
    # self-mask ring_0's left half: those directions are filled from ring_1 where it overlaps, else grey
    masks = {n: np.zeros((120, 160), dtype=bool) for n in colours}
    masks["ring_0"][:, :] = True
    out2, filled2 = lut.stitch(imgs, masks)
    assert tuple(out2[row, col0]) == (pano.GREY,) * 3 and not filled2[row, col0]
    assert out2[row, c45][1] > 150 and out2[row, c45][0] < 30        # ring_1 fills the overlap
    # a missing camera leaves its sector grey
    out3, filled3 = lut.stitch({k: v for k, v in imgs.items() if k != "ring_2"})
    assert not filled3[row, 5] and tuple(out3[row, 5]) == (pano.GREY,) * 3


def test_pano_jpeg_under_512kib_and_sector_ages():
    lut = pano.PanoLUT(pano.nominal_ring(160, 120))
    rng = np.random.default_rng(1)
    imgs = {n: rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for n in pano.RING_CAMS}   # worst case: noise
    out, _ = lut.stitch(imgs)
    data = pano.encode_jpeg(out, 70)
    assert data[:2] == b"\xff\xd8" and len(data) < 512 * 1024
    ages = pano.SectorAges()
    ages.update("ring_0", 10.0)
    ages.update("ring_9", 3.0)                         # unknown camera ignored
    a = ages.ages(11.5)
    assert a["ring_0"] == 1.5 and a["ring_1"] is None and set(a) == set(pano.RING_CAMS)


def test_stitch_rejects_wrong_size():
    lut = pano.PanoLUT(pano.nominal_ring(160, 120))
    with pytest.raises(ValueError):
        lut.stitch({"ring_0": np.zeros((240, 320, 3), dtype=np.uint8)})


# ------------------------------------------------------------------------------------------- crown camera
def test_crown_footprint_at_1_5_m_within_5_percent_of_7_2_mm_per_px():
    with world_env("orchard_lite"):
        m = camera_rig.build_perception_model()
    c = camera_rig.camera_id(m, camera_rig.CROWN_CAM)
    fp = camera_rig.footprint_mm_per_px(float(m.cam_fovy[c]), 320, 240, 1.5)
    assert abs(fp - 7.2) / 7.2 <= 0.05, fp
    assert abs(camera_rig.footprint_mm_per_px(float(m.cam_fovy[c]), 320, 240, 0.9) - 4.3) < 0.1
    ring = camera_rig.camera_id(m, "ring_0")
    assert abs(camera_rig.footprint_mm_per_px(float(m.cam_fovy[ring]), 320, 240, 1.5) - 11.2) < 0.3  # OS 2.5 ~11


# ------------------------------------------------------------------------------------------- self-mask (pure)
def test_dilate_fill_convex_and_coverage():
    m = np.zeros((20, 20), dtype=bool)
    m[10, 10] = True
    d2 = self_mask.dilate(m, 2)
    assert d2.sum() == 25 and d2[8, 8] and d2[12, 12] and not d2[7, 10]
    tri = self_mask.fill_convex((20, 20), np.array([[2.0, 2.0], [17.0, 2.0], [2.0, 17.0]]))
    assert tri[3, 3] and not tri[16, 16] and tri.sum() > 80
    assert self_mask.coverage(d2, m) == 1.0 and self_mask.coverage(m, d2) < 1.0
    assert self_mask.coverage(m, np.zeros_like(m)) == 1.0
    s = self_mask.StaticSelfMask(1)
    seg = np.full((20, 20, 2), -1, dtype=np.int32)
    seg[5, 5] = [3, int(mujoco.mjtObj.mjOBJ_GEOM)]
    robot = np.array([False, False, False, True])
    mk = s.build("ring_0", [seg], robot)
    assert mk[4, 4] and mk[6, 6] and mk.sum() == 9
    assert s.get("ring_0", np.ones((20, 20), dtype=bool)).all() and s.get("nope") is None


# ------------------------------------------------------------------------------------------- self-mask (GL)
@pytest.fixture(scope="module")
def lite():
    with world_env("orchard_lite"):
        m = camera_rig.build_perception_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


@pytest.mark.gl
def test_self_mask_valid_at_ext_0_0p6_1p6(lite):
    """OS 6 / WP5: the static ring masks (built once at ext 0) still cover every robot pixel at ext 0, 0.6 and
    1.6 m; the crown camera, tilted down at the robot, is masked from its own seg pass at every extension."""
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    m, d = lite
    robot = self_mask.robot_geoms(m)
    ren = camera_rig.CameraRenderer(m, "ring_0", 160, 120, cull_dist=4.0)
    crown = camera_rig.CameraRenderer(m, camera_rig.CROWN_CAM, 320, 240, cull_dist=4.0)
    sm = self_mask.StaticSelfMask(2)
    report = {}
    try:
        set_ext(m, 0.0)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        for n in camera_rig.RING_CAMS:
            sm.build(n, [ren.use(n).seg(d)], robot)
        assert sm.masks["ring_0"].sum() > 0          # the front ring camera sees the basket / gantry
        for ext in (0.0, 0.6, 1.6):
            set_ext(m, ext, tilt_deg=-60.0)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            for n in camera_rig.RING_CAMS:
                rp = self_mask.seg_robot(ren.use(n).seg(d), robot)
                cov = self_mask.coverage(sm.get(n), rp)
                report[(ext, n)] = (int(rp.sum()), cov)
                assert cov == 1.0, (ext, n, cov)
            cseg = crown.seg(d)
            rp = self_mask.seg_robot(cseg, robot)
            cm = self_mask.mask_from_seg(cseg, robot, 2)
            assert rp.sum() > 0 and self_mask.coverage(cm, rp) == 1.0
            report[(ext, "crown")] = int(rp.sum())
        # the crown sees less of the robot the higher it is
        assert report[(0.0, "crown")] > report[(0.6, "crown")] > report[(1.6, "crown")]
    finally:
        set_ext(m, 0.0)
        ren.close()
        crown.close()
    print("self-mask report", report)


@pytest.mark.gl
def test_arm_hull_mask_covers_arm_pixels_in_front_ring(lite):
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    m, d = lite
    arm = self_mask.arm_geoms(m)
    assert arm
    arm_set = np.zeros(m.ngeom, dtype=bool)
    arm_set[arm] = True
    ren = camera_rig.CameraRenderer(m, "ring_0", 160, 120, cull_dist=4.0)
    try:
        seg = ren.seg(d)
    finally:
        ren.close()
    arm_px = self_mask.seg_robot(seg, arm_set)
    cid = camera_rig.camera_id(m, "ring_0")
    hull = self_mask.arm_hull_mask(m, d, camera_rig.camera_pose(d, cid), camera_rig.intrinsics(m, "ring_0", 160, 120),
                                   arm)
    assert self_mask.coverage(hull, arm_px) == 1.0
    assert hull.sum() < hull.size                        # a hull, not the whole frame


@pytest.mark.gl
def test_rendered_ring_depth_band_matches_the_sim_band_rays(lite):
    """OS 2.5 cross-check, both paths: the band from a rendered ring depth frame (perception process) and the
    band from the ring pixel rays (sim process, ``RingDepthBand``) give the same trunk centre."""
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    from atharv_nandi.core import lidar_model as lm
    m, d = lite
    tc = d.xpos[m.body("tree_L1-1").id][:2].copy()
    L = lm.LidarModel(m)
    res = {}
    try:
        for side in (-1.0, 1.0):
            place_robot(m, d, tc[0], tc[1] + side * 1.2)
            gz = float(d.xpos[L.trunk][2]) - lm.NOMINAL_TRUNK_Z_M
            cam = camera_rig.ring_facing(m, d, tc)
            B = lm.RingDepthBand(L, cams=[cam], noise_sigma=0.0)
            P, O, cams = B.points(d, tc, gz, noise=False)
            assert cams == [cam]
            ray = lm.band_cross_check(P, O, tc, 0.055)
            cid = camera_rig.camera_id(m, cam)
            ren = camera_rig.CameraRenderer(m, cam, 160, 120, cull_dist=4.0)
            try:
                depth = ren.depth(d)
            finally:
                ren.close()
            Pr = lm.depth_band_points(depth, d.cam_xpos[cid], lm.optical_rot(d, cid),
                                      camera_rig.intrinsics(m, cam, 160, 120), gz, max_depth=4.0)
            rnd = lm.band_cross_check(Pr, d.cam_xpos[cid], tc, 0.055)
            assert ray["ok"] and rnd["ok"], (side, ray, rnd)
            cr, cd = np.asarray(ray["centre"]), np.asarray(rnd["centre"])
            res[side] = (cam, len(P), len(Pr), float(np.linalg.norm(cr - cd)), float(np.linalg.norm(cd - tc)))
            assert np.linalg.norm(cr - cd) <= 0.005 and np.linalg.norm(cd - tc) <= 0.01, res[side]
    finally:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
    print("ring band render vs rays", res)
