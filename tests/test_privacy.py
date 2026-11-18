"""WP5 privacy filter, FilteredFrame-only sinks, MP4 / strip assemblers, parcel mask and the mjvScene cull
(MVP-SPEC 5.4, 5.9; ORCHARD-MVP-SPEC 3.7, 5.6, 6)."""
import math
import struct

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import camera_rig, lidar_model as lm, privacy_filter as pf, scene_cull, self_mask
from atharv_nandi.core.detector_model import Pose, intrinsics_from_fovy
from atharv_nandi.sim.world import actors
from test_pano import cam_forward_xy, gl_available, place_person, place_robot, world_env

GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)


def _fwd_camera(pos=(0.0, 0.0, 1.0)):
    """Optical frame looking along world +x (x right = -y, y down = -z)."""
    R_cam_in_world = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]]).astype(float)
    return Pose.from_frame_in_world(np.asarray(pos, float), R_cam_in_world)


def _frame(**kw):
    f = pf.PrivacyFilter(**kw)
    intr = intrinsics_from_fovy(60.0, 160, 120)
    return f, intr, _fwd_camera()


# ------------------------------------------------------------------------------------------- FilteredFrame gate
def test_filtered_frame_only_from_the_filter_and_sinks_refuse_raw():
    with pytest.raises(pf.PrivacyError):
        pf.FilteredFrame(object(), "x", 0.0, np.zeros((2, 2, 3)), None, np.zeros((2, 2)), {})
    f, intr, pose = _frame()
    res = f.apply("head_cam", 1.0, np.full((120, 160, 3), 77, np.uint8), pose, intr)
    fr = res.frame
    assert isinstance(fr, pf.FilteredFrame) and fr.provenance == "simulated"
    with pytest.raises(ValueError):
        fr.rgb[0, 0, 0] = 1                                     # read-only
    got = []
    sink = pf.FilteredSink(got.append, "ros")
    sink.publish(fr)
    for raw in (np.zeros((120, 160, 3), np.uint8), fr.rgb, b"jpeg", {"rgb": fr.rgb}):
        with pytest.raises(pf.PrivacyError):
            sink.publish(raw)                                  # unfiltered publish refused
    assert got == [fr] and sink.count == 1
    assert pf.require_filtered(fr) is fr
    assert fr.jpeg()[:2] == b"\xff\xd8"


def test_truth_mask_detector_mask_and_drop_rules():
    f, intr, pose = _frame(dilate_px=2, expand_frac=0.2)
    rgb = np.full((120, 160, 3), 200, np.uint8)
    person = pf.geom_box_corners(int(mujoco.mjtGeom.mjGEOM_BOX), np.array([0.2, 0.2, 0.85]), np.array([3.0, 0.0, 0.85]),
                                 np.eye(3))
    r = f.apply("ring_0", 0.0, rgb, pose, intr, persons=[person])
    uv, z = pf.project_points(person, pose, intr)
    u0, v0 = np.floor(uv.min(0)).astype(int)
    u1, v1 = np.ceil(uv.max(0)).astype(int)
    m = r.frame.masked
    assert m[max(v0, 0):min(v1, 119), u0:u1].all()            # the projected box is masked
    assert r.frame.rgb[60, (u0 + u1) // 2].tolist() == [pf.GREY] * 3
    assert r.stats["truth_pct"] > 0 and r.stats["detector_pct"] == 0
    # physics mode: the truth mask is not used; the detector mask (cluster) is, expanded by 20 %
    fp = pf.PrivacyFilter(mode="physics", expand_frac=0.2)
    cl = lm.Cluster(np.array([2.8, -0.2, 0.0]), np.array([3.2, 0.2, 2.0]), 12, np.array([3.0, 0, 1.0]))
    r2 = fp.apply("ring_0", 0.0, rgb, pose, intr, persons=[person], clusters=[cl])
    assert r2.stats["truth_pct"] == 0 and r2.stats["detector_pct"] > 0
    uvc, _ = pf.project_points(cl.corners(), pose, intr)
    w = uvc[:, 0].max() - uvc[:, 0].min()
    left = int(math.floor(uvc[:, 0].min() - 0.09 * w))
    assert r2.frame.masked[60, max(left, 0)]                   # 20 % expansion reaches past the raw box
    r3 = fp.apply("ring_0", 0.0, rgb, pose, intr, clusters=[cl.as_dict()])          # dict clusters too
    assert r3.stats["detector_pct"] == r2.stats["detector_pct"]
    # a cluster partly behind the camera, or closer than 0.3 m, cannot be bounded: the frame is dropped
    near = lm.Cluster(np.array([-0.5, -0.3, 0.0]), np.array([0.6, 0.3, 2.0]), 5, np.zeros(3))
    r4 = fp.apply("ring_0", 0.0, rgb, pose, intr, clusters=[near])
    assert r4.frame is None and r4.dropped and r4.reason == "cluster_unbounded"
    close = lm.Cluster(np.array([0.2, -0.1, 0.8]), np.array([0.25, 0.1, 1.2]), 5, np.zeros(3))
    assert fp.apply("ring_0", 0.0, rgb, pose, intr, clusters=[close]).reason == "cluster_unbounded"
    # out of view (behind) is ignored, not a drop
    behind = lm.Cluster(np.array([-3.0, -0.2, 0.0]), np.array([-2.6, 0.2, 2.0]), 5, np.zeros(3))
    assert fp.apply("ring_0", 0.0, rgb, pose, intr, clusters=[behind]).frame is not None
    # truth person straddling the camera plane drops in assisted mode
    pnear = pf.geom_box_corners(int(mujoco.mjtGeom.mjGEOM_BOX), np.array([0.5, 0.2, 0.85]), np.array([0.1, 0, .85]),
                                np.eye(3))
    assert f.apply("ring_0", 0.0, rgb, pose, intr, persons=[pnear]).reason == "person_unbounded"
    assert fp.counts["dropped"] == 2 and f.counts["dropped"] == 1
    with pytest.raises(ValueError):
        pf.PrivacyFilter(mode="nope")


def test_parcel_mask_drops_pixels_beyond_the_polygon():
    intr = intrinsics_from_fovy(60.0, 160, 120)
    pose = _fwd_camera((0.0, 0.0, 1.0))
    depth = np.full((120, 160), 4.0, np.float32)               # a wall 4 m ahead, x = 4
    poly = [[-1.0, -10.0], [10.0, -10.0], [10.0, 0.0], [-1.0, 0.0]]   # parcel only at y <= 0 (image right half)
    m = pf.parcel_mask(depth, pose, intr, poly)
    assert m[:, :80].all() and not m[:, 81:].any()
    f = pf.PrivacyFilter(parcel=poly)
    r = f.apply("crown_cam", 0.0, np.full((120, 160, 3), 50, np.uint8), pose, intr, depth=depth)
    assert 49.0 <= r.stats["parcel_pct"] <= 51.5 and r.stats["masked_pct"] == r.stats["parcel_pct"]
    assert (r.frame.rgb[:, :80] == pf.GREY).all() and (r.frame.rgb[:, 82:] == 50).all()
    assert np.isnan(r.frame.depth[:, :80]).all()
    # no-return pixels are not touched
    d2 = depth.copy()
    d2[:, :40] = np.inf
    assert not pf.parcel_mask(d2, pose, intr, poly)[:, :40].any()
    assert pf.points_in_polygon(np.array([0.5, 20.0]), np.array([-1.0, -1.0]), poly).tolist() == [True, False]


# ------------------------------------------------------------------------------------------- strips and MP4
def _filtered(n=3, shape=(60, 80)):
    f = pf.PrivacyFilter()
    intr = intrinsics_from_fovy(60.0, shape[1], shape[0])
    out = []
    for k in range(n):
        rgb = np.zeros(shape + (3,), np.uint8)
        rgb[..., k % 3] = 40 + 60 * k
        out.append(f.apply("crown_cam", float(k), rgb, _fwd_camera(), intr).frame)
    return out


def test_mp4_assembler_accepts_only_filtered_frames():
    frames = _filtered(4)
    mp4 = pf.Mp4Assembler(fps=4.0, prefer_ffmpeg=False)
    with pytest.raises(pf.PrivacyError):
        mp4.add(np.zeros((60, 80, 3), np.uint8))
    with pytest.raises(pf.PrivacyError):
        mp4.add(frames[0].rgb)
    with pytest.raises(ValueError):
        pf.Mp4Assembler().encode()
    for f in frames:
        mp4.add(f)
    with pytest.raises(ValueError):
        mp4.add(_filtered(1, (30, 40))[0])                      # one size per MP4
    data, enc = mp4.encode()
    assert enc == "mjpeg_mp4"
    boxes = pf.mp4_boxes(data)
    assert [b[0] for b in boxes] == [b"ftyp", b"mdat", b"moov"] and sum(b[2] for b in boxes) == len(data)
    moov = data[boxes[2][1]:]
    i = moov.index(b"stsz")
    count = struct.unpack(">I", moov[i + 12:i + 16])[0]
    sizes = struct.unpack(f">{count}I", moov[i + 16:i + 16 + 4 * count])
    assert count == 4
    i = moov.index(b"stco")
    off = struct.unpack(">I", moov[i + 12:i + 16])[0]
    for s in sizes:                                           # every sample is a JPEG
        assert data[off:off + 2] == b"\xff\xd8" and data[off + s - 2:off + s] == b"\xff\xd9"
        off += s
    assert b"jpeg" in moov and b"vide" in moov


def test_strip_and_compose_take_only_filtered():
    frames = _filtered(3)
    st = pf.StripAssembler("strip_mid", height=30)
    with pytest.raises(pf.PrivacyError):
        st.add(np.zeros((60, 80, 3), np.uint8))
    with pytest.raises(ValueError):
        st.build()
    for f in frames:
        st.add(f)
    strip = st.build(5.0)
    assert isinstance(strip, pf.FilteredFrame) and strip.shape == (30, 120) and strip.stats["sources"]
    with pytest.raises(pf.PrivacyError):
        pf.compose_gather("pano", 0.0, [np.zeros((4, 4, 3))], (4, 4), [(np.arange(16), np.arange(16), np.ones(16))])
    with pytest.raises(pf.PrivacyError):
        pf.compose_gather("pano", 0.0, [], (4, 4), [])


# ------------------------------------------------------------------------------------------- GL: world fixture
@pytest.fixture(scope="module")
def lite():
    with world_env("orchard_lite"):
        m = camera_rig.build_perception_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def _reset(m, d):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)


def _person_pixels(seg, m):
    return np.isin(seg[..., 0], actors.person_geoms(m)) & (seg[..., 1] == GEOM)


# ------------------------------------------------------------------------------------------- cull
@pytest.mark.gl
def test_scene_cull_reduces_geoms_keeps_actors_and_model_unchanged(lite):
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    m, d = lite
    place_robot(m, d, 0.0, 0.0)
    place_person(m, d, (9.0, 0.5))                             # 9 m away: beyond the 4 m cull, kept anyway
    fp0 = scene_cull.model_fingerprint(m)
    plain = camera_rig.CameraRenderer(m, "ring_0", 160, 120)
    fast = camera_rig.CameraRenderer(m, "ring_0", 160, 120, cull_dist=4.0)
    slow = camera_rig.CameraRenderer(m, "ring_0", 160, 120)
    slow.culler = scene_cull.SceneCuller(m, slow.renderer.scene, 4.0, force_slow=True)
    try:
        img_plain = plain.rgb(d)
        n_all = plain.last_cull[1]
        img_fast = fast.rgb(d)
        b, a = fast.last_cull
        objs_fast = [fast.renderer.scene.geoms[i].objid for i in range(a)]
        img_slow = slow.rgb(d)
        objs_slow = [slow.renderer.scene.geoms[i].objid for i in range(slow.last_cull[1])]
        assert fast.culler.fast and not slow.culler.fast
        assert b == n_all and a < b                            # scene geom count reduced
        assert objs_fast == objs_slow and np.array_equal(img_fast, img_slow)   # both compaction paths agree
        pg = set(actors.person_geoms(m))
        assert pg <= set(objs_fast)                            # persons (mocap) never culled
        seg = fast.seg(d)                                      # segid renumbered: the seg decode works
        assert seg.shape == (120, 160, 2) and seg[..., 0].max() < m.ngeom
        assert scene_cull.model_fingerprint(m) == fp0          # model fields unchanged
        assert img_plain.shape == img_fast.shape
        print(f"cull ring_0 at home: {b} -> {a} scene geoms")
    finally:
        for r in (plain, fast, slow):
            r.close()
        _reset(m, d)


# ------------------------------------------------------------------------------------------- positive control
POSES = {   # camera -> (distance along its ground heading, lateral offset)
    "head_cam": (1.8, 0.0), "ring_0": (2.6, 0.0), "ring_1": (2.4, 0.3), "ring_2": (2.4, -0.3), "ring_3": (2.8, 0.0),
    "crown_cam": (3.0, 0.2),
}


def _lidar_clusters(m, d):
    L = lm.LidarModel(m, seed=7)
    return lm.nonmap_clusters(L.sweep(d, lm.SAFETY_GROUPS, noise=True).values(), L.map_geom,
                              d.xpos[L.trunk][:2])


@pytest.mark.gl
def test_gl_positive_control_person_masked_100pct_in_every_camera(lite):
    """MVP-SPEC 5.4: a person at known poses in view of every camera; the seg pass must show >= 200 person pixels
    (else the test fails: the person was never visible); the truth mask and the detector mask each cover 100 %."""
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    m, d = lite
    place_robot(m, d, 0.0, -2.0, yaw=0.0)                      # alley between the hedgerows, facing +x
    ren = camera_rig.CameraRenderer(m, "ring_0", 160, 120, cull_dist=4.0)
    truth_f = pf.PrivacyFilter("assisted", dilate_px=4)
    det_f = pf.PrivacyFilter("physics", expand_frac=0.2)
    report = {}
    try:
        for cam, (dist, lat) in POSES.items():
            for k, yaw_off in enumerate((0.0, 1.2)):          # two poses per camera, turned person
                cid = camera_rig.camera_id(m, cam)
                fwd = cam_forward_xy(d, cid)
                side = np.array([-fwd[1], fwd[0]])
                xy = d.cam_xpos[cid][:2] + fwd * dist + side * (lat + 0.25 * k)
                place_person(m, d, xy, yaw=math.atan2(fwd[1], fwd[0]) + yaw_off)
                ren.use(cam)
                seg = ren.seg(d)
                rgb, depth = ren.rgbd(d)
                person = _person_pixels(seg, m)
                n = int(person.sum())
                assert n >= 200, f"{cam} pose {k}: only {n} person pixels: the person was never visible"
                pose = camera_rig.camera_pose(d, cid)
                intr = camera_rig.intrinsics(m, cam, 160, 120)
                rt = truth_f.apply(cam, 0.0, rgb, pose, intr, depth, pf.person_boxes(m, d), [])
                cl = _lidar_clusters(m, d)
                rd = det_f.apply(cam, 0.0, rgb, pose, intr, depth, [], cl)
                assert rt.frame is not None and rd.frame is not None, (cam, rt.reason, rd.reason)
                tcov = self_mask.coverage(rt.frame.meta["masks"]["truth"], person)
                dcov = self_mask.coverage(rd.frame.meta["masks"]["detector"], person)
                report[(cam, k)] = (n, tcov, dcov)
                assert tcov == 1.0 and dcov == 1.0, (cam, k, n, tcov, dcov)
                assert not np.any((rt.frame.rgb != pf.GREY).any(axis=-1) & person)   # no person pixel survives
    finally:
        ren.close()
        _reset(m, d)
    recall = np.mean([v[2] for v in report.values()])
    print("positive control (person px, truth cov, detector cov):", report, "detector recall", recall)


@pytest.mark.gl
def test_gl_worker_behind_canopy_masked_100pct(lite):
    """OS WP5: a worker partly visible through a canopy is masked 100 % (safety rays see through foliage)."""
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    m, d = lite
    tc = d.xpos[m.body("tree_L1-1").id][:2].copy()
    place_robot(m, d, tc[0], tc[1] - 2.0, yaw=math.pi / 2)    # facing the tree: ring_0 looks at it
    cid = camera_rig.camera_id(m, "ring_0")
    ren = camera_rig.CameraRenderer(m, "ring_0", 160, 120, cull_dist=4.0)
    try:
        best = None
        for off in (0.7, 0.9, 1.1, 1.3):                       # behind the canopy along the view, small sideways
            for lat in (0.0, 0.35, -0.35):
                place_person(m, d, (tc[0] + lat, tc[1] + off), yaw=-math.pi / 2)
                seg = ren.seg(d)
                n = int(_person_pixels(seg, m).sum())
                if n >= 30 and (best is None or n < best[0]):
                    best = (n, off, lat)
        assert best is not None, "worker never visible through the canopy"
        n, off, lat = best
        place_person(m, d, (tc[0] + lat, tc[1] + off), yaw=-math.pi / 2)
        seg = ren.seg(d)
        person = _person_pixels(seg, m)
        rgb, depth = ren.rgbd(d)
        canopy = np.isin(seg[..., 0], np.nonzero(m.geom_group == 3)[0]) & (seg[..., 1] == GEOM)
        pose = camera_rig.camera_pose(d, cid)
        intr = camera_rig.intrinsics(m, "ring_0", 160, 120)
        rt = pf.PrivacyFilter("assisted").apply("ring_0", 0.0, rgb, pose, intr, depth, pf.person_boxes(m, d), [])
        rd = pf.PrivacyFilter("physics").apply("ring_0", 0.0, rgb, pose, intr, depth, [], _lidar_clusters(m, d))
        assert canopy.sum() > 100                              # the canopy really is in front
        assert self_mask.coverage(rt.frame.masked, person) == 1.0
        assert self_mask.coverage(rd.frame.masked, person) == 1.0
        print(f"worker through canopy: {n} visible person px at +{off} m / lateral {lat}; masked 100 %")
    finally:
        ren.close()
        _reset(m, d)


@pytest.mark.gl
def test_gl_pixels_beyond_parcel_polygon_dropped(lite):
    """OS 5.6: crown pixels whose depth falls beyond the field geofence are dropped before any sink."""
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    import sys
    import os
    from atharv_nandi.nodes import perception_node
    fence = perception_node.field_geofence("orchard_lite")
    assert fence and len(fence) >= 3
    m, d = lite
    xmax = max(p[0] for p in fence)
    place_robot(m, d, xmax - 1.5, 0.0, yaw=0.0)                # 1.5 m inside the east edge, facing out
    from test_pano import set_ext
    set_ext(m, 0.6, tilt_deg=-30.0)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    cid = camera_rig.camera_id(m, camera_rig.CROWN_CAM)
    ren = camera_rig.CameraRenderer(m, camera_rig.CROWN_CAM, 320, 240, cull_dist=4.0)
    try:
        rgb, depth = ren.rgbd(d)
    finally:
        ren.close()
    pose = camera_rig.camera_pose(d, cid)
    intr = camera_rig.intrinsics(m, camera_rig.CROWN_CAM, 320, 240)
    r = pf.PrivacyFilter("assisted", parcel=fence).apply("crown_cam", 0.0, rgb, pose, intr, depth,
                                                         pf.person_boxes(m, d), [])
    parcel = r.frame.meta["masks"]["parcel"]
    assert r.stats["parcel_pct"] > 5.0                         # ground beyond the fence is in view
    # every dropped pixel lies beyond the fence; every kept ground pixel inside
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    z = depth.astype(float)
    P = np.stack([(uu + 0.5 - intr.cx) / intr.fx * z, (vv + 0.5 - intr.cy) / intr.fy * z, z], -1).reshape(-1, 3)
    Pw = pose.inverse_apply(P).reshape(H, W, 3)
    assert (Pw[parcel][:, 0] > xmax - 1e-6).all()
    finite = np.isfinite(z) & (z < 50.0)
    assert (Pw[finite & ~parcel][:, 0] <= xmax + 1e-6).all()
    assert (r.frame.rgb[parcel] == pf.GREY).all()
    set_ext(m, 0.0)
    _reset(m, d)
    print(f"parcel mask: {r.stats['parcel_pct']:.1f} % of the crown frame dropped beyond x = {xmax}")


# ------------------------------------------------------------------------------------------- edge cases
def test_self_mask_and_cull_edge_cases():
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><geom type="plane" size="5 5 .1"/>
      <body name="trunk"><freejoint/><geom size=".1"/><body name="arm_link"><geom size=".05" pos=".3 0 0"/></body>
      </body><body name="tool_0" pos="1 0 0"><freejoint/><geom size=".02"/></body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with pytest.raises(KeyError):
        self_mask.subtree_bodies(m, "nope")
    rg = self_mask.robot_geoms(m)
    assert rg.tolist() == [False, True, True, True]           # trunk subtree + the tool free body
    assert not self_mask.fill_convex((10, 10), np.array([[1.0, 1.0], [2.0, 2.0]])).any()
    assert not self_mask.fill_convex((10, 10), np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])).any()
    assert not self_mask.fill_convex((10, 10), np.array([[-9.0, -9.0], [-8.0, -9.0], [-8.0, -8.0]])).any()
    with pytest.raises(ValueError):
        self_mask.StaticSelfMask().build("ring_0", [], rg)
    intr = intrinsics_from_fovy(60.0, 64, 48)
    assert not self_mask.arm_hull_mask(m, d, _fwd_camera((0, 0, 0)), intr, geoms=[]).any()
    behind = _fwd_camera((5.0, 0.0, 0.0))                     # the arm is behind this camera
    assert not self_mask.arm_hull_mask(m, d, behind, intr, geoms=[2]).any()
    straddle = _fwd_camera((0.3, 0.0, 0.0))                   # the camera sits inside the arm geom box
    assert self_mask.arm_hull_mask(m, d, straddle, intr, geoms=[2]).any()
    # cull with an empty scene and the slow path
    sc = mujoco.MjvScene(m, maxgeom=1)
    cu = scene_cull.SceneCuller(m, sc, 4.0)
    assert not cu.fast and cu.keep_mask([0, 0, 0]).size == 0 and cu.cull([0, 0, 0]) == (0, 0)
    sc2 = mujoco.MjvScene(m, maxgeom=50)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, sc2)
    n = sc2.ngeom
    assert scene_cull.SceneCuller(m, sc2, 100.0).cull([0, 0, 0]) == (n, n)
    assert pf._clipped_reaches_image(np.array([[0.0, 0.0, -1.0]]), intr) is False


# ------------------------------------------------------------------------------------------- review round 1
def test_compose_takes_no_caller_pixels_and_frames_stay_read_only():
    """WP5 review: the old compose(name, stamp, rgb, sources) turned any raw array into a FilteredFrame next to one
    unrelated source; FilteredFrame arrays could be made writable again and attributes re-assigned."""
    src = _filtered(1, (4, 4))[0]
    raw = np.full((480, 640, 3), 200, np.uint8)
    with pytest.raises(pf.PrivacyError):
        pf.compose("anything", 0.0, raw, [src])                 # the raw-rgb entry point is gone
    for arr in (src.rgb, src.masked):
        with pytest.raises(ValueError):
            arr.setflags(write=True)
        b = arr
        while isinstance(b, np.ndarray):                       # no view in the chain can be made writable
            with pytest.raises(ValueError):
                b.setflags(write=True)
            b = b.base
        assert isinstance(b, bytes)                            # backed by immutable bytes
    with pytest.raises(pf.PrivacyError):
        src.rgb = raw
    with pytest.raises(pf.PrivacyError):
        del src.masked
    # every composed pixel is a source pixel (weighted mean); out-of-range tables are refused
    g = pf.compose_gather("x", 0.0, [src], (2, 2), [(np.arange(4), np.array([0, 5, 10, 15]), np.ones(4))])
    assert np.array_equal(g.rgb.reshape(-1, 3), src.rgb.reshape(-1, 3)[[0, 5, 10, 15]])
    with pytest.raises(ValueError):
        pf.compose_gather("x", 0.0, [src], (2, 2), [(np.arange(4), np.array([0, 5, 10, 99]), np.ones(4))])
    with pytest.raises(ValueError):
        pf.compose_gather("x", 0.0, [src], (2, 2), [(np.arange(4), np.arange(4), -np.ones(4))])
    # a masked source pixel never reaches the output: it is grey and masked there
    f = pf.PrivacyFilter(mode="physics")
    intr = intrinsics_from_fovy(60.0, 160, 120)
    cl = lm.Cluster(np.array([2.8, -0.2, 0.0]), np.array([3.2, 0.2, 2.0]), 12, np.array([3.0, 0, 1.0]))
    fr = f.apply("ring_0", 0.0, np.full((120, 160, 3), 60, np.uint8), _fwd_camera(), intr, clusters=[cl]).frame
    st = pf.StripAssembler("s", height=60)
    st.add(fr)
    st.add(fr)
    strip = st.build(1.0)
    assert strip.shape == (60, 160)
    sub = np.asarray(fr.masked)[::2, ::2]
    assert np.array_equal(strip.masked[:, :80], sub) and (strip.rgb[:, :80][sub] == pf.GREY).all()
    assert (strip.rgb[:, :80][~sub] == 60).all()


def test_compose_pano_matches_the_lut_stitch():
    from atharv_nandi.core import pano
    lut = pano.PanoLUT(pano.nominal_ring(160, 120), 320, 128)
    f = pf.PrivacyFilter()
    intr = intrinsics_from_fovy(60.0, 160, 120)
    rng = np.random.default_rng(3)
    frames, imgs, masks, extra = {}, {}, {}, {}
    for k, cam in enumerate(lut.cams):
        rgb = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        person = pf.geom_box_corners(int(mujoco.mjtGeom.mjGEOM_BOX), np.array([0.2, 0.2, 0.5]),
                                     np.array([3.0, 0.3 * k, 1.0]), np.eye(3))
        fr = f.apply(cam.name, 0.0, rgb, _fwd_camera(), intr, persons=[person]).frame
        frames[cam.name] = fr
        em = np.zeros((120, 160), bool)
        em[100:, :] = True                                     # a self-mask band
        extra[cam.name] = em
        imgs[cam.name] = np.asarray(fr.rgb)
        masks[cam.name] = np.asarray(fr.masked) | em
    p = pf.compose_pano(lut, frames, 1.0, extra, {"ages": {}})
    rgb, filled = lut.stitch(imgs, masks)
    assert np.array_equal(p.rgb, rgb) and np.array_equal(p.masked, ~filled)
    assert p.stats["sources"] == [c.name for c in lut.cams]
    with pytest.raises(pf.PrivacyError):
        pf.compose_pano(lut, {lut.cams[0].name: imgs[lut.cams[0].name]}, 1.0)


def _head_like_camera(height=0.4, pitch_deg=35.0):
    """Optical frame at (0, 0, height) looking along +x, pitched down: x right = -y."""
    a = math.radians(pitch_deg)
    fwd = np.array([math.cos(a), 0.0, -math.sin(a)])
    right = np.array([0.0, -1.0, 0.0])
    down = np.cross(fwd, right)
    return Pose.from_frame_in_world(np.array([0.0, 0.0, height]), np.column_stack([right, down, fwd]))


def test_near_part_out_of_view_does_not_drop_and_the_rest_is_masked():
    """WP5 review: a 2 m cluster top 84 deg off a head camera's 35 deg-down axis has optical z < 0.3 m, which
    dropped the frame for a fully boundable person 1.5-2 m ahead. Now only a near part that reaches the image drops."""
    intr = intrinsics_from_fovy(70.0, 160, 120)
    pose = _head_like_camera()
    rgb = np.full((120, 160, 3), 90, np.uint8)
    for mode in ("physics", "assisted"):
        f = pf.PrivacyFilter(mode=mode)
        for dist in (1.2, 1.5):            # from the camera (about 1.55-1.85 m from the trunk)
            cl = {"lo": [dist - 0.1, -0.15, 0.0], "hi": [dist + 0.1, 0.15, 2.0]}
            person = pf.geom_box_corners(int(mujoco.mjtGeom.mjGEOM_BOX), np.array([0.1, 0.15, 0.85]),
                                         np.array([dist, 0.0, 0.85]), np.eye(3))
            _, z = pf.project_points(pf._dict_corners(cl), pose, intr)
            assert z.min() < 0.3                               # the case that used to drop
            r = f.apply("head_cam", 0.0, rgb, pose, intr, persons=[person], clusters=[cl])
            assert r.frame is not None, (mode, dist, r.reason)
            # every in-image point of the person body (in front of the near plane) is masked
            samp = np.concatenate([person, (person[:4] + person[4:]) / 2.0,
                                   np.array([[dist, y, zz] for y in (-0.15, 0, 0.15) for zz in np.linspace(0, 1.7, 9)])])
            uv, zz = pf.project_points(samp, pose, intr)
            ok = (zz > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < 160) & (uv[:, 1] >= 0) & (uv[:, 1] < 120)
            assert ok.any()
            assert r.frame.masked[uv[ok, 1].astype(int), uv[ok, 0].astype(int)].all(), (mode, dist)
    # a near part that is in view still drops; a box wholly behind the camera is ignored
    f = pf.PrivacyFilter(mode="physics")
    close = {"lo": [0.15, -0.1, 0.1], "hi": [0.25, 0.1, 0.3]}
    assert f.apply("head_cam", 0.0, rgb, pose, intr, clusters=[close]).reason == "cluster_unbounded"
    behind = {"lo": [-2.0, -0.2, 0.0], "hi": [-1.5, 0.2, 2.0]}
    r = f.apply("head_cam", 0.0, rgb, pose, intr, clusters=[behind])
    assert r.frame is not None and r.stats["detector_pct"] == 0
    assert len(pf._clip_z(np.zeros((0, 3)), None, 0.3, True)) == 0
