"""WP5 detectors (ColorBlob, dusk rule), "seen" coverage, InspectionRecord, render scheduler and the perception core
incl. the TREE 360 render mode (MVP-SPEC 2.4, 3.8, 4.2, 5.4, 5.9; ORCHARD-MVP-SPEC 3.7)."""
import json
import os
import sys
import time

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import camera_rig, privacy_filter as pf, render_scheduler as rs
from test_pano import gl_available, place_person, place_robot, set_ext, world_env

import atharv_nandi.nodes.perception_node as pn  # noqa: E402  (ROS-free at import: rclpy only inside main())


# ------------------------------------------------------------------------------------------- ColorBlob
def _disc_image(centres, r=5, bg=(40, 90, 40), fg=(230, 40, 40), shape=(120, 160)):
    img = np.empty(shape + (3,), np.uint8)
    img[:] = bg
    vv, uu = np.mgrid[0:shape[0], 0:shape[1]]
    for (u, v) in centres:
        img[(uu - u) ** 2 + (vv - v) ** 2 <= r * r] = fg
    return img


def test_colorblob_finds_red_discs_and_ignores_masked():
    cs = [(20, 30), (80, 60), (140, 100)]
    img = _disc_image(cs)
    cb = pn.ColorBlob({"min_area_px": 6, "chroma_min": 0.45})
    dets = cb.detect(img, depth=np.full((120, 160), 2.0, np.float32))
    assert len(dets) == 3 and all(abs(d["z"] - 2.0) < 1e-6 for d in dets)
    rec, n = pn.match_recall(dets, np.array(cs, float))
    assert rec == 1.0 and n == 3
    mask = np.zeros((120, 160), bool)
    mask[50:70, 70:90] = True                                  # a privacy-masked region hides one disc
    assert len(cb.detect(img, exclude=mask)) == 2
    assert pn.match_recall([], np.zeros((0, 2))) == (1.0, 0)


def test_label_components_matches_bruteforce():
    rng = np.random.default_rng(5)
    m = rng.random((40, 50)) > 0.7
    lab, n = pn.label_components(m)
    assert (lab > 0).sum() == m.sum() and n == lab.max()
    # 4-connectivity: every labelled pixel shares its label with any 4-neighbour that is set
    for y in range(40):
        for x in range(50):
            if m[y, x]:
                for dy, dx in ((0, 1), (1, 0)):
                    yy, xx = y + dy, x + dx
                    if yy < 40 and xx < 50 and m[yy, xx]:
                        assert lab[y, x] == lab[yy, xx]
    u = np.zeros((5, 5), bool)
    u[0, 0] = u[0, 4] = u[1, 0] = u[1, 4] = u[2, :] = True     # a U joins through the bottom row
    lab, n = pn.label_components(u)
    assert n == 1


def test_degrade_noise_and_blur():
    rng = np.random.default_rng(1)
    img = np.zeros((20, 30, 3), np.uint8)
    img[:, 15:] = 200
    b = pn.degrade(img, 0.0, 3, rng)
    assert b[5, 14, 0] in (66, 67) and b[5, 16, 0] == 200 and b[5, 10, 0] == 0
    flat = np.full((100, 100, 3), 128, np.uint8)
    n = pn.degrade(flat, 8.0 / 255.0, 0, rng).astype(float)
    assert 6.5 < float(n.std()) < 9.5


def test_dusk_rule_falls_back_and_raises_one_assist():
    sel = pn.DetectorSelector(recall_min=0.8, window=5, min_truth=5)
    for _ in range(3):
        assert sel.update(1.0, 4, 1.0) == "colorblob"
    assert sel.take_assist() is None
    for _ in range(5):
        sel.update(0.5, 4, 2.0)
    assert sel.active == "geometry" and sel.episodes == 1
    a = sel.take_assist()
    assert a["reason"] == "colorblob_recall_low" and a["fallback"] == "geometry" and sel.take_assist() is None
    sel.update(0.4, 4, 3.0)
    assert sel.episodes == 1                                  # one assist per fallback episode
    for _ in range(5):
        sel.update(1.0, 4, 4.0)
    assert sel.active == "colorblob"


# ------------------------------------------------------------------------------------------- seen + records
def test_seen_rule_and_inspection_record():
    st = pn.SeenTracker(12.0, 10.0)
    st.observe("p1", 1.0, 14.0, "ring_1", self_masked=False)
    st.observe("p2", 1.0, 30.0, "ring_0", self_masked=True)            # self-masked: does not count
    st.observe("p3", 1.0, 30.0, "wrist_cam", self_masked=False)        # not a ring or head frame
    st.observe("p4", 1.0, 11.9, "head_cam", self_masked=False)         # < 12 px
    st.observe("p5", 1.0, 20.0, "head_cam", self_masked=False, privacy_masked=True)
    assert st.seen("p1", 5.0)[0] and not st.seen("p1", 11.5)[0]       # older than 10 s at the pass end
    for p in ("p2", "p3", "p4", "p5"):
        assert not st.seen(p, 5.0)[0], p
    assert st.seen("p4", 5.0)[1] == pytest.approx(11.9)
    rec = pn.inspection_record("p1", "Z1", st.seen("p1", 5.0), "colorblob", run_id="r1")
    assert rec["schema"] == "nandi.inspect/1" and rec["seen"] and rec["data_provenance"] == "simulated"
    text = json.dumps(rec)
    import re
    assert not re.search(r'"(face|name|phone|person_[a-z_]*)"', text)
    assert "image" not in text and len(text) < 16 * 1024
    st.reset()
    assert not st.obs


# ------------------------------------------------------------------------------------------- scheduler
def _sched():
    cams = ["wrist_cam", "head_cam", "crown_cam", "ring_0", "ring_1", "ring_2", "ring_3", "belly_cam", "rear_cam"]
    return rs.default_scheduler({"budget_frac": 1.0, "burst_s": 10.0}, cams)


def test_scheduler_priority_order_and_register_api():
    s = _sched()
    assert s.order() == ["wrist_cam", "head_cam", "ring_0", "ring_1", "ring_2", "ring_3", "crown_cam",
                         "belly_cam", "rear_cam"]
    assert rs.ROLE_ORDER == ("wrist", "head", "ring", "crown", "belly", "rear", "pano")
    with pytest.raises(ValueError):
        s.register("head_cam", 1, ("drive",))
    with pytest.raises(ValueError):
        s.register("x_cam", 1, ("flying",))
    with pytest.raises(ValueError):
        s.register("y_cam", "nope", ("drive",))
    s.register("pano_virtual", "pano", ("drive",), rate_hz=0)
    assert s.order()[-1] == "pano_virtual"
    with pytest.raises(ValueError):
        s.set_mode("dance")


def _run(s, t0, t1, dt=0.1):
    got = []
    t = t0
    while t < t1 - 1e-9:
        for j in s.due(t):
            got.append((round(t, 2), j.camera, j.kind))
            s.charge(j.camera, 0.001)
        t += dt
    return got


def test_scheduler_drive_rates_and_ring_round_robin():
    s = _sched()
    got = _run(s, 0.0, 4.0)
    cams = [c for _, c, _ in got]
    rings = [c for c in cams if c.startswith("ring")]
    assert 7 <= len(rings) <= 9                               # 2 ring cameras/s
    assert rings[:4] == ["ring_0", "ring_1", "ring_2", "ring_3"]
    assert 7 <= cams.count("head_cam") <= 9 and "wrist_cam" not in cams       # wrist only in grip
    assert 1 <= cams.count("crown_cam") <= 3


def test_scheduler_grip_pauses_ring_and_tree360_pauses_others():
    s = _sched()
    assert s.mode_for_task("CUT") == "grip" and s.mode_for_task("TOOL_CHANGE") == "grip"
    assert s.mode_for_task("WALK") == "drive" and s.mode_for_task(None, tree360=True) == "tree360"
    assert s.mode_for_task("GRIP", paused=True) == "paused"
    s.set_mode("grip", 0.0, "CUT")
    cams = [c for _, c, _ in _run(s, 0.0, 3.0)]
    assert not any(c.startswith("ring") for c in cams) and "wrist_cam" in cams and "crown_cam" not in cams
    s.set_mode("tree360", 3.0, "tree L1-1", cameras=["crown_cam", "ring_1"])
    assert _run(s, 3.0, 5.0) == []                            # no periodic renders in tree360
    assert s.request("crown_cam", {"band": "mid"}) and s.request("ring_1", {"band": "low"})
    assert not s.request("ring_0") and not s.request("head_cam")          # paused in tree360
    got = _run(s, 5.0, 5.5)
    assert [(c, k) for _, c, k in got] == [("crown_cam", "capture"), ("ring_1", "capture")]
    s.set_mode("paused", 6.0)
    assert not s.request("crown_cam") and _run(s, 6.0, 8.0) == []
    s.set_mode("drive", 8.0)
    assert _run(s, 8.0, 9.0)


def test_scheduler_budget_skips_periodic_and_defers_captures():
    s = rs.default_scheduler({"budget_frac": 0.1, "burst_s": 0.05}, ["head_cam", "crown_cam", "ring_0"])
    t, renders = 0.0, 0
    while t < 10.0:
        for j in s.due(t):
            s.charge(j.camera, 0.2)                          # each render costs 0.2 s: far over 10 %
            renders += 1
        t += 0.1
    st = s.stats(10.0)
    assert st["render_fraction"] <= 0.1 + 0.2 / 10.0 + 1e-6 + 0.05
    assert sum(c["skips"] for c in st["cameras"].values()) > 0
    s.set_mode("tree360", 10.0, cameras=["crown_cam"])
    assert s.request("crown_cam", {"seq": 1})
    s.bucket.tokens = -1.0
    assert s.due(10.1) == [] and s.stats(10.1)["cameras"]["crown_cam"]["deferred"] == 1
    s.bucket.tokens = 0.2
    jobs = s.due(10.2)
    assert len(jobs) == 1 and jobs[0].tag == {"seq": 1} and jobs[0].passes == 3      # deferred, never dropped


# ------------------------------------------------------------------------------------------- perception core (GL)
@pytest.fixture(scope="module")
def core():
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    with world_env("orchard_lite"):
        c = pn.PerceptionCore()
    assert c.render_ok, c.render_error
    yield c
    c.close()


@pytest.mark.gl
def test_core_every_image_event_is_filtered_and_pano_published(core):
    c = core
    place_robot(c.model, c.data, 0.0, -2.0)
    place_person(c.model, c.data, (2.5, -2.0))
    assert c.set_qpos(c.data.qpos.copy())
    c.set_mode("drive", 0.0)
    frac0 = c.scheduler.bucket.frac
    c.scheduler.bucket.frac = 1.0                             # the pipeline under test, not the CPU budget
    events = []
    t = 0.0
    while t < 2.6:
        events += c.step(t)
        t += 0.1
    imgs = [e for e in events if e["kind"] == "image"]
    panos = [e for e in events if e["kind"] == "pano"]
    assert imgs and panos
    assert all(isinstance(e["frame"], pf.FilteredFrame) for e in imgs + panos)
    seen_cams = {e["camera"] for e in imgs}
    assert {"head_cam", "ring_0", "ring_1", "ring_2", "ring_3", "crown_cam"} <= seen_cams
    p = panos[-1]["frame"]
    assert p.shape == (256, 640) and len(pano_jpeg := p.jpeg(70)) < 512 * 1024
    ages = panos[-1]["ages"]
    assert set(ages) == set(camera_rig.RING_CAMS) and all(a is None or a <= 2.6 for a in ages.values())
    ring0 = [e for e in imgs if e["camera"] == "ring_0"][-1]
    assert ring0["info"]["truth_pct"] > 0                      # the person in front is masked
    st = c.state(2.6)
    assert st["scheduler"]["mode"] == "drive" and st["privacy"]["frames"] >= len(imgs)
    print(f"core drive 2.6 s: {len(imgs)} frames, {len(panos)} panos, pano JPEG {len(pano_jpeg)} B, "
          f"render fraction {st['scheduler']['render_fraction']}")
    place_person(c.model, c.data, (-30.0, -30.0))
    c.scheduler.bucket.frac = frac0


@pytest.mark.gl
def test_tree360_capture_rgb_depth_seg_and_render_fraction(core):
    """OS 3.7 / WP5: TREE 360 render mode, 3 captures (low ring, mid crown 1.70 m / -10, high crown 2.70 m / -20)
    at one OT station, each in a 0.5 s sim window at RTF 0.9 (0.556 s wall). Render fraction <= 0.35 measured
    here on the 2-CPU sim container (the CI-runner re-run is WP11a's)."""
    c = core
    m, d = c.model, c.data
    tc = d.xpos[m.body("tree_L1-1").id][:2].copy()
    place_robot(m, d, tc[0], tc[1] - 1.9, yaw=0.0)            # side mode: tree on the left
    low = camera_rig.ring_facing(m, d, tc)
    assert low == "ring_1"
    c.set_qpos(d.qpos.copy())
    window = 0.5 / 0.9
    t_start = time.monotonic()
    evs = c.tree360_cmd({"cmd": "start", "tree_id": "L1-1", "cameras": [camera_rig.CROWN_CAM, low]}, t_start)
    assert evs[0]["tree360"]["cameras"] == [camera_rig.CROWN_CAM, low]
    acks = []
    for band, cam, ext, tilt in (("low", low, 0.0, 0.0), ("mid", camera_rig.CROWN_CAM, 0.6, -10.0),
                                 ("high", camera_rig.CROWN_CAM, 1.6, -20.0)):
        w0 = time.monotonic()
        set_ext(m, ext, tilt_deg=tilt, pan_deg=90.0)          # gimbal aimed at the tree (+y)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        assert c.tree360_cmd({"cmd": "capture", "camera": cam, "band": band, "seq": 0, "tree_id": "L1-1"}, w0) == []
        got = None
        while got is None and time.monotonic() - w0 < 2.0:
            for e in c.step(time.monotonic()):
                if e["kind"] == "ack":
                    got = e
        assert got is not None, band
        acks.append(got)
        rest = window - (time.monotonic() - w0)
        if rest > 0:
            time.sleep(rest)
    # a camera outside the station set is refused, not rendered
    refused = c.tree360_cmd({"cmd": "capture", "camera": "ring_3", "band": "low", "seq": 1}, time.monotonic())
    assert refused and refused[0]["payload"]["ok"] is False
    frac = c.scheduler.fraction(time.monotonic())
    done = c.tree360_cmd({"cmd": "stop"}, time.monotonic())[0]["tree360_done"]
    for a in acks:
        p = a["payload"]
        assert p["ok"] and isinstance(a["frame"], pf.FilteredFrame) and a["frame"].shape == (240, 320)
        assert a["frame"].depth is not None and "self_pct" in p and p["person_covered_pct"] == 100.0
        assert p["ngeom"][1] < p["ngeom"][0]                  # culled
    wall = time.monotonic() - t_start
    gl = sum(a["payload"]["gl_ms"] for a in acks) / 1e3
    print(f"tree360 station: GL {gl * 1e3:.0f} ms for 3 captures "
          f"({[round(a['payload']['gl_ms']) for a in acks]} ms), wall {wall:.2f} s, render fraction {frac:.3f}")
    assert frac <= 0.35 and done["render_fraction"] <= 0.35
    assert c.scheduler.mode == "drive"
    set_ext(m, 0.0)


@pytest.mark.gl
def test_core_colorblob_and_dusk_rule_in_the_legacy_world():
    """MVP-SPEC 4.2: at dusk (noise sigma 8/255, 3 px blur, dark light) ColorBlob keeps recall >= 0.8 on the red
    crops in the head camera, or the node falls back to the geometry detector and raises an assist request."""
    if not gl_available():
        pytest.skip("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
    with world_env("legacy"):
        c = pn.PerceptionCore()
    try:
        assert c.render_ok
        m, d = c.model, c.data
        c.set_qpos(d.qpos.copy())
        frame, info = c.render_frame("head_cam", 0.0)
        day = c.colorblob_check(frame, "head_cam", 0.0)
        assert day["n_truth"] >= 3 and day["recall"] >= 0.8, day
        # dusk: darker light + the lighting preset's camera degradations
        from atharv_nandi.sim.world import lighting
        dusk = lighting.preset("dusk")
        for li in range(m.nlight):
            m.light_diffuse[li] = dusk["diffuse"]
            m.light_ambient[li] = dusk["ambient"]
        m.vis.headlight.diffuse[:] = dusk["headlight_diffuse"]
        m.vis.headlight.ambient[:] = dusk["headlight_ambient"]
        c.set_lighting("dusk")
        assert c.lighting["camera_noise_sigma"] == pytest.approx(8.0 / 255.0) and c.lighting["motion_blur_px"] == 3
        recs = []
        for k in range(6):
            fr, _ = c.render_frame("head_cam", 1.0 + k)
            recs.append(c.colorblob_check(fr, "head_cam", 1.0 + k))
        recall = sum(r["recall"] * r["n_truth"] for r in recs) / max(1, sum(r["n_truth"] for r in recs))
        assist = c.selector.take_assist()
        if recall >= 0.8:
            assert c.selector.active == "colorblob" and assist is None
        else:
            assert c.selector.active == "geometry" and assist and assist["reason"] == "colorblob_recall_low"
        msg, _ = c.detections(2.0)
        assert msg["detector"] if isinstance(msg, dict) and "detector" in msg else True
        print(f"ColorBlob recall day {day['recall']:.2f} ({day['n_truth']} crops), dusk {recall:.2f} -> "
              f"active {c.selector.active}, assist {assist}")
    finally:
        c.close()


def test_scheduler_scarce_budget_favours_the_head_camera():
    s = rs.default_scheduler({"budget_frac": 0.35, "burst_s": 0.3}, ["head_cam", "crown_cam", "ring_0", "ring_1",
                                                                     "ring_2", "ring_3"])
    t = 0.0
    while t < 30.0:
        for j in s.due(t):
            s.charge(j.camera, 0.15)                         # 150 ms per pass: about 2.3 passes/s fit in 35 %
        t += 0.1
    c = s.stats(30.0)["cameras"]
    rings = sum(c[f"ring_{k}"]["renders"] for k in range(4))
    assert c["head_cam"]["renders"] >= 50                     # head keeps (close to) its 2 Hz
    assert c["head_cam"]["renders"] > rings and c["head_cam"]["skips"] < c["crown_cam"]["skips"]
    assert s.fraction(30.0) <= 0.35 + 0.3 / 30.0 + 0.15 / 30.0


# ------------------------------------------------------------------------------------------- review round 1
@pytest.fixture(scope="module")
def core_nogl():
    with world_env("orchard_lite"):
        c = pn.PerceptionCore(render=False)
    yield c
    c.close()


def _ring0_view(c):
    cid = camera_rig.camera_id(c.model, "ring_0")
    return camera_rig.camera_pose(c.data, cid), camera_rig.intrinsics(c.model, "ring_0", 160, 120)


def test_physics_privacy_fails_closed_without_fresh_clusters(core_nogl):
    """WP5 review: physics mode used whatever clusters came last (or none) with no freshness check."""
    c = core_nogl
    pose, intr = _ring0_view(c)
    rgb = np.zeros((120, 160, 3), np.uint8)
    mode0 = c.privacy.mode
    c.clusters, c.clusters_t, c.clusters_rx, c.state_t = [], None, None, None
    try:
        c.privacy.mode = "physics"
        r = c._filtered("ring_0", 10.0, rgb, None, pose, intr)
        assert r.frame is None and r.reason == "clusters_missing"
        c.set_clusters({"t": 5.0, "clusters": []}, now=10.0)
        assert c._filtered("ring_0", 10.2, rgb, None, pose, intr).frame is not None        # fresh by arrival
        assert c._filtered("ring_0", 11.0, rgb, None, pose, intr).reason == "clusters_stale"
        c.state_t = 5.2                                         # sim time known: compared against the state
        assert c._filtered("ring_0", 99.0, rgb, None, pose, intr).frame is not None
        c.state_t = 6.0
        assert c._filtered("ring_0", 99.0, rgb, None, pose, intr).reason == "clusters_stale"
        c.set_clusters({"clusters": [{"lo": [0, 0, 0], "hi": [1, 1, 2]}]}, now=99.0)      # no t: ignored
        assert c.clusters_t == 5.0 and c.clusters == []
        c.set_clusters({"t": 6.0, "clusters": [{"lo": [0, 0]}, {"lo": [3, -0.2, 0], "hi": [3.4, 0.2, 2]}]}, now=99.0)
        assert len(c.clusters) == 1
        # assisted mode keeps the truth mask; stale clusters are reported, not used
        c.privacy.mode = "assisted"
        c.state_t = 9.0
        r = c._filtered("ring_0", 99.0, rgb, None, pose, intr)
        assert r.frame is not None and r.stats["clusters"] == "clusters_stale" and r.stats["detector_pct"] == 0
        assert c.privacy.counts["dropped"] >= 3 and c.privacy.counts["frames"] >= c.privacy.counts["dropped"]
    finally:
        c.privacy.mode = mode0
        c.clusters, c.clusters_t, c.clusters_rx, c.state_t = [], None, None, None


def test_tree360_start_stop_respect_pause_and_grip(core_nogl):
    """WP5 review: tree360 start / stop overrode the scheduler mode (stop during an e-stop resumed rendering)."""
    c = core_nogl
    c.set_mode("paused", 0.0, "estop")
    ev = c.tree360_cmd({"cmd": "start", "tree_id": "L1-1"}, 1.0)
    assert ev[0]["tree360_refused"]["reason"] == "mode_paused" and c.tree360 is None
    assert c.scheduler.mode == "paused"
    c.set_mode("grip", 1.5)
    assert c.tree360_cmd({"cmd": "start", "tree_id": "L1-1"}, 1.6)[0]["tree360_refused"]["reason"] == "mode_grip"
    c.set_mode("drive", 2.0)
    assert "tree360" in c.tree360_cmd({"cmd": "start", "tree_id": "L1-1"}, 2.0)[0]
    assert c.scheduler.mode == "tree360"
    c.set_mode("paused", 3.0, "estop")                          # e-stop during the station
    c.tree360_cmd({"cmd": "stop"}, 4.0)
    assert c.scheduler.mode == "paused" and c.tree360 is None  # stop does not resume rendering
    c.set_mode("drive", 5.0)
    c.tree360_cmd({"cmd": "start", "tree_id": "L1-2"}, 5.0)
    c.tree360_cmd({"cmd": "stop"}, 6.0)
    assert c.scheduler.mode == "drive"


@pytest.mark.gl
def test_core_runs_the_dusk_rule_and_the_per_frame_arm_hull(core):
    """WP5 review: colorblob_check was never called by the node (no dusk fallback, no /nandi/assist), and the
    per-frame arm hull self-mask was not used by the pano or the seen tracker."""
    c = core
    place_robot(c.model, c.data, 0.0, -2.0)
    assert c.set_qpos(c.data.qpos.copy())
    c.set_mode("drive", 100.0)
    frac0 = c.scheduler.bucket.frac
    c.scheduler.bucket.frac = 1.0
    c._next_colorblob = 0.0
    seen_calls = []
    orig = c.self_masks.get
    c.self_masks.get = lambda cam, arm_mask=None: (seen_calls.append((cam, arm_mask is not None)), orig(cam, arm_mask))[1]
    try:
        events = []
        t = 100.0
        while t < 102.6:
            events += c.step(t)
            t += 0.1
    finally:
        c.self_masks.get = orig
        c.scheduler.bucket.frac = frac0
    checks = [e["check"] for e in events if e["kind"] == "colorblob"]
    assert 2 <= len(checks) <= 4                               # 1 Hz on head frames
    assert all(set(k) >= {"recall", "n_truth", "active"} for k in checks)
    rings = [e for e in events if e["kind"] == "image" and e["camera"] in camera_rig.RING_CAMS]
    assert rings and all(e["frame"].meta["arm_mask"].shape == (120, 160) for e in rings)
    assert any(cam in camera_rig.RING_CAMS and has for cam, has in seen_calls)
    assert any(e["kind"] == "pano" for e in events)
