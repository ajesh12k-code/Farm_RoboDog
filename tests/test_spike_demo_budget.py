"""WP11a spike logic (tools/spike_demo_budget.py), ROS-free and GL-free.

The GL timings themselves are measurements (budget-spike), not tests. These tests pin the
stub's model budget (ORCHARD-MVP-SPEC 2.6), the orchard_demo layout, the mjvScene post-filter cull
(3.7: > 4 m dropped, ngeom compacted, model untouched) and the go / no-go arithmetic.
"""
import math
import os
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from tools import spike_demo_budget as sdb  # noqa: E402


@pytest.fixture(scope="module")
def stub(monkeypatch_module):
    m, info = sdb.build_stub()
    return m, info


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    mp.setenv("NANDI_TERRAIN", "flat")
    mp.delenv("NANDI_CROPS", raising=False)
    mp.delenv("NANDI_MAST_H", raising=False)
    mp.delenv("NANDI_PAYLOAD_KG", raising=False)
    yield mp
    mp.undo()


def _scene_at_station(m, k=0, camera="crown_cam"):
    d = mujoco.MjData(m)
    sdb.reset(m, d)
    key_q = d.qpos.copy()
    sdb.set_station_pose(m, d, sdb.ot1_stations()[k], key_q)
    scn = mujoco.MjvScene(m, maxgeom=20000)
    opt = mujoco.MjvOption()
    for gi, on in enumerate(sdb.RENDER_GROUPS):
        opt.geomgroup[gi] = on
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    mujoco.mjv_updateScene(m, d, opt, mujoco.MjvPerturb(), cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    return d, scn, cam.fixedcamid


def test_stub_matches_model_budget(stub):
    m, info = stub
    c = sdb.stub_counts(m, info)
    assert c["nv"] == sdb.NV_TARGET == 151
    assert c["eq_rows_at_rest"] <= sdb.EQ_ROWS_MAX == 63
    assert c["eq_rows_at_rest"] == 63          # 3 rack welds x 6 + 5 connect x 3 + 5 weld x 6
    assert c["nmocap"] == 10
    assert c["n_trees"] == 17
    assert c["render_geoms_per_tree_max"] <= sdb.RENDER_GEOMS_PER_TREE_MAX
    assert m.opt.timestep == pytest.approx(0.002)


def test_stub_physics_is_stable(stub):
    m, _ = stub
    d = mujoco.MjData(m)
    sdb.reset(m, d)
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))
    assert d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0
    assert d.qpos[2] > 0.25                    # robot still standing
    # pool fruit still hang (connect / weld hold): within 5 cm of their build position
    for k in range(sdb.N_POOL):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"pool_{k}")
        assert np.linalg.norm(d.xpos[bid] - m.body_pos[bid]) < 0.05


def test_layout_follows_spec_and_decisions():
    trees = sdb.orchard_demo_layout()
    assert len(trees) == 17
    by_row = {}
    for t in trees[1:]:
        by_row.setdefault(t["id"].split("-")[0], []).append(t)
    assert {r: len(v) for r, v in by_row.items()} == {"R1": 4, "R2": 4, "R3": 4, "R4": 4}
    xs = [t["x"] for t in by_row["R1"]]
    assert np.allclose(np.diff(xs), 2.5)       # NHB 5 x 2.5 m mango (OQ-02 B)
    assert all(1.8 <= t["H"] <= 2.8 for t in by_row["R1"])
    ot1 = trees[0]
    r_clear = ot1["canopy_r"] + sdb.STANDOFF_M + 0.6
    for t in trees[1:]:
        gap = math.hypot(t["x"] - ot1["x"], t["y"] - ot1["y"]) - t["canopy_r"]
        assert gap >= r_clear, t["id"]


def test_ot1_stations_keep_tree_on_left():
    for x, y, hd in sdb.ot1_stations():
        r = math.hypot(x - sdb.OT1["x"], y - sdb.OT1["y"])
        assert r == pytest.approx(sdb.OT1["canopy_r"] + sdb.STANDOFF_M)
        left = np.array([math.cos(hd + math.pi / 2), math.sin(hd + math.pi / 2)])
        to_tree = np.array([sdb.OT1["x"] - x, sdb.OT1["y"] - y]) / r
        assert float(left @ to_tree) == pytest.approx(1.0)


def test_cam_quat_looks_along_yaw_and_pitch():
    q = sdb.cam_quat(90, -20)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q))
    fwd = -R.reshape(3, 3)[:, 2]
    assert fwd == pytest.approx([0.0, math.cos(math.radians(20)), -math.sin(math.radians(20))], abs=1e-9)


def test_cull_drops_far_geoms_compacts_and_leaves_model_untouched(stub):
    m, _ = stub
    fp = sdb.model_fingerprint(m)
    d, scn, cid = _scene_at_station(m)
    n0 = scn.ngeom
    before = [(scn.geoms[i].objtype, scn.geoms[i].objid) for i in range(n0)]
    culler = sdb.SceneCuller(m, scn)
    assert culler.fast                          # vectorised path verified on this MuJoCo build
    nb, na = culler.cull(d.cam_xpos[cid])
    assert nb == n0 and na == scn.ngeom
    assert na < n0 and na <= sdb.SCENE_GEOMS_AFTER_CULL_MAX
    kept = [(scn.geoms[i].objtype, scn.geoms[i].objid) for i in range(na)]
    # order preserved (a subsequence) and segids renumbered to scene indices
    it = iter(before)
    assert all(any(k == b for b in it) for k in kept)
    assert [scn.geoms[i].segid for i in range(na)] == list(range(na))
    cam = d.cam_xpos[cid]
    geom_t = int(mujoco.mjtObj.mjOBJ_GEOM)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    kept_ids = {oid for ot, oid in kept if ot == geom_t}
    assert floor in kept_ids                    # planes are never culled
    for ot, oid in before:
        if ot != geom_t or m.geom_type[oid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if m.body_mocapid[m.geom_bodyid[oid]] >= 0:
            assert oid in kept_ids              # actors are never culled
            continue
        excess = np.linalg.norm(d.geom_xpos[oid] - cam) - m.geom_rbound[oid] - sdb.CULL_DIST_M
        if excess > 1e-3:
            assert oid not in kept_ids
        elif excess < -1e-3:
            assert oid in kept_ids
    assert sdb.model_fingerprint(m) == fp


def test_cull_fast_path_matches_field_copy_fallback(stub):
    m, _ = stub
    d, scn_a, cid = _scene_at_station(m, 3)
    _, scn_b, _ = _scene_at_station(m, 3)
    fast = sdb.SceneCuller(m, scn_a)
    slow = sdb.SceneCuller(m, scn_b)
    slow.fast = False
    fast.cull(d.cam_xpos[cid])
    slow.cull(d.cam_xpos[cid])
    assert scn_a.ngeom == scn_b.ngeom
    for i in range(scn_a.ngeom):
        a, b = scn_a.geoms[i], scn_b.geoms[i]
        assert (a.objtype, a.objid, a.type, a.segid) == (b.objtype, b.objid, b.type, b.segid)
        assert np.array_equal(a.pos, b.pos) and np.array_equal(a.rgba, b.rgba)


def test_lidar_has_1620_rays_and_masks_differ_only_by_foliage(stub):
    m, _ = stub
    lid = sdb.Lidar(m)
    assert lid.n == 1620
    d = mujoco.MjData(m)
    sdb.reset(m, d)
    sdb.set_station_pose(m, d, (4.0, 4.5, 0.0), d.qpos.copy())
    s = lid.sweep(d, sdb.SAFETY_MASK, count_groups=True)
    c = lid.sweep(d, sdb.COVERAGE_MASK, count_groups=True)
    assert s["foliage_hits"] == 0               # safety rays never see foliage
    assert c["hits"] >= s["hits"] - c["foliage_hits"]
    assert sdb.SAFETY_MASK[sdb.GRP_FOLIAGE] == 0 and sdb.COVERAGE_MASK[sdb.GRP_FOLIAGE] == 1


def test_coverage_rays_count_foliage(stub):
    m, _ = stub
    d = mujoco.MjData(m)
    sdb.reset(m, d)
    key_q = d.qpos.copy()
    crown = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "crown_cam")
    pts = sdb.coverage_points(sdb.OT1)
    assert all(len(v) == sdb.COVERAGE_POINTS_PER_BAND for v in pts.values())
    no_fol, fol = 0, 0
    for st in sdb.ot1_stations():
        sdb.set_station_pose(m, d, st, key_q)
        o = d.cam_xpos[crown].copy()
        no_fol += int(np.sum(sdb.coverage_visible(m, d, o, pts["mid"], sdb.SAFETY_MASK)))
        fol += int(np.sum(sdb.coverage_visible(m, d, o, pts["mid"], sdb.COVERAGE_MASK)))
    assert fol < no_fol                         # foliage occludes some canopy points


def _render(cap_ms, ngeom_after=150):
    row = {"capture_ms": {"p90": cap_ms}, "ngeom_after": {"max": ngeom_after}}
    return {"cull_4m": {"ring": row, "crown_mid": row, "crown_high": row,
                        "station_9pass_equiv_ms": {"p90": 3 * cap_ms * 1.5}},
            "scene_geoms_after_cull_max": ngeom_after}


def _ack(within=24, n=24, rtf=0.9):
    return {"acks_within_window": within, "n_acks": n, "ack_sim_s": {"p90": 0.3}, "rtf_during_captures": rtf}


def test_verdict_go_and_each_no_go():
    assert sdb.verdict(_render(220), _ack())["go"] is True
    assert sdb.verdict(_render(220), _ack(within=23))["go"] is False
    assert sdb.verdict(_render(220), _ack(rtf=0.8))["go"] is False
    assert sdb.verdict(_render(220, ngeom_after=900), _ack())["go"] is False
    # station mean fraction (report-only, loose): 9-pass station cost over 165/8/0.9 s
    assert sdb.verdict(_render(2000), _ack())["report_only"]["A_station_mean_frac"]["ok"] is False
    assert "A_station_mean_frac" not in sdb.verdict(_render(220), _ack())["criteria"]
    assert sdb.verdict(_render(220))["complete"] is False


def test_token_bucket_replay_reports_needed_burst():
    tb = sdb.token_bucket_skips([400, 400, 400], 0.5 / 0.9, frac=0.35, burst_s=0.3)
    assert tb["skips"] >= 1
    tb2 = sdb.token_bucket_skips([400, 400, 400], 0.5 / 0.9, frac=0.35, burst_s=tb["burst_s_for_zero_skips"] + 0.01)
    assert tb2["skips"] == 0


def test_culled_scene_stays_under_700_geoms_in_the_alleys(stub):
    m, _ = stub
    c = sdb.culled_scene_counts(m, sdb.ALLEY_POSES[:4] + sdb.ot1_stations()[:2])
    assert c["after"]["max"] <= sdb.SCENE_GEOMS_AFTER_CULL_MAX
    assert c["after"]["max"] < c["before"]["max"]


# ------------------------------------------------------------------ review fixes (WP11a fixer)
def _scene_ids(scn):
    geom_t = int(mujoco.mjtObj.mjOBJ_GEOM)
    return {scn.geoms[i].objid for i in range(scn.ngeom) if scn.geoms[i].objtype == geom_t}


def test_cull_never_drops_actors(stub):
    """People and animals are privacy truth (MVP-SPEC 5.4) and safety-relevant: never culled, any distance."""
    m, _ = stub
    actor_geoms = {g for g in range(m.ngeom) if m.body_mocapid[m.geom_bodyid[g]] >= 0}
    assert len(actor_geoms) == sdb.NMOCAP_TARGET
    far_seen = 0
    for k in range(len(sdb.ot1_stations())):
        d, scn, cid = _scene_at_station(m, k)
        before = _scene_ids(scn) & actor_geoms
        cam = d.cam_xpos[cid]
        far_seen += sum(np.linalg.norm(d.geom_xpos[g] - cam) - m.geom_rbound[g] > sdb.CULL_DIST_M for g in before)
        sdb.SceneCuller(m, scn).cull(cam)
        assert before <= _scene_ids(scn), k
    assert far_seen > 0                         # the test does exercise actors beyond 4 m


def test_station_pose_carries_rack_tools_with_the_base(stub):
    m, _ = stub
    d = mujoco.MjData(m)
    sdb.reset(m, d)
    key_q = d.qpos.copy()
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    tools = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"tool_{k}") for k in range(sdb.N_TOOLS)]
    pools = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"pool_{k}") for k in range(sdb.N_POOL)]

    def rel(dd):
        R = dd.xmat[trunk].reshape(3, 3)
        return [R.T @ (dd.xpos[b] - dd.xpos[trunk]) for b in tools]
    rel0 = rel(d)
    pool0 = d.xpos[pools].copy()
    for st in sdb.ot1_stations()[:4]:
        sdb.set_station_pose(m, d, st, key_q)
        for a, b in zip(rel0, rel(d)):
            assert np.allclose(a, b, atol=1e-9)
        assert np.allclose(d.xpos[pools], pool0)          # pool fruit hang on trees: not moved
    # and at a station the tools are within reach of the trunk (they were ~5 m away before the fix)
    for b in tools:
        assert np.linalg.norm(d.xpos[b] - d.xpos[trunk]) < 0.5


def test_token_bucket_needed_burst_is_exact_with_refill_cap():
    # cheap first capture: the refill cap binds; the old estimate (0.262) still skipped one
    w = 0.5 / 0.9
    tb = sdb.token_bucket_skips([50, 600, 600], w, 0.35, 0.3)
    need = tb["burst_s_for_zero_skips"]
    assert sdb.token_bucket_skips([50, 600, 600], w, 0.35, need)["skips"] == 0
    assert sdb.token_bucket_skips([50, 600, 600], w, 0.35, need - 0.002)["skips"] >= 1
    for costs in ([400, 400, 400], [100, 900, 50], [600, 50, 600], [10, 10, 10]):
        tb = sdb.token_bucket_skips(costs, w, 0.35, 0.3)
        n = tb["burst_s_for_zero_skips"]
        assert sdb.token_bucket_skips(costs, w, 0.35, n)["skips"] == 0
        if n > 0.002:
            assert sdb.token_bucket_skips(costs, w, 0.35, n - 0.002)["skips"] >= 1


def test_verdict_gates_on_render_budget_replay_not_station_mean():
    # 400 ms per 2-pass capture -> 600 ms per 3-pass capture: station mean fraction tiny, bucket skips
    v = sdb.verdict(_render(400), _ack())
    assert v["report_only"]["A_station_mean_frac"]["ok"] is True
    assert v["criteria"]["E_render_budget_no_skips"]["ok"] is False
    assert v["go"] is False
    need = v["criteria"]["E_render_budget_no_skips"]["burst_s_for_zero_skips"]
    assert sdb.verdict(_render(400), _ack(), burst_s=need)["go"] is True


def test_verdict_rtf_gate_states_its_tolerance():
    c = sdb.verdict(_render(100), _ack(rtf=0.885))["criteria"]["C_rtf_during_captures"]
    assert c["ok"] is True and c["limit"] == pytest.approx(sdb.RTF_TARGET - sdb.RTF_TOLERANCE)
    assert c["tolerance"] == sdb.RTF_TOLERANCE
    assert sdb.verdict(_render(100), _ack(rtf=0.87))["criteria"]["C_rtf_during_captures"]["ok"] is False


def test_git_arg_is_validated_and_run_identity_carries_tool_blob(monkeypatch):
    assert sdb.check_git_arg("6e435bb0a1b2") == "6e435bb0a1b2"
    assert sdb.check_git_arg("6e435bb0a1b2-dirty") == "6e435bb0a1b2-dirty"
    for bad in ("", "unknown", "HEAD", "nogit", "6e435bb0a1b2+dirty"):
        with pytest.raises(SystemExit):
            sdb.check_git_arg(bad)
    monkeypatch.setenv("NANDI_GIT_SHA", "6e435bb0a1b2-dirty")
    ident = sdb.identity("stub")
    assert ident["run_id"].endswith("-6e435bb0a1b2-dirty")
    import hashlib
    data = open(sdb.__file__, "rb").read()
    assert ident["tool_blob"] == hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_rack_welds_hold_the_tools_to_the_trunk(stub):
    # MjSpec's default weld data anchors at (0, 1, 0): the tools then swung 1 m out on a soft pendulum.
    m, _ = stub
    for k in range(sdb.N_TOOLS):
        eid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"rack_weld_{k}")
        assert np.allclose(m.eq_data[eid][0:3], 0.0)
    d = mujoco.MjData(m)
    sdb.reset(m, d)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    tools = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"tool_{k}") for k in range(sdb.N_TOOLS)]

    def rel(dd):
        R = dd.xmat[trunk].reshape(3, 3)
        return np.array([R.T @ (dd.xpos[b] - dd.xpos[trunk]) for b in tools])

    def relq(dd):   # |dot| of tool and trunk quats: 1 when the relative rotation is unchanged
        return np.array([abs(float(np.dot(dd.xquat[b], dd.xquat[trunk]))) for b in tools])
    r0, q0 = rel(d), relq(d)
    assert np.all(np.linalg.norm(r0, axis=1) < 0.5)
    for _ in range(int(round(3.0 / m.opt.timestep))):     # 3 s sim
        mujoco.mj_step(m, d)
        assert np.max(np.abs(rel(d) - r0)) < 0.01
    assert np.allclose(relq(d), q0, atol=1e-3)
    for b in tools:
        assert d.xpos[b][2] > 0.0                         # above the floor, not hanging through it


def test_import_and_build_leave_the_environment_alone():
    # importing the tool (pytest collection) or building the stub must not flip NANDI_TERRAIN for
    # other tests, which expect farm_scene's default rough world
    import subprocess
    code = ("import os, sys; from tools import spike_demo_budget as s; "
            "s.build_stub(); print(os.environ.get('NANDI_TERRAIN'), os.environ.get('LP_NUM_THREADS'))")
    env = {k: v for k, v in os.environ.items() if k not in ("NANDI_TERRAIN", "LP_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[-2:] == ["None", "None"]
    assert sdb.ENV_DEFAULTS == {"LP_NUM_THREADS": "1", "NANDI_TERRAIN": "flat"}


def test_verdict_section_json_is_not_double_nested(tmp_path, monkeypatch):
    import json
    for k in sdb.ENV_DEFAULTS:            # main() applies env defaults: keep them inside this test
        monkeypatch.setenv(k, os.environ.get(k, sdb.ENV_DEFAULTS[k]))
    rp, ap_, op = tmp_path / "render.json", tmp_path / "ack.json", tmp_path / "v.json"
    rp.write_text(json.dumps({"render": _render(220)}))
    ap_.write_text(json.dumps({"ack": _ack()}))
    assert sdb.main(["verdict", "--render", str(rp), "--ack", str(ap_), "--out", str(op)]) == 0
    v = json.loads(op.read_text())["verdict"]
    assert "go" in v and "verdict" not in v
