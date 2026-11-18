"""WP25 TREE 360 recorder (ORCHARD-MVP-SPEC 1.2 step 2, 2.3, 5.3, 5.6; MISSION-PRESETS 0.6, 0.8, 2).

Unit tier: ``Tree360Run`` on a pure-Python stand-in (``FakeIO``: unicycle hexapod with a lagging gait, a 0.2 m/s
mast, analytic cameras, the real WP5 PrivacyFilter / FilteredFrames): sequence, snake vs stop-and-go, pause,
protective stop, re-entry and gap, crown refusal, two-sided transit, files, manifest, records. One unit test drives
the ``tree360`` hook on the composed orchard_demo world (command errors, start, abort, published records).
``slow``: the full OT-1 visit on orchard_demo with the WP24 mast, WP5 trunk fit, WP6 fields and the scripted worker
(GL captures when a display is available), and the two-sided hedgerow visit of R1-2.
"""
import json
import math
import os

import numpy as np
import pytest

from atharv_nandi.core import coverage as cv
from atharv_nandi.core import manifest as mf
from atharv_nandi.core import orbit as ob
from atharv_nandi.core import tree360 as t3
from atharv_nandi.core.detector_model import Pose, intrinsics_from_fovy
from atharv_nandi.core.privacy_filter import PrivacyFilter

OT1_ENTRY = {"tree_id": "OT-1", "physical_tag": "01", "block_id": "B0", "row_id": "OT", "crop": "mango",
             "cultivar": "Dashehari", "form": "grafted", "geo": {"frame": "nandi_local_v1", "x": 6.0, "y": -3.5},
             "height_m": 2.2, "canopy_r_m": 1.0, "trunk_r_m": 0.06, "trunk_h_m": 0.77, "basin_r_m": 0.8,
             "orbit_clear": True, "alley_sides": ["all"]}
FENCE = [[-6.5, -6.5], [16.8, -6.5], [16.8, 17.5], [-6.5, 17.5]]


def _look(eye, fwd):
    z = np.asarray(fwd, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Pose.from_frame_in_world(np.asarray(eye, float), np.stack([x, y, z], axis=1))


class FakeIO:
    """Kinematic stand-in for ``SimIO`` (no MuJoCo)."""

    def __init__(self, tree_xy=(6.0, -3.5), v_bias=0.9, w_bias=0.9, lag=0.25):
        self.t = 0.0
        self.x, self.y, self.h = 0.0, -6.0, 0.0
        self.v = self.w = 0.0
        self.cmd = (0.0, 0.0)
        self.tree_xy = tree_xy
        self.v_bias, self.w_bias, self.lag = v_bias, w_bias, lag
        self.ext, self.ext_target = 0.0, 0.0
        self.requests = []
        self.refuse = {}                 # ext -> reason
        self.pan = self.tilt = 0.0
        self.inh = {"hold": None, "pause": False, "stop": False, "stop_reason": None, "walk_block": False}
        self.gait = object()
        self.privacy = PrivacyFilter("assisted")
        self.captures = []
        self.ext_on_arc = []
        self.phase_of = None

    # simulation step (the test loop)
    def step(self, dt=0.02):
        self.t += dt
        vx, yaw = self.cmd
        v_t, w_t = vx * ob.V_PER_CMD * self.v_bias, yaw * ob.W_PER_CMD * self.w_bias
        self.v += (v_t - self.v) * dt / self.lag
        self.w += (w_t - self.w) * dt / self.lag
        self.x += self.v * math.cos(self.h) * dt
        self.y += self.v * math.sin(self.h) * dt
        self.h = ob.wrap(self.h + self.w * dt)
        d = self.ext_target - self.ext
        self.ext += max(-0.2 * dt, min(0.2 * dt, d))

    # IO
    def now(self):
        return self.t

    def pose(self):
        return (self.x, self.y), self.h

    def velocity(self):
        return abs(self.v), self.w

    def yaw_rate_deg(self):
        return abs(math.degrees(self.w))

    def tilt_deg(self):
        return 0.5

    def truth_centre(self, tree_id):
        return self.tree_xy

    def rtk(self):
        return "FIXED"

    def inhibit(self):
        return dict(self.inh)

    def drive(self, vx, yaw):
        self.cmd = (float(vx), float(yaw))

    def stand(self):
        self.cmd = (0.0, 0.0)

    def teleport(self, trunk_xy, heading):
        self.x, self.y, self.h = float(trunk_xy[0]), float(trunk_xy[1]), float(heading)
        self.v = self.w = 0.0

    def mast_request(self, ext, reason):
        self.requests.append((round(self.t, 2), round(ext, 3), reason))
        for e, why in self.refuse.items():
            if abs(e - ext) < 1e-6 and ext > self.ext:
                return {"allowed": False, "reason": why, "detail": "test"}
        self.ext_target = float(ext)
        return {"allowed": True, "reason": "", "detail": None}

    def mast_release(self):
        self.requests.append((round(self.t, 2), None, "release"))

    def mast_state(self):
        return {"ext_m": round(self.ext, 4), "moving": abs(self.ext - self.ext_target) > 1e-4,
                "lens_z": 1.10 + self.ext, "gimbal_deg": [round(math.degrees(self.pan), 2),
                                                          round(math.degrees(self.tilt), 2)],
                "arbiter_source": "preset"}

    def aim(self, pan, tilt):
        self.pan, self.tilt = pan, tilt

    def request_fit(self, tree):
        pass

    def cancel_fit(self):
        pass

    def trunk_fit(self, tree_id):
        return None

    def ring_facing(self, target_xy):
        return "ring_1"

    def crown_camera(self):
        return "crown_cam"

    def backend_name(self):
        return "geometry"

    def coverage_ray_fn(self):
        return None

    def on_start(self, run):
        pass

    def close(self):
        pass

    def capture(self, camera, band, seq, tree_id):
        ref = ob.ref_point((self.x, self.y), self.h)
        if camera.startswith("ring"):
            yaw, tilt, z, fovy = self.h + math.pi / 2, math.radians(-25), 0.98, 83.6
        else:
            yaw, tilt, z, fovy = self.h + self.pan, self.tilt, 1.10 + self.ext, 60.0
        fwd = (math.cos(tilt) * math.cos(yaw), math.cos(tilt) * math.sin(yaw), math.sin(tilt))
        eye = (ref[0], ref[1], z)
        pose = _look(eye, fwd)
        intr = intrinsics_from_fovy(fovy, 320, 240)
        res = self.privacy.apply(camera, self.t, np.zeros((240, 320, 3), np.uint8), pose, intr)
        self.captures.append((round(self.t, 2), camera, band, round(self.ext, 3)))
        return t3.CaptureResult(camera, pose, intr, fovy, res.frame, np.asarray(res.frame.masked), False, "ok",
                                dict(res.stats), eye, fwd, "geometry")


def make_run(tmp_path, params=None, entry=None, registry=None, io=None, fence=FENCE):
    io = io or FakeIO()
    p = {"entry": "teleport", "measure": "assisted", "capture": "geometry", "out_root": str(tmp_path),
         "run_id": "unit", "device_id": "nandi-sim-01", "mp4": True}
    p.update(params or {})
    cfg = t3.Tree360Config.from_params(p)
    entry = entry or OT1_ENTRY
    run = t3.Tree360Run(entry, registry or [entry], cfg, io, geofence=fence)
    return run, io


def drive(run, io, t_max=400.0, hook=None):
    while not run.finished and io.t < t_max:
        io.step()
        if hook:
            hook(run, io)
        run.tick()
        if run.phase == "arc":
            io.ext_on_arc.append(io.ext)
    return run.result


# ------------------------------------------------------------------------------------------- config
def test_config_validation():
    cfg = t3.Tree360Config.from_params({"stations": 8, "bands": ["low", "mid", "high"]})
    assert cfg.ext_of("mid") == pytest.approx(0.60) and cfg.ext_of("high") == pytest.approx(1.60)
    for bad in ({"stations": 3}, {"standoff_m": 0.5}, {"v_mps": 0.2}, {"arc_mode": "spiral"},
                {"crown_walk": "hop"}, {"bands": ["top"]}, {"high_lens_m": 3.0}, {"nope": 1},
                {"reason": "whim"}):
        with pytest.raises(ValueError):
            t3.Tree360Config.from_params(bad)


# ------------------------------------------------------------------------------------------- full visit
def test_full_snake_visit(tmp_path):
    run, io = make_run(tmp_path)
    res = drive(run, io)
    k = res["kpi"]
    assert res["status"] == "done" and res["mode"] == "full" and res["crown_walk"] == "snake"
    assert k["frames"] == 24 and all(v == {"low", "mid", "high"} for v in run.captured.values())
    assert len(run.captured) == 8
    # snake: raise to mid, one crown move per station (mid <-> high), stow; the crown stays up on the arcs
    exts = [e for _, e, why in io.requests if e is not None]
    assert exts[0] == pytest.approx(0.6) and exts[-1] == 0.0
    assert exts[1:-1] == [1.6, 0.6] * 4
    assert min(io.ext_on_arc) >= 0.59
    assert io.requests[-1][2] == "release"
    # O-06 (sim budget 150 s computed; gate 165 s)
    assert 140.0 <= k["orbit_s"] <= t3.ORBIT_MAX_S, k["orbit_s"]
    assert k["radius_rms_truth_m"] <= 0.05 and k["az_ok"] and k["falls"] == 0
    assert k["coverage_ok"] and res["coverage_pass"]
    for b in cv.BANDS:
        assert res["coverage"]["bands"][b]["pct"] >= 90.0
    # files: frames, depth-free geometry frames, tags, strips, mp4, coverage, plan, manifest
    td = res["tree_dir"]
    lines = [json.loads(x) for x in open(os.path.join(td, "frames.jsonl"))]
    assert len(lines) == 24
    for key in ("theta_tree_deg", "R_m", "band", "camera", "cam_pos", "cam_fwd", "rtk", "t", "gimbal_deg", "ext_m",
                "steady", "station"):
        assert key in lines[0]
    assert {ln["band"] for ln in lines} == {"low", "mid", "high"}
    for b in cv.BANDS:
        assert os.path.isfile(os.path.join(td, f"strip_{b}.jpg"))
        assert os.path.isfile(os.path.join(td, f"{b}.mp4"))
        assert len([f for f in os.listdir(os.path.join(td, b)) if f.endswith(".jpg")]) == 8
    ok, root = mf.verify(td)
    assert ok and root == res["merkle_root"]
    assert k["bytes"] <= t3.BYTES_MAX
    # records
    rec, ds = res["record"], res["deepscan"]
    assert rec["schema"] == "nandi.tree/1" and rec["data_provenance"] == "simulated" and rec["tenant"] == "sim-nandi"
    raw = json.dumps(rec)
    assert len(raw.encode()) <= t3.RECORD_MAX_BYTES
    for forbidden in ("rgb", "image", "jpeg", "person", "face", "name"):
        assert forbidden not in rec
    assert rec["coverage"]["mode"] == "full" and rec["media"]["manifest_root"] == root
    assert rec["next_actions"] == ["deep_scan"]
    assert set(rec["assisted"]) >= {"station_teleport", "orbit_truth_radius", "kinematic_mast"}
    assert ds["schema"] == "nandi.deepscan/1" and ds["merkle_root"] == root
    assert ds["analysers"] == ["H-02", "H-03", "H-04", "H-06"] and ds["analysers_optional"] == ["H-05"]
    assert ds["reason"] == "operator" and ds["mission_id"].startswith("nandi-sim-01-")
    assert {c["path"].split("/", 2)[2] for c in ds["chunks"]} == {c["path"] for c in mf.list_chunks(td)}
    assert os.path.isfile(os.path.join(run.run_dir, "deepscan_OT-1.json"))
    assert os.path.isfile(os.path.join(run.run_dir, "record_OT-1.json"))


def test_tampered_media_fails_manifest(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})
    res = drive(run, io)
    td = res["tree_dir"]
    with open(os.path.join(td, "low", "0000.jpg"), "ab") as fh:
        fh.write(b"x")
    assert not mf.verify(td)[0]


def test_stop_and_go_costs_and_stows_on_arcs(tmp_path):
    run, io = make_run(tmp_path, {"crown_walk": "stop_and_go", "mp4": False})
    res = drive(run, io)
    k = res["kpi"]
    assert res["status"] == "done" and res["crown_walk"] == "stop_and_go" and k["frames"] == 24
    assert max(io.ext_on_arc) <= 0.01                     # rule 7: crown to 0 before every arc
    assert 215.0 <= k["orbit_s"] <= 260.0                # OS 1.2: ~232 s computed
    assert k["orbit_ok"] is None                          # O-06 gate is for snake; stop-and-go is reported


def test_person_pause_freezes_and_resumes(tmp_path):
    base, io0 = make_run(tmp_path / "a", {"mp4": False})
    t_base = drive(base, io0)["kpi"]["orbit_s"]
    run, io = make_run(tmp_path / "b", {"mp4": False})
    seen = {}

    def person(r, i):
        if r.phase == "arc" and "t0" not in seen and r.k_station == 2:
            seen["t0"] = i.t
        on = "t0" in seen and seen["t0"] <= i.t < seen["t0"] + 6.0
        i.inh["pause"] = on
        if on and r.paused:
            seen.setdefault("clock", r.clock)
            assert r.clock == seen["clock"]              # every timer frozen
            assert i.cmd == (0.0, 0.0)                   # gait stopped
    res = drive(run, io, hook=person)
    k = res["kpi"]
    assert res["status"] == "done" and k["pauses"] == 1
    assert k["pause_s"] == pytest.approx(6.0, abs=0.1)
    assert k["orbit_s"] == pytest.approx(t_base, abs=3.0)                # excl. the pause
    assert k["orbit_s_incl_pause"] == pytest.approx(k["orbit_s"] + k["pause_s"], abs=0.05)


def test_person_hold_too_long_asks_assist_then_halts(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})

    def person(r, i):
        i.inh["pause"] = r.phase in ("arc", "station", "raise") and i.t > 20.0
    res = drive(run, io, hook=person)
    assert res["status"] == "aborted" and res["abort_reason"] == "person_hold"
    assert res["kpi"]["pause_s"] == pytest.approx(t3.PAUSE_ABORT_S, abs=0.1)


@pytest.mark.parametrize("inh,reason", [({"stop": True, "stop_reason": "person_hold"}, "person_hold"),
                                        ({"hold": "estop"}, "estop")])
def test_safety_stop_halts_ordered(tmp_path, inh, reason):
    run, io = make_run(tmp_path, {"mp4": False})

    def stop(r, i):
        if r.k_station == 3 and r.phase == "arc":
            i.inh.update(inh)
    res = drive(run, io, hook=stop)
    assert res["status"] == "aborted" and res["abort_reason"] == reason
    assert io.cmd == (0.0, 0.0)
    # ordered (MP 2 "Abort = ordered stop, stow"): stow requested, waited for, preset source released
    assert io.requests[-2][1] == 0.0 and "abort" in io.requests[-2][2]
    assert io.requests[-1][2] == "release" and io.ext <= t3.CROWN_TOL_M
    assert "abort_stowed" in res["notes"]
    assert run.orbit_value == {"active": False}
    assert mf.verify(res["tree_dir"])[0]                    # partial media still sealed
    assert res["deepscan"]["status"] == "aborted" and res["record"]["next_actions"] == ["revisit", "tree360"]
    assert not res["coverage_pass"] and not res["deepscan_requested"]
    assert not os.path.exists(os.path.join(run.run_dir, "deepscan_OT-1.json"))


def test_radius_error_reentry_then_gap(tmp_path):
    """OS 5.3: |e| > 0.30 m -> stop, re-enter at the nearest station (2 x), then mark the gap."""
    run, io = make_run(tmp_path, {"mp4": False})
    kicks = {"n": 0, "armed": True}

    def kick(r, i):
        if r.phase == "arc" and r.k_station >= 2 and kicks["armed"] and kicks["n"] < 3:
            c = i.tree_xy
            ref = ob.ref_point((i.x, i.y), i.h)
            th = math.atan2(ref[1] - c[1], ref[0] - c[0])
            if ob.wrap(th - r.station_now.theta) > math.radians(20):
                i.x += 0.45 * math.cos(th)                   # a slip outward of 0.45 m
                i.y += 0.45 * math.sin(th)
                kicks["n"] += 1
                kicks["armed"] = False
        if r.phase == "station":
            kicks["armed"] = True
    res = drive(run, io, t_max=600.0, hook=kick)
    k = res["kpi"]
    assert res["status"] == "done"
    assert k["reentries"] == 2 and k["gaps"] == 1 and k["gap_deg"] > 0
    assert any(n.startswith("gap_marked") for n in res["notes"])


def test_crown_refusal_is_logged_and_coverage_is_honest(tmp_path):
    io = FakeIO()
    io.refuse[1.6] = "wind"
    run, io = make_run(tmp_path, {"mp4": False}, io=io)
    res = drive(run, io)
    assert res["status"] == "done"
    assert all(v == {"low", "mid"} for v in run.captured.values())
    assert any(e["event"] == "refused" and e["reason"] == "wind" for e in res["crown_events"])
    assert res["coverage"]["bands"]["high"]["pct"] < 90.0   # never extrapolated
    assert not res["coverage_pass"] and "high" in res["coverage_why"]
    # done but coverage failed: no Deep Scan request, the record asks for a revisit
    assert not res["deepscan_requested"] and res["record"]["next_actions"] == ["revisit", "tree360"]
    assert not os.path.exists(os.path.join(run.run_dir, "deepscan_OT-1.json"))


def test_walk_inhibit_switches_to_stop_and_go(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})

    def inhibit(r, i):
        i.inh["walk_block"] = r.phase == "arc" and i.ext > 0.02
    res = drive(run, io, hook=inhibit)
    assert res["status"] == "done" and res["crown_walk"] == "stop_and_go"
    assert any(n.startswith("crown_walk:stop_and_go:walk_inhibit") for n in res["notes"])
    assert res["kpi"]["frames"] == 24


def test_two_sided_hedgerow_with_transit(tmp_path):
    row = [{"tree_id": f"R1-{k + 1}", "x": 2.0 + 2.5 * k, "y": 2.5, "canopy_r_m": 0.99, "height_m": 2.2,
            "crop": "mango", "alley_sides": ["-y", "+y"], "orbit_clear": False, "trunk_r_m": 0.055}
           for k in range(4)]
    io = FakeIO(tree_xy=(4.5, 2.5))
    io.x, io.y = 4.5, -1.0
    run, io = make_run(tmp_path, {"mp4": False}, entry=row[1], registry=row, io=io)
    assert run.plan.mode == "two_sided"
    res = drive(run, io, t_max=800.0)
    k = res["kpi"]
    assert res["status"] == "done", res["abort_reason"]
    assert any(n.startswith("transit_m:") for n in res["notes"])
    assert k["min_neighbour_gap_m"] >= ob.ARC_CLEAR_M
    sides = {s.side for arc in run.plan.stations for s in arc}
    assert sides == {"-y", "+y"}
    for b in cv.BANDS:
        sd = res["coverage"]["bands"][b]["sides"]
        assert min(sd.values()) >= cv.TWO_SIDED_MIN_DEG, (b, sd)
    assert res["record"]["coverage"]["mode"] == "two_sided"
    assert res["deepscan"]["gap_deg"] > 0


def test_blocked_tree_is_refused(tmp_path):
    ring = [{"tree_id": f"N{k}", "x": 6.0 + 2.0 * math.cos(k * math.pi / 3), "y": -3.5 + 2.0 * math.sin(k * math.pi / 3),
             "canopy_r_m": 1.0} for k in range(6)]
    run, io = make_run(tmp_path, {"mp4": False}, registry=[OT1_ENTRY] + ring)
    res = drive(run, io)
    assert res["status"] == "aborted" and res["abort_reason"] == "orbit_blocked" and res["tree_dir"] is None


def test_progress_payload(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})
    for _ in range(600):
        io.step()
        run.tick()
    p = run.progress()
    assert p["tree_id"] == "OT-1" and p["phase"] in t3.PHASES and p["stations"] == 8
    assert set(p["coverage_pct"]) == set(cv.BANDS)
    json.dumps(p)
    assert run.orbit_value["active"] and run.orbit_value["arc"]




# ------------------------------------------------------------------------------------------- review fixes
def _no_raise(io):
    return [e for _, e, why in io.requests if e is not None and e > 0.0]


def test_walk_that_never_arrives_aborts_before_raise(tmp_path):
    """No cmd_vel bridge / stuck: the approach times out and the run halts; it never raises the crown at home."""
    io = FakeIO(v_bias=0.0, w_bias=0.0)
    run, io = make_run(tmp_path, {"entry": "walk", "mp4": False}, io=io)
    res = drive(run, io, t_max=400.0)
    assert res["status"] == "aborted" and res["abort_reason"] == "approach_timeout"
    assert _no_raise(io) == [] and res["kpi"]["frames"] == 0
    assert res["record"]["next_actions"] == ["revisit", "tree360"] and not res["deepscan_requested"]
    assert not os.path.exists(os.path.join(run.run_dir, "deepscan_OT-1.json"))


def test_off_station_after_approach_refuses_raise(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})
    moved = {}

    def shove(r, i):
        if r.phase == "approach" and i.t > 0.5 and not moved:
            moved["x"] = True
            i.x += 0.6                                      # pushed off the entry station before the raise
    res = drive(run, io, hook=shove)
    assert res["status"] == "aborted" and res["abort_reason"] == "approach_off_station"
    assert _no_raise(io) == []


def test_reentry_that_never_arrives_aborts(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})

    def kick(r, i):
        if r.phase == "arc" and r.k_station >= 1 and i.v_bias > 0:
            c = i.tree_xy
            ref = ob.ref_point((i.x, i.y), i.h)
            th = math.atan2(ref[1] - c[1], ref[0] - c[0])
            if ob.wrap(th - r.station_now.theta) > math.radians(20):
                i.x += 0.45 * math.cos(th)
                i.y += 0.45 * math.sin(th)
                i.v_bias = i.w_bias = 0.0                    # and then the legs stop answering
    res = drive(run, io, t_max=600.0, hook=kick)
    assert res["status"] == "aborted" and res["abort_reason"] == "reentry_timeout"
    assert io.ext <= t3.CROWN_TOL_M and io.requests[-1][2] == "release"


@pytest.mark.parametrize("bad", ["../../escaped", "a/b", ".hidden", "x" * 65, "/abs", "a..b"])
def test_run_id_is_one_safe_path_segment(tmp_path, bad):
    with pytest.raises(ValueError):
        make_run(tmp_path, {"run_id": bad})
    with pytest.raises(ValueError):
        make_run(tmp_path, {"device_id": bad})
    run, _ = make_run(tmp_path, {"run_id": "2026-11-18_ok.1"})
    assert os.path.dirname(os.path.dirname(run.tree_dir)) == os.path.join(str(tmp_path), "2026-11-18_ok.1")


def test_internal_fault_becomes_ordered_abort(tmp_path):
    """A fault inside the run never escapes tick(): ordered abort ``internal_error:<type>``; the run finishes."""
    (tmp_path / "file").write_text("x")                   # out_root under a regular file: makedirs fails
    run, io = make_run(tmp_path, {"mp4": False, "out_root": str(tmp_path / "file" / "runs")})
    res = drive(run, io)                                    # tick() must not raise
    assert run.finished and res["status"] == "aborted"
    assert res["abort_reason"].startswith("internal_error:")
    # a fault mid-orbit: stowed, released, partial media sealed
    run, io = make_run(tmp_path / "b", {"mp4": False})
    real = io.capture

    def flaky(camera, band, seq, tree_id):
        if seq == 7:
            raise RuntimeError("renderer gone")
        return real(camera, band, seq, tree_id)
    io.capture = flaky
    res = drive(run, io)
    assert res["status"] == "aborted" and res["abort_reason"] == "internal_error:RuntimeError"
    assert io.ext <= t3.CROWN_TOL_M and io.requests[-1][2] == "release"
    assert mf.verify(res["tree_dir"])[0]


def test_walk_inhibit_stands_in_stop_and_go(tmp_path):
    run, io = make_run(tmp_path, {"crown_walk": "stop_and_go", "mp4": False})
    driving, blocked = 0, 0
    while not run.finished and io.t < 500.0:
        io.step()
        io.inh["walk_block"] = run.phase == "arc" and 30.0 <= io.t < 45.0
        run.tick()
        if io.inh["walk_block"]:
            blocked += 1
            driving += int(io.cmd != (0.0, 0.0))
    assert blocked > 100 and driving == 0
    assert run.result["status"] == "done" and run.result["kpi"]["frames"] == 24


def test_walk_inhibit_too_long_halts(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})

    def inhibit(r, i):
        i.inh["walk_block"] = r.phase == "arc" or i.inh["walk_block"]
    res = drive(run, io, hook=inhibit)
    assert res["status"] == "aborted" and res["abort_reason"] == "stability"
    assert any(n == "crown_walk:stop_and_go:walk_inhibit" for n in res["notes"])     # stowed after 2 s first


def test_world_reset_aborts_the_run(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})
    st = {}

    def reset(r, i):
        if r.k_station == 2 and r.phase == "station" and "t" not in st:
            st["t"] = i.t
            i.t = 0.0                                        # /reset_world: sim time back to 0
            i.ext = i.ext_target = 0.0
    res = drive(run, io, hook=reset)
    assert res["status"] == "aborted" and res["abort_reason"] == "world_reset"
    assert io.t < 1.0                                       # at once, not after a timer measured on the old clock


def test_operator_pause_resume(tmp_path):
    run, io = make_run(tmp_path, {"mp4": False})
    st = {}

    def op(r, i):
        if r.k_station == 2 and r.phase == "arc" and "on" not in st:
            st["on"] = i.t
            r.pause(True)
        if "on" in st and "off" not in st and i.t >= st["on"] + 130.0:
            st["off"] = i.t
            assert r.paused and r.assist is None and not r.finished     # no person-hold timers
            r.pause(False)
        if "on" in st and "off" not in st and r.paused:
            assert i.cmd == (0.0, 0.0)
    res = drive(run, io, t_max=600.0, hook=op)
    assert res["status"] == "done" and res["kpi"]["pause_s"] == pytest.approx(130.0, abs=0.1)


def test_perception_capture_is_labelled(tmp_path):
    io = FakeIO()
    io.backend_name = lambda: "perception"
    run, io = make_run(tmp_path, {"mp4": False}, io=io)
    res = drive(run, io)
    assert {"capture_perception_remote", "coverage_geometry_only"} <= set(res["record"]["assisted"])


def test_low_light_aborts(tmp_path):
    io = FakeIO()
    io.light_ok = lambda: False
    run, io = make_run(tmp_path, {"mp4": False}, io=io)
    res = drive(run, io)
    assert res["status"] == "aborted" and res["abort_reason"] == "low_light" and _no_raise(io) == []
    io = FakeIO()
    run, io = make_run(tmp_path / "b", {"mp4": False}, io=io)

    def dusk(r, i):
        i.light_ok = (lambda: False) if r.k_station >= 3 else (lambda: True)
    res = drive(run, io, hook=dusk)
    assert res["status"] == "aborted" and res["abort_reason"] == "low_light" and io.ext <= t3.CROWN_TOL_M


def test_arm_not_parked_refuses(tmp_path):
    io = FakeIO()
    io.arm_parked = lambda: False
    run, io = make_run(tmp_path, {"mp4": False}, io=io)
    res = drive(run, io)
    assert res["status"] == "aborted" and res["abort_reason"] == "arm_not_parked" and _no_raise(io) == []


def test_disk_low_records_hybrid_or_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(t3, "disk_free", lambda p: 30 * 2 ** 20)
    run, io = make_run(tmp_path / "a")
    res = drive(run, io)
    assert res["status"] == "done" and any(n.startswith("disk_low:hybrid") for n in res["notes"])
    assert not run.cfg.mp4 and not run.cfg.write_depth and not res["files"]["mp4"]
    monkeypatch.setattr(t3, "disk_free", lambda p: 1 * 2 ** 20)
    run, io = make_run(tmp_path / "b")
    res = drive(run, io)
    assert res["status"] == "aborted" and res["abort_reason"] == "disk_low"


# ------------------------------------------------------------------------------------------- the hook
def test_hook_commands_start_abort_publish(tmp_path, monkeypatch):
    monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
    monkeypatch.setenv("NANDI_ACTORS", "0")
    monkeypatch.setenv("NANDI_RUNS_DIR", str(tmp_path))
    import atharv_nandi.sim.hooks as hr
    from atharv_nandi.core import farm_scene, sim_driver as sd
    from atharv_nandi.sim.hooks import tree360 as h3
    m = farm_scene.load_farm_model()
    drv = sd.SimDriver(m, hooks=hr.load_all(names=["mast", "lidar", "tree360"]))
    drv.bus.values["tree360_gait"] = drv.gait
    drv.deliver(h3.CMD_TOPIC, {"cmd": "start", "tree_id": "NOPE"})
    drv.deliver(h3.CMD_TOPIC, "not json")
    # a console command never picks an output path: out_root refused, run_id one safe segment
    drv.deliver(h3.CMD_TOPIC, {"cmd": "start", "tree_id": "OT-1", "params": {"out_root": "/proc/nope"}})
    drv.deliver(h3.CMD_TOPIC, {"cmd": "start", "tree_id": "OT-1", "params": {"run_id": "../../escaped"}})
    drv.deliver(h3.CMD_TOPIC, {"cmd": "pause"})
    drv.wait(0.1)
    errs = [json.loads(x) for _, tp, x in drv.published if tp == h3.PROGRESS_TOPIC]
    assert len(errs) == 5 and all("error" in e for e in errs), errs
    assert "out_root" in errs[2]["error"] and "run_id" in errs[3]["error"]
    assert h3.BUS_KEY not in drv.bus.values
    assert not drv.hook_faults
    drv.deliver(h3.CMD_TOPIC, {"cmd": "start", "tree_id": "OT-1",
                               "params": {"entry": "teleport", "capture": "geometry", "mp4": False, "run_id": "hook"}})
    drv.wait(4.0)
    run = drv.bus.values.get(h3.BUS_KEY)
    assert run is not None and run.phase in ("raise", "station")
    assert drv.bus.values["orbit"]["active"]
    # console Pause / Resume
    drv.deliver(h3.CMD_TOPIC, {"cmd": "pause"})
    drv.wait(0.5)
    assert run.paused and run.op_paused and run.progress()["operator_paused"]
    clock = run.clock
    drv.wait(1.0)
    assert run.clock == clock                              # every run timer frozen
    drv.deliver(h3.CMD_TOPIC, {"cmd": "resume"})
    drv.wait(0.2)
    assert not run.paused and run.clock > clock
    drv.wait(1.0)
    drv.deliver(h3.CMD_TOPIC, {"cmd": "abort"})
    drv.wait(0.1)
    # ordered: the crown stows first (MP 2), the run is detached only once it is down
    assert run.aborting and drv.bus.values["orbit"] == {"active": False}
    for _ in range(80):
        if h3.BUS_KEY not in drv.bus.values:
            break
        drv.wait(0.1)
    assert h3.BUS_KEY not in drv.bus.values and drv.bus.values["orbit"] == {"active": False}
    assert float(drv.bus.values["mast"]["ext_m"]) <= 0.01
    assert "preset" not in drv.bus.values["mast_control"].arb.requests
    res = drv.bus.values[h3.RESULT_KEY]
    assert res["status"] == "aborted" and res["abort_reason"] == "operator_abort"
    topics = [tp for _, tp, _ in drv.published]
    assert h3.RECORD_TOPIC in topics and h3.DEEPSCAN_TOPIC not in topics   # deep scan only on a passed visit
    assert not drv.hook_faults


# ------------------------------------------------------------------------------------------- slow
def _gl():
    from atharv_nandi.sim.hooks import tree360 as h3
    return h3._gl_ok()


@pytest.mark.slow
def test_ot1_visit_orchard_demo(tmp_path, monkeypatch):
    """OS 1.2 step 2 on orchard_demo: full orbit of OT-1 at r1 1.9 m, physics radius (WP5 trunk fit), WP24 mast,
    WP6 fields, the scripted worker walking into the crown-scaled warning field (orbit PAUSES, resumes once the worker
    is parked: the field's home is inside the high-band warning field, see the contract). Gates: coverage >= 90 % per
    band, radius RMS <= 0.10 m, orbit <= 165 s (excl. the pause), 0 falls, <= 30 MB, 0 unmasked person pixels (GL),
    manifest verifies, records written."""
    monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
    monkeypatch.setenv("NANDI_NOVNC_UP", "0")
    monkeypatch.delenv("NANDI_ACTORS", raising=False)
    monkeypatch.setenv("NANDI_SEED", "7")
    import atharv_nandi.sim.hooks as hr
    from atharv_nandi.core import farm_scene, sim_driver as sd
    from atharv_nandi.sim.hooks import tree360 as h3
    m = farm_scene.load_farm_model()
    drv = sd.SimDriver(m, hooks=hr.load_all(names=["mast", "stability", "lidar", "world_events", "tree360"]))
    drv.wait(0.3)
    gl = _gl()
    run = h3.start(drv.bus, drv.m, drv.d, "OT-1", {"entry": "teleport", "capture": "gl" if gl else "geometry",
                                                   "out_root": str(tmp_path), "run_id": "ot1"}, gait=drv.gait)
    flips = []
    while not run.finished and drv.t < 300.0:
        drv.wait(0.5)
        if not flips or flips[-1] != run.paused:
            flips.append(run.paused)
            if flips == [False, True, False]:
                drv.deliver("/nandi/world/cmd", {"actor": "worker_1", "action": "park"})
    res = drv.bus.values.get(h3.RESULT_KEY) or run.result
    k = res["kpi"]
    print("OT-1:", json.dumps(k))
    assert res["status"] == "done", res["abort_reason"]
    assert not drv.hook_faults
    assert k["pauses"] >= 1 and k["pause_s"] > 0                 # the worker paused the orbit, then it resumed
    for b in cv.BANDS:
        assert res["coverage"]["bands"][b]["pct"] >= 90.0, (b, res["coverage"]["bands"][b])
    assert k["radius_rms_truth_m"] <= 0.10 and k["orbit_s"] <= 165.0 and k["falls"] == 0
    assert k["az_ok"] and k["bearing_ok"] and k["steady_pct"] >= 95.0
    assert k["bytes"] <= 30 * 1024 * 1024
    assert mf.verify(res["tree_dir"])[0]
    if gl:
        assert k["person_unmasked_px"] == 0 and k["person_frames"] > 0
    topics = [tp for _, tp, _ in drv.published]
    assert h3.DEEPSCAN_TOPIC in topics and h3.RECORD_TOPIC in topics


@pytest.mark.slow
def test_two_sided_r1_2_orchard_demo(tmp_path, monkeypatch):
    """Hedgerow mango R1-2 (5 x 2.5 m): two-sided, >= 2 x 120 deg per band, arcs >= 0.3 m from the neighbours,
    transit around the row end, 0 falls."""
    monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
    monkeypatch.setenv("NANDI_ACTORS", "0")
    monkeypatch.setenv("NANDI_SEED", "7")
    import atharv_nandi.sim.hooks as hr
    from atharv_nandi.core import farm_scene, sim_driver as sd
    from atharv_nandi.sim.hooks import tree360 as h3
    m = farm_scene.load_farm_model()
    drv = sd.SimDriver(m, hooks=hr.load_all(names=["mast", "stability", "lidar", "tree360"]))
    drv.wait(0.3)
    run = h3.start(drv.bus, drv.m, drv.d, "R1-2", {"entry": "teleport", "capture": "geometry", "mp4": False,
                                                   "out_root": str(tmp_path), "run_id": "r12"}, gait=drv.gait)
    while not run.finished and drv.t < 320.0:
        drv.wait(1.0)
    res = run.result
    k = res["kpi"]
    print("R1-2:", json.dumps(k), res["notes"])
    assert res["status"] == "done", res["abort_reason"]
    assert res["mode"] == "two_sided" and not drv.hook_faults
    for b in cv.BANDS:
        sides = res["coverage"]["bands"][b]["sides"]
        assert len(sides) == 2 and min(sides.values()) >= 120.0, (b, sides)
    assert k["min_neighbour_gap_m"] >= 0.30 - 0.02
    assert k["radius_rms_truth_m"] <= 0.10 and k["falls"] == 0
