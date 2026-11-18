"""WP13 dock with the hub sync stub (MISSION-PRESETS 9; ORCHARD-MVP-SPEC 1.2 step 11, O-20; WP13):

    dock < 3 cm / 5 deg
    retry path (misaligned -> back off 0.3 m and retry (2), then assist)
    manifest closed (hub sync stub: L1 sent by the stub, L2 / L3 queued, local delete only after a hub receipt)
    ``slow`` dock <= 60 s (wall, the whole DOCK / UNLOAD with a low-battery inject, on the 2-CPU capped runner)

Unit tier: the pure logic on a kinematic plant (no MuJoCo), then the real gait on the orchard_demo world
(SimDriver, every atharv_nandi/sim/hooks file loaded). Headless, ROS-free; each unit test <= 10 s wall."""
import contextlib
import json
import math
import os
import time

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import dock_logic as dl
from atharv_nandi.core import energy_model as em
from atharv_nandi.core import farm_scene
from atharv_nandi.core import manifest as mf
from atharv_nandi.core.interfaces import GateDecision
from atharv_nandi.sim.world import compose
from atharv_nandi.sim.world.modules.tools import seat_tools


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


# ---------------------------------------------------------------------- kinematic plant (pure)
class Plant:
    """A kinematic stand-in for the tripod gait: velocities per unit command near the MEASURED ones (vx 0.24 m/s,
    vy 0.42 m/s, yaw 0.55 rad/s), and after a release the last command keeps running COAST_S (the gait finishing
    its cycle: 0.108 m per unit vx, close to the measured coast fit)."""
    VX, VY, WZ, COAST_S, DT = 0.24, 0.42, 0.55, 0.45, 0.02

    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw, self.t = float(x), float(y), float(yaw), 0.0
        self.last = dl.Cmd()
        self.coast = 0.0
        self.holding = False
        self.blocked = False
        self.crown_stowed = True

    def obs(self):
        walking = (not self.last.release) or self.coast > 0
        v = math.hypot(self.VX * self.last.vx, self.VY * self.last.vy) if walking else 0.0
        return dl.Obs(self.t, self.x, self.y, self.yaw, speed=v,
                      yaw_rate=self.WZ * self.last.yaw if walking else 0.0, walking=walking,
                      holding=self.holding, crown_stowed=self.crown_stowed)

    def step(self, cmd):
        if self.holding:
            self.last, self.coast = dl.Cmd(), 0.0
        elif cmd.release:
            if not self.last.release:
                self.coast = self.COAST_S
                self.last = dl.Cmd(self.last.vx, self.last.vy, self.last.yaw, True)
        else:
            self.last, self.coast = cmd, 0.0
        moving = (not self.last.release) or self.coast > 0
        if moving and not self.blocked and not self.holding:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            vx, vy = self.VX * self.last.vx, self.VY * self.last.vy
            self.x += (c * vx - s * vy) * self.DT
            self.y += (s * vx + c * vy) * self.DT
            self.yaw = dl.wrap(self.yaw + self.WZ * self.last.yaw * self.DT)
        if self.coast > 0:
            self.coast = max(0.0, self.coast - self.DT)
            if self.coast == 0.0:
                self.last = dl.Cmd()
        self.t += self.DT


def run_plant(ctl, plant, limit_s=200.0, each=None, update=None):
    upd = update or ctl.update
    while plant.t < limit_s:
        o = plant.obs()
        cmd = upd(o)
        if each is not None:
            each(plant, ctl)
        plant.step(cmd)
        if getattr(ctl, "done", False) or getattr(ctl, "step", "") in ("done", "assist"):
            break
    return ctl


DOCK = dl.DockGeometry(0.0, 0.0, 0.0)                    # orchard_demo: plate at home, reverse entry
# Wall budget of one real-gait sim test (2-CPU capped runner). Was 10 s at about 9.5 s measured before the WP5 merge;
# every SimDriver now also runs the WP5 lidar hook (10 Hz, 1,620 rays, about 4-5 ms per sweep on the orchard_demo
# terrain), measured 13.7-16 s for the 2-start test. 20 s keeps the gait-regression intent with headroom.
GAIT_WALL_S = 20.0


# ---------------------------------------------------------------------- geometry
def test_geometry_errors_staging_and_from_field():
    g = DOCK
    assert g.staging_xy() == pytest.approx((0.95, 0.0))     # reverse entry: staging in front of the plate
    assert g.clear_m == pytest.approx(0.85) and g.staging_m > g.clear_m   # every foot off the plate at staging
    e = g.error(0.6, 0.0, 0.0)
    assert e.remaining_m == pytest.approx(0.6) and e.lateral_m == pytest.approx(0.0) and e.yaw_deg == 0.0
    e = g.error(-0.02, 0.01, math.radians(3))                # 2 cm past, 1 cm left, 3 deg CCW
    assert e.remaining_m == pytest.approx(-0.02) and e.lateral_m == pytest.approx(0.01)
    assert e.yaw_deg == pytest.approx(3.0) and e.dist_m == pytest.approx(math.hypot(0.02, 0.01))
    assert e.ok() and not dl.DockError(0.0, 0.031, 0.0, 0.031).ok() and not dl.DockError(0, 0, 5.0, 0).ok()
    # rotated, forward entry
    g2 = dl.DockGeometry(1.0, 2.0, math.pi / 2, entry=+1, staging_m=0.5)
    assert g2.staging_xy() == pytest.approx((1.0, 1.5))
    e = g2.error(1.0 - 0.02, 1.9, math.pi / 2)              # 10 cm short, 2 cm to the dock-frame +y (= world -x)
    assert e.remaining_m == pytest.approx(0.1) and e.lateral_m == pytest.approx(0.02)
    f = dl.DockGeometry.from_field({"dock": {"xy": [1, 2], "yaw_deg": 90, "entry": "forward", "staging_m": 0.5,
                                              "size_m": [1.0, 0.8]}})
    assert f == g2
    assert dl.DockGeometry.from_field({"dock": {"xy": [0, 0], "size_m": [1.4, 0.8]}}).clear_m == pytest.approx(1.05)
    assert dl.crate_park_geometry((0.5, -1.5)).clear_m == 0.0              # no plate at the crate
    assert dl.DockGeometry.from_field({"dock": {"xy": [0, 0]}}) == DOCK
    with pytest.raises(ValueError):
        dl.DockGeometry(0, 0, 0, entry=0)
    # the coast predictions grow with the command (MEASURED fits)
    assert dl.coast_m(0.15) == pytest.approx(0.0198, abs=1e-3) and dl.coast_m(1.0) == pytest.approx(0.115)
    assert dl.yaw_coast_deg(0.2) < dl.yaw_coast_deg(0.5) and dl.strafe_coast_m(0.12) < dl.strafe_coast_m(0.5)


# ---------------------------------------------------------------------- aligner (pure, kinematic plant)
@pytest.mark.parametrize("start", [(1.5, 0.8, 2.5), (2.5, -0.5, 0.0), (0.0, 1.5, -1.0), (0.6, 0.0, 0.0)])
def test_aligner_docks_within_tolerance_on_the_plant(start):
    p = Plant(*start)
    a = dl.DockAligner(DOCK)
    assert a.start(p.t, xy=(p.x, p.y))
    run_plant(a, p)
    assert a.docked, a.status()
    assert a.final.dist_m < dl.DOCK_TOL_M and abs(a.final.yaw_deg) < dl.DOCK_TOL_DEG
    assert a.retries == 0
    st = a.status()
    assert st["state"] == "docked" and st["dock_s"] > 0 and st["attempts"][-1]["ok"]
    states = [x[1] for x in st["log"]]
    assert states[0] == "approach" and "entry" in states and states[-1] == "docked"


def slip(dy, n=1, when=lambda e: e.remaining_m < 0.25):
    """A scripted slip of ``dy`` (m, dock-frame y) during the entry, at most ``n`` times, once per entry."""
    done = {"n": 0, "entries": set()}

    def each(plant, a):
        e = a.last_error
        key = (len(a.attempts), a.retries)
        if a.state == "entry" and e is not None and when(e) and done["n"] < n and key not in done["entries"]:
            plant.y += dy
            done["n"] += 1
            done["entries"].add(key)
    return each, done


def test_retry_path_entry_abort_then_docked():
    """MP 9: misaligned -> back off 0.3 m and retry: a 6 cm slip leaves the entry cone, the robot backs out, re-aligns
    on the entry line and docks on the second try."""
    p = Plant(1.5, 0.8, 2.5)
    a = dl.DockAligner(DOCK)
    a.start(0.0)
    each, done = slip(0.06)
    run_plant(a, p, each=each)
    assert done["n"] == 1 and a.docked and a.retries == 1
    ev = [x["event"] for x in a.attempts]
    assert ev == ["entry_abort", "verify"]
    assert "backoff" in [x[1] for x in a.status()["log"]]
    assert a.final.ok()


def test_retry_path_verify_misaligned_then_docked():
    """A slip to 3.8 cm off the line at the very end stays inside the entry cone but fails the 3 cm check: back off and retry."""
    p = Plant(1.5, 0.8, 2.5)
    a = dl.DockAligner(DOCK)
    a.start(0.0)
    done = {"n": 0}

    def each(plant, al):                               # put the robot 3.8 cm off the line (cone 4.5 cm) near the end
        e = al.last_error
        if al.state == "entry" and e is not None and e.remaining_m < 0.04 and done["n"] == 0:
            plant.y = DOCK.y + 0.038
            done["n"] += 1
    run_plant(a, p, each=each)
    assert done["n"] == 1 and a.docked and a.retries == 1, a.status()
    v = [x for x in a.attempts if x["event"] == "verify"]
    assert [x["ok"] for x in v] == [False, True]
    assert v[0]["error"]["dist_m"] >= dl.DOCK_TOL_M


def test_two_retries_then_assist_dock_failed():
    p = Plant(1.5, 0.8, 2.5)
    a = dl.DockAligner(DOCK)
    a.start(0.0)
    each, done = slip(0.06, n=10)
    run_plant(a, p, each=each)
    assert a.state == "assist" and a.reason == dl.REASON_DOCK_FAILED
    assert a.retries == dl.MAX_RETRIES == 2 and done["n"] == 3          # first try + 2 retries
    assert [x["event"] for x in a.attempts] == ["entry_abort"] * 3
    assert a.update(p.obs()) == dl.RELEASE                               # done: no more commands


def test_refusals_crown_up_and_too_far():
    a = dl.DockAligner(DOCK)
    assert not a.start(0.0, crown_stowed=False)
    assert a.state == "assist" and a.reason == dl.REASON_CROWN
    assert not a.start(0.0, xy=(6.0, 0.0))
    assert a.state == "assist" and a.reason == dl.REASON_DOCK_FAILED and a.detail == "too_far"
    assert a.start(0.0, xy=(3.0, 0.0)) and a.state == "approach"


def test_estop_hold_pauses_then_resumes_and_docks():
    p = Plant(0.6, 0.02, 0.0)
    a = dl.DockAligner(DOCK)
    a.start(0.0)
    seen = {"held": 0, "cmds_while_held": 0}

    def each(plant, al):
        if al.state == "entry" and seen["held"] == 0 and al.last_error.remaining_m < 0.35:
            plant.holding = True
            seen["held"] = 1
        if plant.holding:
            seen["held"] += 1
            if seen["held"] > 100:                     # 2 s of hold, then released
                plant.holding = False
    orig = a.update

    def upd(o):
        c = orig(o)
        if o.holding and not c.release:
            seen["cmds_while_held"] += 1
        if o.holding:
            assert a.paused
        return c
    run_plant(a, p, each=each, update=upd)
    assert seen["held"] > 100 and seen["cmds_while_held"] == 0
    assert a.docked and a.final.ok() and not a.paused


def test_stuck_and_timeout_go_to_assist():
    p = Plant(2.0, 0.5, 0.0)
    p.blocked = True                                   # turning in place against a basin rim: no motion at all
    a = dl.DockAligner(DOCK)
    a.start(0.0)
    run_plant(a, p)
    assert a.state == "assist" and a.reason == dl.REASON_STUCK and a.detail == "stuck_approach"
    assert p.t < dl.STUCK_S + 2.0
    # a phase timeout (a plant that moves but never converges: a 90 deg yaw bias on every command)
    p = Plant(2.0, 0.5, 0.0)
    a = dl.DockAligner(DOCK)
    a.start(0.0)

    def spin(o):
        c = a.update(o)
        return c if c.release else dl.Cmd.go(0.0, 0.0, 0.3)
    run_plant(a, p, update=spin, limit_s=200.0)
    assert a.state == "assist" and a.detail == "timeout_approach"


# ---------------------------------------------------------------------- charger (pure)
def test_charger_cc_cv_matches_the_analytic_time():
    params = dl.ChargeParams()
    bat = em.Battery(soc0=0.25)
    ch = dl.TimeCompressedCharger(params)
    cap_ah = bat.p.capacity_wh / dl.nominal_v(bat)
    want_real_s = dl.analytic_charge_s(0.25, params, cap_ah)
    ch.start(0.0, bat)
    t, dt, cur = 0.0, 0.02, []
    while ch.state == "charging" and t < 600:
        t += dt
        ch.step(t, bat)
        cur.append((bat.soc, ch.current_a))
    assert ch.state == "full"
    assert ch.real_s == pytest.approx(want_real_s, rel=0.01)
    assert t == pytest.approx(want_real_s / params.compress, rel=0.01)    # time-compressed: sim = real / 240
    assert 15.0 < t < 40.0                                                # 25 % -> full in about half a minute
    assert bat.soc == pytest.approx(1.0 - params.cutoff_a * (1 - params.cv_soc) / params.cc_a, abs=2e-3)
    cc = [i for s, i in cur if s < params.cv_soc - 0.01]
    cv = [i for s, i in cur if s > params.cv_soc + 0.01]
    assert set(round(i, 6) for i in cc) == {params.cc_a}                    # CC
    assert cv and all(b <= a + 1e-9 for a, b in zip(cv, cv[1:])) and cv[-1] < 2 * params.cutoff_a  # CV taper
    st = ch.status(bat)
    assert st["state"] == "full" and st["assumed"] and st["charged_wh"] == pytest.approx(
        (bat.soc - 0.25) * bat.p.capacity_wh, rel=1e-3)
    # disabled (e-stop / off the plate): nothing charged
    b2 = em.Battery(soc0=0.5)
    c2 = dl.TimeCompressedCharger(params)
    c2.start(0.0, b2)
    c2.step(1.0, b2, enabled=False)
    assert b2.soc == pytest.approx(0.5) and c2.current_a == 0.0
    c2.step(2.0, b2)
    assert b2.soc > 0.5 and c2.status(b2)["eta_sim_s"] > 0
    c2.stop(3.0)
    assert c2.state == "idle" and c2.step(4.0, b2) == b2.soc
    with pytest.raises(ValueError):
        dl.ChargeParams(cutoff_a=20.0)
    with env(NANDI_DOCK_CHARGE_X="60"):
        assert dl.ChargeParams.from_env().compress == 60.0


def test_charge_x_env_rejects_non_finite_and_out_of_range_values():
    """NANDI_DOCK_CHARGE_X 'inf' used to pass the >= 1 check and raise OverflowError inside the 50 Hz task; now any
    non-finite, non-numeric or out-of-range value is a ValueError naming the variable when the session is built."""
    for bad in ("inf", "nan", "abc", "0.5", "1e300", "-inf"):
        with env(NANDI_DOCK_CHARGE_X=bad):
            with pytest.raises(ValueError, match="NANDI_DOCK_CHARGE_X"):
                dl.DockUnloadSession(DOCK, battery=em.Battery(soc0=0.3))
    for kw in ({"compress": math.inf}, {"cc_a": math.nan}, {"eta": math.inf}, {"compress": dl.COMPRESS_MAX * 2}):
        with pytest.raises(ValueError):
            dl.ChargeParams(**kw)
    # the largest allowed compression still integrates in a bounded number of sub-steps and ends full
    bat = em.Battery(soc0=0.3)
    ch = dl.TimeCompressedCharger(dl.ChargeParams(compress=dl.COMPRESS_MAX))
    ch.start(0.0, bat)
    t0 = time.perf_counter()
    for k in range(1, 50):
        ch.step(0.02 * k, bat)
    assert ch.state == "full" and bat.soc > 0.98 and time.perf_counter() - t0 < 1.0


# ---------------------------------------------------------------------- hub sync stub (pure)
def _run_dir(tmp_path):
    d = tmp_path / "runs" / "sim-nandi-20261117T0900Z"
    (d / "media").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"run_id": d.name}))
    (d / "tree_records.jsonl").write_text('{"tree_id": "OT-1"}\n')
    (d / "media" / "kf_000.jpg").write_bytes(b"\xff\xd8 not a real jpeg")
    return str(d)


def test_hub_sync_closes_the_manifest_and_gates_local_delete(tmp_path):
    rd = _run_dir(tmp_path)
    s = dl.HubSyncStub(sync=True)
    assert s.verify() == (False, "not_closed") and s.may_delete_local() == (False, "not_closed")
    m = s.close(rd, {"run_id": "sim-nandi-20261117T0900Z", "device_id": "nandi-sim"}, closed_ms=123)
    assert m["final"] is True and m["closed"] == 123 and m["sync"] == "on"
    assert [c["path"] for c in m["chunks"]] == ["media/kf_000.jpg", "meta.json", "tree_records.jsonl"]
    on_disk = json.load(open(os.path.join(rd, "manifest.json")))
    assert on_disk["merkle_root"] == m["merkle_root"] == mf.root_of(mf.list_chunks(rd))
    assert s.verify() == (True, m["merkle_root"])
    assert s.lanes == {"L1": "sent_stub", "L2": "queued", "L3": "queued"}
    assert s.close(rd) is m                                              # idempotent
    # delete-after-receipt rule: no delete before a matching hub receipt for every lane
    assert s.may_delete_local() == (False, "no_hub_receipt")
    assert not s.hub_receipt({"merkle_root": "0" * 64}) and s.refusals == ["receipt_mismatch"]
    assert s.hub_receipt({"merkle_root": m["merkle_root"], "lanes": ["L1"]})
    assert s.may_delete_local() == (False, "lanes_pending")
    assert s.hub_receipt({"merkle_root": m["merkle_root"]})
    assert s.may_delete_local() == (True, "hub_receipt")
    assert os.path.isfile(os.path.join(rd, "media", "kf_000.jpg"))       # the sim never deletes
    st = s.status()
    assert st["manifest_closed"] and st["chunks"] == 3 and st["may_delete_local"] and st["stub"]
    # tampering after the close is detected
    open(os.path.join(rd, "tree_records.jsonl"), "a").write("x")
    assert s.verify()[0] is False


def test_hub_sync_off_still_closes_and_in_memory_chunks():
    s = dl.HubSyncStub(sync=False)
    ch = [{"path": "b.txt", "sha256": "11" * 32}, {"path": "a.txt", "sha256": "22" * 32}]
    m = s.close(chunks=ch, meta={"run_id": "r1"}, closed_ms=5)
    assert s.closed and m["sync"] == "off" and m["final"]
    assert m["merkle_root"] == mf.root_of(ch) and [c["path"] for c in m["chunks"]] == ["a.txt", "b.txt"]
    assert s.lanes == {"L1": "idle", "L2": "idle", "L3": "idle"}
    assert s.verify() == (True, m["merkle_root"]) and s.status()["sync"] == "off"


# ---------------------------------------------------------------------- DOCK / UNLOAD session (pure)
class FakeUnload:
    """WP3 TransferController shape: refuses ``refuse`` first, then unloads in ``busy_ticks`` ticks."""

    def __init__(self, refuse=(), busy_ticks=10, items=2):
        self.refuse = list(refuse)
        self.busy_ticks = busy_ticks
        self.items = items
        self.calls = 0

    def per_tree(self):
        if self.items == 0:
            return {}
        return {"OT-1": {"tray_n": self.items if self.busy_ticks else 0, "crate_n": 0 if self.busy_ticks else
                         self.items, "crate_kg": 0.0 if self.busy_ticks else 0.6}}

    def unload(self, d, inputs=None):
        self.calls += 1
        if self.refuse:
            return GateDecision(False, "transfer", self.refuse.pop(0))
        return GateDecision(True, "transfer")

    def busy(self):
        self.busy_ticks = max(0, self.busy_ticks - 1)
        return self.busy_ticks > 0

    def release_crate_slots(self, d):
        return ["s0", "s1"]

    def crate_summary(self):
        return {"n": self.items, "kg": 0.6, "dropped": 0, "max_impact_mps": 0.39}


def test_session_steps_outputs_and_low_battery_charge(tmp_path):
    rd = _run_dir(tmp_path)
    p = Plant(2.0, -0.6, 2.0)
    bat = em.Battery(soc0=0.2)
    park = dl.crate_park_geometry((0.5, -1.5))
    assert park.x == pytest.approx(0.5) and park.y == pytest.approx(-0.97) and park.entry == 1
    sheets = []
    s = dl.DockUnloadSession(DOCK, park, battery=bat, run_dir=rd, meta={"run_id": "r"},
                             jobsheets=lambda rows: sheets.append(rows) or {"status": "ok", "rows": rows},
                             rows_visited=["R1", "R2"])
    u = FakeUnload(refuse=["gait_moving"])
    assert s.start(0.0, trigger="energy_reserve", unload=u, xy=(p.x, p.y))
    run_plant(s, p, update=lambda o: s.update(o, None, u, object()))
    assert s.step == "done", s.status()
    steps = [e["step"] for e in s.events if e["event"] == "step"]
    assert steps == ["park_crate", "unload", "jobsheets", "dock", "charge", "sync", "done"]
    assert u.calls == 2 and s.unload_refusal == "gait_moving"              # interlock retried, not an assist
    out = s.outputs
    assert out["crate_park"]["dist_m"] < dl.PARK_TOL_M
    assert out["crate"]["n"] == 2 and out["crate"]["per_tree"]["OT-1"]["crate_n"] == 2
    assert out["crate"]["slots_released"] == 2
    assert sheets == [["R1", "R2"]] and out["jobsheets"]["status"] == "ok"
    assert out["dock"]["final"]["dist_m"] < dl.DOCK_TOL_M and abs(out["dock"]["final"]["yaw_deg"]) < dl.DOCK_TOL_DEG
    assert out["charge"]["state"] == "full" and bat.soc > 0.98 and out["charge"]["soc0"] == pytest.approx(0.2)
    assert out["sync"]["manifest_closed"] and out["manifest"]["final"] and mf.verify(rd)[0]
    st = s.status()
    assert st["step"] == "done" and st["dock_state"] == "docked" and st["manifest_closed"]
    assert st["crate"]["n"] == 2 and st["elapsed_s"] > 0 and st["trigger"] == "energy_reserve"


def test_session_without_items_skips_the_crate_and_assist_paths():
    p = Plant(0.9, 0.3, 0.0)
    s = dl.DockUnloadSession(DOCK, dl.crate_park_geometry((0.5, -1.5)), battery=None)
    s.start(0.0, unload=FakeUnload(items=0), xy=(p.x, p.y))
    run_plant(s, p)
    steps = [e["step"] for e in s.events if e["event"] == "step"]
    assert steps == ["jobsheets", "dock", "sync", "done"]                  # no battery: no charge step
    assert s.outputs["jobsheets"]["status"] == "pending_wp9"
    # crown up: refused before any motion
    s2 = dl.DockUnloadSession(DOCK)
    assert not s2.start(0.0, crown_stowed=False) and s2.step == "assist" and s2.reason == dl.REASON_CROWN
    assert s2.update(p.obs()) == dl.RELEASE
    # an unload refusal that does not clear by itself -> assist with its reason
    p3 = Plant(0.0, -0.97, 0.0)
    s3 = dl.DockUnloadSession(DOCK, dl.crate_park_geometry((0.5, -1.5)))
    u3 = FakeUnload(refuse=["crate_out_of_reach"])
    s3.start(0.0, unload=u3, xy=(p3.x, p3.y))
    run_plant(s3, p3, update=lambda o: s3.update(o, None, u3, None))
    assert s3.step == "assist" and s3.reason == "crate_out_of_reach"
    # docking fails -> the session ends in assist with dock_failed, aligner status kept for the console
    p4 = Plant(1.5, 0.8, 2.5)
    s4 = dl.DockUnloadSession(DOCK)
    s4.start(0.0, xy=(p4.x, p4.y))
    each, _ = slip(0.06, n=10)
    run_plant(s4, p4, each=lambda pl, ss: each(pl, ss.dock_aligner))
    assert s4.step == "assist" and s4.reason == dl.REASON_DOCK_FAILED
    assert s4.status()["aligner"]["retries"] == 2 and s4.status()["dock_state"] == "assist"


def _session_to_charge(compress=240.0, soc0=0.3):
    """A pure session docked on the kinematic plant with the charger just started."""
    p = Plant(1.2, 0.0, 0.0)
    bat = em.Battery(soc0=soc0)
    s = dl.DockUnloadSession(DOCK, battery=bat, charger=dl.TimeCompressedCharger(dl.ChargeParams(compress=compress)))
    s.start(0.0, xy=(p.x, p.y))
    while s.step != "charge" and p.t < 200.0:
        p.step(s.update(p.obs()))
    assert s.step == "charge" and s.charger.state == "charging"
    return p, bat, s


def test_charge_off_the_dock_tolerance_faults_closes_the_manifest_and_goes_to_assist(tmp_path):
    """Review finding: the charge step had no exit, so 5 cm off the plate the session sat in 'charge' at 0 A for ever
    and never closed the manifest. Now: off the tolerance for CHARGE_CONTACT_S -> charge_contact_lost, sync still
    closes the manifest, then assist."""
    p, bat, s = _session_to_charge()
    s.syncer = dl.HubSyncStub(True)
    p.x += 0.05                                         # drift off the 3 cm dock tolerance
    soc = bat.soc
    t0 = p.t
    while s.active and p.t < t0 + 600.0:
        p.step(s.update(p.obs()))
    assert s.step == "assist" and s.reason == dl.REASON_CONTACT and s.fault == dl.REASON_CONTACT
    assert dl.CHARGE_CONTACT_S - 0.05 <= p.t - t0 <= dl.CHARGE_CONTACT_S + 0.1
    assert bat.soc == pytest.approx(soc) and s.charger.state == "idle"           # charged nothing off the plate
    assert s.outputs["charge"]["fault"] == dl.REASON_CONTACT and s.outputs["charge"]["error"]["dist_m"] > 0.03
    assert s.syncer.closed and s.outputs["sync"]["manifest_closed"] and s.status()["fault"] == dl.REASON_CONTACT
    assert [e["step"] for e in s.events if e["event"] == "step"][-3:] == ["charge", "sync", "assist"]
    assert any(e["event"] == "charge_fault" for e in s.events)
    # a hold off the plate does not count (the e-stop is the operator's): no fault while held
    p, bat, s = _session_to_charge()
    p.x += 0.05
    p.holding = True
    t0 = p.t
    while p.t < t0 + 3 * dl.CHARGE_CONTACT_S:
        p.step(s.update(p.obs()))
    assert s.step == "charge" and not s.fault
    # back on the plate after a brief slip: no fault, the charge finishes
    p, bat, s = _session_to_charge()
    p.x += 0.05
    for _ in range(int(0.5 * dl.CHARGE_CONTACT_S / Plant.DT)):
        p.step(s.update(p.obs()))
    p.x -= 0.05
    while s.active and p.t < 300.0:
        p.step(s.update(p.obs()))
    assert s.step == "done" and s.outputs["charge"]["state"] == "full" and bat.soc > 0.98


def test_charge_timeout_goes_to_assist_and_hold_time_does_not_count():
    p, bat, s = _session_to_charge(compress=1.0)       # real-time charge: hours of sim, far past any test
    want = dl.analytic_charge_s(0.3, s.charger.p, bat.p.capacity_wh / dl.nominal_v(bat))
    assert s._charge_limit_s == pytest.approx(want * dl.CHARGE_TIMEOUT_FACTOR + dl.CHARGE_TIMEOUT_PAD_S, rel=0.02)
    s._charge_limit_s = 3.0                            # shrink the limit: the path, not the numbers
    p.holding = True
    for _ in range(int(5.0 / Plant.DT)):               # 5 s in a hold: not counted
        p.step(s.update(p.obs()))
    assert s.step == "charge"
    p.holding = False
    while s.active and p.t < 400.0:
        p.step(s.update(p.obs()))
    assert s.step == "assist" and s.reason == dl.REASON_CHARGE_TIMEOUT and s.syncer.closed
    assert s.outputs["charge"]["fault"] == dl.REASON_CHARGE_TIMEOUT


def test_crown_raised_mid_dock_stops_the_gait_and_resumes_when_stowed():
    """Review finding: the crown gate was checked only at start. Now a crown raised mid-dock releases the gait (no
    walking until stowed, MP 8), the phase clock does not run, and the dock resumes once stowed; held up past
    CROWN_WAIT_S -> assist crown_gate."""
    p = Plant(2.5, -0.5, 0.0)
    a = dl.DockAligner(DOCK)
    a.start(0.0, xy=(p.x, p.y))
    moving = n = 0
    while p.t < 200.0 and not a.done:
        if 2.0 <= p.t < 6.0:
            p.crown_stowed = False
        else:
            p.crown_stowed = True
        c = a.update(p.obs())
        if not p.crown_stowed:
            n += 1
            moving += not c.release
            assert a.paused and a.detail == "crown_up"
        p.step(c)
    assert n >= 195 and moving == 0
    assert a.docked and a.final.ok() and a.detail == ""
    assert [x["event"] for x in a.attempts] == ["crown_up", "verify"]
    # held up too long -> assist crown_gate
    p = Plant(2.5, -0.5, 0.0)
    a = dl.DockAligner(DOCK)
    a.start(0.0, xy=(p.x, p.y))
    while p.t < 100.0 and not a.done:
        p.crown_stowed = p.t < 1.0
        p.step(a.update(p.obs()))
    assert a.state == "assist" and a.reason == dl.REASON_CROWN
    assert dl.CROWN_WAIT_S < p.t < dl.CROWN_WAIT_S + 1.5


def test_session_entering_the_dock_step_checks_the_crown():
    """After park / unload / job sheets the dock step starts the aligner with the crown state of that tick."""
    p = Plant(0.0, -0.97, 0.0)
    s = dl.DockUnloadSession(DOCK, dl.crate_park_geometry((0.5, -1.5)))
    u = FakeUnload()
    s.start(0.0, unload=u, xy=(p.x, p.y))

    def upd(o):
        if s.step == "jobsheets":
            p.crown_stowed = False                     # raised during the unload / job sheets
        return s.update(p.obs(), None, u, None)
    run_plant(s, p, update=upd)
    assert s.step == "assist" and s.reason == dl.REASON_CROWN
    assert s.dock_aligner.state == "assist" and s.dock_aligner.reason == dl.REASON_CROWN


def test_crown_stowed_reading_fails_closed():
    """obs_from_driver used to read a missing bus.values['mast'] as stowed. Now only a model without a mast is stowed
    without a reading."""
    from types import SimpleNamespace as NS
    from atharv_nandi.sim.hooks import mast as hm
    with env(NANDI_WORLD="orchard_demo"):
        m = farm_scene.load_farm_model()
    has = hm.has_mast(m)
    drv = lambda values: NS(bus=NS(values=values), m=m)
    assert dl.crown_stowed(drv({"mast": {"ext_m": 0.0}})) is True
    assert dl.crown_stowed(drv({"mast": {"ext_m": 0.5}})) is False
    assert dl.crown_stowed(drv({"mast": {"ext_m": 0.0, "has_mast": False}})) is True
    for values in ({}, {"mast": {}}, {"mast": {"ext_m": None}}, {"mast": {"ext_m": "nan"}}, {"mast": "x"}):
        assert dl.crown_stowed(drv(values)) is (not has), values


def test_retry_strafes_only_with_every_foot_off_the_plate():
    """MP 9.1: misalignment is corrected by reverse-and-re-approach, never by side steps over the dock. Review
    finding: the retry used to strafe at 0.24-0.26 m before the centre (trunk over the plate). Now every strafe runs
    at remaining >= clear_m (plate half-length + FOOT_REACH_M), and a lateral error found nearer walks out first."""
    for slip_fn in (slip(0.06)[0], None):
        p = Plant(1.5, 0.8, 2.5)
        a = dl.DockAligner(DOCK)
        a.start(0.0)
        rems = []

        def each(plant, al, f=slip_fn):
            if f is not None:
                f(plant, al)
            if al.state == "align" and al._moving and al._move_axis == "lat":
                rems.append(al.last_error.remaining_m)
        run_plant(a, p, each=each)
        assert a.docked and rems
        assert min(rems) >= DOCK.clear_m and a.strafe_min_remaining_m >= DOCK.clear_m
    # a lateral error found close to the plate (resumed from a hold at 0.4 m, 5 cm off the line): out first, then strafe
    p = Plant(0.4, 0.05, 0.0)
    a = dl.DockAligner(DOCK)
    a.start(0.0)
    a._to("align", 0.0)
    seen = []
    run_plant(a, p, each=lambda pl, al: seen.append((al._move_axis, al.last_error.remaining_m))
              if al.state == "align" and al._moving else None)
    assert a.docked and a.final.ok()
    lat = [r for ax, r in seen if ax == "lat"]
    assert lat and min(lat) >= DOCK.clear_m and seen[0][0] == "along"


class _FakeReserve:
    def __init__(self, action="continue"):
        self.action, self.reason, self.latched_at = action, ("energy_reserve" if action != "continue" else ""), 1.0

    def clear(self):
        self.action, self.reason, self.latched_at = "continue", "", None


class _FakePlatform:
    """WP7 Platform shape for the plan hand-over: ``plan`` / ``remaining_plan()`` / ``reserve``; applies the
    /nandi/platform/plan messages like ``_cmd_plan`` (``clear`` drops the plan and clears the reserve latch)."""

    def __init__(self, plan=None, latch="continue"):
        from atharv_nandi.core import route_energy as re_
        self.re_ = re_
        self.plan = None if plan is None else re_.RoutePlan.from_dict(plan)
        self.reserve = _FakeReserve(latch)

    def remaining_plan(self):
        return self.plan

    def apply(self, msg):
        if msg.get("clear"):
            self.plan = None
            self.reserve.clear()
        else:
            self.plan = self.re_.RoutePlan.from_dict(msg)


class _FakeDrv:
    def __init__(self):
        self.sent, self.t, self.m = [], 0.0, object()

    def add_task(self, *a, **k):
        pass

    def deliver(self, topic, payload):
        self.sent.append((topic, dict(payload)))


@pytest.mark.parametrize("prior,latch,full", [(None, "continue", True), (None, "return_and_wait", False),
                                              (None, "return_and_wait", True),
                                              ({"route_m": 40.0, "items": 3, "t_work_s": 600.0}, "return_and_wait",
                                               False),
                                              ({"route_m": 40.0, "items": 3, "t_work_s": 600.0}, "return_and_wait",
                                               True)])
def test_runner_holds_the_platform_plan_for_the_whole_session_and_restores_the_prior_one(monkeypatch, prior, latch,
                                                                                         full):
    """Review findings: the WP13 plan went out only in 'charge' (WP7's instant fill fired during an e-stop near the
    plate) and the end sent {clear} over any mission plan, which also cleared the reserve latch. Now the plan is held
    from the session's first tick; at the end the prior plan is re-sent (or cleared when there was none) and the
    reserve latch is kept unless the charge filled the pack."""
    plat = _FakePlatform(prior, latch)
    monkeypatch.setattr(dl, "platform_runtime", lambda m: plat)
    drv = _FakeDrv()
    s = dl.DockUnloadSession(DOCK, battery=em.Battery(soc0=0.3))
    r = dl.DockRunner(drv, s)

    def tick():
        r._platform_plan(drv)
        while drv.sent:
            topic, msg = drv.sent.pop(0)
            assert topic == dl.PLATFORM_PLAN_TOPIC
            plat.apply(msg)
    tick()
    assert plat.plan == (None if prior is None else plat.re_.RoutePlan.from_dict(prior))   # idle: untouched
    s.start(0.0, xy=(1.2, 0.0))
    for step in ("jobsheets", "dock", "charge"):        # every active step holds the WP13 plan
        s.step = step
        tick()
        assert plat.plan == plat.re_.RoutePlan.from_dict(dl.WP13_PLAN), step
    if full:
        s.charger.state = "full"
    s.step = "done"
    tick()
    tick()                                              # the latch restore runs once WP7 has applied the clear
    if prior is None:
        assert plat.plan is None
    else:
        assert plat.plan == plat.re_.RoutePlan.from_dict(prior)
    assert plat.reserve.action == ("continue" if full else latch)
    ev = [e for e in s.events if e["event"] == "platform_plan"]
    assert [e["action"] for e in ev] == ["hold", "cleared" if prior is None else "restored"]
    assert ev[0]["prior_plan"] is (prior is not None) and ev[1]["charged_full"] is full


@pytest.mark.parametrize("exit_by", ["leave_radius", "acknowledge", "new_session"])
def test_runner_keeps_the_plan_after_an_assist_at_home_until_the_robot_leaves_or_the_operator_acks(monkeypatch,
                                                                                                    exit_by):
    """Review finding: a session ending in assist near home (charge_contact_lost, charge_timeout, dock_failed) sent
    {clear} at once, and 5 s later WP7's 'parked at home, no plan' rule filled the pack to 100 % (and cleared the
    reserve latch WP13 had just put back): a charge fault next to a full battery. Now the plan stays held while the
    robot is within WP7's dock radius of home, until it leaves that radius or the operator acknowledges."""
    from atharv_nandi.sim.hooks.platform import DOCK_RADIUS_M
    plat = _FakePlatform(None, "return_and_wait")
    plat.home = (0.0, 0.0)
    monkeypatch.setattr(dl, "platform_runtime", lambda m: plat)
    drv = _FakeDrv()
    s = dl.DockUnloadSession(DOCK, battery=em.Battery(soc0=0.3))
    r = dl.DockRunner(drv, s)

    def tick(xy):
        r._platform_plan(drv, xy)
        while drv.sent:
            plat.apply(drv.sent.pop(0)[1])
    s.start(0.0, xy=(1.2, 0.0))
    s.step = "charge"
    tick((0.0, 0.0))
    held = plat.re_.RoutePlan.from_dict(dl.WP13_PLAN)
    assert plat.plan == held
    s.step, s.reason, s.fault = "assist", dl.REASON_CONTACT, dl.REASON_CONTACT
    for _ in range(3):                                  # at home, no ack: kept (WP7 cannot fill)
        tick((0.04, 0.0))
    assert plat.plan == held and plat.reserve.action == "return_and_wait"
    ev = [e for e in s.events if e["event"] == "platform_plan"]
    assert [e["action"] for e in ev] == ["hold", "kept"] and ev[1]["reason"] == dl.REASON_CONTACT
    if exit_by == "leave_radius":
        tick((DOCK_RADIUS_M + 0.05, 0.0))
    elif exit_by == "acknowledge":
        r.acknowledge()
        tick((0.04, 0.0))
    else:                                               # a new session: still held, released at its end
        s.start(1.0, xy=(1.2, 0.0))
        s.step = "dock"
        tick((0.04, 0.0))
        assert plat.plan == held and not r._acked
        s.step = "done"
        tick((0.0, 0.0))
    tick((0.0, 0.0))
    assert plat.plan is None and plat.reserve.action == "return_and_wait"   # cleared, latch kept (not full)
    assert [e["action"] for e in s.events if e["event"] == "platform_plan"][-1] == "cleared"
    # a 'done' end (charged full) releases at once, and an assist away from home does too
    for step, xy in (("done", (0.0, 0.0)), ("assist", (DOCK_RADIUS_M + 0.2, 0.0))):
        s.start(2.0, xy=(1.2, 0.0))
        s.step = "charge"
        tick(xy)
        assert plat.plan == held
        s.step = step
        tick(xy)
        assert plat.plan is None, step
    # our own plan left on the platform by an earlier runner (never applied {clear}) is not "restored" at the end
    plat.plan = plat.re_.RoutePlan.from_dict(dl.WP13_PLAN)
    s.start(3.0, xy=(1.2, 0.0))
    s.step = "charge"
    tick((0.0, 0.0))
    s.step = "done"
    tick((0.0, 0.0))
    assert plat.plan is None and s.events[-1]["action"] == "cleared"


def test_unload_interlock_retry_is_rate_limited_skips_holds_and_times_out():
    """Review finding: an interlock refusal was retried on every 50 Hz tick with no limit (503 WP3 'refused' events
    in a 10 s e-stop, the 200-entry event buffer flushed) and could wait for ever (arm never parks). Now: no WP3
    call during a hold, retries at most every UNLOAD_RETRY_S, and after UNLOAD_WAIT_S refused (hold time not counted)
    -> assist with the refusal reason."""
    park = dl.crate_park_geometry((0.5, -1.5))
    # e-stop held 10 s in the unload step: WP3 not asked while held; then the unload completes
    p = Plant(0.0, -0.97, 0.0)
    s = dl.DockUnloadSession(DOCK, park)
    u = FakeUnload(refuse=["estop"])
    s.start(0.0, unload=u, xy=(p.x, p.y))
    while s.step != "unload" and p.t < 100.0:
        p.step(s.update(p.obs(), None, u, None))
    assert s.step == "unload"
    p.holding = True
    for _ in range(int(10.0 / Plant.DT)):
        p.step(s.update(p.obs(), None, u, None))
    assert u.calls == 0 and s.step == "unload"
    p.holding = False
    run_plant(s, p, update=lambda o: s.update(o, None, u, None))
    assert s.step == "done" and s.outputs["crate"]["n"] == 2 and u.calls == 2
    # an arm that never parks: ~1 call per UNLOAD_RETRY_S, then assist arm_not_parked after UNLOAD_WAIT_S
    p = Plant(0.0, -0.97, 0.0)
    s = dl.DockUnloadSession(DOCK, park)
    u = FakeUnload(refuse=["arm_not_parked"] * 10000)
    s.start(0.0, unload=u, xy=(p.x, p.y))
    while s.step != "unload" and p.t < 100.0:
        p.step(s.update(p.obs(), None, u, None))
    t0 = p.t
    p.holding = True                                    # 5 s of hold first: not counted, no call
    for _ in range(int(5.0 / Plant.DT)):
        p.step(s.update(p.obs(), None, u, None))
    p.holding = False
    while s.active and p.t < t0 + 200.0:
        p.step(s.update(p.obs(), None, u, None))
    waited = p.t - t0 - 5.0
    assert s.step == "assist" and s.reason == "arm_not_parked"
    assert dl.UNLOAD_WAIT_S - 0.1 <= waited <= dl.UNLOAD_WAIT_S + 0.2
    assert u.calls <= dl.UNLOAD_WAIT_S / dl.UNLOAD_RETRY_S + 2
    assert [e["reason"] for e in s.events if e["event"] == "unload_wait"] == ["arm_not_parked"]   # once per reason


def test_items_without_a_crate_park_pose_are_recorded_as_skipped():
    """Review finding: crate_park=None with items in the tray skipped MP 9 step 2 with no record. Now an
    'unload_skipped' event (reason no_crate_park, tray count) and outputs/status 'crate' say so."""
    p = Plant(0.9, 0.3, 0.0)
    s = dl.DockUnloadSession(DOCK, None)
    u = FakeUnload(items=2)
    s.start(0.0, unload=u, xy=(p.x, p.y))
    run_plant(s, p, update=lambda o: s.update(o, None, u, None))
    assert s.step == "done" and u.calls == 0
    ev = [e for e in s.events if e["event"] == "unload_skipped"]
    assert ev and ev[0]["reason"] == dl.REASON_NO_CRATE_PARK and ev[0]["tray_n"] == 2
    assert s.status()["crate"] == {"crate_id": "crate", "skipped": dl.REASON_NO_CRATE_PARK, "tray_n": 2}
    # an empty tray without a crate park pose: nothing to record
    s2 = dl.DockUnloadSession(DOCK, None)
    s2.start(0.0, unload=FakeUnload(items=0), xy=(p.x, p.y))
    assert not any(e["event"] == "unload_skipped" for e in s2.events) and "crate" not in s2.outputs


def test_charge_contact_band_is_wider_than_the_entry_gate():
    """Review finding: the charge contact check used the 3 cm / 5 deg entry gate, so a dock verified near the edge
    (2.5 cm) faulted on a 5 mm settle. Now the contact band is CHARGE_CONTACT_TOL_M / _DEG (hysteresis)."""
    assert dl.CHARGE_CONTACT_TOL_M > dl.DOCK_TOL_M and dl.CHARGE_CONTACT_TOL_DEG > dl.DOCK_TOL_DEG
    p, bat, s = _session_to_charge()
    p.x += 0.038                                        # 3.8 cm: outside the gate, inside the contact band
    while s.active and p.t < 300.0:
        p.step(s.update(p.obs()))
    assert s.step == "done" and not s.fault and s.outputs["charge"]["state"] == "full"
    p, bat, s = _session_to_charge()
    p.x += dl.CHARGE_CONTACT_TOL_M + 0.005              # past the band: contact lost
    while s.active and p.t < 300.0:
        p.step(s.update(p.obs()))
    assert s.step == "assist" and s.fault == dl.REASON_CONTACT


# ---------------------------------------------------------------------- world module
@pytest.fixture(scope="module")
def demo_model():
    with env(NANDI_WORLD="orchard_demo"):
        return farm_scene.load_farm_model()


def test_world_module_sites_decals_and_registry(demo_model):
    m = demo_model
    compose.discover()
    mods = {x.name: x for x in compose.modules()}
    assert mods["dock_target"].order == 81 > mods["dock"].order
    sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "dock")
    for n in ("dock_target", "dock_staging"):
        assert sid(n) >= 0 and m.site_bodyid[sid(n)] == bid and m.site_group[sid(n)] == 5
    assert np.allclose(m.site_size[sid("dock_target")][:2], dl.DOCK_TOL_M)
    for n in ("dock_lane_l", "dock_lane_r", "dock_tag", "dock_pad_l", "dock_pad_r", "dock_tol_0"):
        g = gid(n)
        assert g >= 0 and m.geom_bodyid[g] == bid and m.geom_group[g] == 4
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    # the geometry read from the compiled model = the field's
    with open(os.path.join(compose.FIELDS_DIR, "orchard_demo.json")) as fh:
        field = json.load(fh)
    gm, gf = dl.DockGeometry.from_model(m), dl.DockGeometry.from_field(field)
    assert gm.x == pytest.approx(gf.x) and gm.y == pytest.approx(gf.y) and gm.yaw == pytest.approx(gf.yaw)
    assert (gm.entry, gm.staging_m) == (gf.entry, pytest.approx(gf.staging_m))
    # the home keyframe stands on the target: docked at the start
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    e = gm.error(d.qpos[0], d.qpos[1], 0.0)
    assert e.dist_m < 1e-6
    # world id folds the module in
    wid, _, _, _, versions = compose.identify("orchard_demo", 7, farm_scene.world_config())
    assert any(v.startswith("dock_target@1[") for v in versions)


def test_world_module_is_absent_from_legacy():
    with env(NANDI_WORLD="legacy"):
        m = farm_scene.load_farm_model()
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "dock_target") < 0
    assert dl.DockGeometry.from_model(m) is None
    _, _, _, _, versions = compose.identify("legacy", 7, farm_scene.world_config())
    assert "dock_target@1" in versions                                   # no fingerprint outside orchards


# ---------------------------------------------------------------------- real gait (SimDriver, orchard_demo)
def _driver(m, start):
    from atharv_nandi.core.sim_driver import SimDriver
    drv = SimDriver(m)
    drv.stand(0.3)
    assert drv.go_to(start)
    drv.stand(0.3)
    return drv


def _dock_run(m, start, inject_dy=0.0, n_inject=1):
    drv = _driver(m, start)
    s = dl.DockUnloadSession(dl.DockGeometry.from_model(m), battery=None, charge=False)
    r = dl.DockRunner(drv, s)
    done = {"n": 0, "keys": set(), "strafe_ticks": 0, "feet_on_plate": 0}
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{k}_foot") for k in ("FR", "FL", "MR", "ML", "RR", "RL")]
    pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "dock_plate")
    g = s.dock_geom

    def slip_obs(dv):
        a = s.dock_aligner
        if a.state == "align" and a._moving and a._move_axis == "lat":     # a strafe tick: any foot over the plate?
            done["strafe_ticks"] += 1
            c, sn = math.cos(g.yaw), math.sin(g.yaw)
            for f in feet:
                dx, dy = dv.d.geom_xpos[f][0] - g.x, dv.d.geom_xpos[f][1] - g.y
                u, v = c * dx + sn * dy, -sn * dx + c * dy
                r = m.geom_size[f][0]
                if abs(u) <= m.geom_size[pg][0] + r and abs(v) <= m.geom_size[pg][1] + r:
                    done["feet_on_plate"] += 1
                    break
        key = (len(a.attempts), a.retries)
        if (inject_dy and a.state == "entry" and a.last_error is not None and a.last_error.remaining_m < 0.25
                and done["n"] < n_inject and key not in done["keys"]):
            dv.d.qpos[1] += inject_dy                   # a scripted 6 cm side slip of the whole robot
            seat_tools(dv.m, dv.d)                      # WP2: racked tools follow a teleported trunk
            mujoco.mj_forward(dv.m, dv.d)
            done["n"] += 1
            done["keys"].add(key)
    drv.add_observer(slip_obs)
    assert r.start()
    drv.run_until(lambda _: s.step in ("done", "assist"), 150.0, chunk_s=0.1)
    return drv, s, done


def test_sim_dock_within_3cm_5deg_from_headland_starts(demo_model):
    """O-20 / MP 9 step 4 on the real tripod gait: from 2 headland starts, docked < 3 cm and < 5 deg, no retry.
    MEASURED 2026-11-17 (14 random starts 1-3 m out in the home apron): 0.2-2.3 cm, 0.2-0.6 deg."""
    t0 = time.perf_counter()
    res = []
    for start in ((2.5, -0.5), (-0.2, 1.2)):
        drv, s, _ = _dock_run(demo_model, start)
        a = s.dock_aligner
        assert s.step == "done" and a.docked, a.status()
        res.append((a.final.dist_m, a.final.yaw_deg, a.retries, a.status()["dock_s"]))
        # cross-check with the truth pose, not the aligner's own numbers
        e = s.dock_geom.error(drv.d.qpos[0], drv.d.qpos[1], drv.heading())
        assert e.dist_m < dl.DOCK_TOL_M and abs(e.yaw_deg) < dl.DOCK_TOL_DEG
        assert [m_ for m_ in drv.published_count if m_ == dl.STATE_TOPIC]
    print("dock results (dist_m, yaw_deg, retries, dock_s):", [tuple(round(v, 4) for v in r) for r in res])
    assert all(r[0] < dl.DOCK_TOL_M and abs(r[1]) < dl.DOCK_TOL_DEG and r[2] == 0 for r in res)
    assert time.perf_counter() - t0 < GAIT_WALL_S


def test_sim_retry_path_after_a_side_slip(demo_model):
    """MP 9 failure path on the real gait: a 6 cm side slip during the entry -> back off 0.3 m, re-align, dock."""
    t0 = time.perf_counter()
    drv, s, done = _dock_run(demo_model, (1.5, 0.8), inject_dy=0.06)
    a = s.dock_aligner
    print("retry:", a.status()["attempts"])
    assert done["n"] == 1 and s.step == "done" and a.docked and a.retries == 1
    first, last = a.attempts[0], a.attempts[-1]
    # the slip either leaves the entry cone (abort) or ends the entry outside 3 cm (verify fails): both back off
    assert first["event"] == "entry_abort" or (first["event"] == "verify" and not first["ok"]), a.attempts
    assert len(a.attempts) == 2 and last["event"] == "verify" and last["ok"]
    assert "backoff" in [x[1] for x in a.status()["log"]]
    assert a.final.dist_m < dl.DOCK_TOL_M and abs(a.final.yaw_deg) < dl.DOCK_TOL_DEG
    # MP 9.1: every strafe (first try and retry) with the trunk at least clear_m out, i.e. no foot on the plate
    assert a.strafe_min_remaining_m is not None and a.strafe_min_remaining_m >= s.dock_geom.clear_m
    assert done["feet_on_plate"] == 0 and done["strafe_ticks"] > 0
    assert time.perf_counter() - t0 < GAIT_WALL_S


def test_sim_estop_near_the_plate_does_not_trigger_the_platform_fill(demo_model):
    """Review finding: an e-stop during the entry (creep 0.027 m/s, under WP7's 0.05 m/s dock speed) within 0.15 m
    of home let WP7's 'parked at home, no plan' rule fill the pack before the charge step (SoC 0.40 -> 1.0, charge
    0 Wh). Now the WP13 plan is held for the whole session: no WP7 fill; the time-compressed charge does the work."""
    t0 = time.perf_counter()
    m = demo_model
    drv = _driver(m, (1.5, 0.3))
    bat = dl.platform_battery(m)
    assert bat is not None
    bat.energy_wh = 0.40 * bat.p.capacity_wh
    s = dl.DockUnloadSession(dl.DockGeometry.from_model(m), battery=bat,
                             charger=dl.TimeCompressedCharger(dl.ChargeParams(compress=2400.0)))
    r = dl.DockRunner(drv, s)
    st = {"held": None, "plat_docked": False, "soc_max": 0.0}

    def obs(dv):
        a = s.dock_aligner
        if s.step == "dock" and a.state == "entry" and a.last_error.remaining_m < 0.12 and st["held"] is None:
            dv.enter_hold("estop")
            st["held"] = dv.t
        if st["held"] is not None and dv.holding and dv.t - st["held"] > 7.0:
            dv.leave_hold()
        if s.step not in ("charge", "sync", "done"):
            st["plat_docked"] |= bool((dv.bus.values.get("platform") or {}).get("docked"))
            st["soc_max"] = max(st["soc_max"], bat.soc)
    drv.add_observer(obs)
    assert r.start()
    drv.run_until(lambda _: s.step in ("done", "assist"), 150.0, chunk_s=0.1)
    ch = s.outputs.get("charge") or {}
    print("estop near plate:", {"held_at": st["held"], "plat_docked": st["plat_docked"], "soc_max": st["soc_max"],
                                "charge": ch})
    assert st["held"] is not None and s.step == "done"
    assert not st["plat_docked"] and st["soc_max"] <= 0.401
    assert ch["state"] == "full" and ch["soc0"] < 0.41 and ch["charged_wh"] > 200.0 and bat.soc > 0.98
    assert time.perf_counter() - t0 < GAIT_WALL_S


def test_sim_charge_fault_at_home_is_not_undone_by_the_platform_fill(demo_model):
    """Review finding (real gait, every hook): a charge_contact_lost at the plate ended in assist, the runner sent
    {clear}, and 5 s later WP7's dock rule filled the pack 0.45 -> 1.0. Now the plan stays held: SoC stays put."""
    t0 = time.perf_counter()
    m = demo_model
    drv = _driver(m, (1.5, 0.3))
    bat = dl.platform_battery(m)
    bat.energy_wh = 0.40 * bat.p.capacity_wh
    s = dl.DockUnloadSession(dl.DockGeometry.from_model(m), battery=bat,
                             charger=dl.TimeCompressedCharger(dl.ChargeParams(compress=240.0)))
    r = dl.DockRunner(drv, s)
    st = {"shifted": None}

    def obs(dv):
        if s.step == "charge" and st["shifted"] is None and bat.soc > 0.45:
            dv.d.qpos[0] += 0.08                        # knocked 8 cm off the plate (still inside WP7's home radius)
            seat_tools(dv.m, dv.d)
            mujoco.mj_forward(dv.m, dv.d)
            st["shifted"] = dv.t
    drv.add_observer(obs)
    assert r.start()
    drv.run_until(lambda _: s.step in ("done", "assist"), 150.0, chunk_s=0.1)
    assert s.step == "assist" and s.fault == dl.REASON_CONTACT, s.status()
    soc = bat.soc
    drv.wait(8.0)
    plat = drv.bus.values.get("platform") or {}
    print("charge fault at home:", {"soc_end": round(soc, 4), "soc_8s": round(bat.soc, 4),
                                    "plat_docked": plat.get("docked")})
    assert bat.soc <= soc + 1e-6 and not plat.get("docked")
    assert dl.platform_runtime(m).plan is not None and drv.bus.values["dock"]["platform_plan_held"]
    r.acknowledge()                                     # operator ack: the hold goes, WP7's rule may run again
    drv.wait(0.1)
    assert dl.platform_runtime(m).plan is None and not drv.bus.values["dock"]["platform_plan_held"]
    assert time.perf_counter() - t0 < GAIT_WALL_S


# ---------------------------------------------------------------------- slow: the whole DOCK / UNLOAD
@pytest.mark.slow
def test_slow_dock_unload_low_battery_inject_under_60s(tmp_path):
    """MVP-SPEC stretch + MP 9 steps 1-6 on the orchard_demo world, every hook loaded: 3 fruit in the tray (2 mango
    OT-1, 1 papaya R3-1), low-battery inject (SoC 0.22) 1.8 m out -> WP7 reserve monitor -> the runner starts
    DOCK/UNLOAD by itself -> park alongside the crate -> WP3 unload -> job sheets (WP9 stub) -> dock < 3 cm / 5 deg ->
    time-compressed charge -> hub sync stub closes the manifest. <= 60 s wall."""
    import mujoco as mj
    from atharv_nandi.core.sim_driver import SimDriver
    from atharv_nandi.sim.hooks import transfer as ht
    w0 = time.perf_counter()
    with env(NANDI_WORLD="orchard_demo"):
        m = farm_scene.load_farm_model()
    drv = SimDriver(m)
    ctrl = ht.controller_for(m)
    pool = ctrl.pool
    drv.stand(0.5)
    pick = lambda tree, n: sorted(f for f, v in pool.fruit.items() if v["tree_id"] == tree and v["in_band"]
                                  and v["harvest_stage"] and not v["bagged"])[:n]
    ids = pick("OT-1", 2) + pick("R3-1", 1)
    for fid in ids:                                   # WP2 / WP8 hand-over at the catcher, then WP3 place
        slot = pool.assign(drv.d, fid)
        pool.release(drv.d, slot)
        mj.mj_forward(m, drv.d)
        sl = pool.slots[slot]
        drv.d.qpos[sl.qadr:sl.qadr + 3] = ctrl.L.handoff_point(drv.d) + np.array([0.0, 0.0, -0.10])
        drv.d.qvel[sl.dadr:sl.dadr + 6] = 0.0
        mj.mj_forward(m, drv.d)
        assert ctrl.place(drv.d, fid, "auto", ht.runtime_for(m).inputs(drv.d, drv.bus))
        assert drv.run_until(lambda _: not ctrl.busy(), 20.0)
    assert drv.go_to((1.8, 0.5))
    drv.stand(0.3)
    rd = _run_dir(tmp_path)
    bat = dl.platform_battery(m)
    assert bat is not None
    c, _, _ = ctrl.crate_frame(drv.d)
    s = dl.DockUnloadSession(dl.DockGeometry.from_model(m), dl.crate_park_geometry(c[:2]), battery=bat, run_dir=rd,
                             meta={"run_id": os.path.basename(rd), "device_id": "nandi-sim"}, rows_visited=["OT", "R3"])
    dl.DockRunner(drv, s, unload=ctrl, auto_low_soc=True)
    drv.wait(1.0)
    assert s.step == "idle"                           # SoC full: nothing starts by itself
    bat.energy_wh = 0.22 * bat.p.capacity_wh          # low-battery inject (below energy_model.LOW_SOC 0.25)
    t0 = drv.t
    plat_docked_in_session = []                       # WP7's instant fill stays off for the whole session
    drv.add_observer(lambda dv: plat_docked_in_session.append(bool((dv.bus.values.get("platform") or {})
                                                                   .get("docked")))
                     if s.active else None)
    assert drv.run_until(lambda _: s.step in ("done", "assist"), 170.0, chunk_s=0.1), s.status()
    wall = time.perf_counter() - w0
    st = s.status()
    out = s.outputs
    print("DOCK/UNLOAD:", json.dumps({"sim_s": round(drv.t - t0, 1), "wall_s": round(wall, 1),
                                      "crate_park": out.get("crate_park"), "dock": out["dock"]["final"],
                                      "retries": out["dock"]["retries"], "crate": out.get("crate"),
                                      "charge": out.get("charge"), "sync": out.get("sync")}))
    assert s.step == "done" and s.trigger == "energy_reserve"
    # dock < 3 cm / 5 deg (truth pose)
    e = s.dock_geom.error(drv.d.qpos[0], drv.d.qpos[1], drv.heading())
    assert e.dist_m < dl.DOCK_TOL_M and abs(e.yaw_deg) < dl.DOCK_TOL_DEG
    # crate count and kg per tree (WP3 numbers), every item delivered
    cr = out["crate"]
    assert cr["n"] == 3 and cr["dropped"] == 0
    assert set(cr["per_tree"]) == {"OT-1", "R3-1"} and cr["per_tree"]["OT-1"]["crate_n"] == 2
    # time-compressed charge: SoC 0.22 -> full over tens of sim seconds, not the platform's instant fill
    ch = out["charge"]
    assert ch["state"] == "full" and ch["soc0"] < 0.25 and bat.soc > 0.98
    assert 10.0 < ch["sim_s"] < 60.0 and ch["real_min"] > 60.0
    assert plat_docked_in_session and not any(plat_docked_in_session)
    # manifest closed and verifiable, L2 / L3 queued, no local delete without a hub receipt
    assert st["manifest_closed"] and mf.verify(rd)[0]
    assert out["sync"]["lanes"] == {"L1": "sent_stub", "L2": "queued", "L3": "queued"}
    assert out["sync"]["may_delete_local"] is False
    # console topics
    assert drv.published_count[dl.STATE_TOPIC] > 0 and drv.published_count[dl.EVENTS_TOPIC] >= 7
    assert drv.bus.values["dock"]["step"] == "done"
    assert wall <= 60.0


def test_sweep_entry_point_is_reproducible_from_the_repo():
    """Review finding: the contract's sweep came from a deleted throwaway script. ``dl.sweep`` (CLI
    ``python3 atharv_nandi/core/dock_logic.py --sweep N --world W --seed S``) re-runs it; one start here."""
    with env(NANDI_WORLD="orchard_demo"):
        rows = dl.sweep("orchard_demo", n=1, seed=3)
    assert len(rows) == 1 and set(rows[0]) >= {"start", "status"}
    r = rows[0]
    assert r["status"] in ("docked", "skipped"), r
    if r["status"] == "docked":
        assert r["dist_cm"] < 100 * dl.DOCK_TOL_M and r["yaw_deg"] < dl.DOCK_TOL_DEG
