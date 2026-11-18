"""WP29 WORK assist (MISSION-PRESETS 7, ORCHARD-MVP-SPEC 5.6): tote, lead-to-marks, tag follow-me, spray disabled.

Acceptance (WP29): lead-to-marks reaches 100 % of MARK trees and stops within 0.3 m; follow 1.5-3 m,
tag lost > 2 s -> stop, untagged person -> stop; no track coordinates in any uplink fixture; spray refused
``not_in_mvp``; <= 10 s.

Two tiers: pure logic on a kinematic unicycle plant (ROS-free, no MuJoCo), then the real tripod gait on
``orchard_lite`` through ``SimDriver`` with the ``follow_tag`` hook and the WP6 ``stability`` hook (fields live).
"""
import contextlib
import json
import math
import os
import random
import time

import pytest

from atharv_nandi.core import work_assist as wa

LITE = os.path.join(os.path.dirname(__file__), "..", "config", "fields", "orchard_lite.json")


def _field():
    with open(LITE, encoding="utf-8") as fh:
        return json.load(fh)


MARKS = [{"tree_id": t, "physical_tag": t, "kind": k, "reason": r, "clock": c, "side": "S", "h_class": h,
          "tool_hint": th}
         for t, k, r, c, h, th in (("L1-1", "prune", "over_capacity", 3, "beech", "lopper"),
                                   ("L1-2", "fruit", "out_of_reach_marked", 11, "upar", "pole_picker"),
                                   ("L2-2", "rootstock_sprout", "sucker", 6, "neeche", "sickle"),
                                   ("L2-3", "skirt_lift", "skirt", 9, "neeche", "saw"))]


# --------------------------------------------------------------------------------------------- kinematic plant
class Plant:
    """Unicycle plant at 20 Hz with a first-order speed lag (0.3 s) and a coast on release, like the gait."""

    def __init__(self, x=0.0, y=0.0, yaw=0.0, dt=0.05):
        self.p = wa.Pose(x, y, yaw)
        self.v = 0.0
        self.w = 0.0
        self.dt = dt
        self.t = 0.0

    def step(self, d: wa.Drive):
        a = min(1.0, self.dt / 0.3)
        tv, tw = (0.0, 0.0) if d.stop else (min(d.v_mps, wa.V_CAP_MPS), d.yaw_rps)
        self.v += a * (tv - self.v)
        self.w += a * (tw - self.w)
        self.p = wa.Pose(self.p.x + self.v * math.cos(self.p.yaw) * self.dt,
                         self.p.y + self.v * math.sin(self.p.yaw) * self.dt, wa._wrap(self.p.yaw + self.w * self.dt))
        self.t += self.dt


class Holder:
    """A tag holder walking a polyline at a constant speed; UWB reading at 10 Hz with noise."""

    def __init__(self, path, speed, seed=3):
        self.path = [tuple(p) for p in path]
        self.speed = speed
        self.rng = random.Random(seed)
        self.drop = (-1.0, -1.0)
        self.last = None
        self.next_t = 0.0

    def xy(self, t):
        s = t * self.speed
        for a, b in zip(self.path, self.path[1:]):
            L = math.dist(a, b)
            if s <= L:
                f = s / L if L else 1.0
                return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            s -= L
        return self.path[-1]

    def reading(self, t, pose):
        if t + 1e-9 >= self.next_t:
            self.next_t = t + 0.1
            if not (self.drop[0] <= t < self.drop[1]):
                hx, hy = self.xy(t)
                r = math.hypot(hx - pose.x, hy - pose.y)
                b = wa._wrap(math.atan2(hy - pose.y, hx - pose.x) - pose.yaw)
                self.last = wa.TagReading(t, r + self.rng.gauss(0, 0.05), b + self.rng.gauss(0, math.radians(2)))
        return self.last

    def base_xy(self, t, pose):
        hx, hy = self.xy(t)
        dx, dy = hx - pose.x, hy - pose.y
        c, s = math.cos(-pose.yaw), math.sin(-pose.yaw)
        return c * dx - s * dy, s * dx + c * dy


def _obs(plant, **kw):
    return wa.Obs(t=plant.t, pose=plant.p, **kw)


def _issued(tags=None):
    tags = tags or wa.TagRegistry("test")
    ok, tid = tags.issue("worker_1", "worker", True, "s1")
    assert ok and tid.startswith("tag-")
    return tags


def _run_lead(field, marks, start=(0.0, 0.0, 0.0), auto_next_s=1.0, hits_fn=None, max_s=400.0):
    w = wa.WorkAssist(field)
    pl = Plant(*start)
    ok, why = w.start("lead_to_marks", {"marks": marks, "auto_next_s": auto_next_s}, _obs(pl))
    assert ok, why
    log = []
    while pl.t < max_s and w.phase not in ("done", "aborted"):
        hits = hits_fn(pl) if hits_fn else []
        d = w.update(_obs(pl, hits=hits))
        log.append((pl.t, d.stop, d.reason))
        if w.phase == "at_stop":
            w.settle_error(pl.p)
        pl.step(d)
    return w, pl, log


# --------------------------------------------------------------------------------------------- planning
def test_plan_lead_orders_rows_and_skips_unknown_trees():
    f = _field()
    stops, skipped = wa.plan_lead(f, MARKS + [{"tree_id": "NOPE-9", "kind": "fruit"}], (0.0, 0.0))
    assert [s.tree_id for s in stops] == ["L1-1", "L1-2", "L2-2", "L2-3"]
    assert skipped == [{"tree_id": "NOPE-9", "reason": "registry_mismatch"}]
    for s in stops:
        tx, ty = wa._tree_xy(wa.tree_index(f)[s.tree_id])
        assert math.hypot(s.xy[0] - tx, s.xy[1] - ty) == pytest.approx(wa.LEAD_STOP_M)
    assert stops[0].jobs[0] == {"kind": "prune", "reason": "over_capacity", "clock": 3, "side": "S",
                                "h_class": "beech", "tool_hint": "lopper"}
    for s in stops:
        assert wa.contains_coordinates(s.jobs) == []


def test_route_goes_round_rows_never_through_them():
    f = _field()
    # from the south alley of row L1 (y 1.0) to the north alley of L1 (y 4.0): row L1 (y 2.5, x 2..4+) is between
    wps = wa.route(f, (3.0, 1.0), (3.0, 4.0))
    assert len(wps) == 3
    x0, x1, _ = wa._row_extent(f, "L1")
    assert wps[0][0] <= x0 - wa.HEADLAND_M + 1e-9 or wps[0][0] >= x1 + wa.HEADLAND_M - 1e-9
    assert wa.route(f, (0.0, 0.0), (5.0, 1.0)) == [(5.0, 1.0)]


# --------------------------------------------------------------------------------------------- lead-to-marks
def test_lead_to_marks_reaches_every_mark_and_stops_within_0_3_m():
    w, pl, _ = _run_lead(_field(), MARKS)
    k = w.kpis()
    assert w.phase == "done"
    assert k["marks_reached"] == k["marks_total"] == 4 and k["reached_pct"] == 100.0
    assert k["stop_err_max_m"] <= wa.LEAD_STOP_TOL_M
    assert k["time_per_job_s"] > 0
    st = w.state(pl.t)
    assert st["mode"] == "lead_to_marks" and st["lead"]["stops"] == 4


def test_lead_waits_at_stop_for_next_and_shows_the_job_sheet_line():
    w = wa.WorkAssist(_field())
    pl = Plant()
    assert w.start("lead_to_marks", {"marks": MARKS[:2]}, _obs(pl))[0]
    while w.phase != "at_stop" and pl.t < 200:
        pl.step(w.update(_obs(pl)))
    st = w.state(pl.t)
    assert st["phase"] == "at_stop" and st["lead"]["tree_id"] == "L1-1"
    assert st["lead"]["jobs"][0]["kind"] == "prune"
    for _ in range(100):                                  # no auto_next: stays
        d = w.update(_obs(pl))
        assert d.stop and d.reason == "at_stop"
        pl.step(d)
    assert w.command("next", _obs(pl)) == (True, "next")
    assert w.phase == "running" and w.stop_k == 1


def test_lead_stops_for_anyone_else_but_not_for_the_follower_behind():
    in_front = {"on": False}

    def hits(pl):
        out = [wa.FieldHit(-1.2, 0.0, "warning", "obstacle", "rear")]          # the scout following
        if in_front["on"]:
            out.append(wa.FieldHit(1.8, 0.1, "warning", "obstacle", "front"))
        return out

    w = wa.WorkAssist(_field())
    pl = Plant()
    assert w.start("lead_to_marks", {"marks": MARKS[:1], "auto_next_s": 1.0}, _obs(pl))[0]
    for _ in range(60):
        d = w.update(_obs(pl, hits=hits(pl)))
        pl.step(d)
    assert w.hold_reason in ("",) and not w.last_drive.stop       # the follower behind does not stop the lead
    in_front["on"] = True
    d = w.update(_obs(pl, hits=hits(pl)))
    assert d.stop and d.reason == "person_hold"
    in_front["on"] = False
    for _ in range(int(wa.PERSON_CLEAR_S / pl.dt) - 2):
        d = w.update(_obs(pl, hits=hits(pl)))
        assert d.stop and d.reason == "person_hold"
        pl.step(d)
    for _ in range(5):
        d = w.update(_obs(pl, hits=hits(pl)))
        pl.step(d)
    assert not d.stop
    # a protective return behind (the follower walked into the robot) is a stop
    d = w.update(_obs(pl, hits=[wa.FieldHit(-0.4, 0.0, "protective", "obstacle", "rear")]))
    assert d.stop and d.reason == "person_hold"
    assert w.kpis()["person_hold_count"] == {"h0": 2}


def test_lead_refusals():
    w = wa.WorkAssist(_field())
    pl = Plant()
    assert w.start("lead_to_marks", {"marks": []}, _obs(pl)) == (False, ["no_job_sheet"])
    assert w.start("lead_to_marks", {"marks": [{"tree_id": "X"}]}, _obs(pl)) == (False, ["registry_mismatch"])
    assert w.start("lead_to_marks", {"marks": MARKS}, _obs(pl, crown_ext_m=0.4)) == (False, ["crown_gate"])
    assert wa.WorkAssist(None).start("lead_to_marks", {"marks": MARKS}, _obs(pl)) == (False, ["registry_mismatch"])
    fence = [(-1.0, -1.0), (5.0, -1.0), (5.0, 3.0), (-1.0, 3.0)]          # L2 stops are outside
    assert w.start("lead_to_marks", {"marks": MARKS}, _obs(pl, geofence=fence)) == (False, ["geofence"])


# --------------------------------------------------------------------------------------------- follow-me
def _follow(holder, max_s, tags=None, hits_fn=None, pl=None, w=None, start=True):
    tags = tags or _issued()
    w = w or wa.WorkAssist(_field(), tags)
    pl = pl or Plant()
    if start:
        ok, why = w.start("follow_me", {}, _obs(pl, tag=holder.reading(pl.t, pl.p)))
        assert ok, why
    log = []
    t_end = pl.t + max_s
    while pl.t < t_end:
        r = holder.reading(pl.t, pl.p)
        d = w.update(_obs(pl, tag=r, hits=hits_fn(pl) if hits_fn else []))
        hx, hy = holder.xy(pl.t)
        log.append((pl.t, math.hypot(hx - pl.p.x, hy - pl.p.y), d.stop, d.reason))
        pl.step(d)
    return w, pl, log


def test_follow_keeps_1_5_to_3_m():
    h = Holder([(2.3, 0.0), (20.0, 0.0), (20.0, 6.0)], 0.15)
    w, pl, log = _follow(h, 150.0)
    rng = [r for t, r, _s, _ in log if t > 5.0]
    assert min(rng) >= wa.FOLLOW_MIN_M - 0.05 and max(rng) <= wa.FOLLOW_MAX_M
    k = w.kpis()
    assert k["in_band_pct"] >= 95.0 and k["protective_violations"] == 0
    assert k["follow_lag_s"] is not None


def test_follow_never_closes_inside_1_5_m_when_the_holder_stops():
    h = Holder([(2.3, 0.0), (6.0, 0.0)], 0.15)          # stops at x 6
    w, pl, log = _follow(h, 60.0)
    assert min(r for _t, r, _s, _ in log) >= wa.FOLLOW_MIN_M - 0.05
    assert log[-1][2]                                   # standing


def test_tag_lost_over_2_s_stops_and_latches():
    h = Holder([(2.3, 0.0), (30.0, 0.0)], 0.15)
    w, pl, _ = _follow(h, 15.0)
    t_drop = pl.t
    t_seen = h.last.t                                     # the last reading before the drop
    h.drop = (t_drop, t_drop + 4.0)
    stop_t = lost_t = None
    while pl.t < t_drop + 3.0:
        d = w.update(_obs(pl, tag=h.reading(pl.t, pl.p)))
        if d.stop and stop_t is None:
            stop_t = pl.t
        if w.latched_reason == "tag_lost" and lost_t is None:
            lost_t = pl.t
        pl.step(d)
    assert stop_t - t_drop <= wa.TAG_STALE_S + 0.15          # stops at once (waits for the tag) ...
    assert wa.TAG_LOST_S < lost_t - t_seen <= wa.TAG_LOST_S + 0.1   # ... latches tag_lost > 2 s after the last fix
    assert w.command("resume", _obs(pl, tag=h.reading(pl.t, pl.p)))[0] is False   # no tag yet
    while pl.t < t_drop + 4.5:
        d = w.update(_obs(pl, tag=h.reading(pl.t, pl.p)))
        assert d.stop and d.reason == "tag_lost"             # the tag is back: still latched until resume
        pl.step(d)
    assert w.command("resume", _obs(pl, tag=h.reading(pl.t, pl.p))) == (True, "resumed")
    for _ in range(40):
        d = w.update(_obs(pl, tag=h.reading(pl.t, pl.p)))
        pl.step(d)
    assert not d.stop
    assert any(e["event"] == "tag_lost" for e in w.events)


def test_tag_dropout_under_2_s_pauses_without_latch():
    h = Holder([(2.3, 0.0), (30.0, 0.0)], 0.15)
    w, pl, _ = _follow(h, 15.0)
    h.drop = (pl.t, pl.t + 1.2)
    w, pl, log = _follow(h, 6.0, tags=w.tags, w=w, pl=pl, start=False)
    assert w.latched_reason == "" and any(r == "tag_stale" for *_x, r in log)
    assert not log[-1][2]


def test_tag_lost_latches_even_while_someone_else_holds_the_robot():
    """The holder without a fresh reading is 'someone else' to the field check: the tag rule must still run."""
    h = Holder([(2.3, 0.0), (30.0, 0.0)], 0.15)
    w, pl, _ = _follow(h, 10.0)
    t0 = pl.t
    h.drop = (t0, t0 + 10.0)
    while pl.t < t0 + 2.3:
        hits = [wa.FieldHit(*h.base_xy(pl.t, pl.p), "warning", "obstacle", "front")]
        pl.step(w.update(_obs(pl, tag=h.reading(pl.t, pl.p), hits=hits)))
    assert w.latched_reason == "tag_lost"


def test_untagged_person_stops_follow_and_the_holder_does_not():
    h = Holder([(2.3, 0.0), (30.0, 0.0)], 0.15)
    other = {"on": False}

    def hits(pl):
        out = [wa.FieldHit(*h.base_xy(pl.t, pl.p), "warning", "obstacle", "front")]   # the holder, always seen
        if other["on"]:
            out.append(wa.FieldHit(1.0, -0.8, "warning", "obstacle", "side"))
        return out

    w, pl, log = _follow(h, 10.0, hits_fn=hits)
    assert not log[-1][2]
    other["on"] = True
    w, pl, log = _follow(h, 2.0, tags=w.tags, w=w, pl=pl, start=False, hits_fn=hits)
    assert all(s and r == "person_hold" for _t, _r, s, r in log)
    other["on"] = False
    w, pl, log = _follow(h, 4.0, tags=w.tags, w=w, pl=pl, start=False, hits_fn=hits)
    assert not log[-1][2]
    assert w.kpis()["person_hold_count"] == {"h0": 1}
    # an animal anywhere in a field stops it too
    d = w.update(_obs(pl, tag=h.reading(pl.t, pl.p), hits=[wa.FieldHit(2.0, 0.0, "warning", "animal", "front")]))
    assert d.stop and d.reason == "animal_hold"


def test_follow_needs_an_issued_adult_tag():
    pl = Plant()
    w = wa.WorkAssist(_field())
    assert w.start("follow_me", {}, _obs(pl)) == (False, ["tag_not_issued"])
    tags = wa.TagRegistry("x")
    assert tags.issue("kid", "child", True) == (False, "tag_not_adult")
    assert tags.issue("w", "worker", False) == (False, "tag_not_adult")        # no adult attestation
    assert not tags.issued
    tags = _issued()
    w = wa.WorkAssist(_field(), tags)
    assert w.start("follow_me", {}, _obs(pl)) == (False, ["tag_stale"])      # issued but not seen


def test_tag_ids_are_pseudonymous_and_rotate_per_shift():
    tags = _issued()
    a = tags.tag_id
    assert "worker" not in a
    b = tags.rotate("s2")
    assert a != b and tags.issued
    assert tags.rotate("s1") == a


# --------------------------------------------------------------------------------------------- tote
def test_tote_carry_limits_and_arrival():
    f = _field()
    pl = Plant()
    w = wa.WorkAssist(f)
    assert w.start("tote_carry", {"tote_kg": 3.5, "dest": {"tree_id": "L1-2"}}, _obs(pl)) == (False, ["payload_full"])
    assert w.start("tote_carry", {"dest": {"tree_id": "L1-2"}}, _obs(pl)) == (False, ["tote_missing"])
    assert w.start("tote_carry", {"tote_kg": 2.0, "dest": {"tree_id": "ZZ"}}, _obs(pl)) == (False, ["unknown_target"])
    ok, _ = w.start("tote_carry", {"tote_kg": 3.0, "dest": {"tree_id": "L2-3"}, "return": True},
                    _obs(pl, soc=0.80))
    assert ok
    soc = 0.80
    while w.phase not in ("done",) and pl.t < 400:
        soc -= 1e-5
        d = w.update(_obs(pl, soc=soc))
        if w.phase == "delivered":
            assert w.settle_error(pl.p) <= wa.TOTE_ARRIVE_M
            assert w.command("ack", _obs(pl, soc=soc)) == (True, "acked")
        pl.step(d)
    k = w.kpis()
    assert w.phase == "done" and k["kg_moved"] == 3.0
    assert k["arrival_err_m"] <= wa.TOTE_ARRIVE_M
    assert k["wh"] > 0 and k["kg_per_wh"] == pytest.approx(3.0 / k["wh"], rel=1e-2)
    assert math.hypot(pl.p.x, pl.p.y) <= wa.TOTE_ARRIVE_M          # back at the origin
    # the route never cut through a row: the robot stayed in the alleys / headlands
    assert w.delivery["returning"]


# --------------------------------------------------------------------------------------------- spray, safety
def test_spray_is_refused_not_in_mvp():
    w = wa.WorkAssist(_field(), _issued())
    pl = Plant()
    assert w.start("spot_spray", {}, _obs(pl)) == (False, ["not_in_mvp"])
    assert w.command("spray", _obs(pl)) == (False, "not_in_mvp")
    assert w.state(0.0)["spray"] == {"enabled": False, "reason": "not_in_mvp"}
    assert [e["reasons"] for e in w.events if e["event"] == "refused"] == [["not_in_mvp"]] * 2


def test_safety_stops_and_latches():
    w, pl, _ = _run_lead(_field(), MARKS[:1], max_s=3.0)
    assert w.active
    for kw, reason in ((dict(holding=True), "estop"), (dict(latched="protective"), "protective"),
                       (dict(local_link_ok=False), "local_link_lost"), (dict(crown_ext_m=0.3), "crown_gate")):
        d = w.update(_obs(pl, **kw))
        assert d.stop and d.reason == reason
    # e-stop / protective latch stay latched below this layer clearing: needs an operator resume
    d = w.update(_obs(pl))
    assert d.stop and d.reason == "estop"
    assert w.command("resume", _obs(pl, holding=True))[0] is False
    assert w.command("resume", _obs(pl)) == (True, "resumed")
    assert not w.update(_obs(pl)).stop


def test_drive_respects_the_speed_cap():
    d = wa.Drive(0.5, 0.0, False)
    assert d.gait()[0] == pytest.approx(min(1.0, wa.V_CAP_MPS / wa.V_PER_UNIT))
    w = wa.WorkAssist(_field())
    pl = Plant()
    assert w.start("lead_to_marks", {"marks": MARKS, "v_mps": 0.9}, _obs(pl))[0]
    assert w.params["v_mps"] == wa.V_CAP_MPS


# --------------------------------------------------------------------------------------------- privacy
def _all_payloads(w, t):
    return [w.state(t), w.uplink_record("run-1", "nandi-sim-01"), *w.events]


def test_no_track_coordinates_or_tag_id_in_any_uplink_fixture():
    from atharv_nandi.core.kis_mapper import KisMapper
    tags = _issued()
    h = Holder([(2.3, 0.0), (20.0, 0.0)], 0.15)
    w, pl, _ = _follow(h, 40.0, tags=tags)
    print("WP29 device track", w.track.summary())
    assert len(w.track) == 400 and 5.0 <= w.track.summary()["holder_walk_m"] <= 7.0   # on the device (truth 6 m)
    lead, pl2, _ = _run_lead(_field(), MARKS)
    km = KisMapper("sim-nandi", "orchard-lite", "nandi-sim-01", 1)
    fixtures = []
    for x, t in ((w, pl.t), (lead, pl2.t)):
        for p in _all_payloads(x, t):
            assert wa.contains_coordinates(p) == [], p
            assert tags.tag_id not in json.dumps(p)
            assert "worker_1" not in json.dumps(p)
        fixtures.append(km.mission(1_700_000_000_000, x.uplink_record("run-1", "nandi-sim-01")))
    for topic, payload in fixtures:
        assert topic.endswith("/tele/mission")
        assert wa.contains_coordinates(payload) == []
        assert payload["personal_data"] == "device_only"
        assert tags.tag_id not in json.dumps(payload)
    # positive control: the detector does flag coordinates
    assert wa.contains_coordinates({"a": {"xy": [1.0, 2.0]}}) == ["a.xy", "a.xy"]
    assert wa.contains_coordinates({"pts": [[1.0, 2.0]]}) == ["pts[0]"]


def test_track_log_is_device_only_and_purged_after_7_days():
    tl = wa.TrackLog()
    for i in range(10):
        tl.append(i * 3600.0, float(i), 0.0)
    assert len(tl) == 10
    tl.append(7 * 86400.0 + 5 * 3600.0 + 1.0, 10.0, 0.0)
    assert len(tl) == 5                                     # h0..h4 older than 7 days: purged
    assert set(tl.summary()) == {"samples", "span_s", "holder_walk_m"}


def test_track_log_purges_without_new_fixes_and_documents_its_cap():
    tl = wa.TrackLog(retention_s=10.0)
    tl.append(0.0, 0.0, 0.0)
    tl.append(1.0, 1.0, 0.0)
    assert tl.summary(now=5.0)["samples"] == 2
    assert tl.summary(now=100.0)["samples"] == 0           # purged with no append
    assert wa.TrackLog()._pts.maxlen == wa.TRACK_MAX_N
    # the controller purges every tick, also while no tag fix arrives
    w = wa.WorkAssist(_field(), _issued())
    w.track = wa.TrackLog(retention_s=10.0)
    w.track.append(0.0, 1.0, 1.0)
    w.update(wa.Obs(t=60.0, pose=wa.Pose(0.0, 0.0, 0.0)))
    assert len(w.track) == 0


# --------------------------------------------------------------------------------------------- fixer regressions
def test_malformed_start_params_are_bad_command_never_raise():
    f = _field()
    ob = wa.Obs(t=0.0, pose=wa.Pose(0.0, 0.0, 0.0))
    for marks in (5, 3.2, True, "L1-1", {"tree_id": "L1-1"}, [5], ["L1-1"]):
        w = wa.WorkAssist(f)
        assert w.start("lead_to_marks", {"marks": marks}, ob) == (False, ["bad_command"]), marks
        assert w.phase == "idle"
    w = wa.WorkAssist(f)
    assert w.start("tote_carry", {"tote_kg": 1, "dest": {"xy": [1, 2]}, "return": "false"}, ob) == \
        (False, ["bad_command"])
    assert w.start("tote_carry", {"tote_kg": 1, "dest": 5}, ob) == (False, ["bad_command"])
    assert w.start("tote_carry", 5, ob) == (False, ["bad_command"])
    assert w.start("tote_carry", {"tote_kg": 1, "dest": {"xy": [1, 2]}, "return": False}, ob) == (True, [])


def test_non_finite_or_out_of_range_tag_readings_never_drive():
    nan, inf = float("nan"), float("inf")
    assert wa.Drive(nan, 0.0, False, "walking").gait() == (0.0, 0.0, 0.0)
    d = wa.Drive(0.1, nan, False, "walking")
    assert d.stop and d.as_dict()["cmd"] == [0.0, 0.0, 0.0]
    for bad in ((nan, nan), (inf, 0.0), (2.0, inf), (-0.5, 0.0), (25.0, 0.0)):
        tags = _issued()
        w = wa.WorkAssist(_field(), tags)
        pose = wa.Pose(0.0, 0.0, 0.0)
        assert w.start("follow_me", {}, wa.Obs(t=0.0, pose=pose, tag=wa.TagReading(0.0, *bad))) == \
            (False, ["tag_stale"]), bad
        assert w.start("follow_me", {}, wa.Obs(t=0.0, pose=pose, tag=wa.TagReading(0.0, 2.5, 0.0)))[0]
        t = 0.0
        for _ in range(10):
            t += 0.05
            dr = w.update(wa.Obs(t=t, pose=pose, tag=wa.TagReading(t, *bad)))
            assert dr.stop and all(math.isfinite(c) for c in dr.gait()), (bad, dr)
            assert json.dumps(dr.as_dict(), allow_nan=False)


def test_fields_unavailable_fails_closed():
    """No (or stale) WP6 returns: nobody can be seen, so no mode starts and a running one stops."""
    pose = wa.Pose(0.0, 0.0, 0.0)
    tag = wa.TagReading(0.0, 2.5, 0.0)
    w = wa.WorkAssist(_field(), _issued())
    ok, why = w.start("follow_me", {}, wa.Obs(t=0.0, pose=pose, tag=tag, fields_ok=False))
    assert not ok and "fields_unavailable" in why
    w2 = wa.WorkAssist(_field())
    assert "fields_unavailable" in w2.start("lead_to_marks", {"marks": MARKS},
                                            wa.Obs(t=0.0, pose=pose, fields_ok=False))[1]
    assert w.start("follow_me", {}, wa.Obs(t=0.0, pose=pose, tag=tag))[0]
    d = w.update(wa.Obs(t=0.05, pose=pose, tag=wa.TagReading(0.05, 2.5, 0.0), fields_ok=False))
    assert d.stop and d.reason == "fields_unavailable"
    assert w.command("resume", wa.Obs(t=0.1, pose=pose, tag=tag, fields_ok=False))[0] is False
    k = w.kpis()
    assert k["false_stops_per_h"] > 0 and k["stops_by_reason"] == {"fields_unavailable": 1}


def test_a_second_person_beside_the_holder_or_in_the_protective_field_stops_follow():
    pose = wa.Pose(0.0, 0.0, 0.0)

    def run(hits, n=30):
        w = wa.WorkAssist(_field(), _issued())
        assert w.start("follow_me", {}, wa.Obs(t=0.0, pose=pose, tag=wa.TagReading(0.0, 2.5, 0.0)))[0]
        t, d = 0.0, None
        for _ in range(n):
            t += 0.05
            d = w.update(wa.Obs(t=t, pose=pose, tag=wa.TagReading(t, 2.5, 0.0), hits=hits))
        return d

    holder = wa.FieldHit(2.5, 0.0, "warning")
    assert not run([holder]).stop                                     # the holder alone: follows
    d = run([holder, wa.FieldHit(2.4, 0.5, "warning")])               # someone at the holder's side
    assert d.stop and d.reason == "person_hold"
    d = run([wa.FieldHit(2.5, 0.35, "protective")])                   # protective: never the holder
    assert d.stop and d.reason == "person_hold"


# --------------------------------------------------------------------------------------------- real gait (MuJoCo)
mujoco = pytest.importorskip("mujoco")


@contextlib.contextmanager
def world_env(**env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_TYPES = None


def _add_actor(spec, atype, aid, masked=False):
    """A person actor body (mocap). ``masked``: its geoms carry a WP6 map prefix, so the WP6 fields do not see it:
    the stand-in for the WP6 tag-holder exemption (change request, the WP29 contract notes)."""
    from atharv_nandi.sim.world import actors as wact
    types = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
             "box": mujoco.mjtGeom.mjGEOM_BOX, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}
    cls = wact.actor_class(atype)
    body = spec.worldbody.add_body(name=f"actor_{cls}_{aid}", pos=[-30.0, -30.0, 0.0], mocap=True)
    counts = {}
    for g in wact.geoms_for(atype):
        k = counts.get(g["role"], 0)
        counts[g["role"]] = k + 1
        size = list(g["size"]) + [0.0] * (3 - len(g["size"]))
        name = f"prop_wp29_{aid}_{g['role']}_{k}" if masked else f"actor_{cls}_{aid}__{g['role']}_{k}"
        kw = dict(name=name, type=types[g["type"]], size=size, rgba=g["rgba"], contype=0, conaffinity=0, group=1,
                  mass=0.001)
        if "fromto" in g:
            kw["fromto"] = g["fromto"]
        else:
            kw["pos"] = g["pos"]
        body.add_geom(**kw)


@pytest.fixture(scope="module")
def lite_model():
    from atharv_nandi.core import farm_scene
    with world_env(NANDI_WORLD="orchard_lite", NANDI_TERRAIN="flat", NANDI_ACTORS="0"):
        w = farm_scene.build_world()
        _add_actor(w.spec, "worker", "tw")                 # tag holder, seen by WP6
        _add_actor(w.spec, "worker", "tx", masked=True)    # tag holder, WP6-exempt stand-in
        _add_actor(w.spec, "adult", "to")                  # someone else
        _add_actor(w.spec, "child", "tc")
        _add_actor(w.spec, "dog", "td")                    # a non-person body (animal)
        return w.spec.compile()


def _drv(m):
    import atharv_nandi.sim.hooks as hr
    from atharv_nandi.core import sim_driver as sd
    from atharv_nandi.sim.hooks import follow_tag as ft
    drv = sd.SimDriver(m, hooks=hr.load_all(names=["mast", "stability", "platform", "follow_tag"]))
    drv.add_task("work_assist_drive", wa.driver_task, 50)
    ft._reset(m, None)
    return drv, ft.runtime_for(m)


def _mocap(m, name):
    return int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)])


def _events(drv):
    return [json.loads(p) for _, tp, p in drv.published if tp == "/nandi/work/events"]


def _published(drv, prefix="/nandi/work/"):
    return [(tp, json.loads(p)) for _, tp, p in drv.published if tp.startswith(prefix)]


def test_gait_lead_to_marks_orchard_lite(lite_model):
    W = time.time()
    drv, rt = _drv(lite_model)
    drv.stand(0.5)
    drv.deliver("/nandi/work/cmd", {"op": "start", "mode": "lead_to_marks",
                                    "params": {"marks": MARKS, "auto_next_s": 3.0}})
    while drv.t < 120.0 and rt.assist.phase not in ("done", "aborted"):
        drv.wait(0.5)
    k = rt.assist.kpis()
    errs = [v.get("stop_err_m") for v in rt.assist.visited]
    print("WP29 lead gait", json.dumps(k), errs, "sim %.1f s wall %.1f s" % (drv.t, time.time() - W))
    assert rt.assist.phase == "done" and k["reached_pct"] == 100.0 and k["marks_reached"] == 4
    assert all(e is not None and e <= wa.LEAD_STOP_TOL_M for e in errs)
    assert drv.bus.values["safety_fields"].get("latched") is False
    for tp, p in _published(drv):
        assert wa.contains_coordinates(p) == [] or tp == "/nandi/work/drive", (tp, p)
    assert time.time() - W <= 10.0 + 5.0       # budget 10 s (+5 s slack for a busy runner slot)


def _follow_gait(m, holder, speed, walk_s, other=False):
    drv, rt = _drv(m)
    mid = _mocap(m, f"actor_person_{holder}")
    drv.d.mocap_pos[mid] = (2.3, 0.0, 0.0)
    drv.stand(0.5)
    drv.deliver("/nandi/work/tag/cmd", {"op": "issue", "actor": holder, "type": "worker", "adult": True, "shift": "s1"})
    drv.wait(0.3)
    drv.deliver("/nandi/work/cmd", {"op": "start", "mode": "follow_me", "params": {}})
    drv.wait(0.1)
    assert rt.assist.phase == "running", _events(drv)
    drv.deliver("/nandi/work/tag/cmd", {"op": "walk", "waypoints": [[2.3 + speed * walk_s, 0.0]],
                                        "speed_mps": speed})
    rng = []
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"actor_person_{holder}")
    t_end = drv.t + walk_s
    while drv.t < t_end:
        drv.wait(0.25)
        if drv.t > 6.0:
            x, y = drv.trunk_xy()
            rng.append(math.hypot(drv.d.xpos[hb][0] - x, drv.d.xpos[hb][1] - y))
    return drv, rt, rng


def test_gait_follow_me_band_tag_lost_and_untagged_person(lite_model):
    """Holder ``tx`` (WP6-exempt stand-in): the follow loop holds 1.5-3 m at 0.15 m/s; tag drop -> stop at once,
    ``tag_lost`` latched after 2 s; an untagged adult steps in -> ``person_hold``."""
    m = lite_model
    W = time.time()
    drv, rt, rng = _follow_gait(m, "tx", 0.15, 30.0)
    k = rt.assist.kpis()
    print("WP29 follow gait (exempt holder)", "true range %.2f..%.2f" % (min(rng), max(rng)), json.dumps(k))
    assert wa.FOLLOW_MIN_M <= min(rng) and max(rng) <= wa.FOLLOW_MAX_M
    assert k["protective_violations"] == 0
    assert drv.bus.values.get("work_tag_holder", {}).get("r_m") == wa.TAG_ASSOC_M
    # untagged adult walks into the warning field from the side
    om = _mocap(m, "actor_person_to")
    x, y = drv.trunk_xy()
    drv.d.mocap_pos[om] = (x + 1.2, y - 0.6, 0.0)
    drv.wait(0.6)
    assert rt.assist.hold_reason == "person_hold", (rt.assist.hold_reason, drv.bus.values.get("safety_fields"))
    x1 = drv.trunk_xy().copy()
    drv.wait(1.0)
    assert math.dist(drv.trunk_xy(), x1) < 0.03
    drv.d.mocap_pos[om] = (-30.0, -30.0, 0.0)
    drv.wait(wa.PERSON_CLEAR_S + 0.6)
    assert rt.assist.hold_reason in ("", "at_gap"), rt.assist.hold_reason
    # the holder walks on; the robot follows again; then the tag drops out while both walk
    drv.deliver("/nandi/work/tag/cmd", {"op": "walk", "waypoints": [[x + 6.0, 0.0]], "speed_mps": 0.15})
    drv.wait(3.0)
    assert not rt.assist.last_drive.stop, rt.assist.hold_reason
    drv.deliver("/nandi/work/tag/cmd", {"op": "drop", "dur_s": 5.0})
    t_seen = rt.assist.tag_seen_t                       # the last fix before the drop
    drv.wait(0.05)
    t_drop = drv.t
    t_lost = None
    while drv.t < t_drop + 3.0:
        drv.wait(0.05)
        if t_lost is None and rt.assist.latched_reason == "tag_lost":
            t_lost = drv.t
    x2 = drv.trunk_xy().copy()
    drv.wait(1.0)
    print("WP29 tag lost %.2f s after the last fix" % (t_lost - t_seen), "wall %.1f s" % (time.time() - W))
    assert wa.TAG_LOST_S < t_lost - t_seen <= wa.TAG_LOST_S + 0.15
    assert math.dist(drv.trunk_xy(), x2) < 0.03
    assert rt.assist.last_drive.stop and drv.bus.values["work_drive"]["stop"]
    assert "work_tag_holder" not in drv.bus.values
    for tp, p in _published(drv):
        if tp != "/nandi/work/drive":
            assert wa.contains_coordinates(p) == [], (tp, p)
        assert rt.tags.tag_id not in json.dumps(p)
    assert time.time() - W <= 15.0


def test_gait_follow_me_with_wp6_seeing_the_holder(lite_model):
    """Today's WP6 (no tag exemption): the holder inside the 2.5 m front warning field caps the robot at 30 %, so
    the follow settles at the warning-field edge. Recorded, and the robot still never closes inside 1.5 m."""
    drv, rt, rng = _follow_gait(lite_model, "tw", 0.15, 30.0)
    k = rt.assist.kpis()
    in_band = sum(wa.FOLLOW_MIN_M <= r <= wa.FOLLOW_MAX_M for r in rng) / len(rng)
    print("WP29 follow gait (WP6 sees holder)", "true range %.2f..%.2f in band %.0f %%" %
          (min(rng), max(rng), 100 * in_band), json.dumps(k))
    assert min(rng) >= wa.FOLLOW_MIN_M
    assert max(rng) <= wa.FOLLOW_MAX_M + 0.5
    assert k["protective_violations"] == 0


def test_gait_tag_refused_for_a_child_and_spray_refused(lite_model):
    drv, rt = _drv(lite_model)
    drv.deliver("/nandi/work/tag/cmd", {"op": "issue", "actor": "tc", "type": "child", "adult": True})
    drv.deliver("/nandi/work/cmd", {"op": "start", "mode": "spot_spray", "params": {}})
    drv.deliver("/nandi/work/cmd", {"op": "spray"})            # two commands in one tick: both handled
    drv.wait(0.1)
    ev = _events(drv)
    assert {"event": "tag_refused", "reason": "tag_not_adult"}.items() <= next(
        e for e in ev if e["event"] == "tag_refused").items()
    assert any(e["event"] == "refused" and e["reasons"] == ["not_in_mvp"] for e in ev)
    assert any(e["event"] == "command" and e["op"] == "spray" and e["msg"] == "not_in_mvp" for e in ev)
    assert not rt.tags.issued


def test_gait_tote_carry_round_trip(lite_model):
    drv, rt = _drv(lite_model)
    drv.stand(0.5)
    drv.deliver("/nandi/work/cmd", {"op": "start", "mode": "tote_carry",
                                    "params": {"tote_kg": 2.5, "dest": {"tree_id": "L1-2"}, "return": True}})
    while drv.t < 80.0 and rt.assist.phase not in ("done", "aborted"):
        drv.wait(0.5)
        if rt.assist.phase == "delivered" and rt.assist.delivery.get("stop_err_m") is not None:
            drv.deliver("/nandi/work/cmd", {"op": "ack"})
    k = rt.assist.kpis()
    print("WP29 tote gait", json.dumps(k), rt.assist.delivery)
    assert rt.assist.phase == "done" and k["kg_moved"] == 2.5
    assert k["arrival_err_m"] <= wa.TOTE_ARRIVE_M and rt.assist.delivery["return_err_m"] <= wa.TOTE_ARRIVE_M
    assert k["wh"] is not None and k["wh"] > 0


def test_gait_malformed_marks_answer_bad_command_and_work_stays_alive(lite_model):
    """A non-list ``params.marks`` used to raise out of the hook (hook_fault -> safe hold, WORK dead)."""
    drv, rt = _drv(lite_model)
    drv.stand(0.5)
    for marks in (5, 3.2, True):
        drv.deliver("/nandi/work/cmd", {"op": "start", "mode": "lead_to_marks", "params": {"marks": marks}})
        drv.wait(0.1)
    assert "follow_tag" not in drv.hook_faults, drv.hook_faults
    assert (drv.bus.values.get("hold") or {}).get("reason") != "hook_fault"
    ev = _events(drv)
    assert sum(e["event"] == "bad_command" for e in ev) == 3, ev
    drv.deliver("/nandi/work/cmd", {"op": "start", "mode": "tote_carry",
                                    "params": {"tote_kg": 1.0, "dest": {"tree_id": "L1-1"}}})
    drv.wait(0.3)
    assert rt.assist.phase == "running", _events(drv)
    # a NaN walk for the holder is refused (it would write NaN into the mocap pose)
    drv.deliver("/nandi/work/tag/cmd", {"op": "issue", "actor": "tw", "type": "worker", "adult": True})
    drv.deliver("/nandi/work/tag/cmd", {"op": "walk", "waypoints": [["nan", "nan"]], "speed_mps": 0.15})
    drv.wait(0.1)
    assert any(e["event"] == "bad_command" and e.get("why") == "waypoints" for e in _events(drv))
    assert all(math.isfinite(c) for c in drv.d.mocap_pos[_mocap(lite_model, "actor_person_tw")])


def test_gait_tag_refused_for_a_non_person_body(lite_model):
    drv, rt = _drv(lite_model)
    for aid in ("actor_animal_td", "actor_animal_td"):
        drv.deliver("/nandi/work/tag/cmd", {"op": "issue", "actor": aid, "type": "worker", "adult": True})
    drv.wait(0.1)
    assert not rt.tags.issued and rt.holder_body() is None
    assert [e["reason"] for e in _events(drv) if e["event"] == "tag_refused"] == ["tag_not_adult"] * 2
    # the full person body name is the same actor as the short id
    drv.deliver("/nandi/work/tag/cmd", {"op": "issue", "actor": "actor_person_tw", "type": "worker", "adult": True})
    drv.wait(0.1)
    assert rt.tags.issued and rt.holder_body() == "actor_person_tw"


def test_gait_follow_refused_and_stopped_without_wp6_fields(lite_model):
    """Without the stability hook (no field returns) FOLLOW-ME must not start: person detection fails closed."""
    import atharv_nandi.sim.hooks as hr
    from atharv_nandi.core import sim_driver as sd
    from atharv_nandi.sim.hooks import follow_tag as ft
    m = lite_model
    drv = sd.SimDriver(m, hooks=hr.load_all(names=["mast", "platform", "follow_tag"]))
    drv.add_task("work_assist_drive", wa.driver_task, 50)
    ft._reset(m, None)
    rt = ft.runtime_for(m)
    drv.d.mocap_pos[_mocap(m, "actor_person_tw")] = (2.6, 0.0, 0.0)
    drv.d.mocap_pos[_mocap(m, "actor_person_to")] = (1.0, -0.4, 0.0)       # untagged adult right in front
    drv.stand(0.5)
    drv.deliver("/nandi/work/tag/cmd", {"op": "issue", "actor": "tw", "type": "worker", "adult": True})
    drv.wait(0.3)
    x0 = drv.trunk_xy().copy()
    drv.deliver("/nandi/work/cmd", {"op": "start", "mode": "follow_me", "params": {}})
    drv.wait(3.0)
    assert rt.assist.phase == "idle"
    assert any(e["event"] == "refused" and "fields_unavailable" in e["reasons"] for e in _events(drv))
    assert math.dist(drv.trunk_xy(), x0) < 0.05
    drv.d.mocap_pos[_mocap(m, "actor_person_to")] = (-30.0, -30.0, 0.0)
