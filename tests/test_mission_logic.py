"""Mission executor: command parsing, reject reasons, planning, pick loop, counting and capacity rules.

The pick loop runs against a toy world (unicycle trunk + scripted harvest node), so the full state
machine is exercised without ROS or MuJoCo. tools/mission_sim_check.py runs it in the MuJoCo world.
"""
import json
import os
import math
from datetime import datetime, timedelta, timezone

import pytest

from atharv_nandi.core import geo_frame as gf
from atharv_nandi.core import mission_logic as ml
from atharv_nandi.core import nav_logic as nav

ORIGIN = gf.SiteOrigin(12.5, -30.0)
UTC0 = datetime(2026, 11, 17, 6, 0, tzinfo=timezone.utc)
ROW_N = {"crop_item_2": (1.03, 0.53, 0.035), "crop_item_4": (2.5, 0.52, 0.035), "crop_item_6": (3.27, 0.44, 0.035)}
OTHERS = {"crop_item_1": (0.62, 0.0, 0.035), "crop_item_3": (1.7, -0.5, 0.035), "crop_item_5": (2.53, -0.39, 0.035)}
ZONE_N = {"local_polygon": [[0.4, 0.15], [3.9, 0.15], [3.9, 1.2], [0.4, 1.2]]}


def cmd(**kw):
    d = {"mission_id": "m1", "mode": "harvest_zone", "zone": ZONE_N,
         "not_after": (UTC0 + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")}
    d.update(kw)
    return json.dumps(d)


def obs(t=0.0, **kw):
    base = dict(t=t, utc=UTC0 + timedelta(seconds=t), pose=(0.0, 0.0, 0.0), crops={**ROW_N, **OTHERS},
                in_basket=0, in_basket_names=(), soc=1.0)
    base.update(kw)
    return ml.Observation(**base)


def controller(**kw):
    return ml.MissionController(ml.MissionConfig(origin=ORIGIN, **kw))


# ------------------------------------------------------------------ parsing + rejections
def test_command_invalid_cases():
    c = controller()
    for raw in ("not json", json.dumps([1]), cmd(mission_id=""), cmd(mode="dance"), cmd(not_after="soon"),
                cmd(max_picks=0), cmd(max_picks=1.5)):
        ok, reason, _ = c.submit(raw, obs())
        assert not ok and reason == "command_invalid" and c.state == "rejected"


def test_reject_expired():
    c = controller()
    ok, reason, _ = c.submit(cmd(not_after=(UTC0 - timedelta(seconds=1)).isoformat()), obs())
    assert (ok, reason, c.state) == (False, "expired", "rejected")
    assert c.status()["mission_id"] == "m1" and c.status()["reason_code"] == "expired"


def test_reject_zone_invalid():
    c = controller()
    for zone in ({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1]]]}, {"local_polygon": [[0, 0], [float("nan"), 1], [1, 1]]}, None):
        ok, reason, _ = c.submit(cmd(zone=zone), obs())
        assert reason == "zone_invalid"


def test_reject_zone_outside_geofence():
    fence = [(-1.0, -1.0), (3.0, -1.0), (3.0, 2.0), (-1.0, 2.0)]  # zone runs to x = 3.9
    ok, reason, _ = controller(geofence=fence).submit(cmd(), obs())
    assert reason == "zone_outside_geofence"
    ok, reason, _ = controller(geofence=[(-2, -3), (8, -3), (8, 3), (-2, 3)]).submit(cmd(), obs())
    assert ok and reason == ""


def test_reject_estopped_battery_low_basket_full():
    assert controller().submit(cmd(), obs(estopped=True))[1] == "estopped"
    assert controller().submit(cmd(), obs(soc=0.20))[1] == "battery_low"   # route + 20 % reserve (0.21 with the 100 Wh pack)
    full = obs(in_basket=6, in_basket_names=("a", "b", "c", "d", "e", "f"))
    assert controller().submit(cmd(), full)[1] == "basket_full"
    assert controller().submit(cmd(), obs(soc=0.5))[0]


def test_busy_does_not_disturb_running_mission():
    c = controller()
    assert c.submit(cmd(), obs())[0]
    c.step(obs(0.1))
    ok, reason, _ = c.submit(cmd(mission_id="m2"), obs(0.2))
    assert (ok, reason) == (False, "busy")
    assert c.mission_id == "m1" and c.state == "running"


# ------------------------------------------------------------------ planning
def test_waypoints_follow_the_row_inside_the_zone():
    zone = gf.zone_to_local(ZONE_N, ORIGIN)
    crops = {n: p[:2] for n, p in {**ROW_N, **OTHERS}.items()}
    wps, order = ml.plan_mission("harvest_zone", zone, crops, (0.0, 0.0), (0.0, 0.0))
    picks = [w for w in wps if w.kind == "pick"]
    assert order == ["crop_item_2", "crop_item_4", "crop_item_6"]   # west -> east along the row
    assert [w.crop for w in picks] == order
    for w in picks:
        assert gf.point_in_polygon(w.x, w.y, zone)                   # stops inside the zone
        cx, cy, _ = ROW_N[w.crop]
        assert ml.crop_in_pick_window((w.x, w.y, w.heading), (cx, cy))
        for n, p in OTHERS.items():                                  # lane clear of other crops
            assert math.hypot(w.x - p[0], w.y - p[1]) >= ml.CROP_CLEAR_M
    assert wps[0].kind == "transit" and abs(wps[0].x) < 0.05         # side-on entry, away from crop_item_1


def test_two_rows_boustrophedon():
    zone = [(0.4, -2.0), (3.9, -2.0), (3.9, 2.0), (0.4, 2.0)]
    crops = {f"s{i}": (float(i), -0.6) for i in (1, 2, 3)}
    crops.update({f"n{i}": (float(i), 0.6) for i in (1, 2, 3)})
    wps, order = ml.plan_mission("harvest_zone", zone, crops, (0.0, 0.0), (0.0, 0.0))
    picks = [w for w in wps if w.kind == "pick"]
    assert order == ["s1", "s2", "s3", "n3", "n2", "n1"]           # east along one row, back west on the next
    assert [round(w.heading, 2) for w in picks] == [0.0] * 3 + [round(math.pi, 2)] * 3
    assert all(gf.point_in_polygon(w.x, w.y, zone) for w in picks)
    # each lane is on the outside of its row, away from the other row
    assert all(w.y < -0.6 for w in picks[:3]) and all(w.y > 0.6 for w in picks[3:])


def test_patrol_without_crops_walks_the_zone_vertices():
    zone = [(0, 0), (2, 0), (2, 1), (0, 1)]
    wps, order = ml.plan_mission("patrol", zone, {}, (0.0, 0.0), (0.0, 0.0))
    assert order == [] and [(w.x, w.y) for w in wps] == zone


def test_capacity_rule():
    assert ml.BasketCapacity().limit == 6                  # 6 crops < 1.2 kg / 0.15 kg = 8
    assert ml.BasketCapacity(10, 1.2).limit == 8           # mass binds
    assert ml.BasketCapacity(6, 0.5).limit == 3
    cap = ml.BasketCapacity(2, 1.2)
    assert not cap.is_full(1) and cap.is_full(2)


def test_crop_source_selection_and_parsers():
    truth = {"crop_item_1": (0.62, 0.0, 0.035)}
    det = ml.parse_detections(json.dumps({"detections": [{"name": "crop_item_1", "x": 0.6, "y": 0.01}, {"x": 1}]}))
    assert det == {"crop_item_1": (0.6, 0.01, 0.035)}
    assert ml.choose_crops(truth, det, 0.4) == (det, "detections")
    assert ml.choose_crops(truth, det, 1.5) == (truth, "truth")
    assert ml.choose_crops(truth, {}, 0.1) == (truth, "truth")
    assert ml.parse_estop('{"estop": true}') and ml.parse_estop('{"state": "ESTOP"}')
    assert not ml.parse_estop('{"estop": false}') and not ml.parse_estop("garbage")


def test_geofence_sources():
    site = {"fallback_geofence": {"x_min": -2, "x_max": 8, "y_min": -3, "y_max": 3}}
    fence, src = ml.geofence_from_config(None, site)
    assert src == "site_sim.yaml" and len(fence) == 4
    fence, src = ml.geofence_from_config({"geofence": {"local_polygon": [[0, 0], [1, 0], [1, 1]]}}, site)
    assert src == "safety.yaml" and fence == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert ml.geofence_from_config({}, {}) == (None, "none")


# ------------------------------------------------------------------ toy world closed loop
class ToyWorld:
    """Unicycle trunk, crops as points, a harvest node that takes the nearest crop."""

    def __init__(self, crops, succeed=lambda name, n: True, harvest_s=8.0, soc_drain=0.0):
        self.pose = (0.0, 0.0, 0.0)
        self.crops = dict(crops)
        self.basket = []
        self.succeed, self.harvest_s, self.soc_drain = succeed, harvest_s, soc_drain
        self.t, self.soc = 0.0, 1.0
        self.harvest_until, self.harvest_target = None, None
        self.calls, self.n_harvest, self.walking = [], 0, False
        self.estop, self.max_speed = False, 0.0

    def observe(self):
        mode = "ik_harvest" if self.harvest_until is not None else "rl_balance"
        return obs(self.t, pose=self.pose, crops=dict(self.crops), in_basket=len(self.basket),
                   in_basket_names=tuple(self.basket), harvest_mode=mode, walking=self.walking,
                   trunk_speed=0.0, soc=self.soc, estopped=self.estop)

    def run(self, c, until=lambda w: False, max_t=900.0, dt=0.1):
        while self.t < max_t:
            act = c.step(self.observe())
            self.calls += act.calls
            if act.cmd is not None:
                self.pose = nav.unicycle_step(self.pose, act.cmd[0], act.cmd[1], dt)
                self.walking = True
            else:
                self.walking = False
            if "auto_harvest" in act.calls:
                name = ml.predicted_harvest_target(self.pose, self.crops, self.basket)
                c.on_harvest_reply(name is not None, f"Harvesting {name}" if name else "No crop in reach")
                if name:
                    self.n_harvest += 1
                    self.harvest_until, self.harvest_target = self.t + self.harvest_s, name
            if "release_arm" in act.calls:
                self.harvest_until = None
            if self.harvest_until is not None and self.t >= self.harvest_until:
                if self.succeed(self.harvest_target, self.n_harvest):
                    self.basket.append(self.harvest_target)
                self.harvest_until = None
            self.soc -= self.soc_drain * dt
            self.t += dt
            if c.state in ("done", "failed", "aborted") or until(self):
                c.step(self.observe())  # exit actions
                return


def test_full_mission_picks_row_and_returns_home():
    w = ToyWorld({**ROW_N, **OTHERS})
    c = controller()
    assert c.submit(cmd(), w.observe())[0]
    w.run(c)
    st = c.status()
    assert (st["state"], st["reason_code"]) == ("done", "completed")
    assert st["picked"] == 3 and st["attempted"] == 3 and st["basket_count"] == 3
    assert sorted(w.basket) == sorted(ROW_N)
    assert math.hypot(w.pose[0], w.pose[1]) <= nav.ARRIVE_TOL_M and st["progress"] == 1.0


def test_picked_counts_only_game_status_increase_and_retries_once():
    # harvest "succeeds" by its own report but the crop never reaches the basket
    w = ToyWorld({**ROW_N, **OTHERS}, succeed=lambda n, k: False)
    c = controller()
    c.submit(cmd(), w.observe())
    w.run(c)
    st = c.status()
    assert st["picked"] == 0 and st["attempted"] == 6          # 3 crops x (1 try + 1 retry)
    assert all(v.startswith("skipped") for v in st["results"].values()), (st["results"], c.events)
    assert st["state"] == "done"


def test_retry_success_counts_once():
    w = ToyWorld({**ROW_N, **OTHERS}, succeed=lambda n, k: k % 2 == 0)  # every 2nd attempt lands
    c = controller()
    c.submit(cmd(), w.observe())
    w.run(c)
    assert c.picked == 3 and c.attempted == 6


def test_basket_full_returns_home_then_done():
    w = ToyWorld({**ROW_N, **OTHERS})
    c = controller(capacity=ml.BasketCapacity(max_crops=2))
    c.submit(cmd(), w.observe())
    seen = []
    w.run(c, until=lambda world: seen.append(c.state) and False)
    assert "returning" in seen
    assert (c.state, c.reason, c.picked) == ("done", "basket_full", 2)
    assert math.hypot(w.pose[0], w.pose[1]) <= nav.ARRIVE_TOL_M


def test_low_battery_triggers_return():
    w = ToyWorld({**ROW_N, **OTHERS}, soc_drain=0.02)  # 2 %/s: crosses 25 % after ~38 s
    c = controller()
    c.submit(cmd(), w.observe())
    w.run(c)
    assert (c.state, c.reason) == ("done", "battery_low") and c.picked < 3


def test_max_picks():
    w = ToyWorld({**ROW_N, **OTHERS})
    c = controller()
    c.submit(cmd(max_picks=1), w.observe())
    w.run(c)
    assert (c.state, c.reason, c.picked) == ("done", "max_picks", 1)


def test_abort_mid_walk_publishes_zero_and_stops():
    w = ToyWorld({**ROW_N, **OTHERS})
    c = controller()
    c.submit(cmd(), w.observe())
    w.run(c, until=lambda world: world.t > 5.0)
    assert c.state == "running" and w.walking
    ok, _ = c.abort()
    act = c.step(w.observe())
    assert ok and c.state == "aborted" and c.reason == "aborted"
    assert act.cmd is None and act.force_release and {"stop_motion", "release_arm"} <= set(act.calls)
    assert c.step(w.observe()).cmd is None
    assert not c.abort()[0]


def test_estop_mid_mission_fails_within_one_tick():
    w = ToyWorld({**ROW_N, **OTHERS})
    c = controller()
    c.submit(cmd(), w.observe())
    w.run(c, until=lambda world: world.t > 3.0)
    w.estop = True
    act = c.step(w.observe())
    assert (c.state, c.reason) == ("failed", "estopped")
    assert act.cmd is None and act.force_release and "stop_motion" in act.calls


def test_harvest_timeout_releases_arm_and_skips_after_retry():
    w = ToyWorld({**ROW_N, **OTHERS}, harvest_s=1e9)  # arm never finishes
    c = controller(harvest_timeout_s=5.0)
    c.submit(cmd(), w.observe())
    w.run(c, max_t=400)
    assert w.calls.count("release_arm") >= 6, (c.results, c.events)  # 2 timeouts per crop
    assert c.results["crop_item_2"] == "skipped:harvest_timeout"


def test_status_contract_keys():
    c = controller()
    st = c.status()
    for k in ("mission_id", "state", "reason_code", "progress", "waypoint_idx", "target_crop", "attempted",
              "picked", "basket_count", "basket_capacity", "soc", "crop_source"):
        assert k in st
    assert st["state"] == "idle" and 0.0 <= st["progress"] <= 1.0
    json.loads(c.status_json())


def test_ridge_crop_is_approached_from_the_level_side():
    from atharv_nandi.core.farm_scene import RIDGES
    keep_out = ml.ridge_keep_out(RIDGES, "rough")
    assert ml.ridge_keep_out(RIDGES, "flat") == []
    zone = gf.zone_to_local(ZONE_N, ORIGIN)
    crops = {n: p[:2] for n, p in {**ROW_N, **OTHERS}.items()}
    wps, order = ml.plan_mission("harvest_zone", zone, crops, (0.0, 0.0), (0.0, 0.0), keep_out)
    picks = {w.crop: w for w in wps if w.kind == "pick"}
    for w in picks.values():
        assert not any(lo <= w.x <= hi for lo, hi in keep_out)          # every stop stands level
        assert ml.crop_in_pick_window((w.x, w.y, w.heading), crops[w.crop])
    # crop_item_4 (x 2.5) sits just past ridge_2: picked facing west on the way back, after crop_item_6
    assert picks["crop_item_4"].heading == pytest.approx(math.pi) and order[-1] == "crop_item_4"
    assert len({round(w.y, 3) for w in picks.values()}) == 1             # one straight lane per row


def test_harvest_of_a_nearer_crop_is_attributed_by_name():
    c = controller()
    w = ToyWorld({**ROW_N, **OTHERS})
    c.submit(cmd(), w.observe())
    w.run(c, until=lambda world: c.phase == "harvesting")
    w.harvest_target = "crop_item_4"  # the harvest node takes a different crop than the stop's target
    w.run(c, until=lambda world: "crop_item_4" in c.results)
    assert c.picked == 1 and c.results["crop_item_4"] == "picked"
    assert "crop_item_2" not in c.results and c.attempts["crop_item_2"] == 0  # attempt not spent
    w.run(c)
    assert (c.state, c.picked, c.attempted) == ("done", 3, 3)
    assert set(c.results.values()) <= {"picked", "in_basket"}


# ------------------------------------------------------------------ review fixes
def test_non_finite_max_picks_is_command_invalid_not_a_crash():
    for raw in ('NaN', 'Infinity', '-Infinity', '1e400'):
        text = cmd()[:-1] + f', "max_picks": {raw}}}'
        c = controller()
        ok, reason, _ = c.submit(text, obs())
        assert (ok, reason, c.state) == (False, "command_invalid", "rejected")


def test_parser_crash_inside_check_is_command_invalid(monkeypatch):
    def boom(*a, **k):
        raise OverflowError("boom")
    monkeypatch.setattr(ml, "plan_mission", boom)
    c = controller()
    assert c.submit(cmd(), obs())[:2] == (False, "command_invalid")


def test_iso8601_extended_forms():
    ref = datetime(2030, 1, 1, tzinfo=timezone.utc)
    for text, delta in (("2030-01-01T00:00:00.5Z", timedelta(microseconds=500000)),
                        ("2030-01-01T00:00:00.123456789Z", timedelta(microseconds=123456)),
                        ("2030-01-01T05:30:00+0530", timedelta(0)), ("2030-01-01T05:30:00+05:30", timedelta(0)),
                        ("2030-01-01T05:00:00+05", timedelta(0)), ("2030-01-01 00:00Z", timedelta(0)),
                        ("2030-01-01T00:00:00,25z", timedelta(microseconds=250000)),
                        ("2030-01-01T00:00:00", timedelta(0)), ("2030-01-01", timedelta(0))):
        assert ml.parse_iso8601(text) == ref + delta, text
    for bad in ("soon", "2030-13-01T00:00:00Z", "2030-01-01T00:00:00+5", ""):
        with pytest.raises(ValueError):
            ml.parse_iso8601(bad)
    assert controller().submit(cmd(not_after="2030-01-01T00:00:00.5Z"), obs())[0]


def test_planned_path_must_stay_inside_the_geofence():
    # zone inside the fence, but the north lane (y ~0.91) would leave it: the south lane is used
    zone = {"local_polygon": [[0.4, 0.15], [3.9, 0.15], [3.9, 0.69], [0.4, 0.69]]}
    fence = [(-2.0, -3.0), (8.0, -3.0), (8.0, 0.7), (-2.0, 0.7)]
    c = controller(geofence=fence)
    ok, reason, detail = c.submit(cmd(zone=zone), obs())
    assert ok, (reason, detail)
    assert all(gf.point_in_polygon(w.x, w.y, fence) for w in c.waypoints)
    # a fence too tight for either lane: rejected, not accepted with lanes outside it
    tight = [(-2.0, 0.1), (8.0, 0.1), (8.0, 0.7), (-2.0, 0.7)]
    c = controller(geofence=tight)
    ok, reason, detail = c.submit(cmd(zone=zone), obs(pose=(0.0, 0.3, 0.0)))
    assert (ok, reason) == (False, "zone_outside_geofence") and "planned" in detail
    # the robot itself outside the fence
    ok, reason, detail = controller(geofence=tight).submit(cmd(zone=zone), obs())
    assert (ok, reason) == (False, "zone_outside_geofence") and "robot" in detail


def test_geofence_breach_during_mission_fails_and_stops():
    fence = [(-2.0, -3.0), (8.0, -3.0), (8.0, 3.0), (-2.0, 3.0)]
    w = ToyWorld({**ROW_N, **OTHERS})
    c = controller(geofence=fence)
    assert c.submit(cmd(), w.observe())[0]
    w.run(c, until=lambda world: world.t > 3.0)
    w.pose = (0.0, 3.5, 0.0)
    act = c.step(w.observe())
    assert (c.state, c.reason) == ("failed", "geofence_breach")
    assert act.cmd is None and act.force_release and "stop_motion" in act.calls


def test_locomotion_mode_rl_rejected_and_fails_running_mission():
    c = controller()
    ok, reason, _ = c.submit(cmd(), obs(loco_mode="rl"))
    assert (ok, reason) == (False, "locomotion_mode")
    assert controller().submit(cmd(), obs(loco_mode=""))[1] == "locomotion_mode"
    c = controller()
    assert c.submit(cmd(), obs())[0]
    c.step(obs(0.1))
    act = c.step(obs(0.2, loco_mode="rl"))
    assert (c.state, c.reason) == ("failed", "locomotion_mode") and act.cmd is None and act.force_release


def test_external_stop_aborts_and_releases():
    w = ToyWorld({**ROW_N, **OTHERS})
    c = controller()
    c.submit(cmd(), w.observe())
    w.run(c, until=lambda world: world.t > 5.0)
    assert c.state == "running" and w.walking
    assert c.external_stop("operator stop (/nandi/stop_motion)")
    act = c.step(w.observe())
    assert (c.state, c.reason) == ("aborted", "aborted") and "operator stop" in c.detail
    assert act.cmd is None and act.force_release and {"stop_motion", "release_arm"} <= set(act.calls)
    assert c.step(w.observe()).cmd is None
    assert not c.external_stop("again")


def test_busy_rejection_answers_the_sender_with_its_own_status():
    c = controller()
    assert c.submit(cmd(), obs())[0]
    c.step(obs(0.1))
    assert c.submit(cmd(mission_id="m2"), obs(0.2))[1] == "busy"
    assert len(c.replies) == 1
    r = c.replies[0]
    assert (r["mission_id"], r["state"], r["reason_code"], r["active_mission_id"]) == ("m2", "rejected", "busy", "m1")
    assert set(c.status()) <= set(r)
    assert c.status()["mission_id"] == "m1" and c.state == "running"
    # no sim state yet (idle): an ordinary rejection of its own
    c2 = controller()
    assert c2.submit(cmd(), obs(pose=None))[1] == "busy" and c2.state == "rejected" and c2.mission_id == "m1"


def test_arm_path_out_of_reach_retries_from_a_closer_stop():
    c = controller()
    w = ToyWorld({**ROW_N, **OTHERS})
    c.submit(cmd(), w.observe())
    w.run(c, until=lambda world: c.phase == "harvesting")
    before = c.waypoints[c.wp_idx]
    w.harvest_until = None
    c.on_harvest_reply(False, "Arm path out of reach (0.9 cm off)")
    c.step(w.observe())
    after = c.waypoints[c.wp_idx]
    assert c.phase == "walking" and after.crop == before.crop
    cx, cy, _ = ROW_N[after.crop]
    assert abs(after.y - cy) == pytest.approx(ml.CLOSER_LATERAL_M)


def test_attempt_on_another_crop_is_charged_to_that_crop():
    # stop for crop_a, but crop_b (a later target) is nearer and inside the pick window
    c = controller()
    pose = (0.0, 0.0, 0.0)
    crops = {"crop_a": (0.5, 0.45, 0.035), "crop_b": (0.45, 0.0, 0.035)}
    c.targets = ["crop_a", "crop_b"]
    c.waypoints = [ml.Waypoint(0.0, 0.0, "pick", "crop_a", 0.0), ml.Waypoint(0.0, -0.4, "pick", "crop_b", 0.0)]
    c.state, c.mode, c.mission_id = "running", "harvest_zone", "m"
    c._set_phase("settling", 0.0)
    o = obs(1.0, pose=pose, crops=crops)
    assert ml.predicted_harvest_target(pose, crops, ()) == "crop_b"
    act = c.step(o)
    assert act.calls == ["auto_harvest"] and c.attempts == {"crop_b": 1}
    c.on_harvest_reply(True, "Harvesting crop_b")
    c.step(obs(2.0, pose=pose, crops=crops, harvest_mode="ik_harvest"))
    c.step(obs(10.0, pose=pose, crops=crops, harvest_result="Drop missed: crop_b not in basket"))
    c.step(obs(12.0, pose=pose, crops=crops))
    assert c.attempts.get("crop_a", 0) == 0 and c.attempts["crop_b"] == 1
    assert "crop_b" not in c.results and c.phase == "settling"
    # crop_b was tried from this stop already: the stop now slides instead of trying crop_b again
    act = c.step(obs(13.0, pose=pose, crops=crops))
    assert act.calls == [] and c.phase == "walking" and c.waypoints[0].x != 0.0


def test_crop_behind_the_trunk_is_not_harvested_from_another_stop():
    c = controller()
    pose = (0.0, 0.0, 0.0)
    crops = {"crop_a": (0.5, 0.45, 0.035), "crop_b": (-0.3, 0.45, 0.035)}  # crop_b behind, nearer
    c.targets = ["crop_a", "crop_b"]
    c.waypoints = [ml.Waypoint(0.0, 0.0, "pick", "crop_a", 0.0)]
    c.state, c.mode, c.mission_id = "running", "harvest_zone", "m"
    c._set_phase("settling", 0.0)
    assert ml.predicted_harvest_target(pose, crops, ()) == "crop_b"
    act = c.step(obs(1.0, pose=pose, crops=crops))
    assert "auto_harvest" not in act.calls and c.phase == "walking" and c.attempts == {}


def test_unblock_stop_for_the_crop_6_stop_of_the_default_field():
    # the crop_item_6 stop (x 2.82, lane y 0.91, facing east) has crop_item_4 nearer, behind the trunk
    crops = {**ROW_N, **OTHERS}
    w = ml.Waypoint(2.82, 0.91, "pick", "crop_item_6", 0.0)
    assert ml.predicted_harvest_target((w.x, w.y, 0.0), crops, ()) == "crop_item_4"
    alt = ml.unblock_stop(w, crops, ())
    pose = (alt.x, alt.y, alt.heading)
    assert alt.crop == "crop_item_6" and alt.y == w.y
    assert ml.predicted_harvest_target(pose, crops, ()) == "crop_item_6"
    assert ml.crop_in_pick_window(pose, crops["crop_item_6"][:2])
    # no stop works when every candidate is outside the fence
    assert ml.unblock_stop(w, crops, (), fence=[(-1, -1), (2, -1), (2, 2), (-1, 2)]) is None


def test_console_start_command_is_accepted():
    # what web_bridge publishes for "Start zone A" (console_protocol.mission_command)
    with open(os.path.join(os.path.dirname(__file__), "..", "config", "missions", "zone_a.geojson")) as fh:
        geo = json.load(fh)
    raw = json.dumps({"action": "start", "zone": "zone_a", "source": "config/missions/zone_a.geojson", "mission": geo})
    c = controller()
    ok, reason, detail = c.submit(raw, obs())
    assert ok, (reason, detail)
    assert c.mission_id.startswith("console-zone_a-") and c.mode == "harvest_zone"
    assert c.not_after == UTC0 + timedelta(seconds=ml.CONSOLE_TTL_S)
    st = c.status()
    assert st["basket"] == {"n": 0, "capacity": 6}
    assert controller().submit(json.dumps({"action": "stop"}), obs())[1] == "command_invalid"


def test_safety_yaml_polygon_form_and_disabled():
    site = {"fallback_geofence": {"x_min": -2, "x_max": 8, "y_min": -3, "y_max": 3}}
    poly = [[-1.5, -2.0], [7.0, -2.0], [7.0, 2.0], [-1.5, 2.0]]
    fence, src = ml.geofence_from_config({"geofence": {"enabled": True, "polygon": poly}}, site)
    assert src == "safety.yaml" and fence == [tuple(p) for p in poly]
    fence, src = ml.geofence_from_config({"geofence": {"enabled": False, "polygon": poly}}, site)
    assert src == "site_sim.yaml"
    from atharv_nandi.core.geo_frame import load_flat_yaml
    cfg = load_flat_yaml(os.path.join(os.path.dirname(__file__), "..", "config", "safety.yaml"))
    fence, src = ml.geofence_from_config(cfg, site)
    assert src == "safety.yaml" and len(fence) >= 3
