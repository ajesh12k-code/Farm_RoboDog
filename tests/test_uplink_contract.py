"""WP9 uplink contract (MVP-SPEC 5.9, OS 5.4 / 5.6): no uplink payload with an image, binary, personal key or person
coordinates; leads never leave as findings; next_actions from the enum; sim-* tenants only; 16 KiB per payload."""
import json

import pytest

from atharv_nandi.core import kis_mapper as km
from atharv_nandi.core import records as rec
from atharv_nandi.core.tasks import common as tc

IDENT = rec.identity("run-up-1", "nandi-sim", "nandi-sim-20261118T0900Z")
ENTRY = {"tree_id": "R1-2", "physical_tag": "M-2", "block_id": "B1-mango-hdp", "row_id": "R1", "crop": "mango",
         "cultivar": "Dashehari", "form": "grafted", "geo": {"frame": "nandi_local_v1", "x": 4.5, "y": 2.5}}
TREE = {"tree_id": "R1-2", "physical_tag": "M-2", "pos": (4.5, 2.5)}
TS = 1_794_000_000_000


def full_record():
    b = rec.TreeRecordBuilder(IDENT, ENTRY)
    b.merge_tree360({"coverage": {"mode": "two_sided", "sides": ["-y", "+y"], "masked_pct": 0.4},
                     "media": {"manifest_root": "cd" * 32, "keyframe_paths": [f"k/{i}.jpg" for i in range(24)]},
                     "next_actions": ["deep_scan"], "detector": ["simtruth"], "assisted": ["truth_radius"]})
    b.merge_perception({"symptoms": [{"class": "hopper", "band": "mid", "clock": 2, "h_class": "beech", "h_m": 1.4,
                                      "unit": "per_panicle", "count": 3, "score": 0.8, "stage": "lead"}],
                        "leaf": {"status": "ok", "n": 6, "underside_n": 3,
                                 "findings": [{"class": "mealybug", "stage": "lead", "score": 0.7}],
                                 "revisit_leaves": ["R1-2:leaf:3"]},
                        "fruit": {"visible_n": 12, "reachable_n": 4, "est_kg": 2.6,
                                  "harvest_window": {"from": "2026-06-17", "to": "2026-07-07", "basis": "calendar"}}},
                       "leaf_scan")
    b.add_marks([tc.make_mark(TREE, "fruit", (4.8, 2.2, 1.6), "out_of_reach_marked", stage="mature",
                              tool_hint="pole_picker", target_id="R1-2:f04")])
    return b.build()


def up():
    return rec.Uplink(device="nandi-sim")


def test_every_stream_payload_is_clean_and_enveloped():
    u = up()
    r = full_record()
    m = tc.make_mark(TREE, "skirt_lift", (4.6, 2.0, 0.5), "over_capacity", diameter_mm=18.0, tool_hint="lopper",
                     target_id="R1-2:prune:2")
    h = rec.harvest_record_v2(tc.harvest_record("R1-2", "R1-2:f01", "mango", "clip", "picked", pedicel_mm=20.1),
                              IDENT, ENTRY, t=12.0)
    msgs = [u.tree(TS, r), u.mark(TS, m), u.harvest_v2(TS, h),
            u.mission_summary(TS, {"schema": rec.MISSION_SCHEMA, **IDENT, "pass": True, "sim_s": 900.0}),
            u.kpi(TS, {"run_id": IDENT["run_id"], "device_id": "nandi-sim", "date": "2026-11-18", "rows": 3}),
            u.alert(TS, {"id": "stuck", "t": 30.0, "reason": "sink", "attempt": 1, "armed": True})]
    for topic, payload in msgs:
        assert topic.startswith("kis/v1/sim-nandi/sim-orchard/nandi-sim/tele/")
        assert rec.contract_violations(payload) == [], (topic, rec.contract_violations(payload))
        assert payload["data_provenance"] == "simulated" and {"v", "ts", "ts_src", "boot", "seq"} <= set(payload)
        assert km.check_size(payload, km.MAX_TELE_BYTES, topic) <= km.MAX_TELE_BYTES
    tree = msgs[0][1]
    assert "keyframe_paths" not in tree["media"] and tree["media"]["keyframes_n"] == 24
    assert "findings" not in tree["leaf"] and tree["leaf"]["leads_n"] == 1 and "geo" not in tree
    assert all(s["stage"] == "lead" for s in tree["symptoms"])


def test_person_events_leave_only_as_hourly_counts_without_coordinates():
    u = up()
    for t in (10.0, 50.0, 3700.0):
        assert u.alert(TS, {"id": "person_hold", "t": t, "x": 3.2, "y": 4.1, "range_m": 1.2, "bearing": 0.4}) is None
    assert u.alert(TS, {"id": "orbit_pause", "t": 20.0, "xy": [1, 2]}) is None
    agg = u.person_aggregates(TS)
    assert [p["count"] for _, p in agg] == [3, 1] and [p["hour"] for _, p in agg] == [0, 1]
    for _, p in agg:
        assert rec.contract_violations(p, person_event=True) == []
        assert not ({"x", "y", "xy", "t_sim", "range_m", "bearing"} & set(p))


def test_alert_ids_map_to_the_kis_v1_set():
    u = up()
    for src, want in (("protective_stop", "obstacle_stop"), ("animal_hold", "obstacle_stop"), ("rtk_lost", "rtk_lost"),
                      ("low_battery_return", "low_battery_return"), ("local_link_lost_hold", "local_link_lost_hold"),
                      ("cloud_link_lost_hold", "cloud_link_lost_hold"), ("strike", "crown_retract"),
                      ("sink", "stuck"), ("fallen", "stability_stop"), ("geofence", "geofence"),
                      ("orbit_blocked", "orbit_blocked"), ("tool_fault", "tool_fault")):
        _, p = u.alert(TS, {"id": src, "t": 1.0})
        assert p["id"] == want and p["id"] in rec.UPLINK_ALERT_IDS
    assert u.alert(TS, {"id": "gust", "t": 1.0}) is None          # not an uplink alert (local toast only)


@pytest.mark.parametrize("body,rule", [
    ({"image": "x"}, "image_key"), ({"jpeg_b64": "x"}, "image_key"), ({"raw": b"\xff\xd8"}, "binary"),
    ({"note": "data:image/jpeg;base64,/9j/4AAQ"}, "binary"), ({"worker_id": "w1"}, "personal_key"),
    ({"name": "Sita"}, "personal_key"), ({"phone": "9"}, "personal_key"), ({"person_track": []}, "personal_key"),
    ({"findings": ["anthracnose"]}, "advice_key"), ({"treatment": "copper"}, "advice_key"),
    ({"symptoms": [{"class": "x", "stage": "confirmed"}]}, "lead_not_lead"),
    ({"next_actions": ["spray_now"]}, "next_action"), ({"data_provenance": "measured"}, "provenance"),
    ({"tenant": "kisaanu"}, "tenant"),
])
def test_each_forbidden_payload_is_refused(body, rule):
    u = up()
    with pytest.raises(ValueError):
        u.mission_summary(TS, body)
    assert any(r.startswith(rule) for r in u.refused[-1][1])


def test_mark_with_a_bad_field_is_refused_and_seq_has_no_gap():
    u = up()
    m = tc.make_mark(TREE, "fruit", (4.8, 2.2, 1.6), "out_of_reach_marked", tool_hint="pole_picker")
    _, p0 = u.mark(TS, m)
    with pytest.raises(ValueError):
        u.mark(TS, dict(m, kind="diagnosis"))
    _, p1 = u.mark(TS, m)
    assert p1["seq"] == p0["seq"] + 1


def test_non_sim_tenant_refused():
    with pytest.raises(km.TenantRefused):
        rec.Uplink(tenant="acme-farms")


def test_numpy_scalars_are_numbers_not_binary():
    import numpy as np
    assert rec.contract_violations({"pct": np.float64(93.2), "n": np.int64(3)}) == []
    assert rec.contract_violations({"img": np.zeros(3)})            # an array is refused (key and value)
    assert rec.plain({"a": np.float32(1.5), "b": (np.int64(2),), "c": float("nan")}) == {"a": 1.5, "b": [2],
                                                                                          "c": None}
    json.dumps(rec.plain({"a": np.arange(3)}))
