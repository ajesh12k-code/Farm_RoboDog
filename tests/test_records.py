"""WP9 records (MP 0.6, OS 5.6, O-22): TreeRecord <= 16 KiB with no image or personal keys, leads stay leads,
next_actions only from the enum, MarkRecord and HarvestRecord v2 checks. ROS-free, MuJoCo-free."""
import json

import pytest

from atharv_nandi.core import records as rec
from atharv_nandi.core.tasks import common as tc

IDENT = rec.identity("20261118T090000Z-tour_lite-7", "nandi-sim", "nandi-sim-20261118T0900Z")
ENTRY = {"tree_id": "OT-1", "physical_tag": "OT-1", "block_id": "B0-mango-isolated", "row_id": "OT", "crop": "mango",
         "cultivar": "Dashehari", "form": "grafted", "geo": {"frame": "nandi_local_v1", "x": 6.0, "y": -3.5}}
TREE = {"tree_id": "OT-1", "physical_tag": "OT-1", "pos": (6.0, -3.5)}


def lead(cls="mealybug", unit="trunk", band="low", clock=7, score=0.8, stage="lead"):
    return {"class": cls, "band": band, "clock": clock, "h_class": "neeche", "h_m": 0.3, "unit": unit, "count": 1,
            "score": score, "stage": stage}


def wp25_fragment():
    return {"schema": "nandi.tree/1", **IDENT, "tree_id": "OT-1", "presets": ["tree360"],
            "coverage": {"az_deg_by_band": {"low": 360, "mid": 360, "high": 350}, "sides": ["N", "E", "S", "W"],
                         "mode": "full", "frames": 24, "seen": 0.94, "masked_pct": 1.2,
                         "pct_by_band": {"low": 95.0, "mid": 93.0, "high": 91.0}, "gap_deg": 0},
            "media": {"manifest_root": "ab" * 32,
                      "keyframe_paths": [f"tree360/OT-1/mid/{i:04d}.jpg" for i in range(24)]},
            "detector": ["simtruth"], "assisted": ["truth_radius"], "next_actions": ["deep_scan"]}


def builder():
    return rec.TreeRecordBuilder(IDENT, ENTRY, zone_id="orchard_demo")


def test_identity_refuses_a_non_sim_tenant_and_mission_id_shape():
    with pytest.raises(ValueError):
        rec.identity("r", "nandi-sim", tenant="acme-prod")
    assert rec.identity("r")["tenant"] == "sim-nandi"
    from datetime import datetime, timezone
    assert rec.mission_id("nandi-sim", datetime(2026, 11, 18, 9, 5, tzinfo=timezone.utc)) == "nandi-sim-20261118T0905Z"


def test_full_record_merges_every_preset_and_validates():
    b = builder()
    b.merge_scan(rec.scan_fragment(ENTRY, "S", ["low", "mid"], [lead("anthracnose", "per_leaf", "mid", 3)],
                                   {"height_m": 2.2, "width_m": 2.0, "gap_pct": None, "lean_deg": 0.0},
                                   ["tree360", "leaf_scan"], 2), t=40.0)
    b.merge_tree360(wp25_fragment(), t=200.0)
    b.merge_perception({"symptoms": [lead()], "trunk": {"frass": False, "gum": True, "collar_rot": False,
                                                       "mud_tubes": False, "weevil_holes": False, "band_ok": True},
                        "leaf": {"status": "ok", "n": 6, "underside_n": 3, "findings": [lead("hopper", "per_leaf")]},
                        "detector": ["simtruth"], "assisted": ["simtruth_leads"]}, "leaf_scan", t=300.0)
    marks = [tc.make_mark(TREE, "fruit", (6.5, -3.2, 1.4), "out_of_reach_marked", stage="mature",
                          tool_hint="pole_picker", target_id="OT-1:f09")]
    v1 = [tc.harvest_record("OT-1", "OT-1:f01", "mango", "clip", "picked", pedicel_mm=19.8, pedicel_error_mm=-0.2)]
    b.merge_harvest(v1, marks, kg=0.6, t=400.0)
    r = b.build()
    assert rec.validate_tree_record(r) == []
    assert r["presets"] == ["scan", "tree360", "leaf_scan", "harvest"]
    assert r["coverage"]["mode"] == "full"                 # TREE 360 coverage replaced the SCAN one
    assert r["harvest"] == {"picked_n": 1, "marked_n": 1, "kg": 0.6, "pedicel_mm_mean": 19.8}
    assert r["visit_t"] == 40.0 and r["marks_n"] == 1
    assert "mark_jobs:1" in r["next_actions"] and "human_inspect" in r["next_actions"]
    assert all(rec.valid_next_action(a) for a in r["next_actions"])
    assert rec.json_size(r) <= rec.TREE_RECORD_MAX_BYTES


def test_scan_never_replaces_tree360_coverage():
    b = builder()
    b.merge_tree360(wp25_fragment())
    b.merge_scan(rec.scan_fragment(ENTRY, "N", ["low"], [], {}, [], 1))
    assert b.build()["coverage"]["mode"] == "full"


def test_symptom_that_is_not_a_lead_is_refused():
    with pytest.raises(ValueError):
        builder().add_symptoms([lead(stage="confirmed")])
    r = builder().build()
    r["symptoms"] = [lead(stage="confirmed")]
    assert any(e.startswith("symptom_stage") for e in rec.validate_tree_record(r))


def test_oversize_record_is_trimmed_in_order_then_refused():
    b = builder()
    frag = wp25_fragment()
    frag["media"]["keyframe_paths"] = [f"tree360/OT-1/high/{'x' * 40}{i:05d}.jpg" for i in range(400)]
    b.merge_tree360(frag)
    b.add_symptoms([lead(cls=f"c{i}", clock=1 + i % 12, unit="per_shoot", band="mid") for i in range(60)])
    r = b.build()
    assert rec.json_size(r) <= rec.TREE_RECORD_MAX_BYTES
    assert r["trimmed"][0] == "media.keyframe_paths" and r["media"]["keyframes_n"] == 400
    huge = dict(r, cultivar="x" * 20000)
    with pytest.raises(ValueError):
        rec.fit_tree_record(huge)


@pytest.mark.parametrize("key,value,rule", [
    ("worker_name", "Ram", "personal_key"), ("phone", "98", "personal_key"), ("face_box", [1, 2], "personal_key"),
    ("person_xy", [1, 2], "personal_key"), ("thumbnail", "x", "image_key"), ("rgb", [0, 0, 0], "image_key"),
    ("blob", b"\x00\x01", "binary"), ("note", "data:image/png;base64,AAAA", "binary"),
    ("frame", "A" * 400, "binary"), ("diagnosis", "anthracnose", "advice_key"),
    ("treatment", "spray", "advice_key"), ("tag_id", "tag-1a2b", "personal_key"), ("track", [[0, 0]], "personal_key"),
])
def test_contract_scan_catches_each_forbidden_field(key, value, rule):
    r = builder().build()
    r["extra"] = {key: value}
    errs = rec.validate_tree_record(r)
    assert any(e.startswith(rule) for e in errs), errs


def test_next_actions_enum_only():
    assert rec.valid_next_action("mark_jobs:3") and not rec.valid_next_action("mark_jobs:x")
    for bad in ("spray", "apply_fungicide", "treat", "", None):
        assert not rec.valid_next_action(bad)
    r = builder().build()
    r["next_actions"] = ["revisit", "spray_imidacloprid"]
    assert "next_action:spray_imidacloprid" in rec.validate_tree_record(r)


def test_mark_checks_and_dedupe():
    m = tc.make_mark(TREE, "prune", (6.4, -3.0, 1.5), "over_capacity", diameter_mm=20.0, tool_hint="lopper",
                     target_id="OT-1:prune:4")
    assert rec.validate_mark(m) == [] and m["paste_required"] is True
    bad = dict(m, kind="spray", clock=13, tool_hint="drone", paste_required=False)
    errs = rec.validate_mark(bad)
    assert {"kind:spray", "clock:13", "tool_hint:drone", "paste_required"} <= set(errs)
    assert len(rec.dedupe_marks([m, dict(m), dict(m, target_id="other")])) == 2


def test_harvest_record_v2():
    v1 = tc.harvest_record("OT-1", "OT-1:f01", "mango", "clip", "picked", pedicel_mm=19.8, cell=3)
    h = rec.harvest_record_v2(v1, IDENT, ENTRY, t=123.4, mass_kg=0.25, stage="mature", zone_id="orchard_demo",
                              assisted=["station_teleport"])
    assert h["schema"] == "nandi.harvest/2" and h["outcome"] == "delivered" and h["outcome_v1"] == "picked"
    assert h["mass_g"] == 250.0 and h["block_id"] == "B0-mango-isolated" and h["physical_tag"] == "OT-1"
    assert "geo" not in h and rec.validate_harvest(h) == []
    assert rec.validate_harvest(dict(h, outcome="eaten"))[0] == "outcome:eaten"
    json.dumps(h, allow_nan=False)
