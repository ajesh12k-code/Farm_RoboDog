"""WP26 symptom leads (ORCHARD-MVP-SPEC 2.5, 3.3, 4.4, 5.6): seasonal class catalogue, lead rule, next actions,
per-unit aggregation, block sampling, view geometry, SimTruth canopy recall, ColorBlob baseline, record fragment.
ROS-free, < 10 s each."""
import json

import numpy as np
import pytest

from atharv_nandi.core.perception import fruit_track as ft
from atharv_nandi.core.perception import pest_detect as pd
from atharv_nandi.sim.world import compose, crops
from atharv_nandi.sim.world import trees as gen

CFG = pd.load_config()
PROFILES = crops.load_profiles()

SPEC_CLASSES = {   # WP26 deliverable list
    "mango": {"sooty_mould", "powdery_mildew", "anthracnose", "malformation", "webber", "mealybug", "shoot_gall",
              "stem_borer_frass", "gummosis", "hopper_honeydew"},
    "guava": {"wilt", "fruit_fly", "bark_caterpillar_webbing"},
    "papaya": {"prsv", "leaf_curl", "collar_rot"},
    "banana": {"sigatoka", "bunchy_top", "weevil_gum"},
}


@pytest.fixture(scope="module")
def demo():
    return {p["tree_id"]: p for p in gen.world_plans(compose.load_field("orchard_demo", 7), 7)}


def test_catalogue_covers_every_spec_class_per_crop():
    for crop, want in SPEC_CLASSES.items():
        have = {c["class"] for c in pd.active_classes(PROFILES[crop], "demo_all_fruiting")}
        assert want <= have, (crop, want - have)
    sig = next(c for c in pd.active_classes(PROFILES["banana"], "mature") if c["class"] == "sigatoka")
    assert sig["graded"] is True
    hop = next(c for c in pd.active_classes(PROFILES["mango"], "panicle") if c["class"] == "hopper_honeydew")
    assert hop["distance_only"] and hop["unit"] == "per_panicle"


def test_classes_follow_the_season():
    m = PROFILES["mango"]
    pan = {c["class"] for c in pd.active_classes(m, "panicle")}
    mat = {c["class"] for c in pd.active_classes(m, "mature")}
    assert {"powdery_mildew", "malformation", "hopper_honeydew", "mealybug"} <= pan
    assert not {"powdery_mildew", "malformation"} & mat
    assert {"sooty_mould", "anthracnose", "webber", "shoot_gall"} <= mat
    assert "shoot_gall" not in pan                      # catalogue-only class keeps its own seasons
    assert {"stem_borer_frass", "gummosis"} <= pan & mat


def test_lead_rule_and_enumerated_next_actions():
    with pytest.raises(ValueError):
        pd.lead("anthracnose", "per_leaf", [0, 0, 0.5], 0.5, "low", 6, 0.9, "simtruth", stage="confirmed")
    with pytest.raises(ValueError):
        pd.lead("anthracnose", "per_tree", [0, 0, 0.5], 0.5, "low", 6, 0.9, "simtruth")
    ld = pd.lead("gummosis", "trunk", [0, 0, 0.3], 0.3, "low", 6, 0.8, "simtruth")
    assert ld["stage"] == "lead" and ld["h_class"] == "neeche"
    sy = [pd.lead("sooty_mould", "per_shoot", [0, 0, 1.5], 1.5, "mid", 12, 0.9, "simtruth"),
          pd.lead("anthracnose", "per_leaf", [0, 0, 0.5], 0.5, "low", 6, 0.9, "simtruth"), ld]
    acts = pd.next_actions(sy, trunk_leads=1, leaf_leads=1, marks_n=3, coverage_partial=True)
    assert acts == ["revisit", "tree360", "leaf_scan", "trunk_scan", "human_inspect", "mark_jobs:3"]
    acts2 = pd.next_actions(sy, did=["tree360", "leaf_scan", "trunk_scan"])
    assert acts2 == ["deep_scan"]
    for a in acts + acts2:
        assert a in pd.NEXT_ACTIONS or a.startswith("mark_jobs:")
    assert pd.next_actions([pd.lead("webber", "per_shoot", [0, 0, 1], 1, "mid", 12, 0.2, "simtruth")]) == []


def test_dedupe_and_unit_summary():
    a = pd.lead("anthracnose", "per_leaf", [0, 0, 0.5], 0.5, "low", 6, 0.7, "simtruth", truth="d1")
    b = pd.lead("anthracnose", "per_leaf", [0.01, 0, 0.5], 0.5, "low", 6, 0.9, "simtruth", truth="d1")
    c = pd.lead("anthracnose", "per_leaf", [0.3, 0, 0.5], 0.5, "low", 6, 0.8, "simtruth", truth="d2")
    h = pd.lead("hopper_honeydew", "per_panicle", [0, 0, 1.5], 1.5, "mid", 12, 0.8, "simtruth")
    g = pd.lead("gummosis", "trunk", [0, 0, 0.3], 0.3, "low", 6, 0.8, "simtruth")
    rows = pd.dedupe_leads([a, b, c, h, g])
    leaf = [r for r in rows if r["class"] == "anthracnose"]
    assert len(leaf) == 2 and max(r["views"] for r in leaf) == 2 and max(r["score"] for r in leaf) == 0.9
    u = pd.unit_summary(rows, {"per_leaf": 8, "per_panicle": 10})
    assert u["anthracnose"]["pct"] == 25.0 and u["anthracnose"]["stage"] == "lead"
    assert u["hopper_honeydew"]["per_unit"] == 0.1
    assert u["gummosis"]["present"] is True and u["gummosis"]["n_units"] == 1
    bs = pd.block_summary([{"tree_id": "a", "symptoms": rows}, {"tree_id": "b", "symptoms": []}], {"n": 2})
    assert bs["pct_trees_affected"]["gummosis"] == 50.0 and bs["sampling_plan"] == {"n": 2}


def test_block_sampling_plan_is_stratified_and_deterministic():
    field = compose.load_field("orchard_demo", 7)
    entries = gen.registry(field, sorted(PROFILES))
    plan = pd.block_sampling_plan(entries)
    assert plan["n"] == min(6, len(entries)) and plan["basis"] == "stratified_by_direction"
    assert sum(1 for s in pd.STRATA if plan["strata"][s]) >= 3
    assert plan == pd.block_sampling_plan(entries)
    mango = pd.block_sampling_plan(entries, crop="mango")
    assert mango["requested_n"] == 8 and mango["n"] == 5              # 5 mango trees: never more than the block
    assert set(mango["tree_ids"]) == {e["tree_id"] for e in entries if e["crop"] == "mango"}
    two = pd.block_sampling_plan(entries, n=4, seed=1)
    assert len({pd.stratum_of((e["x"] if "x" in e else e["geo"]["x"], e["y"] if "y" in e else e["geo"]["y"]),
                              np.mean([[(x.get("x") or x["geo"]["x"]), (x.get("y") or x["geo"]["y"])]
                                       for x in entries], axis=0))
                 for e in entries if e["tree_id"] in two["tree_ids"]}) >= 3


def test_view_projection_round_trip_and_camera_convention():
    v = pd.View(pos=[1, 2, 1], fwd=[0.3, -0.5, -0.4])
    p = np.array([1.5, 1.4, 0.6])
    u, vv, z = v.project(p)
    assert np.allclose(v.unproject(u, vv, z), p)
    R = v.rot()
    assert np.allclose(R[:, 2], -v.fwd) and np.allclose(pd.quat_mat(v.quat()), R)
    assert v.project(v.pos - v.fwd) is None


def test_canopy_leads_recall_from_tree360_frames(demo):
    """Canopy-level leads (>= 5 cm decals) from the 8-station TREE 360 frames: SimTruth recall >= 0.9."""
    hits = n = 0
    for p in demo.values():
        views = ft.orbit_views(p)
        leads = pd.dedupe_leads(pd.canopy_leads(p, PROFILES[p["crop"]], views))
        truth = pd.canopy_truth_ids(p, PROFILES[p["crop"]], views)
        r = pd.recall(leads, truth, True, truth)
        hits, n = hits + r[1], n + r[2]
        assert all(ld["stage"] == "lead" and ld["detector"] == "simtruth" for ld in leads)
    assert n >= 30 and hits / n >= 0.9
    print(f"canopy lead recall {hits}/{n} = {hits / n:.3f}")


def test_colorblob_finds_class_coloured_blobs_on_a_synthetic_frame():
    img = np.zeros((60, 80, 3), np.float32)
    img[:] = [0.18, 0.50, 0.18]                                      # leaf green
    img[10:20, 10:20] = [0.95, 0.95, 0.92]                           # mealybug white
    img[30:40, 40:52] = [0.08, 0.08, 0.08]                           # sooty mould
    classes = [c for c in pd.active_classes(PROFILES["mango"], "demo_all_fruiting")
               if c["class"] in ("mealybug", "sooty_mould")]
    blobs = pd.ColorBlobLeadDetector().detect((img * 255).astype(np.uint8), classes)
    got = {b["class"]: b for b in blobs}
    assert set(got) == {"mealybug", "sooty_mould"}
    assert abs(got["mealybug"]["u"] - 14.5) < 1 and got["sooty_mould"]["area_px"] == 120
    assert all(b["stage"] == "lead" for b in blobs)


def test_record_fragment_is_small_and_has_no_truth_or_images(demo):
    from atharv_nandi.core.perception import leaf_scan as ls, trunk_scan as ts
    p = demo["OT-1"]
    lr = ls.run_leaf_scan(p, PROFILES["mango"])
    tr = ts.run_trunk_scan(p, PROFILES["mango"])
    sc = ft.scout(p, PROFILES["mango"], "2026-06-20", crown_gate_ok=True)
    can = pd.dedupe_leads(pd.canopy_leads(p, PROFILES["mango"], ft.orbit_views(p)))
    sym = pd.dedupe_leads(can + pd.dedupe_leads(lr.leads) + tr.symptoms)
    frag = pd.record_fragment(p, sym, leaf=lr.leaf_block, trunk=tr.trunk, fruit=sc.fruit,
                              marks_n=len(sc.marks) + len(tr.prune_flags), did=["leaf_scan", "trunk_scan"],
                              n_units={"per_leaf": lr.leaf_block["n"]})
    s = json.dumps(frag)
    assert len(s.encode()) <= 16 * 1024                        # MP 0.6: TreeRecord <= 16 KiB
    i = max(s.find("\"truth"), s.find("truth_ids"))
    assert i < 0 and "jpg" not in s and "rgb" not in s, s[max(0, i - 200):i + 50]
    assert all(r["stage"] == "lead" for r in frag["symptoms"])
    assert frag["next_actions"] and frag["next_actions"][-1].startswith("mark_jobs:")
    assert not any(w in s.lower() for w in ("spray", "pesticide", "fungicide", "insecticide", "dose"))
