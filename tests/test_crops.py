"""WP23 crop profiles (ORCHARD-MVP-SPEC 4.4; MISSION-PRESETS 4-6). Pure Python, no MuJoCo, each test < 1 s."""
import copy

import pytest

from atharv_nandi.sim.world import crops

CROPS = ("mango", "guava", "papaya", "banana")


@pytest.fixture(scope="module")
def profiles():
    return crops.load_profiles()


def test_four_profiles_load_and_validate(profiles):
    assert set(CROPS) <= set(profiles)
    for name, p in profiles.items():
        assert crops.validate(p) == [], name
        assert p["schema"] == crops.SCHEMA and p["crop"] == name


def test_project_scope_oq01_mango_guava_harvest_papaya_banana_scout_mark(profiles):
    assert profiles["mango"]["mvp_scope"] == "harvest" and profiles["mango"]["priority"] == 1
    assert profiles["guava"]["mvp_scope"] == "harvest" and profiles["guava"]["priority"] == 2
    assert profiles["papaya"]["mvp_scope"] == "scout_mark"
    assert profiles["banana"]["mvp_scope"] == "scout_mark"


def test_methods_tools_and_fixed_pool_kinds(profiles):
    m, g, p, b = (profiles[c] for c in CROPS)
    assert (m["harvest_method"], m["tool"], m["pool_kind"], m["pull_release"]) == ("clip", "T2", "connect", False)
    assert m["one_round"] is True
    assert (g["harvest_method"], g["pool_kind"]) == ("clip", "weld") and "twist" in g["methods"]
    assert (p["harvest_method"], p["tool"], p["pool_kind"], p["one_per_trip"]) == ("twist", "T4", "weld", True)
    assert (b["harvest_method"], b["pool_kind"], b["tool"]) == ("mark", "none", None)
    assert p["placement_speed_max_m_s"] == 0.5 and p["jaw_clearance_mm"] == 15


def test_mango_pedicel_default_and_range(profiles):
    ped = profiles["mango"]["pedicel_leave_cm"]
    assert (ped["default"], ped["min"], ped["max"]) == (2.0, 1.0, 5.0)


def test_twist_crops_have_swept_tau(profiles):
    assert profiles["guava"]["tau_detach_nm"] == 0.15
    assert profiles["papaya"]["tau_detach_nm"] == 0.3      # WP23 sweep, the WP23 contract notes
    for c in ("guava", "papaya"):
        assert profiles[c]["twist_min_roll_deg"] == 30


def test_mango_maturity_never_uses_skin_colour(profiles):
    m = profiles["mango"]
    assert m["skin_colour_cue"] is False
    assert "skin_colour" not in m["maturity_cues"]
    assert len({tuple(s["rgba"]) for s in m["stages"]}) == 1
    assert set(m["maturity_cues"]) == {"shoulder_rise", "stem_end_colour", "days_since_fruit_set"}
    mature = {"shoulder_rise_mm": 3.0, "stem_end_colour": "yellowing"}
    assert crops.stage_from_cues(m, mature) == "mature"
    assert crops.stage_from_cues(m, {"shoulder_rise_mm": 0.5, "stem_end_colour": "green"}) == "immature"
    # a skin cue in the input changes nothing for mango
    assert crops.stage_from_cues(m, dict(mature, skin="dark_green")) == "mature"
    assert crops.stage_from_cues(m, {"skin": "light_green_yellow"}) is None


def test_validate_rejects_a_mango_profile_with_a_skin_cue(profiles):
    bad = copy.deepcopy(profiles["mango"])
    bad["stages"][1]["rgba"] = [0.9, 0.7, 0.1, 1.0]
    assert any("stage colours differ" in e for e in crops.validate(bad))
    bad = copy.deepcopy(profiles["mango"])
    bad["maturity_cues"] = bad["maturity_cues"] + ["skin_colour"]
    assert any("skin cue" in e for e in crops.validate(bad))
    bad = copy.deepcopy(profiles["guava"])
    bad["tau_detach_nm"] = None
    assert any("tau_detach" in e for e in crops.validate(bad))
    bad = copy.deepcopy(profiles["banana"])
    bad["pool_kind"] = "weld"
    assert any("MARK-only" in e for e in crops.validate(bad))


def test_validate_rejects_mango_skin_colour_any_spelling(profiles):
    """OS 4.4 holds for mango whatever the flag says and however the cue is spelled."""
    bad = copy.deepcopy(profiles["mango"])
    bad["skin_colour_cue"] = True
    bad["maturity_cues"] = bad["maturity_cues"] + ["skin_colour"]
    bad["stages"][1]["rgba"] = [0.9, 0.8, 0.1, 1.0]
    errs = crops.validate(bad)
    assert any("skin_colour_cue must be false" in e for e in errs)
    assert any("stage colours differ" in e for e in errs) and any("skin cue" in e for e in errs)
    for cue in ("peel_color", "color", "colour", "skin_hue", "rind_colour", "Peel-Color"):
        bad = copy.deepcopy(profiles["mango"])
        bad["maturity_cues"] = bad["maturity_cues"] + [cue]
        assert any("skin cue" in e for e in crops.validate(bad)), cue
    bad = copy.deepcopy(profiles["mango"])
    bad["stages"][1]["cues"] = dict(bad["stages"][1]["cues"], color="yellow")
    assert any("skin cue" in e for e in crops.validate(bad))
    # the legitimate mango colour cue is the stem end, not the skin
    assert not crops.is_skin_colour_cue("stem_end_colour") and crops.validate(profiles["mango"]) == []
    assert crops.stage_from_cues(profiles["mango"], {"color": "yellow"}) is None


def test_colour_crops_use_colour(profiles):
    for c in ("guava", "papaya"):
        p = profiles[c]
        assert p["skin_colour_cue"] is True
        assert len({tuple(s["rgba"]) for s in p["stages"]}) == 2
    assert crops.stage_from_cues(profiles["guava"], {"skin": "light_green_yellow"}) == "mature"
    assert crops.stage_from_cues(profiles["papaya"], {"orange_streaks": 4}) == "colour_break"


def test_harvest_window_from_registry_dates(profiles):
    m = profiles["mango"]
    w = crops.harvest_window(m, {"fruit_set_date": "2026-03-17"}, today="2026-06-20")
    assert (w["from"], w["to"], w["basis"]) == ("2026-06-09", "2026-06-29", "days_since_fruit_set")
    assert w["open"] is True
    # only a bagging date: fruit set = bagging - 28 d, basis calendar (OQ-15 A)
    w = crops.harvest_window(m, {"bagging_date": "2026-04-14"})
    assert w["basis"] == "calendar" and w["from"] == "2026-06-09"
    b = crops.harvest_window(profiles["banana"], {"shooting_date": "2026-03-12"}, today="2026-06-20")
    assert (b["from"], b["to"]) == ("2026-06-10", "2026-07-10") and b["open"]
    assert crops.harvest_window(m, {}) is None


def test_stage_for_age(profiles):
    assert crops.stage_for_age(profiles["mango"], 60) == "immature"
    assert crops.stage_for_age(profiles["mango"], 90) == "mature"
    assert crops.stage_for_age(profiles["banana"], 100) == "mature"
    # fractional ages count completed days: 83.5 d is day 83, still immature (no gap between 83 and 84)
    assert crops.stage_for_age(profiles["mango"], 83.5) == "immature"
    assert crops.stage_for_age(profiles["mango"], 83.999) == "immature"
    assert crops.stage_for_age(profiles["mango"], 84.0) == "mature"
    assert crops.stage_for_age(profiles["banana"], 89.9) == "immature"
    bad = copy.deepcopy(profiles["mango"])
    bad["stages"][1]["days_since_fruit_set"] = [86, 120]          # a 2-day hole between the stages
    assert any("tile whole days" in e for e in crops.validate(bad))


def test_up_calendar_windows(profiles):
    assert crops.calendar_window(profiles["mango"], "Dashehari") == ("2026-06-08", "2026-07-07")
    assert crops.calendar_window(profiles["mango"], "Amrapali") == ("2026-07-01", "2026-08-10")
    assert crops.calendar_window(profiles["guava"]) == ("2026-11-01", "2027-01-31")   # mrig bahar over New Year
    assert crops.calendar_window(profiles["papaya"]) is None


def test_harvest_decision_reason_codes(profiles):
    m, p, b = profiles["mango"], profiles["papaya"], profiles["banana"]
    ripe_in = {"stage": "mature", "in_band": True}
    assert crops.harvest_decision(m, ripe_in) == ("pick", "")
    assert crops.harvest_decision(m, {"stage": "mature", "in_band": False}) == ("mark", "out_of_reach_marked")
    assert crops.harvest_decision(m, {"stage": "immature", "in_band": True}) == ("skip", "")
    assert crops.harvest_decision(m, dict(ripe_in, bagged=True)) == ("skip", "bagged")
    assert crops.harvest_decision(m, ripe_in, method="twist") == ("skip", "not_in_mvp")
    assert crops.harvest_decision(p, {"stage": "colour_break", "in_band": True, "clearance_mm": 10}) == \
        ("mark", "no_clearance")
    assert crops.harvest_decision(p, {"stage": "colour_break", "in_band": True, "clearance_mm": 40}) == ("pick", "")
    assert crops.harvest_decision(b, {"stage": "mature", "in_band": False}) == ("mark", "human_cut")


def test_prune_decision_capacity_and_paste(profiles):
    m = profiles["mango"]
    assert crops.prune_decision(m, "rootstock_sprout", 6, 0.22) == ("cut", "")
    assert crops.prune_decision(m, "skirt_lift", 20, 0.55) == ("mark", "paste_required")
    assert crops.prune_decision(m, "water_shoot", 16, 1.6) == ("mark", "out_of_reach_marked")
    assert crops.prune_decision(m, "skirt_lift", 10, 0.5, keep=True) == ("skip", "keep")
    assert crops.paste_required(m, 12.5) and not crops.paste_required(m, 12.0)
    assert crops.prune_decision(profiles["banana"], "banana_desucker", 60, 0.1) == ("mark", "human_cut")


def test_banana_deleaf_respects_the_functional_leaf_floor(profiles):
    b = profiles["banana"]
    assert crops.min_functional_leaves(b, "flowering") == 9 and crops.min_functional_leaves(b, "harvest") == 4
    assert crops.min_functional_leaves(profiles["mango"], "flowering") is None
    leaves = [{"id": f"L{i}", "functional": True, "sigatoka_pct": 0.0} for i in range(8)]
    leaves += [{"id": "S1", "functional": True, "sigatoka_pct": 40.0}, {"id": "S2", "functional": True,
               "sigatoka_pct": 20.0}, {"id": "D1", "functional": False}]
    plan = crops.deleaf_plan(b, leaves, "flowering")       # 10 functional, floor 9: dead + 1 worst Sigatoka
    assert plan == ["D1", "S1"]
    assert crops.deleaf_plan(b, leaves, "harvest") == ["D1", "S1", "S2"]


def test_symptom_classes_by_season(profiles):
    m = profiles["mango"]
    assert {c["class"] for c in crops.symptom_classes(m, "panicle")} >= {"powdery_mildew", "malformation"}
    assert "powdery_mildew" not in {c["class"] for c in crops.symptom_classes(m, "mature")}
    assert {c["class"] for c in crops.symptom_classes(m, "demo_all_fruiting", "trunk")} >= {"gummosis"}


def test_ribbon_rota_and_height_classes():
    assert crops.ribbon_shooting_date("blue", {"blue": "2026-03-12"}) == "2026-03-12"
    assert crops.ribbon_shooting_date("pink", {"blue": "2026-03-12"}) is None
    assert [crops.h_class(z) for z in (0.5, 0.8, 1.2, 2.0)] == ["neeche", "beech", "beech", "upar"]


def test_profiles_dir_override_and_digest(tmp_path, monkeypatch, profiles):
    import yaml
    for c in CROPS:
        (tmp_path / f"{c}.yaml").write_text(yaml.safe_dump(profiles[c]))
    d0 = crops.config_digest(str(tmp_path))
    monkeypatch.setenv("NANDI_CROPS_CONFIG", str(tmp_path))
    assert set(crops.load_profiles()) == set(CROPS)
    lychee = copy.deepcopy(profiles["guava"])
    lychee["crop"] = "lychee"
    (tmp_path / "lychee.yaml").write_text(yaml.safe_dump(lychee))
    assert "lychee" in crops.load_profiles() and crops.config_digest() != d0
    (tmp_path / "broken.yaml").write_text(yaml.safe_dump({"crop": "broken"}))
    with pytest.raises(crops.CropProfileError):
        crops.load_profiles()
