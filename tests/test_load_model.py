"""BT3 of the physics alignment spec v1: atharv_nandi/core/load_model.py ratings, basket, arm, pick gate."""
import math

import pytest

from atharv_nandi.core import load_model as lm


@pytest.mark.parametrize("reach,kg", [(0.0, 2.0), (0.3, 2.0), (0.5, 2.0), (0.5000001, 1.5), (0.7, 1.5),
                                      (0.8, 1.5), (0.80001, 0.0), (1.2, 0.0)])
def test_reach_table_boundaries(reach, kg):
    assert lm.arm_payload_limit(reach) == kg


@pytest.mark.parametrize("bad", [-0.01, math.nan, math.inf])
def test_reach_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        lm.arm_payload_limit(bad)


def test_basket_volume_binds_over_mass():
    cap = lm.basket_capacity()
    assert cap["volume_l"] == 10.8 and cap["bulk_density_kg_per_l"] == 0.6
    assert cap["volume_limit_kg"] == pytest.approx(10.8 * 0.6)
    assert cap["mass_limit_kg"] == 10.0
    assert cap["limit_kg"] == pytest.approx(6.48) and cap["binding"] == "volume"
    # dense produce (1.2 kg/L ASSUMED for the test): the volume limit 12.96 kg no longer binds
    dense = lm.basket_capacity(bulk_density_kg_per_l=1.2)
    assert dense["binding"] == "mass" and dense["limit_kg"] == 10.0
    # standing or slow walk: the 15 kg absolute applies to the mass rating
    slow = lm.basket_capacity(walking=True, speed_scale=0.5, bulk_density_kg_per_l=2.0)
    assert slow["mass_limit_kg"] == 15.0 and slow["limit_kg"] == 15.0
    assert lm.basket_capacity(walking=False, bulk_density_kg_per_l=2.0)["limit_kg"] == 15.0


def test_check_basket_kg_and_counts():
    assert lm.check_basket(6.0)["ok"]
    over = lm.check_basket(7.0)
    assert not over["ok"] and over["binding"] == "volume"
    # 12 mangoes at the typical 0.35 kg = 4.2 kg
    r = lm.check_basket({"mango": 12})
    assert r["kg"] == pytest.approx(12 * 0.35) and r["ok"]
    assert not lm.check_basket({"guava": 40})["ok"]  # 40 x 0.175 = 7.0 kg > 6.48
    with pytest.raises(ValueError):
        lm.check_basket(-1.0)


def test_banana_is_assist_only():
    for name in ("banana", "banana_bunch"):
        r = lm.check_pick(name, 20.0, 0.5, 0.3)
        assert r["decision"] == lm.DECISION_ASSIST_ONLY and not r["ok"]
    assert lm.crop_profile("banana").mass_min_kg == 15.0 and lm.crop_profile("banana").mass_max_kg == 40.0


def test_papaya_2kg_at_0p7_is_rejected():
    r = lm.check_pick("papaya", 2.0, 0.0, 0.7)
    assert r["decision"] == lm.DECISION_REJECT and not r["ok"] and "assist" in r["reasons"][0]
    # the same papaya close in (0.4 m): 2.0 kg total with no tool is exactly the 2.0 kg rating
    assert lm.check_pick("papaya", 2.0, 0.0, 0.4)["decision"] == lm.DECISION_OK


def test_mango_with_tool_at_0p7_is_ok():
    r = lm.check_pick("mango", 0.4, 0.6, 0.7)
    assert r["decision"] == lm.DECISION_OK and r["ok"]
    assert r["total_kg"] == pytest.approx(1.0) and r["limit_kg"] == 1.5
    assert r["method"] == lm.METHOD_CLIP_PEDICEL
    assert lm.check_pick("mango", 0.4, 0.6, 0.85)["decision"] == lm.DECISION_REJECT  # > 0.8 m: no load


def test_pick_checks_basket_and_unknown_crop():
    assert lm.check_pick("guava", 0.2, 0.3, 0.4, basket_kg=6.4)["decision"] == lm.DECISION_REJECT
    assert lm.check_pick("guava", None, 0.3, 0.4)["total_kg"] == pytest.approx(0.25 + 0.3)  # unknown mass -> max
    with pytest.raises(KeyError):
        lm.crop_profile("durian")


def test_mount_ratings_table():
    r = lm.MOUNT_RATINGS
    assert set(r) == {"basket_payload", "tool_flange", "mast_head_ring", "mast_base", "front_mount", "rear_mount",
                      "tow_dock"}
    assert (r["basket_payload"].rated_kg, r["basket_payload"].max_kg) == (10.0, 15.0)
    assert (r["mast_head_ring"].rated_kg, r["mast_head_ring"].max_kg) == (1.5, 3.0)
    assert (r["mast_base"].moment_nm, r["mast_base"].force_n) == (290.0, 185.0)
    assert r["front_mount"].rated_kg == r["rear_mount"].rated_kg == 5.0
    assert r["tow_dock"].force_n == 50.0
    for rating in r.values():
        assert rating.tag in ("MEASURED", "CITED", "DERIVED", "ASSUMED")


def test_mass_and_com_shift():
    r = lm.total_mass_and_com_shift(25.0, (0.0, 0.0, 0.3), [(5.0, (-0.34, 0.0, 0.3))])
    assert r["total_kg"] == 30.0
    assert r["shift"][0] == pytest.approx(5 * -0.34 / 30.0) and r["shift"][2] == pytest.approx(0.0)
    r = lm.total_mass_and_com_shift(31.8, (0.01, 0.0, 0.3), [(5.0, (0.34, 0, 0)), (5.0, (-0.34, 0, 0))])
    assert r["shift"][0] == pytest.approx((31.8 * 0.01) / 41.8 - 0.01)  # a pair at +-x keeps x centred


def test_rms_torque_and_window():
    r = lm.rms_torque([[3.0, -50.0], [-4.0, 50.0]])
    assert r["rms_nm"] == pytest.approx([math.sqrt(12.5), 50.0])
    assert r["max_rms_nm"] == 50.0 and r["over"] == [1] and r["limit_nm"] == 48.0
    assert lm.rms_torque([])["max_rms_nm"] is None
    w = lm.TorqueRmsWindow(window_s=1.0)
    for k in range(30):
        w.add(k * 0.1, [100.0 if k < 10 else 10.0])
    assert w.summary()["max_rms_nm"] == pytest.approx(10.0)  # the 100 Nm samples aged out


def test_energy_remaining():
    p = lm.electrical_power_w(146.0)
    assert p == pytest.approx(146.0 / 0.6 + 15.0)  # ~258 W
    e = lm.energy_remaining(146.0)
    assert e["hours"] == pytest.approx(500.0 / p) and 1.9 < e["hours"] < 2.0
    assert lm.energy_remaining(146.0, soc=0.5)["remaining_wh"] == 250.0
    assert lm.energy_remaining(0.0, used_wh=600.0)["remaining_wh"] == 0.0
    assert lm.electrical_power_w(-50.0) == 15.0
