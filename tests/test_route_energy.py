"""WP7 route-energy estimator, reserve, Ready check / mid-tour monitor, endurance numbers, and the estimator
error against the integrated energy of real physics runs (ROS-free, headless, each <= 10 s)."""
import json
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

from atharv_nandi.core import energy_model as em  # noqa: E402
from atharv_nandi.core import route_energy as re_  # noqa: E402
from test_platform import Run, mb  # noqa: E402

P = em.EnergyParams()
TRUE = re_.Coefficients(wh_per_m=0.21, wh_per_item=0.31, p_idle_w=17.0, wh_per_crown_m=0.03)


def test_formula_seed_and_return():
    c = re_.Coefficients.seed(P, 0.18)
    assert c.wh_per_m == pytest.approx((220.0 - 15.0) / 0.18 / 3600.0)
    assert (c.wh_per_item, c.p_idle_w, c.source) == (0.25, 15.0, "seed")
    plan = re_.RoutePlan(route_m=100.0, items=4, t_work_s=1800.0, crown_m=3.2, return_m=40.0)
    assert c.energy_wh(plan) == pytest.approx(c.wh_per_m * 100 + 0.25 * 4 + 15.0 * 0.5 + c.wh_per_crown_m * 3.2)
    assert c.return_wh(40.0) == pytest.approx(c.wh_per_m * 40 + 15.0 * 40 / 0.18 / 3600)
    assert re_.ASSUMED_CROWN_WH_PER_M == pytest.approx((12 / 0.2 + 0.8 * 9.81 / 0.5) / 3600)
    with pytest.raises(ValueError):
        re_.RoutePlan(route_m=-1.0)
    with pytest.raises(ValueError):
        re_.RoutePlan(items=1.5)
    assert re_.RoutePlan.from_dict({"route_m": 3, "items": 2}).features() == (3.0, 2.0, 0.0, 0.0)


def test_tracker_measures_coefficients_and_keeps_unexcited_at_seed():
    rng = random.Random(3)
    tr = re_.CoefficientTracker(re_.Coefficients.seed(P, 0.18))
    for _ in range(40):                 # segments: walk / stand / item mixes, 2 % energy noise
        f = re_.RoutePlan(route_m=rng.uniform(0, 1.0), items=rng.choice([0, 0, 1]), t_work_s=5.0,
                          crown_m=rng.choice([0.0, 0.0, 0.6]))
        tr.add_segment(re_.Segment(f.route_m, f.items, 5.0, f.crown_m, TRUE.energy_wh(f) * rng.uniform(0.98, 1.02)))
    c = tr.coefficients()
    assert c.source == "measured" and c.n_segments == 40
    assert c.wh_per_m == pytest.approx(0.21, rel=0.05)
    assert c.p_idle_w == pytest.approx(17.0, rel=0.05)
    assert c.wh_per_item == pytest.approx(0.31, rel=0.10)
    tr2 = re_.CoefficientTracker(re_.Coefficients.seed(P, 0.18), segment_s=1.0)
    for _ in range(30):                 # walking only: items and crown never excited -> stay at the seed
        tr2.update(0.5, route_m=0.09, energy_wh=TRUE.energy_wh(re_.RoutePlan(route_m=0.09, t_work_s=0.5)))
    c2 = tr2.coefficients()
    assert c2.n_segments == 15
    assert c2.wh_per_item == pytest.approx(0.25, rel=1e-3) and c2.wh_per_crown_m == pytest.approx(
        re_.ASSUMED_CROWN_WH_PER_M, rel=1e-3)
    tr2.update(-1.0, energy_wh=5.0)     # bad increments are ignored
    assert tr2.total.t_s == pytest.approx(15.0)
    tr2.reset()
    assert tr2.coefficients().source == "seed"
    with pytest.raises(ValueError):
        re_.CoefficientTracker(TRUE, segment_s=0.0)


def test_run_card_round_trip_and_fallback(tmp_path):
    card = {"energy": re_.run_card_energy(TRUE, 500.0, integrated_wh=10.0, predicted_wh=11.0)}
    assert card["energy"]["estimator_error"] == pytest.approx(0.1)
    path = tmp_path / "run_card.json"
    path.write_text(json.dumps(card))
    c = re_.Coefficients.from_run_card(str(path))
    assert c.source == "run_card" and c.wh_per_m == pytest.approx(0.21) and c.p_idle_w == pytest.approx(17.0)
    assert re_.Coefficients.from_run_card(card).wh_per_item == pytest.approx(0.31)
    seed = re_.Coefficients.seed(P)
    assert re_.Coefficients.from_run_card(str(tmp_path / "missing.json")) == seed
    assert re_.Coefficients.from_run_card({"energy": {"coefficients": {"wh_per_m": 1}}}) == seed
    assert re_.Coefficients.from_run_card({"energy": {"coefficients": {**TRUE.to_dict(), "wh_per_m": -1}}}) == seed
    (tmp_path / "bad.json").write_text("{not json")
    assert re_.Coefficients.from_run_card(str(tmp_path / "bad.json")) == seed
    with pytest.raises(ValueError):
        re_.estimator_error(1.0, 0.0)


def test_ready_check_refuses_below_reserve():
    c = TRUE
    plan = re_.RoutePlan(route_m=400.0, items=6, t_work_s=3600.0, crown_m=10.0, return_m=50.0)
    need = c.energy_wh(plan)
    res = re_.reserve_soc(50.0, c, 500.0)
    assert res == pytest.approx(c.return_wh(50.0) / 500.0 + 0.15)
    ok = re_.ready_check_energy(0.9, plan, c, 500.0)
    assert ok.ok and ok.reason == "" and ok.projected_end_soc == pytest.approx(0.9 - need / 500.0)
    soc_edge = res + need / 500.0
    assert re_.ready_check_energy(soc_edge + 1e-6, plan, c, 500.0).ok
    bad = re_.ready_check_energy(soc_edge - 1e-3, plan, c, 500.0)
    assert (bad.ok, bad.reason) == (False, "energy_reserve") and "reserve" in bad.detail
    assert set(bad.to_dict()) >= {"ok", "reason", "projected_end_soc", "reserve", "need_wh"}


def test_reserve_monitor_latches_return_and_wait():
    mon = re_.ReserveMonitor(500.0, confirm_s=1.0)
    rem = re_.RoutePlan(route_m=300.0, t_work_s=2400.0, return_m=60.0)
    assert mon.update(0.0, 0.9, rem, TRUE) == re_.ACTION_CONTINUE
    low = 0.3
    assert not re_.ready_check_energy(low, rem, TRUE, 500.0).ok
    assert mon.update(1.0, low, rem, TRUE) == re_.ACTION_CONTINUE      # breach must persist 1 s
    assert mon.update(1.5, 0.9, rem, TRUE) == re_.ACTION_CONTINUE      # cleared: timer restarts
    mon.update(2.0, low, rem, TRUE)
    assert mon.update(3.0, low, rem, TRUE) == re_.ACTION_RETURN_AND_WAIT
    assert mon.update(4.0, 0.95, rem, TRUE) == re_.ACTION_RETURN_AND_WAIT   # latched
    st = mon.status()
    assert st["reason"] == "energy_reserve" and st["latched_at"] == 3.0
    mon.clear()
    assert mon.action == re_.ACTION_CONTINUE


def test_throughput_reports_ha_and_trees_per_charge():
    th = re_.throughput(TRUE, 500.0)
    usable = 500.0 * 0.85 - TRUE.return_wh(100.0)
    scan_t = 2100.0 / 0.18 + 1600 * 3.0
    assert th["scan_h_per_ha"] == pytest.approx(scan_t / 3600, abs=0.01)
    assert th["scan_h_per_ha"] == pytest.approx(4.6, abs=0.1)          # spec 1.3: ~4.6 h/ha with stops
    assert th["scan_h_per_ha_no_stops"] == pytest.approx(3.3, abs=0.1)  # spec 1.3: ~3.3 h/ha
    assert th["ha_per_charge"] == pytest.approx(usable / (0.21 * 2100 + 17.0 * scan_t / 3600), rel=1e-3)
    assert th["trees_per_charge"] > 10 and th["usable_wh_per_charge"] == pytest.approx(usable, abs=0.01)
    assert th["assumed"] is True


# ------------------------------------------------------------------ physics: estimator error on a lite route
# orchard_lite (WP4) has not landed: the legacy world (default ridged terrain) stands in. The calibration run A
# (seed 7) writes the "last run card"; the scored run B (seed 11) has a different walk / stop mix. A cross-terrain
# card (A on flat ground) is reported too: it is the harder case (no gate in the spec), kept as a regression guard.
ROUTE_A = [(4.0, 0.0), (12.0, 0.5), (6.0, 0.0), (12.0, 0.5), (6.0, 0.0), (8.0, 0.5), (3.0, 0.0)]
ROUTE_B = [(3.0, 0.0), (16.0, 0.5), (9.0, 0.0), (6.0, 0.5), (4.0, 0.0), (14.0, 0.5), (8.0, 0.0), (6.0, 0.5)]


def _drive(run, route):
    for dur, vx in route:
        run.run(dur, vx=vx)


def _features(p):
    return re_.RoutePlan(route_m=p.route_m, items=p.items, t_work_s=p.tracker.total.t_s, crown_m=p.crown_m)


def _card(tmp_path_factory, terrain):
    m = mb.build(NANDI_TERRAIN=terrain, NANDI_CROPS=1, NANDI_SEED=7)
    a = Run(m)
    _drive(a, ROUTE_A)
    p = a.p
    coef = p.tracker.coefficients()
    c = {"run_id": f"wp7-cal-A-{terrain or 'ridges'}",
         "energy": re_.run_card_energy(coef, p.params.capacity_wh, p.e_total_wh, coef.energy_wh(_features(p)))}
    path = tmp_path_factory.mktemp("card") / "run_card.json"
    path.write_text(json.dumps(c))
    return str(path), c


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    return _card(tmp_path_factory, None)


@pytest.fixture(scope="module")
def flat_card(tmp_path_factory):
    return _card(tmp_path_factory, "flat")


def _score(card, monkeypatch):
    """Run B with the card as NANDI_ENERGY_CARD: (start error, mid error, numbers)."""
    path, _ = card
    monkeypatch.setenv("NANDI_ENERGY_CARD", path)
    m = mb.build(NANDI_SEED=11, NANDI_CROPS=1)          # default terrain (ridges), other seed
    b = Run(m)
    _drive(b, ROUTE_B[:4])
    p = b.p
    assert p.coef_seed.source == "run_card"            # seeded from the last run card via NANDI_ENERGY_CARD
    e_half, f_half = p.e_total_wh, _features(p)
    live = p.tracker.coefficients()
    _drive(b, ROUTE_B[4:])
    f_all = _features(p)
    integrated = p.e_total_wh
    # (1) Ready-check projection: run-card coefficients on the whole planned route (planned = walked metres)
    pred_start = p.coef_seed.energy_wh(f_all)
    # (2) mid-tour projection: energy so far + live coefficients (seed + this run) on the remaining route
    rem = re_.RoutePlan(route_m=f_all.route_m - f_half.route_m, t_work_s=f_all.t_work_s - f_half.t_work_s)
    pred_mid = e_half + live.energy_wh(rem)
    # (3) the honest mid-tour number: the projection of the REMAINING energy alone against what the remaining half
    # actually took (pred_mid adds the measured first half e_half as exact truth, which dilutes the error)
    rem_err = re_.estimator_error(live.energy_wh(rem), integrated - e_half)
    return (re_.estimator_error(pred_start, integrated), re_.estimator_error(pred_mid, integrated),
            dict(route_m=f_all.route_m, t_s=f_all.t_work_s, integrated=integrated, pred_start=pred_start,
                 pred_mid=pred_mid, rem_err=rem_err, p=p))


def test_estimator_within_15_percent_of_integrated_energy(card, flat_card, monkeypatch, capsys):
    err_start, err_mid, n = _score(card, monkeypatch)
    x_start, x_mid, _ = _score(flat_card, monkeypatch)
    c_a = card[1]
    th = re_.throughput(n["p"].tracker.coefficients(), n["p"].params.capacity_wh)
    with capsys.disabled():
        print(f"\n[WP7 estimator] card A (ridges, seed 7) coefficients {c_a['energy']['coefficients']}, "
              f"A self-error {c_a['energy']['estimator_error']:.4f}\n"
              f"[WP7 estimator] B (ridges, seed 11): {n['route_m']:.2f} m, {n['t_s']:.0f} s, integrated "
              f"{n['integrated']:.4f} Wh; start pred {n['pred_start']:.4f} Wh err {err_start:.4f}; mid pred "
              f"{n['pred_mid']:.4f} Wh err {err_mid:.4f}; remaining-half-only err {n['rem_err']:.4f}\n"
              f"[WP7 estimator] cross-terrain card (A on flat) -> B: start err {x_start:.4f}, mid err {x_mid:.4f}\n"
              f"[WP7 endurance] measured coefficients {th['coefficients']}: SCAN {th['ha_per_charge']} ha/charge "
              f"({th['scan_wh_per_ha']} Wh/ha, {th['scan_h_per_ha']} h/ha), TREE 360 {th['trees_per_charge']} "
              f"trees/charge ({th['tree360_wh_per_tree']} Wh/tree), {th['wh_per_100m']} Wh/100 m")
    # scope (review): features are the WALKED route (planning error excluded); run B is 5.95 m / 66 s on the legacy
    # ridges world, not orchard_lite. rem_err is the mid-tour number without the measured first half.
    assert err_start <= 0.15 and err_mid <= 0.15 and n["rem_err"] <= 0.15
    assert c_a["energy"]["estimator_error"] <= 0.15
    assert x_mid <= 0.15            # the in-run coefficients absorb a wrong-terrain card
    assert x_start <= 0.25          # regression guard only (no spec gate across terrains)
    assert th["ha_per_charge"] > 0 and th["trees_per_charge"] > 0


def test_hook_mid_tour_breach_triggers_return_and_wait(card):
    m = mb.build(NANDI_TERRAIN="flat", NANDI_CROPS=1)
    r = Run(m)
    r.run(2.0, vx=0.5)
    p = r.p
    r.bus.deliver("/nandi/platform/plan", {"route_m": 50.0, "items": 2, "t_work_s": 600.0, "return_m": 20.0})
    r.run(2.0, vx=0.5)
    assert p.reserve.action == re_.ACTION_CONTINUE
    bat = r.msgs("/nandi/platform/battery")[-1]
    assert bat["projected_end_soc"] is not None and bat["projected_end_soc"] > bat["reserve"]
    rem = p.remaining_plan()
    assert rem.route_m < 50.0 and rem.t_work_s == pytest.approx(600.0 - 2.0, abs=0.05)
    p.battery.energy_wh = 0.16 * p.params.capacity_wh     # pack drained (scripted) -> projection < reserve
    r.run(1.5, vx=0.5)
    alerts = [a for a in r.msgs("/nandi/platform/alert") if a["id"] == "low_battery_return"]
    assert len(alerts) == 1 and alerts[0]["reason"] == "energy_reserve"
    assert r.bus.values["platform"]["energy_action"] == re_.ACTION_RETURN_AND_WAIT
    r.bus.deliver("/nandi/platform/plan", {"clear": True})
    r.run(0.1)
    assert p.plan is None and p.reserve.action == re_.ACTION_CONTINUE


def test_hook_counts_items_and_crown_travel():
    m = mb.build(NANDI_TERRAIN="flat", NANDI_CROPS=1)
    r = Run(m)
    ext = {"v": 0.0}

    def drive(run):
        ext["v"] = min(1.6, ext["v"] + 0.2 * mb.CTRL_DT)   # WP24 would share this; 0.2 m/s ramp
        run.bus.values["mast"] = {"ext_m": ext["v"]}
        run.bus.values["platform_items"] = 3 if run.d.time > 1.0 else 0
    r.on_tick = drive
    r.run(4.0)
    p = r.p
    assert p.items == 3
    assert p.crown_m == pytest.approx(0.8 - 0.2 * mb.CTRL_DT, abs=0.01)
    e = r.msgs("/nandi/energy")[-1]
    assert e["items"] == 3 and e["crown_m"] == pytest.approx(p.crown_m, abs=1e-3)
