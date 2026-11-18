"""Energy model: power formula, SoC integration on a synthetic torque trace, route estimate."""
import math

import pytest

from atharv_nandi.core.energy_model import (
    ASSUMED_CAPACITY_WH, ASSUMED_WALK_POWER_W, Battery, EnergyParams, electrical_power_w, mechanical_power_w, route_soc_needed,
)


def test_defaults_are_the_documented_assumptions():
    p = EnergyParams()
    assert (p.capacity_wh, p.eta, p.p_idle_w, p.low_soc) == (500.0, 0.6, 15.0, 0.25)
    assert ASSUMED_CAPACITY_WH == 500.0  # 500 Wh pack (physics alignment v1; was 100)


def test_power_is_abs_tau_qdot_over_eta_plus_idle():
    p = EnergyParams()
    # +6 W, |-2 * 3| = 6 W, braking joints still cost (abs), NaN ignored
    assert mechanical_power_w([2.0, -2.0, 1.0], [3.0, 3.0, float("nan")]) == pytest.approx(12.0)
    assert electrical_power_w([2.0, -2.0], [3.0, 3.0], p) == pytest.approx(12.0 / 0.6 + 15.0)


def test_soc_integration_on_synthetic_torque_trace():
    """18 joints at tau = 5 sin(wt) Nm, qdot = 2 cos(wt) rad/s for 600 s at 50 Hz."""
    p = EnergyParams()
    b = Battery(p)
    dt, w = 0.02, 2 * math.pi * 1.8
    expected_wh, prev_w = 0.0, None
    socs = []
    for i in range(int(600 / dt) + 1):
        t = i * dt
        tau, qd = [5 * math.sin(w * t)] * 18, [2 * math.cos(w * t)] * 18
        pw = electrical_power_w(tau, qd, p)
        if prev_w is not None:  # zero-order hold, like Battery.update
            expected_wh += prev_w * dt / 3600.0
        prev_w = pw
        socs.append(b.update(t, pw))
    # mean |10 sin cos| = 10/pi per joint -> 18 * 3.183 / 0.6 + 15 = 110.5 W -> 18.4 Wh in 10 min
    assert expected_wh == pytest.approx((18 * 10 / math.pi / 0.6 + 15.0) * 600 / 3600, rel=0.01)
    assert b.energy_wh == pytest.approx(ASSUMED_CAPACITY_WH - expected_wh, abs=1e-6)
    assert all(b2 <= b1 for b1, b2 in zip(socs, socs[1:]))  # monotonic
    assert b.source == "joint_states"


def test_idle_only_fallback_and_flag():
    b = Battery(EnergyParams())
    for i in range(3601):
        b.update(i * 1.0, None)  # one hour without joint_states
    assert b.source == "idle_only"
    assert b.energy_wh == pytest.approx(ASSUMED_CAPACITY_WH - 15.0, abs=1e-6)
    st = b.status()
    assert set(st) >= {"soc", "power_w", "energy_wh", "est_remaining_min"}
    assert st["est_remaining_min"] == pytest.approx((ASSUMED_CAPACITY_WH - 15.0) / 15.0 * 60.0, rel=0.02)


def test_time_backwards_and_gaps_do_not_charge_or_burst():
    b = Battery(EnergyParams())
    b.update(10.0, 100.0)
    b.update(5.0, 100.0)   # clock went backwards: ignored
    assert b.energy_wh == ASSUMED_CAPACITY_WH
    b.update(1000.0, 100.0)  # sim paused for 990 s: at most 5 s are drained
    assert b.energy_wh == pytest.approx(ASSUMED_CAPACITY_WH - 100.0 * 5 / 3600)


def test_route_estimate_and_reserve():
    p = EnergyParams()
    need = route_soc_needed(12.0, 3, 160.0, p)
    walk = ASSUMED_WALK_POWER_W * (12.0 / 0.12) / 3600
    assert need == pytest.approx((walk + 0.75 + 15.0 * 160 / 3600) / ASSUMED_CAPACITY_WH + 0.20)


def test_params_from_dict_and_validation():
    p = EnergyParams.from_dict({"capacity_wh": 50, "eta": 0.5, "unknown": 3})
    assert p.capacity_wh == 50.0 and p.eta == 0.5 and p.p_idle_w == 15.0
    with pytest.raises(ValueError):
        EnergyParams(eta=0.0)


def test_site_sim_energy_reaches_the_mission_config():
    """Review F2: site_sim.yaml overrides the energy_model defaults through EnergyParams.from_dict, so the pack and
    walking-power values the running mission uses must be the aligned ones, not the stale 100 Wh / 60 W."""
    import os
    from atharv_nandi.core import energy_model as em
    from atharv_nandi.core import geo_frame as gf
    from atharv_nandi.core import mission_logic as ml
    origin, cfg = gf.load_site(os.path.join(os.path.dirname(__file__), "..", "config", "missions", "site_sim.yaml"))
    mc = ml.config_from_site(origin, cfg, None)
    assert mc.energy.capacity_wh == em.ASSUMED_CAPACITY_WH == 500.0
    assert mc.energy.walk_power_w == em.ASSUMED_WALK_POWER_W
    assert mc.energy.walk_speed_mps == em.ASSUMED_WALK_SPEED_MPS
    assert mc.capacity.max_kg == 6.5
