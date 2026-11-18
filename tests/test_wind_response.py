"""WP6 ordered wind response and tripod wind tipping (ORCHARD-MVP-SPEC 2.3 rule 4, 6; WP6).

Acceptance:
    gust     11 m/s at mid-stride, ext 1.6, flat and 3 deg (forced up by the harness): 6-foot stance <= 0.5 s, no fall,
             margin > 0 throughout the retract (``slow``)
    tipping  tripod-stance wind tipping speed with the crown up (measurement: config/crown_gates.json ``tipwind``)
The harness honours the hook's ``motion_inhibit.stability.stance`` request by stopping the gait (``gait.stop``), the
job locomotion does in the live stack (change request, the WP6 contract notes). The margin during the retract is the
6-foot margin with the wind moment (drag moment / (m g) added to the CoM), so the wind load is in it.
"""
import json
import math
import os
import sys

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import stability as st

import test_crown_gates as cg  # noqa: E402  (basin_world, place, set_cross_slope)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATES_JSON = os.path.join(REPO, "config", "crown_gates.json")
GUST_MPS = 11.0


def test_tripod_wind_tipping_speed_is_measured_and_above_the_walk_limit():
    with open(GATES_JSON, encoding="utf-8") as fh:
        tw = json.load(fh)["tipwind"]
    assert tw["meta"]["run_id"].startswith("wp6-tipwind-") and tw["ext"] == 1.6
    assert {r["dir_deg"] for r in tw["rows"]} == {0.0, 90.0, 180.0, 270.0}
    v = tw["tripod_tip_speed_min_mps"]
    print(f"\n[tipwind] tripod stance, ext 1.6: tip speed min {v} m/s (wind toward {tw['worst_dir_deg']} deg), "
          f"rows {[(r['dir_deg'], r['v_tip_static_mps'], r['v_tip_dynamic_step_mps']) for r in tw['rows']]}")
    assert v > st.WIND_CROWN_WALK_MAX_MPS                       # the 8 m/s crown-up walk limit sits below tipping
    for r in tw["rows"]:
        assert r["v_tip_dynamic_step_mps"] <= r["v_tip_static_mps"] + 1e-6       # a step load tips earlier
        assert not r["check_0p9"]["tipped"]


def _wind_margin(m, d, drag) -> float:
    """6-foot margin with the wind: the drag moment about the ground (sum f_xy h) shifts the effective CoM by
    M / (m g) (the quasi-static tipping criterion of tools/crown_gate_matrix.py tipwind), feet hull along gravity."""
    feet = [d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot")] for leg in st.LEG_NAMES]
    z0 = float(np.mean([p[2] for p in feet]))
    mx = my = 0.0
    for gid, f in (drag.last_forces or {}).items():
        h = float(d.geom_xpos[gid][2]) - z0
        mx += f[0] * h
        my += f[1] * h
    mass = float(m.body_subtreemass[1])
    g = float(np.linalg.norm(m.opt.gravity))
    R = st.gravity_frame(np.asarray(m.opt.gravity, float))
    com = R @ (d.subtree_com[1] + np.array([mx, my, 0.0]) / (mass * g))
    return st.com_margin(com[:2], st.convex_hull([(R @ p)[:2] for p in feet]))


def gust_case(cross_deg: float, dir_deg: float, seed: int = 7) -> dict:
    m, _ = cg.basin_world(seed, terrain="flat")
    cg.set_cross_slope(m, cross_deg)
    import atharv_nandi.sim.hooks as hook_registry
    from atharv_nandi.core import sim_driver as sd
    from atharv_nandi.sim.hooks import stability as hs
    drv = sd.SimDriver(m, hooks=hook_registry.load_all(names=["mast", "platform", "stability"]))
    cg.place(drv, 3.0, -1.0)
    drv.wait(1.0)
    drv.bus.values["mast_control"].arb.reset(1.6)            # forced up by the harness
    drv.wait(0.5)
    from atharv_nandi.sim.hooks import platform as hp
    plat = hp._STATE[id(m)]

    def honour_stance(dr):
        mi = (dr.bus.values.get("motion_inhibit") or {}).get("stability") or {}
        if mi.get("stance") and dr.gait.is_walking:
            dr.gait.stop()

    drv.add_task("wp6_stance", honour_stance, rate_hz=50.0)
    drv.walk(cg.WALK_VX)
    drv.wait(2.0)
    # mid-stride: wait for the gait phase to pass a quarter cycle (a tripod at the top of its swing)
    while (drv.gait.gait_phase % math.pi) < 0.5 * math.pi - 0.15 or (drv.gait.gait_phase % math.pi) > 0.5 * math.pi:
        drv.run_steps(1)
    t_gust = drv.t
    drv.deliver("/nandi/wind/cmd", {"speed_mps": 0.0, "dir_deg": dir_deg,
                                    "gusts": [{"t0": 0.0, "peak_mps": GUST_MPS, "rise_s": 0.2, "hold_s": 12.0,
                                               "fall_s": 1.0}]})
    t_req = t_stance = t_retract = None
    min_wind_m, max_tilt, fell = math.inf, 0.0, False
    exts, cop_trace = [], []
    while drv.t - t_gust < 11.0:
        drv.run_steps(10)
        s = drv.bus.values["stability"]
        ms = drv.bus.values["mast"]
        if t_req is None and s["stance_request"]:
            t_req = drv.t
        if t_req is not None and t_stance is None and len(s["contacts"]) == 6 and not drv.gait.is_walking:
            t_stance = drv.t
        if t_retract is None and ms["arbiter_source"] == "safety" and ms["target_ext"] <= 1e-6:
            t_retract = drv.t
        if t_retract is not None and ms["ext_m"] > 1e-6:
            cm = _wind_margin(m, drv.d, plat.drag)
            min_wind_m = min(min_wind_m, cm)
            cop_trace.append((round(drv.t - t_retract, 2), round(cm, 4), len(s["contacts"])))
        exts.append(ms["ext_m"])
        rel = max(abs(s["tilt_deg"]["roll"] - s["slope_deg"]["roll"]), abs(s["tilt_deg"]["pitch"] - s["slope_deg"]["pitch"]))
        max_tilt = max(max_tilt, rel)
        fell = fell or drv.d.qpos[2] < 0.15 or rel > 35.0
    w = drv.bus.values.get("wind") or {}
    out = {"cross": cross_deg, "dir_deg": dir_deg, "t_gust": round(t_gust, 3),
           "flag_after_s": None if t_req is None else round(t_req - t_gust, 3),
           "stance_after_request_s": None if (t_req is None or t_stance is None) else round(t_stance - t_req, 3),
           "retract_after_request_s": None if (t_req is None or t_retract is None) else round(t_retract - t_req, 3),
           "ext_end": round(exts[-1], 4), "min_wind_margin_during_retract_m": round(min_wind_m, 4), "wind_margin_low": sorted(cop_trace, key=lambda x: x[1])[:3],
           "max_rel_tilt_deg": round(max_tilt, 2), "fell": fell, "wind_peak_seen_mps": w.get("speed_mps"),
           "retracts": [e["reason"] for e in hs._RT[0].retract_log]}
    drv.close()
    return out


@pytest.mark.slow
@pytest.mark.parametrize("cross_deg", [0.0, 3.0])
def test_gust_at_mid_stride_ordered_stance_then_retract(cross_deg):
    rows = [gust_case(cross_deg, d) for d in (90.0, 180.0)]        # lateral (OS 2.3) and the worst tipping direction
    for r in rows:
        print(f"\n[gust] {r}")
    for r in rows:
        assert r["flag_after_s"] is not None and r["flag_after_s"] <= 0.5, r
        assert r["stance_after_request_s"] is not None and r["stance_after_request_s"] <= st.STANCE_WITHIN_S, r
        assert r["retract_after_request_s"] is not None and r["retract_after_request_s"] <= st.STANCE_WITHIN_S + 0.05, r
        assert not r["fell"], r
        assert r["min_wind_margin_during_retract_m"] > 0.0, r
        assert r["ext_end"] == pytest.approx(0.0, abs=1e-6), r
