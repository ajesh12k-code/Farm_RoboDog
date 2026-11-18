"""BT1 / BT2 / BT5 of the physics alignment spec v1: atharv_nandi/core/stability.py against closed forms.

Pure numpy, no MuJoCo. Every expected value is computed in closed form next to the assert.
"""
import json
import math

import numpy as np
import pytest

from atharv_nandi.core import stability as st

TOL = 1e-9


def regular_polygon(n, R, phase=0.0, z=0.0):
    return [(R * math.cos(phase + 2 * math.pi * k / n), R * math.sin(phase + 2 * math.pi * k / n), z) for k in range(n)]


def square(a, z=0.0):
    return [(-a, -a, z), (a, -a, z), (a, a, z), (-a, a, z)]


# --- BT1: analytic polygons --------------------------------------------------------------------
def test_hull_is_ccw_and_drops_interior_and_collinear_points():
    pts = [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2), (1, 1), (0, 1)]
    hull = st.convex_hull(pts)
    assert hull == [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    area = 0.5 * sum(hull[i][0] * hull[(i + 1) % 4][1] - hull[(i + 1) % 4][0] * hull[i][1] for i in range(4))
    assert area > 0  # counter-clockwise


@pytest.mark.parametrize("x0,y0", [(0.0, 0.0), (0.05, -0.02), (-0.11, 0.07)])
def test_square_margin_and_tip_closed_form(x0, y0):
    a, h = 0.25, 0.32
    poly2 = st.support_polygon([p[:2] for p in square(a)])
    margin = st.com_margin((x0, y0), poly2)
    assert margin == pytest.approx(a - max(abs(x0), abs(y0)), abs=TOL)
    tips = st.tip_angles((x0, y0, h), st.support_polygon_3d(square(a)))
    expect = sorted(math.degrees(math.atan2(d, h)) for d in (a - x0, a + x0, a - y0, a + y0))
    assert sorted(tips) == pytest.approx(expect, abs=TOL)
    assert st.min_tip_angle((x0, y0, h), st.support_polygon_3d(square(a))) == pytest.approx(expect[0], abs=TOL)


def test_square_com_outside_gives_negative_margin_and_tip():
    a, h = 0.25, 0.3
    poly2 = st.support_polygon([p[:2] for p in square(a)])
    assert st.com_margin((a + 0.1, 0.0), poly2) == pytest.approx(-0.1, abs=TOL)
    tip = st.min_tip_angle((a + 0.1, 0.0, h), st.support_polygon_3d(square(a)))
    assert tip == pytest.approx(math.degrees(math.atan2(-0.1, h)), abs=TOL)


def test_equilateral_triangle_centroid():
    R, h = 0.3, 0.33
    tri = regular_polygon(3, R, phase=0.4)
    poly2 = st.support_polygon([p[:2] for p in tri])
    assert len(poly2) == 3
    assert st.com_margin((0.0, 0.0), poly2) == pytest.approx(R / 2.0, abs=TOL)  # inradius = R cos(60 deg)
    tips = st.tip_angles((0.0, 0.0, h), st.support_polygon_3d(tri))
    assert tips == pytest.approx([math.degrees(math.atan2(R / 2.0, h))] * 3, abs=TOL)


def test_regular_hexagon_offset_com():
    R, h, dx = 0.26, 0.31, 0.04
    hexa = regular_polygon(6, R)  # vertices at 0, 60, ... deg: edges have apothem R sqrt(3)/2
    apothem = R * math.sqrt(3) / 2.0
    poly2 = st.support_polygon([p[:2] for p in hexa])
    assert len(poly2) == 6
    # edge normals at 30, 90, ..., 330 deg; CoM shifted +x by dx: d_i = apothem - dx cos(normal_i)
    d = [apothem - dx * math.cos(math.radians(30 + 60 * k)) for k in range(6)]
    assert st.com_margin((dx, 0.0), poly2) == pytest.approx(min(d), abs=TOL)
    tips = st.tip_angles((dx, 0.0, h), st.support_polygon_3d(hexa))
    assert sorted(tips) == pytest.approx(sorted(math.degrees(math.atan2(v, h)) for v in d), abs=TOL)


def test_tilted_gravity_reduces_tip_angle_by_the_tilt():
    """Square stance, CoM centred at height h, gravity tilted by th toward +x: the +x edge's tip angle
    is atan2(a, h) - th, the -x edge's atan2(a, h) + th (rigid rotation of the whole problem)."""
    a, h, th = 0.25, 0.3, math.radians(12.0)
    g = (math.sin(th), 0.0, -math.cos(th))
    edges = st.edge_tip_angles((0.0, 0.0, h), st.support_polygon_3d(square(a), g), g)
    by_dir = {tuple(np.round(e["outward"], 6)): e["tip_deg"] for e in edges}
    base = math.degrees(math.atan2(a, h))
    out_px = max(edges, key=lambda e: e["outward"][0])
    out_mx = min(edges, key=lambda e: e["outward"][0])
    assert out_px["tip_deg"] == pytest.approx(base - math.degrees(th), abs=1e-9)
    assert out_mx["tip_deg"] == pytest.approx(base + math.degrees(th), abs=1e-9)
    assert len(by_dir) == 4


def test_sloped_ground_rigid_rotation_matches_flat():
    """Rotate feet + CoM + gravity together: tip angles must not change."""
    a, h = 0.22, 0.3
    feet, com = square(a), (0.03, -0.02, h)
    flat = sorted(st.tip_angles(com, st.support_polygon_3d(feet)))
    c, s = math.cos(0.3), math.sin(0.3)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    feet_r = [Rx @ np.array(p) for p in feet]
    g_r = Rx @ np.array([0, 0, -1.0])
    rot = sorted(st.tip_angles(Rx @ np.array(com), st.support_polygon_3d(feet_r, g_r), g_r))
    assert rot == pytest.approx(flat, abs=1e-9)


@pytest.mark.parametrize("pts", [[], [(0, 0)], [(0, 0), (1, 0)], [(0, 0), (0.5, 0), (1, 0)], [(1, 1)] * 4])
def test_degenerate_polygons_have_no_margin_and_are_unsupported(pts):
    poly = st.support_polygon(pts)
    assert len(poly) < 3
    assert st.com_margin((0.5, 0.0), poly) == -math.inf
    assert st.tip_angles((0, 0, 0.3), [(*p, 0.0) for p in poly]) == []
    assert st.min_tip_angle((0, 0, 0.3), [(*p, 0.0) for p in poly]) == -math.inf
    assert st.infer_mode(len(pts) if len(poly) >= 3 else min(len(pts), 2), None, [0.1] * 6) == st.MODE_UNSUPPORTED
    v = st.classify(st.MODE_UNSUPPORTED, {"n_contacts": len(pts)})
    assert not v["ok"] and st.INHIBIT_WALK in v["inhibit"] and st.INHIBIT_ARM_PICK in v["inhibit"]


def test_head_push_to_tip_closed_form():
    assert st.head_push_to_tip(30.0, 0.2, 2.5, 0.0) == pytest.approx(30.0 * 9.81 * 0.2 / 2.5, abs=TOL)
    assert st.head_push_to_tip(30.0, -0.01, 2.5, 0.0) == 0.0
    a, h, zh = 0.25, 0.3, 2.2
    com = (0.05, 0.0, h)  # nearest edge is +x at d = a - 0.05
    f = st.min_head_push_to_tip(31.8, com, st.support_polygon_3d(square(a)), (0.03, 0.0, zh))
    assert f == pytest.approx(31.8 * 9.81 * (a - 0.05) / zh, abs=TOL)


def test_cop_is_force_weighted_mean():
    pts = [(0, 0, 0), (1, 0, 0), (0, 2, 0)]
    c = st.cop(pts, [1.0, 1.0, 2.0])
    assert c.tolist() == pytest.approx([0.25, 1.0, 0.0], abs=TOL)
    assert st.cop(pts, [0, 0, 0]) is None


# --- BT2: slope from foot points vs IMU quaternion ---------------------------------------------
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def axis_quat(axis, ang):
    v = np.asarray(axis, dtype=float) * math.sin(ang / 2.0)
    return np.array([math.cos(ang / 2.0), *v])


def body_quat(yaw_deg, pitch_up_deg, roll_left_deg):
    q = quat_mul(axis_quat((0, 0, 1), math.radians(yaw_deg)), axis_quat((0, 1, 0), -math.radians(pitch_up_deg)))
    return quat_mul(q, axis_quat((1, 0, 0), math.radians(roll_left_deg)))


FEET_BODY = [(0.26, -0.243, -0.31), (0.26, 0.243, -0.31), (0.0, -0.253, -0.31),
             (0.0, 0.253, -0.31), (-0.26, -0.243, -0.31), (-0.26, 0.243, -0.31)]


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (0, 7, 0), (0, 0, -4), (35, 5, 3), (-120, -9, 8), (170, 3, -10)])
def test_slope_points_and_imu_agree(yaw, pitch, roll):
    q = body_quat(yaw, pitch, roll)
    R = st.quat_to_mat(q)
    rng = np.random.default_rng(abs(yaw) + 7)
    pts = [R @ np.array(p) + np.array([1.0, -2.0, 0.31]) + rng.normal(0, 0.001, 3) for p in FEET_BODY]  # 1 mm noise
    sp = st.slope_from_points(pts, st.yaw_from_quat(q))
    si = st.slope_from_quat(q)
    assert sp[0] == pytest.approx(si[0], abs=0.5) and sp[1] == pytest.approx(si[1], abs=0.5)
    if roll == 0:
        assert si[0] == pytest.approx(pitch, abs=1e-9) and si[1] == pytest.approx(0.0, abs=1e-9)
    if pitch == 0:
        assert si[1] == pytest.approx(roll, abs=1e-9) and si[0] == pytest.approx(0.0, abs=1e-9)


def test_slope_from_points_sign_convention_and_degenerate():
    # plane z = 0.1 x (nose up when heading +x), z = -0.05 y (left side down)
    pts = [(x, y, 0.1 * x - 0.05 * y) for x in (-0.3, 0.0, 0.3) for y in (-0.25, 0.25)]
    p, r = st.slope_from_points(pts, 0.0)
    assert p == pytest.approx(math.degrees(math.atan(0.1)), abs=1e-9)
    assert r == pytest.approx(math.degrees(math.atan(-0.05)), abs=1e-9)
    # heading -x (yaw pi): the same hill is nose down
    p2, r2 = st.slope_from_points(pts, math.pi)
    assert p2 == pytest.approx(-p, abs=1e-9) and r2 == pytest.approx(-r, abs=1e-9)
    assert st.slope_from_points([(0, 0, 0), (1, 1, 1)], 0.0) is None
    assert st.slope_from_points([(0, 0, 0), (1, 0, 0), (2, 0, 0)], 0.0) is None


# --- modes / classify ------------------------------------------------------------------------
def test_infer_mode():
    assert st.infer_mode(6, None, [0.12] * 6) == st.MODE_STAND
    assert st.infer_mode(6, 0.8, [0.12] * 6) == st.MODE_STAND
    assert st.infer_mode(6, None, [0.45, -0.45] * 3) == st.MODE_SCAN_WIDE
    assert st.infer_mode(3, None, [0.12] * 6) == st.MODE_TRIPOD_WALK
    assert st.infer_mode(6, 2.2, [0.45] * 6) == st.MODE_MAST_UP
    assert st.infer_mode(2, 2.2, [0.45] * 6) == st.MODE_UNSUPPORTED


def test_classify_stand_ok_and_low_margin():
    ok = st.classify(st.MODE_STAND, {"n_contacts": 6, "margin_m": 0.22, "tip_angle_deg": 35, "pitch_deg": 0, "roll_deg": 0})
    assert ok == {"ok": True, "inhibit": [], "reasons": []}
    bad = st.classify(st.MODE_STAND, {"n_contacts": 6, "margin_m": 0.09, "tip_angle_deg": 14})
    assert not bad["ok"] and st.INHIBIT_ARM_PICK in bad["inhibit"] and st.INHIBIT_MAST_UP in bad["inhibit"]


def test_classify_tripod_limits():
    v = st.classify(st.MODE_TRIPOD_WALK, {"n_contacts": 3, "margin_m": 0.06, "tip_angle_deg": 10.0})
    assert v["ok"]
    v = st.classify(st.MODE_TRIPOD_WALK, {"n_contacts": 3, "margin_m": 0.04, "tip_angle_deg": 9.0})
    assert st.INHIBIT_WALK in v["inhibit"] and len(v["reasons"]) == 2


def test_classify_slopes():
    v = st.classify(st.MODE_STAND, {"n_contacts": 6, "pitch_deg": 7.0, "roll_deg": 0.0})
    assert v["inhibit"] == [st.INHIBIT_WALK_FORWARD_UPHILL]
    v = st.classify(st.MODE_STAND, {"n_contacts": 6, "pitch_deg": -7.0, "roll_deg": 11.0})
    assert v["inhibit"] == [st.INHIBIT_WALK]


def test_classify_mast_up():
    base = {"n_contacts": 6, "margin_m": 0.15, "pitch_deg": 2, "roll_deg": 1, "basket_kg": 5, "mast_h_m": 2.2,
            "head_kg": 1.5, "head_push_to_tip_n": 30.0}
    v = st.classify(st.MODE_MAST_UP, base)
    assert v["inhibit"] == [st.INHIBIT_WALK]  # stance only
    for k, val, flag in (("margin_m", 0.07, st.INHIBIT_MAST_UP), ("roll_deg", 6.0, st.INHIBIT_MAST_UP),
                         ("basket_kg", 11.0, st.INHIBIT_MAST_UP), ("head_kg", 2.0, st.INHIBIT_MAST_UP),
                         ("head_push_to_tip_n", 14.0, st.INHIBIT_MAST_UP), ("n_contacts", 5, st.INHIBIT_MAST_UP),
                         ("head_force_n", 15.0, st.INHIBIT_MAST_TRIP), ("mast_base_moment_nm", 40.0, st.INHIBIT_MAST_TRIP)):
        v = st.classify(st.MODE_MAST_UP, {**base, k: val})
        assert flag in v["inhibit"], k
    # 3.0 kg head is allowed in the wide stance on <= 5 deg, not on 6 deg
    assert st.INHIBIT_MAST_UP not in st.classify(st.MODE_MAST_UP, {**base, "head_kg": 3.0, "stance_wide": True})["inhibit"]
    assert st.INHIBIT_MAST_UP in st.classify(st.MODE_MAST_UP, {**base, "head_kg": 3.0, "stance_wide": True,
                                                               "pitch_deg": 6.0})["inhibit"]


def test_classify_basket_and_unknown_mode():
    v = st.classify(st.MODE_STAND, {"n_contacts": 6, "basket_kg": 6.5, "basket_limit_kg": 6.48})
    assert v["inhibit"] == [st.INHIBIT_BASKET_LOAD]
    with pytest.raises(ValueError):
        st.classify("FLYING", {})


# --- BT5: /nandi/stability JSON ------------------------------------------------------------------
def _status(**over):
    kw = dict(t=12.3456, mode=st.MODE_STAND, n_contacts=6, com=[-0.04, 0.0, 0.312], margin_m=0.2204,
              tip_angle_deg=35.14, pitch_deg=0.5, roll_deg=-0.2, head_push_to_tip_n=None, basket_kg=0.3,
              basket_limit_kg=6.48, basket_binding="volume", arm_reach_m=0.09, arm_limit_kg=2.0,
              leg_rms_nm_max=12.5, inhibit=[], reasons=[])
    kw.update(over)
    return st.make_status(**kw)


@pytest.mark.parametrize("over", [{}, {"mode": st.MODE_MAST_UP, "head_push_to_tip_n": 41.2, "inhibit": ["walk"],
                                        "reasons": ["mast up"]},
                                   {"mode": st.MODE_UNSUPPORTED, "n_contacts": 2, "margin_m": -math.inf,
                                    "tip_angle_deg": float("nan"), "com": None, "arm_reach_m": None}])
def test_status_round_trip_and_finite(over):
    s = _status(**over)
    text = st.build_stability_status(s)
    assert st.parse_stability_status(text) == s
    assert set(s) == set(st.STATUS_KEYS)
    assert s["advisory"] is True and s["enforced"] is False

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, float):
            assert math.isfinite(o)
    walk(json.loads(text))
    if over.get("mode") == st.MODE_UNSUPPORTED:
        assert s["margin_m"] is None and s["tip_angle_deg"] is None


def test_status_rejects_bad_payloads():
    s = _status()
    bad = dict(s)
    bad["margin_m"] = math.inf
    with pytest.raises(ValueError):
        st.build_stability_status(bad)
    missing = dict(s)
    del missing["inhibit"]
    with pytest.raises(ValueError):
        st.build_stability_status(missing)
    assert st.parse_stability_status("not json") is None
    assert st.parse_stability_status(json.dumps({**s, "mode": "FLYING"})) is None
    assert st.parse_stability_status('{"t": 1}') is None
    assert st.parse_stability_status(json.dumps(s).replace("0.2204", "NaN")) is None


# =================================================================================================
# WP6: predicted tripod margin, gate table, gates, stance set, crown supervisor (ORCHARD-MVP-SPEC 2.3, 5.2)
# =================================================================================================
FEET6 = {"FR": (0.3, -0.25, 0.0), "FL": (0.3, 0.25, 0.0), "MR": (0.0, -0.3, 0.0), "ML": (0.0, 0.3, 0.0),
         "RR": (-0.3, -0.25, 0.0), "RL": (-0.3, 0.25, 0.0)}


def _tri_margin_closed_form(com_xy, a, b, c):
    """Signed distance of com to triangle abc (inside +): min over edges of the perpendicular distance."""
    pts = [a, b, c]
    out = math.inf
    for i in range(3):
        p, q = np.array(pts[i][:2]), np.array(pts[(i + 1) % 3][:2])
        e = q - p
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        r = pts[(i + 2) % 3][:2]
        if np.dot(np.array(r) - p, n) < 0:
            n = -n
        out = min(out, float(np.dot(np.array(com_xy[:2]) - p, n)))
    return out


def test_predicted_tripod_margin_flat_is_the_closed_form_min_of_both_tripods():
    com = (0.02, 0.01, 0.30)
    r = st.predicted_tripod_margin(com, FEET6, (0, 0, -9.81))
    a = _tri_margin_closed_form(com, FEET6["FR"], FEET6["ML"], FEET6["RR"])
    b = _tri_margin_closed_form(com, FEET6["FL"], FEET6["MR"], FEET6["RL"])
    assert r["A"] == pytest.approx(a, abs=1e-9) and r["B"] == pytest.approx(b, abs=1e-9)
    assert r["margin_m"] == pytest.approx(min(a, b), abs=1e-9)


def test_predicted_margin_assumes_the_rack_side_downhill():
    """Gravity seen from the body toward +y (rack uphill) is mirrored to -y: the worst case. h tan(a) shift."""
    com = (0.0, 0.0, 0.30)
    a = math.radians(5.0)
    g_up = (0.0, 9.81 * math.sin(a), -9.81 * math.cos(a))        # rack (-y) side uphill
    g_dn = (0.0, -9.81 * math.sin(a), -9.81 * math.cos(a))       # rack side downhill
    wc_up = st.predicted_tripod_margin(com, FEET6, g_up)
    wc_dn = st.predicted_tripod_margin(com, FEET6, g_dn)
    raw_up = st.predicted_tripod_margin(com, FEET6, g_up, worst_case=False)
    assert wc_up["margin_m"] == pytest.approx(wc_dn["margin_m"], abs=1e-12)
    assert wc_up["gravity_b"][1] < 0
    # the CoM projection moves 0.30 tan(5 deg) toward -y: the right-side edge of tripod B (MR) loses that much
    flat = st.predicted_tripod_margin(com, FEET6, (0, 0, -1))
    assert flat["margin_m"] - wc_dn["margin_m"] == pytest.approx(0.30 * math.tan(a), abs=0.02)
    assert raw_up["margin_m"] >= wc_up["margin_m"] - 1e-12


def test_predicted_margin_takes_the_worse_side_when_the_load_is_off_the_rack_side():
    """A CoM well left (+y, an arm reaching left with a fruit): rack uphill is the worst case; the prediction is
    never more optimistic than the stance it measures, and reports which side binds."""
    a = math.radians(5.0)
    g_dn = (0.0, -9.81 * math.sin(a), -9.81 * math.cos(a))       # rack (-y) side downhill
    left = (0.0, 0.08, 0.30)
    wc = st.predicted_tripod_margin(left, FEET6, g_dn)
    raw_dn = st.predicted_tripod_margin(left, FEET6, g_dn, worst_case=False)
    raw_up = st.predicted_tripod_margin(left, FEET6, (0.0, -g_dn[1], g_dn[2]), worst_case=False)
    assert raw_up["margin_m"] < raw_dn["margin_m"]
    assert wc["margin_m"] == pytest.approx(raw_up["margin_m"], abs=1e-12) and wc["side"] == "rack_uphill"
    right = (0.0, -0.02, 0.30)
    assert st.predicted_tripod_margin(right, FEET6, g_dn)["side"] == "rack_downhill"
    # terrain-roll mirror: same rule, the trunk's own roll kept
    for roll in (-4.0, 4.0):
        r = math.radians(roll)
        g = (0.0, 9.81 * math.sin(r), -9.81 * math.cos(r))        # roll > 0 = left up = gravity toward -y
        g = (0.0, -g[1], g[2])
        w = st.predicted_tripod_margin(left, FEET6, g, terrain_roll_deg=roll)
        assert w["margin_m"] <= st.predicted_tripod_margin(left, FEET6, g, worst_case=False)["margin_m"] + 1e-12
        assert w["side"] == "rack_uphill"


def test_predicted_margin_arm_park_shift_and_hexa():
    com = (0.0, 0.0, 0.3)
    base = st.predicted_tripod_margin(com, FEET6)["margin_m"]
    moved = st.predicted_tripod_margin(com, FEET6, com_shift_b=(0.05, 0.0, 0.0))["margin_m"]
    assert moved < base
    # nearest hexagon edge: FL (0.3, 0.25) - ML (0, 0.3): |cross| / length = 0.09 / sqrt(0.0925)
    assert st.hexa_margin(com, FEET6) == pytest.approx(0.09 / math.sqrt(0.0925), abs=1e-9)
    assert st.hexa_margin((0.0, 0.0, 0.3), FEET6, com_shift_b=(0.29, 0.0, 0.0)) == pytest.approx(0.01, abs=1e-9)


def test_mast_com_shift_nested_sections():
    # sections 0.15/0.13/0.12 move 1/3, 2/3, 3/3 of dext; head 0.4 moves all of it
    dz = st.mast_com_shift(1.6, (0.15, 0.13, 0.12), 0.4, 38.5)
    moved = 0.15 / 3 + 0.13 * 2 / 3 + 0.12 + 0.4
    assert dz[2] == pytest.approx(moved * 1.6 / 38.5, abs=1e-12) and dz[0] == 0.0


def test_gate_table_interpolates_and_falls_back(tmp_path):
    env = {"PARK": {"0": {"0": 0.10, "5": 0.06, "10": 0.02}, "1.6": {"0": 0.09, "5": 0.05, "10": 0.01}},
           "T4": {"0": {"0": 0.11, "5": 0.07, "10": 0.03}, "1.6": {"0": 0.10, "5": 0.04, "10": 0.02}}}
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"matrix": {"envelope": env, "meta": {"run_id": "wp6-matrix-x"}}}))
    t = st.CrownGateTable.load(str(p))
    assert t.run_id == "wp6-matrix-x"
    assert t.margin(0.0, 2.5, "PARK") == pytest.approx(0.08, abs=1e-12)
    assert t.margin(0.8, 0.0, "PARK") == pytest.approx(0.095, abs=1e-12)
    assert t.margin(1.6, 5.0) == pytest.approx(0.04, abs=1e-12)            # worst arm: T4 at 5 deg
    assert t.margin(1.6, 12.0, "PARK") == pytest.approx(-0.006, abs=1e-12)  # extrapolated, falling
    assert t.max_tilt(1.6, 0.04, "PARK") == pytest.approx(6.2, abs=0.11)
    fb = st.CrownGateTable.load(str(tmp_path / "missing.json"))
    assert fb.run_id == "" and fb.margin(1.6, 5.0) == pytest.approx(0.030, abs=1e-12)


def test_shipped_gate_table_is_the_wp6_matrix():
    t = st.CrownGateTable.load()
    assert t.run_id.startswith("wp6-matrix-")
    # OS 2.3: flat crown-up tripod margin >= 0.04 m at the full layout -> crown-up walking is not dropped
    assert t.margin(1.6, 0.0) >= st.WALK_START_TRIPOD_M
    assert t.margin(1.6, st.CROWN_WALK_MAX_SLOPE_DEG) >= st.WALK_START_TRIPOD_M


def test_gate_raise_rule_1():
    ok = st.gate_raise(0.0, 1.6, 0.20, 1.0, 5.0, False, True, 6)
    assert ok and ok.gate == st.GATE_RAISE
    assert st.gate_raise(1.6, 0.0, -1.0, 9.0, 30.0, True, False, 0)          # a retract is never gated
    cases = [((0.0, 1.6, 0.12, 1.0, 5.0, False, True, 6), st.R_STABILITY),
             ((0.0, 1.6, 0.20, 3.5, 5.0, False, True, 6), st.R_SLOPE),
             ((0.0, 1.6, 0.20, 1.0, 10.0, False, True, 6), st.R_WIND),
             ((0.0, 1.6, 0.20, 1.0, 5.0, True, True, 6), st.R_WIND),
             ((0.0, 1.6, 0.20, 1.0, 5.0, False, False, 6), st.R_STANCE),
             ((0.0, 1.6, 0.20, 1.0, 5.0, False, True, 5), st.R_STANCE)]
    for args, reason in cases:
        dec = st.gate_raise(*args)
        assert not dec and dec.reason == reason, (args, dec)
    assert st.gate_raise(0.0, 1.6, 0.20, 1.0, 5.0, False, True, 6, estop=True).reason == st.R_ESTOP
    assert st.gate_raise(0.0, 1.6, 0.20, 1.0, 5.0, False, True, 6, tilt_deg=6.0).reason == st.R_SLOPE


def test_gate_crown_walk_rule_2():
    assert st.gate_crown_walk(1.6, 0.045, 0.15, 5.0, False, 1.0)
    assert st.gate_crown_walk(0.0, -1.0, 0.6, 20.0, True, 9.0)              # stowed: rule 2 does not apply
    assert st.gate_crown_walk(1.6, 0.039, 0.15, 5.0, False, 1.0).reason == st.R_STABILITY
    assert st.gate_crown_walk(1.6, 0.035, 0.15, 5.0, False, 1.0, starting=False)    # continue gate 0.03
    assert st.gate_crown_walk(1.6, 0.029, 0.15, 5.0, False, 1.0, starting=False).reason == st.R_STABILITY
    assert st.gate_crown_walk(1.6, 0.09, 0.20, 5.0, False, 1.0).reason == st.R_SPEED
    assert st.gate_crown_walk(1.6, 0.09, 0.15, 8.0, False, 1.0).reason == st.R_WIND
    assert st.gate_crown_walk(1.6, 0.09, 0.15, 2.0, True, 1.0).reason == st.R_WIND
    assert st.gate_crown_walk(1.6, 0.09, 0.15, 2.0, False, 2.5).reason == st.R_SLOPE


def test_slope_dwell_and_static_gates():
    assert st.gate_walk_slope(5.0, 0.0) and not st.gate_walk_slope(5.1, 0.0)
    assert not st.gate_walk_slope(4.0, 4.0)                                 # 5.6 deg tilt on the fall line
    assert st.gate_dwell(5.0, 20.0) and not st.gate_dwell(5.0, 60.0) and st.gate_dwell(3.0, 60.0)
    assert st.gate_static_margin(0.03) and not st.gate_static_margin(0.029)
    assert st.gate_static_margin(0.01, "tool_change").gate == "static_margin:tool_change"


def test_check_step_flags_the_wp27_seeded_contradictions():
    t = st.CrownGateTable.load()
    assert st.check_step(1.6, 0.15, 0.0, table=t) == []
    assert st.check_step(1.6, 0.15, 2.0, table=t) == []
    bad = st.check_step(1.6, 0.15, 5.0, table=t)                            # crown-up walk on 5 deg
    assert bad and bad[0].reason == st.R_SLOPE
    bad = st.check_step(1.6, 0.20, 0.0, table=t)                            # 0.20 m/s crown-up
    assert bad and bad[0].reason == st.R_SPEED
    assert st.check_step(0.0, 0.6, 5.0, table=t) == []                      # stowed crossing of the 5 deg strip
    assert st.check_step(0.0, 0.3, 10.0, table=t)[0].gate == st.GATE_SLOPE  # 10 deg patch refused
    assert st.check_step(0.0, 0.0, 4.0, walking=False, dwell_s=60.0, table=t)[0].gate == st.GATE_DWELL
    assert st.check_step(0.0, 0.0, 4.0, walking=False, raise_to=1.6, table=t)[0].reason == st.R_SLOPE
    assert st.check_step(0.0, 0.0, 2.0, walking=False, raise_to=1.6, table=t) == []


def test_stance_set_60ms_hysteresis():
    s = st.StanceSet()
    assert s.update(0.0, ["FR", "FL"]) == {"FR", "FL"}
    assert s.update(0.05, ["FR"]) == {"FR", "FL"}            # FL left 50 ms ago: still in
    assert s.update(0.061, ["FR"]) == {"FR"}
    assert s.update(0.07, [], flags=["ML"]) == {"FR", "ML"}  # gait flag counts as stance


def test_crown_walk_gate_hysteresis_and_filter():
    g = st.CrownWalkGate()
    assert not g.start(0.039)
    assert g.start(0.041)
    assert g.update(0.0, 0.031)                 # above the continue gate
    assert g.update(0.1, 0.025)                 # below, but not for 0.3 s yet
    assert g.update(0.35, 0.025)
    assert not g.update(0.40, 0.025)            # 0.3 s below 0.03 -> fails
    assert not g.update(0.5, 0.05)              # stays failed until a new start
    g.start(0.05)
    assert g.update(0.0, 0.02) and g.update(0.2, 0.031) and g.update(0.45, 0.02)   # a good sample resets the filter


def test_tilt_filter_is_a_half_second_mean_relative_to_the_slope():
    f = st.TiltFilter()
    for k in range(26):
        v = f.update(k * 0.02, 9.0 if k % 2 else 3.0, 0.0, slope_roll_deg=0.0)
    assert v == pytest.approx(6.0, abs=0.25) and f.full(0.5)
    f.reset()
    for k in range(26):
        v = f.update(k * 0.02, 9.0, 0.0, slope_roll_deg=5.0)
    assert v == pytest.approx(4.0, abs=1e-9)


def _sup_run(sup, t0, t1, **kw):
    out = []
    t = t0
    while t <= t1 + 1e-9:
        out.append((t, sup.update(t, **kw)))
        t = round(t + 0.02, 6)
    return out


BASE = dict(ext=1.6, walking=True, six_foot_stance=False, tripod_m=0.06, roll_deg=1.0, pitch_deg=1.0)


def test_crown_supervisor_ordered_wind_response_walking():
    sup = st.CrownSupervisor()
    _sup_run(sup, 0.0, 1.0, **BASE)
    r = sup.update(1.02, **{**BASE, "wind_mps": 8.5})
    assert r["stance_request"] and r["retract"] is None and r["wind_state"] == st.W_STANCE
    assert "wind_stance" in r["walk_inhibit"]
    r = sup.update(1.20, **{**BASE, "wind_mps": 8.5, "walking": False, "six_foot_stance": True})
    assert r["retract"] == "wind" and r["wind_state"] == st.W_RETRACT              # stance reached -> retract
    r = sup.update(1.22, **{**BASE, "wind_mps": 8.5, "walking": False, "six_foot_stance": True, "retracting": True})
    assert r["retract"] is None and r["walk_inhibit"]          # the retract runs: no new event, walking inhibited
    r = sup.update(1.24, **{**BASE, "wind_mps": 8.5, "walking": False, "six_foot_stance": True})
    assert r["retract"] == "wind"                              # not running (refused, e-stop hold): sent again
    r = sup.update(9.0, **{**BASE, "ext": 0.0, "walking": False, "six_foot_stance": True})
    assert r["wind_state"] == st.W_IDLE and not r["walk_inhibit"]                    # stowed: walking allowed


def test_crown_supervisor_wind_retracts_after_half_a_second_without_stance():
    sup = st.CrownSupervisor()
    sup.update(0.0, **{**BASE, "gust": True})
    assert sup.update(0.48, **{**BASE, "gust": True})["retract"] is None
    assert sup.update(0.50, **{**BASE, "gust": True})["retract"] == "gust"


def test_crown_supervisor_standing_thresholds():
    stand = {**BASE, "walking": False, "six_foot_stance": True}
    sup = st.CrownSupervisor()
    assert sup.update(0.0, **{**stand, "wind_mps": 9.0})["retract"] is None        # < 10 standing is fine
    assert sup.update(0.02, **{**stand, "wind_mps": 10.0})["retract"] == "wind"


def test_crown_supervisor_rule_5_and_margin_triggers():
    sup = st.CrownSupervisor()
    assert sup.update(0.0, **{**BASE, "pitch_deg": 12.5})["retract"] == "tilt"
    sup = st.CrownSupervisor()
    runs = _sup_run(sup, 0.0, 0.6, **{**BASE, "roll_deg": 7.0})
    hits = [t for t, r in runs if r["retract"] == "tilt_filtered"]
    assert hits and hits[0] >= 0.47                                             # only once the window is full
    sup = st.CrownSupervisor()
    sup.update(0.0, **BASE)
    runs = _sup_run(sup, 0.02, 0.5, **{**BASE, "tripod_m": 0.025})
    hits = [t for t, r in runs if r["retract"] == "margin"]
    assert hits and hits[0] == pytest.approx(0.32, abs=0.021)
    sup = st.CrownSupervisor()
    assert sup.update(0.0, **{**BASE, "tripod_m": 0.035})["retract"] == "margin"  # start gate 0.04
    for kw, reason in (({"local_link_ok": False}, "local_link_lost"), ({"low_battery_return": True},
                                                                       "low_battery_return")):
        sup = st.CrownSupervisor()
        assert sup.update(0.0, **{**BASE, **kw})["retract"] == reason
    sup = st.CrownSupervisor()
    assert sup.update(0.0, **{**BASE, "ext": 0.0, "pitch_deg": 20.0})["retract"] is None   # stowed: nothing


def test_crown_supervisor_reraise_once_after_30_s():
    sup = st.CrownSupervisor()
    stand = {**BASE, "walking": False, "six_foot_stance": True}
    assert sup.update(0.0, **{**stand, "pitch_deg": 13.0})["retract"] == "tilt"
    assert sup.reraise_due(20.0, True, True) is None
    assert sup.reraise_due(31.0, False, True) is None
    assert sup.reraise_due(31.0, True, True) == pytest.approx(1.6)
    assert sup.reraise_due(40.0, True, True) is None                            # once
    sup.update(41.0, **{**stand, "local_link_ok": False})
    assert sup.reraise_due(80.0, True, True) is None                            # link loss: never by itself


def test_stability_latch_only_stopped_or_work():
    lt = st.StabilityLatch()
    assert not lt.update(0.0, "WALK", -0.1, 3)
    assert not lt.update(1.0, "WALK", -0.1, 3)                                  # walking: reported, not latched
    assert not lt.update(2.0, "WORK", 0.005, 2)                                 # degenerate: not gated
    assert not lt.update(3.0, "WORK", 0.005, 6)
    assert lt.update(3.31, "WORK", 0.005, 6)
    assert not lt.reset(False) and lt.reset(True)


def test_basin_keepout_and_crown_path_gate():
    r = st.basin_keepout_radius(0.7)
    assert r == pytest.approx(0.7 + 0.15 + math.pi * 0.10 / (2 * math.tan(math.radians(30))), abs=1e-12)
    basins = [(2.0, 2.5, 0.7)]
    across = [(x, 2.5 - 0.775) for x in np.linspace(0.5, 3.5, 31)]
    beside = [(x, 2.5 - (r + st.CROWN_PATH_HULL_R_M + 0.05)) for x in np.linspace(0.5, 3.5, 31)]
    assert not st.gate_crown_path(1.6, across, basins) and st.gate_crown_path(1.6, beside, basins)
    assert st.gate_crown_path(0.0, across, basins)                       # stowed: rims may be crossed
