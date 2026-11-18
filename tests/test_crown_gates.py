"""WP6 crown gate matrix and crown walk (ORCHARD-MVP-SPEC 2.3, 6; WP6).

Acceptance:
    matrix   ext {0, 0.6, 1.6} x cross {0, 2, 3, 5} deg both directions x pitch {0, 5, 10} deg (both signs: the WP0c
             finding that rear-down pitch binds), full layout (rack downhill and uphill, 3 kg basket, arm PARK and the
             worst T4 pose) -> table written with run id (config/crown_gates.json, tools/crown_gate_matrix.py);
             6-foot margin > 0.13 m; raise and walk gates as tabled
    walk     6 s at 0.15 m/s, ext 0.6 and 1.6, cross 0 and 2 deg, rough + basins, seeds 7/11/23: no crown
             auto-retract and predicted tripod margin >= 0.03 m throughout (``slow``; failure -> stop-and-go default)
Plus the live WP24 integration of the WP6 raise gate.
"""
import contextlib
import json
import math
import os
import sys

import mujoco
import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.core import farm_scene
from atharv_nandi.core import sim_driver as sd
from atharv_nandi.core import stability as st
from atharv_nandi.core.interfaces import MastRequest
from atharv_nandi.sim.world import compose

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GATES_JSON = os.path.join(REPO, "config", "crown_gates.json")
WORLD_ENV = ("NANDI_WORLD", "NANDI_TERRAIN", "NANDI_SEED", "NANDI_CROPS", "NANDI_MAST_H", "NANDI_MAST_HEAD_KG",
             "NANDI_DT", "NANDI_PAYLOAD_KG", "NANDI_BUND_H", "NANDI_ACTORS")


@pytest.fixture(scope="module")
def gates():
    with open(GATES_JSON, encoding="utf-8") as fh:
        return json.load(fh)


# ====================================================================== the matrix table
def test_matrix_covers_the_acceptance_grid_with_a_run_id(gates):
    mx = gates["matrix"]
    assert mx["meta"]["run_id"].startswith("wp6-matrix-") and mx["meta"]["mujoco"]
    rows = mx["rows"]
    keys = {(r["arm"], r["ext"], r["roll_deg"], r["pitch_deg"]) for r in rows}
    for arm in ("PARK", "T4"):
        for ext in (0.0, 0.6, 1.6):
            for roll in (0.0, 2.0, -2.0, 3.0, -3.0, 5.0, -5.0):
                for pitch in (0.0, 5.0, 10.0, -5.0, -10.0):
                    assert (arm, ext, roll, pitch) in keys, (arm, ext, roll, pitch)
    assert {r["rack"] for r in rows} >= {"downhill", "uphill", "level"}
    assert mx["layout"]["basket_kg"] == 3.0 and mx["layout"]["head_kg"] == 0.6
    assert mx["layout"]["mass_kg"]["PARK"] > 38.0 and mx["layout"]["mass_kg"]["T4"] > mx["layout"]["mass_kg"]["PARK"]


def test_matrix_six_foot_margin_and_no_fall(gates):
    mx = gates["matrix"]
    every = mx["rows"] + mx.get("heading_rows", [])
    assert min(r["hexa"] for r in every) > st.RAISE_MIN_HEXA_M == 0.13
    assert mx["min_hexa"] > 0.13 and not mx["any_fell"]
    print(f"\n[matrix] {mx['meta']['run_id']}: min 6-foot {mx['min_hexa']} m, "
          f"flat crown-up tripod {mx['flat_crown_up_min_tripod']} m, unstable tripod rows {mx['unstable_tripod_rows']}")


def test_raise_and_walk_gates_as_tabled(gates):
    """Rule 1 holds wherever the raise is allowed (<= 3 deg cross, <= 5 deg tilt); rule 2 start gate holds on the
    crown-up walking slope cap; the flat crown-up margin keeps crown-up walking (no forced stop-and-go)."""
    mx = gates["matrix"]
    for r in mx["rows"]:
        if abs(r["roll_deg"]) <= st.RAISE_MAX_CROSS_DEG and r["tilt_deg"] <= st.WALK_MAX_TILT_DEG + 0.05:
            assert r["hexa"] >= st.RAISE_MIN_HEXA_M, r
    t = st.CrownGateTable.load()
    for ext in (0.6, 1.6):
        assert t.margin(ext, st.CROWN_WALK_MAX_SLOPE_DEG) >= st.WALK_START_TRIPOD_M
    assert mx["flat_crown_up_min_tripod"] >= st.WALK_START_TRIPOD_M
    # the stowed 5 deg crossing keeps the continue gate on every fall line (OS 5.2)
    assert t.margin(0.0, st.WALK_MAX_TILT_DEG) >= st.WALK_CONTINUE_TRIPOD_M


def test_predicted_margin_matches_the_measured_rows(gates):
    """controllers.stability.predicted_tripod_margin on each settled matrix state (930 rows) against the measured
    tripod margin: never optimistic (it is the worse of the measured cross-slope and its mirror). Against the worse
    of the measured row and its measured mirror row (roll -> -roll): optimistic by the settle asymmetry only,
    <= 2 mm at PARK within |pitch| <= 5 deg (the crown-up walking range), <= 6 mm at the worst T4 pose (WORK, the
    0.03 m swap gate) and <= 10 mm anywhere on the grid."""
    p = gates["matrix"]["predicted_vs_measured_m"]
    assert p["n"] >= 900 and p["max"] <= 0.0
    mv = gates["matrix"]["predicted_vs_mirror_m"]
    assert mv["PARK_pitch_le_5"]["max"] <= 0.002 and mv["T4_pitch_le_5"]["max"] <= 0.006
    assert all(-1e-9 <= v["min"] and v["max"] <= 0.010 for v in mv.values())


def test_worst_case_never_assumes_the_rack_side_when_the_arm_loads_the_other(gates):
    """WP0c's worst T4 pose (1.8 kg fruit) loads the left side: on rack-uphill rows the measured margin is worse
    than rack-downhill, so a rack-downhill-only prediction would be optimistic by up to 33 mm; the T4 envelope
    keeps the measured worst (rack uphill) rows."""
    rows = {(r["arm"], r["ext"], r["roll_deg"], r["pitch_deg"]): r for r in gates["matrix"]["rows"]}
    up, dn = rows[("T4", 1.6, -5.0, -10.0)], rows[("T4", 1.6, 5.0, -10.0)]
    assert up["tripod"] < dn["tripod"] - 0.02
    assert dn["predicted"] <= up["tripod"] + 0.010


def test_matrix_rows_reproduce_live():
    from tools import crown_gate_matrix as cgm
    models, rest, q = cgm._models()
    with open(GATES_JSON, encoding="utf-8") as fh:
        stored = {(r["arm"], r["ext"], r["roll_deg"], r["pitch_deg"]): r for r in json.load(fh)["matrix"]["rows"]}
    for arm, ext, roll, pitch in (("PARK", 1.6, 5.0, -10.0), ("PARK", 0.0, 0.0, 0.0), ("T4", 1.6, -3.0, 5.0),
                                  ("PARK", 0.6, 2.0, -5.0)):
        r = cgm.matrix_case(models, rest, ext, arm, q, roll, pitch)
        s = stored[(arm, ext, roll, pitch)]
        assert r["tripod"] == pytest.approx(s["tripod"], abs=5e-4) and r["hexa"] == pytest.approx(s["hexa"], abs=5e-4)


# ====================================================================== live WP24 integration
@contextlib.contextmanager
def world_env(**env):
    saved = {k: os.environ.get(k) for k in WORLD_ENV}
    try:
        for k in WORLD_ENV:
            os.environ.pop(k, None)
        for k, v in env.items():
            os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _noop(spec, field, ctx):
    return None


def basin_world(seed: int = 7, terrain: str = "rough"):
    """orchard_lite terrain without the trees: the crown walk measures stability, so no wood or canopy can strike the
    crown head (WP24's domain). ``rough``: the field's noise (amp 0.01 m) everywhere, i.e. without its flat home
    pad and gentle alleys, plus the trunk basins with their 0.10 m rims (OS 3.6); ``flat``: no noise."""
    import tempfile
    with open(os.path.join(compose.FIELDS_DIR, "orchard_lite.json"), encoding="utf-8") as fh:
        field = json.load(fh)
    if terrain == "rough":
        field["terrain"]["noise"]["flat_zones"] = []
        field["terrain"]["noise"]["gentle_zones"] = []
    tmp = tempfile.mkdtemp(prefix="wp6_field_")
    with open(os.path.join(tmp, "orchard_lite.json"), "w", encoding="utf-8") as fh:
        json.dump(field, fh)
    with world_env(NANDI_WORLD="orchard_lite", NANDI_TERRAIN=terrain, NANDI_SEED=seed, NANDI_ACTORS="0"), \
            compose.temporary_module("trees", compose.ORDER["trees"], _noop), \
            compose.temporary_module("fruit_pool", compose.ORDER["fruit_pool"], _noop):
        cfg = farm_scene.world_config()
        w = compose.build(cfg["world"], cfg["seed"], farm_scene.ROBOT_XML, cfg, fields_dir=tmp)
        return w.spec.compile(), w.field


def set_cross_slope(m, roll_deg: float):
    """Tilted gravity = a plane falling to -y (rack side) by roll_deg (WP0c set_slope convention)."""
    a = math.radians(roll_deg)
    m.opt.gravity[:] = [0.0, -9.81 * math.sin(a), -9.81 * math.cos(a)]


def _driver(m, names=("mast", "stability")):
    return sd.SimDriver(m, hooks=hook_registry.load_all(names=list(names)))


def test_wp6_raise_gate_refuses_on_cross_slope_and_while_walking():
    m, _ = basin_world(terrain="flat")
    set_cross_slope(m, 4.0)
    drv = _driver(m)
    drv.wait(1.0)
    mrt = drv.bus.values["mast_control"]
    s = drv.bus.values["stability"]
    assert s["slope_deg"]["roll"] == pytest.approx(4.0, abs=0.6)      # gravity-relative: left up, rack side down
    dec = mrt.submit(MastRequest("preset", 1.6, "test", t=drv.t))
    assert not dec and dec.reason == "slope" and dec.values.get("detail") == st.GATE_RAISE, dec
    set_cross_slope(m, 2.0)
    drv.wait(1.0)
    assert mrt.submit(MastRequest("preset", 0.6, "test", t=drv.t))    # 2 deg: allowed
    drv.close()


def test_hook_state_reports_predicted_margin_and_gates():
    m, _ = basin_world(terrain="flat")
    drv = _driver(m)
    drv.wait(1.0)
    s = drv.bus.values["stability"]
    assert s["stance"] == sorted(st.LEG_NAMES)
    assert 0.08 < s["predicted_tripod_m"] < 0.13 and s["hexa_m"] > 0.2 and not s["degenerate"]
    assert s["gates"]["walk"] and s["gates"]["slope"] and s["gates"]["raise_hexa_ok"]
    assert s["table_run_id"].startswith("wp6-matrix-")
    assert s["margin_m"] == s["predicted_tripod_m"] and s["stance_margin_m"] > s["margin_m"]   # WP2 gates on margin_m
    assert drv.bus.values["work_field"]["person"] is False
    published = [p for _t, topic, p in drv.published if topic == "/nandi/stability/crown"]
    assert published and json.loads(published[-1])["worst_case"] == "both_sides+park"
    drv.close()


# ====================================================================== crown walk (slow)
WALK_VX = 0.58                      # ~0.15 m/s on the aligned gait (tripod_gait: vx 1 ~ 0.24-0.27 m/s)


def place(drv, x: float, y: float):
    """Teleport the robot to (x, y) on the heightfield: lifted so no foot starts inside the terrain."""
    m, d = drv.m, drv.d
    d.qpos[0:2] = [x, y]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    pen = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        names = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "",
                 mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "")
        if any(n.endswith("_foot") for n in names):
            pen = min(pen, float(c.dist))
    d.qpos[2] += -pen + 0.005
    mujoco.mj_forward(m, d)
    try:                                              # WP2: put the welded tools back on their weld pose
        from atharv_nandi.sim.world.modules import tools as tools_mod
        tools_mod.seat_tools(m, d)
        mujoco.mj_forward(m, d)
    except (ImportError, AttributeError):
        pass


def crown_walk(seed: int, ext: float, cross_deg: float, across_rim: bool = False) -> dict:
    """Crown-up walk past the basin of L1-1. Default: the path the crown-up gate allows (``gate_crown_path``: hull
    5 cm outside the rim footprint, closer than any TREE 360 orbit stand-off), on rough ground. ``across_rim``:
    the left feet cross the 0.10 m rim (the measurement behind the gate)."""
    m, field = basin_world(seed)
    tree = next(t for t in field["trees"] if t["tree_id"] == "L1-1")
    tx, ty = float(tree["geo"]["x"]), float(tree["geo"]["y"])
    r_b = float(tree.get("basin_r_m", 0.7))
    set_cross_slope(m, cross_deg)
    drv = _driver(m)
    if across_rim:
        y0 = ty - (r_b + 0.075)
    else:
        y0 = ty - (st.basin_keepout_radius(r_b) + st.CROWN_PATH_HULL_R_M + 0.05)
    path = [(x, y0) for x in np.linspace(tx - 1.5, tx + 0.5, 21)]
    assert bool(st.gate_crown_path(ext, path, [(tx, ty, r_b)])) is (not across_rim)
    place(drv, tx - 1.5 if across_rim else tx - 0.45, y0)
    drv.wait(1.5)
    mrt = drv.bus.values["mast_control"]
    dec = mrt.submit(MastRequest("preset", ext, "crown_walk_test", t=drv.t))
    assert dec, dec
    drv.wait(ext / 0.2 + 1.0)
    assert drv.bus.values["mast"]["ext_m"] == pytest.approx(ext, abs=1e-6)
    from atharv_nandi.sim.hooks import stability as hs
    rt = hs._RT[0]
    rt.min_predicted = math.inf
    p0, t0 = drv.d.qpos[0:2].copy(), drv.t
    drv.walk(WALK_VX)
    min_pred, max_tilt, fell = math.inf, 0.0, False
    while drv.t - t0 < 6.0:
        drv.wait(0.02)
        s = drv.bus.values["stability"]
        if s["walking"] and s["predicted_tripod_m"] is not None:
            min_pred = min(min_pred, s["predicted_tripod_m"])
        max_tilt = max(max_tilt, abs(s["tilt_deg"]["roll"] - s["slope_deg"]["roll"]),
                       abs(s["tilt_deg"]["pitch"] - s["slope_deg"]["pitch"]))
        fell = fell or drv.d.qpos[2] < 0.15
    dist = float(np.linalg.norm(drv.d.qpos[0:2] - p0))
    ms = drv.bus.values["mast"]
    out = {"seed": seed, "ext": ext, "cross": cross_deg, "across_rim": across_rim, "speed_mps": round(dist / 6.0, 4),
           "min_predicted_m": round(min_pred, 4), "retracts": list(rt.retract_log), "strikes": ms["strike_count"],
           "ext_end": ms["ext_m"], "arbiter": ms["arbiter_source"], "max_rel_tilt_deg": round(max_tilt, 2),
           "tilt_filtered_deg": s["tilt_filtered_deg"], "fell": fell}
    drv.close()
    return out


@pytest.mark.slow
@pytest.mark.parametrize("seed", [7, 11, 23])
def test_crown_walk_no_auto_retract_and_tripod_above_0p03(seed):
    rows = [crown_walk(seed, ext, cross) for ext in (0.6, 1.6) for cross in (0.0, 2.0)]
    for r in rows:
        print(f"\n[crown walk] {r}")
    for r in rows:
        assert not r["fell"], r
        assert r["retracts"] == [] and r["ext_end"] == pytest.approx(r["ext"]), r
        assert r["min_predicted_m"] >= st.WALK_CONTINUE_TRIPOD_M, r
        assert 0.10 <= r["speed_mps"] <= st.CROWN_WALK_MAX_MPS + 0.01, r


@pytest.mark.slow
def test_crown_up_across_a_basin_rim_trips_the_margin_gate_without_a_fall():
    """The measurement behind gate_crown_path: crossing a 0.10 m rim crown-up at ext 1.6 drops the predicted tripod
    margin below 0.03 m and the margin gate retracts (no fall). Rims are crossed stowed."""
    rows = [crown_walk(seed, 1.6, 0.0, across_rim=True) for seed in (7, 23)]
    for r in rows:
        print(f"\n[rim crossing] {r}")
        assert not r["fell"]
        assert r["min_predicted_m"] < st.WALK_CONTINUE_TRIPOD_M
        assert any(e["reason"] in ("margin", "tilt", "tilt_filtered") for e in r["retracts"]) or r["arbiter"] == "safety"
