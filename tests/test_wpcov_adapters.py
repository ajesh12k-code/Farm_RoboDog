"""WPCOV coverage push for the core adapters the unit tier never ran: the ROS 2 control-panel client (stand-in
rclpy), the Tk widget helpers (stand-in tkinter), the PPO balance policy loader (stand-in SB3), the headless
mjvScene cull (no GL needed: ``mjv_updateScene`` only) and the WP9 scope extractors. ROS-free, no display."""
import importlib
import sys
import threading
import types

import mujoco
import numpy as np
import pytest


# ====================================================================== stand-in modules
class _Rec:
    """Records constructor kwargs and attribute writes (a message / widget stand-in)."""

    def __init__(self, *a, **k):
        self.args, self.kw = a, dict(k)
        for key, v in k.items():
            setattr(self, key, v)


class _Vec:
    x = y = z = 0.0


class _Twist:
    def __init__(self):
        self.linear, self.angular = _Vec(), _Vec()


class _Pub:
    def __init__(self, topic):
        self.topic, self.sent = topic, []

    def publish(self, msg):
        self.sent.append(msg)


class _Future:
    def __init__(self, result=None, done=True):
        self._r, self._done, self.cancelled = result, done, False

    def done(self):
        return self._done

    def result(self):
        return self._r

    def cancel(self):
        self.cancelled = True


class _Client:
    def __init__(self, name, up=True, fut=None):
        self.name, self.up, self.fut, self.removed = name, up, fut, []

    def wait_for_service(self, timeout_sec=None):
        return self.up

    def call_async(self, req):
        return self.fut

    def remove_pending_request(self, fut):
        self.removed.append(fut)


class _Node:
    def __init__(self, name):
        self.name, self.pubs, self.subs, self.clients = name, {}, {}, {}

    def create_publisher(self, typ, topic, qos):
        self.pubs[topic] = _Pub(topic)
        return self.pubs[topic]

    def create_client(self, typ, name):
        self.clients[name] = _Client(name)
        return self.clients[name]

    def create_subscription(self, typ, topic, cb, qos):
        self.subs[topic] = cb


@pytest.fixture
def ros_client(monkeypatch):
    spin = {"ok": iter([True, False]), "n": 0}

    def ok():
        return next(spin["ok"], False)

    def spin_once(node, timeout_sec=None):
        spin["n"] += 1
    rclpy = types.ModuleType("rclpy")
    rclpy.ok, rclpy.spin_once = ok, spin_once
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    geo, geo_msg = types.ModuleType("geometry_msgs"), types.ModuleType("geometry_msgs.msg")
    geo_msg.Twist = _Twist
    std, std_msg = types.ModuleType("std_msgs"), types.ModuleType("std_msgs.msg")
    std_msg.Float64MultiArray = std_msg.String = _Rec
    srv, srv_srv = types.ModuleType("std_srvs"), types.ModuleType("std_srvs.srv")
    srv_srv.Trigger = types.SimpleNamespace(Request=lambda: "req")
    for name, mod in (("rclpy", rclpy), ("rclpy.node", node_mod), ("geometry_msgs", geo),
                      ("geometry_msgs.msg", geo_msg), ("std_msgs", std), ("std_msgs.msg", std_msg),
                      ("std_srvs", srv), ("std_srvs.srv", srv_srv)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "atharv_nandi.core.ros_client", raising=False)
    mod = importlib.import_module("atharv_nandi.core.ros_client")
    yield mod, spin
    sys.modules.pop("atharv_nandi.core.ros_client", None)


def _wait_for(pred, n=200):
    ev = threading.Event()
    for _ in range(n):
        if pred():
            return True
        ev.wait(0.01)
    return pred()


def test_ros_client_topics_callbacks_and_publishers(ros_client):
    mod, spin = ros_client
    from atharv_nandi.core.key_drive import DriveCommand
    c = mod.NandiROSClient()
    assert _wait_for(lambda: spin["n"] >= 1)                         # the background spin ran, then rclpy.ok False
    assert set(c.subs) == {"/nandi/sim_qpos", "/nandi/locomotion_status", "/nandi/harvest_status",
                           "/nandi/game_status", "/nandi/workspace_wireframe"}
    assert c.loco_status["ok"] is False and c.telemetry == (0.0, 0.0, 0.0)
    qpos = [0.0] * 40
    qpos[0], qpos[2] = 1.5, 0.4
    c.subs["/nandi/sim_qpos"](types.SimpleNamespace(data=qpos))
    assert c.telemetry[0] == pytest.approx(1.5) and c.telemetry[1] == pytest.approx(0.4)
    c.subs["/nandi/locomotion_status"](types.SimpleNamespace(data='{"mode": "tripod"}'))
    c.subs["/nandi/harvest_status"](types.SimpleNamespace(data="not json"))
    c.subs["/nandi/game_status"](types.SimpleNamespace(data="{}"))
    assert c.loco_status["ok"] and not c.harvest_status["ok"]
    c.subs["/nandi/workspace_wireframe"](types.SimpleNamespace(data='{"a": 1}'))
    assert c.wireframe_shown and c.wireframe_json == '{"a": 1}'
    c.subs["/nandi/workspace_wireframe"](types.SimpleNamespace(data=""))
    assert not c.wireframe_shown and c.wireframe_json == '{"a": 1}'     # last non-empty kept
    c.send_drive(DriveCommand(0.2, -0.1, 0.3, True))
    m = c.pubs["/nandi/cmd_vel"].sent[-1]
    assert (m.linear.x, m.linear.y, m.angular.z, m.linear.z) == (0.2, -0.1, 0.3, 1.0)
    c.send_drive(DriveCommand(0.0, 0.0, 0.0, False))
    assert c.pubs["/nandi/cmd_vel"].sent[-1].linear.z == 0.0
    c.send_gait_params([9.0, 9.0, 9.0, 9.0])
    assert all(isinstance(v, float) for v in c.pubs["/nandi/gait_params"].sent[-1].data)
    c.send_arm([1, 2, 3, 4, 5, 6, 7])
    assert c.pubs["/nandi/manual_arm"].sent[-1].data == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    c.send_wireframe("")
    assert c.pubs["/nandi/workspace_wireframe"].sent[-1].data == ""


def test_ros_client_service_calls_offline_ok_and_timeout(ros_client, monkeypatch):
    mod, _ = ros_client
    c = mod.NandiROSClient()
    got = []
    c.service_clients["stop"].up = False
    c.call_service("stop", lambda ok, msg: got.append(("stop", ok, msg)))
    c.service_clients["mode"].fut = _Future(types.SimpleNamespace(success=True, message="tripod"))
    c.call_service("mode", lambda ok, msg: got.append(("mode", ok, msg)))
    c.service_clients["harvest"].fut = _Future(None)                  # done, but no response object
    c.call_service("harvest", lambda ok, msg: got.append(("harvest", ok, msg)))
    c.service_clients["reset"].fut = _Future(types.SimpleNamespace(success=True, message="ok"))
    c.call_service("reset")                                          # no callback: must not raise
    assert _wait_for(lambda: len(got) == 3)
    assert sorted(got) == [("harvest", False, "Timeout"), ("mode", True, "tripod"), ("stop", False, "Service offline")]

    # timeout: the pending request is cancelled and dropped (WPCOV review)
    monkeypatch.setattr(mod, "SERVICE_POLL_S", 0.001)
    monkeypatch.setattr(mod, "SERVICE_POLL_N", 3)
    fut = _Future(None, done=False)
    c.service_clients["release"].fut = fut
    c.call_service("release", lambda ok, msg: got.append(("release", ok, msg)))
    assert _wait_for(lambda: len(got) == 4)
    assert got[-1] == ("release", False, "Timeout") and fut.cancelled
    assert c.service_clients["release"].removed == [fut]


# ---------------------------------------------------------------------- Tk widgets
class _Widget:
    def __init__(self, parent=None, **kw):
        self.parent, self.kw, self.packed, self.bound, self.items = parent, dict(kw), None, {}, {}
        self.value = None
        self.scrolled = []

    def __getitem__(self, k):
        return self.kw[k]

    def config(self, **kw):
        self.kw.update(kw)
    configure = config

    def pack(self, **kw):
        self.packed = kw

    def set(self, v):
        self.value = v

    def bind(self, ev, fn):
        self.bound[ev] = fn

    def bind_all(self, ev, fn):
        self.bound[ev] = fn

    def create_window(self, xy, window=None, anchor=None):
        self.items[1] = window
        return 1

    def itemconfigure(self, i, **kw):
        self.items[(i, "cfg")] = kw

    def bbox(self, _):
        return (0, 0, 10, 10)

    def yview(self, *a):
        return None

    def yview_scroll(self, n, what):
        self.scrolled.append(n)

    def winfo_reqheight(self):
        return self.kw.get("_req", 500)

    def winfo_height(self):
        return self.kw.get("_h", 100)


@pytest.fixture
def gw(monkeypatch):
    tk = types.ModuleType("tkinter")
    for name in ("Label", "Button", "Frame", "Scale", "Canvas", "Scrollbar"):
        setattr(tk, name, type(name, (_Widget,), {}))
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.delitem(sys.modules, "atharv_nandi.core.gui_widgets", raising=False)
    mod = importlib.import_module("atharv_nandi.core.gui_widgets")
    yield mod
    sys.modules.pop("atharv_nandi.core.gui_widgets", None)


def test_gui_widgets_labels_buttons_dimming(gw):
    parent = _Widget(bg=gw.BG)
    lab = gw.make_label(parent, "hi", size=12)
    assert lab.kw["bg"] == gw.BG and lab.kw["font"] == ("Sans", 12, "bold")
    assert gw.make_label(parent, "x", bold=False).kw["font"] == ("Sans", 10)
    btn = gw.make_button(parent, "STOP", "#c00", cmd=print)
    assert btn.kw["takefocus"] == 0 and btn.kw["command"] is print       # Space never clicks a focused button
    gw.set_button_dimmed(btn, True, "#c00")
    assert btn.kw["bg"] == gw.DIM_BG and btn.kw["fg"] == gw.DIM_FG
    gw.set_button_dimmed(btn, False, "#c00")
    assert btn.kw["bg"] == "#c00" and btn.kw["fg"] == "white"


def test_gui_widgets_slider_and_scrollable(gw):
    got = []
    sc = gw.make_slider_row(_Widget(bg=gw.BG), "hip", -1.0, 1.0, 0.01, 0.25, got.append)
    assert sc.value == 0.25 and sc.kw["takefocus"] == 0
    sc.kw["command"]("0.5")
    assert got == [0.5]
    root = _Widget()
    inner = gw.make_scrollable(root, 300)
    canvas = inner.parent
    inner.bound["<Configure>"](None)
    canvas.bound["<Configure>"](types.SimpleNamespace(width=280))
    assert canvas.kw["scrollregion"] == (0, 0, 10, 10) and canvas.items[(1, "cfg")] == {"width": 280}
    root.bound["<Button-4>"](None)
    root.bound["<Button-5>"](None)
    root.bound["<MouseWheel>"](types.SimpleNamespace(delta=120))
    root.bound["<MouseWheel>"](types.SimpleNamespace(delta=-120))
    assert canvas.scrolled == [-2, 2, -1, 1]
    inner.kw["_req"] = 50                                            # content shorter than the view: no scroll
    root.bound["<Button-5>"](None)
    assert canvas.scrolled == [-2, 2, -1, 1]


# ---------------------------------------------------------------------- PPO balance policy loader
@pytest.fixture(scope="module")
def farm():
    from atharv_nandi.core.farm_scene import load_farm_model
    return load_farm_model()


def test_find_policy_zip_prefers_final_then_most_steps(tmp_path):
    from atharv_nandi.core import rl_balance_policy as rp
    assert rp.find_policy_zip(str(tmp_path)) is None
    for n in ("farmdog6_balance_100_steps.zip", "nandi_balance_20_steps.zip", "nandi_balance_bad_steps.zip"):
        (tmp_path / n).write_bytes(b"")
    assert rp.find_policy_zip(str(tmp_path)).endswith("farmdog6_balance_100_steps.zip")
    (tmp_path / "farmdog6_balance_final.zip").write_bytes(b"")
    assert rp.find_policy_zip(str(tmp_path)).endswith("farmdog6_balance_final.zip")
    (tmp_path / "nandi_balance_final.zip").write_bytes(b"")
    assert rp.find_policy_zip(str(tmp_path)).endswith("nandi_balance_final.zip")


class _FakePolicy:
    def __init__(self, shape, action=None):
        self.observation_space = types.SimpleNamespace(shape=shape)
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action, None


def test_policy_loader_degrades_with_a_reason(farm, monkeypatch, tmp_path):
    from atharv_nandi.core import rl_balance_policy as rp
    monkeypatch.setattr(rp, "HAS_SB3", True)
    monkeypatch.setattr(rp, "find_policy_zip", lambda models_dir=rp.MODELS_DIR: None)
    rl = rp.RLBalanceController(0.02, model=farm)
    assert rl.policy is None and "no policy zip" in rl.load_error

    monkeypatch.setattr(rp, "find_policy_zip", lambda models_dir=rp.MODELS_DIR: str(tmp_path / "p.zip"))

    def boom(path, device=None):
        raise ValueError("corrupt")
    monkeypatch.setattr(rp, "PPO", types.SimpleNamespace(load=boom))
    rl = rp.RLBalanceController(0.02, model=farm)
    assert rl.policy is None and "failed to load" in rl.load_error and "corrupt" in rl.load_error

    monkeypatch.setattr(rp, "PPO", types.SimpleNamespace(load=lambda p, device=None: _FakePolicy((83,))))
    rl = rp.RLBalanceController(0.02, model=farm)
    assert rl.policy is None and "observation shape" in rl.load_error


def test_policy_predict_clips_and_filters(farm, monkeypatch, tmp_path):
    from atharv_nandi.core import rl_balance_policy as rp
    act = np.full(rp.ACTION_DIM, 5.0, dtype=np.float32)
    monkeypatch.setattr(rp, "HAS_SB3", True)
    monkeypatch.setattr(rp, "find_policy_zip", lambda models_dir=rp.MODELS_DIR: str(tmp_path / "p.zip"))
    monkeypatch.setattr(rp, "PPO", types.SimpleNamespace(load=lambda p, device=None: _FakePolicy((rp.OBS_DIM,), act)))
    rl = rp.RLBalanceController(0.02, model=farm)
    assert rl.policy is not None and rl.policy_path.endswith("p.zip") and rl.load_error is None
    q = rl.data.qpos.copy()
    arm, legs = rl.predict(q, q)
    assert np.all(rl.last_action == 1.0)                               # clipped to [-1, 1]
    assert len(arm) == 5 and len(legs) == 18
    raw = rl.home_arm + rl.arm_scale
    assert np.allclose(arm, 0.65 * rl.home_arm + 0.35 * raw)
    assert np.allclose(legs, rl.home_legs + rl.leg_scale)
    rest = rl.reset_arm_to_rest()
    assert np.all(rl.last_action == 0.0) and len(rest) == 5


# ---------------------------------------------------------------------- headless scene cull
def _scene(m, maxgeom=4000):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    scn = mujoco.MjvScene(m, maxgeom=maxgeom)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.0, 0.0, 0.5)
    cam.distance = 3.0
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, cam, int(mujoco.mjtCatBit.mjCAT_ALL), scn)
    return scn


def _rows(scn):
    return sorted((int(g.objtype), int(g.objid)) for g in (scn.geoms[i] for i in range(scn.ngeom)))


def test_scene_cull_fast_and_slow_agree_without_gl(farm):
    from atharv_nandi.core import scene_cull as sc
    fp0 = sc.model_fingerprint(farm)
    a, b = _scene(farm), _scene(farm)
    fast, slow = sc.SceneCuller(farm, a, 2.0), sc.SceneCuller(farm, b, 2.0, force_slow=True)
    assert not slow.fast
    n0 = a.ngeom
    ra, rb = fast.cull([0.0, 0.0, 1.0]), slow.cull([0.0, 0.0, 1.0])
    assert ra == rb and ra[0] == n0 and ra[1] < n0
    assert _rows(a) == _rows(b)
    assert [b.geoms[i].segid for i in range(b.ngeom)] == list(range(b.ngeom))
    if fast.fast:
        assert [a.geoms[i].segid for i in range(a.ngeom)] == list(range(a.ngeom))
    assert sc.model_fingerprint(farm) == fp0                           # render-only: no model field touched
    assert fast.cull([0.0, 0.0, 1.0]) == (ra[1], ra[1])                # nothing more to drop


def test_scene_cull_small_and_empty_scenes(farm):
    from atharv_nandi.core import scene_cull as sc
    tiny = mujoco.MjvScene(farm, maxgeom=1)
    cu = sc.SceneCuller(farm, tiny)
    assert not cu.fast                                                 # no stride to measure
    tiny.ngeom = 0
    assert cu.keep_mask([0, 0, 0]).size == 0 and cu.cull([0, 0, 0]) == (0, 0)


# ---------------------------------------------------------------------- WP9 scope extractors
def _table():
    from atharv_nandi.core import scope_kpi as sk
    return sk, sk.KpiTable("run-1", "sim-dev", date="2026-11-18")


def _row(t, metric):
    return next(r for r in t.rows if r["metric"] == metric)


def test_scope_extractors_gate_and_report():
    sk, t = _table()
    sk.scan_kpis(t, {"trees": [{"seen": True, "bands": ["low", "mid"]}, {"seen": False, "bands": ["low"]}],
                     "laterals_stepped": 0, "triage": {"tp": 0, "fp": 0, "fn": 2}})
    assert _row(t, "scan.trees_seen_frac")["pass"] is False and _row(t, "scan.triage_precision")["value"] is None
    sk.scan_kpis(t, {})
    sk.tree360_kpis(t, {"mode": "two_sided", "crown_walk": "stop_and_go",
                        "coverage": {"bands": {"low": {"pct": 60.0, "sides": {"A": 130, "B": 125}, "frames": 3},
                                               "high": {"pct": 0, "sides": {"A": 0}, "frames": 0}}},
                        "kpi": {"radius_rms_truth_m": 0.05, "orbit_s": 200.0, "bytes": 3 * 1024 * 1024,
                                "person_unmasked_px": 0}})
    assert _row(t, "tree360.coverage_low_pct")["pass"] is None                 # two-sided: reported only
    assert _row(t, "tree360.two_sided_min_deg")["value"] == 125
    assert _row(t, "tree360.orbit_s")["pass"] is None and _row(t, "tree360.mb_per_tree")["value"] == 3.0
    sk.tree360_kpis(t, {"mode": "full", "crown_walk": "snake", "bands_planned": ["low"],
                        "coverage": {"bands": {"low": {"pct": 95.0}, "mid": {"pct": 50.0}}},
                        "kpi": {"orbit_s": 150.0}})
    assert [r["pass"] for r in t.rows if r["metric"] == "tree360.coverage_low_pct"][-1] is True
    assert [r["pass"] for r in t.rows if r["metric"] == "tree360.orbit_s"][-1] is True
    sk.leaf_kpis(t, {"status": "skipped"})
    sk.leaf_kpis(t, {"status": "ok", "n": 6, "imaged_frac": 1.0, "underside_frac": 0.5, "footprint_mm_px_max": 0.4,
                     "median_s_per_leaf": 8.0, "min_clearance_m": 0.03})
    sk.trunk_kpis(t, {"azimuths_imaged": 2, "mode": "two"})
    sk.scout_kpis(t, {"visible_n": 4, "reachable_n": 3}, {"count_error": 0.05, "stage_acc": 0.95,
                                                          "window_equal": True, "truth_visible_n": 4})
    sk.scout_kpis(t, {}, None)
    assert [r["value"] for r in t.rows if r["metric"] == "fruit_scout.reach_ratio"] == [0.75, None]
    sk.harvest_kpis(t, [{"crop": "mango", "results": [{"outcome": "picked", "pedicel_error_mm": -3.0},
                                                      {"outcome": "failed"}, {"outcome": "marked"}],
                         "skipped_targets": 1, "place_mps_max": 0.3},
                        {"results": [{"crop": "papaya", "outcome": "delivered"}]}])
    assert _row(t, "harvest.mango_success")["value"] == 0.5 and _row(t, "harvest.mango_success")["aggregate"]
    assert _row(t, "harvest.pedicel_err_mm_max")["value"] == 3.0 and _row(t, "harvest.papaya_place_mps_max")
    sk.harvest_kpis(t, [])
    sk.prune_kpis(t, [{"results": [{"outcome": "cut"}, {"outcome": "marked"}, {"outcome": "skipped"}],
                       "keep_cuts": 0, "green_cuts": None}])
    assert _row(t, "prune.cut_frac")["value"] == 0.5 and _row(t, "prune.green_leaf_cuts")["value"] is None
    sk.prune_kpis(t, [])
    sk.dock_kpis(t, {"dock": {"final": {"dist_m": 0.02, "yaw_deg": -3.0}}, "expected_items": 4,
                     "crate": {"n": 4, "impact_mps_max": 1.0, "per_tree": {"OT-1": {"crate_n": 3}, "G-2": {"n": 1}}}})
    assert _row(t, "dock.error_cm")["value"] == 2.0 and _row(t, "dock.attribution_frac")["value"] == 1.0
    sk.dock_kpis(t, {"dock": {"final": {"dist_m": None, "yaw_deg": None}}})
    sk.safety_kpis(t, {"person_pixels_measured": False, "gust_stance_s": 0.3})
    sk.safety_kpis(t, {"unmasked_person_px": 2})
    assert [r["pass"] for r in t.rows if r["metric"] == "safety.unmasked_person_px"] == [None, False]
    sk.records_kpis(t, [{"a": 1}], 0, ["m1"], ["R1", "R2"], {"R1": {}}, marks_truth_n=2, marks_n=1)
    sk.records_kpis(t, [], 1, [], [], {})
    assert _row(t, "records.jobsheet_rows_frac")["value"] == 0.5 and _row(t, "marks.completeness")["value"] == 0.5
    assert {r["metric"] for r in t.failed()} >= {"scan.trees_seen_frac", "records.jobsheet_missing_marks"}
    assert "harvest.mango_success" not in {r["metric"] for r in t.failed()}
    assert "harvest.mango_success" in {r["metric"] for r in t.failed(include_aggregate=True)}


def test_scope_cmp_and_physics_mode_is_never_gated():
    sk, t = _table()
    assert sk._cmp("ge", "x", 1) is False and sk._cmp("eq", "a", "a") is True and sk._cmp("bogus", 1, 1) is None
    assert sk._cmp("true", True, True) is True and sk._cmp(None, 1, 1) is None and sk._cmp("le", None, 1) is None
    p = sk.KpiTable("r", "d", mode="physics")
    assert p.add("scan.trees_seen_frac", 0.1)["pass"] is None and p.date
    assert sk.median([None, 3, 1, 2]) == 2 and sk.median([]) is None


def test_run_card_and_markdown():
    sk, t = _table()
    t.add("scan.trees_seen_frac", 0.5, 2)
    t.add("harvest.mango_success", 0.5, 2)
    t.add("custom.metric", 1, units="u")
    card = sk.run_card(ident={"run_id": "run-1", "device_id": "sim-dev"}, date="2026-11-18", world="orchard_lite",
                       season="demo_all_fruiting", seed=7, program="tour", detector="simtruth", mode="assisted",
                       assisted=["b", "a", "a"], scripted_delays=[{"what": "person", "s": 2}], table=t,
                       counts={"trees": 2}, tour={"pass": False, "fail_reasons": ["x"], "sim_s": 10}, notes=["n1"])
    assert card["assisted"] == ["a", "b"] and card["failed_gates"] == ["scan.trees_seen_frac"]
    assert card["failed_aggregate"] == ["harvest.mango_success"] and card["season_banner"]
    md = sk.format_card(card)
    assert "Tour pass: FAIL" in md and "Fail reasons: x" in md and "Scripted delays: person 2 s" in md
    assert "(aggregate)" in md and md.rstrip().endswith("- n1")
    card2 = sk.run_card(ident={"run_id": "r", "device_id": "d"}, date="d", world="w", season="up", seed=1,
                        program="p", detector="simtruth", mode="assisted", assisted=[], scripted_delays=[],
                        table=sk.KpiTable("r", "d"), counts={}, tour={"pass": True})
    md2 = sk.format_card(card2)
    assert "Tour pass: PASS" in md2 and "Assisted: none" in md2 and "season_banner" not in md2
