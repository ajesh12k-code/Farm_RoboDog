"""WP0 compose + hook registries (ROS-free, headless, each test well under 10 s).

Covers MVP-SPEC 3.1 / 5.1 and ORCHARD-MVP-SPEC 3.1: modules compose in declared ORDER without editing
farm_scene.py; with 10 pool free bodies + 3 tools + 10 mocap appended the legacy crop slices, harvest,
recorder and mission crop counts still resolve; the home keyframe is generated; world_id; hooks run at
their declared rate and the substep point runs every physics step.
"""
import contextlib
import hashlib
import json
import os
import sys
import types

import mujoco
import numpy as np
import pytest

import atharv_nandi.sim.hooks as hook_registry
from atharv_nandi.nodes import run_recorder
from atharv_nandi.core import farm_scene
from atharv_nandi.core import harvest_logic as hl
from atharv_nandi.core import model_layout as mlay
from atharv_nandi.sim.world import compose

N_TOOLS, N_POOL, N_MOCAP = 3, 10, 10
POOL_CONNECT = 5          # slots 0-4 connect (mango), 5-9 weld (guava, papaya): ORCHARD-MVP-SPEC 3.4


# ---------------------------------------------------------------------- the extended test world
def _attach_tools(spec, field, ctx):
    """3 tools x (free joint + 2 finger slides), 2 appended actuators each, a rack weld each (active)."""
    for i in range(N_TOOLS):
        b = spec.worldbody.add_body(name=f"test_tool_{i}", pos=[-1.0 - 0.3 * i, 1.5, 0.05])
        b.add_freejoint(name=f"test_tool_{i}_free")
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.03, 0.02], mass=0.3)
        for side, y in (("l", 0.02), ("r", -0.02)):
            f = b.add_body(name=f"test_tool_{i}_jaw_{side}", pos=[0.05, y, 0])
            f.add_joint(name=f"test_tool_{i}_jaw_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0],
                        range=[-0.01, 0.01])
            f.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.005, 0.01], mass=0.02)
            spec.add_actuator(name=f"test_tool_{i}_jaw_{side}", target=f"test_tool_{i}_jaw_{side}",
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        spec.add_equality(name=f"test_rack_{i}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=f"test_tool_{i}", active=True)
    ctx.set_home("test_tool_0_jaw_l", [0.004])


def _attach_pool(spec, field, ctx):
    """10 fruit-pool free bodies parked under the floor, each with an inactive equality to the world."""
    for i in range(N_POOL):
        b = spec.worldbody.add_body(name=f"test_pool_{i}", pos=[5.0 + 0.2 * i, -3.0, -1.0])
        b.add_freejoint(name=f"test_pool_{i}_free")
        b.add_geom(name=f"test_pool_{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.04, 0, 0], mass=0.2,
                   contype=1, conaffinity=0)
        kind = mujoco.mjtEq.mjEQ_CONNECT if i < POOL_CONNECT else mujoco.mjtEq.mjEQ_WELD
        spec.add_equality(name=f"test_pool_eq_{i}", type=kind, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=f"test_pool_{i}", active=False)
    # a joint-less "mast section" whose body_pos a mutation provider owns (test_sim_state)
    spec.body("trunk").add_body(name="test_mast_s1", pos=[-0.30, 0.0, 0.05]).add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.1, 0], mass=0.05, contype=0, conaffinity=0)


def _attach_actors(spec, field, ctx):
    for i in range(N_MOCAP):
        b = spec.worldbody.add_body(name=f"test_actor_{i}", pos=[2.0 + i, 4.0, 0.0], mocap=True)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.2, 0.6, 0], contype=0, conaffinity=0)
    ctx.set_home_mocap("test_actor_0", [2.0, 4.0, 0.9])


@contextlib.contextmanager
def extended_world():
    """Tools (30), fruit pool (55) and actors (70) registered as compose modules, like WP2 / WP23 / WP4 will."""
    with compose.temporary_module("test_tools", compose.ORDER["tools"], _attach_tools), \
            compose.temporary_module("test_pool", compose.ORDER["fruit_pool"], _attach_pool), \
            compose.temporary_module("test_actors", compose.ORDER["actors"], _attach_actors):
        yield


def build_extended():
    with extended_world():
        model = farm_scene.load_farm_model()
        lay = mlay.current()
    return model, lay


# ---------------------------------------------------------------------- ordering
def test_modules_compose_in_declared_order_without_editing_farm_scene():
    calls = []

    def make(tag):
        def attach(spec, field, ctx):
            calls.append(tag)
            spec.worldbody.add_body(name=f"order_{tag}", pos=[0, 9, 0]).add_geom(size=[0.01, 0, 0])
        return attach

    # registered late-band first, so registration order cannot be what orders them
    with compose.temporary_module("zz_late", 60, make("late")), compose.temporary_module("aa_early", 45, make("early")):
        names = [m.name for m in compose.modules()]
        model = farm_scene.load_farm_model()
    assert calls == ["early", "late"]
    assert names.index("robot") < names.index("legacy_crops") < names.index("legacy_terrain") \
        < names.index("aa_early") < names.index("zz_late")
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    assert bid("crop_item_6") < bid("order_early") < bid("order_late")
    assert "aa_early" not in [m.name for m in compose.modules()]  # temporary registration is undone


def test_order_bands_match_the_orchard_spec():
    assert compose.ORDER == {"robot": 0, "legacy_crops": 10, "tools": 30, "terrain": 40, "fixtures": 40,
                             "trees": 50, "fruit_pool": 55, "actors": 70, "dock": 80}
    with pytest.raises(ValueError):
        compose.register_module("bad", 101, lambda *a: None)


def test_temporary_module_restores_a_shadowed_module():
    before = {m.name: m for m in compose.modules()}["legacy_crops"]
    with compose.temporary_module("legacy_crops", 10, lambda *a: None, version="x"):
        assert {m.name: m for m in compose.modules()}["legacy_crops"].version == "x"
    assert {m.name: m for m in compose.modules()}["legacy_crops"] is before


def test_temporary_module_before_discovery_keeps_the_shadowed_module():
    """Shadowing a module before any build (fresh process) must not drop it once the block exits."""
    import subprocess
    code = ("from atharv_nandi.sim.world import compose\n"
            "with compose.temporary_module(\"fruit_pool\", compose.ORDER[\"fruit_pool\"], lambda *a: None):\n"
            "    pass\n"
            "mods = {m.name: m for m in compose.modules()}\n"
            "assert \"fruit_pool\" in mods and mods[\"fruit_pool\"].attach_fn.__module__ == \"atharv_nandi.sim.world.modules.fruit_pool\"\n")
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    r = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_discover_imports_every_module_file():
    names = compose.discover(force=True)
    assert all(not n.startswith("_") for n in names)
    for n in names:
        assert f"atharv_nandi.sim.world.modules.{n}" in sys.modules


# ---------------------------------------------------------------------- legacy world unchanged
LEGACY_KEY_SHA16 = "097381b78a0f6f47"   # key_qpos of the default world (6 crops, rough, seed 7) before WP0


def test_legacy_home_qpos_hash_unchanged():
    m = farm_scene.load_farm_model()
    assert m.nq == 32 + 7 * 6 and m.nu == 25 and m.nmocap == 0
    assert hashlib.sha256(np.asarray(m.key_qpos[0]).tobytes()).hexdigest()[:16] == LEGACY_KEY_SHA16
    lay = mlay.current()
    assert lay.world == "legacy" and lay.is_legacy_shape


def test_legacy_sim_qpos_is_byte_identical_to_qpos():
    m = farm_scene.load_farm_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(20):
        mujoco.mj_step(m, d)
    lay = mlay.current()
    # /nandi/sim_qpos is data.qpos.tolist(); the layout's legacy accessor is exactly that vector
    assert np.asarray(lay.legacy_qpos(d)).tobytes() == np.asarray(d.qpos.tolist()).tobytes()


# ---------------------------------------------------------------------- extended world
def test_extended_world_keeps_robot_prefix_and_legacy_crops():
    model, lay = build_extended()
    assert mlay.frozen_violations(lay) == []
    assert lay.nq == 32 + 7 * 6 + N_TOOLS * (7 + 2) + N_POOL * 7
    assert lay.nv == 31 + 6 * 6 + N_TOOLS * (6 + 2) + N_POOL * 6
    assert lay.nu == 25 + 2 * N_TOOLS and lay.nmocap == N_MOCAP and lay.neq == N_TOOLS + N_POOL
    assert lay.actuators[:25] == mlay.ROBOT_ACTUATORS
    assert all(lay.act_id(f"test_tool_{i}_jaw_l") >= 25 for i in range(N_TOOLS))
    assert lay.crop_names == [f"crop_item_{k}" for k in range(1, 7)]
    assert [a for _, a in lay.crops] == [32 + 7 * k for k in range(6)]
    assert lay.qpos_adr("test_tool_0_free") == 32 + 7 * 6  # module joints come after the legacy crops
    assert not lay.is_legacy_shape
    with pytest.raises(mlay.LayoutError):
        lay.legacy_qpos(np.zeros(lay.nq))


def test_extended_world_crop_slices_harvest_recorder_and_mission_counts_resolve():
    model, lay = build_extended()
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    q = d.qpos.copy()
    # crop slices (layout and the legacy helper agree, and match the body positions)
    for k, name in enumerate(lay.crop_names, start=1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert lay.crop_qpos(k) == farm_scene.crop_qpos_slice(k) == lay.crop_slice(name)
        np.testing.assert_allclose(d.xpos[bid], q[lay.crop_qpos(k)][:3])
        np.testing.assert_allclose(hl.crop_pos(q, name), d.xpos[bid])
    # recorder and mission executor counts: exactly the 6 legacy crops, never a tool or pool body
    rec = run_recorder.crop_positions(q, lay)
    assert sorted(rec) == sorted(lay.crop_names)
    assert mlay.legacy_crop_positions(q, lay) == rec == lay.crop_positions(q)
    assert run_recorder.crop_positions(q) == rec  # the process's current layout is used by default
    assert mlay.n_legacy_crops_for(len(q), lay) == 6
    # the pre-WP0 rule would have read (nq - 32) // 7 = 13 "crops" out of tool and pool joints
    assert (len(q) - 32) // 7 != 6
    # harvest: moving a pool fruit is not a world reset, moving a crop is
    moved_pool = q.copy()
    s = lay.qpos_slice("test_pool_9_free")
    moved_pool[s.start:s.start + 3] += 1.0
    assert not hl.world_reset_detected(q, moved_pool, lay)
    assert not hl.world_reset_detected(q, moved_pool)          # current layout by default
    moved_crop = q.copy()
    moved_crop[lay.crop_qpos(3).start] += 0.5
    assert hl.world_reset_detected(q, moved_crop, lay)
    target, _ = hl.select_target(q, lay.crop_names)
    assert target in lay.crop_names


def test_generated_keyframe_covers_every_joint_and_mocap():
    model, lay = build_extended()
    key = np.asarray(model.key_qpos[0])
    assert key.shape == (model.nq,) and np.all(np.isfinite(key))
    np.testing.assert_array_equal(key[:32], farm_scene.mujoco.MjModel.from_xml_path(farm_scene.ROBOT_XML).key_qpos[0][:32])
    adr = lay.qpos_adr("test_tool_1_free")
    np.testing.assert_allclose(key[adr:adr + 3], [-1.3, 1.5, 0.05])
    assert key[lay.qpos_adr("test_tool_0_jaw_l")] == pytest.approx(0.004)     # ctx.set_home override
    mid = lay.mocap_id("test_actor_0")
    np.testing.assert_allclose(model.key_mpos[0].reshape(-1, 3)[mid], [2.0, 4.0, 0.9])  # set_home_mocap
    np.testing.assert_allclose(model.key_mpos[0].reshape(-1, 3)[lay.mocap_id("test_actor_3")], [5.0, 4.0, 0.0])
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_step(model, d)
    assert np.all(np.isfinite(d.qpos))


def test_a_jointed_body_inside_the_robot_prefix_is_refused():
    def bad(spec, field, ctx):
        b = spec.body("trunk").add_body(name="bad_link", pos=[0, 0, 0.2])
        b.add_joint(name="bad_hinge", type=mujoco.mjtJoint.mjJNT_HINGE)
        b.add_geom(size=[0.01, 0, 0], mass=0.01)

    with compose.temporary_module("bad_robot_joint", 5, bad):
        with pytest.raises(mlay.LayoutError, match="frozen robot layout"):
            farm_scene.load_farm_model()


# ---------------------------------------------------------------------- world id
def test_world_id_is_deterministic_and_tracks_world_seed_versions_and_registry(monkeypatch, tmp_path):
    w, wid = farm_scene.world_identity()
    assert w == "legacy" and len(wid) == 12 and wid == farm_scene.build_world().world_id
    assert wid == farm_scene.world_identity()[1]
    monkeypatch.setenv("NANDI_SEED", "11")
    assert farm_scene.world_identity()[1] != wid
    monkeypatch.delenv("NANDI_SEED")
    monkeypatch.setenv("NANDI_CROPS", "3")
    assert farm_scene.world_identity()[1] != wid
    monkeypatch.delenv("NANDI_CROPS")
    monkeypatch.setenv("NANDI_WORLD", "orchard_lite")
    assert farm_scene.world_identity()[0] == "orchard_lite" and farm_scene.world_identity()[1] != wid
    monkeypatch.delenv("NANDI_WORLD")
    with compose.temporary_module("legacy_crops", 10, farm_scene._attach_legacy_crops, version="2"):
        assert farm_scene.world_identity()[1] != wid
    assert farm_scene.world_identity()[1] == wid
    v = ["a@1"]
    assert compose.world_id("w", 7, v, "") != compose.world_id("w", 7, v, "abc")
    (tmp_path / "orchard_x.json").write_text(json.dumps({"trees": [{"tree_id": "OT-1"}], "name": "x"}))
    field = compose.load_field("orchard_x", 7, str(tmp_path))
    assert field["world"] == "orchard_x" and field["seed"] == 7 and field["name"] == "x"
    h = compose.registry_hash(field)
    assert len(h) == 40 and h != compose.registry_hash({"trees": [{"tree_id": "OT-2"}]})
    assert compose.registry_hash({}) == "" and compose.load_field("nope", 1, str(tmp_path)) == {"world": "nope", "seed": 1}


def test_compose_rng_is_per_module_and_seeded():
    a = compose.ComposeContext("legacy", 7)
    b = compose.ComposeContext("legacy", 7)
    assert a.rng("trees").uniform() == b.rng("trees").uniform()
    assert a.rng("trees").uniform() != compose.ComposeContext("legacy", 8).rng("trees").uniform()
    assert a.rng("trees").uniform() != a.rng("actors").uniform()


def test_set_home_on_an_unknown_joint_fails_loudly():
    def attach(spec, field, ctx):
        ctx.set_home("no_such_joint", [0.0])

    with compose.temporary_module("bad_home", 90, attach):
        with pytest.raises(KeyError, match="no_such_joint"):
            farm_scene.load_farm_model()


# ---------------------------------------------------------------------- hooks
def _stepper():
    m = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.002"/><worldbody><body>'
                                       '<freejoint/><geom size="0.1"/></body></worldbody></mujoco>')
    return m, mujoco.MjData(m)


def _run(reg, m, d, seconds, bus=None):
    bus = bus or hook_registry.Bus()
    for _ in range(int(round(seconds / m.opt.timestep))):
        reg.substep(m, d, bus)
        mujoco.mj_step(m, d)
        reg.tick(m, d, bus)
    return bus


def test_hook_declared_at_50_hz_runs_50_per_sim_second():
    m, d = _stepper()
    reg = hook_registry.HookRegistry()
    seen = {"50": [], "25": [], "7": []}
    reg.register("h50", lambda m_, d_, t, bus: seen["50"].append(t))
    reg.register("h25", lambda m_, d_, t, bus: seen["25"].append(t), rate_hz=25)
    reg.register("h7", lambda m_, d_, t, bus: seen["7"].append(t), rate_hz=7)
    _run(reg, m, d, 1.0)
    assert abs(len(seen["50"]) - 50) <= 1
    assert abs(len(seen["25"]) - 25) <= 1
    assert abs(len(seen["7"]) - 7) <= 1
    assert np.allclose(np.diff(seen["50"][1:]), 0.02)
    # a second sim second keeps the rate (no drift, no burst)
    _run(reg, m, d, 1.0)
    assert abs(len(seen["50"]) - 100) <= 1


def test_substep_hook_runs_before_every_physics_step():
    m, d = _stepper()
    reg = hook_registry.HookRegistry()
    times = []
    reg.register("mast_interp", lambda m_, d_, t, bus: times.append(t), every_step=True, mutates_model=True)
    assert reg.has_substep and reg.model_mutating() == ["mast_interp"]
    _run(reg, m, d, 0.2)
    assert len(times) == 100
    assert times[0] == 0.0 and np.allclose(np.diff(times), 0.002)  # called before the step it precedes
    assert reg.table() == [{"name": "mast_interp", "rate_hz": None, "every_step": True, "mutates_model": True,
                            "topics_in": []}]


def test_hook_reset_rebases_and_calls_reset_fn():
    m, d = _stepper()
    reg = hook_registry.HookRegistry()
    resets = []
    h = reg.register("r", lambda *a: None, reset=lambda m_, d_: resets.append(1))
    _run(reg, m, d, 0.5)
    assert h.calls > 0
    mujoco.mj_resetData(m, d)
    reg.reset(m, d)
    assert h.calls == 0 and resets == [1]
    _run(reg, m, d, 1.0)
    assert abs(h.calls - 50) <= 1


def test_hook_registration_rules_and_bus():
    reg = hook_registry.HookRegistry()
    reg.register("a", lambda *a: None, topics_in=["/nandi/mast/cmd"])
    with pytest.raises(ValueError):
        reg.register("a", lambda *a: None)
    with pytest.raises(ValueError):
        reg.register("b", lambda *a: None, rate_hz=0)
    with pytest.raises(ValueError):
        reg.register("c", lambda *a: None, topics_in=["mast/cmd"])
    assert reg.topics_in() == ["/nandi/mast/cmd"] and len(reg) == 1
    reg.unregister("a")
    assert len(reg) == 0
    bus = hook_registry.Bus()
    bus.publish("/nandi/beacon", {"state": "amber"})
    with pytest.raises(ValueError):
        bus.publish("beacon", 1)
    assert bus.drain() == [("/nandi/beacon", '{"state":"amber"}')] and bus.drain() == []
    bus.deliver("/nandi/mast/cmd", {"ext": 0.6})
    assert bus.latest("/nandi/mast/cmd") == {"ext": 0.6} and bus.latest("/nandi/x", 3) == 3


def test_load_all_calls_register_of_every_hook_file(monkeypatch):
    mod = types.ModuleType("atharv_nandi.sim.hooks.wp0_dummy")
    mod.register = lambda reg: reg.register("wp0_dummy", lambda *a: None, rate_hz=10)
    monkeypatch.setitem(sys.modules, "atharv_nandi.sim.hooks.wp0_dummy", mod)
    reg = hook_registry.load_all(names=["wp0_dummy"])
    assert [h.name for h in reg.hooks()] == ["wp0_dummy"]
    broken = types.ModuleType("atharv_nandi.sim.hooks.wp0_broken")
    monkeypatch.setitem(sys.modules, "atharv_nandi.sim.hooks.wp0_broken", broken)
    with pytest.raises(AttributeError):
        hook_registry.load_all(names=["wp0_broken"])
    # the real directory loads (no hook files yet is fine)
    assert isinstance(hook_registry.load_all(), hook_registry.HookRegistry)
    assert all(not n.startswith("_") for n in hook_registry.hook_files())


# ---------------------------------------------------------------------- interface stubs (plug-in contracts)
from atharv_nandi.core import interfaces as ifc  # noqa: E402


def test_mast_request_validation_and_arbiter_priority():
    preset = ifc.MastRequest("preset", 1.6, "tree360 high band", t=3.0)
    recovery = ifc.MastRequest("recovery", 0.0, "low_battery_return", t=1.0)
    safety = ifc.MastRequest("safety", 0.0, "wind", t=0.5)
    estop = ifc.MastRequest("estop", None, "operator", t=0.1)
    assert ifc.arbitrate([preset, recovery, safety, estop]) is estop
    assert ifc.arbitrate([preset, recovery, safety]) is safety
    assert ifc.arbitrate([preset, recovery]) is recovery
    newer = ifc.MastRequest("preset", 0.6, "scan mid band", t=4.0)
    assert ifc.arbitrate([preset, newer]) is newer and ifc.arbitrate([]) is None
    assert estop.priority < safety.priority < recovery.priority < preset.priority
    for bad in (dict(source="web", target_ext=0.1, reason="x"), dict(source="preset", target_ext=1.7, reason="x"),
                dict(source="preset", target_ext=-0.1, reason="x"), dict(source="preset", target_ext=0.1, reason="")):
        with pytest.raises(ValueError):
            ifc.MastRequest(**bad)


def test_gate_decision_needs_a_reason_to_refuse():
    ok = ifc.GateDecision.ok("crown_raise", margin_m=0.14)
    assert ok and ok.values == {"margin_m": 0.14}
    no = ifc.GateDecision.refuse("crown_walk", "crown_gate", tripod_margin_m=0.029)
    assert not no and no.reason == "crown_gate"
    with pytest.raises(ValueError):
        ifc.GateDecision(False, "crown_walk")
    with pytest.raises(ValueError):
        ifc.GateDecision(True, "")


def test_tree_observation_and_preset_step_shapes():
    obs = ifc.TreeObservation("OT-1", "mid", 12.5, "crown_cam", (1.9, 0, 1.7, 1, 0, 0, 0), azimuth_deg=45.0)
    assert obs.detections == [] and obs.masked_pct == 0.0
    with pytest.raises(ValueError):
        ifc.TreeObservation("OT-1", "top", 0.0, "crown_cam", (0,) * 7)
    with pytest.raises(ValueError):
        ifc.TreeObservation("OT-1", "low", 0.0, "ring_0", (0, 0, 0))
    step = ifc.PresetStep("tree360", "orbit_station", {"stations": 8}, ("rtk_fixed",), ("person_pause",), 165.0)
    assert step.params["stations"] == 8 and step.timeout_s == 165.0


def test_null_stubs_satisfy_their_protocols():
    tool, transfer, mast = ifc.NullToolActuation(), ifc.NullTransferControl(), ifc.NullMastControl()
    assert isinstance(tool, ifc.ToolActuation) and isinstance(transfer, ifc.TransferControl)
    assert isinstance(mast, ifc.MastControl)
    tool.set(0.5)
    tool.open_blade()
    tool.stop()
    tool.set(1.0)                      # refused after stop (e-stop codes against this)
    assert tool.state()["value"] == 0.0 and tool.state()["stopped"] and tool.log[-1] == "set refused (stopped)"
    transfer.pause()
    assert transfer.paused and not transfer.busy()
    transfer.resume()
    assert not transfer.paused
    d = mast.submit(ifc.MastRequest("preset", 0.6, "scan"))
    assert not d and d.reason == "not_in_mvp"
    mast.brake()
    assert mast.state()["braked"] and mast.state()["ext_m"] == 0.0
