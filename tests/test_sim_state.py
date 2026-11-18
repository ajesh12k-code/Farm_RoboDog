"""WP0 sim-state envelope + mutation-provider registry (MVP-SPEC 3.2 / 5.7, ORCHARD-MVP-SPEC 3.2).

``camera_rig.set_state`` round trip restores qpos, mocap, eq_active and every registered provider's model
mutations (two dummy providers: a mast section body_pos and pool eq_data + hidden geoms); a world_id
mismatch is refused. ROS-free, headless, each test well under 10 s.
"""
import contextlib
import json

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import camera_rig
from atharv_nandi.core import farm_scene
from atharv_nandi.core import model_layout as mlay
from atharv_nandi.core import sim_state
from test_compose import N_POOL, extended_world


@pytest.fixture(autouse=True)
def isolated_providers():
    """These tests define their own providers: hide the ones real WP hook / module files registered at import
    (WP1 ``beacon``, later WP24 ``mast`` ...) and restore them afterwards. Discovery imports each file once, so
    a hidden provider is not re-registered during the test."""
    sim_state.discover_providers()
    saved = dict(sim_state._PROVIDERS)  # noqa: SLF001 - test isolation of the module registry
    sim_state._PROVIDERS.clear()        # noqa: SLF001
    try:
        yield
    finally:
        sim_state._PROVIDERS.clear()    # noqa: SLF001
        sim_state._PROVIDERS.update(saved)  # noqa: SLF001


def _ids(m, kind, names):
    return [mujoco.mj_name2id(m, kind, n) for n in names]


def _mast_body(m):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "test_mast_s1")


def _pool_eqs(m):
    return _ids(m, mujoco.mjtObj.mjOBJ_EQUALITY, [f"test_pool_eq_{i}" for i in range(N_POOL)])


def _pool_geoms(m):
    return _ids(m, mujoco.mjtObj.mjOBJ_GEOM, [f"test_pool_{i}_geom" for i in range(N_POOL)])


def _pack_mast(m):
    return {"body_pos": [float(v) for v in m.body_pos[_mast_body(m)]]}


def _unpack_mast(m, data):
    m.body_pos[_mast_body(m)] = data["body_pos"]


def _pack_pool(m):
    return {"eq_data": [[float(v) for v in m.eq_data[e]] for e in _pool_eqs(m)],
            "hidden_geoms": [int(g) for g in _pool_geoms(m) if m.geom_contype[g] == 0]}


def _unpack_pool(m, data):
    for e, row in zip(_pool_eqs(m), data["eq_data"]):
        m.eq_data[e] = row
    hidden = set(data["hidden_geoms"])
    for g in _pool_geoms(m):
        m.geom_contype[g] = 0 if g in hidden else 1
        m.geom_rgba[g, 3] = 0.0 if g in hidden else 1.0


@contextlib.contextmanager
def two_providers():
    sim_state.register_provider("test_mast", _pack_mast, _unpack_mast, version=1)
    sim_state.register_provider("test_pool", _pack_pool, _unpack_pool, version=2)
    try:
        yield
    finally:
        sim_state.unregister_provider("test_mast")
        sim_state.unregister_provider("test_pool")


def _live_world():
    """A 'sim process' model + a separately built 'renderer' model of the same extended world."""
    with extended_world():
        sim_m = farm_scene.load_farm_model()
        wid = mlay.current().world_id
        ren_m = camera_rig.build_perception_model()
    assert camera_rig.current_world_id() == wid
    return sim_m, ren_m, wid


def _mutate(m, d):
    rng = np.random.default_rng(23)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(50):
        d.ctrl[:] = m.key_ctrl[0] + rng.uniform(-0.05, 0.05, m.nu)
        mujoco.mj_step(m, d)
    d.mocap_pos[:] = rng.uniform(-3, 3, d.mocap_pos.shape)
    q = rng.normal(size=d.mocap_quat.shape)
    d.mocap_quat[:] = q / np.linalg.norm(q, axis=1, keepdims=True)
    eqs = _pool_eqs(m)
    d.eq_active[eqs[0]] = 1         # a mango slot assigned (connect)
    d.eq_active[eqs[7]] = 1         # a guava slot assigned (weld)
    d.eq_active[0] = 0              # a tool left its rack weld
    m.eq_data[eqs[0], 0:3] = [0.01, -0.02, 0.03]
    m.eq_data[eqs[7], 3:10] = [0.1, 0.2, 0.3, 0.7071, 0.0, 0.7071, 0.0]
    m.geom_contype[_pool_geoms(m)[4]] = 0
    m.body_pos[_mast_body(m)] = [-0.30, 0.0, 0.05 + 0.533]


def test_set_state_round_trip_restores_qpos_mocap_eq_active_and_every_provider():
    with two_providers():
        sim_m, ren_m, wid = _live_world()
        sim_d, ren_d = mujoco.MjData(sim_m), mujoco.MjData(ren_m)
        _mutate(sim_m, sim_d)
        text = sim_state.encode(sim_state.pack(sim_m, sim_d, wid))
        state = json.loads(text)
        assert set(state["mutations"]["providers"]) == {"test_mast", "test_pool"}
        assert state["mutations"]["providers"]["test_pool"]["v"] == 2

        ok, reason = camera_rig.set_state(ren_m, ren_d, text)   # default: this world's id
        assert ok, reason
        assert ren_d.qpos.tobytes() == sim_d.qpos.tobytes()      # bit-exact
        assert ren_d.mocap_pos.tobytes() == sim_d.mocap_pos.tobytes()
        assert ren_d.mocap_quat.tobytes() == sim_d.mocap_quat.tobytes()
        np.testing.assert_array_equal(ren_d.eq_active, sim_d.eq_active)
        mb = _mast_body(ren_m)
        np.testing.assert_array_equal(ren_m.body_pos[mb], sim_m.body_pos[_mast_body(sim_m)])
        np.testing.assert_array_equal(ren_m.eq_data[_pool_eqs(ren_m)], sim_m.eq_data[_pool_eqs(sim_m)])
        np.testing.assert_array_equal(ren_m.geom_contype[_pool_geoms(ren_m)], sim_m.geom_contype[_pool_geoms(sim_m)])
        assert ren_d.time == sim_d.time
        # mutations were applied BEFORE kinematics: the moved mast section is where the sim has it
        mujoco.mj_kinematics(sim_m, sim_d)
        np.testing.assert_allclose(ren_d.xpos[mb], sim_d.xpos[_mast_body(sim_m)], atol=1e-12)
        crop = mujoco.mj_name2id(ren_m, mujoco.mjtObj.mjOBJ_BODY, "crop_item_3")
        np.testing.assert_allclose(ren_d.xpos[crop], ren_d.qpos[mlay.Layout.from_model(ren_m).crop_qpos(3)][:3])
        # and packing the renderer again reproduces the message
        assert sim_state.encode(sim_state.pack(ren_m, ren_d, wid)) == text


def test_world_id_mismatch_is_refused_and_nothing_is_written():
    with two_providers():
        sim_m, ren_m, wid = _live_world()
        sim_d, ren_d = mujoco.MjData(sim_m), mujoco.MjData(ren_m)
        _mutate(sim_m, sim_d)
        state = sim_state.pack(sim_m, sim_d, "000000000000")
        before = ren_d.qpos.copy(), ren_m.body_pos.copy()
        ok, reason = camera_rig.set_state(ren_m, ren_d, state)
        assert not ok and "world_id mismatch" in reason
        np.testing.assert_array_equal(ren_d.qpos, before[0])
        np.testing.assert_array_equal(ren_m.body_pos, before[1])


def test_legacy_world_ids_differ_from_the_extended_world():
    legacy = farm_scene.world_identity()[1]
    with extended_world():
        assert farm_scene.world_identity()[1] != legacy


def test_refusals_size_provider_version_schema():
    with two_providers():
        sim_m, ren_m, wid = _live_world()
        d = mujoco.MjData(sim_m)
        mujoco.mj_resetDataKeyframe(sim_m, d, 0)
        good = sim_state.pack(sim_m, d, wid)
        rd = mujoco.MjData(ren_m)
        cases = {
            "qpos has": dict(good, qpos=good["qpos"][:-1]),
            "mocap_pos has": dict(good, mocap_pos=good["mocap_pos"][:-3]),
            "eq_active has": dict(good, eq_active=[]),
            "schema": dict(good, schema="other/1"),
            "mutations version": dict(good, mutations={"v": 9, "providers": {}}),
            "no provider registered": dict(good, mutations={"v": 1, "providers": dict(
                good["mutations"]["providers"], ghost={"v": 1, "data": {}})}),
            "version 1 != 2": dict(good, mutations={"v": 1, "providers": dict(
                good["mutations"]["providers"], test_pool={"v": 1, "data": {}})}),
            "lacks mutations": dict(good, mutations={"v": 1, "providers": {"test_mast": good["mutations"]["providers"]["test_mast"]}}),
        }
        for needle, st in cases.items():
            ok, reason = camera_rig.set_state(ren_m, rd, st, world_id=wid)
            assert not ok and needle in reason, (needle, reason)
        ok, reason = camera_rig.set_state(ren_m, rd, "{not json")
        assert not ok and "undecodable" in reason
        with pytest.raises(sim_state.SimStateError):
            sim_state.decode(json.dumps({"schema": "x"}))
        # non-strict receivers may lack a provider block (e.g. a tool that only renders qpos)
        partial = dict(good, mutations={"v": 1, "providers": {}})
        assert sim_state.apply(ren_m, rd, partial, wid, strict_providers=False) == (True, "ok")
        assert sim_state.apply(ren_m, rd, partial, None, strict_providers=True)[0] is False


def test_provider_pack_must_return_a_dict():
    sim_state.register_provider("test_bad", lambda m: [1, 2], lambda m, d: None)
    try:
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        with pytest.raises(sim_state.SimStateError):
            sim_state.pack(m, mujoco.MjData(m), "x")
    finally:
        sim_state.unregister_provider("test_bad")
    with pytest.raises(ValueError):
        sim_state.register_provider("", lambda m: {}, lambda m, d: None)


def test_legacy_world_round_trip_through_the_perception_model():
    m = farm_scene.load_farm_model()
    wid = mlay.current().world_id
    pm = camera_rig.build_perception_model()
    d, pd = mujoco.MjData(m), mujoco.MjData(pm)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(25):
        mujoco.mj_step(m, d)
    st = sim_state.pack(m, d, wid)
    assert st["nmocap"] == 0 and st["neq"] == 0 and st["mutations"] == {"v": 1, "providers": {}}
    assert camera_rig.set_state(pm, pd, st) == (True, "ok")
    assert pd.qpos.tobytes() == d.qpos.tobytes()
    # same result as the legacy set_qpos path
    pd2 = mujoco.MjData(pm)
    assert camera_rig.set_qpos(pm, pd2, d.qpos.tolist())
    np.testing.assert_array_equal(pd.xpos, pd2.xpos)


# ---------------------------------------------------------------------- malformed messages (never raise, never half-write)
def test_malformed_messages_are_refused_without_raising_or_writing():
    calls = []

    def unpack_log(m, data):
        calls.append(data)
        m.body_pos[_mast_body(m)] = data.get("body_pos", m.body_pos[_mast_body(m)])

    sim_state.register_provider("test_mast", _pack_mast, unpack_log, version=1)
    try:
        sim_m, ren_m, wid = _live_world()
        d = mujoco.MjData(sim_m)
        mujoco.mj_resetDataKeyframe(sim_m, d, 0)
        good = sim_state.pack(sim_m, d, wid)
        good["mutations"]["providers"]["test_mast"]["data"]["body_pos"] = [9.0, 9.0, 9.0]
        rd = mujoco.MjData(ren_m)
        mutations = good["mutations"]
        cases = {
            "qpos strings": dict(good, qpos=["bad"] * ren_m.nq),
            "qpos number": dict(good, qpos=5),
            "qpos nan": dict(good, qpos=[float("nan")] * ren_m.nq),
            "qpos inf": dict(good, qpos=[float("inf")] + good["qpos"][1:]),
            "qpos nested": dict(good, qpos=[[0.0]] * ren_m.nq),
            "mocap bool": dict(good, mocap_pos=[True] * len(good["mocap_pos"])),
            "eq_active 2": dict(good, eq_active=[2] * ren_m.neq),
            "eq_active str": dict(good, eq_active="1" * ren_m.neq),
            "t nan": dict(good, t=float("nan")),
            "mutations v str": dict(good, mutations={"v": "x", "providers": {}}),
            "mutations list": dict(good, mutations=[1]),
            "providers list": dict(good, mutations={"v": 1, "providers": [1]}),
            "block not dict": dict(good, mutations={"v": 1, "providers": {"test_mast": 5}}),
            "data not dict": dict(good, mutations={"v": 1, "providers": {"test_mast": {"v": 1, "data": 3}}}),
            "state list": [1, 2],
        }
        assert mutations["providers"]["test_mast"]["v"] == 1
        before = rd.qpos.copy(), ren_m.body_pos.copy()
        for label, st in cases.items():
            ok, reason = camera_rig.set_state(ren_m, rd, st, world_id=wid)
            assert not ok and reason, label
            assert sim_state.check(ren_m, st, wid)[0] is False, label
        assert calls == [], "a provider was unpacked for a refused message"
        np.testing.assert_array_equal(rd.qpos, before[0])
        np.testing.assert_array_equal(ren_m.body_pos, before[1])
        assert camera_rig.set_state(ren_m, rd, good, world_id=wid) == (True, "ok")
    finally:
        sim_state.unregister_provider("test_mast")


def test_a_raising_unpack_rolls_back_the_providers_already_applied():
    def boom(m, data):
        raise KeyError("eq_data")

    sim_state.register_provider("test_mast", _pack_mast, _unpack_mast, version=1)
    sim_state.register_provider("test_zz_boom", lambda m: {}, boom, version=1)   # unpacked after test_mast
    try:
        sim_m, ren_m, wid = _live_world()
        d = mujoco.MjData(sim_m)
        mujoco.mj_resetDataKeyframe(sim_m, d, 0)
        sim_m.body_pos[_mast_body(sim_m)] = [-0.3, 0.0, 0.7]
        st = sim_state.pack(sim_m, d, wid)
        rd = mujoco.MjData(ren_m)
        before = ren_m.body_pos.copy(), rd.qpos.copy()
        ok, reason = camera_rig.set_state(ren_m, rd, st, world_id=wid)
        assert not ok and "test_zz_boom" in reason and "unpack failed" in reason
        np.testing.assert_array_equal(ren_m.body_pos, before[0])
        np.testing.assert_array_equal(rd.qpos, before[1])
    finally:
        sim_state.unregister_provider("test_mast")
        sim_state.unregister_provider("test_zz_boom")


def test_encode_and_decode_reject_non_finite_numbers():
    m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>")
    d = mujoco.MjData(m)
    d.qpos[0] = float("nan")
    with pytest.raises(sim_state.SimStateError):
        sim_state.encode(sim_state.pack(m, d, "x"))
    text = sim_state.encode(sim_state.pack(m, mujoco.MjData(m), "x"))
    assert "NaN" not in text and json.loads(text)["nq"] == 7
    with pytest.raises(sim_state.SimStateError):
        sim_state.decode(text.replace('"t":0.0', '"t":NaN'))
    ok, reason = camera_rig.set_state(m, d, text.replace('"t":0.0', '"t":Infinity'), world_id=None)
    assert not ok and "non-finite" in reason


def test_providers_declared_in_hook_files_are_discovered_by_every_process(tmp_path, monkeypatch):
    import atharv_nandi.sim.hooks as hook_registry
    (tmp_path / "wp0_prov_probe.py").write_text(
        "from atharv_nandi.core import sim_state\n"
        "sim_state.register_provider('wp0_prov_probe', lambda m: {}, lambda m, d: None)\n"
        "def register(hooks):\n    pass\n")
    monkeypatch.setattr(hook_registry, "__path__", list(hook_registry.__path__) + [str(tmp_path)])
    monkeypatch.setattr(hook_registry, "HOOKS_DIR", str(tmp_path))
    try:
        assert "wp0_prov_probe" in sim_state.discover_providers(force=True)
    finally:
        sim_state.unregister_provider("wp0_prov_probe")
        import sys
        sys.modules.pop("atharv_nandi.sim.hooks.wp0_prov_probe", None)
