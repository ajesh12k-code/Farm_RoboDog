"""WP4 world events, lighting and the world_events hook (ORCHARD-MVP-SPEC 1.2, 3.6, 3.7, 4.5).

Acceptance (WP4): reset after a tour diff-equal to a fresh build; 2,000 steps ``orchard_lite``
within budget (``slow``). Plus the scheduler, the WP7 hand-off of wind / rain, dusk and dawn lighting with its
sim-state provider, and the world-state topic.
"""
import json
import time

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene, sim_state
from atharv_nandi.core import world_events as wev
from atharv_nandi.core.sim_driver import SimDriver
from atharv_nandi.sim.hooks import Bus, HookRegistry, load_all
from atharv_nandi.sim.world import field_spec, lighting

PHYSICS_BUDGET_MS = 2.0     # ORCHARD-MVP-SPEC 3.7: physics budget 2 ms per step


@pytest.fixture
def world(monkeypatch):
    def build(name, **env):
        monkeypatch.setenv("NANDI_WORLD", name)
        monkeypatch.setenv("NANDI_NOVNC", "0")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return farm_scene.load_farm_model()
    return build


def model_bytes(m) -> bytes:
    buf = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return buf.tobytes()


# ---------------------------------------------------------------------- scheduler (pure)
def test_scheduler_triggers():
    s = wev.EventScheduler(
        [{"id": "a", "type": "marker", "trigger": {"mode": "at_s", "t": 2.0}},
         {"id": "b", "type": "marker", "trigger": {"mode": "robot_within", "xy": [5, 0], "r": 1.0, "after_s": 3.0}},
         {"id": "c", "type": "light", "preset": "dusk", "trigger": {"mode": "cmd", "after_s": 0.5}},
         {"id": "d", "type": "marker", "trigger": {"mode": "always", "repeat_s": 1.0}}],
        [{"id": "dog", "start": {"mode": "at_s", "t": 1.0}}, {"id": "cat", "start": {"mode": "cmd"}}])
    ids = lambda evs: [e.id for e in evs]
    assert ids(s.update(0.0, (0, 0))) == ["d"]
    assert ids(s.update(0.5, (0, 0))) == []
    assert ids(s.update(1.0, (0, 0))) == ["d", "actor:dog:start"]
    assert ids(s.update(2.0, (4.5, 0))) == ["a", "d"]                 # b arms at 2.0
    assert ids(s.update(4.9, (9, 0))) == ["d"]                        # still pending (armed)
    assert ids(s.update(5.0, (9, 0))) == ["b"]
    s.fire("c", 6.0)
    assert ids(s.update(6.2, None)) == ["d"]
    assert ids(s.update(6.5, None)) == ["c"]
    assert ids(s.update(50.0, (5, 0))) == ["d"]                       # once-only events stay fired
    with pytest.raises(KeyError):
        s.fire("zzz", 1.0)
    s.reset()
    assert s.fired_ids() == [] and ids(s.update(0.0)) == ["d"]
    with pytest.raises(ValueError):
        wev.EventScheduler([{"id": "x", "type": "earthquake"}])
    with pytest.raises(ValueError):
        wev.EventScheduler([{"id": "x", "type": "marker"}, {"id": "x", "type": "marker"}])
    with pytest.raises(ValueError):
        wev.Trigger.from_dict({"mode": "sometimes"})


def test_parse_command():
    assert wev.parse_command('{"fire": "gust_11"}') == {"fire": "gust_11"}
    assert wev.parse_command(b'{"light": "dusk", "seq": 3}') == {"light": "dusk", "seq": 3}
    assert wev.parse_command({"actor": "dog_1", "action": "start"})["actor"] == "dog_1"
    for bad in ("not json", "[1]", '{"drop_table": 1}', '{"seq": 1}', 5):
        assert wev.parse_command(bad) is None
    # value types are checked too (review: these faulted the hook and safe-held the robot)
    for bad in ('{"fog": "thick"}', '{"light": ["dusk"]}', '{"fog": NaN}', '{"fog": Infinity}', '{"fog": true}',
                '{"reset_counters": "no"}', '{"reset_counters": 1}', '{"fire": 7}', '{"fire": ""}',
                '{"actor": {"id": 1}}', '{"actor": "dog_1", "action": "explode"}', '{"action": "start"}',
                '{"light": "dusk", "seq": [1]}'):
        cmd, why = wev.check_command(bad)
        assert cmd is None and why != "ok", bad
    assert wev.parse_command('{"fog": 1e9}') == {"fog": 1.0}                # clamped to [0, 1]
    assert wev.parse_command('{"fog": -3}') == {"fog": 0.0}
    raw = {"fire": "gust_11", "seq": 4}
    assert wev.parse_command(raw) == raw and wev.parse_command(raw) is not raw


def test_every_field_event_and_actor_start_parses():
    for w in field_spec.ORCHARD_WORLDS:
        f = field_spec.load(w)
        s = wev.EventScheduler(f.get("events") or [], f.get("actors") or [])
        assert s.events
    demo = field_spec.load("orchard_demo")
    ev = {e["id"]: e for e in demo["events"]}
    assert ev["gust_11"]["script"]["gusts"][0]["peak_mps"] == 11.0
    assert {"light_dusk", "light_dawn", "fog_dust", "rain_shower"} <= set(ev)


# ---------------------------------------------------------------------- lighting
def test_lighting_presets_and_ready_thresholds():
    assert set(lighting.PRESETS) >= {"morning", "noon", "overcast", "dusk", "dawn"}
    for p in lighting.PRESETS.values():
        assert np.linalg.norm(p["dir"]) == pytest.approx(1.0) and p["dir"][2] < 0
    dusk, dawn, noon = (lighting.state(n) for n in ("dusk", "dawn", "noon"))
    assert noon["camera_ok"] and not dusk["camera_ok"] and not dawn["camera_ok"]
    assert dawn["dawn_mode_ok"] and dawn["dawn_mode_window"]          # LEAF SCAN dawn mode (OS 4.5)
    assert dusk["camera_noise_sigma"] == pytest.approx(8 / 255) and dusk["motion_blur_px"] == 3
    assert lighting.fog_range(0.0) is None and lighting.fog_range(2.0)[0] == 0.0
    with pytest.raises(KeyError):
        lighting.preset("eclipse")


def test_orchard_sun_and_lighting_provider_round_trip(world):
    m = world("orchard_lite")
    lid = lighting.sun_id(m)
    sim_state.discover_providers()
    assert lid >= 0
    base = lighting.snapshot(m)
    assert np.allclose(base["dir"], lighting.PRESETS["morning"]["dir"])
    assert lighting.apply_model(m, "dusk", fog=0.5)
    packed = sim_state.pack_mutations(m)["providers"]["lighting"]
    assert packed["data"]["diffuse"] == pytest.approx(lighting.PRESETS["dusk"]["diffuse"])
    other = farm_scene.load_farm_model()                              # a renderer's own copy
    d = mujoco.MjData(other)
    st = sim_state.pack(m, mujoco.MjData(m), "x")
    ok, why = sim_state.apply(other, d, sim_state.decode(sim_state.encode(st)), None)
    assert ok, why
    assert lighting.snapshot(other) == lighting.snapshot(m)
    leg = world("legacy")
    assert lighting.snapshot(leg) is None and not lighting.apply_model(leg, "dusk")
    assert "lighting" not in sim_state.pack_mutations(leg)["providers"]     # legacy: empty mutations block
    world("orchard_lite")
    assert "lighting" in sim_state.pack_mutations(leg)["providers"] and lighting._pack(leg) == {}


# ---------------------------------------------------------------------- hook
def _hooked(m):
    from atharv_nandi.sim.hooks import world_events as hook
    reg, bus = HookRegistry(), Bus()
    hook.register(reg)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    reg.reset(m, d)
    return hook, reg, bus, d


def _tick(m, d, reg, bus, secs, dt=0.02):
    for _ in range(int(round(secs / dt))):
        d.time += dt
        reg.tick(m, d, bus)


def test_wind_and_rain_events_go_to_the_wp7_platform_commands(world):
    m = world("orchard_lite")
    hook, reg, bus, d = _hooked(m)
    _tick(m, d, reg, bus, 0.1)
    assert bus.latest(wev.WIND_CMD_TOPIC) is None
    bus.deliver(hook.CMD_TOPIC, {"fire": "gust_11"})
    _tick(m, d, reg, bus, 0.04)
    script = bus.latest(wev.WIND_CMD_TOPIC)
    assert script["gusts"][0]["peak_mps"] == 11.0
    # the fire time travels with the hand-off: gust t0 is seconds after t_fire, not after WP7's read
    assert script["t_fire"] == pytest.approx(0.12, abs=0.021) and script["event"] == "gust_11"
    events = [json.loads(p) for t, p in bus.outbox if t == hook.EVENTS_TOPIC]
    assert any(e["type"] == "wind" and e["id"] == "gust_11" for e in events)
    mm = world("orchard_demo_monsoon")
    hook, reg, bus, d = _hooked(mm)
    _tick(mm, d, reg, bus, 0.04)
    assert bus.latest(wev.WEATHER_CMD_TOPIC)["rain"]["mm_h"] == 15.0     # monsoon rain from t = 0
    assert bus.latest(wev.WEATHER_CMD_TOPIC)["t_fire"] <= 0.02
    _tick(mm, d, reg, bus, 2.0)
    bus.deliver(hook.CMD_TOPIC, {"fire": "rain_shower"})
    _tick(mm, d, reg, bus, 0.04)
    rain = bus.latest(wev.WEATHER_CMD_TOPIC)
    assert rain["event"] == "rain_shower" and rain["t_fire"] >= 2.0 and rain["rain"]["t0"] == 0.0
    assert bus.values["world"]["light"]["preset"] == "overcast"


BAD_COMMANDS = ('{"fog": "thick"}', '{"light": ["dusk"]}', '{"fog": NaN}', '{"reset_counters": "no"}',
                '{"actor": ["x"]}', '{"fire": {"a": 1}}', b'\xff\xfe', '{"fog": 1e400}', "null", "[]")


def test_malformed_commands_are_refused_and_never_fault_the_hook(world):
    """Review: a malformed /nandi/world/cmd faulted world_events (hook_fault safe-hold, actors frozen)."""
    m = world("orchard_lite")
    drv = SimDriver(m, hooks=load_all(names=["world_events"]))
    from atharv_nandi.sim.hooks import world_events as hook
    drv.wait(0.1)
    rt = hook.runtime_for(m)
    rt.counter.count = 3
    for bad in BAD_COMMANDS:
        drv.deliver(hook.CMD_TOPIC, bad)
        drv.wait(0.04)
    assert not drv.hook_faults and not drv.holding, drv.hook_faults
    refused = [json.loads(p) for _t, tp, p in drv.published if tp == hook.EVENTS_TOPIC and '"cmd_refused"' in p]
    assert len(refused) == len(BAD_COMMANDS) and all(e["reason"] for e in refused)
    assert rt.counter.count == 3                                          # "no" did not reset the counter
    assert bus_fog(drv) == 0.0 and m.vis.map.fogend == pytest.approx(lighting.snapshot(m)["fogend"])
    drv.deliver(hook.CMD_TOPIC, '{"fog": 1e9}')                          # out of range: clamped, not published raw
    drv.wait(0.04)
    assert bus_fog(drv) == 1.0
    drv.deliver(hook.CMD_TOPIC, {"reset_counters": True, "seq": 9})
    drv.wait(0.04)
    assert rt.counter.count == 0
    assert drv.hooks.faulted() == {} and rt.cmd_seen == drv.bus.arrival_count(hook.CMD_TOPIC)    # all consumed


def bus_fog(drv):
    return drv.bus.values["world"]["light"]["fog"]


def test_the_same_command_object_delivered_twice_is_applied_twice(world):
    """Review: de-duplication by identity dropped a repeated in-process command (interned string literal)."""
    m = world("orchard_lite")
    hook, reg, bus, d = _hooked(m)
    for _ in range(2):
        bus.deliver(hook.CMD_TOPIC, '{"fire": "light_dusk"}')
        _tick(m, d, reg, bus, 0.04)
    fired = [json.loads(p) for t, p in bus.outbox if t == hook.EVENTS_TOPIC and '"light_dusk"' in p]
    assert len(fired) == 2
    reg.reset(m, d)                                                      # nothing delivered before it is replayed
    _tick(m, d, reg, bus, 0.04)
    assert sum('"light_dusk"' in p for t, p in bus.outbox if t == hook.EVENTS_TOPIC) == 2


def test_dusk_dawn_and_fog_commands_change_the_model_and_publish_state(world):
    m = world("orchard_demo")
    hook, reg, bus, d = _hooked(m)
    before = lighting.snapshot(m)
    bus.deliver(hook.CMD_TOPIC, '{"fire": "light_dusk"}')
    _tick(m, d, reg, bus, 0.04)
    assert lighting.snapshot(m)["diffuse"] == pytest.approx(lighting.PRESETS["dusk"]["diffuse"])
    st = bus.values["world"]
    assert st["light"]["preset"] == "dusk" and not st["light"]["camera_ok"]
    bus.deliver(hook.CMD_TOPIC, {"light": "dawn", "fog": 0.4})
    _tick(m, d, reg, bus, 0.04)
    assert bus.values["world"]["light"]["preset"] == "dawn" and bus.values["world"]["light"]["fog"] == 0.4
    assert m.vis.map.fogend < before["fogend"] or before["fogend"] != m.vis.map.fogend
    bus.deliver(hook.CMD_TOPIC, {"light": "eclipse"})
    _tick(m, d, reg, bus, 0.04)
    assert any('"cmd_refused"' in p for t, p in bus.outbox)
    bus.deliver(hook.CMD_TOPIC, "garbage")
    _tick(m, d, reg, bus, 0.04)
    states = [json.loads(p) for t, p in bus.outbox if t == hook.STATE_TOPIC]
    assert states and all(s["world"] == "orchard_demo" and s["source"] == "sim_truth" for s in states)
    reg.reset(m, d)
    assert lighting.snapshot(m) == before


def test_reset_after_a_tour_is_diff_equal_to_a_fresh_build(world):
    m = world("orchard_demo")
    drv = SimDriver(m, hooks=load_all(names=["world_events"]))
    from atharv_nandi.sim.hooks import world_events as hook
    drv.walk(0.8, 0.0, 0.0, seconds=3.0)
    for cmd in ({"fire": "light_dusk"}, {"fire": "fog_dust"}, {"fire": "gust_11"},
                {"actor": "worker_1", "action": "start"}, {"actor": "lying_1", "action": "start"}):
        drv.deliver(hook.CMD_TOPIC, cmd)
        drv.wait(0.1)
    drv.walk(0.0, 0.0, 0.8, seconds=2.0)
    drv.wait(2.0)
    rt = hook.runtime_for(m)
    assert rt.scheduler.fired_ids() and rt.actors["worker_1"].state != "idle"
    assert lighting.snapshot(m)["diffuse"] != lighting.PRESETS["morning"]["diffuse"]
    moved = drv.d.mocap_pos.copy()
    drv.reset()

    fresh_m = farm_scene.load_farm_model()
    fresh = SimDriver(fresh_m, hooks=load_all(names=["world_events"]))
    assert model_bytes(m) == model_bytes(fresh_m)                         # every model field, lighting included
    for name in ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "eq_active", "ctrl", "qacc_warmstart"):
        assert np.array_equal(getattr(drv.d, name), getattr(fresh.d, name)), name
    assert not np.array_equal(moved, drv.d.mocap_pos)
    assert sim_state.pack(m, drv.d, "w") == sim_state.pack(fresh_m, fresh.d, "w")
    assert rt.scheduler.fired_ids() == [] and rt.counter.count == 0
    assert all(a.state in ("idle", "parked") for a in rt.actors.values())
    # and the next run replays identically
    for x in (drv, fresh):
        x.walk(0.6, 0.0, 0.2, seconds=1.0)
    assert drv.d.qpos.tobytes() == fresh.d.qpos.tobytes()
    assert np.array_equal(drv.d.mocap_pos, fresh.d.mocap_pos)


def _gl_renderer(m):
    try:
        return mujoco.Renderer(m, 96, 128)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no offscreen GL backend (set MUJOCO_GL / DISPLAY): {exc}")


@pytest.mark.gl
def test_fog_reaches_the_rendered_image(world):
    """Review: fog changed only model fields (mjRND_FOG off, range baked into the MjrContext): 0.0 pixel change."""
    from atharv_nandi.core import camera_rig
    from atharv_nandi.core.sim_driver import MjRenderer, View
    world("orchard_lite")
    m = camera_rig.build_perception_model()                              # the perception process's model
    assert lighting.sun_id(m) >= 0
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    _gl_renderer(m).close()
    view = View.look([-3.0, -3.0, 2.0], [4.0, 3.0, 0.5])
    r = MjRenderer(m, 128, 96)
    cam = camera_rig.CameraRenderer(m, camera_rig.HEAD_CAM, 128, 96)
    clear, clear_cam = r.render(m, d, view).astype(int), cam.rgb(d).astype(int)
    clear_rgbd, clear_rgbd_depth = cam.rgbd(d)
    clear_rgbd = clear_rgbd.astype(int)
    clear_depth = cam.depth(d)
    ctx0, depth_map0 = cam.renderer._mjr_context, int(cam.renderer._mjr_context.readDepthMap)
    assert not lighting.fog_active(m)
    assert lighting.apply_model(m, "morning", 0.9)                        # what the fog command does
    assert lighting.fog_active(m)
    fog, fog_cam = r.render(m, d, view).astype(int), cam.rgb(d).astype(int)
    fog_rgbd, depth = cam.rgbd(d)
    diff = {"driver": float(np.abs(fog - clear).mean()), "camera": float(np.abs(fog_cam - clear_cam).mean()),
            "rgbd": float(np.abs(fog_rgbd.astype(int) - clear_rgbd).mean())}
    print("WP4 fog mean pixel change (0-255):", diff)
    assert min(diff.values()) > 5.0, diff
    assert np.isfinite(depth).all()
    # review: the fog-driven MjrContext rebuild dropped readDepthMap (reversed-Z), so the one-pass rgbd depth
    # came back wrong (median 0.861 -> 0.707 m, max error 1941.8 m) with no error. Fog does not move geometry:
    # the rgbd depth and the two-pass depth are unchanged by it
    assert cam.renderer._mjr_context is not ctx0 and cam.renderer._mjr_context.readDepthMap == depth_map0
    assert np.allclose(depth, clear_rgbd_depth, atol=1e-4), float(np.nanmax(np.abs(depth - clear_rgbd_depth)))
    assert np.allclose(cam.depth(d), clear_depth, atol=1e-4)
    # towards the fog colour, and back to the clear image once the fog is cleared
    target = np.array(lighting.FOG_RGBA[:3]) * 255
    assert np.abs(fog.reshape(-1, 3).mean(0) - target).sum() < np.abs(clear.reshape(-1, 3).mean(0) - target).sum()
    lighting.restore(m, lighting.snapshot(farm_scene.load_farm_model()))
    assert not lighting.fog_active(m)
    assert np.array_equal(r.render(m, d, view).astype(int), clear)
    r.close()
    leg = world("legacy")                                                  # legacy: the renderer is not touched
    rl = _gl_renderer(leg)
    assert not lighting.sync_renderer(rl) and rl.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] == 0
    rl.close()


@pytest.mark.slow
def test_2000_steps_orchard_lite_within_budget(world):
    out = {}
    for w in ("legacy", "orchard_lite", "orchard_demo"):
        m = world(w)
        drv = SimDriver(m, hooks=load_all())
        drv.walk(0.6, 0.0, 0.1)                                       # walking + turning: feet on the hfield
        t0 = time.perf_counter()
        drv.run_steps(2000)
        wall = time.perf_counter() - t0
        out[w] = {"ms_per_step": round(1000 * wall / 2000, 4), "mj_step_ms": round(1000 * drv.wall_in_step / 2000, 4),
                  "rtf": round(2000 * m.opt.timestep / wall, 2), "hook_calls": {h.name: h.calls for h in drv.hooks.hooks()},
                  "faults": dict(drv.hook_faults)}
        assert not drv.hook_faults and drv.diverged is None
    print("WP4 2000-step budget:", json.dumps(out))
    assert out["orchard_lite"]["ms_per_step"] <= PHYSICS_BUDGET_MS, out
    assert out["orchard_lite"]["hook_calls"]["world_events"] == pytest.approx(2000 * 0.002 * 50, abs=1)


# ---------------------------------------------------------------------- review round 2
def test_every_world_cmd_within_one_tick_is_applied_in_order(world):
    """Review: the hook kept only the newest /nandi/world/cmd per 20 ms tick (bus.inbox.pop): {light: dusk}
    followed 4 ms later by {fog: 0.5} published only the fog event, dusk vanished without a refusal."""
    m = world("orchard_lite")
    drv = SimDriver(m, hooks=load_all(names=["world_events"]))
    from atharv_nandi.sim.hooks import world_events as hook
    for _ in range(30):
        drv.step()
    drv.deliver(hook.CMD_TOPIC, {"light": "dusk", "seq": 1})
    drv.step()
    drv.step()
    drv.deliver(hook.CMD_TOPIC, {"fog": 0.5, "seq": 2})
    drv.wait(0.1)
    ev = [json.loads(p) for _t, tp, p in drv.published if tp == hook.EVENTS_TOPIC]
    lights = [e for e in ev if e["type"] == "light"]
    assert [(e["seq"], e["preset"], e["fog"]) for e in lights] == [(1, "dusk", 0.0), (2, "dusk", 0.5)], ev
    assert bus_fog(drv) == 0.5 and drv.bus.values["world"]["light"]["preset"] == "dusk"
    assert not any(e["type"] in ("cmd_lost", "cmd_refused") for e in ev)


def test_world_cmds_beyond_the_bus_window_give_cmd_lost(world):
    m = world("orchard_lite")
    hook, reg, bus, d = _hooked(m)
    _tick(m, d, reg, bus, 0.04)
    from atharv_nandi.sim.hooks import HISTORY_N
    n = HISTORY_N + 5
    for i in range(n):                                                     # all within one tick
        bus.deliver(hook.CMD_TOPIC, {"fog": i / 100.0, "seq": i})
    _tick(m, d, reg, bus, 0.02)
    ev = [json.loads(p) for t, p in bus.outbox if t == hook.EVENTS_TOPIC]
    lost = [e for e in ev if e["type"] == "cmd_lost"]
    seqs = [e["seq"] for e in ev if e["type"] == "light"]
    assert len(lost) == 1 and lost[0]["n"] == 5
    assert seqs == list(range(5, n))                                       # the window, oldest first
    assert hook.runtime_for(m).cmd_lost == 5
    _tick(m, d, reg, bus, 0.04)                                            # nothing replayed on the next tick
    assert sum(json.loads(p)["type"] == "light" for t, p in bus.outbox if t == hook.EVENTS_TOPIC) == len(seqs)


def test_field_events_are_validated_at_load():
    """Review: field_spec.validate never checked events[]; a typo raised inside apply_event during a tick."""
    base = field_spec.load("orchard_demo")
    assert field_spec.validate(base) == []
    actor = (base.get("actors") or [{"id": "worker_1"}])[0]["id"]
    bad = (({"id": "x", "type": "light", "preset": "eclipse"}, "light preset"),
           ({"id": "x", "type": "light"}, "light preset"),
           ({"id": "x", "type": "fog", "density": "thick"}, "fog density"),
           ({"id": "x", "type": "fog", "density": float("nan")}, "fog density"),
           ({"id": "x", "type": "actor"}, "actor None"),
           ({"id": "x", "type": "actor", "actor": "ghost"}, "actor 'ghost'"),
           ({"id": "x", "type": "actor", "actor": actor, "action": "explode"}, "action"),
           ({"id": "x", "type": "rain", "mm_h": "lots"}, "mm_h"),
           ({"id": "x", "type": "earthquake"}, "type"),
           ({"id": "x", "type": "marker", "trigger": {"mode": "sometimes"}}, "trigger"),
           ({"id": "x", "type": "marker", "trigger": {"mode": "at_s", "t": "soon"}}, "trigger"),
           ({"type": "marker"}, "id"),
           ("marker", "not an object"))
    for ev, want in bad:
        f = json.loads(json.dumps(base, allow_nan=True))
        f["events"] = list(f.get("events") or []) + [ev]
        errs = field_spec.validate(f)
        assert errs and any(want in e for e in errs), (ev, errs)
    f = json.loads(json.dumps(base))
    f["events"] = list(f["events"]) + [dict(f["events"][0])]
    assert any("duplicate event id" in e for e in field_spec.validate(f))
    for w in field_spec.ORCHARD_WORLDS:                                    # the shipped fields stay valid
        assert field_spec.validate(field_spec.load(w, validate_field=False)) == [], w


def test_a_bad_field_event_is_refused_not_a_hook_fault(world):
    """Backstop for a field loaded without validation: the event is refused, the hook keeps running."""
    m = world("orchard_lite")
    hook, reg, bus, d = _hooked(m)
    rt = hook.runtime_for(m)
    rt.scheduler = wev.EventScheduler([{"id": "typo", "type": "light", "trigger": {"mode": "at_s", "t": 0.05}}])
    _tick(m, d, reg, bus, 0.1)
    ev = [json.loads(p) for t, p in bus.outbox if t == hook.EVENTS_TOPIC]
    assert any(e["type"] == "event_refused" and e["id"] == "typo" for e in ev), ev
    assert reg.faulted() == {}
