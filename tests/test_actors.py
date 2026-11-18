"""WP4 actors (ORCHARD-MVP-SPEC 2.6, 5.2; MVP-SPEC 4.2, 5.4).

Acceptance (WP4): actors contype 0; refuse noVNC. Plus: 4 person types + ladder picker + pole
picker + animals (dog, monkey, cattle / nilgai), animals never classed as persons, scripted motion, triggers.
"""
import math
import os
import shutil
import subprocess

import mujoco
import numpy as np
import pytest

from atharv_nandi.core import farm_scene
from atharv_nandi.sim.hooks import Bus, HookRegistry
from atharv_nandi.sim.world import actors, field_spec


@pytest.fixture
def world(monkeypatch):
    def build(name, **env):
        monkeypatch.setenv("NANDI_WORLD", name)
        monkeypatch.setenv("NANDI_NOVNC", "0")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return farm_scene.load_farm_model()
    return build


def body_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


# ---------------------------------------------------------------------- composition
def test_every_actor_is_a_contype0_mocap_body_in_group1(world):
    m = world("orchard_demo")
    table = actors.geom_table(m)
    assert table
    for g, (cls, aid, role) in table.items():
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0, (aid, role)
        assert m.geom_group[g] == actors.ACTOR_GROUP
        assert m.body_mocapid[m.geom_bodyid[g]] >= 0
    bodies = actors.actor_bodies(m)
    f = field_spec.load("orchard_demo")
    assert len(bodies) == len(f["actors"]) + 1                       # + the ladder picker's fruit prop
    types = {a["type"] for a in f["actors"]}
    assert {"adult", "worker", "child", "crouching", "lying"} & types >= {"worker", "child", "crouching", "lying"}
    assert {"ladder_picker", "pole_picker", "dog", "monkey", "cattle", "nilgai"} <= types
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for a in f["actors"]:
        spec = actors.ActorSpec.from_dict(a)
        mid = bodies[actors.body_name(spec)]
        want = actors.PARK_XY if spec.parked else spec.home[:2]
        assert d.mocap_pos[mid][:2] == pytest.approx(want)        # the home keyframe carries the start poses


def test_persons_and_animals_are_classified_apart(world):
    m = world("orchard_demo")
    persons, animals = set(actors.person_geoms(m)), set(actors.animal_geoms(m))
    assert persons and animals and not persons & animals
    table = actors.geom_table(m)
    assert {table[g][1] for g in animals} == {"dog_1", "monkey_1", "cattle_1", "nilgai_1"}
    props = [g for g, (c, a, r) in table.items() if r in actors.PROP_ROLES]
    assert props and not set(props) & persons                        # ladders, poles, fruit are not person pixels
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    def zspan(aid, roles):
        gs = [g for g, (c, a, r) in table.items() if a == aid and r in roles]
        lo = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in gs)
        hi = max(d.geom_xpos[g][2] + m.geom_rbound[g] for g in gs)
        return lo, hi
    lo, hi = zspan("ladder_1", actors.PERSON_ROLES)                  # ladder picker: person at 0.8-2.5 m
    assert 0.75 <= lo < 1.0 and 2.45 <= hi <= 2.75                    # 0.8-2.5 m (OS 5.2 band top 2.5 m)
    lo, hi = zspan("pole_1", ("pole",))
    assert 3.0 <= hi <= 4.0                                           # 3-4 m pole picker
    lo, hi = zspan("child_1", actors.PERSON_ROLES)
    assert hi == pytest.approx(1.0, abs=0.08)
    lying_top = max(max(g["fromto"][2], g["fromto"][5]) + g["size"][0] if "fromto" in g else g["pos"][2] + g["size"][0]
                    for g in actors.geoms_for("lying"))
    assert lying_top <= 0.3                                           # person lying down: 0.25 m high
    assert actors.parse_geom_name("crate_floor") is None and actors.parse_geom_name("actor_x__y_0") is None


# ---------------------------------------------------------------------- privacy: refuse noVNC
def test_novnc_state_inference():
    assert actors.novnc_up({}) is True                                # start.sh starts noVNC unless field mode
    assert actors.novnc_up({"NANDI_FIELD_MODE": "1"}) is False
    assert actors.novnc_up({"NANDI_NOVNC": "0"}) is False
    assert actors.novnc_up({"NANDI_NOVNC": "1", "NANDI_FIELD_MODE": "1"}) is False   # field mode: start.sh skips it
    # review: values start.sh does not match must not switch persons on (start.sh starts noVNC for them)
    for v in ("TRUE", "On", " 1", "1 ", "Yes"):
        assert actors.novnc_up({"NANDI_FIELD_MODE": v}) is True, v
        assert actors.persons_policy({"NANDI_FIELD_MODE": v})[1] is False, v
    for v in ("OFF", " 0", "False", "nope"):
        assert actors.novnc_up({"NANDI_NOVNC": v}) is True, v
    # the decision start.sh exports wins; anything but exactly "0" is "up"
    assert actors.novnc_up({"NANDI_NOVNC_UP": "0", "NANDI_FIELD_MODE": "0"}) is False
    assert actors.novnc_up({"NANDI_NOVNC_UP": "1", "NANDI_NOVNC": "0"}) is True
    assert actors.novnc_up({"NANDI_NOVNC_UP": "", "NANDI_FIELD_MODE": "1"}) is True
    assert actors.persons_policy({"NANDI_NOVNC": "0"}) == (True, True, "ok")
    assert actors.persons_policy({"NANDI_ACTORS": "0"})[:2] == (False, False)
    on, persons, why = actors.persons_policy({})
    assert on and not persons and "novnc" in why
    with pytest.raises(actors.ActorPolicyError):
        actors.persons_policy({"NANDI_ACTORS": "1"})


START_SH = os.path.join(os.path.dirname(__file__), "..", "docker", "start.sh")


def _start_sh_decision(env: dict) -> str:
    """Run start.sh's own noVNC decision lines (the `case` patterns) in bash and return NANDI_NOVNC_UP."""
    lines = [ln for ln in open(START_SH).read().splitlines()
             if ln.startswith(('case "${NANDI_FIELD_MODE', 'case "${NANDI_NOVNC', 'if [ "$FIELD" = 1 ]; then NOVNC=0',
                               'export NANDI_NOVNC_UP'))]
    assert len(lines) == 4, lines
    script = "\n".join(lines + ['printf %s "$NANDI_NOVNC_UP"'])
    full = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **env}
    return subprocess.run(["bash", "-c", script], env=full, capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("env", [{}, {"NANDI_FIELD_MODE": "1"}, {"NANDI_FIELD_MODE": "true"}, {"NANDI_FIELD_MODE": "TRUE"},
                                 {"NANDI_FIELD_MODE": "On"}, {"NANDI_FIELD_MODE": " 1"}, {"NANDI_FIELD_MODE": "0"},
                                 {"NANDI_NOVNC": "0"}, {"NANDI_NOVNC": "off"}, {"NANDI_NOVNC": "OFF"},
                                 {"NANDI_NOVNC": "1"}, {"NANDI_NOVNC": "1", "NANDI_FIELD_MODE": "yes"},
                                 {"NANDI_NOVNC": "", "NANDI_FIELD_MODE": ""}])
def test_privacy_lockout_matches_the_launcher_exactly(env):
    """Review: actors.novnc_up parsed the env differently from docker/start.sh (case, whitespace, NANDI_NOVNC)."""
    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    decided = _start_sh_decision(env)
    assert decided in ("0", "1")
    assert actors.novnc_up(env) is (decided == "1"), (env, decided)
    assert actors.novnc_up({**env, "NANDI_NOVNC_UP": decided}) is (decided == "1")


def test_run_sh_forwards_the_world_and_privacy_knobs():
    run = open(os.path.join(os.path.dirname(__file__), "..", "docker", "run.sh")).read()
    for var in ("NANDI_WORLD", "NANDI_NOVNC", "NANDI_ACTORS", "NANDI_FIELD_MODE"):
        assert f"-e {var} " in run or f"-e {var}\n" in run, var


def test_person_actors_are_refused_while_novnc_is_up(world, monkeypatch):
    monkeypatch.delenv("NANDI_FIELD_MODE", raising=False)
    monkeypatch.setenv("NANDI_WORLD", "orchard_demo")
    monkeypatch.setenv("NANDI_NOVNC", "1")
    m = farm_scene.load_farm_model()                                  # auto: persons left out, animals stay
    assert not actors.person_geoms(m) and actors.animal_geoms(m)
    monkeypatch.setenv("NANDI_ACTORS", "1")                           # explicit request: refused
    with pytest.raises(actors.ActorPolicyError):
        farm_scene.load_farm_model()
    with pytest.raises(actors.ActorPolicyError):
        farm_scene.world_identity()
    monkeypatch.setenv("NANDI_ACTORS", "0")
    assert not actors.actor_bodies(farm_scene.load_farm_model())
    monkeypatch.setenv("NANDI_NOVNC", "0")
    monkeypatch.delenv("NANDI_ACTORS")
    assert actors.person_geoms(farm_scene.load_farm_model())


# ---------------------------------------------------------------------- scripted motion
def _rt(**kw):
    d = {"id": "a", "type": "adult", "home": [0, 0, 0], "script": {}, "start": {"mode": "cmd"}}
    d.update(kw)
    return actors.ActorRuntime(actors.ActorSpec.from_dict(d))


def test_script_walks_at_speed_with_holds_and_ends():
    rt = _rt(script={"waypoints": [[2, 0], [2, 2]], "speed_mps": 1.0, "holds_s": [1.0, 0.0], "on_end": "home"})
    assert rt.update(5.0) == (0.0, 0.0, 0.0) and rt.state == "idle"
    rt.start(10.0)
    assert rt.update(11.0)[:2] == pytest.approx((1.0, 0.0))
    x, y, yaw = rt.update(12.5)
    assert (x, y) == pytest.approx((2.0, 0.0)) and rt.state == "active"       # holding at waypoint 1
    x, y, yaw = rt.update(13.5)
    assert (x, y) == pytest.approx((2.0, 0.5)) and yaw == pytest.approx(math.pi / 2)
    rt.update(20.0)
    assert rt.state == "done" and rt.xy == (0.0, 0.0)                          # on_end home
    park = _rt(script={"waypoints": [[1, 0]], "on_end": "park"})
    park.start(0.0)
    park.update(5.0)
    assert park.state == "parked" and park.xy == actors.PARK_XY
    loop = _rt(script={"waypoints": [[1, 0], [0, 0]], "speed_mps": 1.0, "loop": True})
    loop.start(0.0)
    assert loop.update(2.5)[:2] == pytest.approx((0.5, 0.0)) and loop.state == "active"
    loop.stop()
    assert loop.state == "done"


def test_robot_frame_script_is_relative_to_the_robot_at_the_trigger():
    rt = _rt(home=[5, 5, 0], script={"frame": "robot", "waypoints": [[1.0, 2.0]], "speed_mps": 100.0})
    rt.start(0.0, robot_pose=(10.0, 0.0, math.pi / 2))
    rt.update(10.0)
    assert rt.xy == pytest.approx((8.0, 1.0))                                   # x forward = +y world
    parked = _rt(parked=True)
    assert parked.state == "parked" and parked.xy == actors.PARK_XY
    parked.start(0.0)
    assert parked.path[0] == (0.0, 0.0)                                         # unparks at home


def test_ladder_picker_person_is_in_the_os52_band():
    """OS 5.2 "person at 1-2.5 m height" (review: the person spanned 1.36-2.9 m)."""
    def zs(p):
        if "fromto" in p:
            f, r = p["fromto"], p["size"][0]
            return min(f[2], f[5]) - r, max(f[2], f[5]) + r
        return p["pos"][2] - p["size"][-1], p["pos"][2] + p["size"][-1]
    person = [zs(p) for p in actors.geoms_for("ladder_picker") if p["role"] in actors.PERSON_ROLES]
    head = [zs(p) for p in actors.geoms_for("ladder_picker") if p["role"] == "head"]
    lo, hi = min(z for z, _ in person), max(z for _, z in person)
    assert hi <= 2.5 + 1e-9 and lo >= actors.LADDER_FEET_M - 1e-9 and lo < 1.0
    assert all(1.0 <= a and b <= 2.5 + 1e-9 for a, b in head)


def test_ladder_picker_drops_fruit_periodically():
    rt = _rt(type="ladder_picker", drop_every_s=10.0, home=[0, 0, 0])
    x, y, z, falling = rt.fruit_drop(0.1)
    assert falling and 1.8 < z < actors.LADDER_HAND_Z_M <= 2.5
    assert rt.fruit_drop(5.0)[2:] == (0.04, False)                              # landed
    assert _rt().fruit_drop(1.0) is None
    with pytest.raises(KeyError):
        actors.actor_class("dragon")
    with pytest.raises(ValueError):
        actors.specs_from_field({"actors": [{"id": "a", "type": "dog"}, {"id": "a", "type": "dog"}]}, {"NANDI_NOVNC": "0"})


# ---------------------------------------------------------------------- hook-driven actors
def _hook(m):
    from atharv_nandi.sim.hooks import world_events as hook
    reg, bus = HookRegistry(), Bus()
    hook.register(reg)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    reg.reset(m, d)
    return hook, reg, bus, d


def _advance(m, d, reg, bus, secs, dt=0.02):
    n = int(round(secs / dt))
    for _ in range(n):
        d.time += dt
        reg.tick(m, d, bus)


def test_worker_enters_after_the_robot_reaches_ot1_and_dog_crosses_on_command(world):
    m = world("orchard_demo")
    hook, reg, bus, d = _hook(m)
    rt = hook.runtime_for(m)
    worker = rt.actors["worker_1"]
    mid = rt.mocap["worker_1"]
    home = d.mocap_pos[mid].copy()
    _advance(m, d, reg, bus, 1.0)
    assert worker.state == "idle"
    d.qpos[0:2] = (6.0 + 1.9, -3.5)                                   # robot on the OT-1 orbit
    _advance(m, d, reg, bus, 39.0)
    assert worker.state == "idle"                                     # after_s 40
    _advance(m, d, reg, bus, 3.0)
    assert worker.state == "active"
    _advance(m, d, reg, bus, 6.0)
    assert not np.allclose(d.mocap_pos[mid], home)
    assert d.mocap_pos[mid][2] == pytest.approx(rt.grid.surface_at(*d.mocap_pos[mid][:2]))
    # the ladder picker and the loops run from t = 0
    assert rt.actors["ladder_1"].state != "parked" and rt.actors["monkey_1"].state == "active"
    fruit = d.mocap_pos[rt.prop_mocap["ladder_1"]]
    assert math.dist(fruit[:2], d.mocap_pos[rt.mocap["ladder_1"]][:2]) < 1.0 and 0.0 < fruit[2] < 2.7
    # dog on command, relative to the robot
    bus.deliver(hook.CMD_TOPIC, '{"actor": "dog_1", "action": "start"}')
    _advance(m, d, reg, bus, 0.04)
    assert rt.actors["dog_1"].state == "active"
    state = bus.values["world"]
    assert {a["id"] for a in state["actors"]} == set(rt.actors)
    assert any('"actor:worker_1:start"' in p for t, p in bus.outbox if t == hook.EVENTS_TOPIC)
