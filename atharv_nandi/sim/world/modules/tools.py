"""WP2 compose module ``tools`` (order 30): T2 CLIPPER, T4 TWIST, T5 PRUNER in their rack cradles, plus 4 debris
mocap bodies (ORCHARD-MVP-SPEC 2.4, 2.6, 3.4; MVP-SPEC 2.3, 3.1).

Each tool is a free body attached from ``atharv_nandi/sim/models/AtharvNandi/tools/<tool>.xml`` (6 + 2 dof; actuators
``<tool>_roll`` and ``<tool>_blade`` / ``twist_jaw`` appended, 25..30 in compose order). This module adds, per tool:

    rack_<tool>   weld trunk -> tool, ACTIVE at compile: relpose ``X_rack`` = the cradle point of the WP1 rack
                  (sites ``cradle_T2/T4/T5``, WP0c frozen) with the tool axis pointing down (tool-down docking) and
                  the yaw the arm reaches the cradle with at the WP0c frozen gantry value
    grip_<tool>   weld arm_wrist -> tool, INACTIVE at compile: relpose ``X_nom`` (handle_ref on tool_flange)
    <exclude>     tool bodies x (arm_wrist, finger_left, finger_right, arm_forearm): a mounted tool and the
                  fingers that hold it never fight the weld (0 tool-finger contacts by construction)

and writes the tool's home-keyframe qpos at the rack pose (``ctx.set_home``), so reset puts every tool back in
its cradle with zero weld error. Welds use ``solref 0.006 1`` (3 dt). Geoms compile parked (contype 16 /
conaffinity 0: collide with nothing); ``set_contact(m, tool_id, active)`` switches the contact geoms of the
mounted tool to contype 8 / conaffinity 1 (and the body-level broadphase masks) and back.

Debris: ``debris_0..3`` mocap bodies with one capsule each (group 0, contype 0: safety rays see them, nothing
collides), parked off the field and invisible (alpha 0). The prune cut (``controllers.tools.cut_logic.Debris``)
reshapes one to the cut shoot and drops it kinematically (OS 3.4 "Prune and tidy", 0 DOF).

Order 56 (``tools_catch``, after the WP23 pool): inactive welds ``catch_clipper_<k>`` (slots 0-9) and ``catch_twist_<k>``
(weld slots 5-9) from the tool head to the pool body, switched on by the hook after a cut or a twist release with the
measured relative pose (assisted ``weld_grasp``). ``seat_tools(m, d)`` puts every tool (and a caught fruit) on its
active weld pose: call it after teleporting the robot.

When the module composes (``tools_enabled``): ``ctx.config["tools"]`` if set (bool or on/off), else
``NANDI_TOOLS`` = ``on`` | ``off`` | ``auto`` (default ``auto`` = the orchard worlds). The legacy world gets
nothing, so its model, qpos and ``/nandi/sim_state`` stay byte-identical.

Sim-state providers (registered at import, limited to models with tools):

    tools   v1  {"tools": {T2|T4|T5: {"rack": eq_data[11], "grip": eq_data[11], "geoms": [[contype, conaffinity]],
                 "bodies": [[contype, conaffinity]]}}, "catch": {<tool>_<k>: eq_data[11]}}   the mutated model fields of
                the mount and of the assisted catch welds (MVP-SPEC 3.6)
    debris  v1  {"debris": [{"k", "size": [3], "rgba": [4], "source": str|null}]}   (poses are mocap, in qpos' place)
"""
import math
import os
import weakref
from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np

from atharv_nandi import paths
from atharv_nandi.core import sim_state
from atharv_nandi.core.tools import registry as reg
from atharv_nandi.sim.world import compose

VERSION = "1"
TOOLS_DIR = os.path.join(paths.ROBOT_MODEL_DIR, "tools")
ORCHARD_WORLDS = ("orchard_lite", "orchard_demo", "orchard_demo_monsoon")
N_DEBRIS = 4
DEBRIS_RADIUS = 0.006
DEBRIS_PARK = (-20.0, -22.0, 0.0)
DEBRIS_RGBA = (0.42, 0.33, 0.18, 1.0)
SHOULDER_BODIES = ("arm_carriage", "arm_turret", "arm_upper")


def debris_name(k: int) -> str:
    return f"debris_{k}"


def tool_mount(ctx) -> Optional[str]:
    """The tool of a rigid build, or None for the rack (default).

    ``ctx.config["tool_mount"]`` if set, else ``NANDI_TOOL_MOUNT`` = ``rack`` (default, also ``""``) |
    ``rigid:<T2|T4|T5>`` (MVP-SPEC 2.3 "Rigid builds", the pilot's manual swap, for baseline tests and RL): only
    that tool is built, held on the flange at X_nom by its grip weld from compile (``_attach_rigid``: design rule 1
    keeps it a free body), no rack welds, the other two tools are not built. A rigid tool is mounted for the whole
    run: change requests are refused ``rigid_mount``."""
    v = ctx.config.get("tool_mount")
    if v is None:
        v = os.environ.get("NANDI_TOOL_MOUNT", "")
    v = str(v).strip()
    if v.lower() in ("", "rack"):
        return None
    if v.lower().startswith("rigid:"):
        try:
            return reg.get(v.split(":", 1)[1].strip().upper()).tool_id
        except KeyError:
            pass
    raise ValueError(f"NANDI_TOOL_MOUNT={v!r}: use rack or rigid:<T2|T4|T5>")


def tools_enabled(field: dict, ctx) -> bool:
    v = ctx.config.get("tools")
    if v is None:
        v = os.environ.get("NANDI_TOOLS", "auto").strip().lower() or "auto"
    rigid = tool_mount(ctx)
    if isinstance(v, bool):
        on = v
    else:
        v = str(v).strip().lower()
        if v in ("1", "on", "true", "yes"):
            on = True
        elif v in ("0", "off", "false", "no"):
            on = False
        elif v == "auto":
            on = rigid is not None or str(field.get("world", ctx.world)) in ORCHARD_WORLDS
        else:
            raise ValueError(f"NANDI_TOOLS={v!r}: use on, off or auto")
    if rigid is not None and not on:
        raise ValueError(f"NANDI_TOOL_MOUNT=rigid:{rigid} with the tools switched off (NANDI_TOOLS)")
    return on


def _fingerprint(field: dict, ctx) -> str:
    if not tools_enabled(field, ctx):
        return "off"
    rigid = tool_mount(ctx)
    return f"rigid:{rigid}" if rigid else "on"


# ---------------------------------------------------------------------------------------------- rack geometry
def _quat_mul(a, b) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _quat_axis(axis, angle) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    return q


def rack_yaw(shoulder0: Sequence[float], cradle: Sequence[float], gantry: float) -> float:
    """Arm yaw that reaches the cradle tool-down: wrist pivot = cradle + (0, 0, flange) seen from the shoulder."""
    wrist = np.asarray(cradle, float) + np.array([0.0, 0.0, reg.FLANGE_X_M])
    v = wrist - (np.asarray(shoulder0, float) + np.array([gantry, 0.0, 0.0]))
    return math.atan2(v[1], v[0])


def rack_relpose(shoulder0, cradle, gantry) -> (np.ndarray, np.ndarray):
    """X_rack: (pos, quat) of the racked tool body in the trunk frame (handle_ref on the cradle point, tool axis
    down, yaw of the docking arm): the pose the tool has when the arm at its docking pose holds it at X_nom."""
    yaw = rack_yaw(shoulder0, cradle, gantry)
    q = _quat_mul(_quat_axis([0, 0, 1], yaw), _quat_axis([0, 1, 0], math.pi / 2.0))
    return np.asarray(cradle, float).copy(), q


def _spec_shoulder0(spec) -> np.ndarray:
    return sum(np.asarray(spec.body(n).pos, float) for n in SHOULDER_BODIES)


def _attach(spec, field, ctx):
    if not tools_enabled(field, ctx):
        return
    key = spec.keys[0] if spec.keys else None
    kq = np.asarray(key.qpos, float) if key is not None else np.zeros(0)
    trunk_pos = kq[0:3] if kq.size >= 7 else np.asarray(spec.body("trunk").pos, float)
    trunk_quat = kq[3:7] if kq.size >= 7 else np.asarray(spec.body("trunk").quat, float)
    shoulder0 = _spec_shoulder0(spec)
    rigid = tool_mount(ctx)
    for t in reg.all_tools():
        if rigid is not None:
            if t.tool_id == rigid:
                _attach_rigid(spec, t, ctx, kq)
            continue
        site = spec.site(t.cradle_site)
        if site is None:
            raise KeyError(f"{t.title}: rack cradle site {t.cradle_site} missing in the robot MJCF (WP1)")
        rel_pos, rel_quat = rack_relpose(shoulder0, site.pos, t.cradle_gantry)
        child = mujoco.MjSpec.from_file(os.path.join(TOOLS_DIR, t.xml))
        frame = spec.worldbody.add_frame()
        body = frame.attach_body(child.body(t.body), "", "")
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, trunk_quat)
        wpos = trunk_pos + R.reshape(3, 3) @ rel_pos
        wquat = _quat_mul(trunk_quat, rel_quat)
        body.pos, body.quat = wpos.tolist(), wquat.tolist()
        ctx.set_home(t.free_joint, list(wpos) + list(wquat))
        for b2 in t.bodies:
            for b1 in reg.EXCLUDE_ROBOT_BODIES:
                spec.add_exclude(bodyname1=b1, bodyname2=b2)
        e = spec.add_equality(name=t.rack_eq, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=reg.TRUNK_BODY, name2=t.body, active=True)
        e.data = [0.0, 0.0, 0.0, *rel_pos.tolist(), *rel_quat.tolist(), 1.0]
        e.solref = list(reg.WELD_SOLREF)
        g = spec.add_equality(name=t.grip_eq, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=reg.WRIST_BODY, name2=t.body, active=False)
        g.data = [0.0, 0.0, 0.0, *reg.X_NOM_POS, *reg.X_NOM_QUAT, 1.0]
        g.solref = list(reg.WELD_SOLREF)
    for k in range(N_DEBRIS):
        pos = [DEBRIS_PARK[0] + 0.5 * k, DEBRIS_PARK[1], DEBRIS_PARK[2]]
        b = spec.worldbody.add_body(name=debris_name(k), pos=pos, mocap=True)
        b.add_geom(name=f"{debris_name(k)}_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[DEBRIS_RADIUS, 0.03, 0],
                   contype=0, conaffinity=0, group=0, rgba=[*DEBRIS_RGBA[:3], 0.0], mass=0.001)
        ctx.set_home_mocap(debris_name(k), pos)


def _home_wrist_pose(spec, key_qpos) -> (np.ndarray, np.ndarray):
    """World (pos, quat) of ``arm_wrist`` at the home keyframe: compile a key-less copy of the spec so far."""
    s2 = spec.copy()
    for k in list(s2.keys):
        s2.delete(k)
    m2 = s2.compile()
    d2 = mujoco.MjData(m2)
    q = np.asarray(key_qpos, float).reshape(-1)
    n = min(q.size, m2.nq)
    d2.qpos[:n] = q[:n]
    mujoco.mj_kinematics(m2, d2)
    b = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, reg.WRIST_BODY)
    return np.array(d2.xpos[b]), np.array(d2.xquat[b])


def _attach_rigid(spec, t: reg.ToolSpec, ctx, key_qpos) -> None:
    """MVP-SPEC 2.3 rigid build, as far as design rule 1 allows: only this tool is built, a free body like in the
    rack build (its roll / blade joints come after the pool, so qpos[0:32] stays frozen; attached to ``arm_wrist``
    itself they would sit at qpos 14-15 and shift the legs, which ``model_layout.assert_frozen`` refuses), held at
    X_nom by its ``grip_<tool>`` weld ACTIVE from compile, no rack weld, contact geoms compiled mounted, the same
    arm ``<exclude>`` pairs, and a home keyframe on the flange (zero weld error at reset)."""
    wpos, wquat = _home_wrist_pose(spec, key_qpos)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, wquat)
    pos = wpos + R.reshape(3, 3) @ np.asarray(reg.X_NOM_POS, float)
    quat = _quat_mul(wquat, reg.X_NOM_QUAT)
    child = mujoco.MjSpec.from_file(os.path.join(TOOLS_DIR, t.xml))
    for n in t.collide_geoms:
        g = child.geom(n)
        g.contype, g.conaffinity = reg.ACTIVE_CONTYPE, reg.ACTIVE_CONAFFINITY
    frame = spec.worldbody.add_frame()
    body = frame.attach_body(child.body(t.body), "", "")
    body.pos, body.quat = pos.tolist(), quat.tolist()
    ctx.set_home(t.free_joint, list(pos) + list(quat))
    for b2 in t.bodies:
        for b1 in reg.EXCLUDE_ROBOT_BODIES:
            spec.add_exclude(bodyname1=b1, bodyname2=b2)
    g = spec.add_equality(name=t.grip_eq, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=reg.WRIST_BODY, name2=t.body, active=True)
    g.data = [0.0, 0.0, 0.0, *reg.X_NOM_POS, *reg.X_NOM_QUAT, 1.0]
    g.solref = list(reg.WELD_SOLREF)


def catch_eq_name(tool_name: str, slot: int) -> str:
    return f"catch_{tool_name}_{slot}"


CATCH_SLOTS = {"clipper": tuple(range(10)), "twist": tuple(range(5, 10))}   # T2 clips connect + weld slots;
#                                                                            T4 twists weld slots (guava, papaya)


def _attach_catch(spec, field, ctx):
    """Order 56 (after the WP23 pool): inactive welds head -> pool_<k> for the assisted catch after a cut or a
    twist release (assisted ``weld_grasp``: the stem clamp / jaws hold the fruit). 0 rows while inactive."""
    if not tools_enabled(field, ctx) or spec.body("pool_0") is None:
        return
    for tool_name, slots in CATCH_SLOTS.items():
        if spec.body(f"{tool_name}_head") is None:           # a rigid build has one tool only
            continue
        for k in slots:
            if spec.body(f"pool_{k}") is None:
                continue
            e = spec.add_equality(name=catch_eq_name(tool_name, k), type=mujoco.mjtEq.mjEQ_WELD,
                                  objtype=mujoco.mjtObj.mjOBJ_BODY, name1=f"{tool_name}_head", name2=f"pool_{k}",
                                  active=False)
            e.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            e.solref = list(reg.WELD_SOLREF)


compose.register_module("tools", compose.ORDER["tools"], _attach, version=VERSION, fingerprint=_fingerprint)
compose.register_module("tools_catch", compose.ORDER["fruit_pool"] + 1, _attach_catch, version=VERSION,
                        fingerprint=_fingerprint)


# ---------------------------------------------------------------------------------------------- runtime helpers
_HANDLES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def handles(m) -> Optional[reg.ToolHandles]:
    """``ToolHandles`` of a model (cached per model), ``None`` for a world without tools."""
    if m in _HANDLES:
        return _HANDLES[m]
    h = reg.ToolHandles.from_model(m)
    _HANDLES[m] = h
    return h


def debris_ids(m) -> List[dict]:
    out = []
    for k in range(N_DEBRIS):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, debris_name(k))
        if b < 0:
            return []
        out.append({"k": k, "body": b, "mocap": int(m.body_mocapid[b]), "geom": int(m.body_geomadr[b])})
    return out


def set_contact(m, tool_id: str, active: bool) -> None:
    """Mounted (active) or parked contact masks on the tool's contact geoms and on its bodies' broadphase masks
    (MuJoCo filters body pairs by the OR of their geoms first)."""
    h = handles(m)
    ids = h[tool_id]
    if ids.rigid:
        active = True                          # a rigid build is always mounted
    ct, ca =(reg.ACTIVE_CONTYPE, reg.ACTIVE_CONAFFINITY) if active else (reg.PARKED_CONTYPE, reg.PARKED_CONAFFINITY)
    for g in ids.collide:
        m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
    for b in ids.bodies:
        gs = [g for g in ids.geoms if m.geom_bodyid[g] == b]
        m.body_contype[b] = int(np.bitwise_or.reduce([int(m.geom_contype[g]) for g in gs])) if gs else 0
        m.body_conaffinity[b] = int(np.bitwise_or.reduce([int(m.geom_conaffinity[g]) for g in gs])) if gs else 0


def _weld_target(m, d, eq: int):
    """World (pos, quat) that body2 of weld ``eq`` must have to satisfy it exactly (anchor 0 convention)."""
    b1 = int(m.eq_obj1id[eq])
    e = np.asarray(m.eq_data[eq], float)
    R1 = np.asarray(d.xmat[b1]).reshape(3, 3)
    pos = np.asarray(d.xpos[b1]) + R1 @ e[3:6]
    return pos, _quat_mul(d.xquat[b1], e[6:10])


def seat_tools(m, d, tol_m: float = 0.0) -> List[str]:
    """Put every tool (and a caught pool fruit) exactly where its ACTIVE weld holds it, at rest.

    For code that teleports the robot (writes the root qpos) in a world with tools: without this the welds yank
    the tools across the jump. Runs ``mj_kinematics`` first; with ``tol_m`` > 0 only bodies further than that
    from their weld pose move. Returns the names of the bodies moved."""
    h = handles(m)
    if h is None:
        return []
    moved = []
    for _pass in range(2):                     # tools first, then fruit caught by a (now seated) tool head
        mujoco.mj_kinematics(m, d)
        eqs = [(i.rack_eq if not i.rigid and d.eq_active[i.rack_eq] else i.grip_eq, i.free_qadr, i.free_dadr, i.body)
               for i in h.tools.values()] if _pass == 0 else []
        if _pass == 1:
            for e in catch_ids(m).values():
                if d.eq_active[e]:
                    b = int(m.eq_obj2id[e])
                    j = int(m.body_jntadr[b])
                    eqs.append((e, int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), b))
        for eq, qa, da, body in eqs:
            if not d.eq_active[eq]:
                continue
            pos, quat = _weld_target(m, d, eq)
            if tol_m > 0 and float(np.linalg.norm(pos - d.xpos[body])) <= tol_m:
                continue
            d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = pos, quat
            d.qvel[da:da + 6] = 0.0
            moved.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body))
    mujoco.mj_kinematics(m, d)
    return moved


def is_active(m, tool_id: str) -> bool:
    ids = handles(m)[tool_id]
    return int(m.geom_contype[ids.collide[0]]) == reg.ACTIVE_CONTYPE


# ---------------------------------------------------------------------------------------------- providers
def _fl(a) -> List[float]:
    return [float(v) for v in np.asarray(a).reshape(-1)]


def catch_ids(m) -> Dict[tuple, int]:
    """(tool name, pool slot) -> id of the assisted catch weld (empty without tools or pool)."""
    out = {}
    for tool_name, slots in CATCH_SLOTS.items():
        for k in slots:
            e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, catch_eq_name(tool_name, k))
            if e >= 0:
                out[(tool_name, k)] = e
    return out


def _pack_tools(m) -> dict:
    h = handles(m)
    if h is None:
        return {}
    out = {}
    for k, ids in h.tools.items():
        out[k] = {"grip": _fl(m.eq_data[ids.grip_eq])} if ids.rigid else {"rack": _fl(m.eq_data[ids.rack_eq]),
                                                                           "grip": _fl(m.eq_data[ids.grip_eq])}
        out[k]["geoms"] = [[int(m.geom_contype[g]), int(m.geom_conaffinity[g])] for g in ids.collide]
        out[k]["bodies"] = [[int(m.body_contype[b]), int(m.body_conaffinity[b])] for b in ids.bodies]
    catch = catch_ids(m)
    return {"tools": out, "catch": {f"{n}_{k}": _fl(m.eq_data[e]) for (n, k), e in sorted(catch.items())}}


def _num_list(v, n, what):
    if not isinstance(v, list) or len(v) != n or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                       or not math.isfinite(x) for x in v):
        raise ValueError(f"tools {what}: needs {n} finite numbers")


def _int_pairs(v, n, what):
    if not isinstance(v, list) or len(v) != n:
        raise ValueError(f"tools {what}: needs {n} [contype, conaffinity] pairs")
    for p in v:
        if not isinstance(p, list) or len(p) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0
                                                         for x in p):
            raise ValueError(f"tools {what}: bad pair {p!r}")


def _unpack_tools(m, payload: dict) -> None:
    if not payload:
        return
    h = handles(m)
    if h is None:
        raise ValueError("tools mutations for a world without tools")
    tools = payload.get("tools")
    if not isinstance(tools, dict) or set(tools) != set(h.tools):
        raise ValueError(f"tools.tools must hold {sorted(h.tools)}")
    for k, ids in h.tools.items():
        t = tools[k]
        keys = {"grip", "geoms", "bodies"} if ids.rigid else {"rack", "grip", "geoms", "bodies"}
        if not isinstance(t, dict) or set(t) != keys:
            raise ValueError(f"tools.{k}: keys must be {', '.join(sorted(keys))}")
        if not ids.rigid:
            _num_list(t["rack"], 11, f"{k}.rack")
        _num_list(t["grip"], 11, f"{k}.grip")
        _int_pairs(t["geoms"], len(ids.collide), f"{k}.geoms")
        _int_pairs(t["bodies"], len(ids.bodies), f"{k}.bodies")
    catch = {f"{n}_{k}": e for (n, k), e in catch_ids(m).items()}
    got = payload.get("catch", {})
    if not isinstance(got, dict) or set(got) != set(catch):
        raise ValueError(f"tools.catch must hold {sorted(catch)}")
    for key in catch:
        _num_list(got[key], 11, f"catch {key}")
    for k, ids in h.tools.items():
        t = tools[k]
        if not ids.rigid:
            m.eq_data[ids.rack_eq] = t["rack"]
        m.eq_data[ids.grip_eq] = t["grip"]
        for g, (ct, ca) in zip(ids.collide, t["geoms"]):
            m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        for b, (ct, ca) in zip(ids.bodies, t["bodies"]):
            m.body_contype[b], m.body_conaffinity[b] = ct, ca
    for key, e in catch.items():
        m.eq_data[e] = got[key]


_DEBRIS_SOURCE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def debris_sources(m) -> Dict[int, Optional[str]]:
    """k -> name of the cut geom a debris body stands for (None = parked); per model."""
    if m not in _DEBRIS_SOURCE:
        _DEBRIS_SOURCE[m] = {k: None for k in range(N_DEBRIS)}
    return _DEBRIS_SOURCE[m]


def _pack_debris(m) -> dict:
    ids = debris_ids(m)
    if not ids:
        return {}
    src = debris_sources(m)
    return {"debris": [{"k": i["k"], "size": _fl(m.geom_size[i["geom"]]), "rgba": _fl(m.geom_rgba[i["geom"]]),
                        "source": src.get(i["k"])} for i in ids]}


def _unpack_debris(m, payload: dict) -> None:
    if not payload:
        return
    ids = debris_ids(m)
    if not ids:
        raise ValueError("debris mutations for a world without debris bodies")
    items = payload.get("debris")
    if not isinstance(items, list) or len(items) != len(ids):
        raise ValueError(f"debris.debris must list {len(ids)} entries")
    for it, i in zip(items, ids):
        if not isinstance(it, dict) or it.get("k") != i["k"]:
            raise ValueError("debris entries must be objects in k order")
        _num_list(it.get("size"), 3, f"debris {i['k']} size")
        _num_list(it.get("rgba"), 4, f"debris {i['k']} rgba")
        if it.get("source") is not None and not isinstance(it["source"], str):
            raise ValueError("debris source must be a string or null")
    src = debris_sources(m)
    for it, i in zip(items, ids):
        m.geom_size[i["geom"]] = it["size"]
        m.geom_rgba[i["geom"]] = it["rgba"]
        src[i["k"]] = it.get("source")


sim_state.register_provider("tools", _pack_tools, _unpack_tools, version=1, applies=lambda m: handles(m) is not None)
sim_state.register_provider("debris", _pack_debris, _unpack_debris, version=1, applies=lambda m: bool(debris_ids(m)))
