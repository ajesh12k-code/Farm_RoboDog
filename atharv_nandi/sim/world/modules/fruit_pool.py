"""WP23 compose module ``fruit_pool`` (order 55): 10 fixed-slot fruit proxy bodies (ORCHARD-MVP-SPEC 3.4).

Slots 0-4 carry a ``connect`` equality (mango: the anchor at the pedicel top is a ball joint, so the fruit swings),
slots 5-9 a ``weld`` (guava, papaya: anchor at the fruit CoM, torquescale 1, so gravity gives a force and no
moment and rows 3-5 of ``efc_force`` are torques). MuJoCo fixes equality types at compile time, so a slot never
changes kind. Each slot is a world-child free body ``pool_<k>`` (joint ``pool_<k>_free``) with:

    geom ``pool_<k>_fruit``    ellipsoid, group 1, contype 1 / conaffinity 0 while assigned (0 / 0 parked):
                               never collides with wood (1/0) or other fruit, does collide with the robot and
                               the ground (1/1)
    geom ``pool_<k>_pedicel``  visual capsule (group 1, contype 0, massless) from the stem end to the anchor
    sites ``pool_<k>_stem_end``, ``pool_<k>_cut_k`` (``pedicel_leave_cm`` above the stem end), ``pool_<k>_anchor``

Because the pedicel and ``cut_k`` live on the fruit body and the connect keeps the anchor coincident, both move
with a swinging fruit. Parked slots are invisible, contactless and held by their (active) equality at a park
spot off the field, so the pool's equality rows are constant (5 x 3 + 5 x 6 = 45). Free joints get rotational
damping ``POOL_DAMPING`` and armature ``POOL_ARMATURE`` (tuned by WP23, the WP23 contract notes); their home
qpos is written explicitly (the qpos0-fill gotcha: an appended free joint left at zero quaternion gives NaN).

Registers (at import, limited to models with pool bodies) the ``pool`` sim-state provider: per slot the
assignment (fruit_id, tree_id), ``eq_data`` and every
model field an assignment mutates, so a renderer that applies ``/nandi/sim_state`` shows the same fruit.
The runtime (assign / release / recycle) is ``controllers.fruit_pool.FruitPool``.
"""
import math
import weakref
from typing import Dict, List, Optional

import mujoco
import numpy as np

from atharv_nandi.core import sim_state
from atharv_nandi.sim.world import compose

VERSION = "1"
N_SLOTS = 10
N_CONNECT = 5                       # slots 0-4 connect, 5-9 weld (OS 3.4)
POOL_DAMPING = 0.04                 # N.m.s/rad on the free joint (OS 3.4 assumed 0.002; tuned over every demo mango,
                                    # 0.05 and 0.2 N.s bumps: worst settle 0.55 / 0.89 s; see contract)
POOL_ARMATURE = 1e-4
POOL_SOLREF = (0.006, 1.0)          # >= 2 dt at dt 0.002 / 0.003 (treesim probe value)
FRUIT_FRICTION = (1.0, 0.05, 0.02)
FRUIT_CONDIM = 6                    # like the legacy crops: a released fruit stops instead of rolling away
PARK_Z = 1.0
GRP_FRUIT = 1
# parked templates: mango-like on connect slots, guava-like on weld slots (overwritten on every assignment)
TEMPLATE = {"connect": {"size": (0.034, 0.031, 0.0425), "mass": 0.25, "pedicel": 0.06, "leave": 0.02},
            "weld": {"size": (0.034, 0.034, 0.035), "mass": 0.15, "pedicel": 0.03, "leave": 0.01}}


def slot_kind(k: int) -> str:
    return "connect" if k < N_CONNECT else "weld"


def park_pos(k: int) -> List[float]:
    return [-20.0 + 0.5 * k, -20.0, PARK_Z]


def names(k: int) -> Dict[str, str]:
    b = f"pool_{k}"
    return {"body": b, "joint": f"{b}_free", "fruit": f"{b}_fruit", "pedicel": f"{b}_pedicel", "eq": f"pool_eq_{k}",
            "cut_k": f"{b}_cut_k", "stem_end": f"{b}_stem_end", "anchor": f"{b}_anchor"}


def ellipsoid_inertia(mass: float, size) -> List[float]:
    a, b, c = size
    return [mass / 5.0 * (b * b + c * c), mass / 5.0 * (a * a + c * c), mass / 5.0 * (a * a + b * b)]


def _attach(spec, field, ctx):
    if not ctx.config.get("tree_plans"):
        return        # no orchard trees in this world: no pool (legacy world unchanged)
    for k in range(N_SLOTS):
        n, kind = names(k), slot_kind(k)
        t = TEMPLATE[kind]
        sx, sy, sz = t["size"]
        pos = park_pos(k)
        b = spec.worldbody.add_body(name=n["body"], pos=pos)
        j = b.add_freejoint(name=n["joint"])
        j.damping = POOL_DAMPING
        j.armature = POOL_ARMATURE
        b.add_geom(name=n["fruit"], type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[sx, sy, sz], mass=t["mass"],
                   rgba=[0.9, 0.7, 0.1, 0.0], group=GRP_FRUIT, contype=0, conaffinity=0,
                   condim=FRUIT_CONDIM, friction=list(FRUIT_FRICTION))
        stem = [0.0, 0.0, sz]
        top = [0.0, 0.0, sz + t["pedicel"]]
        b.add_geom(name=n["pedicel"], type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=stem + top, size=[0.0025, 0, 0],
                   mass=0.0, rgba=[0.35, 0.45, 0.15, 0.0], group=GRP_FRUIT, contype=0, conaffinity=0)
        b.add_site(name=n["stem_end"], pos=stem, size=[0.003, 0, 0], group=5, rgba=[1, 1, 1, 0])
        b.add_site(name=n["cut_k"], pos=[0.0, 0.0, sz + t["leave"]], size=[0.003, 0, 0], group=5, rgba=[1, 0, 0, 0])
        b.add_site(name=n["anchor"], pos=top, size=[0.003, 0, 0], group=5, rgba=[1, 1, 1, 0])
        if kind == "connect":
            e = spec.add_equality(name=n["eq"], type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                  name1=n["body"], active=True)
            e.data = top + [0.0] * 8               # anchor in body1 frame; the compiler fills the world anchor
        else:
            e = spec.add_equality(name=n["eq"], type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                  name1=n["body"], active=True)
            # anchor (body2 = world frame) at the CoM, relpose 0 / identity, torquescale 1
            e.data = list(pos) + [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        e.solref = list(POOL_SOLREF)
        ctx.set_home(n["joint"], list(pos) + [1.0, 0.0, 0.0, 0.0])


compose.register_module("fruit_pool", compose.ORDER["fruit_pool"], _attach, version=VERSION,
                        fingerprint=lambda field, ctx: f"d={POOL_DAMPING:g},a={POOL_ARMATURE:g},n={N_SLOTS}")


# ---------------------------------------------------------------------- slot ids + shared assignment state
class SlotIds:
    __slots__ = ("k", "kind", "body", "jnt", "qadr", "dadr", "fruit", "pedicel", "eq", "cut_k", "stem_end", "anchor")

    def __init__(self, m, k: int):
        n = names(k)
        oid = lambda t, s: mujoco.mj_name2id(m, t, s)  # noqa: E731
        self.k, self.kind = k, slot_kind(k)
        self.body = oid(mujoco.mjtObj.mjOBJ_BODY, n["body"])
        self.jnt = oid(mujoco.mjtObj.mjOBJ_JOINT, n["joint"])
        self.qadr = int(m.jnt_qposadr[self.jnt])
        self.dadr = int(m.jnt_dofadr[self.jnt])
        self.fruit = oid(mujoco.mjtObj.mjOBJ_GEOM, n["fruit"])
        self.pedicel = oid(mujoco.mjtObj.mjOBJ_GEOM, n["pedicel"])
        self.eq = oid(mujoco.mjtObj.mjOBJ_EQUALITY, n["eq"])
        self.cut_k = oid(mujoco.mjtObj.mjOBJ_SITE, n["cut_k"])
        self.stem_end = oid(mujoco.mjtObj.mjOBJ_SITE, n["stem_end"])
        self.anchor = oid(mujoco.mjtObj.mjOBJ_SITE, n["anchor"])


_STATE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def model_state(m) -> Optional[dict]:
    """{'slots': [SlotIds], 'assign': {k: {fruit_id, tree_id}}, 'harvested': set} or None without a pool."""
    st = _STATE.get(m)
    if st is None:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pool_0") < 0:
            return None
        slots = [SlotIds(m, k) for k in range(N_SLOTS)]
        st = {"slots": slots, "assign": {}, "harvested": set(),
              "park": {s.k: pack_slot(m, s) for s in slots}}   # compiled parked template, before any write
        _STATE[m] = st
    return st


def _fl(a) -> List[float]:
    return [float(v) for v in np.asarray(a).reshape(-1)]


def pack_slot(m, s: SlotIds) -> dict:
    return {"eq_data": _fl(m.eq_data[s.eq]),
            "fruit": {"size": _fl(m.geom_size[s.fruit]), "rgba": _fl(m.geom_rgba[s.fruit]),
                      "contype": int(m.geom_contype[s.fruit]), "conaffinity": int(m.geom_conaffinity[s.fruit]),
                      "rbound": float(m.geom_rbound[s.fruit]), "aabb": _fl(m.geom_aabb[s.fruit])},
            "pedicel": {"pos": _fl(m.geom_pos[s.pedicel]), "quat": _fl(m.geom_quat[s.pedicel]),
                        "size": _fl(m.geom_size[s.pedicel]), "rgba": _fl(m.geom_rgba[s.pedicel]),
                        "rbound": float(m.geom_rbound[s.pedicel]), "aabb": _fl(m.geom_aabb[s.pedicel])},
            "sites": {"cut_k": _fl(m.site_pos[s.cut_k]), "stem_end": _fl(m.site_pos[s.stem_end]),
                      "anchor": _fl(m.site_pos[s.anchor])},
            "body": {"contype": int(m.body_contype[s.body]), "conaffinity": int(m.body_conaffinity[s.body]),
                     "mass": float(m.body_mass[s.body]), "inertia": _fl(m.body_inertia[s.body]),
                     "subtreemass": float(m.body_subtreemass[s.body]), "invweight0": _fl(m.body_invweight0[s.body]),
                     "dof_invweight0": _fl(m.dof_invweight0[s.dadr:s.dadr + 6]),
                     "dof_M0": _fl(m.dof_M0[s.dadr:s.dadr + 6])}}


def sync_dof_m0(m, s: SlotIds) -> None:
    """``dof_M0`` of a slot from its current ``body_mass`` / ``body_inertia`` (+ armature).

    The pool bodies are MuJoCo "simple" bodies (``body_simple`` 1: CoM at the body origin, principal axes = body
    axes), whose diagonal mass matrix is taken from the compiled ``dof_M0``, not from ``body_mass``. Without this an
    assigned fruit moves like the parked template (0.25 kg connect, 0.15 kg weld): a released 1.54 kg papaya fell at
    100.8 m/s^2 (WP3 change request, measured)."""
    a = s.dadr
    m.dof_M0[a:a + 3] = float(m.body_mass[s.body]) + m.dof_armature[a:a + 3]
    m.dof_M0[a + 3:a + 6] = np.asarray(m.body_inertia[s.body]) + m.dof_armature[a + 3:a + 6]


def unpack_slot(m, s: SlotIds, f: dict) -> None:
    m.eq_data[s.eq] = f["eq_data"]
    g = f["fruit"]
    m.geom_size[s.fruit], m.geom_rgba[s.fruit] = g["size"], g["rgba"]
    m.geom_contype[s.fruit], m.geom_conaffinity[s.fruit] = g["contype"], g["conaffinity"]
    m.geom_rbound[s.fruit], m.geom_aabb[s.fruit] = g["rbound"], g["aabb"]
    p = f["pedicel"]
    m.geom_pos[s.pedicel], m.geom_quat[s.pedicel], m.geom_size[s.pedicel] = p["pos"], p["quat"], p["size"]
    m.geom_rgba[s.pedicel], m.geom_rbound[s.pedicel], m.geom_aabb[s.pedicel] = p["rgba"], p["rbound"], p["aabb"]
    for key, sid in (("cut_k", s.cut_k), ("stem_end", s.stem_end), ("anchor", s.anchor)):
        m.site_pos[sid] = f["sites"][key]
    b = f["body"]
    m.body_contype[s.body], m.body_conaffinity[s.body] = b["contype"], b["conaffinity"]   # broadphase filter
    m.body_mass[s.body], m.body_inertia[s.body] = b["mass"], b["inertia"]
    m.body_subtreemass[s.body], m.body_invweight0[s.body] = b["subtreemass"], b["invweight0"]
    m.dof_invweight0[s.dadr:s.dadr + 6] = b["dof_invweight0"]
    if "dof_M0" in b:
        m.dof_M0[s.dadr:s.dadr + 6] = b["dof_M0"]
    else:                     # a v1 payload from before dof_M0 was packed: derive it from mass and inertia
        sync_dof_m0(m, s)


def _pack_pool(m) -> dict:
    st = model_state(m)
    if st is None:
        return {}
    out = []
    for s in st["slots"]:
        a = st["assign"].get(s.k) or {}
        out.append({"slot": s.k, "kind": s.kind, "fruit_id": a.get("fruit_id"), "tree_id": a.get("tree_id"),
                    "eq_id": s.eq, "fields": pack_slot(m, s)})
    return {"slots": out, "harvested": sorted(st["harvested"])}


def _check_fields(m, s: SlotIds, f) -> None:
    """Shape check of one slot's ``fields`` against ``pack_slot`` (keys, lengths, finite numbers)."""
    ref = pack_slot(m, s)
    if isinstance(f, dict) and isinstance(f.get("body"), dict) and "dof_M0" not in f["body"]:
        ref["body"].pop("dof_M0")      # optional: v1 payloads packed before it was added (unpack derives it)
    elif isinstance(f, dict) and isinstance(f.get("body"), dict):
        m0 = f["body"]["dof_M0"]
        if isinstance(m0, list) and any(isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0
                                        for v in m0):
            raise ValueError(f"pool slot {s.k} fields.body.dof_M0: must be > 0")

    def walk(want, got, path):
        if isinstance(want, dict):
            if not isinstance(got, dict) or set(got) != set(want):
                raise ValueError(f"pool {path}: keys {sorted(got) if isinstance(got, dict) else type(got).__name__}"
                                 f" != {sorted(want)}")
            for k in want:
                walk(want[k], got[k], f"{path}.{k}")
        elif isinstance(want, list):
            if not isinstance(got, list) or len(got) != len(want):
                raise ValueError(f"pool {path}: needs a list of {len(want)} numbers")
            for v in got:
                walk(0.0, v, path)
        else:
            if isinstance(got, bool) or not isinstance(got, (int, float)) or not math.isfinite(got):
                raise ValueError(f"pool {path}: {got!r} is not a finite number")
            if isinstance(want, int) and not isinstance(want, bool) and got != int(got):
                raise ValueError(f"pool {path}: {got!r} is not an integer")
    walk(ref, f, f"slot {s.k} fields")


def _unpack_pool(m, payload: dict) -> None:
    """Validate the whole payload (slots, ids, field shapes, harvested ids) first, then write."""
    st = model_state(m)
    if not payload:
        return
    if st is None:
        raise ValueError("pool mutations for a world without a fruit pool")
    slots = payload.get("slots")
    if not isinstance(slots, list) or len(slots) != N_SLOTS:
        raise ValueError(f"pool.slots must list {N_SLOTS} slots")
    seen = set()
    for item in slots:
        if not isinstance(item, dict):
            raise ValueError("pool slot entries must be objects")
        k = item.get("slot")
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < N_SLOTS or k in seen \
                or item.get("eq_id") != st["slots"][k].eq:
            raise ValueError(f"pool slot {k!r}: bad, repeated slot or eq_id")
        seen.add(k)
        for key in ("fruit_id", "tree_id"):
            if item.get(key) is not None and not isinstance(item[key], str):
                raise ValueError(f"pool slot {k}: {key} must be a string or null")
        _check_fields(m, st["slots"][k], item.get("fields"))
    harvested = payload.get("harvested", [])
    if not isinstance(harvested, list) or any(not isinstance(h, str) or not h for h in harvested):
        raise ValueError("pool.harvested must be a list of fruit ids")
    for item in slots:
        k = item["slot"]
        unpack_slot(m, st["slots"][k], item["fields"])
        if item.get("fruit_id"):
            st["assign"][k] = {"fruit_id": item["fruit_id"], "tree_id": item.get("tree_id")}
        else:
            st["assign"].pop(k, None)
    st["harvested"] = set(harvested)


# Registered at import (WP0 rule) and limited per model: a world without pool bodies packs no ``pool`` key.
sim_state.register_provider("pool", _pack_pool, _unpack_pool, version=1,
                            applies=lambda m: model_state(m) is not None)
