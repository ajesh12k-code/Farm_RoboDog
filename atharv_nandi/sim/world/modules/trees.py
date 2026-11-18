"""WP23 compose module ``trees`` (order 50): one static, joint-free body per registry tree (ORCHARD-MVP-SPEC 3.3).

The geometry comes from ``world.trees`` plans (pure Python); this file only writes them into the ``MjSpec``:
wood capsules (group 0; keep wood contype 1 / conaffinity 0, prune candidates contype 0), merged foliage meshes
(group 3), leaf quads (group 3) with a ``<tree>_leaf_k_<k>`` site each, decals (group 4), static fruit, bags and
the banana bunch (group 1) and the query-only foliage envelope (group 5, contype 0). No joints, so nv does not
depend on the orchard size (design rule 13). Worlds without a registry (``legacy``) get nothing.

Also registers (at import, limited to models with hideable tree geoms) the ``hidden_geoms`` sim-state provider:
static fruit / bag / prune geoms that the fruit pool or
the cut rule hid (alpha 0, contype 0; their compiled alpha is 1 and contype 0, so un-hiding restores them).
"""
import hashlib
import json
import re
import weakref
from typing import Dict, List, Optional, Sequence

import mujoco
import numpy as np

from atharv_nandi.core import sim_state
from atharv_nandi.sim.world import compose, crops
from atharv_nandi.sim.world import trees as gen

VERSION = "1"
_GT = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
       "box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
       "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "mesh": mujoco.mjtGeom.mjGEOM_MESH}

_LAST: Dict[str, object] = {"world_id": None, "plans": []}
_RECENT: List[tuple] = []           # (world_id, plans) of the last composed orchard worlds, newest first
_RECENT_MAX = 8
_MODEL_PLANS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def plans_for(field: dict, ctx) -> List[dict]:
    """Plans of this compose run (cached on ``ctx.config`` so the fruit-pool module reuses them)."""
    if "tree_plans" not in ctx.config:
        ctx.config["tree_plans"] = gen.world_plans(field, ctx.seed)
    return ctx.config["tree_plans"]


def current_plans() -> List[dict]:
    """Plans of the world this process composed last (``[]`` before any orchard build). Prefer
    ``plans_for_model(m)``, which does not depend on what was composed after ``m``'s world."""
    return list(_LAST["plans"])


def plans_match(m, plans: Sequence[dict]) -> bool:
    """Do ``plans`` describe the trees compiled into ``m`` (every tree body at its position, every non-mesh
    tree geom by name and size, and no other tree body)?"""
    n_tree_bodies = sum(1 for b in range(m.nbody) if m.body_parentid[b] == 0
                        and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("tree_"))
    if n_tree_bodies != len(plans):
        return False
    for plan in plans:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, plan["body"])
        if b < 0 or not np.allclose(m.body_pos[b], plan["pos"], atol=1e-6):
            return False
        for g in plan["geoms"]:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g["name"])
            if gid < 0 or m.geom_bodyid[gid] != b:
                return False
            if g["type"] in ("ellipsoid", "sphere", "box") and "size" in g:
                n = len(g["size"])
                if not np.allclose(m.geom_size[gid][:n], g["size"], atol=1e-6):
                    return False
    return True


def bind_plans(m, plans: Sequence[dict]) -> None:
    """Record ``plans`` as the plans of model ``m`` (``plans_for_model`` returns them from then on)."""
    _MODEL_PLANS[m] = list(plans)


def plans_for_model(m) -> List[dict]:
    """The plans ``m`` was compiled from: bound with ``bind_plans`` or matched (bodies, positions, geom names and
    sizes) against the orchard worlds this process composed recently. ``[]`` for a model without trees or
    whose plans are not known here."""
    got = _MODEL_PLANS.get(m)
    if got is not None:
        return list(got)
    for _wid, plans in _RECENT:
        if plans and plans_match(m, plans):
            _MODEL_PLANS[m] = plans
            return list(plans)
    return []


def current_world_id() -> Optional[str]:
    return _LAST["world_id"]


def add_plan(spec, plan: dict):
    """Write one tree plan into ``spec`` as a static world child body."""
    body = spec.worldbody.add_body(name=plan["body"], pos=list(plan["pos"]))
    for mesh in plan["meshes"]:
        spec.add_mesh(name=mesh["name"], uservert=mesh["vert"], userface=mesh["face"],
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    for g in plan["geoms"]:
        kw = dict(name=g["name"], type=_GT[g["type"]], group=g["group"], contype=g["contype"],
                  conaffinity=g["conaffinity"], rgba=g["rgba"], mass=0.0)
        if g["type"] == "mesh":
            kw["meshname"] = g["mesh"]
        else:
            if "fromto" in g:
                kw["fromto"] = g["fromto"]
            if "pos" in g:
                kw["pos"] = g["pos"]
            if "quat" in g:
                kw["quat"] = g["quat"]
            kw["size"] = (list(g["size"]) + [0.0, 0.0, 0.0])[:3]
        body.add_geom(**kw)
    for s in plan["sites"]:
        body.add_site(name=s["name"], pos=s["pos"], size=[0.005, 0, 0], group=gen.GRP_QUERY, rgba=[1, 1, 1, 0])
    return body


def _attach(spec, field, ctx):
    plans = plans_for(field, ctx)
    for plan in plans:
        add_plan(spec, plan)
    _LAST["world_id"] = ctx.config.get("world_id")
    _LAST["plans"] = plans
    if plans:
        _RECENT[:] = [(ctx.config.get("world_id"), plans)] + [r for r in _RECENT if r[1] is not plans]
        del _RECENT[_RECENT_MAX:]


def _fingerprint(field, ctx) -> str:
    """Season + config digests + default-registry hash: a yaml edit changes ``world_id``."""
    entries = gen.registry(field) if not field.get("trees") else None
    if not field.get("trees") and not entries:
        return ""
    reg = "field" if field.get("trees") else hashlib.sha1(
        json.dumps(entries, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{gen.season_name(field)},t={gen.config_digest()[:8]},c={crops.config_digest()[:8]},r={reg}"


compose.register_module("trees", compose.ORDER["trees"], _attach, version=VERSION, fingerprint=_fingerprint)


# ---------------------------------------------------------------------- hidden geoms (sim-state provider)
_HIDEABLE = re.compile(r"^tree_.+_(fruit|bag|prune)_\d+$")
_STATE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _model_state(m) -> dict:
    st = _STATE.get(m)
    if st is None:
        ids = [g for g in range(m.ngeom)
               if _HIDEABLE.match(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        st = {"hideable": ids, "hideable_set": set(ids), "hidden": set()}
        _STATE[m] = st
    return st


def hideable_geoms(m) -> List[int]:
    return list(_model_state(m)["hideable"])


def hidden(m) -> List[int]:
    return sorted(_model_state(m)["hidden"])


def hide_geom(m, gid: int) -> None:
    """Hide a static fruit / bag / prune geom (alpha 0, contype 0, conaffinity 0). Used by the pool and WP2."""
    st = _model_state(m)
    if gid not in st["hideable_set"]:
        raise ValueError(f"geom {gid} is not a hideable tree geom")
    m.geom_rgba[gid, 3] = 0.0
    m.geom_contype[gid] = 0
    m.geom_conaffinity[gid] = 0
    st["hidden"].add(int(gid))


def unhide_geom(m, gid: int) -> None:
    st = _model_state(m)
    if gid not in st["hideable_set"]:
        raise ValueError(f"geom {gid} is not a hideable tree geom")
    m.geom_rgba[gid, 3] = 1.0      # compiled alpha of every hideable geom (world.trees invariant)
    m.geom_contype[gid] = 0        # hideable geoms are contype 0 in the compiled world
    m.geom_conaffinity[gid] = 0
    st["hidden"].discard(int(gid))


def _pack_hidden(m) -> dict:
    st = _model_state(m)
    if not st["hideable"]:
        return {}
    return {"geoms": sorted(st["hidden"])}


def _unpack_hidden(m, payload: dict) -> None:
    st = _model_state(m)
    if not payload:
        return
    want = payload.get("geoms", [])
    if not isinstance(want, list) or any(not isinstance(g, int) or isinstance(g, bool) for g in want):
        raise ValueError("hidden_geoms.geoms must be a list of geom ids")
    bad = [g for g in want if g not in st["hideable_set"]]
    if bad:
        raise ValueError(f"hidden_geoms: geoms {bad[:5]} are not hideable tree geoms")
    want = set(want)
    for g in st["hideable"]:
        if g in want:
            hide_geom(m, g)
        elif g in st["hidden"] or m.geom_rgba[g, 3] != 1.0:
            unhide_geom(m, g)


def _has_hideable(m) -> bool:
    return bool(_model_state(m)["hideable"])


# Registered at import (WP0 rule) and limited per model: a world without tree fruit / bags / prune geoms (legacy)
# packs no ``hidden_geoms`` key, whatever else this process composed (sim_state ``applies``).
sim_state.register_provider("hidden_geoms", _pack_hidden, _unpack_hidden, version=1, applies=_has_hideable)
