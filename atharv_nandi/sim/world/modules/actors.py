"""WP4 compose module ``actors`` (order 70): people, ladder / pole pickers and animals as mocap bodies.

Orchard worlds only. Each actor of the field (``world.actors.specs_from_field``, after the noVNC privacy
policy) becomes a mocap body ``actor_<class>_<id>`` at its home pose (or parked off-field) on the terrain, with
contype 0 / conaffinity 0 geoms in group 1 (``world.actors`` naming). A ladder picker with ``drop_every_s``
also gets a mocap fruit prop ``actor_prop_<id>``. The home keyframe carries the start poses
(``ctx.set_home_mocap``), so a reset puts every actor back.
"""
import hashlib
import json
import math

import mujoco

from atharv_nandi.sim.world import actors, compose, field_spec, terrain_orchard

ORDER = compose.ORDER["actors"]   # 70
VERSION = "1"
_TYPES = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
          "box": mujoco.mjtGeom.mjGEOM_BOX, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}


def _size3(g):
    s = list(g["size"])
    return s + [0.0] * (3 - len(s))


def _add_geoms(body, prefix: str, geoms):
    counts = {}
    for g in geoms:
        k = counts.get(g["role"], 0)
        counts[g["role"]] = k + 1
        kw = dict(name=f"{prefix}__{g['role']}_{k}", type=_TYPES[g["type"]], size=_size3(g), rgba=g["rgba"],
                  contype=0, conaffinity=0, group=actors.ACTOR_GROUP, mass=0.001)
        if "fromto" in g:
            kw["fromto"] = g["fromto"]
        else:
            kw["pos"] = g["pos"]
        body.add_geom(**kw)


def attach(spec, field, ctx):
    if not field_spec.is_orchard(field):
        return
    grid = terrain_orchard.grid_for(field, ctx.seed)
    for a in actors.specs_from_field(field):
        x, y = (actors.PARK_XY if a.parked else a.home[:2])
        z = grid.surface_at(x, y)
        quat = actors.yaw_quat(math.radians(a.home[2]))
        name = actors.body_name(a)
        body = spec.worldbody.add_body(name=name, pos=[x, y, z], quat=quat, mocap=True)
        _add_geoms(body, name, actors.geoms_for(a.type))
        ctx.set_home_mocap(name, [x, y, z], quat)
        if a.drop_every_s > 0:
            pname = actors.prop_body_name(a.id)
            prop = spec.worldbody.add_body(name=pname, pos=[actors.PARK_XY[0], actors.PARK_XY[1], 0.0], mocap=True)
            prop.add_geom(name=f"{pname}__fruit_0", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.045, 0.04, 0.055],
                          rgba=[0.85, 0.75, 0.15, 1.0], contype=0, conaffinity=0, group=actors.PROP_GROUP, mass=0.001)
            ctx.set_home_mocap(pname, [actors.PARK_XY[0], actors.PARK_XY[1], 0.0])


def fingerprint(field, ctx) -> str:
    if not field_spec.is_orchard(field):
        return ""
    on, persons, _ = actors.persons_policy()
    h = hashlib.sha1(json.dumps(field.get("actors"), sort_keys=True).encode()).hexdigest()[:10]
    return f"actors={int(on)},persons={int(persons)},a={h}"


compose.register_module("actors", ORDER, attach, version=VERSION, fingerprint=fingerprint)
