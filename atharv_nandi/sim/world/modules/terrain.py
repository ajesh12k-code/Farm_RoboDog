"""WP4 compose module ``terrain`` (order 40, after ``legacy_terrain``): orchard heightfield, ground patches, sun.

Orchard worlds only (field ``kind: orchard``); the legacy world is untouched (byte-identical model).

Adds, from ``world.terrain_orchard.TerrainGrid``:
    hfield ``orchard_terrain`` + geom ``orchard_ground`` (group 0, friction from the field, priority 1 so the
        ground friction wins over the foot's 0.8: the monsoon alleys' 0.4 must reach the contact)
    patch boxes / cylinders ``patch_<id>`` (group 0) filling their pits flush to z 0:
        mud     friction 0.3 (OS 3.6), priority 2, normal stiffness
        sink    soft contact tuned for 3-5 cm foot penetration (assumption, OS 3.6), friction 0.5
        flooded flooded-basin floor (monsoon): friction 0.3 + the sink softness, plus a water disc
                ``water_<tree>`` (visual, group 4)
    a text element ``nandi_world`` (``world=<w>;seed=<s>``) so hooks and tools identify the field from the
    compiled model alone; the sun light (``world.lighting``)
and removes what the legacy terrain module left (``ridge_*``, ``clod_*``); the robot MJCF ``floor`` plane stays
as a visual backdrop at z −0.002 with contype 0 (the hfield is the ground).
"""

import mujoco

from atharv_nandi.sim.world import compose, field_spec, lighting, terrain_orchard

ORDER = compose.ORDER["terrain"]           # 40; name "terrain" sorts after "legacy_terrain" in the band
VERSION = "1"
HFIELD_NAME = "orchard_terrain"
GROUND_GEOM = "orchard_ground"
GROUND_SOLREF = (0.01, 1.0)                # farm_scene GROUND_SOLREF: hard orchard ground
GROUND_RGBA = [0.36, 0.27, 0.18, 1.0]
FLOOR_Z = -0.002
WORLD_TEXT = "nandi_world"

# Sink softness (ASSUMED soil; tuned so a standing 6-foot robot sinks 3-5 cm, tests/test_terrain_orchard.py):
# solref timeconst / dampratio and solimp (dmin, dmax, width). MEASURED penetration is in the WP4 contract notes.
SINK_SOLREF = (0.10, 1.0)
SINK_SOLIMP = (0.9, 0.95, 0.05, 0.5, 2.0)
MUD_SOLREF = (0.02, 1.0)
G = mujoco.mjtGeom


def world_text(world: str, seed: int) -> str:
    return f"world={world};seed={int(seed)}"


def parse_world_text(model):
    """(world, seed) from a compiled model's ``nandi_world`` text, or None (legacy / foreign model)."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, WORLD_TEXT)
    if tid < 0:
        return None
    adr, n = int(model.text_adr[tid]), int(model.text_size[tid])
    raw = bytes(model.text_data[adr:adr + n]).split(b"\0", 1)[0].decode()
    kv = dict(p.split("=", 1) for p in raw.split(";") if "=" in p)
    return kv.get("world"), int(kv.get("seed", 7))


def _delete_legacy(spec):
    for g in list(spec.worldbody.geoms):
        n = g.name or ""
        if n.startswith(("ridge_", "clod_")):
            spec.delete(g)


def _patch_geom(spec, p: terrain_orchard.PatchSpec):
    top = 0.0
    depth = p.depth_m
    kw = dict(name=f"patch_{p.id}", group=0, friction=[p.friction, 0.02, 0.01], priority=2)
    if p.kind == "mud":
        kw.update(solref=list(MUD_SOLREF), rgba=[0.30, 0.22, 0.14, 1.0])
    elif p.kind in ("sink", "flooded"):
        kw.update(solref=list(SINK_SOLREF), solimp=list(SINK_SOLIMP),
                  rgba=[0.26, 0.20, 0.13, 1.0] if p.kind == "sink" else [0.22, 0.20, 0.17, 1.0])
    else:
        raise ValueError(f"patch {p.id}: kind {p.kind!r} not mud|sink|flooded")
    if p.r_m > 0:
        spec.worldbody.add_geom(type=G.mjGEOM_CYLINDER, pos=[p.center[0], p.center[1], top - depth / 2.0],
                                size=[p.r_m, depth / 2.0, 0], **kw)
    else:
        spec.worldbody.add_geom(type=G.mjGEOM_BOX, pos=[p.center[0], p.center[1], top - depth / 2.0],
                                size=[p.half[0], p.half[1], depth / 2.0], **kw)
    if p.kind == "flooded":
        spec.worldbody.add_geom(name=f"water_{p.id[len('flood_'):]}", type=G.mjGEOM_CYLINDER,
                                pos=[p.center[0], p.center[1], 0.07], size=[p.r_m + 0.1, 0.001, 0],
                                contype=0, conaffinity=0, group=4, rgba=[0.35, 0.42, 0.45, 0.55])


def attach(spec, field, ctx):
    if not field_spec.is_orchard(field):
        lighting.unregister_provider()
        return
    lighting.register_provider()
    if ctx.config.get("terrain") == "orchard":
        raise ValueError("NANDI_TERRAIN=orchard is the legacy tilted test orchard; it cannot be combined with "
                         f"NANDI_WORLD={field.get('world')} (unset NANDI_TERRAIN)")
    _delete_legacy(spec)
    floor = spec.geom("floor")
    if floor is not None:
        floor.contype, floor.conaffinity = 0, 0
        floor.pos = [0.0, 0.0, FLOOR_Z]
    grid = terrain_orchard.grid_for(field, ctx.seed)
    data, rng, base = grid.hfield_data()
    x0, x1, y0, y1 = grid.x0, grid.x1, grid.y0, grid.y1
    spec.add_hfield(name=HFIELD_NAME, size=[(x1 - x0) / 2.0, (y1 - y0) / 2.0, rng, 0.1],
                    nrow=grid.nrow, ncol=grid.ncol, userdata=data.reshape(-1).tolist())
    cfg = grid.cfg
    spec.worldbody.add_geom(name=GROUND_GEOM, type=G.mjGEOM_HFIELD, hfieldname=HFIELD_NAME,
                            pos=[(x0 + x1) / 2.0, (y0 + y1) / 2.0, base], group=0, priority=1,
                            friction=[cfg.friction, 0.02, 0.01], solref=list(GROUND_SOLREF), rgba=GROUND_RGBA)
    for p in cfg.patches:
        _patch_geom(spec, p)
    lighting.apply_spec(spec, (field.get("lighting") or {}).get("default", lighting.DEFAULT_PRESET))
    spec.add_text(name=WORLD_TEXT, data=world_text(field["world"], ctx.seed))


def fingerprint(field, ctx) -> str:
    if not field_spec.is_orchard(field):
        return ""
    import hashlib
    import json
    h = hashlib.sha1(json.dumps([field.get("terrain"), field.get("lighting")], sort_keys=True).encode()).hexdigest()
    return f"t={h[:10]}"


compose.register_module("terrain", ORDER, attach, version=VERSION, fingerprint=fingerprint)
