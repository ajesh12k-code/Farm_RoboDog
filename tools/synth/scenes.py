"""Synthetic scenes (CAMERA-TRAINING C.2), built through the WP0 compose interface only.

Until WP4, WP23, WP26, WP37 and WP39 merge, the generator uses WP40's own **stub modules** (stylised
trees, fruit, decals, ground props, a person proxy, animals, a dock with tags, distractors). Each stub is a
``world.compose.Module`` with the standard ``attach_fn(spec, field, ctx)`` signature and ORDER band, and
the scene's ``world_id`` is ``compose.world_id`` over ``compose.module_versions``. The stubs are NOT put in
the global registry (``atharv_nandi/sim/world/modules`` stays untouched and the live world never sees them).

``source: compose`` selects already-registered compose modules by name instead (``compose.modules()``),
so a later dataset can use WP4/WP23/WP26/WP37/WP39 geometry with no edit to their files. Their geoms are
labelled from geom-name patterns (``label_patterns``) until those packages publish truth records.

Truth: every labelled geom belongs to exactly one instance ``(class, attrs)``; every geom has a Head C
semantic class. Geom groups: 0 wood/ground, 1 fruit, 3 foliage, 4 decals, thin label geoms and actors,
5 query-only (never rendered, never labelled). Group 2 is reserved for the solo (occlusion) pass.
"""
import hashlib
import math
import re
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import labels as L
from .dr import hsv_rgb, render_buffer

STUB_WORLD = "synth_stub"
EXTENT_M = 10.0          # stat.extent: fog and clip planes scale with it
ZNEAR, ZFAR = 0.002, 6.0  # fractions of the extent: 2 cm .. 60 m


# ------------------------------------------------------------------------------------------- plan + truth
@dataclass
class ScenePlan:
    seed: int
    split: str = "train"
    kind: str = "orchard"            # orchard | probe
    bank: str = "bank_a"
    species: Tuple[str, ...] = ("mango",)
    season: str = "fruiting"

    def to_dict(self) -> dict:
        return {"seed": int(self.seed), "split": self.split, "kind": self.kind, "bank": self.bank,
                "species": list(self.species), "season": self.season}


@dataclass
class Instance:
    iid: int
    cls: str
    attrs: dict
    geoms: List[str]
    anchor: Tuple[float, float, float]     # a point on the object (wrist views aim here)
    normal: Tuple[float, float, float] = (1.0, 0.0, 0.0)


class TruthRecorder:
    """Collects (geom name -> instance, semantic) while the stub modules attach geometry."""

    def __init__(self, registry: L.ClassRegistry):
        self.reg = registry
        self.instances: List[Instance] = []
        self.semantic: Dict[str, str] = {}      # geom name -> semantic class
        self.scene_tags: Dict[str, object] = {}
        self._n = 0

    def geom_name(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}/{self._n}"

    def instance(self, cls: str, attrs: dict, anchor, normal=(1.0, 0.0, 0.0)) -> Instance:
        if cls not in self.reg.classes or not self.reg.classes[cls].mask:
            raise L.ClassConfigError(f"{cls}: not a mask class, cannot own geoms")
        self.reg.check_attrs(cls, attrs)
        inst = Instance(len(self.instances), cls, dict(attrs), [], tuple(float(x) for x in anchor),
                        tuple(float(x) for x in normal))
        self.instances.append(inst)
        return inst


# ------------------------------------------------------------------------------------------- geometry


def _mj():
    import mujoco
    return mujoco


def quat_z_to(n: Sequence[float]) -> List[float]:
    """Quaternion (w, x, y, z) rotating +z onto ``n``."""
    n = np.asarray(n, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, n))
    if c > 1 - 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    if c < -1 + 1e-9:
        return [0.0, 1.0, 0.0, 0.0]
    axis = np.cross(z, n)
    axis /= np.linalg.norm(axis)
    a = math.acos(c) / 2.0
    return [math.cos(a)] + list(axis * math.sin(a))


def quat_axis(axis: Sequence[float], angle: float) -> List[float]:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return [math.cos(angle / 2)] + list(axis * math.sin(angle / 2))


def add_geom(spec, rec: TruthRecorder, *, kind: str, size, pos, semantic: str, rgba=(0.5, 0.5, 0.5, 1.0),
             group: int = 0, quat=None, material: Optional[str] = None, inst: Optional[Instance] = None,
             fromto=None, prefix: Optional[str] = None) -> str:
    mj = _mj()
    gtype = {"sphere": mj.mjtGeom.mjGEOM_SPHERE, "ellipsoid": mj.mjtGeom.mjGEOM_ELLIPSOID,
             "box": mj.mjtGeom.mjGEOM_BOX, "capsule": mj.mjtGeom.mjGEOM_CAPSULE,
             "cylinder": mj.mjtGeom.mjGEOM_CYLINDER, "plane": mj.mjtGeom.mjGEOM_PLANE}[kind]
    name = rec.geom_name(prefix or (inst.cls if inst is not None else f"sem_{semantic}"))
    kw = dict(name=name, type=gtype, size=[float(s) for s in size] + [0.0] * (3 - len(size)),
              rgba=[float(c) for c in rgba], group=int(group), contype=0, conaffinity=0)
    if fromto is not None:
        kw["fromto"] = [float(v) for v in fromto]
    else:
        kw["pos"] = [float(v) for v in pos]
        if quat is not None:
            kw["quat"] = [float(v) for v in quat]
    if material:
        kw["material"] = material
    spec.worldbody.add_geom(**kw)
    rec.semantic[name] = semantic
    if inst is not None:
        inst.geoms.append(name)
    return name


def _rgb_from_hue(rng, hue_rng, sat=(0.35, 0.75), val=(0.3, 0.7)) -> Tuple[float, float, float, float]:
    r, g, b = hsv_rgb(rng.uniform(*hue_rng), rng.uniform(*sat), rng.uniform(*val))
    return (r, g, b, 1.0)


# ------------------------------------------------------------------------------------------- stub modules
# attach_fn(spec, field, ctx): field["plan"] is the ScenePlan dict, ctx.config carries the recorder ("truth"),
# the synth config ("cfg") and runtime state shared between stubs ("trees").
def _plan(field) -> dict:
    return field["plan"]


def attach_environment(spec, field, ctx):
    """Order 40: ground plane, sky, sun, textures of the scene's DR bank, query-only proxy."""
    mj = _mj()
    rec: TruthRecorder = ctx.config["truth"]
    cfg = ctx.config["cfg"]
    plan = _plan(field)
    bank = cfg["banks"][plan["bank"]]
    rng = ctx.rng("environment")
    spec.stat.extent = EXTENT_M
    spec.stat.center = [0.0, 0.0, 0.0]
    spec.visual.map.znear = ZNEAR
    spec.visual.map.zfar = ZFAR
    ds = cfg["dataset"]
    # room for the padded render that barrel distortion needs (dr.render_buffer)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = render_buffer(int(ds["width"]), int(ds["height"]))
    spec.visual.quality.offsamples = int(ds.get("msaa_samples", 4))
    spec.visual.headlight.ambient = [0.1, 0.1, 0.1]
    spec.visual.headlight.diffuse = [0.15, 0.15, 0.15]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    builtin = {"checker": mj.mjtBuiltin.mjBUILTIN_CHECKER, "gradient": mj.mjtBuiltin.mjBUILTIN_GRADIENT,
               "flat": mj.mjtBuiltin.mjBUILTIN_FLAT}[bank.get("builtin", "checker")]
    sky1, sky2 = bank["sky"]
    spec.add_texture(name="sky", type=mj.mjtTexture.mjTEXTURE_SKYBOX, builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
                     width=64, height=64, rgb1=list(sky1), rgb2=list(sky2))
    for tex, hue_key, sat, val, rep in (("soil", "soil_hue", (0.3, 0.6), (0.25, 0.5), 30),
                                        ("bark", "bark_hue", (0.3, 0.6), (0.2, 0.45), 3),
                                        ("leaf", "leaf_hue", (0.4, 0.8), (0.2, 0.5), 4)):
        c1 = _rgb_from_hue(rng, bank[hue_key], sat, val)[:3]
        c2 = _rgb_from_hue(rng, bank[hue_key], sat, val)[:3]
        spec.add_texture(name=tex, type=mj.mjtTexture.mjTEXTURE_2D, builtin=builtin, width=64, height=64,
                         rgb1=list(c1), rgb2=list(c2), mark=mj.mjtMark.mjMARK_RANDOM, random=0.05,
                         markrgb=list(c2))
        mat = spec.add_material(name=tex, texrepeat=[rep, rep], texuniform=True)
        mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = tex
    spec.add_texture(name="tag", type=mj.mjtTexture.mjTEXTURE_2D, builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
                     width=64, height=64, rgb1=[0.05, 0.05, 0.05], rgb2=[0.95, 0.95, 0.95])
    mat = spec.add_material(name="tag", texrepeat=[2, 2])
    mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "tag"
    spec.worldbody.add_light(name="sun", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, dir=[0.3, 0.2, -1.0],
                             diffuse=[0.8, 0.8, 0.8], ambient=[0.3, 0.3, 0.3], specular=[0.1, 0.1, 0.1],
                             castshadow=bool(cfg["dataset"].get("shadows", False)), pos=[0.0, 0.0, 8.0])
    add_geom(spec, rec, kind="plane", size=[25.0, 25.0, 0.1], pos=[0, 0, 0], semantic="ground",
             material="soil", group=0, prefix="ground")
    # query-only proxy (e.g. a LiDAR foliage envelope): group 5, never rendered, never labelled
    add_geom(spec, rec, kind="box", size=[0.5, 0.5, 0.5], pos=[0, 0, 1.5], semantic="other",
             rgba=(1, 0, 1, 0.3), group=L.QUERY_ONLY_GROUP, prefix="query")


def _probe(plan) -> bool:
    return plan["kind"] == "probe"


def attach_trees(spec, field, ctx):
    """Order 50: stylised trees (trunk, limbs, canopy blobs) with trunk-base items, band, traps, props."""
    rec: TruthRecorder = ctx.config["truth"]
    cfg = ctx.config["cfg"]
    plan = _plan(field)
    sc = cfg["scene"]
    rng = ctx.rng("trees")
    probe = _probe(plan)
    n_trees = 1 if probe else int(rng.integers(sc["trees"][0], sc["trees"][1] + 1))
    trees = []
    for t in range(n_trees):
        sp = plan["species"][t % len(plan["species"])]
        tx = t * float(sc["tree_spacing_m"])
        trunk_r = rng.uniform(0.08, 0.14) if sp in ("mango", "guava") else rng.uniform(0.07, 0.10)
        trunk_h = rng.uniform(0.9, 1.4) if sp != "papaya" else rng.uniform(1.6, 2.0)
        add_geom(spec, rec, kind="cylinder", size=[trunk_r, trunk_h / 2], pos=[tx, 0, trunk_h / 2],
                 semantic="trunk", material="bark", group=0, prefix="trunk")
        crown_c = np.array([tx, 0.0, trunk_h + (0.55 if sp != "papaya" else 0.25)])
        nb = int(rng.integers(sc["canopy_blobs"][0], sc["canopy_blobs"][1] + 1))
        blobs = []
        crown_r = rng.uniform(0.7, 1.0) if sp in ("mango", "guava") else rng.uniform(0.45, 0.65)
        for b in range(nb):
            off = rng.normal(0, 1, 3) * np.array([0.35, 0.35, 0.18]) * crown_r
            radii = rng.uniform(0.35, 0.6, 3) * crown_r
            c = crown_c + off
            blobs.append((c, radii))
            add_geom(spec, rec, kind="ellipsoid", size=radii, pos=c, semantic="foliage", material="leaf",
                     group=3, prefix="canopy")
        for k in range(3):     # limbs
            a = rng.uniform(0, 2 * math.pi)
            end = crown_c + np.array([math.cos(a) * 0.4 * crown_r, math.sin(a) * 0.4 * crown_r, 0.1])
            add_geom(spec, rec, kind="capsule", size=[trunk_r * 0.45], pos=None, semantic="trunk",
                     material="bark", group=0, fromto=[tx, 0, trunk_h * 0.9, *end], prefix="limb")
        trees.append({"x": tx, "species": sp, "trunk_r": trunk_r, "trunk_h": trunk_h, "crown_c": crown_c,
                      "crown_r": crown_r, "blobs": blobs})
        _trunk_items(spec, rec, rng, trees[-1], probe)
    ctx.config["trees"] = trees


def _surface_point(rng, blobs) -> Tuple[np.ndarray, np.ndarray]:
    """A point just outside a random canopy blob's surface, and its outward normal."""
    c, radii = blobs[int(rng.integers(0, len(blobs)))]
    d = rng.normal(0, 1, 3)
    d[2] = abs(d[2]) * 0.4 - 0.15        # mostly sideways, a little up or down
    d /= np.linalg.norm(d)
    p = c + d * radii
    n = d / np.maximum(radii, 1e-6)
    n /= np.linalg.norm(n)
    return p, n


def _trunk_items(spec, rec, rng, tree, probe):
    tx, r, h = tree["x"], tree["trunk_r"], tree["trunk_h"]
    # trunk-base findings: thin geoms on the bark, a sprout capsule at the base
    kinds = ["lesion", "gum", "frass", "sprout"] if probe else \
        list(rng.choice(["lesion", "gum", "frass", "sprout"], size=int(rng.integers(0, 3)), replace=False))
    for kind in kinds:
        a = rng.uniform(0, 2 * math.pi)
        n = np.array([math.cos(a), math.sin(a), 0.0])
        z = rng.uniform(0.08, 0.4)
        inst = rec.instance("trunk_base", {"kind": kind}, [tx + n[0] * r, n[1] * r, z], n)
        if kind == "sprout":
            base = np.array([tx, 0, 0.03]) + n * (r + 0.01)
            tip = base + n * rng.uniform(0.12, 0.25) + np.array([0, 0, rng.uniform(0.1, 0.25)])
            add_geom(spec, rec, kind="capsule", size=[0.008], pos=None, fromto=[*base, *tip], semantic="trunk",
                     rgba=(0.25, 0.5, 0.15, 1), group=4, inst=inst)
        else:
            col = {"lesion": (0.15, 0.08, 0.05, 1), "gum": (0.75, 0.55, 0.2, 1), "frass": (0.6, 0.5, 0.35, 1)}[kind]
            s = rng.uniform(0.02, 0.05)
            p = np.array([tx, 0, z]) + n * (r + 0.002)
            add_geom(spec, rec, kind="box", size=[s, s * rng.uniform(0.6, 1.4), 0.002], pos=p, quat=quat_z_to(n),
                     semantic="trunk", rgba=col, group=4, inst=inst)
    # band (or a missing band: image-level tag only), ants on the band
    p_band = 1.0 if probe else 0.6
    if rng.random() < p_band:
        state = "ok" if probe or rng.random() < 0.75 else "torn"
        zb = rng.uniform(0.5, 0.75)
        inst = rec.instance("band", {"state": state}, [tx + r, 0, zb])
        hh = 0.03 if state == "ok" else 0.015
        add_geom(spec, rec, kind="cylinder", size=[r + 0.006, hh], pos=[tx, 0, zb], semantic="trunk",
                 rgba=(0.85, 0.85, 0.75, 1), group=4, inst=inst)
        if probe or rng.random() < 0.3:
            a = rng.uniform(0, 2 * math.pi)
            n = np.array([math.cos(a), math.sin(a), 0.0])
            ia = rec.instance("ants_on_band", {}, [tx + n[0] * r, n[1] * r, zb], n)
            add_geom(spec, rec, kind="box", size=[0.025, 0.015, 0.002], pos=np.array([tx, 0, zb]) + n * (r + 0.009),
                     quat=quat_z_to(n), semantic="trunk", rgba=(0.08, 0.05, 0.03, 1), group=4, inst=ia)
    else:
        rec.scene_tags["band_missing"] = True
    # ribbon colour: an attribute-only class (mask: false), the ribbon geom is semantic `other`
    if probe or rng.random() < 0.4:
        colour = str(rng.choice(["red", "yellow", "blue", "white"]))
        rec.scene_tags.setdefault("ribbon_colour", []).append(colour)
        rgb = {"red": (0.85, 0.1, 0.1, 1), "yellow": (0.9, 0.85, 0.1, 1), "blue": (0.1, 0.25, 0.85, 1),
               "white": (0.95, 0.95, 0.95, 1)}[colour]
        add_geom(spec, rec, kind="cylinder", size=[r + 0.004, 0.012], pos=[tx, 0, 1.0 if h > 1.05 else h * 0.9],
                 semantic="other", rgba=rgb, group=4, prefix="ribbon")
    if probe or rng.random() < 0.3:     # sleeve (tree guard) around the lower trunk
        inst = rec.instance("sleeve", {}, [tx + r, 0, 0.25])
        add_geom(spec, rec, kind="cylinder", size=[r + 0.03, 0.2], pos=[tx, 0, 0.21], semantic="trunk",
                 rgba=(0.7, 0.75, 0.7, 1), group=4, inst=inst)
    if probe or rng.random() < 0.4:     # prop pole under a limb
        a = rng.uniform(0, 2 * math.pi)
        foot = np.array([tx + math.cos(a) * 0.9, math.sin(a) * 0.9, 0.0])
        top = np.array([tx + math.cos(a) * 0.35, math.sin(a) * 0.35, h + 0.2])
        inst = rec.instance("prop", {}, (foot + top) / 2)
        add_geom(spec, rec, kind="capsule", size=[0.025], pos=None, fromto=[*foot, *top], semantic="trunk",
                 rgba=(0.55, 0.42, 0.25, 1), group=0, inst=inst)
    if probe or rng.random() < 0.5:     # mulch ring
        rr = rng.uniform(0.5, 0.9)
        inst = rec.instance("mulch", {}, [tx + rr * 0.8, 0, 0.01])
        add_geom(spec, rec, kind="cylinder", size=[rr, 0.006], pos=[tx, 0, 0.006], semantic="ground",
                 rgba=(0.45, 0.33, 0.18, 1), group=4, inst=inst)


def attach_fruit(spec, field, ctx):
    """Order 55: fruit per species and stage, bags, panicles, leaf decals, sooty sheen, traps."""
    rec: TruthRecorder = ctx.config["truth"]
    cfg = ctx.config["cfg"]
    plan = _plan(field)
    rng = ctx.rng("fruit")
    probe = _probe(plan)
    sc, sp_cfg, stage_col = cfg["scene"], cfg["species"], cfg["stage_colour"]
    season = plan["season"]
    dsz = ctx.config.get("decal_size_m", 0.02)
    for tree in ctx.config["trees"]:
        sp = tree["species"]
        spc = sp_cfg[sp]
        nf = int(rng.integers(sc["fruit_per_tree"][0], sc["fruit_per_tree"][1] + 1))
        if season == "flowering" and not probe:
            nf = nf // 3
        for k in range(nf):
            stage = str(rng.choice(spc["stages"]))
            col = stage_col[stage]
            rgba = _rgb_from_hue(rng, col["hue"], col["sat"], col["val"])
            p, n = _surface_point(rng, tree["blobs"])
            if sp == "papaya":     # papaya fruit sits on the stem under the crown
                a = rng.uniform(0, 2 * math.pi)
                n = np.array([math.cos(a), math.sin(a), 0.0])
                p = np.array([tree["x"], 0, tree["trunk_h"] - rng.uniform(0.05, 0.4)]) + n * (tree["trunk_r"] + 0.06)
            inst = rec.instance("fruit", {"species": sp, "stage": stage}, p, n)
            r = rng.uniform(*spc["size_m"])
            asp = rng.uniform(*spc["aspect"])
            if spc["shape"] == "bunch":
                for j in range(int(rng.integers(6, 11))):
                    a = 2 * math.pi * j / 8 + rng.uniform(-0.2, 0.2)
                    c = p + np.array([math.cos(a) * 0.05, math.sin(a) * 0.05, -0.02 * (j // 8)])
                    tip = c + np.array([math.cos(a) * 0.06, math.sin(a) * 0.06, r * asp * 0.6])
                    add_geom(spec, rec, kind="capsule", size=[r], pos=None, fromto=[*c, *tip], semantic="fruit",
                             rgba=rgba, group=1, inst=inst)
            elif spc["shape"] == "sphere":
                add_geom(spec, rec, kind="sphere", size=[r], pos=p, semantic="fruit", rgba=rgba, group=1, inst=inst)
            else:
                tilt = quat_axis(rng.normal(0, 1, 3), rng.uniform(0, 0.4))
                add_geom(spec, rec, kind="ellipsoid", size=[r, r * rng.uniform(0.85, 1.0), r * asp], pos=p,
                         quat=tilt, semantic="fruit", rgba=rgba, group=1, inst=inst)
            if sp == "mango" and (probe and k == 0 or rng.random() < 0.15):     # fruit bag over this mango
                state = "torn" if rng.random() < 0.3 else "ok"
                bag = rec.instance("bag", {"state": state}, p + n * 0.02, n)
                add_geom(spec, rec, kind="box", size=[r * 1.4, r * 1.4, r * asp * 1.3], pos=p + n * 0.012,
                         quat=quat_z_to(n) if state == "torn" else None, semantic="other",
                         rgba=(0.9, 0.88, 0.8, 1) if state == "ok" else (0.8, 0.75, 0.6, 1), group=4, inst=bag)
        # panicles (flowering, or a few late ones)
        npan = (int(rng.integers(2, 6)) if season == "flowering" else int(rng.integers(0, 2))) if sp == "mango" else 0
        if probe:
            npan = max(npan, 1)
        for _ in range(npan):
            p, n = _surface_point(rng, tree["blobs"])
            cond = str(rng.choice(["normal", "malformed", "mildew"]))
            inst = rec.instance("panicle", {"condition": cond}, p, n)
            tip = p + n * 0.05 + np.array([0, 0, rng.uniform(0.12, 0.25)])
            rgba = {"normal": (0.9, 0.8, 0.55, 1), "malformed": (0.55, 0.5, 0.3, 1), "mildew": (0.92, 0.92, 0.9, 1)}[cond]
            add_geom(spec, rec, kind="capsule", size=[rng.uniform(0.02, 0.035)], pos=None, fromto=[*p, *tip],
                     semantic="foliage", rgba=rgba, group=4, inst=inst)
        # leaf symptom decals and sooty sheen (thin geoms on the canopy surface)
        nd = int(rng.integers(sc["decals_per_tree"][0], sc["decals_per_tree"][1] + 1))
        symptoms = cfg_classes_values(rec, "leaf_symptom", "symptom")
        for _ in range(nd):
            p, n = _surface_point(rng, tree["blobs"])
            sym = str(rng.choice(symptoms))
            inst = rec.instance("leaf_symptom", {"symptom": sym}, p, n)
            s = dsz * rng.uniform(0.7, 1.3)
            add_geom(spec, rec, kind="box", size=[s, s * rng.uniform(0.6, 1.0), 0.002], pos=p + n * 0.004,
                     quat=quat_z_to(n), semantic="foliage", rgba=_decal_rgba(rng, sym, ctx), group=4, inst=inst)
        if probe or rng.random() < 0.3:
            p, n = _surface_point(rng, tree["blobs"])
            inst = rec.instance("sooty_sheen", {}, p, n)
            add_geom(spec, rec, kind="ellipsoid", size=[0.12, 0.09, 0.003], pos=p + n * 0.004, quat=quat_z_to(n),
                     semantic="foliage", rgba=(0.12, 0.12, 0.1, 1), group=4, inst=inst)
        # traps hang from the canopy edge
        if probe or rng.random() < 0.5:
            kind = "me" if rng.random() < 0.5 else "sticky"
            p, n = _surface_point(rng, tree["blobs"])
            p = p + n * 0.08 - np.array([0, 0, 0.15])
            inst = rec.instance("trap", {"kind": kind, "catch_bin": str(rng.choice(["none", "low", "mid", "high"])),
                                         "present": True}, p, n)
            if kind == "me":
                add_geom(spec, rec, kind="cylinder", size=[0.05, 0.09], pos=p, semantic="other",
                         rgba=(0.85, 0.9, 0.9, 1), group=4, inst=inst)
            else:
                add_geom(spec, rec, kind="box", size=[0.1, 0.002, 0.07], pos=p, quat=quat_z_to(n), semantic="other",
                         rgba=(0.95, 0.9, 0.1, 1), group=4, inst=inst)
        if probe and tree is ctx.config["trees"][0]:
            # probe scenes add one fruit per species on the ground so every species is present
            pass


def cfg_classes_values(rec: TruthRecorder, cls: str, attr: str) -> List[str]:
    return [str(v) for v in rec.reg.classes[cls].attrs[attr]]


def _decal_rgba(rng, sym: str, ctx) -> Tuple[float, float, float, float]:
    base = {"sooty_mould": (0.1, 0.1, 0.1), "mildew": (0.9, 0.9, 0.88), "anthracnose": (0.25, 0.15, 0.08),
            "mealybug": (0.92, 0.92, 0.92), "sigatoka": (0.35, 0.25, 0.1)}.get(sym, None)
    if base is None:
        base = hsv_rgb(rng.uniform(20, 60), rng.uniform(0.4, 0.8), rng.uniform(0.3, 0.8))
    op = float(ctx.config.get("decal_opacity", 1.0))
    leaf = np.array([0.2, 0.45, 0.15])
    mixed = op * np.asarray(base) + (1 - op) * leaf          # opacity as colour mix: geoms stay opaque in seg
    return (float(mixed[0]), float(mixed[1]), float(mixed[2]), 1.0)


def attach_ground_items(spec, field, ctx):
    """Order 45 band (terrain/fixtures): weeds, wet patches, puddles, broken limbs, fallen fruit, laterals."""
    rec: TruthRecorder = ctx.config["truth"]
    plan = _plan(field)
    cfg = ctx.config["cfg"]
    rng = ctx.rng("ground_items")
    probe = _probe(plan)
    trees = ctx.config.get("trees") or [{"x": 0.0, "species": plan["species"][0]}]
    x_lo, x_hi = -2.5, trees[-1]["x"] + 2.5
    for side in (-1, 1):       # drip laterals along the row
        if probe or rng.random() < 0.6:
            y = side * rng.uniform(0.5, 0.8)
            inst = rec.instance("drip_lateral", {}, [(x_lo + x_hi) / 2, y, 0.01])
            add_geom(spec, rec, kind="capsule", size=[0.008], pos=None, fromto=[x_lo, y, 0.009, x_hi, y, 0.009],
                     semantic="other", rgba=(0.05, 0.05, 0.05, 1), group=4, inst=inst)

    def ground_xy(near=1.6):
        t = trees[int(rng.integers(0, len(trees)))]
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.3, near)
        return t, np.array([t["x"] + math.cos(a) * d, math.sin(a) * d])

    for _ in range(1 if probe else int(rng.integers(0, 5))):
        _, xy = ground_xy(2.5)
        inst = rec.instance("weed", {}, [*xy, 0.05])
        for j in range(3):
            o = rng.normal(0, 0.04, 2)
            add_geom(spec, rec, kind="ellipsoid", size=[0.04, 0.03, rng.uniform(0.05, 0.12)], pos=[*(xy + o), 0.04],
                     semantic="foliage", rgba=_rgb_from_hue(rng, (75, 120), (0.5, 0.8), (0.3, 0.6)), group=3, inst=inst)
    if probe or rng.random() < 0.4:
        _, xy = ground_xy()
        inst = rec.instance("wet_patch", {}, [*xy, 0.003])
        add_geom(spec, rec, kind="ellipsoid", size=[rng.uniform(0.2, 0.5), rng.uniform(0.15, 0.4), 0.003],
                 pos=[*xy, 0.0], semantic="ground", rgba=(0.2, 0.15, 0.1, 1), group=4, inst=inst)
    if probe or rng.random() < (0.6 if plan["season"] == "monsoon" else 0.2):
        _, xy = ground_xy(3.0)
        inst = rec.instance("puddle", {}, [*xy, 0.004])
        add_geom(spec, rec, kind="cylinder", size=[rng.uniform(0.25, 0.6), 0.003], pos=[*xy, 0.002],
                 semantic="water", rgba=(0.35, 0.42, 0.5, 1), group=4, inst=inst)
    if probe or rng.random() < 0.3:
        _, xy = ground_xy(2.0)
        a = rng.uniform(0, math.pi)
        d = np.array([math.cos(a), math.sin(a)]) * rng.uniform(0.3, 0.6)
        inst = rec.instance("broken_limb", {}, [*xy, 0.04])
        add_geom(spec, rec, kind="capsule", size=[rng.uniform(0.02, 0.04)], pos=None,
                 fromto=[*(xy - d), 0.035, *(xy + d), 0.035], semantic="trunk", rgba=(0.4, 0.3, 0.2, 1), group=0,
                 inst=inst)
    nff = len(cfg["species"]) if probe else int(rng.integers(0, 4))
    for k in range(nff):
        t, xy = ground_xy(1.5)
        sp = list(plan["species"])[k % len(plan["species"])] if not probe else t["species"]
        inst = rec.instance("fallen_fruit", {"species": sp}, [*xy, 0.03])
        col = cfg["stage_colour"]["overripe"]
        add_geom(spec, rec, kind="ellipsoid", size=[0.04, 0.035, 0.03], pos=[*xy, 0.03], semantic="fruit",
                 rgba=_rgb_from_hue(rng, col["hue"], col["sat"], col["val"]), group=1, inst=inst)


def attach_actors(spec, field, ctx):
    """Order 70: person proxies (whole body, no face geometry), animals, crates, tools, wire, distractors."""
    rec: TruthRecorder = ctx.config["truth"]
    plan = _plan(field)
    sc = ctx.config["cfg"]["scene"]
    rng = ctx.rng("actors")
    probe = _probe(plan)
    trees = ctx.config.get("trees") or [{"x": 0.0}]
    x_mid = (trees[0]["x"] + trees[-1]["x"]) / 2

    def spot(d_lo=2.0, d_hi=5.0):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(d_lo, d_hi)
        return np.array([x_mid + math.cos(a) * d, math.sin(a) * d])

    for _ in range(1 if probe else (int(rng.integers(1, 3)) if rng.random() < sc["p_person"] else 0)):
        add_person(spec, rec, rng, spot(2.0, 4.5))
    if probe or rng.random() < sc["p_animal"]:
        kinds = ["dog", "cattle", "nilgai", "monkey"]
        for kind in (kinds if probe else [str(rng.choice(kinds))]):
            add_animal(spec, rec, rng, spot(2.5, 6.0), kind)
    if probe or rng.random() < 0.5:
        xy = spot(1.5, 3.5)
        inst = rec.instance("crate", {}, [*xy, 0.15])
        add_geom(spec, rec, kind="box", size=[0.25, 0.18, 0.15], pos=[*xy, 0.15],
                 quat=quat_axis([0, 0, 1], rng.uniform(0, math.pi)), semantic="other",
                 rgba=_rgb_from_hue(rng, (0, 360), (0.5, 0.9), (0.4, 0.8)), group=4, inst=inst)
    if probe or rng.random() < 0.4:
        xy = spot(1.0, 3.0)
        inst = rec.instance("tool", {}, [*xy, 0.02])
        a = rng.uniform(0, math.pi)
        d = np.array([math.cos(a), math.sin(a)]) * 0.15
        add_geom(spec, rec, kind="capsule", size=[0.015], pos=None, fromto=[*(xy - d), 0.016, *(xy + d), 0.016],
                 semantic="other", rgba=(0.8, 0.15, 0.1, 1), group=4, inst=inst)
    if probe or rng.random() < sc["p_fallen_wire"]:
        xy = spot(1.0, 3.0)
        a = rng.uniform(0, math.pi)
        d = np.array([math.cos(a), math.sin(a)]) * rng.uniform(1.0, 2.0)
        inst = rec.instance("fallen_wire", {}, [*xy, 0.01])
        add_geom(spec, rec, kind="capsule", size=[0.006], pos=None, fromto=[*(xy - d), 0.007, *(xy + d), 0.007],
                 semantic="other", rgba=(0.3, 0.3, 0.32, 1), group=4, inst=inst)
    if probe or rng.random() < sc["p_distractors"]:
        # unlabelled distractors: a rope between posts, a net panel, birds
        p0, p1 = spot(1.5, 3.0), spot(1.5, 3.0)
        add_geom(spec, rec, kind="capsule", size=[0.006], pos=None, fromto=[*p0, 1.2, *p1, 1.1], semantic="other",
                 rgba=(0.75, 0.65, 0.4, 1), group=4, prefix="rope")
        c = spot(2.5, 4.0)
        for j in range(5):
            add_geom(spec, rec, kind="capsule", size=[0.003], pos=None,
                     fromto=[c[0], c[1] - 0.5 + 0.25 * j, 0.0, c[0], c[1] - 0.5 + 0.25 * j, 1.2], semantic="other",
                     rgba=(0.2, 0.5, 0.25, 1), group=4, prefix="net")
        for j in range(int(rng.integers(0, 3))):
            b = spot(1.0, 4.0)
            add_geom(spec, rec, kind="ellipsoid", size=[0.08, 0.04, 0.035], pos=[*b, rng.uniform(1.8, 3.5)],
                     semantic="other", rgba=(0.15, 0.15, 0.18, 1), group=4, prefix="bird")


def add_person(spec, rec: TruthRecorder, rng, xy) -> Instance:
    """Whole-body person proxy: capsules and one plain head ellipsoid. No face geometry, no identity attrs."""
    s = rng.uniform(0.85, 1.08)                 # size
    shirt = _rgb_from_hue(rng, (0, 360), (0.2, 0.9), (0.2, 0.9))
    trousers = _rgb_from_hue(rng, (0, 360), (0.1, 0.7), (0.1, 0.6))
    skin = (0.55, 0.42, 0.32, 1.0)
    yaw = rng.uniform(0, 2 * math.pi)
    side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    base = np.array([xy[0], xy[1], 0.0])
    inst = rec.instance("person", {}, base + np.array([0, 0, 1.0 * s]))
    for sgn in (-1, 1):
        hip = base + side * 0.09 * sgn * s + np.array([0, 0, 0.9 * s])
        add_geom(spec, rec, kind="capsule", size=[0.06 * s], pos=None, fromto=[*(base + side * 0.1 * sgn * s + [0, 0, 0.06]), *hip],
                 semantic="other", rgba=trousers, group=4, inst=inst)
        sh = base + side * 0.2 * sgn * s + np.array([0, 0, 1.38 * s])
        hand = sh + side * 0.05 * sgn + np.array([0, 0, -0.58 * s])
        add_geom(spec, rec, kind="capsule", size=[0.04 * s], pos=None, fromto=[*sh, *hand], semantic="other",
                 rgba=shirt, group=4, inst=inst)
    add_geom(spec, rec, kind="capsule", size=[0.15 * s], pos=None,
             fromto=[*(base + [0, 0, 0.98 * s]), *(base + [0, 0, 1.33 * s])], semantic="other", rgba=shirt,
             group=4, inst=inst)
    add_geom(spec, rec, kind="ellipsoid", size=[0.09 * s, 0.09 * s, 0.11 * s], pos=base + [0, 0, 1.6 * s],
             semantic="other", rgba=skin, group=4, inst=inst)
    return inst


def add_animal(spec, rec: TruthRecorder, rng, xy, kind: str) -> Instance:
    dims = {"dog": (0.32, 0.12, 0.35), "cattle": (0.8, 0.3, 0.75), "nilgai": (0.75, 0.25, 0.8),
            "monkey": (0.22, 0.12, 0.3)}[kind]
    length, girth, leg = (d * rng.uniform(0.85, 1.15) for d in dims)
    col = {"dog": (0.6, 0.45, 0.3, 1), "cattle": (0.85, 0.82, 0.78, 1), "nilgai": (0.45, 0.4, 0.4, 1),
           "monkey": (0.55, 0.5, 0.42, 1)}[kind]
    yaw = rng.uniform(0, 2 * math.pi)
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    side = np.array([-fwd[1], fwd[0], 0.0])
    body_c = np.array([xy[0], xy[1], leg + girth])
    inst = rec.instance("animal", {"kind": kind}, body_c)
    add_geom(spec, rec, kind="ellipsoid", size=[length, girth, girth], pos=body_c, quat=quat_axis([0, 0, 1], yaw),
             semantic="other", rgba=col, group=4, inst=inst)
    for fx in (-0.7, 0.7):
        for sy in (-1, 1):
            top = body_c + fwd * fx * length + side * sy * girth * 0.6
            add_geom(spec, rec, kind="capsule", size=[girth * 0.22], pos=None, fromto=[*top, top[0], top[1], 0.02],
                     semantic="other", rgba=col, group=4, inst=inst)
    head = body_c + fwd * (length + girth * 0.6) + np.array([0, 0, girth * 0.8])
    add_geom(spec, rec, kind="ellipsoid", size=[girth * 0.7, girth * 0.5, girth * 0.5], pos=head, semantic="other",
             rgba=col, group=4, inst=inst)
    return inst


DOCK_XY = (-1.5, -3.5)       # dock entry axis +x


def attach_dock(spec, field, ctx):
    """Order 80: stub straddle dock with a backboard tag and an apron tag (RTH-DOCKING E.5/E.6)."""
    rec: TruthRecorder = ctx.config["truth"]
    plan = _plan(field)
    rng = ctx.rng("dock")
    probe = _probe(plan)
    if not (probe or rng.random() < ctx.config["cfg"]["scene"]["p_dock"]):
        ctx.config["dock"] = None
        return
    x, y = DOCK_XY
    add_geom(spec, rec, kind="box", size=[0.35, 0.22, 0.03], pos=[x, y, 0.03], semantic="other",
             rgba=(0.35, 0.35, 0.38, 1), group=0, prefix="dock_base")
    add_geom(spec, rec, kind="box", size=[0.02, 0.25, 0.25], pos=[x + 0.37, y, 0.28], semantic="other",
             rgba=(0.85, 0.85, 0.85, 1), group=0, prefix="dock_board")
    cover = str(ctx.config.get("dock_tag_cover", "clear"))
    back = rec.instance("dock_tag", {"role": "backboard", "cover": cover}, [x + 0.345, y, 0.3], (-1, 0, 0))
    add_geom(spec, rec, kind="box", size=[0.002, 0.09, 0.09], pos=[x + 0.348, y, 0.3], semantic="other",
             material="tag", rgba=(1, 1, 1, 1), group=4, inst=back)
    apron = rec.instance("dock_tag", {"role": "apron", "cover": cover}, [x, y, 0.062], (0, 0, 1))
    add_geom(spec, rec, kind="box", size=[0.065, 0.065, 0.002], pos=[x, y, 0.062], semantic="other",
             material="tag", rgba=(1, 1, 1, 1), group=4, inst=apron)
    if cover != "clear":     # dust blotches or leaves partly covering the tags (tag dropout is trained)
        for tag_c, normal in (((x + 0.345, y, 0.3), np.array([-1.0, 0, 0])), ((x, y, 0.064), np.array([0, 0, 1.0]))):
            for _ in range(int(rng.integers(1, 4))):
                tc = np.asarray(tag_c)
                lat = rng.uniform(-0.05, 0.05, 3)
                lat -= normal * np.dot(lat, normal)
                p = tc + lat + normal * 0.003
                if cover == "dust":
                    add_geom(spec, rec, kind="ellipsoid", size=[0.03, 0.025, 0.0015], pos=p, quat=quat_z_to(normal),
                             semantic="other", rgba=(0.62, 0.52, 0.38, 1), group=4, prefix="tag_dust")
                else:
                    add_geom(spec, rec, kind="ellipsoid", size=[0.045, 0.02, 0.0015], pos=p,
                             quat=quat_z_to(normal), semantic="foliage", rgba=(0.3, 0.45, 0.15, 1), group=4,
                             prefix="tag_leaf")
    ctx.config["dock"] = {"x": x, "y": y, "back": [x + 0.348, y, 0.3], "apron": [x, y, 0.062]}


# order bands per compose.ORDER (terrain/fixtures 40, trees 50, fruit pool 55, actors 70, dock 80)
def _fp_plan(field, ctx) -> str:
    return hashlib.sha1(L.json_bytes(field["plan"])).hexdigest()[:8]


def stub_modules():
    from atharv_nandi.sim.world import compose
    return [
        compose.Module("synth_environment", 40, attach_environment, version="1", fingerprint=_fp_plan),
        compose.Module("synth_trees", 50, attach_trees, version="1"),
        compose.Module("synth_ground_items", 52, attach_ground_items, version="1"),
        compose.Module("synth_fruit", 55, attach_fruit, version="1"),
        compose.Module("synth_actors", 70, attach_actors, version="1"),
        compose.Module("synth_dock", 80, attach_dock, version="1"),
    ]


# ------------------------------------------------------------------------------------------- scene
@dataclass
class Scene:
    plan: ScenePlan
    model: object
    data: object
    world_id: str
    versions: List[str]
    instances: List[Instance]
    geom_inst: np.ndarray          # geom id -> instance index (-1 none)
    geom_sem: np.ndarray           # geom id -> semantic index
    geom_group0: np.ndarray        # original geom groups (restored after solo passes)
    inst_geoms: List[np.ndarray]   # instance -> geom ids
    scene_tags: dict
    trees: list
    dock: Optional[dict]
    registry: L.ClassRegistry


def _finish(spec, rec: TruthRecorder, plan: ScenePlan, registry, world_id, versions, trees, dock) -> Scene:
    mj = _mj()
    model = spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    sky = registry.sem_id("sky")
    geom_inst = np.full(model.ngeom, -1, dtype=np.int64)
    geom_sem = np.full(model.ngeom, registry.sem_id("other"), dtype=np.int64)
    name_to_id = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g): g for g in range(model.ngeom)}
    for name, sem in rec.semantic.items():
        geom_sem[name_to_id[name]] = registry.sem_id(sem)
    inst_geoms = []
    for inst in rec.instances:
        ids = np.array([name_to_id[n] for n in inst.geoms], dtype=np.int64)
        if ids.size == 0:
            raise L.ClassConfigError(f"instance {inst.iid} ({inst.cls}) has no geom")
        geom_inst[ids] = inst.iid
        inst_geoms.append(ids)
    groups = np.array(model.geom_group, dtype=np.int64)
    # group 5 (query-only) is never labelled; group 2 is reserved for the solo pass
    geom_inst[groups == L.QUERY_ONLY_GROUP] = -1
    # ... and never moved into the solo pass either (it would inflate the occlusion denominator)
    inst_geoms = [ids[groups[ids] != L.QUERY_ONLY_GROUP] for ids in inst_geoms]
    if np.any(groups == L.SOLO_GROUP):
        raise L.ClassConfigError("a scene geom uses the reserved solo group 2")
    del sky
    return Scene(plan, model, data, world_id, versions, rec.instances, geom_inst, geom_sem, groups, inst_geoms,
                 dict(rec.scene_tags), trees, dock, registry)


def build_stub_scene(plan: ScenePlan, cfg: dict, registry: L.ClassRegistry, knobs: Optional[dict] = None,
                     modules: Optional[list] = None) -> Scene:
    """Compose a stub scene: WP0 ComposeContext + Module attach functions, in ORDER, then compile."""
    mj = _mj()
    from atharv_nandi.sim.world import compose
    rec = TruthRecorder(registry)
    knobs = dict(knobs or {})
    field = {"world": STUB_WORLD, "seed": int(plan.seed), "plan": plan.to_dict()}
    ctx = compose.ComposeContext(world=STUB_WORLD, seed=int(plan.seed),
                                 config={"truth": rec, "cfg": cfg, **knobs})
    mods = sorted(modules if modules is not None else stub_modules(), key=lambda m: (m.order, m.name))
    spec = mj.MjSpec()
    spec.modelname = STUB_WORLD
    for m in mods:
        m.attach_fn(spec, field, ctx)
    versions = compose.module_versions(field, ctx, mods)
    wid = compose.world_id(STUB_WORLD, plan.seed, versions + [L.json_bytes(knobs).decode().strip()], "")
    return _finish(spec, rec, plan, registry, wid, versions, ctx.config.get("trees", []), ctx.config.get("dock"))


# ------------------------------------------------------------------------------------------- compose source
def label_patterns(registry: L.ClassRegistry, cfg: dict) -> List[Tuple[re.Pattern, str, str]]:
    """Geom-name regex -> (class, instance unit) for registry-composed worlds (``compose_label_patterns``).

    A value is a regex (one instance per geom) or ``{regex: ..., instance: geom|body}``."""
    pats = cfg.get("compose_label_patterns") or {}
    out = []
    for cls, rx in pats.items():
        if cls not in registry.classes or not registry.classes[cls].mask:
            raise L.ClassConfigError(f"compose_label_patterns: {cls!r} is not a mask class")
        per = "geom"
        if isinstance(rx, dict):
            rx, per = rx["regex"], rx.get("instance", "geom")
        if per not in ("geom", "body"):
            raise L.ClassConfigError(f"compose_label_patterns.{cls}: instance must be geom or body")
        out.append((re.compile(rx), cls, per))
    return out


def build_compose_scene(plan: ScenePlan, cfg: dict, registry: L.ClassRegistry, module_names: Sequence[str],
                        base_xml: Optional[str] = None, world: str = "synth_compose") -> Scene:
    """Select registered compose modules by name (WP4/WP23/WP26/WP37/WP39 once merged) on top of the stub
    environment. Geoms are labelled by ``compose_label_patterns``; one instance per matching body."""
    mj = _mj()
    from atharv_nandi.sim.world import compose
    rec = TruthRecorder(registry)
    reg_mods = {m.name: m for m in compose.modules()}
    missing = [n for n in module_names if n not in reg_mods]
    if missing:
        raise KeyError(f"compose modules not registered: {missing}")
    field = compose.load_field(world, plan.seed)
    field["plan"] = plan.to_dict()
    ctx = compose.ComposeContext(world=world, seed=int(plan.seed), config={"truth": rec, "cfg": cfg})
    spec = mj.MjSpec.from_file(base_xml) if base_xml else mj.MjSpec()
    env = compose.Module("synth_environment", 40, attach_environment, version="1", fingerprint=_fp_plan)
    mods = sorted([env] + [reg_mods[n] for n in module_names], key=lambda m: (m.order, m.name))
    known = {g.name for g in spec.geoms}
    for m in mods:
        m.attach_fn(spec, field, ctx)
    pats = label_patterns(registry, cfg)
    by_key: Dict[Tuple[str, str], Instance] = {}
    for g in spec.geoms:
        if g.name in known or g.name in rec.semantic or not g.name:
            continue
        for rx, cls, per in pats:
            if rx.search(g.name):
                key = (cls, (g.parent.name if g.parent is not None else "") if per == "body" else g.name)
                if key not in by_key:
                    by_key[key] = rec.instance(cls, {}, list(g.pos))
                by_key[key].geoms.append(g.name)
                rec.semantic[g.name] = registry.classes[cls].semantic
                break
    for g in spec.geoms:
        if g.name and g.name not in rec.semantic:
            rec.semantic[g.name] = "other"
    versions = compose.module_versions(field, ctx, mods)
    wid = compose.world_id(world, plan.seed, versions, compose.registry_hash(field))
    trees = []
    for t in field.get("trees") or []:
        xy = t.get("pos") or (t.get("geo") or {}).get("local_xy") or [t.get("x"), t.get("y")]
        if xy and xy[0] is not None:
            trees.append({"x": float(xy[0]), "y": float(xy[1]), "species": t.get("crop", "")})
    return _finish(spec, rec, plan, registry, wid, versions, trees, None)


# ------------------------------------------------------------------------------------------- views
@dataclass
class View:
    rig: str
    eye: np.ndarray
    target: np.ndarray
    fovy: float

    def mjv_camera(self):
        mj = _mj()
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        d = np.asarray(self.target, float) - np.asarray(self.eye, float)
        dist = float(np.linalg.norm(d))
        f = d / dist
        cam.lookat[:] = self.target
        cam.distance = dist
        cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
        cam.elevation = math.degrees(math.asin(float(np.clip(f[2], -1, 1))))
        return cam

    def to_dict(self) -> dict:
        return {"rig": self.rig, "eye": [round(float(v), 6) for v in self.eye],
                "target": [round(float(v), 6) for v in self.target], "fovy_deg": round(float(self.fovy), 6)}


def _u(rng, r) -> float:
    return float(rng.uniform(float(r[0]), float(r[1])))


def sample_view(scene: Scene, cams: dict, rng: np.random.Generator, rig: Optional[str] = None) -> View:
    """One camera pose of a real Nandi rig at a station pose (WP25/WP26) with jitter, or a dock approach pose."""
    rigs = [r for r in ("ring", "crown", "wrist", "head", "belly", "rear", "dock") if r in cams]
    if scene.dock is None:
        rigs = [r for r in rigs if r != "dock"]
    if rig is None:
        w = np.array([float(cams[r]["weight"]) for r in rigs])
        rig = rigs[int(rng.choice(len(rigs), p=w / w.sum()))]
    c = cams[rig]
    jit = float(cams.get("pose_jitter_m", 0.05))
    trees = scene.trees or [{"x": 0.0, "crown_c": np.array([0, 0, 1.5]), "trunk_h": 1.0}]
    tree = trees[int(rng.integers(0, len(trees)))]
    tx, ty = float(tree["x"]), float(tree.get("y", 0.0))
    a = rng.uniform(0, 2 * math.pi)
    radial = np.array([math.cos(a), math.sin(a), 0.0])
    fovy = _u(rng, c["fovy_deg"])
    if rig in ("ring", "crown", "head", "rear"):
        h = float(rng.choice(c["heights_m"])) if "heights_m" in c else _u(rng, c["height_m"])
        d = _u(rng, c["standoff_m"])
        eye = np.array([tx, ty, 0]) + radial * d + np.array([0, 0, h])
        target = np.array([tx, ty, _u(rng, c["target_z_m"])]) + rng.normal(0, 0.15, 3) * np.array([1, 1, 0.5])
    elif rig == "wrist":
        cands = [i for i in scene.instances if i.cls in ("fruit", "leaf_symptom", "panicle", "trunk_base", "band",
                                                          "trap", "sooty_sheen", "bag", "ants_on_band")]
        if not cands:
            cands = scene.instances
        inst = cands[int(rng.integers(0, len(cands)))]
        n = np.asarray(inst.normal, float)
        n = n + rng.normal(0, 0.25, 3)
        n /= np.linalg.norm(n)
        target = np.asarray(inst.anchor, float)
        eye = target + n * _u(rng, c["standoff_m"])
        eye[2] = max(eye[2], 0.08)
    elif rig == "belly":
        h = _u(rng, c["height_m"])
        xy = np.array([tx, ty]) + radial[:2] * rng.uniform(0.4, 1.4)
        eye = np.array([xy[0], xy[1], h])
        target = eye + np.array([rng.normal(0, 0.05), rng.normal(0, 0.05), -h])
    else:   # dock approach: staging, pre-entry (head cam on the backboard tag), or belly over the apron tag
        dk = scene.dock
        stage = str(rng.choice(["staging", "pre_entry", "belly"]))
        back, apron = np.asarray(dk["back"]), np.asarray(dk["apron"])
        if stage == "belly":
            eye = apron + np.array([rng.normal(0, 0.03), rng.normal(0, 0.03), 0.29])
            target = apron + np.array([rng.normal(0, 0.01), rng.normal(0, 0.01), 0.0])
            fovy = 90.0
        else:
            off = float(c["staging_m"]) if stage == "staging" else float(c["pre_entry_m"]) + 0.35
            eye = np.array([dk["x"] - off, dk["y"] + rng.normal(0, 0.05), 0.55])
            target = back + rng.normal(0, 0.02, 3)
    eye = eye + rng.normal(0, jit, 3) * np.array([1, 1, 0.3])
    eye[2] = max(eye[2], 0.05)
    yaw = math.radians(rng.normal(0, float(cams.get("yaw_jitter_deg", 4.0))))
    d = target - eye
    cs, sn = math.cos(yaw), math.sin(yaw)
    target = eye + np.array([cs * d[0] - sn * d[1], sn * d[0] + cs * d[1], d[2]])
    return View(rig, eye, target, fovy)
