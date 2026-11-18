"""World composition registry (WP0; MVP-SPEC 3.1, ORCHARD-MVP-SPEC 3.1).

A work package adds content to the world by dropping one file into ``atharv_nandi/sim/world/modules/`` that calls
``register_module(name, order, attach_fn)`` at import time. It never edits another package's file and
never edits ``farm_scene.py``. ``build()`` loads the robot MJCF, then calls every registered module in
ascending ``order`` (ties by name), then generates the home keyframe in code (nq varies by world).

ORDER bands (ORCHARD-MVP-SPEC 3.1):
    robot 0 · legacy crops 10 · tools 30 · terrain and fixtures 40 · trees 50 · fruit pool 55 · actors 70 ·
    dock and crate 80
Joints are numbered in body-tree order, and every module adds its bodies as world children after the
ones before it, so a module's joints land after the robot and legacy crops (design rule 1).

``attach_fn(spec, field, ctx)``:
    spec   the ``mujoco.MjSpec`` being built (edit it in place)
    field  the field dict of the world (``config/fields/<world>.json`` when present, WP4), always with
           ``world`` and ``seed``
    ctx    a ``ComposeContext``: world, seed, config knobs, ``rng(name)`` (deterministic per module) and
           ``set_home(joint, values)`` to override a joint's home-keyframe qpos

``world_id = sha1(world, seed, module versions, tree registry hash)[:12]``: every consumer of
``/nandi/sim_state`` checks it, so two processes that composed different worlds never exchange state.
"""
import contextlib
import glob
import hashlib
import importlib
import json
import os
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi import paths

REPO = paths.REPO
MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules")
FIELDS_DIR = os.path.join(REPO, "config", "fields")

ORDER = {"robot": 0, "legacy_crops": 10, "tools": 30, "terrain": 40, "fixtures": 40, "trees": 50,
         "fruit_pool": 55, "actors": 70, "dock": 80}
WORLDS_DEFAULT = "legacy"


@dataclass
class Module:
    name: str
    order: int
    attach_fn: Callable
    version: str = "1"
    # Optional fingerprint(field, ctx) -> str: parameters that change the composed world (folded into world_id).
    fingerprint: Optional[Callable] = None


@dataclass
class ComposeContext:
    world: str
    seed: int
    config: dict = dc_field(default_factory=dict)
    home: Dict[str, List[float]] = dc_field(default_factory=dict)
    home_mocap: Dict[str, Tuple[List[float], List[float]]] = dc_field(default_factory=dict)

    def rng(self, name: str) -> np.random.Generator:
        """A generator seeded from (seed, name): a module's draws never shift another module's."""
        h = int(hashlib.sha1(f"{self.seed}:{name}".encode()).hexdigest()[:8], 16)
        return np.random.default_rng(h)

    def set_home(self, joint: str, values: Sequence[float]):
        """Home-keyframe qpos for ``joint`` (default: the robot key for robot joints, else qpos0)."""
        self.home[joint] = [float(v) for v in values]

    def set_home_mocap(self, body: str, pos: Sequence[float], quat: Sequence[float] = (1, 0, 0, 0)):
        self.home_mocap[body] = ([float(v) for v in pos], [float(v) for v in quat])


_REGISTRY: Dict[str, Module] = {}
_DISCOVERED = False


def register_module(name: str, order: int, attach_fn: Callable, version: str = "1",
                    fingerprint: Optional[Callable] = None) -> Module:
    """Register (or replace, same name) a world module. Call at import time of ``atharv_nandi/sim/world/modules/<name>.py``."""
    if not isinstance(order, int) or not 0 <= order <= 100:
        raise ValueError(f"module {name!r}: order {order!r} outside 0..100")
    mod = Module(name, order, attach_fn, str(version), fingerprint)
    _REGISTRY[name] = mod
    return mod


def unregister_module(name: str) -> None:
    _REGISTRY.pop(name, None)


def modules() -> List[Module]:
    """Registered modules in build order (order, then name)."""
    return sorted(_REGISTRY.values(), key=lambda m: (m.order, m.name))


def discover(force: bool = False) -> List[str]:
    """Import every ``atharv_nandi/sim/world/modules/*.py`` once so its ``register_module`` call runs. Returns module names."""
    global _DISCOVERED
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(MODULES_DIR, "*.py")))
    names = [n for n in names if not n.startswith("_")]
    if _DISCOVERED and not force:
        return names
    for n in names:
        importlib.import_module(f"{__package__}.modules.{n}")
    _DISCOVERED = True
    return names


# ---------------------------------------------------------------------- field + world id
def world_name() -> str:
    return (os.environ.get("NANDI_WORLD", "") or WORLDS_DEFAULT).strip().lower()


def load_field(world: str, seed: int, fields_dir: str = FIELDS_DIR) -> dict:
    """``config/fields/<world>.json`` if it exists (WP4 owns those files), else a bare field."""
    path = os.path.join(fields_dir, f"{world}.json")
    field: dict = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            field = json.load(fh)
    field = dict(field)
    field["world"], field["seed"] = world, int(seed)
    return field


def registry_hash(field: dict) -> str:
    """sha1 of the tree registry (``trees``) in canonical JSON; '' when the field has none."""
    trees = field.get("trees")
    if not trees:
        return ""
    return hashlib.sha1(json.dumps(trees, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def module_versions(field: dict, ctx: ComposeContext, mods: Optional[List[Module]] = None) -> List[str]:
    out = []
    for m in (mods if mods is not None else modules()):
        fp = m.fingerprint(field, ctx) if m.fingerprint else ""
        out.append(f"{m.name}@{m.version}" + (f"[{fp}]" if fp else ""))
    return out


def world_id(world: str, seed: int, versions: Sequence[str], reg_hash: str = "") -> str:
    payload = json.dumps([world, int(seed), list(versions), reg_hash], separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------- build
@dataclass
class Composed:
    spec: object
    world: str
    seed: int
    world_id: str
    field: dict
    versions: List[str]


def _spec_nq(spec) -> int:
    from atharv_nandi.core.model_layout import JNT_NQ
    return sum(JNT_NQ[int(j.type)] for j in spec.joints)


def _generate_keyframe(spec, ctx: ComposeContext):
    """Make key 0 cover every joint: robot key values for the prefix, module overrides, qpos0 for the rest.

    Skipped (spec untouched) when the key already has nq entries and no override or mocap needs filling,
    which is the legacy world: its key is byte-identical to the pre-WP0 one.
    """
    import mujoco
    if not spec.keys:
        spec.add_key(name="home")
    key = spec.keys[0]
    have = np.asarray(key.qpos, dtype=np.float64).reshape(-1)
    n_mocap_bodies = sum(1 for b in spec.bodies if getattr(b, "mocap", False))
    if have.size == _spec_nq(spec) and not ctx.home and not ctx.home_mocap and n_mocap_bodies == 0:
        return
    model = spec.compile()
    qpos = np.array(model.qpos0, dtype=np.float64)
    n = min(have.size, model.nq)
    qpos[:n] = have[:n]
    for joint, values in ctx.home.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            raise KeyError(f"set_home: joint {joint!r} not in the composed model")
        adr = int(model.jnt_qposadr[jid])
        qpos[adr:adr + len(values)] = values
    key.qpos = qpos.tolist()
    if model.nmocap:
        mpos = np.zeros((model.nmocap, 3))
        mquat = np.zeros((model.nmocap, 4))
        for b in range(model.nbody):
            mid = int(model.body_mocapid[b])
            if mid < 0:
                continue
            mpos[mid], mquat[mid] = model.body_pos[b], model.body_quat[b]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            if name in ctx.home_mocap:
                mpos[mid], mquat[mid] = ctx.home_mocap[name]
        key.mpos = mpos.reshape(-1).tolist()
        key.mquat = mquat.reshape(-1).tolist()


def identify(world: str, seed: int, config: Optional[dict] = None, fields_dir: str = FIELDS_DIR,
             discover_modules: bool = True):
    """(world_id, field, ctx, modules, versions) without building anything: cheap, for consumers that check ids."""
    if discover_modules:
        discover()
    mods = modules()
    field = load_field(world, seed, fields_dir)
    ctx = ComposeContext(world=world, seed=int(seed), config=dict(config or {}))
    versions = module_versions(field, ctx, mods)
    wid = world_id(world, seed, versions, registry_hash(field))
    ctx.config.setdefault("world_id", wid)
    return wid, field, ctx, mods, versions


def build(world: str, seed: int, robot_xml: str, config: Optional[dict] = None,
          fields_dir: str = FIELDS_DIR, discover_modules: bool = True) -> Composed:
    """Compose the world: robot, then every registered module in ORDER, then the generated home keyframe."""
    import mujoco
    wid, field, ctx, mods, versions = identify(world, seed, config, fields_dir, discover_modules)
    spec = mujoco.MjSpec.from_file(robot_xml)
    for m in mods:
        m.attach_fn(spec, field, ctx)
    _generate_keyframe(spec, ctx)
    return Composed(spec=spec, world=world, seed=int(seed), world_id=wid, field=field, versions=versions)


@contextlib.contextmanager
def temporary_module(name: str, order: int, attach_fn: Callable, version: str = "1",
                     fingerprint: Optional[Callable] = None):
    """Register a module for the duration of a ``with`` block (tests, probes); restores any module it shadowed."""
    # Discover first: if the shadowed module file were first imported inside the block, its register_module
    # call would land in the block and the exit would drop it for the rest of the process (import runs once).
    discover()
    previous = _REGISTRY.get(name)
    register_module(name, order, attach_fn, version, fingerprint)
    try:
        yield _REGISTRY[name]
    finally:
        if previous is None:
            _REGISTRY.pop(name, None)
        else:
            _REGISTRY[name] = previous
