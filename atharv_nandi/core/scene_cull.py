"""mjvScene camera-distance post-filter (WP5; ORCHARD-MVP-SPEC 3.7, WP11a budget spike).

After ``mjv_updateScene`` (``mujoco.Renderer.update_scene``) the perception model drops the scene geoms that are
farther than ``max_dist`` (4 m) from the rendering camera and compacts ``scene.ngeom``. The mjvScene API permits
this (the scene is a plain output buffer of the update and an input of ``mjr_render``). **No model field is
mutated**, so nothing has to be restored and nothing of it is in ``/nandi/sim_state``: render-only culling is not
sim state (OS 3.2). Geom groups (``scene.flags`` / ``opt.geomgroup``) are the second lever and stay with the
caller.

Distance = ``|geom pos - camera| - geom_rbound`` (the bounding sphere), so a tree whose canopy reaches into the
4 m sphere is kept. Always kept, at any distance:

* planes and height fields (ground, terrain),
* scene entries that are not model geoms (sites, decor, labels),
* every geom of a **mocap body**: people, animals and debris actors. Person and animal pixels are privacy truth
  (MVP-SPEC 5.4) and safety-relevant, so the cull never removes them.

What it does remove (static scenery beyond 4 m: far trees, laterals, bunds, a far parcel boundary) is the
assisted-register entry ``scene_cull_4m`` (OS 3.9). WP11a measured 894 -> median 125 scene geoms at the OT-1
stations and a 16-31 % shorter capture (``budget-spike``).

Compaction runs on a raw byte view of the ``mjvGeom`` array (vectorised). The layout is verified once at
construction with a sentinel write/read; when it does not match, a per-field copy is used instead (same result,
``tests/test_privacy.py`` checks both). ``segid`` is renumbered to the new scene index, because
``mujoco.Renderer`` decodes a segmentation image with ``scene.geoms[segid]`` (segid >= ngeom raises).
"""
import ctypes
import hashlib
from typing import Sequence, Tuple

import mujoco
import numpy as np

CULL_DIST_M = 4.0
_GEOM_FIELDS = ("type", "dataid", "objtype", "objid", "category", "matid", "texcoord", "segid", "size", "pos",
                "mat", "rgba", "emission", "specular", "shininess", "reflectance", "label", "camdist",
                "modelrbound", "transparent")


class SceneCuller:
    """Post-filter one ``mujoco.MjvScene`` by distance from the camera. Bind it to the scene it filters."""

    _INT_OFF = {"type": 0, "objtype": 8, "objid": 12, "segid": 28}

    def __init__(self, model: mujoco.MjModel, scene: mujoco.MjvScene, max_dist: float = CULL_DIST_M,
                 force_slow: bool = False):
        self.model, self.scene, self.max_dist = model, scene, float(max_dist)
        self.rbound = np.asarray(model.geom_rbound, dtype=np.float64).copy()
        G = mujoco.mjtGeom
        self.always = np.isin(np.asarray(model.geom_type), [int(G.mjGEOM_PLANE), int(G.mjGEOM_HFIELD)])
        if model.ngeom:
            self.always |= np.asarray(model.body_mocapid)[np.asarray(model.geom_bodyid)] >= 0
        self.raw = None
        self.fast = False if force_slow else self._map()
        self.last: Tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------ raw layout
    def _map(self) -> bool:
        sc = self.scene
        if sc.maxgeom < 2:
            return False
        g0, g1 = sc.geoms[0], sc.geoms[1]
        a0 = g0.size.ctypes.data
        stride = g1.size.ctypes.data - a0
        base = a0 - 32                      # 8 int32 fields precede size[3] in mjvGeom (MuJoCo 3.3)
        self.pos_off = g0.pos.ctypes.data - base
        if stride <= 0 or self.pos_off != 44:
            return False
        buf = (ctypes.c_uint8 * (stride * sc.maxgeom)).from_address(base)
        self.raw = np.ctypeslib.as_array(buf).reshape(sc.maxgeom, stride)
        self.stride = stride
        old = (g1.type, g1.objtype, g1.objid, g1.segid)
        g1.type, g1.objtype, g1.objid, g1.segid = 5, 7, 12345, 4321
        ok = (self._ints("type")[1] == 5 and self._ints("objtype")[1] == 7 and self._ints("objid")[1] == 12345
              and self._ints("segid")[1] == 4321)
        g1.type, g1.objtype, g1.objid, g1.segid = old
        if not ok:
            self.raw = None
        return bool(ok)

    def _ints(self, field: str) -> np.ndarray:
        o = self._INT_OFF[field]
        return self.raw[:, o:o + 4].copy().view(np.int32).ravel()

    # ------------------------------------------------------------------ filter
    def keep_mask(self, cam_pos: Sequence[float]) -> np.ndarray:
        sc, n = self.scene, self.scene.ngeom
        if n == 0:
            return np.zeros(0, dtype=bool)
        if self.fast:
            pos = self.raw[:n, self.pos_off:self.pos_off + 12].copy().view(np.float32).reshape(n, 3)
            objtype = self._ints("objtype")[:n]
            objid = self._ints("objid")[:n]
        else:
            pos = np.array([sc.geoms[i].pos for i in range(n)], dtype=np.float32).reshape(n, 3)
            objtype = np.array([sc.geoms[i].objtype for i in range(n)], dtype=np.int32)
            objid = np.array([sc.geoms[i].objid for i in range(n)], dtype=np.int32)
        is_geom = (objtype == int(mujoco.mjtObj.mjOBJ_GEOM)) & (objid >= 0) & (objid < self.model.ngeom)
        dist = np.linalg.norm(pos.astype(np.float64) - np.asarray(cam_pos, dtype=np.float64), axis=1)
        gid = np.where(is_geom, objid, 0)
        far = is_geom & ~self.always[gid] & ((dist - self.rbound[gid]) > self.max_dist)
        return ~far

    def cull(self, cam_pos: Sequence[float]) -> Tuple[int, int]:
        """Filter the scene in place; returns ``(ngeom_before, ngeom_after)``."""
        sc = self.scene
        n = sc.ngeom
        keep = self.keep_mask(cam_pos)
        idx = np.nonzero(keep)[0]
        k = int(idx.size)
        if k == n:
            self.last = (n, n)
            return self.last
        if self.fast:
            self.raw[:k] = self.raw[idx]
            self.raw[:k, 28:32] = np.arange(k, dtype=np.int32).view(np.uint8).reshape(k, 4)
        else:
            for dst, src in enumerate(idx):
                if dst != src:
                    gs, gd = sc.geoms[int(src)], sc.geoms[dst]
                    for f in _GEOM_FIELDS:
                        if hasattr(gs, f):
                            v = getattr(gs, f)
                            if isinstance(v, np.ndarray):
                                getattr(gd, f)[...] = v
                            else:
                                setattr(gd, f, v)
                sc.geoms[dst].segid = dst
        sc.ngeom = k
        self.last = (n, k)
        return self.last


def model_fingerprint(m: mujoco.MjModel) -> str:
    """Hash of the model fields a render-side change could touch (the cull test: identical before and after)."""
    h = hashlib.sha1()
    for a in (m.geom_rgba, m.geom_group, m.geom_size, m.geom_pos, m.geom_quat, m.geom_contype,
              m.geom_conaffinity, m.geom_rbound, m.body_pos, m.body_quat, m.cam_pos, m.cam_quat,
              m.mat_rgba, m.light_pos, m.light_dir):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]
