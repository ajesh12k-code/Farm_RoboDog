"""Robot self-masks for the ring and crown cameras (WP5; MVP-SPEC 2.4; ORCHARD-MVP-SPEC 2.3, 6).

Ring cameras are rigid on the fixed mast ring, so their mask is **static**: a boolean mask of robot pixels per
camera, made once from a segmentation pass at the home pose and dilated 2 px (``StaticSelfMask``). The arm moves
through the front camera's view, so the per-frame mask adds the **arm's projected hull** at the current arm pose
(``arm_hull_mask``: the 8 corners of every arm / gripper / mounted-tool geom box, projected, convex-hull filled,
dilated). The crown camera moves with the mast and the gimbal: its mask comes from the segmentation pass of the
same capture (a TREE 360 capture is RGB + depth + seg), ``mask_from_seg``.

Validity (``coverage``): every robot pixel of a fresh segmentation pass must lie inside the mask. The tests check
this for the ring at mast extension 0, 0.6 and 1.6 m (the moving sections are above and behind the ring) and for
the crown camera at the same extensions, tilted down at the robot.

Robot geoms: every geom of the ``trunk`` subtree (``robot_geoms``) plus geoms of free bodies welded to it (rack
tools), by name prefix ``tool_``. The seg pass reports (objid, objtype) per pixel; objtype mjOBJ_GEOM.
"""
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

ROBOT_ROOT = "trunk"
ARM_BODY_PREFIXES = ("arm_", "gripper", "finger", "gantry")
TOOL_PREFIXES = ("tool_",)
DILATE_PX = 2


def subtree_bodies(model, root: str = ROBOT_ROOT) -> np.ndarray:
    import mujoco
    r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    if r < 0:
        raise KeyError(f"self_mask: body {root!r} not in model")
    parent = np.asarray(model.body_parentid)
    inside = np.zeros(model.nbody, dtype=bool)
    inside[r] = True
    for b in range(model.nbody):        # parents precede children in MuJoCo's body order
        if parent[b] >= 0 and inside[parent[b]] and b != 0:
            inside[b] = True
    return inside


def robot_geoms(model, root: str = ROBOT_ROOT, tool_prefixes: Sequence[str] = TOOL_PREFIXES) -> np.ndarray:
    """bool per geom: geoms of the robot subtree and of the rack tools (separate free bodies)."""
    import mujoco
    inside = subtree_bodies(model, root)
    out = inside[np.asarray(model.geom_bodyid)].copy()
    for b in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith(tuple(tool_prefixes)):
            out |= np.asarray(model.geom_bodyid) == b
    return out


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Square (Chebyshev) dilation by ``px`` pixels, pure numpy."""
    m = np.asarray(mask, dtype=bool)
    if px <= 0 or not m.any():
        return m.copy()
    H, W = m.shape
    out = m.copy()
    for _ in range(int(px)):
        n = out.copy()
        n[1:, :] |= out[:-1, :]
        n[:-1, :] |= out[1:, :]
        n[:, 1:] |= out[:, :-1]
        n[:, :-1] |= out[:, 1:]
        n[1:, 1:] |= out[:-1, :-1]
        n[:-1, :-1] |= out[1:, 1:]
        n[1:, :-1] |= out[:-1, 1:]
        n[:-1, 1:] |= out[1:, :-1]
        out = n
    return out


def seg_robot(seg: np.ndarray, robot: np.ndarray) -> np.ndarray:
    """Robot pixels of a ``mujoco.Renderer`` segmentation image (H, W, 2: objid, objtype)."""
    import mujoco
    objid, objtype = seg[..., 0], seg[..., 1]
    is_geom = (objtype == int(mujoco.mjtObj.mjOBJ_GEOM)) & (objid >= 0) & (objid < len(robot))
    out = np.zeros(objid.shape, dtype=bool)
    out[is_geom] = robot[objid[is_geom]]
    return out


def mask_from_seg(seg: np.ndarray, robot: np.ndarray, dilate_px: int = DILATE_PX) -> np.ndarray:
    return dilate(seg_robot(seg, robot), dilate_px)


def coverage(mask: np.ndarray, robot_pixels: np.ndarray) -> float:
    """Share of the robot pixels that the mask covers (1.0 when there are none)."""
    n = int(np.sum(robot_pixels))
    return 1.0 if n == 0 else float(np.sum(mask & robot_pixels)) / n


def fill_convex(shape, pts_uv: np.ndarray) -> np.ndarray:
    """Fill the convex hull of 2-D points (pixel coordinates) into a bool mask."""
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    P = np.asarray(pts_uv, dtype=np.float64)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 3:
        return m
    hull = _hull(P)
    if len(hull) < 3:
        return m
    u0, v0 = np.floor(hull.min(axis=0)).astype(int)
    u1, v1 = np.ceil(hull.max(axis=0)).astype(int)
    u0, v0, u1, v1 = max(0, u0), max(0, v0), min(W - 1, u1), min(H - 1, v1)
    if u0 > u1 or v0 > v1:
        return m
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1].astype(np.float64)
    inside = np.ones(uu.shape, dtype=bool)
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        cross = (b[0] - a[0]) * (vv + 0.5 - a[1]) - (b[1] - a[1]) * (uu + 0.5 - a[0])
        inside &= cross >= -1e-9
    m[v0:v1 + 1, u0:u1 + 1] = inside
    return m


def _hull(P: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain, counter-clockwise in (u, v)."""
    pts = sorted(set(map(tuple, np.round(P, 6))))
    if len(pts) < 3:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def arm_geoms(model, prefixes: Sequence[str] = ARM_BODY_PREFIXES, tool_prefixes: Sequence[str] = TOOL_PREFIXES):
    import mujoco
    out = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith(tuple(prefixes)) or n.startswith(tuple(tool_prefixes)):
            if float(model.geom_rgba[g][3]) > 0.0 or int(model.geom_group[g]) == 2:
                out.append(g)
    return out


def arm_hull_mask(model, data, pose, intr, geoms: Optional[Iterable[int]] = None, dilate_px: int = DILATE_PX,
                  near: float = 1e-3) -> np.ndarray:
    """Projected convex hull of the arm (and a mounted tool) in one camera; empty when out of view."""
    from atharv_nandi.core.privacy_filter import geom_box_corners
    gs = list(arm_geoms(model) if geoms is None else geoms)
    shape = (intr.height, intr.width)
    if not gs:
        return np.zeros(shape, dtype=bool)
    out = np.zeros(shape, dtype=bool)
    for g in gs:
        C = geom_box_corners(model.geom_type[g], model.geom_size[g], data.geom_xpos[g], data.geom_xmat[g])
        P = pose.apply(C)                     # optical frame
        front = P[:, 2] > near
        if not front.any():
            continue
        pts = [P[front]]
        if not front.all():                   # clip the box edges at the near plane (the box is convex)
            for i, j in _BOX_EDGES:
                zi, zj = P[i, 2], P[j, 2]
                if (zi > near) != (zj > near):
                    s = (near - zi) / (zj - zi)
                    pts.append((P[i] + s * (P[j] - P[i]))[None, :])
        Q = np.concatenate(pts, axis=0)
        Q[:, 2] = np.maximum(Q[:, 2], near)
        uv = np.stack([intr.cx + intr.fx * Q[:, 0] / Q[:, 2], intr.cy + intr.fy * Q[:, 1] / Q[:, 2]], axis=1)
        uv = np.clip(uv, -4.0 * max(shape), 5.0 * max(shape))
        out |= fill_convex(shape, uv)
    return dilate(out, dilate_px)


# corner index = 4*ix + 2*iy + iz (geom_box_corners order); edges differ in exactly one bit
_BOX_EDGES = [(a, a ^ bit) for a in range(8) for bit in (1, 2, 4) if a < a ^ bit]


class StaticSelfMask:
    """Per-camera static masks (ring), built from segmentation passes, plus the per-frame arm hull."""

    def __init__(self, dilate_px: int = DILATE_PX):
        self.dilate_px = int(dilate_px)
        self.masks: Dict[str, np.ndarray] = {}

    def build(self, camera: str, segs: Iterable[np.ndarray], robot: np.ndarray) -> np.ndarray:
        """Union over one or more seg passes (home pose; more poses only widen it), dilated."""
        acc = None
        for s in segs:
            r = seg_robot(s, robot)
            acc = r if acc is None else (acc | r)
        if acc is None:
            raise ValueError("self_mask: no segmentation pass")
        self.masks[camera] = dilate(acc, self.dilate_px)
        return self.masks[camera]

    def get(self, camera: str, arm_mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        m = self.masks.get(camera)
        if m is None:
            return arm_mask
        return m if arm_mask is None else (m | arm_mask)
