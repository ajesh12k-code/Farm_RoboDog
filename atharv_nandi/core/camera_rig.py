"""Perception-only MuJoCo model for Atharv Nandi: the shared farm world plus two cameras.

* ``head_cam``  - on the trunk front, looking forward, pitched down (default fovy 70, 35 deg down).
* ``wrist_cam`` - on the ``arm_wrist`` (gripper) body, looking along the gripper.

The robot MJCF (WP1) already carries the 4 ring cameras ``ring_0..3`` and the ``crown_cam`` on the crown head;
the crown gimbal is ``cam_quat[crown_cam]``, written by the WP24 mast provider through ``set_state``, so a
perception render shows the raised crown and the aimed view. WP5 adds the segmentation pass, the TREE 360
``capture`` (RGB + depth + seg) and the mjvScene camera-distance post-filter (``cull_dist``, OS 3.7).

The cameras are added programmatically to the same ``MjSpec`` that ``farm_scene`` compiles for
every live node, so the robot MJCF and ``farm_scene.py`` stay untouched and ``nq`` / the qpos layout
are identical to ``farm_scene.load_farm_model()`` (cameras add no degrees of freedom). The physics
model used by ``sim_runner`` is unchanged: this model only ever has state copied in (``set_state`` from
``/nandi/sim_state``: qpos, mocap, eq_active and module mutations; ``set_qpos`` from ``/nandi/sim_qpos`` in the
legacy world) and is never stepped.

Conventions: MuJoCo cameras look along their local -z with +y up. ``camera_pose`` converts to the
computer-vision optical frame (x right, y down, z forward) that ``detector_model`` uses.
"""
import math
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from atharv_nandi.core import farm_scene
from atharv_nandi.core import sim_state as sim_state_mod
from atharv_nandi.core.detector_model import Intrinsics, Pose, intrinsics_from_fovy, load_config

HEAD_CAM = "head_cam"
WRIST_CAM = "wrist_cam"
CAMERA_NAMES = (HEAD_CAM, WRIST_CAM)     # the cameras this module adds; the ring and crown are in the robot MJCF
RING_CAMS = ("ring_0", "ring_1", "ring_2", "ring_3")
CROWN_CAM = "crown_cam"
RING_YAW_DEG = {"ring_0": 0.0, "ring_1": 90.0, "ring_2": 180.0, "ring_3": 270.0}
# Geom groups a camera renders (OS 2.5): 0 wood/ground, 1 fruit/actors, 2 robot, 3 foliage, 4 decals; never 5
# (query-only: foliage envelope, rope lines, laterals, robot mass-only geoms). MuJoCo's default MjvOption shows
# only groups 0-2, which would hide every canopy and decal from the cameras.
RENDER_GROUPS = (1, 1, 1, 1, 1, 0)
TRUNK_BODY = "trunk"
WRIST_BODY = "arm_wrist"
# MuJoCo camera frame (x right, y up, z back) -> optical frame (x right, y down, z forward).
_MJ_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


def _camera_quat(pitch_down_deg: float) -> np.ndarray:
    """Quaternion of a camera looking along the parent's +x, pitched down, image-right = parent -y."""
    p = math.radians(pitch_down_deg)
    x_axis = np.array([0.0, -1.0, 0.0])
    y_axis = np.array([math.sin(p), 0.0, math.cos(p)])
    z_axis = np.cross(x_axis, y_axis)  # = -forward
    mat = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns = camera axes in the parent frame
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    return quat


_WORLD_ID: Optional[str] = None   # world_id of the last perception spec built in this process


def _farm_spec() -> mujoco.MjSpec:
    # The same composed world as load_farm_model() (read-only reuse of the shared builder), so the perception
    # model cannot drift from the physics world. Its world_id is kept for set_state.
    global _WORLD_ID
    world = farm_scene.build_world()
    _WORLD_ID = world.world_id
    return world.spec


def build_perception_spec(cfg: Optional[dict] = None) -> mujoco.MjSpec:
    cam_cfg = (cfg if cfg is not None else load_config()).get("camera", {})
    spec = _farm_spec()
    trunk = spec.body(TRUNK_BODY)
    wrist = spec.body(WRIST_BODY)
    if trunk is None or wrist is None:
        raise RuntimeError("camera_rig: trunk / arm_wrist body not found in the farm model")
    trunk.add_camera(
        name=HEAD_CAM,
        pos=list(cam_cfg.get("head_pos_trunk", [0.35, 0.0, 0.06])),
        quat=list(_camera_quat(float(cam_cfg.get("head_pitch_down_deg", 35.0)))),
        fovy=float(cam_cfg.get("head_fovy_deg", 70.0)),
    )
    wrist.add_camera(
        name=WRIST_CAM,
        pos=list(cam_cfg.get("wrist_pos_body", [0.02, 0.0, 0.03])),
        quat=list(_camera_quat(0.0)),
        fovy=float(cam_cfg.get("wrist_fovy_deg", 60.0)),
    )
    # Offscreen buffer large enough for the 640x480 benchmark resolution.
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), 640)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), 480)
    return spec


def build_perception_model(cfg: Optional[dict] = None) -> mujoco.MjModel:
    """Farm world + head_cam + wrist_cam. Never step it; copy qpos in and call ``mj_kinematics``."""
    return build_perception_spec(cfg).compile()


def camera_id(model: mujoco.MjModel, name: str) -> int:
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        raise KeyError(f"camera {name!r} not in model")
    return cid


def intrinsics(model: mujoco.MjModel, name: str, width: int, height: int) -> Intrinsics:
    return intrinsics_from_fovy(float(model.cam_fovy[camera_id(model, name)]), width, height)


def camera_pose(data: mujoco.MjData, cid: int) -> Pose:
    """World -> optical-frame transform of camera ``cid`` (needs kinematics up to date)."""
    rot = data.cam_xmat[cid].reshape(3, 3) @ _MJ_TO_OPTICAL
    return Pose.from_frame_in_world(data.cam_xpos[cid], rot)


def body_pose(model: mujoco.MjModel, data: mujoco.MjData, body: str = TRUNK_BODY) -> Pose:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return Pose.from_frame_in_world(data.xpos[bid], data.xmat[bid].reshape(3, 3))


def crop_truth(model: mujoco.MjModel, data: mujoco.MjData) -> Tuple[List[str], np.ndarray]:
    names = farm_scene.crop_body_names(model)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    return names, np.array([data.xpos[i] for i in ids], dtype=np.float64).reshape(-1, 3)


def same_view(a: Optional[Pose], b: Optional[Pose], pos_tol_m: float = 0.01, ang_tol_deg: float = 1.0) -> bool:
    """True when two camera poses are within ``pos_tol_m`` and ``ang_tol_deg`` of each other."""
    if a is None or b is None:
        return False
    # camera centre in world = -R^T t
    ca, cb = -a.R.T @ a.t, -b.R.T @ b.t
    if float(np.linalg.norm(ca - cb)) > pos_tol_m:
        return False
    cos_ang = (float(np.trace(a.R @ b.R.T)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_ang)))) <= ang_tol_deg


def scene_state(data: mujoco.MjData) -> np.ndarray:
    """Everything in qpos that can change the picture besides the camera pose: the gantry, arm and
    gripper ([7:14]), legs ([14:32]) and every crop free joint ([32:]). The trunk free joint
    ([0:7]) is left out because ``same_view`` already covers the camera pose."""
    return np.array(data.qpos[7:], dtype=np.float64)


def same_scene(a: Optional[np.ndarray], b: Optional[np.ndarray], tol: float = 0.005) -> bool:
    """True when two ``scene_state`` vectors differ by at most ``tol`` (rad or m) in every entry."""
    if a is None or b is None or a.shape != b.shape:
        return False
    return bool(np.max(np.abs(a - b), initial=0.0) <= tol)


def set_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos) -> bool:
    """Copy a /nandi/sim_qpos vector in and update kinematics + cameras. False on nq mismatch."""
    q = np.asarray(qpos, dtype=np.float64)
    if q.shape[0] != model.nq:
        return False
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)
    return True


def current_world_id() -> str:
    """world_id of the world this process's perception model was composed from (computed if none built yet)."""
    return _WORLD_ID if _WORLD_ID is not None else farm_scene.world_identity()[1]


def set_state(model: mujoco.MjModel, data: mujoco.MjData, state, world_id: Optional[str] = "") -> Tuple[bool, str]:
    """Apply a /nandi/sim_state message (dict or JSON text) and update kinematics + cameras (WP0, MVP-SPEC 3.2).

    Refuses (returns ``(False, reason)``, writes nothing) when the state's ``world_id`` differs from this world's
    (``world_id=""``, the default, means ``current_world_id()``; ``None`` skips the check, for tests that built
    the model themselves), or on a size or provider mismatch. Otherwise writes qpos, mocap poses, ``eq_active``
    and every registered provider's model mutations (``controllers.sim_state``; providers from
    ``sim_state.discover_providers``), THEN runs ``mj_kinematics``. A malformed message is refused, never raised.
    ``set_qpos`` stays for the legacy world."""
    try:
        st = sim_state_mod.decode(state) if isinstance(state, (str, bytes)) else state
    except (ValueError, sim_state_mod.SimStateError) as exc:
        return False, f"undecodable sim_state: {exc}"
    sim_state_mod.discover_providers()  # once per process: the same provider set as the physics node
    expected = current_world_id() if world_id == "" else world_id
    ok, reason = sim_state_mod.apply(model, data, st, expected)  # never raises on bad input
    if not ok:
        return False, reason
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)
    return True, "ok"


class CameraRenderer:
    """Offscreen RGB + depth + segmentation for one camera (GL backend from MUJOCO_GL: glfw/egl/osmesa/cgl).

    Create it in the thread that will render (GL contexts are thread-bound). ``cull_dist`` (metres) enables the
    WP5 mjvScene camera-distance post-filter (``scene_cull.SceneCuller``, OS 3.7) after every scene update; the
    model is never mutated. ``last_cull`` is ``(ngeom_before, ngeom_after)`` of the newest update.
    """

    def __init__(self, model: mujoco.MjModel, camera: str, width: int, height: int, fast: bool = True,
                 cull_dist: Optional[float] = None):
        self.camera = camera
        self.model = model
        self.cid = camera_id(model, camera)
        self.width, self.height = int(width), int(height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        if fast:
            # Shadows and reflections dominate llvmpipe frame time and add nothing for detection.
            flags = self.renderer.scene.flags
            flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.opt = mujoco.MjvOption()
        for g, on in enumerate(RENDER_GROUPS):
            self.opt.geomgroup[g] = on
        self.culler = None
        self.last_cull: Tuple[int, int] = (0, 0)
        if cull_dist is not None and cull_dist > 0:
            from atharv_nandi.core.scene_cull import SceneCuller
            self.culler = SceneCuller(model, self.renderer.scene, float(cull_dist))

    def use(self, camera: str) -> "CameraRenderer":
        """Point this renderer at another camera of the same model (one renderer per resolution, MVP-SPEC 3.7)."""
        self.cid = camera_id(self.model, camera)
        self.camera = camera
        return self

    def _sync_fog(self):
        from atharv_nandi.sim.world import lighting      # orchard fog (WP4): no-op in models without a ``sun`` light
        lighting.sync_renderer(self.renderer)

    def _update(self, data: mujoco.MjData):
        self.renderer.update_scene(data, camera=self.cid, scene_option=self.opt)
        if self.culler is not None:
            self.culler.scene = self.renderer.scene
            self.last_cull = self.culler.cull(data.cam_xpos[self.cid])
        else:
            n = self.renderer.scene.ngeom
            self.last_cull = (n, n)

    def rgb(self, data: mujoco.MjData) -> np.ndarray:
        self._sync_fog()
        self._update(data)
        return self.renderer.render().copy()

    def depth(self, data: mujoco.MjData) -> np.ndarray:
        """Metric depth along the optical axis (z), shape (H, W)."""
        self.renderer.enable_depth_rendering()
        try:
            self._update(data)
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()

    def seg(self, data: mujoco.MjData) -> np.ndarray:
        """Segmentation pass: (H, W, 2) int32 of (objid, objtype); background -1."""
        self.renderer.enable_segmentation_rendering()
        try:
            self._update(data)
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()

    def capture(self, data: mujoco.MjData, with_seg: bool = True) -> dict:
        """One TREE 360 / privacy capture: RGB + metric depth (one GL pass, two readbacks) and a segmentation pass.
        Returns ``{rgb, depth, seg, ms: {rgbd, seg, total}, ngeom: (before, after cull)}``."""
        import time as _t
        t0 = _t.perf_counter()
        rgb, depth = self.rgbd(data)
        t1 = _t.perf_counter()
        seg = self.seg(data) if with_seg else None
        t2 = _t.perf_counter()
        return {"rgb": rgb, "depth": depth, "seg": seg, "ngeom": self.last_cull,
                "ms": {"rgbd": 1e3 * (t1 - t0), "seg": 1e3 * (t2 - t1), "total": 1e3 * (t2 - t0)}}

    def rgbd(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """RGB (H, W, 3) uint8 and metric z-depth (H, W) from ONE GL pass (half the cost of rgb+depth).

        Uses the Renderer's GL context directly (mujoco 3.3 attribute names); falls back to two
        passes if those internals change.
        """
        r = self.renderer
        ctx, rect = getattr(r, "_mjr_context", None), getattr(r, "_rect", None)
        if ctx is None or rect is None:
            return self.rgb(data), self.depth(data)
        self._sync_fog()
        ctx = r._mjr_context            # sync may rebuild the context on a fog change
        self._update(data)
        if getattr(r, "_gl_context", None) is not None:
            r._gl_context.make_current()
        h, w = rect.height, rect.width
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        zbuf = np.empty((h, w), dtype=np.float32)
        mujoco.mjr_render(rect, r.scene, ctx)
        mujoco.mjr_readPixels(rgb, zbuf, rect, ctx)
        return np.flipud(rgb).copy(), np.flipud(self._linear_depth(zbuf, r.model)).copy()

    @staticmethod
    def _linear_depth(zbuf: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        # Same conversion as mujoco.Renderer (3.3, reversed-Z): window depth -> eye-space z in metres.
        extent = model.stat.extent
        zfar = np.float32(model.vis.map.zfar * extent)
        znear = np.float32(model.vis.map.znear * extent)
        c_coef = -(zfar + znear) / (zfar - znear)
        d_coef = -(np.float32(2) * zfar * znear) / (zfar - znear)
        c_coef = np.float32(-0.5) * c_coef - np.float32(0.5)
        d_coef = np.float32(-0.5) * d_coef
        return (d_coef / (zbuf.astype(np.float64) + c_coef)).astype(np.float32)

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass


def footprint_mm_per_px(fovy_deg: float, width: int, height: int, dist_m: float) -> float:
    """Horizontal ground-sample distance (mm/px) of a pinhole camera at ``dist_m`` (OS 2.5 footprint table)."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    hfov = 2.0 * math.atan((width / 2.0) / f)
    return 1e3 * 2.0 * dist_m * math.tan(hfov / 2.0) / width


def ring_facing(model: mujoco.MjModel, data: mujoco.MjData, target_xy) -> str:
    """The ring camera whose optical axis points closest to ``target_xy`` (the TREE 360 low band camera)."""
    best, best_d = RING_CAMS[0], math.inf
    for n in RING_CAMS:
        c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)
        if c < 0:
            continue
        fwd = -data.cam_xmat[c].reshape(3, 3)[:, 2]
        v = np.asarray(target_xy, dtype=float)[:2] - data.cam_xpos[c][:2]
        ang = abs(math.atan2(fwd[0] * v[1] - fwd[1] * v[0], fwd[0] * v[0] + fwd[1] * v[1]))
        if ang < best_d:
            best, best_d = n, ang
    return best


class RenderBudget:
    """Token bucket that caps the share of wall time spent in GL rendering.

    Tokens (seconds of render time) accrue at ``frac`` per wall second up to ``burst_s``. A render
    may start while the balance is positive; its measured duration is then charged. When the
    balance is negative the caller falls back to geometry-only detection, so a slow software GL
    (llvmpipe) degrades the camera instead of stealing CPU from physics.
    """

    def __init__(self, frac: float, burst_s: float = 0.3):
        self.frac = max(0.0, float(frac))
        self.burst_s = float(burst_s)
        self.tokens = self.burst_s
        self._last: Optional[float] = None
        self.spent_s = 0.0
        self.renders = 0
        self.skips = 0

    def _refill(self, now: float):
        if self._last is not None:
            self.tokens = min(self.burst_s, self.tokens + self.frac * max(0.0, now - self._last))
        self._last = now

    def allow(self, now: float) -> bool:
        self._refill(now)
        ok = self.frac > 0.0 and self.tokens > 0.0
        if not ok:
            self.skips += 1
        return ok

    def charge(self, seconds: float):
        self.tokens -= float(seconds)
        self.spent_s += float(seconds)
        self.renders += 1
