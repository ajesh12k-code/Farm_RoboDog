#!/usr/bin/env python3
"""Atharv Nandi perception node: cameras, ring pano, TREE 360 captures, privacy filter, detections (WP5).

Subscribes  /nandi/sim_state                 (String JSON, WP0; preferred: qpos, mocap, eq_active, mutations incl.
                                              the WP24 mast extension and crown gimbal)
            /nandi/sim_qpos                  (Float64MultiArray, legacy world only)
            /nandi/lidar/clusters            (String JSON from atharv_nandi/sim/hooks/lidar.py: non-map LiDAR clusters, 10 Hz)
            /nandi/perception/mode           (String JSON {"mode": drive|grip|tree360|paused, "task_state"?})
            /nandi/perception/tree360        (String JSON: {"cmd": "start", "tree_id", "cameras": [...]} |
                                              {"cmd": "capture", "camera", "band", "seq", "tree_id"} | {"cmd": "stop"})
            /nandi/perception/zone_pass      (String JSON {"cmd": "start"|"end", "zone_id", "plant_ids"?})
Publishes   /nandi/crop_detections           (String JSON, <= 5 Hz; contract in README 'Perception')
            /nandi/camera/<cam>/compressed   (sensor_msgs/CompressedImage JPEG, privacy-filtered; head, wrist,
                                              ring_0..3, crown)
            /nandi/camera/pano/compressed    (CompressedImage, <= 1 Hz) + /nandi/camera/pano/info (sector ages JSON)
            /nandi/perception/capture_ack    (String JSON per TREE 360 capture)
            /nandi/perception/state          (String JSON 1 Hz: scheduler mode, render fraction, privacy counts)
            /nandi/records/inspection        (String JSON nandi.inspect/1 per plant at the end of a zone pass)
            /nandi/assist                    (String JSON when the dusk rule falls back to the geometry detector)

Everything except ``main`` is ROS-free: ``PerceptionCore`` holds the never-stepped perception model
(``atharv_nandi/core/camera_rig.py``), one renderer per resolution, the render scheduler, the privacy filter, the ring
self-masks, the pano LUT, ``ColorBlob`` and the "seen" tracker, and returns its outputs as events, so the tests
drive it without ROS. **Every image leaves through ``privacy_filter.FilteredSink``**: a raw array cannot reach a
publisher (MVP-SPEC 5.4).

Detectors (MVP-SPEC 5.4): ``sim_truth_noise`` (``atharv_nandi/core/detector_model.py``) on the head camera, and
``ColorBlob`` (chromaticity threshold on the RGB already rendered, no extra GL pass). The run card names the one
that ran (``detector`` in every message). Dusk / dawn (``world.lighting``): camera noise and a horizontal motion
blur are applied to every rendered frame (``degrade``); when the ColorBlob recall (against the projected truth,
the assisted check) drops below 0.8, the node falls back to the geometry detector and raises an assist request.

GL backend: MUJOCO_GL (glfw on the container's Xvfb display, egl/osmesa elsewhere, cgl on macOS).
"""
import json
import math
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

# llvmpipe spawns one render thread per core by default; one is enough for 160x120 and keeps the
# camera off the cores the physics loop needs. Must be set before the GL context exists.
os.environ.setdefault("LP_NUM_THREADS", "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from atharv_nandi import node_params  # noqa: E402
from atharv_nandi import paths
from atharv_nandi.core import camera_rig, pano, privacy_filter, render_scheduler, self_mask  # noqa: E402
from atharv_nandi.core.detector_model import (  # noqa: E402
    CONFIG_PATH, DetectorModel, build_message, config_sha256, encode_message, load_config,
)

MAX_DETECTION_RATE_HZ = 5.0
INSPECT_SCHEMA = "nandi.inspect/1"
CAMERA_TOPICS = {   # camera -> privacy-filtered JPEG topic (the WP5 contract notes)
    "head_cam": "/nandi/camera/head/compressed",
    "wrist_cam": "/nandi/camera/wrist/compressed",
    "crown_cam": "/nandi/camera/crown/compressed",
    "ring_0": "/nandi/camera/ring_0/compressed",
    "ring_1": "/nandi/camera/ring_1/compressed",
    "ring_2": "/nandi/camera/ring_2/compressed",
    "ring_3": "/nandi/camera/ring_3/compressed",
}


# ============================================================================== ColorBlob + dusk + seen (ROS-free)
def degrade(rgb: np.ndarray, sigma: float, blur_px: int, rng: np.random.Generator) -> np.ndarray:
    """Dusk stub (MVP-SPEC 4.2): Gaussian noise ``sigma`` (fraction of 255) and a ``blur_px`` horizontal box blur."""
    img = rgb.astype(np.float32)
    k = int(blur_px)
    if k > 1:
        pad = np.pad(img, ((0, 0), (k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
        c = np.cumsum(pad, axis=1, dtype=np.float64)
        c = np.concatenate([np.zeros_like(c[:, :1]), c], axis=1)
        img = ((c[:, k:] - c[:, :-k]) / k).astype(np.float32)
    if sigma > 0:
        img = img + rng.normal(0.0, 255.0 * sigma, img.shape).astype(np.float32)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-connected component labels (0 = background) by union-find over runs; pure numpy + Python."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    parent = [0]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    nxt = 1
    for y in range(H):
        row = mask[y]
        if not row.any():
            continue
        xs = np.nonzero(row)[0]
        breaks = np.nonzero(np.diff(xs) > 1)[0]
        starts = np.concatenate([[xs[0]], xs[breaks + 1]])
        ends = np.concatenate([xs[breaks], [xs[-1]]])
        for a, b in zip(starts, ends):
            above = labels[y - 1, a:b + 1] if y > 0 else np.zeros(0, dtype=np.int32)
            ids = np.unique(above[above > 0])
            if ids.size == 0:
                parent.append(nxt)
                lab = nxt
                nxt += 1
            else:
                roots = sorted({find(int(i)) for i in ids})
                lab = roots[0]
                for r in roots[1:]:
                    parent[r] = lab
            labels[y, a:b + 1] = lab
    if nxt == 1:
        return labels, 0
    lut = np.array([find(i) for i in range(nxt)], dtype=np.int32)
    _, dense = np.unique(lut, return_inverse=True)
    return dense[labels].astype(np.int32), int(dense.max())


class ColorBlob:
    """Red-crop blob detector on RGB (no extra GL pass): chromaticity r/(r+g+b) >= ``chroma_min``."""

    name = "colorblob"

    def __init__(self, cfg: Optional[dict] = None):
        c = dict(cfg or {})
        self.min_area = int(c.get("min_area_px", 6))
        self.chroma_min = float(c.get("chroma_min", 0.45))
        self.v_min = float(c.get("v_min", 0.03))

    def mask(self, rgb: np.ndarray, exclude: Optional[np.ndarray] = None) -> np.ndarray:
        f = rgb[..., :3].astype(np.float32) / 255.0
        s = f.sum(axis=-1) + 1e-6
        m = (f[..., 0] / s >= self.chroma_min) & (f.max(axis=-1) >= self.v_min)
        if exclude is not None:
            m &= ~exclude
        return m

    def detect(self, rgb: np.ndarray, depth: Optional[np.ndarray] = None,
               exclude: Optional[np.ndarray] = None) -> List[dict]:
        lab, n = label_components(self.mask(rgb, exclude))
        out = []
        for k in range(1, n + 1):
            ys, xs = np.nonzero(lab == k)
            if len(xs) < self.min_area:
                continue
            det = {"u": float(xs.mean()), "v": float(ys.mean()), "area_px": int(len(xs)),
                   "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "class": "ripe"}
            if depth is not None:
                z = depth[ys, xs]
                z = z[np.isfinite(z)]
                det["z"] = float(np.median(z)) if len(z) else None
            out.append(det)
        return out


def match_recall(dets: Sequence[dict], truth_uv: np.ndarray, gate_px: float = 6.0) -> Tuple[float, int]:
    """Share of visible truth points with a detection centroid within ``gate_px`` (greedy, one-to-one)."""
    n = len(truth_uv)
    if n == 0:
        return 1.0, 0
    used, hit = set(), 0
    for t in truth_uv:
        best, bd = None, gate_px
        for i, d in enumerate(dets):
            if i in used:
                continue
            dd = math.hypot(d["u"] - t[0], d["v"] - t[1])
            if dd <= bd:
                best, bd = i, dd
        if best is not None:
            used.add(best)
            hit += 1
    return hit / n, n


class DetectorSelector:
    """Dusk rule (MVP-SPEC 4.2): ColorBlob while its rolling recall >= ``recall_min``, else the geometry detector
    plus one assist request (per fallback episode)."""

    def __init__(self, recall_min: float = 0.8, window: int = 10, min_truth: int = 5):
        self.recall_min = float(recall_min)
        self.window = deque(maxlen=int(window))
        self.min_truth = int(min_truth)
        self.active = ColorBlob.name
        self.assist: Optional[dict] = None
        self.episodes = 0

    def update(self, recall: float, n_truth: int, t: float = 0.0) -> str:
        if n_truth > 0:
            self.window.append((recall * n_truth, n_truth))
        hits = sum(h for h, _ in self.window)
        tot = sum(n for _, n in self.window)
        if tot >= self.min_truth:
            r = hits / tot
            if r < self.recall_min and self.active == ColorBlob.name:
                self.active = "geometry"
                self.episodes += 1
                self.assist = {"kind": "assist_request", "reason": "colorblob_recall_low", "recall": round(r, 3),
                               "fallback": "geometry", "t": round(float(t), 3)}
            elif r >= self.recall_min and self.active != ColorBlob.name:
                self.active = ColorBlob.name
        return self.active

    def take_assist(self) -> Optional[dict]:
        a, self.assist = self.assist, None
        return a


class SeenTracker:
    """MVP-SPEC 2.4 "seen": a plant counts as seen when it had a >= 12 px bbox, not self-masked, in at least one
    ring or head frame within the last ``window_s`` of its zone pass."""

    CAMERAS = ("head_cam",) + camera_rig.RING_CAMS

    def __init__(self, min_bbox_px: float = 12.0, window_s: float = 10.0):
        self.min_px, self.window_s = float(min_bbox_px), float(window_s)
        self.obs: Dict[str, List[Tuple[float, float, str]]] = {}    # plant -> [(t, bbox_px, camera)]

    def observe(self, plant_id: str, t: float, bbox_px: float, camera: str, self_masked: bool,
                privacy_masked: bool = False):
        if camera not in self.CAMERAS or self_masked or privacy_masked:
            return
        self.obs.setdefault(plant_id, []).append((float(t), float(bbox_px), camera))

    def seen(self, plant_id: str, t_end: float) -> Tuple[bool, float, Optional[float]]:
        """(seen, best_px within the window, last_seen_t)."""
        rows = [r for r in self.obs.get(plant_id, ()) if t_end - self.window_s <= r[0] <= t_end]
        good = [r for r in rows if r[1] >= self.min_px]
        best = max((r[1] for r in rows), default=0.0)
        last = max((r[0] for r in good), default=None)
        return bool(good), best, last

    def reset(self):
        self.obs.clear()


def inspection_record(plant_id: str, zone_id: str, seen: Tuple[bool, float, Optional[float]], detector: str,
                      run_id: str = "sim", device_id: str = "nandi-sim", geo: Optional[dict] = None,
                      counts_by_stage: Optional[dict] = None) -> dict:
    """``nandi.inspect/1`` (MVP-SPEC 5.9). No image, no personal field."""
    s, best, last = seen
    return {"schema": INSPECT_SCHEMA, "run_id": run_id, "device_id": device_id, "tenant": "sim-nandi",
            "data_provenance": "simulated", "zone_id": zone_id, "plant_id": plant_id, "geo": geo or {},
            "seen": bool(s), "best_px": round(float(best), 1), "last_seen_t": None if last is None else round(last, 3),
            "counts_by_stage": counts_by_stage or {}, "detector": detector}


# ============================================================================== perception core (ROS-free)
def field_geofence(world: Optional[str] = None) -> Optional[list]:
    """The parcel polygon of the composed world (field JSON ``geofence``), or None."""
    w = world or os.environ.get("NANDI_WORLD", "legacy")
    path = paths.config_path("fields", f"{w}.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            fence = json.load(fh).get("geofence")
    except (OSError, ValueError):
        return None
    return fence if fence and len(fence) >= 3 else None


class PerceptionCore:
    """The perception pipeline without ROS. ``step(now)`` renders what the scheduler allows and returns events."""

    def __init__(self, cfg: Optional[dict] = None, render: bool = True, model: Optional[mujoco.MjModel] = None,
                 geofence: Optional[list] = None, seed: int = 7):
        self.cfg = cfg if cfg is not None else load_config()
        c = self.cfg
        cams, node = c.get("cameras", {}), c.get("node", {})
        self.model = model if model is not None else camera_rig.build_perception_model(c)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0) if self.model.nkey else mujoco.mj_resetData(
            self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.rng = np.random.default_rng(seed)
        self.head = str(c["camera"].get("name", camera_rig.HEAD_CAM))
        self.sizes = {self.head: (int(c["camera"]["width"]), int(c["camera"]["height"])),
                      camera_rig.WRIST_CAM: (int(cams.get("wrist_width", 160)), int(cams.get("wrist_height", 120))),
                      camera_rig.CROWN_CAM: (int(cams.get("crown_width", 320)), int(cams.get("crown_height", 240)))}
        for n in camera_rig.RING_CAMS:
            self.sizes[n] = (int(cams.get("ring_width", 160)), int(cams.get("ring_height", 120)))
        self.capture_size = (int(cams.get("capture_width", 320)), int(cams.get("capture_height", 240)))
        self.seg_div = max(1, int(cams.get("capture_seg_div", 2)))
        self.cameras = [n for n in self.sizes if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, n) >= 0]
        self.cull_dist = float(cams.get("cull_dist_m", 4.0))
        self.scheduler = render_scheduler.default_scheduler(c.get("scheduler"), self.cameras)
        pc = c.get("privacy", {})
        fence = geofence if geofence is not None else (field_geofence() if pc.get("parcel_mask", True) else None)
        self.privacy = privacy_filter.PrivacyFilter(str(pc.get("mode", "assisted")), int(pc.get("dilate_px", 4)),
                                                    float(pc.get("expand_frac", 0.2)), float(pc.get("near_m", 0.3)),
                                                    fence)
        self.robot = self_mask.robot_geoms(self.model)
        self.self_masks = self_mask.StaticSelfMask(int(c.get("self_mask", {}).get("dilate_px", 2)))
        self.detector = DetectorModel(c["detector"])
        self.intr = camera_rig.intrinsics(self.model, self.head, *self.sizes[self.head])
        self.colorblob = ColorBlob(c.get("colorblob"))
        self.selector = DetectorSelector(float(c.get("colorblob", {}).get("recall_min", 0.8)))
        sc = c.get("seen", {})
        self.seen = SeenTracker(float(sc.get("min_bbox_px", 12)), float(sc.get("window_s", 10.0)))
        pcfg = c.get("pano", {})
        self.pano_enabled = bool(pcfg.get("enabled", True)) and all(n in self.cameras for n in camera_rig.RING_CAMS)
        self.pano_period = 1.0 / max(1e-3, float(pcfg.get("rate_hz", 1.0)))
        self.pano_quality = int(pcfg.get("jpeg_quality", 70))
        self.lut = None
        if self.pano_enabled:
            rw, rh = self.sizes[camera_rig.RING_CAMS[0]]
            ring = pano.ring_from_model(self.model, self.data, rw, rh)
            self.lut = pano.PanoLUT(ring, int(pcfg.get("width", 640)), int(pcfg.get("height", 256)),
                                    float(pcfg.get("el_min_deg", pano.EL_MIN_DEG)),
                                    float(pcfg.get("el_max_deg", pano.EL_MAX_DEG)),
                                    float(pcfg.get("feather_deg", pano.FEATHER_DEG)))
        self.ages = pano.SectorAges()
        self._next_pano = 0.0
        self.latest: Dict[str, privacy_filter.FilteredFrame] = {}
        self.clusters: List[dict] = []
        self.clusters_t: Optional[float] = None      # sim time of the newest cluster message (payload "t")
        self.clusters_rx: Optional[float] = None     # node clock when it arrived
        self.clusters_max_age = float(pc.get("clusters_max_age_s", 0.35))
        self.state_t: Optional[float] = None         # sim time of the newest applied /nandi/sim_state
        cb = c.get("colorblob", {})
        self.colorblob_period = 1.0 / max(1e-3, float(cb.get("check_rate_hz", 1.0)))
        self._next_colorblob = 0.0
        self.arm = self_mask.arm_geoms(self.model)
        self.lighting = {"camera_noise_sigma": 0.0, "motion_blur_px": 0}
        self.renderers: Dict[Tuple[int, int], camera_rig.CameraRenderer] = {}
        self.render_ok = False
        self.render_error = ""
        self.tree360: Optional[dict] = None
        self._depth = self._depth_pose = self._depth_scene = None
        self.state_ok = False
        if render and bool(node.get("render", True)):
            try:
                self._renderer(self.head)
                self.render_ok = True
                self._build_self_masks()
                if camera_rig.CROWN_CAM in self.cameras:     # warm the TREE 360 capture renderer (first pass is slow)
                    self._renderer(camera_rig.CROWN_CAM, self.capture_size).rgbd(self.data)
            except Exception as exc:  # no GL backend: geometry only
                self.render_ok = False
                self.render_error = str(exc)

    # ------------------------------------------------------------------ renderers
    def _renderer(self, camera: str, size: Optional[Tuple[int, int]] = None) -> camera_rig.CameraRenderer:
        w, h = size or self.sizes[camera]
        r = self.renderers.get((w, h))
        if r is None:
            r = camera_rig.CameraRenderer(self.model, camera, w, h, cull_dist=self.cull_dist or None)
            self.renderers[(w, h)] = r
        return r.use(camera)

    def _build_self_masks(self):
        for n in camera_rig.RING_CAMS:
            if n in self.cameras:
                self.self_masks.build(n, [self._renderer(n).seg(self.data)], self.robot)

    def close(self):
        for r in self.renderers.values():
            r.close()
        self.renderers.clear()

    # ------------------------------------------------------------------ inputs
    def set_state(self, msg) -> Tuple[bool, str]:
        ok, why = camera_rig.set_state(self.model, self.data, msg)
        self.state_ok = self.state_ok or ok
        if ok:
            self.state_t = float(self.data.time)
        return ok, why

    def set_qpos(self, qpos) -> bool:
        ok = camera_rig.set_qpos(self.model, self.data, qpos)
        self.state_ok = self.state_ok or ok
        return ok

    def set_clusters(self, payload: Optional[dict], now: Optional[float] = None):
        """Newest ``/nandi/lidar/clusters``. Malformed entries are dropped from the list but the message still
        counts as fresh only when it parses as a whole (``t`` a number, ``clusters`` a list)."""
        p = payload if isinstance(payload, dict) else {}
        cl = p.get("clusters")
        t = p.get("t")
        if not isinstance(cl, list) or isinstance(t, bool) or not isinstance(t, (int, float)):
            return
        good = [c for c in cl if isinstance(c, dict) and len(c.get("lo") or ()) == 3 and len(c.get("hi") or ()) == 3]
        self.clusters = good
        self.clusters_t = float(t)
        self.clusters_rx = None if now is None else float(now)

    def clusters_fresh(self, now: float) -> Tuple[bool, str]:
        """Are the clusters from (about) the scene being rendered? Against the sim_state time when both carry sim
        time, else by arrival age on the node clock. Never received = not fresh."""
        if self.clusters_t is None:
            return False, "clusters_missing"
        if self.state_t is not None:
            age = abs(self.state_t - self.clusters_t)
        elif self.clusters_rx is not None:
            age = float(now) - self.clusters_rx
        else:
            age = 0.0
        if age > self.clusters_max_age:
            return False, "clusters_stale"
        return True, ""

    def set_lighting(self, preset: Optional[str]):
        from atharv_nandi.sim.world import lighting
        try:
            st = lighting.state(preset) if preset else {"camera_noise_sigma": 0.0, "motion_blur_px": 0}
        except KeyError:
            return
        self.lighting = {"camera_noise_sigma": float(st["camera_noise_sigma"]),
                         "motion_blur_px": int(st["motion_blur_px"])}

    def set_mode(self, mode: str, now: float, reason: str = "", cameras: Optional[Sequence[str]] = None):
        self.scheduler.set_mode(mode, now, reason, cameras)

    def tree360_cmd(self, msg: dict, now: float) -> List[dict]:
        cmd = msg.get("cmd")
        if cmd == "start":
            mode = self.scheduler.mode
            if mode in ("paused", "grip"):        # e-stop / pause, or the arm is gripping: not now
                return [{"kind": "state", "tree360_refused": {"tree_id": msg.get("tree_id"), "reason": f"mode_{mode}"}}]
            cams = list(msg.get("cameras") or [camera_rig.CROWN_CAM])
            prev = self.tree360["prev_mode"] if self.tree360 is not None else mode
            self.tree360 = {"tree_id": msg.get("tree_id"), "cameras": cams, "t0": now, "captures": 0,
                            "prev_mode": prev}
            self.set_mode("tree360", now, f"tree360 {msg.get('tree_id')}", cams)
            return [{"kind": "state", "tree360": dict(self.tree360)}]
        if cmd == "capture":
            cam = str(msg.get("camera", camera_rig.CROWN_CAM))
            ok = self.scheduler.request(cam, {k: msg.get(k) for k in ("tree_id", "band", "seq")}, passes=3)
            if not ok:
                return [{"kind": "ack", "topic": "/nandi/perception/capture_ack",
                         "payload": {"ok": False, "reason": "camera_not_enabled", "camera": cam,
                                     "mode": self.scheduler.mode, **{k: msg.get(k) for k in ("tree_id", "band",
                                                                                            "seq")}}}]
            return []
        if cmd == "stop":
            frac = self.scheduler.fraction(now)
            info = dict(self.tree360 or {}, render_fraction=round(frac, 4))
            prev = (self.tree360 or {}).get("prev_mode", "drive")
            self.tree360 = None
            # only leave tree360 if we are still in it: a pause (e-stop) or GRIP that came in meanwhile stays
            if self.scheduler.mode == "tree360":
                self.set_mode(prev if prev not in ("tree360", "paused", "grip") else "drive", now, "tree360 done")
            return [{"kind": "state", "tree360_done": info}]
        return []

    # ------------------------------------------------------------------ one frame
    def _filtered(self, camera: str, stamp: float, rgb, depth, cam_pose, intr, meta=None):
        persons = privacy_filter.person_boxes(self.model, self.data) if self.privacy.mode == "assisted" else []
        fresh, why = self.clusters_fresh(stamp)
        if not fresh and self.privacy.mode == "physics":
            # physics mode has no truth mask: without current LiDAR clusters there is no person mask, so fail
            # closed (MVP-SPEC 5.4). Assisted mode keeps the truth mask and reports the gap in the stats.
            self.privacy.counts["frames"] += 1
            return self.privacy._drop(why, {"mode": self.privacy.mode})
        clusters = self.clusters if fresh else []
        far = 0.98 * float(self.model.vis.map.zfar) * float(self.model.stat.extent)   # background: no return
        res = self.privacy.apply(camera, stamp, rgb, cam_pose, intr, depth, persons, clusters, no_return_m=far,
                                 meta=meta)
        res.stats["clusters"] = "ok" if fresh else why
        return res

    def render_frame(self, camera: str, now: float, capture: bool = False,
                     tag: Optional[dict] = None) -> Tuple[Optional[privacy_filter.FilteredFrame], dict]:
        size = self.capture_size if capture else self.sizes[camera]
        r = self._renderer(camera, size)
        t0 = time.perf_counter()
        if capture:
            # RGB + depth at the capture size; the seg pass (privacy truth, crown self-mask) at half size on the
            # ring-size renderer (OS 3.7 measured a 160 x 120 seg pass), upsampled 2x nearest
            rgb, depth = r.rgbd(self.data)
            hs = (size[0] // self.seg_div, size[1] // self.seg_div)
            seg = self._renderer(camera, hs).seg(self.data)
            if self.seg_div > 1:
                seg = np.repeat(np.repeat(seg, self.seg_div, axis=0), self.seg_div, axis=1)[:size[1], :size[0]]
            r.use(camera)
        else:
            rgb, depth = r.rgbd(self.data)
            seg = None
        gl_s = time.perf_counter() - t0
        self.scheduler.charge(camera, gl_s)
        sigma, blur = self.lighting["camera_noise_sigma"], self.lighting["motion_blur_px"]
        if sigma > 0 or blur > 1:
            rgb = degrade(rgb, sigma, blur, self.rng)
        cid = camera_rig.camera_id(self.model, camera)
        pose = camera_rig.camera_pose(self.data, cid)
        intr = camera_rig.intrinsics(self.model, camera, *size)
        info: Dict[str, Any] = {"camera": camera, "gl_ms": round(1e3 * gl_s, 2), "ngeom": list(r.last_cull)}
        if seg is not None:
            from atharv_nandi.sim.world import actors
            person = np.isin(seg[..., 0], actors.person_geoms(self.model)) & (
                seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            robot_px = self_mask.seg_robot(seg, self.robot)
            info["person_px"] = int(person.sum())
            info["self_pct"] = round(100.0 * robot_px.sum() / robot_px.size, 3)
        res = self._filtered(camera, now, rgb, depth, pose, intr, meta=dict(tag or {}))
        info.update(res.stats)
        if res.frame is not None and camera in camera_rig.RING_CAMS and self.arm:
            # per-frame arm hull (MVP-SPEC 2.4): the arm entering a ring image is self-masked in this frame
            res.frame.meta["arm_mask"] = self_mask.arm_hull_mask(self.model, self.data, pose, intr, self.arm,
                                                                 self.self_masks.dilate_px)
        if res.frame is not None and seg is not None:
            m = res.frame.masked
            p = info.get("person_px", 0)
            info["person_covered_pct"] = 100.0 if p == 0 else round(100.0 * float(np.sum(m & person)) / p, 3)
            res.frame.meta["self_mask"] = self_mask.dilate(robot_px, 2)
        if res.frame is None:
            info["dropped"] = res.reason
        if camera == self.head and depth is not None:
            self._depth, self._depth_pose = depth, pose
            self._depth_scene = camera_rig.scene_state(self.data)
        return res.frame, info

    # ------------------------------------------------------------------ tick
    def step(self, now: float) -> List[dict]:
        events: List[dict] = []
        if not self.state_ok:
            return events
        if self.render_ok:
            for job in self.scheduler.due(now):
                capture = job.kind == "capture"
                frame, info = self.render_frame(job.camera, now, capture, job.tag)
                if capture:
                    if self.tree360 is not None:
                        self.tree360["captures"] += 1
                    events.append({"kind": "ack", "topic": "/nandi/perception/capture_ack",
                                   "payload": dict(info, ok=frame is not None, **(job.tag or {})),
                                   "frame": frame})
                elif frame is not None:
                    events.append({"kind": "image", "camera": job.camera, "frame": frame, "info": info})
                    if job.camera in camera_rig.RING_CAMS:
                        self.latest[job.camera] = frame
                        self.ages.update(job.camera, now)
                        self._observe_seen(job.camera, frame, now)
                    elif job.camera == self.head:
                        self._observe_seen(job.camera, frame, now)
                        if now >= self._next_colorblob:      # dusk rule (MVP-SPEC 4.2), rate-limited
                            self._next_colorblob = now + self.colorblob_period
                            events.append({"kind": "colorblob", "check": self.colorblob_check(frame, job.camera,
                                                                                               now)})
            if self.lut is not None and self.scheduler.mode == "drive" and now >= self._next_pano and self.latest:
                events.append(self.compose_pano(now))
                self._next_pano = now + self.pano_period
        return events

    def compose_pano(self, now: float) -> dict:
        frames = {n: f for n, f in self.latest.items()}
        extra = {n: self.self_masks.get(n, f.meta.get("arm_mask")) for n, f in frames.items()}
        frame = privacy_filter.compose_pano(self.lut, frames, now, extra, {"ages": self.ages.ages(now)})
        return {"kind": "pano", "frame": frame, "ages": self.ages.ages(now)}

    # ------------------------------------------------------------------ detections and "seen"
    def _observe_seen(self, camera: str, frame: privacy_filter.FilteredFrame, now: float):
        names, truth = camera_rig.crop_truth(self.model, self.data)
        if not names:
            return
        cid = camera_rig.camera_id(self.model, camera)
        pose = camera_rig.camera_pose(self.data, cid)
        w, h = frame.shape[1], frame.shape[0]
        intr = camera_rig.intrinsics(self.model, camera, w, h)
        uv, z = privacy_filter.project_points(truth, pose, intr)
        sm = self.self_masks.get(camera, frame.meta.get("arm_mask"))
        r_m = float(self.cfg["detector"].get("crop_radius_m", 0.035))
        for n, (u, v), zz in zip(names, uv, z):
            if not (zz > 0.05 and 0 <= u < w and 0 <= v < h):
                continue
            ui, vi = int(u), int(v)
            if frame.depth is not None:
                dz = frame.depth[vi, ui]
                if np.isfinite(dz) and dz < zz - r_m - 0.03:
                    continue          # occluded
            px = 2.0 * r_m * intr.fx / zz
            selfm = bool(sm[vi, ui]) if sm is not None else False
            self.seen.observe(n, now, px, camera, selfm, bool(frame.masked[vi, ui]))

    def detections(self, now: float) -> Tuple[dict, Optional[dict]]:
        """The ``/nandi/crop_detections`` message (head camera, ``sim_truth_noise``) and an assist request."""
        cid = camera_rig.camera_id(self.model, self.head)
        cam = camera_rig.camera_pose(self.data, cid)
        depth = None
        if (self._depth is not None and camera_rig.same_view(cam, self._depth_pose)
                and camera_rig.same_scene(camera_rig.scene_state(self.data), self._depth_scene)):
            depth = self._depth
        names, truth = camera_rig.crop_truth(self.model, self.data)
        trunk = camera_rig.body_pose(self.model, self.data)
        dets = self.detector.detect(truth, names, cam, trunk, self.intr, depth)
        msg = build_message(now, self.head, str(self.cfg["detector"]["name"]), config_sha256(), dets)
        return msg, self.selector.take_assist()

    def colorblob_check(self, frame: privacy_filter.FilteredFrame, camera: str, now: float) -> dict:
        """ColorBlob on a filtered frame + recall against the projected visible truth (assisted check)."""
        dets = self.colorblob.detect(np.asarray(frame.rgb), None if frame.depth is None else np.asarray(frame.depth),
                                     exclude=np.asarray(frame.masked))
        names, truth = camera_rig.crop_truth(self.model, self.data)
        cid = camera_rig.camera_id(self.model, camera)
        pose = camera_rig.camera_pose(self.data, cid)
        intr = camera_rig.intrinsics(self.model, camera, frame.shape[1], frame.shape[0])
        uv, z = privacy_filter.project_points(truth, pose, intr) if len(truth) else (np.zeros((0, 2)), np.zeros(0))
        vis = []
        r_m = float(self.cfg["detector"].get("crop_radius_m", 0.035))
        for (u, v), zz in zip(uv, z):
            if zz > 0.05 and 2 <= u < intr.width - 2 and 2 <= v < intr.height - 2:
                dz = None if frame.depth is None else frame.depth[int(v), int(u)]
                if dz is None or not np.isfinite(dz) or dz >= zz - r_m - 0.03:
                    if 2.0 * r_m * intr.fx / zz >= 3.0:
                        vis.append((u, v))
        rec, n = match_recall(dets, np.asarray(vis).reshape(-1, 2))
        active = self.selector.update(rec, n, now)
        return {"recall": rec, "n_truth": n, "n_det": len(dets), "active": active}

    def inspection_records(self, zone_id: str, plant_ids: Sequence[str], t_end: float, run_id: str = "sim") -> \
            List[dict]:
        return [inspection_record(p, zone_id, self.seen.seen(p, t_end), self.selector.active, run_id)
                for p in plant_ids]

    def state(self, now: float) -> dict:
        return {"scheduler": self.scheduler.stats(now), "privacy": dict(self.privacy.counts),
                "render_ok": self.render_ok, "render_error": self.render_error[:200],
                "tree360": self.tree360, "detector_active": self.selector.active,
                "pano": self.lut is not None}


# ============================================================================== ROS node
def _jpeg(frame: privacy_filter.FilteredFrame, quality: int) -> bytes:
    return privacy_filter.require_filtered(frame).jpeg(quality)


def main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import Float64MultiArray, String

    class PerceptionNode(Node):
        def __init__(self):
            super().__init__("nandi_perception")
            node_params.declare(self, "nandi_perception")  # WPR: config/nandi.yaml + NANDI_* env (env wins)
            self.declare_parameter("config_path", CONFIG_PATH)
            path = self.get_parameter("config_path").get_parameter_value().string_value or CONFIG_PATH
            self.cfg = load_config(path)
            self.core = PerceptionCore(self.cfg)
            node_cfg = self.cfg["node"]
            self.det_period = 1.0 / min(MAX_DETECTION_RATE_HZ, float(node_cfg.get("detection_rate_hz", 5.0)))
            self.jpeg_quality = int(node_cfg.get("jpeg_quality", 70))
            if not self.core.render_ok:
                self.get_logger().warning(f"Offscreen rendering unavailable ({self.core.render_error}); "
                                          "geometry-only detections, no images")
            self.img_pubs, self.sinks = {}, {}
            for cam in self.core.cameras:
                topic = CAMERA_TOPICS.get(cam, f"/nandi/camera/{cam}/compressed")
                short = topic.split("/")[3]
                pub = self.create_publisher(CompressedImage, topic, 2)
                self.sinks[cam] = privacy_filter.FilteredSink(lambda f, p=pub, c=short: self._pub_img(p, f, c), cam)
            pano_pub = self.create_publisher(CompressedImage, "/nandi/camera/pano/compressed", 2)
            self.pano_sink = privacy_filter.FilteredSink(lambda f: self._pub_img(pano_pub, f, "pano",
                                                                                  self.core.pano_quality), "pano")
            self.pano_info = self.create_publisher(String, "/nandi/camera/pano/info", 2)
            self.det_pub = self.create_publisher(String, "/nandi/crop_detections", 10)
            self.ack_pub = self.create_publisher(String, "/nandi/perception/capture_ack", 10)
            self.state_pub = self.create_publisher(String, "/nandi/perception/state", 2)
            self.insp_pub = self.create_publisher(String, "/nandi/records/inspection", 10)
            self.assist_pub = self.create_publisher(String, "/nandi/assist", 10)
            self.create_subscription(String, "/nandi/sim_state", self.on_state, 10)
            self.create_subscription(Float64MultiArray, "/nandi/sim_qpos", self.on_qpos, 10)
            self.create_subscription(String, "/nandi/lidar/clusters", self.on_clusters, 10)
            self.create_subscription(String, "/nandi/perception/mode", self.on_mode, 10)
            self.create_subscription(String, "/nandi/perception/tree360", self.on_tree360, 10)
            self.create_subscription(String, "/nandi/perception/zone_pass", self.on_zone, 10)
            self.create_subscription(String, "/nandi/world/state", self.on_world, 10)
            self._pending_state = None
            self._have_state = False
            self.zones: Dict[str, dict] = {}
            tick = float(self.cfg.get("scheduler", {}).get("tick_hz", 10.0))
            self.create_timer(1.0 / max(1.0, tick), self.tick)
            self.create_timer(self.det_period, self.tick_detections)
            self.create_timer(1.0, self.tick_state)
            self.get_logger().info(
                f"Perception: cameras {self.core.cameras}, render={'on' if self.core.render_ok else 'off'}, "
                f"budget {self.core.scheduler.bucket.frac:.0%}, privacy {self.core.privacy.mode}, "
                f"parcel mask {'on' if self.core.privacy.parcel else 'off'}, config sha256 {config_sha256(path)[:12]}")

        def _now(self) -> float:
            return time.monotonic()

        def _pub_img(self, pub, frame, frame_id, quality=None):
            img = CompressedImage()
            img.header.stamp = self.get_clock().now().to_msg()
            img.header.frame_id = frame_id
            img.format = "jpeg"
            img.data = _jpeg(frame, quality or self.jpeg_quality)
            pub.publish(img)

        def on_state(self, msg):
            self._pending_state = msg.data
            self._have_state = True

        def on_qpos(self, msg):
            if not self._have_state:
                self.core.set_qpos(msg.data)

        def on_clusters(self, msg):
            try:
                self.core.set_clusters(json.loads(msg.data), self._now())
            except ValueError:
                pass

        def on_world(self, msg):
            try:
                st = json.loads(msg.data)
            except ValueError:
                return
            light = st.get("light") if isinstance(st, dict) else None
            if isinstance(light, dict):
                self.core.set_lighting(light.get("preset"))

        def on_mode(self, msg):
            try:
                st = json.loads(msg.data)
                mode = st.get("mode") or self.core.scheduler.mode_for_task(st.get("task_state"))
                self.core.set_mode(mode, self._now(), str(st.get("reason", "")))
            except (ValueError, AttributeError) as exc:
                self.get_logger().warning(f"bad /nandi/perception/mode: {exc}")

        def on_tree360(self, msg):
            try:
                for ev in self.core.tree360_cmd(json.loads(msg.data), self._now()):
                    self._emit(ev)
            except (ValueError, AttributeError) as exc:
                self.get_logger().warning(f"bad /nandi/perception/tree360: {exc}")

        def on_zone(self, msg):
            try:
                st = json.loads(msg.data)
            except ValueError:
                return
            if st.get("cmd") == "end":
                names, _ = camera_rig.crop_truth(self.core.model, self.core.data)
                for rec in self.core.inspection_records(str(st.get("zone_id", "")), st.get("plant_ids") or names,
                                                        self._now()):
                    self.insp_pub.publish(String(data=json.dumps(rec, separators=(",", ":"))))

        def _emit(self, ev):
            k = ev["kind"]
            if k == "image":
                self.sinks[ev["camera"]].publish(ev["frame"])
            elif k == "pano":
                self.pano_sink.publish(ev["frame"])
                self.pano_info.publish(String(data=json.dumps({"ages": ev["ages"]})))
            elif k == "ack":
                payload = {kk: vv for kk, vv in ev["payload"].items()}
                self.ack_pub.publish(String(data=json.dumps(payload, separators=(",", ":"), default=str)))
                if ev.get("frame") is not None and ev["payload"].get("camera") in self.sinks:
                    self.sinks[ev["payload"]["camera"]].publish(ev["frame"])
            elif k == "state":
                self.state_pub.publish(String(data=json.dumps(ev, default=str)))

        def tick(self):
            if self._pending_state is not None:
                ok, why = self.core.set_state(self._pending_state)
                self._pending_state = None
                if not ok:
                    self.get_logger().warning(f"sim_state refused: {why}", throttle_duration_sec=10.0)
            for ev in self.core.step(self._now()):
                self._emit(ev)

        def tick_detections(self):
            if not self.core.state_ok:
                return
            msg, assist = self.core.detections(self.get_clock().now().nanoseconds * 1e-9)
            self.det_pub.publish(String(data=encode_message(msg)))
            if assist:
                self.assist_pub.publish(String(data=json.dumps(assist)))

        def tick_state(self):
            self.state_pub.publish(String(data=json.dumps(self.core.state(self._now()), default=str)))

    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():  # a real error, not a publish racing a SIGTERM/SIGINT shutdown
            raise
    finally:
        node.core.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
