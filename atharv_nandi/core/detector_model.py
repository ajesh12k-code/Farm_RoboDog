"""Simulated crop detector for Atharv Nandi: ground truth + pinhole camera + noise (pure numpy).

This is the ``sim_truth_noise`` detector that feeds ``/nandi/crop_detections``. It stands in for a
real vision model (edge detector) with the error budget of a cheap RGB-D pipeline:

* pinhole projection of the true crop centres (camera optical frame: x right, y down, z forward),
* field-of-view cut (centre must project inside the image) and range cut (<= ``max_range_m``),
* occlusion by a depth-buffer test when a rendered depth map is supplied,
* Gaussian position noise, sigma = ``pos_sigma_base_m + pos_sigma_range_frac * range`` per axis,
* random misses (``miss_rate``) and Poisson false positives (``fp_rate_per_frame``), placed on
  random pixel rays no deeper than the ground plane ``fp_ground_z_m``,
* a confidence model that falls with range and distance from the image centre.

Every parameter lives in ``config/perception.yaml``; the RNG is seeded so runs are reproducible.
Ids: with ``oracle_ids: false`` (default) ``id_hint`` is a track id (``trk_<n>``) from a
nearest-neighbour tracker that associates in the world (odometry) frame, so ids survive turns
and consumers cannot cheat with body names.

No ROS and no MuJoCo imports here; ``camera_rig`` turns MuJoCo camera poses into the
world->optical transforms used below.
"""
import ast
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from atharv_nandi import paths

CONFIG_PATH = paths.config_path("perception.yaml")
MAX_MESSAGE_BYTES = 16 * 1024


# ----------------------------------------------------------------------------- config
def _parse_scalar(text: str):
    t = text.strip()
    if t == "" or t in ("null", "~", "None"):
        return None
    low = t.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        return [] if not inner else [_parse_scalar(p) for p in inner.split(",")]
    if (t[0] == t[-1]) and t[0] in "'\"" and len(t) >= 2:
        return t[1:-1]
    try:
        return ast.literal_eval(t) if any(c.isdigit() for c in t) else t
    except (ValueError, SyntaxError):
        return t


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def parse_simple_yaml(text: str) -> dict:
    """Parse the YAML subset used by perception.yaml (nested mappings, scalars, inline lists)."""
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, rest = line.strip().partition(":")
        if not sep:
            raise ValueError(f"perception.yaml: cannot parse line {raw!r}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest.strip() == "":
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(rest)
    return root


def load_config(path: Optional[str] = None) -> dict:
    """Load perception.yaml (PyYAML when installed, else the built-in subset parser)."""
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return parse_simple_yaml(text)


def config_sha256(path: Optional[str] = None) -> str:
    with open(path or CONFIG_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ----------------------------------------------------------------------------- camera maths
@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> Intrinsics:
    """MuJoCo convention: square pixels, ``fovy`` spans the full image height."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return Intrinsics(f, f, width / 2.0, height / 2.0, int(width), int(height))


def project(p_cam: np.ndarray, intr: Intrinsics):
    """Optical-frame points (N,3) -> pixel coords (N,2) and depth z (N,). z<=0 gives nan pixels."""
    p = np.atleast_2d(np.asarray(p_cam, dtype=np.float64))
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.cx + intr.fx * p[:, 0] / z
        v = intr.cy + intr.fy * p[:, 1] / z
    bad = z <= 0
    u[bad] = np.nan
    v[bad] = np.nan
    return np.stack([u, v], axis=1), z


def backproject(uv: np.ndarray, z: np.ndarray, intr: Intrinsics) -> np.ndarray:
    uv = np.atleast_2d(uv)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    x = (uv[:, 0] - intr.cx) / intr.fx * z
    y = (uv[:, 1] - intr.cy) / intr.fy * z
    return np.stack([x, y, z], axis=1)


@dataclass
class Pose:
    """Rigid transform world -> frame: p_frame = R @ p_world + t."""
    R: np.ndarray
    t: np.ndarray

    def apply(self, p_world: np.ndarray) -> np.ndarray:
        return np.atleast_2d(p_world) @ self.R.T + self.t

    def inverse_apply(self, p_frame: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(p_frame) - self.t) @ self.R

    @staticmethod
    def from_frame_in_world(pos: np.ndarray, rot_frame_to_world: np.ndarray) -> "Pose":
        """A frame at ``pos`` whose axes (columns of the matrix) are given in world coordinates."""
        R = np.asarray(rot_frame_to_world, dtype=np.float64).reshape(3, 3).T
        return Pose(R, -R @ np.asarray(pos, dtype=np.float64))


# ----------------------------------------------------------------------------- tracker
class NearestTracker:
    """Greedy nearest-neighbour tracker; ids are ``trk_<n>``.

    ``DetectorModel`` feeds it world-frame (odometry-frame) points, so tracks stay put while the
    robot turns or walks. Associating in the trunk frame would swap ids between crops on a turn.
    """

    def __init__(self, gate_m: float, max_age: int, smoothing: float = 0.5):
        self.gate_m = float(gate_m)
        self.smoothing = float(smoothing)  # EMA weight of the new measurement in the track state
        self.max_age = int(max_age)
        self._tracks: Dict[str, dict] = {}
        self._next = 1

    def reset(self):
        self._tracks.clear()
        self._next = 1

    def assign(self, points: Sequence[Sequence[float]]) -> List[str]:
        ids: List[Optional[str]] = [None] * len(points)
        pairs = []
        for i, p in enumerate(points):
            for tid, tr in self._tracks.items():
                d = float(np.linalg.norm(np.asarray(p) - tr["xyz"]))
                if d <= self.gate_m:
                    pairs.append((d, i, tid))
        used = set()
        for d, i, tid in sorted(pairs):
            if ids[i] is None and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i, p in enumerate(points):
            if ids[i] is None:
                ids[i] = f"trk_{self._next}"
                self._next += 1
            p = np.asarray(p, dtype=np.float64)
            old = self._tracks.get(ids[i])
            xyz = p if old is None else (1.0 - self.smoothing) * old["xyz"] + self.smoothing * p
            self._tracks[ids[i]] = {"xyz": xyz, "age": 0}
            used.add(ids[i])
        for tid in list(self._tracks):
            if tid not in used:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]
        return ids  # type: ignore[return-value]


# ----------------------------------------------------------------------------- detector
class DetectorModel:
    """Ground truth -> noisy detections. One instance per camera stream (it owns RNG + tracker)."""

    def __init__(self, cfg: dict, seed: Optional[int] = None):
        self.cfg = dict(cfg)
        self.rng = np.random.default_rng(self.cfg.get("seed", 0) if seed is None else seed)
        self.tracker = NearestTracker(self.cfg.get("track_gate_m", 0.25), self.cfg.get("track_max_age", 10))

    def sigma_at(self, range_m: float) -> float:
        return float(self.cfg["pos_sigma_base_m"]) + float(self.cfg["pos_sigma_range_frac"]) * float(range_m)

    def visible(self, p_cam: np.ndarray, intr: Intrinsics, depth: Optional[np.ndarray] = None):
        """Boolean mask of crop centres that pass the FOV, range and (optional) occlusion tests."""
        c = self.cfg
        uv, z = project(p_cam, intr)
        rng_m = np.linalg.norm(np.atleast_2d(p_cam), axis=1)
        m = float(c.get("fov_margin_px", 0.0))
        with np.errstate(invalid="ignore"):
            ok = (z > 0) & (uv[:, 0] >= m) & (uv[:, 0] < intr.width - m) \
                & (uv[:, 1] >= m) & (uv[:, 1] < intr.height - m)
        ok &= (rng_m >= float(c.get("min_range_m", 0.0))) & (rng_m <= float(c["max_range_m"]))
        if depth is not None:
            r = float(c.get("crop_radius_m", 0.035)) + float(c.get("occlusion_margin_m", 0.03))
            for i in np.flatnonzero(ok):
                col = min(int(uv[i, 0]), depth.shape[1] - 1)
                row = min(int(uv[i, 1]), depth.shape[0] - 1)
                if float(depth[row, col]) < z[i] - r:
                    ok[i] = False
        return ok, uv, z, rng_m

    def _confidence(self, rng_m: float, u: float, v: float, intr: Intrinsics) -> float:
        c = self.cfg
        edge = max(abs(u - intr.cx) / intr.cx, abs(v - intr.cy) / intr.cy)
        conf = float(c["conf_base"]) - float(c["conf_range_slope"]) * rng_m \
            - float(c["conf_edge_penalty"]) * min(1.0, edge) + self.rng.normal(0.0, float(c["conf_sigma"]))
        return float(np.clip(conf, float(c["conf_min"]), 1.0))

    def _bbox(self, u: float, v: float, z: float, intr: Intrinsics) -> List[int]:
        half = intr.fy * float(self.cfg.get("crop_radius_m", 0.035)) / max(z, 1e-6)
        n = self.rng.normal(0.0, float(self.cfg.get("bbox_sigma_px", 1.0)), size=4)
        u0, u1 = sorted((u - half + n[0], u + half + n[2]))  # noise can cross tiny (far) boxes
        v0, v1 = sorted((v - half + n[1], v + half + n[3]))
        cu = lambda x: int(round(float(np.clip(x, 0, intr.width - 1))))  # noqa: E731
        cv = lambda y: int(round(float(np.clip(y, 0, intr.height - 1))))  # noqa: E731
        return [cu(u0), cv(v0), cu(u1), cv(v1)]

    def detect(self, truth_world: np.ndarray, names: Sequence[str], cam: Pose, trunk: Pose,
               intr: Intrinsics, depth: Optional[np.ndarray] = None) -> List[dict]:
        """One frame. ``cam``/``trunk`` map world points into the optical / trunk frame."""
        c = self.cfg
        truth_world = np.atleast_2d(np.asarray(truth_world, dtype=np.float64)).reshape(-1, 3)
        dets: List[dict] = []
        if len(truth_world):
            p_cam = cam.apply(truth_world)
            ok, uv, z, rng_m = self.visible(p_cam, intr, depth)
            for i in range(len(truth_world)):
                # Always draw the same number of variates per crop so one crop's visibility does not
                # shift another crop's noise (keeps runs comparable when the view changes).
                miss = self.rng.random() < float(c["miss_rate"])
                noise = self.rng.normal(0.0, 1.0, size=3)
                if not ok[i] or miss:
                    continue
                # Noise is isotropic, so adding it in the world frame is the same distribution.
                p_w = truth_world[i] + noise * self.sigma_at(rng_m[i])
                dets.append({
                    "name": names[i],
                    "xyz_world": p_w,
                    "xyz_trunk": trunk.apply(p_w)[0],
                    "conf": self._confidence(rng_m[i], uv[i, 0], uv[i, 1], intr),
                    "bbox_px": self._bbox(uv[i, 0], uv[i, 1], z[i], intr),
                })
        n_fp = int(self.rng.poisson(float(c["fp_rate_per_frame"])))
        lo, hi = c.get("fp_range_m", [0.4, float(c["max_range_m"])])
        ground_z = float(c.get("fp_ground_z_m", 0.0))
        cam_origin = cam.inverse_apply(np.zeros(3))[0]
        for _ in range(n_fp):
            uv_fp = np.array([[self.rng.uniform(0, intr.width), self.rng.uniform(0, intr.height)]])
            u01 = self.rng.random()
            # A false positive sits on something the camera can see: the depth along a downward
            # ray stops at the ground plane, so no FP is placed below the soil.
            ray_w = cam.inverse_apply(backproject(uv_fp, np.array([1.0]), intr))[0] - cam_origin
            z_hi = float(hi)
            if ray_w[2] < 0.0:
                z_hi = min(z_hi, (ground_z - cam_origin[2]) / ray_w[2])
            z_lo = min(float(lo), z_hi)
            if z_hi <= 0.0:
                continue  # camera at or below the ground plane: nothing to place
            zf = z_lo + u01 * (z_hi - z_lo)
            p_w = cam.inverse_apply(backproject(uv_fp, np.array([zf]), intr))
            dets.append({
                "name": None,
                "xyz_world": p_w[0],
                "xyz_trunk": trunk.apply(p_w)[0],
                "conf": float(self.rng.uniform(*c.get("fp_conf", [0.25, 0.55]))),
                "bbox_px": self._bbox(uv_fp[0, 0], uv_fp[0, 1], zf, intr),
            })
        dets = [d for d in dets if d["conf"] >= float(c.get("conf_threshold", 0.0))]
        dets.sort(key=lambda d: -d["conf"])
        dets = dets[: int(c.get("max_detections", 32))]
        # Associate in the world frame: the truth trunk pose plays the role of odometry, so a turn
        # does not drag every track across the trunk frame onto a neighbouring crop.
        track_ids = self.tracker.assign([d["xyz_world"] for d in dets])
        out = []
        for d, tid in zip(dets, track_ids):
            oracle = bool(c.get("oracle_ids", False)) and d["name"] is not None
            out.append({
                "id_hint": d["name"] if oracle else tid,
                "xyz_trunk": [round(float(x), 4) for x in d["xyz_trunk"]],
                "conf": round(d["conf"], 3),
                "bbox_px": d["bbox_px"],
                "_truth": d["name"],  # stripped by build_message; handy for tests / evaluation
            })
        return out


def build_message(t: float, camera: str, model_name: str, sha256: str, detections: List[dict]) -> dict:
    """The /nandi/crop_detections payload (contract in README 'Perception')."""
    dets = [{k: v for k, v in d.items() if not k.startswith("_")} for d in detections]
    return {
        "t": round(float(t), 3),
        "frame": "trunk",
        "camera": camera,
        "model": {"name": model_name, "sha256": sha256},
        "detections": dets,
    }


def encode_message(msg: dict) -> str:
    """Compact JSON; drops the lowest-confidence detections if the payload would exceed 16 KiB."""
    s = json.dumps(msg, separators=(",", ":"))
    while len(s.encode()) > MAX_MESSAGE_BYTES and msg["detections"]:
        msg = dict(msg, detections=msg["detections"][:-1])
        s = json.dumps(msg, separators=(",", ":"))
    return s


def validate_message(msg: dict) -> List[str]:
    """Schema check for consumers/tests; returns a list of problems (empty = valid)."""
    errs = []
    for key, typ in (("t", (int, float)), ("frame", str), ("camera", str), ("model", dict), ("detections", list)):
        if not isinstance(msg.get(key), typ):
            errs.append(f"{key}: missing or wrong type")
    if errs:
        return errs
    if msg["frame"] != "trunk":
        errs.append("frame must be 'trunk'")
    if not isinstance(msg["model"].get("name"), str) or len(str(msg["model"].get("sha256", ""))) != 64:
        errs.append("model: needs name and a 64-hex sha256")
    for i, d in enumerate(msg["detections"]):
        if not isinstance(d.get("id_hint"), str):
            errs.append(f"detections[{i}].id_hint")
        xyz = d.get("xyz_trunk")
        if not (isinstance(xyz, list) and len(xyz) == 3 and all(isinstance(v, (int, float)) for v in xyz)):
            errs.append(f"detections[{i}].xyz_trunk")
        if not (isinstance(d.get("conf"), (int, float)) and 0.0 <= d["conf"] <= 1.0):
            errs.append(f"detections[{i}].conf")
        bb = d.get("bbox_px")
        if not (isinstance(bb, list) and len(bb) == 4 and all(isinstance(v, int) for v in bb)
                and bb[0] <= bb[2] and bb[1] <= bb[3]):
            errs.append(f"detections[{i}].bbox_px")
        if set(d) - {"id_hint", "xyz_trunk", "conf", "bbox_px"}:
            errs.append(f"detections[{i}]: unexpected keys")
    if len(json.dumps(msg, separators=(",", ":")).encode()) > MAX_MESSAGE_BYTES:
        errs.append("payload larger than 16 KiB")
    return errs
