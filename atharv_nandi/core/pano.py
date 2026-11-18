"""Ring panorama: equirect LUT stitch of the 4 ring cameras (WP5; MVP-SPEC 2.4, 3.7; ORCHARD-MVP-SPEC 2.3).

Geometry. The fixed ring on the crown mast carries 4 cameras at yaw 0/90/180/270 (trunk frame), pitched 25 deg
down, fovy 83.6 deg at 4:3 (HFOV 100 deg), lens 0.98 m. Neighbouring cameras overlap by 10 deg on the centre
row. ``ring_from_model`` reads the yaw / pitch / fovy of each camera from the compiled model (relative to the
trunk), so the LUT follows the MJCF.

Strip. A numpy LUT gather into a ``W x H`` (640 x 256) equirect strip: azimuth 360 deg (column 0 = +180 deg,
the centre column = straight ahead, left of centre = robot left), elevation ``el_min..el_max``. Each pano pixel
maps to (camera, u, v) for every camera that sees it; in the overlaps the pixel is a **feathered blend**: a
camera's weight ramps from 0 at its left/right image edge to 1 at ``feather_deg`` (5 deg) inside it.
**Self-masked pixels** (``self_mask``) get weight 0, so they are filled from the neighbouring camera where it
sees the same direction, else left grey (128). Pixels of a frame the privacy filter masked are already grey.

Elevation band (computed, ``tests/test_pano.py``). The spec's -67..+17 deg is the vertical FOV on the camera's
centre column. Off-axis, a pinhole frustum covers less elevation: at a seam (45 deg off-axis) the 4 cameras
reach only about -59..+7 deg, so the -67..+17 strip would have 8,952 uncovered corner pixels. The strip
therefore uses ``EL_MIN_DEG`` -56 .. ``EL_MAX_DEG`` +5, where **every pixel is covered by at least one camera**.
The seam overlap is at least 5 deg on every row from -56 to -1 deg (90 % of the rows) and tapers to about
1.1 deg at +5 deg (computed at the 0.56 deg column pitch), where the feather ramp is correspondingly narrower.

Sector ages. ``SectorAges`` keeps the stamp of the newest frame per camera; the web JSON carries the age of each
sector (a pano sector can be 2 s stale: the ring runs round-robin at 2 cameras/s; the pano is not a safety
input).

Output. JPEG q70 at <= 1 Hz (``encode_jpeg``), 8-9 KB measured on ``orchard_lite`` (a pure-noise worst case stays
< 512 KiB, tested). ``PanoLUT.stitch`` works on arrays; the perception node feeds it only the arrays of
privacy-filtered frames (``privacy_filter.FilteredFrame``) and wraps the result with ``privacy_filter.compose``, so
the published pano is itself a FilteredFrame.

Conventions: camera optical frame x right, y down, z forward (``detector_model``); pinhole with
f = (H/2)/tan(fovy/2) and the principal point at the image centre (MuJoCo).
"""
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PANO_W, PANO_H = 640, 256
EL_MIN_DEG, EL_MAX_DEG = -56.0, 5.0
FEATHER_DEG = 5.0
GREY = 128
RING_CAMS = ("ring_0", "ring_1", "ring_2", "ring_3")
MAX_JPEG_BYTES = 512 * 1024


@dataclass(frozen=True)
class RingCamera:
    name: str
    yaw_deg: float           # optical axis azimuth in the trunk frame (0 = +x forward, CCW)
    pitch_down_deg: float    # optical axis below the horizon
    fovy_deg: float
    width: int
    height: int

    @property
    def f(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)

    @property
    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.width / 2.0) / self.f))

    def rot(self) -> np.ndarray:
        """Trunk -> optical rotation (rows = optical x, y, z axes in the trunk frame)."""
        y, p = math.radians(self.yaw_deg), math.radians(self.pitch_down_deg)
        z = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])
        x = np.array([math.sin(y), -math.cos(y), 0.0])
        return np.stack([x, np.cross(z, x), z])

    def project(self, dirs_trunk: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trunk-frame directions (..., 3) -> (u, v, in_front)."""
        P = np.asarray(dirs_trunk, dtype=np.float64) @ self.rot().T
        z = P[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.f * P[..., 0] / z + self.width / 2.0
            v = self.f * P[..., 1] / z + self.height / 2.0
        return u, v, z > 0

    def backproject(self, u, v) -> np.ndarray:
        """Pixel coordinates -> unit trunk-frame directions."""
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        p = np.stack([(u - self.width / 2.0) / self.f, (v - self.height / 2.0) / self.f, np.ones_like(u)], -1)
        p = p / np.linalg.norm(p, axis=-1, keepdims=True)
        return p @ self.rot()


def nominal_ring(width: int = 160, height: int = 120, fovy_deg: float = 83.6, pitch_deg: float = 25.0):
    return [RingCamera(n, 90.0 * k, pitch_deg, fovy_deg, width, height) for k, n in enumerate(RING_CAMS)]


def ring_from_model(model, data, width: int, height: int, names: Sequence[str] = RING_CAMS,
                    trunk: str = "trunk") -> List[RingCamera]:
    """Ring cameras with yaw / pitch measured from the compiled model (needs kinematics up to date)."""
    import mujoco
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, trunk)
    Rt = data.xmat[b].reshape(3, 3)
    out = []
    for n in names:
        c = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)
        if c < 0:
            raise KeyError(f"pano: camera {n!r} not in model")
        fwd = Rt.T @ (-data.cam_xmat[c].reshape(3, 3)[:, 2])    # MuJoCo camera looks along local -z
        yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
        pitch = -math.degrees(math.asin(max(-1.0, min(1.0, fwd[2]))))
        out.append(RingCamera(n, yaw, pitch, float(model.cam_fovy[c]), width, height))
    return out


def pano_dirs(W: int = PANO_W, H: int = PANO_H, el_min: float = EL_MIN_DEG, el_max: float = EL_MAX_DEG):
    """(H, W, 3) unit trunk-frame directions plus azimuth (W,) and elevation (H,) in degrees."""
    az = 180.0 - 360.0 * (np.arange(W) + 0.5) / W
    el = el_max - (el_max - el_min) * (np.arange(H) + 0.5) / H
    A, E = np.meshgrid(np.radians(az), np.radians(el))
    D = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1)
    return D, az, el


class PanoLUT:
    """Precomputed gather tables: per camera the flat pano indices it covers, its pixel indices and weights."""

    def __init__(self, cams: Sequence[RingCamera], W: int = PANO_W, H: int = PANO_H, el_min: float = EL_MIN_DEG,
                 el_max: float = EL_MAX_DEG, feather_deg: float = FEATHER_DEG):
        self.cams = list(cams)
        self.W, self.H = int(W), int(H)
        self.feather_deg = float(feather_deg)
        D, self.az, self.el = pano_dirs(W, H, el_min, el_max)
        self.valid: List[np.ndarray] = []
        self.entries: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []   # (pano flat idx, cam flat idx, w)
        for cam in self.cams:
            u, v, front = cam.project(D)
            ok = front & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
            self.valid.append(ok)
            deg_per_px = cam.hfov_deg / cam.width
            margin = np.minimum(u, cam.width - u) * deg_per_px
            w = np.clip(margin / self.feather_deg, 1e-3, 1.0) if self.feather_deg > 0 else np.ones_like(u)
            pi = np.nonzero(ok.ravel())[0]
            ui = np.clip(np.floor(u.ravel()[pi]).astype(np.int64), 0, cam.width - 1)
            vi = np.clip(np.floor(v.ravel()[pi]).astype(np.int64), 0, cam.height - 1)
            self.entries.append((pi, vi * cam.width + ui, w.ravel()[pi].astype(np.float32)))
        self.count = np.sum(self.valid, axis=0)

    # ---------------------------------------------------------------- coverage report
    def uncovered(self) -> int:
        return int(np.sum(self.count == 0))

    def seam_overlap_deg(self) -> np.ndarray:
        """(n_seams, H): overlap width in degrees between each pair of neighbouring cameras, per row."""
        deg_per_col = 360.0 / self.W
        n = len(self.cams)
        return np.array([np.sum(self.valid[k] & self.valid[(k + 1) % n], axis=1) * deg_per_col for k in range(n)])

    # ---------------------------------------------------------------- stitch
    def stitch(self, images: Mapping[str, np.ndarray], masks: Optional[Mapping[str, np.ndarray]] = None) -> \
            Tuple[np.ndarray, np.ndarray]:
        """(pano RGB uint8 (H, W, 3), filled bool (H, W)). ``images[name]`` (h, w, 3) uint8; ``masks[name]``
        True = pixel unusable (self-mask, privacy grey). Missing cameras leave their sectors grey."""
        acc = np.zeros((self.H * self.W, 3), dtype=np.float32)
        wsum = np.zeros(self.H * self.W, dtype=np.float32)
        for cam, (pi, ci, w) in zip(self.cams, self.entries):
            img = images.get(cam.name)
            if img is None:
                continue
            if img.shape[0] != cam.height or img.shape[1] != cam.width:
                raise ValueError(f"pano: {cam.name} is {img.shape[1]}x{img.shape[0]}, LUT wants "
                                 f"{cam.width}x{cam.height}")
            flat = img.reshape(-1, img.shape[-1])[:, :3]
            ww = w
            mk = None if masks is None else masks.get(cam.name)
            if mk is not None:
                ww = w * (~mk.reshape(-1)[ci]).astype(np.float32)
            wsum[pi] += ww          # pi is unique per camera, so a plain fancy-index add is exact
            acc[pi] += flat[ci].astype(np.float32) * ww[:, None]
        filled = wsum > 1e-6
        out = np.full((self.H * self.W, 3), GREY, dtype=np.float32)
        out[filled] = acc[filled] / wsum[filled, None]
        return (np.clip(np.rint(out), 0, 255).astype(np.uint8).reshape(self.H, self.W, 3),
                filled.reshape(self.H, self.W))


class SectorAges:
    """Stamp of the newest frame per ring camera; ``ages(now)`` for the web JSON (MVP-SPEC 3.7)."""

    def __init__(self, names: Sequence[str] = RING_CAMS):
        self.stamps: Dict[str, Optional[float]] = {n: None for n in names}

    def update(self, name: str, stamp: float):
        if name in self.stamps:
            self.stamps[name] = float(stamp)

    def ages(self, now: float) -> Dict[str, Optional[float]]:
        return {n: (None if s is None else round(max(0.0, now - s), 3)) for n, s in self.stamps.items()}


def encode_jpeg(rgb: np.ndarray, quality: int = 70) -> bytes:
    from PIL import Image  # python3-pil in the image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=int(quality))
    data = buf.getvalue()
    if len(data) >= MAX_JPEG_BYTES:
        raise ValueError(f"pano JPEG {len(data)} B >= {MAX_JPEG_BYTES} B")
    return data
