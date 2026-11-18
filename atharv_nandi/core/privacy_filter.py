"""Privacy filter, FilteredFrame, parcel-boundary mask and the filtered-only sinks (WP5; MVP-SPEC 5.4, 5.9;
ORCHARD-MVP-SPEC 5.6).

Every image sink (ROS topics, recorder, web, MQTT, TREE 360 strips, MP4) takes **only** ``FilteredFrame`` objects.
A ``FilteredFrame`` can be made by ``PrivacyFilter.apply`` (a camera frame) or a gather composer (``compose_gather`` / ``compose_pano`` /
``StripAssembler``: every output pixel is a weighted mean of source-frame pixels, no caller pixels enter) and
nothing else: the constructor checks a private token, the arrays are bytes-backed read-only, the attributes cannot
be re-assigned, and
``require_filtered`` / ``FilteredSink`` / ``Mp4Assembler`` / ``StripAssembler`` raise ``PrivacyError`` on a raw
array.

Masks (pixels set to grey 128; ``stats`` reports each share of the frame):

* **truth mask** (assisted / demo mode): the person actors (``world.actors.person_geoms``: body, head and limbs;
  never animals or carried props) projected through the intrinsics: the 8 corners of every geom's oriented box,
  per person a bbox, dilated ``dilate_px``.
* **detector mask** (both modes): the non-map LiDAR clusters within 8 m (``lidar_model.nonmap_clusters``),
  projected as boxes (z widened to a person height) and **expanded by 20 %**.
* A person or cluster whose part **nearer than** ``near_m`` 0.3 m (optical z; or behind the camera) reaches the
  image **cannot be bounded** and **drops the frame**; a near part outside the view does not (the rest of the box
  is clipped at the near plane and masked); ``apply`` then returns ``None`` with the reason. In assisted mode
  this also holds for the truth boxes.
* **parcel mask** (OS 5.6): pixels whose depth back-projects to a world point outside the parcel polygon (the field
  JSON ``geofence``: neighbouring parcels, huts, dwellings) are dropped before any sink. Pixels with no depth
  return are not touched (there is nothing behind them). Coverage lost this way is ``parcel_pct``; WP25 reports
  it as ``masked_pct`` in O-01.

Physics / ColorBlob mode uses the detector mask only (``mode="physics"``).

Future detectors: person detection only; no face recognition, no identity or re-identification features
(privacy rule); Apache-2.0 licence required (licence rule). Nothing here extracts any person attribute: a person is a
bbox to grey out.

MP4: ``Mp4Assembler`` writes an H.264 MP4 through ``ffmpeg`` when one is on PATH, else a pure-Python ISO-BMFF MP4
with a Motion-JPEG track (sample entry ``jpeg``; plays in ffmpeg/VLC, not in every browser). Either way it accepts
only FilteredFrames of one size.
"""
import io
import math
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

GREY = 128
DILATE_PX = 4
EXPAND_FRAC = 0.20
NEAR_M = 0.30
MODES = ("assisted", "physics")
_TOKEN = object()


class PrivacyError(TypeError):
    """A raw frame reached a sink that takes only FilteredFrames."""


def _frozen(a: np.ndarray, dtype) -> np.ndarray:
    """A copy of ``a`` backed by an immutable ``bytes`` buffer: ``setflags(write=True)`` raises, and ``.base`` is
    the ``bytes`` object, so neither the array nor its base can be made writable again."""
    a = np.ascontiguousarray(a, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=dtype).reshape(a.shape)


class FilteredFrame:
    """A privacy-filtered image. Only ``PrivacyFilter.apply`` and the gather composers (``compose_gather``,
    ``compose_pano``, ``StripAssembler``) construct one. Arrays are read-only for good (bytes-backed) and the
    attributes cannot be re-assigned."""

    __slots__ = ("camera", "stamp", "rgb", "depth", "masked", "stats", "provenance", "meta")

    def __init__(self, token, camera: str, stamp: float, rgb: np.ndarray, depth: Optional[np.ndarray],
                 masked: np.ndarray, stats: Dict[str, Any], provenance: str = "simulated",
                 meta: Optional[Dict[str, Any]] = None):
        if token is not _TOKEN:
            raise PrivacyError("FilteredFrame is made only by the privacy filter")
        put = object.__setattr__
        put(self, "camera", str(camera))
        put(self, "stamp", float(stamp))
        put(self, "rgb", _frozen(rgb, np.uint8))
        put(self, "depth", None if depth is None else _frozen(depth, np.float32))
        put(self, "masked", _frozen(masked, bool))
        put(self, "stats", dict(stats))
        put(self, "provenance", str(provenance))
        put(self, "meta", dict(meta or {}))

    def __setattr__(self, key, value):
        raise PrivacyError(f"FilteredFrame.{key} is read-only")

    def __delattr__(self, key):
        raise PrivacyError(f"FilteredFrame.{key} is read-only")

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.rgb.shape[0]), int(self.rgb.shape[1])

    def jpeg(self, quality: int = 70) -> bytes:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(self.rgb)).save(buf, format="JPEG", quality=int(quality))
        return buf.getvalue()


def require_filtered(frame) -> "FilteredFrame":
    if not isinstance(frame, FilteredFrame):
        raise PrivacyError(f"unfiltered frame refused ({type(frame).__name__}); sinks take FilteredFrame only")
    return frame


class FilteredSink:
    """Wraps a publish callable (ROS publisher, recorder, web) so that only FilteredFrames get through."""

    def __init__(self, fn: Callable[["FilteredFrame"], Any], name: str = "sink"):
        self.fn, self.name, self.count = fn, name, 0

    def publish(self, frame) -> Any:
        require_filtered(frame)
        self.count += 1
        return self.fn(frame)


# ---------------------------------------------------------------------------------------------- geometry
def project_points(points_world: np.ndarray, pose, intr) -> Tuple[np.ndarray, np.ndarray]:
    """World points -> (uv (N, 2), z (N,)); ``pose`` is a ``detector_model.Pose`` (world -> optical)."""
    P = pose.apply(np.asarray(points_world, dtype=np.float64))
    z = P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([intr.cx + intr.fx * P[:, 0] / z, intr.cy + intr.fy * P[:, 1] / z], axis=1)
    return uv, z


def box_in_view(uv: np.ndarray, z: np.ndarray, intr) -> bool:
    """Does the convex set spanned by these points possibly reach the image? (conservative)."""
    front = z > 1e-9
    if not front.any():
        return False
    if not front.all():
        return True      # straddles the camera plane: may reach anywhere
    u, v = uv[:, 0], uv[:, 1]
    return not (u.max() < 0 or u.min() >= intr.width or v.max() < 0 or v.min() >= intr.height)


_BOX_EDGES = [(a, a ^ bit) for a in range(8) for bit in (1, 2, 4) if a < a ^ bit]


def _clipped_reaches_image(P_opt: np.ndarray, intr, eps: float = 1e-3) -> bool:
    """Does any part of these boxes (8 corners each, optical frame) in front of the camera reach the image? Each
    box is clipped at the plane z = ``eps`` before projecting, so a box partly behind the camera is handled."""
    P_opt = np.asarray(P_opt, dtype=np.float64)
    blocks = [P_opt[k:k + 8] for k in range(0, len(P_opt), 8)] if len(P_opt) % 8 == 0 else [P_opt]
    for B in blocks:
        front = B[:, 2] >= eps
        if not front.any():
            continue
        pts = [B[front]]
        if len(B) == 8:
            for i, j in _BOX_EDGES:
                zi, zj = B[i, 2], B[j, 2]
                if (zi >= eps) != (zj >= eps):
                    t = (eps - zi) / (zj - zi)
                    pts.append((B[i] + t * (B[j] - B[i]))[None, :])
        Q = np.concatenate(pts, axis=0)
        u = intr.cx + intr.fx * Q[:, 0] / Q[:, 2]
        v = intr.cy + intr.fy * Q[:, 1] / Q[:, 2]
        if not (u.max() < 0 or u.min() >= intr.width or v.max() < 0 or v.min() >= intr.height):
            return True
    return False


NEAR_EPS = 1e-3


def _clip_z(V: np.ndarray, edges, c: float, keep_above: bool) -> np.ndarray:
    """Vertices of conv(V) clipped by the half-space z >= c (``keep_above``) or z <= c (optical frame). ``edges``:
    the polytope's edges (a box: ``_BOX_EDGES``) or None (every vertex pair: exact for a convex hull, since the
    segment between two hull points lies in the hull)."""
    V = np.asarray(V, dtype=np.float64)
    if len(V) == 0:
        return V.reshape(0, 3)
    s = V[:, 2] - c if keep_above else c - V[:, 2]
    inside = s >= 0
    if inside.all() or not inside.any():
        return V if inside.all() else V[:0]
    pts = [V[inside]]
    pairs = edges if edges is not None else [(i, j) for i in range(len(V)) for j in range(i + 1, len(V))]
    for i, j in pairs:
        if inside[i] != inside[j]:
            t = s[i] / (s[i] - s[j])
            pts.append((V[i] + t * (V[j] - V[i]))[None, :])
    return np.concatenate(pts, axis=0)


def _uv_reaches_image(Q: np.ndarray, intr) -> bool:
    """Points all in front of the camera (z > 0): does the bbox of their projection overlap the image?
    (conservative for the convex set they span)."""
    u = intr.cx + intr.fx * Q[:, 0] / Q[:, 2]
    v = intr.cy + intr.fy * Q[:, 1] / Q[:, 2]
    return not (u.max() < 0 or u.min() >= intr.width or v.max() < 0 or v.min() >= intr.height)


def bbox_mask(shape: Tuple[int, int], u0: float, v0: float, u1: float, v1: float) -> np.ndarray:
    H, W = shape
    m = np.zeros((H, W), dtype=bool)
    a, b = max(0, int(math.floor(u0))), min(W, int(math.ceil(u1)) + 1)
    c, d = max(0, int(math.floor(v0))), min(H, int(math.ceil(v1)) + 1)
    if a < b and c < d:
        m[c:d, a:b] = True
    return m


def geom_box_corners(gtype: int, size: np.ndarray, pos: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """8 world corners of a geom's oriented bounding box (MuJoCo geom types)."""
    import mujoco
    G = mujoco.mjtGeom
    s = np.asarray(size, dtype=np.float64)
    t = int(gtype)
    if t == int(G.mjGEOM_SPHERE):
        h = np.array([s[0]] * 3)
    elif t == int(G.mjGEOM_CAPSULE):
        h = np.array([s[0], s[0], s[0] + s[1]])
    elif t == int(G.mjGEOM_CYLINDER):
        h = np.array([s[0], s[0], s[1]])
    else:   # box, ellipsoid, mesh (rbound handled by caller)
        h = s[:3]
    R = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return np.asarray(pos)[None, :] + (signs * h[None, :]) @ R.T


def person_boxes(model, data) -> List[np.ndarray]:
    """Per person actor (truth): (8k, 3) world corners of its body / head / limb geoms (never props, animals)."""
    from atharv_nandi.sim.world import actors
    by_actor: Dict[str, List[np.ndarray]] = {}
    table = actors.geom_table(model)
    for g in actors.person_geoms(model):
        aid = table[g][1]
        by_actor.setdefault(aid, []).append(
            geom_box_corners(model.geom_type[g], model.geom_size[g], data.geom_xpos[g], data.geom_xmat[g]))
    return [np.concatenate(v, axis=0) for _, v in sorted(by_actor.items())]


def points_in_polygon(x: np.ndarray, y: np.ndarray, poly: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorised even-odd point-in-polygon."""
    P = np.asarray(poly, dtype=np.float64)
    inside = np.zeros(np.shape(x), dtype=bool)
    n = len(P)
    for i in range(n):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xint)
    return inside


def parcel_mask(depth: np.ndarray, pose, intr, polygon: Sequence[Sequence[float]],
                no_return_m: Optional[float] = None) -> np.ndarray:
    """True where the pixel's depth back-projects to a world point outside the parcel polygon."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    z = depth.astype(np.float64)
    good = np.isfinite(z) & (z > 0)
    if no_return_m is not None:
        good &= z < no_return_m
    z = np.where(good, z, 1.0)                 # placeholder depth where there is no return (masked out below)
    x = (uu + 0.5 - intr.cx) / intr.fx * z
    y = (vv + 0.5 - intr.cy) / intr.fy * z
    P = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    Pw = pose.inverse_apply(P)
    inside = points_in_polygon(Pw[:, 0], Pw[:, 1], polygon).reshape(H, W)
    return good & ~inside


# ---------------------------------------------------------------------------------------------- filter
@dataclass
class FilterResult:
    frame: Optional[FilteredFrame]
    dropped: bool
    reason: str
    stats: Dict[str, Any]


class PrivacyFilter:
    """Masks persons (truth and/or LiDAR detector) and pixels beyond the parcel, or drops the frame."""

    def __init__(self, mode: str = "assisted", dilate_px: int = DILATE_PX, expand_frac: float = EXPAND_FRAC,
                 near_m: float = NEAR_M, parcel: Optional[Sequence[Sequence[float]]] = None,
                 provenance: str = "simulated"):
        if mode not in MODES:
            raise ValueError(f"privacy mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.dilate_px = int(dilate_px)
        self.expand_frac = float(expand_frac)
        self.near_m = float(near_m)
        self.parcel = [list(map(float, p)) for p in parcel] if parcel else None
        self.provenance = provenance
        self.counts = {"frames": 0, "dropped": 0}

    def _boxes_mask(self, shape, boxes: Iterable[np.ndarray], pose, intr, pad_px: float, expand: float,
                    what: str) -> Tuple[np.ndarray, Optional[str]]:
        """Union of the projected boxes (one bbox per entry of ``boxes``; an entry is 8k corners: k oriented boxes,
        e.g. the geoms of one person). Each box is clipped at the near plane ``near_m`` (optical z): the part in
        front is projected and masked; the part nearer than ``near_m`` (or straddling the camera plane) has no
        bounded projection, so if it reaches the image the frame is dropped (MVP-SPEC 5.4). A part that is near but
        outside the view does not drop the frame (a 2 m cluster top 84 deg off a head camera's axis, say)."""
        m = np.zeros(shape, dtype=bool)
        H, W = shape
        for corners in boxes:
            P_all = pose.apply(np.asarray(corners, dtype=np.float64))
            blocks = [P_all[k:k + 8] for k in range(0, len(P_all), 8)] if len(P_all) % 8 == 0 else [P_all]
            uvs = []
            for B in blocks:
                edges = _BOX_EDGES if len(B) == 8 else None
                V = B
                if B[:, 2].min() < self.near_m:
                    near_part = _clip_z(B, edges, self.near_m, keep_above=False)
                    near_front = _clip_z(near_part, None, NEAR_EPS, keep_above=True)
                    if len(near_front) and _uv_reaches_image(near_front, intr):
                        return m, f"{what}_unbounded"
                    V = _clip_z(B, edges, self.near_m, keep_above=True)
                if len(V):
                    uvs.append(np.stack([intr.cx + intr.fx * V[:, 0] / V[:, 2],
                                         intr.cy + intr.fy * V[:, 1] / V[:, 2]], axis=1))
            if not uvs:
                continue
            uv = np.concatenate(uvs, axis=0)
            u0, v0 = uv.min(axis=0)
            u1, v1 = uv.max(axis=0)
            if u1 < 0 or u0 >= W or v1 < 0 or v0 >= H:
                continue
            # expansion relative to the box's in-image extent (equal to the full extent for a box inside the frame)
            du = (min(u1, W) - max(u0, 0.0)) * expand / 2.0
            dv = (min(v1, H) - max(v0, 0.0)) * expand / 2.0
            m |= bbox_mask(shape, u0 - du - pad_px, v0 - dv - pad_px, u1 + du + pad_px, v1 + dv + pad_px)
        return m, None

    def apply(self, camera: str, stamp: float, rgb: np.ndarray, pose, intr, depth: Optional[np.ndarray] = None,
              persons: Sequence[np.ndarray] = (), clusters: Sequence[Any] = (),
              no_return_m: Optional[float] = None, meta: Optional[dict] = None) -> FilterResult:
        """Filter one camera frame. ``persons``: truth corner sets (``person_boxes``); ``clusters``:
        ``lidar_model.Cluster`` objects (or dicts with ``lo`` / ``hi``)."""
        self.counts["frames"] += 1
        H, W = rgb.shape[:2]
        shape = (H, W)
        stats: Dict[str, Any] = {"mode": self.mode}
        truth = np.zeros(shape, dtype=bool)
        if self.mode == "assisted":
            truth, why = self._boxes_mask(shape, persons, pose, intr, self.dilate_px, 0.0, "person")
            if why:
                return self._drop(why, stats)
        boxes = [c.corners() if hasattr(c, "corners") else _dict_corners(c) for c in clusters]
        det, why = self._boxes_mask(shape, boxes, pose, intr, 0.0, self.expand_frac, "cluster")
        if why:
            return self._drop(why, stats)
        parcel = np.zeros(shape, dtype=bool)
        if self.parcel is not None and depth is not None:
            parcel = parcel_mask(depth, pose, intr, self.parcel, no_return_m)
        privacy = truth | det
        masked = privacy | parcel
        out = np.array(rgb[..., :3], dtype=np.uint8, copy=True)
        out[masked] = GREY
        n = float(H * W)
        stats.update({"truth_pct": round(100.0 * truth.sum() / n, 3), "detector_pct": round(100.0 * det.sum() / n, 3),
                      "privacy_pct": round(100.0 * privacy.sum() / n, 3),
                      "parcel_pct": round(100.0 * parcel.sum() / n, 3),
                      "masked_pct": round(100.0 * masked.sum() / n, 3)})
        d_out = None
        if depth is not None:
            d_out = np.array(depth, dtype=np.float32, copy=True)
            d_out[masked] = np.nan
        frame = FilteredFrame(_TOKEN, camera, stamp, out, d_out, masked, stats, self.provenance, meta)
        frame.meta["masks"] = {"truth": truth, "detector": det, "parcel": parcel}
        return FilterResult(frame, False, "ok", stats)

    def _drop(self, reason: str, stats: dict) -> FilterResult:
        self.counts["dropped"] += 1
        stats = dict(stats, dropped=reason)
        return FilterResult(None, True, reason, stats)


def _dict_corners(c: dict) -> np.ndarray:
    lo, hi = np.asarray(c["lo"], dtype=float), np.asarray(c["hi"], dtype=float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def compose(*_args, **_kw):
    """Removed (WP5 review): ``compose(name, stamp, rgb, sources)`` took the derived image verbatim, so any raw
    array became a FilteredFrame next to one unrelated source. Use ``compose_gather`` (or ``compose_pano``,
    ``StripAssembler``): they build the pixels themselves from the source frames."""
    raise PrivacyError("compose(rgb, sources) was removed: build derived images with compose_gather / compose_pano "
                       "/ StripAssembler, which take pixels only from FilteredFrames")


def compose_gather(name: str, stamp: float, sources: Sequence[FilteredFrame], out_hw: Tuple[int, int],
                   tables: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                   extra_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
                   stats: Optional[dict] = None) -> FilteredFrame:
    """A derived image (pano, strip) whose every pixel is a weighted mean of source-frame pixels; no caller pixel
    data enters. ``tables[k] = (out_idx, src_idx, w)`` for ``sources[k]``: flat output pixel index, flat source
    pixel index, weight >= 0 (``out_idx`` unique per source). Source pixels that are masked in the FilteredFrame
    (or in ``extra_masks[k]``, e.g. a self-mask) get weight 0; output pixels with no weight are grey and masked."""
    srcs = [require_filtered(s) for s in sources]
    if not srcs:
        raise PrivacyError("compose needs at least one FilteredFrame source")
    if len(tables) != len(srcs):
        raise ValueError(f"compose: {len(tables)} tables for {len(srcs)} sources")
    H, W = int(out_hw[0]), int(out_hw[1])
    acc = np.zeros((H * W, 3), dtype=np.float32)
    wsum = np.zeros(H * W, dtype=np.float32)
    for k, (f, (pi, ci, w)) in enumerate(zip(srcs, tables)):
        pi, ci = np.asarray(pi, dtype=np.int64), np.asarray(ci, dtype=np.int64)
        w = np.asarray(w, dtype=np.float32)
        h, wd = f.shape
        if len(pi) and (pi.min() < 0 or pi.max() >= H * W or ci.min() < 0 or ci.max() >= h * wd):
            raise ValueError(f"compose: table {k} indexes outside the output or source {f.camera}")
        if np.any(w < 0):
            raise ValueError("compose: negative weight")
        bad = np.asarray(f.masked).reshape(-1)
        em = None if extra_masks is None else extra_masks[k]
        if em is not None:
            bad = bad | np.asarray(em, dtype=bool).reshape(-1)
        ww = w * (~bad[ci]).astype(np.float32)
        wsum[pi] += ww                     # pi unique per source: a plain fancy-index add is exact
        acc[pi] += np.asarray(f.rgb).reshape(-1, 3)[ci].astype(np.float32) * ww[:, None]
    filled = wsum > 1e-6
    out = np.full((H * W, 3), GREY, dtype=np.float32)
    out[filled] = acc[filled] / wsum[filled, None]
    rgb = np.clip(np.rint(out), 0, 255).astype(np.uint8).reshape(H, W, 3)
    st = dict(stats or {})
    st.setdefault("sources", [s.camera for s in srcs])
    return FilteredFrame(_TOKEN, name, stamp, rgb, None, ~filled.reshape(H, W), st, srcs[0].provenance)


def compose_pano(lut, frames: Dict[str, FilteredFrame], stamp: float,
                 extra_masks: Optional[Dict[str, Optional[np.ndarray]]] = None,
                 stats: Optional[dict] = None, name: str = "pano") -> FilteredFrame:
    """The ring pano from FilteredFrames through a ``pano.PanoLUT``'s gather tables (same pixels as
    ``lut.stitch`` with the frames' privacy masks plus ``extra_masks``)."""
    srcs, tables, extra = [], [], []
    for cam, ent in zip(lut.cams, lut.entries):
        f = frames.get(cam.name)
        if f is None:
            continue
        f = require_filtered(f)
        if f.shape != (cam.height, cam.width):
            raise ValueError(f"pano: {cam.name} is {f.shape[1]}x{f.shape[0]}, LUT wants {cam.width}x{cam.height}")
        srcs.append(f)
        tables.append(ent)
        extra.append((extra_masks or {}).get(cam.name))
    return compose_gather(name, stamp, srcs, (lut.H, lut.W), tables, extra, stats)


# ---------------------------------------------------------------------------------------------- strips and MP4
class StripAssembler:
    """A band strip (keyframes side by side, scaled to ``height``) from FilteredFrames only."""

    def __init__(self, name: str, height: int = 120):
        self.name, self.height = name, int(height)
        self.frames: List[FilteredFrame] = []

    def add(self, frame) -> None:
        self.frames.append(require_filtered(frame))

    def build(self, stamp: float = 0.0) -> FilteredFrame:
        if not self.frames:
            raise ValueError("strip: no frames")
        steps = [max(1, int(round(f.shape[0] / self.height))) for f in self.frames]
        sub = [(len(range(0, f.shape[0], st)), len(range(0, f.shape[1], st))) for f, st in zip(self.frames, steps)]
        hmin = min(h for h, _ in sub)
        W = sum(w for _, w in sub)
        tables, x0 = [], 0
        for f, st, (_, w) in zip(self.frames, steps, sub):
            rr, cc = np.meshgrid(np.arange(hmin), np.arange(w), indexing="ij")
            out_idx = (rr * W + x0 + cc).ravel()
            src_idx = (rr * st * f.shape[1] + cc * st).ravel()
            tables.append((out_idx, src_idx, np.ones(out_idx.size, dtype=np.float32)))
            x0 += w
        return compose_gather(self.name, stamp, self.frames, (hmin, W), tables, stats={"n": len(self.frames)})


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + kind + payload


def _full(kind: bytes, version: int, flags: int, payload: bytes) -> bytes:
    return _box(kind, struct.pack(">I", (version << 24) | flags) + payload)


def mjpeg_mp4(jpegs: Sequence[bytes], width: int, height: int, fps: float) -> bytes:
    """A minimal ISO-BMFF MP4 with one Motion-JPEG video track (sample entry ``jpeg``)."""
    timescale = 1000
    dur = max(1, int(round(timescale / fps)))
    n = len(jpegs)
    total = dur * n
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")
    mdat_payload = b"".join(jpegs)
    mdat = _box(b"mdat", mdat_payload)
    chunk_off = len(ftyp) + 8
    matrix = struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)
    mvhd = _full(b"mvhd", 0, 0, struct.pack(">IIII", 0, 0, timescale, total) + struct.pack(">IH", 0x00010000, 0x0100)
                 + b"\0" * 10 + matrix + b"\0" * 24 + struct.pack(">I", 2))
    tkhd = _full(b"tkhd", 0, 3, struct.pack(">IIIII", 0, 0, 1, 0, total) + b"\0" * 8 + struct.pack(">hhhH", 0, 0, 0, 0)
                 + matrix + struct.pack(">II", width << 16, height << 16))
    mdhd = _full(b"mdhd", 0, 0, struct.pack(">IIII", 0, 0, timescale, total) + struct.pack(">HH", 0x55C4, 0))
    hdlr = _full(b"hdlr", 0, 0, struct.pack(">I", 0) + b"vide" + b"\0" * 12 + b"VideoHandler\0")
    vmhd = _full(b"vmhd", 0, 1, b"\0" * 8)
    dref = _full(b"dref", 0, 0, struct.pack(">I", 1) + _full(b"url ", 0, 1, b""))
    dinf = _box(b"dinf", dref)
    entry = (b"\0" * 6 + struct.pack(">H", 1) + b"\0" * 16 + struct.pack(">HH", width, height)
             + struct.pack(">II", 0x00480000, 0x00480000) + struct.pack(">I", 0) + struct.pack(">H", 1)
             + bytes([12]) + b"Motion JPEG".ljust(31, b"\0") + struct.pack(">Hh", 24, -1))
    stsd = _full(b"stsd", 0, 0, struct.pack(">I", 1) + _box(b"jpeg", entry))
    stts = _full(b"stts", 0, 0, struct.pack(">III", 1, n, dur))
    stsc = _full(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, n, 1))
    stsz = _full(b"stsz", 0, 0, struct.pack(">II", 0, n) + b"".join(struct.pack(">I", len(j)) for j in jpegs))
    stco = _full(b"stco", 0, 0, struct.pack(">II", 1, chunk_off))
    stbl = _box(b"stbl", stsd + stts + stsc + stsz + stco)
    minf = _box(b"minf", vmhd + dinf + stbl)
    mdia = _box(b"mdia", mdhd + hdlr + minf)
    trak = _box(b"trak", tkhd + mdia)
    moov = _box(b"moov", mvhd + trak)
    return ftyp + mdat + moov


def mp4_boxes(data: bytes) -> List[Tuple[bytes, int, int]]:
    """Top-level (type, offset, size) of an MP4 (the assembler test)."""
    out, off = [], 0
    while off + 8 <= len(data):
        size, kind = struct.unpack(">I4s", data[off:off + 8])
        if size < 8:
            break
        out.append((kind, off, size))
        off += size
    return out


class Mp4Assembler:
    """Per-band MP4 from FilteredFrames only (OS 5.6). ``write`` returns the encoder that was used."""

    def __init__(self, fps: float = 4.0, quality: int = 80, prefer_ffmpeg: bool = True):
        self.fps, self.quality, self.prefer_ffmpeg = float(fps), int(quality), bool(prefer_ffmpeg)
        self.frames: List[FilteredFrame] = []

    def add(self, frame) -> None:
        f = require_filtered(frame)
        if self.frames and f.shape != self.frames[0].shape:
            raise ValueError(f"mp4: frame {f.shape} differs from {self.frames[0].shape}")
        self.frames.append(f)

    def encode(self) -> Tuple[bytes, str]:
        if not self.frames:
            raise ValueError("mp4: no frames")
        h, w = self.frames[0].shape
        ff = shutil.which("ffmpeg") if self.prefer_ffmpeg else None
        if ff:
            raw = b"".join(np.ascontiguousarray(f.rgb).tobytes() for f in self.frames)
            cmd = [ff, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r",
                   str(self.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   "-f", "mp4", "-"]
            try:
                res = subprocess.run(cmd, input=raw, capture_output=True, timeout=120, check=True)
                return res.stdout, "ffmpeg_h264"
            except (subprocess.SubprocessError, OSError):
                pass
        return mjpeg_mp4([f.jpeg(self.quality) for f in self.frames], w, h, self.fps), "mjpeg_mp4"

    def write(self, path: str) -> str:
        data, enc = self.encode()
        with open(path, "wb") as fh:
            fh.write(data)
        return enc
