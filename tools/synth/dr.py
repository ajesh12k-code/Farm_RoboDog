"""Domain randomisation (CAMERA-TRAINING C.2): knob sampler and the image-space post-process.

Knobs live in ``config/synth.yaml`` under ``dr``. Each has ``dist`` (uniform | normal | choice | bernoulli),
``range`` / ``values`` / ``p``, a curriculum ``stage`` and a ``default``. A knob is sampled when its stage is
<= ``curriculum_stage`` and takes its default otherwise. Draws come from a per-image generator, so one
image's draws never shift another's.

Geometry-changing DR (lens distortion) is applied to RGB, seg and depth alike with the same nearest-pixel
map from a padded render (see the geometry section), so masks still equal the (distorted) seg render.
Photometric DR (exposure, white balance, noise, blur, rain, drops, lens dirt, glare, JPEG) only touches
RGB; labels are untouched.
"""
import colorsys
import io
import math
from typing import Dict, Optional, Tuple

import numpy as np

DISTS = ("uniform", "normal", "choice", "bernoulli")
WEATHER = ("clear", "haze", "fog", "dust", "winter_fog")


class DRConfigError(ValueError):
    pass


def validate(dr_cfg: dict) -> None:
    """Refuse a knob without dist, support, stage or default."""
    if not isinstance(dr_cfg, dict) or not dr_cfg:
        raise DRConfigError("dr: empty")
    for name, k in dr_cfg.items():
        if not isinstance(k, dict):
            raise DRConfigError(f"dr.{name}: not a mapping")
        dist = k.get("dist")
        if dist not in DISTS:
            raise DRConfigError(f"dr.{name}: dist {dist!r} not in {DISTS}")
        if "stage" not in k or not isinstance(k["stage"], int) or k["stage"] < 0:
            raise DRConfigError(f"dr.{name}: missing or bad curriculum stage")
        if "default" not in k:
            raise DRConfigError(f"dr.{name}: missing default")
        if dist in ("uniform", "normal"):
            r = k.get("range")
            if not (isinstance(r, (list, tuple)) and len(r) == 2 and float(r[0]) <= float(r[1])):
                raise DRConfigError(f"dr.{name}: range must be [lo, hi]")
        elif dist == "choice":
            v = k.get("values")
            if not v:
                raise DRConfigError(f"dr.{name}: choice needs values")
            w = k.get("weights")
            if w is not None and (len(w) != len(v) or min(w) < 0 or sum(w) <= 0):
                raise DRConfigError(f"dr.{name}: bad weights")
        elif dist == "bernoulli":
            p = k.get("p")
            if p is None or not 0.0 <= float(p) <= 1.0:
                raise DRConfigError(f"dr.{name}: bernoulli needs p in [0, 1]")


def sample(dr_cfg: dict, rng: np.random.Generator, stage: int) -> Dict[str, object]:
    """One draw of every knob, in sorted name order (stable across Python dict orders)."""
    out: Dict[str, object] = {}
    for name in sorted(dr_cfg):
        k = dr_cfg[name]
        if int(k["stage"]) > int(stage):
            out[name] = k["default"]
            continue
        dist = k["dist"]
        if dist == "uniform":
            lo, hi = (float(x) for x in k["range"])
            out[name] = float(rng.uniform(lo, hi))
        elif dist == "normal":
            lo, hi = (float(x) for x in k["range"])
            mu = float(k.get("mean", 0.5 * (lo + hi)))
            sd = float(k.get("std", (hi - lo) / 4.0))
            out[name] = float(np.clip(rng.normal(mu, sd), lo, hi))
        elif dist == "choice":
            vals = list(k["values"])
            w = k.get("weights")
            p = None if w is None else np.asarray(w, dtype=np.float64) / float(np.sum(w))
            out[name] = vals[int(rng.choice(len(vals), p=p))]
        else:
            out[name] = bool(rng.random() < float(k["p"]))
    return out


# ------------------------------------------------------------------------------------------- colour
def kelvin_rgb(k: float) -> Tuple[float, float, float]:
    """Black-body colour (Tanner Helland fit), normalised to max 1."""
    t = float(np.clip(k, 1000.0, 40000.0)) / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * ((t - 60) ** -0.1332047592)
    g = (99.4708025861 * math.log(t) - 161.1195681661) if t <= 66 else 288.1221695283 * ((t - 60) ** -0.0755148492)
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    rgb = np.clip(np.array([r, g, b]), 0, 255) / 255.0
    return tuple(float(x) for x in rgb / max(rgb.max(), 1e-6))


def hsv_rgb(h_deg: float, s: float, v: float) -> Tuple[float, float, float]:
    return colorsys.hsv_to_rgb((h_deg % 360.0) / 360.0, float(np.clip(s, 0, 1)), float(np.clip(v, 0, 1)))


def sun_dir(elev_deg: float, az_deg: float) -> np.ndarray:
    """Direction the light travels (MuJoCo light ``dir``): from the sun towards the ground."""
    e, a = math.radians(elev_deg), math.radians(az_deg)
    to_sun = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return -to_sun


def weather_fog(weather: str) -> Optional[Tuple[float, float, Tuple[float, float, float, float]]]:
    """(fogstart, fogend, rgba) as fractions of ``stat.extent`` (10 m in the stub scenes), or None."""
    return {
        "clear": None,
        "haze": (1.0, 6.0, (0.80, 0.83, 0.88, 1.0)),
        "fog": (0.2, 2.0, (0.78, 0.80, 0.82, 1.0)),
        "dust": (0.4, 3.0, (0.72, 0.62, 0.48, 1.0)),
        "winter_fog": (0.05, 0.8, (0.85, 0.86, 0.88, 1.0)),
    }[weather]


# ------------------------------------------------------------------------------------------- geometry
# Brown k1, k2 on a pinhole render: the stored image follows x_d = x_u * g(r_u) / r_u with
# g(r) = r (1 + k1 r^2 + k2 r^4) in normalised coordinates. Two limits keep that honest on wide rigs
# (ring fovy 78-86, belly 88-92, plus the fovy jitter):
#  - g must be strictly increasing (slope >= MIN_SLOPE) on [0, MAX_STRETCH * r_corner], so every output
#    pixel has exactly one undistorted source (a folded model has none, or two);
#  - barrel distortion pulls sources from outside the frame, so the pinhole frame is rendered padded (same
#    fy, centred principal point) by at most MAX_STRETCH of the corner radius; nothing is clipped to an edge.
# A sampled (k1, k2) that breaks either limit is scaled down by one factor s in (0, 1]; the effective k1, k2
# (what the image follows) go in the meta intrinsics and s in ``distortion_scale``.
MAX_STRETCH = 1.2
MIN_SLOPE = 0.25


def render_buffer(w: int, h: int) -> Tuple[int, int]:
    """Offscreen buffer that holds the widest padded render for a w x h output."""
    return int(math.ceil(w * MAX_STRETCH)) + 4, int(math.ceil(h * MAX_STRETCH)) + 4


def _min_slope_coeff(k1: float, k2: float, t_max: float) -> float:
    """min over t = r^2 in [0, t_max] of 3 k1 t + 5 k2 t^2 (so g'(r) = 1 + s * that)."""
    ts = [0.0, t_max]
    if k2 > 0:
        tv = -3.0 * k1 / (10.0 * k2)
        if 0.0 < tv < t_max:
            ts.append(tv)
    return min(3.0 * k1 * t + 5.0 * k2 * t * t for t in ts)


def effective_distortion(k1: float, k2: float, r_corner: float) -> Tuple[float, float, float]:
    """(k1, k2, s): the sampled coefficients scaled by the largest s in [0, 1] that keeps g' >= MIN_SLOPE on
    [0, MAX_STRETCH r_corner] and the corner's source inside MAX_STRETCH r_corner. Both conditions are
    linear in s at a fixed r, so this closed form is exact."""
    R = MAX_STRETCH * r_corner
    s = 1.0
    a = _min_slope_coeff(k1, k2, R * R)
    if a < 0:
        s = min(s, (MIN_SLOPE - 1.0) / a)
    c = R * (k1 * R * R + k2 * R ** 4)           # g(R) = R + s c must reach r_corner
    if c < 0:
        s = min(s, (r_corner - R) / c)
    s = max(0.0, min(1.0, s))
    if s < 1.0:
        s *= 1.0 - 1e-9                           # strictly inside the feasible set
    return k1 * s, k2 * s, s


def forward_radius(r, k1: float, k2: float):
    r2 = r * r
    return r * (1.0 + k1 * r2 + k2 * r2 * r2)


class DistortionMap:
    """Nearest-pixel sources for every output pixel, indexing a render padded by (pad_x, pad_y)."""

    def __init__(self, sv, su, k1, k2, scale, pad_x, pad_y, render_w, render_h, xs, ys):
        self.sv, self.su = sv, su
        self.k1, self.k2, self.scale = k1, k2, scale
        self.pad_x, self.pad_y = pad_x, pad_y
        self.render_w, self.render_h = render_w, render_h
        self.xs, self.ys = xs, ys                    # continuous sources in render pixels (for checks)


def distortion_map(w: int, h: int, fy: float, k1: float, k2: float) -> Optional[DistortionMap]:
    """Map for radial distortion of a w x h output with focal fy (px), or None when no distortion applies.

    The inverse is a bisection on the monotone branch of g (60 halvings: exact to float precision), so the
    stored image and its labels follow the recorded effective (k1, k2). Sources index a pinhole render of
    (render_w, render_h) = (w + 2 pad_x, h + 2 pad_y) with the same fy and a centred principal point.
    """
    if abs(k1) < 1e-12 and abs(k2) < 1e-12:
        return None
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r_corner = math.hypot(cx, cy) / fy
    k1e, k2e, s = effective_distortion(float(k1), float(k2), r_corner)
    if abs(k1e) < 1e-12 and abs(k2e) < 1e-12:
        return None
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    xd, yd = (u - cx) / fy, (v - cy) / fy
    rd = np.hypot(xd, yd)
    lo, hi = np.zeros_like(rd), np.full_like(rd, MAX_STRETCH * r_corner)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = forward_radius(mid, k1e, k2e) < rd
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    ru = 0.5 * (lo + hi)
    ratio = np.divide(ru, rd, out=np.ones_like(rd), where=rd > 0)
    xs, ys = xd * ratio * fy, yd * ratio * fy          # source offsets from the centre, px
    pad_x = max(0, int(math.ceil(float(np.abs(xs).max()) - cx - 0.5 + 1e-6)))
    pad_y = max(0, int(math.ceil(float(np.abs(ys).max()) - cy - 0.5 + 1e-6)))
    rw, rh = w + 2 * pad_x, h + 2 * pad_y
    xs, ys = xs + cx + pad_x, ys + cy + pad_y
    su = np.clip(np.rint(xs), 0, rw - 1).astype(np.int64)    # no-op by construction of the pad
    sv = np.clip(np.rint(ys), 0, rh - 1).astype(np.int64)
    return DistortionMap(sv, su, k1e, k2e, s, pad_x, pad_y, rw, rh, xs, ys)


def apply_map(img: np.ndarray, m: Optional[DistortionMap]) -> np.ndarray:
    """Sample the (padded) render; with no map the image is returned unchanged."""
    if m is None:
        return img
    if img.shape[0] != m.render_h or img.shape[1] != m.render_w:
        raise ValueError(f"render is {img.shape[1]}x{img.shape[0]}, map expects {m.render_w}x{m.render_h}")
    return img[m.sv, m.su]


# ------------------------------------------------------------------------------------------- photometric
def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return img
    k = 2 * r + 1
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode="edge")
    c = np.cumsum(np.cumsum(pad, axis=0, dtype=np.float64), axis=1)
    c = np.pad(c, ((1, 0), (1, 0), (0, 0)))
    h, w = img.shape[:2]
    s = c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]
    return s / (k * k)


def _motion_blur(img: np.ndarray, length: int, horizontal: bool) -> np.ndarray:
    if length <= 1:
        return img
    axis = 1 if horizontal else 0
    acc = np.zeros_like(img, dtype=np.float64)
    for s in range(length):
        acc += np.roll(img, s - length // 2, axis=axis)
    return acc / length


def postprocess_rgb(rgb: np.ndarray, knobs: dict, rng: np.random.Generator, camera: str,
                    tag_box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Photometric DR on a uint8 RGB frame. Deterministic given ``rng``. Labels are never touched."""
    img = rgb.astype(np.float64)
    h, w = img.shape[:2]
    wb = float(knobs.get("white_balance", 1.0))
    img *= float(knobs.get("exposure", 1.0)) * np.array([wb, 1.0, 2.0 - wb])
    img = _box_blur(img, int(knobs.get("defocus_px", 0)))
    img = _motion_blur(img, int(knobs.get("motion_blur_px", 0)), horizontal=bool(rng.random() < 0.7))
    if knobs.get("dock_tag_glare") and tag_box is not None:
        x0, y0, x1, y1 = tag_box
        cy, cx = (y0 + y1) / 2.0 + rng.uniform(-3, 3), (x0 + x1) / 2.0 + rng.uniform(-3, 3)
        rad = max(3.0, 0.4 * max(x1 - x0, y1 - y0))
        yy, xx = np.mgrid[0:h, 0:w]
        img += 230.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad * rad))[..., None]
    if knobs.get("rain_streaks"):
        n = int(rng.integers(40, 120))
        for _ in range(n):
            x, y, ln = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(6, 18))
            ys = np.arange(y, min(h, y + ln))
            xs = np.clip(x + ((ys - y) * 0.15).astype(np.int64), 0, w - 1)
            img[ys, xs] = 0.6 * img[ys, xs] + 0.4 * 220.0
    if knobs.get("lens_drops"):
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(int(rng.integers(2, 6))):
            cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(6, 20)
            m = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
            if m.any():
                img[m] = 0.5 * img[m] + 0.5 * img[m].mean(axis=0)
    dirt = float(knobs.get("belly_lens_dirt", 0.0)) if camera == "belly" else 0.0
    if dirt > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        cov = np.zeros((h, w))
        for _ in range(int(rng.integers(3, 8))):
            cy, cx, r = rng.uniform(0, h), rng.uniform(0, w), rng.uniform(10, 45)
            cov = np.maximum(cov, np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
        a = (dirt * cov)[..., None]
        img = (1 - a) * img + a * np.array([120.0, 100.0, 75.0])
    sigma = float(knobs.get("sensor_noise_sigma", 0.0))
    if sigma > 0:
        img += rng.normal(0.0, sigma, img.shape)
    out = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    q = int(knobs.get("jpeg_quality", 100))
    if q < 100:
        out = jpeg_roundtrip(out, q)
    return out


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    """JPEG artefacts via Pillow when present (deterministic on one host); unchanged without Pillow."""
    try:
        from PIL import Image
    except Exception:       # pragma: no cover - the sim image ships Pillow
        return rgb
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=int(quality), optimize=False, subsampling=2)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
