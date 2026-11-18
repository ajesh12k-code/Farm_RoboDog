"""Orchard heightfield and ground patches (WP4; ORCHARD-MVP-SPEC 3.6, 4.2). Pure numpy, no MuJoCo.

The terrain is one height grid (``TerrainGrid``) that ``world/modules/terrain.py`` compiles into a MuJoCo
hfield. Every feature is >= 0 m except the pits under the patch boxes, so the visual floor plane can stay at
z ≈ 0 and nothing that is walked on hides under it:

    noise      seeded smooth noise (bilinear on a coarse grid, >= 0); amplitude per zone: 0 in ``flat_zones``
               (home pad, slope cores, patches), ``gentle_amp_m`` in ``gentle_zones`` (the R1 alleys: <= 1°),
               ``amp_m`` elsewhere
    basins     trunk basins: floor at 0 inside ``basin_r_m``, a rim of crest height ``rim_h_m`` (0.10 m deep
               basin, OS 3.6) and crest width ``rim_crest_m`` outside it, half-cosine flanks at ``edge_deg``
    bunds      straight bunds ``h_m`` x crest ``crest_m`` (0.10 x 0.3 m, OS 3.6), flanks at ``edge_deg``
               (30°, the walkable edge of farm_scene BUND_EDGE_DEG, review F7)
    slopes     planar cores tilted ``deg`` along ``rise`` (+x / -x / +y / -y): z = tan(deg) * distance from
               the low edge. Outside the core the feature is ``max(0, top - tan(deg) * distance to the core's
               high edge)``: the fall past the high edge and the side faces are never steeper than ``deg``
               anywhere in the footprint (``SlopeSpec.outer``: the core grown by its run on the sides and past
               the high edge). The 5° headland strip (stowed crossing) and the 10° patch (refusal demo)
    pits       under every mud / sink / flooded-basin patch: the grid is lowered ``depth_m`` and the patch box
               (modules/terrain.py) fills it flush to z = 0 with its own friction and softness

``slope_deg_at`` / ``height_at`` sample the grid bilinearly (MuJoCo triangulates each cell; the difference
is below a millimetre at 0.05 m cells), for placement, the Ready check (local cross-slope) and tests.
"""
import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RES_M = 0.05
SLOPE_CREST_R_M = 0.25      # rounding of the slope features' high-edge ridge (no gradient kink there)


def _rect(z) -> Tuple[float, float, float, float]:
    return float(z["x"][0]), float(z["x"][1]), float(z["y"][0]), float(z["y"][1])


def _cos_step(u: np.ndarray) -> np.ndarray:
    """0 for u <= 0, 1 for u >= 1, half-cosine in between."""
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def square_gradient(z: np.ndarray, k: int, res: float) -> Tuple[np.ndarray, np.ndarray]:
    """(gx, gy) of every k x k cell square of grid z (rows = y), at the square's centre: each component is the
    mean of the two opposite edge differences (the bilinear patch's gradient there). Both components belong to
    the same point; pairing the bottom-edge x difference with the left-edge y difference (a corner stencil)
    reads up to sqrt(2) x the true slope where two faces meet."""
    gx = ((z[:-k, k:] - z[:-k, :-k]) + (z[k:, k:] - z[k:, :-k])) / (2.0 * k * res)
    gy = ((z[k:, :-k] - z[:-k, :-k]) + (z[k:, k:] - z[:-k, k:])) / (2.0 * k * res)
    return gx, gy


def flank_run(h: float, edge_deg: float) -> float:
    """Horizontal run of a half-cosine flank of height h whose steepest slope is edge_deg.

    max slope of h * (1 - cos(pi u / L)) / 2 is pi h / (2 L)  ->  L = pi h / (2 tan(edge))."""
    return math.pi * h / (2.0 * math.tan(math.radians(edge_deg)))


def ridge_profile(dist: np.ndarray, h: float, crest_half: float, edge_deg: float) -> np.ndarray:
    """Height of a straight / ring ridge at distance ``dist`` from its centre line: flat crest of half
    width ``crest_half`` at height h, half-cosine flanks."""
    run = flank_run(h, edge_deg)
    d = np.abs(dist)
    return h * (1.0 - _cos_step((d - crest_half) / run))


@dataclass
class SlopeSpec:
    id: str
    deg: float
    core: Tuple[float, float, float, float]       # x0, x1, y0, y1
    rise: str = "+x"

    @classmethod
    def from_dict(cls, d: dict) -> "SlopeSpec":
        if d.get("rise", "+x") not in ("+x", "-x", "+y", "-y"):
            raise ValueError(f"slope {d.get('id')!r}: rise {d.get('rise')!r} not in +x / -x / +y / -y")
        return cls(d["id"], float(d["deg"]), _rect(d["core"]), d.get("rise", "+x"))

    @property
    def run_m(self) -> float:
        """Core length along the rise: the face reach (top / tan(deg)) on the sides and past the high edge."""
        x0, x1, y0, y1 = self.core
        return (x1 - x0) if self.rise in ("+x", "-x") else (y1 - y0)

    def outer(self) -> Tuple[float, float, float, float]:
        """Footprint (x0, x1, y0, y1): everything the feature raises above 0."""
        x0, x1, y0, y1 = self.core
        L = self.run_m
        if self.rise in ("+x", "-x"):
            return (x0 - (L if self.rise == "-x" else 0.0), x1 + (L if self.rise == "+x" else 0.0), y0 - L, y1 + L)
        return (x0 - L, x1 + L, y0 - (L if self.rise == "-y" else 0.0), y1 + (L if self.rise == "+y" else 0.0))


@dataclass
class PatchSpec:
    id: str
    kind: str                                      # mud | sink | flooded
    center: Tuple[float, float]
    half: Tuple[float, float]                      # box half extents (x, y); flooded basins use r
    depth_m: float = 0.06
    friction: float = 0.3
    penetration_m: float = 0.0                     # sink target (assumption 3-5 cm, OS 3.6)
    r_m: float = 0.0                               # > 0: round patch (flooded basin)

    @classmethod
    def from_dict(cls, d: dict) -> "PatchSpec":
        return cls(d["id"], d["kind"], (float(d["center"][0]), float(d["center"][1])),
                   (float(d.get("half", [0, 0])[0]), float(d.get("half", [0, 0])[1])), float(d.get("depth_m", 0.06)),
                   float(d.get("friction", 0.3)), float(d.get("penetration_m", 0.0)), float(d.get("r_m", 0.0)))


@dataclass
class TerrainConfig:
    extent: Tuple[float, float, float, float]     # x0, x1, y0, y1
    res_m: float = DEFAULT_RES_M
    friction: float = 0.8                          # ground friction (monsoon alleys 0.4)
    noise_amp_m: float = 0.010
    noise_cell_m: float = 0.5
    gentle_amp_m: float = 0.003
    flat_zones: List[Tuple[float, float, float, float]] = dc_field(default_factory=list)
    gentle_zones: List[Tuple[float, float, float, float]] = dc_field(default_factory=list)
    zone_ramp_m: float = 0.5
    rim_h_m: float = 0.10
    rim_crest_m: float = 0.15
    rim_edge_deg: float = 30.0
    flooded_basins: bool = False
    flood: dict = dc_field(default_factory=dict)
    basins: List[Tuple[str, float, float, float]] = dc_field(default_factory=list)   # (tree_id, x, y, r)
    bunds: List[dict] = dc_field(default_factory=list)
    slopes: List[SlopeSpec] = dc_field(default_factory=list)
    patches: List[PatchSpec] = dc_field(default_factory=list)

    @classmethod
    def from_field(cls, field: dict) -> "TerrainConfig":
        tc = field.get("terrain") or {}
        ex = tc.get("extent", {"x": [-6, 16], "y": [-6, 18]})
        noise = tc.get("noise", {})
        bas = tc.get("basins", {})
        cfg = cls(extent=_rect(ex), res_m=float(tc.get("res_m", DEFAULT_RES_M)),
                  friction=float(tc.get("friction", 0.8)),
                  noise_amp_m=float(noise.get("amp_m", 0.010)), noise_cell_m=float(noise.get("cell_m", 0.5)),
                  gentle_amp_m=float(noise.get("gentle_amp_m", 0.003)),
                  flat_zones=[_rect(z) for z in noise.get("flat_zones", [])],
                  gentle_zones=[_rect(z) for z in noise.get("gentle_zones", [])],
                  zone_ramp_m=float(noise.get("zone_ramp_m", 0.5)),
                  rim_h_m=float(bas.get("rim_h_m", 0.10)), rim_crest_m=float(bas.get("rim_crest_m", 0.15)),
                  rim_edge_deg=float(bas.get("edge_deg", 30.0)), flooded_basins=bool(bas.get("flooded", False)),
                  flood=dict(bas.get("flood", {})),
                  bunds=[dict(b) for b in tc.get("bunds", [])],
                  slopes=[SlopeSpec.from_dict(s) for s in tc.get("slopes", [])],
                  patches=[PatchSpec.from_dict(p) for p in tc.get("patches", [])])
        for t in field.get("trees") or []:
            r = t.get("basin_r_m")
            if r:
                cfg.basins.append((t["tree_id"], float(t["geo"]["x"]), float(t["geo"]["y"]), float(r)))
        if cfg.flooded_basins:
            f = cfg.flood
            for tid, x, y, r in cfg.basins:
                cfg.patches.append(PatchSpec(f"flood_{tid}", "flooded", (x, y), (0.0, 0.0),
                                             float(f.get("depth_m", 0.06)), float(f.get("friction", 0.3)),
                                             float(f.get("penetration_m", 0.04)), r_m=r))
        return cfg


class TerrainGrid:
    """Height grid z[row, col] over ``extent``; row i is y0 + i * res, column j is x0 + j * res."""

    def __init__(self, cfg: TerrainConfig, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        x0, x1, y0, y1 = cfg.extent
        self.res = float(cfg.res_m)
        self.ncol = int(round((x1 - x0) / self.res)) + 1
        self.nrow = int(round((y1 - y0) / self.res)) + 1
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x0 + (self.ncol - 1) * self.res, y0 + (self.nrow - 1) * self.res
        xs = self.x0 + self.res * np.arange(self.ncol)
        ys = self.y0 + self.res * np.arange(self.nrow)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.layers: Dict[str, np.ndarray] = {}
        self.z = self._build()

    # ---------------------------------------------------------------- layers
    def _zone_dist(self, rect) -> np.ndarray:
        """Distance (m) from each grid point to an axis-aligned rectangle (0 inside)."""
        x0, x1, y0, y1 = rect
        dx = np.maximum(np.maximum(x0 - self.X, self.X - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - self.Y, self.Y - y1), 0.0)
        return np.hypot(dx, dy)

    def amplitude(self) -> np.ndarray:
        c = self.cfg
        amp = np.full(self.X.shape, c.noise_amp_m)
        ramp = max(c.zone_ramp_m, 1e-6)
        for rect in c.gentle_zones:
            w = np.clip(self._zone_dist(rect) / ramp, 0.0, 1.0)
            amp = np.minimum(amp, c.gentle_amp_m + (c.noise_amp_m - c.gentle_amp_m) * w)
        flats = list(c.flat_zones) + [s.outer() for s in c.slopes]
        for p in c.patches:
            hx, hy = (p.r_m, p.r_m) if p.r_m > 0 else p.half
            flats.append((p.center[0] - hx, p.center[0] + hx, p.center[1] - hy, p.center[1] + hy))
        for rect in flats:
            w = np.clip(self._zone_dist(rect) / ramp, 0.0, 1.0)
            amp = np.minimum(amp, c.noise_amp_m * w)
        return amp

    def _noise(self) -> np.ndarray:
        c = self.cfg
        rng = np.random.default_rng(int(np.uint32(self.seed * 7919 + 4)))
        cell = c.noise_cell_m
        nx = int(math.ceil((self.x1 - self.x0) / cell)) + 2
        ny = int(math.ceil((self.y1 - self.y0) / cell)) + 2
        coarse = rng.uniform(0.0, 1.0, size=(ny, nx))
        u = (self.X - self.x0) / cell
        v = (self.Y - self.y0) / cell
        i0, j0 = np.floor(v).astype(int), np.floor(u).astype(int)
        fu, fv = u - j0, v - i0
        n = (coarse[i0, j0] * (1 - fu) * (1 - fv) + coarse[i0, j0 + 1] * fu * (1 - fv)
             + coarse[i0 + 1, j0] * (1 - fu) * fv + coarse[i0 + 1, j0 + 1] * fu * fv)
        return n * self.amplitude()

    def _basins(self) -> np.ndarray:
        c = self.cfg
        z = np.zeros(self.X.shape)
        half = c.rim_crest_m / 2.0
        for _tid, x, y, r in c.basins:
            rho = np.hypot(self.X - x, self.Y - y)
            z = np.maximum(z, ridge_profile(rho - (r + half), c.rim_h_m, half, c.rim_edge_deg))
        return z

    def _bunds(self) -> np.ndarray:
        z = np.zeros(self.X.shape)
        for b in self.cfg.bunds:
            a, e = np.asarray(b["from"], float), np.asarray(b["to"], float)
            h, crest = float(b.get("h_m", 0.10)), float(b.get("crest_m", 0.30))
            d = e - a
            L = float(np.hypot(*d))
            t = np.clip(((self.X - a[0]) * d[0] + (self.Y - a[1]) * d[1]) / (L * L), 0.0, 1.0)
            dist = np.hypot(self.X - (a[0] + t * d[0]), self.Y - (a[1] + t * d[1]))
            z = np.maximum(z, ridge_profile(dist, h, crest / 2.0, float(b.get("edge_deg", 30.0))))
        return z

    def _slope_one(self, s: SlopeSpec) -> np.ndarray:
        """``max(0, top - g * D)``, D = distance to the core's high edge: the core plane g * run inside the core
        (D = length - run there), and every face outside it is a g-Lipschitz cone / plane, so the slope is at
        most ``deg`` everywhere (the old cosine fall / blend faces reached 15-34°, review)."""
        x0, x1, y0, y1 = s.core
        g = math.tan(math.radians(s.deg))
        if s.rise in ("+x", "-x"):
            along, (a0, a1) = self.X, (x0, x1)
            across, (c0, c1) = self.Y, (y0, y1)
        else:
            along, (a0, a1) = self.Y, (y0, y1)
            across, (c0, c1) = self.X, (x0, x1)
        run = (along - a0) if s.rise.startswith("+") else (a1 - along)
        length = a1 - a0
        e = np.maximum(np.maximum(c0 - across, across - c1), 0.0)        # distance outside the core, across
        D = np.hypot(run - length, e)
        # the ridge along the high edge is rounded over SLOPE_CREST_R_M (parabolic cap, slope g * D / rho): the
        # gradient is continuous, still <= g, and the core is exact up to rho from the high edge
        rho = min(SLOPE_CREST_R_M, length / 4.0)
        drop = np.where(D >= rho, D, rho / 2.0 + D * D / (2.0 * rho))
        return np.maximum(0.0, g * (length - drop))

    def _slopes(self) -> np.ndarray:
        z = np.zeros(self.X.shape)
        for s in self.cfg.slopes:
            z = np.maximum(z, self._slope_one(s))
        return z

    def _pits(self) -> np.ndarray:
        """Negative layer: the hollows under the patch boxes (inset by one cell so the box edge lies on z 0)."""
        z = np.zeros(self.X.shape)
        inset = self.res
        for p in self.cfg.patches:
            if p.r_m > 0:
                inside = np.hypot(self.X - p.center[0], self.Y - p.center[1]) <= p.r_m - inset
            else:
                inside = ((np.abs(self.X - p.center[0]) <= p.half[0] - inset)
                          & (np.abs(self.Y - p.center[1]) <= p.half[1] - inset))
            z = np.where(inside, np.minimum(z, -p.depth_m), z)
        return z

    def _build(self) -> np.ndarray:
        self.layers = {"noise": self._noise(), "basins": self._basins(), "bunds": self._bunds(),
                       "slopes": self._slopes(), "pits": self._pits()}
        pos = self.layers["noise"] + np.maximum(np.maximum(self.layers["basins"], self.layers["bunds"]),
                                                self.layers["slopes"])
        pits = self.layers["pits"]
        # walkable surface: the patch boxes fill their pits flush to z 0
        self.surface = np.where(pits < 0, 0.0, pos)
        return np.where(pits < 0, pits, pos)

    # ---------------------------------------------------------------- queries
    @property
    def zmin(self) -> float:
        return float(self.z.min())

    @property
    def zmax(self) -> float:
        return float(self.z.max())

    def _ij(self, x: float, y: float):
        u = (x - self.x0) / self.res
        v = (y - self.y0) / self.res
        j = int(np.clip(math.floor(u), 0, self.ncol - 2))
        i = int(np.clip(math.floor(v), 0, self.nrow - 2))
        return i, j, u - j, v - i

    def height_at(self, x: float, y: float) -> float:
        """Bilinear ground height (m) at world (x, y); 0 outside the grid."""
        if not (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1):
            return 0.0
        i, j, fu, fv = self._ij(x, y)
        z = self.z
        return float(z[i, j] * (1 - fu) * (1 - fv) + z[i, j + 1] * fu * (1 - fv)
                     + z[i + 1, j] * (1 - fu) * fv + z[i + 1, j + 1] * fu * fv)

    def surface_at(self, x: float, y: float) -> float:
        """Height of what a foot stands on: the grid, or z 0 on a mud / sink / flooded patch box."""
        if not (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1):
            return 0.0
        i, j, fu, fv = self._ij(x, y)
        s = self.surface
        return float(s[i, j] * (1 - fu) * (1 - fv) + s[i, j + 1] * fu * (1 - fv)
                     + s[i + 1, j] * (1 - fu) * fv + s[i + 1, j + 1] * fu * fv)

    def gradient_at(self, x: float, y: float, h: float = 0.1) -> Tuple[float, float]:
        return ((self.height_at(x + h, y) - self.height_at(x - h, y)) / (2 * h),
                (self.height_at(x, y + h) - self.height_at(x, y - h)) / (2 * h))

    def slope_deg_at(self, x: float, y: float, h: float = 0.1) -> float:
        gx, gy = self.gradient_at(x, y, h)
        return math.degrees(math.atan(math.hypot(gx, gy)))

    def cross_slope_deg(self, x: float, y: float, heading_rad: float, h: float = 0.2) -> float:
        """Slope across a heading (deg; + = ground rises to the robot's left)."""
        gx, gy = self.gradient_at(x, y, h)
        lx, ly = -math.sin(heading_rad), math.cos(heading_rad)
        return math.degrees(math.atan(gx * lx + gy * ly))

    def max_slope_deg(self, rect, baseline_m: float = 0.2) -> float:
        """Steepest slope (deg) inside an axis-aligned rect: the gradient of every ``baseline_m`` square of grid
        nodes at its centre (``square_gradient``)."""
        x0, x1, y0, y1 = rect
        k = max(1, int(round(baseline_m / self.res)))
        jm = (self.X[0] >= x0) & (self.X[0] <= x1)
        im = (self.Y[:, 0] >= y0) & (self.Y[:, 0] <= y1)
        sub = self.z[np.ix_(im, jm)]
        if sub.shape[0] <= k or sub.shape[1] <= k:
            return 0.0
        gx, gy = square_gradient(sub, k, self.res)
        return float(np.degrees(np.arctan(np.hypot(gx, gy).max())))

    def hfield_data(self) -> Tuple[np.ndarray, float, float]:
        """(normalised data in [0, 1] row-major, elevation range m, base z m) for a MuJoCo hfield:
        world z = base + data * range."""
        lo, hi = self.zmin, self.zmax
        rng = max(hi - lo, 1e-4)
        return (self.z - lo) / rng, rng, lo

    def summary(self) -> dict:
        return {"nrow": self.nrow, "ncol": self.ncol, "res_m": self.res, "zmin_m": round(self.zmin, 4),
                "zmax_m": round(self.zmax, 4), "n_basins": len(self.cfg.basins), "n_bunds": len(self.cfg.bunds),
                "n_slopes": len(self.cfg.slopes), "n_patches": len(self.cfg.patches)}


_CACHE: Dict[Tuple[str, int], TerrainGrid] = {}


def grid_for(field: dict, seed: Optional[int] = None) -> TerrainGrid:
    """The (cached) grid of a field: every module and the hooks share one per (world, seed, terrain)."""
    import json
    s = int(field.get("seed", 7) if seed is None else seed)
    key = (json.dumps([field.get("world"), field.get("terrain"), field.get("trees")], sort_keys=True), s)
    g = _CACHE.get(key)
    if g is None:
        if len(_CACHE) > 8:
            _CACHE.clear()
        g = _CACHE[key] = TerrainGrid(TerrainConfig.from_field(field), s)
    return g


def rect_of(zone: Sequence[float]) -> Tuple[float, float, float, float]:
    return tuple(float(v) for v in zone)  # type: ignore[return-value]
