"""Mast LiDAR, low safety planes, ray masks, trunk circle fit and LiDAR clusters (WP5; MVP-SPEC 2.4, 5.4;
ORCHARD-MVP-SPEC 2.5, 5.1).

Sensors (``mj_multiRay`` on the live ``MjData`` of the sim process, terrain hfield rays by ``_HfieldCaster``;
``atharv_nandi/sim/hooks/lidar.py`` at 10 Hz):

* **mast LiDAR**: from the ``mast_lidar`` site (ring height, lens 0.98 m), 360 rays x 3 beams at 0, -15 and
  -30 deg (Livox Mid-360 class). The -30 deg beam meets the ground at about 1.7 m.
* **low safety planes**: 2 virtual 270 deg scanners at 0.18 m world height on the hull corners (front-left
  looking out at +45 deg, rear-right at -135 deg; 270 rays each, 540 in total). AGV practice, assumption.

Noise: Gaussian sigma 2 cm on the range, min range 0.1 m, max range 8 m (MVP-SPEC 2.4). ``noise=False`` for the
range-accuracy tests.

**Ray groups (OS 2.5).** Geom groups: 0 tree wood and ground, 1 fruit and actors, 2 robot, 3 foliage, 4 decals,
5 query-only (foliage envelope, rope lines, laterals, robot mass-only geoms). Safety rays use ``SAFETY_GROUPS``
{0, 1} and never see foliage; coverage rays use ``COVERAGE_GROUPS`` {0, 1, 3}, so canopy self-occlusion counts.
The robot (group 2 / 5) is never hit: rays pass through it. Fidelity register: the sim LiDAR sees through foliage
on the safety mask; a real LiDAR does not.

**Trunk fit (OS 2.5).** A mango trunk is 0.35 H tall (about 0.77 m on a 2.2 m tree) and scaffolds start at about
0.6 m, so the 0.98 m mast LiDAR hits scaffolds or canopy, not the trunk. ``TrunkFitter`` fits a circle to the
**low-plane** hits at 0.18 m around the expected trunk (registry prior, gate 0.5 m): the cluster nearest the prior
(sprouts, props and weeds fall in other clusters), then a fixed-radius geometric least-squares fit with the
registry ``trunk_r_m`` (seeded by a sensor-facing fixed-radius RANSAC and trimmed at ``trim_m`` 4 cm: sprouts that
touch the trunk at 0.18 m, e.g. with the robot on a basin rim, join its cluster) (at 3 m a 0.11 m trunk spans only
about 2 rays of a 1 deg scanner per scan, so the fitter
accumulates ``window`` scans; with sigma 2 cm a free-radius fit is biased by 2-4 cm, measured, so the free fit
(RANSAC + geometric least squares, wide arc required) runs only when no registry radius is given). If the fit
fails (too few points, basin edge, residual too high) the orbit falls back to ``fit_canopy_edge``: the canopy-edge
ellipse of the coverage-mask mast hits (direct ellipse fit when the visible arc is wide, else a circle with the
registry ``canopy_r_m``). Every result carries its ``method`` so R_meas is labelled. ``RingDepthBand`` / ``band_cross_check`` cross-check
the low-plane fit on a ring-depth band at 0.2-0.5 m (OS 2.5), see the section at the end of this module.

**Clusters (MVP-SPEC 5.4).** ``nonmap_clusters`` groups the safety-mask hits that are not map geoms (map = geoms
welded to the world that are not mocap: ground, terrain, trees, fixtures; plus the WP23 fruit pool bodies, which are
registry fruit, not person candidates) within 8 m into xy clusters (0.3 m gap);
a cluster whose top is below 0.15 m above the local ground (a crop, debris) is not a person candidate.
Each cluster is an axis-aligned box whose z range is widened from the ground to at least ``PERSON_TOP_M`` (2.0 m,
assumption: a LiDAR only samples a person at its beam heights). The privacy filter projects these boxes into every
camera frame as the detector mask. Classifying by geom identity is the assisted shortcut ``map_masked_lidar``
(OS 5.2); a physics-mode map mask would come from the field registry.

No ROS here; MuJoCo only for the ray casts.
"""
import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

SAFETY_GROUPS = (1, 1, 0, 0, 0, 0)       # {0, 1}: wood/ground, fruit/actors; never foliage
COVERAGE_GROUPS = (1, 1, 0, 1, 0, 0)     # {0, 1, 3}: plus foliage, so canopy self-occlusion counts
MAST_SITE = "mast_lidar"
TRUNK_BODY = "trunk"
MAST_BEAMS_DEG = (0.0, -15.0, -30.0)
MAST_RAYS_PER_BEAM = 360
LOW_PLANE_RAYS = 270
LOW_PLANE_FOV_DEG = 270.0
LOW_PLANE_Z_M = 0.18
NOMINAL_TRUNK_Z_M = 0.34                 # trunk body height at the home pose (low-plane mount offsets)
# Hull corners (trunk frame, trunk box half-size 0.34 x 0.14): front-left looks out at +45 deg, rear-right at -135.
LOW_PLANES = (
    {"name": "low_fl", "pos_xy": (0.34, 0.14), "yaw_deg": 45.0},
    {"name": "low_rr", "pos_xy": (-0.34, -0.14), "yaw_deg": -135.0},
)
RANGE_MIN_M = 0.10
RANGE_MAX_M = 8.0
NOISE_SIGMA_M = 0.02
CLUSTER_GAP_M = 0.30
CLUSTER_RANGE_M = 8.0
PERSON_TOP_M = 2.0
MIN_TOP_M = 0.15                         # lower clusters (crops on the ground, debris) are not person candidates


@dataclass
class Scan:
    """One scanner's sweep in the world frame. ``ranges`` is ``inf`` where there is no valid return."""
    name: str
    origin: np.ndarray
    dirs: np.ndarray
    ranges: np.ndarray
    geom: np.ndarray
    angles: np.ndarray            # scanner-frame azimuth per ray (rad)
    beam_deg: Optional[float] = None
    t: float = 0.0
    beams: Optional[np.ndarray] = None   # per-ray beam elevation (deg), mast only

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.ranges)

    def points(self) -> np.ndarray:
        v = self.valid
        return self.origin[None, :] + self.dirs[v] * self.ranges[v, None]

    def hit_geoms(self) -> np.ndarray:
        return self.geom[self.valid]


def _mast_dirs_local() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dirs, az, beam = [], [], []
    for el in MAST_BEAMS_DEG:
        e = math.radians(el)
        for k in range(MAST_RAYS_PER_BEAM):
            a = 2.0 * math.pi * k / MAST_RAYS_PER_BEAM - math.pi
            dirs.append([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            az.append(a)
            beam.append(el)
    return np.asarray(dirs), np.asarray(az), np.asarray(beam)


def _plane_dirs_local(yaw_deg: float, n: int = LOW_PLANE_RAYS, fov_deg: float = LOW_PLANE_FOV_DEG):
    half = math.radians(fov_deg) / 2.0
    az = np.linspace(-half, half, n) + math.radians(yaw_deg)
    return np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], axis=1), az


HF_BLOCK_CELLS = 8    # coarse level block (cells)
HF_STEP_FRAC = 0.25  # fine step / cell size (0.5 missed about 3 in 100,000 crest-grazing rays; 0.25 none, measured)


class _HfieldCaster:
    """Height-field rays without ``mj_multiRay``. MuJoCo's ray / hfield test walks every cell of the ray's xy box,
    so on the orchard_demo terrain (501 x 481 cells) one sweep cost about 67 ms instead of about 1 ms (measured,
    2-CPU sim container). Here every ray that can reach the terrain band is marched in two levels: half-block steps
    against a dilated block-max grid flag the intervals that can touch the surface, then only those get quarter-cell
    steps against MuJoCo's own two triangles per cell and a bisection (same returns as ``mj_multiRay`` to about
    0.05 mm, measured; about 4 ms per sweep). The terrain is the compiled hfield and is treated as static (the
    block grid is built once). ``mj_multiRay`` then runs on a private copy of the model with the hfield
    moved out of the requested groups; the copy's ray fields are re-synced from the live model on every cast, so
    the live model is never written. Only an axis-aligned hfield on a static body is handled; any other hfield
    stays on the plain ``mj_multiRay`` path."""

    FIELDS = ("geom_rgba", "geom_matid", "mat_rgba", "geom_group", "geom_size", "geom_rbound", "geom_aabb",
              "geom_type", "geom_dataid", "geom_bodyid")

    def __init__(self, model: mujoco.MjModel, d0: mujoco.MjData, gid: int):
        self.m, self.gid = model, int(gid)
        hid = int(model.geom_dataid[gid])
        self.nr, self.nc = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
        adr = int(model.hfield_adr[hid])
        self.grid = model.hfield_data[adr:adr + self.nr * self.nc].reshape(self.nr, self.nc)   # live view
        self.size = model.hfield_size[hid]                                                       # live view
        self.group = int(model.geom_group[gid])
        self.copy = copy.copy(model)
        # static terrain (compiled hfield): min / max and a dilated block-max grid (normalised heights) for the
        # coarse level, padded with a -inf ring so rays entering from outside the box are still flagged
        g = np.asarray(self.grid, dtype=np.float64)
        self.hmin, self.hmax = float(g.min()), float(g.max())
        cx, cy = 2.0 * float(self.size[0]) / (self.nc - 1), 2.0 * float(self.size[1]) / (self.nr - 1)
        self.fine = HF_STEP_FRAC * min(cx, cy)
        self.inv_cx, self.inv_cy = 1.0 / cx, 1.0 / cy
        self.flat = self.grid.reshape(-1)                                   # live view
        B = HF_BLOCK_CELLS
        nbr, nbc = int(math.ceil((self.nr - 1) / B)), int(math.ceil((self.nc - 1) / B))
        bm = np.full((nbr + 2, nbc + 2), -np.inf)
        for i in range(nbr):
            for j in range(nbc):
                bm[i + 1, j + 1] = g[i * B:min((i + 1) * B, self.nr - 1) + 1,
                                     j * B:min((j + 1) * B, self.nc - 1) + 1].max()
        dm = bm.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                dm = np.maximum(dm, np.roll(np.roll(bm, di, 0), dj, 1))
        self.dmax = dm
        self.block_x, self.block_y = B * cx, B * cy
        self.block_m = min(self.block_x, self.block_y)

    def _sync(self, groups) -> Optional[np.ndarray]:
        """Copy the ray fields, then put the hfield in a group the cast does not request (None: no free group)."""
        free = [k for k in range(len(groups)) if not groups[k]]
        if not free:
            return None
        for f in self.FIELDS:
            getattr(self.copy, f)[...] = getattr(self.m, f)
        self.copy.geom_group[self.gid] = free[0]
        return self.copy

    def _height(self, d, x, y):
        p = d.geom_xpos[self.gid]
        u = (x - (p[0] - self.size[0])) * self.inv_cx
        v = (y - (p[1] - self.size[1])) * self.inv_cy
        inside = (u >= 0) & (u <= self.nc - 1) & (v >= 0) & (v <= self.nr - 1)
        c0 = np.clip(u.astype(np.int64), 0, self.nc - 2)
        r0 = np.clip(v.astype(np.int64), 0, self.nr - 2)
        fu, fv = u - c0, v - r0
        k = r0 * self.nc + c0
        f = self.flat
        h00, h01, h10, h11 = f.take(k), f.take(k + 1), f.take(k + self.nc), f.take(k + self.nc + 1)
        # MuJoCo's two triangles per cell (diagonal (r, c)-(r+1, c+1)): matches mj_ray to 0.05 mm (measured)
        h = np.where(fv >= fu, h00 + (h11 - h10) * fu + (h10 - h00) * fv, h00 + (h01 - h00) * fu + (h11 - h01) * fv)
        return np.where(inside, p[2] + h * self.size[2], -np.inf)

    def cast(self, d, origin, dirs, rmax) -> np.ndarray:
        """Distance to the terrain surface per ray (inf: no hit within ``rmax``, a scalar or one per ray).
        Two levels: a coarse march (half a block per step) against the dilated block-max heights flags the only
        intervals where the ray can be at or below the surface; just those get the fine march and bisection."""
        n = len(dirs)
        out = np.full(n, np.inf)
        p = d.geom_xpos[self.gid]
        sx, sy, sz = float(self.size[0]), float(self.size[1]), float(self.size[2])
        zlo, zhi = p[2] + self.hmin * sz - 1e-6, p[2] + self.hmax * sz + 1e-6
        oz, dz = float(origin[2]), dirs[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (zhi - oz) / dz, (zlo - oz) / dz
        flat = np.abs(dz) < 1e-9
        rmax = np.broadcast_to(np.asarray(rmax, dtype=np.float64), (n,))
        t0 = np.where(flat, 0.0, np.minimum(ta, tb))
        t1 = np.where(flat, np.where((oz >= zlo) & (oz <= zhi), rmax, -1.0), np.maximum(ta, tb))
        t0, t1 = np.maximum(t0, 0.0), np.minimum(t1, rmax)
        idx = np.nonzero(t1 >= t0)[0]
        if len(idx) == 0:
            return out
        # coarse level: half-block steps; any surface point an interval can reach lies in the dilated block max
        cs = 0.5 * self.block_m
        kc = int(math.ceil(float((t1[idx] - t0[idx]).max()) / cs))
        Tc = np.minimum(t0[idx, None] + cs * np.arange(kc + 1)[None, :], t1[idx, None])
        D = dirs[idx]
        X = origin[0] + Tc * D[:, 0:1]
        Y = origin[1] + Tc * D[:, 1:2]
        Z = origin[2] + Tc * D[:, 2:3]
        bc = np.floor((X - p[0] + sx) / self.block_x).astype(np.int64) + 1     # +1: the padded ring
        br = np.floor((Y - p[1] + sy) / self.block_y).astype(np.int64) + 1
        ok = (bc >= 0) & (bc < self.dmax.shape[1]) & (br >= 0) & (br < self.dmax.shape[0])
        bound = np.where(ok, self.dmax[np.clip(br, 0, self.dmax.shape[0] - 1), np.clip(bc, 0, self.dmax.shape[1] - 1)],
                         -np.inf) * sz + p[2]
        zmin = np.minimum(Z[:, :-1], Z[:, 1:])
        flag = (zmin <= bound[:, :-1]) & (Tc[:, 1:] > Tc[:, :-1] - 1e-12)
        flag[:, 0] |= (Z[:, 0] <= bound[:, 0])
        rows, cols = np.nonzero(flag)                 # row-major: per ray in t order
        if len(rows) == 0:
            return out
        # fine level, in rounds: each round fine-samples only the nearest not yet tested flagged interval of every
        # unresolved ray (a ray that dips into the ground flags every interval behind its first hit)
        m_ = int(math.ceil(cs / self.fine)) + 1
        s = np.linspace(0.0, 1.0, m_)[None, :]
        start = np.r_[0, np.nonzero(np.diff(rows))[0] + 1]
        rank = np.arange(len(rows)) - np.repeat(start, np.diff(np.r_[start, len(rows)]))
        done = np.zeros(len(idx), dtype=bool)
        hit_r, hit_hi, hit_lo = [], [], []
        for k in range(int(rank.max()) + 1):
            sel = np.nonzero((rank == k) & ~done[rows])[0]
            if len(sel) == 0:
                continue
            rr, cc = rows[sel], cols[sel]
            ta_, tb_ = Tc[rr, cc][:, None], Tc[rr, np.minimum(cc + 1, kc)][:, None]
            T = ta_ + (tb_ - ta_) * s
            Dr = D[rr]
            below = (origin[2] + T * Dr[:, 2:3]) <= self._height(d, origin[0] + T * Dr[:, 0:1],
                                                                 origin[1] + T * Dr[:, 1:2])
            anyb = below.any(axis=1)
            if not anyb.any():
                continue
            pc = below[anyb].argmax(axis=1)
            Tb = T[anyb]
            j = np.arange(len(pc))
            hit_r.append(rr[anyb])
            hit_hi.append(Tb[j, pc])
            hit_lo.append(np.where(pc > 0, Tb[j, np.maximum(pc - 1, 0)], Tb[j, pc]))
            done[rr[anyb]] = True
        if not hit_r:
            return out
        pr, hi, lo = np.concatenate(hit_r), np.concatenate(hit_hi), np.concatenate(hit_lo)
        Dh = D[pr]
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            q = origin[None, :] + mid[:, None] * Dh
            b = q[:, 2] <= self._height(d, q[:, 0], q[:, 1])
            hi, lo = np.where(b, mid, hi), np.where(b, lo, mid)
        out[idx[pr]] = hi
        return out


class LidarModel:
    """Ray casts for the mast LiDAR and the two low planes on a live model (sim process)."""

    def __init__(self, model: mujoco.MjModel, seed: int = 7, noise_sigma: float = NOISE_SIGMA_M,
                 range_max: float = RANGE_MAX_M, range_min: float = RANGE_MIN_M,
                 low_plane_z: float = LOW_PLANE_Z_M, nominal_trunk_z: float = NOMINAL_TRUNK_Z_M):
        self.m = model
        self.trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TRUNK_BODY)
        if self.trunk < 0:
            raise RuntimeError("lidar_model: no trunk body")
        self.mast_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, MAST_SITE)
        self.rng = np.random.default_rng(seed)
        self.noise_sigma = float(noise_sigma)
        self.range_max, self.range_min = float(range_max), float(range_min)
        self.mast_local, self.mast_az, self.mast_beam = _mast_dirs_local()
        dz = float(low_plane_z) - float(nominal_trunk_z)
        self.planes = []
        for p in LOW_PLANES:
            d, az = _plane_dirs_local(p["yaw_deg"])
            self.planes.append({"name": p["name"], "pos": np.array([p["pos_xy"][0], p["pos_xy"][1], dz]),
                                "dirs": d, "az": az})
        self.map_geom = map_geoms(model)
        self._hf: List[_HfieldCaster] = []
        d0 = mujoco.MjData(model)
        mujoco.mj_kinematics(model, d0)
        for g in range(model.ngeom):
            if (model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD and model.body_weldid[model.geom_bodyid[g]] == 0
                    and np.allclose(d0.geom_xmat[g], np.eye(3).ravel(), atol=1e-9)):
                self._hf.append(_HfieldCaster(model, d0, g))
        if len(self._hf) > 1:          # one private model copy handles one hfield; more stay on mj_multiRay
            self._hf = []

    @property
    def n_rays(self) -> int:
        return (len(self.mast_local) if self.mast_site >= 0 else 0) + sum(len(p["dirs"]) for p in self.planes)

    def _cast(self, d: mujoco.MjData, origin, dirs_world: np.ndarray, groups, noise: bool):
        n = len(dirs_world)
        gid = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n, dtype=np.float64)
        dirs_world = np.ascontiguousarray(dirs_world, dtype=np.float64)
        vec = dirs_world.ravel()
        origin = np.asarray(origin, dtype=np.float64)
        grp = np.asarray(groups, dtype=np.uint8)
        hf = self._hf[0] if self._hf and grp[self._hf[0].group] else None
        mray = hf._sync(grp) if hf is not None else None
        if mray is None:
            hf = None
        mujoco.mj_multiRay(mray if hf is not None else self.m, d, origin, vec, grp,
                           1, -1, gid, dist, n, self.range_max)
        r = dist.copy()
        r[gid < 0] = np.inf
        if hf is not None:
            rh = hf.cast(d, origin, dirs_world, np.minimum(r, self.range_max))   # march only up to the first hit
            nearer = rh < r
            r[nearer] = rh[nearer]
            gid[nearer] = hf.gid
        if noise and self.noise_sigma > 0:
            hit = gid >= 0
            r[hit] = r[hit] + self.rng.normal(0.0, self.noise_sigma, int(hit.sum()))
        r[(r < self.range_min) | (r > self.range_max)] = np.inf
        gid[~np.isfinite(r)] = -1
        return r, gid

    def sweep(self, d: mujoco.MjData, groups: Sequence[int] = SAFETY_GROUPS, noise: bool = True,
              mast: bool = True, low: bool = True) -> Dict[str, Scan]:
        R = d.xmat[self.trunk].reshape(3, 3)
        p0 = d.xpos[self.trunk]
        out: Dict[str, Scan] = {}
        if mast and self.mast_site >= 0:
            org = d.site_xpos[self.mast_site].copy()
            dw = self.mast_local @ R.T
            r, g = self._cast(d, org, dw, groups, noise)
            out["mast"] = Scan("mast", org, dw, r, g, self.mast_az.copy(), None, float(d.time), self.mast_beam)
        if low:
            for p in self.planes:
                org = p0 + R @ p["pos"]
                dw = p["dirs"] @ R.T
                r, g = self._cast(d, org, dw, groups, noise)
                out[p["name"]] = Scan(p["name"], org, dw, r, g, p["az"].copy(), 0.0, float(d.time))
        return out


# ---------------------------------------------------------------------------------------------- map / clusters
POOL_BODY_RE = r"pool_\d+$"            # WP23 fruit proxy pool slots (world/modules/fruit_pool.py names)


def registry_free_bodies(model: mujoco.MjModel) -> np.ndarray:
    """bool per body: a free body the field registry accounts for, so never a person candidate: the WP23 fruit pool
    slots (``pool_<k>``: a target fruit on the tree during HARVEST / PICK, in the gripper, in the tray or parked)."""
    import re
    rx = re.compile(POOL_BODY_RE)
    return np.array([bool(rx.match(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""))
                     for b in range(model.nbody)], dtype=bool)


def map_geoms(model: mujoco.MjModel) -> np.ndarray:
    """bool per geom: a map geom (welded to the world and not on a mocap body: ground, terrain, trees, fixtures) or
    a registry fruit body (``registry_free_bodies``: a pooled fruit hanging on the tree is still the tree's fruit)."""
    if model.ngeom == 0:
        return np.zeros(0, dtype=bool)
    body = np.asarray(model.geom_bodyid)
    weld = np.asarray(model.body_weldid)[body]
    mocap = np.asarray(model.body_mocapid)[body] >= 0
    return ((weld == 0) & ~mocap) | registry_free_bodies(model)[body]


@dataclass
class Cluster:
    lo: np.ndarray            # world AABB min (x, y, z)
    hi: np.ndarray            # world AABB max
    n: int
    centroid: np.ndarray
    raw_lo: np.ndarray = field(default_factory=lambda: np.zeros(3))
    raw_hi: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def corners(self) -> np.ndarray:
        lo, hi = self.lo, self.hi
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    def as_dict(self) -> dict:
        return {"lo": [round(float(v), 3) for v in self.lo], "hi": [round(float(v), 3) for v in self.hi],
                "n": int(self.n), "c": [round(float(v), 3) for v in self.centroid]}

    @staticmethod
    def from_dict(dct: dict) -> "Cluster":
        lo, hi = np.asarray(dct["lo"], dtype=float), np.asarray(dct["hi"], dtype=float)
        c = np.asarray(dct.get("c", (lo + hi) / 2.0), dtype=float)
        return Cluster(lo, hi, int(dct.get("n", 1)), c, lo.copy(), hi.copy())


def cluster_xy(points: np.ndarray, gap: float = CLUSTER_GAP_M) -> List[np.ndarray]:
    """Single-linkage clusters in xy (indices), gap ``gap``. O(n^2) on at most a few hundred points."""
    n = len(points)
    if n == 0:
        return []
    xy = np.asarray(points)[:, :2]
    labels = -np.ones(n, dtype=int)
    cur = 0
    for i in range(n):
        if labels[i] >= 0:
            continue
        labels[i] = cur
        stack = [i]
        while stack:
            j = stack.pop()
            near = np.nonzero((labels < 0) & (np.sum((xy - xy[j]) ** 2, axis=1) <= gap * gap))[0]
            labels[near] = cur
            stack.extend(near.tolist())
        cur += 1
    return [np.nonzero(labels == k)[0] for k in range(cur)]


def nonmap_clusters(scans: Iterable[Scan], map_mask: np.ndarray, sensor_xy: Optional[Sequence[float]] = None,
                    max_range: float = CLUSTER_RANGE_M, gap: float = CLUSTER_GAP_M, ground_z: float = 0.0,
                    person_top: float = PERSON_TOP_M, min_top: float = MIN_TOP_M) -> List[Cluster]:
    """Clusters of the non-map hits within ``max_range`` (xy) of ``sensor_xy``; boxes widened to a person height.
    A cluster whose highest hit is below ``ground_z + min_top`` (a crop on the ground, debris) is not a person
    candidate; a lying person reaches the 0.18 m low planes."""
    pts = []
    for s in scans:
        v = s.valid
        if not v.any():
            continue
        p = s.origin[None, :] + s.dirs[v] * s.ranges[v, None]
        g = s.geom[v]
        keep = ~map_mask[g]
        pts.append(p[keep])
    if not pts:
        return []
    P = np.concatenate(pts, axis=0)
    if sensor_xy is not None and len(P):
        P = P[np.hypot(P[:, 0] - sensor_xy[0], P[:, 1] - sensor_xy[1]) <= max_range]
    out = []
    for idx in cluster_xy(P, gap):
        q = P[idx]
        lo, hi = q.min(axis=0), q.max(axis=0)
        if hi[2] < ground_z + min_top:
            continue
        blo, bhi = lo.copy(), hi.copy()
        blo[2] = min(lo[2], ground_z)
        bhi[2] = max(hi[2], ground_z + person_top)
        out.append(Cluster(blo, bhi, int(len(idx)), q.mean(axis=0), lo, hi))
    return out


# ---------------------------------------------------------------------------------------------- circle fits
def fit_circle_algebraic(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Kasa fit: centre (2,), radius. Needs >= 3 non-collinear points."""
    xy = np.asarray(xy, dtype=np.float64)
    A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
    b = np.sum(xy ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = math.sqrt(max(0.0, float(sol[2] + c @ c)))
    return c, r


def fit_circle_geometric(xy: np.ndarray, c0: Sequence[float], r0: float, fixed_r: Optional[float] = None,
                         iters: int = 30) -> Tuple[np.ndarray, float, float]:
    """Gauss-Newton on sum(|p - c| - r)^2. ``fixed_r`` keeps the radius. Returns (centre, radius, rms)."""
    xy = np.asarray(xy, dtype=np.float64)
    c = np.asarray(c0, dtype=np.float64).copy()
    r = float(fixed_r if fixed_r is not None else r0)
    for _ in range(iters):
        dvec = xy - c
        dist = np.linalg.norm(dvec, axis=1)
        dist = np.maximum(dist, 1e-9)
        res = dist - r
        J = np.column_stack([-dvec[:, 0] / dist, -dvec[:, 1] / dist] + ([] if fixed_r is not None
                                                                      else [-np.ones(len(xy))]))
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step[:2]
        if fixed_r is None:
            r = float(r + step[2])
        if float(np.max(np.abs(step))) < 1e-10:
            break
    rms = float(np.sqrt(np.mean((np.linalg.norm(xy - c, axis=1) - r) ** 2)))
    return c, r, rms


def arc_span_deg(xy: np.ndarray, c: Sequence[float]) -> float:
    """Angular span (deg) of the points around ``c`` (the largest gap removed)."""
    a = np.sort(np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0]))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, a[:1] + 2 * math.pi]))
    return math.degrees(2 * math.pi - float(gaps.max()))


@dataclass
class TrunkFit:
    ok: bool
    method: str
    centre: Optional[np.ndarray] = None
    radius: Optional[float] = None
    rms: Optional[float] = None
    n: int = 0
    arc_deg: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict:
        return {"ok": self.ok, "method": self.method, "reason": self.reason, "n": self.n,
                "arc_deg": round(self.arc_deg, 1),
                "centre": None if self.centre is None else [round(float(v), 4) for v in self.centre],
                "radius": None if self.radius is None else round(float(self.radius), 4),
                "rms": None if self.rms is None else round(float(self.rms), 4)}


class TrunkFitter:
    """Circle fit on the 0.18 m low-plane hits around a registry prior, accumulated over ``window`` scans."""

    def __init__(self, window: int = 10, gate_m: float = 0.5, r_bounds: Tuple[float, float] = (0.03, 0.25),
                 cluster_gap_m: float = 0.08, min_points: int = 3, free_arc_deg: float = 60.0,
                 max_rms_m: float = 0.04, trim_m: float = 0.04, seed: int = 7):
        self.window = int(window)
        self.gate_m = float(gate_m)
        self.r_bounds = r_bounds
        self.cluster_gap_m = float(cluster_gap_m)
        self.min_points = int(min_points)
        self.free_arc_deg = float(free_arc_deg)
        self.max_rms_m = float(max_rms_m)
        self.trim_m = float(trim_m)
        self.rng = np.random.default_rng(seed)
        self._scans: List[Tuple[np.ndarray, np.ndarray]] = []
        self._rays: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []   # (origin xy, unit dir xy, range)

    def reset(self):
        self._scans = []
        self._rays = []

    def add(self, points_world: np.ndarray, origins_world: Optional[np.ndarray] = None):
        """Add one sweep's low-plane hit points (world frame, N x 3 or N x 2) and, optionally, each point's sensor
        origin (N x 3 or 3,): the fixed-radius fit then starts behind the visible surface, away from the sensor."""
        p = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
        if p.size == 0:
            p = np.zeros((0, 2))
        ray = np.full((len(p), 2), np.nan)
        if origins_world is not None and len(p):
            o = np.broadcast_to(np.asarray(origins_world, dtype=np.float64)[..., :2], (len(p), 2))
            v = p[:, :2] - o
            ray = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        self._scans.append((p[:, :2].copy(), ray))
        if len(self._scans) > self.window:
            self._scans = self._scans[-self.window:]

    def add_scans(self, scans: Dict[str, Scan]):
        """Add one sweep of the low planes: the hits, and every ray as free space up to its range (the free-space
        constraint keeps the fitted circle off the rays that passed the trunk)."""
        pts, org, O, D, R = [], [], [], [], []
        for k, s in scans.items():
            if k == "mast":
                continue
            q = s.points()
            pts.append(q)
            org.append(np.broadcast_to(s.origin, q.shape))
            d2 = s.dirs[:, :2]
            n2 = np.maximum(np.linalg.norm(d2, axis=1), 1e-9)
            O.append(np.broadcast_to(s.origin[:2], d2.shape))
            D.append(d2 / n2[:, None])
            R.append(np.where(np.isfinite(s.ranges), s.ranges, RANGE_MAX_M) * n2)
        if pts:
            self.add(np.concatenate(pts, axis=0), np.concatenate(org, axis=0))
            self._rays.append((np.concatenate(O), np.concatenate(D), np.concatenate(R)))
            if len(self._rays) > self.window:
                self._rays = self._rays[-self.window:]
        else:
            self.add(np.zeros((0, 3)))

    def _near(self, prior_xy: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        if not self._scans:
            return np.zeros((0, 2)), np.zeros((0, 2))
        P = np.concatenate([a for a, _ in self._scans], axis=0)
        V = np.concatenate([b for _, b in self._scans], axis=0)
        near = np.hypot(P[:, 0] - prior_xy[0], P[:, 1] - prior_xy[1]) <= self.gate_m
        return P[near], V[near]

    def points_near(self, prior_xy: Sequence[float]) -> np.ndarray:
        return self._near(prior_xy)[0]

    def fit(self, prior_xy: Sequence[float], prior_r: Optional[float] = None) -> TrunkFit:
        P, V = self._near(prior_xy)
        if len(P) < self.min_points:
            return TrunkFit(False, "low_plane_circle", n=int(len(P)), reason="too_few_points")
        # keep the cluster nearest the prior (sprouts, a prop or a weed form other clusters)
        groups = cluster_xy(np.column_stack([P, np.zeros(len(P))]), self.cluster_gap_m)
        best = min(groups, key=lambda idx: float(np.min(np.hypot(P[idx, 0] - prior_xy[0], P[idx, 1] - prior_xy[1]))))
        Q, QV = P[best], V[best]
        if len(Q) < self.min_points:
            return TrunkFit(False, "low_plane_circle", n=int(len(Q)), reason="too_few_points")
        lo, hi = self.r_bounds
        free = None
        # A free-radius fit only without a registry radius: with sigma 2 cm on a 5.5 cm trunk it is biased (the
        # radius shrinks and the centre moves by 2-4 cm, measured); the fixed-radius fit is not.
        if prior_r is None and len(Q) >= 5:
            c, r = self._ransac(Q)
            if c is not None and lo <= r <= hi:
                c, r, rms = fit_circle_geometric(Q, c, r)
                span = arc_span_deg(Q, c)
                if lo <= r <= hi and rms <= self.max_rms_m and span >= self.free_arc_deg:
                    free = TrunkFit(True, "low_plane_circle", c, r, rms, int(len(Q)), span)
        if free is not None:
            return free
        if prior_r is None:
            return TrunkFit(False, "low_plane_circle", n=int(len(Q)), reason="no_free_fit_no_radius_prior")
        # fixed-radius fit: start behind the visible surface (along the mean sensor ray; else toward the prior),
        # so Gauss-Newton cannot settle on the mirrored circle in front of the arc
        m = Q.mean(axis=0)
        okv = np.all(np.isfinite(QV), axis=1)
        u = QV[okv].mean(axis=0) if okv.any() else np.asarray(prior_xy, dtype=float) - m
        nu = float(np.linalg.norm(u))
        c0 = m + (u / nu * prior_r if nu > 1e-9 else np.zeros(2))
        c, r, rms = fit_circle_geometric(Q, c0, prior_r, fixed_r=float(prior_r))
        if self.trim_m > 0 and len(Q) >= 2 * self.min_points and nu > 1e-9:
            # Pruning sprouts and stubs touch the trunk at 0.18 m (one cluster; a robot on a basin rim puts the
            # plane through the sprout roots) and pull the least-squares circle 3-10 cm off. Seed from the
            # fixed-radius RANSAC circle with the most hits that face their own sensor ray (a hit on the far side of
            # a circle cannot be seen, so the circle whose back runs along the trunk front and the sprout loses),
            # then refit on the hits within trim_m of it only.
            cr = _ransac_fixed_r_facing(Q, QV, float(prior_r), self.trim_m, self.rng)
            if cr is not None:
                ct, _, _, keep = _trimmed_circle(Q, cr, float(prior_r), True, self.trim_m)
                if ct is not None and keep.sum() >= max(self.min_points, 0.6 * len(Q)):
                    c, Q = ct, Q[keep]
        if self._rays:
            c = self._free_space_refine(Q, c, float(prior_r))
        rms = float(np.sqrt(np.mean((np.linalg.norm(Q - c, axis=1) - r) ** 2)))
        if rms > self.max_rms_m:
            return TrunkFit(False, "low_plane_circle_fixed_r", c, r, rms, int(len(Q)), arc_span_deg(Q, c),
                            reason="residual_too_high")
        return TrunkFit(True, "low_plane_circle_fixed_r", c, r, rms, int(len(Q)), arc_span_deg(Q, c))

    def _free_space_refine(self, Q: np.ndarray, c0: np.ndarray, r: float, sigma: float = NOISE_SIGMA_M,
                           half: float = 0.06) -> np.ndarray:
        """Coarse-to-fine grid search around ``c0`` for the centre that best fits the hits (|p - c| = r) while no ray
        passes through the circle before its own return (free space). At 3 m a 1 deg scanner puts only ~2 rays on a
        0.11 m trunk, so the hits alone leave the centre loosely placed across the rays; the free rays on both sides
        bound it."""
        O = np.concatenate([o for o, _, _ in self._rays])
        D = np.concatenate([dd for _, dd, _ in self._rays])
        R = np.concatenate([rr for _, _, rr in self._rays])
        # keep the rays that pass near c0
        w = c0[None, :] - O
        tpar = np.clip(np.sum(w * D, axis=1), 0.0, None)
        near = np.linalg.norm(O + D * tpar[:, None] - c0[None, :], axis=1) <= r + 2 * half
        O, D, R = O[near], D[near], R[near]
        c = np.asarray(c0, dtype=np.float64)
        for h, n in ((half, 31), (half / 10.0, 21)):
            g = np.linspace(-h, h, n)
            C = c[None, :] + np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)
            hit = np.linalg.norm(Q[None, :, :] - C[:, None, :], axis=2) - r
            cost = np.sum(hit ** 2, axis=1) / sigma ** 2
            if len(O):
                wv = C[:, None, :] - O[None, :, :]
                t = np.sum(wv * D[None, :, :], axis=2)
                # free space ends one sigma short of the measured return (a hit ray grazes its own circle)
                t = np.clip(t, 0.0, np.maximum(R[None, :] - 2.0 * sigma, 0.0))
                dist = np.linalg.norm(O[None, :, :] + D[None, :, :] * t[:, :, None] - C[:, None, :], axis=2)
                pen = np.clip(r - dist, 0.0, None)
                cost = cost + np.sum(pen ** 2, axis=1) / (0.25 * sigma) ** 2
            c = C[int(np.argmin(cost))]
        return c

    def _ransac(self, Q: np.ndarray, iters: int = 60, tol: float = 0.02):
        best_in, best = -1, (None, 0.0)
        n = len(Q)
        for _ in range(iters):
            idx = self.rng.choice(n, 3, replace=False)
            a, b, c = Q[idx]
            if abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-6:
                continue
            cc, rr = fit_circle_algebraic(Q[idx])
            if not (self.r_bounds[0] <= rr <= self.r_bounds[1]):
                continue
            inl = np.abs(np.linalg.norm(Q - cc, axis=1) - rr) <= tol
            if int(inl.sum()) > best_in:
                best_in, best = int(inl.sum()), fit_circle_algebraic(Q[inl]) if inl.sum() >= 3 else (cc, rr)
        return best


def fit_ellipse(xy: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Direct least-squares ellipse (Fitzgibbon 1999): (centre, semi-axes (a >= b), angle) or None."""
    x, y = xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()
    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    S = D.T @ D
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2.0
    C[1, 1] = -1.0
    try:
        w, v = np.linalg.eig(np.linalg.solve(S + 1e-12 * np.eye(6), C))
    except np.linalg.LinAlgError:
        return None
    w, v = np.real(w), np.real(v)
    k = [i for i in range(6) if w[i] > 0 and np.isfinite(w[i])]
    if not k:
        return None
    A, B, Cc, Dd, E, F = v[:, k[0]]
    den = B * B - 4 * A * Cc
    if den >= 0:
        return None
    x0 = (2 * Cc * Dd - B * E) / den
    y0 = (2 * A * E - B * Dd) / den
    num = 2 * (A * E * E + Cc * Dd * Dd - B * Dd * E + den * F)
    root = math.sqrt((A - Cc) ** 2 + B * B)
    try:
        a1 = -math.sqrt(num * ((A + Cc) + root)) / den
        a2 = -math.sqrt(num * ((A + Cc) - root)) / den
    except ValueError:
        return None
    ang = 0.5 * math.atan2(-B, Cc - A)
    c = np.array([x0 + xy[:, 0].mean(), y0 + xy[:, 1].mean()])
    axes = np.array(sorted([abs(a1), abs(a2)], reverse=True))
    if not np.all(np.isfinite(axes)) or axes[1] < 0.02 * axes[0]:
        return None                     # degenerate (points on a line)
    return c, axes, ang


def fit_canopy_edge(points: np.ndarray, prior_xy: Sequence[float], canopy_r: float, gate_m: float = 0.6,
                    min_z: float = 0.3, ellipse_arc_deg: float = 150.0) -> TrunkFit:
    """Fallback (OS 2.5): the canopy-edge ellipse from the coverage-mask mast hits near the registry canopy
    (points above ``min_z``, so ground returns are left out). With a wide visible arc a direct ellipse fit gives
    the centre (``canopy_edge_ellipse``); otherwise a fixed-radius circle with the registry ``canopy_r_m``
    (``canopy_edge_circle``). Coarse by nature: the label tells the orbit it is not a trunk fit."""
    P = np.asarray(points, dtype=np.float64) if len(points) else np.zeros((0, 3))
    if P.ndim == 2 and P.shape[1] >= 3:
        P = P[P[:, 2] >= min_z]
    P = P[:, :2] if len(P) else np.zeros((0, 2))
    if len(P):
        d = np.hypot(P[:, 0] - prior_xy[0], P[:, 1] - prior_xy[1])
        P = P[d <= canopy_r + gate_m]
    if len(P) < 5:
        return TrunkFit(False, "canopy_edge_circle", n=int(len(P)), reason="too_few_points")
    span = arc_span_deg(P, prior_xy)
    if span >= ellipse_arc_deg and len(P) >= 10:
        e = fit_ellipse(P)
        if e is not None and e[1][0] <= 2.0 * (canopy_r + gate_m) and \
                math.hypot(e[0][0] - prior_xy[0], e[0][1] - prior_xy[1]) <= canopy_r + gate_m:
            c, axes, _ = e
            res = np.linalg.norm(P - c, axis=1) - float(axes.mean())
            return TrunkFit(True, "canopy_edge_ellipse", c, float(axes.mean()), float(np.sqrt(np.mean(res ** 2))),
                            int(len(P)), arc_span_deg(P, c))
    m = P.mean(axis=0)
    u = np.asarray(prior_xy, dtype=float) - m
    nu = float(np.linalg.norm(u))
    c0 = m + (u / nu * canopy_r if nu > 1e-9 else np.zeros(2))
    c, r, rms = fit_circle_geometric(P, c0, canopy_r, fixed_r=float(canopy_r))
    return TrunkFit(True, "canopy_edge_circle", c, r, rms, int(len(P)), arc_span_deg(P, c))


# ---------------------------------------------------------------------------------------------- messages
def laser_scan(scan: Scan, frame_id: str, decimals: int = 3) -> dict:
    """``sensor_msgs/LaserScan``-shaped JSON (no return = ``null``; JSON has no inf)."""
    a = scan.angles
    inc = float(a[1] - a[0]) if len(a) > 1 else 0.0
    rr = [None if not math.isfinite(v) else round(float(v), decimals) for v in scan.ranges]
    return {"header": {"frame_id": frame_id, "stamp": round(scan.t, 4)}, "angle_min": round(float(a[0]), 6),
            "angle_max": round(float(a[-1]), 6), "angle_increment": round(inc, 8), "range_min": RANGE_MIN_M,
            "range_max": RANGE_MAX_M, "ranges": rr}


# ---------------------------------------------------------------------------------------------- ring-depth band
# OS 2.5: the low-plane trunk fit is cross-checked on a ring-depth band at 0.2-0.5 m above the ground, below the
# scaffolds (from ~0.6 m) and above the basin rim / mulch. In the perception process the band comes from a rendered
# ring depth frame (``depth_band_points``); in the sim process (no GL) ``RingDepthBand`` casts the same ring-camera
# pixel rays (160 x 120, fovy from the MJCF) over the band's image ROI only, with the camera's solid geom groups
# {0, 1, 3} (a camera sees foliage; decals are thin quads on the bark), so both paths share one back-projection.
RING_CAMS = ("ring_0", "ring_1", "ring_2", "ring_3")
RING_BAND_Z_M = (0.20, 0.50)
RING_DEPTH_WH = (160, 120)
RING_DEPTH_SIGMA_M = 0.01             # depth noise, assumption (stereo / ToF class at 1-3 m)
RING_BAND_GROUPS = (1, 1, 0, 1, 0, 0)
RING_BAND_AGREE_M = 0.03              # centre agreement with the low-plane fit (2 cm fit budget + 1 cm)
RING_BAND_OFF_TRUNK_M = 0.05          # band centre beyond r + this from the prior and the primary: off the trunk
RING_BAND_RADIUS_TOL_M = 0.02         # free band radius vs the registry radius (wrong registry radius check)
RING_BAND_FREE_ARC_DEG = 100.0        # the free band radius is judged only on a free-fit arc this wide
_MJ_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


def optical_rot(d: mujoco.MjData, cid: int) -> np.ndarray:
    """Optical frame (x right, y down, z forward) axes in world coordinates (columns)."""
    return d.cam_xmat[cid].reshape(3, 3) @ _MJ_TO_OPTICAL


def backproject(uu: np.ndarray, vv: np.ndarray, z: np.ndarray, cam_pos: np.ndarray, rot_opt_world: np.ndarray,
                fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pixel centres (u, v) with optical depth z -> world points (N x 3). Same convention as the privacy parcel
    mask (``(u + 0.5 - cx) / fx * z``)."""
    z = np.asarray(z, dtype=np.float64).ravel()
    x = (np.asarray(uu, dtype=np.float64).ravel() + 0.5 - cx) / fx * z
    y = (np.asarray(vv, dtype=np.float64).ravel() + 0.5 - cy) / fy * z
    P = np.stack([x, y, z], axis=-1)
    return P @ np.asarray(rot_opt_world, dtype=np.float64).T + np.asarray(cam_pos, dtype=np.float64)


def depth_band_points(depth: np.ndarray, cam_pos: np.ndarray, rot_opt_world: np.ndarray, intr, ground_z: float,
                      band: Tuple[float, float] = RING_BAND_Z_M, max_depth: Optional[float] = None,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
    """World points of a depth frame whose height above ``ground_z`` lies in ``band`` (a rendered ring depth frame
    in the perception process). ``mask`` True = drop (self-mask). ``intr`` has ``fx, fy, cx, cy``."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    z = np.asarray(depth, dtype=np.float64)
    good = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        good &= z < max_depth
    if mask is not None:
        good &= ~np.asarray(mask, dtype=bool)
    P = backproject(uu[good], vv[good], z[good], cam_pos, rot_opt_world, intr.fx, intr.fy, intr.cx, intr.cy)
    h = P[:, 2] - float(ground_z)
    return P[(h >= band[0]) & (h <= band[1])]


class RingDepthBand:
    """Ring-depth band points around a trunk prior: the ring cameras' pixel rays cast over the band ROI only."""

    def __init__(self, lidar: "LidarModel", width: int = RING_DEPTH_WH[0], height: int = RING_DEPTH_WH[1],
                 cams: Sequence[str] = RING_CAMS, band: Tuple[float, float] = RING_BAND_Z_M,
                 noise_sigma: float = RING_DEPTH_SIGMA_M, max_depth: float = 4.0, seed: int = 7):
        self.L = lidar
        m = lidar.m
        self.W, self.H = int(width), int(height)
        self.band = (float(band[0]), float(band[1]))
        self.noise_sigma = float(noise_sigma)
        self.max_depth = float(max_depth)
        self.rng = np.random.default_rng(seed)
        self.cams = []
        for n in cams:
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n)
            if cid >= 0:
                f = 0.5 * self.H / math.tan(math.radians(float(m.cam_fovy[cid])) / 2.0)
                self.cams.append((n, cid, f))
        self.last_rays = 0

    def roi(self, d: mujoco.MjData, cid: int, f: float, prior_xy: Sequence[float], half: float, ground_z: float):
        """Pixel box (u0, u1, v0, v1) of the band box around the prior in camera ``cid``, or None (behind / out)."""
        Ropt = optical_rot(d, cid)
        p = d.cam_xpos[cid]
        xs = (prior_xy[0] - half, prior_xy[0] + half)
        ys = (prior_xy[1] - half, prior_xy[1] + half)
        zs = (ground_z + self.band[0], ground_z + self.band[1])
        C = np.array([[x, y, z] for x in xs for y in ys for z in zs])
        Q = (C - p) @ Ropt                    # world -> optical
        if np.any(Q[:, 2] <= 0.05):
            return None
        u = Q[:, 0] / Q[:, 2] * f + self.W / 2.0
        v = Q[:, 1] / Q[:, 2] * f + self.H / 2.0
        u0, u1 = max(int(math.floor(u.min())), 0), min(int(math.ceil(u.max())), self.W)
        v0, v1 = max(int(math.floor(v.min())), 0), min(int(math.ceil(v.max())), self.H)
        if u1 <= u0 or v1 <= v0:
            return None
        return u0, u1, v0, v1

    def points(self, d: mujoco.MjData, prior_xy: Sequence[float], ground_z: float, half: float = 0.3,
               noise: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """(band points N x 3, their camera origins N x 3, cameras used)."""
        P, O, used = [], [], []
        self.last_rays = 0
        for name, cid, f in self.cams:
            box = self.roi(d, cid, f, prior_xy, half, ground_z)
            if box is None:
                continue
            u0, u1, v0, v1 = box
            Ropt = optical_rot(d, cid)
            p = d.cam_xpos[cid].copy()
            vv, uu = np.mgrid[v0:v1, u0:u1]
            uu, vv = uu.ravel(), vv.ravel()
            dop = np.stack([(uu + 0.5 - self.W / 2.0) / f, (vv + 0.5 - self.H / 2.0) / f, np.ones(len(uu))], -1)
            nrm = np.linalg.norm(dop, axis=1)
            dw = (dop / nrm[:, None]) @ Ropt.T
            r, _ = self.L._cast(d, p, dw, RING_BAND_GROUPS, noise=False)
            self.last_rays += len(dw)
            z = r / nrm                      # range along the ray -> optical depth
            if noise and self.noise_sigma > 0:
                fin = np.isfinite(z)
                z[fin] = z[fin] + self.rng.normal(0.0, self.noise_sigma, int(fin.sum()))
            ok = np.isfinite(z) & (z > 0) & (z < self.max_depth)
            if not ok.any():
                continue
            Q = backproject(uu[ok], vv[ok], z[ok], p, Ropt, f, f, self.W / 2.0, self.H / 2.0)
            h = Q[:, 2] - float(ground_z)
            sel = (h >= self.band[0]) & (h <= self.band[1])
            if sel.any():
                P.append(Q[sel])
                O.append(np.broadcast_to(p, Q[sel].shape))
                used.append(name)
        if not P:
            return np.zeros((0, 3)), np.zeros((0, 3)), used
        return np.concatenate(P), np.concatenate(O), used


def _ransac_fixed_r(Q: np.ndarray, r: float, view: Optional[np.ndarray], tol_m: float, rng, iters: int = 120):
    """Fixed-radius RANSAC: two points give two centres at distance r; keep the one behind the visible surface
    (along the camera view direction ``view``, xy), score by inliers. Returns the best centre or None."""
    n = len(Q)
    if n < 2:
        return None
    best, best_in = None, -1
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        a, b = Q[i], Q[j]
        mvec = 0.5 * (a + b)
        dv = b - a
        h2 = r * r - 0.25 * float(dv @ dv)
        if h2 < 0 or float(dv @ dv) < 1e-8:
            continue
        nrm = np.array([-dv[1], dv[0]]) / math.sqrt(float(dv @ dv))
        for s in (1.0, -1.0):
            c = mvec + s * math.sqrt(h2) * nrm
            if view is not None and float((c - mvec) @ view) < 0:
                continue                      # the mirrored circle in front of the surface
            inl = int(np.sum(np.abs(np.linalg.norm(Q - c, axis=1) - r) <= tol_m))
            if inl > best_in:
                best, best_in = c, inl
    return best


def _trimmed_circle(Q: np.ndarray, c0: np.ndarray, r0: float, fixed: bool, tol_m: float, iters: int = 5):
    """Geometric circle fit with residual trimming: sprouts, water shoots and prune stubs on the trunk base touch the
    trunk (one cluster), so their points are dropped as outliers (|residual| > ``tol_m``) and the fit is redone."""
    c, r = np.asarray(c0, dtype=np.float64), float(r0)
    keep = np.abs(np.linalg.norm(Q - c, axis=1) - r) <= tol_m      # start from the seed's inliers
    if keep.sum() < 3:
        keep = np.ones(len(Q), dtype=bool)
    for _ in range(iters):
        if keep.sum() < 3:
            break
        c, r, _ = fit_circle_geometric(Q[keep], c, r, fixed_r=r0 if fixed else None)
        res = np.abs(np.linalg.norm(Q - c, axis=1) - r)
        new = res <= tol_m
        if np.array_equal(new, keep):
            break
        keep = new
    if keep.sum() < 3:
        return None, None, None, keep
    rms = float(np.sqrt(np.mean((np.linalg.norm(Q[keep] - c, axis=1) - r) ** 2)))
    return c, r, rms, keep


BAND_FIELDS = ("method", "fit", "ok", "reason", "n", "inliers", "arc_deg", "centre", "radius", "rms", "free_radius",
               "radius_delta_m", "free_arc_deg", "radius_mismatch", "off_prior_m", "delta_m", "agree")


def band_result(**kw) -> dict:
    """A ``cross_check`` payload with every contract field (None / 0 / False defaults), updated with ``kw``."""
    out = {"method": "ring_depth_band", "fit": "free", "ok": False, "reason": "", "n": 0, "inliers": 0,
           "arc_deg": 0.0, "centre": None, "radius": None, "rms": None, "free_radius": None, "radius_delta_m": None,
           "free_arc_deg": None, "radius_mismatch": None, "off_prior_m": None, "delta_m": None, "agree": None}
    out.update(kw)
    return out


def band_fit(points: np.ndarray, origins: Optional[np.ndarray], prior_xy: Sequence[float],
             prior_r: Optional[float], gate_m: float = 0.5, tol_m: float = 0.02, min_inlier_frac: float = 0.4,
             r_bounds: Tuple[float, float] = (0.03, 0.25), min_points: int = 6, seed: int = 7) -> dict:
    """The primary-independent part of ``band_cross_check``: the circle fit on the band points (deterministic for
    given points, prior and seed, so the hook caches it with the 2 Hz band cast).

    With a registry radius the fixed-radius fit is followed by a **free-radius** trimmed fit started from its centre
    (``free_radius``, ``radius_delta_m`` = free - registry): a wrong registry radius shifts the fixed-radius band
    centre the same way as the fixed-radius low-plane centre, so the centres alone would still agree."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3) if len(points) else np.zeros((0, 3))
    O = None if origins is None or not len(P) else np.broadcast_to(np.asarray(origins, dtype=np.float64), P.shape)
    out = band_result(fit="fixed_r" if prior_r is not None else "free")
    if len(P):
        near = np.hypot(P[:, 0] - prior_xy[0], P[:, 1] - prior_xy[1]) <= gate_m
        P = P[near]
        O = O[near] if O is not None else None
    if len(P) >= min_points:
        groups = cluster_xy(np.column_stack([P[:, :2], np.zeros(len(P))]), 0.08)
        best = min(groups, key=lambda i: float(np.min(np.hypot(P[i, 0] - prior_xy[0], P[i, 1] - prior_xy[1]))))
        P = P[best]
        O = O[best] if O is not None else None
    out["n"] = int(len(P))
    if len(P) < min_points:
        out["reason"] = "too_few_points"
        return out
    Q = P[:, :2]
    if prior_r is not None:
        m = Q.mean(axis=0)
        u = (Q - O[:, :2]).mean(axis=0) if O is not None else np.asarray(prior_xy, dtype=float) - m
        nu = float(np.linalg.norm(u))
        view = u / nu if nu > 1e-9 else None
        c0 = _ransac_fixed_r(Q, float(prior_r), view, tol_m, np.random.default_rng(seed))
        if c0 is None:
            c0 = m + (view * float(prior_r) if view is not None else np.zeros(2))
        c, r, rms, keep = _trimmed_circle(Q, c0, float(prior_r), True, tol_m)
    else:
        c, r = TrunkFitter(r_bounds=r_bounds, seed=seed)._ransac(Q, tol=tol_m)
        if c is None:
            out["reason"] = "no_free_fit"
            return out
        c, r, rms, keep = _trimmed_circle(Q, c, r, False, tol_m)
    if c is None:
        out["reason"] = "too_few_inliers"
        return out
    out.update(inliers=int(keep.sum()), arc_deg=round(arc_span_deg(Q[keep], c), 1),
               centre=[round(float(v), 4) for v in c], radius=round(float(r), 4), rms=round(float(rms), 4))
    out["off_prior_m"] = round(float(np.hypot(c[0] - prior_xy[0], c[1] - prior_xy[1])), 4)
    if prior_r is not None:
        cf, rf, _, kf = _trimmed_circle(Q, c, float(prior_r), False, tol_m)
        if cf is not None and r_bounds[0] <= rf <= r_bounds[1] and \
                kf.sum() >= max(min_points, min_inlier_frac * len(Q)):
            out.update(free_radius=round(float(rf), 4), radius_delta_m=round(float(rf) - float(prior_r), 4),
                       free_arc_deg=round(arc_span_deg(Q[kf], cf), 1))
    if keep.sum() < max(min_points, min_inlier_frac * len(Q)):
        out["reason"] = "too_few_inliers"
    elif not (r_bounds[0] <= r <= r_bounds[1]):
        out.update(reason="radius_out_of_bounds", centre=None, radius=None, rms=None, off_prior_m=None)
    else:
        out["ok"] = True
    return out


def band_compare(fit: dict, prior_xy: Sequence[float], primary: Optional["TrunkFit"] = None,
                 agree_m: float = RING_BAND_AGREE_M, off_trunk_m: float = RING_BAND_OFF_TRUNK_M,
                 radius_tol_m: float = RING_BAND_RADIUS_TOL_M,
                 free_arc_min_deg: float = RING_BAND_FREE_ARC_DEG) -> dict:
    """``band_fit`` result -> the ``cross_check`` payload against the low-plane fit ``primary``.

    * **off_trunk**: a fit whose centre is farther than ``radius + off_trunk_m`` from both the prior and the primary
      centre is not on the trunk (close in, under ~0.65 m beside the tree, the tree-side ring camera cannot see the
      band and the front camera sees only a pruning sprout): ``ok`` False, ``reason: off_trunk``, ``agree`` None.
    * **agree** = centre ``delta_m`` <= ``agree_m`` AND (no free radius, or ``|radius_delta_m|`` <= ``radius_tol_m``);
      None when either fit failed or ``primary`` is not a low-plane fit. ``radius_mismatch`` True when the free band
      radius disagrees with the registry radius. The free radius is judged only on a free-fit arc of at least
      ``free_arc_min_deg`` (100 deg; on narrower arcs at 1.5-2.2 m it ran 1-8 cm off, measured), else
      ``radius_mismatch`` is None and only the centres count."""
    out = dict(fit)
    judged = out.get("radius_delta_m") is not None and (out.get("free_arc_deg") or 0.0) >= free_arc_min_deg
    out["radius_mismatch"] = bool(abs(float(out["radius_delta_m"])) > radius_tol_m) if judged else None
    have_prim = primary is not None and primary.ok and primary.centre is not None and \
        primary.method.startswith("low_plane")
    if out["ok"]:
        c = np.asarray(out["centre"], dtype=float)
        lim = float(out["radius"]) + off_trunk_m
        d_ref = float(np.hypot(c[0] - prior_xy[0], c[1] - prior_xy[1]))
        if have_prim:
            d_ref = min(d_ref, float(np.linalg.norm(c - np.asarray(primary.centre, dtype=float))))
        if d_ref > lim:
            out.update(ok=False, reason="off_trunk")
    if out["ok"] and have_prim:
        dlt = float(np.linalg.norm(np.asarray(out["centre"]) - np.asarray(primary.centre)))
        out["delta_m"] = round(dlt, 4)
        out["agree"] = bool(dlt <= agree_m and not out["radius_mismatch"])
    return out


def band_cross_check(points: np.ndarray, origins: Optional[np.ndarray], prior_xy: Sequence[float],
                     prior_r: Optional[float], primary: Optional["TrunkFit"] = None,
                     agree_m: float = RING_BAND_AGREE_M, gate_m: float = 0.5, tol_m: float = 0.02,
                     min_inlier_frac: float = 0.4, r_bounds: Tuple[float, float] = (0.03, 0.25),
                     min_points: int = 6, seed: int = 7, off_trunk_m: float = RING_BAND_OFF_TRUNK_M,
                     radius_tol_m: float = RING_BAND_RADIUS_TOL_M,
                     free_arc_min_deg: float = RING_BAND_FREE_ARC_DEG) -> dict:
    """Circle fit on the ring-depth band points and its agreement with the low-plane fit ``primary``
    (``band_compare(band_fit(...))``).

    The points within ``gate_m`` of the prior, cluster nearest the prior (0.08 m gap), then a trimmed geometric fit
    (fixed registry radius when given, started behind the visible surface along the mean camera ray, plus a
    free-radius check; else RANSAC + free radius). Returns ``{method: ring_depth_band, fit, ok, reason, n, inliers,
    arc_deg, centre, radius, rms, free_radius, radius_delta_m, free_arc_deg, radius_mismatch, off_prior_m, delta_m,
    agree}``. A
    cross-check only: it never replaces the low-plane result."""
    fit = band_fit(points, origins, prior_xy, prior_r, gate_m=gate_m, tol_m=tol_m, min_inlier_frac=min_inlier_frac,
                   r_bounds=r_bounds, min_points=min_points, seed=seed)
    return band_compare(fit, prior_xy, primary, agree_m=agree_m, off_trunk_m=off_trunk_m, radius_tol_m=radius_tol_m,
                        free_arc_min_deg=free_arc_min_deg)


def _ransac_fixed_r_facing(Q: np.ndarray, V: np.ndarray, r: float, tol_m: float, rng, iters: int = 80):
    """Fixed-radius RANSAC on low-plane hits with their unit ray directions ``V`` (sensor -> hit, NaN = unknown): a
    hit counts as an inlier only on the sensor-facing half of the circle ((p - c) . v <= 0). Vectorised: ``iters``
    random pairs, both circles through each pair. Returns the centre with the most inliers, or None."""
    n = len(Q)
    if n < 2:
        return None
    i = rng.integers(0, n, iters)
    j = (i + rng.integers(1, n, iters)) % n                  # j != i
    a, b = Q[i], Q[j]
    dv = b - a
    dd = np.sum(dv * dv, axis=1)
    h2 = r * r - 0.25 * dd
    good = (h2 >= 0) & (dd >= 1e-8)
    if not good.any():
        return None
    a, dv, dd, h2 = a[good], dv[good], dd[good], h2[good]
    mvec = a + 0.5 * dv
    nrm = np.column_stack([-dv[:, 1], dv[:, 0]]) / np.sqrt(dd)[:, None]
    off = np.sqrt(h2)[:, None] * nrm
    C = np.concatenate([mvec + off, mvec - off])            # (2k, 2)
    Vz = np.where(np.all(np.isfinite(V), axis=1)[:, None], V, 0.0)
    W = Q[None, :, :] - C[:, None, :]                       # (2k, n, 2)
    inl = (np.abs(np.linalg.norm(W, axis=2) - r) <= tol_m) & (np.einsum("knd,nd->kn", W, Vz) <= 0.0)
    return C[int(np.argmax(inl.sum(axis=1)))]
