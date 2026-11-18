"""WP5 hook ``lidar`` (10 Hz, sim process): mast LiDAR, low safety planes, clusters and the trunk fit
(MVP-SPEC 2.4, 5.4; ORCHARD-MVP-SPEC 2.5).

Runs ``controllers.lidar_model`` on the **live** ``MjData`` (it sees the current mocap actors, contype and eq state)
and publishes, as JSON on the hook bus (``sim_runner`` turns each into a ``std_msgs/String``):

``/nandi/scan``           mast LiDAR, ``{"beams": [{"elevation_deg", ...LaserScan fields}, x3]}`` (safety mask {0, 1})
``/nandi/scan_low``       the two 0.18 m planes, ``{"planes": [{"name", "origin", ...LaserScan fields}, x2]}``
``/nandi/lidar/clusters`` non-map clusters within 8 m (``{"clusters": [{"lo", "hi", "n", "c"}], "t"}``) for the
                          perception privacy filter (detector mask) and the safety layer
``/nandi/lidar/trunk_fit`` after a request on ``/nandi/lidar/trunk_fit_request``
                          ``{"tree_id", "prior_xy": [x, y], "trunk_r_m"?, "canopy_r_m"?}``: the low-plane circle fit
                          (``method`` ``low_plane_circle_fixed_r`` / ``low_plane_circle``) or the canopy-edge
                          fallback (``canopy_edge_circle``, coverage mask {0, 1, 3}), with ``R_meas`` (robot to centre)
                          and ``cross_check``: the OS 2.5 ring-depth band (0.2-0.5 m above the ground, ring-camera pixel
                          rays over the band ROI, ``lidar_model.RingDepthBand``) circle fit and its centre ``delta_m``
                          / ``agree`` (<= 3 cm and no ``radius_mismatch`` of the free band radius) against the
                          low-plane fit, ``off_trunk`` when the band centre is off both the prior and the fit;
                          up to 2.5 m range, every contract field on every payload, never replaces the fit

The newest sweep, clusters and fit are also in ``bus.values["lidar"]`` for other hooks (WP6 safety fields, WP25
orbit). Scans carry noise (sigma 2 cm, seeded per world reset). Cost: 1,620 rays about 1.0-1.4 ms per sweep on the
2-CPU sim container on orchard_lite and about 4-5 ms on the orchard_demo hfield terrain (measured; terrain rays go
through ``lidar_model._HfieldCaster``), so 12-50 ms/s at 10 Hz. No model field is written (``mutates_model`` False).
"""
import math
from typing import Optional, Tuple

import numpy as np

from atharv_nandi.core import lidar_model as lm

RATE_HZ = 10.0
SCAN_TOPIC = "/nandi/scan"
SCAN_LOW_TOPIC = "/nandi/scan_low"
CLUSTERS_TOPIC = "/nandi/lidar/clusters"
FIT_TOPIC = "/nandi/lidar/trunk_fit"
FIT_REQUEST_TOPIC = "/nandi/lidar/trunk_fit_request"
RING_BAND_MAX_RANGE_M = 2.5          # the ring-depth band cross-check runs up to this robot-to-prior range


def load_lidar_config() -> dict:
    """The ``lidar:`` section of ``config/perception.yaml`` (empty when unreadable: module defaults apply)."""
    try:
        from atharv_nandi.core.detector_model import load_config
        return dict(load_config().get("lidar") or {})
    except Exception:  # noqa: BLE001 - a hook must never fail on config
        return {}


def _pos_num(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) and v > 0 else None


def parse_fit_request(req) -> Tuple[Optional[dict], str]:
    """Validate a ``/nandi/lidar/trunk_fit_request`` -> (clean request or None, error ('' when fine)).
    ``prior_xy``: 2 finite numbers; ``trunk_r_m`` / ``canopy_r_m``: absent / null or a positive finite number.
    A malformed request is refused (one ``ok: false`` reply), never raised: a hook exception would disable the
    LiDAR and safe-hold the runner."""
    if not isinstance(req, dict):
        return None, "not_an_object"
    p = req.get("prior_xy")
    if p is None:
        return None, "no_prior_xy"
    if not isinstance(p, (list, tuple)) or len(p) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in p):
        return None, "prior_xy_not_2_numbers"
    out = {"tree_id": req.get("tree_id"), "prior_xy": [float(p[0]), float(p[1])]}
    for k in ("trunk_r_m", "canopy_r_m"):
        v = req.get(k)
        if v is None:
            out[k] = None
            continue
        r = _pos_num(v)
        if r is None:
            return None, f"{k}_not_positive_number"
        out[k] = r
    return out, ""


class LidarHook:
    def __init__(self, cfg: Optional[dict] = None, seed: int = 7):
        c = dict(cfg or {})
        self.publish_scans = bool(c.get("publish_scans", True))
        self.noise = float(c.get("noise_sigma_m", lm.NOISE_SIGMA_M))
        self.range_max = float(c.get("range_max_m", lm.RANGE_MAX_M))
        self.cluster_range = float(c.get("cluster_range_m", lm.CLUSTER_RANGE_M))
        self.seed = int(seed)
        self.model = None
        self.lidar: Optional[lm.LidarModel] = None
        self.fitter = lm.TrunkFitter(window=int(c.get("trunk_window", 10)), gate_m=float(c.get("trunk_gate_m", 0.5)))
        rb = dict(c.get("ring_band") or {})
        self.band_enabled = bool(rb.get("enabled", True))
        z = rb.get("z_m", lm.RING_BAND_Z_M)
        self.band_z = (float(z[0]), float(z[1]))
        self.band_sigma = float(rb.get("noise_sigma_m", lm.RING_DEPTH_SIGMA_M))
        self.band_max_range = float(rb.get("max_range_m", RING_BAND_MAX_RANGE_M))
        self.band_agree = float(rb.get("agree_m", lm.RING_BAND_AGREE_M))
        self.band_off_trunk = float(rb.get("off_trunk_m", lm.RING_BAND_OFF_TRUNK_M))
        self.band_radius_tol = float(rb.get("radius_tol_m", lm.RING_BAND_RADIUS_TOL_M))
        self.band_free_arc = float(rb.get("free_arc_min_deg", lm.RING_BAND_FREE_ARC_DEG))
        self.band_period = 1.0 / max(float(rb.get("rate_hz", 2.0)), 1e-6)
        self._band_cache = None      # (key, t, points, origins, cams, rays, band fit): a static trunk
        self.band: Optional[lm.RingDepthBand] = None
        self.request: Optional[dict] = None
        self._req_count = 0
        self.last: dict = {}

    def _bind(self, m):
        if self.model is not m:
            self.model = m
            self.lidar = lm.LidarModel(m, seed=self.seed, noise_sigma=self.noise, range_max=self.range_max)
            self.band = lm.RingDepthBand(self.lidar, band=self.band_z, noise_sigma=self.band_sigma,
                                         seed=self.seed) if self.band_enabled else None
            self.fitter.reset()

    def _empty_cc(self, reason: str, req: dict, range_m: Optional[float] = None) -> dict:
        """A cross_check payload that carries every contract field (out_of_range, error:<exc>)."""
        return lm.band_result(fit="fixed_r" if req.get("trunk_r_m") is not None else "free", reason=reason,
                              radius_mismatch=None, cams=[], rays=0,
                              range_m=None if range_m is None else round(range_m, 3), band_t=None)

    def cross_check(self, d, req: dict, primary: lm.TrunkFit) -> Optional[dict]:
        """OS 2.5 ring-depth band (0.2-0.5 m) cross-check of the low-plane fit; None when disabled or no ring
        cameras. Beyond ``max_range_m`` from the robot it is not run (``reason: out_of_range``): at 160 x 120 a
        0.11 m trunk is under 3 px wide at 3 m, and the orbit stands off 0.9-1.5 m. The band cast **and** the band
        circle fit are cached at ``rate_hz`` (both depend only on the points, prior and radius); each tick only
        compares the cached fit with the current low-plane fit."""
        if self.band is None or not self.band.cams:
            return None
        prior = req["prior_xy"]
        xy = d.xpos[self.lidar.trunk][:2]
        rng = math.hypot(prior[0] - xy[0], prior[1] - xy[1])
        if rng > self.band_max_range:
            return self._empty_cc("out_of_range", req, rng)
        ground_z = float(d.xpos[self.lidar.trunk][2]) - lm.NOMINAL_TRUNK_Z_M
        r = req.get("trunk_r_m")
        half = (float(r) if r is not None else 0.15) + 0.25
        key = (tuple(prior), r)
        t = float(d.time)
        bc = self._band_cache
        if bc is None or bc[0] != key or not (0.0 <= t - bc[1] < self.band_period):
            P, O, cams = self.band.points(d, prior, ground_z, half=half)
            bfit = lm.band_fit(P, O, prior, r)
            bc = self._band_cache = (key, t, P, O, cams, self.band.last_rays, bfit)
        _, tb, P, O, cams, rays, bfit = bc
        out = lm.band_compare(bfit, prior, primary, agree_m=self.band_agree, off_trunk_m=self.band_off_trunk,
                              radius_tol_m=self.band_radius_tol, free_arc_min_deg=self.band_free_arc)
        out.update(cams=list(cams), rays=rays, range_m=round(rng, 3), band_t=round(tb, 4))
        return out

    def reset(self, m=None, d=None):
        self.request, self.last = None, {}
        self._band_cache = None
        self.fitter.reset()
        if m is not None:
            self.model = None
            self._bind(m)

    def tick(self, m, d, t: float, bus):
        self._bind(m)
        L = self.lidar
        scans = L.sweep(d, lm.SAFETY_GROUPS, noise=True)
        trunk_xy = d.xpos[L.trunk][:2]
        ground_z = float(d.xpos[L.trunk][2]) - lm.NOMINAL_TRUNK_Z_M      # local ground under the robot
        clusters = lm.nonmap_clusters(scans.values(), L.map_geom, trunk_xy, max_range=self.cluster_range,
                                      ground_z=ground_z)
        self.fitter.add_scans(scans)
        payload_c = {"t": round(float(t), 4), "frame": "world", "clusters": [c.as_dict() for c in clusters],
                     "assisted": ["map_masked_lidar"]}
        bus.publish(CLUSTERS_TOPIC, payload_c)
        if self.publish_scans:
            if "mast" in scans:
                s = scans["mast"]
                beams = []
                for el in lm.MAST_BEAMS_DEG:
                    sel = s.beams == el
                    sub = lm.Scan("mast", s.origin, s.dirs[sel], s.ranges[sel], s.geom[sel], s.angles[sel], el, s.t)
                    beams.append(dict(lm.laser_scan(sub, "mast_lidar"), elevation_deg=el))
                bus.publish(SCAN_TOPIC, {"beams": beams})
            planes = [dict(lm.laser_scan(s, k), name=k, origin=[round(float(v), 4) for v in s.origin])
                      for k, s in scans.items() if k != "mast"]
            bus.publish(SCAN_LOW_TOPIC, {"planes": planes})
        n = bus.arrival_count(FIT_REQUEST_TOPIC)
        if n != self._req_count:
            self._req_count = n
            req = bus.latest(FIT_REQUEST_TOPIC)
            self.request, bad = parse_fit_request(req)
            if bad and bad != "no_prior_xy":       # no prior_xy: a cancel, as before
                tid = req.get("tree_id") if isinstance(req, dict) else None
                bus.publish(FIT_TOPIC, {"ok": False, "method": "none", "reason": f"bad_request:{bad}",
                                        "tree_id": tid if isinstance(tid, (str, int)) else None,
                                        "t": round(float(t), 4)})
        fit = None
        if self.request is not None:
            try:
                fit = self.fit(d, self.request)
            except Exception as exc:  # noqa: BLE001 - a bad request must not disable the hook (WP0 hook rules)
                fit = {"ok": False, "method": "none", "reason": f"fit_error:{type(exc).__name__}",
                       "tree_id": self.request.get("tree_id")}
                self.request = None
            bus.publish(FIT_TOPIC, dict(fit, t=round(float(t), 4)))
        self.last = {"t": float(t), "clusters": payload_c["clusters"], "trunk_fit": fit,
                     "n_rays": L.n_rays}
        bus.values["lidar"] = self.last

    def fit(self, d, req: dict) -> dict:
        req, bad = parse_fit_request(req)
        if bad:
            raise ValueError(f"bad trunk fit request: {bad}")
        prior = list(req["prior_xy"])
        r = req.get("trunk_r_m")
        res = self.fitter.fit(prior, None if r is None else float(r))
        try:
            cc = self.cross_check(d, req, res)
        except Exception as exc:  # noqa: BLE001 - the cross-check must never cost the primary fit
            xy = d.xpos[self.lidar.trunk][:2]
            cc = self._empty_cc(f"error:{type(exc).__name__}", req,
                                math.hypot(prior[0] - xy[0], prior[1] - xy[1]))
        if not res.ok and req.get("canopy_r_m") is not None:
            cov = self.lidar.sweep(d, lm.COVERAGE_GROUPS, noise=True, low=False)
            pts = cov["mast"].points() if "mast" in cov else np.zeros((0, 3))
            res2 = lm.fit_canopy_edge(pts, prior, float(req["canopy_r_m"]))
            res2.reason = f"low_plane:{res.reason}" if res2.ok else f"{res2.reason}|low_plane:{res.reason}"
            res = res2
        out = res.as_dict()
        out["tree_id"] = req.get("tree_id")
        if cc is not None:
            out["cross_check"] = cc
        if res.centre is not None:
            xy = d.xpos[self.lidar.trunk][:2]
            out["R_meas"] = round(math.hypot(res.centre[0] - xy[0], res.centre[1] - xy[1]), 4)
        return out


_HOOK: Optional[LidarHook] = None


def register(hooks):
    global _HOOK
    cfg = load_lidar_config()
    _HOOK = LidarHook(cfg)
    hooks.register("lidar", _HOOK.tick, rate_hz=float(cfg.get("rate_hz", RATE_HZ)), mutates_model=False, topics_in=(FIT_REQUEST_TOPIC,),
                   reset=_HOOK.reset)
