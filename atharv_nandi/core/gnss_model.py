"""GNSS-RTK model, RTK supervisor and pose estimate for Atharv Nandi (WP7; pure Python, no ROS, no MuJoCo).

MVP-SPEC 2.5: antenna site on the mast head (``gnss_antenna``; falls back to ``imu`` / the trunk until WP1 adds
it) -> ``geo_frame`` with the fictitious site origin. States NONE / FLOAT (0.5 m) / FIXED (0.02 m), ASSUMED
horizontal 1-sigma per axis; vertical 2x. Scripted dropouts. RTK (re)convergence: after start or a dropout the
receiver reports FLOAT for ``converge_s`` before FIXED (the console badge goes FLOAT -> FIXED at start).

``RtkSupervisor`` is the mission-side rule (MVP-SPEC 5.6, MISSION-PRESETS 0.2 and 1):
    start gate      a preset needs FIXED; FLOAT only for RETURN HOME and SAFE PARK -> else reason ``rtk``
    mid-row loss    FIXED -> FLOAT / NONE: finish the current item, pause motion, wait 30 s; not recovered ->
                    return on odometry to the last FIXED waypoint. Back to FIXED -> resume.

``PoseEstimator`` fuses odometry (drift proportional to distance, hal-sim-to-real 4: 2 % ASSUMED) with the fix by a
per-axis Kalman update, so a FIXED fix pulls the estimate to the fix and a dropout grows the covariance.
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from atharv_nandi.core import geo_frame as gf

NONE, FLOAT, FIXED = "NONE", "FLOAT", "FIXED"
STATES = (NONE, FLOAT, FIXED)
SIGMA_H_M = {FIXED: 0.02, FLOAT: 0.5}          # ASSUMED horizontal 1-sigma per axis
SIGMA_V_FACTOR = 2.0                           # vertical sigma = 2 x horizontal (ASSUMED)
CONVERGE_S = 5.0                               # ASSUMED RTK convergence FLOAT -> FIXED
# sensor_msgs/NavSatStatus
NAVSAT_NO_FIX, NAVSAT_FIX, NAVSAT_SBAS_FIX, NAVSAT_GBAS_FIX = -1, 0, 1, 2
SERVICE_GPS = 1
COVARIANCE_TYPE_DIAGONAL_KNOWN = 2
NAVSAT_STATUS = {NONE: NAVSAT_NO_FIX, FLOAT: NAVSAT_FIX, FIXED: NAVSAT_GBAS_FIX}

REASON_RTK = "rtk"
RTK_WAIT_S = 30.0                              # MVP-SPEC 5.6: pause and wait 30 s
FLOAT_OK_PRESETS = ("return_home", "safe_park")
ANTENNA_SITES = ("gnss_antenna", "imu")        # first one present in the model


@dataclass(frozen=True)
class Dropout:
    t0: float
    dur: float
    state: str = NONE   # NONE or FLOAT during the dropout

    def __post_init__(self):
        if self.state not in (NONE, FLOAT):
            raise ValueError("a dropout degrades to NONE or FLOAT")
        if not (math.isfinite(self.t0) and math.isfinite(self.dur) and self.dur > 0):
            raise ValueError("dropout t0/dur must be finite, dur > 0")


@dataclass
class GnssScript:
    """Scripted receiver state. ``start_state`` at t = 0 (FLOAT: converges to FIXED after converge_s)."""
    dropouts: List[Dropout] = field(default_factory=list)
    start_state: str = FLOAT
    converge_s: float = CONVERGE_S

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "GnssScript":
        d = d or {}
        drops = [Dropout(float(x["t0"]), float(x["dur"]), str(x.get("state", NONE)).upper())
                 for x in d.get("dropouts", [])]
        start = str(d.get("start_state", FLOAT)).upper()
        if start not in STATES:
            raise ValueError(f"start_state {start!r} not in {STATES}")
        return cls(dropouts=drops, start_state=start, converge_s=float(d.get("converge_s", CONVERGE_S)))

    def state_at(self, t: float) -> str:
        """Receiver state at sim time t (dropouts win; after one ends, FLOAT for converge_s, then FIXED)."""
        last_end = None
        for dr in self.dropouts:
            if dr.t0 <= t < dr.t0 + dr.dur:
                return dr.state
            end = dr.t0 + dr.dur
            if end <= t and (last_end is None or end > last_end):
                last_end = end
        if self.start_state == NONE and last_end is None:
            return NONE
        ref = last_end if last_end is not None else (0.0 if self.start_state == FLOAT else None)
        if ref is None:
            return FIXED
        return FLOAT if t - ref < self.converge_s else FIXED


class GnssModel:
    """Fix generator: truth (local ENU) + seeded Gaussian error per state -> NavSatFix-compatible dict."""

    def __init__(self, origin: gf.SiteOrigin, script: Optional[GnssScript] = None, seed: int = 7,
                 sigma_h: Optional[dict] = None):
        self.origin = origin
        self.script = script or GnssScript()
        self.sigma_h = dict(SIGMA_H_M if sigma_h is None else sigma_h)
        self._seed = int(seed)
        self.rng = random.Random(self._seed)
        self._scripted = list(self.script.dropouts)

    def reset(self):
        """Back to t = 0: same noise sequence, injected dropouts dropped."""
        self.rng = random.Random(self._seed)
        self.script.dropouts = list(self._scripted)

    def inject(self, t: float, dur: float, state: str = NONE):
        """Runtime dropout (console / test harness): ``state`` until t + dur, then reconverge."""
        self.script.dropouts.append(Dropout(float(t), float(dur), state.upper()))

    def state_at(self, t: float) -> str:
        return self.script.state_at(t)

    def sample(self, t: float, x: float, y: float, z: float) -> dict:
        st = self.state_at(t)
        msg = {"header": {"stamp": round(float(t), 6), "frame_id": "gnss_antenna"},
               "status": {"status": NAVSAT_STATUS[st], "service": SERVICE_GPS}, "rtk": st,
               "position_covariance_type": COVARIANCE_TYPE_DIAGONAL_KNOWN, "assumed": True}
        if st == NONE:
            msg.update(latitude=None, longitude=None, altitude=None, position_covariance=[0.0] * 9,
                       local=None, sigma_h_m=None)
            return msg
        sh = self.sigma_h[st]
        sv = sh * SIGMA_V_FACTOR
        ex, ey, ez = self.rng.gauss(0.0, sh), self.rng.gauss(0.0, sh), self.rng.gauss(0.0, sv)
        mx, my, mz = x + ex, y + ey, z + ez
        lat, lon, alt = self.origin.to_wgs84(mx, my, mz)
        msg.update(latitude=lat, longitude=lon, altitude=alt,
                   position_covariance=[sh * sh, 0.0, 0.0, 0.0, sh * sh, 0.0, 0.0, 0.0, sv * sv],
                   local=[round(mx, 4), round(my, 4), round(mz, 4)], sigma_h_m=sh)
        return msg


def validate_navsatfix(msg: dict) -> list:
    """Schema check of a /nandi/gnss/fix payload. [] when valid."""
    errs = []
    for k in ("header", "status", "latitude", "longitude", "altitude", "position_covariance",
              "position_covariance_type", "rtk"):
        if k not in msg:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if msg["rtk"] not in STATES:
        errs.append("rtk not in NONE/FLOAT/FIXED")
    elif msg["status"].get("status") != NAVSAT_STATUS[msg["rtk"]]:
        errs.append("status.status does not match rtk")
    if len(msg["position_covariance"]) != 9:
        errs.append("position_covariance needs 9 entries")
    if msg["rtk"] != NONE:
        for k in ("latitude", "longitude", "altitude"):
            if not isinstance(msg[k], float) or not math.isfinite(msg[k]):
                errs.append(f"{k} not finite")
    return errs


# ------------------------------------------------------------------ mission-side RTK rule
def rtk_start_gate(rtk: str, preset: str = "", float_ok: Optional[bool] = None) -> Tuple[bool, str]:
    """(ok, reason): FIXED to start; FLOAT only for RETURN HOME / SAFE PARK (MP 0.2, RTK-fix rule). ``float_ok`` (the
    preset registry's flag) overrides the ``FLOAT_OK_PRESETS`` name list; the Ready check (``ready_check._common``)
    passes it, so the live start gate and this rule are one function."""
    if rtk == FIXED:
        return True, ""
    allow_float = (preset or "").lower() in FLOAT_OK_PRESETS if float_ok is None else bool(float_ok)
    if rtk == FLOAT and allow_float:
        return True, ""
    return False, REASON_RTK


A_CONTINUE, A_FINISH_ITEM, A_PAUSE, A_RETURN_ODOM = "continue", "finish_item", "pause", "return_odometry"


class RtkSupervisor:
    """Mid-row RTK loss rule. ``update`` returns the action; ``waypoint`` records FIXED waypoints."""

    def __init__(self, wait_s: float = RTK_WAIT_S):
        self.wait_s = float(wait_s)
        self.reset()

    def reset(self):
        self.action = A_CONTINUE
        self.lost_at: Optional[float] = None
        self.armed = False        # a loss counts only after the first FIXED (start is the start gate decides)
        self.paused_at: Optional[float] = None
        self.last_fixed_wp: Optional[Tuple[float, float]] = None
        self.last_fixed_pos: Optional[Tuple[float, float]] = None
        self.events: List[dict] = []

    def waypoint(self, xy: Sequence[float], rtk: str):
        """A waypoint reached: remembered as the return target only while FIXED."""
        if rtk == FIXED:
            self.last_fixed_wp = (float(xy[0]), float(xy[1]))

    def observe(self, xy: Sequence[float], rtk: str):
        """Pose while FIXED (fallback return target when the mission sent no waypoints)."""
        if rtk == FIXED:
            self.last_fixed_pos = (float(xy[0]), float(xy[1]))

    @property
    def return_target(self) -> Optional[Tuple[float, float]]:
        return self.last_fixed_wp if self.last_fixed_wp is not None else self.last_fixed_pos

    def update(self, t: float, rtk: str, item_active: bool = False) -> str:
        prev = self.action
        if rtk == FIXED:
            self.armed = True
            if self.action != A_CONTINUE:
                self.events.append({"t": t, "event": "rtk_recovered", "after_s": round(t - (self.lost_at or t), 3)})
            self.action, self.lost_at, self.paused_at = A_CONTINUE, None, None
            return self.action
        if not self.armed:
            return self.action
        if self.lost_at is None:
            self.lost_at = t
            self.events.append({"t": t, "event": "rtk_lost", "rtk": rtk, "reason": REASON_RTK})
        if self.action in (A_CONTINUE, A_FINISH_ITEM):
            if item_active:
                self.action = A_FINISH_ITEM
            else:
                self.action, self.paused_at = A_PAUSE, t
        elif self.action == A_PAUSE and t - self.paused_at >= self.wait_s - 1e-9:
            self.action = A_RETURN_ODOM
        if self.action != prev and self.action != A_CONTINUE:
            self.events.append({"t": t, "event": self.action, "target": self.return_target})
        return self.action

    def status(self, t: Optional[float] = None) -> dict:
        waited = None if (self.paused_at is None or t is None) else round(t - self.paused_at, 3)
        return {"action": self.action, "reason": REASON_RTK if self.action != A_CONTINUE else "",
                "lost_at": self.lost_at, "paused_at": self.paused_at, "waited_s": waited,
                "last_fixed_wp": list(self.last_fixed_wp) if self.last_fixed_wp else None,
                "return_target": list(self.return_target) if self.return_target else None}


# ------------------------------------------------------------------ pose estimate
ODOM_DRIFT_FRAC = 0.02          # hal-sim-to-real 4: 2 % of distance (ASSUMED), 1-sigma growth
ODOM_HEADING_BIAS_DEG = 1.0     # ASSUMED 1-sigma heading bias of the odometry, redrawn per run


class PoseEstimator:
    """Odometry + GNSS, per-axis Kalman on x, y. Yaw from the IMU orientation (passed in)."""

    def __init__(self, seed: int = 7, drift_frac: float = ODOM_DRIFT_FRAC):
        self._seed = int(seed)
        self.drift_frac = float(drift_frac)
        self.reset()

    def reset(self, xy: Sequence[float] = (0.0, 0.0)):
        rng = random.Random(self._seed + 1)
        self._scale = 1.0 + rng.gauss(0.0, self.drift_frac)
        self._bias = math.radians(rng.gauss(0.0, ODOM_HEADING_BIAS_DEG))
        self.x, self.y = float(xy[0]), float(xy[1])
        self.var0 = 0.0               # variance right after the last correction
        self.dist_since_corr = 0.0    # odometry metres since the last correction
        self._truth_prev: Optional[Tuple[float, float]] = None
        self.dist_since_fix = 0.0
        self.source = "init"

    def predict(self, truth_xy: Sequence[float]):
        """Odometry step from the truth delta, with scale and heading-bias error; variance grows with distance."""
        tx, ty = float(truth_xy[0]), float(truth_xy[1])
        if self._truth_prev is not None:
            dx, dy = tx - self._truth_prev[0], ty - self._truth_prev[1]
            c, s = math.cos(self._bias), math.sin(self._bias)
            self.x += self._scale * (c * dx - s * dy)
            self.y += self._scale * (s * dx + c * dy)
            step = math.hypot(dx, dy)
            self.dist_since_fix += step
            self.dist_since_corr += step
        self._truth_prev = (tx, ty)

    @property
    def var(self) -> float:
        """Systematic odometry error grows LINEARLY with distance (scale + heading bias), not as a random walk."""
        k = math.hypot(self.drift_frac, math.radians(ODOM_HEADING_BIAS_DEG))
        return self.var0 + (k * self.dist_since_corr) ** 2

    def correct(self, fix_local: Optional[Sequence[float]], sigma_h: Optional[float], rtk: str):
        if fix_local is None or sigma_h is None or rtk == NONE:
            self.source = "odometry"
            return
        r = sigma_h * sigma_h
        v = self.var
        k = v / (v + r) if v + r > 0 else 1.0
        if self.source == "init":
            k = 1.0
        self.x += k * (fix_local[0] - self.x)
        self.y += k * (fix_local[1] - self.y)
        self.var0 = (1.0 - k) * v if self.source != "init" else r
        self.dist_since_corr = 0.0
        if rtk == FIXED:
            self.dist_since_fix = 0.0
        self.source = "rtk_fixed" if rtk == FIXED else "rtk_float"

    def estimate(self, t: float, yaw: float) -> dict:
        sd = math.sqrt(max(self.var, 0.0))
        return {"t": round(float(t), 6), "x": round(self.x, 4), "y": round(self.y, 4), "yaw": round(float(yaw), 5),
                "sigma_xy_m": round(sd, 4), "source": self.source, "dist_since_fix_m": round(self.dist_since_fix, 3),
                "assumed": True}
