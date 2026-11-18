"""WP13 dock: alignment, time-compressed charge and the hub sync stub (MISSION-PRESETS 9; ORCHARD-MVP-SPEC 1.2
step 11, O-20; MVP-SPEC 4 "Dock (stretch)").

Pure Python except the thin ``DockRunner`` glue at the bottom, which drives a WP0b ``SimDriver`` gait from a task.
WP37 replaces this module with Dock v2 at the C+ boundary (RTH-DOCKING E.5-E.10); everything here is v1.

Geometry (``DockGeometry``)
    The docked pose is the trunk at the ``dock_ref`` site (WP4 ``dock`` body, plate centre) heading along the dock
    yaw. The WP4 backboard stands behind the plate (-x of the dock frame), so the default entry is ``reverse``: the
    robot stands at the staging point ``STAGING_M`` in front of the plate, heading the dock yaw, and backs straight
    in. ``remaining`` is the distance still to go along the entry direction, ``lateral`` the offset across it.

Alignment (``DockAligner``), a state machine fed with ``Obs`` (truth trunk pose in the sim; the real robot reads the
dock tag) and returning normalised gait commands ``Cmd`` (as ``/nandi/cmd_vel``):

    idle -> approach (walk to the staging point) -> align (turn to the dock yaw, then strafe the lateral error out,
    then the along-axis error) -> entry (straight creep, yaw held by a P-term, no side steps, stop early by the
    predicted gait coast) -> settle -> verify
    Strafes run only with every foot off the plate (``remaining`` >= ``DockGeometry.clear_m`` = plate half-length +
    FOOT_REACH_M; MP 9.1 "never by side-steps over the dock"): a lateral error found nearer is fixed by walking
    straight out to the staging point first.
    verify: error < 3 cm and < 5 deg (O-20) -> docked
            else back off straight out (at least 0.30 m, MP 9 "back off 0.3 m and retry (2), then assist") to the
            staging point -> align -> entry; third failure -> assist (reason ``dock_failed``)
    entry aborts early (back off) when |lateral| or |yaw| leave the entry cone.
    A safe hold (e-stop) freezes the state machine with the gait released; so does a crown raised mid-dock (MP 8 "no
    walking until stowed"; assist ``crown_gate`` after CROWN_WAIT_S); a phase timeout -> assist.

Charge (``TimeCompressedCharger``): CC at ``cc_a`` up to ``cv_soc``, then a CV taper I = cc_a (1 - soc) /
(1 - cv_soc) down to ``cutoff_a`` (or ``full_soc``), all in real-equivalent seconds = sim seconds x ``compress``.
It adds the charged energy to a WP7 ``energy_model.Battery`` (``energy_wh``, capped at capacity). ASSUMED numbers.

hub sync stub (``HubSyncStub``, delete-after-receipt rule): closes the run manifest (``controllers.manifest``, final, Merkle root),
marks lane L1 (manifest + leads) sent by the stub and L2 / L3 queued; local data may be deleted only after a hub
receipt naming the same Merkle root. The sim never deletes anything.

DOCK / UNLOAD (``DockUnloadSession``, MP 9 steps 1-6): park alongside the crate, WP3 unload, job sheets (WP9
callback), dock, charge, sync, with an outputs dict (crate count and kg per tree, dock state, manifest closed).
A charge fault (off the dock tolerance for CHARGE_CONTACT_S outside a hold: ``charge_contact_lost``; no "full" within
the analytic time x CHARGE_TIMEOUT_FACTOR + pad: ``charge_timeout``) stops the charge; sync still closes the manifest,
then the session goes to assist with that reason.
"""
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DOCK_TOL_M = 0.03            # O-20 / MP 9: dock error < 3 cm
DOCK_TOL_DEG = 5.0           # O-20 / MP 9: and < 5 deg
STAGING_M = 0.95             # staging point in front of the plate along the dock axis: every foot clear of the
                             # default plate (CLEAR, below) with 10 cm to spare, so the strafe there is off the plate
PLATE_HALF_M = 0.50          # WP4 plate half-length along the dock axis (field dock.size_m [1.0, 0.8])
FOOT_REACH_M = 0.35          # MEASURED (orchard_demo home keyframe): outer feet 0.26 m fore / aft of the trunk, 0.03 m
                             # radius; + 0.06 m ASSUMED gait swing. Trunk this far past the plate edge = feet off it
BACKOFF_M = 0.30             # MP 9: back off 0.3 m and retry
MAX_RETRIES = 2              # MP 9: retry (2), then assist
APPROACH_TOL_M = 0.10        # walking to the staging point: close enough to start the fine alignment
APPROACH_MAX_M = 4.0         # farther than this from the staging point: RETURN HOME first (WP28)
STUCK_S = 8.0                # commanding motion but moved < STUCK_M and turned < STUCK_DEG for this long -> stuck
STUCK_M = 0.02
STUCK_DEG = 3.0
REASON_STUCK = "stuck"
ALIGN_TOL_M = 0.012          # fine alignment at staging: lateral aim (well inside the 3 cm budget)
ALIGN_TOL_DEG = 2.0          # fine alignment: yaw aim
ALIGN_MIN_M = 0.20           # the entry starts between these distances before the target (the along error at
ALIGN_MAX_M = 1.05           # staging does not matter: the entry servoes it out)
MOVE_MIN_S = 0.12            # an alignment move runs at least this long before the coast check may stop it
YAW_CMD_MIN = 0.06
VY_CMD_MIN = 0.12
ENTRY_CONE_M = 0.045         # entry aborts (back off) if |lateral| grows past this
ENTRY_CONE_DEG = 6.0         # or |yaw error| past this
ENTRY_VX_MIN = 0.15          # normalised creep command (MEASURED 0.027 m/s on orchard_demo flat at the plate)
ENTRY_VX_MAX = 0.50
ENTRY_GAIN = 1.2             # vx = clamp(VX_MIN + GAIN * remaining)
YAW_HOLD_GAIN = 2.0          # yaw command per rad of heading error during entry / back-off
# Gait coast after ``release()`` (the gait finishes its cycle): MEASURED on orchard_demo flat, 3 s at each vx,
# 2026-11-17: vx 0.15 -> 19.5 mm, 0.3 -> 39.6 mm, 0.5 -> 64.4 mm, 1.0 -> 115 mm; fit 3 mm + 112 mm per unit command.
COAST_M0 = 0.003
COAST_M_PER_CMD = 0.112
SETTLE_SPEED_MPS = 0.01      # standing: trunk speed below this ...
SETTLE_RATE_RPS = 0.03       # ... and yaw rate below this ...
SETTLE_S = 0.4               # ... for this long
ALIGN_MAX_ITER = 12          # fine-alignment move/settle rounds before the attempt counts as failed
PHASE_TIMEOUT_S = {"approach": 90.0, "align": 40.0, "entry": 40.0, "backoff": 20.0, "settle": 6.0}

STATES = ("idle", "approach", "align", "entry", "settle", "verify", "backoff", "docked", "assist")
REASON_DOCK_FAILED = "dock_failed"        # RTH-DOCKING E.8 / MP 0.5 absorb note
REASON_CROWN = "crown_gate"               # MP 9 precondition: crown stowed
REASON_NO_DOCK = "no_dock"                # world without the WP4 dock
REASON_CONTACT = "charge_contact_lost"    # charging: off the dock tolerance for CHARGE_CONTACT_S
REASON_CHARGE_TIMEOUT = "charge_timeout"  # charging: no "full" within the analytic time x factor + pad
CROWN_WAIT_S = 30.0          # crown raised mid-dock: hold (MP 8 "no walking until stowed") at most this long -> assist
CHARGE_CONTACT_S = 2.0       # charging off the dock tolerance (not in a hold) this long -> charge fault
CHARGE_TIMEOUT_FACTOR = 1.5  # charge step limit = analytic sim time x factor + pad (hold time not counted)
CHARGE_TIMEOUT_PAD_S = 30.0
COMPRESS_MAX = 1.0e5         # NANDI_DOCK_CHARGE_X upper bound (real s per sim s)
CHARGE_MAX_SUBSTEPS = 1000   # per charger step (bounds the CPU per tick whatever the compression)
CHARGE_CONTACT_TOL_M = 0.045  # charging: contact band (= ENTRY_CONE_M), wider than the 3 cm entry gate (hysteresis: a dock
                             # verified near the edge does not fault on a standing-gait settle or a small nudge)
CHARGE_CONTACT_TOL_DEG = 7.5  # and 1.5 x the 5 deg gate
UNLOAD_RETRY_S = 0.5         # unload interlock refusal (gait_moving / arm_not_parked / estop / busy): retry at most this often
UNLOAD_WAIT_S = 30.0         # interlock still refusing after this long (hold time not counted) -> assist with its reason
REASON_NO_CRATE_PARK = "no_crate_park"   # items in the tray but no crate park pose: WP3 unload skipped (recorded)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def coast_m(vx_cmd: float) -> float:
    """Predicted trunk travel after ``release()`` at a held normalised command (MEASURED fit, see COAST_*)."""
    return COAST_M0 + COAST_M_PER_CMD * abs(vx_cmd)


def yaw_coast_deg(yaw_cmd: float) -> float:
    """Predicted heading change after ``release()`` while turning in place (MEASURED orchard_demo flat, 2 s turns:
    yaw 0.2 -> 2.8-4.5 deg, 0.5 -> 7.0-10 deg, 1.0 -> 12.6 deg; fit 0.5 deg + 14 deg per unit command)."""
    return 0.5 + 14.0 * abs(yaw_cmd)


def strafe_coast_m(vy_cmd: float) -> float:
    """Predicted lateral travel after ``release()`` while strafing (MEASURED: vy 0.12 -> 12 mm, 0.25 -> 31 mm,
    0.5 -> 50 mm; fit 2 mm + 100 mm per unit command, conservative at the low end)."""
    return 0.002 + 0.10 * abs(vy_cmd)


# ---------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class DockError:
    remaining_m: float      # still to go along the entry direction (> 0 short of the target, < 0 overshoot)
    lateral_m: float        # across the dock axis (+ = dock-frame +y)
    yaw_deg: float          # robot heading minus dock heading, wrapped
    dist_m: float           # horizontal trunk-to-target distance

    def ok(self, tol_m: float = DOCK_TOL_M, tol_deg: float = DOCK_TOL_DEG) -> bool:
        return self.dist_m < tol_m and abs(self.yaw_deg) < tol_deg

    def to_dict(self) -> dict:
        return {"remaining_m": round(self.remaining_m, 4), "lateral_m": round(self.lateral_m, 4),
                "yaw_deg": round(self.yaw_deg, 3), "dist_m": round(self.dist_m, 4)}


@dataclass(frozen=True)
class DockGeometry:
    x: float
    y: float
    yaw: float                     # rad, docked heading
    entry: int = -1                # -1 reverse in (backboard behind the plate, WP4 layout), +1 forward in
    staging_m: float = STAGING_M
    plate_half_m: float = PLATE_HALF_M   # 0: no plate (the crate park pose)

    def __post_init__(self):
        if self.entry not in (-1, 1) or not self.staging_m > 0 or not self.plate_half_m >= 0:
            raise ValueError("entry must be -1 or +1, staging_m > 0 and plate_half_m >= 0")

    @property
    def clear_m(self) -> float:
        """Smallest ``remaining`` with every foot off the plate (MP 9.1: no side steps over the dock)."""
        return self.plate_half_m + FOOT_REACH_M if self.plate_half_m > 0 else 0.0

    @property
    def axis(self) -> Tuple[float, float]:
        return math.cos(self.yaw), math.sin(self.yaw)

    @property
    def entry_dir(self) -> Tuple[float, float]:
        """World direction the trunk moves while entering (entry -1: backwards, opposite the docked heading)."""
        ux, uy = self.axis
        return self.entry * ux, self.entry * uy

    def point_before(self, d_m: float) -> Tuple[float, float]:
        """The point ``d_m`` before the target on the entry line."""
        ex, ey = self.entry_dir
        return self.x - d_m * ex, self.y - d_m * ey

    def staging_xy(self) -> Tuple[float, float]:
        return self.point_before(self.staging_m)

    def error(self, x: float, y: float, yaw: float) -> DockError:
        ex, ey = self.entry_dir
        dx, dy = self.x - x, self.y - y
        remaining = dx * ex + dy * ey
        ux, uy = self.axis
        lateral = -(-uy * dx + ux * dy)          # robot offset along the dock-frame +y
        return DockError(remaining, lateral, math.degrees(wrap(yaw - self.yaw)), math.hypot(dx, dy))

    @classmethod
    def from_field(cls, field: dict) -> "DockGeometry":
        dk = field["dock"]
        x, y = (float(v) for v in dk["xy"])
        entry = dk.get("entry", "reverse")
        size = dk.get("size_m") or [2 * PLATE_HALF_M]
        return cls(x, y, math.radians(float(dk.get("yaw_deg", 0.0))), -1 if entry == "reverse" else 1,
                   float(dk.get("staging_m", STAGING_M)), 0.5 * float(size[0]))

    @classmethod
    def from_model(cls, m, d=None, entry: int = -1, staging_m: float = STAGING_M) -> Optional["DockGeometry"]:
        """From the WP4 ``dock`` body / ``dock_ref`` site of a compiled model (None without a dock)."""
        import mujoco
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "dock")
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "dock_ref")
        if bid < 0 or sid < 0:
            return None
        # world-static body: compute from the model so no mj_forward is needed
        w, qx, qy, qz = (float(v) for v in m.body_quat[bid])
        yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        bx, by = float(m.body_pos[bid][0]), float(m.body_pos[bid][1])
        sx, sy = float(m.site_pos[sid][0]), float(m.site_pos[sid][1])
        c, s = math.cos(yaw), math.sin(yaw)
        st = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "dock_staging")
        if st >= 0:                                # WP13 module: staging point on the entry line
            off = float(m.site_pos[st][0]) - sx
            if abs(off) > 1e-6:
                entry, staging_m = (-1 if off > 0 else 1), abs(off)
        pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "dock_plate")
        half = float(m.geom_size[pg][0]) if pg >= 0 else PLATE_HALF_M
        return cls(bx + c * sx - s * sy, by + s * sx + c * sy, yaw, entry, staging_m, half)


# ---------------------------------------------------------------------- aligner
@dataclass
class Obs:
    t: float
    x: float
    y: float
    yaw: float
    speed: float = 0.0          # trunk horizontal speed m/s
    yaw_rate: float = 0.0       # rad/s
    walking: bool = False       # the gait still has phase to run
    holding: bool = False       # safe hold / e-stop
    crown_stowed: bool = True


@dataclass
class Cmd:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    release: bool = True        # True: no command, finish the gait cycle and stand

    @classmethod
    def go(cls, vx=0.0, vy=0.0, yaw=0.0) -> "Cmd":
        return cls(clamp(vx, -1, 1), clamp(vy, -1, 1), clamp(yaw, -1, 1), False)


RELEASE = Cmd()


class DockAligner:
    """Dock alignment state machine (module docstring). ``update(obs) -> Cmd``; ``status()`` for the console."""

    def __init__(self, geom: DockGeometry, tol_m: float = DOCK_TOL_M, tol_deg: float = DOCK_TOL_DEG,
                 max_retries: int = MAX_RETRIES, backoff_m: float = BACKOFF_M):
        self.g = geom
        self.tol_m, self.tol_deg = float(tol_m), float(tol_deg)
        self.max_retries, self.backoff_m = int(max_retries), float(backoff_m)
        self.reset()

    def reset(self):
        self.state = "idle"
        self.reason = ""
        self.retries = 0
        self.detail = ""
        self._prog: Optional[Tuple[float, float, float, float]] = None
        self.attempts: List[dict] = []     # one per verify: {t, error, ok} (and entry aborts)
        self.t_start: Optional[float] = None
        self.t_phase: Optional[float] = None
        self.t_docked: Optional[float] = None
        self.paused = False
        self.final: Optional[DockError] = None
        self.last_error: Optional[DockError] = None
        self._after_settle = "verify"
        self._settle_since: Optional[float] = None
        self._align_iter = 0
        self._moving = False
        self._move_t0 = 0.0
        self._move_axis = ""
        self._last_sign: Dict[str, bool] = {}
        self._floor: Dict[str, float] = {}
        self._entry_vx = 0.0
        self._backoff_from: Optional[Tuple[float, float]] = None
        self._crown_since: Optional[float] = None
        self.strafe_min_remaining_m: Optional[float] = None   # smallest ``remaining`` at which a strafe ran
        self._log: List[Tuple[float, str, str]] = []

    # -- control
    def start(self, t: float, crown_stowed: bool = True, xy: Optional[Sequence[float]] = None) -> bool:
        """Start docking. Refused (assist) with the crown up, or farther than APPROACH_MAX_M from the staging point
        (the long way home is RETURN HOME, WP28; this approach is a straight walk across the open home apron)."""
        self.reset()
        self.t_start = t
        if not crown_stowed:
            self._to("assist", t, REASON_CROWN)
            return False
        if xy is not None:
            sx, sy = self.g.staging_xy()
            if math.hypot(sx - float(xy[0]), sy - float(xy[1])) > APPROACH_MAX_M:
                self.detail = "too_far"
                self._to("assist", t, REASON_DOCK_FAILED)
                return False
        self._to("approach", t)
        return True

    @property
    def done(self) -> bool:
        return self.state in ("docked", "assist")

    @property
    def docked(self) -> bool:
        return self.state == "docked"

    def _to(self, state: str, t: float, reason: str = ""):
        self.state, self.t_phase = state, t
        if reason:
            self.reason = reason
        self._moving = False
        self._settle_since = None
        self._log.append((round(t, 3), state, reason))

    def _settle_then(self, nxt: str, t: float):
        self._after_settle = nxt
        self._to("settle", t)

    def _timed_out(self, t: float) -> bool:
        lim = PHASE_TIMEOUT_S.get(self.state)
        return lim is not None and self.t_phase is not None and t - self.t_phase > lim

    def update(self, o: Obs) -> Cmd:
        if self.state in ("idle", "docked", "assist"):
            return RELEASE
        self.last_error = self.g.error(o.x, o.y, o.yaw)
        if o.holding:                          # e-stop: freeze, gait released; the phase clock restarts on resume
            self.paused = True
            self.t_phase = o.t
            self._settle_since = None
            return RELEASE
        if not o.crown_stowed:                 # MP 9 precondition / MP 8 "no walking until stowed": hold the same way
            if self._crown_since is None:
                self._crown_since = o.t
                self.attempts.append({"t": round(o.t, 3), "event": "crown_up", "phase": self.state})
            self.paused = True
            self.t_phase = o.t
            self._settle_since = None
            self.detail = "crown_up"
            if o.t - self._crown_since > CROWN_WAIT_S:
                self._to("assist", o.t, REASON_CROWN)
            return RELEASE
        if self._crown_since is not None:
            self._crown_since = None
            if self.detail == "crown_up":
                self.detail = ""
        if self.paused:
            self.paused = False
            self.t_phase = o.t
            if self.state in ("entry", "backoff", "align"):   # re-evaluate from a standstill
                self._settle_then("align" if self.state != "entry" else "verify_entry", o.t)
        if self._timed_out(o.t):
            phase = self.state
            self.detail = f"timeout_{phase}"
            self.attempts.append({"t": round(o.t, 3), "event": "timeout", "phase": phase})
            self._to("assist", o.t, REASON_DOCK_FAILED)
            return RELEASE
        cmd = getattr(self, "_" + self.state)(o)
        if cmd.release or self.state in ("idle", "docked", "assist", "settle"):
            self._prog = None
            return cmd
        if self._prog is None:
            self._prog = (o.t, o.x, o.y, o.yaw)
        elif (math.hypot(o.x - self._prog[1], o.y - self._prog[2]) >= STUCK_M
              or abs(math.degrees(wrap(o.yaw - self._prog[3]))) >= STUCK_DEG):
            self._prog = (o.t, o.x, o.y, o.yaw)
        elif o.t - self._prog[0] > STUCK_S:
            self.detail = f"stuck_{self.state}"
            self.attempts.append({"t": round(o.t, 3), "event": "stuck", "phase": self.state})
            self._to("assist", o.t, REASON_STUCK)
            return RELEASE
        return cmd

    # -- phases
    def _approach(self, o: Obs) -> Cmd:
        sx, sy = self.g.staging_xy()
        dist = math.hypot(sx - o.x, sy - o.y)
        if dist <= APPROACH_TOL_M:
            self._align_iter = 0
            self._settle_then("align", o.t)
            return RELEASE
        ang = wrap(math.atan2(sy - o.y, sx - o.x) - o.yaw)
        if abs(ang) > 0.35:
            return Cmd.go(0.0, 0.0, 1.0 if ang > 0 else -1.0)
        return Cmd.go(clamp(1.5 * dist, 0.3, 1.0), 0.0, 1.5 * ang)

    def _align(self, o: Obs) -> Cmd:
        """Fine alignment on the entry line: yaw to the dock heading, then the lateral offset,
        then the along-axis distance into [ALIGN_MIN_M, ALIGN_MAX_M] before the target. One degree of freedom per
        move; each move stops early by the predicted gait coast, settles, and is re-checked."""
        e = self.last_error
        yaw_err = math.radians(-e.yaw_deg)
        # the aims relax after half the rounds (still well inside the 3 cm / 5 deg budget): a gait whose smallest
        # step is longer than the aim would otherwise limit-cycle
        relax = min(1.5, 1.0 + max(0, self._align_iter - ALIGN_MAX_ITER // 2) / (ALIGN_MAX_ITER // 2))
        yaw_ok = abs(e.yaw_deg) <= ALIGN_TOL_DEG * relax
        lat_ok = abs(e.lateral_m) <= ALIGN_TOL_M * relax
        along_ok = ALIGN_MIN_M <= e.remaining_m <= max(ALIGN_MAX_M, self._stage_m() + 0.10)
        if not self._moving:
            if yaw_ok and lat_ok and along_ok:
                self._entry_start(o)
                return RELEASE
            self._align_iter += 1
            if self._align_iter > ALIGN_MAX_ITER:
                return self._fail(o, "align_not_converged")
            self._moving = True
            self._move_t0 = o.t
            # MP 9.1: never a side step with a foot on the plate. A lateral error found nearer than ``clear_m``
            # is fixed by walking straight out to the staging point first (the along move), then strafing there
            lat_clear = e.remaining_m >= self.g.clear_m
            self._move_axis = ("yaw" if not yaw_ok else "lat" if not lat_ok and lat_clear
                               else "along")
            # an overshoot (this move on the same axis goes the other way) halves that axis' command floor
            sign = {"yaw": -e.yaw_deg, "lat": -e.lateral_m, "along": e.remaining_m}[self._move_axis] >= 0
            prev = self._last_sign.get(self._move_axis)
            if prev is not None and prev != sign:
                self._floor[self._move_axis] = max(0.25, self._floor.get(self._move_axis, 1.0) * 0.5)
            self._last_sign[self._move_axis] = sign
        early = o.t - self._move_t0 < MOVE_MIN_S
        axis = self._move_axis
        if axis == "yaw":
            mag = clamp(abs(yaw_err) * 2.0, YAW_CMD_MIN * self._floor.get("yaw", 1.0), 0.6)
            if not early and abs(e.yaw_deg) <= yaw_coast_deg(mag):
                self._settle_then("align", o.t)
                return RELEASE
            return Cmd.go(0.0, 0.0, math.copysign(mag, yaw_err))
        if axis == "lat":
            # lateral error in the robot frame (yaw is aligned to within ALIGN_TOL_DEG here): dock +y = robot +y
            # when the docked heading is the dock yaw; strafe toward -lateral
            mag = clamp(abs(e.lateral_m) * 2.5, VY_CMD_MIN * self._floor.get("lat", 1.0), 0.4)
            if e.remaining_m < self.g.clear_m:          # drifted toward the plate mid-strafe: stop, walk out first
                self._settle_then("align", o.t)
                return RELEASE
            if not early and abs(e.lateral_m) <= strafe_coast_m(mag):
                self._settle_then("align", o.t)
                return RELEASE
            r = e.remaining_m
            self.strafe_min_remaining_m = r if self.strafe_min_remaining_m is None else min(
                self.strafe_min_remaining_m, r)
            return Cmd.go(0.0, math.copysign(mag, -e.lateral_m), YAW_HOLD_GAIN * yaw_err)
        # along: straight along the entry line to the staging point (positive = move along the entry direction)
        want = self._stage_m()
        go = e.remaining_m - want
        mag = clamp(abs(go) * 2.0, 0.2, 0.8)
        if not early and abs(go) <= coast_m(mag) + 0.02:
            self._settle_then("align", o.t)
            return RELEASE
        return Cmd.go(self.g.entry * math.copysign(mag, go), 0.0, YAW_HOLD_GAIN * yaw_err)

    def _stage_m(self) -> float:
        """Where along moves and back-offs end: the staging point, at least 8 cm past ``clear_m``."""
        return max(self.g.staging_m, self.g.clear_m + 0.08)

    def _entry_start(self, o: Obs):
        self._to("entry", o.t)

    def _entry(self, o: Obs) -> Cmd:
        e = self.last_error
        if abs(e.lateral_m) > ENTRY_CONE_M or abs(e.yaw_deg) > ENTRY_CONE_DEG:
            self.attempts.append({"t": round(o.t, 3), "event": "entry_abort", "error": e.to_dict()})
            return self._retry(o, "entry_cone")
        vx = clamp(ENTRY_VX_MIN + ENTRY_GAIN * max(0.0, e.remaining_m), ENTRY_VX_MIN, ENTRY_VX_MAX)
        if e.remaining_m <= coast_m(vx):
            self._settle_then("verify", o.t)
            return RELEASE
        self._entry_vx = vx
        yaw_err = math.radians(-e.yaw_deg)
        return Cmd.go(self.g.entry * vx, 0.0, YAW_HOLD_GAIN * yaw_err)

    def _settle(self, o: Obs) -> Cmd:
        still = (not o.walking) and o.speed < SETTLE_SPEED_MPS and abs(o.yaw_rate) < SETTLE_RATE_RPS
        if not still:
            self._settle_since = None
            return RELEASE
        if self._settle_since is None:
            self._settle_since = o.t
        if o.t - self._settle_since < SETTLE_S:
            return RELEASE
        nxt = self._after_settle
        if nxt == "verify_entry":          # resumed after a hold mid-entry: continue unless already in
            e = self.last_error
            nxt = "verify" if e.remaining_m <= coast_m(ENTRY_VX_MIN) else "entry"
        self._to(nxt, o.t)
        return self.update(o) if nxt not in ("verify",) else self._verify(o)

    def _verify(self, o: Obs) -> Cmd:
        e = self.last_error
        ok = e.ok(self.tol_m, self.tol_deg)
        self.attempts.append({"t": round(o.t, 3), "event": "verify", "ok": ok, "error": e.to_dict()})
        if ok:
            self.final = e
            self.t_docked = o.t
            self._to("docked", o.t)
            return RELEASE
        return self._retry(o, "misaligned")

    def _retry(self, o: Obs, why: str) -> Cmd:
        if self.retries >= self.max_retries:
            self.final = self.last_error
            self._to("assist", o.t, REASON_DOCK_FAILED)
            return RELEASE
        self.retries += 1
        self._backoff_from = (o.x, o.y)
        self._to("backoff", o.t, why)
        return RELEASE

    def _fail(self, o: Obs, why: str) -> Cmd:
        self.attempts.append({"t": round(o.t, 3), "event": why})
        return self._retry(o, why)

    def _backoff(self, o: Obs) -> Cmd:
        """Straight out along the dock axis (MP 9: at least ``backoff_m``) to the staging point, every foot off the
        plate, then re-align there: the retry's lateral correction is never a side step over the dock (MP 9.1)."""
        e = self.last_error
        vx = 0.5
        if e.remaining_m >= max(self.backoff_m, self._stage_m()) - coast_m(vx):
            self._align_iter = 0
            self._settle_then("align", o.t)
            return RELEASE
        yaw_err = math.radians(-e.yaw_deg)
        return Cmd.go(-self.g.entry * vx, 0.0, YAW_HOLD_GAIN * yaw_err)

    # -- report
    def status(self) -> dict:
        e = self.last_error
        return {"state": self.state, "reason": self.reason, "detail": self.detail, "retries": self.retries,
                "paused": self.paused,
                "strafe_min_remaining_m": None if self.strafe_min_remaining_m is None else round(
                    self.strafe_min_remaining_m, 4),
                "error": None if e is None else e.to_dict(),
                "final": None if self.final is None else self.final.to_dict(),
                "tol": {"m": self.tol_m, "deg": self.tol_deg},
                "dock_s": None if self.t_docked is None or self.t_start is None else round(self.t_docked -
                                                                                        self.t_start, 3),
                "attempts": list(self.attempts),
                "log": [list(x) for x in self._log[-20:]]}


# ---------------------------------------------------------------------- charger
@dataclass(frozen=True)
class ChargeParams:
    cc_a: float = 10.0          # ASSUMED charger current (about 0.5 C of the 19.5 Ah WP7 pack)
    cv_soc: float = 0.90        # ASSUMED CC -> CV hand-over
    cutoff_a: float = 1.0       # ASSUMED CV termination (about C/20)
    full_soc: float = 0.999
    eta: float = 0.95           # ASSUMED charge efficiency (wall -> cell)
    compress: float = 240.0     # real seconds per sim second (time-compressed charge, MVP-SPEC 4)

    def __post_init__(self):
        vals = (self.cc_a, self.cv_soc, self.cutoff_a, self.full_soc, self.eta, self.compress)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError("invalid charge params: every value must be a finite number")
        if not (self.cc_a > 0 and 0 < self.cv_soc < 1 and 0 < self.cutoff_a < self.cc_a and 0 < self.eta <= 1
                and 1 <= self.compress <= COMPRESS_MAX and self.cv_soc < self.full_soc <= 1):
            raise ValueError("invalid charge params")

    @classmethod
    def from_env(cls) -> "ChargeParams":
        """``NANDI_DOCK_CHARGE_X``: a finite number in [1, COMPRESS_MAX]; anything else is a ValueError naming the
        variable (raised when the session is built, never inside the 50 Hz task)."""
        raw = os.environ.get("NANDI_DOCK_CHARGE_X", "").strip()
        if not raw:
            return cls()
        msg = f"NANDI_DOCK_CHARGE_X={raw!r}: must be a finite number in [1, {COMPRESS_MAX:g}]"
        try:
            return cls(compress=float(raw))
        except ValueError:
            raise ValueError(msg) from None


def nominal_v(battery) -> float:
    from atharv_nandi.core.energy_model import ASSUMED_NOMINAL_CELL_V
    return ASSUMED_NOMINAL_CELL_V * float(battery.p.cells_s)


def charge_current_a(soc: float, p: ChargeParams) -> float:
    """CC below ``cv_soc``, then the linear CV taper to 0 at soc 1 (ASSUMED shape)."""
    if soc < p.cv_soc:
        return p.cc_a
    return p.cc_a * max(0.0, 1.0 - soc) / (1.0 - p.cv_soc)


def analytic_charge_s(soc0: float, p: ChargeParams, capacity_ah: float) -> float:
    """Real-equivalent seconds from ``soc0`` to termination (cutoff current or full_soc), closed form."""
    t = 0.0
    s = max(0.0, min(1.0, soc0))
    if s < p.cv_soc:
        t += (p.cv_soc - s) * capacity_ah * 3600.0 / (p.cc_a * p.eta)
        s = p.cv_soc
    # CV: ds/dt = eta I / C = eta cc (1 - s) / ((1 - cv) C)  ->  1 - s decays with tau = (1 - cv) C / (eta cc)
    tau = (1.0 - p.cv_soc) * capacity_ah * 3600.0 / (p.eta * p.cc_a)
    s_cut = max(1.0 - p.cutoff_a * (1.0 - p.cv_soc) / p.cc_a, s)
    s_end = min(p.full_soc, s_cut)
    if s_end > s:
        t += tau * math.log((1.0 - s) / (1.0 - s_end))
    return t


class TimeCompressedCharger:
    """Charges a WP7 ``Battery`` while docked (``step(t, battery)`` each tick). States idle | charging | full."""

    def __init__(self, params: ChargeParams = ChargeParams()):
        self.p = params
        self.reset()

    def reset(self):
        self.state = "idle"
        self.t0: Optional[float] = None
        self.t_last: Optional[float] = None
        self.t_full: Optional[float] = None
        self.soc0: Optional[float] = None
        self.current_a = 0.0
        self.charged_wh = 0.0
        self.real_s = 0.0

    def start(self, t: float, battery):
        self.reset()
        self.state, self.t0, self.t_last, self.soc0 = "charging", t, t, float(battery.soc)

    def stop(self, t: float):
        if self.state == "charging":
            self.state = "idle"
        self.current_a = 0.0
        self.t_last = t

    def step(self, t: float, battery, enabled: bool = True) -> float:
        """Integrate since the last call; ``enabled`` False (e-stop, contact lost) charges nothing. Returns soc."""
        if self.state != "charging":
            return float(battery.soc)
        dt = max(0.0, t - (self.t_last if self.t_last is not None else t))
        self.t_last = t
        if not enabled:
            self.current_a = 0.0
            return float(battery.soc)
        cap_wh = float(battery.p.capacity_wh)
        real = dt * self.p.compress
        # sub-steps keep the CV taper exact enough at large compression (<= 10 real s each)
        n = min(CHARGE_MAX_SUBSTEPS, max(1, int(math.ceil(real / 10.0)))) if math.isfinite(real) else 1
        h = real / n
        for _ in range(n):
            soc = battery.energy_wh / cap_wh
            i = charge_current_a(soc, self.p)
            if i < self.p.cutoff_a - 1e-12 or soc >= self.p.full_soc:
                self._full(t, i)
                return float(battery.soc)
            d_ah = self.p.eta * i * h / 3600.0
            d_wh = min(d_ah * nominal_v(battery), cap_wh - battery.energy_wh)
            battery.energy_wh += d_wh
            self.charged_wh += d_wh
            self.real_s += h
            self.current_a = i
        return float(battery.soc)

    def _full(self, t: float, i: float):
        self.state, self.t_full, self.current_a = "full", t, 0.0

    def eta_real_min(self, soc: float, capacity_ah: float) -> float:
        if self.state != "charging":
            return 0.0
        return analytic_charge_s(soc, self.p, capacity_ah) / 60.0

    def status(self, battery=None) -> dict:
        out = {"state": self.state, "current_a": round(self.current_a, 3), "charged_wh": round(self.charged_wh, 3),
               "real_min": round(self.real_s / 60.0, 2), "compress": self.p.compress,
               "sim_s": None if self.t0 is None else round((self.t_full or self.t_last or self.t0) - self.t0, 3),
               "soc0": None if self.soc0 is None else round(self.soc0, 4), "assumed": True}
        if battery is not None:
            cap_ah = float(battery.p.capacity_wh) / nominal_v(battery)
            out["soc"] = round(float(battery.soc), 4)
            out["eta_real_min"] = round(self.eta_real_min(float(battery.soc), cap_ah), 1)
            out["eta_sim_s"] = round(out["eta_real_min"] * 60.0 / self.p.compress, 1)
        return out


# ---------------------------------------------------------------------- hub sync stub
LANES = ("L1", "L2", "L3")   # MP 0.6: L1 manifest + leads, L2 320 px thumbnails, L3 full frames (hub drive)


class HubSyncStub:
    """Closes the run manifest and queues the uplink lanes; local deletion only after a matching hub receipt."""

    def __init__(self, sync: bool = True):
        self.sync = bool(sync)
        self.reset()

    def reset(self):
        self.manifest: Optional[dict] = None
        self.run_dir: Optional[str] = None
        self.closed = False
        self.lanes: Dict[str, str] = {k: "idle" for k in LANES}
        self.receipt: Optional[dict] = None
        self.refusals: List[str] = []

    def close(self, run_dir: Optional[str] = None, meta: Optional[dict] = None, closed_ms: Optional[int] = None,
              chunks: Optional[Sequence[dict]] = None) -> dict:
        """Close the manifest: write ``manifest.json`` (final) for ``run_dir``, or build one in memory from
        ``chunks`` ``[{path, sha256, bytes?, kind?}]``. Idempotent: a second close returns the first manifest."""
        from atharv_nandi.core import manifest as mf
        if self.closed:
            return self.manifest
        meta = dict(meta or {})
        if run_dir is not None:
            self.manifest = mf.write_manifest(run_dir, meta, final=True, closed_ms=closed_ms)
            self.run_dir = run_dir
        else:
            ch = sorted((dict(c) for c in (chunks or [])), key=lambda c: c["path"])
            self.manifest = {"v": 1, "mission_id": meta.get("run_id"), "final": True, "closed": closed_ms,
                             "chunks": ch, "merkle_root": mf.root_of(ch), "data_provenance": "simulated"}
        self.closed = True
        self.manifest["sync"] = "on" if self.sync else "off"
        if self.sync:
            self.lanes = {"L1": "sent_stub", "L2": "queued", "L3": "queued"}
        return self.manifest

    @property
    def merkle_root(self) -> Optional[str]:
        return None if self.manifest is None else self.manifest["merkle_root"]

    def verify(self) -> Tuple[bool, str]:
        """Recompute the root from disk (run_dir) and compare; (False, 'not_closed') before close."""
        from atharv_nandi.core import manifest as mf
        if not self.closed:
            return False, "not_closed"
        if self.run_dir is None:
            return True, self.merkle_root
        return mf.verify(self.run_dir)

    def hub_receipt(self, receipt: dict) -> bool:
        """A hub receipt ``{merkle_root, lanes?: [..]}``: accepted only for the closed manifest's root."""
        if not self.closed or not isinstance(receipt, dict) or receipt.get("merkle_root") != self.merkle_root:
            self.refusals.append("receipt_mismatch")
            return False
        self.receipt = dict(receipt)
        for k in receipt.get("lanes", LANES):
            if k in self.lanes:
                self.lanes[k] = "received"
        return True

    def may_delete_local(self) -> Tuple[bool, str]:
        """delete-after-receipt rule: local data may go only after a hub receipt for every lane. The sim never deletes."""
        if not self.closed:
            return False, "not_closed"
        if self.receipt is None:
            return False, "no_hub_receipt"
        if any(v != "received" for v in self.lanes.values()):
            return False, "lanes_pending"
        return True, "hub_receipt"

    def status(self) -> dict:
        ok, why = self.may_delete_local()
        return {"sync": "on" if self.sync else "off", "manifest_closed": self.closed,
                "merkle_root": self.merkle_root, "chunks": 0 if self.manifest is None else len(self.manifest["chunks"]),
                "lanes": dict(self.lanes), "receipt": self.receipt is not None, "may_delete_local": ok,
                "delete_reason": why, "stub": True}


# ---------------------------------------------------------------------- DOCK / UNLOAD session
PARK_OFFSET_M = 0.53     # WP3: crate centre 0.53 m to the right of the trunk axis when parked alongside
PARK_TOL_M = 0.03
PARK_TOL_DEG = 3.0
SESSION_STEPS = ("park_crate", "unload", "jobsheets", "dock", "charge", "sync", "done")


def crate_park_geometry(crate_xy: Sequence[float], heading: float = 0.0, offset_m: float = PARK_OFFSET_M,
                        approach_m: float = 0.5) -> DockGeometry:
    """Park pose with the crate centre ``offset_m`` to the robot's right, heading ``heading`` (WP3 hand-over),
    reached by walking forward along ``heading`` (entry +1) from ``approach_m`` behind it."""
    cx, cy = float(crate_xy[0]), float(crate_xy[1])
    lx, ly = -math.sin(heading), math.cos(heading)            # robot left
    return DockGeometry(cx + offset_m * lx, cy + offset_m * ly, heading, +1, approach_m, 0.0)


class DockUnloadSession:
    """MISSION-PRESETS 9: park alongside the crate -> WP3 unload -> job sheets (WP9) -> dock -> charge -> sync.

    ``unload``: object with WP3's TransferController shape (``unload(d, inputs)``, ``busy()``,
    ``release_crate_slots(d)``, ``crate_summary()``, ``per_tree()``) or None (skip, nothing to unload).
    ``jobsheets(rows) -> dict``: WP9's generator; default records it as pending (WP9 owns job sheets).
    """

    def __init__(self, dock: DockGeometry, crate_park: Optional[DockGeometry] = None, *, battery=None,
                 charger: Optional[TimeCompressedCharger] = None, sync: Optional[HubSyncStub] = None,
                 run_dir: Optional[str] = None, meta: Optional[dict] = None,
                 jobsheets: Optional[Callable[[List[str]], dict]] = None, rows_visited: Sequence[str] = (),
                 crate_id: str = "crate", dock_id: str = "dock", charge: bool = True):
        self.dock_geom, self.crate_geom = dock, crate_park
        self.battery = battery
        self.charger = charger or TimeCompressedCharger(ChargeParams.from_env())
        self.syncer = sync or HubSyncStub(True)
        self.run_dir, self.meta = run_dir, dict(meta or {})
        self.jobsheets_fn = jobsheets
        self.rows_visited = list(rows_visited)
        self.crate_id, self.dock_id, self.do_charge = crate_id, dock_id, bool(charge)
        self.reset()

    def reset(self):
        self.step = "idle"
        self.reason = ""
        self.trigger = ""
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self.aligner: Optional[DockAligner] = None
        self.dock_aligner = DockAligner(self.dock_geom)
        self.park_aligner = None if self.crate_geom is None else DockAligner(
            self.crate_geom, tol_m=PARK_TOL_M, tol_deg=PARK_TOL_DEG)
        self.unload_started = False
        self.unload_refusal = ""
        self.outputs: Dict[str, Any] = {}
        self.charger.reset()
        self.syncer.reset()
        self.events: List[dict] = []
        self._xy: Optional[Tuple[float, float]] = None
        self._crown = True
        self.fault = ""                    # charge fault (contact lost / timeout): sync still runs, then assist
        self._charge_t: Optional[float] = None
        self._charge_active_s = 0.0
        self._charge_limit_s = 0.0
        self._off_dock_since: Optional[float] = None
        self._unload_next_t: Optional[float] = None   # next WP3 unload retry (interlock refusal), rate-limited
        self._unload_wait_s = 0.0                     # time spent refused outside a hold (UNLOAD_WAIT_S)
        self._unload_t: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.step not in ("idle", "done", "assist")

    def _event(self, t, ev, **kw):
        self.events.append({"t": round(t, 3), "event": ev, **kw})

    def start(self, t: float, crown_stowed: bool = True, trigger: str = "operator", unload=None,
              xy: Optional[Sequence[float]] = None) -> bool:
        self.reset()
        self.trigger, self.t_start = trigger, t
        self._crown = bool(crown_stowed)
        self._xy = None if xy is None else (float(xy[0]), float(xy[1]))
        if not crown_stowed:
            self.step, self.reason = "assist", REASON_CROWN
            self._event(t, "refused", reason=REASON_CROWN)
            return False
        items = unload is not None and self._has_items(unload)
        if self.park_aligner is not None and items:
            self._enter("park_crate", t)
        else:
            if items:                       # MP 9 step 2 cannot run without a crate park pose: say so, do not drop it
                self.outputs["crate"] = {"crate_id": self.crate_id, "skipped": REASON_NO_CRATE_PARK,
                                         "tray_n": self._tray_n(unload)}
                self._event(t, "unload_skipped", reason=REASON_NO_CRATE_PARK, tray_n=self._tray_n(unload))
            self._enter("jobsheets", t)
        return True

    @staticmethod
    def _tray_n(unload) -> Optional[int]:
        try:
            return int(sum(v.get("tray_n", 0) for v in unload.per_tree().values()))
        except Exception:
            return None

    @staticmethod
    def _has_items(unload) -> bool:
        try:
            return any(v.get("tray_n", 0) for v in unload.per_tree().values())
        except Exception:
            return True

    def _enter(self, step: str, t: float, reason: str = ""):
        self.step = step
        if reason:
            self.reason = reason
        self._event(t, "step", step=step, reason=reason)
        if step == "park_crate":
            self.aligner = self.park_aligner
            if not self.aligner.start(t, crown_stowed=self._crown, xy=self._xy):
                self.step, self.reason = "assist", self.aligner.reason
        elif step == "dock":
            self.aligner = self.dock_aligner
            if not self.aligner.start(t, crown_stowed=self._crown, xy=self._xy):
                self.step, self.reason = "assist", self.aligner.reason
        elif step != "assist":
            self.aligner = None
        if step == "done":
            self.t_end = t

    def update(self, o: Obs, d=None, unload=None, transfer_inputs=None) -> Cmd:
        """One tick. ``d`` (MjData) and ``unload`` / ``transfer_inputs`` only for the WP3 unload step."""
        if not self.active:
            return RELEASE
        self._xy = (o.x, o.y)
        self._crown = bool(o.crown_stowed)
        if self.step in ("park_crate", "dock"):
            cmd = self.aligner.update(o)
            if self.aligner.state == "assist":
                self._enter("assist", o.t, self.aligner.reason)
            elif self.aligner.docked:
                if self.step == "park_crate":
                    self.outputs["crate_park"] = self.aligner.final.to_dict()
                    self._enter("unload", o.t)
                else:
                    self.outputs["dock"] = self.aligner.status()
                    self._enter("charge" if self.do_charge and self.battery is not None else "sync", o.t)
                    if self.step == "charge":
                        self._charge_start(o.t)
            return cmd
        if self.step == "unload":
            return self._unload(o, d, unload, transfer_inputs)
        if self.step == "jobsheets":
            rows = list(self.rows_visited)
            self.outputs["jobsheets"] = (self.jobsheets_fn(rows) if self.jobsheets_fn is not None
                                         else {"status": "pending_wp9", "rows": rows})
            self._enter("dock", o.t)
            return RELEASE
        if self.step == "charge":
            return self._charge(o)
        if self.step == "sync":
            self.outputs["manifest"] = self.syncer.close(self.run_dir, self.meta, closed_ms=int(round(o.t * 1000)))
            self.outputs["sync"] = self.syncer.status()
            if self.fault:
                self._enter("assist", o.t, self.fault)
            else:
                self._enter("done", o.t)
            return RELEASE
        return RELEASE

    def _charge_start(self, t: float):
        self.charger.start(t, self.battery)
        cap_ah = float(self.battery.p.capacity_wh) / nominal_v(self.battery)
        self._charge_t, self._charge_active_s, self._off_dock_since = t, 0.0, None
        self._charge_limit_s = (analytic_charge_s(float(self.battery.soc), self.charger.p, cap_ah) /
                                self.charger.p.compress * CHARGE_TIMEOUT_FACTOR + CHARGE_TIMEOUT_PAD_S)

    def _charge(self, o: Obs) -> Cmd:
        """Charge while on the dock tolerance and not in a hold. Off the tolerance for CHARGE_CONTACT_S, or no
        "full" within the limit (hold time not counted): stop, record the fault, still close the manifest (sync),
        then assist with the fault as the reason."""
        dt = max(0.0, o.t - (self._charge_t if self._charge_t is not None else o.t))
        self._charge_t = o.t
        on_dock = self.dock_geom.error(o.x, o.y, o.yaw).ok(CHARGE_CONTACT_TOL_M, CHARGE_CONTACT_TOL_DEG)   # band > gate
        self.charger.step(o.t, self.battery, enabled=(not o.holding) and on_dock)
        if self.charger.state == "full":
            self.outputs["charge"] = self.charger.status(self.battery)
            self._enter("sync", o.t)
            return RELEASE
        fault = ""
        if o.holding:
            self._off_dock_since = None
        else:
            self._charge_active_s += dt
            if on_dock:
                self._off_dock_since = None
            elif self._off_dock_since is None:
                self._off_dock_since = o.t
            if self._off_dock_since is not None and o.t - self._off_dock_since >= CHARGE_CONTACT_S - 1e-9:
                fault = REASON_CONTACT
            elif self._charge_active_s > self._charge_limit_s:
                fault = REASON_CHARGE_TIMEOUT
        if fault:
            self.charger.stop(o.t)
            self.fault = fault
            self.outputs["charge"] = {**self.charger.status(self.battery), "fault": fault,
                                      "error": self.dock_geom.error(o.x, o.y, o.yaw).to_dict()}
            self._event(o.t, "charge_fault", reason=fault)
            self._enter("sync", o.t)
        return RELEASE

    def _unload(self, o: Obs, d, unload, inputs) -> Cmd:
        if unload is None:
            self._enter("jobsheets", o.t)
            return RELEASE
        if not self.unload_started:
            dt = max(0.0, o.t - (self._unload_t if self._unload_t is not None else o.t))
            self._unload_t = o.t
            if o.holding:                   # e-stop / safe hold: wait without asking WP3 (no refusal flood), not counted
                return RELEASE
            self._unload_wait_s += dt if self.unload_refusal else 0.0
            if self.unload_refusal and self._unload_wait_s > UNLOAD_WAIT_S:
                self._enter("assist", o.t, self.unload_refusal)       # e.g. the arm never parks: no endless wait
                return RELEASE
            if self._unload_next_t is not None and o.t + 1e-9 < self._unload_next_t:
                return RELEASE              # rate limit: WP3 logs (and publishes) one event per refused call
            g = unload.unload(d, inputs) if inputs is not None else unload.unload(d)
            if not g:
                if g.reason == "empty":
                    self._enter("jobsheets", o.t)
                    return RELEASE
                # interlocks (gait still moving, arm off PARK) clear by themselves: retry next tick;
                # anything else is an assist
                if g.reason in ("gait_moving", "arm_not_parked", "estop", "busy"):
                    if g.reason != self.unload_refusal:
                        self._event(o.t, "unload_wait", reason=g.reason)
                    self.unload_refusal = g.reason
                    self._unload_next_t = o.t + UNLOAD_RETRY_S
                    return RELEASE
                self._enter("assist", o.t, g.reason)
                return RELEASE
            self.unload_started = True
            self._event(o.t, "unload_start")
            return RELEASE
        if unload.busy():
            return RELEASE
        released = unload.release_crate_slots(d)
        self.outputs["crate"] = {"crate_id": self.crate_id, **unload.crate_summary(),
                                 "per_tree": {k: {"crate_n": v.get("crate_n", 0), "crate_kg": v.get("crate_kg", 0.0)}
                                              for k, v in unload.per_tree().items()},
                                 "slots_released": len(released)}
        self._enter("jobsheets", o.t)
        return RELEASE

    def status(self) -> dict:
        al = self.aligner.status() if self.aligner is not None else None
        out = {"step": self.step, "reason": self.reason, "fault": self.fault, "trigger": self.trigger,
               "dock_id": self.dock_id,
               "crate_id": self.crate_id, "aligner": al,
               "dock_state": ("docked" if self.dock_aligner.docked else self.dock_aligner.state),
               "charge": self.charger.status(self.battery) if self.battery is not None else None,
               "sync": self.syncer.status(), "manifest_closed": self.syncer.closed,
               "elapsed_s": None if self.t_start is None or self.t_end is None else round(self.t_end - self.t_start, 3)}
        if "crate" in self.outputs:
            out["crate"] = self.outputs["crate"]
        return out


# ---------------------------------------------------------------------- SimDriver glue (WP0b)
def obs_from_driver(drv) -> Obs:
    """Truth trunk pose / speed from the driver's MjData (the sim stands in for the dock tag)."""
    d = drv.d
    return Obs(t=float(d.time), x=float(d.qpos[0]), y=float(d.qpos[1]), yaw=drv.heading(),
               speed=float(math.hypot(d.qvel[0], d.qvel[1])), yaw_rate=float(d.qvel[5]),
               walking=bool(drv.gait.is_active), holding=bool(drv.holding), crown_stowed=crown_stowed(drv))


CROWN_STOWED_EXT_M = 0.02


def crown_stowed(drv) -> bool:
    """WP24 ``bus.values["mast"]``: stowed = ``ext_m`` <= CROWN_STOWED_EXT_M. Fails closed: a model with a mast but no
    (or an unreadable) mast state counts as raised; only a model without a mast counts as stowed."""
    mast = drv.bus.values.get("mast")
    if isinstance(mast, dict):
        if mast.get("has_mast") is False:
            return True
        try:
            ext = float(mast["ext_m"])
        except (KeyError, TypeError, ValueError):
            ext = float("nan")
        if math.isfinite(ext):
            return ext <= CROWN_STOWED_EXT_M
    try:
        from atharv_nandi.sim.hooks import mast as hm
        return not hm.has_mast(drv.m)
    except Exception:  # pragma: no cover - hooks always importable in the sim tree
        return False


def apply_cmd(drv, cmd: Cmd):
    if cmd.release:
        drv.gait.release()
    else:
        drv.gait.set_command(cmd.vx, cmd.vy, cmd.yaw, hold=True)


PLATFORM_PLAN_TOPIC = "/nandi/platform/plan"
WP13_PLAN = {"route_m": 0.0, "items": 0, "t_work_s": 0.0, "crown_m": 0.0, "return_m": 0.0}   # "busy, nothing left"
ROUTE_RETURN_AND_WAIT = "return_and_wait"    # = route_energy.ACTION_RETURN_AND_WAIT
STATE_TOPIC = "/nandi/dock/state"
EVENTS_TOPIC = "/nandi/dock/events"
STATE_PERIOD_S = 0.5


def _nothing_left(plan) -> bool:
    """A plan with no route, items, crown travel or return left and under 1 s of work: WP13_PLAN or a spent one."""
    return (float(plan.route_m) <= 0.0 and int(plan.items) <= 0 and float(plan.crown_m) <= 0.0
            and float(plan.return_m) <= 0.0 and float(plan.t_work_s) < 1.0)


class DockRunner:
    """Runs a ``DockUnloadSession`` as a SimDriver task (50 Hz) and publishes ``/nandi/dock/state`` (2 Hz and on
    step change) and ``/nandi/dock/events``; ``bus.values["dock"]`` holds the same status.

    ``auto_low_soc``: start the session by itself when the WP7 reserve monitor says ``return_and_wait`` (the
    low-battery return, MVP-SPEC stretch) or the SoC drops below ``energy_model.LOW_SOC``.
    While the session is active it holds a WP13 work plan on the WP7 platform so its instant "parked at home" fill
    does not pre-empt the time-compressed charge (change request to WP7 in the WP13 contract notes), and puts the
    platform's previous plan back afterwards (``_platform_plan``)."""

    def __init__(self, drv, session: DockUnloadSession, *, unload=None, auto_low_soc: bool = False,
                 rate_hz: float = 50.0, name: str = "dock_unload"):
        self.drv, self.s, self.unload_ctrl = drv, session, unload
        self.auto_low_soc = auto_low_soc
        self.name = name
        self._next_pub = 0.0
        self._ev_sent = 0
        self._last_step = None
        self._plan_sent = False
        self._prior_plan: Optional[dict] = None
        self._restore_latch: Optional[tuple] = None
        self._acked = False               # operator acknowledged the assist: the post-assist plan hold may go
        self._kept_logged = False
        drv.add_task(name, self.tick, rate_hz=rate_hz)

    def acknowledge(self):
        """Operator acknowledges an assist (console DOCK/UNLOAD tile, /nandi/dock/cmd [PROPOSED]): releases the plan
        held after the assist, so WP7's dock rule may run again (a robot parked on the plate then reads as docked)."""
        self._acked = True

    def start(self, trigger: str = "operator") -> bool:
        o = obs_from_driver(self.drv)
        return self.s.start(o.t, o.crown_stowed, trigger, unload=self.unload_ctrl, xy=(o.x, o.y))

    def _low_soc(self) -> bool:
        from atharv_nandi.core.energy_model import LOW_SOC
        p = self.drv.bus.values.get("platform") or {}
        if p.get("energy_action") == "return_and_wait":
            return True
        soc = p.get("soc")
        if soc is None and self.s.battery is not None:
            soc = self.s.battery.soc
        return soc is not None and float(soc) < LOW_SOC

    def tick(self, drv):
        o = obs_from_driver(drv)
        if self.auto_low_soc and self.s.step == "idle" and self._low_soc():
            self.s.start(o.t, o.crown_stowed, "energy_reserve", unload=self.unload_ctrl, xy=(o.x, o.y))
        inputs = None
        if self.unload_ctrl is not None and self.s.step == "unload":
            from atharv_nandi.sim.hooks import transfer as ht
            rt = ht.runtime_for(drv.m)
            inputs = rt.inputs(drv.d, drv.bus) if rt is not None else None
        cmd = self.s.update(o, drv.d, self.unload_ctrl, inputs)
        if self.s.active or self.s.step != self._last_step:
            apply_cmd(drv, cmd)
        self._platform_plan(drv, (o.x, o.y))
        self._publish(drv, o.t)

    def _hold_after_assist(self, p, xy) -> bool:
        """After the session ends in assist (charge fault, dock_failed, unload refusal ...) the plan stays held while
        the robot is inside WP7's dock radius of home and the operator has not acknowledged: otherwise WP7's "parked at
        home, no plan" rule fills the pack 5 s later and the console shows a full battery next to a charge fault."""
        if self.s.step != "assist" or self._acked or p is None or xy is None:
            return False
        home = getattr(p, "home", None)
        if home is None:
            return False
        try:
            from atharv_nandi.sim.hooks.platform import DOCK_RADIUS_M as radius
        except Exception:  # pragma: no cover - hooks always importable in the sim tree
            radius = 0.15
        return math.hypot(xy[0] - home[0], xy[1] - home[1]) <= radius + 1e-9

    def _platform_plan(self, drv, xy=None):
        """While the session is active (every step, not only ``charge``) the WP7 platform holds a WP13 work plan,
        so its instant "parked at home, no plan" fill never fires: not in an e-stop near the plate, not during the
        slow entry creep, not between steps. The plan the platform held before (a mission layer's) is put back when
        the session ends, and its reserve latch is kept unless the charge filled the pack (WP7's own dock rule clears
        it after a fill too)."""
        p = platform_runtime(drv.m)
        if self._restore_latch is not None:                 # WP7 has applied our {clear}: put the latch back
            if p is None or p.plan is None:
                if p is not None:
                    act, reason, latched_at = self._restore_latch
                    p.reserve.action, p.reserve.reason, p.reserve.latched_at = act, reason, latched_at
                self._restore_latch = None
        active = self.s.active
        if active:
            self._acked, self._kept_logged = False, False
        if not active and self._plan_sent and self._hold_after_assist(p, xy if xy is not None else self.s._xy):
            if not self._kept_logged:
                self.s._event(drv.t, "platform_plan", action="kept", reason=self.s.reason)
                self._kept_logged = True
            return
        if active and not self._plan_sent:
            prior = None if p is None else p.remaining_plan()
            if prior is not None and _nothing_left(prior):
                prior = None                # e.g. our own plan left by an earlier runner: restoring it would block WP7
            self._prior_plan = None if prior is None else dict(prior.__dict__)
            drv.deliver(PLATFORM_PLAN_TOPIC, dict(WP13_PLAN))
            self._plan_sent = True
            self.s._event(drv.t, "platform_plan", action="hold", prior_plan=self._prior_plan is not None)
        elif not active and self._plan_sent:
            full = self.s.charger.state == "full"
            latch = None
            if p is not None and p.reserve.action == ROUTE_RETURN_AND_WAIT:
                latch = (p.reserve.action, p.reserve.reason, p.reserve.latched_at)
            if self._prior_plan is not None:
                drv.deliver(PLATFORM_PLAN_TOPIC, dict(self._prior_plan))   # restores it; WP7 keeps the latch
                if full and p is not None:
                    p.reserve.clear()
                action = "restored"
            else:
                drv.deliver(PLATFORM_PLAN_TOPIC, {"clear": True})          # WP7 clears the latch with the plan
                if not full and latch is not None:
                    self._restore_latch = latch
                action = "cleared"
            self.s._event(drv.t, "platform_plan", action=action, charged_full=full,
                          reserve_latch_kept=bool(latch is not None and not full))
            self._plan_sent = False
            self._prior_plan = None

    def _publish(self, drv, t):
        st = self.s.status()
        st["platform_plan_held"] = self._plan_sent
        drv.bus.values["dock"] = st
        changed = self.s.step != self._last_step
        self._last_step = self.s.step
        if changed or t + 1e-9 >= self._next_pub:
            drv.bus.publish(STATE_TOPIC, st)
            self._next_pub = t + STATE_PERIOD_S
        while self._ev_sent < len(self.s.events):
            drv.bus.publish(EVENTS_TOPIC, self.s.events[self._ev_sent])
            self._ev_sent += 1


def platform_runtime(m):
    """The WP7 platform hook's per-model runtime (None when the hook is not loaded). Private state: change request to
    WP7 for a public accessor (the WP13 contract notes)."""
    try:
        from atharv_nandi.sim.hooks import platform as hp
    except Exception:  # pragma: no cover - hooks always importable in the sim tree
        return None
    return getattr(hp, "_STATE", {}).get(id(m))


def platform_battery(m):
    """The WP7 platform hook's Battery for model ``m`` (None when the hook is not loaded). Reads the hook's
    private per-model state: change request to WP7 for a public accessor (the WP13 contract notes)."""
    p = platform_runtime(m)
    return None if p is None else p.battery


# ---------------------------------------------------------------------- acceptance sweep (re-runnable)
def sweep(world: str = "orchard_demo", n: int = 10, seed: int = 1, r_min: float = 1.0, r_max: float = 3.0,
          bearing_deg: float = 69.0, limit_s: float = 150.0) -> List[dict]:
    """The contract's dock sweep: ``n`` random starts ``r_min``-``r_max`` m from the plate on the entry side (bearing
    within ±``bearing_deg`` of the entry line), one model for the world, a fresh SimDriver per start, real gait.
    One row per start: ``{start, status (docked / assist / skipped), reason, retries, dist_cm, yaw_deg, sim_s,
    wall_s, strafe_min_m}`` with the truth-pose error. ``python3 -m atharv_nandi.core.dock_logic --sweep N --world W
    --seed S`` prints the rows and a summary (keep N <= 15 per capped-runner call)."""
    import random
    import time as _time
    os.environ["NANDI_WORLD"] = world
    from atharv_nandi.core import farm_scene
    from atharv_nandi.core.sim_driver import SimDriver
    m = farm_scene.load_farm_model()
    g = DockGeometry.from_model(m)
    if g is None:
        raise ValueError(f"world {world!r} has no dock")
    rng = random.Random(seed)
    ex, ey = g.entry_dir
    back = math.atan2(-ey, -ex)              # from the plate towards the entry side
    rows = []
    for _ in range(n):
        r, b = rng.uniform(r_min, r_max), back + math.radians(rng.uniform(-bearing_deg, bearing_deg))
        start = (round(g.x + r * math.cos(b), 3), round(g.y + r * math.sin(b), 3))
        w0 = _time.perf_counter()
        drv = SimDriver(m)
        drv.stand(0.3)
        if not drv.go_to(start):
            rows.append({"start": start, "status": "skipped", "reason": "go_to_failed"})
            continue
        drv.stand(0.3)
        s = DockUnloadSession(g, battery=None, charge=False)
        DockRunner(drv, s)
        t0 = drv.t
        s.start(t0, crown_stowed(drv), xy=(float(drv.d.qpos[0]), float(drv.d.qpos[1])))
        drv.run_until(lambda _: s.step in ("done", "assist"), limit_s, chunk_s=0.1)
        e = g.error(float(drv.d.qpos[0]), float(drv.d.qpos[1]), drv.heading())
        a = s.dock_aligner
        rows.append({"start": start, "status": "docked" if s.step == "done" else s.step, "reason": s.reason,
                     "retries": a.retries, "dist_cm": round(e.dist_m * 100, 2), "yaw_deg": round(abs(e.yaw_deg), 2),
                     "sim_s": round(drv.t - t0, 1), "wall_s": round(_time.perf_counter() - w0, 1),
                     "strafe_min_m": a.strafe_min_remaining_m})
    return rows


if __name__ == "__main__":  # pragma: no cover - CLI around sweep()
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(description="WP13 dock acceptance sweep (real gait)")
    ap.add_argument("--sweep", type=int, default=10, help="number of random starts")
    ap.add_argument("--world", default="orchard_demo")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    res = sweep(args.world, args.sweep, args.seed)
    for row in res:
        print(_json.dumps(row))
    ok = [x for x in res if x["status"] == "docked"]
    run = [x for x in res if x["status"] != "skipped"]
    print(_json.dumps({"world": args.world, "seed": args.seed, "starts": len(res), "skipped": len(res) - len(run),
                       "docked": len(ok), "first_try": sum(1 for x in ok if x["retries"] == 0),
                       "dist_cm_max": max((x["dist_cm"] for x in ok), default=None),
                       "yaw_deg_max": max((x["yaw_deg"] for x in ok), default=None)}))
