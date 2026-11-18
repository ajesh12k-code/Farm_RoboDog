"""18-DOF alternating tripod gait (walk, strafe, turn) with trunk-levelling balance feedback.

Pure Python (no ROS, no MuJoCo). Output order matches ctrl[7:25] / qpos[14:32]:
    legs FR, FL, MR, ML, RR, RL  x  (hip, thigh, calf)

Sign conventions, derived from models/AtharvNandi/atharv_nandi.xml
(verified against MuJoCo forward kinematics in tests/test_tripod_gait.py):

  Leg layout (trunk frame, x fwd, y left, z up): hips at x = +0.26 (F), 0 (M), -0.26 (R);
  y = -0.14/-0.15 (right) and +0.14/+0.15 (left). Each thigh link starts 0.065 m further out.

  Thigh/calf joints: hinge axis +y, links of L = 0.23 m hanging along -z. A rotation of +t about
  +y maps (0, 0, -L) to (-L sin t, 0, -L cos t), so a POSITIVE thigh angle swings the foot BACKWARD
  (-x). With thigh = t and calf = c the foot sits at distance R = 2 L cos(c/2) from the thigh
  joint, in direction alpha = t + c/2 from straight down (foot x = -R sin(alpha)). Home
  (t, c) = (0.80, -1.60) gives alpha = 0 (foot under the hip) and R = 0.3205 m.
    * stride: change thigh only -> alpha changes, R fixed. foot x = dx  <=>  thigh -= asin(dx / R).
    * extension: dR/dc = -L sin(c/2) = +0.165 m/rad at home, so raising calf by dc (towards 0)
      lengthens the leg; thigh -= dc / 2 keeps alpha (the foot stays under the hip).
    * lift: calf more negative shortens the leg (foot rises ~7 cm at full lift).

  Hip joints: hinge axis +x for every leg. Rotation by +p about +x maps (y, z) to
  (y cos p - z sin p, y sin p + z cos p); with the foot at z ~= -0.32 the foot moves +y by
  ~0.31 m/rad on BOTH sides, so a POSITIVE hip angle moves the foot LEFT (+y). Home splays
  the feet outward: hip = -0.12 (right legs), +0.12 (left legs).

  Body motion from stance feet (feet fixed on the ground, body frame): a stance foot moving
  -x drives the body +x; moving -y drives the body +y. For a CCW yaw rate w (turn left, +z) a
  ground point at body position (px, py) moves at (w py, -w px) relative to the body, i.e.
    * left legs (py > 0) push their feet FORWARD and right legs push BACKWARD
      (differential stride: left legs walk "backward", right legs "forward");
    * front legs (px > 0) push their feet RIGHT (-y), rear legs LEFT (+y) (tangential term).
  So per leg i: stride_x = vx * Sx - yaw * Sw * py_i / Rref and stride_y = vy * Sy + yaw * Sw * px_i / Rref.

  Balance (levelling): for small tilts the world height of body point (px, py) changes by
  dz = -sin(pitch) px + cos(pitch) sin(roll) py (third row of R). pitch > 0 is NOSE-DOWN, so the
  front hips are low -> front legs EXTEND, rear legs retract. roll > 0 lifts the LEFT side -> left
  legs retract, right legs extend. Extension = -dz * BALANCE_FRACTION * balance_gain, clamped.
"""
import math
from typing import List, Optional, Sequence, Tuple

from atharv_nandi.core.locomotion_logic import CMD_EPS, GaitParams, clamp

LEG_NAMES = ("FR", "FL", "MR", "ML", "RR", "RL")
SIDE = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # +1 = left (+y)
TRIPOD_PHASE = (0.0, math.pi, math.pi, 0.0, 0.0, math.pi)  # tripod A = FR, ML, RR; B = FL, MR, RL

LINK_L = 0.23
HOME_HIP_ABS, HOME_THIGH, HOME_CALF = 0.12, 0.80, -1.60
LEG_REACH = 2.0 * LINK_L * math.cos(HOME_CALF / 2.0)          # 0.3205 m thigh joint -> foot centre
EXT_PER_CALF_RAD = -LINK_L * math.sin(HOME_CALF / 2.0)        # 0.165 m of leg length per rad of calf
_THIGH_OFFSET_Y = 0.065
LATERAL_ARM = (-_THIGH_OFFSET_Y * math.sin(HOME_HIP_ABS)       # 0.310 m of foot y per rad of hip
               + LEG_REACH * math.cos(HOME_HIP_ABS))

# Nominal foot positions in the trunk frame (hip position + thigh offset + home splay).
HIP_X = (0.26, 0.26, 0.0, 0.0, -0.26, -0.26)
_HIP_Y_ABS = (0.14, 0.14, 0.15, 0.15, 0.14, 0.14)
_SPLAY_Y = _THIGH_OFFSET_Y * math.cos(HOME_HIP_ABS) + LEG_REACH * math.sin(HOME_HIP_ABS)
FOOT_Y = tuple(s * (h + _SPLAY_Y) for s, h in zip(SIDE, _HIP_Y_ABS))  # ~ +-0.243 / +-0.253 m
TURN_REF_RADIUS = 0.26

# Stride amplitudes (foot travel = 2x these per stance) per unit command, before speed_scale.
# STRIDE_X_PER_CMD 0.045 (was 0.09 = the original 0.28 rad thigh swing). With the kp 300 legs the 0.09 stride
# made the trunk pitch-bounce: at vx 1, 0 kg, flat the robot spent 70% of the walk on fewer than 3 feet
# (pre-alignment 20%) and the rear stance feet bounced 13 mm clear in early stance (review F3). MEASURED
# (tools/physics/measure_baseline.walk, vx 1, 6 s, flat / ridges / rough seeds 1-6, 0 / 10 kg, 2026-11-17):
#   stride 0.050: < 3 feet 0.13-0.33 (rough seed 2, 0 kg over the 0.30 gate)
#   stride 0.045: < 3 feet 0.08-0.27, speed 0.22-0.27 m/s (pre-alignment 0.21-0.31), foot pen <= 9.0 mm
#   stride 0.040: < 3 feet 0.04-0.26 but flat 0.24 m/s, under 0.8 x the pre-alignment 0.31 m/s
# Rough-clod tilt transients of 10-14 deg at 0 kg occur at every stride tried (one or two seeds of six).
STRIDE_X_PER_CMD = 0.045  # m
STRIDE_Y_PER_CMD = 0.068  # m  (= the original 0.22 rad hip swing)
# TURN_STRIDE_PER_CMD 0.035 (was 0.06): MEASURED in-place turn yaw 1, 6 s, flat: 0.06 gave 97 dps with 7 calf-floor
# ticks and 64% of the time on < 3 feet; 0.035 gives 59 dps at 0 kg / -51 dps at 10 kg (pre-alignment 60 dps),
# 0 calf-floor ticks, foot penetration <= 7.7 mm, < 3 feet 42-53% (pre-alignment 23%, still open).
TURN_STRIDE_PER_CMD = 0.035  # m of tangential foot travel at TURN_REF_RADIUS for yaw = 1
# Front and middle legs on one side are in opposite tripods and meet at the stride extremes:
# their feet are 0.26 m apart at home, 2 x 0.095 + 2 x 0.03 (foot radius) = 0.25 m.
MAX_STRIDE_X = 0.095
MAX_STRIDE_Y = 0.07
# Calf flex at mid-swing for step_height_scale = 1. 0.44 (was 0.42) after the kp 300 retune: MEASURED sweep
# 0.34-0.50 (tools/physics/measure_baseline.walk, vx 1, 6 s, 0 / 10 kg): 0.44 keeps the walking foot penetration
# <= 9.1 mm on flat / ridges / rough (0.42: 10.1 mm on ridges) with no calf-to-floor contact on flat; its peak
# target rate 0.44 * 2 pi * 1.8 Hz = 4.98 rad/s stays under MAX_JOINT_SPEED_RPS (DERIVED).
LIFT_CALF_RAD = 0.44

BALANCE_FRACTION = 0.5    # fraction of the measured hip height error corrected at balance_gain = 1
MAX_BALANCE_EXT = 0.04    # m, per-leg clamp on the levelling extension

RAMP_S = 0.15             # start/stop envelope and command smoothing time constant
# Leg joint speed cap (CITED AK80-64 rated speed ~4.2-5.0 rad/s; store.tmotor.com goods-1202, 2026-11-17).
# MuJoCo position actuators cannot cap velocity, so every output target moves at most
# MAX_JOINT_SPEED_RPS * dt per tick from the previous one (0.10 rad per 50 Hz tick). This applies to stop()
# too: the legs slew back to home at the cap instead of jumping.
MAX_JOINT_SPEED_RPS = 5.0
JOINT_LIMITS = ((-0.7, 0.7), (-1.5, 2.5), (-2.6, -0.5))  # hip, thigh, calf (MJCF ranges)

TWO_PI = 2.0 * math.pi


def leg_extension_to_joints(ext_m: float) -> Tuple[float, float]:
    """(d_thigh, d_calf) that lengthens a home-pose leg by ext_m along its axis (foot stays under hip)."""
    d_calf = ext_m / EXT_PER_CALF_RAD
    return -0.5 * d_calf, d_calf


def balance_extensions(roll: float, pitch: float, gain: float) -> List[float]:
    """Per-leg extension (m, + = longer leg) that levels the trunk; see module docstring for signs."""
    sp, cp_sr = math.sin(pitch), math.cos(pitch) * math.sin(roll)
    k = BALANCE_FRACTION * gain
    return [clamp(k * (sp * HIP_X[i] - cp_sr * FOOT_Y[i]), -MAX_BALANCE_EXT, MAX_BALANCE_EXT) for i in range(6)]


def balance_corrections(roll: float, pitch: float, gain: float) -> List[float]:
    """18 joint deltas (hip, thigh, calf per leg) implementing balance_extensions."""
    out = []
    for ext in balance_extensions(roll, pitch, gain):
        d_thigh, d_calf = leg_extension_to_joints(ext)
        out.extend([0.0, d_thigh, d_calf])
    return out


def leg_strides(vx: float, vy: float, yaw: float, speed_scale: float = 1.0) -> List[Tuple[float, float]]:
    """Per-leg (stride_x, stride_y) half-amplitudes in metres: + means the body is pushed +x / +y."""
    out = []
    for i in range(6):
        sx = speed_scale * (STRIDE_X_PER_CMD * vx - TURN_STRIDE_PER_CMD * yaw * FOOT_Y[i] / TURN_REF_RADIUS)
        sy = speed_scale * (STRIDE_Y_PER_CMD * vy + TURN_STRIDE_PER_CMD * yaw * HIP_X[i] / TURN_REF_RADIUS)
        out.append((clamp(sx, -MAX_STRIDE_X, MAX_STRIDE_X), clamp(sy, -MAX_STRIDE_Y, MAX_STRIDE_Y)))
    return out


def home_leg_targets() -> List[float]:
    out = []
    for s in SIDE:
        out.extend([s * HOME_HIP_ABS, HOME_THIGH, HOME_CALF])
    return out


class TripodGaitGenerator:
    """Alternating tripod gait.

    Phase convention per leg: phase = gait_phase + TRIPOD_PHASE[i]; swing = sin(phase),
    lift = max(0, cos(phase)). The foot is in the air for cos > 0 (phase -pi/2 .. pi/2) while
    swing runs -1 -> +1 (foot carried back -> front); in stance swing runs +1 -> -1 and the
    planted foot sweeps front -> back, pushing the body along +stride.
    """

    def __init__(self, dt: float = 0.02, params: Optional[GaitParams] = None):
        self.dt = dt
        self.params = params or GaitParams()
        self.cmd = (0.0, 0.0, 0.0)   # target (vx, vy, yaw)
        self.vel = (0.0, 0.0, 0.0)   # smoothed command actually driving the strides
        self.is_held = False
        self.remaining_phase = 0.0   # single-step budget (rad of gait phase)
        self.gait_phase = 0.0        # [0, 2pi) since the walk started
        self.amp = 0.0               # start/stop envelope 0..1
        self.last_targets = None     # previous output (rate limit reference); None = no limit on the first step

    # --- commands ---------------------------------------------------------------------
    def set_params(self, params: GaitParams):
        self.params = params

    def set_command(self, vx: float, vy: float, yaw: float = 0.0, hold: bool = False):
        """hold=True: walk until release(); hold=False: one gait cycle. All-zero => release()."""
        if abs(vx) < CMD_EPS and abs(vy) < CMD_EPS and abs(yaw) < CMD_EPS:
            self.release()
            return
        self.cmd = (float(vx), float(vy), float(yaw))
        if hold:
            self.is_held = True
            self.remaining_phase = 0.0
        elif self.is_held:
            self.is_held = False
            self.remaining_phase = self._phase_to_cycle_end()
        elif self.remaining_phase <= 0.05:
            self.remaining_phase = TWO_PI

    def release(self):
        """Finish the current gait cycle, then stop (a running single step just completes)."""
        if self.is_held:
            self.is_held = False
            self.remaining_phase = self._phase_to_cycle_end()

    def stop(self):
        """Stop now: back to the standing pose on the next step()."""
        self.is_held = False
        self.remaining_phase = 0.0
        self.cmd = self.vel = (0.0, 0.0, 0.0)
        self.amp = 0.0
        self.gait_phase = 0.0

    def _phase_to_cycle_end(self) -> float:
        return (TWO_PI - self.gait_phase) % TWO_PI

    @property
    def is_active(self) -> bool:
        """A walk is requested (held, or a single-step cycle still has phase left)."""
        return self.is_held or self.remaining_phase > 0.0

    @property
    def is_walking(self) -> bool:
        """Legs are moving (requested, or still ramping down)."""
        return self.is_active or self.amp > 0.0

    # --- generator --------------------------------------------------------------------
    def _advance(self):
        active = self.is_active
        ramp = self.dt / RAMP_S
        self.amp = min(1.0, self.amp + ramp) if active else max(0.0, self.amp - ramp)
        if not (active or self.amp > 0.0):
            self.gait_phase = 0.0
            self.cmd = self.vel = (0.0, 0.0, 0.0)
            return
        a = 1.0 - math.exp(-self.dt / RAMP_S)
        self.vel = tuple(v + a * (c - v) for v, c in zip(self.vel, self.cmd))
        d_phase = TWO_PI * self.params.freq_hz * self.dt
        self.gait_phase = (self.gait_phase + d_phase) % TWO_PI
        if not self.is_held and self.remaining_phase > 0.0:
            self.remaining_phase = max(0.0, self.remaining_phase - d_phase)

    def step(self, roll: float = 0.0, pitch: float = 0.0) -> List[float]:
        """Advance one dt and return 18 leg joint targets; roll/pitch (rad) feed the levelling."""
        self._advance()
        p = self.params
        targets = home_leg_targets()
        if self.amp > 0.0:
            strides = leg_strides(*self.vel, speed_scale=p.speed_scale)
            lift_amp = LIFT_CALF_RAD * p.step_height_scale * self.amp
            for i, (sx, sy) in enumerate(strides):
                phase = self.gait_phase + TRIPOD_PHASE[i]
                swing, lift = math.sin(phase), max(0.0, math.cos(phase))
                targets[3 * i] += self.amp * sy * swing / LATERAL_ARM
                targets[3 * i + 1] -= math.asin(clamp(self.amp * sx * swing / LEG_REACH, -1.0, 1.0))
                targets[3 * i + 2] -= lift_amp * lift
        if p.balance_gain > 0.0:
            targets = [t + c for t, c in zip(targets, balance_corrections(roll, pitch, p.balance_gain))]
        out = clamp_leg_targets(targets)
        if self.last_targets is not None:
            out = rate_limit(self.last_targets, out, MAX_JOINT_SPEED_RPS * self.dt)
        self.last_targets = out
        return out


def clamp_leg_targets(targets: Sequence[float]) -> List[float]:
    return [clamp(t, *JOINT_LIMITS[j % 3]) for j, t in enumerate(targets)]


def rate_limit(prev: Sequence[float], targets: Sequence[float], max_step: float) -> List[float]:
    """Move each target at most max_step (rad) away from prev."""
    return [p + clamp(t - p, -max_step, max_step) for p, t in zip(prev, targets)]
