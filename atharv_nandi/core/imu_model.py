"""IMU noise layer for Atharv Nandi (WP7; pure Python, no ROS, no MuJoCo).

MVP-SPEC 2.5: the ``imu`` site carries ``framequat`` / ``gyro`` / ``accelerometer`` sensors (imu_quat, imu_gyro,
imu_acc); this module degrades them with the hal-sim-to-real 4 layer before publishing ``/nandi/imu``
(sensor_msgs/Imu-compatible JSON):

    gyro    white noise 0.01 rad/s (1 sigma, ASSUMED) + bias random walk 1e-4 rad/s/sqrt(s)  (slow bias, ASSUMED)
    accel   white noise 0.05 m/s^2 + bias random walk 1e-3 m/s^2/sqrt(s)                    (ASSUMED)
    orient  small-angle noise 0.2 deg per axis on the reported quaternion                     (ASSUMED)

The layer is ON by default in the platform hook (``NANDI_IMU_NOISE=0`` publishes ideal values); every message says
which (``noise: true/false``) so a KPI run records whether the layer was on (hal-sim-to-real 4 rule).
"""
import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

GRAVITY = 9.81


@dataclass(frozen=True)
class ImuNoise:
    gyro_sigma: float = 0.01          # rad/s
    gyro_bias_rw: float = 1e-4        # rad/s/sqrt(s)
    acc_sigma: float = 0.05           # m/s^2
    acc_bias_rw: float = 1e-3         # m/s^2/sqrt(s)
    orient_sigma_deg: float = 0.2

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"ImuNoise.{k} must be finite and >= 0")


IDEAL = ImuNoise(0.0, 0.0, 0.0, 0.0, 0.0)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _small_rot(rx, ry, rz):
    ang = math.sqrt(rx * rx + ry * ry + rz * rz)
    if ang < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    s = math.sin(ang / 2) / ang
    return (math.cos(ang / 2), rx * s, ry * s, rz * s)


def quat_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


class ImuModel:
    """Stateful noise layer. ``sample(t, quat_wxyz, gyro, acc)`` -> sensor_msgs/Imu-compatible dict."""

    def __init__(self, noise: ImuNoise = ImuNoise(), seed: int = 7, enabled: bool = True):
        self.noise = noise if enabled else IDEAL
        self.enabled = bool(enabled)
        self._seed = int(seed)
        self.reset()

    def reset(self):
        self.rng = random.Random(self._seed + 101)
        self.gyro_bias = [0.0, 0.0, 0.0]
        self.acc_bias = [0.0, 0.0, 0.0]
        self._last_t: Optional[float] = None

    def _walk_bias(self, dt: float):
        if dt <= 0:
            return
        g, a = self.noise.gyro_bias_rw * math.sqrt(dt), self.noise.acc_bias_rw * math.sqrt(dt)
        for i in range(3):
            self.gyro_bias[i] += self.rng.gauss(0.0, g) if g else 0.0
            self.acc_bias[i] += self.rng.gauss(0.0, a) if a else 0.0

    def sample(self, t: float, quat_wxyz: Sequence[float], gyro: Sequence[float], acc: Sequence[float]) -> dict:
        t = float(t)
        if self._last_t is not None and t > self._last_t:
            self._walk_bias(min(t - self._last_t, 5.0))
        if self._last_t is None or t > self._last_t:
            self._last_t = t
        n = self.noise
        gs, as_ = n.gyro_sigma, n.acc_sigma
        w = [float(gyro[i]) + self.gyro_bias[i] + (self.rng.gauss(0.0, gs) if gs else 0.0) for i in range(3)]
        a = [float(acc[i]) + self.acc_bias[i] + (self.rng.gauss(0.0, as_) if as_ else 0.0) for i in range(3)]
        q = tuple(float(v) for v in quat_wxyz)
        if n.orient_sigma_deg:
            s = math.radians(n.orient_sigma_deg)
            q = _quat_mul(q, _small_rot(self.rng.gauss(0, s), self.rng.gauss(0, s), self.rng.gauss(0, s)))
            nq = math.sqrt(sum(v * v for v in q))
            q = tuple(v / nq for v in q)
        so, sg, sa = math.radians(n.orient_sigma_deg) ** 2, gs * gs, as_ * as_
        return {
            "header": {"stamp": round(t, 6), "frame_id": "imu"},
            # sensor_msgs/Imu order is x, y, z, w
            "orientation": {"x": q[1], "y": q[2], "z": q[3], "w": q[0]},
            "orientation_covariance": [so, 0, 0, 0, so, 0, 0, 0, so],
            "angular_velocity": {"x": w[0], "y": w[1], "z": w[2]},
            "angular_velocity_covariance": [sg, 0, 0, 0, sg, 0, 0, 0, sg],
            "linear_acceleration": {"x": a[0], "y": a[1], "z": a[2]},
            "linear_acceleration_covariance": [sa, 0, 0, 0, sa, 0, 0, 0, sa],
            "noise": self.enabled,
            "assumed": True,
        }


def validate_imu(msg: dict) -> list:
    errs = []
    for k in ("header", "orientation", "orientation_covariance", "angular_velocity", "angular_velocity_covariance",
              "linear_acceleration", "linear_acceleration_covariance", "noise"):
        if k not in msg:
            errs.append(f"missing {k}")
    if errs:
        return errs
    q = msg["orientation"]
    nq = math.sqrt(sum(q[k] ** 2 for k in "xyzw"))
    if abs(nq - 1.0) > 1e-6:
        errs.append("orientation not unit")
    for k in ("orientation_covariance", "angular_velocity_covariance", "linear_acceleration_covariance"):
        if len(msg[k]) != 9:
            errs.append(f"{k} needs 9 entries")
    return errs
