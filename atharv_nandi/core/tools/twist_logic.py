"""Twist release for guava and papaya (T4 TWIST; WP2 runtime owner of the ORCHARD-MVP-SPEC 3.4 rule).
Pure Python, no MuJoCo, no ROS.

Release statistic (OS 3.4): the weld's rotational reaction on the fruit projected onto the **stem / pedicel axis**,
minus the rest baseline captured at assignment, low-passed over 40 ms, above ``tau_detach`` for 3 consecutive
ticks, AND an accumulated relative roll of >= 30 deg. The fruit is welded at its CoM with torquescale 1, so
gravity gives a force and no moment and the baseline is ~0 for a hanging fruit.

Axis and roll as built (the WP23 contract leaves both to WP2):

    pedicel axis  unit(anchor - stem_end) of the pool slot: the pedicel line the stalk breaks along. For papaya
                  the profile ``stem_axis`` is the fruit's long axis (z), but the peduncle runs radially into the
                  trunk, the line the WP0c T4 cradle approach (tool axis radial, horizontal) rolls about; for guava
                  it is the hanging pedicel
    roll          the T4 ``twist_roll`` joint travel since the jaws closed on the fruit (``arm_roll``). A stiff
                  weld lets the fruit itself turn < 1 deg (WP23 sweep: 0.1 deg papaya at 1.2 N.m), so the
                  fruit's own rotation cannot reach 30 deg; the pads slip once the stalk holds, and the tool roll
                  is the travel the real twist needs (fidelity register ``logical_twist_release``)

So a hanging papaya never releases (no roll); full jaw closure without roll never releases (roll gate); a
twist whose stalk torque stays at 0.8 tau never releases (torque gate); above tau with >= 30 deg roll it does.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

LPF_S = 0.040
TICKS = 3
MIN_ROLL_DEG = 30.0


def pedicel_axis(stem_end: Sequence[float], anchor: Sequence[float]) -> np.ndarray:
    v = np.asarray(anchor, float) - np.asarray(stem_end, float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("pedicel has zero length (stem end on the anchor)")
    return v / n


def axis_torque(tau_world: Sequence[float], axis: Sequence[float]) -> float:
    return float(np.dot(np.asarray(tau_world, float), np.asarray(axis, float)))


@dataclass
class TwistRelease:
    tau_detach: float
    dt: float                                   # control period (s)
    lpf_s: float = LPF_S
    ticks: int = TICKS
    min_roll_deg: float = MIN_ROLL_DEG
    baseline: Optional[float] = None            # stem torque at rest, captured at assignment
    roll0: Optional[float] = None               # tool roll (rad) when the jaws closed
    filtered: float = 0.0
    above: int = 0
    peak: float = 0.0
    roll_deg: float = 0.0
    released: bool = False
    log: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if not (self.tau_detach > 0 and math.isfinite(self.tau_detach)):
            raise ValueError(f"tau_detach must be > 0 (got {self.tau_detach})")
        if not self.dt > 0:
            raise ValueError("dt must be > 0")

    def capture_baseline(self, stem_torque: float):
        """At assignment (fruit hanging untouched)."""
        self.baseline = float(stem_torque)

    def arm_roll(self, roll_rad: float):
        """Jaws closed on the fruit: the roll is counted from here."""
        self.roll0 = float(roll_rad)

    def update(self, stem_torque: float, roll_rad: Optional[float]) -> bool:
        """One control tick; True once (and from then on) when the stalk lets go."""
        if self.released:
            return True
        if self.baseline is None:
            self.baseline = float(stem_torque)
        x = abs(float(stem_torque) - self.baseline)
        a = self.dt / (self.lpf_s + self.dt)
        self.filtered += a * (x - self.filtered)
        self.peak = max(self.peak, self.filtered)
        self.above = self.above + 1 if self.filtered > self.tau_detach else 0
        if roll_rad is None or self.roll0 is None:
            self.roll_deg = 0.0
        else:
            self.roll_deg = abs(math.degrees(float(roll_rad) - self.roll0))
        if self.above >= self.ticks and self.roll_deg >= self.min_roll_deg:
            self.released = True
        return self.released
