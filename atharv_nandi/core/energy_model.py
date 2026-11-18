"""Pure battery / power model for Atharv Nandi (no ROS, no MuJoCo).

Power draw (electrical, W):
    P = sum_j |tau_j * qdot_j| / eta + P_idle
where tau_j * qdot_j is the mechanical power of joint j (from /nandi/joint_states effort x velocity).
Every constant below is an ASSUMPTION for the simulator, not a measured value of real hardware.

State of charge is integrated from P with an injected dt (sim seconds). ``update`` and ``consume`` never raise it;
only ``reset`` and ``charge_full`` (docked at home on the ASSUMED charger, no charge-rate model) do.

When no joint_states arrive (the topic is published by the safety package and may be absent), the
model drains at P_idle only and the caller flags the estimate as 'idle_only'.

WP7 (MVP-SPEC 2.5, ORCHARD-MVP-SPEC 1.3): the pack is a 24 V class LFP pack (8S, 25.6 V nominal, ASSUMED) with a
flat open-circuit-voltage curve and a series resistance, so ``Battery.status()`` is a superset of
``sensor_msgs/BatteryState`` (voltage, current, charge, capacity, percentage, power_supply_* enums) plus
``projected_end_soc`` and ``reserve`` once a route projection is set (``set_projection``; see route_energy.py).
``Battery.consume`` drains an exactly integrated energy (the platform hook integrates sum |tau qdot| every
physics step), so the battery and the integrated-energy reference of the estimator test are the same number.
"""
import json
import math
from dataclasses import dataclass
from typing import Optional, Sequence

# ASSUMED: not measured on hardware. Configurable through EnergyParams (config/missions/site_sim.yaml).
ASSUMED_CAPACITY_WH = 500.0   # ASSUMED usable pack energy (500 Wh Li-ion, 3.2 kg at ~155 Wh/kg; MJCF battery geom)
ASSUMED_ETA = 0.6             # ASSUMED motor + driver + gearbox efficiency (mechanical / electrical)
ASSUMED_P_IDLE_W = 15.0       # ASSUMED compute + sensors + servo holding baseline
# DERIVED from MEASURED walking mechanical power (tools/physics/measure_baseline.walk mech_power_w = sum |tau qdot|,
# vx 1, 6 s, AT3 walks, 2026-11-17): 83-108 W at 0 kg, 101-121 W at 10 kg (flat / ridges / rough seeds 1-6).
# Worst case 121 / 0.6 + 15 = 217 W, rounded up to 220 W (was 60 W, about 4x low; review F2). Route estimates only;
# the live SoC integrates /nandi/joint_states.
ASSUMED_WALK_POWER_W = 220.0
ASSUMED_PICK_ENERGY_WH = 0.25  # ASSUMED energy per pick-and-place attempt (route estimates only)
# ASSUMED average incl. turns and slow-down at waypoints (route estimates only). MEASURED straight walking: 0.22-0.27
# m/s at vx 1 (flat / ridges / rough, 0-10 kg, 2026-11-17); lower near waypoints (nav VX_MIN 0.35).
ASSUMED_WALK_SPEED_MPS = 0.12

# ASSUMED pack electrics (MVP-SPEC 2.5: 24 V LFP, YAML). 8S LiFePO4, 3.2 V nominal per cell.
ASSUMED_CELLS_S = 8
ASSUMED_NOMINAL_CELL_V = 3.2  # LiFePO4 nominal cell voltage (8S -> 25.6 V)
ASSUMED_R_INT_OHM = 0.05      # ASSUMED pack series resistance (cells + BMS + wiring)
# LFP open-circuit voltage per cell vs SoC (typical LiFePO4 curve, ASSUMED; flat 3.25-3.35 V plateau).
LFP_OCV_TABLE = ((0.0, 2.50), (0.05, 3.00), (0.10, 3.20), (0.20, 3.25), (0.40, 3.28), (0.60, 3.30),
                 (0.80, 3.33), (0.90, 3.35), (0.97, 3.40), (1.0, 3.45))
RESERVE_PACK_FRACTION = 0.15  # MVP-SPEC 2.5 / MP 0.2: reserve = SoC to return to the headland + 15 % of the pack

# sensor_msgs/BatteryState enums (the subset used here).
POWER_SUPPLY_STATUS_CHARGING = 1
POWER_SUPPLY_STATUS_DISCHARGING = 2
POWER_SUPPLY_STATUS_FULL = 4
POWER_SUPPLY_HEALTH_GOOD = 1
POWER_SUPPLY_TECHNOLOGY_LIFE = 4
BATTERY_STATE_FIELDS = ("voltage", "temperature", "current", "charge", "capacity", "design_capacity", "percentage",
                        "power_supply_status", "power_supply_health", "power_supply_technology", "present",
                        "cell_voltage", "cell_temperature", "location", "serial_number")

LOW_SOC = 0.25                # below this a running mission returns home
RESERVE_FRACTION = 0.20       # a mission needs route estimate + this much SoC to be accepted
POWER_EMA_TAU_S = 10.0        # smoothing for est_remaining_min


@dataclass(frozen=True)
class EnergyParams:
    capacity_wh: float = ASSUMED_CAPACITY_WH
    eta: float = ASSUMED_ETA
    p_idle_w: float = ASSUMED_P_IDLE_W
    walk_power_w: float = ASSUMED_WALK_POWER_W
    pick_energy_wh: float = ASSUMED_PICK_ENERGY_WH
    walk_speed_mps: float = ASSUMED_WALK_SPEED_MPS
    low_soc: float = LOW_SOC
    reserve: float = RESERVE_FRACTION
    cells_s: float = ASSUMED_CELLS_S
    r_int_ohm: float = ASSUMED_R_INT_OHM
    reserve_pack: float = RESERVE_PACK_FRACTION

    def __post_init__(self):
        if not (self.capacity_wh > 0 and 0 < self.eta <= 1 and self.p_idle_w >= 0 and self.walk_speed_mps > 0
                and self.cells_s >= 1 and self.r_int_ohm >= 0 and 0 <= self.reserve_pack < 1):
            raise ValueError("invalid energy params")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "EnergyParams":
        d = d or {}
        kw = {}
        for k in cls.__dataclass_fields__:
            if k in d:
                kw[k] = float(d[k])
        return cls(**kw)


def mechanical_power_w(effort: Sequence[float], velocity: Sequence[float]) -> float:
    """sum |tau * qdot| over joints present in both arrays; non-finite entries count as 0."""
    total = 0.0
    for tau, qd in zip(effort, velocity):
        p = float(tau) * float(qd)
        if math.isfinite(p):
            total += abs(p)
    return total


def electrical_power_w(effort: Sequence[float], velocity: Sequence[float], params: EnergyParams) -> float:
    return mechanical_power_w(effort, velocity) / params.eta + params.p_idle_w


class Battery:
    """SoC integrator. Feed power samples with sim-time stamps; time going backwards is ignored."""

    def __init__(self, params: EnergyParams = EnergyParams(), soc0: float = 1.0):
        self.p = params
        self.energy_wh = max(0.0, min(1.0, soc0)) * params.capacity_wh
        self.power_w = params.p_idle_w
        self._ema_w = params.p_idle_w
        self._last_t: Optional[float] = None
        self.source = "idle_only"  # 'joint_states' once a torque sample has been integrated
        self.projected_end_soc: Optional[float] = None  # set_projection (route_energy)
        self.reserve_soc: Optional[float] = None

    @property
    def soc(self) -> float:
        return self.energy_wh / self.p.capacity_wh

    def reset(self, soc0: float = 1.0):
        self.energy_wh = max(0.0, min(1.0, soc0)) * self.p.capacity_wh
        self._last_t = None

    def charge_full(self, t: Optional[float] = None):
        """Docked on the (ASSUMED) charger: the pack is full at once (the same rule as mission_executor's dock
        reset; there is no charge-rate model). The only way SoC rises besides ``reset``."""
        self.energy_wh = self.p.capacity_wh
        self._last_t = t

    def update(self, t: float, power_w: Optional[float] = None) -> float:
        """Integrate the interval since the previous call at the current power (zero-order hold).

        power_w=None means "no joint_states": idle-only drain. Returns the SoC.
        """
        self.source = "idle_only" if power_w is None else "joint_states"
        if power_w is None:
            power_w = self.p.p_idle_w
        power_w = max(0.0, float(power_w)) if math.isfinite(power_w) else self.p.p_idle_w
        if self._last_t is not None and t > self._last_t:
            dt = min(t - self._last_t, 5.0)  # a huge gap (sim paused) is not drained as a burst
            self.energy_wh = max(0.0, self.energy_wh - self.power_w * dt / 3600.0)
            a = 1.0 - math.exp(-dt / POWER_EMA_TAU_S)
            self._ema_w += a * (power_w - self._ema_w)
        if self._last_t is None or t > self._last_t:
            self._last_t = t
        self.power_w = power_w
        return self.soc

    def est_remaining_min(self) -> float:
        p = max(self._ema_w, 1e-6)
        return self.energy_wh / p * 60.0

    def consume(self, t: float, energy_wh: float, power_w: Optional[float] = None) -> float:
        """Drain an exactly integrated energy (Wh, >= 0) for the interval ending at sim time ``t``.

        Used by the platform hook, which integrates sum |tau qdot| / eta + P_idle every physics step. A negative
        or non-finite energy drains nothing (no charging model: SoC never rises). ``power_w`` is the mean power of
        the interval, for the status and the remaining-time EMA. Returns the SoC."""
        e = float(energy_wh)
        if math.isfinite(e) and e > 0.0:
            self.energy_wh = max(0.0, self.energy_wh - e)
        if power_w is not None and math.isfinite(power_w):
            pw = max(0.0, float(power_w))
            if self._last_t is not None and t > self._last_t:
                a = 1.0 - math.exp(-min(t - self._last_t, 5.0) / POWER_EMA_TAU_S)
                self._ema_w += a * (pw - self._ema_w)
            self.power_w = pw
        self.source = "joint_states"
        if self._last_t is None or t > self._last_t:
            self._last_t = t
        return self.soc

    # ------------------------------------------------------------------ pack electrics (ASSUMED)
    def ocv_v(self) -> float:
        return lfp_ocv_cell_v(self.soc) * self.p.cells_s

    def current_a(self) -> float:
        """Pack discharge current (A, > 0 discharging) at the present power: P = (OCV - I R) I, solved for I."""
        return pack_current_a(self.power_w, self.ocv_v(), self.p.r_int_ohm)

    def voltage_v(self) -> float:
        return self.ocv_v() - self.current_a() * self.p.r_int_ohm

    def set_projection(self, projected_end_soc: Optional[float], reserve: Optional[float]):
        """Route projection from route_energy (None clears it): published as projected_end_soc and reserve."""
        self.projected_end_soc = None if projected_end_soc is None else float(projected_end_soc)
        self.reserve_soc = None if reserve is None else float(reserve)

    def status(self) -> dict:
        """Legacy keys (soc, power_w, energy_wh, est_remaining_min, source, assumed) plus the
        sensor_msgs/BatteryState fields (SI units; ``current`` negative while discharging, as in BatteryState)
        and ``projected_end_soc`` / ``reserve`` (None until a projection is set)."""
        cap_ah = self.p.capacity_wh / (ASSUMED_NOMINAL_CELL_V * self.p.cells_s)
        cur = self.current_a()
        v = self.ocv_v() - cur * self.p.r_int_ohm
        full = self.soc >= 0.9995
        proj, res = self.projected_end_soc, self.reserve_soc
        return {
            "soc": round(self.soc, 5),
            "power_w": round(self.power_w, 2),
            "energy_wh": round(self.energy_wh, 4),
            "est_remaining_min": round(self.est_remaining_min(), 1),
            "source": self.source,
            "assumed": True,
            # sensor_msgs/BatteryState-compatible block (MVP-SPEC 5.7 /nandi/battery_status)
            "voltage": round(v, 3),
            "temperature": None,
            "current": round(-cur, 3),
            "charge": round(cap_ah * self.soc, 4),
            "capacity": round(cap_ah, 4),
            "design_capacity": round(cap_ah, 4),
            "percentage": round(self.soc, 5),
            "power_supply_status": POWER_SUPPLY_STATUS_FULL if full else POWER_SUPPLY_STATUS_DISCHARGING,
            "power_supply_health": POWER_SUPPLY_HEALTH_GOOD,
            "power_supply_technology": POWER_SUPPLY_TECHNOLOGY_LIFE,
            "present": True,
            "cell_voltage": [round(v / self.p.cells_s, 4)] * int(self.p.cells_s),
            "cell_temperature": [],
            "location": "trunk",
            "serial_number": "sim",
            "projected_end_soc": None if proj is None else round(proj, 5),
            "reserve": None if res is None else round(res, 5),
        }

    def status_json(self) -> str:
        return json.dumps(self.status())


def route_energy_wh(distance_m: float, picks: int, duration_s: float, params: EnergyParams) -> float:
    """Planning estimate (ASSUMED constants): walking + picks + idle over the mission duration."""
    walk_s = max(0.0, distance_m) / params.walk_speed_mps
    return (params.walk_power_w * walk_s / 3600.0
            + params.pick_energy_wh * max(0, picks)
            + params.p_idle_w * max(0.0, duration_s) / 3600.0)


def route_soc_needed(distance_m: float, picks: int, duration_s: float, params: EnergyParams) -> float:
    """SoC fraction a mission needs: route estimate + reserve. Reject with 'battery_low' if soc < this."""
    return route_energy_wh(distance_m, picks, duration_s, params) / params.capacity_wh + params.reserve


# ------------------------------------------------------------------ pack electrics and BatteryState schema
def lfp_ocv_cell_v(soc: float) -> float:
    """LFP open-circuit cell voltage at ``soc`` (piecewise linear in LFP_OCV_TABLE, ASSUMED curve)."""
    s = max(0.0, min(1.0, float(soc)))
    for (s0, v0), (s1, v1) in zip(LFP_OCV_TABLE, LFP_OCV_TABLE[1:]):
        if s <= s1:
            return v0 + (v1 - v0) * (s - s0) / (s1 - s0)
    return LFP_OCV_TABLE[-1][1]


def pack_current_a(power_w: float, ocv_v: float, r_ohm: float) -> float:
    """Discharge current for a terminal power P with a Thevenin pack: R I^2 - OCV I + P = 0 (smaller root).

    Above the maximum transferable power OCV^2 / 4R the current saturates at OCV / 2R."""
    p = max(0.0, float(power_w))
    if r_ohm <= 0.0:
        return p / max(ocv_v, 1e-9)
    disc = ocv_v * ocv_v - 4.0 * r_ohm * p
    if disc <= 0.0:
        return ocv_v / (2.0 * r_ohm)
    return (ocv_v - math.sqrt(disc)) / (2.0 * r_ohm)


def validate_battery_state(msg: dict) -> list:
    """Schema check of a /nandi/battery_status payload (BatteryState-compatible + projection). [] when valid."""
    errs = []
    if not isinstance(msg, dict):
        return ["not an object"]
    for k in BATTERY_STATE_FIELDS + ("projected_end_soc", "reserve", "soc"):
        if k not in msg:
            errs.append(f"missing {k}")
    if errs:
        return errs
    for k in ("voltage", "current", "charge", "capacity", "design_capacity", "percentage", "soc"):
        v = msg[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            errs.append(f"{k} not a finite number")
    if not errs:
        if not 0.0 <= msg["percentage"] <= 1.0:
            errs.append("percentage outside [0, 1]")
        if msg["current"] > 1e-9:
            errs.append("current > 0 (no charging model; BatteryState: negative while discharging)")
        if msg["charge"] > msg["capacity"] + 1e-6:
            errs.append("charge > capacity")
    for k in ("power_supply_status", "power_supply_health", "power_supply_technology"):
        if not isinstance(msg[k], int) or isinstance(msg[k], bool) or not 0 <= msg[k] <= 8:
            errs.append(f"{k} not a BatteryState enum")
    if not isinstance(msg["present"], bool):
        errs.append("present not bool")
    if not isinstance(msg["cell_voltage"], list):
        errs.append("cell_voltage not a list")
    for k in ("projected_end_soc", "reserve"):
        v = msg[k]
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)):
            errs.append(f"{k} not null or a finite number")
    return errs
