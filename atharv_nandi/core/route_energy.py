"""Route-energy estimator, reserve and endurance numbers for Atharv Nandi (WP7; pure Python, no ROS, no MuJoCo).

MVP-SPEC 2.5 / ORCHARD-MVP-SPEC 1.3 / MISSION-PRESETS 0.2:

    E = Wh_per_m x route_m + Wh_per_item x items + P_idle x t_work + Wh_per_crown_m x crown_m

``t_work`` is the whole mission duration (the idle draw runs all the time), so ``Wh_per_m`` is the walking energy
ABOVE idle per metre and ``Wh_per_crown_m`` the crown-mast energy per metre of mast travel (raise, snake, stow).
The coefficients are measured in this run (``CoefficientTracker``: ridge regression of integrated energy on route
segments, pulled toward the seed) and seeded from the last run card (``Coefficients.from_run_card``), or from the
ASSUMED ``EnergyParams`` when there is no card.

Reserve: the SoC needed to walk back to the headland from where the robot will be + 15 % of the pack.
Behaviour: ``ready_check_energy`` refuses a tour whose projected end SoC is below the reserve at its end (reason
``energy_reserve``); mid-tour, ``ReserveMonitor`` latches ``return_and_wait`` when the projection breaches the
reserve (finish the item, crown stow, return to the headland and wait; WP28 RETURN HOME runs it).

``throughput`` turns the coefficients into the grower-facing numbers of ORCHARD-MVP-SPEC 1.3 (ha per charge for
SCAN, trees per charge for TREE 360): report only, never gates. Every default below is an ASSUMPTION.
"""
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from atharv_nandi.core.energy_model import RESERVE_PACK_FRACTION, EnergyParams

REASON_ENERGY_RESERVE = "energy_reserve"   # MISSION-PRESETS 0.4 reason code
ACTION_CONTINUE = "continue"
ACTION_RETURN_AND_WAIT = "return_and_wait"

# ASSUMED crown-mast drive: about 12 W while moving at 0.2 m/s plus lifting the 0.8 kg moving mass at eta 0.5
# -> 12 / 0.2 / 3600 + 0.8 * 9.81 / 0.5 / 3600 ~ 0.021 Wh per metre of travel. The sim mast is kinematic (no motor
# torque), so the platform hook adds this model energy to the integrated electrical energy.
ASSUMED_CROWN_MOVE_W = 12.0
ASSUMED_CROWN_SPEED_MPS = 0.2
ASSUMED_CROWN_MOVING_KG = 0.8
ASSUMED_CROWN_ETA = 0.5
ASSUMED_CROWN_WH_PER_M = (ASSUMED_CROWN_MOVE_W / ASSUMED_CROWN_SPEED_MPS
                          + ASSUMED_CROWN_MOVING_KG * 9.81 / ASSUMED_CROWN_ETA) / 3600.0

# Field throughput assumptions (ORCHARD-MVP-SPEC 1.3; report only).
HDP_TREES_PER_HA = 800            # 5 x 2.5 m block
HDP_ALLEY_M_PER_HA = 2100.0       # ~2.1 km of alley per ha incl. headlands (computed in the spec)
HDP_TREE_SIDES_PER_HA = 1600
SCAN_SPEED_MPS = 0.18             # speed_scale 0.5, stowed (measured, probe)
SCAN_STOP_S = 3.0                 # stop per tree-side for the burst
TREE360_S = 155.0                 # TREE 360 per tree (computed in the spec)
TREE360_ORBIT_M = 2 * math.pi * 1.2 + 0.6   # one lap at r 1.2 m + overlap (ASSUMED)
TREE360_TRANSIT_M = 5.0           # walk to the next flagged tree (ASSUMED)
TREE360_CROWN_M = 2 * 0.6 + 4 * 0.6         # raise to mid, stow, ~2 snakes mid <-> high (ASSUMED)
RETURN_ALLOWANCE_M = 100.0        # walk back to the headland at the end of a charge (ASSUMED)
TB_PER_HA_TREE360 = 1.1           # real Pi stills, 1.35 GB per tree (computed in the spec)

FEATURES = ("route_m", "items", "t_work_h", "crown_m")
COEF_KEYS = ("wh_per_m", "wh_per_item", "p_idle_w", "wh_per_crown_m")


@dataclass(frozen=True)
class RoutePlan:
    """A (remaining) route: metres walked, items worked, total duration, crown-mast travel; plus the walk back to
    the headland from the route's END (for the reserve)."""
    route_m: float = 0.0
    items: int = 0
    t_work_s: float = 0.0
    crown_m: float = 0.0
    return_m: float = 0.0

    def __post_init__(self):
        for k in ("route_m", "t_work_s", "crown_m", "return_m"):
            v = getattr(self, k)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValueError(f"RoutePlan.{k} must be a finite number >= 0")
        if not (isinstance(self.items, int) and self.items >= 0):
            raise ValueError("RoutePlan.items must be an int >= 0")

    @classmethod
    def from_dict(cls, d: dict) -> "RoutePlan":
        return cls(route_m=float(d.get("route_m", 0.0)), items=int(d.get("items", 0)),
                   t_work_s=float(d.get("t_work_s", 0.0)), crown_m=float(d.get("crown_m", 0.0)),
                   return_m=float(d.get("return_m", 0.0)))

    def features(self) -> Tuple[float, float, float, float]:
        return (self.route_m, float(self.items), self.t_work_s / 3600.0, self.crown_m)


@dataclass(frozen=True)
class Coefficients:
    wh_per_m: float
    wh_per_item: float
    p_idle_w: float
    wh_per_crown_m: float = ASSUMED_CROWN_WH_PER_M
    walk_speed_mps: float = SCAN_SPEED_MPS   # for the duration of a return walk
    source: str = "seed"                     # seed | run_card | measured
    n_segments: int = 0

    def vector(self) -> Tuple[float, float, float, float]:
        return (self.wh_per_m, self.wh_per_item, self.p_idle_w, self.wh_per_crown_m)

    def energy_wh(self, plan: RoutePlan) -> float:
        return sum(c * f for c, f in zip(self.vector(), plan.features()))

    def return_wh(self, return_m: float) -> float:
        """Walk back to the headland: walking energy + idle for the walk time."""
        r = max(0.0, float(return_m))
        return self.wh_per_m * r + self.p_idle_w * (r / self.walk_speed_mps) / 3600.0

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @classmethod
    def seed(cls, params: EnergyParams = EnergyParams(), walk_speed_mps: Optional[float] = None) -> "Coefficients":
        """From the ASSUMED EnergyParams: walking above idle per metre at the planning speed."""
        v = float(walk_speed_mps or params.walk_speed_mps)
        return cls(wh_per_m=max(0.0, params.walk_power_w - params.p_idle_w) / v / 3600.0,
                   wh_per_item=params.pick_energy_wh, p_idle_w=params.p_idle_w,
                   wh_per_crown_m=ASSUMED_CROWN_WH_PER_M, walk_speed_mps=v, source="seed")

    @classmethod
    def from_dict(cls, d: dict, source: str = "run_card") -> "Coefficients":
        vals = {}
        for k in COEF_KEYS + ("walk_speed_mps",):
            if k in d:
                v = float(d[k])
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f"coefficient {k} must be finite and >= 0")
                vals[k] = v
        missing = [k for k in COEF_KEYS[:3] if k not in vals]
        if missing:
            raise ValueError(f"run card coefficients missing {missing}")
        return cls(source=source, n_segments=int(d.get("n_segments", 0)), **vals)

    @classmethod
    def from_run_card(cls, card, fallback: Optional["Coefficients"] = None) -> "Coefficients":
        """Seed from the last run card (dict or path): ``card["energy"]["coefficients"]``. A missing or malformed
        card falls back to ``fallback`` (default: the EnergyParams seed)."""
        fb = fallback or cls.seed()
        try:
            if isinstance(card, str):
                if not os.path.isfile(card):
                    return fb
                with open(card, encoding="utf-8") as fh:
                    card = json.load(fh)
            return cls.from_dict(card["energy"]["coefficients"], source="run_card")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return fb


def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting (4 x 4; a is SPD by construction)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        if abs(piv) < 1e-300:
            raise ValueError("singular system")
        for r in range(c + 1, n):
            f = m[r][c] / piv
            if f:
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


@dataclass
class Segment:
    route_m: float = 0.0
    items: float = 0.0
    t_s: float = 0.0
    crown_m: float = 0.0
    energy_wh: float = 0.0

    def features(self) -> Tuple[float, float, float, float]:
        return (self.route_m, self.items, self.t_s / 3600.0, self.crown_m)


class CoefficientTracker:
    """Measures the estimator coefficients in this run.

    Feed it increments (``update``) from the integrated energy; it cuts them into segments of ``segment_s`` sim
    seconds and fits ``E_seg = c . f_seg`` by ridge regression toward the seed coefficients: each coefficient k
    has a Gaussian prior N(seed_k, (prior_rel * max(seed_k, floor_k))^2) against an observation noise
    ``sigma_wh`` per segment, so a coefficient the run never excites (no items, no crown motion) stays at the
    seed, and one it does excite converges to the measured value within a few segments."""

    FLOORS = (0.05, 0.05, 5.0, 0.01)

    def __init__(self, seed: Coefficients, segment_s: float = 5.0, prior_rel: float = 0.5, sigma_wh: float = 0.02):
        if segment_s <= 0 or prior_rel <= 0 or sigma_wh <= 0:
            raise ValueError("segment_s, prior_rel and sigma_wh must be > 0")
        self.seed = seed
        self.segment_s = float(segment_s)
        self.prior_rel = float(prior_rel)
        self.sigma_wh = float(sigma_wh)
        self.segments: List[Segment] = []
        self._cur = Segment()
        self.total = Segment()
        self._coef: Optional[Coefficients] = None

    def reset(self):
        self.segments, self._cur, self.total, self._coef = [], Segment(), Segment(), None

    def update(self, dt_s: float, route_m: float = 0.0, items: float = 0.0, crown_m: float = 0.0,
               energy_wh: float = 0.0):
        """Add one interval's increments (all >= 0)."""
        vals = (dt_s, route_m, items, crown_m, energy_wh)
        if not all(math.isfinite(v) and v >= 0 for v in vals):
            return
        for s in (self._cur, self.total):
            s.t_s += dt_s
            s.route_m += route_m
            s.items += items
            s.crown_m += crown_m
            s.energy_wh += energy_wh
        if self._cur.t_s >= self.segment_s - 1e-9:
            self.add_segment(self._cur)
            self._cur = Segment()

    def add_segment(self, seg: Segment):
        self.segments.append(seg)
        self._coef = None

    def coefficients(self) -> Coefficients:
        if self._coef is not None:
            return self._coef
        if not self.segments:
            self._coef = self.seed
            return self._coef
        s2 = self.sigma_wh ** 2
        prior = self.seed.vector()
        lam = [s2 / (self.prior_rel * max(p, f)) ** 2 for p, f in zip(prior, self.FLOORS)]
        a = [[0.0] * 4 for _ in range(4)]
        b = [0.0] * 4
        for seg in self.segments:
            f = seg.features()
            for i in range(4):
                b[i] += f[i] * seg.energy_wh
                for j in range(4):
                    a[i][j] += f[i] * f[j]
        for i in range(4):
            a[i][i] += lam[i]
            b[i] += lam[i] * prior[i]
        x = [max(0.0, v) for v in _solve(a, b)]
        self._coef = Coefficients(wh_per_m=x[0], wh_per_item=x[1], p_idle_w=x[2], wh_per_crown_m=x[3],
                                  walk_speed_mps=self.seed.walk_speed_mps, source="measured",
                                  n_segments=len(self.segments))
        return self._coef


# ------------------------------------------------------------------ reserve, Ready check, mid-tour monitor
def reserve_soc(return_m: float, coeffs: Coefficients, capacity_wh: float,
                pack_fraction: float = RESERVE_PACK_FRACTION) -> float:
    """SoC to walk ``return_m`` back to the headland + ``pack_fraction`` of the pack."""
    return coeffs.return_wh(return_m) / capacity_wh + pack_fraction


def projected_end_soc(soc: float, remaining: RoutePlan, coeffs: Coefficients, capacity_wh: float) -> float:
    return float(soc) - coeffs.energy_wh(remaining) / capacity_wh


@dataclass(frozen=True)
class EnergyCheck:
    ok: bool
    reason: str
    projected_end_soc: float
    reserve: float
    need_wh: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def ready_check_energy(soc: float, plan: RoutePlan, coeffs: Coefficients, capacity_wh: float,
                       pack_fraction: float = RESERVE_PACK_FRACTION) -> EnergyCheck:
    """Ready check (MP 0.2): projected end SoC >= reserve at the tour end, else ``energy_reserve``."""
    need = coeffs.energy_wh(plan)
    end = projected_end_soc(soc, plan, coeffs, capacity_wh)
    res = reserve_soc(plan.return_m, coeffs, capacity_wh, pack_fraction)
    if end < res:
        return EnergyCheck(False, REASON_ENERGY_RESERVE, end, res, need,
                           f"projected end SoC {end:.3f} < reserve {res:.3f} (coefficients {coeffs.source})")
    return EnergyCheck(True, "", end, res, need)


class ReserveMonitor:
    """Mid-tour projection check. A breach that persists ``confirm_s`` sim seconds latches ``return_and_wait``
    (reason energy_reserve) until ``clear()`` (operator, or the dock charged the pack)."""

    def __init__(self, capacity_wh: float, pack_fraction: float = RESERVE_PACK_FRACTION, confirm_s: float = 1.0):
        self.capacity_wh = float(capacity_wh)
        self.pack_fraction = float(pack_fraction)
        self.confirm_s = float(confirm_s)
        self.clear()

    def clear(self):
        self.action = ACTION_CONTINUE
        self.reason = ""
        self.latched_at: Optional[float] = None
        self._breach_since: Optional[float] = None
        self.last: Optional[EnergyCheck] = None

    def update(self, t: float, soc: float, remaining: RoutePlan, coeffs: Coefficients) -> str:
        chk = ready_check_energy(soc, remaining, coeffs, self.capacity_wh, self.pack_fraction)
        self.last = chk
        if self.action == ACTION_RETURN_AND_WAIT:
            return self.action
        if chk.ok:
            self._breach_since = None
        else:
            if self._breach_since is None:
                self._breach_since = t
            if t - self._breach_since >= self.confirm_s - 1e-9:
                self.action, self.reason, self.latched_at = ACTION_RETURN_AND_WAIT, REASON_ENERGY_RESERVE, t
        return self.action

    def status(self) -> dict:
        out = {"action": self.action, "reason": self.reason, "latched_at": self.latched_at}
        if self.last is not None:
            out.update(projected_end_soc=round(self.last.projected_end_soc, 5), reserve=round(self.last.reserve, 5))
        return out


# ------------------------------------------------------------------ estimator error and endurance (report only)
def estimator_error(predicted_wh: float, integrated_wh: float) -> float:
    """|predicted - integrated| / integrated (the <= 15 % acceptance)."""
    if integrated_wh <= 0:
        raise ValueError("integrated energy must be > 0")
    return abs(predicted_wh - integrated_wh) / integrated_wh


def throughput(coeffs: Coefficients, capacity_wh: float, pack_fraction: float = RESERVE_PACK_FRACTION,
               return_m: float = RETURN_ALLOWANCE_M) -> dict:
    """ha per charge (SCAN, stowed, with 3 s stops) and trees per charge (TREE 360) from the coefficients.

    Usable energy per charge = pack x (1 - reserve fraction) - the walk back to the headland. Report only."""
    usable = capacity_wh * (1.0 - pack_fraction) - coeffs.return_wh(return_m)
    scan_t = HDP_ALLEY_M_PER_HA / SCAN_SPEED_MPS + HDP_TREE_SIDES_PER_HA * SCAN_STOP_S
    scan_nostop_t = HDP_ALLEY_M_PER_HA / SCAN_SPEED_MPS
    scan_wh = coeffs.energy_wh(RoutePlan(route_m=HDP_ALLEY_M_PER_HA, t_work_s=scan_t))
    t360_wh = coeffs.energy_wh(RoutePlan(route_m=TREE360_ORBIT_M + TREE360_TRANSIT_M,
                                         t_work_s=TREE360_S + TREE360_TRANSIT_M / SCAN_SPEED_MPS,
                                         crown_m=TREE360_CROWN_M))
    ha = usable / scan_wh if scan_wh > 0 else math.inf
    trees = usable / t360_wh if t360_wh > 0 else math.inf
    return {
        "usable_wh_per_charge": round(usable, 2),
        "scan_wh_per_ha": round(scan_wh, 2),
        "scan_h_per_ha": round(scan_t / 3600.0, 2),
        "scan_h_per_ha_no_stops": round(scan_nostop_t / 3600.0, 2),
        "ha_per_charge": round(ha, 3),
        "wh_per_100m": round(coeffs.energy_wh(RoutePlan(route_m=100.0, t_work_s=100.0 / SCAN_SPEED_MPS)), 3),
        "tree360_wh_per_tree": round(t360_wh, 3),
        "trees_per_charge": round(trees, 1),
        "tree360_h_per_ha": round(HDP_TREES_PER_HA * (TREE360_S + TREE360_TRANSIT_M / SCAN_SPEED_MPS) / 3600, 1),
        "tree360_tb_per_ha": TB_PER_HA_TREE360,
        "coefficients": coeffs.to_dict(),
        "assumed": True,
    }


def run_card_energy(coeffs: Coefficients, capacity_wh: float, integrated_wh: Optional[float] = None,
                    predicted_wh: Optional[float] = None) -> dict:
    """The ``energy`` block of a run card: coefficients (seed for the next run), estimator error, endurance."""
    out = {"coefficients": coeffs.to_dict(), "throughput": throughput(coeffs, capacity_wh), "assumed": True}
    if integrated_wh is not None:
        out["integrated_wh"] = round(integrated_wh, 4)
    if integrated_wh is not None and predicted_wh is not None and integrated_wh > 0:
        out["predicted_wh"] = round(predicted_wh, 4)
        out["estimator_error"] = round(estimator_error(predicted_wh, integrated_wh), 4)
    return out
