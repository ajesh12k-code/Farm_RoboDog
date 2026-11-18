"""Load ratings and capacity of Atharv Nandi: crop profiles, mount ratings, basket, arm, energy.

Pure Python + numpy (no MuJoCo, no ROS). Numbers are fixed by the physics alignment spec v1
(2026-11-17) sections 4 and 5; each carries a tag:
    MEASURED  measured in the sim (audit ref: mass-load 49a9cb0, contact-solver c98754d,
              mounting-stability 25efcff; all local scratch branches)
    CITED     from a datasheet / listing (URL in the comment)
    DERIVED   arithmetic shown next to it
    ASSUMED   a design assumption; replace with a measurement or a citation before field use

Status is advisory (spec G5): these checks feed /nandi/stability; nothing enforces them yet.

Ratings table (mount site -> rating):
    basket_payload   10 kg walking, 15 kg absolute (stand / slow walk speed_scale <= 0.5);
                     software limit = min(10 kg, 10.8 L x 0.6 kg/L = 6.5 kg by volume)
    tool_flange      tool + produce by horizontal reach from the shoulder:
                     <= 0.5 m: 2.0 kg, <= 0.8 m: 1.5 kg, > 0.8 m: 0 kg
    mast_head_ring   1.5 kg rated; 3.0 kg only in the SCAN_WIDE stance on slopes <= 5 deg
    mast_base        design load 290 Nm / 185 N (2x the measured worst case, mounting audit E/F/I)
    front_mount      5 kg, used in pairs with rear_mount
    rear_mount       5 kg, used in pairs with front_mount
    tow_dock         50 N pull PLACEHOLDER; an unpaired rear load <= 5 kg
"""
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

G = 9.81

# --- crops ----------------------------------------------------------------------------------------
METHOD_GRASP = "grasp"           # placeholder sphere: plain 2-finger grasp
METHOD_CLIP_PEDICEL = "clip_pedicel"
METHOD_CLIP_OR_TWIST = "clip_or_twist"
METHOD_TWIST_OR_CLIP_TRUNK = "twist_or_clip_from_trunk"
METHOD_ASSIST_MARK = "assist_mark"

DECISION_OK = "OK"
DECISION_REJECT = "REJECT"
DECISION_ASSIST_ONLY = "ASSIST_ONLY"


@dataclass(frozen=True)
class CropProfile:
    name: str
    mass_min_kg: float
    mass_max_kg: float
    method: str
    assist_only: bool
    tag: str
    note: str = ""

    @property
    def mass_typ_kg(self) -> float:
        return 0.5 * (self.mass_min_kg + self.mass_max_kg)


CROP_PROFILES: Dict[str, CropProfile] = {p.name: p for p in (
    # MEASURED: the 0.15 kg r=0.035 m sphere of farm_scene / the robot MJCF (ground placeholder only).
    CropProfile("placeholder_sphere", 0.15, 0.15, METHOD_GRASP, False, "MEASURED",
                "farm_scene crop sphere, r 0.035 m"),
    # ASSUMED until cited (spec v1 section 5): mango 0.20-0.50 kg, clip leaving a 1-2 cm pedicel.
    CropProfile("mango", 0.20, 0.50, METHOD_CLIP_PEDICEL, False, "ASSUMED",
                "clip the stem leaving a 1-2 cm pedicel (limits sap burn)"),
    # ASSUMED until cited: guava 0.10-0.25 kg, clip or twist.
    CropProfile("guava", 0.10, 0.25, METHOD_CLIP_OR_TWIST, False, "ASSUMED", "clip or twist"),
    # ASSUMED until cited: papaya 0.5-2.5 kg, twist / clip from the trunk; pick only if tool + fruit
    # fits the reach rating, otherwise assist.
    CropProfile("papaya", 0.5, 2.5, METHOD_TWIST_OR_CLIP_TRUNK, False, "ASSUMED",
                "twist or clip from the trunk; assist if over the reach rating"),
    # Project brief (2026-11-17): a banana bunch is 15-40 kg, beyond the basket and the arm.
    CropProfile("banana_bunch", 15.0, 40.0, METHOD_ASSIST_MARK, True, "PROJECT_BRIEF",
                "assist / mark only: exceeds the basket and the arm"),
)}
CROP_ALIASES = {"banana": "banana_bunch", "sphere": "placeholder_sphere", "placeholder": "placeholder_sphere"}


def crop_profile(name: str) -> CropProfile:
    key = CROP_ALIASES.get(name.strip().lower(), name.strip().lower())
    if key not in CROP_PROFILES:
        raise KeyError(f"unknown crop {name!r}; known: {sorted(CROP_PROFILES)}")
    return CROP_PROFILES[key]


# --- mount ratings ------------------------------------------------------------------------------
@dataclass(frozen=True)
class MountRating:
    site: str
    rated_kg: Optional[float]
    max_kg: Optional[float]
    tag: str
    note: str = ""
    force_n: Optional[float] = None
    moment_nm: Optional[float] = None


# Basket. MEASURED basis (mounting audit 25efcff): tripod belly 0.155 m at 15 kg, knee scrape from
# 20 kg at leg kp 120; re-validated by Package A's acceptance tests after the kp change.
BASKET_RATED_WALK_KG = 10.0
BASKET_MAX_KG = 15.0                 # stand / slow walk only
BASKET_SLOW_SPEED_SCALE = 0.5        # "slow walk" = speed_scale <= 0.5
# DERIVED from the MJCF basket: floor 0.64 x 0.24 m, walls 0.07 m -> 0.64 * 0.24 * 0.07 = 0.01075 m^3
BASKET_VOLUME_L = 10.8
BULK_DENSITY_KG_PER_L = 0.6          # ASSUMED bulk fruit (loose fruit with voids)
BASKET_VOLUME_LIMIT_KG = BASKET_VOLUME_L * BULK_DENSITY_KG_PER_L  # DERIVED 10.8 * 0.6 = 6.48 (spec: ~6.5)

# Tool flange: (max horizontal reach from the shoulder m, max tool + produce kg). DERIVED (spec v1
# section 5): shoulder budget 0.6 x 30 Nm = 18 Nm, minus 2.67 Nm arm self-weight (MEASURED, mass-load
# audit 49a9cb0) -> 15.33 Nm / (9.81 * 0.8 m) = 1.95 kg at 0.8 m, rated 1.5 kg; elbow budget
# 0.6 x 15 Nm = 9 Nm -> ~1.8 kg at 0.45 m (forearm + tool), rated 2.0 kg inside 0.5 m.
ARM_REACH_TABLE: Tuple[Tuple[float, float], ...] = ((0.5, 2.0), (0.8, 1.5))
ARM_MAX_REACH_M = ARM_REACH_TABLE[-1][0]

MAST_HEAD_RATED_KG = 1.5
MAST_HEAD_WIDE_KG = 3.0              # SCAN_WIDE stance, slope <= 5 deg only
# DERIVED (spec v1 section 5): 2x the measured worst case of mounting audit 25efcff E/F/I
# (290 / 2 = 145 Nm, 185 / 2 = 92.5 N measured).
MAST_BASE_DESIGN_MOMENT_NM = 290.0
MAST_BASE_DESIGN_FORCE_N = 185.0
END_MOUNT_RATED_KG = 5.0             # front_mount / rear_mount, used in pairs (ASSUMED)
TOW_PULL_N = 50.0                    # PLACEHOLDER (ASSUMED)
REAR_UNPAIRED_MAX_KG = 5.0           # DERIVED CoM shift: see total_mass_and_com_shift()

MOUNT_RATINGS: Dict[str, MountRating] = {r.site: r for r in (
    MountRating("basket_payload", BASKET_RATED_WALK_KG, BASKET_MAX_KG, "MEASURED",
                f"walk {BASKET_RATED_WALK_KG:g} kg; stand / speed_scale <= {BASKET_SLOW_SPEED_SCALE} up to "
                f"{BASKET_MAX_KG:g} kg; volume {BASKET_VOLUME_L} L x {BULK_DENSITY_KG_PER_L} kg/L"),
    MountRating("tool_flange", ARM_REACH_TABLE[0][1], ARM_REACH_TABLE[0][1], "DERIVED",
                "tool + produce by reach: <= 0.5 m 2.0 kg, <= 0.8 m 1.5 kg, > 0.8 m 0"),
    MountRating("mast_head_ring", MAST_HEAD_RATED_KG, MAST_HEAD_WIDE_KG, "DERIVED",
                "3.0 kg only in SCAN_WIDE on slopes <= 5 deg"),
    MountRating("mast_base", None, None, "DERIVED", "design load, 2x measured worst case",
                force_n=MAST_BASE_DESIGN_FORCE_N, moment_nm=MAST_BASE_DESIGN_MOMENT_NM),
    MountRating("front_mount", END_MOUNT_RATED_KG, END_MOUNT_RATED_KG, "ASSUMED", "used in pairs with rear_mount"),
    MountRating("rear_mount", END_MOUNT_RATED_KG, END_MOUNT_RATED_KG, "ASSUMED", "used in pairs with front_mount"),
    MountRating("tow_dock", REAR_UNPAIRED_MAX_KG, REAR_UNPAIRED_MAX_KG, "ASSUMED",
                "50 N pull PLACEHOLDER; unpaired rear load <= 5 kg", force_n=TOW_PULL_N),
)}


# --- basket -------------------------------------------------------------------------------------
def basket_capacity(walking: bool = True, speed_scale: float = 1.0,
                    bulk_density_kg_per_l: float = BULK_DENSITY_KG_PER_L) -> dict:
    """Mass and volume limits of the dorsal basket; the smaller one binds."""
    if walking and speed_scale > BASKET_SLOW_SPEED_SCALE:
        mass_limit = BASKET_RATED_WALK_KG
    else:
        mass_limit = BASKET_MAX_KG
    vol_limit = BASKET_VOLUME_L * bulk_density_kg_per_l
    binding = "volume" if vol_limit < mass_limit else "mass"
    return {"mass_limit_kg": mass_limit, "volume_l": BASKET_VOLUME_L,
            "bulk_density_kg_per_l": bulk_density_kg_per_l, "volume_limit_kg": vol_limit,
            "limit_kg": min(mass_limit, vol_limit), "binding": binding}


def basket_mass(load: Union[float, Mapping[str, int]]) -> float:
    """kg of a basket load given as kg or as {crop name: count} (typical mass per crop)."""
    if isinstance(load, Mapping):
        return float(sum(crop_profile(c).mass_typ_kg * int(n) for c, n in load.items()))
    return float(load)


def check_basket(load: Union[float, Mapping[str, int]], walking: bool = True, speed_scale: float = 1.0) -> dict:
    """Is this basket load within limits? Never above BASKET_MAX_KG whatever the mode."""
    kg = basket_mass(load)
    if kg < 0 or not math.isfinite(kg):
        raise ValueError(f"bad basket load {kg}")
    cap = basket_capacity(walking, speed_scale)
    ok = kg <= cap["limit_kg"] + 1e-9 and kg <= BASKET_MAX_KG
    return {"ok": ok, "kg": kg, "limit_kg": cap["limit_kg"], "binding": cap["binding"],
            "mass_limit_kg": cap["mass_limit_kg"], "volume_limit_kg": cap["volume_limit_kg"]}


# --- arm ----------------------------------------------------------------------------------------
def arm_payload_limit(reach_m: float) -> float:
    """Max tool + produce (kg) at a horizontal reach from the shoulder (spec v1 section 5)."""
    if reach_m < 0 or not math.isfinite(reach_m):
        raise ValueError(f"bad reach {reach_m}")
    for max_reach, kg in ARM_REACH_TABLE:
        if reach_m <= max_reach + 1e-12:
            return kg
    return 0.0


def check_pick(crop: str, est_mass_kg: Optional[float], tool_mass_kg: float, reach_m: float,
               basket_kg: Optional[float] = None) -> dict:
    """Gate a pick / clip: decision OK, REJECT (over the arm or basket limit) or ASSIST_ONLY."""
    prof = crop_profile(crop)
    mass = prof.mass_max_kg if est_mass_kg is None else float(est_mass_kg)
    reasons: List[str] = []
    if prof.assist_only:
        return {"decision": DECISION_ASSIST_ONLY, "ok": False, "crop": prof.name, "method": prof.method,
                "total_kg": mass + tool_mass_kg, "limit_kg": arm_payload_limit(reach_m),
                "reasons": [f"{prof.name} {prof.mass_min_kg:g}-{prof.mass_max_kg:g} kg: assist / mark only"]}
    limit = arm_payload_limit(reach_m)
    total = mass + float(tool_mass_kg)
    if total > limit + 1e-9:
        reasons.append(f"tool + fruit {total:.2f} kg > {limit:.2f} kg at reach {reach_m:.2f} m"
                       + ("; assist instead" if prof.name == "papaya" else ""))
    if basket_kg is not None:
        b = check_basket(float(basket_kg) + mass)
        if not b["ok"]:
            reasons.append(f"basket would hold {b['kg']:.2f} kg > {b['limit_kg']:.2f} kg ({b['binding']})")
    decision = DECISION_REJECT if reasons else DECISION_OK
    return {"decision": decision, "ok": not reasons, "crop": prof.name, "method": prof.method,
            "total_kg": total, "limit_kg": limit, "reasons": reasons}


# --- mass and CoM -------------------------------------------------------------------------------
def total_mass_and_com_shift(base_mass_kg: float, base_com: Sequence[float],
                             loads: Iterable[Tuple[float, Sequence[float]]]) -> dict:
    """Combine the robot with point loads [(kg, (x, y, z))]: total mass, new CoM, CoM shift.

    Example (DERIVED, spec v1 section 5): 5 kg at the rear_mount x = -0.34 m on a 25 kg robot with its
    CoM at x = 0 shifts the CoM by 5 * -0.34 / 30 = -0.057 m (on the 31.8 kg target: -0.046 m).
    """
    m = float(base_mass_kg)
    moment = m * np.asarray(base_com, dtype=np.float64)
    for kg, pos in loads:
        m += float(kg)
        moment = moment + float(kg) * np.asarray(pos, dtype=np.float64)
    com = moment / m
    return {"total_kg": m, "com": com.tolist(),
            "shift": (com - np.asarray(base_com, dtype=np.float64)).tolist()}


# --- leg torque RMS -------------------------------------------------------------------------------
# CITED: AK80-64 rated (continuous) torque 48 Nm, peak 120 Nm
# (store.tmotor.com goods-1202; robotshop cubemars ak80-64; read 2026-11-17).
LEG_CONTINUOUS_NM = 48.0
LEG_PEAK_NM = 120.0
RMS_WINDOW_S = 10.0                  # ASSUMED thermal window for the continuous-torque check


def rms_torque(window: Sequence[Sequence[float]], limit_nm: float = LEG_CONTINUOUS_NM) -> dict:
    """RMS torque per joint over a window of effort samples (rows = samples, cols = joints)."""
    a = np.asarray(window, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.size == 0:
        return {"rms_nm": [], "max_rms_nm": None, "over": [], "limit_nm": limit_nm}
    rms = np.sqrt(np.mean(a * a, axis=0))
    return {"rms_nm": rms.tolist(), "max_rms_nm": float(rms.max()),
            "over": [int(i) for i in np.nonzero(rms > limit_nm)[0]], "limit_nm": limit_nm}


class TorqueRmsWindow:
    """Rolling RMS of leg efforts over RMS_WINDOW_S (feed (t, efforts) samples)."""

    def __init__(self, window_s: float = RMS_WINDOW_S):
        self.window_s = window_s
        self._t: List[float] = []
        self._rows: List[List[float]] = []

    def add(self, t: float, efforts: Sequence[float]):
        self._t.append(float(t))
        self._rows.append([float(e) for e in efforts])
        while self._t and self._t[0] < t - self.window_s:
            self._t.pop(0)
            self._rows.pop(0)

    def summary(self) -> dict:
        return rms_torque(self._rows)


# --- energy -------------------------------------------------------------------------------------
PACK_WH = 500.0                      # ASSUMED 500 Wh Li-ion pack (~155 Wh/kg, 3.2 kg); = energy_model after Package A
DRIVE_EFFICIENCY = 0.6               # ASSUMED motor + driver efficiency, mechanical -> electrical
HOTEL_LOAD_W = 15.0                  # ASSUMED compute + sensors
# MEASURED walking mechanical power 146-202 W (mass-load audit 49a9cb0, M9).
# DERIVED: 146 / 0.6 + 15 = 258 W electrical -> 500 / 258 = 1.94 h.
WALK_MECH_W_RANGE = (146.0, 202.0)


def electrical_power_w(p_mech_w: float) -> float:
    """Electrical draw estimate: P_mech / efficiency + hotel load (negative mech power counts as 0)."""
    return max(0.0, float(p_mech_w)) / DRIVE_EFFICIENCY + HOTEL_LOAD_W


def energy_remaining(p_mech_w: float, soc: Optional[float] = None, used_wh: Optional[float] = None,
                     pack_wh: float = PACK_WH) -> dict:
    """Remaining energy and runtime at the given mechanical power. Give soc (0..1) or used_wh."""
    if soc is not None:
        rem = max(0.0, min(1.0, float(soc))) * pack_wh
    elif used_wh is not None:
        rem = max(0.0, pack_wh - float(used_wh))
    else:
        rem = pack_wh
    p = electrical_power_w(p_mech_w)
    return {"remaining_wh": rem, "p_elec_w": p, "hours": rem / p, "pack_wh": pack_wh}
