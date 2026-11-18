"""Weather stub for Atharv Nandi: rain, wet foliage, ambient temperature (WP7; pure Python, no ROS).

ORCHARD-MVP-SPEC 4.5 Ready checks: rain no-go; wet foliage no-go for LEAF SCAN and the contact presets; ambient
temperature > 42 C no-go, > 38 C degraded (ASSUMED thresholds for the agronomist and the safety case). The Ready
check itself (and the season matrix) is WP27's; this module produces the weather state and the per-preset
weather verdicts it consumes.

The state is scripted, never measured: a date and a start hour (sim t = 0), rain events, and an UP (North India /
Lucknow plain) diurnal temperature from approximate monthly normal maxima and minima (ASSUMED, IMD-normal-like
values rounded; replace with station data). Foliage stays wet ``dry_h`` hours (ASSUMED 2 h) after rain stops;
optional morning dew (off by default because LEAF SCAN dawn mode runs before 08:00).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# (mean daily max, mean daily min) C per month, Lucknow plain (ASSUMED approximations of the normals).
UP_MONTHLY_MAX_MIN_C = {1: (22.0, 8.0), 2: (26.0, 11.0), 3: (32.0, 16.0), 4: (38.0, 22.0), 5: (41.0, 26.0),
                        6: (39.0, 28.0), 7: (34.0, 27.0), 8: (33.0, 26.0), 9: (33.0, 25.0), 10: (32.0, 20.0),
                        11: (28.0, 14.0), 12: (23.0, 9.0)}
T_MIN_HOUR = 6.0          # daily minimum near sunrise (ASSUMED)
T_MAX_HOUR = 15.0         # daily maximum mid-afternoon (ASSUMED)
HEAT_NO_GO_C = 42.0
HEAT_DEGRADED_C = 38.0
WET_DRY_H = 2.0
DEW_UNTIL_HOUR = 8.0
DAWN_MODE_UNTIL_HOUR = 8.0

GO, DEGRADED, NO_GO = "go", "degraded", "no_go"
R_RAIN, R_WET, R_HEAT = "rain", "wet_foliage", "heat"

# Preset classes. A preset id belongs to a class when it equals a class id or starts with "<class id>_"
# (per-crop and per-mode ids: harvest_mango, clip_prune, leaf_scan_dawn, work_assist, ...). An id that matches no
# class is treated as a contact preset (the strictest weather rule), so a new preset is never let through wet.
CONTACT_PRESETS = ("harvest", "prune", "work", "clip", "pick")
LEAF_PRESETS = ("leaf_scan",)
NON_CONTACT_PRESETS = ("scan", "tree360", "tree_360", "trunk_scan", "base_scan", "trunk_base_scan", "fruit_scout",
                       "mark")
WEATHER_EXEMPT = ("return_home", "safe_park", "recovery", "dock", "unload", "estop_recovery")


@dataclass(frozen=True)
class RainEvent:
    t0: float
    dur: float
    mm_h: float = 5.0

    def __post_init__(self):
        if not (math.isfinite(self.t0) and self.dur > 0 and self.mm_h >= 0):
            raise ValueError("rain event needs finite t0, dur > 0, mm_h >= 0")


@dataclass
class WeatherScript:
    month: int = 11
    start_hour: float = 9.0
    rain: List[RainEvent] = field(default_factory=list)
    temp_c: Optional[float] = None          # fixed override (heat demos)
    temp_offset_c: float = 0.0              # e.g. +5 for a heat wave
    dry_h: float = WET_DRY_H
    dew: bool = False
    wet_at_start: bool = False              # foliage wet from rain before t = 0

    def __post_init__(self):
        if not 1 <= int(self.month) <= 12:
            raise ValueError("month must be 1..12")
        if not 0.0 <= self.start_hour < 24.0:
            raise ValueError("start_hour must be in [0, 24)")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "WeatherScript":
        d = dict(d or {})
        rain = [RainEvent(float(r["t0"]), float(r["dur"]), float(r.get("mm_h", 5.0))) for r in d.pop("rain", [])]
        kw = {}
        for k, v in d.items():
            if k == "month":
                kw[k] = int(v)
            elif k in ("dew", "wet_at_start"):
                kw[k] = bool(v)
            elif k == "temp_c":
                kw[k] = None if v is None else float(v)
            elif k in cls.__dataclass_fields__:
                kw[k] = float(v)
        return cls(rain=rain, **kw)

    def hour(self, t: float) -> float:
        return (self.start_hour + t / 3600.0) % 24.0


def diurnal_temp_c(month: int, hour: float) -> float:
    """Cosine between the monthly mean min (06:00) and max (15:00), rising and falling halves."""
    tmax, tmin = UP_MONTHLY_MAX_MIN_C[int(month)]
    h = hour % 24.0
    if T_MIN_HOUR <= h <= T_MAX_HOUR:
        x = (h - T_MIN_HOUR) / (T_MAX_HOUR - T_MIN_HOUR)        # 0 at min, 1 at max
    else:
        span = 24.0 - (T_MAX_HOUR - T_MIN_HOUR)
        x = 1.0 - (((h - T_MAX_HOUR) % 24.0) / span)            # 1 at max, 0 at next min
    return tmin + (tmax - tmin) * (1.0 - math.cos(math.pi * x)) / 2.0


class WeatherStub:
    def __init__(self, script: Optional[WeatherScript] = None):
        self.script = script or WeatherScript()
        self._scripted = list(self.script.rain)
        self._scripted_temp = self.script.temp_c

    def reset(self):
        """Back to the configured script: runtime rain and a runtime temp_c override are dropped."""
        self.script.rain = list(self._scripted)
        self.script.temp_c = self._scripted_temp

    def set_temp(self, temp_c: Optional[float]):
        """Runtime fixed temperature override (None = diurnal model again)."""
        if temp_c is not None and not (isinstance(temp_c, (int, float)) and not isinstance(temp_c, bool)
                                       and math.isfinite(temp_c) and -20.0 <= temp_c <= 60.0):
            raise ValueError("temp_c must be null or a number in [-20, 60]")
        self.script.temp_c = None if temp_c is None else float(temp_c)

    def set_rain(self, t: float, dur: float, mm_h: float = 5.0):
        """Runtime rain event (console / harness)."""
        self.script.rain.append(RainEvent(float(t), float(dur), float(mm_h)))

    def _rain_at(self, t: float) -> Tuple[float, Optional[float]]:
        """(mm/h now, end time of the latest finished rain)."""
        now, last_end = 0.0, None
        for r in self.script.rain:
            if r.t0 <= t < r.t0 + r.dur:
                now = max(now, r.mm_h)
            elif r.t0 + r.dur <= t:
                last_end = max(last_end or -math.inf, r.t0 + r.dur)
        return now, last_end

    def state(self, t: float) -> dict:
        s = self.script
        hour = s.hour(t)
        mm_h, last_end = self._rain_at(t)
        raining = mm_h > 0.0
        wet_rain = raining or (last_end is not None and t - last_end < s.dry_h * 3600.0)
        wet_start = s.wet_at_start and t < s.dry_h * 3600.0
        dew = s.dew and hour < DEW_UNTIL_HOUR
        temp = s.temp_c if s.temp_c is not None else diurnal_temp_c(s.month, hour) + s.temp_offset_c
        return {"t": round(float(t), 3), "month": int(s.month), "hour": round(hour, 3), "rain": raining,
                "rain_mm_h": mm_h, "wet_foliage": bool(wet_rain or wet_start or dew),
                "wet_cause": "rain" if wet_rain or wet_start else ("dew" if dew else None),
                "temp_c": round(temp, 2), "dawn_mode_window": hour < DAWN_MODE_UNTIL_HOUR,
                "source": "stub", "assumed": True}


def _in_class(p: str, ids) -> bool:
    return any(p == c or p.startswith(c + "_") for c in ids)


def preset_class(preset: str) -> str:
    """exempt | leaf | contact | non_contact | unknown (unknown is judged like contact)."""
    p = str(preset).strip().lower()
    if _in_class(p, WEATHER_EXEMPT):
        return "exempt"
    if _in_class(p, LEAF_PRESETS):
        return "leaf"
    if _in_class(p, CONTACT_PRESETS):
        return "contact"
    if _in_class(p, NON_CONTACT_PRESETS):
        return "non_contact"
    return "unknown"


def preset_verdict(state: dict, preset: str) -> dict:
    """Weather part of the Ready check for one preset class: {level: go|degraded|no_go, reasons: [...]}.

    rain -> no_go (except recovery / dock / return home / safe park); wet foliage -> no_go for LEAF SCAN and the
    contact presets (and any id of no known class); temp > 42 C -> no_go, > 38 C -> degraded (all non-exempt
    presets). Ids match a class exactly or by the "<class>_" prefix (``preset_class``)."""
    cls = preset_class(preset)
    if cls == "exempt":
        return {"level": GO, "reasons": []}
    reasons: List[Dict[str, str]] = []
    if state["rain"]:
        reasons.append({"code": R_RAIN, "level": NO_GO})
    if state["wet_foliage"] and cls in ("leaf", "contact", "unknown"):
        reasons.append({"code": R_WET, "level": NO_GO})
    if state["temp_c"] > HEAT_NO_GO_C:
        reasons.append({"code": R_HEAT, "level": NO_GO})
    elif state["temp_c"] > HEAT_DEGRADED_C:
        reasons.append({"code": R_HEAT, "level": DEGRADED})
    level = NO_GO if any(r["level"] == NO_GO for r in reasons) else (DEGRADED if reasons else GO)
    return {"level": level, "reasons": reasons}
