"""Pure mission executor logic for Atharv Nandi (no ROS, no MuJoCo).

Contract (README "Missions"):
    in   /nandi/mission/command  String JSON {mission_id, mode: harvest_zone|patrol, zone, not_after, max_picks?}
                                 or an orchard preset command {mission_id, mode: <PRESET_MODES>, not_after, plan}
    in   /nandi/mission/abort    Trigger
    out  /nandi/mission_status   String JSON, 2 Hz (see MissionController.status())

The ROS node (atharv_nandi/nodes/mission_executor.py) only gathers observations, calls step() at 10 Hz and carries
out the returned Actions; everything that decides anything lives here and is unit-tested
(tests/test_mission_logic.py) and closed-loop tested in bare MuJoCo (tools/mission_sim_check.py).

Counting rule: 'picked' moves ONLY when /nandi/game_status in_basket rises (independent truth);
the harvest node's own result text is recorded but never counted.

Orchard presets (WP27, MISSION-PRESETS 0.1): MODES = LEGACY_MODES + PRESET_MODES. Every preset mode maps to the one
internal cmd/mission type with ``plan {source in console | zone_alert | edge_hub, preset, version, targets, tree_ids,
params}`` (until the command-naming question is settled). ``parse_command`` validates both shapes; the legacy ``MissionController`` executes only
the legacy modes (a preset command is refused ``command_invalid``: presets go through ``PresetQueue``, and the
preset executor hook is WP9 / WP38's). ``PresetQueue`` keeps the console queue with ETA, projected SoC and free
disk; ``load_mission`` / ``tour_gate_violations`` / ``tour_estimate`` read config/missions/orchard_tour.yaml and
recovery_demo.yaml and check every scripted step against the WP6 gate table (tests/test_tour_gates.py).
"""
import json
import math
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from atharv_nandi import paths
from atharv_nandi.core import geo_frame as gf
from atharv_nandi.core import nav_logic as nav
from atharv_nandi.core.energy_model import EnergyParams, route_soc_needed

# ------------------------------------------------------------------ constants
STATES = ("idle", "accepted", "running", "returning", "done", "rejected", "failed", "aborted")
ACTIVE_STATES = ("accepted", "running", "returning")
LEGACY_MODES = ("harvest_zone", "patrol")
# MISSION-PRESETS 0.1 orchard modes (WP27)
PRESET_MODES = ("scan_rows", "tree360", "trunk_scan", "leaf_scan", "fruit_scout", "harvest_tree", "prune_tree",
                "tote_carry", "lead_to_marks", "follow_me", "return_home", "safe_park", "stuck_recover",
                "estop_recover", "self_right_spike", "dock_unload")
MODES = LEGACY_MODES + PRESET_MODES
PLAN_SOURCES = ("console", "zone_alert", "edge_hub")     # plan.source until the command naming is settled

REASONS = (  # reason_code values
    "", "expired", "zone_invalid", "zone_outside_geofence", "estopped", "battery_low", "basket_full",
    "busy", "command_invalid", "completed", "no_targets", "max_picks", "aborted", "fallen",
    "locomotion_mode", "geofence_breach",
)
TRIPOD_MODE = "tripod"    # /nandi/locomotion_status 'mode' a mission may walk with (nav_logic steers it)

REACH_M = 1.08            # = controllers.harvest_logic.REACH_M (auto_harvest refuses beyond this)
STANDOFF_X_M = 0.45       # stop with the crop this far ahead of the trunk centre ...
LATERAL_M = 0.38          # ... and at least this far to the side: clear of the feet (|y| ~0.25 + 0.03 foot)
STANDOFF_RANGE_M = (0.36, 0.56)  # trunk-frame x the arm reaches at LATERAL_M (IK map, flat ground)
CLOSER_STANDOFF_M = 0.38  # retry stop after "out of arm reach": nearer ...
CLOSER_LATERAL_M = 0.33   # ... and less to the side (still clear of the feet while standing)
ROW_GAP_M = 0.30          # crops whose y differ by more than this belong to different rows
CROP_CLEAR_M = 0.40       # a lane must stay this far (centre to centre) from crops it is not picking
PICK_WINDOW_X = (0.22, 0.75)   # trunk-frame x window in which a crop may be picked
PICK_WINDOW_Y = 0.65           # trunk-frame |y| limit for a pick
PICK_ARRIVE_TOL_M = 0.08  # pick stops and home; transit waypoints use nav.ARRIVE_TOL_M (0.15 m)
BLOCK_SHIFT_M = 0.20      # shift a stop this far along the row when a nearer crop would be harvested
ALIGN_TOL_RAD = 0.15
SETTLE_SPEED_MPS = 0.05
SETTLE_MIN_S = 0.6
SETTLE_MAX_S = 4.0
HARVEST_TIMEOUT_S = 60.0
HARVEST_START_GRACE_S = 3.0    # after a successful auto_harvest call, wait this long to see ik_harvest
DROP_SETTLE_S = 1.5            # after the arm is back, wait for game_status to catch up
MAX_ATTEMPTS = 2               # one try + one retry per crop
WAYPOINT_TIMEOUT_S = 90.0      # walking to one waypoint longer than this: skip it
CROP_MASS_KG = 0.15            # farm_scene crop mass
DETECTIONS_FRESH_S = 1.0
CONSOLE_TTL_S = 1800.0         # a console start command (no not_after) is valid this long
FENCE_SAMPLE_M = 0.05          # planned path segments are checked against the geofence at this spacing


@dataclass(frozen=True)
class BasketCapacity:
    max_crops: int = 6
    max_kg: float = 1.2
    crop_mass_kg: float = CROP_MASS_KG

    @property
    def limit(self) -> int:
        """Crops that fit: the stricter of the count and the mass limit."""
        by_mass = int(math.floor(self.max_kg / self.crop_mass_kg + 1e-9)) if self.crop_mass_kg > 0 else self.max_crops
        return max(0, min(self.max_crops, by_mass))

    def is_full(self, count: int) -> bool:
        return count >= self.limit


@dataclass(frozen=True)
class MissionCommand:
    mission_id: str
    mode: str
    zone: Optional[dict]
    not_after: datetime
    max_picks: Optional[int] = None
    plan: Optional[dict] = None          # orchard preset commands only (WP27)


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    kind: str                    # 'transit' | 'pick' | 'patrol' | 'home'
    crop: Optional[str] = None
    heading: Optional[float] = None  # face this way on arrival (pick stops face along the row)


@dataclass
class Observation:
    """Everything step() looks at. Times: t = monotonic seconds; utc = wall clock (for not_after)."""
    t: float
    utc: datetime
    pose: Optional[Tuple[float, float, float]] = None   # trunk x, y, yaw
    trunk_speed: float = 0.0
    walking: bool = False
    crops: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    crop_source: str = "truth"
    in_basket: Optional[int] = None
    in_basket_names: Sequence[str] = ()
    fallen: bool = False
    harvest_mode: str = "rl_balance"
    harvest_phase: str = "idle"
    harvest_target: Optional[str] = None
    harvest_result: str = ""
    estopped: bool = False
    soc: float = 1.0
    loco_mode: str = TRIPOD_MODE   # /nandi/locomotion_status 'mode' ('' = not seen yet)


@dataclass
class Actions:
    cmd: Optional[Tuple[float, float]] = None   # desired (vx, yaw); None = stand still
    calls: List[str] = field(default_factory=list)  # 'auto_harvest' | 'release_arm' | 'stop_motion'
    force_release: bool = False


# ------------------------------------------------------------------ parsing / acceptance
class Rejection(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason, self.detail = reason, detail or reason


_ISO_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[Tt ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?"
    r"(?:([Zz])|([+-])(\d{2})(?::?(\d{2}))?)?$")


def parse_iso8601(text) -> datetime:
    """ISO 8601 / RFC 3339 date-time -> aware datetime (no offset = UTC).

    Python 3.10's datetime.fromisoformat only takes 3 or 6 fraction digits and '+HH:MM' offsets, so
    the common extended forms are normalised first: any number of fraction digits (truncated to
    microseconds), 'Z', '+HH:MM', '+HHMM', '+HH', ',' as the decimal sign, a space for 'T'.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("not_after missing")
    s = text.strip()
    m = _ISO_RE.match(s)
    if m:
        date, hh, mm, ss, frac, z, sign, oh, om = m.groups()
        s = f"{date}T{hh}:{mm}:{ss or '00'}"
        if frac:
            s += "." + (frac + "000000")[:6]
        if z:
            s += "+00:00"
        elif sign:
            s += f"{sign}{oh}:{om or '00'}"
    dt = datetime.fromisoformat(s)  # date-only and anything else 3.10 understands; ValueError otherwise
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def from_console_start(d: dict, now: Optional[datetime]) -> dict:
    """The web console's stock start command -> the mission contract.

    web_bridge (console_protocol.mission_command) publishes
    {"action": "start", "zone": "<name>", "source": "config/missions/<file>", "mission": {GeoJSON}}:
    mission_id becomes 'console-<zone>-<unix s>', zone the embedded GeoJSON, not_after now + 30 min.
    """
    now = now or datetime.now(timezone.utc)
    name = d.get("zone") if isinstance(d.get("zone"), str) else "zone"
    out = {k: v for k, v in d.items() if k not in ("action", "zone", "source", "mission")}
    out.setdefault("mission_id", f"console-{name}-{int(now.timestamp())}")
    out.setdefault("mode", "harvest_zone")
    out.setdefault("not_after", (now + timedelta(seconds=CONSOLE_TTL_S)).isoformat())
    out["zone"] = d.get("mission")
    return out


def parse_command(raw, now: Optional[datetime] = None) -> MissionCommand:
    """JSON text or dict -> MissionCommand; raises Rejection('command_invalid' | 'zone_invalid')."""
    try:
        d = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (ValueError, TypeError):
        raise Rejection("command_invalid", "command is not JSON")
    if not isinstance(d, dict):
        raise Rejection("command_invalid", "command is not a JSON object")
    if d.get("action") is not None:
        if d.get("action") != "start" or not isinstance(d.get("mission"), dict):
            raise Rejection("command_invalid", f"action {d.get('action')!r}: only 'start' with a 'mission' GeoJSON")
        d = from_console_start(d, now)
    mid = d.get("mission_id")
    if not isinstance(mid, str) or not mid.strip() or len(mid) > 128:
        raise Rejection("command_invalid", "mission_id missing")
    mode = d.get("mode", "harvest_zone")
    if mode not in MODES:
        raise Rejection("command_invalid", f"mode {mode!r} not in {MODES}")
    try:
        not_after = parse_iso8601(d.get("not_after"))
    except ValueError as exc:
        raise Rejection("command_invalid", f"not_after: {exc}")
    if mode in PRESET_MODES:
        return MissionCommand(mid.strip(), mode, None, not_after, None, parse_plan(d.get("plan")))
    zone = d.get("zone")
    if not isinstance(zone, dict):
        raise Rejection("zone_invalid", "zone missing")
    mp = d.get("max_picks")
    if mp is not None:
        # json.loads accepts NaN / Infinity (and 1e400 -> inf): int() of those raises, so test finiteness first
        if (isinstance(mp, bool) or not isinstance(mp, (int, float)) or not math.isfinite(mp) or mp < 1
                or mp != int(mp)):
            raise Rejection("command_invalid", "max_picks must be a positive integer")
        mp = int(mp)
    return MissionCommand(mid.strip(), mode, zone, not_after, mp)


def mission_id_of(raw, now: Optional[datetime] = None) -> str:
    try:
        d = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        if isinstance(d, dict) and d.get("action") == "start" and "mission_id" not in d:
            d = from_console_start(d, now)
        mid = d.get("mission_id") if isinstance(d, dict) else None
        return mid if isinstance(mid, str) else ""
    except (ValueError, TypeError):
        return ""


def zone_inside_geofence(zone_pts: Sequence[gf.Point], fence: Optional[Sequence[gf.Point]]) -> bool:
    """Every zone vertex inside the fence (exact for convex fences; documented limitation)."""
    if not fence:
        return True
    return all(gf.point_in_polygon(x, y, fence) for x, y in zone_pts)


def path_inside_geofence(start: Tuple[float, float], wps: Sequence["Waypoint"],
                         fence: Optional[Sequence[gf.Point]], step: float = FENCE_SAMPLE_M) -> bool:
    """The polyline start -> waypoints stays inside the fence (segments sampled every `step` m, so a
    non-convex fence is handled too)."""
    if not fence:
        return True
    pts = [start] + [(w.x, w.y) for w in wps]
    if not gf.point_in_polygon(pts[0][0], pts[0][1], fence):
        return False
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step)))
        for k in range(1, n + 1):
            f = k / n
            if not gf.point_in_polygon(x0 + f * (x1 - x0), y0 + f * (y1 - y0), fence):
                return False
    return True


# ------------------------------------------------------------------ planning
def group_rows(crops: Dict[str, Tuple[float, float]], gap: float = ROW_GAP_M) -> List[List[str]]:
    """Cluster crop names into rows by y (sorted south -> north), each row sorted west -> east."""
    names = sorted(crops, key=lambda n: crops[n][1])
    rows: List[List[str]] = []
    last_y = None
    for n in names:
        y = crops[n][1]
        if last_y is None or y - last_y > gap:
            rows.append([])
        rows[-1].append(n)
        last_y = y
    return [sorted(r, key=lambda n: crops[n][0]) for r in rows]


def _clear_of(pt: Tuple[float, float], others: Dict[str, Tuple[float, float]], skip: Sequence[str], dist: float) -> bool:
    return all(math.hypot(pt[0] - p[0], pt[1] - p[1]) >= dist for n, p in others.items() if n not in skip)


def stop_pose(crop_x: float, direction: int, keep_out_x: Sequence[Tuple[float, float]] = ()) -> Tuple[float, int]:
    """(stop x, facing direction) for a crop: STANDOFF_X_M behind it in the walking direction.

    keep_out_x are trunk-x bands where the robot would stand pitched (a leg on a soil ridge lowers
    the crop in the arm's frame by 3-5 cm, past the arm's reach). The stop slides within
    STANDOFF_RANGE_M to leave the band; failing that the crop is approached from the far side,
    facing back along the row.
    """
    def blocked(x):
        return any(lo <= x <= hi for lo, hi in keep_out_x)
    lo, hi = STANDOFF_RANGE_M
    offsets = [STANDOFF_X_M] + [o for k in range(1, 21) for o in (STANDOFF_X_M - 0.01 * k, STANDOFF_X_M + 0.01 * k)
                                if lo <= o <= hi]
    for d in (direction, -direction):
        for off in offsets:
            if not blocked(crop_x - d * off):
                return crop_x - d * off, d
    return crop_x - direction * STANDOFF_X_M, direction


def plan_row_stops(row: Sequence[str], crops: Dict[str, Tuple[float, float]], zone: Sequence[gf.Point],
                   all_crops: Dict[str, Tuple[float, float]], direction: int,
                   keep_out_x: Sequence[Tuple[float, float]] = (),
                   fence: Optional[Sequence[gf.Point]] = None) -> Tuple[List[Waypoint], str]:
    """Pick stops for one row walked in `direction` (+1 east, -1 west).

    One straight lane per row, LATERAL_M outside the row's outermost crop on that side, so walking
    between stops never brings a foot near a crop of the row. A side whose stops leave the geofence
    is used only when both do (the mission is then rejected); otherwise the side that keeps more
    stops inside the zone and clear of crops outside the row wins; on a tie, the side farther from
    other crops (then 'north'). Stops facing the walking direction come first in row order; crops that can
    only be picked level from the far side (see stop_pose) follow on the way back.
    Returns (waypoints, side) with side in {'north', 'south'}.
    """
    order = list(row) if direction > 0 else list(reversed(row))
    options = []
    for side, s in (("north", 1.0), ("south", -1.0)):
        lane_y = (max if s > 0 else min)(crops[n][1] for n in row) + s * LATERAL_M
        ahead, back = [], []
        for n in order:
            x, d = stop_pose(crops[n][0], direction, keep_out_x)
            (ahead if d == direction else back).append(
                Waypoint(x, lane_y, "pick", n, 0.0 if d > 0 else math.pi))
        stops = ahead + list(reversed(back))
        inside = sum(gf.point_in_polygon(w.x, w.y, zone) for w in stops)
        clear = sum(_clear_of((w.x, w.y), all_crops, row, CROP_CLEAR_M) for w in stops)
        margin = min([math.hypot(w.x - p[0], w.y - p[1]) for w in stops
                      for n, p in all_crops.items() if n not in row] or [1e9])
        in_fence = all(gf.point_in_polygon(w.x, w.y, fence) for w in stops) if fence else True
        options.append((in_fence, inside + clear, round(margin, 2), s > 0, stops, side))
    options.sort(key=lambda o: o[:4], reverse=True)
    stops, side = options[0][4], options[0][5]
    return stops, side


def plan_mission(mode: str, zone: Sequence[gf.Point], crops: Dict[str, Tuple[float, float]],
                 start: Tuple[float, float], home: Tuple[float, float],
                 keep_out_x: Sequence[Tuple[float, float]] = (),
                 fence: Optional[Sequence[gf.Point]] = None) -> Tuple[List[Waypoint], List[str]]:
    """Waypoints for a mission and the crops it targets (inside the zone, in visiting order).

    harvest_zone / patrol with crops: rows in boustrophedon order starting from the row nearest the
    start; each row is entered through a transit waypoint level with the robot's current x so the robot
    reaches the lane side-on, away from crops. patrol without crops: the zone's vertices.
    """
    targets = {n: p for n, p in crops.items() if gf.point_in_polygon(p[0], p[1], zone)}
    wps: List[Waypoint] = []
    if not targets:
        if mode == "patrol":
            wps = [Waypoint(x, y, "patrol") for x, y in zone]
        return wps, []
    rows = group_rows(targets)
    rows.sort(key=lambda r: abs(sum(targets[n][1] for n in r) / len(r) - start[1]))
    pos = start
    order: List[str] = []
    for row in rows:
        mean_x = sum(targets[n][0] for n in row) / len(row)
        direction = 1 if pos[0] <= mean_x else -1
        stops, _ = plan_row_stops(row, targets, zone, crops, direction, keep_out_x, fence)
        first = stops[0]
        if abs(pos[1] - first.y) > 0.3:  # move to the lane side-on, then walk along it
            wps.append(Waypoint(pos[0], first.y, "transit"))
        kind = "pick" if mode == "harvest_zone" else "patrol"
        wps += [Waypoint(w.x, w.y, kind, w.crop if kind == "pick" else None, w.heading) for w in stops]
        order += [w.crop for w in stops]
        pos = (stops[-1].x, stops[-1].y)
    return wps, order


def return_path(pos: Tuple[float, float], home: Tuple[float, float], home_yaw: float = 0.0) -> List[Waypoint]:
    """Back along the current lane to the home x, then across to home (mirrors the way in)."""
    wps = []
    if abs(pos[1] - home[1]) > 0.3 and abs(pos[0] - home[0]) > 0.3:
        wps.append(Waypoint(home[0], pos[1], "transit"))
    wps.append(Waypoint(home[0], home[1], "home", None, home_yaw))
    return wps


def path_length(start: Tuple[float, float], wps: Sequence[Waypoint]) -> float:
    total, (px, py) = 0.0, start
    for w in wps:
        total += math.hypot(w.x - px, w.y - py)
        px, py = w.x, w.y
    return total


def restandoff(w: Waypoint, crop_xy: Tuple[float, float], standoff: float = STANDOFF_X_M,
               lateral: float = LATERAL_M) -> Waypoint:
    """The pick stop for a crop at crop_xy, keeping w's side of the row and walking direction."""
    cx, cy = crop_xy
    side = 1.0 if w.y >= cy else -1.0
    d = 1.0 if math.cos(w.heading or 0.0) >= 0 else -1.0
    return Waypoint(cx - d * standoff, cy + side * lateral, "pick", w.crop, w.heading)


UNDER_BODY_HALF = (0.375, 0.175)  # = harvest_logic.UNDER_BODY_HALF_X / _Y


def trunk_xy(pose: Tuple[float, float, float], p: Tuple[float, float]) -> Tuple[float, float]:
    x, y, th = pose
    dx, dy = p[0] - x, p[1] - y
    return math.cos(th) * dx + math.sin(th) * dy, -math.sin(th) * dx + math.cos(th) * dy


def predicted_harvest_target(pose: Tuple[float, float, float], crops: Dict[str, Sequence[float]],
                             in_basket_names: Sequence[str]) -> Optional[str]:
    """The crop /nandi/auto_harvest will choose: nearest not in the basket and not under the body."""
    best, best_d = None, REACH_M
    for n, p in crops.items():
        if n in in_basket_names:
            continue
        fx, fy = trunk_xy(pose, p[:2])
        if abs(fx) < UNDER_BODY_HALF[0] and abs(fy) < UNDER_BODY_HALF[1]:
            continue
        d = math.hypot(p[0] - pose[0], p[1] - pose[1])
        if d <= best_d:
            best, best_d = n, d
    return best


UNBLOCK_WINDOW_MARGIN_M = 0.08  # an unblocked stop keeps the target this far inside PICK_WINDOW_X (arrival slack)


def unblock_stop(w: Waypoint, crops: Dict[str, Sequence[float]], in_basket_names: Sequence[str],
                 keep_out_x: Sequence[Tuple[float, float]] = (),
                 fence: Optional[Sequence[gf.Point]] = None) -> Optional[Waypoint]:
    """A stop on w's lane from which auto_harvest would take w.crop rather than a nearer crop.

    Tried in order: the far side of the crop facing back along the row (the blocker is then behind
    the trunk and usually out of reach), then a slide along the row (either way), at most
    BLOCK_SHIFT_M and never so far that the target nears the pick window's edge. A candidate counts
    only if the target is in the pick window, it is the predicted harvest target there, the stop is
    level (not in keep_out_x) and inside the fence. None if no candidate works.
    """
    cx, cy = crops[w.crop][0], crops[w.crop][1]
    d = 1.0 if math.cos(w.heading or 0.0) >= 0 else -1.0
    lo, hi = STANDOFF_RANGE_M
    cands = []
    for k in range(0, 21):  # far side: STANDOFF_X_M, then +-1 cm steps inside STANDOFF_RANGE_M
        for off in ((STANDOFF_X_M,) if k == 0 else (STANDOFF_X_M - 0.01 * k, STANDOFF_X_M + 0.01 * k)):
            if lo <= off <= hi:
                cands.append(Waypoint(cx + d * off, w.y, "pick", w.crop, math.pi if d > 0 else 0.0))
    fx = (cx - w.x) * d  # how far the target is ahead of the current stop
    fx_lo, fx_hi = PICK_WINDOW_X[0] + UNBLOCK_WINDOW_MARGIN_M, PICK_WINDOW_X[1] - UNBLOCK_WINDOW_MARGIN_M
    # sliding the stop forward by s puts the target at fx - s: keep that inside [fx_lo, fx_hi]
    for s in (min(BLOCK_SHIFT_M, fx - fx_lo), -min(BLOCK_SHIFT_M, fx_hi - fx)):
        if abs(s) > 0.01:
            cands.append(Waypoint(w.x + d * s, w.y, "pick", w.crop, w.heading))
    for c in cands:
        pose = (c.x, c.y, c.heading or 0.0)
        if any(a <= c.x <= b for a, b in keep_out_x):
            continue
        if fence and not gf.point_in_polygon(c.x, c.y, fence):
            continue
        if crop_in_pick_window(pose, (cx, cy)) and predicted_harvest_target(pose, crops, in_basket_names) == w.crop:
            return c
    return None


def crop_in_pick_window(pose: Tuple[float, float, float], crop_xy: Tuple[float, float]) -> bool:
    x, y, th = pose
    dx, dy = crop_xy[0] - x, crop_xy[1] - y
    fx = math.cos(th) * dx + math.sin(th) * dy
    fy = -math.sin(th) * dx + math.cos(th) * dy
    return (PICK_WINDOW_X[0] <= fx <= PICK_WINDOW_X[1] and abs(fy) <= PICK_WINDOW_Y
            and math.hypot(dx, dy) <= REACH_M)


# ------------------------------------------------------------------ executor
@dataclass
class MissionConfig:
    origin: gf.SiteOrigin
    home: Tuple[float, float] = (0.0, 0.0)
    home_yaw: float = 0.0
    geofence: Optional[List[gf.Point]] = None
    capacity: BasketCapacity = BasketCapacity()
    energy: EnergyParams = EnergyParams()
    harvest_timeout_s: float = HARVEST_TIMEOUT_S
    keep_out_x: Sequence[Tuple[float, float]] = ()  # x bands (e.g. soil ridges) where pick stops are not placed


LEG_SPAN_X_M = 0.38  # trunk centre to the front/rear foot at full stride (0.26 hip + 0.095 stride + 0.03 foot)


def ridge_keep_out(ridges, terrain: str, margin: float = 0.02) -> List[Tuple[float, float]]:
    """Trunk-x bands in which some foot stands on one of farm_scene.RIDGES ((name, x, height, half width), ...)."""
    if terrain == "flat":
        return []
    return [(x - w - LEG_SPAN_X_M - margin, x + w + LEG_SPAN_X_M + margin) for _name, x, _h, w in ridges]


def config_from_site(origin: gf.SiteOrigin, cfg: dict, geofence: Optional[List[gf.Point]]) -> MissionConfig:
    """MissionConfig from site_sim.yaml contents (see config/missions/site_sim.yaml)."""
    home = cfg.get("home") or {}
    basket = cfg.get("basket") or {}
    return MissionConfig(
        origin=origin,
        home=(float(home.get("x", 0.0)), float(home.get("y", 0.0))),
        home_yaw=float(home.get("yaw", 0.0)),
        geofence=geofence,
        capacity=BasketCapacity(int(basket.get("max_crops", 6)), float(basket.get("max_kg", 1.2))),
        energy=EnergyParams.from_dict(cfg.get("energy")),
    )


def geofence_from_config(safety_cfg: Optional[dict], site_cfg: dict) -> Tuple[Optional[List[gf.Point]], str]:
    """Geofence polygon (local metres) and where it came from.

    config/safety.yaml (read-only; owned by the safety package) wins when it has an enabled
    'geofence' key in one of the forms {polygon: [[x, y], ...]} (the safety supervisor's own),
    {local_polygon: [[x, y], ...]} or {x_min, x_max, y_min, y_max}. Otherwise the site's
    fallback_geofence is used; with neither, no geofence check is made ('none').
    """
    def rect(d):
        return [(float(d["x_min"]), float(d["y_min"])), (float(d["x_max"]), float(d["y_min"])),
                (float(d["x_max"]), float(d["y_max"])), (float(d["x_min"]), float(d["y_max"]))]

    for src, d in (("safety.yaml", (safety_cfg or {}).get("geofence")), ("site_sim.yaml", site_cfg.get("fallback_geofence"))):
        if not isinstance(d, dict) or d.get("enabled", True) is False:
            continue
        try:
            for key in ("polygon", "local_polygon"):
                if key in d:
                    return gf.validate_ring(d[key], closed_required=False), src
            return rect(d), src
        except (KeyError, TypeError, ValueError):
            continue
    return None, "none"


class MissionController:
    """State machine. Call submit() / abort() from events and step(obs) at 10 Hz."""

    def __init__(self, cfg: MissionConfig):
        self.cfg = cfg
        self.state = "idle"
        self.reason = ""
        self.detail = ""
        self.mission_id = ""
        self.mode = ""
        self.not_after: Optional[datetime] = None
        self.max_picks: Optional[int] = None
        self.waypoints: List[Waypoint] = []
        self.targets: List[str] = []
        self.wp_idx = 0
        self.return_wps: List[Waypoint] = []
        self.ret_idx = 0
        self.return_reason = ""
        self.phase = "idle"
        self.phase_t = 0.0
        self.target: Optional[str] = None
        self.attempted = 0
        self.picked = 0
        self.basket_count = 0
        self.attempts: Dict[str, int] = {}
        self.results: Dict[str, str] = {}
        self.baseline_in_basket: Optional[int] = None
        self.harvest_seen_running = False
        self.harvest_reply: Optional[Tuple[bool, str]] = None
        self.last_obs: Optional[Observation] = None
        self.soc = 1.0
        self.crop_source = "truth"
        self.events: List[str] = []
        self.replies: List[dict] = []   # one-off /nandi/mission_status answers (busy rejections)
        self._tried: Optional[str] = None            # crop the current auto_harvest call is expected to take
        self._proxy_tried: Dict[str, set] = {}       # stop crop -> later targets already tried from its stop
        self._replanned = set()
        self._replanned_for_block: Dict[str, set] = {}
        self._basket_before: Tuple[str, ...] = ()
        self._exit_actions = False

    # -------------------------------------------------- events
    def submit(self, raw, obs: Observation) -> Tuple[bool, str, str]:
        """Handle a /nandi/mission/command. Returns (accepted, reason_code, detail)."""
        mid = mission_id_of(raw, obs.utc)
        try:
            if self.state in ACTIVE_STATES:
                raise Rejection("busy", f"mission {self.mission_id} is {self.state}")
            try:
                cmd = parse_command(raw, obs.utc)
                mid = cmd.mission_id
                self._check(cmd, obs)
            except Rejection:
                raise
            except Exception as exc:  # a malformed command must never take the executor down
                raise Rejection("command_invalid", f"{type(exc).__name__}: {exc}")
        except Rejection as rej:
            if rej.reason == "busy" and self.state in ACTIVE_STATES:
                # keep the running mission's status; answer the sender with a one-off status of its own
                self.events.append(f"rejected {mid or '?'}: busy")
                self.replies.append(self.rejection_status(mid, rej.reason, rej.detail))
                return False, rej.reason, rej.detail
            self._reset_counters()
            self.state, self.reason, self.detail, self.mission_id = "rejected", rej.reason, rej.detail, mid
            self.events.append(f"rejected {mid or '?'}: {rej.reason} ({rej.detail})")
            return False, rej.reason, rej.detail
        self._start(cmd, obs)
        return True, "", self.detail

    def _check(self, cmd: MissionCommand, obs: Observation):
        if cmd.mode not in LEGACY_MODES:
            raise Rejection("command_invalid", f"mode {cmd.mode!r} is an orchard preset: it runs through "
                                               "PresetQueue and the preset executor, not the legacy executor")
        if obs.utc >= cmd.not_after:
            raise Rejection("expired", f"not_after {cmd.not_after.isoformat()} is past")
        try:
            zone = gf.zone_to_local(cmd.zone, self.cfg.origin)
        except gf.ZoneError as exc:
            raise Rejection("zone_invalid", str(exc))
        if not zone_inside_geofence(zone, self.cfg.geofence):
            raise Rejection("zone_outside_geofence", "zone leaves the safety geofence")
        if obs.estopped:
            raise Rejection("estopped", "e-stop is active")
        if obs.pose is None or obs.in_basket is None:
            raise Rejection("busy", "no robot state yet (is sim_runner running?)")
        if obs.loco_mode != TRIPOD_MODE:
            # nav_logic steers with yaw and reverse, which the RL policy ignores: tripod gait only
            raise Rejection("locomotion_mode", f"locomotion mode is {obs.loco_mode or 'unknown'!r}, "
                                               f"missions walk with {TRIPOD_MODE!r} only (toggle it back)")
        if self.cfg.geofence and not gf.point_in_polygon(obs.pose[0], obs.pose[1], self.cfg.geofence):
            raise Rejection("zone_outside_geofence", "the robot is outside the safety geofence")
        if self.cfg.capacity.is_full(obs.in_basket) and cmd.mode == "harvest_zone":
            raise Rejection("basket_full", f"basket holds {obs.in_basket}/{self.cfg.capacity.limit}")
        flat = {n: (p[0], p[1]) for n, p in obs.crops.items() if n not in obs.in_basket_names}
        wps, order = plan_mission(cmd.mode, zone, flat, obs.pose[:2], self.cfg.home, self.cfg.keep_out_x,
                                  self.cfg.geofence)
        ret = return_path((wps[-1].x, wps[-1].y) if wps else obs.pose[:2], self.cfg.home)
        if not path_inside_geofence(obs.pose[:2], wps + ret, self.cfg.geofence):
            raise Rejection("zone_outside_geofence", "the planned lanes or the way home leave the safety geofence")
        dist = path_length(obs.pose[:2], wps + ret)
        picks = len(order) if cmd.mode == "harvest_zone" else 0
        need = route_soc_needed(dist, picks, dist / self.cfg.energy.walk_speed_mps + 30.0 * picks, self.cfg.energy)
        if obs.soc < need:
            raise Rejection("battery_low", f"soc {obs.soc:.2f} < route {need - self.cfg.energy.reserve:.2f} + reserve")
        self._plan = (zone, wps, order, dist, need)

    def _reset_counters(self):
        self.waypoints, self.targets, self.wp_idx = [], [], 0
        self.return_wps, self.ret_idx, self.return_reason = [], 0, ""
        self.target, self.attempted, self.picked = None, 0, 0
        self.attempts, self.results = {}, {}
        self.baseline_in_basket = None
        self.harvest_reply = None
        self.phase = "idle"
        self._replanned = set()
        self._replanned_for_block = {}
        self._tried, self._proxy_tried = None, {}

    def _start(self, cmd: MissionCommand, obs: Observation):
        zone, wps, order, dist, need = self._plan
        self._reset_counters()
        self.mission_id, self.mode = cmd.mission_id, cmd.mode
        self.not_after, self.max_picks = cmd.not_after, cmd.max_picks
        self.waypoints, self.targets = wps, order
        self.basket_count = obs.in_basket or 0
        self.state, self.reason = "accepted", ""
        self.detail = f"{len(order)} crops, {len(wps)} waypoints, {dist:.1f} m, needs soc {need:.2f}"
        self.phase, self.phase_t = "accepted", obs.t
        self.events.append(f"accepted {cmd.mission_id}: {self.detail}")

    def abort(self, obs: Optional[Observation] = None) -> Tuple[bool, str]:
        if self.state not in ACTIVE_STATES:
            return False, f"No active mission (state {self.state})"
        self._finish("aborted", "aborted", "operator abort")
        return True, f"Mission {self.mission_id} aborted"

    def external_stop(self, detail: str) -> bool:
        """Someone else stopped the robot (/nandi/stop_motion from the console or the Tk panel, or a
        world reset): end the mission as aborted so it does not walk again on the next tick."""
        if self.state not in ACTIVE_STATES:
            return False
        self._finish("aborted", "aborted", detail)
        return True

    def on_harvest_reply(self, ok: bool, message: str):
        self.harvest_reply = (bool(ok), str(message))

    # -------------------------------------------------- helpers
    def _finish(self, state: str, reason: str, detail: str = ""):
        self.state, self.reason, self.detail = state, reason, detail or reason
        self.phase = "stopped"
        self._exit_actions = True
        self.events.append(f"{state} {self.mission_id}: {reason} {detail}".strip())

    def _set_phase(self, name: str, t: float):
        self.phase, self.phase_t = name, t

    def _begin_return(self, reason: str, obs: Observation):
        self.state, self.return_reason = "returning", reason
        self.return_wps = return_path(obs.pose[:2], self.cfg.home, self.cfg.home_yaw)
        self.ret_idx = 0
        self._set_phase("walking", obs.t)
        self.events.append(f"returning: {reason}")

    def _next_target_wp(self, obs: Observation):
        """Skip pick waypoints whose crop is already in the basket or done."""
        while self.wp_idx < len(self.waypoints):
            w = self.waypoints[self.wp_idx]
            if w.kind == "pick" and (w.crop in obs.in_basket_names or w.crop in self.results):
                if w.crop in obs.in_basket_names and w.crop not in self.results:
                    self.results[w.crop] = "in_basket"
                self.wp_idx += 1
                continue
            return w
        return None

    def _update_basket(self, obs: Observation):
        if obs.in_basket is not None:
            self.basket_count = obs.in_basket

    # -------------------------------------------------- main step
    def step(self, obs: Observation) -> Actions:
        self.last_obs = obs
        self.soc = obs.soc
        self.crop_source = obs.crop_source
        self._update_basket(obs)
        act = Actions()
        if self._exit_actions:
            self._exit_actions = False
            act.force_release = True
            act.calls += ["stop_motion", "release_arm"]
            return act
        if self.state not in ACTIVE_STATES:
            return act

        # safety first: these preempt everything, every tick
        if obs.estopped:
            self._finish("failed", "estopped", "e-stop during mission")
            return self.step(obs)
        if obs.fallen:
            self._finish("failed", "fallen", "robot fell")
            return self.step(obs)
        if obs.loco_mode != TRIPOD_MODE:
            self._finish("failed", "locomotion_mode", f"locomotion switched to {obs.loco_mode or 'unknown'!r}")
            return self.step(obs)
        if obs.pose is None:
            return act
        if self.cfg.geofence and not gf.point_in_polygon(obs.pose[0], obs.pose[1], self.cfg.geofence):
            self._finish("failed", "geofence_breach",
                         f"trunk at ({obs.pose[0]:.2f}, {obs.pose[1]:.2f}) left the safety geofence")
            return self.step(obs)

        if self.state == "accepted":
            self.state = "running"
            self._set_phase("walking", obs.t)
            return act

        if self.state == "running":
            if self.phase not in ("harvesting", "verifying"):
                reason = self._return_trigger(obs)
                if reason:
                    self._begin_return(reason, obs)
                    return act
            return self._run(obs, act)
        return self._returning(obs, act)

    def _return_trigger(self, obs: Observation) -> str:
        if obs.soc < self.cfg.energy.low_soc:
            return "battery_low"
        if self.not_after is not None and obs.utc >= self.not_after:
            return "expired"
        if self.mode == "harvest_zone" and self.cfg.capacity.is_full(self.basket_count):
            return "basket_full"
        if self.max_picks is not None and self.picked >= self.max_picks:
            return "max_picks"
        return ""

    def _run(self, obs: Observation, act: Actions) -> Actions:
        if self.phase == "walking":
            w = self._next_target_wp(obs)
            if w is None:
                reason = "completed" if (self.targets or self.mode == "patrol") else "no_targets"
                self._begin_return(reason, obs)
                return act
            if obs.t - self.phase_t > WAYPOINT_TIMEOUT_S:
                if w.crop:
                    self.results[w.crop] = "skipped:waypoint_timeout"
                self.wp_idx += 1
                self._set_phase("walking", obs.t)
                return act
            self.target = w.crop
            # pick stops are approached from behind; the generic 0.15 m tolerance would stop short
            c = nav.goto(obs.pose, (w.x, w.y), tol=PICK_ARRIVE_TOL_M if w.kind == "pick" else nav.ARRIVE_TOL_M)
            if c.arrived:
                if w.kind == "pick":
                    self._set_phase("aligning", obs.t)
                else:
                    self.wp_idx += 1
                return act
            act.cmd = (c.vx, c.yaw)
            return act

        w = self.waypoints[self.wp_idx]
        if self.phase == "aligning":
            if w.heading is not None:
                c = nav.turn_to(obs.pose[2], w.heading, ALIGN_TOL_RAD)
                if not c.arrived and obs.t - self.phase_t < 15.0:
                    act.cmd = (0.0, c.yaw)
                    return act
            self._set_phase("settling", obs.t)
            return act

        if self.phase == "settling":
            dt = obs.t - self.phase_t
            still = (not obs.walking) and obs.trunk_speed < SETTLE_SPEED_MPS
            if (still and dt >= SETTLE_MIN_S) or dt >= SETTLE_MAX_S:
                crop = obs.crops.get(w.crop)
                if crop is None or not crop_in_pick_window(obs.pose, crop[:2]):
                    # the crop was nudged or the stop is off: re-plan this stop once, else skip it
                    if crop is not None and w.crop not in self._replanned:
                        self._replanned.add(w.crop)
                        self.waypoints[self.wp_idx] = restandoff(w, crop[:2])
                        self._set_phase("walking", obs.t)
                        return act
                    self.results[w.crop] = "skipped:out_of_window"
                    self.wp_idx += 1
                    self._set_phase("walking", obs.t)
                    return act
                if self.mode == "patrol":
                    self.results[w.crop] = "visited"
                    self.wp_idx += 1
                    self._set_phase("walking", obs.t)
                    return act
                # auto_harvest takes the nearest free crop; if that is not ours, a nearer crop is in the way
                nearest = predicted_harvest_target(obs.pose, obs.crops, obs.in_basket_names)
                tried = w.crop
                if nearest is not None and nearest != w.crop:
                    if (nearest in self.targets and nearest not in self.results
                            and self.attempts.get(nearest, 0) < MAX_ATTEMPTS
                            and nearest not in self._proxy_tried.get(w.crop, ())
                            and crop_in_pick_window(obs.pose, obs.crops[nearest][:2])):
                        # a later target of ours, in the pick window: take it now, charged to that crop
                        self._proxy_tried.setdefault(w.crop, set()).add(nearest)
                        tried = nearest
                    elif (nearest not in self._replanned_for_block.get(w.crop, ())
                          and (alt := unblock_stop(w, obs.crops, obs.in_basket_names, self.cfg.keep_out_x,
                                                   self.cfg.geofence)) is not None):
                        self._replanned_for_block.setdefault(w.crop, set()).add(nearest)
                        self.events.append(f"{w.crop}: auto_harvest would take {nearest}; stop moved to "
                                           f"({alt.x:.2f}, {alt.y:.2f}) facing {math.degrees(alt.heading or 0):.0f} deg")
                        self.waypoints[self.wp_idx] = alt
                        self._set_phase("walking", obs.t)
                        return act
                    else:
                        self.results[w.crop] = f"skipped:blocked_by_{nearest}"
                        self.events.append(f"skipped {w.crop}: auto_harvest would take {nearest}")
                        self.wp_idx += 1
                        self._set_phase("walking", obs.t)
                        return act
                self.baseline_in_basket = obs.in_basket
                self._basket_before = tuple(obs.in_basket_names)
                self._tried = tried
                self.attempts[tried] = self.attempts.get(tried, 0) + 1
                self.attempted += 1
                self.harvest_reply = None
                self.harvest_seen_running = False
                act.calls.append("auto_harvest")
                self._set_phase("harvesting", obs.t)
            return act

        if self.phase == "harvesting":
            dt = obs.t - self.phase_t
            if obs.harvest_mode == "ik_harvest":
                self.harvest_seen_running = True
            if self.harvest_reply is not None and not self.harvest_reply[0]:
                return self._attempt_over(obs, act, f"refused:{self.harvest_reply[1]}")
            if dt > self.cfg.harvest_timeout_s:
                act.calls.append("release_arm")
                return self._attempt_over(obs, act, "harvest_timeout")
            done = (self.harvest_seen_running and obs.harvest_mode != "ik_harvest") or \
                   (self.harvest_reply is not None and not self.harvest_seen_running and dt > HARVEST_START_GRACE_S
                    and obs.harvest_mode != "ik_harvest")
            if done:
                self._set_phase("verifying", obs.t)
            return act

        if self.phase == "verifying":
            gained = (obs.in_basket or 0) - (self.baseline_in_basket or 0)
            if gained > 0 or obs.t - self.phase_t >= DROP_SETTLE_S:
                return self._attempt_over(obs, act, obs.harvest_result or "")
            return act
        return act

    def _attempt_over(self, obs: Observation, act: Actions, note: str) -> Actions:
        w = self.waypoints[self.wp_idx]
        gained = max(0, (obs.in_basket or 0) - (self.baseline_in_basket or 0))
        self.picked += gained  # the only place 'picked' moves: game_status in_basket rose
        crop = w.crop
        tried, self._tried = (self._tried or crop), None
        new = [n for n in obs.in_basket_names if n not in self._basket_before]
        for n in new:
            self.results[n] = "picked"
            self.events.append(f"picked {n} (in_basket {obs.in_basket})")
        if crop in obs.in_basket_names:
            self.results[crop] = "picked"
            self.wp_idx += 1
            self._set_phase("walking", obs.t)
        elif tried != crop:
            # a later target was tried from this stop: the attempt is that crop's, this one is still to do
            if tried not in obs.in_basket_names:
                if self.attempts.get(tried, 0) >= MAX_ATTEMPTS:
                    self.results[tried] = f"skipped:{note}" if note else "skipped:not_in_basket"
                    self.events.append(f"skipped {tried}: {note}")
                else:
                    self.events.append(f"attempt {self.attempts.get(tried)} on {tried} (from the {crop} stop) "
                                       f"failed: {note}; it is retried at its own stop")
            self._set_phase("settling", obs.t)
        elif gained > 0:
            # auto_harvest took a different (nearer) crop; this one is still to do, attempt not spent
            self.attempts[crop] = max(0, self.attempts.get(crop, 1) - 1)
            self._set_phase("settling", obs.t)
        elif self.attempts.get(crop, 0) < MAX_ATTEMPTS:
            self.events.append(f"attempt {self.attempts.get(crop)} on {crop} failed: {note}; retrying")
            low = note.lower()
            if ("out of reach" in low or "out of arm reach" in low or "move closer" in low) and crop in obs.crops:
                # retry from a nearer stop instead of repeating the same unreachable pose
                self._replanned.add(crop)
                self.waypoints[self.wp_idx] = restandoff(w, obs.crops[crop][:2], CLOSER_STANDOFF_M, CLOSER_LATERAL_M)
                self._set_phase("walking", obs.t)
            else:
                self._set_phase("settling", obs.t)
        else:
            self.results[crop] = f"skipped:{note}" if note else "skipped:not_in_basket"
            self.events.append(f"skipped {crop}: {note}")
            self.wp_idx += 1
            self._set_phase("walking", obs.t)
        return act

    def _returning(self, obs: Observation, act: Actions) -> Actions:
        if self.ret_idx >= len(self.return_wps):
            self._finish("done", self.return_reason)
            return self.step(obs)
        w = self.return_wps[self.ret_idx]
        if self.phase == "aligning":
            c = nav.turn_to(obs.pose[2], w.heading or 0.0, 0.2)
            if c.arrived or obs.t - self.phase_t > 15.0:
                self.ret_idx += 1
                self._set_phase("walking", obs.t)
            else:
                act.cmd = (0.0, c.yaw)
            return act
        c = nav.goto(obs.pose, (w.x, w.y), tol=PICK_ARRIVE_TOL_M if w.kind == "home" else nav.ARRIVE_TOL_M)
        if c.arrived or obs.t - self.phase_t > WAYPOINT_TIMEOUT_S:
            if w.heading is not None and c.arrived:
                self._set_phase("aligning", obs.t)
            else:
                self.ret_idx += 1
                self._set_phase("walking", obs.t)
            return act
        act.cmd = (c.vx, c.yaw)
        return act

    # -------------------------------------------------- status
    def progress(self) -> float:
        if self.state == "done":
            return 1.0
        if self.state in ("idle", "rejected", "accepted"):
            return 0.0
        n = max(1, len(self.waypoints))
        base = min(self.wp_idx, n) / n * 0.9
        if self.state == "returning":
            base = 0.9 + 0.1 * self.ret_idx / max(1, len(self.return_wps))
        return round(min(1.0, base), 3)

    def status(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "state": self.state,
            "reason_code": self.reason if self.state not in ("running", "returning") else (
                self.return_reason if self.state == "returning" else ""),
            "detail": self.detail,
            "mode": self.mode,
            "phase": self.phase,
            "progress": self.progress(),
            "waypoint_idx": self.wp_idx if self.state != "returning" else len(self.waypoints) + self.ret_idx,
            "waypoints_total": len(self.waypoints) + len(self.return_wps),
            "target_crop": self.target if self.state == "running" else None,
            "attempted": self.attempted,
            "picked": self.picked,
            "basket_count": self.basket_count,
            "basket_capacity": self.cfg.capacity.limit,
            "basket": {"n": self.basket_count, "capacity": self.cfg.capacity.limit},  # console panel form
            "soc": round(self.soc, 4),
            "crop_source": self.crop_source,
            "results": dict(self.results),
        }

    def status_json(self) -> str:
        return json.dumps(self.status())

    def rejection_status(self, mission_id: str, reason: str, detail: str) -> dict:
        """A one-off status answering a command that was rejected without replacing the active
        mission's status (busy): same keys as status(), plus active_mission_id."""
        return {
            "mission_id": mission_id, "state": "rejected", "reason_code": reason, "detail": detail,
            "mode": "", "phase": "idle", "progress": 0.0, "waypoint_idx": 0, "waypoints_total": 0,
            "target_crop": None, "attempted": 0, "picked": 0, "basket_count": self.basket_count,
            "basket_capacity": self.cfg.capacity.limit,
            "basket": {"n": self.basket_count, "capacity": self.cfg.capacity.limit}, "soc": round(self.soc, 4),
            "crop_source": self.crop_source, "results": {}, "active_mission_id": self.mission_id,
        }


# ------------------------------------------------------------------ crop source selection
def choose_crops(truth: Dict[str, Tuple[float, float, float]], detections: Optional[Dict[str, Tuple[float, float, float]]],
                 detections_age_s: Optional[float]) -> Tuple[Dict[str, Tuple[float, float, float]], str]:
    """Fresh (< 1 s) /nandi/crop_detections win; otherwise qpos truth. Returns (crops, source)."""
    if detections and detections_age_s is not None and 0.0 <= detections_age_s < DETECTIONS_FRESH_S:
        return dict(detections), "detections"
    return dict(truth), "truth"


def parse_detections(raw) -> Dict[str, Tuple[float, float, float]]:
    """Lenient reader for /nandi/crop_detections (String JSON, world frame).

    Accepts {"detections": [{"name"|"id": str, "x": .., "y": .., "z"?: ..} | {"name", "pos": [x, y, z]}]}
    or a bare list of the same. Entries without a usable name or position are dropped.
    """
    try:
        d = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (ValueError, TypeError):
        return {}
    items = d.get("detections", []) if isinstance(d, dict) else d
    out = {}
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("id") or it.get("label")
        pos = it.get("pos") or it.get("position") or [it.get("x"), it.get("y"), it.get("z", 0.035)]
        try:
            x, y = float(pos[0]), float(pos[1])
            z = float(pos[2]) if len(pos) > 2 and pos[2] is not None else 0.035
        except (TypeError, ValueError, IndexError):
            continue
        if isinstance(name, str) and math.isfinite(x) and math.isfinite(y):
            out[name] = (x, y, z)
    return out


def parse_estop(raw) -> bool:
    """Lenient reader for /nandi/safety_status: true if any of estop/estopped/e_stop is true or state says so."""
    try:
        d = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (ValueError, TypeError):
        return False
    if not isinstance(d, dict):
        return False
    for k in ("estop", "estopped", "e_stop", "estop_active"):
        if bool(d.get(k)):
            return True
    st = str(d.get("state", "")).lower()
    return st in ("estop", "estopped", "e_stop", "emergency_stop")


# ================================================================== orchard presets (WP27)
# MISSION-PRESETS 0.1 / 0.7 / 10, ORCHARD-MVP-SPEC 0 rule 14. Everything below is pure (no ROS, no MuJoCo); the
# preset registry is controllers.presets, the Ready check controllers.ready_check, the gates controllers.stability.
QUEUE_MAX = 16                    # console queue length (assumption)
TRANSIT_V_MPS = 0.18              # stowed walking between presets (the MP 0.3 stowed cap)
# gait command (normalised vx) that walks 0.18 m/s: measured 0.179 m/s over 4 m on orchard_lite flat (runner,
# 2026-11-18); vx 0.5 (speed_scale 0.5, the OS 1.2 note) walks only 0.140 m/s, vx 0.6 0.165, vx 0.7 0.194.
TRANSIT_CMD_VX = 0.65
TRANSIT_DETOUR = 1.3              # straight line x this (assumption; = ready_check.DETOUR_FACTOR)
STOWED_V_MAX_MPS = 0.18           # MP 0.3: stowed <= 0.18 m/s
SPEED_CAP_MPS = 0.30              # MP 0.3: cap 0.3 m/s
SLOPE_DECLARE_TOL_DEG = 0.5       # a scripted slope may understate the measured terrain by at most this much
# standing foot centres in the trunk frame (measured, runner: orchard_lite, 1.5 s stand, feasibility layout)
FEET_B = ((0.266, -0.244), (0.266, 0.243), (0.006, -0.253), (0.006, 0.253), (-0.254, -0.244), (-0.254, 0.243))
PATH_SAMPLE_M = 0.25
MISSIONS_DIR = paths.config_path("missions")
SEGMENT_KINDS = ("walk", "raise", "dwell", "event")
MISSION_SCHEMA = "nandi.mission/1"


def parse_plan(plan) -> dict:
    """The ``plan`` block of a preset cmd/mission: {source, preset, version?, targets?, tree_ids?, params?}."""
    if not isinstance(plan, dict):
        raise Rejection("command_invalid", "a preset command needs a plan object")
    src = plan.get("source")
    if src not in PLAN_SOURCES:
        raise Rejection("command_invalid", f"plan.source {src!r} not in {PLAN_SOURCES}")
    pid = plan.get("preset")
    if not isinstance(pid, str) or not pid:
        raise Rejection("command_invalid", "plan.preset missing")
    for k, typ in (("targets", dict), ("params", dict), ("tree_ids", list)):
        if k in plan and plan[k] is not None and not isinstance(plan[k], typ):
            raise Rejection("command_invalid", f"plan.{k} must be a {typ.__name__}")
    return dict(plan)


def from_preset_start(msg, registry, field_doc: Optional[dict] = None, source: str = "console",
                      now: Optional[datetime] = None, device: str = "nandi-sim") -> dict:
    """Console ``preset_start{preset, targets{tree_ids | row_ids | block_id}, params}`` (MP 0.7) -> the cmd/mission
    dict (``presets.to_mission_command``). Raises Rejection with the preset reason code (param_invalid,
    preset_unknown, registry_mismatch, not_in_mvp). Targets are always checked against the tree registry:
    ``field_doc`` None refuses a preset with targets ``registry_mismatch``."""
    from atharv_nandi.core import presets as ps
    try:
        d = json.loads(msg) if isinstance(msg, (str, bytes)) else msg
    except (ValueError, TypeError):
        raise Rejection("command_invalid", "preset_start is not JSON")
    if not isinstance(d, dict) or not isinstance(d.get("preset"), str):
        raise Rejection("command_invalid", "preset_start needs a preset id")
    try:
        preset = registry.get(d["preset"])
        return ps.to_mission_command(preset, d.get("targets"), d.get("params"), source=source, now=now,
                                     device=device, field_doc=field_doc)
    except ps.PresetError as exc:
        raise Rejection(exc.reason, exc.detail)


# ------------------------------------------------------------------ queue with ETA, SoC, disk
@dataclass
class QueueItem:
    qid: str
    cmd: dict
    preset: str
    estimate: dict                       # presets.Preset.estimate (+ ready_check energy fields)
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    level: str = "go"                    # Ready level at enqueue time
    reasons: Tuple[str, ...] = ()
    confirmed: bool = False              # operator confirmed a degraded Ready check


class PresetQueue:
    """The console queue (OS 5.5, MP 0.7 ``queue_reorder``): items in order, each with its start and end point.
    ``projection`` walks the queue from the robot's state: transit (straight line x 1.3 at 0.18 m/s) + the preset
    estimate gives the ETA; the WP7 route-energy coefficients give the SoC after each item and the reserve check
    (return energy from the item's end + 15 % of the pack); the disk estimate gives the free disk. An item whose
    projection fails is ``blocked`` with energy_reserve / disk_low, so the console shows it before it starts, and
    ``pop`` (the dispatch) refuses a blocked head item with that reason. ``add`` needs the item's Ready level
    (go / degraded + confirm); ``pop`` can re-run the Ready check (``recheck``) at dispatch. Thread-safe: the status
    timer and the preset_start / queue_reorder callbacks may run on different threads."""

    LEVELS = ("go", "degraded", "no_go")

    def __init__(self, coeffs=None, capacity_wh: Optional[float] = None, max_len: int = QUEUE_MAX):
        from atharv_nandi.core.route_energy import Coefficients
        self.coeffs = coeffs or Coefficients.seed()
        self.capacity_wh = float(capacity_wh or EnergyParams().capacity_wh)
        self.max_len = int(max_len)
        self.items: List[QueueItem] = []
        self._n = 0
        self._lock = threading.RLock()

    def __len__(self):
        with self._lock:
            return len(self.items)

    def add(self, cmd: dict, estimate: dict, start_xy, end_xy, level: Optional[str] = None,
            reasons: Sequence[str] = (), confirmed: bool = False) -> str:
        """Enqueue a preset command with its Ready check result (``level`` is required: no level = not checked)."""
        if level not in self.LEVELS:
            raise Rejection("not_checked", f"Ready check level {level!r}: run the Ready check before queueing")
        if level == "no_go":
            raise Rejection(reasons[0] if reasons else "not_checked", "Ready check is no_go")
        if level == "degraded" and not confirmed:
            raise Rejection("confirm_required", f"Ready check degraded ({list(reasons)}): operator confirm needed")
        with self._lock:
            if len(self.items) >= self.max_len:
                raise Rejection("queue_full", f"queue holds {self.max_len} items")
            mid = cmd.get("mission_id", "")
            taken = {i.cmd.get("mission_id") for i in self.items}
            if mid and mid in taken:
                # suffix until unique: "m" -> "m-<n+1>" could itself be queued already (WPCOV review)
                k = self._n + 1
                while f"{mid}-{k}" in taken:
                    k += 1
                mid = f"{mid}-{k}"
            self._n += 1
            qid = f"q{self._n}"
            preset = (cmd.get("plan") or {}).get("preset", "")
            self.items.append(QueueItem(qid, dict(cmd, mission_id=mid), preset, dict(estimate), tuple(start_xy),
                                        tuple(end_xy), level, tuple(reasons), bool(confirmed)))
            return qid

    def remove(self, qid: str) -> bool:
        with self._lock:
            n = len(self.items)
            self.items = [i for i in self.items if i.qid != qid]
            return len(self.items) < n

    def reorder(self, order: Sequence[str]):
        """``queue_reorder``: ``order`` must be a permutation of the queued ids."""
        with self._lock:
            ids = [i.qid for i in self.items]
            if sorted(order) != sorted(ids) or len(set(order)) != len(order):
                raise Rejection("command_invalid", f"queue_reorder needs a permutation of {ids}")
            by = {i.qid: i for i in self.items}
            self.items = [by[q] for q in order]

    def pop(self, soc: float, disk_free_mb: float, robot_xy, home_xy=(0.0, 0.0),
            recheck=None) -> Optional[QueueItem]:
        """Dispatch the head item (None = empty queue). Refuses (Rejection, the item stays queued) when the
        projection blocks it (energy_reserve / disk_low from the robot's state now) or when ``recheck(item)`` -- the
        Ready check at dispatch, returning a ReadyResult or (level, reasons) -- is no_go, or degraded without the
        operator confirm the item was queued with."""
        with self._lock:
            if not self.items:
                return None
            head = self.items[0]
            row = self._project([head], soc, disk_free_mb, robot_xy, home_xy)["items"][0]
            if row["blocked"]:
                raise Rejection(row["blocked"][0], f"{head.qid} blocked at dispatch: {row['blocked']}")
            if recheck is not None:
                r = recheck(head)
                level, reasons = ((r.level, list(r.reasons)) if hasattr(r, "level") else (r[0], list(r[1])))
                if level not in self.LEVELS or level == "no_go":
                    raise Rejection(reasons[0] if reasons else "not_checked",
                                    f"{head.qid}: Ready check at dispatch is {level}")
                if level == "degraded" and not head.confirmed:
                    raise Rejection("confirm_required", f"{head.qid}: Ready check at dispatch degraded {reasons}")
            return self.items.pop(0)

    def projection(self, soc: float, disk_free_mb: float, robot_xy, home_xy=(0.0, 0.0), t0: float = 0.0) -> dict:
        with self._lock:
            items = list(self.items)
        return self._project(items, soc, disk_free_mb, robot_xy, home_xy, t0)

    def _project(self, items, soc: float, disk_free_mb: float, robot_xy, home_xy=(0.0, 0.0), t0: float = 0.0) -> dict:
        from atharv_nandi.core.route_energy import RoutePlan, reserve_soc
        rows = []
        t, pos, s, disk = float(t0), tuple(robot_xy), float(soc), float(disk_free_mb)
        blocked_at = None
        for it in items:
            transit = math.hypot(it.start_xy[0] - pos[0], it.start_xy[1] - pos[1]) * TRANSIT_DETOUR
            t_start = t + transit / TRANSIT_V_MPS
            t_end = t_start + float(it.estimate.get("t_s", 0.0))
            plan = RoutePlan(route_m=transit + float(it.estimate.get("walk_m", 0.0)),
                             items=int(it.estimate.get("items", 0)), t_work_s=t_end - t,
                             crown_m=float(it.estimate.get("crown_m", 0.0)))
            wh = self.coeffs.energy_wh(plan)
            s_end = s - wh / self.capacity_wh
            ret = math.hypot(home_xy[0] - it.end_xy[0], home_xy[1] - it.end_xy[1]) * TRANSIT_DETOUR
            reserve = reserve_soc(ret, self.coeffs, self.capacity_wh)
            need_disk = 1.5 * float(it.estimate.get("disk_mb", 0.0))
            reasons = []
            if not s_end >= reserve:          # NaN SoC fails closed
                reasons.append("energy_reserve")
            if not disk >= need_disk:
                reasons.append("disk_low")
            row = {"qid": it.qid, "preset": it.preset, "mission_id": it.cmd.get("mission_id"),
                   "eta_start_s": round(t_start, 1), "eta_end_s": round(t_end, 1), "transit_m": round(transit, 2),
                   "energy_wh": round(wh, 2), "soc_end": round(s_end, 4), "reserve_soc": round(reserve, 4),
                   "disk_free_end_mb": round(disk - float(it.estimate.get("disk_mb", 0.0)), 1),
                   "blocked": reasons, "level": it.level}
            if reasons and blocked_at is None:
                blocked_at = it.qid
            rows.append(row)
            t, pos, s, disk = t_end, it.end_xy, s_end, disk - float(it.estimate.get("disk_mb", 0.0))
        return {"items": rows, "eta_s": round(t - t0, 1), "soc_end": round(s, 4), "disk_free_end_mb": round(disk, 1),
                "blocked_at": blocked_at, "coefficients": self.coeffs.source}

    def status(self, soc: float, disk_free_mb: float, robot_xy, home_xy=(0.0, 0.0)) -> dict:
        """The console ``preset_status`` queue block (<= 2 Hz)."""
        with self._lock:
            return dict(self.projection(soc, disk_free_mb, robot_xy, home_xy), n=len(self.items))


# ------------------------------------------------------------------ missions (orchard_tour, recovery_demo)
def load_mission(name_or_path: str) -> dict:
    path = name_or_path if os.path.sep in name_or_path else os.path.join(MISSIONS_DIR, f"{name_or_path}.yaml")
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_mission(doc, registry) -> List[str]:
    """Shape of a nandi.mission/1 file: ids unique, presets known, every segment well formed ([] = valid)."""
    errs = []
    if not isinstance(doc, dict) or doc.get("schema") != MISSION_SCHEMA:
        return [f"mission schema must be {MISSION_SCHEMA}"]
    ids = [s.get("id") for s in doc.get("steps") or []]
    if not ids or len(ids) != len(set(ids)) or not all(isinstance(i, str) and i for i in ids):
        errs.append("mission steps need unique ids")
    for s in doc.get("steps") or []:
        sid = s.get("id", "?")
        if s.get("preset") is not None and s["preset"] not in registry:
            errs.append(f"{sid}: unknown preset {s['preset']!r}")
        if s.get("preset") is None and not isinstance(s.get("t_s"), (int, float)):
            errs.append(f"{sid}: a step without a preset needs t_s")
        for i, g in enumerate(s.get("motion") or []):
            if not isinstance(g, dict) or g.get("kind") not in SEGMENT_KINDS:
                errs.append(f"{sid}[{i}]: segment kind not in {SEGMENT_KINDS}")
                continue
            if g["kind"] == "walk" and (len(g.get("path") or []) < 2 or "v_mps" not in g):
                errs.append(f"{sid}[{i}]: walk needs path (>= 2 points) and v_mps")
            if g["kind"] in ("raise", "dwell") and not isinstance(g.get("at"), list) and not g.get("station"):
                errs.append(f"{sid}[{i}]: {g['kind']} needs at [x, y] or a WP0c station")
            if g.get("station") and wp0c_station(g["station"]) is None:
                errs.append(f"{sid}[{i}]: unknown WP0c station {g['station']!r}")
            if g.get("known_issue") and g["known_issue"] not in [k.get("id") for k in doc.get("known_issues") or []]:
                errs.append(f"{sid}[{i}]: known_issue {g['known_issue']!r} not listed in known_issues")
            if g["kind"] == "raise" and "to_m" not in g:
                errs.append(f"{sid}[{i}]: raise needs to_m")
            if g["kind"] == "dwell" and "dwell_s" not in g:
                errs.append(f"{sid}[{i}]: dwell needs dwell_s")
            if g["kind"] == "event" and not g.get("expect_refused"):
                errs.append(f"{sid}[{i}]: event needs expect_refused (the reason the gate must give)")
            if "slope_deg" not in g and g["kind"] != "event":
                errs.append(f"{sid}[{i}]: declare slope_deg (the scripted local slope)")
    return errs


def footprint_slope(grid, x: float, y: float, heading: float, feet=FEET_B) -> Tuple[float, float]:
    """(cross_deg, pitch_deg) of the least-squares plane through the terrain under the six standing feet (the
    settled body tilt a station or path point gives). cross > 0 = ground rises to the robot's left; pitch > 0 =
    rises ahead."""
    import numpy as np
    c, s = math.cos(heading), math.sin(heading)
    A, z = [], []
    for u, v in feet:
        px, py = x + u * c - v * s, y + u * s + v * c
        A.append((u, v, 1.0))
        z.append(grid.surface_at(px, py))
    (gu, gv, _), *_ = np.linalg.lstsq(np.asarray(A), np.asarray(z), rcond=None)
    return math.degrees(math.atan(gv)), math.degrees(math.atan(gu))


def _samples(path, step=PATH_SAMPLE_M):
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(L / step)))
        hd = math.atan2(y1 - y0, x1 - x0)
        for k in range(n + 1):
            a = k / n
            yield x0 + a * (x1 - x0), y0 + a * (y1 - y0), hd


_STATIONS: Optional[Dict[str, dict]] = None


def wp0c_station(target_id: str) -> Optional[dict]:
    """The WP0c frozen station {xy, heading_deg, ...} of a target id (config/fields/orchard_candidates.json)."""
    global _STATIONS
    if _STATIONS is None:
        path = os.path.join(os.path.dirname(MISSIONS_DIR), "fields", "orchard_candidates.json")
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        _STATIONS = {t["id"]: t["station"] for t in doc.get("frozen", {}).get("targets", []) if t.get("station")}
    return _STATIONS.get(target_id)


def resolve_segment(g: dict) -> dict:
    """A segment with ``station`` gets ``at`` and ``heading_deg`` from the WP0c frozen station."""
    if g.get("station") and "at" not in g:
        s = wp0c_station(g["station"])
        if s is None:
            raise ValueError(f"unknown WP0c station {g['station']!r}")
        g = dict(g, at=list(s["xy"]), heading_deg=float(s["heading_deg"]))
    return g


def segment_slope(grid, g: dict) -> Tuple[float, float]:
    """Worst (|cross|, |pitch|) the terrain gives along a walk path or at a raise / dwell / event point (at the
    station heading when known, else 4 headings: a dwell may turn)."""
    if grid is None or (g["kind"] == "event" and not g.get("at")):
        return 0.0, 0.0
    worst = (0.0, 0.0, 0.0)
    if g["kind"] == "walk":
        pts = list(_samples(g["path"]))
    elif g.get("heading_deg") is not None:
        pts = [(g["at"][0], g["at"][1], math.radians(float(g["heading_deg"])))]
    else:
        pts = [(g["at"][0], g["at"][1], h * math.pi / 4) for h in range(4)]
    for x, y, hd in pts:
        cr, pi = footprint_slope(grid, x, y, hd)
        from atharv_nandi.core.stability import slope_tilt_deg
        tl = slope_tilt_deg(cr, pi)
        if tl > worst[2]:
            worst = (abs(cr), abs(pi), tl)
    return worst[0], worst[1]


def _basins(field_doc) -> List[Tuple[float, float, float]]:
    return [(float(t["geo"]["x"]), float(t["geo"]["y"]), float(t.get("basin_r_m") or 0.0))
            for t in (field_doc or {}).get("trees", []) if t.get("basin_r_m")]


CLEARANCE_MARGIN_M = 0.10         # swept / route clearance margin (OS 2.3 rules 1-2)
NEAR_TREE_M = 1.0                 # a tree's alley clear height applies within canopy_r + this of the path (assumption)
STOWED_LENS_M, CROWN_HEAD_HALF_M = 1.10, 0.06


def clearance_violation(g: dict, field_doc: Optional[dict]) -> Optional[dict]:
    """MP 0.2 / OS 2.3 rule 2: walking under the registry ``alley_clear_height_m`` of every tree the path passes
    (within canopy_r + 1 m) with the head top + 0.10 m below it."""
    if g["kind"] != "walk" or not field_doc:
        return None
    head = STOWED_LENS_M + float(g.get("ext_m", 0.0)) + CROWN_HEAD_HALF_M + CLEARANCE_MARGIN_M
    pts = [(p[0], p[1]) for p in _samples(g["path"])]
    worst = None
    for t in field_doc.get("trees", []):
        x, y = float(t["geo"]["x"]), float(t["geo"]["y"])
        reach = float(t.get("canopy_r_m") or 0.0) + NEAR_TREE_M
        if any(math.hypot(px - x, py - y) <= reach for px, py in pts):
            clear = min(t.get("alley_clear_height_m") or [9.9])
            if clear < head - 1e-9 and (worst is None or clear < worst[1]):
                worst = (t["tree_id"], clear)
    if worst is None:
        return None
    return {"gate": "alley_clearance", "reason": "crown_clearance",
            "detail": f"head + margin {head:.2f} m > alley clear height {worst[1]} m at {worst[0]}"}


def check_segment(g: dict, grid=None, field_doc=None, table=None) -> List[dict]:
    """Every violation of one scripted motion segment against the WP6 gate table, the speed caps and the slope
    rules (ORCHARD-MVP-SPEC 0 rule 14). [] = consistent. An ``event`` segment must be REFUSED with its
    ``expect_refused`` reason (the scripted recovery is consistent only if the gates really trip)."""
    from atharv_nandi.core import stability as st
    table = table or st.CrownGateTable.load()
    g = resolve_segment(g)
    out = []
    ext = float(g.get("ext_m", 0.0))
    declared = float(g.get("slope_deg", 0.0))
    m_cross, m_pitch = segment_slope(grid, g)
    m_tilt = st.slope_tilt_deg(m_cross, m_pitch)
    if g["kind"] != "event" and m_tilt > declared + SLOPE_DECLARE_TOL_DEG:
        out.append({"gate": "terrain", "reason": "slope_understated",
                    "detail": f"scripted {declared} deg, terrain {m_tilt:.2f} deg"})
    cross = max(declared, m_cross)
    pitch = m_pitch if m_tilt > declared else 0.0
    wind = float(g.get("wind_mps", 0.0))
    if g["kind"] == "walk":
        v = float(g["v_mps"])
        if v > SPEED_CAP_MPS + 1e-9:
            out.append({"gate": "speed_cap", "reason": "speed", "detail": f"{v} m/s > cap {SPEED_CAP_MPS}"})
        if ext <= st.CROWN_UP_EXT_M and v > STOWED_V_MAX_MPS + 1e-9:
            out.append({"gate": "speed_stowed", "reason": "speed", "detail": f"{v} m/s > stowed {STOWED_V_MAX_MPS}"})
        for dec in st.check_step(ext, v, cross, pitch, wind_mps=wind, walking=True, table=table):
            out.append({"gate": dec.gate, "reason": dec.reason, "detail": dict(dec.values)})
        dec = st.gate_crown_path(ext, [(p[0], p[1]) for p in _samples(g["path"])], _basins(field_doc))
        if not dec:
            out.append({"gate": dec.gate, "reason": dec.reason, "detail": dict(dec.values)})
        cv = clearance_violation(g, field_doc)
        if cv:
            out.append(cv)
    elif g["kind"] == "raise":
        for dec in st.check_step(ext, 0.0, cross, pitch, wind_mps=wind, walking=False, table=table,
                                 raise_to=float(g["to_m"])):
            out.append({"gate": dec.gate, "reason": dec.reason, "detail": dict(dec.values)})
    elif g["kind"] == "dwell":
        for dec in st.check_step(ext, 0.0, cross, pitch, wind_mps=wind, walking=False, table=table,
                                 dwell_s=float(g["dwell_s"])):
            out.append({"gate": dec.gate, "reason": dec.reason, "detail": dict(dec.values)})
    else:  # event: the injected condition must be refused with the expected reason
        walking = bool(g.get("walking", True))
        decs = st.check_step(ext, float(g.get("v_mps", 0.0)), cross, pitch, wind_mps=wind,
                             walking=walking, table=table, raise_to=g.get("raise_to"),
                             dwell_s=float(g.get("dwell_s", 0.0)))
        reasons = {d.reason for d in decs}
        if g["expect_refused"] not in reasons:
            out.append({"gate": "event", "reason": "not_refused",
                        "detail": f"expected {g['expect_refused']!r}, gates gave {sorted(reasons)}"})
    return out


def tour_gate_violations(doc: dict, field_doc: Optional[dict] = None, grid=None, table=None) -> List[dict]:
    """Rule 14 check of a whole mission: [{step, segment, gate, reason, detail}] ([] = every scripted step agrees
    with the WP6 gate table, the speed caps and the slope rules)."""
    out = []
    for s in doc.get("steps") or []:
        for i, g in enumerate(s.get("motion") or []):
            for v in check_segment(g, grid, field_doc, table):
                out.append(dict(v, step=s["id"], segment=i, known_issue=g.get("known_issue")))
    return out


def preset_gate_violations(preset, table=None) -> List[dict]:
    """Rule 14 for a preset default: every segment of its default profile at the slope it may meet (the segment's
    ``slope_deg``, else the preset's Ready-check slope limit) passes the WP6 gates and the speed caps."""
    out = []
    for i, g in enumerate(preset.gate_profile()):
        seg = dict(g, slope_deg=float(g.get("slope_deg", preset.slope_limit_deg)))
        if seg["kind"] == "walk":
            seg["path"] = [(0.0, 0.0), (1.0, 0.0)]
        else:
            seg["at"] = [0.0, 0.0]
        for v in check_segment(seg, None, None, table):
            out.append(dict(v, preset=preset.id, segment=i))
    return out


def _path_len(path) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def tour_steps(doc: dict, registry, field_doc: Optional[dict], ctx=None) -> List[dict]:
    """Per step: Ready check (tour context + the step's ctx overrides; robot at the end of the previous motion),
    expected level, and the time: preset estimate + transit walks (segments with ``transit: true``) + fixed t_s."""
    from atharv_nandi.core import ready_check as rc
    season = rc.season_table()
    from atharv_nandi.core import stability as st
    table = st.CrownGateTable.load()
    base = dict(doc.get("ctx") or {})
    pos = tuple((field_doc or {}).get("home", {}).get("xy", (0.0, 0.0)))
    rows = []
    mounted = None
    for s in doc.get("steps") or []:
        transit_s = sum(_path_len(g["path"]) / float(g["v_mps"]) for g in s.get("motion") or []
                        if g.get("kind") == "walk" and g.get("transit"))
        transit_m = sum(_path_len(g["path"]) for g in s.get("motion") or []
                        if g.get("kind") == "walk" and g.get("transit"))
        row = {"id": s["id"], "preset": s.get("preset"), "transit_s": round(transit_s, 1),
               "transit_m": round(transit_m, 2), "budget_s": s.get("budget_s"), "expect": s.get("expect") or {}}
        first_walk = next((g for g in s.get("motion") or [] if g.get("kind") == "walk"), None)
        start = tuple(first_walk["path"][0]) if first_walk else pos
        if s.get("preset"):
            kw = dict(base)
            kw.update(s.get("ctx") or {})
            kw.setdefault("robot_xy", start)
            kw.setdefault("tool_mounted", mounted)
            c = ctx(**kw) if ctx else rc.ReadyContext.nominal(**kw)
            r = rc.check(registry.get(s["preset"]), c, s.get("targets"), s.get("params"), field_doc, season, table)
            if r.level != "no_go" and registry.get(s["preset"]).tool_for(r.params):
                mounted = registry.get(s["preset"]).tool_for(r.params)
            row.update(level=r.level, reasons=r.reasons, chips=r.chips, adjust=r.adjust,
                       preset_s=float(r.estimate.get("t_s", 0.0)) if r.level != "no_go" else 0.0,
                       estimate=r.estimate)
            if r.level == "no_go":          # a refused step (10 b) shows the reason for t_s, it does not run
                row["preset_s"] = float(s.get("t_s", 0.0))
        else:
            row.update(level=None, reasons=[], preset_s=float(s.get("t_s", 0.0)), estimate={})
        row["t_s"] = round(row["preset_s"] + transit_s + float(s.get("extra_s", 0.0)), 1)
        rows.append(row)
        for g in s.get("motion") or []:
            if g.get("kind") == "walk":
                pos = tuple(g["path"][-1])
            elif g.get("at") or g.get("station"):
                pos = tuple(resolve_segment(g)["at"])
    return rows


def tour_estimate(doc: dict, registry, field_doc: Optional[dict]) -> dict:
    rows = tour_steps(doc, registry, field_doc)
    total = sum(r["t_s"] for r in rows)
    return {"steps": rows, "total_s": round(total, 1), "budget_s": doc.get("budget_s"), "gate_s": doc.get("gate_s")}
