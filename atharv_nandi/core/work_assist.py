"""WP29 WORK assist: SAMPLE / TOOL TOTE, LEAD-TO-MARKS, tag-based FOLLOW-ME, SPOT SPRAY disabled.

MISSION-PRESETS 7 and ORCHARD-MVP-SPEC 5.6 (personal data, DP-13). Pure Python, no MuJoCo, no ROS: the hook
``atharv_nandi/sim/hooks/follow_tag.py`` simulates the UWB tag, feeds ``Obs`` in at 20 Hz and hands the ``Drive`` out; a driver
task (``driver_task``, SimDriver) or the live relay (``/nandi/work/drive``, change request to WP27) walks it.
Every number marked (assumption) is one; (measured) numbers come from the WP29 probe runs in the WP29 contract notes.

Modes (MISSION-PRESETS 0.1 ``mission_logic.MODES`` names: ``tote_carry``, ``lead_to_marks``, ``follow_me``)

    tote_carry      carry a lidded sample / tool tote (<= 3 kg, not a field crate, OQ-20) to a tree stop, a headland
                    point, the dock or the tag holder; arrive within 0.5 m; wait for ``ack`` (unloaded); optional
                    return. Outputs kg moved, Wh, kg/Wh.
    lead_to_marks   walk ahead of a scout or pruner to every MARKed tree of a job sheet, stop 1.5 m from the trunk
                    (within 0.3 m of the stop point), show the job sheet lines, wait for ``next`` (or ``auto_next_s``).
                    Stops for anyone else (any non-animal return in a WP6 field outside the rear sector, where the
                    follower walks). Outputs MARKs visited, time per job.
    follow_me       follow an adult worker wearing an issued UWB tag (no vision identity, privacy rule; OQ-17) at
                    1.5-3 m, v <= 0.18 m/s (0.3 m/s hard cap). Tag lost: the robot stops after 0.5 s without a reading
                    and latches ``tag_lost`` after 2 s (resume needs the tag back AND an operator ``resume``). Anyone
                    else in a WP6 field (a protective return, or a warning return not within 0.45 m of the tag) ->
                    ``person_hold`` stop. Missing or stale WP6 returns -> ``fields_unavailable`` stop (fail closed).
    spot_spray      refused, reason ``not_in_mvp`` (Phase D, OQ-16). Never runs.

Interlocks common to every mode: crown stowed (ext <= 0.02 m) to start and to move; e-stop / safe hold, a WP6
protective or animal latch, local safety link loss -> stop; the WP6 warning speed cap and protective latch act on
the gait below this layer (sim_driver / locomotion_service), this layer only adds stops.

Personal data (OS 5.6, DP-13): the follow / lead track (tag holder positions) is kept by ``TrackLog`` in memory on the
device only, purged after 7 days at most (assumption; capped at TRACK_MAX_N fixes), never published. Every published
payload (``state``, events, ``uplink_record``) carries no coordinates and no tag id; ``person_hold`` leaves the robot
only as a count per hour.
Tag ids are pseudonymous and rotated per shift (``TagRegistry``); tags are issued only to adult workers.
"""
import hashlib
import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "nandi.work/1"
MODES = ("tote_carry", "lead_to_marks", "follow_me", "spot_spray")
MVP_MODES = ("tote_carry", "lead_to_marks", "follow_me")

# ---------------------------------------------------------------------------------------------- numbers
TOTE_MAX_KG = 3.0            # MP 7 / OQ-20: <= 3 kg; crates (20-25 kg, assumption) are Phase D
TOTE_ARRIVE_M = 0.5          # MP 7 KPI: arrival <= 0.5 m
LEAD_STOP_M = 1.5            # MP 7: stop distance 1.5 m from the tree
LEAD_STOP_TOL_M = 0.3        # MP 7 KPI: stop within 0.3 m of the target
FOLLOW_MIN_M = 1.5           # MP 7: follow distance 1.5-3 m
FOLLOW_MAX_M = 3.0
FOLLOW_SET_M = 2.2           # set point inside the band (assumption)
V_STOWED_MPS = 0.18          # MP 0.3 stowed speed (speed_scale 0.5, measured, probe)
V_CAP_MPS = 0.30             # MP 0.3 hard cap
V_MIN_MPS = 0.05             # slowest commanded walk in a final approach (assumption)
TAG_STALE_S = 0.5            # no reading this long -> stop (the robot waits for the tag)
TAG_LOST_S = 2.0             # MP 7 / WP29 acceptance: tag lost > 2 s -> stop, latched tag_lost
TAG_ASSOC_M = 0.45           # a WARNING-field return within this of the tag position is the tag holder (assumption:
                             # body radius ~0.25 m + tag noise ~0.15 m at 3 m); protective returns are never the holder
TAG_RANGE_MAX_M = 20.0       # a UWB reading outside 0..20 m (or not finite) is no reading (fail closed)
FIELDS_STALE_S = 0.25        # WP6 field returns older than this (or missing) = person detection down -> stop
PERSON_CLEAR_S = 2.0         # person_hold lifts after the fields stayed clear of others this long (assumption)
CROWN_STOWED_M = 0.02        # crown "stowed" for WORK assist (ext_m)
TRACK_RETENTION_S = 7 * 86400.0   # OS 5.6: track log device-only, 7 days at most (assumption)
TRACK_MAX_N = 200000         # in-memory cap: about 5.6 h at the 10 Hz tag rate; older points go first (the 7 days
                             # are an upper bound, not a guarantee to keep)
TRACK_STEP_M = 0.5           # walked-distance decimation of the track summary
LEAD_FOLLOWER_FAR_M = 5.0    # lead mode with a tag: wait while the follower is farther than this (assumption)
HEADLAND_M = 1.5             # headland turn points this far beyond the row ends (assumption)
WAYPOINT_PASS_M = 0.35       # an intermediate waypoint counts as passed inside this radius
# gait calibration (measured, orchard_lite flat, SimDriver, 4 s walks): v = 0.245 m/s per unit vx for vx 0.3..1.0;
# yaw 0.5 -> 23 deg/s in place (0.80 rad/s per unit); coast after release ~0.55 s x v
V_PER_UNIT = 0.245
YAW_PER_UNIT = 0.80
COAST_S = 0.55
YAW_MAX_RPS = 0.40
TURN_IN_PLACE_RAD = 0.5
K_HEADING = 1.2
K_APPROACH = 0.6
K_FOLLOW = 0.5
K_BEARING = 1.0

REASONS = ("not_in_mvp", "payload_full", "tote_missing", "no_job_sheet", "registry_mismatch", "tag_not_issued",
           "tag_not_adult", "tag_lost", "tag_stale", "person_hold", "animal_hold", "protective", "crown_gate",
           "local_link_lost", "estop", "geofence", "follower_far", "unknown_target", "busy", "bad_command",
           "fields_unavailable")
# stops with no hazard in a field (the robot stopped for its own sensing / link): the pure-layer proxy for the MP 7
# "false stops/h" KPI (a person_hold on a mis-associated holder return would also be false; not separable here)
FALSE_STOP_REASONS = ("tag_stale", "tag_lost", "local_link_lost", "fields_unavailable")
BOOL_PARAMS = ("return", "tote_on_deck")
ADULT_PERSON_TYPES = ("worker", "adult", "ladder_picker", "pole_picker", "crouching")   # WP4 person types, not child
COORD_KEYS = ("x", "y", "xy", "pos", "position", "lat", "lon", "lng", "latitude", "longitude", "track", "path",
              "waypoints", "geo", "pose", "points", "tag_xy", "bearing", "bearing_deg", "tag_id")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def tag_ok(r) -> bool:
    """A usable UWB reading: finite time, range in 0..TAG_RANGE_MAX_M, finite bearing."""
    if r is None:
        return False
    try:
        t, rg, b = float(r.t), float(r.range_m), float(r.bearing_rad)
    except (TypeError, ValueError, AttributeError):
        return False
    return math.isfinite(t) and math.isfinite(b) and math.isfinite(rg) and 0.0 <= rg <= TAG_RANGE_MAX_M


def _sane(obs: "Obs") -> "Obs":
    """``obs`` with an unusable tag reading dropped (treated as no reading: the tag rules then stop the robot)."""
    if obs.tag is not None and not tag_ok(obs.tag):
        return replace(obs, tag=None)
    return obs


def _fin(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# ---------------------------------------------------------------------------------------------- data
@dataclass
class Pose:
    x: float
    y: float
    yaw: float


@dataclass
class TagReading:
    """One UWB fix in the robot base frame (x forward, y left): range and bearing from the trunk centre."""
    t: float
    range_m: float
    bearing_rad: float


@dataclass
class FieldHit:
    """A WP6 field return (base frame) as this layer sees it: which field, its class and its sector."""
    x: float
    y: float
    field: str = "warning"          # protective | warning
    cls: str = "obstacle"           # obstacle | animal
    sector: str = "front"           # front | side | rear


@dataclass
class Obs:
    t: float
    pose: Pose
    holding: bool = False                     # sim_runner / driver safe hold (e-stop, hook fault)
    latched: str = ""                         # WP6 field latch reason ("" = none)
    hits: List[FieldHit] = field(default_factory=list)
    tag: Optional[TagReading] = None          # newest reading (None = never)
    crown_ext_m: float = 0.0
    local_link_ok: bool = True
    soc: Optional[float] = None
    geofence: Optional[List[Tuple[float, float]]] = None
    fields_ok: bool = True                    # WP6 field returns present and fresh (<= FIELDS_STALE_S); False = blind


@dataclass
class Drive:
    """What the base must do now. ``v_mps`` forward, ``yaw_rps`` CCW; ``stop`` = release the gait."""
    v_mps: float = 0.0
    yaw_rps: float = 0.0
    stop: bool = True
    reason: str = ""

    def __post_init__(self):
        # fail closed: a non-finite command is a stop, never a walk
        if not self.stop and not (math.isfinite(float(self.v_mps)) and math.isfinite(float(self.yaw_rps))):
            self.v_mps, self.yaw_rps, self.stop, self.reason = 0.0, 0.0, True, "bad_command"

    def gait(self) -> Tuple[float, float, float]:
        """Normalised gait command (as /nandi/cmd_vel, -1..1) from the measured calibration."""
        if self.stop or not (math.isfinite(self.v_mps) and math.isfinite(self.yaw_rps)):
            return 0.0, 0.0, 0.0
        v = _clip(self.v_mps, 0.0, V_CAP_MPS)
        return (_clip(v / V_PER_UNIT, 0.0, 1.0), 0.0, _clip(self.yaw_rps / YAW_PER_UNIT, -1.0, 1.0))

    def as_dict(self) -> dict:
        vx, vy, yaw = self.gait()
        return {"v_mps": round(self.v_mps, 4), "yaw_rps": round(self.yaw_rps, 4), "stop": self.stop,
                "reason": self.reason, "cmd": [round(vx, 4), round(vy, 4), round(yaw, 4)]}


STOP = Drive()


# ---------------------------------------------------------------------------------------------- privacy
def contains_coordinates(obj: Any, path: str = "") -> List[str]:
    """Paths of every coordinate-like key (``COORD_KEYS``) or numeric pair / triple list in ``obj``."""
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if str(k).lower() in COORD_KEYS:
                out.append(p)
            out += contains_coordinates(v, p)
    elif isinstance(obj, (list, tuple)):
        if 2 <= len(obj) <= 3 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
            out.append(path or "[]")
        else:
            for i, v in enumerate(obj):
                out += contains_coordinates(v, f"{path}[{i}]")
    return out


class TagRegistry:
    """Issued UWB / BLE tags (OQ-17 A). Adult designated workers only; pseudonymous ids rotated per shift.

    ``issue(holder, holder_type, adult_attested, shift)``: ``holder`` is the device-local holder reference (in the sim
    the actor id) and never leaves this object; the returned tag id is ``tag-`` + 8 hex of SHA-256(salt, shift,
    holder), so the same worker gets a new id every shift and the id alone does not name anyone."""

    def __init__(self, salt: str = "nandi-sim"):
        self.salt = str(salt)
        self.shift = "s0"
        self._holder: Optional[str] = None
        self.tag_id: Optional[str] = None
        self.issued_t: Optional[float] = None

    def _id(self, holder: str) -> str:
        h = hashlib.sha256(f"{self.salt}|{self.shift}|{holder}".encode()).hexdigest()[:8]
        return f"tag-{h}"

    def issue(self, holder: str, holder_type: Optional[str], adult_attested: bool, shift: Optional[str] = None,
              t: float = 0.0) -> Tuple[bool, str]:
        if holder_type is not None and holder_type not in ADULT_PERSON_TYPES:
            return False, "tag_not_adult"
        if not adult_attested:
            return False, "tag_not_adult"
        if shift is not None:
            self.shift = str(shift)
        self._holder = str(holder)
        self.tag_id = self._id(self._holder)
        self.issued_t = float(t)
        return True, self.tag_id

    def rotate(self, shift: str) -> Optional[str]:
        """New shift: the same holder gets a new pseudonymous id."""
        self.shift = str(shift)
        if self._holder is not None:
            self.tag_id = self._id(self._holder)
        return self.tag_id

    def revoke(self):
        self._holder, self.tag_id, self.issued_t = None, None, None

    @property
    def issued(self) -> bool:
        return self.tag_id is not None

    @property
    def holder(self) -> Optional[str]:
        """Device-local only (the sim hook needs it to find the tag body); never published."""
        return self._holder


class TrackLog:
    """Device-only track of the tag holder (personal data, DP-13): world positions in memory, purged after
    ``retention_s``. ``summary()`` is the only thing that may leave the robot: counts and distances, no positions."""

    def __init__(self, retention_s: float = TRACK_RETENTION_S, max_n: int = TRACK_MAX_N):
        self.retention_s = float(retention_s)
        self._pts: Deque[Tuple[float, float, float]] = deque(maxlen=int(max_n))
        self.dist_m = 0.0
        self._anchor: Optional[Tuple[float, float]] = None

    def append(self, t: float, x: float, y: float):
        # walked distance on a 0.5 m decimation, so the tag noise (about 0.07 m at 2 m) does not add up
        if self._anchor is None:
            self._anchor = (float(x), float(y))
        elif math.hypot(x - self._anchor[0], y - self._anchor[1]) >= TRACK_STEP_M:
            self.dist_m += math.hypot(x - self._anchor[0], y - self._anchor[1])
            self._anchor = (float(x), float(y))
        self._pts.append((float(t), float(x), float(y)))
        self.purge(t)

    def purge(self, now: float):
        while self._pts and now - self._pts[0][0] > self.retention_s:
            self._pts.popleft()

    def __len__(self):
        return len(self._pts)

    def local_points(self) -> List[Tuple[float, float, float]]:
        """For an on-device viewer only."""
        return list(self._pts)

    def clear(self):
        self._pts.clear()
        self.dist_m = 0.0
        self._anchor = None

    def summary(self, now: Optional[float] = None) -> dict:
        if now is not None:
            self.purge(now)
        span = (self._pts[-1][0] - self._pts[0][0]) if len(self._pts) > 1 else 0.0
        return {"samples": len(self._pts), "span_s": round(span, 1), "holder_walk_m": round(self.dist_m, 2)}


class HourlyCounter:
    """``person_hold_count`` per hour of sim time (aggregate only, OS 5.6)."""

    def __init__(self):
        self.buckets: Dict[int, int] = {}

    def add(self, t: float):
        h = int(max(0.0, t) // 3600.0)
        self.buckets[h] = self.buckets.get(h, 0) + 1

    def as_dict(self) -> Dict[str, int]:
        return {f"h{h}": n for h, n in sorted(self.buckets.items())}

    @property
    def total(self) -> int:
        return sum(self.buckets.values())


# ---------------------------------------------------------------------------------------------- field geometry
def _tree_xy(tree: dict) -> Tuple[float, float]:
    g = tree.get("geo") or {}
    return float(g.get("x", tree.get("x", 0.0))), float(g.get("y", tree.get("y", 0.0)))


def tree_index(fld: dict) -> Dict[str, dict]:
    return {str(t["tree_id"]): t for t in (fld.get("trees") or [])}


def _row_trees(fld: dict, row_id: str) -> List[dict]:
    return [t for t in (fld.get("trees") or []) if t.get("row_id") == row_id]


def _row_extent(fld: dict, row_id: str) -> Tuple[float, float, float]:
    """(x_min, x_max, y) of a row's trunks."""
    ts = _row_trees(fld, row_id)
    xs = [_tree_xy(t)[0] for t in ts]
    ys = [_tree_xy(t)[1] for t in ts]
    return min(xs), max(xs), sum(ys) / len(ys)


def lane_side(tree: dict, from_y: Optional[float] = None) -> int:
    """-1 for the -y alley, +1 for +y: the alley side nearer ``from_y`` among ``alley_sides``."""
    sides = [s for s in (tree.get("alley_sides") or ["-y", "+y"])]
    opts = []
    if "all" in sides or "-y" in sides:
        opts.append(-1)
    if "all" in sides or "+y" in sides:
        opts.append(1)
    if not opts:
        opts = [-1]
    if from_y is None or len(opts) == 1:
        return opts[0]
    ty = _tree_xy(tree)[1]
    return min(opts, key=lambda s: abs(ty + s * LEAD_STOP_M - from_y))


def stop_point(tree: dict, side: int, stop_m: float = LEAD_STOP_M) -> Tuple[float, float]:
    """The lead / tote stop: ``stop_m`` from the trunk across the row, in the alley on ``side``."""
    x, y = _tree_xy(tree)
    return x, y + side * stop_m


def point_in_polygon(p: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    x, y = float(p[0]), float(p[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1:
            inside = not inside
    return inside


def _rows_between(fld: dict, y0: float, y1: float, x: float, margin: float = 1.2) -> bool:
    """True if a tree row lies strictly between y0 and y1 at x (a straight line would cross it)."""
    lo, hi = min(y0, y1), max(y0, y1)
    for r in fld.get("rows") or []:
        if not _row_trees(fld, r.get("row_id")):
            continue
        x0, x1, ry = _row_extent(fld, r["row_id"])
        if lo + 0.3 < ry < hi - 0.3 and x0 - margin <= x <= x1 + margin:
            return True
    return False


def route(fld: Optional[dict], start: Sequence[float], goal: Sequence[float]) -> List[Tuple[float, float]]:
    """Waypoints from ``start`` to ``goal`` that never cut through a tree row: when a row lies between them the
    route goes round the nearer headland (``HEADLAND_M`` beyond the row ends of every crossed row)."""
    sx, sy = float(start[0]), float(start[1])
    gx, gy = float(goal[0]), float(goal[1])
    if fld is None or not (_rows_between(fld, sy, gy, sx) or _rows_between(fld, sy, gy, gx)):
        return [(gx, gy)]
    lo, hi = min(sy, gy), max(sy, gy)
    xs = []
    for r in fld.get("rows") or []:
        if _row_trees(fld, r.get("row_id")):
            x0, x1, ry = _row_extent(fld, r["row_id"])
            if lo + 0.3 < ry < hi - 0.3:
                xs += [x0, x1]
    west, east = min(xs) - HEADLAND_M, max(xs) + HEADLAND_M
    cost = {hx: abs(sx - hx) + abs(gx - hx) for hx in (west, east)}
    hx = min(cost, key=cost.get)
    return [(hx, sy), (hx, gy), (gx, gy)]


@dataclass
class Stop:
    tree_id: str
    physical_tag: Optional[str]
    xy: Tuple[float, float]
    jobs: List[dict]
    row_id: str = ""


def job_lines(marks: Iterable[dict]) -> Dict[str, List[dict]]:
    """tree_id -> job sheet lines (kind, reason, clock, side, h_class, tool_hint): MarkRecord fields, no coordinates."""
    out: Dict[str, List[dict]] = {}
    for mk in marks:
        if not isinstance(mk, dict) or "tree_id" not in mk:
            continue
        line = {k: mk.get(k) for k in ("kind", "reason", "clock", "side", "h_class", "tool_hint", "paste_required")
                if mk.get(k) is not None}
        out.setdefault(str(mk["tree_id"]), []).append(line)
    return out


def plan_lead(fld: dict, marks: Iterable[dict], start_xy: Sequence[float],
              stop_m: float = LEAD_STOP_M) -> Tuple[List[Stop], List[dict]]:
    """Order the MARKed trees of a job sheet into stops: rows nearest first, serpentine within a row, each row on
    the alley side nearer the robot. Returns (stops, skipped) where skipped carries ``registry_mismatch``."""
    idx = tree_index(fld)
    lines = job_lines(marks)
    skipped = [{"tree_id": tid, "reason": "registry_mismatch"} for tid in lines if tid not in idx]
    by_row: Dict[str, List[dict]] = {}
    for tid in lines:
        if tid in idx:
            by_row.setdefault(str(idx[tid].get("row_id", "")), []).append(idx[tid])
    stops: List[Stop] = []
    cx, cy = float(start_xy[0]), float(start_xy[1])
    rows = dict(by_row)
    while rows:
        def row_cost(rid):
            ts = rows[rid]
            ys = sum(_tree_xy(t)[1] for t in ts) / len(ts)
            return abs(ys - cy) + 0.1 * min(abs(_tree_xy(t)[0] - cx) for t in ts)
        rid = min(rows, key=row_cost)
        ts = rows.pop(rid)
        side = lane_side(ts[0], cy)
        xs = [_tree_xy(t)[0] for t in ts]
        ascending = abs(min(xs) - cx) <= abs(max(xs) - cx)
        ts.sort(key=lambda t: _tree_xy(t)[0], reverse=not ascending)
        for t in ts:
            p = stop_point(t, side, stop_m)
            stops.append(Stop(str(t["tree_id"]), t.get("physical_tag"), p, lines[str(t["tree_id"])], rid))
            cx, cy = p
    return stops, skipped


# ---------------------------------------------------------------------------------------------- steering
class Navigator:
    """Waypoint follower for the tripod gait: turn in place beyond 0.5 rad, else walk with a heading P-term;
    the last waypoint is a stop (slow-down ``K_APPROACH``, release one coast distance short)."""

    def __init__(self, waypoints: Sequence[Sequence[float]], v_max: float = V_STOWED_MPS, tol_m: float = 0.15):
        self.wps = [(float(p[0]), float(p[1])) for p in waypoints]
        self.k = 0
        self.v_max = float(v_max)
        self.tol_m = float(tol_m)
        self.done = not self.wps

    @property
    def goal(self) -> Optional[Tuple[float, float]]:
        return self.wps[-1] if self.wps else None

    def remaining_m(self, pose: Pose) -> float:
        if self.done or not self.wps:
            return 0.0
        d = math.hypot(self.wps[self.k][0] - pose.x, self.wps[self.k][1] - pose.y)
        for a, b in zip(self.wps[self.k:], self.wps[self.k + 1:]):
            d += math.dist(a, b)
        return d

    def update(self, pose: Pose, v_scale: float = 1.0) -> Drive:
        if self.done:
            return Drive(reason="arrived")
        while True:
            tx, ty = self.wps[self.k]
            dist = math.hypot(tx - pose.x, ty - pose.y)
            last = self.k == len(self.wps) - 1
            if not last and dist <= WAYPOINT_PASS_M:
                self.k += 1
                continue
            break
        v_max = self.v_max * v_scale
        if last:
            coast = COAST_S * min(v_max, max(V_MIN_MPS, K_APPROACH * dist))
            if dist <= max(self.tol_m * 0.5, coast + 0.02):
                self.done = True
                return Drive(reason="arrived")
        ang = _wrap(math.atan2(ty - pose.y, tx - pose.x) - pose.yaw)
        if abs(ang) > TURN_IN_PLACE_RAD:
            return Drive(0.0, math.copysign(YAW_MAX_RPS, ang), False, "turning")
        v = v_max if not last else _clip(K_APPROACH * dist, V_MIN_MPS, v_max)
        v *= max(0.0, math.cos(ang))
        return Drive(max(v, V_MIN_MPS if last else 0.0), _clip(K_HEADING * ang, -YAW_MAX_RPS, YAW_MAX_RPS), False,
                     "walking")


# ---------------------------------------------------------------------------------------------- controller
class WorkAssist:
    """The WORK assist state machine. ``start(mode, params, obs)`` checks the preconditions; ``update(obs)`` returns
    the ``Drive`` every tick; ``command(op)`` takes the console ops; ``state()`` is the published state (no
    coordinates); ``uplink_record()`` the aggregate record (no coordinates, no tag id, no personal fields)."""

    def __init__(self, fld: Optional[dict] = None, tags: Optional[TagRegistry] = None):
        self.field = fld
        self.tags = tags if tags is not None else TagRegistry()
        self.track = TrackLog()
        self.person_holds = HourlyCounter()
        self.events: List[dict] = []
        self._reset_run()

    # ------------------------------------------------------------------ run state
    def _reset_run(self):
        self.mode: Optional[str] = None
        self.phase = "idle"                 # idle | running | at_stop | delivered | paused | stopped | done | aborted
        self.hold_reason = ""               # why the robot is standing still while running ("" = moving)
        self.latched_reason = ""            # tag_lost / protective / estop: needs an operator resume
        self.params: Dict[str, Any] = {}
        self.nav: Optional[Navigator] = None
        self.stops: List[Stop] = []
        self.skipped: List[dict] = []
        self.stop_k = 0
        self.visited: List[dict] = []
        self.t0: Optional[float] = None
        self.t_end: Optional[float] = None
        self.t_stop_arrive: Optional[float] = None
        self.t_leg0: Optional[float] = None
        self.person_since: Optional[float] = None
        self.clear_since: Optional[float] = None
        self.person_hold_active = False
        self.tag_seen_t: Optional[float] = None
        self.tag_world: Optional[Tuple[float, float]] = None
        self.tag_vel = (0.0, 0.0)
        self._tag_t_prev: Optional[float] = None
        self.follow_n = 0
        self.follow_in_band = 0
        self.follow_sum = 0.0
        self.follow_max = 0.0
        self.follow_min = math.inf
        self.follow_lags: List[float] = []
        self.kg_moved = 0.0
        self.soc0: Optional[float] = None
        self.soc_last: Optional[float] = None
        self.delivery: Optional[dict] = None
        self.stops_by_reason: Dict[str, int] = {}
        self.protective_events = 0
        self.last_drive = STOP
        self._last_latched = ""
        self.t_last = 0.0

    def _event(self, t: float, kind: str, **kw):
        ev = {"t": round(float(t), 3), "event": kind, "mode": self.mode, **kw}
        self.events.append(ev)
        return ev

    def _count_stop(self, reason: str):
        self.stops_by_reason[reason] = self.stops_by_reason.get(reason, 0) + 1

    @property
    def active(self) -> bool:
        return self.phase in ("running", "at_stop", "delivered", "paused", "stopped")

    # ------------------------------------------------------------------ preconditions and start
    def preconditions(self, mode: str, params: dict, obs: Obs) -> List[str]:
        """Reasons the mode cannot start now (empty = ready). WP27 ``ready_check`` adds the MP 0.2 common set (RTK,
        SoC reserve, disk, season, ...); these are the WORK-assist specific rows plus the safety basics."""
        why: List[str] = []
        obs = _sane(obs)
        if mode == "spot_spray":
            return ["not_in_mvp"]
        if mode not in MVP_MODES or not isinstance(params, dict):
            return ["bad_command"]
        marks = params.get("marks")
        if marks is not None and not (isinstance(marks, list) and all(isinstance(mk, dict) for mk in marks)):
            return ["bad_command"]            # a job sheet is a list of MarkRecord dicts
        if any(k in params and not isinstance(params[k], bool) for k in BOOL_PARAMS):
            return ["bad_command"]            # "return": "false" must not mean True
        dest = params.get("dest")
        if dest is not None and not isinstance(dest, dict):
            return ["bad_command"]
        if self.active:
            why.append("busy")
        if obs.holding:
            why.append("estop")
        if obs.latched:
            why.append("animal_hold" if obs.latched == "animal" else "protective")
        if not obs.local_link_ok:
            why.append("local_link_lost")
        if obs.crown_ext_m > CROWN_STOWED_M:
            why.append("crown_gate")
        if not obs.fields_ok:
            why.append("fields_unavailable")  # no WP6 returns: nobody could be seen, so nothing may start
        if mode == "tote_carry":
            kg = _fin(params.get("tote_kg"), None)
            if not params.get("tote_on_deck", True) or kg is None:
                why.append("tote_missing")
            elif kg > TOTE_MAX_KG + 1e-9 or kg < 0:
                why.append("payload_full")
            if self._tote_goal(params, obs) is None:
                why.append("unknown_target")
        elif mode == "lead_to_marks":
            marks = params.get("marks") or []
            if not marks:
                why.append("no_job_sheet")
            elif self.field is None:
                why.append("registry_mismatch")
            else:
                stops, _sk = plan_lead(self.field, marks, (obs.pose.x, obs.pose.y), self._stop_m(params))
                if not stops:
                    why.append("registry_mismatch")
        elif mode == "follow_me":
            if not self.tags.issued:
                why.append("tag_not_issued")
            elif obs.tag is None or obs.t - obs.tag.t > TAG_STALE_S:
                why.append("tag_stale")
        if obs.geofence and not point_in_polygon((obs.pose.x, obs.pose.y), obs.geofence):
            why.append("geofence")
        if mode in ("tote_carry", "lead_to_marks") and obs.geofence and not why:
            for p in self._planned_points(mode, params, obs):
                if not point_in_polygon(p, obs.geofence):
                    why.append("geofence")
                    break
        return why

    def _stop_m(self, params: dict) -> float:
        return _clip(_fin(params.get("stop_m"), LEAD_STOP_M), 1.0, 3.0)

    def _planned_points(self, mode: str, params: dict, obs: Obs) -> List[Tuple[float, float]]:
        start = (obs.pose.x, obs.pose.y)
        pts: List[Tuple[float, float]] = []
        if mode == "lead_to_marks":
            stops, _ = plan_lead(self.field, params.get("marks") or [], start, self._stop_m(params))
            cur = start
            for s in stops:
                pts += route(self.field, cur, s.xy)
                cur = s.xy
        else:
            g = self._tote_goal(params, obs)
            if g is not None:
                pts = route(self.field, start, g)
        return pts

    def _tote_goal(self, params: dict, obs: Obs) -> Optional[Tuple[float, float]]:
        dest = params.get("dest") or {}
        if not isinstance(dest, dict):
            return None
        if "tree_id" in dest:
            if self.field is None:
                return None
            tr = tree_index(self.field).get(str(dest["tree_id"]))
            if tr is None:
                return None
            return stop_point(tr, lane_side(tr, obs.pose.y), self._stop_m(params))
        if dest.get("to") == "dock":
            h = ((self.field or {}).get("dock") or (self.field or {}).get("home") or {}).get("xy", [0.0, 0.0])
            return float(h[0]), float(h[1])
        if dest.get("to") == "tag":
            if obs.tag is None or not self.tags.issued:
                return None
            tx, ty = self._tag_world(obs.pose, obs.tag)
            d = math.hypot(tx - obs.pose.x, ty - obs.pose.y)
            if d <= LEAD_STOP_M:
                return obs.pose.x, obs.pose.y
            f = (d - LEAD_STOP_M) / d
            return obs.pose.x + f * (tx - obs.pose.x), obs.pose.y + f * (ty - obs.pose.y)
        if "xy" in dest:
            p = dest["xy"]
            if isinstance(p, (list, tuple)) and len(p) >= 2 and _fin(p[0]) is not None and _fin(p[1]) is not None:
                return float(p[0]), float(p[1])
        return None

    def start(self, mode: str, params: Optional[dict], obs: Obs) -> Tuple[bool, List[str]]:
        obs = _sane(obs)
        params = dict(params) if isinstance(params, dict) else ({} if params is None else params)
        why = self.preconditions(mode, params, obs)
        if why:
            self._event(obs.t, "refused", requested=mode, reasons=why)
            return False, why
        keep_tags = self.tags
        self._reset_run()
        self.tags = keep_tags
        self.mode, self.params, self.phase = mode, params, "running"
        self.t0 = self.t_leg0 = float(obs.t)
        self.soc0 = self.soc_last = obs.soc
        v_max = _clip(_fin(params.get("v_mps"), V_STOWED_MPS), 0.05, V_CAP_MPS)
        self.params["v_mps"] = v_max
        if mode == "lead_to_marks":
            self.stops, self.skipped = plan_lead(self.field, params.get("marks") or [], (obs.pose.x, obs.pose.y),
                                                 self._stop_m(params))
            self._leg_to(0, obs)
        elif mode == "tote_carry":
            g = self._tote_goal(params, obs)
            self.params["_origin"] = (obs.pose.x, obs.pose.y)
            self.nav = Navigator(route(self.field, (obs.pose.x, obs.pose.y), g), v_max, tol_m=0.2)
            self.delivery = {"kg": round(float(params["tote_kg"]), 3), "dest": self._dest_label(params)}
        elif mode == "follow_me":
            self.tag_seen_t = obs.tag.t if obs.tag else None
        self._event(obs.t, "started", stops=len(self.stops), skipped=len(self.skipped))
        return True, []

    def _dest_label(self, params: dict) -> str:
        d = params.get("dest") or {}
        if "tree_id" in d:
            return f"tree:{d['tree_id']}"
        if d.get("to") in ("dock", "tag"):
            return str(d["to"])
        return "point"

    def _leg_to(self, k: int, obs: Obs):
        self.stop_k = k
        if k >= len(self.stops):
            self.nav = None
            return
        self.nav = Navigator(route(self.field, (obs.pose.x, obs.pose.y), self.stops[k].xy),
                             self.params.get("v_mps", V_STOWED_MPS), tol_m=0.15)
        self.t_leg0 = float(obs.t)

    # ------------------------------------------------------------------ operator commands
    def command(self, op: str, obs: Obs, **kw) -> Tuple[bool, str]:
        obs = _sane(obs)
        t = obs.t
        if op == "spray":
            self._event(t, "refused", requested="spot_spray", reasons=["not_in_mvp"])
            return False, "not_in_mvp"
        if op in ("abort", "halt"):
            if self.mode is None:
                return False, "not running"
            self._finish(t, "aborted")
            return True, "aborted"
        if not self.active:
            return False, "not running"
        if op == "pause":
            self.phase = "paused"
            self._event(t, "paused")
            return True, "paused"
        if op == "resume":
            if obs.holding or obs.latched:
                return False, "refused: safety latch"
            if not obs.fields_ok:
                return False, "refused: fields_unavailable"
            if self.latched_reason == "tag_lost" and (obs.tag is None or t - obs.tag.t > TAG_STALE_S):
                return False, "refused: tag not seen"
            self.latched_reason = ""
            if self.phase in ("paused", "stopped"):
                self.phase = "at_stop" if self.t_stop_arrive is not None else (
                    "delivered" if self.delivery and self.delivery.get("t_arrive") is not None
                    and not self.delivery.get("acked") else "running")
            self._event(t, "resumed")
            return True, "resumed"
        if op in ("next", "ack"):
            if self.mode == "lead_to_marks" and self.phase == "at_stop":
                self._leave_stop(t, obs, "operator")
                return True, "next"
            if self.mode == "tote_carry" and self.phase == "delivered":
                self._tote_acked(t, obs)
                return True, "acked"
            return False, "nothing to acknowledge"
        return False, f"unknown op {op!r}"

    def _leave_stop(self, t: float, obs: Obs, by: str):
        s = self.stops[self.stop_k]
        self.visited[-1]["dwell_s"] = round(t - (self.t_stop_arrive or t), 2)
        self.visited[-1]["released_by"] = by
        self.t_stop_arrive = None
        self.phase = "running"
        self._event(t, "mark_done", tree_id=s.tree_id, by=by)
        self._leg_to(self.stop_k + 1, obs)
        if self.nav is None:
            self._finish(t, "done")

    def _tote_acked(self, t: float, obs: Obs):
        self.delivery["acked"] = True
        self.kg_moved += float(self.delivery["kg"])
        self._event(t, "tote_unloaded", kg=self.delivery["kg"])
        if self.params.get("return"):
            o = self.params["_origin"]
            self.nav = Navigator(route(self.field, (obs.pose.x, obs.pose.y), o), self.params["v_mps"], tol_m=0.2)
            self.phase = "running"
            self.delivery["returning"] = True
        else:
            self._finish(t, "done")

    def _finish(self, t: float, how: str):
        self.phase = how
        self.t_end = float(t)
        self.nav = None
        self.hold_reason = ""
        self._event(t, how)

    # ------------------------------------------------------------------ tick
    def _tag_world(self, pose: Pose, r: TagReading) -> Tuple[float, float]:
        a = pose.yaw + r.bearing_rad
        return pose.x + r.range_m * math.cos(a), pose.y + r.range_m * math.sin(a)

    def _others(self, obs: Obs) -> Tuple[bool, bool]:
        """(someone other than the tag holder / follower is in a field, an animal is) from the WP6 hits."""
        tag_b = None
        if obs.tag is not None and obs.t - obs.tag.t <= TAG_STALE_S:
            tag_b = (obs.tag.range_m * math.cos(obs.tag.bearing_rad), obs.tag.range_m * math.sin(obs.tag.bearing_rad))
        person = animal = False
        for h in obs.hits:
            if h.cls == "animal":
                animal = True
                continue
            if tag_b is not None and h.field == "warning" and \
                    math.hypot(h.x - tag_b[0], h.y - tag_b[1]) <= TAG_ASSOC_M:
                continue
            if self.mode == "lead_to_marks" and h.sector == "rear" and h.field == "warning":
                continue                      # the scout / pruner following the robot
            if self.mode == "tote_carry" and h.field == "warning":
                continue                      # tote: common interlocks only (the WP6 30 % speed cap below)
            # (the holder exemption above is for the warning field only: a protective-field return is never the
            # holder, so someone at the holder's side who is close enough to matter still stops the robot)
            person = True
        return person, animal

    def update(self, obs: Obs) -> Drive:
        obs = _sane(obs)
        t = float(obs.t)
        self.t_last = t
        self.track.purge(t)
        if obs.soc is not None:
            self.soc_last = obs.soc
        if obs.latched and obs.latched != self._last_latched and self.active:
            self.protective_events += 1
            self._event(t, "safety_latch", reason=obs.latched)
        self._last_latched = obs.latched
        # tag bookkeeping (any mode: tote "to tag" and lead "follower far" use it too)
        if obs.tag is not None and (self.tag_seen_t is None or obs.tag.t > self.tag_seen_t):
            self.tag_seen_t = obs.tag.t
            w = self._tag_world(obs.pose, obs.tag)
            if self.tag_world is not None and self.mode is not None:
                dt = max(1e-3, obs.tag.t - (self._tag_t_prev if self._tag_t_prev is not None else obs.tag.t - 0.1))
                vx, vy = (w[0] - self.tag_world[0]) / dt, (w[1] - self.tag_world[1]) / dt
                a = min(1.0, dt / 1.0)
                self.tag_vel = (self.tag_vel[0] + a * (vx - self.tag_vel[0]),
                                self.tag_vel[1] + a * (vy - self.tag_vel[1]))
            self._tag_t_prev = obs.tag.t
            self.tag_world = w
            if self.active and self.mode in ("follow_me", "lead_to_marks"):
                self.track.append(t, w[0], w[1])
        if not self.active:
            self.last_drive = STOP
            return STOP
        drive = self._tick(obs)
        if drive.stop and drive.reason and drive.reason != self.hold_reason and drive.reason not in (
                "arrived", "at_stop", "delivered", "paused", "done", "too_close", "at_gap", "turning", "walking"):
            self._count_stop(drive.reason)
            self._event(t, "hold", reason=drive.reason)
        self.hold_reason = drive.reason if drive.stop else ""
        self.last_drive = drive
        return drive

    def _safety_stop(self, obs: Obs) -> Optional[str]:
        if obs.holding:
            return "estop"
        if obs.latched:
            return "animal_hold" if obs.latched == "animal" else "protective"
        if not obs.local_link_ok:
            return "local_link_lost"
        if obs.crown_ext_m > CROWN_STOWED_M:
            return "crown_gate"
        if not obs.fields_ok:
            return "fields_unavailable"
        return None

    def _tick(self, obs: Obs) -> Drive:
        t = obs.t
        s = self._safety_stop(obs)
        if s is not None:
            if s in ("estop", "protective", "animal_hold"):
                self.latched_reason = self.latched_reason or s
            return Drive(reason=s)
        if self.latched_reason and self.latched_reason not in ("tag_lost",):
            # a safety latch cleared below this layer: stay stopped until the operator resumes WORK
            return Drive(reason=self.latched_reason)
        if self.phase == "paused":
            return Drive(reason="paused")
        if self.mode == "follow_me":
            # the tag first: a lost tag latches after TAG_LOST_S even while someone else holds the robot (the tag
            # holder without a fresh reading is "someone else" to the field check below)
            g = self._tag_gate(obs)
            if g is not None:
                return g
        # others in the fields: person_hold (lifts after PERSON_CLEAR_S clear)
        person, animal = self._others(obs)
        if animal:
            self.person_hold_active, self.clear_since = True, None
            return Drive(reason="animal_hold")
        if self.mode in ("lead_to_marks", "follow_me"):
            if person:
                if not self.person_hold_active:
                    self.person_hold_active = True
                    self.person_holds.add(t)
                self.clear_since = None
                return Drive(reason="person_hold")
            if self.person_hold_active:
                if self.clear_since is None:
                    self.clear_since = t
                if t - self.clear_since < PERSON_CLEAR_S:
                    return Drive(reason="person_hold")
                self.person_hold_active = False
                self._event(t, "person_clear")
        if self.mode == "follow_me":
            return self._tick_follow(obs)
        if self.mode == "lead_to_marks":
            return self._tick_lead(obs)
        return self._tick_tote(obs)

    # ------------------------------------------------------------------ modes
    def _tag_gate(self, obs: Obs) -> Optional[Drive]:
        """Follow-me tag rule: no reading for TAG_STALE_S -> stop (wait); for more than TAG_LOST_S -> latched
        ``tag_lost`` (resume needs the tag back and an operator ``resume``)."""
        t = obs.t
        if self.latched_reason == "tag_lost":
            return Drive(reason="tag_lost")
        age = math.inf if self.tag_seen_t is None else t - self.tag_seen_t
        if age > TAG_LOST_S:
            self.latched_reason = "tag_lost"
            self.phase = "stopped"
            self._event(t, "tag_lost", age_s=round(min(age, 1e6), 2))
            return Drive(reason="tag_lost")
        if age > TAG_STALE_S or obs.tag is None:
            return Drive(reason="tag_stale")
        return None

    def _tick_follow(self, obs: Obs) -> Drive:
        t = obs.t
        g = self._tag_gate(obs)
        if g is not None:
            return g
        r, b = obs.tag.range_m, obs.tag.bearing_rad
        self.follow_n += 1
        self.follow_sum += r
        self.follow_max = max(self.follow_max, r)
        self.follow_min = min(self.follow_min, r)
        if FOLLOW_MIN_M - 1e-9 <= r <= FOLLOW_MAX_M + 1e-9:
            self.follow_in_band += 1
        sp = math.hypot(*self.tag_vel)
        if sp > 0.05:
            # time headway: how long the holder took to walk the current gap (the robot is this far behind in time)
            self.follow_lags.append(r / sp)
        if obs.geofence and self.tag_world is not None and not point_in_polygon(self.tag_world, obs.geofence):
            return Drive(reason="geofence")
        if obs.geofence:
            ahead = (obs.pose.x + 0.6 * math.cos(obs.pose.yaw), obs.pose.y + 0.6 * math.sin(obs.pose.yaw))
            if not point_in_polygon(ahead, obs.geofence):
                return Drive(reason="geofence")
        v_max = self.params.get("v_mps", V_STOWED_MPS)
        if abs(b) > TURN_IN_PLACE_RAD:
            return Drive(0.0, math.copysign(YAW_MAX_RPS, b), False, "turning")
        if r <= FOLLOW_MIN_M:
            return Drive(reason="too_close")
        # feed-forward: the tag holder's speed along the robot heading, plus a P-term on the range error
        ff = self.tag_vel[0] * math.cos(obs.pose.yaw) + self.tag_vel[1] * math.sin(obs.pose.yaw)
        v = _clip(max(0.0, ff) + K_FOLLOW * (r - FOLLOW_SET_M), 0.0, v_max)
        if v < 0.02:
            return Drive(reason="at_gap")
        return Drive(max(v, V_MIN_MPS), _clip(K_BEARING * b, -YAW_MAX_RPS, YAW_MAX_RPS), False, "walking")

    def _tick_lead(self, obs: Obs) -> Drive:
        t = obs.t
        if self.phase == "at_stop":
            auto = _fin(self.params.get("auto_next_s"), None)
            if auto is not None and t - (self.t_stop_arrive or t) >= auto:
                self._leave_stop(t, obs, "auto")
                if self.phase != "running":
                    return Drive(reason="done")
            else:
                return Drive(reason="at_stop")
        if self.nav is None:
            self._finish(t, "done")
            return Drive(reason="done")
        if self.tags.issued and obs.tag is not None and t - obs.tag.t <= TAG_STALE_S \
                and obs.tag.range_m > _fin(self.params.get("follower_far_m"), LEAD_FOLLOWER_FAR_M):
            return Drive(reason="follower_far")
        d = self.nav.update(obs.pose)
        if self.nav.done:
            s = self.stops[self.stop_k]
            err = math.hypot(obs.pose.x - s.xy[0], obs.pose.y - s.xy[1])
            self.phase = "at_stop"
            self.t_stop_arrive = t
            rec = {"tree_id": s.tree_id, "physical_tag": s.physical_tag, "jobs": s.jobs,
                   "leg_s": round(t - (self.t_leg0 or t), 2), "release_err_m": round(err, 3)}
            self.visited.append(rec)
            self._event(t, "mark_reached", tree_id=s.tree_id, physical_tag=s.physical_tag, jobs=len(s.jobs),
                        release_err_m=round(err, 3))
            return Drive(reason="at_stop")
        return d

    def _tick_tote(self, obs: Obs) -> Drive:
        t = obs.t
        if self.phase == "delivered":
            return Drive(reason="delivered")
        if self.nav is None:
            self._finish(t, "done")
            return Drive(reason="done")
        d = self.nav.update(obs.pose)
        if self.nav.done:
            g = self.nav.goal
            err = math.hypot(obs.pose.x - g[0], obs.pose.y - g[1])
            if self.delivery.get("returning"):
                self.delivery["return_err_m"] = round(err, 3)
                self._finish(t, "done")
                return Drive(reason="done")
            self.phase = "delivered"
            self.delivery["t_arrive"] = round(t - (self.t0 or t), 2)
            self.delivery["release_err_m"] = round(err, 3)
            self._event(t, "tote_arrived", dest=self.delivery["dest"], release_err_m=round(err, 3))
            return Drive(reason="delivered")
        return d

    # ------------------------------------------------------------------ measured arrival (after the gait settled)
    def settle_error(self, pose: Pose) -> Optional[float]:
        """Distance of the settled robot from the current target (lead stop or tote goal); recorded on the visit."""
        g = None
        if self.mode == "lead_to_marks" and self.phase == "at_stop" and self.visited:
            g = self.stops[self.stop_k].xy
            rec = self.visited[-1]
        elif self.mode == "tote_carry" and self.phase == "delivered":
            g = self.nav.goal if self.nav else None
            rec = self.delivery
        if g is None:
            return None
        err = round(math.hypot(pose.x - g[0], pose.y - g[1]), 3)
        rec["stop_err_m"] = err
        return err

    # ------------------------------------------------------------------ outputs
    def tag_status(self, t: float) -> dict:
        age = None if self.tag_seen_t is None else round(t - self.tag_seen_t, 2)
        return {"issued": self.tags.issued, "seen": age is not None and age <= TAG_STALE_S,
                "age_s": age, "lost": self.latched_reason == "tag_lost"}

    def kpis(self) -> dict:
        t_stop = self.t_end if self.t_end is not None else self.t_last
        t_run = (t_stop - self.t0) if self.t0 is not None else 0.0
        k: Dict[str, Any] = {"run_s": round(max(0.0, t_run), 2)}
        if self.mode == "lead_to_marks":
            n = len(self.stops)
            errs = [v.get("stop_err_m") for v in self.visited if v.get("stop_err_m") is not None]
            k.update({"marks_total": n + len(self.skipped), "marks_reached": len(self.visited),
                      "reached_pct": round(100.0 * len(self.visited) / max(1, n + len(self.skipped)), 1),
                      "stop_err_max_m": round(max(errs), 3) if errs else None,
                      "time_per_job_s": round(sum(v["leg_s"] + v.get("dwell_s", 0.0) for v in self.visited)
                                              / max(1, len(self.visited)), 2),
                      "skipped": [dict(s) for s in self.skipped]})
        elif self.mode == "tote_carry":
            wh = self.wh_used()
            k.update({"kg_moved": round(self.kg_moved, 3), "wh": None if wh is None else round(wh, 3),
                      "kg_per_wh": round(self.kg_moved / wh, 4) if wh else None,
                      "arrival_err_m": (self.delivery or {}).get("stop_err_m")})
        elif self.mode == "follow_me":
            n = max(1, self.follow_n)
            hours = max(t_run, 1e-9) / 3600.0
            lags = sorted(self.follow_lags)
            k.update({"in_band_pct": round(100.0 * self.follow_in_band / n, 1),
                      "range_mean_m": round(self.follow_sum / n, 3) if self.follow_n else None,
                      "range_min_m": None if not self.follow_n else round(self.follow_min, 3),
                      "range_max_m": None if not self.follow_n else round(self.follow_max, 3),
                      "follow_lag_s": round(lags[len(lags) // 2], 2) if lags else None,
                      "stops_per_h": round(sum(self.stops_by_reason.values()) / hours, 1),
                      "false_stops_per_h": round(sum(n for r, n in self.stops_by_reason.items()
                                                     if r in FALSE_STOP_REASONS) / hours, 1),
                      "protective_violations": self.protective_events})
        k["stops_by_reason"] = dict(self.stops_by_reason)
        k["person_hold_count"] = self.person_holds.as_dict()
        return k

    def wh_used(self) -> Optional[float]:
        if self.soc0 is None or self.soc_last is None:
            return None
        from atharv_nandi.core.energy_model import EnergyParams
        return max(0.0, (self.soc0 - self.soc_last) * EnergyParams().capacity_wh)

    def state(self, t: float) -> dict:
        """The published state (``/nandi/work/state``): no coordinates, no tag id."""
        st = {"schema": SCHEMA, "t": round(float(t), 3), "mode": self.mode, "phase": self.phase,
              "hold": self.hold_reason, "latched": self.latched_reason, "tag": self.tag_status(t),
              "drive": {"v_mps": round(self.last_drive.v_mps, 3), "stop": self.last_drive.stop},
              "spray": {"enabled": False, "reason": "not_in_mvp"}, "kpi": self.kpis()}
        if self.mode == "lead_to_marks":
            cur = self.stops[self.stop_k] if self.stop_k < len(self.stops) else None
            st["lead"] = {"stop": self.stop_k, "stops": len(self.stops),
                          "tree_id": cur.tree_id if cur else None, "physical_tag": cur.physical_tag if cur else None,
                          "jobs": cur.jobs if (cur and self.phase == "at_stop") else []}
        if self.mode == "tote_carry" and self.delivery:
            st["tote"] = {k: v for k, v in self.delivery.items()}
        return st

    def uplink_record(self, run_id: str = "sim", device_id: str = "nandi-sim") -> dict:
        """Aggregate WORK record for the uplink (kis/v1 ``tele/mission``): no coordinates, no tag id, no track."""
        return {"schema": SCHEMA, "run_id": run_id, "device_id": device_id, "tenant": "sim-nandi",
                "data_provenance": "simulated", "mode": self.mode, "outcome": self.phase,
                "kpi": self.kpis(), "personal_data": "device_only"}


# ---------------------------------------------------------------------------------------------- SimDriver binding
def driver_task(drv) -> None:
    """``SimDriver.add_task('work_assist_drive', driver_task, 50)``: walks the hook's ``bus.values['work_drive']``
    (the in-process twin of the live ``/nandi/work/drive`` -> ``/nandi/cmd_vel`` relay). A stale drive (> 0.5 s, the
    cmd_vel dead-man) or a stop releases the gait."""
    wd = drv.bus.values.get("work_drive")
    if not isinstance(wd, dict):
        return
    if wd.get("stop", True) or drv.t - float(wd.get("t", -1e9)) > 0.5:
        if getattr(drv, "_work_walking", False):
            drv.gait.release()
            drv._work_walking = False
        return
    vx, vy, yaw = wd["cmd"]
    drv.gait.set_command(vx, vy, yaw, hold=True)
    drv._work_walking = True
