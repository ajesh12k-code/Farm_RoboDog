"""WP28 recovery behaviours (ORCHARD-MVP-SPEC 5.3; MISSION-PRESETS 8; WP28).

Pure Python (no ROS, no MuJoCo) except the thin ``RecoveryRunner`` glue at the bottom, which applies the motion
requests of ``atharv_nandi/sim/hooks/recovery.py`` to a WP0b ``SimDriver`` gait. Every clock is injected (sim seconds).

Behaviours (one active at a time, ``RecoveryManager`` picks by priority)::

    FALLEN        fall latch (tilt > 35 deg for 0.3 s, or the runner hold reason ``fall``): power off (hold), motion
                  locked, the manual checklist (crown lock, lift points, two people, laterals and tools, mast
                  inspected, person clear). Never ends by itself; self-righting is a spike only (tools/)
    ESTOP_RECOVER after an e-stop hold: local confirm -> operator confirm (in that order, both after the hold
                  started, the local one only with the local link up) -> checklist (tools safe, fields clear,
                  stability) -> re-arm (``stand_up``: the runner leaves its hold, motion still locked) -> crown
                  retract (the WP24 arbiter refuses every request during the hold, so the retract follows the
                  re-arm) -> stand -> ready, resume from the last completed tree, never mid-cut
    LOCAL_LINK    local safety link lost (WP7 ``link.local.state == lost``, heartbeat > 0.5 s): stop (gait stop,
                  motion locked) -> tools safe -> crown retract -> sit in place -> hold; no autonomous motion
                  (hardware e-stop requirement). Restored: "Link was lost: Resume / Halt" (operator ack), still locked until then; Halt
                  stays parked and locked until a later Resume (also on CLOUD_LINK and the SAFE_PARK hold)
    CLOUD_LINK    cloud console lost, local up (heartbeat > 5 s): queue paused, 0 new targets, SAFE PARK (alley
                  edge <= 3 m) -> hold standing; restored or a local supervisor ack -> resume
    RETURN_HOME   low SoC (WP7 ``energy_action == return_and_wait``) or operator: finish / abandon the current item
                  safely -> crown stow -> route home along the alleys: RTK FIXED or FLOAT at 0.18 m/s on the pose
                  estimate; on odometry (RTK NONE) row-following against registry trunks at 0.10 m/s -> arrive
                  <= 0.5 m from home
    STUCK         progress < 2 cm in 5 s while the gait runs, or stance-foot sink deeper than SINK_DEPTH_M: stop ->
                  crown stow (if up) -> back up 0.2 m -> retry with step height x 1.5 -> retry on an alternative heading
                  +-30 deg -> assist
                  (2 retries per incident; a new stall within 1.5 m / 60 s of the last continues the same ladder)
    SAFE_PARK     operator / tool fault / end of shift (and inside CLOUD_LINK): arm PARK -> blade / jaw open ->
                  crown stow -> if the gates allow and the local link is up, move <= 3 m to the nearest alley edge or
                  headland (out of the alley lane, out of a basin) -> sit -> beacon -> hold

Every behaviour returns ``Actions`` (what to do now) from ``update(Obs)``. Crown requests are WP24 ``recovery``
requests (e-stop > safety > recovery > preset); WP6 sends its own ``safety`` retracts (link loss, low battery).
``Actions.lock`` means "refuse every motion command but recovery's own": the driver glue stops the gait every
step, the live stack reads ``motion_inhibit.recovery`` (change request to ``locomotion_logic.MotionInhibitGate``).

"Sit" is a 6-foot stance with the gait stopped: the tripod gait has no crouch pose, and WP6 measured that the belly
rests do not reach the ground even in a de-powered sit-down (the WP6 contract notes). Numbers marked ASSUMED are
not measured on hardware.
"""
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------------------------- constants
LOCAL_TIMEOUT_S = 0.5            # OS 5.3: local heartbeat (detection is WP7's LinkMonitor)
CLOUD_TIMEOUT_S = 5.0
TOOLS_SAFE_S = 3.0               # MP 8 KPI: stopped + tools safe <= 3 s
CROWN_STOW_S = 8.5               # MP 8 KPI: crown <= 8.5 s (1.60 m at 0.2 m/s = 8.0 s + 0.5 s stance)
STANCE_WAIT_S = 0.5              # ordered retract: 6-foot stance <= 0.5 s before the retract
CROWN_STOWED_M = 0.02            # = stability.CROWN_UP_EXT_M
SAFE_PARK_MAX_M = 3.0            # MP 8: move <= 3 m to the alley edge / headland
STOP_SPEED_MPS = 0.02            # = safety_logic.STOP_SPEED_MPS
STOP_STILL_S = 0.2               # stopped = below STOP_SPEED_MPS for this long (as the WP6 e-stop timing)
SIT_SETTLE_S = 0.5
STUCK_WINDOW_S = 5.0             # OS 5.3: progress < 2 cm in 5 s
STUCK_PROGRESS_M = 0.02
SINK_DEPTH_M = 0.05              # ASSUMED: deeper than the 3-5 cm soft layer the robot is designed to walk on (OS 3.6;
                                 # WP4 measured 3.6-3.7 cm standing on sink_1, 2.1 mm on mud; walking it at 0.18 m/s)
SINK_SUSTAIN_S = 1.0
BACKUP_M = 0.20                  # OS 5.3: back up 0.2 m
STEP_HEIGHT_RETRY = 1.5          # OS 5.3: step height x 1.5
ALT_HEADING_DEG = 30.0           # MP 8: alternative heading +-30 deg
STUCK_RETRIES = 2                # OS 5.3: 2 retries, then assist
RETRY_FREE_M = 0.50              # a retry counts as recovered after this much progress out of the stall point ...
RETRY_BUDGET_S = 20.0            # ... within this long; a new stall inside the retry fails it at once
INCIDENT_RADIUS_M = 1.5          # a stall this close to the last one (within INCIDENT_S) continues its ladder
INCIDENT_S = 60.0
BACKUP_TIMEOUT_S = 6.0
RETRY_VX = 0.8                   # retry stride (normalised). MEASURED on the sink root line: 0.5 recovered fewer seeds
VX_MPS_PER_CMD = 0.275           # MEASURED (orchard_demo flat, 2026-11-18, 4 s steady): 0.268-0.281 m/s per unit vx
                                 # over vx 0.3-1.0 (0.08 m/s at 0.3, 0.199 at 0.72, 0.27 at 1.0); WP13 measured 0.24
RTH_SPEED_RTK_MPS = 0.18         # MP 8 / RTH-DOCKING: FIXED or FLOAT
RTH_SPEED_ODOM_MPS = 0.10        # odometry only: row-following at 0.1 m/s
STOWED_SPEED_MPS = 0.18          # MP 0.3 stowed walking cap
HOME_TOL_M = 0.50                # MP 8 KPI: arrive <= 0.5 m from home
ARRIVE_AIM_M = 0.12              # the controller aims this close on the estimate
WAYPOINT_TOL_M = 0.30
ARRIVE_SETTLE_S = 2.0            # stand at home before the verify (the estimate settles on FLOAT)
RTH_TIMEOUT_S = 600.0
ITEM_FINISH_S = 20.0             # finish the current item at most this long, then abandon it safely
FALL_TILT_DEG = 35.0             # = safety.yaml fall.tilt_deg
FALL_SUSTAIN_S = 0.3
HULL_HALF_M = 0.40               # robot hull half-width incl. the feet (WP6 CROWN_PATH_HULL_R_M)
EDGE_CLEAR_M = 0.10              # park target this far outside the canopy footprint + hull
BASIN_RIM_M = 0.30               # rim crest outside basin_r (WP4 terrain basins)
ROUTE_MARGIN_M = 0.10            # RETURN HOME grid: extra clearance beyond the hull (the shortcut check uses the hull)
LANDMARK_RANGE_M = 3.0           # registry trunks within this range are row-following landmarks
LANDMARK_SIGMA_M = 0.03          # ASSUMED relative trunk fit noise (WP5 low-plane trunk fit stands in for it)
LANDMARK_GAIN = 0.10             # per observation per tick
YAW_GAIN = 1.5                   # heading P-term (normalised yaw per rad)
TURN_IN_PLACE_RAD = 0.35

WAIT_OUTCOMES = ("assist", "halted")    # done, but the behaviour keeps the robot until the operator answers
PRIORITY = ("fallen", "estop_recover", "local_link", "cloud_link", "return_home", "stuck", "safe_park")
MODES = {"fallen": "fallen", "estop_recover": "estop_recover", "local_link": "safe_park",
         "cloud_link": "safe_park", "return_home": "return_home", "stuck": "stuck_recover",
         "safe_park": "safe_park"}             # MISSION-PRESETS 0.1 mission modes
# MISSION-PRESETS 0.5 reason codes / OS 5.4 tele/alert ids used here
R_LOCAL, R_CLOUD, R_STUCK, R_SINK, R_ENERGY = "local_link_lost", "cloud_link_lost", "stuck", "sink", "energy_reserve"
A_LOCAL_HOLD, A_CLOUD_HOLD, A_STUCK, A_LOW_BAT = ("local_link_lost_hold", "cloud_link_lost_hold", "stuck",
                                                  "low_battery_return")
FALLEN_CHECKLIST = ("crown_lock", "lift_points", "two_people", "laterals_tools", "mast_inspected", "person_clear")
ESTOP_CHECKLIST = ("tools_safe", "fields_clear", "stability")


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


# ---------------------------------------------------------------------------------------------- I/O types
@dataclass
class Obs:
    """What the behaviours see. ``x, y`` is the navigation pose (WP7 pose estimate); ``prog_x, prog_y`` the progress
    pose for the stall detector (sim: truth, assisted ``progress_from_truth``; robot: RTK FIXED + visual odometry)."""
    t: float
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    walking: bool = False
    holding: bool = False
    hold_reason: str = ""
    ext_m: float = 0.0
    mast_source: Optional[str] = None
    tools_safe: Optional[bool] = None     # None: no tool system / unknown (counts as safe once commanded)
    link_local: str = "unknown"           # unknown | up | lost (WP7)
    link_cloud: str = "unknown"
    rtk: str = "FIXED"                    # FIXED | FLOAT | NONE (WP7)
    energy_action: str = "continue"       # WP7 reserve monitor
    tilt_deg: float = 0.0
    sink_m: float = 0.0                   # mean penetration of the feet standing in soft ground
    item_active: bool = False
    walk_ok: bool = True                  # no WP6 stance / walk / stop inhibit
    fields_clear: bool = True             # WP6 protective field clear
    stability_ok: bool = True             # WP6 stability clear
    prog_x: Optional[float] = None
    prog_y: Optional[float] = None
    landmarks: Tuple[Tuple[float, float, float, float], ...] = ()   # (tree_x, tree_y, rel_x_body, rel_y_body)
    last_completed: Optional[str] = None  # mission layer: last tree completed (resume point)
    current_item: Optional[str] = None
    cmd_vx: Optional[float] = None        # the gait's translational command (body frame); None = unknown (forward)
    cmd_vy: Optional[float] = None


@dataclass
class Motion:
    """A gait request (normalised like /nandi/cmd_vel). ``stop``: hard stop now (gait.stop); ``release``: finish
    the cycle and stand; else walk (vx, vy, yaw). ``step_height`` None = the gait default."""
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    release: bool = True
    stop: bool = False
    step_height: Optional[float] = None

    @classmethod
    def go(cls, vx=0.0, vy=0.0, yaw=0.0, step_height=None) -> "Motion":
        return cls(clamp(vx, -1, 1), clamp(vy, -1, 1), clamp(yaw, -1, 1), False, False, step_height)

    @classmethod
    def halt(cls) -> "Motion":
        return cls(stop=True)

    @property
    def moving(self) -> bool:
        return not self.release and not self.stop and (abs(self.vx) + abs(self.vy) + abs(self.yaw)) > 1e-9

    def to_dict(self) -> dict:
        return {"vx": _r(self.vx), "vy": _r(self.vy), "yaw": _r(self.yaw), "release": self.release,
                "stop": self.stop, "step_height": self.step_height}


STAND = Motion()


@dataclass
class Actions:
    motion: Optional[Motion] = None       # None: recovery does not drive the gait
    lock: bool = False                    # refuse every motion command except ``motion``
    arm_park: bool = False
    tools_open: bool = False
    crown: Optional[float] = None         # ensure a WP24 ``recovery`` request to this extension (0 = stow)
    crown_reason: str = ""
    queue_pause: bool = False
    new_targets: bool = True
    beacon: Optional[str] = None          # WP1 beacon state request (stopped / going_home / assist ...)
    power_off: bool = False               # enter the safe hold (fall)
    stand_up: bool = False                # leave the safe hold (e-stop recovery, both confirmations)
    prompt: str = ""                      # console prompt (OS 5.3 "Operator prompt")
    alerts: List[dict] = field(default_factory=list)

    def merge_alerts(self, alerts: List[dict]) -> "Actions":
        self.alerts.extend(alerts)
        return self


def drive_to(o: Obs, target: Sequence[float], speed_mps: float, xy: Optional[Tuple[float, float]] = None,
             step_height: Optional[float] = None) -> Motion:
    """Turn toward ``target`` in place when the bearing error exceeds TURN_IN_PLACE_RAD, else walk at
    ``speed_mps`` with a heading P-term (SimDriver.go_to steering, speed capped)."""
    x, y = xy if xy is not None else (o.x, o.y)
    ang = wrap(math.atan2(target[1] - y, target[0] - x) - o.yaw)
    if abs(ang) > TURN_IN_PLACE_RAD:
        return Motion.go(0.0, 0.0, 1.0 if ang > 0 else -1.0, step_height)
    vx = clamp(speed_mps / VX_MPS_PER_CMD, 0.1, 1.0)
    return Motion.go(vx, 0.0, YAW_GAIN * ang, step_height)


# ---------------------------------------------------------------------------------------------- field map
@dataclass
class _Tree:
    tree_id: str
    x: float
    y: float
    canopy_r: float
    basin_r: float
    row_id: str


@dataclass
class _Row:
    row_id: str
    y: float
    x0: float          # canopy footprint along the row (min x - canopy, max x + canopy)
    x1: float
    edge: float        # half-width of the row's canopy / basin footprint across the row


class FieldMap:
    """The registry geometry recovery needs: trees (canopy, basin), rows, home, geofence, slope cores, soft patches.
    Built from a ``nandi.field/1`` dict (``world.field_spec.load``) or by hand in tests."""

    def __init__(self, trees: Sequence[dict], rows: Sequence[dict], home: Sequence[float] = (0.0, 0.0),
                 geofence: Optional[Sequence[Sequence[float]]] = None, keepout: Sequence[Sequence[float]] = ()):
        self.trees = [_Tree(str(t.get("tree_id", f"t{i}")), float(t["x"]), float(t["y"]), float(t.get("canopy_r", 1.0)),
                            float(t.get("basin_r", 0.8)), str(t.get("row_id", ""))) for i, t in enumerate(trees)]
        self.home = (float(home[0]), float(home[1]))
        self.geofence = [(float(p[0]), float(p[1])) for p in geofence] if geofence else None
        self.keepout = [tuple(float(v) for v in k) for k in keepout]     # (x0, x1, y0, y1): slope cores, docks ...
        self.rows: List[_Row] = []
        for r in rows:
            rid = str(r["row_id"])
            ts = [t for t in self.trees if t.row_id == rid]
            if not ts:
                continue
            edge = max(max(t.canopy_r, t.basin_r + BASIN_RIM_M) for t in ts)
            self.rows.append(_Row(rid, float(r["y"]), min(t.x - t.canopy_r for t in ts),
                                  max(t.x + t.canopy_r for t in ts), edge))
        self.rows.sort(key=lambda r: r.y)

    @classmethod
    def from_field(cls, fld: dict) -> "FieldMap":
        trees = []
        for t in fld.get("trees") or []:
            g = t.get("geo") or {}
            trees.append({"tree_id": t.get("tree_id"), "x": g.get("x", t.get("x", 0.0)), "y": g.get("y", t.get("y", 0.0)),
                          "canopy_r": t.get("canopy_r_m", 1.0), "basin_r": t.get("basin_r_m", 0.8),
                          "row_id": t.get("row_id", "")})
        home = (fld.get("home") or {}).get("xy", [0.0, 0.0])
        keep = []
        for s in ((fld.get("terrain") or {}).get("slopes") or []):
            c = s.get("core") or {}
            if "x" in c and "y" in c:
                keep.append((c["x"][0] - 0.5, c["x"][1] + 0.5, c["y"][0] - 0.5, c["y"][1] + 0.5))
        crate = fld.get("crate") or {}
        if crate.get("xy"):
            cx, cy = crate["xy"]
            sx, sy = (crate.get("size_m") or [0.6, 0.4])[:2]
            keep.append((cx - sx / 2 - HULL_HALF_M, cx + sx / 2 + HULL_HALF_M, cy - sy / 2 - HULL_HALF_M,
                         cy + sy / 2 + HULL_HALF_M))
        return cls(trees, fld.get("rows") or [], home, fld.get("geofence"), keep)

    # -- predicates
    def in_geofence(self, p) -> bool:
        if not self.geofence:
            return True
        x, y = p
        inside = False
        poly = self.geofence
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
                inside = not inside
            j = i
        return inside

    def clear(self, p, hull: float = HULL_HALF_M) -> bool:
        """Robot hull at ``p`` outside every canopy footprint, basin rim and keep-out box, inside the geofence."""
        x, y = float(p[0]), float(p[1])
        for t in self.trees:
            if math.hypot(x - t.x, y - t.y) < max(t.canopy_r, t.basin_r + BASIN_RIM_M) + hull - 1e-9:
                return False
        for (x0, x1, y0, y1) in self.keepout:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return False
        return self.in_geofence((x, y))

    def segment_clear(self, a, b, hull: float = HULL_HALF_M, step: float = 0.1) -> bool:
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        for i in range(1, n + 1):
            u = i / n
            if not self.clear((a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])), hull):
                return False
        return True

    def rows_covering(self, x: float) -> List[_Row]:
        return [r for r in self.rows if r.x0 - 1e-9 <= x <= r.x1 + 1e-9]

    def alley_of(self, p) -> Optional[Tuple[Optional[_Row], Optional[_Row]]]:
        """(row below, row above) bounding ``p`` across the rows that cover its x; None outside every row's span."""
        rows = self.rows_covering(p[0])
        if not rows:
            return None
        below = [r for r in rows if r.y <= p[1]]
        above = [r for r in rows if r.y > p[1]]
        return (below[-1] if below else None, above[0] if above else None)

    # -- SAFE PARK target
    def park_target(self, p, max_m: float = SAFE_PARK_MAX_M) -> Tuple[Optional[Tuple[float, float]], str]:
        """Nearest clear alley-edge or headland point within ``max_m`` of ``p`` -> (target, kind), or (None, why).
        Alley edge: just outside the canopy / basin footprint of the bounding row (the lane stays free). Headland:
        past the end of the rows. Open ground (no row covers x): where it stands if clear, else the nearest clear point.
        """
        x, y = float(p[0]), float(p[1])
        cands: List[Tuple[float, Tuple[float, float], str]] = []
        al = self.alley_of((x, y))
        if al is not None:
            for r in al:
                if r is None:
                    continue
                side = 1.0 if y >= r.y else -1.0
                ye_t = r.y + side * (r.edge + HULL_HALF_M + EDGE_CLEAR_M)
                for dx in (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0, 1.25, -1.25, 1.5, -1.5):
                    q = (x + dx, ye_t)
                    if self.clear(q):
                        cands.append((math.hypot(q[0] - x, q[1] - y), q, "alley_edge"))
                        break
            rows = self.rows_covering(x)
            for xe in (min(r.x0 for r in rows) - HULL_HALF_M - EDGE_CLEAR_M,
                       max(r.x1 for r in rows) + HULL_HALF_M + EDGE_CLEAR_M):
                q = (xe, y)
                if self.clear(q):
                    cands.append((abs(xe - x), q, "headland"))
        if al is None or al[0] is None or al[1] is None:
            # not between two rows: open ground / headland beside the block (no traffic lane to clear)
            kind = "headland" if self.rows else "open"
            if self.clear((x, y)):
                cands.append((0.0, (x, y), kind))
            else:
                for rr in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
                    found = None
                    for k in range(16):
                        a = 2 * math.pi * k / 16
                        q = (x + rr * math.cos(a), y + rr * math.sin(a))
                        if self.clear(q):
                            found = q
                            break
                    if found:
                        cands.append((rr, found, kind))
                        break
        cands = [c for c in cands if c[0] <= max_m + 1e-9]
        if not cands:
            return None, "no_clear_edge_within_3m"
        cands.sort(key=lambda c: c[0])
        return cands[0][1], cands[0][2]

    # -- RETURN HOME route
    def route_home(self, p, home: Optional[Sequence[float]] = None) -> List[Tuple[float, float]]:
        """Waypoints from ``p`` to home: straight if the hull sweep is clear; else the shortest path on a 0.25 m grid
        over the registry map (cells where the hull + ROUTE_MARGIN_M clears every canopy / basin rim and keep-out box,
        inside the geofence), string-pulled to the fewest clear segments. Falls back to alley centreline ->
        headland -> home when the grid finds nothing (start or home inside an obstacle footprint)."""
        h = tuple(home) if home is not None else self.home
        p = (float(p[0]), float(p[1]))
        if self.segment_clear(p, h):
            return [h]
        path = self._grid_path(p, h)
        if path:
            return path
        return self._alley_route(p, h)

    def _grid_path(self, p, h, res: float = 0.25) -> List[Tuple[float, float]]:
        import heapq
        hull = HULL_HALF_M + ROUTE_MARGIN_M
        if self.geofence:
            xs, ys = [q[0] for q in self.geofence], [q[1] for q in self.geofence]
        else:
            xs = [p[0], h[0]] + [t.x for t in self.trees]
            ys = [p[1], h[1]] + [t.y for t in self.trees]
        x0, y0 = min(xs) - 1.0, min(ys) - 1.0
        nx, ny = int((max(xs) + 1.0 - x0) / res) + 1, int((max(ys) + 1.0 - y0) / res) + 1
        cell = lambda q: (int(round((q[0] - x0) / res)), int(round((q[1] - y0) / res)))  # noqa: E731
        pos = lambda c: (x0 + c[0] * res, y0 + c[1] * res)  # noqa: E731
        free: Dict[Tuple[int, int], bool] = {}

        def ok(c):
            if c not in free:
                free[c] = 0 <= c[0] < nx and 0 <= c[1] < ny and self.clear(pos(c), hull)
            return free[c]
        s, g = cell(p), cell(h)
        free[s] = free[g] = True                 # start / goal cells are where the robot is / must go
        dist = {s: 0.0}
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
        heap = [(math.hypot(g[0] - s[0], g[1] - s[1]), s)]
        steps = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        while heap:
            _, c = heapq.heappop(heap)
            if c == g:
                break
            for dx, dy, w in steps:
                n = (c[0] + dx, c[1] + dy)
                if not ok(n):
                    continue
                nd = dist[c] + w
                if nd < dist.get(n, math.inf):
                    dist[n], prev[n] = nd, c
                    heapq.heappush(heap, (nd + math.hypot(g[0] - n[0], g[1] - n[1]), n))
        if g not in dist:
            return []
        cells = [g]
        while cells[-1] != s:
            cells.append(prev[cells[-1]])
        pts = [pos(c) for c in reversed(cells)][1:-1] + [h]
        out: List[Tuple[float, float]] = []
        cur, i = p, 0
        while i < len(pts):
            j = len(pts) - 1
            while j > i and not self.segment_clear(cur, pts[j]):
                j -= 1
            out.append((round(pts[j][0], 3), round(pts[j][1], 3)) if j < len(pts) - 1 else pts[j])
            cur, i = pts[j], j + 1
        return out

    def _alley_route(self, p, h) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        al = self.alley_of(p)
        if al is not None:
            lo, hi = al
            if lo is not None and hi is not None:
                yc = 0.5 * (lo.y + hi.y)
            elif lo is not None:
                yc = lo.y + lo.edge + HULL_HALF_M + 0.6
            else:
                yc = hi.y - hi.edge - HULL_HALF_M - 0.6
            rows = self.rows_covering(p[0])
            x_end = (min(r.x0 for r in rows) - HULL_HALF_M - 0.2 if h[0] <= p[0]
                     else max(r.x1 for r in rows) + HULL_HALF_M + 0.2)
            if abs(p[1] - yc) > 0.3:
                pts.append((p[0], yc))
            pts.append((x_end, yc))
        pts.append(h)
        # drop corners whose shortcut is clear
        out: List[Tuple[float, float]] = []
        cur = p
        i = 0
        while i < len(pts):
            j = len(pts) - 1
            while j > i and not self.segment_clear(cur, pts[j]):
                j -= 1
            out.append(pts[j])
            cur = pts[j]
            i = j + 1
        return out

    def nearest_trees(self, p, rng: float = LANDMARK_RANGE_M) -> List[_Tree]:
        return [t for t in self.trees if math.hypot(t.x - p[0], t.y - p[1]) <= rng]


# ---------------------------------------------------------------------------------------------- crown (ordered)
class CrownStow:
    """MP 8 CROWN AUTO-RETRACT, ordered: stop the gait -> 6-foot stance (<= 0.5 s) -> WP24 ``recovery`` retract to 0
    at 0.2 m/s -> stowed. No walking until stowed (the caller keeps the gait stopped). WP28 never re-raises: every
    recovery behaviour ends stowed (the OS 5.3 one re-raise after 30 s is the preset's, WP25)."""

    def __init__(self, reason: str):
        self.reason = reason
        self.phase = "idle"
        self.t0: Optional[float] = None
        self.t_retract: Optional[float] = None
        self.t_stowed: Optional[float] = None
        self.ext0 = 0.0

    def start(self, o: Obs):
        self.t0, self.ext0 = o.t, o.ext_m
        self.phase = "stowed" if o.ext_m <= CROWN_STOWED_M else "stance"
        if self.phase == "stowed":
            self.t_stowed = o.t

    @property
    def done(self) -> bool:
        return self.phase == "stowed"

    def update(self, o: Obs, a: Actions):
        if self.phase == "idle":
            self.start(o)
        if o.ext_m <= CROWN_STOWED_M:
            if self.phase != "stowed":
                self.phase, self.t_stowed = "stowed", o.t
            return
        if self.phase == "stance":
            a.motion = a.motion or Motion.halt()
            if (not o.walking and o.speed < 0.05) or o.t - self.t0 >= STANCE_WAIT_S - 1e-9:
                self.phase, self.t_retract = "retract", o.t
        if self.phase == "retract":
            a.crown, a.crown_reason = 0.0, self.reason

    def status(self) -> dict:
        return {"phase": self.phase, "ext0_m": _r(self.ext0), "t_retract": _r(self.t_retract),
                "stow_s": None if self.t_stowed is None or self.t0 is None else _r(self.t_stowed - self.t0)}


# ---------------------------------------------------------------------------------------------- behaviours
class Behaviour:
    name = "behaviour"

    def __init__(self):
        self.step = "idle"
        self.t0: Optional[float] = None
        self.t_step: Optional[float] = None
        self._recorded = False                    # in the manager history
        self.outcome: Optional[str] = None        # done: ok / home / resumed / halted / assist / manual / preempted
        self.reason = ""
        self.log: List[Tuple[float, str, str]] = []
        self.alerts: List[dict] = []

    @property
    def active(self) -> bool:
        return self.t0 is not None and self.outcome is None

    def _to(self, step: str, t: float, why: str = ""):
        self.step, self.t_step = step, t
        self.log.append((round(t, 3), step, why))

    def _done(self, t: float, outcome: str, why: str = ""):
        self.outcome = outcome
        self._to("done", t, why or outcome)

    def _alert(self, t: float, aid: str, **kw):
        self.alerts.append({"t": round(t, 3), "id": aid, **kw})

    def take_alerts(self) -> List[dict]:
        out, self.alerts = self.alerts, []
        return out

    def preempt(self, t: float, by: str):
        if self.active:
            self._done(t, "preempted", by)

    def status(self) -> dict:
        return {"behaviour": self.name, "step": self.step, "outcome": self.outcome, "reason": self.reason,
                "t0": _r(self.t0), "log": self.log[-12:]}


class _StopWatch:
    """Stopped = speed below STOP_SPEED_MPS (and the gait not walking) continuously for STOP_STILL_S."""

    def __init__(self):
        self.below: Optional[float] = None
        self.t_stopped: Optional[float] = None

    def update(self, o: Obs) -> bool:
        if self.t_stopped is not None:
            return True
        if o.speed < STOP_SPEED_MPS and not o.walking:
            if self.below is None:
                self.below = o.t
            if o.t - self.below >= STOP_STILL_S - 1e-9:
                self.t_stopped = self.below
                return True
        else:
            self.below = None
        return False


class SafePark(Behaviour):
    """MP 8 SAFE PARK. ``move`` False (or the local link not up, or a WP6 inhibit) sits where it stands."""
    name = "safe_park"

    def __init__(self, fmap: Optional[FieldMap] = None, move: bool = True, reason: str = "operator"):
        super().__init__()
        self.fmap, self.allow_move, self.reason = fmap, move, reason
        self.crown = CrownStow("safe_park")
        self.t_tools: Optional[float] = None
        self.target: Optional[Tuple[float, float]] = None
        self.target_kind = ""
        self.xy0: Optional[Tuple[float, float]] = None
        self.moved_m = 0.0
        self.t_move: Optional[float] = None
        self.sw = _StopWatch()
        self.det = StallDetector()
        self._last_motion: Optional[Motion] = None

    def start(self, o: Obs):
        self.t0, self.xy0 = o.t, (o.x, o.y)
        self._to("make_safe", o.t, self.reason)

    def update(self, o: Obs) -> Actions:
        a = Actions(arm_park=True, tools_open=True, beacon="stopped", lock=False)
        if not self.active:
            return a
        if o.holding:                                   # e-stop: freeze (the e-stop recovery owns the robot)
            a.motion = Motion.halt()
            return a
        if self.step == "make_safe":
            a.motion = Motion.halt()
            if o.tools_safe is not False:
                self.t_tools = o.t
                self._to("stow", o.t)
            elif o.t - self.t0 > TOOLS_SAFE_S:
                self.t_tools = None
                self._alert(o.t, "tool_fault", reason="tools_not_safe")
                self._to("stow", o.t, "tools_not_safe")
        if self.step == "stow":
            a.motion = a.motion or Motion.halt()
            self.crown.update(o, a)
            if self.crown.done:
                self._choose(o)
        if self.step == "move":
            a.crown, a.crown_reason = 0.0, "safe_park"
            d = math.hypot(o.x - self.xy0[0], o.y - self.xy0[1])
            self.moved_m = max(self.moved_m, d)
            if o.link_local != "up" or not o.walk_ok:
                self._to("sit", o.t, "move_stopped:" + ("local_link" if o.link_local != "up" else "inhibit"))
            elif math.hypot(self.target[0] - o.x, self.target[1] - o.y) <= ARRIVE_AIM_M:
                self._to("sit", o.t, "at_target")
            elif d > SAFE_PARK_MAX_M + 0.2 or o.t - self.t_move > 40.0:
                self._to("sit", o.t, "move_limit")
            elif (stall := _own_stall(self.det, o, self._last_motion)) is not None:
                self._alert(o.t, A_STUCK, reason=stall, behaviour=self.name)
                self._to("sit", o.t, "move_stopped:" + stall)       # park where it stands (<= 3 m moved anyway)
            else:
                a.motion = drive_to(o, self.target, STOWED_SPEED_MPS)
                self._last_motion = a.motion
        if self.step == "sit":
            a.motion = STAND
            if self.sw.update(o) or o.t - self.t_step > 3.0:
                self._to("hold", o.t)
        if self.step == "hold":
            a.motion, a.lock = STAND, True
        return a

    def _choose(self, o: Obs):
        if not self.allow_move:
            self._to("sit", o.t, "no_move")
            return
        if o.link_local != "up":
            self._to("sit", o.t, "local_link_not_up")
            return
        if not o.walk_ok:
            self._to("sit", o.t, "gates")
            return
        if self.fmap is None:
            self._to("sit", o.t, "no_field_map")
            return
        tgt, kind = self.fmap.park_target((o.x, o.y))
        if tgt is None:
            self._to("sit", o.t, kind)
            return
        self.target, self.target_kind = tgt, kind
        if math.hypot(tgt[0] - o.x, tgt[1] - o.y) <= ARRIVE_AIM_M:
            self._to("sit", o.t, "already_" + kind)
            return
        self.t_move = o.t
        self._to("move", o.t, kind)

    @property
    def holding(self) -> bool:
        return self.step == "hold"

    def release(self, t: float, outcome: str = "resumed"):
        if self.active:
            self._done(t, outcome)

    def status(self) -> dict:
        s = super().status()
        s.update({"tools_safe_s": None if self.t_tools is None else _r(self.t_tools - self.t0),
                  "crown": self.crown.status(), "target": None if self.target is None else [_r(v) for v in self.target],
                  "target_kind": self.target_kind, "moved_m": _r(self.moved_m)})
        return s


class LocalLinkLoss(Behaviour):
    """MP 8 LOCAL SAFETY LINK LOSS: stop -> tools safe -> crown retract -> sit -> hold, no autonomous motion until
    the link is back AND the operator answered "Resume / Halt"."""
    name = "local_link"

    def __init__(self):
        super().__init__()
        self.reason = R_LOCAL
        self.crown = CrownStow(R_LOCAL)
        self.sw = _StopWatch()
        self.t_tools: Optional[float] = None
        self.t_restored: Optional[float] = None
        self.stop_xy: Optional[Tuple[float, float]] = None
        self.max_drift_m = 0.0

    def start(self, o: Obs):
        self.t0 = o.t
        self._to("stop", o.t)
        self._alert(o.t, A_LOCAL_HOLD, reason=R_LOCAL)

    def update(self, o: Obs) -> Actions:
        a = Actions(motion=Motion.halt(), lock=True, arm_park=True, tools_open=True, beacon="stopped",
                    queue_pause=True, new_targets=False)
        if not self.active:
            return a
        if self.t_tools is None and o.tools_safe is not False:
            self.t_tools = o.t
        stopped = self.sw.update(o)
        if stopped and self.stop_xy is None:
            self.stop_xy = (o.x, o.y)
        if self.stop_xy is not None and self.t_restored is None:
            self.max_drift_m = max(self.max_drift_m, math.hypot(o.x - self.stop_xy[0], o.y - self.stop_xy[1]))
        if self.step == "stop" and stopped:
            self._to("retract", o.t)
        if self.step in ("retract", "stop"):
            self.crown.update(o, a)
            a.motion = Motion.halt()
            if self.step == "retract" and self.crown.done:
                self._to("hold", o.t)
        if self.step == "hold":
            a.crown, a.crown_reason = 0.0, R_LOCAL
            if o.link_local == "up":
                self.t_restored = o.t
                self._to("await_confirm", o.t, "link_restored")
        if self.step == "await_confirm":
            a.prompt = "Link was lost: Resume / Halt"
            if o.link_local == "lost":
                self.t_restored = None
                self._to("hold", o.t, "lost_again")
        return a

    def ack(self, t: float, answer: str) -> Tuple[bool, str]:
        if self.step != "await_confirm":
            return False, f"refused: link not restored (step {self.step})"
        if answer not in ("resume", "halt"):
            return False, "answer resume or halt"
        self._done(t, "resumed" if answer == "resume" else "halted")
        return True, self.outcome

    def status(self) -> dict:
        s = super().status()
        s.update({"stopped_s": None if self.sw.t_stopped is None else _r(self.sw.t_stopped - self.t0),
                  "tools_safe_s": None if self.t_tools is None else _r(self.t_tools - self.t0),
                  "crown": self.crown.status(), "drift_m": _r(self.max_drift_m, 4)})
        return s


class CloudLinkLoss(Behaviour):
    """MP 8 CLOUD LINK LOSS (local up): queue paused, 0 new targets, SAFE PARK at the alley edge, hold. Restored
    (or a local supervisor ack, which works while the cloud is still lost) -> resume."""
    name = "cloud_link"

    def __init__(self, fmap: Optional[FieldMap] = None):
        super().__init__()
        self.reason = R_CLOUD
        self.park = SafePark(fmap, move=True, reason=R_CLOUD)
        self.t_restored: Optional[float] = None

    def start(self, o: Obs):
        self.t0 = o.t
        self.park.start(o)
        self._to("safe_park", o.t)
        self._alert(o.t, A_CLOUD_HOLD, reason=R_CLOUD)

    def update(self, o: Obs) -> Actions:
        a = self.park.update(o)
        a.queue_pause, a.new_targets = True, False
        if not self.active:
            return a
        if self.step == "safe_park" and self.park.holding:
            self._to("hold", o.t)
        if self.step == "hold" and o.link_cloud == "up":
            self.t_restored = o.t
            self._to("await_confirm", o.t, "link_restored")
        if self.step == "await_confirm":
            a.prompt = "Console link was lost: Resume / Halt"
            if o.link_cloud == "lost":
                self._to("hold", o.t, "lost_again")
        if self.step in ("hold", "await_confirm", "safe_park"):
            a.prompt = a.prompt or "Console link lost: parked at the alley edge"
        return a

    def ack(self, t: float, answer: str, local: bool = False) -> Tuple[bool, str]:
        """``local``: the local supervisor (handheld / dock link) may resume while the cloud is still lost (MP 8)."""
        if answer not in ("resume", "halt"):
            return False, "answer resume or halt"
        if self.step not in ("await_confirm",) and not (local and self.step in ("hold", "safe_park")):
            return False, f"refused: console link not restored (step {self.step}); a local ack can resume"
        self.park.release(t, "resumed")
        self._done(t, "resumed" if answer == "resume" else "halted")
        return True, self.outcome

    def status(self) -> dict:
        s = super().status()
        s["park"] = self.park.status()
        return s


class ReturnHome(Behaviour):
    """MP 8 RETURN HOME on the WP7 pose estimate. RTK FIXED / FLOAT: 0.18 m/s. RTK NONE (odometry only): 0.10 m/s,
    row-following: registry trunks seen within LANDMARK_RANGE_M correct the estimate (an offset kept here; the WP7
    PoseEstimator is not modified). Arrive <= 0.5 m from home (aims ARRIVE_AIM_M on the estimate, settles, checks)."""
    name = "return_home"

    def __init__(self, fmap: Optional[FieldMap] = None, home: Optional[Sequence[float]] = None, reason: str = "operator"):
        super().__init__()
        self.fmap, self.reason = fmap, reason
        self.home = tuple(home) if home is not None else (fmap.home if fmap is not None else (0.0, 0.0))
        self.crown = CrownStow("return_home")
        self.route: List[Tuple[float, float]] = []
        self.wp = 0
        self.corr = [0.0, 0.0]
        self.mode = ""
        self.mode_s: Dict[str, float] = {"rtk": 0.0, "odometry": 0.0}
        self._last_t: Optional[float] = None
        self.n_landmarks = 0
        self.t_arrive: Optional[float] = None
        self.est_err_m: Optional[float] = None
        self.det = StallDetector()               # the manager's detector is off while a behaviour runs
        self._last_motion: Optional[Motion] = None

    def start(self, o: Obs):
        self.t0 = o.t
        self._to("finish_item" if o.item_active else "stow", o.t)
        if self.reason == R_ENERGY:
            self._alert(o.t, A_LOW_BAT, reason=R_ENERGY)

    def pose(self, o: Obs) -> Tuple[float, float]:
        return (o.x + self.corr[0], o.y + self.corr[1])

    def _landmarks(self, o: Obs):
        """Row-following correction: each relative trunk observation implies a robot position."""
        c, s = math.cos(o.yaw), math.sin(o.yaw)
        for (tx, ty, rx, ry) in o.landmarks:
            ix, iy = tx - (c * rx - s * ry), ty - (s * rx + c * ry)
            px, py = self.pose(o)
            self.corr[0] += LANDMARK_GAIN * (ix - px)
            self.corr[1] += LANDMARK_GAIN * (iy - py)
            self.n_landmarks += 1

    def update(self, o: Obs) -> Actions:
        a = Actions(arm_park=True, beacon="going_home")
        if not self.active:
            if self.outcome == "assist":             # waits for the operator: stays parked and locked
                a.motion, a.lock, a.beacon = STAND, True, "assist"
                a.prompt = "Return home failed: assist (Take over / Skip / Halt)"
            return a
        dt = 0.0 if self._last_t is None else max(0.0, o.t - self._last_t)
        self._last_t = o.t
        if o.holding:
            a.motion = Motion.halt()
            return a
        if self.step == "finish_item":
            a.arm_park = False                      # the item layer finishes; recovery only waits
            if not o.item_active:
                self._to("stow", o.t, "item_done")
            elif o.t - self.t0 >= ITEM_FINISH_S:
                self._to("stow", o.t, "item_abandoned")
        if self.step == "stow":
            a.tools_open = True
            a.motion = Motion.halt()
            self.crown.update(o, a)
            if self.crown.done:
                p = self.pose(o)
                self.route = self.fmap.route_home(p, self.home) if self.fmap is not None else [self.home]
                self.wp = 0
                self._to("drive", o.t, f"{len(self.route)} waypoints")
        if self.step == "drive":
            a.crown, a.crown_reason = 0.0, "return_home"
            if o.t - self.t0 > RTH_TIMEOUT_S:
                self._alert(o.t, "rth_failed", reason="timeout")
                self._done(o.t, "assist", "timeout")
                a.motion, a.lock, a.beacon = STAND, True, "assist"
                return a
            odom = o.rtk not in ("FIXED", "FLOAT")
            self.mode = "odometry" if odom else "rtk"
            self.mode_s[self.mode] += dt
            if odom:
                self._landmarks(o)
            else:
                self.corr = [0.0, 0.0]
            p = self.pose(o)
            last = self.wp == len(self.route) - 1
            tgt = self.route[self.wp]
            dist = math.hypot(tgt[0] - p[0], tgt[1] - p[1])
            if not last and dist <= WAYPOINT_TOL_M:
                self.wp += 1
                tgt = self.route[self.wp]
                last = self.wp == len(self.route) - 1
            elif last and dist <= ARRIVE_AIM_M:
                self._to("settle", o.t)
            if self.step == "drive":
                stall = _own_stall(self.det, o, self._last_motion)
                if stall is not None:
                    self._alert(o.t, A_STUCK, reason=stall, behaviour=self.name)
                    self._alert(o.t, "rth_failed", reason=stall)
                    self._done(o.t, "assist", stall)
                    a.motion, a.lock, a.beacon = STAND, True, "assist"
                    a.prompt = "Return home stuck: assist (Take over / Skip / Halt)"
                    return a
                a.motion = drive_to(o, tgt, RTH_SPEED_ODOM_MPS if odom else RTH_SPEED_RTK_MPS, xy=p)
                self._last_motion = a.motion
        if self.step == "settle":
            a.motion = STAND
            self._last_motion = None
            self.det.reset()
            if o.t - self.t_step >= ARRIVE_SETTLE_S:
                p = self.pose(o)
                self.est_err_m = math.hypot(p[0] - self.home[0], p[1] - self.home[1])
                if self.est_err_m <= ARRIVE_AIM_M + 0.08:
                    self.t_arrive = o.t
                    self._done(o.t, "home")
                else:
                    self._to("drive", o.t, f"re-aim {self.est_err_m:.2f} m")
        return a

    def status(self) -> dict:
        s = super().status()
        s.update({"mode": self.mode, "mode_s": {k: _r(v, 1) for k, v in self.mode_s.items()},
                  "route": [[_r(x), _r(y)] for x, y in self.route], "wp": self.wp,
                  "corr_m": [_r(c, 3) for c in self.corr], "landmarks": self.n_landmarks,
                  "crown": self.crown.status(), "est_err_m": _r(self.est_err_m),
                  "home_s": None if self.t_arrive is None else _r(self.t_arrive - self.t0, 1)})
        return s


class StallDetector:
    """OS 5.3 detection: progress < 2 cm in 5 s while the gait runs (on the progress pose), or the stance feet sunk
    deeper than SINK_DEPTH_M for SINK_SUSTAIN_S while walking. Progress is the displacement ALONG the commanded
    direction (``cmd_vx``, ``cmd_vy`` rotated by the heading at the window start; unknown = forward), so a robot that
    rocks in place or slides back is stalled; turning in place (no translational command) is never a stall.
    ``update`` returns ``stuck`` / ``sink`` / None."""

    def __init__(self, window_s: float = STUCK_WINDOW_S, progress_m: float = STUCK_PROGRESS_M,
                 sink_m: float = SINK_DEPTH_M):
        self.window_s, self.progress_m, self.sink_thr = window_s, progress_m, sink_m
        self.reset()

    def reset(self):
        self.hist: List[Tuple[float, float, float, float, float]] = []
        self.walk_since: Optional[float] = None
        self.sink_since: Optional[float] = None
        self.last_progress: Optional[float] = None

    def update(self, o: Obs) -> Optional[str]:
        px = o.prog_x if o.prog_x is not None else o.x
        py = o.prog_y if o.prog_y is not None else o.y
        vx = 1.0 if o.cmd_vx is None else float(o.cmd_vx)
        vy = 0.0 if o.cmd_vy is None else float(o.cmd_vy)
        if not o.walking or o.holding or math.hypot(vx, vy) < 0.05:
            self.reset()
            return None
        if self.walk_since is None:
            self.walk_since = o.t
        n = math.hypot(vx, vy)
        c, s = math.cos(o.yaw), math.sin(o.yaw)
        self.hist.append((o.t, px, py, (c * vx - s * vy) / n, (s * vx + c * vy) / n))
        while self.hist and self.hist[0][0] < o.t - self.window_s - 0.05:
            self.hist.pop(0)
        if o.sink_m > self.sink_thr:
            if self.sink_since is None:
                self.sink_since = o.t
        else:
            self.sink_since = None
        if o.t - self.walk_since >= self.window_s - 1e-9 and self.hist:
            t0, x0, y0, ux, uy = self.hist[0]
            if o.t - t0 >= self.window_s - 0.03:
                prog = (px - x0) * ux + (py - y0) * uy
                self.last_progress = prog
                if prog < self.progress_m:
                    return R_SINK if self.sink_since is not None else R_STUCK
        if self.sink_since is not None and o.t - self.sink_since >= SINK_SUSTAIN_S - 1e-9:
            return R_SINK
        return None


def _own_stall(det: StallDetector, o: Obs, last: Optional[Motion]) -> Optional[str]:
    """Stall check for a behaviour's own moves (RETURN HOME drive, SAFE PARK move): the manager's detector is off while
    a behaviour runs. Progress is measured along the behaviour's last translational command (walking assumed while
    it commands one, so a gait that no longer moves its legs still counts); none / turning in place resets it."""
    if last is None or not last.moving:
        det.reset()
        return None
    return det.update(Obs(**{**o.__dict__, "walking": True, "cmd_vx": last.vx, "cmd_vy": last.vy}))


class StuckRecover(Behaviour):
    """MP 8 STUCK / SINK ladder, cumulative: stop -> back up 0.2 m (step height x 1.5) -> rung 1: retry straight with
    step height x 1.5 -> stop -> back up 0.2 m -> rung 2: retry on the alternative heading (+30 deg, still x 1.5) ->
    assist (STUCK_RETRIES = 2). A retry is ``recovered`` after BACKUP_M + RETRY_FREE_M of progress along the retry
    heading within RETRY_BUDGET_S (the whole body past the obstacle); a stall inside the retry fails it at once.
    A back-up that does not reach 0.2 m within BACKUP_TIMEOUT_S (soft ground) goes on with the retry."""
    name = "stuck"

    def __init__(self, cause: str = R_STUCK, attempt: int = 0, heading: Optional[float] = None, at=None):
        super().__init__()
        self.reason = cause
        self.attempt = attempt               # retries already used in this incident
        self.h0 = heading
        self.at = at
        self.bk_from: Optional[Tuple[float, float]] = None
        self.retry_from: Optional[Tuple[float, float]] = None
        self.det = StallDetector()
        self.attempts: List[dict] = []
        self.heading = heading
        self.step_h: Optional[float] = None
        self.t_free: Optional[float] = None
        self.crown = CrownStow(R_STUCK)

    def start(self, o: Obs):
        self.t0 = o.t
        if self.h0 is None:
            self.h0 = o.yaw
        if self.at is None:
            self.at = (self._px(o), self._py(o))
        self._to("stop", o.t, self.reason)
        self._alert(o.t, A_STUCK, reason=self.reason, attempt=self.attempt)

    @staticmethod
    def _px(o):
        return o.prog_x if o.prog_x is not None else o.x

    @staticmethod
    def _py(o):
        return o.prog_y if o.prog_y is not None else o.y

    def update(self, o: Obs) -> Actions:
        a = Actions(beacon="assist" if self.outcome == "assist" else None)
        if not self.active:
            if self.outcome == "assist":
                a.motion, a.lock, a.prompt = STAND, True, "Stuck: assist (Take over / Skip / Halt)"
            return a
        if o.holding:
            a.motion = Motion.halt()
            return a
        if self.step == "stop":
            a.motion = STAND
            if o.t - self.t_step >= 0.6 and not o.walking:
                if self.attempt >= STUCK_RETRIES:
                    return self._assist(o, a)
                if o.ext_m > CROWN_STOWED_M:          # MP 8 interlock: crown stowed before any recovery motion
                    self.crown = CrownStow(R_STUCK)
                    self._to("stow", o.t, "crown_up")
                else:
                    self.bk_from = (self._px(o), self._py(o))
                    self._to("backup", o.t, f"attempt {self.attempt + 1}")
        if self.step == "stow":
            a.motion = Motion.halt()
            self.crown.update(o, a)
            if self.crown.done:
                self.bk_from = (self._px(o), self._py(o))
                self._to("backup", o.t, f"attempt {self.attempt + 1}")
        if self.step in ("backup", "turn", "retry"):
            a.crown, a.crown_reason = 0.0, R_STUCK     # keep it stowed while the ladder moves
        if self.step == "backup":
            d = math.hypot(self._px(o) - self.bk_from[0], self._py(o) - self.bk_from[1])
            if d >= BACKUP_M or o.t - self.t_step >= BACKUP_TIMEOUT_S:
                self.attempts.append({"attempt": self.attempt + 1, "backup_m": _r(d), "t": _r(o.t)})
                self.attempt += 1
                alt = 0.0 if self.attempt == 1 else math.radians(ALT_HEADING_DEG)
                self.heading = wrap(self.h0 + alt)
                self.step_h = STEP_HEIGHT_RETRY
                self._to("turn" if alt else "retry", o.t)
                self.det.reset()
                self.retry_from = (self._px(o), self._py(o))
            else:
                a.motion = Motion.go(-0.6, 0.0, YAW_GAIN * wrap(self.h0 - o.yaw), STEP_HEIGHT_RETRY)
        if self.step == "turn":
            err = wrap(self.heading - o.yaw)
            if abs(err) <= math.radians(6.0):
                self.det.reset()
                self.retry_from = (self._px(o), self._py(o))
                self._to("retry", o.t)
            elif o.t - self.t_step > 15.0:
                self._to("retry", o.t, "turn_timeout")
            else:
                a.motion = Motion.go(0.0, 0.0, 1.0 if err > 0 else -1.0, self.step_h)
        if self.step == "retry":
            a.motion = Motion.go(RETRY_VX, 0.0, YAW_GAIN * wrap(self.heading - o.yaw), self.step_h)
            prog = ((self._px(o) - self.retry_from[0]) * math.cos(self.heading)
                    + (self._py(o) - self.retry_from[1]) * math.sin(self.heading))
            stall = self.det.update(Obs(**{**o.__dict__, "walking": True, "cmd_vx": RETRY_VX, "cmd_vy": 0.0}))
            if stall == R_SINK and (self.det.last_progress is None or self.det.last_progress >= STUCK_PROGRESS_M):
                stall = None                      # sinking while it still progresses: keep the higher step going
            if prog >= RETRY_FREE_M + BACKUP_M:
                self.t_free = o.t
                self.attempts[-1].update(result="recovered", progress_m=_r(prog))
                self._done(o.t, "recovered", f"attempt {self.attempt}")
                a.motion = STAND
            elif stall is not None or o.t - self.t_step >= RETRY_BUDGET_S:
                self.attempts[-1].update(result=stall or "timeout", progress_m=_r(prog))
                self._to("stop", o.t, stall or "retry_timeout")
        return a

    def _assist(self, o: Obs, a: Actions) -> Actions:
        self._alert(o.t, A_STUCK, reason=self.reason, assist=True, attempts=self.attempt)
        self._done(o.t, "assist", self.reason)
        a.motion, a.lock, a.beacon, a.prompt = STAND, True, "assist", "Stuck: assist (Take over / Skip / Halt)"
        return a

    def status(self) -> dict:
        s = super().status()
        s.update({"attempt": self.attempt, "attempts": self.attempts, "heading_deg": None if self.heading is None
                  else _r(math.degrees(self.heading), 1)})
        return s


class EstopRecovery(Behaviour):
    """MP 8 E-STOP RECOVERY. The interlock (``tests/test_recovery_behaviours.py`` mutation test): ``stand_up`` and any
    motion only after a valid local confirm and then a valid operator confirm, both after the hold began."""
    name = "estop_recover"

    def __init__(self, reason: str = "estop", item_active: bool = False, current_item: Optional[str] = None,
                 last_completed: Optional[str] = None):
        super().__init__()
        self.reason = reason
        self.confirms: Dict[str, float] = {}
        self.refused: List[Tuple[float, str, str]] = []
        self.checklist: Dict[str, bool] = {k: False for k in ESTOP_CHECKLIST}
        self.crown = CrownStow("estop_recover")
        self.aborted_item = current_item if item_active else None
        self.resume_from = last_completed
        self.t_hold: Optional[float] = None
        self.t_standup: Optional[float] = None

    def start(self, o: Obs):
        self.t0 = self.t_hold = o.t
        self._to("await_local", o.t)

    def confirm(self, t: float, kind: str, local_link_up: bool = True, holding: bool = True) -> Tuple[bool, str]:
        def no(why):
            self.refused.append((round(t, 3), kind, why))
            return False, why
        if not self.active:
            return no("no e-stop recovery running")
        if self.t_hold is None or t < self.t_hold:
            return no("confirm older than the e-stop")
        if kind == "local":
            if self.step != "await_local":
                return no(f"local confirm not expected (step {self.step})")
            if not local_link_up:
                return no("local link not up")
            self.confirms["local"] = t
            self._to("await_operator", t)
            return True, "local confirmed"
        if kind == "operator":
            if self.step != "await_operator" or "local" not in self.confirms:
                return no("operator confirm needs the local confirm first")
            if t <= self.confirms["local"]:
                return no("operator confirm must follow the local confirm")
            self.confirms["operator"] = t
            self._to("checklist", t)
            return True, "operator confirmed"
        return no(f"unknown confirm {kind!r}")

    @property
    def confirmed(self) -> bool:
        return ("local" in self.confirms and "operator" in self.confirms
                and self.confirms["operator"] > self.confirms["local"]
                >= (self.t_hold if self.t_hold is not None else math.inf))

    def update(self, o: Obs) -> Actions:
        a = Actions(motion=Motion.halt(), lock=True, beacon=None, queue_pause=True, new_targets=False,
                    prompt="E-stop: local confirm, then operator confirm")
        if not self.active:
            return a
        if self.step in ("await_local", "await_operator") and not o.holding and self.t_standup is None:
            a.prompt = "E-stop released elsewhere: motion stays locked until both confirmations"
        if self.step == "checklist":
            if not self.confirmed:                       # defensive: never past here without both
                self._to("await_local", o.t, "confirm_lost")
                return a
            self.checklist = {"tools_safe": o.tools_safe is not False, "fields_clear": bool(o.fields_clear),
                              "stability": bool(o.stability_ok)}
            a.prompt = "E-stop checklist: " + ", ".join(k for k, v in self.checklist.items() if not v) if not all(
                self.checklist.values()) else "E-stop checklist passed"
            if all(self.checklist.values()):
                self._to("rearm", o.t)
        if self.step == "rearm":
            if self.confirmed:
                a.stand_up = o.holding                     # the runner leaves the hold (re-arm, still locked)
                if not o.holding:
                    self.t_standup = o.t
                    self._to("crown_retract", o.t)
        if self.step == "crown_retract":
            self.crown.update(o, a)
            a.motion = Motion.halt()
            if self.crown.done:
                self._to("stand", o.t)
        if self.step == "stand":
            a.motion = STAND
            if o.t - self.t_step >= 1.0:
                self._done(o.t, "ready")
                a.lock = False
        if self.step in ("crown_retract", "stand") and o.holding:
            self._to("await_local", o.t, "estop_again")          # a new e-stop: start over
            self.confirms.clear()
            self.t_hold = o.t
        return a

    def status(self) -> dict:
        s = super().status()
        s.update({"confirms": {k: _r(v) for k, v in self.confirms.items()}, "refused": self.refused[-6:],
                  "checklist": self.checklist, "crown": self.crown.status(), "aborted_item": self.aborted_item,
                  "resume_from": self.resume_from})
        return s


class Fallen(Behaviour):
    """MP 8 FALLEN: power off (hold), lock; manual checklist + person-clear confirmation. Never ends by itself into
    motion: ``manual`` means the people on site may lift it; the e-stop latch reset stays the supervisor's."""
    name = "fallen"

    def __init__(self, reason: str = "fall"):
        super().__init__()
        self.reason = reason
        self.items: Dict[str, Optional[float]] = {k: None for k in FALLEN_CHECKLIST}

    def start(self, o: Obs):
        self.t0 = o.t
        self._to("power_off", o.t)
        self._alert(o.t, "fallen", reason=self.reason, tilt_deg=_r(o.tilt_deg, 1), checklist=list(FALLEN_CHECKLIST))

    def check(self, t: float, item: str) -> Tuple[bool, str]:
        if item not in self.items:
            return False, f"unknown checklist item {item!r}"
        if item == "person_clear" and any(self.items[k] is None for k in FALLEN_CHECKLIST if k != "person_clear"):
            return False, "person_clear is the last item"
        self.items[item] = t
        if all(v is not None for v in self.items.values()):
            self._done(t, "manual", "checklist complete")
        return True, item

    def update(self, o: Obs) -> Actions:
        a = Actions(motion=Motion.halt(), lock=True, power_off=True, beacon=None, queue_pause=True, new_targets=False,
                    prompt="Fallen: manual recovery")
        if self.step == "power_off":
            self._to("checklist", o.t)
        return a

    def status(self) -> dict:
        s = super().status()
        s["checklist"] = {k: v is not None for k, v in self.items.items()}
        return s


# ---------------------------------------------------------------------------------------------- manager
class RecoveryManager:
    """Picks and runs one behaviour by priority (FALLEN > E-STOP > LOCAL LINK > CLOUD LINK > RETURN HOME > STUCK >
    SAFE PARK); exposes the commands of the console recovery menu (``command``)."""

    def __init__(self, fmap: Optional[FieldMap] = None, home: Optional[Sequence[float]] = None, auto: bool = True):
        self.fmap = fmap
        self.home = tuple(home) if home is not None else (fmap.home if fmap is not None else (0.0, 0.0))
        self.auto = auto
        self.active: Optional[Behaviour] = None
        self.history: List[dict] = []
        self.stall = StallDetector()
        self.fall_since: Optional[float] = None
        self.last_stall: Optional[Tuple[float, float, float, int, float]] = None   # t, x, y, attempts, heading
        self.alerts: List[dict] = []
        self.pending_ack_link: List[str] = []
        self.last: Optional[Actions] = None
        self.last_obs: Optional[Obs] = None
        self.rth_done_at: Optional[Tuple[float, float]] = None
        self.cloud_resumed_lost = False         # a local ack resumed while the cloud is still lost: no re-trigger
        self.fall_manual: Optional[str] = None  # FALLEN checklist done, robot still in the fall hold: tilted / righted
        self.energy_rth_aborted = False         # operator aborted the low-SoC RETURN HOME: no auto restart

    # -- helpers
    def _rank(self, name: str) -> int:
        return PRIORITY.index(name)

    def _start(self, b: Behaviour, o: Obs):
        cur = self.active
        if cur is not None and cur.active:
            cur.preempt(o.t, b.name)
            self._finish()
        elif cur is not None and cur.outcome in WAIT_OUTCOMES and not cur._recorded:
            self._finish()
        self.active = b
        b.start(o)

    def _finish(self):
        b = self.active
        if b is not None and not b._recorded:
            b._recorded = True
            self.history.append({"behaviour": b.name, "outcome": b.outcome, "t0": _r(b.t0), "log": b.log[-20:],
                                 "status": b.status()})

    def _can_take(self, name: str) -> bool:
        b = self.active
        if b is not None and not b.active and b.outcome in WAIT_OUTCOMES:
            # an assist / an operator Halt waits for the operator: only the safety behaviours may take over
            return self._rank(name) < self._rank("return_home")
        return b is None or not b.active or self._rank(name) < self._rank(b.name)

    @staticmethod
    def _halted_actions() -> Actions:
        """Operator answered Halt (link-loss prompt, SAFE PARK hold): stay parked and locked, crown stowed, queue
        paused, no new targets, until an explicit Resume (or a safety behaviour takes over)."""
        return Actions(motion=STAND, lock=True, arm_park=True, tools_open=True, crown=0.0, crown_reason="halted",
                       queue_pause=True, new_targets=False, beacon="stopped",
                       prompt="Halted by the operator: Resume to continue")

    # -- operator / console commands (/nandi/recovery/cmd; ``source`` "local" = the local safety link transport)
    def command(self, t: float, cmd: dict, o: Optional[Obs] = None, source: str = "console") -> Tuple[bool, str]:
        """``source``: ``console`` (``/nandi/recovery/cmd``) or ``local`` (``/nandi/recovery/local``: the handheld /
        dock local-link transport). The e-stop local confirm and a local link-loss ack are accepted only from
        ``local``, the operator confirm only from ``console``, so one console client cannot give both confirmations."""
        o = o or self.last_obs
        if not isinstance(cmd, dict):
            return False, "command must be an object"
        if source not in ("console", "local"):
            return False, f"unknown command source {source!r}"
        b = self.active
        if source == "local" and not ("confirm" in cmd or "ack" in cmd):
            return False, "the local channel takes only {confirm: local} and {ack}"
        if "confirm" in cmd:
            if not isinstance(b, EstopRecovery) or not b.active:
                return False, "no e-stop recovery running"
            kind = str(cmd["confirm"])
            if kind == "local" and source != "local":
                b.refused.append((round(t, 3), kind, "local confirm only on the local channel"))
                return False, "refused: the local confirm comes only from the local safety link (handheld / dock)"
            if kind == "operator" and source != "console":
                b.refused.append((round(t, 3), kind, "operator confirm only on the console channel"))
                return False, "refused: the operator confirm comes only from the console"
            return b.confirm(t, kind, local_link_up=(o is not None and o.link_local == "up"),
                             holding=bool(o and o.holding))
        if "ack" in cmd:
            ans = str(cmd["ack"])
            local = bool(cmd.get("local"))
            if local and source != "local":
                return False, "refused: a local ack comes only from the local safety link (handheld / dock)"
            local = local or source == "local"
            if b is not None and not b.active and b.outcome == "halted":
                if ans == "resume":
                    self._finish()
                    self.active = None
                    return True, "resumed after halt"
                if ans == "halt":
                    return True, "halted"
                return False, "answer resume or halt"
            if isinstance(b, LocalLinkLoss) and b.active:
                ok, msg = b.ack(t, ans)
                if ok:
                    self.pending_ack_link.append("local")
                return ok, msg
            if isinstance(b, CloudLinkLoss) and b.active:
                ok, msg = b.ack(t, ans, local=local)
                if ok:
                    self.pending_ack_link.append("cloud")
                    self.cloud_resumed_lost = o is not None and o.link_cloud == "lost"
                return ok, msg
            if isinstance(b, SafePark) and b.active and b.holding:
                b.release(t, "resumed" if ans == "resume" else "halted")
                return True, b.outcome
            if isinstance(b, StuckRecover) and b.outcome == "assist":
                self.active = None
                return True, "assist acknowledged"
            if isinstance(b, ReturnHome) and b.outcome == "assist":
                self.active = None
                return True, "assist acknowledged"
            return False, "nothing to acknowledge"
        if "check" in cmd:
            if not isinstance(b, Fallen):
                return False, "no fallen checklist running"
            return b.check(t, str(cmd["check"]))
        if "run" in cmd:
            what = str(cmd["run"])
            if o is None:
                return False, "no observation yet"
            if what == "safe_park":
                if not self._can_take("safe_park"):
                    return False, f"refused: {self.active.name} running"
                self._start(SafePark(self.fmap, move=True, reason="operator"), o)
                return True, "safe_park started"
            if what == "return_home":
                if not self._can_take("return_home"):
                    return False, f"refused: {self.active.name} running"
                if o.rtk not in ("FIXED", "FLOAT", "NONE"):
                    return False, "rtk"
                self._start(ReturnHome(self.fmap, self.home, reason="operator"), o)
                return True, "return_home started"
            return False, f"unknown behaviour {what!r} (safe_park, return_home; the others are triggered)"
        if cmd.get("abort"):
            if isinstance(b, (SafePark, ReturnHome, StuckRecover)) and (b.active or b.outcome == "assist"):
                if isinstance(b, ReturnHome) and b.reason == R_ENERGY:
                    self.energy_rth_aborted = True      # until the reserve monitor leaves return_and_wait
                b.preempt(t, "operator_abort")
                self._finish()
                self.active = None
                return True, "aborted"
            return False, "nothing abortable running"
        return False, "unknown command"

    # -- the 50 Hz step
    def update(self, o: Obs) -> Actions:
        self.last_obs = o
        if o.link_cloud == "up":
            self.cloud_resumed_lost = False
        if o.energy_action != "return_and_wait":
            self.energy_rth_aborted = False
        b = self.active
        # FALLEN (sticky)
        if o.tilt_deg > FALL_TILT_DEG:
            self.fall_since = o.t if self.fall_since is None else self.fall_since
        else:
            self.fall_since = None
        fallen = (self.fall_since is not None and o.t - self.fall_since >= FALL_SUSTAIN_S - 1e-9) or (
            o.holding and o.hold_reason == "fall")
        if self.fall_manual is not None:
            # the FALLEN checklist is done and people handle the robot in the (still latched) fall hold: no new FALLEN
            # until the supervisor resets the hold, or it tips over again after being righted
            if not o.holding:
                self.fall_manual = None
            elif o.tilt_deg <= FALL_TILT_DEG:
                self.fall_manual = "righted"
            elif self.fall_manual == "righted":
                self.fall_manual = None
            fallen = fallen and self.fall_manual is None
        if self.auto and fallen and not isinstance(b, Fallen):
            self._start(Fallen("fall"), o)
        elif self.auto and o.holding and o.hold_reason != "fall" and self._can_take("estop_recover") and not (
                isinstance(b, EstopRecovery) and b.active):
            self._start(EstopRecovery(o.hold_reason or "estop", o.item_active, o.current_item, o.last_completed), o)
        elif self.auto and o.link_local == "lost" and self._can_take("local_link"):
            self._start(LocalLinkLoss(), o)
        elif (self.auto and o.link_cloud == "lost" and o.link_local == "up" and self._can_take("cloud_link")
              and not (isinstance(b, ReturnHome) and b.active) and not self.cloud_resumed_lost):
            self._start(CloudLinkLoss(self.fmap), o)
        elif (self.auto and o.energy_action == "return_and_wait" and self._can_take("return_home")
              and not self._at_home(o) and not self.energy_rth_aborted):
            self._start(ReturnHome(self.fmap, self.home, reason=R_ENERGY), o)
        else:
            cause = self.stall.update(o) if (b is None or not b.active) else None
            if b is not None and b.active:
                self.stall.reset()
            if self.auto and cause is not None and self._can_take("stuck"):
                self._start(self._stuck_for(o, cause), o)
        b = self.active
        if b is not None and not b.active and b.outcome == "halted":
            a = self._halted_actions()
        else:
            a = b.update(o) if b is not None else Actions()
        if isinstance(b, StuckRecover) and b.at is not None:
            self.last_stall = (o.t, b.at[0], b.at[1], b.attempt, b.h0 if b.h0 is not None else o.yaw)
        if b is not None:
            a.merge_alerts(b.take_alerts())
            if b.outcome == "halted":
                self._finish()                            # recorded once; it stays active (parked, locked)
            elif b.outcome is not None and b.outcome != "assist":
                if isinstance(b, ReturnHome) and b.outcome == "home":
                    self.rth_done_at = (o.x, o.y)
                if isinstance(b, Fallen) and b.outcome == "manual":
                    self.fall_manual = "righted" if o.tilt_deg <= FALL_TILT_DEG else "tilted"
                self._finish()
                self.active = None
                if isinstance(b, EstopRecovery) and b.outcome == "ready":
                    a.lock = False
            elif b.outcome == "assist" and not isinstance(b, (StuckRecover, ReturnHome)):
                self._finish()
                self.active = None
        self.last = a
        return a

    def _at_home(self, o: Obs) -> bool:
        return math.hypot(o.x - self.home[0], o.y - self.home[1]) <= HOME_TOL_M

    def _stuck_for(self, o: Obs, cause: str) -> StuckRecover:
        px, py = StuckRecover._px(o), StuckRecover._py(o)
        ls = self.last_stall
        attempt, heading = 0, o.yaw
        if ls is not None and o.t - ls[0] <= INCIDENT_S and math.hypot(px - ls[1], py - ls[2]) <= INCIDENT_RADIUS_M:
            attempt, heading = ls[3], ls[4]
        return StuckRecover(cause, attempt=attempt, heading=heading, at=(px, py) if ls is None else None)

    def status(self) -> dict:
        b = self.active
        return {"active": None if b is None else b.name, "mode": None if b is None else MODES.get(b.name),
                "behaviour": None if b is None else b.status(),
                "history": [{"behaviour": h["behaviour"], "outcome": h["outcome"], "t0": h["t0"]} for h in self.history[-8:]]}


# ---------------------------------------------------------------------------------------------- self-right gate
def self_right_gate(person_clear: bool, confirmed: bool, crown_stowed: bool, arm_folded: bool) -> Tuple[bool, str]:
    """MP 8 SELF-RIGHT (spike only): person-clear + confirmation, crown stowed, arm folded."""
    if not person_clear:
        return False, "person_hold"
    if not confirmed:
        return False, "needs_confirmation"
    if not crown_stowed:
        return False, "crown_gate"
    if not arm_folded:
        return False, "arm_not_folded"
    return True, ""


# ---------------------------------------------------------------------------------------------- SimDriver glue
class RecoveryRunner:
    """Applies ``bus.values["recovery"]`` (written by ``atharv_nandi/sim/hooks/recovery.py``) to a WP0b ``SimDriver``: an observer
    (every step, after the tasks) so a lock beats any task's gait command before the next gait tick; arms the hook.
    Motion: ``stop`` -> gait.stop(); ``release`` -> gait.release(); walk -> gait.set_command(hold=True); step height
    through ``gait.set_params``. ``power_off`` -> enter_hold("fall"); ``stand_up`` -> leave_hold(); ``arm_park`` ->
    the arm targets go to the home (PARK) pose."""

    def __init__(self, drv, arm: bool = True):
        self.drv = drv
        self.applied_motion = 0
        self.lock_stops = 0
        self.stand_ups = 0
        self._step_h_default = drv.gait.params.step_height_scale
        self._step_h = self._step_h_default
        self._parked = False
        self._last_seq = None
        self._rel = None
        if arm:
            drv.bus.values["recovery_arm"] = True
        drv.add_observer(self._observe)

    def _set_step_height(self, sh: Optional[float]):
        from atharv_nandi.core.locomotion_logic import GaitParams
        want = self._step_h_default if sh is None else float(sh)
        if abs(want - self._step_h) > 1e-9:
            p = self.drv.gait.params
            self.drv.gait.set_params(GaitParams(p.speed_scale, want, p.freq_hz, p.balance_gain))
            self._step_h = want

    def _observe(self, drv):
        drv.bus.values["gait_cmd"] = tuple(float(c) for c in drv.gait.cmd)
        rec = drv.bus.values.get("recovery")
        if not isinstance(rec, dict):
            return
        act = rec.get("actions") or {}
        seq = rec.get("seq")
        if act.get("power_off") and not drv.holding:
            drv.enter_hold("fall")
        if act.get("stand_up") and drv.holding and seq != self._last_seq:
            drv.leave_hold()
            self.stand_ups += 1
        self._last_seq = seq
        if act.get("arm_park"):
            if not self._parked:
                from atharv_nandi.core import model_layout as mlay
                drv.command(list(drv.home_ctrl[mlay.ARM_CTRL]))
                self._parked = True
        else:
            self._parked = False
        mo = act.get("motion")
        g = drv.gait
        if mo is not None and not drv.holding:
            # recovery owns the gait: whatever a task commanded this step is overridden before the next gait tick
            self._set_step_height(mo.get("step_height"))
            if mo.get("stop"):
                if g.is_active or any(abs(c) > 0 for c in g.cmd):
                    g.stop()
                    self.lock_stops += 1
                self._rel = None
            elif mo.get("release"):
                self._stand(g)
            else:
                g.set_command(mo.get("vx", 0.0), mo.get("vy", 0.0), mo.get("yaw", 0.0), hold=True)
                self.applied_motion += 1
                self._rel = None
        else:
            self._rel = None
            if mo is None:
                self._set_step_height(None)
            if act.get("lock") and not drv.holding and (g.is_active or any(abs(c) > 1e-9 for c in g.cmd)):
                g.stop()
                self.lock_stops += 1

    def _stand(self, g):
        """Finish the current gait cycle and stand, even if a task re-commands the gait every step: the remaining
        phase and the command of the first release are kept (a task's re-hold is undone)."""
        if self._rel is None:
            if g.is_held:
                g.release()
            self._rel = (tuple(g.cmd), g.remaining_phase)
            return
        cmd0, left = self._rel
        if g.is_held or tuple(g.cmd) != cmd0:
            g.is_held = False
            g.remaining_phase = min(left, g._phase_to_cycle_end()) if left > 0 else 0.0
            if left <= 0.0:
                g.stop()
                self.lock_stops += 1
            else:
                g.cmd = cmd0
        self._rel = (cmd0, g.remaining_phase)


def seeded_noise(seed: int):
    """Gaussian noise source for the assisted relative trunk observations (reproducible per seed)."""
    rng = random.Random(int(seed) + 2801)
    return lambda: rng.gauss(0.0, LANDMARK_SIGMA_M)
