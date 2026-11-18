"""TREE 360 orbit planner and closed-loop side-mode orbit controller (WP25; ORCHARD-MVP-SPEC 2.1 "Orbit mode",
5.1, 5.3 TREE 360 rows; MISSION-PRESETS 2).

Pure Python, no MuJoCo, no ROS.

Geometry. The orbit reference point is the **mast axis** (``mast_base`` at trunk (-0.30, 0); the ring and crown
lenses sit on it), so the radius ``R`` is lens-to-trunk-axis and the stand-off is ``d = R - canopy_r`` (lens to
canopy surface, OS 2.3). ``r1 = clamp(canopy_r + d, 1.2, 3.0)`` (MISSION-PRESETS 0.8). The robot walks **CCW with
the tree on its left** (side mode only, OS 2.1): at tree azimuth ``theta`` the reference point is at
``tree + R (cos theta, sin theta)`` and the steady heading is ``theta + 90 deg - atan(mast_x / R)`` (``station_heading``:
the reference point sits behind the turning centre, so on the circle it crabs ~9 deg outward of the heading).

Full vs two-sided (OS rule 12, MP 2 step 5, OQ-06 A adaptive). On a 1 deg grid of azimuths the robot hull (a disc of
``HULL_R_M`` around the reference point, ASSUMED: the hexapod's legs reach ~0.45 m from the trunk centre, the mast is
0.30 m behind it) is **blocked** where a neighbour canopy (registry disc ``canopy_r``) comes within
``ARC_CLEAR_M`` (0.3 m) plus ``ARC_PLAN_TOL_M`` (0.10 m: the closed loop holds the radius within it, so the flown arc
keeps 0.3 m) of it, or the point leaves the geofence; WP5 LiDAR clusters in the annulus count as neighbours
(``clusters_as_neighbours``). No blocked azimuth (and the registry says
``orbit_clear``) -> ``full``: ``n`` stations every 360/n deg and a closing arc back to the start (360 deg, the
first station's frames are the overlap). Otherwise ``two_sided``: per alley side (registry ``alley_sides``) the longest free run within +-90 deg of
the side's direction (``side_arcs``) gets ``max(2, round(n * len / 360) + 1)`` stations spread end to end; ``gap_deg``
= 360 - covered arc. Runs shorter than ``MIN_ARC_DEG`` are not used; nothing left -> ``blocked``. Between the arcs
``transit_path`` (grid A*) walks around the row end.

Controller (OS 5.1)::

    omega = v / R_cmd + k_r (R_meas - R_cmd) + k_b * (bearing_err - b_ff),   b_ff = atan(mast_x / R_cmd)

``bearing_err`` = (bearing of the tree centre from the reference point, relative to the heading) - 90 deg; positive
when the robot heads outward. With ``e = R - R_cmd`` the linearised loop is ``e' = v b``, ``b' = -(k_r + v/R^2) e -
k_b b``; ``k_r`` 1.6 rad/s/m and ``k_b`` 0.7 1/s give w_n ~0.5 rad/s, zeta ~0.7 at 0.15 m/s (computed). The gait takes
normalised commands; ``GaitMap`` turns (v, omega) into (vx, yaw) with the MEASURED nominal gains (vx 0.55 ->
0.153 m/s, yaw 0.08 -> 0.085 rad/s while walking, orchard_demo flat) plus two slow integrators on the measured
speed and yaw rate, so slip on rough ground and basins does not bias the radius.

Radius rules (OS 5.3): |e| > 0.15 m -> slow to 0.10 m/s; |e| > 0.30 m -> stop, re-enter at the nearest station
(at most ``REENTRY_MAX`` 2 times per orbit), then mark the gap and go on.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

R_MIN_M, R_MAX_M = 1.2, 3.0          # MP 0.8 r1 clamp
STANDOFF_DEFAULT_M = 0.9             # MP 0.8 standoff_m default (OS 1.2 step 2: r1 = 1.9 m on OT-1)
MAST_X_M = -0.30                     # mast_base in the trunk frame (OS 2.3)
HULL_R_M = 0.55                      # ASSUMED hull disc around the reference point (legs reach ~0.45 m + mast)
ARC_CLEAR_M = 0.30                   # OS 5.3: neighbour canopy < 0.3 m from the hull blocks the arc
ARC_PLAN_TOL_M = 0.10               # planned on top of it: the closed loop holds the radius within 0.10 m (O-02)
MIN_ARC_DEG = 30.0                   # free arcs shorter than this are not orbited
ANNULUS_HALF_M = 0.60                # MP 2 precondition: annulus R +- 0.6 m clear in map and LiDAR
SIDE_AZ_DEG = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}
GRID_DEG = 1.0
V_ORBIT_MPS = 0.15                   # MP 0.8 v_mps default and max (crown up)
V_SLOW_MPS = 0.10                    # OS 5.3: radius error > 0.15 m
R_ERR_SLOW_M = 0.15
R_ERR_STOP_M = 0.30
REENTRY_MAX = 2
K_R = 1.6                            # rad/s per m of radius error
K_B = 0.7                            # 1/s on the bearing error
OMEGA_MAX = 0.6                      # rad/s clamp of the controller output
# gait map (normalised cmd), MEASURED nominal gains
V_PER_CMD = 0.27                     # m/s per unit vx (vx 0.6 -> ~0.16 m/s on flat, orchard_demo)
W_PER_CMD = 1.03                     # rad/s per unit yaw (in-place yaw 1 -> 59 deg/s, tripod_gait)
VX_CMD_MAX, YAW_CMD_MAX = 0.9, 0.8
KI_V, KI_W = 0.6, 0.6                # integrator gains (cmd per (m/s) s, cmd per (rad/s) s)
INT_V_MAX, INT_W_MAX = 0.3, 0.25
STOP_LEAD_S = 0.25                   # stop this long before the station (gait ramp-down)


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def ref_point(trunk_xy: Sequence[float], heading: float, mast_x: float = MAST_X_M) -> Tuple[float, float]:
    """World xy of the orbit reference point (mast axis) from the trunk centre and heading."""
    return (float(trunk_xy[0]) + mast_x * math.cos(heading), float(trunk_xy[1]) + mast_x * math.sin(heading))


def trunk_from_ref(ref_xy: Sequence[float], heading: float, mast_x: float = MAST_X_M) -> Tuple[float, float]:
    return (float(ref_xy[0]) - mast_x * math.cos(heading), float(ref_xy[1]) - mast_x * math.sin(heading))


def orbit_radius(canopy_r: float, standoff_m: float = STANDOFF_DEFAULT_M) -> float:
    """MP 0.8: r1 = clamp(canopy_r + standoff_m, 1.2, 3.0)."""
    return min(R_MAX_M, max(R_MIN_M, float(canopy_r) + float(standoff_m)))


def point_on_orbit(centre: Sequence[float], R: float, theta: float) -> Tuple[float, float]:
    return (float(centre[0]) + R * math.cos(theta), float(centre[1]) + R * math.sin(theta))


def heading_at(theta: float) -> float:
    """CCW, tree on the left: tangent heading at azimuth theta."""
    return wrap(theta + math.pi / 2.0)


def station_heading(theta: float, R: float, mast_x: float = MAST_X_M) -> float:
    """Steady orbit heading at a station: the tangent turned in by the reference-point crab atan(mast_x / R)
    (``OrbitController.b_ff``), so the mast axis moves along the circle."""
    return wrap(heading_at(theta) - math.atan2(mast_x, R))


def _side_of_az(az: float, sides: Sequence[str]) -> str:
    from atharv_nandi.core.coverage import side_of_az
    return side_of_az(az, sides)


def _in_poly(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


@dataclass
class Station:
    k: int
    theta: float                 # rad, tree azimuth of the reference point (unwrapped along the orbit)
    xy: Tuple[float, float]      # reference point (mast axis) world xy
    heading: float               # rad
    side: str                    # "full" or "A" / "B" (two-sided arcs)
    capture: bool = True         # False for the closing point of a full orbit

    def trunk_xy(self) -> Tuple[float, float]:
        return trunk_from_ref(self.xy, self.heading)

    def as_dict(self) -> dict:
        return {"k": self.k, "theta_deg": round(math.degrees(self.theta), 2), "xy": [round(v, 4) for v in self.xy],
                "heading_deg": round(math.degrees(self.heading), 2), "side": self.side, "capture": self.capture}


@dataclass
class OrbitPlan:
    tree_id: str
    centre: Tuple[float, float]
    R: float
    standoff_m: float
    mode: str                                  # full | two_sided | blocked
    arcs: List[Tuple[float, float]]            # (theta_start, theta_end) rad, CCW, theta_end > theta_start
    stations: List[List[Station]]              # per arc
    gap_deg: float
    blocked_by: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def n_capture(self) -> int:
        return sum(1 for arc in self.stations for s in arc if s.capture)

    def all_stations(self) -> List[Station]:
        return [s for arc in self.stations for s in arc]

    def arc_length_m(self) -> float:
        return sum(self.R * (b - a) for a, b in self.arcs)

    def as_dict(self) -> dict:
        return {"tree_id": self.tree_id, "centre": [round(v, 4) for v in self.centre], "R_m": round(self.R, 4),
                "standoff_m": self.standoff_m, "mode": self.mode, "gap_deg": round(self.gap_deg, 1),
                "arcs_deg": [[round(math.degrees(a), 1), round(math.degrees(b), 1)] for a, b in self.arcs],
                "stations": [[s.as_dict() for s in arc] for arc in self.stations], "blocked_by": self.blocked_by,
                "reason": self.reason}


def blocked_azimuths(centre: Sequence[float], R: float, neighbours: Sequence[dict], hull_r: float = HULL_R_M,
                     clear_m: float = ARC_CLEAR_M, geofence: Optional[Sequence[Sequence[float]]] = None,
                     grid_deg: float = GRID_DEG) -> Tuple[List[bool], List[str]]:
    """Per grid azimuth (0 = +x, CCW): is the hull there within ``clear_m`` of a neighbour canopy (or outside the
    geofence)? ``neighbours``: dicts with ``tree_id``, ``xy`` and ``canopy_r_m``. Also returns who blocks."""
    n = int(round(360.0 / grid_deg))
    out, who = [], set()
    for i in range(n):
        th = math.radians(i * grid_deg)
        x, y = point_on_orbit(centre, R, th)
        b = False
        for nb in neighbours:
            gap = math.hypot(x - nb["xy"][0], y - nb["xy"][1]) - float(nb["canopy_r_m"]) - hull_r
            if gap < clear_m:
                b = True
                who.add(str(nb.get("tree_id", "?")))
        if geofence and not _in_poly(x, y, geofence):
            b = True
            who.add("geofence")
        out.append(b)
    return out, sorted(who)


def free_arcs(blocked: Sequence[bool], grid_deg: float = GRID_DEG) -> List[Tuple[float, float]]:
    """Contiguous free azimuth intervals (deg, start < end, may run past 360), longest first."""
    n = len(blocked)
    if not any(blocked):
        return [(0.0, 360.0)]
    if all(blocked):
        return []
    start = next(i for i in range(n) if blocked[i])          # begin scanning right after a blocked cell
    arcs, run = [], None
    for j in range(1, n + 1):
        i = (start + j) % n
        if not blocked[i]:
            if run is None:
                run = start + j
        elif run is not None:
            arcs.append(((run - 0.5) * grid_deg, (start + j - 0.5) * grid_deg))
            run = None
    if run is not None:
        arcs.append(((run - 0.5) * grid_deg, (start + n + 0.5) * grid_deg))
    arcs.sort(key=lambda a: a[1] - a[0], reverse=True)
    return arcs


def side_arcs(blocked: Sequence[bool], sides: Sequence[str], grid_deg: float = GRID_DEG,
              min_deg: float = MIN_ARC_DEG) -> List[Tuple[float, float, str]]:
    """Two-sided arcs (OS rule 12, OQ-06 A): per alley side, the longest free run of grid azimuths within +-90 deg
    of the side's direction (``+x`` 0, ``+y`` 90, ``-x`` 180, ``-y`` 270 deg). ``(start_deg, end_deg, side)``, CCW
    (end > start, may pass 360), runs shorter than ``min_deg`` dropped, ordered by start azimuth."""
    n = len(blocked)
    out = []
    for s in sides:
        c = SIDE_AZ_DEG[s]
        idx = [i for i in range(n) if abs((i * grid_deg - c + 180.0) % 360.0 - 180.0) <= 90.0 - 0.5 * grid_deg]
        # idx is one contiguous circular window starting at c - 90 (walk it in CCW order)
        start = int(round(((c - 90.0) % 360.0) / grid_deg))
        order = [(start + j) % n for j in range(n) if (start + j) % n in set(idx)]
        best, run = None, []
        for i in order + [None]:
            if i is not None and not blocked[i]:
                run.append(i)
                continue
            if run and (best is None or len(run) > len(best)):
                best = run
            run = []
        if best and (len(best) - 1) * grid_deg >= min_deg:
            a = best[0] * grid_deg
            b = a + (len(best) - 1) * grid_deg
            out.append((a, b, s))
    out.sort(key=lambda x: x[0])
    return out


def plan_orbit(tree: dict, neighbours: Sequence[dict] = (), standoff_m: float = STANDOFF_DEFAULT_M,
               n_stations: int = 8, arc_mode: str = "auto", hull_r: float = HULL_R_M,
               geofence: Optional[Sequence[Sequence[float]]] = None,
               start_theta: Optional[float] = None) -> OrbitPlan:
    """Plan one radius of TREE 360 around ``tree`` (``tree_id``, ``xy``, ``canopy_r_m``, ``orbit_clear``).

    ``arc_mode`` auto | full | two_sided (MP 0.8). ``full`` is refused (falls back to two-sided) when the annulus is
    blocked: coverage is measured, never assumed (OS rule 12). ``start_theta`` (rad) picks the full orbit's entry
    azimuth (default: the azimuth facing the nearest alley, i.e. -90 deg, the robot coming from y < tree y)."""
    if arc_mode not in ("auto", "full", "two_sided"):
        raise ValueError(f"arc_mode {arc_mode!r}")
    n_stations = int(n_stations)
    if not 4 <= n_stations <= 16:
        raise ValueError("stations must be 4..16 (MP 0.8)")
    centre = (float(tree["xy"][0]), float(tree["xy"][1]))
    R = orbit_radius(tree["canopy_r_m"], standoff_m)
    blocked, who = blocked_azimuths(centre, R, neighbours, hull_r, ARC_CLEAR_M + ARC_PLAN_TOL_M, geofence)
    clear = not any(blocked)
    tid = str(tree.get("tree_id", "?"))
    full = clear and arc_mode != "two_sided" and (bool(tree.get("orbit_clear", True)) or arc_mode == "full")
    if full:
        th0 = -math.pi / 2.0 if start_theta is None else float(start_theta)
        sts = []
        for k in range(n_stations + 1):
            th = th0 + 2.0 * math.pi * k / n_stations
            sts.append(Station(k, th, point_on_orbit(centre, R, th), station_heading(th, R), "full", k < n_stations))
        return OrbitPlan(tid, centre, R, float(standoff_m), "full", [(th0, th0 + 2 * math.pi)], [sts], 0.0, who)
    alley = [s for s in (tree.get("alley_sides") or []) if s in SIDE_AZ_DEG] or ["-y", "+y"]
    use = side_arcs(blocked, alley)
    if not use:
        return OrbitPlan(tid, centre, R, float(standoff_m), "blocked", [], [], 360.0, who, "orbit_blocked")
    arcs, stations, k = [], [], 0
    for (a, b, side) in use:
        ta, tb = math.radians(a), math.radians(b)
        n = max(2, int(round(n_stations * (b - a) / 360.0)) + 1)
        sts = []
        for j in range(n):
            th = ta + (tb - ta) * j / (n - 1)
            sts.append(Station(k, th, point_on_orbit(centre, R, th), station_heading(th, R), side, True))
            k += 1
        arcs.append((ta, tb))
        stations.append(sts)
    gap = 360.0 - sum(math.degrees(b - a) for a, b in arcs)
    reason = "orbit_blocked" if not clear else ""
    return OrbitPlan(tid, centre, R, float(standoff_m), "two_sided", arcs, stations, gap, who, reason)


def entry_xy(entry: dict) -> Tuple[float, float]:
    """A registry entry's trunk xy: top-level ``x``/``y`` (WP23 default registry) or ``geo`` (WP4 field trees)."""
    if "x" in entry and "y" in entry:
        return float(entry["x"]), float(entry["y"])
    g = entry.get("geo") or {}
    return float(g["x"]), float(g["y"])


def neighbours_of(entry: dict, registry: Sequence[dict], within_m: float = 6.0) -> List[dict]:
    """Registry neighbours of ``entry`` as planner dicts (``tree_id``, ``xy``, ``canopy_r_m``)."""
    x, y = entry_xy(entry)
    out = []
    for e in registry:
        if e.get("tree_id") == entry.get("tree_id"):
            continue
        ex, ey = entry_xy(e)
        if math.hypot(ex - x, ey - y) <= within_m:
            out.append({"tree_id": e["tree_id"], "xy": (ex, ey), "canopy_r_m": float(e.get("canopy_r_m") or 0.5)})
    return out


def tree_of(entry: dict) -> dict:
    return {"tree_id": entry["tree_id"], "xy": entry_xy(entry), "canopy_r_m": float(entry["canopy_r_m"]),
            "orbit_clear": bool(entry.get("orbit_clear", False)), "height_m": float(entry.get("height_m") or 2.0),
            "alley_sides": list(entry.get("alley_sides") or ["-y", "+y"]),
            "basin_r_m": entry.get("basin_r_m"), "crop": entry.get("crop", "")}


def clusters_as_neighbours(clusters: Sequence[dict], centre: Sequence[float], R: float,
                           band_m: float = ANNULUS_HALF_M) -> List[dict]:
    """WP5 LiDAR clusters (non-map returns, ``{lo, hi}`` world boxes) that reach into the annulus ``R +- band_m``
    as planner obstacles (a disc around the box): MP 2 precondition "annulus clear in map **and** LiDAR"."""
    out = []
    for k, c in enumerate(clusters or ()):
        try:
            lo, hi = c["lo"], c["hi"]
            cx, cy = (float(lo[0]) + float(hi[0])) / 2.0, (float(lo[1]) + float(hi[1])) / 2.0
            r = math.hypot(float(hi[0]) - float(lo[0]), float(hi[1]) - float(lo[1])) / 2.0
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        dist = math.hypot(cx - float(centre[0]), cy - float(centre[1]))
        if R - band_m - r <= dist <= R + band_m + r:
            out.append({"tree_id": f"lidar_{k}", "xy": (cx, cy), "canopy_r_m": r})
    return out


def basins_of(registry: Sequence[dict]) -> List[Tuple[float, float, float]]:
    """``(x, y, basin_r_m)`` of every registry tree with a basin (WP6 ``gate_crown_path`` input)."""
    out = []
    for e in registry:
        br = e.get("basin_r_m")
        if br:
            x, y = entry_xy(e)
            out.append((x, y, float(br)))
    return out


def crown_path_ok(plan: "OrbitPlan", ext: float, basins: Sequence[Tuple[float, float, float]]):
    """WP6 ``gate_crown_path`` on the planned arcs (trunk centre points every ~0.25 m): crown-up walking must keep
    the hull outside every basin rim footprint (else the orbit runs stop-and-go)."""
    from atharv_nandi.core.stability import gate_crown_path
    path = []
    for a, b in plan.arcs:
        n = max(2, int(abs(b - a) * plan.R / 0.25))
        for i in range(n + 1):
            th = a + (b - a) * i / n
            path.append(trunk_from_ref(point_on_orbit(plan.centre, plan.R, th), station_heading(th, plan.R)))
    return gate_crown_path(ext, path, basins)


# ====================================================================== transit (two-sided: other alley)
def transit_path(start: Sequence[float], goal: Sequence[float], obstacles: Sequence[Tuple[float, float, float]],
                 geofence: Optional[Sequence[Sequence[float]]] = None, step_m: float = 0.25,
                 margin_m: float = HULL_R_M + 0.10) -> Optional[List[Tuple[float, float]]]:
    """Waypoints (start excluded, goal included) from ``start`` to ``goal`` that keep the hull ``margin_m`` clear of
    every obstacle disc ``(x, y, r)`` and inside the geofence: 8-connected A* on a ``step_m`` grid, then
    line-of-sight shortcuts. None when there is no path. Used between the two arcs of a two-sided TREE 360
    (MP 2 step 5: "repeat from the other alley"), crown stowed."""
    import heapq
    obst = [(float(x), float(y), float(r) + margin_m) for x, y, r in obstacles]

    def free(x: float, y: float) -> bool:
        if geofence and not _in_poly(x, y, geofence):
            return False
        return all((x - ox) ** 2 + (y - oy) ** 2 >= orr * orr for ox, oy, orr in obst)

    def seg_free(p, q) -> bool:
        n = max(1, int(math.hypot(q[0] - p[0], q[1] - p[1]) / (step_m / 2.0)))
        return all(free(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(n + 1))

    s = (float(start[0]), float(start[1]))
    g = (float(goal[0]), float(goal[1]))
    if seg_free(s, g):
        return [g]
    if geofence:
        xs = [p[0] for p in geofence]
        ys = [p[1] for p in geofence]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    else:
        x0, x1 = min(s[0], g[0]) - 8.0, max(s[0], g[0]) + 8.0
        y0, y1 = min(s[1], g[1]) - 8.0, max(s[1], g[1]) + 8.0
    nx, ny = int((x1 - x0) / step_m) + 1, int((y1 - y0) / step_m) + 1

    def cell(p):
        return (min(nx - 1, max(0, int(round((p[0] - x0) / step_m)))),
                min(ny - 1, max(0, int(round((p[1] - y0) / step_m)))))

    def pos(c):
        return (x0 + c[0] * step_m, y0 + c[1] * step_m)

    cs, cg = cell(s), cell(g)
    ok_cache: Dict[Tuple[int, int], bool] = {}

    def ok(c):
        v = ok_cache.get(c)
        if v is None:
            v = ok_cache[c] = free(*pos(c))
        return v

    openq = [(0.0, 0.0, cs)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cost = {cs: 0.0}
    nbr = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    found = False
    while openq:
        _, c0, c = heapq.heappop(openq)
        if c == cg:
            found = True
            break
        if c0 > cost.get(c, math.inf) + 1e-9:
            continue
        for dx, dy, w in nbr:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < nx and 0 <= n[1] < ny):
                continue
            if n != cg and not ok(n):
                continue
            nc = c0 + w
            if nc < cost.get(n, math.inf) - 1e-9:
                cost[n] = nc
                came[n] = c
                h = math.hypot(cg[0] - n[0], cg[1] - n[1])
                heapq.heappush(openq, (nc + h, nc, n))
    if not found:
        return None
    cells = [cg]
    while cells[-1] != cs:
        cells.append(came[cells[-1]])
    pts = [s] + [pos(c) for c in reversed(cells[1:-1])] + [g]
    out, i = [], 0                       # line-of-sight shortcuts
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not seg_free(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def path_length(start: Sequence[float], waypoints: Sequence[Sequence[float]]) -> float:
    total, p = 0.0, start
    for q in waypoints:
        total += math.hypot(q[0] - p[0], q[1] - p[1])
        p = q
    return total


# ====================================================================== closed loop
def orbit_error(ref_xy: Sequence[float], heading: float, centre: Sequence[float], R_cmd: float,
                R_meas: Optional[float] = None) -> Tuple[float, float, float]:
    """(radius error R_meas - R_cmd, bearing error rad, tree azimuth theta of the reference point)."""
    dx, dy = float(centre[0]) - float(ref_xy[0]), float(centre[1]) - float(ref_xy[1])
    R = math.hypot(dx, dy) if R_meas is None else float(R_meas)
    bearing = wrap(math.atan2(dy, dx) - heading)            # tree bearing relative to the heading (left = +90)
    theta = math.atan2(-dy, -dx)
    return R - R_cmd, wrap(bearing - math.pi / 2.0), theta


def omega_cmd(v: float, R_cmd: float, e_r: float, bearing_err: float, k_r: float = K_R, k_b: float = K_B,
              omega_max: float = OMEGA_MAX) -> float:
    """OS 5.1 side-mode law: omega = v / R_cmd + k_r e_r + k_b bearing_err (clamped)."""
    w = v / R_cmd + k_r * e_r + k_b * bearing_err
    return max(-omega_max, min(omega_max, w))


def speed_for(e_r: float, v_nominal: float = V_ORBIT_MPS) -> float:
    """OS 5.3: |e| > 0.15 m -> 0.10 m/s; |e| > 0.30 m -> 0 (stop and re-enter)."""
    a = abs(e_r)
    if a > R_ERR_STOP_M:
        return 0.0
    if a > R_ERR_SLOW_M:
        return min(v_nominal, V_SLOW_MPS)
    return v_nominal


class GaitMap:
    """(v m/s, omega rad/s) -> normalised gait (vx, yaw), with integrators on the measured speed and yaw rate."""

    def __init__(self, v_per_cmd: float = V_PER_CMD, w_per_cmd: float = W_PER_CMD):
        self.v_per_cmd, self.w_per_cmd = float(v_per_cmd), float(w_per_cmd)
        self.iv = self.iw = 0.0

    def reset(self):
        self.iv = self.iw = 0.0

    def __call__(self, v: float, w: float, v_meas: Optional[float], w_meas: Optional[float],
                 dt: float) -> Tuple[float, float]:
        if v > 1e-6 and v_meas is not None:
            self.iv = max(-INT_V_MAX, min(INT_V_MAX, self.iv + KI_V * (v - v_meas) * dt))
        if v > 1e-6 and w_meas is not None:
            self.iw = max(-INT_W_MAX, min(INT_W_MAX, self.iw + KI_W * (w - w_meas) * dt))
        vx = max(0.0, min(VX_CMD_MAX, v / self.v_per_cmd + self.iv)) if v > 1e-6 else 0.0
        yaw = max(-YAW_CMD_MAX, min(YAW_CMD_MAX, w / self.w_per_cmd + self.iw))
        return vx, yaw


@dataclass
class ArcStep:
    vx: float
    yaw: float
    v: float
    omega: float
    e_r: float
    bearing_err: float
    theta: float
    done: bool
    stop: bool          # radius error > 0.30 m: stop and re-enter


class OrbitController:
    """Drives one arc from the current pose to ``theta_end`` (unwrapped, CCW)."""

    def __init__(self, centre: Sequence[float], R_cmd: float, v_nominal: float = V_ORBIT_MPS,
                 mast_x: float = MAST_X_M):
        self.centre = (float(centre[0]), float(centre[1]))
        self.R_cmd = float(R_cmd)
        # steady-state crab of the reference point: it sits mast_x behind the turning centre, so on the circle it
        # moves at atan(mast_x omega / v) = atan(mast_x / R) off the heading; the bearing error is taken against it
        self.b_ff = math.atan2(float(mast_x), self.R_cmd)
        self.v_nominal = min(float(v_nominal), V_ORBIT_MPS)
        self.gait = GaitMap()
        self.theta_unwrapped: Optional[float] = None
        self.errors: List[float] = []

    def start(self, theta_now: float, theta_end: float):
        self.theta_unwrapped = float(theta_now)
        self.theta_end = float(theta_end)
        self.gait.reset()

    def _unwrap(self, theta: float) -> float:
        if self.theta_unwrapped is None:
            self.theta_unwrapped = theta
        self.theta_unwrapped += wrap(theta - self.theta_unwrapped)
        return self.theta_unwrapped

    def step(self, ref_xy: Sequence[float], heading: float, dt: float, R_meas: Optional[float] = None,
             v_meas: Optional[float] = None, w_meas: Optional[float] = None,
             centre: Optional[Sequence[float]] = None) -> ArcStep:
        c = self.centre if centre is None else centre
        e_r, b_err, th = orbit_error(ref_xy, heading, c, self.R_cmd, R_meas)
        th_u = self._unwrap(th)
        self.errors.append(e_r)
        v = speed_for(e_r, self.v_nominal)
        lead = self.v_nominal * STOP_LEAD_S / self.R_cmd
        done = th_u >= self.theta_end - lead
        if done or v <= 0.0:
            return ArcStep(0.0, 0.0, 0.0, 0.0, e_r, b_err, th_u, done, v <= 0.0 and not done)
        w = omega_cmd(v, self.R_cmd, e_r, wrap(b_err - self.b_ff))
        vx, yaw = self.gait(v, w, v_meas, w_meas, dt)
        return ArcStep(vx, yaw, v, w, e_r, b_err, th_u, False, False)


def rms(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else 0.0


def nearest_station(stations: Sequence[Station], theta: float, after_k: int = -1) -> Station:
    """The capture station nearest in azimuth to ``theta`` with index > ``after_k`` (re-entry point)."""
    cand = [s for s in stations if s.k > after_k] or list(stations)
    return min(cand, key=lambda s: abs(wrap(s.theta - theta)))


def arc_poses_base(ref_xy: Sequence[float], heading: float, centre: Sequence[float], R: float, theta_from: float,
                   theta_to: float, step_m: float = 0.25) -> List[List[float]]:
    """The planned arc as trunk poses (x, y, yaw) in the base frame of the current trunk pose (WP6 ``orbit.arc``)."""
    tx, ty = trunk_from_ref(ref_xy, heading)
    c, s = math.cos(heading), math.sin(heading)
    n = max(1, int(abs(theta_to - theta_from) * R / step_m))
    out = []
    for i in range(n + 1):
        th = theta_from + (theta_to - theta_from) * i / n
        h = station_heading(th, R)
        px, py = trunk_from_ref(point_on_orbit(centre, R, th), h)
        dx, dy = px - tx, py - ty
        out.append([round(dx * c + dy * s, 3), round(-dx * s + dy * c, 3), round(wrap(h - heading), 4)])
    return out


def entry_pose(plan: OrbitPlan) -> Dict[str, float]:
    """Entry station: the first station's trunk pose (MP 2 step 1: entry at r1, tangent heading)."""
    s = plan.stations[0][0]
    tx, ty = s.trunk_xy()
    return {"xy": [tx, ty], "heading_deg": math.degrees(s.heading)}
