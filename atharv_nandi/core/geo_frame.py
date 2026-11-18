"""Site frame for Atharv Nandi missions: WGS84 (EPSG:4326) <-> local ENU metres, zone validation.

Pure Python (no ROS, no MuJoCo). The sim world frame IS the local ENU frame of the site origin:
    x = East (m), y = North (m), z = Up (m), origin = (lat0, lon0, alt0) from config/missions/site_sim.yaml.

Conversion is a local tangent-plane (equirectangular) approximation on the WGS84 ellipsoid: east and
north scale by the prime-vertical (N) and meridional (M) radii of curvature at lat0. It is an exact
inverse pair, so a round trip is lossless to float precision; against a full geodetic -> ECEF -> ENU
transform the horizontal error stays at the centimetre level within 500 m (tests/test_geo_frame.py
measures it). Good enough for a field a few hundred metres across; not for kilometre-scale zones.
"""
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from atharv_nandi import paths

WGS84_A = 6378137.0                 # semi-major axis (m)
WGS84_F = 1.0 / 298.257223563       # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared

DEFAULT_SITE_FILE = paths.config_path("missions", "site_sim.yaml")

Point = Tuple[float, float]


class ZoneError(ValueError):
    """A zone polygon is malformed; str(exc) says why (reason code 'zone_invalid')."""


@dataclass(frozen=True)
class SiteOrigin:
    lat0: float
    lon0: float
    alt0: float = 0.0
    name: str = "site"

    def __post_init__(self):
        for v in (self.lat0, self.lon0, self.alt0):
            if not math.isfinite(v):
                raise ValueError("site origin must be finite")
        if not (-89.0 < self.lat0 < 89.0) or not (-180.0 <= self.lon0 <= 180.0):
            raise ValueError("site origin out of range (|lat| < 89, |lon| <= 180)")

    @property
    def radii(self) -> Tuple[float, float]:
        """(M, N): meridional and prime-vertical radii of curvature at lat0 (m)."""
        s = math.sin(math.radians(self.lat0))
        w = math.sqrt(1.0 - WGS84_E2 * s * s)
        n = WGS84_A / w
        m = WGS84_A * (1.0 - WGS84_E2) / (w ** 3)
        return m, n

    def to_local(self, lat: float, lon: float, alt: Optional[float] = None) -> Tuple[float, float, float]:
        """WGS84 degrees -> local ENU metres (x east, y north, z up)."""
        m, n = self.radii
        dlon = (lon - self.lon0 + 180.0) % 360.0 - 180.0
        x = math.radians(dlon) * (n + self.alt0) * math.cos(math.radians(self.lat0))
        y = math.radians(lat - self.lat0) * (m + self.alt0)
        z = 0.0 if alt is None else alt - self.alt0
        return x, y, z

    def to_wgs84(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float, float]:
        """Local ENU metres -> (lat, lon, alt) in degrees / metres. Exact inverse of to_local."""
        m, n = self.radii
        lat = self.lat0 + math.degrees(y / (m + self.alt0))
        lon = self.lon0 + math.degrees(x / ((n + self.alt0) * math.cos(math.radians(self.lat0))))
        lon = (lon + 180.0) % 360.0 - 180.0
        return lat, lon, self.alt0 + z


# ------------------------------------------------------------------ reference (tests only)
def geodetic_to_ecef(lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
    la, lo = math.radians(lat), math.radians(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(la) ** 2)
    return ((n + alt) * math.cos(la) * math.cos(lo),
            (n + alt) * math.cos(la) * math.sin(lo),
            (n * (1.0 - WGS84_E2) + alt) * math.sin(la))


def exact_enu(origin: SiteOrigin, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
    """Rigorous geodetic -> ENU (via ECEF), used to measure the approximation error."""
    x0, y0, z0 = geodetic_to_ecef(origin.lat0, origin.lon0, origin.alt0)
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    dx, dy, dz = x - x0, y - y0, z - z0
    la, lo = math.radians(origin.lat0), math.radians(origin.lon0)
    e = -math.sin(lo) * dx + math.cos(lo) * dy
    nn = -math.sin(la) * math.cos(lo) * dx - math.sin(la) * math.sin(lo) * dy + math.cos(la) * dz
    u = math.cos(la) * math.cos(lo) * dx + math.cos(la) * math.sin(lo) * dy + math.sin(la) * dz
    return e, nn, u


# ------------------------------------------------------------------ site config
def _parse_scalar(text: str):
    t = text.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        return t[1:-1]
    if t.startswith("["):  # inline flow list of numbers, e.g. safety.yaml's geofence polygon
        try:
            return json.loads(t)
        except ValueError:
            return t
    if t in ("true", "false"):
        return t == "true"
    try:
        return float(t) if any(c in t for c in ".eE") else int(t)
    except ValueError:
        return t


def load_flat_yaml(path: str) -> dict:
    """Read a YAML file with PyYAML when available, else a minimal 'key: value' / nested-map reader.

    The fallback understands what config/missions/site_sim.yaml uses (nested maps by indentation,
    scalars, comments); CI's minimal requirements do not ship PyYAML.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val.strip() == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            parent[key] = _parse_scalar(val)
    return root


def load_site(path: Optional[str] = None) -> Tuple[SiteOrigin, dict]:
    """(origin, full config dict) from site_sim.yaml (env NANDI_SITE_FILE overrides the path)."""
    path = path or os.environ.get("NANDI_SITE_FILE") or DEFAULT_SITE_FILE
    cfg = load_flat_yaml(path)
    o = cfg.get("origin", {})
    origin = SiteOrigin(float(o["lat"]), float(o["lon"]), float(o.get("alt", 0.0)), str(cfg.get("name", "site")))
    return origin, cfg


# ------------------------------------------------------------------ polygons
def _finite_pair(p) -> Point:
    try:
        x, y = float(p[0]), float(p[1])
    except (TypeError, ValueError, IndexError, KeyError):
        raise ZoneError("vertex is not a coordinate pair")
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ZoneError("vertex is NaN or infinite")
    return x, y


def validate_ring(ring: Sequence, closed_required: bool = True) -> List[Point]:
    """Check a linear ring (GeoJSON rules): >= 4 positions, first == last, finite, non-zero area.

    Returns the ring as a list of (a, b) pairs WITHOUT the closing repeat.
    """
    if not isinstance(ring, (list, tuple)):
        raise ZoneError("ring is not a list")
    pts = [_finite_pair(p) for p in ring]
    if closed_required:
        if len(pts) < 4:
            raise ZoneError(f"ring has {len(pts)} positions, needs >= 4 (closed)")
        if pts[0] != pts[-1]:
            raise ZoneError("ring is not closed (first position != last)")
        pts = pts[:-1]
    else:
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3:
            raise ZoneError(f"polygon has {len(pts)} distinct vertices, needs >= 3")
    if len(set(pts)) < 3:
        raise ZoneError("ring has fewer than 3 distinct vertices")
    if abs(polygon_area(pts)) < 1e-12:
        raise ZoneError("ring has zero area")
    return pts


def polygon_area(pts: Sequence[Point]) -> float:
    """Signed shoelace area (positive = counter-clockwise)."""
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def point_in_polygon(x: float, y: float, pts: Sequence[Point]) -> bool:
    """Even-odd ray cast; points exactly on an edge may go either way."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y):
            xc = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < xc:
                inside = not inside
        j = i
    return inside


def polygon_bounds(pts: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _geojson_ring(zone: dict) -> Sequence:
    """Outer ring of a GeoJSON Polygon / Feature / single-feature FeatureCollection."""
    t = zone.get("type")
    if t == "FeatureCollection":
        feats = zone.get("features") or []
        if len(feats) != 1:
            raise ZoneError("FeatureCollection must hold exactly one Polygon feature")
        return _geojson_ring(feats[0])
    if t == "Feature":
        geom = zone.get("geometry")
        if not isinstance(geom, dict):
            raise ZoneError("Feature has no geometry")
        return _geojson_ring(geom)
    if t != "Polygon":
        raise ZoneError(f"zone type {t!r} is not a Polygon")
    coords = zone.get("coordinates")
    if not isinstance(coords, list) or not coords:
        raise ZoneError("Polygon has no coordinates")
    return coords[0]  # holes are ignored (documented)


def zone_to_local(zone, origin: SiteOrigin) -> List[Point]:
    """Validate a mission zone and return its outer ring in local metres (no closing repeat).

    Accepted forms:
        GeoJSON Polygon (or Feature / FeatureCollection of one), EPSG:4326 [lon, lat] order
        {"local_polygon": [[x, y], ...]} in site metres (closing repeat optional)
    Raises ZoneError (reason 'zone_invalid').
    """
    if not isinstance(zone, dict):
        raise ZoneError("zone must be a JSON object")
    if "local_polygon" in zone:
        return validate_ring(zone["local_polygon"], closed_required=False)
    ring = validate_ring(_geojson_ring(zone), closed_required=True)
    for lon, lat in ring:
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise ZoneError("coordinate out of EPSG:4326 range (expects [lon, lat])")
    return [tuple(origin.to_local(lat, lon)[:2]) for lon, lat in ring]


def local_to_geojson(pts: Iterable[Point], origin: SiteOrigin) -> dict:
    """Local polygon -> closed GeoJSON Polygon in EPSG:4326 ([lon, lat], 9 decimals ~ 0.1 mm)."""
    ring = []
    for x, y in pts:
        lat, lon, _ = origin.to_wgs84(x, y)
        ring.append([round(lon, 9), round(lat, 9)])
    ring.append(list(ring[0]))
    return {"type": "Polygon", "coordinates": [ring]}
