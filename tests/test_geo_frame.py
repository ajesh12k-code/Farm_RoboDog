"""Site frame: WGS84 <-> local ENU round trip and accuracy, site config, zone validation."""
import json
import math
import os

import pytest

from atharv_nandi.core import geo_frame as gf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ORIGINS = [gf.SiteOrigin(12.5, -30.0, 0.0), gf.SiteOrigin(-41.3, 174.8, 120.0), gf.SiteOrigin(60.0, 10.0, 5.0)]


@pytest.mark.parametrize("origin", ORIGINS)
def test_round_trip_under_1cm_within_500m(origin):
    worst = 0.0
    for r in (0.0, 1.0, 50.0, 250.0, 500.0):
        for k in range(16):
            a = 2 * math.pi * k / 16
            x, y, z = r * math.cos(a), r * math.sin(a), 3.0
            lat, lon, alt = origin.to_wgs84(x, y, z)
            x2, y2, z2 = origin.to_local(lat, lon, alt)
            worst = max(worst, math.hypot(x2 - x, y2 - y), abs(z2 - z))
    assert worst < 0.01


@pytest.mark.parametrize("origin", ORIGINS)
def test_approximation_matches_rigorous_enu_within_500m(origin):
    worst = 0.0
    for k in range(24):
        a = 2 * math.pi * k / 24
        lat, lon, _ = origin.to_wgs84(500.0 * math.cos(a), 500.0 * math.sin(a))
        e, n, _ = gf.exact_enu(origin, lat, lon, origin.alt0)
        x, y, _ = origin.to_local(lat, lon)
        worst = max(worst, math.hypot(e - x, n - y))
    assert worst < 0.05  # centimetre level at the edge of a 1 km field
    print(f"worst horizontal error at 500 m: {worst * 100:.2f} cm")


def test_axes_are_east_and_north():
    o = ORIGINS[0]
    lat, lon, _ = o.to_wgs84(10.0, 0.0)
    assert lon > o.lon0 and lat == pytest.approx(o.lat0)
    lat, lon, _ = o.to_wgs84(0.0, 10.0)
    assert lat > o.lat0 and lon == pytest.approx(o.lon0)
    # one degree of latitude is ~110.6 km near 12.5 N
    assert o.to_local(o.lat0 + 1e-3, o.lon0)[1] == pytest.approx(110.6, abs=0.3)


def test_antimeridian_wrap():
    o = gf.SiteOrigin(-17.0, 179.9999)
    lat, lon, _ = o.to_wgs84(50.0, 0.0)
    assert lon < 0  # wrapped to -180..
    x, _, _ = o.to_local(lat, lon)
    assert x == pytest.approx(50.0, abs=1e-6)


def test_invalid_origin_rejected():
    with pytest.raises(ValueError):
        gf.SiteOrigin(float("nan"), 0.0)
    with pytest.raises(ValueError):
        gf.SiteOrigin(89.5, 0.0)


def test_site_config_loads_fictitious_origin():
    origin, cfg = gf.load_site(os.path.join(ROOT, "config", "missions", "site_sim.yaml"))
    assert (origin.lat0, origin.lon0) == (12.5, -30.0)
    assert cfg["basket"]["max_crops"] == 6 and cfg["energy"]["capacity_wh"] == 500.0


def test_minimal_yaml_fallback_parser_matches(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_yaml(name, *a, **k):
        if name == "yaml":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    cfg = gf.load_flat_yaml(os.path.join(ROOT, "config", "missions", "site_sim.yaml"))
    assert cfg["origin"]["lat"] == 12.5 and cfg["origin"]["lon"] == -30.0
    assert cfg["energy"]["eta"] == 0.6 and cfg["name"] == "nandi-sim-site"


def _poly(ring):
    return {"type": "Polygon", "coordinates": [ring]}


def test_zone_validation_rejects_bad_rings():
    o = ORIGINS[0]
    ok = [[-30.0, 12.5], [-29.9999, 12.5], [-29.9999, 12.5001], [-30.0, 12.5]]
    assert len(gf.zone_to_local(_poly(ok), o)) == 3
    bad = {
        "open ring": ok[:-1] + [[-30.0, 12.50001]],
        "too few": [[-30.0, 12.5], [-29.9999, 12.5], [-30.0, 12.5]],
        "nan": [[-30.0, 12.5], [float("nan"), 12.5], [-29.9999, 12.5001], [-30.0, 12.5]],
        "zero area": [[-30.0, 12.5], [-29.9999, 12.5], [-29.9998, 12.5], [-30.0, 12.5]],
        "lat/lon swapped out of range": [[12.5, -30.0], [12.5, 95.0], [12.6, 95.0], [12.5, -30.0]],
    }
    for name, ring in bad.items():
        with pytest.raises(gf.ZoneError):
            gf.zone_to_local(_poly(ring), o)
    for zone in ({"type": "Point", "coordinates": [0, 0]}, {"type": "Polygon"}, "zone", {"local_polygon": [[0, 0], [1, 1]]}):
        with pytest.raises(gf.ZoneError):
            gf.zone_to_local(zone, o)


def test_local_polygon_and_feature_forms():
    o = ORIGINS[0]
    pts = gf.zone_to_local({"local_polygon": [[0, 0], [2, 0], [2, 1], [0, 1]]}, o)
    assert pts == [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    gj = gf.local_to_geojson(pts, o)
    back = gf.zone_to_local({"type": "Feature", "geometry": gj}, o)
    assert all(math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-3 for a, b in zip(pts, back))


def test_zone_a_file_covers_the_north_row():
    origin, _ = gf.load_site(os.path.join(ROOT, "config", "missions", "site_sim.yaml"))
    with open(os.path.join(ROOT, "config", "missions", "zone_a.geojson")) as fh:
        zone = json.load(fh)
    pts = gf.zone_to_local(zone, origin)
    assert gf.point_in_polygon(1.03, 0.53, pts) and gf.point_in_polygon(3.27, 0.44, pts)
    assert not gf.point_in_polygon(0.62, 0.0, pts) and not gf.point_in_polygon(1.7, -0.5, pts)


def test_point_in_polygon_concave():
    u = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    assert gf.point_in_polygon(0.5, 2.5, u) and gf.point_in_polygon(2.5, 2.5, u)
    assert not gf.point_in_polygon(1.5, 2.0, u)
    assert gf.polygon_area(u) == pytest.approx(7.0)


def test_flat_yaml_fallback_reads_inline_lists_and_booleans(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def no_yaml(name, *a, **k):
        if name == "yaml":
            raise ImportError("no PyYAML")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_yaml)
    p = tmp_path / "s.yaml"
    p.write_text("geofence:\n  enabled: true\n  polygon: [[-1.5, -2.0], [7.0, -2.0], [7.0, 2.0]]  # keep-in\n")
    from atharv_nandi.core.geo_frame import load_flat_yaml
    assert load_flat_yaml(str(p)) == {"geofence": {"enabled": True, "polygon": [[-1.5, -2.0], [7.0, -2.0], [7.0, 2.0]]}}
