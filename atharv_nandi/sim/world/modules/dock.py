"""WP13 compose module ``dock_target`` (order 81): the docking aids on WP4's ``dock`` body. Orchard worlds only.

WP4 (``fixtures.py``) owns the ``dock`` module (order 80): plate ``dock_plate``, backboard ``dock_backboard``, site
``dock_ref`` and the crate. This module only adds what the WP13 dock logic (``controllers.dock_logic``) and the
console need, all on the ``dock`` body (world-static, no joint, no collision: contype 0 everywhere):

    sites (group 5, query)   ``dock_target``  the docked trunk pose (plate centre, docked heading), box of the O-20
                                               tolerance half extents (3 cm)
                             ``dock_staging`` the staging point ``STAGING_M`` in front of the plate on the entry line
    decals (group 4)         ``dock_lane_l`` / ``dock_lane_r`` painted entry lane edges from the staging point to
                             the plate, ``dock_tag`` the apron tag on the plate (visual stand-in for the dock tag the
                             real robot reads), ``dock_pad_l`` / ``dock_pad_r`` the two charge contact pads (visual only;
                             the pads, their forces and the de-energised handshake are Dock v2, WP37), and
                             ``dock_tol`` the tolerance square around the target

WP37 replaces this file with Dock v2 at the C+ boundary (one dock module per world).
"""
import json
import math

import mujoco

from atharv_nandi.core import dock_logic as dl
from atharv_nandi.sim.world import compose, field_spec

NAME = "dock_target"
ORDER = compose.ORDER["dock"] + 1          # 81: after WP4 built the dock body
VERSION = "1"
GRP_DECAL, GRP_QUERY = 4, 5
LANE_HALF_GAP_M = 0.36                     # lane edges just outside the stance (feet |y| <= 0.28 + 0.03 radius)
LANE_W_M = 0.025
DECAL_H_M = 0.001
LANE_RGBA = [0.95, 0.85, 0.15, 1.0]
TAG_RGBA = [0.05, 0.05, 0.05, 1.0]
PAD_RGBA = [0.72, 0.45, 0.20, 1.0]
TOL_RGBA = [0.10, 0.80, 0.30, 0.6]
G = mujoco.mjtGeom


def geometry(field: dict) -> dl.DockGeometry:
    return dl.DockGeometry.from_field(field)


def _find_body(spec, name):
    try:
        return spec.body(name)
    except Exception:  # pragma: no cover - older bindings
        return spec.find_body(name)


def _decal(body, name, pos, size, rgba, quat=(1, 0, 0, 0)):
    body.add_geom(name=name, type=G.mjGEOM_BOX, pos=list(pos), size=list(size), quat=list(quat),
                  group=GRP_DECAL, contype=0, conaffinity=0, rgba=rgba)


def attach(spec, field, ctx):
    if not field_spec.is_orchard(field) or not isinstance(field.get("dock"), dict):
        return
    body = _find_body(spec, "dock")
    if body is None:
        return
    g = geometry(field)
    sx, sy = (float(v) for v in field["dock"].get("size_m", [1.0, 0.8]))
    # dock body frame = docked heading at the plate centre (WP4: body pos = dock xy, quat = dock yaw)
    z = 0.004 + DECAL_H_M
    st = g.staging_m * (-g.entry)              # staging point along the body +x (reverse entry: in front)
    body.add_site(name="dock_target", type=G.mjGEOM_BOX, pos=[0, 0, z],
                  size=[dl.DOCK_TOL_M, dl.DOCK_TOL_M, DECAL_H_M], group=GRP_QUERY, rgba=[1, 1, 1, 0])
    body.add_site(name="dock_staging", pos=[st, 0, z], size=[0.03, 0, 0], group=GRP_QUERY, rgba=[1, 1, 1, 0])
    # lane edges from the staging point to the plate's far edge (towards the backboard)
    x0, x1 = st, -math.copysign(sx / 2.0, st)
    cx, half = (x0 + x1) / 2.0, abs(x1 - x0) / 2.0
    for side, y in (("l", LANE_HALF_GAP_M), ("r", -LANE_HALF_GAP_M)):
        _decal(body, f"dock_lane_{side}", [cx, y, z], [half, LANE_W_M / 2.0, DECAL_H_M], LANE_RGBA)
    _decal(body, "dock_tag", [0.0, 0.0, z], [0.06, 0.06, DECAL_H_M], TAG_RGBA)
    for side, y in (("l", 0.09), ("r", -0.09)):
        _decal(body, f"dock_pad_{side}", [-0.10, y, z + DECAL_H_M], [0.05, 0.03, DECAL_H_M], PAD_RGBA)
    t = dl.DOCK_TOL_M
    for k, (px, py, hx, hy) in enumerate(((0, t, t, 0.003), (0, -t, t, 0.003), (t, 0, 0.003, t), (-t, 0, 0.003, t))):
        _decal(body, f"dock_tol_{k}", [px, py, z + 2 * DECAL_H_M], [hx, hy, DECAL_H_M], TOL_RGBA)
    del sy


def _fingerprint(field, ctx) -> str:
    if not field_spec.is_orchard(field) or not isinstance(field.get("dock"), dict):
        return ""
    g = geometry(field)
    return json.dumps({"entry": g.entry, "staging_m": g.staging_m, "tol": [dl.DOCK_TOL_M, dl.DOCK_TOL_DEG]},
                      sort_keys=True, separators=(",", ":"))


compose.register_module(NAME, ORDER, attach, version=VERSION, fingerprint=_fingerprint)
