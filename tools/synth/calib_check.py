"""Sim calibration targets and check (CAMERA-TRAINING C.2 side-task, C.8).

The generator renders a marker board (``calibration.board`` in synth.yaml: a white plate with a grid of
dark square marker geoms) per camera rig from several poses, with the ground-truth intrinsics and
extrinsics of each view. This module then runs a calibration routine (Zhang's homography method, closed
form, numpy only) on the marker centres and checks that it recovers fx, fy, cx, cy within ``tolerance``
(1 %), and every view's camera centre within ``tolerance`` of the camera-board distance (extrinsics). Marker identities come from the segmentation pass (each marker is its own geom): the stand-in for
AprilTag decoding. Centres are the centroid of each marker's pixels in the id image.

    python3 -m tools.synth.calib_check <dataset_dir>        # check the calib/ targets of a dataset

Files: ``calib/<rig>/view_<k>.png`` (RGB), ``calib/<rig>/view_<k>_ids.png`` (0 = none, j + 1 = marker j),
``calib/truth.json`` (board geometry, per-view intrinsics and extrinsics).
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

from . import labels as L
from . import scenes as S

BOARD_X = 0.0          # board plane x = 0 (facing +x), centre at z = BOARD_Z
BOARD_Z = 1.0


def attach_board(spec, field, ctx):
    """Compose module (order 90): white plate + one geom per marker, in the board's (u, v) grid."""
    mj = S._mj()
    b = ctx.config["board"]
    rec: S.TruthRecorder = ctx.config["truth"]
    spec.stat.extent = S.EXTENT_M
    spec.stat.center = [0.0, 0.0, 0.0]
    spec.visual.map.znear = S.ZNEAR
    spec.visual.map.zfar = S.ZFAR
    spec.visual.headlight.ambient = [0.6, 0.6, 0.6]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    cols, rows, pitch, ms = int(b["cols"]), int(b["rows"]), float(b["pitch_m"]), float(b["marker_m"])
    hw, hh = (cols + 1) * pitch / 2, (rows + 1) * pitch / 2
    spec.worldbody.add_geom(name="board", type=mj.mjtGeom.mjGEOM_BOX, size=[0.004, hw, hh], pos=[BOARD_X - 0.004, 0, BOARD_Z],
                            rgba=[0.95, 0.95, 0.95, 1], group=0, contype=0, conaffinity=0)
    pts = []
    for r in range(rows):
        for c in range(cols):
            u = (c - (cols - 1) / 2) * pitch
            v = (r - (rows - 1) / 2) * pitch
            spec.worldbody.add_geom(name=f"marker/{r}/{c}", type=mj.mjtGeom.mjGEOM_BOX, size=[0.0005, ms / 2, ms / 2],
                                    pos=[BOARD_X + 0.0005, u, BOARD_Z + v], rgba=[0.05, 0.05, 0.05, 1], group=0,
                                    contype=0, conaffinity=0)
            pts.append((u, v))
    ctx.config["board_points"] = pts
    del rec


def board_world(u: float, v: float) -> np.ndarray:
    return np.array([BOARD_X + 0.001, u, BOARD_Z + v])


def build_board(cfg: dict):
    import mujoco
    from atharv_nandi.sim.world import compose
    board = cfg["calibration"]["board"]
    ctx = compose.ComposeContext(world="synth_calib", seed=0, config={"board": board, "truth": None})
    mod = compose.Module("synth_calib_board", 90, attach_board, version="1")
    spec = mujoco.MjSpec()
    mod.attach_fn(spec, {"world": "synth_calib", "seed": 0}, ctx)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    first = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "marker/0/0")
    return m, d, np.array(ctx.config["board_points"]), first


def board_views(rig_cfg: dict, n: int, w: int, h: int, board: dict, rng: np.random.Generator) -> List[S.View]:
    """Camera poses that keep the whole board in view, varying azimuth/elevation (board tilt in the camera)."""
    fovy = float(np.mean(rig_cfg["fovy_deg"]))
    hw = (int(board["cols"]) + 1) * float(board["pitch_m"]) / 2
    hh = (int(board["rows"]) + 1) * float(board["pitch_m"]) / 2
    half_v = math.tan(math.radians(fovy) / 2)
    half_h = half_v * w / h
    dist = 1.25 * max(hw / half_h, hh / half_v) + 0.05
    views = []
    for k in range(n):
        az = math.radians([-32, 30, -15, 22, 0, -25, 12, 35][k % 8] + rng.uniform(-3, 3))
        el = math.radians([18, -15, -25, 20, 28, -8, 10, -20][k % 8] + rng.uniform(-3, 3))
        eye = np.array([BOARD_X + dist * math.cos(el) * math.cos(az), dist * math.cos(el) * math.sin(az),
                        BOARD_Z + dist * math.sin(el)])
        tgt = np.array([BOARD_X, rng.uniform(-0.15, 0.15) * hw, BOARD_Z + rng.uniform(-0.15, 0.15) * hh])
        views.append(S.View("calib", eye, tgt, fovy))
    return views


def render_targets(cfg: dict, out_dir: str) -> List[str]:
    """Render the board per rig, write RGB + id images and truth.json. Returns the relative paths written."""
    import mujoco
    from .gen_dataset import extrinsics, gl_camera_pose, intrinsics
    cal = cfg["calibration"]
    W, H = int(cfg["dataset"]["width"]), int(cfg["dataset"]["height"])
    m, d, pts, first = build_board(cfg)
    n_mark = len(pts)
    r = mujoco.Renderer(m, H, W)
    written, truth = [], {"board": {**cal["board"], "points_uv": pts.round(9).tolist(),
                                    "plane": "x=0 facing +x, (u, v) = (y, z - 1.0)"}, "rigs": {}}
    try:
        for rig in cal["rigs"]:
            rng = np.random.default_rng(sum(map(ord, rig)))
            rig_views = []
            for k, view in enumerate(board_views(cfg["cameras"][rig], int(cal["views"]), W, H, cal["board"], rng)):
                m.vis.global_.fovy = view.fovy
                cam = view.mjv_camera()
                r.update_scene(d, camera=cam)
                rgb = r.render().copy()
                ext = extrinsics(*gl_camera_pose(r.scene))
                r.enable_segmentation_rendering()
                seg = r.render().copy()
                r.disable_segmentation_rendering()
                gid = L.geom_map(seg)
                ids = np.where((gid >= first) & (gid < first + n_mark), gid - first + 1, 0).astype(np.uint8)
                base = f"calib/{rig}/view_{k}"
                for rel, img in ((f"{base}.png", rgb), (f"{base}_ids.png", ids)):
                    p = os.path.join(out_dir, rel)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "wb") as fh:
                        fh.write(L.png_bytes(img))
                    written.append(rel)
                rig_views.append({"view": k, "rgb": f"{base}.png", "ids": f"{base}_ids.png", **view.to_dict(),
                                  "intrinsics": intrinsics(W, H, view.fovy), "extrinsics": ext})
            truth["rigs"][rig] = rig_views
    finally:
        r.close()
    p = os.path.join(out_dir, "calib", "truth.json")
    with open(p, "wb") as fh:
        fh.write(L.json_bytes(truth))
    written.append("calib/truth.json")
    return written


# ------------------------------------------------------------------------------------------- calibration
def marker_centres(ids: np.ndarray, n_mark: int, min_px: int = 6) -> Dict[int, Tuple[float, float]]:
    """Centroid (u, v) in pixel-index coordinates of every marker fully inside the frame."""
    h, w = ids.shape
    lab = ids.reshape(-1).astype(np.int64)
    vv, uu = np.mgrid[0:h, 0:w]
    n = np.bincount(lab, minlength=n_mark + 1)
    su = np.bincount(lab, weights=uu.reshape(-1), minlength=n_mark + 1)
    sv = np.bincount(lab, weights=vv.reshape(-1), minlength=n_mark + 1)
    edge = np.unique(np.concatenate([ids[0], ids[-1], ids[:, 0], ids[:, -1]]))
    out = {}
    for j in range(1, n_mark + 1):
        if n[j] < min_px or j in edge:
            continue
        out[j - 1] = (float(su[j] / n[j]), float(sv[j] / n[j]))
    return out


def _normalise(p: np.ndarray):
    c = p.mean(axis=0)
    s = math.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.c_[p, np.ones(len(p))] @ T.T
    return ph, T


def _null_vector(A: np.ndarray) -> np.ndarray:
    """Least-squares null vector of A via the small Gram matrix (a tall SVD is slow under BLAS threading)."""
    w, v = np.linalg.eigh(A.T @ A)
    return v[:, 0]


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalised DLT, src (board uv) -> dst (pixels)."""
    a, Ta = _normalise(src)
    b, Tb = _normalise(dst)
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    Hn = _null_vector(np.asarray(rows)).reshape(3, 3)
    Hm = np.linalg.inv(Tb) @ Hn @ Ta
    return Hm / Hm[2, 2]


def zhang_intrinsics(Hs: List[np.ndarray]) -> np.ndarray:
    def v(Hm, i, j):
        h = Hm.T
        return np.array([h[i, 0] * h[j, 0], h[i, 0] * h[j, 1] + h[i, 1] * h[j, 0], h[i, 1] * h[j, 1],
                         h[i, 2] * h[j, 0] + h[i, 0] * h[j, 2], h[i, 2] * h[j, 1] + h[i, 1] * h[j, 2], h[i, 2] * h[j, 2]])
    V = []
    for Hm in Hs:
        V.append(v(Hm, 0, 1))
        V.append(v(Hm, 0, 0) - v(Hm, 1, 1))
    b11, b12, b22, b13, b23, b33 = _null_vector(np.asarray(V))
    v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 ** 2)
    lam = b33 - (b13 ** 2 + v0 * (b12 * b13 - b11 * b23)) / b11
    alpha = math.sqrt(lam / b11)
    beta = math.sqrt(lam * b11 / (b11 * b22 - b12 ** 2))
    gamma = -b12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - b13 * alpha ** 2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])


def extrinsic_from_h(K: np.ndarray, Hm: np.ndarray):
    Ki = np.linalg.inv(K)
    h1, h2, h3 = Hm[:, 0], Hm[:, 1], Hm[:, 2]
    lam = 1.0 / np.linalg.norm(Ki @ h1)
    r1, r2 = lam * Ki @ h1, lam * Ki @ h2
    t = lam * Ki @ h3
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt, t


def check(out_dir: str, tolerance: float = 0.01) -> dict:
    """Recover intrinsics per rig from the dataset's calib targets and compare with the truth."""
    with open(os.path.join(out_dir, "calib", "truth.json"), encoding="utf-8") as fh:
        truth = json.load(fh)
    uv = np.asarray(truth["board"]["points_uv"])
    res = {"ok": True, "tolerance": tolerance, "rigs": {}}
    for rig, views in truth["rigs"].items():
        Hs, gts, cents = [], [], []
        for v in views:
            ids = L.png_read(os.path.join(out_dir, v["ids"]))
            c = marker_centres(ids, len(uv))
            if len(c) < 8:
                continue
            idx = sorted(c)
            Hs.append(homography(uv[idx], np.array([c[i] for i in idx])))
            gts.append(v)
            cents.append(len(idx))
        if len(Hs) < 3:
            res["rigs"][rig] = {"ok": False, "reason": f"only {len(Hs)} usable views"}
            res["ok"] = False
            continue
        K = zhang_intrinsics(Hs)
        gt = gts[0]["intrinsics"]
        err = {"fx": abs(K[0, 0] - gt["fx"]) / gt["fx"], "fy": abs(K[1, 1] - gt["fy"]) / gt["fy"],
               "cx": abs(K[0, 2] - gt["cx"]) / gt["cx"], "cy": abs(K[1, 2] - gt["cy"]) / gt["cy"]}
        # extrinsics: camera centre in the board frame vs truth (relative to the camera-board distance)
        pos_err = []
        for Hm, v in zip(Hs, gts):
            R, t = extrinsic_from_h(K, Hm)
            c_board = -R.T @ t                          # camera centre in board (u, v, n) coordinates
            p = np.asarray(v["extrinsics"]["position"])
            c_true = np.array([p[1], p[2] - BOARD_Z, p[0] - BOARD_X - 0.001])
            pos_err.append(float(np.linalg.norm(c_board - c_true) / np.linalg.norm(c_true)))
        ok = bool(max(err.values()) <= tolerance and max(pos_err) <= tolerance)
        res["rigs"][rig] = {"ok": ok, "views": len(Hs), "markers_per_view": cents,
                            "K": np.round(K, 4).tolist(), "truth": {k: gt[k] for k in ("fx", "fy", "cx", "cy")},
                            "rel_err": {k: round(float(e), 6) for k, e in err.items()},
                            "max_rel_err": round(float(max(err.values())), 6),
                            "skew": round(float(K[0, 1]), 5),
                            "extrinsic_centre_rel_err_max": round(max(pos_err), 6)}
        res["ok"] = res["ok"] and ok
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="check the sim calibration targets of a dataset")
    ap.add_argument("dataset")
    ap.add_argument("--tolerance", type=float, default=0.01)
    a = ap.parse_args(argv)
    rep = check(a.dataset, a.tolerance)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
