"""Dataset manifest, Merkle root, renderer fingerprint and dataset id (CAMERA-TRAINING C.3).

    dataset id = nds-<yyyymmdd>-<merkle8>

The Merkle tree's leaves are every data file (``sha256("file:" + path + ":" + sha256(bytes))``, sorted by path)
plus the identity leaves: effective ``synth.yaml`` sha, classes sha, ``shadows``, renderer fingerprint sha,
MuJoCo version. So the id changes with the config (``shadows`` included) and with the renderer fingerprint
(GL vendor / renderer / version, Mesa version, ``LP_NUM_THREADS``, ``MUJOCO_GL``): the capped runner, the CI runner
and a GPU host produce different ids for the same ``synth.yaml``, honestly. The generator git sha is recorded
but is not an id leaf (code changes that alter output change the file leaves anyway).

``compare_labels`` is the cross-host check: COCO annotations and semantic PNGs equal, depth within 1 mm
(or one float16 step where that is coarser), RGB bytes ignored.
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

from . import labels as L

SCHEMA = "nandi.synth_manifest/1"
FINGERPRINT_KEYS = ("gl_vendor", "gl_renderer", "gl_version", "mesa_version", "LP_NUM_THREADS", "MUJOCO_GL",
                    "mujoco_version")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def renderer_fingerprint() -> Dict[str, str]:
    """GL strings of the current context (call with a ``mujoco.Renderer`` alive) plus the env that shapes it."""
    import mujoco
    fp = {k: "unknown" for k in FINGERPRINT_KEYS}
    try:
        from OpenGL import GL
        fp["gl_vendor"] = (GL.glGetString(GL.GL_VENDOR) or b"").decode()
        fp["gl_renderer"] = (GL.glGetString(GL.GL_RENDERER) or b"").decode()
        fp["gl_version"] = (GL.glGetString(GL.GL_VERSION) or b"").decode()
    except Exception as exc:       # pragma: no cover - depends on the GL stack
        fp["gl_error"] = type(exc).__name__
    m = re.search(r"Mesa (\S+)", fp.get("gl_version", ""))
    fp["mesa_version"] = m.group(1) if m else "none"
    fp["LP_NUM_THREADS"] = os.environ.get("LP_NUM_THREADS", "")
    fp["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "")
    fp["mujoco_version"] = mujoco.__version__
    return fp


def fingerprint_sha(fp: dict) -> str:
    return hashlib.sha256(L.json_bytes({k: fp.get(k, "") for k in sorted(fp)})).hexdigest()


def merkle_root(leaves: List[bytes]) -> str:
    """Binary sha256 tree over the leaf hashes in order; an odd node is promoted unchanged."""
    if not leaves:
        return hashlib.sha256(b"empty").hexdigest()
    level = [hashlib.sha256(b"\x00" + x).digest() for x in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest())
            else:
                nxt.append(level[i])
        level = nxt
    return level[0].hex()


def identity_leaves(config_sha: str, classes_sha: str, shadows: bool, fp_sha: str, mujoco_version: str) -> List[bytes]:
    return [f"id:config:{config_sha}".encode(), f"id:classes:{classes_sha}".encode(),
            f"id:shadows:{'on' if shadows else 'off'}".encode(), f"id:renderer:{fp_sha}".encode(),
            f"id:mujoco:{mujoco_version}".encode()]


def file_leaves(files: Dict[str, str]) -> List[bytes]:
    return [f"file:{p}:{files[p]}".encode() for p in sorted(files)]


def dataset_id(root: str, date: str) -> str:
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"date {date!r} is not yyyymmdd")
    return f"nds-{date}-{root[:8]}"


def build_manifest(out_dir: str, rel_files: List[str], *, cfg: dict, classes_path: str, fingerprint: dict,
                   scene_seeds: dict, splits: dict, date: Optional[str], git_sha: str, mujoco_version: str) -> dict:
    from .gen_dataset import config_sha
    files = {p: sha256_file(os.path.join(out_dir, p)) for p in sorted(set(rel_files))}
    c_sha = config_sha(cfg)
    k_sha = sha256_file(classes_path)
    fp_sha = fingerprint_sha(fingerprint)
    shadows = bool(cfg["dataset"]["shadows"])
    leaves = identity_leaves(c_sha, k_sha, shadows, fp_sha, mujoco_version) + file_leaves(files)
    root = merkle_root(leaves)
    date = date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
    man = {"schema": SCHEMA, "dataset_id": dataset_id(root, date), "date": date, "merkle_root": root,
           "config_sha256": c_sha, "classes_sha256": k_sha, "shadows": shadows,
           "renderer_fingerprint": fingerprint, "renderer_fingerprint_sha256": fp_sha,
           "mujoco_version": mujoco_version, "generator": {"git_sha": git_sha, "package": "tools.synth"},
           "scene_seeds": scene_seeds, "splits": splits, "data_provenance": "simulated",
           "n_files": len(files), "files": files}
    with open(os.path.join(out_dir, "manifest.json"), "wb") as fh:
        fh.write(L.json_bytes(man))
    return man


def load(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


UNHASHED = ("manifest.json", "run_stats.json")


def untracked_files(out_dir: str, listed) -> List[str]:
    """Files under ``out_dir`` that the manifest does not list (``manifest.json``, ``run_stats.json`` and
    ``__pycache__`` aside): a stale file from another run must not ride along with a verified id."""
    out = []
    for root, dirs, fnames in os.walk(out_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(fnames):
            rel = os.path.relpath(os.path.join(root, f), out_dir).replace(os.sep, "/")
            if rel not in listed and rel not in UNHASHED:
                out.append(rel)
    return out


def verify(out_dir: str) -> dict:
    """Recompute every file sha, the Merkle root and the id. Refuse a non-simulated image, a missing file
    and any file the manifest does not list. Never raises on a damaged dataset: problems are reported."""
    try:
        man = load(out_dir)
    except (OSError, ValueError) as exc:
        return {"ok": False, "problems": [f"no readable manifest.json ({type(exc).__name__})"], "dataset_id": None,
                "n_files": 0}
    problems = []
    for p, sha in man["files"].items():
        full = os.path.join(out_dir, p)
        if not os.path.isfile(full):
            problems.append(f"missing {p}")
            continue
        if sha256_file(full) != sha:
            problems.append(f"sha mismatch {p}")
        if p.startswith("meta/"):
            try:
                with open(full, encoding="utf-8") as fh:
                    prov = json.load(fh).get("data_provenance")
            except (OSError, ValueError, AttributeError):
                prov = None
            if prov != "simulated":
                problems.append(f"non-simulated provenance {p}")
    listed = set(man["files"])
    for extra in untracked_files(out_dir, listed):
        problems.append(f"untracked {extra}")
    leaves = identity_leaves(man["config_sha256"], man["classes_sha256"], man["shadows"],
                             fingerprint_sha(man["renderer_fingerprint"]), man["mujoco_version"]) + file_leaves(man["files"])
    root = merkle_root(leaves)
    if root != man["merkle_root"]:
        problems.append("merkle root mismatch")
    if dataset_id(root, man["date"]) != man["dataset_id"]:
        problems.append("dataset id mismatch")
    return {"ok": not problems, "problems": problems, "dataset_id": man["dataset_id"], "n_files": len(man["files"])}


# ------------------------------------------------------------------------------------------- cross host
def _f16_step(x: np.ndarray) -> np.ndarray:
    return np.abs(np.spacing(np.abs(x).astype(np.float16))).astype(np.float64)


def compare_labels(dir_a: str, dir_b: str, depth_tol_m: float = 0.001) -> dict:
    """Cross-host equality: COCO annotations, semantic PNGs, splits equal; depth within tolerance; RGB ignored."""
    ma, mb = load(dir_a), load(dir_b)
    problems = []
    fa = {p for p in ma["files"] if not p.startswith("images/")}
    fb = {p for p in mb["files"] if not p.startswith("images/")}
    if fa != fb:
        problems.append(f"file sets differ: {sorted(fa ^ fb)[:5]}")
    worst_depth = 0.0
    n_pix_over = 0
    for p in sorted(fa & fb):
        a, b = os.path.join(dir_a, p), os.path.join(dir_b, p)
        if p.startswith("annotations/"):
            ja, jb = (json.load(open(x, encoding="utf-8")) for x in (a, b))
            if ja["annotations"] != jb["annotations"] or ja["categories"] != jb["categories"]:
                problems.append(f"annotations differ {p}")
            if [i["file_name"] for i in ja["images"]] != [i["file_name"] for i in jb["images"]]:
                problems.append(f"image lists differ {p}")
        elif p.startswith("semantic/") or p.startswith("calib/") and p.endswith("_ids.png"):
            if not np.array_equal(L.png_read(a), L.png_read(b)):
                problems.append(f"semantic differs {p}")
        elif p.startswith("depth/"):
            da, db = (np.load(x)["depth"].astype(np.float64) for x in (a, b))
            diff = np.abs(da - db)
            tol = np.maximum(depth_tol_m, _f16_step(np.maximum(da, db)))
            n_pix_over += int(np.sum(diff > tol + 1e-12))
            worst_depth = max(worst_depth, float(diff.max()) if diff.size else 0.0)
        elif p == "splits.json":
            if json.load(open(a, encoding="utf-8")) != json.load(open(b, encoding="utf-8")):
                problems.append("splits differ")
    if n_pix_over:
        problems.append(f"{n_pix_over} depth pixels beyond tolerance")
    return {"ok": not problems, "problems": problems, "max_depth_diff_m": worst_depth,
            "ids": [ma["dataset_id"], mb["dataset_id"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="verify a dataset manifest, or compare two datasets' labels")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args(argv)
    if len(a.dirs) == 1:
        rep = verify(a.dirs[0])
    else:
        rep = compare_labels(a.dirs[0], a.dirs[1])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
