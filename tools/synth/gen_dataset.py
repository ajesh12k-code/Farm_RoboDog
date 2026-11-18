"""WP40 dataset generator (CAMERA-TRAINING C.2, C.3): RGB + seg + depth -> COCO, semantic, depth, metadata.

    MUJOCO_GL=glfw xvfb-run -a python3 -m tools.synth.gen_dataset --out /tmp/nds [--images 60] [--date 20261117]

Per image: image-level DR on the model (sun, ambient, colour temperature, fog/haze/dust, fovy), one RGB pass,
one depth pass, one segmentation pass (fog off), solo segmentation passes for the occlusion fraction
(instances whose projected bounds do not overlap share a pass), lens distortion on all three, then the
photometric post-process on RGB only. Output layout (flat; split membership in ``splits.json``):

    images/<name>.png  semantic/<name>.png  depth/<name>.npz (float16 m, 0 = no return)  meta/<name>.json
    annotations/instances_<split>.json  splits.json  calib/...  manifest.json  run_stats.json (not hashed)

Byte identity holds for the same seed on the same host; ``run_stats.json`` (wall clock) is outside the
manifest. ``data_provenance`` is always ``simulated``; the generator has no input path for real frames.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import CLASSES_YAML, SYNTH_YAML, __version__, load_yaml
from . import dr as DR
from . import labels as L
from . import manifest as MF
from . import scenes as S

SPLITS = ("train", "val", "test_seed", "test_bank", "test_species")


class SynthConfigError(ValueError):
    pass


# ------------------------------------------------------------------------------------------- config
def load_config(path: str = SYNTH_YAML, overrides: Optional[dict] = None) -> dict:
    cfg = load_yaml(path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def validate_config(cfg: dict) -> None:
    ds = cfg.get("dataset") or {}
    if ds.get("data_provenance", "simulated") != "simulated":
        raise SynthConfigError("data_provenance must be `simulated`: real frames never enter lane 0 (C.7, OQ-48)")
    for k in ("real_frames", "input_frames", "real_images", "field_frames"):
        if k in ds or k in cfg:
            raise SynthConfigError(f"`{k}` is not accepted: lane 0 is synthetic only (C.7, OQ-48)")
    if not isinstance(ds.get("shadows"), bool):
        raise SynthConfigError("dataset.shadows must be an explicit on/off (it changes the dataset id)")
    if ds.get("source", "stub") not in ("stub", "compose"):
        raise SynthConfigError("dataset.source must be stub or compose")
    DR.validate(cfg["dr"])
    sp = cfg["splits"]
    if set(sp["train_species"]) & set(sp["heldout_species"]):
        raise SynthConfigError("train and held-out species overlap")
    if set(sp["train_banks"]) & set(sp["heldout_banks"]):
        raise SynthConfigError("train and held-out DR banks overlap")
    for b in list(sp["train_banks"]) + list(sp["heldout_banks"]):
        if b not in cfg["banks"]:
            raise SynthConfigError(f"unknown bank {b!r}")
    for s in list(sp["train_species"]) + list(sp["heldout_species"]):
        if s not in cfg["species"]:
            raise SynthConfigError(f"unknown species {s!r}")
    unknown = set(sp["fractions"]) - set(SPLITS)
    if unknown:
        raise SynthConfigError(f"unknown splits {sorted(unknown)}")
    fr = {s: v for s, v in sp["fractions"].items()}
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 for v in fr.values()):
        raise SynthConfigError("splits.fractions must be numbers >= 0")
    if not any(v > 0 for v in fr.values()):
        raise SynthConfigError("splits.fractions: at least one split needs a positive fraction")
    for k in ("images", "images_per_scene", "width", "height"):
        v = ds.get(k)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise SynthConfigError(f"dataset.{k} must be an integer >= 1, got {v!r}")


def config_sha(cfg: dict) -> str:
    """sha256 of the effective config in canonical JSON (comments and key order do not count)."""
    return hashlib.sha256(L.json_bytes(cfg)).hexdigest()


# ------------------------------------------------------------------------------------------- planning
def _seed(*parts) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def split_counts(n_scenes: int, fractions: Dict[str, float]) -> Dict[str, int]:
    """Largest remainder; every split with a positive fraction gets >= 1 scene when there are enough."""
    names = [s for s in SPLITS if float(fractions.get(s, 0)) > 0]
    tot = sum(float(fractions[s]) for s in names)
    raw = {s: n_scenes * float(fractions[s]) / tot for s in names}
    cnt = {s: int(math.floor(raw[s])) for s in names}
    for s in sorted(names, key=lambda s: -(raw[s] - cnt[s])):
        if sum(cnt.values()) >= n_scenes:
            break
        cnt[s] += 1
    if n_scenes >= len(names):
        for s in names:
            while cnt[s] == 0:
                donor = max(names, key=lambda x: cnt[x])
                cnt[donor] -= 1
                cnt[s] += 1
    return cnt


def plan_scenes(cfg: dict) -> List[S.ScenePlan]:
    ds, sp = cfg["dataset"], cfg["splits"]
    n_scenes = max(1, int(math.ceil(int(ds["images"]) / int(ds["images_per_scene"]))))
    counts = split_counts(n_scenes, sp["fractions"])
    plans = []
    for split in SPLITS:
        for i in range(counts.get(split, 0)):
            seed = _seed(ds["seed"], split, i)
            rng = np.random.default_rng(seed)
            banks = sp["heldout_banks"] if split == "test_bank" else sp["train_banks"]
            species = sp["heldout_species"] if split == "test_species" else sp["train_species"]
            k = int(rng.integers(1, min(2, len(species)) + 1))
            chosen = tuple(str(s) for s in rng.choice(species, size=k, replace=False))
            plans.append(S.ScenePlan(seed=seed, split=split, kind="orchard", bank=str(rng.choice(banks)),
                                     species=chosen, season=str(rng.choice(cfg["scene"]["seasons"]))))
    return plans


SCENE_KNOBS = ("decal_size_m", "decal_opacity", "dock_tag_cover")


def scene_knobs(cfg: dict, plan: S.ScenePlan) -> dict:
    rng = np.random.default_rng(_seed(plan.seed, "scene_dr"))
    knobs = DR.sample(cfg["dr"], rng, int(cfg.get("curriculum_stage", 0)))
    return {k: knobs[k] for k in SCENE_KNOBS if k in knobs}


def build_scene(cfg: dict, plan: S.ScenePlan, registry: L.ClassRegistry) -> S.Scene:
    knobs = scene_knobs(cfg, plan)
    if cfg["dataset"].get("source", "stub") == "compose":
        return S.build_compose_scene(plan, cfg, registry, cfg["dataset"].get("compose_modules") or [],
                                     world=cfg["dataset"].get("compose_world", "synth_compose"))
    return S.build_stub_scene(plan, cfg, registry, knobs)


# ------------------------------------------------------------------------------------------- rendering
_GL = {}


def gl_context(width: int, height: int):
    """One offscreen GL context per process (creating one per scene costs ~0.25 s on llvmpipe)."""
    from mujoco import gl_context as glc
    key = (int(width), int(height))
    if key not in _GL:
        if glc.GLContext is None:
            raise RuntimeError("no offscreen GL backend (set MUJOCO_GL / DISPLAY)")
        if not _GL:
            import atexit
            atexit.register(_free_gl)
        _GL[key] = glc.GLContext(int(width), int(height))
    _GL[key].make_current()
    return _GL[key]


def _free_gl():
    for c in _GL.values():
        try:
            c.free()
        except Exception:
            pass
    _GL.clear()


class SceneRenderer:
    """Offscreen renderer per scene on the shared GL context: RGB + depth in one pass (two readbacks),
    a segmentation pass and solo segmentation passes. Same conversions as ``mujoco.Renderer`` 3.3."""

    def __init__(self, scene: S.Scene, width: int, height: int, shadows: bool):
        import mujoco
        self.mj = mujoco
        self.scene = scene
        self.m, self.d = scene.model, scene.data
        self.w, self.h = int(width), int(height)
        if self.w > self.m.vis.global_.offwidth or self.h > self.m.vis.global_.offheight:
            raise ValueError(f"{self.w}x{self.h} exceeds the offscreen buffer of the scene")
        self.shadows = bool(shadows)
        self.gl = gl_context(max(int(self.m.vis.global_.offwidth), 640), max(int(self.m.vis.global_.offheight), 480))
        self.scn = mujoco.MjvScene(self.m, maxgeom=6000)
        self.ctx = mujoco.MjrContext(self.m, mujoco.mjtFontScale.mjFONTSCALE_50.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self.ctx)
        self.ctx.readDepthMap = mujoco.mjtDepthMap.mjDEPTH_ZEROFAR
        self.set_render_size(self.w, self.h)
        self.pert = mujoco.MjvPerturb()
        self.opt = mujoco.MjvOption()
        for g in range(6):
            self.opt.geomgroup[g] = 1 if g in (0, 1, 3, 4) else 0     # group 5 query-only stays hidden
        self.solo_opt = mujoco.MjvOption()
        for g in range(6):
            self.solo_opt.geomgroup[g] = 1 if g == L.SOLO_GROUP else 0
        self.passes = 0
        self.fog = False

    def set_render_size(self, rw: int, rh: int):
        """Render size of the next passes (the output size, or the padded size lens distortion needs)."""
        rw, rh = int(rw), int(rh)
        if rw > self.m.vis.global_.offwidth or rh > self.m.vis.global_.offheight:
            raise ValueError(f"render {rw}x{rh} exceeds the offscreen buffer of the scene")
        if getattr(self, "rw", None) == rw and getattr(self, "rh", None) == rh:
            return
        self.rw, self.rh = rw, rh
        self.rect = self.mj.MjrRect(0, 0, rw, rh)
        self._rgb = np.empty((rh, rw, 3), dtype=np.uint8)
        self._z = np.empty((rh, rw), dtype=np.float32)

    def close(self):
        try:
            self.ctx.free()
        except Exception:
            pass

    def _update(self, opt):
        mj = self.mj
        self.gl.make_current()      # another renderer in the process may have switched the context
        mj.mjv_updateScene(self.m, self.d, opt, self.pert, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scn)

    def _linear_depth(self, z: np.ndarray) -> np.ndarray:
        extent = self.m.stat.extent
        zfar = np.float32(self.m.vis.map.zfar * extent)
        znear = np.float32(self.m.vis.map.znear * extent)
        c = -(zfar + znear) / (zfar - znear)
        dd = -(np.float32(2) * zfar * znear) / (zfar - znear)
        c = np.float32(-0.5) * c - np.float32(0.5)
        dd = np.float32(-0.5) * dd
        return (dd / (z.astype(np.float64) + c)).astype(np.float32)

    def _seg(self) -> np.ndarray:
        """(H, W, 2) objid, objtype like ``mujoco.Renderer`` segmentation (-1, -1 for background)."""
        mj = self.mj
        f = self.scn.flags
        f[mj.mjtRndFlag.mjRND_SEGMENT] = 1
        f[mj.mjtRndFlag.mjRND_IDCOLOR] = 1
        mj.mjr_render(self.rect, self.scn, self.ctx)
        mj.mjr_readPixels(self._rgb, None, self.rect, self.ctx)
        f[mj.mjtRndFlag.mjRND_SEGMENT] = 0
        f[mj.mjtRndFlag.mjRND_IDCOLOR] = 0
        img = np.flipud(self._rgb).astype(np.uint32)
        segid = img[..., 0] + img[..., 1] * 256 + img[..., 2] * 65536
        n = self.scn.ngeom
        table = np.full((n + 1, 2), -1, dtype=np.int32)
        for i in range(n):
            g = self.scn.geoms[i]
            if g.segid != -1 and g.segid + 1 <= n:
                table[g.segid + 1] = (g.objid, g.objtype)
        segid = np.where(segid <= n, segid, 0)
        return table[segid]

    # -- image-level DR on the model
    def apply_knobs(self, knobs: dict, view: S.View):
        mj, m = self.mj, self.m
        sun = mj.mj_name2id(m, mj.mjtObj.mjOBJ_LIGHT, "sun")
        if sun >= 0:
            m.light_dir[sun] = DR.sun_dir(knobs["sun_elevation_deg"], knobs["sun_azimuth_deg"])
            col = np.asarray(DR.kelvin_rgb(knobs["colour_temp_k"])) * float(knobs["sun_intensity"])
            m.light_diffuse[sun] = col
            m.light_ambient[sun] = [float(knobs["ambient"])] * 3
            m.light_castshadow[sun] = 1 if self.shadows else 0
        fog = DR.weather_fog(str(knobs["weather"]))
        self.fog = fog is not None
        if fog is not None:
            m.vis.map.fogstart, m.vis.map.fogend = fog[0], fog[1]
            m.vis.rgba.fog[:] = fog[2]
        m.vis.global_.fovy = float(view.fovy) + float(knobs.get("fovy_jitter_deg", 0.0))
        mj.mj_forward(m, self.d)

    def _flags(self, seg: bool):
        f = self.scn.flags
        rf = self.mj.mjtRndFlag
        f[rf.mjRND_SHADOW] = 1 if (self.shadows and not seg) else 0
        f[rf.mjRND_REFLECTION] = 0
        f[rf.mjRND_FOG] = 1 if (self.fog and not seg) else 0
        f[rf.mjRND_SKYBOX] = 1

    def render_all(self, view: S.View) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mj = self.mj
        self.cam = view.mjv_camera()
        self._update(self.opt)
        self._flags(seg=False)
        mj.mjr_render(self.rect, self.scn, self.ctx)
        mj.mjr_readPixels(self._rgb, self._z, self.rect, self.ctx)
        rgb = np.flipud(self._rgb).copy()
        depth = np.flipud(self._linear_depth(self._z)).copy()
        self._flags(seg=True)
        seg = self._seg()
        self.cam_pose = gl_camera_pose(self.scn)
        self.passes += 2
        return rgb, depth, seg

    def render_solo(self, inst_ids: List[int]) -> np.ndarray:
        """Seg pass with only these instances' geoms (moved to the solo group), others hidden."""
        m = self.m
        ids = np.concatenate([self.scene.inst_geoms[i] for i in inst_ids])
        if ids.size == 0:
            return np.full((self.rh, self.rw, 2), -1, dtype=np.int32)
        m.geom_group[ids] = L.SOLO_GROUP
        try:
            self._update(self.solo_opt)
            self._flags(seg=True)
            seg = self._seg()
        finally:
            m.geom_group[ids] = self.scene.geom_group0[ids]
        self.passes += 1
        return seg


def gl_camera_pose(scn):
    """Mono camera = mean of the two stereo GL cameras (each is offset by ipd / 2; mjr_render averages them)."""
    c0, c1 = scn.camera[0], scn.camera[1]
    pos = (np.array(c0.pos) + np.array(c1.pos)) / 2.0
    fwd = np.array(c0.forward) + np.array(c1.forward)
    up = np.array(c0.up) + np.array(c1.up)
    return pos, fwd / np.linalg.norm(fwd), up / np.linalg.norm(up)


def intrinsics(w: int, h: int, fovy_deg: float) -> dict:
    fy = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return {"fx": fy, "fy": fy, "cx": w / 2.0 - 0.5, "cy": h / 2.0 - 0.5, "width": w, "height": h}


def extrinsics(pos, forward, up) -> dict:
    """World -> camera (OpenCV: x right, y down, z forward)."""
    f = np.asarray(forward, float)
    u = np.asarray(up, float)
    right = np.cross(f, u)
    R = np.stack([right, -u, f])
    t = -R @ np.asarray(pos, float)
    return {"R": R.round(9).tolist(), "t": t.round(9).tolist(), "position": np.asarray(pos, float).round(9).tolist()}


def project_rects(scene: S.Scene, inst_ids, K: dict, ext: dict) -> Dict[int, Tuple[float, float, float, float]]:
    """Conservative screen rects from geom bounding spheres (full frame if near or behind the camera)."""
    R, t = np.asarray(ext["R"]), np.asarray(ext["t"])
    m, d = scene.model, scene.data
    W, H = K["width"], K["height"]
    out = {}
    for i in inst_ids:
        x0, y0, x1, y1 = math.inf, math.inf, -math.inf, -math.inf
        for g in scene.inst_geoms[i]:
            pc = R @ d.geom_xpos[g] + t
            r = float(m.geom_rbound[g]) * 1.05 + 1e-3
            if pc[2] <= r + 0.01:
                x0, y0, x1, y1 = 0, 0, W, H
                break
            u = K["fx"] * pc[0] / pc[2] + K["cx"]
            v = K["fy"] * pc[1] / pc[2] + K["cy"]
            rp = K["fy"] * r / math.sqrt(max(pc[2] ** 2 - r * r, 1e-9)) * 1.25 + 3
            x0, y0, x1, y1 = min(x0, u - rp), min(y0, v - rp), max(x1, u + rp), max(y1, v + rp)
        out[i] = (x0, y0, x1, y1)
    return out


def solo_groups(rects: Dict[int, tuple]) -> List[List[int]]:
    """Greedy: instances whose rects do not overlap share a solo pass (they cannot occlude each other)."""
    groups: List[Tuple[List[int], List[tuple]]] = []
    for i in sorted(rects):
        r = rects[i]
        for g, rs in groups:
            if all(r[2] <= q[0] or q[2] <= r[0] or r[3] <= q[1] or q[3] <= r[1] for q in rs):
                g.append(i)
                rs.append(r)
                break
        else:
            groups.append(([i], [r]))
    return [g for g, _ in groups]


def label_view(sr: SceneRenderer, view: S.View, knobs: dict, rng: np.random.Generator, min_px: int,
               season: str) -> dict:
    """Render one view and derive every label from the seg pass. Returns arrays + annotation records."""
    scene = sr.scene
    reg = scene.registry
    sr.apply_knobs(knobs, view)
    fovy = float(sr.m.vis.global_.fovy)
    K = intrinsics(sr.w, sr.h, fovy)
    dmap = DR.distortion_map(sr.w, sr.h, K["fy"], float(knobs.get("distortion_k1", 0.0)),
                             float(knobs.get("distortion_k2", 0.0)))
    # the effective coefficients the stored image follows (the sampled ones may be scaled down, see dr.py)
    K["k1"], K["k2"] = (dmap.k1, dmap.k2) if dmap is not None else (0.0, 0.0)
    K["distortion_scale"] = dmap.scale if dmap is not None else 1.0
    Kr = K                                   # intrinsics of the pinhole render the map samples
    if dmap is not None and (dmap.pad_x or dmap.pad_y):
        # padded render with the same fy: barrel distortion samples real pixels, never a smeared frame edge
        fovy_r = math.degrees(2.0 * math.atan((dmap.render_h / 2.0) / K["fy"]))
        Kr = intrinsics(dmap.render_w, dmap.render_h, fovy_r)
        sr.m.vis.global_.fovy = fovy_r
        sr.set_render_size(dmap.render_w, dmap.render_h)
    try:
        rgb, depth, seg = sr.render_all(view)
        ext = extrinsics(*sr.cam_pose)
        gid = DR.apply_map(L.geom_map(seg), dmap)
        inst = L.instance_map(gid, scene.geom_inst)
        vis = L.visible_instances(inst)
        keep = sorted(i for i, c in vis.items() if c >= min_px)
        small = sorted(i for i, c in vis.items() if c < min_px)
        full: Dict[int, int] = {}
        if keep:
            for grp in solo_groups(project_rects(scene, keep, Kr, ext)):
                solo = DR.apply_map(L.instance_map(L.geom_map(sr.render_solo(grp)), scene.geom_inst), dmap)
                sv = L.visible_instances(solo)
                for i in grp:
                    full[i] = max(sv.get(i, 0), vis[i])
    finally:
        sr.m.vis.global_.fovy = fovy
        sr.set_render_size(sr.w, sr.h)
    depth = DR.apply_map(depth, dmap)
    rgb = DR.apply_map(rgb, dmap)
    sem = L.semantic_map(gid, scene.geom_sem, reg.sem_id("sky"))
    depth = np.where(gid >= 0, depth, 0.0).astype(np.float16)
    anns = []
    tag_box = None
    for i in keep:
        ins = scene.instances[i]
        mask = inst == i
        bbox = L.bbox_of(mask)
        attrs = dict(ins.attrs)
        attrs["season"] = season
        attrs["occlusion"] = round(1.0 - vis[i] / float(full[i]), 4) if full.get(i) else 0.0
        reg.check_attrs(ins.cls, attrs)
        anns.append({"instance_id": int(i), "class": ins.cls, "category_id": reg.cat_id[ins.cls], "bbox": bbox,
                     "area": int(vis[i]), "segmentation": L.rle_encode(mask), "iscrowd": 0, "attrs": attrs})
        if ins.cls == "dock_tag" and tag_box is None:
            tag_box = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    rgb = DR.postprocess_rgb(rgb, knobs, rng, view.rig, tag_box)
    return {"rgb": rgb, "sem": sem, "depth": depth, "gid": gid, "inst": inst, "annotations": anns, "K": K,
            "K_render": Kr, "extrinsics": ext, "ignored_small": [int(i) for i in small]}


# ------------------------------------------------------------------------------------------- generate
def _git_sha() -> str:
    env = os.environ.get("NANDI_GIT_SHA", "").strip()
    if env:
        return env
    try:
        import subprocess
        from . import REPO
        out = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _write(path: str, blob: bytes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(blob)


OUTPUT_ENTRIES = ("images", "semantic", "depth", "meta", "annotations", "calib", "splits.json", "manifest.json",
                  "run_stats.json")


def prepare_out_dir(out_dir: str, overwrite: bool = False) -> None:
    """Refuse a non-empty ``out_dir``; with ``overwrite`` remove only the generator's own layout first."""
    if not os.path.isdir(out_dir):
        if os.path.exists(out_dir):
            raise SynthConfigError(f"--out {out_dir} exists and is not a directory")
        os.makedirs(out_dir)
        return
    entries = sorted(os.listdir(out_dir))
    if not entries:
        return
    if not overwrite:
        raise SynthConfigError(f"--out {out_dir} is not empty ({entries[:5]}); use an empty folder or --overwrite")
    foreign = [e for e in entries if e not in OUTPUT_ENTRIES]
    if foreign:
        raise SynthConfigError(f"--out {out_dir} holds files the generator does not write: {foreign[:5]}")
    for e in entries:
        p = os.path.join(out_dir, e)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def generate(out_dir: str, cfg: Optional[dict] = None, classes_path: str = CLASSES_YAML, date: Optional[str] = None,
             calib: bool = True, fingerprint_override: Optional[dict] = None, log=print,
             overwrite: bool = False) -> dict:
    """Generate a dataset into ``out_dir``. Returns the manifest (plus ``run_stats``).

    ``out_dir`` must be empty or absent: a stale file from an earlier run would otherwise sit next to the new
    manifest (and an export would pick it up). ``overwrite=True`` first removes an earlier generator layout
    and still refuses anything the generator does not write."""
    import mujoco
    cfg = cfg if cfg is not None else load_config()
    validate_config(cfg)
    reg = L.ClassRegistry(load_yaml(classes_path))
    ds = cfg["dataset"]
    W, H, n_img, per = int(ds["width"]), int(ds["height"]), int(ds["images"]), int(ds["images_per_scene"])
    shadows = bool(ds["shadows"])
    min_px = L.min_visible_px(W, int(ds.get("min_visible_px_at_320", 12)))
    lvl = int(ds.get("png_compress_level", 1))
    stage = int(cfg.get("curriculum_stage", 0))
    prepare_out_dir(out_dir, overwrite)
    plans = plan_scenes(cfg)
    t0 = time.perf_counter()
    coco: Dict[str, dict] = {}
    splits: Dict[str, List[str]] = {s: [] for s in SPLITS}
    by_cam: Dict[str, List[str]] = {}
    files: List[str] = []
    fingerprint = None
    n_done, ann_id, img_id, passes = 0, 1, 1, 0
    per_scene_s = []
    used: List[S.ScenePlan] = []
    for si, plan in enumerate(plans):
        if n_done >= n_img:
            break
        ts = time.perf_counter()
        used.append(plan)
        scene = build_scene(cfg, plan, reg)
        sr = SceneRenderer(scene, W, H, shadows)
        if fingerprint is None:
            fingerprint = fingerprint_override or MF.renderer_fingerprint()
        try:
            for k in range(min(per, n_img - n_done)):
                rng = np.random.default_rng(_seed(plan.seed, "image", k))
                knobs = DR.sample(cfg["dr"], rng, stage)
                knobs.update(scene_knobs(cfg, plan))
                view = S.sample_view(scene, cfg["cameras"], rng)
                out = label_view(sr, view, knobs, rng, min_px, plan.season)
                name = f"{plan.split}_{plan.seed:010d}_{k:03d}_{view.rig}"
                rels = {"image": f"images/{name}.png", "semantic": f"semantic/{name}.png",
                        "depth": f"depth/{name}.npz", "meta": f"meta/{name}.json"}
                _write(os.path.join(out_dir, rels["image"]), L.png_bytes(out["rgb"], lvl))
                _write(os.path.join(out_dir, rels["semantic"]), L.png_bytes(out["sem"], lvl))
                os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
                L.npz_bytes_write(os.path.join(out_dir, rels["depth"]), depth=out["depth"])
                tags = {c: v for c, v in scene.scene_tags.items()}
                meta = {"schema": "nandi.synth_image/1", "name": name, "split": plan.split, "files": rels,
                        "scene": {**plan.to_dict(), "world_id": scene.world_id, "versions": scene.versions},
                        "image_seed": [int(plan.seed), "image", k], "camera": {**view.to_dict(), "intrinsics": out["K"],
                                                                              "extrinsics": out["extrinsics"]},
                        "dr": {k2: (round(v, 6) if isinstance(v, float) else v) for k2, v in sorted(knobs.items())},
                        "renderer_fingerprint": fingerprint, "data_provenance": "simulated",
                        "image_tags": tags, "ignored_small": out["ignored_small"],
                        "min_visible_px": min_px, "semantic_classes": reg.semantic}
                _write(os.path.join(out_dir, rels["meta"]), L.json_bytes(meta))
                files.extend(rels.values())
                c = coco.setdefault(plan.split, {"images": [], "annotations": []})
                c["images"].append({"id": img_id, "file_name": rels["image"], "width": W, "height": H,
                                    "camera": view.rig, "split": plan.split, "scene_seed": int(plan.seed),
                                    "semantic": rels["semantic"], "depth": rels["depth"], "meta": rels["meta"],
                                    "data_provenance": "simulated"})
                for a in out["annotations"]:
                    c["annotations"].append({"id": ann_id, "image_id": img_id, **{kk: a[kk] for kk in (
                        "category_id", "bbox", "area", "segmentation", "iscrowd", "attrs", "instance_id")}})
                    ann_id += 1
                splits[plan.split].append(name)
                by_cam.setdefault(f"{plan.split}/{view.rig}", []).append(name)
                img_id += 1
                n_done += 1
            passes += sr.passes
        finally:
            sr.close()
        per_scene_s.append(round(time.perf_counter() - ts, 4))
    info = {"description": "Atharv Nandi synthetic dataset (WP40, lane 0)", "version": __version__,
            "config_sha256": config_sha(cfg), "data_provenance": "simulated", "generator": "tools.synth"}
    cats = reg.coco_categories()
    for split, c in sorted(coco.items()):
        doc = {"info": {**info, "split": split}, "licenses": [], "categories": cats, **c}
        rel = f"annotations/instances_{split}.json"
        _write(os.path.join(out_dir, rel), L.json_bytes(doc))
        files.append(rel)
    split_doc = {"schema": "nandi.synth_splits/1", "splits": {s: v for s, v in splits.items() if v},
                 "scene_seeds": {s: sorted(p.seed for p in used if p.split == s) for s in SPLITS if splits[s]},
                 "banks": {s: sorted({p.bank for p in used if p.split == s}) for s in SPLITS if splits[s]},
                 "species": {s: sorted({x for p in used if p.split == s for x in p.species}) for s in SPLITS
                             if splits[s]},
                 "per_camera": {k: v for k, v in sorted(by_cam.items())}}
    _write(os.path.join(out_dir, "splits.json"), L.json_bytes(split_doc))
    files.append("splits.json")
    gen_s = time.perf_counter() - t0
    calib_report = None
    if calib:
        from . import calib_check
        tc = time.perf_counter()
        files.extend(calib_check.render_targets(cfg, out_dir))
        calib_report = {"render_s": round(time.perf_counter() - tc, 3)}
    man = MF.build_manifest(out_dir, files, cfg=cfg, classes_path=classes_path, fingerprint=fingerprint or {},
                            scene_seeds=split_doc["scene_seeds"], splits=split_doc["splits"], date=date,
                            git_sha=_git_sha(), mujoco_version=mujoco.__version__)
    stats = {"images": n_done, "scenes": len(per_scene_s), "gen_s": round(gen_s, 3),
             "img_per_s": round(n_done / gen_s, 3) if gen_s > 0 else None, "render_passes": passes,
             "per_scene_s": per_scene_s, "width": W, "height": H, "calib": calib_report,
             "annotations": ann_id - 1, "dataset_id": man["dataset_id"],
             "cpu_count": os.cpu_count(), "note": "wall clock; not part of the manifest"}
    _write(os.path.join(out_dir, "run_stats.json"), L.json_bytes(stats))
    log(f"[synth] {n_done} images in {gen_s:.2f} s = {stats['img_per_s']} img/s, {ann_id - 1} annotations, "
        f"id {man['dataset_id']}")
    man["run_stats"] = stats
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=SYNTH_YAML)
    ap.add_argument("--classes", default=CLASSES_YAML)
    ap.add_argument("--images", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--shadows", choices=["on", "off"])
    ap.add_argument("--date", help="yyyymmdd for the dataset id (default: today, UTC)")
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace an earlier dataset in --out")
    a = ap.parse_args(argv)
    ov: dict = {"dataset": {}}
    for k in ("images", "width", "height", "seed"):
        if getattr(a, k) is not None:
            ov["dataset"][k] = getattr(a, k)
    if a.shadows:
        ov["dataset"]["shadows"] = a.shadows == "on"
    try:
        cfg = load_config(a.config, ov)
        man = generate(a.out, cfg, a.classes, date=a.date, calib=not a.no_calib, overwrite=a.overwrite)
    except SynthConfigError as exc:
        print(f"gen_dataset: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset_id": man["dataset_id"], **man["run_stats"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
