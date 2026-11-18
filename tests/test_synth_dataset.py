"""WP40 synthetic dataset generator (CAMERA-TRAINING C.2, C.8).

Unit tier: config and DR validation, splits, class registry, RLE / COCO checks, probe-scene class coverage,
YOLO export. `gl` tier (skips cleanly without an offscreen GL backend): 60 images at 320 x 240 on 2 CPU at
>= 4 img/s, COCO validates, masks equal the seg render, calibration within 1 %.
"""
import copy
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from tools.synth import dr as DR  # noqa: E402
from tools.synth import export_yolo as EY  # noqa: E402
from tools.synth import gen_dataset as G  # noqa: E402
from tools.synth import labels as L  # noqa: E402
from tools.synth import scenes as S  # noqa: E402


def _gl_available() -> bool:
    code = ("import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco/>'); "
            "r = mujoco.Renderer(m, 8, 8); r.close()")
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30).returncode == 0
    except Exception:
        return False


GL = _gl_available()
needs_gl = pytest.mark.skipif(not GL, reason="no offscreen GL backend (set MUJOCO_GL / DISPLAY)")


@pytest.fixture(scope="module")
def cfg():
    return G.load_config()


@pytest.fixture(scope="module")
def reg():
    return L.load_registry()


def probe_scene(cfg, reg, seed=11):
    plan = S.ScenePlan(seed=seed, split="train", kind="probe", bank="bank_a", species=("mango",), season="fruiting")
    return S.build_stub_scene(plan, cfg, reg, G.scene_knobs(cfg, plan))


# ------------------------------------------------------------------------------------------- config / DR
def test_config_validates_and_lane0_defaults(cfg):
    assert cfg["dataset"]["width"] == 320 and cfg["dataset"]["height"] == 240
    assert cfg["dataset"]["shadows"] is False            # lane 0 uses shadows off (explicit field)
    assert cfg["dataset"]["data_provenance"] == "simulated"


@pytest.mark.parametrize("override, msg", [
    ({"dataset": {"shadows": "maybe"}}, "shadows"),
    ({"dataset": {"data_provenance": "field"}}, "simulated"),
    ({"dataset": {"real_frames": "/data/field"}}, "synthetic only"),
    ({"splits": {"heldout_species": ["mango"]}}, "overlap"),
    ({"splits": {"heldout_banks": ["bank_a"]}}, "overlap"),
    ({"dr": {"exposure": {"dist": "uniform", "range": [1, 0], "stage": 0, "default": 1}}}, "range"),
    ({"dr": {"exposure": {"dist": "gamma", "range": [0, 1], "stage": 0, "default": 1}}}, "dist"),
    ({"dr": {"exposure": {"dist": "uniform", "range": [0, 1], "stage": -1, "default": 1}}}, "stage"),
    ({"dataset": {"images_per_scene": 0}}, "images_per_scene"),
    ({"dataset": {"images": 0}}, "dataset.images"),
    ({"splits": {"fractions": {"train": 0, "val": 0, "test_seed": 0, "test_bank": 0, "test_species": 0}}},
     "positive fraction"),
    ({"splits": {"fractions": {"train": -0.1}}}, ">= 0"),
])
def test_config_validation_refuses(cfg, override, msg):
    bad = G.deep_merge(cfg, override)
    with pytest.raises(ValueError, match=msg):
        G.validate_config(bad)


def test_every_dr_knob_has_range_distribution_and_stage(cfg):
    for name, k in cfg["dr"].items():
        assert k["dist"] in DR.DISTS, name
        assert isinstance(k["stage"], int), name
        assert "default" in k, name
        assert any(x in k for x in ("range", "values", "p")), name


def test_dr_sample_is_seed_deterministic_and_respects_curriculum(cfg):
    a = DR.sample(cfg["dr"], np.random.default_rng(5), 2)
    b = DR.sample(cfg["dr"], np.random.default_rng(5), 2)
    assert a == b
    s0 = DR.sample(cfg["dr"], np.random.default_rng(5), 0)
    for name, k in cfg["dr"].items():
        if k["stage"] > 0:
            assert s0[name] == k["default"], name
        if k["dist"] == "uniform":
            lo, hi = k["range"]
            assert lo <= a[name] <= hi
    assert a["weather"] in DR.WEATHER


def test_photometric_dr_never_touches_labels_and_is_deterministic():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    knobs = {"exposure": 1.2, "white_balance": 0.95, "defocus_px": 1, "motion_blur_px": 2, "sensor_noise_sigma": 3.0,
             "jpeg_quality": 75, "rain_streaks": True, "lens_drops": True, "belly_lens_dirt": 0.4,
             "dock_tag_glare": True}
    a = DR.postprocess_rgb(img, knobs, np.random.default_rng(1), "belly", (10, 10, 20, 20))
    b = DR.postprocess_rgb(img, knobs, np.random.default_rng(1), "belly", (10, 10, 20, 20))
    assert a.dtype == np.uint8 and a.shape == img.shape and np.array_equal(a, b)
    assert not np.array_equal(a, img)


def test_distortion_map_is_identity_at_zero_and_shared_by_all_layers():
    assert DR.distortion_map(32, 24, 20.0, 0.0, 0.0) is None
    m = DR.distortion_map(32, 24, 20.0, 0.05, 0.0)
    seg = np.arange(32 * 24).reshape(24, 32)
    out = DR.apply_map(seg, m)
    assert out.shape == seg.shape and out[12, 16] == seg[12, 16]  # centre fixed
    assert set(np.unique(out)) <= set(np.unique(seg))           # nearest pixel: no new label values


# ------------------------------------------------------------------------------------------- splits
def test_split_counts_cover_every_split():
    c = G.split_counts(10, {"train": 0.6, "val": 0.1, "test_seed": 0.1, "test_bank": 0.1, "test_species": 0.1})
    assert sum(c.values()) == 10 and min(c.values()) >= 1 and c["train"] == 6


def test_plans_are_disjoint_by_seed_bank_and_species(cfg):
    big = G.deep_merge(cfg, {"dataset": {"images": 600}})
    plans = G.plan_scenes(big)
    seeds = [p.seed for p in plans]
    assert len(seeds) == len(set(seeds)) == 100
    train_banks = {p.bank for p in plans if p.split != "test_bank"}
    assert train_banks.isdisjoint({p.bank for p in plans if p.split == "test_bank"})
    held = set(cfg["splits"]["heldout_species"])
    for p in plans:
        if p.split == "test_species":
            assert set(p.species) <= held
        else:
            assert held.isdisjoint(p.species)


# ------------------------------------------------------------------------------------------- classes
def test_every_mask_class_maps_to_a_geom_in_the_probe_scene(cfg, reg):
    sc = probe_scene(cfg, reg)
    have = {}
    for inst in sc.instances:
        have.setdefault(inst.cls, 0)
        have[inst.cls] += len(sc.inst_geoms[inst.iid])
    missing = [c for c in reg.box_classes if not have.get(c)]
    assert not missing, f"mask classes without a geom: {missing}"
    # mask: false classes own no geom (attribute or image-level tag only)
    for n, c in reg.classes.items():
        if not c.mask:
            assert n not in have
    assert sc.scene_tags.get("ribbon_colour")


def test_label_geoms_follow_the_group_rules(cfg, reg):
    sc = probe_scene(cfg, reg)
    groups = sc.geom_group0
    assert np.all(sc.geom_inst[groups == L.QUERY_ONLY_GROUP] == -1)    # group 5 never labelled
    assert np.any(groups == L.QUERY_ONLY_GROUP)
    assert not np.any(groups == L.SOLO_GROUP)                          # group 2 reserved for solo passes
    for inst in sc.instances:
        want = reg.classes[inst.cls].group
        assert set(groups[sc.inst_geoms[inst.iid]]) == {want}, inst.cls
    # distinct geoms: no geom belongs to two instances
    all_ids = np.concatenate(sc.inst_geoms)
    assert all_ids.size == np.unique(all_ids).size


def test_scene_world_id_is_deterministic_and_seeded(cfg, reg):
    a, b, c = probe_scene(cfg, reg, 3), probe_scene(cfg, reg, 3), probe_scene(cfg, reg, 4)
    assert a.world_id == b.world_id != c.world_id
    assert a.model.ngeom == b.model.ngeom
    assert any(v.startswith("synth_trees@") for v in a.versions)


def test_compose_source_selects_registered_modules_by_name(cfg, reg):
    import mujoco
    from atharv_nandi.sim.world import compose

    def attach(spec, field, ctx):
        spec.worldbody.add_geom(name="wp4_crate_7", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2],
                                pos=[1, 1, 0.2], group=4, contype=0, conaffinity=0)

    c2 = G.deep_merge(cfg, {"compose_label_patterns": {"crate": r"crate"}})
    with compose.temporary_module("wp40_test_crates", 80, attach):
        sc = S.build_compose_scene(S.ScenePlan(seed=1), c2, reg, ["wp40_test_crates"])
    crates = [i for i in sc.instances if i.cls == "crate"]
    assert len(crates) == 1 and len(sc.inst_geoms[crates[0].iid]) == 1
    with pytest.raises(KeyError):
        S.build_compose_scene(S.ScenePlan(seed=1), c2, reg, ["not_registered"])


# ------------------------------------------------------------------------------------------- RLE / COCO
def test_rle_roundtrip_and_bbox():
    rng = np.random.default_rng(2)
    for _ in range(20):
        m = rng.random((17, 23)) < 0.3
        m[0, 0] = bool(rng.integers(0, 2))
        r = L.rle_encode(m)
        assert sum(r["counts"]) == m.size
        assert np.array_equal(L.rle_decode(r), m)
    m = np.zeros((10, 10), bool)
    m[2:5, 3:9] = True
    assert L.bbox_of(m) == [3, 2, 6, 3]


def _tiny_coco():
    m = np.zeros((8, 10), bool)
    m[1:4, 2:6] = True
    return {"info": {}, "images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 8}],
            "categories": [{"id": 1, "name": "fruit"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": L.bbox_of(m), "area": int(m.sum()),
                             "segmentation": L.rle_encode(m), "iscrowd": 0}]}


def test_coco_validator_accepts_good_and_refuses_bad():
    good = _tiny_coco()
    assert L.validate_coco(good)["annotations"] == 1
    for mutate in (lambda c: c["annotations"][0].update(area=99),
                   lambda c: c["annotations"][0].update(bbox=[0, 0, 11, 2]),
                   lambda c: c["annotations"][0].update(category_id=7),
                   lambda c: c["annotations"].append(dict(c["annotations"][0])),
                   lambda c: c["annotations"][0]["segmentation"]["counts"].append(1)):
        bad = copy.deepcopy(good)
        mutate(bad)
        with pytest.raises(L.CocoError):
            L.validate_coco(bad)


def test_png_roundtrip_8_and_16_bit(tmp_path):
    rng = np.random.default_rng(3)
    for img in (rng.integers(0, 255, (5, 7, 3), dtype=np.uint8), rng.integers(0, 255, (5, 7), dtype=np.uint8),
                rng.integers(0, 65535, (5, 7), dtype=np.uint16)):
        p = tmp_path / "x.png"
        p.write_bytes(L.png_bytes(img))
        assert np.array_equal(L.png_read(str(p)), img)


def test_min_visible_px_scales_with_width():
    assert L.min_visible_px(320) == 12 and L.min_visible_px(640) == 24


# ------------------------------------------------------------------------------------------- YOLO
def _manifest(ds, rels, splits):
    from tools.synth import CLASSES_YAML
    from tools.synth import manifest as MF
    return MF.build_manifest(str(ds), rels, cfg=G.load_config(), classes_path=CLASSES_YAML, fingerprint={},
                             scene_seeds={}, splits=splits, date="20261117", git_sha="x", mujoco_version="3.3.7")


def test_yolo_export_lines_and_crown_tiling(tmp_path):
    ds = tmp_path / "ds"
    (ds / "annotations").mkdir(parents=True)
    (ds / "images").mkdir()
    img = np.zeros((480, 640, 3), np.uint8)
    (ds / "images" / "c.png").write_bytes(L.png_bytes(img))
    m1 = np.zeros((480, 640), bool)
    m1[10:30, 20:60] = True        # top-left tile only
    m2 = np.zeros((480, 640), bool)
    m2[200:280, 300:340] = True    # straddles all four tiles
    anns = [{"id": i + 1, "image_id": 1, "category_id": 1 + i, "bbox": L.bbox_of(m), "area": int(m.sum()),
             "segmentation": L.rle_encode(m), "iscrowd": 0} for i, m in enumerate((m1, m2))]
    coco = {"info": {}, "images": [{"id": 1, "file_name": "images/c.png", "width": 640, "height": 480,
                                    "camera": "crown"}],
            "categories": [{"id": 1, "name": "fruit"}, {"id": 2, "name": "person"}], "annotations": anns}
    (ds / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    _manifest(ds, ["images/c.png", "annotations/instances_train.json"], {"train": ["c"]})
    out = tmp_path / "yolo"
    rep = EY.export(str(ds), str(out))
    line = (out / "labels" / "train" / "c.txt").read_text().splitlines()
    assert len(line) == 2 and line[0].split()[0] == "0" and line[1].split()[0] == "1"
    cx, cy, w, h = map(float, line[0].split()[1:])
    assert abs(cx - 40 / 640) < 1e-6 and abs(cy - 20 / 480) < 1e-6 and abs(w - 40 / 640) < 1e-6
    assert (out / "classes.txt").read_text().split() == ["fruit", "person"]
    assert rep["boxes"]["train"] == 2
    out2 = tmp_path / "yolo_t"
    rep2 = EY.export(str(ds), str(out2), tile=(320, 240))
    tiles = sorted(os.listdir(out2 / "labels" / "train"))
    assert len(tiles) == 4
    assert rep2["boxes"]["train"] == 1 + 4      # fruit in one tile, the straddling box in all four
    t00 = (out2 / "labels" / "train" / "c_t0_0.txt").read_text().splitlines()
    assert len(t00) == 2


def test_yolo_export_refuses_stale_files_and_a_non_empty_out(tmp_path):
    """Review WP40 #1: a stale annotation file next to a verified manifest must not reach data.yaml."""
    ds = tmp_path / "ds"
    (ds / "annotations").mkdir(parents=True)
    (ds / "images").mkdir()
    (ds / "images" / "a.png").write_bytes(L.png_bytes(np.zeros((24, 32, 3), np.uint8)))
    coco = {"info": {}, "images": [{"id": 1, "file_name": "images/a.png", "width": 32, "height": 24}],
            "categories": [{"id": 1, "name": "fruit"}], "annotations": []}
    (ds / "annotations" / "instances_train.json").write_text(json.dumps(coco))
    _manifest(ds, ["images/a.png", "annotations/instances_train.json"], {"train": ["a"]})
    rep = EY.export(str(ds), str(tmp_path / "y"))
    assert rep["splits"] == ["train"] and rep["dataset_id"].startswith("nds-20261117-")
    with pytest.raises(EY.ExportError, match="not empty"):
        EY.export(str(ds), str(tmp_path / "y"))
    assert EY.export(str(ds), str(tmp_path / "y"), overwrite=True)["splits"] == ["train"]
    # a stale split from an earlier run in the same folder: verify flags it, export refuses
    (ds / "annotations" / "instances_val.json").write_text(json.dumps(coco))
    from tools.synth import manifest as MF
    v = MF.verify(str(ds))
    assert not v["ok"] and "untracked annotations/instances_val.json" in v["problems"]
    with pytest.raises(EY.ExportError, match="does not verify"):
        EY.export(str(ds), str(tmp_path / "y2"))
    assert not (tmp_path / "y2").exists()


def test_generate_refuses_a_non_empty_out_dir(tmp_path):
    d = tmp_path / "nds"
    G.prepare_out_dir(str(d))                       # absent: created
    assert d.is_dir()
    (d / "annotations").mkdir()
    (d / "annotations" / "instances_val.json").write_text("{}")
    (d / "manifest.json").write_text("{}")
    with pytest.raises(G.SynthConfigError, match="not empty"):
        G.prepare_out_dir(str(d))
    with pytest.raises(G.SynthConfigError, match="not empty"):
        G.generate(str(d), G.load_config(), calib=False, log=lambda s: None)
    (d / "notes.txt").write_text("mine")
    with pytest.raises(G.SynthConfigError, match="does not write"):
        G.prepare_out_dir(str(d), overwrite=True)  # never deletes a file it did not write
    (d / "notes.txt").unlink()
    G.prepare_out_dir(str(d), overwrite=True)
    assert os.listdir(d) == []


WIDE = [(320, 240, 78.0), (320, 240, 88.0), (320, 240, 90.0), (320, 240, 94.0), (640, 480, 94.0)]
KS = [(-0.06, -0.01), (-0.06, 0.0), (0.06, 0.01), (0.06, -0.01), (-0.06, 0.01), (-0.02, 0.0)]


@pytest.mark.parametrize("w, h, fovy", WIDE)
@pytest.mark.parametrize("k1, k2", KS)
def test_distortion_map_follows_the_recorded_model_on_wide_rigs(w, h, fovy, k1, k2):
    """Review WP40 #3: the forward model of the recorded (effective) k1, k2 applied to every source lands on
    its output pixel; no source is clipped to the frame edge; the model is monotone where it is used."""
    fy = G.intrinsics(w, h, fovy)["fy"]
    m = DR.distortion_map(w, h, fy, k1, k2)
    assert m is not None and 0.0 < m.scale <= 1.0
    assert abs(m.k1 / k1 - m.scale) < 1e-9 and (k2 == 0 or abs(m.k2 / k2 - m.scale) < 1e-9)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rcx, rcy = cx + m.pad_x, cy + m.pad_y
    bw, bh = DR.render_buffer(w, h)
    assert m.render_w <= bw and m.render_h <= bh
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)

    def forward(xs, ys):
        xu, yu = (xs - rcx) / fy, (ys - rcy) / fy
        r2 = xu * xu + yu * yu
        f = 1.0 + m.k1 * r2 + m.k2 * r2 * r2
        return xu * f * fy + cx, yu * f * fy + cy

    fu, fv = forward(m.xs, m.ys)
    assert np.max(np.hypot(fu - u, fv - v)) < 1e-6          # exact inverse of the recorded model
    # continuous sources lie inside the padded render, so rounding never clips
    assert m.xs.min() >= -0.5 and m.ys.min() >= -0.5
    assert m.xs.max() <= m.render_w - 0.5 and m.ys.max() <= m.render_h - 0.5
    # monotone radial model up to the widest source used
    rmax = float(np.max(np.hypot(m.xs - rcx, m.ys - rcy))) / fy
    r = np.linspace(0.0, rmax, 2001)
    # nearest-pixel error only: half a pixel diagonal times the local stretch of the model
    slope = 1.0 + 3 * m.k1 * r ** 2 + 5 * m.k2 * r ** 4
    stretch = float(np.max(np.maximum(slope, 1.0 + m.k1 * r ** 2 + m.k2 * r ** 4)))
    fu, fv = forward(m.su.astype(float), m.sv.astype(float))
    assert np.max(np.hypot(fu - u, fv - v)) <= 0.5 * np.sqrt(2.0) * max(stretch, 1.0) + 1e-6
    assert np.all(1.0 + 3 * m.k1 * r ** 2 + 5 * m.k2 * r ** 4 >= DR.MIN_SLOPE - 1e-9)
    if k1 < 0 and k2 <= 0:
        assert m.pad_x > 0 and m.pad_y > 0                 # barrel: rendered with a margin


def test_distortion_scale_is_one_when_the_sampled_model_is_already_valid():
    fy = G.intrinsics(320, 240, 45.0)["fy"]                 # crown-like
    m = DR.distortion_map(320, 240, fy, -0.02, 0.0)
    assert m.scale == 1.0 and m.k1 == -0.02


def test_no_ultralytics_anywhere_in_the_generator():
    d = os.path.join(REPO, "tools", "synth")
    for f in os.listdir(d):
        if f.endswith(".py"):
            src = open(os.path.join(d, f), encoding="utf-8").read()
            assert not re.search(r"^\s*(import|from)\s+ultralytics", src, re.M), f


# ------------------------------------------------------------------------------------------- gl tier
@pytest.fixture(scope="module")
def dataset60(tmp_path_factory, cfg):
    if not GL:
        pytest.skip("no offscreen GL backend")
    out = str(tmp_path_factory.mktemp("nds60"))
    man = G.generate(out, cfg, date="20261117", calib=True)
    return out, man


@needs_gl
@pytest.mark.gl
def test_60_images_at_320x240_at_least_4_img_per_s(dataset60):
    out, man = dataset60
    st = man["run_stats"]
    print(f"WP40 throughput: {st['images']} images {st['width']}x{st['height']} in {st['gen_s']} s = "
          f"{st['img_per_s']} img/s, {st['render_passes']} GL passes, {st['annotations']} annotations, "
          f"cpu_count {st['cpu_count']}")
    assert st["images"] == 60 and st["width"] == 320 and st["height"] == 240
    assert st["img_per_s"] >= 4.0


@needs_gl
@pytest.mark.gl
def test_coco_validates_and_semantic_agrees(dataset60, reg):
    out, man = dataset60
    n_img = 0
    for split, names in man["splits"].items():
        coco = json.load(open(os.path.join(out, "annotations", f"instances_{split}.json")))
        rep = L.validate_coco(coco)
        n_img += rep["images"]
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        for im in coco["images"][:4]:
            sem = L.png_read(os.path.join(out, im["semantic"]))
            depth = np.load(os.path.join(out, im["depth"]))["depth"]
            assert sem.shape == (240, 320) and depth.dtype == np.float16
            assert np.all((depth == 0) == (sem == reg.sem_id("sky")))
            for a in coco["annotations"]:
                if a["image_id"] != im["id"]:
                    continue
                m = L.rle_decode(a["segmentation"])
                want = reg.sem_id(reg.classes[cats[a["category_id"]]].semantic)
                assert np.all(sem[m] == want)
                assert 0.0 <= a["attrs"]["occlusion"] <= 1.0
                assert a["area"] >= 12
    assert n_img == 60


@needs_gl
@pytest.mark.gl
def test_masks_equal_the_seg_render(cfg, reg):
    """Every COCO mask is exactly the instance's pixels in the seg pass, and our seg decode equals
    mujoco.Renderer's own segmentation output."""
    import mujoco
    sc = probe_scene(cfg, reg)
    sr = G.SceneRenderer(sc, 320, 240, shadows=False)
    rng = np.random.default_rng(0)
    checked = 0
    try:
        for k in range(6):
            view = S.sample_view(sc, cfg["cameras"], rng)
            knobs = DR.sample(cfg["dr"], rng, 0)
            out = G.label_view(sr, view, knobs, rng, 12, "fruiting")
            for a in out["annotations"]:
                m = L.rle_decode(a["segmentation"])
                assert np.array_equal(m, out["inst"] == a["instance_id"])
                assert L.bbox_of(m) == a["bbox"] and int(m.sum()) == a["area"]
                checked += 1
            # independent seg render with the public API, same model state and camera
            r = mujoco.Renderer(sc.model, 240, 320)
            try:
                opt = mujoco.MjvOption()
                for g in range(6):
                    opt.geomgroup[g] = 1 if g in (0, 1, 3, 4) else 0
                r.update_scene(sc.data, camera=view.mjv_camera(), scene_option=opt)
                r.enable_segmentation_rendering()
                ref = r.render()
            finally:
                r.close()
            ref_gid = L.geom_map(ref)
            ref_gid = DR.apply_map(ref_gid, DR.distortion_map(320, 240, out["K"]["fy"], out["K"]["k1"], out["K"]["k2"]))
            assert np.array_equal(ref_gid, out["gid"])
    finally:
        sr.close()
    assert checked > 10


@needs_gl
@pytest.mark.gl
def test_barrel_distortion_samples_a_padded_render_not_the_frame_edge(cfg, reg):
    """Review WP40 #3, end to end: with strong barrel distortion the labels equal an independent
    mujoco.Renderer seg pass of the padded pinhole frame put through the recorded map."""
    import math
    import mujoco
    sc = probe_scene(cfg, reg)
    sr = G.SceneRenderer(sc, 320, 240, shadows=False)
    rng = np.random.default_rng(3)
    padded = 0
    try:
        for k in range(4):
            view = S.sample_view(sc, cfg["cameras"], rng)
            knobs = DR.sample(cfg["dr"], rng, 0)
            knobs["distortion_k1"], knobs["distortion_k2"] = -0.06, -0.01
            out = G.label_view(sr, view, knobs, rng, 12, "fruiting")
            K, Kr = out["K"], out["K_render"]
            m = DR.distortion_map(320, 240, K["fy"], -0.06, -0.01)
            assert (m.k1, m.k2) == (K["k1"], K["k2"]) and K["distortion_scale"] == m.scale
            assert (Kr["width"], Kr["height"]) == (m.render_w, m.render_h) and Kr["fy"] == pytest.approx(K["fy"])
            padded += int(m.pad_x > 0)
            assert out["rgb"].shape == (240, 320, 3) and out["depth"].shape == (240, 320)
            fovy0 = float(sc.model.vis.global_.fovy)
            sc.model.vis.global_.fovy = math.degrees(2 * math.atan((m.render_h / 2.0) / K["fy"]))
            r = mujoco.Renderer(sc.model, m.render_h, m.render_w)
            try:
                opt = mujoco.MjvOption()
                for g in range(6):
                    opt.geomgroup[g] = 1 if g in (0, 1, 3, 4) else 0
                r.update_scene(sc.data, camera=view.mjv_camera(), scene_option=opt)
                r.enable_segmentation_rendering()
                ref = r.render()
            finally:
                r.close()
                sc.model.vis.global_.fovy = fovy0
            assert np.array_equal(DR.apply_map(L.geom_map(ref), m), out["gid"])
    finally:
        sr.close()
    assert padded == 4


@needs_gl
@pytest.mark.gl
def test_calibration_recovers_intrinsics_within_1_percent(dataset60):
    from tools.synth import calib_check
    out, _ = dataset60
    rep = calib_check.check(out, 0.01)
    for rig, r in rep["rigs"].items():
        print(f"WP40 calib {rig}: max rel err {r.get('max_rel_err')} intrinsics, "
              f"{r.get('extrinsic_centre_rel_err_max')} camera centre, {r.get('views')} views")
    assert rep["ok"], rep
    assert set(rep["rigs"]) == {"ring", "crown", "wrist", "head", "belly", "rear"}


@needs_gl
@pytest.mark.gl
def test_written_splits_are_disjoint_by_seed_bank_and_species(dataset60, cfg):
    out, man = dataset60
    sp = json.load(open(os.path.join(out, "splits.json")))
    seeds = [s for v in sp["scene_seeds"].values() for s in v]
    assert len(seeds) == len(set(seeds))
    names = [n for v in sp["splits"].values() for n in v]
    assert len(names) == len(set(names)) == 60
    assert set(sp["splits"]) == {"train", "val", "test_seed", "test_bank", "test_species"}
    others = {b for s, v in sp["banks"].items() if s != "test_bank" for b in v}
    assert others.isdisjoint(sp["banks"]["test_bank"])
    held = set(cfg["splits"]["heldout_species"])
    for split in sp["splits"]:
        coco = json.load(open(os.path.join(out, "annotations", f"instances_{split}.json")))
        species = {a["attrs"]["species"] for a in coco["annotations"] if "species" in a["attrs"]}
        if split == "test_species":
            assert species and species <= held
        else:
            assert species.isdisjoint(held), split
    assert any(k.startswith("val/") for k in sp["per_camera"])        # per-camera validation lists
