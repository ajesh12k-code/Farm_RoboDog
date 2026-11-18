"""WP40 manifest, dataset id and determinism (CAMERA-TRAINING C.3, C.8).

Unit tier: Merkle root, id derivation (config incl. `shadows`, classes, renderer fingerprint), tamper
detection. `gl` tier: same seed on the same host gives the same bytes and id; changing synth.yaml, `shadows`
or the renderer fingerprint changes the id; labels are equal across "hosts" (a different llvmpipe thread
count in a subprocess, and an injected fingerprint) while RGB bytes are not compared.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from tools.synth import CLASSES_YAML  # noqa: E402
from tools.synth import gen_dataset as G  # noqa: E402
from tools.synth import labels as L  # noqa: E402
from tools.synth import manifest as MF  # noqa: E402
from test_synth_dataset import GL, needs_gl  # noqa: E402

FP = {"gl_vendor": "Mesa", "gl_renderer": "llvmpipe (LLVM 15.0.7, 128 bits)", "gl_version": "4.5 Mesa 23.2.1",
      "mesa_version": "23.2.1", "LP_NUM_THREADS": "", "MUJOCO_GL": "glfw", "mujoco_version": "3.3.7"}


def test_merkle_root_is_deterministic_and_order_and_content_sensitive():
    leaves = [b"a", b"b", b"c"]
    assert MF.merkle_root(leaves) == MF.merkle_root(list(leaves))
    assert MF.merkle_root(leaves) != MF.merkle_root([b"b", b"a", b"c"])
    assert MF.merkle_root(leaves) != MF.merkle_root([b"a", b"b", b"d"])
    assert MF.merkle_root(leaves) != MF.merkle_root(leaves + [b"c"])     # no odd-leaf duplication collision
    assert len(MF.merkle_root([])) == 64


def test_dataset_id_format():
    assert MF.dataset_id("0123456789abcdef" * 4, "20261117") == "nds-20261117-01234567"
    with pytest.raises(ValueError):
        MF.dataset_id("00" * 32, "2026-11-17")


def _fake_dataset(tmp_path, cfg, fp=FP, name="ds"):
    d = tmp_path / name
    (d / "meta").mkdir(parents=True)
    (d / "images").mkdir()
    (d / "images" / "a.png").write_bytes(L.png_bytes(np.zeros((4, 4, 3), np.uint8)))
    (d / "meta" / "a.json").write_bytes(L.json_bytes({"data_provenance": "simulated"}))
    return str(d), MF.build_manifest(str(d), ["images/a.png", "meta/a.json"], cfg=cfg, classes_path=CLASSES_YAML,
                                     fingerprint=fp, scene_seeds={}, splits={}, date="20261117", git_sha="x",
                                     mujoco_version="3.3.7")


def test_id_changes_with_config_shadows_and_fingerprint(tmp_path):
    cfg = G.load_config()
    _, base = _fake_dataset(tmp_path, cfg, name="a")
    _, same = _fake_dataset(tmp_path, cfg, name="b")
    assert base["dataset_id"] == same["dataset_id"]
    _, shad = _fake_dataset(tmp_path, G.deep_merge(cfg, {"dataset": {"shadows": True}}), name="c")
    _, knob = _fake_dataset(tmp_path, G.deep_merge(cfg, {"dr": {"exposure": {"range": [0.8, 1.2]}}}), name="d")
    _, gpu = _fake_dataset(tmp_path, cfg, fp={**FP, "gl_vendor": "NVIDIA Corporation", "mesa_version": "none"},
                           name="e")
    _, thr = _fake_dataset(tmp_path, cfg, fp={**FP, "LP_NUM_THREADS": "1"}, name="f")
    ids = {base["dataset_id"], shad["dataset_id"], knob["dataset_id"], gpu["dataset_id"], thr["dataset_id"]}
    assert len(ids) == 5
    assert base["renderer_fingerprint"] == FP and base["generator"]["git_sha"] == "x"


def test_verify_detects_tampering_and_non_simulated_provenance(tmp_path):
    cfg = G.load_config()
    d, man = _fake_dataset(tmp_path, cfg)
    assert MF.verify(d)["ok"]
    with open(os.path.join(d, "images", "a.png"), "ab") as fh:
        fh.write(b"x")
    rep = MF.verify(d)
    assert not rep["ok"] and any("sha mismatch" in p for p in rep["problems"])
    d2, _ = _fake_dataset(tmp_path, cfg, name="p")
    with open(os.path.join(d2, "meta", "a.json"), "wb") as fh:
        fh.write(L.json_bytes({"data_provenance": "field"}))
    man2 = MF.load(d2)
    man2["files"]["meta/a.json"] = MF.sha256_file(os.path.join(d2, "meta", "a.json"))
    with open(os.path.join(d2, "manifest.json"), "wb") as fh:
        fh.write(L.json_bytes(man2))
    rep2 = MF.verify(d2)
    assert any("non-simulated" in p for p in rep2["problems"]) and any("merkle" in p for p in rep2["problems"])


def test_verify_reports_a_missing_meta_file_instead_of_crashing(tmp_path):
    """Review WP40 #2: a deleted meta/ file is a problem in the report, not a FileNotFoundError."""
    d, _ = _fake_dataset(tmp_path, G.load_config())
    os.remove(os.path.join(d, "meta", "a.json"))
    rep = MF.verify(d)
    assert rep["ok"] is False and "missing meta/a.json" in rep["problems"]
    assert MF.main([d]) == 1
    os.remove(os.path.join(d, "manifest.json"))
    assert MF.verify(d)["ok"] is False


def test_verify_flags_files_the_manifest_does_not_list(tmp_path):
    d, _ = _fake_dataset(tmp_path, G.load_config())
    os.makedirs(os.path.join(d, "annotations"))
    with open(os.path.join(d, "annotations", "instances_val.json"), "w") as fh:
        fh.write("{}")
    with open(os.path.join(d, "run_stats.json"), "w") as fh:     # unhashed by design: not flagged
        fh.write("{}")
    rep = MF.verify(d)
    assert not rep["ok"] and rep["problems"] == ["untracked annotations/instances_val.json"]


# ------------------------------------------------------------------------------------------- gl tier
SMALL = {"dataset": {"images": 12, "images_per_scene": 6}}


def _gen(out, overrides=None, fp=None, calib=False):
    cfg = G.load_config(overrides=G.deep_merge(SMALL, overrides or {}))
    return G.generate(str(out), cfg, date="20261117", calib=calib, fingerprint_override=fp, log=lambda s: None)


def _bytes(d):
    man = MF.load(str(d))
    return {p: open(os.path.join(str(d), p), "rb").read() for p in man["files"]}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    if not GL:
        pytest.skip("no offscreen GL backend")
    out = tmp_path_factory.mktemp("base")
    return out, _gen(out, calib=True)


@needs_gl
@pytest.mark.gl
def test_same_seed_same_host_same_bytes_and_id(base, tmp_path):
    a, ma = base
    mb = _gen(tmp_path / "b", calib=True)
    assert ma["dataset_id"] == mb["dataset_id"]
    ba, bb = _bytes(a), _bytes(tmp_path / "b")
    assert ba.keys() == bb.keys() and len(ba) > 12 * 4
    diff = [p for p in ba if ba[p] != bb[p]]
    assert not diff, diff[:5]
    assert open(os.path.join(str(a), "manifest.json"), "rb").read() == \
        open(os.path.join(str(tmp_path / "b"), "manifest.json"), "rb").read()
    assert MF.verify(str(a))["ok"]
    print(f"WP40 identity: {len(ba)} files byte-identical, id {ma['dataset_id']}")


@needs_gl
@pytest.mark.gl
def test_id_changes_with_synth_yaml_and_shadows(base, tmp_path):
    _, ma = base
    m_seed = _gen(tmp_path / "seed", {"dataset": {"seed": 8}})
    m_shadow = _gen(tmp_path / "shadow", {"dataset": {"shadows": True}}, calib=True)
    m_knob = _gen(tmp_path / "knob", {"dr": {"sensor_noise_sigma": {"range": [0.0, 2.0]}}})
    ids = {ma["dataset_id"], m_seed["dataset_id"], m_shadow["dataset_id"], m_knob["dataset_id"]}
    assert len(ids) == 4
    assert m_shadow["shadows"] is True and ma["shadows"] is False
    # shadows change RGB only: labels stay equal
    rep = MF.compare_labels(str(base[0]), str(tmp_path / "shadow"))
    assert rep["ok"], rep["problems"]
    print(f"WP40 ids: base {ma['dataset_id']}, seed 8 {m_seed['dataset_id']}, shadows {m_shadow['dataset_id']}, "
          f"knob {m_knob['dataset_id']}; shadow-run labels equal")


@needs_gl
@pytest.mark.gl
def test_renderer_fingerprint_changes_the_id_not_the_labels(base, tmp_path):
    a, ma = base
    fp = {**ma["renderer_fingerprint"], "gl_vendor": "NVIDIA Corporation", "gl_renderer": "NVIDIA L4/PCIe/SSE2",
          "mesa_version": "none"}
    mb = _gen(tmp_path / "gpu", fp=fp, calib=True)
    assert mb["dataset_id"] != ma["dataset_id"]
    assert MF.compare_labels(str(a), str(tmp_path / "gpu"))["ok"]


@needs_gl
@pytest.mark.gl
def test_cross_host_label_equality_with_a_different_llvmpipe_thread_count(base, tmp_path):
    """A second process with LP_NUM_THREADS=1 stands in for another host: different renderer fingerprint and
    id; labels, masks and semantic equal; depth within 1 mm (or one float16 step); RGB bytes not compared."""
    a, ma = base
    out = tmp_path / "lp1"
    code = ("from tools.synth import gen_dataset as G; "
            "cfg = G.load_config(overrides=%r); "
            "G.generate(%r, cfg, date='20261117', calib=True, log=lambda s: None)") % (SMALL, str(out))
    env = dict(os.environ, LP_NUM_THREADS="1")
    res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=150)
    assert res.returncode == 0, res.stderr[-2000:]
    mb = MF.load(str(out))
    assert mb["renderer_fingerprint"]["LP_NUM_THREADS"] == "1"
    assert mb["dataset_id"] != ma["dataset_id"]
    rep = MF.compare_labels(str(a), str(out))
    rgb_same = sum(open(os.path.join(str(a), p), "rb").read() == open(os.path.join(str(out), p), "rb").read()
                   for p in ma["files"] if p.startswith("images/"))
    print(f"WP40 cross-host stand-in: labels ok={rep['ok']}, max depth diff {rep['max_depth_diff_m']} m, "
          f"RGB files byte-equal {rgb_same}/12 (not asserted), ids {rep['ids']}")
    assert rep["ok"], rep["problems"]
