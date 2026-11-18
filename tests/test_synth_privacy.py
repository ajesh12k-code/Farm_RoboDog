"""WP40 privacy rules for synthetic data (CAMERA-TRAINING C.2, C.7; privacy rule; OQ-48).

`person` is whole-body only: no face geometry, no identity attribute, in the class file, in the scene truth
and in the written COCO. Lane 0 refuses any real-frame input and any non-simulated provenance.
"""
import copy
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from tools.synth import gen_dataset as G  # noqa: E402
from tools.synth import labels as L  # noqa: E402
from tools.synth import load_yaml, CLASSES_YAML  # noqa: E402
from tools.synth import scenes as S  # noqa: E402
from test_synth_dataset import GL, needs_gl  # noqa: E402

FACE_WORDS = ("face", "eye", "nose", "mouth", "ear", "brow", "lip", "beard", "hair")
IDENTITY_WORDS = ("identity", "person_id", "name", "age", "gender", "sex", "ethnicity", "skin", "caste", "religion",
                  "re_id", "track_identity", "biometric")


def test_person_class_has_no_attributes_and_identity_attrs_are_forbidden():
    cfg = load_yaml(CLASSES_YAML)
    person = [c for c in cfg["classes"] if c["name"] == "person"]
    assert len(person) == 1 and person[0]["head"] == "B" and person[0]["mask"] is True
    assert not person[0].get("attrs")
    for w in ("face", "identity", "age", "gender", "ethnicity", "person_id"):
        assert w in cfg["forbidden_attrs"]
    for c in cfg["classes"]:
        for k in (c.get("attrs") or {}):
            assert not any(w == k or k.startswith(w) for w in IDENTITY_WORDS + ("face",)), (c["name"], k)


@pytest.mark.parametrize("cls, attrs", [
    ("person", {"pose": ["standing"]}),
    ("person", {"face_box": [True]}),
    ("animal", {"identity": ["a", "b"]}),
    ("crate", {"age": ["new"]}),
])
def test_class_validation_refuses_identity_or_person_attributes(cls, attrs):
    cfg = copy.deepcopy(load_yaml(CLASSES_YAML))
    for c in cfg["classes"]:
        if c["name"] == cls:
            c["attrs"] = attrs
    with pytest.raises(L.ClassConfigError):
        L.validate_classes(cfg)


def test_class_validation_refuses_dropping_the_forbidden_list():
    cfg = copy.deepcopy(load_yaml(CLASSES_YAML))
    cfg["forbidden_attrs"] = [a for a in cfg["forbidden_attrs"] if a != "face"]
    with pytest.raises(L.ClassConfigError, match="face"):
        L.validate_classes(cfg)


def test_truth_records_refuse_identity_attributes_at_runtime():
    reg = L.load_registry()
    rec = S.TruthRecorder(reg)
    with pytest.raises(L.ClassConfigError):
        rec.instance("person", {"gender": "f"}, (0, 0, 0))
    with pytest.raises(L.ClassConfigError):
        rec.instance("person", {"face": True}, (0, 0, 0))
    with pytest.raises(L.ClassConfigError):
        reg.check_attrs("person", {"occlusion": 0.1, "identity": "x"})
    reg.check_attrs("person", {"occlusion": 0.1, "season": "winter"})     # generic attrs only


def test_person_proxy_has_no_face_geometry():
    import mujoco
    cfg = G.load_config()
    reg = L.load_registry()
    persons = 0
    for seed in (1, 2, 3):
        plan = S.ScenePlan(seed=seed, kind="probe")
        sc = S.build_stub_scene(plan, cfg, reg, G.scene_knobs(cfg, plan))
        m = sc.model
        for inst in sc.instances:
            if inst.cls != "person":
                continue
            persons += 1
            assert inst.attrs == {}
            ids = sc.inst_geoms[inst.iid]
            types = [int(m.geom_type[g]) for g in ids]
            assert set(types) <= {int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)}
            assert all(int(m.geom_matid[g]) == -1 for g in ids)          # no texture, so no painted face
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ids]
            assert not any(w in n for n in names for w in FACE_WORDS)
            # the head is one plain ellipsoid with a single colour: no eyes, nose, mouth or ears
            top = max(ids, key=lambda g: m.geom_pos[g][2])
            assert int(m.geom_type[top]) == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
            near_head = [g for g in ids if np.linalg.norm(m.geom_pos[g] - m.geom_pos[top]) < 0.2 and g != top]
            assert not near_head
    assert persons == 3


def test_lane0_refuses_real_frames():
    cfg = G.load_config()
    for bad in ({"dataset": {"data_provenance": "field"}}, {"dataset": {"input_frames": "/x"}},
                {"real_images": ["a.jpg"]}):
        with pytest.raises(G.SynthConfigError):
            G.validate_config(G.deep_merge(cfg, bad))


@needs_gl
@pytest.mark.gl
def test_written_person_annotations_carry_no_identity(tmp_path):
    cfg = G.load_config(overrides={"dataset": {"images": 12, "images_per_scene": 6},
                                   "scene": {"p_person": 1.0}})
    man = G.generate(str(tmp_path), cfg, date="20261117", calib=False, log=lambda s: None)
    reg = L.load_registry()
    n_person = 0
    for split in man["splits"]:
        coco = json.load(open(tmp_path / "annotations" / f"instances_{split}.json"))
        pid = [c["id"] for c in coco["categories"] if c["name"] == "person"][0]
        assert [c for c in coco["categories"] if c["name"] == "person"][0]["attrs"] == {}
        for a in coco["annotations"]:
            if a["category_id"] == pid:
                n_person += 1
                assert set(a["attrs"]) <= set(reg.generic_attrs), a["attrs"]
            assert not set(a["attrs"]) & set(reg.forbidden)
    for f in os.listdir(tmp_path / "meta"):
        meta = json.load(open(tmp_path / "meta" / f))
        assert meta["data_provenance"] == "simulated"
    print(f"WP40 privacy: {n_person} person annotations, attrs only {list(reg.generic_attrs)}")
    assert n_person > 0
