"""WP40 synthetic dataset generator (CAMERA-TRAINING C.2, C.3, C.7, C.8; ORCHARD-MVP-SPEC 3.10).

Offline only, never in the live stack. Lane 0: CPU llvmpipe on the capped runner or a CI runner.

    scenes.py       stub scenes through the WP0 compose interface (later: registered compose modules)
    dr.py           domain randomisation sampler (config/synth.yaml `dr`), photometric post-process
    labels.py       class registry (config/synth_classes.yaml), seg -> instance masks, RLE, COCO checks
    gen_dataset.py  renders RGB + seg + depth, writes COCO / semantic PNG / depth npz / metadata
    manifest.py     sha256 per file, Merkle root, renderer fingerprint, dataset id nds-<yyyymmdd>-<merkle8>
    export_yolo.py  derived YOLO-txt (our own exporter; Ultralytics tooling is never used)
    calib_check.py  sim calibration board per camera rig; recovers intrinsics within 1 %

Run from the repo root: ``python3 -m tools.synth.gen_dataset --out /tmp/nds`` (needs a GL backend:
``MUJOCO_GL=glfw xvfb-run -a ...`` in the sim image).
"""
import os
import sys

__version__ = "1.0.0"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(REPO, "config")
SYNTH_YAML = os.path.join(CONFIG_DIR, "synth.yaml")
CLASSES_YAML = os.path.join(CONFIG_DIR, "synth_classes.yaml")


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
