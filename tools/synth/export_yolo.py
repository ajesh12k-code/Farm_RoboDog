"""Derived YOLO-txt export from the canonical COCO files (CAMERA-TRAINING C.2).

Our own exporter: the YOLO text format is fine to use, Ultralytics tooling is never used (licence rule; the
WP41 registry scan fails on any ``ultralytics``). One line per box: ``<class> <cx> <cy> <w> <h>``,
normalised to the image size, class index = position in ``classes.txt`` (the COCO category order).

Crown frames (7.2 mm/px) can be tiled for fruit (``--tile 320x240 --tile-rigs crown``): a frame larger
than the tile is cut into a grid of tiles, each box is re-derived from its RLE mask inside the tile and
kept when at least ``min_px`` pixels remain. Only a dataset whose manifest verifies is exported, and only the
splits and images that manifest lists.

    python3 -m tools.synth.export_yolo <dataset_dir> --out <yolo_dir> [--tile 320x240 --tile-rigs crown]
"""
import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import labels as L
from . import manifest as MF

EXPORT_ENTRIES = ("labels", "images", "classes.txt", "data.yaml")


class ExportError(ValueError):
    pass


def _prepare_out(out_dir: str, overwrite: bool) -> None:
    """Refuse a non-empty ``out_dir`` (stale labels would mix with this export); ``overwrite`` removes an
    earlier export's own layout and still refuses anything else."""
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        return
    entries = sorted(os.listdir(out_dir))
    if not entries:
        return
    if not overwrite:
        raise ExportError(f"--out {out_dir} is not empty ({entries[:5]}); use an empty folder or --overwrite")
    foreign = [e for e in entries if e not in EXPORT_ENTRIES and not e.endswith(".txt")]
    if foreign:
        raise ExportError(f"--out {out_dir} holds files an export does not write: {foreign[:5]}")
    for e in entries:
        p = os.path.join(out_dir, e)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def _line(cls: int, bbox: Sequence[float], w: int, h: int) -> str:
    x, y, bw, bh = bbox
    return f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"


def tiles(w: int, h: int, tw: int, th: int) -> List[Tuple[int, int]]:
    xs = list(range(0, max(1, w - tw + 1), tw)) or [0]
    ys = list(range(0, max(1, h - th + 1), th)) or [0]
    if xs[-1] + tw < w:
        xs.append(w - tw)
    if ys[-1] + th < h:
        ys.append(h - th)
    return [(x, y) for y in ys for x in xs]


def export(dataset_dir: str, out_dir: str, tile: Optional[Tuple[int, int]] = None,
           tile_rigs: Sequence[str] = ("crown",), min_px: Optional[int] = None,
           overwrite: bool = False) -> dict:
    """Export the splits the dataset's manifest lists, after ``manifest.verify`` passes (so a stale or
    untracked annotation file never reaches training). ``out_dir`` must be empty unless ``overwrite``."""
    rep = MF.verify(dataset_dir)
    if not rep["ok"]:
        raise ExportError(f"{dataset_dir} does not verify: {rep['problems'][:5]}")
    man = MF.load(dataset_dir)
    listed = set(man["files"])
    ann_dir = os.path.join(dataset_dir, "annotations")
    splits = sorted(man["splits"])
    for split in splits:
        if f"annotations/instances_{split}.json" not in listed:
            raise ExportError(f"split {split} has no annotation file in the manifest")
    _prepare_out(out_dir, overwrite)
    names: Optional[List[str]] = None
    counts: Dict[str, int] = {}
    for split in splits:
        with open(os.path.join(ann_dir, f"instances_{split}.json"), encoding="utf-8") as fh:
            coco = json.load(fh)
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        if names is None:
            names = [c["name"] for c in cats]
        cls_of = {c["id"]: i for i, c in enumerate(cats)}
        by_img: Dict[int, list] = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        lab_dir = os.path.join(out_dir, "labels", split)
        os.makedirs(lab_dir, exist_ok=True)
        listing = []
        for im in coco["images"]:
            if im["file_name"] not in listed:
                raise ExportError(f"{im['file_name']} is not in the manifest of {man['dataset_id']}")
            stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
            w, h = im["width"], im["height"]
            anns = by_img.get(im["id"], [])
            if tile and im.get("camera") in tile_rigs and (w > tile[0] or h > tile[1]):
                tw, th = tile
                mp = min_px or L.min_visible_px(tw)
                img = L.png_read(os.path.join(dataset_dir, im["file_name"]))
                masks = [(cls_of[a["category_id"]], L.rle_decode(a["segmentation"])) for a in anns]
                for (x0, y0) in tiles(w, h, tw, th):
                    tstem = f"{stem}_t{x0}_{y0}"
                    lines = []
                    for c, m in masks:
                        sub = m[y0:y0 + th, x0:x0 + tw]
                        if int(sub.sum()) >= mp:
                            lines.append(_line(c, L.bbox_of(sub), tw, th))
                    tile_img = os.path.join(out_dir, "images", split, f"{tstem}.png")
                    os.makedirs(os.path.dirname(tile_img), exist_ok=True)
                    with open(tile_img, "wb") as fh:
                        fh.write(L.png_bytes(img[y0:y0 + th, x0:x0 + tw]))
                    with open(os.path.join(lab_dir, f"{tstem}.txt"), "w", encoding="utf-8") as fh:
                        fh.write("".join(l + "\n" for l in lines))
                    listing.append(os.path.relpath(tile_img, out_dir))
                    counts[split] = counts.get(split, 0) + len(lines)
                continue
            lines = [_line(cls_of[a["category_id"]], a["bbox"], w, h) for a in anns]
            with open(os.path.join(lab_dir, f"{stem}.txt"), "w", encoding="utf-8") as fh:
                fh.write("".join(l + "\n" for l in lines))
            listing.append(os.path.relpath(os.path.join(dataset_dir, im["file_name"]), out_dir))
            counts[split] = counts.get(split, 0) + len(lines)
        with open(os.path.join(out_dir, f"{split}.txt"), "w", encoding="utf-8") as fh:
            fh.write("".join(p + "\n" for p in listing))
    names = names or []
    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as fh:
        fh.write("".join(n + "\n" for n in names))
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"# derived from {os.path.abspath(dataset_dir)} by tools.synth.export_yolo (no Ultralytics)\n")
        fh.write(f"# dataset_id: {man['dataset_id']}\n")
        fh.write(f"path: {os.path.abspath(out_dir)}\n")
        for s in splits:
            fh.write(f"{s}: {s}.txt\n")
        fh.write(f"nc: {len(names)}\nnames: [{', '.join(names)}]\n")
    return {"dataset_id": man["dataset_id"], "splits": splits, "classes": len(names), "boxes": counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="export YOLO-txt labels from a synth dataset")
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", help="WxH, e.g. 320x240")
    ap.add_argument("--tile-rigs", default="crown")
    ap.add_argument("--overwrite", action="store_true", help="replace an earlier export in --out")
    a = ap.parse_args(argv)
    tile = tuple(int(x) for x in a.tile.lower().split("x")) if a.tile else None
    try:
        rep = export(a.dataset, a.out, tile, a.tile_rigs.split(","), overwrite=a.overwrite)
    except ExportError as exc:
        print(f"export_yolo: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
