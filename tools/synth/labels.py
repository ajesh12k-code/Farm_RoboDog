"""Labels from one extra segmentation pass (CAMERA-TRAINING C.2).

``Renderer.enable_segmentation_rendering`` gives (objid, objtype) per pixel. Each geom maps to a truth
record (class, instance, attrs) and a Head C semantic class, built by the scene. Boxes and RLE come from
the visible mask; occlusion comes from a solo render (``gen_dataset``). Geoms in group 5 (query-only) are
never labelled.

Also here: the class registry (``config/synth_classes.yaml``) with its privacy validation, COCO RLE
encode/decode, a COCO validator (no pycocotools in the image), and byte-deterministic PNG / npz writers.
"""
import json
import struct
import zipfile
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import CLASSES_YAML, load_yaml

QUERY_ONLY_GROUP = 5
SOLO_GROUP = 2          # free group the solo (occlusion) pass moves geoms into
HEADS = ("A", "B")


class ClassConfigError(ValueError):
    pass


# ------------------------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class ClassDef:
    name: str
    head: str
    mask: bool
    group: Optional[int]
    semantic: str
    attrs: Dict[str, Tuple]


class ClassRegistry:
    def __init__(self, cfg: dict):
        validate_classes(cfg)
        self.cfg = cfg
        self.semantic: List[str] = list(cfg["semantic"])
        self.generic_attrs = tuple(cfg.get("generic_attrs", ()))
        self.forbidden = tuple(cfg.get("forbidden_attrs", ()))
        self.classes: Dict[str, ClassDef] = {}
        for c in cfg["classes"]:
            self.classes[c["name"]] = ClassDef(c["name"], c["head"], bool(c["mask"]), c.get("group"),
                                               c["semantic"], {k: tuple(v) for k, v in (c.get("attrs") or {}).items()})
        # COCO category ids: 1.. in file order over the mask classes (box + RLE).
        self.box_classes = [n for n, c in self.classes.items() if c.mask]
        self.cat_id = {n: i + 1 for i, n in enumerate(self.box_classes)}

    def sem_id(self, name: str) -> int:
        return self.semantic.index(name)

    def coco_categories(self) -> List[dict]:
        out = []
        for n in self.box_classes:
            c = self.classes[n]
            out.append({"id": self.cat_id[n], "name": n, "supercategory": f"head_{c.head}",
                        "attrs": {k: list(v) for k, v in c.attrs.items()}})
        return out

    def check_attrs(self, cls: str, attrs: dict) -> None:
        """Refuse an attribute the class does not declare (or a forbidden one, always)."""
        c = self.classes[cls]
        for k, v in attrs.items():
            if k in self.forbidden:
                raise ClassConfigError(f"{cls}: forbidden attribute {k!r} (privacy rule)")
            if k in self.generic_attrs:
                continue
            if k not in c.attrs:
                raise ClassConfigError(f"{cls}: undeclared attribute {k!r}")
            if v not in c.attrs[k]:
                raise ClassConfigError(f"{cls}.{k}: value {v!r} not in {list(c.attrs[k])}")


def validate_classes(cfg: dict) -> None:
    if not isinstance(cfg, dict) or "classes" not in cfg or "semantic" not in cfg:
        raise ClassConfigError("synth_classes: needs `semantic` and `classes`")
    sem = list(cfg["semantic"])
    if "sky" not in sem:
        raise ClassConfigError("semantic: needs `sky` (background)")
    forbidden = set(cfg.get("forbidden_attrs") or ())
    for must in ("face", "identity", "age", "gender"):
        if must not in forbidden:
            raise ClassConfigError(f"forbidden_attrs must list {must!r} (privacy rule)")
    names = set()
    for c in cfg["classes"]:
        n = c.get("name")
        if not n or n in names:
            raise ClassConfigError(f"class name {n!r} missing or duplicated")
        names.add(n)
        if c.get("head") not in HEADS:
            raise ClassConfigError(f"{n}: head must be one of {HEADS}")
        if not isinstance(c.get("mask"), bool):
            raise ClassConfigError(f"{n}: mask must be true or false")
        if c.get("semantic") not in sem:
            raise ClassConfigError(f"{n}: semantic {c.get('semantic')!r} not in {sem}")
        if c["mask"]:
            g = c.get("group")
            if not isinstance(g, int) or not 0 <= g <= 4 or g == SOLO_GROUP:
                raise ClassConfigError(f"{n}: mask class needs a render group in 0..4 other than {SOLO_GROUP}")
        attrs = c.get("attrs") or {}
        bad = forbidden.intersection(attrs)
        if bad:
            raise ClassConfigError(f"{n}: forbidden attributes {sorted(bad)} (privacy rule)")
        if n == "person" and attrs:
            raise ClassConfigError("person: no attributes allowed (whole body only, no identity, C.7)")


def load_registry(path: str = CLASSES_YAML) -> ClassRegistry:
    return ClassRegistry(load_yaml(path))


# ------------------------------------------------------------------------------------------- RLE
def rle_encode(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE (column-major, first run is zeros)."""
    h, w = mask.shape
    flat = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    if flat.size == 0:
        return {"size": [h, w], "counts": []}
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs = [0] + runs
    return {"size": [int(h), int(w)], "counts": [int(r) for r in runs]}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in counts:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape((h, w), order="F")


def bbox_of(mask: np.ndarray) -> List[int]:
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    return [int(xs[0]), int(ys[0]), int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1)]


# ------------------------------------------------------------------------------------------- seg -> labels
def geom_map(seg: np.ndarray) -> np.ndarray:
    """(H, W) geom id per pixel, -1 where no geom (sky, skybox, non-geom objects)."""
    import mujoco
    objid, objtype = seg[..., 0], seg[..., 1]
    return np.where(objtype == int(mujoco.mjtObj.mjOBJ_GEOM), objid, -1).astype(np.int64)


def lookup(table: np.ndarray, gid: np.ndarray, fill: int) -> np.ndarray:
    out = np.full(gid.shape, fill, dtype=table.dtype)
    ok = gid >= 0
    out[ok] = table[gid[ok]]
    return out


def instance_map(gid: np.ndarray, geom_inst: np.ndarray) -> np.ndarray:
    return lookup(geom_inst, gid, -1)


def semantic_map(gid: np.ndarray, geom_sem: np.ndarray, sky: int) -> np.ndarray:
    return lookup(geom_sem, gid, sky).astype(np.uint8)


def min_visible_px(width: int, at_320: int = 12) -> int:
    """12 px at 320 wide, linear in width (24 at 640)."""
    return max(1, int(round(at_320 * width / 320.0)))


def visible_instances(inst: np.ndarray) -> Dict[int, int]:
    ids, counts = np.unique(inst[inst >= 0], return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts)}


# ------------------------------------------------------------------------------------------- COCO check
class CocoError(ValueError):
    pass


def validate_coco(coco: dict, check_masks: bool = True) -> dict:
    """Structural COCO instances check (keys, unique ids, refs, bbox in image, RLE sums, area, bbox = mask)."""
    for k in ("info", "images", "annotations", "categories"):
        if k not in coco:
            raise CocoError(f"missing top-level key {k!r}")
    imgs = {}
    for im in coco["images"]:
        for k in ("id", "file_name", "width", "height"):
            if k not in im:
                raise CocoError(f"image missing {k!r}")
        if im["id"] in imgs:
            raise CocoError(f"duplicate image id {im['id']}")
        imgs[im["id"]] = im
    cats = {c["id"] for c in coco["categories"]}
    if len(cats) != len(coco["categories"]):
        raise CocoError("duplicate category id")
    seen = set()
    for a in coco["annotations"]:
        for k in ("id", "image_id", "category_id", "bbox", "area", "segmentation", "iscrowd"):
            if k not in a:
                raise CocoError(f"annotation missing {k!r}")
        if a["id"] in seen:
            raise CocoError(f"duplicate annotation id {a['id']}")
        seen.add(a["id"])
        if a["image_id"] not in imgs:
            raise CocoError(f"annotation {a['id']}: unknown image")
        if a["category_id"] not in cats:
            raise CocoError(f"annotation {a['id']}: unknown category")
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > im["width"] or y + h > im["height"]:
            raise CocoError(f"annotation {a['id']}: bbox {a['bbox']} outside the image")
        seg = a["segmentation"]
        if not isinstance(seg, dict) or seg.get("size") != [im["height"], im["width"]]:
            raise CocoError(f"annotation {a['id']}: RLE size mismatch")
        if sum(seg["counts"]) != im["height"] * im["width"]:
            raise CocoError(f"annotation {a['id']}: RLE counts do not sum to H*W")
        if check_masks:
            m = rle_decode(seg)
            if int(m.sum()) != int(a["area"]):
                raise CocoError(f"annotation {a['id']}: area != mask pixels")
            if bbox_of(m) != list(a["bbox"]):
                raise CocoError(f"annotation {a['id']}: bbox != mask bbox")
    return {"images": len(imgs), "annotations": len(seen), "categories": len(cats)}


# ------------------------------------------------------------------------------------------- writers
def png_bytes(img: np.ndarray, level: int = 1) -> bytes:
    """Minimal deterministic PNG (8-bit grey or RGB; 16-bit grey for uint16)."""
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.dtype == np.uint16:
        depth, ctype, raw = 16, 0, img.astype(">u2")
    else:
        img = img.astype(np.uint8, copy=False)
        depth, ctype = 8, (0 if img.ndim == 2 else 2)
        raw = img
    rows = raw.reshape(h, -1).view(np.uint8).reshape(h, -1)
    data = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(data, level)) + chunk(b"IEND", b""))


def png_read(path: str) -> np.ndarray:
    """Reader for the files ``png_bytes`` writes (filter 0 only)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    pos, idat, hdr = 8, b"", None
    while pos < len(blob):
        n = struct.unpack(">I", blob[pos:pos + 4])[0]
        tag, payload = blob[pos + 4:pos + 8], blob[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            idat += payload
        pos += 12 + n
    w, h, depth, ctype = hdr[0], hdr[1], hdr[2], hdr[3]
    ch = 3 if ctype == 2 else 1
    bpp = ch * depth // 8
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * bpp)
    if np.any(raw[:, 0] != 0):
        raise ValueError("png_read: only filter type 0 is supported")
    px = raw[:, 1:]
    if depth == 16:
        return px.copy().view(">u2").reshape(h, w).astype(np.uint16)
    return px.reshape(h, w, ch) if ch == 3 else px.reshape(h, w).copy()


def npz_bytes_write(path: str, **arrays) -> None:
    """npz with a fixed zip timestamp (np.savez stamps the wall clock, which breaks byte identity)."""
    import io
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue())


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
