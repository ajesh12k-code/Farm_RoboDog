"""Run manifest: SHA-256 per file and a Merkle root over runs/<run_id>/* (platform mission-manifest style).

Tree (as the platform's hub drive layout describes it):
    chunks   every regular file under the run directory except manifest.json / manifest.sig /
             *.tmp, sorted by relative POSIX path
    leaf     SHA-256(path_utf8 || 0x00 || sha256(file) as 32 raw bytes)
    node     SHA-256(left || right); an odd node at the end of a level is carried up unchanged
             (never duplicated, so two different file lists cannot share a root)
    empty    merkle_root of no files = SHA-256(b"")

Not signed: the sim has no device key, so there is no manifest.sig. `kind` values (meta, log,
report) extend the platform's mcap/still/video/finding enum and are PROPOSED.
"""
import hashlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXCLUDE = {"manifest.json", "manifest.sig"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def leaf_hash(rel_path: str, file_sha256_hex: str) -> bytes:
    return hashlib.sha256(rel_path.encode("utf-8") + b"\x00" + bytes.fromhex(file_sha256_hex)).digest()


def merkle_root(leaves: Sequence[bytes]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = list(leaves)
    while len(level) > 1:
        nxt = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def _kind(rel: str) -> str:
    if rel.endswith(".jsonl"):
        return "log"
    if rel in ("meta.json",):
        return "meta"
    return "report"


def list_chunks(run_dir: str) -> List[Dict[str, Any]]:
    chunks = []
    for root, dirs, files in os.walk(run_dir):
        dirs.sort()
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, run_dir).replace(os.sep, "/")
            if rel in EXCLUDE or name.endswith(".tmp") or not os.path.isfile(full):
                continue
            chunks.append({"path": rel, "bytes": os.path.getsize(full), "sha256": sha256_file(full),
                           "kind": _kind(rel)})
    chunks.sort(key=lambda c: c["path"])
    return chunks


def root_of(chunks: Sequence[Dict[str, Any]]) -> str:
    return merkle_root([leaf_hash(c["path"], c["sha256"]) for c in sorted(chunks, key=lambda c: c["path"])])


def build_manifest(run_dir: str, meta: Optional[Dict[str, Any]] = None, final: bool = False,
                   closed_ms: Optional[int] = None) -> Dict[str, Any]:
    meta = meta or {}
    chunks = list_chunks(run_dir)
    return {
        "v": 1,
        "mission_id": meta.get("run_id") or os.path.basename(os.path.normpath(run_dir)),
        "tenant": meta.get("tenant"), "site": meta.get("site"), "device": meta.get("device_id"),
        "mode": "record",
        "started": meta.get("start_utc_ms"),
        "closed": closed_ms,
        "ts_src": "sim",
        "part": 1,
        "final": bool(final),
        "chunks": chunks,
        "merkle_root": root_of(chunks),
        "merkle": "leaf=sha256(path||0x00||sha256(file)), node=sha256(l||r), odd carried up, sorted by path",
        "sw": {"runtime": meta.get("sw")},
        "data_provenance": "simulated",
    }


def write_manifest(run_dir: str, meta: Optional[Dict[str, Any]] = None, final: bool = False,
                   closed_ms: Optional[int] = None) -> Dict[str, Any]:
    import json
    m = build_manifest(run_dir, meta, final, closed_ms)
    path = os.path.join(run_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    return m


def verify(run_dir: str) -> Tuple[bool, str]:
    """Recompute the root from the files on disk and compare with manifest.json."""
    import json
    with open(os.path.join(run_dir, "manifest.json"), "r", encoding="utf-8") as fh:
        m = json.load(fh)
    root = root_of(list_chunks(run_dir))
    return root == m.get("merkle_root"), root
