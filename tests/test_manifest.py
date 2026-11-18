"""Run manifest: per-file SHA-256 and Merkle root (atharv_nandi/core/manifest.py)."""
import hashlib
import json

import pytest

from atharv_nandi.core import manifest as mf


def make_run(d):
    (d / "meta.json").write_text('{"run_id": "r"}\n')
    (d / "kis_v1.jsonl").write_text('{"topic": "a"}\n{"topic": "b"}\n')
    (d / "harvest_attempts.jsonl").write_text('{"attempt": 1}\n')
    return d


def test_file_sha256(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert mf.sha256_file(str(p)) == hashlib.sha256(b"abc").hexdigest()


def test_leaf_is_path_nul_raw_digest():
    h = hashlib.sha256(b"x").hexdigest()
    assert mf.leaf_hash("a/b.json", h) == hashlib.sha256(b"a/b.json\x00" + bytes.fromhex(h)).digest()


def test_merkle_root_hand_computed():
    l = [hashlib.sha256(bytes([i])).digest() for i in range(3)]
    n01 = hashlib.sha256(l[0] + l[1]).digest()
    assert mf.merkle_root(l) == hashlib.sha256(n01 + l[2]).hexdigest()  # odd leaf carried up
    assert mf.merkle_root(l[:1]) == l[0].hex()
    assert mf.merkle_root([]) == hashlib.sha256(b"").hexdigest()


def test_root_stable_for_same_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = make_run(tmp_path / "a"), make_run(tmp_path / "b")
    assert mf.build_manifest(str(a))["merkle_root"] == mf.build_manifest(str(b))["merkle_root"]
    assert mf.build_manifest(str(a))["merkle_root"] == mf.build_manifest(str(a))["merkle_root"]


def test_root_changes_after_one_byte_edit(tmp_path):
    d = make_run(tmp_path)
    before = mf.build_manifest(str(d))["merkle_root"]
    data = bytearray((d / "kis_v1.jsonl").read_bytes())
    data[3] ^= 0x01
    (d / "kis_v1.jsonl").write_bytes(bytes(data))
    assert mf.build_manifest(str(d))["merkle_root"] != before


def test_root_changes_on_rename(tmp_path):
    d = make_run(tmp_path)
    before = mf.build_manifest(str(d))["merkle_root"]
    (d / "harvest_attempts.jsonl").rename(d / "attempts.jsonl")
    assert mf.build_manifest(str(d))["merkle_root"] != before


def test_manifest_excludes_itself_and_verifies(tmp_path):
    d = make_run(tmp_path)
    m = mf.write_manifest(str(d), {"run_id": "r", "device_id": "nandi-sim-01", "sw": "atharv-nandi x"}, final=True)
    assert [c["path"] for c in m["chunks"]] == ["harvest_attempts.jsonl", "kis_v1.jsonl", "meta.json"]
    assert m["data_provenance"] == "simulated" and m["final"] is True and m["device"] == "nandi-sim-01"
    ok, _ = mf.verify(str(d))
    assert ok
    m2 = mf.write_manifest(str(d), {"run_id": "r"})  # rewriting the manifest never changes the root
    assert m2["merkle_root"] == m["merkle_root"]
    (d / "meta.json").write_text('{"run_id": "R"}\n')
    assert mf.verify(str(d))[0] is False


def test_subdirectories_use_posix_paths(tmp_path):
    d = make_run(tmp_path)
    (d / "sub").mkdir()
    (d / "sub" / "x.json").write_text("{}")
    (d / "y.tmp").write_text("partial")
    paths = [c["path"] for c in mf.build_manifest(str(d))["chunks"]]
    assert "sub/x.json" in paths and "y.tmp" not in paths
    assert json.loads(json.dumps(paths)) == sorted(paths)


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_root_matches_reference_for_n_leaves(n):
    leaves = [hashlib.sha256(str(i).encode()).digest() for i in range(n)]
    level = leaves
    while len(level) > 1:
        nxt = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
        level = nxt + ([level[-1]] if len(level) % 2 else [])
    assert mf.merkle_root(leaves) == level[0].hex()
