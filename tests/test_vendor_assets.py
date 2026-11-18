"""The web console must work offline: Three.js is vendored, checksummed, and nothing points off-site."""
import hashlib
import json
import os
import re

import pytest

from atharv_nandi.core import web_scene

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB = os.path.join(ROOT, "atharv_nandi", "web")
VENDOR = os.path.join(WEB, "vendor")
SUMS = os.path.join(VENDOR, "SHA256SUMS")


def _sums():
    out = {}
    with open(SUMS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                digest, path = line.split(maxsplit=1)
                out[path.strip().lstrip("*")] = digest
    return out


def _own_sources():
    """index.html and the console's own *.js / *.css (vendored third-party code is covered by SHA256SUMS)."""
    files = []
    for name in sorted(os.listdir(WEB)):
        if name.endswith((".html", ".js", ".css")):
            files.append(os.path.join(WEB, name))
    return files


def test_sha256sums_cover_vendored_files_and_match():
    sums = _sums()
    assert "three/build/three.module.js" in sums
    assert "three/examples/jsm/controls/OrbitControls.js" in sums
    on_disk = set()
    for dirpath, _, names in os.walk(os.path.join(VENDOR, "three")):
        for n in names:
            on_disk.add(os.path.relpath(os.path.join(dirpath, n), VENDOR))
    assert on_disk == set(sums), "every vendored file must be listed in SHA256SUMS (and only those)"
    for rel, digest in sums.items():
        with open(os.path.join(VENDOR, rel), "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, rel


def test_vendored_three_is_r160():
    with open(os.path.join(VENDOR, "three", "build", "three.module.js"), encoding="utf-8") as f:
        head = f.read(4000)
    assert re.search(r"REVISION\s*=\s*'160'", head)


def _importmap():
    with open(os.path.join(WEB, "index.html"), encoding="utf-8") as f:
        html = f.read()
    m = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    assert m, "index.html must declare an importmap"
    return json.loads(m.group(1))["imports"]


def test_importmap_points_at_vendored_files_that_the_bridge_serves():
    imports = _importmap()
    assert imports["three"] == "/vendor/three/build/three.module.js"
    assert imports["three/addons/"] == "/vendor/three/examples/jsm/"
    assert web_scene.resolve_static_path(WEB, imports["three"]) is not None


def test_every_imported_addon_is_vendored():
    pat = re.compile(r"""from\s+['"]three/addons/([^'"]+)['"]""")
    used = set()
    for path in _own_sources():
        with open(path, encoding="utf-8") as f:
            used.update(pat.findall(f.read()))
    assert used, "expected at least OrbitControls"
    for rel in used:
        full = os.path.join(VENDOR, "three", "examples", "jsm", rel)
        assert os.path.isfile(full), f"addon {rel} imported but not vendored"
        with open(full, encoding="utf-8") as f:  # addons may only import 'three' itself
            deps = re.findall(r"""from\s+['"]([^'"]+)['"]""", f.read())
        assert set(deps) <= {"three"}, (rel, deps)


# The one allowed occurrence: choosing ws:/wss: for the same-origin WebSocket.
SAME_ORIGIN_WS = "location.protocol === 'https:'"


@pytest.mark.parametrize("path", _own_sources(), ids=os.path.basename)
def test_no_off_site_urls_in_console_sources(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().replace(SAME_ORIGIN_WS, "")
    assert "http" not in text.lower(), f"{os.path.basename(path)} references an http(s) URL"
    assert not re.search(r"wss?://(?!\$\{location\.host\})", text), "WebSocket must be same-origin"
    assert "//cdn" not in text and "unpkg" not in text and "jsdelivr" not in text
