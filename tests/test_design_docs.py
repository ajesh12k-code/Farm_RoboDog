"""Checks for the pilot design set in docs/ (ROS-free, no network, no renderer).

Links resolve, no external plain-HTTP URLs or IP addresses, every /nandi/* name in docs/hld.md is either
in code under atharv_nandi/ or marked [PROPOSED], every /nandi/* name in atharv_nandi/ is listed in docs/hld.md,
Mermaid blocks are structurally sound, and the fidelity register rates every row.
"""
import os
import re

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(ROOT, "docs")
DOC_FILES = (
    "README.md",
    "hld.md",
    "state-machines.md",
    "safety-case.md",
    "test-plan.md",
    "sim-fidelity-register.md",
    "hal-sim-to-real.md",
    "runbook-field.md",
    "cap-libraries.md",
)
MERMAID_TYPES = ("flowchart", "graph", "stateDiagram-v2", "sequenceDiagram")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
NAME_RE = re.compile(r"(/nandi/[a-z0-9_]+(?:/[a-z0-9_]+)*|/compute_work_envelope)")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def doc_paths():
    return [os.path.join(DOCS, f) for f in DOC_FILES]


def mermaid_blocks(text):
    return re.findall(r"```mermaid\n(.*?)```", text, flags=re.S)


def sim_strings():
    """Every quoted /nandi/* (and /compute_work_envelope) string in the Python code under atharv_nandi/."""
    found = set()
    for dirpath, _, files in os.walk(os.path.join(ROOT, "atharv_nandi")):
        for f in files:
            if f.endswith(".py"):
                found |= set(re.findall(r"['\"](/nandi/[a-z0-9_]+(?:/[a-z0-9_]+)*|/compute_work_envelope)['\"]", read(os.path.join(dirpath, f))))
    return found


@pytest.mark.parametrize("name", DOC_FILES)
def test_design_doc_exists_and_is_not_empty(name):
    path = os.path.join(DOCS, name)
    assert os.path.isfile(path), path
    assert read(path).lstrip().startswith("# ")


@pytest.mark.parametrize("path", doc_paths() + [os.path.join(ROOT, "README.md")])
def test_relative_links_resolve(path):
    broken = []
    for target in LINK_RE.findall(read(path)):
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            continue
        rel = target.split("#", 1)[0]
        if not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(path), rel))):
            broken.append(target)
    assert not broken, f"{os.path.relpath(path, ROOT)}: broken links {broken}"


@pytest.mark.parametrize("path", doc_paths() + [os.path.join(ROOT, "README.md")])
def test_plain_http_only_to_localhost(path):
    bad = [m for m in re.findall(r"http://[^\s)>\]`'\"]+", read(path))
           if not re.match(r"http://(localhost|127\.0\.0\.1)([:/]|$)", m)]
    assert not bad, bad


@pytest.mark.parametrize("path", doc_paths())
def test_no_ip_addresses_except_loopback(path):
    ips = set(re.findall(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", read(path)))
    assert ips <= {"127.0.0.1"}, ips


def test_hld_contract_names_are_built_or_marked_proposed():
    in_code = sim_strings()
    assert "/nandi/sim_qpos" in in_code and "/nandi/auto_harvest" in in_code  # the grep itself works
    unknown = []
    for line in read(os.path.join(DOCS, "hld.md")).splitlines():
        for name in NAME_RE.findall(line):
            if name not in in_code and "[PROPOSED]" not in line:
                unknown.append(name)
    assert not unknown, f"names neither in atharv_nandi/ nor marked [PROPOSED]: {sorted(set(unknown))}"


def test_hld_lists_every_built_topic_and_service():
    hld = read(os.path.join(DOCS, "hld.md"))
    missing = sorted(n for n in sim_strings() if f"`{n}`" not in hld)
    assert not missing, f"in atharv_nandi/ but missing from docs/hld.md: {missing}"


def test_hld_rows_marked_built_are_really_in_code():
    in_code = sim_strings()
    wrong = []
    for line in read(os.path.join(DOCS, "hld.md")).splitlines():
        if line.startswith("| `/") and "`atharv_nandi/` [BUILT]" in line:
            name = NAME_RE.search(line).group(1)
            if name not in in_code:
                wrong.append(name)
    assert not wrong, wrong


def test_mermaid_blocks_are_structurally_sound():
    count = 0
    for path in doc_paths():
        for block in mermaid_blocks(read(path)):
            count += 1
            lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
            kind = lines[0].split()[0]
            where = f"{os.path.basename(path)} block starting {lines[0]!r}"
            assert kind in MERMAID_TYPES, where
            for ln in lines[1:]:
                assert ln.count('"') % 2 == 0, f"{where}: unbalanced quote in {ln!r}"
                for o, c in ("[]", "()", "{}"):
                    assert ln.count(o) == ln.count(c), f"{where}: unbalanced {o}{c} in {ln!r}"
                assert ";" not in ln, f"{where}: ';' ends a Mermaid statement early in {ln!r}"
            if kind == "stateDiagram-v2":
                state = r"(\[\*\]|[A-Za-z_][A-Za-z0-9_]*)"
                for ln in lines[1:]:
                    assert re.fullmatch(rf"{state} --> {state}(: [^:]+)?", ln), f"{where}: bad line {ln!r}"
    assert count >= 6  # node graph, sequence, mission, harvest, arm, e-stop


def test_state_tables_cover_every_diagram_state():
    text = read(os.path.join(DOCS, "state-machines.md"))
    for block in mermaid_blocks(text):
        if not block.startswith("stateDiagram-v2"):
            continue
        states = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*) -->", block)) | set(re.findall(r"--> ([A-Za-z_][A-Za-z0-9_]*)", block))
        for s in states:
            assert re.search(rf"(\| |, ){re.escape(s)}( \||,)", text), f"state {s} has no table row"


def test_fidelity_register_rates_every_row_and_covers_scope():
    rows = [ln for ln in read(os.path.join(DOCS, "sim-fidelity-register.md")).splitlines()
            if ln.startswith("| ") and not ln.startswith("| Item") and not ln.startswith("|---")]
    assert rows
    items = {}
    for ln in rows:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        items[cells[0]] = cells[1]
        assert re.match(r"^(faithful|approximated|not simulated)\b", cells[1]), ln
    for required in ("Contact and soil", "Servo torque and backlash", "Gripper friction", "Crop detachment",
                     "Camera and detector", "Real-time factor under llvmpipe", "Battery model", "Ground-truth pose"):
        assert required in items, required


def test_readme_links_the_design_set():
    readme = read(os.path.join(ROOT, "README.md"))
    assert "## Docs" in readme
    for name in DOC_FILES[1:]:
        assert f"docs/{name}" in readme, name
