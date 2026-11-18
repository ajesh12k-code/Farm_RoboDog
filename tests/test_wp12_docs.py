"""WP12 docs checks (ROS-free, no network, no renderer): the contract fold-in, the assisted register, the orchard
hazards and beacon map of the safety case, the state tables against the code, and the test plan's file references.

ORCHARD-MVP-SPEC 3.9 (assisted register), 5.4 (contract additions), 7 (risks); WP12 ("every /nandi/* name in code or
[PROPOSED]"; hazards; fidelity entries; correction 0.083 m). The structural checks shared with the pilot design set
stay in tests/test_design_docs.py (WP0); this file only adds what WP12 promises on top.
"""
import ast
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
import test_design_docs as dd  # noqa: E402  (shared parsers: sim strings, Mermaid blocks)

ROOT = dd.ROOT
DOCS = dd.DOCS
SIM = os.path.join(ROOT, "atharv_nandi")

TAG_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")

# ORCHARD-MVP-SPEC 5.4: the kis/v1 ``tele/alert`` ids
TELE_ALERT_IDS = ("orbit_blocked", "crown_retract", "stuck", "local_link_lost_hold", "cloud_link_lost_hold",
                  "low_battery_return", "self_right_failed", "tool_fault", "person_hold_count")
# ORCHARD-MVP-SPEC 3.9: the assisted shortcuts every run card may list
OS39_ASSISTED = ("kinematic_mast", "virtual_gimbal", "rigid_mast", "geometric_crown_strike", "fruit_proxy_pool",
                 "logical_twist_release", "logical_cut", "kinematic_debris", "static_canopy", "lidar_through_foliage",
                 "decal_symptoms", "truth_orbit_range", "truth_tree_registry", "truth_base_pose_during_dwell",
                 "no_contact_underside_view", "scene_cull_4m")


def read_doc(name):
    return dd.read(os.path.join(DOCS, name))


def section(text, start_heading, stop_level="## "):
    """Text from the heading line that starts with ``start_heading`` to the next heading of ``stop_level``."""
    i = text.index(start_heading)
    j = text.find("\n" + stop_level, i + len(start_heading))
    return text[i:] if j < 0 else text[i:j]


def table_rows(text):
    return [ln for ln in text.splitlines() if ln.startswith("| ") and not ln.startswith("|---")]


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


# ---------------------------------------------------------------------- contract fold-in (hld.md section 7)
def fold_rows():
    sec = section(read_doc("hld.md"), "## 7 Orchard MVP contract")
    sec = sec.split("\n### ", 1)[0]          # the topic index only, not 7.1 / 7.2
    out = {}
    for ln in table_rows(sec):
        m = re.match(r"^\| `(/nandi/[a-z0-9_/]+)` \|", ln)
        if m:
            assert m.group(1) not in out, f"hld.md section 7 lists {m.group(1)} twice"
            out[m.group(1)] = cells(ln)
    return out


def test_hld_section7_status_matches_the_code():
    in_code = dd.sim_strings()
    stale = []
    for name, c in fold_rows().items():
        built = "`atharv_nandi/` [BUILT]" in c[-1]
        if built != (name in in_code) or (not built and "[PROPOSED]" not in c[-1]):
            stale.append(f"{name}: '{c[-1]}' but in code = {name in in_code}")
    assert not stale, stale


def unconsumed_console_commands():
    """Console command topics (``HUB_TOPICS_OUT``) that nothing in atharv_nandi/ reads: the name is quoted in no file but
    console_protocol.py, and the console itself does not subscribe to it (``HUB_TOPICS_IN``)."""
    from atharv_nandi.core import console_protocol as cp
    quoted = {}
    for dirpath, _, files in os.walk(SIM):
        for f in files:
            if f.endswith(".py") and f != "console_protocol.py":
                for n in re.findall(r"['\"](/nandi/[a-z0-9_/]+)['\"]", dd.read(os.path.join(dirpath, f))):
                    quoted.setdefault(n, set()).add(f)
    return {t for t in cp.HUB_TOPICS_OUT.values() if t not in quoted and t not in cp.HUB_TOPICS_IN.values()}


def test_hld_marks_console_commands_without_a_consumer():
    """Every "JSON in" topic the console publishes but nothing consumes says "consumer [PROPOSED]" in section 7 and
    has a row in section 6, and no row claims that for a topic that has a consumer."""
    rows = fold_rows()
    dead = {n for n in unconsumed_console_commands() if n in rows and "JSON in" in rows[n][1]}
    assert {"/nandi/preset/run", "/nandi/dock/cmd"} <= dead, dead   # remove from here when their hooks land
    marked = {n for n, c in rows.items() if "consumer [PROPOSED]" in c[-1]}
    assert marked == dead, f"section 7 'consumer [PROPOSED]' {sorted(marked)} != no consumer in atharv_nandi/ {sorted(dead)}"
    sec6 = section(read_doc("hld.md"), "## 6 Contract conflicts")
    assert all(f"`{n}`" in sec6 for n in dead), sec6
    readme = dd.read(os.path.join(ROOT, "README.md"))
    assert "All 94 are in code" not in readme and all(f"`{n}`" in readme for n in dead)


def test_every_nandi_name_in_code_is_in_hld():
    """WP12 acceptance: every /nandi/* name in code is listed in hld.md."""
    hld = read_doc("hld.md")
    missing = sorted(n for n in dd.sim_strings() if f"`{n}`" not in hld)
    assert not missing, f"names in atharv_nandi/ missing from docs/hld.md: {missing}"


def test_os54_contract_additions_have_a_status_row():
    sec = section(read_doc("hld.md"), "### 7.1 Orchard contract additions", "### ")
    for name in ("/nandi/mast/state", "/nandi/mast/cmd", "/nandi/preset/run", "/nandi/preset/state",
                 "/nandi/tree/record", "/nandi/tree360/progress", "/nandi/mark", "/nandi/jobsheet",
                 "/nandi/deepscan/request", "/nandi/wind", "/nandi/link/local", "/nandi/link/cloud"):
        assert f"`{name}`" in sec, name
    ids = TELE_ALERT_IDS
    in_code = set()
    for dirpath, _, files in os.walk(SIM):
        for f in files:
            if f.endswith(".py"):
                in_code |= set(re.findall(r"['\"]([a-z_]+)['\"]", dd.read(os.path.join(dirpath, f))))
    for i in ids:
        assert f"`{i}`" in sec, f"tele/alert id {i} missing from hld.md 7.1"
        assert i in in_code, f"hld.md 7.1 says every tele/alert id is in code, {i} is not"


# ---------------------------------------------------------------------- assisted register (fidelity register 2)
def _strs(node):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _strs(node.body) + _strs(node.orelse)
    if isinstance(node, ast.BinOp):
        return _strs(node.left) + _strs(node.right)
    return []


def _name(t):
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute):
        return t.attr
    if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant):
        return str(t.slice.value)
    return ""


def assisted_tags_in_code():
    """{tag: {files}} for every snake_case string the code puts into an ``assisted`` list, set, dict key or
    function (``"assisted": [...]``, ``ASSISTED = (...)``, ``x.assisted += [...]``, ``_assisted.add(...)``,
    ``assisted=[...]`` keyword, ``def assisted(): return [...]``)."""
    found = {}
    for dirpath, _, files in os.walk(SIM):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            for n in ast.walk(ast.parse(dd.read(path))):
                vals = []
                if isinstance(n, ast.Dict):
                    for k, v in zip(n.keys, n.values):
                        if isinstance(k, ast.Constant) and k.value == "assisted":
                            vals += _strs(v)
                elif isinstance(n, ast.AugAssign) and "assisted" in _name(n.target).lower():
                    vals += _strs(n.value)
                elif isinstance(n, ast.FunctionDef) and "assisted" in n.name.lower():
                    for r in ast.walk(n):
                        if isinstance(r, (ast.Return, ast.Assign)) and r.value is not None:
                            vals += _strs(r.value)
                elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                    tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                    if n.value is not None and any("assisted" in _name(t).lower() for t in tg):
                        vals += _strs(n.value)
                elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                      and n.func.attr in ("append", "add", "extend") and "assisted" in _name(n.func.value).lower()):
                    for a in n.args:
                        vals += _strs(a)
                elif isinstance(n, ast.keyword) and n.arg and "assisted" in n.arg.lower():
                    vals += _strs(n.value)
                for v in vals:
                    if TAG_RE.match(v):
                        found.setdefault(v, set()).add(os.path.relpath(path, ROOT))
    return found


def register_items():
    sec = section(read_doc("sim-fidelity-register.md"), "## 2 Assisted register")
    out = {}
    for ln in table_rows(sec):
        c = cells(ln)
        m = re.fullmatch(r"`([a-z0-9_]+)`", c[0])
        if m:
            out[m.group(1)] = c
    return out


def test_assisted_scan_finds_known_tags():
    tags = assisted_tags_in_code()
    for t in ("weld_grasp", "logical_cut", "map_masked_lidar", "kinematic_mast", "station_teleport", "truth_tag_pose"):
        assert t in tags, t  # the scan itself works


def test_every_assisted_tag_in_code_has_a_register_row():
    items = register_items()
    missing = {t: sorted(f) for t, f in assisted_tags_in_code().items() if t not in items}
    assert not missing, f"assisted tags in atharv_nandi/ without a row in sim-fidelity-register.md section 2: {missing}"


def test_register_rows_marked_built_are_in_code():
    """A section 2 row marked [BUILT] names a tag that is a string somewhere in atharv_nandi/ (an assisted list, or the
    module that applies it, e.g. scene_cull_4m). Behaviour without a tag string is marked [BUILT, no tag]."""
    blob = []
    for dirpath, _, files in os.walk(SIM):
        blob += [dd.read(os.path.join(dirpath, f)) for f in files if f.endswith(".py")]
    blob = "\n".join(blob)
    wrong = [tag for tag, c in register_items().items() if "[BUILT]" in c[2] and tag not in blob]
    assert not wrong, f"rows marked [BUILT] whose tag is not in atharv_nandi/ (use [BUILT, no tag] or [OS 3.9]): {wrong}"


def test_assisted_register_covers_os39_and_wp_change_requests():
    items = register_items()
    os39 = OS39_ASSISTED
    wp12_cr = ("dock_funnel_weld", "truth_fiducial_pose", "time_compressed_charge", "time_compressed_policy",
               "rigid_pad_compliance", "synthetic_raster_gap")
    wp_rows = ("truth_target_pose", "truth_base_pose", "station_teleport", "tray_teleport_unload", "weld_tool_mount",
               "weld_grasp", "hfield_terrain_0p05", "kinematic_actors", "soft_contact_sink", "noncolliding_laterals",
               "static_rope", "lighting_presets_assumed", "kinematic_transfer")
    missing = [t for t in (*os39, *wp12_cr, *wp_rows) if t not in items]
    assert not missing, missing
    for t, c in items.items():
        assert re.match(r"^(faithful|approximated|not simulated)\b", c[1]), (t, c[1])
    # WP12 acceptance names these three fidelity entries explicitly
    for t in ("rigid_mast", "geometric_crown_strike", "lidar_through_foliage", "static_canopy"):
        assert t in items, t


# ---------------------------------------------------------------------- safety case
def hazard_lines():
    return [ln for ln in table_rows(read_doc("safety-case.md")) if re.match(r"^\| HZ-\d\d ", ln)]


def hazard_rows():
    return {cells(ln)[0]: ln for ln in hazard_lines()}


def test_safety_case_has_the_wp12_orchard_hazards():
    ids = [cells(ln)[0] for ln in hazard_lines()]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    assert not dup, f"safety-case.md repeats hazard ids {dup}"
    rows = hazard_rows()
    assert len(rows) >= 20, sorted(rows)
    blob = "\n".join(rows.values()).lower()
    for words in (("crown strike", "snag"), ("wind",), ("fall envelope",), ("debris",), ("bunch",), ("blade",),
                  ("jaw",), ("orbit", "worker"), ("lateral",), ("animal",), ("local safety link",), ("beacon",)):
        assert all(w in blob for w in words), words
    for hz, ln in rows.items():
        c = cells(ln)
        assert re.fullmatch(r"S[1-4]( .*)?", c[3]) and re.fullmatch(r"L[1-4]", c[4]), hz
        assert re.fullmatch(r"L[1-4]", c[7]), hz


def test_safety_case_risk_cells_and_summary_match_the_table():
    """Risk = S x L in every row (the worst S when a row gives two), and the summary sentence names the top risk."""
    risk, sev = {}, {}
    for hz, ln in hazard_rows().items():
        c = cells(ln)
        s = max(int(x) for x in re.findall(r"S([1-4])", c[3]))
        l = int(c[4][1])
        assert int(c[5]) == s * l, (hz, c[3], c[4], c[5])
        risk[hz], sev[hz] = s * l, s
    top = max(risk.values())
    text = read_doc("safety-case.md")
    m = re.search(r"^(HZ-\d\d) has the highest risk before mitigation \(S(\d) x L(\d) = (\d+)\)", text, re.M)
    assert m, "safety-case.md needs the 'HZ-nn has the highest risk before mitigation (S x L = n)' summary"
    assert [k for k, v in risk.items() if v == top] == [m.group(1)] and int(m.group(4)) == top, (m.group(0), top)
    s4 = sorted(k for k, v in sev.items() if v == 4)
    summary = text[m.start():text.index("\n", m.start())]
    assert all(k in summary for k in s4), (s4, summary)


def test_safety_case_beacon_map_matches_the_code():
    from atharv_nandi.sim.hooks import beacon  # noqa: WPS433  (imports mujoco and controllers.sim_state; no ROS)
    sec = section(read_doc("safety-case.md"), "## 5 Beacon map")
    rows = [cells(ln) for ln in table_rows(sec) if re.match(r"^\| \d+ \|", ln)]
    names = {tuple(beacon.RED): "red", tuple(beacon.AMBER): "amber", tuple(beacon.CYAN): "cyan",
             tuple(beacon.BLUE): "blue", tuple(beacon.GREEN): "green"}
    assert [r[1].strip("`") for r in rows] == list(beacon.PRIORITY)
    for r in rows:
        st = r[1].strip("`")
        pat = beacon.DEFAULT_PATTERNS[st]
        assert int(r[0]) == beacon.PRIORITY.index(st) + 1
        assert r[2] == names[tuple(pat["rgb"])], st
        assert r[3] == pat["mode"], st
        assert float(r[4]) == pytest.approx(pat["hz"]), st
    assert set(beacon.SAFETY_STATES) == {"estop", "fault"} and "hook_fault" in beacon.FAULT_HOLD_REASONS


def test_safety_case_states_the_margin_correction_with_the_matrix_run():
    text = read_doc("safety-case.md")
    sec = section(text, "## 8 Tripod-margin correction")
    assert "0.083" in sec and "0.114" in sec
    with open(os.path.join(ROOT, "config", "crown_gates.json"), encoding="utf-8") as fh:
        run_id = json.load(fh)["matrix"]["meta"]["run_id"]
    assert run_id in sec, f"safety case 8 must cite the live matrix run {run_id}"


# ---------------------------------------------------------------------- state machines vs code
STATE_HEADERS = ("State", "Step", "Phase")


def state_table(sec_heading):
    """First-column names of the state tables only (header State / Step / Phase), not the transition tables."""
    sec = section(read_doc("state-machines.md"), sec_heading)
    names, header, prev_row = [], None, False
    for ln in sec.splitlines():
        is_row = ln.startswith("| ")
        if is_row and not prev_row:
            header = cells(ln)[0]          # first line of a table
        elif is_row and header in STATE_HEADERS:
            names += [x.strip().strip("`") for x in cells(ln)[0].split(",")]
        prev_row = is_row or ln.startswith("|---")
    return names


def mermaid_edges(sec_heading, index=0):
    """(from, to, label) of the ``index``-th stateDiagram block of a section."""
    sec = section(read_doc("state-machines.md"), sec_heading)
    block = dd.mermaid_blocks(sec)[index]
    out = []
    for ln in block.splitlines():
        m = re.match(r"^\s*(\S+) --> ([^:\s]+)\s*(?::\s*(.*))?$", ln)
        if m:
            out.append((m.group(1), m.group(2), (m.group(3) or "").strip()))
    return out


def test_tree360_phase_table_matches_the_code():
    from atharv_nandi.core import tree360
    assert sorted(state_table("## 8 TREE 360 phases")) == sorted(tree360.PHASES)
    edges = mermaid_edges("## 8 TREE 360 phases")
    nodes = {a for a, _, _ in edges} | {b for _, b, _ in edges}
    assert nodes - {"[*]"} == set(tree360.PHASES), nodes
    src = dd.read(os.path.join(SIM, "core", "tree360.py"))
    guard = re.search(r'if self\.phase not in \(([^)]*)\):\n\s*self\.phase = "stow"', src)
    no_stow = set(re.findall(r'"([a-z]+)"', guard.group(1)))
    moving = set(tree360.PHASES) - no_stow - {"stow", "done", "aborted"}
    aborts = {a for a, b, lab in edges if b == "stow" and lab.startswith("abort")}
    assert aborts == moving, f"abort -> stow edges {sorted(aborts)} != the code's {sorted(moving)}"
    assert ("reentry", "station") in {(a, b) for a, b, _ in edges}


def test_dock_tables_match_the_code():
    from atharv_nandi.core import dock_logic
    names = set(state_table("## 10 Dock alignment"))
    assert set(dock_logic.STATES) <= names
    assert set(dock_logic.SESSION_STEPS) <= names
    sec = section(read_doc("state-machines.md"), "## 10 Dock alignment")
    order = re.search(r"in this order: ([^;]*);", sec).group(1)
    assert re.findall(r"`([a-z_]+)`", order) == list(dock_logic.SESSION_STEPS)


def test_mission_and_work_tables_match_the_code():
    from atharv_nandi.core import mission_logic
    names = set(state_table("## 2 Mission lifecycle"))
    assert set(mission_logic.STATES) <= names
    src = dd.read(os.path.join(SIM, "core", "work_assist.py"))
    phases = re.search(r'self\.phase = "idle"\s*# ([a-z_| ]+)', src).group(1).replace(" ", "").split("|")
    assert set(phases) <= set(state_table("## 11 WORK assist"))


def test_recovery_priority_matches_the_code():
    from atharv_nandi.core import recovery_behaviours as rb
    sec = section(read_doc("state-machines.md"), "## 9 Recovery behaviours")
    doc = re.search(r"by priority ([A-Z -]+(?: > [A-Z -]+)+)\.", sec).group(1).split(" > ")
    code = {"fallen": "FALLEN", "estop_recover": "E-STOP RECOVERY", "local_link": "LOCAL LINK",
            "cloud_link": "CLOUD LINK", "return_home": "RETURN HOME", "stuck": "STUCK", "safe_park": "SAFE PARK"}
    assert doc == [code[p] for p in rb.PRIORITY]


def test_estop_recovery_machine_matches_the_code():
    """Section 9: the e-stop recovery diagram and step table use EstopRecovery's own step names, and a new hold
    restarts exactly from the steps the code restarts from."""
    import inspect
    from atharv_nandi.core import recovery_behaviours as rb
    src = inspect.getsource(rb.EstopRecovery)
    steps = set(re.findall(r'self\._to\("([a-z_]+)"', src)) | {"done"}
    edges = mermaid_edges("## 9 Recovery behaviours", 0)
    nodes = ({a for a, _, _ in edges} | {b for _, b, _ in edges}) - {"[*]"}
    assert nodes == steps, f"diagram {sorted(nodes)} != code {sorted(steps)}"
    assert set(state_table("## 9 Recovery behaviours")) >= steps
    restart = set(re.findall(r'"([a-z_]+)"', re.search(r"if self\.step in \(([^)]*)\) and o\.holding", src).group(1)))
    doc_restart = {a for a, b, lab in edges if b == "await_local" and lab.startswith("new hold")}
    assert doc_restart == restart, (doc_restart, restart)
    assert re.search(r"o\.t - self\.t_step >= 1\.0", src) and any(
        a == "stand" and b == "done" and "1.0 s" in lab for a, b, lab in edges)


def test_mast_brake_release_matches_the_latch():
    """Section 7: the end of a hold releases only the hold brake; a latched brake waits for release("estop")."""
    from atharv_nandi.core import mast
    hold, latched = mast.MastArbiter(), mast.MastArbiter()
    hold.brake(latch=False)
    latched.brake()                                  # explicit brake / estop request: latches by default
    assert hold.braked and not hold.brake_latched and latched.brake_latched
    sec = section(read_doc("state-machines.md"), "## 7 Crown mast")
    edges = mermaid_edges("## 7 Crown mast", 0)
    rel = [lab for a, b, lab in edges if a == "braked" and b == "up"]
    assert rel and "latched brake released" in rel[0] and "hold brake ended" in rel[0], rel
    assert "brake_latched" in sec and 'release("estop")' in sec


def test_engine_active_states_match_the_code():
    src = dd.read(os.path.join(SIM, "core", "workflow", "engine.py"))
    active = re.findall(r'"([a-z]+)"', re.search(r"ACTIVE_STATES = \(([^)]*)\)", src).group(1))
    sec = section(read_doc("state-machines.md"), "## 11 WORK assist")
    for s in active:
        assert f"`{s}`" in sec, s


# ---------------------------------------------------------------------- test plan and README
def test_test_plan_test_files_exist_or_are_proposed():
    bad = []
    for ln in read_doc("test-plan.md").splitlines():
        for f in re.findall(r"`(tests/test_[a-z0-9_]+\.py)", ln):
            if not os.path.exists(os.path.join(ROOT, f)) and "[PROPOSED]" not in ln:
                bad.append(f)
    assert not bad, f"test-plan.md names test files that do not exist (mark the row [PROPOSED]): {sorted(set(bad))}"


def test_test_plan_states_the_image_python():
    text = read_doc("test-plan.md")
    assert "3.10.12" in text


def test_readme_points_at_the_fold_in():
    readme = dd.read(os.path.join(ROOT, "README.md"))
    docs = section(readme, "## Docs", "## ")
    assert "docs/hld.md" in docs and "section 7" in docs
    assert "tests/test_wp12_docs.py" in readme
