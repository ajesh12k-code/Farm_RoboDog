"""WPCOV coverage gate tool (tools/coverage_gate.py): line >= 85 % and branch >= 75 % on atharv_nandi/core/, read
from a coverage JSON report (coverage.py has one fail_under for its combined total)."""
import json

from tools import coverage_gate as cg


def _report(files):
    return {"files": {p: {"summary": {"num_statements": s, "covered_lines": c, "num_branches": b,
                                      "covered_branches": cb}} for p, (s, c, b, cb) in files.items()}}


REP = _report({"atharv_nandi/core/a.py": (100, 90, 40, 32), "/ws/atharv_nandi/core/sub/b.py": (100, 80, 60, 40),
               "atharv_nandi/sim/hooks/c.py": (100, 0, 10, 0), "atharv_nandi/core/empty.py": (0, 0, 0, 0)})


def test_summarize_only_counts_the_prefix():
    rows, tot = cg.summarize(REP)
    assert {r["module"] for r in rows} == {"a.py", "sub/b.py", "empty.py"}
    assert tot["statements"] == 200 and tot["line_pct"] == 85.0 and tot["branch_pct"] == 72.0
    assert next(r for r in rows if r["module"] == "empty.py")["line_pct"] == 100.0


def test_gate_line_and_branch_separately():
    _, tot = cg.summarize(REP)
    assert cg.gate(tot) == ["branch 72.00 % < 75.0 %"]
    assert cg.gate(tot, line=86, branch=70) == ["line 85.00 % < 86 %"]
    assert cg.gate(tot, branch=72) == []
    assert cg.gate(cg.summarize({"files": {}})[1]) == ["no file under the prefix in the report"]


def test_main_prints_table_and_exit_code(tmp_path, capsys):
    p = tmp_path / "cov.json"
    p.write_text(json.dumps(REP))
    assert cg.main([str(p)]) == 1
    out = capsys.readouterr().out
    assert "sub/b.py" in out and "FAIL: branch" in out and out.index("sub/b.py") < out.index("a.py")
    assert cg.main([str(p), "--branch", "70", "--md"]) == 0
    out = capsys.readouterr().out
    assert "| `sub/b.py` |" in out and "PASS: line 85.00 %" in out


def test_new_modules_gate_at_90_line(tmp_path, capsys):
    rows, _ = cg.summarize(REP)
    assert cg.gate_new(rows, ["atharv_nandi/core/a.py"]) == []                       # 90 % passes
    assert cg.gate_new(rows, ["/ws/atharv_nandi/core/sub/b.py"]) == ["new module sub/b.py line 80.00 % < 90.0 %"]
    assert cg.gate_new(rows, ["atharv_nandi/core/gone.py"]) == ["new module gone.py not in the report"]
    assert cg.gate_new(rows, ["atharv_nandi/sim/hooks/c.py"]) == []                  # outside the prefix
    p = tmp_path / "cov.json"
    p.write_text(json.dumps(REP))
    assert cg.main([str(p), "--branch", "70", "--new", "atharv_nandi/core/a.py"]) == 0
    assert cg.main([str(p), "--branch", "70", "--new", "atharv_nandi/core/sub/b.py"]) == 1
    assert "FAIL: new module sub/b.py" in capsys.readouterr().out
