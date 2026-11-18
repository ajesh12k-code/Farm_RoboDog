#!/usr/bin/env python3
"""Coverage gate on ``atharv_nandi/core/`` (WPCOV; MVP-SPEC 6).

coverage.py has one ``fail_under`` for its combined (line + branch) total, so the two gates the spec names are
read here from a coverage JSON report: line >= 85 % and branch >= 75 % over every file under the package prefix, and
line >= 90 % for each new module named with ``--new`` (MVP-SPEC 6 "New modules start at >= 90 %"; the caller names
them, e.g. ``git diff --name-only --diff-filter=A origin/feat/nandi-pilot -- atharv_nandi/core/``).
Prints a per-module table (worst first) and exits 1 when a gate fails.

    pytest -p pytest_cov --cov --cov-config=.coveragerc --cov-report=json:cov.json ...   # or: coverage json
    python3 tools/coverage_gate.py cov.json [--prefix atharv_nandi/core/] [--line 85] [--branch 75] [--md]
                                    [--new atharv_nandi/core/x.py ...] [--new-line 90]

Several partial runs (the capped runner, file by file) are combined first: ``COVERAGE_FILE=.coverage.<n>`` per run,
then ``coverage combine --rcfile=.coveragerc .coverage.*`` and ``coverage json --rcfile=.coveragerc``.
"""
import argparse
import json
import sys
from typing import Dict, List, Tuple

LINE_GATE = 85.0
BRANCH_GATE = 75.0
NEW_LINE_GATE = 90.0
PREFIX = "atharv_nandi/core/"


def _pct(hit: int, n: int) -> float:
    return 100.0 * hit / n if n else 100.0


def summarize(report: dict, prefix: str = PREFIX) -> Tuple[List[dict], Dict[str, float]]:
    """(per-module rows, totals) for the files under ``prefix`` of a coverage JSON report."""
    rows = []
    tot = {"statements": 0, "covered": 0, "branches": 0, "covered_branches": 0}
    for path, f in sorted((report.get("files") or {}).items()):
        p = path.replace("\\", "/")
        i = p.find(prefix)
        if i < 0:
            continue
        s = f["summary"]
        row = {"module": p[i + len(prefix):], "statements": int(s["num_statements"]),
               "covered": int(s["covered_lines"]), "branches": int(s.get("num_branches", 0)),
               "covered_branches": int(s.get("covered_branches", 0))}
        row["line_pct"] = _pct(row["covered"], row["statements"])
        row["branch_pct"] = _pct(row["covered_branches"], row["branches"])
        rows.append(row)
        for k in tot:
            tot[k] += row[k]
    totals = {"line_pct": _pct(tot["covered"], tot["statements"]),
              "branch_pct": _pct(tot["covered_branches"], tot["branches"]), "files": len(rows), **tot}
    return rows, totals


def gate(totals: Dict[str, float], line: float = LINE_GATE, branch: float = BRANCH_GATE) -> List[str]:
    """Failed gate messages ([] = pass). No file under the prefix is a failure, not a pass."""
    if not totals.get("files"):
        return ["no file under the prefix in the report"]
    out = []
    if totals["line_pct"] < line:
        out.append(f"line {totals['line_pct']:.2f} % < {line} %")
    if totals["branch_pct"] < branch:
        out.append(f"branch {totals['branch_pct']:.2f} % < {branch} %")
    return out


def gate_new(rows: List[dict], new: List[str], prefix: str = PREFIX, line: float = NEW_LINE_GATE) -> List[str]:
    """Failed new-module messages ([] = pass): each module in ``new`` (a path, with or without the prefix) must be
    in the report with line coverage >= ``line``. A new module missing from the report fails (never measured).
    Paths outside the prefix are ignored (the gate is on the package only)."""
    by_mod = {r["module"]: r for r in rows}
    out = []
    for n in new:
        p = n.replace("\\", "/")
        i = p.find(prefix)
        if i < 0:
            continue
        mod = p[i + len(prefix):]
        r = by_mod.get(mod)
        if r is None:
            out.append(f"new module {mod} not in the report")
        elif r["line_pct"] < line:
            out.append(f"new module {mod} line {r['line_pct']:.2f} % < {line} %")
    return out


def table(rows: List[dict], totals: Dict[str, float], md: bool = False) -> str:
    rows = sorted(rows, key=lambda r: (r["line_pct"], r["branch_pct"], r["module"]))
    if md:
        out = ["| module | stmts | line % | branches | branch % |", "|---|---:|---:|---:|---:|"]
        out += [f"| `{r['module']}` | {r['statements']} | {r['line_pct']:.1f} | {r['branches']} | "
                f"{r['branch_pct']:.1f} |" for r in rows]
        out.append(f"| **total** | {totals['statements']} | **{totals['line_pct']:.2f}** | {totals['branches']} | "
                   f"**{totals['branch_pct']:.2f}** |")
    else:
        out = [f"{'module':44s} {'stmts':>6s} {'line%':>6s} {'branches':>8s} {'branch%':>7s}"]
        out += [f"{r['module']:44s} {r['statements']:6d} {r['line_pct']:6.1f} {r['branches']:8d} "
                f"{r['branch_pct']:7.1f}" for r in rows]
        out.append(f"{'TOTAL':44s} {totals['statements']:6d} {totals['line_pct']:6.2f} {totals['branches']:8d} "
                   f"{totals['branch_pct']:7.2f}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("report", help="coverage JSON report (coverage json / --cov-report=json)")
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--line", type=float, default=LINE_GATE)
    ap.add_argument("--branch", type=float, default=BRANCH_GATE)
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--new", nargs="*", default=[], help="new modules (gated at --new-line)")
    ap.add_argument("--new-line", type=float, default=NEW_LINE_GATE)
    a = ap.parse_args(argv)
    with open(a.report, encoding="utf-8") as fh:
        report = json.load(fh)
    rows, totals = summarize(report, a.prefix)
    print(table(rows, totals, a.md))
    bad = gate(totals, a.line, a.branch) + gate_new(rows, a.new, a.prefix, a.new_line)
    print(("FAIL: " + "; ".join(bad)) if bad else
          f"PASS: line {totals['line_pct']:.2f} % >= {a.line} %, branch {totals['branch_pct']:.2f} % >= {a.branch} %")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
