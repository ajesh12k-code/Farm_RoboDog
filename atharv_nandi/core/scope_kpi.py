"""WP9 per-scope KPIs and the run card (pure Python, no ROS, no MuJoCo).

Spec: ORCHARD-MVP-SPEC 1.2 step 12 (run card: ``run_id``, ``device_id``, date, ``data_provenance: simulated``,
detector, ``assisted[]``, scripted delays, per-preset KPIs, TreeRecord and MARK counts), 1.3 (O-01..O-23, gates in
the assisted / SimTruth column; physics or ColorBlob rows are *reported*, never gated), MVP-SPEC 5.7
(``/nandi/kpi``, ``/nandi/kpi/scopes``), run-identity rule (every KPI row carries device_id, run_id and date).

A KPI row: ``{kpi, scope, metric, value, units, gate, pass, n, mode, detector, note, run_id, device_id, date,
data_provenance}``. ``pass`` is True / False against the gate, None when the row is report-only or has no value.
Per-pick success rates (O-10..O-12) are aggregate KPIs over the readiness runs: on one tour they are reported with
their gate but marked ``aggregate: true`` and never decide the tour pass (OS 1.2 tour pass definition).
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

CARD_SCHEMA = "nandi.run_card/1"

# metric -> (kpi id, comparator, gate, units, aggregate). Comparators: ge / le / eq / zero / true.
GATES: Dict[str, tuple] = {
    # SCAN (MP 1)
    "scan.trees_seen_frac": ("MP1", "ge", 0.95, "ratio", False),
    "scan.bands_ge2_frac": ("MP1", "ge", 1.0, "ratio", False),
    "scan.laterals_stepped": ("P-08", "zero", 0, "count", False),
    "scan.triage_precision": ("MP1", "ge", 0.9, "ratio", False),
    "scan.triage_recall": ("MP1", "ge", 0.9, "ratio", False),
    # TREE 360 (O-01..O-06, O-21, O-22)
    "tree360.coverage_low_pct": ("O-01", "ge", 90.0, "%", False),
    "tree360.coverage_mid_pct": ("O-01", "ge", 90.0, "%", False),
    "tree360.coverage_high_pct": ("O-01", "ge", 90.0, "%", False),
    "tree360.two_sided_min_deg": ("O-01", "ge", 120.0, "deg", False),
    "tree360.radius_rms_m": ("O-02", "le", 0.10, "m", False),
    "tree360.az_ok": ("O-03", "true", True, "bool", False),
    "tree360.bearing_ok": ("O-03", "true", True, "bool", False),
    "tree360.steady_pct": ("O-05", "ge", 95.0, "%", False),
    "tree360.orbit_s": ("O-06", "le", 165.0, "s", False),
    "tree360.person_unmasked_px": ("O-21", "zero", 0, "px", False),
    "tree360.mb_per_tree": ("O-22", "le", 30.0, "MB", False),
    "tree360.radius_reentries": ("OS1.2", "le", 1, "count", False),
    # LEAF SCAN (O-08)
    "leaf_scan.imaged_frac": ("O-08", "ge", 0.90, "ratio", False),
    "leaf_scan.underside_frac": ("O-08", "ge", 0.40, "ratio", False),
    "leaf_scan.footprint_mm_px_max": ("O-08", "le", 1.0, "mm/px", False),
    "leaf_scan.median_s_per_leaf": ("O-08", "le", 10.0, "s", False),
    "leaf_scan.min_clearance_m": ("O-08", "ge", 0.02, "m", False),
    # TRUNK SCAN (MP 3b)
    "trunk_scan.azimuths_imaged": ("MP3b", "ge", 2, "count", False),
    # FRUIT SCOUT (O-09, O-15, O-16)
    "fruit_scout.count_error": ("O-09", "le", 0.10, "ratio", False),
    "fruit_scout.stage_acc": ("O-09", "ge", 0.90, "ratio", False),
    "fruit_scout.window_equal": ("O-09", "true", True, "bool", False),
    "fruit_scout.reach_ratio": ("O-16", None, None, "ratio", False),
    # HARVEST / PRUNE (O-10..O-15)
    "harvest.mango_success": ("O-10", "ge", 0.80, "ratio", True),
    "harvest.guava_success": ("O-11", "ge", 0.80, "ratio", True),
    "harvest.papaya_success": ("O-12", "ge", 0.80, "ratio", True),
    "harvest.pedicel_err_mm_max": ("O-10", "le", 5.0, "mm", False),
    "harvest.skipped_targets_max": ("OS1.2", "le", 1, "count", False),
    "harvest.papaya_place_mps_max": ("O-12", "le", 0.5, "m/s", False),
    "prune.cut_frac": ("O-13", "ge", 0.80, "ratio", False),
    "prune.keep_cuts": ("O-13", "zero", 0, "count", False),
    "prune.green_leaf_cuts": ("O-14", "zero", 0, "count", False),
    "prune.skipped_targets_max": ("OS1.2", "le", 1, "count", False),
    "marks.completeness": ("O-15", "ge", 1.0, "ratio", False),
    # DOCK / UNLOAD (O-20)
    "dock.error_cm": ("O-20", "le", 3.0, "cm", False),
    "dock.yaw_deg": ("O-20", "le", 5.0, "deg", False),
    "dock.items_in_crate_frac": ("O-20", "ge", 1.0, "ratio", False),
    "dock.impact_mps_max": ("O-20", "le", 2.0, "m/s", False),
    "dock.attribution_frac": ("O-20", "ge", 1.0, "ratio", False),
    # safety (O-17, O-18, P-05..P-08, O-21)
    "safety.falls": ("O-17", "zero", 0, "count", False),
    "safety.protective_violations": ("P-05", "zero", 0, "count", False),
    "safety.unmasked_person_px": ("O-21", "zero", 0, "px", False),
    "safety.motion_without_confirm": ("O-19", "zero", 0, "count", False),
    "safety.min_tripod_margin_crown_up_m": ("O-17", "ge", 0.03, "m", False),
    "safety.gust_stance_s": ("O-18", "le", 0.5, "s", False),
    "safety.crown_stow_s": ("O-18", "le", 8.5, "s", False),
    "safety.estop_stationary_s": ("MVP1.2", "le", 0.5, "s", False),
    # data (O-22, O-23)
    "records.tree_record_max_bytes": ("O-22", "le", 16384, "B", False),
    "records.contract_violations": ("O-22", "zero", 0, "count", False),
    "records.jobsheet_missing_marks": ("O-23", "zero", 0, "count", False),
    "records.jobsheet_rows_frac": ("O-23", "ge", 1.0, "ratio", False),
    # tour
    "tour.sim_s": ("OS1.1", "le", 1080.0, "s", False),
    "tour.steps_halted": ("OS1.2", "zero", 0, "count", False),
}


def _cmp(op: Optional[str], value: Any, gate: Any) -> Optional[bool]:
    if op is None or value is None:
        return None
    try:
        if op == "ge":
            return float(value) >= float(gate) - 1e-9
        if op == "le":
            return float(value) <= float(gate) + 1e-9
        if op == "zero":
            return float(value) == 0.0
        if op == "true":
            return bool(value) is True
        if op == "eq":
            return value == gate
    except (TypeError, ValueError):
        return False
    return None


class KpiTable:
    """Collects KPI rows for one run. ``mode`` assisted | physics; ``detector`` simtruth | colorblob."""

    def __init__(self, run_id: str, device_id: str, date: Optional[str] = None, mode: str = "assisted",
                 detector: str = "simtruth"):
        self.run_id, self.device_id = run_id, device_id
        self.date = date or datetime.now(timezone.utc).date().isoformat()
        self.mode, self.detector = mode, detector
        self.rows: List[dict] = []

    def add(self, metric: str, value: Any, n: Optional[int] = None, note: str = "", scope: Optional[str] = None,
            gate_override: Any = None, units: Optional[str] = None, mode: Optional[str] = None) -> dict:
        kpi, op, gate, u, agg = GATES.get(metric, (None, None, None, units or "", False))
        if gate_override is not None:
            gate = gate_override
        if isinstance(value, float):
            value = round(value, 4)
        # gates hold only in the assisted / SimTruth column (OS 1.3); other modes are reported
        mode = mode or self.mode
        gated = mode == "assisted" and self.detector == "simtruth"
        row = {"kpi": kpi, "scope": scope or metric.split(".")[0], "metric": metric, "value": value,
               "units": units or u, "gate": gate, "op": op, "pass": _cmp(op, value, gate) if gated else None,
               "aggregate": bool(agg), "n": n, "mode": mode, "detector": self.detector, "note": note,
               "run_id": self.run_id, "device_id": self.device_id, "date": self.date,
               "data_provenance": "simulated"}
        self.rows.append(row)
        return row

    def failed(self, include_aggregate: bool = False) -> List[dict]:
        return [r for r in self.rows if r["pass"] is False and (include_aggregate or not r["aggregate"])]

    def by_scope(self) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {}
        for r in self.rows:
            out.setdefault(r["scope"], []).append(r)
        return out


# ---------------------------------------------------------------------- scope extractors
def scan_kpis(t: KpiTable, out: dict):
    trees = out.get("trees") or []
    n = len(trees)
    seen = sum(1 for x in trees if x.get("seen"))
    t.add("scan.trees_seen_frac", seen / n if n else None, n)
    t.add("scan.bands_ge2_frac", (sum(1 for x in trees if len(x.get("bands") or ()) >= 2) / n) if n else None, n)
    t.add("scan.laterals_stepped", out.get("laterals_stepped"), note=out.get("laterals_note", ""))
    tp, fp, fn = (int(out.get("triage", {}).get(k, 0)) for k in ("tp", "fp", "fn"))
    t.add("scan.triage_precision", tp / (tp + fp) if tp + fp else (1.0 if not fn else None), tp + fp)
    t.add("scan.triage_recall", tp / (tp + fn) if tp + fn else 1.0, tp + fn)


def tree360_kpis(t: KpiTable, res: dict):
    k = res.get("kpi") or {}
    bands = ((res.get("coverage") or {}).get("bands") or {})
    two = res.get("mode") == "two_sided"
    for b in ("low", "mid", "high"):
        if b in bands and b in (res.get("bands_planned") or bands):
            if two:
                t.add(f"tree360.coverage_{b}_pct", bands[b].get("pct"), note="two-sided: reported, gate is per side",
                      gate_override=None, mode="report")
            else:
                t.add(f"tree360.coverage_{b}_pct", bands[b].get("pct"))
    planned = res.get("bands_planned") or [b for b in bands if (bands[b].get("frames") or 0) > 0]
    if two:
        mins = [min((bands[b].get("sides") or {"-": 0}).values()) for b in bands if b in planned]
        t.add("tree360.two_sided_min_deg", min(mins) if mins else None, len(mins))
    t.add("tree360.radius_rms_m", k.get("radius_rms_truth_m"), note="truth radius (assisted pose)")
    t.add("tree360.az_ok", k.get("az_ok"))
    t.add("tree360.bearing_ok", k.get("bearing_ok"))
    t.add("tree360.steady_pct", k.get("steady_pct"))
    t.add("tree360.orbit_s", k.get("orbit_s"), note="full orbit gate; two-sided / snake_mid reported" if two or
          res.get("crown_walk") != "snake" else "", mode=None if not two and res.get("crown_walk") == "snake"
          else "report")
    if k.get("person_unmasked_px") is not None:
        t.add("tree360.person_unmasked_px", k.get("person_unmasked_px"))
    if k.get("bytes") is not None:
        t.add("tree360.mb_per_tree", float(k["bytes"]) / (1024 * 1024))
    t.add("tree360.radius_reentries", k.get("reentries", 0))


def leaf_kpis(t: KpiTable, leaf: dict):
    if (leaf or {}).get("status") != "ok":
        t.add("leaf_scan.imaged_frac", None, note=f"status {(leaf or {}).get('status')}")
        return
    for m in ("imaged_frac", "underside_frac", "footprint_mm_px_max", "median_s_per_leaf", "min_clearance_m"):
        t.add(f"leaf_scan.{m}", leaf.get(m), leaf.get("n"))


def trunk_kpis(t: KpiTable, res: dict):
    t.add("trunk_scan.azimuths_imaged", res.get("azimuths_imaged"), note=res.get("mode", ""))


def scout_kpis(t: KpiTable, fruit: dict, assisted: dict):
    a = assisted or {}
    t.add("fruit_scout.count_error", a.get("count_error"), a.get("truth_visible_n"))
    t.add("fruit_scout.stage_acc", a.get("stage_acc"), a.get("stage_n"))
    t.add("fruit_scout.window_equal", a.get("window_equal"))
    vis, reach = (fruit or {}).get("visible_n"), (fruit or {}).get("reachable_n")
    t.add("fruit_scout.reach_ratio", (reach / vis) if vis else None, vis, note="report only (O-16)")


def harvest_kpis(t: KpiTable, steps: Sequence[dict]):
    """Aggregate over every HARVEST step of the run (O-10..O-12 aggregate)."""
    by_crop: Dict[str, List[dict]] = {}
    ped, skipped, place = [], [], []
    for s in steps:
        res = s.get("results") or []
        for r in res:
            by_crop.setdefault((r.get("crop") or s.get("crop") or "?"), []).append(r)
            if isinstance(r.get("pedicel_error_mm"), (int, float)):
                ped.append(abs(float(r["pedicel_error_mm"])))
        skipped.append(int(s.get("skipped_targets", 0)))
        if s.get("place_mps_max") is not None:
            place.append(float(s["place_mps_max"]))
    for crop, key in (("mango", "harvest.mango_success"), ("guava", "harvest.guava_success"),
                      ("papaya", "harvest.papaya_success")):
        rs = [r for r in by_crop.get(crop, []) if r.get("outcome") not in ("marked",)]
        if rs:
            ok = sum(1 for r in rs if r.get("outcome") in ("picked", "delivered"))
            t.add(key, ok / len(rs), len(rs), note="aggregate KPI over the readiness runs; one tour reported")
    t.add("harvest.pedicel_err_mm_max", max(ped) if ped else None, len(ped))
    t.add("harvest.skipped_targets_max", max(skipped) if skipped else 0, len(skipped))
    if place:
        t.add("harvest.papaya_place_mps_max", max(place), len(place))


def prune_kpis(t: KpiTable, steps: Sequence[dict]):
    cands = cuts = 0
    keep: Optional[int] = 0
    green: Optional[int] = 0
    skipped = []
    for s in steps:
        res = s.get("results") or []
        cands += sum(1 for r in res if r.get("outcome") != "marked")
        cuts += sum(1 for r in res if r.get("outcome") in ("cut", "delivered"))
        # None = not audited on that step: the whole row is report-only (no value), never a passed gate
        keep = None if keep is None or s.get("keep_cuts", 0) is None else keep + int(s.get("keep_cuts", 0))
        green = None if green is None or s.get("green_cuts", 0) is None else green + int(s.get("green_cuts", 0))
        skipped.append(int(s.get("skipped_targets", 0)))
    t.add("prune.cut_frac", cuts / cands if cands else None, cands)
    t.add("prune.keep_cuts", keep, note="" if keep is not None else "not audited (no cut-event audit)")
    t.add("prune.green_leaf_cuts", green, note="" if green is not None else "not audited (no cut-event audit)")
    t.add("prune.skipped_targets_max", max(skipped) if skipped else 0, len(skipped))


def dock_kpis(t: KpiTable, out: dict):
    d = (out.get("dock") or {}).get("final") or {}
    if d:
        t.add("dock.error_cm", None if d.get("dist_m") is None else 100.0 * float(d["dist_m"]))
        t.add("dock.yaw_deg", None if d.get("yaw_deg") is None else abs(float(d["yaw_deg"])))
    cr = out.get("crate") or {}
    n = cr.get("n")
    exp = out.get("expected_items")
    if exp:
        t.add("dock.items_in_crate_frac", (n or 0) / exp, exp)
    if cr.get("impact_mps_max") is not None:
        t.add("dock.impact_mps_max", cr.get("impact_mps_max"))
    if cr.get("per_tree") is not None and n:
        att = sum(int(v.get("crate_n", v.get("n", 0))) for v in (cr.get("per_tree") or {}).values())
        t.add("dock.attribution_frac", att / n, n)


def safety_kpis(t: KpiTable, safety: dict):
    for m in ("falls", "protective_violations", "motion_without_confirm"):
        t.add(f"safety.{m}", int(safety.get(m, 0)))
    # O-21: only a GL capture counts person pixels; with geometry capture the row is report-only, never "ok"
    if safety.get("person_pixels_measured") is False:
        t.add("safety.unmasked_person_px", None, note="not measured (geometry capture; a GL run counts it)",
              mode="report")
    else:
        t.add("safety.unmasked_person_px", int(safety.get("unmasked_person_px", 0)))
    for m in ("min_tripod_margin_crown_up_m", "gust_stance_s", "crown_stow_s", "estop_stationary_s"):
        if safety.get(m) is not None:
            t.add(f"safety.{m}", safety[m])


def records_kpis(t: KpiTable, tree_records: Sequence[dict], violations: int, missing: Sequence[str],
                 rows_visited: Sequence[str], sheets: Dict[str, dict], marks_truth_n: Optional[int] = None,
                 marks_n: Optional[int] = None):
    from atharv_nandi.core import records as rec
    sizes = [rec.json_size(r) for r in tree_records]
    t.add("records.tree_record_max_bytes", max(sizes) if sizes else None, len(sizes))
    t.add("records.contract_violations", int(violations))
    t.add("records.jobsheet_missing_marks", len(missing), note=",".join(missing[:3]))
    if rows_visited:
        t.add("records.jobsheet_rows_frac", sum(1 for r in rows_visited if r in sheets) / len(rows_visited),
              len(rows_visited))
    if marks_truth_n:
        t.add("marks.completeness", (marks_n or 0) / marks_truth_n, marks_truth_n,
              note="MARK records / truth visible out-of-band jobs of the trees worked")


# ---------------------------------------------------------------------- run card
def run_card(*, ident: Dict[str, str], date: str, world: str, season: str, seed: int, program: str,
             detector: str, mode: str, assisted: Iterable[str], scripted_delays: Sequence[dict], table: KpiTable,
             counts: Dict[str, int], tour: dict, notes: Sequence[str] = (), sw: str = "") -> dict:
    """The ORCHARD-MVP-SPEC 1.2 step 12 run card."""
    scopes = {}
    for scope, rows in table.by_scope().items():
        gated = [r for r in rows if r["pass"] is not None]
        scopes[scope] = {"rows": len(rows), "gated": len(gated), "failed": [r["metric"] for r in gated
                                                                           if r["pass"] is False]}
    return {"schema": CARD_SCHEMA, "run_id": ident["run_id"], "mission_id": ident.get("mission_id"),
            "device_id": ident["device_id"], "tenant": ident.get("tenant"), "date": date,
            "data_provenance": "simulated", "world": world, "season": season,
            "season_banner": ("demo_all_fruiting: all four species fruit at once; not agronomically realistic"
                              if season == "demo_all_fruiting" else ""),
            "seed": seed, "program": program, "detector": detector, "mode": mode, "sw": sw,
            "assisted": sorted(set(assisted)), "scripted_delays": list(scripted_delays),
            "counts": dict(counts), "tour": dict(tour), "scopes": scopes, "kpis": list(table.rows),
            "failed_gates": [r["metric"] for r in table.failed()],
            "failed_aggregate": [r["metric"] for r in table.rows if r["pass"] is False and r["aggregate"]],
            "notes": list(notes)}


def format_card(card: dict) -> str:
    """Markdown run card (what the operator reads after step 12)."""
    t = card["tour"]
    out = [f"# Atharv Nandi run card (SIMULATED)",
           "",
           f"run `{card['run_id']}` · device `{card['device_id']}` · date {card['date']} · "
           f"data_provenance **{card['data_provenance']}** · world `{card['world']}` seed {card['seed']} · "
           f"program `{card['program']}` · detector `{card['detector']}` · mode `{card['mode']}`",
           ""]
    if card.get("season_banner"):
        out += [f"> {card['season_banner']}", ""]
    out += [f"**Tour pass: {'PASS' if t.get('pass') else 'FAIL'}** · sim {t.get('sim_s')} s "
            f"(gate {t.get('gate_s')} s, time {'ok' if t.get('time_ok') else 'over'}) · wall {t.get('wall_s')} s · "
            f"steps {t.get('steps_done')}/{t.get('steps_n')} done, {t.get('steps_halted')} halted",
            ""]
    if t.get("fail_reasons"):
        out += ["Fail reasons: " + "; ".join(t["fail_reasons"]), ""]
    out += ["Assisted: " + (", ".join(card["assisted"]) or "none"), ""]
    if card["scripted_delays"]:
        out += ["Scripted delays: " + "; ".join(f"{d.get('what')} {d.get('s')} s" for d in card["scripted_delays"]),
                ""]
    out += ["Counts: " + ", ".join(f"{k} {v}" for k, v in card["counts"].items()), "",
            "| KPI | metric | value | gate | pass | n | note |", "|---|---|---|---|---|---|---|"]
    for r in card["kpis"]:
        p = "-" if r["pass"] is None else ("ok" if r["pass"] else "FAIL")
        g = "" if r["gate"] is None else f"{r['op']} {r['gate']}"
        agg = " (aggregate)" if r.get("aggregate") else ""
        out.append(f"| {r['kpi'] or ''} | {r['metric']} | {r['value']} | {g} | {p}{agg} | {r['n'] or ''} | "
                   f"{r['note']} |")
    if card.get("notes"):
        out += ["", "Notes:"] + [f"- {n}" for n in card["notes"]]
    return "\n".join(out) + "\n"


def median(xs: Sequence[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    return statistics.median(xs) if xs else None
