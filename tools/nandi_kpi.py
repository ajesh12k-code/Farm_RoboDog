#!/usr/bin/env python3
"""KPIs of one Atharv Nandi run directory:  python3 tools/nandi_kpi.py runs/<run_id>

Reads meta.json, harvest_attempts.jsonl, events.jsonl and kis_v1.jsonl, prints a table and writes
kpi.json (then refreshes manifest.json so the Merkle root covers it). Every number is a row carrying
device_id, run_id and date. All values are SIMULATED (data_provenance "simulated").

    picks, attempts, pick_success_rate   from basket truth (attempt "success"), not the node's verdict
    picks_per_hour                       picks / run hours
    drops                                attempts whose crop was lifted (rose >= 3 cm) but did not end in the basket
    false_grasp_reports                  grasp reported ok although the crop never rose 3 cm
    verdict_disagreements                node's own result != basket truth
    falls, estops, interventions(+/hour) events.jsonl
    distance_walked_m                    sum of horizontal tele/pose steps; teleports skipped (world_reset events, or > 2 m/s)
    energy_per_pick_wh                   tele/batt energy used / picks (only when a battery stream exists)
    cycle_<phase>_p50_s / _p90_s         per-phase and total attempt time
WP9 orchard tour: ``--card`` checks and prints the run card the tour wrote (run_card.json -> run_card.md, OS 1.2
step 12), every KPI row with device_id, run_id and date (run-identity rule).
Run hours: meta.duration_s if set (headless runs, sim time), else last kis/v1 ts - meta.start_utc_ms.
"""
import argparse
import datetime as _dt
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
from atharv_nandi.core import manifest as mf  # noqa: E402
from atharv_nandi.core import run_log as rl  # noqa: E402

MAX_WALK_SPEED_MPS = 2.0  # faster than this between two pose messages is a reset/teleport, not walking
# world_reset is detected from /nandi/game_status (about 2 Hz) after the teleport; the teleport step lies in
# [reset - RESET_LAG_MS, reset + RESET_LEAD_MS] (lead covers clock skew between the recorder and the bridge)
RESET_LAG_MS = 2000.0
RESET_LEAD_MS = 250.0


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (numpy's default method); None for no values."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def kis_payloads(lines: List[Dict[str, Any]], stream: str) -> List[Dict[str, Any]]:
    suffix = f"/tele/{stream}"
    return [ln["payload"] for ln in lines if str(ln.get("topic", "")).endswith(suffix)
            and isinstance(ln.get("payload"), dict)]


def distance_walked(poses: List[Dict[str, Any]], reset_ms: Sequence[float] = ()) -> float:
    """Sum of horizontal steps between consecutive tele/pose messages, minus teleports.

    A step is not walking when (a) it is faster than MAX_WALK_SPEED_MPS, or (b) it is the largest
    step overlapping the RESET_LAG_MS window before a world_reset event (events.jsonl utc_ms, same
    wall clock as pose ts). (b) catches the normal live case: a reset after a minute of walking
    moves the trunk well under 1 m, which (a) alone would count as walked distance."""
    pts = []
    for p in sorted(poses, key=lambda p: (p.get("ts", 0), p.get("seq", 0))):
        enu = p.get("enu") or {}
        pts.append((float(enu.get("e", 0.0)), float(enu.get("n", 0.0)), float(p.get("ts", 0))))
    steps = []  # [length, t0, t1, walked]
    for a, b in zip(pts, pts[1:]):
        step = math.dist(a[:2], b[:2])
        dt = max(0.05, (b[2] - a[2]) / 1000.0)
        steps.append([step, a[2], b[2], step / dt <= MAX_WALK_SPEED_MPS])
    for r in reset_ms:
        lo, hi = float(r) - RESET_LAG_MS, float(r) + RESET_LEAD_MS
        near = [s for s in steps if s[2] >= lo and s[1] <= hi]
        if near and all(s[3] for s in near):  # not already dropped by the speed filter
            max(near, key=lambda s: s[0])[3] = False
    return sum(s[0] for s in steps if s[3])


def battery_energy_wh(batts: List[Dict[str, Any]]) -> Optional[float]:
    """Energy used: drop in energy_wh if reported, else the integral of power_w over ts."""
    batts = sorted(batts, key=lambda b: b.get("ts", 0))
    e = [b["energy_wh"] for b in batts if isinstance(b.get("energy_wh"), (int, float))]
    if len(e) >= 2:
        return max(0.0, float(e[0]) - float(e[-1]))
    pw = [(b["ts"], b["power_w"]) for b in batts if isinstance(b.get("power_w"), (int, float))]
    if len(pw) >= 2:
        wh = 0.0
        for (t0, p0), (t1, p1) in zip(pw, pw[1:]):
            wh += abs(0.5 * (p0 + p1)) * max(0.0, (t1 - t0) / 1000.0) / 3600.0
        return wh
    return None


def run_hours(meta: Dict[str, Any], kis: List[Dict[str, Any]]) -> Optional[float]:
    if isinstance(meta.get("duration_s"), (int, float)):
        return float(meta["duration_s"]) / 3600.0
    start = meta.get("start_utc_ms")
    ts = [ln["payload"].get("ts") for ln in kis if isinstance(ln.get("payload"), dict)
          and isinstance(ln["payload"].get("ts"), (int, float))]
    if start and ts:
        return max(0.0, (max(ts) - start) / 3_600_000.0)
    return None


def compute(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    attempts = rl.read_jsonl(os.path.join(run_dir, "harvest_attempts.jsonl"))
    events = rl.read_jsonl(os.path.join(run_dir, "events.jsonl"))
    kis = rl.read_jsonl(os.path.join(run_dir, "kis_v1.jsonl"))

    device_id = meta.get("device_id", rl.DEVICE_ID)
    run_id = meta.get("run_id", os.path.basename(os.path.normpath(run_dir)))
    date = str(meta.get("start_utc", ""))[:10] or _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    rows: List[Dict[str, Any]] = []

    def row(metric: str, value, units: str, n: Optional[int] = None, note: str = ""):
        if isinstance(value, float):
            value = round(value, 4)
        r = {"device_id": device_id, "run_id": run_id, "date": date, "metric": metric, "value": value,
             "units": units, "data_provenance": "simulated"}
        if n is not None:
            r["n"] = n
        if note:
            r["note"] = note
        rows.append(r)

    hours = run_hours(meta, kis)
    picks = sum(1 for a in attempts if a.get("success"))
    n_att = len(attempts)
    drops = sum(1 for a in attempts if a.get("outcome") == "lost_after_grasp")
    false_grasp = sum(1 for a in attempts if a.get("outcome") == "false_grasp_report")
    disagree = sum(1 for a in attempts if a.get("agrees") is False)
    count = {k: sum(1 for e in events if e.get("kind") == k) for k in ("fall", "estop", "intervention")}
    batt = kis_payloads(kis, "batt")
    energy = battery_energy_wh(batt)

    row("run_hours", hours, "h", note="sim time" if "duration_s" in meta else "wall time")
    row("attempts", n_att, "count")
    row("picks", picks, "count", note="basket truth")
    row("pick_success_rate", picks / n_att if n_att else None, "ratio", n_att)
    row("picks_per_hour", picks / hours if hours else None, "picks/h")
    row("drops", drops, "count", note="crop lifted, then lost before the basket")
    row("false_grasp_reports", false_grasp, "count", note="grasp reported ok, crop never lifted")
    row("verdict_disagreements", disagree, "count", note="harvest node result vs basket truth")
    row("falls", count["fall"], "count")
    row("estops", count["estop"], "count")
    row("interventions", count["intervention"], "count", note="manual cmd while a mission ran")
    row("interventions_per_hour", count["intervention"] / hours if hours else None, "1/h")
    resets = [e["utc_ms"] for e in events if e.get("kind") == "world_reset" and isinstance(e.get("utc_ms"), (int, float))]
    poses = kis_payloads(kis, "pose")
    row("distance_walked_m", distance_walked(poses, resets), "m", len(poses),
        note=f"{len(resets)} world_reset step(s) excluded" if resets else "")
    row("energy_per_pick_wh", (energy / picks) if (energy is not None and picks) else None, "Wh/pick",
        note="" if batt else "no tele/batt stream in this run")

    per_phase: Dict[str, List[float]] = {}
    for a in attempts:
        for ph, d in (a.get("phase_durations_s") or {}).items():
            per_phase.setdefault(ph, []).append(float(d))
        if isinstance(a.get("duration_s"), (int, float)):
            per_phase.setdefault("total", []).append(float(a["duration_s"]))
    for ph in sorted(per_phase):
        vals = per_phase[ph]
        row(f"cycle_{ph}_p50_s", percentile(vals, 50), "s", len(vals))
        row(f"cycle_{ph}_p90_s", percentile(vals, 90), "s", len(vals))
    return {"device_id": device_id, "run_id": run_id, "date": date, "sw": meta.get("sw"),
            "data_provenance": "simulated", "rows": rows}


def format_table(kpi: Dict[str, Any]) -> str:
    out = [f"Atharv Nandi KPIs (SIMULATED)  device={kpi['device_id']}  run={kpi['run_id']}  date={kpi['date']}",
           f"{'metric':<30} {'value':>12}  {'units':<9} {'n':>4}  note", "-" * 78]
    for r in kpi["rows"]:
        v = r["value"]
        vs = "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
        out.append(f"{r['metric']:<30} {vs:>12}  {r['units']:<9} {str(r.get('n', '')):>4}  {r.get('note', '')}")
    return "\n".join(out)


def write(run_dir: str, kpi: Dict[str, Any]):
    rl.write_json(os.path.join(run_dir, "kpi.json"), kpi)
    meta = json.load(open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8"))
    final, closed = False, None
    mpath = os.path.join(run_dir, "manifest.json")
    if os.path.isfile(mpath):
        old = json.load(open(mpath, "r", encoding="utf-8"))
        final, closed = bool(old.get("final")), old.get("closed")
    mf.write_manifest(run_dir, meta, final=final, closed_ms=closed)


def run_card(run_dir: str, write_files: bool = True) -> Dict[str, Any]:
    """WP9 orchard tour run card (OS 1.2 step 12): the ``run_card.json`` the tour wrote (``controllers.demo_tour``),
    checked for the run-identity fields, printed as Markdown; the manifest is refreshed so its Merkle root
    covers the card. Raises ValueError when the card or an identity field is missing."""
    from atharv_nandi.core import scope_kpi as sk
    path = os.path.join(run_dir, "run_card.json")
    if not os.path.isfile(path):
        raise ValueError(f"{run_dir}: no run_card.json (not a WP9 tour run)")
    with open(path, "r", encoding="utf-8") as fh:
        card = json.load(fh)
    missing = [k for k in ("run_id", "device_id", "date", "data_provenance", "detector", "assisted",
                           "scripted_delays", "kpis", "counts") if k not in card]
    if missing or card.get("data_provenance") != "simulated":
        raise ValueError(f"run card incomplete: {missing or 'data_provenance'}")
    for r in card["kpis"]:
        if not (r.get("run_id") and r.get("device_id") and r.get("date")):
            raise ValueError(f"KPI row {r.get('metric')} without run_id / device_id / date (run-identity rule)")
    if write_files:
        with open(os.path.join(run_dir, "run_card.md"), "w", encoding="utf-8") as fh:
            fh.write(sk.format_card(card))
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.isfile(meta_path):
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            final, closed = False, None
            mpath = os.path.join(run_dir, "manifest.json")
            if os.path.isfile(mpath):
                old = json.load(open(mpath, "r", encoding="utf-8"))
                final, closed = bool(old.get("final")), old.get("closed")
            mf.write_manifest(run_dir, meta, final=final, closed_ms=closed)
    return card


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="runs/<run_id>")
    ap.add_argument("--no-write", action="store_true", help="print only")
    ap.add_argument("--card", action="store_true", help="WP9 orchard tour run card (run_card.json -> run_card.md)")
    args = ap.parse_args(argv)
    if args.card:
        from atharv_nandi.core import scope_kpi as sk
        card = run_card(args.run_dir, write_files=not args.no_write)
        print(sk.format_card(card))
        return 0 if card["tour"].get("pass") else 1
    kpi = compute(args.run_dir)
    print(format_table(kpi))
    if not args.no_write:
        write(args.run_dir, kpi)
        print(f"wrote {os.path.join(args.run_dir, 'kpi.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
