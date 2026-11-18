"""WP9 orchard tour: the tour state machine, the pass definition, records, job sheets, uplink and run card.

Spec: ORCHARD-MVP-SPEC 1.2 (demo script, tour pass definition, sim-time budget, ``tour_lite``), 1.3 (KPIs), 5.6
(data products); MISSION-PRESETS 0.4 (failure semantics), 0.6 (records), 9 (DOCK/UNLOAD job sheets), 10 (missions);
WP9.

Layers (the first two are pure Python: no ROS, no MuJoCo):

    TourProgram     the steps: ``config/missions/orchard_tour.yaml`` (WP27, read only) or the built-in ``tour_lite``
                    (orchard_lite, OS 1.2 "tour_lite"); each step is a preset (Ready check + run) or a scripted part.
    TourRunner      the state machine. Per step: Ready check (WP27 ``ready_check.check`` on the backend's context and
                    the step's ctx) -> backend.run_step -> merge outputs into TreeRecords, MARKs, HarvestRecords ->
                    pass bookkeeping. Steps whose ``expect`` is a refusal (OS 1.2 step 10 b) pass when the Ready check
                    or the gates refuse them with that reason (and, when the backend can probe, again with the
                    context measured on the spot: ``refusal_probe``). Pause / resume / abort take effect inside a
                    step too: a backend calls ``runner.check_control()`` (SimBackend: a 10 Hz driver task), which
                    blocks while paused and raises ``StepAborted`` on abort (the step ends ``aborted``). At the end: job sheets per row visited, uplink (kis/v1
                    PROPOSED streams), KPI table, run card, all under ``runs/<run_id>/``.
    SimBackend      runs every step on a WP0b ``SimDriver`` with every hook loaded: SCAN (walk the alley, crown mid,
                    3 s bursts abeam, WP26 canopy leads), TREE 360 (WP25 ``hooks.tree360``), LEAF / TRUNK SCAN and
                    FRUIT SCOUT (WP26 planners, standing for their time model), HARVEST / PRUNE (WP8 ``TreeRunner``),
                    RECOVERY tour part (WP7 gust event, WP6 / WP27 refusals, e-stop hold with scripted confirmations),
                    DOCK/UNLOAD (WP13 ``DockUnloadSession`` with this package's job sheet generator). Transits walk
                    the step's ``motion`` paths (or a WP29 headland route) at ``TRANSIT_VX``.

Tour pass definition (OS 1.2): (1) every safety gate holds: 0 falls, 0 protective-field violations, 0 unmasked person
pixels, 0 motion without confirmation, and the timed safety gates of the steps that ran (O-18 gust stance <= 0.5 s and
crown stowed <= 8.5 s, e-stop stationary <= 0.5 s; a step that should measure one and did not fails the tour); (2) per-step skip limits: <= 1 skipped target per harvest / prune step, <= 1
radius re-entry per orbit, no step halted, refused or skipped (a preset step the backend cannot run is a failure,
never a silent pass). Per-pick success rates are aggregate KPIs, not pass conditions. The
sim-time gate (18 min tour, 5 min tour_lite) is reported as ``time_ok`` next to the pass (WP9 acceptance gates it in
the tour tests).
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from atharv_nandi import paths
from atharv_nandi.core import job_sheet as js  # noqa: E402
from atharv_nandi.core import records as rec  # noqa: E402
from atharv_nandi.core import scope_kpi as sk  # noqa: E402

REPO = paths.REPO
STATE_SCHEMA = "nandi.demo_state/1"
TRANSIT_VX = 0.65                 # WP27 TRANSIT_CMD_VX: 0.179 m/s stowed (measured, WP27 runner)
CROWN_UP_VX = 0.5                 # 0.140 m/s (measured, WP27): <= 0.15 m/s with the crown above 0
SKIP_LIMIT = 1                    # OS 1.2: <= 1 skipped target per harvest / prune step
REENTRY_LIMIT = 1                 # OS 1.2: <= 1 radius re-entry per orbit
FALL_TILT_DEG = 35.0              # OS 5.3 fall latch
PROTECTIVE_STOP_S = 0.5           # stationary within 0.5 s of a protective latch (MVP-SPEC 1.2)
OPERATOR_ACK_S = 2.0               # scripted operator ack of a cleared protective stop (run card)
MOTION_EPS_M = 0.02               # moved more than this during an e-stop hold = motion without confirmation
HARVEST_PRESETS = ("harvest_mango", "harvest_guava", "harvest_papaya", "harvest_banana")
PRUNE_PRESETS = ("prune", "prune_banana")
STATUSES = ("pending", "ready", "running", "done", "refused", "skipped", "halted", "aborted")
# timed safety gates (OS 1.2 step 10 / E-STOP row): metric -> (limit s, the step kind that must measure it)
TIMED_SAFETY = {"gust_stance_s": (0.5, "gust"), "crown_stow_s": (8.5, "gust"), "estop_stationary_s": (0.5, "estop")}


class StepAborted(BaseException):
    """Raised inside a running step when the operator aborts (``TourRunner.check_control``). A BaseException so the
    ``except Exception`` of the WP8 / WP13 / WP25 state machines does not turn the abort into a step error."""

# OS 1.2 "tour_lite" (per PR, ~4.3 min sim computed): SCAN 2 trees, two-sided TREE 360 with 4 stations on a hedgerow
# mango, walks, 1 mango clip, 1 papaya twist, unload, 1 person pause. World orchard_lite (OS 4.1).
TOUR_LITE: Dict[str, Any] = {
    "schema": "nandi.mission/1", "id": "tour_lite", "classification": "Public", "world": "orchard_lite",
    "season": "demo_all_fruiting", "budget_s": 258, "gate_s": 300,
    "ctx": {"season_profile": "demo_all_fruiting", "month": 11, "frames_age_h": 0.1, "marks_n": 0},
    "steps": [
        {"id": "s00_ready", "os_step": "0", "title": "Ready check green", "t_s": 5, "budget_s": 5},
        {"id": "l01_scan", "os_step": "1", "title": "SCAN L1 (2 hedgerow mango), south alley, crown mid, 0.15 m/s",
         "preset": "scan", "targets": {"row_ids": ["L1"]}, "params": {"sides": "L", "crown_band": "mid"},
         "budget_s": 45, "alley_y": 0.55},
        {"id": "l02_tree360", "os_step": "2",
         "title": "TREE 360 two-sided on hedgerow mango L1-2, 4 stations (worker actor walks in: pause)",
         "preset": "tree360", "targets": {"tree_ids": ["L1-2"]},
         "params": {"arc_mode": "two_sided", "stations": 4}, "budget_s": 98,
         "person_actor": "worker_1"},
        {"id": "l03_harvest_mango", "os_step": "5", "title": "HARVEST L2-1 mango: CLIPPER, 1 in-band mature fruit",
         "preset": "harvest_mango", "targets": {"tree_ids": ["L2-1"]},
         "params": {"expected_n": 1, "max_items": 1, "pedicel_leave_cm": 2.0}, "budget_s": 26},
        {"id": "l04_harvest_papaya", "os_step": "8", "title": "HARVEST L2-3 papaya: TWIST, one per trip",
         "preset": "harvest_papaya", "targets": {"tree_ids": ["L2-3"]}, "params": {"expected_n": 1},
         "budget_s": 31},
        {"id": "l05_dock_unload", "os_step": "11", "title": "DOCK/UNLOAD: crate, dock, charge, sync, Hindi job sheets",
         "preset": "dock_unload", "params": {"items": 2}, "budget_s": 48},
        {"id": "s12_report", "os_step": "12", "title": "Report: run card", "t_s": 5, "budget_s": 5},
    ],
}


# ====================================================================== program
@dataclass
class TourStep:
    id: str
    os_step: str = ""
    title: str = ""
    preset: Optional[str] = None
    targets: Optional[dict] = None
    params: dict = field(default_factory=dict)
    ctx: dict = field(default_factory=dict)
    expect: dict = field(default_factory=dict)
    t_s: float = 0.0
    budget_s: float = 0.0
    motion: List[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def scripted(self) -> bool:
        return self.preset is None

    @property
    def expect_refusal(self) -> Optional[str]:
        return self.expect.get("reason") if self.expect.get("level") == "no_go" else None

    def tree_ids(self) -> List[str]:
        """Registry tree ids: the Ready check's resolved ids (a target may name the painted ``physical_tag``)."""
        if self.extra.get("_resolved"):
            return list(self.extra["_resolved"])
        return list((self.targets or {}).get("tree_ids") or [])

    def row_ids(self) -> List[str]:
        return list((self.targets or {}).get("row_ids") or [])


@dataclass
class TourProgram:
    id: str
    world: str
    season: str
    budget_s: float
    gate_s: float
    steps: List[TourStep]
    ctx: dict = field(default_factory=dict)
    known_issues: List[dict] = field(default_factory=list)

    @classmethod
    def from_doc(cls, doc: dict) -> "TourProgram":
        known = ("id", "os_step", "title", "preset", "targets", "params", "ctx", "expect", "t_s", "budget_s",
                 "motion")
        steps = []
        for s in doc.get("steps") or []:
            steps.append(TourStep(id=s["id"], os_step=str(s.get("os_step", "")), title=s.get("title", ""),
                                  preset=s.get("preset"), targets=s.get("targets"), params=dict(s.get("params") or {}),
                                  ctx=dict(s.get("ctx") or {}), expect=dict(s.get("expect") or {}),
                                  t_s=float(s.get("t_s") or 0.0), budget_s=float(s.get("budget_s") or 0.0),
                                  motion=list(s.get("motion") or []),
                                  extra={k: v for k, v in s.items() if k not in known}))
        return cls(id=doc["id"], world=doc.get("world", "orchard_demo"), season=doc.get("season", ""),
                   budget_s=float(doc.get("budget_s") or 0.0), gate_s=float(doc.get("gate_s") or 1080.0),
                   steps=steps, ctx=dict(doc.get("ctx") or {}), known_issues=list(doc.get("known_issues") or []))

    def select(self, step_ids: Optional[Sequence[str]]) -> "TourProgram":
        if not step_ids or step_ids == "full":
            return self
        want = list(step_ids)
        unknown = [s for s in want if s not in {st.id for st in self.steps}]
        if unknown:
            raise ValueError(f"unknown tour steps {unknown}")
        p = copy.copy(self)
        p.steps = [s for s in self.steps if s.id in want]
        return p


def load_program(name: str = "tour_lite") -> TourProgram:
    """``tour_lite`` (built in) or a ``config/missions/<name>.yaml`` / path (WP27's ``orchard_tour``)."""
    if name == "tour_lite":
        return TourProgram.from_doc(copy.deepcopy(TOUR_LITE))
    from atharv_nandi.core import mission_logic as ml
    return TourProgram.from_doc(ml.load_mission(name))


def load_field(world: str) -> dict:
    with open(os.path.join(REPO, "config", "fields", f"{world}.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ====================================================================== step results
@dataclass
class StepResult:
    """What a backend returns for one step. ``status`` done | skipped | halted | refused | aborted."""
    status: str = "done"
    reason: str = ""
    sim_s: float = 0.0
    tree_fragments: List[Tuple[str, str, dict]] = field(default_factory=list)   # (kind, tree_id, fragment)
    marks: List[dict] = field(default_factory=list)
    harvest: List[dict] = field(default_factory=list)       # WP8 per-item results (v1) with tree_id, crop
    harvest_kind: str = "harvest"
    kg_by_tree: Dict[str, float] = field(default_factory=dict)
    skipped_targets: int = 0
    radius_reentries: int = 0
    kpi: Dict[str, Any] = field(default_factory=dict)       # scope-specific raw numbers for scope_kpi
    events: List[dict] = field(default_factory=list)
    alerts: List[dict] = field(default_factory=list)
    scripted_delays: List[dict] = field(default_factory=list)
    assisted: List[str] = field(default_factory=list)
    rows_visited: List[str] = field(default_factory=list)
    refused_reason: str = ""
    notes: List[str] = field(default_factory=list)


class Backend:
    """Backend protocol (duck type). ``t`` = sim seconds since the tour started."""

    t: float = 0.0
    detector: str = "simtruth"
    mode: str = "assisted"

    def context(self, step: TourStep) -> dict:            # ReadyContext keyword overrides (robot state)
        return {}

    def run_step(self, step: TourStep, ready: Any, runner: "TourRunner") -> StepResult:
        raise NotImplementedError

    def safety(self) -> dict:                             # cumulative safety counters
        return {"falls": 0, "protective_violations": 0, "unmasked_person_px": 0, "motion_without_confirm": 0}

    def assisted(self) -> List[str]:
        return []

    def close(self):
        pass


# ====================================================================== runner
class TourRunner:
    def __init__(self, program: TourProgram, backend: Backend, *, run_id: Optional[str] = None,
                 device_id: str = "nandi-sim", runs_dir: Optional[str] = None, seed: int = 7,
                 field_doc: Optional[dict] = None, registry=None, publish: Optional[Callable[[str, Any], None]] = None,
                 now: Optional[datetime] = None, write: bool = True, ready: bool = True):
        self.program = program
        self.backend = backend
        self.seed = int(seed)
        self.now = now or datetime.now(timezone.utc)
        self.run_id = run_id or f"{self.now:%Y%m%dT%H%M%SZ}-{program.id}-{seed}"
        self.ident = rec.identity(self.run_id, device_id, rec.mission_id(device_id, self.now))
        self.field = field_doc if field_doc is not None else load_field(program.world)
        self.entries = {e["tree_id"]: e for e in self.field.get("trees") or []}
        self.zone_id = self.field.get("zone_id") or self.field.get("world") or program.world
        self.registry = registry
        self.use_ready = ready
        self.publish = publish or (lambda topic, payload: None)
        self.runs_dir = runs_dir or os.environ.get("NANDI_RUNS_DIR") or os.path.join(REPO, "runs")
        self.run_dir = os.path.join(self.runs_dir, self.run_id)
        self.write = write
        self.trees: Dict[str, rec.TreeRecordBuilder] = {}
        self.marks: List[dict] = []
        self.harvest: List[dict] = []
        self.step_rows: List[dict] = []
        self.events: List[dict] = []
        self.alerts: List[dict] = []
        self.scripted_delays: List[dict] = []
        self.assisted: set = set()
        self.rows_visited: List[str] = []
        self.paused = False
        self.abort_requested = False
        self.abort_reason = "operator"
        self.state = "idle"
        self.current: Optional[str] = None
        self.wall0 = None
        self.result: Optional[dict] = None
        self.kpi_steps: Dict[str, List[dict]] = {}

    # ---------------------------------------------------------------- control
    def pause(self):
        self.paused = True
        self._emit("paused")

    def resume(self):
        self.paused = False
        self._emit("resumed")

    def abort(self, reason: str = "operator"):
        self.abort_requested = True
        self.abort_reason = reason
        self._emit("abort_requested", reason=reason)

    def check_control(self):
        """Called by a backend inside a step: blocks (wall time, sim frozen) while paused, raises ``StepAborted``
        once an abort is requested."""
        while self.paused and not self.abort_requested:
            time.sleep(0.05)
        if self.abort_requested:
            raise StepAborted(getattr(self, "abort_reason", "operator"))

    def _emit(self, event: str, **kw):
        ev = {"t": round(float(getattr(self.backend, "t", 0.0)), 2), "event": event, **kw}
        self.events.append(ev)
        self.publish("/nandi/demo/events", ev)

    def status(self) -> dict:
        """``/nandi/demo/state`` (MVP-SPEC 5.7): step list with status and times, current step, pass so far."""
        return {"schema": STATE_SCHEMA, "run_id": self.run_id, "mission_id": self.ident["mission_id"],
                "program": self.program.id, "world": self.program.world, "state": self.state,
                "current": self.current, "paused": self.paused, "t_sim": round(float(self.backend.t), 1),
                "steps": [{k: r.get(k) for k in ("id", "os_step", "title", "status", "reason", "sim_s", "budget_s")}
                          for r in self.step_rows],
                "pending": [s.id for s in self.program.steps if s.id not in {r["id"] for r in self.step_rows}],
                "tree_records": len(self.trees), "marks": len(rec.dedupe_marks(self.marks)),
                "harvest_records": len(self.harvest), "safety": self.backend.safety(),
                "pass": (self.result or {}).get("tour", {}).get("pass"), "data_provenance": "simulated"}

    # ---------------------------------------------------------------- ready check
    def ready_check(self, step: TourStep, measured: Optional[dict] = None):
        """``measured`` (a refusal probe's context read on the spot) replaces the step's scripted ``ctx``."""
        if step.scripted or not self.use_ready:
            return None
        from atharv_nandi.core import presets as ps, ready_check as rc
        if self.registry is None:
            self.registry = ps.Registry.load()
        kw = dict(self.program.ctx)
        kw.update(self.backend.context(step))
        kw.update(step.ctx if measured is None else measured)
        ctx = rc.ReadyContext.nominal(**{k: v for k, v in kw.items() if k in rc.ReadyContext.__dataclass_fields__})
        return rc.check(self.registry.get(step.preset), ctx, step.targets, step.params, self.field)

    # ---------------------------------------------------------------- run
    def run(self) -> dict:
        self.wall0 = time.perf_counter()
        self.state = "running"
        self._emit("tour_start", program=self.program.id, world=self.program.world)
        for step in self.program.steps:
            while self.paused and not self.abort_requested:
                time.sleep(0.05)
            if self.abort_requested:
                self.step_rows.append(self._row(step, "aborted", "operator"))
                continue
            self.run_one(step)
        return self.finish()

    def run_one(self, step: TourStep) -> dict:
        self.current = step.id
        t0 = float(self.backend.t)
        self._emit("step_start", step=step.id)
        ready = None
        try:
            ready = self.ready_check(step)
        except Exception as exc:  # noqa: BLE001 - a Ready-check bug refuses the step with the reason, never runs it
            ready = None
            res = StepResult(status="refused", reason=f"ready_check_error:{type(exc).__name__}")
            return self._close(step, res, t0, None)
        self.resolve(step, ready)
        want = step.expect_refusal
        if ready is not None and ready.level == "no_go":
            res = StepResult(status="refused", reason=",".join(ready.reasons), refused_reason=",".join(ready.reasons))
            if want:
                res.status = "done" if want in ready.reasons else "halted"
                res.reason = f"refused as expected: {want}" if want in ready.reasons else \
                    f"refused for {ready.reasons}, expected {want}"
                res.notes.append("expected refusal (OS 1.2 step 10 b)")
                if res.status == "done":
                    res = self._probe_refusal(step, want, res)
                # the scripted refusal time (tile shows the reason)
                if res.status != "aborted":
                    self.backend_wait(step.t_s, res)
            return self._close(step, res, t0, ready)
        if want:
            # the gates did not refuse a step that must be refused: a contradiction of the gate table
            res = StepResult(status="halted", reason=f"expected refusal {want} not raised")
            return self._close(step, res, t0, ready)
        try:
            res = self.backend.run_step(step, ready, self)
        except StepAborted as exc:
            res = StepResult(status="aborted", reason=f"{exc.args[0] if exc.args else 'operator'} (inside the step)")
        except Exception as exc:  # noqa: BLE001 - a step crash halts that step and the tour continues stopped
            import traceback
            res = StepResult(status="halted", reason=f"step_error:{type(exc).__name__}: {exc}")
            res.notes.append(traceback.format_exc()[-1500:])
        return self._close(step, res, t0, ready)

    def _probe_refusal(self, step: TourStep, want: str, res: StepResult) -> StepResult:
        """The scripted ``ctx`` refused the step; a backend with ``refusal_probe`` walks the step's motion to the
        spot, measures the context there (terrain truth, robot state) and the Ready check runs again on that: the
        step passes only if the measured world refuses it for the same reason. A backend without a probe keeps the
        scripted refusal, labelled on the card."""
        probe = getattr(self.backend, "refusal_probe", None)
        if probe is None:
            res.notes.append("refusal from the scripted step ctx only (backend has no refusal_probe)")
            return res
        try:
            measured, notes = probe(step, self)
            res.notes += list(notes or [])
            again = self.ready_check(step, measured=measured)
        except StepAborted as exc:
            res.status, res.reason = "aborted", f"{exc.args[0] if exc.args else 'operator'} (inside the step)"
            return res
        except Exception as exc:  # noqa: BLE001 - a probe bug is a halt, never a pass
            res.status, res.reason = "halted", f"refusal_probe_error:{type(exc).__name__}: {exc}"
            return res
        if again is None or again.level != "no_go" or want not in again.reasons:
            res.status = "halted"
            res.reason = (f"measured context {measured} not refused for {want}: "
                          f"{None if again is None else (again.level, again.reasons)}")
        else:
            res.reason += f" (measured on the spot: {measured})"
        return res

    def resolve(self, step: TourStep, ready):
        """Target tree ids of a step: the Ready result's ids, else tags / ids looked up in the registry."""
        raw = list((step.targets or {}).get("tree_ids") or [])
        if ready is not None and getattr(ready, "tree_ids", None) and raw:
            step.extra["_resolved"] = list(ready.tree_ids)
            return
        if raw:
            by_tag = {e.get("physical_tag"): e["tree_id"] for e in self.entries.values()}
            step.extra["_resolved"] = [t if t in self.entries else by_tag.get(t, t) for t in raw]

    def backend_wait(self, s: float, res: StepResult):
        fn = getattr(self.backend, "wait", None)
        if fn and s > 0:
            fn(s)
            res.scripted_delays.append({"what": "refusal shown", "s": s})

    def _row(self, step: TourStep, status: str, reason: str = "", sim_s: float = 0.0) -> dict:
        return {"id": step.id, "os_step": step.os_step, "title": step.title, "preset": step.preset,
                "status": status, "reason": reason, "sim_s": round(sim_s, 1), "budget_s": step.budget_s,
                "skipped_targets": 0, "radius_reentries": 0, "notes": [], "kpi": {}}

    def _close(self, step: TourStep, res: StepResult, t0: float, ready) -> dict:
        sim_s = max(res.sim_s, float(self.backend.t) - t0)
        row = self._row(step, res.status, res.reason, sim_s)
        row.update({"skipped_targets": res.skipped_targets, "radius_reentries": res.radius_reentries,
                    "kpi": res.kpi, "notes": res.notes,
                    "ready": None if ready is None else {"level": ready.level, "reasons": ready.reasons,
                                                         "adjust": ready.adjust, "chips": ready.chips}})
        self.step_rows.append(row)
        self.kpi_steps.setdefault(step.preset or "scripted", []).append({**res.kpi, "step": step.id,
                                                                        "skipped_targets": res.skipped_targets})
        self.assisted |= set(res.assisted)
        self.scripted_delays += [dict(d, step=step.id) for d in res.scripted_delays]
        self.alerts += res.alerts
        self.events += [dict(e, step=step.id) for e in res.events]
        for r in res.rows_visited:
            if r not in self.rows_visited:
                self.rows_visited.append(r)
        try:
            self._merge(step, res)
        except Exception as exc:  # noqa: BLE001 - a record bug halts this step; the tour goes on to the report
            row["status"] = res.status = "halted"
            row["reason"] = res.reason = f"merge_error:{type(exc).__name__}: {exc}"
        self._emit("step_end", step=step.id, status=res.status, reason=res.reason, sim_s=round(sim_s, 1))
        self.publish("/nandi/demo/state", self.status())
        return row

    def _builder(self, tree_id: str) -> rec.TreeRecordBuilder:
        if tree_id not in self.trees:
            self.trees[tree_id] = rec.TreeRecordBuilder(self.ident, self.entries.get(tree_id) or
                                                        {"tree_id": tree_id}, self.zone_id)
        return self.trees[tree_id]

    def _merge(self, step: TourStep, res: StepResult):
        t = float(self.backend.t)
        res.marks = [rec.plain(m) for m in res.marks]
        res.harvest = [rec.plain(h) for h in res.harvest]
        res.tree_fragments = [(k, tid, rec.plain(f)) for k, tid, f in res.tree_fragments]
        for kind, tid, frag in res.tree_fragments:
            b = self._builder(tid)
            if kind == "scan":
                b.merge_scan(frag, t)
            elif kind == "tree360":
                b.merge_tree360(frag, t)
            else:
                b.merge_perception(frag, kind, t)
            row = self.entries.get(tid, {}).get("row_id")
            if row and row not in self.rows_visited:
                self.rows_visited.append(row)
        by_tree: Dict[str, List[dict]] = {}
        for r in res.harvest:
            by_tree.setdefault(r["tree_id"], []).append(r)
        marks_by_tree: Dict[str, List[dict]] = {}
        for m in res.marks:
            marks_by_tree.setdefault(m["tree_id"], []).append(m)
        for tid in set(by_tree) | set(marks_by_tree):
            b = self._builder(tid)
            if res.harvest_kind == "prune":
                b.merge_prune(by_tree.get(tid, []), marks_by_tree.get(tid, []), t)
            else:
                b.merge_harvest(by_tree.get(tid, []), marks_by_tree.get(tid, []), res.kg_by_tree.get(tid), t)
            row = self.entries.get(tid, {}).get("row_id")
            if row and row not in self.rows_visited:
                self.rows_visited.append(row)
        for tid, kg in res.kg_by_tree.items():
            if res.harvest_kind == "unload":
                self._builder(tid).set_harvest_kg(kg)
        self.marks = rec.dedupe_marks(self.marks + list(res.marks))
        for r in res.harvest:
            e = self.entries.get(r["tree_id"])
            self.harvest.append(rec.harvest_record_v2(r, self.ident, e, kind=res.harvest_kind, t=t,
                                                      mass_kg=r.get("mass_kg"), stage=r.get("stage"),
                                                      zone_id=self.zone_id, assisted=res.assisted))
            self.publish("/nandi/records/harvest", self.harvest[-1])
        for m in res.marks:
            self.publish("/nandi/mark", m)

    # ---------------------------------------------------------------- pass definition
    def evaluate(self) -> dict:
        safety = self.backend.safety()
        fails = []
        for k in ("falls", "protective_violations", "unmasked_person_px", "motion_without_confirm"):
            if int(safety.get(k) or 0) > 0:
                fails.append(f"safety:{k}={safety[k]}")
        kinds = {self.step_kind(r) for r in self.step_rows if r["status"] not in ("aborted", "pending", "refused")}
        for k, (limit, kind) in TIMED_SAFETY.items():
            v = safety.get(k)
            if v is not None and float(v) > limit + 1e-9:
                fails.append(f"safety:{k}={v}>{limit}")
            elif v is None and kind in kinds and k in safety:
                fails.append(f"safety:{k}=unmeasured")
        halted = [r for r in self.step_rows if r["status"] in ("halted", "aborted")]
        for r in halted:
            fails.append(f"halted:{r['id']}:{r['reason']}")
        for r in self.step_rows:
            if r["preset"] in HARVEST_PRESETS + PRUNE_PRESETS and r["skipped_targets"] > SKIP_LIMIT:
                fails.append(f"skips:{r['id']}={r['skipped_targets']}")
            if r["preset"] == "tree360" and r["radius_reentries"] > REENTRY_LIMIT:
                fails.append(f"reentries:{r['id']}={r['radius_reentries']}")
            if r["status"] == "refused" and r["preset"] is not None:
                fails.append(f"refused:{r['id']}:{r['reason']}")
            if r["status"] == "skipped" and r["preset"] is not None:
                fails.append(f"skipped:{r['id']}:{r['reason']}")
        sim_s = round(float(self.backend.t), 1)
        return {"pass": not fails, "fail_reasons": fails, "sim_s": sim_s, "gate_s": self.program.gate_s,
                "budget_s": self.program.budget_s, "time_ok": sim_s <= self.program.gate_s,
                "wall_s": round(time.perf_counter() - (self.wall0 or time.perf_counter()), 1),
                "steps_n": len(self.step_rows), "steps_done": sum(1 for r in self.step_rows if r["status"] == "done"),
                "steps_halted": len(halted), "safety": safety}

    @staticmethod
    def step_kind(row: dict) -> str:
        """``gust`` / ``estop`` for the RECOVERY parts that measure a timed safety gate, else the preset."""
        if row.get("preset") == "estop_recovery":
            return "estop"
        if row.get("preset") is None and str(row.get("id", "")).startswith("s10a"):
            return "gust"
        return row.get("preset") or "scripted"

    # ---------------------------------------------------------------- outputs
    def job_sheets(self, rows: Optional[Sequence[str]] = None) -> Dict[str, dict]:
        return js.build_job_sheets(self.marks, self.entries.values(), self.ident,
                                   rows=list(rows if rows is not None else self.rows_visited),
                                   t=float(self.backend.t))

    def tree_records(self, errors: Optional[List[str]] = None) -> List[dict]:
        """Every TreeRecord that builds; one that cannot (still > 16 KiB after trimming, a bad fragment) is left
        out and named in ``errors`` (a contract violation on the card), never a crash of the report."""
        out = []
        for tid, b in self.trees.items():
            try:
                out.append(b.build())
            except Exception as exc:  # noqa: BLE001
                if errors is None:
                    raise
                errors.append(f"{tid}:build:{type(exc).__name__}: {exc}")
        return out

    def finish(self) -> dict:
        self.state = "finishing"
        tour = self.evaluate()
        build_err: List[str] = []
        trees = self.tree_records(build_err)
        sheets = self.job_sheets()
        missing = js.missing_marks(self.marks, sheets)
        viol = list(build_err)
        for r in trees:
            viol += [f"{r['tree_id']}:{v}" for v in rec.validate_tree_record(r)]
        for m in self.marks:
            viol += [f"mark:{v}" for v in rec.validate_mark(m)]
        for h in self.harvest:
            viol += [f"harvest:{v}" for v in rec.validate_harvest(h)]
        for s in sheets.values():
            viol += [f"jobsheet:{v}" for v in js.validate_sheet(s)]
        uplink, uplink_err = self.uplink(trees, tour)
        viol += [f"uplink:{e}" for e in uplink_err]
        table = sk.KpiTable(self.run_id, self.ident["device_id"], self.now.date().isoformat(),
                            mode=getattr(self.backend, "mode", "assisted"),
                            detector=getattr(self.backend, "detector", "simtruth"))
        self._kpis(table, trees, viol, missing, sheets, tour)
        counts = {"tree_records": len(trees), "marks": len(self.marks), "harvest_records": len(self.harvest),
                  "job_sheets": len(sheets), "uplink_messages": len(uplink), "alerts": len(self.alerts)}
        self.assisted |= set(self.backend.assisted())
        card = sk.run_card(ident=self.ident, date=self.now.date().isoformat(), world=self.program.world,
                           season=self.program.season, seed=self.seed, program=self.program.id,
                           detector=getattr(self.backend, "detector", "simtruth"),
                           mode=getattr(self.backend, "mode", "assisted"), assisted=self.assisted,
                           scripted_delays=self.scripted_delays, table=table, counts=counts, tour=tour,
                           notes=[n for r in self.step_rows for n in (r.get("notes") or [])
                                  if not n.startswith("Traceback")][:30], sw=_sw())
        # tele/kpi: the gated KPI rows of the card (metric, value, gate, pass) once the card exists
        kmsg, kerr = self.uplink_kpi(table, card)
        uplink += kmsg
        viol += [f"uplink:{e}" for e in kerr]
        counts["uplink_messages"] = len(uplink)
        card["counts"] = dict(card.get("counts") or {}, uplink_messages=len(uplink))
        self.result = {"schema": "nandi.tour_result/1", **self.ident, "program": self.program.id,
                       "world": self.program.world, "seed": self.seed, "tour": tour, "steps": self.step_rows,
                       "counts": counts, "violations": viol, "jobsheet_missing": missing, "card": card}
        if self.write:
            self._write(trees, sheets, uplink, card)
        self.state = "done"
        self.current = None
        self._emit("tour_end", passed=tour["pass"])
        self.publish("/nandi/kpi", {k: card[k] for k in ("run_id", "device_id", "date", "data_provenance",
                                                         "detector", "assisted", "failed_gates", "tour")})
        self.publish("/nandi/kpi/scopes", {"run_id": self.run_id, "device_id": self.ident["device_id"],
                                           "date": card["date"], "data_provenance": "simulated",
                                           "detector": card["detector"], "assisted": card["assisted"],
                                           "scopes": card["scopes"]})
        for s in sheets.values():
            self.publish("/nandi/jobsheet", s)
        for t in trees:
            self.publish("/nandi/tree/record", t)
        self.publish("/nandi/demo/state", self.status())
        return self.result

    def _kpis(self, table: sk.KpiTable, trees, viol, missing, sheets, tour):
        for k in self.kpi_steps.get("scan", []):
            if k.get("trees") is not None:
                sk.scan_kpis(table, k)
        for k in self.kpi_steps.get("tree360", []):
            if k.get("result"):
                sk.tree360_kpis(table, k["result"])
        for k in self.kpi_steps.get("leaf_scan", []):
            if k.get("leaf") is not None:
                sk.leaf_kpis(table, k["leaf"])
        for k in self.kpi_steps.get("trunk_scan", []):
            if k.get("trunk") is not None:
                sk.trunk_kpis(table, k["trunk"])
        for k in self.kpi_steps.get("fruit_scout", []):
            if k.get("fruit") is not None:
                sk.scout_kpis(table, k["fruit"], k.get("assisted"))
        hs = [k for p in HARVEST_PRESETS for k in self.kpi_steps.get(p, [])]
        if hs:
            sk.harvest_kpis(table, hs)
        ps_ = [k for p in PRUNE_PRESETS for k in self.kpi_steps.get(p, [])]
        if ps_:
            sk.prune_kpis(table, ps_)
        for k in self.kpi_steps.get("dock_unload", []):
            if k.get("dock") is not None or k.get("crate") is not None:
                sk.dock_kpis(table, k)
        sk.safety_kpis(table, tour["safety"])
        planned = sum(int(k.get("marks_planned") or 0) for p in self.kpi_steps for k in self.kpi_steps[p])
        sk.records_kpis(table, trees, len(viol), missing, self.rows_visited, sheets,
                        marks_truth_n=planned or None, marks_n=len(self.marks))
        table.add("tour.sim_s", tour["sim_s"], gate_override=self.program.gate_s)
        table.add("tour.steps_halted", tour["steps_halted"])

    def uplink(self, trees: Sequence[dict], tour: dict) -> Tuple[List[Tuple[str, dict]], List[str]]:
        up = rec.Uplink(device=self.ident["device_id"])
        self._up = up
        t_start = int(self.now.timestamp() * 1000)

        def ts_at(t_sim) -> int:
            """kis/v1 ``ts``: tour start + the sim time of the event (an item / alert at its time; records,
            aggregates and the summary when the tour ends and they are produced)."""
            try:
                return t_start + int(round(1000.0 * max(0.0, float(t_sim))))
            except (TypeError, ValueError):
                return t_end
        t_end = t_start + int(round(1000.0 * float(self.backend.t)))
        ts = t_end
        out, errs = [], []

        def push(fn, *a, **kw):
            try:
                r = fn(*a, **kw)
            except ValueError as exc:
                errs.append(str(exc))
                return
            if r is None:
                return
            if isinstance(r, list):
                out.extend(r)
            else:
                out.append(r)

        for t in trees:
            push(up.tree, ts, t)
        for m in self.marks:
            push(up.mark, ts, m)
        for h in self.harvest:
            push(up.harvest_v2, ts_at(h.get("t_sim")), h)
        for a in self.alerts:
            push(up.alert, ts_at(a.get("t")), a)
        push(up.person_aggregates, ts)
        push(up.mission_summary, ts, {"schema": rec.MISSION_SCHEMA, **self.ident, "program": self.program.id,
                                      "world": self.program.world, "pass": tour["pass"], "sim_s": tour["sim_s"],
                                      "steps": [{"id": r["id"], "status": r["status"]} for r in self.step_rows],
                                      "tree_records": len(trees), "marks": len(self.marks)})
        return out, errs

    def uplink_kpi(self, table: sk.KpiTable, card: dict) -> Tuple[List[Tuple[str, dict]], List[str]]:
        up = getattr(self, "_up", None) or rec.Uplink(device=self.ident["device_id"])
        ts = int(self.now.timestamp() * 1000) + int(round(1000.0 * float(self.backend.t)))
        body = {"schema": "nandi.kpi/1", **self.ident, "program": self.program.id, "date": card.get("date"),
                "detector": card.get("detector"), "mode": card.get("mode"), "pass": card["tour"].get("pass"),
                "failed_gates": list(card.get("failed_gates") or []),
                "rows": [{"metric": r["metric"], "kpi": r["kpi"], "value": r["value"], "gate": r["gate"],
                          "pass": r["pass"], "aggregate": r["aggregate"]} for r in table.rows]}
        try:
            return [up.kpi(ts, body)], []
        except ValueError as exc:
            return [], [str(exc)]

    def _write(self, trees, sheets, uplink, card):
        from atharv_nandi.core import run_log as rl
        d = self.run_dir
        os.makedirs(os.path.join(d, "jobsheets"), exist_ok=True)
        meta = {"run_id": self.run_id, "device_id": self.ident["device_id"], "mission_id": self.ident["mission_id"],
                "tenant": self.ident["tenant"], "start_utc": self.now.isoformat(),
                "start_utc_ms": int(self.now.timestamp() * 1000), "duration_s": float(self.backend.t),
                "world": self.program.world, "seed": self.seed, "program": self.program.id, "sw": _sw(),
                "data_provenance": "simulated"}
        mpath = os.path.join(d, "meta.json")
        old = rl.read_json_dict(mpath)         # a run dir that already has a recorder meta keeps its keys
        if old is not None:
            old.update({k: v for k, v in meta.items() if k not in old})
            meta = old
        rl.write_json(mpath, meta)

        def jl(name, rows):
            with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        jl("tree_records.jsonl", trees)
        jl("marks.jsonl", self.marks)
        jl("harvest_records.jsonl", self.harvest)
        jl("tour_events.jsonl", self.events)
        jl("uplink_kis_v1.jsonl", [{"topic": t, "payload": p} for t, p in uplink])
        for row, s in sheets.items():
            base = os.path.join(d, "jobsheets", f"{row}")
            rl.write_json(base + ".json", s)
            with open(base + ".hi.html", "w", encoding="utf-8") as fh:
                fh.write(js.render_html(s))
            with open(base + ".hi.voice.txt", "w", encoding="utf-8") as fh:
                fh.write("\n".join(js.voice_lines(s)) + "\n")
        rl.write_json(os.path.join(d, "tour.json"), {k: v for k, v in self.result.items() if k != "card"})
        rl.write_json(os.path.join(d, "run_card.json"), card)
        with open(os.path.join(d, "run_card.md"), "w", encoding="utf-8") as fh:
            fh.write(sk.format_card(card))
        # the manifest covers every file of the run (a WP13 sync that closed it keeps its final flag and time)
        from atharv_nandi.core import manifest as mf
        final, closed = False, None
        mpath = os.path.join(d, "manifest.json")
        old = rl.read_json_dict(mpath)
        if old is not None:
            final, closed = bool(old.get("final")), old.get("closed")
        mf.write_manifest(d, meta, final=final, closed_ms=closed)


def _sw() -> str:
    try:
        import subprocess
        h = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           timeout=2).stdout.strip()
        return f"atharv-nandi {h or 'unknown'}"
    except Exception:  # noqa: BLE001
        return "atharv-nandi unknown"


# ====================================================================== geometry helpers (pure)
def path_len(path: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def row_alley_y(field_doc: dict, row_id: str, side: str = "south") -> float:
    """Alley centre beside a row: midway to the neighbour row on that side, else row y -/+ half the 5 m HDP row."""
    rows = sorted(field_doc.get("rows") or [], key=lambda r: float(r["y"]))
    ys = [float(r["y"]) for r in rows]
    me = next((float(r["y"]) for r in rows if r["row_id"] == row_id), None)
    if me is None:                        # a tree outside the rows (OT-1, isolated): its own y
        me = next((float(t["geo"]["y"]) for t in field_doc.get("trees") or [] if t.get("row_id") == row_id), 0.0)
        return me - 2.0 if side == "south" else me + 2.0
    if side == "south":
        lower = [y for y in ys if y < me - 1e-6]
        return (me + max(lower)) / 2.0 if lower else me - 2.0
    upper = [y for y in ys if y > me + 1e-6]
    return (me + min(upper)) / 2.0 if upper else me + 2.0


def _seg_point_dist(a, b, p) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    u = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(ax + u * dx - p[0], ay + u * dy - p[1])


def alley_lines(field_doc: dict) -> List[float]:
    """y of every alley centre line: below the first row, between rows, above the last (orchard rows run along x)."""
    ys = sorted(float(r["y"]) for r in field_doc.get("rows") or [])
    if not ys:
        return [0.0]
    out = [ys[0] - 2.0] + [(a + b) / 2.0 for a, b in zip(ys, ys[1:])] + [ys[-1] + 2.0]
    return out


def plan_route(field_doc: dict, start, goal, clear_m: float = 0.45) -> List[Tuple[float, float]]:
    """Waypoints that keep the trunk ``clear_m`` outside every canopy (keep-out circle canopy_r + clear_m): straight
    if clear, else out to the nearest alley line, along it to the nearer headland, along the headland to the goal's
    alley line and in. A keep-out that contains the start or the goal is ignored (the robot is at that tree)."""
    trees = []
    for e in field_doc.get("trees") or []:
        g = e.get("geo") or {}
        trees.append((float(g["x"]), float(g["y"]), float(e.get("canopy_r_m") or 0.8) + clear_m))
    s, gl = (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
    near = [t for t in trees if math.hypot(s[0] - t[0], s[1] - t[1]) < t[2] or math.hypot(gl[0] - t[0], gl[1] - t[1])
            < t[2]]
    obst = [t for t in trees if t not in near]

    def clear(a, b):
        return all(_seg_point_dist(a, b, t[:2]) >= t[2] for t in obst)

    if clear(s, gl):
        return [gl]
    lines = alley_lines(field_doc)
    ay_s = min(lines, key=lambda y: abs(y - s[1]))
    ay_g = min(lines, key=lambda y: abs(y - gl[1]))
    if trees:
        west = min(t[0] - t[2] for t in trees) - 0.4
        east = max(t[0] + t[2] for t in trees) + 0.4
    else:
        west, east = s[0] - 1.0, s[0] + 1.0
    cands = []
    if abs(ay_s - ay_g) < 1e-6:
        cands.append([(s[0], ay_s), (gl[0], ay_g), gl])
    for hx in (west, east):
        cands.append([(s[0], ay_s), (hx, ay_s), (hx, ay_g), (gl[0], ay_g), gl])
    best = None
    for c in cands:
        pts = [s] + c
        ok = all(clear(a, b) for a, b in zip(pts, pts[1:]))
        L = path_len(pts)
        key = (0 if ok else 1, L)
        if best is None or key < best[0]:
            best = (key, c)
    out = []
    for p in best[1]:
        if not out or math.dist(out[-1], p) > 0.05:
            out.append((round(p[0], 3), round(p[1], 3)))
    return out


def scan_flags(leads: Sequence[dict], fruit_in_view: int, prune_in_view: int, threshold: float = 0.5) -> List[str]:
    """MP 1 step 6 triage flags from what the SCAN pass saw: trunk leads -> trunk_scan; leaf leads -> leaf_scan;
    canopy (panicle / shoot) leads -> tree360; mature fruit in view -> harvest_window; in-band prune candidates in
    view -> prune. Applied to detector leads (the record) and to truth-in-view (the KPI reference)."""
    strong = [ld for ld in leads if float(ld.get("score", 1.0)) >= threshold]
    flags = set()
    if any(ld.get("unit") == "trunk" for ld in strong):
        flags.add("trunk_scan")
    if any(ld.get("unit") == "per_leaf" for ld in strong):
        flags.add("leaf_scan")
    if any(ld.get("unit") in ("per_panicle", "per_shoot") for ld in strong):
        flags.add("tree360")
    if fruit_in_view:
        flags.add("harvest_window")
    if prune_in_view:
        flags.add("prune")
    return sorted(flags)


def triage_score(flags: Sequence[str], truth: Sequence[str]) -> dict:
    f, t = set(flags), set(truth)
    return {"tp": len(f & t), "fp": len(f - t), "fn": len(t - f)}


# ====================================================================== SimDriver backend
ALERT_TOPICS = ("/nandi/platform/alert", "/nandi/recovery/alert", "/nandi/stability/alert")
# scripted actors of the demo script (OS 1.2): the worker in the orbit (step 2), the dog at G-2 (step 6)
STEP_ACTORS = {"s02_tree360": ("worker_1", "person"), "s06_harvest_guava": ("dog_1", "animal")}


class SimBackend(Backend):
    """Every tour step on one ``SimDriver`` world with every ``atharv_nandi/sim/hooks`` file loaded (assisted register: truth
    poses for SCAN views, WP8 station teleports, WP26 planners on truth; each one is named in ``assisted()``)."""

    def __init__(self, world: str = "orchard_lite", seed: int = 7, *, transit: str = "walk",
                 capture: str = "geometry", actors: bool = True, log: Optional[Callable[[str], None]] = None):
        os.environ["NANDI_WORLD"] = world
        os.environ["NANDI_SEED"] = str(int(seed))
        os.environ["NANDI_NOVNC_UP"] = "0"
        os.environ["NANDI_ACTORS"] = "1" if actors else "0"
        from atharv_nandi.core import farm_scene
        from atharv_nandi.core.sim_driver import SimDriver
        from atharv_nandi.core.task_retry import RetryPolicy
        from atharv_nandi.core.tasks import sim_executor as sx, tree_task
        from atharv_nandi.sim.hooks import tools as htools
        from atharv_nandi.sim.world.modules import trees as trees_mod
        from atharv_nandi.sim.world import crops
        self.world, self.seed, self.transit_mode, self.capture = world, int(seed), transit, capture
        self.log = log or (lambda s: None)
        self.m = farm_scene.load_farm_model()
        self.drv = SimDriver(self.m)
        self.rt = htools.install(self.drv)
        self.drv.wait(0.3)
        self.published: List[Tuple[str, Any]] = []
        self.ex = sx.SimTaskExecutor(self.drv, self.rt, publish=lambda tp, p: self.published.append((tp, p)))
        self.tree_runner = sx.TreeRunner(self.ex, policy=RetryPolicy(per_target=1), station_fn=self._station)
        self.plans = {p["tree_id"]: p for p in trees_mod.plans_for_model(self.m)}
        self.profiles = crops.load_profiles()
        self.field = load_field(world)
        # scripted actors stand still at their homes until a step starts them (OS 1.2 steps 2 and 6); park them out
        # of the field meanwhile so a dog at its home does not latch the fields of unrelated steps
        for a in self.field.get("actors") or []:
            self.drv.deliver("/nandi/world/cmd", {"actor": a["id"], "action": "park"})
        self.drv.wait(0.1)
        cand_path = os.path.join(REPO, "config", "fields", "orchard_candidates.json")
        if world == "orchard_demo":
            with open(cand_path, encoding="utf-8") as fh:
                self.stations = tree_task.frozen_stations(json.load(fh))
        else:
            self.stations = {}
        self.today = self.field.get("sim_date")
        self.t0 = float(self.drv.t)
        self._assisted = {"truth_pose_scan_views", "lidar_profile_truth", "simtruth_leads"}
        self._safety = {"falls": 0, "protective_violations": 0, "unmasked_person_px": 0,
                        "motion_without_confirm": 0, "protective_stops": 0, "min_tripod_margin_crown_up_m": None,
                        "gust_stance_s": None, "crown_stow_s": None, "estop_stationary_s": None,
                        "person_pixels_measured": False}
        self._tilt_since = None
        self._fallen = False
        self._latch_prev = False
        self._latch_check: Optional[float] = None
        self._hold_ref = None
        self._own_hold = False
        self._clear_since = None
        self.stop_log: List[dict] = []
        self.scripted_acks: List[dict] = []
        self.alerts: List[dict] = []
        self.drv.on_publish = self._on_publish
        self.drv.add_task("wp9_safety", self._safety_task, 10.0)
        # operator pause / abort inside a step: TourRunner.check_control at 10 Hz sim (blocks / raises StepAborted)
        self._ctl_runner: Optional[TourRunner] = None
        self.drv.add_task("wp9_control", self._control_task, 10.0)
        self._watch: Optional[dict] = None      # stowed-transit watch (max crown ext, min tripod margin)
        self.did: Dict[str, List[str]] = {}
        self.frames_t: Dict[str, float] = {}

    # ---------------------------------------------------------------- basics
    @property
    def t(self) -> float:
        return float(self.drv.t) - self.t0

    def wait(self, s: float):
        self.drv.stand(float(s))

    def assisted(self) -> List[str]:
        return sorted(self._assisted)

    def safety(self) -> dict:
        return dict(self._safety)

    def _on_publish(self, topic: str, text: str):
        if topic in ALERT_TOPICS or topic.endswith("/alert"):
            try:
                a = json.loads(text)
            except ValueError:
                return
            if isinstance(a, dict):
                a.setdefault("t", self.t)
                self.alerts.append(a)

    def _control_task(self, drv):
        r = self._ctl_runner
        if r is not None:
            r.check_control()

    def _safety_task(self, drv):
        import numpy as np
        if self._watch is not None:
            w = self._watch
            w["ext_max"] = max(w["ext_max"], float((drv.bus.values.get("mast") or {}).get("ext_m") or 0.0))
            tm = (drv.bus.values.get("stability") or {}).get("predicted_tripod_m")
            if isinstance(tm, (int, float)) and (drv.bus.values.get("stability") or {}).get("walking"):
                w["tripod_min"] = float(tm) if w["tripod_min"] is None else min(w["tripod_min"], float(tm))
        roll, pitch = drv.roll_pitch()
        tilt = math.degrees(max(abs(roll), abs(pitch)))
        if tilt > FALL_TILT_DEG:
            self._tilt_since = self._tilt_since if self._tilt_since is not None else drv.t
            if drv.t - self._tilt_since >= 0.3 and not self._fallen:
                self._fallen = True
                self._safety["falls"] += 1
        else:
            self._tilt_since = None
        sf = drv.bus.values.get("safety_fields") or {}
        latched = bool(sf.get("latched")) and sf.get("latch_reason") not in ("stability", "")
        if latched and not self._latch_prev:
            self._safety["protective_stops"] += 1
            self._latch_check = drv.t + PROTECTIVE_STOP_S
            x, y = drv.trunk_xy()
            self.stop_log.append({"t": round(self.t, 2), "reason": sf.get("latch_reason"), "detail": sf.get("detail"),
                                  "xy": [round(float(x), 2), round(float(y), 2)]})
            self.log(f"[{self.t:7.1f}] protective latch {sf.get('latch_reason')} {sf.get('detail')} at "
                     f"({x:.2f}, {y:.2f})")
            if not drv.holding:
                # the live safety_supervisor adopts a field latch as an e-stop latch (runner hold); headless, the
                # tour does the same so the hook's hand-over and clear rule apply (hooks/stability.py)
                drv.enter_hold("protective_stop")
                self._own_hold = True
                self._clear_since = None
        self._latch_prev = latched
        if self._own_hold and drv.holding:
            if sf.get("protective_clear") and not sf.get("latched"):
                self._clear_since = self._clear_since if self._clear_since is not None else drv.t
                if drv.t - self._clear_since >= OPERATOR_ACK_S:
                    # the operator clears the stop (scripted, declared on the run card: OS 1.2 step 6 "dog stop
                    # latched and cleared by the operator")
                    drv.leave_hold()
                    self._own_hold = False
                    self.scripted_acks.append({"what": "operator clears the protective stop", "s": OPERATOR_ACK_S,
                                               "t": round(self.t, 2)})
            else:
                self._clear_since = None
        if self._latch_check is not None and drv.t >= self._latch_check:
            self._latch_check = None
            v = float(np.linalg.norm(drv.d.qvel[0:2]))
            if v > 0.05:
                self._safety["protective_violations"] += 1
                if self.stop_log:
                    self.stop_log[-1]["violation_speed_mps"] = round(v, 3)
                self.log(f"[{self.t:7.1f}] protective violation: trunk {v:.3f} m/s {PROTECTIVE_STOP_S} s after the latch")
        st = drv.bus.values.get("stability") or {}
        ext = float((drv.bus.values.get("mast") or {}).get("ext_m") or 0.0)
        if ext > 0.02 and st.get("walking") and isinstance(st.get("predicted_tripod_m"), (int, float)):
            cur = self._safety["min_tripod_margin_crown_up_m"]
            v = float(st["predicted_tripod_m"])
            self._safety["min_tripod_margin_crown_up_m"] = round(v if cur is None else min(cur, v), 4)
        if self._hold_ref is not None:
            moved = float(np.linalg.norm(drv.d.qpos[0:2] - self._hold_ref))
            if moved > MOTION_EPS_M:
                self._safety["motion_without_confirm"] += 1
                self._hold_ref = drv.d.qpos[0:2].copy()

    def context(self, step: TourStep) -> dict:
        b = self.drv.bus.values
        mast = b.get("mast") or {}
        tools = b.get("tools") or {}
        return {"robot_xy": tuple(float(v) for v in self.drv.trunk_xy()), "ext_m": float(mast.get("ext_m") or 0.0),
                "tool_mounted": tools.get("mounted") if isinstance(tools, dict) else None,
                "estop": bool(self.drv.holding) and not step.ctx.get("estop") is False}

    # ---------------------------------------------------------------- motion
    def _walk_to(self, xy, vx: float = TRANSIT_VX, tol: float = 0.15, timeout_s: Optional[float] = None) -> bool:
        drv = self.drv
        tx, ty = float(xy[0]), float(xy[1])
        x0, y0 = drv.trunk_xy()
        timeout_s = timeout_s or 25.0 + 9.0 * math.hypot(tx - x0, ty - y0)

        prog = {"t": drv.t, "xy": (x0, y0), "h": drv.heading(), "stalled": False}

        def drive(_):
            x, y = drv.trunk_xy()
            if math.hypot(tx - x, ty - y) <= tol:
                return True
            # no progress (< 4 cm and < 10 deg in 6 s): give up early, the caller teleports (no sim time burnt)
            if drv.t - prog["t"] >= 6.0:
                dh = abs((drv.heading() - prog["h"] + math.pi) % (2 * math.pi) - math.pi)
                if math.hypot(x - prog["xy"][0], y - prog["xy"][1]) < 0.04 and dh < math.radians(10.0):
                    prog["stalled"] = True
                    return True
                prog.update(t=drv.t, xy=(x, y), h=drv.heading())
            ang = (math.atan2(ty - y, tx - x) - drv.heading() + math.pi) % (2 * math.pi) - math.pi
            if abs(ang) > 0.35:
                drv.gait.set_command(0.0, 0.0, 1.0 if ang > 0 else -1.0, hold=True)
            else:
                drv.gait.set_command(vx, 0.0, max(-1.0, min(1.0, 1.5 * ang)), hold=True)
            return False

        ok = drv.run_until(drive, timeout_s) and not prog["stalled"]
        drv.gait.release()
        return ok

    def _teleport(self, xy, heading_deg: Optional[float] = None):
        from atharv_nandi.core.tasks import sim_executor as sx
        h = math.degrees(self.drv.heading()) if heading_deg is None else heading_deg
        sx.teleport_station(self.drv, {"xy": list(xy), "heading_deg": h}, settle_s=0.8)

    def transit(self, path: Sequence[Sequence[float]], res: StepResult, vx: float = TRANSIT_VX) -> float:
        """Walk ``path`` (waypoints) stowed at <= 0.18 m/s. A waypoint not reached in time is teleported to
        (assisted ``transit_teleport_fallback``, the walk time credited at 0.18 m/s); ``transit: teleport`` does that
        for every leg."""
        dist = 0.0
        self.leave_tree(res)
        path = self.detour_dock(list(path))
        for wp in path:
            if self.in_keepout(wp):
                continue                  # a WP27 path point on a station: the WP8 station teleport goes there
            here = self.drv.trunk_xy()
            leg = math.hypot(float(wp[0]) - here[0], float(wp[1]) - here[1])
            if leg < 0.15:
                continue
            ok = False
            if self.transit_mode == "walk":
                ok = self._walk_to(wp, vx)
            if not ok:
                self._teleport(wp)
                self.wait(leg / 0.18)
                self._assisted.add("transit_teleport_fallback" if self.transit_mode == "walk" else "transit_teleport")
                res.notes.append(f"transit leg to {tuple(round(float(v), 2) for v in wp)} teleported "
                                 f"({leg:.1f} m credited at 0.18 m/s)")
            dist += leg
        self.drv.stand(0.3)
        return dist

    def detour_dock(self, path: List[Sequence[float]]) -> List[Tuple[float, float]]:
        """WP4's dock backboard (0.45 m behind the plate) is a wall: a leg that passes within 0.6 m of it gets two
        waypoints 1.3 m to the side of the dock axis (the goal's side). Finding CR-WP9-01: the WP27 tour path
        home -> (-1.0, 0.5) runs into it."""
        d = self.field.get("dock") or {}
        if not d.get("xy"):
            return [tuple(p) for p in path]
        yaw = math.radians(float(d.get("yaw_deg") or 0.0))
        half = float((d.get("size_m") or [1.0, 0.8])[0]) / 2.0
        c, s_ = math.cos(yaw), math.sin(yaw)
        ox, oy = float(d["xy"][0]), float(d["xy"][1])

        def w(lx, ly):
            return (ox + c * lx - s_ * ly, oy + s_ * lx + c * ly)

        bb_a, bb_b = w(-half - 0.45, -0.45), w(-half - 0.45, 0.45)

        def seg_seg(a, b):
            n = 12
            return min(_seg_point_dist(bb_a, bb_b, (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n))
                       for k in range(n + 1))

        out: List[Tuple[float, float]] = []
        prev = tuple(float(v) for v in self.drv.trunk_xy())
        for p in path:
            p = (float(p[0]), float(p[1]))
            if _seg_point_dist(bb_a, bb_b, p) < 0.6:          # a waypoint at the wall: moved 1.3 m to its side
                lx = c * (p[0] - ox) + s_ * (p[1] - oy)
                ly = -s_ * (p[0] - ox) + c * (p[1] - oy)
                p = w(lx, 1.3 if ly >= 0 else -1.3)
            if seg_seg(prev, p) < 0.6:
                side = 1.0 if (-(p[0] - ox) * s_ + (p[1] - oy) * c) >= 0 else -1.0
                out += [w(0.3, 1.3 * side), w(-half - 1.1, 1.3 * side)]
            out.append(p)
            prev = p
        # the crate (WP4 fixture) is a solid box: legs keep the trunk >= half its diagonal + 0.55 m away
        # (finding CR-WP9-01: the WP27 s06 path (3.0, -5.0) -> (0.0, -1.0) runs over the crate at (0.5, -1.5))
        cr = self.field.get("crate") or {}
        if cr.get("xy"):
            cx, cy = float(cr["xy"][0]), float(cr["xy"][1])
            sz = cr.get("size_m") or [0.6, 0.4]
            rad = 0.5 * math.hypot(float(sz[0]), float(sz[1])) + 0.55
            for _ in range(4):                # insert detour points until every leg clears (a few passes)
                res2: List[Tuple[float, float]] = []
                prev = tuple(float(v) for v in self.drv.trunk_xy())
                changed = False
                for p in out:
                    dp = math.hypot(p[0] - cx, p[1] - cy)
                    if dp < rad:                  # a waypoint at the crate: pushed out radially
                        ux, uy = ((p[0] - cx) / dp, (p[1] - cy) / dp) if dp > 1e-6 else (0.0, 1.0)
                        p = (cx + ux * (rad + 0.1), cy + uy * (rad + 0.1))
                        changed = True
                    if _seg_point_dist(prev, p, (cx, cy)) < rad and math.hypot(prev[0] - cx, prev[1] - cy) >= rad:
                        ax, ay = p[0] - prev[0], p[1] - prev[1]
                        L2 = ax * ax + ay * ay or 1.0
                        u_ = max(0.0, min(1.0, ((cx - prev[0]) * ax + (cy - prev[1]) * ay) / L2))
                        mx, my = prev[0] + u_ * ax - cx, prev[1] + u_ * ay - cy   # closest approach
                        L = math.hypot(mx, my)
                        if L < 1e-6:
                            ax, ay = p[0] - prev[0], p[1] - prev[1]
                            mx, my, L = -ay, ax, math.hypot(ax, ay) or 1.0
                        res2.append((cx + mx / L * (rad + 0.35), cy + my / L * (rad + 0.35)))
                        changed = True
                    res2.append(p)
                    prev = p
                out = res2
                if not changed:
                    break
        return out

    def in_keepout(self, xy, clear_m: float = 0.45) -> bool:
        return any(math.hypot(float(xy[0]) - float(e["geo"]["x"]), float(xy[1]) - float(e["geo"]["y"]))
                   < float(e.get("canopy_r_m") or 0.8) + clear_m for e in self.field.get("trees") or [])

    def leave_tree(self, res: StepResult):
        """After a WP8 station (teleported in, assisted ``station_teleport``) the robot stands on the basin rim inside
        the tree's canopy keep-out, where turning in place stalls; it is put back on the nearest alley line the same
        way (``station_teleport``, the 0.18 m/s walk time credited)."""
        x, y = (float(v) for v in self.drv.trunk_xy())
        for e in self.field.get("trees") or []:
            g = e["geo"]
            r = float(e.get("canopy_r_m") or 0.8) + 0.45
            if math.hypot(x - float(g["x"]), y - float(g["y"])) < r:
                ay = min(alley_lines(self.field), key=lambda v: abs(v - y))
                d = abs(ay - y)
                self._teleport((x, ay), 0.0)
                self.wait(d / 0.18)
                self._assisted.add("station_teleport")
                res.notes.append(f"left the {e['tree_id']} station to the alley y {ay:.2f} (teleport, {d:.1f} m "
                                 f"credited at 0.18 m/s)")
                return

    def route_to(self, goal, res: StepResult) -> float:
        self.leave_tree(res)
        here = tuple(float(v) for v in self.drv.trunk_xy())
        return self.transit(plan_route(self.field, here, goal), res)

    def motion_transits(self, step: TourStep, res: StepResult) -> bool:
        """Walk the step's scripted transit walks that come before its first dwell / scan segment (WP27 paths)."""
        walked = False
        for g in step.motion:
            if g.get("kind") == "walk" and g.get("transit"):
                self.transit(g["path"][1:] if walked else g["path"], res)
                walked = True
            elif g.get("kind") in ("dwell", "raise") or (g.get("kind") == "walk" and not g.get("transit")):
                break
        return walked

    def mast(self, band: str, timeout_s: float = 12.0) -> Tuple[bool, float, str]:
        """WP24 preset request; returns (reached, seconds, last refusal detail)."""
        import numpy as np
        from atharv_nandi.sim.hooks import mast as hm
        target = hm.BANDS[band]
        if target > 0.0:                    # WP24 raise gate: standing (trunk <= 0.05 m/s)
            self.drv.gait.release()
            def still(d):
                s = d.bus.values.get("stability") or {}
                return (float(np.linalg.norm(d.d.qvel[0:2])) < 0.03 and not s.get("walking")
                        and len(s.get("stance") or []) >= 6)
            self.drv.run_until(still, 5.0, chunk_s=0.1)
            self.drv.wait(0.3)
        t0 = self.drv.t
        self.drv.deliver(hm.CMD_TOPIC, {"source": "preset", "band": band, "reason": "wp9_tour"})
        ok = self.drv.run_until(lambda d: abs(float((d.bus.values.get("mast") or {}).get("ext_m") or 0.0) - target)
                                <= 0.02, timeout_s, chunk_s=0.1)
        st = self.drv.bus.values.get("mast") or {}
        return ok, self.drv.t - t0, str(st.get("last_decision") or st.get("reason") or "")

    def alley_point(self, tree_id: str, side: Optional[str] = None) -> Tuple[float, float]:
        e = next(t for t in self.field["trees"] if t["tree_id"] == tree_id)
        g = e["geo"]
        here = self.drv.trunk_xy()
        if side is None:
            side = "south" if here[1] <= float(g["y"]) else "north"
        return float(g["x"]), row_alley_y(self.field, e["row_id"], side)

    # ---------------------------------------------------------------- step dispatch
    def run_step(self, step: TourStep, ready: Any, runner: TourRunner) -> StepResult:
        self.log(f"[{self.t:7.1f}] step {step.id}")
        n_ack, n_stop, n_al = len(self.scripted_acks), len(self.stop_log), len(self.alerts)
        self._ctl_runner = runner
        try:
            res = self._dispatch(step, ready, runner)
        except StepAborted:
            # the operator's abort mid-step: stop where the robot is (the runner hold, a sit-down) and end the step
            self._ctl_runner = None
            self._watch = None
            self._hold_ref = None
            self.drv.gait.release()
            if not self.drv.holding:
                self.drv.enter_hold("operator_abort")
            self.log(f"[{self.t:7.1f}] step {step.id} aborted by the operator")
            raise
        finally:
            self._ctl_runner = None
        res.scripted_delays += self.scripted_acks[n_ack:]
        res.events += [dict(e, event="protective_latch") for e in self.stop_log[n_stop:]]
        res.alerts += self.alerts[n_al:]
        if self.drv.holding and self._own_hold:
            res.notes.append("protective stop still latched at the end of the step")
        return res

    def _dispatch(self, step: TourStep, ready: Any, runner: TourRunner) -> StepResult:
        if step.scripted:
            res = StepResult()
            if step.id.startswith("s10a"):
                return self._gust(step, res)
            stowed_leg = any(g.get("kind") == "walk" and g.get("transit") and float(g.get("ext_m") or 0.0) <= 0.0
                             for g in step.motion)
            if stowed_leg:
                return self._stowed_transit(step, res)
            self.motion_transits(step, res)
            if step.t_s:
                self.wait(step.t_s)
                res.scripted_delays.append({"what": step.title, "s": step.t_s})
            return res
        p = step.preset
        params = dict(ready.params) if ready is not None and getattr(ready, "params", None) else dict(step.params)
        params.update({k: v for k, v in step.params.items() if k not in params})
        if p == "scan":
            return self._scan(step, params, ready)
        if p == "tree360":
            return self._tree360(step, params, ready, runner)
        if p == "leaf_scan":
            return self._leaf(step, params)
        if p == "trunk_scan":
            return self._trunk(step, params)
        if p == "fruit_scout":
            return self._scout(step, params)
        if p in HARVEST_PRESETS:
            return self._harvest(step, params)
        if p in PRUNE_PRESETS:
            return self._prune(step, params)
        if p == "estop_recovery":
            return self._estop(step, params)
        if p == "dock_unload":
            return self._dock(step, params, runner)
        res = StepResult(status="skipped", reason=f"preset {p} has no tour runner in WP9")
        return res

    # ---------------------------------------------------------------- SCAN
    def _scan_views(self, plan: dict, crown_lens: Optional[float]):
        import numpy as np
        from atharv_nandi.core.perception import pest_detect as pdet
        cfg = pdet.load_config()
        p = np.asarray(self.drv.trunk_xy(), float)
        h = self.drv.heading()
        tree = np.asarray(plan["pos"], float)
        H = float(plan["entry"].get("height_m") or 2.2)
        ring = cfg["cameras"]["ring"]
        views = [pdet.look_at([p[0], p[1], 0.98], [tree[0], tree[1], 0.55], fovy_deg=ring["fovy_deg"],
                              width=ring["width"], height=ring["height"], camera="ring", band="low")]
        if crown_lens is not None:
            cr = cfg["cameras"]["crown"]
            lens = [p[0] - 0.30 * math.cos(h), p[1] - 0.30 * math.sin(h), crown_lens]
            views.append(pdet.look_at(lens, [tree[0], tree[1], min(crown_lens - 0.25, 0.75 * H)],
                                      fovy_deg=cr["fovy_deg"], width=cr["width"], height=cr["height"],
                                      camera="crown_cam", band="mid"))
        return views

    def _scan_tree(self, plan: dict, crown_lens: Optional[float], threshold: float, det) -> dict:
        import numpy as np
        from atharv_nandi.core import records as rec_
        from atharv_nandi.core.perception import pest_detect as pdet
        from atharv_nandi.core.tasks import common as tc
        prof = self.profiles[plan["crop"]]
        views = self._scan_views(plan, crown_lens)
        leads = pdet.dedupe_leads(pdet.canopy_leads(plan, prof, views, det))
        truth = pdet.canopy_truth_ids(plan, prof, views)
        units = {c["class"]: c["unit"] for c in pdet.active_classes(prof, plan["season"])}
        truth_leads = [{"unit": units.get(c, "per_leaf"), "score": 1.0} for c in truth.values()]

        def seen(pt):
            q = np.asarray(pt, float)
            return any(v.in_frame(q) for v in views)

        fruit = [f for f in plan["truth"].get("fruit", []) if f.get("harvest_stage") and not f.get("bagged")
                 and seen(f["centre_world"])]
        prune = [x for x in plan["truth"].get("prune", []) if not x.get("keep") and x.get("in_band")
                 and seen(x["cut_world"])]
        flags = scan_flags(leads, len(fruit), len(prune), threshold)
        flags_truth = scan_flags(truth_leads, len(fruit), len(prune), 0.0)
        frag_p = pdet.record_fragment(plan, symptoms=leads, did=self.did.get(plan["tree_id"], []))
        e = plan["entry"]
        side = tc.side_of(plan["pos"][:2], self.drv.trunk_xy())
        frag = rec_.scan_fragment(e, side, [v.band for v in views], frag_p["symptoms"],
                                  {"height_m": e.get("height_m"), "width_m": 2.0 * float(e.get("canopy_r_m") or 0),
                                   "gap_pct": None, "lean_deg": 0.0}, flags, len(views),
                                  detector=frag_p["detector"], assisted=["truth_pose_scan_views",
                                                                         "lidar_profile_truth"])
        return {"frag": frag, "flags": flags, "truth_flags": flags_truth, "bands": [v.band for v in views],
                "seen": True, "leads": len(leads)}

    def _scan(self, step: TourStep, params: dict, ready) -> StepResult:
        from atharv_nandi.core.perception import pest_detect as pdet
        res = StepResult()
        rows = step.row_ids() or sorted({self.field_entry(t)["row_id"] for t in step.tree_ids()})
        crown_mid = params.get("crown_band", "mid") == "mid" and not (ready and ready.adjust.get("crown_band")
                                                                       == "stowed")
        burst = float(params.get("burst_s", 3.0))
        thr = float(params.get("triage_threshold", 0.5))
        det = pdet.SimTruthLeadDetector(pdet.load_config(), seed=self.seed)
        lat0 = self._laterals()
        out_trees, tri = [], {"tp": 0, "fp": 0, "fn": 0}
        scan_path = next((g["path"] for g in step.motion if g.get("kind") == "walk" and not g.get("transit")), None)
        self.motion_transits(step, res)
        for row in rows:
            trees = sorted((t for t in self.field["trees"] if t["row_id"] == row), key=lambda t: t["geo"]["x"])
            if step.tree_ids():
                trees = [t for t in trees if t["tree_id"] in step.tree_ids()]
            if scan_path:
                y = float(scan_path[0][1])
                x0, x1 = float(scan_path[0][0]), float(scan_path[-1][0])
            else:
                y = float(step.extra.get("alley_y", row_alley_y(self.field, row, "south")))
                x0, x1 = float(trees[0]["geo"]["x"]) - 1.2, float(trees[-1]["geo"]["x"]) + 1.0
            if not scan_path:
                self.route_to((x0, y), res)
            lens = None
            if crown_mid:
                ok, dt_, why = self.mast("mid")
                if ok:
                    lens = float((self.drv.bus.values.get("mast") or {}).get("lens_z") or 1.70)
                else:
                    res.notes.append(f"crown_gate: raise refused ({why}); SCAN runs stowed")
                    self.mast("stow")
            vx = CROWN_UP_VX if lens is not None else TRANSIT_VX
            for t in trees:
                tx = float(t["geo"]["x"])
                if not self._walk_to((tx, y), vx, tol=0.12):
                    res.notes.append(f"SCAN: abeam {t['tree_id']} not reached by walking; teleported")
                    self._teleport((tx, y), 0.0 if x1 >= x0 else 180.0)
                    self._assisted.add("transit_teleport_fallback")
                self.drv.stand(burst)
                plan = self.plans[t["tree_id"]]
                s = self._scan_tree(plan, lens, thr, det)
                self.did.setdefault(t["tree_id"], []).append("scan")
                self.frames_t[t["tree_id"]] = self.t
                res.tree_fragments.append(("scan", t["tree_id"], s["frag"]))
                sc = triage_score(s["flags"], s["truth_flags"])
                for k in tri:
                    tri[k] += sc[k]
                out_trees.append({"tree_id": t["tree_id"], "seen": s["seen"], "bands": s["bands"],
                                  "flags": s["flags"], "truth_flags": s["truth_flags"]})
            self._walk_to((x1, y), vx, tol=0.15)
            if lens is not None:
                ok, dt_, why = self.mast("stow")
                if ok:
                    res.notes.append(f"crown stowed at the row end in {dt_:.1f} s")
                else:
                    # the next transit would walk crown-up at 0.18 m/s: a halt, never a silent pass
                    res.status, res.reason = "halted", f"crown not stowed at the {row} row end ({why})"
                    res.notes.append(res.reason)
                    res.rows_visited.append(row)
                    break
            res.rows_visited.append(row)
        lat = self._laterals()
        res.kpi = {"trees": out_trees, "triage": tri,
                   "laterals_stepped": None if lat is None or lat0 is None else lat - lat0,
                   "laterals_note": "" if lat is not None else "no lateral counter in this world"}
        return res

    def _laterals(self) -> Optional[int]:
        w = self.drv.bus.values.get("world") or {}
        c = (w.get("laterals") or {}).get("count") if isinstance(w, dict) else None
        return int(c) if isinstance(c, (int, float)) else None

    def field_entry(self, tree_id: str) -> dict:
        return next(t for t in self.field["trees"] if t["tree_id"] == tree_id)

    # ---------------------------------------------------------------- TREE 360
    def _tree360(self, step: TourStep, params: dict, ready, runner: TourRunner) -> StepResult:
        from atharv_nandi.sim.hooks import tree360 as h3
        res = StepResult()
        tid = step.tree_ids()[0]
        if not self.motion_transits(step, res):
            self.route_to(self.alley_point(tid), res)
        sim = {"entry": "walk" if self.transit_mode == "walk" else "teleport", "capture": self.capture,
               "mp4": False, "out_root": os.path.join(runner.run_dir, "tree360"), "run_id": runner.run_id}
        keep = {k: v for k, v in params.items() if k in ("arc_mode", "standoff_m", "stations", "bands", "crown_walk",
                                                        "settle_s", "capture_s", "v_mps")}
        if keep.get("crown_walk") == "snake_mid" and "high" not in (keep.get("bands") or ["high"]):
            keep["crown_walk"] = "snake"          # no high band: snake walks every arc at mid (= snake_mid)
            res.notes.append("crown_walk snake_mid with bands low+mid -> WP25 snake (arcs at mid)")
        elif keep.get("crown_walk") == "snake_mid":
            # WP27 adjusts TREE 360 to snake_mid (arcs at mid, mid -> high -> mid standing) when the high-band head
            # does not clear alley_clear_height_m; WP25 has snake | stop_and_go only, so the arcs are walked stowed
            # (stop_and_go, OS 2.3 rule 7: slower, never higher than allowed). Change request WP25: snake_mid.
            keep["crown_walk"] = "stop_and_go"
            res.notes.append("crown_walk snake_mid -> stop_and_go (WP25 has no snake_mid; CR to WP25)")
        os.makedirs(sim["out_root"], exist_ok=True)
        actor =STEP_ACTORS.get(step.id) or ((step.extra.get("person_actor"), "person")
                                             if step.extra.get("person_actor") else None)
        try:
            run = h3.start(self.drv.bus, self.m, self.drv.d, tid, {**keep, **sim}, gait=self.drv.gait)
        except ValueError as exc:
            res.status, res.reason = "refused", f"tree360: {exc}"
            return res
        fired, flips = False, []
        t_end = self.drv.t + 480.0
        while not run.finished and self.drv.t < t_end:
            self.drv.wait(0.5)
            prog = run.progress()
            if actor and not fired and prog.get("phase") in ("station", "arc") and int(prog.get("station") or 0) >= 1:
                self.drv.deliver("/nandi/world/cmd", {"actor": actor[0], "action": "start"})
                fired = True
                res.events.append({"t": self.t, "event": "actor_start", "actor": actor[0], "class": actor[1]})
            if fired and (not flips or flips[-1] != run.paused):
                flips.append(run.paused)
                if flips[-2:] == [True, False] or (len(flips) >= 2 and flips[-1] is False):
                    self.drv.deliver("/nandi/world/cmd", {"actor": actor[0], "action": "park"})
        if not run.finished:
            run.abort("wp9_timeout")
            self.drv.run_until(lambda _: run.finished, 30.0, chunk_s=0.5)
        out = self.drv.bus.values.get(h3.RESULT_KEY) or run.result or {}
        k = out.get("kpi") or {}
        res.radius_reentries = int(k.get("reentries") or 0)
        if k.get("person_unmasked_px") is not None:
            self._safety["unmasked_person_px"] += int(k["person_unmasked_px"])
            self._safety["person_pixels_measured"] = True
        if fired and actor:
            self.drv.deliver("/nandi/world/cmd", {"actor": actor[0], "action": "park"})
        if out.get("record"):
            res.tree_fragments.append(("tree360", tid, out["record"]))
        self.did.setdefault(tid, []).append("tree360")
        self.frames_t[tid] = self.t
        res.kpi = {"result": {**{k2: out.get(k2) for k2 in ("status", "mode", "crown_walk", "coverage", "kpi",
                                                            "coverage_pass", "abort_reason")},
                              "bands_planned": list(keep.get("bands") or ["low", "mid", "high"])}}
        for _ in range(int(k.get("pauses") or 0)):
            res.alerts.append({"id": "orbit_pause", "t": self.t})      # person event: aggregated, no place
        res.assisted += ["station_teleport"] if sim["entry"] == "teleport" else []
        if out.get("status") != "done":
            res.status, res.reason = "halted", f"tree360 {out.get('status')}: {out.get('abort_reason')}"
        else:
            res.notes.append(f"TREE 360 {tid}: mode {out.get('mode')}, orbit {k.get('orbit_s')} s, "
                             f"pauses {k.get('pauses')}, reentries {k.get('reentries')}")
        e = self.field_entry(tid)
        res.rows_visited.append(e["row_id"])
        return res

    # ---------------------------------------------------------------- LEAF / TRUNK / SCOUT (WP26 planners)
    def _dwell_here(self, step: TourStep, tid: str, res: StepResult):
        if self.motion_transits(step, res):
            return
        at = next((g.get("at") for g in step.motion if g.get("kind") == "dwell" and g.get("at")), None)
        here = self.drv.trunk_xy()
        if at is not None and math.hypot(float(at[0]) - here[0], float(at[1]) - here[1]) < 0.3:
            return                        # already at the scripted dwell point (WP27 motion)
        self.route_to(tuple(at) if at is not None and not self.in_keepout(at, 0.0) else self.alley_point(tid), res)

    def _leaf(self, step: TourStep, params: dict) -> StepResult:
        from atharv_nandi.core.perception import leaf_scan as ls, pest_detect as pdet, trunk_scan as ts
        res = StepResult()
        tid = step.tree_ids()[0]
        plan = self.plans[tid]
        prof = self.profiles[plan["crop"]]
        self._dwell_here(step, tid, res)
        r = ls.run_leaf_scan(plan, prof, n_leaves=params.get("n_leaves"), demo=True)
        times = [float(x.get("time_s") or 0.0) for x in r.records]
        dwell = 15.0 + 5.0 + sum(times)              # MP 3 positioning + WORK entry + per-leaf time model
        self.drv.stand(dwell)
        tr = ts.run_trunk_scan(plan, prof)
        leads = pdet.dedupe_leads(list(r.leads) + list(tr.symptoms))
        self.did.setdefault(tid, []).extend(["leaf_scan", "trunk_scan"])
        frag = pdet.record_fragment(plan, symptoms=leads, leaf=r.leaf_block, trunk=tr.trunk, did=self.did[tid])
        res.tree_fragments.append(("leaf_scan", tid, frag))
        res.marks += [m for m in (r.marks or []) if isinstance(m, dict) and m.get("schema") == "nandi.mark/1"]
        res.kpi = {"leaf": r.leaf_block, "trunk": {"azimuths_imaged": tr.azimuths_imaged, "mode": tr.mode}}
        res.assisted += ["leaf_scan_kinematic", "simtruth_leads"]
        res.notes.append(f"LEAF SCAN {tid}: {len(r.records)} views, stand {dwell:.0f} s (WP26 time model, arm "
                         f"not moved: kinematic), trunk band {tr.mode} {tr.azimuths_imaged} az")
        return res

    def _trunk(self, step: TourStep, params: dict) -> StepResult:
        from atharv_nandi.core.perception import pest_detect as pdet, trunk_scan as ts
        res = StepResult()
        tid = step.tree_ids()[0]
        plan = self.plans[tid]
        self._dwell_here(step, tid, res)
        tr = ts.run_trunk_scan(plan, self.profiles[plan["crop"]])
        self.drv.stand(float(getattr(tr, "time_s", 0.0) or 20.0))
        self.did.setdefault(tid, []).append("trunk_scan")
        res.tree_fragments.append(("trunk_scan", tid, pdet.record_fragment(plan, symptoms=tr.symptoms,
                                                                           trunk=tr.trunk, did=self.did[tid])))
        res.kpi = {"trunk": {"azimuths_imaged": tr.azimuths_imaged, "mode": tr.mode}}
        res.assisted.append("trunk_scan_kinematic")
        return res

    def _scout(self, step: TourStep, params: dict) -> StepResult:
        from atharv_nandi.core.perception import fruit_track as ft, pest_detect as pdet
        res = StepResult()
        tid = step.tree_ids()[0]
        plan = self.plans[tid]
        self._dwell_here(step, tid, res)
        age = None if tid not in self.frames_t else max(0.0, self.t - self.frames_t[tid]) / 3600.0
        src = params.get("source", "reuse_tree360")
        r = ft.scout(plan, self.profiles[plan["crop"]], self.today, frames_age_h=age, crown_gate_ok=True,
                     source=src, seed=self.seed)
        self.drv.stand(10.0)
        marks = [m for m in (r.marks or []) if isinstance(m, dict) and m.get("schema") == "nandi.mark/1"]
        self.did.setdefault(tid, []).append("fruit_scout")
        res.tree_fragments.append(("fruit_scout", tid, pdet.record_fragment(plan, fruit=r.fruit, marks_n=len(marks),
                                                                            did=self.did[tid])))
        res.marks += marks
        res.kpi = {"fruit": r.fruit, "assisted": r.assisted, "marks_planned": len(marks)}
        res.assisted.append("simtruth_fruit")
        return res

    # ---------------------------------------------------------------- HARVEST / PRUNE (WP8)
    def _station(self, tg):
        from atharv_nandi.core.tasks import sim_executor as sx
        if tg.tool == "T2" and tg.station is None:
            tree = self.plans[tg.tree_id]["pos"]
            tg.station = sx.station_for(tg, tree)
            self._assisted.add("generated_station_t2")
        self._assisted.add("station_teleport")
        return self.tree_runner.default_station(tg)

    def _final(self, results: Sequence[dict]) -> List[dict]:
        last: Dict[str, dict] = {}
        for r in results:
            last[r["target_id"]] = r
        return list(last.values())

    def _work_tree(self, step: TourStep, job, kind: str, crop_mass: Optional[float] = None) -> StepResult:
        res = StepResult(harvest_kind=kind)
        tid = job.tree_id
        if not self.motion_transits(step, res):
            self.route_to(self.alley_point(tid), res)
        actor = STEP_ACTORS.get(step.id)
        if actor:
            self.drv.deliver("/nandi/world/cmd", {"actor": actor[0], "action": "start"})
            res.events.append({"t": self.t, "event": "actor_start", "actor": actor[0], "class": actor[1]})
        if crop_mass is not None:
            self.tree_runner.fruit_mass_kg[job.crop] = float(crop_mass)
            self._assisted.add("fruit_mass_scaled")
        out = self.tree_runner.run_tree(job)
        out = self._resume_after_stop(job, out, res)
        if actor:
            self.drv.deliver("/nandi/world/cmd", {"actor": actor[0], "action": "park"})
        final = self._final(out["results"])
        by_target = {t.target_id: t for t in job.targets}
        for r in final:
            v1 = dict(r.get("record") or {})
            if not v1:
                continue
            v1.setdefault("tree_id", tid)
            v1.setdefault("crop", job.crop)
            # nandi.harvest/2 stage and mass: WP8's v1 item has neither; the target carries the registry stage and
            # the fruit mass (the papaya 1.8 kg override already applied by TreeRunner.assign)
            tg = by_target.get(v1.get("target_id") or r.get("target_id"))
            if tg is not None:
                v1.setdefault("stage", getattr(tg, "stage", None))
                m_kg = float(getattr(tg, "mass_kg", 0.0) or 0.0)
                if kind == "harvest" and m_kg > 0.0:
                    v1.setdefault("mass_kg", m_kg)
            res.harvest.append(v1)
        res.marks += out["marks"]
        res.skipped_targets = sum(1 for r in final if r.get("outcome") in ("skipped", "failed", "halted"))
        res.kpi = {"results": [{"target_id": r["target_id"], "outcome": r.get("outcome"), "reason": r.get("reason"),
                                "crop": job.crop, "pedicel_error_mm": (r.get("record") or {}).get("pedicel_error_mm"),
                                "cycle_s": r.get("cycle_s")} for r in final],
                   "skipped": out.get("skipped"), "deferred": out.get("deferred"),
                   "marks_planned": len(job.marks), "crop": job.crop, "error": out.get("error")}
        if job.crop == "papaya":
            v = [r.get("tcp_mps_by_state", {}) for r in out["results"]]
            sp = [float(x) for d_ in v for k_, x in (d_ or {}).items() if k_ in ("carry", "park", "place")
                  and isinstance(x, (int, float))]
            res.kpi["place_mps_max"] = max(sp) if sp else None
        if out.get("error"):
            res.status, res.reason = "halted", out["error"]
        elif (out.get("loop") or {}).get("halted"):
            res.status, res.reason = "halted", f"target loop halted: {(out.get('loop') or {}).get('halt_reason')}"
        tray = self._tray_per_tree()
        res.kg_by_tree = {tid: tray.get(tid, {}).get("kg", 0.0)} if tid in tray else {}
        res.rows_visited.append(self.field_entry(tid)["row_id"])
        res.notes.append(f"{step.preset} {tid}: " + ", ".join(f"{r['target_id']}:{r.get('outcome')}:{r.get('sim_s')}s:att{r.get('attempts')}"
                                                              f"{':' + r['reason'] if r.get('reason') else ''}"
                                                              for r in final) + f"; marks {len(out['marks'])}")
        return res

    def _resume_after_stop(self, job, out: dict, res: StepResult, wait_s: float = 60.0) -> dict:
        """A protective stop (dog, person) halts the WP8 target loop (``halt_safety``, the hold reads as e-stop).
        Once the operator has cleared the stop, the interrupted targets run again, never mid-cut (MP 8 E-STOP
        RECOVERY "resume from the last completed tree, never mid-cut"; OS 1.2 step 6 "then resume")."""
        halted = [r["target_id"] for r in self._final(out["results"]) if r.get("outcome") == "halted"]
        if not halted:
            return out
        t0 = self.drv.t
        self.drv.run_until(lambda d: not d.holding, wait_s, chunk_s=0.2)
        if self.drv.holding:
            res.notes.append(f"stop not cleared within {wait_s:.0f} s: {len(halted)} target(s) left halted")
            return out
        self.drv.stand(2.0)               # settle after the hold (the stability interlock reads the live margin)
        res.notes.append(f"resumed {len(halted)} target(s) after the protective stop cleared "
                         f"({self.drv.t - t0:.1f} s)")
        job2 = copy.copy(job)
        job2.targets = [t for t in job.targets if t.target_id in halted]
        job2.marks, job2.skipped, job2.deferred = [], [], []
        out2 = self.tree_runner.run_tree(job2)
        out = dict(out)
        out["results"] = list(out["results"]) + list(out2["results"])
        out["marks"] = list(out["marks"]) + list(out2.get("marks") or [])
        out["loop"] = out2.get("loop")
        return out

    def _tray_per_tree(self) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for it in self.ex.tc.items.values():
            if it.get("recycled") or it["state"] not in ("seated", "moving"):
                continue
            p = out.setdefault(it["tree_id"], {"n": 0, "kg": 0.0})
            p["n"] += 1
            p["kg"] = round(p["kg"] + float(it["mass_kg"]), 4)
        return out

    def _harvest(self, step: TourStep, params: dict) -> StepResult:
        from atharv_nandi.core.tasks import tree_task
        tid = step.tree_ids()[0]
        plan = self.plans[tid]
        prof = self.profiles[plan["crop"]]
        method = params.get("method")
        if method in ("mark",) or plan["crop"] == "banana":
            method = None
        job = tree_task.harvest_job(plan, prof, method=method, max_items=params.get("max_items"),
                                    stations=self.stations, exclude=self.rt.pool.harvested)
        mass = params.get("fruit_kg") if plan["crop"] == "papaya" else None
        return self._work_tree(step, job, "harvest", mass)

    def _prune(self, step: TourStep, params: dict) -> StepResult:
        from atharv_nandi.core.tasks import banana_tidy, tree_task
        tid = step.tree_ids()[0]
        plan = self.plans[tid]
        prof = self.profiles[plan["crop"]]
        jobs = [j for j in (params.get("jobs") or []) if not j.startswith("banana_")] or None
        job = tree_task.prune_job(plan, prof, jobs=jobs, stations=self.stations)
        if plan["crop"] == "banana":
            job.marks.extend(m for m in banana_tidy.banana_marks(plan) if m["kind"] == "bunch")
        res = self._work_tree(step, job, "prune")
        # not audited: the WP8 planner never plans keep / green targets, but no cut-event audit counts them, so the
        # O-13 / O-14 rows are report-only (None), never a passed gate
        res.kpi["keep_cuts"] = None
        res.kpi["green_cuts"] = None
        res.notes.append("keep-wood / green cuts: not audited (the WP8 planner never plans keep or green targets)")
        return res

    def _stowed_transit(self, step: TourStep, res: StepResult) -> StepResult:
        """A scripted transit with the crown stowed (OS 1.2 step 1 b, the 5 deg headland strip): the crown is stowed
        first and watched over the walk; it must stay <= 0.03 m (the step halts otherwise). The lowest predicted
        tripod margin while walking is reported (``tripod_min_m``; OS 1.3 O-17 gates it with the crown up)."""
        ext = float((self.drv.bus.values.get("mast") or {}).get("ext_m") or 0.0)
        if ext > 0.02:
            ok, _, why = self.mast("stow")
            if not ok:
                res.status, res.reason = "halted", f"crown not stowed before the transit ({why})"
                return res
        self._watch = {"ext_max": 0.0, "tripod_min": None}
        try:
            self.motion_transits(step, res)
            if step.t_s:
                self.wait(step.t_s)
                res.scripted_delays.append({"what": step.title, "s": step.t_s})
        finally:
            w, self._watch = self._watch, None
        res.kpi = {"crown_ext_max_m": round(w["ext_max"], 3),
                   "tripod_min_m": None if w["tripod_min"] is None else round(w["tripod_min"], 3)}
        res.notes.append(f"stowed transit: crown ext max {w['ext_max']:.3f} m, predicted tripod margin min "
                         f"{res.kpi['tripod_min_m']} m")
        if w["ext_max"] > 0.03:
            res.status, res.reason = "halted", f"crown not stowed on the transit (ext {w['ext_max']:.3f} m)"
        return res

    def refusal_probe(self, step: TourStep, runner: TourRunner) -> Tuple[dict, List[str]]:
        """OS 1.2 step 10 b on the real world: walk the step's transit motion to the patch edge, then read the
        terrain under every ``expect_refused`` event point (the world's heightfield, ``terrain_orchard``; the robot
        does not step onto a 10 deg patch) and the robot's state; the Ready check runs again on that instead of the
        step's scripted ``ctx``. A real WP24 raise on the patch is not attempted (assisted ``terrain_truth_probe``)."""
        from atharv_nandi.sim.world import terrain_orchard as to
        res = StepResult()
        self._ctl_runner = runner
        try:
            self.motion_transits(step, res)
        except StepAborted:
            self.drv.gait.release()
            raise
        finally:
            self._ctl_runner = None
        grid = to.grid_for(self.field, self.seed)
        pts = [g["at"] for g in step.motion if g.get("kind") == "event" and g.get("at")]
        if not pts:
            pts = [tuple(float(v) for v in self.drv.trunk_xy())]
        slopes = [grid.slope_deg_at(float(p[0]), float(p[1])) for p in pts]
        measured = {k: v for k, v in self.context(step).items() if k in ("ext_m", "estop", "tool_mounted")}
        measured["cross_slope_deg"] = round(max(slopes), 2)
        self._assisted.add("terrain_truth_probe")
        notes = res.notes + [f"refusal probe: terrain slope {[round(s, 1) for s in slopes]} deg at {pts} "
                             f"(walked to {tuple(round(float(v), 2) for v in self.drv.trunk_xy())})"]
        return measured, notes

    # ---------------------------------------------------------------- RECOVERY tour part
    def _gust(self, step: TourStep, res: StepResult) -> StepResult:
        """OS 1.2 step 10 a: walk with the crown at mid, 11 m/s gust -> 6-foot stance <= 0.5 s, then the crown
        retracts (WP6 stance request, WP24 backstop retract)."""
        walks = [g for g in step.motion if g.get("kind") == "walk"]
        if walks:
            self.transit(walks[0]["path"], res)
        ok, _, why = self.mast("mid")
        if not ok:
            res.status, res.reason = "halted", f"crown raise refused before the gust: {why}"
            return res
        tgt = walks[1]["path"][-1] if len(walks) > 1 else (self.drv.trunk_xy()[0] - 1.5, self.drv.trunk_xy()[1])
        self._walk_to(tgt, CROWN_UP_VX, timeout_s=3.0)          # walking crown-mid when the gust arrives
        self.drv.gait.set_command(CROWN_UP_VX, 0.0, 0.0, hold=True)
        self.drv.deliver("/nandi/world/cmd", {"fire": "gust_11"})
        t0 = self.drv.t
        onset_rel, onset_how = self.gust_onset("gust_11")
        t_on = t0 + onset_rel              # O-18 clock: the gust's truth onset (not the anemometer's detection)
        stance = None
        stowed = None
        t_det = None
        while self.drv.t - t0 < 16.0:
            self.drv.wait(0.02)
            st = self.drv.bus.values.get("stability") or {}
            w = self.drv.bus.values.get("wind") or {}
            if t_det is None and (w.get("gust") or float(w.get("speed_mps") or 0.0) >= 10.0):
                t_det = self.drv.t
            if self.drv.t < t_on - 1e-9:
                continue
            # the 6-foot stance reached (the same test as the WP24 raise gate in mast()), not the stance request
            if stance is None and not st.get("walking") and len(st.get("stance") or []) >= 6:
                stance = self.drv.t - t_on
            if stance is not None and float((self.drv.bus.values.get("mast") or {}).get("ext_m") or 1) <= 0.02:
                stowed = self.drv.t - t_on
                break
        self.drv.gait.release()
        self.mast("stow")
        self.drv.stand(2.0)
        self._safety["gust_stance_s"] = None if stance is None else round(stance, 2)
        self._safety["crown_stow_s"] = None if stowed is None else round(stowed, 2)
        det = None if t_det is None else round(t_det - t_on, 2)
        res.kpi = {"gust_stance_s": stance, "crown_stow_s": stowed, "gust_detect_s": det}
        res.notes.append(f"gust: 6-foot stance {stance} s, crown stowed {stowed} s after the gust onset "
                         f"({onset_how}, {onset_rel:.2f} s after the command); anemometer detection {det} s")
        if stance is None or stowed is None:
            res.status, res.reason = "halted", "gust: no 6-foot stance / no retract"
        elif stance > TIMED_SAFETY["gust_stance_s"][0] or stowed > TIMED_SAFETY["crown_stow_s"][0]:
            # measured: the timed safety gate fails the tour (evaluate); the step itself ran to its end
            res.reason = f"O-18 over the gate: stance {stance:.2f} s (<= 0.5) / stow {stowed:.2f} s (<= 8.5)"
        return res

    def gust_onset(self, event_id: str) -> Tuple[float, str]:
        """Seconds from the fire command to the truth onset of the event's first gust: WP7's criterion (the true
        scripted speed reaches its running mean + 5 m/s, ``wind_model.gust_criterion_onset``); the scripted gust
        start ``t0`` if that cannot be computed."""
        ev = next((e for e in self.field.get("events") or [] if e.get("id") == event_id), None) or {}
        script = ev.get("script") or {}
        gusts = script.get("gusts") or []
        t_start = float(gusts[0].get("t0", 0.0)) if gusts else 0.0
        try:
            from atharv_nandi.core import wind_model as wm
            ws = wm.WindScript.from_dict(script)
            g = gusts[0]
            t1 = t_start + float(g.get("rise_s", 0.0)) + float(g.get("hold_s", 0.0)) + 1.0
            on = wm.gust_criterion_onset(ws, 0.0, t1)
            if on is not None:
                return float(on), "WP7 truth gust criterion"
        except Exception:  # noqa: BLE001
            pass
        return t_start, "scripted gust start"

    def _estop(self, step: TourStep, params: dict) -> StepResult:
        """OS 1.2 step 10 e: e-stop (the runner hold the supervisor and the handheld use), stationary check, then
        the checklist with the local and the operator confirmation (scripted fixed 5 s each, on the run card); no
        motion without both; stand up."""
        import numpy as np
        res = StepResult()
        self.motion_transits(step, res)
        confirm = float(params.get("confirm_s", 5.0))
        self.drv.gait.set_command(TRANSIT_VX, 0.0, 0.0, hold=True)
        self.drv.wait(1.0)
        self.drv.enter_hold("estop")
        t0 = self.drv.t
        stopped = self.drv.run_until(lambda d: float(np.linalg.norm(d.d.qvel[0:2])) < 0.02, 3.0, chunk_s=0.02)
        # not stationary within the 3 s window: the elapsed window is recorded (it fails the <= 0.5 s gate), never
        # None (a None row would drop out of the card)
        self._safety["estop_stationary_s"] = round(self.drv.t - t0, 3)
        if not stopped:
            res.status, res.reason = "halted", f"e-stop: not stationary within {self.drv.t - t0:.1f} s"
        elif self._safety["estop_stationary_s"] > TIMED_SAFETY["estop_stationary_s"][0]:
            # measured: the timed safety gate fails the tour (evaluate); the step itself ran to its end
            res.reason = (f"e-stop: stationary after {self._safety['estop_stationary_s']} s "
                          f"(gate {TIMED_SAFETY['estop_stationary_s'][0]} s)")
        self.drv.wait(0.3)
        self._hold_ref = self.drv.d.qpos[0:2].copy()
        self.drv.gait.set_command(1.0, 0.0, 0.0, hold=True)      # a stray drive command during the hold
        self.drv.wait(confirm)
        res.scripted_delays.append({"what": "e-stop local confirmation", "s": confirm})
        self.drv.wait(confirm)
        res.scripted_delays.append({"what": "e-stop operator confirmation", "s": confirm})
        self._hold_ref = None
        self.drv.leave_hold()
        self.drv.gait.release()
        self.mast("stow")
        self.drv.stand(5.0)
        res.kpi = {"estop_stationary_s": self._safety["estop_stationary_s"]}
        res.assisted.append("estop_via_driver_hold")
        return res

    # ---------------------------------------------------------------- DOCK / UNLOAD (WP13)
    def _dock(self, step: TourStep, params: dict, runner: TourRunner) -> StepResult:
        from atharv_nandi.core import dock_logic as dl
        res = StepResult(harvest_kind="unload")
        if not self.motion_transits(step, res):
            self.route_to((1.8, 0.5), res)
        self.mast("stow")
        os.makedirs(runner.run_dir, exist_ok=True)
        c, _, _ = self.ex.tc.crate_frame(self.drv.d)
        bat = dl.platform_battery(self.m)
        n_items = sum(v["n"] for v in self._tray_per_tree().values())
        sheets_out: Dict[str, Any] = {}

        def jobsheets(rows):
            sheets = runner.job_sheets(list(rows) or None)
            missing = js.missing_marks(runner.marks, sheets)
            sheets_out.update(sheets)
            return {"status": "generated", "rows": sorted(sheets), "n_marks": sum(s["n_marks"] for s in
                                                                               sheets.values()),
                    "missing": missing, "lang": list(js.LANGS)}

        s = dl.DockUnloadSession(dl.DockGeometry.from_model(self.m), dl.crate_park_geometry(c[:2]), battery=bat,
                                 run_dir=runner.run_dir, meta={"run_id": runner.run_id,
                                                               "device_id": runner.ident["device_id"]},
                                 jobsheets=jobsheets, rows_visited=list(runner.rows_visited))
        dr = dl.DockRunner(self.drv, s, unload=self.ex.tc)
        dr.start("operator")
        self.drv.run_until(lambda _: s.step in ("done", "assist"), 240.0, chunk_s=0.1)
        out = s.outputs
        cr = out.get("crate") or {}
        res.kg_by_tree = {k: float(v.get("crate_kg", 0.0)) for k, v in (cr.get("per_tree") or {}).items()}
        res.kpi = {"dock": out.get("dock"), "crate": {**cr, "impact_mps_max": cr.get("max_impact_mps")},
                   "expected_items": n_items, "jobsheets": out.get("jobsheets"), "charge": out.get("charge"),
                   "sync": out.get("sync")}
        res.assisted += ["kinematic_transfer", "dock_truth_pose"]
        if s.step != "done":
            res.status, res.reason = "halted", f"dock/unload {s.step}: {s.reason}"
        res.notes.append(f"DOCK/UNLOAD: crate {cr.get('n')}/{n_items}, dock "
                         f"{((out.get('dock') or {}).get('final') or {})}, jobsheets {(out.get('jobsheets') or {}).get('rows')}")
        return res

    def close(self):
        self.drv.close()


# ====================================================================== CLI
def run_tour(program: str = "tour_lite", seed: int = 7, steps: Optional[Sequence[str]] = None,
             runs_dir: Optional[str] = None, transit: str = "walk", capture: str = "geometry",
             log: Optional[Callable[[str], None]] = None, run_id: Optional[str] = None,
             publish: Optional[Callable[[str, Any], None]] = None, actors: bool = True) -> dict:
    """Headless tour on a fresh SimDriver world (the ``tour_lite`` / ``tour`` tests and ``/nandi/demo/run``)."""
    prog = load_program(program).select(steps)
    be = SimBackend(prog.world, seed, transit=transit, capture=capture, log=log, actors=actors)
    try:
        runner = TourRunner(prog, be, seed=seed, runs_dir=runs_dir, run_id=run_id, publish=publish)
        return runner.run()
    finally:
        be.close()


def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Atharv Nandi orchard tour (WP9), headless on SimDriver")
    ap.add_argument("--program", default="tour_lite", help="tour_lite | orchard_tour | path to a mission yaml")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", default="", help="comma-separated step ids (default: all)")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--transit", default="walk", choices=["walk", "teleport"])
    ap.add_argument("--capture", default="geometry", choices=["geometry", "gl", "auto"])
    ap.add_argument("--no-actors", action="store_true", help="NANDI_ACTORS=0 (segment runs whose actor step is not in the selection)")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    res = run_tour(a.program, a.seed, [s for s in a.steps.split(",") if s] or None, a.runs_dir, a.transit,
                   a.capture, log=lambda s: print(s, flush=True), run_id=a.run_id, actors=not a.no_actors)
    t = res["tour"]
    for r in res["steps"]:
        print(f"  {r['id']:<22} {r['status']:<8} {r['sim_s']:>7.1f} s  {r['reason']}")
        for n in r.get("notes") or []:
            print(f"      - {n[:300]}")
    print(json.dumps({k: t[k] for k in ("pass", "fail_reasons", "sim_s", "gate_s", "time_ok", "wall_s", "safety")},
                     default=str))
    print("violations:", res["violations"][:10])
    print("card failed gates:", res["card"]["failed_gates"])
    return 0 if t["pass"] else 1


if __name__ == "__main__":     # pragma: no cover
    import sys as _sys
    _sys.exit(main())
