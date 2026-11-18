"""Collision-free joint-space planning for the 5-DOF dorsal arm (WPCAP).

Backends, same interface and same validity checker (``CollisionScene``):

* ``ompl``        OMPL 2.0.1 Python bindings (PyPI wheel ``ompl==2.0.1``: manylinux2014 aarch64 / x86_64,
                  cp310-cp313; installed in the sim image by docker/Dockerfile). RRT-Connect in a 5-D
                  ``RealVectorStateSpace`` bounded by the arm joint limits (OMPL's own simplifier is not used:
                  its B-spline pass adds states; both backends share the shortcutter below).
* ``rrtconnect``  pure-Python / NumPy RRT-Connect (Kuffner and LaValle 2000) with random shortcutting. The
                  documented fallback when ``import ompl`` fails (e.g. the ROS-free CI venv on Python 3.11
                  without the wheel). Seeded, so a plan is reproducible.
* ``auto``        ``ompl`` when importable, else ``rrtconnect``.

Both produce a waypoint path that is then re-checked **independently** with ``CollisionScene.motion_valid`` at
``verify_step`` (default 0.005 rad / m max-norm per sub-step), so a result with ``ok`` True is collision-free at
that resolution regardless of the backend's own edge checker. ``time_parameterize`` turns the path into a
stop-at-waypoint trapezoidal trajectory under per-joint velocity / acceleration limits (``config/hal.yaml``),
which is what the HAL position interfaces (``controllers.hal``) execute.

Typical use (WP8 harvest integration)::

    planner = ArmPlanner(model)                        # once per process
    res = planner.plan(data.qpos, q_pregrasp, ignore=[target_fruit_body], data=data, model=model)
    if res.ok: for t, q in res.trajectory.samples(0.02): ...
"""
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import mujoco
import numpy as np

from atharv_nandi.core.planning.collision_scene import CollisionScene

try:  # optional: the ROS-free CI venv may not have the wheel (tests skip the OMPL backend then)
    from ompl import base as _ob, geometric as _og, util as _ou
    _OMPL_ERR = ""
except Exception as exc:  # pragma: no cover - depends on the environment
    _ob = _og = _ou = None
    _OMPL_ERR = f"{type(exc).__name__}: {exc}"

BACKENDS = ("auto", "ompl", "rrtconnect")
DEFAULT_TIME_LIMIT_S = 1.0
DEFAULT_EDGE_STEP = 0.02       # backend edge check: max joint change (rad or m) between checked states
DEFAULT_VERIFY_STEP = 0.005    # independent re-check of the final path
DEFAULT_RANGE = 0.35           # RRT extension step (joint-space Euclidean)
# Joint velocity / acceleration limits for time parameterization (gantry m/s, m/s^2; hinges rad/s, rad/s^2).
# ASSUMED starting values (servo class of docs/sim-fidelity-register.md); config/hal.yaml overrides them.
DEFAULT_VMAX = (0.15, 1.2, 1.0, 1.2, 1.5)
DEFAULT_AMAX = (0.5, 3.0, 2.5, 3.0, 4.0)
_OMPL_SEEDED = False


def ompl_available() -> bool:
    return _ob is not None


def ompl_import_error() -> str:
    return _OMPL_ERR


@dataclass
class Trajectory:
    """Piecewise trapezoidal joint trajectory: knot times ``t`` (s) and knot positions ``q`` (N x 5)."""
    t: np.ndarray
    q: np.ndarray
    vmax: np.ndarray
    amax: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def sample(self, t: float) -> np.ndarray:
        """Position at time t (stop-at-waypoint: each segment is a synchronized trapezoid / triangle)."""
        if t <= 0.0 or len(self.t) == 1:
            return self.q[0].copy()
        if t >= self.t[-1]:
            return self.q[-1].copy()
        i = int(np.searchsorted(self.t, t, side="right") - 1)
        T = self.t[i + 1] - self.t[i]
        s = _trapezoid_fraction((t - self.t[i]) / T if T > 0 else 1.0)
        return self.q[i] + (self.q[i + 1] - self.q[i]) * s

    def samples(self, dt: float) -> List[tuple]:
        n = max(1, int(math.ceil(self.duration / dt)))
        return [(k * self.duration / n, self.sample(k * self.duration / n)) for k in range(n + 1)]


# Normalised trapezoid with 1/4 accel, 1/2 cruise, 1/4 decel: s(u) for u in [0, 1]; peak speed 4/3 * avg.
_ACC_FRAC = 0.25


def _trapezoid_fraction(u: float) -> float:
    a = _ACC_FRAC
    vpk = 1.0 / (1.0 - a)                     # area = vpk * (1 - a) = 1
    if u < a:
        return 0.5 * vpk / a * u * u
    if u > 1.0 - a:
        r = 1.0 - u
        return 1.0 - 0.5 * vpk / a * r * r
    return 0.5 * vpk * a + vpk * (u - a)


def time_parameterize(path: np.ndarray, vmax: Sequence[float] = DEFAULT_VMAX,
                      amax: Sequence[float] = DEFAULT_AMAX) -> Trajectory:
    """Segment durations so that no joint exceeds vmax / amax with the fixed 1/4-1/2-1/4 trapezoid profile."""
    path = np.asarray(path, dtype=np.float64)
    vmax, amax = np.asarray(vmax, dtype=np.float64), np.asarray(amax, dtype=np.float64)
    a = _ACC_FRAC
    vpk = 1.0 / (1.0 - a)
    t = [0.0]
    for qa, qb in zip(path[:-1], path[1:]):
        dq = np.abs(qb - qa)
        # peak velocity vpk * dq / T <= vmax ; peak accel (vpk / a) * dq / T^2 <= amax
        T = max(float(np.max(vpk * dq / vmax)), float(np.max(np.sqrt(vpk / a * dq / amax))), 1e-3)
        t.append(t[-1] + T)
    return Trajectory(np.array(t), path.copy(), vmax, amax)


@dataclass
class PlanResult:
    ok: bool
    backend: str
    reason: str = ""                       # "" | start_invalid | goal_invalid | no_solution | verify_failed
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))
    plan_time_s: float = 0.0               # wall time of plan() incl. simplification and verification
    solve_time_s: float = 0.0              # backend search only
    n_checks: int = 0                      # validity checks (all phases)
    length: float = 0.0                    # joint-space path length
    min_clearance: Dict[str, float] = field(default_factory=dict)
    trajectory: Optional[Trajectory] = None
    contacts: List[dict] = field(default_factory=list)   # violations at the start / goal when refused

    def to_dict(self) -> dict:
        return {"ok": self.ok, "backend": self.backend, "reason": self.reason, "waypoints": len(self.path),
                "plan_time_s": round(self.plan_time_s, 4), "solve_time_s": round(self.solve_time_s, 4),
                "n_checks": self.n_checks, "length": round(self.length, 4),
                "min_clearance": {k: round(v, 4) for k, v in self.min_clearance.items()},
                "duration_s": round(self.trajectory.duration, 3) if self.trajectory else None,
                "contacts": self.contacts}


def _path_length(path: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0


class ArmPlanner:
    """Plans collision-free paths for the dorsal arm in a composed MuJoCo world."""

    def __init__(self, model: mujoco.MjModel, backend: str = "auto", margins: Optional[Dict[str, float]] = None,
                 time_limit_s: float = DEFAULT_TIME_LIMIT_S, edge_step: float = DEFAULT_EDGE_STEP,
                 verify_step: float = DEFAULT_VERIFY_STEP, rrt_range: float = DEFAULT_RANGE, seed: int = 7,
                 vmax: Sequence[float] = DEFAULT_VMAX, amax: Sequence[float] = DEFAULT_AMAX,
                 limits: Optional[np.ndarray] = None):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        if backend == "ompl" and not ompl_available():
            raise RuntimeError(f"OMPL backend requested but `import ompl` failed: {_OMPL_ERR}")
        self.backend = ("ompl" if ompl_available() else "rrtconnect") if backend == "auto" else backend
        self.scene = CollisionScene(model, margins=margins)
        self.limits = np.array(limits if limits is not None else self.scene.limits, dtype=np.float64)
        self.time_limit_s = float(time_limit_s)
        self.edge_step = float(edge_step)
        self.verify_step = float(verify_step)
        self.rrt_range = float(rrt_range)
        self._step = self.edge_step
        self.vmax, self.amax = np.asarray(vmax, dtype=float), np.asarray(amax, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

    @classmethod
    def from_config(cls, model: mujoco.MjModel, doc: Optional[dict] = None, **overrides) -> "ArmPlanner":
        """Planner from the ``planning`` block and the arm joint limits of config/hal.yaml."""
        from atharv_nandi.core.hal import interfaces as hal
        from atharv_nandi.core.planning.collision_scene import ARM_JOINTS
        doc = doc if doc is not None else hal.load_config()
        p = dict(doc.get("planning", {}) or {})
        vmax, amax = hal.joint_limits(doc, ARM_JOINTS)
        kw = {"backend": p.get("backend", "auto"), "margins": p.get("margins"),
              "time_limit_s": p.get("time_limit_s", DEFAULT_TIME_LIMIT_S), "edge_step": p.get("edge_step", DEFAULT_EDGE_STEP),
              "verify_step": p.get("verify_step", DEFAULT_VERIFY_STEP), "rrt_range": p.get("rrt_range", DEFAULT_RANGE),
              "seed": p.get("seed", 7), "vmax": vmax, "amax": amax}
        kw.update(overrides)
        return cls(model, **kw)

    # ------------------------------------------------------------------ public API
    def plan(self, qpos: Optional[np.ndarray], goal_q: Sequence[float], start_q: Optional[Sequence[float]] = None,
             ignore: Iterable = (), ignore_classes: Iterable[str] = (), attach: Iterable[str] = (),
             model: Optional[mujoco.MjModel] = None, data: Optional[mujoco.MjData] = None,
             time_limit_s: Optional[float] = None) -> PlanResult:
        """Path from start_q (default: the arm joints in qpos / data) to goal_q, avoiding every obstacle class.

        ``ignore``: geom / body names or ids to drop for this plan (the pick target and its pedicel);
        ``ignore_classes``: whole classes (e.g. ``("fruit",)``); ``attach``: free bodies carried with the wrist.
        """
        t0 = time.perf_counter()
        sc = self.scene
        sc.detach_all()
        sc.sync(model, data, qpos)
        sc.set_ignore(ignore, ignore_classes)
        for b in attach:
            sc.attach(b)
        n0 = sc.n_checks
        start = np.array(start_q if start_q is not None else sc.d.qpos[sc.qadr], dtype=np.float64)
        goal = np.asarray(goal_q, dtype=np.float64)
        res = PlanResult(False, self.backend)
        for q, why in ((start, "start_invalid"), (goal, "goal_invalid")):
            if not self._valid(q):
                res.reason = why
                res.contacts = sc.contacts(q) if sc.in_limits(q) else [{"class": "limits", "q": q.tolist()}]
                res.n_checks, res.plan_time_s = sc.n_checks - n0, time.perf_counter() - t0
                return res
        budget = self.time_limit_s if time_limit_s is None else float(time_limit_s)
        t_end = t0 + budget
        # Attempt 1 checks edges at edge_step; if the dense re-check finds a thin obstacle the coarse edges
        # skipped (a goal close to wood), attempt 2 re-plans with edges checked at verify_step.
        res.reason = "no_solution"
        for step in dict.fromkeys((self.edge_step, self.verify_step)):
            left = t_end - time.perf_counter()
            if left <= 0.0:
                break
            self._step = step
            ts = time.perf_counter()
            if sc.motion_valid(start, goal, step):
                path = np.vstack([start, goal])
            elif self.backend == "ompl":
                path = self._solve_ompl(start, goal, 0.8 * left)
            else:
                path = self._solve_rrtconnect(start, goal, 0.8 * left)
            res.solve_time_s += time.perf_counter() - ts
            if path is None:
                res.reason = "no_solution"
                continue
            path = self._shortcut(path, budget=max(0.0, min(0.2 * budget, t_end - time.perf_counter())))
            if self.verify(path):
                res.ok, res.path, res.reason = True, path, ""
                res.length = _path_length(path)
                res.min_clearance = self.path_clearance(path)
                res.trajectory = time_parameterize(path, self.vmax, self.amax)
                break
            res.reason = "verify_failed"
        self._step = self.edge_step
        res.n_checks = sc.n_checks - n0
        res.plan_time_s = time.perf_counter() - t0
        return res

    def verify(self, path: np.ndarray, step: Optional[float] = None) -> bool:
        """Independent dense re-check of every segment of a path in the current scene."""
        step = self.verify_step if step is None else step
        return all(self.scene.motion_valid(a, b, step) for a, b in zip(path[:-1], path[1:])) if len(path) > 1 \
            else bool(len(path) and self._valid(path[0]))

    def path_clearance(self, path: np.ndarray, step: float = 0.02, cap: float = 0.10) -> Dict[str, float]:
        """Minimum distance per obstacle class along the path (sampled every ``step``)."""
        out: Dict[str, float] = {}
        for a, b in zip(path[:-1], path[1:]):
            n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
            for k in range(n + 1):
                for c, v in self.scene.clearance(a + (b - a) * (k / n), cap).items():
                    out[c] = min(out.get(c, cap), v)
        return out

    # ------------------------------------------------------------------ helpers
    def _valid(self, q: np.ndarray) -> bool:
        if np.any(q < self.limits[:, 0] - 1e-9) or np.any(q > self.limits[:, 1] + 1e-9):
            return False
        return self.scene.is_valid(q)

    def _edge(self, a: np.ndarray, b: np.ndarray) -> bool:
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / self._step)))
        for k in range(1, n + 1):
            if not self._valid(a + (b - a) * (k / n)):
                return False
        return True

    def _shortcut(self, path: np.ndarray, budget: float, iters: int = 60) -> np.ndarray:
        """Greedy then random shortcutting with the backend edge check (deterministic rng)."""
        pts = [np.asarray(p, dtype=np.float64) for p in path]
        i = 0
        while i < len(pts) - 2:                      # greedy: from each point jump to the farthest visible one
            for j in range(len(pts) - 1, i + 1, -1):
                if self._edge(pts[i], pts[j]):
                    pts = pts[:i + 1] + pts[j:]
                    break
            i += 1
        t_end = time.perf_counter() + budget
        for _ in range(iters):
            if len(pts) < 3 or time.perf_counter() > t_end:
                break
            i, j = sorted(self.rng.choice(len(pts) - 1, 2, replace=False))
            if j - i < 1:
                continue
            ua, ub = self.rng.random(2)
            a = pts[i] + (pts[i + 1] - pts[i]) * ua
            b = pts[j] + (pts[j + 1] - pts[j]) * ub
            if self._edge(a, b):
                pts = pts[:i + 1] + [a, b] + pts[j + 1:]
        return np.array(pts)

    # ------------------------------------------------------------------ OMPL backend
    def _solve_ompl(self, start: np.ndarray, goal: np.ndarray, budget: float) -> Optional[np.ndarray]:
        global _OMPL_SEEDED
        if not _OMPL_SEEDED:
            _ou.setLogLevel(_ou.LOG_WARN)
            _ou.RNG.setSeed(self.seed)
            _OMPL_SEEDED = True
        n = len(start)
        space = _ob.RealVectorStateSpace(n)
        bounds = _ob.RealVectorBounds(n)
        for i in range(n):
            bounds.setLow(i, float(self.limits[i, 0]))
            bounds.setHigh(i, float(self.limits[i, 1]))
        space.setBounds(bounds)
        ss = _og.SimpleSetup(space)
        valid = self._valid

        def checker(s):
            return valid(np.array([s[i] for i in range(n)]))

        ss.setStateValidityChecker(checker)
        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(self._step / space.getMaximumExtent())
        s0, s1 = space.allocState(), space.allocState()
        for i in range(n):
            s0[i], s1[i] = float(start[i]), float(goal[i])
        ss.setStartAndGoalStates(s0, s1)
        planner = _og.RRTConnect(si)
        planner.setRange(self.rrt_range)
        ss.setPlanner(planner)
        solved = ss.solve(budget)
        if not (solved and ss.haveExactSolutionPath()):
            return None
        # No OMPL simplifier: its B-spline smoothing adds states (and checks); both backends share _shortcut.
        p = ss.getSolutionPath()
        path = np.array([[p.getState(k)[i] for i in range(n)] for k in range(p.getStateCount())])
        path[0], path[-1] = start, goal
        return path

    # ------------------------------------------------------------------ pure-Python RRT-Connect
    def _solve_rrtconnect(self, start: np.ndarray, goal: np.ndarray, budget: float,
                          max_nodes: int = 20000) -> Optional[np.ndarray]:
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        n = len(start)
        trees = []
        for root in (start, goal):
            pts = np.zeros((max_nodes, n))
            pts[0] = root
            trees.append({"pts": pts, "par": np.full(max_nodes, -1, dtype=np.int64), "n": 1})

        def nearest(T, q):
            return int(np.argmin(np.sum((T["pts"][:T["n"]] - q) ** 2, axis=1)))

        def add(T, q, parent):
            k = T["n"]
            T["pts"][k], T["par"][k] = q, parent
            T["n"] += 1
            return k

        def extend(T, q):
            """One step from the nearest node toward q: ("reached" | "advanced" | "trapped", node id)."""
            i = nearest(T, q)
            qn = T["pts"][i]
            d = np.linalg.norm(q - qn)
            qs = q if d <= self.rrt_range else qn + (q - qn) * (self.rrt_range / d)
            if not self._edge(qn, qs):
                return "trapped", -1
            k = add(T, qs.copy(), i)
            return ("reached" if d <= self.rrt_range else "advanced"), k

        def connect(T, q):
            while True:
                st, k = extend(T, q)
                if st != "advanced" or T["n"] >= max_nodes:
                    return st, k

        def branch(T, k):
            out = []
            while k >= 0:
                out.append(T["pts"][k].copy())
                k = int(T["par"][k])
            return out

        t_end = time.perf_counter() + budget
        a, b = 0, 1
        while time.perf_counter() < t_end and trees[0]["n"] < max_nodes and trees[1]["n"] < max_nodes:
            q_rand = lo + (hi - lo) * self.rng.random(n)
            st, ka = extend(trees[a], q_rand)
            if st != "trapped":
                st2, kb = connect(trees[b], trees[a]["pts"][ka])
                if st2 == "reached":
                    pa, pb = branch(trees[a], ka), branch(trees[b], kb)
                    path = (pa[::-1] + pb[1:]) if a == 0 else (pb[::-1] + pa[1:])
                    return np.array(path)
            a, b = b, a
        return None
