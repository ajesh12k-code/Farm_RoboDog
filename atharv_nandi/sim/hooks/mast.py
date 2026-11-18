"""WP24 crown mast runtime hooks (ORCHARD-MVP-SPEC 2.3, 3.5; WP24).

Two hooks share one ``MastRuntime`` (a fault in the rate hook leaves the per-step motion and strike check running,
and the other way round):

    mast_step   ``every_step`` (the WP0 / WP0b substep point, before every ``mj_step``), ``mutates_model``:
                e-stop brake from the runner hold, the geometric strike check of the head and the top section,
                one rate-limited arbiter step, then the section ``body_pos`` writes (ext / 3 each, interpolated
                every physics step: at most 0.4 mm per 2 ms step) and the gimbal ``cam_quat`` write.
    mast        50 Hz: ``/nandi/mast/cmd`` requests, the wind / gust / tilt retract backstops, ``bus.values["mast"]``
                (the state; WP7 reads ``ext_m``) and ``bus.values["mast_control"]`` (this runtime, the in-process
                ``MastControl`` for WP6 / WP25 / WP27 / WP28), and ``/nandi/mast/state`` at 10 Hz and on change.

The setpoint has one writer, the arbiter in ``controllers.mast`` (priority e-stop > safety > recovery > preset).

Strike (rule 3). Every physics step every crown head geom (box, anemometer, GNSS puck) and the top section capsule
are distance-checked with
``mujoco.mj_geomDistance`` (MuJoCo >= 3.2; the analytic ``controllers.mast.geom_distance`` when it is missing)
against the obstacle set: registry **wood** (tree ``*_wood_*`` and ``*_prune_*`` geoms in group 0, ``prop_*``,
stub trunks), the **foliage envelope** (group-5 ``*_envelope_*``, stub canopies) and **rope lines** (group-5
``rope_*``). Candidates are the obstacles whose bounding sphere is within ``CANDIDATE_MARGIN_M`` (0.5 m) of the
head or the top section, re-selected every ``CANDIDATE_REFRESH_S`` (0.1 s of sim time). The exact distance is
evaluated every physics step while it is below ``NEAR_M``; farther out it is re-evaluated after
``(dist - NEAR_M) / V_REL_MAX_MPS`` (2.0 m/s ASSUMED closing-speed bound), the earliest time it could reach
``NEAR_M`` (a 0.08 m clearance is re-checked every 15 ms, 0.5 m every 0.22 s; new candidates force a check).
Distance < 0.05 m stops mast motion: upward always, downward when lowering by 5 mm would close on the nearest
obstacle (analytic distance); a ``safety`` retract always runs. Distance <= 0 is a strike event: a ``safety`` retract
to 0 is submitted in the same substep, so the setpoint already moves down on that step (retract within 1 hook
tick), and the state is published at the next tick. The head is contype 0 / conaffinity 0: a strike never pushes
the robot. While the e-stop brake holds, a strike is recorded but the mast does not move (e-stop beats safety);
when the brake is released with the head still at distance <= 0, the retract is submitted then. A non-safety lowering
held by the near stop for ``LOWER_STALL_S`` (1.0 s) keeps holding and is flagged in the state (``lower_blocked``) for
the requester to escalate. The runner-hold brake ends with the hold, and so does a ``brake()`` / ``estop`` request
that arrives while the runner holds (WP6 brakes on the e-stop latch edge, inside the hold: the same e-stop); one that
arrives outside a hold latches (``brake_latched``) until ``release("estop")``, also through a later hold.

Raise gates (rule 1), checked for every raise (target above the current extension), cheapest first. The gates
before the clearance (e-stop, standing, cross-slope, wind, WP6 gates: ``motion_gates``) are checked again every
50 Hz tick while a raise runs, and the clearance once more when a raise resumes after a near-obstacle hold. A
failure holds the raise at once (``held_by`` ``gate:<detail>``); after ``REGATE_DROP_S`` 0.5 s of failing the raise
is dropped (``last_decision`` refused, detail ``regate:<detail>``). The head reference is the centre of the head
box (``geom_xpos``; 0.02 m above the body origin); the head top (``head_top``) is the top of the highest head geom,
the anemometer, 0.11 m above that centre, and the swept bounding sphere ``r_head`` (0.12 m) encloses every head geom.
    estop        the runner holds -> refused (``crown_gate``, detail ``estop``)
    standing     trunk horizontal speed <= 0.05 m/s (``stability``, ``not_standing``)
    cross-slope  |roll| <= 3 deg (``slope``)
    wind         ``values["wind"]`` speed < 10 m/s and no gust (``wind``)
    WP6 gates    every function registered with ``register_gate(fn)``: ``fn(ext_from, ext_to, ctx) -> GateDecision``
                 (``ctx`` = {m, d, bus, request}); a gate that raises refuses (``stability``)
    clearance    the head's swept capsule from the current to the target height clears every obstacle by
                 >= 0.10 m; the head top (anemometer) + 0.10 m stays under ``alley_clear_height_m`` in a tree's
                 alley overhang zone (registry from ``world.modules.trees.plans_for_model``); and no mast-LiDAR
                 return (rays along the mast axis on the swept cylinder, groups 0 / 1 / 3) inside it
                 (``crown_clearance``, detail ``wood:<geom>``, ``envelope:<geom>``, ``rope:<geom>``,
                 ``alley_clear_height:<tree>``, ``lidar:<geom>``)

Backstop retracts (50 Hz, crown up, not braked, no safety retract to 0 already running): wind >= 10 m/s or a
gust -> ``safety`` retract at once when standing, else after ``WIND_STANCE_S`` 0.5 s (rule 4: WP6 stops the gait
into a stance first); instantaneous tilt > 12 deg -> retract (rule 5). The filtered-tilt, margin and link-loss
triggers are WP6 / WP28 requests.

``/nandi/mast/cmd`` (JSON): ``{source: safety|recovery|preset, target_ext | band: stow|low|mid|high, reason,
gimbal_deg?: [pan, tilt]}``, ``{release: <source>}``, ``{aim_deg: [pan, tilt]}``. ``estop`` cannot be requested
(the brake comes only from the runner hold), and a ``safety`` request may only retract (``safety_raise``). An
``aim_deg`` next to a request applies only when the request is accepted. Every message delivered since the last
tick is processed in order (``bus.since``); messages that fell out of the bus window are counted in ``cmd_lost``.
A malformed command is counted in ``rejected`` and ignored; the hook never raises on input.

Sim-state provider ``mast`` (v1): ``{ext_m, gimbal: [pan_rad, tilt_rad]}`` from the model fields (section
``body_pos``, ``cam_quat`` of ``crown_cam``), plus the informational ``crown_lens_z`` (world z of ``crown_cam``) and
``arbiter_source`` when a runtime drives the model in this process (OS 3.5); ``unpack`` writes the model fields
back, so renderers show the raised crown and the gimbal view, and checks the informational ones. Applies only to
models with the WP1 mast bodies.
"""
import json
import math
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple

import mujoco
import numpy as np

from atharv_nandi.core import mast as mc
from atharv_nandi.core import sim_state
from atharv_nandi.core.interfaces import MAST_EXT_MAX_M, MAST_PRIORITY, GateDecision, MastRequest
from atharv_nandi.core.locomotion_logic import quat_to_roll_pitch

SECTIONS = ("mast_s1", "mast_s2", "mast_s3")
HEAD_BODY, HEAD_GEOM, TOP_GEOM, CAMERA, BASE_BODY = "crown_head", "crown_head_g", "mast_s3_g", "crown_cam", "mast_base"
CMD_TOPIC = "/nandi/mast/cmd"
STATE_TOPIC = "/nandi/mast/state"
PROVIDER_KEY = "mast"
PROVIDER_VERSION = 1
RATE_HZ = 50.0
PUBLISH_PERIOD_S = 0.1
CANDIDATE_MARGIN_M = 0.5
CANDIDATE_REFRESH_S = 0.1
DISTMAX_M = 0.5                      # mj_geomDistance search cap; clearance_m above it is reported as None
V_REL_MAX_MPS = 2.0                  # ASSUMED bound on head-obstacle closing speed (walk 0.6 + mast 0.2 + sway)
STANDING_MAX_MPS = 0.05
RAISE_MAX_ROLL_DEG = 3.0             # rule 1 dwell limit
WIND_RAISE_MAX_MPS = 10.0            # rule 1 / rule 4
WIND_STANCE_S = 0.5                  # rule 4: stance within 0.5 s, then retract
REGATE_DROP_S = 0.5                  # a running raise held by a failing gate is dropped after this
LOWER_STALL_S = 1.0                  # a lowering held this long by a near obstacle is flagged ``lower_blocked``
TILT_RETRACT_DEG = 12.0              # rule 5 instantaneous tilt
LIDAR_GROUPS = (0, 1, 3)             # wood / ground, fruit and actors, foliage (robot is group 2, queries 5)
LIDAR_AZIMUTHS = 8
BANDS = {"stow": 0.0, "low": 0.0, "mid": 0.60, "high": 1.60}   # OS 2.3 bands (low = ring camera, stowed)
CMD_SOURCES = ("safety", "recovery", "preset")
_HAS_GEOM_DISTANCE = hasattr(mujoco, "mj_geomDistance")
_GTYPE = {int(mujoco.mjtGeom.mjGEOM_CAPSULE): mc.CAPSULE, int(mujoco.mjtGeom.mjGEOM_CYLINDER): mc.CYLINDER,
          int(mujoco.mjtGeom.mjGEOM_SPHERE): mc.SPHERE, int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): mc.ELLIPSOID,
          int(mujoco.mjtGeom.mjGEOM_BOX): mc.BOX}
_GATES: List[Callable] = []


def register_gate(fn: Callable) -> Callable:
    """WP6 hook point: ``fn(ext_from, ext_to, ctx) -> GateDecision`` is checked on every raise."""
    if fn not in _GATES:
        _GATES.append(fn)
    return fn


def unregister_gate(fn: Callable):
    if fn in _GATES:
        _GATES.remove(fn)


def classify_geom(name: str, group: int) -> Optional[str]:
    """Obstacle kind of a world geom for the strike and clearance checks, or None."""
    if not name:
        return None
    if group == 0 and (name.startswith("prop_") or "_wood" in name or "_prune" in name
                       or (name.startswith("tree_stub_") and name.endswith("_trunk"))):
        return "wood"
    if group == 5 and "_envelope" in name:
        return "envelope"
    if name.startswith("tree_stub_") and name.endswith("_canopy"):
        return "envelope"
    if group == 5 and name.startswith("rope_"):
        return "rope"
    return None


def _id(m, kind, name) -> int:
    return mujoco.mj_name2id(m, kind, name)


def has_mast(m) -> bool:
    return all(_id(m, mujoco.mjtObj.mjOBJ_BODY, s) >= 0 for s in SECTIONS)


# ---------------------------------------------------------------------- gimbal quaternion
def gimbal_quat(pan: float, tilt: float) -> np.ndarray:
    """``cam_quat`` (head frame) of a level-horizon camera looking along pan (about head z, 0 = +x) and tilt
    (positive up). pan = tilt = 0 is the WP1 rest orientation (level, along the head's +x)."""
    f = np.array([math.cos(tilt) * math.cos(pan), math.cos(tilt) * math.sin(pan), math.sin(tilt)])
    r = np.array([math.sin(pan), -math.cos(pan), 0.0])          # camera +x (right)
    u = np.cross(-f, r)                                          # camera +y (up)
    mat = np.column_stack([r, u, -f]).reshape(-1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def gimbal_of_quat(q) -> Tuple[float, float]:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q, float))
    f = -mat.reshape(3, 3)[:, 2]
    return math.atan2(f[1], f[0]), math.asin(max(-1.0, min(1.0, f[2])))


# ---------------------------------------------------------------------- sim-state provider
def _section_ids(m) -> List[int]:
    return [_id(m, mujoco.mjtObj.mjOBJ_BODY, s) for s in SECTIONS]


def _pack(m) -> dict:
    if not has_mast(m):
        return {}
    ext = float(sum(m.body_pos[b, 2] for b in _section_ids(m)))
    out = {"ext_m": round(ext, 9)}
    c = _id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    if c >= 0:
        pan, tilt = gimbal_of_quat(m.cam_quat[c])
        out["gimbal"] = [round(pan, 9), round(tilt, 9)]
    rt = _live_runtime(m)
    if rt is not None:               # OS 3.5 informational fields (``unpack`` checks and ignores them)
        lz = rt.lens_z()
        st = rt.arb.state()
        out["crown_lens_z"] = None if lz is None else round(lz, 6)
        out["arbiter_source"] = st["arbiter_source"]
    return out


_LIVE: List["weakref.ref"] = []


def _live_runtime(m) -> Optional["MastRuntime"]:
    """The runtime bound to model ``m`` in this process (the one the hooks drive), or None."""
    for ref in reversed(list(_LIVE)):        # the newest runtime bound to m wins
        rt = ref()
        if rt is None:
            _LIVE.remove(ref)
        elif rt.m is m and rt.ok and rt.d is not None:
            return rt
    return None


def _num(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"mast {what} must be a finite number, got {v!r}")
    return float(v)


def _unpack(m, data: dict) -> None:
    if not data:
        return
    if not has_mast(m):
        raise ValueError("state carries a mast block but this model has no mast sections")
    ext = _num(data.get("ext_m"), "ext_m")
    if not -1e-6 <= ext <= MAST_EXT_MAX_M + 1e-6:
        raise ValueError(f"mast ext_m {ext} outside [0, {MAST_EXT_MAX_M}]")
    g = data.get("gimbal")
    if g is not None:
        if not isinstance(g, (list, tuple)) or len(g) != 2:
            raise ValueError(f"mast gimbal must be [pan, tilt], got {g!r}")
        pan, tilt = _num(g[0], "gimbal pan"), _num(g[1], "gimbal tilt")
        if abs(pan) > math.pi + 1e-6 or not mc.TILT_MIN_RAD - 1e-6 <= tilt <= mc.TILT_MAX_RAD + 1e-6:
            raise ValueError(f"mast gimbal out of range: {g!r}")
    if data.get("crown_lens_z") is not None:
        _num(data["crown_lens_z"], "crown_lens_z")
    if data.get("arbiter_source") not in (None,) + tuple(MAST_PRIORITY):
        raise ValueError(f"mast arbiter_source {data.get('arbiter_source')!r}")
    for b, z in zip(_section_ids(m), mc.split(ext)):
        m.body_pos[b, 2] = z
    c = _id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    if g is not None and c >= 0:
        m.cam_quat[c] = gimbal_quat(pan, tilt)


sim_state.register_provider(PROVIDER_KEY, _pack, _unpack, version=PROVIDER_VERSION, applies=has_mast)


# ---------------------------------------------------------------------- runtime
class MastRuntime:
    """The arbiter bound to one model; implements ``interfaces.MastControl`` (``submit``, ``brake``, ``state``)."""

    def __init__(self, arbiter: Optional[mc.MastArbiter] = None, use_geom_distance: bool = _HAS_GEOM_DISTANCE,
                 lidar: bool = True):
        self.arb = arbiter if arbiter is not None else mc.MastArbiter()
        self.arb.check = self.check_raise
        self.use_geom_distance = bool(use_geom_distance)
        self.lidar = bool(lidar)
        self.m = None
        self.d = None
        self.bus = None
        self.ok = False
        self._fromto = np.zeros(6)
        self._rays = (np.array([1 if g in LIDAR_GROUPS else 0 for g in range(6)], dtype=np.uint8),
                      np.zeros(1, dtype=np.int32))
        self._cmd_seen = 0
        self.reset_runtime()
        _LIVE.append(weakref.ref(self))

    # -- binding
    def bind(self, m, d):
        if m is self.m and d is self.d:
            return
        self.m, self.d = m, d
        self.ok = has_mast(m)
        if not self.ok:
            return
        O = mujoco.mjtObj
        self.sec = _section_ids(m)
        self.rest_z = [float(m.body_pos[b, 2]) for b in self.sec]
        self.head_b = _id(m, O.mjOBJ_BODY, HEAD_BODY)
        self.base_b = _id(m, O.mjOBJ_BODY, BASE_BODY)
        self.head_g = _id(m, O.mjOBJ_GEOM, HEAD_GEOM)
        self.top_g = _id(m, O.mjOBJ_GEOM, TOP_GEOM)
        self.cam = _id(m, O.mjOBJ_CAMERA, CAMERA)
        self.rest_cam = m.cam_quat[self.cam].copy() if self.cam >= 0 else None
        # half height of the head box along the mast axis: the head top is the box centre + head_half (the box sits
        # 0.02 m above the crown_head body origin, so the body origin is not the reference)
        self.head_half = float(m.geom_size[self.head_g][2]) if self.head_g >= 0 else 0.06
        # every geom of the crown head body (box, anemometer, GNSS puck) is part of the physical head: the strike
        # and near checks measure all of them, and the head top and the swept bounding sphere enclose all of them
        head_gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.head_b] if self.head_b >= 0 else []
        if self.head_g in head_gs:
            head_gs.remove(self.head_g)
        self.head_geoms = ([self.head_g] if self.head_g >= 0 else []) + head_gs
        self.strike_geoms = self.head_geoms + ([self.top_g] if self.top_g >= 0 else [])
        self.head_top, self.r_head = self._head_extent(m)
        self.r_head_xy = math.hypot(*m.geom_size[self.head_g][:2]) if self.head_g >= 0 else 0.085
        obs = []
        for g in range(m.ngeom):
            kind = classify_geom(mujoco.mj_id2name(m, O.mjOBJ_GEOM, g) or "", int(m.geom_group[g]))
            if kind is not None:
                obs.append((g, kind))
        self.obs_ids = np.array([g for g, _ in obs], dtype=np.int64)
        self.obs_kind = [k for _, k in obs]
        self.obs_rbound = np.array([float(m.geom_rbound[g]) for g, _ in obs])
        self.obs_name = [mujoco.mj_id2name(m, O.mjOBJ_GEOM, g) for g, _ in obs]
        self.alleys = self._alley_rules(m)
        self._cands: List[int] = []         # indices into obs_ids
        self._next_refresh = -1.0
        self._next_exact = -1.0
        self._written = None
        self._gimbal_written = None
        ext = float(sum(m.body_pos[b, 2] - z for b, z in zip(self.sec, self.rest_z)))
        if ext > 1e-9 and self.arb.ext <= 1e-9:
            self.arb.ext = mc.clamp_ext(ext)   # a model that carries an extension (unpacked state)

    def _head_extent(self, m) -> Tuple[float, float]:
        """(top offset along the head axis, bounding radius), both from the head box centre, of the union of the
        head geoms (body frame; the WP1 head: anemometer top 0.11 m above the box centre, radius 0.12 m)."""
        if self.head_g < 0 or not self.head_geoms:
            return 0.06, 0.104
        c0 = np.array(m.geom_pos[self.head_g], float)
        top, rad = 0.0, 0.0
        for g in self.head_geoms:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.geom_quat[g])
            R = R.reshape(3, 3)
            s = m.geom_size[g]
            t = int(m.geom_type[g])
            if t == int(mujoco.mjtGeom.mjGEOM_BOX):
                hz = float(sum(abs(R[2, i]) * s[i] for i in range(3)))
            elif t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                cz = abs(float(R[2, 2]))
                hz = cz * float(s[1]) + float(s[0]) * (math.sqrt(max(0.0, 1.0 - cz * cz))
                                                        + (1.0 if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0.0))
            else:
                hz = float(m.geom_rbound[g])
            off = np.array(m.geom_pos[g], float) - c0
            top = max(top, float(off[2]) + hz)
            rad = max(rad, float(np.linalg.norm(off)) + float(m.geom_rbound[g]))
        return top, rad

    @staticmethod
    def _alley_rules(m) -> List[mc.AlleyRule]:
        try:
            from atharv_nandi.sim.world.modules import trees as mt
            plans = mt.plans_for_model(m)
        except Exception:  # noqa: BLE001 - no tree module: no alley rules
            return []
        out = []
        for p in plans:
            out.extend(mc.alley_rules(p.get("entry") or {}, p.get("pos") or (0.0, 0.0, 0.0)))
        return out

    def reset_runtime(self):
        self.arb.reset()
        self.dist = math.inf
        self.dist_geom = ""
        self.dist_kind = ""
        self.dist_idx = -1
        self.near = False
        self.cmd_lost = 0
        self._regate_since: Optional[float] = None
        self._near_held = False             # a near-obstacle hold stopped a raise since the last re-gate
        self.in_strike = False
        self.strikes: List[Dict[str, Any]] = []
        self.rejected = 0
        self.last_reject = ""
        self._hold_brake = False
        self._lower_held_since: Optional[float] = None
        self.lower_blocked: Optional[Dict[str, Any]] = None
        self._wind_since: Optional[float] = None
        self._pub_key = None
        self._next_pub = 0.0
        self._dirty = False
        self._written = None
        self._gimbal_written = None
        self._next_refresh = -1.0
        self.substeps = 0
        self._next_exact = -1.0

    def reset(self, m, d):
        """World reset: model fields back to rest, extension 0, every request dropped."""
        if m is not None and m is self.m and self.ok:
            for b, z in zip(self.sec, self.rest_z):
                m.body_pos[b, 2] = z
            if self.cam >= 0 and self.rest_cam is not None:
                m.cam_quat[self.cam] = self.rest_cam
        self.reset_runtime()

    def tick_reset(self, m, d):
        self._cmd_seen = self.bus.arrival_count(CMD_TOPIC) if self.bus is not None else 0

    # -- geometry helpers
    def axis(self) -> np.ndarray:
        return self.d.xmat[self.base_b].reshape(3, 3)[:, 2].copy()

    def head_pos(self) -> np.ndarray:
        """Centre of the head box (the centre of its bounding sphere ``r_head``), not the body origin."""
        if self.head_g >= 0:
            return self.d.geom_xpos[self.head_g].copy()
        return self.d.xpos[self.head_b].copy()

    def obstacles(self) -> List[mc.Obstacle]:
        d = self.d
        return [mc.Obstacle(self.obs_name[i], self.obs_kind[i], _GTYPE.get(int(self.m.geom_type[g]), "other"),
                            d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3).copy(), self.m.geom_size[g].copy(),
                            float(self.obs_rbound[i])) for i, g in enumerate(self.obs_ids)]

    def _refresh_candidates(self, t: float):
        self._next_refresh = t + CANDIDATE_REFRESH_S
        if not len(self.obs_ids):
            self._cands = []
            return
        d = self.d
        P = d.geom_xpos[self.obs_ids]
        sel = np.zeros(len(self.obs_ids), dtype=bool)
        for g in self.strike_geoms:
            dist = np.linalg.norm(P - d.geom_xpos[g], axis=1) - self.obs_rbound - float(self.m.geom_rbound[g])
            sel |= dist < CANDIDATE_MARGIN_M
        self._cands = [int(i) for i in np.nonzero(sel)[0]]

    def strike_distance(self) -> Tuple[float, str, str]:
        """(min distance, geom, kind) of the head geoms (box, anemometer, GNSS puck) and the top section to the
        current candidates (the nearest obstacle index is kept in ``dist_idx``)."""
        best, who, kind = math.inf, "", ""
        self.dist_idx = -1
        m, d = self.m, self.d
        for i in self._cands:
            og = int(self.obs_ids[i])
            for g in self.strike_geoms:
                if self.use_geom_distance:
                    dist = float(mujoco.mj_geomDistance(m, d, g, og, DISTMAX_M, self._fromto))
                else:
                    dist = self._analytic_distance(g, i)
                if dist < best:
                    best, who, kind = dist, self.obs_name[i], self.obs_kind[i]
                    self.dist_idx = i
        return best, who, kind

    def lowering_closes(self, step_m: float = 0.005) -> bool:
        """True when lowering the head and the top section by ``step_m`` along the mast axis brings them closer
        to the nearest obstacle (analytic distance; only evaluated while ``near``)."""
        i = getattr(self, "dist_idx", -1)
        if i < 0:
            return False
        shift = -self.axis() * float(step_m)
        for g in self.strike_geoms:
            if self._analytic_distance(g, i, shift) < self._analytic_distance(g, i) - 1e-6:
                return True
        return False

    def _analytic_distance(self, g: int, i: int, shift=None) -> float:
        m, d = self.m, self.d
        og = int(self.obs_ids[i])
        ob = mc.Obstacle(self.obs_name[i], self.obs_kind[i], _GTYPE.get(int(m.geom_type[og]), "other"),
                         d.geom_xpos[og], d.geom_xmat[og].reshape(3, 3), m.geom_size[og], float(self.obs_rbound[i]))
        c = d.geom_xpos[g] if shift is None else d.geom_xpos[g] + shift
        if int(m.geom_type[g]) in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * float(m.geom_size[g][1])
            return mc.geom_distance(c - ax, c + ax, float(m.geom_size[g][0]), ob)
        return mc.geom_distance(c, c, float(m.geom_rbound[g]), ob)

    # -- MastControl
    def submit(self, request: MastRequest) -> GateDecision:
        if request.source == "estop":    # same as brake(): joins a running runner hold, else latches
            self.brake()
            self.arb.last = mc.Decision(True, "estop", request.target_ext, "", "", request.t)
            return GateDecision.ok(mc.GATE, target_ext=float(self.arb.ext))
        dec = self.arb.submit(request)
        self._dirty = True
        return dec

    def brake(self):
        """E-stop brake. While the runner holds, the brake is part of that e-stop (WP6 ``EstopActions`` calls it on
        the latch edge, inside the hold) and ends with the hold; outside a hold it latches until
        ``release("estop")``, also through a later hold."""
        if self.bus is not None and self.bus.holding():
            if not self.arb.braked:
                self.arb.brake(latch=False)
            self._hold_brake = True
        else:
            self.arb.brake()
        self._dirty = True

    def release(self, source: str) -> bool:
        self._dirty = True
        return self.arb.release(source)

    def aim(self, pan: float, tilt: float):
        self.arb.aim(pan, tilt)
        self._dirty = True

    def lens_z(self) -> Optional[float]:
        if not self.ok or self.cam < 0 or self.d is None:
            return None
        return float(self.d.cam_xpos[self.cam][2])

    def state(self) -> Dict[str, Any]:
        st = self.arb.state()
        pan, tilt = st["gimbal"]
        last = st["last_decision"]
        st.update({
            "ext_m": round(st["ext_m"], 6), "target_ext": round(st["target_ext"], 6),
            "lens_z": None if self.lens_z() is None else round(self.lens_z(), 4),
            "gimbal": [round(pan, 6), round(tilt, 6)],
            "gimbal_deg": [round(math.degrees(pan), 2), round(math.degrees(tilt), 2)],
            "detail": (last or {}).get("detail", "") if st["gated"] else "",
            "clearance_m": None if not math.isfinite(self.dist) or self.dist >= DISTMAX_M else round(self.dist, 4),
            "clearance_geom": self.dist_geom if math.isfinite(self.dist) else "",
            "near": self.near, "lower_blocked": self.lower_blocked, "strike_count": len(self.strikes),
            "strike": self.strikes[-1] if self.strikes else None,
            "rejected": self.rejected, "cmd_lost": self.cmd_lost, "has_mast": self.ok,
            "t": round(float(self.d.time), 4) if self.d is not None else 0.0})
        return st

    # -- raise gates
    def _standing(self) -> bool:
        return math.hypot(float(self.d.qvel[0]), float(self.d.qvel[1])) <= STANDING_MAX_MPS

    def _roll_pitch(self) -> Tuple[float, float]:
        return quat_to_roll_pitch(*self.d.qpos[3:7])

    def _wind(self) -> Tuple[float, bool]:
        w = (self.bus.values.get("wind") if self.bus is not None else None) or {}
        if not isinstance(w, dict):
            return 0.0, False
        s = w.get("speed_mps")
        s = float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) else 0.0
        return s, bool(w.get("gust"))

    def check_raise(self, ext_from: float, ext_to: float, req: MastRequest) -> GateDecision:
        """Every rule-1 gate (the arbiter's ``check``): ``motion_gates`` then ``clearance``."""
        dec = self.motion_gates(ext_from, ext_to, req)
        if not dec:
            return dec
        return self.clearance(ext_from, ext_to)

    def motion_gates(self, ext_from: float, ext_to: float, req: MastRequest) -> GateDecision:
        """The raise gates without the clearance: e-stop, standing, cross-slope, wind, WP6 gates. Checked when a
        raise is submitted and again every 50 Hz tick while it runs (``_regate``)."""
        def refuse(reason, detail, **v):
            return GateDecision.refuse(mc.GATE, reason, detail=detail, target_ext=ext_to, **v)

        if not self.ok or self.d is None:
            return refuse(mc.GATE, "no_mast")
        if self.bus is not None and self.bus.holding():
            return refuse(mc.GATE, "estop")
        if not self._standing():
            return refuse(mc.R_STABILITY, "not_standing")
        roll, _ = self._roll_pitch()
        if abs(math.degrees(roll)) > RAISE_MAX_ROLL_DEG:
            return refuse(mc.R_SLOPE, f"cross_slope:{math.degrees(roll):.1f}")
        speed, gust = self._wind()
        if gust or speed >= WIND_RAISE_MAX_MPS:
            return refuse(mc.R_WIND, "gust" if gust else f"wind:{speed:.1f}")
        ctx = {"m": self.m, "d": self.d, "bus": self.bus, "request": req}
        for fn in list(_GATES):
            try:
                dec = fn(ext_from, ext_to, ctx)
            except Exception as exc:  # noqa: BLE001 - fail safe
                return refuse(mc.R_STABILITY, f"gate_error:{type(exc).__name__}")
            if not dec:
                return GateDecision.refuse(mc.GATE, dec.reason or mc.R_STABILITY, detail=dec.gate,
                                           target_ext=ext_to)
        return GateDecision.ok(mc.GATE, target_ext=ext_to)

    def clearance(self, ext_from: float, ext_to: float) -> GateDecision:
        """Rule 1 swept-cylinder clearance from the current to the target extension."""
        ax = self.axis()
        c0 = self.head_pos()
        c1 = c0 + ax * (ext_to - ext_from)
        dist, ob = mc.swept_clearance(self.obstacles(), c0, c1, self.r_head)
        if ob is not None and dist < mc.CLEARANCE_M:
            return GateDecision.refuse(mc.GATE, mc.R_CLEARANCE, detail=f"{ob.kind}:{ob.name}", min_clear_m=dist,
                                       target_ext=ext_to)
        margin, rule = mc.alley_clear(c1[:2], float(c1[2]) + self.head_top, self.alleys)
        if rule is not None:
            return GateDecision.refuse(mc.GATE, mc.R_CLEARANCE, detail=f"alley_clear_height:{rule.tree_id}",
                                       margin_m=margin, target_ext=ext_to)
        if self.lidar:
            hit = self.lidar_sweep(c0, ax, ext_to - ext_from)
            if hit is not None:
                return GateDecision.refuse(mc.GATE, mc.R_CLEARANCE, detail=f"lidar:{hit[0]}", range_m=hit[1],
                                           target_ext=ext_to)
        return GateDecision.ok(mc.GATE, min_clear_m=float(dist) if math.isfinite(dist) else 99.0,
                               target_ext=ext_to)

    def lidar_sweep(self, c0, ax, travel: float) -> Optional[Tuple[str, float]]:
        """First mast-LiDAR return inside the swept cylinder: rays along the mast axis from the head top, on the
        axis and on a ring of radius (head half-diagonal + clearance)."""
        m, d = self.m, self.d
        length = max(0.0, float(travel)) + mc.CLEARANCE_M
        top = np.asarray(c0, float) + ax * self.head_top
        e1 = np.cross(ax, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(ax, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        rc = self.r_head_xy + mc.CLEARANCE_M
        origins = [top] + [top + rc * (math.cos(a) * e1 + math.sin(a) * e2)
                           for a in np.linspace(0.0, 2 * math.pi, LIDAR_AZIMUTHS, endpoint=False)]
        groups, gid = self._rays
        vec = np.asarray(ax, float)
        for p in origins:
            r = mujoco.mj_ray(m, d, np.asarray(p, float), vec, groups, 1, -1, gid)
            if 0.0 <= r <= length:
                return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid[0])) or f"geom{int(gid[0])}", float(r))
        return None

    # -- per physics step
    def substep(self, m, d, t, bus):
        self.bind(m, d)
        if not self.ok:
            return
        self.bus = bus
        self.substeps += 1
        holding = bus.holding()
        if holding and not self.arb.braked:
            self.arb.brake(latch=False)
            self._hold_brake = self._dirty = True
        elif not holding and self._hold_brake:
            if self.arb.braked and not self.arb.brake_latched:
                self.arb.release_brake()   # a brake from the runner hold ends with the hold; an explicit one stays
            self._hold_brake = False
            self._dirty = True
        # geometric strike check (rule 3) on the pose of the last step
        if t + 1e-12 >= self._next_refresh:
            self._refresh_candidates(t)
            self._next_exact = -1.0
        if not self._cands:
            self.dist, self.dist_geom, self.dist_kind = math.inf, "", ""
        elif t + 1e-12 >= self._next_exact:
            self.dist, self.dist_geom, self.dist_kind = self.strike_distance()
            # temporal coherence: the distance cannot fall below NEAR_M sooner than at V_REL_MAX_MPS
            self._next_exact = t + max(0.0, min(self.dist, DISTMAX_M) - mc.NEAR_M) / V_REL_MAX_MPS
        self.near = self.dist < mc.NEAR_M
        if self.dist <= mc.STRIKE_M:
            if not self.in_strike:
                self.in_strike = True
                ev = {"t": round(float(t), 4), "geom": self.dist_geom, "kind": self.dist_kind,
                      "dist_m": round(float(self.dist), 4), "ext_m": round(self.arb.ext, 4),
                      "braked": self.arb.braked}
                self.strikes.append(ev)
                self._dirty = True
            # Retract while the head is in the obstacle, also when the strike happened during the e-stop brake
            # (or a brake dropped the retract): once the brake is released, the retract is submitted here.
            w = self.arb.winner()
            if (not self.arb.braked and self.arb.ext > 1e-9
                    and not (w is not None and w.source == "safety" and (w.target_ext or 0.0) <= 1e-9)):
                self.arb.submit(MastRequest("safety", 0.0, "strike", t=float(t)))
                self._dirty = True
        elif self.dist > mc.NEAR_M:
            self.in_strike = False
        # rule 3: < 0.05 m stops mast motion (upward always; downward when lowering closes on the obstacle)
        hold_lower = False
        if self.near and self.arb.target() < self.arb.ext - 1e-9:
            hold_lower = self.lowering_closes()
        ext = self.arb.step(float(m.opt.timestep), hold_raise=self.near, hold_lower=hold_lower)
        if self.arb.held_by == "near_obstacle" and self.arb.raising() is not None:
            self._near_held = True
        self._lower_stall(float(t))
        if ext != self._written:
            for b, z0, dz in zip(self.sec, self.rest_z, mc.split(ext)):
                m.body_pos[b, 2] = z0 + dz
            self._written = ext
        if self.cam >= 0 and self.arb.gimbal != self._gimbal_written:
            if self.arb.gimbal == (0.0, 0.0) and self.rest_cam is not None:
                m.cam_quat[self.cam] = self.rest_cam
            else:
                m.cam_quat[self.cam] = gimbal_quat(*self.arb.gimbal)
            self._gimbal_written = self.arb.gimbal

    def _lower_stall(self, t: float):
        """Flag a non-safety lowering that the near-obstacle hold has stopped for ``LOWER_STALL_S``: the hold
        itself stays (rule 3), the flag tells the requester (WP28 / the operator) to escalate."""
        w = self.arb.winner()
        lowering = (self.arb.held_by == "near_obstacle" and w is not None and self.arb.target() < self.arb.ext - 1e-9)
        if not lowering:
            if self.lower_blocked is not None:
                self._dirty = True
            self._lower_held_since, self.lower_blocked = None, None
            return
        if self._lower_held_since is None:
            self._lower_held_since = t
        if self.lower_blocked is None and t - self._lower_held_since >= LOWER_STALL_S - 1e-9:
            self.lower_blocked = {"since": round(self._lower_held_since, 4), "source": w.source, "reason": w.reason,
                                  "geom": self.dist_geom, "kind": self.dist_kind, "ext_m": round(self.arb.ext, 4),
                                  "target_ext": round(self.arb.target(), 4)}
            self._dirty = True

    # -- 50 Hz
    def tick(self, m, d, t, bus):
        self.bind(m, d)
        self.bus = bus
        bus.values["mast_control"] = self
        if not self.ok:
            bus.values["mast"] = {"ext_m": 0.0, "has_mast": False}
            return
        n = bus.arrival_count(CMD_TOPIC)
        if n != self._cmd_seen:
            msgs, lost = bus.since(CMD_TOPIC, self._cmd_seen)
            self._cmd_seen = n
            if lost:
                self.cmd_lost += lost
                self._dirty = True
            for msg in msgs:                 # every message of this tick, in delivery order
                self.command(msg, t)
        self._regate(t)
        self._backstops(t)
        st = self.state()
        bus.values["mast"] = st
        key = (st["arbiter_source"], st["gated"], st["reason"], st["braked"], st["moving"], st["near"],
               st["strike_count"], st["held_by"], json.dumps(st["last_decision"], sort_keys=True), st["rejected"],
               st["cmd_lost"], st["lower_blocked"] is not None, st["brake_latched"], st["gimbal_clamped"])
        if self._dirty or key != self._pub_key or t + 1e-9 >= self._next_pub:
            bus.publish(STATE_TOPIC, st)
            self._pub_key, self._dirty = key, False
            self._next_pub = t + PUBLISH_PERIOD_S

    def _regate(self, t: float):
        """Rule 1 while a raise runs: the motion gates (e-stop, standing, cross-slope, wind, WP6 gates) every tick,
        plus the clearance when the raise resumes after a near-obstacle hold. A failure holds the raise at once
        (``held_by`` ``gate:<detail>``); still failing after ``REGATE_DROP_S`` the raise is dropped
        (``last_decision`` refused, detail ``regate:<detail>``) and the requester must submit again."""
        w = self.arb.raising()
        if w is None:
            self.arb.raise_hold = ""
            self._regate_since = None
            self._near_held = False
            return
        tgt = mc.clamp_ext(w.target_ext)
        dec = self.motion_gates(self.arb.ext, tgt, w)
        if dec and self._near_held and not self.near:
            dec = self.clearance(self.arb.ext, tgt)
            if dec:
                self._near_held = False
        if dec:
            self.arb.raise_hold = ""
            self._regate_since = None
            return
        detail = str(dec.values.get("detail", "")) if isinstance(dec.values, dict) else ""
        detail = detail or dec.gate
        if self._regate_since is None:
            self._regate_since = t
            self._dirty = True
        self.arb.raise_hold = detail
        if t - self._regate_since >= REGATE_DROP_S - 1e-9:
            self.arb.abort_raise(dec.reason or mc.GATE, detail, t)
            self._regate_since = None
            self._near_held = False
            self._dirty = True

    def _backstops(self, t: float):
        if self.arb.braked or self.arb.ext <= 1e-3:
            self._wind_since = None
            return
        w = self.arb.winner()
        if w is not None and w.source == "safety" and (w.target_ext or 0.0) <= 1e-9:
            return                           # a safety retract to stow is already running
        speed, gust = self._wind()
        if gust or speed >= WIND_RAISE_MAX_MPS:
            if self._wind_since is None:
                self._wind_since = t
            if self._standing() or t - self._wind_since >= WIND_STANCE_S - 1e-9:
                self._wind_since = None
                self.submit(MastRequest("safety", 0.0, "gust" if gust else "wind", t=float(t)))
                return
        else:
            self._wind_since = None
        roll, pitch = self._roll_pitch()
        if max(abs(roll), abs(pitch)) > math.radians(TILT_RETRACT_DEG):
            self.submit(MastRequest("safety", 0.0, "tilt", t=float(t)))

    def command(self, msg: Any, t: float) -> Optional[GateDecision]:
        """One ``/nandi/mast/cmd`` message; malformed input is counted in ``rejected``, never raised."""
        try:
            if isinstance(msg, str):
                msg = json.loads(msg)
            if not isinstance(msg, dict):
                raise ValueError("not an object")
            if "release" in msg:
                src = msg["release"]
                if src not in CMD_SOURCES:
                    raise ValueError(f"release of {src!r}")
                self.release(src)
                return None
            aim = None
            if "aim_deg" in msg:
                p, tl = msg["aim_deg"]
                aim = (math.radians(_num(p, "pan")), math.radians(_num(tl, "tilt")))
                if "source" not in msg:
                    self.aim(*aim)
                    return None
            src = msg.get("source")
            if src not in CMD_SOURCES:
                raise ValueError(f"source {src!r}")
            if "band" in msg:
                if msg["band"] not in BANDS:
                    raise ValueError(f"band {msg['band']!r}")
                target = BANDS[msg["band"]]
            else:
                target = _num(msg.get("target_ext"), "target_ext")
            gim = None
            if msg.get("gimbal_deg") is not None:
                gp, gt = msg["gimbal_deg"]
                gim = (math.radians(_num(gp, "pan")), math.radians(_num(gt, "tilt")))
            req = MastRequest(src, float(target), str(msg.get("reason") or "console"), gimbal=gim, t=float(t))
        except (ValueError, TypeError, KeyError) as exc:
            self.rejected += 1
            self._dirty = True
            self.last_reject = str(exc)
            return None
        dec = self.submit(req)
        if aim is not None and dec:
            self.aim(*aim)                   # the aim of a request applies only when the request is accepted
        return dec


def register(hooks):
    """``mast_step`` (every physics step, mutates the model) and ``mast`` (50 Hz, commands and state)."""
    rt = MastRuntime()
    hooks.register("mast_step", rt.substep, every_step=True, mutates_model=True, reset=rt.reset)
    hooks.register("mast", rt.tick, rate_hz=RATE_HZ, topics_in=(CMD_TOPIC,), reset=rt.tick_reset)
    return rt
