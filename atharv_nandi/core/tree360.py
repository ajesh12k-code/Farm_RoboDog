"""TREE 360 VIDEO recorder (WP25; ORCHARD-MVP-SPEC 1.2 step 2, 2.3 "Bands", 2.5, 5.1, 5.3; MISSION-PRESETS 0.6, 0.8, 2).

One ``Tree360Run`` drives one radius of TREE 360 around one registry tree, ticked at 50 Hz by the ``tree360`` hook
(``atharv_nandi/sim/hooks/tree360.py``) or a ``SimDriver`` task. The logic is pure Python behind a small IO object; ``SimIO``
binds it to MuJoCo and the hook bus (WP24 mast arbiter, WP5 trunk fit, WP6 motion inhibit) and is the only place
that touches ``mujoco``.

Sequence (MP 2 steps 1-8; times are the defaults of MP 0.8):

    approach   walk (crown stowed, <= 0.18 m/s) to the entry station at r1 = canopy_r + d with the steady orbit
               heading (or ``entry: teleport``, assisted ``station_teleport``, for station-tier tests)
    raise      stand, settle 1 s, open chunks, crown to the first band (mid, 3 s) via a ``preset`` request
    station    settle 1 s; gimbal pans to the trunk bearing with the band tilt (solved per tree, coverage.plan_bands);
               capture the tree-side ring camera (low) and the crown band it is at (0.5 s window); the crown
               **snakes** to the other crown band (mid <-> high, 5 s), settle 1 s, capture (0.5 s). Snake mode keeps
               the crown up on the arcs (one crown move per station). ``stop_and_go`` (OS 2.3 rule 7) raises to mid,
               captures, raises to high, captures and stows (8 s) before every arc
    arc        closed loop to the next station (``orbit.OrbitController``: omega = v/R_cmd + k_r e + k_b b,
               tolerance 0.10 m) at 0.15 m/s; R_meas from the WP5 low-plane trunk fit (``measure: physics``) or truth
               (``measure: assisted``, labelled). |e| > 0.15 m -> 0.10 m/s; |e| > 0.30 m -> stop, re-enter at the
               nearest station (2 x per orbit), then mark the gap and re-enter at the next station
    transit    two-sided only: stow, walk the A* path (``orbit.transit_path``) around the row to the other alley
    stow       crown to 0 (3 s from mid), release the preset source; chunks closed
    finish     strips per band (``privacy_filter.StripAssembler``), MP4 per band (``Mp4Assembler``), coverage,
               ``manifest.json`` (SHA-256 per chunk, Merkle root, ``controllers.manifest``), TreeRecord fragment
               (``nandi.tree/1`` coverage + media) and ``nandi.deepscan/1``

Interlocks: a person in the crown-scaled warning field (WP6 ``motion_inhibit.safety_fields.orbit_pause``) **pauses**
the run (gait stopped, every timer frozen, masking continues) until it clears; the console Pause / Resume does the
same without the person-hold timers. A protective / stability latch, an e-stop hold, a fall, an operator abort, a
walk that does not arrive (``approach_timeout`` / ``reentry_timeout`` / ``transit_timeout``, or the trunk off the
entry station before the raise: ``approach_off_station``), walking inhibited by WP6 for 30 s (``stability``), low
light (``low_light``), the arm off PARK (``arm_not_parked``), too little disk (``disk_low``; below 1.5 x 26 MB it
records Hybrid: stills only, no depth, no MP4), a world reset (``world_reset``) or an internal fault
(``internal_error:<type>``) halts it, ordered: stand, stow the crown and wait for it, release the preset source,
partial outputs sealed, status ``aborted``. ``nandi.deepscan/1`` is written and published only when the visit is
done and its coverage passes; otherwise the record asks for ``[revisit, tree360]``.

Frames: ``runs/<run_id>/tree360/<tree_id>/<band>/<seq>.jpg`` (+ ``<seq>_depth.png``, uint16 mm, masked pixels 0),
``frames.jsonl`` (pose tags: theta_tree, R, band, camera pose, RTK, gimbal, ext, steady, mask), ``strip_<band>.jpg``,
``<band>.mp4``, ``coverage.json``, ``plan.json``, ``manifest.json``; next to the tree directory
``deepscan_<tree_id>.json`` and ``record_<tree_id>.json`` (outside the manifest, they carry its root). Only
FilteredFrames are written (WP5 privacy filter on every frame).
"""
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from atharv_nandi import paths
from atharv_nandi.core import coverage as cv
from atharv_nandi.core import orbit as ob

SCHEMA_DEEPSCAN = "nandi.deepscan/1"
SCHEMA_TREE = "nandi.tree/1"
TENANT = "sim-nandi"
STOW_LENS_M = 1.10                 # lens height with the crown stowed (OS 2.3)
EXT_MAX_M = 1.60
V_TRANSIT_MPS = 0.18               # stowed walking cap (MP 0.3)
V_REENTRY_MPS = 0.10               # crown-up re-entry / approach legs near the tree
V_PER_CMD = ob.V_PER_CMD
W_PER_CMD = ob.W_PER_CMD
GOTO_TOL_M = 0.08
GOTO_HEAD_TOL_DEG = 4.0
CROWN_TOL_M = 0.005
CROWN_RATE_MPS = 0.2
STEADY_RATE_DEG_S = 2.0            # O-05: trunk rate < 2 deg/s ...
STEADY_CROWN_S = 1.0               # ... and the crown settled >= 1 s
FALL_TILT_DEG = 35.0
RESEAT_MAX = 2                     # in-place shuffles to re-seat a 6-foot stance before a raise
PAUSE_ASSIST_S = 30.0              # a person holding the orbit this long -> assist request (MP 0.4 step 3)
PAUSE_ABORT_S = 120.0              # ... and this long -> halt with person_hold
AZ_ERR_MAX_DEG = 3.0               # O-03
AZ_TRIM_DEG = 1.5                  # a station short by more than this gets one closed-loop trim before capture
AIM_APPLY_S = 0.06                 # gimbal aim -> capture (WP24 applies the aim in its substep)
RATE_TAU_S = 0.25                  # low-pass of the trunk yaw rate for O-05 (the gait sways at ~2 Hz)
BEARING_P95_MAX_DEG = 5.0
ORBIT_MAX_S = 165.0                # O-06 (snake)
RADIUS_RMS_MAX_M = 0.10            # O-02
BYTES_MAX = 30 * 1024 * 1024       # O-22 sim TREE 360 bytes per tree
RECORD_MAX_BYTES = 16 * 1024       # TreeRecord <= 16 KiB
FIT_MAX_AGE_S = 0.5
FIT_HOLD_S = 1.0
STATION_POS_TOL_M = 0.35          # trunk within this of the entry station before the raise ...
STATION_R_TOL_M = 0.30            # ... and |R - R_cmd| within the OS 5.3 stop threshold
WALK_BLOCK_SWITCH_S = 2.0         # snake: WP6 walk inhibit this long on an arc -> stow, stop-and-go
WALK_BLOCK_MAX_S = 30.0           # walking still inhibited this long -> halt ``stability``
EST_BYTES_PER_TREE = 26 * 1024 * 1024   # MP 2 sim estimate (computed)
EST_BYTES_HYBRID = 2 * 1024 * 1024      # keyframes only (no depth, no MP4)
DISK_FACTOR = 1.5                 # MP 0.2: free disk >= 1.5 x the preset estimate
ARM_PARK_TOL = 0.10               # rad / m per arm joint vs the home keyframe
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
ANALYSERS = ["H-02", "H-03", "H-04", "H-06"]
ANALYSERS_OPTIONAL = ["H-05"]
REASONS = ("triage", "operator", "zone_alert", "block_sample")
BANDS = cv.BANDS


@dataclass
class Tree360Config:
    """MISSION-PRESETS 0.8 params (+ sim knobs). ``from_params`` validates ranges and enums."""
    arc_mode: str = "auto"                  # auto | full | two_sided
    standoff_m: float = ob.STANDOFF_DEFAULT_M
    stations: int = 8
    bands: Tuple[str, ...] = BANDS
    mid_lens_m: float = 1.70
    high_lens_m: float = 2.70
    crown_walk: str = "snake"               # snake | stop_and_go
    settle_s: float = 1.0
    capture_s: float = 0.5
    v_mps: float = ob.V_ORBIT_MPS
    measure: str = "physics"                # physics (WP5 trunk fit) | assisted (truth, labelled)
    entry: str = "walk"                     # walk | teleport (assisted station_teleport)
    privacy: str = "assisted"               # PrivacyFilter mode
    capture: str = "auto"                   # auto | gl | geometry | perception
    reentry_max: int = ob.REENTRY_MAX
    width: int = 320
    height: int = 240
    jpeg_quality: int = 80
    write_depth: bool = True
    mp4: bool = True
    run_id: str = ""
    device_id: str = "nandi-sim-01"
    out_root: str = ""
    reason: str = "operator"
    priority: int = 2

    @classmethod
    def from_params(cls, params: Optional[dict] = None) -> "Tree360Config":
        params = dict(params or {})
        known = {f.name for f in fields(cls)}
        bad = sorted(set(params) - known)
        if bad:
            raise ValueError(f"tree360: unknown params {bad}")
        if "bands" in params:
            params["bands"] = tuple(params["bands"])
        cfg = cls(**params)
        cfg.validate()
        return cfg

    def validate(self):
        enums = {"arc_mode": ("auto", "full", "two_sided"), "crown_walk": ("snake", "stop_and_go"),
                 "measure": ("physics", "assisted"), "entry": ("walk", "teleport"),
                 "privacy": ("assisted", "physics"), "capture": ("auto", "gl", "geometry", "perception"),
                 "reason": REASONS}
        for k, allowed in enums.items():
            if getattr(self, k) not in allowed:
                raise ValueError(f"tree360: {k} {getattr(self, k)!r} not in {allowed}")
        if not 0.9 - 1e-9 <= float(self.standoff_m) <= 1.5 + 1e-9:
            raise ValueError("tree360: standoff_m must be 0.9..1.5 (MP 0.8)")
        if not 4 <= int(self.stations) <= 16:
            raise ValueError("tree360: stations must be 4..16 (MP 0.8)")
        if not 0.0 < float(self.v_mps) <= ob.V_ORBIT_MPS + 1e-9:
            raise ValueError("tree360: v_mps must be in (0, 0.15] (crown-up cap)")
        if not self.bands or any(b not in BANDS for b in self.bands):
            raise ValueError(f"tree360: bands must be a non-empty subset of {BANDS}")
        for k in ("mid_lens_m", "high_lens_m"):
            z = float(getattr(self, k))
            if not STOW_LENS_M <= z <= STOW_LENS_M + EXT_MAX_M + 1e-9:
                raise ValueError(f"tree360: {k} must be {STOW_LENS_M}..{STOW_LENS_M + EXT_MAX_M} m")
        if float(self.settle_s) < 0 or float(self.capture_s) <= 0:
            raise ValueError("tree360: settle_s >= 0 and capture_s > 0")
        # run_id / device_id become path parts and file names: one safe segment, no traversal
        for k in ("run_id", "device_id"):
            v = getattr(self, k)
            if not isinstance(v, str) or (v and (not RUN_ID_RE.match(v) or ".." in v)):
                raise ValueError(f"tree360: {k} must match [A-Za-z0-9_][A-Za-z0-9_.-]{{0,63}} without '..'")
        if not self.device_id:
            raise ValueError("tree360: device_id must not be empty")
        if not isinstance(self.out_root, str) or "\x00" in self.out_root:
            raise ValueError("tree360: out_root must be a path string")

    def ext_of(self, band: str) -> float:
        if band == "mid":
            return round(float(self.mid_lens_m) - STOW_LENS_M, 4)
        if band == "high":
            return round(float(self.high_lens_m) - STOW_LENS_M, 4)
        return 0.0

    def as_dict(self) -> dict:
        return {f.name: (list(getattr(self, f.name)) if f.name == "bands" else getattr(self, f.name))
                for f in fields(self)}


@dataclass
class CaptureResult:
    """One capture: world -> optical ``pose``, ``intr``, ``fovy_deg``; ``frame`` a FilteredFrame or None (geometry
    backend without pixels or a dropped frame); ``mask`` the privacy mask (H, W) used for coverage."""
    camera: str
    pose: Any
    intr: Any
    fovy_deg: float
    frame: Any = None
    mask: Optional[np.ndarray] = None
    dropped: bool = False
    reason: str = "ok"
    stats: Dict[str, Any] = field(default_factory=dict)
    cam_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_fwd: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    backend: str = "geometry"
    parcel: Optional[np.ndarray] = None       # WP5 parcel mask (coverage: depth-consistent only)
    depth_raw: Optional[np.ndarray] = None    # unmasked depth, in memory only (never written)


def mission_id(device_id: str, utc: Optional[float] = None) -> str:
    """Platform mission id ``{device}-{UTC yyyymmddThhmmZ}``."""
    return f"{device_id}-{time.strftime('%Y%m%dT%H%MZ', time.gmtime(utc if utc is not None else time.time()))}"


def default_run_id() -> str:
    return os.environ.get("NANDI_RUN_ID") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-tree360"


def default_out_root() -> str:
    return os.environ.get("NANDI_RUNS_DIR") or paths.RUNS_DIR


def disk_free(path: str) -> int:
    """Free bytes on the filesystem that ``path`` (or its nearest existing parent) is on."""
    p = os.path.abspath(path or ".")
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return int(shutil.disk_usage(p).free)


def p95(values: Sequence[float]) -> float:
    return float(np.percentile(np.asarray(values, float), 95)) if len(values) else 0.0


# ====================================================================== recorder (files)
class Recorder:
    """Writes the TREE 360 media of one tree: FilteredFrames only (``privacy_filter.require_filtered``)."""

    def __init__(self, tree_dir: str, bands: Sequence[str], cfg: Tree360Config):
        self.dir = tree_dir
        self.cfg = cfg
        self.bands = list(bands)
        if os.path.isdir(self.dir) and os.listdir(self.dir):
            k = 2                      # a second visit in the same run: keep the first one (never delete)
            while os.path.exists(f"{self.dir}.v{k}"):
                k += 1
            os.rename(self.dir, f"{self.dir}.v{k}")
        os.makedirs(self.dir, exist_ok=True)
        for b in self.bands:
            os.makedirs(os.path.join(self.dir, b), exist_ok=True)
        self.tags_path = os.path.join(self.dir, "frames.jsonl")
        self._tags = open(self.tags_path, "w", encoding="utf-8")
        self.frames: Dict[str, List[Any]] = {b: [] for b in self.bands}
        self.keyframes: List[str] = []
        self.closed = False

    def write_frame(self, band: str, seq: int, frame, tags: dict) -> List[str]:
        """Frame (if any) + depth + one tag line. Returns the written relative paths."""
        from atharv_nandi.core.privacy_filter import require_filtered
        paths = []
        if frame is not None:
            require_filtered(frame)
            rel = f"{band}/{seq:04d}.jpg"
            with open(os.path.join(self.dir, rel), "wb") as fh:
                fh.write(frame.jpeg(self.cfg.jpeg_quality))
            paths.append(rel)
            self.keyframes.append(rel)
            self.frames[band].append(frame)
            if self.cfg.write_depth and frame.depth is not None:
                from PIL import Image
                dmm = np.nan_to_num(np.asarray(frame.depth, np.float64) * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
                rel_d = f"{band}/{seq:04d}_depth.png"
                Image.fromarray(np.clip(dmm, 0, 65535).astype(np.uint16)).save(os.path.join(self.dir, rel_d))
                paths.append(rel_d)
        self._tags.write(json.dumps(dict(tags, files=paths), sort_keys=True, separators=(",", ":")) + "\n")
        self._tags.flush()
        return paths

    def write_json(self, name: str, obj: Any):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, default=_json_default)
            fh.write("\n")

    def assemble(self, stamp: float) -> Dict[str, Any]:
        """Strips and MP4s per band from the recorded FilteredFrames."""
        from atharv_nandi.core.privacy_filter import Mp4Assembler, StripAssembler
        out: Dict[str, Any] = {"strips": {}, "mp4": {}}
        for b, frs in self.frames.items():
            if not frs:
                continue
            st = StripAssembler(f"strip_{b}", height=120)
            for f in frs:
                st.add(f)
            strip = st.build(stamp)
            rel = f"strip_{b}.jpg"
            with open(os.path.join(self.dir, rel), "wb") as fh:
                fh.write(strip.jpeg(self.cfg.jpeg_quality))
            out["strips"][b] = rel
            if self.cfg.mp4:
                mp = Mp4Assembler(fps=4.0, quality=self.cfg.jpeg_quality)
                for f in frs:
                    if f.shape == frs[0].shape:
                        mp.add(f)
                rel_v = f"{b}.mp4"
                out["mp4"][b] = {"path": rel_v, "encoder": mp.write(os.path.join(self.dir, rel_v))}
        return out

    def close(self):
        if not self.closed:
            self._tags.close()
            self.closed = True

    def bytes_total(self) -> int:
        total = 0
        for root, _, files in os.walk(self.dir):
            for n in files:
                total += os.path.getsize(os.path.join(root, n))
        return total


def _json_default(o):
    for attr in ("tolist", "item"):
        fn = getattr(o, attr, None)
        if callable(fn):
            return fn()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


# ====================================================================== the run
PHASES = ("init", "approach", "raise", "station", "arc", "reentry", "transit", "stow", "finish", "done", "aborted")


class Tree360Run:
    """One TREE 360 visit (module docstring). ``tick()`` at 50 Hz; ``result`` once ``phase`` is done/aborted."""

    def __init__(self, entry: dict, registry: Sequence[dict], cfg: Tree360Config, io,
                 geofence: Optional[Sequence[Sequence[float]]] = None, clusters: Sequence[dict] = (),
                 canopy: Optional[cv.Canopy] = None):
        self.cfg, self.io = cfg, io
        self.entry = dict(entry)
        self.tree = ob.tree_of(entry)
        self.tree_id = self.tree["tree_id"]
        self.geofence = [list(map(float, p)) for p in geofence] if geofence else None
        self.registry = list(registry)
        nbrs = ob.neighbours_of(entry, self.registry)
        self.nbrs = nbrs
        self.min_nbr_gap = math.inf
        R0 = ob.orbit_radius(self.tree["canopy_r_m"], cfg.standoff_m)
        self.lidar_obstacles = ob.clusters_as_neighbours(clusters, self.tree["xy"], R0)
        self.plan = ob.plan_orbit(self.tree, nbrs + self.lidar_obstacles, cfg.standoff_m, cfg.stations,
                                  cfg.arc_mode, geofence=self.geofence)
        # stand-off actually flown (r1 is clamped to 1.2..3.0 m) -> per-tree band plan (OS 2.3)
        self.d = self.plan.R - self.tree["canopy_r_m"]
        self.bandplan = cv.plan_bands(self.d, mid_lens_m=cfg.mid_lens_m, high_lens_m=cfg.high_lens_m)
        self.crown_walk = cfg.crown_walk
        self.notes: List[str] = []
        if self.lidar_obstacles:
            self.notes.append(f"lidar_annulus_obstacles:{len(self.lidar_obstacles)}")
        crown_bands = [b for b in ("mid", "high") if b in cfg.bands]
        if self.crown_walk == "snake" and crown_bands:
            ext_walk = max(cfg.ext_of(b) for b in crown_bands)
            g = ob.crown_path_ok(self.plan, ext_walk, ob.basins_of(self.registry))
            if not g.allowed:
                self.crown_walk = "stop_and_go"
                self.notes.append(f"crown_walk:stop_and_go:{g.reason}")
        self.canopy = canopy if canopy is not None else cv.canopy_from_entry(dict(entry, geo={
            "x": self.tree["xy"][0], "y": self.tree["xy"][1]}))
        samples = io.surface_samples(self.canopy, self.bandplan) if hasattr(io, "surface_samples") else None
        self.cov = cv.CoverageAccumulator(self.canopy, self.bandplan, samples=samples,
                                          alley_sides=self.tree["alley_sides"])
        self.run_id = cfg.run_id or default_run_id()
        self.mission_id = mission_id(cfg.device_id)
        root = cfg.out_root or default_out_root()
        self.run_dir = os.path.join(root, self.run_id, "tree360")
        self.tree_dir = os.path.join(self.run_dir, self.tree_id)
        self.rec: Optional[Recorder] = None
        # state
        self.phase = "init"
        self.status = "running"
        self.abort_reason = ""
        self.clock = 0.0                 # run clock: sim time minus pauses
        self.t0: Optional[float] = None
        self.t_last: Optional[float] = None
        self.orbit_t0: Optional[float] = None
        self.orbit_t1: Optional[float] = None
        self.pause_s = 0.0
        self.pauses = 0
        self.paused = False
        self.pause_since: Optional[float] = None     # sim time a person started holding the orbit
        self.op_paused = False                       # console Pause / Resume (MP 2 Console)
        self.op_pauses = 0
        self._aborting = False
        self.deepscan_requested = False
        self.assist: Optional[dict] = None
        self.seq = 0
        self.k_station = -1
        self.station_now: Optional[ob.Station] = None
        self.theta_target: Optional[float] = None
        self.reentries = 0
        self.gaps: List[Tuple[float, float]] = []
        self.r_err_ctrl: List[float] = []
        self.r_err_truth: List[float] = []
        self.az_err: List[float] = []
        self.bearing_err: List[float] = []
        self.steady = [0, 0]
        self.fit_methods: Dict[str, int] = {}
        self.crown_events: List[dict] = []
        self.captured: Dict[int, set] = {}
        self.max_tilt = 0.0
        self.rate_f = 0.0                # trunk yaw rate (deg/s), low-passed over the capture window
        self.falls = 0
        self.person_frames = 0
        self.person_unmasked_px = 0
        self.crown_ext_t = 0.0           # run clock of the last crown motion
        self.centre_est = self.tree["xy"]
        self.fit_t: Optional[float] = None
        self.result: Optional[dict] = None
        self.orbit_value: dict = {"active": False}
        self._orbit_value_t = -1.0
        self._script = self._main()

    # ------------------------------------------------------------------ public
    @property
    def finished(self) -> bool:
        return self.phase in ("done", "aborted")

    def abort(self, reason: str = "operator_abort"):
        if not self.finished and not self.abort_reason:
            self.abort_reason = reason

    @property
    def aborting(self) -> bool:
        return self._aborting and not self.finished

    def pause(self, on: bool = True):
        """Console Pause / Resume: stand, every run timer frozen (no person-hold timers), resume where it was."""
        if self.finished or self._aborting:
            return
        if on and not self.op_paused:
            self.op_pauses += 1
        self.op_paused = bool(on)

    def tick(self):
        """One 50 Hz tick: interlocks, then one resume of the script (unless paused). Never raises: an internal
        fault becomes an ordered abort ``internal_error:<type>`` (a raising hook would safe-hold the robot)."""
        if self.finished:
            return
        try:
            self._tick()
        except Exception as exc:          # noqa: BLE001
            self._fail(exc)

    def _fail(self, exc: BaseException):
        why = f"internal_error:{type(exc).__name__}"
        self.notes.append(f"{why}: {exc}"[:240])
        if self._aborting:
            self._finish_safe("aborted")
        else:
            self._abort(why)

    def _tick(self):
        io = self.io
        now = io.now()
        if self.t0 is None:
            self.t0 = self.t_last = now
            self.io.on_start(self)
        if now < self.t_last - 1e-6:
            # sim time went backwards: /reset_world (robot home, mast reset); the run cannot go on
            self.t_last = now
            if not self._aborting:
                self.abort_reason = self.abort_reason or "world_reset"
                self._abort(self.abort_reason)
            else:
                self._finish_safe("aborted")
            return
        dt = max(0.0, now - self.t_last)
        self.t_last = now
        if self._aborting:
            self.clock += dt
            self._step_script()
            return
        tilt = io.tilt_deg()
        self.max_tilt = max(self.max_tilt, tilt)
        self.rate_f += (io.yaw_rate_deg() - self.rate_f) * min(1.0, dt / RATE_TAU_S)
        inh = io.inhibit()
        why = self.abort_reason
        if not why:
            if tilt > FALL_TILT_DEG:
                self.falls += 1
                why = "fall"
            elif inh.get("hold"):
                why = "estop" if inh.get("hold") == "estop" else f"hold:{inh.get('hold')}"
            elif inh.get("stop"):
                why = inh.get("stop_reason") or "person_hold"
        if why:
            self._abort(why)
            return
        person = bool(inh.get("pause")) and self.phase not in ("init", "approach", "finish")
        if person or (self.op_paused and self.phase not in ("init", "finish")):
            if not self.paused:
                self.paused = True
                self.pauses += 1
                io.stand()
            self.pause_s += dt
            if person:
                if self.pause_since is None:
                    self.pause_since = now
                if now - self.pause_since >= PAUSE_ASSIST_S and not self.assist:
                    self.assist = {"kind": "assist_request", "reason": "person_hold", "t": round(now, 3)}
                if now - self.pause_since >= PAUSE_ABORT_S - 1e-9:
                    self._abort("person_hold")
                    return
            else:
                self.pause_since = None
            self._update_orbit_value(now, force=False)
            return
        if self.paused:
            self.paused = False
            self.assist = None
            self.pause_since = None
        self.clock += dt
        self._step_script()
        if not self.finished and not self._aborting:
            self._update_orbit_value(now)

    def _step_script(self):
        """One resume of the current script (main or the abort stow)."""
        try:
            next(self._script)
        except StopIteration:
            if not self.finished:
                if self._aborting:
                    self._finish_safe("aborted")
                else:
                    self._abort("internal_error:script_ended")
        except _Abort as exc:
            if self._aborting:
                self._finish_safe("aborted")
            else:
                self._abort(str(exc))

    def progress(self) -> dict:
        """``/nandi/tree360/progress`` payload (console chips and live coverage, <= 2 Hz)."""
        summ = self.cov.summary()
        n_st = self.plan.n_capture
        return {"tree_id": self.tree_id, "phase": self.phase, "status": self.status, "mode": self.plan.mode,
                "crown_walk": self.crown_walk, "station": max(0, self.k_station + 1), "stations": n_st,
                "R_m": round(self.plan.R, 3), "standoff_m": round(self.d, 3), "gap_deg": round(self.gap_deg(), 1),
                "paused": self.paused, "pauses": self.pauses, "operator_paused": self.op_paused,
                "aborting": self.aborting, "frames": self.seq,
                "coverage_pct": summ["pct_by_band"], "masked_pct": summ["masked_pct"],
                "r_err_m": round(self.r_err_ctrl[-1], 3) if self.r_err_ctrl else None,
                "reentries": self.reentries, "abort_reason": self.abort_reason or None, "assist": self.assist,
                "orbit_s": round(self._orbit_s(), 2), "t": round(self.io.now(), 3)}

    def gap_deg(self) -> float:
        return self.plan.gap_deg + sum(math.degrees(b - a) for a, b in self.gaps)

    # ------------------------------------------------------------------ waits (run clock)
    def _wait(self, seconds: float) -> Iterator[None]:
        t_end = self.clock + float(seconds)
        while self.clock < t_end - 1e-9:
            yield

    # ------------------------------------------------------------------ geometry helpers
    def _ref(self) -> Tuple[Tuple[float, float], float]:
        xy, h = self.io.pose()
        return ob.ref_point(xy, h), h

    def _centre(self) -> Tuple[Tuple[float, float], str]:
        """Tree centre used by the controller: WP5 fit (physics) or truth (assisted); fit gaps fall back to the
        last good fit for ``FIT_HOLD_S``, then to the registry position (``registry_rtk``)."""
        if self.cfg.measure == "assisted":
            return self.io.truth_centre(self.tree_id), "truth"
        fit = self.io.trunk_fit(self.tree_id)
        now = self.io.now()
        if fit is not None and fit.get("ok") and fit.get("centre") is not None \
                and now - float(fit.get("t", now)) <= FIT_MAX_AGE_S:
            self.centre_est = (float(fit["centre"][0]), float(fit["centre"][1]))
            self.fit_t = now
            return self.centre_est, str(fit.get("method") or "fit")
        if self.fit_t is not None and now - self.fit_t <= FIT_HOLD_S:
            return self.centre_est, "fit_hold"
        return self.tree["xy"], "registry_rtk"

    def _theta_now(self) -> float:
        ref, _ = self._ref()
        c, _ = self._centre()
        return math.atan2(ref[1] - c[1], ref[0] - c[0])

    # ------------------------------------------------------------------ motion primitives
    def _stand(self, settle: float = 0.0) -> Iterator[None]:
        self.io.stand()
        if settle > 0:
            yield from self._wait(settle)

    def _walk_blocked(self) -> bool:
        """WP6 walk inhibit (``stability.walk / stance``): stand this tick (the caller yields)."""
        if self.io.inhibit().get("walk_block"):
            self.io.stand()
            return True
        return False

    def _require_arm_park(self):
        """MP 0.3 / MP 2 interlock: the arm is PARKed while walking."""
        fn = getattr(self.io, "arm_parked", None)
        if fn is not None and not fn():
            raise _Abort("arm_not_parked")

    def _goto(self, trunk_goal: Sequence[float], heading_goal: float, v_max: float,
              timeout_s: Optional[float] = None, fail: str = "goto_timeout") -> Iterator[None]:
        """Walk the trunk to ``trunk_goal`` (turn in place, walk, slow near the goal) and turn to ``heading_goal``.
        Not there within ``timeout_s`` (default from the distance) -> ``_Abort(fail)``: never carry on from
        wherever the robot happens to be (no cmd_vel bridge, stuck, slipping)."""
        io = self.io
        self._require_arm_park()
        gx, gy = float(trunk_goal[0]), float(trunk_goal[1])
        if timeout_s is None:
            (x0, y0), _ = io.pose()
            timeout_s = math.hypot(gx - x0, gy - y0) / (0.6 * v_max) + 30.0
        t_end = self.clock + timeout_s
        while True:
            (x, y), h = io.pose()
            dist = math.hypot(gx - x, gy - y)
            if dist <= GOTO_TOL_M:
                break
            ang = ob.wrap(math.atan2(gy - y, gx - x) - h)
            if dist < 0.25 and abs(ang) > math.radians(60):
                break                  # close, but the goal is beside / behind: the heading turn and the arc fix it
            if self.clock >= t_end:
                raise _Abort(fail)
            if self._walk_blocked():
                yield
                continue
            if abs(ang) > math.radians(25):
                io.drive(0.0, 0.6 if ang > 0 else -0.6)
            else:
                v = min(v_max, 0.08 + 0.5 * dist)
                io.drive(v / V_PER_CMD, max(-0.8, min(0.8, 1.5 * ang)))
            yield
        yield from self._turn_to(heading_goal, timeout_s=30.0, fail=fail)

    def _turn_to(self, heading_goal: float, timeout_s: float = 30.0, fail: str = "turn_timeout") -> Iterator[None]:
        io = self.io
        t_end = self.clock + timeout_s
        while True:
            _, h = io.pose()
            err = ob.wrap(heading_goal - h)
            if abs(err) <= math.radians(GOTO_HEAD_TOL_DEG):
                break
            if self.clock >= t_end:
                raise _Abort(fail)
            if self._walk_blocked():
                yield
                continue
            io.drive(0.0, max(-0.6, min(0.6, 1.2 * err)) if abs(err) < 0.3 else (0.6 if err > 0 else -0.6))
            yield
        yield from self._stand(0.0)

    def _check_station(self, st: ob.Station, fail: str):
        """Before a raise: the trunk is at the station and the radius within the OS 5.3 stop threshold."""
        (x, y), _ = self.io.pose()
        sx, sy = st.trunk_xy()
        ref, _ = self._ref()
        c, _ = self._centre()
        r_err = math.hypot(ref[0] - c[0], ref[1] - c[1]) - self.plan.R
        off = math.hypot(sx - x, sy - y)
        if off > STATION_POS_TOL_M or abs(r_err) > STATION_R_TOL_M:
            self.notes.append(f"{fail}:off_m={off:.2f}:r_err_m={r_err:+.2f}")
            raise _Abort(fail)

    def _crown_to(self, band_or_ext, why: str, tries_s: float = 6.0) -> Iterator[None]:
        """Crown to a band (preset request to the WP24 arbiter); waits until it is there. A refusal is retried
        (standing and gates can come green) for ``tries_s``, then logged; the run goes on without that band."""
        io = self.io
        ext = self.cfg.ext_of(band_or_ext) if isinstance(band_or_ext, str) else float(band_or_ext)
        st = io.mast_state()
        if abs(float(st.get("ext_m", 0.0)) - ext) <= CROWN_TOL_M:
            return
        t_end = self.clock + tries_s
        dec = None
        reseats = 0
        while True:
            dec = io.mast_request(ext, f"tree360 {self.tree_id} {why}")
            if dec.get("allowed"):
                break
            if dec.get("reason") == "not_standing" and reseats < RESEAT_MAX and self.clock >= t_end - tries_s + 1.0:
                # a foot off the ground (basin rim, lateral): shuffle in place to re-seat the 6-foot stance
                reseats += 1
                self.crown_events.append({"t": round(io.now(), 3), "event": "reseat", "target_ext": ext})
                t_sh = self.clock + 0.6
                while self.clock < t_sh:
                    io.drive(0.0, 0.3 if reseats % 2 else -0.3)
                    yield
                yield from self._settle()
                t_end = self.clock + tries_s
                continue
            if self.clock >= t_end:
                self.crown_events.append({"t": round(io.now(), 3), "event": "refused", "target_ext": ext,
                                          "reason": dec.get("reason"), "detail": dec.get("detail")})
                return
            yield from self._wait(0.5)
        timeout = abs(float(st.get("ext_m", 0.0)) - ext) / CROWN_RATE_MPS + 3.0
        t_end = self.clock + timeout
        while self.clock < t_end:
            st = io.mast_state()
            if abs(float(st.get("ext_m", 0.0)) - ext) <= CROWN_TOL_M and not st.get("moving"):
                break
            if st.get("arbiter_source") in ("safety", "estop", "recovery"):
                self.crown_events.append({"t": round(io.now(), 3), "event": "preempted",
                                          "source": st.get("arbiter_source"), "reason": st.get("arbiter_reason")})
                break
            yield
        self.crown_ext_t = self.clock

    def _crown_band(self) -> Optional[str]:
        """The crown band the head is at now (None if between bands or stowed)."""
        ext = float(self.io.mast_state().get("ext_m", 0.0))
        for b in ("mid", "high"):
            if b in self.cfg.bands and abs(ext - self.cfg.ext_of(b)) <= 0.01:
                return b
        return None

    # ------------------------------------------------------------------ capture
    def _require_light(self):
        """MP 0.2 camera presets: light >= threshold; MP 2 failure row: low light -> abort ``low_light``."""
        fn = getattr(self.io, "light_ok", None)
        if fn is not None and not fn():
            raise _Abort("low_light")

    def _capture(self, st: ob.Station, bands: Sequence[str]):
        io = self.io
        self._require_light()
        centre, _ = self._centre()
        ref, h = self._ref()
        theta = math.atan2(ref[1] - centre[1], ref[0] - centre[0])
        done = self.captured.setdefault(st.k, set())
        for band in bands:
            if band in done:
                continue
            if band == "low":
                cam = io.ring_facing(self.tree["xy"])
            else:
                if self._crown_band() != band:
                    continue
                cam = io.crown_camera()
            res: CaptureResult = io.capture(cam, band, self.seq, self.tree_id)
            fwd = np.asarray(res.cam_fwd, float)
            tc = np.asarray(self.io.truth_centre(self.tree_id), float)
            to_tree = tc - np.asarray(res.cam_pos[:2], float)
            bear_err = abs(math.degrees(math.atan2(fwd[0] * to_tree[1] - fwd[1] * to_tree[0],
                                                   fwd[0] * to_tree[0] + fwd[1] * to_tree[1])))
            if band != "low":
                self.bearing_err.append(bear_err)
            az_err = abs(math.degrees(ob.wrap(theta - st.theta)))
            self.az_err.append(az_err)
            rate = self.rate_f
            steady = rate < STEADY_RATE_DEG_S and (band == "low" or self.clock - self.crown_ext_t >= STEADY_CROWN_S - 0.03)
            self.steady[0] += int(steady)
            self.steady[1] += 1
            rec = self.cov.add_frame(band, res.pose, res.intr, res.fovy_deg, io.coverage_ray_fn(),
                                     mask=res.mask, dropped=res.dropped, side=st.side, seq=self.seq,
                                     parcel=res.parcel, depth=res.depth_raw)
            ms = io.mast_state()
            tags = {"seq": self.seq, "tree_id": self.tree_id, "band": band, "camera": cam, "t": round(io.now(), 4),
                    "station": st.k, "side": st.side, "theta_tree_deg": round(math.degrees(theta), 2),
                    "theta_plan_deg": round(math.degrees(st.theta), 2), "az_err_deg": round(az_err, 2),
                    "R_m": round(math.hypot(ref[0] - centre[0], ref[1] - centre[1]), 4), "R_cmd_m": round(self.plan.R, 4),
                    "cam_pos": [round(float(v), 4) for v in res.cam_pos],
                    "cam_fwd": [round(float(v), 4) for v in res.cam_fwd],
                    "gimbal_deg": ms.get("gimbal_deg"), "ext_m": ms.get("ext_m"), "lens_z": ms.get("lens_z"),
                    "rtk": io.rtk(), "steady": bool(steady), "yaw_rate_deg_s": round(rate, 3),
                    "bearing_err_deg": round(bear_err, 2), "dropped": bool(res.dropped), "reason": res.reason,
                    "masked_pct": res.stats.get("masked_pct"), "backend": res.backend,
                    "person_px": res.stats.get("person_px"), "person_unmasked_px": res.stats.get("person_unmasked_px"),
                    "seen": rec["seen"], "occluded": rec["occluded"], "masked": rec["masked"]}
            if res.stats.get("person_px"):
                self.person_frames += 1
            self.person_unmasked_px += int(res.stats.get("person_unmasked_px") or 0)
            if self.rec is not None:
                self.rec.write_frame(band, self.seq, res.frame, tags)
            done.add(band)
            self.seq += 1

    def _aim(self, band: str):
        """Gimbal: pan to the trunk bearing (head frame = trunk heading), the band's solved tilt."""
        if band not in ("mid", "high"):
            return
        centre, _ = self._centre()
        ref, h = self._ref()
        cxy = self.io.camera_xy(self.io.crown_camera()) if hasattr(self.io, "camera_xy") else ref
        pan = ob.wrap(math.atan2(centre[1] - cxy[1], centre[0] - cxy[0]) - h)
        self.io.aim(pan, math.radians(self.bandplan.tilt(band)))

    # ------------------------------------------------------------------ the script
    def _main(self) -> Iterator[None]:
        cfg = self.cfg
        if self.plan.mode == "blocked":
            raise _Abort("orbit_blocked")
        # MP 2 preconditions: light, arm PARK, disk >= 1.5 x the estimate (else Record -> Hybrid, else refuse)
        self._require_light()
        self._require_arm_park()
        free = disk_free(os.path.dirname(self.run_dir))
        if free < DISK_FACTOR * EST_BYTES_PER_TREE:
            if free < DISK_FACTOR * EST_BYTES_HYBRID:
                raise _Abort("disk_low")
            # Hybrid: keyframe stills only (the station frames), no depth PNGs, no MP4 (platform Record -> Hybrid)
            self.cfg = cfg = replace(cfg, write_depth=False, mp4=False)
            self.notes.append(f"disk_low:hybrid:free_mb={free / 2 ** 20:.1f}")
        self.rec = Recorder(self.tree_dir, cfg.bands, cfg)
        self.rec.write_json("plan.json", {"orbit": self.plan.as_dict(), "bands": self.bandplan.as_dict(),
                                          "config": cfg.as_dict(), "crown_walk": self.crown_walk,
                                          "notes": self.notes})
        self.io.request_fit(self.tree)
        first = self.plan.stations[0][0]
        self.phase = "approach"
        if cfg.entry == "teleport":
            self.io.teleport(first.trunk_xy(), first.heading)
            yield from self._wait(1.5)
        else:
            yield from self._goto(first.trunk_xy(), first.heading, V_TRANSIT_MPS, fail="approach_timeout")
        yield from self._stand(cfg.settle_s)
        self._check_station(first, "approach_off_station")
        self.orbit_t0 = self.clock
        for ai, arc in enumerate(self.plan.stations):
            if ai > 0:
                yield from self._transit(arc[0])
                self._check_station(arc[0], "transit_off_station")
            self.phase = "raise"
            if self.crown_walk == "snake":
                first_band = "mid" if "mid" in cfg.bands else ("high" if "high" in cfg.bands else None)
                if first_band:
                    self._aim(first_band)
                    yield from self._crown_to(first_band, "raise")
            for j, st in enumerate(arc):
                if j > 0:
                    yield from self._arc_to(st, arc)
                self.station_now, self.k_station = st, st.k
                if st.capture:
                    yield from self._station(st, arc)
        self.phase = "stow"
        yield from self._stand(0.0)
        yield from self._crown_to(0.0, "stow")
        self.io.mast_release()
        self.orbit_t1 = self.clock
        self.phase = "finish"
        self._finish("done")
        return

    def _settle(self, extra_max_s: float = 1.5) -> Iterator[None]:
        """Settle ``settle_s`` after a stop, then up to ``extra_max_s`` more until the trunk is steady (O-05: yaw rate
        < 2 deg/s; the gait finishes its cycle after the stop)."""
        yield from self._stand(self.cfg.settle_s)
        t_end = self.clock + extra_max_s
        while self.clock < t_end and (self.rate_f >= 0.75 * STEADY_RATE_DEG_S
                                      or self.io.velocity()[0] > 0.02):
            yield

    def _station(self, st: ob.Station, arc: Sequence[ob.Station] = ()) -> Iterator[None]:
        cfg = self.cfg
        self.phase = "station"
        yield from self._settle()
        if arc and ob.wrap(self._theta_now() - st.theta) < -math.radians(AZ_TRIM_DEG):
            yield from self._arc_to(st, arc)          # short of the station (O-03 <= 3 deg): one closed-loop trim
            self.phase = "station"
            yield from self._settle()
        if self.crown_walk == "stop_and_go":
            seq = [b for b in ("mid", "high") if b in cfg.bands]
            if not seq:
                self._capture(st, ["low"])
                yield from self._wait(cfg.capture_s)
                return
            for i, b in enumerate(seq):
                self._aim(b)
                yield from self._crown_to(b, f"station {st.k}")
                if self.clock - self.crown_ext_t < cfg.settle_s:
                    yield from self._wait(cfg.settle_s - (self.clock - self.crown_ext_t))
                self._capture(st, (["low"] if i == 0 and "low" in cfg.bands else []) + [b])
                yield from self._wait(cfg.capture_s)
            yield from self._crown_to(0.0, f"stow before arc {st.k}")
            return
        cur = self._crown_band()
        first = (["low"] if "low" in cfg.bands else []) + ([cur] if cur else [])
        if cur:
            self._aim(cur)
            yield from self._wait(AIM_APPLY_S)      # the arbiter writes cam_quat on the next physics steps
        self._capture(st, first)
        yield from self._wait(cfg.capture_s)
        others = [b for b in ("mid", "high") if b in cfg.bands and b != cur]
        for b in others:
            self._aim(b)
            yield from self._crown_to(b, f"snake {st.k}")
            yield from self._wait(cfg.settle_s)
            self._capture(st, [b])
            yield from self._wait(cfg.capture_s)
        if "low" in cfg.bands and "low" not in self.captured.get(st.k, set()):
            self._capture(st, ["low"])

    def _arc_to(self, st: ob.Station, arc: Sequence[ob.Station]) -> Iterator[None]:
        """Closed-loop arc to station ``st`` (unwrapped theta), radius rules of OS 5.3."""
        cfg, io = self.cfg, self.io
        self.phase = "arc"
        self.theta_target = st.theta
        self._require_arm_park()
        while True:
            ctl = ob.OrbitController(self.plan.centre, self.plan.R, cfg.v_mps)
            th_now = self._theta_now()
            if ob.wrap(th_now - st.theta) > -math.radians(1.0):
                return                      # already at (or just past) the station
            # unwrap the current azimuth into the arc's frame (just below the target)
            th_u = st.theta + ob.wrap(th_now - st.theta)
            if th_u > st.theta + 1e-6:
                th_u -= 2 * math.pi
            ctl.start(th_u, st.theta)
            stopped = False
            blocked_t = None
            dt = 1.0 / 50.0
            v_f = w_f = None
            while True:
                if io.inhibit().get("walk_block"):
                    # WP6 holds walking (tripod margin, wind, stance): stand while it holds (never drive)
                    io.stand()
                    if blocked_t is None:
                        blocked_t = self.clock
                    if self.crown_walk == "snake" and self.clock - blocked_t > WALK_BLOCK_SWITCH_S:
                        # crown-up walking held: stow and continue stop-and-go (OS 2.3 rule 7)
                        yield from self._crown_to(0.0, "walk_inhibit stow")
                        self.crown_walk = "stop_and_go"
                        self.notes.append("crown_walk:stop_and_go:walk_inhibit")
                        blocked_t = self.clock
                        continue
                    if self.clock - blocked_t > WALK_BLOCK_MAX_S:
                        raise _Abort("stability")
                    yield
                    continue
                if blocked_t is not None:
                    blocked_t = None
                    ctl.gait.reset()             # no integrator wind-up from the standing ticks
                    v_f = w_f = None
                centre, method = self._centre()
                self.fit_methods[method] = self.fit_methods.get(method, 0) + 1
                ref, h = self._ref()
                v_m, w_m = io.velocity()
                v_f = v_m if v_f is None else v_f + 0.2 * (v_m - v_f)
                w_f = w_m if w_f is None else w_f + 0.2 * (w_m - w_f)
                step = ctl.step(ref, h, dt, v_meas=v_f, w_meas=w_f, centre=centre)
                self.r_err_ctrl.append(step.e_r)
                for nb in self.nbrs:
                    g = math.hypot(ref[0] - nb["xy"][0], ref[1] - nb["xy"][1]) - nb["canopy_r_m"] - ob.HULL_R_M
                    self.min_nbr_gap = min(self.min_nbr_gap, g)
                tc = io.truth_centre(self.tree_id)
                self.r_err_truth.append(math.hypot(ref[0] - tc[0], ref[1] - tc[1]) - self.plan.R)
                if step.done:
                    break
                if step.stop:
                    stopped = True
                    break
                io.drive(step.vx, step.yaw)
                yield
            yield from self._stand(0.0)
            if not stopped:
                return
            # |e| > 0.30 m: stop, re-enter at the nearest station (2 x), then mark the gap
            self.phase = "reentry"
            th = self._theta_now()
            if self.reentries < cfg.reentry_max:
                self.reentries += 1
                near = min(arc, key=lambda s: abs(ob.wrap(s.theta - th)))
                yield from self._goto(near.trunk_xy(), near.heading, V_REENTRY_MPS, fail="reentry_timeout")
                if near.k == st.k:
                    return
                self.phase = "arc"
                continue
            prev_th = st.theta + ob.wrap(th - st.theta)
            if prev_th < st.theta:
                self.gaps.append((prev_th, st.theta))
            self.notes.append(f"gap_marked:{st.k}")
            yield from self._goto(st.trunk_xy(), st.heading, V_REENTRY_MPS, fail="reentry_timeout")
            return

    def _transit(self, st: ob.Station) -> Iterator[None]:
        """Two-sided: stow, walk around the row to the other alley's first station (crown stowed)."""
        self.phase = "transit"
        yield from self._stand(0.0)
        yield from self._crown_to(0.0, "transit")
        (x, y), _ = self.io.pose()
        obst = [(t["xy"][0], t["xy"][1], t["canopy_r_m"]) for t in ob.neighbours_of(self.entry, self.registry, 12.0)]
        obst.append((self.tree["xy"][0], self.tree["xy"][1], self.tree["canopy_r_m"]))
        obst += [(o["xy"][0], o["xy"][1], o["canopy_r_m"]) for o in self.lidar_obstacles]
        goal = st.trunk_xy()
        path = ob.transit_path((x, y), goal, obst, self.geofence)
        if path is None:
            raise _Abort("transit_no_path")
        self.notes.append(f"transit_m:{ob.path_length((x, y), path):.2f}")
        for i, wp in enumerate(path):
            last = i == len(path) - 1
            if last:
                yield from self._goto(wp, st.heading, V_TRANSIT_MPS, timeout_s=120.0, fail="transit_timeout")
            else:
                yield from self._goto_pass(wp, V_TRANSIT_MPS)

    def _goto_pass(self, wp: Sequence[float], v_max: float, tol: float = 0.25) -> Iterator[None]:
        io = self.io
        self._require_arm_park()
        t_end = self.clock + 120.0
        while True:
            (x, y), h = io.pose()
            dist = math.hypot(wp[0] - x, wp[1] - y)
            if dist <= tol:
                return
            if self.clock >= t_end:
                raise _Abort("transit_timeout")
            if self._walk_blocked():
                yield
                continue
            ang = ob.wrap(math.atan2(wp[1] - y, wp[0] - x) - h)
            if abs(ang) > math.radians(30):
                io.drive(0.0, 0.6 if ang > 0 else -0.6)
            else:
                io.drive(v_max / V_PER_CMD, max(-0.8, min(0.8, 1.5 * ang)))
            yield

    # ------------------------------------------------------------------ end
    def _abort(self, reason: str):
        """Ordered halt (MP 2 "Abort = ordered stop, stow"): stand now, then the abort script stows the crown
        (waits until it is at 0, or until the arbiter refuses / a safety source preempts it), releases the preset
        source and only then finishes ``aborted``. Never raises."""
        if self.finished or self._aborting:
            return
        self.abort_reason = self.abort_reason or reason
        self._aborting = True
        self.paused = False
        self.orbit_value = {"active": False}
        if self.phase not in ("init", "finish"):
            self.phase = "stow"
        try:
            self.io.stand()
        except Exception as exc:          # noqa: BLE001
            self.notes.append(f"abort_stand_error:{type(exc).__name__}")
        self._script = self._abort_script(self.abort_reason)
        self._step_script()

    def _abort_script(self, reason: str) -> Iterator[None]:
        io = self.io
        stowed = False
        try:
            ext0 = float(io.mast_state().get("ext_m", 0.0))
        except Exception:                 # noqa: BLE001
            ext0 = 0.0
        if ext0 <= CROWN_TOL_M:
            stowed = True
        else:
            try:
                dec = io.mast_request(0.0, f"tree360 {self.tree_id} abort {reason}")
            except Exception as exc:      # noqa: BLE001 - no mast control: nothing to stow with
                dec = {"allowed": False, "reason": f"{type(exc).__name__}: {exc}"}
            if not dec.get("allowed"):
                # the e-stop brake refuses it: the mast holds where it is (ordered, not stowed)
                self.crown_events.append({"t": round(io.now(), 3), "event": "abort_stow_refused",
                                          "reason": dec.get("reason")})
            else:
                t_end = self.clock + ext0 / CROWN_RATE_MPS + 3.0
                while self.clock < t_end:
                    st = io.mast_state()
                    if abs(float(st.get("ext_m", 0.0))) <= CROWN_TOL_M and not st.get("moving"):
                        stowed = True
                        break
                    if st.get("arbiter_source") in ("safety", "estop", "recovery"):
                        self.crown_events.append({"t": round(io.now(), 3), "event": "preempted",
                                                  "source": st.get("arbiter_source"),
                                                  "reason": st.get("arbiter_reason")})
                        break
                    io.stand()
                    yield
        try:
            io.mast_release()             # never leave the preset request behind in the arbiter
        except Exception:                 # noqa: BLE001
            pass
        self.notes.append("abort_stowed" if stowed else "abort_not_stowed")
        self._finish_safe("aborted")

    def _finish_safe(self, status: str):
        """``_finish`` that cannot raise: a fault while writing the outputs still ends the run (minimal result)."""
        if self.finished:
            return
        try:
            self._finish(status)
        except Exception as exc:          # noqa: BLE001
            self.status = status
            self.notes.append(f"finish_error:{type(exc).__name__}: {exc}"[:240])
            try:
                if self.rec is not None:
                    self.rec.close()
            except Exception:             # noqa: BLE001
                pass
            self.result = {"status": status, "abort_reason": self.abort_reason or None, "tree_id": self.tree_id,
                           "mode": self.plan.mode, "crown_walk": self.crown_walk, "tree_dir": None, "kpi": {},
                           "coverage": None, "coverage_pass": False, "coverage_why": "finish_error", "files": {},
                           "merkle_root": None, "record": None, "deepscan": None, "deepscan_requested": False,
                           "plan": self.plan.as_dict(), "bands": self.bandplan.as_dict(), "notes": self.notes,
                           "crown_events": self.crown_events, "error": f"{type(exc).__name__}: {exc}"}
            self.orbit_value = {"active": False}
            self.phase = "aborted" if status == "aborted" else "done"

    def _orbit_s(self) -> float:
        if self.orbit_t0 is None:
            return 0.0
        end = self.orbit_t1 if self.orbit_t1 is not None else self.clock
        return end - self.orbit_t0

    def _update_orbit_value(self, now: float, force: bool = True):
        """``bus.values["orbit"]`` for WP6: active from the raise to the stow, the planned arc in the base frame."""
        active = self.phase in ("raise", "station", "arc", "reentry") and not self.finished
        if not active:
            self.orbit_value = {"active": False}
            return
        if now - self._orbit_value_t < 0.5 and self.orbit_value.get("active"):
            return
        self._orbit_value_t = now
        ref, h = self._ref()
        th = self._theta_now()
        to = self.theta_target if (self.theta_target is not None and self.phase == "arc") else None
        if to is None:
            arc = [[0.0, 0.0, 0.0]]
        else:
            th_u = to + ob.wrap(th - to)
            if th_u > to:
                th_u -= 2 * math.pi
            arc = ob.arc_poses_base(ref, h, self.plan.centre, self.plan.R, th_u, to)
        self.orbit_value = {"active": True, "tree_id": self.tree_id, "arc": arc}

    def _finish(self, status: str):
        self.status = status
        self.phase = "finish"
        self.orbit_value = {"active": False}
        self.io.cancel_fit()
        summ = self.cov.summary()
        mode = self.plan.mode
        cov_ok, cov_why = cv.coverage_pass(summ, "full" if mode == "full" else "two_sided")
        kpi = self._kpis(summ, cov_ok, cov_why)
        files: Dict[str, Any] = {}
        root, chunks = None, []
        if self.rec is not None:
            try:
                files = self.rec.assemble(self.io.now())
            except ValueError as exc:
                files = {"error": str(exc)}
            self.rec.write_json("coverage.json", {"summary": summ, "pass": cov_ok, "why": cov_why,
                                                  "frames": self.cov.log})
            self.rec.close()
            from atharv_nandi.core import manifest as mf
            man = mf.write_manifest(self.tree_dir, {"run_id": self.run_id, "device_id": self.cfg.device_id,
                                                    "tenant": TENANT, "site": self.entry.get("block_id"),
                                                    "sw": "nandi-sim wp25"}, final=True,
                                    closed_ms=int(time.time() * 1000))
            root, chunks = man["merkle_root"], man["chunks"]
            kpi["bytes"] = self.rec.bytes_total()
            kpi["bytes_ok"] = kpi["bytes"] <= BYTES_MAX
        # Deep Scan is requested only for a finished visit whose coverage passes (O-01); otherwise the
        # record asks for a revisit and no request is written or published
        self.deepscan_requested = status == "done" and bool(cov_ok)
        record = self._tree_record(summ, root)
        deepscan = self._deepscan(summ, root, chunks)
        if self.rec is not None:
            os.makedirs(self.run_dir, exist_ok=True)
            outs = [(f"record_{self.tree_id}.json", record)]
            if self.deepscan_requested:
                outs.append((f"deepscan_{self.tree_id}.json", deepscan))
            for name, obj in outs:
                with open(os.path.join(self.run_dir, name), "w", encoding="utf-8") as fh:
                    json.dump(obj, fh, indent=1, sort_keys=True, default=_json_default)
                    fh.write("\n")
        self.result = {"status": status, "abort_reason": self.abort_reason or None, "tree_id": self.tree_id,
                       "mode": mode, "crown_walk": self.crown_walk, "tree_dir": self.tree_dir if self.rec else None,
                       "kpi": kpi, "coverage": summ, "coverage_pass": cov_ok, "coverage_why": cov_why,
                       "files": files, "merkle_root": root, "record": record, "deepscan": deepscan,
                       "deepscan_requested": self.deepscan_requested,
                       "plan": self.plan.as_dict(), "bands": self.bandplan.as_dict(), "notes": self.notes,
                       "crown_events": self.crown_events}
        self.phase = "aborted" if status == "aborted" else "done"

    def _kpis(self, summ: dict, cov_ok: bool, cov_why: str) -> dict:
        orbit_s = self._orbit_s()
        rt = self.r_err_truth
        return {"orbit_s": round(orbit_s, 2), "orbit_s_incl_pause": round(orbit_s + self.pause_s, 2),
                "pause_s": round(self.pause_s, 2), "pauses": self.pauses,
                "orbit_ok": orbit_s <= ORBIT_MAX_S if self.crown_walk == "snake" else None,
                "radius_rms_truth_m": round(ob.rms(rt), 4), "radius_max_truth_m": round(max(map(abs, rt)), 4) if rt else 0.0,
                "radius_rms_ctrl_m": round(ob.rms(self.r_err_ctrl), 4),
                "radius_ok": ob.rms(rt) <= RADIUS_RMS_MAX_M if rt else None, "radius_n": len(rt),
                "az_err_max_deg": round(max(self.az_err), 2) if self.az_err else None,
                "az_ok": (max(self.az_err) <= AZ_ERR_MAX_DEG) if self.az_err else None,
                "bearing_p95_deg": round(p95(self.bearing_err), 2) if self.bearing_err else None,
                "bearing_ok": (p95(self.bearing_err) <= BEARING_P95_MAX_DEG) if self.bearing_err else None,
                "steady_pct": round(100.0 * self.steady[0] / self.steady[1], 1) if self.steady[1] else None,
                "frames": self.seq, "dropped": sum(1 for r in self.cov.log if r.get("dropped")),
                "coverage_ok": cov_ok, "coverage_why": cov_why, "reentries": self.reentries,
                "gaps": len(self.gaps), "gap_deg": round(self.gap_deg(), 1), "falls": self.falls,
                "max_tilt_deg": round(self.max_tilt, 2), "fit_methods": dict(self.fit_methods),
                "min_neighbour_gap_m": round(self.min_nbr_gap, 3) if math.isfinite(self.min_nbr_gap) else None,
                "crown_events": len(self.crown_events), "person_frames": self.person_frames,
                "person_unmasked_px": self.person_unmasked_px if self.io.backend_name() == "gl" else None}

    def _assisted(self) -> List[str]:
        out = ["kinematic_mast", "virtual_gimbal", "coverage_model_rays"]
        if self.cfg.entry == "teleport":
            out.append("station_teleport")
        if self.cfg.measure == "assisted":
            out.append("orbit_truth_radius")
        if self.cfg.privacy == "assisted":
            out.append("privacy_truth_mask")
        backend = self.io.backend_name()
        if backend == "geometry":
            out.append("capture_geometry_only")
        elif backend == "perception":
            # the pixels stay in the WP5 node (no keyframes here); coverage is the geometry model's, not measured
            out += ["capture_perception_remote", "coverage_geometry_only"]
        return out

    def _tree_record(self, summ: dict, root: Optional[str]) -> dict:
        e = self.entry
        rec = {"schema": SCHEMA_TREE, "run_id": self.run_id, "mission_id": self.mission_id,
               "device_id": self.cfg.device_id, "tenant": TENANT, "data_provenance": "simulated",
               "block_id": e.get("block_id"), "row_id": e.get("row_id"), "tree_id": self.tree_id,
               "physical_tag": e.get("physical_tag"), "crop": e.get("crop"), "cultivar": e.get("cultivar"),
               "form": e.get("form"), "geo": {"x": self.tree["xy"][0], "y": self.tree["xy"][1],
                                             "frame": (e.get("geo") or {}).get("frame", "nandi_local_v1")},
               "visit_t": round(self.io.now(), 3), "presets": ["tree360"],
               "coverage": {"az_deg_by_band": summ["az_deg_by_band"],
                            "sides": {b: v["sides"] for b, v in summ["bands"].items()},
                            "mode": "full" if self.plan.mode == "full" else "two_sided",
                            "frames": self.seq, "seen": summ["seen"], "masked_pct": summ["masked_pct"],
                            "pct_by_band": summ["pct_by_band"], "gap_deg": round(self.gap_deg(), 1)},
               "media": {"manifest_root": root, "keyframe_paths": list(self.rec.keyframes[:24]) if self.rec else []},
               "detector": "SimTruth", "assisted": self._assisted(),
               "next_actions": ["deep_scan"] if self.deepscan_requested else ["revisit", "tree360"]}
        return rec

    def _deepscan(self, summ: dict, root: Optional[str], chunks: Sequence[dict]) -> dict:
        return {"schema": SCHEMA_DEEPSCAN, "mission_id": self.mission_id, "run_id": self.run_id,
                "device_id": self.cfg.device_id, "tenant": TENANT, "data_provenance": "simulated",
                "tree_id": self.tree_id, "physical_tag": self.entry.get("physical_tag"), "crop": self.entry.get("crop"),
                "reason": self.cfg.reason, "priority": int(self.cfg.priority), "status": self.status,
                "bands": list(self.cfg.bands), "az_coverage": summ["az_deg_by_band"],
                "coverage_pct": summ["pct_by_band"], "masked_pct": summ["masked_pct"],
                "mode": "full" if self.plan.mode == "full" else "two_sided", "gap_deg": round(self.gap_deg(), 1),
                "chunks": [{"path": f"tree360/{self.tree_id}/{c['path']}", "sha256": c["sha256"], "bytes": c["bytes"]}
                           for c in chunks],
                "merkle_root": root, "manifest": f"tree360/{self.tree_id}/manifest.json",
                "analysers": list(ANALYSERS), "analysers_optional": list(ANALYSERS_OPTIONAL),
                "lanes": {"L1": "manifest + leads", "L2": "320 px thumbnails (sim frames are 320 x 240)",
                          "L3": "full frames via the hub drive"},
                "term": "Deep Scan: proposed term, not defined by the platform yet (OQ-09)"}


class _Abort(Exception):
    pass


# ====================================================================== MuJoCo / bus IO
class SimIO:
    """``Tree360Run`` IO on one MuJoCo model/data and the hook bus (sim process or ``SimDriver``).

    ``gait``: the driver's ``TripodGaitGenerator`` (in-process); without one, drive commands go to
    ``bus.values["tree360_drive"]`` and ``/nandi/tree360/drive`` (live bridge, change request in the contract).
    ``capture``: ``gl`` (CameraRenderer RGB + depth on this model + PrivacyFilter), ``geometry`` (no pixels: the
    PrivacyFilter's truth/detector masks on a blank frame, coverage from geometry), ``perception`` (geometry here +
    ``/nandi/perception/tree360`` capture requests to the WP5 node)."""

    FIT_REQUEST_TOPIC = "/nandi/lidar/trunk_fit_request"
    DRIVE_TOPIC = "/nandi/tree360/drive"
    PERCEPTION_TOPIC = "/nandi/perception/tree360"

    def __init__(self, m, d, bus, gait=None, capture: str = "geometry", privacy: str = "assisted",
                 geofence: Optional[Sequence[Sequence[float]]] = None, width: int = 320, height: int = 240):
        import mujoco
        self.mj = mujoco
        self.m, self.d, self.bus, self.gait = m, d, bus, gait
        self.capture_mode = capture
        self.width, self.height = int(width), int(height)
        self.geofence = geofence
        from atharv_nandi.core.privacy_filter import PrivacyFilter
        self.privacy = PrivacyFilter(privacy, parcel=geofence)
        self._renderer = None
        self._person_geoms = None
        self._ray = None
        self._cams = {}
        self._truth = {}
        self.last_drive = (0.0, 0.0)
        self._arm_qadr: Optional[List[int]] = None

    # time / pose
    def now(self) -> float:
        return float(self.d.time)

    def pose(self):
        q = self.d.qpos
        w, x, y, z = q[3:7]
        return (float(q[0]), float(q[1])), math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def velocity(self) -> Tuple[float, float]:
        v = self.d.qvel
        return float(math.hypot(v[0], v[1])), float(v[5])

    def yaw_rate_deg(self) -> float:
        return abs(math.degrees(float(self.d.qvel[5])))

    def tilt_deg(self) -> float:
        w, x, y, z = self.d.qpos[3:7]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))

    def truth_centre(self, tree_id: str) -> Tuple[float, float]:
        c = self._truth.get(tree_id)
        if c is None:
            from atharv_nandi.sim.world import trees as trees_mod
            bid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, trees_mod.body_name(tree_id))
            c = (float(self.d.xpos[bid][0]), float(self.d.xpos[bid][1])) if bid >= 0 else None
            if c is None:
                raise KeyError(f"tree body for {tree_id!r} not in the model")
            self._truth[tree_id] = c
        return c

    def rtk(self) -> Optional[str]:
        plat = self.bus.values.get("platform") or {}
        g = plat.get("gnss") if isinstance(plat, dict) else None
        if isinstance(g, dict):
            return g.get("fix") or g.get("status")
        return plat.get("rtk") if isinstance(plat, dict) else None

    def inhibit(self) -> dict:
        hold = self.bus.values.get("hold")
        mi = self.bus.values.get("motion_inhibit") or {}
        fs = mi.get("safety_fields") or {}
        stb = mi.get("stability") or {}
        out = {"hold": (hold or {}).get("reason") if hold else None, "pause": bool(fs.get("orbit_pause")),
               "stop": bool(fs.get("stop")) or bool(stb.get("stop")), "stop_reason": None,
               "walk_block": bool(stb.get("walk")) or bool(stb.get("stance"))}
        if out["stop"]:
            rs = list(fs.get("reasons") or []) + list(stb.get("reasons") or [])
            out["stop_reason"] = ("person_hold" if any(r in ("protective", "bumper") for r in rs)
                                  else (rs[0] if rs else "person_hold"))
        return out

    # motion
    def drive(self, vx: float, yaw: float):
        vx, yaw = float(vx), float(yaw)
        self.last_drive = (vx, yaw)
        if self.gait is not None:
            self.gait.set_command(vx, 0.0, yaw, hold=True)
        else:
            self.bus.values["tree360_drive"] = {"vx": vx, "yaw": yaw, "t": self.now()}

    def stand(self):
        self.last_drive = (0.0, 0.0)
        if self.gait is not None:
            if self.gait.is_active:
                self.gait.release()
        else:
            self.bus.values["tree360_drive"] = {"vx": 0.0, "yaw": 0.0, "stand": True, "t": self.now()}

    def teleport(self, trunk_xy: Sequence[float], heading: float):
        """Assisted ``station_teleport`` (station-tier tests): trunk at ``trunk_xy`` on the ground, ``heading``."""
        m, d, mj = self.m, self.d, self.mj
        gid = np.zeros(1, dtype=np.int32)
        grp = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

        def ground(x, y):
            dist = mj.mj_ray(m, d, np.array([x, y, 3.0]), np.array([0.0, 0.0, -1.0]), grp, 1, -1, gid)
            return 3.0 - dist if dist >= 0 else 0.0

        z0 = float(d.qpos[2]) - ground(d.qpos[0], d.qpos[1])
        x, y = float(trunk_xy[0]), float(trunk_xy[1])
        d.qpos[0:3] = [x, y, ground(x, y) + z0 + 0.005]
        d.qpos[3:7] = [math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)]
        d.qvel[0:6] = 0.0
        mj.mj_forward(m, d)
        try:
            from atharv_nandi.sim.world.modules import tools as tools_mod
            tools_mod.seat_tools(m, d)      # the rack welds would yank the tools across the jump
        except ImportError:
            pass
        mj.mj_forward(m, d)

    # mast
    def _mc(self):
        mc = self.bus.values.get("mast_control")
        if mc is None:
            raise _Abort("no_mast_control")
        return mc

    def mast_request(self, ext: float, reason: str) -> dict:
        from atharv_nandi.core.interfaces import MastRequest
        dec = self._mc().submit(MastRequest("preset", float(ext), reason, t=self.now()))
        return {"allowed": bool(dec.allowed), "reason": dec.reason, "detail": (dec.values or {}).get("detail")
                if isinstance(dec.values, dict) else None}

    def light_ok(self) -> bool:
        """World light >= the camera threshold (``bus.values["world"]["light"]["camera_ok"]``; unknown -> ok)."""
        w = self.bus.values.get("world")
        light = w.get("light") if isinstance(w, dict) else None
        return bool(light.get("camera_ok", True)) if isinstance(light, dict) else True

    def arm_parked(self) -> bool:
        """Arm joints at the home keyframe (PARK) within ``ARM_PARK_TOL``; a model without the arm or keyframe -> ok."""
        if self._arm_qadr is None:
            from atharv_nandi.core.arm_ik_solver import ARM_JOINTS
            adr = []
            for n in ARM_JOINTS:
                j = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, n)
                if j >= 0:
                    adr.append(int(self.m.jnt_qposadr[j]))
            self._arm_qadr = adr if (adr and self.m.nkey) else []
        if not self._arm_qadr:
            return True
        q, q0 = self.d.qpos, self.m.key_qpos[0]
        return all(abs(float(q[a]) - float(q0[a])) <= ARM_PARK_TOL for a in self._arm_qadr)

    def mast_release(self):
        mc = self.bus.values.get("mast_control")
        if mc is not None:
            mc.release("preset")

    def mast_state(self) -> dict:
        st = self.bus.values.get("mast")
        return st if isinstance(st, dict) else {"ext_m": 0.0}

    def aim(self, pan: float, tilt: float):
        self._mc().aim(float(pan), float(tilt))

    # trunk fit (WP5 lidar hook, in-process delivery)
    def request_fit(self, tree: dict):
        payload = {"tree_id": tree["tree_id"], "prior_xy": [float(tree["xy"][0]), float(tree["xy"][1])],
                   "canopy_r_m": float(tree["canopy_r_m"])}
        r = self._entry_trunk_r(tree["tree_id"])
        if r:
            payload["trunk_r_m"] = r
        self.bus.deliver(self.FIT_REQUEST_TOPIC, payload)

    def cancel_fit(self):
        self.bus.deliver(self.FIT_REQUEST_TOPIC, {"tree_id": None})

    def _entry_trunk_r(self, tree_id: str) -> Optional[float]:
        return getattr(self, "trunk_r", {}).get(tree_id)

    def trunk_fit(self, tree_id: str) -> Optional[dict]:
        lid = self.bus.values.get("lidar") or {}
        fit = lid.get("trunk_fit") if isinstance(lid, dict) else None
        if not isinstance(fit, dict):
            return None
        if fit.get("tree_id") not in (None, tree_id):
            return None
        return fit

    # cameras
    def _cid(self, name: str) -> int:
        c = self._cams.get(name)
        if c is None:
            c = self._cams[name] = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_CAMERA, name)
        return c

    def ring_facing(self, target_xy) -> str:
        from atharv_nandi.core.camera_rig import ring_facing
        return ring_facing(self.m, self.d, target_xy)

    def crown_camera(self) -> str:
        return "crown_cam"

    def camera_xy(self, name: str) -> Tuple[float, float]:
        self.mj.mj_kinematics(self.m, self.d)
        p = self.d.cam_xpos[self._cid(name)]
        return float(p[0]), float(p[1])

    def backend_name(self) -> str:
        return "gl" if self._renderer is not None else self.capture_mode

    def coverage_ray_fn(self):
        if self._ray is None:
            self._ray = cv.mujoco_ray_fn(self.m, self.d)
        return self._ray

    def surface_samples(self, canopy: cv.Canopy, plan: cv.BandPlan):
        from atharv_nandi.sim.world import trees as trees_mod
        bid = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_BODY, trees_mod.body_name(canopy.tree_id))
        if bid < 0:
            return None
        ray = cv.mujoco_surface_ray_fn(self.m, self.d)
        return cv.surface_samples_model(canopy, plan, ray, lambda g: int(self.m.geom_bodyid[g]), bid)

    def capture(self, camera: str, band: str, seq: int, tree_id: str) -> CaptureResult:
        from atharv_nandi.core import camera_rig
        from atharv_nandi.core.detector_model import intrinsics_from_fovy
        from atharv_nandi.core.privacy_filter import person_boxes
        mj, m, d = self.mj, self.m, self.d
        mj.mj_kinematics(m, d)
        mj.mj_camlight(m, d)
        cid = self._cid(camera)
        fovy = float(m.cam_fovy[cid])
        intr = intrinsics_from_fovy(fovy, self.width, self.height)
        pose = camera_rig.camera_pose(d, cid)
        pos = tuple(float(v) for v in d.cam_xpos[cid])
        fwd = tuple(float(v) for v in -d.cam_xmat[cid].reshape(3, 3)[:, 2])
        lid = self.bus.values.get("lidar") or {}
        clusters = lid.get("clusters") or [] if isinstance(lid, dict) else []
        persons = person_boxes(m, d)
        backend = self.capture_mode
        rgb, depth, person_px = None, None, None
        if backend == "gl":
            try:
                if self._renderer is None:
                    self._renderer = camera_rig.CameraRenderer(m, camera, self.width, self.height, cull_dist=4.0)
                self._renderer.use(camera)
                rgb, depth = self._renderer.rgbd(d)
                if self._person_geoms is None:
                    from atharv_nandi.sim.world import actors
                    self._person_geoms = actors.person_geoms(m)
                if self._person_geoms:
                    seg = self._renderer.seg(d)
                    person_px = np.isin(seg[..., 0], self._person_geoms) & (
                        seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM))
            except Exception as exc:      # noqa: BLE001 - no GL: fall back to geometry, reported per frame
                self.capture_mode = backend = "geometry"
                self._gl_error = f"{type(exc).__name__}: {exc}"
        if rgb is None:
            rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if backend == "perception":
            self.bus.publish(self.PERCEPTION_TOPIC, {"cmd": "capture", "camera": camera, "band": band, "seq": seq,
                                                     "tree_id": tree_id})
        res = self.privacy.apply(camera, self.now(), rgb, pose, intr, depth=depth, persons=persons,
                                 clusters=clusters, meta={"band": band, "seq": seq, "tree_id": tree_id})
        stats = dict(res.stats)
        if person_px is not None:
            stats["person_px"] = int(person_px.sum())
            stats["person_unmasked_px"] = 0 if res.dropped else int(np.sum(person_px & ~np.asarray(res.frame.masked)))
        mask = parcel = None
        if not res.dropped:
            mk = res.frame.meta.get("masks") or {}
            if "truth" in mk:
                mask = np.asarray(mk["truth"], bool) | np.asarray(mk["detector"], bool)
                parcel = np.asarray(mk["parcel"], bool)
            else:
                mask = np.asarray(res.frame.masked, bool)
        frame = None if (res.dropped or backend == "perception") else res.frame
        return CaptureResult(camera, pose, intr, fovy, frame, mask, res.dropped, res.reason, stats,
                             pos, fwd, backend, parcel, depth)

    def on_start(self, run: "Tree360Run"):
        self.trunk_r = {run.tree_id: float(run.entry.get("trunk_r_m") or 0.0) or None}

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
