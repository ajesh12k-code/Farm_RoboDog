"""Pure run-evidence logic: run ids, git identity, run metadata, per-attempt harvest events,
interventions and joint effort statistics. No ROS; run_recorder.py and tools/headless_eval.py feed it.

Run directory (runs/<run_id>/, git-ignored by runs/.gitignore):
    meta.json                 run identity: device, git sha + dirty flag, config hashes, NANDI_* env
    kis_v1.jsonl              every kis/v1 message the bridge produced ({"topic", "retain", "payload"})
    harvest_attempts.jsonl    one line per auto_harvest attempt (AttemptTracker)
    events.jsonl              falls, resets, interventions, e-stops (one line each)
    joint_effort_summary.json peak / RMS |effort| and |velocity| per actuator vs the MJCF forcerange
    manifest.json             SHA-256 per file + Merkle root (atharv_nandi/core/manifest.py)
    kpi.json                  tools/nandi_kpi.py
"""
import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEVICE_ID = "nandi-sim-01"
ARM_HARVEST_MODE = "ik_harvest"
# Basket truth is read this long after the harvest node reports the end of an attempt: the crop
# may still be falling into the basket, and /nandi/game_status is only published at 2 Hz.
SETTLE_S = 1.5
# A crop whose centre rose this much during an attempt was really lifted by the gripper.
LIFTED_M = 0.03


# ---------------------------------------------------------------------- identity
def utc_stamp(t: Optional[_dt.datetime] = None) -> str:
    t = t or _dt.datetime.now(_dt.timezone.utc)
    return t.astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(utc: str, short_sha: str, seed: Any) -> str:
    """<utc yyyymmddThhmmssZ>-<short git sha>-<seed>; every part made slug-safe."""
    def clean(s: Any) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", str(s)) or "x"
    return f"{clean(utc)}-{clean(short_sha)[:12].lower()}-{clean(seed).lower()}"


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def _git_dir(repo: str) -> Optional[str]:
    dot = os.path.join(repo, ".git")
    if os.path.isdir(dot):
        return dot
    text = _read(dot)  # worktree: "gitdir: /path/to/.git/worktrees/x"
    if text and text.startswith("gitdir:"):
        gd = text.split(":", 1)[1].strip()
        gd = gd if os.path.isabs(gd) else os.path.join(repo, gd)
        return gd if os.path.isdir(gd) else None
    return None


def git_sha(repo: str) -> Optional[str]:
    """HEAD commit, read from the .git files (no git binary, no safe.directory issues)."""
    env = os.environ.get("NANDI_GIT_SHA")
    if env:
        return env.strip()
    gd = _git_dir(repo)
    if not gd:
        return None
    head = _read(os.path.join(gd, "HEAD"))
    if not head:
        return None
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    common = _read(os.path.join(gd, "commondir"))
    roots = [gd] + ([os.path.normpath(os.path.join(gd, common))] if common else [])
    for root in roots:
        sha = _read(os.path.join(root, ref))
        if sha:
            return sha
        packed = _read(os.path.join(root, "packed-refs")) or ""
        for line in packed.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def git_dirty(repo: str) -> Optional[bool]:
    """True/False from `git status --porcelain` (runs/ is ignored); None when git cannot tell."""
    env = os.environ.get("NANDI_GIT_DIRTY")
    if env in ("0", "1"):
        return env == "1"
    try:
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", repo, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def sw_version(sha: Optional[str], dirty: Optional[bool]) -> str:
    """Version string written into every record (SW-11): 'atharv-nandi <short sha>[-dirty]'."""
    short = (sha or "unknown")[:7]
    return f"atharv-nandi {short}{'-dirty' if dirty else ''}"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def build_meta(run_id: str, repo: str, config_paths: Iterable[str], env: Optional[Dict[str, str]] = None,
               start_utc: Optional[str] = None, device_id: str = DEVICE_ID,
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    env = os.environ if env is None else env
    sha, dirty = git_sha(repo), git_dirty(repo)
    configs = {}
    for p in config_paths:
        rel = os.path.relpath(p, repo) if os.path.isabs(p) else p
        full = p if os.path.isabs(p) else os.path.join(repo, p)
        configs[rel] = sha256_file(full) if os.path.isfile(full) else None
    meta = {
        "run_id": run_id,
        "device_id": device_id,
        "git_sha": sha,
        "dirty": dirty,
        "sw": sw_version(sha, dirty),
        "config_sha256": configs,
        "NANDI_TERRAIN": env.get("NANDI_TERRAIN", "rough"),
        "NANDI_SEED": env.get("NANDI_SEED", "7"),
        "NANDI_CROPS": env.get("NANDI_CROPS", "6"),
        "start_utc": start_utc or _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds"),
        "data_provenance": "simulated",
    }
    if extra:
        meta.update(extra)
    return meta


def write_json(path: str, obj: Any):
    # Per-process tmp name: telemetry_bridge and run_recorder can start in the same second with the same run_id
    # and write the same meta.json; a shared tmp let one os.replace() steal the other's file (FileNotFoundError).
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def read_json_dict(path: str) -> Optional[Dict[str, Any]]:
    """A JSON object file, or None when it is missing, unreadable, not JSON or not an object (closes the file; the
    old ``json.load(open(...))`` call sites leaked the handle and raised on a non-object, WPCOV review)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


class JsonlWriter:
    """Append-only JSONL, flushed per line so a live run directory is always readable."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")
        self.lines = 0

    def write(self, obj: Any):
        self._fh.write(json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n")
        self._fh.flush()
        self.lines += 1

    def write_line(self, line: str):
        self._fh.write(line.rstrip("\n") + "\n")
        self._fh.flush()
        self.lines += 1

    def close(self):
        self._fh.close()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue  # a half-written last line of a live run
    return out


# ---------------------------------------------------------------------- harvest attempts
class AttemptTracker:
    """Turns the /nandi/harvest_status + /nandi/game_status streams into one event per attempt.

    An attempt starts when the arm mode enters ik_harvest and ends when it leaves it. Its success is
    NOT the harvest node's own verdict: it is the basket truth from game_status (in_basket count
    went up and the target is listed in in_basket_names) read SETTLE_S after the end. The node's
    verdict is kept as reported_success, and `agrees` flags a false positive / false negative.

    Feed in time order (t = seconds on any monotonic clock): harvest(t, status), game(t, status),
    crop_positions(positions). poll(t) returns attempts finalised since the last call.
    """

    def __init__(self, sw: str = "", run_id: str = "", device_id: str = DEVICE_ID,
                 settle_s: float = SETTLE_S, wall_ms: Optional[Callable[[], int]] = None):
        self.sw, self.run_id, self.device_id, self.settle_s = sw, run_id, device_id, settle_s
        self.wall_ms = wall_ms
        self.game: Optional[Dict[str, Any]] = None
        self.crops: Dict[str, Sequence[float]] = {}
        self.current: Optional[Dict[str, Any]] = None
        self.pending: List[Dict[str, Any]] = []   # ended, waiting for basket truth
        self.done: List[Dict[str, Any]] = []
        self.count = 0
        self._last_mode = "rl_balance"
        self._last_phase = "idle"

    def crop_positions(self, positions: Dict[str, Sequence[float]]):
        self.crops = dict(positions)
        a = self.current
        if a is not None and a["crop_id"] in self.crops:
            z = float(self.crops[a["crop_id"]][2])
            a["_max_z"] = z if a.get("_max_z") is None else max(a["_max_z"], z)
            z0 = a["crop_xyz"][2] if a["crop_xyz"] is not None else None
            # Phase in which a lifted crop fell back below the lift threshold (where it was lost).
            if (z0 is not None and a.get("lost_in_phase") is None and a["_max_z"] - z0 >= LIFTED_M
                    and z - z0 < LIFTED_M):
                a["lost_in_phase"] = self._last_phase

    def game_status(self, t: float, game: Dict[str, Any]):
        prev = self.game
        if prev is not None and game.get("elapsed_s", 0.0) + 0.5 < prev.get("elapsed_s", 0.0):
            # World reset: basket truth after this point says nothing about earlier attempts.
            self._finalise_all(t, prev, reason="world_reset")
        self.game = dict(game)
        self._finalise_due(t)

    def harvest_status(self, t: float, h: Dict[str, Any]):
        mode, phase = h.get("mode", "rl_balance"), h.get("phase", "idle")
        if mode == ARM_HARVEST_MODE and self._last_mode != ARM_HARVEST_MODE:
            self._start(t, h)
        if self.current is not None:
            if mode == ARM_HARVEST_MODE:
                if self.current["crop_id"] is None and h.get("target"):
                    self.current["crop_id"] = h["target"]
                    xyz = self.crops.get(h["target"])
                    self.current["crop_xyz"] = [round(float(v), 4) for v in xyz] if xyz is not None else None
                if phase != self._last_phase or not self.current["phases"]:
                    self.current["phases"].append({"phase": phase, "t": round(t - self.current["t_start"], 3)})
            else:
                self._end(t, h)
        self._last_mode, self._last_phase = mode, phase

    def _start(self, t: float, h: Dict[str, Any]):
        self._finalise_all(t, self.game or {}, reason="next_attempt")  # truth before this one moves the basket
        self.count += 1
        target = h.get("target")
        g = self.game or {}
        xyz = self.crops.get(target) if target else None
        self.current = {
            "attempt": self.count,
            "run_id": self.run_id,
            "device_id": self.device_id,
            "sw": self.sw,
            "crop_id": target,
            "crop_xyz": [round(float(v), 4) for v in xyz] if xyz is not None else None,
            "t_start": round(t, 3),
            "start_utc_ms": self.wall_ms() if self.wall_ms else None,
            "basket_before": int(g.get("in_basket", 0)),
            "names_before": list(g.get("in_basket_names", [])),
            "phases": [],
            "_max_z": float(xyz[2]) if xyz is not None else None,
        }

    def _end(self, t: float, h: Dict[str, Any]):
        a = self.current
        self.current = None
        a["t_end"] = round(t, 3)
        a["duration_s"] = round(t - a["t_start"], 3)
        a["result"] = h.get("last_result", "")
        a["reported_success"] = a["result"].startswith("Harvested")
        a["phase_durations_s"] = phase_durations(a["phases"], a["duration_s"])
        a["_due"] = t + self.settle_s
        self.pending.append(a)

    def _finalise_due(self, t: float):
        g = self.game
        while self.pending and g is not None and t >= self.pending[0]["_due"]:
            self._finalise(self.pending.pop(0), g, reason="basket_truth")

    def poll(self, t: float) -> List[Dict[str, Any]]:
        """Finalise attempts whose settle time has passed; return (and forget) all finalised ones."""
        self._finalise_due(t)
        out, self.done = self.done, []
        return out

    def flush(self, t: float) -> List[Dict[str, Any]]:
        """End of run: finalise whatever is pending with the latest basket state."""
        if self.current is not None:
            self._end(t, {"last_result": "Run ended during harvest"})
        self._finalise_all(t, self.game or {}, reason="run_end")
        out, self.done = self.done, []
        return out

    def _finalise_all(self, t: float, g: Dict[str, Any], reason: str):
        while self.pending:
            self._finalise(self.pending.pop(0), g, reason=reason)

    def _finalise(self, a: Dict[str, Any], g: Dict[str, Any], reason: str):
        a.pop("_due", None)
        after = int(g.get("in_basket", 0))
        names = list(g.get("in_basket_names", []))
        a["basket_after"] = after
        a["basket_delta"] = after - a["basket_before"]
        target_in = a["crop_id"] in names and a["crop_id"] not in a["names_before"]
        a["success"] = bool(a["basket_delta"] >= 1 and target_in)
        a["agrees"] = a["success"] == a["reported_success"]
        a["truth_source"] = reason
        # The node's grasp check passed iff it went on to carry. crop_max_rise_m (highest crop z during
        # the attempt minus its start z, from sim_qpos) tells a real lift from a false grasp report, and
        # a crop that rose but ended outside the basket was lost after the grasp (in carry or release).
        a["grasp_reported_ok"] = any(p["phase"] == "carry" for p in a["phases"])
        end = self.crops.get(a["crop_id"]) if a["crop_id"] else None
        a["crop_xyz_end"] = [round(float(v), 4) for v in end] if end is not None else None
        if end is not None and a["crop_xyz"] is not None:
            a["crop_moved_m"] = round(math.dist(a["crop_xyz"], [float(v) for v in end]), 4)
        else:
            a["crop_moved_m"] = None
        rise = a.pop("_max_z", None)
        a["crop_max_rise_m"] = (round(rise - a["crop_xyz"][2], 4)
                                if rise is not None and a["crop_xyz"] is not None else None)
        lifted = a["crop_max_rise_m"] is not None and a["crop_max_rise_m"] >= LIFTED_M
        a["outcome"] = ("picked" if a["success"] else
                        "lost_after_grasp" if lifted else
                        "false_grasp_report" if a["grasp_reported_ok"] and a["crop_max_rise_m"] is not None else
                        "not_picked")
        a.pop("names_before", None)
        self.done.append(a)


def phase_durations(phases: List[Dict[str, Any]], total_s: float) -> Dict[str, float]:
    """Sum of time spent in each phase (a phase seen twice, e.g. after a regrasp, adds up)."""
    out: Dict[str, float] = {}
    for i, p in enumerate(phases):
        end = phases[i + 1]["t"] if i + 1 < len(phases) else total_s
        out[p["phase"]] = round(out.get(p["phase"], 0.0) + max(0.0, end - p["t"]), 3)
    return out


# ---------------------------------------------------------------------- interventions + events
class InterventionDetector:
    """A manual command (cmd_vel / manual_arm) while a mission or harvest runs is an intervention.

    Repeated messages of one held command count once: a new intervention needs `gap_s` of quiet.
    """

    def __init__(self, gap_s: float = 1.0):
        self.gap_s = gap_s
        self.mission_active = False
        self._last: Dict[str, float] = {}
        self.count = 0

    def manual(self, t: float, source: str, nonzero: bool = True) -> bool:
        if not nonzero or not self.mission_active:
            return False
        last = self._last.get(source)
        self._last[source] = t
        if last is not None and t - last < self.gap_s:
            return False
        self.count += 1
        return True


class EdgeCounter:
    """Rising-edge counter for boolean conditions (e-stop engaged, fallen)."""

    def __init__(self):
        self.state = False
        self.count = 0

    def update(self, value: bool) -> bool:
        rising = bool(value) and not self.state
        self.state = bool(value)
        if rising:
            self.count += 1
        return rising


# ---------------------------------------------------------------------- joint effort statistics
class EffortStats:
    """Peak and RMS of |effort| and |velocity| per actuator/joint name."""

    def __init__(self):
        self.n: Dict[str, int] = {}
        self.peak_e: Dict[str, float] = {}
        self.ss_e: Dict[str, float] = {}
        self.peak_v: Dict[str, float] = {}
        self.ss_v: Dict[str, float] = {}

    def add(self, names: Sequence[str], effort: Sequence[float], velocity: Sequence[float] = ()):
        for i, name in enumerate(names):
            e = abs(float(effort[i])) if i < len(effort) else None
            v = abs(float(velocity[i])) if i < len(velocity) else None
            if e is not None and not math.isfinite(e):
                e = None
            if v is not None and not math.isfinite(v):
                v = None
            if e is None and v is None:
                continue
            self.n[name] = self.n.get(name, 0) + 1
            if e is not None:
                self.peak_e[name] = max(self.peak_e.get(name, 0.0), e)
                self.ss_e[name] = self.ss_e.get(name, 0.0) + e * e
            if v is not None:
                self.peak_v[name] = max(self.peak_v.get(name, 0.0), v)
                self.ss_v[name] = self.ss_v.get(name, 0.0) + v * v

    @property
    def samples(self) -> int:
        return max(self.n.values()) if self.n else 0

    def summary(self, force_limits: Optional[Dict[str, float]] = None, source: str = "") -> Dict[str, Any]:
        """{"actuators": {name: {...}}} with % of the MJCF forcerange where one is known."""
        force_limits = force_limits or {}
        acts = {}
        for name, n in self.n.items():
            rms_e = math.sqrt(self.ss_e[name] / n) if name in self.ss_e else None
            rms_v = math.sqrt(self.ss_v[name] / n) if name in self.ss_v else None
            lim = force_limits.get(name)
            row = {
                "samples": n,
                "peak_abs_effort": _r4(self.peak_e.get(name)),
                "rms_abs_effort": _r4(rms_e),
                "peak_abs_velocity": _r4(self.peak_v.get(name)),
                "rms_abs_velocity": _r4(rms_v),
                "force_limit": lim,
                "peak_pct_of_limit": _pct(self.peak_e.get(name), lim),
                "rms_pct_of_limit": _pct(rms_e, lim),
            }
            acts[name] = row
        worst = max((r["peak_pct_of_limit"] for r in acts.values() if r["peak_pct_of_limit"] is not None),
                    default=None)
        return {"source": source, "samples": self.samples, "max_peak_pct_of_limit": worst,
                "units": {"effort": "N or N*m (actuator force)", "velocity": "m/s or rad/s"},
                "actuators": acts}


def _r4(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 4)


def _pct(v: Optional[float], lim: Optional[float]) -> Optional[float]:
    if v is None or not lim:
        return None
    return round(100.0 * float(v) / float(lim), 2)


def mjcf_force_limits(model) -> Dict[str, float]:
    """{actuator name and its joint name: max |force|} from the compiled MjModel's forcerange."""
    import mujoco  # local: keeps this module importable without MuJoCo for the pure parts
    out: Dict[str, float] = {}
    for i in range(model.nu):
        lo, hi = model.actuator_forcerange[i]
        if not model.actuator_forcelimited[i] and lo == 0 and hi == 0:
            continue
        lim = float(max(abs(lo), abs(hi)))
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        out[name] = lim
        jid = model.actuator_trnid[i][0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) if jid >= 0 else None
        if jname:
            out[jname] = lim
    return out
