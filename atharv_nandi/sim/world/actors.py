"""Field actors: people, ladder and pole pickers, animals (WP4; ORCHARD-MVP-SPEC 2.6, 5.2; MVP-SPEC 4.2, 5.4).

Every actor is one **mocap body** with contype 0 / conaffinity 0 geoms (it is seen, never pushed: safety rays
and cameras see it, the contact solver does not) in geom group ``ACTOR_GROUP`` (1: produce and actors,
MVP-SPEC 2.4, inside the safety-ray mask {0, 1}). Names encode what an actor is, so a renderer, the privacy
mask and the safety layer classify geoms from the compiled model alone::

    body  actor_<class>_<id>                 class: person | animal
    geom  actor_<class>_<id>__<role>_<k>     role:  body | head | limb (person or animal pixels)
                                                    ladder | pole (props carried by a person: NOT person pixels)
    prop  actor_prop_<id>__fruit_0           a mocap fruit a ladder picker drops (debris-like, not a person)

Animals are never person pixels and never handled by person logic (OS 5.2): ``person_geoms`` /
``animal_geoms`` keep them apart.

**Privacy lock-out (MVP-SPEC 5.4).** noVNC shows the unfiltered viewer, so person actors cannot be enabled while
noVNC is up. ``persons_policy(env)``: ``NANDI_ACTORS=0`` → no actors; noVNC up (``novnc_up``: the decision
``docker/start.sh`` exports as ``NANDI_NOVNC_UP``, else start.sh's own rule on ``NANDI_FIELD_MODE`` /
``NANDI_NOVNC`` with its exact patterns) → person actors are left out (``auto``) or the build is **refused**
with ``ActorPolicyError`` (``NANDI_ACTORS=1``). Animals stay.

Scripted motion (``ActorRuntime``) is pure Python: waypoints in the world frame or in the robot frame at the
trigger (``frame: robot``), a walking speed, per-waypoint holds, then ``stay`` / ``home`` / ``park``. The
hook ``atharv_nandi/sim/hooks/world_events.py`` writes the poses into ``data.mocap_pos`` / ``mocap_quat`` at 50 Hz.
Sizes and speeds are ASSUMED.
"""
import math
import os
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ACTOR_GROUP = 1
PROP_GROUP = 1
PARK_XY = (-30.0, -30.0)        # parked actors (inactive scripts) sit far outside the field
SKIN = [0.80, 0.55, 0.42, 1.0]
CLOTH = [0.20, 0.30, 0.65, 1.0]
CLOTH2 = [0.75, 0.35, 0.20, 1.0]
WOOD = [0.55, 0.42, 0.25, 1.0]
PERSON_ROLES = ("body", "head", "limb")
PROP_ROLES = ("ladder", "pole", "fruit")


class ActorPolicyError(RuntimeError):
    """Person actors requested while noVNC (unfiltered video) is up: refused (MVP-SPEC 5.4)."""


# ---------------------------------------------------------------------- geometry per type
def _cap(role, a, b, r, rgba):
    return {"role": role, "type": "capsule", "fromto": [*a, *b], "size": [r], "rgba": rgba}


def _sph(role, p, r, rgba):
    return {"role": role, "type": "sphere", "pos": list(p), "size": [r], "rgba": rgba}


def _box(role, p, half, rgba):
    return {"role": role, "type": "box", "pos": list(p), "size": list(half), "rgba": rgba}


def _ell(role, p, half, rgba):
    return {"role": role, "type": "ellipsoid", "pos": list(p), "size": list(half), "rgba": rgba}


def _person(h: float, rgba=CLOTH) -> List[dict]:
    """Standing person of height h (feet at z 0): torso+legs capsule, head sphere, arms."""
    head_r = 0.11 * h / 1.7
    r = 0.16 * h / 1.7
    return [_cap("body", (0, 0, r), (0, 0, h - 2 * head_r - r * 0.6), r, rgba),
            _sph("head", (0, 0, h - head_r), head_r, SKIN),
            _cap("limb", (0, -r - 0.03, 0.55 * h), (0.05, -r - 0.05, 0.82 * h), 0.04 * h / 1.7, SKIN),
            _cap("limb", (0, r + 0.03, 0.55 * h), (0.05, r + 0.05, 0.82 * h), 0.04 * h / 1.7, SKIN)]


def _ladder(top: float) -> List[dict]:
    """A-frame orchard ladder leaning toward +x (the tree), rails and rungs every 0.3 m."""
    g = []
    for s in (-0.22, 0.22):
        g.append(_cap("ladder", (-0.35, s, 0.02), (0.05, s, top), 0.025, WOOD))
    n = int(top / 0.3)
    for k in range(1, n + 1):
        z = 0.3 * k
        x = -0.35 + 0.40 * z / top
        g.append(_cap("ladder", (x, -0.22, z), (x, 0.22, z), 0.015, WOOD))
    return g


def _shift(geoms: List[dict], dx: float, dz: float) -> List[dict]:
    out = []
    for p in geoms:
        q = dict(p)
        if "fromto" in q:
            f = list(q["fromto"])
            q["fromto"] = [f[0] + dx, f[1], f[2] + dz, f[3] + dx, f[4], f[5] + dz]
        else:
            q["pos"] = [q["pos"][0] + dx, q["pos"][1], q["pos"][2] + dz]
        out.append(q)
    return out


# OS 5.2 "person at 1-2.5 m height": a 1.7 m picker standing on a 0.8 m step spans 0.8-2.5 m, so the head and
# torso are inside the band and the top of the head is at its 2.5 m limit (a 1.7 m person cannot fit inside a
# 1.5 m band; the legs below 1 m are the documented deviation, wp4.md)
LADDER_FEET_M = 0.8
LADDER_PERSON_H_M = 1.7
LADDER_HAND_Z_M = round(LADDER_FEET_M + 0.82 * LADDER_PERSON_H_M, 2)    # raised hand (the _person arm tip)


def _ladder_picker() -> List[dict]:
    g = _ladder(2.0) + _shift(_person(LADDER_PERSON_H_M, CLOTH2), -0.2, LADDER_FEET_M)
    g.append(_cap("pole", (-0.1, 0.2, LADDER_FEET_M + 1.0), (0.9, 0.3, LADDER_FEET_M + 2.0), 0.015, WOOD))
    return g


def _pole_picker() -> List[dict]:
    return _person(1.65) + [_cap("pole", (0.1, -0.2, 1.0), (1.6, -0.3, 3.6), 0.016, WOOD)]


def _quadruped(length: float, height: float, rgba, head_up: float = 0.1) -> List[dict]:
    body_h = 0.35 * height
    leg_top = height - body_h
    g = [_ell("body", (0, 0, leg_top + body_h / 2), (length / 2, 0.22 * length / 2 + 0.05, body_h / 2), rgba),
         _sph("head", (length / 2 + 0.08 * length, 0, height - body_h / 2 + head_up), 0.12 * length + 0.03, rgba)]
    for sx in (-0.38 * length, 0.38 * length):
        for sy in (-0.1 * length, 0.1 * length):
            g.append(_cap("limb", (sx, sy, 0.03), (sx, sy, leg_top + 0.02), 0.03 + 0.012 * length, rgba))
    return g


def _monkey() -> List[dict]:
    brown = [0.45, 0.35, 0.25, 1.0]
    return [_ell("body", (0, 0, 0.30), (0.13, 0.11, 0.20), brown), _sph("head", (0.05, 0, 0.58), 0.08, SKIN),
            _cap("limb", (0.08, -0.1, 0.05), (0.06, -0.1, 0.25), 0.03, brown),
            _cap("limb", (0.08, 0.1, 0.05), (0.06, 0.1, 0.25), 0.03, brown),
            _cap("limb", (-0.12, 0, 0.15), (-0.45, 0, 0.35), 0.015, brown)]


# type -> (class, builder, nominal height m, label). Heights / lengths ASSUMED (MVP-SPEC 4.2, OS 5.2).
ACTOR_TYPES: Dict[str, Tuple[str, Callable[[], List[dict]], float, str]] = {
    "adult": ("person", lambda: _person(1.7), 1.7, "adult walker 1.7 m"),
    "worker": ("person", lambda: _person(1.65, CLOTH2), 1.65, "orchard worker 1.65 m"),
    "child": ("person", lambda: _person(1.0, [0.9, 0.7, 0.2, 1]), 1.0, "child 1.0 m"),
    "crouching": ("person", lambda: [_ell("body", (0, 0, 0.45), (0.30, 0.22, 0.38), CLOTH),
                                     _sph("head", (0.15, 0, 0.80), 0.10, SKIN)], 0.9, "crouching worker 0.9 m"),
    "lying": ("person", lambda: [_cap("body", (-0.75, 0, 0.13), (0.65, 0, 0.13), 0.13, CLOTH),
                                 _sph("head", (0.80, 0, 0.12), 0.11, SKIN)], 0.25, "person lying 0.25 m"),
    "ladder_picker": ("person", _ladder_picker, 2.8, "picker on a ladder, feet 0.8 m (person 0.8-2.5 m), with a pole"),
    "pole_picker": ("person", _pole_picker, 3.6, "worker with a 3.5 m pole picker"),
    "dog": ("animal", lambda: _quadruped(0.65, 0.50, [0.55, 0.45, 0.30, 1.0]), 0.5, "dog"),
    "monkey": ("animal", _monkey, 0.66, "rhesus monkey"),
    "cattle": ("animal", lambda: _quadruped(2.0, 1.40, [0.85, 0.82, 0.75, 1.0], 0.0), 1.4, "cattle"),
    "nilgai": ("animal", lambda: _quadruped(1.9, 1.40, [0.50, 0.45, 0.48, 1.0], 0.15), 1.4, "nilgai"),
}


def actor_class(atype: str) -> str:
    if atype not in ACTOR_TYPES:
        raise KeyError(f"actor type {atype!r} not in {sorted(ACTOR_TYPES)}")
    return ACTOR_TYPES[atype][0]


def geoms_for(atype: str) -> List[dict]:
    return ACTOR_TYPES[atype][1]()


def body_name(spec: "ActorSpec") -> str:
    return f"actor_{spec.cls}_{spec.id}"


def prop_body_name(actor_id: str) -> str:
    return f"actor_prop_{actor_id}"


# ---------------------------------------------------------------------- privacy policy
def _flag(v: Optional[str]) -> Optional[bool]:
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "on", "yes"):
        return True
    if s in ("0", "false", "off", "no"):
        return False
    return None


# docker/start.sh decides with these exact shell `case` patterns (case-sensitive, no trimming); the Python side
# must reach the same answer for the same environment, so it matches the strings exactly too.
FIELD_MODE_ON = ("1", "true", "yes", "on")       # start.sh: case "${NANDI_FIELD_MODE:-0}" in 1|true|yes|on)
NOVNC_OFF = ("0", "false", "no", "off")          # start.sh: case "${NANDI_NOVNC:-1}" in 0|false|no|off)


def novnc_up(env: Mapping[str, str] = os.environ) -> bool:
    """Is the unfiltered noVNC desktop up? Fail-closed: True unless the launcher's own rule says it is not.

    ``NANDI_NOVNC_UP`` (exported by ``docker/start.sh`` with the decision it made: ``0`` / ``1``) wins; any
    value but exactly ``0`` counts as up. Without it (tests, a bare process) the rule of ``start.sh`` is
    applied to the same variables with the same exact patterns: noVNC is down only for ``NANDI_FIELD_MODE``
    in ``FIELD_MODE_ON`` or ``NANDI_NOVNC`` in ``NOVNC_OFF``."""
    decided = env.get("NANDI_NOVNC_UP")
    if decided is not None:
        return decided != "0"
    if env.get("NANDI_FIELD_MODE", "0") in FIELD_MODE_ON:
        return False
    return env.get("NANDI_NOVNC", "1") not in NOVNC_OFF


def persons_policy(env: Mapping[str, str] = os.environ) -> Tuple[bool, bool, str]:
    """(actors_on, persons_on, reason). Raises ``ActorPolicyError`` for NANDI_ACTORS=1 with noVNC up."""
    req = _flag(env.get("NANDI_ACTORS"))
    if req is False:
        return False, False, "NANDI_ACTORS=0"
    if novnc_up(env):
        if req is True:
            raise ActorPolicyError("person actors refused: noVNC is up (unfiltered video, MVP-SPEC 5.4); "
                                   "set NANDI_FIELD_MODE=1 or NANDI_NOVNC=0, or NANDI_ACTORS=0")
        return True, False, "novnc_up: person actors left out (privacy lock-out)"
    return True, True, "ok"


# ---------------------------------------------------------------------- field entries and scripts
@dataclass
class Script:
    waypoints: List[Tuple[float, float]] = dc_field(default_factory=list)
    frame: str = "world"                       # world | robot (robot pose at the trigger, x forward, y left)
    speed_mps: float = 1.0
    holds_s: List[float] = dc_field(default_factory=list)   # hold after reaching waypoint k
    loop: bool = False
    on_end: str = "stay"                       # stay | home | park
    face: str = "path"                         # path | fixed

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Script":
        d = d or {}
        return cls([(float(p[0]), float(p[1])) for p in d.get("waypoints", [])], d.get("frame", "world"),
                   float(d.get("speed_mps", 1.0)), [float(h) for h in d.get("holds_s", [])],
                   bool(d.get("loop", False)), d.get("on_end", "stay"), d.get("face", "path"))


@dataclass
class ActorSpec:
    id: str
    type: str
    home: Tuple[float, float, float]           # x, y, yaw_deg
    script: Script
    start: dict                                # {mode: always|at_s|robot_within|cmd|never, ...}
    parked: bool = False                       # start parked off-field (e.g. lying person: WP6 test only)
    drop_every_s: float = 0.0                  # ladder picker: drop a fruit every n s (0 = never)
    label: str = ""

    @property
    def cls(self) -> str:
        return actor_class(self.type)

    @classmethod
    def from_dict(cls, d: dict) -> "ActorSpec":
        h = d.get("home", [0, 0, 0])
        a = cls(str(d["id"]), str(d["type"]), (float(h[0]), float(h[1]), float(h[2]) if len(h) > 2 else 0.0),
                Script.from_dict(d.get("script")), dict(d.get("start", {"mode": "cmd"})),
                bool(d.get("parked", False)), float(d.get("drop_every_s", 0.0)), d.get("label", ""))
        actor_class(a.type)
        return a


def specs_from_field(field: dict, env: Mapping[str, str] = os.environ) -> List[ActorSpec]:
    """The actors this process composes for ``field``, after the privacy policy."""
    entries = list(field.get("actors") or [])
    if not entries:
        return []
    on, persons, _reason = persons_policy(env)
    if not on:
        return []
    out = [ActorSpec.from_dict(e) for e in entries]
    ids = [a.id for a in out]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate actor ids in field {field.get('world')!r}: {ids}")
    return [a for a in out if persons or a.cls != "person"]


def yaw_quat(yaw_rad: float) -> List[float]:
    return [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)]


class ActorRuntime:
    """Scripted pose of one actor over sim time. States: idle (home) → active → done (stay | home | park)."""

    def __init__(self, spec: ActorSpec):
        self.spec = spec
        self.reset()

    def reset(self):
        s = self.spec
        self.state = "parked" if s.parked else "idle"
        self.t0: Optional[float] = None
        self.path: List[Tuple[float, float]] = []
        self.holds: List[float] = []
        self.xy = (PARK_XY if s.parked else (s.home[0], s.home[1]))
        self.yaw = math.radians(s.home[2])
        self.drops = 0

    def start(self, t: float, robot_pose: Optional[Tuple[float, float, float]] = None):
        """Begin the script at sim time t (robot_pose = (x, y, yaw) for ``frame: robot``)."""
        sc = self.spec.script
        pts = list(sc.waypoints)
        if sc.frame == "robot":
            rx, ry, ryaw = robot_pose if robot_pose is not None else (0.0, 0.0, 0.0)
            c, s = math.cos(ryaw), math.sin(ryaw)
            pts = [(rx + c * px - s * py, ry + s * px + c * py) for px, py in pts]
        start_xy = (self.spec.home[0], self.spec.home[1]) if self.state == "parked" else self.xy
        self.path = [start_xy] + pts
        self.holds = [0.0] + [sc.holds_s[k] if k < len(sc.holds_s) else 0.0 for k in range(len(pts))]
        self.t0 = float(t)
        self.state = "active"

    def stop(self):
        if self.state == "active":
            self.state = "done"

    def park(self):
        self.state = "parked"
        self.xy = PARK_XY

    def _duration(self) -> float:
        v = max(self.spec.script.speed_mps, 1e-6)
        legs = sum(math.dist(a, b) for a, b in zip(self.path, self.path[1:]))
        return legs / v + sum(self.holds)

    def _walk(self, tau: float) -> Tuple[Tuple[float, float], Optional[float], bool]:
        """Position, heading (None = keep) and done at tau seconds into the path."""
        v = max(self.spec.script.speed_mps, 1e-6)
        for k in range(len(self.path)):
            h = self.holds[k]
            if tau <= h:
                return self.path[k], None, False
            tau -= h
            if k + 1 == len(self.path):
                return self.path[k], None, True
            a, b = self.path[k], self.path[k + 1]
            d = math.dist(a, b)
            if d > 1e-9 and tau <= d / v:
                f = tau * v / d
                return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), math.atan2(b[1] - a[1], b[0] - a[0]), False
            tau -= d / v
        return self.path[-1], None, True

    def update(self, t: float) -> Tuple[float, float, float]:
        """Advance to sim time t; returns (x, y, yaw)."""
        if self.state == "active" and self.path:
            tau = float(t) - float(self.t0)
            if self.spec.script.loop:
                tau = tau % max(self._duration(), 1e-6)
            xy, yaw, done = self._walk(tau)
            self.xy = xy
            if yaw is not None and self.spec.script.face == "path":
                self.yaw = yaw
            if done and not self.spec.script.loop:
                end = self.spec.script.on_end
                self.state = "done"
                if end == "home":
                    self.xy, self.yaw = (self.spec.home[0], self.spec.home[1]), math.radians(self.spec.home[2])
                elif end == "park":
                    self.park()
        return self.xy[0], self.xy[1], self.yaw

    # ladder picker: one fruit falls from the hand every drop_every_s while the actor is not parked
    def fruit_drop(self, t: float, hand_z: float = LADDER_HAND_Z_M) -> Optional[Tuple[float, float, float, bool]]:
        """(x, y, z, falling) of the dropped fruit prop, or None for actors without drops."""
        p = self.spec.drop_every_s
        if p <= 0:
            return None
        hx, hy = self.xy[0] + 0.6 * math.cos(self.yaw), self.xy[1] + 0.6 * math.sin(self.yaw)
        if self.state == "parked":
            return PARK_XY[0], PARK_XY[1], 0.0, False
        phase = float(t) % p
        fall_s = math.sqrt(2.0 * hand_z / 9.81)
        if phase < fall_s:
            return hx, hy, max(0.04, hand_z - 0.5 * 9.81 * phase * phase), True
        return hx, hy, 0.04, False


# ---------------------------------------------------------------------- compiled-model classification
def parse_geom_name(name: Optional[str]) -> Optional[Tuple[str, str, str]]:
    """``actor_<class>_<id>__<role>_<k>`` → (class, id, role); props → ('prop', id, role); else None."""
    if not name or not name.startswith("actor_") or "__" not in name:
        return None
    head, tail = name.split("__", 1)
    role = tail.rsplit("_", 1)[0]
    rest = head[len("actor_"):]
    cls, _, aid = rest.partition("_")
    if cls not in ("person", "animal", "prop") or not aid:
        return None
    return cls, aid, role


def geom_table(model) -> Dict[int, Tuple[str, str, str]]:
    """geom id → (class, actor id, role) for every actor geom of a compiled model."""
    import mujoco
    out = {}
    for g in range(model.ngeom):
        p = parse_geom_name(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g))
        if p is not None:
            out[g] = p
    return out


def person_geoms(model) -> List[int]:
    """Geoms whose pixels are person pixels (privacy truth): person actors, roles body / head / limb."""
    return sorted(g for g, (c, _a, r) in geom_table(model).items() if c == "person" and r in PERSON_ROLES)


def animal_geoms(model) -> List[int]:
    return sorted(g for g, (c, _a, r) in geom_table(model).items() if c == "animal")


def actor_bodies(model) -> Dict[str, int]:
    """body name → mocap id for every actor / prop body."""
    import mujoco
    out = {}
    for b in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith("actor_") and int(model.body_mocapid[b]) >= 0:
            out[n] = int(model.body_mocapid[b])
    return out


def hull_distance(robot_xy: Sequence[float], actor_xy: Sequence[float], hull_r: float = 0.45) -> float:
    """Planar distance (m) from an approximate robot hull circle to an actor reference point."""
    return max(0.0, math.dist(robot_xy[:2], actor_xy[:2]) - hull_r)
