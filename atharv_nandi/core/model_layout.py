"""Name-based model layout for Atharv Nandi (WP0, MVP-SPEC 3.2, design rules 1-2).

Every index into qpos / qvel / ctrl / mocap / eq comes from a name lookup here, never from a literal.
A ``Layout`` is built once from a compiled ``MjModel`` (``Layout.from_model``) and is plain data after
that: it serialises to JSON (``to_json`` / ``from_json``) so a process that never compiles the world
(or a browser) can use the same indices. ``sim_runner`` publishes it once as the ``layout`` parameter.

Frozen robot table (design rule 1): qpos ``[0:32]`` is the robot (root free joint ``[0:7]``, arm
``[7:14]``, legs ``[14:32]``) and actuators ``0..24`` are the robot's (arm and gripper 0..6, legs
7..24). The legacy crops ``crop_item_1..N`` follow contiguously from qpos 32. Anything with a joint
that a work package adds is appended after them (compose ORDER, ``atharv_nandi/sim/world/compose.py``), and every
new actuator gets an index >= 25. ``frozen_violations`` lists every breach; ``farm_scene`` refuses to
hand out a model that has one.

The RL code (``rl_balance_policy.py``, ``rl_training/``) keeps its literal slices; the parity test in
``tests/test_model_layout.py`` guards them against this table.
"""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

LAYOUT_VERSION = 1

ROBOT_NQ = 32
ROBOT_NV = 31
ROBOT_NU = 25
ROOT_QPOS = slice(0, 7)
ARM_QPOS = slice(7, 14)      # gantry, yaw, shoulder, elbow, wrist_pitch, finger L, finger R
LEG_QPOS = slice(14, 32)     # (FR, FL, MR, ML, RR, RL) x (hip, thigh, calf)
ARM_CTRL = slice(0, 7)
LEG_CTRL = slice(7, 25)
CROP_PREFIX = "crop_item_"
CROP_FREE_NQ = 7

_LEGS = ("FR", "FL", "MR", "ML", "RR", "RL")
_LEG_PARTS = ("hip", "thigh", "calf")
# (joint name, qpos address) of the frozen robot prefix.
ROBOT_JOINTS: Tuple[Tuple[str, int], ...] = (
    ("root", 0),
    ("gantry_slide", 7), ("arm_yaw", 8), ("arm_shoulder", 9), ("arm_elbow", 10), ("arm_wrist_pitch", 11),
    ("finger_left_joint", 12), ("finger_right_joint", 13),
) + tuple((f"{leg}_{part}_joint", 14 + 3 * i + j) for i, leg in enumerate(_LEGS) for j, part in enumerate(_LEG_PARTS))
# Actuator names 0..24 in index order.
ROBOT_ACTUATORS: Tuple[str, ...] = (
    "gantry_slide", "arm_yaw", "arm_shoulder", "arm_elbow", "arm_wrist_pitch", "finger_left", "finger_right",
) + tuple(f"{leg}_{part}" for leg in _LEGS for part in _LEG_PARTS)

# mjtJoint values (kept literal so this module imports without MuJoCo).
JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE = 0, 1, 2, 3
JNT_NQ = {JNT_FREE: 7, JNT_BALL: 4, JNT_SLIDE: 1, JNT_HINGE: 1}
JNT_NV = {JNT_FREE: 6, JNT_BALL: 3, JNT_SLIDE: 1, JNT_HINGE: 1}


class LayoutError(ValueError):
    """A name is not in the layout, or a legacy-only accessor was used outside the legacy world."""


@dataclass(frozen=True)
class Layout:
    nq: int
    nv: int
    nu: int
    nmocap: int
    neq: int
    joints: Dict[str, Tuple[int, int, int]]    # name -> (qpos adr, dof adr, mjtJoint type)
    actuators: Tuple[str, ...]                  # actuator names in index order ("" for unnamed)
    actuator_joint: Tuple[str, ...]             # joint driven by each actuator ("" if not a joint transmission)
    equalities: Tuple[str, ...]                 # equality names in index order
    mocap: Dict[str, int]                       # mocap body name -> mocap index
    crops: Tuple[Tuple[str, int], ...]          # (crop_item_k, qpos adr) for k = 1..n, in order
    world: str = "legacy"
    world_id: str = ""
    version: int = LAYOUT_VERSION
    _act_idx: Dict[str, int] = field(default_factory=dict, compare=False, repr=False)
    _eq_idx: Dict[str, int] = field(default_factory=dict, compare=False, repr=False)
    _crop_idx: Dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        self._act_idx.update({n: i for i, n in enumerate(self.actuators) if n})
        self._eq_idx.update({n: i for i, n in enumerate(self.equalities) if n})
        self._crop_idx.update({n: a for n, a in self.crops})

    # ------------------------------------------------------------------ build
    @classmethod
    def from_model(cls, model, world: str = "legacy", world_id: str = "") -> "Layout":
        import mujoco  # only needed here
        obj = mujoco.mjtObj

        def name(kind, i):
            return mujoco.mj_id2name(model, kind, i) or ""

        joints = {}
        for j in range(model.njnt):
            n = name(obj.mjOBJ_JOINT, j)
            if n:
                joints[n] = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), int(model.jnt_type[j]))
        act_joint = []
        for i in range(model.nu):
            is_joint = int(model.actuator_trntype[i]) in (int(mujoco.mjtTrn.mjTRN_JOINT),
                                                         int(mujoco.mjtTrn.mjTRN_JOINTINPARENT))
            act_joint.append(name(obj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0])) if is_joint else "")
        mocap = {}
        for b in range(model.nbody):
            mid = int(model.body_mocapid[b])
            if mid >= 0:
                mocap[name(obj.mjOBJ_BODY, b) or f"mocap_{mid}"] = mid
        crops, k = [], 1
        while True:
            bid = mujoco.mj_name2id(model, obj.mjOBJ_BODY, f"{CROP_PREFIX}{k}")
            if bid < 0:
                break
            jnt = int(model.body_jntadr[bid])
            if jnt < 0 or int(model.jnt_type[jnt]) != JNT_FREE:
                raise LayoutError(f"{CROP_PREFIX}{k} has no free joint")
            crops.append((f"{CROP_PREFIX}{k}", int(model.jnt_qposadr[jnt])))
            k += 1
        return cls(nq=int(model.nq), nv=int(model.nv), nu=int(model.nu), nmocap=int(model.nmocap),
                   neq=int(model.neq), joints=joints,
                   actuators=tuple(name(obj.mjOBJ_ACTUATOR, i) for i in range(model.nu)),
                   actuator_joint=tuple(act_joint),
                   equalities=tuple(name(obj.mjOBJ_EQUALITY, i) for i in range(model.neq)),
                   mocap=mocap, crops=tuple(crops), world=world, world_id=world_id)

    # ------------------------------------------------------------------ lookups
    def qpos_adr(self, joint: str) -> int:
        try:
            return self.joints[joint][0]
        except KeyError:
            raise LayoutError(f"joint {joint!r} not in layout") from None

    def dof_adr(self, joint: str) -> int:
        try:
            return self.joints[joint][1]
        except KeyError:
            raise LayoutError(f"joint {joint!r} not in layout") from None

    def qpos_slice(self, joint: str) -> slice:
        adr, _, typ = self.joints.get(joint, (None, None, None))
        if adr is None:
            raise LayoutError(f"joint {joint!r} not in layout")
        return slice(adr, adr + JNT_NQ[typ])

    def dof_slice(self, joint: str) -> slice:
        _, adr, typ = self.joints.get(joint, (None, None, None))
        if adr is None:
            raise LayoutError(f"joint {joint!r} not in layout")
        return slice(adr, adr + JNT_NV[typ])

    def act_id(self, name: str) -> int:
        try:
            return self._act_idx[name]
        except KeyError:
            raise LayoutError(f"actuator {name!r} not in layout") from None

    def eq_id(self, name: str) -> int:
        try:
            return self._eq_idx[name]
        except KeyError:
            raise LayoutError(f"equality {name!r} not in layout") from None

    def mocap_id(self, body: str) -> int:
        try:
            return self.mocap[body]
        except KeyError:
            raise LayoutError(f"mocap body {body!r} not in layout") from None

    # ------------------------------------------------------------------ legacy crops
    @property
    def n_legacy_crops(self) -> int:
        return len(self.crops)

    @property
    def crop_names(self) -> List[str]:
        return [n for n, _ in self.crops]

    def crop_qpos(self, k: int) -> slice:
        """qpos slice of legacy crop k's free joint (k is 1-based)."""
        if not 1 <= k <= len(self.crops):
            raise LayoutError(f"crop {k} not in layout (1..{len(self.crops)})")
        a = self.crops[k - 1][1]
        return slice(a, a + CROP_FREE_NQ)

    def crop_slice(self, name: str) -> slice:
        try:
            a = self._crop_idx[name]
        except KeyError:
            raise LayoutError(f"crop {name!r} not in layout") from None
        return slice(a, a + CROP_FREE_NQ)

    def crop_positions(self, qpos: Sequence[float]) -> Dict[str, List[float]]:
        """{crop_item_k: [x, y, z]} for every legacy crop, read from a full-world qpos."""
        if len(qpos) != self.nq:
            raise LayoutError(f"qpos has {len(qpos)} entries, layout nq is {self.nq}")
        return {n: [float(v) for v in qpos[a:a + 3]] for n, a in self.crops}

    @property
    def is_legacy_shape(self) -> bool:
        """True when nothing is appended after the legacy crops (the qpos shape of the pre-WP0 world)."""
        return (self.nq == ROBOT_NQ + CROP_FREE_NQ * len(self.crops) and self.nu == ROBOT_NU
                and self.nmocap == 0 and self.neq == 0)

    def legacy_qpos(self, data_or_qpos) -> List[float]:
        """The /nandi/sim_qpos vector. Valid in the legacy world only (design rule 4)."""
        if self.world != "legacy" or not self.is_legacy_shape:
            raise LayoutError(f"legacy_qpos is valid only in the legacy world (world {self.world!r})")
        q = getattr(data_or_qpos, "qpos", data_or_qpos)
        return [float(v) for v in q]

    # ------------------------------------------------------------------ serialise
    def to_dict(self) -> dict:
        return {"v": self.version, "world": self.world, "world_id": self.world_id, "nq": self.nq, "nv": self.nv,
                "nu": self.nu, "nmocap": self.nmocap, "neq": self.neq,
                "joints": {n: list(v) for n, v in self.joints.items()}, "actuators": list(self.actuators),
                "actuator_joint": list(self.actuator_joint), "equalities": list(self.equalities),
                "mocap": dict(self.mocap), "crops": [list(c) for c in self.crops]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        if int(d.get("v", 0)) != LAYOUT_VERSION:
            raise LayoutError(f"layout version {d.get('v')} != {LAYOUT_VERSION}")
        return cls(nq=int(d["nq"]), nv=int(d["nv"]), nu=int(d["nu"]), nmocap=int(d["nmocap"]), neq=int(d["neq"]),
                   joints={n: tuple(int(x) for x in v) for n, v in d["joints"].items()},
                   actuators=tuple(d["actuators"]), actuator_joint=tuple(d["actuator_joint"]),
                   equalities=tuple(d["equalities"]), mocap={n: int(v) for n, v in d["mocap"].items()},
                   crops=tuple((str(n), int(a)) for n, a in d["crops"]), world=str(d.get("world", "legacy")),
                   world_id=str(d.get("world_id", "")))

    @classmethod
    def from_json(cls, text: str) -> "Layout":
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------- frozen-table guard
def frozen_violations(layout: Layout) -> List[str]:
    """Every breach of design rule 1 (empty list = OK)."""
    bad = []
    for name, adr in ROBOT_JOINTS:
        got = layout.joints.get(name)
        if got is None or got[0] != adr:
            bad.append(f"joint {name}: qpos adr {got[0] if got else None} != {adr}")
    if layout.joints.get("root", (0, 0, -1))[2] != JNT_FREE:
        bad.append("joint root is not a free joint")
    if tuple(layout.actuators[:ROBOT_NU]) != ROBOT_ACTUATORS:
        bad.append(f"actuators 0..24 changed: {list(layout.actuators[:ROBOT_NU])}")
    for k, (name, adr) in enumerate(layout.crops, start=1):
        want = ROBOT_NQ + CROP_FREE_NQ * (k - 1)
        if adr != want:
            bad.append(f"{name}: qpos adr {adr} != {want} (legacy crops must follow the robot contiguously)")
    robot_and_crops = ROBOT_NQ + CROP_FREE_NQ * len(layout.crops)
    for name, (adr, _, _) in layout.joints.items():
        is_robot = name in dict(ROBOT_JOINTS)
        is_crop = any(adr == a for _, a in layout.crops)
        if not is_robot and not is_crop and adr < robot_and_crops:
            bad.append(f"joint {name} at qpos {adr} is inside the robot/legacy-crop prefix")
    return bad


def assert_frozen(layout: Layout) -> Layout:
    bad = frozen_violations(layout)
    if bad:
        raise LayoutError("frozen robot layout violated: " + "; ".join(bad))
    return layout


# ---------------------------------------------------------------------- process-wide current layout
_CURRENT: Optional[Layout] = None


def set_current(layout: Optional[Layout]) -> None:
    """Record the layout of the world this process built (``farm_scene.load_farm_model`` calls it)."""
    global _CURRENT
    _CURRENT = layout


def current() -> Optional[Layout]:
    return _CURRENT


def n_legacy_crops_for(qpos_len: int, layout: Optional[Layout] = None) -> int:
    """Legacy crop count for a qpos vector: from ``layout`` (or the current one) when it matches the length,
    else the pre-WP0 rule ``(len - 32) // 7`` (only right for the legacy shape)."""
    lay = layout if layout is not None else _CURRENT
    if lay is not None and lay.nq == qpos_len:
        return lay.n_legacy_crops
    return max(0, (qpos_len - ROBOT_NQ) // CROP_FREE_NQ)


def legacy_crop_positions(qpos: Sequence[float], layout: Optional[Layout] = None) -> Dict[str, List[float]]:
    """{crop_item_k: [x, y, z]} of the legacy crops in a full-world qpos (recorder, mission executor).

    Uses ``layout`` (or the current one) when its nq matches, so appended module joints are never read as
    crops; otherwise the pre-WP0 rule (legacy shape only)."""
    n = n_legacy_crops_for(len(qpos), layout)
    return {f"{CROP_PREFIX}{k}": [float(v) for v in qpos[ROBOT_NQ + CROP_FREE_NQ * (k - 1):
                                                         ROBOT_NQ + CROP_FREE_NQ * (k - 1) + 3]]
            for k in range(1, n + 1)}
