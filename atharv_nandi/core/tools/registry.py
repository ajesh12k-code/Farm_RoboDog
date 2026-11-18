"""ToolSpec registry for the orchard tools T2 CLIPPER, T4 TWIST, T5 PRUNER (WP2; ORCHARD-MVP-SPEC 2.4, MVP-SPEC 2.3).

One frozen ``ToolSpec`` per tool holds everything a consumer needs without reading the MJCF: names of the
body, joints, actuators, sites, equalities and contact-active geoms; the design relpose ``X_nom`` of the tool
in the ``arm_wrist`` frame (the nominal-pose lock writes it, never the measured pose); the rack cradle and
the WP0c frozen gantry value used for docking; masses, TCP offsets and the cut capacity numbers.

``ToolHandles.from_model(m)`` resolves every name to MuJoCo ids once (design rule 3: every index from a name
lookup). Actuators are appended in the order of ``TOOL_IDS``, so they are 25..30 in a composed orchard world,
but consumers use ``handles.act_id`` and never the numbers.

Numbers are labelled: (frozen) from the WP0c sweep (feasibility-report 5.5), (spec) from
ORCHARD-MVP-SPEC 2.4 / MVP-SPEC 2.3 / 3.4, (assumption) everything else. Pure data plus a MuJoCo id lookup;
no ROS.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

TOOL_IDS = ("T2", "T4", "T5")          # actuator order: T2 roll, blade; T4 roll, jaw; T5 roll, blade
PICKER = "T1"                          # the built-in parallel gripper: "no tool mounted"
FLANGE_SITE = "tool_flange"            # arm_wrist site at x 0.09: the tool's handle_ref locks onto it
FLANGE_X_M = 0.09                      # (frozen) arm_wrist x of tool_flange
WRIST_BODY = "arm_wrist"
TRUNK_BODY = "trunk"
# Arm bodies a mounted tool never collides with (explicit <exclude> pairs, MVP-SPEC 2.3 "Contacts").
EXCLUDE_ROBOT_BODIES = ("arm_wrist", "finger_left", "finger_right", "arm_forearm")

# Contact classes (MVP-SPEC 3.1 bitmask, adapted to this model where robot, ground and wood are contype 1):
#   parked tool   contype 16 / conaffinity 0: collides with nothing (bit 16 is in no conaffinity)
#   active tool   the listed ``collide_geoms`` get contype 8 / conaffinity 1: ground, tree wood, pool fruit and
#                 the robot except the excluded arm bodies; blades, V-trap and catcher stay contype 0 (logical cut)
PARKED_CONTYPE, PARKED_CONAFFINITY = 16, 0
ACTIVE_CONTYPE, ACTIVE_CONAFFINITY = 8, 1

WELD_SOLREF = (0.006, 1.0)             # (spec) 3 dt at dt 0.002; 0.004 is the 2 dt floor and is not used


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str                       # T2 / T4 / T5
    name: str                          # body name and name prefix: clipper / twist / pruner
    title: str
    xml: str                           # file in atharv_nandi/sim/models/AtharvNandi/tools/
    kind: str                          # cut | twist | prune
    roll_joint: str
    act_joint: str                     # blade hinge (cut, prune) or jaw slide (twist)
    roll_range: Tuple[float, float]    # rad
    act_range: Tuple[float, float]     # joint range: blade 0 (open) .. close_rad (closed); jaw 0 .. 0.16 m opening
    ctrl_range: Tuple[float, float]    # actuator ctrl: blade 0..1 (1 = closed); jaw = opening in m
    cradle_site: str
    cradle_gantry: float               # (frozen) WP0c docking gantry value for this cradle
    mass_kg: float                     # (assumption) whole tool
    tcp_from_wrist_m: float            # (frozen, WP0c TOOLS) TCP distance from the wrist pivot along the tool axis
    collide_geoms: Tuple[str, ...]     # geoms that touch the world while the tool is mounted
    close_rad: float = 0.0             # blade travel open -> closed (rad)
    lever_m: float = 0.0               # blade pivot -> cut point (m)
    blade_forcerange_nm: float = 0.0   # |actuator torque| cap on the blade joint
    max_stem_mm: Optional[float] = None
    jaw_max_m: float = 0.0
    crops: Tuple[str, ...] = ()
    tasks: Tuple[str, ...] = ()

    # ---- names
    @property
    def body(self) -> str:
        return self.name

    @property
    def free_joint(self) -> str:
        return f"{self.name}_free"

    @property
    def roll_act(self) -> str:
        return self.roll_joint

    @property
    def act_act(self) -> str:
        return self.act_joint

    @property
    def tcp_site(self) -> str:
        return f"{self.name}_tcp"

    @property
    def handle_site(self) -> str:
        return f"{self.name}_handle_ref"

    @property
    def blade_a(self) -> Optional[str]:
        return f"{self.name}_blade_a" if self.kind in ("cut", "prune") else None

    @property
    def blade_b(self) -> Optional[str]:
        return f"{self.name}_blade_b" if self.kind in ("cut", "prune") else None

    @property
    def head_body(self) -> str:
        return f"{self.name}_head"

    @property
    def moving_body(self) -> str:
        return f"{self.name}_blade" if self.kind in ("cut", "prune") else f"{self.name}_jaw"

    @property
    def bodies(self) -> Tuple[str, ...]:
        return (self.body, self.head_body, self.moving_body)

    @property
    def rack_eq(self) -> str:
        return f"rack_{self.name}"

    @property
    def grip_eq(self) -> str:
        return f"grip_{self.name}"

    @property
    def pos_sensor(self) -> str:
        """jointpos sensor of the blade hinge (T2, T5) or of the roll (T4): OS 2.4 'blade jointpos' / 'roll jointpos'."""
        return f"{self.name}_roll_pos" if self.kind == "twist" else f"{self.act_joint}_pos"

    @property
    def range_site(self) -> Optional[str]:
        """Site of the TCP rangefinder (T2 only, OS 2.4): at the TCP, z along the tool axis. The hook casts the ray
        at 50 Hz while the tool is mounted (an MJCF rangefinder would cost 6.4 us on every physics step)."""
        return f"{self.name}_range" if self.kind == "cut" else None

    @property
    def range_name(self) -> Optional[str]:
        return f"{self.name}_tcp_range" if self.kind == "cut" else None

    @property
    def touch_sensors(self) -> Tuple[str, ...]:
        return (f"{self.name}_jaw_a_touch", f"{self.name}_jaw_b_touch") if self.kind == "twist" else ()

    # ---- numbers
    @property
    def tcp_local(self) -> Tuple[float, float, float]:
        """TCP in the tool body frame (origin = handle_ref = tool_flange when mounted)."""
        return (self.tcp_from_wrist_m - FLANGE_X_M, 0.0, 0.0)

    @property
    def capacity_n(self) -> float:
        """Commanded cut capacity at the cut point: forcerange / lever (MVP-SPEC 3.4 rule 2)."""
        return self.blade_forcerange_nm / self.lever_m if self.lever_m > 0 else 0.0


# X_nom: design relpose of the tool body in the arm_wrist frame (MVP-SPEC 2.3 step 7): the handle_ref sits on the
# tool_flange site, tool axis = wrist +x. (spec: lock writes X_nom, not X_meas)
X_NOM_POS = (FLANGE_X_M, 0.0, 0.0)
X_NOM_QUAT = (1.0, 0.0, 0.0, 0.0)

TOOLS: Dict[str, ToolSpec] = {
    "T2": ToolSpec(
        tool_id="T2", name="clipper", title="T2 CLIPPER", xml="clipper.xml", kind="cut",
        roll_joint="clipper_roll", act_joint="clipper_blade", roll_range=(-1.5708, 1.5708),
        act_range=(0.0, 0.6), ctrl_range=(0.0, 1.0), cradle_site="cradle_T2", cradle_gantry=-0.02,
        mass_kg=0.28, tcp_from_wrist_m=0.20, collide_geoms=("clipper_handle", "clipper_frame"),
        close_rad=0.6, lever_m=0.05, blade_forcerange_nm=4.0,   # (spec) 4 N.m / 0.05 m = 80 N at the cut point
        crops=("mango", "guava"), tasks=("mango_clip", "guava_clip")),
    "T4": ToolSpec(
        tool_id="T4", name="twist", title="T4 TWIST", xml="twist_picker.xml", kind="twist",
        roll_joint="twist_roll", act_joint="twist_jaw", roll_range=(-4.7124, 4.7124),
        act_range=(0.0, 0.16), ctrl_range=(0.0, 0.16), cradle_site="cradle_T4", cradle_gantry=-0.06,
        mass_kg=0.35, tcp_from_wrist_m=0.25,   # WP0c assumed 0.20 (change request: jaws must clear the fruit)
        collide_geoms=("twist_handle", "twist_frame", "twist_jaw_a", "twist_jaw_b"), jaw_max_m=0.16,
        crops=("papaya", "guava"), tasks=("papaya_twist", "guava_twist")),
    "T5": ToolSpec(
        tool_id="T5", name="pruner", title="T5 PRUNER", xml="pruner.xml", kind="prune",
        roll_joint="pruner_roll", act_joint="pruner_blade", roll_range=(-1.5708, 1.5708),
        act_range=(0.0, 0.6), ctrl_range=(0.0, 1.0), cradle_site="cradle_T5", cradle_gantry=-0.06,
        mass_kg=0.30, tcp_from_wrist_m=0.21, collide_geoms=("pruner_handle", "pruner_frame"),
        close_rad=0.6, lever_m=0.03, blade_forcerange_nm=24.0, max_stem_mm=12.0,  # (assumption) 800 N at the hook
        crops=("mango", "guava", "banana"),
        tasks=("rootstock_sprout", "skirt_lift", "dead_twig", "malformed_panicle", "banana_tidy")),
}

# T4 jaw geometry (tool frame, see twist_picker.xml): the fixed jaw's inner face is at y = +JAW_FIXED_Y, the moving
# jaw's inner face at y = JAW_FIXED_Y - opening. A fruit of width w is gripped centred at y = JAW_FIXED_Y - w / 2.
JAW_FIXED_Y = 0.07

# (assumption) cut forces at the cut point, N. MVP-SPEC 3.4 rule 2 needs F_cut per crop; the crop profiles
# (WP23) carry none yet, so a profile's ``cut_force_n`` wins over these. Tomato pedicel shear 17.4 N is the
# only cited value (MVP-SPEC 2.3); a mango pedicel is woodier.
F_CUT_N = {"mango": 40.0, "guava": 30.0}
# (assumption) green-wood shear for the pruner: F = PRUNE_SHEAR_N_PER_MM2 x pi d^2 / 4 (12 mm -> 565 N).
PRUNE_SHEAR_N_PER_MM2 = 5.0


def get(tool_id: str) -> ToolSpec:
    try:
        return TOOLS[tool_id]
    except KeyError:
        raise KeyError(f"unknown tool {tool_id!r}; tools are {TOOL_IDS}") from None


def by_name(name: str) -> ToolSpec:
    for t in TOOLS.values():
        if t.name == name or t.tool_id == name:
            return t
    raise KeyError(f"unknown tool {name!r}")


def all_tools() -> List[ToolSpec]:
    return [TOOLS[k] for k in TOOL_IDS]


def actuator_names() -> List[str]:
    """The six appended actuators in compose order (25..30 in a composed orchard world)."""
    out = []
    for t in all_tools():
        out += [t.roll_act, t.act_act]
    return out


def f_cut_n(crop: str, profile: Optional[dict] = None) -> float:
    if profile and profile.get("cut_force_n") is not None:
        return float(profile["cut_force_n"])
    try:
        return F_CUT_N[crop]
    except KeyError:
        raise KeyError(f"no cut force for crop {crop!r} (add cut_force_n to its profile)") from None


def prune_force_n(diameter_mm: float) -> float:
    return PRUNE_SHEAR_N_PER_MM2 * math.pi * float(diameter_mm) ** 2 / 4.0


def tool_for_task(task: str) -> Optional[ToolSpec]:
    for t in all_tools():
        if task in t.tasks:
            return t
    return None


# ---------------------------------------------------------------------------------------------- MuJoCo handles
@dataclass
class ToolIds:
    spec: ToolSpec
    body: int
    bodies: Tuple[int, ...]
    free_qadr: int
    free_dadr: int
    roll_qadr: int
    roll_dadr: int
    act_qadr: int
    act_dadr: int
    roll_act: int
    act_act: int
    rack_eq: int
    grip_eq: int
    tcp: int
    handle: int
    blade_a: int
    blade_b: int
    cradle: int
    geoms: Tuple[int, ...]
    collide: Tuple[int, ...]
    touch_adr: Tuple[int, ...]
    pos_adr: int = -1                  # sensordata address of ``spec.pos_sensor`` (-1: model without it)
    range_site: int = -1               # site id of ``spec.range_site`` (-1: none)
    rigid: bool = False                # NANDI_TOOL_MOUNT=rigid:<tool>: grip weld on from compile, no rack weld


@dataclass
class ToolHandles:
    """Every WP2 name of a composed world resolved to MuJoCo ids (``None`` when the world has no tools)."""
    tools: Dict[str, ToolIds] = field(default_factory=dict)
    trunk: int = -1
    wrist: int = -1
    flange: int = -1
    rigid: Optional[str] = None        # the tool id of a rigid build (``NANDI_TOOL_MOUNT=rigid:<tool>``), else None

    @classmethod
    def from_model(cls, m) -> Optional["ToolHandles"]:
        import mujoco
        o = mujoco.mjtObj

        def nid(t, n):
            return mujoco.mj_name2id(m, t, n) if n else -1

        present = [t for t in all_tools() if nid(o.mjOBJ_BODY, t.body) >= 0]
        if not present:
            return None
        h = cls(trunk=nid(o.mjOBJ_BODY, TRUNK_BODY), wrist=nid(o.mjOBJ_BODY, WRIST_BODY),
                flange=nid(o.mjOBJ_SITE, FLANGE_SITE))
        for t in present:
            b = nid(o.mjOBJ_BODY, t.body)
            bodies = tuple(nid(o.mjOBJ_BODY, n) for n in t.bodies)
            if min(bodies) < 0:
                raise KeyError(f"{t.title}: body missing in the model ({t.bodies})")
            jf, jr, ja = (nid(o.mjOBJ_JOINT, n) for n in (t.free_joint, t.roll_joint, t.act_joint))
            rigid = nid(o.mjOBJ_EQUALITY, t.rack_eq) < 0      # rigid build: grip weld only, no rack weld
            if min(jf, jr, ja) < 0:
                raise KeyError(f"{t.title}: joint missing")
            geoms = tuple(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies)
            collide = tuple(nid(o.mjOBJ_GEOM, n) for n in t.collide_geoms)
            if min(collide) < 0:
                raise KeyError(f"{t.title}: collide geom missing ({t.collide_geoms})")
            touch = []
            for s in t.touch_sensors:
                sid = nid(o.mjOBJ_SENSOR, s)
                if sid < 0:
                    raise KeyError(f"{t.title}: sensor {s} missing")
                touch.append(int(m.sensor_adr[sid]))
            ps = nid(o.mjOBJ_SENSOR, t.pos_sensor)
            ids = ToolIds(
                spec=t, body=b, bodies=bodies,
                free_qadr=int(m.jnt_qposadr[jf]), free_dadr=int(m.jnt_dofadr[jf]),
                roll_qadr=int(m.jnt_qposadr[jr]), roll_dadr=int(m.jnt_dofadr[jr]),
                act_qadr=int(m.jnt_qposadr[ja]), act_dadr=int(m.jnt_dofadr[ja]),
                roll_act=nid(o.mjOBJ_ACTUATOR, t.roll_act), act_act=nid(o.mjOBJ_ACTUATOR, t.act_act),
                rack_eq=nid(o.mjOBJ_EQUALITY, t.rack_eq), grip_eq=nid(o.mjOBJ_EQUALITY, t.grip_eq),
                tcp=nid(o.mjOBJ_SITE, t.tcp_site), handle=nid(o.mjOBJ_SITE, t.handle_site),
                blade_a=nid(o.mjOBJ_SITE, t.blade_a), blade_b=nid(o.mjOBJ_SITE, t.blade_b),
                cradle=nid(o.mjOBJ_SITE, t.cradle_site), geoms=geoms, collide=collide, touch_adr=tuple(touch),
                pos_adr=int(m.sensor_adr[ps]) if ps >= 0 else -1, range_site=nid(o.mjOBJ_SITE, t.range_site),
                rigid=rigid)
            need = ("roll_act", "act_act", "grip_eq", "tcp", "handle", "cradle") + (() if rigid else ("rack_eq",))
            for attr in need:
                if getattr(ids, attr) < 0:
                    raise KeyError(f"{t.title}: {attr} missing in the model")
            h.tools[t.tool_id] = ids
            if rigid:
                h.rigid = t.tool_id
        if h.rigid is None and len(h.tools) != len(TOOL_IDS):
            raise KeyError(f"tools missing in the model: {sorted(set(TOOL_IDS) - set(h.tools))}")
        return h

    def __getitem__(self, tool_id: str) -> ToolIds:
        return self.tools[tool_id]

    def act_ids(self) -> List[int]:
        out = []
        for k in TOOL_IDS:
            if k in self.tools:
                out += [self.tools[k].roll_act, self.tools[k].act_act]
        return out


def quat_from_mat(R: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix (numpy-free helper for the module and tests)."""
    import numpy as np
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(9))
    return tuple(float(v) for v in q)
